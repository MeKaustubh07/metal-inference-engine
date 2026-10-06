"""Order-invariant decisions: score every option as a continuation of the same context.

Listing options inside a prompt ("A) ... B) ...") gives each one a different position and lets later options see
earlier ones, so the answer can depend on the order. Here every option is scored alone: the context is prefilled
once, each option continues from its own copy of the context's state (the attention layers' K/V and, for the
hybrid model, the DeltaNet recurrent state: a recurrence has no attention mask, so isolation means a copy), every
option starts at the same position, and the options of a group run in one packed forward pass. When the context is
too big to copy within the memory budget, options run one at a time on the context's own state, rewound after
each. Shuffle the options and each one's score is unchanged (tests/test_decision.py).

An option is scored as a whole word: its tokens, then a token that cannot continue its last word or number (a
space, punctuation, a newline, an end token). That is P(the answer starts with the option as a unit): "1" is not
credited with answers that start "10", and "No" is not penalized because the model would go on "No, the capital
is ...". (Requiring the turn to end right after the option instead would measure how terse the model is.)

score_options -> log P(option | context) per option (+ optionally the hidden state after the option: one embedding
per option, as in JEV-style decision networks). decide -> the three question types of a decision dataset:
choice (pick one of N), boolean (yes / no), score (an ordinal label, e.g. 1-5). Zero-shot: no trained head.
The *_steps generators yield between units of work (each context chunk, each option group), so the server can
interleave decode steps for running streams; score_options / score / decide run them to completion.
"""
from dataclasses import dataclass

import torch

from chat import format_chat

LOGIT_ROWS = 32                       # output head over at most this many rows at once ([32, 248k] fp32 = 32 MB)
FORK_BUDGET = 256 * 2**20             # bytes of forked states alive at once; more options run in several groups
PREFILL_CHUNK = 512                   # context tokens per forward (bounds the [heads, T, S] attention scores)


@dataclass
class OptionScore:
    logprob: float                    # log P(option tokens | context), summed over the option's tokens
    tokens: int
    embedding: torch.Tensor | None    # final-norm hidden state after the option's last token (CPU, fp32)

    @property
    def mean_logprob(self) -> float:
        return self.logprob / self.tokens


def _token_logprobs(model, h: torch.Tensor, targets: list[int], vocab: int) -> torch.Tensor:
    """log P(targets[i] | row i) for hidden rows h [N, hidden], the head applied LOGIT_ROWS rows at a time."""
    out = []
    t = torch.tensor(targets, device=h.device)
    for r in range(0, len(targets), LOGIT_ROWS):
        lg = model.head(h[r:r + LOGIT_ROWS])[:, :vocab].float()
        out.append(lg.gather(1, t[r:r + LOGIT_ROWS, None])[:, 0] - torch.logsumexp(lg, dim=1))
    return torch.cat(out).cpu() if out else torch.zeros(0)


def _boundary_logprobs(model, h: torch.Tensor, mask: torch.Tensor, vocab: int) -> torch.Tensor:
    """log P(next token is a boundary | row i) for hidden rows h [N, hidden]."""
    out = []
    for r in range(0, len(h), LOGIT_ROWS):
        lg = model.head(h[r:r + LOGIT_ROWS])[:, :vocab].float()
        out.append(torch.logsumexp(lg.masked_fill(~mask, float("-inf")), dim=1) - torch.logsumexp(lg, dim=1))
    return torch.cat(out).cpu() if out else torch.zeros(0)


_BOUNDARY: dict[int, torch.Tensor] = {}


def boundary_mask(tokenizer, vocab: int) -> torch.Tensor:
    """[vocab] bool: tokens that cannot continue the previous word or number, i.e. special tokens and tokens that
    start with whitespace or punctuation. Tokens starting with a letter or digit continue it, and so do byte pieces
    of a character (they decode to U+FFFD). Built once per tokenizer (0.2 s for 248k tokens)."""
    if id(tokenizer) not in _BOUNDARY:
        texts = (tokenizer.decode([i]) for i in range(vocab))
        _BOUNDARY[id(tokenizer)] = torch.tensor([bool(t) and (t in tokenizer.special or t[0].isspace()
                                                               or not (t[0].isalnum() or t[0] == "\ufffd"))
                                                 for t in texts])
    return _BOUNDARY[id(tokenizer)]


def _fork_bytes(state) -> int:
    kv = state.kv
    return 2 * kv.k.numel() * kv.k.element_size() + (state.S.numel() + state.conv_tail.numel()) * 4


def _run(gen):
    """Drive a *_steps generator to completion -> its return value."""
    try:
        while True:
            next(gen)
    except StopIteration as done:
        return done.value


def score_options_steps(model, context_ids: list[int], options: list[list[int]], vocab: int,
                        embeddings: bool = False, budget: int = FORK_BUDGET, chunk: int = PREFILL_CHUNK,
                        boundary: torch.Tensor | None = None):
    """Generator form of score_options: yields after each context chunk and each option group."""
    if not context_ids or not options or any(not o for o in options):
        raise ValueError("need a non-empty context and non-empty options")
    ctx = model.new_state(len(context_ids) + max(map(len, options)))
    ids = torch.tensor(context_ids)
    for i in range(0, len(ids), chunk):                               # chunked, like the scheduler's prefill
        last = model.forward(ids[i:i + chunk], state=ctx, last_only=True)
        yield
    first = last[0, :vocab].float()
    first_lp = (first - torch.logsumexp(first, dim=0)).cpu()          # shared: every option's first token
    del last, first
    group = max(1, budget // _fork_bytes(ctx))
    # pack in a canonical order (by token ids), so any permutation of the options produces the identical batch:
    # a row's rounding can depend on where it lands (the batched kernels group rows), so this makes invariance exact
    order = sorted(range(len(options)), key=lambda i: options[i])
    canonical, scores = [options[i] for i in order], []
    mask = boundary.to(model.b.device) if boundary is not None else None
    # one option at a time (a context too big to copy within the budget, e.g. Tiny Aya at its cap: 288 MiB at 4K,
    # 576 at 8K): score it on the context's own state, then rewind that, instead of copying it. An option only writes
    # K/V at positions n_ctx.. and reads up to its own end, so the rows it leaves behind are never read by the next;
    # the recurrent state (Qwen3.5's DeltaNet) is updated in place, so it is put back from a copy taken once
    n_ctx = ctx.length
    saved = (ctx.S.clone(), ctx.conv_tail.clone()) if group == 1 and ctx.S.numel() else None
    for g in range(0, len(canonical), group):
        opts = canonical[g:g + group]
        if group > 1:
            h = model.packed_hidden([(torch.tensor(o), ctx.fork()) for o in opts])
        else:
            try:
                h = model.packed_hidden([(torch.tensor(opts[0]), ctx)])
            finally:
                ctx.kv.length = n_ctx
                if saved:
                    ctx.S.copy_(saved[0])
                    ctx.conv_tail.copy_(saved[1])
        ends = torch.tensor([len(o) for o in opts]).cumsum(0).tolist()
        rows = [e - len(o) + j for o, e in zip(opts, ends) for j in range(len(o) - 1)]
        lp = _token_logprobs(model, h[torch.tensor(rows, device=h.device, dtype=torch.long)] if rows else h[:0],
                             [t for o in opts for t in o[1:]], vocab)
        last = h[torch.tensor([e - 1 for e in ends], device=h.device, dtype=torch.long)]
        bound = _boundary_logprobs(model, last, mask, vocab) if mask is not None else None
        k = 0
        for n, (o, e) in enumerate(zip(opts, ends)):
            rest = lp[k:k + len(o) - 1].sum().item(); k += len(o) - 1
            emb = h[e - 1].float().cpu() if embeddings else None
            if bound is None:
                scores.append(OptionScore(first_lp[o[0]].item() + rest, len(o), emb))
            else:                                             # the boundary counts as one more scored token
                scores.append(OptionScore(first_lp[o[0]].item() + rest + bound[n].item(), len(o) + 1, emb))
        del h
        yield
    out = [None] * len(options)
    for i, sc in zip(order, scores):
        out[i] = sc
    return out


def score_options(model, context_ids: list[int], options: list[list[int]], vocab: int,
                  embeddings: bool = False, budget: int = FORK_BUDGET, chunk: int = PREFILL_CHUNK,
                  boundary: torch.Tensor | None = None) -> list[OptionScore]:
    """Each option's log-probability as a continuation of context_ids, every option from its own fork of the
    context's state, all options of a group in one packed pass. vocab: the tokenizer's vocabulary size (the
    output head is padded beyond it). With boundary (see boundary_mask), each score also includes log P(the next
    token ends the option's last word), so an option is scored as a whole word, not as a prefix of longer ones."""
    return _run(score_options_steps(model, context_ids, options, vocab, embeddings, budget, chunk, boundary))


KINDS = ("choice", "boolean", "score")


def prompt_ids(tokenizer, context: str, question: str, chat: bool, chat_style) -> list[int]:
    """The shared context: chat-formatted as the user turn, ending where the assistant's answer begins."""
    text = f"{context}\n\n{question}" if context else question
    if chat:
        return tokenizer.encode(format_chat([{"role": "user", "content": text}], style=chat_style))  # writes BOS
    return tokenizer.encode(text + "\nAnswer:", add_bos=True)


def option_ids(tokenizer, option: str, chat: bool) -> list[int]:
    """An option's tokens where the answer begins: at the start of the assistant turn (chat), or after a space
    following a plain "Answer:". A continuation, so never a BOS."""
    if chat:
        return tokenizer.encode(option)
    return tokenizer.encode(option if option[:1].isspace() else " " + option)


def prepare(engine, kind: str, context: str, question: str, options: list[str] | None = None,
            chat: bool = True) -> tuple[list[str], list[int], list[list[int]]]:
    """Validate and tokenize (CPU only) -> (options, context ids, option ids). boolean options default to Yes / No;
    score options are ordinal labels from lowest to highest."""
    if kind not in KINDS:
        raise ValueError(f"type must be one of {KINDS}")
    if kind == "boolean":
        options = options or ["Yes", "No"]
        if len(options) != 2:
            raise ValueError("boolean takes exactly two options: the yes and the no answer")
    elif not options or len(options) < 2:
        raise ValueError(f"{kind} needs at least two options")
    if any(not o.strip() for o in options):
        raise ValueError("options must be non-empty")
    tok = engine.tokenizer
    return (options, prompt_ids(tok, context, question, chat, engine.chat_style),
            [option_ids(tok, o, chat) for o in options])


def score_steps(engine, kind: str, options: list[str], ctx: list[int], opt_ids: list[list[int]],
                length_normalize: bool = True, embeddings: bool = False, chunk: int = PREFILL_CHUNK):
    """Generator form of score: the model work for a prepared question (run it on the thread that owns the GPU)."""
    vocab = engine.tokenizer.vocab_size()
    scores = yield from score_options_steps(engine.model, ctx, opt_ids, vocab, embeddings, chunk=chunk,
                                            boundary=boundary_mask(engine.tokenizer, vocab))
    key = torch.tensor([s.mean_logprob if length_normalize else s.logprob for s in scores], dtype=torch.float64)
    # normalize over the scores sorted by value: a sum's rounding depends on its order, so this keeps the
    # probabilities, like the scores, bit-identical however the options are ordered
    probs = (key - torch.logsumexp(key.sort().values, dim=0)).exp().tolist()
    out = {"type": kind, "options": [
        {"text": o, "logprob": s.logprob, "mean_logprob": s.mean_logprob, "tokens": s.tokens, "probability": p}
        | ({"embedding": s.embedding.tolist()} if embeddings else {})
        for o, s, p in zip(options, scores, probs)],
        "usage": {"prompt_tokens": len(ctx), "option_tokens": sum(map(len, opt_ids))}}
    best = min(range(len(options)), key=lambda i: (-probs[i], opt_ids[i]))   # exact ties: canonical order
    if kind == "boolean":
        out["decision"], out["probability"] = best == 0, probs[0]
    elif kind == "score":
        out["decision"], out["expected"] = best, sum(i * p for i, p in enumerate(probs))
    else:
        out["decision"] = best
    return out


def score(engine, kind: str, options: list[str], ctx: list[int], opt_ids: list[list[int]],
          length_normalize: bool = True, embeddings: bool = False, chunk: int = PREFILL_CHUNK) -> dict:
    """The model work for a prepared question. Probabilities are a softmax over the options' log-probabilities (per
    token if length_normalize): they compare only these options."""
    return _run(score_steps(engine, kind, options, ctx, opt_ids, length_normalize, embeddings, chunk))


def decide(engine, kind: str, context: str, question: str, options: list[str] | None = None, chat: bool = True,
           length_normalize: bool = True, embeddings: bool = False) -> dict:
    """prepare + score in one call (library use; the server splits them across threads)."""
    options, ctx, opt_ids = prepare(engine, kind, context, question, options, chat)
    return score(engine, kind, options, ctx, opt_ids, length_normalize, embeddings)
