"""Autoregressive generation loops: prefill the prompt once, then decode one token at a time."""
from collections.abc import Iterator

import torch

from sampler import SamplingParams, sample


def _next_logits(model, tokenizer, ids_new: list[int], state) -> torch.Tensor:
    """Run the new tokens through the model; return last-position logits over real vocab ids only."""
    logits = model.forward(torch.tensor(ids_new), state=state, last_only=True)[0]
    return logits[: tokenizer.vocab_size()].cpu()          # ids past the tokenizer vocab are padding rows


def generate_greedy(model, tokenizer, prompt: str, max_new_tokens: int, eos_ids: set[int],
                    use_cache: bool = True) -> list[int]:
    """Always take the highest-scoring next token."""
    ids = tokenizer.encode(prompt, add_bos=True)                      # raw text: BOS first, if the model has one
    new: list[int] = []
    if use_cache:
        state = model.new_state(len(ids) + max_new_tokens)
        logits = _next_logits(model, tokenizer, ids, state)            # prefill: whole prompt at once
    for _ in range(max_new_tokens):
        if not use_cache:                                              # recompute the whole sequence every step
            logits = _next_logits(model, tokenizer, ids + new, None)
        next_id = int(logits.argmax())
        if next_id in eos_ids:
            break
        new.append(next_id)
        if use_cache and len(new) < max_new_tokens:                   # no forward after the last token
            logits = _next_logits(model, tokenizer, [next_id], state)  # decode: one token, cached context
    return new


def generate_stream(model, tokenizer, prompt_ids: list[int], params: SamplingParams,
                    max_new_tokens: int, eos_ids: set[int], record: dict | None = None) -> Iterator[str]:
    """Sample token by token and yield text as soon as it forms complete characters.

    record (optional): filled with {"ids": generated ids, "stop": "eos" | "length"}.
    """
    gen = torch.Generator().manual_seed(params.seed) if params.seed is not None else None
    state = model.new_state(len(prompt_ids) + max_new_tokens)
    logits = _next_logits(model, tokenizer, list(prompt_ids), state)
    history = list(prompt_ids)
    new: list[int] = []
    emitted = ""
    stop = "length"
    for _ in range(max_new_tokens):
        next_id = sample(logits, history, params, gen)
        if next_id in eos_ids:
            stop = "eos"
            break
        history.append(next_id)
        new.append(next_id)
        text = tokenizer.decode(new)
        if not text.endswith("�"):          # wait until a multi-byte character is complete
            yield text[len(emitted):]
            emitted = text
        if len(new) < max_new_tokens:
            logits = _next_logits(model, tokenizer, [next_id], state)
    text = tokenizer.decode(new)                   # flush anything still held back
    if len(text) > len(emitted):
        yield text[len(emitted):]
    if record is not None:
        record["ids"], record["stop"] = new, stop
