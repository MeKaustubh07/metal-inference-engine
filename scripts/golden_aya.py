"""Answer key for Tiny Aya (Cohere2) that fits in 8 GB: transformers' own decoder layers, run in fp32 on the CPU ONE AT
A TIME, each built from its weights in the safetensors files and freed before the next.

The whole model in fp32 is 13.4 GB; one layer is ~0.31 GB. StreamedCohere2 does what Cohere2Model.forward does (the
same causal and sliding-window masks from transformers' mask helpers, the same rotary embedding, the same
DynamicCache) but materialises one Cohere2DecoderLayer at a time; the embedding rows and the tied head are read
from the bf16 table in slices. All prompts move in lockstep: each layer is built once per step and every prompt goes
through it (one prompt at a time inside the layer, so no padding), and the head runs once per step for all of them.
Checked against the full transformers model on small random models (tests/test_cohere2.py).

usage: golden_aya.py [model_dir] [out_dir]     (defaults: models/tiny-aya-global, tests/golden_aya; gitignored)
       golden_aya.py --long [model_dir] [out_dir] (default out: tests/golden_aya_long; gitignored)
Saved per prompt (<out_dir>/<i>.pt): text, chat, ids (HF tokenizer); for the raw prompts embed, l{i}_out for all 36
layers, final_norm and the logits of every position; for the chat prompts final_norm of every position and the logits
of the last 8 (the ~360-token template preamble makes every layer and every logit too large to keep); greedy: 10
argmax tokens, no stopping, and step_logits: the logits each of the 9 decode steps produced.

--long (M5): one prompt past the 4096-token sliding window, ~600 tokens of public-domain text in each of 8 languages
(scripts/long_texts.py, downloaded on demand), ~4.8K tokens with BOS. It is prefilled in LONG_CHUNK-token chunks over
one DynamicCache (exact past the window: tests/test_window.py). Saved (<out_dir>/0.pt): ids, final_norm of every
position, the logits of the rows in long_rows() (every 64th, the window's edge 4090-4100, the last 8) and 10 greedy
tokens with the step_logits of the 9 decode steps.
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.models.cohere2.modeling_cohere2 import Cohere2DecoderLayer, Cohere2LayerNorm, Cohere2RotaryEmbedding

HEAD_ROWS = 32768                       # rows of the tied head widened to fp32 at a time (256 MB at hidden 2048)


class StreamedCohere2:
    """transformers' Cohere2 forward pass with one decoder layer in memory at a time."""

    def __init__(self, model_dir: str):
        d = Path(model_dir)
        self.config = AutoConfig.from_pretrained(d)
        self.config._attn_implementation = "eager"
        index = d / "model.safetensors.index.json"
        weight_map = json.load(open(index))["weight_map"] if index.exists() else None
        files = sorted(set(weight_map.values())) if weight_map else ["model.safetensors"]
        self.files = {f: safe_open(str(d / f), framework="pt") for f in files}
        self.where = weight_map or {n: files[0] for n in self.files[files[0]].keys()}
        if not self.config.tie_word_embeddings or "lm_head.weight" in self.where:   # transformers would use lm_head
            raise ValueError("the answer key reads the head from embed_tokens; this checkpoint has a separate lm_head")
        self.rotary = Cohere2RotaryEmbedding(self.config)
        self.norm = Cohere2LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)
        self.norm.weight.data = self._get("model.norm.weight").float()

    def _get(self, name: str) -> torch.Tensor:
        return self.files[self.where[name]].get_tensor(name)

    def _rows(self, name: str, lo: int, hi: int) -> torch.Tensor:
        return self.files[self.where[name]].get_slice(name)[lo:hi]

    def layer(self, i: int) -> Cohere2DecoderLayer:
        """Decoder layer i with its weights widened to fp32 (built on the meta device, so nothing is initialised)."""
        prefix = f"model.layers.{i}."
        with torch.device("meta"):
            layer = Cohere2DecoderLayer(self.config, i)
        weights = {n[len(prefix):]: self._get(n).float() for n in self.where if n.startswith(prefix)}
        layer.load_state_dict(weights, strict=True, assign=True)
        return layer.eval()

    def embed(self, ids: list[int]) -> torch.Tensor:
        table = self.files[self.where["model.embed_tokens.weight"]].get_slice("model.embed_tokens.weight")
        return torch.stack([table[i:i + 1][0] for i in ids]).float()

    def head(self, h: torch.Tensor) -> torch.Tensor:
        """The tied head, times logit_scale: [N, hidden] -> [N, vocab], the table widened HEAD_ROWS rows at a time."""
        V = self.config.vocab_size
        out = torch.cat([h @ self._rows("model.embed_tokens.weight", r, min(r + HEAD_ROWS, V)).float().T
                         for r in range(0, V, HEAD_ROWS)], dim=-1)
        return out * self.config.logit_scale

    @torch.no_grad()
    def run_all(self, batch: list[list[int]], caches: list[DynamicCache],
                captures: list[dict | None] | None = None) -> list[torch.Tensor]:
        """Each sequence's new tokens, continuing its own cache -> its final-norm hidden states [T, hidden], as
        Cohere2Model.forward computes them, with each decoder layer built once for all the sequences."""
        c = self.config
        captures = captures or [None] * len(batch)
        hs, ctx = [], []
        for ids, cache, cap in zip(batch, caches, captures):       # masks from the caches BEFORE any layer updates them
            start = cache.get_seq_length()
            h = self.embed(ids)[None]                                            # [1, T, hidden]
            position_ids = torch.arange(start, start + len(ids))[None]
            kw = dict(config=c, inputs_embeds=h, attention_mask=None, past_key_values=cache, position_ids=position_ids)
            masks = {"full_attention": create_causal_mask(**kw),
                     "sliding_attention": create_sliding_window_causal_mask(**kw)}
            hs.append(h)
            ctx.append((masks, self.rotary(h, position_ids), position_ids))
            if cap is not None:
                cap["embed"] = h[0].clone()
        for i in range(c.num_hidden_layers):
            layer = self.layer(i)
            for j, (cache, (masks, pe, position_ids), cap) in enumerate(zip(caches, ctx, captures)):
                hs[j] = layer(hs[j], attention_mask=masks[c.layer_types[i]], position_embeddings=pe,
                              past_key_values=cache, use_cache=True, position_ids=position_ids)
                if cap is not None:
                    cap[f"l{i}_out"] = hs[j][0].clone()
            del layer
        out = [self.norm(h)[0] for h in hs]
        for h, cap in zip(out, captures):
            if cap is not None:
                cap["final_norm"] = h.clone()
        return out

    def greedy_all(self, prompts: list[list[int]], n: int, captures: list[dict | None] | None = None,
                   keep_from: list[int] | None = None, log=None):
        """Prefill every prompt, then n argmax tokens each (no stopping), all in lockstep -> (per prompt the logits of
        its positions from keep_from[i] on (default: all), per prompt its new tokens, per prompt the logits of its
        n - 1 decode steps [n - 1, vocab])."""
        caches = [DynamicCache(config=self.config) for _ in prompts]
        hs = self.run_all(prompts, caches, captures)
        if log:
            log(f"prefill: {sum(map(len, prompts))} tokens of {len(prompts)} prompts")
        rows = [h[(keep_from or [0] * len(hs))[i]:] for i, h in enumerate(hs)]
        logits = list(self.head(torch.cat(rows)).split([len(r) for r in rows]))
        out, steps = [[int(lg[-1].argmax())] for lg in logits], [[] for _ in prompts]
        for _ in range(n - 1):
            hs = self.run_all([[o[-1]] for o in out], caches)
            for o, st, lg in zip(out, steps, self.head(torch.cat([h[-1:] for h in hs]))):
                o.append(int(lg.argmax()))
                st.append(lg)
            if log:
                log(f"token {len(out[0])}/{n}")
        return logits, out, [torch.stack(st) if st else torch.empty(0) for st in steps]

    def greedy(self, ids: list[int], n: int, capture: dict | None = None) -> tuple[torch.Tensor, list[int]]:
        """One prompt: (logits of every position, n argmax tokens)."""
        logits, out, _ = self.greedy_all([ids], n, [capture])
        return logits[0], out[0]


RAW = ["The capital of France is",
       "भारत की राजधानी नई दिल्ली है और",
       "عاصمة مصر هي",
       "中国的首都是",
       "Mji mkuu wa Kenya ni",
       "def fibonacci(n):",
       ("The history of computing is a story of abstraction. Each generation of engineers built tools that hid the "
        "details of the layer below, so the next generation could think in bigger pieces.")]
CHAT = ["What is the capital of France? Answer in one word.",
        "भारत की राजधानी क्या है? एक शब्द में उत्तर दें।"]


LONG_TOKENS_PER_LANGUAGE = 600          # 8 languages -> ~4.8K tokens, past the 4096-token window
LONG_CHUNK = 512                        # prefill tokens per pass (bounds the [heads, T, S] attention scores)


def long_rows(n: int) -> list[int]:
    """The rows whose logits the long key keeps: every 64th, the window's edge (4090-4100), the last 8."""
    return sorted(set(range(0, n, 64)) | set(range(4090, min(4101, n))) | set(range(max(0, n - 8), n)))


def main_long(d: str, out: Path) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import long_texts
    tok, ref = AutoTokenizer.from_pretrained(d), StreamedCohere2(d)

    class Enc:
        @staticmethod
        def encode(text):
            return tok(text, add_special_tokens=False)["input_ids"]
    ids = tok(long_texts.build(Enc, LONG_TOKENS_PER_LANGUAGE))["input_ids"]          # BOS first: a raw prompt
    t0, cache, hs = time.perf_counter(), DynamicCache(config=ref.config), []
    for i in range(0, len(ids), LONG_CHUNK):
        hs.append(ref.run_all([ids[i:i + LONG_CHUNK]], [cache])[0])
        print(f"  {time.perf_counter() - t0:5.0f} s  prefill {min(i + LONG_CHUNK, len(ids))}/{len(ids)}", flush=True)
    h = torch.cat(hs)
    rows = long_rows(len(ids))
    logits = ref.head(h[rows])
    greedy, steps = [int(logits[-1].argmax())], []
    for _ in range(9):
        lg = ref.head(ref.run_all([[greedy[-1]]], [cache])[0][-1:])[0]
        greedy.append(int(lg.argmax())); steps.append(lg)
    torch.save({"ids": torch.tensor(ids), "final_norm": h.clone(), "rows": torch.tensor(rows), "logits": logits.clone(),
                "greedy": torch.tensor(greedy), "step_logits": torch.stack(steps),
                "tokens_per_language": LONG_TOKENS_PER_LANGUAGE, "languages": list(long_texts.SOURCES)}, out / "0.pt")
    print(f"{len(ids)} tokens, {len(rows)} logit rows -> {tok.decode(greedy)!r} in {time.perf_counter() - t0:.0f} s")


def main() -> None:
    if "--long" in sys.argv:
        args = [a for a in sys.argv[1:] if a != "--long"]
        out = Path(args[1] if len(args) > 1 else "tests/golden_aya_long")
        out.mkdir(parents=True, exist_ok=True)
        return main_long(args[0] if args else "models/tiny-aya-global", out)
    d = sys.argv[1] if len(sys.argv) > 1 else "models/tiny-aya-global"
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "tests/golden_aya")
    out.mkdir(parents=True, exist_ok=True)
    tok, ref = AutoTokenizer.from_pretrained(d), StreamedCohere2(d)
    prompts = [(t, False, tok(t)["input_ids"]) for t in RAW]
    prompts += [(t, True, list(tok.apply_chat_template([{"role": "user", "content": t}], tokenize=True,
                                                       add_generation_prompt=True)["input_ids"])) for t in CHAT]
    t0 = time.perf_counter()
    caps = [{} for _ in prompts]                     # chat prompts: only final_norm is kept (every position)
    keep = [len(ids) - 8 if chat else 0 for _, chat, ids in prompts]
    logits, greedy, steps = ref.greedy_all([ids for _, _, ids in prompts], 10, caps, keep,
                                    log=lambda m: print(f"  {time.perf_counter() - t0:5.0f} s  {m}", flush=True))
    for n, ((text, chat, ids), lg, gr, st, cap) in enumerate(zip(prompts, logits, greedy, steps, caps)):
        rec = {"text": text, "chat": chat, "ids": torch.tensor(ids), "greedy": torch.tensor(gr), "logits": lg.clone(),
               "logits_from": keep[n], "step_logits": st.clone(),
               **({"final_norm": cap["final_norm"]} if chat else cap)}
        torch.save(rec, out / f"{n}.pt")
        print(f"{n}: {len(ids)} tokens{' (chat)' if chat else ''} -> {tok.decode(gr)!r}", flush=True)
    print(f"{len(prompts)} prompts in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
