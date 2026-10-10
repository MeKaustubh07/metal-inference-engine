"""Per-language 4-bit damage: KL divergence of a build's next-token distribution from a reference, on long text.

usage: lang_kl.py ref aya|qwen OUT_DIR                     reference (Tiny Aya fp32 on the CPU, layer by layer as
                                                          scripts/golden_aya.py; Qwen3.5-2B HF bf16), one file per language
       lang_kl.py ours MODEL BACKEND QT|- REF_DIR OUT.json [--keep none|shipped|topN]
                                                          this engine (QT "-": quantized at load; --keep: the INT8 tensors)
       lang_kl.py mlx MODEL_DIR REF_DIR OUT.json [--recipe default|mixed_4_6|q4km_bits]
                                                          MLX-LM (run with an MLX-LM venv's python); with --recipe the HF
                                                          checkpoint is quantized in memory under that recipe
       lang_kl.py llama GGUF REF_DIR OUT_DIR              llama.cpp's own llama-perplexity --kl-divergence on each file
       lang_kl.py table RESULTS_DIR                       a markdown table of every result in RESULTS_DIR

Text: two 512-token windows per language (8 languages) cut from the public-domain works of scripts/long_texts.py at 25%
and 60% of each work, past its famous opening; Tiny Aya's windows start with BOS, Qwen3.5's have none. Scored: positions
256-510 of each window (every one with 256+ tokens of context), as llama-perplexity scores its second half.
The reference is stored as llama.cpp's KL base file (tools/perplexity/perplexity.cpp): "_logits_", int32 n_ctx, n_vocab,
n_chunk, the tokens, then per chunk n_ctx - 1 - n_ctx/2 rows of nv = 2*((n_vocab+1)/2) + 4 uint16: float32 scale and
min_log_prob, then each logit as rint((logit - min) / scale), min = max(min logit, max - 16). Every build is scored with
llama.cpp's formula: sum over tokens with base log p > -16 of p_base * (log p_base - log q). llama.cpp reads these files
exactly as score() does (its base perplexity equals ours to 5 digits).

Tiny Aya's mlx recipes (group 64, affine, MLX-LM's rounding): default = 4 bits everywhere (= mlx_lm convert -q, the
tied embedding/head included); mixed_4_6 = MLX-LM's recipe (v_proj and down_proj at 6 bits in llama.cpp Q4_K_M's layers,
lm_head at 6; Tiny Aya has no lm_head, its tied embedding stays 4-bit); q4km_bits = the official Q4_K_M GGUF's bit layout
(mixed_4_6 plus the tied embedding at 6 bits).
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import long_texts  # noqa: E402

N_CTX = 512
LANGS = tuple(long_texts.SOURCES)                  # en fr de es hi ar zh ja
AT = (0.25, 0.60)                                  # where in each work the two windows start
SPAN = 6000                                        # characters tokenized per window (more than N_CTX tokens in each)
MAGIC = b"_logits_"


def windows(encode, bos: int | None) -> list[dict]:
    """Two windows per language: BOS (if any) + text tokens, N_CTX ids each, from a word boundary where there are spaces."""
    out, n_text = [], N_CTX - (bos is not None)
    for lang in LANGS:
        text = long_texts.section(lang)
        for k, at in enumerate(AT):
            off = int(len(text) * at)
            if lang not in ("zh", "ja"):
                while off < len(text) and not text[off].isspace():
                    off += 1
                while off < len(text) and text[off].isspace():
                    off += 1
            ids = encode(text[off:off + SPAN])[:n_text]
            assert len(ids) == n_text, (lang, k, len(ids))
            out.append(dict(lang=lang, k=k, char_off=off, ids=([bos] if bos is not None else []) + ids))
    return out


# --- llama.cpp's KL base file -------------------------------------------------------------------------------------

def nv_of(n_vocab: int) -> int:
    return 2 * ((n_vocab + 1) // 2) + 4


def encode_rows(logits: np.ndarray) -> np.ndarray:
    """fp32 logits [R, V] -> uint16 rows [R, nv] (llama.cpp's log_softmax into uint16)."""
    R, V = logits.shape
    out = np.zeros((R, nv_of(V)), dtype=np.uint16)
    for r in range(R):
        lg = logits[r].astype(np.float32)
        mx, mn = float(lg.max()), max(float(lg.min()), float(lg.max()) - 16.0)
        lse = float(np.log(np.exp((lg - mx).astype(np.float64)).sum()))
        scale = np.float32((mx - mn) / 65535.0)
        out[r, :4] = np.array([scale, np.float32(mn - mx - lse)], dtype=np.float32).view(np.uint16)
        out[r, 4:4 + V] = np.where(lg > mn, np.rint((lg - mn) * np.float32(1.0 / scale)), 0).astype(np.uint16)
    return out


def write_header(f, n_vocab: int, chunks: list[list[int]]) -> None:
    f.write(MAGIC)
    np.array([N_CTX, n_vocab, len(chunks)], dtype=np.int32).tofile(f)
    np.array([t for c in chunks for t in c], dtype=np.int32).tofile(f)


def read(path) -> tuple[int, int, np.ndarray, np.ndarray]:
    """-> (n_ctx, n_vocab, tokens [n_chunk, n_ctx], rows uint16 [n_chunk, n_ctx - 1 - n_ctx/2, nv], memory-mapped)."""
    with open(path, "rb") as f:
        assert f.read(8) == MAGIC, path
        n_ctx, n_vocab, n_chunk = (int(x) for x in np.fromfile(f, dtype=np.int32, count=3))
        tokens = np.fromfile(f, dtype=np.int32, count=n_ctx * n_chunk).reshape(n_chunk, n_ctx)
        off = f.tell()
    rows = np.memmap(path, dtype=np.uint16, mode="r", offset=off,
                     shape=(n_chunk, n_ctx - 1 - n_ctx // 2, nv_of(n_vocab)))
    return n_ctx, n_vocab, tokens, rows


def score(rows, logits: np.ndarray, targets: np.ndarray, block: int = 16) -> dict:
    """llama.cpp's per-row statistics of logits [R, V] against base rows [R, nv]; targets [R] = the next tokens.
    -> arrays kld, same_top, nll, nll_base. In blocks of rows (a float64 row of 262K entries is 2 MB)."""
    parts = []
    for i in range(0, len(logits), block):
        b, lg, t = np.asarray(rows[i:i + block]), logits[i:i + block].astype(np.float64), targets[i:i + block]
        V = lg.shape[1]
        head = np.ascontiguousarray(b[:, :4]).view(np.float32)                # scale, min_log_prob
        plb = head[:, :1] * b[:, 4:4 + V].astype(np.float32) + head[:, 1:2]
        lg -= lg.max(axis=1, keepdims=True)
        logq = lg - np.log(np.exp(lg).sum(axis=1, keepdims=True))
        kld = np.where(plb > -16.0, np.exp(plb.astype(np.float64)) * (plb - logq), 0.0).sum(axis=1)
        idx = np.arange(len(lg))
        parts.append(dict(kld=kld, same_top=b[:, 4:4 + V].argmax(axis=1) == logits[i:i + block].argmax(axis=1),
                          nll=-logq[idx, t], nll_base=-plb[idx, t].astype(np.float64)))
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def summarize(parts: list[dict]) -> dict:
    a = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    return dict(n=int(len(a["kld"])), kld_mean=float(a["kld"].mean()), kld_median=float(np.median(a["kld"])),
                kld_p99=float(np.percentile(a["kld"], 99)), same_top=float(a["same_top"].mean()),
                ln_ppl_ratio=float(a["nll"].mean() - a["nll_base"].mean()),
                ppl=float(np.exp(a["nll"].mean())), ppl_base=float(np.exp(a["nll_base"].mean())))


def run_all(ref_dir: str, out: str, logits_of, meta: dict) -> None:
    """logits_of(ids: np.ndarray) -> fp32 logits [N_CTX, vocab]; every language file in ref_dir scored, saved to out."""
    res, t0 = {}, time.perf_counter()
    for f in sorted(Path(ref_dir).glob("*.kld")):
        n_ctx, V, toks, rows = read(f)
        first, parts = n_ctx // 2, []
        for c in range(len(toks)):
            lg = logits_of(toks[c])
            assert lg.shape == (n_ctx, V), (lg.shape, V)
            parts.append(score(rows[c], lg[first:n_ctx - 1], toks[c, first + 1:]))
        r = res[f.stem] = summarize(parts)
        print(f"{f.stem}: KLD {r['kld_mean']:.4f} (median {r['kld_median']:.4f}, p99 {r['kld_p99']:.3f}) same top "
              f"{r['same_top']:.1%} ln(PPL ratio) {r['ln_ppl_ratio']:+.4f} PPL {r['ppl']:.2f} vs {r['ppl_base']:.2f}  "
              f"[{time.perf_counter() - t0:.0f} s]", flush=True)
    json.dump(res | {"_meta": meta}, open(out, "w"), indent=1)


# --- references ---------------------------------------------------------------------------------------------------

def ref_aya(out: Path) -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch
    from transformers import AutoTokenizer
    from transformers.cache_utils import DynamicCache
    from golden_aya import StreamedCohere2
    d = str(ROOT / "models/tiny-aya-global")
    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(d)
    W = windows(lambda t: tok(t, add_special_tokens=False)["input_ids"], tok.bos_token_id)
    json.dump(W, open(out / "windows.json", "w"))
    ref, first = StreamedCohere2(d), N_CTX // 2
    for g in range(0, len(LANGS), 2):                  # two languages (4 windows) per pass over the 36 layers
        group = [w for w in W if w["lang"] in LANGS[g:g + 2]]
        hs = ref.run_all([w["ids"] for w in group], [DynamicCache(config=ref.config) for _ in group])
        for lang in LANGS[g:g + 2]:
            mine = [i for i, w in enumerate(group) if w["lang"] == lang]
            with open(out / f"{lang}.kld", "wb") as f:
                write_header(f, ref.config.vocab_size, [group[i]["ids"] for i in mine])
                for i in mine:
                    encode_rows(ref.head(hs[i][first:N_CTX - 1]).numpy()).tofile(f)
            print(f"wrote {lang}.kld", flush=True)


def ref_qwen(out: Path) -> None:
    import torch
    from tokenizers import Tokenizer
    from transformers import AutoModelForCausalLM
    d = str(ROOT / "models/qwen3.5-2b")
    torch.set_grad_enabled(False)
    tok = Tokenizer.from_file(f"{d}/tokenizer.json")                 # the engine's ids (see scripts/golden_qwen35.py)
    W = windows(lambda t: tok.encode(t).ids, None)
    json.dump(W, open(out / "windows.json", "w"))
    model = AutoModelForCausalLM.from_pretrained(d, dtype=torch.bfloat16, attn_implementation="eager").eval()
    first = N_CTX // 2
    for lang in LANGS:
        mine = [w for w in W if w["lang"] == lang]
        with open(out / f"{lang}.kld", "wb") as f:
            for i, w in enumerate(mine):
                lg = model(torch.tensor([w["ids"]]), use_cache=False).logits[0, first:N_CTX - 1].float().numpy()
                if i == 0:
                    write_header(f, lg.shape[1], [x["ids"] for x in mine])
                encode_rows(lg).tofile(f)
        print(f"wrote {lang}.kld", flush=True)


# --- builds -------------------------------------------------------------------------------------------------------

def ours(model: str, backend: str, qt: str, ref_dir: str, out: str, keep: str | None) -> None:
    import torch
    sys.path.insert(0, str(ROOT / "src"))
    import engine
    if keep:                                           # an INT4 keep-INT8 list in place of the shipped policy
        spec = engine.MODELS[model]
        src = json.load(open(ROOT / spec["policy"]))
        names = ([] if keep == "none" else src["keep_int8"] if keep == "shipped" else
                 [k for k, v in sorted(src["damage"].items(), key=lambda kv: -(kv[1]["pos0"] + kv[1]["later"]))
                  [:int(keep[3:])]])
        pol = Path(out).with_suffix(".policy.json")
        json.dump({"keep_int8": names}, open(pol, "w"), indent=1)
        spec["policy"] = str(pol.resolve())            # ROOT / an absolute path == that path
        print(f"{model} INT4, kept INT8: {names or '(only the tied embedding)'}", flush=True)
    torch.set_grad_enabled(False)
    m = engine.load_engine(model, backend, None if qt == "-" else qt).model

    def logits_of(ids):
        return m.forward(torch.tensor(ids.astype(np.int64)), state=m.new_state(N_CTX + 8)).float().cpu().numpy()
    run_all(ref_dir, out, logits_of, dict(model=model, backend=backend, qt=qt, keep=keep))


def mlx(path: str, ref_dir: str, out: str, recipe: str | None) -> None:
    import mlx.core as mx
    from mlx_lm.utils import load, quantize_model
    if recipe is None:
        model, _ = load(path)
    else:
        model, _, config = load(path, lazy=True, return_config=True)
        L = len(model.layers)

        def pred(p: str, module) -> dict:
            parts, bits = p.split("."), 4
            i = int(parts[2]) if len(parts) > 2 and parts[1] == "layers" and parts[2].isdigit() else None
            more = i is not None and (i < L // 8 or i >= 7 * L // 8 or (i - L // 8) % 3 == 2)   # llama.cpp's use_more_bits
            if recipe in ("mixed_4_6", "q4km_bits") and more and (p.endswith("v_proj") or p.endswith("down_proj")):
                bits = 6
            if recipe == "q4km_bits" and p.endswith("embed_tokens"):
                bits = 6
            return {"group_size": 64, "bits": bits, "mode": "affine"}
        assert recipe in ("default", "mixed_4_6", "q4km_bits"), recipe
        model, _ = quantize_model(model, config, 64, 4, quant_predicate=pred)
        mx.eval(model.parameters())

    def logits_of(ids):
        return np.array(model(mx.array(ids[None].astype(np.int32)))[0].astype(mx.float32))
    run_all(ref_dir, out, logits_of, dict(engine="mlx-lm", model=path, recipe=recipe))


def llama(gguf: str, ref_dir: str, out_dir: str) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for f in sorted(Path(ref_dir).glob("*.kld")):
        log = subprocess.run(["llama-perplexity", "-m", gguf, "--kl-divergence-base", str(f), "--kl-divergence",
                              "-c", str(N_CTX), "-b", str(N_CTX), "-ngl", "99"], capture_output=True, text=True)
        (Path(out_dir) / f"{f.stem}.txt").write_text(log.stdout + log.stderr)
        r = llama_result(Path(out_dir) / f"{f.stem}.txt")
        print(f"{f.stem}: KLD {r['kld_mean']:.4f} same top {r['same_top']:.1%} ln(PPL ratio) {r['ln_ppl_ratio']:+.4f}",
              flush=True)


def llama_result(path: Path) -> dict:
    t = path.read_text()
    g = lambda k: float(re.search(k + r"\s*:\s*(-?[\d.]+)", t).group(1))
    return dict(kld_mean=g(r"Mean\s+KLD"), same_top=g(r"Same top p") / 100,
                ln_ppl_ratio=g(r"Mean ln\(PPL\(Q\)/PPL\(base\)\)"), ppl_base=g(r"Mean PPL\(base\)"))


def table(results: str) -> None:
    rows = {p.stem: {k: v for k, v in json.load(open(p)).items() if k in LANGS and isinstance(v, dict)}
            for p in sorted(Path(results).glob("*.json")) if not p.name.endswith(".policy.json")}
    for d in sorted(p for p in Path(results).iterdir() if p.is_dir()):
        rows[d.name] = {f.stem: llama_result(f) for f in d.glob("*.txt") if f.stem in LANGS}
    print("| build | " + " | ".join(LANGS) + " | mean |\n|---|" + "---:|" * (len(LANGS) + 1))
    for name, r in rows.items():
        if all(l in r for l in LANGS):
            print(f"| {name} | " + " | ".join(f"{r[l]['kld_mean']:.4f}" for l in LANGS) +
                  f" | {sum(r[l]['kld_mean'] for l in LANGS) / len(LANGS):.4f} |")


def main() -> None:
    a = sys.argv[1:]
    opt = lambda flag: a[a.index(flag) + 1] if flag in a else None
    if not a:
        sys.exit(__doc__)
    if a[0] == "ref":
        out = Path(a[2]); out.mkdir(parents=True, exist_ok=True)
        {"aya": ref_aya, "qwen": ref_qwen}[a[1]](out)
    elif a[0] == "ours":
        ours(*a[1:6], keep=opt("--keep"))
    elif a[0] == "mlx":
        mlx(*a[1:4], recipe=opt("--recipe"))
    elif a[0] == "llama":
        llama(*a[1:4])
    elif a[0] == "table":
        table(a[1])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
