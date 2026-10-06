"""Load generator: closed-loop clients at several concurrency levels against a running server.

usage: loadgen.py [--url http://127.0.0.1:8000] [--levels 1,2,4,8] [--requests 16] [--max-tokens 64]
                  [--temperature 0.7] [--out FILE.md]

Each client streams a completion, sends the next one as soon as it finishes, and records TTFT (time to the first
streamed token), TPOT (mean gap between later tokens) and end-to-end latency. Aggregate throughput is total
generated tokens / wall time (read from the server's own counter). Requests set the temperature (default 0.7) and
a fixed seed and leave top-k / top-p to the model's own values, so the server's per-token sampling cost is included; the
prompts ask for long answers so most requests run to max_tokens and levels stay comparable.
"""
import argparse
import asyncio
import json
import statistics
import time

import httpx

PROMPTS = ["Write a short story about a robot learning to paint.",
           "Explain how a CPU cache works to a new programmer.",
           "List ten facts about the ocean, one per line.",
           "Describe the water cycle in detail.",
           "What are the trade-offs between SQL and NoSQL databases?",
           "Write a poem about the city at night.",
           "Explain gradient descent step by step.",
           "Give a recipe for a simple vegetable soup."]


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))] if xs else float("nan")


async def one(client, url, prompt, max_tokens, temperature=0.7, seed=0):
    t0 = time.perf_counter()
    stamps = []
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": temperature,
            "seed": seed, "stream": True}
    async with client.stream("POST", f"{url}/v1/chat/completions", json=body) as r:
        if r.status_code != 200:
            return {"status": r.status_code}
        async for line in r.aiter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                event = json.loads(line[6:])
                if "error" in event:                           # server-side failure mid-stream
                    return {"status": "error"}
                if event["choices"][0].get("delta", {}).get("content"):
                    stamps.append(time.perf_counter())
    end = time.perf_counter()
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    return {"status": 200, "ttft": stamps[0] - t0 if stamps else None, "tpot": statistics.mean(gaps) if gaps else None,
            "e2e": end - t0, "chunks": len(stamps)}


async def level(url, conc, n_req, max_tokens, temperature):
    async with httpx.AsyncClient(timeout=600) as client:
        before = await usage(client, url)
        queue = list(range(n_req))
        out = []

        async def worker(wid):
            while queue:
                i = queue.pop()
                out.append(await one(client, url, PROMPTS[i % len(PROMPTS)], max_tokens, temperature, seed=i))

        t0 = time.perf_counter()
        await asyncio.gather(*[worker(w) for w in range(conc)])
        wall = time.perf_counter() - t0
        after = await usage(client, url)
    ok = [r for r in out if r["status"] == 200]
    gen = after - before
    return {"conc": conc, "ok": len(ok), "rejected": len(out) - len(ok), "wall": wall, "tok_s": gen / wall,
            "ttft50": pct([r["ttft"] for r in ok if r["ttft"]], 50), "ttft95": pct([r["ttft"] for r in ok if r["ttft"]], 95),
            "tpot50": pct([r["tpot"] for r in ok if r["tpot"]], 50), "e2e50": pct([r["e2e"] for r in ok], 50),
            "e2e95": pct([r["e2e"] for r in ok], 95)}


async def usage(client, url) -> float:
    """Server-side generated-token counter (exact, independent of how text was chunked)."""
    for line in (await client.get(f"{url}/metrics")).text.splitlines():
        if line.startswith("engine_generation_tokens_total "):
            return float(line.split()[1])
    return 0.0


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--levels", default="1,2,4,8")
    ap.add_argument("--requests", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    async with httpx.AsyncClient(timeout=600) as c:
        ready = (await c.get(f"{a.url}/ready")).json()
        # warm-up: every prompt shape, and batch sizes up to 8, so first-use costs stay out of the measured levels
        await asyncio.gather(*[one(c, a.url, p, 8, a.temperature) for p in PROMPTS])
    rows = []
    for conc in [int(x) for x in a.levels.split(",")]:
        r = await level(a.url, conc, max(a.requests, conc), a.max_tokens, a.temperature)
        rows.append(r)
        print(f"conc {conc}: {r['tok_s']:.1f} tok/s  TTFT p50 {r['ttft50'] * 1e3:.0f} ms  "
              f"TPOT p50 {r['tpot50'] * 1e3:.1f} ms  e2e p50 {r['e2e50']:.2f}s  ({r['ok']} ok, {r['rejected']} rejected)")
    table = [f"Model `{ready['model']}`, backend `{ready['backend']}`, {a.requests} requests per level, "
             f"max_tokens {a.max_tokens}, temperature {a.temperature} (top-k, top-p: the model's), streaming chat "
             f"completions.\n",
             "| concurrency | throughput (tok/s) | TTFT p50 | TTFT p95 | TPOT p50 | e2e p50 | e2e p95 | rejected |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        table.append(f"| {r['conc']} | {r['tok_s']:.1f} | {r['ttft50'] * 1e3:.0f} ms | {r['ttft95'] * 1e3:.0f} ms | "
                     f"{r['tpot50'] * 1e3:.1f} ms | {r['e2e50']:.2f} s | {r['e2e95']:.2f} s | {r['rejected']} |")
    print("\n".join(table))
    if a.out:
        with open(a.out, "w") as f:
            f.write("\n".join(table) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
