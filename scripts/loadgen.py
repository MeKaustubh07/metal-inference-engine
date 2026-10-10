"""Load generator: closed-loop clients at several concurrency levels against a running server.

usage: loadgen.py [--url http://127.0.0.1:8000] [--levels 1,2,4,8] [--requests 16] [--max-tokens 64]
                  [--temperature 0.7] [--burst N --rounds R] [--out FILE.md]

Each client streams a completion, sends the next one as soon as it finishes, and records TTFT (time to the first
streamed token), TPOT (mean gap between later tokens) and end-to-end latency. Aggregate throughput is total
generated tokens / wall time (read from the server's own counter). Requests set the temperature (default 0.7) and
a fixed seed and leave top-k / top-p to the model's own values, so the server's per-token sampling cost is included; the
prompts ask for long answers so most requests run to max_tokens and levels stay comparable.

--burst N --rounds R (instead of the levels): N chats sent at once, all awaited, R times; TTFT p50/p95 over the N x R
requests (arrivals that come together share the engine's first prefill passes: the case a pinned preamble speeds up).
Both modes read /metrics before the measured requests and after them: the header gives the pinned preamble's KV units,
the end the prefix cache's hits and bypasses among the measured requests and, once the server is idle, whether
kv_blocks_free + prefix_pinned_units == kv_blocks_total (no unit leaked or lost; N/A on a server without them).
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


async def burst(url, n, rounds, max_tokens, temperature):
    """n chats sent at once and awaited, `rounds` times (seeds differ per request, as in level)."""
    out = []
    async with httpx.AsyncClient(timeout=600, limits=httpx.Limits(max_connections=None)) as client:   # all at once
        for k in range(rounds):
            t0 = time.perf_counter()
            got = await asyncio.gather(*[one(client, url, PROMPTS[i % len(PROMPTS)], max_tokens, temperature,
                                             seed=k * n + i) for i in range(n)])
            wall = time.perf_counter() - t0
            ok = [r for r in got if r["status"] == 200]
            ttft = [r["ttft"] for r in ok if r["ttft"]]
            rej = sum(r["status"] == 429 for r in got)              # a full queue; anything else is a failure
            print(f"round {k + 1}: TTFT p50 {pct(ttft, 50) * 1e3:.0f} ms  max {pct(ttft, 100) * 1e3:.0f} ms  "
                  f"wall {wall:.2f}s  ({len(ok)} ok, {rej} rejected, {n - len(ok) - rej} failed)")
            out += got
    ok = [r for r in out if r["status"] == 200]
    ttft = [r["ttft"] for r in ok if r["ttft"]]
    rej = sum(r["status"] == 429 for r in out)
    return {"n": n, "rounds": rounds, "ok": len(ok), "rejected": rej, "failed": len(out) - len(ok) - rej,
            "timed": len(ttft),
            "ttft50": pct(ttft, 50), "ttft95": pct(ttft, 95), "e2e50": pct([r["e2e"] for r in ok], 50)}


async def scrape(client, url) -> dict[str, float]:
    """The server's unlabelled /metrics samples, by name without the engine_ prefix (empty if it has none)."""
    r = await client.get(f"{url}/metrics")
    if r.status_code != 200:
        return {}
    pairs = [line.split() for line in r.text.splitlines() if line.startswith("engine_") and "{" not in line]
    return {p[0][7:]: float(p[1]) for p in pairs if len(p) == 2}


async def usage(client, url) -> float:
    """Server-side generated-token counter (exact, independent of how text was chunked)."""
    return (await scrape(client, url)).get("generation_tokens_total", 0.0)


async def idle(client, url, timeout=10.0) -> tuple[dict[str, float], float]:
    """/metrics once the server has no request in flight, or after timeout -> (samples, requests still in flight)."""
    deadline = time.perf_counter() + timeout
    while True:
        m = await scrape(client, url)
        busy = sum(m.get(k, 0) for k in ("running_requests", "prefilling_requests", "waiting_requests"))
        if not busy or time.perf_counter() > deadline:
            return m, busy
        await asyncio.sleep(0.05)


def pinned(m) -> str:
    """The header's note on the pinned preamble (nothing from a server that does not report it)."""
    if "prefix_pinned_units" not in m:
        return ""
    return f", prefix cache: {m['prefix_pinned_units']:.0f} KV units pinned" if m["prefix_pinned_units"] else \
        ", no prefix cache"


def footer(before, after, busy, served) -> list[str]:
    """Prefix cache hits and bypasses among the measured requests; the idle pool's units all accounted for."""
    if "prefix_hits_total" in after:
        hits, byp = (after.get(k, 0) - before.get(k, 0) for k in ("prefix_hits_total", "prefix_bypasses_total"))
        lines = [f"prefix cache: {hits:.0f} hits, {byp:.0f} bypasses over the {served} requests measured"]
        pt, pf, pre = (after.get(k, 0) - before.get(k, 0)
                       for k in ("prompt_tokens_total", "prefill_tokens_total", "requests_preempted_total"))
        lines.append(f"prefill: {pf:.0f} tokens prefilled for {pt:.0f} prompt tokens ({pre:.0f} preemptions)")
    else:
        lines = ["prefix cache: N/A (the server reports no prefix_hits_total)"]
    if "kv_blocks_free" not in after or "kv_blocks_total" not in after:
        lines.append("KV pool: N/A (the server reports no kv_blocks_free / kv_blocks_total)")
    elif busy:
        lines.append(f"KV pool: N/A (the server still has {busy:.0f} requests in flight)")
    else:
        free, pin, total = after["kv_blocks_free"], after.get("prefix_pinned_units", 0.0), after["kv_blocks_total"]
        ok = free + pin == total
        lines.append(f"KV pool, server idle: {free:.0f} free + {pin:.0f} pinned {'==' if ok else '!='} {total:.0f} "
                     f"units: {'ok' if ok else 'MISMATCH'}")
    return lines


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--levels", default="1,2,4,8")
    ap.add_argument("--requests", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--burst", type=int, default=0, help="N chats sent at once, --rounds times (instead of --levels)")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.burst < 0 or a.rounds < 1:
        ap.error("--burst must be at least 0 and --rounds at least 1")
    async with httpx.AsyncClient(timeout=600) as c:
        ready = (await c.get(f"{a.url}/ready")).json()
        # warm-up: every prompt shape, and batch sizes up to 8, so first-use costs stay out of the measured levels
        await asyncio.gather(*[one(c, a.url, p, 8, a.temperature) for p in PROMPTS])
        before = await scrape(c, a.url)
    sampling = f"max_tokens {a.max_tokens}, temperature {a.temperature} (top-k, top-p: the model's)"
    if a.burst:
        r = await burst(a.url, a.burst, a.rounds, a.max_tokens, a.temperature)
        served = r["ok"]
        print(f"burst {a.burst} x {a.rounds}: TTFT p50 {r['ttft50'] * 1e3:.0f} ms  p95 {r['ttft95'] * 1e3:.0f} ms  "
              f"e2e p50 {r['e2e50']:.2f}s  ({r['ok']} ok, {r['rejected']} rejected, {r['failed']} failed, "
              f"{r['timed']} with a first token)")
        table = [f"Model `{ready['model']}`, backend `{ready['backend']}`, bursts of {a.burst} chats sent at once, "
                 f"{a.rounds} rounds, {sampling}, streaming chat completions{pinned(before)}.\n",
                 "| burst | rounds | TTFT p50 | TTFT p95 | e2e p50 | rejected | failed |",
                 "|---:|---:|---:|---:|---:|---:|---:|",
                 f"| {a.burst} | {a.rounds} | {r['ttft50'] * 1e3:.0f} ms | {r['ttft95'] * 1e3:.0f} ms | "
                 f"{r['e2e50']:.2f} s | {r['rejected']} | {r['failed']} |"]
    else:
        rows = []
        for conc in [int(x) for x in a.levels.split(",")]:
            r = await level(a.url, conc, max(a.requests, conc), a.max_tokens, a.temperature)
            rows.append(r)
            print(f"conc {conc}: {r['tok_s']:.1f} tok/s  TTFT p50 {r['ttft50'] * 1e3:.0f} ms  TPOT p50 "
                  f"{r['tpot50'] * 1e3:.1f} ms  e2e p50 {r['e2e50']:.2f}s  ({r['ok']} ok, {r['rejected']} rejected)")
        served = sum(r["ok"] for r in rows)
        table = [f"Model `{ready['model']}`, backend `{ready['backend']}`, {a.requests} requests per level, "
                 f"{sampling}, streaming chat completions{pinned(before)}.\n",
                 "| concurrency | throughput (tok/s) | TTFT p50 | TTFT p95 | TPOT p50 | e2e p50 | e2e p95 | rejected |",
                 "|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in rows:
            table.append(f"| {r['conc']} | {r['tok_s']:.1f} | {r['ttft50'] * 1e3:.0f} ms | {r['ttft95'] * 1e3:.0f} ms"
                         f" | {r['tpot50'] * 1e3:.1f} ms | {r['e2e50']:.2f} s | {r['e2e95']:.2f} s | {r['rejected']} |")
    async with httpx.AsyncClient(timeout=600) as c:
        after, busy = await idle(c, a.url)
    table += [""] + footer(before, after, busy, served)
    print("\n".join(table))
    if a.out:
        with open(a.out, "w") as f:
            f.write("\n".join(table) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
