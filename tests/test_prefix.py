"""The pinned chat preamble (src/server/prefix.py; Scheduler / create_app(prefix_cache=True), serve.py --prefix-cache):
it builds where it can (P = the probe chats' common prefix in whole blocks) and is refused where it cannot; a sequence
that borrows it and prefills the rest == a fresh one prefilled with a chunk boundary at P, bit for bit; borrow and the
write guard refuse what would break the holder; served with the cache == without it (P fewer prefill tokens per hit),
with bypasses, misses whose rings wrap, preemption and the never-fit check; HTTP and /metrics.

usage: test_prefix.py [--target qwen|tiny|aya-int4|aya-int8]   the serving target (tests/serving_targets.py; default
qwen). Prefill chunks of 128 on tiny (a ring of 10 blocks: hits up to 160 tokens), 512 on aya (289 blocks: 4,624).
qwen has no windowed pool: the cache is refused, nothing else changes, the other checks print N/A. On tiny the pools
are NaN-poisoned (a stale read shows); on aya (Metal, not exact) served greedy runs follow the near-tie rule against
references prefilled as a hit is (split at P), and GPU memory and Metal aborts are gated."""
import asyncio
import functools
import queue
import sys
import time
from types import SimpleNamespace

import httpx
import torch

sys.path.insert(0, "src")
sys.path.insert(0, "tests")
from chat import format_chat
from sampler import SamplingParams
from server.app import create_app
from server.metrics import Metrics
from server.prefix import PROBES, PinnedPrefix
from server.scheduler import Scheduler
from serving_targets import PROMPTS, Served, gpu_gates, load_target, na, notes, target_from_argv, unload_target

torch.set_grad_enabled(False)
results = []
def check(name, ok):
    results.append(bool(ok)); print(f"{'PASS' if ok else 'FAIL'}  {name}{notes()}")

GREEDY = SamplingParams(temperature=0, repetition_penalty=1.0)
GR = {"temperature": 0, "repetition_penalty": 1.0}
CHATS = ["Explain in detail how a steam engine works.", "Describe the history of the Roman Empire.",
         "Tell me a long story about a brave knight."]          # hits; long answers (tiny: none ends within 60 tokens)
FILLER = "Inference engines turn trained weights into answers, one token at a time. "
M_HIT, BS, LEN = 40, 16, 8192                                   # a hit's max_tokens, block size, max_model_len
TARGET = target_from_argv()
T = load_target(TARGET)
eng, tok = T.engine, T.engine.tokenizer
V, EOS, name = tok.vocab_size(), set(eng.eos_ids), T.name
CHUNK = 128 if TARGET == "tiny" else 512
POISON = TARGET == "tiny"                                       # Metal's kernels are not proven on NaN-poisoned pools
if POISON:                                                      # the Scheduler passes no poison flag
    eng.model.new_paged_pool = functools.partial(eng.model.new_paged_pool, poison=True)
EXPECT_P = {"tiny": 64, "aya-int4": 352, "aya-int8": 352}.get(TARGET)


def chat_ids(text):                                             # as /v1/chat/completions encodes a one-line chat
    return tok.encode(format_chat([{"role": "user", "content": text}], style=eng.chat_style), add_bos=False)


def longest(make, limit):
    """The longest FILLER x k whose ids make(text) are at most `limit` long -> (ids, text)."""
    l1, l9 = len(make(FILLER)), len(make(FILLER * 9))
    k = max(1, 1 + int((limit - l1) * 8 / (l9 - l1)))
    while k > 1 and len(make(FILLER * k)) > limit:
        k -= 1
    while len(make(FILLER * (k + 1))) <= limit:
        k += 1
    return make(FILLER * k), FILLER * k


def drain(reqs, limit=None):
    """Wait for the requests to finish; False if they don't within the limit (a queue that hangs fails, not stalls)."""
    end = time.perf_counter() + (limit or (900 if T.gpu else 120))
    try:
        for r in reqs:
            while r.out.get(timeout=max(0.01, end - time.perf_counter()))[0] == "token":
                pass
    except queue.Empty:
        return False
    return True


def pinned_kv(pre):
    """A copy of the holder's K and V, every unit it pins."""
    kv = pre.state.kv.pool
    idx = torch.tensor([u for t in pre.state.kv.tables for u in t], device=kv.k.device)
    return kv.k[:, idx].clone(), kv.v[:, idx].clone()


def intact(pre, snap):                                          # bit-unchanged (a NaN would also fail torch.equal)
    k, v = pinned_kv(pre)
    return torch.equal(k, snap[0]) and torch.equal(v, snap[1])


def holder(blocks, poison):
    """A pool shaped as the scheduler's, its preamble pinned as the scheduler pins it -> (pool, PinnedPrefix)."""
    pool = eng.model.new_paged_pool(blocks, BS, max_seqs=4, max_chunk=CHUNK, poison=poison)
    pre, _ = PinnedPrefix.build(SimpleNamespace(pool=pool, eng=eng, tok=tok, model=eng.model, prefill_chunk=CHUNK))
    return pool, pre


def reference(ids, n, cut=0):
    """Greedy alone on a contiguous cache, prefilled in CHUNK pieces from 0, split again at `cut` (a hit: at P, as the
    holder and the borrower compute it) -> (tokens, logits rows: T.agree's reference)."""
    m = eng.model
    st = m.new_state(len(ids) + n + 1)
    bounds = sorted(set(range(0, cut, CHUNK)) | set(range(cut, len(ids), CHUNK)) | {len(ids)})
    for a, b in zip(bounds, bounds[1:]):
        lg = m.forward(torch.tensor(ids[a:b]), state=st, last_only=True)[0]
    toks, rows = [], []
    for _ in range(n):
        rows.append(lg[:V].float().cpu())
        t = int(lg[:V].argmax())
        if t in EOS:
            break
        toks.append(t)
        lg = m.forward(torch.tensor([t]), state=st)[0]
    return toks, rows


def serve(reqs, kv_blocks, cache, max_batch=4):
    """Every (ids, max_tokens, kind) submitted at once to a fresh Scheduler (chunks of CHUNK), all waited for. -> (the
    Requests, its counters, the holder bit-unchanged, every unit but the pinned free once idle, prefix attaches, the
    prompt + generated lengths it preempted)."""
    s = Scheduler(eng, Metrics(), max_batch=max_batch, kv_blocks=kv_blocks, max_model_len=LEN, prefill_chunk=CHUNK,
                  prefix_cache=cache)
    snap, attached, preempted = (pinned_kv(s.prefix) if s.prefix else None), [], []
    if s.prefix:
        attach = s.prefix.attach
        s.prefix.attach = lambda st: (attached.append(1), attach(st))[-1]
    preempt = s._preempt
    s._preempt = lambda r: (preempted.append(len(r.prompt_ids) + len(r.generated)), preempt(r))[-1]
    with s.cond:                                                # all queued before the engine thread looks (RLock)
        got = [s.submit(ids, GREEDY, n) for ids, n, _ in reqs]
    drain(got)
    a = s.pool.allocator
    held, idle = snap is None or intact(s.prefix, snap), a.num_free == a.num_blocks - s.pinned
    counters = dict(s.metrics.counters)
    s.shutdown(5)
    return got, counters, held, idle, len(attached), preempted


async def http(cache, kv_blocks, posts, max_batch=4, limit=None):
    """posts [(url, body)] to a fresh app, concurrently (limit: each must answer within it) -> (responses or
    'timeout', Served, /metrics values)."""
    app = create_app(eng, max_batch=max_batch, max_waiting=8, kv_blocks=kv_blocks, max_model_len=LEN,
                     prefill_chunk=CHUNK, prefix_cache=cache)
    sched = app.state.scheduler
    served = Served(sched)
    async def post(c, url, body):
        try:
            return await asyncio.wait_for(c.post(url, json=body), limit)
        except asyncio.TimeoutError:
            return "timeout"
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t", timeout=1200) as c:
        rs = await asyncio.gather(*[post(c, u, b) for u, b in posts])
        m = (await c.get("/metrics")).text
    sched.shutdown(5)
    return rs, served, {l.split()[0]: float(l.split()[1]) for l in m.splitlines() if l.startswith("engine_")}


def reply(r):                                                   # the reply's text, None for an error
    if r.status_code != 200:
        return None
    ch = r.json()["choices"][0]
    return ch["message"]["content"] if "message" in ch else ch["text"]


# ---------------------------------------------------------------- 1. the cache builds, or is refused with a note
if TARGET == "qwen":                                            # no GroupedKVPool, no TemplateChat
    na("the cache builds: note 'ok', P = the probe chats' common prefix in whole blocks, pinned = blocks_for(P)")
    na("at the serving target's prefill chunk of 8 (a 2-block ring) the cache is refused with a note (tiny)")
    runs = []
    for cache in (True, False):
        s = Scheduler(eng, Metrics(), max_batch=2, kv_blocks=64, prefill_chunk=T.prefill_chunk, prefix_cache=cache)
        with s.cond:
            reqs = [s.submit(ids, GREEDY, 8) for ids in (chat_ids(CHATS[0]), T.raw_ids(PROMPTS[0]))]
        drain(reqs)
        a = s.pool.allocator
        runs.append((s.prefix, s.prefix_note, s.pinned, [r.generated for r in reqs], dict(s.metrics.counters),
                     s.metrics.gauges["prefix_pinned_units"], a.num_free == a.num_blocks))
        s.shutdown(5)
    del s
    (pre, note, pinned, ids_on, c_on, g_on, idle_on), (_, _, _, ids_off, c_off, _, idle_off) = runs
    check(f"{name}: the cache is refused ({note!r}) and nothing else changes: a chat and a raw prompt give the same ids "
          f"and counters as with the cache off (prefill tokens {c_on['prefill_tokens_total']}), nothing pinned, every "
          f"block free once idle", pre is None and note != "ok" and pinned == g_on == 0 and ids_on == ids_off
          and c_on == c_off and c_on["prefix_hits_total"] == c_on["prefix_bypasses_total"] == 0 and idle_on and idle_off)
    usable = False
else:
    na(f"{name}: the cache is refused and nothing else changes (qwen: no windowed KV pool)")
    s = Scheduler(eng, Metrics(), max_batch=2, kv_blocks=64, prefill_chunk=CHUNK, prefix_cache=True)
    enc = [chat_ids(t) for t in PROBES]
    common = next((i for i, (a, b) in enumerate(zip(*enc)) if a != b), min(map(len, enc)))
    probe = eng.model.new_paged_pool(1, BS, max_seqs=1, max_chunk=CHUNK)   # its blocks_for: units held by n positions
    blocks_for, G, R = probe.blocks_for, probe.kv.n_groups, probe.kv.ring
    pre, a = s.prefix, s.pool.allocator
    usable = pre is not None
    P, N, PINNED, CAP = (len(pre.ids), pre.n, s.pinned, pre.cap) if usable else (0, 0, 0, 0)
    check(f"{name}: the cache builds (note {s.prefix_note!r}): P = {P}, the {common}-token common prefix of the probe "
          f"chats in whole blocks (expected {EXPECT_P}); {PINNED} pinned units = blocks_for(P) ({N} blocks x {G} groups);"
          f" cap {CAP} = the {R}-block ring; free {a.num_free} = {a.num_blocks} - pinned, in /metrics too",
          usable and s.prefix_note == "ok" and P == common // BS * BS == EXPECT_P and pre.ids == enc[0][:P]
          and PINNED == pre.units == blocks_for(P) == N * G and N * BS == P and CAP == R * BS
          and a.num_free == a.num_blocks - PINNED == s.metrics.gauges["kv_blocks_free"]
          and s.metrics.gauges["prefix_pinned_units"] == PINNED)
    s.shutdown(5)
    del s, pre
    if TARGET == "tiny":                                        # the serving target's own chunk
        s = Scheduler(eng, Metrics(), max_batch=2, kv_blocks=64, prefill_chunk=T.prefill_chunk, prefix_cache=True)
        a = s.pool.allocator
        check(f"{name}: at prefill chunks of {T.prefill_chunk} (a {s.pool.kv.ring}-block ring) the cache is refused "
              f"({s.prefix_note!r}): nothing pinned, every block free", s.prefix is None and s.prefix_note != "ok"
              and s.pinned == s.metrics.gauges["prefix_pinned_units"] == 0 and a.num_free == a.num_blocks)
        s.shutdown(5)
        del s
    else:
        na(f"{name}: at the serving target's prefill chunk of 8 the cache is refused with a note (tiny)")
T.tidy()

if not usable:                                                  # qwen: refused (a failed build has failed above)
    for line in ["2. a borrower's logits == a fresh sequence split at P (torch.equal), over decode steps",
                 "2. the holder bit-unchanged; borrowers' bytes_used and free count their own units",
                 "3. borrow refuses a non-empty sequence, another pool's sequence, too many blocks (ValueError)",
                 "3. a write to a borrowed block raises RuntimeError before any poison; the holder bit-unchanged",
                 "4. served with the cache == without it / greedy alone (hits, a bypass, misses, a ring that wraps)",
                 "4. prefill tokens with the cache == without it - P x hits; hit / bypass counters; no preemption",
                 "4. the holder bit-unchanged; free == total - pinned once idle",
                 "5. preemption with the cache: hits borrow again, prefill counted after P, outputs as in 4",
                 "6. never-fit: a miss over num_blocks - pinned is refused at submit, a hit of the same size served",
                 "6. never-fit over HTTP: 400, not a hang; the hit 200",
                 "7. HTTP chat completion with the cache == without it",
                 "7. /metrics: engine_prefix_hits_total, engine_prefix_bypasses_total, engine_prefix_pinned_units"]:
        na(line)
else:
    # ------------------------------------------------------------ 2. model level: borrow + the rest == split at P
    pool, pre = holder(64, POISON)
    snap, eq, starts, shared, owned = pinned_kv(pre), [], [], [], []
    m = eng.model
    for text in CHATS:
        ids = chat_ids(text)
        a, b = pool.new_sequence(), pool.new_sequence()
        pre.attach(a)
        starts.append(a.length == P and a.kv.shared == N)
        for i in range(0, P, CHUNK):                            # the preamble as the holder took it
            m.forward_packed([(torch.tensor(ids[i:min(i + CHUNK, P)]), b)])
        la, lb = (m.forward(torch.tensor(ids[P:]), state=st) for st in (a, b))   # every row of the rest
        ok = torch.equal(la, lb)
        for _ in range(4):
            t = int(la[-1, :V].argmax())
            la, lb = (m.decode_batch([t], [st]) for st in (a, b))
            ok = ok and torch.equal(la, lb)
        eq.append(ok)
        shared.append(all(a.kv.tables[g][:N] == pre.state.kv.tables[g][:N] for g in range(G)))
        owned.append(a.kv.bytes_used() == (sum(map(len, a.kv.tables)) - N * G) * pool.unit_bytes)
        a.free(); b.free()
    check(f"{name}: {len(CHATS)} chats borrowing the {P}-token preamble and prefilling the rest: logits == a fresh "
          f"sequence of the same pool prefilled as ids[:P] then ids[P:] (torch.equal, every row), and over 4 decode "
          f"steps ({sum(eq)}/{len(eq)})", all(eq) and all(starts) and all(shared))
    check(f"{name}: the holder's K and V bit-unchanged afterwards; the borrowers' bytes_used count their own units; "
          f"freed, they leave free == total - pinned ({pool.allocator.num_free} == {pool.allocator.num_blocks} - "
          f"{PINNED})", intact(pre, snap) and all(owned)
          and pool.allocator.num_free == pool.allocator.num_blocks - PINNED)

    # ------------------------------------------------------------ 3. borrow refuses; the write guard (poisoned pool)
    pool_b, pre_b = holder(-(-PINNED // G) + 12, True)          # poisoned: a retagged ring slot is NaN-filled first
    snap_b, kv, refused = pinned_kv(pre_b), pool_b.kv, 0
    st = pool_b.new_sequence()
    m.forward(torch.tensor(chat_ids(CHATS[0])[:3]), state=st)
    for seq, n in ((st, N), (pool.new_sequence(), N), (pool_b.new_sequence(), N + 1)):   # non-empty, other pool, > holder
        try:
            seq.kv.borrow(pre_b.state.kv, n)
        except ValueError:
            refused += 1
        seq.free()
    check(f"{name}: borrow refuses a non-empty sequence, a sequence of another pool and {N + 1} blocks of a {N}-block "
          f"holder (ValueError, {refused}/3)", refused == 3)
    st = pool_b.new_sequence()
    pre_b.attach(st)
    gf, gs = kv.sliding.index(False), kv.sliding.index(True)
    x = torch.randn(1, *kv.k.shape[3:], device=kv.k.device)
    tags, raised = [list(t) for t in st.kv.tags], 0
    writes = [lambda: st.kv.unit(gf, 0, write=True),           # a full group's block 0
              lambda: st.kv.unit(gs, R, write=True),           # a sliding group's block R: its ring wraps onto slot 0
              lambda: st.kv.write(kv.groups[gs][0], P - 1, x, x)]   # a layer's K/V at the preamble's last position
    wrap = CAP - P <= 2 * CHUNK                                 # tiny: a forward that wraps the ring is cheap
    if wrap:
        st2 = pool_b.new_sequence()
        pre_b.attach(st2)
        fill = (chat_ids(CHATS[0]) * (CAP // 8))[:CAP + 8 - P]
        m.forward(torch.tensor(fill[:CAP - P]), state=st2)      # up to the cap: the ring's own slots
        writes.append(lambda: m.forward(torch.tensor(fill[CAP - P:]), state=st2))   # past it: onto the shared slots
    for w in writes:
        try:
            w()
        except RuntimeError:
            raised += 1
    ok = raised == len(writes) and intact(pre_b, snap_b) and st.kv.tags == tags
    st.free()
    if wrap:
        st2.free()
    check(f"{name}: a write to a borrowed block (a full group's block 0, a sliding group's block {R} onto ring slot 0, "
          f"K/V at position {P - 1}{', a forward past the cap' if wrap else ''}) raises RuntimeError before any poison: "
          f"the holder bit-unchanged ({raised}/{len(writes)}); all freed, free == total - pinned",
          ok and pool_b.allocator.num_free == pool_b.allocator.num_blocks - PINNED)
    del pool, pre, pool_b, pre_b, st, seq, a, b, la, lb, writes, kv, snap, snap_b
    if wrap:
        del st2
    T.tidy()

    # ------------------------------------------------------------ 4. served: cache on == off, P fewer prefill tokens
    LONG = min(LEN, eng.max_model_len) >= CAP + 2 * BS + 8       # INT8 serves 4,096 tokens, under its 4,624-token cap:
    if LONG:                                                    # no chat can bypass and no ring can wrap there
        LONG_CHAT, LONG_CHAT_TEXT = longest(chat_ids, CAP - 1)  # + max_tokens past the cap: a bypass (it wraps)
        BYPASS_N = CAP - len(LONG_CHAT) + 8
        LONG_RAW, _ = longest(T.raw_ids, CAP + 2 * BS)          # longer than the ring: wraps in its prefill
    REQS = [(chat_ids(c), M_HIT, "hit") for c in CHATS] + [(T.raw_ids(PROMPTS[0]), 12, "miss")] \
        + ([(LONG_CHAT, BYPASS_N, "bypass"), (LONG_RAW, 8, "miss")] if LONG else [])
    HITS = sum(k == "hit" for _, _, k in REQS)
    REFS = [reference(ids, n, P if k == "hit" else 0) for ids, n, k in REQS]
    PLAIN0 = reference(REQS[0][0], M_HIT)                       # the cache-off app's hit (7): no split at P
    T.tidy()
    full = [blocks_for(len(ids) + n) - (PINNED if k == "hit" else 0) for ids, n, k in REQS]
    # room for everything at once (no preemption); aya: one long request is ~15 x the rest (1,159 units), so the
    # second is admitted only once the first has left and one is budgeted
    units = PINNED + sum(full[:4]) + (max(full[4:], default=0) if T.gpu else sum(full[4:])) + G * 4
    KV4 = -(-units // G)
    def agree(reqs, refs, tag):                                 # every served greedy run ended normally, == its reference
        ok = [T.agree(ref, r.generated, r.finish_reason, what=f"{tag} {j}") for j, (ref, r) in enumerate(zip(refs, reqs))]
        return all(ok) and all(r.finish_reason in ("stop", "length") for r in reqs)
    on, c_on, held, idle, attaches, pre_on = serve(REQS, KV4, True)
    T.tidy()
    off, c_off, _, idle_off, _, pre_off = serve(REQS, KV4, False)
    T.tidy()
    lens = [len(ids) for ids, _, _ in REQS]
    what = (f"a {lens[4]}-token chat + {BYPASS_N} new > the {CAP}-token cap: a bypass; raw prompts of {lens[3]} and "
            f"{lens[5]} > {CAP} tokens: misses, rings wrapping" if LONG else f"a raw prompt of {lens[3]}: a miss; at "
            f"most {eng.max_model_len} tokens a request, so none can bypass the {CAP}-token cap")
    check(f"{name}: {len(REQS)} requests ({HITS} chats: hits; {what}) in a {KV4}-block pool: with the cache == "
          f"{'without it == ' if T.exact else ''}greedy alone (hits split at P)",
          agree(on, REFS, "request") and (not T.exact or [r.generated for r in on] == [r.generated for r in off])
          and (not LONG or lens[5] > CAP and lens[4] <= CAP < lens[4] + BYPASS_N))
    check(f"{name}: prefill tokens {c_on['prefill_tokens_total']} with the cache == {c_off['prefill_tokens_total']} "
          f"without - {P} x {HITS} hits; hits {c_on['prefix_hits_total']}, bypasses {c_on['prefix_bypasses_total']} "
          f"(without: 0, 0); attaches {attaches}; no preemption",
          c_on["prefill_tokens_total"] == c_off["prefill_tokens_total"] - P * HITS
          and c_off["prefill_tokens_total"] == sum(lens) and c_on["prefix_hits_total"] == attaches == HITS
          and c_on["prefix_bypasses_total"] == int(LONG) and c_off["prefix_hits_total"] == c_off["prefix_bypasses_total"] == 0
          and c_on["requests_preempted_total"] == c_off["requests_preempted_total"] == 0 and not pre_on + pre_off)
    check(f"{name}: the holder's K and V bit-unchanged after serving; once idle, free == total - pinned (without the "
          f"cache: == total)", held and idle and idle_off)

    # ------------------------------------------------------------ 5. preemption with the cache on
    hl = lens[:HITS]
    free5 = sum(blocks_for(L + 1) - PINNED for L in hl) + G * (HITS - 1)   # all admitted at once, with headroom
    KV5 = -(-(free5 + PINNED) // G)
    dry = sum(blocks_for(L + M_HIT - 1) - PINNED for L in hl)  # what they hold at the end: more than the pool's
    got, c5, held, idle, attaches, pre5 = serve(REQS[:HITS], KV5, True, max_batch=HITS)
    T.tidy()
    want = sum(L - P for L in hl) + sum(n - P for n in pre5)    # every (re)admission prefills after P
    check(f"{name}: {HITS} hits in a {KV5}-block pool ({KV5 * G - PINNED} units not pinned, {dry} needed at the end): "
          f"{len(pre5)} preemptions, outputs as in check 4; {attaches} attaches (= hits + preemptions), prefill tokens "
          f"{c5['prefill_tokens_total']} == {want} (each counted after P); holder bit-unchanged; all but the pinned free",
          len(pre5) >= 1 and c5["requests_preempted_total"] == len(pre5) and agree(got, REFS, "hit")
          and attaches == HITS + len(pre5) and c5["prefill_tokens_total"] == want and held and idle
          and c5["prefix_hits_total"] == HITS and c5["prefix_bypasses_total"] == 0)

    # ------------------------------------------------------------ 6. never-fit: own units vs num_blocks - pinned
    S = lens[0] + M_HIT
    need = blocks_for(S)
    KV6 = -(-need // G)
    miss = T.raw_ids(PROMPTS[0])
    s = Scheduler(eng, Metrics(), max_batch=2, kv_blocks=KV6, max_model_len=LEN, prefill_chunk=CHUNK, prefix_cache=True)
    total = s.pool.allocator.num_blocks
    try:                                                        # accepted, it would wait forever ahead of the hit
        s.cancel(s.submit(miss, GREEDY, S - len(miss))); refused = False
    except ValueError:
        refused = True
    hit = s.submit(REQS[0][0], GREEDY, M_HIT)
    ended = drain([hit])
    s.shutdown(5)
    check(f"{name}: never-fit: a miss of {S} positions needing {need} units (> the {total - PINNED} not pinned, <= all "
          f"{total}) is refused at submit (ValueError); a hit of the same size ({need - PINNED} own units) is accepted "
          f"and served as in check 4", total - PINNED < need <= total and refused and ended and agree([hit], REFS[:1], "hit"))
    del s, hit
    rs, _, _ = asyncio.run(http(True, KV6, [
        ("/v1/completions", {"prompt": PROMPTS[0], "max_tokens": S - len(miss), **GR}),
        ("/v1/chat/completions", {"messages": [{"role": "user", "content": CHATS[0]}], "max_tokens": M_HIT, **GR})],
        max_batch=2, limit=120))
    codes = [r if r == "timeout" else r.status_code for r in rs]
    check(f"{name}: never-fit over HTTP (create_app(prefix_cache=True)): the miss gets {codes[0]} (not a hang), the "
          f"hit {codes[1]}", codes == [400, 200] and "never fit" in rs[0].json()["detail"])
    del rs
    T.tidy()

    # ------------------------------------------------------------ 7. HTTP: cache on == off; /metrics
    posts = [("/v1/chat/completions", {"messages": [{"role": "user", "content": CHATS[0]}], "max_tokens": M_HIT, **GR})] \
        + ([("/v1/chat/completions", {"messages": [{"role": "user", "content": LONG_CHAT_TEXT}],
                                      "max_tokens": BYPASS_N, **GR})] if LONG else []) \
        + [("/v1/completions", {"prompt": PROMPTS[0], "max_tokens": 12, **GR})]
    which = [0, 4, 3] if LONG else [0, 3]                       # their REQS
    runs = []
    for cache in (True, False):
        rs, served, mx = asyncio.run(http(cache, KV4, posts))
        reqs = [served.last(REQS[i][0]) for i in which]
        runs.append(([r.status_code for r in rs], [reply(r) for r in rs], reqs, mx))
        del rs, served
        T.tidy()
    (codes, texts, reqs_on, m_on), (codes_off, texts_off, reqs_off, m_off) = runs
    ok = codes == codes_off == [200] * len(posts) and all(t == tok.decode(r.generated) for t, r in zip(texts, reqs_on))
    if T.exact:
        ok = ok and texts == texts_off and [r.generated for r in reqs_on] == [r.generated for r in reqs_off]
    check(f"{name}: HTTP with the cache (a hit, {'a bypass, ' if LONG else ''}a raw prompt) == without it{'' if T.exact else ' (near-ties)'}"
          f" == greedy alone", ok and agree(reqs_on, [REFS[i] for i in which], "on")
          and agree(reqs_off, [PLAIN0] + [REFS[i] for i in which[1:]], "off"))
    keys = ["engine_prefix_hits_total", "engine_prefix_bypasses_total", "engine_prefix_pinned_units",
            "engine_kv_blocks_free"]
    got_on, got_off = [m_on.get(k) for k in keys], [m_off.get(k) for k in keys]
    check(f"{name}: /metrics with the cache: hits, bypasses, pinned units, free {got_on} == [1, {int(LONG)}, {PINNED}, "
          f"{KV4 * G - PINNED}]; without: {got_off}", got_on == [1, int(LONG), PINNED, KV4 * G - PINNED]
          and got_off == [0, 0, 0, KV4 * G])
    del runs, reqs_on, reqs_off, on, off, got, probe, blocks_for, m
    T.tidy()

# ---------------------------------------------------------------- 8. GPU targets: memory and aborts; free the model
gpu_gates(T, check)                                             # while the target is loaded (nothing on a CPU target)
freed = "the GPU target's model is freed and the MPS cache emptied" if T.gpu else "the target's model is freed"
if POISON:
    del eng.model.new_paged_pool
del eng, T
check(freed, unload_target(TARGET))

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
