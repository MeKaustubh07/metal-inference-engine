"""OpenAI-compatible HTTP API over the engine: completions and chat completions, streaming via SSE.

Endpoints: POST /v1/completions, POST /v1/chat/completions, POST /v1/decide (order-invariant choice / boolean /
score decisions, src/decision.py), GET /health (process alive), GET /ready (model loaded and queue has room: load
balancers should route here), GET /metrics (Prometheus text format).
"""
import asyncio
import contextlib
import json
import logging
import time
import uuid

from fastapi import FastAPI, HTTPException, Request as HTTPRequest
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

import decision
from chat import format_chat
from sampler import SamplingParams
from server.metrics import Metrics
from server.scheduler import QueueFull, Scheduler

log = logging.getLogger("engine.server")
MAX_PROMPT_CHARS = 65536                  # bounds tokenization work per request before the token-length check
MAX_OPTIONS, MAX_OPTION_CHARS, MAX_OPTION_TOKENS = 16, 1000, 64     # /v1/decide: bounds the forked states' memory
DISCONNECT_POLL_S = 1.0                   # how often a waiting handler checks whether its client is still there


class Sampling(BaseModel):
    max_tokens: int | None = Field(None, ge=1, le=4096)
    max_completion_tokens: int | None = Field(None, ge=1, le=4096)   # the newer OpenAI name for max_tokens
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.8, gt=0.0, le=1.0)
    top_k: int = Field(20, ge=0)
    repetition_penalty: float = Field(1.0, ge=1.0, le=2.0)
    seed: int | None = None
    stop: str | list[str] | None = None
    n: int = Field(1, ge=1, le=1)          # one choice per request
    stream: bool = False

    def limit(self, default: int) -> int:
        return self.max_completion_tokens or self.max_tokens or default

    def stops(self) -> list[str]:
        stops = [self.stop] if isinstance(self.stop, str) else (self.stop or [])
        return [s for s in stops if s][:4]


class CompletionBody(Sampling):
    prompt: str = Field(max_length=MAX_PROMPT_CHARS)


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(system|user|assistant)$")
    content: str = Field(max_length=MAX_PROMPT_CHARS)


class DecideBody(BaseModel):
    type: str = Field(pattern="^(choice|boolean|score)$")
    question: str = Field(min_length=1, max_length=MAX_PROMPT_CHARS)
    context: str = Field("", max_length=MAX_PROMPT_CHARS)
    options: list[str] | None = Field(None, min_length=2, max_length=MAX_OPTIONS)
    chat: bool = True                      # the context as a chat user turn; options are the assistant's answer
    length_normalize: bool = True          # compare options by log-probability per token (they differ in length)
    embeddings: bool = False               # also return each option's final hidden state


class ChatBody(Sampling):
    messages: list[ChatMessage] = Field(min_length=1, max_length=256)
    enable_thinking: bool = False


class AsyncSink:
    """Bridges the engine thread to the event loop: put() is called from the engine thread and wakes the awaiting
    handler without tying up a worker thread per request (a blocking queue.get in a thread pool would)."""

    def __init__(self):
        self.loop = asyncio.get_running_loop()
        self.q: asyncio.Queue = asyncio.Queue()

    def put(self, item) -> None:
        self.loop.call_soon_threadsafe(self.q.put_nowait, item)


class OutputFilter:
    """Turns the raw generated text into ("reasoning" | "content", text) pieces.

    thinking: the prompt already opened <think>, so text before </think> is reasoning (Qwen3.5 thinking mode).
    stops: the content is cut at the first stop string; a tail that could still grow into one is held back.
    """

    def __init__(self, thinking: bool, stops: list[str]):
        self.in_reasoning, self.stops, self.buf = thinking, stops, ""
        self.keep = max((len(s) for s in stops), default=1) - 1
        self.stopped = False
        self.lead = False                                     # drop the newlines between </think> and the answer

    def feed(self, piece: str) -> list[tuple[str, str]]:
        out = []
        if self.in_reasoning:
            reasoning, sep, piece = piece.partition("</think>")   # a single special token: it arrives whole
            if reasoning:
                out.append(("reasoning", reasoning))
            if not sep:
                return out
            self.in_reasoning, self.lead = False, True
        if self.lead:                                         # they may arrive in later pieces than </think>
            piece = piece.lstrip("\n")
            self.lead = not piece
        self.buf += piece
        hits = [i for i in (self.buf.find(s) for s in self.stops) if i >= 0]
        if hits:
            text, self.buf, self.stopped = self.buf[: min(hits)], "", True
        else:
            cut = max(0, len(self.buf) - self.keep)
            text, self.buf = self.buf[:cut], self.buf[cut:]
        if text:
            out.append(("content", text))
        return out

    def flush(self) -> list[tuple[str, str]]:
        text, self.buf = self.buf, ""
        return [("content", text)] if text else []


def create_app(engine, max_batch: int = 8, max_waiting: int = 64, kv_blocks: int = 1024,
               max_model_len: int | None = None, drain_timeout: float = 2.0, prefill_chunk: int = 512,
               batch_wait_ms: float = 5.0, lock_weights: bool = False) -> FastAPI:
    cap = getattr(engine, "max_model_len", None)
    if max_model_len is None:                           # default: the model's own limit (else 4096)
        max_model_len = cap or 4096
    elif cap and max_model_len > cap:                   # a model's own limit wins (registry and model)
        log.info(f"max_model_len {max_model_len} -> {cap}, {engine.name}'s limit")
        max_model_len = cap
    metrics = Metrics()
    sched = Scheduler(engine, metrics, max_batch=max_batch, max_waiting=max_waiting, kv_blocks=kv_blocks,
                      max_model_len=max_model_len, prefill_chunk=prefill_chunk, batch_wait_ms=batch_wait_ms,
                      lock_weights=lock_weights)
    if lock_weights:
        if sched.lock_error:
            log.warning(f"could not lock the weights in memory ({sched.lock_error}); an idle server may be paged out")
        else:
            log.info(f"weights locked in memory: {sched.weights_locked / 2**30:.2f} GiB")

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        yield
        # uvicorn runs this after its own graceful wait (open HTTP connections finish first), so whatever is still
        # in the scheduler has no client left: stop quickly instead of serving it
        await asyncio.to_thread(sched.shutdown, drain_timeout)

    app = FastAPI(title="InferenceEngine", version="1.0", lifespan=lifespan)
    app.state.scheduler, app.state.metrics, app.state.engine = sched, metrics, engine

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request, exc):
        # the default handler echoes the offending input; ensure_ascii escapes lone surrogates ("\ud800" in the
        # request JSON), which would otherwise make rendering the 422 itself fail with a 500
        body = json.dumps({"detail": jsonable_encoder(exc.errors())}, ensure_ascii=True, default=str)
        return Response(body, status_code=422, media_type="application/json")

    async def submit(prompt: str, body: Sampling, default_max: int, add_bos: bool):
        ids = await asyncio.to_thread(engine.tokenizer.encode, prompt, add_bos)   # CPU work off the event loop
        params = SamplingParams(temperature=body.temperature, top_k=body.top_k, top_p=body.top_p,
                                repetition_penalty=body.repetition_penalty, seed=body.seed)
        try:
            return sched.submit(ids, params, body.limit(default_max), out=AsyncSink()), len(ids)
        except QueueFull:
            raise HTTPException(429, "server is at capacity; retry with backoff", headers={"Retry-After": "1"})
        except ValueError as e:
            raise HTTPException(400, str(e))

    async def events(req, http: HTTPRequest, thinking: bool, stops: list[str]):
        """Yield ("reasoning" | "content", text) pieces, then ("finish", reason). Cancels the request if the client
        disconnects, the handler is cancelled, or a stop string is hit."""
        filt = OutputFilter(thinking, stops)
        loop = asyncio.get_running_loop()
        next_check = loop.time() + DISCONNECT_POLL_S
        try:
            while True:
                try:
                    kind, payload = await asyncio.wait_for(req.out.q.get(), DISCONNECT_POLL_S)
                except asyncio.TimeoutError:
                    kind = None
                if loop.time() >= next_check:                 # on a timer, not only when idle: tokens keep coming
                    if await http.is_disconnected():
                        return
                    next_check = loop.time() + DISCONNECT_POLL_S
                if kind is None:
                    continue
                if kind == "error":
                    raise RuntimeError(payload)
                if kind == "done":
                    for item in filt.flush():
                        yield item
                    yield "finish", payload
                    return
                for item in filt.feed(payload):
                    yield item
                if filt.stopped:
                    sched.cancel(req)
                    yield "finish", "stop"
                    return
        finally:
            if req.finish_reason is None:
                sched.cancel(req)                             # client gone or handler cancelled

    def respond(req, n_prompt: int, chat: bool, body: Sampling, http: HTTPRequest, thinking: bool = False):
        rid = f"{'chatcmpl' if chat else 'cmpl'}-{uuid.uuid4().hex[:12]}"
        created, obj = int(time.time()), ("chat.completion" if chat else "text_completion")
        stream_events = events(req, http, thinking, body.stops())

        def chunk(delta: dict | None = None, text: str = "", finish: str | None = None) -> str:
            choice = {"index": 0, "finish_reason": finish}
            if chat:
                choice["delta"] = delta or {}
            else:
                choice["text"] = text
            return "data: " + json.dumps({"id": rid, "object": obj + (".chunk" if chat else ""), "created": created,
                                          "model": engine.name, "choices": [choice]}) + "\n\n"

        if body.stream:
            async def gen():
                final = None
                try:
                    if chat:
                        yield chunk({"role": "assistant", "content": ""})
                    async for kind, text in stream_events:
                        if kind == "finish":
                            final = text
                            yield chunk(finish=text)
                        elif chat:
                            yield chunk({"reasoning_content" if kind == "reasoning" else "content": text})
                        elif kind == "content":
                            yield chunk(text=text)
                    yield "data: [DONE]\n\n"
                except RuntimeError as e:
                    yield "data: " + json.dumps({"error": {"message": str(e)}}) + "\n\n"
                finally:
                    _log(req, n_prompt, final)
            return StreamingResponse(gen(), media_type="text/event-stream")

        async def collect():
            parts, reasoning, reason = [], [], None
            try:
                async for kind, text in stream_events:
                    if kind == "finish":
                        reason = text
                    else:
                        (reasoning if kind == "reasoning" else parts).append(text)
            except RuntimeError as e:
                raise HTTPException(500, str(e))
            finally:
                _log(req, n_prompt, reason)
            if reason is None:                                # client left: nobody to answer
                raise HTTPException(499, "client closed request")
            text = "".join(parts)
            if chat:
                msg = {"role": "assistant", "content": text}
                if thinking:
                    msg["reasoning_content"] = "".join(reasoning).strip()
                choice = {"message": msg}
            else:
                choice = {"text": text}
            return JSONResponse({"id": rid, "object": obj, "created": created, "model": engine.name,
                                 "choices": [{"index": 0, "finish_reason": reason, **choice}],
                                 "usage": {"prompt_tokens": n_prompt, "completion_tokens": len(req.generated),
                                           "total_tokens": n_prompt + len(req.generated)}})
        return collect()

    def _log(req, n_prompt: int, told: str | None = None) -> None:
        """told: the finish_reason the client was sent, if any. It comes first: a stop string ends the request
        through cancel(), but it finished normally. A request cancelled while running (client gone) is stopped by
        the engine at its next step, after this line runs, so its own finish_reason may still be unset."""
        ttft = (req.first_token_at - req.arrived) if req.first_token_at else None
        reason = told or req.finish_reason or ("cancelled" if req.cancelled else None)
        log.info(json.dumps({"event": "request_finished", "id": req.id, "prompt_tokens": n_prompt,
                             "completion_tokens": len(req.generated), "finish_reason": reason,
                             "ttft_s": round(ttft, 4) if ttft else None,
                             "latency_s": round(time.perf_counter() - req.arrived, 4)}))

    @app.post("/v1/completions")
    async def completions(body: CompletionBody, http: HTTPRequest):
        req, n = await submit(body.prompt, body, 128, add_bos=True)      # raw text: BOS first, if the model has one
        out = respond(req, n, False, body, http)
        return out if body.stream else await out

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatBody, http: HTTPRequest):
        roles = [m.role for m in body.messages]
        if "system" in roles[1:]:
            raise HTTPException(400, "a system message may only be the first message")
        if "user" not in roles:
            raise HTTPException(400, "at least one user message is required")
        if sum(len(m.content) for m in body.messages) > MAX_PROMPT_CHARS:
            raise HTTPException(400, f"messages exceed {MAX_PROMPT_CHARS} characters in total")
        if body.enable_thinking and not getattr(engine, "thinking", False):
            raise HTTPException(400, f"{engine.name} has no thinking mode")
        try:                                              # a model's own template may refuse the conversation
            prompt = format_chat([m.model_dump() for m in body.messages], style=engine.chat_style,
                                 enable_thinking=body.enable_thinking)
        except ValueError as e:                           # e.g. roles that do not alternate (Tiny Aya)
            raise HTTPException(400, str(e))
        req, n = await submit(prompt, body, 256, add_bos=False)          # the chat template writes BOS itself
        out = respond(req, n, True, body, http, body.enable_thinking)
        return out if body.stream else await out

    @app.post("/v1/decide")
    async def decide(body: DecideBody, http: HTTPRequest):
        if len(body.context) + len(body.question) > MAX_PROMPT_CHARS:
            raise HTTPException(400, f"context + question exceed {MAX_PROMPT_CHARS} characters")
        if body.options and any(len(o) > MAX_OPTION_CHARS for o in body.options):
            raise HTTPException(400, f"an option exceeds {MAX_OPTION_CHARS} characters")
        try:                                              # tokenizing is CPU work: off the event loop and the engine
            options, ctx, opt_ids = await asyncio.to_thread(decision.prepare, engine, body.type, body.context,
                                                            body.question, body.options, body.chat)
        except ValueError as e:
            raise HTTPException(400, str(e))
        if max(map(len, opt_ids)) > MAX_OPTION_TOKENS:
            raise HTTPException(400, f"an option exceeds {MAX_OPTION_TOKENS} tokens")
        if len(ctx) + max(map(len, opt_ids)) > max_model_len:
            raise HTTPException(400, f"prompt ({len(ctx)} tokens) + longest option exceeds {max_model_len}")
        try:                                              # stepwise on the engine thread: decode keeps going
            fut = sched.run_job(lambda: decision.score_steps(engine, body.type, options, ctx, opt_ids,
                                                             body.length_normalize, body.embeddings,
                                                             sched.prefill_chunk))
        except QueueFull:
            raise HTTPException(429, "server is at capacity; retry with backoff", headers={"Retry-After": "1"})
        waiter = asyncio.wrap_future(fut)
        try:
            while True:
                try:
                    result = await asyncio.wait_for(asyncio.shield(waiter), DISCONNECT_POLL_S)
                    break
                except asyncio.TimeoutError:
                    if await http.is_disconnected():      # nobody to answer: the engine drops the job
                        raise HTTPException(499, "client closed request")
        finally:
            if not fut.done():
                fut.cancel()
        metrics.inc("decisions_total")
        return {"id": f"dec-{uuid.uuid4().hex[:12]}", "object": "decision", "model": engine.name, **result}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        if not sched.ready():
            raise HTTPException(503, "not ready")
        return {"status": "ready", "model": engine.name, "backend": engine.model.b.name}

    @app.get("/metrics")
    def prometheus():
        sched.refresh_gauges()                            # queue gauges as of now, not as of the last engine step
        return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")

    return app
