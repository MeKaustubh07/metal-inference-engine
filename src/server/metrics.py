"""Minimal Prometheus metrics (text exposition format), hand-written: counters, gauges, histograms."""
import threading

LATENCY_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0)


class Histogram:
    def __init__(self, name: str, help_: str, lock: threading.Lock, buckets=LATENCY_BUCKETS):
        self.name, self.help, self.buckets, self.lock = name, help_, buckets, lock
        self.counts = [0] * (len(buckets) + 1)
        self.sum = 0.0
        self.n = 0

    def observe(self, v: float) -> None:
        with self.lock:                                   # the engine thread observes while the event loop renders
            self.sum += v; self.n += 1
            i = next((i for i, b in enumerate(self.buckets) if v <= b), len(self.buckets))
            self.counts[i] += 1

    def render(self) -> list[str]:
        out = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} histogram"]
        cum = 0
        for b, c in zip(self.buckets, self.counts):
            cum += c
            out.append(f'{self.name}_bucket{{le="{b}"}} {cum}')
        out.append(f'{self.name}_bucket{{le="+Inf"}} {self.n}')
        out += [f"{self.name}_sum {self.sum:.6f}", f"{self.name}_count {self.n}"]
        return out

    def quantile(self, q: float) -> float:
        """Upper bucket bound containing quantile q (for logs/tests; Prometheus computes it server-side)."""
        if not self.n:
            return 0.0
        target, cum = q * self.n, 0
        for b, c in zip(self.buckets, self.counts):
            cum += c
            if cum >= target:
                return b
        return float("inf")


class Metrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.counters = {"requests_total": 0, "requests_rejected_total": 0, "requests_preempted_total": 0,
                         "requests_finished_total": 0, "prompt_tokens_total": 0, "generation_tokens_total": 0,
                         "decode_steps_total": 0, "decode_sequences_total": 0,   # ratio = mean decode batch size
                         "prefill_steps_total": 0, "prefill_tokens_total": 0,   # ratio = mean packed prefill size
                         "decisions_total": 0, "jobs_rejected_total": 0}          # /v1/decide answered / turned away
        self.gauges = {"running_requests": 0, "prefilling_requests": 0, "waiting_requests": 0, "kv_blocks_free": 0,
                       "kv_blocks_total": 0, "kv_unit_bytes": 0, "kv_bytes_held": 0, "weights_locked_bytes": 0,
                       "waiting_jobs": 0}
        self.ttft = Histogram("engine_time_to_first_token_seconds", "Time from arrival to the first generated token",
                              self.lock)
        self.tpot = Histogram("engine_time_per_output_token_seconds", "Time between consecutive generated tokens",
                              self.lock, (0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.2, 0.5, 1.0))
        self.e2e = Histogram("engine_request_latency_seconds", "Arrival to last token, completed requests only",
                             self.lock)

    def inc(self, name: str, v: int = 1) -> None:
        with self.lock:
            self.counters[name] += v

    def set(self, name: str, v: float) -> None:
        with self.lock:
            self.gauges[name] = v

    def render(self) -> str:
        with self.lock:
            lines = []
            for k, v in self.counters.items():
                lines += [f"# TYPE engine_{k} counter", f"engine_{k} {v}"]
            for k, v in self.gauges.items():
                lines += [f"# TYPE engine_{k} gauge", f"engine_{k} {v}"]
            for h in (self.ttft, self.tpot, self.e2e):
                lines += h.render()
            return "\n".join(lines) + "\n"
