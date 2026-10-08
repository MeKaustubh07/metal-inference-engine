"""Run every test script in sequence (each needs the model weights) and summarize. Exit 1 if any fails.

usage: run_tests.py [--quick] [--save DIR]
  --quick     skips the cache, quantization, Qwen3.5, Tiny Aya model / long / quant, serving, prefill and decision
              suites on their real models; the serving, prefill and decision suites still run on the tiny target
              (tests/serving_targets.py: a small random Cohere2 on the CPU)
An entry may carry arguments ("test_server --target tiny"); its report and saved output are named test_server@tiny.
  --save DIR  writes each suite's output to DIR/<suite>.txt and this summary, with timings, to DIR/run_tests.txt
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTS = ["test_tokenizer", "test_aya", "test_cohere2", "test_window", "test_sampling", "test_cache", "test_paged",
         "test_kernels", "test_native", "test_quant", "test_qwen35", "test_qwen35_2b", "test_aya_model",
         "test_aya_long", "test_aya_long_quant", "test_aya_quant", "test_server", "test_prefill", "test_decision",
         "test_server --target tiny", "test_prefill --target tiny", "test_decision --target tiny"]
SLOW = {"test_cache", "test_quant", "test_qwen35", "test_qwen35_2b", "test_aya_model", "test_aya_long",
        "test_aya_long_quant", "test_aya_quant", "test_server", "test_prefill", "test_decision"}


def main() -> None:
    quick = "--quick" in sys.argv
    save = Path(sys.argv[sys.argv.index("--save") + 1]) if "--save" in sys.argv else None
    failed, summary, skipped = [], [], 0
    def say(line: str) -> None:
        print(line); summary.append(line)
    start = time.perf_counter()
    for entry in TESTS:
        if quick and entry in SLOW:
            continue
        script, *args = entry.split()
        name = script + "".join(f"@{a}" for a in args if not a.startswith("--"))   # "test_server --target tiny"
        t0 = time.perf_counter()
        r = subprocess.run([sys.executable, f"tests/{script}.py", *args], cwd=ROOT, capture_output=True, text=True)
        passes, fails, skips = r.stdout.count("PASS"), r.stdout.count("FAIL"), r.stdout.count("SKIP")
        nas = r.stdout.count("N/A")                                    # checks that do not apply to a target
        skipped += skips
        status = "ok  " if r.returncode == 0 else "FAIL"
        say(f"{status} {name:19s} {passes:3d} pass {fails:2d} fail  {time.perf_counter() - t0:6.1f}s"
            + (f"  ({nas} n/a)" if nas else "")
            + (f"  ({skips} skipped: needs files not in the repo)" if skips else ""))
        if save:
            (save / f"{name}.txt").write_text(r.stdout)
        if r.returncode != 0:
            failed.append(name)
            say("\n".join(l for l in r.stdout.splitlines() if "FAIL" in l)[:2000] + "\n" + r.stderr[-1500:])
    verdict = "ALL PASSED" if not failed else "FAILED: " + ", ".join(failed)
    say(f"\n{verdict}{f', {skipped} skipped' if skipped else ''}  ({time.perf_counter() - start:.0f} s in all)")
    if save:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        head = f"run_tests.py {' '.join(sys.argv[1:])} at {time.strftime('%Y-%m-%d %H:%M')} on commit {commit}"
        (save / "run_tests.txt").write_text(head + " (+ uncommitted changes, if any)\n\n" + "\n".join(summary) + "\n")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
