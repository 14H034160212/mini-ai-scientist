"""Run the mini AI Scientist over every rule x seed for one model.

  python scripts/run_eval.py --model Qwen/Qwen2.5-7B-Instruct --seeds 3
  python scripts/run_eval.py --model Qwen/Qwen2.5-7B-Instruct --no-check   # ablation
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mini_scientist import Lab, RULES, Episode, run_batch  # noqa: E402
from mini_scientist.backend import HFBackend, VLLMBackend  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--rules", default="", help="comma-separated rule names (default: all)")
    ap.add_argument("--max-turns", type=int, default=15)
    ap.add_argument("--budget", type=int, default=30)
    ap.add_argument("--no-check", action="store_true", help="ablation: remove the hypothesis-check tool")
    ap.add_argument("--no-compute", action="store_true", help="ablation: remove the Python compute tool")
    ap.add_argument("--gate", action="store_true", help="refuse submissions that contradict the observed data")
    ap.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    ap.add_argument("--think", action="store_true", help="enable thinking mode (Qwen3 / Qwen3.5)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    rules = [r for r in RULES if not args.rules or r.name in args.rules.split(",")]
    tools = tuple(t for t in ("experiment", "compute", "check", "submit")
                  if not (t == "check" and args.no_check) and not (t == "compute" and args.no_compute))
    episodes = [Episode(Lab(r, budget=args.budget, seed=s), max_turns=args.max_turns, tools=tools, gate=args.gate)
                for r in rules for s in range(args.seeds)]

    t0 = time.time()
    backend = VLLMBackend(args.model, thinking=args.think) if args.backend == "vllm" \
        else HFBackend(args.model, thinking=args.think)
    results = run_batch(backend, episodes)
    elapsed = time.time() - t0

    tag = args.model.split("/")[-1] + ("_nocheck" if args.no_check else "") + ("_nocompute" if args.no_compute else "") \
        + ("_think" if args.think else "") + ("_gate" if args.gate else "")
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    with open(out / f"{tag}.jsonl", "w") as f:
        for e, r in zip(episodes, results):
            f.write(json.dumps({**r, "model": args.model, "tools": list(tools), "trace": e.trace}) + "\n")

    n = len(results)
    print(f"{tag}: solved {sum(r['correct'] for r in results)}/{n} in {elapsed:.0f}s")
    for tier in (1, 2, 3):
        rs = [r for r in results if r["tier"] == tier]
        if rs:
            print(f"  tier {tier}: {sum(r['correct'] for r in rs)}/{len(rs)}  "
                  f"mean acc {sum(r['accuracy'] for r in rs) / len(rs):.2f}  "
                  f"mean experiments {sum(r['experiments'] for r in rs) / len(rs):.1f}")


if __name__ == "__main__":
    main()
