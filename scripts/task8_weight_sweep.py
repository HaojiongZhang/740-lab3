"""Task 8.4/8.5: predict and sweep matmul's Prime-core work weight.

This script reads the matched-clock IPC result from task8_cluster_compare.py.
It predicts the per-Prime-thread work ratio as

    (Prime IPC / Gold IPC) * (Prime clock / Gold clock),

then sweeps ``prime_weight`` around that prediction using eight pinned workers.
Run it with both clusters locked at the performance clocks used for Part B.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from task8_common import (Case, Profiler, add_measurement_args, load_csv,
                          require_cluster_locks, run_cases, write_csv,
                          write_summary)


def parse_weights(text: str) -> tuple[float, ...]:
    try:
        weights = tuple(float(value) for value in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--weights must be comma-separated numbers") from exc
    if not weights or any(weight <= 0 for weight in weights):
        raise argparse.ArgumentTypeError("--weights must contain positive values")
    return weights


def predicted_weight(pairs_path: Path, gold_khz: int, prime_khz: int) -> tuple[float, dict[str, str]]:
    rows = load_csv(pairs_path)
    match = next((row for row in rows if row.get("workload") == "matmul"), None)
    if match is None:
        raise SystemExit(f"{pairs_path} has no matmul matched-clock row.")
    gold_ipc, prime_ipc = float(match["gold_ipc"]), float(match["prime_ipc"])
    prediction = (prime_ipc / gold_ipc) * (prime_khz / gold_khz)
    return prediction, match


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_measurement_args(ap, "results/task8_weight_sweep.csv", default_pause=20.0)
    ap.add_argument("--pairs", type=Path, default=Path("results/task8_cluster_compare_pairs.csv"),
                    help="matched-clock Task 8.1/8.2 pair summary")
    ap.add_argument("--weights", type=parse_weights,
                    default=parse_weights("0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4"),
                    help="comma-separated Prime work weights to sweep")
    ap.add_argument("--allow-dvfs", action="store_true",
                    help="permit unlocked exploratory runs; not suitable for report comparisons")
    args = ap.parse_args()

    profiler = Profiler()
    state = require_cluster_locks(profiler.device, allow_dvfs=args.allow_dvfs)
    gold_khz, prime_khz = state["gold_min_khz"], state["prime_min_khz"]
    if gold_khz is None or prime_khz is None:
        raise SystemExit("could not read the locked Gold/Prime frequencies")
    predicted, pair = predicted_weight(args.pairs, gold_khz, prime_khz)
    print(f"Task 8.4 prediction from matched-clock matmul IPC: {predicted:.3f} "
          f"(IPC ratio {float(pair['prime_ipc']) / float(pair['gold_ipc']):.3f}, "
          f"clock ratio {prime_khz / gold_khz:.3f})")

    cases = tuple(
        Case(label=f"matmul_weight_{weight:.2f}", workload="matmul", core="all", threads=8,
             cluster="mixed", params={"iterations": 30, "pin": True, "prime_weight": weight},
             metadata={"prime_weight": weight, "gold_khz": gold_khz, "prime_khz": prime_khz,
                       "predicted_prime_weight": predicted})
        for weight in args.weights
    )
    rows, log = run_cases(profiler=profiler, args=args, cases=cases,
                          phase="task8_4_5", counters=None)
    try:
        summary_path, summary = write_summary(log, rows, ("case", "prime_weight"))
        even = next((row for row in summary if abs(float(row["prime_weight"]) - 1.0) < 1e-9), None)
        if even is None:
            raise RuntimeError("the sweep must include W=1.0 to report speedup over an even split")
        even_time = float(even["mean_time_us_per_iter"])
        for row in summary:
            row["speedup_vs_even_split"] = even_time / float(row["mean_time_us_per_iter"])
            row["distance_from_predicted_weight"] = abs(float(row["prime_weight"]) - predicted)
        best = min(summary, key=lambda row: float(row["mean_time_us_per_iter"]))
        prediction_path = log.path.with_name(log.path.stem + "_prediction.csv")
        write_csv(prediction_path, [{
            "matched_clock_khz": pair["matched_clock_khz"],
            "matched_clock_gold_ipc": pair["gold_ipc"],
            "matched_clock_prime_ipc": pair["prime_ipc"],
            "part_b_gold_khz": gold_khz,
            "part_b_prime_khz": prime_khz,
            "predicted_prime_weight": predicted,
            "best_measured_prime_weight": best["prime_weight"],
            "best_time_us_per_iter": best["mean_time_us_per_iter"],
            "best_speedup_vs_even": best["speedup_vs_even_split"],
        }])
        write_csv(summary_path, summary)
        print(f"\nbest W={float(best['prime_weight']):.2f}: "
              f"{float(best['speedup_vs_even_split']):.3f}x versus even split")
        print(f"wrote {log.path}, {log.jsonl_path}, {summary_path}, and {prediction_path}")
    finally:
        log.close()


if __name__ == "__main__":
    main()
