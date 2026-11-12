"""Task 8.1/8.2: compare Gold C0 and Prime C7 at one matched, locked clock.

Before running, lock both policies at the printed common OPP.  The default is
the highest common OPP read from the board, not a hard-coded board revision.

    python scripts/task8_cluster_compare.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from task8_common import (Case, Profiler, add_measurement_args,
                          read_opps, require_common_clock_lock, run_cases, write_csv,
                          write_summary)


WORKLOADS = ("saxpy", "reduction", "convolution", "matmul")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_measurement_args(ap, "results/task8_cluster_compare.csv", default_pause=15.0)
    ap.add_argument("--clock-khz", type=int, default=None,
                    help="common Gold/Prime OPP; default: highest common OPP read from the board")
    ap.add_argument("--show-common-opps", action="store_true",
                    help="print common Gold/Prime OPPs and exit without measuring")
    args = ap.parse_args()

    profiler = Profiler()
    if args.show_common_opps:
        common = sorted(set(read_opps(profiler.device, "gold"))
                        & set(read_opps(profiler.device, "prime")))
        if not common:
            raise SystemExit("No common Gold/Prime OPPs were readable from cpufreq.")
        print("common Gold/Prime OPPs (kHz): " + " ".join(map(str, common)))
        print(f"recommended matched clock: {common[-1]} kHz ({common[-1] / 1000:.1f} MHz)")
        return
    common_khz = require_common_clock_lock(profiler.device, args.clock_khz)
    print(f"matched-clock comparison: both clusters locked at {common_khz / 1000:.1f} MHz")

    cases = tuple(
        Case(label=f"{workload}_{cluster}_c{core}", workload=workload,
             core=core, threads=1, cluster=cluster,
             metadata={"purpose": "Task 8.1 matched-clock single-core comparison"})
        for workload in WORKLOADS
        for cluster, core in (("gold", 0), ("prime", 7))
    )
    rows, log = run_cases(profiler=profiler, args=args, cases=cases,
                          phase="task8_1_2", counters="stalls",
                          common_clock_khz=common_khz)
    try:
        summary_path, summary = write_summary(log, rows, ("workload", "cluster", "core"))
        by_key = {(row["workload"], row["cluster"]): row for row in summary}
        pairs = []
        print("\nMatched-clock Gold vs Prime summary:")
        for workload in WORKLOADS:
            gold, prime = by_key[(workload, "gold")], by_key[(workload, "prime")]
            gold_time = float(gold["mean_time_us_per_iter"])
            prime_time = float(prime["mean_time_us_per_iter"])
            gold_ipc = float(gold["mean_derived_ipc"])
            prime_ipc = float(prime["mean_derived_ipc"])
            gold_stall = float(gold["mean_derived_backend_stall_pct"])
            prime_stall = float(prime["mean_derived_backend_stall_pct"])
            row = {
                "workload": workload,
                "matched_clock_khz": common_khz,
                "gold_time_us_per_iter": gold_time,
                "prime_time_us_per_iter": prime_time,
                "prime_to_gold_time_ratio": prime_time / gold_time,
                "gold_ipc": gold_ipc,
                "prime_ipc": prime_ipc,
                "prime_to_gold_ipc_ratio": prime_ipc / gold_ipc,
                "gold_backend_stall_pct": gold_stall,
                "prime_backend_stall_pct": prime_stall,
                "prime_higher_ipc_and_backend_stall": prime_ipc > gold_ipc and prime_stall > gold_stall,
                "gold_cov_pct": gold["sample_cov_pct"],
                "prime_cov_pct": prime["sample_cov_pct"],
            }
            pairs.append(row)
            marker = "  <-- Task 8.2 candidate" if row["prime_higher_ipc_and_backend_stall"] else ""
            print(f"  {workload:12s}  time P/G={row['prime_to_gold_time_ratio']:.3f}  "
                  f"IPC G/P={gold_ipc:.3f}/{prime_ipc:.3f}  "
                  f"backend-stall G/P={gold_stall:.2f}/{prime_stall:.2f}%{marker}")
        pairs_path = log.path.with_name(log.path.stem + "_pairs.csv")
        write_csv(pairs_path, pairs)
        print(f"\nwrote {log.path}, {log.jsonl_path}, {summary_path}, and {pairs_path}")
    finally:
        log.close()


if __name__ == "__main__":
    main()
