"""Task 8.3: find the fastest practical Gold/Prime core mask for each workload.

The default set covers the two complete clusters, all eight cores, and every
seven-core all-but-one mask.  Thus it directly answers whether all cores win
and, if not, which one is best left idle.  It is intentionally not an
uninterpretable exhaustive search of all 255 masks.

Lock both clusters at the intended performance clocks first, normally:

    python scripts/power_pin.py gold 3628800
    python scripts/power_pin.py prime 4396800
    python scripts/task8_mask_sweep.py
"""

from __future__ import annotations

import argparse

from task8_common import (GOLD_CORES, PRIME_CORES, Case, Profiler,
                          add_measurement_args, core_list, mask_hex,
                          require_cluster_locks, run_cases, write_csv,
                          write_summary)


WORKLOADS = ("saxpy", "reduction", "convolution", "matmul")


def mask_cases() -> tuple[tuple[str, tuple[int, ...]], ...]:
    all_cores = (*GOLD_CORES, *PRIME_CORES)
    return (
        ("gold6", GOLD_CORES),
        ("prime2", PRIME_CORES),
        ("all8", all_cores),
        *((f"all_minus_c{core}", tuple(c for c in all_cores if c != core))
          for core in all_cores),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_measurement_args(ap, "results/task8_mask_sweep.csv", default_pause=20.0)
    ap.add_argument("--allow-dvfs", action="store_true",
                    help="permit unlocked exploratory runs; not suitable for report comparisons")
    args = ap.parse_args()

    profiler = Profiler()
    state = require_cluster_locks(profiler.device, allow_dvfs=args.allow_dvfs)
    print("mask sweep clock locks: "
          f"Gold={state['gold_min_khz']} kHz, Prime={state['prime_min_khz']} kHz")

    cases = tuple(
        Case(label=f"{workload}_{label}", workload=workload, core=cores,
             threads=len(cores), cluster="mixed" if set(cores) & set(PRIME_CORES) and set(cores) & set(GOLD_CORES)
             else ("gold" if set(cores) <= set(GOLD_CORES) else "prime"),
             metadata={"mask_label": label, "mask_hex": mask_hex(cores),
                       "omitted_core": next((core for core in (*GOLD_CORES, *PRIME_CORES)
                                             if core not in cores), "")})
        for workload in WORKLOADS
        for label, cores in mask_cases()
    )
    rows, log = run_cases(profiler=profiler, args=args, cases=cases,
                          phase="task8_3", counters=None)
    try:
        summary_path, summary = write_summary(log, rows, ("workload", "case", "core_mask", "cores", "threads"))
        # Recover the human-readable mask label from the case name.  It contains
        # the workload as a prefix, which is unambiguous for the four fixed names.
        for row in summary:
            workload = row["workload"]
            row["mask_label"] = row["case"][len(workload) + 1:]
        best_rows = []
        print("\nTask 8.3 best versus worst:")
        for workload in WORKLOADS:
            options = [row for row in summary if row["workload"] == workload]
            best = min(options, key=lambda row: float(row["mean_time_us_per_iter"]))
            worst = max(options, key=lambda row: float(row["mean_time_us_per_iter"]))
            seven_core = [row for row in options if int(row["threads"]) == 7]
            best_seven = min(seven_core, key=lambda row: float(row["mean_time_us_per_iter"]))
            omitted = next(core for core in (*GOLD_CORES, *PRIME_CORES)
                           if str(core) not in best_seven["cores"].split(","))
            out = {
                "workload": workload,
                "best_mask": best["mask_label"],
                "best_mask_hex": best["core_mask"],
                "best_time_us_per_iter": best["mean_time_us_per_iter"],
                "worst_mask": worst["mask_label"],
                "worst_mask_hex": worst["core_mask"],
                "worst_time_us_per_iter": worst["mean_time_us_per_iter"],
                "worst_to_best_ratio": (float(worst["mean_time_us_per_iter"])
                                        / float(best["mean_time_us_per_iter"])),
                "best_seven_core_mask": best_seven["mask_label"],
                "best_core_to_leave_idle": omitted,
                "best_seven_time_us_per_iter": best_seven["mean_time_us_per_iter"],
            }
            best_rows.append(out)
            print(f"  {workload:12s} best={out['best_mask']:14s} "
                  f"worst/best={out['worst_to_best_ratio']:.3f}x  "
                  f"best 7-core={out['best_seven_core_mask']} (idle C{omitted})")
        best_path = log.path.with_name(log.path.stem + "_best.csv")
        write_csv(best_path, best_rows)
        # Include the readable mask label in the report-facing summary too.
        write_csv(summary_path, summary)
        print(f"\nwrote {log.path}, {log.jsonl_path}, {summary_path}, and {best_path}")
    finally:
        log.close()


if __name__ == "__main__":
    main()
