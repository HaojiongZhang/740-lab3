"""Task 9.1--9.3: GPU coalescing, a cache/occupancy hypothesis, and energy.

The main stride sweep holds the buffer at 64 MiB, as required for a DRAM rather
than cache experiment. The Task 9.2 cases distinguish two explanations for a
large-stride loss:

* a 64 KiB buffer should recover if cache residency is important; and
* changing the number of active work-items at fixed 64 MiB tests whether
  memory-level parallelism/occupancy is important.

Run with ADB in non-root mode:
    adb unroot
    python scripts/task9_memory.py

Use --skip-power for a timing-only pass. The default also runs the two 35 s
whole-board modeled-power windows required by Task 9.3.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from task9_common import (GpuCase, Profiler, ThermalReader, add_measurement_args,
                          mean_metric, precheck_temperature, require_shell_uid,
                          run_cases, gpu_task, write_csv, write_summary)


KB_DRAM = 65_536
GLOBAL_DEFAULT = 262_144
LOCAL_DEFAULT = 256
KERNEL_REPS = 100
STRIDES = (1, 2, 4, 8, 16, 32, 64)


def coalesce_case(label: str, *, stride: int, kb: int = KB_DRAM,
                  global_items: int = GLOBAL_DEFAULT,
                  local_items: int = LOCAL_DEFAULT,
                  kernel_reps: int = KERNEL_REPS,
                  phase_note: str = "") -> GpuCase:
    return GpuCase(
        label=label,
        params={
            "mode": "coalesce",
            "stride": stride,
            "kb": kb,
            "global": global_items,
            "local": local_items,
            "reps": kernel_reps,
        },
        metadata={
            "phase_note": phase_note,
            "purpose": "useful bandwidth; stride is measured in 4-byte floats",
        },
    )


def add_args(ap: argparse.ArgumentParser) -> None:
    add_measurement_args(ap, "results/task9_memory.csv", default_pause=5.0)
    ap.add_argument("--skip-power", action="store_true",
                    help="collect Tasks 9.1--9.2 timing only")
    ap.add_argument("--power-seconds", type=int, default=35,
                    help="whole-board power-model window; must be at least 30 seconds")
    ap.add_argument("--power-pause", type=float, default=90.0,
                    help="cooldown seconds before each power window")


def power_row(*, profiler: Profiler, thermal: ThermalReader, case: GpuCase,
              timing_gbps: float, args: argparse.Namespace) -> dict[str, Any]:
    if args.power_seconds < 30:
        raise SystemExit("--power-seconds must be at least 30")
    if args.power_pause:
        print(f"cooling {args.power_pause:.0f} s before {case.label} power window ...",
              flush=True)
        time.sleep(args.power_pause)
    precheck_temperature(thermal, args.max_temp)
    estimate = profiler.power(gpu_task(case), seconds=args.power_seconds)
    if not estimate.get("available") or estimate.get("power_w") is None:
        raise RuntimeError(f"{case.label}: no power estimate: "
                           f"{estimate.get('reason', estimate)}")
    temps = thermal.read()
    return {
        "phase": "task9_3",
        "case": case.label,
        "stride_floats": case.params["stride"],
        "byte_spacing": case.params["stride"] * 4,
        "kb": case.params["kb"],
        "global_items": case.params["global"],
        "local_items": case.params["local"],
        "power_kernel_reps": case.params["reps"],
        "timing_gbps": timing_gbps,
        "power_w": float(estimate["power_w"]),
        "energy_per_useful_gb": float(estimate["power_w"]) / timing_gbps,
        "energy_unit": "J/GB",
        "power_window_s": estimate.get("window_s"),
        "power_scope": estimate.get("scope"),
        "hottest_temp_after_c": max(temps.values()) if temps else None,
        "power_warnings": "; ".join(map(str, estimate.get("warnings") or [])),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_args(ap)
    args = ap.parse_args()
    if args.power_pause < 0:
        raise SystemExit("--power-pause cannot be negative")

    primary = tuple(
        coalesce_case(f"stride_{stride}", stride=stride, phase_note="Task 9.1 stride sweep")
        for stride in STRIDES
    )
    hypothesis = (
        coalesce_case("stride1_cache64k", stride=1, kb=64,
                      phase_note="Task 9.2 cache-resident control"),
        coalesce_case("stride16_cache64k", stride=16, kb=64,
                      phase_note="Task 9.2 cache-resident large-stride test"),
        coalesce_case("stride16_global16k", stride=16, global_items=16_384,
                      phase_note="Task 9.2 low-occupancy test"),
        coalesce_case("stride16_global1m", stride=16, global_items=1_048_576,
                      phase_note="Task 9.2 high-occupancy test"),
    )

    profiler = Profiler()
    rows, log = run_cases(profiler=profiler, args=args,
                          cases=(*primary, *hypothesis), phase="task9_1_2")
    try:
        summary_path, summary = write_summary(
            log, rows,
            ("case", "mode", "stride_floats", "byte_spacing", "kb",
             "global_items", "local_items", "kernel_reps"),
        )
        by_case = {row["case"]: row for row in summary}
        coalescing_rows = [
            {
                "stride_floats": stride,
                "byte_spacing": stride * 4,
                "mean_gbps": by_case[f"stride_{stride}"]["mean_gbps"],
                "sample_cov_pct": by_case[f"stride_{stride}"]["sample_cov_pct"],
                "mean_compute_us": by_case[f"stride_{stride}"]["mean_compute_us_per_iter"],
                "kb": KB_DRAM,
                "global_items": GLOBAL_DEFAULT,
                "local_items": LOCAL_DEFAULT,
            }
            for stride in STRIDES
        ]
        coalescing_path = log.path.with_name(log.path.stem + "_coalescing.csv")
        write_csv(coalescing_path, coalescing_rows)
        hypothesis_rows = [
            {
                "case": case.label,
                "hypothesis_test": case.metadata["phase_note"],
                "stride_floats": case.params["stride"],
                "byte_spacing": case.params["stride"] * 4,
                "kb": case.params["kb"],
                "global_items": case.params["global"],
                "local_items": case.params["local"],
                "mean_gbps": by_case[case.label]["mean_gbps"],
                "sample_cov_pct": by_case[case.label]["sample_cov_pct"],
                "mean_compute_us": by_case[case.label]["mean_compute_us_per_iter"],
            }
            for case in hypothesis
        ]
        hypothesis_path = log.path.with_name(log.path.stem + "_hypothesis.csv")
        write_csv(hypothesis_path, hypothesis_rows)
        print(f"\nwrote {log.path}, {log.jsonl_path}, {summary_path},")
        print(f"      {coalescing_path}, and {hypothesis_path}")

        if not args.skip_power:
            require_shell_uid(profiler)
            thermal = ThermalReader(profiler.device)
            power_cases = (
                coalesce_case("stride1_power", stride=1, kernel_reps=5_000,
                              phase_note="Task 9.3 power window"),
                coalesce_case("stride16_power", stride=16, kernel_reps=5_000,
                              phase_note="Task 9.3 power window"),
            )
            power_rows = [
                power_row(
                    profiler=profiler,
                    thermal=thermal,
                    case=case,
                    timing_gbps=mean_metric(
                        [row for row in rows if row["case"] == f"stride_{case.params['stride']}"],
                        "gbps",
                    ),
                    args=args,
                )
                for case in power_cases
            ]
            power_path = log.path.with_name(log.path.stem + "_power.csv")
            write_csv(power_path, power_rows)
            print(f"wrote {power_path}")
    finally:
        log.close()


if __name__ == "__main__":
    main()
