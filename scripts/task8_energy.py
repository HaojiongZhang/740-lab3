"""Task 8.6--8.8: match Prime matmul with Gold, then compare energy/work.

The script pins the required OPPs itself, always using enumerated cpufreq
setpoints.  It first finds the lowest Gold frequency at which six Gold workers
match one top-clock Prime worker on matmul.  It then measures whole-board power
for both matmul and SAXPY configurations and writes J/GFLOP or J/GB.

Run last, then restore deployment DVFS with ``python scripts/power_pin.py restore``.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import Any

import power_pin
from task8_common import (CURRENT_DEVICE, Case, Profiler, ThermalReader, TrialLog,
                          add_measurement_args, cov_pct, make_task, mean,
                          precheck_temperature, read_opps, write_csv)


def add_energy_args(ap: argparse.ArgumentParser) -> None:
    add_measurement_args(ap, "results/task8_energy.csv", default_pause=20.0)
    ap.add_argument("--power-seconds", type=int, default=35,
                    help="power-estimation window; must be at least 30 seconds")
    ap.add_argument("--power-pause", type=float, default=60.0,
                    help="extra cooling time before each power window")
    ap.add_argument("--skip-power", action="store_true",
                    help="collect Task 8.6 throughput only; omit Task 8.7/8.8 power windows")


def run_repeated(*, profiler: Profiler, thermal: ThermalReader, log: TrialLog,
                 case: Case, phase: str, reps: int, pause: float, max_temp: float,
                 position: int) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    for trial in range(1, reps + 1):
        precheck_temperature(thermal, max_temp)
        result = profiler.run(make_task(case))
        row = log.record(result, case, phase=phase, trial=trial, position=position)
        position += 1
        rows.append(row)
        if not result.ok:
            raise RuntimeError(f"{case.label}: {result.error or 'measurement failed'}")
        metric = result.gflops if result.gflops is not None else result.gbps
        unit = "GFLOP/s" if result.gflops is not None else "GB/s"
        print(f"  {phase:16s} {case.label:20s} trial {trial}/{reps}: "
              f"{result.compute_us_per_iter:.1f} us/iter  {metric:.2f} {unit}")
        if trial != reps and pause:
            time.sleep(pause)
    return rows, position


def throughput(rows: list[dict[str, Any]], field: str) -> tuple[float, float | None]:
    values = [float(row[field]) for row in rows if row.get(field) not in (None, "")]
    if not values:
        raise RuntimeError(f"no {field} was reported for this configuration")
    return statistics.fmean(values), cov_pct(values)


def power_record(*, profiler: Profiler, thermal: ThermalReader, case: Case,
                 phase: str, freq_khz: int, throughput_value: float, throughput_unit: str,
                 args: argparse.Namespace) -> dict[str, Any]:
    if args.power_seconds < 30:
        raise SystemExit("--power-seconds must be at least 30; the model is not validated below that")
    if args.power_pause:
        print(f"  cooling {args.power_pause:.0f} s before {phase} power window ...", flush=True)
        time.sleep(args.power_pause)
    precheck_temperature(thermal, args.max_temp)
    estimate = profiler.power(make_task(case), seconds=args.power_seconds)
    if not estimate.get("available") or estimate.get("power_w") is None:
        raise RuntimeError(f"{phase}: no power estimate: {estimate.get('reason', estimate)}")
    watts = float(estimate["power_w"])
    temps = thermal.read()
    return {
        "phase": phase,
        "workload": case.workload,
        "configuration": case.label,
        "frequency_khz": freq_khz,
        "threads": case.threads or 1,
        "throughput": throughput_value,
        "throughput_unit": throughput_unit,
        "power_w": watts,
        "energy_per_unit": watts / throughput_value,
        "energy_unit": "J/GFLOP" if throughput_unit == "GFLOP/s" else "J/GB",
        "power_window_s": estimate.get("window_s"),
        "power_scope": estimate.get("scope"),
        "hottest_temp_after_c": max(temps.values()) if temps else None,
        "power_warnings": "; ".join(map(str, estimate.get("warnings") or [])),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_energy_args(ap)
    args = ap.parse_args()
    if args.reps < 1 or args.pause < 0 or args.power_pause < 0:
        raise SystemExit("--reps must be >= 1; pause values must be non-negative")

    profiler = Profiler()
    CURRENT_DEVICE.set(profiler.device)
    thermal = ThermalReader(profiler.device)
    gold_opps = read_opps(profiler.device, "gold")
    prime_opps = read_opps(profiler.device, "prime")
    if not gold_opps or not prime_opps:
        raise SystemExit("could not read both cpufreq ladders")
    prime_top = max(prime_opps)

    # Task 8.6 specifies this exact comparison.  Thirty iterations amortise
    # process startup and are also the handout's Task 8.5 matmul setting.
    prime_matmul = Case("matmul_prime_c7", "matmul", 7, 1, "prime",
                        params={"iterations": 30})
    gold_matmul = lambda khz: Case(f"matmul_gold6_{khz}", "matmul", "gold", 6, "gold",
                                   params={"iterations": 30, "pin": True},
                                   metadata={"gold_khz": khz})
    prime_saxpy = Case("saxpy_prime_c7", "saxpy", 7, 1, "prime")
    gold_saxpy = lambda khz: Case(f"saxpy_gold6_{khz}", "saxpy", "gold", 6, "gold",
                                  params={"pin": True}, metadata={"gold_khz": khz})

    log = TrialLog(args.output, args.append)
    position = 1
    all_rows: list[dict[str, Any]] = []
    power_rows: list[dict[str, Any]] = []
    gold_sweep_rows: list[dict[str, Any]] = []
    try:
        print(f"pinning Prime at its highest enumerated OPP: {prime_top / 1000:.1f} MHz")
        power_pin.pin_cpu("prime", prime_top)
        prime_rows, position = run_repeated(profiler=profiler, thermal=thermal, log=log,
                                             case=prime_matmul, phase="task8_6_matmul",
                                             reps=args.reps, pause=args.pause,
                                             max_temp=args.max_temp, position=position)
        all_rows.extend(prime_rows)
        prime_gflops, prime_cov = throughput(prime_rows, "gflops")
        print(f"Prime C7 matmul: {prime_gflops:.2f} GFLOP/s (CoV {prime_cov:.2f}%)")

        chosen_khz: int | None = None
        chosen_gold_rows: list[dict[str, Any]] = []
        for khz in gold_opps:
            print(f"pinning Gold at {khz / 1000:.1f} MHz")
            power_pin.pin_cpu("gold", khz)
            rows, position = run_repeated(profiler=profiler, thermal=thermal, log=log,
                                           case=gold_matmul(khz), phase="task8_6_matmul",
                                           reps=args.reps, pause=args.pause,
                                           max_temp=args.max_temp, position=position)
            all_rows.extend(rows)
            gold_gflops, gold_cov = throughput(rows, "gflops")
            gold_sweep_rows.append({
                "gold_frequency_khz": khz,
                "gold_frequency_mhz": khz / 1000.0,
                "gold6_matmul_gflops": gold_gflops,
                "sample_cov_pct": gold_cov,
                "matches_top_prime": gold_gflops >= prime_gflops,
            })
            print(f"  Gold6 at {khz / 1000:.1f} MHz: {gold_gflops:.2f} GFLOP/s "
                  f"(CoV {gold_cov:.2f}%)")
            if gold_gflops >= prime_gflops:
                chosen_khz, chosen_gold_rows = khz, rows
                break
        if chosen_khz is None:
            raise RuntimeError("six Gold cores did not match top-clock Prime matmul at any enumerated Gold OPP")
        gold_gflops, gold_cov = throughput(chosen_gold_rows, "gflops")
        print(f"\nlowest matching Gold OPP: {chosen_khz / 1000:.1f} MHz "
              f"({gold_gflops:.2f} vs {prime_gflops:.2f} GFLOP/s)")

        # Task 8.8 uses exactly the frequency/configuration pair selected above.
        power_pin.pin_cpu("prime", prime_top)
        prime_saxpy_rows, position = run_repeated(profiler=profiler, thermal=thermal, log=log,
                                                   case=prime_saxpy, phase="task8_8_saxpy",
                                                   reps=args.reps, pause=args.pause,
                                                   max_temp=args.max_temp, position=position)
        all_rows.extend(prime_saxpy_rows)
        prime_gbps, prime_saxpy_cov = throughput(prime_saxpy_rows, "gbps")
        power_pin.pin_cpu("gold", chosen_khz)
        gold_saxpy_rows, position = run_repeated(profiler=profiler, thermal=thermal, log=log,
                                                  case=gold_saxpy(chosen_khz), phase="task8_8_saxpy",
                                                  reps=args.reps, pause=args.pause,
                                                  max_temp=args.max_temp, position=position)
        all_rows.extend(gold_saxpy_rows)
        gold_gbps, gold_saxpy_cov = throughput(gold_saxpy_rows, "gbps")

        comparison_rows = [
            {"workload": "matmul", "configuration": "Prime C7", "frequency_khz": prime_top,
             "threads": 1, "throughput": prime_gflops, "throughput_unit": "GFLOP/s",
             "throughput_cov_pct": prime_cov},
            {"workload": "matmul", "configuration": "Gold C0--5", "frequency_khz": chosen_khz,
             "threads": 6, "throughput": gold_gflops, "throughput_unit": "GFLOP/s",
             "throughput_cov_pct": gold_cov},
            {"workload": "saxpy", "configuration": "Prime C7", "frequency_khz": prime_top,
             "threads": 1, "throughput": prime_gbps, "throughput_unit": "GB/s",
             "throughput_cov_pct": prime_saxpy_cov},
            {"workload": "saxpy", "configuration": "Gold C0--5", "frequency_khz": chosen_khz,
             "threads": 6, "throughput": gold_gbps, "throughput_unit": "GB/s",
             "throughput_cov_pct": gold_saxpy_cov},
        ]
        if not args.skip_power:
            power_pin.pin_cpu("prime", prime_top)
            power_rows.append(power_record(profiler=profiler, thermal=thermal, case=prime_matmul,
                                           phase="task8_7_matmul_prime", freq_khz=prime_top,
                                           throughput_value=prime_gflops, throughput_unit="GFLOP/s", args=args))
            power_pin.pin_cpu("gold", chosen_khz)
            power_rows.append(power_record(profiler=profiler, thermal=thermal, case=gold_matmul(chosen_khz),
                                           phase="task8_7_matmul_gold", freq_khz=chosen_khz,
                                           throughput_value=gold_gflops, throughput_unit="GFLOP/s", args=args))
            power_pin.pin_cpu("prime", prime_top)
            power_rows.append(power_record(profiler=profiler, thermal=thermal, case=prime_saxpy,
                                           phase="task8_8_saxpy_prime", freq_khz=prime_top,
                                           throughput_value=prime_gbps, throughput_unit="GB/s", args=args))
            power_pin.pin_cpu("gold", chosen_khz)
            power_rows.append(power_record(profiler=profiler, thermal=thermal, case=gold_saxpy(chosen_khz),
                                           phase="task8_8_saxpy_gold", freq_khz=chosen_khz,
                                           throughput_value=gold_gbps, throughput_unit="GB/s", args=args))

        throughput_path = log.path.with_name(log.path.stem + "_throughput.csv")
        gold_sweep_path = log.path.with_name(log.path.stem + "_gold_sweep.csv")
        power_path = log.path.with_name(log.path.stem + "_power.csv")
        write_csv(throughput_path, comparison_rows)
        write_csv(gold_sweep_path, gold_sweep_rows)
        if power_rows:
            write_csv(power_path, power_rows)
        print(f"\nwrote {log.path}, {log.jsonl_path}, {throughput_path}, and {gold_sweep_path}"
              + (f", {power_path}" if power_rows else ""))
    finally:
        log.close()


if __name__ == "__main__":
    main()
