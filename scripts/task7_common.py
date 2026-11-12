"""Shared measurement and recording helpers for the Task 7 experiment drivers.

The drivers deliberately use the public ``Profiler.run`` and
``Profiler.run_parallel`` APIs.  They keep every raw trial, including the
aggressors' durations, so a result can be audited instead of being reduced to
one printed slowdown number.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

# These drivers are invoked as ``python scripts/task7_*.py`` from the repo
# root, but keeping the root explicit also makes direct invocation reliable.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from profiler import Profiler, Result, Task, ThermalReader


TaskFactory = Callable[[], Task]

# These are the metrics Task 7 calls for.  A metric that was not requested by
# the Task's counter preset is written as an empty CSV cell rather than 0.
DERIVED_METRICS = (
    "ipc",
    "effective_ghz",
    "backend_stall_pct",
    "slot_backend_pct",
    "mem_stall_pct",
    "l1d_miss_pct",
    "l2d_miss_pct",
)

CSV_FIELDS = (
    "run_id",
    "timestamp",
    "scenario",
    "trial",
    "position_in_trial",
    "victim_name",
    "victim_compute_us",
    "victim_us_per_iter",
    "victim_wall_us",
    "victim_gflops",
    "victim_gbps",
    "victim_fingerprint",
    "victim_temp_c",
    "victim_too_hot",
    *(f"victim_{metric}" for metric in DERIVED_METRICS),
    "aggressor_compute_us_json",
    "aggressor_wall_us_json",
    "all_aggressors_longer_than_victim",
    "error",
)


def build_parser(description: str, default_output: str) -> argparse.ArgumentParser:
    """Return the common CLI parser used by every Task 7 driver."""
    ap = argparse.ArgumentParser(description=description,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--reps", type=int, default=3,
                    help="timed trials for every scenario")
    ap.add_argument("--pause", type=float, default=15.0,
                    help="seconds idle between dispatches")
    ap.add_argument("--max-temp", type=float, default=85.0,
                    help="stop before a dispatch at/above this hottest-zone temperature; 0 disables")
    ap.add_argument("--output", default=default_output,
                    help="raw-trial CSV path; a JSONL log and summary CSV sit alongside it")
    ap.add_argument("--append", action="store_true",
                    help="append to an existing raw CSV/JSONL instead of refusing to overwrite it")
    return ap


def _duration_us(result: Result) -> float | None:
    """Use the kernel's wall time when it exists, otherwise its compute time."""
    return result.wall_us if result.wall_us is not None else result.compute_us


def _result_json(result: Result) -> dict:
    return {
        "task": result.task.name,
        "ok": result.ok,
        "compute_us": result.compute_us,
        "compute_us_per_iter": result.compute_us_per_iter,
        "wall_us": result.wall_us,
        "gflops": result.gflops,
        "gbps": result.gbps,
        "fingerprint": result.fingerprint,
        "fields": result.fields,
        "counters": result.counters,
        "derived": result.derived,
        "temp_c": result.temp_c,
        "temps": result.temps,
        "too_hot": result.too_hot,
        "error": result.error,
        "raw": result.raw,
    }


def _open_outputs(args: argparse.Namespace) -> tuple[Path, csv.DictWriter, object, object]:
    output = Path(args.output)
    if output.suffix != ".csv":
        raise SystemExit("--output must end in .csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output.with_suffix(".jsonl")
    exists = output.exists() or raw_path.exists()
    if exists and not args.append:
        raise SystemExit(
            f"{output} or {raw_path} already exists. Use --append to keep adding data, "
            "or choose a new --output path.")
    csv_file = output.open("a" if args.append else "w", newline="", encoding="utf-8")
    raw_file = raw_path.open("a" if args.append else "w", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    if not args.append or output.stat().st_size == 0:
        writer.writeheader()
        csv_file.flush()
    return output, writer, csv_file, raw_file


def _check_temperature(thermal: ThermalReader, max_temp: float) -> None:
    if max_temp <= 0:
        return
    temps = thermal.read()
    if temps and max(temps.values()) >= max_temp:
        zone, temp = max(temps.items(), key=lambda pair: pair[1])
        raise RuntimeError(
            f"temperature precheck stopped the run: {temp:.1f} C on {zone}; "
            "let the board cool, then resume with --append")


def _record_trial(
    *,
    writer: csv.DictWriter,
    csv_file: object,
    raw_file: object,
    run_id: str,
    scenario: str,
    trial: int,
    position: int,
    results: Sequence[Result],
) -> dict:
    """Persist one victim-alone or victim-plus-aggressors dispatch immediately."""
    victim, aggressors = results[0], results[1:]
    victim_duration = _duration_us(victim)
    aggressor_compute = {r.task.name: r.compute_us for r in aggressors}
    aggressor_wall = {r.task.name: _duration_us(r) for r in aggressors}
    overlap_known = victim_duration is not None and all(_duration_us(r) is not None for r in aggressors)
    all_longer = (all(_duration_us(r) > victim_duration for r in aggressors)
                  if aggressors and overlap_known else None)
    errors = [f"{r.task.name}: {r.error or 'failed'}" for r in results if not r.ok]

    row = {
        "run_id": run_id,
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "scenario": scenario,
        "trial": trial,
        "position_in_trial": position,
        "victim_name": victim.task.name,
        "victim_compute_us": victim.compute_us,
        "victim_us_per_iter": victim.compute_us_per_iter,
        "victim_wall_us": victim.wall_us,
        "victim_gflops": victim.gflops,
        "victim_gbps": victim.gbps,
        "victim_fingerprint": victim.fingerprint,
        "victim_temp_c": victim.temp_c,
        "victim_too_hot": victim.too_hot,
        "aggressor_compute_us_json": json.dumps(aggressor_compute, sort_keys=True),
        "aggressor_wall_us_json": json.dumps(aggressor_wall, sort_keys=True),
        "all_aggressors_longer_than_victim": all_longer,
        "error": "; ".join(errors),
    }
    row.update({f"victim_{metric}": victim.derived.get(metric) for metric in DERIVED_METRICS})
    writer.writerow(row)
    csv_file.flush()
    raw_file.write(json.dumps({
        "run_id": run_id,
        "scenario": scenario,
        "trial": trial,
        "position_in_trial": position,
        "results": [_result_json(result) for result in results],
    }) + "\n")
    raw_file.flush()
    return row


def _mean_or_none(values: list[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    return statistics.fmean(clean) if clean else None


def _cov_pct(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = statistics.fmean(values)
    return statistics.stdev(values) / mean * 100 if mean else None


def _write_summary(output: Path, rows: Sequence[dict]) -> Path:
    """Summarise the current invocation and print the slowdown of each scenario."""
    by_scenario: dict[str, list[dict]] = {}
    for row in rows:
        if not row["error"] and row["victim_us_per_iter"] is not None:
            by_scenario.setdefault(row["scenario"], []).append(row)
    if "alone" not in by_scenario:
        raise RuntimeError("no successful victim-alone measurements were recorded")

    baseline = statistics.fmean(float(r["victim_us_per_iter"]) for r in by_scenario["alone"])
    fields = ("scenario", "n", "mean_victim_us_per_iter", "sample_cov_pct", "min_victim_us_per_iter",
              "max_victim_us_per_iter", "slowdown_vs_alone", "mean_victim_gbps", "mean_victim_ipc",
              "mean_effective_ghz", "mean_backend_stall_pct", "mean_mem_stall_pct",
              "mean_l1d_miss_pct", "mean_l2d_miss_pct")
    summary_path = output.with_name(output.stem + "_summary.csv")
    with summary_path.open("w", newline="", encoding="utf-8") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=fields)
        writer.writeheader()
        print("\nVictim summary (ratio > 1 means slower than alone):")
        for scenario, scenario_rows in by_scenario.items():
            times = [float(r["victim_us_per_iter"]) for r in scenario_rows]
            mean = statistics.fmean(times)
            summary = {
                "scenario": scenario,
                "n": len(times),
                "mean_victim_us_per_iter": mean,
                "sample_cov_pct": _cov_pct(times),
                "min_victim_us_per_iter": min(times),
                "max_victim_us_per_iter": max(times),
                "slowdown_vs_alone": mean / baseline,
                "mean_victim_gbps": _mean_or_none([r["victim_gbps"] for r in scenario_rows]),
                "mean_victim_ipc": _mean_or_none([r["victim_ipc"] for r in scenario_rows]),
                "mean_effective_ghz": _mean_or_none([r["victim_effective_ghz"] for r in scenario_rows]),
                "mean_backend_stall_pct": _mean_or_none([r["victim_backend_stall_pct"] for r in scenario_rows]),
                "mean_mem_stall_pct": _mean_or_none([r["victim_mem_stall_pct"] for r in scenario_rows]),
                "mean_l1d_miss_pct": _mean_or_none([r["victim_l1d_miss_pct"] for r in scenario_rows]),
                "mean_l2d_miss_pct": _mean_or_none([r["victim_l2d_miss_pct"] for r in scenario_rows]),
            }
            writer.writerow(summary)
            cov = summary["sample_cov_pct"]
            cov_text = f"{cov:.2f}%" if cov is not None else "n/a"
            print(f"  {scenario:26s} {mean:10.1f} us/iter  "
                  f"slowdown {summary['slowdown_vs_alone']:.3f}x  CoV {cov_text}")
    return summary_path


def run_experiment(
    *,
    profiler: Profiler,
    args: argparse.Namespace,
    victim: TaskFactory,
    cases: Sequence[tuple[str, Sequence[TaskFactory]]],
) -> None:
    """Run every case repeatedly, rotating their order to limit heat-order bias."""
    if args.reps < 1:
        raise SystemExit("--reps must be at least 1")
    if args.pause < 0:
        raise SystemExit("--pause cannot be negative")
    if not any(name == "alone" and not aggressors for name, aggressors in cases):
        raise ValueError("cases must contain an aggressor-free scenario named 'alone'")

    output, writer, csv_file, raw_file = _open_outputs(args)
    thermal = ThermalReader(profiler.device)
    run_id = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    rows: list[dict] = []
    try:
        for trial in range(1, args.reps + 1):
            # Every configuration occupies each early/late position over a full
            # cycle, avoiding a fixed "all-core case always runs hottest" bias.
            offset = (trial - 1) % len(cases)
            ordered_cases = [*cases[offset:], *cases[:offset]]
            for position, (scenario, aggressors) in enumerate(ordered_cases, start=1):
                _check_temperature(thermal, args.max_temp)
                tasks = [victim(), *(factory() for factory in aggressors)]
                results = (profiler.run_parallel(tasks) if aggressors else [profiler.run(tasks[0])])
                row = _record_trial(writer=writer, csv_file=csv_file, raw_file=raw_file,
                                    run_id=run_id, scenario=scenario, trial=trial,
                                    position=position, results=results)
                rows.append(row)
                if row["error"]:
                    raise RuntimeError(row["error"])
                duration = row["victim_us_per_iter"]
                if duration is None:
                    raise RuntimeError(f"{scenario}: victim did not report a compute time")
                print(f"trial {trial}/{args.reps}  {scenario:26s} "
                      f"victim={duration:.1f} us/iter", flush=True)
                if aggressors and row["all_aggressors_longer_than_victim"] is not True:
                    print("  WARNING: an aggressor did not outlast the victim. Increase its "
                          "iterations before using this trial for Task 7.", file=sys.stderr,
                          flush=True)
                if position != len(ordered_cases) and args.pause:
                    time.sleep(args.pause)
        summary_path = _write_summary(output, rows)
        print(f"\nwrote {output}, {output.with_suffix('.jsonl')}, and {summary_path}")
    finally:
        csv_file.close()
        raw_file.close()
