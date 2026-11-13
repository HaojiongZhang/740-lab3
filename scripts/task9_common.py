"""Shared measurement and logging helpers for Task 9's GPU experiments.

Task 9 is deliberately separate from Task 8: GPU OpenCL dispatches must run
through an ordinary adb shell session, while Task 8's CPU clock controls
require adb root. These helpers reject a root ADB daemon before a sweep starts,
rotate case order between trials, and persist raw data immediately.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from profiler import Backend, Profiler, Result, Task, ThermalReader


RESULT_FIELDS = (
    "run_id", "timestamp", "phase", "case", "mode",
    "stride_floats", "byte_spacing", "kb", "active_lanes", "density",
    "rows", "cols", "global_items", "local_items", "kernel_reps",
    "trial", "position_in_trial", "compute_us", "compute_us_per_iter",
    "wall_us", "gflops", "gbps", "fingerprint", "temp_c", "too_hot",
    "error", "metadata_json",
)

SUMMARY_METRICS = (
    "compute_us_per_iter", "gflops", "gbps", "temp_c",
)


@dataclass(frozen=True)
class GpuCase:
    """One gpu_memlab invocation and report-facing metadata."""

    label: str
    params: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


def add_measurement_args(ap: argparse.ArgumentParser, default_output: str,
                         *, default_pause: float = 5.0) -> None:
    ap.add_argument("--trials", type=int, default=3,
                    help="independent process launches per configuration")
    ap.add_argument("--pause", type=float, default=default_pause,
                    help="idle seconds between launches")
    ap.add_argument("--max-temp", type=float, default=85.0,
                    help="stop before a launch at/above this hottest-zone temperature; 0 disables")
    ap.add_argument("--output", default=default_output,
                    help="raw CSV output path; JSONL and summary CSV are written beside it")
    ap.add_argument("--append", action="store_true",
                    help="append raw trials instead of refusing to overwrite existing output")


def require_shell_uid(profiler: Profiler) -> None:
    """Fail with the actionable remedy when the Adreno ICD cannot be reached."""
    out, err, rc = profiler.device.shell("id", timeout=30)
    if rc != 0:
        raise SystemExit(f"could not identify the ADB shell user: {err or out}")
    if "uid=0" in out:
        raise SystemExit(
            "Task 9 GPU runs require the normal ADB shell user. Run adb unroot, "
            "wait for the device, then rerun this script. The Adreno OpenCL ICD is "
            "not visible from the root/su domain."
        )


def gpu_task(case: GpuCase) -> Task:
    return Task("gpu_memlab", Backend.GPU, dict(case.params), name=case.label)


def _json_result(result: Result) -> dict[str, Any]:
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
        "temp_c": result.temp_c,
        "temps": result.temps,
        "error": result.error,
        "raw": result.raw,
    }


class TrialLog:
    """Incremental CSV/JSONL writer so a partial board run remains usable."""

    def __init__(self, output_name: str, append: bool):
        self.path = Path(output_name)
        if self.path.suffix != ".csv":
            raise SystemExit("--output must end in .csv")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.path.with_suffix(".jsonl")
        exists = self.path.exists() or self.jsonl_path.exists()
        if exists and not append:
            raise SystemExit(
                f"{self.path} or {self.jsonl_path} already exists. "
                "Use --append or choose --output."
            )
        self.csv_file = self.path.open("a" if append else "w", newline="", encoding="utf-8")
        self.jsonl_file = self.jsonl_path.open("a" if append else "w", encoding="utf-8")
        self.writer = csv.DictWriter(self.csv_file, fieldnames=RESULT_FIELDS)
        if not append or self.path.stat().st_size == 0:
            self.writer.writeheader()
            self.csv_file.flush()
        self.run_id = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")

    def record(self, result: Result, case: GpuCase, *, phase: str,
               trial: int, position: int) -> dict[str, Any]:
        p = case.params
        stride = p.get("stride", "")
        row: dict[str, Any] = {name: "" for name in RESULT_FIELDS}
        row.update({
            "run_id": self.run_id,
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "phase": phase,
            "case": case.label,
            "mode": p.get("mode", ""),
            "stride_floats": stride,
            "byte_spacing": int(stride) * 4 if isinstance(stride, int) else "",
            "kb": p.get("kb", ""),
            "active_lanes": p.get("active", ""),
            "density": p.get("density", ""),
            "rows": p.get("rows", ""),
            "cols": p.get("cols", ""),
            "global_items": p.get("global", ""),
            "local_items": p.get("local", ""),
            "kernel_reps": p.get("reps", ""),
            "trial": trial,
            "position_in_trial": position,
            "compute_us": result.compute_us,
            "compute_us_per_iter": result.compute_us_per_iter,
            "wall_us": result.wall_us,
            "gflops": result.gflops,
            "gbps": result.gbps,
            "fingerprint": result.fingerprint,
            "temp_c": result.temp_c,
            "too_hot": result.too_hot,
            "error": result.error or "",
            "metadata_json": json.dumps(case.metadata, sort_keys=True),
        })
        self.writer.writerow(row)
        self.csv_file.flush()
        self.jsonl_file.write(json.dumps({
            "run_id": self.run_id,
            "phase": phase,
            "trial": trial,
            "position_in_trial": position,
            "case": case.label,
            "params": case.params,
            "metadata": case.metadata,
            "result": _json_result(result),
        }, default=str) + "\n")
        self.jsonl_file.flush()
        return row

    def close(self) -> None:
        self.csv_file.close()
        self.jsonl_file.close()


def precheck_temperature(thermal: ThermalReader, max_temp: float) -> None:
    if max_temp <= 0:
        return
    temps = thermal.read()
    if temps and max(temps.values()) >= max_temp:
        zone, temp = max(temps.items(), key=lambda item: item[1])
        raise RuntimeError(
            f"temperature precheck stopped the run at {temp:.1f} C on {zone}; "
            "let the board cool, then rerun with --append"
        )


def run_cases(*, profiler: Profiler, args: argparse.Namespace,
              cases: Sequence[GpuCase], phase: str) -> tuple[list[dict[str, Any]], TrialLog]:
    """Run cases in a different cyclic order each trial and log every result."""
    if args.trials < 1:
        raise SystemExit("--trials must be at least 1")
    if args.pause < 0:
        raise SystemExit("--pause cannot be negative")
    if not cases:
        raise ValueError("Task 9 sweep has no cases")
    require_shell_uid(profiler)
    thermal = ThermalReader(profiler.device)
    log = TrialLog(args.output, args.append)
    rows: list[dict[str, Any]] = []
    try:
        for trial in range(1, args.trials + 1):
            offset = (trial - 1) % len(cases)
            ordered = [*cases[offset:], *cases[:offset]]
            for position, case in enumerate(ordered, start=1):
                precheck_temperature(thermal, args.max_temp)
                result = profiler.run(gpu_task(case))
                row = log.record(result, case, phase=phase, trial=trial, position=position)
                rows.append(row)
                if not result.ok:
                    raise RuntimeError(f"{case.label}: {result.error or 'measurement failed'}")
                rate, unit = (
                    (result.gflops, "GFLOP/s") if result.gflops is not None
                    else (result.gbps, "GB/s")
                )
                rate_text = f"  {rate:.2f} {unit}" if rate is not None else ""
                print(f"trial {trial}/{args.trials}  {case.label:36s}  "
                      f"{result.compute_us_per_iter:.1f} us{rate_text}", flush=True)
                if position != len(ordered) and args.pause:
                    time.sleep(args.pause)
    except BaseException:
        log.close()
        raise
    return rows, log


def _numbers(rows: Sequence[dict[str, Any]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(field)
        if value not in (None, ""):
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                pass
    return values


def mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def cov_pct(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    avg = statistics.fmean(values)
    return statistics.stdev(values) / avg * 100.0 if avg else None


def aggregate(rows: Sequence[dict[str, Any]],
              group_fields: Sequence[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("error") or row.get("compute_us_per_iter") in (None, ""):
            continue
        key = tuple(str(row.get(field, "")) for field in group_fields)
        grouped.setdefault(key, []).append(row)
    summaries: list[dict[str, Any]] = []
    for key, group in grouped.items():
        out: dict[str, Any] = dict(zip(group_fields, key))
        out["n"] = len(group)
        for metric in SUMMARY_METRICS:
            values = _numbers(group, metric)
            out[f"mean_{metric}"] = mean(values)
            if metric == "compute_us_per_iter":
                out["sample_cov_pct"] = cov_pct(values)
                out["min_compute_us_per_iter"] = min(values) if values else None
                out["max_compute_us_per_iter"] = max(values) if values else None
        summaries.append(out)
    return summaries


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for field_name in row:
            if field_name not in fields:
                fields.append(field_name)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(log: TrialLog, rows: Sequence[dict[str, Any]],
                  group_fields: Sequence[str]) -> tuple[Path, list[dict[str, Any]]]:
    summary = aggregate(rows, group_fields)
    path = log.path.with_name(log.path.stem + "_summary.csv")
    write_csv(path, summary)
    return path, summary


def mean_metric(rows: Sequence[dict[str, Any]], field: str) -> float:
    values = _numbers(rows, field)
    if not values:
        raise RuntimeError(f"no {field} reported")
    return statistics.fmean(values)
