"""Shared helpers for the Task 8 experiment drivers.

The scripts deliberately keep the frequency controls separate from measurement:
``power_pin.py`` changes device state, while these helpers verify and record
that state beside every result.  This makes a partially completed sweep
auditable and avoids silently comparing Gold and Prime at different clocks.
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
from typing import Any, Callable, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from profiler import (Backend, Device, Profiler, Result, ThermalReader,
                      convolution, matmul, reduction, saxpy)


GOLD_CORES = (0, 1, 2, 3, 4, 5)
PRIME_CORES = (6, 7)
POLICIES = {"gold": "policy0", "prime": "policy6"}

# These are exactly the four CPU workloads and stable run lengths from Task 6.
# They make Task 8 directly comparable with the preceding scaling experiment.
TASK8_PARAMS: dict[str, dict[str, int]] = {
    "saxpy": {"size": 16_777_216, "iterations": 100},
    "reduction": {"size": 16_777_216, "iterations": 120},
    "convolution": {"iterations": 150},
    "matmul": {"size": 1024, "iterations": 10},
}
_BUILDERS: dict[str, Callable[..., Any]] = {
    "saxpy": saxpy,
    "reduction": reduction,
    "convolution": convolution,
    "matmul": matmul,
}

DERIVED_FIELDS = (
    "ipc", "effective_ghz", "scaling_cur_freq_ghz", "backend_stall_pct",
    "mem_stall_pct", "slot_backend_pct", "l1d_miss_pct", "l2d_miss_pct",
)
RESULT_FIELDS = (
    "run_id", "timestamp", "phase", "case", "workload", "cluster", "core",
    "core_mask", "cores", "threads", "prime_weight", "common_clock_khz",
    "gold_min_khz", "gold_max_khz", "prime_min_khz", "prime_max_khz",
    "trial", "position_in_trial", "compute_us", "time_us_per_iter", "wall_us",
    "gflops", "gbps", "fingerprint", "temp_c", "too_hot",
    *(f"derived_{name}" for name in DERIVED_FIELDS),
    "error", "metadata_json",
)
SUMMARY_METRICS = (
    "time_us_per_iter", "gflops", "gbps", "derived_ipc",
    "derived_effective_ghz", "derived_backend_stall_pct",
    "derived_mem_stall_pct", "temp_c",
)


@dataclass(frozen=True)
class Case:
    """One serial CPU configuration in a Task 8 sweep."""

    label: str
    workload: str
    core: object
    threads: int | None
    cluster: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def add_measurement_args(ap: argparse.ArgumentParser, default_output: str,
                         *, default_pause: float = 20.0) -> None:
    ap.add_argument("--reps", type=int, default=3,
                    help="independent process launches per configuration")
    ap.add_argument("--pause", type=float, default=default_pause,
                    help="idle seconds between launches")
    ap.add_argument("--max-temp", type=float, default=85.0,
                    help="stop before a launch at/above this hottest-zone temperature; 0 disables")
    ap.add_argument("--output", default=default_output,
                    help="raw CSV output path; JSONL and summary CSV are written beside it")
    ap.add_argument("--append", action="store_true",
                    help="append raw trials instead of refusing to overwrite existing output")


def _shell_text(device: Device, command: str) -> str:
    out, err, rc = device.shell(command, timeout=30)
    if rc != 0:
        return ""
    return out.strip()


def read_policy(device: Device, cluster: str) -> dict[str, int | None]:
    """Read the two cpufreq limits for one cluster."""
    policy = POLICIES[cluster]
    base = f"/sys/devices/system/cpu/cpufreq/{policy}"
    result: dict[str, int | None] = {}
    for name in ("scaling_min_freq", "scaling_max_freq"):
        text = _shell_text(device, f"cat {base}/{name}")
        result[name] = int(text) if text.isdigit() else None
    return result


def read_opps(device: Device, cluster: str) -> list[int]:
    policy = POLICIES[cluster]
    text = _shell_text(device,
                       f"cat /sys/devices/system/cpu/cpufreq/{policy}/scaling_available_frequencies")
    return sorted({int(value) for value in text.split() if value.isdigit()})


def frequency_snapshot(device: Device) -> dict[str, int | None]:
    gold, prime = read_policy(device, "gold"), read_policy(device, "prime")
    return {
        "gold_min_khz": gold["scaling_min_freq"],
        "gold_max_khz": gold["scaling_max_freq"],
        "prime_min_khz": prime["scaling_min_freq"],
        "prime_max_khz": prime["scaling_max_freq"],
    }


def require_common_clock_lock(device: Device, requested_khz: int | None) -> int:
    """Pick a common OPP and refuse to run unless both policies are locked to it."""
    gold, prime = read_opps(device, "gold"), read_opps(device, "prime")
    common = sorted(set(gold) & set(prime))
    if not common:
        raise SystemExit("No common Gold/Prime OPPs were readable from cpufreq.")
    target = requested_khz if requested_khz is not None else common[-1]
    if target not in common:
        raise SystemExit(
            f"{target} kHz is not common to Gold and Prime. Common OPPs: {' '.join(map(str, common))}")
    state = frequency_snapshot(device)
    if (state["gold_min_khz"], state["gold_max_khz"],
            state["prime_min_khz"], state["prime_max_khz"]) != (target, target, target, target):
        raise SystemExit(
            f"Task 8.1 requires both clusters locked at {target} kHz. Run:\n"
            f"  python scripts/power_pin.py gold {target}\n"
            f"  python scripts/power_pin.py prime {target}\n"
            "Then rerun this script. The lock must read back as min=max on both policies.")
    return target


def require_cluster_locks(device: Device, *, allow_dvfs: bool = False) -> dict[str, int | None]:
    """Ensure Part B/C timing comparisons run with both cluster clocks fixed."""
    state = frequency_snapshot(device)
    locked = (state["gold_min_khz"] is not None
              and state["gold_min_khz"] == state["gold_max_khz"]
              and state["prime_min_khz"] is not None
              and state["prime_min_khz"] == state["prime_max_khz"])
    if not locked and not allow_dvfs:
        raise SystemExit(
            "Both clusters must be frequency-locked for this comparison. For top-clock runs:\n"
            "  python scripts/power_pin.py gold 3628800\n"
            "  python scripts/power_pin.py prime 4396800\n"
            "Use --allow-dvfs only for exploratory data you will not compare in the report.")
    if not locked:
        print("WARNING: recording a DVFS-controlled run; do not compare it with locked results.",
              file=sys.stderr)
    return state


def core_list(core: object) -> tuple[int, ...]:
    if isinstance(core, int):
        return (core,)
    if isinstance(core, (tuple, list)):
        return tuple(int(value) for value in core)
    if core == "gold":
        return GOLD_CORES
    if core == "prime":
        return PRIME_CORES
    if core == "all":
        return (*GOLD_CORES, *PRIME_CORES)
    raise ValueError(f"Task 8 case needs an explicit known core set, got {core!r}")


def mask_hex(cores: Iterable[int]) -> str:
    mask = 0
    for core in cores:
        mask |= 1 << int(core)
    return f"0x{mask:02x}"


def make_task(case: Case, *, counters: str | None = None):
    """Construct one of Task 8's CPU workloads with the documented parameters."""
    if case.workload not in _BUILDERS:
        raise ValueError(f"unknown Task 8 workload {case.workload!r}")
    params = dict(TASK8_PARAMS[case.workload])
    params.update(case.params)
    params.update({"core": case.core, "threads": case.threads,
                   "counters": counters, "name": case.label})
    return _BUILDERS[case.workload](Backend.CPU, **params)


def _json_result(result: Result) -> dict[str, Any]:
    return {
        "task": result.task.name, "ok": result.ok,
        "compute_us": result.compute_us, "compute_us_per_iter": result.compute_us_per_iter,
        "wall_us": result.wall_us, "gflops": result.gflops, "gbps": result.gbps,
        "fingerprint": result.fingerprint, "fields": result.fields,
        "counters": result.counters, "derived": result.derived,
        "temp_c": result.temp_c, "temps": result.temps, "error": result.error,
        "raw": result.raw,
    }


class TrialLog:
    """Incremental raw CSV/JSONL writer shared by the Task 8 drivers."""

    def __init__(self, output_name: str, append: bool):
        self.path = Path(output_name)
        if self.path.suffix != ".csv":
            raise SystemExit("--output must end in .csv")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.path.with_suffix(".jsonl")
        exists = self.path.exists() or self.jsonl_path.exists()
        if exists and not append:
            raise SystemExit(
                f"{self.path} or {self.jsonl_path} already exists. Use --append or choose --output.")
        self.csv_file = self.path.open("a" if append else "w", newline="", encoding="utf-8")
        self.jsonl_file = self.jsonl_path.open("a" if append else "w", encoding="utf-8")
        self.writer = csv.DictWriter(self.csv_file, fieldnames=RESULT_FIELDS)
        if not append or self.path.stat().st_size == 0:
            self.writer.writeheader()
            self.csv_file.flush()
        self.run_id = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")

    def record(self, result: Result, case: Case, *, phase: str, trial: int,
               position: int, common_clock_khz: int | None = None) -> dict[str, Any]:
        cores = core_list(case.core)
        row: dict[str, Any] = {name: "" for name in RESULT_FIELDS}
        row.update({
            "run_id": self.run_id,
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "phase": phase,
            "case": case.label,
            "workload": case.workload,
            "cluster": case.cluster,
            "core": cores[0] if len(cores) == 1 else "",
            "core_mask": mask_hex(cores),
            "cores": ",".join(map(str, cores)),
            "threads": case.threads or 1,
            "prime_weight": case.params.get("prime_weight", ""),
            "common_clock_khz": common_clock_khz or "",
            "trial": trial,
            "position_in_trial": position,
            "compute_us": result.compute_us,
            "time_us_per_iter": result.compute_us_per_iter,
            "wall_us": result.wall_us,
            "gflops": result.gflops,
            "gbps": result.gbps,
            "fingerprint": result.fingerprint,
            "temp_c": result.temp_c,
            "too_hot": result.too_hot,
            "error": result.error or "",
            "metadata_json": json.dumps(case.metadata, sort_keys=True),
        })
        row.update(frequency_snapshot(CURRENT_DEVICE.get()))
        row.update({f"derived_{name}": result.derived.get(name, "") for name in DERIVED_FIELDS})
        self.writer.writerow(row)
        self.csv_file.flush()
        self.jsonl_file.write(json.dumps({
            "run_id": self.run_id, "phase": phase, "trial": trial,
            "position_in_trial": position, "case": case.label,
            "metadata": case.metadata, "result": _json_result(result),
        }) + "\n")
        self.jsonl_file.flush()
        return row

    def close(self) -> None:
        self.csv_file.close()
        self.jsonl_file.close()


# TrialLog records a frequency snapshot immediately after every dispatch.  A
# tiny holder avoids threading a Device argument through every caller while
# keeping each script's Profiler as the source of truth.
class _CurrentDevice:
    device: Device | None = None

    def set(self, device: Device) -> None:
        self.device = device

    def get(self) -> Device:
        if self.device is None:
            raise RuntimeError("no Task 8 device is registered")
        return self.device


CURRENT_DEVICE = _CurrentDevice()


def precheck_temperature(thermal: ThermalReader, max_temp: float) -> None:
    if max_temp <= 0:
        return
    temps = thermal.read()
    if temps and max(temps.values()) >= max_temp:
        zone, temp = max(temps.items(), key=lambda item: item[1])
        raise RuntimeError(
            f"temperature precheck stopped the run at {temp:.1f} C on {zone}; "
            "let the board cool, then rerun with --append")


def run_cases(*, profiler: Profiler, args: argparse.Namespace, cases: Sequence[Case],
              phase: str, counters: str | None = None,
              common_clock_khz: int | None = None) -> tuple[list[dict[str, Any]], TrialLog]:
    """Run serial cases in a rotating order and persist every result immediately."""
    if args.reps < 1:
        raise SystemExit("--reps must be at least 1")
    if args.pause < 0:
        raise SystemExit("--pause cannot be negative")
    if not cases:
        raise ValueError("Task 8 sweep has no cases")
    CURRENT_DEVICE.set(profiler.device)
    thermal = ThermalReader(profiler.device)
    log = TrialLog(args.output, args.append)
    rows: list[dict[str, Any]] = []
    try:
        for trial in range(1, args.reps + 1):
            offset = (trial - 1) % len(cases)
            ordered = [*cases[offset:], *cases[:offset]]
            for position, case in enumerate(ordered, start=1):
                precheck_temperature(thermal, args.max_temp)
                result = profiler.run(make_task(case, counters=counters))
                row = log.record(result, case, phase=phase, trial=trial, position=position,
                                 common_clock_khz=common_clock_khz)
                rows.append(row)
                if not result.ok:
                    raise RuntimeError(f"{case.label}: {result.error or 'measurement failed'}")
                print(f"trial {trial}/{args.reps}  {case.label:30s}  "
                      f"{result.compute_us_per_iter:.1f} us/iter", flush=True)
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


def aggregate(rows: Sequence[dict[str, Any]], group_fields: Sequence[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("error") or row.get("time_us_per_iter") in (None, ""):
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
            if metric == "time_us_per_iter":
                out["sample_cov_pct"] = cov_pct(values)
                out["min_time_us_per_iter"] = min(values) if values else None
                out["max_time_us_per_iter"] = max(values) if values else None
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


def load_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except FileNotFoundError:
        raise SystemExit(f"Missing {path}. Run the preceding Task 8 script first.") from None
