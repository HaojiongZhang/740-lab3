"""Task 9 GPU memlab sweeps.

Run inside the lab container, as the normal adb shell user (not adb root):

    python scripts/task9_memlab.py --section timing
    python scripts/task9_memlab.py --section power

The timing sections cover Tasks 9.1, 9.2, 9.4, 9.5, 9.7 and 9.8.  Task 9.6
is a written mitigation discussion.  Results are written under results/.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from profiler import Backend, Device, Profiler, Task


OUT = ROOT / "results"


def make_task(label: str, params: dict[str, Any]) -> Task:
    return Task("gpu_memlab", Backend.GPU, params=dict(params), name=label)


def run_point(profiler: Profiler, label: str, params: dict[str, Any],
              repeats: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for trial in range(1, repeats + 1):
        result = profiler.run(make_task(label, params))
        if not result.ok:
            raise RuntimeError(f"{label} trial {trial}: {result.error or 'GPU run failed'}")
        fields = result.fields
        rows.append({
            "trial": trial,
            "time_us": result.compute_us,
            "gbps": result.gbps,
            "gflops": result.gflops,
            "useful_gbps": fields.get("useful_gbps"),
            "lane_util_pct": fields.get("lane_util_pct"),
            "nnz": fields.get("nnz"),
            "params_json": json.dumps(params, sort_keys=True),
        })
    metric = "gflops" if params.get("mode") == "divergence" else (
        "gbps" if params.get("mode") == "coalesce" else "time_us")
    values = [float(row[metric]) for row in rows if row[metric] is not None]
    mean = statistics.fmean(values)
    cov = (statistics.stdev(values) / mean * 100.0
           if len(values) > 1 and mean else 0.0)
    summary = {
        "case": label,
        "mode": params["mode"],
        **{key: value for key, value in params.items() if key != "mode"},
        "mean_time_us": statistics.fmean(float(r["time_us"]) for r in rows),
        "mean_gbps": statistics.fmean(float(r["gbps"]) for r in rows
                                      if r["gbps"] is not None) if any(r["gbps"] is not None for r in rows) else None,
        "mean_gflops": statistics.fmean(float(r["gflops"]) for r in rows
                                        if r["gflops"] is not None) if any(r["gflops"] is not None for r in rows) else None,
        "mean_useful_gbps": statistics.fmean(float(r["useful_gbps"]) for r in rows
                                             if r["useful_gbps"] is not None) if any(r["useful_gbps"] is not None for r in rows) else None,
        "mean_lane_util_pct": statistics.fmean(float(r["lane_util_pct"]) for r in rows
                                               if r["lane_util_pct"] is not None) if any(r["lane_util_pct"] is not None for r in rows) else None,
        "sample_cov_pct": cov,
        "trials": len(rows),
    }
    for row in rows:
        row.update({"case": label, "mode": params["mode"]})
    return {"rows": rows, "summary": summary}


def timing_cases() -> list[tuple[str, dict[str, Any], str]]:
    cases: list[tuple[str, dict[str, Any], str]] = []
    for stride in (1, 2, 4, 8, 16, 32, 64):
        cases.append((f"t9_1_stride{stride}", {
            "mode": "coalesce", "kb": 65536, "stride": stride,
            "global": 262144, "local": 256, "reps": 100,
        }, "9.1"))
    for kb in (8192, 65536):
        cases.append((f"t9_2_kb{kb}_stride16", {
            "mode": "coalesce", "kb": kb, "stride": 16,
            "global": 262144, "local": 256, "reps": 100,
        }, "9.2"))
    for active in (64, 32, 16, 8, 4, 2, 1):
        cases.append((f"t9_4_active{active}", {
            "mode": "divergence", "kb": 65536, "active": active,
            "global": 262144, "local": 256, "reps": 1000,
        }, "9.4"))
    for local in (8, 16, 32, 64, 128, 256):
        cases.append((f"t9_5_local{local}", {
            "mode": "divergence", "kb": 65536, "active": 64,
            "global": 262144, "local": local, "reps": 1000,
        }, "9.5"))
    for rows, cols, tag in ((4000, 4000, "4000"), (4096, 4096, "4096")):
        for density in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
            cases.append((f"t9_{'7' if rows == 4000 else '8'}_spmv_{tag}_{density:g}", {
                "mode": "spmv", "rows": rows, "cols": cols,
                "density": density, "local": 256, "reps": 10,
            }, "9.7" if rows == 4000 else "9.8"))
        cases.append((f"t9_{'7' if rows == 4000 else '8'}_densemv_{tag}", {
            "mode": "densemv", "rows": rows, "cols": cols,
            "local": 256, "reps": 10,
        }, "9.7" if rows == 4000 else "9.8"))
    return cases


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--section", choices=("timing", "power"), default="timing")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")
    profiler = Profiler(Device())
    if args.section == "timing":
        raw: list[dict[str, Any]] = []
        summary: list[dict[str, Any]] = []
        for label, params, task in timing_cases():
            result = run_point(profiler, label, params, args.repeats)
            for row in result["rows"]:
                row["task"] = task
            result["summary"]["task"] = task
            raw.extend(result["rows"])
            summary.append(result["summary"])
            value = result["summary"].get("mean_gbps") or result["summary"].get("mean_gflops") or result["summary"]["mean_time_us"]
            print(f"{label:32s} {value:10.3f}  CoV {result['summary']['sample_cov_pct']:.2f}%")
        output = Path(args.output) if args.output else OUT / "task9_memlab_timing.csv"
        write_rows(output, raw)
        write_rows(output.with_name(output.stem + "_summary.csv"), summary)
        (output.with_suffix(".json")).write_text(json.dumps({"raw": raw, "summary": summary}, indent=2))
        print(f"wrote {output}, {output.with_name(output.stem + '_summary.csv')}, and {output.with_suffix('.json')}")
        return
    power_rows = []
    cases = [
        ("t9_3_stride1", {"mode": "coalesce", "kb": 65536, "stride": 1,
                          "global": 262144, "local": 256, "reps": 5000}),
        ("t9_3_stride16", {"mode": "coalesce", "kb": 65536, "stride": 16,
                           "global": 262144, "local": 256, "reps": 5000}),
    ]
    for label, params in cases:
        print(f"power window: {label} (35 seconds)")
        row = profiler.power(make_task(label, params), seconds=35)
        row.update({"case": label, "mode": params["mode"], "stride": params["stride"],
                    "kb": params["kb"], "reps": params["reps"]})
        power_rows.append(row)
    output = Path(args.output) if args.output else OUT / "task9_power.csv"
    write_rows(output, power_rows)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
