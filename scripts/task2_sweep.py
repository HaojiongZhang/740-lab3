#!/usr/bin/env python3
"""Run Task 2.3--2.5 measurements on the board.

Run inside the lab container from the repository root.  The script deliberately
keeps profiler output verbatim in a log so the report can cite the measurements.
It restores the deployment frequency state when it exits.
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import time

FREQS = [768000, 1497600, 2227200, 2880000, 3648000, 4396800]
ROOT = pathlib.Path(__file__).resolve().parents[1]
PIN = ROOT / "scripts" / "power_pin.py"
PROF = ROOT / "profiler.py"


def run(cmd: list[str], log, check: bool = True) -> None:
    line = "$ " + " ".join(cmd)
    print(line, flush=True)
    log.write(line + "\n")
    log.flush()
    p = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT)
    print(p.stdout, end="", flush=True)
    log.write(p.stdout)
    log.flush()
    if check and p.returncode:
        raise SystemExit(p.returncode)


def pin(khz: int, log) -> None:
    run([sys.executable, str(PIN), "prime", str(khz)], log)
    time.sleep(2)


def task23(log, repeats: int) -> None:
    for khz in FREQS:
        pin(khz, log)
        # Use sufficiently long dispatches; profiler reports timing, IPC and
        # effective frequency.  The variants are the handout's CPU workloads.
        run([sys.executable, str(PROF), "matmul", "cpu", "--core", "prime",
             "--variant", "eigen", "-p", "size=1024", "--repeat", str(repeats)], log)
        run([sys.executable, str(PROF), "convolution", "cpu", "--core", "prime",
             "--variant", "neon", "-p", "cin=32", "-p", "cout=32",
             "-p", "H=56", "-p", "W=56", "--repeat", str(repeats)], log)
        run([sys.executable, str(PROF), "saxpy", "cpu", "--core", "prime",
             "--variant", "neon", "-p", "size=16777216", "-p", "iterations=150",
             "--repeat", str(repeats)], log)


def task24(log, workload: str, repeats: int) -> None:
    # Counters are collected at every frequency so the outlier can be explained
    # using IPC, backend stalls, and slot-backend stalls.
    for khz in FREQS:
        pin(khz, log)
        if workload == "matmul":
            cmd = ["matmul", "cpu", "--core", "prime", "--variant", "eigen",
                   "-p", "size=1024", "--counters", "basic", "--repeat", str(repeats)]
        elif workload == "conv":
            cmd = ["convolution", "cpu", "--core", "prime", "--variant", "neon",
                   "-p", "cin=32", "-p", "cout=32", "-p", "H=56", "-p", "W=56",
                   "--counters", "basic", "--repeat", str(repeats)]
        else:
            cmd = ["saxpy", "cpu", "--core", "prime", "--variant", "neon",
                   "-p", "size=16777216", "-p", "iterations=150",
                   "--counters", "basic", "--repeat", str(repeats)]
        run([sys.executable, str(PROF), *cmd], log)


def task25(log, workload: str, repeats: int) -> None:
    # Compare the maximum clock with its nearest half-clock OPP, then repeat
    # with a cache-fitting working set.  Matmul's N=256 is cache-fitting;
    # SAXPY uses 64 KiB, and convolution uses a small 8x8x8x8 shape.
    high, half = FREQS[-1], FREQS[3]
    for khz in (high, half):
        pin(khz, log)
        if workload == "matmul":
            params = ["-p", "size=1024"]
            small = ["-p", "size=256"]
            base = ["matmul", "cpu", "--core", "prime", "--variant", "eigen"]
        elif workload == "conv":
            params = ["-p", "cin=32", "-p", "cout=32", "-p", "H=56", "-p", "W=56"]
            small = ["-p", "cin=8", "-p", "cout=8", "-p", "H=8", "-p", "W=8"]
            base = ["convolution", "cpu", "--core", "prime", "--variant", "neon"]
        else:
            params = ["-p", "size=16777216", "-p", "iterations=150"]
            small = ["-p", "size=16384", "-p", "iterations=1000"]
            base = ["saxpy", "cpu", "--core", "prime", "--variant", "neon"]
        run([sys.executable, str(PROF), *base, *params, "--counters", "basic",
             "--repeat", str(repeats)], log)
        run([sys.executable, str(PROF), *base, *small, "--counters", "basic",
             "--repeat", str(repeats)], log)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("23", "24", "25", "all"), default="all")
    ap.add_argument("--outlier", choices=("matmul", "conv", "saxpy"), default="saxpy")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--output", type=pathlib.Path, default=ROOT / "results" / "task2_3_5.log")
    a = ap.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with a.output.open("w") as log:
            if a.phase in ("23", "all"):
                task23(log, a.repeat)
            if a.phase in ("24", "all"):
                task24(log, a.outlier, a.repeat)
            if a.phase in ("25", "all"):
                task25(log, a.outlier, a.repeat)
    finally:
        subprocess.run([sys.executable, str(PIN), "restore"], cwd=ROOT)


if __name__ == "__main__":
    main()
