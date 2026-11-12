"""Task 7.4: matmul victim with cache counters vs a Prime-core SAXPY aggressor.

Run this inside the lab container:
    python scripts/task7_matmul_cache.py

The output summary includes the victim's IPC and L1D/L2 miss percentages in
both cases.  Use those rates, not just the slowdown, when identifying the
shared resource responsible for interference.
"""

from __future__ import annotations

from task7_common import build_parser, run_experiment
from profiler import Backend, Profiler, matmul, saxpy


def main() -> None:
    ap = build_parser(__doc__.splitlines()[0], "results/task7_matmul_cache.csv")
    ap.add_argument("--victim-size", type=int, default=512)
    ap.add_argument("--victim-iterations", type=int, default=10)
    ap.add_argument("--aggressor-size", type=int, default=16_777_216)
    ap.add_argument("--aggressor-iterations", type=int, default=150,
                    help="raise this if SAXPY ends before the matmul victim")
    args = ap.parse_args()

    def victim():
        return matmul(Backend.CPU, size=args.victim_size, iterations=args.victim_iterations,
                      core=7, counters="cache", name="victim_matmul_c7")

    def prime_saxpy():
        return saxpy(Backend.CPU, size=args.aggressor_size,
                     iterations=args.aggressor_iterations, core=6,
                     name="aggressor_saxpy_c6")

    cases = (
        ("alone", ()),
        ("prime_saxpy_c6", (prime_saxpy,)),
    )
    run_experiment(profiler=Profiler(), args=args, victim=victim, cases=cases)


if __name__ == "__main__":
    main()
