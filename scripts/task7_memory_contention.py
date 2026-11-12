"""Task 7.1/7.2: reduction victim vs one Prime or six Gold SAXPY aggressors.

Run this inside the lab container:
    python scripts/task7_memory_contention.py

SAXPY has very low arithmetic intensity and writes every element, making it a
strong DRAM-bandwidth aggressor.  The three cases are rotated each trial to
avoid always measuring the all-Gold case when the board is hottest.
"""

from __future__ import annotations

from task7_common import build_parser, run_experiment
from profiler import Backend, Profiler, reduction, saxpy


def main() -> None:
    ap = build_parser(__doc__.splitlines()[0], "results/task7_memory_contention.csv")
    ap.add_argument("--size", type=int, default=16_777_216,
                    help="vector length for both reduction and SAXPY")
    ap.add_argument("--victim-iterations", type=int, default=300)
    ap.add_argument("--aggressor-iterations", type=int, default=150,
                    help="raise this if the script warns that SAXPY ended first")
    args = ap.parse_args()

    def victim():
        return reduction(Backend.CPU, size=args.size, iterations=args.victim_iterations,
                         core=7, counters="stalls", name="victim_reduction_c7")

    def saxpy_on(core: int):
        return lambda: saxpy(Backend.CPU, size=args.size, iterations=args.aggressor_iterations,
                             core=core, name=f"aggressor_saxpy_c{core}")

    cases = (
        ("alone", ()),
        ("one_prime_saxpy_c6", (saxpy_on(6),)),
        ("six_gold_saxpy_c0_c5", tuple(saxpy_on(core) for core in range(6))),
    )
    run_experiment(profiler=Profiler(), args=args, victim=victim, cases=cases)


if __name__ == "__main__":
    main()
