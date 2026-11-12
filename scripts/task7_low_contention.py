"""Task 7.3: reduction victim vs six compute-heavy Gold-cluster matmuls.

Run this inside the lab container:
    python scripts/task7_low_contention.py

The aggressor cores are busy, but their small matrix multiply has much higher
arithmetic intensity than SAXPY.  That makes this a useful contrast with the
DRAM-saturating experiment in task7_memory_contention.py.
"""

from __future__ import annotations

from task7_common import build_parser, run_experiment
from profiler import Backend, Profiler, matmul, reduction


def main() -> None:
    ap = build_parser(__doc__.splitlines()[0], "results/task7_low_contention.csv")
    ap.add_argument("--victim-size", type=int, default=16_777_216)
    ap.add_argument("--victim-iterations", type=int, default=300)
    ap.add_argument("--aggressor-size", type=int, default=256,
                    help="small enough to encourage cache reuse")
    ap.add_argument("--aggressor-iterations", type=int, default=200,
                    help="raise this if a matmul ends before the victim")
    args = ap.parse_args()

    def victim():
        return reduction(Backend.CPU, size=args.victim_size, iterations=args.victim_iterations,
                         core=7, counters="stalls", name="victim_reduction_c7")

    def matmul_on(core: int):
        return lambda: matmul(Backend.CPU, size=args.aggressor_size,
                              iterations=args.aggressor_iterations, core=core,
                              name=f"aggressor_matmul_c{core}")

    cases = (
        ("alone", ()),
        ("six_gold_matmul_c0_c5", tuple(matmul_on(core) for core in range(6))),
    )
    run_experiment(profiler=Profiler(), args=args, victim=victim, cases=cases)


if __name__ == "__main__":
    main()
