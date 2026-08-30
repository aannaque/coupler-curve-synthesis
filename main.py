"""
Reproduction entry point for:

  A. Aannaque, "Algebraic Redundancy, Point-Based Extensions, and Practical
  Selection for Fully Analytical Four-Bar Coupler-Curve Synthesis" (2026),
  extending R. Wu, R. Li, S. Bai, Mech. Mach. Theory 155 (2021) 104070.

Runs each reproduction script as a separate process (identical to invoking
it directly, e.g. `python coupler_curve_synthesis.py`) so behavior here is
exactly what a reviewer running the individual scripts by hand would see.

Usage:
    python main.py             baseline validation + this paper's Fig. 1 and
                                Fig. 2 (roughly 1-2 minutes)
    python main.py --sweep     also reproduce Table 1, the identifiability
                                sweep (slow: several minutes; ~36 solver
                                calls)
"""
import argparse
import subprocess
import sys


def run(script, label):
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}", flush=True)
    subprocess.run([sys.executable, script], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sweep', action='store_true',
                         help='also run identifiability_sweep.py (Table 1; several minutes)')
    args = parser.parse_args()

    run('coupler_curve_synthesis.py',
        "Step 1/3: independent re-derivation of the coupler-curve equation, "
        "reproduction of Wu, Li & Bai (2021) Tables 1-4 and Figs. 3-5, and the "
        "codimension-6 redundancy certificate example (this paper's Section 3)")

    run('precision_point_synthesis.py',
        "Step 2/3: precision-point synthesis demos -> this paper's Fig. 1 "
        "(noise-free recovery) and Fig. 2 (noisy recovery, Section 4)")

    if args.sweep:
        run('identifiability_sweep.py',
            "Step 3/3: identifiability sweep -> this paper's Table 1 "
            "(Section 5.2; slow, ~36 solver calls)")
    else:
        print("\nSkipping the identifiability sweep (Table 1) -- pass --sweep "
              "to run it (several minutes).")

    print("\nDone. Generated figures: fig3_case1_loop1.png, fig4_case1_loop2.png, "
          "fig5_case2.png (Wu et al. reproduction), demoA_exact_points.png "
          "(paper Fig. 1), demoB_noisy_points.png (paper Fig. 2).")


if __name__ == '__main__':
    main()
