"""
Extension #3b: how many points / how much noise does the exact-seeded fit
pipeline (precision_point_synthesis.py) actually need before the recovered
linkage reliably matches the ground truth, rather than landing on one of the
several visibly-different but similarly-well-fitting mechanisms seen in
Demo B? This is a concrete, quantifiable identifiability study the original
paper (and the classical precision-point-synthesis literature it cites)
does not provide for this method.

Fixed ground-truth linkage, swept over:
  - N  = number of sampled task points   in {15, 30, 60}
  - sigma = point coordinate noise std   in {0, 0.002, 0.005}
For each (N, sigma) cell, run several independent trials (different random
point subset + noise draw) through synthesize_from_points(), and record
whether the TOP-RANKED candidate matches ground truth within tolerance.
"""

import numpy as np
import time
import sys

from coupler_curve_synthesis import trace_linkage
from precision_point_synthesis import synthesize_from_points, SYM_ORDER

TRUTH = dict(m=0.12, h=0.22, b1=0.05, b2=-0.03, c1=0.31, c2=0.02, l2=0.28, l3=0.33, l4=0.19)
MATCH_TOL = 0.02  # absolute parameter tolerance to call a candidate "ground truth"


def matches_truth(entry, tol=MATCH_TOL):
    return all(abs(entry[k] - TRUTH[k]) <= tol for k in SYM_ORDER)


def run_sweep(Ns=(15, 30, 60), sigmas=(0.0, 0.002, 0.005), trials=4, seed=0):
    branch0 = trace_linkage(TRUTH, n=4000)[0]
    rng = np.random.default_rng(seed)

    results = {}
    t00 = time.time()
    for N in Ns:
        for sigma in sigmas:
            hits_top1 = 0
            hits_top3 = 0
            for t in range(trials):
                idx = rng.choice(len(branch0), size=N, replace=False)
                pts = branch0[idx].copy()
                if sigma > 0:
                    pts += rng.normal(scale=sigma, size=pts.shape)
                try:
                    cands = synthesize_from_points(pts, n_trace=500, verbose=False, max_nfev=150)
                except Exception:
                    cands = []
                if cands and matches_truth(cands[0]):
                    hits_top1 += 1
                if any(matches_truth(c) for c in cands[:3]):
                    hits_top3 += 1
            results[(N, sigma)] = (hits_top1 / trials, hits_top3 / trials)
            print(f"  N={N:>3} sigma={sigma:.3f}: "
                  f"top1_success={hits_top1}/{trials}  top3_success={hits_top3}/{trials}  "
                  f"[{time.time()-t00:.0f}s elapsed]", flush=True)
    return results


def print_grid(results, Ns, sigmas, which=0, label="top-1 success rate"):
    print(f"\n{label}:")
    print(f"{'N \\\\ sigma':>10}", *[f"{s:>8.3f}" for s in sigmas])
    for N in Ns:
        row = [results[(N, s)][which] for s in sigmas]
        print(f"{N:>10}", *[f"{v:>8.2f}" for v in row])


if __name__ == '__main__':
    Ns = (15, 30, 60)
    sigmas = (0.0, 0.002, 0.005)
    res = run_sweep(Ns=Ns, sigmas=sigmas, trials=4, seed=0)
    print_grid(res, Ns, sigmas, which=0, label="Top-1 success rate (recovers ground truth exactly)")
    print_grid(res, Ns, sigmas, which=1, label="Top-3 success rate (ground truth is among the top 3)")
