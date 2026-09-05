"""
Reproduces Table 1 of the paper (Section 5.2): how many points / how much
noise does the exact-seeded fit pipeline (precision_point_synthesis.py)
actually need before the recovered linkage reliably matches the ground
truth, and how much of that reliability is specifically due to (a) seeding
from the exact algebraic solver rather than at random, and (b) enumerating
all six cognates rather than taking just one?

For a fixed ground-truth linkage, swept over:
  - N     = number of sampled task points   in {15, 30, 60}
  - sigma = point coordinate noise std      in {0, 0.002, 0.005}
three success rates are tracked per (N, sigma) cell:
  - single_exact : a single, arbitrary (not cherry-picked) exact-solver
                   seed, polished
  - best_exact   : the best (lowest geometric-RMS) of all six exact-solver
                   seeds, polished  -- this is what solve_from_points()/
                   solve_case() returns and is the method of Section 4
  - best_blind   : the best of six candidates seeded UNIFORMLY AT RANDOM
                   within the point cloud's bounding box, polished by the
                   identical procedure -- a size-matched baseline

Table 1 pools two independent runs (different RNG seeds) of ten trials
per cell into n=20 for tighter statistics, and reports 95% Clopper-Pearson
confidence intervals rather than point estimates alone. The same pipeline
is then repeated at N=30 for two further linkages of different Grashof
type (a double-crank and a double-rocker) to check the pattern is not an
artifact of the one linkage used throughout the rest of the paper.

This is slow: each (N, sigma, linkage) cell costs roughly 10-25s per trial
(dominated by the exact solver's resultant-based elimination), so the
full run (13 cells x 20 trials = 260 exact-method calls, plus a matching
number of blind-baseline calls) takes on the order of an hour. Reduce
TRIALS_PER_RUN below for a quicker, noisier check.
"""
import contextlib
import io
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import beta

from coupler_curve_synthesis import trace_linkage, solve_case
from precision_point_synthesis import SYM_ORDER, geometric_residual

LINKAGES = {
    'truth1_crankrocker_l4': dict(m=0.12, h=0.22, b1=0.05, b2=-0.03, c1=0.31, c2=0.02,
                                   l2=0.28, l3=0.33, l4=0.19),
    'truth3_doublecrank': dict(m=0.12, h=0.10, b1=0.10, b2=0.0, c1=0.13, c2=0.0,
                                l2=0.25, l3=0.30, l4=0.25),
    'truth4_doublerocker': dict(m=0.05, h=0.05, b1=0.05, b2=-0.03, c1=0.31, c2=0.02,
                                 l2=0.28, l3=0.10, l4=0.19),
}
MATCH_TOL = 0.02
TRIALS_PER_RUN = 10  # two runs of this size are pooled -> n = 2*TRIALS_PER_RUN


# --------------------------------------------------------------------------
# Fitting / seeding / polishing (self-contained here, mirroring
# precision_point_synthesis.py, so this script has no other side effects on
# the shared bound_scale/n_trace defaults used by the demo figures)
# --------------------------------------------------------------------------
def design_matrix(points):
    x, y = points[:, 0], points[:, 1]
    r2 = x * x + y * y
    return np.column_stack([x * r2 ** 2, y * r2 ** 2, x * x * r2, x * y * r2, y * y * r2,
                             x ** 3, x * x * y, x * y * y, y ** 3, x * x, x * y, y * y, x, y,
                             np.ones_like(x)])


def fit_sextic(points):
    M = design_matrix(points)
    rhs = -(points[:, 0] ** 2 + points[:, 1] ** 2) ** 3
    kfit, *_ = np.linalg.lstsq(M, rhs, rcond=None)
    return {i + 1: float(kfit[i]) for i in range(15)}


def polish(p0, points, n_trace=300, max_nfev=120, bound_scale=6.0):
    p0 = np.asarray(p0, dtype=float)
    scale = float(np.ptp(points, axis=0).max()) * bound_scale
    lo = np.array([-scale] * 6 + [1e-4] * 3)
    hi = np.array([scale] * 9)
    p0c = np.clip(p0, lo, hi)
    sol = least_squares(geometric_residual, p0c, args=(points, n_trace), method='trf',
                         bounds=(lo, hi), max_nfev=max_nfev, diff_step=1e-6)
    rms = float(np.sqrt(2 * sol.cost / len(points)))
    e = dict(zip(SYM_ORDER, sol.x))
    e['rms'] = rms
    return e


def exact_seeds_polished(points):
    """All six exact-solver seeds, polished, in solve_case's own (arbitrary)
    order -- i.e. NOT sorted by fit quality, so seeds[0] is a fair stand-in
    for "a single exact seed, not cherry-picked"."""
    kstar = fit_sextic(points)
    with contextlib.redirect_stdout(io.StringIO()):
        seeds = solve_case(kstar, "sweep")
    out = []
    for s in seeds:
        try:
            out.append(polish([s[k] for k in SYM_ORDER], points))
        except Exception:
            continue
    return out


def blind_seeds_polished(points, rng, k=6):
    extent = float(np.ptp(points, axis=0).max())
    lo_xy = points.min(axis=0)
    hi_xy = points.max(axis=0)
    out = []
    for _ in range(k):
        p0 = [rng.uniform(-extent, extent), rng.uniform(-extent, extent),
              rng.uniform(lo_xy[0], hi_xy[0]), rng.uniform(lo_xy[1], hi_xy[1]),
              rng.uniform(lo_xy[0], hi_xy[0]), rng.uniform(lo_xy[1], hi_xy[1]),
              rng.uniform(0.05 * extent, 1.5 * extent),
              rng.uniform(0.05 * extent, 1.5 * extent),
              rng.uniform(0.05 * extent, 1.5 * extent)]
        try:
            out.append(polish(p0, points))
        except Exception:
            continue
    return out


def matches(entry, truth, tol=MATCH_TOL):
    return entry is not None and all(abs(entry[k] - truth[k]) <= tol for k in SYM_ORDER)


# --------------------------------------------------------------------------
# Sweep
# --------------------------------------------------------------------------
def run_cell(truth, N, sigma, trials, rng, branch0):
    hits_single, hits_best_exact, hits_best_blind = 0, 0, 0
    for _ in range(trials):
        idx = rng.choice(len(branch0), size=N, replace=False)
        pts = branch0[idx].copy()
        if sigma > 0:
            pts += rng.normal(scale=sigma, size=pts.shape)

        ex_cands = exact_seeds_polished(pts)
        bl_cands = blind_seeds_polished(pts, rng)

        single = ex_cands[0] if ex_cands else None
        best_ex = min(ex_cands, key=lambda e: e['rms']) if ex_cands else None
        best_bl = min(bl_cands, key=lambda e: e['rms']) if bl_cands else None

        hits_single += int(matches(single, truth))
        hits_best_exact += int(matches(best_ex, truth))
        hits_best_blind += int(matches(best_bl, truth))
    return hits_single, hits_best_exact, hits_best_blind


def clopper_pearson(x, n, conf=0.95):
    alpha = 1 - conf
    lo = 0.0 if x == 0 else beta.ppf(alpha / 2, x, n - x + 1)
    hi = 1.0 if x == n else beta.ppf(1 - alpha / 2, x + 1, n - x)
    return lo, hi


def two_pooled_runs(truth, N, sigma, branch0, seed_a, seed_b):
    """Two independent TRIALS_PER_RUN-trial runs, pooled to n=2*TRIALS_PER_RUN."""
    s1, e1, b1 = run_cell(truth, N, sigma, TRIALS_PER_RUN, np.random.default_rng(seed_a), branch0)
    s2, e2, b2 = run_cell(truth, N, sigma, TRIALS_PER_RUN, np.random.default_rng(seed_b), branch0)
    n = 2 * TRIALS_PER_RUN
    return dict(n=n, single=s1 + s2, best_exact=e1 + e2, best_blind=b1 + b2)


def main():
    t00 = time.time()

    print("=== Primary sweep: truth1 (crank-rocker), full N x sigma grid ===", flush=True)
    truth = LINKAGES['truth1_crankrocker_l4']
    branch0 = trace_linkage(truth, n=4000)[0]
    primary = {}
    for N in (15, 30, 60):
        for sigma in (0.0, 0.002, 0.005):
            t0 = time.time()
            r = two_pooled_runs(truth, N, sigma, branch0, seed_a=100 + N, seed_b=500 + N)
            primary[(N, sigma)] = r
            print(f"  N={N:>3} sigma={sigma:.3f}: single={r['single']}/{r['n']} "
                  f"best_exact={r['best_exact']}/{r['n']} best_blind={r['best_blind']}/{r['n']} "
                  f"[{time.time()-t0:.0f}s, total {time.time()-t00:.0f}s]", flush=True)

    print("\n=== Generality check: 2 more linkages (different Grashof type), N=30 ===", flush=True)
    generality = {}
    for name in ('truth3_doublecrank', 'truth4_doublerocker'):
        t_link = LINKAGES[name]
        branches = trace_linkage(t_link, n=4000)
        b0 = max(branches.values(), key=len)
        for sigma in (0.0, 0.002):
            t0 = time.time()
            r = two_pooled_runs(t_link, 30, sigma, b0, seed_a=200, seed_b=600)
            generality[(name, sigma)] = r
            print(f"  {name} sigma={sigma:.3f}: single={r['single']}/{r['n']} "
                  f"best_exact={r['best_exact']}/{r['n']} best_blind={r['best_blind']}/{r['n']} "
                  f"[{time.time()-t0:.0f}s, total {time.time()-t00:.0f}s]", flush=True)

    print("\n=== SUMMARY (Table 1 + text figures of Section 5.2) ===")
    print(f"{'cell':>28}  {'single':>16}  {'best_exact':>18}  {'best_blind':>16}")
    for k, r in {**primary, **generality}.items():
        n = r['n']
        s_lo, s_hi = clopper_pearson(r['single'], n)
        e_lo, e_hi = clopper_pearson(r['best_exact'], n)
        b_lo, b_hi = clopper_pearson(r['best_blind'], n)
        print(f"{str(k):>28}  "
              f"{r['single']}/{n}={r['single']/n:.2f} CI[{s_lo:.2f},{s_hi:.2f}]  "
              f"{r['best_exact']}/{n}={r['best_exact']/n:.2f} CI[{e_lo:.2f},{e_hi:.2f}]  "
              f"{r['best_blind']}/{n}={r['best_blind']/n:.2f} CI[{b_lo:.2f},{b_hi:.2f}]")

    total_single = sum(r['single'] for r in {**primary, **generality}.values())
    total_n = sum(r['n'] for r in {**primary, **generality}.values())
    lo, hi = clopper_pearson(total_single, total_n)
    print(f"\nSingle-exact-seed, pooled across ALL cells: {total_single}/{total_n} "
          f"CI[{lo:.3f},{hi:.3f}]")
    print(f"Total wall time: {time.time()-t00:.0f}s")


if __name__ == '__main__':
    main()
