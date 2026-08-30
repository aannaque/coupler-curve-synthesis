"""
Extension #2 to Wu, Li & Bai (2021): bridging DISCRETE TASK POINTS to the
fully-analytical coupler-curve synthesis method of coupler_curve_synthesis.py.

The paper's method needs an exact 15-coefficient algebraic sextic as input.
Real design problems instead hand you a set of points the coupler point
should pass through/near. This module closes that gap in three stages:

  1. FIT   - the sextic is LINEAR in (k1..k15), so N>=15 sampled points give a
             standard linear least-squares fit for the unconstrained curve
             (no realizability constraint enforced yet).
  2. SEED  - feed that fitted curve straight into solve_case(): it returns up
             to six algebraically exact candidate linkages (three cognates x
             two assembly branches). This is the key benefit over blind
             optimization -- it supplies several genuinely different,
             globally-spread starting points instead of one arbitrary guess,
             so a subsequent local optimizer is far less likely to get stuck
             on the wrong branch.
  3. POLISH - each seed is refined by nonlinear least-squares against the
             TRUE geometric (point-to-curve) distance, not the algebraic
             residual used for the linear fit. This matters: minimizing
             algebraic distance on a degree-6 implicit curve is known to be
             severely ill-conditioned/biased (the classical Bookstein/Taubin
             problem in algebraic curve fitting), and step 1 alone is not a
             reliable estimator once there is any noise -- see
             demo_noisy_points() below, which shows the fit landing on a
             different (but comparably good-fitting) linkage than the
             ground truth once realistic point noise is added. This is an
             honest limitation worth stating, not hidden: sparse/noisy
             precision points do not, in general, pin down a unique
             four-bar linkage -- there can be several visibly different
             mechanisms that all reproduce the given points almost equally
             well. Returning the six-candidate spread (ranked by fit) makes
             that multiplicity visible instead of silently picking one.

Candidates are ranked by the true geometric RMS distance to the input
points (not by k1..k15 relation residuals, which will not be ~0 in general
here since fitted data rarely lies EXACTLY on a realizable curve).
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from coupler_curve_synthesis import solve_case
from mechanism_quality import classify_linkage, practical_score

SYM_ORDER = ('m', 'h', 'b1', 'b2', 'c1', 'c2', 'l2', 'l3', 'l4')


# --------------------------------------------------------------------------
# Stage 1: linear fit of k1..k15 from points (Gamma is LINEAR in the k_i)
# --------------------------------------------------------------------------
def fit_sextic_from_points(points):
    """points: (N,2) array of (x,y). Returns a {1:...,...,15:...} dict of the
    least-squares coefficients of the normalized CCE (leading x^6 coeff = 1).
    Needs N >= 15 for a well-posed fit (use more for noise averaging)."""
    points = np.asarray(points, dtype=float)
    if len(points) < 15:
        raise ValueError(f"need >=15 points for a well-posed linear fit, got {len(points)}")
    x, y = points[:, 0], points[:, 1]
    r2 = x * x + y * y
    M = np.column_stack([
        x * r2 ** 2, y * r2 ** 2, x * x * r2, x * y * r2, y * y * r2,
        x ** 3, x * x * y, x * y * y, y ** 3, x * x, x * y, y * y, x, y, np.ones_like(x),
    ])
    rhs = -(r2 ** 3)
    kfit, *_ = np.linalg.lstsq(M, rhs, rcond=None)
    return {i + 1: float(kfit[i]) for i in range(15)}


# --------------------------------------------------------------------------
# Stage 3 support: fast vectorized forward kinematics + geometric residual
# (a numpy re-implementation of coupler_curve_synthesis.trace_linkage, which
# uses a Python for-loop unsuitable for repeated calls inside an optimizer)
# --------------------------------------------------------------------------
def _trace_vectorized(p, n=800):
    """p = (m,h,b1,b2,c1,c2,l2,l3,l4). Returns (M,2) array of coupler-point
    samples over both assembly branches of the DC dyad, or None if the
    linkage cannot be assembled at all (circles never intersect)."""
    mv, hv, b1v, b2v, c1v, c2v, l2v, l3v, l4v = p
    beta = np.linspace(0, 2 * np.pi, n)
    Ax = b1v + l2v * np.cos(beta)
    Ay = b2v + l2v * np.sin(beta)
    vx, vy = c1v - Ax, c2v - Ay
    d = np.hypot(vx, vy)
    valid = (d > 1e-9) & (d <= l3v + l4v) & (d >= abs(l3v - l4v))
    if not np.any(valid):
        return None
    Ax, Ay, vx, vy, d = Ax[valid], Ay[valid], vx[valid], vy[valid], d[valid]
    a_ = (l3v ** 2 - l4v ** 2 + d ** 2) / (2 * d)
    h_sq = l3v ** 2 - a_ ** 2
    ok = h_sq >= 0
    if not np.any(ok):
        return None
    Ax, Ay, vx, vy, d, a_, h_sq = (arr[ok] for arr in (Ax, Ay, vx, vy, d, a_, h_sq))
    h_ = np.sqrt(h_sq)
    midx, midy = Ax + a_ * vx / d, Ay + a_ * vy / d
    perpx, perpy = -vy / d, vx / d
    out = []
    for sign in (1, -1):
        Dx, Dy = midx + sign * h_ * perpx, midy + sign * h_ * perpy
        theta = np.arctan2(Dy - Ay, Dx - Ax)
        ct, st = np.cos(theta), np.sin(theta)
        out.append(np.column_stack([Ax + ct * mv - st * hv, Ay + st * mv + ct * hv]))
    return np.vstack(out)


def geometric_residual(p, points, n_trace=800):
    """Vector of nearest-point-on-curve distances for each target point --
    the residual scipy.optimize.least_squares minimizes the sum of squares
    of. Returns a large constant residual (not a crash) if `p` cannot be
    assembled, so the optimizer can still take a gradient step away from it."""
    curve_pts = _trace_vectorized(p, n=n_trace)
    if curve_pts is None or len(curve_pts) < 3:
        return np.full(len(points), 1.0)
    dists, _ = cKDTree(curve_pts).query(points)
    return dists


# --------------------------------------------------------------------------
# Full pipeline
# --------------------------------------------------------------------------
def synthesize_from_points(points, n_trace=2000, verbose=True, max_nfev=300, bound_scale=6.0):
    """Fit -> six algebraic seeds -> geometric polish -> rank by fit RMS.
    Returns a list of dicts (sorted best-first), each with keys m,h,b1,b2,
    c1,c2,l2,l3,l4, plus 'rms' (geometric RMS distance to the input points)
    and 'converged' (whether least_squares reported success).

    The polish is bounded to +/-bound_scale x the input points' spatial
    extent (link lengths additionally clamped to be positive). Without this,
    an occasional degenerate/noisy seed lets the optimizer wander to
    absurdly large or near-zero link lengths -- numerically unstable and,
    empirically, dramatically slower to (not) converge -- for a candidate
    that would never be a sane design pick anyway."""
    points = np.asarray(points, dtype=float)
    scale = float(np.ptp(points, axis=0).max()) * bound_scale
    lo = np.array([-scale, -scale, -scale, -scale, -scale, -scale, 1e-4, 1e-4, 1e-4])
    hi = np.array([scale] * 6 + [scale, scale, scale])

    kstar = fit_sextic_from_points(points)
    seeds = solve_case(kstar, "precision-point fit")
    if verbose:
        print(f"  linear fit -> {len(seeds)} algebraic seed(s) from solve_case")

    results = []
    for i, seed in enumerate(seeds, 1):
        p0 = np.clip(np.array([float(seed[k]) for k in SYM_ORDER]), lo, hi)
        sol = least_squares(geometric_residual, p0, args=(points, n_trace),
                             method='trf', bounds=(lo, hi), max_nfev=max_nfev, diff_step=1e-6)
        rms = float(np.sqrt(2 * sol.cost / len(points)))
        entry = dict(zip(SYM_ORDER, sol.x))
        entry['rms'] = rms
        entry['converged'] = sol.success
        entry['quality'] = classify_linkage(entry)
        results.append(entry)
        if verbose:
            print(f"    seed {i}: geometric RMS after polish = {rms:.5f}   "
                  f"({entry['quality']['grashof']}, "
                  f"worst transmission dev={entry['quality']['worst_transmission_deviation_deg']})")

    results.sort(key=lambda r: r['rms'])
    return results


def rank_practical(results, rms_tolerance=1.5):
    """Among candidates whose RMS is within `rms_tolerance`x the best RMS
    (i.e. statistically about as good a fit), re-sort by practical_score so
    a real crank-rocker with a healthy transmission angle is preferred over
    an equally-fitting but unbuildable/poor-quality mechanism."""
    if not results:
        return results
    best_rms = results[0]['rms']
    tied = [r for r in results if r['rms'] <= best_rms * rms_tolerance]
    rest = [r for r in results if r['rms'] > best_rms * rms_tolerance]
    tied.sort(key=lambda r: practical_score(r['quality']))
    return tied + rest


def plot_synthesis(points, results, truth, top_n, fname, title):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(points[:, 0], points[:, 1], c='black', s=18, zorder=5, label='target points')

    truth_pts = _trace_vectorized([truth[k] for k in SYM_ORDER], n=2000)
    ax.plot(truth_pts[:, 0], truth_pts[:, 1], '-', color='gray', lw=4, alpha=0.35,
            zorder=1, label='ground-truth curve')

    colors = ['crimson', 'royalblue', 'seagreen', 'darkorange']
    for i, r in enumerate(results[:top_n]):
        p = [r[k] for k in SYM_ORDER]
        curve = _trace_vectorized(p, n=2000)
        if curve is None:
            continue
        ax.plot(curve[:, 0], curve[:, 1], '-', color=colors[i % len(colors)], lw=1.4,
                zorder=3, label=f"candidate {i+1} (rms={r['rms']:.4f})")

    ax.set_aspect('equal')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title(title)
    ax.legend(fontsize=8, loc='best')
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  saved {fname}")


def print_synthesis_table(results, name="synthesis result", top=None):
    print(f"--- {name}: ranked candidates (best fit first) ---")
    header = f"{'#':>2} {'rms':>10} " + " ".join(f"{k:>8}" for k in SYM_ORDER) + \
             f"  {'crank?':>7} {'trans.dev':>10}  grashof"
    print(header)
    for i, r in enumerate(results[:top] if top else results, 1):
        q = r.get('quality') or classify_linkage(r)
        print(f"{i:>2} {r['rms']:>10.5f} " + " ".join(f"{r[k]:>8.4f}" for k in SYM_ORDER) +
              f"  {str(q['l2_is_crank']):>7} {str(q['worst_transmission_deviation_deg']):>10}  {q['grashof']}")
    print()


# --------------------------------------------------------------------------
# Demonstrations
# --------------------------------------------------------------------------
def demo_exact_points():
    """No noise: confirms the pipeline recovers the ground-truth linkage
    (up to cognate/branch ambiguity) essentially exactly."""
    from coupler_curve_synthesis import trace_linkage
    print("=== Demo A: exact (noise-free) points sampled from a known linkage ===")
    truth = dict(m=0.12, h=0.22, b1=0.05, b2=-0.03, c1=0.31, c2=0.02, l2=0.28, l3=0.33, l4=0.19)
    branch0 = trace_linkage(truth, n=4000)[0]
    rng = np.random.default_rng(7)
    pts = branch0[rng.choice(len(branch0), size=25, replace=False)]

    results = synthesize_from_points(pts)
    print_synthesis_table(results, "Demo A")
    print("Ground truth:", {k: truth[k] for k in SYM_ORDER})
    best = results[0]
    err = max(abs(best[k] - truth[k]) for k in SYM_ORDER)
    print(f"Best candidate max-parameter error vs ground truth: {err:.4f}  (rms fit={best['rms']:.2e})\n")

    practical = rank_practical(results, rms_tolerance=1.2)
    if practical[0] is not results[0]:
        print("Practicality re-rank changes the top pick: the best-FIT candidate "
              f"is not a true crank ({results[0]['quality']['grashof']}), but a "
              f"nearly-as-good fit is ({practical[0]['quality']['grashof']}, "
              f"rms={practical[0]['rms']:.5f} vs {results[0]['rms']:.5f}).")
        print("Whether that trade is worth it depends on whether the design truly")
        print("needs l2 to be a continuously-driven crank.\n")

    plot_synthesis(pts, results, truth, top_n=1, fname='demoA_exact_points.png',
                    title='Demo A: exact points -> near-perfect recovery')


def demo_noisy_points():
    """Realistic case: modest point noise. Demonstrates that several visibly
    DIFFERENT linkages can fit the same sparse, noisy point set almost
    equally well -- an honest limitation of precision-point synthesis in
    general, not specific to this method. Ranking by geometric RMS still
    picks a linkage that reproduces the data about as well as the noise
    floor allows."""
    from coupler_curve_synthesis import trace_linkage
    print("=== Demo B: noisy points (more realistic measurement scenario) ===")
    truth = dict(m=0.12, h=0.22, b1=0.05, b2=-0.03, c1=0.31, c2=0.02, l2=0.28, l3=0.33, l4=0.19)
    branch0 = trace_linkage(truth, n=4000)[0]
    rng = np.random.default_rng(42)
    idx = rng.choice(len(branch0), size=20, replace=False)
    noise_level = 0.003
    pts = branch0[idx] + rng.normal(scale=noise_level, size=(20, 2))

    results = synthesize_from_points(pts)
    print_synthesis_table(results, "Demo B -- ranked by fit (RMS) alone")
    print(f"Point noise std was {noise_level}; best candidate RMS = {results[0]['rms']:.5f}")
    print("Ground truth:", {k: truth[k] for k in SYM_ORDER})
    print("Note how the top few candidates can have similar RMS while looking")
    print("like visibly different mechanisms -- expected with only 20 noisy points.\n")

    practical = rank_practical(results, rms_tolerance=1.7)
    print_synthesis_table(practical, "Demo B -- re-ranked: fit-tied candidates sorted by practicality")
    print("Among candidates within 1.7x the best RMS, this prefers an actual")
    print("crank-rocker with a healthy transmission angle over a fit-equivalent")
    print("mechanism that isn't a true crank or is near a jamming configuration.\n")

    plot_synthesis(pts, results, truth, top_n=3, fname='demoB_noisy_points.png',
                    title='Demo B: noisy points -> several distinct near-equal fits')


if __name__ == '__main__':
    demo_exact_points()
    demo_noisy_points()
