"""
Regeneration of the results of:

  R. Wu, R. Li, S. Bai, "A fully analytical method for coupler-curve synthesis
  of planar four-bar linkages", Mechanism and Machine Theory 155 (2021) 104070.

The script re-derives, from first principles (rigid-body loop-closure of the
two dyads AB and DC, eliminating the coupler orientation theta via Cramer's
rule + cos^2+sin^2=1), the coupler-curve equation (CCE) coefficients K_0..K_15
of Eq. (3a)-(3b) of the paper as *symbolic* functions of the nine linkage
parameters (m, h, b1, b2, c1, c2, l2, l3, l4).

It then follows the paper's fully-analytical synthesis procedure (Section 3,
Fig. 2):
  1. Solve Eqs. (7a)-(7d) [given explicitly in the paper, general in k1*..k9*]
     for the pivot coordinates b1, b2, c1, c2  (six solutions -> 3 cognates x
     2 assembly modes).
  2. Solve Eqs. (8a)-(8b) [also given explicitly] for m, h as linear
     functions of l3.
  3. Use the (independently re-derived here) combinations k3+k5, k6+k9,
     k10-k12 -- which the paper states are linear in l2^2 and l4^2 (property 2)
     but does not spell out algebraically -- to solve for l2, l3, l4.
  4. Back-substitute to get the final m, h.

Case 1 (Eq. 13) and Case 2 (Eq. 16) of the paper are reproduced, matching
Tables 1-4. Coupler-curve tracing (forward kinematics) reproduces Figs. 3-5.
"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction

x, y = sp.symbols('x y')
m, h, b1, b2, c1, c2, l2, l3, l4 = sp.symbols('m h b1 b2 c1 c2 l2 l3 l4', real=True)
b1_, b2_, c1_, c2_ = sp.symbols('b1_ b2_ c1_ c2_', real=True)
k1s, k2s, k3s, k4s, k5s, k6s, k7s, k8s, k9s = sp.symbols(
    'k1s k2s k3s k4s k5s k6s k7s k8s k9s')


# --------------------------------------------------------------------------
# 1. Symbolic derivation of the coupler-curve equation coefficients K_i
# --------------------------------------------------------------------------
def derive_cce_coefficients():
    """Eliminate theta from the two rigid-dyad constraints

        |r + Q(theta) a' - b|^2 = l2^2      (link AB)
        |r + Q(theta) d' - c|^2 = l4^2      (link DC)

    where a' = (-m,-h), d' = (l3-m,-h) are the local coordinates of A, D in
    the coupler frame P-xy, r=(x,y) is the coupler point and Q is the
    rotation matrix. Writing each constraint as Lc*cos(theta)+Ls*sin(theta)
    = const (Pythagorean trick, since the cos^2/sin^2 terms combine to a
    constant), Cramer's rule gives cos(theta), sin(theta) as rational
    functions of x,y; substituting into cos^2+sin^2=1 and clearing
    denominators yields the sextic CCE directly (degree 6, matching Eq. 3a).
    """
    def Lc_Ls(px, py, p1, p2):
        Lc = 2 * px * (x - p1) + 2 * py * (y - p2)
        Ls = 2 * px * (y - p2) - 2 * py * (x - p1)
        return Lc, Ls

    Ax, Ay = -m, -h
    Dx, Dy = l3 - m, -h

    R2 = (x - b1) ** 2 + (y - b2) ** 2 + Ax ** 2 + Ay ** 2
    L2c, L2s = Lc_Ls(Ax, Ay, b1, b2)
    R4 = (x - c1) ** 2 + (y - c2) ** 2 + Dx ** 2 + Dy ** 2
    L4c, L4s = Lc_Ls(Dx, Dy, c1, c2)

    Delta = sp.expand(L2c * L4s - L2s * L4c)
    NumC = sp.expand((l2 ** 2 - R2) * L4s - (l4 ** 2 - R4) * L2s)
    NumS = sp.expand(L2c * (l4 ** 2 - R4) - L4c * (l2 ** 2 - R2))
    CCE = sp.expand(NumC ** 2 + NumS ** 2 - Delta ** 2)

    P = sp.Poly(CCE, x, y)

    def coeff(px, py):
        return P.coeff_monomial(x ** px * y ** py) if (px or py) else P.coeff_monomial(1)

    mono = {0: (6, 0), 1: (5, 0), 2: (0, 5), 3: (4, 0), 4: (3, 1), 5: (0, 4),
            6: (3, 0), 7: (2, 1), 8: (1, 2), 9: (0, 3), 10: (2, 0), 11: (1, 1),
            12: (0, 2), 13: (1, 0), 14: (0, 1), 15: (0, 0)}
    K = {i: coeff(*mono[i]) for i in mono}
    return K


print("Deriving the coupler-curve equation symbolically (Eq. 3a-3b)...")
K = derive_cce_coefficients()

# The three combinations that fix l2, l3, l4 (paper's "property 2"), obtained
# here from the coefficients above rather than taken from the paper (which
# only states they exist, without giving the closed form).
expr_k3pk5 = sp.expand(sp.cancel((K[3] + K[5]) / K[0]))
expr_k6pk9 = sp.expand(sp.cancel((K[6] + K[9]) / K[0]))
expr_k10mk12 = sp.expand(sp.cancel((K[10] - K[12]) / K[0]))

# sanity check against the paper's Eq. (13) / Table 2 row 1 values
_check = {m: sp.Rational(1, 10), h: sp.Rational(15, 100), b1: sp.Rational(-2, 10),
          b2: 0, c1: sp.Rational(2, 10), c2: sp.Rational(-2, 10),
          l2: sp.Rational(15, 100), l3: sp.Rational(4, 10), l4: sp.Rational(35, 100)}
_k1 = float((K[1] / K[0]).subs(_check))
assert abs(_k1 - 0.05) < 1e-9, "self-check failed: symbolic CCE derivation is wrong"
print("Self-check passed (k1 = 0.05 reproduced from Table 2, row 1).\n")


# --------------------------------------------------------------------------
# 2. Paper's Eqs. (7a)-(7d): position parameters b1,b2,c1,c2 from k1*..k9*
# --------------------------------------------------------------------------
def eqs_7a_7d():
    eq7a = (-4 * b1 ** 2 - 4 * b1 * c1 + 4 * b2 ** 2 + 4 * b2 * c2 - 4 * c1 ** 2
            + 4 * c2 ** 2 + 2 * k2s * b2 - 2 * k1s * b1 - 2 * k1s * c1
            + 2 * k2s * c2 - k3s + k5s)
    eq7b = (-2 * k1s * b2 - 2 * k2s * b1 - 2 * k1s * c2 - k4s - 2 * k2s * c1
            - 8 * b1 * b2 - 4 * b1 * c2 - 4 * b2 * c1 - 8 * c1 * c2)
    eq7c = (8 * c1 ** 3 - 24 * c1 * c2 ** 2 - 8 * k2s * c1 * c2 - 4 * k1s * c2 ** 2
            + 4 * k1s * c1 ** 2 + 2 * k3s * c1 - 2 * k5s * c1 - 2 * k4s * c2
            - k8s + k6s)
    eq7d = (8 * c2 ** 3 - 24 * c1 ** 2 * c2 - 8 * k1s * c1 * c2 - 4 * k2s * c1 ** 2
            + 4 * k2s * c2 ** 2 - 2 * k4s * c1 - 2 * k3s * c2 + 2 * k5s * c2
            - k7s + k9s)
    return eq7a, eq7b, eq7c, eq7d


def eqs_8a_8b():
    eq8a = (2 * b1 * m - 2 * b2 * h - 2 * c1 * m + 2 * c2 * h - 4 * b1 * l3
            - 2 * c1 * l3 - k1s * l3)
    eq8b = (2 * b1 * h + 2 * b2 * m - 2 * c1 * h - 2 * c2 * m - 4 * b2 * l3
            - 2 * c2 * l3 - k2s * l3)
    return eq8a, eq8b


def _to_exact(v):
    """Coerce a python float/int/sympy number to an exact Rational so that
    downstream symbolic elimination (resultants) runs in QQ, not RR -- sympy's
    Euclidean-algorithm-based routines can fail to detect zero and raise
    PolynomialDivisionFailed when working over inexact (float) coefficients.
    The denominator is capped: an *exact* decimal string (e.g. from a
    Fraction/Rational) can have a huge denominator once many such values are
    combined by the resultant, making the exact arithmetic extremely slow;
    limit_denominator keeps this fast while staying far below float noise."""
    if isinstance(v, (sp.Rational, sp.Integer)):
        return v
    return sp.Rational(Fraction(float(v)).limit_denominator(10 ** 6))


def _real_roots_of_univariate(expr, sym, tol=1e-6):
    """Coefficients of a univariate polynomial -> real roots via numpy.
    This never invokes sympy's resultant/gcd machinery, so it is safe even
    when `expr` has floating-point coefficients (which happens once numeric
    values have been substituted for the other variables)."""
    P = sp.Poly(sp.expand(expr), sym)
    coeffs = [complex(c) for c in P.all_coeffs()]
    if len(coeffs) < 2:
        return []
    return [float(z.real) for z in np.roots(coeffs) if abs(z.imag) < tol]


def _dedup(values, key=lambda v: v, tol=1e-6):
    """Collapse near-duplicate entries (e.g. a resultant root of multiplicity
    2, which numpy.roots legitimately returns twice)."""
    out = []
    for v in values:
        if not any(all(abs(a - b) < tol for a, b in zip(key(v), key(o))) for o in out):
            out.append(v)
    return out


def _solve_position_system(eq7a_n, eq7b_n, eq7c_n, eq7d_n):
    """Solve Eqs. (7a)-(7d) for (b1,b2,c1,c2) robustly for ANY numeric k1*..k9*
    (not just the paper's hand-picked "nice" fractions), by combining exact
    resultant elimination (while coefficients are still exact rationals) with
    plain numpy root-finding for the univariate reductions (safe regardless
    of numeric domain). sympy's solve()/solve_poly_system() were found to be
    unreliable (silently returning no solutions, or raising
    PolynomialDivisionFailed) on "generic" floating-point coefficients."""
    # --- c1,c2 from the bivariate pair (7c)-(7d): eliminate c2 via an exact
    #     resultant (both equations are still purely symbolic/rational here) ---
    R = sp.resultant(sp.expand(eq7c_n), sp.expand(eq7d_n), c2)
    c1_candidates = _dedup(_real_roots_of_univariate(R, c1), key=lambda v: (v,))

    cd_solutions = []
    for cc1 in c1_candidates:
        eqc = eq7c_n.subs(c1, cc1)  # quadratic in c2 -> direct root-find, no resultant needed
        eqd = eq7d_n.subs(c1, cc1)
        for cc2 in _real_roots_of_univariate(eqc, c2):
            if abs(complex(eqd.subs(c2, cc2))) < 1e-4:
                cd_solutions.append((cc1, cc2))
    cd_solutions = _dedup(cd_solutions)

    # --- b1,b2 given c1,c2: (7b) is affine (linear) in b2, so isolate it and
    #     substitute into (7a) -- pure substitution, never resultant/gcd.
    #     (Near a point where the b2-coefficient of (7b) vanishes, this
    #     division amplifies noise into a spurious large root, so every
    #     candidate is re-checked against the ORIGINAL equations below.) ---
    results = []
    for (cc1, cc2) in cd_solutions:
        eq7a_cd = eq7a_n.subs({c1: cc1, c2: cc2})
        eq7b_cd = eq7b_n.subs({c1: cc1, c2: cc2})
        P_b2 = sp.Poly(eq7b_cd, b2)
        a_, c_ = P_b2.all_coeffs()  # eq7b_cd = a_*b2 + c_  (a_,c_ depend on b1)
        b2_of_b1 = sp.cancel(-c_ / a_)
        num, _den = sp.fraction(sp.together(eq7a_cd.subs(b2, b2_of_b1)))
        for bb1 in _dedup(_real_roots_of_univariate(num, b1), key=lambda v: (v,)):
            bb2 = complex(b2_of_b1.subs(b1, bb1))
            if abs(bb2.imag) < 1e-6:
                results.append({'b1': bb1, 'b2': bb2.real, 'c1': cc1, 'c2': cc2})

    # Discard artifacts of the division above (spurious roots near a
    # vanishing denominator) by re-checking against the original equations,
    # then drop any remaining near-duplicates.
    checked = []
    for r in results:
        subs_all = {b1: r['b1'], b2: r['b2'], c1: r['c1'], c2: r['c2']}
        if abs(complex(eq7a_n.subs(subs_all))) < 1e-6 and abs(complex(eq7b_n.subs(subs_all))) < 1e-6:
            checked.append(r)
    return _dedup(checked, key=lambda r: (r['b1'], r['b2'], r['c1'], r['c2']))


def solve_case(kstar, name):
    """Full pipeline: kstar -> six sets of (m,h,b1,b2,c1,c2,l2,l3,l4). Works
    for ANY given CCE coefficients k1*..k15* (not just the paper's two
    worked examples)."""
    print(f"=== {name} ===")
    kstar = {i: _to_exact(v) for i, v in kstar.items()}
    subs_k = {k1s: kstar[1], k2s: kstar[2], k3s: kstar[3], k4s: kstar[4],
              k5s: kstar[5], k6s: kstar[6], k7s: kstar[7], k8s: kstar[8],
              k9s: kstar[9]}

    eq7a, eq7b, eq7c, eq7d = eqs_7a_7d()
    eq7a_n, eq7b_n, eq7c_n, eq7d_n = (e.subs(subs_k) for e in (eq7a, eq7b, eq7c, eq7d))

    results = _solve_position_system(eq7a_n, eq7b_n, eq7c_n, eq7d_n)
    print(f"  -> {len(results)} (b1,b2,c1,c2) solutions (should be 6)\n")

    # Step 2: m,h as linear functions of l3 (Eqs 8a-8b). Solved with plain
    # numpy linear algebra, not sp.solve: both are genuinely linear (2x2), so
    # this is exact up to float precision and -- unlike sympy's symbolic
    # solve -- cannot hang on a near-singular/degenerate candidate (which
    # does happen when solve_case is fed noisy, non-"nice" fitted
    # coefficients, e.g. from precision_point_synthesis.py).
    eq8a, eq8b = eqs_8a_8b()
    eq8a_n = eq8a.subs(subs_k)
    eq8b_n = eq8b.subs(subs_k)

    for r in results:
        subs_bc = {b1: r['b1'], b2: r['b2'], c1: r['c1'], c2: r['c2']}
        a = sp.expand(eq8a_n.subs(subs_bc))
        bnd = sp.expand(eq8b_n.subs(subs_bc))
        Pa, Pb = sp.Poly(a, m, h, l3), sp.Poly(bnd, m, h, l3)
        am, ah, al = (float(Pa.coeff_monomial(v)) for v in (m, h, l3))
        bm, bh, bl = (float(Pb.coeff_monomial(v)) for v in (m, h, l3))
        alpha, beta = np.linalg.solve(np.array([[am, ah], [bm, bh]]), np.array([-al, -bl]))
        r['alpha'] = float(alpha)  # m = alpha*l3
        r['beta'] = float(beta)  # h = beta*l3

    # Step 3: l2,l3,l4 from k3+k5, k6+k9, k10-k12 (linear in l2^2,l3^2,l4^2).
    # Also solved with numpy for the same reason.
    L2, L3, L4 = sp.symbols('L2 L3 L4')
    target_A = float(kstar[3] + kstar[5])
    target_B = float(kstar[6] + kstar[9])
    target_C = float(kstar[10] - kstar[12])

    for r in results:
        subs_bcmh = {b1: r['b1'], b2: r['b2'], c1: r['c1'], c2: r['c2'],
                     m: r['alpha'] * l3, h: r['beta'] * l3}
        row, const = [], []
        for expr, target in ((expr_k3pk5, target_A), (expr_k6pk9, target_B), (expr_k10mk12, target_C)):
            e = sp.expand(expr.subs(subs_bcmh)).subs({l2 ** 2: L2, l3 ** 2: L3, l4 ** 2: L4})
            P = sp.Poly(e, L2, L3, L4)
            row.append([float(P.coeff_monomial(v)) for v in (L2, L3, L4)])
            const.append(target - float(P.coeff_monomial(1)))
        L2v, L3v, L4v = np.linalg.solve(np.array(row), np.array(const))
        if min(L2v, L3v, L4v) < 0:
            r['_invalid'] = True
            continue
        r['l2'], r['l3'], r['l4'] = float(np.sqrt(L2v)), float(np.sqrt(L3v)), float(np.sqrt(L4v))
        r['m'] = r['alpha'] * r['l3']
        r['h'] = r['beta'] * r['l3']

    return [r for r in results if not r.get('_invalid')]


def print_table(results, name):
    print(f"--- {name}: final linkage parameters (m,h,b1,b2,c1,c2,l2,l3,l4) ---")
    header = f"{'No.':>3} {'m':>9} {'h':>9} {'b1':>8} {'b2':>8} {'c1':>8} {'c2':>8} {'l2':>8} {'l3':>8} {'l4':>8}"
    print(header)
    for i, r in enumerate(results, 1):
        vals = [float(r[k]) for k in ('m', 'h', 'b1', 'b2', 'c1', 'c2', 'l2', 'l3', 'l4')]
        print(f"{i:>3} " + " ".join(f"{v:>9.4f}" if k in ('m', 'h') else f"{v:>8.4f}"
                                     for k, v in zip(('m', 'h', 'b1', 'b2', 'c1', 'c2', 'l2', 'l3', 'l4'), vals)))
    print()


# --------------------------------------------------------------------------
# 2b. Redundancy analysis: the 15 CCE coefficients k1..k15 come from only 9
#     free linkage parameters, so a genuine coupler curve cannot specify all
#     15 independently. A numeric rank check of the Jacobian of
#     (m,h,b1,b2,c1,c2,l2,l3,l4) -> (k1,...,k15) is exactly 9 everywhere
#     generic (see the development notes) -- i.e. valid coefficient-vectors
#     form a smooth 9-dimensional variety inside the 15-dimensional space of
#     all such sextic coefficients, so EXACTLY 6 independent algebraic
#     relations must hold among k1..k15. The paper's own algorithm only ever
#     consumes k1..k10 and k12 (11 numbers) to pin the 9 parameters, so:
#       - 4 of the 6 relations are simply "k11, k13, k14, k15 are functions
#         of k1..k10,k12" (the paper notes these 4 are unused, but never
#         states -- or exploits -- that they are therefore *determined*);
#       - the other 2 are hiding inside k1..k10,k12 itself: k7+k8 and
#         k10+k12 are EXTRA combinations the paper never forms (it only uses
#         k8-k6, k9-k7, k4 from {k6,k7,k8,k9}, and only k10-k12 from
#         {k10,k12}), and both come out fully determined too, verified below
#         to machine precision against six independent cognate solutions.
#     All 6 are evaluated the same way: solve for the 9 parameters from
#     k1..k10,k12 (as solve_case already does), then plug back into the
#     *other* combinations of K_i/K_0 and compare to the given targets.
# --------------------------------------------------------------------------
_VALIDATION_ARGS = (m, h, b1, b2, c1, c2, l2, l3, l4)
_VALIDATION_EXPRS = {
    'k11': K[11] / K[0], 'k13': K[13] / K[0], 'k14': K[14] / K[0], 'k15': K[15] / K[0],
    'k7+k8': sp.cancel((K[7] + K[8]) / K[0]), 'k10+k12': sp.cancel((K[10] + K[12]) / K[0]),
}
_VALIDATION_FUNCS = {name: sp.lambdify(_VALIDATION_ARGS, e, 'numpy')
                      for name, e in _VALIDATION_EXPRS.items()}


def validate_coupler_curve(kstar, result):
    """Given one solved linkage `result` (a dict from solve_case/
    solve_from_polynomial) and the original `kstar` dict of 15 target
    coefficients, evaluate the 6 combinations NOT used by the solve
    (k11, k13, k14, k15, k7+k8, k10+k12) and return their residuals against
    the given targets. All ~0 confirms kstar is a genuine four-bar coupler
    curve; a non-negligible residual means it is not (no four-bar linkage
    traces that curve exactly -- e.g. it was fit/approximated, or corrupted)."""
    vals = tuple(float(result[str(s)]) for s in _VALIDATION_ARGS)
    targets = {'k11': kstar[11], 'k13': kstar[13], 'k14': kstar[14], 'k15': kstar[15],
               'k7+k8': kstar[7] + kstar[8], 'k10+k12': kstar[10] + kstar[12]}
    return {name: _VALIDATION_FUNCS[name](*vals) - float(targets[name]) for name in targets}


def solve_from_polynomial(k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15,
                           name="user polynomial"):
    """Convenience entry point: given the 15 coefficients of a coupler curve
    already normalized to the form

        (x^2+y^2)^3 + (k1 x+k2 y)(x^2+y^2)^2 + (k3 x^2+k4 xy+k5 y^2)(x^2+y^2)
        + k6 x^3+k7 x^2y+k8 xy^2+k9 y^3 + k10 x^2+k11 xy+k12 y^2
        + k13 x+k14 y+k15 = 0

    (i.e. the coefficient of x^6 is 1 -- divide your raw polynomial through by
    that coefficient first if it isn't), return the list of up to six
    dictionaries {m,h,b1,b2,c1,c2,l2,l3,l4} of four-bar linkages that trace
    it. k11, k13, k14, k15 -- and, less obviously, k7+k8 and k10+k12 -- are
    NOT used by the solve (see the redundancy-analysis note above) but are
    returned alongside a residual check (all 6, not just the 4 the paper
    points out) so you can confirm the given polynomial is actually a
    genuine coupler curve.
    """
    kstar = {1: k1, 2: k2, 3: k3, 4: k4, 5: k5, 6: k6, 7: k7, 8: k8, 9: k9,
              10: k10, 11: k11, 12: k12, 13: k13, 14: k14, 15: k15}
    results = solve_case(kstar, name)
    for r in results:
        r['_residual'] = validate_coupler_curve(kstar, r)
    return results


# --------------------------------------------------------------------------
# 3. Case 1 (Eq. 13) and Case 2 (Eq. 16) input coefficients
# --------------------------------------------------------------------------
case1_kstar = {
    1: sp.Rational(5, 100), 2: sp.Rational(2, 10), 3: sp.Rational(-109375, 1000000),
    4: sp.Rational(18, 100), 5: sp.Rational(-29375, 1000000), 6: sp.Rational(875, 100000),
    7: sp.Rational(-4375, 1000000), 8: sp.Rational(-1525, 100000), 9: sp.Rational(-44375, 1000000),
    10: sp.Rational(107375, 10000000), 11: sp.Rational(1425, 1000000), 12: sp.Rational(214375, 100000000),
    13: sp.Rational(8525, 10000000), 14: sp.Rational(107375, 100000000), 15: sp.Rational(-479375025, 10000000000000),
}

case2_kstar = {
    1: sp.Rational(-2, 3), 2: sp.Integer(0), 3: sp.Rational(203, 1800),
    4: sp.Integer(0), 5: sp.Rational(53, 1800), 6: sp.Rational(17, 1800),
    7: sp.Integer(0), 8: sp.Rational(17, 1800), 9: sp.Integer(0),
    10: sp.nsimplify(-0.00333264, rational=False), 11: sp.Integer(0),
    12: sp.nsimplify(-0.00277708, rational=False),
    13: sp.nsimplify(0.00000833, rational=False), 14: sp.Integer(0),
    15: sp.nsimplify(0.000025, rational=False),
}

# --------------------------------------------------------------------------
# 4. Coupler-curve tracing (forward kinematics) and plotting (Figs. 3-5)
# --------------------------------------------------------------------------
def implicit_curve(kstar, X, Y):
    k = {i: float(kstar[i]) for i in range(1, 16)}
    r2 = X ** 2 + Y ** 2
    return (r2 ** 3 + (k[1] * X + k[2] * Y) * r2 ** 2
            + (k[3] * X ** 2 + k[4] * X * Y + k[5] * Y ** 2) * r2
            + k[6] * X ** 3 + k[7] * X ** 2 * Y + k[8] * X * Y ** 2 + k[9] * Y ** 3
            + k[10] * X ** 2 + k[11] * X * Y + k[12] * Y ** 2
            + k[13] * X + k[14] * Y + k[15])


def trace_linkage(r, n=3000):
    """Forward-kinematics trace of the coupler point P for both assembly
    branches of the DC dyad (the two 'loops'/circuits of the CCE)."""
    b1v, b2v, c1v, c2v = (float(r[k_]) for k_ in ('b1', 'b2', 'c1', 'c2'))
    l2v, l3v, l4v = (float(r[k_]) for k_ in ('l2', 'l3', 'l4'))
    mv, hv = float(r['m']), float(r['h'])
    Bf = np.array([b1v, b2v])
    Cf = np.array([c1v, c2v])

    pts = {0: [], 1: []}
    for beta in np.linspace(0, 2 * np.pi, n):
        A = Bf + l2v * np.array([np.cos(beta), np.sin(beta)])
        v = Cf - A
        d = np.linalg.norm(v)
        if d < 1e-9 or d > l3v + l4v or d < abs(l3v - l4v):
            continue
        a_ = (l3v ** 2 - l4v ** 2 + d ** 2) / (2 * d)
        h_sq = l3v ** 2 - a_ ** 2
        if h_sq < 0:
            continue
        h_ = np.sqrt(h_sq)
        mid = A + a_ * v / d
        perp = np.array([-v[1], v[0]]) / d
        for branch, sign in ((0, 1), (1, -1)):
            D = mid + sign * h_ * perp
            theta = np.arctan2(D[1] - A[1], D[0] - A[0])
            ct, st = np.cos(theta), np.sin(theta)
            P = A + np.array([ct * mv - st * hv, st * mv + ct * hv])
            pts[branch].append(P)
    return {b: np.array(v) for b, v in pts.items() if v}


def plot_cognates(results, kstar, indices, branch, fname, xr, yr, titles):
    fig, axes = plt.subplots(1, len(indices), figsize=(5 * len(indices), 5))
    if len(indices) == 1:
        axes = [axes]
    xs = np.linspace(*xr, 500)
    ys = np.linspace(*yr, 500)
    X, Y = np.meshgrid(xs, ys)
    Gamma = implicit_curve(kstar, X, Y)

    for ax, idx, title in zip(axes, indices, titles):
        r = results[idx]
        ax.contour(X, Y, Gamma, levels=[0], colors='red', linewidths=1)
        traces = trace_linkage(r)
        if branch in traces:
            pts = traces[branch]
            ax.plot(pts[:, 0], pts[:, 1], '.', color='red', markersize=0.6)

        b1v, b2v, c1v, c2v = (float(r[k_]) for k_ in ('b1', 'b2', 'c1', 'c2'))
        l2v = float(r['l2'])
        mv, hv = float(r['m']), float(r['h'])
        # draw one representative pose of the linkage
        beta0 = np.pi / 2
        A = np.array([b1v, b2v]) + l2v * np.array([np.cos(beta0), np.sin(beta0)])
        traces_full = trace_linkage(r, n=3000)
        # pick a D consistent with beta0 on the requested branch
        Cf = np.array([c1v, c2v])
        v = Cf - A
        d = np.linalg.norm(v)
        l3v, l4v = float(r['l3']), float(r['l4'])
        a_ = (l3v ** 2 - l4v ** 2 + d ** 2) / (2 * d)
        h_ = np.sqrt(max(l3v ** 2 - a_ ** 2, 0))
        mid = A + a_ * v / d
        perp = np.array([-v[1], v[0]]) / d
        sign = 1 if branch == 0 else -1
        D = mid + sign * h_ * perp
        theta = np.arctan2(D[1] - A[1], D[0] - A[0])
        ct, st = np.cos(theta), np.sin(theta)
        P = A + np.array([ct * mv - st * hv, st * mv + ct * hv])

        Bf = np.array([b1v, b2v])
        ax.plot([Bf[0], A[0]], [Bf[1], A[1]], 'k-o', markerfacecolor='white')
        ax.plot([Cf[0], D[0]], [Cf[1], D[1]], 'k-o', markerfacecolor='white')
        tri = np.array([A, D, P, A])
        ax.fill(tri[:, 0], tri[:, 1], color='cyan', alpha=0.5)
        ax.plot(tri[:, 0], tri[:, 1], 'k-')
        ax.plot(*P, 'ko', markerfacecolor='white')

        ax.set_xlim(*xr)
        ax.set_ylim(*yr)
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title)

    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  saved {fname}")


if __name__ == '__main__':
    res1 = solve_case(case1_kstar, 'Case 1')
    print_table(res1, 'Case 1 (cf. Table 2)')

    res2 = solve_case(case2_kstar, 'Case 2')
    print_table(res2, 'Case 2 (cf. Table 4)')

    # Identify rows of res1 matching Table 2's numbering (1..6) so that we
    # reproduce "Fig. 3: cognates from solutions 1,3,5" and
    # "Fig. 4: cognates from solutions 2,4,6" using the paper's own labels.
    def match_table2_row(r, table2_row):
        keys = ('m', 'h', 'b1', 'b2', 'c1', 'c2', 'l2', 'l3', 'l4')
        return all(abs(float(r[k_]) - table2_row[k_]) < 1e-3 for k_ in keys)

    table2 = [
        dict(m=0.1, h=0.15, b1=-0.2, b2=0, c1=0.2, c2=-0.2, l2=0.15, l3=0.4, l4=0.35),
        dict(m=0.3, h=-0.15, b1=0.2, b2=-0.2, c1=-0.2, c2=0, l2=0.35, l3=0.4, l4=0.15),
        dict(m=0.313, h=0.1565, b1=0.2, b2=-0.2, c1=-0.025, c2=0.1, l2=0.3354, l3=0.2935, l4=0.1258),
        dict(m=-0.0196, h=-0.1565, b1=-0.025, b2=0.1, c1=0.2, c2=-0.2, l2=0.1258, l3=0.2935, l4=0.3354),
        dict(m=-0.0156, h=0.1248, b1=-0.025, b2=0.1, c1=-0.2, c2=0, l2=0.1577, l3=0.0676, l4=0.1803),
        dict(m=0.0832, h=-0.1248, b1=-0.2, b2=0, c1=-0.025, c2=0.1, l2=0.1803, l3=0.0676, l4=0.1577),
    ]
    order = [None] * 6
    for i, r in enumerate(res1):
        for j, t2 in enumerate(table2):
            if match_table2_row(r, t2):
                order[j] = i
    print("res1 index matching Table-2 row 1..6:", order)

    print("\nPlotting Fig. 3 (first loop, solutions 1,3,5) and Fig. 4 (second loop, 2,4,6)...")
    plot_cognates(res1, case1_kstar, [order[0], order[2], order[4]], branch=0,
                  fname='fig3_case1_loop1.png', xr=(-0.35, 0.35), yr=(-0.35, 0.35),
                  titles=['Cognate I (sol. 1)', 'Cognate II (sol. 3)', 'Cognate III (sol. 5)'])
    plot_cognates(res1, case1_kstar, [order[1], order[3], order[5]], branch=1,
                  fname='fig4_case1_loop2.png', xr=(-0.35, 0.35), yr=(-0.35, 0.35),
                  titles=['Cognate I (sol. 2)', 'Cognate II (sol. 4)', 'Cognate III (sol. 6)'])

    print("Plotting Fig. 5 (Case 2 cognates)...")
    plot_cognates(res2, case2_kstar, [0, 1, 2], branch=0,
                  fname='fig5_case2.png', xr=(-0.2, 0.3), yr=(-0.2, 0.3),
                  titles=['Cognate I', 'Cognate II', 'Cognate III'])

    print("\nDone. Compare fig3_case1_loop1.png / fig4_case1_loop2.png / fig5_case2.png "
          "to Figs. 3-5 of the paper.")

    # ------------------------------------------------------------------
    # Example: solve_from_polynomial() on a curve that has NOTHING to do
    # with the paper, to demonstrate the method is fully general -- any
    # polynomial of the required 15-term sextic form, once normalized to a
    # leading x^6 coefficient of 1, can be handed to it directly.
    # ------------------------------------------------------------------
    print("\n=== Example: an arbitrary coupler curve (not from the paper) ===")
    example_k = {1: -0.9424242424242424, 2: -0.30303030303030304, 3: 0.18806385664819472,
                 4: 0.19493333333333335, 5: -0.04294177796715472, 6: 0.03725473388202942,
                 7: -0.025446685878962538, 8: 0.06230353709270574, 9: -0.012028059286898425,
                 10: -0.008954410818041966, 11: -0.010284213164594626, 12: 0.0012296104339375322,
                 13: -0.0003069885411220314, 14: 0.001448635131217276, 15: 6.917952722995226e-05}
    example_res = solve_from_polynomial(**{f'k{i}': v for i, v in example_k.items()},
                                         name="arbitrary example")
    print_table(example_res, "arbitrary example")
    print("Consistency check, all 6 relations (should be ~0 for a genuine coupler curve; row 2 shown):")
    for k_, v_ in example_res[1]['_residual'].items():
        print(f"    {k_:>8}: {v_: .3e}")

    print("\n=== Redundancy check on the paper's own Case 1 ===")
    for i, r in enumerate(res1, 1):
        resid = validate_coupler_curve(case1_kstar, r)
        worst = max(abs(v) for v in resid.values())
        print(f"  sol {i}: worst residual = {worst:.2e}")
