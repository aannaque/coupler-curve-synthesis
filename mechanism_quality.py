"""
Extension #3a: practical-usability filter for the candidate linkages returned
by solve_case() / synthesize_from_points().

The exact method (and the point-fitting pipeline built on it) can return
several algebraically/geometrically valid four-bar linkages for the same
coupler curve -- but "traces the right curve" says nothing about whether the
mechanism is actually buildable as a continuously-motor-driven device. Two
classical, purely geometric checks settle that:

  - Grashof condition: whether the input link (AB, length l2) can complete a
    FULL rotation about its fixed pivot B, as opposed to only oscillating
    (rocking) -- a linkage with a full crank is far more useful in practice
    (a single continuous-rotation motor suffices) than one needing a
    reciprocating actuator.
  - Transmission angle: the angle at joint D between the coupler (link AD)
    and the output rocker (link DC). It should stay well away from 0 deg/180
    deg through the whole cycle -- near those values the mechanism is close
    to a singular (locked/jamming) configuration and small friction or load
    variations cause large output-force/velocity errors. This module reports
    the worst-case deviation from the ideal 90 deg seen over the cycle
    (design rule of thumb: keep it under ~50 deg, i.e. transmission angle
    within roughly [40 deg, 140 deg]).

Both are computed directly from the same forward-kinematics sweep already
used for coupler-curve tracing, so classify_linkage() is cheap (a few
milliseconds) and can be applied to re-rank ties in fit quality: among
several candidates with statistically indistinguishable RMS, prefer the one
that is an actual crank-rocker with a healthy transmission angle -- that is
the one a real designer would build.
"""

import numpy as np

SYM_ORDER = ('m', 'h', 'b1', 'b2', 'c1', 'c2', 'l2', 'l3', 'l4')


def classify_linkage(params, n=2000):
    """params: dict (or sequence in SYM_ORDER order) of the 9 linkage
    parameters. Returns a dict describing:
      l1                       - the (derived) frame length |BC|
      grashof                  - Grashof classification (which link, if any,
                                  can fully rotate)
      l2_is_crank              - True iff the INPUT link AB can complete a
                                  full rotation (the practically important
                                  case: a motor at B can drive it directly)
      assemblable_fraction_pct - % of the 0..360 deg sweep of AB's angle for
                                  which the linkage can actually be
                                  assembled at all (100% for any true crank)
      worst_transmission_deviation_deg -
                                  max over the assemblable cycle of
                                  |90 - transmission_angle|; smaller is
                                  better, over ~50 is a poor/jamming-prone
                                  design
    """
    if isinstance(params, dict):
        _m, _h, b1, b2, c1, c2, l2, l3, l4 = (float(params[k]) for k in SYM_ORDER)
    else:
        _m, _h, b1, b2, c1, c2, l2, l3, l4 = (float(v) for v in params)

    l1 = float(np.hypot(c1 - b1, c2 - b2))
    lengths = {'l1(frame)': l1, 'l2(input)': l2, 'l3(coupler)': l3, 'l4(output)': l4}
    names = list(lengths.keys())
    vals = np.array([lengths[nm] for nm in names])
    order = np.argsort(vals)
    s, l = vals[order[0]], vals[order[-1]]
    p, q = vals[order[1]], vals[order[2]]
    is_grashof = (s + l) <= (p + q) + 1e-9
    shortest_name = names[order[0]]

    if not is_grashof:
        gtype, l2_is_crank = 'non-Grashof (triple-rocker)', False
    elif shortest_name == 'l1(frame)':
        gtype, l2_is_crank = 'double-crank (drag-link)', True
    elif shortest_name == 'l2(input)':
        gtype, l2_is_crank = 'crank-rocker (l2 is the crank)', True
    elif shortest_name == 'l4(output)':
        gtype, l2_is_crank = 'crank-rocker (l4 is the crank; l2 only rocks)', False
    else:
        gtype, l2_is_crank = 'double-rocker (coupler is shortest)', False

    beta = np.linspace(0, 2 * np.pi, n)
    Ax, Ay = b1 + l2 * np.cos(beta), b2 + l2 * np.sin(beta)
    vx, vy = c1 - Ax, c2 - Ay
    d = np.hypot(vx, vy)
    valid = (d > 1e-9) & (d <= l3 + l4) & (d >= abs(l3 - l4))
    assemblable_pct = round(float(np.mean(valid)) * 100, 1)
    if not np.any(valid):
        return dict(l1=round(l1, 4), grashof=gtype, l2_is_crank=l2_is_crank,
                    assemblable_fraction_pct=0.0, worst_transmission_deviation_deg=None)

    Ax, Ay, vx, vy, d = Ax[valid], Ay[valid], vx[valid], vy[valid], d[valid]
    a_ = (l3 ** 2 - l4 ** 2 + d ** 2) / (2 * d)
    h_sq = l3 ** 2 - a_ ** 2
    ok = h_sq >= 0
    Ax, Ay, vx, vy, d, a_, h_sq = (arr[ok] for arr in (Ax, Ay, vx, vy, d, a_, h_sq))
    hh = np.sqrt(h_sq)
    midx, midy = Ax + a_ * vx / d, Ay + a_ * vy / d
    perpx, perpy = -vy / d, vx / d

    worst_dev = 0.0
    for sign in (1, -1):
        Dx, Dy = midx + sign * hh * perpx, midy + sign * hh * perpy
        DA = np.column_stack([Ax - Dx, Ay - Dy])
        DC = np.column_stack([c1 - Dx, c2 - Dy])
        cosang = np.clip(np.sum(DA * DC, axis=1) / (l3 * l4), -1, 1)
        ang = np.degrees(np.arccos(cosang))
        worst_dev = max(worst_dev, float(np.max(np.abs(90 - ang))))

    return dict(l1=round(l1, 4), grashof=gtype, l2_is_crank=l2_is_crank,
                assemblable_fraction_pct=assemblable_pct,
                worst_transmission_deviation_deg=round(worst_dev, 1))


def practical_score(quality):
    """Single scalar to sort candidates by practicality (lower is better):
    heavily penalize not being a true crank and poor/undefined transmission
    quality, so a designer's default ranking prefers buildable mechanisms."""
    if quality['worst_transmission_deviation_deg'] is None:
        return 1e6
    penalty = 0 if quality['l2_is_crank'] else 200
    return penalty + quality['worst_transmission_deviation_deg']
