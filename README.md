# Fully Analytical Four-Bar Coupler-Curve Synthesis

Reference Python implementation for the paper:
**"Algebraic Redundancy, Point-Based Extensions, and Practical Selection for
Fully Analytical Four-Bar Coupler-Curve Synthesis"** (A. Aannaque, 2026),
extending R. Wu, R. Li, S. Bai, *Mech. Mach. Theory* 155 (2021) 104070.

## Overview

This code extends the fully analytical synthesis method of Wu, Li & Bai by
providing:

* An independent symbolic re-derivation of the coupler-curve equation (CCE)
  coefficients using `sympy` (not taken from the original paper's
  intermediate formulas), verified to reproduce their Tables 1-4 and Figs.
  3-5 exactly.
* A constant-time algebraic validity certificate for any 15-coefficient CCE,
  identifying two previously unremarked algebraic relations alongside the
  four the original method leaves unused (six total, matching the proven
  codimension-6 redundancy).
* A complete fit-seed-polish pipeline (`scipy.optimize.least_squares`,
  trust-region-reflective) to synthesize a linkage directly from discrete
  task points rather than a known exact CCE.
* A Grashof/transmission-angle classifier to select, among several
  algebraically valid cognates, the one that is an actual buildable
  crank-driven mechanism.
* A controlled identifiability/noise-sensitivity study of the point-based
  pipeline, including a tested (and reported as unsuccessful) attempt at
  fixing it via coordinate normalization and Tikhonov regularization
  (Section 5.3 of the paper).

## Requirements

```
pip install -r requirements.txt
```

Tested with Python 3.13; SymPy's on-import symbolic derivation takes
roughly 10-15 seconds the first time `coupler_curve_synthesis` is imported
in a process.

## Files

| File | Role |
|---|---|
| `coupler_curve_synthesis.py` | Core re-derivation of the CCE, `solve_case`/`solve_from_polynomial` (the exact solver), `validate_coupler_curve` (the six-relation redundancy certificate), and reproduction of Wu et al.'s Tables 1-4 and Figs. 3-5. |
| `precision_point_synthesis.py` | The fit-seed-polish pipeline (`synthesize_from_points`) and the noise-free/noisy demonstrations that produce this paper's Fig. 1 and Fig. 2. Imports `coupler_curve_synthesis` and `mechanism_quality`. |
| `mechanism_quality.py` | `classify_linkage` (Grashof condition, transmission-angle quality) and `practical_score`/`rank_practical` used to re-rank near-tied fit candidates. |
| `identifiability_sweep.py` | The $N \times \sigma$ Monte Carlo sweep reproducing Table 1 (Section 5.2). Slow (~36 solver calls, several minutes). |
| `main.py` | Convenience wrapper that runs the above in the right order; see Usage. |

There is no separate `validate_coupler_curve.py`, `synthesis_pipeline.py`, or
`generate_figures.py` script -- that functionality lives inside
`coupler_curve_synthesis.py` and `precision_point_synthesis.py` as noted
above, and figures are produced as a side effect of running those files
directly (via `if __name__ == '__main__':`) rather than by a separate
plotting script.

## Usage

Simplest path -- run everything in order:

```
python main.py            # baseline validation + Fig. 1 and Fig. 2 (~1-2 min)
python main.py --sweep    # also reproduce Table 1 (slow: several minutes)
```

Or run each piece individually:

1. **Baseline validation and redundancy certificate:**
   `python coupler_curve_synthesis.py` -- reproduces Wu et al.'s Tables 1-4
   and Figs. 3-5, then runs the codimension-6 validity check
   (`validate_coupler_curve`) on both their worked example and an
   independently generated linkage.
2. **Precision-point synthesis (Fig. 1, Fig. 2):**
   `python precision_point_synthesis.py` -- fits a sextic to sampled points
   from a known linkage (noise-free, then noisy), seeds the exact solver,
   polishes geometrically, and re-ranks by practicality.
3. **Identifiability sweep (Table 1):**
   `python identifiability_sweep.py` -- the $N\in\{15,30,60\}$,
   $\sigma\in\{0,0.002,0.005\}$ sensitivity study.

To use the solver on your own coupler-curve coefficients or task points,
import `solve_from_polynomial` or `synthesize_from_points` directly; see
their docstrings for the expected input format.

## Citation

If you use this code in your research, please cite the associated paper.

## License

MIT -- see `LICENSE`.
