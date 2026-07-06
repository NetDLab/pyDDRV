r"""Sum-of-Squares (SOS) exponential-rate certification — the Table I baseline.

The paper benchmarks Algorithm 1 against **SOSTOOLS** (MATLAB). This module
provides the same comparison in pure Python via `SumOfSquares` + `picos` + an SDP
solver (cvxopt), so no MATLAB is needed.

Unlike the data-driven RLF core, this is **model-based**: it needs the polynomial
vector field symbolically. For a polynomial system :math:`\dot x = f(x)` it seeks
a homogeneous Lyapunov function :math:`V` of degree ``degree`` (``2`` = quadratic
:math:`x^\top P x`; ``4, 6, ...`` are richer, less conservative) certifying a
local exponential rate :math:`\alpha` on the box :math:`Q_R=\{|x_i|\le R\}`:

.. math::

    \dot V(x) = \nabla V \cdot f(x) \;\le\; -(\text{degree})\,\alpha\, V(x),
    \qquad x \in Q_R,

where the ``degree`` factor normalizes so that :math:`\alpha` is the **state-norm**
exponential rate (:math:`V \sim \lVert x\rVert^{\text{degree}}`), comparable
across degrees and to the RLF rate.

enforced by a Positivstellensatz certificate with SOS multipliers
:math:`\sigma_i \ge 0` on the box constraints :math:`g_i(x)=R^2-x_i^2`:

.. math::

    -\nabla V\!\cdot\! f - 2\alpha V - \sum_i \sigma_i\,g_i \ \text{is SOS},
    \quad \sigma_i \ \text{SOS},
    \quad V - \epsilon\lVert x\rVert^2 \ \text{is SOS}.

For fixed :math:`\alpha` this is an SDP feasibility problem; the largest
certifiable :math:`\alpha` is found by bisection.

**Two certified regions — read this before comparing to RLF.**
``certify_rate_sos`` enforces the decrease only *pointwise on* :math:`Q_R`, which
soundly certifies the largest **inner** sub-level set :math:`\{V\le c\}\subseteq
Q_R` -- a *smaller* region than RLF certifies. For an apples-to-apples comparison
(both certifying over :math:`Q_R`), use ``certify_rate_sos_box``, which certifies
that :math:`Q_R` lies inside an **invariant** sub-level set :math:`\{V\le1\}` via a
V--s alternation (the bilinear :math:`\sigma\!\cdot\!V` term is broken by
alternating). The box version fails on draws whose trajectories leave :math:`Q_R`
(divergent or excursing) -- because invariance is strictly stronger than RLF's
recurrence -- which is exactly why the inner-ellipse version overstates SOS.

Install the optional dependencies with ``pip install -e ".[sos]"``.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
import sympy as sp


def build_polynomial_field(A, B, monomials, xs) -> List[sp.Expr]:
    """Assemble ``f_i = A_i . x + B_i . monomials(x)`` as sympy expressions.

    Parameters
    ----------
    A : (d, d) array_like
        Linear part.
    B : (d, m) array_like
        Coefficients of the nonlinear monomials.
    monomials : sequence of sympy expressions, length m
        The nonlinear monomials (e.g. ``[x1**2, x1*x2, x2**2]``).
    xs : sequence of sympy symbols, length d
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    d = len(xs)
    f = []
    for i in range(d):
        lin = sum(A[i, j] * xs[j] for j in range(d))
        nl = sum(B[i, k] * monomials[k] for k in range(len(monomials)))
        f.append(sp.expand(lin + nl))
    return f


_SOLVER_CACHE = {}


def available_solver():
    """Return ``"mosek"`` if it is installed AND licensed, else ``"cvxopt"``.

    MOSEK is far more robust on these SDPs (cvxopt hangs on some 3D draws), but
    needs an academic license at ``~/mosek/mosek.lic``. The check runs a trivial
    solve once and caches the result.
    """
    if "default" not in _SOLVER_CACHE:
        _SOLVER_CACHE["default"] = "cvxopt"
        try:
            import picos
            if "mosek" in picos.solvers.available_solvers():
                p = picos.Problem()
                x = picos.RealVariable("x")
                p.set_objective("min", x)
                p.add_constraint(x >= 1)
                p.solve(solver="mosek")          # raises if unlicensed
                _SOLVER_CACHE["default"] = "mosek"
        except Exception:
            pass
    return _SOLVER_CACHE["default"]


def _feasible(f, xs, R, alpha, eps, degree, deg_sigma, solver, max_iters):
    from SumOfSquares import SOSProblem, poly_variable

    if solver == "cvxopt":
        # Cap iterations so an ill-conditioned draw can't hang the solve.
        import cvxopt
        cvxopt.solvers.options["maxiters"] = max_iters
        cvxopt.solvers.options["show_progress"] = False

    prob = SOSProblem()
    V = poly_variable("V", list(xs), degree, hom=True)          # homogeneous deg `degree`
    sig = [poly_variable(f"s{i}", list(xs), deg_sigma) for i in range(len(xs))]
    g = [R**2 - xi**2 for xi in xs]
    Vdot = sum(sp.diff(V, xi) * fi for xi, fi in zip(xs, f))
    # For homogeneous V of degree `degree`=2d, V ~ ||x||^{2d}; the condition
    # Vdot <= -(degree) alpha V makes ||x|| decay at rate alpha (so alpha is the
    # state-norm exponential rate, comparable across degrees and to the RLF rate).
    expr = sp.expand(-Vdot - degree * alpha * V - sum(s * gi for s, gi in zip(sig, g)))
    prob.add_sos_constraint(expr, list(xs))
    for s in sig:
        prob.add_sos_constraint(s, list(xs))
    # Positive definiteness: V - eps * ||x||^degree is SOS.
    proper = sp.expand(V - eps * (sum(xi**2 for xi in xs)) ** (degree // 2))
    prob.add_sos_constraint(proper, list(xs))
    try:
        prob.solve(solver=solver)
        return prob.status in ("optimal", "feasible")
    except Exception:
        return False


def certify_rate_sos(f: Sequence[sp.Expr], xs: Sequence[sp.Symbol], R: float,
                     lo: float = 0.0, hi: float = 1.5, iters: int = 18,
                     eps: float = 1e-3, degree: int = 2, deg_sigma: int = None,
                     solver: str = "auto", max_iters: int = 100) -> float:
    r"""Largest exponential rate ``alpha`` SOS-certifiable for ``f`` on ``Q_R``.

    Bisects ``alpha`` in ``[lo, hi]``. Returns ``nan`` if even ``lo`` is
    infeasible (no certificate of this degree on the box).

    Parameters
    ----------
    f : sequence of sympy expressions
        Polynomial vector field, ``len(f) == len(xs)``.
    xs : sequence of sympy symbols
    R : float
        Box half-width (region ``Q_R``).
    degree : int, default 2
        Degree of the (homogeneous) Lyapunov function ``V`` -- 2 = quadratic,
        4, 6, ... give richer, less conservative certificates at higher SDP cost.
        Must be even.
    deg_sigma : int, optional
        Degree of the SOS box multipliers (defaults to ``degree``).
    solver : str
        ``"auto"`` (default) picks MOSEK if licensed, else cvxopt; or name one
        explicitly (``"mosek"``, ``"cvxopt"``).
    max_iters : int
        Cap on the cvxopt iterations (robustness against bad draws).
    """
    if degree % 2:
        raise ValueError("degree must be even")
    if deg_sigma is None:
        deg_sigma = degree
    if solver == "auto":
        solver = available_solver()

    def feas(a):
        return _feasible(f, xs, R, a, eps, degree, deg_sigma, solver, max_iters)

    if not feas(lo):
        return float("nan")
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if feas(mid):
            lo = mid
        else:
            hi = mid
    return lo


# --------------------------------------------------------------------------- #
# Box (invariant sub-level set) certification — the FAIR comparison to RLF
# --------------------------------------------------------------------------- #
def _linearization(f, xs):
    """Jacobian of f at the origin (the linear part A)."""
    d = len(xs)
    zero = {x: 0 for x in xs}
    return np.array([[float(sp.diff(f[i], xs[j]).subs(zero)) for j in range(d)]
                     for i in range(d)])


def _init_V(f, xs, R, degree=2):
    """Initial V of the given degree from the linearization, scaled so Q_R ⊆ {V≤0.9}.

    Quadratic ``x^T P x`` (P solving the linearization Lyapunov eqn); for
    ``degree > 2`` the initial guess is ``(x^T P x)^{degree/2}``. Returns None if
    the linearization is not Hurwitz.
    """
    import itertools
    from scipy.linalg import solve_continuous_lyapunov
    A = _linearization(f, xs)
    try:
        P = solve_continuous_lyapunov(A.T, -np.eye(len(xs)))
    except Exception:
        return None
    if np.min(np.linalg.eigvalsh(0.5 * (P + P.T))) <= 0:
        return None
    Vq = sum(P[i, j] * xs[i] * xs[j] for i in range(len(xs)) for j in range(len(xs)))
    V = sp.expand(Vq ** (degree // 2))
    corners = np.array(list(itertools.product([-R, R], repeat=len(xs))))
    subs = [dict(zip(xs, c)) for c in corners]
    cmax = max(float(V.subs(d)) for d in subs)
    return sp.expand(V * (0.9 / cmax))


def _set_cvxopt(solver, max_iters):
    if solver == "cvxopt":
        import cvxopt
        cvxopt.solvers.options["maxiters"] = max_iters
        cvxopt.solvers.options["show_progress"] = False


def _vs_s_step(f, xs, V, alpha, box, solver, max_iters, degree):
    """V fixed -> solve for multipliers; return numeric s1 or None.

    Rate normalized as Vdot <= -(degree) alpha V on {V<=1}, so alpha is the
    state-norm rate (V ~ ||x||^degree). Multiplier degrees scale with V's degree.
    """
    from SumOfSquares import SOSProblem, poly_variable
    _set_cvxopt(solver, max_iters)
    prob = SOSProblem()
    s1 = poly_variable("s1", list(xs), degree)
    lam = [poly_variable(f"l{i}", list(xs), degree - 2) for i in range(len(xs))]
    Vdot = sum(sp.diff(V, xi) * fi for xi, fi in zip(xs, f))
    prob.add_sos_constraint(sp.expand(-(Vdot + degree * alpha * V) - s1 * (1 - V)), list(xs))
    prob.add_sos_constraint(s1, list(xs))
    prob.add_sos_constraint(sp.expand((1 - V) - sum(l * g for l, g in zip(lam, box))), list(xs))
    for l in lam:
        prob.add_sos_constraint(l, list(xs))
    try:
        prob.solve(solver=solver)
        if prob.status not in ("optimal", "feasible"):
            return None
        return sp.expand(prob.subs_with_sol(s1))
    except Exception:
        return None


def _vs_v_step(f, xs, s1, alpha, box, solver, max_iters, eps, degree):
    """s1 fixed -> solve for V (PD, degree `degree`); return numeric V or None."""
    from SumOfSquares import SOSProblem, poly_variable
    _set_cvxopt(solver, max_iters)
    prob = SOSProblem()
    V = poly_variable("V", list(xs), degree, hom=True)
    lam = [poly_variable(f"m{i}", list(xs), degree - 2) for i in range(len(xs))]
    Vdot = sum(sp.diff(V, xi) * fi for xi, fi in zip(xs, f))
    prob.add_sos_constraint(sp.expand(-(Vdot + degree * alpha * V) - s1 * (1 - V)), list(xs))
    prob.add_sos_constraint(sp.expand((1 - V) - sum(l * g for l, g in zip(lam, box))), list(xs))
    for l in lam:
        prob.add_sos_constraint(l, list(xs))
    prob.add_sos_constraint(sp.expand(V - eps * (sum(xi**2 for xi in xs)) ** (degree // 2)),
                            list(xs))
    try:
        prob.solve(solver=solver)
        if prob.status not in ("optimal", "feasible"):
            return None
        return sp.expand(prob.subs_with_sol(V))
    except Exception:
        return None


def certify_rate_sos_box(f: Sequence[sp.Expr], xs: Sequence[sp.Symbol], R: float,
                         lo: float = 0.0, hi: float = 1.0, iters: int = 12,
                         rounds: int = 3, eps: float = 1e-4, degree: int = 2,
                         solver: str = "auto", max_iters: int = 100) -> float:
    r"""Largest rate certifiable with ``Q_R`` INSIDE an invariant set ``{V≤1}``.

    The fair region match to RLF: rather than the inner ellipse
    ``{V≤c}⊆Q_R`` of :func:`certify_rate_sos`, this certifies that the whole box
    sits inside an invariant sub-level set decreasing at rate ``alpha``. The
    bilinear ``sigma*V`` term is handled by a V--s alternation initialised from
    the linearization Lyapunov function. Returns ``nan`` if no certificate of this
    ``degree`` exists.

    ``degree`` (even, default 2) is the degree of the Lyapunov function ``V``. A
    quadratic fails on draws that leave ``Q_R`` (excursing); a higher-degree ``V``
    can shape a level set that contains the box and bends around the excursion,
    certifying systems a quadratic cannot -- at higher SDP cost.
    """
    if degree % 2:
        raise ValueError("degree must be even")
    if solver == "auto":
        solver = available_solver()
    box = [R**2 - xi**2 for xi in xs]

    def feasible(alpha):
        V = _init_V(f, xs, R, degree)
        if V is None:
            return False
        for _ in range(rounds):
            s1 = _vs_s_step(f, xs, V, alpha, box, solver, max_iters, degree)
            if s1 is None:
                return False
            V = _vs_v_step(f, xs, s1, alpha, box, solver, max_iters, eps, degree)
            if V is None:
                return False
        return True

    if not feasible(lo):
        return float("nan")
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if feasible(mid):
            lo = mid
        else:
            hi = mid
    return lo
