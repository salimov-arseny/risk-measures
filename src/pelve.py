"""
PELVE: Probability Equivalent Level of VaR and ES.

Implements numerical computation of the PELVE coefficient

    Pi_eps(X) = c such that ES_{1 - c*eps}(X) = VaR_{1 - eps}(X),

for several distributions commonly used in actuarial science:
Uniform, Pareto, Exponential, Normal, Lognormal, Gamma, Weibull.

The implementation follows the 3rd-year course paper of A.E. Salimov
(MSU, Faculty of Mechanics and Mathematics, Department of Probability
Theory, 2025) supervised by Prof. G.I. Falin.

References
----------
Li H., Wang R. (2021). PELVE: Probability Equivalent Level of VaR and ES.
Journal of Econometrics, 234(1), 353-370.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import scipy.stats as stats
from scipy.integrate import quad
from scipy.optimize import brentq


_QUAD_KW = dict(epsabs=1e-14, epsrel=1e-14, limit=1000)
_BRENTQ_KW = dict(maxiter=10000, xtol=1e-14)


@dataclass(frozen=True)
class PelveResult:
    """Result of a PELVE computation."""

    c: float
    eps: float
    var: float


def value_at_risk(isf: Callable[[float], float], eps: float) -> float:
    """VaR at confidence level ``1 - eps`` via the inverse survival function."""
    return isf(eps)


def expected_shortfall(isf: Callable[[float], float], c: float, eps: float) -> float:
    """ES at confidence level ``1 - c*eps`` computed as

        ES_{1 - c*eps}(X) = (1 / (c*eps)) * integral_{1-c*eps}^{1} F^{-1}(q) dq
                          = integral_0^1 S^{-1}(c*eps*t) dt.

    The survival-function form is numerically stable for very small ``eps``.
    """
    integral, _ = quad(lambda t: isf(c * eps * t), 0.0, 1.0, **_QUAD_KW)
    return integral


def find_pelve(isf: Callable[[float], float], eps: float) -> PelveResult:
    """Solve ``ES_{1 - c*eps}(X) = VaR_{1 - eps}(X)`` for ``c`` in ``[1, 1/eps]``.

    Parameters
    ----------
    isf : callable
        Inverse survival function ``S^{-1}(q)`` of the loss distribution.
    eps : float
        Tail probability, ``eps`` in ``(0, 1)``.

    Returns
    -------
    PelveResult
        The PELVE coefficient together with the input ``eps`` and the
        corresponding VaR value.
    """
    var = value_at_risk(isf, eps)

    def equation(c: float) -> float:
        return expected_shortfall(isf, c, eps) - var

    c_sol = brentq(equation, 1.0, 1.0 / eps, **_BRENTQ_KW)
    return PelveResult(c=c_sol, eps=eps, var=var)


def normal_isf(sigma: float = 1.0) -> Callable[[float], float]:
    """Inverse survival function of ``N(0, sigma^2)``."""
    return lambda q: stats.norm.isf(q, loc=0.0, scale=sigma)


def lognormal_isf(sigma: float = 1.0) -> Callable[[float], float]:
    """Inverse survival function of ``LN(0, sigma^2)``."""
    return lambda q: stats.lognorm.isf(q, sigma)


def gamma_isf(alpha: float, beta: float, x0: float = 0.0) -> Callable[[float], float]:
    """Inverse survival function of the shifted Gamma distribution
    with density

        p(x) = alpha^beta * (x - x0)^(beta - 1) * exp(-alpha*(x - x0)) / Gamma(beta),

    supported on ``[x0, +infty)``.
    """
    return lambda q: x0 + stats.gamma.isf(q, beta, scale=1.0 / alpha)


def weibull_isf(k: float, scale: float = 1.0) -> Callable[[float], float]:
    """Inverse survival function of the Weibull distribution ``W(k, scale)``."""
    return lambda q: stats.weibull_min.isf(q, k, scale=scale)


def exponential_isf(lam: float = 1.0) -> Callable[[float], float]:
    """Inverse survival function of ``Exp(lambda)``."""
    return lambda q: stats.expon.isf(q, scale=1.0 / lam)


def pareto_isf(k: float, x0: float = 1.0) -> Callable[[float], float]:
    """Inverse survival function of ``Pareto(k, x0)``."""
    return lambda q: stats.pareto.isf(q, k, scale=x0)


def uniform_isf(a: float = 0.0, b: float = 1.0) -> Callable[[float], float]:
    """Inverse survival function of ``R[a, b]``."""
    return lambda q: stats.uniform.isf(q, loc=a, scale=b - a)


def pelve_normal(eps: float, sigma: float = 1.0) -> PelveResult:
    return find_pelve(normal_isf(sigma), eps)


def pelve_lognormal(eps: float, sigma: float = 1.0) -> PelveResult:
    return find_pelve(lognormal_isf(sigma), eps)


def pelve_gamma(eps: float, alpha: float, beta: float, x0: float = 0.0) -> PelveResult:
    return find_pelve(gamma_isf(alpha, beta, x0), eps)


def pelve_weibull(eps: float, k: float, scale: float = 1.0) -> PelveResult:
    return find_pelve(weibull_isf(k, scale), eps)


def pelve_exponential(eps: float, lam: float = 1.0) -> PelveResult:
    return find_pelve(exponential_isf(lam), eps)


def pelve_pareto(eps: float, k: float, x0: float = 1.0) -> PelveResult:
    return find_pelve(pareto_isf(k, x0), eps)


def pelve_uniform(eps: float, a: float = 0.0, b: float = 1.0) -> PelveResult:
    return find_pelve(uniform_isf(a, b), eps)


def pelve_pareto_closed_form(k: float) -> float:
    """Closed-form PELVE for ``Pareto(k, x0)`` with ``k > 1``:

        Pi_eps(X) = (k / (k - 1))^k.
    """
    if k <= 1.0:
        raise ValueError("Closed-form PELVE for Pareto requires k > 1.")
    return (k / (k - 1.0)) ** k


def pelve_exponential_closed_form() -> float:
    """Closed-form PELVE for ``Exp(lambda)``: ``Pi_eps(X) = e``."""
    return math.e


def pelve_uniform_closed_form() -> float:
    """Closed-form PELVE for ``R[a, b]`` with ``eps <= 1/2``: ``Pi_eps(X) = 2``."""
    return 2.0


def pelve_curve(
    isf: Callable[[float], float],
    eps_values: np.ndarray,
) -> np.ndarray:
    """Vectorised computation of ``Pi_eps(X)`` over an array of ``eps`` values."""
    out = np.empty_like(eps_values, dtype=float)
    for i, eps in enumerate(eps_values):
        out[i] = find_pelve(isf, float(eps)).c
    return out


__all__ = [
    "PelveResult",
    "value_at_risk",
    "expected_shortfall",
    "find_pelve",
    "normal_isf",
    "lognormal_isf",
    "gamma_isf",
    "weibull_isf",
    "exponential_isf",
    "pareto_isf",
    "uniform_isf",
    "pelve_normal",
    "pelve_lognormal",
    "pelve_gamma",
    "pelve_weibull",
    "pelve_exponential",
    "pelve_pareto",
    "pelve_uniform",
    "pelve_pareto_closed_form",
    "pelve_exponential_closed_form",
    "pelve_uniform_closed_form",
    "pelve_curve",
]
