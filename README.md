# Probability Equivalent Risk Measures (PELVE)

Numerical implementation and theoretical study of the **PELVE** coefficient (*Probability Equivalent Level of VaR and ES*) for distributions arising in actuarial and financial mathematics.

This project accompanies the 3rd-year course paper:

> Salimov A.E. *Probability Equivalent Risk Measures.* M.V. Lomonosov Moscow State University, Faculty of Mechanics and Mathematics, Department of Probability Theory, 2025. Supervisor: Prof. G.I. Falin, Dr. Sc.

The PDF and LaTeX source are in [`paper/`](paper/).

## Problem statement

Let $X$ be a positive absolutely continuous random variable modelling aggregate insurance claims or financial losses, with CDF $F_X$ and survival function $S_X = 1 - F_X$. For a confidence level $p \in [0, 1)$:

$$\mathrm{VaR}_p(X) = F_X^{-1}(p),\qquad \mathrm{ES}_p(X) = \frac{1}{1 - p}\int_p^1 F_X^{-1}(q)\,dq.$$

The **PELVE** $\Pi_\varepsilon(X)$ is the constant $c \in [1, 1/\varepsilon]$ such that

$$\mathrm{ES}_{1 - c\varepsilon}(X) = \mathrm{VaR}_{1 - \varepsilon}(X).$$

Introduced by Li & Wang (2021), PELVE quantifies the regulatory impact of the Basel III transition from $\mathrm{VaR}_{0.99}$ to $\mathrm{ES}_{0.975}$. The threshold is $c = 2.5$: if $\Pi_\varepsilon(X) > 2.5$, capital requirements increase.

## Main results

**Existence and uniqueness** (Prop. 2.1). If $\mathbb{E}X < \infty$ and $\mathbb{E}X \leq \mathrm{VaR}_{1-\varepsilon}(X)$, the equation has a unique solution.

**Invariance** (Thm. 3.1). For any $\lambda > 0,\, a \in \mathbb{R}$: $\Pi_\varepsilon(\lambda X + a) = \Pi_\varepsilon(X)$. Convex increasing transformations do not decrease PELVE, so it is a measure of tail heaviness.

**Convergence** (Thm. 3.2). $X_n \xrightarrow{d} X$ together with uniform integrability implies $\Pi_\varepsilon(X_n) \to \Pi_\varepsilon(X)$.

**Limit theorem** (Thm. 4.2). If $H(x) = -\ln S_X(x)$ satisfies

$$\lim_{x \to \infty} H'(x) = \infty,\qquad \lim_{x \to \infty} \frac{H''(x)}{(H'(x))^2} = 0,$$

then $\lim_{\varepsilon \to 0} \Pi_\varepsilon(X) = e \approx 2.71828$.

## Closed-form PELVE values

| Distribution | $\Pi_\varepsilon(X)$ | Range |
|---|---|---|
| $R[a, b]$ | $2$ | $\varepsilon \leq 1/2$ |
| $\mathrm{Pareto}(k, x_0)$ | $\left(\dfrac{k}{k - 1}\right)^k$ | $k > 1$ |
| $\mathrm{Exp}(\lambda)$ | $e$ | for all $\lambda > 0$ |
| $\mathrm{Gamma}(\alpha, \beta, x_0)$ | $e \cdot (1 + O(1/L^2)),\; L = -\ln\varepsilon$ | $\varepsilon \to 0$ |
| $W(k, 1)$ | $e \cdot (1 + O(1/L))$ | $\varepsilon \to 0$ |
| $N(0, \sigma^2)$ | $\to e$ | $\varepsilon \to 0$ |
| $LN(0, \sigma^2)$ | $\to e$ | $\varepsilon \to 0$ |

## Numerical implementation

The defining equation is rewritten in terms of the survival function:

$$\int_0^1 S_X^{-1}(c\varepsilon t)\,dt = S_X^{-1}(\varepsilon).$$

Using $S_X^{-1}$ (SciPy's `stats.dist.isf`) instead of $F_X^{-1}(1 - \cdot)$ is crucial for numerical stability at small $\varepsilon$, where $1 - \varepsilon$ and $1 - c\varepsilon$ collapse to the same double-precision value. The root in $c$ on $[1, 1/\varepsilon]$ is found via Brent's method on the integral evaluated by adaptive quadrature.

## Repository layout

```
risk-measures/
├── src/pelve.py              # core module
├── notebooks/                # original research notebooks
├── tests/test_pelve.py       # pytest checks of closed forms and limits
├── paper/                    # thesis (PDF + LaTeX)
├── requirements.txt
└── LICENSE
```

## Install and run

```bash
git clone https://github.com/salimov-arseny/risk-measures.git
cd risk-measures
pip install -r requirements.txt
pytest tests/ -v
```

Usage example:

```python
from src.pelve import pelve_normal, pelve_pareto, pelve_pareto_closed_form

result = pelve_normal(eps=0.01)
print(result.c)            # ≈ 2.5768

print(pelve_pareto(eps=1e-3, k=4).c)     # ≈ 3.16
print(pelve_pareto_closed_form(k=4))     # (4/3)^4 ≈ 3.16
```

## References

Full bibliography (20 entries) in `paper/thesis.tex`. Key works:

- Li H., Wang R. (2021). *PELVE: Probability Equivalent Level of VaR and ES.* Journal of Econometrics, 234(1), 353–370.
- McNeil A., Frey R., Embrechts P. (2015). *Quantitative Risk Management.* Princeton University Press.
- BCBS (2016). *Minimum capital requirements for market risk.* Basel Committee on Banking Supervision.
- Falin G.I. (1994). *Mathematical analysis of insurance risks.* (in Russian).

## License

[MIT](LICENSE)
