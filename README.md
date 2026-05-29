# Вероятностно эквивалентные меры риска (PELVE)

Численная реализация и теоретическое исследование показателя **PELVE** (*Probability Equivalent Level of VaR and ES*) для распределений, используемых в страховой и финансовой математике.

Проект сопровождает курсовую работу:

> Салимов А.Е. *Вероятностно эквивалентные меры риска.* МГУ им. М.В. Ломоносова, механико-математический факультет, кафедра теории вероятностей, 2025. Научный руководитель — проф., д.ф.-м.н. Г.И. Фалин.

PDF курсовой и исходник LaTeX лежат в [`paper/`](paper/).

## Постановка задачи

Пусть $X$ — положительная абсолютно непрерывная случайная величина, моделирующая суммарный страховой иск или финансовые потери, с функцией распределения $F_X$ и функцией выживания $S_X = 1 - F_X$. Для уровня доверия $p \in [0, 1)$ определены:

$$\mathrm{VaR}_p(X) = F_X^{-1}(p),\qquad \mathrm{ES}_p(X) = \frac{1}{1 - p}\int_p^1 F_X^{-1}(q)\,dq.$$

**PELVE** $\Pi_\varepsilon(X)$ — это константа $c \in [1, 1/\varepsilon]$, такая что

$$\mathrm{ES}_{1 - c\varepsilon}(X) = \mathrm{VaR}_{1 - \varepsilon}(X).$$

Показатель введён в работе Li & Wang (2021) для оценки последствий перехода Базельского комитета от $\mathrm{VaR}_{0.99}$ к $\mathrm{ES}_{0.975}$. Пороговое значение — $c = 2.5$: если $\Pi_\varepsilon(X) > 2.5$, регуляторная нагрузка вырастет.

## Основные результаты

**Существование и единственность** (утв. 2.1). Если $\mathbb{E}X < \infty$ и $\mathbb{E}X \leq \mathrm{VaR}_{1-\varepsilon}(X)$, то решение уравнения единственно.

**Инвариантность** (теор. 3.1). Для любых $\lambda > 0,\, a \in \mathbb{R}$: $\Pi_\varepsilon(\lambda X + a) = \Pi_\varepsilon(X)$. Выпуклое вниз преобразование не уменьшает PELVE — то есть показатель чувствителен к тяжести хвоста.

**Сходимость** (теор. 3.2). $X_n \xrightarrow{d} X$ + равномерная интегрируемость $\Rightarrow \Pi_\varepsilon(X_n) \to \Pi_\varepsilon(X)$.

**Предельная теорема** (теор. 4.2). Если $H(x) = -\ln S_X(x)$ удовлетворяет

$$\lim_{x \to \infty} H'(x) = \infty,\qquad \lim_{x \to \infty} \frac{H''(x)}{(H'(x))^2} = 0,$$

то $\lim_{\varepsilon \to 0} \Pi_\varepsilon(X) = e \approx 2.71828$.

## Аналитические значения PELVE

| Распределение | $\Pi_\varepsilon(X)$ | Условие |
|---|---|---|
| $R[a, b]$ | $2$ | $\varepsilon \leq 1/2$ |
| $\mathrm{Pareto}(k, x_0)$ | $\left(\dfrac{k}{k - 1}\right)^k$ | $k > 1$ |
| $\mathrm{Exp}(\lambda)$ | $e$ | для всех $\lambda > 0$ |
| $\mathrm{Gamma}(\alpha, \beta, x_0)$ | $e \cdot (1 + O(1/L^2)),\; L = -\ln\varepsilon$ | $\varepsilon \to 0$ |
| $W(k, 1)$ | $e \cdot (1 + O(1/L))$ | $\varepsilon \to 0$ |
| $N(0, \sigma^2)$ | $\to e$ | $\varepsilon \to 0$ |
| $LN(0, \sigma^2)$ | $\to e$ | $\varepsilon \to 0$ |

Численные значения для практических $\varepsilon$:

| $\varepsilon$ | $R[0,1]$ | $\mathrm{Par}(2,1)$ | $\mathrm{Par}(4,1)$ | $\mathrm{Par}(8,1)$ | $\mathrm{Exp}$ | $N(0,1)$ | $\Gamma(1,2)$ | $\Gamma(1,10)$ | $LN(0,1)$ |
|---|---|---|---|---|---|---|---|---|---|
| 0.100 | 2 | 4 | 3.16 | 2.91 | $e$ | 2.46 | 2.65 | 2.55 | 3.23 |
| 0.050 | 2 | 4 | 3.16 | 2.91 | $e$ | 2.51 | 2.67 | 2.59 | 3.19 |
| 0.010 | 2 | 4 | 3.16 | 2.91 | $e$ | 2.58 | 2.69 | 2.64 | 3.13 |
| 0.005 | 2 | 4 | 3.16 | 2.91 | $e$ | 2.59 | 2.70 | 2.65 | 3.11 |

## Численная реализация

Основное уравнение перепишем через функцию выживания:

$$\int_0^1 S_X^{-1}(c\varepsilon t)\,dt = S_X^{-1}(\varepsilon).$$

Использование $S_X^{-1}$ вместо $F_X^{-1}(1 - \cdot)$ принципиально для устойчивости при малых $\varepsilon$: иначе $1 - \varepsilon$ и $1 - c\varepsilon$ совпадают по машинному представлению double. В библиотеке SciPy для этого есть функция `stats.dist.isf`.

Решение находится методом Брента: интегрируем по $t$ с помощью `scipy.integrate.quad`, далее `scipy.optimize.brentq` ищет корень по $c$ на $[1, 1/\varepsilon]$.

## Структура репозитория

```
risk-measures/
├── src/pelve.py              # основной модуль
├── notebooks/                # исходные ноутбуки исследования
├── tests/test_pelve.py       # pytest-проверки формул и асимптотик
├── paper/                    # курсовая (PDF + LaTeX)
├── requirements.txt
└── LICENSE
```

## Установка и запуск

```bash
git clone https://github.com/salimov-arseny/risk-measures.git
cd risk-measures
pip install -r requirements.txt
pytest tests/ -v
```

Пример использования:

```python
from src.pelve import pelve_normal, pelve_pareto, pelve_pareto_closed_form

result = pelve_normal(eps=0.01)
print(result.c)            # ≈ 2.5768

print(pelve_pareto(eps=1e-3, k=4).c)     # ≈ 3.16
print(pelve_pareto_closed_form(k=4))     # (4/3)^4 ≈ 3.16
```

## Литература

Полный список (20 источников) — в `paper/thesis.tex`. Ключевые работы:

- Li H., Wang R. (2021). *PELVE: Probability Equivalent Level of VaR and ES.* Journal of Econometrics, 234(1), 353–370.
- McNeil A., Frey R., Embrechts P. (2015). *Quantitative Risk Management.* Princeton University Press.
- BCBS (2016). *Minimum capital requirements for market risk.* Basel Committee on Banking Supervision.
- Фалин Г.И. (1994). *Математический анализ рисков в страховании.* Российский юридический издательский дом.

## Лицензия

[MIT](LICENSE)
