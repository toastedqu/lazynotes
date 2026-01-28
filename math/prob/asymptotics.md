---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---
# Asymptotics
Study notes from {cite:t}`handbook_math` and {cite:t}`prob_lifesaver`.

## Inequalities
### Boole's Inequality (Union Bound)
- **What**: The probability that $\ge$1 event happens is no more than the sum of individual probabilities.
- **Why**: Overlaps can only reduce the total.
- **How**: For events $A_1,\dots,A_n$:
$$
P\left(\bigcup_i A_i\right)\leq \sum_i P(A_i)
$$

&nbsp;

### Markov's Inequality
- **What**: A nonnegative RV can't be large often, unless its mean is large.
- **How**: Let $X\ge0,a>0$:
$$
P(X\ge a)\le\frac{E[X]}{a}
$$

```{tip} Derivation
:class: dropdown
Let $\bm{1}_A$ be the indicator of event A.

Since $X\ge a\bm{1}_{\{X\ge a\}}$,

$$
E[X]\ge E[a\bm{1}_{\{X\ge a\}}]=aP(X\ge a)
$$
```

&nbsp;

### Chebyshev's Inequality
- **What**: Deviations of size $a$ are rare unless STD is at least on the scale of $a$.
- **How**: Let $\mu=E[X],\sigma^2=Var[X], a>0$:
$$
P(|X-\mu|\ge a)\le\frac{\sigma^2}{a^2}
$$

```{tip} Derivation
:class: dropdown
Apply Markov to nonnegative RV $(X-\mu)^2$:

$$
P(|X-\mu|\ge a)=P((X-\mu)^2\ge a^2)\le\frac{E[(X-\mu)^2]}{a^2}=\frac{\sigma^2}{a^2}
$$
```

&nbsp;

### Cauchy-Schwarz Inequality
- **What**: The size of the alignment between 2 RVs is limited by the sizes of themselves (measured by second moments).
- **How**: For square-integrable $X,Y$,
$$
|E[XY]|\le \sqrt{E[X^2]}\sqrt{E[Y^2]}
$$

```{tip} Derivation
:class: dropdown
1. $\forall t\in\mathbb{R}$:

$$
0\le E[(X-tY)^2]=E[X^2]-2tE[XY]+t^2E[Y^2]
$$

2. Quadratic formula:

$$
a>0,at^2+bt+c\ge 0 \Leftrightarrow b^2-4ac\le 0
$$

3. $a=E[Y^2],b=-2E[XY],c=E[X^2]$:

$$\begin{align*}
(-2E[XY])^2-4E[Y^2]E[X^2]&\le 0 \\
(E[XY])^2&\le E[X^2]E[Y^2] \\
|E[XY]|&\le \sqrt{E[X^2]}\sqrt{E[Y^2]}
\end{align*}$$
```

&nbsp;

### Jensen's Inequality
- **What**: A convex function punishes variability.
    - "Applying it after averaging" is better than "Averaging after applying it".
- **How**: If $f(\cdot)$ is convex and $X$ is integrable:
$$
f(E[X])\le E[f(X)]
$$

```{tip} Derivation
:class: dropdown
Convexity - Supporting hyperplane:
- A convex function ALWAYS lies above at least one tangent hyperplane.

1. If $f(\cdot)$ is convex, at any point $x_0$ in its domain, there exists a slope/subgradient $m$ such that:

$$
\forall x: f(x)\ge f(x_0)+m(x-x_0)
$$

2. Let $x_0=E[X]$, then

$$
f(X)\ge f(E[X])+m(X-E[X])
$$

3. Take expectation:

$$
E[f(X)]\ge f(E[X])+m(E[X]-E[X])=f(E[X])
$$
```

&nbsp;

## Law of Large Numbers
- **What**: The avg outcome gets closer to the true mean if a random process is repeated many times.
    - **Weak LLN**: Gets close in probability (high chance).
    - **Strong LLN**: Gets close certainly ($p=1$).
- **Why**: Randomness cancels out when you avg lots of independent trials.
- **How**:
    - Define:
        - $X_1,X_2,\dots$: i.i.d. RVs w/ $E[X_i]=\mu,\text{Var}[X_i]=\sigma^2$.
        - $\bar{X}_n=\frac 1 n\sum_{i=1}^nX_i$: Sample mean.
    - Weak LLN:
$$
\bar{X}_n\xrightarrow{P}\mu\Longleftrightarrow\forall\varepsilon>0:\lim_{n\rightarrow\infty}P(|\bar{X}_n-\mu|>\varepsilon)=0
$$
    - Strong LLN:
$$
\bar{X}_n\xrightarrow{P=1}\mu
$$

```{tip} Derivation
:class: dropdown
Weak LLN:
1. Variance of sum:
$$
Var\left[\sum_{i=1}^nX_i\right]=\sum_{i=1}^nVar[X_i]=n\sigma^2
$$

2. Variance of sample mean:
$$
Var(\bar{X}_n)=Var\left(\frac 1 n\sum_{i=1}^nX_i\right)=\frac{1}{n^2}Var\left(\sum_{i=1}^nX_i\right)=\frac{\sigma^2}{n}
$$

3. Chebyshev's Inequality:
$$
P(|\bar{X}_n-\mu|\ge\varepsilon)\le\frac{Var[\bar{X}_n]}{\varepsilon^2}=\frac{\sigma^2}{n\varepsilon^2}
$$
```

&nbsp;

## Central Limit Theorem
- **What**: The sum of many i.i.d. RVs with finite mean & variance approximates the Normal distribution, regardless of their original distribution.
- **Why**: Ask God.
- **How**:
    - Define:
        - $X_1,X_2,\dots$: i.i.d. RVs w/ $E[X_i]=\mu,\text{Var}[X_i]=\sigma^2\in(0,\infty)$.
        - $\bar{X}_n=\frac 1 n\sum_{i=1}^nX_i$: Sample mean.
    - CLT: As $n\rightarrow\infty$,
$$\begin{align*}
&\text{Sum}:  &&\frac{S_n-n\mu}{\sigma\sqrt{n}}\xrightarrow{d}N(0,1) \\
&\text{Mean}: &&\frac{\sqrt{n}(\bar{X}_n-\mu)}{\sigma}\xrightarrow{d}N(0,1)
\end{align*}$$

```{tip} Derivation
:class: dropdown
Notations:
- Standardized variable: $Y_i:=\frac{X_i-\mu}{\sigma}$
- Standardized sum: $Z_n:=\frac{1}{\sqrt n}\sum_{i=1}^nY_i$
- CLT: $Z_n\xrightarrow{d}N(0,1)$

1. Characteristic Function:
    1. Lévy's continuity theorem:
    $$
    \forall t,\varphi_{Z_n}(t)\rightarrow\varphi_Z(t)\ \Longleftrightarrow\ Z_n\xrightarrow{d} Z
    $$
    2. CF of $N(0,1)$: $\exp\left(-\frac{t^2}{2}\right)$
    3. We aim to prove: $\varphi_{Z_n}(t)\rightarrow\exp\left(-\frac{t^2}{2}\right)$
2. Independence of $Y_i$:
$$\begin{align*}
\varphi_{Z_n}(t)&=E\left[\exp\left(it\frac{1}{\sqrt n}\sum_{i=1}^nY_i\right)\right] \\
&=\prod_{i=1}^nE\left[\exp\left(it\frac{Y_i}{\sqrt n}\right)\right] \\
&=\left[\varphi_Y\left(\frac{t}{\sqrt n}\right)\right]^n
\end{align*}$$
3. Taylor Expansion near 0:
    1. If a function $f$ has a continuous second derivative near 0, then as $u\rightarrow0$,
    $$
    f(u)=f(0)+f'(0)u+\frac{1}{2}f''(0)u^2+o(u^2)
    $$
    where $o(u^2)$ means a negligible term compared to $u^2$ as $u\rightarrow0$.
    2. Since $E[Y]=0,E[Y^2]=1$, as $u\rightarrow0$,
    $$
    \varphi_Y(u)=E[e^{iuY}]=1+0\cdot u-\frac{u^2}{2}+o(u^2)
    $$
    3. Let $u=\frac{t}{\sqrt n}$:
    $$
    \varphi_Y(\frac{t}{\sqrt n})=1-\frac{t^2}{2n}+o(\frac{1}{n})
    $$
    4. Back to Step 2:
    $$
    \varphi_{Z_n}(t)=\left(1-\frac{t^2}{2n}+o(\frac{1}{n})\right)^n
    $$
    5. Classic limit:
    $$\begin{align*}
    &\lim_{n\rightarrow\infty}(1+\frac{a}{n})^n=e^a \\
    &\lim_{n\rightarrow\infty}(1-\frac{t^2}{2n})=e^{-\frac{t^2}{2}} \\
    &\varphi_{Z_n}(t)\rightarrow e^{-\frac{t^2}{2}}
    \end{align*}$$
    Back to Step 1.
```