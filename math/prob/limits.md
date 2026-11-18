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
# Inequalities & Limit Theorems
Study notes from {cite:t}`math` and {cite:t}`prob_lifesaver`.
Tail inequalities → finite-sample concentration → asymptotic limit theorems.

Notations:
- $X$: Random variable.
- $Y$: Second random variable.
- $X_1,\dots,X_n$: Random variables, i.i.d. unless stated.
- $\mu=E[X]$: Mean.
- $\sigma^2=Var[X]$: Variance.
- $\bar{X}_n=\frac{1}{n}\sum_{i=1}^nX_i$: Sample mean.
- $S_n=\sum_{i=1}^nX_i$: Sum.
- $\varepsilon>0$: Deviation threshold.

Overrides Tier 1: $n$ = #samples here (not #features).

&nbsp;

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

## Concentration

### Chernoff Bound
- **What**: Markov's inequality on $e^{\lambda X}$, minimized over $\lambda>0$.
- **Why**: Markov & Chebyshev tails decay only polynomially.
    - Sums of independent light-tailed RVs have exponentially small tails → need a bound that uses all moments.
- **How**:
    1. $X\ge a\Leftrightarrow e^{\lambda X}\ge e^{\lambda a}$ ← $e^{\lambda x}$ increasing for $\lambda>0$.
    2. Markov on $e^{\lambda X}$ → bound = $e^{-\lambda a}\times$ [MGF](basics.md#moment-generating-function).
    3. Independent sum → MGF factorizes → product of per-term MGFs.
    4. Pick the best $\lambda$ → tightest exponential rate.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\lambda$: Tilt parameter.
    - $a$: Threshold.
    - $M_X(\lambda)=E[e^{\lambda X}]$: MGF of $X$.
    - $p_i$: Success prob of Bernoulli $X_i$.
    - $\mu_S=E[S_n]=\sum_{i=1}^np_i$: Mean of the Bernoulli sum.
    - $\delta$: Relative deviation.

Generic bound:

$$\begin{align*}
&\text{Upper tail}: &&P(X\ge a)\le\inf_{\lambda>0}e^{-\lambda a}M_X(\lambda) \\
&\text{Lower tail}: &&P(X\le a)\le\inf_{\lambda>0}e^{\lambda a}M_X(-\lambda)
\end{align*}$$

i.i.d. mean, $a>\mu$:

$$
P(\bar X_n\ge a)\le e^{-nI(a)},\qquad I(a)=\sup_{\lambda\in\mathbb{R}}\left[\lambda a-\log M_X(\lambda)\right]
$$

- $I(a)$: Rate function = Legendre–Fenchel transform of the cumulant generating function $\log M_X$.
- Bernoulli($p$), $p<a<1$ → $I(a)=\text{KL}(a\,\|\,p)=a\log\frac{a}{p}+(1-a)\log\frac{1-a}{1-p}$.

Poisson trials (multiplicative form), independent $X_i\sim\text{Bernoulli}(p_i)$:
- Upper tail, $\delta>0$:

    $$
    P(S_n\ge(1+\delta)\mu_S)\le\left(\frac{e^\delta}{(1+\delta)^{1+\delta}}\right)^{\mu_S}\le\exp\left(-\frac{\delta^2\mu_S}{2+\delta}\right)
    $$

- Lower tail, $0<\delta<1$:

    $$
    P(S_n\le(1-\delta)\mu_S)\le\left(\frac{e^{-\delta}}{(1-\delta)^{1-\delta}}\right)^{\mu_S}\le\exp\left(-\frac{\delta^2\mu_S}{2}\right)
    $$

- Two-sided, $0<\delta<1$:

    $$
    P(|S_n-\mu_S|\ge\delta\mu_S)\le2\exp\left(-\frac{\delta^2\mu_S}{3}\right)
    $$
```

```{tip} Derivation
:class: dropdown
Generic bound:
1. $\forall\lambda>0$: $P(X\ge a)=P(e^{\lambda X}\ge e^{\lambda a})\le e^{-\lambda a}E[e^{\lambda X}]$ ← Markov.
2. Holds $\forall\lambda>0$ → take the inf.
3. Lower tail: apply step 1 to $-X$.
4. i.i.d. → $M_{S_n}(\lambda)=M_X(\lambda)^n$:

    $$
    P(\bar X_n\ge a)=P(S_n\ge na)\le\exp\left(-n\left[\lambda a-\log M_X(\lambda)\right]\right)
    $$

5. $\lambda\le0$ adds nothing for $a>\mu$ ← Jensen: $\log M_X(\lambda)\ge\lambda\mu$ → $\lambda a-\log M_X(\lambda)\le\lambda(a-\mu)\le0$.

Poisson trials:
1. $M_{X_i}(\lambda)=1+p_i(e^\lambda-1)\le\exp\left(p_i(e^\lambda-1)\right)$ ← $1+x\le e^x$.
2. Independence → $M_{S_n}(\lambda)\le\exp\left(\mu_S(e^\lambda-1)\right)$.
3. Generic bound at $a=(1+\delta)\mu_S$:

    $$
    P(S_n\ge(1+\delta)\mu_S)\le\exp\left(\mu_S(e^\lambda-1)-\lambda(1+\delta)\mu_S\right)
    $$

4. Minimizer $\lambda=\ln(1+\delta)$ → $\left(e^\delta/(1+\delta)^{1+\delta}\right)^{\mu_S}$.
5. Simplify: $\ln(1+\delta)\ge\frac{2\delta}{2+\delta}$ → $(1+\delta)\ln(1+\delta)-\delta\ge\frac{\delta^2}{2+\delta}$.
6. Lower tail: same steps w/ $\lambda=-\ln(1-\delta)$, then $(1-\delta)\ln(1-\delta)\ge-\delta+\frac{\delta^2}{2}$.
7. Two-sided: $\delta<1$ → $2+\delta<3$ → both tails $\le e^{-\delta^2\mu_S/3}$ → union bound.
```

```{note} Example
:class: dropdown
Gaussian, $X\sim N(0,\sigma^2)$, $M_X(\lambda)=e^{\lambda^2\sigma^2/2}$:
- $\inf_\lambda e^{-\lambda a+\lambda^2\sigma^2/2}$ at $\lambda=a/\sigma^2$ → $P(X\ge a)\le e^{-a^2/(2\sigma^2)}$.
- $a=3\sigma$: Chernoff $\approx0.011$ · Chebyshev (two-sided) $\le1/9\approx0.111$ · true $\approx0.00135$.

Rare events, $n=1000$ i.i.d. Bernoulli($0.1$) → $\mu_S=100$. Bound $P(S_n\ge150)$, i.e., $\delta=0.5$:

| Bound | Value |
|:--|:--|
| Exact binomial tail | $4.5\times10^{-7}$ |
| KL Chernoff $e^{-n\,\text{KL}(0.15\Vert0.1)}$ | $4.9\times10^{-6}$ |
| Multiplicative $\left(e^{0.5}/1.5^{1.5}\right)^{100}$ | $2.0\times10^{-5}$ |
| Simplified $e^{-\delta^2\mu_S/(2+\delta)}=e^{-10}$ | $4.5\times10^{-5}$ |
| Hoeffding $e^{-2\cdot50^2/1000}=e^{-5}$ | $6.7\times10^{-3}$ |
| Chebyshev $90/50^2$ | $3.6\times10^{-2}$ |
```

```{attention} Q&A
:class: dropdown
*Why exponential and not polynomial decay?*

- Markov uses $E[X]$. Chebyshev uses $E[X^2]$.
- Chernoff uses the MGF → all moments at once → exponential decay for light tails.

*Is the exponent tight?*

- Yes, for i.i.d. sums. **Cramér's theorem**: $\frac1n\log P(\bar X_n\ge a)\to-I(a)$ for $a>\mu$.
- Chernoff loses only a subexponential prefactor.
- → Basis of large deviations theory.

*Cons?*

- Needs $M_X(\lambda)<\infty$ for some $\lambda>0$ → ❌ heavy tails (log-normal, Pareto, Student-t).
- Needs independence (or other control of the joint MGF).
- Needs the MGF, or a bound on it, per distribution → Hoeffding & Bernstein are ready-made MGF bounds.

*Multiplicative vs additive (Hoeffding) form?*

- Additive exponent: $2\delta^2\mu_S^2/n=2\delta^2\mu_Sp$, $p=\mu_S/n$.
- Multiplicative exponent: $\delta^2\mu_S/(2+\delta)$.
- Small $p$ → multiplicative wins by a factor $\approx1/p$ ← rare events have small variance.

*Typical uses?*

- Randomized algorithms: hashing, load balancing, sketches.
- Error counts on a test set; sample size to estimate a small probability.
- Union bound over many events → exponential tail → cost only $\log(\#\text{events})$.
```

&nbsp;

### Sub-Gaussian Random Variable
- **What**: RV whose centered MGF is at most a Gaussian's MGF.
- **Why**: Gaussian-type concentration w/o Gaussianity.
    - One class covers Gaussian, bounded & Rademacher RVs.
    - Closed under independent sums → bounds for $\bar X_n$, maxima, random matrices, generalization.
- **How**:
    - MGF bound → Chernoff → Gaussian-shaped tail.
    - Independent sum → MGFs multiply → variance proxies add.
    - **Variance proxy** $\tau^2$ plays the role of $\sigma^2$, but only bounds it from above.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\tau^2$: Variance proxy.
    - $N$: #RVs in a maximum.
    - $K$: Scale constant in equivalent definitions.

Definition:

$$
E\left[e^{\lambda(X-E[X])}\right]\le e^{\lambda^2\tau^2/2},\quad\forall\lambda\in\mathbb{R}
$$

Properties:
- Tail:

    $$
    P(X-E[X]\ge\varepsilon)\le e^{-\varepsilon^2/(2\tau^2)},\qquad P(|X-E[X]|\ge\varepsilon)\le2e^{-\varepsilon^2/(2\tau^2)}
    $$

- Variance: $\sigma^2\le\tau^2$.
- Scaling: $cX$ is sub-Gaussian w/ $c^2\tau^2$.
- Independent sum: $X_i$ w/ $\tau_i^2$ → $S_n$ w/ $\sum_{i=1}^n\tau_i^2$.
- i.i.d. mean:

    $$
    P(|\bar X_n-\mu|\ge\varepsilon)\le2\exp\left(-\frac{n\varepsilon^2}{2\tau^2}\right)
    $$

- Maximum, $E[X_i]=0$, any dependence:

    $$
    E\left[\max_{i\le N}X_i\right]\le\tau\sqrt{2\ln N}
    $$

- Equivalent definitions for centered $X$; $K$ changes only by universal constant factors:
    - Tail: $P(|X|\ge\varepsilon)\le2e^{-\varepsilon^2/K^2}$.
    - Moments: $(E|X|^q)^{1/q}\le K\sqrt q$, $\forall q\ge1$.
    - MGF of square: $E[e^{X^2/K^2}]\le2$.
```

```{tip} Derivation
:class: dropdown
1. Tail: Chernoff → $P(X-E[X]\ge\varepsilon)\le\inf_\lambda e^{-\lambda\varepsilon+\lambda^2\tau^2/2}$. Minimizer $\lambda=\varepsilon/\tau^2$ → $e^{-\varepsilon^2/(2\tau^2)}$.
2. Variance: expand both sides at $\lambda\to0$:

    $$
    1+\frac{\lambda^2\sigma^2}{2}+o(\lambda^2)\le1+\frac{\lambda^2\tau^2}{2}+o(\lambda^2)\ \Rightarrow\ \sigma^2\le\tau^2
    $$

3. Sum: independence → $E[e^{\lambda(S_n-E[S_n])}]=\prod_iE[e^{\lambda(X_i-E[X_i])}]\le e^{\lambda^2\sum_i\tau_i^2/2}$.
4. i.i.d. mean: $\bar X_n-\mu=\frac1n(S_n-n\mu)$ → proxy $\frac{n\tau^2}{n^2}=\frac{\tau^2}{n}$ → step 1.
5. Maximum, $\lambda>0$:

    $$
    e^{\lambda E[\max_iX_i]}\le E[e^{\lambda\max_iX_i}]\le\sum_{i=1}^NE[e^{\lambda X_i}]\le Ne^{\lambda^2\tau^2/2}
    $$

    - 1st step ← Jensen. 2nd step ← max $\le$ sum of nonnegative terms.
    - → $E[\max_iX_i]\le\frac{\ln N}{\lambda}+\frac{\lambda\tau^2}{2}$. Minimizer $\lambda=\sqrt{2\ln N}/\tau$ → $\tau\sqrt{2\ln N}$.
```

```{note} Example
:class: dropdown
✅ Sub-Gaussian:
- $N(\mu,\sigma^2)$: equality in the definition → $\tau^2=\sigma^2$.
- Rademacher ($\pm1$ w/ prob $\frac12$): $E[e^{\lambda X}]=\cosh\lambda=\sum_k\frac{\lambda^{2k}}{(2k)!}\le\sum_k\frac{\lambda^{2k}}{2^kk!}=e^{\lambda^2/2}$ → $\tau^2=1$.
    - ← $(2k)!\ge2^kk!$.
- Bounded in $[a,b]$: $\tau^2=\frac{(b-a)^2}{4}$ ← Hoeffding's lemma ([Hoeffding's Inequality](#hoeffdings-inequality)).

❌ Not sub-Gaussian:
- Exponential, Poisson, $\chi^2$ → **sub-exponential** (tail $e^{-c\varepsilon}$).
- Student-t, Pareto, log-normal → heavy-tailed (no MGF for $\lambda>0$).
```

```{attention} Q&A
:class: dropdown
*Why is the variance proxy not the variance?*

- $\sigma^2\le\tau^2$. Equality for the Gaussian.
- Bernoulli($p$), small $p$: smallest valid $\tau^2\gg p(1-p)$ ← the MGF bound must cover the rare jump of size $\approx1$ for every $\lambda$.
- → Sub-Gaussian bounds can be loose for small-variance RVs → Bernstein.

*Sub-Gaussian vs CLT?*

- CLT: Gaussian *shape* as $n\to\infty$. No finite-$n$ guarantee.
- Sub-Gaussian: Gaussian-type tail *upper bound* for every $n$. No shape claim.

*Why does $E[\max]\le\tau\sqrt{2\ln N}$ matter?*

- Max of $N$ noisy estimates grows only like $\sqrt{\ln N}$.
- → Union over many hypotheses/arms costs only $\log N$ samples.
- → Size of the winner's curse when you select the best of $N$ models on noisy scores.

*Squares of sub-Gaussians?*

- Sub-exponential (e.g., $\chi^2$) → Bernstein-type bounds, not Hoeffding-type.
```

&nbsp;

### Hoeffding's Inequality
- **What**: Sub-Gaussian tail bound for the mean of independent bounded RVs.
- **Why**: Need a finite-$n$, distribution-free guarantee for $\bar X_n$ from the range alone.
- **How**:
    1. Bounded in $[a_i,b_i]$ → sub-Gaussian w/ $\tau_i^2=(b_i-a_i)^2/4$ ← **Hoeffding's lemma**.
    2. Independence → variance proxies add.
    3. Chernoff → Gaussian-shaped tail w/ worst-case variance.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $a_i$: Lower bound: $a_i\le X_i$ a.s.
    - $b_i$: Upper bound: $X_i\le b_i$ a.s.
    - $\delta$: Failure probability.

Independent, not nec. identical, $X_i\in[a_i,b_i]$:

$$
P\left(\bar X_n-E[\bar X_n]\ge\varepsilon\right)\le\exp\left(-\frac{2n^2\varepsilon^2}{\sum_{i=1}^n(b_i-a_i)^2}\right)
$$

- Lower tail: same bound. Two-sided: $\times2$.

i.i.d. in $[a,b]$:

$$
P(|\bar X_n-\mu|\ge\varepsilon)\le2\exp\left(-\frac{2n\varepsilon^2}{(b-a)^2}\right)
$$

i.i.d. in $[0,1]$, w/ prob $\ge1-\delta$:
- Width at fixed $n$:

    $$
    |\bar X_n-\mu|\le\sqrt{\frac{\ln(2/\delta)}{2n}}
    $$

- Sample complexity for accuracy $\varepsilon$:

    $$
    n\ge\frac{\ln(2/\delta)}{2\varepsilon^2}
    $$

**Hoeffding's lemma**: $X\in[a,b]$ a.s. →

$$
E\left[e^{\lambda(X-E[X])}\right]\le e^{\lambda^2(b-a)^2/8}
$$
```

```{tip} Derivation
:class: dropdown
Hoeffding's lemma:
1. WLOG $E[X]=0$ → $a\le0\le b$.
2. Convexity of $e^{\lambda x}$ on $[a,b]$:

    $$
    e^{\lambda x}\le\frac{b-x}{b-a}e^{\lambda a}+\frac{x-a}{b-a}e^{\lambda b}
    $$

3. Take expectation, $E[X]=0$. Let $p=\frac{-a}{b-a}\in[0,1]$, $h=\lambda(b-a)$:

    $$
    E[e^{\lambda X}]\le(1-p)e^{\lambda a}+pe^{\lambda b}=e^{L(h)},\qquad L(h)=-ph+\ln(1-p+pe^h)
    $$

4. $L(0)=0$. $L'(0)=0$. $L''(h)=q(1-q)\le\frac14$, w/ $q=\frac{pe^h}{1-p+pe^h}\in[0,1]$.
5. Taylor w/ Lagrange remainder → $L(h)\le\frac{h^2}{8}=\frac{\lambda^2(b-a)^2}{8}$.

Hoeffding:
1. Chernoff + independence + lemma, $t=n\varepsilon$:

    $$
    P(S_n-E[S_n]\ge t)\le e^{-\lambda t}\prod_{i=1}^nE\left[e^{\lambda(X_i-E[X_i])}\right]\le\exp\left(-\lambda t+\frac{\lambda^2}{8}\sum_{i=1}^n(b_i-a_i)^2\right)
    $$

2. Minimizer $\lambda=4t/\sum_i(b_i-a_i)^2$ → $\exp\left(-2t^2/\sum_i(b_i-a_i)^2\right)$.
3. $t=n\varepsilon$ → mean form.
```

```{note} Example
:class: dropdown
Test-set error of 1 fixed classifier $h$, 0-1 loss $\in[0,1]$:
- $R(h)$: true error. $\hat R_n(h)$: error on $n$ i.i.d. test samples.
- $P(|\hat R_n(h)-R(h)|\ge\varepsilon)\le2e^{-2n\varepsilon^2}$.
- $\varepsilon=0.01$, $\delta=0.05$ → $n\ge\frac{\ln40}{2\times10^{-4}}\approx18{,}445$.

Finite class $\mathcal{H}$, all $h$ at once:
- Union bound → $n\ge\frac{\ln(2|\mathcal{H}|/\delta)}{2\varepsilon^2}$.
- → Cost of $|\mathcal{H}|$ is only logarithmic.
```

````{important} Code
:class: dropdown
```python
import math

class BernoulliTails:
    ## Upper tail P(X̄_n - p >= eps) for i.i.d. Bernoulli(p): exact value vs bounds
    def __init__(self, n, p):
        self.n, self.p = n, p

    def __call__(self, eps):
        n, p, q = self.n, self.p, self.p + eps
        ## smallest count k with k/n >= q; tolerance absorbs float error
        k0 = math.ceil(n * q - 1e-9)
        exact = sum(math.comb(n, k) * p**k * (1 - p)**(n - k) for k in range(k0, n + 1))
        kl = q * math.log(q / p) + (1 - q) * math.log((1 - q) / (1 - p))
        return {
            "exact": exact,
            "chebyshev": p * (1 - p) / (n * eps**2),  ## two-sided bound, polynomial in n
            "hoeffding": math.exp(-2 * n * eps**2),   ## uses range [0,1] only
            "chernoff_kl": math.exp(-n * kl),         ## optimal exponential rate
        }

## Example
print(BernoulliTails(100, 0.5)(0.1))
## exact≈0.028, chebyshev=0.25, hoeffding≈0.135, chernoff_kl≈0.134 -> max variance: Hoeffding ≈ KL
print(BernoulliTails(1000, 0.1)(0.05))
## exact≈4.5e-07, chebyshev=0.036, hoeffding≈6.7e-03, chernoff_kl≈4.9e-06 -> small variance: Hoeffding loose
```
````

```{attention} Q&A
:class: dropdown
*Pros?*

- Distribution-free: needs only range & independence.
- Non-asymptotic: valid for every $n$.
- Exponential tail → $n\propto\ln(1/\delta)$.

*Cons?*

- Ignores variance → loose when $\sigma^2\ll(b-a)^2/4$ → [Bernstein's Inequality](#bernsteins-inequality).
- ❌ Unbounded RVs.
- ❌ Dependent data → martingale version: [Azuma–Hoeffding Inequality](process.md#azuma-hoeffding-inequality).

*Hoeffding vs Chebyshev?*

- Samples for accuracy $\varepsilon$ w/ prob $\ge1-\delta$:
    - Chebyshev: $n\ge\frac{\sigma^2}{\varepsilon^2\delta}$.
    - Hoeffding: $n\ge\frac{(b-a)^2\ln(2/\delta)}{2\varepsilon^2}$.
- $\delta=10^{-6}$ → factor $10^6$ vs $\ln(2\times10^6)\approx14.5$.

*Hoeffding vs CLT?*

- CLT: approximation, no finite-$n$ guarantee.
- Hoeffding = Gaussian tail w/ worst-case variance $(b-a)^2/4$.
- Bernoulli($\frac12$) has exactly that variance → Hoeffding exponent = CLT exponent.

*Why does it fail for the model picked by ERM?*

- ERM selects $h$ using the same data → $\hat R_n(h)$ is no longer a mean of terms independent of $h$.
- → Need a uniform bound over $\mathcal{H}$: union bound, VC, or Rademacher via [McDiarmid's Inequality](#mcdiarmids-inequality).
- Same reason: test set reuse for model selection breaks the guarantee.

*Sampling without replacement?*

- Bound still holds (Hoeffding, 1963).

*Relation to Chernoff?*

- Bernoulli: Pinsker → $\text{KL}(q\|p)\ge2(q-p)^2$.
- → Hoeffding = KL-Chernoff relaxed by Pinsker.
```

&nbsp;

### Bernstein's Inequality
- **What**: Variance-aware exponential tail bound for sums of independent bounded RVs.
- **Why**: Hoeffding uses only the range → loose when $\sigma^2\ll(b-a)^2/4$.
    - e.g., rare events, small $p$, low-noise losses.
- **How**:
    1. Bounded centered terms → higher moments $\le\sigma^2M^{k-2}$.
    2. → MGF $\approx$ Gaussian MGF w/ true variance, inflated by $1/(1-\lambda M/3)$.
    3. Chernoff → 2 regimes:
        - Small $\varepsilon$ → Gaussian tail w/ true $\sigma^2$.
        - Large $\varepsilon$ → exponential tail w/ scale $M$.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $M$: Bound $|X_i-E[X_i]|\le M$ a.s.
    - $v=\sum_{i=1}^nVar[X_i]$: Total variance.
    - $t$: Deviation of the sum.
    - $\delta$: Failure probability.

Independent, not nec. identical:

$$
P(S_n-E[S_n]\ge t)\le\exp\left(-\frac{t^2/2}{v+Mt/3}\right)
$$

- One-sided: only $X_i-E[X_i]\le M$ is needed. Two-sided: $|X_i-E[X_i]|\le M$, $\times2$.

i.i.d. mean, $t=n\varepsilon$, $v=n\sigma^2$:

$$
P(|\bar X_n-\mu|\ge\varepsilon)\le2\exp\left(-\frac{n\varepsilon^2}{2\sigma^2+2M\varepsilon/3}\right)
$$

Regimes:
- $\varepsilon\ll\sigma^2/M$ → $\approx2e^{-n\varepsilon^2/(2\sigma^2)}$: sub-Gaussian w/ the true variance.
- $\varepsilon\gg\sigma^2/M$ → $\approx2e^{-3n\varepsilon/(2M)}$: sub-exponential.

W/ prob $\ge1-\delta$:

$$
|\bar X_n-\mu|\le\sqrt{\frac{2\sigma^2\ln(2/\delta)}{n}}+\frac{2M\ln(2/\delta)}{3n}
$$

- $\sigma^2\lesssim M^2\ln(2/\delta)/n$ → $\frac1n$ term dominates → width $O(1/n)$, not $O(1/\sqrt n)$.
```

```{tip} Derivation
:class: dropdown
1. Let $Y_i=X_i-E[X_i]$, $|Y_i|\le M$, $\sigma_i^2=Var[X_i]$. For $k\ge2$:

    $$
    |E[Y_i^k]|\le E[Y_i^2|Y_i|^{k-2}]\le\sigma_i^2M^{k-2}
    $$

2. Series of the MGF, $0<\lambda<3/M$. Use $k!\ge2\cdot3^{k-2}$:

    $$
    E[e^{\lambda Y_i}]\le1+\sum_{k\ge2}\frac{\lambda^k\sigma_i^2M^{k-2}}{2\cdot3^{k-2}}=1+\frac{\lambda^2\sigma_i^2}{2(1-\lambda M/3)}\le\exp\left(\frac{\lambda^2\sigma_i^2}{2(1-\lambda M/3)}\right)
    $$

3. Independence → $\log E[e^{\lambda(S_n-E[S_n])}]\le\frac{\lambda^2v}{2(1-\lambda M/3)}$.
4. Chernoff w/ $\lambda=\frac{t}{v+Mt/3}$ → $1-\frac{\lambda M}{3}=\frac{v}{v+Mt/3}$:

    $$
    -\lambda t+\frac{\lambda^2v}{2(1-\lambda M/3)}=-\frac{t^2}{v+Mt/3}+\frac{t^2}{2(v+Mt/3)}=-\frac{t^2}{2(v+Mt/3)}
    $$

5. Confidence width: set exponent $=\ln(2/\delta)$, solve the quadratic in $\varepsilon$, use $\sqrt{x+y}\le\sqrt x+\sqrt y$.
```

```{note} Example
:class: dropdown
$n=1000$ i.i.d. Bernoulli($0.1$), bound $P(\bar X_n-0.1\ge0.05)=P(S_n\ge150)$:
- $\sigma^2=0.09$, $M=0.9$, $t=50$, $v=90$.
- Hoeffding: $e^{-2\cdot50^2/1000}=e^{-5}\approx6.7\times10^{-3}$.
- Bernstein: $\exp\left(-\frac{1250}{90+15}\right)\approx6.8\times10^{-6}$.
- Exact: $\approx4.5\times10^{-7}$.
- → ~1000× tighter than Hoeffding ← $\sigma^2=0.09\ll\frac14$.
```

```{attention} Q&A
:class: dropdown
*When does Bernstein beat Hoeffding?*

- $\sigma^2\ll(b-a)^2/4$: rare events, small $p$, sparse or low-noise losses.
- Samples for accuracy $\varepsilon$: Hoeffding $\propto\frac{1}{\varepsilon^2}$ vs Bernstein $\propto\frac{\sigma^2}{\varepsilon^2}+\frac{M}{\varepsilon}$.
- $\sigma^2\lesssim M\varepsilon$ → $n\propto\frac1\varepsilon$ instead of $\frac1{\varepsilon^2}$.

*When is Bernstein worse?*

- $\sigma^2$ near its max for the range. Rademacher: $\sigma^2=1$, $M=1$.
    - Hoeffding: $e^{-n\varepsilon^2/2}$.
    - Bernstein: $e^{-n\varepsilon^2/(2+2\varepsilon/3)}$ → slightly worse.

*$\sigma^2$ unknown?*

- **Empirical Bernstein**: plug in the sample variance + an extra $O(\ln(1/\delta)/n)$ term.
- Used in bandits (UCB-V) & adaptive A/B testing.

*Unbounded RVs?*

- Moment condition $E|X_i-E[X_i]|^k\le\frac{k!}{2}\sigma_i^2M^{k-2}$, $\forall k\ge2$ → same form w/ $M$ in place of $M/3$.
- Covers sub-exponential RVs ($\chi^2$, exponential).

*Relation to Bennett & CLT?*

- **Bennett**: $\exp\left(-\frac{v}{M^2}h\left(\frac{Mt}{v}\right)\right)$, $h(u)=(1+u)\ln(1+u)-u$ → tighter.
- Bernstein = Bennett relaxed by $h(u)\ge\frac{u^2}{2(1+u/3)}$.
- Small deviations → CLT-like Gaussian tail. Large deviations → Poisson-like tail.
```

&nbsp;

### McDiarmid's Inequality
- **What**: Concentration of $f(X_1,\dots,X_n)$ around $E[f]$, for independent $X_i$ & bounded coordinate influence.
- **Why**: Many statistics are not sums.
    - e.g., worst-case generalization gap over a class, #empty bins, U-statistics.
    - Hoeffding covers only sums → McDiarmid generalizes it.
- **How**:
    1. Reveal $X_1,\dots,X_n$ one at a time → **Doob martingale** from $E[f]$ to $f$.
    2. Each reveal moves the conditional mean within an interval of length $\le c_i$.
    3. Hoeffding's lemma per step → product of MGFs → Gaussian-shaped tail.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $f:\mathcal{X}^n\to\mathbb{R}$: Function of $n$ independent inputs.
    - $c_i$: Bounded-difference constant of coordinate $i$.

Bounded differences:

$$
\sup_{x_1,\dots,x_n,x_i'}\left|f(x_1,\dots,x_i,\dots,x_n)-f(x_1,\dots,x_i',\dots,x_n)\right|\le c_i,\quad\forall i
$$

Bound, $X_1,\dots,X_n$ independent, not nec. identical:

$$
P\left(f(X_1,\dots,X_n)-E[f]\ge\varepsilon\right)\le\exp\left(-\frac{2\varepsilon^2}{\sum_{i=1}^nc_i^2}\right)
$$

- Lower tail: same bound. Two-sided: $\times2$.
- Equivalent: $f-E[f]$ is sub-Gaussian w/ $\tau^2=\frac14\sum_ic_i^2$.
- Hoeffding = special case: $f=\bar X_n$, $X_i\in[a_i,b_i]$ → $c_i=\frac{b_i-a_i}{n}$.
```

```{tip} Derivation
:class: dropdown
1. Doob martingale: $Z_k=E[f\mid X_1,\dots,X_k]$. $Z_0=E[f]$. $Z_n=f$.
2. Increment $D_k=Z_k-Z_{k-1}$. Independence → $Z_k=g_k(X_1,\dots,X_k)$ w/ $g_k(x_{1:k})=E[f(x_{1:k},X_{k+1:n})]$.
3. Given $X_{1:k-1}$, changing $x_k$ moves $g_k$ by $\le c_k$ → $D_k$ lies in an interval of length $\le c_k$.
4. Conditional Hoeffding's lemma → $E[e^{\lambda D_k}\mid X_{1:k-1}]\le e^{\lambda^2c_k^2/8}$.
5. Tower rule → $E[e^{\lambda(f-E[f])}]\le e^{\lambda^2\sum_kc_k^2/8}$.
6. Chernoff, $\lambda=4\varepsilon/\sum_kc_k^2$ → $\exp(-2\varepsilon^2/\sum_kc_k^2)$.

- Step 2 needs independence ← the average over $X_{k+1:n}$ must not depend on $x_k$.
- General martingale version: [Azuma–Hoeffding Inequality](process.md#azuma-hoeffding-inequality).
```

```{note} Example
:class: dropdown
Rademacher generalization bound:
- $\mathcal{G}$: loss class, functions $z\mapsto\ell(h,z)\in[0,1]$. $Z_1,\dots,Z_n$ i.i.d. samples.
- $\Phi=\sup_{g\in\mathcal{G}}\left(E[g(Z)]-\frac1n\sum_{i=1}^ng(Z_i)\right)$: worst-case generalization gap.
- Change 1 sample → $\Phi$ changes by $\le\frac1n$ → $c_i=\frac1n$.
- McDiarmid → w/ prob $\ge1-\delta$: $\Phi\le E[\Phi]+\sqrt{\frac{\ln(1/\delta)}{2n}}$.
- Symmetrization → $E[\Phi]\le2\mathfrak{R}_n(\mathcal{G})$, w/ $\mathfrak{R}_n(\mathcal{G})=E\left[\sup_{g}\frac1n\sum_i\sigma_ig(Z_i)\right]$, $\sigma_i$ i.i.d. Rademacher.
- → $\forall g\in\mathcal{G}$: $E[g]\le\frac1n\sum_ig(Z_i)+2\mathfrak{R}_n(\mathcal{G})+\sqrt{\frac{\ln(1/\delta)}{2n}}$.

Balls into bins:
- $n$ balls, independent random bins. $f$ = #empty bins.
- Move 1 ball → $f$ changes by $\le1$ → $P(|f-E[f]|\ge\varepsilon)\le2e^{-2\varepsilon^2/n}$.
- $f$ is not a sum of independent terms, yet it concentrates.
```

```{attention} Q&A
:class: dropdown
*Pros?*

- Any function, not only sums.
- Needs only independence & bounded influence → no need to know the distribution of $f$.

*Cons?*

- Worst-case $c_i$ → ignores variance → loose. Variance-aware alternatives: Talagrand / Bousquet-type inequalities.
- 1 rare input w/ a large effect → large $c_i$ → useless bound.
- Needs independence.

*Concentrates around what?*

- Around $E[f]$, not around the target quantity.
- → Bound $E[f]$ separately, e.g., symmetrization → Rademacher complexity.

*Why $\sum c_i^2$ and not $(\sum c_i)^2$?*

- Independent fluctuations add in quadrature, like variances.
- $c_i=c$ → deviation scale $c\sqrt n$, not $cn$.

*Relation to Hoeffding & Azuma?*

- Hoeffding = McDiarmid for sums.
- McDiarmid = Azuma on the Doob martingale w/ a sharper constant.
    - Azuma assumes $|D_k|\le c_k$ → interval length $2c_k$ → $e^{-\varepsilon^2/(2\sum c_k^2)}$.
    - McDiarmid uses interval length $c_k$ → $e^{-2\varepsilon^2/\sum c_k^2}$.
```

&nbsp;

## Limit Theorems

### Modes of Convergence
- **What**: Senses in which RVs $Z_n$ approach a limit $Z$.
    - **Almost sure**, **in probability**, **in $L^p$**, **in distribution**.
- **Why**: Each mode answers a different question.
    - a.s.: Does 1 infinite run settle?
    - P: Is a large deviation unlikely at a fixed large $n$?
    - $L^p$: Is the average error size small?
    - d: Is the law (shape) close? → approximations, tests, CIs.
- **How**: Strongest → weakest:
    - a.s. & $L^p$ → in P → in d.
    - Converses fail in general. Extra conditions restore some of them.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $Z_n$: Sequence of RVs.
    - $Z$: Limit RV. Same probability space as $Z_n$, except for d.
    - $F_Z$: CDF of $Z$.
    - $p\ge1$: Moment order.
    - $q\ge p$: Higher moment order.
    - $c$: Constant.

Definitions:
- Almost surely: $Z_n\xrightarrow{a.s.}Z\Leftrightarrow P\left(\lim_{n\to\infty}Z_n=Z\right)=1$.
- In probability: $Z_n\xrightarrow{P}Z\Leftrightarrow\forall\varepsilon>0:\lim_{n\to\infty}P(|Z_n-Z|>\varepsilon)=0$.
- In $L^p$: $Z_n\xrightarrow{L^p}Z\Leftrightarrow\lim_{n\to\infty}E|Z_n-Z|^p=0$. $p=2$ → **mean square**.
- In distribution: $Z_n\xrightarrow{d}Z\Leftrightarrow\lim_{n\to\infty}F_{Z_n}(z)=F_Z(z)$ at every continuity point $z$ of $F_Z$.
    - ⇔ $E[g(Z_n)]\to E[g(Z)]$ for all bounded continuous $g$ (**Portmanteau**).
    - ⇔ $\varphi_{Z_n}(t)\to\varphi_Z(t)$, $\forall t$ (**Lévy continuity**; used in the CLT derivation).

Implications:

$$\begin{align*}
&\text{a.s.}\Rightarrow P\Rightarrow d \\
&L^q\Rightarrow L^p\Rightarrow P,\quad q\ge p\ge1 \\
&Z_n\xrightarrow{d}c\Rightarrow Z_n\xrightarrow{P}c
\end{align*}$$

Partial converses:
- $\xrightarrow{P}$ → $\xrightarrow{a.s.}$ along some subsequence.
- $\sum_nP(|Z_n-Z|>\varepsilon)<\infty$, $\forall\varepsilon>0$ → $\xrightarrow{a.s.}$ ← Borel–Cantelli.
- $\xrightarrow{P}$ & $\{|Z_n|^p\}$ uniformly integrable ⇔ $\xrightarrow{L^p}$ (for $Z_n\in L^p$).
- $\xrightarrow{d}$ → $\xrightarrow{a.s.}$ for copies w/ the same laws on a new probability space (**Skorokhod**).
```

```{tip} Derivation
:class: dropdown
1. a.s. ⇒ P: $\mathbf{1}\{|Z_n-Z|>\varepsilon\}\to0$ a.s. → dominated convergence → $P(|Z_n-Z|>\varepsilon)\to0$.
2. $L^p$ ⇒ P: Markov on $|Z_n-Z|^p$ → $P(|Z_n-Z|>\varepsilon)\le\frac{E|Z_n-Z|^p}{\varepsilon^p}\to0$.
3. $L^q$ ⇒ $L^p$: Jensen → $(E|W|^p)^{1/p}\le(E|W|^q)^{1/q}$ for $q\ge p$.
4. P ⇒ d: sandwich the CDF:

    $$
    F_Z(z-\varepsilon)-P(|Z_n-Z|>\varepsilon)\le F_{Z_n}(z)\le F_Z(z+\varepsilon)+P(|Z_n-Z|>\varepsilon)
    $$

    - $n\to\infty$, then $\varepsilon\to0$ at a continuity point $z$ → $F_{Z_n}(z)\to F_Z(z)$.
5. d to $c$ ⇒ P: $c\pm\varepsilon$ are continuity points of the point-mass CDF:

    $$
    P(|Z_n-c|>\varepsilon)\le F_{Z_n}(c-\varepsilon)+1-F_{Z_n}(c+\varepsilon)\to0+1-1=0
    $$
```

```{note} Example
:class: dropdown
Counterexamples, $U\sim\text{Unif}(0,1)$, $Z\sim N(0,1)$:
1. P ⇏ a.s. & $L^p$ ⇏ a.s.: independent $Z_n\sim\text{Bernoulli}(1/n)$.
    - $E|Z_n|^p=\frac1n\to0$ → $\xrightarrow{L^p}0$ & $\xrightarrow{P}0$.
    - $\sum_n\frac1n=\infty$ & independence → $Z_n=1$ infinitely often a.s. (2nd Borel–Cantelli) → ❌ a.s.
2. a.s. ⇏ $L^1$ & P ⇏ $L^1$: $Z_n=n\mathbf{1}\{U<1/n\}$.
    - $U>0$ → $Z_n=0$ once $n>1/U$ → $\xrightarrow{a.s.}0$.
    - $E[Z_n]=1$, $\forall n$ → ❌ $L^1$.
3. d ⇏ P: $Z_n=-Z$.
    - Same law as $Z$ → $\xrightarrow{d}Z$.
    - $|Z_n-Z|=2|Z|$ does not shrink → ❌ P.
4. d ⇏ moments: example 2 → $\xrightarrow{d}0$, but $E[Z_n]=1\ne0$.
```

```{dropdown} Table: Modes of Convergence
| Mode | Controls | Needs joint law? | On this page |
|:--|:--|:--|:--|
| a.s. | Whole path of 1 run | Yes | Strong LLN |
| P | Deviation at each fixed $n$ | Yes | Weak LLN; consistency |
| $L^2$ | Mean squared error | Yes | $E(\bar X_n-\mu)^2=\sigma^2/n\to0$ |
| d | Law only | No (marginals only) | CLT |
```

```{attention} Q&A
:class: dropdown
*Which mode is consistency?*

- Consistent estimator: $\hat\theta_n\xrightarrow{P}\theta$. Strongly consistent: $\xrightarrow{a.s.}$.
- MSE $\to0$ ($L^2$) → consistent.

*Why does convergence in d not need a joint law?*

- It compares only the CDFs of $Z_n$ & $Z$ → each $Z_n$ can live on a different probability space.

*Common misconception about $\xrightarrow{d}$?*

- ❌ "$Z_n\approx Z$ for large $n$". ✅ Only the laws are close.
- e.g., $Z_n=-Z$: same law, values never close.

*Why does d to a constant give P?*

- A point-mass limit fixes *where* the mass goes → law alone pins down the values.
- → Basis of Slutsky: one limit is constant → joint convergence is free.

*How do LLN & CLT fit?*

- Weak LLN: $\xrightarrow{P}$. Strong LLN: $\xrightarrow{a.s.}$. Finite variance → also $\xrightarrow{L^2}$.
- CLT: $\xrightarrow{d}$ only. $\sqrt n(\bar X_n-\mu)$ does not converge in P to any RV.
```

&nbsp;

### Law of Large Numbers
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

### Central Limit Theorem
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
    &\lim_{n\rightarrow\infty}(1-\frac{t^2}{2n})^n=e^{-\frac{t^2}{2}} \\
    &\varphi_{Z_n}(t)\rightarrow e^{-\frac{t^2}{2}}
    \end{align*}$$
    Back to Step 1.
```

&nbsp;

### Continuous Mapping Theorem
- **What**: Continuous functions preserve convergence a.s., in P & in d.
- **Why**: Prove a limit once for a building block ($\bar X_n$ via LLN/CLT) → get limits for any continuous statistic of it for free.
- **How**: Apply $g$ to both sides of the limit.
    - $g$ may be discontinuous on a set the limit hits w/ prob 0.
    - For several sequences: need *joint* convergence first.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $Z_n$: Random vectors in $\mathbb{R}^d$.
    - $Z$: Limit random vector.
    - $g:\mathbb{R}^d\to\mathbb{R}^k$: Measurable map.
    - $D_g$: Set of discontinuity points of $g$.

Theorem, mode $\in\{\text{a.s.},P,d\}$:

$$
Z_n\xrightarrow{\text{mode}}Z\ \text{ \& }\ P(Z\in D_g)=0\ \Rightarrow\ g(Z_n)\xrightarrow{\text{mode}}g(Z)
$$

- ❌ Not for $L^p$.
```

```{tip} Derivation
:class: dropdown
$g$ continuous everywhere:
1. a.s.: $Z_n(\omega)\to Z(\omega)$ → $g(Z_n(\omega))\to g(Z(\omega))$ by continuity, on the same prob-1 set.
2. d: $h$ bounded continuous → $h\circ g$ bounded continuous → $E[h(g(Z_n))]\to E[h(g(Z))]$ (Portmanteau).
3. P: $\xrightarrow{P}$ ⇔ every subsequence has a further subsequence that converges a.s. → apply step 1 to it.
```

```{note} Example
:class: dropdown
1. $\bar X_n\xrightarrow{a.s.}\mu$ → $\bar X_n^2\xrightarrow{a.s.}\mu^2$, $e^{\bar X_n}\xrightarrow{a.s.}e^\mu$.
2. $\mu\ne0$ → $1/\bar X_n\xrightarrow{a.s.}1/\mu$ ← $g(z)=1/z$ is discontinuous only at $0$, & the limit $\mu\ne0$.
3. CLT + $g(z)=z^2$ → $\frac{n(\bar X_n-\mu)^2}{\sigma^2}\xrightarrow{d}\chi^2_1$ → Wald / chi-square tests.
4. Plug-in estimators: $\hat\theta_n=g(\bar X_n)\xrightarrow{P}g(\mu)$ → method of moments is consistent.
```

```{attention} Q&A
:class: dropdown
*Is marginal convergence enough for $g(Z_n,W_n)$?*

- ❌. $Z\sim N(0,1)$, $Z_n=Z$, $W_n=-Z$.
    - $Z_n\xrightarrow{d}N(0,1)$, $W_n\xrightarrow{d}N(0,1)$.
    - $Z_n+W_n=0$, not $N(0,2)$.
- Need $(Z_n,W_n)$ to converge jointly. Exception: one limit is a constant → [Slutsky's Theorem](#slutskys-theorem).

*What breaks if $g$ is discontinuous at the limit?*

- $Z_n=1/n\to0$, $g(z)=\mathbf{1}\{z>0\}$ → $g(Z_n)=1$, but $g(0)=0$.
- Fine if the limit puts prob 0 on $D_g$, e.g., $\text{sign}(Z_n)$ w/ continuous $Z$.

*Why not $L^p$?*

- $Z_n=\sqrt n\mathbf{1}\{U<1/n\}$, $U\sim\text{Unif}(0,1)$ → $E|Z_n|=\frac{1}{\sqrt n}\to0$.
- $g(z)=z^2$ → $E[Z_n^2]=1$, $\forall n$ → $g(Z_n)\not\xrightarrow{L^1}0$.
- $L^p$ needs control of size, not only of values → continuity is not enough.
```

&nbsp;

### Slutsky's Theorem
- **What**: $Z_n\xrightarrow{d}Z$ & $W_n\xrightarrow{P}c$ (constant) → sums, products & ratios converge in d.
- **Why**: Asymptotic statistics replace unknown params (e.g., $\sigma$) w/ consistent estimates.
    - Need the limit law to stay the same after the swap.
- **How**:
    1. Constant limit → $(Z_n,W_n)\xrightarrow{d}(Z,c)$ jointly, for free.
    2. Continuous mapping → $Z_n+W_n$, $W_nZ_n$, $Z_n/W_n$.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $Z_n$: RVs w/ $Z_n\xrightarrow{d}Z$.
    - $Z$: Limit RV.
    - $W_n$: RVs w/ $W_n\xrightarrow{P}c$.
    - $\hat\sigma_n^2=\frac{1}{n-1}\sum_{i=1}^n(X_i-\bar X_n)^2$: Sample variance.

Theorem:

$$\begin{align*}
&Z_n+W_n\xrightarrow{d}Z+c \\
&W_nZ_n\xrightarrow{d}cZ \\
&Z_n/W_n\xrightarrow{d}Z/c,\quad c\ne0
\end{align*}$$

- Vector/matrix version: $W_n\xrightarrow{P}C$ (constant matrix) → $W_nZ_n\xrightarrow{d}CZ$.

Studentized mean, i.i.d., $0<\sigma^2<\infty$:

$$
T_n=\frac{\sqrt n(\bar X_n-\mu)}{\hat\sigma_n}\xrightarrow{d}N(0,1)
$$
```

```{tip} Derivation
:class: dropdown
Theorem:
1. $W_n-c\xrightarrow{P}0$ → $(Z_n,W_n)-(Z_n,c)\xrightarrow{P}0$.
2. $(Z_n,c)\xrightarrow{d}(Z,c)$.
3. Lemma: $A_n\xrightarrow{d}A$ & $B_n-A_n\xrightarrow{P}0$ → $B_n\xrightarrow{d}A$. → $(Z_n,W_n)\xrightarrow{d}(Z,c)$.
4. CMT w/ $g(z,w)=z+w$, $zw$, $z/w$ (continuous where $w\ne0$).

Studentized mean:
1. $\hat\sigma_n^2=\frac{n}{n-1}\left[\frac1n\sum_iX_i^2-\bar X_n^2\right]\xrightarrow{P}E[X^2]-\mu^2=\sigma^2$ ← WLLN + CMT.
2. CMT ($\sqrt{\cdot}$, $1/\cdot$, $\sigma>0$) → $\sigma/\hat\sigma_n\xrightarrow{P}1$.
3. Split:

    $$
    T_n=\underbrace{\frac{\sqrt n(\bar X_n-\mu)}{\sigma}}_{\xrightarrow{d}N(0,1)\text{ (CLT)}}\cdot\underbrace{\frac{\sigma}{\hat\sigma_n}}_{\xrightarrow{P}1}\xrightarrow{d}N(0,1)
    $$
```

```{note} Example
:class: dropdown
Asymptotic CI for $\mu$ w/ unknown $\sigma$, any finite-variance data:
- $\bar X_n\pm z_{\alpha/2}\frac{\hat\sigma_n}{\sqrt n}$ → coverage $\to1-\alpha$.
- $z_{\alpha/2}$: upper $\alpha/2$ quantile of $N(0,1)$ ($1.96$ for $\alpha=0.05$).
- Same logic → z-test / Wald test w/ estimated SE.
```

```{attention} Q&A
:class: dropdown
*Why must the limit of $W_n$ be a constant?*

- Random limit → joint law of $(Z_n,W_n)$ is not determined by the marginals.
- $Z_n=Z$, $W_n=-Z$, $Z\sim N(0,1)$ → $Z_n+W_n=0$, not $N(0,2)$.

*t-test vs z-test?*

- Gaussian data → $T_n\sim t_{n-1}$ exactly.
- Non-Gaussian data → $T_n\xrightarrow{d}N(0,1)$ by Slutsky. $t_{n-1}\to N(0,1)$ too.
- → Both tests are asymptotically valid for any finite-variance data.

*$\frac1n$ or $\frac{1}{n-1}$ in $\hat\sigma_n^2$?*

- Both work: $\frac{n}{n-1}\to1$.

*What does Slutsky not give?*

- A finite-$n$ error rate.
- Skewed data → slow convergence. **Berry–Esseen**: CDF error $=O\left(\frac{E|X-\mu|^3}{\sigma^3\sqrt n}\right)$.
```

&nbsp;

### Delta Method
- **What**: First-order Taylor expansion → asymptotic normal law of $g(\text{estimator})$.
- **Why**: CLT gives the law of $\bar X_n$ only.
    - Need SEs & CIs for transformed estimates: log, ratio, odds, relative lift.
- **How**:
    1. CLT for $\bar X_n$.
    2. Linearize: $g(\bar X_n)-g(\mu)\approx g'(\mu)(\bar X_n-\mu)$.
    3. Variance scales by $g'(\mu)^2$. Vector case: $\nabla g^\top\Sigma\nabla g$.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $g$: Function, differentiable at $\mu$.
    - $\mathbf{X}_i\in\mathbb{R}^d$: i.i.d. random vectors.
    - $\boldsymbol\mu$: Mean of $\mathbf{X}_i$.
    - $\Sigma$: Covariance of $\mathbf{X}_i$.
    - $\bar{\mathbf{X}}_n$: Sample mean of $\mathbf{X}_i$.
    - $J\in\mathbb{R}^{k\times d}$: Jacobian of $g:\mathbb{R}^d\to\mathbb{R}^k$ at $\boldsymbol\mu$.

Univariate, $g'(\mu)\ne0$:

$$
\sqrt n\left(g(\bar X_n)-g(\mu)\right)\xrightarrow{d}N\left(0,g'(\mu)^2\sigma^2\right)
$$

- Plug-in SE: $\text{SE}[g(\bar X_n)]\approx\frac{|g'(\bar X_n)|\,\hat\sigma_n}{\sqrt n}$ (valid if $g'$ continuous at $\mu$; Slutsky + CMT).
    - $\hat\sigma_n$: Sample std, as in [Slutsky's Theorem](#slutskys-theorem).
- Any estimator w/ $\sqrt n(\hat\theta_n-\theta)\xrightarrow{d}N(0,v)$ → $\sqrt n(g(\hat\theta_n)-g(\theta))\xrightarrow{d}N(0,g'(\theta)^2v)$.

Multivariate:

$$
\sqrt n\left(g(\bar{\mathbf{X}}_n)-g(\boldsymbol\mu)\right)\xrightarrow{d}N\left(\mathbf{0},J\Sigma J^\top\right)
$$

- Scalar $g$ → variance $\nabla g(\boldsymbol\mu)^\top\Sigma\nabla g(\boldsymbol\mu)$.

Second order, $g'(\mu)=0$, $g''(\mu)\ne0$:

$$
n\left(g(\bar X_n)-g(\mu)\right)\xrightarrow{d}\frac{\sigma^2g''(\mu)}{2}\chi^2_1
$$

- Rate $n$, not $\sqrt n$. Limit is not Gaussian.
```

```{tip} Derivation
:class: dropdown
First order:
1. Differentiability at $\mu$: $g(x)-g(\mu)=\left(g'(\mu)+r(x)\right)(x-\mu)$, w/ $r(x)\to0$ as $x\to\mu$, $r(\mu)=0$.
2. $\bar X_n\xrightarrow{P}\mu$ & $r$ continuous at $\mu$ → $r(\bar X_n)\xrightarrow{P}0$ (CMT).
3. Multiply by $\sqrt n$:

    $$
    \sqrt n\left(g(\bar X_n)-g(\mu)\right)=\underbrace{\left(g'(\mu)+r(\bar X_n)\right)}_{\xrightarrow{P}g'(\mu)}\underbrace{\sqrt n(\bar X_n-\mu)}_{\xrightarrow{d}N(0,\sigma^2)}
    $$

4. Slutsky → $N(0,g'(\mu)^2\sigma^2)$.
5. Multivariate: same steps w/ $J$; $J\cdot N(\mathbf{0},\Sigma)\sim N(\mathbf{0},J\Sigma J^\top)$.

Second order:
1. $g(x)-g(\mu)=\frac{g''(\mu)}{2}(x-\mu)^2+o\left((x-\mu)^2\right)$.
2. $n(\bar X_n-\mu)^2=\sigma^2\left[\frac{\sqrt n(\bar X_n-\mu)}{\sigma}\right]^2\xrightarrow{d}\sigma^2\chi^2_1$ (CLT + CMT).
3. Remainder $\times n$ $\xrightarrow{P}0$ → Slutsky.

Why linearization is safe: $\bar X_n-\mu=O_P(1/\sqrt n)$ → quadratic term $O_P(1/n)$ → vanishes after $\sqrt n$ scaling.
```

```{note} Example
:class: dropdown
Log, $\mu>0$:
- $g=\ln$, $g'(\mu)=1/\mu$ → $\sqrt n(\ln\bar X_n-\ln\mu)\xrightarrow{d}N(0,\sigma^2/\mu^2)$.
- $\text{SE}[\ln\bar X_n]\approx\frac{\hat\sigma_n}{\bar X_n\sqrt n}$ = coefficient of variation $/\sqrt n$.

Ratio metric in an A/B test, e.g., CTR = clicks / views, randomized by user:
- Per user $i$: $V_i$ = #views, $C_i$ = #clicks, means $\mu_V$, $\mu_C$. Users i.i.d.; views within a user are correlated.
- $\hat R=\bar C_n/\bar V_n$, $R=\mu_C/\mu_V$, $g(v,c)=c/v$, $\nabla g=\left(-\frac{\mu_C}{\mu_V^2},\frac{1}{\mu_V}\right)$.
- Delta method:

    $$
    Var[\hat R]\approx\frac{1}{n\mu_V^2}\left(\sigma_C^2-2R\,\sigma_{VC}+R^2\sigma_V^2\right)
    $$

    - $\sigma_V^2$: Per-user variance of views.
    - $\sigma_C^2$: Per-user variance of clicks.
    - $\sigma_{VC}$: Per-user covariance of views & clicks.
- ❌ Naive binomial $\hat R(1-\hat R)/\sum_iV_i$: treats views as i.i.d. → ignores within-user correlation → usually too small SE → false positives.

Relative lift, independent groups B (treatment) & A (control) w/ means $\mu_B,\mu_A$ & sample means $\bar Y_B,\bar Y_A$:
- $\hat\Delta=\bar Y_B/\bar Y_A-1$ → $Var[\hat\Delta]\approx\frac{Var[\bar Y_B]}{\mu_A^2}+\frac{\mu_B^2\,Var[\bar Y_A]}{\mu_A^4}$.
```

```{attention} Q&A
:class: dropdown
*Assumptions?*

- Estimator is $\sqrt n$-asymptotically normal.
- $g$ differentiable at $\mu$, $g'(\mu)\ne0$.
- $n$ large enough → $g$ is nearly linear over $\pm$ a few SEs.

*What if $g'(\mu)=0$?*

- First-order limit is $N(0,0)$ → degenerate, wrong rate.
- → Second-order delta method: rate $n$, scaled $\chi^2_1$ limit.

*When does it fail in practice?*

- Small $n$, or strong curvature over the SE scale: ratio w/ denominator mean near 0, log near 0.
- Heavy tails → slow CLT.
- → Bootstrap; for a ratio CI, Fieller's method.

*What is a variance-stabilizing transform?*

- Choose $g$ w/ $g'(\mu)\,\sigma(\mu)=\text{const}$ → asymptotic variance does not depend on $\mu$.
- Poisson ($\sigma^2=\mu$): $g(x)=2\sqrt x$ → $N(0,1)$.
- Bernoulli ($\sigma^2=p(1-p)$): $g(x)=\arcsin\sqrt x$ → $N(0,\frac14)$.

*How does it relate to MLE?*

- $\hat\theta$ MLE → $g(\hat\theta)$ is the MLE of $g(\theta)$ (invariance).
- Asymptotic variance per sample: $g'(\theta)^2/I(\theta)$, $I$: Fisher information.
```

&nbsp;
