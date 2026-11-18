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
# Distributions
Named distributions: what each models, its parameters, moments, transforms & relations.
Study notes from {cite:t}`math` and {cite:t}`prob_lifesaver`.

The math of each distribution is contained in its Math block, following this structure:
- PDF/CDF
- Support & Constraints
- Parameters (stable parameterization)
    - How parameters affect location, spread, shape.
- Shape & tails
- Moments (at least Mean & Variance)
- Entropy
- Transformations (only choose the ones that are applicable)
    - Affine transform
    - Recursion
    - Sum
    - Product/Ratio
    - Log/Exp
    - Min/Max
- Marginals/Conditionals (if multivariate)
- Sampling
- Relationship w/ other distributions (including KL if applicable)

Notations:
- $X$: Random variable; $\mathbf{X}$ if vector-valued.
- $x$: Value of $X$.
- $f$: PDF (continuous) or PMF (discrete).
- $F$: CDF.
- $\Phi$: Standard Normal CDF.
- $M_X(t)$: MGF.
- $H$: Entropy (discrete).
- $h$: Differential entropy (continuous).
- $D_{KL}(P\|Q)$: KL divergence.
- $\text{Unif}(a,b)$: Uniform distribution on $(a,b)$.
- $\bm{1}\{\cdot\}$: Indicator.

&nbsp;

## Discrete
### Bernoulli
- **What**: Description of **a random event** w/ ONLY 2 possible outcomes, w/ a fixed probability of one of them happening.
- **Why**:
    - "Yes/No" situations show up everywhere.
    - Once we can model one event, we can model multiple (i.e., Binomial).

```{note} Math
:class: dropdown
Notation:
$$
X\sim\text{Bern}(p)
$$

PMF/CDF:
$$\begin{align*}
&\text{PMF:} && P(X=x)=p^x(1-p)^{1-x} \\
&\text{CDF:} && F(x)=\begin{cases}
1-p & x=0 \\
1   & x=1
\end{cases}
\end{align*}$$
- Support: $x\in\{0,1\}$
- Params: 
    - $p=P(X=1)\in[0,1]$: Probability of $X=1$.

Shape:
- 2 points.
    - Right point: $P(X=1)$
    - Left point: $P(X=0)$

Moments:
|  |  |
|------ |------- |
| Mean   | $E[X]=p$ |
| Variance | $Var[X]=p(1-p)$ |
| MGF    | $M_X(t)=(1-p)+pe^t$ |

Entropy:
$$
H(X)=-p\log p-(1-p)\log(1-p)
$$
- $\arg\max_pH(p)=\frac{1}{2}$
- $\max_pH(p)=\log2$
- $p\updownarrow$ $\longrightarrow$ $H(p)\rightarrow 0$

Transformations:
|  |  |
|:-------- |:------- |
| Sum (i.i.d.)   | $\sum_{i=1}^nX_i\sim\text{Bin}(n,p)$ |
| Product (*independent*) | $X\sim\text{Bern}(p),Y\sim\text{Bern}(q)\rightarrow XY\sim\text{Bern}(pq)$ |

Sampling:
1. Draw $U\sim\text{Unif}(0,1)$.
2. Set $X=\bm{1}\{U\leq p\}$.

KLD:
- $P=\text{Bern}(p),Q=\text{Bern}(q),p,q\in(0,1)$:
$$
D_{KL}(P||Q)=p\log\frac{p}{q}+(1-p)\log\frac{1-p}{1-q}
$$
```

&nbsp;

### Binomial
- **What**: Probability of **#times** an event occurs out of a fixed number of independent, identical binary trials.
- **Why**: To monitor repeated chance events.

```{note} Math
:class: dropdown
Notation:
$$
X\sim\text{Bin}(n,p)
$$

PMF/CDF:
$$\begin{align*}
&\text{PMF:} && P(X=k)=\begin{pmatrix}
n \\ k
\end{pmatrix}p^k(1-p)^{n-k} \\
&\text{CDF:} && P(X\leq x)=\sum_{k=0}^{\lfloor x\rfloor}\begin{pmatrix}
n \\ k
\end{pmatrix}p^k(1-p)^{n-k}
\end{align*}$$
- Support: $x\in\{0,\cdots,n\}$: #times the event occurs.
- Params: 
    - $n\in\mathbb{N}$: #trials.
    - $p\in[0,1]$: Probability that the event occurs per trial.

Shape:
- Unimodal (i.e., one peak/mode) at $m=\lfloor(n+1)p\rfloor$.
- Skewness:
    - $p<\frac{1}{2}$: Right (mass → 0)
    - $p=\frac{1}{2}$: Symmetric
    - $p>\frac{1}{2}$: Left (mass → $n$)

Moments:
|  |  |
|:------ |:------- |
| Mean   | $E[X]=np$ |
| Variance | $Var[X]=np(1-p)$ |
| MGF    | $M_X(t)=(1-p+pe^t)^n$ |

Entropy:
$$
H(X)=-\sum_{x=0}^nP(X=x)\log P(X=x)
$$
- $\forall n\in\mathbb{N}: \arg\max_pH(p)=\frac{1}{2}$

Transformations:
|  |  |
|:-------- |:------- |
| Recursion                                | $\frac{P(X=k+1)}{P(X=k)}=\frac{n-k}{k+1}\frac{p}{1-p}$   |
| Sum                                      | $X_n\sim\text{Bin}(n,p),X_m\sim\text{Bin}(m,p)$ <br>$\rightarrow X=X_n+X_m\sim\text{Bin}(n+m,p)$  |
| Approximation by Normal Distribution | $X\sim\text{Bin}(n,p)\rightarrow \frac{X-E[X]}{\sqrt{Var[X]}} \xRightarrow[n\to\infty]{} N(0,1)$ |

Sampling:
1. Draw $n$ independent $U_i\sim\text{Unif}(0,1)$.
2. Set $X_i=\bm{1}\{U_i\leq p\}$.
3. Output $X=\sum_iX_i$.

Relationship w/ other distributions:
- Poisson limit:
$$
X_n\sim\text{Bin}(n,\frac{\lambda}{n})\longrightarrow X_n \xRightarrow[n\to\infty]{} \text{Pois}(\lambda)
$$
- Normal approximation (CLT):
$$
X\sim\text{Bin}(n,p)\rightarrow \frac{X-E[X]}{\sqrt{Var[X]}} \xRightarrow[n\to\infty]{} N(0,1)
$$

KLD:
- $P=\text{Bin}(n,p),Q=\text{Bin}(n,q),p,q\in(0,1)$:
$$
D_{KL}(P||Q)=n\left[p\log\frac{p}{q}+(1-p)\log\frac{1-p}{1-q}\right]
$$
```

&nbsp;

### Geometric
- **What**: #Bernoulli trials until the 1st success.
- **Why**: Model "how many tries until it works?".
    - Discrete analog of the Exponential waiting time.
- **How**: Repeat independent Bernoulli($p$) trials → stop at the 1st success → count all trials, incl. the success.

```{note} Math
:class: dropdown
Notation:

$$
X\sim\text{Geom}(p)
$$

PMF/CDF:

$$\begin{align*}
&\text{PMF:} && P(X=k)=(1-p)^{k-1}p \\
&\text{CDF:} && F(x)=1-(1-p)^{\lfloor x\rfloor},\quad x\geq1
\end{align*}$$

- Support: $x\in\{1,2,\dots\}$: #trials until the 1st success (incl. it).
- Params:
    - $p\in(0,1]$: Success probability per trial.
- Failures variant: $Y=X-1\in\{0,1,\dots\}$: #failures before the 1st success.
    - $P(Y=k)=(1-p)^kp$.
    - Libraries differ: `scipy.stats.geom` counts trials; `torch.distributions.Geometric` counts failures.

Shape ($0<p<1$):
- Mode at 1.
- Strictly decreasing: each step multiplies the mass by $1-p$.
- Right-skewed.

Moments:
|  |  |
|:------ |:------- |
| Mean   | $E[X]=\frac1p$ (failures variant: $\frac{1-p}{p}$) |
| Variance | $Var[X]=\frac{1-p}{p^2}$ (both variants) |
| MGF    | $M_X(t)=\frac{pe^t}{1-(1-p)e^t},\quad t<-\log(1-p)$ |

Entropy:

$$
H(X)=\frac{-p\log p-(1-p)\log(1-p)}{p}
$$

- = Bernoulli entropy × $E[X]$ ← $X$ & the trial sequence determine each other.
- Max among distributions on $\{1,2,\dots\}$ w/ fixed mean.

Transformations:
|  |  |
|:-------- |:------- |
| Recursion (Memoryless) | $P(X>s+k\|X>s)=P(X>k),\quad s,k\in\{0,1,\dots\}$<br>Only memoryless distribution on $\{1,2,\dots\}$. |
| Min (*independent*) | $X\sim\text{Geom}(p),Y\sim\text{Geom}(q)\rightarrow\min(X,Y)\sim\text{Geom}(1-(1-p)(1-q))$ |
| Sum (*i.i.d.*) | $\sum_{i=1}^rX_i$ = #trials until the $r$-th success<br>$\rightarrow\sum_{i=1}^r(X_i-1)\sim\text{NB}(r,p)$ |

Sampling:
- Inverse CDF ($p<1$):
    1. Draw $U\sim\text{Unif}(0,1)$.
    2. Output $X=\left\lceil\frac{\log U}{\log(1-p)}\right\rceil$.

Relationship w/ other distributions:
- Negative Binomial: $X-1\sim\text{NB}(1,p)$.
- Exponential (discretized):

$$
E\sim\text{Exp}(\lambda)\longrightarrow\lceil E\rceil\sim\text{Geom}(1-e^{-\lambda})
$$

- Exponential (limit): $p\to0\rightarrow pX\Rightarrow\text{Exp}(1)$.

KLD:
- $P=\text{Geom}(p),Q=\text{Geom}(q),p,q\in(0,1)$ (same for the failures variant):

$$
D_{KL}(P\|Q)=\log\frac pq+\frac{1-p}{p}\log\frac{1-p}{1-q}
$$
```

```{attention} Q&A
:class: dropdown
*Expected #rolls of a fair die until the 1st 6?*

- $X\sim\text{Geom}(\frac16)\rightarrow E[X]=6$.
- Expected #non-6 rolls before it: $5$ (failures variant).

*Why is $E[X]=\frac1p$?*

- 1st-step analysis: $E[X]=1+(1-p)E[X]$ ← fail w.p. $1-p$ → start over (memoryless).
- Solve → $E[X]=\frac1p$.

*Why is it memoryless?*

- $P(X>k)=(1-p)^k\rightarrow\frac{P(X>s+k)}{P(X>s)}=(1-p)^k=P(X>k)$.
- Past failures carry no info ← trials independent & identical.
- Gambler's fallacy = denial of this property.

*When does it NOT fit?*

- $p$ changes across trials (learning, fatigue) → hazard not constant.
- Trials dependent.

*How does it relate to Exponential?*

- Both: constant hazard → memoryless.
- Geometric = discrete time; Exponential = continuous time.
- $\lceil\text{Exp}\rceil$ is Geometric; scaled Geometric → Exponential as $p\to0$.
```

&nbsp;

### Negative Binomial
- **What**: #failures before the $r$-th success in Bernoulli trials.
- **Why**: Overdispersed counts.
    - Poisson forces $Var=$ mean. Real counts usually have $Var>$ mean.
    - NB adds 1 param $r$ → $Var=\mu+\frac{\mu^2}{r}>\mu$ ($\mu>0$).
- **How**: 2 equivalent stories.
    - **Trials**: Repeat Bernoulli($p$) trials → stop at the $r$-th success → count failures.
    - **Gamma–Poisson**: Draw a rate $\lambda\sim$ Gamma → draw a count $\sim\text{Pois}(\lambda)$. Rate spread → extra variance.

```{note} Math
:class: dropdown
Notation:

$$
X\sim\text{NB}(r,p)
$$

PMF/CDF:

$$\begin{align*}
&\text{PMF:} && P(X=k)=\binom{k+r-1}{k}p^r(1-p)^k \\
&\text{CDF:} && F(x)=\sum_{k=0}^{\lfloor x\rfloor}P(X=k)=\frac{B(p;r,\lfloor x\rfloor+1)}{B(r,\lfloor x\rfloor+1)}
\end{align*}$$

- Support: $x\in\{0,1,\dots\}$: #failures before the $r$-th success.
- Params:
    - $r>0$: #successes to wait for.
        - Integer in the trials story.
        - Any real $r>0$ via $\binom{k+r-1}{k}=\frac{\Gamma(k+r)}{k!\,\Gamma(r)}$.
    - $p\in(0,1]$: Success probability per trial.
- $B(\cdot;\cdot,\cdot)$, $B(\cdot,\cdot)$: Incomplete & complete Beta functions (see Beta).
- Mean–dispersion form: $\mu=\frac{r(1-p)}{p}$, $p=\frac{r}{r+\mu}$.
- Other variants: #trials until the $r$-th success ($=X+r$); some sources swap $p$ & $1-p$.

Shape ($0<p<1$):
- $r\leq1$: Mode at 0, strictly decreasing.
- $r>1$: Unimodal at $\left\lfloor\frac{(r-1)(1-p)}{p}\right\rfloor$ (2 modes if the ratio is an integer).
- Right-skewed.

Moments:
|  |  |
|:------ |:------- |
| Mean   | $E[X]=\frac{r(1-p)}{p}=\mu$ |
| Variance | $Var[X]=\frac{r(1-p)}{p^2}=\mu+\frac{\mu^2}{r}$ |
| MGF    | $M_X(t)=\left(\frac{p}{1-(1-p)e^t}\right)^r,\quad t<-\log(1-p)$ |

Entropy:

$$
H(X)=-\sum_{k=0}^\infty P(X=k)\log P(X=k)
$$

- No closed form.

Transformations:
|  |  |
|:-------- |:------- |
| Recursion | $\frac{P(X=k+1)}{P(X=k)}=\frac{k+r}{k+1}(1-p)$ |
| Sum (*independent, common $p$*) | $X_i\sim\text{NB}(r_i,p)\rightarrow\sum_iX_i\sim\text{NB}(\sum_ir_i,p)$ |

Sampling:
- Gamma–Poisson (any real $r>0$, $p<1$):
    1. Draw $\lambda\sim\text{Gamma}(r,\frac{p}{1-p})$.
    2. Output $X\sim\text{Pois}(\lambda)$.
- Geometric sum (integer $r$): Add $r$ i.i.d. failures-variant Geometrics.

Relationship w/ other distributions:
- Geometric: $\text{NB}(1,p)$ = failures variant of $\text{Geom}(p)$.
- Gamma–Poisson mixture:

$$
\lambda\sim\text{Gamma}\left(r,\frac{p}{1-p}\right),\ X|\lambda\sim\text{Pois}(\lambda)\longrightarrow X\sim\text{NB}(r,p)
$$

- Poisson limit: $r\to\infty$ w/ $\mu$ fixed → $\text{Pois}(\mu)$.
- [GLM](../../ml/supervised.md#glm): NB regression for overdispersed counts; dispersion $\alpha=\frac1r\rightarrow Var=\mu+\alpha\mu^2$.

KLD:
- Same $r$: $P=\text{NB}(r,p),Q=\text{NB}(r,q),p,q\in(0,1)$:

$$
D_{KL}(P\|Q)=r\log\frac pq+\frac{r(1-p)}{p}\log\frac{1-p}{1-q}
$$

- Different $r$: No closed form.
```

```{tip} Derivation
:class: dropdown
*Why is a Gamma mixture of Poissons Negative Binomial?*

1. Let $\lambda\sim\text{Gamma}(r,\beta)$, $X|\lambda\sim\text{Pois}(\lambda)$. Integrate out $\lambda$:

    $$
    P(X=k)=\int_0^\infty e^{-\lambda}\frac{\lambda^k}{k!}\cdot\frac{\beta^r}{\Gamma(r)}\lambda^{r-1}e^{-\beta\lambda}d\lambda=\frac{\beta^r}{k!\,\Gamma(r)}\cdot\frac{\Gamma(k+r)}{(1+\beta)^{k+r}}
    $$

2. Regroup:

    $$
    P(X=k)=\frac{\Gamma(k+r)}{k!\,\Gamma(r)}\left(\frac{\beta}{1+\beta}\right)^r\left(\frac{1}{1+\beta}\right)^k
    $$

3. Set $p=\frac{\beta}{1+\beta}$ ($\Leftrightarrow\beta=\frac{p}{1-p}$) → NB($r,p$) PMF.
4. Variance by the law of total variance:

    $$
    Var[X]=E[Var(X|\lambda)]+Var(E[X|\lambda])=E[\lambda]+Var[\lambda]=\frac r\beta+\frac{r}{\beta^2}=\mu+\frac{\mu^2}{r}
    $$

    - Poisson noise ($E[\lambda]$) + rate spread ($Var[\lambda]$).
```

```{attention} Q&A
:class: dropdown
*Why not just use Poisson for counts?*

- Poisson: $Var=$ mean. Real counts (clicks, word frequencies, read counts) often have $Var\gg$ mean.
- Poisson fit on overdispersed data → standard errors too small → false significance.

*Where does the extra variance come from?*

- Rate heterogeneity: each unit has its own $\lambda\sim$ Gamma.
- Total variance = Poisson noise + spread of rates.

*What does $r$ control?*

- $r$⬇️ → overdispersion⬆️.
- $r\to\infty$ → Poisson.

*Can NB model underdispersion ($Var<$ mean)?*

- ❌. $Var=\mu+\frac{\mu^2}{r}\geq\mu$ always.
- Binomial has $Var=np(1-p)<np$ → use it (or another underdispersed model).

*Which parameterization does a library use?*

- `scipy.stats.nbinom`: #failures before $n$ successes, `p` = success probability → matches this page.
- `torch.distributions.NegativeBinomial`: #successes before `total_count` failures → its `probs` = $1-p$ here.
- Always check before you fit or compare.
```

&nbsp;

### Categorical
- **What**: Bernoulli BUT w/ >2 possible outcomes.
- **Why**: Many random events have >2 possible outcomes, so Bernoulli isn't enough.

```{note} Math
:class: dropdown
Notation:
$$
X\sim\text{Cat}(\mathbf{p})
$$

PMF/CDF:
$$\begin{align*}
&\text{PMF:} && P(X=k)=p_k \\
&\text{CDF:} && F(x)=P(X\leq x)=\sum_{i=1}^{\lfloor x\rfloor}p_i
\end{align*}$$
- Support: $x\in\{1,\dots,K\}$
- Params:
    - $\mathbf{p}=(p_1,\dots,p_K)$: Probabilities for all categories. 
        - $p_k=P(X=k)\in[0,1]$: Probability of category $k$.
        - $\sum_i p_i=1$

Shape:
- $K$ point masses (a bar plot over categories)

Moments:
- Assume one-hot view: $Y\in\{0,1\}^K$ by $Y_k=\mathbf 1\{X=k\}$.

|  |  |
|:------ |:------- |
| Mean   | $E[Y]=\mathbf{p}$ |
| Variance | $Var[Y_k]=p_k(1-p_k)$ |
| Covariance | $Cov(Y_i,Y_j)=-p_ip_j,\quad\forall i\neq j$ |
| MGF (scalar view)    | $M_X(t)=\sum_{k=1}^Kp_ke^{tk}$ |

Entropy:
$$
H(X)=-\sum_{k=1}^K p_k\log p_k
$$

- $\arg\max_{\mathbf p} H(\mathbf p)$ is the uniform distribution $p_k=\frac1K$.
- $\max H=\log K$.
- If $p_k\to 1$ & others $\to 0$, then $H\to 0$.

Sampling:
1. Draw $U\sim\text{Unif}(0,1)$.
2. Set $X=\min\{j:\sum_{i=1}^j p_i\ge U\}$
   - (i.e., pick the first category whose cumulative probability exceeds $U$.)

KLD:
- $P=\text{Cat}(\mathbf p),\ Q=\text{Cat}(\mathbf q)$, $p_k,q_k\in(0,1)$:
$$
D_{KL}(P|Q)=\sum_{k=1}^K p_k\log\frac{p_k}{q_k}
$$
```

&nbsp;

### Multinomial
- **What**: Binomial BUT w/ >2 possible outcomes.
- **Why**: Many random events have >2 possible outcomes, so Binomial isn't enough.

```{note} Math
:class: dropdown
Notation:
$$
\mathbf{X}\sim\text{Mult}(n,\mathbf{p})
$$

PMF/CDF:
$$\begin{align*}
&\text{PMF:} && P(\mathbf{X}=\mathbf{x})=\frac{n!}{\prod_{k=1}^K x_k!}\prod_{k=1}^K p_k^{x_k} \\
&\text{CDF:} && F(\mathbf{x})=P(\mathbf{X}\le \mathbf{x})=\sum_{\substack{\mathbf{y}\in\mathbb{N}_0^K:\ \sum_k y_k=n,\forall k:\ y_k\le x_k}}\frac{n!}{\prod_{k=1}^K y_k!}\prod_{k=1}^K p_k^{y_k}
\end{align*}$$
- Support: $\mathbf{x}=(x_1,\dots,x_K)\in\mathbb{N}_0^K$
    - $x_k$: #occurrences of category $k$.
    - $\sum_{k=1}^K x_k=n$.
- Params:
    - $n\in\mathbb{N}$: #trials.
    - $\mathbf{p}=(p_1,\dots,p_K)$: Probabilities for all categories.
        - $p_k\in[0,1]$: Probability of category $k$ per trial.
        - $\sum_i p_i=1$.

Shape:
- A distribution over $K$-dimensional count vectors (mass on $\sum_k x_k=n$)

Moments:
|  |  |
|:------ |:------- |
| Mean   | $E[\mathbf{X}]=n\mathbf{p}$ |
| Variance | $Var[X_k]=np_k(1-p_k)$ |
| Covariance | $Cov(X_i,X_j)=-np_ip_j,\quad\forall i\neq j$ |
| MGF    | $M_{\mathbf{X}}(\mathbf{t})=\left(\sum_{k=1}^K p_k e^{t_k}\right)^n$ |

Entropy:
$$
H(\mathbf{X})=-\sum_{\substack{\mathbf{x}\in\mathbb{N}_0^K:\ \sum_k x_k=n}} P(\mathbf{X}=\mathbf{x})\log P(\mathbf{X}=\mathbf{x})
$$
- $\arg\max_{\mathbf p} H(\mathbf{X})$ occurs at the uniform distribution $p_k=\frac1K$ (for fixed $n,K$).
- No simple closed form for $\max H$.
- If $p_k\to 1$ & others $\to 0$, $\mathbf{X}$ concentrates on $(0,\dots,n,\dots,0)$, $H\to 0$.

Sampling:
1. Draw $X^{(1)},\dots,X^{(n)}\overset{i.i.d.}{\sim}\text{Cat}(\mathbf{p})$.
2. Set $X_k=\sum_{m=1}^n \mathbf{1}\{X^{(m)}=k\}$ for each $k$.
   - (i.e., count how many times each category appears.)

KLD:
- $P=\text{Mult}(n,\mathbf p),\ Q=\text{Mult}(n,\mathbf q)$, $p_k,q_k\in(0,1)$:
$$
D_{KL}(P|Q)=n\sum_{k=1}^K p_k\log\frac{p_k}{q_k}
$$
```

&nbsp;

### Poisson
- **What**: Probability of #occurrences of an event in a fixed time/space interval, given an average occurrence rate.
- **Why**: "How many times will something happen?"

```{note} Math
:class: dropdown
Notation:
$$
X\sim\text{Pois}(\lambda)
$$

PMF/CDF:
$$\begin{align*}
&\text{PMF:} && P(X=k)=e^{-\lambda}\frac{\lambda^k}{k!} \\
&\text{CDF:} && P(X\leq x)=e^{-\lambda}\sum_{k=0}^{\lfloor x\rfloor}\frac{\lambda^k}{k!}
\end{align*}$$
- Support: $x\in\{0,1,\dots\}$: #times the event occurs.
- Params:
    - $\lambda>0$: Avg occurrence rate.

Shape:
- For small $\lambda$, mass is concentrated near 0 & is right-skewed.
- For large $\lambda$, it approaches Normal distribution shape.

Moments:
|  |  |
|:------ |:------- |
| Mean   | $E[X]=\lambda$ |
| Variance | $Var[X]=\lambda$ |
| MGF    | $M_X(t)=\exp(\lambda(e^t-1))$ |

Transformations:
|  |  |
|:-------- |:------- |
| Recursion                                | $P(X=k+1)=\frac{\lambda}{k+1}P(X=k)$   |
| Sum (independent)         | $X_i\sim\text{Pois}(\lambda_i)\rightarrow \sum_{i=1}^nX_i\sim\text{Pois}\left(\sum_{i=1}^n\lambda_i\right)$  |

Sampling:
- Knuth's product method (for small $\lambda$):
    1. Set $L=e^{-\lambda},k=0,P=1$.
    2. While $P>L$:
        1. $k\leftarrow k+1$.
        2. $P\leftarrow PU_k,\quad U_k\sim\text{Unif}(0,1)$.
    3. Output $X=k-1$.
- CDF inversion:
    1. Draw $U\sim\text{Unif}(0,1)$.
    2. Output smallest $k$ with $F_\lambda(k)\geq U$.

Relationship w/ other distributions:
- Poisson limit for Binomial:
$$
X_n\sim\text{Bin}(n,\frac{\lambda}{n})\longrightarrow X_n \xRightarrow[n\to\infty]{} \text{Pois}(\lambda)
$$
- Normal approximation: For large $\lambda$:
$$
X\approx N(\lambda,\lambda)
$$

KLD:
- $P=\text{Pois}(\lambda),Q=\text{Pois}(\mu)$:
$$
D_{KL}(P||Q)=\lambda\log\frac{\lambda}{\mu}+\mu-\lambda
$$
```

&nbsp;

## Continuous
### Uniform
- **What**: Constant density on an interval $[a,b]$.
- **Why**: Base of all sampling.
    - Every univariate distribution = a transform of $\text{Unif}(0,1)$ (inverse CDF).
    - Max entropy on a bounded interval → no preference inside $[a,b]$.
- **How**: Equal-length subintervals → equal probability.

```{note} Math
:class: dropdown
Notation:

$$
X\sim\text{Unif}(a,b)
$$

PDF/CDF:

$$\begin{align*}
&\text{PDF:} && f(x)=\frac{1}{b-a},\quad x\in[a,b] \\
&\text{CDF:} && F(x)=\begin{cases}
0 & x<a \\
\frac{x-a}{b-a} & a\leq x\leq b \\
1 & x>b
\end{cases}
\end{align*}$$

- Support: $x\in[a,b]$.
- Params:
    - $a,b\in\mathbb{R}$, $a<b$: Endpoints (location & width $b-a$).

Shape:
- Flat: every point in $[a,b]$ is a mode.
- Symmetric about $\frac{a+b}{2}$.
- Bounded → no tails.

Moments:
|  |  |
|:------ |:------- |
| Mean   | $E[X]=\frac{a+b}{2}$ |
| Variance | $Var[X]=\frac{(b-a)^2}{12}$ |
| MGF    | $M_X(t)=\frac{e^{tb}-e^{ta}}{t(b-a)},\ t\neq0;\quad M_X(0)=1$ |

Entropy:

$$
h(X)=\log(b-a)
$$

- Max among distributions supported on $[a,b]$.
- $b-a<1\rightarrow h<0$ (differential entropy can be negative).

Transformations:
|  |  |
|:-------- |:------- |
| Affine | $c>0\rightarrow cX+d\sim\text{Unif}(ca+d,cb+d)$ |
| Probability integral transform | $X$ continuous w/ CDF $F$ $\rightarrow F(X)\sim\text{Unif}(0,1)$ |
| Inverse CDF | $U\sim\text{Unif}(0,1)\rightarrow F^{-1}(U)\sim F$<br>$F^{-1}(u)=\inf\{x:F(x)\geq u\}$ (any $F$, incl. discrete) |
| Log | $U\sim\text{Unif}(0,1)\rightarrow-\log U\sim\text{Exp}(1)$ |
| Order statistics (*i.i.d.*) | $U_1,\dots,U_m\sim\text{Unif}(0,1)\rightarrow U_{(\ell)}\sim\text{Beta}(\ell,m-\ell+1)$<br>$U_{(\ell)}$: $\ell$-th smallest. |
| Sum (*i.i.d.*) | $U_1+U_2$: Triangular on $[0,2]$, peak at 1 |

Sampling:
1. Draw $U\sim\text{Unif}(0,1)$ (PRNG).
2. Output $X=a+(b-a)U$.

Relationship w/ other distributions:
- Beta: $\text{Unif}(0,1)=\text{Beta}(1,1)$.
- Exponential: $-\frac{\log U}{\lambda}\sim\text{Exp}(\lambda)$.
- Normal: Box–Muller maps 2 Uniforms → 2 i.i.d. $N(0,1)$.
- Dirichlet: $\text{Dir}(1,\dots,1)$ = uniform on the simplex.

KLD:
- $P=\text{Unif}(a_0,b_0),Q=\text{Unif}(a_1,b_1)$:

$$
D_{KL}(P\|Q)=\begin{cases}
\log\frac{b_1-a_1}{b_0-a_0} & [a_0,b_0]\subseteq[a_1,b_1] \\
\infty & \text{otherwise}
\end{cases}
$$
```

```{tip} Derivation
:class: dropdown
*Why does $F^{-1}(U)$ have CDF $F$?*

1. Generalized inverse: $F^{-1}(u)\leq x\Leftrightarrow u\leq F(x)$ ← $F$ non-decreasing & right-continuous.
2. $P(F^{-1}(U)\leq x)=P(U\leq F(x))=F(x)$ ← $U$ uniform.

*Why is $F(X)\sim\text{Unif}(0,1)$ for continuous $F$?*

1. Strictly increasing $F$: $P(F(X)\leq u)=P(X\leq F^{-1}(u))=F(F^{-1}(u))=u$.
2. Flat parts of $F$ carry 0 probability → same result for any continuous $F$.
3. Discrete $F$ → $F(X)$ takes countably many values → ❌ uniform.
```

```{attention} Q&A
:class: dropdown
*Why $\frac{1}{12}$ in the variance?*

- $\text{Unif}(0,1)$: $E[X^2]=\frac13$, $E[X]^2=\frac14\rightarrow Var=\frac1{12}$.
- Scale by $b-a$ → variance × $(b-a)^2$.

*$E[\max]$ of $m$ i.i.d. $\text{Unif}(0,1)$?*

- $P(\max\leq x)=x^m\rightarrow\max\sim\text{Beta}(m,1)\rightarrow E[\max]=\frac{m}{m+1}$.
- Symmetric: $E[\min]=\frac{1}{m+1}$.

*Is a uniform prior "uninformative"?*

- ❌. Uniform on $\theta$ ⇏ uniform on a nonlinear $g(\theta)$.
    - E.g., $\theta\sim\text{Unif}(0,1)\rightarrow\theta^2$ has density $\frac{1}{2\sqrt{x}}$, not flat.
- Reparameterization-invariant alternative: [Jeffreys prior](bayes.md#jeffreys-prior).

*Is there a uniform distribution on $\mathbb{R}$?*

- ❌. Constant density on $\mathbb{R}$ cannot integrate to 1.
- Used only as an improper prior.

*Why not sample a Normal by inverse CDF?*

- $\Phi^{-1}$ has no closed form → numerical approximation (works, but costs more).
- Box–Muller avoids $\Phi^{-1}$.
```

&nbsp;

### Normal/Gaussian
- **What**: Probability of the sum/avg of many independent random outcomes.
    - Most outcomes land near the mean.
    - Very large/small outcomes are rare.

```{note} Math
:class: dropdown
Notation:
$$
X\sim N(\mu,\sigma^2)
$$

PDF/CDF:
$$\begin{align*}
&\text{PDF:} && f(x)=\frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right) \\
&\text{CDF:} && F(x)=\Phi\left(\frac{x-\mu}{\sigma}\right) \\
&\text{Standard CDF:} && \Phi(z)=\frac{1}{\sqrt{2\pi}}\int_{-\infty}^z\exp\left(-\frac{t^2}{2}\right)dt
\end{align*}$$
- Support: $x\in\mathbb{R}$: A random outcome.
- Params:
    - $\mu\in\mathbb{R}$: Mean (center location).
    - $\sigma>0$: Standard deviation (scale).

Shape:
- Unimodal at $\mu$.
- Symmetric about $\mu$.
- Gaussian tails:
$$
P(|X-\mu|\ge t)\leq 2\exp\left(-\frac{t^2}{2\sigma^2}\right)
$$

Moments:
|  |  |
|:------ |:------- |
| Mean   | $E[X]=\mu$ |
| Variance | $Var[X]=\sigma^2$ |
| MGF    | $M_X(t)=\exp\left(\mu t+\frac{1}{2}\sigma^2t^2\right)$ |

Entropy:
$$
h(X)=\frac{1}{2}\log(2\pi e \sigma^2)
$$
- Independent of $\mu$.
- **Max** among all continuous distributions with fixed $\sigma^2$.

Transformations:
|  |  |
|:-------- |:------- |
| Linearity                     | $a\neq0\rightarrow aX+b\sim N(a\mu+b,a^2\sigma^2)$ |
| Sum (*independence*)                                        | $\sum_{i=1}^{n}X_i\sim N\left(\sum_{i=1}^{n}\mu_i,\sum_{i=1}^{n}\sigma_i^2\right)$ |
| Standardization | $Z=\frac{X-\mu}{\sigma}\sim N(0,1)$                                        |
| Inflection Points | $f''_X(x)=0\Longleftrightarrow x=\mu\pm\sigma$ |
| Empirical Probabilities | $P(\|X-\mu\|\leq\sigma)\approx 0.68$<br>$P(\|X-\mu\|\leq2\sigma)\approx 0.95$<br>$P(\|X-\mu\|\leq3\sigma)\approx 0.997$ |

Sampling:
- Box-Muller:
    1. Sample $U_1,U_2\sim\text{Unif}(0,1)$.
    2. Set $Z_1=\sqrt{-2\log U_1}\cos(2\pi U_2),Z_2=\sqrt{-2\log U_1}\sin(2\pi U_2)$.
    3. Output $Z_1,Z_2\sim N(0,1)$.
- Sampling (after Box-Muller):
    1. Sample $Z\sim N(0,1)$.
    2. Output $X=\mu+\sigma Z$.

Relationship w/ other distributions:
- Chi-square:
$$
Z_i\overset{i.i.d.}{\sim}N(0,1)\longrightarrow\sum_{i=1}^nZ_i^2\sim\chi_n^2
$$
- Chi-square & Student t:
$$
Z\sim N(0,1),V\sim\chi_v^2\longrightarrow\frac{Z}{\sqrt{\frac{V}{v}}}\sim t_v
$$
- Lognormal:
$$
X\sim N(\mu,\sigma^2)\longrightarrow \exp(X)\sim\text{LogN}(\mu,\sigma^2)
$$


KLD:
- $P=N(\mu_0,\sigma_0^2),Q=N(\mu_1,\sigma_1^2)$:
$$
D_{KL}(P||Q)=\log\frac{\sigma_1}{\sigma_0}+\frac{\sigma_0^2+(\mu_0-\mu_1)^2}{2\sigma_1^2}-\frac{1}{2}
$$
```

&nbsp;

#### Multivariate Normal
- **What**: Normal distribution over vectors $\mathbf{X}\in\mathbb{R}^d$.
    - ⇔ Every linear combination $\mathbf{a}^T\mathbf{X}$ is univariate Normal.
- **Why**: Model correlated continuous variables jointly.
    - Closed under affine maps, marginalization & conditioning → exact inference in closed form.
- **How**: Shift & shear standard Normal noise.
    1. Draw $d$ i.i.d. $N(0,1)$ → $\mathbf{Z}$.
    2. Mix via $L$ ($\Sigma=LL^T$) → correlations.
    3. Shift by $\boldsymbol\mu$.

```{note} Math
:class: dropdown
Notation:

$$
\mathbf{X}\sim N(\boldsymbol\mu,\Sigma)
$$

PDF/CDF:

$$\begin{align*}
&\text{PDF:} && f(\mathbf{x})=\frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}}\exp\left(-\frac12(\mathbf{x}-\boldsymbol\mu)^T\Sigma^{-1}(\mathbf{x}-\boldsymbol\mu)\right) \\
&\text{CDF:} && F(\mathbf{x})=P(X_1\leq x_1,\dots,X_d\leq x_d)\quad\text{(no closed form; numerical)}
\end{align*}$$

- Support: $\mathbf{x}\in\mathbb{R}^d$.
- Params:
    - $\boldsymbol\mu\in\mathbb{R}^d$: Mean vector (center).
    - $\Sigma\in\mathbb{R}^{d\times d}$: Covariance; symmetric, PSD ($\Sigma\succeq0$).
        - PD ($\Sigma\succ0$) → the density above exists.
        - Singular → degenerate: mass on the affine subspace $\boldsymbol\mu+\text{col}(\Sigma)$; no density on $\mathbb{R}^d$.
- $d$: Dimension.
- $|\Sigma|$: Determinant.
- $\Lambda=\Sigma^{-1}$: Precision matrix.

Shape:
- Unimodal at $\boldsymbol\mu$.
- Density contours = ellipsoids $\Delta^2=c$.
    - $\Delta^2=(\mathbf{x}-\boldsymbol\mu)^T\Sigma^{-1}(\mathbf{x}-\boldsymbol\mu)$: Squared **Mahalanobis distance**.
    - Axes ∥ eigenvectors of $\Sigma$; half-lengths $\propto\sqrt{\text{eigenvalues}}$.

Moments:
|  |  |
|:------ |:------- |
| Mean   | $E[\mathbf{X}]=\boldsymbol\mu$ |
| Covariance | $Cov[\mathbf{X}]=\Sigma$ |
| MGF    | $M_{\mathbf{X}}(\mathbf{t})=\exp\left(\mathbf{t}^T\boldsymbol\mu+\frac12\mathbf{t}^T\Sigma\mathbf{t}\right)$ |

Entropy ($\Sigma\succ0$):

$$
h(\mathbf{X})=\frac12\log\left((2\pi e)^d|\Sigma|\right)
$$

- Independent of $\boldsymbol\mu$.
- Max among all distributions on $\mathbb{R}^d$ w/ covariance $\Sigma$.
- $|\Sigma|$⬆️ (generalized variance) → $h$⬆️.

Transformations:
|  |  |
|:-------- |:------- |
| Affine | $A\in\mathbb{R}^{q\times d},\mathbf{b}\in\mathbb{R}^q\rightarrow A\mathbf{X}+\mathbf{b}\sim N(A\boldsymbol\mu+\mathbf{b},A\Sigma A^T)$ |
| Linear combination | $\mathbf{a}^T\mathbf{X}\sim N(\mathbf{a}^T\boldsymbol\mu,\mathbf{a}^T\Sigma\mathbf{a})$ |
| Whitening ($\Sigma\succ0$) | $\Sigma=LL^T\rightarrow L^{-1}(\mathbf{X}-\boldsymbol\mu)\sim N(\mathbf{0},I_d)$ |
| Sum (*independent*) | $\mathbf{X}+\mathbf{Y}\sim N(\boldsymbol\mu_X+\boldsymbol\mu_Y,\Sigma_X+\Sigma_Y)$ |
| Quadratic form ($\Sigma\succ0$) | $\Delta^2\sim\chi^2_d$; singular $\Sigma$ of rank $r$ → use $\Sigma^+$ → $\chi^2_r$ |

Marginals/Conditionals:
- Partition:

$$
\mathbf{X}=\begin{pmatrix}\mathbf{X}_1\\ \mathbf{X}_2\end{pmatrix},\quad
\boldsymbol\mu=\begin{pmatrix}\boldsymbol\mu_1\\ \boldsymbol\mu_2\end{pmatrix},\quad
\Sigma=\begin{pmatrix}\Sigma_{11} & \Sigma_{12}\\ \Sigma_{21} & \Sigma_{22}\end{pmatrix},\quad\Sigma_{21}=\Sigma_{12}^T
$$

- Marginal: $\mathbf{X}_1\sim N(\boldsymbol\mu_1,\Sigma_{11})$ (drop the other blocks).
- Conditional ($\Sigma_{22}\succ0$):

$$\begin{align*}
\mathbf{X}_1|\mathbf{X}_2=\mathbf{x}_2&\sim N(\boldsymbol\mu_{1|2},\Sigma_{1|2}) \\
\boldsymbol\mu_{1|2}&=\boldsymbol\mu_1+\Sigma_{12}\Sigma_{22}^{-1}(\mathbf{x}_2-\boldsymbol\mu_2) \\
\Sigma_{1|2}&=\Sigma_{11}-\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}
\end{align*}$$

- $\Sigma_{1|2}$: **Schur complement** of $\Sigma_{22}$ in $\Sigma$; independent of $\mathbf{x}_2$.
- Precision form ($\Sigma\succ0$): $\Sigma_{1|2}=\Lambda_{11}^{-1}$, $\boldsymbol\mu_{1|2}=\boldsymbol\mu_1-\Lambda_{11}^{-1}\Lambda_{12}(\mathbf{x}_2-\boldsymbol\mu_2)$.
- Independence: $\mathbf{X}_1\perp\mathbf{X}_2\Leftrightarrow\Sigma_{12}=0$ (← jointly Gaussian).

Sampling:
1. Factor $\Sigma=LL^T$ (Cholesky; $L$ lower-triangular; needs $\Sigma\succ0$).
2. Draw $\mathbf{Z}\sim N(\mathbf{0},I_d)$ (e.g., Box–Muller).
3. Output $\mathbf{X}=\boldsymbol\mu+L\mathbf{Z}$.
- Any $A$ w/ $AA^T=\Sigma$ works, e.g., eigen: $\Sigma=QDQ^T\rightarrow A=QD^{1/2}$.

Relationship w/ other distributions:
- Normal: $d=1\rightarrow N(\mu,\sigma^2)$.
- Chi-squared: $\Delta^2\sim\chi^2_d$.
- [GP](bayes.md#gp): Infinite-dim generalization; every finite subset is MVN.

KLD:
- $P=N(\boldsymbol\mu_0,\Sigma_0),Q=N(\boldsymbol\mu_1,\Sigma_1)$, both $d$-dim & PD:

$$
D_{KL}(P\|Q)=\frac12\left[\text{tr}(\Sigma_1^{-1}\Sigma_0)+(\boldsymbol\mu_1-\boldsymbol\mu_0)^T\Sigma_1^{-1}(\boldsymbol\mu_1-\boldsymbol\mu_0)-d+\log\frac{|\Sigma_1|}{|\Sigma_0|}\right]
$$

- $d=1$ → the Normal KLD above.
- VAE: $Q=N(\mathbf{0},I_d)$, $\Sigma_0=\text{diag}(\boldsymbol\sigma^2)$ → $\frac12\sum_{j=1}^d\left(\sigma_j^2+\mu_{0,j}^2-1-\log\sigma_j^2\right)$.
```

```{tip} Derivation
:class: dropdown
*Why is the conditional mean linear in $\mathbf{x}_2$ & the covariance a Schur complement?*

1. Define $\mathbf{W}=\mathbf{X}_1-\Sigma_{12}\Sigma_{22}^{-1}\mathbf{X}_2$ (affine in $\mathbf{X}$ → jointly Gaussian w/ $\mathbf{X}_2$).
2. $Cov(\mathbf{W},\mathbf{X}_2)=\Sigma_{12}-\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{22}=0$ → $\mathbf{W}\perp\mathbf{X}_2$ (jointly Gaussian).
3. $\mathbf{X}_1=\mathbf{W}+\Sigma_{12}\Sigma_{22}^{-1}\mathbf{X}_2$ → fix $\mathbf{X}_2=\mathbf{x}_2$; $\mathbf{W}$ keeps its distribution.
4. $E[\mathbf{W}]=\boldsymbol\mu_1-\Sigma_{12}\Sigma_{22}^{-1}\boldsymbol\mu_2$ → $\boldsymbol\mu_{1|2}=\boldsymbol\mu_1+\Sigma_{12}\Sigma_{22}^{-1}(\mathbf{x}_2-\boldsymbol\mu_2)$.
5. $Cov[\mathbf{W}]=\Sigma_{11}-2\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}+\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{22}\Sigma_{22}^{-1}\Sigma_{21}=\Sigma_{11}-\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}$.

*Where does the KLD come from?*

1. $\log\frac{f_P}{f_Q}=\frac12\log\frac{|\Sigma_1|}{|\Sigma_0|}-\frac12(\mathbf{x}-\boldsymbol\mu_0)^T\Sigma_0^{-1}(\mathbf{x}-\boldsymbol\mu_0)+\frac12(\mathbf{x}-\boldsymbol\mu_1)^T\Sigma_1^{-1}(\mathbf{x}-\boldsymbol\mu_1)$.
2. Trick: $E[\mathbf{v}^TM\mathbf{v}]=\text{tr}(M\,Cov[\mathbf{v}])+E[\mathbf{v}]^TME[\mathbf{v}]$.
3. Under $P$: $E[(\mathbf{x}-\boldsymbol\mu_0)^T\Sigma_0^{-1}(\mathbf{x}-\boldsymbol\mu_0)]=\text{tr}(I_d)=d$.
4. Under $P$: $E[(\mathbf{x}-\boldsymbol\mu_1)^T\Sigma_1^{-1}(\mathbf{x}-\boldsymbol\mu_1)]=\text{tr}(\Sigma_1^{-1}\Sigma_0)+(\boldsymbol\mu_0-\boldsymbol\mu_1)^T\Sigma_1^{-1}(\boldsymbol\mu_0-\boldsymbol\mu_1)$.
5. Take $E_P$ of step 1 & substitute → formula.
```

````{important} Code
:class: dropdown
```python
import torch

class MVN:
    def __init__(self, mu, Sigma):
        self.mu, self.Sigma = mu, Sigma                ## mu: (d,), Sigma: (d, d)
        self.L = torch.linalg.cholesky(Sigma)          ## Sigma = L L^T; fails if Sigma not PD

    def sample(self, m):
        z = torch.randn(m, self.mu.shape[0])           ## Z ~ N(0, I_d), shape (m, d)
        return self.mu + z @ self.L.T                  ## row-wise x = mu + L z -> Cov = L L^T = Sigma

    def condition(self, idx1, idx2, x2):
        ## X_1 | X_2 = x2 via the Schur complement
        S11 = self.Sigma[idx1][:, idx1]
        S12 = self.Sigma[idx1][:, idx2]
        S22 = self.Sigma[idx2][:, idx2]
        K = torch.linalg.solve(S22, S12.T).T          ## K = S12 S22^{-1}; solve avoids an explicit inverse
        mu_c = self.mu[idx1] + K @ (x2 - self.mu[idx2])
        Sigma_c = S11 - K @ S12.T                      ## S12^T = S21
        return MVN(mu_c, Sigma_c)

## Example
mu = torch.tensor([0.0, 1.0])
Sigma = torch.tensor([[2.0, 0.8],
                      [0.8, 1.0]])
p = MVN(mu, Sigma)
print(p.sample(5).shape)                               ## torch.Size([5, 2])
c = p.condition([0], [1], torch.tensor([2.0]))
print(c.mu, c.Sigma)                                   ## mean 0 + 0.8*(2-1) = 0.8; var 2 - 0.8^2 = 1.36
```
````

```{attention} Q&A
:class: dropdown
*Why must $\Sigma$ be PSD?*

- $Var[\mathbf{a}^T\mathbf{X}]=\mathbf{a}^T\Sigma\mathbf{a}\geq0$ for every $\mathbf{a}$.
- Density needs PD: $|\Sigma|>0$ & $\Sigma^{-1}$ exist.
- Singular $\Sigma$ ← a linear combination w/ 0 variance → mass on a lower-dim subspace.

*Uncorrelated ⇒ independent?*

- Only if jointly Gaussian.
- Counterexample: $X\sim N(0,1)$, random sign $S=\pm1$ independent of $X$, $Y=SX$.
    - $Y\sim N(0,1)$ & $Cov(X,Y)=E[S]E[X^2]=0$.
    - $|Y|=|X|$ → dependent.
    - $(X,Y)$ not jointly Gaussian ← $X+Y=0$ w.p. $\frac12$.

*Gaussian marginals ⇒ jointly Gaussian?*

- ❌. Same counterexample.

*Why is $\Sigma_{1|2}$ independent of $\mathbf{x}_2$?*

- $\mathbf{X}_1$ = linear function of $\mathbf{X}_2$ + independent Gaussian residual $\mathbf{W}$.
- Observation shifts the mean only; residual spread stays.
- $\Sigma_{1|2}\preceq\Sigma_{11}$ ← $\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}\succeq0$: conditioning never adds uncertainty.

*Covariance zero vs precision zero?*

- $\Sigma_{ij}=0\Leftrightarrow X_i\perp X_j$ (marginally).
- $\Lambda_{ij}=0\Leftrightarrow X_i\perp X_j\mid$ all other components.
- Sparse precision = Gaussian graphical model.

*Why Cholesky for sampling?*

- $O(d^3)$ once, then $O(d^2)$ per sample; triangular & numerically stable.
- Fails on near-singular $\Sigma$ → add jitter $\epsilon I$.

*Where is the mass in high dimensions?*

- $\mathbf{Z}\sim N(\mathbf{0},I_d)$: $\|\mathbf{Z}\|^2\sim\chi^2_d$ → mean $d$, variance $2d$.
- → $\|\mathbf{Z}\|\approx\sqrt d$: mass on a thin shell, not near the mode at $\mathbf{0}$.
```

&nbsp;

### Lognormal
- **What**: Probability of the product of many independent, **positive** random growth factors.
    - If the **log** of a variable is **Normal**, then the variable is **Lognormal**.
- **Why**:
    - Many quantities are made by multiplying & can't go below 0.
    - BUT once they take a log, they behave like normal.

```{note} Math
:class: dropdown
Notation:
$$
X\sim\text{LogN}(\mu,\sigma^2)\Leftrightarrow\log X\sim N(\mu,\sigma^2)
$$

PDF/CDF:
$$\begin{align*}
&\text{PDF:} && f(x)=\frac{1}{x\sqrt{2\pi}\sigma}\exp\left(-\frac{(\log x-\mu)^2}{2\sigma^2}\right), x>0 \\
&\text{CDF:} && F(x)=\Phi\left(\frac{\log x-\mu}{\sigma}\right), x>0
\end{align*}$$
- Support: $x\in(0,\infty)$: A random growth factor.
- Params:
    - $\mu\in\mathbb{R}$: Mean of $\log X$.
    - $\sigma>0$: Standard deviation of $\log X$.

Shape:
- Unimodal at $e^{\mu-\sigma^2}$.
- Strictly right-skewed.

Moments:
|  |  |
|:------ |:------- |
| Mean     | $E[X]=\exp\left(\mu+\frac{\sigma^2}{2}\right)$  |
| Variance | $Var[X]=\left(e^{\sigma^2}-1\right)\exp\left(2\mu+\sigma^2\right)$ |
| MGF    | Non-existent |

Entropy:
$$
h(X)=\mu+\log(\sigma\sqrt{2\pi})+\frac 1 2
$$

Transformations:
|  |  |
|:-------- |:------- |
| Scale    | $a>0\rightarrow aX\sim\text{LogN}(\mu+\log a,\sigma^2)$                                                                                                         |
| Power    | $k\neq0\rightarrow X^k\sim\text{LogN}(k\mu,k^2\sigma^2)$                                                                                                |
| Product  | $X_1\sim\text{LogN}(\mu_1,\sigma_1^2),X_2\sim\text{LogN}(\mu_2,\sigma_2^2)$<br>$\rightarrow X_1X_2\sim\text{LogN}(\mu_1+\mu_2,\sigma_1^2+\sigma_2^2)$ |
| Ratio    | If independent as above, $\frac{X_1}{X_2}\sim\text{LogN}(\mu_1-\mu_2,\sigma_1^2+\sigma_2^2)$                                                                    |

Sampling:
1. Sample $Z\sim N(0,1)$.
2. Output $X=\exp(\mu+\sigma Z)$.

KLD:
- $P=\text{LogN}(\mu_0,\sigma_0^2),Q=\text{LogN}(\mu_1,\sigma_1^2)$:
$$
D_{KL}(P||Q)=D_{KL}(\log P||\log Q)
$$
```

&nbsp;

### Exponential
- **What**: Probability of the waiting time till the next event when events happen randomly at an avg rate.
- **Why**: A good match with Poisson distribution.
    - If events happen randomly at a steady avg rate, then "no event yet" gets exponentially unlikely as time passes.
        - Constant-rate randomness forces exponential waiting.
        - Poisson randomness over time is memoryless.

```{note} Math
:class: dropdown
Notation:
$$
X\sim\text{Exp}(\lambda)
$$

PDF/CDF:
$$\begin{align*}
&\text{PDF:} && f(x)=\lambda e^{-\lambda x}, x\geq 0 \\
&\text{CDF:} && F(x)=1-e^{-\lambda x}, x\geq 0
\end{align*}$$
- Support: $x\in[0,\infty)$: Waiting time.
- Params:
    - $\lambda>0$: Avg event occurrence rate (i.e., time scale).

Shape:
- Unimodal at 0.
- Density strictly decreasing in $x$.

Moments:
|  |  |
|:------ |:------- |
| Mean     | $E[X]=\frac 1 \lambda$  |
| Variance | $Var[X]=\frac 1 \lambda^2$ |
| All moments    | $E[X^k]=\frac{k!}{\lambda^k}$ |
| MGF | $M_X(t)=E[e^{tX}]=\frac \lambda {\lambda-t},\quad \forall t<\lambda$ |

Entropy:
$$
h(X)=1-\log\lambda
$$
- Spread ↑ → Entropy ↑.

Transformations:
|  |  |
|:-------- |:------- |
| Scale    | $a>0\rightarrow aX\sim\text{Exp}(\frac\lambda a)$ |
| Recursion (Memoryless)    | $P(X>s+x\|X>s)=P(X>x)$<br>No matter how long you've waited, you are starting over. |
| Sum (*i.i.d.*) | $X_i\overset{i.i.d.}{\sim}\text{Exp}(\lambda)\rightarrow\sum_{i=1}^nX_i\sim\text{Gamma}(n,\text{rate}=\lambda)$ |

Sampling:
1. Sample $U\sim\text{Unif}(0,1)$.
2. Output $X=\frac{-\log U}{\lambda}$.

Relationship w/ other distributions:
- Gamma special case:
$$
\text{Exp}(\lambda)=\text{Gamma}(1,\text{rate}=\lambda)
$$
- Uniform:
$$
U\sim\text{Unif}(0,1)\Leftrightarrow\frac{-\log U}{\lambda}\sim\text{Exp}(\lambda)
$$

KLD:
- $P=\text{Exp}(\lambda_0),Q=\text{Exp}(\lambda_1)$:
$$
D_{KL}(P||Q)=\log\frac{\lambda_0}{\lambda_1}+\frac{\lambda_1}{\lambda_0}-1
$$
```

&nbsp;

### Laplace
- **What**: Two-sided Exponential: density $\propto e^{-|x-\mu|/b}$.
- **Why**: Sharp peak + heavier-than-Normal tails.
    - Log-density = $-\frac{|x-\mu|}{b}$ (L1) → prior ↔ Lasso; likelihood ↔ absolute loss (median).
- **How**: Draw an Exponential magnitude → attach a random sign → shift by $\mu$.

```{note} Math
:class: dropdown
Notation:

$$
X\sim\text{Laplace}(\mu,b)
$$

PDF/CDF:

$$\begin{align*}
&\text{PDF:} && f(x)=\frac{1}{2b}\exp\left(-\frac{|x-\mu|}{b}\right) \\
&\text{CDF:} && F(x)=\begin{cases}
\frac12\exp\left(\frac{x-\mu}{b}\right) & x<\mu \\
1-\frac12\exp\left(-\frac{x-\mu}{b}\right) & x\geq\mu
\end{cases}
\end{align*}$$

- Support: $x\in\mathbb{R}$.
- Params:
    - $\mu\in\mathbb{R}$: Location (mean = median = mode).
    - $b>0$: Scale.

Shape:
- Symmetric about $\mu$.
- Cusp at $\mu$: $f$ not differentiable there.
- Exponential tails $e^{-|x-\mu|/b}$.
    - Heavier than Normal ($e^{-x^2}$); lighter than Student-t (polynomial).
    - Excess kurtosis 3 (Normal: 0).

Moments:
|  |  |
|:------ |:------- |
| Mean   | $E[X]=\mu$ |
| Variance | $Var[X]=2b^2$ |
| MGF    | $M_X(t)=\frac{e^{\mu t}}{1-b^2t^2},\quad\|t\|<\frac1b$ |

Entropy:

$$
h(X)=\log(2be)
$$

- Max among distributions on $\mathbb{R}$ w/ fixed $E[|X-\mu|]$.

Transformations:
|  |  |
|:-------- |:------- |
| Affine | $c\neq0\rightarrow cX+d\sim\text{Laplace}(c\mu+d,\|c\|b)$ |
| Absolute value | $\|X-\mu\|\sim\text{Exp}(\frac1b)$ |
| Difference (*i.i.d.*) | $E_1,E_2\sim\text{Exp}(\lambda)\rightarrow E_1-E_2\sim\text{Laplace}(0,\frac1\lambda)$ |
| Scale mixture of Normals | $V\sim\text{Exp}(\frac{1}{2b^2}),\ X\|V\sim N(\mu,V)\rightarrow X\sim\text{Laplace}(\mu,b)$ |

Sampling:
1. Draw $E\sim\text{Exp}(1)$ & a fair sign $S\in\{-1,+1\}$.
2. Output $X=\mu+bSE$.

Relationship w/ other distributions:
- Exponential: $|X-\mu|$ is Exponential; difference of 2 i.i.d. Exponentials is Laplace.
- [Lasso](../../ml/supervised.md#lasso-regression): Gaussian likelihood + i.i.d. $\text{Laplace}(0,b)$ prior on weights $\mathbf{w}$:

$$
-\log p(\mathbf{w}|\mathcal{D})=\frac{1}{2\sigma^2}\sum_{i=1}^m(y_i-\mathbf{w}^T\mathbf{x}_i)^2+\frac1b\|\mathbf{w}\|_1+\text{const}
$$

- $\sigma^2$: Noise variance.
- × $\sigma^2$ → MAP = Lasso w/ penalty weight $\lambda=\frac{\sigma^2}{b}$.
- Laplace likelihood ↔ absolute loss: MLE of $\mu$ = sample median.

KLD:
- $P=\text{Laplace}(\mu_0,b_0),Q=\text{Laplace}(\mu_1,b_1)$:

$$
D_{KL}(P\|Q)=\log\frac{b_1}{b_0}+\frac{|\mu_0-\mu_1|}{b_1}+\frac{b_0}{b_1}\exp\left(-\frac{|\mu_0-\mu_1|}{b_0}\right)-1
$$
```

```{tip} Derivation
:class: dropdown
*Why is $E_1-E_2$ Laplace?*

1. $M_{E_1-E_2}(t)=M_{E_1}(t)M_{E_2}(-t)=\frac{\lambda}{\lambda-t}\cdot\frac{\lambda}{\lambda+t}=\frac{1}{1-t^2/\lambda^2}$, $|t|<\lambda$.
2. Match $\frac{1}{1-b^2t^2}$ → $\text{Laplace}(0,\frac1\lambda)$.

*Where does the KLD come from?*

1. $D_{KL}=E_P\left[\log\frac{b_1}{b_0}-\frac{|X-\mu_0|}{b_0}+\frac{|X-\mu_1|}{b_1}\right]=\log\frac{b_1}{b_0}-1+\frac{E_P|X-\mu_1|}{b_1}$ ← $E_P|X-\mu_0|=b_0$.
2. For $\delta=|\mu_1-\mu_0|$: $E_P|X-\mu_1|=\delta+b_0e^{-\delta/b_0}$ (integrate the 2 tails).
3. Substitute → formula.
```

```{attention} Q&A
:class: dropdown
*Why does a Laplace prior give exact zeros but a Gaussian prior does not?*

- $-\log$ prior: Laplace → $\frac{|w|}{b}$, cusp at 0; Gaussian → $\frac{w^2}{2\tau^2}$, 0 gradient at 0.
    - $\tau^2$: Gaussian prior variance.
- Cusp → MAP stays at exactly 0 while the data gradient is below the penalty threshold.
- Only the MAP is sparse. Laplace-prior posterior mean & samples are nonzero a.s. (Bayesian Lasso is not sparse).

*Laplace vs Normal errors?*

- Normal errors → MLE = least squares → conditional mean.
- Laplace errors → MLE = least absolute deviations → conditional median → robust to outliers.

*Is Laplace heavy-tailed?*

- Heavier than Normal, but all moments exist & MGF exists near 0.
- Polynomial tails (outlier-dominated data) → Student-t.

*Where else does it appear?*

- Differential privacy: Laplace mechanism adds $\text{Laplace}(0,\frac{\Delta_1}{\varepsilon})$ noise to a query.
    - $\Delta_1$: L1 sensitivity of the query.
    - $\varepsilon$: Privacy budget.
```

&nbsp;

### Gamma
- **What**: Description of **wait time** till a certain number of random events have happened. 
    - Exponential: till 1 occurrence.
    - Gamma: till $k$ occurrences.
- **Why**: Generalization of Exponential distribution.

```{note} Math
:class: dropdown
Notation:
$$
X\sim\text{Gamma}(\alpha,\beta)
$$

PDF/CDF:
$$\begin{align*}
&\text{PDF:} && f(x)=\frac{\beta^\alpha}{\Gamma(\alpha)}x^{\alpha-1}e^{-\beta x} \\
&\text{CDF:} && F(x)=\frac{\gamma(\alpha,\beta x)}{\Gamma(\alpha)}
\end{align*}$$
- Support: $x\ge 0$: Wait time.
- Params:
    - $\alpha>0$: Shape.
    - $\beta>0$: Rate.

Gamma Functions:
- **Complete**: Full area under a Gamma-shaped curve.
$$\begin{align*}
&\Gamma(x)=\int_0^\infty t^{x-1}e^{-t}dt \\
&\Gamma(x+1)=x\Gamma(x) \\
&\Gamma(n)=(n-1)!,\quad n\in\mathbb{Z},n>0
\end{align*}$$
- **Lower incomplete**: Left.
$$
\gamma(\alpha,x)=\int_0^x t^{\alpha-1}e^{-t}dt
$$
- **Upper incomplete**: Right.
$$
\Gamma(\alpha,x)=\int_x^\infty t^{\alpha-1}e^{-t}dt
$$
- Relationship:
$$
\Gamma(\alpha)=\gamma(\alpha,x)+\Gamma(\alpha,x)
$$

Shape:
- Shape w.r.t. $\alpha$:
    - $\alpha\in(0,1)$: Blows up at 0 then decreases.
    - $\alpha=1$: Exponential distribution.
    - $\alpha>1$: Unimodal at $\frac{\alpha-1}{\beta}$.
- Tail behavior:
    - $x\rightarrow0: f(x)\propto x^{\alpha-1}$
    - $x\rightarrow\infty: f(x)\approx \text{poly}(x)e^{-\beta x}$
        - i.e., Exponential tail with rate $\beta$.

Moments:
|  |  |
|:------ |:------- |
| Mean     | $E[X]=\frac\alpha\beta$  |
| Variance | $Var[X]=\frac{\alpha}{\beta^2}$ |
| All moments    | $E[X^k]=\frac{\Gamma(\alpha+k)}{\Gamma(\alpha)}\frac{1}{\beta^k}$ |
| MGF | $M_X(t)=\left(\frac{\beta}{\beta-t}\right)^\alpha,\quad t<\beta$ |

Entropy:
$$
h(X)=\alpha-\log\beta+\log\Gamma(\alpha)+(1-\alpha)\varphi(\alpha)
$$
- $\varphi(\alpha)=\frac{\Gamma'(\alpha)}{\Gamma(\alpha)}$: Digamma function.
- Rate: $\beta$↑ → $h$↓.
- Shape: $\alpha$↑ → approximates Normal distribution.

Transformations:
|  |  |
|:-------- |:------- |
| Scale    | $c>0\rightarrow cX\sim\text{Gamma}(\alpha,\frac\beta c)$ |
| Recursion (*shape increment*) | $E\sim\text{Exp}(\beta)\rightarrow X+E\sim\text{Gamma}(\alpha+1,\beta)$ |
| Sum (*independent, common rate*) | $X_i\sim\text{Gamma}(\alpha_i,\beta)\rightarrow\sum_{i=1}^nX_i\sim\text{Gamma}(\sum_{i=1}^n\alpha_i,\beta)$ |
| Ratio (*independent, common rate*) | $Y\sim\text{Gamma}(\gamma,\beta)\rightarrow\frac{X}{X+Y}\sim\text{Beta}(\alpha,\gamma)$ |
```

&nbsp;

#### Chi-Squared
- **What**: Sum of $k$ squared i.i.d. standard Normals.
- **Why**: Distribution of squared Normal errors.
    - Sample variance, goodness-of-fit & likelihood-ratio statistics = sums of (approx.) squared Normals.
- **How**: Square $k$ independent $N(0,1)$ → add.
    - = $\text{Gamma}(\frac k2,\frac12)$ (rate $\frac12$).

```{note} Math
:class: dropdown
Notation:

$$
X\sim\chi^2_k
$$

PDF/CDF:

$$\begin{align*}
&\text{PDF:} && f(x)=\frac{1}{2^{k/2}\Gamma(k/2)}x^{k/2-1}e^{-x/2},\quad x>0 \\
&\text{CDF:} && F(x)=\frac{\gamma(k/2,\,x/2)}{\Gamma(k/2)}
\end{align*}$$

- Support: $x\in(0,\infty)$.
- Params:
    - $k>0$: Degrees of freedom (dof); integer in the sum-of-squares story. Overrides $k$ = class idx.
- $\Gamma$, $\gamma$: Complete & lower incomplete Gamma functions (see Gamma).

Shape:
- $k=1$: Blows up at 0, then decreases.
- $k=2$: $\text{Exp}(\frac12)$, mode at 0.
- $k>2$: Unimodal at $k-2$.
- Right-skewed; skewness $\sqrt{8/k}$ → $k$⬆️ → $\approx N(k,2k)$.

Moments:
|  |  |
|:------ |:------- |
| Mean   | $E[X]=k$ |
| Variance | $Var[X]=2k$ |
| MGF    | $M_X(t)=(1-2t)^{-k/2},\quad t<\frac12$ |

Entropy:

$$
h(X)=\frac k2+\log\left(2\Gamma\left(\frac k2\right)\right)+\left(1-\frac k2\right)\varphi\left(\frac k2\right)
$$

- $\varphi$: Digamma function (see Gamma).
- = Gamma entropy w/ $\alpha=\frac k2,\beta=\frac12$.

Transformations:
|  |  |
|:-------- |:------- |
| Sum (*independent*) | $X_i\sim\chi^2_{k_i}\rightarrow\sum_iX_i\sim\chi^2_{\sum_ik_i}$ |
| Scale | $c>0\rightarrow cX\sim\text{Gamma}(\frac k2,\frac{1}{2c})$ |
| Ratio (*independent*) | $X_i\sim\chi^2_{k_i}\rightarrow\frac{X_1/k_1}{X_2/k_2}\sim\text{F}_{k_1,k_2}$ (F-distribution) |

Sampling:
- Integer $k$: Draw $Z_1,\dots,Z_k\overset{i.i.d.}{\sim}N(0,1)$ → output $\sum_iZ_i^2$.
- Any $k>0$: Draw $\text{Gamma}(\frac k2,\frac12)$.

Relationship w/ other distributions:
- Normal (definition):

$$
Z_i\overset{i.i.d.}{\sim}N(0,1)\longrightarrow\sum_{i=1}^kZ_i^2\sim\chi^2_k
$$

- Gamma: $\chi^2_k=\text{Gamma}(\frac k2,\frac12)$ (rate).
- Exponential: $\chi^2_2=\text{Exp}(\frac12)$.
- Sample variance of Normal data:

$$
X_i\overset{i.i.d.}{\sim}N(\mu,\sigma^2)\longrightarrow\frac{(m-1)S^2}{\sigma^2}\sim\chi^2_{m-1},\ \text{independent of }\bar X
$$

- $\bar X=\frac1m\sum_{i=1}^mX_i$: Sample mean.
- $S^2=\frac{1}{m-1}\sum_{i=1}^m(X_i-\bar X)^2$: Sample variance.
- Student-t: $\frac{Z}{\sqrt{V/k}}\sim t_k$, $V\sim\chi^2_k$ independent of $Z$.
- MVN: Squared Mahalanobis distance $\sim\chi^2_d$.
- Asymptotic tests:
    - Pearson goodness-of-fit statistic → $\chi^2$ w/ dof = #categories − 1 − #fitted params.
    - Wilks: $2\log$(likelihood ratio) → $\chi^2$ w/ dof = #constrained params (regularity conditions).

KLD:
- $P=\chi^2_{k_0},Q=\chi^2_{k_1}$ (Gamma KLD w/ common rate):

$$
D_{KL}(P\|Q)=\frac{k_0-k_1}{2}\varphi\left(\frac{k_0}{2}\right)-\log\Gamma\left(\frac{k_0}{2}\right)+\log\Gamma\left(\frac{k_1}{2}\right)
$$
```

```{attention} Q&A
:class: dropdown
*Why $m-1$ dof for the sample variance?*

- Residuals satisfy $\sum_i(X_i-\bar X)=0$ → live in an $(m-1)$-dim subspace.
- Rotate onto that subspace → sum of $m-1$ independent squared Normals.

*How do you build a CI for $\sigma^2$?*

- Pivot $\frac{(m-1)S^2}{\sigma^2}\sim\chi^2_{m-1}$ → invert:

    $$
    \left[\frac{(m-1)S^2}{\chi^2_{m-1,1-\alpha/2}},\ \frac{(m-1)S^2}{\chi^2_{m-1,\alpha/2}}\right]
    $$

    - $\chi^2_{k,q}$: $q$-quantile of $\chi^2_k$.
    - $1-\alpha$: Confidence level.
- Asymmetric around $S^2$ ← $\chi^2$ is right-skewed.

*Is the $\chi^2$ variance test robust?*

- ❌. It needs Normal data.
- Heavy tails (excess kurtosis) → wrong coverage, even for large $m$.
- Contrast: the t-test for a mean is robust for large $m$ (CLT; i.i.d., finite variance).

*Why $\chi^2_k\approx N(k,2k)$ for large $k$?*

- CLT on a sum of $k$ i.i.d. $Z^2$ w/ mean 1 & variance 2.
```

&nbsp;

### Beta
- **What**: Description of **uncertainty about a probability**.
- **Why**: We don't always know the actual probabilities.
    - We need something to monitor probabilities themselves.
    - This thing updates cleanly when we get new info about the probability.
    - This thing represents prior beliefs.

```{note} Math
:class: dropdown
Notation:
$$
X\sim\text{Beta}(\alpha,\beta)
$$

PDF/CDF:
$$\begin{align*}
&\text{PDF:} && f(x)=\frac{1}{B(\alpha,\beta)}x^{\alpha-1}(1-x)^{\beta-1} \\
&\text{CDF:} && F(x)=\frac{B(x;\alpha,\beta)}{B(\alpha,\beta)}
\end{align*}$$
- Support: $x\in[0,1]$: Probability.
- Params:
    - Standard:
        - $\alpha>0$: Behavior control near 0.
        - $\beta>0$: Behavior control near 1.
    - Interpretable:
        - $\mu=E[X]=\frac{\alpha}{\alpha+\beta}\in(0,1)$: Mean.
        - $\kappa=\alpha+\beta>0$: Concentration.
            - $\alpha=\mu\kappa,\quad\beta=(1-\mu)\kappa$.

Beta Function:
$$\begin{align*}
&B(\alpha,\beta)=\int_0^1t^{\alpha-1}(1-t)^{\beta-1}dt=\frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)} \\
&B(x;\alpha,\beta)=\int_0^xt^{\alpha-1}(1-t)^{\beta-1}dt \\
&B(\alpha+1,\beta)=\frac{\alpha}{\alpha+\beta}B(\alpha,\beta) \\
&B(\alpha,\beta+1)=\frac{\beta}{\alpha+\beta}B(\alpha,\beta)
\end{align*}$$

Shape:
- Location: Mass sits at $\mu$.
- Spread: For fixed $\mu$, larger $\kappa$ concentrates around $\mu$ (smaller variance).
- Skewness: For fixed $\kappa$, moving $\mu$ toward 0 or 1 shifts skew accordingly.
- Tail behavior:
    - $x\rightarrow0: f(x)\propto x^{\alpha-1}$
    - $x\rightarrow1: f(x)\propto (1-x)^{\beta-1}$
- Typical global shapes:
    - $\alpha>1,\beta>1$: Unimodal at $\frac{\alpha-1}{\alpha+\beta-2}$.
    - $\alpha<1,\beta<1$: U-shaped (mass near both ends).
    - $\alpha<1,\beta>1$: J-shaped (mass near 0).
    - $\alpha>1,\beta<1$: Reverse J-shaped (mass near 1).
    - $\alpha=\beta=1$: Uniform(0,1).

Moments:
|  |  |
|:------ |:------- |
| Mean     | $E[X]=\mu$  |
| Variance | $Var[X]=\frac{\mu(1-\mu)}{\kappa+1}$ |
| All moments    | $(a)_k=a(a+1)\cdots(a+k-1)\rightarrow E[X^k]=\frac{(\alpha)_k}{(\alpha+\beta)_k}$ |

Sampling:
- Gamma ratio:
    1. Sample $G_1\sim\Gamma(\alpha,1),G_2\sim\Gamma(\beta,1)$.
    2. Output $X=\frac{G_1}{G_1+G_2}$.
```

&nbsp;

#### Dirichlet
- **What**: Beta BUT over $K$-category probability vectors.
- **Why**: Uncertainty about a Categorical/Multinomial $\mathbf{p}$.
    - Conjugate → posterior update = add counts.
- **How**: Draw $K$ independent Gammas → normalize to sum 1.

```{note} Math
:class: dropdown
Notation:

$$
\mathbf{X}\sim\text{Dir}(\boldsymbol\alpha)
$$

PDF/CDF:

$$\begin{align*}
&\text{PDF:} && f(\mathbf{x})=\frac{1}{B(\boldsymbol\alpha)}\prod_{k=1}^Kx_k^{\alpha_k-1} \\
&\text{CDF:} && \text{No closed form (rarely used).}
\end{align*}$$

- Support: $\mathbf{x}\in\Delta^{K-1}=\{\mathbf{x}\in\mathbb{R}^K:x_k>0,\sum_kx_k=1\}$: Probability vector (simplex).
    - Density w.r.t. $(x_1,\dots,x_{K-1})$; $x_K=1-\sum_{k<K}x_k$.
- Params:
    - $\boldsymbol\alpha=(\alpha_1,\dots,\alpha_K)$, $\alpha_k>0$: Concentration params (pseudo-counts).
    - $\alpha_0=\sum_k\alpha_k$: Total concentration.
- Multivariate Beta function:

$$
B(\boldsymbol\alpha)=\frac{\prod_{k=1}^K\Gamma(\alpha_k)}{\Gamma(\alpha_0)}
$$

Shape:
- Location: Mass centers on $\frac{\boldsymbol\alpha}{\alpha_0}$.
- Spread: For fixed mean, $\alpha_0$⬆️ → mass concentrates around the mean.
- All $\alpha_k=1$: Uniform on the simplex.
- All $\alpha_k>1$: Unimodal at $x_k=\frac{\alpha_k-1}{\alpha_0-K}$.
- $\alpha_k<1$: Mass pushed to corners & edges → sparse $\mathbf{x}$.

Moments:
|  |  |
|:------ |:------- |
| Mean   | $E[X_k]=\frac{\alpha_k}{\alpha_0}$ |
| Variance | $Var[X_k]=\frac{\alpha_k(\alpha_0-\alpha_k)}{\alpha_0^2(\alpha_0+1)}$ |
| Covariance | $Cov(X_i,X_j)=-\frac{\alpha_i\alpha_j}{\alpha_0^2(\alpha_0+1)},\quad i\neq j$ |
| Log moment | $E[\log X_k]=\varphi(\alpha_k)-\varphi(\alpha_0)$ |
| MGF    | Exists (bounded support); no elementary closed form |

Entropy:

$$
h(\mathbf{X})=\log B(\boldsymbol\alpha)+(\alpha_0-K)\varphi(\alpha_0)-\sum_{k=1}^K(\alpha_k-1)\varphi(\alpha_k)
$$

- $\varphi$: Digamma function (see Gamma).
- Max at $\boldsymbol\alpha=\mathbf{1}$ (uniform): $h=-\log\left((K-1)!\right)$.

Transformations:
|  |  |
|:-------- |:------- |
| Aggregation | Merge $i,j$: $(\dots,X_i+X_j,\dots)\sim\text{Dir}(\dots,\alpha_i+\alpha_j,\dots)$ |
| Marginal | $X_k\sim\text{Beta}(\alpha_k,\alpha_0-\alpha_k)$ |
| Neutrality | $\frac{(X_2,\dots,X_K)}{1-X_1}\sim\text{Dir}(\alpha_2,\dots,\alpha_K)$, independent of $X_1$ |
| Normalized Gammas | $G_k\sim\text{Gamma}(\alpha_k,1)$ independent $\rightarrow\frac{\mathbf{G}}{\sum_kG_k}\sim\text{Dir}(\boldsymbol\alpha)$<br>Independent of $\sum_kG_k\sim\text{Gamma}(\alpha_0,1)$. |

Sampling:
1. Draw $G_k\sim\text{Gamma}(\alpha_k,1)$ independently, $k=1,\dots,K$.
2. Output $X_k=\frac{G_k}{\sum_lG_l}$.

Relationship w/ other distributions:
- Beta: $K=2\rightarrow X_1\sim\text{Beta}(\alpha_1,\alpha_2)$.
- Categorical/Multinomial: Conjugate prior ([Conjugacy](bayes.md#conjugacy)):

$$
\mathbf{p}\sim\text{Dir}(\boldsymbol\alpha),\ \mathbf{c}|\mathbf{p}\sim\text{Mult}(n,\mathbf{p})\longrightarrow\mathbf{p}|\mathbf{c}\sim\text{Dir}(\boldsymbol\alpha+\mathbf{c})
$$

- $\mathbf{c}$: Count vector, $\sum_kc_k=n$.
- Posterior mean $\frac{\alpha_k+c_k}{\alpha_0+n}$ → $\alpha_k$ = pseudo-counts; $\alpha_k=1$ = add-one (Laplace) smoothing.
- Dirichlet-Multinomial: Integrate out $\mathbf{p}$ → overdispersed counts (multivariate Beta-Binomial).

KLD:
- $P=\text{Dir}(\boldsymbol\alpha),Q=\text{Dir}(\boldsymbol\beta)$, $\beta_0=\sum_k\beta_k$:

$$
D_{KL}(P\|Q)=\log\Gamma(\alpha_0)-\sum_{k=1}^K\log\Gamma(\alpha_k)-\log\Gamma(\beta_0)+\sum_{k=1}^K\log\Gamma(\beta_k)+\sum_{k=1}^K(\alpha_k-\beta_k)\left(\varphi(\alpha_k)-\varphi(\alpha_0)\right)
$$
```

```{tip} Derivation
:class: dropdown
*Why are normalized Gammas Dirichlet?*

1. Joint density of independent $G_k\sim\text{Gamma}(\alpha_k,1)$: $\prod_k\frac{g_k^{\alpha_k-1}e^{-g_k}}{\Gamma(\alpha_k)}$.
2. Change variables: $S=\sum_kG_k$, $X_k=\frac{G_k}{S}$ ($k<K$) → Jacobian $S^{K-1}$.
3. Substitute $g_k=Sx_k$:

    $$
    \frac{S^{\alpha_0-1}e^{-S}}{\Gamma(\alpha_0)}\cdot\frac{\Gamma(\alpha_0)}{\prod_k\Gamma(\alpha_k)}\prod_kx_k^{\alpha_k-1}
    $$

4. Factorizes → $S\sim\text{Gamma}(\alpha_0,1)$ ⊥ $\mathbf{X}\sim\text{Dir}(\boldsymbol\alpha)$.

*Why is the posterior $\text{Dir}(\boldsymbol\alpha+\mathbf{c})$?*

1. Multinomial likelihood kernel: $\prod_kp_k^{c_k}$.
2. Dirichlet prior kernel: $\prod_kp_k^{\alpha_k-1}$.
3. Multiply → $\prod_kp_k^{\alpha_k+c_k-1}$ → $\text{Dir}(\boldsymbol\alpha+\mathbf{c})$.
```

````{important} Code
:class: dropdown
```python
import torch

class Dirichlet:
    def __init__(self, alpha):
        self.alpha = alpha                                  ## (K,), all > 0

    def sample(self, m):
        ## independent G_k ~ Gamma(alpha_k, rate=1)
        g = torch.distributions.Gamma(self.alpha, torch.ones_like(self.alpha)).sample((m,))
        return g / g.sum(dim=-1, keepdim=True)              ## normalize -> point on the simplex

    def update(self, counts):
        ## conjugacy: Dir(alpha) prior + Multinomial counts -> Dir(alpha + counts)
        return Dirichlet(self.alpha + counts)

    def mean(self):
        return self.alpha / self.alpha.sum()

## Example
prior = Dirichlet(torch.tensor([1.0, 1.0, 1.0]))           ## uniform on the 2-simplex
x = prior.sample(4)
print(x.shape)                                              ## torch.Size([4, 3]); each row sums to 1
post = prior.update(torch.tensor([5.0, 0.0, 2.0]))
print(post.mean())                                          ## tensor([0.6000, 0.1000, 0.3000])
```
````

```{attention} Q&A
:class: dropdown
*What does $\alpha_0$ control?*

- Prior strength = pseudo-sample size.
- $\alpha_0$⬆️ → $Var[X_k]$⬇️ ($\propto\frac{1}{\alpha_0+1}$ for a fixed mean).

*Why use $\alpha_k<1$ in topic models?*

- Mass near corners → sparse vectors.
- Latent Dirichlet Allocation: each document uses few topics; each topic uses few words.

*Limitations?*

- All covariances negative & fixed by the means & $\alpha_0$ → no free correlation structure.
- Positive correlations → logistic-Normal (softmax of an MVN).

*How does it relate to Beta?*

- $K=2$ → Beta.
- Each marginal $X_k$ is Beta (aggregation of the other components).
```

&nbsp;

### Student-t
- **What**: $\frac{Z}{\sqrt{V/\nu}}$, $Z\sim N(0,1)$ ⊥ $V\sim\chi^2_\nu$.
    - = Normal w/ random (Gamma) precision → heavy-tailed bell curve.
- **Why**:
    - Small-sample inference: $\sigma$ unknown → estimate by $S$ → extra uncertainty → tails heavier than Normal.
    - Robust modeling: Outliers less surprising → less influence on the fit.
- **How**: Draw a precision $\lambda\sim\text{Gamma}(\frac\nu2,\frac\nu2)$ → draw $N(\mu,\frac{\sigma^2}{\lambda})$.
    - Occasional small $\lambda$ → wide Normal → heavy tails.

```{note} Math
:class: dropdown
Notation:

$$
X\sim t_\nu(\mu,\sigma^2),\quad t_\nu=t_\nu(0,1)
$$

PDF/CDF:

$$\begin{align*}
&\text{PDF:} && f(x)=\frac{\Gamma\left(\frac{\nu+1}{2}\right)}{\Gamma\left(\frac\nu2\right)\sqrt{\nu\pi}\,\sigma}\left(1+\frac{(x-\mu)^2}{\nu\sigma^2}\right)^{-\frac{\nu+1}{2}} \\
&\text{CDF:} && F(x)=1-\frac12\frac{B\left(\frac{\nu}{\nu+z^2};\frac\nu2,\frac12\right)}{B\left(\frac\nu2,\frac12\right)},\quad z=\frac{x-\mu}{\sigma}\geq0
\end{align*}$$

- $x<\mu$: $F(x)=1-F(2\mu-x)$ (symmetry).
- Support: $x\in\mathbb{R}$.
- Params:
    - $\nu>0$: Degrees of freedom (tail weight).
    - $\mu\in\mathbb{R}$: Location (median = mode; mean if $\nu>1$).
    - $\sigma>0$: Scale (≠ std).
- $B(\cdot;\cdot,\cdot)$, $B(\cdot,\cdot)$: Incomplete & complete Beta functions (see Beta).

Shape:
- Symmetric bell, unimodal at $\mu$.
- Polynomial tails: $f(x)\propto|x|^{-(\nu+1)}$ as $|x|\to\infty$ → heavier than Normal for every finite $\nu$.
- $\nu$⬇️ → tails⬆️; $\nu\to\infty$ → Normal.

Moments:
|  |  |
|:------ |:------- |
| Mean   | $E[X]=\mu$, $\nu>1$ (undefined for $\nu\leq1$) |
| Variance | $Var[X]=\frac{\nu}{\nu-2}\sigma^2$, $\nu>2$ ($\infty$ for $1<\nu\leq2$) |
| Finite moments | $s\geq0$: $E[\|X\|^s]<\infty\Leftrightarrow s<\nu$ |
| Excess kurtosis | $\frac{6}{\nu-4}$, $\nu>4$ |
| MGF    | Non-existent |

Entropy:

$$
h(X)=\frac{\nu+1}{2}\left[\varphi\left(\frac{\nu+1}{2}\right)-\varphi\left(\frac\nu2\right)\right]+\log\left(\sqrt\nu\,B\left(\frac\nu2,\frac12\right)\right)+\log\sigma
$$

- $\varphi$: Digamma function (see Gamma).
- $\nu=1$ (Cauchy): $h=\log(4\pi\sigma)$.
- $\nu\to\infty$: $h\to\frac12\log(2\pi e\sigma^2)$ (Normal).

Transformations:
|  |  |
|:-------- |:------- |
| Affine | $a\neq0\rightarrow aX+b\sim t_\nu(a\mu+b,a^2\sigma^2)$ |
| Square | $T\sim t_\nu\rightarrow T^2\sim\text{F}_{1,\nu}$ |
| Scale mixture of Normals | $\lambda\sim\text{Gamma}(\frac\nu2,\frac\nu2),\ X\|\lambda\sim N(\mu,\frac{\sigma^2}{\lambda})\rightarrow X\sim t_\nu(\mu,\sigma^2)$ |

Sampling:
1. Draw $Z\sim N(0,1)$ & $V\sim\chi^2_\nu$ independently.
2. Output $X=\mu+\sigma\frac{Z}{\sqrt{V/\nu}}$.

Relationship w/ other distributions:
- Normal & Chi-squared: Definition $\frac{Z}{\sqrt{V/\nu}}$.
- Normal limit: $\nu\to\infty\rightarrow N(\mu,\sigma^2)$ ← $\frac V\nu\to1$ (LLN).
- Cauchy: $\nu=1$:

$$
f(x)=\frac{1}{\pi\sigma}\frac{1}{1+\left(\frac{x-\mu}{\sigma}\right)^2}
$$

- t-statistic (Normal data, $\bar X$ & $S$ as in Chi-Squared):

$$
X_i\overset{i.i.d.}{\sim}N(\mu,\sigma^2)\longrightarrow\frac{\bar X-\mu}{S/\sqrt m}\sim t_{m-1}
$$

- Bayesian: Normal data, unknown mean & variance, conjugate prior → posterior of the mean & posterior predictive are Student-t.
- [Robust Regression](bayes.md#robust-regression): Student-t errors via the Gamma scale mixture.

KLD:
- No closed form in general → numerical integration or Monte Carlo.
```

```{attention} Q&A
:class: dropdown
*Why $t$ instead of $N(0,1)$ for small-sample tests?*

- $S$ replaces $\sigma$ → the ratio has extra randomness → heavier tails → wider CIs.
- $m$⬆️ → $S\to\sigma$ → $t_{m-1}\to N(0,1)$.

*Why is it robust to outliers?*

- Score of $-\log f$ in the residual $\varepsilon=x-\mu$:

    $$
    \frac{\partial}{\partial\varepsilon}\left(-\log f\right)=\frac{(\nu+1)\varepsilon}{\nu\sigma^2+\varepsilon^2}\xrightarrow{|\varepsilon|\to\infty}0
    $$

- Large residuals → vanishing pull on $\mu$. Normal: pull $\frac{\varepsilon}{\sigma^2}$ grows w/o bound.

*What breaks at small $\nu$?*

- $1<\nu\leq2$: Infinite variance → finite-variance CLT fails.
- $\nu\leq1$: No mean → LLN fails.
    - Cauchy: mean of $m$ i.i.d. Cauchy = same Cauchy → never concentrates.

*Is $\sigma$ the standard deviation?*

- ❌. $Var=\frac{\nu}{\nu-2}\sigma^2>\sigma^2$ ($\nu>2$).

*Why Student-t in t-SNE?*

- Low-dim similarities use a $t_1$ (Cauchy) kernel.
- Heavy tails → moderately dissimilar points can sit far apart → less crowding.

*How do you choose $\nu$?*

- Fix a small $\nu$ or learn it (MLE / prior).
- $\nu$⬇️ → more robust, less efficient if data are truly Normal.
```

&nbsp;

## Exponential Family
- **What**: Distributions w/ log-density = (natural param)$^T$(sufficient statistic) − normalizer + param-free base term.
- **Why**: Unification. Prove a property once → it holds for every member.
- **How**: Factor the log-density into 3 parts.
    1. **Base measure** $r(x)$: no param.
    2. **Sufficient statistic** $T(x)$: enters linearly w/ the **natural param** $\boldsymbol\eta$.
    3. **Log-partition** $A(\boldsymbol\eta)$: normalizer; no $x$.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\boldsymbol\eta\in\Omega\subseteq\mathbb{R}^s$: Natural param.
- Misc:
    - $s$: #natural params.
    - $T(x)\in\mathbb{R}^s$: Sufficient statistic.
    - $r(x)\geq0$: Base measure (often $h(x)$; renamed ← $h$ = differential entropy on this page).
    - $A(\boldsymbol\eta)$: Log-partition function.

Definition:

$$
f(x|\boldsymbol\eta)=r(x)\exp\left(\boldsymbol\eta^TT(x)-A(\boldsymbol\eta)\right),\qquad A(\boldsymbol\eta)=\log\int r(x)\exp\left(\boldsymbol\eta^TT(x)\right)dx
$$

- Discrete $x$: $\int\to\sum$.
- $\Omega=\{\boldsymbol\eta:A(\boldsymbol\eta)<\infty\}$: Natural param space (convex).
- **Minimal**: Components of $T$ affinely independent → $\boldsymbol\eta$ identifiable.

Properties ($\boldsymbol\eta$ in the interior of the natural parameter space):
- Mean: $\nabla A(\boldsymbol\eta)=E[T(X)]=\boldsymbol\mu$ (**mean param**).
- Covariance: $\nabla^2A(\boldsymbol\eta)=Cov[T(X)]\succeq0$ → $A$ convex (strictly if minimal) → $\boldsymbol\eta\leftrightarrow\boldsymbol\mu$ bijective (if minimal).
- Fisher information: $\mathcal{I}(\boldsymbol\eta)=\nabla^2A(\boldsymbol\eta)$.
- Cumulants of $T$: $\log E\left[e^{\mathbf{u}^TT(X)}\right]=A(\boldsymbol\eta+\mathbf{u})-A(\boldsymbol\eta)$.
- Sufficiency: i.i.d. $x_1,\dots,x_m$ → $\sum_{i=1}^mT(x_i)$ is sufficient; dim $s$ for every $m$. See [Sufficient Statistics](bayes.md#sufficient-statistics).
- MLE = moment matching:

$$
\ell(\boldsymbol\eta)=\boldsymbol\eta^T\sum_{i=1}^mT(x_i)-mA(\boldsymbol\eta)+\text{const}\ \Longrightarrow\ \nabla A(\hat{\boldsymbol\eta})=\frac1m\sum_{i=1}^mT(x_i)
$$

- $\ell$: Log-likelihood; concave in $\boldsymbol\eta$.
- Solution exists iff the empirical mean of $T$ lies in the interior of the set of attainable means.
- Conjugate prior ([Conjugacy](bayes.md#conjugacy)):

$$
\pi(\boldsymbol\eta|\boldsymbol\tau,\nu_0)\propto\exp\left(\boldsymbol\eta^T\boldsymbol\tau-\nu_0A(\boldsymbol\eta)\right)\ \xrightarrow{\text{data}}\ \boldsymbol\tau+\sum_{i=1}^mT(x_i),\ \ \nu_0+m
$$

- $\boldsymbol\tau\in\mathbb{R}^s$: Prior pseudo-sum of sufficient statistics.
- $\nu_0>0$: Prior pseudo-count.
- Max entropy:

$$
\arg\max_f\left(-\int f\log\frac fr\,dx\right)\ \text{s.t.}\ E_f[T(X)]=\mathbf{t}\ \Longrightarrow\ f(x)\propto r(x)\exp\left(\boldsymbol\eta^TT(x)\right)
$$

- $\mathbf{t}$: Target moments; $\boldsymbol\eta$ = Lagrange multipliers chosen to meet them.
- $r\equiv1$ → ordinary $h$. E.g., fixed mean & variance on $\mathbb{R}$ → Normal; fixed mean on $[0,\infty)$ → Exponential.
- KLD = **Bregman divergence** of $A$:

$$
D_{KL}(f_{\boldsymbol\eta_1}\|f_{\boldsymbol\eta_2})=A(\boldsymbol\eta_2)-A(\boldsymbol\eta_1)-(\boldsymbol\eta_2-\boldsymbol\eta_1)^T\nabla A(\boldsymbol\eta_1)
$$
```

```{tip} Derivation
:class: dropdown
*Why is $\nabla A=E[T]$ & $\nabla^2A=Cov[T]$?*

1. $\nabla A(\boldsymbol\eta)=\frac{\int T(x)r(x)e^{\boldsymbol\eta^TT(x)}dx}{\int r(x)e^{\boldsymbol\eta^TT(x)}dx}=E[T(X)]$ (differentiate under $\int$; valid in the interior of $\Omega$).
2. Differentiate again: $\nabla^2A=E[TT^T]-E[T]E[T]^T=Cov[T]$.

*Why is MLE moment matching?*

1. $\nabla\ell=\sum_iT(x_i)-m\nabla A(\boldsymbol\eta)=0$.
2. → $E_{\hat{\boldsymbol\eta}}[T]=\frac1m\sum_iT(x_i)$.
3. Hessian $=-m\,Cov[T]\preceq0$ → global max.

*Why is KLD a Bregman divergence?*

1. $\log\frac{f_{\boldsymbol\eta_1}}{f_{\boldsymbol\eta_2}}=(\boldsymbol\eta_1-\boldsymbol\eta_2)^TT(x)-A(\boldsymbol\eta_1)+A(\boldsymbol\eta_2)$ ($r$ cancels).
2. Take $E_{\boldsymbol\eta_1}$: $E[T]=\nabla A(\boldsymbol\eta_1)$ → formula.

*Why does max entropy give this form?*

1. Lagrangian: $-\int f\log\frac fr+\boldsymbol\eta^T\left(\int fT-\mathbf{t}\right)+\lambda_0\left(\int f-1\right)$.
2. Pointwise derivative in $f$: $-\log\frac fr-1+\boldsymbol\eta^TT+\lambda_0=0$.
3. → $f=r\exp(\boldsymbol\eta^TT+\lambda_0-1)$; $\lambda_0$ normalizes → $A(\boldsymbol\eta)=1-\lambda_0$.
```

```{note} Example
:class: dropdown
- Bernoulli: $p^x(1-p)^{1-x}=\exp\left(x\log\frac{p}{1-p}+\log(1-p)\right)$.
    - $\eta=\log\frac p{1-p}$ (logit), $T(x)=x$, $A(\eta)=\log(1+e^\eta)$, $r(x)=1$.
    - $A'(\eta)=\frac{1}{1+e^{-\eta}}=p$ (sigmoid); $A''(\eta)=p(1-p)$ = variance.
- Poisson: $e^{-\lambda}\frac{\lambda^x}{x!}=\frac1{x!}\exp(x\log\lambda-\lambda)$.
    - $\eta=\log\lambda$, $A(\eta)=e^\eta$ → $A'=A''=\lambda$ → mean = variance.
- Normal: $-\frac{(x-\mu)^2}{2\sigma^2}-\log\sigma=\frac{\mu}{\sigma^2}x-\frac{1}{2\sigma^2}x^2-\frac{\mu^2}{2\sigma^2}-\log\sigma$.
    - $T(x)=(x,x^2)$, $\boldsymbol\eta=\left(\frac{\mu}{\sigma^2},-\frac{1}{2\sigma^2}\right)$.
```

```{dropdown} Table: Members
| Distribution | $T(x)$ | $\boldsymbol\eta$ | $A(\boldsymbol\eta)$ | $r(x)$ |
|:--|:--|:--|:--|:--|
| $\text{Bern}(p)$ | $x$ | $\log\frac{p}{1-p}$ | $\log(1+e^\eta)$ | $1$ |
| $\text{Bin}(n,p)$, $n$ fixed | $x$ | $\log\frac{p}{1-p}$ | $n\log(1+e^\eta)$ | $\binom nx$ |
| $\text{NB}(r,p)$, $r$ fixed | $x$ | $\log(1-p)$ | $-r\log(1-e^\eta)$ | $\binom{x+r-1}{x}$ |
| $\text{Cat}(\mathbf{p})$ | one-hot $\mathbf{y}$ | $\eta_k=\log p_k$ | $\log\sum_ke^{\eta_k}$ | $1$ |
| $\text{Mult}(n,\mathbf{p})$, $n$ fixed | counts $\mathbf{x}$ | $\eta_k=\log p_k$ | $n\log\sum_ke^{\eta_k}$ | $\frac{n!}{\prod_kx_k!}$ |
| $\text{Pois}(\lambda)$ | $x$ | $\log\lambda$ | $e^\eta$ | $\frac1{x!}$ |
| $N(\mu,\sigma^2)$ | $(x,x^2)$ | $\left(\frac{\mu}{\sigma^2},-\frac{1}{2\sigma^2}\right)$ | $-\frac{\eta_1^2}{4\eta_2}-\frac12\log(-2\eta_2)$ | $\frac{1}{\sqrt{2\pi}}$ |
| $\text{Exp}(\lambda)$ | $x$ | $-\lambda$ | $-\log(-\eta)$ | $1$ |
| $\text{Gamma}(\alpha,\beta)$ | $(\log x,x)$ | $(\alpha,-\beta)$ | $\log\Gamma(\eta_1)-\eta_1\log(-\eta_2)$ | $\frac1x$ |
| $\text{Beta}(\alpha,\beta)$ | $(\log x,\log(1-x))$ | $(\alpha,\beta)$ | $\log B(\eta_1,\eta_2)$ | $\frac{1}{x(1-x)}$ |
| $\text{Dir}(\boldsymbol\alpha)$ | $(\log x_1,\dots,\log x_K)$ | $\boldsymbol\alpha$ | $\log B(\boldsymbol\eta)$ | $\frac{1}{\prod_kx_k}$ |

- Categorical/Multinomial rows: overcomplete ($\boldsymbol\eta$ identifiable only up to an additive constant) → $\nabla A$ = softmax. Drop 1 class for a minimal form.
- In the NB row, $r$ = #successes (NB param), not the base measure.

❌ Not members:
- $\text{Unif}(a,b)$, unknown $a,b$: Support depends on the params.
- $\text{Laplace}(\mu,b)$, unknown $\mu$: $|x-\mu|$ is not $\boldsymbol\eta(\mu)^TT(x)$ for a finite $T$ (✅ member if $\mu$ known: $T(x)=|x-\mu|$).
- Student-t: $-\frac{\nu+1}{2}\log\left(1+\frac{(x-\mu)^2}{\nu\sigma^2}\right)$ is not linear in a finite $T(x)$.
- Binomial / NB w/ unknown $n$ / $r$.
- Mixtures (e.g., GMM): log of a sum.
```

```{attention} Q&A
:class: dropdown
*Is the exponential family the Exponential distribution?*

- ❌. Exponential distribution = 1 member of the family.

*Why is MLE easy for members?*

- $\ell(\boldsymbol\eta)$ concave → every stationary point is a global max.
- Stationarity = moment matching: model $E[T]$ = empirical mean of $T$.
- Hessian $=-m\,Cov[T]$ → Newton / IRLS.

*Why do members have conjugate priors?*

- Likelihood depends on data only via $\sum_iT(x_i)$ & $m$, in exp-linear form.
- Prior of the same form → product keeps the form → update = add statistics.

*How does it connect to GLMs?*

- [GLM](../../ml/supervised.md#glm) response = member; canonical link sets $\boldsymbol\eta=\mathbf{x}^T\mathbf{w}$.
    - Bernoulli → logistic regression; Categorical → softmax regression; Poisson → Poisson regression.
- GLM page form $\exp\left(\frac{y\theta-b(\theta)}{a(\phi)}+c(y,\phi)\right)$, $\phi$ fixed → $T(y)=y$, $\eta=\frac{\theta}{a(\phi)}$, $A=\frac{b(\theta)}{a(\phi)}$.

*Why are softmax & cross-entropy everywhere?*

- Categorical in natural params: $\nabla A$ = softmax.
- NLL = cross-entropy; gradient w.r.t. $\boldsymbol\eta$ = softmax − one-hot = $E[T]-T$.

*Which families are NOT members, and why does it matter?*

- i.i.d., regular, param-independent support + sufficient statistic of fixed dim for every $m$ ⇒ exponential family (Pitman–Koopman–Darmois).
- Uniform escapes ← support depends on params (sufficient statistic $(\min,\max)$).
- Student-t, mixtures: no fixed-dim sufficient statistic → no closed-form MLE / conjugacy.

*How does EM handle mixtures of members?*

- Latent assignments → complete-data likelihood is exponential family.
- M-step = moment matching on expected sufficient statistics.

*Natural vs mean params: when to use which?*

- Mean params $\boldsymbol\mu$: MLE is the sample mean of $T$; interpretable.
- Natural params $\boldsymbol\eta$: unconstrained in many cases (logit, log-rate) → linear models act here.
```

&nbsp;
