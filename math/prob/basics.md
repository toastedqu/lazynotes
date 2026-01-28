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
# Basics
Study notes from {cite:t}`handbook_math` and {cite:t}`prob_lifesaver`.

## Probability
### Sample Space 
- **What**: Set of all possible outcomes of a random experiment ($\Omega$).
    - e.g., Coin flip: $\Omega=\{\text{head}, \text{tail}\}$.
    - e.g., Time till the universe ends: $\Omega=[0,\infty)$.

### Event
- **What**: A subset of the sample space ($A\subseteq\Omega$).
    - e.g., Coin flip is heads: $A=\{\text{head}\}$.
    - e.g., The universe ends within 60 seconds: $A=[0,60]$.

### σ-algebra
- **What**: A menu of allowed events in the sample space w/ 3 conditions ($\mathcal{F}$):
    1. **It contains the whole space**: $\Omega\in\mathcal{F}$.
    2. **It is closed under complements**: $A\in\mathcal{F}\rightarrow A^{c}=\Omega\backslash A \in\mathcal{F}$.
    3. **It is closed under countable unions**: $A_1,A_2,\cdots\in\mathcal{F}\rightarrow \bigcup_{i=1}^\infty A_i\in\mathcal{F}$.
- **Why**:
    - *Why these 3 specific conditions?*
        - These 3 closure rules are the minimum for events to behave like sensible boolean (yes/no) questions & for probability to work reliably in infinite processes.
        - Rule 1 ← "Something happened for sure" is an event.
        - Rule 2 ← "$A$ happened" and "$A$ didn't happen" are both events.
        - Rule 3 ← "At least one of these events happened" is an event.
    - *Why countable unions?*
        - **Countable union**: An OR-list we can list one-by-one in a sequence.
            - e.g., Set of rational numbers $\mathbb{Q}$.
        - **Uncountable union**: An OR-list we cannot list one-by-one in a sequence.
            - e.g., Set of every real number between 0 and 1 $[0,1]$.
        - We cannot assign probabilities to EVERY subset in an uncountable union.
            - e.g., It's impossible to assign a probability to a Vitali set in $[0,1]$.
            - We can still have σ-algebra on $[0,1]$, but it won't work on EVERY subset of it.
- **How**: Allowed events if $A_1,A_2,\cdots\in\mathcal{F}$:

| Operation              | Notation                   |
|:---------------------- |:-------------------------- |
| Complement             | $A^{c}$                    |
| Countable union        | $\bigcup_{i=1}^\infty A_i$ |
| Finite union           | $\bigcup_{i=1}^n A_i$      |
| Countable intersection | $\bigcap_{i=1}^\infty A_i$ |
| Finite intersection    | $\bigcap_{i=1}^n A_i$      |
| Difference             | $A\backslash B$            |
| Empty set              | $\varnothing$              |

&nbsp;

### Probability
- **What**: A real function $P:\mathcal{F}\rightarrow [0,1]$ satisfying the 3 Kolmogorov axioms:
    - **Non-negativity**: $\forall A\in\mathcal{F}: P(A)\geq 0$
    - **Normalization**: $P(\Omega)=1$ (and $P(\varnothing)=0$)
    - **Countable additivity**: If $A_i$ are mutually exclusive events, then $P(\bigcup_{i=1}^\infty A_i)=\sum_{i=1}^\infty P(A_i)$
- **Why**:
    - *Why these 3 specific axioms?*
        - These 3 axioms are the minimal rules for "measuring uncertainty as a number" to behave consistently w/ ordinary logic about events AND to work for infinite processes.
        - Axiom 1 ← A negative probability means "less than impossible". What?
        - Axiom 2 ← "Anything in the sample space can happen" is a certainty, so 1.
        - Axiom 3 ← If events cannot happen together, then the chance of "one of them happens" should be the sum of their chances.
- **How**: Rules for probability:

| Rule                  | Formula                                                                                                            |
|:-------------------------------- |:-------------------------------------------------------------------------------------------------------------------- |
| Complement                       | $P(A^c)=1-P(A)$                                                                                                      |
| Monotonicity                     | $A\subseteq B\rightarrow P(A)\leq P(B)$                                                                              |
| Difference                       | $A\subseteq B\rightarrow P(B\backslash A)=P(B)-P(A)$                                                                 |
| Inclusion-Exclusion (2 events)   | $P(A\cup B)=P(A)+P(B)-P(A\cap B)$                                                                                    |
| Boole's Inequality (union bound) | $P\left(\bigcup_i A_i\right)\leq \sum_i P(A_i)$                                                                      |
| Mutual exclusivity (disjoint)    | $A\cap B=\varnothing\rightarrow P(A\cap B)=0$                                                                        |
| Independence                     | $P(A\cap B)=P(A)P(B)$                                                                                                |
| Continuity from above            | $A_1\subseteq A_2\subseteq\cdots\rightarrow P!\left(\bigcup_{i=1}^\infty A_i\right)=\lim_{i\rightarrow\infty}P(A_i)$ |
| Continuity from below            | $A_1\supseteq A_2\supseteq\cdots\rightarrow P!\left(\bigcap_{i=1}^\infty A_i\right)=\lim_{i\rightarrow\infty}P(A_i)$ |

&nbsp;

### Conditional Probability
- **What**: Probability of event $B$ when event $A$ has already occurred.
$$
P(B|A)=\frac{P(A\cap B)}{P(A)}
$$
- **Why**: Conditioning updates probabilities when we learn new info.
- **How**: Rules for conditional probability:

| Rule           | Formula                                                                                         |
|:------------------------|:-----------------------------------------------------------------------------------------------|
| Multiplication           | $P(A\cap B)=P(A\mid B)P(B)=P(B\mid A)P(A)$                                                      |
| Chain rule               | $P\left(\bigcap_{i=1}^n A_i\right)=\prod_{i=1}^n P\left(A_i\mid \bigcap_{j=1}^{i-1} A_j\right)$ |
| Law of total probability | $P(B)=\sum_i P(B\mid A_i)P(A_i)$                                                                |
| Bayes' theorem           | $P(A\mid B)=\frac{P(B\mid A)P(A)}{P(B)}$                                                        |
| Independence             | $P(A\mid B)=P(A)$                                                                               |

&nbsp;

### Probability Space
- **What**: $(\Omega,\mathcal{F},P)$
- **Why**:
    - We need a container that packages everything we need to do probability in a mathematically consistent way.
    - It specifies what random outcomes are allowed.
    - It specifies how likely each event is.
    - It provides the base for random variables ← ANY random variable is defined on this space.
    - It functions as the ground truth model for a random event. We can derive everything from it.

&nbsp;

## Random Variables & Distributions
### Random Variable
- **What**: A variable which takes its value randomly from a subset of $\mathbb{R}$.

&nbsp;

### Distribution
- **What**: The probability that the random variable $X$ takes a value lower than $x$:
$$\begin{align*}
&\text{Discrete:}   && F(x)=P(X\leq x)=\sum_{x_i\leq x}P(X=x_i) \\
&\text{Continuous:} && F(x)=P(X\leq x)=\int_{-\infty}^xf(t)dt
\end{align*}$$
- **Why**: Distribution tells us everything probabilistic about a random variable.
- **How**: Properties:
    - $F(-\infty)=0, F(\infty)=1$
    - $F(x)$ is non-decreasing.
    - $F(x)$ is continuous on the right (left if defined as $P(X<x)$).
    - $F'(x)=f(x)$ if $f(x)$ is continuous.
    - No probability at any point for continuous distributions: $\forall c\in\mathbb{R}: P(X=c)=0$

&nbsp;

### Expected Value / Mean
- **What**: Average outcome if repeating a random process multiple times (i.e., weigh each possible outcome by its likelihood).
$$\begin{align*}
&\text{Discrete:}   && E[X]=\sum_i x_ip_i \\
&\text{Continuous:} && E[X]=\int_{-\infty}^{\infty}xf(x)dx
\end{align*}$$
- **How**:

| Property                             | Formula                          |
| ------------------------------------ | ---------------------------------- |
| Linearity *(no independence needed)* | $E[aX+bY+c]=aE[X]+bE[Y]+c$         |
| Monotonicity                         | $X\leq Y\rightarrow E[X]\leq E[Y]$ |
| Product rule *(independence)*        | $E[XY]=E[X]E[Y]$                   |
| Law of total expectation             | $E[X]=E[E[X\|Y]]$                   |

&nbsp;

### Variance
- **What**: Average deviation between outcomes and mean.
$$\begin{align*}
&\text{Discrete:}   && Var[X]=E[(X-\mu^2)]=\sum_i (x_i-\mu)^2p_i \\
&\text{Continuous:} && Var[X]=E[(X-\mu^2)]=\int_{-\infty}^{\infty}(x-\mu)^2f(x)dx
\end{align*}$$
- **Why**:
    - *Why squared?*
        - Mathematically convenient (see **How**).
        - Smooth/Differentiable.
        - Punishes big surprises more.
            - BUT WHY CARE?
            - 1) Big misses are often disproportionately bad.
            - 2) Big surprises dominate what we care about, in various contexts.
    - *Why care?*
        - It tells us how **unreliable** the mean is.
- **How**:

| Property                         | Formula                          |
| -------------------------------- | ---------------------------------- |
| Scale                            | $Var[aX]=a^2 Var[X]$               |
| Shift                            | $Var[X+c]=Var[X]$                  |
| Sum                              | $Var[X+Y]=Var[X]+Var[Y]+2Cov[X,Y]$ |
| Relationship w/ Mean | $Var[X]=E[X^2]-E[X]^2$             |
| Law of total variance            | $Var[X]=E[Var[X\|Y]]+Var[E[X\|Y]]$   |
| Chebyshev Inequality             | $\forall\lambda>0: P(\|X-\mu\|\geq \lambda\sigma)\leq\frac{1}{\lambda^2}$<br>It's very unlikely for $X$ to be farther from $\mu$ than a multiple of $\sigma$. |

&nbsp;

### Moment
- **What**: Mean of a power of a random variable.
- **Why**:
    - *Why was moment created?*
        - Moments let us compress an entire probability distribution into a few meaningful numbers
        - They can summarize the shape quickly:
            - 1st moment = average value
            - 2nd moment = spread
            - 2nd central moment = variance
            - 3rd central moment = skewness (left/right)
            - 4th central moment = kurtosis (tail/heaviness)
        - They are super useful for approximations for complicated distributions.
        - They connect directly to prediction & risk.
    - *Why the name "moment"?*
        - In physics, "moment" measures how mass is distributed.
        - In probability, "moment" measures how probability mass is distributed.
- **How**:
    - $k$th Moment: $\mu_k=E[X^k]$.
    - $k$th Central moment: $\mu_k=E[(X-E[X])^k]$.

&nbsp;

### Moment Generating Function
- **What**: Avg exponential weighting of a random variable.
    - → A tail-sensitive fingerprint that lets us read off moments by differentiating at 0.
$$
M_X(t)=E[e^{tX}]
$$
    - $t$: A knob that changes what values of $X$ we emphasize.
    - $t>0$ → $M_X(t)$ grows fast for large $X$ → Highly sensitive to the **right tail**.
    - $t<0$ → $M_X(t)$ shrinks large $X$ → Highly sensitive to the **left tail**.
    - "If we weigh outcomes exponentially toward big/small values, what avg weight do we get?"
- **Why**: A compression of a random variable's distribution into one smooth function.
    - *Why exponential?*
        - Perfect sum of independent vars: $e^{t(X+Y)}=e^{tX}e^{tY}$
            - Who cares?
            - ← Adding random vars is usually very hard.
            - ← Multiplying functions is usually much easier.
            - "Hard convolution" → "Easy multiplication".
        - Derivatives create powers of $X$:
            - $\frac{d}{dt}e^{tX}=Xe^{tX}$
            - $\frac{d^2}{dt^2}e^{tX}=X^2e^{tX}$
            - ...
    - *Why "moment generating"?*
        - Try differentiate at $t=0$:
            - $M'_X(0)=E[X]$
            - $M''_X(0)=E[X^2]$
            - ...
    - *Why MGF sometimes doesn't exist?*
        - If a distribution has heavy tails, $e^{tX}$ can explode.
        - → $E[e^{tX}]\rightarrow\infty$
- **How**:

| Property             | Formula                                  |
| -------------------- | ---------------------------------------- |
| Normalization        | $M_X(0)=1$                               |
| Moment generation    | $M_X^{(k)}(0)=E[X^k]$                    |
| Linearity            | $Y=aX+b\rightarrow M_Y(t)=e^{bt}M_X(at)$ |
| Sum (*independence*) | $M_{\sum X_i}(t)=\prod_iM_{X_i}(t)$      |
| Convexity            | $M_X(t)$ is convex in $t$                |

&nbsp;

### Characteristic Function
- **What**: Avg of a complex rotation of a random variable.
    - → A frequency fingerprint of the distribution that always exists.
$$
\varphi_X(t)=E[e^{itX}]
$$
    - $|e^{itX}|=1$ → CF spins around the unit circle in the complex plane.
        - Outcome $X=x$ is an arrow of length 1 at angle $tx$.
        - Expectation averages all arrows.
    - If distribution is very spread out → Arrows **point in many directions & cancel out** → $\varphi_X(t)$ small.
    - If distribution is very concentrated → Arrows **line up & don't cancel** → $\varphi_X(t)$ large.
    - "For each frequency $t$, how much does the distribution 'line up'?" (i.e., Fourier transform)
- **Why**:
    - *Why does a unit circle spin uniquely determine a distribution?*
        - CF = Fourier Transform:
$$
\varphi(t)=\int_{-\infty}^{\infty}e^{itx}f(x)dx
$$
        - Fourier Transform: "How much of each wave $e^{itx}$ is inside this function?"
        - Fourier Transform is **one-to-one** → Knowing all freq info is enough to reconstruct the original signal.
            - If we know the coefficient for EVERY freq $t$, we know the function's expansion in that wave-basis.
        - → Two diff distributions can't have the exact same coordinates in a basis.
    - *Why CF always exists?*
        - It's just an arrow on a unit circle. It never explodes or vanishes.
- **How**:

| Property                | Formula                                                                                    |
| ----------------------- | ------------------------------------------------------------------------------------------ |
| Normalization           | $M_X(0)=1$                                                                                 |
| Moment generation       | $\varphi_X^{(k)}(0)=i^kE[X^k]$                                                             |
| Linearity               | $Y=aX+b\rightarrow \varphi_Y(t)=e^{itb}\varphi_X(at)$                                      |
| Sum (*independence*)    | $\varphi_{\sum X_i}(t)=\prod_i\varphi_{X_i}(t)$                                            |
| Lévy's continuity theorem | $\forall t,\varphi_{X_n}(t)\rightarrow\varphi_X(t)\ \Longleftrightarrow\ X_n\Rightarrow X$ |
| Relationship w/ MGF   | $M_X(t)=\varphi_X(-it)$                                                                    |


&nbsp;

## Named Distributions
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

### Discrete
#### Bernoulli
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
| Product        | $X,Y\sim\text{Bern}(p)(q)\rightarrow XY\sim\text{Bern}(pq)$ |

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

#### Binomial
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
    - $p=P(X=x)\in[0,1]$: Probability that the event occurs per trial.

Shape:
- Unimodal (i.e., one peak/mode) at $m=\lfloor(n+1)p\rfloor$.
- Skewness:
    - $p\leq\frac{1}{2}$: Right (mass → 0)
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
| Approximation by Normal Distribution | $X\sim\text{Bin}(n,p)\rightarrow \frac{X-E[X]}{Var[X]} \xRightarrow[n\to\infty]{} N(0,1)$ |

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
X\sim\text{Bin}(n,p)\rightarrow \frac{X-E[X]}{Var[X]} \xRightarrow[n\to\infty]{} N(0,1)
$$

KLD:
- $P=\text{Bin}(n,p),Q=\text{Bin}(n,q),p,q\in(0,1)$:
$$
D_{KL}(P||Q)=n\left[p\log\frac{p}{q}+(1-p)\log\frac{1-p}{1-q}\right]
$$
```

&nbsp;

#### Categorical
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

#### Multinomial
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

#### Poisson
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
&\text{CDF:} && P(X\leq x)=e^{-\lambda}\sum_{k=0}^{x}\frac{\lambda^k}{k!}
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

### Continuous
#### Normal/Gaussian
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

#### Lognormal
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
    - $\mu\in\mathbb{R}$: Mean.
    - $\sigma>0$: Standard deviation.

Shape:
- Unimodal at $\mu$.
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
| Scale    | $a>0\rightarrow aX\sim\text{LogN}(\mu+\log c,\sigma^2)$                                                                                                         |
| Power    | $k\in\mathbb{R}\rightarrow X^k\sim\text{LogN}(k\mu,k^2\sigma^2)$                                                                                                |
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

#### Exponential
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
- Strictly decreasing as $\lambda$ increases.

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
- $P=\text{Exp}(\lambda_0),Q=\text{LogN}(\lambda_1)$:
$$
D_{KL}(P||Q)=\log\frac{\lambda_0}{\lambda_1}+\frac{\lambda_1}{\lambda_0}-1
$$
```

&nbsp;

#### Gamma
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
    - $x\rightarrow1: f(x)\approx \text{poly}(x)e^{-\beta x}$
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
- $\varphi(\alpha)=\frac{\Gamma'(x)}{\Gamma(x)}$: Digamma function.
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

#### Beta
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