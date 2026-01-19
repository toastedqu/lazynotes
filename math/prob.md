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
# Probability Theory
Study notes from {cite:t}`handbook_math` and {cite:t}`prob_lifesaver`.

## Basics (Probability)
### Sample Space 
- **What**: Set of all possible outcomes of a random experiment ($\Omega$).
    - e.g., Coin flip: $\Omega=\{\text{head}, \text{tail}\}$.
    - e.g., Time till the universe ends: $\Omega=[0,\infty)$.

### Event
- **What**: A subset of the sample space ($A\subseteq\Omega$).
    - e.g., Coin flip is heads: $A=\{\text{head}\}$.
    - e.g., The universe ends within 60 seconds: $A=[0,60]$.

### σ-algebra
- **What**: A menu of allowed events in the sample space with 3 conditions ($\mathcal{F}$):
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
        - These 3 axioms are the minimal rules for "measuring uncertainty as a number" to behave consistently with ordinary logic about events AND to work for infinite processes.
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
- **How**:
    - **Discrete**: Takes countable values.
        - pmf: $p(x)=P(X=x)$
    - **Continuous**: Takes values in intervals.
        - pdf: $P(a\leq X\leq b)=\int_a^bf(x)dx$
        - No probability at any point: $\forall c\in\mathbb{R}: P(X=c)=0$

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
| Relationship with Mean | $Var[X]=E[X^2]-E[X]^2$             |
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
| Lévy continuity theorem | $\forall t,\varphi_{X_n}(t)\rightarrow\varphi_X(t)\ \Longleftrightarrow\ X_n\Rightarrow X$ |
| Relationship with MGF   | $M_X(t)=\varphi_X(-it)$                                                                    |


&nbsp;

## Named Distributions
### Discrete
#### Binomial
- **What**: Probability of #times an event $A$ happens out of a fixed number of independent, identical binary trials.
$$\begin{align*}
&\text{PMF:} && P(X_n=k)=\begin{pmatrix}
n \\ k
\end{pmatrix}p^k(1-p)^{n-k} \\
&\text{CDF:} && P(X_n\leq x)=\sum_{k=0}^{\lfloor x\rfloor}\begin{pmatrix}
n \\ k
\end{pmatrix}p^k(1-p)^{n-k}
\end{align*}$$
    - $n$: #trials.
    - $p$: $P(A)$ per trial.
    - $k$: $A$ takes place exactly $k$ times.
    - $P(X_n=k)$: Probability that $A$ takes place exactly $k$ times.
- **How**: $X_n\sim\text{Bin}(n,p)$:

| Property | Formula |
|:---------------------------------------- |:----------------------------------------------- |
| Mean                           | $E[X_n]=np$ |
| Variance                                 | $Var[X_n]=np(1-p)$    |
| Recursion                                | $P(X_n=k+1)=\frac{n-k}{k+1}\frac{p}{q}P(X_n=k)$   |
| Sum                                      | $X_n\sim\text{Bin}(n,p),X_m\sim\text{Bin}(m,p)$ <br>$\rightarrow X=X_n+X_m\sim\text{Bin}(n+m,p)$  |
| Approximation by Normal Distribution | $X_n\sim\text{Bin}(n,p)\rightarrow \frac{X_n-\mu}{\sigma} \xRightarrow[n\to\infty]{} N(0,1)$ |

&nbsp;

#### Poisson
- **What**: Probability of #times an event $A$ happens in a fixed time/space interval, given an average occurrence rate.
$$\begin{align*}
&\text{PMF:} && P(X=k)=\frac{\lambda^k}{k!}e^{-\lambda} \\
&\text{CDF:} && P(X\leq x)=e^{-\lambda}\sum_{k=0}^{\lfloor x\rfloor}\frac{\lambda^k}{k!}
\end{align*}$$
    - $\lambda$: Avg occurrence rate.
- **How**: $X\sim\text{Pois}(\lambda)$:

| Property                                   | Formula                                                                                                                |
| ------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| Mean                             | $E[X]=\lambda$                                                                                                         |
| Variance                                   | $Var[X]=\lambda$                                                                                                       |
| Recursion                                  | $P(X=k+1)=\frac{\lambda}{k+1}P(X=k)$                                                                                   |
| Sum                                        | $X_n\sim\text{Pois}(\lambda_n),X_m\sim\text{Pois}(\lambda_m)$<br>$\rightarrow X=X_n+X_m\sim\text{Pois}(\lambda_n+\lambda_m)$ |
| Approximation by Binomial Distribution | $X_n\sim\text{Bin}(n,\frac{\lambda}{n})\rightarrow X_n \xRightarrow[n\to\infty]{} \text{Pois}(\lambda)$                                        |

&nbsp;

### Continuous
#### Normal/Gaussian
- **What**: Probability of the sum/avg of many independent random outcomes.
    - Most outcomes land near the mean.
    - Very large/small outcomes are rare.
$$\begin{align*}
&\text{PDF:} && f(x)=\frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right) \\
&\text{CDF:} && F(x)=\Phi\left(\frac{x-\mu}{\sigma}\right)=\frac{1}{2}\left[1+\text{erf}\left(\frac{x-\mu}{\sqrt{2}\sigma}\right)\right] \\
&\text{Standard CDF:} && \Phi(z)=\frac{1}{\sqrt{2\pi}}\int_{-\infty}^z\exp\left(-\frac{t^2}{2}\right)dt
\end{align*}$$
- **How**: $X\sim N(\mu,\sigma^2)$:

| Property                                   | Formula                                                                                                                |
| ------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| Mean                             | $E[X]=\mu$                                                                                                         |
| Variance                                   | $Var[X]=\sigma^2$                                                                                                       |
| Linearity                     | $a,b\in\mathbb{R},a\neq0\rightarrow aX+b\sim N(a\mu+b,a^2\sigma^2)$                                                                       |
| Sum (*independence*)                                        | $\sum_{i=1}^{n}X_i\sim N\left(\sum_{i=1}^{n}\mu_i,\sum_{i=1}^{n}\sigma_i^2\right)$ |
| Standardization | $Z=\frac{X-\mu}{\sigma}\sim N(0,1)$                                        |
| Inflection Points | $f''_X(x)=0\Longleftrightarrow x=\mu\pm\sigma$ |
| Empirical Probabilities | $P(\|X-\mu\|\leq\sigma)\approx 0.68$<br>$P(\|X-\mu\|\leq2\sigma)\approx 0.95$<br>$P(\|X-\mu\|\leq3\sigma)\approx 0.997$ |
| Max Entropy (*among all continuous distributions*) | $h(X)=\frac{1}{2}\log(2\pi e \sigma^2)$ |

&nbsp;

#### Lognormal
- **What**: Probability of the product of many independent, **positive** random growth factors.
    - If the **log** of a variable is **Normal**, then the variable is **Lognormal**.
$$\begin{align*}
&\text{PDF:} && f(x)=\frac{1}{x\sqrt{2\pi}\sigma}\exp\left(-\frac{(\log x-\mu)^2}{2\sigma^2}\right), x>0 \\
&\text{CDF:} && F(x)=\Phi\left(\frac{\log x-\mu}{\sigma}\right), x>0
\end{align*}$$
- **Why**:
    - Many quantities are made by multiplying & can't go below 0.
    - BUT once they take a log, they behave like normal.
- **How**:

| Property | Formula                                                                                                                                                              |
| -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Mean     | $E[X]=\exp\left(\mu+\frac{\sigma^2}{2}\right)$                                                                                                                       |
| Variance | $Var[X]=\left(e^{\sigma^2}-1\right)\exp\left(2\mu+\sigma^2\right)$                                                                                                   |
| Scale    | $a>0\rightarrow aX\sim\text{LogNormal}(\mu+\log c,\sigma^2)$                                                                                                         |
| Power    | $k\in\mathbb{R}\rightarrow X^r\sim\text{LogNormal}(r\mu,r^2\sigma^2)$                                                                                                |
| Product  | $X_1\sim\text{LogNormal}(\mu_1,\sigma_1^2),X_2\sim\text{LogNormal}(\mu_2,\sigma_2^2)$<br>$\rightarrow X_1X_2\sim\text{LogNormal}(\mu_1+\mu_2,\sigma_1^2+\sigma_2^2)$ |
| Ratio    | If independent as above, $\frac{X_1}{X_2}\sim\text{LogNormal}(\mu_1-\mu_2,\sigma_1^2+\sigma_2^2)$                                                                    |

&nbsp;

#### Exponential
- **What**: Probability of the waiting time till the next event when events happen randomly at an avg rate.
$$\begin{align*}
&\text{PDF:} && f(x)=\lambda e^{-\lambda x}, x\geq 0 \\
&\text{CDF:} && F(x)=1-e^{-\lambda x}, x\geq 0
\end{align*}$$
- **How**:
    - Mean: $E[X]=\frac{1}{\lambda}$
    - Variance: $Var[X]=\frac{1}{\lambda^2}$
    - Memoryless: $P(X>s+x|X>s)=P(X>x)$<br>No matter how long you've waited, you are starting over.