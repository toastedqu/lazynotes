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

## Basics
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
- **What**: Average deviation between outcomes and expected value.
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
        - It tells us how **unreliable** the expected value is.
- **How**:

| Property                         | Formula                          |
| -------------------------------- | ---------------------------------- |
| Scale                            | $Var[aX]=a^2 Var[X]$               |
| Shift                            | $Var[X+c]=Var[X]$                  |
| Sum                              | $Var[X+Y]=Var[X]+Var[Y]+2Cov[X,Y]$ |
| Relationship with Expected Value | $Var[X]=E[X^2]-E[X]^2$             |
| Law of total variance            | $Var[X]=E[Var[X\|Y]]+Var[E[X\|Y]]$   |
| Chebyshev Inequality             | $\forall\lambda>0: P(\|X-\mu\|\geq \lambda\sigma)\leq\frac{1}{\lambda^2}$<br>It's very unlikely for $X$ to be farther from $\mu$ than a multiple of $\sigma$. |

&nbsp;

### Moment of Order $n$
- **What**: Expected value of a random variable raised to the $n$th power.
- **Why**: Moments let us compress an entire probability distribution into a few meaningful numbers
    - They can summarize the shape quickly:
        - 1st moment = average value
        - 2nd moment = spread
        - 3rd moment = skewness (left/right)
        - 4th moment = kurtosis (tail/heaviness)
    - They are super useful for approximations for complicated distributions.
    - They connect directly to prediction & risk.
- **How**: $E[X^n]$.

&nbsp;

## Distribution (Discrete)
### Binomial
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
- **How**:

| Property | Formula |
|:---------------------------------------- |:----------------------------------------------- |
| Expected Value                           | $E[X_n]=np$ |
| Variance                                 | $Var[X_n]=np(1-p)$    |
| Recursion                                | $P(X_n=k+1)=\frac{n-k}{k+1}\frac{p}{q}P(X_n=k)$   |
| Sum                                      | $X_n\sim\text{Bin}(n,p),X_m\sim\text{Bin}(m,p)$ <br>$\rightarrow X=X_n+X_m\sim\text{Bin}(n+m,p)$  |
| Approximation by Normal Distribution | $X_n\sim\text{Bin}(n,p)\rightarrow \frac{X_n-\mu}{\sigma} \xRightarrow[n\to\infty]{} N(0,1)$ |

&nbsp;

### Poisson
- **What**: Probability of #times an event $A$ happens in a fixed time/space interval, given an average occurrence rate.
$$\begin{align*}
&\text{PMF:} && P(X=k)=\frac{\lambda^k}{k!}e^{-\lambda} \\
&\text{CDF:} && P(X\leq x)=e^{-\lambda}\sum_{k=0}^{\lfloor x\rfloor}\frac{\lambda^k}{k!}
\end{align*}$$
    - $\lambda$: Avg occurrence rate.
- **How**:

| Property                                   | Formula                                                                                                                |
| ------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| Expected Value                             | $E[X]=\lambda$                                                                                                         |
| Variance                                   | $Var[X]=\lambda$                                                                                                       |
| Recursion                                  | $P(X=k+1)=\frac{\lambda}{k+1}P(X=k)$                                                                                   |
| Sum                                        | $X_n\sim\text{Pois}(\lambda_n),X_m\sim\text{Pois}(\lambda_m)$<br>$\rightarrow X=X_n+X_m\sim\text{Pois}(\lambda_n+\lambda_m)$ |
| Approximation by Binomial Distribution | $X_n\sim\text{Bin}(n,\frac{\lambda}{n})\rightarrow X_n \xRightarrow[n\to\infty]{} \text{Pois}(\lambda)$                                        |

## Distribution (Continuous)
### Normal
