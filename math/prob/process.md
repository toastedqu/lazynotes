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
# Stochastic Processes
Random quantities indexed by time.
Foundations → Markov chains → random walks & martingales → continuous-time jumps → continuous paths & SDEs.

Notations:
- $t$: Time; $t\in\{0,1,2,\ldots\}$ (discrete) or $t\in[0,\infty)$ (continuous).
- $X_t$: Random state at time $t$ (scalar or vector).
- $\mathcal{S}$: State space.
- $i,j$: States in a discrete $\mathcal{S}$.
- $n$: #steps.
- $\mathbb{P}$: Probability.
- $\mathbb{E}$: Expectation.
- $P$: Transition matrix, $P_{ij}=\mathbb{P}(X_{t+1}=j\mid X_t=i)$.
- $\pi_t$: Distribution of $X_t$ as a row vector.
- $\pi$: Stationary distribution as a row vector.
- $\mathcal{F}_t$: Filtration (info available at time $t$).
- $\tau$: Stopping time / hitting time.
- $W_t$: Standard Brownian motion.
- $\lambda$: Event rate.

Overrides: $i,j$ = states, ❌sample/feature idx. $n$ = #steps, ❌#features. $P$ = transition matrix; probability uses $\mathbb{P}$. $X_t$ = random state, ❌matrix.

&nbsp;

## Foundations

### Stochastic Process
- **What**: Collection of random variables $\{X_t\}_{t\in\mathcal{T}}$ on one probability space, indexed by time.
- **Why**: Model quantities that evolve randomly.
    - Dependence across time is the object of study, ❌iid.
- **How**: 2 views of the same object:
    - Fix $t$ → random variable $X_t$.
    - Fix outcome $\omega$ → **sample path** $t\mapsto X_t(\omega)$ (one random function).

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\mathcal{T}$: Index set (time).
    - $\Omega$: Sample space.

Definition:

$$
X:\mathcal{T}\times\Omega\to\mathcal{S},\qquad (t,\omega)\mapsto X_t(\omega)
$$

- Each $X_t$ is a random variable.

**Finite-dimensional distributions** (fdds):

$$
\mathbb{P}(X_{t_1}\in A_1,\ldots,X_{t_k}\in A_k),\qquad t_1<\cdots<t_k
$$

- **Kolmogorov extension**: a consistent family of fdds (on $\mathbb{R}^d$-valued states) defines a process.
- Fdds fix the law on the product σ-algebra.
- Fdds do NOT fix sample-path properties (continuity, measurability in $t$).
```

```{dropdown} Table: Process Taxonomy
| | Discrete state | Continuous state |
|:--|:--|:--|
| **Discrete time** | [Markov chain](#markov-chain), [random walk](#random-walk) | AR(1), Gaussian random walk |
| **Continuous time** | [Poisson process](#poisson-process), [CTMC](#ctmc) | [Brownian motion](#brownian-motion), [SDE](#sde), [GP](bayes.md#gp) |
```

```{attention} Q&A
:class: dropdown
*Do fdds determine the sample paths?*

- No.
- e.g., $U\sim\text{Unif}[0,1]$, $X_t=0$, $Y_t=\mathbb{1}\{t=U\}$.
- $\mathbb{P}(Y_t\neq0)=\mathbb{P}(U=t)=0$ for every $t$ → same fdds.
- $X$ has continuous paths. $Y$ has a jump on every path.
- → Path properties need an extra construction (a continuous modification).

*How does a stochastic process differ from a random vector?*

- Finite $\mathcal{T}$ → it is a random vector.
- Infinite $\mathcal{T}$ → a random function → new questions: long-run limits, path regularity, hitting times.
```

&nbsp;

### Filtration
- **What**: Increasing family of σ-algebras $\mathcal{F}_s\subseteq\mathcal{F}_t$ for $s\le t$.
- **Why**: Formalize "no peeking into the future".
    - Needed to define martingales, stopping times & non-anticipating strategies.
- **How**:
    - **Natural filtration**: $\mathcal{F}_t=\sigma(X_0,\ldots,X_t)$ = everything learned from the path so far.
    - **Adapted**: $X_t$ is $\mathcal{F}_t$-measurable → value of $X_t$ known at time $t$.

&nbsp;

#### Stopping Time
- **What**: Random time $\tau$ with $\{\tau\le t\}\in\mathcal{F}_t$ for all $t$.
- **Why**: Model decisions "stop now" that use only past & present info.
- **How**: Test: at each $t$, can you tell whether $\tau$ has happened w/o looking ahead?
    - ✅ First hitting time of a set: $\tau_A=\min\{t\ge0:X_t\in A\}$.
    - ✅ Constant time; min/max of 2 stopping times.
    - ❌ Last exit time from a set; time of the path's maximum on $[0,T]$ ← both need the future.

&nbsp;

### Stationarity
- **What**: Distribution invariant to time shifts.
    - **Strict**: all fdds are shift-invariant.
    - **Weak (WSS)**: constant mean & autocovariance depending only on the lag.
- **Why**: Time-invariant statistics.
    - Stationarity + [ergodicity](#ergodicity) → estimate them from 1 long path.
    - Prerequisite for [entropy rate](../info.md#entropy-rate), spectral analysis & classical time-series models.
- **How**: Check whether shifting the time window changes the statistics you care about.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $h$: Time lag.
    - $\mu$: Constant mean.
    - $\gamma(h)$: Autocovariance at lag $h$.

Strict:

$$
(X_{t_1+h},\ldots,X_{t_k+h})\overset{d}{=}(X_{t_1},\ldots,X_{t_k})\qquad\forall k,t_1,\ldots,t_k,h
$$

Weak (WSS), with $\mathbb{E}[X_t^2]<\infty$:

$$
\mathbb{E}[X_t]=\mu,\qquad \text{Cov}(X_t,X_{t+h})=\gamma(h)
$$

- Strict + finite 2nd moment → WSS.
- WSS ↛ strict.
- Exception: WSS Gaussian process → strict ← Gaussian fdds are fixed by mean & covariance.
```

```{note} Example
:class: dropdown
**AR(1)**: $X_t=\phi X_{t-1}+\varepsilon_t$, $\varepsilon_t\overset{iid}{\sim}\mathcal{N}(0,\sigma^2)$.

- $|\phi|<1$ & $X_0\sim\mathcal{N}\left(0,\frac{\sigma^2}{1-\phi^2}\right)$ → stationary:

    $$
    \text{Var}(X_t)=\frac{\sigma^2}{1-\phi^2},\qquad \gamma(h)=\phi^{|h|}\frac{\sigma^2}{1-\phi^2}
    $$

    - $\sigma^2$: Noise variance.
    - Var: solve $v=\phi^2v+\sigma^2$.
- $|\phi|<1$, other $X_0$ → $\text{Var}(X_t)=\phi^{2t}\text{Var}(X_0)+\frac{\sigma^2(1-\phi^{2t})}{1-\phi^2}$ → only asymptotically stationary.
- $\phi=1$ → Gaussian random walk → $\text{Var}(X_t)=\text{Var}(X_0)+t\sigma^2$ grows → ❌stationary.
```

```{attention} Q&A
:class: dropdown
*Does stationary mean iid?*

- No. Values can be strongly dependent (AR(1) w/ $\phi=0.99$).
- Only the joint law is invariant to shifts.

*Does stationary mean the path is constant?*

- No. The distribution is constant, the path is not.

*When is a Markov chain stationary?*

- When it starts from its [stationary distribution](#stationary-distribution) $\pi$ → $\pi_t=\pi$ for all $t$.

*How do you make a nonstationary series stationary?*

- Remove trend/seasonality, or difference it: random walk $X_t$ → increments $X_t-X_{t-1}$ are iid.
```

&nbsp;

### Ergodicity
- **What**: Time avg along 1 path = ensemble avg.
- **Why**: Often only 1 path exists: 1 time series, 1 MCMC chain, 1 RL trajectory.
    - → Need time avgs to estimate expectations.
- **How**: Stationary + no invariant sub-region that traps the path → 1 long path visits everything in the right proportions.

```{note} Math
:class: dropdown
**Birkhoff ergodic theorem**: $\{X_t\}$ stationary & ergodic, $\mathbb{E}|f(X_0)|<\infty$:

$$
\frac{1}{n}\sum_{t=0}^{n-1}f(X_t)\xrightarrow{\text{a.s.}}\mathbb{E}[f(X_0)]
$$

- $f$: Any integrable function of the state.
- **Ergodic**: every shift-invariant event has probability 0 or 1.
```

```{note} Example
:class: dropdown
Stationary but NOT ergodic:

- $Z\sim\mathcal{N}(0,1)$, $X_t=Z$ for all $t$.
- Time avg $=Z$. Ensemble avg $=0$.
- 1 path never sees other values of $Z$.
```

```{attention} Q&A
:class: dropdown
*Does stationarity imply ergodicity?*

- No. See the Example. A stationary process is a mixture of ergodic components.

*Why does MCMC need ergodicity?*

- MCMC reports time avgs of 1 chain as posterior expectations → valid only if the chain is ergodic. See [MCMC](sampling.md#mcmc).

*How does ergodicity relate to RL?*

- Average-reward RL assumes every policy induces an ergodic chain → long-run avg reward does not depend on the start state.

*What breaks w/o ergodicity?*

- Absorbing outcomes (bankruptcy, death) → time avg of 1 agent ≠ avg over many agents.
- → Maximizing expected value can still ruin almost every single path.
```

&nbsp;

## Markov Chains

### Markov Chain
- **What**: Discrete-time process whose next state depends only on the curr state.
- **Why**: Tractable memory.
    - Full history grows w/o bound.
    - Markov → 1 matrix $P$ describes all dynamics.
- **How**:
    1. Pick states & transition matrix $P$ (row $i$ = next-state distribution from $i$).
    2. Distribution evolves by right-multiplication: $\pi_{t+1}=\pi_tP$.
    3. $n$-step transitions = $P^n$.

```{note} Math
:class: dropdown
Markov property (time-homogeneous):

$$
\mathbb{P}(X_{t+1}=j\mid X_t=i,X_{t-1},\ldots,X_0)=\mathbb{P}(X_{t+1}=j\mid X_t=i)=P_{ij}
$$

- Row-stochastic: $P_{ij}\ge0$, $\sum_jP_{ij}=1$.

Path probability:

$$
\mathbb{P}(X_0=i_0,\ldots,X_n=i_n)=\pi_0(i_0)\prod_{t=1}^{n}P_{i_{t-1}i_t}
$$

Distribution after $n$ steps:

$$
\pi_n=\pi_0P^n
$$

**Chapman–Kolmogorov**:

$$
P^{(a+b)}_{ij}=\sum_{l\in\mathcal{S}}P^{(a)}_{il}P^{(b)}_{lj}\quad\Leftrightarrow\quad P^{a+b}=P^aP^b
$$

- $P^{(a)}_{ij}$: $\mathbb{P}(X_{t+a}=j\mid X_t=i)$.
- $l$: Intermediate state at time $t+a$.
```

````{important} Code
:class: dropdown
```python
import numpy as np

class MarkovChain:
    def __init__(self, P, seed=0):
        self.P = np.asarray(P, dtype=float)
        assert np.allclose(self.P.sum(axis=1), 1), "rows must sum to 1"
        self.rng = np.random.default_rng(seed)

    def sample(self, x0, n):
        ## next state depends only on the curr state -> 1 row lookup per step
        path = [x0]
        for _ in range(n):
            path.append(self.rng.choice(len(self.P), p=self.P[path[-1]]))
        return np.array(path)

    def distribution(self, pi0, n):
        ## row vector times P^n (Chapman-Kolmogorov)
        return np.asarray(pi0, dtype=float) @ np.linalg.matrix_power(self.P, n)

## Example: 0 = sunny, 1 = rainy
mc = MarkovChain([[0.9, 0.1], [0.5, 0.5]])
print(mc.sample(0, 10).shape)          ## (11,)
print(mc.distribution([1, 0], 50))     ## ~[0.8333 0.1667]
```
````

```{attention} Q&A
:class: dropdown
*Does Markov mean the future is independent of the past?*

- Only conditionally: future ⫫ past | present.
- Unconditionally, $X_{t+1}$ & $X_{t-1}$ are usually dependent.

*How do you make a non-Markov process Markov?*

- Augment the state.
- Order-$k$ dependence → state $(X_{t-k+1},\ldots,X_t)$.
- e.g., $n$-gram LM = Markov chain over the last $n-1$ tokens.
- Same trick turns a non-Markov env into an [MDP](../../rl/intro.md#mdp).

*Time-homogeneous vs inhomogeneous?*

- Homogeneous: same $P$ at every step.
- Inhomogeneous: $P_t$ changes w/ $t$ → $\pi_n=\pi_0P_1\cdots P_n$.
- e.g., the [diffusion forward process](../../dl/gen/diffusion.md#forward-process) uses step-dependent noise $\beta_t$.

*Why row vectors?*

- Convention: $\pi_{t+1}=\pi_tP$ (row) ⇔ $\pi_{t+1}^T=P^T\pi_t^T$ (column).
- Mixing conventions transposes $P$ → common bug.
```

&nbsp;

### State Classification
- **What**: Partition of states by reachability & return behavior.
- **Why**: Long-run behavior depends only on these properties.
    - Does a stationary distribution exist? Is it unique? Does $\pi_n$ converge to it?
- **How**: Ask in order:
    1. **Communicate** ($i\leftrightarrow j$): each is reachable from the other. Equivalence classes = communicating classes. **Irreducible**: 1 class.
    2. **Recurrent / Transient**: return to $i$ w.p. $1$ / $<1$.
    3. **Positive / Null recurrent**: finite / infinite mean return time.
    4. **Period** $d(i)$: gcd of possible return times. **Aperiodic**: $d(i)=1$.
    5. **Absorbing**: $P_{ii}=1$.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $T_i=\min\{n\ge1:X_n=i\}$: First return time to $i$.
    - $\mathbb{P}_i$: Probability given $X_0=i$.
    - $\mathbb{E}_i$: Expectation given $X_0=i$.

Definitions:

$$
\begin{align*}
&\text{Recurrent:} && \mathbb{P}_i(T_i<\infty)=1\\
&\text{Positive recurrent:} && \mathbb{E}_i[T_i]<\infty\\
&\text{Period:} && d(i)=\gcd\{n\ge1:P^n_{ii}>0\}
\end{align*}
$$

Properties:
- Recurrent $\Leftrightarrow\sum_{n\ge1}P^n_{ii}=\infty$ (expected #returns is infinite).
- Recurrence, transience & period are **class properties** → shared by all states in a communicating class.
- Finite chain → at least 1 recurrent class & no null recurrent state.
- Finite irreducible → positive recurrent.
- Irreducible + any $P_{ii}>0$ → aperiodic.
```

```{attention} Q&A
:class: dropdown
*Why does periodicity matter?*

- $P=\begin{bmatrix}0&1\\1&0\end{bmatrix}$ has $\pi=(\frac12,\frac12)$.
- But $\pi_n$ alternates between $(1,0)$ & $(0,1)$ from $X_0=0$ → never converges.
- Fix: **lazy chain** $\frac12(I+P)$ → aperiodic, same $\pi$.

*Example of null recurrence?*

- Simple symmetric random walk on $\mathbb{Z}$: returns w.p. 1, but $\mathbb{E}[T_0]=\infty$.
- Needs an infinite state space.

*What happens to transient states in the long run?*

- Visited finitely often w.p. 1 → their long-run probability is 0.
```

&nbsp;

### Stationary Distribution
- **What**: Distribution unchanged by 1 step: $\pi P=\pi$.
- **Why**: Long-run behavior.
    - Long-run fraction of time in each state.
    - Target of MCMC, PageRank scores, steady state of queues.
- **How**: Solve the linear system $\pi P=\pi$, $\sum_i\pi_i=1$.
    - Equivalently: left eigenvector of $P$ w/ eigenvalue 1, normalized.

```{note} Math
:class: dropdown
Definition:

$$
\pi=\pi P,\qquad \pi_i\ge0,\qquad \sum_i\pi_i=1
$$

Existence & uniqueness:
- Irreducible chain has a stationary distribution $\Leftrightarrow$ positive recurrent.
- Then $\pi$ is unique & $\pi_i=\dfrac{1}{\mathbb{E}_i[T_i]}$ (mean return time).
- Finite irreducible → unique $\pi$ w/ all $\pi_i>0$.

**Ergodic theorem** (irreducible, positive recurrent, any $X_0$):

$$
\frac1n\sum_{t=0}^{n-1}f(X_t)\xrightarrow{\text{a.s.}}\sum_i\pi_if(i)
$$

- $f$: Any function w/ $\sum_i\pi_i|f(i)|<\infty$.
- Aperiodicity is NOT needed for time avgs.

**Convergence theorem** (irreducible, aperiodic, positive recurrent):

$$
\lim_{n\to\infty}P^n_{ij}=\pi_j\qquad\forall i,j
$$

- → $\pi_n\to\pi$ from any start.
```

```{tip} Derivation
:class: dropdown
*Why does eigenvalue 1 always exist?*

1. Rows sum to 1 → $P\mathbf{1}=\mathbf{1}$ → 1 is an eigenvalue of $P$.
2. $P$ & $P^T$ share eigenvalues → a left eigenvector w/ eigenvalue 1 exists.
3. Perron–Frobenius (nonnegative matrix) → a nonnegative choice exists → normalize it to a distribution.
4. Every eigenvalue $\lambda$ of $P$ satisfies $|\lambda|\le1$ ← $P$ maps the $\ell_\infty$ ball into itself.
```

```{note} Example
:class: dropdown
**2-state chain**: $P=\begin{bmatrix}1-a&a\\b&1-b\end{bmatrix}$, $a,b\in(0,1]$.

$$
\pi=\left(\frac{b}{a+b},\frac{a}{a+b}\right)
$$

- Weather chain ($a=0.1$, $b=0.5$) → $\pi=(0.833,0.167)$.

**PageRank** (random surfer):
1. Surfer follows a random out-link w.p. $\alpha$, jumps to a uniform random page w.p. $1-\alpha$.
2. Transition matrix $G=\alpha P+(1-\alpha)\frac1N\mathbf{1}\mathbf{1}^T$, $N$ = #pages.
3. Teleport → every entry of $G$ is positive → irreducible & aperiodic → unique $\pi$.
4. PageRank score of page $i$ = $\pi_i$. Power iteration error shrinks by a factor $\le\alpha$ per step (in $L_1$). Common $\alpha=0.85$.
- Dangling page (no out-links) → replace its row by uniform.
```

````{important} Code
:class: dropdown
```python
import numpy as np

class StationaryDistribution:
    def __call__(self, P):
        P = np.asarray(P, dtype=float)
        S = len(P)
        ## (P^T - I) pi^T = 0 has rank S-1 (irreducible) -> replace 1 redundant row by sum(pi) = 1
        A = P.T - np.eye(S)
        A[-1] = 1.0
        b = np.zeros(S)
        b[-1] = 1.0
        return np.linalg.solve(A, b)

## Example: weather chain from Markov Chain above
print(StationaryDistribution()([[0.9, 0.1], [0.5, 0.5]]))   ## [0.8333 0.1667]
```
````

```{attention} Q&A
:class: dropdown
*Stationary distribution = limiting distribution?*

- Not always. Periodic chain → $\pi$ exists but $\pi_n$ oscillates.
- Equal when the chain is also aperiodic.

*When are there multiple stationary distributions?*

- ≥2 closed recurrent classes → each has its own $\pi^{(c)}$ → every convex combination is stationary.

*How does this relate to RL?*

- Fixed policy → MDP becomes a Markov chain → its stationary distribution $d^\pi$ weights states in on-policy objectives.
```

&nbsp;

### Detailed Balance
- **What**: $\pi_iP_{ij}=\pi_jP_{ji}$ for all $i,j$: equal probability flow both ways across every edge.
- **Why**: Local check for stationarity.
    - Avoid solving the global system $\pi P=\pi$.
    - Design tool: MCMC kernels are built to satisfy it.
- **How**: Sum both sides over $i$ → $\pi P=\pi$.

```{tip} Derivation
:class: dropdown
*Why does detailed balance imply stationarity?*

$$
(\pi P)_j=\sum_i\pi_iP_{ij}=\sum_i\pi_jP_{ji}=\pi_j\sum_iP_{ji}=\pi_j
$$
```

```{attention} Q&A
:class: dropdown
*Is detailed balance necessary for stationarity?*

- No. Sufficient only.
- e.g., 3-state cycle w/ $P_{i,i+1}=0.9$, $P_{i,i-1}=0.1$ (mod 3): uniform $\pi$ is stationary (columns sum to 1), but flow $0\to1$ ≠ flow $1\to0$.

*What does "reversible" mean?*

- Chain started at $\pi$ has the same law forward & backward in time.
- Reversible $\Leftrightarrow$ detailed balance.
- **Kolmogorov criterion**: reversible $\Leftrightarrow$ every cycle has equal products of transition probs in both directions.

*Why do reversible chains have nice spectra?*

- $D^{1/2}PD^{-1/2}$ is symmetric, $D=\text{diag}(\pi)$ → all eigenvalues real → clean [mixing](#mixing-time) analysis.

*Which common chains are reversible?*

- Random walk on an undirected graph, birth–death chains, Metropolis–Hastings, Gibbs (random-scan).
```

&nbsp;

### Mixing Time
- **What**: #steps until the distribution is within $\varepsilon$ of $\pi$ in total variation, from the worst start.
- **Why**: Stationarity says nothing about speed.
    - Sets MCMC burn-in, power-iteration cost, random-walk sampling cost.
- **How**: Distance to $\pi$ shrinks geometrically.
    - Rate = second-largest eigenvalue modulus $\lambda_\star$.
    - Spectral gap $1-\lambda_\star$ large → fast mixing.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $d(n)$: Worst-case distance to $\pi$ after $n$ steps.
    - $\lambda$: Eigenvalue of $P$ (overrides rate $\lambda$ in this section).
    - $\lambda_\star$: Largest $|\lambda|$ over eigenvalues $\lambda\ne1$.
    - $t_\text{rel}=1/(1-\lambda_\star)$: Relaxation time.
    - $\pi_\text{min}=\min_i\pi_i$.

Definition:

$$
d(n)=\max_i\|P^n(i,\cdot)-\pi\|_\text{TV},\qquad t_\text{mix}(\varepsilon)=\min\{n:d(n)\le\varepsilon\}
$$

- Convention: $t_\text{mix}=t_\text{mix}(1/4)$.

Spectral bounds (finite, reversible, irreducible):

$$
(t_\text{rel}-1)\log\frac{1}{2\varepsilon}\le t_\text{mix}(\varepsilon)\le t_\text{rel}\log\frac{1}{\varepsilon\,\pi_\text{min}}
$$

**Cheeger inequality** (reversible):

$$
\frac{\Phi_\star^2}{2}\le1-\lambda_2\le2\Phi_\star,\qquad \Phi_\star=\min_{A:\pi(A)\le\frac12}\frac{\sum_{i\in A,j\notin A}\pi_iP_{ij}}{\pi(A)}
$$

- $\lambda_2$: Second-largest eigenvalue (signed).
- $\Phi_\star$: Bottleneck ratio = flow out of the worst set relative to its mass.
```

```{note} Example
:class: dropdown
2-state chain ($P$ from [Stationary Distribution](#stationary-distribution)):

- Eigenvalues: $1$ & $1-a-b$.
- $\pi_n-\pi=(1-a-b)^n(\pi_0-\pi)$ → geometric decay at rate $|1-a-b|$.
- Weather chain: $|1-0.1-0.5|=0.4$ → error ⬇️ by 60% per step.
- $a=b=1$ → eigenvalue $-1$ → periodic → no convergence.
```

```{attention} Q&A
:class: dropdown
*Why use a lazy chain?*

- $\frac12(I+P)$ maps eigenvalues $\lambda\to\frac{1+\lambda}{2}\ge0$ → no oscillation.
- Cost: at most ~2× slower.

*Why do multimodal targets mix slowly?*

- Low-probability region between modes = bottleneck → small $\Phi_\star$ → small gap (Cheeger) → huge $t_\text{mix}$.

*What is the cutoff phenomenon?*

- Some chains stay far from $\pi$, then drop to near $\pi$ in a short window.
- e.g., riffle shuffle: ~7 shuffles for 52 cards.
```

&nbsp;

### Hitting Time
- **What**: First time the chain enters a target set $A$: $\tau_A=\min\{t\ge0:X_t\in A\}$.
- **Why**: "How long until…?" & "Which happens first?" questions.
    - Gambler's ruin, #flips until HH, time to failure, absorption in an absorbing chain.
- **How**: **First-step analysis**: condition on the first move → linear equations.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $h_i=\mathbb{P}_i(\tau_A<\infty)$: Hitting probability.
    - $k_i=\mathbb{E}_i[\tau_A]$: Expected hitting time.

First-step equations:

$$
\begin{align*}
h_i&=\begin{cases}1 & i\in A\\ \sum_jP_{ij}h_j & i\notin A\end{cases}\\
k_i&=\begin{cases}0 & i\in A\\ 1+\sum_jP_{ij}k_j & i\notin A\end{cases}
\end{align*}
$$

- Take the minimal nonnegative solution (matters for infinite $\mathcal{S}$).

Absorbing chain (transient states first):

$$
P=\begin{bmatrix}Q&R\\0&I\end{bmatrix},\qquad N=(I-Q)^{-1}
$$

- $Q$: Transient → transient block.
- $R$: Transient → absorbing block.
- $N_{ij}$: Expected #visits to transient $j$ from transient $i$ (**fundamental matrix**).
- Expected steps to absorption: $N\mathbf{1}$.
- Absorption probabilities: $NR$.
```

```{tip} Derivation
:class: dropdown
*Why $N=(I-Q)^{-1}$?*

1. $Q^n_{ij}$ = prob of being at transient $j$ after $n$ steps from $i$.
2. Expected visits $=\sum_{n\ge0}Q^n$.
3. Transient → $Q^n\to0$ → geometric series converges to $(I-Q)^{-1}$.
```

```{note} Example
:class: dropdown
*Expected #fair-coin flips until HH?*

- States: $0$ = no progress, $1$ = last flip H.
- $k_0=1+\frac12k_1+\frac12k_0$
- $k_1=1+\frac12\cdot0+\frac12k_0$
- → $k_1=4$, $k_0=6$.
- HT needs only 4 flips ← after an H, a miss (H) keeps the progress.
```

````{important} Code
:class: dropdown
```python
import numpy as np

class AbsorbingChain:
    def __init__(self, Q, R):
        Q, R = np.asarray(Q, dtype=float), np.asarray(R, dtype=float)
        ## N[i, j] = expected #visits to transient j from transient i = sum_n Q^n
        self.N = np.linalg.inv(np.eye(len(Q)) - Q)
        self.steps = self.N.sum(axis=1)    ## expected time to absorption
        self.absorb = self.N @ R           ## P(absorbed in each absorbing state)

## Example: flip until HH; transient = {no progress, last flip H}, absorbing = {HH}
chain = AbsorbingChain(Q=[[0.5, 0.5], [0.5, 0.0]], R=[[0.0], [0.5]])
print(chain.steps)    ## [6. 4.]
print(chain.absorb.ravel())   ## [1. 1.]
```
````

```{attention} Q&A
:class: dropdown
*Why does HH take longer than HT?*

- Both have probability $\frac14$ per window.
- HH occurrences overlap (HHH has 2) → they cluster → longer gaps between clusters.

*Can $h_i=1$ but $k_i=\infty$?*

- Yes. Symmetric random walk on $\mathbb{Z}$ hits 1 w.p. 1, but the expected time is infinite.
```

&nbsp;

### HMM
- **Name**: Hidden Markov Model
- **What**: Markov chain over hidden states, each state emits an observation.
- **Why**: Observed sequence = noisy view of simple latent dynamics.
    - e.g., POS tags → words, phonemes → audio, market regimes → returns.
- **How**: 3 classic problems:
    1. **Likelihood** $p(y_{1:T})$ → **forward algorithm** (sum over paths by DP).
    2. **Decoding** $\arg\max_{z_{1:T}}p(z_{1:T}\mid y_{1:T})$ → **Viterbi** (max instead of sum).
    3. **Learning** → **Baum–Welch** = EM w/ forward-backward posteriors.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $y_t$: Observation at time $t$ (discrete symbol).
    - $T$: Sequence length.
- Params:
    - $\pi_0$: Initial hidden-state distribution.
    - $P$: Hidden-state transition matrix.
    - $b_i(y)=p(y_t=y\mid z_t=i)$: Emission prob.
- Misc:
    - $z_t$: Hidden state at time $t$.

Model:

$$
p(z_{1:T},y_{1:T})=\pi_0(z_1)b_{z_1}(y_1)\prod_{t=2}^{T}P_{z_{t-1}z_t}b_{z_t}(y_t)
$$

Forward ($\alpha_t(i)=p(y_{1:t},z_t=i)$):

$$
\alpha_1(i)=\pi_0(i)b_i(y_1),\qquad \alpha_t(j)=b_j(y_t)\sum_i\alpha_{t-1}(i)P_{ij},\qquad p(y_{1:T})=\sum_i\alpha_T(i)
$$

Backward ($\beta_t(i)=p(y_{t+1:T}\mid z_t=i)$):

$$
\beta_T(i)=1,\qquad \beta_t(i)=\sum_jP_{ij}b_j(y_{t+1})\beta_{t+1}(j)
$$

Posteriors:

$$
\gamma_t(i)=\frac{\alpha_t(i)\beta_t(i)}{p(y_{1:T})},\qquad \xi_t(i,j)=\frac{\alpha_t(i)P_{ij}b_j(y_{t+1})\beta_{t+1}(j)}{p(y_{1:T})}
$$

- $\gamma_t(i)$: $p(z_t=i\mid y_{1:T})$.
- $\xi_t(i,j)$: $p(z_t=i,z_{t+1}=j\mid y_{1:T})$.

Viterbi:

$$
\delta_1(i)=\pi_0(i)b_i(y_1),\qquad \delta_t(j)=b_j(y_t)\max_i\delta_{t-1}(i)P_{ij}
$$

- $\delta_t(j)$: Prob of the best path ending in $j$ at $t$.
- Store $\arg\max_i$ as backpointers → trace back from $\arg\max_j\delta_T(j)$.

Baum–Welch M-step:

$$
\pi_0(i)\leftarrow\gamma_1(i),\qquad P_{ij}\leftarrow\frac{\sum_{t=1}^{T-1}\xi_t(i,j)}{\sum_{t=1}^{T-1}\gamma_t(i)},\qquad b_i(v)\leftarrow\frac{\sum_{t}\gamma_t(i)\mathbb{1}\{y_t=v\}}{\sum_t\gamma_t(i)}
$$

- $v$: Observation symbol.
- Cost: $O(T|\mathcal{S}|^2)$ for forward, backward & Viterbi, vs $O(|\mathcal{S}|^T)$ brute force.
```

````{important} Code
:class: dropdown
```python
import numpy as np

class HMM:
    def __init__(self, pi0, P, B):
        ## pi0: (S,) initial; P: (S, S) transitions; B: (S, V) emission probs
        self.pi0, self.P, self.B = (np.asarray(a, dtype=float) for a in (pi0, P, B))

    def log_likelihood(self, y):
        ## forward algorithm; renormalize every step -> no underflow on long sequences
        alpha = self.pi0 * self.B[:, y[0]]
        log_p = 0.0
        for t in range(1, len(y)):
            c = alpha.sum()
            if c == 0:                         ## prefix impossible under the model
                return -np.inf
            log_p += np.log(c)
            alpha = (alpha / c @ self.P) * self.B[:, y[t]]
        return log_p + np.log(alpha.sum())

    def viterbi(self, y):
        ## same recursion as forward, but max instead of sum (in log space)
        with np.errstate(divide="ignore"):
            lP, lB = np.log(self.P), np.log(self.B)
            delta = np.log(self.pi0) + lB[:, y[0]]
        back = []
        for t in range(1, len(y)):
            scores = delta[:, None] + lP             ## (S_prev, S_next)
            back.append(scores.argmax(axis=0))
            delta = scores.max(axis=0) + lB[:, y[t]]
        path = [int(delta.argmax())]
        for bp in reversed(back):
            path.append(int(bp[path[-1]]))
        return path[::-1]

## Example: 2 hidden states, 2 symbols
hmm = HMM(pi0=[0.5, 0.5], P=[[0.9, 0.1], [0.1, 0.9]], B=[[0.8, 0.2], [0.2, 0.8]])
y = [0, 0, 1, 1, 1]
print(hmm.log_likelihood(y))   ## a negative scalar
print(hmm.viterbi(y))          ## [0, 0, 1, 1, 1]
```
````

```{attention} Q&A
:class: dropdown
*Viterbi path vs per-step most likely states?*

- Viterbi: best whole sequence.
- $\arg\max_i\gamma_t(i)$ per step: maximizes expected #correct states, but the sequence can contain a zero-prob transition.

*Why log-space or scaling?*

- $\alpha_t$ is a product of $2t$ probs → underflows for long $T$.

*Cons?*

- Discrete states → hidden memory is limited to $\log_2|\mathcal{S}|$ bits per step.
- Observations independent given the state → no direct long-range dependence.
- Baum–Welch finds a local optimum only.

*HMM vs CRF?*

- HMM: generative $p(z,y)$.
- Linear-chain CRF: discriminative $p(z\mid y)$ → arbitrary overlapping features of $y$.
- Same DP structure (forward-backward, Viterbi).

*How does HMM relate to state-space models?*

- HMM = [state-space model](bayes.md#state-space-models) w/ finite discrete states.
- Linear-Gaussian states → Kalman filter. Same predict → update recursion.
```

&nbsp;

### Kalman Filter
- **What**: Exact recursive posterior $p(\mathbf{z}_t\mid\mathbf{y}_{1:t})$ for a linear-Gaussian state-space model.
- **Why**: Track a hidden continuous state from noisy observations, online.
    - Linear maps & conditioning keep Gaussians Gaussian → posterior = mean + covariance → $O(1)$ memory in $t$.
- **How**: Repeat 2 steps per time step:
    1. **Predict**: push the posterior through the dynamics → mean moves, covariance grows by process noise.
    2. **Update**: compare the new observation w/ its prediction (innovation) → correct the mean by gain × innovation → covariance shrinks.
    - Gain = trust ratio: uncertain prediction / precise sensor → follow the observation; reverse → follow the model.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{z}_t\in\mathbb{R}^{d_z}$: Hidden state = $X_t$ of this page; named $\mathbf{z}_t$ as in [HMM](#hmm).
    - $\mathbf{y}_t\in\mathbb{R}^{d_y}$: Observation.
- Params:
    - $A\in\mathbb{R}^{d_z\times d_z}$: Dynamics matrix.
    - $H\in\mathbb{R}^{d_y\times d_z}$: Observation matrix.
    - $Q\in\mathbb{R}^{d_z\times d_z}$: Process-noise covariance.
    - $R\in\mathbb{R}^{d_y\times d_y}$: Observation-noise covariance.
- Intermediate values:
    - $\hat{\mathbf{z}}_{t\mid t-1}$: Predicted (prior) mean; textbook $\hat{\mathbf{x}}^-$.
    - $\Sigma_{t\mid t-1}$: Predicted covariance; textbook $P^-$. $\Sigma$, ❌$P$ ← page $P$ = transition matrix.
    - $\hat{\mathbf{z}}_{t\mid t}$: Filtered (posterior) mean.
    - $\Sigma_{t\mid t}$: Filtered covariance.
    - $\mathbf{e}_t$: Innovation.
    - $S_t$: Innovation covariance; invertible if $R\succ0$.
    - $I$: Identity, $d_z\times d_z$.
    - $K_t\in\mathbb{R}^{d_z\times d_y}$: Kalman gain.

Model:

$$
\begin{align*}
\mathbf{z}_t&=A\mathbf{z}_{t-1}+\mathbf{w}_t, && \mathbf{w}_t\sim\mathcal{N}(\mathbf{0},Q) \\
\mathbf{y}_t&=H\mathbf{z}_t+\mathbf{v}_t, && \mathbf{v}_t\sim\mathcal{N}(\mathbf{0},R) \\
\mathbf{z}_0&\sim\mathcal{N}(\hat{\mathbf{z}}_{0\mid 0},\Sigma_{0\mid 0})
\end{align*}
$$

- $\mathbf{w}_t$: Process noise.
- $\mathbf{v}_t$: Observation noise.
- All noises & $\mathbf{z}_0$ mutually independent; noises white (independent across $t$).

Inference (filtering): $p(\mathbf{z}_t\mid\mathbf{y}_{1:t})=\mathcal{N}(\hat{\mathbf{z}}_{t\mid t},\Sigma_{t\mid t})$.

Predict:

$$
\hat{\mathbf{z}}_{t\mid t-1}=A\hat{\mathbf{z}}_{t-1\mid t-1},\qquad
\Sigma_{t\mid t-1}=A\Sigma_{t-1\mid t-1}A^T+Q
$$

Update:

$$
\begin{align*}
\mathbf{e}_t&=\mathbf{y}_t-H\hat{\mathbf{z}}_{t\mid t-1} \\
S_t&=H\Sigma_{t\mid t-1}H^T+R \\
K_t&=\Sigma_{t\mid t-1}H^TS_t^{-1} \\
\hat{\mathbf{z}}_{t\mid t}&=\hat{\mathbf{z}}_{t\mid t-1}+K_t\mathbf{e}_t \\
\Sigma_{t\mid t}&=(I-K_tH)\Sigma_{t\mid t-1}
\end{align*}
$$

Joseph form (same value at optimal $K_t$; valid for any gain; better preserves symmetry & PSD under round-off):

$$
\Sigma_{t\mid t}=(I-K_tH)\Sigma_{t\mid t-1}(I-K_tH)^T+K_tRK_t^T
$$

Likelihood (prediction-error decomposition):

$$
\log p(\mathbf{y}_{1:T})=\sum_{t=1}^{T}\log\mathcal{N}(\mathbf{y}_t;\,H\hat{\mathbf{z}}_{t\mid t-1},\,S_t)
$$

- $T$: Sequence length.
- Cost per step: $O(d_z^3+d_y^3)$ (matrix products & $S_t^{-1}$).
```

```{tip} Derivation
:class: dropdown
*Why are predict & update exact?*

1. **Predict**: $\mathbf{z}_{t-1}\mid\mathbf{y}_{1:t-1}\sim\mathcal{N}(\hat{\mathbf{z}}_{t-1\mid t-1},\Sigma_{t-1\mid t-1})$, $\mathbf{w}_t$ independent.
    - Linear map + independent Gaussian → Gaussian w/ mean $A\hat{\mathbf{z}}_{t-1\mid t-1}$, cov $A\Sigma_{t-1\mid t-1}A^T+Q$.
2. **Joint** of $(\mathbf{z}_t,\mathbf{y}_t)$ given $\mathbf{y}_{1:t-1}$ is Gaussian:

$$
\begin{pmatrix}\mathbf{z}_t\\ \mathbf{y}_t\end{pmatrix}\sim\mathcal{N}\left(\begin{pmatrix}\hat{\mathbf{z}}_{t\mid t-1}\\ H\hat{\mathbf{z}}_{t\mid t-1}\end{pmatrix},\begin{pmatrix}\Sigma_{t\mid t-1} & \Sigma_{t\mid t-1}H^T\\ H\Sigma_{t\mid t-1} & S_t\end{pmatrix}\right)
$$

- Cross-cov: $Cov[\mathbf{z}_t,H\mathbf{z}_t+\mathbf{v}_t]=\Sigma_{t\mid t-1}H^T$ ← $\mathbf{v}_t$ independent.

3. **Gaussian conditioning** ($\mathbb{E}[\mathbf{a}\mid\mathbf{b}]=\boldsymbol{\mu}_a+\Sigma_{ab}\Sigma_{bb}^{-1}(\mathbf{b}-\boldsymbol{\mu}_b)$, $Cov[\mathbf{a}\mid\mathbf{b}]=\Sigma_{aa}-\Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba}$):

$$
\begin{align*}
\hat{\mathbf{z}}_{t\mid t}&=\hat{\mathbf{z}}_{t\mid t-1}+\Sigma_{t\mid t-1}H^TS_t^{-1}\mathbf{e}_t=\hat{\mathbf{z}}_{t\mid t-1}+K_t\mathbf{e}_t \\
\Sigma_{t\mid t}&=\Sigma_{t\mid t-1}-\Sigma_{t\mid t-1}H^TS_t^{-1}H\Sigma_{t\mid t-1}=(I-K_tH)\Sigma_{t\mid t-1}
\end{align*}
$$

*Why does the Joseph form hold?*

1. Any gain $K$: error $\mathbf{z}_t-\hat{\mathbf{z}}_{t\mid t}=(I-KH)(\mathbf{z}_t-\hat{\mathbf{z}}_{t\mid t-1})-K\mathbf{v}_t$.
2. 2 independent terms → $\Sigma_{t\mid t}=(I-KH)\Sigma_{t\mid t-1}(I-KH)^T+KRK^T$.
3. Optimal $K=\Sigma_{t\mid t-1}H^TS_t^{-1}$ → $KS_tK^T=\Sigma_{t\mid t-1}H^TK^T$ → expansion collapses to $(I-KH)\Sigma_{t\mid t-1}$.
```

```{note} Example
:class: dropdown
1-D random walk: $A=H=1$, $Q=1$, $R=2$, $\hat{z}_{0\mid 0}=0$, $\Sigma_{0\mid 0}=1$, $y_1=2$.
- Predict: $\hat{z}_{1\mid 0}=0$, $\Sigma_{1\mid 0}=1+1=2$.
- Update: $e_1=2$, $S_1=2+2=4$, $K_1=\frac{2}{4}=0.5$.
- Posterior: $\hat{z}_{1\mid 1}=0+0.5\cdot 2=1$, $\Sigma_{1\mid 1}=(1-0.5)\cdot 2=1$.
- $\Sigma_{1\mid 1}=\Sigma_{0\mid 0}$ → already at steady state: $K_t=0.5$ $\forall t$, independent of the data.
```

````{important} Code
:class: dropdown
```python
import numpy as np

class KalmanFilter:
    def __init__(self, A, H, Q, R, z0, Sigma0):
        self.A, self.H, self.Q, self.R = A, H, Q, R
        self.z, self.Sigma = z0, Sigma0                  ## posterior mean & cov at t-1

    def predict(self):
        ## push the Gaussian through linear dynamics; Q adds process uncertainty
        self.z = self.A @ self.z
        self.Sigma = self.A @ self.Sigma @ self.A.T + self.Q

    def update(self, y):
        e = y - self.H @ self.z                          ## innovation: surprise in the obs
        S = self.H @ self.Sigma @ self.H.T + self.R
        K = np.linalg.solve(S, self.H @ self.Sigma).T    ## K = Sigma H^T S^{-1}, no explicit inverse
        self.z = self.z + K @ e
        I_KH = np.eye(len(self.z)) - K @ self.H
        ## Joseph form: same value, better preserves symmetric PSD under round-off
        self.Sigma = I_KH @ self.Sigma @ I_KH.T + K @ self.R @ K.T
        return self.z, self.Sigma

## Example: constant velocity, state = (position, velocity), observe position only
kf = KalmanFilter(A=np.array([[1.0, 1.0], [0.0, 1.0]]), H=np.array([[1.0, 0.0]]),
                  Q=0.1 * np.eye(2), R=np.array([[1.0]]),
                  z0=np.array([0.0, 1.0]), Sigma0=np.eye(2))
kf.predict()                                ## prior mean (1, 1)
z, Sigma = kf.update(np.array([2.0]))
print(z.round(3))       ## [1.677 1.323] -> velocity also corrected via pos-vel covariance
print(Sigma.round(3))   ## [[0.677 0.323]
                        ##  [0.323 0.777]]
```
````

```{attention} Q&A
:class: dropdown
*How does it relate to the HMM forward algorithm?*

- Same predict → update recursion on $p(\mathbf{z}_t\mid\mathbf{y}_{1:t})$.
- HMM: discrete states → sum over states. Kalman: Gaussian states → closed-form Gaussian integral.
- $\alpha_t$ ↔ $(\hat{\mathbf{z}}_{t\mid t},\Sigma_{t\mid t})$ + normalizer $p(\mathbf{y}_t\mid\mathbf{y}_{1:t-1})=\mathcal{N}(\mathbf{e}_t;\mathbf{0},S_t)$.
- Backward pass ↔ Rauch–Tung–Striebel (RTS) smoother. Both = [state-space models](bayes.md#state-space-models).

*Why is the gain a trust ratio?*

- Scalar, $H=1$: $K_t=\frac{\Sigma_{t\mid t-1}}{\Sigma_{t\mid t-1}+R}$.
- $R\to 0$ → $K_t\to 1$ → copy the observation.
- $\Sigma_{t\mid t-1}\to 0$ → $K_t\to 0$ → ignore the observation.

*Does the covariance depend on the data?*

- No. $\Sigma_{t\mid t-1}$, $S_t$, $K_t$ use only $A,H,Q,R,\Sigma_{0\mid 0}$ → precompute offline.
- Only the mean uses $\mathbf{y}_t$.

*What is the steady-state gain?*

- Time-invariant $A,H,Q,R$ + $(A,H)$ detectable + $(A,Q^{1/2})$ stabilizable → $\Sigma_{t\mid t-1}\to\Sigma_\infty$ for any $\Sigma_{0\mid 0}$.
- $\Sigma_\infty$ solves the discrete algebraic Riccati equation: $\Sigma_\infty=A\Sigma_\infty A^T-A\Sigma_\infty H^T(H\Sigma_\infty H^T+R)^{-1}H\Sigma_\infty A^T+Q$.
- → constant gain $K_\infty$ → cheap fixed linear filter.

*What if the noise is not Gaussian?*

- Same recursion = best **linear** MMSE estimator (only means & covariances used).
- No longer the exact posterior. Multimodal posteriors are lost.

*What if dynamics or observations are nonlinear?*

- **EKF**: linearize $f,h$ w/ Jacobians at the current estimate → Kalman equations. Breaks under strong nonlinearity.
- **UKF**: propagate deterministic sigma points through $f,h$ → refit mean & cov. No Jacobians; often more accurate than EKF.
- **Particle filter**: weighted samples → any nonlinear/non-Gaussian model. See [SMC](sampling.md#smc).
- $f,h$: nonlinear dynamics & observation functions replacing $A,H$.

*Why the Joseph form?*

- $(I-K_tH)\Sigma_{t\mid t-1}$ = difference of matrices → round-off → asymmetric or not PSD.
- Joseph = sum of PSD terms → PSD in exact arithmetic; more robust to round-off; also correct for suboptimal $K$.

*How to learn $A,H,Q,R$?*

- Maximize the prediction-error log-likelihood w/ gradients, or EM w/ the RTS smoother.

*Link to Bayesian linear regression?*

- State = weights, $A=I$, $Q=0$, time-varying $H_t=\mathbf{x}_t^T$ → Kalman = recursive [Bayesian linear regression](bayes.md#bayesian-linear-regression) (recursive least squares).
- $Q>0$ → weights drift → tracks non-stationary data.

*Cons?*

- Gaussian posterior → unimodal only.
- Needs a known (or learned) $Q,R$. Wrong $Q,R$ → over/under-confident estimates.
- $O(d_z^3)$ per step → expensive for large states.
```

&nbsp;

## Random Walks & Martingales

### Random Walk
- **What**: Partial sums of iid steps: $S_n=S_0+\sum_{t=1}^n\xi_t$.
- **Why**: Simplest process w/ dependence across time.
    - Discrete prototype of [Brownian motion](#brownian-motion).
    - Models gambler's wealth, MCMC proposals, graph exploration.
- **How**: Each step adds an independent increment → independent, stationary increments.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\xi_t$: Step at time $t$, iid.
    - $p$: $\mathbb{P}(\xi_t=+1)$ for the simple walk.
    - $q=1-p$.

Simple walk on $\mathbb{Z}$ ($\xi_t=\pm1$):

$$
\mathbb{E}[S_n-S_0]=n(p-q),\qquad \text{Var}(S_n)=4npq
$$

- Symmetric ($p=\frac12$): mean 0, variance $n$ → typical distance $\sim\sqrt n$.

**Pólya recurrence**: simple symmetric walk on $\mathbb{Z}^d$:
- $d=1,2$ → recurrent (returns w.p. 1).
- $d\ge3$ → transient. $d=3$ return prob $\approx0.34$.

Random walk on an undirected connected graph:

$$
P=D^{-1}A,\qquad \pi_i=\frac{\deg(i)}{2|E|}
$$

- $A$: Adjacency matrix.
- $D$: Degree matrix.
- $|E|$: #edges.
- Reversible ← $\pi_iP_{ij}=\frac{1}{2|E|}=\pi_jP_{ji}$ for every edge.
- Periodic $\Leftrightarrow$ bipartite.

**Donsker**: steps w/ mean 0, variance $\sigma^2$:

$$
\left(\frac{S_{\lfloor nt\rfloor}}{\sigma\sqrt n}\right)_{t\ge0}\xrightarrow{d}(W_t)_{t\ge0}
$$

- CLT for whole paths → Brownian motion is universal.
```

```{attention} Q&A
:class: dropdown
*Why $\sqrt n$, not $n$?*

- Independent steps → variances add → std $\propto\sqrt n$.
- Positive & negative steps cancel.

*Why can a drunk man find home, but a drunk bird may not?*

- Pólya: $d=2$ recurrent, $d=3$ transient.
- Higher dimension → more directions to escape → return prob $<1$.

*Is a random walk stationary?*

- No. $\text{Var}(S_n)$ grows w/ $n$.
- Its increments are stationary.
```

&nbsp;

#### Gambler's Ruin
- **What**: Simple random walk on $\{0,\ldots,N\}$, absorbed at $0$ & $N$.
- **Why**: Canonical "which boundary first?" problem.
    - Template for [hitting times](#hitting-time) & [optional stopping](#optional-stopping-theorem).
- **How**: Solve the first-step recursion $h_i=ph_{i+1}+qh_{i-1}$ w/ $h_0=0$, $h_N=1$.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $N$: Upper absorbing boundary (target wealth).
    - $h_i$: $\mathbb{P}$(reach $N$ before $0$ | start at $i$).
    - $D_i$: Expected duration from $i$.

Win probability:

$$
h_i=\begin{cases}\dfrac{i}{N} & p=\frac12\\[2mm] \dfrac{1-(q/p)^i}{1-(q/p)^N} & p\ne\frac12\end{cases}
$$

Expected duration (fair game):

$$
D_i=i(N-i)
$$
```

```{tip} Derivation
:class: dropdown
*How does optional stopping give these in 2 lines?*

1. Fair: $S_n$ is a martingale, $\tau$ = exit time.
2. $\mathbb{E}[S_\tau]=i$ → $N h_i+0\cdot(1-h_i)=i$ → $h_i=i/N$.
3. $S_n^2-n$ is also a martingale → $\mathbb{E}[S_\tau^2]-\mathbb{E}[\tau]=i^2$ → $N^2\frac iN-D_i=i^2$ → $D_i=i(N-i)$.
4. Unfair: $(q/p)^{S_n}$ is a martingale ← $p(q/p)^{s+1}+q(q/p)^{s-1}=(q/p)^s(q+p)$.
5. → $(q/p)^i=h_i(q/p)^N+(1-h_i)$ → formula above.
```

```{attention} Q&A
:class: dropdown
*Fair game vs an infinitely rich casino?*

- $N\to\infty$ → $h_i=i/N\to0$ → ruin w.p. 1.

*Small house edge matters how much?*

- $p<\frac12$ → $(q/p)^N$ explodes → $h_i\approx(p/q)^{N-i}$ → exponentially small in the distance to the target.
```

&nbsp;

### Martingale
- **What**: Adapted, integrable process w/ $\mathbb{E}[X_{t+1}\mid\mathcal{F}_t]=X_t$.
- **Why**: Model of a fair game.
    - Given today's info, no expected gain tomorrow.
    - Unlocks strong tools w/o independence: optional stopping, convergence, concentration.
- **How**: Check 3 conditions: adapted, $\mathbb{E}|X_t|<\infty$, conditional mean of next = curr.
    - **Submartingale** ($\ge$): tends up. **Supermartingale** ($\le$): tends down.

```{note} Math
:class: dropdown
Definition:

$$
\mathbb{E}[X_{t+1}\mid\mathcal{F}_t]=X_t\quad\Rightarrow\quad\mathbb{E}[X_t]=\mathbb{E}[X_0]\;\;\forall t
$$

Common martingales:
- Sum of independent zero-mean variables.
- Product of independent nonnegative mean-1 variables.
- $S_n^2-n$ for the simple symmetric random walk.
- **Doob martingale**: $M_t=\mathbb{E}[Y\mid\mathcal{F}_t]$ for any integrable $Y$.
- Likelihood ratio $L_n=\prod_{t=1}^n\frac{q(\xi_t)}{p(\xi_t)}$ for $\xi_t\overset{iid}{\sim}p$, if $q$ is supported within $p$'s support.

Properties:
- **Martingale convergence**: $\sup_t\mathbb{E}|X_t|<\infty$ → $X_t$ converges a.s. to an integrable limit.
- Nonnegative supermartingale → converges a.s.
- Convex $\varphi$ of a martingale (integrable) → submartingale (Jensen).
```

```{attention} Q&A
:class: dropdown
*Martingale vs Markov?*

- Independent concepts.
- Random walk w/ drift: Markov, ❌martingale.
- Martingale whose step size depends on older history: martingale, ❌Markov.

*Why is a Bayesian posterior mean a martingale?*

- $\mathbb{E}[\theta\mid\mathcal{D}_{1:t}]$ is a Doob martingale.
- → You cannot predict the direction of your own next belief update.
- Coherent forecasts $\mathbb{P}(A\mid\mathcal{F}_t)$ of a fixed event $A$ form a martingale too.
- Calibration alone is weaker ← a long-run frequency property, ❌martingale.

*How do martingales appear in RL?*

- True $V^\pi$ → TD errors $\delta_t=r_{t+1}+\gamma V^\pi(s_{t+1})-V^\pi(s_t)$ satisfy $\mathbb{E}[\delta_t\mid s_0,a_0,\ldots,s_t]=0$ → martingale differences.
- Conditioning must stop before $a_t$. Given $a_t$ too: $\mathbb{E}[\delta_t\mid\ldots,s_t,a_t]=Q^\pi(s_t,a_t)-V^\pi(s_t)\ne0$ in general.
- Stochastic approximation convergence proofs (TD, Q-learning, SGD) bound such martingale noise terms.

*Does $\mathbb{E}[X_t]$ constant imply martingale?*

- No. Conditional mean must hold given every history.
```

&nbsp;

#### Optional Stopping Theorem
- **What**: $\mathbb{E}[X_\tau]=\mathbb{E}[X_0]$ for a martingale $X$ & stopping time $\tau$, under boundedness conditions.
- **Why**: No rule for *when* to stop beats a fair game.
    - Computes hitting probs & expected times in 1 line.
- **How**: Holds if any 1 condition holds:
    1. $\tau$ bounded.
    2. $\tau<\infty$ a.s. & $|X_{t\wedge\tau}|\le K$ for all $t$.
    3. $\mathbb{E}[\tau]<\infty$ & increments bounded: $|X_{t+1}-X_t|\le K$.

```{note} Example
:class: dropdown
**Doubling strategy** (counterexample):
1. Bet $1$ on a fair coin. On loss, double the bet. Stop at first win.
2. $\tau$ ~ Geometric($\frac12$) → $\tau<\infty$ a.s., $\mathbb{E}[\tau]=2$.
3. Final profit $X_\tau=1$ always → $\mathbb{E}[X_\tau]=1\ne0=\mathbb{E}[X_0]$.
4. Fails conditions 2 & 3 ← losses before the win & bet sizes are unbounded.
```

```{attention} Q&A
:class: dropdown
*What is Wald's identity?*

- iid $\xi_t$ w/ finite mean, stopping time $\tau$ w/ $\mathbb{E}[\tau]<\infty$:

    $$
    \mathbb{E}\left[\sum_{t=1}^{\tau}\xi_t\right]=\mathbb{E}[\tau]\,\mathbb{E}[\xi_1]
    $$

- Follows from OST on the martingale $\sum_{t\le n}\xi_t-n\mathbb{E}[\xi_1]$.

*Why does doubling fail in practice?*

- Finite bankroll → bounded strategy → OST holds → expected profit 0.
- Small chance of a huge loss balances the frequent small wins.
```

&nbsp;

#### Azuma–Hoeffding Inequality
- **What**: Gaussian-like tail bound for martingales w/ bounded increments.
- **Why**: Hoeffding needs independence.
    - Online learning, bandits, RL & SGD produce dependent sums that are still martingales.
- **How**: Bounded steps $|X_t-X_{t-1}|\le c_t$ → same tail shape as an independent sum.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $c_t$: Bound on the $t$-th increment.
    - $\varepsilon$: Deviation.

Azuma–Hoeffding:

$$
\mathbb{P}(|X_n-X_0|\ge\varepsilon)\le2\exp\left(-\frac{\varepsilon^2}{2\sum_{t=1}^nc_t^2}\right)
$$

**McDiarmid** (bounded differences): independent $\xi_1,\ldots,\xi_n$; changing $\xi_t$ alone changes $g$ by at most $c_t$:

$$
\mathbb{P}\left(g(\xi_{1:n})-\mathbb{E}[g]\ge\varepsilon\right)\le\exp\left(-\frac{2\varepsilon^2}{\sum_{t=1}^nc_t^2}\right)
$$

- $g$: Function of $n$ independent inputs.
- Proof: Azuma-type argument on the Doob martingale $\mathbb{E}[g\mid\xi_{1:t}]$.
- Constant 2 vs Azuma's $\frac12$ ← each Doob increment lies in an interval of length $c_t$.
```

```{attention} Q&A
:class: dropdown
*Where does McDiarmid appear in ML?*

- Generalization bounds: empirical risk changes by $\le\frac1m$ when 1 bounded-loss sample changes → deviation $O(\sqrt{\log(1/\delta)/m})$.

*When is Azuma too loose?*

- Uses only worst-case step size.
- Small conditional variance → Freedman/Bernstein-type martingale bounds are tighter.
```

&nbsp;

## Continuous-Time Jump Processes

### Poisson Process
- **What**: Counting process w/ independent increments & $N_{t+s}-N_s\sim\text{Pois}(\lambda t)$.
- **Why**: Model of completely random arrivals at a constant rate.
    - Server requests, photon counts, neuron spikes, rare failures.
- **How**: 3 equivalent views:
    1. **Counts**: $N_0=0$, independent increments, Poisson counts.
    2. **Gaps**: interarrival times iid [$\text{Exp}(\lambda)$](dist.md#exponential).
    3. **Local**: given the whole past $\mathcal{F}_t$, $\mathbb{P}(\text{1 arrival in }[t,t+\Delta]\mid\mathcal{F}_t)=\lambda\Delta+o(\Delta)$, $\mathbb{P}(\ge2\mid\mathcal{F}_t)=o(\Delta)$.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $N_t$: #arrivals in $[0,t]$.
    - $A_k$: Time of the $k$-th arrival.

Counts:

$$
\mathbb{P}(N_t=k)=e^{-\lambda t}\frac{(\lambda t)^k}{k!},\qquad \mathbb{E}[N_t]=\text{Var}(N_t)=\lambda t
$$

Arrival times: $A_k\sim\text{Gamma}(k,\text{rate}=\lambda)$ ← sum of $k$ iid $\text{Exp}(\lambda)$.

Properties:
- **Superposition**: independent $\text{PP}(\lambda_1)+\text{PP}(\lambda_2)=\text{PP}(\lambda_1+\lambda_2)$.
- **Thinning**: keep each arrival independently w.p. $\rho$ → $\text{PP}(\lambda\rho)$; kept & dropped streams are independent.
- **Conditional uniformity**: given $N_t=k$, arrival times = sorted $k$ iid $\text{Unif}[0,t]$.
- **Inhomogeneous**: rate $\lambda(t)$ → $N_t\sim\text{Pois}\left(\int_0^t\lambda(s)ds\right)$.
```

````{important} Code
:class: dropdown
```python
import numpy as np

class PoissonProcess:
    def __init__(self, rate, seed=0):
        self.rate, self.rng = rate, np.random.default_rng(seed)

    def sample(self, T):
        ## interarrival gaps are iid Exp(rate) -> cumulative sums = arrival times
        times = []
        t = self.rng.exponential(1 / self.rate)
        while t < T:
            times.append(t)
            t += self.rng.exponential(1 / self.rate)
        return np.array(times)

## Example
pp = PoissonProcess(rate=2.0)
counts = [len(pp.sample(T=10.0)) for _ in range(2000)]
print(np.mean(counts), np.var(counts))   ## both ~ rate * T = 20
```
````

```{attention} Q&A
:class: dropdown
*What is the inspection (bus) paradox?*

- Arrive at a random time $t$. Gap that contains $t$ has expected length $\to\frac2\lambda$, not $\frac1\lambda$.
- ← Long gaps cover more time → more likely to contain $t$.
- Residual wait is still $\text{Exp}(\lambda)$ ← memoryless.

*When does a Poisson model fail?*

- Overdispersion: $\text{Var}>\text{mean}$ in real counts → negative binomial / Cox (random-rate) process.
- Self-excitation (one event triggers more, e.g., retweets, aftershocks) → Hawkes process.
- Regular spacing (heartbeats) → renewal process w/ non-exponential gaps.

*Why is Poisson the limit of rare events?*

- Many independent trials, each w/ small prob → [Binomial → Poisson](dist.md#poisson).
```

&nbsp;

### CTMC
- **Name**: Continuous-Time Markov Chain
- **What**: Markov process on a discrete state space in continuous time.
- **Why**: Real events do not occur on a clock grid.
    - Queues, chemical reactions, epidemics (SIR), reliability, molecular evolution.
- **How**:
    1. In state $i$, wait a holding time $\sim\text{Exp}(q_i)$.
    2. Jump to $j\ne i$ w.p. $q_{ij}/q_i$ (**embedded jump chain**).

```{note} Math
:class: dropdown
Notations:
- Params:
    - $Q$: Generator (rate) matrix.
    - $q_{ij}\ge0$: Jump rate $i\to j$, $i\ne j$.
    - $q_i=\sum_{j\ne i}q_{ij}$: Total exit rate of $i$.
- Misc:
    - $P(t)$: Transition matrix over time $t$, $P(t)_{ij}=\mathbb{P}(X_{s+t}=j\mid X_s=i)$.

Generator:

$$
Q_{ij}=q_{ij}\;(i\ne j),\qquad Q_{ii}=-q_i\qquad\Rightarrow\quad \text{rows sum to }0
$$

Transitions (finite $\mathcal{S}$):

$$
P(t)=e^{tQ},\qquad P'(t)=P(t)Q\;\;\text{(forward)},\qquad P'(t)=QP(t)\;\;\text{(backward)}
$$

Stationary distribution & detailed balance:

$$
\pi Q=0,\qquad \pi_iq_{ij}=\pi_jq_{ji}
$$

**Uniformization**: pick $\Lambda\ge\max_iq_i$:

$$
\tilde P=I+\frac{Q}{\Lambda},\qquad P(t)=\sum_{k\ge0}e^{-\Lambda t}\frac{(\Lambda t)^k}{k!}\tilde P^k
$$

- CTMC = DTMC $\tilde P$ whose steps occur at $\text{PP}(\Lambda)$ times.
- Poisson process = CTMC on $\{0,1,\ldots\}$ w/ $q_{i,i+1}=\lambda$.
```

```{note} Example
:class: dropdown
**M/M/1 queue**: arrivals $\text{PP}(\lambda)$, 1 server w/ $\text{Exp}(\mu)$ service.

- Birth–death CTMC on #customers in system: $q_{i,i+1}=\lambda$ ($i\ge0$), $q_{i,i-1}=\mu$ ($i\ge1$).
- Load $\rho=\lambda/\mu<1$ → detailed balance $\pi_i\lambda=\pi_{i+1}\mu$:

    $$
    \pi_i=(1-\rho)\rho^i,\qquad \mathbb{E}[\#\text{in system}]=\frac{\rho}{1-\rho}
    $$

- Little's law $L=\lambda W$ → mean time in system $W=\frac{1}{\mu-\lambda}$.
- $\rho\to1$ → delay explodes ← why servers must not run near 100% utilization.
```

````{important} Code
:class: dropdown
```python
import numpy as np

class CTMC:
    def __init__(self, Q, seed=0):
        self.Q = np.asarray(Q, dtype=float)
        assert np.allclose(self.Q.sum(axis=1), 0), "generator rows must sum to 0"
        self.rng = np.random.default_rng(seed)

    def sample(self, x0, T):
        ## Gillespie: Exp(q_i) holding time, then jump w.p. q_ij / q_i
        t, x, path = 0.0, x0, [(0.0, x0)]
        while True:
            rate = -self.Q[x, x]
            if rate == 0:                          ## absorbing state
                return path
            t += self.rng.exponential(1 / rate)
            if t >= T:
                return path
            jump = self.Q[x].clip(min=0) / rate    ## drops the negative diagonal
            x = self.rng.choice(len(self.Q), p=jump)
            path.append((t, x))

## Example: on/off switch, pi = (2/3, 1/3)
T = 1000.0
path = CTMC([[-1.0, 1.0], [2.0, -2.0]]).sample(0, T)
ends = [t for t, _ in path[1:]] + [T]
frac0 = sum(t1 - t0 for (t0, x), t1 in zip(path, ends) if x == 0) / T
print(frac0)   ## ~0.667
```
````

```{attention} Q&A
:class: dropdown
*Why exponential holding times?*

- Markov in continuous time → time already spent in $i$ must not matter → memoryless.
- Exponential is the only memoryless continuous distribution.

*CTMC vs DTMC?*

- DTMC: transition probs per step.
- CTMC: transition rates per unit time; $P(t)=e^{tQ}$.
- Small $\Delta$: $P(\Delta)\approx I+\Delta Q$.
```

&nbsp;

## Continuous Paths

### Brownian Motion
- **What**: Continuous-path process w/ independent Gaussian increments $W_t-W_s\sim\mathcal{N}(0,t-s)$.
    - Alias: Wiener process.
- **Why**: Universal continuous noise.
    - Scaling limit of every finite-variance random walk (Donsker) → appears whenever many small independent shocks add up.
    - Building block of [SDEs](#sde) → diffusion models, Langevin sampling, finance.
- **How**: Random walk w/ steps $\pm\sqrt{\Delta t}$ every $\Delta t$ → let $\Delta t\to0$.

```{note} Math
:class: dropdown
Definition:
1. $W_0=0$.
2. Independent increments.
3. $W_t-W_s\sim\mathcal{N}(0,t-s)$ for $s<t$.
4. Continuous paths.

Properties:
- $\mathbb{E}[W_t]=0$, $\text{Cov}(W_s,W_t)=\min(s,t)$ → Gaussian process w/ kernel $\min(s,t)$.
- **Scaling**: $c^{-1/2}W_{ct}$ is a Brownian motion for $c>0$.
- **Martingales**: $W_t$, $W_t^2-t$, $\exp(\theta W_t-\frac12\theta^2t)$.
- **Quadratic variation**: partitions of $[0,t]$ w/ mesh $\to0$:

    $$
    \sum_k\left(W_{t_{k+1}}-W_{t_k}\right)^2\xrightarrow{L^2}t
    $$

- Paths are nowhere differentiable a.s. & have infinite total variation on every interval.
- **Reflection principle** ($a>0$):

    $$
    \mathbb{P}\left(\max_{s\le t}W_s\ge a\right)=2\,\mathbb{P}(W_t\ge a)
    $$

- Hitting time $\tau_a$ of level $a\ne0$: $\tau_a<\infty$ a.s., but $\mathbb{E}[\tau_a]=\infty$.
- First-passage density (from reflection, $a>0$):

    $$
    f_{\tau_a}(t)=\frac{a}{\sqrt{2\pi t^3}}\exp\left(-\frac{a^2}{2t}\right),\quad t>0
    $$

    - Tail $\sim t^{-1/2}$ → infinite mean.
- $d$-dim Brownian motion: $d$ independent coordinates.
```

````{important} Code
:class: dropdown
```python
import numpy as np

class BrownianMotion:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def sample(self, T, n_steps, dim=1):
        dt = T / n_steps
        ## independent N(0, dt) increments -> std scales as sqrt(dt), not dt
        dW = self.rng.normal(0.0, np.sqrt(dt), size=(n_steps, dim))
        return np.vstack([np.zeros((1, dim)), np.cumsum(dW, axis=0)])

## Example
W = BrownianMotion().sample(T=1.0, n_steps=100_000)
print(W.shape)                           ## (100001, 1)
print((np.diff(W, axis=0) ** 2).sum())   ## quadratic variation ~ T = 1.0
```
````

```{attention} Q&A
:class: dropdown
*Why is the path nowhere differentiable?*

- $|W_{t+\Delta}-W_t|\sim\sqrt\Delta$ → difference quotient $\sim1/\sqrt\Delta\to\infty$.

*Why does nonzero quadratic variation matter?*

- Smooth functions: $\sum(\Delta f)^2\to0$.
- Brownian motion: $\sum(\Delta W)^2\to t$ → $(dW)^2=dt$ is first order.
- → 2nd-order Taylor term survives → ordinary chain rule fails → [Itô calculus](#ito-calculus).

*Is Brownian motion stationary?*

- No. $\text{Var}(W_t)=t$ grows.
- Increments are stationary.
- Stationary continuous analog: [OU process](#ornstein-uhlenbeck-process).
```

&nbsp;

(ito-calculus)=
### Itô Calculus
- **What**: Calculus for integrals against Brownian motion, w/ a correction term from quadratic variation.
- **Why**: Brownian paths have infinite variation → Riemann–Stieltjes integrals $\int H\,dW$ fail.
    - Need a rule for $d f(X_t)$ when $X_t$ is driven by noise.
- **How**:
    1. **Itô integral**: limit of $\sum_kH_{t_k}(W_{t_{k+1}}-W_{t_k})$ w/ **left** endpoints → non-anticipating.
    2. **Itô's lemma**: Taylor-expand $f$ to 2nd order, keep $(dW)^2=dt$.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $H_t$: Adapted integrand w/ $\mathbb{E}\int_0^tH_s^2ds<\infty$.
    - $\mu_t$: Drift of an Itô process.
    - $\sigma_t$: Volatility (noise scale) of an Itô process.
    - $f(t,x)$: Function, $C^1$ in $t$ & $C^2$ in $x$.

Itô integral properties:

$$
\mathbb{E}\left[\int_0^tH_s\,dW_s\right]=0,\qquad \mathbb{E}\left[\left(\int_0^tH_s\,dW_s\right)^2\right]=\mathbb{E}\int_0^tH_s^2\,ds
$$

- 2nd identity: **Itô isometry**.
- $\int_0^tH_s\,dW_s$ is a martingale in $t$.

Itô process & multiplication table:

$$
dX_t=\mu_t\,dt+\sigma_t\,dW_t,\qquad dt\cdot dt=0,\quad dt\cdot dW_t=0,\quad dW_t\cdot dW_t=dt
$$

**Itô's lemma**:

$$
df(t,X_t)=\left(\partial_tf+\mu_t\,\partial_xf+\frac12\sigma_t^2\,\partial_{xx}f\right)dt+\sigma_t\,\partial_xf\,dW_t
$$

- $\frac12\sigma_t^2\partial_{xx}f$: Itô correction; absent in ordinary calculus.

Multivariate ($dX_t=\boldsymbol{\mu}_t\,dt+\Sigma_t\,dW_t$, $X_t\in\mathbb{R}^d$, $W_t\in\mathbb{R}^r$):

$$
df=\left(\partial_tf+\nabla f^T\boldsymbol{\mu}_t+\frac12\text{tr}\left(\Sigma_t\Sigma_t^T\nabla^2f\right)\right)dt+\nabla f^T\Sigma_t\,dW_t
$$

- $\Sigma_t\in\mathbb{R}^{d\times r}$: Noise matrix.
- $\nabla^2f$: Hessian in $x$.
- Rule: $dW^{(a)}dW^{(b)}=\mathbb{1}\{a=b\}\,dt$ ← independent coordinates.
```

```{tip} Derivation
:class: dropdown
*Why is $\int_0^tW_s\,dW_s=\frac12W_t^2-\frac12t$, not $\frac12W_t^2$?*

1. Apply Itô's lemma to $f(x)=x^2$, $X=W$ ($\mu=0$, $\sigma=1$).
2. $d(W_t^2)=2W_t\,dW_t+\frac12\cdot2\,dt=2W_t\,dW_t+dt$.
3. Integrate: $W_t^2=2\int_0^tW_s\,dW_s+t$.
4. Check: LHS mean $t$; $\int W\,dW$ is a martingale w/ mean 0 → consistent.

*Geometric Brownian motion* $dS_t=\mu S_t\,dt+\sigma S_t\,dW_t$ ($\mu,\sigma$ constants):

1. $f=\log S$: $\partial_xf=\frac1S$, $\partial_{xx}f=-\frac1{S^2}$.
2. $d\log S_t=\left(\mu-\frac12\sigma^2\right)dt+\sigma\,dW_t$.
3. → $S_t=S_0\exp\left(\left(\mu-\frac12\sigma^2\right)t+\sigma W_t\right)$.
```

```{attention} Q&A
:class: dropdown
*Why left endpoints?*

- $H_{t_k}$ is known before the increment $W_{t_{k+1}}-W_{t_k}$ → a trading/control strategy cannot use the future.
- → Integral is a martingale.
- With smooth integrators, the choice does not matter. With $W$, it does ← quadratic variation.

*Itô vs Stratonovich?*

- Stratonovich: midpoint rule → ordinary chain rule, but ❌martingale.
- Scalar conversion: $dX=b\,dt+\sigma(X)\circ dW$ ⇔ $dX=\left(b+\frac12\sigma\sigma'\right)dt+\sigma\,dW$.
- Itô: finance, ML, probability. Stratonovich: physics limits of smooth noise, manifolds.
- Additive noise ($\sigma$ constant) → both agree.
```

&nbsp;

### SDE
- **Name**: Stochastic Differential Equation
- **What**: ODE w/ Brownian noise: $dX_t=f(X_t,t)\,dt+g(X_t,t)\,dW_t$.
- **Why**: Continuous-time dynamics w/ randomness.
    - Generative [diffusion (Score SDE)](../../dl/gen/diffusion.md#score-sde), Langevin sampling, physics, finance, continuous-time models of SGD.
- **How**: Each $dt$: deterministic push $f\,dt$ + random kick of std $g\sqrt{dt}$.
    - **Euler–Maruyama** simulation: $X_{t+\Delta}=X_t+f\Delta+g\sqrt\Delta\,z$, $z\sim\mathcal{N}(0,I)$.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $f(x,t)$: Drift.
    - $g(x,t)$: Diffusion coefficient.
- Misc:
    - $\Delta$: Step size.
    - $z\sim\mathcal{N}(0,I)$: Fresh Gaussian noise each step.

Integral form (the actual meaning of the notation):

$$
X_t=X_0+\int_0^tf(X_s,s)\,ds+\int_0^tg(X_s,s)\,dW_s
$$

- 2nd integral: Itô integral.
- Existence & uniqueness: $f,g$ globally Lipschitz in $x$ & of linear growth.

Euler–Maruyama:

$$
X_{k+1}=X_k+f(X_k,t_k)\Delta+g(X_k,t_k)\sqrt\Delta\,z_k
$$

- Strong order $\frac12$ (pathwise error $\propto\Delta^{1/2}$).
- Weak order 1 (error in expectations $\propto\Delta$), for smooth coefficients.
- Milstein (scalar): add $\frac12g\,\partial_xg\left((\Delta W_k)^2-\Delta\right)$ → strong order 1 under smoothness.
    - Multi-dim noise → may need iterated Itô integrals (Lévy areas).
```

````{important} Code
:class: dropdown
```python
import numpy as np

class EulerMaruyama:
    def __init__(self, f, g, seed=0):
        self.f, self.g = f, g                  ## drift f(x, t), diffusion g(x, t) (scalar/diagonal)
        self.rng = np.random.default_rng(seed)

    def solve(self, x0, T, n_steps):
        dt = T / n_steps
        x = np.array(x0, dtype=float)
        for k in range(n_steps):
            t = k * dt
            ## Brownian increment ~ N(0, dt): sqrt(dt) is the only change from Euler for ODEs
            dW = self.rng.normal(0.0, np.sqrt(dt), size=x.shape)
            x = x + self.f(x, t) * dt + self.g(x, t) * dW
        return x

## Example: OU process dX = -theta X dt + sigma dW, 10k paths in parallel
theta, sigma = 1.0, 0.5
em = EulerMaruyama(lambda x, t: -theta * x, lambda x, t: sigma)
xT = em.solve(np.zeros(10_000), T=5.0, n_steps=500)
print(xT.var())   ## ~ sigma^2 / (2 theta) = 0.125
```
````

```{attention} Q&A
:class: dropdown
*Why not just add $g\Delta z$ like an ODE step?*

- Noise std must scale as $\sqrt\Delta$ ← variance of $W$ is additive in time.
- $g\Delta z$ → noise vanishes as $\Delta\to0$ → deterministic ODE.

*SDE vs ODE w/ random initial condition?*

- ODE: randomness only at $t=0$ → paths smooth.
- SDE: fresh noise at every instant → paths rough.
- Both can share the same marginals $p_t$ → [probability-flow ODE](#fokker-planck-equation).
```

&nbsp;

#### Fokker–Planck Equation
- **What**: PDE for the density $p_t$ of an SDE solution.
- **Why**: SDE describes 1 particle. Fokker–Planck describes the whole cloud.
    - → stationary distributions, convergence, probability-flow ODE.
- **How**: Probability is conserved → $\partial_tp=-\nabla\cdot J$, flux $J$ = drift transport − diffusion spreading.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $p_t(x)$: Density of $X_t$.
    - $D=gg^T$: Diffusion matrix.

General:

$$
\partial_tp_t=-\sum_a\partial_a\left(f_a\,p_t\right)+\frac12\sum_{a,b}\partial_a\partial_b\left(D_{ab}\,p_t\right)
$$

- $a,b$: Coordinate indices of $x$.

Scalar, state-independent $g(t)$:

$$
\partial_tp_t=-\nabla\cdot(f\,p_t)+\frac12g^2\Delta p_t
$$

- $\Delta$: Laplacian here.

**Probability-flow ODE** (same marginals $p_t$, deterministic):

$$
\frac{dx}{dt}=f(x,t)-\frac12g(t)^2\nabla_x\log p_t(x)
$$

- ← $\frac12g^2\Delta p_t=\nabla\cdot\left(\frac12g^2p_t\nabla\log p_t\right)$ → diffusion term rewritten as a drift.

Stationary density: solve $\partial_tp=0$.
```

```{attention} Q&A
:class: dropdown
*Forward vs backward Kolmogorov equation?*

- Forward (Fokker–Planck): evolves the density $p_t$ forward in time.
- Backward: evolves an expectation $u(x,t)=\mathbb{E}[\varphi(X_T)\mid X_t=x]$:

    $$
    \partial_tu+f\cdot\nabla u+\frac12g^2\Delta u=0,\qquad u(x,T)=\varphi(x)
    $$

- They are adjoint. Same pair as $\pi_{t+1}=\pi_tP$ vs $u_t=Pu_{t+1}$ for chains.
```

&nbsp;

#### Ornstein–Uhlenbeck Process
- **What**: Mean-reverting Gaussian SDE $dX_t=-\theta(X_t-\mu)\,dt+\sigma\,dW_t$.
- **Why**: Simplest stationary continuous-time process.
    - Brownian motion pulled back toward a mean.
    - Continuous analog of AR(1).
- **How**: Drift pulls toward $\mu$ ∝ distance. Noise pushes away. Balance → stationary variance $\frac{\sigma^2}{2\theta}$.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\theta>0$: Mean-reversion rate.
    - $\mu$: Long-run mean.
    - $\sigma$: Noise scale.

Solution:

$$
X_t=\mu+(X_0-\mu)e^{-\theta t}+\sigma\int_0^te^{-\theta(t-s)}dW_s
$$

Moments (fixed $X_0$):

$$
\mathbb{E}[X_t]=\mu+(X_0-\mu)e^{-\theta t},\qquad \text{Var}(X_t)=\frac{\sigma^2}{2\theta}\left(1-e^{-2\theta t}\right)
$$

Stationary law & autocovariance:

$$
X_\infty\sim\mathcal{N}\left(\mu,\frac{\sigma^2}{2\theta}\right),\qquad \text{Cov}(X_t,X_{t+h})=\frac{\sigma^2}{2\theta}e^{-\theta|h|}
$$

Exact discretization (step $\Delta$) = AR(1) w/ $\phi=e^{-\theta\Delta}$:

$$
X_{t+\Delta}=\mu+(X_t-\mu)e^{-\theta\Delta}+\sqrt{\frac{\sigma^2}{2\theta}\left(1-e^{-2\theta\Delta}\right)}\,z
$$
```

```{attention} Q&A
:class: dropdown
*How does OU relate to diffusion models?*

- VP-SDE $dx=-\frac12\beta(t)x\,dt+\sqrt{\beta(t)}\,dW$ = time-changed OU w/ $\mu=0$, $\theta=\frac12\beta$, $\sigma^2=\beta$.
- Stationary variance $\frac{\sigma^2}{2\theta}=1$ → noise converges to $\mathcal{N}(0,I)$. See [Score SDE](../../dl/gen/diffusion.md#score-sde).

*How does OU relate to Langevin dynamics?*

- OU = [Langevin](#langevin-dynamics) for a Gaussian target ← quadratic potential → linear drift.

*Why is OU special?*

- Doob's theorem: every nondegenerate stationary Gaussian Markov process in continuous time (continuous in probability) is an OU process.
```

&nbsp;

#### Langevin Dynamics
- **What**: SDE $dX_t=\nabla\log p(X_t)\,dt+\sqrt2\,dW_t$ w/ stationary distribution $p$.
- **Why**: Sample from $p$ using only its score $\nabla\log p$.
    - Normalizing constant cancels in the gradient → unnormalized densities suffice.
    - Basis of [NCSN](../../dl/gen/diffusion.md#ncsn) sampling, SGLD, MALA.
- **How**: Gradient ascent on $\log p$ + calibrated noise.
    - Gradient concentrates mass on high density. Noise spreads it. Balance = $p$.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $U(x)=-\log p(x)+\text{const}$: Potential (energy).
- Hyperparams:
    - $\delta$: Step size.
    - $T_\text{temp}$: Temperature.

Overdamped Langevin:

$$
dX_t=-\nabla U(X_t)\,dt+\sqrt{2T_\text{temp}}\,dW_t\quad\Rightarrow\quad p_\infty(x)\propto e^{-U(x)/T_\text{temp}}
$$

- $T_\text{temp}=1$ → $p_\infty=p$.

Discretization (ULA = unadjusted Langevin algorithm, same convention as [NCSN](../../dl/gen/diffusion.md#ncsn)):

$$
x_{k+1}=x_k+\frac{\delta}{2}\nabla\log p(x_k)+\sqrt\delta\,z_k
$$

- Euler–Maruyama of the SDE w/ time step $\frac\delta2$.
- Biased for $\delta>0$. MALA adds a Metropolis–Hastings accept step → exact $p$.
```

```{tip} Derivation
:class: dropdown
*Why is $p\propto e^{-U}$ stationary?*

1. Fokker–Planck w/ $f=-\nabla U$, $g=\sqrt2$: $\partial_tp=\nabla\cdot(p\nabla U)+\Delta p=\nabla\cdot(p\nabla U+\nabla p)$.
2. $p\propto e^{-U}$ → $\nabla p=-p\nabla U$.
3. → flux $p\nabla U+\nabla p=0$ → $\partial_tp=0$.
```

````{important} Code
:class: dropdown
```python
import numpy as np

class LangevinSampler:
    def __init__(self, score, step, seed=0):
        self.score, self.step = score, step    ## score(x) = grad log p(x)
        self.rng = np.random.default_rng(seed)

    def sample(self, x0, n_iters):
        x = np.array(x0, dtype=float)
        for _ in range(n_iters):
            ## half-step up log p + noise of variance step -> stationary ~ p as step -> 0
            noise = self.rng.normal(size=x.shape)
            x = x + 0.5 * self.step * self.score(x) + np.sqrt(self.step) * noise
        return x

## Example: target N(2, 1) -> score = -(x - 2); normalizer never needed
xs = LangevinSampler(lambda x: -(x - 2.0), step=0.01).sample(np.zeros(10_000), 5_000)
print(xs.mean(), xs.var())   ## ~2.0, ~1 / (1 - step / 4) = 1.0025
```
````

```{attention} Q&A
:class: dropdown
*How big is the ULA bias?*

- Target $\mathcal{N}(0,1)$: $x_{k+1}=(1-\frac\delta2)x_k+\sqrt\delta z_k$.
- Stationary variance solves $v=(1-\frac\delta2)^2v+\delta$ → $v=\frac{1}{1-\delta/4}>1$.
- → Too wide by $O(\delta)$.

*How does Langevin relate to SGD?*

- SGD = gradient descent + minibatch noise.
- Minibatch noise is anisotropic, state-dependent & not scaled to the step size → ❌Langevin, ❌exact stationary $p$.
- SGLD injects calibrated Gaussian noise & decays the step → approximate posterior samples.

*How does it relate to simulated annealing?*

- Lower temperature $T_\text{temp}$ → $e^{-U/T_\text{temp}}$ concentrates on minima of $U$. See [SA](../../ml/optim.md#sa).

*Why does Langevin fail on multimodal targets?*

- Crossing a low-density gap needs a rare large noise excursion → exponentially slow mixing.
- Fix: anneal over noise levels → [NCSN](../../dl/gen/diffusion.md#ncsn) / diffusion.
```

&nbsp;
