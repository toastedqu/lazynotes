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
# Value-Based Methods

Estimate return → act greedily w.r.t. the estimate. Order: exact backups → sampled backups → neural approximation. Study notes from {cite:t}`rlbook`; prerequisite: [MDPs, returns & Bellman equations](intro.md#bellman-equations).

Notations:
- $s_t$: State at decision $t$.
- $a_t$: Action at decision $t$.
- $r_{t+1}$: Reward after $a_t$.
- $\mathcal{S}$: State space.
- $\mathcal{A}$: Discrete action space.
- $p(s'|s,a)$: Transition probability.
- $r(s,a)$: Expected immediate reward.
- $\pi(a|s)$: Target policy.
- $\mu(a|s)$: Behavior policy.
- $\gamma$: Discount factor; $0\leq\gamma<1$ unless stated otherwise.
- $\alpha$: Step size.
- $\epsilon$: Epsilon-greedy exploration probability.
- $T$: Terminal time.
- $G_t$: Discounted return from $t$.
- $V^\pi$: True state value under $\pi$.
- $Q^\pi$: True action value under $\pi$.
- $V^*$: Optimal state value.
- $Q^*$: Optimal action value.
- $V$: Current state-value estimate.
- $Q$: Current action-value estimate.
- $\mathcal{T}^\pi$: [Bellman expectation operator](intro.md#bellman-equations).
- $\mathcal{T}^*$: Bellman optimality operator.
- $d_t$: True-termination indicator after $a_t$; $0$ at a time-limit cutoff.
- $\delta_t$: TD error.
- $\phi$: Value-network params.
- $\bar\phi$: Target-network params.
- $\mathcal{D}$: Replay buffer.

RL overrides: $a$ = action; $r$ = reward; $V/Q$ = scalar-valued functions, not matrices.
Code: reward at index `t` = $r_{t+1}$; `terminated` = $d_t$.

&nbsp;

## Planning with a Known Model

### Dynamic Programming
- **What**: Repeated Bellman backups w/ a known model.
- **Why**: Trajectory enumeration repeats work ← many trajectories share the same successor states.
- **How**: Store one value per state → back up reward + discounted successor values → reuse the updated values.

```{note} Math
:class: dropdown
Process:

$$
Q_V(s,a)=r(s,a)+\gamma\sum_{s'}p(s'|s,a)V(s')
$$

- $Q_V$: One-step lookahead from estimate $V$.
- Terminal $s'$ → $V(s')=0$.
- Evaluation backup: $V(s)\leftarrow\sum_a\pi(a|s)Q_V(s,a)$.
- Optimality backup: $V(s)\leftarrow\max_aQ_V(s,a)$.
```

````{important} Code
:class: dropdown
```python
import numpy as np

class TabularModel:
    def __init__(self, transition, reward, gamma):
        self.p = np.asarray(transition, dtype=float)  ## [S, A, S]
        self.r = np.asarray(reward, dtype=float)     ## [S, A]
        self.gamma = gamma

    def __call__(self, value):
        ## p omits terminal mass -> terminal continuation = 0; r keeps terminal reward
        return self.r + self.gamma * np.einsum("sat,t->sa", self.p, value)

## Example
model = TabularModel([[[0.0], [0.0]]], [[1.0, 2.0]], gamma=0.9)
print(model(np.zeros(1)))  ## [[1. 2.]]
```
````

```{attention} Q&A
:class: dropdown
*Is DP model-free RL?*

- No. DP plans w/ known $p$ & $r$. It needs no sampled experience.
- TD & Q-learning use the same backup w/ one sample instead of the expectation.

*Limits?*

- Dense sweep cost: $O(|\mathcal{S}|^2|\mathcal{A}|)$. Sparse $p$ ⬇️ cost.
- #states grows exponentially w/ #state variables → enumeration fails for large or continuous spaces.
- Wrong model → wrong values. DP has no data to correct the model.
```

&nbsp;

### Policy Evaluation
- **What**: Compute $V^\pi$ for a fixed $\pi$.
- **Why**: Policy improvement needs the value of the curr policy.
- **How**: Repeat evaluation backups until values stop changing, or solve the linear Bellman system.

```{note} Math
:class: dropdown
Process:

$$
P^\pi_{ss'}=\sum_a\pi(a|s)p(s'|s,a),\qquad
\mathbf{r}^\pi_s=\sum_a\pi(a|s)r(s,a)
$$

- $P^\pi$: Policy-induced transition matrix over nonterminal states.
- $\mathbf{r}^\pi$: Policy-induced expected reward vector.

$$
\mathbf{v}^\pi=\mathbf{r}^\pi+\gamma P^\pi\mathbf{v}^\pi
\quad\Leftrightarrow\quad
(I-\gamma P^\pi)\mathbf{v}^\pi=\mathbf{r}^\pi
$$

- $\mathbf{v}^\pi$: Vector of $V^\pi(s)$ over states.
- $I$: Identity matrix.
- $\gamma<1$ → $I-\gamma P^\pi$ is invertible → unique solution. Solve the system; do not form the inverse.
- Iterative form: $\mathbf{v}\leftarrow\mathcal{T}^\pi\mathbf{v}$. Each sweep shrinks the max error by a factor of at least $\gamma$.
```

````{important} Code
:class: dropdown
```python
import numpy as np

## Reuses TabularModel from rl/value.md.
class PolicyEvaluation:
    def __init__(self, model):
        self.model = model

    def __call__(self, policy):
        ## Average dynamics & rewards over the policy -> one Markov chain
        p = np.einsum("sa,sat->st", policy, self.model.p)
        r = (policy * self.model.r).sum(axis=1)
        ## Exact solve of (I - gamma P) v = r
        return np.linalg.solve(np.eye(len(r)) - self.model.gamma * p, r)

## Example
model = TabularModel([[[1.0], [0.0]]], [[1.0, 3.0]], gamma=0.5)
print(PolicyEvaluation(model)(np.array([[1.0, 0.0]])))  ## [2.]
```
````

```{attention} Q&A
:class: dropdown
*Iterative vs. exact evaluation?*

- Linear solve: $O(|\mathcal{S}|^3)$ for dense $P^\pi$. Exact up to float precision.
- Iteration: $O(|\mathcal{S}|^2)$ per dense sweep. Stop at a residual tolerance.
- Improvement often needs only approximate values → truncated evaluation (modified policy iteration).
```

&nbsp;

### Policy Iteration
- **What**: Alternate policy evaluation & greedy policy improvement.
- **Why**: Reach an optimal policy through a sequence of policies that never get worse.
- **How**:
    1. Evaluate the curr policy.
    2. In each state, choose the action w/ the largest lookahead value.
    3. Repeat until the policy stops changing.

```{note} Math
:class: dropdown
Process:

$$
\pi'(s)\in\arg\max_a Q^\pi(s,a)
\quad\Longrightarrow\quad V^{\pi'}(s)\geq V^\pi(s)\ \ \forall s
$$

- $\pi'$: Greedy deterministic policy w.r.t. $Q^\pi$.
- Weaker sufficient condition: $\mathbb{E}_{a\sim\pi'(\cdot|s)}Q^\pi(s,a)\geq V^\pi(s)$ for every $s$.
- **Generalized policy iteration**: Evaluation & improvement interact, even if neither step is exact.
```

```{tip} Derivation
:class: dropdown
*Why does greedy improvement never decrease value?*

1. $\pi'$ is greedy w.r.t. $Q^\pi$ → $\mathcal{T}^{\pi'}V^\pi\geq V^\pi$.
2. $\mathcal{T}^{\pi'}$ is monotone: $u\geq v$ → $\mathcal{T}^{\pi'}u\geq\mathcal{T}^{\pi'}v$.
3. Apply $\mathcal{T}^{\pi'}$ $\ell$ times & induct → $(\mathcal{T}^{\pi'})^\ell V^\pi\geq V^\pi$ for every $\ell$.
4. $\mathcal{T}^{\pi'}$ is a $\gamma$-contraction → $(\mathcal{T}^{\pi'})^\ell V^\pi\to V^{\pi'}$ → $V^{\pi'}\geq V^\pi$.
5. Policy stops changing → $\mathcal{T}^*V^\pi=\mathcal{T}^{\pi'}V^\pi=V^\pi$ → $V^\pi=V^*$.
```

````{important} Code
:class: dropdown
```python
import numpy as np

## Reuses TabularModel and PolicyEvaluation from rl/value.md.
class PolicyIteration:
    def __init__(self, model):
        self.model = model
        self.evaluate = PolicyEvaluation(model)

    def __call__(self):
        states, actions = self.model.r.shape
        choice = np.zeros(states, dtype=int)
        while True:
            ## One-hot rows -> deterministic policy matrix
            value = self.evaluate(np.eye(actions)[choice])
            ## Fixed tie-breaking prevents cycling among equal policies.
            improved = self.model(value).argmax(axis=1)
            if np.array_equal(improved, choice):
                return value, choice
            choice = improved

## Example
model = TabularModel([[[1.0], [0.0]]], [[1.0, 3.0]], gamma=0.5)
print(PolicyIteration(model)())  ## (array([3.]), array([1]))
```
````

```{attention} Q&A
:class: dropdown
*When does exact policy iteration terminate?*

- Finite discounted MDP + exact evaluation + consistent tie-breaking → finite termination.
    - ← Finitely many deterministic policies & no policy repeats: each change strictly improves some state or is a one-time tie-break.
- Approximate evaluation or numerical noise → monotonic improvement can fail.

*Why not improve w/ immediate reward alone?*

- Greedy immediate reward ignores delayed consequences.
- The lookahead must keep the evaluated continuation value.
```

&nbsp;

### Value Iteration
- **What**: Repeated Bellman optimality backups.
- **Why**: Full policy evaluation in every iteration is expensive.
- **How**: Apply one optimality backup per sweep → stop at a small residual → extract the greedy policy.

```{note} Math
:class: dropdown
Process:

$$
V(s)\leftarrow\max_a\left[r(s,a)+\gamma\sum_{s'}p(s'|s,a)V(s')\right]
$$

$$
\|V-V^*\|_\infty
\leq\frac{\|\mathcal{T}^*V-V\|_\infty}{1-\gamma},
\qquad
\|V^{\pi_V}-V^*\|_\infty\leq\frac{2\gamma\|V-V^*\|_\infty}{1-\gamma}
$$

- $\|\cdot\|_\infty$: Largest absolute statewise error.
- $\pi_V$: Greedy policy w.r.t. $V$.
- Left bound: residual → certified value error for exact finite discounted backups.
- Right bound: small value error → near-optimal greedy policy. The bound grows as $\gamma\to1$.
```

````{important} Code
:class: dropdown
```python
import numpy as np

## Reuses TabularModel from rl/value.md.
class ValueIteration:
    def __init__(self, model, tolerance):
        self.model, self.tolerance = model, tolerance

    def __call__(self):
        value = np.zeros(len(self.model.r))
        while True:
            ## One optimality backup for every state
            updated = self.model(value).max(axis=1)
            if np.max(np.abs(updated - value)) <= self.tolerance:
                ## Greedy policy from the final values
                return updated, self.model(updated).argmax(axis=1)
            value = updated

## Example
model = TabularModel([[[1.0], [0.0]]], [[1.0, 3.0]], gamma=0.5)
print(ValueIteration(model, tolerance=1e-8)())  ## (array([3.]), array([1]))
```
````

```{attention} Q&A
:class: dropdown
*Synchronous vs. asynchronous updates?*

- Synchronous: every backup reads the frozen previous value vector.
- In-place/asynchronous: backups read fresher values. Convergence needs every state updated infinitely often.

*Finite horizon?*

- Start w/ $V_T=0$. Back up $V_{T-1},\ldots,V_0$ once each.
- The optimal action depends on the remaining time. No fixed-point iteration is necessary.

*Policy iteration vs. value iteration?*

- Policy iteration: expensive iterations, often few improvement rounds.
- Value iteration: cheap sweeps, often many of them.
- Modified policy iteration: a few evaluation sweeps between improvements → interpolates between the two.
```

&nbsp;

## Learning from Returns

### Monte Carlo Prediction
- **What**: Value estimate = average of sampled complete returns.
- **Why**: Estimate value w/o a model & w/o bootstrapping.
- **How**: Finish an episode → compute returns backward → average them at each visited state.

```{note} Math
:class: dropdown
Process:

$$
V(s)\leftarrow V(s)+\frac{1}{N(s)}[G_t-V(s)]
$$

- $N(s)$: Count of return observations for $s$, including this one.
- **First visit**: Use only the first visit to $s$ in each episode.
- **Every visit**: Use all visits. Returns from one episode are correlated.
- Constant $\alpha$ instead of $1/N(s)$ → recency-weighted average, not the sample mean.
```

````{important} Code
:class: dropdown
```python
## Reuses DiscountedReturn from rl/intro.md.
class MonteCarlo:
    def __init__(self, gamma):
        self.returns = DiscountedReturn(gamma)
        self.value, self.count = {}, {}

    def step(self, keys, rewards):
        returns = self.returns(rewards)
        seen = set()
        for key, ret in zip(keys, returns):
            ## First visit only: skip later visits in the same episode
            if key in seen:
                continue
            seen.add(key)
            ## Incremental sample mean
            self.count[key] = self.count.get(key, 0) + 1
            old = self.value.get(key, 0.0)
            self.value[key] = old + (ret - old) / self.count[key]

## Example
mc = MonteCarlo(gamma=1.0)
mc.step(["s", "s"], [1, 2])
print(mc.value["s"])  ## 3.0: first visit, not last visit
```
````

```{attention} Q&A
:class: dropdown
*Why high variance?*

- The target includes all later reward & transition randomness.
- No bootstrap bias. Long trajectories → noisy returns.

*Does MC need a Markov state?*

- No for estimation. Return averages estimate conditional returns under a fixed sampling distribution.
- Control: the state can omit decision-relevant information → the best policy over it can be poor.
- MC is less hurt by non-Markov states than TD ← no bootstrap from successor estimates.

*Can MC handle a rollout cutoff before termination?*

- Pure MC needs the complete return.
- Add an estimated continuation value → bootstrapped target, not pure MC.
```

&nbsp;

#### Monte Carlo Control
- **What**: MC action-value estimation alternated w/ policy improvement.
- **Why**: W/o a model, state values cannot rank actions.
- **How**: Average returns per visited $(s,a)$ → make the policy epsilon-greedy w.r.t. $Q$ → collect another episode.

```{note} Math
:class: dropdown
Process:

$$
Q(s_t,a_t)\leftarrow Q(s_t,a_t)
+\alpha[G_t-Q(s_t,a_t)],\qquad
\pi\leftarrow\epsilon\text{-greedy}(Q)
$$

- Hold the policy fixed during each episode.
- $\alpha=1/N(s,a)$ → first-visit sample average (code below). Constant $\alpha$ → tracks changing values.
- $N(s,a)$: Count of first-visit return observations for $(s,a)$, including this one.
- **Exploring starts**: Start episodes from every $(s,a)$ pair. Needs a resettable environment.
- **Epsilon-soft policy**: Every action keeps probability $\geq\epsilon/|\mathcal{A}|$ → no exploring starts necessary.
```

````{important} Code
:class: dropdown
```python
## Reuses MonteCarlo here and EpsilonGreedy from rl/intro.md.
class MonteCarloControl(MonteCarlo):
    def __init__(self, actions, gamma, epsilon):
        super().__init__(gamma)
        self.actions = actions
        self.select = EpsilonGreedy(epsilon)

    def __call__(self, state):
        ## Keys are (state, action) pairs -> MC averages give Q
        q = [self.value.get((state, a), 0.0) for a in range(self.actions)]
        return self.select(q)

## Example
agent = MonteCarloControl(actions=2, gamma=1.0, epsilon=0.0)
agent.step([("s", 0)], [2.0])
print(agent("s"))  ## 0
```
````

```{attention} Q&A
:class: dropdown
*Why not become fully greedy at once?*

- Unvisited actions keep wrong values forever.
- Fixed $\epsilon$ → keeps exploration, but learns the best epsilon-soft policy, not $\pi^*$.
- **GLIE** (greedy in the limit w/ infinite exploration): standard exploration condition. It alone does not prove convergence for every MC-control variant.
```

&nbsp;

### Importance Sampling
- **What**: Reweight behavior-policy returns to estimate a target-policy expectation.
- **Why**: Logged trajectories often come from a different policy.
- **How**: Multiply action-probability ratios over the relevant trajectory suffix.

```{note} Math
:class: dropdown
$$
w_{t:T-1}
=\prod_{\ell=t}^{T-1}\frac{\pi(a_\ell|s_\ell)}{\mu(a_\ell|s_\ell)}
$$

- $w_{t:T-1}$: Importance weight of the trajectory suffix.
- Same environment under both policies → transition probabilities cancel.
- For $Q^\pi(s_t,a_t)$: $a_t$ is given → product starts at $t+1$.

$$
\hat V_{\mathrm{ordinary}}(s)=\frac1m\sum_{i=1}^m w_iG_i,
\qquad
\hat V_{\mathrm{weighted}}(s)=\frac{\sum_{i=1}^m w_iG_i}{\sum_{i=1}^m w_i}
$$

- $w_i$: Weight of sampled suffix $i$ that starts at $s$.
- $G_i$: Return of sampled suffix $i$.
- Ordinary IS: unbiased w/ coverage & integrable returns. Variance can be very large or infinite.
- Weighted IS: biased for finite $m$, consistent under suitable conditions, usually much lower variance.
```

```{tip} Derivation
:class: dropdown
*Why do only policy ratios remain?*

1. $\mathbb{E}_\pi[G]=\sum_\tau p_\pi(\tau)G(\tau)$.
2. Multiply & divide by $p_\mu(\tau)$ wherever $p_\pi(\tau)>0$.
3. → $\mathbb{E}_\mu[(p_\pi/p_\mu)G]$.
4. Both trajectory probabilities contain the same factors $p(s_{\ell+1}|s_\ell,a_\ell)$ → they cancel → only policy ratios remain.
5. Per-decision IS: weight $r_{\ell+1}$ only by ratios for decisions $t,\ldots,\ell$. Later ratios have expectation 1 → dropping them keeps the estimate unbiased & can ⬇️ variance.
- $\tau$: Trajectory.
- $p_\pi(\tau)$: Probability of $\tau$ under $\pi$.
```

````{important} Code
:class: dropdown
```python
import numpy as np

class ImportanceSampling:
    def __init__(self, weighted):
        self.weighted = weighted

    def __call__(self, returns, weights):
        returns, weights = np.asarray(returns), np.asarray(weights)
        ## Weighted IS normalizes by total weight; ordinary IS by sample count
        denominator = weights.sum() if self.weighted else len(weights)
        if denominator <= 0:
            raise ValueError("No usable target-policy mass in this batch")
        return float(np.dot(weights, returns) / denominator)

## Example
print(ImportanceSampling(True)([2.0, 6.0], [1.0, 3.0]))  ## 5.0
```
````

```{attention} Q&A
:class: dropdown
*What can IS not correct?*

- Missing support: $\mu(a|s)=0$ where $\pi(a|s)>0$.
- Wrong or missing behavior probabilities.
- Unobserved confounders: the logged state omits information that drove the logged decision.

*Why not use full-trajectory ratios everywhere?*

- Products of ratios → variance grows w/ horizon.
- Log-space products prevent overflow/underflow. They do not reduce variance.
- Weight clipping trades bias for lower variance. Weighted IS does not fix missing support.
```

&nbsp;

## Bootstrapped Prediction

### TD Learning
- **Name**: Temporal-Difference Learning
- **What**: Value update toward sampled reward + estimated next value.
- **Why**: MC must wait until the episode ends & ignores known successor values.
- **How**: Compare the curr prediction w/ a one-step target → move the prediction toward it.

```{note} Math
:class: dropdown
Process:

$$
\delta_t=r_{t+1}+\gamma(1-d_t)V(s_{t+1})-V(s_t),
\qquad
V(s_t)\leftarrow V(s_t)+\alpha\delta_t
$$

- **TD(0)**: This one-step state-value update.
- Sample trajectories from $\pi$ → the update estimates $V^\pi$.
```

```{note} Example
:class: dropdown
- $V(s_t)=2$, reward $1$, $V(s_{t+1})=4$, $\gamma=0.5$ → target $3$, error $1$.
- $\alpha=0.1$ → updated value $2.1$.
- Terminal transition → target $1$ → updated value $1.9$.
```

````{important} Code
:class: dropdown
```python
import numpy as np

class TDLearning:
    def __init__(self, states, alpha, gamma):
        self.value = np.zeros(states)
        self.alpha, self.gamma = alpha, gamma

    def step(self, s, reward, next_s, terminated):
        ## True termination -> no bootstrap; a time-limit cutoff still bootstraps
        continuation = 0.0 if terminated else self.value[next_s]
        delta = reward + self.gamma * continuation - self.value[s]
        self.value[s] += self.alpha * delta
        return delta

## Example
td = TDLearning(states=2, alpha=0.1, gamma=0.5)
td.value[:] = [2.0, 4.0]
td.step(0, 1.0, 1, False)
print(td.value[0])  ## 2.1
```
````

```{attention} Q&A
:class: dropdown
*Why is the TD target biased?*

- $V\neq V^\pi$ → the bootstrap differs from the true continuation value.
- Exact $V^\pi$ → the target is conditionally unbiased for $V^\pi(s_t)$.
- Cost of the bias: much lower variance than the full return.

*When does tabular TD(0) converge?*

- Fixed policy, finite MDP, bounded rewards, every state visited infinitely often.
- Per-state step sizes w/ $\sum\alpha=\infty$ & $\sum\alpha^2<\infty$.
- Constant step size → tracks change, but does not converge exactly.

*TD vs. DP?*

- Both bootstrap from a value estimate.
- DP takes an expectation w/ a model. TD uses one sampled transition & no model.
```

&nbsp;

### N-Step Returns
- **What**: $h$ sampled rewards + a value bootstrap.
- **Why**: One-step TD trusts an imperfect estimate. MC waits for noisy complete returns.
- **How**: Wait $h$ steps → sum the discounted rewards → bootstrap unless the episode truly ends.

```{note} Math
:class: dropdown
$$
G_t^{(h)}
=\sum_{\ell=0}^{h-1}\gamma^\ell r_{t+\ell+1}
+\gamma^h V(s_{t+h}),\qquad t+h<T
$$

- $h$: Backup length; positive integer.
- $G_t^{(h)}$: $h$-step return target.
- Episode ends before $t+h$ → sum rewards through $r_T$ & omit the bootstrap.
- Time-limit cutoff before $T$ → bootstrap from the last observed state.
```

````{important} Code
:class: dropdown
```python
## Reuses DiscountedReturn from rl/intro.md.
class NStepReturn:
    def __init__(self, gamma):
        self.returns = DiscountedReturn(gamma)

    def __call__(self, rewards, next_value, terminated):
        if len(rewards) == 0:
            raise ValueError("A backup needs at least one transition")
        ## Discounted rewards + gamma^h * bootstrap, unless truly terminated
        bootstrap = 0.0 if terminated else next_value
        return self.returns(rewards, bootstrap)[0]

## Example
print(NStepReturn(0.5)([1, 2], next_value=4, terminated=False))  ## 3.0
```
````

```{dropdown} Table: Backup Families
| Method | Environment model | Sampled reward depth | Bootstrap |
|:--|:--|:--|:--|
| DP | Required | None; full expectation | Yes |
| TD(0) | No | One step | Yes |
| $h$-step TD | No | $h$ steps or termination | At nonterminal endpoint |
| Monte Carlo | No | Complete episode | No |
```

```{attention} Q&A
:class: dropdown
*What changes as $h$ increases?*

- Bootstrap error is scaled by $\gamma^h$ → less bias from $V$.
- More sampled randomness → more variance.
- Longer wait before each update.
- This bias/variance trend is intuition, not a monotonic theorem. The best $h$ is task-dependent.

*Off-policy multi-step backups?*

- Intermediate actions follow $\mu$, not $\pi$.
- A final $\max$ does not make the intermediate path greedy.
- Correct w/ IS ratios or trace cutting, or accept the bias (e.g., Rainbow's uncorrected $h$-step targets).
```

&nbsp;

### Eligibility Traces
- **What**: Decaying memory of recently visited states that receives each TD error.
- **Why**: Spread delayed credit online w/o storing $h$-step returns for every $h$.
- **How**: Mark the curr state → apply the TD error to all marked states in proportion to their trace → decay all traces.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\lambda$: Trace-decay parameter in $[0,1]$.
- Misc:
    - $e_t(s)$: Accumulating eligibility of state $s$.

Process:

$$
G_t^\lambda
=(1-\lambda)\sum_{h=1}^{T-t-1}\lambda^{h-1}G_t^{(h)}
+\lambda^{T-t-1}G_t
$$

- **Forward view**: $G_t^\lambda$ = episodic $\lambda$-return.
- $G_t^{(h)}$: [$h$-step return](#n-step-returns) w/ fixed values.
- Last term: all remaining weight goes to the full return. $\lambda=1$ → MC. $\lambda=0$ → TD(0).

$$
\begin{align*}
e_t(s)&=\gamma\lambda e_{t-1}(s)+\mathbf{1}[s=s_t],\\
\delta_t&=r_{t+1}+\gamma(1-d_t)V(s_{t+1})-V(s_t),\\
V(s)&\leftarrow V(s)+\alpha\delta_t e_t(s)\quad\forall s.
\end{align*}
$$

- **Backward view**: Online trace updates above.
- Reset traces at each episode boundary.
- **Accumulating trace**: Adds $1$ per visit. **Replacing trace**: Sets the visited state's trace to $1$.
```

````{important} Code
:class: dropdown
```python
import numpy as np

## Reuses TDLearning from rl/value.md.
class TDLambda(TDLearning):
    def __init__(self, states, alpha, gamma, lam):
        super().__init__(states, alpha, gamma)
        self.lam = lam
        self.trace = np.zeros(states)

    def step(self, s, reward, next_s, terminated, episode_end):
        continuation = 0.0 if terminated else self.value[next_s]
        delta = reward + self.gamma * continuation - self.value[s]
        ## Decay old credit, then mark the curr state (accumulating trace)
        self.trace *= self.gamma * self.lam
        self.trace[s] += 1
        ## One TD error updates every state in proportion to its trace
        self.value += self.alpha * delta * self.trace
        if terminated or episode_end:
            self.trace.fill(0)
        return delta

## Example
td = TDLambda(2, alpha=1.0, gamma=1.0, lam=0.5)
td.step(0, 0, 1, False, False)
td.step(1, 2, None, True, True)
print(td.value)  ## [1. 2.]
```
````

```{attention} Q&A
:class: dropdown
*Are the forward & backward views identical?*

- Offline updates (values frozen during the episode, applied at the end) → exact equivalence.
- Online TD($\lambda$) changes values during the episode → only approximately equivalent.
- True online TD($\lambda$) gives exact equivalence to an online forward view.

*Can plain traces be used off-policy?*

- Not in general. Exploratory actions break the continuation being credited.
- Use off-policy corrections or cut traces after non-greedy actions (e.g., Watkins's Q($\lambda$)).
```

&nbsp;

## Sampled Control

### SARSA
- **Name**: State–Action–Reward–State–Action
- **What**: On-policy TD control that bootstraps from the next action actually selected.
- **Why**: Learn the value of the exploratory policy that acts.
- **How**: Select the next action → bootstrap from its $Q$ → update the curr pair.

```{note} Math
:class: dropdown
Process:

$$
\delta_t=r_{t+1}+\gamma(1-d_t)Q(s_{t+1},a_{t+1})-Q(s_t,a_t),
\qquad
Q(s_t,a_t)\leftarrow Q(s_t,a_t)+\alpha\delta_t
$$

- Draw $a_{t+1}$ from the policy being evaluated & improved.
- Execute that same $a_{t+1}$ next. Do not resample it after the update.
```

````{important} Code
:class: dropdown
```python
import numpy as np

class SARSA:
    def __init__(self, states, actions, alpha, gamma):
        self.q = np.zeros((states, actions))
        self.alpha, self.gamma = alpha, gamma

    def step(self, s, a, reward, next_s, next_a, terminated):
        ## Bootstrap from the action the policy will actually take
        continuation = 0.0 if terminated else self.q[next_s, next_a]
        delta = reward + self.gamma * continuation - self.q[s, a]
        self.q[s, a] += self.alpha * delta
        return delta

## Example
agent = SARSA(2, 2, alpha=1.0, gamma=0.5)
agent.q[1] = [2, 8]
agent.step(0, 0, 1, 1, 0, False)
print(agent.q[0, 0])  ## 2.0, using the chosen action's value 2
```
````

```{attention} Q&A
:class: dropdown
*Why does SARSA take the safer cliff-walking route?*

- Persistent exploration sometimes steps off the cliff.
- SARSA values include these future exploratory mistakes → path near the edge looks worse.
- This is sensitivity to the behavior policy, not a risk-sensitive objective.

*Does SARSA converge to $Q^*$?*

- Yes, under: finite MDP, bounded rewards, every pair visited infinitely often, decaying per-pair steps & GLIE.
- Fixed $\epsilon$ → learns epsilon-greedy values, not $Q^*$.
```

&nbsp;

#### Expected SARSA
- **Name**: Expected State–Action–Reward–State–Action
- **What**: TD control that averages next-action values under a target policy.
- **Why**: The sampled next action adds variance.
- **How**: Replace $Q(s',a')$ w/ its expectation under $\pi$.

```{note} Math
:class: dropdown
Process:

$$
\delta_t=r_{t+1}+\gamma(1-d_t)
\sum_{a'}\pi(a'|s_{t+1})Q(s_{t+1},a')-Q(s_t,a_t)
$$

- $\pi=\mu$ → on-policy. $\pi\neq\mu$ → off-policy, w/o importance ratios for one step.
- Greedy $\pi$ → one-step Q-learning.
```

````{important} Code
:class: dropdown
```python
## Reuses SARSA from rl/value.md.
class ExpectedSARSA(SARSA):
    def step(self, s, a, reward, next_s, next_policy, terminated):
        ## Policy-weighted next value replaces the sampled next action
        continuation = (
            0.0 if terminated else self.q[next_s] @ next_policy
        )
        delta = reward + self.gamma * continuation - self.q[s, a]
        self.q[s, a] += self.alpha * delta
        return delta

## Example
agent = ExpectedSARSA(2, 2, alpha=1.0, gamma=0.5)
agent.q[1] = [2, 8]
agent.step(0, 0, 1, 1, [0.75, 0.25], False)
print(agent.q[0, 0])  ## 2.75
```
````

```{attention} Q&A
:class: dropdown
*What variance remains?*

- Reward & transition randomness.
- The expectation removes only the next-action sampling variance.

*Cost?*

- One sum over $|\mathcal{A}|$ per update. ❌ for large or continuous action spaces w/o a closed form.
```

&nbsp;

### Q-Learning
- **What**: Off-policy TD control w/ a greedy next-action target.
- **Why**: Learn optimal values while the behavior policy still explores.
- **How**: Take any action w/ adequate coverage → bootstrap from the largest next $Q$ → act greedily w.r.t. the learned $Q$.

```{note} Math
:class: dropdown
Process:

$$
\delta_t=r_{t+1}+\gamma(1-d_t)\max_{a'}Q(s_{t+1},a')-Q(s_t,a_t),
\qquad
Q(s_t,a_t)\leftarrow Q(s_t,a_t)+\alpha\delta_t
$$

$$
\sum_{\text{visits}}\alpha(s,a)=\infty,\qquad
\sum_{\text{visits}}\alpha(s,a)^2<\infty
$$

- Per-pair step-size conditions + every pair visited infinitely often + bounded rewards + finite discounted MDP → tabular convergence to $Q^*$.
```

````{important} Code
:class: dropdown
```python
## Reuses SARSA's table initialization from rl/value.md.
class QLearning(SARSA):
    def step(self, s, a, reward, next_s, terminated):
        ## Greedy target, independent of the next behavior action
        continuation = 0.0 if terminated else self.q[next_s].max()
        delta = reward + self.gamma * continuation - self.q[s, a]
        self.q[s, a] += self.alpha * delta
        return delta

## Example
agent = QLearning(2, 2, alpha=1.0, gamma=0.5)
agent.q[1] = [2, 8]
agent.step(0, 0, 1, 1, False)
print(agent.q[0, 0])  ## 5.0, regardless of the behavior's next action
```
````

```{dropdown} Table: One-Step Control Targets
| Method | Continuation | Target policy | Next-action sampling noise |
|:--|:--|:--|:--|
| SARSA | $Q(s',a')$ | Same as behavior | Yes |
| Expected SARSA | $\sum_{a'}\pi(a'\mid s')Q(s',a')$ | Specified $\pi$ | No |
| Q-learning | $\max_{a'}Q(s',a')$ | Greedy | No |
```

```{attention} Q&A
:class: dropdown
*Why is Q-learning off-policy, even w/ epsilon-greedy behavior?*

- The target assumes greedy continuation. Epsilon-greedy behavior is a different policy.
- The sampled action $a_t$ need not maximize $Q$.

*Does convergence mean safe training?*

- No. Exploration can cause large losses before $Q$ is accurate.
- Asymptotic convergence says nothing about a finite interaction budget.

*Why can the max overestimate?*

- Unbiased noisy estimates → $\mathbb{E}[\max_a\hat Q(a)]\geq\max_a\mathbb{E}[\hat Q(a)]$ (Jensen; $\max$ is convex).
- The max selects positive errors. This is **maximization bias**.
- Not every estimate is too high; the bias is a tendency.
```

&nbsp;

#### Double Q-Learning
- **What**: Q-learning w/ 2 estimators: one selects the next action, the other evaluates it. {cite:p}`hasselt2010double`
- **Why**: One noisy estimate for both selection & evaluation → maximization bias.
- **How**: Update one table w/ its own argmax & the other table's value. Choose the updated table at random.

```{note} Math
:class: dropdown
Process:

$$
a^*=\arg\max_{a'}Q_1(s_{t+1},a'),\qquad
Q_1(s_t,a_t)\leftarrow Q_1(s_t,a_t)
+\alpha[r_{t+1}+\gamma(1-d_t)Q_2(s_{t+1},a^*)-Q_1(s_t,a_t)]
$$

- $Q_1$: First action-value estimator.
- $Q_2$: Second action-value estimator.
- $a^*$: Action selected by the estimator being updated.
- Swap roles to update $Q_2$. Behavior: e.g., epsilon-greedy w.r.t. $Q_1+Q_2$.
```

````{important} Code
:class: dropdown
```python
import numpy as np

class DoubleQLearning:
    def __init__(self, states, actions, alpha, gamma, seed=0):
        self.q = np.zeros((2, states, actions))
        self.alpha, self.gamma = alpha, gamma
        self.rng = np.random.default_rng(seed)

    def step(self, s, a, reward, next_s, terminated):
        ## Pick which table to update; the other one evaluates
        update = int(self.rng.integers(2))
        own, other = self.q[update], self.q[1 - update]
        continuation = 0.0
        if not terminated:
            ## Select w/ own table (random ties), evaluate w/ the other table
            ties = np.flatnonzero(own[next_s] == own[next_s].max())
            best = self.rng.choice(ties)
            continuation = other[next_s, best]
        delta = reward + self.gamma * continuation - own[s, a]
        own[s, a] += self.alpha * delta
        return delta

## Example
agent = DoubleQLearning(1, 2, alpha=1.0, gamma=0.9)
agent.step(0, 1, 3, None, True)
print(agent.q[:, 0, 1].sum())  ## 3.0
```
````

```{attention} Q&A
:class: dropdown
*Does double estimation give unbiased values?*

- No. It removes the selection/evaluation coupling only.
- It can underestimate.
- Correlated estimators → weaker effect.

*Cost?*

- 2× memory. Each table sees about half the updates → slower learning per table.
```

&nbsp;

## Function Approximation

### Semi-Gradient Value Learning
- **What**: Fit a parametric value function; treat the bootstrap target as a constant.
- **Why**: Tables cannot generalize to unseen states or scale to large state spaces.
- **How**: Build a target from reward + next-state estimate → differentiate only the curr prediction.

```{note} Math
:class: dropdown
Model:

$$
V_\phi(s)=\phi^\top\mathbf{x}(s)
$$

- $\mathbf{x}(s)$: State-feature vector. A neural network can replace the linear model.

Training:

$$
\begin{align*}
y_t&=r_{t+1}+\gamma(1-d_t)V_\phi(s_{t+1}),\\
\mathcal{L}(\phi)&=\tfrac12[\operatorname{stopgrad}(y_t)-V_\phi(s_t)]^2,\\
\phi&\leftarrow\phi+\alpha[y_t-V_\phi(s_t)]\nabla_\phi V_\phi(s_t).
\end{align*}
$$

- $y_t$: Bootstrap regression target.
- $\operatorname{stopgrad}$: Keep the value; block its gradient.
- The update is not the gradient of any fixed objective → "semi-gradient".
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class SemiGradientTD(nn.Module):
    def __init__(self, features, gamma):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(features))
        self.gamma = gamma

    def forward(self, x):
        return x @ self.weight

    def loss(self, x, reward, next_x, terminated):
        ## no_grad = stopgrad: the target is a constant label
        with torch.no_grad():
            target = reward + self.gamma * (~terminated).float() * self(next_x)
        return 0.5 * (self(x) - target).square().mean()

## Example
value = SemiGradientTD(features=2, gamma=0.9)
loss = value.loss(torch.eye(2), torch.ones(2),
                  torch.zeros(2, 2), torch.tensor([True, True]))
loss.backward()
print(value.weight.grad)  ## tensor([-0.5000, -0.5000])
```
````

```{attention} Q&A
:class: dropdown
*What is lost vs. a table?*

- An update at one state changes other states through shared params.
- The function class may not contain $V^\pi$ or $Q^*$.
- 3 separate error sources: optimization, approximation & distribution mismatch.

*Why not backpropagate through the target?*

- Full gradient through one sampled target → naive residual gradient: minimizes squared TD error, not Bellman error → wrong values in stochastic environments.
- An unbiased gradient of the squared expected Bellman residual needs 2 independent next-state samples from the same $(s,a)$: the **double-sampling problem**.

*Does linear on-policy semi-gradient TD converge?*

- Yes, under standard conditions, to the TD fixed point.
- That fixed point is not the best linear fit to $V^\pi$.
- Bound: $\overline{\mathrm{VE}}(\phi_{\mathrm{TD}})\leq\frac{1}{1-\gamma}\min_\phi\overline{\mathrm{VE}}(\phi)$.
- $\overline{\mathrm{VE}}$: Mean squared value error, weighted by the on-policy state distribution.
```

&nbsp;

### Deadly Triad
- **What**: Possible divergence from function approximation + bootstrapping + off-policy training.
- **Why**: Each element is useful alone. The combination explains divergence that no single element causes.
- **How**: Errors feed back.
    - Shared params → one update changes many values.
    - Bootstrapping → those wrong values become new targets.
    - Off-policy update distribution → projected Bellman backup need not be a contraction → errors can grow, even if every state gets updates.

```{attention} Q&A
:class: dropdown
*Does the combination always diverge?*

- No. It permits divergence; it does not force it.
- Linear off-policy TD can diverge (e.g., Baird's counterexample).
- On-policy linear TD converges under standard assumptions. Nonlinear control does not inherit this result.

*How to avoid it?*

- Remove one element: tabular values, MC targets, or on-policy sampling.
- Linear off-policy prediction: gradient-TD methods (true gradient of the projected Bellman error) converge w/ all 3 elements.

*Do replay & target networks remove the triad?*

- No. They ⬇️ sample correlation & target drift. They do not prove convergence.
- Monitor value magnitudes, TD errors, state coverage & actual return separately.
```

&nbsp;

### DQN
- **Name**: Deep Q-Network {cite:p}`mnih2015human`
- **What**: Neural Q-learning w/ experience replay & a delayed target network.
- **Why**: Neural Q-learning is unstable.
    - Consecutive transitions are correlated → correlated updates → inefficient, unstable SGD.
    - Each update moves the bootstrap target → regression toward a moving label.
- **How**:
    1. Act epsilon-greedily; store transitions in $\mathcal{D}$.
    2. Sample minibatches from $\mathcal{D}$; regress $Q(s_t,a_t)$ toward targets from the frozen target network.
    3. Copy online weights into the target network every fixed number of updates.

```{note} Math
:class: dropdown
Model:

$$
Q_\phi(s,\cdot)\in\mathbb{R}^{|\mathcal{A}|}
$$

Training:

$$
y_t=r_{t+1}+\gamma(1-d_t)\max_{a'}Q_{\bar\phi}(s_{t+1},a'),
\qquad
\mathcal{L}(\phi)=\mathbb{E}_{\mathcal{D}}
\left[\ell_\kappa(Q_\phi(s_t,a_t)-\operatorname{stopgrad}(y_t))\right]
$$

- $y_t$: Q-learning target from the frozen target network.
- $\ell_\kappa$: Huber loss. Squared for $|\cdot|\leq\kappa$, linear above.
- $\kappa$: Positive Huber threshold.
- One forward pass outputs all action values. Only the taken action gets a loss.
```

````{important} Code
:class: dropdown
```python
from copy import deepcopy
import random
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN:
    def __init__(self, q, gamma, lr, capacity, sync_every):
        ## Target network: frozen copy that supplies labels
        self.q, self.target = q, deepcopy(q).requires_grad_(False).eval()
        self.gamma, self.sync_every = gamma, sync_every
        self.optimizer = torch.optim.Adam(q.parameters(), lr=lr)
        self.replay = deque(maxlen=capacity)
        self.updates = 0

    @torch.no_grad()
    def bootstrap(self, next_s):
        return self.target(next_s).max(dim=1).values

    def step(self, batch_size):
        if len(self.replay) < batch_size:
            raise ValueError("Collect a full replay batch before updating")
        ## Uniform replay sample breaks temporal correlation
        s, a, r, next_s, terminal = zip(*random.sample(self.replay, batch_size))
        s, next_s = torch.stack(s), torch.stack(next_s)
        device = s.device
        a = torch.tensor(a, dtype=torch.long, device=device)
        r = torch.tensor(r, dtype=s.dtype, device=device)
        terminal = torch.tensor(terminal, dtype=torch.bool, device=device)
        with torch.no_grad():
            target = r.clone()
            active = ~terminal
            ## Never evaluate a terminal placeholder as a next state.
            if active.any():
                target[active] += self.gamma * self.bootstrap(next_s[active])
        ## Regress only the taken action's Q
        prediction = self.q(s).gather(1, a[:, None]).squeeze(1)
        loss = F.smooth_l1_loss(prediction, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.updates += 1
        ## Periodic hard copy -> labels stay fixed between syncs
        if self.updates % self.sync_every == 0:
            self.target.load_state_dict(self.q.state_dict())
        return loss.item()

## Example
q = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))
agent = DQN(q, gamma=0.9, lr=1e-3, capacity=8, sync_every=2)
agent.replay.append((torch.zeros(2), 0, 1.0, torch.ones(2), False))
agent.replay.append((torch.ones(2), 1, 2.0, torch.zeros(2), True))
print(isinstance(agent.step(batch_size=2), float))  ## True
```
````

```{attention} Q&A
:class: dropdown
*Why 2 networks?*

- Online $Q$ changes every step. Target $Q$ gives labels that stay fixed between syncs.
- Slower sync → more stable labels, but slower value propagation.
- A target network alone is not [Double DQN](#double-dqn).

*Why replay?*

- Reuse expensive transitions → ⬆️ sample efficiency.
- Random minibatches → ⬇️ correlation between consecutive samples.
- Store observation, action, reward, next observation & true termination flag.
- Old transitions come from old policies → replay requires an off-policy method (e.g., Q-learning). It does not guarantee coverage.

*Atari-specific tricks in the original DQN?*

- Stack the last 4 frames → partial observability ⬇️ (e.g., velocity).
- Clip rewards to $[-1,1]$ → one LR across games. Cost: the agent cannot tell large rewards from small ones.
- Anneal $\epsilon$ from 1 to a small final value.

*Why not softmax the outputs?*

- Q values are unbounded expected returns. Softmax removes their scale.
- Discrete actions → explicit $\max$. Continuous actions need an inner optimization or an actor (e.g., [DDPG](policy.md#ddpg)).
```

&nbsp;

#### Double DQN
- **Name**: Double Deep Q-Network {cite:p}`hasselt2016deep`
- **What**: DQN that selects the next action w/ the online network & evaluates it w/ the target network.
- **Why**: DQN's target network still selects & evaluates w/ the same estimates → overestimation.
- **How**: Keep replay & target sync. Change only the target: online argmax, target value.

```{note} Math
:class: dropdown
Process:

$$
y_t=r_{t+1}+\gamma(1-d_t)
Q_{\bar\phi}\!\left(s_{t+1},\arg\max_{a'}Q_\phi(s_{t+1},a')\right)
$$

- $y_t$: Double DQN regression target.
- Online & target networks are correlated → bias ⬇️, not removed.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

## Reuses DQN from rl/value.md.
class DoubleDQN(DQN):
    @torch.no_grad()
    def bootstrap(self, next_s):
        ## Online net selects, target net evaluates
        action = self.q(next_s).argmax(dim=1, keepdim=True)
        return self.target(next_s).gather(1, action).squeeze(1)

## Example
agent = DoubleDQN(nn.Linear(2, 2), gamma=0.9, lr=1e-3,
                  capacity=8, sync_every=2)
print(agent.bootstrap(torch.zeros(3, 2)).shape)  ## torch.Size([3])
```
````

```{attention} Q&A
:class: dropdown
*Why not train a second independent network as in Double Q-learning?*

- The target network already exists → no extra network, no extra params.
- Cost: the 2 networks are not independent.

*Double vs. dueling?*

- Double changes the bootstrap target.
- Dueling changes the network architecture.
- They are compatible.
```

&nbsp;

#### Dueling Networks
- **What**: Q network w/ a state-value stream & a centered action-advantage stream. {cite:p}`wang2016dueling`
- **Why**: In many states, the action barely matters. One shared value is easier to learn than $|\mathcal{A}|$ separate ones.
- **How**: Shared features → scalar value head + per-action score head → center the scores → add.

```{note} Math
:class: dropdown
Forward:

$$
Q_\phi(s,a)=v_\phi(s)+u_\phi(s,a)
-\frac{1}{|\mathcal{A}|}\sum_{a'}u_\phi(s,a')
$$

- $v_\phi$: Scalar value stream.
- $u_\phi$: Raw action-score stream.
- Centering removes the ambiguity ← adding a constant to $v_\phi$ & subtracting it from $u_\phi$ gives the same $Q$.
- $v_\phi(s)$ = mean of $Q_\phi(s,\cdot)$ over actions; it is $V^\pi$ only for a uniform policy.
- Paper also tests max-centering: then $v_\phi(s)=\max_aQ_\phi(s,a)$. Mean-centering was more stable.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class DuelingQ(nn.Module):
    def __init__(self, features, hidden, actions):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(features, hidden), nn.ReLU())
        self.value = nn.Linear(hidden, 1)
        self.advantage = nn.Linear(hidden, actions)

    def forward(self, x):
        h = self.body(x)
        advantage = self.advantage(h)
        ## Mean-center scores -> v & u are identifiable
        return self.value(h) + advantage - advantage.mean(dim=-1, keepdim=True)

## Example
print(DuelingQ(2, 8, 3)(torch.zeros(2, 2)).shape)  ## torch.Size([2, 3])
```
````

```{attention} Q&A
:class: dropdown
*Does the advantage head output zero-mean values?*

- Raw outputs: no. The centered term: yes.
- Uniform-mean centering differs from the $\pi$-weighted identity $\mathbb{E}_{a\sim\pi}A^\pi(s,a)=0$.

*What does each update gain?*

- Every update trains $v_\phi$, whatever action was taken → shared state value learns faster.
- No change to the target or the loss.
```

&nbsp;

#### Prioritized Replay
- **What**: Replay that samples transitions w/ large TD errors more often. {cite:p}`schaul2015prioritized`
- **Why**: Uniform replay spends equal compute on transitions w/ unequal learning signal.
- **How**: Sample by priority → weight each loss to correct the sampling bias → update priorities from new TD errors.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\eta$: Prioritization exponent; paper's $\alpha$.
    - $\zeta$: Importance-correction exponent; paper's $\beta$.
    - $\varepsilon_p$: Positive priority floor.

Process:

$$
p_i=|\delta_i|+\varepsilon_p,\qquad
P(i)=\frac{p_i^\eta}{\sum_jp_j^\eta},\qquad
\tilde w_i=\left(|\mathcal{D}|\,P(i)\right)^{-\zeta},\qquad
w_i=\frac{\tilde w_i}{\max_j\tilde w_j}
$$

- $p_i$: Priority of replay item $i$. **Proportional** variant.
- $\delta_i$: Latest TD error of item $i$.
- $P(i)$: Sampling probability of item $i$. $\eta=0$ → uniform.
- $\tilde w_i$: IS weight. $\zeta=1$ → full correction to the uniform-replay expectation.
- $w_i$: Normalized loss weight; $\leq1$ → weights only scale updates down.
- $j$: Replay-item index here, not a feature index.
- **Rank-based** variant: $p_i=1/\operatorname{rank}(i)$, rank by $|\delta_i|$.
- New transitions get the max priority so far → high chance of early replay.
- Anneal $\zeta$ linearly from $\zeta_0$ to $1$ → full correction by the end of training.
- Paper defaults: $\eta=0.6$, $\zeta_0=0.4$ (proportional); $\eta=0.7$, $\zeta_0=0.5$ (rank-based).
```

````{important} Code
:class: dropdown
```python
import numpy as np

class PrioritizedSampler:
    def __init__(self, exponent, correction, floor, seed=0):
        self.exponent, self.correction, self.floor = exponent, correction, floor
        self.rng = np.random.default_rng(seed)

    def __call__(self, td_errors, batch_size):
        ## Proportional priority, sharpened by the exponent
        priority = (np.abs(td_errors) + self.floor) ** self.exponent
        probability = priority / priority.sum()
        index = self.rng.choice(len(priority), batch_size, p=probability)
        ## IS weights undo the non-uniform sampling
        weight = (len(priority) * probability[index]) ** (-self.correction)
        ## Divide by the buffer's largest weight -> weights <= 1
        max_weight = (len(priority) * probability.min()) ** (-self.correction)
        return index, weight / max_weight

## Example
sampler = PrioritizedSampler(exponent=0.0, correction=1.0, floor=1e-6)
index, weight = sampler(np.array([1.0, 5.0]), batch_size=2)
print(weight)  ## [1. 1.]
```
````

```{attention} Q&A
:class: dropdown
*How does PER plug into DQN?*

- Replace uniform indices w/ prioritized samples.
- Compute per-item Huber losses, multiply by $w_i$, then average.
- Write back $|\delta_i|$ of sampled items as new priorities. Keep $\varepsilon_p>0$ → every item keeps a chance.
- Use a sum-tree for $O(\log|\mathcal{D}|)$ sampling & updates. The reference sampler scans the buffer in $O(|\mathcal{D}|)$.

*Proportional vs. rank-based?*

- Proportional: uses the TD-error magnitude directly → sensitive to outliers.
- Rank-based: uses only the order → expected to be robust to outliers & error scale.
- On Atari, the 2 variants usually give similar results.

*Does PER fix off-policy learning?*

- No. $w_i$ corrects toward uniform replay, not toward the curr policy's state distribution.
- Large TD errors can be irreducible noise in stochastic environments.
- Stale priorities can misdirect updates.
```

&nbsp;

#### C51
- **Name**: Categorical DQN w/ 51 atoms {cite:p}`bellemare2017distributional`
- **What**: DQN that predicts a categorical return distribution for each action.
- **Why**: Expected return discards the rest of the return distribution.
    - The distribution gives a richer training signal.
    - Better control performance on Atari; the reason for the gain is not settled.
- **How**:
    1. Fix $N$ return atoms on $[V_{\min},V_{\max}]$.
    2. Output one softmax over atoms for each action.
    3. Shift & shrink the next-state atoms by reward & $\gamma$ → project them back onto the fixed atoms.
    4. Minimize cross-entropy to the projected target.
    5. Act greedily w.r.t. the distribution mean.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $N$: #atoms; $51$ in the paper.
    - $V_{\min}$: Lowest atom; $-10$ on Atari.
    - $V_{\max}$: Highest atom; $10$ on Atari.
- Misc:
    - $\ell$: Atom index, $0\leq\ell\leq N-1$.
    - $\rho_\ell(s,a)$: Online-net probability of atom $\ell$.
    - $\bar\rho_\ell(s,a)$: Target-net probability of atom $\ell$.

Model:

$$
z_\ell=V_{\min}+\ell\,\Delta z,\qquad
\Delta z=\frac{V_{\max}-V_{\min}}{N-1},\qquad
Q_\phi(s,a)=\sum_{\ell}z_\ell\,\rho_\ell(s,a)
$$

- $z_\ell$: Fixed return atom.
- $\Delta z$: Atom spacing.

Training:

$$
\begin{align*}
a^*&=\arg\max_{a'}\sum_\ell z_\ell\,\bar\rho_\ell(s_{t+1},a'),\\
\hat z_\ell&=\operatorname{clip}\!\left(r_{t+1}+\gamma(1-d_t)z_\ell,\;V_{\min},\;V_{\max}\right),\\
\tilde\rho_{\ell'}&=\sum_\ell\left[1-\frac{|\hat z_\ell-z_{\ell'}|}{\Delta z}\right]_0^1\bar\rho_\ell(s_{t+1},a^*),\\
\mathcal{L}(\phi)&=-\sum_{\ell'}\tilde\rho_{\ell'}\log\rho_{\ell'}(s_t,a_t).
\end{align*}
$$

- $\hat z_\ell$: Shifted, shrunk & clipped atom.
- $\tilde\rho_{\ell'}$: Projected target probability of atom $\ell'$.
- $[\cdot]_0^1$: Clip to $[0,1]$ → each $\hat z_\ell$ splits its mass between its 2 nearest atoms by linear interpolation.
- Cross-entropy = $\mathrm{KL}(\tilde\rho\,\|\,\rho)$ + a constant in $\phi$.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class C51(nn.Module):
    def __init__(self, features, actions, atoms=51, v_min=-10.0, v_max=10.0):
        super().__init__()
        self.actions, self.atoms = actions, atoms
        self.net = nn.Linear(features, actions * atoms)
        self.register_buffer("z", torch.linspace(v_min, v_max, atoms))

    def forward(self, x):
        ## One log-softmax over atoms per action: [B, A, N]
        logits = self.net(x).view(-1, self.actions, self.atoms)
        return logits.log_softmax(dim=-1)

    def q(self, x):
        ## Greedy actions use the distribution mean
        return (self(x).exp() * self.z).sum(dim=-1)

    @torch.no_grad()
    def target(self, reward, next_x, terminated, gamma):
        ## Call on the target network in practice
        z, dz = self.z, self.z[1] - self.z[0]
        best = self.q(next_x).argmax(dim=1)
        rows = torch.arange(len(best), device=best.device)
        p = self(next_x).exp()[rows, best]  ## [B, N]
        ## Shift & shrink every atom, then clip to the support
        tz = reward[:, None] + gamma * (~terminated).float()[:, None] * z
        b = (tz.clamp(float(z[0]), float(z[-1])) - z[0]) / dz
        lo, hi = b.floor().long(), b.ceil().long()
        proj = torch.zeros_like(p)
        ## Split mass between the 2 neighbors; exact hit (lo == hi) keeps all
        proj.scatter_add_(1, lo, p * (hi - b + (lo == hi).float()))
        proj.scatter_add_(1, hi, p * (b - lo))
        return proj

    def loss(self, x, a, target_probs):
        log_p = self(x)[torch.arange(len(a), device=a.device), a]
        return -(target_probs * log_p).sum(dim=-1).mean()

## Example
net = C51(features=2, actions=2, atoms=3, v_min=-1.0, v_max=1.0)
proj = net.target(torch.tensor([0.5]), torch.zeros(1, 2),
                  torch.tensor([True]), gamma=0.9)
print(proj)  ## tensor([[0.0000, 0.5000, 0.5000]]): 0.5 splits between z=0 & z=1
print(net.loss(torch.zeros(1, 2), torch.tensor([0]), proj).shape)  ## torch.Size([])
```
````

```{attention} Q&A
:class: dropdown
*Why project?*

- $r+\gamma z_\ell$ moves atoms off the fixed support.
- Cross-entropy needs both distributions on the same support → project.

*Why cross-entropy & not the Wasserstein metric?*

- Distributional evaluation operator: a $\gamma$-contraction in the maximal Wasserstein metric. It is not a contraction in KL or total variation.
- Wasserstein loss from sampled transitions has biased gradients → paper uses projected cross-entropy.
- Result: a gap between the theory & the algorithm.

*Does distributional control converge?*

- Evaluation: yes, in the maximal Wasserstein metric.
- Control: the distributional optimality operator is not a contraction in general. The mean still contracts like $\mathcal{T}^*$.

*How to choose $[V_{\min},V_{\max}]$?*

- Returns outside the range are clipped → biased mean.
- Atari works w/ $\pm10$ because rewards are clipped to $[-1,1]$.
- Wider range w/ fixed $N$ → coarser atoms.

*Why does the distribution help if the policy uses only the mean?*

- Not settled. The paper reports large Atari gains & suggests a richer learning signal.
```

&nbsp;

#### Rainbow
- **What**: DQN + 6 extensions: Double, prioritized replay, dueling, multi-step, distributional & Noisy Nets. {cite:p}`hessel2018rainbow`
- **Why**: The extensions were developed separately → unclear if their gains combine.
- **How**:
    1. Build a [C51](#c51) target from an [$h$-step return](#n-step-returns).
    2. Select the bootstrap action w/ the online net; evaluate it w/ the target net (Double).
    3. Use a dueling architecture w/ distributional outputs.
    4. Prioritize replay by the KL loss.
    5. Explore w/ Noisy Nets (learned param noise) instead of epsilon-greedy.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $h$: Multi-step length, as in [N-Step Returns](#n-step-returns).
    - $V_{\min}$: Lowest atom, as in [C51](#c51).
    - $V_{\max}$: Highest atom, as in C51.
- Misc:
    - $z_\ell$: Fixed return atom, as in C51.
    - $\bar\rho_\ell$: Target-net probability of atom $\ell$, as in C51.

Training:

$$
\begin{align*}
a^*&=\arg\max_{a'}Q_\phi(s_{t+h},a'),\\
\hat z_\ell&=\operatorname{clip}\!\left(\sum_{u=0}^{h-1}\gamma^ur_{t+u+1}+\gamma^hz_\ell,\;V_{\min},\;V_{\max}\right).
\end{align*}
$$

- $a^*$: Online-net greedy action (Double).
- $u$: Step offset inside the $h$-step window.
- Episode ends inside the window → drop the $\gamma^hz_\ell$ term.
- Project $\hat z_\ell$ w/ target-net probabilities $\bar\rho_\ell(s_{t+h},a^*)$; then use the [C51](#c51) loss.
- Paper settings: $h=3$, $51$ atoms on $[-10,10]$, priority exponent $0.5$, IS exponent annealed $0.4\to1$.
```

```{dropdown} Table: Rainbow Components
| Component | Fixes | Effect of removing it (median, 57 Atari games) |
|:--|:--|:--|
| Prioritized replay | Uniform replay wastes updates | Large drop; one of the 2 most important |
| Multi-step ($h=3$) | Slow 1-step value propagation | Large drop, early & final; one of the 2 most important |
| Distributional | Mean-only target | 3rd most important; no difference for the first 40M frames, then lags |
| Noisy Nets | Undirected epsilon-greedy exploration | Worse in aggregate; large drops in some games |
| Dueling | Separate value per action | No significant aggregate difference |
| Double | Overestimation | Limited median difference |
```

```{attention} Q&A
:class: dropdown
*Why did Double matter little?*

- Authors' hypothesis: value clipping to $[-10,10]$ counteracts overestimation.
- Wider support → Double can matter more.

*Is the $h$-step target off-policy correct?*

- No. Rainbow uses uncorrected $h$-step returns from replay → targets mix in the behavior policy's actions.

*Why prioritize by KL loss, not TD error?*

- KL is what the distributional agent minimizes.
- Authors' hypothesis: KL is more robust to stochastic returns, since it can still decrease when returns are not deterministic.
```

&nbsp;
