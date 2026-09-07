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

Estimate return → improve actions: exact backups, sampled backups, then neural approximation. Foundations follow {cite:t}`rlbook`; assumes [MDPs, returns & Bellman equations](intro.md#bellman-equations).

Notations:
- $s_t$: State at decision $t$.
- $a_t$: Action at decision $t$.
- $r_{t+1}$: Reward after $a_t$.
- $\mathcal{S}$: State space.
- $\mathcal{A}$: Discrete action space.
- $p(s'|s,a)$: Transition distribution.
- $r(s,a)$: Expected immediate reward.
- $\pi(a|s)$: Target policy.
- $\mu(a|s)$: Behavior policy.
- $\gamma$: Discount factor; $0\leq\gamma<1$ unless stated otherwise.
- $\alpha$: Learning rate.
- $\epsilon$: Epsilon-greedy exploration probability.
- $T$: Terminal time.
- $G_t$: Discounted return from $t$.
- $V^\pi$: True state value under $\pi$.
- $Q^\pi$: True action value under $\pi$.
- $V$: Current state-value estimate.
- $Q$: Current action-value estimate.
- $d_t$: True-termination indicator after $a_t$, not an artificial time cutoff.
- $\delta_t$: Temporal-difference error.
- $\phi$: Value-network params.
- $\bar\phi$: Target-network params.
- $\mathcal{D}$: Experience dataset/replay buffer.

RL overrides: $a$ denotes an action; $V/Q$ denote scalar-valued functions. Code rewards at index `t` mean $r_{t+1}$; `terminated` means $d_t$.

&nbsp;

## Planning with a Known Model

### Dynamic Programming
- **What**: Solving a decision problem through repeated model-based Bellman backups.
- **Why**: Shared future subproblems make explicit trajectory enumeration wasteful.
- **How**: Store values → combine immediate reward with successor values → reuse updated estimates.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $P_{sas'}$: Probability of transition to nonterminal $s'$ from $(s,a)$.
    - $R_{sa}$: Expected immediate reward.

Process:

$$
Q_V(s,a)=R_{sa}+\gamma\sum_{s'}P_{sas'}V(s')
$$

- $Q_V$: One-step action lookahead using continuation estimate $V$.
- $P$ can be substochastic: missing row mass is terminal probability, with continuation value $0$.
- Terminal rewards remain included in $R$.
- Evaluation backup: $V_{\mathrm{new}}(s)=\sum_a\pi(a|s)Q_V(s,a)$.
- Optimality backup: $V_{\mathrm{new}}(s)=\max_a Q_V(s,a)$.
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
        ## Terminal transition mass is omitted from p, not from r.
        return self.r + self.gamma * np.einsum("sat,t->sa", self.p, value)

## Example
model = TabularModel([[[0.0], [0.0]]], [[1.0, 2.0]], gamma=0.9)
print(model(np.zeros(1)))  ## [[1., 2.]]
```
````

```{attention} Q&A
:class: dropdown
*Is DP model-free RL?*

- No. It plans using known transitions/rewards; no sampled experience is required.
- The same backup underlies model-free sampled updates.

*What limits it?*

- Dense backup cost $O(|\mathcal{S}|^2|\mathcal{A}|)$ per sweep; sparsity can reduce it.
- State enumeration is the main obstacle in large/continuous problems.
```

&nbsp;

### Policy Evaluation
- **What**: Computing the value of a fixed policy.
- **Why**: Policy improvement needs an estimate of its current consequences.
- **How**: Average lookahead over policy actions; repeat until consistent or solve the linear system.

```{note} Math
:class: dropdown
$$
\begin{align*}
P^\pi_{ss'}&=\sum_a\pi(a|s)P_{sas'},&
\mathbf{r}^\pi_s&=\sum_a\pi(a|s)R_{sa},\\
\mathbf{v}^\pi&=\mathbf{r}^\pi+\gamma P^\pi\mathbf{v}^\pi,&
(I-\gamma P^\pi)\mathbf{v}^\pi&=\mathbf{r}^\pi.
\end{align*}
$$

- $P_{sas'}$: Nonterminal continuation kernel.
- $R_{sa}$: Expected immediate reward.
- $P^\pi$: Policy-induced continuation matrix.
- $\mathbf{r}^\pi$: Policy-induced reward vector.
- $\mathbf{v}^\pi$: Vector of state values.
- $I$: Identity matrix.
- $\gamma<1$ → unique solution for finite spaces; solve rather than explicitly invert.
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
        p = np.einsum("sa,sat->st", policy, self.model.p)
        r = (policy * self.model.r).sum(axis=1)
        return np.linalg.solve(np.eye(len(r)) - self.model.gamma * p, r)

## Example
model = TabularModel([[[1.0], [0.0]]], [[1.0, 3.0]], gamma=0.5)
print(PolicyEvaluation(model)(np.array([[1.0, 0.0]])))  ## [2.]
```
````

```{attention} Q&A
:class: dropdown
*Iterative vs. exact evaluation?*

- Linear solve: cubic dense cost in state count, up to numerical precision.
- Iteration: cheaper individual sweeps; stops at a chosen residual tolerance.
- Approximate evaluation may suffice for improvement; it is not an exact policy-value solution.
```

&nbsp;

### Policy Iteration
- **What**: Alternating policy evaluation & greedy policy improvement.
- **Why**: A one-step improvement over correct continuation values improves the full policy.
- **How**:
    1. Evaluate the current policy.
    2. Choose maximizing lookahead actions.
    3. Repeat until the policy no longer changes.

```{note} Math
:class: dropdown
$$
\pi'(s)\in\arg\max_a Q^\pi(s,a)
\quad\Longrightarrow\quad V^{\pi'}(s)\geq V^\pi(s)
$$

- $\pi'$: Greedily improved deterministic policy.
- More generally, $\mathbb{E}_{a\sim\pi'}Q^\pi(s,a)\geq V^\pi(s)$ for every state suffices.
- **Generalized policy iteration**: Evaluation & improvement interact even when neither is completed exactly.
```

```{tip} Derivation
:class: dropdown
1. Improvement gives $\mathcal{T}^{\pi'}V^\pi\geq V^\pi$.
2. Bellman evaluation is monotone → repeated application preserves & propagates the inequality.
3. Contraction gives $(\mathcal{T}^{\pi'})^\ell V^\pi\to V^{\pi'}$.
4. Therefore $V^{\pi'}\geq V^\pi$; a policy greedy w.r.t. its own value satisfies the optimality equation.
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

- Finite discounted MDP, exact evaluation & consistent tie-breaking → finitely many deterministic policies.
- Approximate evaluation or numerical noise can spoil exact monotonicity.

*Why not improve using immediate reward alone?*

- The lookahead must retain the evaluated continuation value.
- Greedy immediate reward ignores delayed consequences.
```

&nbsp;

### Value Iteration
- **What**: Repeated Bellman optimality backups.
- **Why**: Avoid fully evaluating each intermediate policy.
- **How**: Improve & evaluate one step at a time; extract a greedy policy from the resulting values.

```{note} Math
:class: dropdown
Process:

$$
V_{\mathrm{new}}(s)
=\max_a\left[r(s,a)+\gamma\sum_{s'}p(s'|s,a)V(s')\right]
$$

$$
\|V-V^*\|_\infty
\leq\frac{\|\mathcal{T}^*V-V\|_\infty}{1-\gamma}
$$

- $\mathcal{T}^*$: Bellman optimality operator.
- $V^*$: Optimal value.
- Residual → value-error certificate for exact finite discounted backups.
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
            q = self.model(value)
            updated = q.max(axis=1)
            if np.max(np.abs(updated - value)) <= self.tolerance:
                return updated, self.model(updated).argmax(axis=1)
            value = updated

## Example
model = TabularModel([[[1.0], [0.0]]], [[1.0, 3.0]], gamma=0.5)
print(ValueIteration(model, tolerance=1e-8)())  ## values [3.], action [1]
```
````

```{attention} Q&A
:class: dropdown
*Synchronous vs. asynchronous updates?*

- Synchronous sweeps use a frozen previous value vector.
- In-place/asynchronous backups reuse fresher values; convergence needs every relevant state updated sufficiently often.

*What about finite horizons?*

- Start with $V_T=0$; back up $V_{T-1},\ldots,V_0$ once each.
- The optimal action depends on remaining time; no infinite fixed-point iteration is needed.

*Which DP method should I choose?*

- Policy iteration spends more per iteration but often needs fewer improvement rounds.
- Value iteration uses cheap optimality sweeps.
- Modified policy iteration performs several evaluation sweeps between improvements.
```

&nbsp;

## Learning from Returns

### Monte Carlo Prediction
- **What**: Value estimation by averaging sampled complete returns.
- **Why**: Estimate consequences without a transition model or a learned bootstrap target.
- **How**: Finish an episode → compute returns backward → average returns at visited states.

```{note} Math
:class: dropdown
Process:

$$
V(s)\leftarrow V(s)+\frac{1}{N(s)}[G_t-V(s)]
$$

- $N(s)$: Number of included return observations for $s$, including this one.
- **First visit**: Include only the earliest visit to $s$ within each episode.
- **Every visit**: Include all visits; within-episode observations are correlated.
- Constant $\alpha$ instead of $1/N(s)$ gives recency weighting, not the arithmetic sample mean.
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
            if key in seen:
                continue
            seen.add(key)
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

- Targets include all later reward/transition randomness.
- No bootstrap bias, but long trajectories can have noisy returns.

*Does the estimator require a Markov state?*

- Return averaging can estimate conditional returns for a fixed sampling distribution without Markov structure.
- Treating those estimates as a sufficient state for control still requires an adequate state representation.

*Can MC handle nonterminal rollout cutoffs?*

- Pure MC needs the complete return.
- Appending an estimated continuation value makes the target bootstrapped, not pure MC.
```

&nbsp;

#### Monte Carlo Control
- **What**: Return-based action-value estimation interleaved with policy improvement.
- **Why**: Without a model, state values alone cannot rank untried actions.
- **How**: Average returns for visited $(s,a)$ pairs → improve an exploratory policy → collect another episode.

```{note} Math
:class: dropdown
Process:

$$
Q(s_t,a_t)\leftarrow Q(s_t,a_t)
+\alpha[G_t-Q(s_t,a_t)],\qquad
\pi\leftarrow\epsilon\text{-greedy}(Q)
$$

- Hold the policy fixed during each episode.
- $\alpha=1/N(s,a)$ gives the first-visit sample average used below; constant $\alpha$ instead tracks changing values.
- $N(s,a)$: Number of first-visit return observations for the pair, including the current one.
- Exploring starts assume access to all required initial state–action pairs; epsilon-soft policies avoid that reset assumption.
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
*Why not become fully greedy immediately?*

- Unvisited actions can retain incorrect values forever.
- Fixed epsilon keeps exploration but evaluates an exploratory policy, not necessarily an optimal deterministic one.
- GLIE is a standard exploration requirement; it alone is not a blanket convergence proof for every MC-control variant.
```

&nbsp;

### Importance Sampling
- **What**: Reweighting behavior-policy returns to estimate a target-policy expectation.
- **Why**: Logged trajectories may come from a different policy.
- **How**: Multiply action-probability ratios over the relevant trajectory suffix.

```{note} Math
:class: dropdown
$$
w_{t:T-1}
=\prod_{\ell=t}^{T-1}\frac{\pi(a_\ell|s_\ell)}{\mu(a_\ell|s_\ell)}
$$

- $w_{t:T-1}$: Trajectory-suffix importance weight.
- Shared initial state & environment → transition likelihoods cancel.
- For $Q^\pi(s_t,a_t)$, condition on the initial action too → product starts at $t+1$.

$$
\hat V_{\mathrm{ordinary}}(s)=\frac1m\sum_{i=1}^m w_iG_i,
\qquad
\hat V_{\mathrm{weighted}}(s)=\frac{\sum_{i=1}^m w_iG_i}{\sum_{i=1}^m w_i}
$$

- $w_i$: Weight for sampled suffix $i$ starting at $s$.
- $G_i$: Return of sampled suffix $i$.
- $m$: Number of sampled suffixes, using the global sample-count notation.
- Ordinary IS is unbiased with coverage & integrable returns; variance can be enormous or infinite.
- Self-normalized/weighted IS has finite-sample bias but is consistent under suitable sampling/integrability conditions.
```

```{tip} Derivation
:class: dropdown
1. Write $\mathbb{E}_\pi[G]=\sum_\tau p_\pi(\tau)G(\tau)$.
2. Multiply & divide by $p_\mu(\tau)$ wherever $p_\pi(\tau)>0$.
3. Recognize $\mathbb{E}_\mu[(p_\pi/p_\mu)G]$.
4. The same environment kernel appears in both trajectory distributions; only policy ratios remain.
5. Per-decision IS weights reward $r_{\ell+1}$ only through decisions $t,\ldots,\ell$, avoiding irrelevant later ratios.
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
*What cannot be corrected?*

- Missing support: $\mu(a|s)=0$ where $\pi(a|s)>0$.
- Wrong/missing behavior propensities.
- Unobserved confounding in logged decisions; a state that omits decision-relevant information can invalidate the assumed ratios.

*Why not use full-trajectory ratios everywhere?*

- Products amplify variance with horizon.
- Log-space products prevent some numerical overflow/underflow, not statistical variance.
- Clipping weights trades bias for stability; weighted IS does not repair missing support.
```

&nbsp;

## Bootstrapped Prediction

### TD Learning
- **Name**: Temporal-Difference Learning
- **What**: Value learning from sampled rewards plus estimated continuation values.
- **Why**: Update before an episode ends; reuse what is already known about successor states.
- **How**: Compare the current prediction with a one-step-ahead target.

```{note} Math
:class: dropdown
Process:

$$
\delta_t=r_{t+1}+\gamma(1-d_t)V(s_{t+1})-V(s_t),
\qquad
V(s_t)\leftarrow V(s_t)+\alpha\delta_t
$$

- This is **TD(0)**: One-step state-value prediction.
- Sample trajectories from $\pi$ to evaluate $V^\pi$ with this uncorrected update.
```

```{note} Example
:class: dropdown
- $V(s_t)=2$, reward $1$, $V(s_{t+1})=4$, $\gamma=0.5$ → target $3$, error $1$.
- $\alpha=0.1$ → updated value $2.1$.
- If the transition terminates, target $1$ → updated value $1.9$ instead.
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
*Why is the target biased?*

- Until $V=V^\pi$, the bootstrap generally differs from true continuation value.
- It usually removes much of the variance from waiting for the full return.
- With exact $V^\pi$, the one-step target is conditionally unbiased for $V^\pi(s_t)$.

*When does tabular TD(0) converge?*

- Fixed policy, finite suitable task, bounded reward/noise conditions, sufficient state visits & per-state step sizes satisfying $\sum\alpha=\infty$, $\sum\alpha^2<\infty$.
- Constant step sizes can track changes but do not generally converge exactly.

*Model vs. bootstrap: the same thing?*

- No. TD bootstraps from a value estimate without predicting a next state.
- DP bootstraps too, but takes an expectation using a model.
```

&nbsp;

### N-Step Returns
- **What**: Several sampled rewards followed by a value bootstrap.
- **Why**: Trade delayed noisy evidence against an imperfect continuation estimate.
- **How**: Wait for a short segment → accumulate rewards → bootstrap unless the segment truly terminates.

```{note} Math
:class: dropdown
$$
G_t^{(h)}
=\sum_{\ell=0}^{h-1}\gamma^\ell r_{t+\ell+1}
+\gamma^h V(s_{t+h}),\qquad t+h<T
$$

- $h$: Backup length; positive integer.
- $G_t^{(h)}$: $h$-step return target.
- Reaching $T$ → sum only observed rewards through $r_T$ & omit the bootstrap.
- An artificial cutoff before $T$ retains a bootstrap from the last pre-reset state.
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
        bootstrap = 0.0 if terminated else next_value
        return self.returns(rewards, bootstrap)[0]

## Example
print(NStepReturn(0.5)([1, 2], next_value=4, terminated=False))  ## 3.0
```
````

```{dropdown} Table: Backup Families
| Method | Environment model | Sampled reward depth | Bootstrap |
|:--|:--|:--|:--|
| DP | Required | Full expectation at each backup | Yes |
| TD(0) | No | One step | Yes |
| $h$-step TD | No | $h$ steps or termination | At nonterminal endpoint |
| Monte Carlo | No | Complete episode | No |
```

```{attention} Q&A
:class: dropdown
*What changes as the backup gets longer?*

- Bootstrap error is multiplied by $\gamma^h$ before termination.
- More sampled randomness & more delay before an update.
- Bias/variance trends are useful intuition, not a universal monotonic theorem.

*What about off-policy multi-step backups?*

- Intermediate sampled actions follow the behavior policy.
- A final max does not retroactively make that path an on-policy optimal continuation.
- Valid off-policy corrections or explicitly biased approximations are required.
```

&nbsp;

### Eligibility Traces
- **What**: Decaying memory that distributes TD errors to recently visited states or features.
- **Why**: Propagate delayed evidence online without storing every backup length separately.
- **How**: Mark the current state → apply its TD error to active traces → decay traces over time.

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

- $G_t^\lambda$: Episodic forward-view lambda-return.
- $G_t^{(h)}$: $h$-step return with fixed continuation estimates.
- Last term keeps the remaining mixture mass; $\lambda=1$ gives MC, $\lambda=0$ gives one-step TD.

$$
\begin{align*}
e_t(s)&=\gamma\lambda e_{t-1}(s)+\mathbf{1}[s=s_t],\\
\delta_t&=r_{t+1}+\gamma(1-d_t)V(s_{t+1})-V(s_t),\\
V(s)&\leftarrow V(s)+\alpha\delta_t e_t(s).
\end{align*}
$$

- Reset traces at an episode boundary; don't carry eligibility into a reset episode.
- Accumulating traces add visits; replacing traces set visited-state eligibility to $1$.
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
        self.trace *= self.gamma * self.lam
        self.trace[s] += 1
        self.value += self.alpha * delta * self.trace
        if terminated or episode_end:
            self.trace.fill(0)
        return delta

## Example
td = TDLambda(2, alpha=1.0, gamma=1.0, lam=0.5)
td.step(0, 0, 1, False, False)
td.step(1, 2, None, True, True)
print(td.value)  ## [1., 2.]
```
````

```{attention} Q&A
:class: dropdown
*Are forward & backward views exactly identical?*

- Frozen values over an episode + accumulated offline updates give the classic equivalence.
- Conventional online TD(lambda) changes values during the episode → not exactly that fixed-value forward view.
- True-online variants address a different, online forward-view equivalence.

*Can plain traces be used off-policy?*

- Not generally. Behavior actions can invalidate the continuation being credited.
- Off-policy trace corrections/cutting are needed; do not blindly combine accumulating traces with a max backup.
```

&nbsp;

## Sampled Control

### SARSA
- **Name**: State–Action–Reward–State–Action
- **What**: On-policy TD control using the next action actually selected.
- **Why**: Learn consequences of the exploratory policy that will actually act.
- **How**: Select the next action → bootstrap from its Q estimate → update the current pair.

```{note} Math
:class: dropdown
Process:

$$
\delta_t=r_{t+1}+\gamma(1-d_t)Q(s_{t+1},a_{t+1})-Q(s_t,a_t),
\qquad
Q(s_t,a_t)\leftarrow Q(s_t,a_t)+\alpha\delta_t
$$

- Draw $a_{t+1}$ from the same policy being evaluated/improved.
- When continuing the episode, execute that selected next action rather than resampling after the update.
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
*Why does SARSA take a safer cliff-walking route?*

- Persistent exploration can accidentally step off the cliff.
- SARSA values include those future exploratory mistakes.
- This is sensitivity to the behavior policy, not an explicit risk-sensitive objective.

*Convergence to optimal Q?*

- Finite suitable task, bounded rewards, infinite state–action coverage, diminishing per-pair steps & GLIE.
- Fixed epsilon generally learns an exploratory-policy control solution, not $Q^*$.
```

&nbsp;

#### Expected SARSA
- **Name**: Expected State–Action–Reward–State–Action
- **What**: TD control averaging the next action under a specified target policy.
- **Why**: Remove next-action sampling noise when the action sum is tractable.
- **How**: Replace the sampled next Q with its policy-weighted expectation.

```{note} Math
:class: dropdown
Process:

$$
\delta_t=r_{t+1}+\gamma(1-d_t)
\sum_{a'}\pi(a'|s_{t+1})Q(s_{t+1},a')-Q(s_t,a_t)
$$

- $\pi=\mu$ → on-policy; a different target policy gives an off-policy version.
- Greedy target → one-step Q-learning.
```

````{important} Code
:class: dropdown
```python
## Reuses SARSA from rl/value.md.
class ExpectedSARSA(SARSA):
    def step(self, s, a, reward, next_s, next_policy, terminated):
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
- Computing the full action expectation removes only conditional next-action sampling variance.
```

&nbsp;

### Q-Learning
- **What**: Off-policy TD control with a greedy next-action target.
- **Why**: Learn optimal continuation values while behavior remains exploratory.
- **How**: Sample any adequately covered pair → bootstrap from the largest next Q → improve the greedy policy.

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

- These per-pair step-size conditions + infinite coverage, bounded rewards & a finite discounted MDP give tabular convergence to $Q^*$.
- $Q^*$: Optimal action value.
```

````{important} Code
:class: dropdown
```python
## Reuses SARSA's table initialization from rl/value.md.
class QLearning(SARSA):
    def step(self, s, a, reward, next_s, terminated):
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
*Why is Q-learning off-policy even with epsilon-greedy behavior?*

- The backup assumes greedy continuation; epsilon-greedy behavior is different.
- The current sampled action need not maximize Q.

*Does convergence mean safe training?*

- No. An exploratory behavior policy can incur large losses before values are learned.
- An asymptotic theorem says nothing about a finite interaction budget.

*Why can the max overestimate?*

- For unbiased noisy estimates, $\mathbb{E}[\max_a\hat Q(a)]\geq\max_a\mathbb{E}[\hat Q(a)]$.
- Selection favors positive errors; not every approximation error or state must be an overestimate.
```

&nbsp;

#### Double Q-Learning
- **What**: Two-estimator Q-learning that separates action selection from evaluation. {cite:p}`hasselt2010double`
- **Why**: Reusing the same noisy estimate for both introduces maximization bias.
- **How**: Update one table using its maximizing action & the other table's value; alternate tables.

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
- Swap roles for updates to $Q_2$; behavior can be epsilon-greedy w.r.t. $Q_1+Q_2$.
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
        update = int(self.rng.integers(2))
        own, other = self.q[update], self.q[1 - update]
        continuation = 0.0
        if not terminated:
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
*Does double estimation guarantee unbiased values?*

- No. It targets selection/evaluation coupling, not every source of error.
- Can underestimate; correlation between estimators weakens the separation.
```

&nbsp;

## Function Approximation

### Semi-Gradient Value Learning
- **What**: Fitting a parametric value estimate while treating its bootstrap target as fixed.
- **Why**: Tabular values cannot generalize across large state spaces.
- **How**: Build a target from observed rewards & continuation estimates; differentiate only the current prediction.

```{note} Math
:class: dropdown
Model:

$$
V_\phi(s)=\phi^\top\mathbf{x}(s)
$$

- $\mathbf{x}(s)$: State-feature vector; a neural network can replace the linear model.

Training:

$$
\begin{align*}
y_t&=r_{t+1}+\gamma(1-d_t)V_\phi(s_{t+1}),\\
\mathcal{L}(\phi)&=\tfrac12[\operatorname{stopgrad}(y_t)-V_\phi(s_t)]^2,\\
\phi&\leftarrow\phi+\alpha[y_t-V_\phi(s_t)]\nabla_\phi V_\phi(s_t).
\end{align*}
$$

- $y_t$: Bootstrap regression target, using the global output notation.
- $\operatorname{stopgrad}$: Preserve value but block its derivative.
- The update is not the full gradient of a squared Bellman residual.
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
*What is lost compared with a table?*

- Updating one state changes others through shared params.
- The representable value functions may exclude $V^\pi$ or $Q^*$.
- Optimization, approximation & distribution mismatch become separate error sources.

*Why not backpropagate through the target?*

- That defines a different objective/update, not the intended TD semi-gradient.
- An unbiased gradient of a squared expected Bellman residual generally needs independent next-state samples for the same pair: the double-sampling problem.
```

&nbsp;

### Deadly Triad
- **What**: Potential instability from combining approximation, bootstrapping & off-policy learning.
- **Why**: Shared predictions can reinforce unsupported targets under a mismatched sampling distribution.
- **How**: An erroneous value becomes another target → updates propagate & amplify the error.

```{attention} Q&A
:class: dropdown
*Does combining all three always diverge?*

- No. The combination permits divergence; it does not force it in every task.
- Even linear off-policy TD can diverge.
- On-policy linear TD has convergence results under appropriate assumptions; arbitrary nonlinear control does not inherit them.

*Do replay & target networks solve the theorem?*

- They reduce correlation & target drift, not prove general convergence.
- Track value magnitudes, target errors, coverage & actual return separately.
```

&nbsp;

### DQN
- **Name**: Deep Q-Network {cite:p}`mnih2015human`
- **What**: Neural Q-learning with experience replay & a delayed target network.
- **Why**: Correlated transitions & rapidly moving bootstrap targets destabilize neural Q-learning.
- **How**:
    1. Act epsilon-greedily; store transitions.
    2. Sample replay minibatches; regress selected-action Q toward frozen targets.
    3. Periodically copy online weights into the target network.

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

- $y_t$: Frozen Q-learning target.
- $\ell_\kappa$: Huber loss; squared near zero, linear in large residual magnitude.
- $\kappa$: Positive Huber transition threshold.
- Only the selected action is regressed; outputs are values, not class probabilities.
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
        prediction = self.q(s).gather(1, a[:, None]).squeeze(1)
        loss = F.smooth_l1_loss(prediction, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.updates += 1
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
*Why two networks?*

- Online Q changes every optimizer step; target Q supplies temporarily stationary labels.
- Delayed copying trades target freshness for slower feedback.
- A target network alone is not [Double DQN](#double-dqn).

*Why replay?*

- Reuse expensive transitions & reduce sequential sample correlation.
- Store immutable observations, action, reward, next pre-reset observation & true termination.
- A mixed replay distribution can become stale; replay does not guarantee independent samples or coverage.

*What belongs around the update kernel?*

- Epsilon-greedy collection, environment reset handling & replay warm-up.
- Match observation preprocessing between collection & updates.
- The snippet uses vector observations, CPU or a consistent device, and no dropout/batch normalization.
- Huber threshold & example hyperparams illustrate an update, not tuned defaults.

*Why not softmax the outputs?*

- Q values are unrestricted expected returns; softmax would erase their scale.
- Discrete actions allow an explicit max. Continuous actions require inner optimization or an actor.
```

&nbsp;

#### Double DQN
- **Name**: Double Deep Q-Network {cite:p}`hasselt2016deep`
- **What**: DQN selecting next actions online & evaluating them with the target network.
- **Why**: A delayed target still overselects its own positive estimation errors.
- **How**: Keep replay & target synchronization; split the target's selection and evaluation roles.

```{note} Math
:class: dropdown
Process:

$$
y_t=r_{t+1}+\gamma(1-d_t)
Q_{\bar\phi}\!\left(s_{t+1},\arg\max_{a'}Q_\phi(s_{t+1},a')\right)
$$

- $y_t$: Double-DQN regression target.
- Online & target networks are correlated; bias reduction is not guaranteed unbiasedness.
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
*Double vs. dueling?*

- Double changes the bootstrap target.
- Dueling changes the network parameterization.
- They can be combined.
```

&nbsp;

#### Dueling Networks
- **What**: Q parameterization with state-value & centered action-advantage streams. {cite:p}`wang2016dueling`
- **Why**: Many actions share most of their value through the state itself.
- **How**: Learn shared state features; split into scalar value & per-action scores; center scores before adding.

```{note} Math
:class: dropdown
Forward:

$$
Q_\phi(s,a)=v_\phi(s)+u_\phi(s,a)
-\frac{1}{|\mathcal{A}|}\sum_{a'}u_\phi(s,a')
$$

- $v_\phi$: Scalar value stream.
- $u_\phi$: Raw action-score stream.
- Centering removes the additive ambiguity between streams.
- $v_\phi$ is the uniform action mean of represented Q, not automatically $V^\pi$ for a nonuniform policy.
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
        return self.value(h) + advantage - advantage.mean(dim=-1, keepdim=True)

## Example
print(DuelingQ(2, 8, 3)(torch.zeros(2, 2)).shape)  ## torch.Size([2, 3])
```
````

```{attention} Q&A
:class: dropdown
*Does the advantage stream itself have zero mean?*

- Raw outputs need not; the centered contribution does.
- Centering under a uniform action average differs from the policy-weighted $A^\pi$ identity.
```

&nbsp;

#### Prioritized Replay
- **What**: Replay sampling biased toward transitions with large learning errors. {cite:p}`schaul2015prioritized`
- **Why**: Uniform replay spends equal effort on transitions with unequal learning signal.
- **How**: Sample using priorities → weight losses for sampling bias → refresh priorities after updates.

```{note} Math
:class: dropdown
Process:

$$
p_i=|\delta_i|+\varepsilon_p,\qquad
P(i)=\frac{p_i^\eta}{\sum_jp_j^\eta},\qquad
w_i=\left(\frac{1}{N P(i)}\right)^\zeta
$$

- $p_i$: Positive transition priority.
- $\delta_i$: Current TD residual for replay item $i$.
- $\varepsilon_p$: Positive priority floor.
- $P(i)$: Sampling probability of item $i$.
- $\eta$: Prioritization exponent; $0$ gives uniform replay.
- $N$: Buffer size.
- $\zeta$: Importance-correction exponent; $1$ corrects to a uniform-buffer expectation before normalization.
- $w_i$: Loss weight; treat priorities/weights as fixed during the value update.
- Here $j$ indexes replay items rather than features.
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
        priority = (np.abs(td_errors) + self.floor) ** self.exponent
        probability = priority / priority.sum()
        index = self.rng.choice(len(priority), batch_size, p=probability)
        weight = (len(priority) * probability[index]) ** (-self.correction)
        return index, weight

## Example
sampler = PrioritizedSampler(exponent=0.0, correction=1.0, floor=1e-6)
index, weight = sampler(np.array([1.0, 5.0]), batch_size=2)
print(weight)  ## [1., 1.]
```
````

```{attention} Q&A
:class: dropdown
*How does this integrate with DQN?*

- Replace uniform indices with prioritized samples.
- Use unreduced per-item Huber losses, multiply by $w_i$, then average.
- Refresh sampled priorities from detached absolute TD errors; retain a positive chance of replaying every item.

*Does it fix off-policy learning?*

- Weights correct prioritized sampling toward the uniform replay objective, not toward the current policy's trajectory distribution.
- Large TD errors can be irreducible noise; stale priorities can misallocate updates.
- The reference sampler scans the buffer; tree-based sampling matters only when buffer scale warrants it.
```

&nbsp;
