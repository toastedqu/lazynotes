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
# Introduction

Sequential decision-making: specify the objective → estimate consequences → improve behavior. Foundations follow {cite:t}`rlbook`; continue with [value-based methods](value.md), [policy-based methods](policy.md), then [models & planning](model.md).

Notations:
- $s_t$: State before decision $t$.
- $\mathcal{S}$: State space.
- $a_t$: Action at decision $t$.
- $\mathcal{A}$: Action space.
- $r_{t+1}$: Reward received after $a_t$.
- $p(s'|s,a)$: Next-state transition distribution.
- $r(s,a)$: Expected immediate reward given $(s,a)$.
- $\rho_0$: Initial-state distribution.
- $\pi(a|s)$: Target policy.
- $\mu(a|s)$: Behavior policy generating data.
- $\gamma$: Discount factor; $0\leq\gamma<1$ unless stated otherwise.
- $T$: Terminal time; actions run from $0$ through $T-1$.
- $G_t$: Discounted return starting at $t$.
- $V^\pi(s)$: State value under $\pi$.
- $Q^\pi(s,a)$: Action value under $\pi$.
- $A^\pi(s,a)$: Advantage under $\pi$.
- $J(\pi)$: Expected return from $\rho_0$.

RL overrides: $a$ denotes an action, $r$ a reward, $V/Q/A$ scalar-valued functions rather than matrices.

&nbsp;

## Decision Problems

### RL
- **Name**: Reinforcement Learning
- **What**: Learning behavior to maximize expected cumulative reward.
- **Why**: Actions change future opportunities; feedback does not label the best action.
- **How**:
    1. Observe → act → receive reward & next observation.
    2. Estimate consequences from experience.
    3. Improve behavior → collect different experience.

```{dropdown} Table: Course Map
| Page | Central question | Main tools |
|:--|:--|:--|
| Introduction | What exactly are we optimizing? | MDPs, returns, Bellman equations, exploration |
| [Value](value.md) | How good is each state/action? | DP, MC, TD, SARSA, Q-learning, DQN |
| [Policy](policy.md) | How should policy params change? | Policy gradients, actor–critic, PPO, SAC |
| [Model](model.md) | What happens if we act? | Learned dynamics, Dyna, search, MPC |
```

```{attention} Q&A
:class: dropdown
*How is RL different from supervised learning?*

- Feedback scores chosen behavior, not every available action.
- Actions influence later inputs → data distribution depends on the learner.
- Delayed rewards → temporal credit assignment, not merely fitting observed labels.

*Must data be online or non-i.i.d.?*

- Interaction usually creates correlated trajectories; replay can reduce correlation.
- Offline RL learns from a fixed dataset. Neither online collection nor non-i.i.d. sampling defines RL.
- Even i.i.d. sampled transitions retain a sequential decision objective.

*Prediction vs. control?*

- **Prediction**: Evaluate a specified policy.
- **Control**: Find a better policy.

*Does maximizing reward imply consciousness?*

- An optimization criterion, not a theory of subjective experience.
- A useful controller can exploit narrow regularities w/o a general world model.
```

&nbsp;

### MDP
- **Name**: Markov Decision Process
- **What**: Fully observed, Markovian sequential decision problem.
- **Why**: A sufficient state makes consequences depend on the present, not the entire history.
- **How**: State + action → joint next-state/reward draw → next decision.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $p(s',r|s,a)$: Joint next-state/reward kernel.
    - $\mathcal{M}$: Decision problem including its objective.

$$
\mathcal{M}=(\mathcal{S},\mathcal{A},p,r,\rho_0,\gamma),\qquad
r(s,a)=\mathbb{E}[r_{t+1}|s_t=s,a_t=a]
$$

- The joint kernel specifies stochastic rewards; its next-state marginal plus $r(s,a)$ suffices for expected-return Bellman equations.

$$
\Pr(s_{t+1},r_{t+1}|s_0,a_0,\ldots,s_t,a_t)
=p(s_{t+1},r_{t+1}|s_t,a_t)
$$

- **Markov property**: State screens off history, conditional on the action.
- **Stationarity**: The kernel does not change with $t$; a separate assumption.
- Finite horizon → include time-to-go in the state or use time-indexed values/policies.
- Terminal state → absorbing transition, subsequent rewards $0$.
```

```{note} Example
:class: dropdown
- Robot position alone: same location, different velocity → different next position under the same action.
- Position + velocity may be Markov for the chosen dynamics.
- Hidden battery degradation or another adapting agent can invalidate that state description.
```

```{attention} Q&A
:class: dropdown
*Does Markov mean deterministic or memoryless behavior?*

- Stochastic transitions are allowed.
- The environment needs no history beyond a sufficient state. An agent may use memory to construct that state.

*Is the MDP known?*

- The mathematical problem exists whether or not the agent knows its kernel.
- Known kernel → [planning](model.md#model-based-rl); unknown kernel → learn values/policies directly or learn a model.

*Is a finite action set required?*

- No. Sums become integrals for continuous spaces; maximization and approximation become harder.
- Tabular convergence results do not transfer automatically to continuous spaces or neural approximators.
```

&nbsp;

### POMDP
- **Name**: Partially Observable Markov Decision Process
- **What**: An MDP whose latent state is observed only indirectly.
- **Why**: Sensors often hide variables needed to predict consequences.
- **How**: Infer state from observation/action history → choose an action from that information.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $o_t$: Observation at $t$.
- Misc:
    - $O(o'|s',a)$: Observation likelihood after an action.
    - $h_t$: Available observation/action/reward history.
    - $b_t(s)$: Belief $\Pr(s_t=s|h_t)$.

$$
b_{t+1}(s')
=\frac{O(o_{t+1}|s',a_t)\sum_s p(s'|s,a_t)b_t(s)}
{\sum_{\bar s}O(o_{t+1}|\bar s,a_t)\sum_s p(\bar s|s,a_t)b_t(s)}
$$

- This form assumes the observation includes all informative feedback; incorporate reward likelihood too if reward reveals state.
- A belief is a sufficient information state when the model & Bayesian update are correct.
- Policy $\pi(a|b)$ acts in a fully observed belief-state MDP.
```

```{attention} Q&A
:class: dropdown
*Why not treat the latest observation as the state?*

- Identical observations can hide different states → incompatible optimal actions.
- Frame stacks approximate short-term missing information; recurrent policies summarize longer history.
- Neither guarantees a sufficient state.

*Does randomizing actions solve partial observability?*

- No. Memory/belief tracking addresses missing state information.
- Information-gathering actions can improve later decisions.
```

&nbsp;

### Policies & Trajectories
- **What**: An action-selection rule & the interaction sequence it induces.
- **Why**: Behavior optimization depends on both chosen actions & resulting state visits.
- **How**: Alternate policy draws with environment draws.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\tau$: Trajectory $(s_0,a_0,r_1,\ldots,s_T)$.
    - $p(s',r|s,a)$: Joint transition/reward kernel.
    - $d_\gamma^\pi(s)$: Normalized discounted state-visitation distribution.

$$
p_\pi(\tau)=\rho_0(s_0)\prod_{t=0}^{T-1}
\pi(a_t|s_t)\,p(s_{t+1},r_{t+1}|s_t,a_t)
$$

$$
d_\gamma^\pi(s)=(1-\gamma)\sum_{t=0}^{\infty}\gamma^t\Pr_\pi(s_t=s),
\qquad
J(\pi)=\frac{\mathbb{E}_{s\sim d_\gamma^\pi,\ a\sim\pi}[r(s,a)]}{1-\gamma}
$$

- Episodic tasks use an absorbing terminal state to define the infinite sum.
- Stationary deterministic policy: one action per state.
- Stationary stochastic policy: an action distribution per state.
```

```{attention} Q&A
:class: dropdown
*Can an optimal policy be deterministic?*

- Yes for finite, fully observed, discounted MDPs with the ordinary expected-return objective.
- Finite horizon: generally time-dependent unless time is part of the state.
- Entropy constraints, restricted memory, occupancy constraints, or games can change the answer.

*Why care about the state distribution?*

- Improving action probabilities changes which states will be encountered.
- A policy can perform well on its training states but visit unfamiliar states after an update.
- This dependency drives [policy gradients](policy.md) & distribution-shift problems.
```

&nbsp;

## Objectives & Consequences

### Rewards and Returns
- **What**: Immediate feedback & its discounted accumulation.
- **Why**: Locally attractive actions can sacrifice larger future rewards.
- **How**: Specify success → aggregate rewards over the task horizon.

```{note} Math
:class: dropdown
$$
\begin{align*}
G_t&=\sum_{\ell=0}^{T-t-1}\gamma^\ell r_{t+\ell+1},&
G_T&=0,\\
G_t&=r_{t+1}+\gamma G_{t+1},&
J(\pi)&=\mathbb{E}_\pi[G_0].
\end{align*}
$$

- Bounded rewards $|r_{t+1}|\leq R_{\max}$ & $\gamma<1$ imply $|G_t|\leq R_{\max}/(1-\gamma)$.
- $R_{\max}$: Uniform reward-magnitude bound.
- $\gamma=1$ is valid for bounded finite-horizon rewards; random-length episodes need integrability, not merely eventual termination.

$$
J_{\mathrm{avg}}(\pi)=\lim_{H\to\infty}\frac{1}{H}
\mathbb{E}_\pi\!\left[\sum_{t=0}^{H-1}r_{t+1}\right]
$$

- $J_{\mathrm{avg}}$: Average-reward objective, when the limit exists.
- $H$: Number of observed decisions.
- Average reward is not an undiscounted infinite sum; recurrent-class assumptions determine initial-state dependence.
```

```{note} Example
:class: dropdown
- Reward sequence $(1,2,3)$ & $\gamma=\tfrac12$ → $G_0=1+1+\tfrac34=2.75$.
- Choice: terminate for $2$, or receive $0$ then $5$ → delayed choice wins iff $5\gamma>2$.
```

````{important} Code
:class: dropdown
```python
import numpy as np

class DiscountedReturn:
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, rewards, bootstrap=0.0):
        ## Nonterminal cutoffs bootstrap; true terminal return is zero.
        returns = np.empty(len(rewards), dtype=float)
        future = bootstrap
        for t in reversed(range(len(rewards))):
            future = rewards[t] + self.gamma * future
            returns[t] = future
        return returns

## Example
print(DiscountedReturn(0.5)([1, 2, 3]))  ## [2.75, 3.5, 3.0]
```
````

```{attention} Q&A
:class: dropdown
*What does discounting change?*

- $\gamma=0$ → immediate expected reward only; $\gamma$ near $1$ → more weight on distant outcomes.
- $1/(1-\gamma)$ is the sum of discount weights, not a hard horizon.
- Discounting changes the objective; it is not merely a numerical stabilizer.

*Termination vs. truncation?*

- **Termination**: The task ends → no bootstrap.
- **Truncation**: Collection stops but the task could continue → bootstrap from the final pre-reset state.
- An actual finite-horizon deadline is termination; include time-to-go in the state.
- Never bootstrap from a reset observation supplied by an auto-reset wrapper.

*Which reward transformations preserve the optimal policy?*

- Positive scaling preserves ordinary expected-return rankings.
- A constant per step preserves rankings for fixed-length or discounted continuing trajectories.
- Variable episode length breaks constant-shift invariance: a living bonus may reward delaying termination.
- Reward clipping or arbitrary dense bonuses can change the task.

*Sparse vs. dense rewards?*

- Sparse rewards specify success directly but make discovery & credit assignment hard.
- Dense rewards guide learning but may favor proxy behavior over the intended outcome.
- Expected-return maximization is risk-neutral; it does not enforce tail-risk or safety constraints.
```

&nbsp;

#### Potential-Based Reward Shaping
- **What**: Reward augmentation by a discounted potential difference. {cite:p}`ng1999policy`
- **Why**: Add intermediate guidance w/o changing optimal action ordering.
- **How**: Reward increases in a state potential; discount the next-state potential consistently.

```{note} Math
:class: dropdown
$$
r'_{t+1}=r_{t+1}+\gamma\Phi(s_{t+1})-\Phi(s_t),
\qquad
G'_0=G_0-\Phi(s_0)+\gamma^T\Phi(s_T)
$$

- $\Phi(s)$: Bounded state potential.
- $r'_{t+1}$: Shaped reward.
- $G'_0$: Shaped return.
- $\gamma<1$ & bounded $\Phi$ → terminal term vanishes as $T\to\infty$.
- Episodic implementation → set terminal potentials to $0$.
- Then $Q'^\pi(s,a)=Q^\pi(s,a)-\Phi(s)$ → same maximizing actions.
```

````{important} Code
:class: dropdown
```python
class PotentialShaping:
    def __init__(self, potential, gamma):
        self.potential, self.gamma = potential, gamma

    def __call__(self, s, reward, next_s, terminated):
        next_phi = 0.0 if terminated else self.potential(next_s)
        return reward + self.gamma * next_phi - self.potential(s)

## Example
shape = PotentialShaping(lambda distance: -distance, gamma=0.9)
print(shape(2, 0, 1, False))  ## 1.1
```
````

```{attention} Q&A
:class: dropdown
*Does any progress bonus work?*

- No. The discounted difference makes bonuses telescope.
- Nonzero terminal potentials can change episode rankings; a mismatched discount can reward cycles.
```

&nbsp;

### Value Functions
- **What**: Expected future return conditional on a state or state–action pair.
- **Why**: Compare consequences without retaining every possible future trajectory.
- **How**: Average returns under a specified continuation policy.

```{note} Math
:class: dropdown
$$
\begin{align*}
V^\pi(s)&=\mathbb{E}_\pi[G_t|s_t=s],\\
Q^\pi(s,a)&=\mathbb{E}_\pi[G_t|s_t=s,a_t=a],\\
V^\pi(s)&=\mathbb{E}_{a\sim\pi(\cdot|s)}[Q^\pi(s,a)],\\
A^\pi(s,a)&=Q^\pi(s,a)-V^\pi(s),\\
\mathbb{E}_{a\sim\pi(\cdot|s)}[A^\pi(s,a)]&=0.
\end{align*}
$$

$$
V^*(s)=\sup_\pi V^\pi(s),\qquad
Q^*(s,a)=\sup_\pi Q^\pi(s,a),\qquad
J(\pi)=\mathbb{E}_{s_0\sim\rho_0}[V^\pi(s_0)]
$$

- $V^*$: Optimal state value.
- $Q^*$: Optimal action value.
- Values depend on reward, discount & horizon; finite-horizon values generally depend on time.
```

```{note} Example
:class: dropdown
- $Q^\pi(s,a_1)=2$, $Q^\pi(s,a_2)=6$ & $\pi=(\tfrac34,\tfrac14)$.
- $V^\pi(s)=3$; advantages $(-1,3)$ → policy-weighted mean $0$.
- The best action need not have positive absolute return.
```

```{attention} Q&A
:class: dropdown
*Why learn Q instead of V?*

- $Q$ directly ranks actions.
- $V$ needs a model for lookahead or a separate policy for acting.

*Is Q the immediate reward?*

- No: current reward + consequences under the continuation policy.
- $Q^\pi$ assumes $\pi$ after the first action, even when that action differs from $\pi$.

*Does a value estimate provide uncertainty?*

- No. An expected-return estimate is neither a return distribution nor a confidence interval.
- A high value can be accurate or an extrapolation error.
```

&nbsp;

### Bellman Equations
- **What**: Recursive consistency conditions for expected return.
- **Why**: Long-horizon reasoning can be decomposed into one-step predictions.
- **How**: Immediate reward + discounted next-state value; average or maximize over the next action.

```{note} Math
:class: dropdown
$$
\begin{align*}
V^\pi(s)
&=\sum_a\pi(a|s)\left[r(s,a)+\gamma\sum_{s'}p(s'|s,a)V^\pi(s')\right],\\
Q^\pi(s,a)
&=r(s,a)+\gamma\sum_{s'}p(s'|s,a)\sum_{a'}\pi(a'|s')Q^\pi(s',a'),\\
V^*(s)
&=\max_a\left[r(s,a)+\gamma\sum_{s'}p(s'|s,a)V^*(s')\right],\\
Q^*(s,a)
&=r(s,a)+\gamma\sum_{s'}p(s'|s,a)\max_{a'}Q^*(s',a').
\end{align*}
$$

- Finite spaces shown; integrals/suprema replace sums/maxima when needed.
- Terminal continuation values are $0$.

$$
\begin{align*}
\mathcal{T}^\pi v&=\mathbf{r}^\pi+\gamma P^\pi v,\\
\|\mathcal{T}^\pi u-\mathcal{T}^\pi v\|_\infty
&\leq\gamma\|u-v\|_\infty,\\
\|\mathcal{T}^*u-\mathcal{T}^*v\|_\infty
&\leq\gamma\|u-v\|_\infty.
\end{align*}
$$

- $\mathcal{T}^\pi$: Bellman expectation operator.
- $\mathcal{T}^*$: Bellman optimality operator.
- $v$: Candidate vector of state values; local vector notation.
- $u$: Another candidate value vector.
- $\mathbf{r}^\pi$: Expected immediate reward vector under $\pi$.
- $P^\pi$: Policy-induced transition matrix.
- $\|\cdot\|_\infty$: Largest absolute statewise error.
```

```{tip} Derivation
:class: dropdown
1. Substitute $G_t=r_{t+1}+\gamma G_{t+1}$ into the value definition.
2. Condition on the first action & next state.
3. Markov property + stationary policy → remaining expectation is $V^\pi(s_{t+1})$.
4. Optimal continuation → replace the policy-weighted action average with a maximum.
5. An expectation cannot enlarge a uniform error; $|\max_a f(a)-\max_a g(a)|\leq\max_a|f(a)-g(a)|$.
6. Multiply by $\gamma<1$ → contraction → unique fixed point & convergence of exact repeated backups.
```

```{attention} Q&A
:class: dropdown
*Equation vs. algorithm?*

- Bellman equations characterize values; [dynamic programming](value.md#dynamic-programming) solves them using a model.
- [TD learning](value.md#td-learning) approximates backups from sampled transitions.

*Why is maximization inside the next-state expectation for Q?*

- The next action can depend on the next state actually observed.
- Moving the maximum outside commits to one next action before observing that state.

*When does the contraction argument fail?*

- $\gamma=1$: no strict sup-norm contraction in general; finite-horizon backward induction still works.
- Neural projection/optimization: the full update is not the exact Bellman operator.
- Off-policy sampling + approximation + bootstrapping can diverge despite the underlying contraction.
```

&nbsp;

## Exploration & Data

### Exploration and Exploitation
- **What**: Choosing information-gathering actions vs. currently preferred actions.
- **Why**: Rewards of chosen actions cannot identify consequences of actions never tried.
- **How**: Reserve decisions for uncertainty reduction; exploit accumulated evidence on the rest.

```{dropdown} Table: Exploration Mechanisms
| Mechanism | Principle | Limitation |
|:--|:--|:--|
| $\epsilon$-greedy | Sometimes choose uniformly | Ignores which actions are uncertain |
| Optimistic initialization | Unvisited actions start attractive | Optimism may disappear before discovery |
| Upper confidence bound | Value + uncertainty bonus | A bandit bonus alone does not solve deep exploration |
| Thompson sampling | Sample a plausible model, act greedily | Needs useful posterior uncertainty |
| Entropy regularization | Reward policy diversity | Diversity is not calibrated uncertainty |
| Intrinsic reward | Reward novelty/information | Can chase stochastic distractions or alter the task |
```

```{attention} Q&A
:class: dropdown
*Why can random action noise fail?*

- Sparse reward may require a coherent sequence of individually unattractive actions.
- Deep exploration maintains a hypothesis long enough to reach informative states.

*Exploration vs. credit assignment?*

- Exploration discovers rewarding trajectories.
- Credit assignment identifies which earlier decisions caused the observed reward.
- Better return estimators do not discover an unseen reward by themselves.
```

&nbsp;

#### Epsilon-Greedy
- **What**: Greedy action selection mixed with uniform random exploration.
- **Why**: Keep discrete actions reachable while favoring estimated value.
- **How**: Flip an exploration coin; otherwise sample among maximizing actions.

```{note} Math
:class: dropdown
$$
\pi(a|s)=\frac{\epsilon}{|\mathcal{A}|}
+(1-\epsilon)\frac{\mathbf{1}[a\in\mathcal{A}^*(s)]}{|\mathcal{A}^*(s)|}
$$

- $\epsilon$: Exploration probability.
- $\mathcal{A}^*(s)$: Actions maximizing the current value estimate.
- $\mathbf{1}[\cdot]$: Indicator.
```

````{important} Code
:class: dropdown
```python
import numpy as np

class EpsilonGreedy:
    def __init__(self, epsilon, seed=0):
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

    def __call__(self, q):
        q = np.asarray(q)
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(len(q)))
        ## Random ties avoid preferring the first unseen action.
        return int(self.rng.choice(np.flatnonzero(q == q.max())))

## Example
print(EpsilonGreedy(0.0)([1.0, 3.0]))  ## 1
```
````

```{attention} Q&A
:class: dropdown
*Does decaying epsilon guarantee convergence?*

- No. Every required state–action pair must still receive enough visits.
- **GLIE**: Greedy in the Limit with Infinite Exploration; both parts matter.
- Decay too quickly → permanently miss a good action.
```

&nbsp;

### Bandits
- **What**: Decision problems without action-dependent future state consequences.
- **Why**: Isolate exploration from temporal credit assignment.
- **How**: Select an arm → observe its reward → update its estimate.

```{note} Math
:class: dropdown
$$
\mathcal{R}_H=Hq_*-\mathbb{E}\!\left[\sum_{t=0}^{H-1}r_{t+1}\right],
\qquad q_*=\max_a q(a)
$$

- $\mathcal{R}_H$: Expected cumulative regret over $H$ decisions.
- $H$: Decision budget.
- $q(a)$: Arm mean reward.
- $q_*$: Best arm mean.
```

```{attention} Q&A
:class: dropdown
*Contextual bandit vs. MDP?*

- Contextual bandit observes a context before choosing; the action does not control later contexts.
- MDP actions change later states → greedy immediate reward need not optimize return.

*Why not treat full trajectories as arms?*

- Action-sequence count grows combinatorially.
- Ignores shared subproblems & adaptation to intermediate observations.
```

&nbsp;

#### UCB
- **Name**: Upper Confidence Bound
- **What**: Optimistic action selection from a value estimate plus an uncertainty bonus.
- **Why**: Explore arms that could plausibly beat the current best.
- **How**: Try unobserved arms first; then maximize an upper estimate of reward.

```{note} Math
:class: dropdown
$$
a_t\in\arg\max_a\left[
\hat q_t(a)+c\sqrt{\frac{\log t}{N_t(a)}}\right]
$$

- $\hat q_t(a)$: Empirical mean from prior pulls.
- $N_t(a)$: Number of prior pulls of $a$.
- $c$: Exploration coefficient; guarantees need assumptions & calibrated choice.
- Unpulled arms have infinite priority; use the expression after initialization.
```

````{important} Code
:class: dropdown
```python
import numpy as np

class UCB:
    def __init__(self, actions, coefficient):
        self.count = np.zeros(actions, dtype=int)
        self.mean = np.zeros(actions)
        self.coefficient = coefficient

    def __call__(self):
        unseen = np.flatnonzero(self.count == 0)
        if len(unseen):
            return int(unseen[0])
        bonus = self.coefficient * np.sqrt(
            np.log(self.count.sum()) / self.count
        )
        return int(np.argmax(self.mean + bonus))

    def step(self, action, reward):
        self.count[action] += 1
        self.mean[action] += (
            reward - self.mean[action]
        ) / self.count[action]

## Example
bandit = UCB(2, coefficient=1.0)
bandit.step(0, 1.0)
print(bandit())  ## 1: still unobserved
```
````

```{attention} Q&A
:class: dropdown
*What assumptions matter?*

- Standard logarithmic-regret results assume stationary rewards & suitable bounded/sub-Gaussian tails.
- Nonstationarity needs forgetting or change detection.
- Neural Q estimates do not inherit count-based bandit confidence guarantees.
```

&nbsp;

#### Thompson Sampling
- **What**: Greedy action selection under a posterior-sampled reward model.
- **Why**: Explore according to uncertainty about which action is best.
- **How**: Update posterior → sample plausible arm means → select the largest.

```{note} Math
:class: dropdown
$$
\tilde q(a)\sim\operatorname{Beta}(u_a,v_a),\qquad
a_t=\arg\max_a\tilde q(a)
$$

- $\tilde q(a)$: Sampled Bernoulli success probability.
- $u_a$: Prior success parameter + observed successes.
- $v_a$: Prior failure parameter + observed failures.
- Bernoulli reward $r\in\{0,1\}$ → $u_a\leftarrow u_a+r$, $v_a\leftarrow v_a+1-r$.
```

````{important} Code
:class: dropdown
```python
import numpy as np

class ThompsonSampling:
    def __init__(self, actions, seed=0):
        self.success = np.ones(actions)
        self.failure = np.ones(actions)
        self.rng = np.random.default_rng(seed)

    def __call__(self):
        return int(np.argmax(self.rng.beta(self.success, self.failure)))

    def step(self, action, reward):
        if reward not in (0, 1):
            raise ValueError("Beta-Bernoulli updates need reward 0 or 1")
        self.success[action] += reward
        self.failure[action] += 1 - reward

## Example
bandit = ThompsonSampling(2)
bandit.step(0, 1)
print(bandit.success)  ## [2., 1.]
```
````

```{attention} Q&A
:class: dropdown
*Does posterior sampling mean adding arbitrary noise?*

- No. Posterior spread should shrink as evidence accumulates.
- Model misspecification can produce confidently wrong exploration.
```

&nbsp;

### On-Policy and Off-Policy Learning
- **What**: Learning about the data-generating policy vs. a different target policy.
- **Why**: Separate the policy being evaluated/improved from the source of experience.
- **How**: Match collection to the target, correct distribution mismatch, or use a valid off-policy backup.

```{note} Math
:class: dropdown
$$
\pi=\mu\quad\text{(on-policy)},\qquad
\pi\neq\mu\quad\text{(off-policy)}
$$

- Importance-sampling coverage: $\pi(a|s)>0\Rightarrow\mu(a|s)>0$ at relevant states.
- [Importance sampling](value.md#importance-sampling) corrects likelihoods; arbitrary replay does not make an estimator unbiased.
```

```{dropdown} Table: Independent Axes
| Axis | Alternatives | Distinction |
|:--|:--|:--|
| Target vs. behavior | On-policy / off-policy | Which policy does the update concern? |
| Data access | Online / offline | Can new environment data be collected? |
| Dynamics use | Model-free / model-based | Is a transition model used? |
| Representation | Tabular / approximation | Separate entries or shared params? |
| Learned decision object | Value / policy / both | How are actions chosen? |
```

```{attention} Q&A
:class: dropdown
*Why can Q-learning use exploratory actions without importance ratios?*

- A transition at $(s,a)$ is evidence about that conditional environment transition.
- Its target maximizes the next action rather than averaging the behavior policy.
- Tabular coverage & step-size conditions remain necessary.

*Is replay always compatible with policy gradients?*

- No. A naive likelihood-ratio gradient on old data uses the wrong trajectory distribution.
- PPO reuses a recent rollout with a local surrogate; SAC uses an off-policy critic objective.

*Why is offline RL harder than replay-based online RL?*

- No new interaction to correct values of unsupported actions.
- Greedy improvement can exploit extrapolation error outside the dataset.
- Behavior regularization or pessimistic values limit unsupported improvement; logged data alone cannot identify unseen consequences.
```

&nbsp;

### Experimental Evaluation
- **What**: Estimating deployed-policy performance & learning efficiency.
- **Why**: Training loss, exploratory return & cherry-picked runs can misrepresent control quality.
- **How**:
    1. Fix task version, reward, horizon & deployment policy.
    2. Train independent runs; evaluate frozen policies on held-out episodes.
    3. Report return distribution, environment interactions & compute budget.
    4. Separate tuning choices from final evaluation.

```{attention} Q&A
:class: dropdown
*Which measurements answer different questions?*

- **Return**: Task success under the evaluation protocol.
- **Sample efficiency**: Return vs. real environment interactions; distinguish reused/model-generated transitions.
- **Compute efficiency**: Return vs. wall-clock or compute budget.
- **Reliability**: Variation across training seeds, tasks & evaluation episodes.
- **Safety**: Constraint violations & tail outcomes, not just the mean.

*Why can critic loss fall while return worsens?*

- Targets move, coverage changes, or value extrapolation misleads the actor.
- Low sampled Bellman error does not guarantee accurate values on newly visited states.

*What is the unit of replication?*

- Training runs for claims about learning reliability.
- Repeated episodes from one frozen policy estimate only that policy's evaluation uncertainty.
- Do not treat correlated episodes as independent training runs.

*Must evaluation disable all randomness?*

- Evaluate the intended deployment policy.
- Greedy evaluation fits epsilon-greedy control; replacing a stochastic policy with its mean creates a different policy.
- Freeze learning & exploration settings consistently; never select the best seed after seeing test returns.
```

&nbsp;
