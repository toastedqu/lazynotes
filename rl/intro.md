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

Sequential decision-making: objective → consequences → behavior. Study notes from {cite:t}`rlbook`.

Notations:
- $s_t$: State before decision $t$.
- $\mathcal{S}$: State space.
- $a_t$: Action at decision $t$.
- $\mathcal{A}$: Action space.
- $r_{t+1}$: Reward received after $a_t$.
- $p(s',r|s,a)$: Joint next-state/reward kernel.
- $p(s'|s,a)$: Next-state transition distribution.
- $r(s,a)$: Expected immediate reward given $(s,a)$.
- $\rho_0$: Initial-state distribution.
- $\pi(a|s)$: Policy.
- $\gamma$: Discount factor; $0\leq\gamma<1$ unless stated otherwise.
- $T$: Terminal time; actions run from $0$ through $T-1$.
- $G_t$: Discounted return from $t$.
- $V^\pi(s)$: State value under $\pi$.
- $Q^\pi(s,a)$: Action value under $\pi$.
- $J(\pi)$: Expected return from $\rho_0$.

Override: $a$ = action, not a generic scalar.

Override: $r$ = reward.

Override: $V,Q,A$ = scalar-valued functions, not matrices or random variables.

&nbsp;

## Decision Problems

### RL
- **Name**: Reinforcement Learning
- **What**: Learn behavior that maximizes expected cumulative reward.
- **Why**: No labeled best action; consequences arrive late.
- **How**:
    1. Observe → act → receive reward & next observation.
    2. Estimate consequences from experience.
    3. Improve behavior → collect different experience.

```{dropdown} Table: Course Map
| Page | Central question | Main tools |
|:--|:--|:--|
| Introduction | What do we optimize? | MDPs, returns, Bellman equations, exploration |
| [Value](value.md) | How good is each state/action? | DP, MC, TD, SARSA, Q-learning, DQN |
| [Policy](policy.md) | How should policy params change? | Policy gradients, actor–critic, PPO, SAC |
| [Model](model.md) | What happens if we act? | Learned dynamics, Dyna, search, MPC |
```

```{attention} Q&A
:class: dropdown
*How is RL different from supervised learning?*

- Feedback scores the chosen action only, ❌label for every action.
- Actions change later inputs → data distribution depends on the learner.
- Delayed reward → **temporal credit assignment**: find which earlier decision caused the outcome.

*Must data be online or non-i.i.d.?*

- No. Offline RL learns from a fixed dataset.
- Interaction usually gives correlated trajectories; replay reduces correlation.
- I.i.d. sampled transitions still serve a sequential objective.

*Prediction vs. control?*

- **Prediction**: Evaluate a given policy.
- **Control**: Find a better policy.

*Can every goal be written as a scalar reward?*

- **Reward hypothesis**: every goal = maximize expected cumulative scalar reward. Contested.
- Some tasks have no Markov reward: some sets of acceptable policies, policy orderings & trajectory orderings {cite:p}`abel2021expressivity`.
    - e.g., "always move in the same direction" in a grid w/ state $(x,y)$.
- Hard constraints, risk preferences & multiple objectives need extended formulations.

*Does reward maximization imply consciousness?*

- No. Reward maximization is an optimization criterion, not a theory of subjective experience.
- A capable controller can exploit narrow regularities w/o a general world model.
```

&nbsp;

### MDP
- **Name**: Markov Decision Process
- **What**: Sequential decision problem w/ a fully observed Markov state.
- **Why**: Tractable recursion.
    - Full history grows w/o bound.
    - Markov state → future depends only on curr state & action → one-step recursion (Bellman).
- **How**: Agent sees state → picks action → environment samples next state & reward.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\mathcal{M}$: Decision problem, incl. its objective.

$$
\mathcal{M}=(\mathcal{S},\mathcal{A},p,r,\rho_0,\gamma),\qquad
r(s,a)=\mathbb{E}[r_{t+1}|s_t=s,a_t=a]
$$

- Joint kernel $p(s',r|s,a)$ allows stochastic rewards.
- Expected-return Bellman equations need only $p(s'|s,a)$ & $r(s,a)$.

$$
\Pr(s_{t+1},r_{t+1}|s_0,a_0,\ldots,s_t,a_t)
=p(s_{t+1},r_{t+1}|s_t,a_t)
$$

- **Markov property**: Given the action, curr state screens off history.
- **Stationarity**: Kernel does not change with $t$. Separate assumption.
- Finite horizon → add time-to-go to the state, or use time-indexed values & policies.
- Terminal state → absorbing, all later rewards $0$.
```

```{note} Example
:class: dropdown
- Robot position alone: same position, different velocity → different next position under the same action.
- Position + velocity can be Markov for the chosen dynamics.
- Hidden battery wear or another adapting agent breaks that state description.
```

```{attention} Q&A
:class: dropdown
*Does Markov mean deterministic or memoryless behavior?*

- No. Stochastic transitions are allowed.
- The environment needs no history beyond a sufficient state. An agent can use memory to build that state.

*How do you make a non-Markov problem Markov?*

- Add the missing info to the state: velocity, time-to-go, recent frames, recurrent memory.
- Full history is always Markov, but it grows w/o bound → summarize it.

*Is the MDP known?*

- The problem exists whether or not the agent knows the kernel.
- Known kernel → [planning](model.md#model-based-rl).
- Unknown kernel → learn values/policies directly (model-free) or learn a model (model-based).

*Is a finite action set required?*

- No. Integrals replace sums for continuous spaces; maximization & approximation get harder.
- Tabular convergence results do not transfer automatically to continuous spaces or neural approximators.
```

&nbsp;

### POMDP
- **Name**: Partially Observable Markov Decision Process
- **What**: MDP whose latent state is seen only through observations.
- **Why**: Sensors hide variables needed to predict consequences.
- **How**: Track a belief over states from history → act on the belief.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $o_t$: Observation at $t$.
- Misc:
    - $O(o'|s',a)$: Observation likelihood after action $a$ leads to $s'$.
    - $h_t$: Observation/action/reward history up to $t$.
    - $b_t(s)$: Belief $\Pr(s_t=s|h_t)$.

$$
b_{t+1}(s')
=\frac{O(o_{t+1}|s',a_t)\sum_s p(s'|s,a_t)b_t(s)}
{\sum_{\bar s}O(o_{t+1}|\bar s,a_t)\sum_s p(\bar s|s,a_t)b_t(s)}
$$

- This form assumes the observation carries all state info. If reward reveals state, add the reward likelihood.
- Correct model + exact Bayes update → belief is a sufficient statistic of history.
- Policy $\pi(a|b)$ acts in a fully observed **belief MDP**.
```

```{attention} Q&A
:class: dropdown
*Why not treat the latest observation as the state?*

- Same observation can hide different states → different optimal actions.
- Frame stacks cover short-term missing info; recurrent policies summarize longer history.
- Neither guarantees a sufficient state.

*Does random action choice solve partial observability?*

- No. Memory or belief tracking supplies the missing state info.
- Information-gathering actions can improve later decisions.

*Why is exact POMDP planning hard?*

- Belief MDP state = point on a probability simplex → continuous even for finite $\mathcal{S}$.
- Exact solutions scale badly → practice uses approximate beliefs, sampled beliefs or learned memory.
```

&nbsp;

### Policies & Trajectories
- **What**: Action-selection rule & the interaction sequence it induces.
- **Why**: Return depends on chosen actions & on the states those actions reach.
- **How**: Alternate policy draws w/ environment draws.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\tau$: Trajectory $(s_0,a_0,r_1,\ldots,s_T)$.
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

- Episodic task → absorbing terminal state defines the infinite sum.
- Stationary deterministic policy: one action per state.
- Stationary stochastic policy: one action distribution per state.
```

```{attention} Q&A
:class: dropdown
*Can an optimal policy be deterministic?*

- Yes for finite, fully observed, discounted MDPs w/ the standard expected-return objective.
- Finite horizon: optimal policy is generally time-dependent unless time is in the state.
- Entropy terms, limited memory, occupancy constraints or multi-agent games can make stochastic policies strictly better.

*Why care about the state distribution?*

- Changing action probabilities changes which states the agent visits.
- A policy can do well on training states, then reach unfamiliar states after an update.
- This dependency drives [policy gradients](policy.md) & distribution-shift problems.
```

&nbsp;

## Objectives & Consequences

### Rewards & Returns
- **What**: Immediate feedback & its discounted sum.
- **Why**: Locally attractive actions can sacrifice larger future reward.
- **How**: Define success as reward → sum discounted rewards over the horizon.

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

- Bounded rewards $|r_{t+1}|\leq R_{\max}$ & $\gamma<1$ → $|G_t|\leq R_{\max}/(1-\gamma)$.
- $R_{\max}$: Uniform reward-magnitude bound.
- $\gamma=1$ is valid for bounded finite-horizon rewards. Random-length episodes need integrable returns, not only eventual termination.

$$
J_{\mathrm{avg}}(\pi)=\lim_{H\to\infty}\frac{1}{H}
\mathbb{E}_\pi\!\left[\sum_{t=0}^{H-1}r_{t+1}\right]
$$

- $J_{\mathrm{avg}}$: Average-reward objective, if the limit exists.
- $H$: Number of decisions.
- Average reward ≠ undiscounted infinite sum. Recurrence assumptions decide whether it depends on the start state.
```

```{note} Example
:class: dropdown
- Rewards $(1,2,3)$ & $\gamma=\tfrac12$ → $G_0=1+1+\tfrac34=2.75$.
- Choice: terminate for $2$, or get $0$ then $5$ → delayed choice wins iff $5\gamma>2$.
```

````{important} Code
:class: dropdown
```python
import numpy as np

class DiscountedReturn:
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, rewards, bootstrap=0.0):
        ## Truncated rollout → bootstrap = value estimate of the last state; true terminal → 0.
        returns = np.empty(len(rewards), dtype=float)
        future = bootstrap
        ## Backward pass: G_t = r_{t+1} + gamma * G_{t+1}.
        for t in reversed(range(len(rewards))):
            future = rewards[t] + self.gamma * future
            returns[t] = future
        return returns

## Example
print(DiscountedReturn(0.5)([1, 2, 3]))  ## [2.75 3.5  3.  ]
```
````

```{attention} Q&A
:class: dropdown
*What does discounting change?*

- $\gamma=0$ → immediate expected reward only. $\gamma\to1$ → more weight on distant outcomes.
- $1/(1-\gamma)$ = sum of discount weights, not a hard horizon.
- Discounting changes the objective. It is not only a numerical stabilizer.

*Why discount at all?*

- Bounded infinite sums in continuing tasks.
- $\gamma$ = per-step survival probability: termination w/ prob $1-\gamma$ per step, independent of the trajectory → expected undiscounted return = discounted return.
- Preference for earlier reward; lower variance of long-horizon targets.

*Termination vs. truncation?*

- **Termination**: Task ends → no bootstrap.
- **Truncation**: Collection stops, task could continue → bootstrap from the last pre-reset state.
- A real finite-horizon deadline is termination; put time-to-go in the state.
- Never bootstrap from a reset observation from an auto-reset wrapper.

*Which reward transformations keep the optimal policy?*

- Positive scaling: ✅.
- Constant shift per step: ✅ for fixed-length episodes or continuing discounted tasks.
- Variable episode length: ❌ constant shift. A living bonus rewards delaying termination; a living cost rewards ending early.
- Clipping or arbitrary dense bonuses: ❌ can change the task.
- Potential-based shaping: ✅ (next section).

*Sparse vs. dense rewards?*

- Sparse: specifies success directly; discovery & credit assignment are hard.
- Dense: guides learning; can favor proxy behavior over the intended outcome (**reward hacking**).
- Expected-return maximization is risk-neutral → ❌tail-risk or safety constraints.
```

&nbsp;

#### Potential-Based Reward Shaping
- **What**: Reward plus a discounted potential difference {cite:p}`ng1999policy`
- **Why**: Dense guidance w/o changing the optimal policy.
- **How**: Reward moves toward high-potential states → bonuses telescope → total bonus depends only on start & end states.

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
- Episodic tasks → set terminal potentials to $0$.
- Then $Q'^\pi(s,a)=Q^\pi(s,a)-\Phi(s)$ → same maximizing actions.
```

````{important} Code
:class: dropdown
```python
class PotentialShaping:
    def __init__(self, potential, gamma):
        self.potential = potential
        self.gamma = gamma

    def __call__(self, s, reward, next_s, terminated):
        ## Terminal potential = 0 → bonuses telescope to -Phi(s_0).
        next_phi = 0.0 if terminated else self.potential(next_s)
        return reward + self.gamma * next_phi - self.potential(s)

## Example
shape = PotentialShaping(lambda distance: -distance, gamma=0.9)
print(shape(2, 0, 1, False))  ## 1.1: one step closer to the goal
```
````

```{attention} Q&A
:class: dropdown
*Does any progress bonus work?*

- No. Only the discounted difference telescopes.
- Nonzero terminal potentials can change episode rankings.
- Mismatched discount → bonus can reward cycles.

*Why potential-based specifically?*

- Without knowledge of the MDP, potential-based form is the only shaping that guarantees an unchanged optimal policy.

*What is the ideal potential?*

- $\Phi=V^*$ → shaped $Q'^*(s,a)=Q^*(s,a)-V^*(s)\leq0$, equal to $0$ only for optimal actions.
- → Greedy one-step choice is optimal: long-horizon credit assignment disappears.

*Does shaping change learning or only the fixed point?*

- It changes learning speed, not the optimal policy.
- Same experience → shaped Q-learning from $Q_0$ & unshaped Q-learning from $Q_0+\Phi$ make identical updates {cite:p}`wiewiora2003potential`.
- Behavior also matches for advantage-based policies (e.g., greedy, Boltzmann): $\Phi(s)$ shifts all actions equally.
- Bad potential → slower learning, not a wrong optimum.
```

&nbsp;

### Value Functions
- **What**: Expected return from a state or state–action pair under a policy.
- **Why**: Rank actions by long-run consequences w/o enumerating every future.
- **How**: Average future returns, given the policy for all later decisions.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $A^\pi(s,a)$: Advantage under $\pi$.

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

$$
\pi^*(a|s)>0\Rightarrow a\in\arg\max_{a'}Q^*(s,a')
$$

- $\pi^*$: Optimal policy. In finite discounted MDPs, any policy greedy w.r.t. $Q^*$ is optimal from every start state.
- Values depend on reward, discount & horizon. Finite-horizon values generally depend on time.
```

```{note} Example
:class: dropdown
- $Q^\pi(s,a_1)=2$, $Q^\pi(s,a_2)=6$ & $\pi=(\tfrac34,\tfrac14)$.
- $V^\pi(s)=\tfrac34\cdot2+\tfrac14\cdot6=3$ → advantages $(-1,3)$ → policy-weighted mean $0$.
- The best action need not have a positive absolute value; it needs the largest $Q$.
```

```{attention} Q&A
:class: dropdown
*Why learn Q instead of V?*

- $Q$ ranks actions directly → act greedily w/o a model.
- $V$ needs a model for lookahead, or a separate policy for acting.

*Is Q the immediate reward?*

- No. $Q$ = curr reward + consequences under the continuation policy.
- $Q^\pi$ follows $\pi$ after the first action, even if that action differs from $\pi$.

*Why use the advantage?*

- Removes the state-dependent baseline $V^\pi(s)$ → only the relative action quality remains.
- Lower-variance [policy gradients](policy.md) w/o bias.

*Does a value estimate give uncertainty?*

- No. An expected-return estimate is neither a return distribution nor a confidence interval.
- A high value can be accurate or an extrapolation error.
```

&nbsp;

### Bellman Equations
- **What**: Recursive consistency conditions for expected return.
- **Why**: Split long-horizon reasoning into one-step predictions.
- **How**: Value = immediate reward + discounted next-state value; average (policy) or maximize (optimal) over the next action.

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

- Finite spaces shown. Integrals & suprema replace sums & maxima when needed.
- Terminal continuation values are $0$.

$$
\begin{align*}
\mathcal{T}^\pi\mathbf{v}&=\mathbf{r}^\pi+\gamma P^\pi\mathbf{v},\\
\|\mathcal{T}^\pi\mathbf{u}-\mathcal{T}^\pi\mathbf{v}\|_\infty
&\leq\gamma\|\mathbf{u}-\mathbf{v}\|_\infty,\\
\|\mathcal{T}^*\mathbf{u}-\mathcal{T}^*\mathbf{v}\|_\infty
&\leq\gamma\|\mathbf{u}-\mathbf{v}\|_\infty.
\end{align*}
$$

- $\mathcal{T}^\pi$: Bellman expectation operator.
- $\mathcal{T}^*$: Bellman optimality operator.
- $\mathbf{v}\in\mathbb{R}^{|\mathcal{S}|}$: Candidate state-value vector.
- $\mathbf{u}\in\mathbb{R}^{|\mathcal{S}|}$: Another candidate state-value vector.
- $\mathbf{r}^\pi\in\mathbb{R}^{|\mathcal{S}|}$: Expected immediate reward per state under $\pi$.
- $P^\pi\in\mathbb{R}^{|\mathcal{S}|\times|\mathcal{S}|}$: Policy-induced transition matrix.
- $\|\cdot\|_\infty$: Largest absolute statewise error.

$$
\mathbf{v}^\pi=(I-\gamma P^\pi)^{-1}\mathbf{r}^\pi
$$

- $\mathbf{v}^\pi$: Vector of $V^\pi(s)$ over all states.
- $I$: Identity matrix.
- $\gamma<1$ & $P^\pi$ row-stochastic → $I-\gamma P^\pi$ is invertible.
```

```{tip} Derivation
:class: dropdown
1. Substitute $G_t=r_{t+1}+\gamma G_{t+1}$ into the value definition.
2. Condition on the first action & the next state.
3. Markov property + stationary policy → remaining expectation is $V^\pi(s_{t+1})$.
4. Optimal continuation → replace the policy-weighted action average w/ a maximum.
5. An expectation cannot enlarge a uniform error. Also $|\max_a f(a)-\max_a g(a)|\leq\max_a|f(a)-g(a)|$.
6. Multiply by $\gamma<1$ → contraction → unique fixed point; repeated exact backups converge to it (Banach).
```

```{attention} Q&A
:class: dropdown
*Equation vs. algorithm?*

- Bellman equations characterize values.
- [Dynamic programming](value.md#dynamic-programming) solves them w/ a known model.
- [TD learning](value.md#td-learning) approximates backups from sampled transitions.

*Why does policy evaluation have a closed form but control does not?*

- Expectation equation is linear in $\mathbf{v}$ → one linear solve, $O(|\mathcal{S}|^3)$.
- Optimality equation has a $\max$ → nonlinear → iterate (value iteration, policy iteration).

*Why is the max inside the next-state expectation for $Q^*$?*

- The next action can depend on the next state actually reached.
- A max outside the expectation commits to one next action before seeing that state.

*When does the contraction argument fail?*

- $\gamma=1$: no strict sup-norm contraction in general. Finite-horizon backward induction still works.
- Neural projection or optimization: the full update is not the exact Bellman operator.
- Off-policy sampling + approximation + bootstrapping can diverge ([deadly triad](value.md#deadly-triad)).
```

&nbsp;

## Exploration & Data

### Exploration & Exploitation
- **What**: Trade-off: gather information vs. use the curr best estimate.
- **Why**: Rewards of chosen actions say nothing about actions never tried.
- **How**: Spend some decisions on reducing uncertainty; exploit evidence on the rest.

```{dropdown} Table: Exploration Mechanisms
| Mechanism | Principle | Limitation |
|:--|:--|:--|
| $\epsilon$-greedy | Sometimes choose uniformly | Ignores which actions are uncertain |
| Boltzmann | Sample $\propto\exp(\beta\hat Q)$ | Ignores uncertainty; scale-sensitive |
| Optimistic initialization | Unvisited actions start attractive | Optimism can fade before discovery |
| Upper confidence bound | Value + uncertainty bonus | Bandit bonus alone ❌deep exploration |
| Thompson sampling | Sample a plausible model, act greedily | Needs useful posterior uncertainty |
| Entropy regularization | Reward policy diversity | Diversity ≠ calibrated uncertainty |
| Intrinsic reward | Reward novelty/information | Can chase stochastic distractions or alter the task |
```

```{attention} Q&A
:class: dropdown
*Why can random action noise fail?*

- Sparse reward can need a long, coherent sequence of individually unattractive actions.
- Per-step noise rarely produces that sequence.
- **Deep exploration**: keep one hypothesis long enough to reach informative states.

*Exploration vs. credit assignment?*

- Exploration discovers rewarding trajectories.
- Credit assignment finds which earlier decisions caused the reward.
- Better return estimators do not discover an unseen reward.
```

&nbsp;

#### Epsilon-Greedy
- **What**: Greedy action w/ prob $1-\epsilon$; uniform random action w/ prob $\epsilon$.
- **Why**: Keep every discrete action reachable while favoring estimated value.
- **How**: Flip an exploration coin. Heads → random action. Tails → best estimated action, random tie-break.

```{note} Math
:class: dropdown
$$
\pi(a|s)=\frac{\epsilon}{|\mathcal{A}|}
+(1-\epsilon)\frac{\mathbf{1}[a\in\mathcal{A}^*(s)]}{|\mathcal{A}^*(s)|}
$$

- $\epsilon$: Exploration probability.
- $\mathcal{A}^*(s)$: Actions that maximize the curr value estimate.
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
        ## Explore: uniform over all actions, incl. the greedy one.
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(len(q)))
        ## Random ties avoid always preferring the first action.
        return int(self.rng.choice(np.flatnonzero(q == q.max())))

## Example
print(EpsilonGreedy(0.0)([1.0, 3.0]))  ## 1
```
````

```{attention} Q&A
:class: dropdown
*Pros?*

- Simple; one hyperparam; works w/ any value estimate.

*Cons?*

- Undirected: explores a terrible action as often as a nearly-best one.
- Ignores uncertainty: a well-tested action & an untested action get the same exploration.
- Combination-lock tasks → #episodes to reach the reward can grow exponentially in lock length.
- ❌Continuous actions w/o a discretization.

*Does decaying epsilon guarantee convergence?*

- No. Every needed state–action pair must still get enough visits.
- **GLIE**: Greedy in the Limit w/ Infinite Exploration; both parts matter.
- Decay too fast → a good action can be missed forever.
```

&nbsp;

#### Boltzmann Exploration
- **What**: Softmax over action-value estimates w/ a temperature.
- **Why**: $\epsilon$-greedy explores bad & near-best actions equally.
- **How**: Higher estimated value → higher probability; temperature sets greediness.

```{note} Math
:class: dropdown
$$
\pi(a|s)=\frac{\exp(\beta\hat Q(s,a))}{\sum_{a'}\exp(\beta\hat Q(s,a'))}
$$

- $\hat Q(s,a)$: Curr action-value estimate.
- $\beta>0$: Inverse temperature.
- $\beta\to\infty$ → greedy (uniform over ties).
- $\beta\to0$ → uniform random.
```

````{important} Code
:class: dropdown
```python
import numpy as np

class Boltzmann:
    def __init__(self, beta, seed=0):
        self.beta = beta
        self.rng = np.random.default_rng(seed)

    def probs(self, q):
        z = self.beta * np.asarray(q, dtype=float)
        ## Subtract max → no overflow; softmax ignores a constant shift.
        w = np.exp(z - z.max())
        return w / w.sum()

    def __call__(self, q):
        p = self.probs(q)
        return int(self.rng.choice(len(p), p=p))

## Example
print(Boltzmann(1.0).probs([1.0, 3.0]).round(3))  ## [0.119 0.881]
```
````

```{attention} Q&A
:class: dropdown
*Pros?*

- Graded exploration: near-best actions get tried more than clearly bad ones.
- Smooth in $\hat Q$ → differentiable policy.

*Cons?*

- Scale-sensitive: same $\beta$ acts differently when rewards are rescaled.
- Ignores uncertainty: two actions w/ equal estimates get equal probability, whatever their sample counts.
- Needs a temperature schedule; bad schedule → over-exploration or early lock-in.

*How does it relate to maximum-entropy RL?*

- Softmax over $Q$ = the optimal policy form under an entropy-regularized objective ([maximum-entropy RL](policy.md#maximum-entropy-rl)).
- There, the soft $Q$ includes future entropy; here, $\hat Q$ is an ordinary estimate.
```

&nbsp;

### Bandits
- **What**: Decision problem where actions earn reward but do not change future states.
- **Why**: Isolate exploration from temporal credit assignment.
- **How**: Pull an arm → observe its reward → update its estimate.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $q(a)$: Mean reward of arm $a$.
    - $q_*=\max_a q(a)$: Best arm mean.
    - $\Delta_a=q_*-q(a)$: Gap of arm $a$.
    - $N_H(a)$: Pulls of arm $a$ in $H$ decisions.

$$
\mathcal{R}_H=Hq_*-\mathbb{E}\!\left[\sum_{t=0}^{H-1}r_{t+1}\right]
=\sum_a\Delta_a\,\mathbb{E}[N_H(a)]
$$

- $\mathcal{R}_H$: Expected cumulative regret over $H$ decisions.
- $H$: Decision budget.
- Low regret ⇔ few pulls of large-gap arms.
```

```{attention} Q&A
:class: dropdown
*What regret is achievable?*

- Pure greedy: can lock onto a bad arm → linear in $H$.
- Fixed $\epsilon$: constant exploration rate → linear in $H$.
- Instance-dependent lower bound: any consistent algorithm has $\Omega(\log H)$ regret.
- UCB & Thompson sampling reach logarithmic regret under standard assumptions.

*Sample average vs. constant step size?*

- Sample average $\hat q\leftarrow\hat q+\frac1N(r-\hat q)$: unbiased for stationary arms.
- Constant $\alpha$: exponential recency weighting → tracks nonstationary arms; never fully converges.

*Contextual bandit vs. MDP?*

- Contextual bandit: sees a context before acting; action does not control later contexts.
- MDP: actions change later states → greedy immediate reward need not maximize return.

*Why not treat full trajectories as arms?*

- #action sequences grows combinatorially.
- Ignores shared subproblems & adaptation to intermediate observations.
```

&nbsp;

#### UCB
- **Name**: Upper Confidence Bound {cite:p}`auer2002finite`
- **What**: Greedy action selection on value estimate + uncertainty bonus.
- **Why**: Optimism under uncertainty: try arms that could still be the best.
- **How**: Rarely pulled arm → large bonus → gets tried. Bonus shrinks as pulls accumulate.

```{note} Math
:class: dropdown
Process:
1. Pull each arm once.
2. At each later decision $t$, pull

    $$
    a_t\in\arg\max_a\left[
    \hat q_t(a)+c\sqrt{\frac{\log t}{N_t(a)}}\right]
    $$

3. Update $N_t(a_t)$ & $\hat q_t(a_t)$ w/ the observed reward.

- $t$: Number of prior pulls over all arms.
- $\hat q_t(a)$: Empirical mean of prior pulls of $a$.
- $N_t(a)$: Number of prior pulls of $a$.
- $c$: Exploration coefficient.
- **UCB1**: rewards in $[0,1]$, $c=\sqrt{2}$, natural log → $O\!\left(\sum_{a:\Delta_a>0}\frac{\log H}{\Delta_a}\right)$ regret, uniformly over $H$.
```

````{important} Code
:class: dropdown
```python
import numpy as np

class UCB:
    def __init__(self, actions, coefficient=np.sqrt(2)):
        self.count = np.zeros(actions, dtype=int)
        self.mean = np.zeros(actions)
        self.coefficient = coefficient

    def __call__(self):
        ## Unpulled arms have infinite bonus → pull each once first.
        unseen = np.flatnonzero(self.count == 0)
        if len(unseen):
            return int(unseen[0])
        bonus = self.coefficient * np.sqrt(
            np.log(self.count.sum()) / self.count
        )
        return int(np.argmax(self.mean + bonus))

    def step(self, action, reward):
        ## Incremental sample mean.
        self.count[action] += 1
        self.mean[action] += (
            reward - self.mean[action]
        ) / self.count[action]

## Example
bandit = UCB(2)
bandit.step(0, 1.0)
print(bandit())  ## 1: still unpulled
```
````

```{attention} Q&A
:class: dropdown
*Pros?*

- Directed: explores where uncertainty is large.
- UCB1 needs no prior & no tuning for rewards in $[0,1]$; logarithmic regret.
- Deterministic → reproducible.

*Cons?*

- Bonus needs a known reward range or tail bound.
- Deterministic → batched or delayed feedback repeats the same arm.
- Neural $Q$ estimates do not inherit count-based confidence guarantees.

*What assumptions matter?*

- Stationary rewards; bounded or sub-Gaussian tails.
- Nonstationarity → forgetting or change detection.

*Where else does UCB appear?*

- Tree search: UCB at each node → [UCT](model.md#mcts-with-uct).
```

&nbsp;

#### Thompson Sampling
- **What**: Act greedily on one posterior sample of arm means {cite:p}`thompson1933likelihood`
- **Why**: Explore each arm in proportion to its probability of being the best.
- **How**: Uncertain arm → wide posterior → sometimes sampled high → gets tried. Posterior narrows w/ evidence.

```{note} Math
:class: dropdown
Process:
1. Prior $\operatorname{Beta}(1,1)$ (uniform) per arm.
2. Sample $\tilde q(a)\sim\operatorname{Beta}(u_a,v_a)$ for every arm.
3. Pull $a_t=\arg\max_a\tilde q(a)$.
4. Observe $r\in\{0,1\}$ → $u_{a_t}\leftarrow u_{a_t}+r$, $v_{a_t}\leftarrow v_{a_t}+1-r$.

- $\tilde q(a)$: Sampled Bernoulli success probability.
- $u_a$: Prior success param + observed successes.
- $v_a$: Prior failure param + observed failures.
- **Probability matching**: $\Pr(a_t=a)=\Pr(a\text{ is best}\mid\text{data})$.
```

````{important} Code
:class: dropdown
```python
import numpy as np

class ThompsonSampling:
    def __init__(self, actions, seed=0):
        ## Beta(1, 1) = uniform prior on each arm's success rate.
        self.success = np.ones(actions)
        self.failure = np.ones(actions)
        self.rng = np.random.default_rng(seed)

    def __call__(self):
        ## One posterior draw per arm → act greedily on the draw.
        return int(np.argmax(self.rng.beta(self.success, self.failure)))

    def step(self, action, reward):
        if reward not in (0, 1):
            raise ValueError("Beta-Bernoulli updates need reward 0 or 1")
        self.success[action] += reward
        self.failure[action] += 1 - reward

## Example
bandit = ThompsonSampling(2)
bandit.step(0, 1)
print(bandit.success)  ## [2. 1.]
```
````

```{attention} Q&A
:class: dropdown
*Pros?*

- Bernoulli arms, uniform prior → asymptotically matches the Lai–Robbins lower bound {cite:p}`kaufmann2012thompson`.
- Randomized → handles batched & delayed feedback well.
- Prior knowledge enters naturally.

*Cons?*

- Needs a prior & a tractable posterior; deep RL needs approximations.
- Misspecified model → confidently wrong exploration.

*Is posterior sampling the same as adding arbitrary noise?*

- No. Posterior spread shrinks as evidence grows. Fixed noise does not.
```

&nbsp;

### On-Policy & Off-Policy Learning
- **What**: Learn about the data-generating policy vs. a different target policy.
- **Why**: Separate the policy being evaluated/improved from the source of experience.
- **How**: Collect w/ the target, correct the distribution mismatch, or use an off-policy backup.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\mu(a|s)$: Behavior policy that generates the data.

$$
\pi=\mu\quad\text{(on-policy)},\qquad
\pi\neq\mu\quad\text{(off-policy)}
$$

- Importance-sampling coverage: $\pi(a|s)>0\Rightarrow\mu(a|s)>0$ at relevant states.
- [Importance sampling](value.md#importance-sampling) corrects likelihoods. Arbitrary replay does not make an estimator unbiased.
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
*Pros & cons of on-policy?*

- ✅ Simple, stable; data matches the policy being improved.
- ❌ Sample-inefficient: data goes stale after a few updates → discarded (PPO reuses each batch for only a few epochs).

*Pros & cons of off-policy?*

- ✅ Reuses replay, demonstrations & logs; can explore w/ one policy & learn another.
- ❌ Distribution mismatch → high-variance corrections or instability w/ function approximation.

*Why can Q-learning use exploratory actions w/o importance ratios?*

- A transition at $(s,a)$ is evidence about that conditional environment transition.
- Its target maximizes over the next action instead of averaging over $\mu$.
- Tabular coverage & step-size conditions still apply.

*Is replay always compatible w/ policy gradients?*

- No. A naive likelihood-ratio gradient on old data uses the wrong trajectory distribution.
- PPO reuses a recent rollout w/ a clipped local surrogate. SAC uses an off-policy critic objective.

*Why is offline RL harder than replay-based online RL?*

- No new interaction to correct values of unsupported actions.
- Greedy improvement exploits extrapolation error outside the dataset.
- Behavior regularization or pessimistic values limit unsupported improvement. Logged data alone cannot identify unseen consequences.
```

&nbsp;

### Experimental Evaluation
- **What**: Estimate deployed-policy performance & learning efficiency.
- **Why**: Training loss, exploratory return & cherry-picked runs misrepresent control quality.
- **How**:
    1. Fix task version, reward, horizon & deployment policy.
    2. Train independent runs. Evaluate frozen policies on held-out episodes.
    3. Report return distribution, environment interactions & compute budget.
    4. Separate tuning from final evaluation.

```{attention} Q&A
:class: dropdown
*Which measurements answer which question?*

- **Return**: Task success under the evaluation protocol.
- **Sample efficiency**: Return vs. real environment interactions. Count reused & model-generated transitions separately.
- **Compute efficiency**: Return vs. wall-clock or compute budget.
- **Reliability**: Variation across training seeds, tasks & evaluation episodes.
- **Safety**: Constraint violations & tail outcomes, not only the mean.

*Why can critic loss fall while return gets worse?*

- Targets move, coverage changes, or value extrapolation misleads the actor.
- Low sampled Bellman error ≠ accurate values on newly visited states.

*What is the unit of replication?*

- Training runs, for claims about learning reliability.
- Repeated episodes of one frozen policy measure only that policy's evaluation noise.
- Do not treat correlated episodes as independent training runs.

*How to aggregate a few runs over many tasks?*

- Point estimates (mean/median) from 3–10 runs are unreliable.
- Report interquartile mean (IQM) w/ stratified bootstrap confidence intervals {cite:p}`agarwal2021deep`.
- IQM: robust to outlier tasks & runs; smaller CIs than the median.

*Must evaluation disable all randomness?*

- Evaluate the intended deployment policy.
- Greedy evaluation fits epsilon-greedy control. Replacing a stochastic policy w/ its mean creates a different policy.
- Freeze learning & exploration consistently. Never select the best seed after seeing test returns.
```

&nbsp;
