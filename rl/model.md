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
# Model-Based RL & Planning

Study notes from {cite:t}`rlbook`: environment models → simulated learning, search & receding-horizon control. [MDPs](intro.md#mdp) define the task; [dynamic programming](value.md#dynamic-programming) covers full-state Bellman sweeps.

Notations:
- $s_t$: State before action $t$.
- $a_t$: Action at time $t$.
- $r_{t+1}$: Reward received after $a_t$.
- $p(s'|s,a)$: Environment transition distribution.
- $r(s,a)$: Expected immediate reward $\mathbb{E}[r_{t+1}|s_t=s,a_t=a]$.
- $\gamma\in[0,1)$: Discount factor.
- $\pi(a|s)$: Policy.
- $V^\pi(s)$: Expected return under $\pi$.
- $Q(s,a)$: Action-value estimate.
- $\mathcal{A}(s)$: Legal actions at state $s$.
- $\mathcal{D}$: Real transition dataset.
- $\psi$: Learned model params.
- $\hat M_\psi$: Learned joint simulator for next state, reward & termination.
- $d_t$: True-termination indicator for the transition after $a_t$.
- $c_t$: Episode/rollout boundary indicator, including artificial cutoffs.
- $H$: Planning/model-rollout horizon (#model steps); ≠ episode length.
- $\hat V(s)$: Leaf value estimate at a nonterminal planning cutoff.
- $o_t$: Observation at time $t$.
- $z_t$: Learned latent state at time $t$.

Override: $a$ denotes an RL action, possibly vector-valued, rather than a generic scalar.

&nbsp;

## Foundations

### Model-Based RL
- **Name**: Model-Based Reinforcement Learning
- **What**: Control via action-conditioned predictions of future outcomes.
- **Why**: Real interaction is costly → plan/learn from hypothetical experience instead.
- **How**:
    1. Get a model: known rules/physics/simulator, learned predictor, or both.
    2. Use it: improve values/policies, search from the curr state, or optimize action sequences.
    3. Act in the real env → update learned parts from the real outcome.

```{dropdown} Table: Model Usage
| Approach | Model access | Computation | Deployed action selection |
|:--|:--|:--|:--|
| Model-free Q-learning / actor-critic | No predictive environment model required | Learn from real or recorded transitions | Value maximization / actor |
| Dynamic programming | Explicit transition probabilities & rewards | Expected backups across states | Stored policy / value maximization |
| Dyna-Q | Learned simulator | Real + simulated value updates | Value maximization |
| Expectimax | Enumerable model outcomes | Decision/chance lookahead | Root action |
| MCTS with UCT | Generative simulator | Selective sampled lookahead | Root visit counts |
| MPC / PETS | Known / learned dynamics | Optimize a finite action sequence | First action, then replan |
| MBPO | Learned probabilistic ensemble | Short simulated data → actor-critic | Actor; ❌online search |
| Dreamer | Learned latent dynamics | Backprop value grads through imagined latent rollouts | Actor; ❌online search |
| AlphaZero / MuZero | Known rules / learned latent dynamics | Search + policy/value learning | Search, optionally distilled policy |
```

```{attention} Q&A
:class: dropdown
*Planning vs learning?*

- **Planning**: Model → better decisions. ❌Learning required.
- **Learning**: Experience → better estimates. ❌Model required.
- Model-based RL can combine model learning, planning & direct value/policy learning.

*Does training in a simulator make an algorithm model-based?*

- ❌Not automatically. Simulator used only as the env → still model-free.
- Model-based = query hypothetical outcomes for planning or synthetic updates.
- Generative model → start from any chosen state w/o real interaction.
- Exact game simulator ≠ accurate model of the physical system it imitates.

*What counts as a model?*

- Probability table, sample generator, deterministic physics, or action-conditioned latent dynamics.
- ❌Value function alone ← predicts return, ❌counterfactual transitions.
- ❌Differentiability, ❌observation reconstruction, ❌explicit density required.

*Pros?*

- Dynamics reusable across policies & reward functions.
- Extra compute → fewer real interactions.

*Cons?*

- ❌Universal sample-efficiency gain ← model error, fitting cost & planning cost can dominate.
- Task-specific latent models (e.g., MuZero) may not transfer to a new reward.
```

&nbsp;

## Learning Environment Models

### Transition, Reward & Termination Models
- **What**: Supervised predictors of action-conditioned env outcomes.
- **Why**: Planning needs consequences. A value estimate alone gives none.
- **How**:
    1. Collect $(s_t,a_t,r_{t+1},s_{t+1},d_t)$ over visited states/actions.
    2. Fit deterministic outcomes or conditional distributions.
    3. Query hypothetical transitions. Keep known reward/termination rules; ❌relearn them.
    4. Collect fresh real data where the curr controller operates.

```{note} Math
:class: dropdown
Model:

$$
(s',r',d)\sim\hat M_\psi(\cdot|s,a),\qquad
\hat M_\psi=\hat p_\psi(s'|s,a)\hat p_\psi(r',d|s,a,s')
$$

- $s'$: Next state.
- $r'$: Reward after the queried action.
- $d$: True termination after that action.
- $\hat p_\psi$: Learned conditional probability mass/density.

- Joint outcome model → keeps dependencies among next state, reward & termination.
- Separate reward & termination heads → assume conditional independence given their inputs. Add conditioning if it fails.

Training:

$$
\mathcal{L}_{\text{joint}}(\psi)
=-\frac1m\sum_{i=1}^{m}
\log\hat M_\psi(s_i',r_i',d_i|s_i,a_i)
$$

- Categorical next state → cross-entropy.
- Tabular maximum likelihood → empirical transition frequencies.
- Deterministic continuous next state/reward → squared error:

$$
\mathcal{L}_{\text{det}}
=\frac1m\sum_i
\left[
\|s_i'-f_\psi(s_i,a_i)\|_2^2
+\lambda_r(r_i'-\hat r_\psi(s_i,a_i,s_i'))^2
+\lambda_d\,\operatorname{BCE}(d_i,\hat q_\psi(s_i,a_i,s_i'))
\right]
$$

- $f_\psi$: Next-state predictor; often $s+\Delta_\psi(s,a)$.
- $\hat r_\psi$: Conditional mean reward predictor.
- $\hat q_\psi$: Conditional termination probability.
- $\lambda_r$: Reward-loss weight.
- $\lambda_d$: Termination-loss weight.

- Gaussian continuous next state → heteroskedastic negative log-likelihood:

$$
\mathcal{L}_{p,i}
=\frac12\sum_j
\left[
\frac{(s'_{ij}-f_{\psi,j}(s_i,a_i))^2}{\sigma_{\psi,j}^2(s_i,a_i)}
+\log\sigma_{\psi,j}^2(s_i,a_i)
\right]
$$

- $\sigma_{\psi,j}^2$: Predicted conditional variance for state coordinate $j$.
- Constant $\frac12\log(2\pi)$ per coordinate omitted.
- Stochastic reward → analogous scalar Gaussian loss, or a categorical/distributional model.
- Fixed Gaussian variance → squared error. Learned variance → also pays the log-variance term.

$$
\operatorname{BCE}(d,q)=-d\log q-(1-d)\log(1-q)
$$

- $q$: Predicted probability of true termination.
- Time limit / rollout cutoff → $c_t=1$ w/ $d_t=0$ possible.
```

````{important} Code
:class: dropdown
```python
import torch
from torch import nn
from torch.nn import functional as F

class EnvironmentModel(nn.Module):
    def __init__(self, state_dim, action_dim, width=32):
        super().__init__()
        self.transition = nn.Sequential(
            nn.Linear(state_dim + action_dim, width), nn.Tanh(),
            nn.Linear(width, 2 * state_dim))
        self.outcome = nn.Sequential(
            nn.Linear(2 * state_dim + action_dim, width), nn.Tanh(),
            nn.Linear(width, 3))

    def forward(self, s, a, next_state=None):
        delta, raw_var = self.transition(torch.cat((s, a), -1)).chunk(2, -1)
        mean = s + delta
        variance = F.softplus(raw_var) + 1e-6
        sp = mean if next_state is None else next_state
        reward, raw_reward_var, done_logit = self.outcome(
            torch.cat((s, a, sp), -1)).unbind(-1)
        return mean, variance, reward, F.softplus(raw_reward_var) + 1e-6, done_logit

    def loss(self, s, a, sp, reward, terminated, stochastic=True):
        ## Train outcome heads on the observed next state, not a free-running prediction.
        mean, var, rmean, rvar, logit = self(s, a, sp)
        state_error, reward_error = (sp - mean).square(), (reward - rmean).square()
        if stochastic:
            state_loss = 0.5 * (state_error / var + var.log()).sum(-1)
            reward_loss = 0.5 * (reward_error / rvar + rvar.log())
        else:
            state_loss, reward_loss = state_error.sum(-1), reward_error
        return (state_loss + reward_loss).mean() + F.binary_cross_entropy_with_logits(
            logit, terminated.float())

    @torch.no_grad()
    def sample(self, s, a, stochastic=True):
        mean, var, *_ = self(s, a)
        sp = mean + var.sqrt() * torch.randn_like(mean) if stochastic else mean
        _, _, rmean, rvar, logit = self(s, a, sp)
        reward = rmean + rvar.sqrt() * torch.randn_like(rmean) if stochastic else rmean
        done = torch.bernoulli(logit.sigmoid()).bool()
        return sp, reward, done

## Example: batches (m, state_dim), (m, action_dim); scalar targets (m,).
model = EnvironmentModel(2, 1)
s, a = torch.zeros(2, 2), torch.ones(2, 1)
loss = model.loss(s, a, s + 0.1, torch.ones(2), torch.tensor([False, True]))
loss.backward()
assert model.sample(s, a)[0].shape == (2, 2)
```
````

```{attention} Q&A
:class: dropdown
*When is deterministic regression wrong?*

- Squared error → conditional mean, ❌representative sampled outcome.
- 2 possible next positions on opposite sides of an obstacle → their mean can lie inside it.
- Gaussian head → state-dependent spread, ❌arbitrary multimodality. Use mixtures, categorical outcomes or richer generators.
- Code: `stochastic=False` → deterministic state & reward. Termination stays Bernoulli. Variance heads then have ❌statistical meaning.

*Must rewards be sampled?*

- Risk-neutral expected return + Markov state → correct conditional mean reward suffices.
- Risk-sensitive criteria or reward-dependent state inference → need more than the mean.
- Reward & termination dependent given $(s,a,s')$ → code's separate heads cannot reproduce their joint.

*Why learn termination?*

- Missing termination → fictitious future reward after failure/success.
- Premature termination → suppressed future value.
- Use true termination labels. ❌Generic `done` flags that merge termination & truncation.
- Autoreset envs → store the final observation, ❌the next episode's reset observation.
- True finite-horizon task → remaining time belongs in the state. Its endpoint = task termination, ❌collector cutoff.

*What to check before trusting a model?*

- Held-out one-step likelihood/error, reward error & termination calibration.
- Free-running multi-step predictions from real states.
- Split correlated data by trajectory/time, ❌only shuffled adjacent transitions.
- Coverage of proposed actions, rare terminal events & constraints.
- Low avg prediction error ≠ good control.
- Partial observability → model a sufficient history/belief/latent state. ❌Treat an aliased observation as Markov.
```

&nbsp;

### Model Error & Uncertainty
- **What**: Mismatch between predicted & real consequences.
- **Why**: Planner optimizes model predictions, incl. their errors.
- **How**:
    1. Separate outcome randomness from uncertainty about the dynamics.
    2. Diagnose errors on states/actions the curr planner proposes.
    3. Restrict unsupported predictions. Refresh real data. Shorten untrusted rollouts.

```{note} Math
:class: dropdown
- **Compounding error**: fixed action seq, same start state, deterministic dynamics $L$-Lipschitz in state, one-step error $\le\varepsilon$:

$$
e_{h+1}\le L e_h+\varepsilon,\qquad
e_h\le
\begin{cases}
h\varepsilon,&L=1,\\
\varepsilon\frac{L^h-1}{L-1},&L\ne1.
\end{cases}
$$

- $e_h$: State-prediction error after $h$ simulated steps; $e_0=0$.
- $L\ge0$: Lipschitz constant of the true transition function in state.
- $\varepsilon$: Uniform one-step state error.
- Feedback actions → need a bound on closed-loop dynamics, ❌fixed-action dynamics only.

- **Simulation lemma**: finite discounted MDPs, same reward bound $|r|,|\hat r|\le R_{\max}$, termination = absorbing zero-reward state:

$$
\begin{align*}
\varepsilon_r&=\sup_{s,a}|r(s,a)-\hat r(s,a)|,\\
\varepsilon_p&=\sup_{s,a}\|p(\cdot|s,a)-\hat p(\cdot|s,a)\|_1,\\
\|V^\pi-\hat V^\pi\|_\infty
&\le
\frac{\varepsilon_r}{1-\gamma}
+\frac{\gamma R_{\max}\varepsilon_p}{(1-\gamma)^2}
\equiv\Delta.
\end{align*}
$$

- $R_{\max}$: Bound on absolute expected immediate rewards in both models.
- $\hat r$: Learned expected reward after marginalizing modeled outcomes.
- $\hat p$: Learned transition kernel including absorbing-state entry.
- $\varepsilon_r$: Worst-case reward error.
- $\varepsilon_p$: Worst-case transition error in $\ell_1$ (= 2 × total variation).
- $\hat V^\pi$: Policy value in the learned model; ≠ leaf approximator $\hat V$.
- $\Delta$: Uniform policy-value error bound.

$$
V^*(s)-V^{\hat\pi}(s)\le2\Delta
$$

- $V^*$: True optimal value.
- $\hat\pi$: Exactly optimal policy in the learned model.
- Approximate planning → add its model-space suboptimality.
- Dataset-avg errors ❌establish these uniform bounds.

- **Uncertainty decomposition**: ensemble prediction of 1 scalar output at fixed $(s,a)$:

$$
\operatorname{Var}(Z)
=\mathbb{E}_E[\operatorname{Var}(Z|E)]
+\operatorname{Var}_E(\mathbb{E}[Z|E])
$$

- $Z$: Predicted scalar outcome.
- $E$: Random ensemble-member index.
- 1st term: Within-model noise.
- 2nd term: Between-model disagreement.
```

```{tip} Derivation
:class: dropdown
1. Subtract the two policy Bellman equations:

    $$
    V^\pi-\hat V^\pi
    =(r^\pi-\hat r^\pi)
    +\gamma P^\pi(V^\pi-\hat V^\pi)
    +\gamma(P^\pi-\hat P^\pi)\hat V^\pi.
    $$

    - $r^\pi$: True reward vector averaged over policy actions.
    - $\hat r^\pi$: Model reward vector averaged over policy actions.
    - $P^\pi$: True policy-induced transition matrix.
    - $\hat P^\pi$: Model policy-induced transition matrix.

2. Take sup norms; use $\|\hat V^\pi\|_\infty\le R_{\max}/(1-\gamma)$ & rearrange:

    $$
    (1-\gamma)\|V^\pi-\hat V^\pi\|_\infty
    \le\varepsilon_r+\frac{\gamma\varepsilon_pR_{\max}}{1-\gamma}.
    $$

3. Split via the true-optimal policy $\pi^*$:

    $$
    V^*-V^{\hat\pi}
    =\underbrace{(V^{\pi^*}-\hat V^{\pi^*})}_{\le\Delta}
    +\underbrace{(\hat V^{\pi^*}-\hat V^{\hat\pi})}_{\le0}
    +\underbrace{(\hat V^{\hat\pi}-V^{\hat\pi})}_{\le\Delta}.
    $$

    - Middle term $\le0$ ← $\hat\pi$ is optimal in the learned model.
```

```{attention} Q&A
:class: dropdown
*Epistemic vs aleatoric uncertainty?*

- **Epistemic**: Uncertainty about dynamics/rewards ← limited data. Reducible w/ informative data.
- **Aleatoric**: Outcome randomness given the modeled info. Persists even w/ the true conditional distribution.
- Hidden state can look like randomness → richer state info can ⬇️ it.
- Ensemble disagreement = proxy, ❌calibrated Bayesian uncertainty. All members can share one bias.

*Why does small one-step error become a bad policy?*

- Prediction error → next input leaves the training distribution → later predictions unsupported.
- Optimization selects overestimated trajectories: false rewards, impossible shortcuts, missing termination.
- More search → exploits model errors more effectively.
- More simulations ⬇️ sampling error, ❌model bias.

*What helps?*

- Known reward/termination/constraint rules, recent real data, bootstrap ensembles, short horizons, conservative handling of unsupported actions.
- Penalize epistemic uncertainty when caution matters.
- High aleatoric variance alone ≠ evidence that more data helps.
- Short rollouts ⬇️ accumulated error, ❌first-step out-of-distribution error.
- Reward model correct on the data can still be exploited off the data.
- Wrong task reward = separate specification problem.
```

&nbsp;

#### PETS
- **Name**: Probabilistic Ensembles with Trajectory Sampling {cite:p}`chua2018deep`
- **What**: Probabilistic ensemble dynamics + particle-based MPC.
- **Why**: Mean trajectory → discards uncertainty relevant to action choice.
- **How**:
    1. Train probabilistic dynamics models on bootstrap resamples of real data.
    2. Propagate particles through candidate action sequences.
    3. Optimize mean predicted return w/ [CEM](#cem). Execute the 1st action → replan.

```{note} Math
:class: dropdown
Objective:

$$
\hat J(\mathbf{u})=\frac1P\sum_{i=1}^{P}
\sum_{h=0}^{H-1}\gamma^h r_{h+1}^{(i)}
$$

- $\mathbf{u}$: Candidate open-loop action sequence.
- $P$: Number of trajectory particles.
- $r_{h+1}^{(i)}$: Particle $i$ reward; 0 after its termination.
- **TS∞** propagation: 1 ensemble member per particle for the whole rollout; fresh transition noise each step.
```

````{important} Code
:class: dropdown
```python
import random
from statistics import mean

class ParticleReturns:
    def __init__(self, models, particles, gamma, seed=0, leaf=lambda s: 0.0):
        self.models, self.particles, self.gamma = tuple(models), particles, gamma
        self.rng, self.leaf = random.Random(seed), leaf
        if not self.models or particles < 1 or not 0 <= gamma < 1:
            raise ValueError("Need models, positive particles and 0 <= gamma < 1")

    def __call__(self, state, actions):
        returns = []
        for _ in range(self.particles):
            ## One persistent dynamics hypothesis per trajectory; fresh outcome noise per step.
            model = self.rng.choice(self.models)
            s, total, discount, done = state, 0.0, 1.0, False
            for action in actions:
                s, reward, done = model(s, action)
                total += discount * reward
                discount *= self.gamma
                if done:
                    break
            returns.append(total + (0.0 if done else discount * self.leaf(s)))
        return mean(returns)

## Example: callbacks return (next_state, reward, true_termination) & never mutate state.
score = ParticleReturns([lambda s, a: (s + a, a, False)], particles=2, gamma=0.9)
assert abs(score(0, [1, 1]) - 1.9) < 1e-12
```
````

```{attention} Q&A
:class: dropdown
*Does particle sampling make control risk-averse?*

- ❌. Avg over particles → optimizes an estimated expectation, ❌safety guarantee.

*How does the code map to PETS?*

- `ParticleReturns` = TS∞ particle evaluator. Learned callbacks can wrap `EnvironmentModel.sample`.
- `leaf` bootstrap = optional extension, ❌part of PETS.

*Must the ensemble member stay fixed?*

- PETS tests both: **TS1** (resample member each step) & **TS∞** (fixed member per particle).
- TS∞ keeps 1 dynamics hypothesis per trajectory.
- Neither guarantees calibrated uncertainty.
```

&nbsp;

## Learning from Simulated Experience
- **What**: Background planning: model → simulated transitions → value/policy updates.

### Dyna-Q
- **What**: Q-learning on real experience + learned-simulator experience {cite:p}`sutton1990integrated`.
- **Why**: 1 real transition → only 1 direct update. A model → many more updates from the same data.
- **How**:
    1. Act → apply a real [Q-learning](value.md#q-learning) update.
    2. Update the transition/reward/termination model.
    3. Sample previously observed state-action pairs → generate modeled outcomes.
    4. Apply the same update to each simulated transition.

```{note} Math
:class: dropdown
Process:

$$
Q(s,a)\leftarrow Q(s,a)+\alpha
\left[r'+\gamma(1-d)\max_{a'\in\mathcal{A}(s')}Q(s',a')-Q(s,a)\right]
$$

- $\alpha\in(0,1]$: Update step size.
- $r'$: Real or modeled immediate reward.
- $d$: True termination of the corresponding transition.
- $a'$: Candidate next action.
- Terminal transition → target $r'$.
- Artificial simulation cutoff alone → keep the bootstrap.
- $N_{\text{plan}}$: #simulated backups per real transition.
```

````{important} Code
:class: dropdown
```python
import random
from collections import defaultdict

class DynaQ:
    def __init__(self, n_actions, gamma=0.9, alpha=0.5, seed=0):
        self.n_actions, self.gamma, self.alpha = n_actions, gamma, alpha
        self.q = defaultdict(lambda: [0.0] * n_actions)
        self.model, self.rng = {}, random.Random(seed)

    def act(self, s, epsilon=0.1):
        if self.rng.random() < epsilon:
            return self.rng.randrange(self.n_actions)
        best = max(self.q[s])
        return self.rng.choice([a for a, q in enumerate(self.q[s]) if q == best])

    def target(self, outcome):
        sp, reward, terminated = outcome
        return reward if terminated else reward + self.gamma * max(self.q[sp])

    def backup(self, key, outcome):
        s, a = key
        self.q[s][a] += self.alpha * (self.target(outcome) - self.q[s][a])

    def observe(self, s, a, reward, sp, terminated, planning_steps=10):
        key, outcome = (s, a), (sp, reward, terminated)
        self.backup(key, outcome)
        ## Deterministic tabular model: overwrite the last observed outcome.
        self.model[key] = outcome
        for _ in range(planning_steps):
            imagined = self.rng.choice(tuple(self.model))
            self.backup(imagined, self.model[imagined])

## Example: reward at state 1 propagates back to state 0 through planning.
agent = DynaQ(n_actions=1, alpha=1.0)
agent.observe(0, 0, 0.0, 1, False, planning_steps=0)
agent.observe(1, 0, 1.0, 2, True, planning_steps=20)
assert abs(agent.q[0][0] - 0.9) < 1e-12
```
````

```{attention} Q&A
:class: dropdown
*Is Dyna-Q model-based or model-free?*

- Model-based architecture w/ a model-free update rule.
- Direct & simulated learning share 1 $Q$.
- Learned controller acts w/o consulting the model online.

*How is it different from experience replay?*

- Replay reuses recorded outcomes. Dyna generates outcomes from a model.
- Deterministic last-observation table → sampled Dyna updates = replay of those entries.
- Probabilistic model → can generate alternative outcomes or average over them.
- Function-approximation model → can generalize to unseen pairs.

*What breaks in the compact implementation?*

- Assumes stationary deterministic outcomes, hashable states & 1 finite legal action set at every nonterminal state.
- Stochastic env → need outcome counts/distributions or a generative model. Overwriting 1 random outcome ≠ transition-distribution estimate.
- Planning reuses knowledge, ❌new evidence.
- Unknown shortcuts → need real exploration. Stale model → need new observations.
```

&nbsp;

#### Prioritized Sweeping
- **What**: Model-based backups ordered by Bellman-residual size {cite:p}`moore1993prioritized`.
- **Why**: Uniform planning wastes updates on settled estimates.
- **How**:
    1. Queue state-action pairs w/ large model Bellman residuals.
    2. Back up the highest-priority pair.
    3. Re-queue predecessor pairs whose targets depend on the changed state's value.

```{note} Math
:class: dropdown
Process:

$$
\begin{align*}
\operatorname{priority}(s,a)
&=\left|\mathbb{E}_{\hat M_\psi}
\left[r'+\gamma(1-d)\max_{a'}Q(s',a')\mid s,a\right]-Q(s,a)\right|,\\
\operatorname{Pred}(s)
&=\{(u,a):\Pr_{\hat M_\psi}(s'=s,d=0\mid u,a)>0\}.
\end{align*}
$$

- $r'$: Modeled reward.
- $d$: Modeled true termination.
- $u$: Predecessor state.
- $\operatorname{Pred}(s)$: State-action pairs whose continuing outcomes can reach $s$.
- Enqueue only residuals above threshold $\eta>0$.
- Stochastic models → expected backups or suitable estimates.
```

````{important} Code
:class: dropdown
```python
import heapq
from collections import defaultdict
from itertools import count
## Reuses DynaQ from this page.

class PrioritizedSweeping(DynaQ):
    def __init__(self, n_actions, gamma=0.9, threshold=1e-8):
        super().__init__(n_actions, gamma=gamma, alpha=1.0)
        self.predecessors = defaultdict(set)
        self.heap, self.serial, self.threshold = [], count(), threshold

    def enqueue(self, key):
        s, a = key
        priority = abs(self.target(self.model[key]) - self.q[s][a])
        if priority > self.threshold:
            heapq.heappush(self.heap, (-priority, next(self.serial), key))

    def observe(self, s, a, reward, sp, terminated, planning_steps=10):
        key = (s, a)
        if key in self.model:
            old_sp, _, old_done = self.model[key]
            if not old_done:
                self.predecessors[old_sp].discard(key)
        self.model[key] = (sp, reward, terminated)
        if not terminated:
            self.predecessors[sp].add(key)
        self.enqueue(key)
        for _ in range(planning_steps):
            if not self.heap:
                break
            _, _, key = heapq.heappop(self.heap)
            state, action = key
            ## Recompute stale priorities' targets; full deterministic backup.
            if abs(self.target(self.model[key]) - self.q[state][action]) <= self.threshold:
                continue
            self.backup(key, self.model[key])
            for predecessor in self.predecessors[state]:
                self.enqueue(predecessor)

## Example: terminal reward → predecessor update, without uniform sampling.
agent = PrioritizedSweeping(n_actions=1)
agent.observe(0, 0, 0.0, 1, False)
agent.observe(1, 0, 1.0, 2, True)
assert abs(agent.q[0][0] - 0.9) < 1e-12
```
````

```{attention} Q&A
:class: dropdown
*Prioritized sweeping vs prioritized replay?*

- Sweeping: model predecessor structure → propagate consequences backward.
- Prioritized replay: resample stored transitions by learning priority. ❌Predecessor model.

*Costs & limits?*

- Maintain reverse dependencies & a queue. Changed model edges → remove obsolete predecessors.
- Compact code allows duplicate heap entries & recomputes targets on pop → duplicates consume budget.
- Indexed priority queue → avoids that overhead at scale.
- ❌Direct real backup in this variant: observations enter the queue first → zero planning budget = ❌value update.
```

&nbsp;

### MBPO
- **Name**: Model-Based Policy Optimization {cite:p}`janner2019trust`
- **What**: Off-policy actor-critic on real data + short branched model rollouts.
- **Why**: Full-episode imagination → model error compounds over long task horizons.
- **How**:
    1. Fit a probabilistic ensemble on real transitions.
    2. Start short model rollouts from states sampled from real replay.
    3. Sample actions from the curr policy → store imagined transitions in model replay.
    4. Train [SAC](policy.md#sac) on a real + model data mix. Refresh model & synthetic data.

```{note} Math
:class: dropdown
Process:

$$
s_0\sim\mathcal{D},\qquad
a_h\sim\pi(\cdot|s_h),\qquad
(s_{h+1},r_{h+1},d_h)\sim\hat M_\psi(\cdot|s_h,a_h),
\quad 0\le h<H.
$$

- $s_0$: Replay branch point; ≠ episode-initial state in general.
- End on $d_h=1$ or at horizon $H$. Horizon end → $c_h=1$, $d_h=0$.
- Nonterminal model cutoff → critic targets keep their next-state bootstrap.
```

````{important} Code
:class: dropdown
```python
import random

class BranchedRollouts:
    def __init__(self, models, policy, horizon, seed=0):
        self.models, self.policy, self.horizon = tuple(models), policy, horizon
        self.rng = random.Random(seed)
        if not self.models or horizon < 1:
            raise ValueError("Need models and a positive rollout horizon")

    def __call__(self, real_states, branches):
        data = []
        for _ in range(branches):
            s = self.rng.choice(real_states)  ## Nonterminal source states from real replay.
            for h in range(self.horizon):
                a = self.policy(s)
                model = self.rng.choice(self.models)
                sp, reward, terminated = model(s, a)
                boundary = terminated or h + 1 == self.horizon
                data.append((s, a, reward, sp, terminated, boundary))
                if terminated:
                    break
                s = sp
        return data

## Example: 1-step branch → collection ends, task does not.
rollout = BranchedRollouts([lambda s, a: (s + a, 1.0, False)],
                          policy=lambda s: 1, horizon=1)
sample = rollout([0, 4], branches=1)[0]
assert sample[-2:] == (False, True)
```
````

```{attention} Q&A
:class: dropdown
*MBPO vs online MPC?*

- MBPO: model → training transitions. Actor selects deployed actions w/o solving a planning problem.
- MPC: model → online optimization at every real step.
- Code covers branching only. Model fitting, replay mixing & SAC updates are separate parts.

*Why not make rollouts arbitrarily long?*

- More synthetic transitions → more accumulated model bias.
- Short branches ⬇️ state drift. New policy actions can still leave the model's support at step 1.
- Branch horizon & synthetic-data ratio = tuning knobs, ❌universal defaults.
```

&nbsp;

### Dreamer
- **What**: Actor-critic trained by backprop through imagined latent-model rollouts {cite:p}`hafner2020dream`.
- **Why**:
    - *Why do we need it?*
        - Online latent planning (e.g., PlaNet's CEM) → costly per step & blind beyond $H$.
        - Model-generated data for model-free updates → ignores the model's gradients.
    - *Why does it work?*
        1. Latent state → compact → many parallel imagined trajectories, ❌pixel generation.
        2. Critic bootstraps beyond $H$ → long-horizon credit w/ short imagination.
        3. Differentiable dynamics + reparameterized actions → analytic value grads for the actor.
- **How**:
    1. **World model**: Learn latent dynamics from replayed real sequences (reconstruction, reward & KL terms).
    2. **Imagine**: From posterior latents of real data, roll out $H$ steps w/ actor & latent transition.
    3. **Critic**: Regress values toward λ-returns of the imagined trajectory.
    4. **Actor**: Maximize λ-returns by backprop through rewards, values & dynamics. World model frozen.
    5. **Act**: Encode the real history → actor action (+ exploration noise) → add data to $\mathcal{D}$.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\theta$: Actor params.
    - $\phi$: Critic params.
- Hyperparams:
    - $\lambda\in[0,1]$: λ-return mixing coeff.
    - $\beta$: KL weight.
- Misc:
    - Paper → page: model $\theta$ → $\psi$, actor $\phi$ → $\theta$, critic $\psi$ → $\phi$.

Model:

$$
\begin{align*}
z_t&\sim q_\psi(z_t|z_{t-1},a_{t-1},o_t),\\
\hat z_t&\sim\hat p_\psi(\hat z_t|z_{t-1},a_{t-1}),\\
\hat r_t&\sim\hat p_\psi(r_t|z_t),\\
\hat o_t&\sim\hat p_\psi(o_t|z_t).
\end{align*}
$$

- $q_\psi$: Representation model (posterior; sees $o_t$).
- $\hat p_\psi(\hat z_t|\cdot)$: Transition model (prior; ❌sees $o_t$). Used for imagination.
- $\hat p_\psi(r_t|z_t)$: Reward model.
- $\hat p_\psi(o_t|z_t)$: Observation decoder; training signal only.
- $r_t$: Reward received w/ $o_t$, i.e., after $a_{t-1}$.

Training:

$$
\mathcal{J}_\text{WM}(\psi)
=\mathbb{E}\Big[\sum_t
\ln\hat p_\psi(o_t|z_t)
+\ln\hat p_\psi(r_t|z_t)
-\beta\operatorname{KL}\big(q_\psi(z_t|z_{t-1},a_{t-1},o_t)\,\big\|\,\hat p_\psi(z_t|z_{t-1},a_{t-1})\big)
\Big]
$$

- Maximize. Expectation over real sequences from $\mathcal{D}$ & posterior samples.

Imagination from a posterior latent $\hat z_0=z_t$:

$$
a_h\sim\pi_\theta(\cdot|\hat z_h),\qquad
\hat z_{h+1}\sim\hat p_\psi(\cdot|\hat z_h,a_h),\qquad
\hat r_{h+1}=\mathbb{E}\big[r\,|\,\hat z_{h+1}\big],
\quad 0\le h<H.
$$

$$
G^\lambda_h
=\hat r_{h+1}+\gamma\Big[(1-\lambda)v_\phi(\hat z_{h+1})+\lambda G^\lambda_{h+1}\Big],
\qquad G^\lambda_H=v_\phi(\hat z_H).
$$

- $G^\lambda_h$: λ-return at imagined step $h$.
- $v_\phi$: Critic.
- $\lambda=0$ → 1-step TD target. $\lambda=1$ → $H$-step model return + bootstrap at $\hat z_H$.
- Paper attaches reward $r_\tau$ to the state $s_\tau$ that emits it. Same recursion, shifted reward index.
- Early-termination tasks → model also predicts a discount per latent; cumulative product weights each step.

Objective:

$$
\max_\theta\ \mathbb{E}\Big[\sum_{h=0}^{H}G^\lambda_h\Big],
\qquad
\min_\phi\ \mathbb{E}\Big[\sum_{h=0}^{H}\tfrac12\big(v_\phi(\hat z_h)-\operatorname{sg}(G^\lambda_h)\big)^2\Big]
$$

- $\operatorname{sg}$: Stop-gradient.
- Actor grad flows through $\hat r$, $v_\phi$ & $\hat p_\psi$ via reparameterized actions & latents. Only $\theta$ updates.
```

````{important} Code
:class: dropdown
```python
import torch
from torch import nn

class DreamerBehavior(nn.Module):
    def __init__(self, latent_dim, action_dim, horizon, gamma, lam):
        super().__init__()
        ## World-model parts: trained by J_WM elsewhere, frozen during behavior learning.
        self.dynamics = nn.Linear(latent_dim + action_dim, latent_dim)  ## deterministic stand-in for RSSM
        self.reward = nn.Linear(latent_dim, 1)
        self.dynamics.requires_grad_(False)
        self.reward.requires_grad_(False)
        self.actor = nn.Linear(latent_dim, 2 * action_dim)
        self.critic = nn.Linear(latent_dim, 1)
        self.horizon, self.gamma, self.lam = horizon, gamma, lam

    def imagine(self, z):
        latents, rewards = [z], []
        for _ in range(self.horizon):
            mean, log_std = self.actor(z).chunk(2, -1)
            ## Reparameterized tanh-Gaussian -> action stays differentiable w.r.t. actor params.
            a = torch.tanh(mean + log_std.exp() * torch.randn_like(mean))
            z = self.dynamics(torch.cat((z, a), -1)).tanh()
            latents.append(z)
            rewards.append(self.reward(z).squeeze(-1))  ## r_{h+1} from z_{h+1}
        return torch.stack(latents, 1), torch.stack(rewards, 1)

    def lambda_returns(self, rewards, values):
        ## Backward recursion; bootstrap from the critic at the horizon.
        g = values[:, -1]
        returns = [g]
        for h in reversed(range(self.horizon)):
            g = rewards[:, h] + self.gamma * ((1 - self.lam) * values[:, h + 1] + self.lam * g)
            returns.append(g)
        return torch.stack(returns[::-1], 1)

    def forward(self, z0):
        latents, rewards = self.imagine(z0)
        returns = self.lambda_returns(rewards, self.critic(latents).squeeze(-1))
        actor_loss = -returns.sum(1).mean()
        ## Critic sees detached latents & targets -> its graph is independent of the actor's.
        values = self.critic(latents.detach()).squeeze(-1)
        critic_loss = 0.5 * (values - returns.detach()).square().sum(1).mean()
        return actor_loss, critic_loss

## Example
agent = DreamerBehavior(latent_dim=4, action_dim=2, horizon=3, gamma=0.9, lam=0.95)
actor_loss, critic_loss = agent(torch.randn(5, 4))  ## 5 posterior start latents
actor_loss.backward(inputs=list(agent.actor.parameters()))  ## actor update ignores critic grads
critic_loss.backward()
print(agent.actor.weight.grad.shape, agent.critic.weight.grad.shape)  ## torch.Size([4, 4]) torch.Size([1, 4])
print(agent.dynamics.weight.grad)  ## None
```
````

```{attention} Q&A
:class: dropdown
*Dreamer vs MBPO?*

- MBPO: model → synthetic transitions → model-free SAC. Grads ❌pass through the dynamics.
- Dreamer: analytic value grads through learned latent dynamics.
- MBPO imagines in env state space. Dreamer imagines in a learned latent space.

*Dreamer vs PlaNet / MPC?*

- PlaNet: CEM search in latent space at every real step → compute per action ⬆️.
- Dreamer: amortized actor → 1 actor forward pass per action, ❌search.
- Reward-only imagination over $H$ → shortsighted. Dreamer's critic covers rewards beyond $H$.

*Why λ-returns?*

- Bias-variance tradeoff.
- $\lambda$⬇️ → more critic reliance → critic bias.
- $\lambda$⬆️ → more imagined-reward reliance → model error & variance over the horizon.

*Why freeze the world model during behavior learning?*

- Actor grads would otherwise reshape the model toward high predicted value, ❌toward accurate predictions.

*Discrete actions?*

- Reparameterization ❌applies directly. Dreamer uses straight-through gradients.

*Failure modes?*

- Actor ascends model grads → can exploit latent-dynamics errors.
- Long backprop chains through dynamics → exploding/vanishing grads.
- Reconstruction objective spends capacity on task-irrelevant pixels.
```

&nbsp;

## Decision-Time Search
- **What**: Planning from the curr state at action time → 1 action choice.

### Lookahead & Expectimax
- **What**: Finite-depth decision/chance tree expansion w/ leaf-value bootstrap.
- **Why**: Only the curr action is needed → full-state planning is wasted work.
- **How**:
    1. Expand legal actions at each decision node.
    2. Average env outcomes at chance nodes.
    3. Stop at termination (value 0) or at planning depth (leaf value).
    4. Back up expected returns → pick the maximizing root action.

```{note} Math
:class: dropdown
Process:

$$
\begin{align*}
V_0(s)&=\hat V(s),\\
Q_h(s,a)&=\mathbb{E}_{\hat M_\psi}
\left[r'+\gamma(1-d)V_{h-1}(s')\mid s,a\right],\\
V_h(s)&=\max_{a\in\mathcal{A}(s)}Q_h(s,a),\qquad h\ge1.
\end{align*}
$$

- $h$: Remaining lookahead depth.
- $V_h$: Depth-limited planned state value.
- $Q_h$: Depth-limited planned action value.
- $r'$: Immediate modeled reward.
- $d$: Modeled true termination.
- Terminal-state values = 0 at every depth, incl. $V_0$. Incoming terminal reward counted once.

$$
\|V_H-V^*\|_\infty\le\gamma^H\|\hat V-V^*\|_\infty
$$

- $V^*$: Optimal value in an exact known model.
- Assumes exact max/expectation & bounded leaf error.
- Learned dynamics → add model error.
```

````{important} Code
:class: dropdown
```python
class Expectimax:
    def __init__(self, actions, outcomes, gamma, leaf=lambda s: 0.0):
        self.actions, self.outcomes, self.gamma, self.leaf = actions, outcomes, gamma, leaf

    def value(self, s, depth):
        legal = tuple(self.actions(s))
        if not legal:
            return 0.0
        if depth == 0:
            return self.leaf(s)
        return max(self.q(s, a, depth) for a in legal)

    def q(self, s, a, depth):
        return sum(prob * (reward + (0.0 if done else
                    self.gamma * self.value(sp, depth - 1)))
                   for prob, sp, reward, done in self.outcomes(s, a))

    def act(self, s, depth):
        legal = tuple(self.actions(s))
        if depth < 1 or not legal:
            raise ValueError("Need positive depth and a nonterminal state")
        return max(legal, key=lambda a: self.q(s, a, depth))

## Example: expectation over outcomes → "safe" wins; max over outcomes would pick "risky".
outcomes = {
    "safe": [(1.0, "end", 0.6, True)],
    "risky": [(0.5, "end", 1.0, True), (0.5, "end", 0.0, True)]
}
planner = Expectimax(lambda s: () if s == "end" else outcomes,
                    lambda s, a: outcomes[a], gamma=0.9)
assert planner.act("start", depth=2) == "safe"
```
````

```{attention} Q&A
:class: dropdown
*Why max over actions but avg over outcomes?*

- Agent controls actions. Agent ❌controls env randomness.
- Max over chance outcomes → assumes control over luck.
- Min over chance outcomes → different objective (worst case).
- Adversarial 2-player search adds opponent decisions. Single-agent RL ❌alternates reward signs.

*Why not evaluate 1 fixed action sequence?*

- Expectimax: later actions depend on the observed successor state.
- Fixed sequence = open-loop → ❌adapts within its hypothetical future.
- Deterministic dynamics → ❌chance branching. Task termination ≠ search cutoff still holds.

*What limits exhaustive search?*

- $\le A$ actions & $\le B$ outcomes per action → unmerged depth-$H$ tree = $O((AB)^H)$.
- Fixed model → memoize over (state, remaining depth) to share repeated subproblems.
- Better leaf values → shallower search, at the cost of learned approximation.
- Omitted tails ≠ 0 automatically.
```

&nbsp;

### MCTS with UCT
- **Name**: Monte Carlo Tree Search with Upper Confidence bounds applied to Trees {cite:p}`kocsis2006bandit`
- **What**: Sampled lookahead tree grown by bandit-guided action selection.
- **Why**: Exhaustive expansion spends equal compute on every branch, incl. bad ones.
- **How**:
    1. **Selection**: Pick actions by value estimate + exploration bonus.
    2. **Chance sampling**: Draw 1 model outcome after each chosen action.
    3. **Expansion**: Add the newly reached decision node → estimate its continuation by rollout or leaf value.
    4. **Backup**: Propagate discounted sampled returns through visited decision edges.
    5. **Act**: Pick the root action by visit count. Exploration bonus guides search, ❌execution.

```{note} Math
:class: dropdown
Process:

$$
a=\arg\max_{a\in\mathcal{A}(s)}
\left[
\bar Q(s,a)+C\sqrt{\frac{\log N(s)}{N(s,a)}}
\right]
$$

- $\bar Q(s,a)$: Mean backed-up return for a decision edge.
- $C>0$: Search exploration coeff; scale depends on return scale.
- $N(s)$: Decision-node visits.
- $N(s,a)$: Visits to the action edge.
- Unvisited actions → tried first, before the finite-count formula applies.

$$
g_h=r_{h+1}+\gamma(1-d_h)g_{h+1},\qquad
\bar Q(s_h,a_h)\leftarrow\bar Q(s_h,a_h)
+\frac{g_h-\bar Q(s_h,a_h)}{N(s_h,a_h)}
$$

- $g_h$: Sampled return from search depth $h$.
- $g_H$: Leaf estimate at a nonterminal cutoff; 0 after true termination.
- Counts increment before the running-mean update.
- Tree: decision node → action edge → sampled chance outcome → next decision node.
- 1 action can reach many outcome children. Simulator sampling → respects outcome probabilities.
```

````{important} Code
:class: dropdown
```python
import math
import random

class UCT:
    def __init__(self, actions, sample, horizon, gamma, exploration=1.0,
                 leaf=lambda s: 0.0, seed=0):
        self.actions, self.sample = actions, sample
        self.horizon, self.gamma, self.exploration = horizon, gamma, exploration
        self.leaf, self.rng = leaf, random.Random(seed)
        if horizon < 1 or not 0 <= gamma < 1 or exploration <= 0:
            raise ValueError("Need positive horizon/exploration and 0 <= gamma < 1")

    def rollout(self, s, depth):
        total, discount = 0.0, 1.0
        for _ in range(depth):
            legal = tuple(self.actions(s))
            if not legal:
                return total
            s, reward, done = self.sample(s, self.rng.choice(legal))
            total += discount * reward
            if done:
                return total
            discount *= self.gamma
        return total + discount * self.leaf(s)

    def simulate(self, node, s, depth):
        legal = tuple(self.actions(s))
        if not legal:
            return 0.0
        if depth == 0:
            return self.leaf(s)
        if not node:
            node.update(visits=0, edges={a: [0, 0.0, {}] for a in legal})
        edges = node["edges"]
        unvisited = [a for a in legal if edges[a][0] == 0]
        if unvisited:
            action = self.rng.choice(unvisited)
        else:
            action = max(legal, key=lambda a: edges[a][1] + self.exploration *
                         math.sqrt(math.log(node["visits"]) / edges[a][0]))
        count, average, outcomes = edges[action]
        ## Chance is sampled, never selected by UCT; resample on every traversal.
        sp, reward, done = self.sample(s, action)
        if done:
            continuation = 0.0
        elif sp not in outcomes:
            outcomes[sp] = {}
            continuation = self.rollout(sp, depth - 1)
        else:
            continuation = self.simulate(outcomes[sp], sp, depth - 1)
        result = reward + self.gamma * continuation
        edges[action][0] = count + 1
        edges[action][1] = average + (result - average) / (count + 1)
        node["visits"] += 1
        return result

    def act(self, s, simulations):
        if simulations < 1 or not tuple(self.actions(s)):
            raise ValueError("Need simulations and a nonterminal root")
        self.root = {}  ## Fresh search avoids stale estimates after model updates.
        for _ in range(simulations):
            self.simulate(self.root, s, self.horizon)
        return max(self.root["edges"],
                   key=lambda a: tuple(self.root["edges"][a][:2]))

## Example: independent stochastic terminal rewards; ❌adversarial sign flip.
rng = random.Random(1)
planner = UCT(
    actions=lambda s: ("safe", "risky"),
    sample=lambda s, a: ("end", 0.6 if a == "safe" else float(rng.random() < 0.5), True),
    horizon=2, gamma=0.9, seed=1)
assert planner.act("start", simulations=2000) == "safe"
```
````

```{attention} Q&A
:class: dropdown
*What does the code assume?*

- Finite legal actions, finite/hashable state outcomes & a stationary Markov generative model.
- Callbacks return fresh outcomes w/o changing the real env.
- Empty legal-action set = terminal state.
- Outcomes share a continuation node by next state. Sampled rewards stay edge-specific returns, ❌cached deterministic rewards.
- Continuous unique next states → ❌revisits → this finite-outcome tree ❌fits continuous states.

*What can more simulations guarantee?*

- Finite-horizon, finite-action, finite-outcome (revisitable tree), bounded-return model → UCT root choice converges to optimal for the supplied model & leaf objective.
- Finite budget → can miss delayed rewards. Asymptotic guarantees ≠ practical runtime bounds.
- Wrong model/leaf values stay wrong w/ unlimited sampling.

*Why no sign changes?*

- All returns use 1 single-agent reward convention.
- 2-player zero-sum search from the player-to-move perspective → needs sign conversion. Different backup convention.

*How do policy/value networks help?*

- Policy priors → guide action exploration.
- Value heads → replace or shorten random rollouts.
- Prior-guided search ≠ the UCT formula above. AlphaZero uses a different exploration bonus.
```

&nbsp;

## Receding-Horizon Control

### MPC
- **Name**: Model Predictive Control
- **What**: Repeated finite-horizon optimization w/ 1st-action execution.
- **Why**: Long open-loop plan → ❌corrects deviations after it starts.
- **How**:
    1. Observe the curr state → optimize a candidate future action sequence.
    2. Execute only its 1st action.
    3. Observe the actual successor → shift/warm-start the proposal → solve again.
    4. Unknown dynamics → learn the model from new transitions.

```{note} Math
:class: dropdown
Objective:

$$
\begin{align*}
\mathbf{u}^*
&\in\arg\max_{\mathbf{u}}\;
\mathbb{E}_{\hat M_\psi}
\left[
\sum_{h=0}^{H-1}\gamma^h w_h r_{t+h+1}
+\gamma^H w_H\hat V(s_{t+H})
\mid s_t,\mathbf{u}
\right],\\
w_h&=\prod_{\ell=0}^{h-1}(1-d_{t+\ell}),\qquad w_0=1,\\
a_t&=u_0^*.
\end{align*}
$$

- $\mathbf{u}=(u_0,\ldots,u_{H-1})$: Open-loop action sequence; actions may be vectors.
- $u_h$: Proposed action at relative time $h$.
- $w_h$: Survival indicator before predicted step $h$.
- Known simulator → replaces $\hat M_\psi$ directly.
- Action bounds → restrict every $u_h$.
- State constraints → need predicted-state checks or an explicit constrained formulation.
```

```{note} Example
:class: dropdown
- Plan 2 pushes to move an object to a target.
- Execute the 1st. Friction leaves the object short.
- Replan from the measured position. ❌Apply the old 2nd push blindly.
- Wrong friction model → still misleads each solve. Replanning = correction, ❌model-error removal.
```

```{attention} Q&A
:class: dropdown
*Is MPC a learning algorithm?*

- ❌. Known dynamics + repeated optimization → ❌training.
- Learned dynamics → MPC becomes a model-based RL/control component.
- Solver: gradients, quadratic programming or sampling. Model-based ≠ differentiable.
- 1st-action & replanning loop in code → [CEM](#cem).

*Open-loop optimization but closed-loop control?*

- Within 1 solve: candidate sequence ❌adapts to hypothetical outcomes.
- Across real steps: state feedback changes the next solve.
- Stochastic dynamics → ≠ optimizing over all contingent future policies in general.

*How long should the horizon be?*

- Longer → more delayed consequences, ⬆️optimization cost, ⬆️accumulated model error.
- Shorter → easier search, ⬇️drift, ⬆️dependence on the terminal value.
- Leaf value must match the reward/discount convention. Leaf = 0 → truncated objective.

*Do replanning or penalties guarantee safety/stability?*

- ❌. Action clipping enforces only action bounds, ❌state constraints.
- Soft penalties → trade violations against reward.
- Finite candidate sampling → can miss feasible/safe solutions.
- Hard guarantees need: suitable model/uncertainty assumptions, constrained optimization, feasibility handling, often terminal conditions or a verified fallback.
- Compute delay & state-estimation error → break the instant-replanning idealization.
```

&nbsp;

#### CEM
- **Name**: Cross-Entropy Method {cite:p}`rubinstein1999cross`
- **What**: Iterative refit of a proposal distribution to high-scoring samples.
- **Why**: Nonlinear action-sequence objectives → useful derivatives may not exist.
- **How**:
    1. Sample candidate sequences from a proposal distribution.
    2. Score them w/ model rollouts → keep an elite subset.
    3. Refit proposal params to the elites. Repeat within the planning budget.
    4. Return the best candidate's 1st action. Re-optimize after the real observation.

```{note} Math
:class: dropdown
Process:

$$
\mathbf{u}^{(i)}\sim q_\xi,\qquad
\xi_{\text{new}}\in\arg\max_\xi
\sum_{i\in\mathcal{E}}\log q_\xi(\mathbf{u}^{(i)})
$$

- $q_\xi$: Parameterized proposal over action sequences.
- $\xi$: Proposal params.
- $\mathcal{E}$: Indices of the highest-scoring elite sequences.
- $\mathbf{u}^{(i)}$: Candidate $i$; flatten horizon × action coordinates.

$$
\mathbf{m}_{\text{new}}
=\frac1{|\mathcal{E}|}\sum_{i\in\mathcal{E}}\mathbf{u}^{(i)},\qquad
\boldsymbol{\sigma}_{\text{new}}^2
=\frac1{|\mathcal{E}|}\sum_{i\in\mathcal{E}}
(\mathbf{u}^{(i)}-\mathbf{m}_{\text{new}})^{\odot2}
$$

- $\mathbf{m}$: Diagonal-Gaussian proposal mean.
- $\boldsymbol{\sigma}^2$: Coordinatewise proposal variance.
- $\odot2$: Elementwise square.
- Above = unconstrained Gaussian maximum-likelihood updates.
- Clip bounded actions → match moments = practical heuristic, ❌exact truncated-Gaussian maximum likelihood.
```

````{important} Code
:class: dropdown
```python
import math
import random
from statistics import mean, pvariance
## Reuses ParticleReturns from this page for model-based sequence scoring.

class CEMMPC:
    def __init__(self, score, horizon, low, high, population, elites, iterations,
                 min_std=0.01, seed=0):
        if not (horizon >= 1 and low < high and 1 <= elites <= population
                and iterations >= 1 and min_std > 0):
            raise ValueError("Invalid horizon, bounds or search budget")
        self.score, self.horizon, self.low, self.high = score, horizon, low, high
        self.population, self.elites, self.iterations = population, elites, iterations
        self.min_std, self.rng = min_std, random.Random(seed)
        self.mean = [(low + high) / 2] * horizon

    def act(self, state):
        center = self.mean[:]
        scale = [(self.high - self.low) / 2] * self.horizon
        best, best_score = None, -math.inf
        for _ in range(self.iterations):
            candidates = [[min(self.high, max(self.low, self.rng.gauss(m, sd)))
                           for m, sd in zip(center, scale)]
                          for _ in range(self.population)]
            scored = [(self.score(state, seq), seq) for seq in candidates]
            if any(not math.isfinite(value) for value, _ in scored):
                raise ValueError("Non-finite model score; do not execute an unchecked action")
            scored.sort(key=lambda pair: pair[0], reverse=True)
            if scored[0][0] > best_score:
                best_score, best = scored[0]
            elite = [seq for _, seq in scored[:self.elites]]
            coordinates = list(zip(*elite))
            center = [mean(values) for values in coordinates]
            scale = [max(self.min_std, math.sqrt(pvariance(values))) for values in coordinates]
        self.mean = center[1:] + [(self.low + self.high) / 2]
        return best[0]  ## Execute ONE scalar action, never the entire optimized sequence.

## Example: scalar-action control; budgets are illustrative, ❌recommended defaults.
def physics(s, a):
    sp = s + a
    return sp, -sp * sp, False

score = ParticleReturns([physics], particles=1, gamma=0.9)
controller = CEMMPC(score, horizon=3, low=-1.0, high=1.0,
                    population=64, elites=8, iterations=4)
state = 2.0
for _ in range(2):
    action = controller.act(state)  ## Replan from the newly observed state each time.
    assert -1.0 <= action <= 1.0
    state, reward, terminated = physics(state, action)
    if terminated:
        break
print(round(state, 2))
```
````

```{attention} Q&A
:class: dropdown
*Random shooting vs CEM?*

- Random shooting: sample once from a fixed proposal.
- CEM: adapt the proposal toward elites, over iterations.
- Scalar code → generalizes coordinatewise to vector actions. Diagonal covariance ignores correlations.

*Why the name cross-entropy?*

- Fit the proposal toward a distribution concentrated on elite events ← maximize their log-likelihood (= minimize cross-entropy).
- ❌Supervised classification, ❌actor network.

*Failure modes?*

- Premature variance collapse, multimodal optima, high-dimensional sequences.
- Variance floor → keeps local sampling, ❌global coverage.
- Finite population/budget → ❌global-optimality guarantee.
- Noisy model scores → lucky candidates win. Use enough particles; re-evaluate finalists.
- Clipping → probability mass piles on action bounds.
- State feasibility = the caller's separate job.
```

&nbsp;

## Learned Representations & Search

### AlphaZero
- **What**: Self-play policy/value learning guided by known-rule tree search {cite:p}`silver2018general`.
- **Why**: Search improves on the raw policy → its decisions become training targets for the network.
- **How**:
    1. Use exact game rules for transitions, legal moves & terminal outcomes.
    2. Guide search w/ policy priors & neural leaf values.
    3. Store search visit distributions & self-play outcomes.
    4. Train policy/value heads toward those targets → repeat.

```{note} Math
:class: dropdown
Training:

$$
\mathcal{L}
=(v_\phi(s)-z)^2
-\sum_a\pi_{\text{search}}(a|s)\log\pi_\theta(a|s)
+\lambda\|\vartheta\|_2^2
$$

- $v_\phi$: Predicted game outcome value.
- $\phi$: Value-head params.
- $z$: Self-play outcome from the player-to-move perspective at stored state $s$. Paper notation; overrides page latent $z_t$ locally.
- $\pi_{\text{search}}$: Normalized root-visit target, optionally temperature-adjusted.
- $\theta$: Policy-head params.
- $\vartheta$: All shared/network params.
- $\lambda$: Weight-decay coeff.
- Board-game outcome prediction = episodic & undiscounted. ≠ discounted leaf targets elsewhere on this page.
```

````{important} Code
:class: dropdown
```python
import torch
from torch import nn
from torch.nn import functional as F

class SearchDistillation(nn.Module):
    def __init__(self, feature_dim, n_actions):
        super().__init__()
        self.policy = nn.Linear(feature_dim, n_actions)
        self.value = nn.Linear(feature_dim, 1)

    def forward(self, features):
        return self.policy(features), self.value(features).squeeze(-1).tanh()

    def loss(self, features, visits, outcome, legal, l2=0.0):
        ## Search has already produced legal root counts; fit its distribution, not one argmax.
        counts = visits.masked_fill(~legal, 0)
        if (counts.sum(-1) <= 0).any():
            raise ValueError("Every target needs positive legal visit mass")
        target = (counts / counts.sum(-1, keepdim=True)).detach()
        logits, value = self(features)
        logp = F.log_softmax(logits.masked_fill(~legal, -torch.inf), dim=-1)
        policy_loss = -(target * logp.masked_fill(~legal, 0)).sum(-1).mean()
        penalty = l2 * sum(p.square().sum() for p in self.parameters())
        return policy_loss + F.mse_loss(value, outcome.detach()) + penalty

## Example: distillation loss only; game simulation & prior-guided search are separate.
learner = SearchDistillation(feature_dim=2, n_actions=3)
loss = learner.loss(torch.zeros(1, 2), torch.tensor([[3., 1., 0.]]),
                    torch.tensor([1.]), torch.tensor([[True, True, False]]))
loss.backward()
assert torch.isfinite(loss)
```
````

```{attention} Q&A
:class: dropdown
*Does AlphaZero learn the game dynamics?*

- ❌. Learns policy/value only. Search uses known transitions & legal moves.

*Why train toward visit counts, ❌the raw policy?*

- Search = policy-improvement operator → visit distribution beats the prior.
- Distill visits → stronger prior → stronger search → loop.

*How does its search differ from UCT?*

- PUCT-style bonus ∝ prior $\pi_\theta(a|s)$ × $\sqrt{N(s)}/(1+N(s,a))$ → prior steers exploration.
- Value head replaces random rollouts.
- Code covers distillation only, ❌the full search system.

*AlphaGo vs AlphaGo Zero vs AlphaZero?*

- AlphaGo: human expert data + RL + rollouts.
- AlphaGo Zero: self-play from scratch, ❌human data, ❌rollouts; Go only.
- AlphaZero: same recipe for chess, shogi & Go. ❌Go-specific symmetry augmentation. 1 continually updated network, ❌evaluator gating.

*Sign convention?*

- Outcome sign follows the stored player's perspective.
- ❌Copy that convention into single-agent backups.
```

&nbsp;

### MuZero
- **What**: Search-guided policy/value/reward learning through learned latent dynamics {cite:p}`schrittwieser2020mastering`.
- **Why**: Exact dynamics often unavailable. Planning ❌needs every observation detail.
- **How**:
    1. Encode observation history → latent root state.
    2. Apply candidate actions to recurrent latent dynamics.
    3. Search w/ predicted rewards, policy priors & values.
    4. Unroll real action sequences → train on observed rewards, return targets & search policies.

```{note} Math
:class: dropdown
Model:

$$
\begin{align*}
z_t^0&=h_\psi(o_{\le t}),\\
(z_t^{j+1},\hat r_{t+j+1})&=g_\psi(z_t^j,a_{t+j}),\\
(\hat\pi_t^j,\hat v_t^j)&=f_\psi(z_t^j).
\end{align*}
$$

- $o_{\le t}$: Observation history through time $t$.
- $z_t^j$: Latent state after $j$ hypothetical actions from root $t$; $z_t^0$ = page's $z_t$.
- $j$: Unroll step; overrides Tier-1 feature idx locally.
- $h_\psi$: Representation network.
- $g_\psi$: Latent dynamics/reward network.
- $f_\psi$: Policy/value prediction network.
- $\hat r_{t+j+1}$: Predicted reward after hypothetical action $a_{t+j}$.
- $\hat\pi_t^j$: Predicted action distribution.
- $\hat v_t^j$: Predicted value.

Training:

$$
\mathcal L_t
=\sum_{j=0}^{K}\left[
\ell_v(\hat v_t^j,v^{\text{target}}_{t+j})
+\ell_\pi(\hat\pi_t^j,\pi^{\text{search}}_{t+j})
\right]
+\sum_{j=0}^{K-1}\ell_r(\hat r_{t+j+1},r_{t+j+1})
+\lambda\|\psi\|_2^2.
$$

- $K$: #training unroll steps; ≠ search horizon $H$ in general.
- $v^{\text{target}}_{t+j}$: Observed return or bootstrapped value target.
- $\pi^{\text{search}}_{t+j}$: Search-derived policy target.
- $\ell_v$: Value prediction loss.
- $\ell_\pi$: Policy cross-entropy.
- $\ell_r$: Reward prediction loss.
- $\lambda$: Regularization coeff.
- Past true termination → stop/mask targets or use absorbing-state conventions.
```

````{important} Code
:class: dropdown
```python
import torch
from torch import nn

class LatentModel(nn.Module):
    def __init__(self, observation_dim, latent_dim, n_actions):
        super().__init__()
        self.n_actions = n_actions
        self.encoder = nn.Linear(observation_dim, latent_dim)
        self.dynamics = nn.Linear(latent_dim + n_actions, latent_dim)
        self.reward = nn.Linear(latent_dim + n_actions, 1)
        self.policy = nn.Linear(latent_dim, n_actions)
        self.value = nn.Linear(latent_dim, 1)

    def forward(self, history_features, actions):
        z = self.encoder(history_features).tanh()
        policies, values, rewards = [], [], []
        for j in range(actions.shape[1] + 1):
            policies.append(self.policy(z))
            values.append(self.value(z).squeeze(-1))
            if j < actions.shape[1]:
                action = torch.nn.functional.one_hot(actions[:, j], self.n_actions).to(z)
                za = torch.cat((z, action), dim=-1)
                rewards.append(self.reward(za).squeeze(-1))
                z = self.dynamics(za).tanh()
        reward_seq = torch.stack(rewards, 1) if rewards else z.new_empty((z.shape[0], 0))
        return torch.stack(policies, 1), torch.stack(values, 1), reward_seq

## Example: recurrent prediction core only; ❌full agent, ❌observation decoder.
model = LatentModel(observation_dim=4, latent_dim=8, n_actions=3)
logits, values, rewards = model(torch.zeros(2, 4), torch.tensor([[0, 1], [1, 2]]))
assert logits.shape == (2, 3, 3)
assert values.shape == (2, 3) and rewards.shape == (2, 2)
```
````

```{attention} Q&A
:class: dropdown
*AlphaZero vs MuZero?*

- AlphaZero: known transitions & game rules → learn policy/value.
- MuZero: learned latent transitions & rewards → learn representation/policy/value jointly.
- MuZero ❌requires pixel reconstruction or physically meaningful latent states.

*MuZero vs Dreamer?*

- Both: learned latent dynamics, ❌observation-space rollouts.
- MuZero: latent trained only by reward/value/policy targets → tree search at action time.
- Dreamer: latent trained by reconstruction + reward + KL → amortized actor, ❌search.

*Is the latent model a calibrated stochastic simulator?*

- ❌Not necessarily. Original MuZero: deterministic latent dynamics, ❌explicit chance-outcome distribution.
- Good predictions on trained action sequences ❌guarantee all counterfactuals.
- Root legal-action masking uses env info. Original MuZero ❌applies rule-based legal-action masks at internal nodes.

*Does training differentiate through the tree search?*

- ❌. Backprop through the recurrent prediction unroll toward search-generated targets.
- Search → supervision. Latent predictions still suffer model error & limited coverage.
```

&nbsp;
