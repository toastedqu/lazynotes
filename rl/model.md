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
- $H$: Planning or model-rollout horizon; distinct from task terminal time $T$.
- $\hat V(s)$: Leaf value estimate at a nonterminal planning cutoff.

Override: $a$ denotes an RL action, possibly vector-valued, rather than a generic scalar.

&nbsp;

## Foundations

### Model-Based RL
- **Name**: Model-Based Reinforcement Learning
- **What**: Control using action-conditioned predictions of future consequences.
- **Why**: Real interaction can be costly; hypothetical experience can inform decisions.
- **How**:
    1. Obtain a model: known rules/physics/simulator, learned predictions, or a combination.
    2. Use predictions to improve values/policies, search from the current state, or optimize action sequences.
    3. Act in the environment; update learned components from the observed outcome.

```{dropdown} Table: Model Usage
| Approach | Model access | Computation | Deployed action selection |
|:--|:--|:--|:--|
| Model-free Q-learning / actor-critic | No predictive environment model required | Learn from real or recorded transitions | Value maximization / actor |
| Dynamic programming | Explicit transition probabilities & rewards | Expected backups across states | Stored policy / value maximization |
| Dyna-Q | Learned simulator | Real + simulated value updates | Value maximization |
| Expectimax | Enumerable model outcomes | Decision/chance lookahead | Root action |
| MCTS with UCT | Generative simulator | Selective sampled lookahead | Root visit counts |
| MPC / PETS | Known / learned dynamics | Optimize a finite action sequence | First action, then replan |
| MBPO | Learned probabilistic ensemble | Short simulated data → actor-critic | Actor; no online search required |
| AlphaZero / MuZero | Known rules / learned latent dynamics | Search + policy/value learning | Search, optionally distilled policy |
```

```{attention} Q&A
:class: dropdown
*Planning versus learning?*

- **Planning**: Improve decisions using a model; learning a model is not required.
- **Learning**: Improve estimates from experience; a predictive model is not required.
- Known-model planning can involve no learning. Model-based RL can combine model learning, planning & direct value/policy learning.

*Does training in a simulator make an algorithm model-based?*

- Not automatically. A simulator used only as the environment still supports model-free learning.
- Model-based usage queries hypothetical consequences for planning or synthetic updates; a generative model can start from specified states without consuming real interaction.
- A software simulator may be exact for its game yet inaccurate for the physical system it approximates.

*What counts as a model?*

- A full probability table, a sample generator, deterministic physics, or action-conditioned latent dynamics.
- A value function alone predicts return, not counterfactual transitions.
- No requirement for differentiability, observation reconstruction, or an explicit density.

*Pros & cons?*

- Reuse dynamics across policies or reward functions; trade extra computation for fewer real interactions.
- No universal sample-efficiency advantage: model misspecification, fitting cost & planning cost can dominate.
- Separately learned reward functions can change while dynamics stay reusable; task-specific latent models may not transfer.
```

&nbsp;

## Learning Environment Models

### Transition, Reward & Termination Models
- **What**: Supervised predictors of action-conditioned environment outcomes.
- **Why**: Planning needs consequences unavailable from a value estimate alone.
- **How**:
    1. Collect $(s_t,a_t,r_{t+1},s_{t+1},d_t)$ across visited states/actions.
    2. Fit deterministic outcomes or conditional distributions.
    3. Query hypothetical transitions; retain known reward/termination rules instead of relearning them.
    4. Collect fresh real data where the evolving controller operates.

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

- The joint outcome model preserves dependencies between next state, reward & termination.
- Separate reward & termination heads assume conditional independence given their inputs; include additional conditioning when that assumption fails.

Training:

$$
\mathcal{L}_{\text{joint}}(\psi)
=-\frac1m\sum_{i=1}^{m}
\log\hat M_\psi(s_i',r_i',d_i|s_i,a_i)
$$

- Categorical next state → cross-entropy; tabular maximum likelihood → empirical transition frequencies.
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
- Stochastic reward → analogous scalar Gaussian loss or an appropriate categorical/distributional model.
- Fixed Gaussian variance → squared error; learned variance also pays the log-variance term.

$$
\operatorname{BCE}(d,q)=-d\log q-(1-d)\log(1-q)
$$

- $q$: Predicted probability of true termination.
- Time-limit or rollout cutoff: $c_t=1$ need not imply $d_t=1$.
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

- Squared-error regression learns a conditional mean, not a representative sampled outcome.
- Two possible next positions on opposite sides of an obstacle → their mean can lie inside it.
- Gaussian heads capture state-dependent spread, not arbitrary multimodality; use mixtures, categorical outcomes or richer generators when needed.
- In the code, `stochastic=False` gives deterministic state/reward predictions; termination remains Bernoulli. Variance heads then have no statistical interpretation.

*Must rewards be sampled?*

- For risk-neutral expected return with a Markov state, the correct conditional mean reward suffices.
- Risk-sensitive criteria or reward-dependent state inference require more than its mean.
- If reward & termination remain dependent given $(s,a,s')$, the code's separate heads cannot reproduce their joint distribution.

*Why learn termination?*

- Missing termination → fictitious future reward after failure/success; premature termination → suppressed future value.
- Use true termination labels, not generic `done` flags merging termination & truncation.
- For autoresetting environments, store the final observation, not the next episode's reset observation.
- A genuine finite-horizon task needs remaining time in the state; its endpoint is task termination, not merely a collector cutoff.

*What should be checked before trusting a model?*

- Held-out one-step likelihood/error, reward error & termination calibration; also free-running multi-step predictions from real states.
- Split correlated data by trajectories/time, not only shuffled adjacent transitions.
- Check coverage of proposed actions, rare terminal events & constraints; low average prediction error does not imply good control.
- Partial observability → model an adequate history/belief/latent state, not an aliased observation as if it were Markov.
```

&nbsp;

### Model Error & Uncertainty
- **What**: Mismatch between predicted consequences & the real environment.
- **Why**: A planner optimizes model predictions, including their mistakes.
- **How**:
    1. Separate random outcomes from uncertainty about the dynamics.
    2. Diagnose errors on states/actions the current planner actually proposes.
    3. Restrict unsupported predictions; refresh real data & shorten untrusted rollouts.

```{note} Math
:class: dropdown
- Fixed action sequence; same initial state; deterministic dynamics $L$-Lipschitz in state; uniform one-step error at most $\varepsilon$:

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
- Feedback actions require a bound on the composed closed-loop dynamics, not just fixed-action dynamics.

- Finite discounted MDPs; same reward bound $|r|,|\hat r|\le R_{\max}$; termination represented by an absorbing zero-reward state:

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
- $\varepsilon_p$: Worst-case transition error in $\ell_1$; twice total variation.
- $\hat V^\pi$: Policy value in the learned model, not a leaf approximator.
- $\Delta$: Uniform policy-value error bound.

$$
V^*(s)-V^{\hat\pi}(s)\le2\Delta
$$

- $V^*$: True optimal value.
- $\hat\pi$: Exactly optimal policy in the learned model.
- Approximate planning adds its model-space suboptimality; dataset-average errors do not establish these uniform bounds.

- Ensemble prediction of one scalar output at a fixed state/action:

$$
\operatorname{Var}(Z)
=\mathbb{E}_E[\operatorname{Var}(Z|E)]
+\operatorname{Var}_E(\mathbb{E}[Z|E])
$$

- $Z$: Predicted scalar outcome.
- $E$: Random ensemble-member index.
- First term: Within-model noise; second: Between-model disagreement.
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

3. Compare the true-optimal policy with the model-optimal policy. Each value translation costs at most $\Delta$; model optimality makes the middle difference nonpositive.
```

```{attention} Q&A
:class: dropdown
*Epistemic versus aleatoric uncertainty?*

- **Epistemic**: Uncertainty about dynamics/rewards from limited data; reducible with informative observations.
- **Aleatoric**: Outcome randomness conditional on the modeled information; persists even with the correct conditional distribution.
- Hidden state can appear as randomness under insufficient observations; richer state information may reduce it.
- Ensemble disagreement is a proxy, not calibrated Bayesian certainty; all members can share the same bias.

*Why does small one-step error become a bad policy?*

- Prediction errors shift the next input away from the training distribution; later predictions then rely on unsupported inputs.
- Optimization selects overestimated trajectories: false rewards, impossible shortcuts, or missing termination.
- More search can exploit model errors more effectively. More simulations remove sampling error, not model bias.

*What helps?*

- Known reward/termination/constraint rules when available; recent real data; bootstrap ensembles; short horizons; conservative treatment of unsupported actions.
- Penalize epistemic uncertainty when caution is appropriate; high aleatoric variance alone is not evidence that collecting more data will help.
- Short rollouts reduce accumulated error, not first-step out-of-distribution error.
- A reward model correct on the dataset can still be exploited outside it; a wrong task reward is a separate specification problem.
```

&nbsp;

#### PETS
- **Name**: Probabilistic Ensembles with Trajectory Sampling {cite:p}`chua2018deep`
- **What**: Ensemble dynamics + particle-based model predictive control.
- **Why**: A mean trajectory discards uncertainty relevant to action choice.
- **How**:
    1. Train probabilistic dynamics models on bootstrap-resampled real datasets.
    2. Propagate particles through candidate action sequences.
    3. Optimize mean predicted return with [CEM](#cem); execute the first action & replan.

```{note} Math
:class: dropdown
Process:

$$
\hat J(\mathbf{u})=\frac1P\sum_{i=1}^{P}
\sum_{h=0}^{H-1}\gamma^h r_{h+1}^{(i)}
$$

- $\mathbf{u}$: Candidate open-loop action sequence.
- $P$: Number of trajectory particles.
- $r_{h+1}^{(i)}$: Particle $i$ reward; zero after its termination.
- Fixed-member propagation: select one ensemble member per particle, then sample fresh transition noise at each step.
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

## Example: model callbacks return (next_state, reward, true_termination), without mutating state.
score = ParticleReturns([lambda s, a: (s + a, a, False)], particles=2, gamma=0.9)
assert abs(score(0, [1, 1]) - 1.9) < 1e-12
```
````

```{attention} Q&A
:class: dropdown
*Does particle sampling make control risk-averse?*

- No. Averaging returns optimizes an estimated expectation, not a safety guarantee.
- The code supplies a generic fixed-member particle evaluator; learned callbacks can wrap `EnvironmentModel.sample`.
- Leaf bootstrapping is optional here, not required by PETS.

*Must the ensemble member stay fixed?*

- PETS studies both resampling each step & retaining one member throughout a particle trajectory.
- Retaining a member preserves one dynamics hypothesis over time; neither choice guarantees calibrated uncertainty.
```

&nbsp;

## Learning from Simulated Experience

### Dyna-Q
- **What**: Q-learning from real experience & a learned simulator {cite:p}`sutton1990integrated`.
- **Why**: A real transition can improve decisions beyond its single direct update.
- **How**:
    1. Act & apply a real [Q-learning](value.md#q-learning) update.
    2. Update the transition/reward/termination model.
    3. Sample previously observed state-action pairs; generate modeled outcomes.
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
- Terminal transition → target $r'$; artificial simulation cutoff alone does not remove the bootstrap.
- Planning budget $N_{\text{plan}}$: Number of simulated backups per real transition.
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

- Model-based architecture using a model-free update rule; direct & simulated learning share $Q$.
- The learned controller can act without consulting the model online.

*How is this different from experience replay?*

- Replay reuses recorded outcomes; Dyna generates outcomes from a model.
- A deterministic last-observation table can make sampled Dyna updates numerically identical to replaying those entries.
- A probabilistic model can generate alternative outcomes or average over them; learned function approximation can generalize to unseen pairs.

*What breaks in the compact implementation?*

- Assumes stationary deterministic outcomes, hashable states & the same finite legal action set at every nonterminal state.
- Stochastic environments require outcome counts/distributions or a generative model; overwriting one random outcome is not a transition-distribution estimator.
- Planning reuses knowledge, not new evidence. Unknown shortcuts need real exploration; stale models need new observations.
```

&nbsp;

#### Prioritized Sweeping
- **What**: Model-based backups ordered by value-change impact {cite:p}`moore1993prioritized`.
- **Why**: Uniform planning spends updates on already settled estimates.
- **How**:
    1. Queue state-action pairs with large model Bellman residuals.
    2. Back up the highest-priority pair.
    3. Reconsider predecessor pairs whose targets depend on the changed state's value.

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
- Enqueue residuals above threshold $\eta>0$; stochastic models use expected backups or suitable estimates.
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
*Prioritized sweeping versus prioritized replay?*

- Sweeping uses model predecessor structure to propagate consequences backward.
- Prioritized replay resamples stored transitions by learning priority; no predecessor model required.

*Costs & limits?*

- Maintain reverse dependencies & a queue; changing model edges must remove obsolete predecessors.
- Compact code permits duplicate heap entries & recomputes targets on pop; duplicates consume budget. An indexed priority queue avoids that overhead at scale.
- No direct real backup in this variant: observations enter the queue first. A zero planning budget therefore performs no value update.
```

&nbsp;

### MBPO
- **Name**: Model-Based Policy Optimization {cite:p}`janner2019trust`
- **What**: Off-policy actor-critic learning from real data & short branched model rollouts.
- **Why**: Full-episode imagination couples long tasks to unreliable model horizons.
- **How**:
    1. Fit a probabilistic ensemble on real transitions.
    2. Start short model rollouts from states sampled from real replay.
    3. Sample actions from the current policy; place imagined transitions in model replay.
    4. Train [SAC](policy.md#sac) on a mixture of real & model data; refresh both model & synthetic data.

```{note} Math
:class: dropdown
Process:

$$
s_0\sim\mathcal{D},\qquad
a_h\sim\pi(\cdot|s_h),\qquad
(s_{h+1},r_{h+1},d_h)\sim\hat M_\psi(\cdot|s_h,a_h),
\quad 0\le h<H.
$$

- Here $s_0$ is a replay branch point, not necessarily an episode-initial state.
- End on $d_h=1$ or at horizon $H$; the latter has $c_h=1$ but $d_h=0$.
- At a nonterminal model cutoff, critic targets retain their next-state bootstrap.
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

## Example: a one-step branch ends collection, not the underlying task.
rollout = BranchedRollouts([lambda s, a: (s + a, 1.0, False)],
                          policy=lambda s: 1, horizon=1)
sample = rollout([0, 4], branches=1)[0]
assert sample[-2:] == (False, True)
```
````

```{attention} Q&A
:class: dropdown
*What makes it different from online MPC?*

- The model supplies training transitions; the actor selects deployed actions without solving a planning problem.
- The snippet implements branching only; model fitting, replay mixing & SAC updates are separate components.

*Why not make the rollouts arbitrarily long?*

- More synthetic transitions also mean greater accumulated model bias.
- Short branches reduce state drift; new policy actions can still leave the model's support immediately.
- Branch horizon & synthetic-data ratio are control knobs, not universal defaults.
```

&nbsp;

## Decision-Time Search

### Lookahead & Expectimax
- **What**: Finite-horizon decision/chance expansion with leaf-value bootstrapping.
- **Why**: Full-state planning is unnecessary when only the current action is needed.
- **How**:
    1. Expand legal actions at each decision node.
    2. Average possible environment outcomes at chance nodes.
    3. Stop at termination or planning depth; use zero or a leaf value respectively.
    4. Back up expected returns; select the maximizing root action.

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
- All terminal-state values are zero, including $V_0$; the incoming terminal reward is counted once.

$$
\|V_H-V^*\|_\infty\le\gamma^H\|\hat V-V^*\|_\infty
$$

- $V^*$: Optimal value in an exact known model.
- Assumes exact maximization/expectation & bounded leaf error; learned dynamics add model error.
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

## Example: expectation over outcomes, not maximization over lucky outcomes.
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
*Why maximize actions but average outcomes?*

- Actions are controlled; environment randomness is not.
- Maximizing chance outcomes assumes control over luck. Minimizing them instead changes the objective to a worst-case problem.
- Adversarial two-player search introduces opponent decisions; ordinary single-agent RL does not alternate reward signs.

*Why not evaluate a single fixed action sequence?*

- Expectimax permits later actions to depend on the observed successor state.
- A fixed sequence is open-loop; it cannot adapt within its hypothetical future.
- Deterministic dynamics remove chance branching, not the distinction between task termination & a search cutoff.

*What limits exhaustive search?*

- With at most $A$ actions & $B$ outcomes per action, an unmerged depth-$H$ tree has exponential growth $O((AB)^H)$.
- Memoization over state & remaining depth can share repeated subproblems when the model is fixed.
- Better leaf values trade learned approximation for shallower search; omitted tails are not automatically zero.
```

&nbsp;

### MCTS with UCT
- **Name**: Monte Carlo Tree Search with Upper Confidence bounds applied to Trees {cite:p}`kocsis2006bandit`
- **What**: Adaptive sampled lookahead using bandit-guided action selection.
- **Why**: Exhaustive decision/chance expansion spends computation on every branch.
- **How**:
    1. **Selection**: Choose actions using value estimates + exploration bonuses.
    2. **Chance sampling**: Draw an actual model outcome after each chosen action.
    3. **Expansion**: Add a newly reached decision node; estimate its continuation by rollout or leaf value.
    4. **Backup**: Propagate discounted sampled returns through visited decision edges.
    5. Select a root action by visit count; the exploration bonus guides search, not execution.

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
- $C>0$: Search exploration coefficient; depends on return scale.
- $N(s)$: Decision-node visits.
- $N(s,a)$: Visits to the action edge.
- Unvisited actions are tried before applying the finite-count expression.

$$
g_h=r_{h+1}+\gamma(1-d_h)g_{h+1},\qquad
\bar Q(s_h,a_h)\leftarrow\bar Q(s_h,a_h)
+\frac{g_h-\bar Q(s_h,a_h)}{N(s_h,a_h)}
$$

- $g_h$: Sampled return from search depth $h$.
- $g_H$: Leaf estimate at a nonterminal cutoff; zero after true termination.
- Counts increment before the running-mean update.
- Decision node → action edge → sampled chance outcome → next decision node.
- The same action can reach multiple outcome children; outcome probabilities are respected by simulator sampling.
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

## Example: independent stochastic terminal rewards; no adversarial sign flip.
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
- Callbacks return fresh outcomes without changing the real environment; empty legal-action sets denote terminal states.
- Outcomes share a continuation node by next state; sampled rewards remain edge-specific returns, not cached deterministic rewards.
- Continuous unique next states can prevent revisits; this compact finite-outcome tree is not a continuous-state search solution.

*What can more simulations guarantee?*

- Under finite-horizon finite-action bounded-return assumptions, UCT consistency concerns the supplied model & leaf objective.
- Finite-budget search may miss delayed rewards; guarantees are not practical runtime bounds.
- Incorrect model/leaf values remain incorrect with unlimited sampling.

*Why no sign changes?*

- All returns use the same single-agent reward convention.
- Two-player zero-sum values expressed from the player-to-move perspective need consistent perspective conversion, but that is a different backup convention.

*How do policy/value networks help?*

- Policy priors guide action exploration; value heads replace or shorten random rollouts.
- Prior-guided search is not identical to the UCT formula above; AlphaZero uses a different exploration bonus.
```

&nbsp;

## Receding-Horizon Control

### MPC
- **Name**: Model Predictive Control
- **What**: Repeated finite-horizon optimization with first-action execution.
- **Why**: A long open-loop plan cannot correct deviations after it starts.
- **How**:
    1. Observe the current state; optimize a candidate future action sequence.
    2. Execute only its first action.
    3. Observe the actual successor; shift/warm-start the proposal & solve again.
    4. Learn the model from new transitions if dynamics are unknown.

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
- Known simulators replace $\hat M_\psi$ directly.
- Action bounds restrict every $u_h$; state constraints require predicted-state checks or an explicit constrained formulation.
```

```{note} Example
:class: dropdown
- Plan two pushes to place an object at a target.
- Execute the first; friction leaves it short.
- Replan from the measured position rather than blindly applying the old second push.
- A wrong friction model can still mislead each solve; replanning is correction, not model-error elimination.
```

```{attention} Q&A
:class: dropdown
*Is MPC itself a learning algorithm?*

- No. Known dynamics + repeated optimization require no training.
- Learned dynamics make it a model-based RL/control component.
- The solve may use gradients, quadratic programming or sampling; model-based does not mean differentiable.
- The [CEM implementation](#cem) below includes the first-action/replanning loop.

*Open-loop optimization but closed-loop control?*

- Within one solve, a candidate sequence does not adapt to hypothetical outcomes.
- Across real steps, state feedback changes the next solve.
- Under stochastic dynamics, this is generally not equivalent to optimizing all contingent future policies.

*How long should the horizon be?*

- Longer → more delayed consequences considered, more optimization cost & accumulated model error.
- Shorter → easier search & less drift, greater dependence on the terminal value.
- A leaf value must match the reward/discount convention; setting it to zero deliberately truncates the objective.

*Do replanning or penalties guarantee safety/stability?*

- No. Action clipping enforces only action bounds, not state constraints.
- Soft penalties trade violations against reward; finite candidate sampling can miss feasible/safe solutions.
- Hard guarantees need suitable model/uncertainty assumptions, constrained optimization, feasibility handling & often terminal conditions or a verified fallback.
- Delayed computation & state-estimation errors also break idealized instantaneous-replanning assumptions.
```

&nbsp;

#### CEM
- **Name**: Cross-Entropy Method {cite:p}`rubinstein1999cross`
- **What**: Iterative proposal-distribution fitting to high-scoring samples.
- **Why**: Nonlinear action-sequence objectives may lack useful derivatives.
- **How**:
    1. Sample candidate sequences from a proposal distribution.
    2. Score them with model rollouts; retain an elite subset.
    3. Refit proposal parameters to elites; repeat within the planning budget.
    4. Return a candidate's first action; re-optimize after the real observation.

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

- $\mathbf{m}$: Diagonal-Gaussian proposal mean; not the behavior policy $\mu$.
- $\boldsymbol{\sigma}^2$: Coordinatewise proposal variance.
- $\odot2$: Elementwise square.
- These are unconstrained Gaussian maximum-likelihood updates; clipping bounded actions then matching moments is a practical heuristic, not exact truncated-Gaussian maximum likelihood.
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

## Example: scalar-action control; budgets illustrate the code, not recommended defaults.
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
*Random shooting versus CEM?*

- Random shooting samples once from a fixed proposal; CEM adapts the proposal using elites.
- Scalar code above generalizes coordinatewise to vector actions; the proposal's diagonal covariance ignores correlations.

*Why the name cross-entropy?*

- Fit the proposal toward a distribution concentrated on elite events by maximizing their log likelihood.
- No supervised classification problem or actor network is required.

*Failure modes?*

- Premature variance collapse, multimodal optima & high-dimensional sequences; a variance floor preserves local sampling, not global coverage.
- No global-optimality guarantee for a finite population/budget.
- Noisy model scores can select lucky candidates; use enough particles & re-evaluate finalists when needed.
- Clipping creates mass on action bounds; state feasibility remains the caller's separate responsibility.
```

&nbsp;

## Learned Representations & Search

### AlphaZero
- **What**: Self-play policy/value learning guided by known-rule tree search {cite:p}`silver2018general`.
- **Why**: Search can improve a learned policy; its decisions can then train the network.
- **How**:
    1. Use exact game rules for transitions, legal moves & terminal outcomes.
    2. Guide search with policy priors & neural leaf values.
    3. Store search visit distributions & self-play outcomes.
    4. Train policy/value predictions toward those targets; repeat.

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
- $z$: Self-play outcome from the player-to-move perspective at stored state $s$.
- $\pi_{\text{search}}$: Normalized root-visit target, optionally temperature-adjusted.
- $\theta$: Policy-head params.
- $\vartheta$: All shared/network params.
- $\lambda$: Weight-decay coefficient.
- Board-game outcome prediction is episodic & undiscounted; it is not the discounted leaf target assumed elsewhere on this page.
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

## Example: the distillation loss only; game simulation & prior-guided search are separate.
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

- No. It learns policy/value predictions; search uses known transitions & legal moves.
- Its exploration uses policy priors, not plain UCT; the compact code implements distillation, not the complete search system.
- Outcome signs follow the stored player's perspective; do not copy that convention into single-agent backups.
```

&nbsp;

### MuZero
- **What**: Search-guided policy/value/reward learning through latent dynamics {cite:p}`schrittwieser2020mastering`.
- **Why**: Exact dynamics may be unavailable; reconstructing every observation detail is unnecessary for planning.
- **How**:
    1. Encode observation history into a latent root state.
    2. Apply candidate actions to recurrent latent dynamics.
    3. Search using predicted rewards, policy priors & values.
    4. Unroll real action sequences; train against observed rewards, return targets & search policies.

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
- $z_t^j$: Latent state after $j$ hypothetical actions from root $t$; overrides the page's generic feature index $j$ locally.
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

- $K$: Number of training unroll steps; not necessarily search horizon $H$.
- $v^{\text{target}}_{t+j}$: Observed return or bootstrapped value target.
- $\pi^{\text{search}}_{t+j}$: Search-derived policy target.
- $\ell_v$: Value prediction loss.
- $\ell_\pi$: Policy cross-entropy.
- $\ell_r$: Reward prediction loss.
- $\lambda$: Regularization coefficient.
- Targets stop/mask or use absorbing-state conventions past true termination.
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

## Example: recurrent prediction core, not a full MuZero agent; no observation decoder.
model = LatentModel(observation_dim=4, latent_dim=8, n_actions=3)
logits, values, rewards = model(torch.zeros(2, 4), torch.tensor([[0, 1], [1, 2]]))
assert logits.shape == (2, 3, 3)
assert values.shape == (2, 3) and rewards.shape == (2, 2)
```
````

```{attention} Q&A
:class: dropdown
*AlphaZero versus MuZero?*

- AlphaZero: known-state transitions & game rules; learn policy/value.
- MuZero: learned latent transitions & rewards; learn representation/policy/value jointly.
- MuZero does not require pixel reconstruction or literal physical-state semantics.

*Is the latent model a calibrated stochastic simulator?*

- Not necessarily. Original MuZero uses deterministic latent dynamics; no explicit chance-outcome distribution.
- Good prediction on trained action sequences does not guarantee all counterfactuals.
- Root legal-action masking uses environment information; original MuZero does not apply rule-based legal-action masks at internal nodes.

*Does training differentiate through the tree search?*

- No requirement: backpropagate through the recurrent prediction unroll toward search-generated targets.
- Search generates supervision; latent predictions remain subject to model error & limited coverage.
```

&nbsp;
