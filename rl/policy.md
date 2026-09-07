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

# Policy-Based Methods & Actor–Critic

Direct policy optimization: likelihood-ratio gradients → advantage estimation → constrained updates → deterministic & entropy-regularized control. Foundations from {cite:t}`rlbook`; assumes [MDPs](intro.md#mdp) & [value functions](intro.md#value-functions).

Notations:
- $s_t$: State at time $t$.
- $a_t$: Action taken in $s_t$.
- $r_{t+1}$: Reward received after $a_t$.
- $r(s,a)$: Expected immediate reward $\mathbb E[r_{t+1}\mid s_t=s,a_t=a]$.
- $p(s'\mid s,a)$: Environment transition distribution.
- $\rho_0$: Initial-state distribution.
- $\gamma$: Discount factor; $0\leq\gamma<1$ unless a suitable finite/terminating problem permits $\gamma=1$.
- $T$: True terminal time; infinite for continuing tasks.
- $G_t$: Return $\sum_{l=0}^{T-t-1}\gamma^l r_{t+l+1}$.
- $\pi_\theta$: Target stochastic policy.
- $\mu$: Behavior policy generating data.
- $\theta$: Actor params.
- $\phi$: Critic params.
- $V^\pi$: State value under $\pi$.
- $Q^\pi$: Action value under $\pi$.
- $A^\pi$: Advantage $Q^\pi-V^\pi$.
- $J$: Expected discounted start-state return $\mathbb E_{\rho_0,\pi_\theta}[G_0]$.
- $d_\gamma^\pi$: Normalized discounted state occupancy $(1-\gamma)\sum_{t\geq0}\gamma^t\Pr_\pi(s_t=\cdot)$.
- $d_t$: Indicator of true environment termination after transition $t$.
- $c_t$: Indicator of episode or collected-segment boundary after transition $t$; $d_t\leq c_t$.
- $b_t$: Bootstrap mask $1-d_t$.
- $\delta_t$: One-step temporal-difference residual.
- $\widehat A_t$: Estimated advantage.
- $\widehat R_t$: Critic's return target.
- $\mathcal R$: Replay buffer.
- $\mathcal H$: Policy entropy; differential entropy for continuous actions.
- $\beta$: Nonnegative entropy coefficient; temperature in maximum-entropy control.
- $\theta_{\mathrm{old}}$: Frozen params of the policy that collected a rollout.
- $\bar\phi$: Slowly updated target-critic params.
- $\operatorname{sg}$: Stop-gradient operator; numerical identity with zero derivative.
- $\widehat{\mathbb E}$: Empirical average over a specified sample batch.
- $\omega_t$: Actor sample weight; $\gamma^t$ for episode-based discounted gradients, $1$ for the usual unweighted rollout surrogate.

RL overrides: $s$ & $a$ denote states & actions, possibly vectors; $t,l$ denote time & time offsets; $A^\pi$ denotes an advantage function rather than a matrix.

&nbsp;

## Policy representation

### Categorical policy
- **What**: State-conditioned distribution over finitely many actions.
- **Why**: Learn action preferences without a nondifferentiable greedy selection rule.
- **How**:
    1. Map state features to one logit per valid action.
    2. Normalize logits → probabilities.
    3. Sample for interaction; differentiate the sampled action's log-probability.

```{note} Math
:class: dropdown
Forward:

$$
\pi_\theta(a=j\mid s)
=\frac{\exp z_{\theta,j}(s)}
{\sum_{q=1}^{C}\exp z_{\theta,q}(s)}.
$$
- $z_\theta(s)\in\mathbb R^C$: Logit vector.
- $C$: Number of discrete actions.
- $j$: Action-category index.
- $q$: Summation index over action categories.

Backward:

$$
\frac{\partial\log\pi_\theta(a\mid s)}{\partial z_{\theta,j}(s)}
=\mathbf 1\{a=j\}-\pi_\theta(j\mid s).
$$
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class CategoricalPolicy(nn.Module):
    def __init__(self, state_dim, actions):
        super().__init__()
        self.logits = nn.Linear(state_dim, actions)

    def forward(self, state, valid=None):
        logits = self.logits(state)
        if valid is not None:
            assert valid.shape == logits.shape
            assert valid.any(dim=-1).all()
            logits = logits.masked_fill(~valid, -torch.inf)
        return logits.log_softmax(dim=-1)

    def sample(self, state, valid=None):
        logp = self(state, valid)
        action = torch.multinomial(logp.exp(), 1)
        return action.squeeze(-1), logp.gather(-1, action).squeeze(-1)

## Example
policy = CategoricalPolicy(3, 2)
action, logp = policy.sample(torch.zeros(2, 3))
assert action.shape == logp.shape == (2,)
```
````

```{attention} Q&A
:class: dropdown
*Why not backpropagate through the sampled integer?*

- Sampling is discrete; the likelihood-ratio estimator differentiates $\log\pi_\theta(a\mid s)$ with $a$ held fixed.
- Argmax during training discards the stochastic sampling assumption behind that estimator.

*How do invalid-action masks interact with PPO?*

- Mask before normalization; at least one valid action per state.
- Use the same state-dependent support when storing old & recomputing new log-probabilities.
- A learned, parameter-dependent support change can invalidate likelihood-ratio identities.

*Does stochasticity imply that an optimal MDP policy must randomize?*

- No. Finite discounted fully observed MDPs admit deterministic stationary optimal policies.
- Randomization still supports exploration; entropy constraints, partial-observation restrictions & game-theoretic objectives can change the answer.
```

&nbsp;

### Gaussian policy
- **What**: State-conditioned normal density over continuous actions.
- **Why**: Enumerating candidate actions is impossible in a continuous action space.
- **How**:
    1. Predict a mean & positive scale for each action coordinate.
    2. Transform independent standard-normal noise into an action.
    3. Combine coordinate log-densities into one joint log-density.

```{note} Math
:class: dropdown
Forward:

$$
\begin{align*}
\epsilon&\sim\mathcal N(0,I),\\
a&=m_\theta(s)+\sigma_\theta(s)\odot\epsilon,\\
\log\pi_\theta(a\mid s)
&=-\frac12\sum_{j=1}^{D}
\left[
\frac{(a_j-m_{\theta,j}(s))^2}{\sigma_{\theta,j}(s)^2}
+2\log\sigma_{\theta,j}(s)+\log(2\pi)
\right].
\end{align*}
$$
- $\epsilon\in\mathbb R^D$: Parameter-independent sampling noise.
- $m_\theta(s)\in\mathbb R^D$: Mean vector; subscript distinguishes it from global batch size $m$.
- $\sigma_\theta(s)\in\mathbb R_{>0}^D$: Coordinate standard deviations.
- $D$: Action dimension.
- $\odot$: Elementwise multiplication.

Backward:

$$
\left.\frac{\partial\log\pi_\theta(a\mid s)}
{\partial m_{\theta,j}(s)}\right|_{a\ \mathrm{fixed}}
=\frac{a_j-m_{\theta,j}(s)}{\sigma_{\theta,j}(s)^2}.
$$
```

````{important} Code
:class: dropdown
```python
import math
import torch
import torch.nn as nn

class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.mean = nn.Linear(state_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        mean = self.mean(state)
        return mean, self.log_std.expand_as(mean)

    def log_prob(self, state, action):
        mean, log_std = self(state)
        standardized = (action - mean) * (-log_std).exp()
        return (-0.5 * standardized.square() - log_std
                - 0.5 * math.log(2 * math.pi)).sum(dim=-1)

    def sample(self, state, reparameterize=False):
        mean, log_std = self(state)
        action = mean + log_std.exp() * torch.randn_like(mean)
        if not reparameterize:
            action = action.detach()  ## score gradient holds the action fixed
        return action, self.log_prob(state, action)

## Example
policy = GaussianPolicy(3, 2)
action, logp = policy.sample(torch.zeros(2, 3))
assert action.shape == (2, 2) and logp.shape == (2,)
```
````

```{attention} Q&A
:class: dropdown
*Score-function gradient versus reparameterization gradient?*

- REINFORCE/PPO: stored actions are constants; differentiate their log-density.
- SAC: retain the sampled action's path to actor params; differentiate the critic through that path.
- A reparameterized action accidentally left attached in a score-function loss changes the estimator; the mean gradient of an unsquashed Gaussian's own sampled log-density even cancels.

*What does a diagonal Gaussian assume?*

- Conditional independence of action coordinates given the state; no learned within-state covariance.
- Mean can depend on every state feature. The example uses state-independent trainable scales; state-dependent scales are another parameterization.
- Unimodality & scale parameterization restrict expressivity; finite, positive scales are required.

*Is the joint log-probability a sum or a mean?*

- Sum over action coordinates; average over batch samples.
- Coordinate averaging changes likelihood ratios & entropy, not merely implementation style.
```

&nbsp;

#### Tanh-squashed Gaussian
- **What**: Invertibly transformed Gaussian density inside fixed action bounds.
- **Why**: Gaussian samples are unbounded; actuator commands are not.
- **How**:
    1. Sample an unconstrained latent action.
    2. Apply $\tanh$, then affine scaling to the environment's action range.
    3. Correct the density for the transform's local volume change.

```{note} Math
:class: dropdown
Forward:

$$
\begin{align*}
u&\sim\mathcal N\!\left(m_\theta(s),
             \operatorname{diag}(\sigma_\theta(s)^2)\right),\\
a_j&=h_j+q_j\tanh u_j,\qquad q_j>0,\\
\log\pi_\theta(a\mid s)
&=\log\mathcal N(u;m_\theta(s),\operatorname{diag}(\sigma_\theta(s)^2))
-\sum_{j=1}^{D}\left[\log q_j+\log(1-\tanh^2 u_j)\right].
\end{align*}
$$
- $u\in\mathbb R^D$: Unsquashed latent action.
- $h_j$: Midpoint of coordinate $j$'s action interval.
- $q_j$: Half-width of that interval.

- Stable identity:

    $$
    \log(1-\tanh^2 u)=2\left[\log 2-u-\operatorname{softplus}(-2u)\right].
    $$
```

```{tip} Derivation
:class: dropdown
1. Change of variables: density equals latent density divided by the absolute Jacobian determinant.
2. Coordinatewise transform → diagonal Jacobian with entries $q_j(1-\tanh^2u_j)$.
3. Log determinant → sum of coordinate log-scales; subtract it from the Gaussian log-density.
4. Store $u$ when sampling; avoid an unstable $\operatorname{atanh}$ inversion near action bounds.
```

````{important} Code
:class: dropdown
```python
import math
import torch
import torch.nn.functional as F
## Reuses GaussianPolicy from rl/policy.md

class SquashedGaussian(GaussianPolicy):
    def __init__(self, state_dim, action_dim, low, high):
        super().__init__(state_dim, action_dim)
        low, high = torch.tensor(low, dtype=torch.float32), torch.tensor(
            high, dtype=torch.float32
        )
        assert low.shape == high.shape == (action_dim,)
        assert (high > low).all()
        self.register_buffer("center", (high + low) / 2)
        self.register_buffer("scale", (high - low) / 2)

    def log_prob(self, state, latent):
        base_logp = super().log_prob(state, latent)
        log_jacobian = self.scale.log() + 2 * (
            math.log(2) - latent - F.softplus(-2 * latent)
        )
        return base_logp - log_jacobian.sum(dim=-1)

    def sample(self, state, reparameterize=False):
        mean, log_std = self(state)
        latent = mean + log_std.exp() * torch.randn_like(mean)
        if not reparameterize:
            latent = latent.detach()
        action = self.center + self.scale * latent.tanh()
        return action, self.log_prob(state, latent), latent

## Example
policy = SquashedGaussian(3, 2, [-2, -1], [2, 3])
action, logp, latent = policy.sample(torch.zeros(2, 3))
assert action.shape == (2, 2) and logp.shape == (2,)
assert torch.allclose(logp, policy.log_prob(torch.zeros(2, 3), latent))
```
````

```{attention} Q&A
:class: dropdown
*Why not clip Gaussian actions and evaluate a Gaussian density at the clipped value?*

- Clipping collapses tail intervals onto boundary points → a mixed distribution with boundary probability masses.
- The original Gaussian density at the clipped value is not that distribution's likelihood.
- A separate valid design stores the pre-clip action as the policy action & treats clipping as part of the environment; then likelihoods must refer to that stored action.

*Can PPO omit the Jacobian?*

- In a ratio only: identical fixed invertible transforms & identical stored action → old/new Jacobians cancel.
- Absolute log-densities, SAC entropy & temperature tuning still require the correction.
- Changing bounds between collection & optimization breaks that cancellation.

*Does squashing preserve Gaussian entropy?*

- No. Saturation compresses volume; increasing latent scale need not increase action entropy.
- Differential entropy depends on action units. Affine scaling shifts it by $\sum_j\log q_j$; choose entropy targets in the same coordinates.
- Floating-point $\tanh$ can round to a bound; stored latents & the stable log-Jacobian avoid taking $\log 0$.
```

&nbsp;

## Stochastic policy gradients

### Policy-gradient theorem
- **What**: Exact return gradient expressed through policy scores & action values. {cite:p}`sutton1999policy`
- **Why**: Changing a policy changes its future state distribution; directly differentiating that distribution is impractical.
- **How**:
    1. Differentiate trajectory likelihoods, not environment transitions.
    2. Credit each action only for rewards it can affect.
    3. Replace sampled future returns with conditional action values.

```{note} Math
:class: dropdown
Objective:

$$
J(\theta)=\mathbb E_{\rho_0,\pi_\theta}[G_0].
$$

$$
\begin{align*}
\nabla_\theta J
&=\mathbb E_{\pi_\theta}\left[
\sum_{t=0}^{T-1}\gamma^t
\nabla_\theta\log\pi_\theta(a_t\mid s_t)
Q^{\pi_\theta}(s_t,a_t)\right]\\
&=\frac{1}{1-\gamma}
\mathbb E_{\substack{s\sim d_\gamma^{\pi_\theta}\\a\sim\pi_\theta(\cdot\mid s)}}
\left[\nabla_\theta\log\pi_\theta(a\mid s)Q^{\pi_\theta}(s,a)\right].
\end{align*}
$$

- Second form requires $\gamma<1$; terminal states are padded with absorbing, zero-reward behavior.
- Assumptions: parameter-independent initial distribution & environment; differentiable policy on fixed support; finite required expectations; justified exchange of differentiation & integration.
- Finite-horizon nonstationarity: include remaining time in the state or index policies & values by time.
- The integrand differentiates the policy score only; the theorem has already accounted for future policy effects.
```

```{tip} Derivation
:class: dropdown
1. Factor a trajectory's probability:

    $$
    p_\theta(\xi)=\rho_0(s_0)
    \prod_{t=0}^{T-1}
    \pi_\theta(a_t\mid s_t)P(s_{t+1},r_{t+1}\mid s_t,a_t).
    $$
    - $\xi$: Complete trajectory including rewards.
    - $P$: Joint next-state/reward kernel; state marginal equals $p$.

2. Apply the likelihood-ratio identity:

    $$
    \nabla_\theta J
    =\int p_\theta(\xi)G_0(\xi)\nabla_\theta\log p_\theta(\xi)\,d\xi
    =\mathbb E\left[G_0\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)\right].
    $$

3. Remove past rewards: conditioned on history before $a_t$, their product with the policy score has zero expectation.

    $$
    \mathbb E_{a_t\sim\pi_\theta}
    [\nabla_\theta\log\pi_\theta(a_t\mid s_t)\mid s_t]
    =\nabla_\theta\int\pi_\theta(a\mid s_t)\,da=0.
    $$

4. Factor the remaining discount:

    $$
    \sum_{l=t}^{T-1}\gamma^l r_{l+1}=\gamma^tG_t.
    $$

5. Condition $G_t$ on $(s_t,a_t)$ → $Q^{\pi_\theta}(s_t,a_t)$; collect discounted state visits → occupancy form.
```

```{attention} Q&A
:class: dropdown
*Where did the gradient of state occupancy go?*

- It did not vanish. The recursive future-policy effects are represented by $Q^\pi$ & discounted visitation in the theorem.
- Treating a replay distribution as fixed is a different approximation, not another proof of the same theorem.

*Can the outer $\gamma^t$ be dropped?*

- Not for the exact start-distribution objective with ordinary episode trajectories.
- Sampling states directly from $d_\gamma^\pi$ absorbs the weighting; the factor $1/(1-\gamma)$ remains a policy-independent scale.
- Uniform rollout sampling with discounted advantages & no outer weight is the common practical surrogate; generally not the exact gradient of $J$, nor automatically an exact average-reward gradient.
- Sum time contributions within each episode, then average episodes. With policy-dependent episode lengths, normalizing by the number of observed transitions can change the objective's weighting.

*Does an exact gradient guarantee global convergence?*

- No. The theorem identifies a derivative, not a global optimizer.
- Stochastic-approximation convergence needs additional smoothness, step-size, bounded-noise & sampling assumptions; general neural parameterizations provide no blanket global-optimum guarantee.

*Can any learned critic replace $Q^\pi$ without bias?*

- No. Approximate action values generally bias the actor update.
- A special exception: a score-feature advantage approximator fitted to the appropriate occupancy-weighted least-squares normal equations preserves the expected score–advantage product.
- An arbitrary neural critic trained on temporal-difference targets does not automatically satisfy that compatibility condition.
```

&nbsp;

### REINFORCE
- **Name**: REward Increment = Nonnegative Factor × Offset Reinforcement × Characteristic Eligibility {cite:p}`williams1992simple`
- **What**: Monte Carlo policy-gradient ascent using sampled returns.
- **Why**: Estimate a policy gradient without a learned action-value function or differentiable environment.
- **How**:
    1. Collect complete episodes under the current policy.
    2. Compute reward-to-go for every sampled action.
    3. Increase/decrease action log-probability according to return above/below an action-independent baseline.

```{note} Math
:class: dropdown
Process:

$$
\widehat g=\frac1m\sum_{i=1}^{m}\sum_{t=0}^{T_i-1}
\gamma^t\nabla_\theta\log\pi_\theta(a_t^{(i)}\mid s_t^{(i)})
\left[G_t^{(i)}-v(s_t^{(i)})\right].
$$
- $v(s)$: Action-independent baseline fixed while estimating the actor gradient.
- $T_i$: Terminal time of episode $i$.

$$
\theta\leftarrow\theta+\alpha\widehat g.
$$
- $\alpha$: Positive actor step size.

Objective:

$$
\mathcal L_{\mathrm{actor}}
=-\frac1m\sum_i\sum_t\gamma^t
\log\pi_\theta(a_t^{(i)}\mid s_t^{(i)})
\operatorname{sg}\!\left(G_t^{(i)}-v(s_t^{(i)})\right).
$$

- Minimize this sample loss; its negative gradient is the estimator above.
- With $v=V_\phi$, fit the baseline separately to Monte Carlo returns.
```

```{tip} Derivation
:class: dropdown
1. An action-independent baseline contributes zero expected score:

    $$
    \mathbb E_{a\sim\pi_\theta}
    [v(s)\nabla_\theta\log\pi_\theta(a\mid s)]
    =v(s)\nabla_\theta\int\pi_\theta(a\mid s)\,da=0.
    $$

2. Subtracting $v(s_t)$ leaves the expected policy gradient unchanged.
3. For a single conditional score term, minimize its squared norm's expectation with respect to the scalar baseline:

    $$
    v^*(s)=
    \frac{\mathbb E[G_t\|z_t\|^2\mid s_t=s]}
         {\mathbb E[\|z_t\|^2\mid s_t=s]}.
    $$
    - $z_t$: Policy score $\nabla_\theta\log\pi_\theta(a_t\mid s_t)$.
    - Positive denominator required; a zero score makes that state's baseline irrelevant to this term.

4. $V^\pi(s)$ is a useful baseline, but need not equal this variance-minimizing baseline; score norms weight actions differently.
```

````{important} Code
:class: dropdown
```python
import torch

class ReinforceLoss:
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, logp, returns, baseline, valid):
        ## Padded complete episodes: all inputs [time, episodes].
        assert logp.ndim == 2
        assert logp.shape == returns.shape == baseline.shape == valid.shape
        time = torch.arange(logp.shape[0], device=logp.device)
        weight = logp.new_tensor(self.gamma).pow(time)[:, None]
        advantage = (returns - baseline).detach()
        terms = torch.where(valid, weight * logp * advantage, 0.0)
        return -terms.sum(dim=0).mean()  ## average episodes, not lengths

## Example
logp = torch.tensor([[-0.7], [-0.4]], requires_grad=True)
returns = torch.tensor([[2.0], [2.0]])  ## rewards [1, 2], gamma=0.5
loss = ReinforceLoss(0.5)(logp, returns, torch.zeros_like(logp),
                          torch.ones_like(logp, dtype=torch.bool))
loss.backward()
assert torch.allclose(logp.grad, torch.tensor([[-2.0], [-1.0]]))
```
````

```{attention} Q&A
:class: dropdown
*Why must the baseline be action-independent?*

- The zero-score identity factors out a quantity fixed before the current action.
- Naively subtracting an action-dependent baseline changes the gradient; special corrections are needed to remove that bias.
- A learned baseline sharing actor features must be stop-gradient in the actor term. Fit its value loss through its own computational path.

*Does fitting a baseline on the same rollout preserve exact finite-sample unbiasedness?*

- Not automatically. Fitted params can depend on that rollout's actions & rewards.
- The clean proof conditions on a fixed baseline or one estimated independently of the current action; same-batch fitting is a practical approximation.
- Batch advantage centering/scaling likewise need not preserve the exact finite-sample unbiased estimator.

*Pros & cons?*

- No bootstrap bias; complete on-policy returns give an unbiased gradient under the stated assumptions.
- Long horizons, noisy rewards & rare successful episodes → high variance; learning waits for returns.
- An artificial cutoff is not a complete episode. Bootstrapping its continuation produces an actor–critic estimator, not pure Monte Carlo REINFORCE.
```

&nbsp;

### Actor–critic
- **What**: Policy optimization guided by a learned value estimator.
- **Why**: Full-return policy gradients are noisy & unavailable until an episode finishes.
- **How**:
    1. Critic predicts future return.
    2. Observed reward plus predicted continuation → a training target.
    3. Actor uses the critic's relative action assessment; critic learns toward the target.

```{note} Math
:class: dropdown
Process:

$$
\begin{align*}
\widehat R_t&=r_{t+1}+\gamma b_tV_\phi(s_{t+1}),\\
\delta_t&=\widehat R_t-V_\phi(s_t).
\end{align*}
$$

- With $V_\phi=V^\pi$ & correct terminal handling:

    $$
    \mathbb E[\delta_t\mid s_t,a_t]=A^\pi(s_t,a_t).
    $$

Objective:

$$
\begin{align*}
\mathcal L_{\mathrm{actor}}
&=-\widehat{\mathbb E}
  [\omega_t\log\pi_\theta(a_t\mid s_t)\operatorname{sg}(\delta_t)],\\
\mathcal L_{\mathrm{critic}}
&=\frac12\widehat{\mathbb E}
  [(V_\phi(s_t)-\operatorname{sg}(\widehat R_t))^2].
\end{align*}
$$

- Empirical-average form is a batch surrogate; exact episode-gradient normalization follows REINFORCE.
- Critic update is a semi-gradient: do not differentiate its bootstrap target.
```

````{important} Code
:class: dropdown
```python
import torch

class ActorCriticLoss:
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, logp, value, next_value, reward, terminated, weight):
        assert logp.ndim == 1
        assert all(x.shape == logp.shape for x in
                   (value, next_value, reward, terminated, weight))
        bootstrap = torch.where(terminated, 0.0, next_value.detach())
        target = (reward + self.gamma * bootstrap).detach()
        advantage = (target - value).detach()
        actor = -(weight.detach() * logp * advantage).mean()
        critic = 0.5 * (value - target).square().mean()
        return actor, critic

## Example
logp = torch.tensor([-0.7, -0.4], requires_grad=True)
value = torch.tensor([1.0, 1.0], requires_grad=True)
losses = ActorCriticLoss(0.5)(
    logp, value, torch.tensor([4.0, 99.0]), torch.tensor([1.0, 2.0]),
    torch.tensor([False, True]), torch.ones(2)
)
sum(losses).backward()
assert value.grad is not None and logp.grad is not None
```
````

```{attention} Q&A
:class: dropdown
*Is REINFORCE with a learned baseline already actor–critic?*

- Terminology varies. Here, actor–critic uses learned values in the actor's return/advantage estimate, typically by bootstrapping.
- A Monte Carlo baseline alone reduces variance without requiring a bootstrapped return estimate.

*Where does actor bias enter?*

- An imperfect $V_\phi(s_t)$ used only as a fixed state baseline does not bias the score expectation.
- An imperfect $V_\phi(s_{t+1})$ inside the return target generally does.
- Exact on-policy values recover the conditional advantage identity; stale policies, critic error & truncated importance weights break that ideal.

*Should actor & critic share parameters?*

- Shared features save computation but couple the optimization problems.
- Separate heads do not prevent value-loss gradients from changing shared actor features.
- Detach advantages/targets; combine only the intended actor, critic & optional entropy gradients.

*What convergence claim survives deep actor–critic?*

- Classical convergence results require specific approximation, sampling & often two-timescale assumptions.
- Simultaneously updating arbitrary neural actor/critic networks is not covered by a generic convergence guarantee.
```

&nbsp;

#### A2C
- **Name**: Advantage Actor–Critic
- **What**: Synchronous batched advantage actor–critic.
- **Why**: Collect decorrelated experience while keeping a shared policy version during each rollout.
- **How**:
    1. Step several environments using the same actor snapshot.
    2. Compute multi-step or GAE advantages & return targets.
    3. Aggregate gradients; apply one synchronized actor/critic update.

```{attention} Q&A
:class: dropdown
*How does A2C differ from vanilla actor–critic?*

- Same advantage-weighted update; synchronous parallel collection & batched optimization specify the implementation pattern.
- Reuse `ActorCriticLoss`, replacing its one-step advantage/target with the multi-step estimates below.
- One rollout update is on-policy at collection time; repeated uncorrected actor steps on that batch introduce policy mismatch.

*Pros & cons?*

- Synchronized policy versions; vectorized computation; no asynchronous gradient races.
- Workers may wait for the slowest environment; no long-lived replay for the plain on-policy estimator.
```

&nbsp;

#### A3C
- **Name**: Asynchronous Advantage Actor–Critic {cite:p}`mnih2016asynchronous`
- **What**: Parallel actor–critic workers applying asynchronous shared updates.
- **Why**: Gather diverse experience without synchronizing every worker's rollout.
- **How**:
    1. Each worker copies shared actor/critic params into local networks.
    2. Collect a short trajectory; bootstrap its end when nonterminal.
    3. Compute local multi-step policy & value gradients, optionally adding an entropy bonus.
    4. Apply gradients to shared params; refresh the worker's local copy.

```{attention} Q&A
:class: dropdown
*A2C versus A3C?*

- A2C: synchronized batch & shared policy snapshot.
- A3C: independent workers & potentially stale gradients; no global barrier.
- Both use advantage actor–critic; asynchrony is a scheduling difference, not a new policy-gradient theorem.

*Is A3C off-policy?*

- Locally collected trajectories follow each worker's local policy; the shared policy may change before its gradient is applied.
- Usually classified as on-policy without replay, but asynchronous staleness introduces a mismatch. No exact current-policy-gradient claim follows.

*Does an entropy bonus make it SAC?*

- No. A local actor entropy bonus encourages randomness at sampled states.
- SAC also includes future entropy in its critic's Bellman targets & performs off-policy soft policy improvement.
```

&nbsp;

### n-step advantage
- **What**: Multi-step bootstrapped return minus a state-value baseline.
- **Why**: Trade noisy full returns against dependence on a one-step critic prediction.
- **How**:
    1. Accumulate rewards until the chosen horizon or segment boundary.
    2. Bootstrap only when the endpoint is not truly terminal.
    3. Subtract the starting state's frozen value prediction.

```{note} Math
:class: dropdown
Process:

$$
\begin{align*}
\widehat R_t^{(h)}
&=\sum_{l=0}^{h-1}\gamma^l r_{t+l+1}
+\gamma^h b_{t+h-1}V_\phi(s_{t+h}),\\
\widehat A_t^{(h)}
&=\widehat R_t^{(h)}-V_\phi(s_t).
\end{align*}
$$
- $h$: Actual number of collected transitions from $t$; capped by the chosen horizon & the first $c=1$ boundary.

- Never cross an episode reset inside this sum.
- Both value predictions come from the same frozen snapshot when preparing actor targets.
- Same return as [n-step value learning](value.md#n-step-returns); the additional subtraction forms an advantage.
```

````{important} Code
:class: dropdown
```python
import torch

class NStepAdvantage:
    def __init__(self, gamma, horizon):
        assert horizon >= 1
        self.gamma, self.horizon = gamma, horizon

    @torch.no_grad()
    def __call__(self, reward, value, next_value, terminated, boundary):
        ## Each array is [time, environments]; next_value uses final observations.
        assert reward.ndim == 2
        assert all(x.shape == reward.shape for x in
                   (value, next_value, terminated, boundary))
        assert ((~terminated) | boundary).all() and boundary[-1].all()
        targets = torch.zeros_like(reward)
        length = reward.shape[0]
        for t in range(length):
            discount = torch.ones_like(reward[t])
            for offset in range(min(self.horizon, length - t)):
                end = t + offset
                stop = (boundary[end] | (offset == self.horizon - 1)
                        | (end == length - 1))
                boot = torch.where(terminated[end], 0.0, next_value[end])
                targets[t] += discount * (
                    reward[end] + self.gamma * stop * boot
                )
                discount = discount * self.gamma * (~stop)
        return targets - value, targets

## Example
r = torch.tensor([[1.0], [2.0], [100.0]])
v = torch.tensor([[4.0], [5.0], [8.0]])
nv = torch.tensor([[5.0], [6.0], [0.0]])
terminal = torch.tensor([[False], [False], [True]])
boundary = torch.tensor([[False], [True], [True]])
adv, target = NStepAdvantage(0.5, 2)(r, v, nv, terminal, boundary)
assert torch.allclose(target[:, 0], torch.tensor([3.5, 5.0, 100.0]))
```
````

```{attention} Q&A
:class: dropdown
*Termination versus truncation?*

- True terminal: $d_t=c_t=1$ → neither bootstrap nor continue the trace.
- Artificial time limit or rollout cutoff: $d_t=0,c_t=1$ → bootstrap, but do not continue into the next collected segment.
- Ordinary transition: $d_t=c_t=0$ → bootstrap & continue.
- An actual finite-horizon task is terminal at its horizon; remaining time must be represented when needed for the Markov property.

*Which next observation should be valued after an automatic reset?*

- The final observation of the ended segment, not the next episode's reset observation.
- Reset observations leak unrelated future values into truncated-episode targets.

*How does horizon affect bias & variance?*

- Longer horizons usually reduce dependence on bootstrap error & increase sampled-return variance.
- At a true terminal endpoint: full return minus a fixed baseline → no bootstrap bias.
- At a nonterminal cutoff: the final critic error remains, scaled by $\gamma^h$.

*Can replayed behavior-policy trajectories supply these returns unchanged?*

- Only when their continuation policy is the policy being evaluated, or when a justified correction is supplied.
- Uncorrected multi-step returns follow behavior actions inside the segment; one final target-policy bootstrap does not fix that mismatch.
```

&nbsp;

### GAE
- **Name**: Generalized Advantage Estimation {cite:p}`schulman2016high`
- **What**: Exponentially weighted mixture of multi-step advantage estimates.
- **Why**: Select a smooth bias–variance tradeoff instead of committing to one bootstrap horizon.
- **How**:
    1. Compute one-step residuals from frozen rollout values.
    2. Accumulate residuals backward with geometrically decaying weight.
    3. Cut recurrence at every segment boundary, while preserving nonterminal bootstraps.

```{note} Math
:class: dropdown
Process:

$$
\begin{align*}
\delta_t&=r_{t+1}+\gamma b_tV_\phi(s_{t+1})-V_\phi(s_t),\\
\widehat A_t^{\mathrm{GAE}}
&=\delta_t+\gamma\lambda(1-c_t)\widehat A_{t+1}^{\mathrm{GAE}},\\
\widehat R_t&=\widehat A_t^{\mathrm{GAE}}+V_\phi(s_t).
\end{align*}
$$
- $\lambda\in[0,1]$: Trace-decay parameter.

- Initialize the backward accumulator to zero after the last collected transition.
- $\widehat R_t$ is a $\lambda$-return target, not necessarily a Monte Carlo return.
```

```{tip} Derivation
:class: dropdown
1. Within a segment, discounted residuals telescope:

    $$
    \sum_{l=0}^{h-1}\gamma^l\delta_{t+l}
    =\widehat A_t^{(h)}.
    $$

2. Form a finite mixture that retains the remaining weight at the endpoint:

    $$
    \widehat A_t^{\mathrm{GAE}}
    =(1-\lambda)\sum_{h=1}^{H-1}\lambda^{h-1}\widehat A_t^{(h)}
    +\lambda^{H-1}\widehat A_t^{(H)}.
    $$
    - $H$: Remaining transitions until this segment's endpoint.

3. Collect each residual's coefficient:

    $$
    \widehat A_t^{\mathrm{GAE}}
    =\sum_{l=0}^{H-1}(\gamma\lambda)^l\delta_{t+l}.
    $$

4. Factor out the first residual → backward recurrence; $1-c_t$ prevents crossing the endpoint.
```

````{important} Code
:class: dropdown
```python
import torch

class GAE:
    def __init__(self, gamma, trace_decay):
        assert 0 <= trace_decay <= 1
        self.gamma, self.trace_decay = gamma, trace_decay

    @torch.no_grad()
    def __call__(self, reward, value, next_value, terminated, boundary):
        assert reward.ndim == 2
        assert all(x.shape == reward.shape for x in
                   (value, next_value, terminated, boundary))
        assert ((~terminated) | boundary).all() and boundary[-1].all()
        bootstrap = torch.where(terminated, 0.0, next_value)
        residual = reward + self.gamma * bootstrap - value
        advantage = torch.empty_like(residual)
        carry = torch.zeros_like(residual[0])
        for t in reversed(range(reward.shape[0])):
            carry = residual[t] + self.gamma * self.trace_decay * (
                ~boundary[t]
            ) * carry
            advantage[t] = carry
        return advantage, advantage + value

## Example
r = torch.tensor([[1.0], [2.0], [100.0]])
v = torch.tensor([[4.0], [5.0], [8.0]])
nv = torch.tensor([[5.0], [6.0], [0.0]])
terminal = torch.tensor([[False], [False], [True]])
boundary = torch.tensor([[False], [True], [True]])
adv, target = GAE(0.5, 1.0)(r, v, nv, terminal, boundary)
assert torch.allclose(adv[:, 0], torch.tensor([-0.5, 0.0, 92.0]))
assert torch.allclose(target[:, 0], torch.tensor([3.5, 5.0, 100.0]))
```
````

```{attention} Q&A
:class: dropdown
*What are the limiting cases?*

- $\lambda=0$: one-step residual.
- $\lambda=1$: return to the segment end, including its nonterminal bootstrap, minus the starting value.
- $\gamma=0$: only the immediate reward matters; trace decay has no effect.

*When is GAE unbiased?*

- Exact $V^\pi$, on-policy continuations & correct boundaries → each component has the correct conditional advantage expectation.
- Approximate values generally introduce bootstrap bias for $\lambda<1$.
- $\lambda=1$ through true termination removes bootstrap bias from the policy-gradient estimator, even with an imperfect fixed state baseline; the advantage estimate itself need not equal $A^\pi$ in expectation.
- $\lambda=1$ at a nonterminal cutoff still depends on the endpoint value estimate.
- The outer $\gamma^t$ issue remains separate: unbiased advantage estimation does not repair an incorrectly weighted actor objective.

*Why two masks?*

- Using $1-c_t$ in the residual incorrectly zeroes time-limit continuation values.
- Using $1-d_t$ in the recurrence can mix the next episode/segment's residuals into the current one.
- If a rollout merely pauses & later continues the same episode, its episode-time index does not reset; only the collected trace ends.
```

&nbsp;

## Conservative policy updates

### TRPO
- **Name**: Trust Region Policy Optimization {cite:p}`schulman2015trust`
- **What**: Policy-surrogate maximization under a distributional step constraint.
- **Why**: A large policy step invalidates advantages & state visits estimated under the previous policy.
- **How**:
    1. Collect old-policy data & estimate advantages.
    2. Find a locally improving direction measured in policy-distribution space.
    3. Backtrack until sampled surrogate improvement & a mean-divergence constraint are satisfied.

```{note} Math
:class: dropdown
Objective:

$$
\begin{align*}
\max_\theta\quad
L(\theta)&=
\mathbb E_{\substack{s\sim d_\gamma^{\pi_{\mathrm{old}}}\\
                    a\sim\pi_{\mathrm{old}}(\cdot\mid s)}}
\left[
\frac{\pi_\theta(a\mid s)}{\pi_{\mathrm{old}}(a\mid s)}
A^{\pi_{\mathrm{old}}}(s,a)
\right],\\
\text{subject to}\quad
\mathbb E_{s\sim d_\gamma^{\pi_{\mathrm{old}}}}
\left[D_{\mathrm{KL}}(\pi_{\mathrm{old}}(\cdot\mid s)
\Vert\pi_\theta(\cdot\mid s))\right]&\leq\varepsilon_{\mathrm{KL}}.
\end{align*}
$$
- $\pi_{\mathrm{old}}$: Policy $\pi_{\theta_{\mathrm{old}}}$.
- $\varepsilon_{\mathrm{KL}}>0$: Trust-region radius.
- $D_{\mathrm{KL}}$: Kullback–Leibler divergence.

Process:

$$
\begin{align*}
F&=\mathbb E[z z^\top],\qquad
z=\left.\nabla_\theta\log\pi_\theta(a\mid s)\right|_{\theta_{\mathrm{old}}},\\
g&=\left.\nabla_\theta L(\theta)\right|_{\theta_{\mathrm{old}}},\\
\Delta\theta&=
\sqrt{\frac{2\varepsilon_{\mathrm{KL}}}{g^\top F^{-1}g}}F^{-1}g.
\end{align*}
$$
- $F$: Fisher information matrix under old-policy sampling.
- $z$: Old-policy score vector.
- $g$: Surrogate gradient.
- $\Delta\theta$: Locally optimal linear-objective/quadratic-constraint step, assuming invertible $F$ & nonzero $g$.

- Large networks: Fisher-vector products & conjugate gradients; damping & backtracking handle approximation error.
```

```{tip} Derivation
:class: dropdown
1. The performance-difference identity is exact:

    $$
    J(\pi_\theta)-J(\pi_{\mathrm{old}})
    =\frac1{1-\gamma}
    \mathbb E_{\substack{s\sim d_\gamma^{\pi_\theta}\\a\sim\pi_\theta}}
    [A^{\pi_{\mathrm{old}}}(s,a)].
    $$

2. Replace new-policy occupancy by old-policy occupancy → a local surrogate, not the exact performance difference.
3. Importance-weight old-policy actions to express that surrogate using collected data.
4. Approximate the mean KL locally by $\tfrac12\Delta\theta^\top F\Delta\theta$.
5. Maximize $g^\top\Delta\theta$ subject to that quadratic constraint → scaled natural-gradient direction $F^{-1}g$.
```

````{important} Code
:class: dropdown
```python
import torch

class TrustRegionStep:
    def __init__(self, radius, damping=0.0):
        assert radius > 0 and damping >= 0
        self.radius, self.damping = radius, damping

    def __call__(self, gradient, fisher):
        ## Small explicit Fisher; large models need matrix-free solves.
        if torch.count_nonzero(gradient) == 0:
            return torch.zeros_like(gradient)
        identity = torch.eye(gradient.numel(), device=gradient.device,
                             dtype=gradient.dtype)
        metric = fisher + self.damping * identity
        direction = torch.linalg.solve(metric, gradient)
        curvature = gradient @ direction
        if curvature <= 0:
            raise ValueError("Fisher metric must give positive gradient curvature")
        return (2 * self.radius / curvature).sqrt() * direction

## Example
fisher = torch.diag(torch.tensor([1.0, 4.0]))
step = TrustRegionStep(0.01)(torch.tensor([1.0, 1.0]), fisher)
assert torch.allclose(0.5 * step @ fisher @ step, torch.tensor(0.01))
```
````

```{attention} Q&A
:class: dropdown
*Does practical TRPO guarantee monotonic improvement?*

- The theoretical bound controls worst-state policy divergence & uses exact quantities.
- Practical TRPO substitutes sampled advantages, average KL, local approximations & finite optimization.
- Those substitutions do not inherit an unconditional monotonic-return guarantee; a low average KL can hide a large change in a rare state.

*Is the code a complete TRPO optimizer?*

- No. It exposes the local natural-gradient step only.
- A complete update also estimates Fisher/advantages, tests the actual surrogate & KL, backtracks, and rejects a step if no candidate passes.
- Uniform rollout states replace discounted occupancy in common implementations; the earlier exact-versus-surrogate distinction still applies.

*Why move to PPO?*

- Avoid a second-order constrained solve & line search; retain a first-order mechanism discouraging excessive policy change.
- That simplification gives up an explicit trust-region constraint.
```

&nbsp;

### PPO
- **Name**: Proximal Policy Optimization {cite:p}`schulman2017proximalpolicyoptimizationalgorithms`
- **What**: First-order policy optimization with clipped or KL-penalized old-policy surrogates.
- **Why**: Reuse a fresh rollout for several updates without unconstrained policy-gradient steps.
- **How**:
    1. Freeze a behavior-policy snapshot; collect actions, old log-probabilities & value predictions.
    2. Prepare fixed advantages & value targets, usually with GAE.
    3. Optimize shuffled minibatches for a limited number of epochs.
    4. Discard the rollout & collect again with the updated policy.

```{note} Math
:class: dropdown
Objective:

$$
\begin{align*}
\varrho_t(\theta)
&=\exp\!\left[
\log\pi_\theta(a_t\mid s_t)
-\operatorname{sg}(\log\pi_{\mathrm{old}}(a_t\mid s_t))
\right],\\
L^{\mathrm{clip}}(\theta)
&=\widehat{\mathbb E}\left[
\omega_t\min\left(
\varrho_t\operatorname{sg}(\widehat A_t),
\operatorname{clip}(\varrho_t,1-\epsilon,1+\epsilon)
\operatorname{sg}(\widehat A_t)
\right)\right].
\end{align*}
$$
- $\varrho_t$: Current/old action-probability ratio; not a reward.
- $\epsilon\in(0,1)$: Clipping width.

$$
\mathcal L=
-L^{\mathrm{clip}}
+\frac{c_V}{2}\widehat{\mathbb E}
[(V_\phi(s_t)-\operatorname{sg}(\widehat R_t))^2]
-\beta\widehat{\mathbb E}
[\omega_t\mathcal H(\pi_\theta(\cdot\mid s_t))].
$$
- $c_V\geq0$: Value-loss coefficient.

- Minimize $\mathcal L$; the value term is not part of the policy-gradient theorem.
- $\omega_t=1$ gives the common rollout objective; episode-weighted aggregation is needed for an exact discounted-gradient interpretation at the old policy.
```

```{tip} Derivation
:class: dropdown
1. Without clipping, $\mathbb E_{\pi_{\mathrm{old}}}[\varrho_t\widehat A_t]$ importance-weights actions at fixed old-policy states.
2. For $\widehat A_t>0$, cap the reward for increasing probability:

    $$
    \min(\varrho_t,1+\epsilon)\widehat A_t.
    $$

3. For $\widehat A_t<0$, cap the reward for decreasing probability:

    $$
    \max(\varrho_t,1-\epsilon)\widehat A_t.
    $$

4. The minimum chooses the pessimistic branch. Harmful changes retain a gradient; sufficiently favorable changes stop earning additional surrogate improvement.
5. The clipped objective lower-bounds the unclipped sample surrogate, not the true return.
```

```{note} Example
:class: dropdown
- Toy clipping width $\epsilon=0.2$:
    - $\widehat A=1,\ \varrho=1.5$ → clipped contribution $1.2$.
    - $\widehat A=-1,\ \varrho=0.5$ → clipped contribution $-0.8$.
    - $\widehat A=-1,\ \varrho=1.5$ → contribution $-1.5$; the harmful probability increase remains penalized.
```

````{important} Code
:class: dropdown
```python
import torch

class PPOClipLoss:
    def __init__(self, epsilon, value_weight, entropy_weight):
        assert 0 < epsilon < 1
        self.epsilon = epsilon
        self.value_weight, self.entropy_weight = value_weight, entropy_weight

    def __call__(self, logp, old_logp, advantage, value, target,
                 entropy, weight):
        assert logp.ndim == 1
        assert all(x.shape == logp.shape for x in
                   (old_logp, advantage, value, target, entropy, weight))
        ratio = (logp - old_logp.detach()).exp()
        advantage, weight = advantage.detach(), weight.detach()
        raw = ratio * advantage
        clipped = ratio.clamp(1 - self.epsilon, 1 + self.epsilon) * advantage
        actor = -(weight * torch.minimum(raw, clipped)).mean()
        critic = 0.5 * (value - target.detach()).square().mean()
        bonus = (weight * entropy).mean()
        return actor + self.value_weight * critic - self.entropy_weight * bonus

## Example
logp = torch.tensor([0.6, 0.2]).log().requires_grad_()
old_logp = torch.tensor([0.5, 0.4]).log().requires_grad_()
loss = PPOClipLoss(0.2, 1.0, 0.0)(
    logp, old_logp, torch.tensor([1.0, -1.0]), torch.zeros(2),
    torch.tensor([1.0, -1.0]), torch.zeros(2), torch.ones(2)
)
loss.backward()
assert old_logp.grad is None and logp.grad is not None
```
````

```{attention} Q&A
:class: dropdown
*What must remain frozen during the rollout's optimization epochs?*

- Stored actions, old log-probabilities, advantages & return targets.
- Old log-probabilities must come from the actual collection policy, including its preprocessing, action masks & action transform.
- Do not overwrite the denominator with the current policy each minibatch; that erases the reference being constrained.
- Shared actor/critic params may move, but value targets & actor advantages remain detached.

*Does clipping guarantee $\varrho_t\in[1-\epsilon,1+\epsilon]$?*

- No. It clips a branch of the objective, not params or probabilities.
- Other samples, shared features, entropy & value gradients can still move an already-clipped action.
- No hard KL bound, monotonic-improvement guarantee or global-convergence guarantee follows.

*Why track KL if ratios are clipped?*

- Detect changes that clipping did not constrain; stop epochs early when a chosen KL budget is exceeded.
- On old-policy samples, $\widehat{\mathbb E}[(\varrho-1)-\log\varrho]$ estimates old-to-new KL when supports match; sampling error remains.
- Clipped fraction describes ratio excursions, not safety or task performance.

*Can PPO train indefinitely on an old replay buffer?*

- No general justification. Action ratios do not correct the changed state distribution or stale advantage estimates.
- A full trajectory ratio can correct trajectory distributions under coverage assumptions, but its variance can grow severely with horizon; clipping adds bias.
- Limited same-rollout reuse is not equivalent to the replay-based Bellman learning used by DDPG/TD3/SAC.

*What implementation mistakes silently change the algorithm?*

- Continuous actions: sum coordinate log-densities before taking the ratio; keep actions fixed.
- Flatten values, advantages & log-probabilities consistently to one scalar per sample; $(m,1)$ versus $(m,)$ can broadcast into an $(m,m)$ loss.
- Use final observations & the two distinct boundary masks when computing GAE.
- Advantage normalization is a practical rescaling, not required by the theorem; compute critic targets before normalizing advantages.

*Is every PPO implementation the clipped variant?*

- No. The original family also includes a KL-penalized surrogate with an adaptively adjusted penalty.
- Value clipping, advantage normalization, gradient clipping & KL early stopping are additional choices; none makes the policy-ratio clipping a hard constraint.
```

&nbsp;

## Deterministic continuous control

### Deterministic policy gradients
- **What**: Return gradients through a deterministic actor & a differentiable action-value function. {cite:p}`silver2014deterministic`
- **Why**: Improve continuous actions without an action-space maximization or score-function sampling noise.
- **How**:
    1. Critic estimates how return changes with the action.
    2. Differentiate the actor's action with respect to its params.
    3. Chain those derivatives to move actions toward higher predicted return.

```{note} Math
:class: dropdown
Model:

$$
a=f_\theta(s).
$$
- $f_\theta$: Deterministic actor; $\mu$ remains reserved for behavior policy.

Training:

$$
\nabla_\theta J
=\frac1{1-\gamma}
\mathbb E_{s\sim d_\gamma^{f_\theta}}
\left[
(D_\theta f_\theta(s))^\top
\left.\nabla_aQ^{f_\theta}(s,a)\right|_{a=f_\theta(s)}
\right].
$$
- $D_\theta f_\theta(s)$: Actor Jacobian; shape action-dimension × parameter-dimension.

- Requires differentiability of $f_\theta$ & $Q^{f_\theta}$ in the relevant arguments, integrability & regularity allowing the gradient/expectation interchange.
- Finite-episode equivalent: expected sum over $\gamma^t$-weighted state visits.
```

```{tip} Derivation
:class: dropdown
1. Write $V^{f_\theta}(s)=Q^{f_\theta}(s,f_\theta(s))$.
2. Differentiate: direct action change gives $(D_\theta f_\theta)^\top\nabla_aQ$; changes to future actions give a recursive future-value term.
3. Unroll that recursion through transitions → discounted occupancy, exactly as in the stochastic theorem.
4. No environment derivative is needed if a learned differentiable critic supplies $\nabla_aQ$.
```

```{attention} Q&A
:class: dropdown
*Is the replay-buffer actor gradient the exact gradient above?*

- Generally no. It replaces target-policy discounted occupancy by replay states & true values by an approximate critic.
- The replay objective holds critic params fixed while changing current actions; it is a policy-improvement surrogate.
- Even a behavior-weighted objective $\mathbb E_{s\sim\nu}[V^{f_\theta}(s)]$ has future-policy derivative terms. Simply replacing occupancy by $\nu$ in the theorem does not produce its exact derivative.

*Why no action importance-sampling ratio?*

- The actor evaluates its own action at each replayed state; it does not average score terms over behavior actions.
- Critic targets similarly evaluate a new target action after a stored transition.
- State/action coverage & critic accuracy still matter; eliminating an action integral does not fix distribution mismatch.

*Why not use deterministic actions in REINFORCE?*

- A point mass has no ordinary continuous density with a usable log-score.
- The deterministic theorem uses a pathwise action derivative instead; stochastic policy gradients approach it only under additional limiting regularity assumptions.

*How does a deterministic policy explore?*

- Use a distinct behavior policy: actor action plus exploration noise, or another exploratory action rule.
- Environment randomness alone need not provide adequate action coverage.
```

&nbsp;

### DDPG
- **Name**: Deep Deterministic Policy Gradient {cite:p}`lillicrap2016continuous`
- **What**: Off-policy deep deterministic actor–critic with replay & target networks.
- **Why**: Extend replay-based value learning to continuous actions without enumerating an argmax.
- **How**:
    1. Interact using noisy actor actions; store transitions in replay.
    2. Regress the critic toward a target critic evaluated at a target actor's next action.
    3. Improve the actor through the online critic.
    4. Slowly move both target networks toward their online networks.

```{note} Math
:class: dropdown
Process:

$$
\begin{align*}
a'&=f_{\bar\theta}(s_{t+1}),\\
y_t&=r_{t+1}+\gamma b_tQ_{\bar\phi}(s_{t+1},a').
\end{align*}
$$
- $\bar\theta$: Target-actor params.
- $a'$: Target action at the next state.
- $y_t$: Detached Bellman target.

Objective:

$$
\begin{align*}
\mathcal L_Q(\phi)
&=\frac12\mathbb E_{\mathcal R}
[(Q_\phi(s_t,a_t)-\operatorname{sg}(y_t))^2],\\
\mathcal L_\pi(\theta)
&=-\mathbb E_{s\sim\mathcal R}[Q_\phi(s,f_\theta(s))].
\end{align*}
$$

- Actor loss: freeze $\phi$, retain $\nabla_aQ_\phi$.

Process:

$$
\begin{align*}
\bar\theta&\leftarrow(1-\tau)\bar\theta+\tau\theta,\\
\bar\phi&\leftarrow(1-\tau)\bar\phi+\tau\phi.
\end{align*}
$$
- $\tau\in(0,1]$: Online-param interpolation weight; smaller → slower target motion.
```

````{important} Code
:class: dropdown
```python
from copy import deepcopy
import torch
import torch.nn as nn

class DDPGKernel:
    def __init__(self, gamma, tau):
        assert 0 <= gamma < 1 and 0 < tau <= 1
        self.gamma, self.tau = gamma, tau

    @staticmethod
    def q(network, state, action):
        return network(torch.cat((state, action), dim=-1)).squeeze(-1)

    def __call__(self, actor, critic, target_actor, target_critic,
                 state, action, reward, next_state, terminated):
        assert reward.ndim == 1 and terminated.shape == reward.shape
        with torch.no_grad():
            next_q = self.q(target_critic, next_state, target_actor(next_state))
            target = reward + self.gamma * torch.where(terminated, 0.0, next_q)
        prediction = self.q(critic, state, action)
        assert prediction.shape == target.shape
        critic_loss = 0.5 * (prediction - target).square().mean()
        ## Freeze critic params, not its derivative with respect to the action.
        critic.requires_grad_(False)
        actor_loss = -self.q(critic, state, actor(state)).mean()
        critic.requires_grad_(True)
        return actor_loss, critic_loss

    @torch.no_grad()
    def update_target(self, target, online):
        for target_param, param in zip(target.parameters(), online.parameters()):
            target_param.lerp_(param, self.tau)

## Example
actor = nn.Sequential(nn.Linear(3, 1), nn.Tanh())
critic = nn.Linear(4, 1)
target_actor, target_critic = deepcopy(actor), deepcopy(critic)
kernel = DDPGKernel(0.9, 0.1)
losses = kernel(actor, critic, target_actor, target_critic,
                torch.zeros(2, 3), torch.zeros(2, 1), torch.ones(2),
                torch.zeros(2, 3), torch.tensor([False, True]))
sum(losses).backward()
assert actor[0].weight.grad is not None and critic.weight.grad is not None
```
````

```{attention} Q&A
:class: dropdown
*Which action trains which network?*

- Critic regression: actual replayed action $a_t$.
- Actor gradient: current action $f_\theta(s_t)$, not the replayed action.
- Bootstrap: target actor's action at the next state.
- Exploration noise affects data collection, not the basic DDPG actor objective.

*Why does off-policy one-step critic learning not need a behavior-action ratio?*

- A stored $(s,a)$ transition samples the same conditional environment dynamics regardless of how that action was selected.
- The next target action is evaluated explicitly rather than taken from the behavior trajectory.
- Replay determines which state–action errors are fitted; insufficient coverage & function approximation still cause bias/instability.

*What does the kernel omit?*

- Data collection, replay storage, optimizers & update scheduling; these are update kernels, not production agents.
- Use separate actor/critic networks without shared params; feed detached replay tensors.
- For a sequential update: optimize critic, recompute actor loss with the updated critic, optimize actor, then update targets.
- Target interpolation covers params; architectures with running-statistic buffers need an explicit buffer policy. Examples use buffer-free critics.

*Does freezing the critic mean using `no_grad` around its actor evaluation?*

- No. That would erase the action derivative & stop actor learning.
- Freeze critic params while retaining the graph from actor action → critic output.
- Online actor & critic are assumed trainable on entry; the kernel restores critic trainability afterward.

*Main failure mode?*

- Actor exploits critic overestimates, producing poor actions that reinforce erroneous targets.
- Replay + bootstrapping + nonlinear approximation has no general convergence guarantee.
- Target networks slow moving targets; they do not make the Bellman/actor iteration a proven contraction in neural parameter space.
```

&nbsp;

#### TD3
- **Name**: Twin Delayed Deep Deterministic Policy Gradient {cite:p}`fujimoto2018addressing`
- **What**: DDPG with twin critics, delayed policy updates & smoothed target actions.
- **Why**: Approximation errors in the critic become exploitable peaks for the actor.
- **How**:
    1. Use the lower of two target-critic estimates in the backup.
    2. Perturb target actions locally to discourage narrow value spikes.
    3. Update critics more often than the actor & target networks.

```{note} Math
:class: dropdown
Process:

$$
\begin{align*}
\widetilde\epsilon&=\operatorname{clip}
(\epsilon,-c_{\mathrm{noise}},c_{\mathrm{noise}}),
\qquad \epsilon\sim\mathcal N(0,\sigma_{\mathrm{noise}}^2I),\\
\widetilde a'
&=\operatorname{clip}\left(
f_{\bar\theta}(s_{t+1})+\widetilde\epsilon,
a_{\mathrm{low}},a_{\mathrm{high}}\right),\\
y_t&=r_{t+1}+\gamma b_t
\min_{j\in\{1,2\}}Q_{\bar\phi_j}(s_{t+1},\widetilde a').
\end{align*}
$$
- $c_{\mathrm{noise}}>0$: Target-noise clipping magnitude.
- $\sigma_{\mathrm{noise}}>0$: Target-noise standard deviation.
- $\widetilde a'$: Smoothed, bounded next action.
- $a_{\mathrm{low}}$: Coordinatewise lower action bound.
- $a_{\mathrm{high}}$: Coordinatewise upper action bound.
- $j$: Critic index in this section.

Objective:

$$
\begin{align*}
\mathcal L_Q&=\frac12\sum_{j=1}^{2}
\mathbb E_{\mathcal R}[(Q_{\phi_j}(s_t,a_t)-\operatorname{sg}(y_t))^2],\\
\mathcal L_\pi&=-\mathbb E_{s\sim\mathcal R}[Q_{\phi_1}(s,f_\theta(s))].
\end{align*}
$$

- Original TD3 actor uses the first online critic, not the minimum.
- Update actor & all targets once per chosen number of critic updates.
- $c_{\mathrm{noise}}$ is a magnitude, not the segment-boundary indicator $c_t$.
```

````{important} Code
:class: dropdown
```python
from copy import deepcopy
import torch
import torch.nn as nn
## Reuses DDPGKernel from rl/policy.md

class TD3Kernel(DDPGKernel):
    def __init__(self, gamma, tau, noise_std, noise_clip,
                 low, high, policy_delay):
        super().__init__(gamma, tau)
        assert policy_delay >= 1 and noise_std >= 0 and noise_clip >= 0
        self.noise_std, self.noise_clip = noise_std, noise_clip
        self.low, self.high, self.policy_delay = low, high, policy_delay

    def __call__(self, update_index, actor, critics, target_actor,
                 target_critics, state, action, reward, next_state, terminated):
        assert len(critics) == len(target_critics) == 2
        assert reward.ndim == 1 and terminated.shape == reward.shape
        with torch.no_grad():
            next_action = target_actor(next_state)
            noise = (self.noise_std * torch.randn_like(next_action)).clamp(
                -self.noise_clip, self.noise_clip
            )
            low = torch.as_tensor(self.low, device=next_action.device)
            high = torch.as_tensor(self.high, device=next_action.device)
            next_action = (next_action + noise).clamp(low, high)
            next_q = torch.minimum(
                self.q(target_critics[0], next_state, next_action),
                self.q(target_critics[1], next_state, next_action)
            )
            target = reward + self.gamma * torch.where(terminated, 0.0, next_q)
        predictions = [self.q(q, state, action) for q in critics]
        assert all(q.shape == target.shape for q in predictions)
        critic_loss = 0.5 * sum((q - target).square().mean() for q in predictions)
        actor_loss = None
        if update_index % self.policy_delay == 0:
            critics[0].requires_grad_(False)
            actor_loss = -self.q(critics[0], state, actor(state)).mean()
            critics[0].requires_grad_(True)
        return actor_loss, critic_loss

## Example
actor = nn.Sequential(nn.Linear(3, 1), nn.Tanh())
critics = [nn.Linear(4, 1), nn.Linear(4, 1)]
kernel = TD3Kernel(0.9, 0.1, 0.1, 0.2, [-1.0], [1.0], 2)
actor_loss, critic_loss = kernel(
    1, actor, critics, deepcopy(actor), deepcopy(critics),
    torch.zeros(2, 3), torch.zeros(2, 1), torch.ones(2),
    torch.zeros(2, 3), torch.tensor([False, True])
)
assert actor_loss is None and critic_loss.ndim == 0
```
````

```{attention} Q&A
:class: dropdown
*What does each modification address?*

- Twin minimum: reduce overestimation in bootstrapped targets.
- Delayed actor/targets: give critics more fitting steps before the policy exploits their current estimates.
- Target smoothing: train on a neighborhood of next actions rather than a single potentially spurious peak.
- None removes all function-approximation error.

*Is target noise the exploration noise?*

- No. Exploration noise changes actions executed in the environment.
- Target noise changes critic backups during training.
- Both can exist, but serve different purposes.

*Does taking a minimum make the target unbiased?*

- No. It can introduce underestimation; correlated critic errors can leave substantial overestimation.
- This is a bias-control heuristic, not an exact correction or uncertainty bound.

*How should the kernel's delayed result be used?*

- Always train both critics.
- When `actor_loss` is not `None`, train the actor & interpolate target actor plus both target critics.
- Do not interpolate targets every critic step if reproducing the original delayed-update algorithm.
- Recompute actor loss after a critic optimizer step rather than mutating params referenced by an outstanding loss graph.
```

&nbsp;

## Entropy-regularized control

### Maximum-entropy RL
- **What**: Return maximization with a reward for conditional action entropy.
- **Why**: Prefer diverse high-value behavior instead of immediately collapsing to one action.
- **How**:
    1. Add entropy at every decision state, including future states.
    2. Evaluate policies with entropy-aware Bellman backups.
    3. Improve toward actions favored jointly by value & entropy.

```{note} Math
:class: dropdown
Objective:

$$
J_\beta(\pi)
=\mathbb E_{\rho_0,\pi}\left[
\sum_{t=0}^{T-1}\gamma^t
\left(r_{t+1}-\beta\log\pi(a_t\mid s_t)\right)
\right].
$$
- $J_\beta$: Entropy-regularized discounted return.

$$
\mathcal H(\pi(\cdot\mid s))
=-\mathbb E_{a\sim\pi(\cdot\mid s)}[\log\pi(a\mid s)].
$$

Model:

$$
\begin{align*}
Q_\beta^\pi(s,a)
&=r(s,a)+\gamma\mathbb E_{s'\sim p(\cdot\mid s,a)}
[V_\beta^\pi(s')],\\
V_\beta^\pi(s)
&=\mathbb E_{a\sim\pi}
[Q_\beta^\pi(s,a)-\beta\log\pi(a\mid s)].
\end{align*}
$$
- $Q_\beta^\pi$: Soft action value; excludes the current action's entropy term, includes future entropy.
- $V_\beta^\pi$: Soft state value; includes entropy of the current decision.

- Terminal continuation value is zero; sampled backups apply $b_t$.

Inference:

$$
\begin{align*}
\pi^*(a\mid s)&=
\exp\!\left(\frac{Q_\beta^*(s,a)-V_\beta^*(s)}{\beta}\right),\\
V_\beta^*(s)&=\beta\log\int
\exp(Q_\beta^*(s,a)/\beta)\,da.
\end{align*}
$$

- $\beta>0$; use a sum for finite actions. Continuous normalizer must be finite.
- The density is relative to the chosen action-space reference measure.
```

```{tip} Derivation
:class: dropdown
1. For a fixed state & fixed action-value function, optimize a normalized action distribution:

    $$
    \max_\pi\int\pi(a)\left[Q(a)-\beta\log\pi(a)\right]\,da,
    \qquad \int\pi(a)\,da=1.
    $$

2. Introduce a multiplier for normalization; setting the density derivative to zero yields $\log\pi(a)=Q(a)/\beta+\text{constant}$.
3. Normalize → exponential-of-value policy.
4. Substitute back → log-sum-exp value.
5. For finite actions, $\beta\to0^+$ recovers the greedy maximum; ties may retain a mixture.
```

````{important} Code
:class: dropdown
```python
import torch

class SoftPolicyImprovement:
    def __init__(self, temperature):
        assert temperature > 0
        self.temperature = temperature

    def __call__(self, action_values):
        logits = action_values / self.temperature
        log_normalizer = logits.logsumexp(dim=-1, keepdim=True)
        probabilities = (logits - log_normalizer).exp()
        soft_value = self.temperature * log_normalizer.squeeze(-1)
        return probabilities, soft_value

## Example
q = torch.tensor([[0.0, 1.0]])
probability, value = SoftPolicyImprovement(0.5)(q)
assert torch.allclose(probability.sum(dim=-1), torch.ones(1))
assert probability[0, 1] > probability[0, 0] and value.shape == (1,)
```
````

```{attention} Q&A
:class: dropdown
*Why is an actor entropy bonus alone insufficient?*

- It rewards randomness at sampled current states, but a reward-only critic does not account for the entropy obtainable later.
- Soft values propagate future entropy through the Bellman equation.
- Entropy regularization changes the task objective; it is not an unbiased estimator of reward-only optimal control.

*Does maximum entropy solve exploration?*

- No. It encourages action diversity, not necessarily discovery of distant rewarding states.
- Sparse/deceptive rewards & poor state coverage can still defeat local policy optimization.

*Why can continuous entropy be negative?*

- A density can exceed one; differential entropy is not a discrete uncertainty count.
- It depends on coordinates & units. On an unbounded action space it can also grow without bound unless rewards or policy constraints counteract expansion.

*What is guaranteed in the exact setting?*

- Finite discounted MDP, bounded rewards, fixed finite $\beta$: the soft optimality operator is a $\gamma$-contraction in sup norm.
- Exact soft value/policy iteration converges under its assumptions.
- Shared neural params, approximate critics & incomplete state coverage do not inherit those exact tabular guarantees.
```

&nbsp;

### SAC
- **Name**: Soft Actor–Critic {cite:p}`haarnoja2018soft`
- **What**: Off-policy stochastic actor–critic approximating soft policy iteration.
- **Why**: Combine replay-based learning with a tractable entropy-regularized continuous policy.
- **How**:
    1. Fit soft action values from replay.
    2. Move the actor toward the exponential-of-value distribution.
    3. Alternate evaluation & improvement instead of solving either exactly.

```{note} Math
:class: dropdown
Objective:

$$
\pi_{\mathrm{new}}(\cdot\mid s)
=\arg\min_{\pi\in\Pi}
D_{\mathrm{KL}}\left(
\pi(\cdot\mid s)\,\middle\|\,
\frac{\exp(Q_\beta^{\pi_{\mathrm{old}}}(s,\cdot)/\beta)}
{Z(s)}
\right).
$$
- $\Pi$: Allowed policy-distribution family.
- $Z(s)$: Normalizing integral of $\exp(Q_\beta^{\pi_{\mathrm{old}}}(s,a)/\beta)$ over actions.

- Equivalent statewise minimization:

    $$
    \mathbb E_{a\sim\pi}
    [\beta\log\pi(a\mid s)-Q_\beta^{\pi_{\mathrm{old}}}(s,a)].
    $$
```

```{tip} Derivation
:class: dropdown
1. Expand the reverse KL:

    $$
    D_{\mathrm{KL}}(\pi\Vert\exp(Q/\beta)/Z)
    =\mathbb E_\pi[\log\pi-Q/\beta]+\log Z.
    $$

2. Multiply by $\beta$; drop $\beta\log Z$, which is constant with respect to the new actor.
3. Differentiate a sampled action reparameterization to optimize the remaining expectation.
4. Approximate values & replayed states give the practical actor loss, not an exact start-state policy-gradient estimator.
```

```{attention} Q&A
:class: dropdown
*Why can a stochastic actor learn off-policy without action importance weights?*

- Replay supplies states & transitions; the actor samples fresh actions from its current distribution at those states.
- Critic targets sample the current policy at next states rather than reuse behavior continuation actions.
- No behavior-action correction is needed for these conditional objectives; replay-state mismatch & extrapolation error still remain.

*What does the soft policy-iteration guarantee require?*

- Exact statewise policy improvement & policy evaluation, with boundedness & coverage assumptions.
- A finite-action tabular result does not prove convergence of continuous neural SAC with shared params, sampled gradients & twin-critic heuristics.
- A restricted unimodal actor cannot represent every desirable multimodal action distribution.

*Which SAC version should be implemented?*

- The original formulation explicitly learned a separate state-value network.
- The twin-Q variant below computes continuation values from target critics & the current actor instead; no separately trained $V$ network.
```

&nbsp;

#### Twin-Q SAC & automatic temperature
- **What**: SAC with twin target critics, reparameterized actor updates & learned entropy temperature. {cite:p}`haarnoja2018softapplications`
- **Why**: Remove a separately fitted value network & adapt the reward–entropy balance.
- **How**:
    1. Sample replay transitions; draw next actions from the current actor.
    2. Fit both critics to a shared entropy-adjusted target.
    3. Improve the actor through fresh reparameterized actions & the lower online critic.
    4. Adjust temperature toward a chosen average entropy floor; slowly update target critics.

```{note} Math
:class: dropdown
Process:

$$
\begin{align*}
a'&\sim\pi_\theta(\cdot\mid s_{t+1}),\\
y_t&=r_{t+1}+\gamma b_t
\left[
\min_{j\in\{1,2\}}Q_{\bar\phi_j}(s_{t+1},a')
-\beta\log\pi_\theta(a'\mid s_{t+1})
\right].
\end{align*}
$$

- $Q_{\phi_j}$ approximates a soft action value; the $\beta$ subscript is suppressed on learned critics.
- $j$ indexes critics; $a'$ & $y_t$ denote next action & Bellman target.
- Target actions come from the current actor; no target actor or separate $V$ network.

Objective:

$$
\begin{align*}
\mathcal L_Q&=\frac12\sum_{j=1}^{2}
\mathbb E_{\mathcal R}
[(Q_{\phi_j}(s_t,a_t)-\operatorname{sg}(y_t))^2],\\
\mathcal L_\pi&=
\mathbb E_{\substack{s\sim\mathcal R\\\epsilon\sim\mathcal N(0,I)}}
\left[
\operatorname{sg}(\beta)\log\pi_\theta(f_\theta(s,\epsilon)\mid s)
-\min_jQ_{\phi_j}(s,f_\theta(s,\epsilon))
\right].
\end{align*}
$$
- $f_\theta(s,\epsilon)$: Reparameterized stochastic actor; overrides the deterministic section's noise-free $f_\theta(s)$.

- Actor derivative: hold critic params fixed; retain both the sampled-action path & explicit log-density dependence on $\theta$.

$$
\begin{align*}
\mathbb E_{s\sim\mathcal R,a\sim\pi_\theta}[-\log\pi_\theta(a\mid s)]
&\geq \mathcal H_*,\\
\beta&=\exp\eta,\\
\mathcal L_\eta
&=-\mathbb E_{s\sim\mathcal R,a\sim\pi_\theta}
\left[
e^\eta\operatorname{sg}\!\left(\log\pi_\theta(a\mid s)+\mathcal H_*\right)
\right].
\end{align*}
$$
- $\mathcal H_*$: Desired minimum average entropy, possibly negative for continuous actions.
- $\eta$: Log-temperature; unconstrained trainable scalar.

- Paper's continuous-control heuristic: $\mathcal H_*=-D$ for normalized $(-1,1)^D$ actions; not a universal optimum.
- $D$: Action dimension.

Process:

$$
\bar\phi_j\leftarrow(1-\tau)\bar\phi_j+\tau\phi_j.
$$
- $\tau\in(0,1]$: Online-param interpolation weight.

- Detach the complete critic target.
- Freeze temperature during the actor update & actor samples during the temperature update.
```

```{tip} Derivation
:class: dropdown
1. At fixed sampled-state weighting, impose an entropy floor while maximizing reward.
2. Add a nonnegative multiplier times entropy minus its target. For a fixed actor, minimize this dual objective over temperature:

    $$
    \mathcal L_\beta
    =\beta\left(
    \mathbb E[-\log\pi_\theta(a\mid s)]-\mathcal H_*
    \right).
    $$

3. Parameterize positive temperature by $\beta=e^\eta$:

    $$
    \frac{\partial\mathcal L_\eta}{\partial\eta}
    =\beta\left(\mathbb E[-\log\pi_\theta]-\mathcal H_*\right).
    $$

4. Entropy below target → negative derivative → gradient descent increases $\eta$ & $\beta$.
5. Entropy above target → positive derivative → decrease temperature; the constraint is average, not a per-state equality.
```

````{important} Code
:class: dropdown
```python
from copy import deepcopy
import torch
import torch.nn as nn
## Reuses DDPGKernel & SquashedGaussian from rl/policy.md

class SACKernel(DDPGKernel):
    def __init__(self, gamma, tau, target_entropy):
        super().__init__(gamma, tau)
        self.target_entropy = target_entropy

    def __call__(self, actor, critics, target_critics, log_temperature,
                 state, action, reward, next_state, terminated):
        assert len(critics) == len(target_critics) == 2
        assert reward.ndim == 1 and terminated.shape == reward.shape
        temperature = log_temperature.exp()
        with torch.no_grad():
            next_action, next_logp, _ = actor.sample(
                next_state, reparameterize=True
            )
            next_q = torch.minimum(
                self.q(target_critics[0], next_state, next_action),
                self.q(target_critics[1], next_state, next_action)
            )
            soft_next = next_q - temperature * next_logp
            target = reward + self.gamma * torch.where(terminated, 0.0, soft_next)
        predictions = [self.q(q, state, action) for q in critics]
        assert all(q.shape == target.shape for q in predictions)
        critic_loss = 0.5 * sum((q - target).square().mean() for q in predictions)
        sampled_action, logp, _ = actor.sample(state, reparameterize=True)
        for critic in critics:
            critic.requires_grad_(False)
        q_min = torch.minimum(
            self.q(critics[0], state, sampled_action),
            self.q(critics[1], state, sampled_action)
        )
        actor_loss = (temperature.detach() * logp - q_min).mean()
        for critic in critics:
            critic.requires_grad_(True)
        ## Exact log-parameterized dual loss: do not backpropagate into the actor.
        temperature_loss = -(
            temperature * (logp.detach() + self.target_entropy)
        ).mean()
        return actor_loss, critic_loss, temperature_loss

## Example
actor = SquashedGaussian(3, 1, [-1.0], [1.0])
critics = [nn.Linear(4, 1), nn.Linear(4, 1)]
log_temperature = torch.tensor(0.0, requires_grad=True)
losses = SACKernel(0.9, 0.1, -0.5)(
    actor, critics, deepcopy(critics), log_temperature,
    torch.zeros(2, 3), torch.zeros(2, 1), torch.ones(2),
    torch.zeros(2, 3), torch.tensor([False, True])
)
sum(losses).backward()
assert log_temperature.grad is not None
assert actor.mean.weight.grad is not None
```
````

```{dropdown} Table: Policy optimization methods
| Method | Actor | Training data | Critic/advantage | Main update control | Main cost or failure |
|:--|:--|:--|:--|:--|:--|
| REINFORCE | Stochastic | Complete on-policy episodes | Monte Carlo return; optional baseline | Step size | High return variance |
| A2C/A3C | Stochastic | Fresh parallel rollouts | Bootstrapped state values | Step size; optional entropy | Critic bias; A3C staleness |
| TRPO | Stochastic | Fresh old-policy rollouts | Usually state value + GAE | Approximate KL constraint | Second-order solve; sampling approximations |
| PPO | Stochastic | Fresh rollouts, limited epoch reuse | Usually state value + GAE | Clipped/penalized surrogate | No hard trust region; stale-data bias |
| DDPG | Deterministic | Replay | One action-value critic | Target actor & critic | Critic exploitation; weak action coverage |
| TD3 | Deterministic | Replay | Twin action-value critics | Pessimistic targets, delay, smoothing | Underestimation; replay extrapolation |
| SAC | Stochastic | Replay | Twin soft action-value critics | Entropy objective & target critics | Entropy/reward scaling; replay extrapolation |

- Categorical stochastic actors support discrete actions; Gaussian actors support continuous actions.
- DDPG/TD3 require differentiable continuous actions; the SAC implementation above is continuous-action SAC.
```

```{attention} Q&A
:class: dropdown
*What is the target-entropy sign convention?*

- Store $\mathcal H_*$ as an entropy, not as a desired log-probability.
- Loss uses $\log\pi+\mathcal H_*$, equivalently the negative of estimated entropy minus target.
- Example: current entropy $-2$, target $-1$ → entropy is too low → increase temperature.
- Negative continuous target entropy is not an error; compare quantities in the same action coordinates.

*Why do implementations sometimes optimize log-temperature linearly?*

- A common surrogate replaces $e^\eta$ in $\mathcal L_\eta$ by $\eta$.
- Its gradient has the same sign & interior zero but lacks the positive factor $\beta$; it changes optimization scaling, not the target sign.
- The code uses the exact exponential-parameterized dual loss; do not describe the linear surrogate as algebraically identical.

*What if the entropy target is impossible?*

- On a bounded action domain, entropy has an upper bound; the policy family may impose a lower achievable maximum.
- An infeasible floor can keep driving temperature upward; no dual equilibrium need exist.
- Automatic tuning is not a guarantee of exploration quality or reward-scale invariance.

*Which params receive which gradients?*

- Critic loss → online critics only; complete target is detached.
- Actor loss → actor only; online critics retain action derivatives but no parameter gradients.
- Temperature loss → log-temperature only.
- Target critics → interpolation only; no optimizer steps.
- For sequential optimizer steps, recompute affected losses rather than reuse graphs across in-place parameter updates.

*Why no separate $V$ or target actor?*

- Sampled next action & log-density estimate the soft state value using target critics directly.
- Current actor generates that action inside a detached target computation.
- Both target critics begin as exact online copies & are slowly interpolated after updates.

*Is SAC's deterministic evaluation action its mean action?*

- Common choice: squash & rescale the latent mean.
- That is generally not the expectation of the squashed distribution, nor necessarily its mode.
- Removing sampling changes the evaluated policy; the training entropy objective does not guarantee that this deterministic choice maximizes reward.

*How should PPO, TD3 & SAC be chosen?*

- Fresh parallel simulation, discrete actions or a simple stochastic-policy pipeline → PPO is a natural candidate.
- Continuous control with expensive environment interaction & useful replay coverage → TD3/SAC can reuse data.
- TD3 separates deterministic control from injected exploration noise; SAC learns a stochastic entropy-regularized policy.
- No universal ranking: interaction cost, compute, reward structure, action support & data coverage determine the tradeoff.

*Does off-policy mean offline-safe?*

- No. Online replay algorithms continue collecting coverage near their changing policies.
- Fixed offline data may not cover actor-proposed actions; bootstrapped critics can extrapolate badly & actors exploit those errors.
- Twin minima are not sufficient offline support constraints. Uncorrected replayed multi-step returns & GAE add behavior-continuation mismatch.
- Actor states sampled uniformly from replay do not become $d_\gamma^\pi$; SAC's replay-weighted policy-improvement objective is not an exact discounted start-distribution gradient.
```

&nbsp;
