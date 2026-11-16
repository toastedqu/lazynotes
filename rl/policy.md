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

# Policy-Based Methods & Actor-Critic

Optimize the policy directly: score-function gradients → advantage estimation → constrained updates → deterministic & entropy-regularized control. Study notes from {cite:t}`rlbook`; assumes [MDPs](intro.md#mdp) & [value functions](intro.md#value-functions).

Notations:
- $s_t$: State at time $t$.
- $a_t$: Action taken in $s_t$.
- $r_{t+1}$: Reward after $a_t$.
- $\rho_0$: Initial-state distribution.
- $\gamma$: Discount factor; $0\leq\gamma<1$ unless a finite/terminating task permits $\gamma=1$.
- $T$: Terminal time; $\infty$ for continuing tasks.
- $G_t$: Return $\sum_{l=0}^{T-t-1}\gamma^l r_{t+l+1}$.
- $\pi_\theta$: Target stochastic policy.
- $\theta$: Actor params.
- $\phi$: Critic params.
- $V^\pi$: State value under $\pi$.
- $Q^\pi$: Action value under $\pi$.
- $A^\pi$: Advantage $Q^\pi-V^\pi$.
- $J$: Expected discounted start-state return $\mathbb E_{\rho_0,\pi_\theta}[G_0]$.
- $d_\gamma^\pi$: Normalized discounted state occupancy $(1-\gamma)\sum_{t\geq0}\gamma^t\Pr_\pi(s_t=\cdot)$.
- $d_t$: True-termination indicator after transition $t$.
- $c_t$: Episode- or collected-segment-boundary indicator after transition $t$; $d_t\leq c_t$.
- $b_t$: Bootstrap mask $1-d_t$.
- $\delta_t$: One-step TD residual.
- $\widehat A_t$: Estimated advantage.
- $\widehat R_t$: Critic return target.
- $\mathcal R$: Replay buffer.
- $\mathcal H$: Policy entropy; differential entropy for continuous actions.
- $\beta$: Entropy coefficient $\geq0$; temperature in maximum-entropy control.
- $\theta_{\mathrm{old}}$: Frozen params of the policy that collected a rollout.
- $\bar\phi$: Target-critic params (slowly updated).
- $\operatorname{sg}$: Stop-gradient; numerical identity w/ zero derivative.
- $\widehat{\mathbb E}$: Empirical average over a stated sample batch.
- $\omega_t$: Actor sample weight; $\gamma^t$ for the exact episodic discounted gradient, $1$ for the usual rollout surrogate.

RL overrides: $s$ & $a$ = states & actions, possibly vectors. $t,l$ = time & time offset. $A^\pi$ = advantage function, not a matrix.

&nbsp;

## Policy representation

### Categorical policy
- **What**: State-conditioned softmax over a finite action set.
- **Why**: Greedy argmax has no gradient → learn smooth action preferences instead.
- **How**:
    1. NN maps state → one logit per valid action.
    2. Softmax → action probabilities.
    3. Sample to act. Differentiate the sampled action's log-prob.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $z_\theta(s)\in\mathbb R^C$: Logit vector.
    - $C$: #actions.
    - $j$: Action idx; overrides the global feature idx.
    - $q$: Summation idx over actions.

Forward:

$$
\pi_\theta(a=j\mid s)
=\frac{\exp z_{\theta,j}(s)}
{\sum_{q=1}^{C}\exp z_{\theta,q}(s)}
$$

Backward:

$$
\frac{\partial\log\pi_\theta(a\mid s)}{\partial z_{\theta,j}(s)}
=\mathbf 1\{a=j\}-\pi_\theta(j\mid s)
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
            ## mask BEFORE softmax -> invalid actions get exactly 0 prob; needs >=1 valid action
            logits = logits.masked_fill(~valid, -torch.inf)
        return logits.log_softmax(dim=-1)

    def sample(self, state, valid=None):
        logp = self(state, valid)
        action = torch.multinomial(logp.exp(), 1)
        ## keep the SAMPLED action's log-prob -> score-function gradient
        return action.squeeze(-1), logp.gather(-1, action).squeeze(-1)

## Example
policy = CategoricalPolicy(3, 2)
action, logp = policy.sample(torch.zeros(2, 3))
print(action.shape, logp.shape)  ## torch.Size([2]) torch.Size([2])
```
````

```{attention} Q&A
:class: dropdown
*Why not backprop through the sampled action?*

- Sampling is discrete → ❌gradient through the integer.
- Score-function estimator differentiates $\log\pi_\theta(a\mid s)$ w/ $a$ held fixed.
- Argmax during training → ❌the stochastic sampling the estimator assumes.

*How do invalid-action masks interact with PPO?*

- Mask before softmax. Keep ≥1 valid action per state.
- Use the same state-dependent mask for stored old log-probs & recomputed new log-probs.
- Learned, param-dependent mask → support changes w/ $\theta$ → likelihood-ratio identities can fail.

*Must an optimal MDP policy randomize?*

- ❌. Finite discounted fully observed MDPs have a deterministic stationary optimal policy.
- Randomization still helps exploration.
- Entropy objectives, partial observability w/ restricted policies & games can require randomization.
```

&nbsp;

### Gaussian policy
- **What**: State-conditioned diagonal Gaussian over continuous actions.
- **Why**: Continuous action space → ❌enumerate actions for a softmax.
- **How**:
    1. NN predicts a mean & a positive std per action coordinate.
    2. Action = mean + std ⊙ standard-normal noise.
    3. Joint log-density = sum of coordinate log-densities.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\epsilon\in\mathbb R^D$: Param-independent sampling noise.
    - $m_\theta(s)\in\mathbb R^D$: Mean vector; subscript separates it from global $m$ (#samples).
    - $\sigma_\theta(s)\in\mathbb R_{>0}^D$: Coordinate stds.
    - $D$: Action dimension.
    - $\odot$: Elementwise product.

Forward:

$$
\begin{align*}
\epsilon&\sim\mathcal N(0,I)\\
a&=m_\theta(s)+\sigma_\theta(s)\odot\epsilon\\
\log\pi_\theta(a\mid s)
&=-\frac12\sum_{j=1}^{D}
\left[
\frac{(a_j-m_{\theta,j}(s))^2}{\sigma_{\theta,j}(s)^2}
+2\log\sigma_{\theta,j}(s)+\log(2\pi)
\right]
\end{align*}
$$
- $j$: Action-coordinate idx.

Backward:

$$
\left.\frac{\partial\log\pi_\theta(a\mid s)}
{\partial m_{\theta,j}(s)}\right|_{a\ \mathrm{fixed}}
=\frac{a_j-m_{\theta,j}(s)}{\sigma_{\theta,j}(s)^2}
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
        ## state-independent log-std: unconstrained param -> std = exp(.) > 0
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        mean = self.mean(state)
        return mean, self.log_std.expand_as(mean)

    def log_prob(self, state, action):
        mean, log_std = self(state)
        standardized = (action - mean) * (-log_std).exp()
        ## SUM over action coords = joint density of a diagonal Gaussian
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
print(action.shape, logp.shape)  ## torch.Size([2, 2]) torch.Size([2])
```
````

```{attention} Q&A
:class: dropdown
*Score-function gradient vs reparameterization gradient?*

- REINFORCE/PPO: stored action = constant → differentiate its log-density.
- SAC: keep the sampled action's path to $\theta$ → differentiate the critic through it.
- Reparameterized action left attached in a score-function loss → different estimator.
    - e.g., unsquashed Gaussian's own reparameterized log-density: $(a-m_\theta)/\sigma_\theta=\epsilon$ → mean grad $=0$.

*What does a diagonal Gaussian assume?*

- Action coordinates independent given the state → ❌learned within-state covariance.
- Mean can depend on every state feature.
- Code: state-independent learned std. State-dependent std is another valid parameterization.
- Unimodal → ❌multimodal action distributions. Std must be finite & $>0$.

*Sum or mean of the joint log-prob?*

- Sum over action coordinates. Mean over batch samples.
- Averaging coordinates changes likelihood ratios & entropy, not only style.
```

&nbsp;

#### Tanh-squashed Gaussian
- **What**: Gaussian latent → $\tanh$ → affine rescale into action bounds. {cite:p}`haarnoja2018soft`
- **Why**: Gaussian samples are unbounded. Actuator commands are bounded.
- **How**:
    1. Sample an unbounded latent action.
    2. Apply $\tanh$, then scale & shift into the action range.
    3. Correct the density for the transform's local volume change.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $u\in\mathbb R^D$: Unsquashed latent action.
    - $h_j$: Midpoint of coordinate $j$'s action interval.
    - $q_j>0$: Half-width of that interval.

Forward:

$$
\begin{align*}
u&\sim\mathcal N\!\left(m_\theta(s),
             \operatorname{diag}(\sigma_\theta(s)^2)\right)\\
a_j&=h_j+q_j\tanh u_j\\
\log\pi_\theta(a\mid s)
&=\log\mathcal N(u;m_\theta(s),\operatorname{diag}(\sigma_\theta(s)^2))
-\sum_{j=1}^{D}\left[\log q_j+\log(1-\tanh^2 u_j)\right]
\end{align*}
$$

- Stable identity:

    $$
    \log(1-\tanh^2 u)=2\left[\log 2-u-\operatorname{softplus}(-2u)\right]
    $$
```

```{tip} Derivation
:class: dropdown
1. Change of variables: action density = latent density ÷ |Jacobian determinant|.
2. Coordinatewise transform → diagonal Jacobian w/ entries $q_j(1-\tanh^2u_j)$.
3. Log-determinant = sum of coordinate log-scales → subtract it from the Gaussian log-density.
4. Store $u$ at sampling time → ❌unstable $\operatorname{atanh}$ near the action bounds.
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
        low = torch.tensor(low, dtype=torch.float32)
        high = torch.tensor(high, dtype=torch.float32)
        self.register_buffer("center", (high + low) / 2)
        self.register_buffer("scale", (high - low) / 2)

    def log_prob(self, state, latent):
        base_logp = super().log_prob(state, latent)
        ## log|da/du| per coord, via the softplus identity -> no log(0) at saturation
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
        ## return the latent too: later log-probs reuse it instead of atanh(action)
        return action, self.log_prob(state, latent), latent

## Example
policy = SquashedGaussian(3, 2, [-2, -1], [2, 3])
action, logp, latent = policy.sample(torch.zeros(2, 3))
print(action.shape, logp.shape)  ## torch.Size([2, 2]) torch.Size([2])
```
````

```{attention} Q&A
:class: dropdown
*Why not clip Gaussian actions & evaluate the Gaussian density at the clipped value?*

- Clipping maps whole tail intervals onto boundary points → mixed distribution w/ point masses at the bounds.
- Gaussian density at the clipped value ≠ likelihood of that mixed distribution.
- Valid alternative: store the pre-clip action as the policy action; treat clipping as part of the environment. Likelihoods then use the stored pre-clip action.

*Can PPO omit the Jacobian?*

- In a ratio only: same fixed invertible transform & same stored latent → old & new Jacobians cancel.
- Absolute log-densities, SAC entropy & temperature tuning still need the correction.
- Bounds changed between collection & optimization → ❌cancellation.

*Does squashing preserve Gaussian entropy?*

- ❌. Saturation compresses volume → larger latent std need not raise action entropy.
- Differential entropy depends on action units. Affine scaling shifts it by $\sum_j\log q_j$ → set entropy targets in the same coordinates.
- Float $\tanh$ can round to a bound → stored latents & the stable log-Jacobian avoid $\log 0$.
```

&nbsp;

## Stochastic policy gradients

### Policy-gradient theorem
- **What**: $\nabla_\theta J$ = expected policy score × action value. {cite:p}`sutton1999policy`
- **Why**: $\theta$ changes the future state distribution → direct differentiation needs the unknown dynamics.
    - Theorem → gradient needs only on-policy samples, policy scores & $Q^\pi$.
- **How**:
    1. Differentiate the trajectory log-likelihood → transition terms drop out (❌depend on $\theta$).
    2. Credit each action only w/ rewards that come after it.
    3. Replace the sampled future return w/ its conditional mean $Q^\pi(s_t,a_t)$.

```{note} Math
:class: dropdown
Objective:

$$
J(\theta)=\mathbb E_{\rho_0,\pi_\theta}[G_0]
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
\left[\nabla_\theta\log\pi_\theta(a\mid s)Q^{\pi_\theta}(s,a)\right]
\end{align*}
$$

- Second form needs $\gamma<1$. Pad terminal states w/ absorbing, zero-reward behavior.
- Assumptions:
    - Initial distribution & environment independent of $\theta$.
    - Policy differentiable on a fixed support.
    - Required expectations finite; gradient & integral can be exchanged.
- Finite-horizon nonstationarity → put remaining time in the state, or index policies & values by time.
- The integrand differentiates only the policy score. The theorem already accounts for future policy effects.
```

```{tip} Derivation
:class: dropdown
1. Factor the trajectory probability:

    $$
    p_\theta(\xi)=\rho_0(s_0)
    \prod_{t=0}^{T-1}
    \pi_\theta(a_t\mid s_t)P(s_{t+1},r_{t+1}\mid s_t,a_t)
    $$
    - $\xi$: Complete trajectory incl. rewards.
    - $P$: Joint next-state/reward kernel.

2. Apply the likelihood-ratio identity $\nabla p_\theta=p_\theta\nabla\log p_\theta$; only policy factors depend on $\theta$:

    $$
    \nabla_\theta J
    =\int p_\theta(\xi)G_0(\xi)\nabla_\theta\log p_\theta(\xi)\,d\xi
    =\mathbb E\left[G_0\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)\right]
    $$

3. Drop past rewards. Given the history before $a_t$, the score has zero mean:

    $$
    \mathbb E_{a_t\sim\pi_\theta}
    [\nabla_\theta\log\pi_\theta(a_t\mid s_t)\mid s_t]
    =\nabla_\theta\int\pi_\theta(a\mid s_t)\,da=0
    $$

4. Factor the remaining discount:

    $$
    \sum_{l=t}^{T-1}\gamma^l r_{l+1}=\gamma^tG_t
    $$

5. Condition $G_t$ on $(s_t,a_t)$ → $Q^{\pi_\theta}(s_t,a_t)$. Collect discounted state visits → occupancy form.
```

```{attention} Q&A
:class: dropdown
*Where did the gradient of the state occupancy go?*

- It did not vanish. $Q^\pi$ & discounted visitation carry the recursive future-policy effects.
- Treating a replay distribution as fixed is a different approximation, not this theorem.

*Can the outer $\gamma^t$ be dropped?*

- ❌ for the exact start-distribution objective w/ ordinary episode trajectories.
- Sample states directly from $d_\gamma^\pi$ → weighting absorbed. $1/(1-\gamma)$ remains a policy-independent scale.
- Common practice: uniform rollout sampling, discounted advantages, no outer weight → a surrogate.
    - Generally ≠ exact gradient of $J$; ≠ automatically an average-reward gradient.
- Sum time terms within each episode, then average episodes. Policy-dependent episode lengths + per-transition normalization → changed weighting.

*Does an exact gradient guarantee global convergence?*

- ❌. The theorem gives a derivative, not a global optimizer.
- Stochastic-approximation convergence needs extra smoothness, step-size, bounded-noise & sampling assumptions.
- General neural parameterizations → ❌blanket global-optimum guarantee.

*Can any learned critic replace $Q^\pi$ w/o bias?*

- ❌. Approximate action values generally bias the actor update.
- Exception (compatible function approximation): critic linear in the policy score $\nabla_\theta\log\pi_\theta$, fitted by occupancy-weighted least squares → expected score × critic = expected score × $Q^\pi$.
- A neural critic trained on TD targets does not satisfy that condition automatically.
- Link: compatible critic weights = [natural policy gradient](#natural-policy-gradient) direction.
```

&nbsp;

### REINFORCE
- **Name**: REward Increment = Nonnegative Factor × Offset Reinforcement × Characteristic Eligibility {cite:p}`williams1992simple`
- **What**: Monte Carlo policy gradient: sampled return × policy score.
- **Why**: Estimate $\nabla_\theta J$ w/o a learned critic or a differentiable environment.
- **How**:
    1. Collect complete episodes under the curr policy.
    2. Compute the reward-to-go of every sampled action.
    3. Return above/below an action-independent baseline → raise/lower that action's log-prob.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\alpha>0$: Actor step size.
- Misc:
    - $v(s)$: Action-independent baseline, fixed while estimating the actor gradient.
    - $T_i$: Terminal time of episode $i$.

Process:

$$
\widehat g=\frac1m\sum_{i=1}^{m}\sum_{t=0}^{T_i-1}
\gamma^t\nabla_\theta\log\pi_\theta(a_t^{(i)}\mid s_t^{(i)})
\left[G_t^{(i)}-v(s_t^{(i)})\right]
$$

$$
\theta\leftarrow\theta+\alpha\widehat g
$$

Objective:

$$
\mathcal L_{\mathrm{actor}}
=-\frac1m\sum_i\sum_t\gamma^t
\log\pi_\theta(a_t^{(i)}\mid s_t^{(i)})
\operatorname{sg}\!\left(G_t^{(i)}-v(s_t^{(i)})\right)
$$

- Minimize this sample loss → its negative gradient = $\widehat g$.
- $v=V_\phi$ → fit the baseline separately to Monte Carlo returns.
```

```{tip} Derivation
:class: dropdown
*Why is a baseline free, & which baseline is best?*

1. An action-independent baseline has zero expected score:

    $$
    \mathbb E_{a\sim\pi_\theta}
    [v(s)\nabla_\theta\log\pi_\theta(a\mid s)]
    =v(s)\nabla_\theta\int\pi_\theta(a\mid s)\,da=0
    $$

2. → Subtracting $v(s_t)$ leaves the expected gradient unchanged.
3. For one conditional score term, minimize the expected squared norm over a scalar baseline:

    $$
    v^*(s)=
    \frac{\mathbb E[G_t\|z_t\|^2\mid s_t=s]}
         {\mathbb E[\|z_t\|^2\mid s_t=s]}
    $$
    - $z_t$: Policy score $\nabla_\theta\log\pi_\theta(a_t\mid s_t)$.
    - Needs a positive denominator. Zero score → that state's baseline does not matter for this term.

4. $V^\pi(s)$ is a good baseline, but ≠ $v^*$ in general ← score norms weight actions differently.
```

````{important} Code
:class: dropdown
```python
import torch

class ReinforceLoss:
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, logp, returns, baseline, valid):
        ## padded complete episodes: all inputs [time, episodes]
        time = torch.arange(logp.shape[0], device=logp.device)
        weight = logp.new_tensor(self.gamma).pow(time)[:, None]  ## exact gamma^t weight
        ## detach: the baseline must not receive actor gradient
        advantage = (returns - baseline).detach()
        terms = torch.where(valid, weight * logp * advantage, 0.0)
        return -terms.sum(dim=0).mean()  ## average episodes, not lengths

## Example
logp = torch.tensor([[-0.7], [-0.4]], requires_grad=True)
returns = torch.tensor([[2.0], [2.0]])  ## rewards [1, 2], gamma=0.5
loss = ReinforceLoss(0.5)(logp, returns, torch.zeros_like(logp),
                          torch.ones_like(logp, dtype=torch.bool))
loss.backward()
print(logp.grad.flatten())  ## tensor([-2., -1.])
```
````

```{attention} Q&A
:class: dropdown
*Why must the baseline be action-independent?*

- Zero-score identity factors out only a quantity fixed before the curr action.
- Action-dependent baseline → changed gradient, unless a special correction removes the bias.
- Baseline shares actor features → stop-gradient it in the actor term. Fit its value loss through its own path.

*Does fitting the baseline on the same rollout keep exact finite-sample unbiasedness?*

- Not automatically ← fitted params can depend on that rollout's actions & rewards.
- Clean proof: fixed baseline, or one estimated independently of the curr action. Same-batch fitting = practical approximation.
- Batch advantage centering/scaling also need not keep the exact unbiased estimator.

*Pros?*

- ❌Bootstrap bias. Complete on-policy returns → unbiased gradient under the stated assumptions.
- Simple: policy only, ❌critic required.

*Cons?*

- Long horizons, noisy rewards & rare successes → high variance.
- Must wait for episode end.
- Artificial cutoff ≠ complete episode. Bootstrapping its continuation → actor–critic, not pure Monte Carlo REINFORCE.
```

&nbsp;

### Actor–critic
- **What**: Policy gradient w/ a learned value critic in the advantage.
- **Why**: Full-return gradients are noisy & wait for episode end.
- **How**:
    1. Critic predicts future return.
    2. Observed reward + predicted continuation → training target.
    3. Actor follows the critic's advantage estimate. Critic regresses toward the target.

```{note} Math
:class: dropdown
Process:

$$
\begin{align*}
\widehat R_t&=r_{t+1}+\gamma b_tV_\phi(s_{t+1})\\
\delta_t&=\widehat R_t-V_\phi(s_t)
\end{align*}
$$

- $V_\phi=V^\pi$ & correct terminal handling →

    $$
    \mathbb E[\delta_t\mid s_t,a_t]=A^\pi(s_t,a_t)
    $$

Objective:

$$
\begin{align*}
\mathcal L_{\mathrm{actor}}
&=-\widehat{\mathbb E}
  [\omega_t\log\pi_\theta(a_t\mid s_t)\operatorname{sg}(\delta_t)]\\
\mathcal L_{\mathrm{critic}}
&=\frac12\widehat{\mathbb E}
  [(V_\phi(s_t)-\operatorname{sg}(\widehat R_t))^2]
\end{align*}
$$

- Empirical average = batch surrogate. Exact episodic normalization follows REINFORCE.
- Critic update = semi-gradient: ❌differentiate its bootstrap target.
```

````{important} Code
:class: dropdown
```python
import torch

class ActorCriticLoss:
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, logp, value, next_value, reward, terminated, weight):
        ## all inputs [batch]; true termination -> no bootstrap
        bootstrap = torch.where(terminated, 0.0, next_value.detach())
        target = (reward + self.gamma * bootstrap).detach()  ## semi-gradient target
        advantage = (target - value).detach()  ## actor must not train the critic
        actor = -(weight.detach() * logp * advantage).mean()
        critic = 0.5 * (value - target).square().mean()
        return actor, critic

## Example
logp = torch.tensor([-0.7, -0.4], requires_grad=True)
value = torch.tensor([1.0, 1.0], requires_grad=True)
actor, critic = ActorCriticLoss(0.5)(
    logp, value, torch.tensor([4.0, 99.0]), torch.tensor([1.0, 2.0]),
    torch.tensor([False, True]), torch.ones(2)
)
print(actor.item(), critic.item())  ## targets [3, 2] -> advantages [2, 1]
```
````

```{attention} Q&A
:class: dropdown
*Is REINFORCE w/ a learned baseline already actor–critic?*

- Terminology varies.
- Here: actor–critic = learned values inside the actor's return/advantage estimate, typically by bootstrapping.
- Monte Carlo baseline alone → variance ⬇️ w/o a bootstrapped return.

*Where does actor bias enter?*

- Imperfect $V_\phi(s_t)$ used only as a fixed state baseline → ❌bias.
- Imperfect $V_\phi(s_{t+1})$ inside the return target → bias in general.
- Exact on-policy values → conditional advantage identity holds. Stale policies, critic error & truncated importance weights break it.

*Should actor & critic share params?*

- Shared features → compute ⬇️, but the 2 optimization problems couple.
- Separate heads do not stop value-loss gradients from changing shared features.
- Detach advantages & targets. Combine only the intended actor, critic & optional entropy gradients.

*What convergence claim survives deep actor–critic?*

- Classical results need specific approximation, sampling &, often, two-timescale assumptions.
- Simultaneous updates of arbitrary neural actor & critic → ❌generic convergence guarantee.
```

&nbsp;

#### A2C
- **Name**: Advantage Actor–Critic
- **What**: Synchronous, batched advantage actor–critic over parallel environments.
- **Why**: Decorrelate experience, & keep one policy version per rollout.
- **How**:
    1. Step several environments w/ the same actor snapshot.
    2. Compute multi-step or GAE advantages & return targets.
    3. Aggregate gradients → one synchronized actor/critic update.

```{attention} Q&A
:class: dropdown
*How does A2C differ from vanilla actor–critic?*

- Same advantage-weighted update. Synchronous parallel collection & batched optimization = implementation pattern.
- Reuse `ActorCriticLoss` w/ the multi-step advantages & targets below.
- 1 update per rollout = on-policy. Repeated uncorrected actor steps on that batch → policy mismatch.

*Pros?*

- One policy version per batch. Vectorized compute. ❌Asynchronous gradient races.

*Cons?*

- Workers wait for the slowest environment.
- Plain on-policy estimator → ❌long-lived replay.
```

&nbsp;

#### A3C
- **Name**: Asynchronous Advantage Actor–Critic {cite:p}`mnih2016asynchronous`
- **What**: Parallel actor–critic workers w/ asynchronous updates to shared params.
- **Why**: Diverse, decorrelated experience w/o a synchronization barrier.
- **How**:
    1. Each worker copies shared actor/critic params into local networks.
    2. Collect a short trajectory. Bootstrap its end if nonterminal.
    3. Compute local multi-step policy & value gradients, optionally w/ an entropy bonus.
    4. Apply gradients to shared params. Refresh the local copy.

```{attention} Q&A
:class: dropdown
*A2C vs A3C?*

- A2C: synchronized batch, one shared policy snapshot.
- A3C: independent workers, possibly stale gradients, ❌global barrier.
- Both = advantage actor–critic. Asynchrony = scheduling difference, not a new policy-gradient theorem.

*Is A3C off-policy?*

- Each worker follows its local policy. The shared policy can change before the gradient is applied.
- Usually called on-policy (❌replay). Staleness still → mismatch → ❌exact curr-policy gradient.

*Does an entropy bonus make A3C into SAC?*

- ❌. A local entropy bonus rewards randomness only at sampled states.
- SAC also puts future entropy into the critic's Bellman targets & does off-policy soft policy improvement.
```

&nbsp;

### n-step advantage
- **What**: $h$-step bootstrapped return minus $V(s_t)$.
- **Why**: Full returns are noisy; 1-step targets depend fully on the critic → $h$ sets the tradeoff.
- **How**:
    1. Sum rewards up to the horizon or the first segment boundary.
    2. Bootstrap only if the endpoint is not truly terminal.
    3. Subtract the start state's frozen value.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $h$: #collected transitions used from $t$; capped by the horizon & the first $c=1$ boundary.

Process:

$$
\begin{align*}
\widehat R_t^{(h)}
&=\sum_{l=0}^{h-1}\gamma^l r_{t+l+1}
+\gamma^h b_{t+h-1}V_\phi(s_{t+h})\\
\widehat A_t^{(h)}
&=\widehat R_t^{(h)}-V_\phi(s_t)
\end{align*}
$$

- Never cross an episode reset inside the sum.
- Both values come from the same frozen snapshot when preparing actor targets.
- Same return as [n-step value learning](value.md#n-step-returns). Subtracting $V_\phi(s_t)$ → advantage.
```

````{important} Code
:class: dropdown
```python
import torch

class NStepAdvantage:
    def __init__(self, gamma, horizon):
        self.gamma, self.horizon = gamma, horizon

    @torch.no_grad()
    def __call__(self, reward, value, next_value, terminated, boundary):
        ## all inputs [time, envs]; next_value = V(final obs of each transition), not the reset obs
        targets = torch.zeros_like(reward)
        length = reward.shape[0]
        for t in range(length):
            discount = torch.ones_like(reward[t])
            for offset in range(min(self.horizon, length - t)):
                end = t + offset
                ## stop at a segment boundary, the horizon, or the end of the rollout
                stop = (boundary[end] | (offset == self.horizon - 1)
                        | (end == length - 1))
                ## bootstrap only at the stopping step & only if not truly terminal
                boot = torch.where(terminated[end], 0.0, next_value[end])
                targets[t] += discount * (
                    reward[end] + self.gamma * stop * boot
                )
                discount = discount * self.gamma * (~stop)  ## 0 after stop -> no more terms
        return targets - value, targets

## Example
r = torch.tensor([[1.0], [2.0], [100.0]])
v = torch.tensor([[4.0], [5.0], [8.0]])
nv = torch.tensor([[5.0], [6.0], [0.0]])
terminal = torch.tensor([[False], [False], [True]])
boundary = torch.tensor([[False], [True], [True]])  ## time-limit cut after t=1
adv, target = NStepAdvantage(0.5, 2)(r, v, nv, terminal, boundary)
print(target.flatten())  ## tensor([  3.5000,   5.0000, 100.0000])
```
````

```{attention} Q&A
:class: dropdown
*Termination vs truncation?*

- True terminal: $d_t=c_t=1$ → ❌bootstrap, ❌continue the trace.
- Time limit or rollout cutoff: $d_t=0,c_t=1$ → bootstrap, but ❌continue into the next segment.
- Ordinary transition: $d_t=c_t=0$ → bootstrap & continue.
- A real finite-horizon task is terminal at its horizon. Add remaining time to the state if the Markov property needs it.

*Which next observation to value after an automatic reset?*

- The ended segment's final observation, not the next episode's reset observation.
- Reset observation → unrelated future value leaks into truncated-episode targets.

*How does the horizon trade bias & variance?*

- Horizon ⬆️ → dependence on bootstrap error ⬇️, sampled-return variance ⬆️.
- True terminal endpoint → full return minus a fixed baseline → ❌bootstrap bias.
- Nonterminal cutoff → final critic error remains, scaled by $\gamma^h$.

*Can replayed behavior-policy trajectories supply these returns unchanged?*

- Only if their continuation policy = the evaluated policy, or w/ a justified correction.
- Uncorrected multi-step returns follow behavior actions inside the segment. One final target-policy bootstrap does not fix that.
```

&nbsp;

### GAE
- **Name**: Generalized Advantage Estimation {cite:p}`schulman2016high`
- **What**: $(\gamma\lambda)$-discounted sum of TD residuals.
- **Why**: One fixed horizon $h$ = one fixed bias–variance point.
    - GAE = exponential mix of all horizons → one knob $\lambda$ tunes the tradeoff smoothly.
- **How**:
    1. Compute 1-step TD residuals from frozen rollout values.
    2. Accumulate them backward w/ decay $\gamma\lambda$.
    3. Reset the accumulator at every segment boundary. Keep nonterminal bootstraps.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\lambda\in[0,1]$: Trace decay.

Process:

$$
\begin{align*}
\delta_t&=r_{t+1}+\gamma b_tV_\phi(s_{t+1})-V_\phi(s_t)\\
\widehat A_t^{\mathrm{GAE}}
&=\delta_t+\gamma\lambda(1-c_t)\widehat A_{t+1}^{\mathrm{GAE}}\\
\widehat R_t&=\widehat A_t^{\mathrm{GAE}}+V_\phi(s_t)
\end{align*}
$$

- Initialize the backward accumulator to 0 after the last collected transition.
- $\widehat R_t$ = $\lambda$-return target, not necessarily a Monte Carlo return.
```

```{tip} Derivation
:class: dropdown
1. Within a segment, discounted residuals telescope:

    $$
    \sum_{l=0}^{h-1}\gamma^l\delta_{t+l}
    =\widehat A_t^{(h)}
    $$

2. Mix horizons geometrically; the endpoint keeps the remaining weight:

    $$
    \widehat A_t^{\mathrm{GAE}}
    =(1-\lambda)\sum_{h=1}^{H-1}\lambda^{h-1}\widehat A_t^{(h)}
    +\lambda^{H-1}\widehat A_t^{(H)}
    $$
    - $H$: Remaining transitions to this segment's endpoint.

3. Collect each residual's coefficient:

    $$
    \widehat A_t^{\mathrm{GAE}}
    =\sum_{l=0}^{H-1}(\gamma\lambda)^l\delta_{t+l}
    $$

4. Factor out the first residual → backward recurrence. $1-c_t$ stops it at the endpoint.
```

````{important} Code
:class: dropdown
```python
import torch

class GAE:
    def __init__(self, gamma, trace_decay):
        self.gamma, self.trace_decay = gamma, trace_decay

    @torch.no_grad()
    def __call__(self, reward, value, next_value, terminated, boundary):
        ## all inputs [time, envs]
        ## residual uses d_t: time-limit cuts still bootstrap
        bootstrap = torch.where(terminated, 0.0, next_value)
        residual = reward + self.gamma * bootstrap - value
        advantage = torch.empty_like(residual)
        carry = torch.zeros_like(residual[0])
        for t in reversed(range(reward.shape[0])):
            ## recurrence uses c_t: never mix residuals across segments
            carry = residual[t] + self.gamma * self.trace_decay * (
                ~boundary[t]
            ) * carry
            advantage[t] = carry
        return advantage, advantage + value  ## (advantage, lambda-return target)

## Example
r = torch.tensor([[1.0], [2.0], [100.0]])
v = torch.tensor([[4.0], [5.0], [8.0]])
nv = torch.tensor([[5.0], [6.0], [0.0]])
terminal = torch.tensor([[False], [False], [True]])
boundary = torch.tensor([[False], [True], [True]])
adv, target = GAE(0.5, 1.0)(r, v, nv, terminal, boundary)
print(adv.flatten(), target.flatten())
## tensor([-0.5000,  0.0000, 92.0000]) tensor([  3.5000,   5.0000, 100.0000])
```
````

```{attention} Q&A
:class: dropdown
*Limiting cases?*

- $\lambda=0$: 1-step residual.
- $\lambda=1$: return to the segment end (incl. its nonterminal bootstrap) minus the start value.
- $\gamma=0$: only the immediate reward matters; $\lambda$ has no effect.

*When is GAE unbiased?*

- Exact $V^\pi$, on-policy continuations & correct boundaries → each component has the correct conditional advantage mean.
- Approximate values & $\lambda<1$ → bootstrap bias in general.
- $\lambda=1$ through true termination → ❌bootstrap bias in the policy-gradient estimator, even w/ an imperfect fixed baseline. The advantage estimate itself need not equal $A^\pi$ in mean.
- $\lambda=1$ at a nonterminal cutoff → still depends on the endpoint value.
- Unbiased advantages do not fix a wrongly weighted actor objective (the outer $\gamma^t$ issue).

*Why 2 masks?*

- $1-c_t$ in the residual → wrongly zeroes time-limit continuation values.
- $1-d_t$ in the recurrence → mixes the next episode/segment's residuals into the curr one.
- Rollout pauses & later resumes the same episode → episode-time idx does not reset; only the collected trace ends.
```

&nbsp;

## Conservative policy updates

### Natural policy gradient
- **What**: Policy gradient preconditioned by the inverse Fisher matrix. {cite:p}`kakade2001natural`
- **Why**: Plain gradient steps are measured in param space.
    - Same param step → tiny or huge policy change, depending on the parameterization.
    - Natural gradient measures the step by KL between policies → parameterization-invariant to first order.
- **How**:
    1. Estimate the policy gradient & the Fisher matrix from the same on-policy samples.
    2. Solve Fisher × direction = gradient.
    3. Step along that direction.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\alpha>0$: Step size.
- Misc:
    - $z$: Policy score $\nabla_\theta\log\pi_\theta(a\mid s)$.
    - $F$: Fisher information matrix of the policy.

Process:

$$
\begin{align*}
F(\theta)&=\mathbb E_{\substack{s\sim d_\gamma^{\pi_\theta}\\a\sim\pi_\theta(\cdot\mid s)}}\left[zz^\top\right]\\
\theta&\leftarrow\theta+\alpha F(\theta)^{-1}\nabla_\theta J
\end{align*}
$$

- Singular $F$ → pseudoinverse or damping $F+\lambda_F I$.
    - $\lambda_F>0$: Damping coefficient.
- $F$ = local metric of mean KL:

    $$
    \mathbb E_{s\sim d_\gamma^{\pi_\theta}}\left[D_{\mathrm{KL}}\left(\pi_\theta(\cdot\mid s)\Vert\pi_{\theta+\Delta\theta}(\cdot\mid s)\right)\right]
    =\tfrac12\Delta\theta^\top F\Delta\theta+O(\|\Delta\theta\|^3)
    $$
    - $D_{\mathrm{KL}}$: Kullback–Leibler divergence.
```

```{tip} Derivation
:class: dropdown
*Why is $F^{-1}\nabla_\theta J$ the steepest-ascent direction?*

1. Maximize the linear gain $g^\top\Delta\theta$ s.t. $\tfrac12\Delta\theta^\top F\Delta\theta\leq\varepsilon$.
    - $g$: $\nabla_\theta J$.
    - $\varepsilon$: KL budget.
2. Lagrangian stationarity: $g-\nu F\Delta\theta=0$.
    - $\nu>0$: Multiplier.
3. → $\Delta\theta=\nu^{-1}F^{-1}g$. Direction = $F^{-1}g$; $\nu$ only sets the length.
4. Plain gradient = same problem w/ the Euclidean metric $\tfrac12\|\Delta\theta\|^2$ instead of KL.
```

````{important} Code
:class: dropdown
```python
import torch

class NaturalPolicyGradient:
    def __init__(self, lr, damping=1e-3):
        self.lr, self.damping = lr, damping

    def direction(self, scores, gradient):
        ## scores: [samples, params] = per-sample grad log pi; gradient: [params]
        fisher = scores.T @ scores / scores.shape[0]  ## empirical E[z z^T]
        ## damping -> invertible F (rank <= #samples)
        fisher = fisher + self.damping * torch.eye(
            gradient.numel(), device=gradient.device, dtype=gradient.dtype
        )
        return torch.linalg.solve(fisher, gradient)  ## F^-1 g w/o forming the inverse

    def __call__(self, scores, gradient):
        return self.lr * self.direction(scores, gradient)

## Example
scores = torch.tensor([[1.0, 0.0], [0.0, 2.0], [-1.0, 0.0], [0.0, -2.0]])  ## F = diag(0.5, 2)
step = NaturalPolicyGradient(0.1, damping=0.0)(scores, torch.tensor([1.0, 1.0]))
print(step)  ## tensor([0.2000, 0.0500]): low-curvature coord gets the bigger step
```
````

```{attention} Q&A
:class: dropdown
*Why is it parameterization-invariant?*

- KL depends only on the distributions, not on the coordinates of $\theta$.
- Smooth invertible reparameterization $\theta=h(\psi)$ w/ Jacobian $M$:
    - $g_\psi=M^\top g_\theta$ & $F_\psi=M^\top F_\theta M$.
    - → $M\,F_\psi^{-1}g_\psi=F_\theta^{-1}g_\theta$ → same first-order policy change.
- Plain gradient → $M M^\top g_\theta$ → policy change depends on the parameterization.

*How does it relate to compatible function approximation?*

- Fit $w^\top z$ to $Q^\pi$ by least squares under $s\sim d_\gamma^\pi$, $a\sim\pi$.
- Normal equations: $Fw=\mathbb E[zQ^\pi]=(1-\gamma)\nabla_\theta J$.
- → $w=(1-\gamma)F^{-1}\nabla_\theta J$: compatible critic weights = natural gradient direction.
- → natural actor–critic: read the step off the critic, ❌explicit Fisher inverse.

*What does NPG do for a tabular softmax policy?*

- Update becomes $\pi_{t+1}(a\mid s)\propto\pi_t(a\mid s)\exp\!\left(\frac{\alpha A^{\pi_t}(s,a)}{1-\gamma}\right)$. {cite:p}`agarwal2021theory`
- = multiplicative weights per state → ❌dependence on the state distribution.
- Exact gradients & $\theta_0=0$ → global convergence:

    $$
    J^*-J(\pi_K)\leq\frac{\log C}{\alpha K}+\frac{1}{(1-\gamma)^2K}
    $$
    - $K$: #NPG iterations.
    - $C$: #actions.
    - $J^*$: Optimal return.
- $\alpha\geq(1-\gamma)^2\log C$ → bound $\frac{2}{(1-\gamma)^2K}$ → ❌dependence on #states & #actions.
- Same exponential form as [soft policy improvement](#maximum-entropy-rl), w/ the old policy as prior.

*Cons?*

- $F$ is $p\times p$ for $p$ params → ❌explicit inverse for NNs → conjugate gradient w/ Fisher-vector products.
- Fixed $\alpha$ → KL change per step still uncontrolled → TRPO fixes the KL radius instead.
- Sampled $F$ & $\nabla_\theta J$ → noisy, can be ill-conditioned → damping needed.
```

&nbsp;

### TRPO
- **Name**: Trust Region Policy Optimization {cite:p}`schulman2015trust`
- **What**: Surrogate-advantage maximization under a mean-KL constraint.
- **Why**: Large policy step → old-policy advantages & state visits stop being valid → collapse.
- **How**:
    1. Collect old-policy data. Estimate advantages.
    2. Compute the natural-gradient direction w/ conjugate gradient.
    3. Scale it to the KL radius.
    4. Backtrack until the sampled surrogate improves & the mean KL stays within the radius.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\varepsilon_{\mathrm{KL}}>0$: Trust-region radius.
- Misc:
    - $\pi_{\mathrm{old}}$: $\pi_{\theta_{\mathrm{old}}}$.
    - $F$: Fisher matrix at $\theta_{\mathrm{old}}$, as in [natural policy gradient](#natural-policy-gradient).
    - $g$: Surrogate gradient at $\theta_{\mathrm{old}}$.

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
\right]\\
\text{s.t.}\quad
\mathbb E_{s\sim d_\gamma^{\pi_{\mathrm{old}}}}
\left[D_{\mathrm{KL}}(\pi_{\mathrm{old}}(\cdot\mid s)
\Vert\pi_\theta(\cdot\mid s))\right]&\leq\varepsilon_{\mathrm{KL}}
\end{align*}
$$

Process:

$$
\Delta\theta=
\sqrt{\frac{2\varepsilon_{\mathrm{KL}}}{g^\top F^{-1}g}}\,F^{-1}g
$$
- $\Delta\theta$: Optimal step for the linear objective & quadratic KL model; needs invertible $F$ & $g\neq0$.

- Large networks: Fisher-vector products + conjugate gradient. Damping & backtracking handle approximation error.
```

```{tip} Derivation
:class: dropdown
1. Performance-difference identity (exact):

    $$
    J(\pi_\theta)-J(\pi_{\mathrm{old}})
    =\frac1{1-\gamma}
    \mathbb E_{\substack{s\sim d_\gamma^{\pi_\theta}\\a\sim\pi_\theta}}
    [A^{\pi_{\mathrm{old}}}(s,a)]
    $$

2. Replace new-policy occupancy w/ old-policy occupancy → local surrogate, not the exact difference.
3. Importance-weight old-policy actions → surrogate computable from collected data.
4. Approximate mean KL locally by $\tfrac12\Delta\theta^\top F\Delta\theta$.
5. Maximize $g^\top\Delta\theta$ s.t. that quadratic ≤ $\varepsilon_{\mathrm{KL}}$ → direction $F^{-1}g$.
6. Pick the length so that $\tfrac12\Delta\theta^\top F\Delta\theta=\varepsilon_{\mathrm{KL}}$ → the scale factor above.
```

````{important} Code
:class: dropdown
```python
import torch
## Reuses NaturalPolicyGradient from rl/policy.md

class TrustRegionStep:
    def __init__(self, radius, damping=1e-3):
        self.radius = radius
        self.npg = NaturalPolicyGradient(lr=1.0, damping=damping)

    def __call__(self, scores, gradient):
        direction = self.npg.direction(scores, gradient)  ## F^-1 g
        curvature = gradient @ direction  ## g^T F^-1 g
        if curvature <= 0:
            return torch.zeros_like(gradient)  ## zero gradient -> no step
        ## scale so 0.5 d^T F d = radius under the DAMPED F; true KL model < radius if damping > 0
        return (2 * self.radius / curvature).sqrt() * direction

## Example
scores = torch.tensor([[1.0, 0.0], [0.0, 2.0], [-1.0, 0.0], [0.0, -2.0]])
step = TrustRegionStep(0.01, damping=0.0)(scores, torch.tensor([1.0, 1.0]))
fisher = scores.T @ scores / 4
print(step, 0.5 * step @ fisher @ step)  ## tensor([0.1789, 0.0447]) tensor(0.0100)
```
````

```{attention} Q&A
:class: dropdown
*Does practical TRPO guarantee monotonic improvement?*

- ❌. The theoretical bound uses worst-state (max) policy divergence & exact quantities.
- Practical TRPO uses sampled advantages, mean KL, local approximations & finite optimization.
- Those substitutions lose the unconditional monotonic-return guarantee.
- Low mean KL can hide a large change in a rare state.

*Is the code a complete TRPO optimizer?*

- ❌. It gives the local natural-gradient step only.
- A full update also: estimates Fisher & advantages, tests the actual surrogate & KL, backtracks, & rejects the step if no candidate passes.
- Common implementations sample states uniformly from rollouts, not from $d_\gamma^\pi$ → the exact-vs-surrogate distinction still applies.

*TRPO vs NPG?*

- Same direction $F^{-1}g$.
- NPG: fixed step size. TRPO: step length set by the KL radius + line search.

*Why move to PPO?*

- ❌Second-order constrained solve & line search. Keep a first-order brake on policy change.
- Cost: ❌explicit trust-region constraint.
```

&nbsp;

### PPO
- **Name**: Proximal Policy Optimization {cite:p}`schulman2017proximalpolicyoptimizationalgorithms`
- **What**: First-order policy optimization w/ a clipped (or KL-penalized) importance-ratio surrogate.
- **Why**: Reuse a fresh rollout for several updates.
    - Plain policy gradient → many steps on one batch → destructive policy change.
    - TRPO → stable but second-order & complex.
- **How**:
    1. Freeze a behavior snapshot. Collect actions, old log-probs & value predictions.
    2. Compute fixed advantages & value targets, usually w/ GAE.
    3. Optimize shuffled minibatches for a few epochs.
    4. Discard the rollout. Collect again w/ the updated policy.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\epsilon\in(0,1)$: Clipping width.
    - $c_V\geq0$: Value-loss coefficient.
- Misc:
    - $\varrho_t$: Curr/old action-probability ratio; not a reward.

Objective:

$$
\begin{align*}
\varrho_t(\theta)
&=\exp\!\left[
\log\pi_\theta(a_t\mid s_t)
-\operatorname{sg}(\log\pi_{\mathrm{old}}(a_t\mid s_t))
\right]\\
L^{\mathrm{clip}}(\theta)
&=\widehat{\mathbb E}\left[
\omega_t\min\left(
\varrho_t\operatorname{sg}(\widehat A_t),
\operatorname{clip}(\varrho_t,1-\epsilon,1+\epsilon)
\operatorname{sg}(\widehat A_t)
\right)\right]
\end{align*}
$$

$$
\mathcal L=
-L^{\mathrm{clip}}
+\frac{c_V}{2}\widehat{\mathbb E}
[(V_\phi(s_t)-\operatorname{sg}(\widehat R_t))^2]
-\beta\widehat{\mathbb E}
[\omega_t\mathcal H(\pi_\theta(\cdot\mid s_t))]
$$

- Minimize $\mathcal L$. The value term is not part of the policy-gradient theorem.
- $\omega_t=1$ → common rollout objective. Exact discounted-gradient reading at the old policy needs episode-weighted aggregation.
```

```{tip} Derivation
:class: dropdown
1. ❌Clipping: $\mathbb E_{\pi_{\mathrm{old}}}[\varrho_t\widehat A_t]$ importance-weights actions at fixed old-policy states.
2. $\widehat A_t>0$ → cap the gain from raising the action's prob:

    $$
    \min(\varrho_t,1+\epsilon)\widehat A_t
    $$

3. $\widehat A_t<0$ → cap the gain from lowering the action's prob:

    $$
    \max(\varrho_t,1-\epsilon)\widehat A_t
    $$

4. $\min$ picks the pessimistic branch.
    - Harmful changes keep their gradient.
    - Favorable changes past the clip earn ❌further surrogate gain.
5. Clipped objective lower-bounds the unclipped sample surrogate, not the true return.
```

```{note} Example
:class: dropdown
- $\epsilon=0.2$:
    - $\widehat A=1,\ \varrho=1.5$ → clipped contribution $1.2$.
    - $\widehat A=-1,\ \varrho=0.5$ → clipped contribution $-0.8$.
    - $\widehat A=-1,\ \varrho=1.5$ → contribution $-1.5$ → harmful increase stays penalized.
```

````{important} Code
:class: dropdown
```python
import torch

class PPOClipLoss:
    def __init__(self, epsilon, value_weight, entropy_weight):
        self.epsilon = epsilon
        self.value_weight, self.entropy_weight = value_weight, entropy_weight

    def __call__(self, logp, old_logp, advantage, value, target,
                 entropy, weight):
        ## all inputs [batch]: one scalar per sample -> no (m,1) vs (m,) broadcast
        ## ratio via log-probs; old_logp frozen = the reference being constrained
        ratio = (logp - old_logp.detach()).exp()
        advantage, weight = advantage.detach(), weight.detach()
        raw = ratio * advantage
        clipped = ratio.clamp(1 - self.epsilon, 1 + self.epsilon) * advantage
        ## pessimistic branch: min over raw & clipped
        actor = -(weight * torch.minimum(raw, clipped)).mean()
        critic = 0.5 * (value - target.detach()).square().mean()
        bonus = (weight * entropy).mean()
        return actor + self.value_weight * critic - self.entropy_weight * bonus

## Example
logp = torch.tensor([0.6, 0.2]).log().requires_grad_()
old_logp = torch.tensor([0.5, 0.4]).log()
loss = PPOClipLoss(0.2, 1.0, 0.0)(
    logp, old_logp, torch.tensor([1.0, -1.0]), torch.zeros(2),
    torch.tensor([1.0, -1.0]), torch.zeros(2), torch.ones(2)
)
print(loss.item())  ## ratios [1.2, 0.5] -> actor -(1.2 - 0.8)/2 = -0.2; critic 0.5
```
````

```{attention} Q&A
:class: dropdown
*What stays frozen during the rollout's optimization epochs?*

- Stored actions, old log-probs, advantages & return targets.
- Old log-probs must come from the actual collection policy, incl. its preprocessing, action masks & action transform.
- ❌Overwrite the denominator w/ the curr policy each minibatch → erases the reference.
- Shared actor/critic params may move. Value targets & actor advantages stay detached.

*Does clipping guarantee $\varrho_t\in[1-\epsilon,1+\epsilon]$?*

- ❌. It clips one branch of the objective, not params or probs.
- Other samples, shared features, entropy & value gradients can still move a clipped action.
- ❌Hard KL bound, ❌monotonic improvement, ❌global convergence.

*Why track KL if ratios are clipped?*

- Detect changes clipping did not stop → early-stop epochs past a KL budget.
- On old-policy samples w/ matching supports, $\widehat{\mathbb E}[(\varrho-1)-\log\varrho]$ estimates old→new KL; sampling error remains.
- Clip fraction measures ratio excursions, not safety or task performance.

*Can PPO train indefinitely on an old replay buffer?*

- ❌General justification. Action ratios do not correct the changed state distribution or stale advantages.
- Full-trajectory ratios correct trajectory distributions under coverage assumptions, but variance can explode w/ horizon. Clipping adds bias.
- Limited same-rollout reuse ≠ the replay-based Bellman learning of DDPG/TD3/SAC.

*Which implementation mistakes silently change the algorithm?*

- Continuous actions: sum coordinate log-densities before the ratio. Keep actions fixed.
- Flatten values, advantages & log-probs to 1 scalar per sample. $(m,1)$ vs $(m,)$ can broadcast into an $(m,m)$ loss.
- GAE: use final observations & the 2 distinct boundary masks.
- Advantage normalization = practical rescaling, not required by the theorem. Compute critic targets before normalizing.

*Is every PPO the clipped variant?*

- ❌. The original paper also has a KL-penalized surrogate w/ an adaptive penalty.
- Value clipping, advantage normalization, gradient clipping & KL early stopping = extra choices. None makes ratio clipping a hard constraint.

*Where is PPO used for LLMs?*

- RLHF/RLVR fine-tuning; see [PPO for LLMs](../nlp/post/rl.md#ppo).
```

&nbsp;

## Deterministic continuous control

### Deterministic policy gradients
- **What**: $\nabla_\theta J$ = actor Jacobian × critic action-gradient. {cite:p}`silver2014deterministic`
- **Why**: Continuous actions → $\arg\max_a Q$ is an inner optimization at every step.
    - Stochastic score-function gradients also integrate over actions → extra variance.
    - Deterministic actor + differentiable critic → follow $\nabla_aQ$ directly.
- **How**:
    1. Critic gives how return changes w/ the action.
    2. Actor Jacobian gives how the action changes w/ the params.
    3. Chain both → move actions toward higher predicted return.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $f_\theta$: Deterministic actor.
    - $D_\theta f_\theta(s)$: Actor Jacobian; shape action-dim × param-dim.

Model:

$$
a=f_\theta(s)
$$

Training:

$$
\nabla_\theta J
=\frac1{1-\gamma}
\mathbb E_{s\sim d_\gamma^{f_\theta}}
\left[
(D_\theta f_\theta(s))^\top
\left.\nabla_aQ^{f_\theta}(s,a)\right|_{a=f_\theta(s)}
\right]
$$

- Needs: $f_\theta$ & $Q^{f_\theta}$ differentiable in the relevant arguments; integrability & regularity for the gradient/expectation exchange.
- Finite-episode form: expected sum over $\gamma^t$-weighted state visits.
```

```{tip} Derivation
:class: dropdown
1. Write $V^{f_\theta}(s)=Q^{f_\theta}(s,f_\theta(s))$.
2. Differentiate:
    - Direct action change → $(D_\theta f_\theta)^\top\nabla_aQ$.
    - Change of future actions → a recursive future-value term.
3. Unroll that recursion through transitions → discounted occupancy, as in the stochastic theorem.
4. Learned differentiable critic supplies $\nabla_aQ$ → ❌environment derivative needed.
```

```{attention} Q&A
:class: dropdown
*Is the replay-buffer actor gradient the exact gradient above?*

- ❌ in general. It replaces target-policy occupancy w/ replay states & true values w/ an approximate critic.
- Replay objective holds critic params fixed while changing curr actions → policy-improvement surrogate.
- Even $\mathbb E_{s\sim\nu}[V^{f_\theta}(s)]$ w/ a fixed behavior state distribution $\nu$ has future-policy derivative terms. Swapping occupancy for $\nu$ in the theorem ≠ its exact derivative.

*Why no action importance ratio?*

- The actor evaluates its own action at each replayed state → ❌average of score terms over behavior actions.
- Critic targets also evaluate a new target action after each stored transition.
- State/action coverage & critic accuracy still matter. ❌Action integral does not fix distribution mismatch.

*Why not plug deterministic actions into REINFORCE?*

- Point mass → ❌ordinary continuous density → ❌usable log-score.
- DPG uses a pathwise action derivative instead.
- Stochastic policy gradient → DPG as the policy variance → 0, only under extra regularity assumptions.

*How does a deterministic policy explore?*

- Use a separate behavior policy: actor action + exploration noise, or another exploratory rule.
- Environment randomness alone need not cover the action space.
```

&nbsp;

### DDPG
- **Name**: Deep Deterministic Policy Gradient {cite:p}`lillicrap2016continuous`
- **What**: Off-policy deep DPG w/ replay & target networks.
- **Why**: DQN-style replay learning needs $\max_a Q$ → ❌tractable for continuous actions.
    - Deterministic actor = learned approximate argmax.
- **How**:
    1. Act w/ noisy actor actions. Store transitions in replay.
    2. Regress the critic toward a target critic, evaluated at the target actor's next action.
    3. Improve the actor through the online critic.
    4. Move both target networks slowly toward their online networks.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\bar\theta$: Target-actor params.
- Hyperparams:
    - $\tau\in(0,1]$: Target interpolation weight; smaller → slower target motion.
- Misc:
    - $a'$: Target action at the next state.
    - $y_t$: Bellman target (detached).

Process:

$$
\begin{align*}
a'&=f_{\bar\theta}(s_{t+1})\\
y_t&=r_{t+1}+\gamma b_tQ_{\bar\phi}(s_{t+1},a')
\end{align*}
$$

Objective:

$$
\begin{align*}
\mathcal L_Q(\phi)
&=\frac12\mathbb E_{\mathcal R}
[(Q_\phi(s_t,a_t)-\operatorname{sg}(y_t))^2]\\
\mathcal L_\pi(\theta)
&=-\mathbb E_{s\sim\mathcal R}[Q_\phi(s,f_\theta(s))]
\end{align*}
$$

- Actor loss: freeze $\phi$, keep $\nabla_aQ_\phi$.

Process:

$$
\begin{align*}
\bar\theta&\leftarrow(1-\tau)\bar\theta+\tau\theta\\
\bar\phi&\leftarrow(1-\tau)\bar\phi+\tau\phi
\end{align*}
$$
```

````{important} Code
:class: dropdown
```python
from copy import deepcopy
import torch
import torch.nn as nn

class DDPGKernel:
    def __init__(self, gamma, tau):
        self.gamma, self.tau = gamma, tau

    @staticmethod
    def q(network, state, action):
        return network(torch.cat((state, action), dim=-1)).squeeze(-1)

    def __call__(self, actor, critic, target_actor, target_critic,
                 state, action, reward, next_state, terminated):
        ## target from TARGET actor & critic; no gradient through the bootstrap
        with torch.no_grad():
            next_q = self.q(target_critic, next_state, target_actor(next_state))
            target = reward + self.gamma * torch.where(terminated, 0.0, next_q)
        ## critic regresses on the REPLAYED action
        critic_loss = 0.5 * (self.q(critic, state, action) - target).square().mean()
        ## actor uses its CURRENT action; freeze critic params, keep d Q / d action
        critic.requires_grad_(False)
        actor_loss = -self.q(critic, state, actor(state)).mean()
        critic.requires_grad_(True)
        return actor_loss, critic_loss

    @torch.no_grad()
    def update_target(self, target, online):
        ## Polyak averaging: target <- (1 - tau) target + tau online
        for target_param, param in zip(target.parameters(), online.parameters()):
            target_param.lerp_(param, self.tau)

## Example
actor = nn.Sequential(nn.Linear(3, 1), nn.Tanh())
critic = nn.Linear(4, 1)
target_actor, target_critic = deepcopy(actor), deepcopy(critic)
kernel = DDPGKernel(0.9, 0.1)
actor_loss, critic_loss = kernel(actor, critic, target_actor, target_critic,
                                 torch.zeros(2, 3), torch.zeros(2, 1), torch.ones(2),
                                 torch.zeros(2, 3), torch.tensor([False, True]))
print(actor_loss.shape, critic_loss.shape)  ## torch.Size([]) torch.Size([])
```
````

```{attention} Q&A
:class: dropdown
*Which action trains which network?*

- Critic regression: replayed action $a_t$.
- Actor gradient: curr action $f_\theta(s_t)$, not the replayed action.
- Bootstrap: target actor's action at the next state.
- Exploration noise affects data collection only, not the basic actor objective.

*Why does off-policy 1-step critic learning need no behavior-action ratio?*

- Stored $(s,a)$ transition samples the same environment dynamics, however $a$ was chosen.
- Next target action is evaluated explicitly, not taken from the behavior trajectory.
- Replay decides which state–action errors get fitted. Poor coverage & function approximation still → bias & instability.

*What does the kernel omit?*

- Data collection, replay storage, optimizers & scheduling → update kernel, not a full agent.
- Use separate actor & critic networks (❌shared params). Feed detached replay tensors.
- Sequential update: optimize critic → recompute actor loss w/ the updated critic → optimize actor → update targets.
- Target interpolation covers params only. Running-statistic buffers need an explicit policy.

*Does "freeze the critic" mean `no_grad` around its actor evaluation?*

- ❌. `no_grad` erases the action derivative → actor stops learning.
- Freeze critic params; keep the graph actor action → critic output.

*Main failure mode?*

- Actor exploits critic overestimates → poor actions reinforce wrong targets.
- Replay + bootstrapping + nonlinear approximation → ❌general convergence guarantee.
- Target networks slow moving targets; they do not make the iteration a proven contraction in param space.
```

&nbsp;

#### TD3
- **Name**: Twin Delayed Deep Deterministic Policy Gradient {cite:p}`fujimoto2018addressing`
- **What**: DDPG + twin critics + delayed actor updates + target-action smoothing.
- **Why**: Critic overestimation ← approximation error + max-seeking actor.
    - Actor exploits the error peaks → bad actions → worse targets.
- **How**:
    1. Back up the lower of 2 target-critic estimates.
    2. Add clipped noise to target actions → ❌narrow value spikes.
    3. Update critics more often than the actor & target networks.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\sigma_{\mathrm{noise}}>0$: Target-noise std.
    - $c_{\mathrm{noise}}>0$: Target-noise clip magnitude; ≠ boundary indicator $c_t$.
- Misc:
    - $\widetilde a'$: Smoothed, bounded next action.
    - $a_{\mathrm{low}}$: Coordinatewise lower action bound.
    - $a_{\mathrm{high}}$: Coordinatewise upper action bound.
    - $j$: Critic idx.

Process:

$$
\begin{align*}
\widetilde\epsilon&=\operatorname{clip}
(\epsilon,-c_{\mathrm{noise}},c_{\mathrm{noise}}),
\qquad \epsilon\sim\mathcal N(0,\sigma_{\mathrm{noise}}^2I)\\
\widetilde a'
&=\operatorname{clip}\left(
f_{\bar\theta}(s_{t+1})+\widetilde\epsilon,
a_{\mathrm{low}},a_{\mathrm{high}}\right)\\
y_t&=r_{t+1}+\gamma b_t
\min_{j\in\{1,2\}}Q_{\bar\phi_j}(s_{t+1},\widetilde a')
\end{align*}
$$

Objective:

$$
\begin{align*}
\mathcal L_Q&=\frac12\sum_{j=1}^{2}
\mathbb E_{\mathcal R}[(Q_{\phi_j}(s_t,a_t)-\operatorname{sg}(y_t))^2]\\
\mathcal L_\pi&=-\mathbb E_{s\sim\mathcal R}[Q_{\phi_1}(s,f_\theta(s))]
\end{align*}
$$

- Original TD3 actor uses the first online critic, not the min.
- Update actor & all targets once per $k$ critic updates (paper: $k=2$).
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
        self.noise_std, self.noise_clip = noise_std, noise_clip
        self.low, self.high, self.policy_delay = low, high, policy_delay

    def __call__(self, update_index, actor, critics, target_actor,
                 target_critics, state, action, reward, next_state, terminated):
        with torch.no_grad():
            next_action = target_actor(next_state)
            ## target smoothing: clipped noise, then clip back into the action box
            noise = (self.noise_std * torch.randn_like(next_action)).clamp(
                -self.noise_clip, self.noise_clip
            )
            low = torch.as_tensor(self.low, device=next_action.device)
            high = torch.as_tensor(self.high, device=next_action.device)
            next_action = (next_action + noise).clamp(low, high)
            ## twin min -> pessimistic target vs overestimation
            next_q = torch.minimum(
                self.q(target_critics[0], next_state, next_action),
                self.q(target_critics[1], next_state, next_action)
            )
            target = reward + self.gamma * torch.where(terminated, 0.0, next_q)
        critic_loss = 0.5 * sum(
            (self.q(q, state, action) - target).square().mean() for q in critics
        )
        actor_loss = None
        ## delayed actor: only every policy_delay critic steps; uses critic 1, not the min
        if update_index % self.policy_delay == 0:
            critics[0].requires_grad_(False)
            actor_loss = -self.q(critics[0], state, actor(state)).mean()
            critics[0].requires_grad_(True)
        return actor_loss, critic_loss

## Example
actor = nn.Sequential(nn.Linear(3, 1), nn.Tanh())
critics = [nn.Linear(4, 1), nn.Linear(4, 1)]
kernel = TD3Kernel(0.9, 0.1, 0.2, 0.5, [-1.0], [1.0], 2)
actor_loss, critic_loss = kernel(
    1, actor, critics, deepcopy(actor), deepcopy(critics),
    torch.zeros(2, 3), torch.zeros(2, 1), torch.ones(2),
    torch.zeros(2, 3), torch.tensor([False, True])
)
print(actor_loss, critic_loss.shape)  ## None torch.Size([]) -> step 1 skips the actor
```
````

```{attention} Q&A
:class: dropdown
*What does each modification fix?*

- Twin min → overestimation in bootstrapped targets ⬇️.
- Delayed actor & targets → critics fit longer before the actor exploits them.
- Target smoothing → train on a neighborhood of next actions, not one spurious peak.
- None removes all function-approximation error.

*Is target noise the exploration noise?*

- ❌. Exploration noise changes actions executed in the environment.
- Target noise changes critic backups during training.
- Both can coexist; they serve different purposes.

*Does the min make the target unbiased?*

- ❌. It can underestimate. Correlated critic errors can still leave large overestimation.
- Bias-control heuristic, not an exact correction or uncertainty bound.

*How to use the kernel's delayed result?*

- Always train both critics.
- `actor_loss` not `None` → train the actor & interpolate the target actor + both target critics.
- To reproduce the original: ❌interpolate targets every critic step.
- Recompute the actor loss after a critic optimizer step. ❌Mutate params referenced by an outstanding graph.
```

&nbsp;

## Entropy-regularized control

### Maximum-entropy RL
- **What**: Return + per-step policy entropy bonus, incl. at future states.
- **Why**: Plain return maximization → policy collapses early onto one action.
    - Entropy term → keep all near-optimal actions in play → robustness & exploration ⬆️.
- **How**:
    1. Add entropy at every decision state, incl. future states.
    2. Evaluate policies w/ entropy-aware (soft) Bellman backups.
    3. Improve toward actions favored jointly by value & entropy.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $J_\beta$: Entropy-regularized discounted return.
    - $r(s,a)$: Expected immediate reward $\mathbb E[r_{t+1}\mid s_t=s,a_t=a]$.
    - $p(s'\mid s,a)$: Transition distribution.
    - $Q_\beta^\pi$: Soft action value; excludes the curr action's entropy, includes future entropy.
    - $V_\beta^\pi$: Soft state value; includes the curr decision's entropy.

Objective:

$$
J_\beta(\pi)
=\mathbb E_{\rho_0,\pi}\left[
\sum_{t=0}^{T-1}\gamma^t
\left(r_{t+1}-\beta\log\pi(a_t\mid s_t)\right)
\right]
$$

- $\mathbb E[-\beta\log\pi(a_t\mid s_t)\mid s_t]=\beta\mathcal H(\pi(\cdot\mid s_t))$ → entropy bonus.
- Soft Bellman equations:

    $$
    \begin{align*}
    Q_\beta^\pi(s,a)
    &=r(s,a)+\gamma\mathbb E_{s'\sim p(\cdot\mid s,a)}
    [V_\beta^\pi(s')]\\
    V_\beta^\pi(s)
    &=\mathbb E_{a\sim\pi}
    [Q_\beta^\pi(s,a)-\beta\log\pi(a\mid s)]
    \end{align*}
    $$
    - Terminal continuation value = 0. Sampled backups apply $b_t$.

- Optimal policy:

    $$
    \begin{align*}
    \pi^*(a\mid s)&=
    \exp\!\left(\frac{Q_\beta^*(s,a)-V_\beta^*(s)}{\beta}\right)\\
    V_\beta^*(s)&=\beta\log\int
    \exp(Q_\beta^*(s,a)/\beta)\,da
    \end{align*}
    $$
    - Needs $\beta>0$. Sum instead of integral for finite actions. Continuous normalizer must be finite.
    - Density is relative to the chosen action-space reference measure.
```

```{tip} Derivation
:class: dropdown
1. Fix a state & a soft action-value function. Optimize a normalized action distribution:

    $$
    \max_\pi\int\pi(a)\left[Q(a)-\beta\log\pi(a)\right]\,da,
    \qquad \int\pi(a)\,da=1
    $$

2. Add a multiplier for normalization. Set the derivative w.r.t. $\pi(a)$ to 0 → $\log\pi(a)=Q(a)/\beta+\text{const}$.
3. Normalize → exponential-of-value (Boltzmann) policy.
4. Substitute back → log-sum-exp value.
5. Finite actions, $\beta\to0^+$ → greedy max. Ties can keep a mixture.
```

````{important} Code
:class: dropdown
```python
import torch

class SoftPolicyImprovement:
    def __init__(self, temperature):
        self.temperature = temperature

    def __call__(self, action_values):
        logits = action_values / self.temperature
        ## logsumexp = numerically stable log-normalizer
        log_normalizer = logits.logsumexp(dim=-1, keepdim=True)
        probabilities = (logits - log_normalizer).exp()  ## Boltzmann policy
        soft_value = self.temperature * log_normalizer.squeeze(-1)  ## V = beta * logsumexp(Q / beta)
        return probabilities, soft_value

## Example
q = torch.tensor([[0.0, 1.0]])
probability, value = SoftPolicyImprovement(0.5)(q)
print(probability, value)  ## tensor([[0.1192, 0.8808]]) tensor([1.0635])
```
````

```{attention} Q&A
:class: dropdown
*Why is an actor entropy bonus alone not enough?*

- It rewards randomness only at sampled curr states. A reward-only critic ignores entropy available later.
- Soft values carry future entropy through the Bellman equation.
- Entropy regularization changes the task objective → ❌unbiased estimator of reward-only optimal control.

*Does maximum entropy solve exploration?*

- ❌. It promotes action diversity, not discovery of distant rewarding states.
- Sparse/deceptive rewards & poor state coverage can still defeat local policy optimization.

*Why can continuous entropy be negative?*

- A density can exceed 1 → differential entropy ≠ a discrete uncertainty count.
- Depends on coordinates & units.
- Unbounded action space → can grow w/o bound unless rewards or policy constraints stop expansion.

*What holds in the exact setting?*

- Finite discounted MDP, bounded rewards, fixed finite $\beta$ → soft optimality operator = $\gamma$-contraction in sup norm.
- Exact soft value/policy iteration converges under its assumptions.
- Shared neural params, approximate critics & incomplete state coverage → ❌exact tabular guarantees.

*How does it relate to natural policy gradient?*

- Both produce exponential-of-value updates per state.
- Soft improvement: $\pi\propto\exp(Q/\beta)$ (uniform prior). Tabular NPG: $\pi_{t+1}\propto\pi_t\exp(\cdot)$ (old policy as prior).
```

&nbsp;

### SAC
- **Name**: Soft Actor–Critic {cite:p}`haarnoja2018soft`
- **What**: Off-policy stochastic actor–critic approximating soft policy iteration.
- **Why**: On-policy methods → poor sample efficiency. DDPG-style off-policy methods → brittle & hyperparam-sensitive.
    - Max-entropy objective + replay + stochastic actor → stable & sample-efficient.
- **How**:
    1. Fit soft action values from replay.
    2. Move the actor toward the exponential-of-value distribution.
    3. Alternate evaluation & improvement; solve neither exactly.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\Pi$: Allowed policy family.
    - $Z(s)$: Normalizer of $\exp(Q_\beta^{\pi_{\mathrm{old}}}(s,a)/\beta)$ over actions.

Objective:

$$
\pi_{\mathrm{new}}(\cdot\mid s)
=\arg\min_{\pi\in\Pi}
D_{\mathrm{KL}}\left(
\pi(\cdot\mid s)\,\middle\|\,
\frac{\exp(Q_\beta^{\pi_{\mathrm{old}}}(s,\cdot)/\beta)}
{Z(s)}
\right)
$$

- Equivalent per-state minimization:

    $$
    \mathbb E_{a\sim\pi}
    [\beta\log\pi(a\mid s)-Q_\beta^{\pi_{\mathrm{old}}}(s,a)]
    $$
```

```{tip} Derivation
:class: dropdown
1. Expand the reverse KL:

    $$
    D_{\mathrm{KL}}(\pi\Vert\exp(Q/\beta)/Z)
    =\mathbb E_\pi[\log\pi-Q/\beta]+\log Z
    $$

2. Multiply by $\beta$. Drop $\beta\log Z$ ← constant w.r.t. the new actor.
3. Reparameterize the sampled action → differentiate the remaining expectation.
4. Approximate values & replay states → practical actor loss, not an exact start-state policy gradient.
```

```{attention} Q&A
:class: dropdown
*How can a stochastic actor learn off-policy w/o action importance weights?*

- Replay supplies states & transitions. The actor samples fresh actions from its curr distribution at those states.
- Critic targets sample the curr policy at next states; ❌reuse of behavior continuation actions.
- → ❌behavior-action correction for these conditional objectives. Replay-state mismatch & extrapolation error remain.

*What does the soft policy-iteration guarantee need?*

- Exact per-state policy improvement & evaluation, plus boundedness & coverage.
- Finite-action tabular result ≠ convergence proof for continuous neural SAC w/ shared params, sampled gradients & twin critics.
- Unimodal actor → ❌every desirable multimodal action distribution.

*Which SAC version to implement?*

- Original: separate state-value network $V$.
- Twin-Q variant below: continuation values from target critics & the curr actor → ❌separate $V$ network.
```

&nbsp;

#### Twin-Q SAC & automatic temperature
- **What**: SAC w/ twin target critics, ❌$V$ network & learned temperature. {cite:p}`haarnoja2018softapplications`
- **Why**: Fixed $\beta$ → best value depends on reward scale & shifts during training.
    - Separate $V$ network = extra model to fit.
- **How**:
    1. Sample replay transitions. Draw next actions from the curr actor.
    2. Fit both critics to one shared entropy-adjusted target.
    3. Improve the actor through fresh reparameterized actions & the lower online critic.
    4. Adjust temperature toward a target average entropy. Slowly update target critics.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\eta$: Log-temperature; unconstrained scalar, $\beta=e^\eta$.
- Hyperparams:
    - $\mathcal H_*$: Target minimum average entropy; can be negative for continuous actions.
    - $\tau\in(0,1]$: Target interpolation weight.
- Misc:
    - $j$: Critic idx.
    - $a'$: Next action.
    - $y_t$: Bellman target.
    - $f_\theta(s,\epsilon)$: Reparameterized stochastic actor; overrides the deterministic $f_\theta(s)$.

Process:

$$
\begin{align*}
a'&\sim\pi_\theta(\cdot\mid s_{t+1})\\
y_t&=r_{t+1}+\gamma b_t
\left[
\min_{j\in\{1,2\}}Q_{\bar\phi_j}(s_{t+1},a')
-\beta\log\pi_\theta(a'\mid s_{t+1})
\right]
\end{align*}
$$

- $Q_{\phi_j}$ approximates a soft action value; $\beta$ subscript dropped on learned critics.
- Next actions come from the curr actor → ❌target actor, ❌separate $V$ network.

Objective:

$$
\begin{align*}
\mathcal L_Q&=\frac12\sum_{j=1}^{2}
\mathbb E_{\mathcal R}
[(Q_{\phi_j}(s_t,a_t)-\operatorname{sg}(y_t))^2]\\
\mathcal L_\pi&=
\mathbb E_{\substack{s\sim\mathcal R\\\epsilon\sim\mathcal N(0,I)}}
\left[
\operatorname{sg}(\beta)\log\pi_\theta(f_\theta(s,\epsilon)\mid s)
-\min_jQ_{\phi_j}(s,f_\theta(s,\epsilon))
\right]
\end{align*}
$$

- Actor gradient: hold critic params fixed. Keep both the sampled-action path & the explicit log-density dependence on $\theta$.

$$
\begin{align*}
\mathbb E_{s\sim\mathcal R,a\sim\pi_\theta}[-\log\pi_\theta(a\mid s)]
&\geq \mathcal H_*\\
\mathcal L_\eta
&=-\mathbb E_{s\sim\mathcal R,a\sim\pi_\theta}
\left[
e^\eta\operatorname{sg}\!\left(\log\pi_\theta(a\mid s)+\mathcal H_*\right)
\right]
\end{align*}
$$

- Paper's continuous-control heuristic: $\mathcal H_*=-D$ for normalized $(-1,1)^D$ actions; not a universal optimum.
    - $D$: Action dimension.

Process:

$$
\bar\phi_j\leftarrow(1-\tau)\bar\phi_j+\tau\phi_j
$$

- Detach the full critic target.
- Freeze temperature during the actor update. Freeze actor samples during the temperature update.
```

```{tip} Derivation
:class: dropdown
1. At fixed sampled-state weighting: maximize reward s.t. an average entropy floor.
2. Add a nonnegative multiplier × (entropy − target). For a fixed actor, minimize the dual over temperature:

    $$
    \mathcal L_\beta
    =\beta\left(
    \mathbb E[-\log\pi_\theta(a\mid s)]-\mathcal H_*
    \right)
    $$

3. Parameterize $\beta=e^\eta>0$:

    $$
    \frac{\partial\mathcal L_\eta}{\partial\eta}
    =\beta\left(\mathbb E[-\log\pi_\theta]-\mathcal H_*\right)
    $$

4. Entropy < target → negative derivative → gradient descent raises $\eta$ & $\beta$.
5. Entropy > target → positive derivative → lower $\beta$. Constraint is on the average, not a per-state equality.
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
        temperature = log_temperature.exp()
        with torch.no_grad():
            ## next action from the CURRENT actor -> no target actor needed
            next_action, next_logp, _ = actor.sample(
                next_state, reparameterize=True
            )
            next_q = torch.minimum(
                self.q(target_critics[0], next_state, next_action),
                self.q(target_critics[1], next_state, next_action)
            )
            ## soft value = min Q - beta * log pi -> future entropy enters the target
            soft_next = next_q - temperature * next_logp
            target = reward + self.gamma * torch.where(terminated, 0.0, soft_next)
        critic_loss = 0.5 * sum(
            (self.q(q, state, action) - target).square().mean() for q in critics
        )
        ## fresh reparameterized action: gradient flows actor -> action -> critic
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
        ## exact log-parameterized dual loss; logp detached -> no actor gradient
        temperature_loss = -(
            temperature * (logp.detach() + self.target_entropy)
        ).mean()
        return actor_loss, critic_loss, temperature_loss

## Example
actor = SquashedGaussian(3, 1, [-1.0], [1.0])
critics = [nn.Linear(4, 1), nn.Linear(4, 1)]
log_temperature = torch.tensor(0.0, requires_grad=True)
losses = SACKernel(0.9, 0.1, -1.0)(
    actor, critics, deepcopy(critics), log_temperature,
    torch.zeros(2, 3), torch.zeros(2, 1), torch.ones(2),
    torch.zeros(2, 3), torch.tensor([False, True])
)
print([loss.shape for loss in losses])  ## 3x torch.Size([]): actor, critic, temperature
```
````

```{attention} Q&A
:class: dropdown
*Target-entropy sign convention?*

- Store $\mathcal H_*$ as an entropy, not as a target log-prob.
- Loss uses $\log\pi+\mathcal H_*$ = −(estimated entropy − target).
- e.g., curr entropy $-2$, target $-1$ → entropy too low → raise temperature.
- Negative continuous target entropy is not an error. Compare in the same action coordinates.

*Why do some implementations optimize log-temperature linearly?*

- Common surrogate: replace $e^\eta$ in $\mathcal L_\eta$ w/ $\eta$.
- Same gradient sign & interior zero, but ❌positive factor $\beta$ → different optimization scaling, same target sign.
- Code uses the exact exponential dual loss. The linear surrogate is not algebraically identical.

*What if the entropy target is impossible?*

- Bounded action domain → entropy has an upper bound. The policy family can lower the achievable max.
- Infeasible floor → temperature keeps rising → ❌dual equilibrium.
- Auto-tuning ≠ guarantee of exploration quality or reward-scale invariance.

*Which params get which gradients?*

- Critic loss → online critics only; full target detached.
- Actor loss → actor only; online critics keep action derivatives but get ❌param gradients.
- Temperature loss → log-temperature only.
- Target critics → interpolation only; ❌optimizer steps.
- Sequential optimizer steps → recompute affected losses; ❌reuse graphs across in-place param updates.

*Why no separate $V$ or target actor?*

- Sampled next action & log-density estimate the soft state value from target critics directly.
- Curr actor generates that action inside a detached target computation.
- Both target critics start as exact copies of the online critics, then interpolate slowly.

*Is SAC's deterministic evaluation action its mean action?*

- Common choice: squash & rescale the latent mean.
- ≠ the mean of the squashed distribution in general; not necessarily its mode.
- ❌Sampling → different evaluated policy. The entropy objective does not guarantee that this deterministic choice maximizes reward.
```

&nbsp;

## Comparison
- **What**: Policy-optimization methods side by side.

```{dropdown} Table: Policy optimization methods
| Method | Actor | Training data | Critic/advantage | Main update control | Main cost or failure |
|:--|:--|:--|:--|:--|:--|
| REINFORCE | Stochastic | Complete on-policy episodes | Monte Carlo return; optional baseline | Step size | High return variance |
| A2C/A3C | Stochastic | Fresh parallel rollouts | Bootstrapped state values | Step size; optional entropy | Critic bias; A3C staleness |
| NPG | Stochastic | Fresh on-policy rollouts | Usually state value + GAE | Fisher preconditioning | Fisher estimation & solve |
| TRPO | Stochastic | Fresh old-policy rollouts | Usually state value + GAE | Approximate KL constraint | Second-order solve; sampling approximations |
| PPO | Stochastic | Fresh rollouts, limited epoch reuse | Usually state value + GAE | Clipped/penalized surrogate | No hard trust region; stale-data bias |
| DDPG | Deterministic | Replay | One action-value critic | Target actor & critic | Critic exploitation; weak action coverage |
| TD3 | Deterministic | Replay | Twin action-value critics | Pessimistic targets, delay, smoothing | Underestimation; replay extrapolation |
| SAC | Stochastic | Replay | Twin soft action-value critics | Entropy objective & target critics | Entropy/reward scaling; replay extrapolation |

- Categorical actors → discrete actions. Gaussian actors → continuous actions.
- DDPG/TD3 need differentiable continuous actions. The SAC code above is continuous-action SAC.
```

```{attention} Q&A
:class: dropdown
*How to choose PPO, TD3 or SAC?*

- Fast parallel simulation, discrete actions or a simple stochastic-policy pipeline → PPO.
- Continuous control w/ expensive interaction & useful replay coverage → TD3/SAC reuse data.
- TD3: deterministic control + injected exploration noise. SAC: learned stochastic, entropy-regularized policy.
- ❌Universal ranking. Interaction cost, compute, reward structure, action support & data coverage decide.

*Does off-policy mean offline-safe?*

- ❌. Online replay methods keep collecting coverage near their changing policies.
- Fixed offline data may not cover actor-proposed actions → bootstrapped critics extrapolate badly → actors exploit the errors.
- Twin minima ≠ sufficient offline support constraints.
- Uncorrected replayed multi-step returns & GAE add behavior-continuation mismatch.
- States sampled uniformly from replay ≠ $d_\gamma^\pi$ → SAC's replay-weighted improvement objective ≠ exact discounted start-distribution gradient.
```

&nbsp;
