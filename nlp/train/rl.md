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
# RL for LLMs
How RL is wired onto a pretrained LM, and how the objective evolved from REINFORCE → PPO → GRPO → its descendants.

Assumes RL and LLMs are each already understood in isolation.

```{dropdown} Table: Shared Notations
| Notation | Meaning |
|:--|:--|
| $x\sim\mathcal{D}$ | Prompt (input token seq) |
| $y$ | Response (output token seq) |
| $y_t$ | $t$-th response token |
| $\|y\|$ | Response length (#tokens) |
| $\mathcal{V}$ | Vocabulary |
| $\pi_\theta$ | Curr policy (the LM being trained) |
| $\pi_{\theta_\text{old}}$ | Rollout policy (generated the curr batch) |
| $\pi_\text{ref}$ | Reference policy (frozen, usually post-SFT) |
| $r(x,y)$ | Scalar reward for a full response |
| $\hat{A}$ | Advantage estimate |
| $V_\psi$ | Value model (critic), parameterized by $\psi$ |
| $G$ | Group size (#responses sampled per prompt) |
| $i$ | Response idx within a group |
| $t$ | Token idx within a response |
| $\epsilon$ | Clip range |
| $\beta$ | KL coeff |
| $\text{sg}[\cdot]$ | Stop-gradient |

$G$ overrides the global $G_t$ (discounted return) — this page never discounts.
```

&nbsp;

## Setup
### MDP Framing
- **What**: Prompt → state, response → action, LM → policy, grader → reward.
- **Why**: RL needs an env; an LM has none.
    - LM = conditional next-token distribution, ❌agent.
    - No transition physics, no per-step reward, no episode past one response → all of it must be manufactured.
- **How**:
    1. **Token-MDP**: $s_t=(x,y_{<t})$, $a_t=y_t$, transition = deterministic concat.
    2. Grader scores the **full** response only → $r_t=0$ for $t<|y|$.
    3. → Collapse to a **contextual bandit**: 1 prompt, 1 action (= whole response), 1 reward.
    4. $\gamma=1$ ← Discounting a fixed-length response has no meaning.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $y_{<t}$: Response tokens before position $t$.
- Misc:
    - $J(\theta)$: Expected reward (the RL objective).

Model (token-MDP):

$$
s_t=(x,y_{<t}),\quad a_t=y_t\in\mathcal{V},\quad p(s_{t+1}|s_t,a_t)=\mathbb{1}[s_{t+1}=(x,y_{\leq t})]
$$

Model (bandit collapse):

$$
\pi_\theta(y|x)=\prod_{t=1}^{|y|}\pi_\theta(y_t|x,y_{<t})
$$

Objective:

$$
J(\theta)=\mathbb{E}_{x\sim\mathcal{D},\ y\sim\pi_\theta(\cdot|x)}[r(x,y)]
$$
```

````{dropdown} Table: RL $\Leftrightarrow$ LLM
| RL | LLM | Note |
|:--|:--|:--|
| Agent | LM | |
| Policy $\pi_\theta$ | LM weights | Initialized from pretrain (+SFT), ❌random |
| State $s_t$ | Prompt + tokens so far | |
| Action $a_t$ | Next token | $\|\mathcal{A}\|=\|\mathcal{V}\|\sim10^5$ |
| Transition $p(s_{t+1}\|s_t,a_t)$ | String concat | Deterministic, known |
| Reward $r_t$ | 0 until EOS, then $r(x,y)$ | Sparse, terminal-only |
| Episode | One response | $T=\|y\|$ |
| Env | Grader (RM / verifier) | Stateless, no dynamics |
````

```{attention} Q&A
:class: dropdown
*Why does the token-MDP collapse to a bandit?*
- Transitions are deterministic & known → ❌dynamics to learn.
- Rewards are terminal-only → $V(s_t)$ is just "expected final score from here", ❌TD signal to bootstrap.
- → Only the sequence-level return is real; the per-step structure carries no extra information.

*Then why do most implementations still compute per-token quantities?*
- Grads flow through $\log\pi_\theta(y_t|\cdot)$ per token regardless.
- The sequence-level advantage is **broadcast** to every token → same scalar, $|y|$ copies.
- → "Token-level" in these algorithms means loss **aggregation**, ❌token-level credit.

*What makes this setting easy compared to classic RL?*
- Strong pretrained prior → ✅Valid rollouts from step 0, ❌random exploration phase.
- Known deterministic dynamics → ❌model error.
- Cheap, parallel, resettable env → arbitrarily many rollouts per prompt.

*What makes it hard?*
- $|\mathcal{V}|^{|y|}$ effective action space.
- Credit assignment: 1 scalar for $10^3$–$10^5$ tokens.
- Reward is a **proxy** → [reward hacking](../rh.md).
- Rollouts dominate wall-clock → strong pressure to reuse each batch → off-policy drift.

*Why is $\gamma=1$?*
- $\gamma<1$ encodes "sooner is better", which is meaningless within one response.
- Discounting would arbitrarily down-weight later tokens of a correct answer.

*Why not just give per-token rewards?*
- Requires a **process reward model** graded per step → expensive to label, easy to hack.
- Outcome-only rewards remain the default ← cheap, verifiable, empirically sufficient.
```

&nbsp;

### Reward
- **What**: Scalar grade of a full response.
- **Why**: RL optimizes a number; "good answer" is not a number.
    - The reward source **is** the design decision — the optimizer only maximizes what it is handed.
- **How**: 3 sources, ordered by how much they survive optimization pressure:
    1. **RLVR**: Programmatic checker → exact, only for verifiable domains.
    2. **RLHF**: Learned RM from human preferences → covers open-ended tasks, hackable.
    3. **RLAIF**: Learned RM from LM-generated preferences → scalable, inherits the judge's biases.
```{attention} Q&A
:class: dropdown
*Why is a learned RM hackable but a verifier is not?*
- RM is a **finite-capacity approximation** of preference → has off-distribution holes.
- Policy training pushes exactly into those holes ← that's where reward is highest.
- Verifier is the ground-truth function itself → no approximation gap to exploit.
- ⚠️ Verifiers are still **spec**-hackable (special-casing tests, right answer w/ fabricated reasoning).

*Why can't we use RLVR everywhere?*
- Needs a cheap, unambiguous checker → math/code/structured output only.
- ❌Open-ended writing, safety, tone, helpfulness.

*Why is reward kept bounded/normalized in practice?*
- Unbounded reward → unbounded advantage → grad explosion.
- Verifiers naturally give $\{0,1\}$; RM scores are typically whitened per batch.
```

&nbsp;

#### RLHF
- **Name**: Reinforcement Learning from Human Feedback {cite:p}`christiano2017deep`
- **What**: Reward = learned RM fit on human pairwise preferences.
- **Why**: Most desirable behavior has no gold answer & no checker.
    - Humans can't *write* the best response, but they can *compare* two.
- **How**:
    1. Collect $(x,y_w,y_l)$ pairs from human labelers.
    2. Fit RM $r_\phi$ under **Bradley-Terry** → scalar score per response.
    3. RL-optimize the policy against $r_\phi$, KL-anchored to $\pi_\text{ref}$.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $y_w$: Preferred (won) response.
    - $y_l$: Dispreferred (lost) response.
- Params:
    - $r_\phi$: RM, parameterized by $\phi$.
- Misc:
    - $\sigma$: Sigmoid.

Model (Bradley-Terry):

$$
p(y_w\succ y_l|x)=\frac{\exp r(x,y_w)}{\exp r(x,y_w)+\exp r(x,y_l)}=\sigma\left(r(x,y_w)-r(x,y_l)\right)
$$

Training:

$$
\mathcal{L}_\text{RM}(\phi)=-\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\left[\log\sigma\left(r_\phi(x,y_w)-r_\phi(x,y_l)\right)\right]
$$
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BradleyTerryRM(nn.Module):
    def __init__(self, backbone, hidden):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(hidden, 1, bias=False)  ## scalar score, no bias

    def forward(self, tokens, last_idx):
        h = self.backbone(tokens)                     ## (B, T, hidden)
        ## score the LAST token only -> one scalar per full response
        h_last = h[torch.arange(h.size(0)), last_idx]
        return self.head(h_last).squeeze(-1)          ## (B,)

def bt_loss(score_w, score_l):
    return -F.logsigmoid(score_w - score_l).mean()

## Example
print(bt_loss(torch.tensor([2.0, 0.5]), torch.tensor([1.0, 1.5])))  ## tensor(0.8133)
```
````

```{attention} Q&A
:class: dropdown
*Who introduced what?*
- Preference RM + RL loop, on control tasks → Christiano et al. 2017.
- Ported to LMs → Ziegler et al. {cite:p}`ziegler2019finetuning`
- Scaled into the standard instruction-following recipe → InstructGPT {cite:p}`ouyang2022training`

*Why is the RM score's absolute offset meaningless?*
- BT loss depends only on the **difference** $r(x,y_w)-r(x,y_l)$.
- → $r_\phi+c$ is an equally good RM ← the shift cancels; the scale does **not** (it sharpens the sigmoid).
- → Never compare raw scores across RMs. Batch whitening is the usual implementation fix, ❌a consequence of BT itself.

*Why does the RM go stale during training?*
- RM was fit on samples from $\pi_\text{ref}$.
- Policy drifts → its samples leave the RM's training distribution → scores become unreliable.
- → The KL anchor is partly an **RM validity constraint**, not just a behavior constraint.

*Why pairwise instead of direct scoring?*
- Absolute ratings are inconsistent across labelers and drift within a session.
- Comparisons are lower-variance and need no shared scale.

*Cons?*
- Expensive & slow ← human labeling.
- Noisy ← inter-annotator disagreement.
- The classic failure mode is RM **overoptimization**: true quality peaks, then falls, while RM score keeps rising.
```

&nbsp;

#### RLAIF
- **Name**: Reinforcement Learning from AI Feedback {cite:p}`bai2022constitutional`
- **What**: RLHF, but the preference labeler is an LM following a written spec.
- **Why**: Human labels don't scale.
    - Cost & latency cap dataset size.
    - Humans are least reliable exactly where it matters most (long, technical, adversarial responses).
- **How**:
    1. Write an explicit spec (a "constitution") of principles.
    2. LM critiques & revises its own responses against a sampled principle → SFT data.
    3. LM picks the preferred response of a pair given a principle → preference data.
    4. Train RM on those labels → standard RLHF from there.

```{attention} Q&A
:class: dropdown
*Why does asking the same model to judge itself work at all?*
- **Evaluation is easier than generation** — recognizing a violation needs less capability than avoiding it.
- The spec sits in-context at judging time but not at generation time → strictly more information.

*Cons?*
- Inherits & amplifies judge biases (position bias, verbosity bias, self-preference).
- Blind spots are **correlated** with the policy's own ← same model family.
- → Mixed with human data in practice, ❌full replacement.

*What is actually new vs RLHF?*
- Only the label **source**. RM training, policy optimization, and KL anchoring are unchanged.
```

&nbsp;

#### RLVR
- **Name**: Reinforcement Learning with Verifiable Rewards {cite:p}`lambert2024tulu`
- **What**: Reward = output of a deterministic checker.
- **Why**: Learned RMs get hacked, and reasoning has a ground truth.
    - A preference RM gives long CoT no useful signal ← "which proof looks nicer" ≠ "which proof is right".
- **How**:
    1. Dataset of prompts with **checkable** answers (final answer, unit tests).
    2. Sample response → extract answer → run checker.
    3. $r=1$ if pass else $0$ (+ optional format reward).
    4. Optimize with any policy-gradient method.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $a^*$: Ground-truth answer.
    - $\text{ext}(y)$: Answer extracted from the response.
    - $\lambda$: Format reward weight.

Model:

$$
r(x,y)=\mathbb{1}[\text{ext}(y)=a^*]+\lambda\cdot\mathbb{1}[y\text{ matches the required format}]
$$
```

```{attention} Q&A
:class: dropdown
*Why did RLVR unlock reasoning models when RLHF didn't?*
- RLHF's ceiling is the RM's judgment; RLVR's ceiling is the task's ground truth.
- → Safe to optimize far harder & far longer without the reward degrading.
- DeepSeek-R1-Zero: pure RLVR on a **base** model, ❌SFT warmup → long CoT & self-verification emerge unprompted {cite:p}`deepseekai2025deepseek`

*Why add a format reward?*
- Makes extraction reliable → the accuracy signal stops being corrupted by parse failures.
- Cheap to satisfy → the policy locks it in early and moves on.

*Cons?*
- Binary reward is **sparse** → 0 gradient when a group is all-pass or all-fail.
- Math/code/structured output only.
- Grades the final answer, ❌the reasoning → CoT may be post-hoc ([representation-level exploitation](../rh.md#representation-level-exploitation)).
- R1-Zero's raw output was poorly readable & language-mixed → R1 added an SFT cold start.

*What breaks if the verifier is imperfect?*
- False negatives (equivalent-but-differently-formatted answers) punish correct reasoning → the policy learns the *format*, not the math.
- → Verifier quality upper-bounds everything downstream.
```

&nbsp;

### KL Penalty
- **What**: Divergence term anchoring $\pi_\theta$ to $\pi_\text{ref}$.
- **Why**: Reward is a proxy; the unconstrained argmax of a proxy is degenerate.
    - Unconstrained → collapse onto whatever the RM scores highest, gibberish included.
    - Drift also invalidates the RM ← it only ever saw $\pi_\text{ref}$'s distribution.
    - Catastrophic forgetting of general capability.
- **How**:
    1. Estimate per-token KL between $\pi_\theta$ and $\pi_\text{ref}$ on sampled tokens.
    2. Subtract $\beta\cdot\text{KL}$ from the reward, or add it to the loss.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\rho=\frac{\pi_\text{ref}(y_t|x,y_{<t})}{\pi_\theta(y_t|x,y_{<t})}$: Reference/policy prob ratio at one sampled token.

Objective (reward-level):

$$
\tilde{r}(x,y)=r_\phi(x,y)-\beta\log\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}
$$

Estimators of $\text{KL}[\pi_\theta\|\pi_\text{ref}]$ from a single sample $y_t\sim\pi_\theta$:

$$\begin{align*}
k_1&=-\log\rho && (\text{on-policy unbiased, high variance, can be}<0) \\
k_2&=\tfrac{1}{2}(\log\rho)^2 && (\text{biased, low variance, always}\geq0) \\
k_3&=\rho-\log\rho-1 && (\text{on-policy unbiased, always}\geq0)
\end{align*}$$
- Unbiasedness of $k_1,k_3$ requires $y_t\sim\pi_\theta$; $k_3$'s non-negativity is unconditional.
```

```{tip} Derivation
:class: dropdown
*Why is $k_3$ both unbiased and non-negative?*

1. The added term has zero mean under $\pi_\theta$:

    $$
    \mathbb{E}_{y_t\sim\pi_\theta}[\rho-1]=\sum_{y_t}\pi_\theta\cdot\frac{\pi_\text{ref}}{\pi_\theta}-1=\sum_{y_t}\pi_\text{ref}-1=0
    $$

2. → $\mathbb{E}[k_3]=\mathbb{E}[-\log\rho]+\mathbb{E}[\rho-1]=\text{KL}[\pi_\theta\|\pi_\text{ref}]$.

3. Non-negativity ← the tangent-line bound on the concave $\log$:

    $$
    \log\rho\leq\rho-1\ \Longrightarrow\ \rho-\log\rho-1\geq0\quad(\text{equality iff }\rho=1)
    $$

4. → $k_3$ is $k_1$ plus a zero-mean **control variate**: identical mean, and $(\rho-1)$ cancels most of $-\log\rho$'s fluctuation.
```

````{important} Code
:class: dropdown
```python
import torch

def kl_estimators(logp, logp_ref):
    ## logp, logp_ref: (B, T) log-probs of the SAMPLED tokens under each policy
    log_rho = logp_ref - logp
    k1 = -log_rho                        ## on-policy unbiased, can go negative
    k2 = 0.5 * log_rho.pow(2)            ## biased, always >= 0
    k3 = log_rho.exp() - log_rho - 1     ## on-policy unbiased AND always >= 0
    return k1, k2, k3

## Example
logp, logp_ref = torch.tensor([[-1.0, -3.0]]), torch.tensor([[-1.5, -2.0]])
for name, v in zip("k1 k2 k3".split(), kl_estimators(logp, logp_ref)):
    print(name, v.mean().item())  ## k1 -0.2500  k2 0.3125  k3 0.4124
```
````

```{attention} Q&A
:class: dropdown
*Why is $k_1$ a problem in practice?*
- Per-sample $-\log\rho<0$ whenever $\pi_\theta(y_t)>\pi_\text{ref}(y_t)$.
- → Minibatch "KL" goes negative even though true KL $\geq0$ → noisy, sign-flipping penalty.

*Reward-level vs loss-level KL?*
- Reward-level: folded into $\tilde{r}$ → flows through the advantage estimator → the critic must model it.
- Loss-level: separate additive term → the advantage stays pure. GRPO does this.

*When should you turn it off?*
- RLVR w/ a trustworthy verifier → the anchor mostly just slows learning.
- DAPO & CISPO drop it entirely. (R1-Zero's stated GRPO objective still carries it.)
- Keep it whenever the reward is a **learned** RM.

*What replaces it once removed?*
- The IS-ratio clip becomes the only trust region.
- But the clip constrains $\pi_\theta$ vs $\pi_{\theta_\text{old}}$ (one batch), ❌vs $\pi_\text{ref}$ (all of training).
- → No bound on cumulative drift. Tolerable only because a verifier can't be hacked by drift alone.

*Which direction is it?*
- Reverse KL, $\text{KL}[\pi_\theta\|\pi_\text{ref}]$ ← we can only sample from $\pi_\theta$.
- → Mode-seeking: the policy is free to drop modes of $\pi_\text{ref}$, but heavily penalized for putting mass where $\pi_\text{ref}$ has none.
- → Partly explains RLHF's well-known diversity loss.
```

&nbsp;

### Training Loop
- **What**: Generate → grade → estimate advantage → update.
- **Why**: The data does not exist until the curr policy makes it.
    - ❌Fixed dataset → every step needs fresh rollouts from a model that just changed.
- **How**: Per iteration,
    1. **Rollout**: Sample prompts; generate $G$ responses each from $\pi_{\theta_\text{old}}$.
    2. **Grade**: Score each response → $r(x,y_i)$.
    3. **Estimate**: Compute $\hat{A}$ (critic, or group statistics).
    4. **Update**: $\mu$ gradient steps over the batch.
    5. $\pi_{\theta_\text{old}}\leftarrow\pi_\theta$; sync weights into the inference engine.

```{attention} Q&A
:class: dropdown
*Why is $\pi_{\theta_\text{old}}\neq\pi_\theta$?*
- Generation dominates cost → each batch is reused for $\mu>1$ updates.
- After the 1st update $\pi_\theta$ has moved, but the data still came from $\pi_{\theta_\text{old}}$.
- → **Semi-on-policy**: exactly on-policy at the 1st step only, then increasingly off-policy.
- → This mismatch is precisely what the IS ratio corrects & the clip bounds.

*What is $\mu$ in practice?*
- $\mu=1$ → fully on-policy, ratio $\equiv1$, clipping is a no-op, ⬆️cost.
- $\mu>1$ → cheaper, but ⬆️drift → trust-region design starts to dominate stability.

*What sits in memory?*

| Model | PPO | GRPO-family |
|:--|:--|:--|
| Policy $\pi_\theta$ (+ optimizer states) | ✅ | ✅ |
| Rollout policy $\pi_{\theta_\text{old}}$ | ✅ | ✅ |
| Reference $\pi_\text{ref}$ | ✅ | ✅/❌ (dropped w/ KL) |
| Critic $V_\psi$ (+ optimizer states) | ✅ | ❌ |
| RM $r_\phi$ | ✅ | ❌ w/ RLVR |

*Why is the loop throughput-bound, not FLOP-bound?*
- Generation is autoregressive & memory-bandwidth-bound; training is one dense parallel forward-backward.
- Long CoT (10k+ tokens) makes it worse → rollout can dominate wall-clock.
- → Async/pipelined rollout, and harder batch reuse, are the main levers.

*Why does one slow rollout stall the whole step?*
- The batch isn't gradable until its **longest** response finishes.
- → Straggler-bound; a length cap is a throughput knob, not just a memory knob.
```

&nbsp;

## Algorithms
### REINFORCE
- **What**: Log-prob gradient weighted by reward.
- **Why**: $r$ is not differentiable w.r.t. $\theta$.
    - Grader is a black box (string match, sandbox, RM) → ❌backprop through it.
    - Sampling $y\sim\pi_\theta$ is itself non-differentiable.
    - → Need $\nabla_\theta$ of an expectation whose **distribution** depends on $\theta$, not whose integrand does.
- **How**:
    1. Sample $y\sim\pi_\theta(\cdot|x)$.
    2. Grade → $r(x,y)$.
    3. Ascend $\nabla_\theta\log\pi_\theta(y|x)$, scaled by $r$.
    4. Subtract a baseline $b(x)$ → same expected gradient, ⬇️variance.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $J(\theta)$: Expected reward.
    - $b(x)$: Baseline, any function of $x$ alone.

Objective:

$$
J(\theta)=\mathbb{E}_{x\sim\mathcal{D},\ y\sim\pi_\theta(\cdot|x)}[r(x,y)]
$$

Gradient:

$$
\nabla_\theta J(\theta)=\mathbb{E}_{x\sim\mathcal{D},\ y\sim\pi_\theta(\cdot|x)}\left[\left(r(x,y)-b(x)\right)\nabla_\theta\log\pi_\theta(y|x)\right]
$$

Per-token expansion:

$$
\nabla_\theta\log\pi_\theta(y|x)=\sum_{t=1}^{|y|}\nabla_\theta\log\pi_\theta(y_t|x,y_{<t})
$$
```

```{tip} Derivation
:class: dropdown
*Where does the log come from, and why is any $b(x)$ free?*

1. Differentiate the objective; only the sampling distribution depends on $\theta$:

    $$
    \nabla_\theta J=\nabla_\theta\sum_y\pi_\theta(y|x)r(x,y)=\sum_y r(x,y)\nabla_\theta\pi_\theta(y|x)
    $$

2. **Log-derivative trick** turns it back into an expectation:

    $$
    \nabla_\theta\pi_\theta=\pi_\theta\frac{\nabla_\theta\pi_\theta}{\pi_\theta}=\pi_\theta\nabla_\theta\log\pi_\theta
    $$

3. → $\nabla_\theta J=\mathbb{E}_{y\sim\pi_\theta}\left[r(x,y)\nabla_\theta\log\pi_\theta(y|x)\right]$.

4. A $y$-independent baseline contributes nothing:

    $$
    \mathbb{E}_{y\sim\pi_\theta}\left[b(x)\nabla_\theta\log\pi_\theta(y|x)\right]=b(x)\sum_y\nabla_\theta\pi_\theta(y|x)=b(x)\nabla_\theta 1=0
    $$

5. → Subtracting $b(x)$ leaves the mean untouched & changes only the variance.
```

````{important} Code
:class: dropdown
```python
import torch

def token_logp(logits, actions):
    ## logits: (B, T, V) from the policy; actions: (B, T) the SAMPLED token ids
    logp = torch.log_softmax(logits.float(), dim=-1)
    return logp.gather(-1, actions.unsqueeze(-1)).squeeze(-1)  ## (B, T)

def reinforce_loss(logits, actions, rewards, baseline=0.0):
    ## rewards: (B,) one scalar per full response
    ## baseline MUST NOT depend on y_i -- a running/EMA scalar, never this batch's own mean
    adv = (rewards - baseline).detach()                        ## constant wrt theta
    seq_logp = token_logp(logits, actions).sum(-1)             ## (B,) = log pi(y|x)
    ## ascend E[A * log pi]  ==  descend -(A * log pi)
    return -(adv * seq_logp).mean()

## Example
logits, actions = torch.randn(4, 3, 5), torch.randint(0, 5, (4, 3))
rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
ema_baseline = 0.4                                             ## from previous batches
print(reinforce_loss(logits, actions, rewards, ema_baseline).item())
```
````

```{attention} Q&A
:class: dropdown
*How does this differ from SFT?*
- SFT: $-\nabla_\theta\log\pi_\theta(y^*|x)$ on a **fixed, external** $y^*$.
- REINFORCE: $-\hat{A}\nabla_\theta\log\pi_\theta(y|x)$ on the model's **own sample** $y$.
- → REINFORCE = SFT on self-generated data, signed & scaled by the advantage.
- $\hat{A}>0$ → imitate it; $\hat{A}<0$ → unlearn it. SFT has no second case.

*Why is variance the central problem?*
- 1 scalar supervises a sample drawn from $|\mathcal{V}|^{|y|}$ possibilities.
- Reward offset is arbitrary → $r\in[99,101]$ makes every sample look good, so every sample gets pushed up.
- → Baseline design **is** the algorithm; everything after this is a better $b$ and a better trust region.

*Does $b$ have to be the value function?*
- ❌. Any $b(x)$ independent of $y$ keeps the estimator unbiased.
- $b(x)=V(x)=\mathbb{E}[r|x]$ is the standard choice & usually works well, but it is **not** the variance-minimizing one. That is $b^*(x)=\frac{\mathbb{E}\left[r\|\nabla_\theta\log\pi_\theta(y|x)\|^2\right]}{\mathbb{E}\left[\|\nabla_\theta\log\pi_\theta(y|x)\|^2\right]}$, a gradient-norm-weighted reward.
- Group mean & leave-one-out estimate $V(x)$ from samples → same role, no critic.

*Why is it strictly on-policy?*
- The expectation is taken under $\pi_\theta$ itself.
- Reuse the batch after one update → $y$ was drawn from a stale distribution → biased gradient.
- → Fixed by importance sampling (PPO onward).
```

&nbsp;

#### RLOO
- **Name**: REINFORCE Leave-One-Out {cite:p}`kool2019buy`
- **What**: REINFORCE w/ baseline = mean reward of the **other** $K-1$ samples.
- **Why**: The critic is the most expensive component of PPO and this setting barely uses it.
    - Critic ≈ policy-sized → ~2× memory, its own optimizer state, its own loss to balance.
    - Terminal-only reward → the critic only ever estimates "expected final score for this prompt", which $K$ samples estimate directly.
- **How**:
    1. Sample $K$ responses per prompt.
    2. Baseline for $y_i$ = mean reward over $j\neq i$.
    3. Plain REINFORCE update — ❌ratio, ❌clip.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $K$: #responses sampled per prompt.
- Misc:
    - $r_i=r(x,y_i)$: Reward of the $i$-th response.
    - $\bar{r}=\frac{1}{K}\sum_{j=1}^{K}r_j$: Group mean reward.

Advantage:

$$
\hat{A}_i=r_i-\frac{1}{K-1}\sum_{j\neq i}r_j=\frac{K}{K-1}\left(r_i-\bar{r}\right)
$$

Objective:

$$
J_\text{RLOO}(\theta)=\mathbb{E}_{x\sim\mathcal{D}}\left[\frac{1}{K}\sum_{i=1}^{K}\hat{A}_i\log\pi_\theta(y_i|x)\right]
$$
```

````{important} Code
:class: dropdown
```python
import torch

def rloo_advantage(rewards):
    ## rewards: (P, K) -> P prompts x K responses
    K = rewards.size(1)
    ## exclude self from own baseline -> b is independent of y_i
    loo_mean = (rewards.sum(1, keepdim=True) - rewards) / (K - 1)
    return rewards - loo_mean

## Example
r = torch.tensor([[1.0, 0.0, 0.0, 1.0]])
print(rloo_advantage(r))  ## tensor([[ 0.6667, -0.6667, -0.6667,  0.6667]])
```
````

```{attention} Q&A
:class: dropdown
*Leave-one-out vs just subtracting the group mean?*
- Including $r_i$ in its own baseline makes $b$ depend on $y_i$ → the unbiasedness proof no longer applies.
- Algebraically the two differ by the scalar $\frac{K}{K-1}$ → same direction, absorbed by the LR.
- → The practical gap is negligible; the theoretical guarantee is not.

*Why drop the clip?*
- Measured: the PPO loss is clipped <5% of the time in RLHF runs {cite:p}`ahmadian2024back`
- Strong SFT init + small updates → the policy never leaves the trust region anyway.
- ⚠️ Long-CoT reasoning work reaches the opposite conclusion ← far more steps, far longer sequences, far larger drift.

*Cons?*
- $K$ rollouts per prompt → generation cost $\times K$.
- ❌IS correction → valid only at $\mu=1$ (one update per batch), the expensive regime.
- Sequence-level only → ❌knob for token-level loss aggregation.
```

&nbsp;

### PPO
- **Name**: Proximal Policy Optimization {cite:p}`schulman2017proximalpolicyoptimizationalgorithms`
- **What**: IS-reweighted policy gradient w/ a clipped trust region.
- **Why**: REINFORCE throws away each batch after a single step.
    - Rollouts dominate wall-clock → must take $\mu>1$ updates per batch.
    - But then the batch is off-policy → the plain gradient is biased.
    - And an unbounded off-policy step can land the policy where the data says nothing → collapse.
- **How**:
    1. Reweight each sampled token by the IS ratio $\rho_t$.
    2. `clip` $\rho_t$ into $[1-\epsilon,1+\epsilon]$, then take the `min` w/ the unclipped term.
    3. → The clipped branch is constant in $\theta$ → **exactly zero** gradient once the update overshoots in the direction $\hat{A}$ wants.
    4. Estimate $\hat{A}_t$ w/ GAE from a learned critic $V_\psi$.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\epsilon$: Clip range.
    - $\lambda$: GAE bias-variance coeff.
- Misc:
    - $s_t=(x,y_{<t})$: State at position $t$.
    - $\delta_t$: TD residual.
    - $\hat{R}_t$: Return target for the critic.

Ratio:

$$
\rho_t(\theta)=\frac{\pi_\theta(y_t|x,y_{<t})}{\pi_{\theta_\text{old}}(y_t|x,y_{<t})}
$$

Objective:

$$
J_\text{PPO}(\theta)=\mathbb{E}_t\left[\min\left(\rho_t(\theta)\hat{A}_t,\ \text{clip}\left(\rho_t(\theta),1-\epsilon,1+\epsilon\right)\hat{A}_t\right)\right]
$$

Gradient-equivalent mask form:

$$
M(\hat{A}_t,\rho_t,\epsilon)=\begin{cases}
0 & \text{if }(\hat{A}_t>0\wedge\rho_t>1+\epsilon)\vee(\hat{A}_t<0\wedge\rho_t<1-\epsilon) \\
1 & \text{otherwise}
\end{cases}
$$

$$
\nabla_\theta J_\text{PPO}(\theta)=\mathbb{E}_t\left[M(\hat{A}_t,\rho_t(\theta),\epsilon)\ \rho_t(\theta)\ \hat{A}_t\ \nabla_\theta\log\pi_\theta(y_t|x,y_{<t})\right]
$$
- Masked terms contribute $(1\pm\epsilon)\hat{A}_t$ to $J$, ❌$0$ — the two forms differ by a $\theta$-independent constant, so only their **gradients** coincide.

Advantage (GAE):

$$
\delta_t=r_t+\gamma V_\psi(s_{t+1})-V_\psi(s_t),\qquad \hat{A}_t=\sum_{l=0}^{|y|-t}(\gamma\lambda)^l\delta_{t+l}
$$
- $V_\psi(s_{|y|+1})=0$: Terminal state.
- $r_t=0$ for $t<|y|$, $r_{|y|}=r(x,y)$.

Critic training:

$$
\mathcal{L}_V(\psi)=\mathbb{E}_t\left[\left(V_\psi(s_t)-\hat{R}_t\right)^2\right]
$$
```

````{dropdown} Table: What the Clip Does
| $\hat{A}_t$ | $\rho_t<1-\epsilon$ | $\rho_t\in[1-\epsilon,1+\epsilon]$ | $\rho_t>1+\epsilon$ |
|:--|:--|:--|:--|
| $>0$ (reinforce) | $\rho_t\hat{A}_t$ → ✅grad | $\rho_t\hat{A}_t$ → ✅grad | $(1+\epsilon)\hat{A}_t$ → const → ❌grad |
| $<0$ (suppress) | $(1-\epsilon)\hat{A}_t$ → const → ❌grad | $\rho_t\hat{A}_t$ → ✅grad | $\rho_t\hat{A}_t$ → ✅grad |

Only 2 of the 4 out-of-range cases are masked: the ones where the policy has already moved **too far in the direction $\hat{A}_t$ asked for**. Moving back toward $\pi_{\theta_\text{old}}$ is always allowed.
````

````{important} Code
:class: dropdown
```python
import torch

def ppo_loss(logp, logp_old, adv, eps=0.2):
    ## logp, logp_old: (B, T) log-probs of the SAMPLED tokens; adv: (B, T)
    ratio = (logp - logp_old).exp()
    unclipped = ratio * adv
    clipped = ratio.clamp(1 - eps, 1 + eps) * adv
    ## min() -> pessimistic bound: kills the grad ONLY on an overshoot, never on a recovery
    return -torch.min(unclipped, clipped).mean()

## Example
logp, logp_old = torch.randn(2, 3), torch.randn(2, 3)
print(ppo_loss(logp, logp_old, torch.randn(2, 3)).item())
```
````

```{attention} Q&A
:class: dropdown
*Why `min`, and not `clip` alone?*
- `clip` alone would also zero the gradient when the ratio drifts **against** the advantage — i.e., while the policy is undoing an earlier overshoot.
- `min` makes $J$ a pessimistic lower bound on the unclipped objective → the clipped branch is selected only when it is the smaller one.
- → Overshoot: no gradient. Recovery: full gradient.

*Why "exactly zero" and not "small"?*
- The clipped branch has no $\theta$ in it → $\nabla_\theta$ of a constant.
- → The token is **deleted** from the update, not damped. This is the property CISPO attacks.

*What does GAE actually buy here?*
- $r_t=0$ for $t<|y|$ → $\delta_t=V_\psi(s_{t+1})-V_\psi(s_t)$ is entirely the **critic's own opinion** of how much $y_t$ moved the expected final score.
- Accurate critic → that is genuine per-token credit, the one thing group baselines cannot provide.
- Inaccurate critic → it is pure noise, injected at every position.
- At $\gamma=\lambda=1$ the sum telescopes to $\hat{A}_t=r(x,y)-V_\psi(s_t)$ → GAE degenerates to a plain learned baseline.
- → $\lambda$ is the dial between "trust the critic's per-token credit" and "trust only the final outcome".

*Why is the critic hard to train here?*
- Cold start ← it is randomly initialized (or head-grafted) while the policy is already strong.
- Non-stationary target ← it must track a policy that moves every step.
- Sparse target ← one scalar per response, credited to every prefix.

*What is $\epsilon$ in practice?*
- $0.2$ is PPO's near-universal default & transfers across models and tasks remarkably well.
- Later methods change **both** what $\epsilon$ constrains and its numeric range: DAPO splits it into $0.2/0.28$; GSPO's sequence ratio needs a range orders of magnitude smaller; CISPO's bound is on a weight, not a mask, and is far larger.

*Cons?*
- 4 large models resident: policy, rollout policy, reference, critic.
- Critic ≈ policy-sized → the memory ceiling on scaling RL.
- Many coupled hyperparams: $\epsilon$, $\lambda$, $\beta$, value-loss coeff, value clip.
```

&nbsp;

### GRPO
- **Name**: Group Relative Policy Optimization {cite:p}`shao2024deepseekmath`
- **What**: PPO w/ the critic replaced by group-normalized rewards.
- **Why**: The critic costs ~half the memory, and what it uniquely provides is hard to get right.
    - Its whole job is prefix-conditioned credit, $V(s_t)=\mathbb{E}[r|x,y_{<t}]$ — valuable in principle, but cold-started, non-stationary, and trained from one scalar per response.
    - Drop it and you only need the **prompt-level** baseline $V(x)$, which $G$ samples estimate directly & without learning.
    - Critic memory is what caps how large an RL run can go.
- **How**:
    1. Sample $G$ responses per prompt from $\pi_{\theta_\text{old}}$.
    2. Grade → $\{r_1,\dots,r_G\}$.
    3. Standardize within the group → $\hat{A}_i$; broadcast the same scalar to every token of $y_i$.
    4. PPO-style clipped update, + an explicit KL term **in the loss**.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $G$: Group size.
    - $\epsilon$: Clip range.
    - $\beta$: KL coeff.
- Misc:
    - $\rho_{i,t}(\theta)=\frac{\pi_\theta(y_{i,t}|x,y_{i,<t})}{\pi_{\theta_\text{old}}(y_{i,t}|x,y_{i,<t})}$: Token IS ratio.
    - $k_3$: Unbiased non-negative KL estimator (see KL Penalty).

Advantage:

$$
\hat{A}_{i,t}=\hat{A}_i=\frac{r_i-\text{mean}\left(\{r_j\}_{j=1}^{G}\right)}{\text{std}\left(\{r_j\}_{j=1}^{G}\right)}
$$

Objective:

$$
J_\text{GRPO}(\theta)=\mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}\left(\min\left(\rho_{i,t}\hat{A}_i,\ \text{clip}\left(\rho_{i,t},1-\epsilon,1+\epsilon\right)\hat{A}_i\right)-\beta k_3\right)\right]
$$
```

````{important} Code
:class: dropdown
```python
import torch

def grpo_advantage(rewards, eps_std=1e-4):
    ## rewards: (P, G) -> P prompts x G responses
    mu = rewards.mean(1, keepdim=True)
    sd = rewards.std(1, keepdim=True)
    return (rewards - mu) / (sd + eps_std)  ## guard: sd == 0 when the group is unanimous

def grpo_loss(logp, logp_old, logp_ref, adv, eps=0.2, beta=0.0):
    ## logp*: (B, T) sampled-token log-probs; adv: (B,) ONE scalar per response
    ratio = (logp - logp_old).exp()
    a = adv.unsqueeze(-1)                                   ## broadcast over tokens
    surr = torch.min(ratio * a, ratio.clamp(1 - eps, 1 + eps) * a)
    log_rho = logp_ref - logp
    k3 = log_rho.exp() - log_rho - 1                        ## KL sits in the LOSS, not the reward
    ## mean over tokens FIRST (the 1/|y_i|), then over responses -> GRPO's length bias
    return -(surr - beta * k3).mean(-1).mean()

## Example
r = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
adv = grpo_advantage(r).flatten()
lp, lp_old, lp_ref = torch.randn(4, 3), torch.randn(4, 3), torch.randn(4, 3)
print(grpo_loss(lp, lp_old, lp_ref, adv).item())
```
````

```{attention} Q&A
:class: dropdown
*Why does a group of samples beat a learned critic?*
- They do **not** estimate the same thing. The critic targets $V(s_t)$ per prefix; the group estimates only $V(x)$, the prompt-level value.
- → GRPO trades away prefix-conditioned credit assignment for a baseline that needs no learning and cannot be miscalibrated.
- Empirically that trade wins ← the critic's per-prefix estimates were mostly noise anyway, and $G\times$ generation parallelizes while $2\times$ memory does not.

*Why divide by $\sigma_G$?*
- Puts advantages on one scale across prompts & across mixed reward sources.
- ⚠️ Not free — it silently reweights prompts (→ Dr. GRPO).

*What happens when the group is unanimous?*
- All $r_i$ equal → $\mu_G=r_i$ → $\hat{A}_i=0$ ∀$i$ → 0 gradient, and $\sigma_G=0$ needs a guard.
- Binary reward → any prompt that is too easy or too hard is **wasted rollout compute**.
- → Motivates DAPO's dynamic sampling & zero-variance filtering.

*Why is KL in the loss instead of the reward?*
- Reward-level KL would pass through group standardization → the penalty gets rescaled by $\sigma_G$, which has nothing to do with it.
- Loss-level keeps $\hat{A}$ purely task reward and $\beta$ interpretable.

*What is $G$ in practice?*
- 8–64. ⬆️$G$ → ⬇️baseline variance & ⬆️chance of a mixed group, but generation cost scales linearly.

*Cons?*
- ❌Token-level credit ← one $\hat{A}_i$ broadcast over $|y_i|$ tokens.
- $\frac{1}{|y_i|}$ sample-level averaging → length bias (→ Dr. GRPO).
- $\sigma_G$ division → difficulty bias (→ Dr. GRPO).
- Token-level $\rho_{i,t}$ weights tokens unequally within one response (→ GSPO).
```

&nbsp;

#### Dr. GRPO
- **Name**: GRPO Done Right {cite:p}`liu2025understanding`
- **What**: GRPO minus both of its normalizations.
- **Why**: GRPO's two divisions look cosmetic but reweight the objective.
    - **Response-level length bias** ← dividing by $|y_i|$.
        - $\hat{A}_i>0$ → shorter correct responses get larger per-token updates.
        - $\hat{A}_i<0$ → longer incorrect responses get **smaller** per-token penalty.
        - → Being wrong is cheaper the longer you are → failed rollouts drift longer & longer.
    - **Question-level difficulty bias** ← dividing by $\sigma_G$.
        - $\sigma_G$ is smallest exactly when the prompt is nearly-solved or nearly-impossible.
        - → Those prompts' advantages get inflated the most.
- **How**:
    1. Drop $\sigma_G$ → $\hat{A}_i=r_i-\mu_G$.
    2. Drop $\frac{1}{|y_i|}$ → divide the summed token loss by a **constant** (the generation budget).

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $L_\text{max}$: Max generation length, fixed for the whole run.

Advantage:

$$
\hat{A}_{i,t}=\hat{A}_i=r_i-\text{mean}\left(\{r_j\}_{j=1}^{G}\right)
$$

Objective:

$$
J_\text{Dr.GRPO}(\theta)=\mathbb{E}\left[\frac{1}{G\cdot L_\text{max}}\sum_{i=1}^{G}\sum_{t=1}^{|y_i|}\min\left(\rho_{i,t}\hat{A}_i,\ \text{clip}\left(\rho_{i,t},1-\epsilon,1+\epsilon\right)\hat{A}_i\right)\right]
$$
```

````{important} Code
:class: dropdown
```python
import torch

def dr_grpo_advantage(rewards):
    ## rewards: (P, G) -> center only, NO /std
    return rewards - rewards.mean(1, keepdim=True)

def dr_grpo_loss(logp, logp_old, adv, mask, l_max, eps=0.2):
    ## mask: (B, T) 1 for real tokens, 0 for padding
    ratio = (logp - logp_old).exp()
    a = adv.unsqueeze(-1)
    surr = torch.min(ratio * a, ratio.clamp(1 - eps, 1 + eps) * a)
    ## denominator is a CONSTANT -> a pure LR rescale, so it cannot reweight samples
    return -(surr * mask).sum() / (logp.size(0) * l_max)

## Example
r = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
print(dr_grpo_advantage(r))  ## tensor([[ 0.2500,  0.2500,  0.2500, -0.7500]])
```
````

```{attention} Q&A
:class: dropdown
*Why does the length bias inflate response length?*
- Per-token gradient magnitude $\propto\hat{A}_i/|y_i|$.
- Wrong & long → each token is barely punished → long wrong answers are cheap to keep.
- → Length can grow while reasoning does not.

*So the "responses get longer = it's learning to reason" story is wrong?*
- Partly. Removing the bias keeps length from growing wildly & sharply shortens **incorrect** responses, while reward still improves.
- → Length growth is a mix of real reasoning and an optimizer artifact; length alone is not evidence of either.

*Why is dividing by $\sigma_G$ worse than batch-level whitening?*
- Batch whitening applies one scale to everything → relative prompt weights unchanged.
- Per-prompt whitening applies a **different** scale per prompt → it is a reweighting, and it up-weights the lowest-information prompts.

*Why a constant denominator rather than a per-sequence one?*
- GRPO's $\frac{1}{|y_i|}$ is applied **per response** → it changes how responses weigh against each other by their length. That is the bias.
- A constant is a pure LR rescale → it cannot reweight anything.
- vs DAPO's $\frac{1}{\sum_i|y_i|}$: that is also batch-wide, so it does not reweight *within* a batch either — the two differ only by the scalar $\frac{G\cdot L_\text{max}}{\sum_i|y_i|}$, i.e., across batches.
- Other constants work too; they only shift gradient norm.

*Cons?*
- Effective LR now varies with how many tokens the batch happens to contain.
- Unnormalized advantages → reward scale must be sane across mixed task types.
```

&nbsp;

#### DAPO
- **Name**: Decoupled Clip and Dynamic sAmpling Policy Optimization {cite:p}`yu2025dapo`
- **What**: GRPO + 4 fixes for long-CoT: clip-higher, dynamic sampling, token-level loss, overlong shaping.
- **Why**: Naive GRPO on long-CoT degenerates in 3 separate ways.
    - **Entropy collapse** → the policy sharpens early & stops exploring.
    - **Dead batches** → unanimous groups give $\hat{A}=0$, so a growing share of each batch contributes no gradient.
    - **Truncation noise** → a cut-off response is scored identically to a wrong one.
    - (+ GRPO's length bias, same diagnosis as Dr. GRPO.)
- **How**:
    1. **Clip-Higher**: decouple the bounds, $\epsilon_\text{low}=0.2$, $\epsilon_\text{high}=0.28$.
    2. **Dynamic Sampling**: keep resampling until every group in the batch has mixed outcomes.
    3. **Token-Level Loss**: divide by the batch's **total** token count, ❌per-sequence mean.
    4. **Overlong Reward Shaping**: soft penalty ramp before the hard length cutoff.
    - Drops the KL term entirely.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\epsilon_\text{low}$: Lower clip bound.
    - $\epsilon_\text{high}$: Upper clip bound.
    - $L_\text{max}$: Hard generation cutoff.
    - $L_\text{cache}$: Width of the soft penalty zone below the cutoff.
- Misc:
    - $a^*$: Ground-truth answer.

Objective:

$$
J_\text{DAPO}(\theta)=\mathbb{E}\left[\frac{1}{\sum_{i=1}^{G}|y_i|}\sum_{i=1}^{G}\sum_{t=1}^{|y_i|}\min\left(\rho_{i,t}\hat{A}_i,\ \text{clip}\left(\rho_{i,t},1-\epsilon_\text{low},1+\epsilon_\text{high}\right)\hat{A}_i\right)\right]
$$

Dynamic sampling constraint:

$$
0<\left|\{y_i:\text{ext}(y_i)=a^*\}\right|<G
$$

Overlong reward shaping:

$$
R_\text{length}(y)=\begin{cases}
0 & |y|\leq L_\text{max}-L_\text{cache} \\
\frac{(L_\text{max}-L_\text{cache})-|y|}{L_\text{cache}} & L_\text{max}-L_\text{cache}<|y|\leq L_\text{max} \\
-1 & |y|>L_\text{max}
\end{cases}
$$

$\hat{A}_i$ keeps GRPO's $\sigma_G$ standardization.
```

````{important} Code
:class: dropdown
```python
import torch

def dapo_loss(logp, logp_old, adv, mask, eps_low=0.2, eps_high=0.28):
    ## adv: (B,) group-standardized, same as GRPO
    ratio = (logp - logp_old).exp()
    a = adv.unsqueeze(-1)
    ## asymmetric: more headroom to RAISE a token than to crush it
    surr = torch.min(ratio * a, ratio.clamp(1 - eps_low, 1 + eps_high) * a)
    ## token-level: one denominator for the whole batch, not one per sequence
    return -(surr * mask).sum() / mask.sum()

def keep_prompt(correct):
    ## correct: (G,) bool -> is_equivalent(ext(y_i), a*) per response
    ## DAPO's reward is +1/-1, so test OUTCOMES, not the reward sum
    n = int(correct.sum())
    return 0 < n < correct.numel()

## Example
print(keep_prompt(torch.tensor([True, True, False, False])))  ## True
print(keep_prompt(torch.tensor([True, True, True, True])))     ## False
```
````

```{attention} Q&A
:class: dropdown
*Why raise only the upper bound?*
- The ratio is **multiplicative**, so the clip is a cap in relative terms.
- $p=0.01$, $\epsilon=0.2$ → can only reach $0.012$. $p=0.9$ → $1.08$, i.e., unconstrained.
- → The symmetric clip is effectively a hard cap on promoting **rare** tokens, and free rein on already-likely ones.
- Rare tokens are the reasoning forks ("However", "Wait", "Recheck") that long CoT needs.

*Why not raise $\epsilon_\text{low}$ too?*
- The lower bound is what stops a token from being crushed toward 0 in one batch.
- Probability driven to ~0 is effectively unrecoverable ← it stops being sampled, so it stops receiving gradient.

*How does clip-higher fix entropy collapse?*
- Up-clipping is what blocks rare tokens from gaining mass, so the distribution could only ever sharpen.
- Relaxing it restores upward mobility → entropy flattens or rises slowly instead of crashing.
- Confirmed empirically: the mean probability of up-clipped tokens is low.

*Why is dynamic sampling not free?*
- Regenerate-until-mixed → variable, sometimes much larger, rollout cost per step.
- Better **step** efficiency, worse **wall-clock** per step.
- Cheaper alternative: just discard unanimous groups and accept a smaller effective batch.

*Why is truncation shaping needed?*
- Default: truncated = wrong → a nearly-correct 32k trace is graded a failure.
- That is label noise on the reward, concentrated on exactly the hardest prompts.
- The ramp separates "too long" from "incorrect" so the policy can learn length control independently.

*DAPO's token-level average vs Dr. GRPO's constant?*
- Both delete GRPO's per-response $\frac{1}{|y_i|}$, so both weight a response proportionally to its length. Their numerators are **identical**.
- They differ only by the batch-wide scalar $\frac{G\cdot L_\text{max}}{\sum_i|y_i|}$ → a per-batch rescale, ❌a within-batch reweighting.
- DAPO: per-token mean → gradient scale is invariant to how many tokens the batch holds, but the denominator is random.
- Dr. GRPO: fixed denominator → no random rescaling across batches, but gradient norm grows with total tokens.

*Cons?*
- Keeps $\sigma_G$ → retains the difficulty bias Dr. GRPO argues against.
- Keeps PPO's hard masking → clipped tokens still contribute nothing (→ CISPO).
- 4 extra hyperparams to tune.
```

&nbsp;

#### GSPO
- **Name**: Group Sequence Policy Optimization {cite:p}`zheng2025group`
- **What**: GRPO w/ the likelihood ratio defined, clipped, and applied at **sequence** level.
- **Why**: The unit of the ratio does not match the unit of the reward.
    - Reward is defined on the whole sequence; $\rho_{i,t}$ constrains one token.
    - → Per-token ratios weight tokens **unequally** within a response at fixed $\hat{A}_i$, and that noise accumulates over $|y_i|$ positions & is amplified by clipping.
    - Worst in MoE ← routing is discrete, so a small weight change flips which experts fire → $\rho_{i,t}$ swings wildly for reasons unrelated to the reward.
- **How**:
    1. Ratio = sequence likelihood ratio, raised to $\frac{1}{|y_i|}$ (geometric mean of the token ratios).
    2. Clip $s_i$, not $\rho_{i,t}$ → the unit of clipping is now a **whole response**.
    3. → Every token in a kept response gets the same weight $s_i$.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $s_i(\theta)$: Length-normalized sequence-level likelihood ratio.

Ratio:

$$
s_i(\theta)=\left(\frac{\pi_\theta(y_i|x)}{\pi_{\theta_\text{old}}(y_i|x)}\right)^{\frac{1}{|y_i|}}=\exp\left(\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}\log\frac{\pi_\theta(y_{i,t}|x,y_{i,<t})}{\pi_{\theta_\text{old}}(y_{i,t}|x,y_{i,<t})}\right)
$$

Objective:

$$
J_\text{GSPO}(\theta)=\mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\min\left(s_i(\theta)\hat{A}_i,\ \text{clip}\left(s_i(\theta),1-\epsilon,1+\epsilon\right)\hat{A}_i\right)\right]
$$
- $\hat{A}_i$: GRPO's group-standardized advantage, unchanged.

Gradient (clipping omitted):

$$\begin{align*}
\nabla_\theta J_\text{GRPO}&=\mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\hat{A}_i\cdot\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}\rho_{i,t}(\theta)\nabla_\theta\log\pi_\theta(y_{i,t}|x,y_{i,<t})\right] \\
\nabla_\theta J_\text{GSPO}&=\mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}s_i(\theta)\hat{A}_i\cdot\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}\nabla_\theta\log\pi_\theta(y_{i,t}|x,y_{i,<t})\right]
\end{align*}$$
```

````{important} Code
:class: dropdown
```python
import torch

def gspo_loss(logp, logp_old, adv, mask, eps):
    ## eps must be ORDERS OF MAGNITUDE below PPO's 0.2 -> see Q&A
    ## geometric mean of the token ratios == arithmetic mean of the log-ratios
    log_s = ((logp - logp_old) * mask).sum(-1) / mask.sum(-1)   ## (B,)
    s = log_s.exp()
    surr = torch.min(s * adv, s.clamp(1 - eps, 1 + eps) * adv)
    return -surr.mean()

## Example
lp, lp_old = torch.randn(2, 5), torch.randn(2, 5)
mask, adv = torch.ones(2, 5), torch.tensor([1.0, -1.0])
print(gspo_loss(lp, lp_old, adv, mask, eps=3e-4).item())
```
````

```{attention} Q&A
:class: dropdown
*What exactly changes in the gradient?*
- GRPO weights token $t$ by its own $\rho_{i,t}$ → **unequal** weights within one response, ranging over $(0,1+\epsilon]$ or $[1-\epsilon,+\infty)$ depending on the sign of $\hat{A}_i$.
- GSPO weights every token of a response by the same $s_i$ → equal weights.
- → Both carry one advantage per response; GSPO just stops adding per-token noise on top of it.

*Is $s_i$ a valid importance weight?*
- ❌. The valid sequence-level IS weight is $\frac{\pi_\theta(y_i|x)}{\pi_{\theta_\text{old}}(y_i|x)}$; the $\frac{1}{|y_i|}$ root destroys that identity.
- $s_i$ is a **trust-region statistic** — a length-comparable measure of how far the response's likelihood moved — ❌an unbiased off-policy correction.
- The paper motivates it by arguing that a single-token ratio "fails to perform distribution correction". ⚠️ One-sample IS is still unbiased in expectation, so that argument is heuristic; the defensible claims are the unit mismatch, the variance accumulation, and the MoE evidence.

*Why must $\epsilon$ be far smaller than PPO's $0.2$?*
- $s_i$ is a geometric mean over $|y_i|$ token ratios → concentrates tightly around 1.
- A $0.2$ window would essentially never bind.
- → Clip ranges are **not comparable** across GSPO and GRPO; they measure different objects.

*What does a clip mean now?*
- Token-level clip drops individual tokens; sequence-level clip drops the **entire response**.
- → Coarser, but the unit of the trust region finally matches the unit of the reward.

*Why does this stabilize MoE RL?*
- Routing flips between $\pi_{\theta_\text{old}}$ and $\pi_\theta$ make per-token likelihoods jump even under a tiny weight change.
- Averaging log-ratios over the sequence washes those spikes out.
- → Removes the need to cache & replay $\pi_{\theta_\text{old}}$'s expert routes.

*Why does it simplify infrastructure?*
- Only the sequence-level likelihood matters → tolerant of small per-token numerical differences between the inference and training engines.
- → Directly attacks the train-inference mismatch failure mode.

*Cons?*
- Whole-response masking → one pathological token can discard an entire rollout's gradient.
- ❌Token-level advantage. (The paper's GSPO-token variant restores that knob and is numerically identical when all token advantages are equal.)
```

&nbsp;

#### CISPO
- **Name**: Clipped IS-weight Policy Optimization {cite:p}`minimax2025minimax`
- **What**: REINFORCE w/ a clipped, stop-gradient IS weight — ❌masking.
- **Why**: PPO-style clipping deletes exactly the tokens worth learning from.
    - Clipped in a masked direction → constant branch → **zero** gradient → the token is gone for the whole batch.
    - Tokens that trip the upper clip are the low-probability ones ("However", "Recheck", "Wait") — the reasoning forks.
    - Long CoT → those appear a handful of times per trace → deleting them deletes the signal.
- **How**:
    1. Compute $\rho_{i,t}$ and clip it → $\hat{\rho}_{i,t}$.
    2. Wrap it in $\text{sg}[\cdot]$ → it is a constant multiplier, ❌a function of $\theta$.
    3. Multiply it onto a plain $\log\pi_\theta$ term → every token keeps a gradient; only its **magnitude** is capped.
    4. Only $\epsilon^\text{IS}_\text{high}$ is tuned; $\epsilon^\text{IS}_\text{low}$ is set large enough to never bind.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\epsilon^\text{IS}_\text{low}$: Lower IS-weight clip bound.
    - $\epsilon^\text{IS}_\text{high}$: Upper IS-weight clip bound.
- Misc:
    - $\hat{\rho}_{i,t}$: Clipped IS weight.

Weight:

$$
\hat{\rho}_{i,t}(\theta)=\text{clip}\left(\rho_{i,t}(\theta),1-\epsilon^\text{IS}_\text{low},1+\epsilon^\text{IS}_\text{high}\right)
$$

Objective:

$$
J_\text{CISPO}(\theta)=\mathbb{E}\left[\frac{1}{\sum_{i=1}^{G}|y_i|}\sum_{i=1}^{G}\sum_{t=1}^{|y_i|}\text{sg}\left[\hat{\rho}_{i,t}(\theta)\right]\hat{A}_i\log\pi_\theta(y_{i,t}|x,y_{i,<t})\right]
$$
```

```{tip} Derivation
:class: dropdown
*Why is this the same as PPO inside the trust region, and different outside?*

1. PPO's unclipped branch, using $\nabla_\theta\rho_t=\rho_t\nabla_\theta\log\pi_\theta$:

    $$
    \nabla_\theta\left(\rho_t\hat{A}_t\right)=\rho_t\hat{A}_t\nabla_\theta\log\pi_\theta
    $$

2. PPO's clipped branch is constant in $\theta$ → $\nabla_\theta=0$.

3. CISPO, with $\text{sg}[\hat{\rho}_t]$ treated as a constant:

    $$
    \nabla_\theta\left(\text{sg}[\hat{\rho}_t]\hat{A}_t\log\pi_\theta\right)=\hat{\rho}_t\hat{A}_t\nabla_\theta\log\pi_\theta
    $$

4. $\rho_t$ inside the clip range → $\hat{\rho}_t=\rho_t$ → **identical** to PPO.

5. $\rho_t$ outside, in one of the 2 masked cases ($\hat{A}_t>0\wedge\rho_t>1+\epsilon$, or $\hat{A}_t<0\wedge\rho_t<1-\epsilon$) → PPO gives $0$, CISPO gives the same direction capped at $\hat{\rho}_t$.

6. $\rho_t$ outside, in the 2 recovery cases → PPO keeps the full $\rho_t$ weight, CISPO caps it → CISPO is the more conservative one here.

7. → CISPO replaces PPO's *delete* with a *saturate*, in both directions.
```

````{important} Code
:class: dropdown
```python
import torch

def cispo_loss(logp, logp_old, adv, mask, eps_high, eps_low=1e9):
    ## eps_low huge -> lower bound never binds (only the upper one is tuned)
    ratio = (logp - logp_old).exp()
    w = ratio.clamp(1 - eps_low, 1 + eps_high).detach()   ## sg[] on the WEIGHT
    ## plain REINFORCE term -> every token keeps a gradient, capped in magnitude
    surr = w * adv.unsqueeze(-1) * logp
    return -(surr * mask).sum() / mask.sum()

## Example
lp, lp_old = torch.randn(2, 5), torch.randn(2, 5)
mask, adv = torch.ones(2, 5), torch.tensor([1.0, -1.0])
print(cispo_loss(lp, lp_old, adv, mask, eps_high=5.0).item())
```
````

```{attention} Q&A
:class: dropdown
*Why stop-gradient the weight?*
- Without $\text{sg}$, gradient would also flow through $\hat{\rho}_t$, adding a term that is not a policy gradient.
- IS weights are **corrections to the sampling distribution**, not part of the objective → they must be constants.

*Is it unbiased?*
- ❌. Weight clipping biases the gradient — acknowledged in the paper.
- Trade: small bias, but zero information loss & lower variance than the unclipped weight.
- Remove the clip → it reduces to the plain token-ratio-weighted policy gradient.

*Why is only the upper bound needed?*
- The variance blow-up comes from weights $\gg1$, i.e., tokens whose probability rose sharply under $\pi_\theta$ → cap those.
- A lower bound is a **floor**: it would *raise* the weight of tokens whose probability collapsed, re-amplifying exactly the samples the policy is correctly abandoning.
- → No variance to gain, and a distortion to pay.

*How does this compare to DAPO's clip-higher?*
- Same diagnosis (rare tokens are being suppressed), different remedy.
- DAPO widens the window in which gradients survive; CISPO removes the window and caps magnitude instead.
- Reported: CISPO matches DAPO's AIME'24 accuracy in ~50% of the training steps, on Qwen2.5-32B-base.

*Cons?*
- ❌Trust region in the PPO sense → nothing bounds how far one update moves the policy; stability rests entirely on the weight cap.
- Biased gradient.
- $\epsilon^\text{IS}_\text{high}$ becomes a sensitive knob — MiniMax report lowering it (w/ the gradient-clipping threshold) to stop late-training degeneration.
```

&nbsp;

## Practice
### Entropy Collapse
- **What**: Policy entropy → 0 early in training.
- **Why**: RL on a pretrained LM is nearly pure exploitation.
    - The policy starts competent → reward maximization sharpens it onto what already works, ❌explores.
    - The symmetric clip is asymmetric **in probability space** → rare tokens can barely gain mass, likely ones are unconstrained.
    - Masking removes clipped tokens entirely → the tokens that would raise entropy contribute nothing.
- **How**: Monitor it as a first-class metric, then widen the upward path.
    - Clip-Higher (DAPO) → more headroom for rare tokens.
    - Saturate instead of mask (CISPO) → rare tokens are never dropped.
    - Entropy bonus / entropy target → direct, but a fragile knob (see Q&A).
    - Track policy entropy **and** the mean generation probability side by side; they move opposite ways.

```{attention} Q&A
:class: dropdown
*Why is it self-reinforcing?*
- ⬇️Entropy → group samples become near-duplicates → $\sigma_G\to0$ and $\hat{A}_i\to0$.
- → ⬇️Gradient → nothing left to push entropy back up.
- The only cure is exploration, which is precisely what was lost.

*Isn't low entropy just the model becoming confident?*
- Confidence **after** the policy is good is fine; collapse **before** it is good is terminal.
- Diagnostic: entropy ⬇️ while reward plateaus → collapse. Reward ⬆️ w/ entropy flat or slowly ⬆️ → healthy.

*Is more entropy always better?*
- ❌. Too high → gibberish, repetition, degenerate loops.
- → Entropy is a band to stay inside, ❌a quantity to maximize.

*Why is an entropy bonus a fragile fix?*
- It rewards uncertainty everywhere, including on tokens the policy has correctly resolved.
- Coefficient too low → no effect; too high → the model is paid to be vague.
- → Structural fixes (clip bounds, no masking) act only where the suppression actually happens.

*How does this connect to the pass@1 vs pass@k gap?*
- RL objectives maximize expected reward = pass@1.
- Sharpening onto one mode raises pass@1 while destroying the diversity that pass@$k$ measures.
- → Entropy collapse and the widely-reported pass@$k$ regression are the same phenomenon seen from two angles.
```

&nbsp;

### Train-Inference Mismatch
- **What**: Token probs from the inference engine $\neq$ recomputed training probs, at **identical** weights. {cite:p}`minimax2025minimax`
- **Why**: Rollout and training are different programs.
    - Rollout: fused/paged attention kernels, low precision, different batching → different reduction order.
    - Training: standard forward pass.
    - → Same $\theta$, numerically different logits.
    - $\rho_{i,t}$ is a **ratio** → a tiny absolute error becomes a large relative error exactly where $\pi$ is small.
    - → The run is silently off-policy from step 0; in the worst case reward never grows at all.
- **How**:
    1. Diagnose: scatter inference vs training probs token-by-token; correlation should be ~1.
    2. Fix the numerics: compute the LM output head in FP32.
    3. Or treat it as genuine off-policyness: use the rollout log-probs as $\pi_{\theta_\text{old}}$ so the IS ratio actually corrects for it.
    4. Or reduce the exposure: a sequence-level ratio (GSPO) averages the per-token noise away.

```{attention} Q&A
:class: dropdown
*Why is this worse than ordinary off-policy drift?*
- Drift is systematic & points in the direction the update intended; even though the clip does **not** hard-bound it, the clip at least removes the incentive to push further.
- This is **noise**, uncorrelated with the update direction, and unbounded in relative terms.
- It is largest on low-probability tokens — the same ones clip-higher and CISPO exist to protect.

*Why did it surface only recently?*
- Long CoT → errors compound over $10^4$ positions.
- Large & MoE models → high-magnitude activations at the output layer, plus discrete routing.
- MiniMax report it did **not** appear in smaller dense softmax-attention models.

*Does the FP32 head fully solve it?*
- It substantially realigns the two: reported correlation ~0.9x → ~0.99x, stable across training.
- ❌Exact. A residual gap remains, which is why algorithm-side robustness is still worth having.

*Why does recomputing $\pi_{\theta_\text{old}}$ under the training kernel hide the bug?*
- Then $\rho\equiv1$ at the first update by construction, so the ratio looks perfectly on-policy.
- But the data was actually drawn from the **inference** distribution → the mismatch becomes an uncorrected bias instead of a visible quantity.
- → $\rho\approx1$ at step 0 is not evidence that the mismatch is absent.

*Why is this a warning about RL results in general?*
- A silent numerics bug and a genuine algorithmic difference look identical from the reward curve.
- → Method comparisons are only meaningful once the train-inference correlation is verified for every arm.
```

&nbsp;

### Design Space
- **What**: The 4 knobs every post-GRPO method turns.
- **Why**: The methods differ by small deltas, so the deltas are the entire content of the comparison.
- **How**:
    1. **Baseline**: critic / group mean / leave-one-out — and whether to divide by $\sigma$.
    2. **Trust-region unit**: token ratio / sequence ratio.
    3. **Trust-region action**: mask (delete the gradient) / cap the weight (keep it).
    4. **Loss aggregation**: per-sequence mean / batch-token mean / fixed constant.

````{dropdown} Table: Algorithms at a Glance
| Method | Advantage | Trust region | Out-of-range | Aggregation | Delta |
|:--|:--|:--|:--|:--|:--|
| REINFORCE | $r-b(x)$ | ❌ | — | Sample mean | Policy gradient |
| RLOO | $r_i-\frac{1}{K-1}\sum_{j\neq i}r_j$ | ❌ | — | Sample mean | ❌Critic, unbiased baseline |
| PPO | GAE w/ critic $V_\psi$ | Token $\rho_t$, $\pm\epsilon$ | Mask | Sample mean | Batch reuse w/o collapse |
| GRPO | $(r_i-\mu_G)/\sigma_G$ | Token $\rho_{i,t}$, $\pm\epsilon$ | Mask | $\frac{1}{\|y_i\|}$ per sample | ❌Critic → ⬇️memory |
| Dr. GRPO | $r_i-\mu_G$ | Token $\rho_{i,t}$, $\pm\epsilon$ | Mask | $\frac{1}{L_\text{max}}$ const | ❌Length bias, ❌difficulty bias |
| DAPO | $(r_i-\mu_G)/\sigma_G$ | Token $\rho_{i,t}$, $[-\epsilon_l,+\epsilon_h]$ | Mask | $\frac{1}{\sum_i\|y_i\|}$ | Room for rare tokens, ❌dead batches |
| GSPO | $(r_i-\mu_G)/\sigma_G$ | **Sequence** $s_i$, $\pm\epsilon$ | Mask (whole response) | $\frac{1}{G}$ per sample | Ratio unit = reward unit |
| CISPO | $(r_i-\mu_G)/\sigma_G$ | Token $\hat{\rho}_{i,t}$, upper only | **Cap** (sg) | $\frac{1}{\sum_i\|y_i\|}$ | ❌Gradient deletion |

$\epsilon_l,\epsilon_h$: Decoupled lower/upper clip bounds.
````

```{attention} Q&A
:class: dropdown
*What has the field actually converged on?*
- **❌Critic.** Every method after PPO drops it; sample-based baselines match it at ~half the memory. Strong pretrained init makes PPO's variance machinery largely redundant here.
- **❌Per-response loss averaging.** GRPO's $\frac{1}{|y_i|}$ is a length reweighting in disguise; every later method removed it.
- **$\sigma_G$ is contested, ❌settled.** Dr. GRPO shows it up-weights the least informative prompts — but DAPO, GSPO, and CISPO all kept it.
- **Trust regions are open.** $\epsilon=0.2$ has held up remarkably well, yet DAPO, GSPO, and CISPO each argue a different part of it is wrong.

*So which one should I run?*
- Verifiable reward, ordinary dense model → GRPO w/ Dr. GRPO's two fixes is a strong, simple default.
- Long CoT, entropy dying → DAPO's clip-higher, or CISPO.
- MoE, or training that collapses for no visible reason → GSPO.
- Learned RM in the loop → keep the KL anchor regardless of which objective you pick.

*What is still unsolved?*
- **Credit assignment.** One scalar for $10^3$–$10^5$ tokens; the token that broke the proof is scored like the whitespace next to it.
- **Sample efficiency.** A binary verifier yields ~1 bit per rollout, and methods need 8–64 rollouts per prompt to build a baseline.
- **Prompts that are never solved.** $r_i=0$ ∀$i$ → $\hat{A}=0$ → no gradient, ever. Curriculum is a workaround, not a fix.
- **Domains without a verifier.** Nearly all progress is math & code; noisy, delayed, subjective, or multi-turn rewards remain hard.
- **Reproducibility.** Most papers report one model family, one verifier, one data mix, one compute budget — and an intervention can change early speed, final ceiling, or both.

*Why is a single checkpoint comparison misleading?*
- Two methods can cross: the faster early learner may plateau below the slower one.
- → Reward-vs-compute curves, not a single number, are the honest comparison.

*What does any of this say about intelligence?*
- Pretraining supplies the hypothesis space; RL only reweights it. Every method here shifts probability mass among behaviors the base model could already produce.
- The unsolved item is credit assignment — attributing one outcome to the specific internal step that caused it.
- → An agent that cannot localize its own error must brute-force the search, which is exactly what 8–64 rollouts per prompt buys.
```

&nbsp;
