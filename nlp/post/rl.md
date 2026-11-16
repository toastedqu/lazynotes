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
# RL
RL on a pretrained LM: problem setup, objectives from REINFORCE → PPO → GRPO & its descendants, failure modes in practice.

General policy-gradient theory: [Policy-Based Methods](../../rl/policy.md).

Notations:
- $x$: Prompt (input token seq)
- $y$: Response (output token seq)
- $y_t$: $t$-th response token
- $y_{<t}$: Response tokens before position $t$
- $|y|$: Response length (#tokens)
- $s_t=(x,y_{<t})$: State at position $t$
- $\mathcal{V}$: Vocabulary
- $\mathcal{D}$: Prompt dataset
- $\pi_\theta$: Curr policy (the LM being trained)
- $\pi_{\theta_\text{old}}$: Rollout policy (generated the curr batch)
- $\pi_\text{ref}$: Reference policy (frozen, usually post-SFT)
- $J(\theta)$: Expected reward (RL objective)
- $r(x,y)$: Scalar reward for a full response
- $r_\phi$: Learned RM, parameterized by $\phi$
- $\text{ext}(y)$: Answer extracted from $y$
- $a^*$: Ground-truth answer
- $\hat{A}$: Advantage estimate
- $V_\psi$: Value model (critic), parameterized by $\psi$
- $\gamma$: Discount; $\gamma=1$ on this page
- $\mu$: #grad steps per rollout batch
- $G$: Group size (#responses sampled per prompt)
- $i$: Response idx within a group
- $t$: Token idx within a response
- $r_i=r(x,y_i)$: Reward of the $i$-th response
- $\mu_G$: Mean of $\{r_i\}_{i=1}^G$
- $\sigma_G$: Std of $\{r_i\}_{i=1}^G$
- $\rho_{i,t}=\frac{\pi_\theta(y_{i,t}|x,y_{i,<t})}{\pi_{\theta_\text{old}}(y_{i,t}|x,y_{i,<t})}$: Token IS ratio ($\rho_t$ w/o a group)
- $\epsilon$: Clip range
- $\beta$: KL coeff
- $L_\text{max}$: Max generation length
- $\text{sg}[\cdot]$: Stop-gradient

&nbsp;

## Setup
### MDP Framing
- **What**: Prompt → state, response → action, LM → policy, grader → reward.
- **Why**: RL needs an env. An LM has none.
    - LM = conditional next-token distribution, ❌agent.
    - ❌Transition physics, ❌per-step reward, ❌episode past one response → build all of them.
- **How**:
    1. **Token-MDP**: state = prompt + tokens so far, action = next token, transition = deterministic concat.
    2. Grader scores only the full response → reward 0 until EOS.
    3. → Collapse to a **contextual bandit**: 1 prompt, 1 action (whole response), 1 reward.
    4. $\gamma=1$ ← $\gamma<1$ would act as a hidden length penalty.

```{note} Math
:class: dropdown
Model:

$$
a_t=y_t\in\mathcal{V},\qquad p(s_{t+1}|s_t,a_t)=\mathbb{1}[s_{t+1}=(x,y_{\leq t})],\qquad r_t=\begin{cases}0 & t<|y|\\ r(x,y) & t=|y|\end{cases}
$$
- Bandit collapse: action = whole response, w/ likelihood

    $$
    \pi_\theta(y|x)=\prod_{t=1}^{|y|}\pi_\theta(y_t|x,y_{<t})
    $$

Training:

$$
\max_\theta J(\theta)=\mathbb{E}_{x\sim\mathcal{D},\ y\sim\pi_\theta(\cdot|x)}[r(x,y)]
$$
```

````{dropdown} Table: RL $\Leftrightarrow$ LLM
| RL | LLM | Note |
|:--|:--|:--|
| Agent | LM | |
| Policy $\pi_\theta$ | LM weights | Init from pretrain (+SFT), ❌random |
| State $s_t$ | Prompt + tokens so far | |
| Action $a_t$ | Next token | $\|\mathcal{A}\|=\|\mathcal{V}\|\sim10^5$ |
| Transition $p(s_{t+1}\|s_t,a_t)$ | String concat | Deterministic, known |
| Reward $r_t$ | 0 until EOS, then $r(x,y)$ | Sparse, terminal-only |
| Episode | One response | $T=\|y\|$ |
| Env | Grader (RM / verifier) | Stateless, ❌dynamics |
````

```{attention} Q&A
:class: dropdown
*Why does the token-MDP collapse to a bandit?*
- Transitions are deterministic & known → ❌dynamics to learn.
- Rewards are terminal-only → env gives ❌per-step signal.
- Per-token credit is possible only through a learned $V(s_t)$ (PPO's critic): $V(s_{t+1})-V(s_t)$ can be $\neq0$.
- ❌Critic → only the sequence-level return is left → bandit.

*Then why do most implementations compute per-token quantities?*
- Grads flow through $\log\pi_\theta(y_t|x,y_{<t})$ per token regardless.
- Critic-free methods broadcast the sequence-level advantage to every token → same scalar, $|y|$ copies.
- → "Token-level" in those algorithms = loss aggregation, ❌token-level credit.

*What makes this setting easy vs classic RL?*
- Strong pretrained prior → ✅valid rollouts from step 0, ❌random exploration phase.
- Known deterministic dynamics → ❌model error.
- Cheap, parallel, resettable env → many rollouts per prompt.

*What makes it hard?*
- $|\mathcal{V}|^{|y|}$ effective action space.
- Credit assignment: 1 scalar for $10^3$–$10^5$ tokens.
- Reward is a proxy → [reward hacking](../rh.md).
- Rollouts dominate wall-clock → pressure to reuse each batch → off-policy drift.

*Why $\gamma=1$?*
- Terminal-only reward → $\gamma<1$ turns the objective into $\mathbb{E}[\gamma^{|y|-1}r(x,y)]$.
- → Implicit length penalty, unrelated to correctness.
- Length control belongs in the reward (e.g., DAPO's overlong shaping), ❌a side effect of $\gamma$.

*Why not give per-token rewards?*
- Needs a **process reward model** (PRM) that grades each step → expensive to label, easy to hack.
- Outcome-only reward stays the default ← cheap, verifiable, empirically sufficient.
```

&nbsp;

### Reward
- **What**: Scalar grade of a full response.
- **Why**: RL optimizes a number. "Good answer" is not a number.
    - The optimizer maximizes only what it gets → the reward source is the design decision.
- **How**: 3 sources, ordered by robustness to optimization pressure:
    1. **RLVR**: Programmatic checker → exact, only for verifiable domains.
    2. **RLHF**: Learned RM from human preferences → covers open-ended tasks, hackable.
    3. **RLAIF**: Learned RM from LM-generated preferences → scalable, inherits the judge's biases.

```{attention} Q&A
:class: dropdown
*Why is a learned RM hackable but a verifier is not?*
- RM = finite-capacity approximation of preference → off-distribution holes.
- Policy training pushes into those holes ← reward is highest there.
- Verifier = the ground-truth function itself → ❌approximation gap to exploit.
- ⚠️ Verifiers are still spec-hackable: special-cased tests, right answer w/ fabricated reasoning.

*Why not use RLVR everywhere?*
- Needs a cheap, unambiguous checker → math, code, structured output only.
- ❌Open-ended writing, safety, tone, helpfulness.

*Why keep reward bounded/normalized?*
- Unbounded reward → unbounded advantage → grad explosion.
- Verifier rewards are bounded by design ($\{0,1\}$, $\pm1$). RM scores are usually whitened per batch.
```

&nbsp;

#### RLHF
- **Name**: Reinforcement Learning from Human Feedback {cite:p}`christiano2017deep`
- **What**: Reward = learned RM fit on human pairwise preferences.
- **Why**: Most desirable behavior has ❌gold answer & ❌checker.
    - Humans cannot write the best response, but they can compare two.
- **How**:
    1. Collect $(x,y_w,y_l)$ triples from human labelers.
    2. Fit RM $r_\phi$ under **Bradley-Terry** → scalar score per response.
    3. RL-optimize the policy against $r_\phi$, KL-anchored to $\pi_\text{ref}$.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $y_w$: Preferred (won) response.
    - $y_l$: Dispreferred (lost) response.
    - $\mathcal{D}_\text{pref}$: Preference dataset of $(x,y_w,y_l)$ triples.
- Misc:
    - $\sigma$: Sigmoid.

Model:

$$
p(y_w\succ y_l|x)=\frac{\exp r(x,y_w)}{\exp r(x,y_w)+\exp r(x,y_l)}=\sigma\left(r(x,y_w)-r(x,y_l)\right)
$$

Training:

$$
\mathcal{L}_\text{RM}(\phi)=-\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}_\text{pref}}\left[\log\sigma\left(r_\phi(x,y_w)-r_\phi(x,y_l)\right)\right]
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
        self.head = nn.Linear(hidden, 1, bias=False)  ## bias cancels in r_w - r_l anyway

    def score(self, tokens, last_idx):
        h = self.backbone(tokens)                     ## (B, T, hidden)
        ## score the LAST token only -> one scalar per full response
        h_last = h[torch.arange(h.size(0)), last_idx]
        return self.head(h_last).squeeze(-1)          ## (B,)

    def forward(self, tok_w, last_w, tok_l, last_l):
        ## BT likelihood depends ONLY on the score difference
        return -F.logsigmoid(self.score(tok_w, last_w) - self.score(tok_l, last_l)).mean()

## Example
rm = BradleyTerryRM(nn.Embedding(10, 8), hidden=8)   ## toy backbone: (B, T) -> (B, T, 8)
tok_w, tok_l = torch.randint(0, 10, (2, 5)), torch.randint(0, 10, (2, 5))
last = torch.tensor([4, 2])                          ## idx of each response's final token
print(rm(tok_w, last, tok_l, last))                  ## scalar loss
```
````

```{attention} Q&A
:class: dropdown
*Who introduced what?*
- Preference RM + RL loop, on control tasks → Christiano et al. 2017.
- Ported to LMs → Ziegler et al. 2019.
- Scaled into the standard instruction-following recipe → InstructGPT {cite:p}`ouyang2022training`

*Why is the absolute RM score meaningless?*
- BT loss depends only on the difference $r(x,y_w)-r(x,y_l)$.
- → $r_\phi+c$ is an equally good RM ← the shift cancels.
- The scale does not cancel → it sharpens the sigmoid.
- → Never compare raw scores across RMs. Batch whitening is the usual fix.

*Why does the RM go stale during training?*
- RM was fit on samples from $\pi_\text{ref}$.
- Policy drifts → its samples leave the RM's training distribution → scores become unreliable.
- → The KL anchor also keeps the RM valid, ❌only a behavior constraint.

*Why pairwise instead of direct scoring?*
- Absolute ratings vary across labelers & drift within a session.
- Comparisons → lower variance, ❌shared scale needed.

*Cons?*
- Expensive & slow ← human labeling.
- Noisy ← inter-annotator disagreement.
- RM overoptimization: true quality peaks, then falls, while RM score keeps rising.
```

&nbsp;

#### RLAIF
- **Name**: Reinforcement Learning from AI Feedback {cite:p}`bai2022constitutional`
- **What**: RLHF w/ an LM as the preference labeler.
- **Why**: Human labels do not scale.
    - Cost & latency cap dataset size.
    - Humans are least reliable on long, technical, adversarial responses ← where labels matter most.
- **How**: Original pipeline = **Constitutional AI**:
    1. Write an explicit spec (a "constitution") of principles.
    2. LM critiques & revises its own responses against a sampled principle → SFT data.
    3. LM picks the preferred response of a pair, given a principle → preference data.
    4. Train RM on those labels → standard RLHF from there.

```{attention} Q&A
:class: dropdown
*Why does a model judging its own outputs work?*
- Evaluation is easier than generation ← recognizing a violation needs less capability than avoiding it.
- The spec is in-context at judging time, ❌at generation time → the judge has more information.

*Cons?*
- Inherits & amplifies judge biases: position, verbosity, self-preference.
- Blind spots correlate w/ the policy's own ← same model family.
- → In practice, mixed w/ human data, ❌full replacement.

*What is new vs RLHF?*
- Only the label source. RM training, policy optimization & KL anchor are unchanged.

*Is the RM or the constitution required?*
- ❌. Steps 1–2 are specific to Constitutional AI.
- **Direct RLAIF**: an LM scores each rollout during RL → ❌RM training, ❌RM staleness, ⬆️judge cost per step {cite:p}`lee2023rlaif`
```

&nbsp;

#### RLVR
- **Name**: Reinforcement Learning with Verifiable Rewards {cite:p}`lambert2024tulu`
- **What**: Reward = output of a deterministic checker.
- **Why**: Learned RMs get hacked. Reasoning has a ground truth.
    - Preference RM gives long CoT ❌useful signal ← "which proof looks nicer" ≠ "which proof is right".
- **How**:
    1. Dataset of prompts w/ checkable answers (final answer, unit tests).
    2. Sample response → extract answer → run checker.
    3. $r=1$ if pass, else $0$ (+ optional format reward).
    4. Optimize w/ any policy-gradient method.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\lambda$: Format reward weight.

Model:

$$
r(x,y)=\mathbb{1}[\text{ext}(y)=a^*]+\lambda\cdot\mathbb{1}[y\text{ matches the required format}]
$$
```

````{important} Code
:class: dropdown
```python
import re

class VerifiableReward:
    def __init__(self, lam=0.1):
        self.lam = lam  ## format reward weight

    def extract(self, y):
        m = re.search(r"<answer>(.*?)</answer>", y, re.S)
        return m.group(1).strip() if m else None

    def __call__(self, y, a_star):
        ## exact match for brevity; real verifiers test math equivalence or run unit tests
        correct = float(self.extract(y) == a_star)
        formatted = re.fullmatch(r"(?s)<think>.*</think>\s*<answer>.*</answer>", y.strip())
        return correct + self.lam * float(formatted is not None)

## Example
r = VerifiableReward()
print(r("<think>6*7</think><answer>42</answer>", "42"))  ## 1.1
print(r("The answer is 42.", "42"))                        ## 0.0
```
````

```{attention} Q&A
:class: dropdown
*Why did RLVR unlock reasoning models when RLHF did not?*
- RLHF ceiling = RM's judgment. RLVR ceiling = task's ground truth.
- → Safe to optimize much harder & longer w/o reward degradation.
- DeepSeek-R1-Zero: pure RLVR on a base model, ❌SFT warmup → long CoT & self-verification emerge {cite:p}`deepseekai2025deepseek`

*Why add a format reward?*
- Reliable extraction → parse failures stop corrupting the accuracy signal.
- Cheap to satisfy → the policy learns it early, then moves on.

*Cons?*
- Binary reward is sparse → w/ group baselines, 0 task grad when a group is all-pass or all-fail.
- Math, code, structured output only.
- Grades the final answer, ❌the reasoning → CoT can be post-hoc ([representation-level exploitation](../rh.md#representation-level-exploitation)).
- R1-Zero output: poor readability, language mixing → R1 added an SFT cold start.

*What breaks if the verifier is imperfect?*
- False negatives (equivalent answer, different format) punish correct reasoning → policy learns the format, ❌the math.
- → Verifier quality upper-bounds everything downstream.
```

&nbsp;

### KL Penalty
- **What**: Divergence term that anchors $\pi_\theta$ to $\pi_\text{ref}$.
- **Why**: Reward = proxy → unconstrained argmax of a proxy is degenerate.
    - ❌Constraint → collapse onto whatever the RM scores highest, gibberish included.
    - Drift invalidates the RM ← it only saw $\pi_\text{ref}$'s distribution.
    - Catastrophic forgetting of general capability.
- **How**:
    1. Estimate per-token KL between $\pi_\theta$ & $\pi_\text{ref}$ on sampled tokens.
    2. Subtract $\beta\cdot\text{KL}$ from the reward, or add it to the loss.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $u_t=\frac{\pi_\text{ref}(y_t|x,y_{<t})}{\pi_\theta(y_t|x,y_{<t})}$: Reference/policy prob ratio at one sampled token.

Objective:

$$
\tilde{r}(x,y)=r_\phi(x,y)-\beta\log\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}
$$
- $\tilde{r}$: KL-shaped reward (reward-level form).
- $\mathbb{E}_{y\sim\pi_\theta}$ of the log term $=\text{KL}[\pi_\theta\|\pi_\text{ref}]$.
- Single-sample estimators of the per-token KL, $y_t\sim\pi_\theta$:

    $$\begin{align*}
    k_1&=-\log u_t \\
    k_2&=\tfrac{1}{2}(\log u_t)^2 \\
    k_3&=u_t-\log u_t-1
    \end{align*}$$
    - $k_1$: Unbiased, can be $<0$, high variance when $\pi_\theta\approx\pi_\text{ref}$.
    - $k_2$: Biased, always $\geq0$, low variance when $\pi_\theta\approx\pi_\text{ref}$.
    - $k_3$: Unbiased, always $\geq0$, low variance when $\pi_\theta\approx\pi_\text{ref}$.
    - Far-apart policies → $k_3$'s variance can exceed $k_1$'s ← $u_t$ is unbounded.
    - Unbiasedness of $k_1,k_3$ requires $y_t\sim\pi_\theta$. Non-negativity of $k_2,k_3$ holds always.
```

```{tip} Derivation
:class: dropdown
*Why is $k_3$ both unbiased and non-negative?*

1. The added term has zero mean under $\pi_\theta$:

    $$
    \mathbb{E}_{y_t\sim\pi_\theta}[u_t-1]=\sum_{y_t}\pi_\theta\cdot\frac{\pi_\text{ref}}{\pi_\theta}-1=\sum_{y_t}\pi_\text{ref}-1=0
    $$

2. → $\mathbb{E}[k_3]=\mathbb{E}[-\log u_t]+\mathbb{E}[u_t-1]=\text{KL}[\pi_\theta\|\pi_\text{ref}]$.

3. Non-negativity ← tangent-line bound on the concave $\log$:

    $$
    \log u\leq u-1\ \Longrightarrow\ u-\log u-1\geq0\quad(\text{equality iff }u=1)
    $$

4. → $k_3$ = $k_1$ + zero-mean **control variate**: same mean.
    - $u_t\approx1$ → $-\log u_t\approx1-u_t$ → $(u_t-1)$ cancels most of $-\log u_t$'s fluctuation.
    - $u_t\gg1$ → no cancellation → variance can grow.
```

````{important} Code
:class: dropdown
```python
import torch

class KLEstimators:
    def __call__(self, logp, logp_ref):
        ## logp, logp_ref: (B, T) log-probs of the SAMPLED tokens (y_t ~ pi_theta)
        log_u = logp_ref - logp
        k1 = -log_u                          ## unbiased, can go negative
        k2 = 0.5 * log_u.pow(2)              ## biased, always >= 0
        k3 = log_u.exp() - log_u - 1         ## unbiased AND always >= 0
        return k1, k2, k3

## Example
logp, logp_ref = torch.tensor([[-1.0, -3.0]]), torch.tensor([[-1.5, -2.0]])
for name, v in zip("k1 k2 k3".split(), KLEstimators()(logp, logp_ref)):
    print(name, round(v.mean().item(), 4))  ## k1 -0.25  k2 0.3125  k3 0.4124
```
````

```{attention} Q&A
:class: dropdown
*Why is $k_1$ a problem in practice?*
- Per-sample $-\log u_t<0$ whenever $\pi_\theta(y_t)<\pi_\text{ref}(y_t)$.
- → Minibatch "KL" can go negative, though true KL $\geq0$ → noisy, sign-flipping penalty.

*Reward-level vs loss-level KL?*
- Reward-level: folded into $\tilde{r}$ → passes through the advantage estimator → the critic must model it.
- Loss-level: separate additive term → advantage stays pure task reward. GRPO uses this.

*Which direction is it?*
- Reverse KL, $\text{KL}[\pi_\theta\|\pi_\text{ref}]$ ← we can only sample from $\pi_\theta$.
- → Mode-seeking: dropping modes of $\pi_\text{ref}$ is cheap; mass where $\pi_\text{ref}\approx0$ is heavily penalized.
- → Partly explains RLHF's diversity loss.

*Does $k_3$ in the loss give the reverse-KL grad?*
- ❌. Per token, w/ the sampled $y_t$ held fixed: $\nabla_\theta k_3=(1-u_t)\nabla_\theta\log\pi_\theta(y_t)$.
- $\mathbb{E}_{\pi_\theta}[\nabla_\theta k_3]=\sum_a(\pi_\theta(a)-\pi_\text{ref}(a))\nabla_\theta\log\pi_\theta(a)=-\sum_a\pi_\text{ref}(a)\nabla_\theta\log\pi_\theta(a)$ ← $\sum_a\pi_\theta\nabla\log\pi_\theta=0$.
- → $=\nabla_\theta\text{KL}[\pi_\text{ref}\|\pi_\theta]$: the forward-KL grad.
- After the 1st update, samples come from $\pi_{\theta_\text{old}}$ → the on-policy unbiasedness no longer holds.
- → Loss-level KL = surrogate regularizer, ❌exact reverse-KL grad. Unbiased value ≠ unbiased grad.

*What policy maximizes the KL-regularized objective?*
- Closed form, per prompt:

    $$
    \pi^*(y|x)=\frac{1}{Z(x)}\pi_\text{ref}(y|x)\exp\left(\frac{r(x,y)}{\beta}\right)
    $$
    - $Z(x)$: Partition function.
- $\beta\to0$ → $\pi^*$ → argmax of $r$. $\beta\to\infty$ → $\pi^*\to\pi_\text{ref}$.
- → RLHF = reweight $\pi_\text{ref}$ by exponentiated reward. [DPO](sft.md#dpo) inverts this formula to remove the RM.

*When should you turn it off?*
- RLVR w/ a trustworthy verifier → the anchor mostly slows learning.
- DAPO & CISPO drop it. R1-Zero's stated GRPO objective keeps it.
- Keep it whenever the reward is a learned RM.

*What replaces it once removed?*
- The IS-ratio clip becomes the only trust region.
- The clip limits the incentive to move $\pi_\theta$ from $\pi_{\theta_\text{old}}$ (one batch), ❌from $\pi_\text{ref}$ (all of training).
- Even per batch, it is ❌hard bound: shared params & large steps can push ratios past $1\pm\epsilon$.
- → ❌Bound on cumulative drift. Acceptable only because drift alone cannot hack a verifier.
```

&nbsp;

### Training Loop
- **What**: Generate → grade → estimate advantage → update.
- **Why**: Training data does not exist until the curr policy generates it.
    - ❌Fixed dataset → every step needs fresh rollouts from a model that just changed.
- **How**: Per iteration,
    1. **Rollout**: Sample prompts. Generate $G$ responses per prompt from $\pi_{\theta_\text{old}}$.
    2. **Grade**: Score each response → $r(x,y_i)$.
    3. **Estimate**: Compute $\hat{A}$ (critic, or group statistics).
    4. **Update**: $\mu$ grad steps over the batch.
    5. $\pi_{\theta_\text{old}}\leftarrow\pi_\theta$. Sync weights into the inference engine.

```{dropdown} Table: Models in Memory
| Model | PPO | GRPO-family |
|:--|:--|:--|
| Policy $\pi_\theta$ (+ optimizer states) | ✅ | ✅ |
| Rollout policy $\pi_{\theta_\text{old}}$ | ✅ | ✅ |
| Reference $\pi_\text{ref}$ | ✅ | ✅/❌ (dropped w/ KL) |
| Critic $V_\psi$ (+ optimizer states) | ✅ | ❌ |
| RM $r_\phi$ | ✅ | ❌ w/ RLVR |
```

```{attention} Q&A
:class: dropdown
*Why $\pi_{\theta_\text{old}}\neq\pi_\theta$?*
- Generation dominates cost → each batch is reused for $\mu>1$ updates.
- After the 1st update, $\pi_\theta$ has moved, but the data still came from $\pi_{\theta_\text{old}}$.
- → **Semi-on-policy**: on-policy at the 1st step only, then increasingly off-policy.
- → The IS ratio corrects this mismatch. The clip limits it.

*What is $\mu$ in practice?*
- $\mu=1$ → fully on-policy, ratio $\equiv1$, clip = no-op, ⬆️cost.
- $\mu>1$ → cheaper, but ⬆️drift → trust-region design dominates stability.

*Why is the loop throughput-bound, ❌FLOP-bound?*
- Generation: autoregressive, memory-bandwidth-bound. Training: one dense parallel forward-backward.
- Long CoT (10k+ tokens) → rollout can dominate wall-clock.
- → Main levers: async/pipelined rollout, more batch reuse.

*Why does one slow rollout stall the whole step?*
- A batch is gradable only after its longest response finishes.
- → Straggler-bound. Length cap = throughput knob, ❌only a memory knob.
```

&nbsp;

## Algorithms
### REINFORCE
- **Name**: REward Increment = Nonnegative Factor × Offset Reinforcement × Characteristic Eligibility {cite:p}`williams1992simple`
- **What**: Log-prob grad weighted by reward. General form: [REINFORCE](../../rl/policy.md#reinforce).
- **Why**: $r$ is not differentiable w.r.t. $\theta$.
    - Grader = black box (string match, sandbox, RM) → ❌backprop through it.
    - Sampling $y\sim\pi_\theta$ is not differentiable.
    - → Need $\nabla_\theta$ of an expectation whose distribution depends on $\theta$.
- **How**:
    1. Sample $y\sim\pi_\theta(\cdot|x)$.
    2. Grade → $r(x,y)$.
    3. Ascend $\nabla_\theta\log\pi_\theta(y|x)$, scaled by $r$.
    4. Subtract a baseline $b(x)$ → same expected grad, ⬇️variance.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $b(x)$: Baseline, any function of $x$ alone.

Objective:

$$
\nabla_\theta J(\theta)=\mathbb{E}_{x\sim\mathcal{D},\ y\sim\pi_\theta(\cdot|x)}\left[\left(r(x,y)-b(x)\right)\nabla_\theta\log\pi_\theta(y|x)\right]
$$
- Bandit case of the [policy-gradient theorem](../../rl/policy.md#policy-gradient-theorem) ← log-derivative trick: $\nabla_\theta\pi_\theta=\pi_\theta\nabla_\theta\log\pi_\theta$.
- $b(x)$ adds $b(x)\nabla_\theta\sum_y\pi_\theta(y|x)=b(x)\nabla_\theta1=0$ → unbiased for any $b(x)$.
- Per-token expansion:

    $$
    \nabla_\theta\log\pi_\theta(y|x)=\sum_{t=1}^{|y|}\nabla_\theta\log\pi_\theta(y_t|x,y_{<t})
    $$
```

````{important} Code
:class: dropdown
```python
import torch

## Reuses ReinforceLoss from rl/policy.md
class REINFORCE:
    def __init__(self, decay=0.9):
        self.pg = ReinforceLoss(gamma=1.0)   ## gamma = 1 on this page
        self.decay = decay
        self.b = 0.0                         ## EMA of PREVIOUS batches' rewards

    def __call__(self, logp, rewards, mask):
        ## logp, mask: (B, T) sampled-token log-probs & validity; rewards: (B,)
        ## terminal reward & gamma = 1 -> every token's return = r(x, y)
        returns = rewards[:, None].expand_as(logp)
        baseline = torch.full_like(logp, self.b)  ## independent of this batch's y
        loss = self.pg(logp.T, returns.T, baseline.T, mask.T.bool())  ## (time, episodes) layout
        self.b = self.decay * self.b + (1 - self.decay) * rewards.mean().item()  ## update AFTER use
        return loss

## Example
logp = torch.randn(4, 3, requires_grad=True)
mask = torch.tensor([[1, 1, 1], [1, 1, 0], [1, 0, 0], [1, 1, 1]])
rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
print(REINFORCE()(logp, rewards, mask).item())
```
````

```{attention} Q&A
:class: dropdown
*How does this differ from SFT?*
- SFT: $-\nabla_\theta\log\pi_\theta(y^*|x)$ on a fixed, external $y^*$.
- REINFORCE: $-\hat{A}\nabla_\theta\log\pi_\theta(y|x)$ on the model's own sample $y$.
- → REINFORCE = SFT on self-generated data, signed & scaled by the advantage.
- $\hat{A}>0$ → imitate it. $\hat{A}<0$ → unlearn it. SFT has no second case.

*Why is variance the central problem?*
- 1 scalar supervises a sample from $|\mathcal{V}|^{|y|}$ possibilities.
- Reward offset is arbitrary → $r\in[99,101]$ pushes up every sample.
- → Baseline design is the algorithm. Every later method = a better $b$ + a better trust region.

*Must $b$ be the value function?*
- ❌. Any $b(x)$ independent of $y$ keeps the estimator unbiased.
- $b(x)=V(x)=\mathbb{E}[r|x]$: standard & good, but ❌variance-minimizing. The optimum is a grad-norm-weighted reward ([derivation](../../rl/policy.md#reinforce)).
- Group mean & leave-one-out estimate $V(x)$ from samples → same role, ❌critic.

*Why must the baseline not use this batch's own mean?*
- Batch mean includes $r(x,y_i)$ → $b$ depends on $y_i$ → the zero-mean proof fails.
- EMA of previous batches, or leave-one-out (→ RLOO), keeps $b$ independent.

*Why strictly on-policy?*
- The expectation is under $\pi_\theta$ itself.
- Reuse the batch after one update → $y$ came from a stale distribution → biased grad.
- → Importance sampling fixes it (PPO onward).
```

&nbsp;

#### RLOO
- **Name**: REINFORCE Leave-One-Out {cite:p}`kool2019buy`
- **What**: REINFORCE w/ baseline = mean reward of the other $G-1$ samples.
- **Why**: Critic = most expensive part of PPO, and this setting barely uses it.
    - Critic ≈ policy-sized → ~2× memory, own optimizer state, own loss to balance.
    - Terminal-only reward → critic estimates only "expected final score for this prompt" → $G$ samples estimate that directly.
- **How**:
    1. Sample $G$ responses per prompt.
    2. Baseline for $y_i$ = mean reward over $j\neq i$.
    3. Plain REINFORCE update: ❌ratio, ❌clip.

```{note} Math
:class: dropdown
Objective:

$$
\nabla_\theta J(\theta)=\mathbb{E}_{x\sim\mathcal{D},\ \{y_i\}_{i=1}^G\sim\pi_\theta(\cdot|x)}\left[\frac{1}{G}\sum_{i=1}^{G}\hat{A}_i\nabla_\theta\log\pi_\theta(y_i|x)\right]
$$
- Exact, ❌approximation ← $y_j$ independent of $y_i$ → $\mathbb{E}[r_j\nabla_\theta\log\pi_\theta(y_i|x)]=0$ for $j\neq i$.
- $\hat{A}_i$: Leave-one-out advantage:

    $$
    \hat{A}_i=r_i-\frac{1}{G-1}\sum_{j\neq i}r_j=\frac{G}{G-1}\left(r_i-\mu_G\right)
    $$
```

````{important} Code
:class: dropdown
```python
import torch

## Reuses ReinforceLoss from rl/policy.md
class RLOO:
    def __init__(self):
        self.pg = ReinforceLoss(gamma=1.0)

    @staticmethod
    def advantage(rewards):
        ## rewards: (P, G); exclude self from own baseline -> b independent of y_i
        G = rewards.size(1)
        loo_mean = (rewards.sum(1, keepdim=True) - rewards) / (G - 1)
        return rewards - loo_mean

    def __call__(self, logp, rewards, mask):
        ## logp, mask: (P*G, T); rewards: (P, G)
        adv = self.advantage(rewards).reshape(-1, 1).expand_as(logp)
        zero = torch.zeros_like(logp)  ## baseline already inside adv
        return self.pg(logp.T, adv.T, zero.T, mask.T.bool())

## Example
r = torch.tensor([[1.0, 0.0, 0.0, 1.0]])
print(RLOO.advantage(r))  ## tensor([[ 0.6667, -0.6667, -0.6667,  0.6667]])
```
````

```{attention} Q&A
:class: dropdown
*Leave-one-out vs subtracting the group mean?*
- $r_i$ in its own baseline → $b$ depends on $y_i$ → unbiasedness proof fails.
- The two differ only by the scalar $\frac{G}{G-1}$ → same direction, absorbed by the LR.
- → Practical gap is negligible. The theoretical guarantee is not.

*Why drop the clip?*
- Measured: PPO loss is clipped <5% of the time per batch in RLHF runs {cite:p}`ahmadian2024back`
- Strong SFT init + small updates → policy stays inside the trust region anyway.
- ⚠️ Long-CoT reasoning work concludes the opposite ← more steps, longer sequences, larger drift.

*Cons?*
- $G$ rollouts per prompt → generation cost $\times G$.
- ❌IS correction → valid only at $\mu=1$, the expensive regime.
- Sequence-level only → ❌knob for token-level loss aggregation.
```

&nbsp;

### PPO
- **Name**: Proximal Policy Optimization {cite:p}`schulman2017proximalpolicyoptimizationalgorithms`
- **What**: IS-reweighted policy grad w/ a clipped trust region. General form: [PPO](../../rl/policy.md#ppo).
- **Why**: Batch reuse.
    - REINFORCE discards each batch after 1 step, but rollouts dominate wall-clock → need $\mu>1$.
    - $\mu>1$ → batch is off-policy → plain grad is biased.
    - Unbounded off-policy step → policy lands where the data says nothing → collapse.
- **How**:
    1. Reweight each sampled token by its IS ratio.
    2. Clip the ratio into $[1-\epsilon,1+\epsilon]$. Take the `min` w/ the unclipped term.
    3. → Update overshoots in the direction $\hat{A}$ wants → token's grad = exactly 0.
    4. Estimate per-token $\hat{A}_t$ w/ GAE from a learned critic $V_\psi$.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\lambda$: GAE bias-variance coeff.

Objective:

$$
J_\text{PPO}(\theta)=\mathbb{E}_t\left[\min\left(\rho_t(\theta)\hat{A}_t,\ \text{clip}\left(\rho_t(\theta),1-\epsilon,1+\epsilon\right)\hat{A}_t\right)\right]
$$
- $\hat{A}_t$: [GAE](../../rl/policy.md#gae) w/ $\gamma=1$:

    $$
    \delta_t=r_t+V_\psi(s_{t+1})-V_\psi(s_t),\qquad \hat{A}_t=\sum_{l=0}^{|y|-t}\lambda^l\delta_{t+l}
    $$
    - $\delta_t$: TD residual.
    - $r_t=0$ for $t<|y|$. $r_{|y|}=r(x,y)$.
    - $V_\psi(s_{|y|+1})=0$: Terminal state.
- $V_\psi$: Fit by regression on return targets:

    $$
    \mathcal{L}_V(\psi)=\mathbb{E}_t\left[\left(V_\psi(s_t)-\hat{R}_t\right)^2\right]
    $$
    - $\hat{R}_t=\hat{A}_t+V_\psi(s_t)$: Return target, frozen at rollout time.
```

```{tip} Derivation
:class: dropdown
*What grad does the clipped objective produce?*

1. Unclipped branch selected → $\nabla_\theta(\rho_t\hat{A}_t)=\rho_t\hat{A}_t\nabla_\theta\log\pi_\theta(y_t|x,y_{<t})$.

2. Clipped branch selected → it has no $\theta$ → grad $=0$.

3. `min` selects the clipped branch iff it is smaller:
    - $\hat{A}_t>0$: $(1+\epsilon)\hat{A}_t<\rho_t\hat{A}_t$ iff $\rho_t>1+\epsilon$.
    - $\hat{A}_t<0$: $(1-\epsilon)\hat{A}_t<\rho_t\hat{A}_t$ iff $\rho_t<1-\epsilon$.

4. → Mask form:

    $$
    \nabla_\theta J_\text{PPO}=\mathbb{E}_t\left[M_t\,\rho_t\hat{A}_t\nabla_\theta\log\pi_\theta(y_t|x,y_{<t})\right],\quad M_t=\begin{cases}
    0 & (\hat{A}_t>0\wedge\rho_t>1+\epsilon)\vee(\hat{A}_t<0\wedge\rho_t<1-\epsilon) \\
    1 & \text{otherwise}
    \end{cases}
    $$

5. Masked terms still contribute $(1\pm\epsilon)\hat{A}_t$ to $J$, ❌$0$ → the two forms match in grad only.
```

````{important} Code
:class: dropdown
```python
import torch

## Reuses GAE & PPOClipLoss from rl/policy.md
class PPO:
    def __init__(self, eps=0.2, lam=0.95, value_weight=0.5):
        self.gae = GAE(gamma=1.0, trace_decay=lam)      ## gamma = 1 on this page
        self.clip = PPOClipLoss(eps, value_weight, entropy_weight=0.0)

    def advantage(self, values_old, reward):
        ## ONCE per rollout: values_old (B, T) from the rollout-time critic; reward (B,)
        ## equal lengths for brevity
        B, T = values_old.shape
        r = torch.zeros(T, B)
        r[-1] = reward                                  ## r_t = 0 until the last token
        v = values_old.detach().T                       ## GAE layout: (time, episodes)
        v_next = torch.cat([v[1:], torch.zeros(1, B)])  ## V(s_{|y|+1}) = 0
        end = torch.zeros(T, B, dtype=torch.bool)
        end[-1] = True                                  ## episode = one response
        adv, target = self.gae(r, v, v_next, end, end)
        return adv.T, target.T                          ## frozen for all mu updates

    def __call__(self, logp, logp_old, values, adv, target):
        ## EACH update: current logp & critic values vs frozen adv & target
        flat = lambda a: a.reshape(-1)
        n = logp.numel()
        return self.clip(flat(logp), flat(logp_old), flat(adv), flat(values),
                         flat(target), torch.zeros(n), torch.ones(n))

## Example
ppo = PPO()
values = torch.randn(2, 3, requires_grad=True)
adv, target = ppo.advantage(values, torch.tensor([1.0, 0.0]))
logp, logp_old = torch.randn(2, 3), torch.randn(2, 3)
print(ppo(logp, logp_old, values, adv, target).item())
```
````

````{dropdown} Table: What the Clip Does
| $\hat{A}_t$ | $\rho_t<1-\epsilon$ | $\rho_t\in[1-\epsilon,1+\epsilon]$ | $\rho_t>1+\epsilon$ |
|:--|:--|:--|:--|
| $>0$ (reinforce) | $\rho_t\hat{A}_t$ → ✅grad | $\rho_t\hat{A}_t$ → ✅grad | $(1+\epsilon)\hat{A}_t$ → const → ❌grad |
| $<0$ (suppress) | $(1-\epsilon)\hat{A}_t$ → const → ❌grad | $\rho_t\hat{A}_t$ → ✅grad | $\rho_t\hat{A}_t$ → ✅grad |

- Masked: 2 of 4 out-of-range cases ← policy already moved too far in the direction $\hat{A}_t$ asked for.
- Moving back toward $\pi_{\theta_\text{old}}$ → always allowed.
````

```{attention} Q&A
:class: dropdown
*Why `min`, ❌`clip` alone?*
- `clip` alone also zeros the grad when the ratio drifts against the advantage, i.e., while the policy undoes an earlier overshoot.
- `min` → $J$ = pessimistic lower bound on the unclipped objective → clipped branch wins only when smaller.
- → Overshoot: ❌grad. Recovery: full grad.

*Why exactly zero, ❌small?*
- Clipped branch has no $\theta$ → $\nabla_\theta$ of a constant.
- → The token is deleted from the update, ❌damped. CISPO attacks this property.

*What does GAE buy here?*
- $r_t=0$ for $t<|y|$ → $\delta_t=V_\psi(s_{t+1})-V_\psi(s_t)$ = critic's own opinion of how much $y_t$ moved the expected final score.
- Accurate critic → real per-token credit, which group baselines cannot give.
- Inaccurate critic → noise at every position.
- $\gamma=\lambda=1$ → sum telescopes to $\hat{A}_t=r(x,y)-V_\psi(s_t)$ → plain learned baseline.
- → $\lambda$ = dial between "trust the critic's per-token credit" and "trust only the final outcome".

*Why is the critic hard to train here?*
- Cold start ← random init (or new head) while the policy is already strong.
- Non-stationary target ← policy moves every step.
- Sparse target ← one scalar per response, credited to every prefix.

*What is $\epsilon$ in practice?*
- $0.2$: near-universal default, transfers well across models & tasks.
- Later methods change what $\epsilon$ constrains & its range:
    - DAPO: split into $0.2/0.28$.
    - GSPO: sequence ratio → range orders of magnitude smaller.
    - CISPO: bound on a weight, ❌a mask → much larger.

*Cons?*
- 4 large models resident: policy, rollout policy, reference, critic.
- Critic ≈ policy-sized → memory ceiling on scaling RL.
- Many coupled hyperparams: $\epsilon$, $\lambda$, $\beta$, value-loss coeff, value clip.
```

&nbsp;

### GRPO
- **Name**: Group Relative Policy Optimization {cite:p}`shao2024deepseekmath`
- **What**: PPO w/ the critic replaced by group-normalized rewards.
- **Why**: Critic ≈ half the memory & hard to train well.
    - Its job = prefix-level credit $V(s_t)$ → cold-started, non-stationary, trained from 1 scalar per response.
    - Without it, only the prompt-level baseline $V(x)$ is needed → $G$ samples estimate it, ❌learning.
- **How**:
    1. Sample $G$ responses per prompt from $\pi_{\theta_\text{old}}$.
    2. Grade → $\{r_1,\dots,r_G\}$.
    3. Standardize within the group → $\hat{A}_i$. Broadcast the same scalar to every token of $y_i$.
    4. PPO-style clipped update + explicit KL term in the loss.

```{note} Math
:class: dropdown
Objective:

$$
J_\text{GRPO}(\theta)=\mathbb{E}_{x\sim\mathcal{D},\ \{y_i\}_{i=1}^G\sim\pi_{\theta_\text{old}}(\cdot|x)}\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}\left(\min\left(\rho_{i,t}\hat{A}_i,\ \text{clip}\left(\rho_{i,t},1-\epsilon,1+\epsilon\right)\hat{A}_i\right)-\beta k_{3,i,t}\right)\right]
$$
- $\hat{A}_i$: Group-standardized advantage, shared by every token of $y_i$:

    $$
    \hat{A}_i=\frac{r_i-\mu_G}{\sigma_G}
    $$
- $k_{3,i,t}$: $k_3$ KL estimator ([KL Penalty](#kl-penalty)) at token $y_{i,t}$.
- Later variants (Dr. GRPO, DAPO, GSPO, CISPO) keep this expectation & write only $\mathbb{E}$.
```

````{important} Code
:class: dropdown
```python
import torch

## Reuses KLEstimators from KL Penalty above
class GRPO:
    def __init__(self, eps=0.2, beta=0.04, eps_std=1e-4):
        self.eps, self.beta, self.eps_std = eps, beta, eps_std
        self.kl = KLEstimators()

    def advantage(self, rewards):
        ## rewards: (P, G); guard: std = 0 when the group is unanimous
        mu, sd = rewards.mean(1, keepdim=True), rewards.std(1, keepdim=True)
        return (rewards - mu) / (sd + self.eps_std)

    def surrogate(self, logp, logp_old, adv, mask):
        ## adv: (B, 1) -> ONE scalar broadcast over every token of y_i
        ratio = (logp - logp_old).exp()
        return torch.min(ratio * adv, ratio.clamp(1 - self.eps, 1 + self.eps) * adv)

    def aggregate(self, per_token, mask):
        ## mean over tokens FIRST (the 1/|y_i|), then over responses -> length bias
        return ((per_token * mask).sum(-1) / mask.sum(-1)).mean()

    def __call__(self, logp, logp_old, rewards, mask, logp_ref=None):
        ## logp*, mask: (P*G, T) sampled-token log-probs; rewards: (P, G)
        adv = self.advantage(rewards).reshape(-1, 1)
        obj = self.surrogate(logp, logp_old, adv, mask)
        if self.beta > 0:
            obj = obj - self.beta * self.kl(logp, logp_ref)[2]  ## k3, in the LOSS
        return -self.aggregate(obj, mask)

## Example
r = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
lp, lp_old, lp_ref = torch.randn(4, 3), torch.randn(4, 3), torch.randn(4, 3)
mask = torch.tensor([[1., 1., 1.], [1., 1., 0.], [1., 0., 0.], [1., 1., 1.]])
print(GRPO().advantage(r))  ## tensor([[ 0.8659,  0.8659, -0.8659, -0.8659]])
print(GRPO()(lp, lp_old, r, mask, lp_ref).item())
```
````

```{attention} Q&A
:class: dropdown
*Why does a group of samples beat a learned critic?*
- They estimate different things. Critic: $V(s_t)$ per prefix. Group: only $V(x)$ per prompt.
- → GRPO trades prefix-level credit for a baseline that needs ❌learning & cannot be miscalibrated.
- Empirically the trade wins ← per-prefix critic estimates were mostly noise; $G\times$ generation parallelizes, $2\times$ memory does not.

*Why divide by $\sigma_G$?*
- One advantage scale across prompts & across mixed reward sources.
- ⚠️ ❌Free: it reweights prompts (→ Dr. GRPO).

*What happens when the group is unanimous?*
- All $r_i$ equal → $\hat{A}_i=0$ ∀$i$ → 0 task grad (only the KL grad remains if $\beta>0$). $\sigma_G=0$ → needs a guard.
- Binary reward → too-easy or too-hard prompts = wasted rollout compute.
- → Motivates DAPO's dynamic sampling.

*Why KL in the loss, ❌in the reward?*
- Reward-level KL passes through group standardization → the penalty is rescaled by $\sigma_G$, which is unrelated to it.
- Loss-level → $\hat{A}$ stays pure task reward & $\beta$ stays interpretable.

*What are $G$ & $\beta$ in practice?*
- DeepSeekMath: $G=64$, $\beta=0.04$. Later work: $G$ = 8–64, often $\beta=0$ w/ RLVR.
- ⬆️$G$ → ⬇️baseline variance & ⬆️chance of a mixed group. Generation cost grows linearly.

*Cons?*
- ❌Token-level credit ← one $\hat{A}_i$ broadcast over $|y_i|$ tokens.
- $\frac{1}{|y_i|}$ per-response averaging → length bias (→ Dr. GRPO).
- $\sigma_G$ division → difficulty bias (→ Dr. GRPO).
- Token-level $\rho_{i,t}$ → unequal token weights within one response (→ GSPO).
```

&nbsp;

#### Dr. GRPO
- **Name**: GRPO Done Right {cite:p}`liu2025understanding`
- **What**: GRPO minus both of its normalizations.
- **Why**: GRPO's two divisions look cosmetic but reweight the objective.
    - **Response-level length bias** ← divide by $|y_i|$:
        - $\hat{A}_i>0$ → shorter correct responses get larger per-token updates.
        - $\hat{A}_i<0$ → longer wrong responses get smaller per-token penalty.
        - → Longer = cheaper to be wrong → failed rollouts grow longer & longer.
    - **Question-level difficulty bias** ← divide by $\sigma_G$:
        - $\sigma_G$ is smallest when the prompt is nearly solved or nearly impossible.
        - → Those prompts' advantages are inflated the most.
- **How**:
    1. Drop $\sigma_G$ → $\hat{A}_i=r_i-\mu_G$.
    2. Drop $\frac{1}{|y_i|}$ → divide the summed token loss by a constant (generation budget).

```{note} Math
:class: dropdown
Objective:

$$
J_\text{Dr.GRPO}(\theta)=\mathbb{E}\left[\frac{1}{G\cdot L_\text{max}}\sum_{i=1}^{G}\sum_{t=1}^{|y_i|}\min\left(\rho_{i,t}\hat{A}_i,\ \text{clip}\left(\rho_{i,t},1-\epsilon,1+\epsilon\right)\hat{A}_i\right)\right]
$$
- $\hat{A}_i=r_i-\mu_G$: Centered advantage, ❌$\sigma_G$.
- $L_\text{max}$: Fixed for the whole run.
```

````{important} Code
:class: dropdown
```python
import torch

## Reuses GRPO from above; overrides only its two normalizations
class DrGRPO(GRPO):
    def __init__(self, l_max, eps=0.2):
        super().__init__(eps, beta=0.0)
        self.l_max = l_max

    def advantage(self, rewards):
        return rewards - rewards.mean(1, keepdim=True)  ## center only, NO /std

    def aggregate(self, per_token, mask):
        ## CONSTANT denominator -> pure LR rescale, cannot reweight responses
        return (per_token * mask).sum() / (mask.size(0) * self.l_max)

## Example
r = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
print(DrGRPO(l_max=8).advantage(r))  ## tensor([[ 0.2500,  0.2500,  0.2500, -0.7500]])
```
````

```{attention} Q&A
:class: dropdown
*Why does the length bias inflate response length?*
- Per-token grad magnitude $\propto\hat{A}_i/|y_i|$.
- Wrong & long → each token barely punished → long wrong answers are cheap to keep.
- → Length can grow while reasoning does not.

*So "longer responses = learning to reason" is wrong?*
- Partly. Bias removed → length stops growing wildly & wrong responses get much shorter, while reward still improves.
- → Length growth = real reasoning + optimizer artifact. Length alone is evidence of neither.

*Why is per-prompt $\sigma_G$ worse than batch-level whitening?*
- Batch whitening: one scale for all → relative prompt weights unchanged.
- Per-prompt: a different scale per prompt → reweighting that up-weights the lowest-information prompts.

*Why a constant denominator, ❌per-sequence?*
- GRPO's $\frac{1}{|y_i|}$ is per response → changes response weights by length. That is the bias.
- Constant → pure LR rescale → cannot reweight.
- vs DAPO's $\frac{1}{\sum_i|y_i|}$: also batch-wide → ❌within-batch reweighting. The two differ by the scalar $\frac{G\cdot L_\text{max}}{\sum_i|y_i|}$, i.e., across batches only.
- Other constants work too → they only shift grad norm.

*Cons?*
- Effective LR varies w/ how many tokens the batch holds.
- Unnormalized advantages → reward scale must be consistent across mixed task types.
```

&nbsp;

#### DAPO
- **Name**: Decoupled Clip and Dynamic sAmpling Policy Optimization {cite:p}`yu2025dapo`
- **What**: GRPO + 4 long-CoT fixes: clip-higher, dynamic sampling, token-level loss, overlong shaping.
- **Why**: Naive GRPO on long CoT fails in 3 ways.
    - **Entropy collapse** → policy sharpens early & stops exploring.
    - **Dead batches** → unanimous groups give $\hat{A}=0$ → growing share of the batch gives ❌grad.
    - **Truncation noise** → cut-off response scored the same as a wrong one.
    - (+ GRPO's length bias, same diagnosis as Dr. GRPO.)
- **How**:
    1. **Clip-Higher**: Decouple the bounds, $\epsilon_\text{low}=0.2$, $\epsilon_\text{high}=0.28$.
    2. **Dynamic Sampling**: Over-sample. Drop unanimous groups. Refill until the batch is full.
    3. **Token-Level Loss**: Divide by the batch's total token count, ❌per-sequence mean.
    4. **Overlong Reward Shaping**: Soft penalty ramp before the hard length cutoff.
    5. Drop the KL term.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\epsilon_\text{low}$: Lower clip bound.
    - $\epsilon_\text{high}$: Upper clip bound.
    - $L_\text{cache}$: Width of the soft penalty zone below $L_\text{max}$.

Objective:

$$
J_\text{DAPO}(\theta)=\mathbb{E}\left[\frac{1}{\sum_{i=1}^{G}|y_i|}\sum_{i=1}^{G}\sum_{t=1}^{|y_i|}\min\left(\rho_{i,t}\hat{A}_i,\ \text{clip}\left(\rho_{i,t},1-\epsilon_\text{low},1+\epsilon_\text{high}\right)\hat{A}_i\right)\right]
$$
- $\hat{A}_i$: Same as GRPO (keeps $\sigma_G$).
- Batch form (code below): one denominator = total tokens over all responses of all prompts in the batch.
- Subject to (dynamic sampling):

    $$
    0<\left|\{y_i:\text{ext}(y_i)=a^*\}\right|<G
    $$
- Reward adds $R_\text{length}(y)$ (overlong shaping):

    $$
    R_\text{length}(y)=\begin{cases}
    0 & |y|\leq L_\text{max}-L_\text{cache} \\
    \frac{(L_\text{max}-L_\text{cache})-|y|}{L_\text{cache}} & L_\text{max}-L_\text{cache}<|y|\leq L_\text{max} \\
    -1 & |y|>L_\text{max}
    \end{cases}
    $$
```

````{important} Code
:class: dropdown
```python
import torch

## Reuses GRPO from above
class DAPO(GRPO):
    def __init__(self, eps_low=0.2, eps_high=0.28):
        super().__init__(eps_low, beta=0.0)  ## KL term dropped
        self.eps_high = eps_high

    def surrogate(self, logp, logp_old, adv, mask):
        ratio = (logp - logp_old).exp()
        ## asymmetric: more room to RAISE a token than to crush it
        return torch.min(ratio * adv, ratio.clamp(1 - self.eps, 1 + self.eps_high) * adv)

    def aggregate(self, per_token, mask):
        ## token-level: one denominator for the whole batch
        return (per_token * mask).sum() / mask.sum()

    @staticmethod
    def keep_group(correct):
        ## correct: (G,) bool; DAPO's reward is +1/-1 -> test OUTCOMES, not the reward sum
        return 0 < int(correct.sum()) < correct.numel()

    @staticmethod
    def overlong_penalty(length, l_max, l_cache):
        if length <= l_max - l_cache:
            return 0.0
        if length <= l_max:
            return ((l_max - l_cache) - length) / l_cache  ## linear ramp 0 -> -1
        return -1.0

## Example
print(DAPO.keep_group(torch.tensor([True, True, False, False])))  ## True
print(DAPO.keep_group(torch.tensor([True, True, True, True])))    ## False
print(DAPO.overlong_penalty(18, l_max=20, l_cache=4))             ## -0.5
```
````

```{attention} Q&A
:class: dropdown
*Why raise only the upper bound?*
- Ratio is multiplicative → the clip saturates the incentive at a relative change.
- $p=0.01$, $\epsilon=0.2$ → incentive stops at $0.012$. $p=0.9$ → stops at $1.08$ → never binds.
- → Symmetric clip limits the push on rare tokens, ❌on likely ones.
- ⚠️ Saturation threshold, ❌hard prob bound ← shared params can still move $p$ further.
- Rare tokens = reasoning forks ("However", "Wait", "Recheck") that long CoT needs.

*Why not raise $\epsilon_\text{low}$ too?*
- The lower bound stops a token being crushed toward 0 in one batch.
- Prob ≈ 0 → never sampled → never gets grad → effectively unrecoverable.

*How does clip-higher fix entropy collapse?*
- Up-clipping limits how much rare tokens gain per batch; likely tokens are unconstrained → sharpening pressure.
- Relaxed upper bound → rare tokens gain more → entropy stays higher.
- ⚠️ Empirical result (DAPO), ❌guarantee. Evidence: mean prob of up-clipped tokens is low.

*Why is dynamic sampling not free?*
- Regenerate until mixed → variable, sometimes much larger, rollout cost per step.
- Better step efficiency, worse wall-clock per step.
- Cheaper alternative: drop unanimous groups & accept a smaller effective batch.

*Why shape truncation?*
- Default: truncated = wrong → a nearly-correct 32k trace is a failure.
- → Reward label noise, concentrated on the hardest prompts.
- Ramp separates "too long" from "incorrect" → policy learns length control separately.

*DAPO token-level average vs Dr. GRPO constant?*
- Both delete GRPO's per-response $\frac{1}{|y_i|}$ → response weight $\propto$ its length. Numerators identical.
- Differ only by the batch-wide scalar $\frac{G\cdot L_\text{max}}{\sum_i|y_i|}$ → per-batch rescale, ❌within-batch reweighting.
- DAPO: per-token mean → grad scale invariant to batch token count; denominator is random.
- Dr. GRPO: fixed denominator → ❌random rescaling; grad norm grows w/ total tokens.

*Cons?*
- Keeps $\sigma_G$ → keeps the difficulty bias Dr. GRPO argues against.
- Keeps PPO's hard masking → clipped tokens still contribute nothing (→ CISPO).
- 4 extra hyperparams.
```

&nbsp;

#### GSPO
- **Name**: Group Sequence Policy Optimization {cite:p}`zheng2025group`
- **What**: GRPO w/ the likelihood ratio defined, clipped & applied per sequence.
- **Why**: Ratio unit ≠ reward unit.
    - Reward is per sequence. $\rho_{i,t}$ constrains one token.
    - → Per-token ratios weight tokens unequally at fixed $\hat{A}_i$ → noise accumulates over $|y_i|$ positions, amplified by clipping.
    - Worst in MoE ← routing is discrete → small weight change flips experts → $\rho_{i,t}$ swings for reasons unrelated to reward.
- **How**:
    1. Ratio = sequence likelihood ratio ^ $\frac{1}{|y_i|}$ (geometric mean of token ratios).
    2. Clip that ratio, ❌$\rho_{i,t}$ → clip unit = whole response.
    3. → Every token in a kept response gets the same weight.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $s_i(\theta)$: Length-normalized sequence likelihood ratio. Overrides $s_t$ (state) in this section only.

Objective:

$$
J_\text{GSPO}(\theta)=\mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\min\left(s_i(\theta)\hat{A}_i,\ \text{clip}\left(s_i(\theta),1-\epsilon,1+\epsilon\right)\hat{A}_i\right)\right]
$$
- $s_i(\theta)$: Geometric mean of token ratios:

    $$
    s_i(\theta)=\left(\frac{\pi_\theta(y_i|x)}{\pi_{\theta_\text{old}}(y_i|x)}\right)^{\frac{1}{|y_i|}}=\exp\left(\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}\log\rho_{i,t}(\theta)\right)
    $$
- $\hat{A}_i$: GRPO's group-standardized advantage, unchanged.
```

```{tip} Derivation
:class: dropdown
*How does the grad differ from GRPO's?* (clipping & KL omitted)

1. GRPO: $\nabla_\theta\rho_{i,t}=\rho_{i,t}\nabla_\theta\log\pi_\theta(y_{i,t}|x,y_{i,<t})$ →

    $$
    \nabla_\theta J_\text{GRPO}=\mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\hat{A}_i\cdot\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}\rho_{i,t}(\theta)\nabla_\theta\log\pi_\theta(y_{i,t}|x,y_{i,<t})\right]
    $$

2. GSPO: $\nabla_\theta s_i=s_i\cdot\frac{1}{|y_i|}\sum_t\nabla_\theta\log\pi_\theta(y_{i,t}|x,y_{i,<t})$ ← chain rule on $\exp(\text{mean log-ratio})$ →

    $$
    \nabla_\theta J_\text{GSPO}=\mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}s_i(\theta)\hat{A}_i\cdot\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}\nabla_\theta\log\pi_\theta(y_{i,t}|x,y_{i,<t})\right]
    $$

3. → GRPO: per-token weight $\rho_{i,t}$. GSPO: one weight $s_i$ for all tokens of $y_i$.
```

````{important} Code
:class: dropdown
```python
import torch

## Reuses GRPO from above; overrides only the ratio
class GSPO(GRPO):
    def __init__(self, eps=3e-4):
        super().__init__(eps, beta=0.0)

    def surrogate(self, logp, logp_old, adv, mask):
        ## geometric mean of token ratios == exp of the mean log-ratio
        log_s = ((logp - logp_old) * mask).sum(-1, keepdim=True) / mask.sum(-1, keepdim=True)
        s = log_s.exp()                                  ## (B, 1): ONE ratio per response
        per_seq = torch.min(s * adv, s.clamp(1 - self.eps, 1 + self.eps) * adv)
        return per_seq.expand_as(logp)                   ## GRPO's per-seq mean returns it unchanged

## Example
r = torch.tensor([[1.0, 0.0]])
lp, lp_old, mask = torch.randn(2, 5), torch.randn(2, 5), torch.ones(2, 5)
print(GSPO()(lp, lp_old, r, mask).item())
```
````

```{attention} Q&A
:class: dropdown
*What changes in the grad?*
- GRPO: token $t$ weighted by its own $\rho_{i,t}$ → unequal weights within one response.
- GSPO: every token weighted by the same $s_i$ → equal weights.
- → Both use one advantage per response. GSPO stops adding per-token noise on top.

*Is $s_i$ a valid importance weight?*
- ❌. Valid sequence-level IS weight = $\frac{\pi_\theta(y_i|x)}{\pi_{\theta_\text{old}}(y_i|x)}$. The $\frac{1}{|y_i|}$ root breaks that identity.
- $s_i$ = trust-region statistic: length-comparable measure of how far the response's likelihood moved, ❌unbiased off-policy correction.
- ⚠️ The paper says a single-token ratio "fails to perform distribution correction". One-sample IS is still unbiased in expectation → that argument is heuristic.
- Defensible claims: unit mismatch, variance accumulation, MoE evidence.

*Why must $\epsilon$ be far smaller than PPO's $0.2$?*
- $s_i$ = geometric mean over $|y_i|$ token ratios → concentrates tightly around 1.
- $0.2$ window → almost never binds.
- Paper: $3\times10^{-4}$ lower / $4\times10^{-4}$ upper.
- → Clip ranges are ❌comparable across GSPO & GRPO; they bound different quantities.

*What does a clip mean now?*
- Token-level clip drops single tokens. Sequence-level clip drops the whole response.
- → Coarser, but trust-region unit = reward unit.

*Why does it stabilize MoE RL?*
- Routing flips between $\pi_{\theta_\text{old}}$ & $\pi_\theta$ → per-token likelihoods jump under tiny weight changes.
- Averaging log-ratios over the sequence smooths those spikes.
- → ❌Need to cache & replay $\pi_{\theta_\text{old}}$'s expert routes (Routing Replay).

*Why does it simplify infrastructure?*
- Only sequence likelihood matters → tolerates small per-token numeric differences between inference & training engines.
- → Directly reduces [train-inference mismatch](#train-inference-mismatch).

*Cons?*
- Whole-response masking → one pathological token can discard a full rollout's grad.
- ❌Token-level advantage. The paper's GSPO-token variant restores it; equal token advantages → identical to GSPO.
```

&nbsp;

#### CISPO
- **Name**: Clipped IS-weight Policy Optimization {cite:p}`minimax2025minimax`
- **What**: REINFORCE w/ a clipped, stop-grad IS weight, ❌masking.
- **Why**: PPO-style clipping deletes the tokens most worth learning from.
    - Clipped in a masked direction → constant branch → 0 grad → token gone for the whole batch.
    - Tokens that hit the upper clip = low-prob ones ("However", "Recheck", "Wait") = reasoning forks.
    - Long CoT → few such tokens per trace → deleting them deletes the signal.
- **How**:
    1. Compute $\rho_{i,t}$. Clip it → $\hat{\rho}_{i,t}$.
    2. Wrap it in $\text{sg}[\cdot]$ → constant multiplier, ❌function of $\theta$.
    3. Multiply onto a plain $\log\pi_\theta$ term → every token keeps a grad; only its magnitude is capped.
    4. Tune only $\epsilon^\text{IS}_\text{high}$. Set $\epsilon^\text{IS}_\text{low}$ so large it never binds.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\epsilon^\text{IS}_\text{low}$: Lower IS-weight clip bound.
    - $\epsilon^\text{IS}_\text{high}$: Upper IS-weight clip bound.

Objective:

$$
J_\text{CISPO}(\theta)=\mathbb{E}\left[\frac{1}{\sum_{i=1}^{G}|y_i|}\sum_{i=1}^{G}\sum_{t=1}^{|y_i|}\text{sg}\left[\hat{\rho}_{i,t}(\theta)\right]\hat{A}_i\log\pi_\theta(y_{i,t}|x,y_{i,<t})\right]
$$
- $\hat{\rho}_{i,t}$: Clipped IS weight:

    $$
    \hat{\rho}_{i,t}(\theta)=\text{clip}\left(\rho_{i,t}(\theta),1-\epsilon^\text{IS}_\text{low},1+\epsilon^\text{IS}_\text{high}\right)
    $$
- $\hat{A}_i$: GRPO's group-standardized advantage.
- Batch form (code below): one denominator over all prompts in the batch, as in DAPO.
```

```{tip} Derivation
:class: dropdown
*Why is this the same as PPO inside the trust region, and different outside?*

1. PPO's unclipped branch, w/ $\nabla_\theta\rho_t=\rho_t\nabla_\theta\log\pi_\theta$:

    $$
    \nabla_\theta\left(\rho_t\hat{A}_t\right)=\rho_t\hat{A}_t\nabla_\theta\log\pi_\theta
    $$

2. PPO's clipped branch is constant in $\theta$ → $\nabla_\theta=0$.

3. CISPO, w/ $\text{sg}[\hat{\rho}_t]$ as a constant:

    $$
    \nabla_\theta\left(\text{sg}[\hat{\rho}_t]\hat{A}_t\log\pi_\theta\right)=\hat{\rho}_t\hat{A}_t\nabla_\theta\log\pi_\theta
    $$

4. $\rho_t$ inside the clip range → $\hat{\rho}_t=\rho_t$ → identical to PPO.

5. $\rho_t$ outside, in one of the 2 masked cases ($\hat{A}_t>0\wedge\rho_t>1+\epsilon$, or $\hat{A}_t<0\wedge\rho_t<1-\epsilon$) → PPO gives $0$. CISPO keeps the grad w/ weight $\hat{\rho}_t$.

6. $\rho_t$ outside, in the 2 recovery cases:
    - $\hat{A}_t<0\wedge\rho_t>1+\epsilon^\text{IS}_\text{high}$ → PPO weight $\rho_t$, CISPO caps it at $1+\epsilon^\text{IS}_\text{high}$ → more conservative.
    - $\hat{A}_t>0\wedge\rho_t<1-\epsilon^\text{IS}_\text{low}$ → a lower clip would floor the weight above $\rho_t$ → amplify. CISPO's huge $\epsilon^\text{IS}_\text{low}$ disables it → same as PPO.

7. → CISPO replaces PPO's delete w/ a saturate.
```

````{important} Code
:class: dropdown
```python
import torch

## Reuses DAPO from above (token-level aggregation, no KL)
class CISPO(DAPO):
    def __init__(self, eps_high, eps_low=1e9):
        super().__init__(eps_low, eps_high)  ## huge eps_low -> lower bound never binds

    def surrogate(self, logp, logp_old, adv, mask):
        ratio = (logp - logp_old).exp()
        w = ratio.clamp(1 - self.eps, 1 + self.eps_high).detach()  ## sg[] on the WEIGHT
        ## REINFORCE term -> every token keeps a grad, only its size is capped
        return w * adv * logp

## Example
r = torch.tensor([[1.0, 0.0]])
lp = torch.randn(2, 5, requires_grad=True)
lp_old, mask = torch.randn(2, 5), torch.ones(2, 5)
CISPO(eps_high=5.0)(lp, lp_old, r, mask).backward()
print((lp.grad != 0).all())  ## tensor(True): no token is masked out
```
````

```{attention} Q&A
:class: dropdown
*Why stop-grad the weight?*
- ❌$\text{sg}$ → grad also flows through $\hat{\rho}_t$ → adds a term that is not a policy grad.
- IS weights correct the sampling distribution, ❌part of the objective → must be constants.

*Is it unbiased?*
- ❌. Weight clipping biases the grad; the paper says so.
- Trade: small bias, but ❌information loss & lower variance than the unclipped weight.
- Remove the clip → plain token-ratio-weighted policy grad.

*Why only the upper bound?*
- Variance blow-up comes from weights $\gg1$ ← tokens whose prob rose sharply under $\pi_\theta$ → cap those.
- A lower bound = floor → raises the weight of tokens whose prob collapsed → re-amplifies samples the policy correctly abandons.
- → ❌Variance gain, only distortion.

*How does this compare to DAPO's clip-higher?*
- Same diagnosis (rare tokens suppressed), different remedy.
- DAPO widens the window where grads survive. CISPO removes the window & caps magnitude instead.
- Reported: matches DAPO's AIME'24 accuracy in ~50% of the training steps, on Qwen2.5-32B-base.

*Cons?*
- ❌Trust region in the PPO sense → nothing bounds one update's step. Stability rests on the weight cap only.
- Biased grad.
- $\epsilon^\text{IS}_\text{high}$ = sensitive knob. MiniMax lowered it (w/ the grad-clipping threshold) to stop late-training degeneration.
```

&nbsp;

## Practice
### Entropy Collapse
- **What**: Policy entropy → 0 early in training.
- **Why**: RL on a pretrained LM is nearly pure exploitation.
    - Policy starts competent → reward maximization sharpens it onto what already works, ❌explores.
    - Symmetric clip is asymmetric in prob space → push on rare tokens saturates early; likely ones are unconstrained.
    - Masking removes clipped tokens → the tokens that would raise entropy contribute nothing.
- **How**: Monitor it as a first-class metric. Widen the upward path.
    - **Clip-Higher** (DAPO) → more headroom for rare tokens.
    - Saturate, ❌mask (CISPO) → rare tokens never dropped.
    - Entropy bonus / entropy target → direct, but fragile.
    - Track policy entropy & mean generation prob together → they move in opposite directions.

```{attention} Q&A
:class: dropdown
*Why is it self-reinforcing?*
- ⬇️Entropy → group samples become near-duplicates → $\sigma_G\to0$ & $\hat{A}_i\to0$.
- → ⬇️Grad → nothing pushes entropy back up.
- Only exploration cures it, and exploration is what was lost.

*Isn't low entropy just confidence?*
- Confidence after the policy is good → fine. Collapse before it is good → terminal.
- Diagnostic: entropy ⬇️ & reward plateaus → collapse. Reward ⬆️ & entropy flat or slowly ⬆️ → healthy.

*Is more entropy always better?*
- ❌. Too high → gibberish, repetition, degenerate loops.
- → Entropy = a band to stay inside, ❌a quantity to maximize.

*Why is an entropy bonus fragile?*
- Rewards uncertainty everywhere, including tokens the policy has correctly resolved.
- Coeff too low → no effect. Too high → model is paid to be vague.
- → Structural fixes (clip bounds, no masking) act only where suppression happens.

*How does this connect to the pass@1 vs pass@$k$ gap?*
- RL maximizes expected reward = pass@1.
- Per prompt, $\text{pass@}k=1-(1-p)^k$ ⬆️ w/ $p$ → ❌regression on one prompt alone.
- Across prompts: sharpening raises $p$ on solvable prompts but can drive $p\to0$ on rare-success prompts.
- → Benchmark pass@$k$ at large $k$ can fall below the base model. Entropy collapse is one proposed mechanism, ❌proven sole cause.
```

&nbsp;

### Train-Inference Mismatch
- **What**: Token probs from the inference engine $\neq$ recomputed training probs, at identical weights. {cite:p}`minimax2025minimax`
- **Why**: Rollout & training are different programs.
    - Rollout: fused/paged attention kernels, low precision, different batching → different reduction order.
    - Training: standard forward pass.
    - → Same $\theta$, numerically different logits.
    - $\rho_{i,t}$ is a ratio → tiny absolute error = large relative error where $\pi$ is small.
    - → Run is silently off-policy from step 0. Worst case: reward never grows.
- **How**:
    1. **Diagnose**: Scatter inference vs training probs per token. Correlation should be ≈1.
    2. **Fix numerics**: Compute the LM output head in FP32.
    3. **Or correct it**: Use rollout log-probs as $\pi_{\theta_\text{old}}$ → IS ratio corrects the gap.
    4. **Or reduce exposure**: Sequence-level ratio (GSPO) averages the per-token noise.

```{attention} Q&A
:class: dropdown
*Why is this worse than ordinary off-policy drift?*
- Drift: systematic, points in the intended update direction. The clip does not hard-bound it, but removes the incentive to push further.
- Mismatch: noise, uncorrelated w/ the update direction, unbounded in relative terms.
- Largest on low-prob tokens ← the ones clip-higher & CISPO protect.

*Why did it surface only recently?*
- Long CoT → errors compound over $10^4$ positions.
- Large & MoE models → high-magnitude activations at the output layer + discrete routing.
- MiniMax: ❌seen in smaller dense softmax-attention models.

*Does the FP32 head fully solve it?*
- Mostly: reported correlation ~0.9x → ~0.99x, stable across training.
- ❌Exact. A residual gap remains → algorithm-side robustness still helps.

*Why does recomputing $\pi_{\theta_\text{old}}$ w/ the training kernel hide the bug?*
- Then $\rho\equiv1$ at the 1st update by construction → looks on-policy.
- Data actually came from the inference distribution → mismatch = uncorrected bias, ❌visible quantity.
- → $\rho\approx1$ at step 0 ≠ evidence of no mismatch.

*Why is this a warning about RL results in general?*
- Silent numerics bug & real algorithmic difference look identical on the reward curve.
- → Method comparisons are valid only after train-inference correlation is verified for every arm.
```

&nbsp;

### Design Space
- **What**: The 4 knobs every post-GRPO method turns.
- **Why**: Methods differ by small deltas → the deltas are the whole comparison.
- **How**:
    1. **Baseline**: Critic / group mean / leave-one-out; divide by $\sigma$ or not.
    2. **Trust-region unit**: Token ratio / sequence ratio.
    3. **Trust-region action**: Mask (delete the grad) / cap the weight (keep it).
    4. **Loss aggregation**: Per-sequence mean / batch-token mean / fixed constant.

````{dropdown} Table: Algorithms at a Glance
| Method | Advantage | Trust region | Out-of-range | Aggregation | Delta |
|:--|:--|:--|:--|:--|:--|
| REINFORCE | $r-b(x)$ | ❌ | — | Sample mean | Policy grad |
| RLOO | $r_i-\frac{1}{G-1}\sum_{j\neq i}r_j$ | ❌ | — | Sample mean | ❌Critic, unbiased baseline |
| PPO | GAE w/ critic $V_\psi$ | Token $\rho_t$, $\pm\epsilon$ | Mask | Sample mean | Batch reuse w/o collapse |
| GRPO | $(r_i-\mu_G)/\sigma_G$ | Token $\rho_{i,t}$, $\pm\epsilon$ | Mask | $\frac{1}{\|y_i\|}$ per sample | ❌Critic → ⬇️memory |
| Dr. GRPO | $r_i-\mu_G$ | Token $\rho_{i,t}$, $\pm\epsilon$ | Mask | $\frac{1}{L_\text{max}}$ const | ❌Length bias, ❌difficulty bias |
| DAPO | $(r_i-\mu_G)/\sigma_G$ | Token $\rho_{i,t}$, $[-\epsilon_l,+\epsilon_h]$ | Mask | $\frac{1}{\sum_i\|y_i\|}$ | Room for rare tokens, ❌dead batches |
| GSPO | $(r_i-\mu_G)/\sigma_G$ | Sequence $s_i$, $\pm\epsilon$ | Mask (whole response) | $\frac{1}{G}$ per sample | Ratio unit = reward unit |
| CISPO | $(r_i-\mu_G)/\sigma_G$ | Token $\hat{\rho}_{i,t}$, upper only | Cap (sg) | $\frac{1}{\sum_i\|y_i\|}$ | ❌Grad deletion |

- $\epsilon_l$: Lower clip bound (DAPO's $\epsilon_\text{low}$).
- $\epsilon_h$: Upper clip bound (DAPO's $\epsilon_\text{high}$).
````

```{attention} Q&A
:class: dropdown
*What has the field converged on?*
- ❌Critic. Every method after PPO drops it. Sample baselines match it at ~half the memory.
- ❌Per-response token averaging. GRPO's $\frac{1}{|y_i|}$ = hidden length reweighting. Dr. GRPO, DAPO & CISPO remove it. GSPO keeps per-response weighting by design (sequence-level ratio).
- $\sigma_G$: contested, ❌settled. Dr. GRPO shows it up-weights the least informative prompts; DAPO, GSPO, CISPO keep it.
- Trust region: open. $\epsilon=0.2$ holds up well, yet DAPO, GSPO & CISPO each attack a different part of it.

*Which one should I run?*
- Verifiable reward, ordinary dense model → GRPO + Dr. GRPO's two fixes = strong, simple default.
- Long CoT, entropy dying → DAPO's clip-higher, or CISPO.
- MoE, or collapse w/o visible cause → GSPO.
- Learned RM in the loop → keep the KL anchor, whatever the objective.

*What is still unsolved?*
- Credit assignment: 1 scalar for $10^3$–$10^5$ tokens. The token that broke the proof scores the same as nearby whitespace.
- Sample efficiency: Binary verifier → ~1 bit per rollout; 8–64 rollouts per prompt to build a baseline.
- Never-solved prompts: $r_i=0$ ∀$i$ → $\hat{A}=0$ → ❌grad, ever. Curriculum is a workaround, ❌fix.
- Domains w/o a verifier: Progress is mostly math & code. Noisy, delayed, subjective, multi-turn rewards stay hard.
- Reproducibility: Most papers report one model family, verifier, data mix & compute budget. An intervention can change early speed, final ceiling, or both.

*Why is a single-checkpoint comparison misleading?*
- Two methods can cross: the faster early learner can plateau below the slower one.
- → Compare reward-vs-compute curves, ❌single numbers.

*What does this say about intelligence?*
- Pretraining supplies the hypothesis space. RL only reweights it → every method here shifts mass among behaviors the base model can already produce.
- Open problem = credit assignment: attribute one outcome to the internal step that caused it.
- → An agent that cannot localize its own error must brute-force the search → 8–64 rollouts per prompt.
```

&nbsp;
