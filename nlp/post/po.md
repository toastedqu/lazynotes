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
# Preference Optimization
- **What**: Fitting a policy to pairwise comparisons w/ a supervised loss.
- **Why**: SFT cannot express "this is better than that".
    - CE raises the demonstrated response & drains the rest only through normalization → ❌way to name a **specific** response as the worse one.
    - Quality is unratable in absolutes but reliably rankable in pairs → the cheap label is a comparison.
    - The [RLHF](rl.md#rlhf) route buys the same signal at the cost of a second model plus a rollout loop.
- **How**:
    1. Collect $(x,y_w,y_l)$ triples.
    2. Define an **implicit reward** as a function of the policy itself → ❌separate RM.
    3. Plug it into a preference likelihood → a classification loss.
    4. Minimize it on the fixed dataset. ❌Sampling, ❌reward model, ❌rollout loop.

```{attention} Q&A
:class: dropdown
*Where does preference data come from?*
- Human labelers comparing 2 responses → expensive, the RLHF standard.
- LM judge → scalable, inherits the judge's biases.
- Constructed → correct vs incorrect answers, strong-model vs weak-model responses, edited vs original.

*Why do on-policy pairs beat off-policy pairs?*
- The loss pushes $\log\pi_\theta(y_l|x)$ **down**; if $y_l$ already has negligible probability under $\pi_\theta$, there is nothing left to suppress.
- Then the gradient mostly reshapes mass elsewhere — off the data entirely.
- → Sample $y_w,y_l$ from the model being trained (or its close relative), then label.

*Why is offline preference optimization structurally weaker than online RL?*
- The dataset is fixed → the objective constrains the policy **only** on responses in $\mathcal{D}$.
- Off that support it says nothing → the optimum is free to put mass on responses no one ever ranked.
- Online RL re-samples, so whatever the policy drifts toward gets graded next step.

*Why is a KL anchor to $\pi_\text{ref}$ present in almost all of them?*
- The preference signal is a **ranking**, which pins down the policy only up to everything it never ranked.
- $\pi_\text{ref}$ supplies the missing prior: stay where you were unless the data says otherwise.
- Methods that drop it (ORPO, SimPO) replace it w/ another anchor — an SFT term or a length-normalized scale.
```

&nbsp;

## DPO
- **Name**: Direct Preference Optimization {cite:p}`rafailov2023direct`
- **What**: Preference classification w/ the LM as its own reward model.
- **Why**: The RLHF stack is expensive & fragile.
    - Separate RM → an extra model to train, store, and overfit.
    - PPO loop → rollouts, critic, clip & KL tuning → the dominant cost & the dominant source of instability.
- **How**:
    1. The KL-constrained RL objective has a **closed-form** optimum.
    2. Invert it → reward = scaled log-ratio of policy to reference.
    3. Substitute into Bradley-Terry → the intractable partition function cancels.
    4. Maximize the likelihood of the observed preferences → binary CE, directly on $\pi_\theta$.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\pi_\theta$: Policy being trained.
- Hyperparams:
    - $\pi_\text{ref}$: Frozen reference policy.
    - $\beta$: Implicit reward scale (= the KL coeff of the RL problem it replaces).
- Misc:
    - $\hat{r}_\theta$: Implicit reward.

Implicit reward:

$$
\hat{r}_\theta(x,y)=\beta\log\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}
$$

Objective:

$$
\mathcal{L}_\text{DPO}(\theta)=-\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)}-\beta\log\frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}\right)\right]
$$

Gradient:

$$
\nabla_\theta\mathcal{L}_\text{DPO}=-\beta\,\mathbb{E}\left[\underbrace{\sigma\left(\hat{r}_\theta(x,y_l)-\hat{r}_\theta(x,y_w)\right)}_{\text{how wrong the implicit RM is}}\left(\nabla_\theta\log\pi_\theta(y_w|x)-\nabla_\theta\log\pi_\theta(y_l|x)\right)\right]
$$
```

```{tip} Derivation
:class: dropdown
*Where does the objective come from, and why does the reward model vanish?*

1. The KL-constrained RL objective:

    $$
    \max_\pi\ \mathbb{E}_{x\sim\mathcal{D},\ y\sim\pi(\cdot|x)}\left[r(x,y)\right]-\beta\,\text{KL}\left(\pi(\cdot|x)\ \|\ \pi_\text{ref}(\cdot|x)\right)
    $$

2. Its optimum is available in closed form (Gibbs / exponential tilting of the reference):

    $$
    \pi^*(y|x)=\frac{1}{Z(x)}\pi_\text{ref}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right),\qquad Z(x)=\sum_{y}\pi_\text{ref}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right)
    $$

    - $Z(x)$: Partition function — a sum over **all** sequences, intractable.

3. Solve for the reward:

    $$
    r(x,y)=\beta\log\frac{\pi^*(y|x)}{\pi_\text{ref}(y|x)}+\beta\log Z(x)
    $$

4. Bradley-Terry depends only on reward **differences** at the same $x$:

    $$
    p(y_w\succ y_l|x)=\sigma\left(r(x,y_w)-r(x,y_l)\right)
    $$

    → $\beta\log Z(x)$ appears in both terms & cancels. ✅Tractable.

5. Substitute $\pi^*\to\pi_\theta$ & maximize the likelihood of the observed preferences → $\mathcal{L}_\text{DPO}$.

→ The LM **is** the reward model: any $\pi_\theta$ implicitly defines a reward, & fitting the preferences fits the policy directly.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn.functional as F

def seq_logp(logits, labels, ignore=-100):
    ## sum of token log-probs over the RESPONSE tokens -> log pi(y|x)
    logits, labels = logits[:, :-1], labels[:, 1:]
    keep = labels != ignore
    safe = labels.masked_fill(~keep, 0)
    lp = F.log_softmax(logits.float(), -1).gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    return (lp * keep).sum(-1)                       ## (B,)

def dpo_loss(pol_w, pol_l, ref_w, ref_l, beta=0.1):
    ## all four args are seq_logp outputs; ref_* come from the FROZEN model (no grad)
    logits = beta * ((pol_w - ref_w) - (pol_l - ref_l))  ## implicit reward margin
    loss = -F.logsigmoid(logits).mean()
    ## diagnostics that matter: accuracy should rise, both rewards typically FALL
    acc = (logits > 0).float().mean()
    return loss, acc, beta * (pol_w - ref_w).mean(), beta * (pol_l - ref_l).mean()

## Example
pw, pl = torch.tensor([-13.0, -21.0]), torch.tensor([-17.0, -24.0])   ## policy log pi(y|x)
rw, rl = torch.tensor([-12.0, -20.0]), torch.tensor([-14.0, -20.0])   ## frozen reference
print([round(t.item(), 4) for t in dpo_loss(pw, pl, rw, rl)])
## [0.5762, 1.0, -0.1, -0.35] -> perfect accuracy, yet BOTH rewards are negative
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- ⬇️Compute ← ❌RM training, ❌sampling loop; 2 models resident instead of 4.
- Stable ← a plain supervised classification loss w/ a bounded gradient.
- Reuses the SFT trainer; the only new machinery is a frozen forward pass.

*Cons?*
- Off-policy → bounded by the coverage of $\mathcal{D}$.
- Both $\log\pi_\theta(y_w|x)$ **and** $\log\pi_\theta(y_l|x)$ usually fall during training.
- Length bias → longer responses tend to win.
- Overfits preference pairs quickly; 1 epoch is standard.

*Why does the chosen response's log-prob go down?*
- The loss constrains only the **difference** $\hat{r}_\theta(x,y_w)-\hat{r}_\theta(x,y_l)$, never either level.
- $y_w$ & $y_l$ share most of their tokens (prefix, style, formatting) → suppressing $y_l$ drags those shared tokens down too.
- The displaced probability mass goes to sequences that appear in **neither** → completely unconstrained by the objective.
- → Track reward accuracy & the two reward levels separately; falling accuracy w/ a growing margin is the failure signature.

*Why is $\pi_\text{ref}$ necessary?*
- It is the KL anchor inherited from step 1 of the derivation: it makes the reward a *relative* quantity & bounds the drift.
- W/o it, the margin can be maximized by degenerate solutions ($\pi_\theta(y_l|x)\to0$ on everything that resembles $y_l$).
- It also cancels prompt difficulty: a prompt where all responses are improbable does not dominate the loss.

*What must $\pi_\text{ref}$ be?*
- The **derivation** assumes Bradley-Terry preferences, ❌anything about who generated them.
- The **practical** requirement is distribution match: public preference sets were sampled from some $\pi_\text{SFT}$, so DPO's authors set $\pi_\text{ref}=\pi_\text{SFT}$ whenever it exists.
- W/o an SFT model, they instead fit $\pi_\text{ref}$ by MLE on the **chosen** responses — explicitly to mitigate the shift between the true (unavailable) reference distribution & the one DPO uses.
- → An arbitrary checkpoint is not an error in the algebra; it is an uncontrolled distribution shift.

*What does $\beta$ control?*
- The implicit-reward scale = the KL strength. ⬇️$\beta$ → more drift from $\pi_\text{ref}$; ⬆️$\beta$ → stay put.
- $0.1$ is the common default; Tülu 3 instead used **length-normalized** DPO w/ $\beta=5$ — the log-ratio is divided by $|y|$, so the scale is not comparable. {cite:p}`lambert2024tulu`

*Is DPO really RL-free?*
- As an algorithm, ✅: ❌reward model, ❌sampling, ❌policy gradient — just BCE on a fixed dataset.
- As a **problem**, ❌: every line of the derivation is the KL-constrained RL objective. DPO removes the loop, not the framing.
```

&nbsp;

### IPO
- **Name**: Identity Preference Optimization {cite:p}`azar2023general`
- **What**: Squared-loss preference matching w/ a finite target margin, ❌Bradley-Terry.
- **Why**: DPO overfits whenever the preferences are (nearly) deterministic.
    - BT maps a preference **probability** to a reward difference; probability $\to1$ maps to difference $\to\infty$.
    - Real datasets have 1 annotation per pair → the empirical probability **is** 0 or 1.
    - → The optimum drives $\pi_\theta(y_l|x)\to0$ & the KL term stops binding, no matter how large $\beta$ is.
- **How**: Replace $-\log\sigma(\cdot)$ — which always rewards more margin — w/ a squared loss that pins the margin at a finite target.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\tau$: Regularization strength (the role $\beta$ plays in DPO).
- Misc:
    - $h_\pi$: Log-ratio margin.

Margin:

$$
h_\pi(y_w,y_l,x)=\log\frac{\pi_\theta(y_w|x)\,\pi_\text{ref}(y_l|x)}{\pi_\theta(y_l|x)\,\pi_\text{ref}(y_w|x)}
$$

Objective:

$$
\mathcal{L}_\text{IPO}(\theta)=\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\left[\left(h_\pi(y_w,y_l,x)-\frac{\tau^{-1}}{2}\right)^2\right]
$$
```

```{attention} Q&A
:class: dropdown
*Why does a squared loss fix the overfitting?*
- $-\log\sigma(z)$ is strictly decreasing in $z$ → the optimizer is **always** paid to grow the margin.
- $(z-c)^2$ penalizes overshoot → the margin is pulled **to** $c=\frac{\tau^{-1}}{2}$ & held there.
- → $\tau$ actually controls the distance from $\pi_\text{ref}$, which is what it was supposed to do all along.

*What exactly is the $\Psi$PO framing?*
- A general objective: maximize $\mathbb{E}[\Psi(p(y\succ y'))]$ minus a KL term.
- DPO = $\Psi=\log\frac{p}{1-p}$ (unbounded, BT-implied); IPO = $\Psi=\text{identity}$ (bounded, no BT assumption).
- → The pathology is the unboundedness of $\Psi$, ❌anything specific to DPO's algebra.

*When does the difference actually show up?*
- Clean, near-deterministic preferences & many epochs → DPO degrades, IPO holds.
- Noisy preferences (multiple annotators, real disagreement) → BT's assumption is closer to true → the gap narrows.

*Cost vs DPO?*
- Identical: same 2 forward passes, same data. Only the scalar loss changes.
```

&nbsp;

### KTO
- **Name**: Kahneman-Tversky Optimization {cite:p}`ethayarajh2024kto`
- **What**: Preference optimization from **unpaired** binary labels.
- **Why**: Pairing is the expensive part of preference data.
    - Production feedback arrives as thumbs-up / thumbs-down on single responses, ❌as comparisons.
    - Forcing pairs discards unmatched examples & invents comparisons nobody made.
- **How**: Score each response against a reference point w/ a prospect-theory value function.
    1. Implicit reward = the DPO log-ratio.
    2. Reference point $z_0$ = the KL between policy & reference, estimated from **mismatched** pairs in the microbatch.
    3. Value = a saturating function of (reward $-$ reference point), flipped in sign for undesirable responses.
    4. Weight the desirable & undesirable classes to handle imbalance.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\lambda_D$: Weight on desirable examples.
    - $\lambda_U$: Weight on undesirable examples.
    - $\lambda_y$: $\lambda_D$ or $\lambda_U$, whichever applies to $y$.
- Hyperparams:
    - $\beta$: Risk-aversion / reward scale.
- Misc:
    - $z_0$: Reference point.
    - $v$: Value function.

Implicit reward & reference point:

$$
\hat{r}_\theta(x,y)=\log\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)},\qquad z_0=\text{KL}\left(\pi_\theta(y'|x)\ \|\ \pi_\text{ref}(y'|x)\right)
$$

Value:

$$
v(x,y)=\begin{cases}\lambda_D\,\sigma\left(\beta\left(\hat{r}_\theta(x,y)-z_0\right)\right) & y\text{ desirable}\\ \lambda_U\,\sigma\left(\beta\left(z_0-\hat{r}_\theta(x,y)\right)\right) & y\text{ undesirable}\end{cases}
$$

Objective:

$$
\mathcal{L}_\text{KTO}(\theta)=\mathbb{E}_{(x,y)\sim\mathcal{D}}\left[\lambda_y-v(x,y)\right]
$$

$z_0$ is estimated by shifting outputs within the microbatch to form mismatched pairs, clamped at 0, and is **not** backpropagated through.
```

```{attention} Q&A
:class: dropdown
*Why a reference point at all?*
- Prospect theory: humans evaluate outcomes as gains/losses **relative to a baseline**, ❌in absolute terms.
- Mechanically it blocks the cheapest exploit: raising the reward of a desirable response by uniformly inflating the policy also raises $z_0$ → no progress.
- → The model is forced to learn what specifically makes an output desirable.

*Why not backprop through $z_0$?*
- It exists to control where the loss saturates, ❌to be optimized.
- Differentiating it makes training unstable, & $z_0$ is a biased estimate anyway.

*How are $\lambda_D,\lambda_U$ set?*
- Both default to 1, then adjusted for class imbalance: keep $\frac{\lambda_D n_D}{\lambda_U n_U}\in\left[1,\frac{3}{2}\right]$.
- $n_D,n_U$: #desirable & #undesirable examples.

*What $\beta$?*
- $[0.01,0.10]$ for larger models already SFT'd; $[0.10,1.00]$ for smaller models run through KTO w/o prior SFT.

*What is the practical payoff?*
- Uses 1-sided feedback → an order of magnitude more usable data in a deployed product.
- Tolerates class imbalance explicitly, which pairwise losses cannot express.

*What is lost vs DPO?*
- The pairwise signal is the cleanest form of preference; splitting a pair into 2 independent labels discards the fact that they answered the **same** prompt.
```

&nbsp;

### ORPO
- **Name**: Odds Ratio Preference Optimization {cite:p}`hong2024orpo`
- **What**: SFT loss + an odds-ratio penalty, one stage, ❌reference model.
- **Why**: SFT has a side effect that a second stage then has to undo.
    - SFT raises the likelihood of the chosen response — and of stylistically similar **rejected** responses along w/ it, ← it has no repulsive term at all.
    - Two stages also mean two datasets, two hyperparameter sets, and a frozen copy of the model in memory.
- **How**:
    1. Length-normalize the sequence log-likelihood → $P_\theta(y|x)$.
    2. Odds $=\frac{P}{1-P}$; the log ratio of chosen-to-rejected odds is the preference score.
    3. Loss = NLL on the chosen response $+\ \lambda\times$ the odds-ratio term. Train from the **base** model in one pass.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\lambda$: Weight on the odds-ratio term.
- Misc:
    - $m$: #tokens in $y$.

Length-normalized likelihood:

$$
\log P_\theta(y|x)=\frac{1}{m}\sum_{t=1}^{m}\log P_\theta(y_t|x,y_{<t})
$$

Odds & odds ratio:

$$
\textbf{odds}_\theta(y|x)=\frac{P_\theta(y|x)}{1-P_\theta(y|x)},\qquad \textbf{OR}_\theta(y_w,y_l)=\frac{\textbf{odds}_\theta(y_w|x)}{\textbf{odds}_\theta(y_l|x)}
$$

Objective:

$$
\mathcal{L}_\text{ORPO}=\mathbb{E}_{(x,y_w,y_l)}\left[\mathcal{L}_\text{SFT}+\lambda\cdot\mathcal{L}_\text{OR}\right],\qquad \mathcal{L}_\text{OR}=-\log\sigma\left(\log\textbf{OR}_\theta(y_w,y_l)\right)
$$
- $\mathcal{L}_\text{SFT}$: Ordinary NLL on $y_w$.
```

```{attention} Q&A
:class: dropdown
*Why odds instead of the plain probability ratio?*
- ⚠️ Not because the odds ratio is smaller — algebraically $\textbf{OR}=\textbf{PR}\cdot\frac{1-P_\theta(y_l|x)}{1-P_\theta(y_w|x)}\geq\textbf{PR}$ whenever $y_w$ leads.
- The argument is about the **distribution of values in practice**: sampled $\log\textbf{PR}$ has far heavier tails than $\log\textbf{OR}$ → the probability ratio discriminates the disfavored response far more extremely.
- → The odds ratio is the *stabler* penalty, which is what a term running **alongside** SFT needs; an over-extreme penalty would fight the NLL term.

*Why keep the SFT term?*
- The odds-ratio term is purely **relative** — it is satisfied by lowering $y_l$ as easily as by raising $y_w$.
- The NLL term pins the chosen response's absolute likelihood → the model still learns to *produce* it.
- → This is also what replaces the missing $\pi_\text{ref}$ anchor.

*What is actually saved?*
- 1 model in memory instead of 2, 1 training run instead of 2, 1 dataset.
- ← The reference model exists only to define a relative reward; ORPO's absolute NLL term serves the same purpose.

*Cons?*
- $\lambda$ is the whole balance between imitation & discrimination, & it is sensitive.
- Length normalization is hard-wired → no way to opt out.
- Less validated at frontier scale than DPO.
```

&nbsp;

### SimPO
- **Name**: Simple Preference Optimization {cite:p}`meng2024simpo`
- **What**: Length-normalized average log-prob as the reward, w/ a target margin, ❌reference model.
- **Why**: DPO's implicit reward is not the quantity decoding actually ranks by.
    - Beam search & sampling rank sequences roughly by **average** log-likelihood per token.
    - DPO's reward is a **sum** of log-ratios against $\pi_\text{ref}$ → a response can win on the training reward yet lose under the model's own decoding metric.
    - The reference model also costs a second resident copy & a forward pass every step.
- **How**:
    1. Reward = average log-prob per token, scaled by $\beta$. ❌$\pi_\text{ref}$.
    2. Bradley-Terry w/ a **target margin** $\gamma$: the chosen must beat the rejected by at least $\gamma$.
    3. Minimize BCE.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\beta$: Reward scale.
    - $\gamma>0$: Target reward margin.

Reward:

$$
\hat{r}_\text{SimPO}(x,y)=\frac{\beta}{|y|}\log\pi_\theta(y|x)=\frac{\beta}{|y|}\sum_{t=1}^{|y|}\log\pi_\theta(y_t|x,y_{<t})
$$

Objective:

$$
\mathcal{L}_\text{SimPO}(\theta)=-\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\left[\log\sigma\left(\frac{\beta}{|y_w|}\log\pi_\theta(y_w|x)-\frac{\beta}{|y_l|}\log\pi_\theta(y_l|x)-\gamma\right)\right]
$$
```

```{attention} Q&A
:class: dropdown
*Why is length normalization load-bearing?*
- W/o it, the learned reward correlates strongly & positively w/ response length → **length exploitation** (longer, lower-quality output).
- W/ it, the reward difference rises for all pairs regardless of their lengths.
- ⚠️ DPO gets a partial version of this for free: the $\pi_\text{ref}$ ratio cancels much of the length dependence.

*What does $\gamma$ do?*
- Demands a strictly positive gap instead of merely "greater than".
- ⬆️$\gamma$ → reward accuracy rises monotonically, but AlpacaEval 2 win rate rises **then falls**.
- ← Large $\gamma$ flattens the reward distribution & lowers the log-likelihood of winning sequences → degeneration.
- → Tune $\gamma$ against generation quality, ❌against reward accuracy.

*What is the cost of dropping $\pi_\text{ref}$?*
- ❌KL anchor → nothing bounds the drift from the SFT model.
- The average-log-prob reward is bounded above by 0, which limits the most degenerate solutions, but it is not a trust region.
- → An SFT regularization term is a common add-back.

*Why does aligning the reward w/ the decoding metric matter?*
- Training optimizes the quantity you write down; serving selects by a different one.
- Any gap between them is a systematic train-inference mismatch — the same failure class as a mismatched chat template.
```

&nbsp;

````{dropdown} Table: Preference Objectives at a Glance
| Method | Implicit reward | Loss shape | $\pi_\text{ref}$ | Data | Length norm | Delta |
|:--|:--|:--|:--|:--|:--|:--|
| DPO | $\beta\log\frac{\pi_\theta}{\pi_\text{ref}}$ | $-\log\sigma(\Delta)$ | ✅ | Pairs | ❌ (partial, via the ratio) | ❌RM, ❌rollouts |
| IPO | Same | $(\Delta-\frac{\tau^{-1}}{2})^2$ | ✅ | Pairs | ❌ | Bounded margin → ❌overfitting |
| KTO | $\log\frac{\pi_\theta}{\pi_\text{ref}}$ | $\lambda_y-\sigma(\beta(\hat{r}-z_0))$ | ✅ | **Unpaired** labels | ❌ | ❌Pairing requirement |
| ORPO | $\log\textbf{odds}_\theta$ | $\mathcal{L}_\text{SFT}+\lambda\mathcal{L}_\text{OR}$ | ❌ | Pairs | ✅ | 1 stage, 1 model |
| SimPO | $\frac{\beta}{\|y\|}\log\pi_\theta$ | $-\log\sigma(\Delta-\gamma)$ | ❌ | Pairs | ✅ | Reward = decoding metric |

$\Delta$: Implicit reward margin between chosen & rejected.
````