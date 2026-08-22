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
# Distillation
- **What**: Training a student to match a teacher's output distribution. {cite:p}`hinton2015distilling`
- **Why**: The teacher's behavior is wanted at the student's cost.
    - Serving a frontier model is expensive, & capability per param is not fixed.
    - A one-hot label carries at most $\log_2|\mathcal{V}|\approx17$ bits per position; the teacher's full distribution carries the relative probabilities of every **wrong** token too.
    - Those relatives encode the teacher's similarity structure ("dark knowledge") — the part a hard label throws away.
- **How**:
    1. Choose the token set to distill on: fixed corpus / teacher samples / student samples.
    2. Teacher forward pass → a next-token distribution at every position.
    3. Minimize a divergence between student & teacher distributions, optionally mixed w/ hard-label CE.
    4. Soften both sides w/ temperature $\tau$ during training.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $p_T(\cdot|x,y_{<t})$: Teacher next-token distribution.
    - $p_S^\theta(\cdot|x,y_{<t})$: Student next-token distribution.
- Params:
    - $z$: Student logits.
- Hyperparams:
    - $\tau$: Temperature.
    - $\alpha$: Weight on the soft (teacher) term.

Objective:

$$
\mathcal{L}_\text{KD}(\theta)=\mathbb{E}_{(x,y)\sim\mathcal{D}}\left[\frac{1}{|y|}\sum_{t=1}^{|y|}\text{KL}\left(p_T(\cdot|x,y_{<t})\ \|\ p_S^\theta(\cdot|x,y_{<t})\right)\right]
$$

W/ temperature & hard labels:

$$
\mathcal{L}=\alpha\tau^2\,\text{KL}\left(p_T^\tau\ \|\ p_S^{\theta,\tau}\right)+(1-\alpha)\,\text{CE}(y,p_S^\theta)
$$
- $p^\tau$: Softmax over logits divided by $\tau$.
- $\tau^2$: Rescaling — softened gradients shrink as $\tau^{-2}$, so this keeps the two terms comparable.

Gradient w.r.t. student logits (at $\tau=1$):

$$
\nabla_z\,\text{KL}(p_T\|p_S^\theta)=p_S^\theta-p_T,\qquad \nabla_z\,\text{CE}(y,p_S^\theta)=p_S^\theta-\mathbf{e}_{y_t}
$$
- $\mathbf{e}_{y_t}$: One-hot at the target token.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn.functional as F

def kd_loss(student_logits, teacher_logits, labels, tau=2.0, alpha=0.9, ignore=-100):
    ## logits and labels are assumed ALREADY next-token aligned (shift applied upstream)
    ## soften BOTH sides with the same temperature
    s_log = F.log_softmax(student_logits / tau, dim=-1)
    t_prob = F.softmax(teacher_logits / tau, dim=-1)
    ## KL(teacher || student), summed over vocab, averaged over live positions
    kl = (t_prob * (t_prob.clamp_min(1e-9).log() - s_log)).sum(-1)   ## (B, T)
    keep = labels != ignore
    soft = (kl * keep).sum() / keep.sum()
    hard = F.cross_entropy(
        student_logits.reshape(-1, student_logits.size(-1)),
        labels.reshape(-1), ignore_index=ignore,
    )
    ## tau^2 restores the gradient scale the softmax division removed
    return alpha * tau**2 * soft + (1 - alpha) * hard

## Example
B, T, V = 2, 3, 10
labels = torch.randint(0, V, (B, T))
print(kd_loss(torch.randn(B, T, V), torch.randn(B, T, V), labels).item())
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- ⬇️Serving cost at a given capability level.
- ⬆️Sample efficiency ← every position supplies $|\mathcal{V}|$ numbers instead of 1 index.
- Transfers behavior the student would not reach from the same data alone.

*Cons?*
- Teacher inference over the whole corpus → expensive, & it must be redone for new data.
- Ceiling is essentially the teacher.
- Token-level KD needs **logits** & a shared tokenizer → white-box only.
- Distilling a commercial API usually violates its terms.

*Why is matching the distribution better than matching the argmax?*
- CE gradient is $p_S-\mathbf{e}_{y_t}$: informative at 1 vocab entry.
- KD gradient is $p_S-p_T$: a target at **every** vocab entry, including which wrong answers are near-misses.
- → Effectively a much denser label per token.

*Forward vs reverse KL?*
- Forward $\text{KL}(p_T\|p_S)$ → **mode-covering**: the student must cover everything the teacher does, hedging where it lacks capacity.
- Reverse $\text{KL}(p_S\|p_T)$ → **mode-seeking**: the student concentrates on a subset of teacher modes, sharper & more fluent, less diverse.
- → Capacity gap is the deciding factor: the wider it is, the more reverse KL is preferred.

*Can the student ever beat the teacher?*
- ✅ In restricted senses: distilling an expensive procedure (CoT, search, ensembling) into one forward pass, or distilling a *filtered* teacher.
- ✅ **Weak-to-strong**: a strong student supervised by a much weaker teacher recovers a large fraction of the gap — supervision **elicits** latent capability rather than transferring it. {cite:p}`burns2023weak`
- ❌ In general: matching a distribution cannot exceed it.

*Different tokenizers?*
- Token-level KD is undefined ← the vocab index spaces don't correspond.
- → Fall back to sequence-level KD, or align tokenizations approximately.
```

&nbsp;

### Sequence-Level KD
- **What**: SFT on teacher-generated sequences. {cite:p}`kim2016sequence`
- **Why**: Token-level KD needs logits & a fixed target corpus, and matches the wrong object.
    - Frontier teachers expose text, ❌logits.
    - What matters at inference is the distribution over **sequences**, not per-position marginals conditioned on a ground-truth prefix.
- **How**:
    1. Generate responses from the teacher for each prompt (sample, or take the beam-search mode).
    2. Optionally filter w/ a verifier.
    3. Plain SFT on (prompt, teacher response).

```{attention} Q&A
:class: dropdown
*Why does it work w/o any logits?*
- Sampling $y\sim p_T$ then doing MLE minimizes $\text{KL}(p_T\|p_S)$ at the **sequence** level, Monte-Carlo estimated.
- The original formulation instead approximates the teacher's sequence distribution by a point mass at its **mode** (beam output) → SFT on that single output.
- → Either way the teacher's distribution enters through its samples, ❌its probabilities.

*Trade-off vs token-level KD?*
- ✅Black-box, tokenizer-agnostic, reuses the SFT trainer, teacher runs once per prompt.
- ❌ ~$\log_2|\mathcal{V}|$ bits per position instead of a full distribution → far weaker signal per token.

*What is the failure mode?*
- It is SFT → exposure bias returns in full: the student only ever sees teacher trajectories, which may be very improbable under itself.
- Sharpening on the teacher's mode also collapses diversity relative to the teacher.

*Is this just "SFT on synthetic data"?*
- Mechanically identical. Most open "distilled" models are exactly this.
- The distinction that matters is the **filter**: unfiltered teacher output inherits every teacher error.
```

&nbsp;

### GKD
- **Name**: Generalized Knowledge Distillation {cite:p}`agarwal2023onpolicy`
- **What**: Distillation on the **student's own** samples.
- **Why**: Fixed teacher data leaves the student unsupervised exactly where it operates.
    - Trained on teacher trajectories, evaluated on its own → the states it actually visits were never labeled.
    - A low-capacity student cannot match the teacher everywhere, so forward KL on teacher data spends its capacity hedging modes it can never represent.
- **How**:
    1. Sample outputs from the **student**.
    2. Teacher scores those exact tokens → target distributions.
    3. Minimize a divergence at every position.
    4. Mix student-generated & fixed data w/ $\lambda$; pick the divergence via JSD($\beta$).

```{note} Math
:class: dropdown
Notations:
- IO:
    - $(X,Y)$: Fixed dataset of prompts & target responses.
    - $p_S(\cdot|x)$: Student sampling distribution (samples treated as data, no gradient through sampling).
- Hyperparams:
    - $\lambda\in[0,1]$: Fraction of student-generated data.
    - $\beta\in[0,1]$: Divergence interpolation.
- Misc:
    - $\mathcal{D}$: Chosen divergence.

Divergence:

$$
\mathcal{D}_{\text{JSD}(\beta)}(P\|Q)=\beta\,\text{KL}\left(P\ \|\ \beta P+(1-\beta)Q\right)+(1-\beta)\,\text{KL}\left(Q\ \|\ \beta P+(1-\beta)Q\right)
$$
- $\beta\to0$ → recovers forward $\text{KL}(P\|Q)$ (up to scale); $\beta\to1$ → reverse $\text{KL}(Q\|P)$; $\beta=0.5$ → standard JSD.

Objective:

$$
\mathcal{L}_\text{GKD}(\theta)=(1-\lambda)\,\mathbb{E}_{(x,y)\sim(X,Y)}\left[\mathcal{D}(p_T\|p_S^\theta)(y|x)\right]+\lambda\,\mathbb{E}_{x\sim X}\,\mathbb{E}_{y\sim p_S(\cdot|x)}\left[\mathcal{D}(p_T\|p_S^\theta)(y|x)\right]
$$
- $\mathcal{D}(p_T\|p_S^\theta)(y|x)$: Token-averaged divergence over the positions of $y$.
- $\lambda=0$ → supervised KD; $\lambda=1$ → fully on-policy.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn.functional as F

def jsd_beta(t_logits, s_logits, beta=0.5):
    ## generalized JSD: beta -> 0 gives forward KL, beta -> 1 gives reverse KL
    logp_t, logp_s = F.log_softmax(t_logits, -1), F.log_softmax(s_logits, -1)
    ## mixture in log-space for numerical stability
    log_mix = torch.logsumexp(
        torch.stack([logp_t + torch.log(torch.tensor(beta).clamp_min(1e-9)),
                     logp_s + torch.log(torch.tensor(1 - beta).clamp_min(1e-9))]), dim=0)
    kl_t = (logp_t.exp() * (logp_t - log_mix)).sum(-1)
    kl_s = (logp_s.exp() * (logp_s - log_mix)).sum(-1)
    return beta * kl_t + (1 - beta) * kl_s            ## (B, T)

def gkd_loss(teacher, student, fixed_batch, student_samples, beta=0.5, lam=0.5):
    ## student_samples: token ids the STUDENT just generated -> detached, they are DATA
    on = jsd_beta(teacher(student_samples), student(student_samples), beta).mean()
    off = jsd_beta(teacher(fixed_batch), student(fixed_batch), beta).mean()
    return (1 - lam) * off + lam * on

## Example
B, T, V = 2, 4, 8
print(jsd_beta(torch.randn(B, T, V), torch.randn(B, T, V)).shape)  ## torch.Size([2, 4])
```
````

```{attention} Q&A
:class: dropdown
*Why is this not RL, given that it samples from the policy?*
- ❌Reward, ❌advantage, ❌policy-gradient estimator.
- The samples are **detached** & used as input positions; the gradient flows only through the divergence at those positions.
- → On-policy *data*, supervised *objective*.

*Which $\beta$?*
- Forward KL ($\beta\to0$) → the student hedges; safe when capacity is close to the teacher's.
- Reverse KL ($\beta\to1$) → the student commits to modes it can actually represent; better under a large capacity gap.
- Reported best settings are task-dependent → treat $\beta$ & $\lambda$ as a 2D sweep, ❌constants.

*Cost vs sequence-level KD?*
- Student generation **every step** + a teacher forward pass on every generated token.
- → Order-of-magnitude more expensive; buy it only when exposure bias is the observed failure.

*Relation to RFT?*
- Both train on self-generated text.
- RFT: verifier filters, target is the sample itself (one-hot), rejected samples discarded.
- GKD: no filter, target is the teacher's full distribution, every sample used.
- → RFT needs a verifier; GKD needs a teacher.
```