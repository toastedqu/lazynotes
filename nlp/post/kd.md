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
Compress a teacher's behavior into a smaller student. Organized by 3 choices: what to match, which divergence, where to measure it.

Notations:
- $x$: Prompt (input token seq)
- $y$: Response (output token seq)
- $y_t$: $t$-th response token
- $y_{<t}$: Response prefix before position $t$
- $|y|$: Response length (#tokens)
- $\mathcal{V}$: Vocabulary
- $\mathcal{D}$: Dataset
- $\theta$: Student params
- $p_T$: Teacher distribution: $p_T(\cdot|x,y_{<t})$ over $\mathcal{V}$, or $p_T(y|x)$ over full sequences
- $p_S$: Student distribution (params $\theta$), same 2 forms
- $D(\cdot\Vert\cdot)$: A divergence

&nbsp;

## Token-Level KD
- **What**: Match the teacher's next-token distribution at every position of a fixed corpus. {cite:p}`hinton2015distilling`
- **Why**: Teacher capability at student serving cost, w/ a denser label than one-hot.
    - Teacher distribution → also ranks every wrong token → teacher's similarity structure ("dark knowledge").
- **How**:
    1. Teacher forward pass over the corpus → distribution over $\mathcal{V}$ at every position.
    2. Soften both sides w/ temperature $\tau$.
    3. Minimize a [divergence](#divergence) to the student, optionally mixed w/ hard-label CE.
    4. Teacher-forced throughout → ❌sampling, ❌rollouts.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\tau$: Temperature.
    - $\alpha$: Weight on the soft (teacher) term.
- Intermediate values:
    - $z$: Student logits at one position.
    - $p^\tau$: Softmax of logits divided by $\tau$.

Objective:

$$
\mathcal{L}_\text{KD}(\theta)=\mathbb{E}_{(x,y)\sim\mathcal{D}}\left[\frac{1}{|y|}\sum_{t=1}^{|y|}D_\text{KL}\left(p_T(\cdot|x,y_{<t})\ \Vert\ p_S(\cdot|x,y_{<t})\right)\right]
$$

- Per position, w/ temperature & hard label:

    $$
    \mathcal{L}=\alpha\tau^2\,D_\text{KL}\left(p_T^\tau\ \Vert\ p_S^\tau\right)+(1-\alpha)\,\text{CE}(y_t,p_S)
    $$

    - $\tau^2$: Soft-term gradient scales as $\tau^{-2}$ → $\tau^2$ keeps both terms comparable.
- Gradient w.r.t. $z$ at $\tau=1$:

    $$
    \nabla_z\,D_\text{KL}(p_T\Vert p_S)=p_S-p_T,\qquad \nabla_z\,\text{CE}(y_t,p_S)=p_S-\mathbf{e}_{y_t}
    $$

    - $\mathbf{e}_{y_t}$: One-hot at the target token.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenLevelKD(nn.Module):
    def __init__(self, tau=2.0, alpha=0.9, ignore=-100):
        super().__init__()
        self.tau, self.alpha, self.ignore = tau, alpha, ignore

    def forward(self, s_logits, t_logits, labels):
        ## logits (B, T, V) & labels (B, T) are ALREADY next-token aligned
        ## t_logits come from a frozen teacher under torch.no_grad()
        ## soften BOTH sides w/ the same temperature
        s_log = F.log_softmax(s_logits / self.tau, dim=-1)
        t_log = F.log_softmax(t_logits / self.tau, dim=-1)
        ## KL(teacher || student) per position, summed over the vocab
        kl = (t_log.exp() * (t_log - s_log)).sum(-1)                 ## (B, T)
        keep = labels != self.ignore
        soft = (kl * keep).sum() / keep.sum()
        hard = F.cross_entropy(s_logits.flatten(0, 1), labels.flatten(), ignore_index=self.ignore)
        ## tau^2 restores the gradient scale that the division by tau removed
        return self.alpha * self.tau**2 * soft + (1 - self.alpha) * hard

## Example
B, T, V = 2, 3, 10
loss = TokenLevelKD()(torch.randn(B, T, V), torch.randn(B, T, V), torch.randint(0, V, (B, T)))
print(loss.shape)  ## torch.Size([])
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- ⬇️Serving cost at a given capability level.
- ⬆️Sample efficiency ← every position supplies $|\mathcal{V}|$ numbers instead of 1 index.
- Transfers behavior the student does not reach from the same data alone.

*Cons?*
- Teacher inference over the whole corpus → expensive; redo it for new data.
- Ceiling ≈ the teacher.
- Needs logits & a shared tokenizer → white-box only.
- Distilling a commercial API usually violates its terms.

*Why match the distribution, not the argmax?*
- CE gradient $p_S-\mathbf{e}_{y_t}$ → target at 1 vocab entry.
- KD gradient $p_S-p_T$ → target at every vocab entry, incl. which wrong tokens are near-misses.
- → Much denser label per token.

*Why $\tau>1$?*
- At $\tau=1$, a confident teacher gives wrong tokens ≈0 probability → they barely enter the loss.
- $\tau$⬆️ → flatter $p_T^\tau$ → wrong-token ratios become visible → dark knowledge transfers.
- High-$\tau$ limit w/ zero-mean logits → KD ≈ matching logits by MSE.

*Why match logits, not hidden states?*
- Logits live in a shared space ($\mathcal{V}$) → directly comparable between 2 models w/ the same tokenizer.
- Hidden states → need a learned projection & a layer mapping → [Feature-Level KD](#feature-level-kd).

*Can the student beat the teacher?*
- ✅ Distill an expensive procedure (CoT, search, ensemble) into 1 forward pass, or distill a filtered teacher.
- ✅ Weak-to-strong: a strong student under a weaker teacher recovers a large fraction of the gap → supervision elicits latent capability. {cite:p}`burns2023weak`
- ❌ At the exact optimum: $p_S=p_T$ reproduces the teacher, no more.
- → Gains above the teacher come from imperfect matching, filtering, or the student's own priors.

*Different tokenizers?*
- Token-level KD is undefined ← the 2 vocab index spaces do not correspond.
- → Use text ([Sequence-Level KD](#sequence-level-kd)), or drop token identity ([ULD](#uld)).
```

&nbsp;

## Feature-Level KD
- **What**: Match the teacher's intermediate representations, not only its outputs. {cite:p}`romero2015fitnets`
- **Why**: Logits show what the teacher outputs, ❌how it computes it.
    - Intermediate layers → extra supervision per layer → deep, thin students train better.
- **How**:
    1. Map each distilled student layer to a teacher layer (e.g., every $k$-th teacher layer).
    2. Project student features to teacher width w/ a learned regressor (linear for transformers).
    3. Minimize a distance (MSE, cosine) between projected student & teacher features.
    4. Usually add the token-level KD loss on top.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $W_l\in\mathbb{R}^{d_T\times d_S}$: Projection for student layer $l$.
- Misc:
    - $\mathbf{h}_S^{(l)}\in\mathbb{R}^{d_S}$: Student hidden state at layer $l$, one position.
    - $\mathbf{h}_T^{(g(l))}\in\mathbb{R}^{d_T}$: Teacher hidden state at the mapped layer.
    - $g$: Layer mapping, student → teacher.
    - $\mathcal{S}$: Set of distilled student layers.

Objective:

$$
\mathcal{L}_\text{feat}=\sum_{l\in\mathcal{S}}\frac{1}{d_T}\left\Vert W_l\mathbf{h}_S^{(l)}-\mathbf{h}_T^{(g(l))}\right\Vert_2^2
$$
- Averaged over positions & data.
- Attention-map variant: match per-head attention matrices instead of hidden states.
    - TinyBERT → MSE on attention scores; MiniLM → KL on attention distributions.
    - Attention matrix shape = seq length × seq length → independent of width → no projection.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureKD(nn.Module):
    def __init__(self, d_s, d_t, layer_map):
        super().__init__()
        self.layer_map = layer_map                     ## {student layer: teacher layer}
        ## 1 learned projection per distilled layer: student width -> teacher width
        self.proj = nn.ModuleDict({str(l): nn.Linear(d_s, d_t, bias=False) for l in layer_map})

    def forward(self, s_hiddens, t_hiddens):
        ## s_hiddens[l]: (B, T, d_s); t_hiddens[g]: (B, T, d_t) from a frozen teacher
        loss = 0.0
        for l, g in self.layer_map.items():
            ## mse_loss averages over B, T & d_t -> the 1/d_T factor in the Math block
            loss = loss + F.mse_loss(self.proj[str(l)](s_hiddens[l]), t_hiddens[g])
        return loss

## Example
B, T = 2, 3
s_h = [torch.randn(B, T, 4) for _ in range(2)]        ## 2-layer student, width 4
t_h = [torch.randn(B, T, 8) for _ in range(4)]        ## 4-layer teacher, width 8
print(FeatureKD(4, 8, {0: 1, 1: 3})(s_h, t_h).shape)  ## torch.Size([])
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- Supervision at every distilled layer → ⬆️signal for deep, thin students.

*Cons?*
- Needs teacher hidden states → white-box only.
- Layer mapping & projection are design choices w/o a canonical answer.
- Independently trained models → hidden coordinates share no meaning → rare across LLM families; common in BERT compression.

*Why does DistilBERT need no projection?*
- Student width = teacher width, & student layers are initialized from every 2nd teacher layer.
- → Coordinates already aligned → cosine loss on hidden states directly.

*Why match attention maps?*
- Shape depends on seq length & heads, ❌width → no width projection; head counts still need a mapping.
- Encodes which tokens interact → transfers structure, not only outputs.
```

&nbsp;

## Sequence-Level KD
- **What**: SFT on teacher-generated sequences. {cite:p}`kim2016sequence`
- **Why**: Token-level KD needs logits & matches the wrong object.
    - Frontier teachers expose text, ❌logits.
    - Inference samples whole sequences → target = sequence distribution, ❌per-position marginals on ground-truth prefixes.
- **How**:
    1. Teacher generates a response per prompt (sample, or beam-search mode).
    2. Optionally filter w/ a verifier.
    3. SFT on (prompt, teacher response).

```{note} Math
:class: dropdown
Objective:

$$
\mathcal{L}_\text{SeqKD}(\theta)=D_\text{KL}\left(p_T(\cdot|x)\ \Vert\ p_S(\cdot|x)\right)=-\mathbb{E}_{y\sim p_T(\cdot|x)}\left[\log p_S(y|x)\right]-H\left(p_T(\cdot|x)\right)
$$
- $H(p_T(\cdot|x))$: Teacher sequence entropy, $\perp\theta$ → drops from the gradient.
- → Sequence-level forward KL $\equiv$ MLE on teacher samples.
- Sum over all $y\in\mathcal{V}^*$ → intractable → 2 surrogates:
    - Monte-Carlo (unbiased up to the constant $-H(p_T)$):

        $$
        \hat{\mathcal{L}}_\text{MC}=-\frac{1}{N}\sum_{n=1}^{N}\log p_S(y^{(n)}|x),\qquad y^{(n)}\sim p_T(\cdot|x)
        $$

    - Mode (original formulation):

        $$
        \hat{\mathcal{L}}_\text{mode}=-\log p_S(\hat{y}|x),\qquad \hat{y}\approx\arg\max_y p_T(y|x)
        $$

        - $\hat{y}$: Beam-search output → point-mass approximation of $p_T(\cdot|x)$, ❌an estimator.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SequenceLevelKD(nn.Module):
    def __init__(self, teacher, max_new=4):
        super().__init__()
        self.teacher, self.max_new = teacher, max_new

    @torch.no_grad()
    def generate(self, x):
        ## teacher writes the target: y ~ p_T(.|x), token by token -> only TEXT survives
        y = x
        for _ in range(self.max_new):
            probs = F.softmax(self.teacher(y)[:, -1], dim=-1)
            y = torch.cat([y, torch.multinomial(probs, 1)], dim=1)
        return y

    def forward(self, student, x):
        seq = self.generate(x)
        logits = student(seq[:, :-1])                  ## position i predicts token i+1
        ## plain SFT on RESPONSE tokens -> MC estimate of KL(p_T || p_S) up to a constant
        resp = logits[:, x.size(1) - 1:]
        return F.cross_entropy(resp.flatten(0, 1), seq[:, x.size(1):].flatten())

## Example
V, d = 10, 8
lm = lambda: nn.Sequential(nn.Embedding(V, d), nn.Linear(d, V))   ## toy LM: (B, L) -> (B, L, V)
x = torch.randint(0, V, (2, 3))
print(SequenceLevelKD(lm())(lm(), x).shape)  ## torch.Size([])
```
````

```{attention} Q&A
:class: dropdown
*Why does it work w/o logits?*
- Teacher distribution enters through its samples, ❌its probabilities.
- MC form → sequence-level forward KL; mode form → point-mass approximation of it.

*Trade-off vs token-level KD?*
- ✅Black-box, tokenizer-agnostic, reuses the SFT trainer, teacher runs once per prompt.
- ❌1 sampled token per position instead of a full distribution → far weaker signal per token.

*Why only forward KL here, when token-level KD can use any divergence?*
- Every divergence needs $\mathbb{E}_{y\sim P}[\cdot]$ for some $P$; only $P=p_T$ is samplable once, offline.
- Reverse KL needs $y\sim p_S$ → changes every step → re-sampling & policy gradient ([MiniLLM](#minillm)).
- Symmetric ones ([JSD](#jsd), [TVD](#tvd)) need both → draw the teacher half offline once.

*Failure mode?*
- It is SFT → full exposure bias: the student sees only teacher trajectories, which can be improbable under itself.
- Mode form sharpens onto the teacher's mode → diversity⬇️ vs the teacher.

*Is this just "SFT on synthetic data"?*
- Mechanically identical. Most open "distilled" models are exactly this.
- The filter matters: unfiltered teacher output inherits every teacher error.
```

&nbsp;

### CoT Distillation
- **Name**: Chain-of-Thought Distillation {cite:p}`li2022explanations`
- **What**: SFT on teacher rationale + answer, ❌answer only.
    - Concurrent: Magister et al. 2022, Ho et al. 2022 (Fine-tune-CoT) → no single origin.
- **Why**: The answer omits the computation that produced it.
    - Label says what, ❌how → the student must rediscover the procedure from the label alone.
- **How**:
    1. Prompt the teacher (few-shot CoT) → rationale + answer.
    2. Optionally keep only samples w/ a correct answer.
    3. SFT the student on the result.

```{attention} Q&A
:class: dropdown
*Why does a rationale help a small model?*
- Rationale splits 1 hard mapping into short steps → each step easier to fit.
- More supervised tokens per example → ⬆️sample efficiency.

*Risk?*
- Unverified rationale → a correct answer reached by wrong steps trains the wrong procedure.
- Teacher rationales are post-hoc text, ❌a faithful trace of the teacher's computation.

*Inference cost?*
- Student trained on rationale → answer generates the rationale at inference → ⬆️latency.
- → [Distilling Step-by-Step](#distilling-step-by-step) removes it.
```

&nbsp;

#### Distilling Step-by-Step
- **What**: CoT distillation as multi-task learning: label & rationale as 2 separate output tasks. {cite:p}`hsieh2023distilling`
- **Why**: Concatenated rationale → student must generate it at inference.
- **How**:
    1. Teacher (few-shot CoT) → rationale + label.
    2. Student input w/ prefix `[label]` → predict label; prefix `[rationale]` → predict rationale.
    3. Shared weights → rationale is training-only supervision.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\lambda$: Rationale-loss weight.
- Misc:
    - $\hat{r}$: Teacher rationale.

Objective:

$$
\mathcal{L}=\mathcal{L}_\text{label}+\lambda\,\mathcal{L}_\text{rationale},\qquad \mathcal{L}_\text{label}=\text{CE}\left(f_\theta([\text{label}];x),\ \hat{y}\right),\quad \mathcal{L}_\text{rationale}=\text{CE}\left(f_\theta([\text{rationale}];x),\ \hat{r}\right)
$$
- $f_\theta$: Student seq2seq model.
- $\hat{y}$: Label (human or teacher).
```

```{attention} Q&A
:class: dropdown
*How much does it buy?*
- 770M T5 > few-shot 540B PaLM on 1 benchmark, w/ 80% of the data.
- Standard fine-tuning of the same T5 does not match it even w/ 100%.
- → Gain is in sample efficiency, not only param count.

*Why multi-task, not concatenation?*
- Concatenation → rationale at inference → ⬆️latency & an extra failure point.
- Separate prefixes → rationale shapes shared representations, then is dropped at inference.
```

&nbsp;

## Divergence
- **What**: Choice of $D$ in $\min_\theta D(p_T\Vert p_S)$.
- **Why**: Capacity gap → $p_S=p_T$ unreachable → divergences disagree on where to put the residual error.
- **How**: Each weights the teacher-student mismatch by a different measure.

```{dropdown} Table: Divergences
| Divergence | Formula | Bounded | Symmetric | Under a capacity gap |
|:--|:--|:--|:--|:--|
| FKLD | $D_\text{KL}(p_T\Vert p_S)$ | ❌ | ❌ | Zero-avoiding → covers all teacher modes |
| RKLD | $D_\text{KL}(p_S\Vert p_T)$ | ❌ | ❌ | Zero-forcing → drops teacher modes |
| JSD($\beta$) | $\beta D_\text{KL}(p_T\Vert M_\beta)+(1-\beta)D_\text{KL}(p_S\Vert M_\beta)$ | ✅ $\leq H_b(\beta)$ | Only at $\beta=0.5$ | Interpolates FKLD ↔ RKLD |
| TVD | $\frac{1}{2}\Vert p_T-p_S\Vert_1$ | ✅ $\leq1$ | ✅ | Balanced |
| SKL / SRKL | $D_\text{KL}(p\Vert\alpha p+(1-\alpha)q)$ | ✅ $\leq\log\frac{1}{\alpha}$ | ❌ | FKLD / RKLD w/ bounded gradient |
- $M_\beta=\beta p_T+(1-\beta)p_S$: Mixture.
- $H_b(\beta)=-\beta\log\beta-(1-\beta)\log(1-\beta)$: Binary entropy.
```

&nbsp;

### FKLD
- **Name**: Forward Kullback-Leibler Divergence
- **What**: $D_\text{KL}(p_T\Vert p_S)$: mismatch averaged under the teacher.
- **Why**: Default. ≡ CE vs a soft label → drops into any SFT trainer.
- **How**: **Zero-avoiding**. $p_T(v)>0$ & $p_S(v)\to0$ → $\log$ ratio $\to\infty$ → student keeps mass wherever the teacher has any.

```{note} Math
:class: dropdown
Definition:

$$
D_\text{KL}(p_T\Vert p_S)=\sum_{v\in\mathcal{V}}p_T(v)\log\frac{p_T(v)}{p_S(v)}=H(p_T,p_S)-H(p_T)
$$
- $H(p_T,p_S)$: CE.
- $H(p_T)$: $\perp\theta$ → minimize FKLD ≡ minimize CE.
- Gradient w.r.t. student logits $z$: $\nabla_z D_\text{KL}(p_T\Vert p_S)=p_S-p_T$ → each entry in $[-1,1]$ → stable at the token level.
```

&nbsp;

### RKLD
- **Name**: Reverse Kullback-Leibler Divergence
- **What**: $D_\text{KL}(p_S\Vert p_T)$: mismatch averaged under the student.
- **Why**: A small student cannot cover a large teacher; FKLD forces it to try.
    - Insufficient capacity → mass smeared onto tokens the teacher only tolerates.
    - → Student samples those tokens at generation time → hallucination.
- **How**: **Zero-forcing**.
    - $p_T(v)\to0$ & $p_S(v)>0$ → $\infty$ penalty.
    - $p_T(v)>0$ & $p_S(v)\to0$ → contributes $0$ → student drops teacher modes for free.

```{note} Math
:class: dropdown
Definition:

$$
D_\text{KL}(p_S\Vert p_T)=\sum_{v\in\mathcal{V}}p_S(v)\log\frac{p_S(v)}{p_T(v)}
$$
- Token level: closed form over $\mathcal{V}$ → same cost as FKLD.
- Sequence level: expectation under $p_S(\cdot|x)$ moves w/ $\theta$ → policy gradient:

    $$
    \nabla_\theta D_\text{KL}(p_S\Vert p_T)=\mathbb{E}_{y\sim p_S(\cdot|x)}\left[\left(\log\frac{p_S(y|x)}{p_T(y|x)}\right)\nabla_\theta\log p_S(y|x)\right]
    $$

    - $\log\frac{p_T}{p_S}$ acts as a reward → RL machinery, ⬆️variance.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class RKLD(nn.Module):
    def forward(self, t_logits, s_logits):
        t_log, s_log = F.log_softmax(t_logits, -1), F.log_softmax(s_logits, -1)
        ## expectation under the STUDENT -> zero-forcing; token level = exact sum over V, no sampling
        return (s_log.exp() * (s_log - t_log)).sum(-1)          ## (B, T)

## Example
print(RKLD()(torch.randn(2, 3, 10), torch.randn(2, 3, 10)).shape)  ## torch.Size([2, 3])
```
````

```{attention} Q&A
:class: dropdown
*Is "RKLD is mode-seeking, FKLD is mode-covering" true?*
- ✅ General: only zero-avoiding (FKLD) vs zero-forcing (RKLD).
- ✅ Typical, ❌guaranteed: restricted family (e.g., unimodal) fit to a multimodal target → FKLD spreads over modes, RKLD locks onto 1.
- ❌ Contested for token-level LLM KD. {cite:p}`wu2024rethinking`
    - FKLD & RKLD share the same optimum & converge to it given enough epochs.
    - At realistic epoch counts: RKLD fits the tail first, FKLD the head.
- → Use the framing to motivate sequence-level objectives, ❌to predict a token-level run.

*Why do the fixes for asymmetric KL converge on bounding the ratio?*
- Both failure modes ← a ratio whose denominator can fall to 0 → unbounded gradient.
- JSD & SKL → mix the numerator into the denominator → ratio capped.
- TVD → drops the ratio entirely.
```

&nbsp;

### JSD
- **Name**: Jensen-Shannon Divergence
- **What**: Weighted sum of the 2 KLs to their mixture.
- **Why**: Bound the KL ratio.
    - Both KLs are unbounded ← denominator $\to0$.
    - Mixture in the denominator → ratio $\leq1/\beta$ or $1/(1-\beta)$ → bounded loss & gradient.
- **How**: $\beta$ interpolates: $\beta\to0$ → FKLD, $\beta\to1$ → RKLD (both after rescaling), $\beta=0.5$ → standard symmetric JSD.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\beta\in[0,1]$: Interpolation weight.
- Misc:
    - $M_\beta=\beta p_T+(1-\beta)p_S$: Mixture.

Definition:

$$
D_{\text{JSD}(\beta)}(p_T\Vert p_S)=\beta\,D_\text{KL}(p_T\Vert M_\beta)+(1-\beta)\,D_\text{KL}(p_S\Vert M_\beta)
$$
- Limits:

    $$
    \lim_{\beta\to0}\frac{1}{\beta}D_{\text{JSD}(\beta)}=D_\text{KL}(p_T\Vert p_S),\qquad \lim_{\beta\to1}\frac{1}{1-\beta}D_{\text{JSD}(\beta)}=D_\text{KL}(p_S\Vert p_T)
    $$

- Bound at $\beta=0.5$: $0\leq D_\text{JSD}\leq\log2$.
- $\sqrt{D_\text{JSD}}$ at $\beta=0.5$ is a metric; $D_\text{JSD}$ itself is not.
- Symmetric only at $\beta=0.5$. In general, roles swap w/ $\beta$:

    $$
    D_{\text{JSD}(\beta)}(P\Vert Q)=D_{\text{JSD}(1-\beta)}(Q\Vert P)
    $$
```

```{tip} Derivation
:class: dropdown
*Why does $\beta\to0$ give FKLD only after dividing by $\beta$?*

1. $M_\beta=p_S+\beta(p_T-p_S)$ → as $\beta\to0$, $M_\beta\to p_S$.
2. Term 1: $\beta D_\text{KL}(p_T\Vert M_\beta)\to\beta D_\text{KL}(p_T\Vert p_S)$ → linear in $\beta$.
3. Term 2: 1st-order term of $D_\text{KL}(P\Vert P+\delta)$ vanishes → $D_\text{KL}(p_S\Vert M_\beta)=O(\beta^2)$.
4. → $D_{\text{JSD}(\beta)}=\beta D_\text{KL}(p_T\Vert p_S)+O(\beta^2)$ → raw value $\to0$; only the rescaled limit is FKLD.
5. → Small $\beta$ shrinks the loss scale → retune LR w/ $\beta$.
```

````{important} Code
:class: dropdown
```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class JSD(nn.Module):
    def __init__(self, beta=0.5):
        super().__init__()
        self.beta = min(max(beta, 1e-9), 1 - 1e-9)   ## keep both logs finite

    def forward(self, t_logits, s_logits):
        t_log, s_log = F.log_softmax(t_logits, -1), F.log_softmax(s_logits, -1)
        b = self.beta
        ## log M = log(b * p_T + (1 - b) * p_S), computed in log-space for stability
        log_m = torch.logsumexp(torch.stack([t_log + math.log(b), s_log + math.log(1 - b)]), dim=0)
        kl_t = (t_log.exp() * (t_log - log_m)).sum(-1)
        kl_s = (s_log.exp() * (s_log - log_m)).sum(-1)
        return b * kl_t + (1 - b) * kl_s                  ## (B, T)

## Example
print(JSD()(torch.randn(2, 3, 10), torch.randn(2, 3, 10)).shape)  ## torch.Size([2, 3])
```
````

```{attention} Q&A
:class: dropdown
*Cost of boundedness?*
- Near-disjoint $p_T$ & $p_S$ → JSD saturates at $\log2$ → gradient ⬇️ toward 0.
- KL in the same regime → large gradient.
- → JSD learns slowly when the student starts far from the teacher.

*Which $\beta$?*
- Capacity close to the teacher → small $\beta$ (FKLD-like, covers modes).
- Large capacity gap → large $\beta$ (RKLD-like, commits to representable modes).
```

&nbsp;

### TVD
- **Name**: Total Variation Distance
- **What**: Half the $L_1$ distance between $p_T$ & $p_S$.
- **Why**: Bounded loss w/o a log ratio.
    - Gradient w.r.t. $p_S$ is a sign, ❌a ratio → 1 outlier token cannot dominate the update.
- **How**: Sum the absolute per-token probability gaps, halve it.

```{note} Math
:class: dropdown
Definition:

$$
D_\text{TV}(p_T\Vert p_S)=\frac{1}{2}\sum_{v\in\mathcal{V}}\left|p_T(v)-p_S(v)\right|=\max_{A\subseteq\mathcal{V}}\left|p_T(A)-p_S(A)\right|
$$
- $A$: Any event (subset of $\mathcal{V}$).
- $0\leq D_\text{TV}\leq1$.
- Pinsker's inequality:

    $$
    D_\text{TV}(P\Vert Q)\leq\sqrt{\tfrac{1}{2}D_\text{KL}(P\Vert Q)}
    $$

    - → Small KL forces small TVD; ❌the reverse. A tiny TVD allows an infinite KL.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TVD(nn.Module):
    def forward(self, t_logits, s_logits):
        ## L1 on PROBABILITIES, not log-probs -> no ratio, bounded in [0, 1]
        return 0.5 * (F.softmax(t_logits, -1) - F.softmax(s_logits, -1)).abs().sum(-1)   ## (B, T)

## Example
print(TVD()(torch.randn(2, 3, 10), torch.randn(2, 3, 10)).shape)  ## torch.Size([2, 3])
```
````

```{attention} Q&A
:class: dropdown
*Why is boundedness a real advantage here?*
- Early in training, teacher & student tails are near-disjoint.
- KL: a token w/ $p_S\to0$ & $p_T>0$ → arbitrarily large term → 1 rare token sets the batch gradient.
- TVD: that term is capped at its probability gap → update tracks the bulk of the distribution.

*Does symmetric beat asymmetric in practice?*
- f-distill: JSD & TVD generally beat SeqKD (FKLD) & ENGINE (RKLD) on 4 generation datasets. {cite:p}`wen2023fdivergence`
- Small human eval (DART, 5 raters, 50 samples): TVD lowest on missing info & hallucination; fluency not significantly different from SeqKD.
- → Neither extreme mode-averaging nor extreme mode-collapsing is ideal.

*Cost?*
- Symmetric → needs expectations under both distributions → teacher & student samples.
- Mitigation: sample the teacher offline once; only the student half stays online.
```

&nbsp;

### SKL
- **Name**: Skew Kullback-Leibler Divergence {cite:p}`lee1999measures`
- **What**: KL w/ the denominator mixed w/ a small share $\alpha$ of the numerator.
- **Why**: Keep FKLD/RKLD behavior, cap the gradient.
    - JSD bounds the ratio but changes the objective's shape at $\beta=0.5$.
    - Small $\alpha$ → coefficient capped at $1/\alpha$ & target ≈ original KL.
- **How**: In $D_\text{KL}(p\Vert q)$, replace $q$ w/ $\alpha p+(1-\alpha)q$.
    - On FKLD → SKL. On RKLD → SRKL.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\alpha\in(0,1)$: Skew coefficient.
- Misc:
    - $\tilde{p}=\alpha p_T+(1-\alpha)p_S$: Skewed mixture.

Definitions:

$$
D_\text{SKL}^{(\alpha)}(p_T,p_S)=D_\text{KL}\left(p_T\ \Vert\ \alpha p_T+(1-\alpha)p_S\right)
$$

$$
D_\text{SRKL}^{(\alpha)}(p_T,p_S)=D_\text{KL}\left(p_S\ \Vert\ (1-\alpha)p_T+\alpha p_S\right)
$$
- $\alpha=0$ → exactly FKLD / RKLD.
- $\alpha=1$ → identically $0$ → keep $\alpha$ small.
- Lee's $\alpha$ is the weight on the other distribution → Lee's $\alpha$ = $1-\alpha$ here.
- Gradient:

    $$
    \nabla_\theta D_\text{SKL}^{(\alpha)}(p_T,p_S)=-(1-\alpha)\sum_{v\in\mathcal{V}}\frac{p_T(v)}{\tilde{p}(v)}\,\nabla_\theta p_S(v)
    $$

    - $\tilde{p}\geq\alpha p_T$ → $\frac{p_T}{\tilde{p}}\leq\frac{1}{\alpha}$ → bounded coefficient, vs unbounded $\frac{p_T}{p_S}$ for plain FKLD.
- SRKL & the $\alpha=0.1$ optimum for LLM KD: DistiLLM. {cite:p}`ko2024distillm`
```

````{important} Code
:class: dropdown
```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SkewKL(nn.Module):
    def __init__(self, alpha=0.1, reverse=False):
        super().__init__()
        self.alpha, self.reverse = alpha, reverse      ## 0 < alpha < 1

    def forward(self, t_logits, s_logits):
        t_log, s_log = F.log_softmax(t_logits, -1), F.log_softmax(s_logits, -1)
        ## SKL: KL(p_T || .), SRKL: KL(p_S || .)
        p, q = (s_log, t_log) if self.reverse else (t_log, s_log)
        ## denominator alpha*p + (1-alpha)*q >= alpha*p -> ratio p/mix <= 1/alpha
        log_mix = torch.logsumexp(torch.stack([p + math.log(self.alpha), q + math.log(1 - self.alpha)]), dim=0)
        return (p.exp() * (p - log_mix)).sum(-1)       ## (B, T)

## Example
t, s = torch.randn(2, 3, 10), torch.randn(2, 3, 10)
print(SkewKL()(t, s).shape, SkewKL(reverse=True)(t, s).shape)  ## torch.Size([2, 3]) torch.Size([2, 3])
```
````

```{attention} Q&A
:class: dropdown
*SKL's $\alpha$ vs JSD's $\beta$: same knob?*
- ❌ $\beta$ interpolates FKLD ↔ RKLD; $\alpha$ stabilizes 1 fixed direction.
- $\beta=0.5$ → a different (symmetric) objective. $\alpha=0.1$ → still FKLD in character, w/ a bounded gradient.
- $\alpha\to1$ → degenerate (loss $\to0$); $\beta\to1$ → RKLD.

*Why $\alpha=0.1$, not smaller?*
- $\alpha$⬆️ → gradient coefficient⬇️ & estimator $L_2$ error bound⬇️ (DistiLLM Thm. 1).
- $\alpha$⬆️ → target drifts from the KL it replaces → $0$ at $\alpha=1$.
- Normalized $L_2$ bound vs $\alpha$ looks convex; minimum at $0.1$ matches the empirical optimum.
- Above $0.1$: SKL degrades mildly, SRKL severely → keep SRKL's $\alpha$ tight.
```

&nbsp;

## On-Policy KD
- **What**: Measure the divergence at positions the student generated.
- **Why**: Teacher-chosen prefixes in training ≠ student-chosen prefixes at inference.
    - Train on $y\sim p_T$, deploy on $y\sim p_S$ → distribution shift in the objective ([exposure bias](sft.md#sft)).
    - Student errors compound over a sequence; a teacher-forced objective never grades a compounded error.
- **How**: Roll out the student → teacher scores those exact tokens → minimize a divergence there.

&nbsp;

### GKD
- **Name**: Generalized Knowledge Distillation {cite:p}`agarwal2023onpolicy`
- **What**: Token-level KD on the student's own samples, w/ a tunable divergence.
- **Why**: Fixed teacher data leaves the student unsupervised where it operates.
    - The states the student visits at inference never get a label.
- **How**:
    1. Sample outputs from the student.
    2. Teacher scores those exact tokens → target distributions.
    3. Minimize a divergence at every position; choose it via [JSD](#jsd)($\beta$).
    4. Mix student-generated & fixed data w/ $\lambda$.
    - Generalizes ImitKD {cite:p}`lin2020autoregressive`, which trained on student trajectories w/ a fixed FKLD-style loss.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $(X,Y)$: Fixed dataset of prompts & target responses.
- Hyperparams:
    - $\lambda\in[0,1]$: Fraction of student-generated data.

Objective:

$$
\mathcal{L}_\text{GKD}(\theta)=(1-\lambda)\,\mathbb{E}_{(x,y)\sim(X,Y)}\left[D(p_T\Vert p_S)(y|x)\right]+\lambda\,\mathbb{E}_{x\sim X}\,\mathbb{E}_{y\sim p_S(\cdot|x)}\left[D(p_T\Vert p_S)(y|x)\right]
$$
- $D(p_T\Vert p_S)(y|x)$: Token-averaged divergence over the positions of $y$.
- $y\sim p_S(\cdot|x)$: Sampled, then detached → no gradient through sampling.
- $\lambda=0$ → supervised KD; $\lambda=1$ → fully on-policy.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

## Reuses JSD from nlp/post/kd.md
class GKD(nn.Module):
    def __init__(self, beta=0.5, lam=0.5):
        super().__init__()
        self.div, self.lam = JSD(beta), lam

    def forward(self, teacher, student, fixed_seq, student_seq, prompt_len):
        ## *_seq: prompt + response ids (B, L); student_seq was sampled by the STUDENT -> detached DATA
        ## positions prompt_len-1 .. L-2 predict the response tokens -> keep only those
        resp = slice(prompt_len - 1, -1)
        with torch.no_grad():
            t_on, t_off = teacher(student_seq)[:, resp], teacher(fixed_seq)[:, resp]
        on = self.div(t_on, student(student_seq)[:, resp]).mean()
        off = self.div(t_off, student(fixed_seq)[:, resp]).mean()
        ## paper picks 1 source per step w/ prob lam; weighting both has the same expected loss
        return (1 - self.lam) * off + self.lam * on

## Example
V, d = 10, 8
lm = lambda: nn.Sequential(nn.Embedding(V, d), nn.Linear(d, V))
fixed, sampled = torch.randint(0, V, (2, 5)), torch.randint(0, V, (2, 5))
print(GKD()(lm(), lm(), fixed, sampled, prompt_len=2).shape)  ## torch.Size([])
```
````

```{attention} Q&A
:class: dropdown
*Why is this not RL, given it samples from the policy?*
- ❌Reward, ❌advantage, ❌policy-gradient estimator.
- Samples are detached input positions; gradient flows only through the divergence there.
- → On-policy data, supervised objective.

*Which $\beta$ & $\lambda$?*
- Best settings are task-dependent → sweep $(\beta,\lambda)$ jointly, ❌fixed constants.

*Cost vs sequence-level KD?*
- Student generation every step + a teacher forward pass on every generated token.
- Generation dominates → overhead scales w/ response length & both model sizes.
- → Pay for it only when exposure bias is the observed failure.

*Relation to [RFT](sft.md#rft)?*
- Both train on self-generated text.
- RFT: verifier filters; target = the sample itself (one-hot); rejected samples dropped.
- GKD: no filter; target = teacher's full distribution; every sample used.
- → RFT needs a verifier; GKD needs a teacher.
```

&nbsp;

### MiniLLM
- **What**: Sequence-level [RKLD](#rkld) minimization via policy gradient. {cite:p}`gu2023minillm`
- **Why**: Token-level RKLD never charges the student for a whole degenerate continuation.
    - Sequence-level RKLD does, but its expectation is under $p_S$ → naive REINFORCE → high variance.
- **How**:
    1. Sample from a teacher-mixed distribution, ❌the raw student.
    2. Reward each token by $\log\frac{p_T}{p_S}$; length-normalize the future reward.
    3. Split the gradient: exact single-step term + REINFORCE long-term term.
    4. Importance-weight to correct for the mixed sampler.
    5. Add an LM loss on pretraining text → keeps general benchmark performance.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\alpha$: Teacher mix-in strength.
- Misc:
    - $r_t=\sum_{v\in\mathcal{V}}p_S(v|y_{<t},x)\log\frac{p_T(v|y_{<t},x)}{p_S(v|y_{<t},x)}$: Single-step term (exact over $\mathcal{V}$).

Objective:

$$
\mathcal{L}(\theta)=D_\text{KL}\left(p_S(\cdot|x)\ \Vert\ p_T(\cdot|x)\right)
$$

Process:
1. Sample $y$ from the mixed sampler:

    $$
    \tilde{p}(y_t|y_{<t},x)=\alpha\,p_T(y_t|y_{<t},x)+(1-\alpha)\,p_S(y_t|y_{<t},x)
    $$

    - $\alpha=0.2$ in the paper.
2. Importance weight, exact (unbiased) → implemented:

    $$
    w_t=\prod_{t'=1}^{t}\frac{p_S(y_{t'}|y_{<t'},x)}{\tilde{p}(y_{t'}|y_{<t'},x)}\quad\longrightarrow\quad w_t\approx\frac{p_S(y_t|y_{<t},x)}{\tilde{p}(y_t|y_{<t},x)}
    $$

    - Product accumulates per-step variance → truncated to the single-step ratio → ⬇️variance, ⬆️bias.
3. Length-normalized future reward:

    $$
    R_{t+1}^\text{Norm}=\frac{1}{|y|-t}\sum_{t'=t+1}^{|y|}\log\frac{p_T(y_{t'}|y_{<t'},x)}{p_S(y_{t'}|y_{<t'},x)}
    $$

    - Divisor = #summed terms. The paper writes $|y|-t-1$, off by 1 vs its own sum range.
4. Gradient:

    $$
    \nabla_\theta\mathcal{L}=-\mathbb{E}_{x\sim\mathcal{D},\,y\sim\tilde{p}(\cdot|x)}\left[\sum_{t=1}^{|y|}w_t\left(\underbrace{\nabla_\theta r_t}_{\text{single-step}}+\underbrace{R_{t+1}^\text{Norm}\,\nabla_\theta\log p_S(y_t|y_{<t},x)}_{\text{long-term}}\right)\right]
    $$

    - Single-step part: exact ← summed over $\mathcal{V}$.
    - Long-term part: REINFORCE → carries the variance; implemented w/ a PPO-style clipped ratio.
5. Update w/ this gradient + $\nabla_\theta\mathcal{L}_\text{PT}$ (LM loss on pretraining text).
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniLLM(nn.Module):
    ## surrogate loss: its gradient = single-step + normalized long-term terms
    ## omits PPO clipping & the pretraining LM loss
    def __init__(self, alpha=0.2):
        super().__init__()
        self.alpha = alpha

    def forward(self, s_logits, t_logits, y):
        ## s_logits, t_logits: (B, T, V) at the positions that predict y: (B, T)
        ## y was sampled from the teacher-mixed sampler -> detached data
        s_log, t_log = F.log_softmax(s_logits, -1), F.log_softmax(t_logits, -1)
        ## single-step: r_t = -KL(p_S || p_T) at each position, exact over V, grad flows
        r = (s_log.exp() * (t_log - s_log)).sum(-1)                         ## (B, T)
        lp_s = s_log.gather(-1, y.unsqueeze(-1)).squeeze(-1)                ## log p_S(y_t)
        with torch.no_grad():
            lp_t = t_log.gather(-1, y.unsqueeze(-1)).squeeze(-1)
            ## truncated importance weight p_S / p_mix at the sampled token
            w = lp_s.exp() / (self.alpha * lp_t.exp() + (1 - self.alpha) * lp_s.exp())
            ## future reward: mean of log(p_T / p_S) over t' > t
            ratio = lp_t - lp_s
            future = ratio.flip(-1).cumsum(-1).flip(-1) - ratio
            n_future = (y.size(1) - 1 - torch.arange(y.size(1), device=y.device)).clamp_min(1)
            R = future / n_future                                           ## last position -> 0
        return -(w * (r + R * lp_s)).sum(-1).mean()

## Example
B, T, V = 2, 4, 10
y = torch.randint(0, V, (B, T))
print(MiniLLM()(torch.randn(B, T, V), torch.randn(B, T, V), y).shape)  ## torch.Size([])
```
````

```{attention} Q&A
:class: dropdown
*Why teacher-mixed sampling?*
- Pure-student sampling → reward hacking: degenerate outputs (repeated phrases) that the teacher scores highly.
- Small students suffer most ← their raw samples are worst.
- Mixing in $p_T$ suppresses those trajectories at the source.
- Importance weight corrects the bias; exact only in its full-product form, which is not the one implemented.

*Why length-normalize?*
- $R_{t+1}$ is a sum of log-ratios; each has expectation $-D_\text{KL}(p_S\Vert p_T)\leq0$ under $p_S$.
- → Longer continuations get a more negative reward in expectation → bias toward short responses.
- Divide by the remaining length → removes the length dependence.

*Why decompose instead of plain REINFORCE?*
- $r_t$ is a full-vocab expectation → closed form → zero sampling variance for that part.
- Early-token errors compound down the sequence → single-step term is high-leverage → worth computing exactly.

*Vs GKD w/ $\beta\to1$?*
- GKD: token-level RKLD, detached samples, supervised gradient.
- MiniLLM: sequence-level RKLD, gradient through the sampling distribution → RL.
- → GKD cheaper & more stable; MiniLLM optimizes the deployed object.
```

&nbsp;

### DistiLLM
- **What**: [SKL](#skl) + adaptive off-policy student outputs from a replay buffer. {cite:p}`ko2024distillm`
- **Why**: On-policy KD is correct but wasteful & noisy.
    - Fresh student-generated outputs (SGO) every step → up to 5.5× runtime, discarded after 1 update.
    - Early SGO are poor & unfamiliar to the teacher → noisy targets.
- **How**:
    1. Loss: SKL or SRKL w/ $\alpha=0.1$ → bounded gradient.
    2. Each step: w/ prob $\phi$, batch from the SGO replay buffer; else from fixed data.
    3. Raise $\phi$ when validation loss rises.
    4. Generate new SGO w/ a prob that decays linearly over training → reuse old SGO late.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\phi$: Prob of an SGO batch (init $0$).
    - $N_\phi$: #increments from $0$ to $1$.
    - $\varepsilon$: Validation-loss tolerance.
- Misc:
    - $s$: Training step.
    - $S$: Total steps.
    - $\mathcal{D}_R$: FIFO replay buffer of SGO (capacity 1000).

Process:
1. Draw $u\sim\text{Unif}(0,1)$.
2. If $u<\phi\left(1-\frac{s}{S}\right)$: generate SGO from the current student → push into $\mathcal{D}_R$.
3. If $u<\phi$: batch from $\mathcal{D}_R$. Else: batch from $\mathcal{D}$.
4. Update $\theta$ w/ $D_\text{SKL}^{(\alpha)}$ or $D_\text{SRKL}^{(\alpha)}$.
5. At each validation: if $\mathcal{L}_\text{val}>\mathcal{L}_\text{ref}+\varepsilon$ → $\phi\leftarrow\min(\phi+1/N_\phi,1)$, $\mathcal{L}_\text{ref}\leftarrow\mathcal{L}_\text{val}$.
- $\mathcal{L}_\text{ref}$: Reference validation loss.
- $1-\frac{s}{S}$: Replay ratio → fresh SGO early, reused SGO late.
```

```{attention} Q&A
:class: dropdown
*Why is off-policy acceptable when GKD argues for on-policy?*
- Target = teacher distribution at given positions → does not change as $\theta$ moves.
- → Stale samples are still validly labeled; no value/advantage estimate goes stale.
- State distribution still drifts → off-policy bias grows as buffered samples lag the student.
- Buffer size trade-off: too small → overfitting; too large → stale samples. 1000 best on most tasks.

*Why start $\phi$ at 0 & raise it?*
- Early SGO are near-garbage → noisy positions dominate.
- Rising validation loss → training-inference mismatch → more SGO helps.
- → Curriculum, ❌a fixed mixture.

*Why decay the generation rate?*
- Early: student changes fast → old SGO go stale → generate often.
- Late: student near convergence → old SGO stay valid → reuse.

*Claimed gain?*
- 2.5–4.3× training speedup vs GKD & MiniLLM, at comparable or better quality.
```

&nbsp;

### ULD
- **Name**: Universal Logit Distillation {cite:p}`boizard2024towards`
- **What**: Cross-tokenizer logit KD via the $L_1$ distance between sorted probability vectors.
- **Why**: Every divergence above sums coordinate-wise over 1 shared $\mathcal{V}$.
    - 2 tokenizers → index $v$ means different tokens on each side → sum meaningless.
    - → Logit KD was limited to same-tokenizer pairs.
- **How**:
    1. Pair positions by index, up to the shorter of the 2 tokenizations.
    2. Sort each side's probability vector descending → drop token identity, keep shape.
    3. Zero-pad the smaller vocab → equal length.
    4. Sum the rank-wise absolute gaps → add to CE w/ weight $\lambda$.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\lambda$: ULD weight ($1.5$ in the paper).
- Misc:
    - $p^\downarrow$: Probability vector sorted descending, zero-padded to length $\max(|\mathcal{V}_T|,|\mathcal{V}_S|)$.
    - $L$: $\min(|y^S|,|y^T|)$, the shorter tokenized length.

Objective:

$$
\mathcal{L}_\text{ULD}=\sum_{t=1}^{L}\left[\text{CE}(t)+\lambda\sum_{r=1}^{\max(|\mathcal{V}_T|,|\mathcal{V}_S|)}\left|p_S^\downarrow(r)-p_T^\downarrow(r)\right|\right]
$$
- $r$: Rank, ❌token id.
- Inner sum $=2D_\text{TV}$ between the 2 sorted profiles.
- Paper presents it as a closed-form $\mathcal{W}_1$ (padding + uniform transport cost); that equivalence is loose → treat it as a rank-sorted $L_1$ heuristic.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ULD(nn.Module):
    def __init__(self, lam=1.5):
        super().__init__()
        self.lam = lam

    def forward(self, s_logits, t_logits, labels):
        ## s_logits (B, T_s, V_s), t_logits (B, T_t, V_t): vocab sizes MAY differ
        L = min(s_logits.size(1), t_logits.size(1))        ## pair positions by index
        s, t = s_logits[:, :L], t_logits[:, :L]
        ## sort descending -> token identity gone, distribution shape kept
        p_s = F.softmax(s, -1).sort(-1, descending=True).values
        p_t = F.softmax(t, -1).sort(-1, descending=True).values
        ## zero-pad the smaller vocab -> equal support size
        V = max(p_s.size(-1), p_t.size(-1))
        p_s, p_t = F.pad(p_s, (0, V - p_s.size(-1))), F.pad(p_t, (0, V - p_t.size(-1)))
        w1 = (p_s - p_t).abs().sum(-1).mean()
        ce = F.cross_entropy(s.flatten(0, 1), labels[:, :L].flatten())   ## student's own tokens
        return ce + self.lam * w1

## Example
s, t = torch.randn(2, 5, 10), torch.randn(2, 4, 16)
print(ULD()(s, t, torch.randint(0, 10, (2, 5))).shape)  ## torch.Size([])
```
````

```{attention} Q&A
:class: dropdown
*What is matched, if not tokens?*
- The sorted profile: how peaked vs flat the teacher is, ❌which token it prefers.
- → Transfers confidence & entropy structure; drops the token identity behind dark knowledge.

*Why does it still help?*
- CE on the text already fixes which token.
- ULD adds the shape on top → more than [Sequence-Level KD](#sequence-level-kd), less than same-tokenizer token-level KD.

*Weak point?*
- Positions paired by index → position $t$ can cover different text on each side.
- Uniform transport cost → ignores which tokens are similar.

*When not worth it?*
- Shared tokenizer → ❌ULD; a vocab-aligned divergence keeps token identity.
- Close tokenizers → explicit token alignment (merge/split spans, min edit distance over token strings) keeps identity, at an alignment-error cost.
```

&nbsp;

## Practice
- **What**: Choose a method along the 3 axes.

```{dropdown} Table: KD Methods
| Method | Sampled from | Target | Divergence | Needs |
|:--|:--|:--|:--|:--|
| Token-Level KD | Fixed corpus | Full distribution | FKLD | Logits, shared tokenizer |
| Sequence-Level KD | Teacher (offline) | Token one-hot | FKLD (sequence) | Text only |
| CoT Distillation | Teacher (offline) | Rationale + label | FKLD (sequence) | Text only |
| Feature-Level KD | Fixed corpus | Hidden states / attention | MSE / KL | Hidden states, layer mapping |
| ULD | Teacher (offline) | Sorted distribution | Sorted $L_1$ | Logits, any tokenizer |
| f-distill | Teacher (offline) + student | Full distribution | JSD / TVD | Logits + student sampling |
| GKD | Student (+ fixed) | Full distribution | JSD($\beta$) | Logits + student sampling |
| MiniLLM | Teacher-mixed | Full distribution | RKLD (sequence) | Logits + RL loop |
| DistiLLM | Student (replayed) | Full distribution | SKL / SRKL | Logits + student sampling |
```

```{attention} Q&A
:class: dropdown
*Decision order?*
1. ❌Logits → [Sequence-Level KD](#sequence-level-kd).
2. ✅Logits, ❌shared tokenizer → [ULD](#uld), or align tokenizations if they are close.
3. ✅Both, small capacity gap → [Token-Level KD](#token-level-kd) w/ [FKLD](#fkld).
4. Large capacity gap → skew or symmetrize ([SKL](#skl), [JSD](#jsd), [TVD](#tvd)) before on-policy.
5. Fine on teacher text, degrades on own rollouts → [on-policy](#on-policy-kd) (GKD; DistiLLM for cost).
6. Verifiable task → [RFT](sft.md#rft) can beat all of them w/o a teacher.

*Most common mistake?*
- Exotic divergence when the real bottleneck is data coverage.
- No divergence supervises a prompt the corpus never contains.

*How to evaluate?*
- ❌The distillation loss: a divergence on the training distribution, not a capability measure.
- Task metrics + a diversity measure ← mode-seeking objectives buy quality by shedding coverage.
```
