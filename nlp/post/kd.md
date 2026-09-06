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
Compressing a teacher's behavior into a smaller student. Organized by the 3 choices that define any KD method: **what** to match, **which** divergence, and **where** to measure it.

Notations:
- $x$: Prompt (input token seq)
- $y$: Response (output token seq)
- $y_t$: $t$-th response token
- $y_{<t}$: Response prefix before position $t$
- $|y|$: Response length (#tokens)
- $\mathcal{V}$: Vocabulary
- $\mathcal{D}$: Dataset
- $\theta$: Student params
- $p_T$: Teacher distribution — $p_T(\cdot|x,y_{<t})$ over $\mathcal{V}$, or $p_T(y|x)$ over full sequences
- $p_S$: Student distribution (parameterized by $\theta$), same 2 forms
- $D(\cdot\Vert\cdot)$: A divergence

&nbsp;

## Token-Level KD
- **What**: Matching the teacher's next-token distribution at every position of a fixed corpus. {cite:p}`hinton2015distilling`
- **Why**: The teacher's behavior is wanted at the student's cost.
    - Serving a frontier model is expensive, & capability per param is not fixed.
    - A one-hot label carries at most $\log_2|\mathcal{V}|\approx17$ bits per position; the teacher's full distribution carries the relative probabilities of every **wrong** token too.
    - Those relatives encode the teacher's similarity structure ("dark knowledge") — the part a hard label throws away.
- **How**:
    1. Teacher forward pass over the corpus → a distribution over $\mathcal{V}$ at every position.
    2. Soften both sides w/ temperature $\tau$.
    3. Minimize a [divergence](#divergence) against the student's distribution, optionally mixed w/ hard-label CE.
    4. Teacher-forced throughout → ❌sampling, ❌rollouts.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $z$: Student logits.
- Hyperparams:
    - $\tau$: Temperature.
    - $\alpha$: Weight on the soft (teacher) term.

Objective:

$$
\mathcal{L}_\text{KD}(\theta)=\mathbb{E}_{(x,y)\sim\mathcal{D}}\left[\frac{1}{|y|}\sum_{t=1}^{|y|}D_\text{KL}\left(p_T\ \Vert\ p_S\right)\right]
$$

W/ temperature & hard labels:

$$
\mathcal{L}=\alpha\tau^2\,D_\text{KL}\left(p_T^\tau\ \Vert\ p_S^\tau\right)+(1-\alpha)\,\text{CE}(y,p_S)
$$
- $p^\tau$: Softmax over logits divided by $\tau$.
- $\tau^2$: Rescaling — softened gradients shrink as $\tau^{-2}$, so this keeps the two terms comparable.

Gradient w.r.t. student logits (at $\tau=1$):

$$
\nabla_z\,D_\text{KL}(p_T\Vert p_S)=p_S-p_T,\qquad \nabla_z\,\text{CE}(y,p_S)=p_S-\mathbf{e}_{y_t}
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

*Why match logits instead of hidden states?*
- Logits live in a **shared** space ($\mathcal{V}$) → directly comparable between any 2 models w/ the same tokenizer.
- Hidden states don't: widths differ, & the coordinates carry no shared meaning across independently trained models → needs a learned projection + an arbitrary alignment.
- Feature matching (the BERT-compression line: TinyBERT, MiniLM) works when student & teacher share an architecture family & are trained together; rare across LLM checkpoints.

*Can the student ever beat the teacher?*
- ✅ In restricted senses: distilling an expensive procedure (CoT, search, ensembling) into one forward pass, or distilling a *filtered* teacher.
- ✅ **Weak-to-strong**: a strong student supervised by a much weaker teacher recovers a large fraction of the gap — supervision **elicits** latent capability rather than transferring it. {cite:p}`burns2023weak`
- ❌ In general: matching a distribution cannot exceed it.

*Different tokenizers?*
- Token-level KD is undefined ← the 2 vocab index spaces don't correspond.
- → [Sequence-Level KD](#sequence-level-kd) instead: it consumes **text**, which is tokenizer-agnostic.
- → Or keep the logits & drop the index: match **sorted** probability vectors ([ULD](#uld)), or align the 2 tokenizations first (merge/split spans, min-edit-distance over token strings) → approximate, & the alignment becomes its own error source.
```

&nbsp;

## Sequence-Level KD
- **What**: SFT on teacher-generated sequences. {cite:p}`kim2016sequence`
- **Why**: Token-level KD needs logits & a fixed target corpus, and matches the wrong object.
    - Frontier teachers expose text, ❌logits.
    - What matters at inference is the distribution over **sequences**, not per-position marginals conditioned on a ground-truth prefix.
- **How**:
    1. Generate responses from the teacher for each prompt (sample, or take the beam-search mode).
    2. Optionally filter w/ a verifier.
    3. Plain SFT on (prompt, teacher response).

```{note} Math
:class: dropdown
Objective:

$$
\mathcal{L}_\text{SeqKD}(\theta)=D_\text{KL}\left(p_T(\cdot|x)\ \Vert\ p_S(\cdot|x)\right)=-\mathbb{E}_{y\sim p_T(\cdot|x)}\left[\log p_S(y|x)\right]-H\left(p_T(\cdot|x)\right)
$$
- $H(p_T(\cdot|x))$: Teacher's sequence entropy — $\perp\theta$ → drops from the gradient.
- → Minimizing sequence-level forward KL $\equiv$ MLE on teacher samples.

Sum is over all $y\in\mathcal{V}^*$ → intractable. 2 tractable surrogates, both dropping the $\perp\theta$ entropy term:

$$
\hat{\mathcal{L}}_\text{MC}=-\frac{1}{N}\sum_{n=1}^{N}\log p_S(y^{(n)}|x),\qquad y^{(n)}\sim p_T(\cdot|x)
$$
- Monte-Carlo CE → an **unbiased** estimator of the KL up to the constant $-H(p_T)$.

$$
\hat{\mathcal{L}}_\text{mode}=-\log p_S(\hat{y}|x),\qquad \hat{y}\approx\arg\max_y p_T(y|x)
$$
- $\hat{y}$: Beam-search output — the original formulation. A point-mass **approximation** of $p_T(\cdot|x)$, ❌an estimator of it.
```

```{attention} Q&A
:class: dropdown
*Why does it work w/o any logits?*
- Sampling $y\sim p_T$ then doing MLE minimizes $D_\text{KL}(p_T\Vert p_S)$ at the **sequence** level, Monte-Carlo estimated.
- The original formulation instead approximates the teacher's sequence distribution by a point mass at its **mode** (beam output) → SFT on that single output.
- → Either way the teacher's distribution enters through its samples, ❌its probabilities.

*Trade-off vs token-level KD?*
- ✅Black-box, tokenizer-agnostic, reuses the SFT trainer, teacher runs once per prompt.
- ❌ ~$\log_2|\mathcal{V}|$ bits per position instead of a full distribution → far weaker signal per token.

*Why is only forward KL available here, when token-level KD can use any divergence?*
- Every divergence needs $\mathbb{E}_{y\sim P}[\cdot]$ for some $P$; only $P=p_T$ is samplable **once, offline**.
- Reverse KL needs $y\sim p_S$, which changes every step → re-sampling + a policy gradient ([MiniLLM](#minillm)).
- Symmetric ones ([JSD](#jsd), [TVD](#tvd)) need both → both sampling costs, halved by drawing the teacher's samples offline once.

*What is the failure mode?*
- It is SFT → exposure bias returns in full: the student only ever sees teacher trajectories, which may be very improbable under itself.
- Sharpening on the teacher's mode also collapses diversity relative to the teacher.

*Is this just "SFT on synthetic data"?*
- Mechanically identical. Most open "distilled" models are exactly this.
- The distinction that matters is the **filter**: unfiltered teacher output inherits every teacher error.
```

&nbsp;

### CoT Distillation
- **Name**: Chain-of-Thought Distillation {cite:p}`hsieh2023distilling`
- **What**: SFT on teacher **rationales**, not only teacher answers.
- **Why**: The answer alone omits the computation that produced it.
    - A label says *what*, ❌*how* → the student must rediscover the intermediate procedure from scratch.
    - → Standard KD needs a lot of data to fit a reasoning task the teacher solves in a few steps.
- **How**:
    1. Few-shot prompt the teacher to emit rationale + answer.
    2. Train the student in a **multi-task** setup: 1 prefix asks for the label, another asks for the rationale.
    3. → Rationales supervise training but need not be generated at inference.

```{attention} Q&A
:class: dropdown
*Why multi-task instead of just concatenating rationale + answer?*
- Concatenation forces the student to generate the rationale at **inference** → ⬆️latency & an extra failure mode.
- Separate prefixes over shared weights → rationale acts as pure training-time supervision.

*How much does it buy?*
- 770M T5 > 540B few-shot PaLM on a benchmark, using 80% of the data — while plain fine-tuning of the same T5 fails to match at 100%.
- → The gain is in **sample** efficiency, not only in param count.

*What is the risk?*
- The rationale is unverified → a correct answer reached by a wrong rationale trains the wrong procedure.
- Teacher rationales are post-hoc text, ❌a faithful trace of the teacher's computation.
```

&nbsp;

## Divergence
- **What**: The choice of $D$ in $\min_\theta D(p_T\Vert p_S)$.
- **Why**: All of them are 0 iff $p_S=p_T$ — that equality is unreachable under a capacity gap, & they disagree completely about where to spend the shortfall.
- **How**: Each weights the teacher-student mismatch by a different measure → different behavior when the student **cannot** match.

&nbsp;

### FKLD
- **Name**: Forward Kullback-Leibler Divergence
- **What**: $D_\text{KL}(p_T\Vert p_S)$ — mismatch averaged under the **teacher**.
- **Why**: The default, & the only one available for free.
    - $\equiv$ CE against a soft label, up to a $\theta$-independent constant → drops into any SFT trainer unchanged.
    - Only divergence whose sequence-level form is estimable from **offline** teacher samples.
- **How**: **Zero-avoiding**. $p_T(v)>0$ w/ $p_S(v)\to0$ → $\log$ ratio $\to\infty$ → the student must keep mass wherever the teacher has any.

```{note} Math
:class: dropdown
Definition:

$$
D_\text{KL}(p_T\Vert p_S)=\sum_{v\in\mathcal{V}}p_T(v)\log\frac{p_T(v)}{p_S(v)}=\underbrace{H(p_T,p_S)}_{\text{CE}}-\underbrace{H(p_T)}_{\perp\theta}
$$

Gradient w.r.t. student logits $z$:

$$
\nabla_z D_\text{KL}(p_T\Vert p_S)=p_S-p_T
$$
- Bounded in $[-1,1]$ per entry → stable at the token level.
```

&nbsp;

### RKLD
- **Name**: Reverse Kullback-Leibler Divergence
- **What**: $D_\text{KL}(p_S\Vert p_T)$ — mismatch averaged under the **student**.
- **Why**: A small student cannot cover a large teacher, & FKLD makes it try anyway.
    - Covering $|\mathcal{V}|$-wide teacher mass w/ insufficient capacity → probability smeared onto tokens the teacher merely tolerates.
    - → Hallucination at generation time: the student samples what it was forced to hedge on.
- **How**: **Zero-forcing**. $p_T(v)\to0$ w/ $p_S(v)>0$ → $\infty$ penalty; $p_T(v)>0$ w/ $p_S(v)\to0$ → contributes $0$ → the student may drop teacher modes for free.

```{note} Math
:class: dropdown
Definition:

$$
D_\text{KL}(p_S\Vert p_T)=\sum_{v\in\mathcal{V}}p_S(v)\log\frac{p_S(v)}{p_T(v)}
$$

Token-level: closed form over $\mathcal{V}$ → same cost as FKLD.

Sequence-level: $\mathbb{E}_{y\sim p_S(\cdot|x)}$ → the expectation moves w/ $\theta$ → policy gradient:

$$
\nabla_\theta D_\text{KL}(p_S\Vert p_T)=\mathbb{E}_{y\sim p_S(\cdot|x)}\left[\left(\log\frac{p_S(y|x)}{p_T(y|x)}\right)\nabla_\theta\log p_S(y|x)\right]
$$
- $\log\frac{p_T}{p_S}$ acts as a **reward** → RL machinery, ⬆️variance.
```

```{attention} Q&A
:class: dropdown
*Is "RKLD is mode-seeking, FKLD is mode-covering" actually true?*
- ✅ As a property of the **exact minimizer** of each divergence under a constrained family — textbook, provable.
- ❌ Contested for **token-level** LLM KD: FKLD & RKLD share the same optimum & converge to it given enough epochs; the observed difference at realistic epoch counts is that RKLD moves the **tail** first & FKLD the **head**. {cite:p}`wu2024rethinking`
- → Treat the framing as motivation for the sequence-level objective; ❌as a prediction about what a token-level run will do.

*Why do all the asymmetric-KL fixes converge on mixing?*
- Both failure modes come from a **ratio w/ a free-falling denominator** → unbounded gradient.
- Mixing the denominator w/ the numerator floors it → JSD, TVD, SKL are all the same repair applied differently.
```

&nbsp;

### JSD
- **Name**: Jensen-Shannon Divergence
- **What**: Average of the 2 KLs against their **mixture**.
- **Why**: Both KLs are unbounded & pick a side; the failure is the ratio's denominator hitting 0.
    - Mixing puts the other distribution in the denominator → the ratio is capped → bounded loss & bounded gradient.
    - At $\beta=0.5$ it is symmetric → penalizes covering-failure & spurious-mass **equally**.
- **How**: Interpolate w/ $\beta$: $\beta\to0$ recovers FKLD, $\beta\to1$ recovers RKLD (both only after rescaling), $\beta=0.5$ is standard JSD.

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

Limits (up to scale):

$$
\lim_{\beta\to0}\frac{1}{\beta}D_{\text{JSD}(\beta)}=D_\text{KL}(p_T\Vert p_S),\qquad \lim_{\beta\to1}\frac{1}{1-\beta}D_{\text{JSD}(\beta)}=D_\text{KL}(p_S\Vert p_T)
$$

Bound at $\beta=0.5$:

$$
0\leq D_\text{JSD}(p_T\Vert p_S)\leq\log 2
$$
- $\sqrt{D_\text{JSD}}$ is a **metric** (triangle inequality holds); $D_\text{JSD}$ itself is not.

Symmetry holds **only** at $\beta=0.5$; in general the roles swap w/ $\beta$:

$$
D_{\text{JSD}(\beta)}(P\Vert Q)=D_{\text{JSD}(1-\beta)}(Q\Vert P)
$$
```

```{tip} Derivation
:class: dropdown
*Why does $\beta\to0$ give FKLD only after dividing by $\beta$?*

1. $M_\beta=p_S+\beta(p_T-p_S)$ → as $\beta\to0$, $M_\beta\to p_S$.
2. Term 1: $\beta D_\text{KL}(p_T\Vert M_\beta)\to\beta D_\text{KL}(p_T\Vert p_S)$ → linear in $\beta$.
3. Term 2: 2nd-order Taylor of KL around a matched pair → $D_\text{KL}(p_S\Vert M_\beta)=O(\beta^2)$ ← 1st-order term of $D_\text{KL}(P\Vert P+\delta)$ vanishes.
4. → $D_{\text{JSD}(\beta)}=\beta D_\text{KL}(p_T\Vert p_S)+O(\beta^2)$, so the raw value $\to0$ & only the **rescaled** limit is FKLD.
5. → Practical consequence: small $\beta$ shrinks the loss scale → LR must be retuned w/ $\beta$, ❌held fixed.
```

&nbsp;

### TVD
- **Name**: Total Variation Distance
- **What**: Half the $L_1$ distance between $p_T$ & $p_S$. {cite:p}`wen2023fdivergence`
- **Why**: JSD is bounded but still log-based; a pure $L_1$ discrepancy is bounded **and** flat-gradient.
    - $\nabla$ is a sign, ❌a ratio → 1 outlier token cannot dominate the update.
    - Symmetric → mode-balancing: neither averaging (FKLD) nor collapsing (RKLD).
- **How**: Sum the absolute per-token probability gaps → the largest achievable disagreement in assigning any event.

```{note} Math
:class: dropdown
Definition:

$$
D_\text{TV}(p_T\Vert p_S)=\frac{1}{2}\sum_{v\in\mathcal{V}}\left|p_T(v)-p_S(v)\right|=\max_{A\subseteq\mathcal{V}}\left|p_T(A)-p_S(A)\right|
$$
- $A$: Any event (subset of $\mathcal{V}$).
- $0\leq D_\text{TV}\leq1$.

Pinsker's inequality:

$$
D_\text{TV}(P\Vert Q)\leq\sqrt{\tfrac{1}{2}D_\text{KL}(P\Vert Q)}
$$
- → KL controls TVD; ❌the reverse. A tiny TVD is compatible w/ an infinite KL.
```

```{attention} Q&A
:class: dropdown
*Why is a bounded divergence a real advantage here, not just aesthetics?*
- Teacher & student have **near-disjoint** low-probability tails early in training.
- Under KL, a token where $p_S\to0$ but $p_T>0$ contributes an arbitrarily large term → the batch gradient is set by 1 rare token.
- Under TVD that term is capped at its probability gap → the update tracks the bulk of the distribution.

*Does symmetric beat asymmetric in practice?*
- On generation tasks, yes: JSD & TVD outperform both SeqKD (FKLD) & ENGINE (RKLD), w/ human raters scoring TVD lowest on missing information & hallucination at equal fluency. {cite:p}`wen2023fdivergence`
- → Evidence that neither extreme mode-averaging nor extreme mode-collapsing is what you want.

*What is the cost?*
- Symmetric → needs expectations under **both** distributions → teacher samples *and* student samples.
- Mitigation: sample the teacher **offline** once; only the student half stays online.
```

&nbsp;

### SKL
- **Name**: Skew Kullback-Leibler Divergence {cite:p}`ko2024distillm`
- **What**: KL against a mixture skewed toward the numerator.
- **Why**: JSD floors the denominator but pays for it by changing the objective's shape at $\beta=0.5$.
    - Wanted: FKLD's (or RKLD's) behavior, minus the exploding gradient.
    - → Mix in only a **small** $\alpha$ → the gradient coefficient is capped at $1/\alpha$, & the target is still essentially the original KL.
- **How**: Replace the denominator $q$ w/ $\alpha p+(1-\alpha)q$ → SKL; the same trick on RKLD → SRKL.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\alpha\in[0,1]$: Skew (self-mixing) coefficient.

Definitions:

$$
D_\text{SKL}^{(\alpha)}(p_T,p_S)=D_\text{KL}\left(p_T\ \Vert\ \alpha p_T+(1-\alpha)p_S\right)
$$

$$
D_\text{SRKL}^{(\alpha)}(p_T,p_S)=D_\text{KL}\left(p_S\ \Vert\ (1-\alpha)p_T+\alpha p_S\right)
$$
- $\alpha=0$ → exactly FKLD / RKLD respectively.
- $\alpha=1$ → identically $0$ (the 2 arguments coincide) → $\alpha$ must stay small.
- Best empirical value for both: $\alpha=0.1$.

Gradient:

$$
\nabla_\theta D_\text{SKL}^{(\alpha)}(p_T,p_S)=-(1-\alpha)\sum_{v\in\mathcal{V}}\mathbf{r}_{p_T,\tilde{p}}(v)\,\nabla_\theta p_S(v),\qquad \tilde{p}=\alpha p_T+(1-\alpha)p_S
$$
- $\mathbf{r}_{a,b}=a/b$: Density ratio.
- $\tilde{p}\geq\alpha p_T$ → $\mathbf{r}_{p_T,\tilde{p}}\leq1/\alpha$ → **bounded** coefficient, vs unbounded $\mathbf{r}_{p_T,p_S}$ for plain KL.

The skew divergence itself is due to {cite:t}`lee2001effectiveness`; SRKL & the LLM-KD analysis are DistiLLM's.
```

```{attention} Q&A
:class: dropdown
*SKL's $\alpha$ vs GKD's $\beta$ — same knob?*
- ❌Opposite meaning. $\beta$ **interpolates** FKLD $\leftrightarrow$ RKLD; $\alpha$ **stabilizes** one fixed direction.
- $\beta=0.5$ → a genuinely different (symmetric) objective. $\alpha=0.1$ → still FKLD in character, just gradient-clipped by construction.
- → $\alpha\to1$ is degenerate (loss $\to0$), $\beta\to1$ is not.

*Why $\alpha=0.1$ & not smaller?*
- $\alpha$⬆️ → gradient coefficient⬇️ (stability) **and** $L_2$ error between the empirical & true divergence⬇️ (their Thm. 1).
- $\alpha$⬆️ also drags the target away from the KL it is standing in for, → $0$ at $\alpha=1$.
- Their normalized $L_2$ (estimator error ÷ gradient coefficient) bottoms out at $0.1$, matching the accuracy optimum on their tasks.
- SRKL degrades faster than SKL as $\alpha$ grows past it → keep SRKL's $\alpha$ tight.
```

&nbsp;

## On-Policy KD
- **What**: Measuring the divergence at positions the **student** generated.
- **Why**: The 2 preceding axes both evaluate on teacher-chosen prefixes, which the student never sees at inference.
    - Train on $y\sim p_T$, deploy on $y\sim p_S$ → distribution shift baked into the objective ([exposure bias](sft.md#sft)).
    - The student's own errors compound over a sequence, & no teacher-forced objective ever grades a compounded error.
- **How**: Roll out the student → teacher scores those exact tokens → minimize a divergence there.

&nbsp;

### GKD
- **Name**: Generalized Knowledge Distillation {cite:p}`agarwal2023onpolicy`
- **What**: Token-level KD on the **student's own** samples, w/ a tunable divergence.
- **Why**: Fixed teacher data leaves the student unsupervised exactly where it operates.
    - Trained on teacher trajectories, evaluated on its own → the states it actually visits were never labeled.
    - A low-capacity student cannot match the teacher everywhere, so FKLD on teacher data spends its capacity hedging modes it can never represent.
- **How**:
    1. Sample outputs from the **student**.
    2. Teacher scores those exact tokens → target distributions.
    3. Minimize a divergence at every position.
    4. Mix student-generated & fixed data w/ $\lambda$; pick the divergence via [JSD](#jsd)($\beta$).
    - Generalizes ImitKD {cite:p}`lin2020autoregressive`, which introduced student-trajectory KD but fixed the divergence to FKLD.

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
- $y\sim p_S(\cdot|x)$: Sampled, then **detached** → no gradient through sampling.
- $\lambda=0$ → supervised KD; $\lambda=1$ → fully on-policy.
```

````{important} Code
:class: dropdown
```python
import math
import torch
import torch.nn.functional as F

def jsd_beta(t_logits, s_logits, beta=0.5):
    ## generalized JSD: beta -> 0 gives forward KL, beta -> 1 gives reverse KL (both after rescaling)
    logp_t, logp_s = F.log_softmax(t_logits, -1), F.log_softmax(s_logits, -1)
    ## mixture in log-space for numerical stability
    ## python floats, NOT torch.tensor -> stays on whatever device the logits live on
    log_b, log_1mb = math.log(max(beta, 1e-9)), math.log(max(1 - beta, 1e-9))
    log_mix = torch.logsumexp(torch.stack([logp_t + log_b, logp_s + log_1mb]), dim=0)
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
- FKLD ($\beta\to0$) → the student hedges; safe when capacity is close to the teacher's.
- RKLD ($\beta\to1$) → the student commits to modes it can actually represent; better under a large capacity gap.
- Reported best settings are task-dependent → treat $\beta$ & $\lambda$ as a 2D sweep, ❌constants.

*Cost vs sequence-level KD?*
- Student generation **every step** + a teacher forward pass on every generated token.
- Generation, not the loss, dominates → the multiplier scales w/ response length & the 2 model sizes, ❌a fixed constant.
- → Buy it only when exposure bias is the observed failure. [DistiLLM](#distillm) reports up to 4.3× training speedup just by attacking this overhead.

*Relation to [RFT](sft.md#rft)?*
- Both train on self-generated text.
- RFT: verifier filters, target is the sample itself (one-hot), rejected samples discarded.
- GKD: no filter, target is the teacher's full distribution, every sample used.
- → RFT needs a verifier; GKD needs a teacher.
```

&nbsp;

### MiniLLM
- **What**: Sequence-level [RKLD](#rkld) minimization via policy gradient. {cite:p}`gu2023minillm`
- **Why**: GKD applies its divergence per token; the object that actually misbehaves is the **sequence** distribution.
    - Token-level RKLD on student samples still never charges the student for a whole degenerate continuation.
    - Sequence-level RKLD does, but its expectation is under $p_S$ → the naive estimator is a high-variance REINFORCE.
- **How**:
    1. Sample from a **teacher-mixed** distribution, ❌the raw student.
    2. Reward each token by $\log\frac{p_T}{p_S}$; length-normalize the future reward.
    3. Split the gradient: a closed-form single-step term + a REINFORCE long-term term.
    4. Importance-weight to stay unbiased under the mixed sampler.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\alpha$: Teacher mix-in strength.
- Misc:
    - $r_t=\sum_{v\in\mathcal{V}}p_S(v|y_{<t},x)\log\frac{p_T(v|y_{<t},x)}{p_S(v|y_{<t},x)}$: Single-step term (closed form over $\mathcal{V}$).
    - $R_{t+1}$: Future log-ratio sum from $t+1$ onward.
    - $w_t$: Importance weight correcting the mixed sampler.

Objective:

$$
\mathcal{L}(\theta)=D_\text{KL}\left(p_S(\cdot|x)\ \Vert\ p_T(\cdot|x)\right)
$$

Mixed sampler:

$$
\tilde{p}(y_t|y_{<t},x)=\alpha\,p_T(y_t|y_{<t},x)+(1-\alpha)\,p_S(y_t|y_{<t},x)
$$

Importance weight — exact (unbiased) vs implemented:

$$
w_t=\prod_{t'=1}^{t}\frac{p_S(y_{t'}|y_{<t'},x)}{\tilde{p}(y_{t'}|y_{<t'},x)}\quad\longrightarrow\quad w_t\approx\frac{p_S(y_t|y_{<t},x)}{\tilde{p}(y_t|y_{<t},x)}
$$
- The product accumulates per-step variance → truncated to the single-step ratio → ⬇️variance, ⬆️bias.

Length-normalized future reward:

$$
R_{t+1}^\text{Norm}=\frac{1}{|y|-t}\sum_{t'=t+1}^{|y|}\log\frac{p_T(y_{t'}|y_{<t'},x)}{p_S(y_{t'}|y_{<t'},x)}
$$
- Divisor = #terms summed. The paper writes $|y|-t-1$, which is off by one against its own summation range.

Gradient:

$$
\nabla_\theta\mathcal{L}=-\mathbb{E}_{x\sim\mathcal{D},\,y\sim\tilde{p}(\cdot|x)}\left[\sum_{t=1}^{|y|}w_t\left(\underbrace{\nabla_\theta r_t}_{\text{single-step}}+\underbrace{R_{t+1}^\text{Norm}\,\nabla_\theta\log p_S(y_t|y_{<t},x)}_{\text{long-term}}\right)\right]
$$
- Single-step part is exact ← summed over $\mathcal{V}$, ❌sampled.
- Long-term part is REINFORCE ← carries the variance.
```

```{attention} Q&A
:class: dropdown
*Why is teacher-mixed sampling needed at all?*
- Pure-student sampling → **reward hacking**: degenerate outputs (repeated phrases) that the teacher happens to score highly.
- Small students hit this hardest ← their raw samples are worst.
- Mixing in $p_T$ suppresses those trajectories at the source; the importance weight is what keeps the correction honest — exactly unbiased only in its full-product form, which is not the one implemented.

*Why length-normalize?*
- $R_{t+1}$ is a **sum** of log-ratios, each typically negative under a trained student → longer continuations accumulate a more negative reward.
- → Bias toward **short** responses. Dividing by the remaining length removes the length dependence.

*Why decompose instead of plain REINFORCE?*
- $r_t$ is a full-vocab expectation → computable in closed form → that part contributes **zero** sampling variance.
- Front-token errors compound down the sequence, so the single-step term is exactly the high-leverage part → worth computing exactly.

*Vs GKD w/ $\beta\to1$?*
- GKD: token-level RKLD, detached samples, supervised gradient, ❌REINFORCE.
- MiniLLM: sequence-level RKLD, gradient flows through the sampling distribution → genuinely RL.
- → GKD is cheaper & stabler; MiniLLM optimizes the object you actually deploy.
```

&nbsp;

### DistiLLM
- **What**: [SKL](#skl) + a replay buffer over student samples. {cite:p}`ko2024distillm`
- **Why**: On-policy KD is right but wasteful, & its objective is still unbounded.
    - Generating fresh student samples every step dominates cost & is thrown away after 1 update.
    - Early student samples are noisy → training on them from step 0 destabilizes.
- **How**:
    1. Objective: SKL / SRKL w/ small $\alpha$ → bounded gradient.
    2. Use student-generated output w/ probability $\phi$, fixed data w/ $1-\phi$.
    3. Schedule $\phi$ **up** over training, driven by validation loss.
    4. Reuse past student samples from a replay buffer → off-policy, amortizing generation.

```{attention} Q&A
:class: dropdown
*Why is off-policy acceptable here when GKD argues for on-policy?*
- The target is the **teacher's** distribution at given positions → it does not change as $\theta$ moves.
- Stale samples are therefore still validly-labeled inputs → unlike RL, no advantage/value estimate goes stale.
- But the **state distribution** does drift → off-policy bias returns once buffered samples lag the current student far enough. Capacity is the knob: too small → overfitting, too large → outdated samples & high bias; they settle on 1000.

*Why start $\phi$ low & raise it?*
- Early student samples are near-garbage → noisy positions dominate the signal.
- Late in training the student's own distribution is the one worth correcting.
- → Curriculum, ❌a fixed mixture.

*What does it actually claim to win?*
- Up to ~4.3× speedup over recent on-policy KD baselines at comparable or better quality.
- Both halves matter: SKL alone stabilizes, the buffer alone saves compute; the paper's ablation shows dropping either loses most of the gain.
```

&nbsp;

### ULD
- **Name**: Universal Logit Distillation {cite:p}`boizard2024towards`
- **What**: Cross-tokenizer logit distillation via optimal transport.
- **Why**: Every divergence above is a **coordinate-wise** sum over $\mathcal{V}$, so it presupposes 1 shared vocab.
    - 2 tokenizers → index $v$ means different things on each side → the sum is meaningless.
    - → Logit-based KD was confined to within-family pairs; everything cross-family fell back to text.
- **How**:
    1. Sort each side's probability vector descending → discard the index identity, keep the **shape**.
    2. Pad the shorter vocab w/ zeros → equal length.
    3. Cost of moving mass between ranks = Wasserstein distance → minimize it.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $p^\downarrow$: Probability vector sorted in descending order.

Objective (per position):

$$
\mathcal{L}_\text{ULD}=\sum_{r=1}^{\max(|\mathcal{V}_T|,|\mathcal{V}_S|)}\left|p_T^\downarrow(r)-p_S^\downarrow(r)\right|
$$
- $r$: Rank, ❌token id.
- $\equiv$ the 1D Wasserstein-1 distance under a uniform ground cost → closed form, ❌an OT solver.
```

```{attention} Q&A
:class: dropdown
*What is actually being matched, if not tokens?*
- The **sorted profile**: how peaked vs flat the teacher is, ❌which token it prefers.
- → Transfers confidence & entropy structure; drops the identity of the "dark knowledge" that token-level KD's main selling point rests on.

*Why does discarding the token identity still help?*
- The sequence-level target already pins down *which* token ← the text is supervised by CE alongside.
- ULD adds the shape on top → strictly more than [Sequence-Level KD](#sequence-level-kd), strictly less than same-tokenizer token-level KD.

*When is it not worth it?*
- Shared tokenizer → ❌ULD. It is a strict downgrade from a real vocab-aligned divergence.
- The alternative family, explicit **token alignment** (merge/split spans, minimum edit distance over token strings), keeps identity but pays an alignment-error cost → prefer it when the 2 tokenizers are close.
```

&nbsp;

## Practice
- **What**: Picking a method from the 3 axes.

```{dropdown} Table: KD Methods
| Method | Sampled from | Target | Divergence | Needs |
|:--|:--|:--|:--|:--|
| Token-Level KD | Fixed corpus | Full distribution | FKLD | Logits, shared tokenizer |
| Sequence-Level KD | Teacher (offline) | Token one-hot | FKLD (sequence) | Text only |
| CoT Distillation | Teacher (offline) | Rationale + label | FKLD (sequence) | Text only |
| ULD | Teacher (offline) | Sorted distribution | $\mathcal{W}_1$ | Logits, any tokenizer |
| f-distill | Teacher + student | Full distribution | JSD / TVD | Logits + student sampling |
| GKD | Student (+ fixed) | Full distribution | JSD($\beta$) | Logits + student sampling |
| MiniLLM | Teacher-mixed | Full distribution | RKLD (sequence) | Logits + RL loop |
| DistiLLM | Student (replayed) | Full distribution | SKL / SRKL | Logits + student sampling |
```

```{attention} Q&A
:class: dropdown
*Decision order?*
1. ❌Logits → [Sequence-Level KD](#sequence-level-kd). No choice to make.
2. ✅Logits, ❌shared tokenizer → [ULD](#uld), or align the tokenizations if they are close.
3. ✅Both, small capacity gap, cheap → [Token-Level KD](#token-level-kd) w/ [FKLD](#fkld).
4. Large capacity gap → skew or symmetrize ([SKL](#skl), [JSD](#jsd), [TVD](#tvd)) before reaching for on-policy.
5. Student is fine on teacher text but degrades on its own rollouts → [on-policy](#on-policy-kd) (GKD → DistiLLM for cost).
6. Verifiable task w/ a checkable answer → [RFT](sft.md#rft) may beat all of them & needs no teacher.

*What is the single most common mistake?*
- Reaching for an exotic divergence when the actual bottleneck is **data coverage**.
- Divergence choice matters only where the teacher is evaluated; no divergence supervises a prompt the corpus never contains.

*How is any of this evaluated?*
- ❌The distillation loss — it is a divergence on the training distribution, not a capability measure.
- Task metrics + a diversity measure, since every mode-seeking objective buys quality by shedding coverage.
```