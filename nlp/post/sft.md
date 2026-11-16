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
# Supervised

Notations:
- $x$: Prompt (input token seq)
- $y$: Response (output token seq)
- $y_t$: $t$-th response token
- $y_{<t}$: Response tokens before position $t$
- $|y|$: Response length (#tokens)
- $y_w,y_l$: Preferred / dispreferred response to the same $x$
- $\mathcal{D}$: Dataset
- $\pi_\theta$: LM policy (next-token distribution)
- $\pi_\text{ref}$: Frozen reference policy (usually the SFT model)

&nbsp;

## SFT
- **What**: Train a next-token prediction model on curated $(x,y)$ pairs.
- **Why**: Shift probability distribution onto desired behavior.
- **How**:
    1. Render $(x,y)$ w/ the chat template, incl. the end-of-turn token.
    2. Mask prompt tokens → loss on response tokens only.
    3. Teacher forcing: condition each $y_t$ on the ground-truth prefix $y_{<t}$.
    4. Minimize NLL.

```{note} Math
:class: dropdown
Objective:

$$
\mathcal{L}_\text{SFT}(\theta)=-\mathbb{E}_{(x,y)\sim\mathcal{D}}\left[\log\pi_\theta(y|x)\right]=-\mathbb{E}_{(x,y)\sim\mathcal{D}}\left[\sum_{t=1}^{|y|}\log\pi_\theta(y_t|x,y_{<t})\right]
$$
```

```{tip} Derivation
:class: dropdown
*What is SFT actually minimizing?*

1. The dataset defines a conditional $p_\mathcal{D}(y|x)$; the model defines $\pi_\theta(y|x)$.
2. Expand the **forward KL** from data to model:

    $$
    \text{KL}(p_\mathcal{D}\|\pi_\theta)=\mathbb{E}_{y\sim p_\mathcal{D}}[\log p_\mathcal{D}(y|x)]-\mathbb{E}_{y\sim p_\mathcal{D}}[\log\pi_\theta(y|x)]=-H(p_\mathcal{D})+H(p_\mathcal{D},\pi_\theta)
    $$

3. $H(p_\mathcal{D})$ is $\theta$-free → minimize forward KL = minimize CE = MLE.
4. Forward KL is **mode-covering**: wherever $p_\mathcal{D}(y|x)>0$, driving $\pi_\theta(y|x)\to0$ costs $\to\infty$.
5. → The model must place mass on every demonstrated response, incl. mutually contradictory ones.
6. → SFT averages the demonstrations; it cannot prefer among them. Preference optimization & RL fill that gap.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SFTLoss(nn.Module):
    def __init__(self, ignore=-100):
        super().__init__()
        self.ignore = ignore  ## label for prompt & pad tokens -> excluded from the loss

    def seq_logp(self, logits, labels):
        ## logits at position t predict token t+1 -> shift by one
        logits, labels = logits[:, :-1], labels[:, 1:]
        keep = labels != self.ignore
        lp = F.log_softmax(logits.float(), -1)
        lp = lp.gather(-1, labels.masked_fill(~keep, 0).unsqueeze(-1)).squeeze(-1)
        return (lp * keep).sum(-1)  ## (B,) = log pi(y|x), response tokens only

    def forward(self, logits, labels):
        ## sum over tokens (= Math), mean over samples
        return -self.seq_logp(logits, labels).mean()

## Example
logits = torch.randn(2, 5, 7)            ## (B, L, |V|)
labels = torch.randint(0, 7, (2, 5))
labels[:, :2] = -100                     ## first 2 tokens = prompt -> masked
print(SFTLoss()(logits, labels).shape)   ## torch.Size([]): scalar loss
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- Dense signal ← every target token is supervised.
- Direct control over format, tone, refusal style, tool syntax, etc.

*Cons?*
- **Imitation ceiling**: Trains toward demonstrated quality → ❌signal to rank outputs above it.
- **Mode-covering**: Contradictory demonstrations get averaged into a blurry compromise.
- Overfitting on small data.

*Why mask the prompt tokens?*
- Target = $\pi_\theta(y|x)$, ❌$p(x)$ → prompt tokens are conditioning, ❌targets.
- Loss on prompts spends gradient on imitating user text.
- ⚠️ Default, ❌law: loss on prompts can help when responses are short & data is small.

*Why does the chat template matter?*
- Role markers & end-of-turn tokens are part of the learned format.
- Train/inference template mismatch → off-distribution input → degraded output.
- ❌Loss on the end-of-turn token → model never learns to stop.

*Sum vs mean over tokens?*
- Sum per sequence (Math) → long responses get proportionally larger gradients.
- Mean over all response tokens in the batch → every token weighted equally → long responses still dominate by count.
- Mean per sequence, then over the batch → every sample weighted equally → tokens of short responses upweighted.
- Each choice = an implicit sample weighting, ❌a neutral default.
- Gradient accumulation w/ token-mean: average per micro-batch ≠ average over all tokens, unless you normalize by the global token count.

*Why pack sequences, & what breaks?*
- Concatenate several samples into one context → ⬇️padding waste → ⬆️throughput.
- Naive packing → tokens attend across sample boundaries → cross-sample contamination.
- Fix: block-diagonal attention mask & position reset per sample.

*Why does SFT on facts the model doesn't know induce hallucination?*
- The target is a confident assertion w/o internal support in the model.
- → The generalizable lesson = assert unsupported facts confidently.
- Empirically: unknown-knowledge examples are fit much slower; once fit, hallucination rises on other questions. {cite:p}`gekhman2024does`
- → Prefer demonstrations of what the model already knows; teach abstention explicitly.

*Why does SFT on the model's own correct outputs beat SFT on stronger off-policy data?*
- An off-policy target can be extremely improbable under the policy → large gradients that move weights far → forgetting & miscalibration.
- On-policy targets sit near the model's own distribution → small, targeted updates.
- → The basis of RFT & [on-policy distillation](kd.md#on-policy-kd).

*What is exposure bias?*
- Training conditions on the ground-truth prefix (teacher forcing); inference conditions on the model's own prefix.
- → The model never trains on the states it reaches after its own mistakes → errors compound over long generations.
- Fix direction: put the model's own samples into training (RFT, on-policy distillation, RL).

*Why not just train more epochs?*
- Post-training sets are $10^3$–$10^6$ samples against $10^9$+ params → memorization is immediate.
- Symptom: train loss ⬇️, held-out win rate flat, generations become templated.
- → Few epochs (1–3); select on held-out metrics, ❌train loss.
```

&nbsp;

### Instruction Tuning
- **What**: SFT on many tasks phrased as natural-language instructions. {cite:p}`wei2021finetuned`
- **Why**: Zero-shot ability does not fall out of single-task fine-tuning.
    - Pretrained LMs are strong few-shot but weak zero-shot ← a bare instruction does not look like pretraining text.
    - Fine-tuning on one task buys that task & nothing else.
- **How**:
    1. Templatize many existing datasets into instruction form, w/ several phrasings each.
    2. Mix them, capped per dataset so no giant set dominates.
    3. SFT.
    4. Evaluate on held-out task clusters, ❌held-out examples.

```{attention} Q&A
:class: dropdown
*Why does it generalize to unseen tasks?*
- The learned skill is "read the instruction, then comply", ❌any individual task.
- Diversity is the driver: held-out performance rises w/ #task clusters & had not saturated in the original ablation.
- Removing the instructions (same data, no template) destroys the gain → the natural-language framing is load-bearing.

*Why did it make small models worse?*
- FLAN swept 422M / 2B / 8B / 68B / 137B: ⬆️held-out at 68B & 137B, ⬇️held-out at 8B & below.
- Proposed cause: small capacity is fully consumed learning the ~40 tuning tasks → nothing left for the meta-skill.
- ⚠️ Today's small models instruction-tune well ← better data & stronger pretraining. FLAN's result holds for its own setup.

*Instruction tuning vs chat SFT?*
- Same objective, different data: many short NLP tasks & single turn vs open-ended multi-turn dialogue.
- Modern mixes contain both, plus code, math, safety, & tool-use traces.

*Why cap each dataset?*
- Source sizes span orders of magnitude → uncapped, one dataset becomes the whole gradient.
- Capping trades raw tokens for task diversity, which is what actually transfers.
```

&nbsp;

### CPT
- **Name**: Continued Pretraining {cite:p}`gururangan2020dont`
- **What**: More next-token training on a raw target-domain corpus.
- **Why**: Some gaps are in the base distribution, ❌in the behavior.
    - Domain jargon, a new language, or a new text modality (code, legal, clinical) may be rare or absent in the original mix.
    - SFT sets are far too small to move what the model knows.
- **How**:
    1. Collect raw domain text, orders of magnitude larger than any SFT set.
    2. **Replay** a slice of the original pretraining mix in every batch.
    3. Re-warm up the LR, then decay again.
    4. Optionally extend tokenizer/context, then SFT as usual on top.

```{attention} Q&A
:class: dropdown
*DAPT vs TAPT?*
- **DAPT** (domain-adaptive) → a large corpus generic to the domain.
- **TAPT** (task-adaptive) → the task's own unlabeled text; tiny but exactly on-distribution.
- Both help & they compose: DAPT → TAPT → SFT.

*Why replay original data?*
- Pure domain text → the model drifts onto it & loses general ability → [catastrophic forgetting](../../dl/issues.md#catastrophic-forgetting).
- Replay keeps the old distribution in the gradient at a fraction of its original cost.

*Why re-warm up the LR?*
- Resuming at the final (tiny) pretraining LR barely moves the weights → CPT does almost nothing.
- Jumping straight to the peak LR spikes the loss & destroys prior capability.
- → Warm up to an intermediate peak, then decay.

*When is CPT the wrong tool?*
- Facts that change → retrieval, ❌weights.
- Format/behavior → SFT; it is far cheaper & more precise.
- → CPT is for a distribution shift: vocabulary, style, language, idioms.

*Why extend the tokenizer?*
- A domain the tokenizer never saw is shredded into many sub-word pieces → ⬆️sequence length, ⬇️effective context.
- ⚠️ New embedding rows start random → they need warm-up & enough data, or they stay noise.
```

&nbsp;

### RFT
- **Name**: Rejection sampling Fine-Tuning {cite:p}`yuan2023scaling`
- **What**: SFT on the model's own samples that pass a filter.
- **Why**: Human demonstrations are the bottleneck, & the model already produces good ones some of the time.
    - Checking a response is far cheaper than writing one.
    - Self-generated targets are already on-distribution → easier to fit than expert text.
- **How**:
    1. Sample $K$ responses per prompt from the curr policy at $T>0$.
    2. Keep the ones a verifier / RM accepts.
    3. Dedup by reasoning path, ❌only by final answer.
    4. Fine-tune the base model on the survivors; optionally repeat.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $K$: #samples per prompt.
- Misc:
    - $\pi_{\theta_\text{old}}$: Policy that generated the samples.
    - $v(x,y)\in\{0,1\}$: Verifier / filter outcome.
    - $\mathcal{D}^+$: Accepted samples.

Process:

1. $y_i\sim\pi_{\theta_\text{old}}(\cdot|x)$ for $i=1,\dots,K$.
2. $\mathcal{D}^+=\{(x,y_i)\mid v(x,y_i)=1\}$.
3. Minimize $\mathcal{L}_\text{SFT}$ on $\mathcal{D}^+$.

Objective (reward-weighted MLE form):

$$
\mathcal{L}_\text{RFT}(\theta)=-\mathbb{E}_{x\sim\mathcal{D},\ y\sim\pi_{\theta_\text{old}}(\cdot|x)}\left[v(x,y)\log\pi_\theta(y|x)\right]
$$
```

```{tip} Derivation
:class: dropdown
*Is RFT a policy-gradient step?*

1. At the first update after sampling, $\pi_{\theta_\text{old}}=\pi_\theta$.
2. Assume samples are drawn at $T=1$ & filtered by $v$ only. Then:

    $$
    \nabla_\theta\mathcal{L}_\text{RFT}=-\mathbb{E}_{y\sim\pi_\theta}\left[v(x,y)\nabla_\theta\log\pi_\theta(y|x)\right]
    $$

3. → This is the [REINFORCE](rl.md#reinforce) gradient w/ $r=v$ & baseline $b=0$.
4. The published recipe breaks both assumptions ($T=0.7$, then dedup) → idealization, ❌identity.
```

````{important} Code
:class: dropdown
```python
import random

class RFT:
    def __init__(self, generate, verify, key=lambda y: y, K=8):
        ## key: reasoning-path identity; Yuan et al. use the list of equations in y
        self.generate, self.verify, self.key, self.K = generate, verify, key, K

    def __call__(self, prompts):
        data = []
        for x in prompts:
            seen = set()
            for _ in range(self.K):
                y = self.generate(x)                              ## sample at T>0 -> diverse paths
                if self.verify(x, y) and self.key(y) not in seen: ## filter, then dedup by path
                    seen.add(self.key(y))
                    data.append((x, y))
        return data   ## -> plain SFT on D+ (SFTLoss above)

## Example: toy "policy" that sometimes reasons correctly
gen = lambda x: random.choice(["2+3=5", "3+2=5", "2+3=6"])
ver = lambda x, y: y.split("=")[1] == "5"
print(RFT(gen, ver, K=8)(["2+3"]))  ## <= 2 pairs: distinct correct paths only
```
````

```{attention} Q&A
:class: dropdown
*So is this RL?*
- The gradient matches REINFORCE only in the idealized case (Derivation).
- In practice $T=0.7$, duplicate reasoning paths are removed, & the fine-tune restarts from the base model. No policy-gradient method does any of these.
- Everything RL adds is absent: ❌baseline/advantage, ❌importance ratio, ❌KL anchor, ❌fresh rollouts per update.
- → Read it as one policy-improvement step (EM-style), ❌an optimization loop.

*Why does it saturate?*
- It can only train on what the policy already produces → the ceiling is pass@$K$ at the sampling budget.
- Prompts the model never solves contribute nothing, ever, no matter how many rounds you run.

*Why dedup?*
- Easy prompts yield many accepted samples → they dominate the filtered set → the mix skews easy.
- Distinct reasoning paths carry information; duplicate final answers do not.

*RFT vs RL, in practice?*
- ✅Stability & infra ← the ordinary SFT trainer, ❌critic, ❌ratio, ❌weight-sync loop.
- ❌Sample efficiency ← every rejected sample is discarded; RL extracts gradient from failures too.
- ✅Safe default when the verifier is reliable but the RL stack is not available.

*What can serve as the filter?*
- Verifier (exact match, unit tests) → precise, verifiable domains only.
- RM / LM judge → broad coverage, hackable → see [reward hacking](../rh.md).
- Human → best & least scalable.

*Why sample at $T>0$?*
- Greedy decoding returns one response per prompt → ❌diversity → the accepted set is tiny.
- Higher $T$ ⬆️coverage of solvable prompts but ⬆️false accepts (right answer, broken reasoning).
```

&nbsp;

#### STaR
- **Name**: Self-Taught Reasoner {cite:p}`zelikman2022star`
- **What**: RFT on rationales, w/ a hinted retry for the failures.
- **Why**: Rationale data does not exist at scale.
    - Final answers are cheap to label; step-by-step reasoning is not.
    - Plain filtering stalls: unsolved problems never produce a rationale, so the training set is permanently the easy subset.
- **How**:
    1. Few-shot prompt the model for rationale + answer.
    2. Keep rationales whose answer is correct.
    3. **Rationalize** the failures: re-prompt w/ the correct answer as a hint, keep the backward rationale it produces.
    4. Fine-tune from the original model on everything kept; repeat.

```{attention} Q&A
:class: dropdown
*Why rationalize instead of dropping failures?*
- W/o it the curriculum never advances past what the model can already solve.
- Rationalization manufactures training signal for exactly the problems it cannot yet solve.

*Why is rationalization risky?*
- The rationale is written w/ the answer known → it can be a post-hoc justification that never derives the answer.
- → Teaches plausible-sounding reasoning that carries no computational weight.

*Why fine-tune from the original model every round?*
- Fine-tuning on top of fine-tunes compounds overfitting to the earlier, easier rounds.
- Restarting keeps the model matched to the current, harder dataset.

*How does this relate to intelligence?*
- The model supplies both the hypotheses & the filter for its own training data; the human supplies only the answer key.
- The limit is sharp: it can bootstrap only what it can already occasionally produce.
- → Self-improvement here is amplification of an existing prior, ❌open-ended discovery.
```

&nbsp;

## PEFT
- **Name**: Parameter-Efficient Fine-Tuning
- **What**: Train a small param subset; freeze the pretrained weights.
- **Why**: Full fine-tuning does not scale to many tasks or small budgets.
    - Optimizer state dominates: mixed-precision Adam keeps 2 fp32 moments + an fp32 master copy → 12 bytes/param, vs 2 for bf16 weights.
    - 1 full checkpoint per task → storage = #tasks × model size.
    - Small dataset + all params trainable → fast overfitting & [catastrophic forgetting](../../dl/issues.md#catastrophic-forgetting).
- **How**:
    1. Freeze $W_0$ everywhere.
    2. Inject a small trainable module, or select a small subset of existing params.
    3. Train only those.
    4. Ship deltas (MB) instead of models (GB).

```{attention} Q&A
:class: dropdown
*Why does it work at all?*
- Hypothesis: adaptation to a downstream task has a low **intrinsic dimension** → the required update lives in a tiny subspace of weight space.
- Support is empirical (matching full FT at 0.01–1% trainable params), ❌proved.
- The pretrained model already contains the capability; the update only has to select it.

*Where does the memory actually go?*
- Weights: frozen → can even be quantized.
- Gradients + optimizer state: $\propto$ trainable params → this is the entire win.
- Activations: essentially unchanged ← you still backprop through the whole frozen network.
- → PEFT does not remove the need for gradient checkpointing on long sequences.

*When is full FT still the right call?*
- Large distribution shift (new language, new modality) → CPT-scale change, ❌a low-rank nudge.
- Abundant data & budget, single deployment target.
- → PEFT's gap to full FT widens as the required change grows.

*Can you stack PEFT on top of preference optimization?*
- ✅ Orthogonal: the adapter is a parameterization, the objective is a loss. LoRA + DPO is routine.
- ⚠️ $\pi_\text{ref}$ becomes free: disable the adapters → the same model is the reference.
```

&nbsp;

### Adapter
- **What**: Small bottleneck MLPs inserted between transformer sublayers. {cite:p}`houlsby2019parameter`
- **Why**: 1 fine-tuned copy per task is unaffordable to store & serve.
- **How**:
    1. After a sublayer, insert: down-project → nonlinearity → up-project.
    2. Wrap it in a residual connection.
    3. Initialize near-identity so the module starts as a no-op.
    4. Train adapters + layer norms only.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{h}\in\mathbb{R}^{d}$: Sublayer output.
- Params:
    - $D\in\mathbb{R}^{b\times d}$: Down-projection.
    - $U\in\mathbb{R}^{d\times b}$: Up-projection.
- Hyperparams:
    - $b\ll d$: Bottleneck width.
- Misc:
    - $\phi$: Nonlinearity.

Forward:

$$
\mathbf{h}\leftarrow\mathbf{h}+U\,\phi\left(D\mathbf{h}\right)
$$
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class Adapter(nn.Module):
    def __init__(self, d, b):
        super().__init__()
        self.D = nn.Linear(d, b, bias=False)   ## down-project to bottleneck b << d
        self.U = nn.Linear(b, d, bias=False)   ## up-project back to d
        nn.init.zeros_(self.U.weight)          ## U=0 -> exact identity at step 0; D random keeps U's grad nonzero

    def forward(self, h):
        ## residual around a nonlinear bottleneck -> sequential, cannot fold into W0
        return h + self.U(torch.relu(self.D(h)))

## Example
h = torch.randn(2, 8)
ad = Adapter(d=8, b=2)
print(torch.allclose(ad(h), h))   ## True: no-op before training
```
````

```{attention} Q&A
:class: dropdown
*Why near-identity init?*
- A randomly initialized inserted module corrupts the pretrained function at step 0 → the model has to recover before it can learn.
- Near-zero $U$ makes the branch a no-op initially.

*Why did adapters lose to LoRA?*
- They are sequential: the branch must run before the next layer → its cost cannot be folded into $W_0$.
- → Permanent extra depth & inference latency at every layer, every token.
- Worse under model parallelism ← an extra synchronization point per layer.

*What survives of the idea?*
- The bottleneck-plus-residual pattern is everywhere (LoRA is its linear, parallel, mergeable cousin).
- Adapters remain attractive when you want a nonlinear task-specific transform.
```

&nbsp;

### LoRA
- **Name**: Low-Rank Adaptation {cite:p}`hu2021lora`
- **What**: Trainable rank-$r$ update added in parallel to a frozen weight matrix.
- **Why**: Adapters buy parameter efficiency by paying inference latency.
    - A sequential module cannot be folded into $W_0$ → the cost is permanent.
    - A parallel linear branch can be added into the weight matrix after training → free at inference.
- **How**:
    1. Factor the update: $\Delta W=BA$ w/ inner dimension $r\ll\min(d_\text{in},d_\text{out})$.
    2. Init $A$ random, $B=0$ → $\Delta W=0$ at step 0, so the model starts unchanged.
    3. Train $A,B$ only, scaling the branch by $\frac{\alpha}{r}$.
    4. At deploy, merge $W\leftarrow W_0+\frac{\alpha}{r}BA$.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{x}\in\mathbb{R}^{d_\text{in}}$: Input vector.
    - $\mathbf{h}\in\mathbb{R}^{d_\text{out}}$: Output vector.
- Params:
    - $A\in\mathbb{R}^{r\times d_\text{in}}$: Down-projection, $A\sim\mathcal{N}(0,\sigma^2)$ at init.
    - $B\in\mathbb{R}^{d_\text{out}\times r}$: Up-projection, $B=0$ at init.
    - $W_0\in\mathbb{R}^{d_\text{out}\times d_\text{in}}$: Frozen pretrained weight.
- Hyperparams:
    - $r$: Rank.
    - $\alpha$: Scaling numerator.

Forward:

$$
\mathbf{h}=W_0\mathbf{x}+\frac{\alpha}{r}BA\mathbf{x}
$$

Backward (w/ $\mathbf{g}=\frac{\partial\mathcal{L}}{\partial\mathbf{h}}$):

$$
\frac{\partial\mathcal{L}}{\partial B}=\frac{\alpha}{r}\,\mathbf{g}\left(A\mathbf{x}\right)^T,\qquad \frac{\partial\mathcal{L}}{\partial A}=\frac{\alpha}{r}\,B^T\mathbf{g}\,\mathbf{x}^T
$$
- $W_0$: Frozen → ❌grad, ❌optimizer state.
- #Trainable params: $r(d_\text{in}+d_\text{out})$ vs $d_\text{in}d_\text{out}$.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False               ## W0 is frozen, forever
        d_out, d_in = base.weight.shape
        self.A = nn.Parameter(torch.randn(r, d_in) * 0.01)   ## random -> nonzero grad path
        self.B = nn.Parameter(torch.zeros(d_out, r))         ## zero -> delta W = 0 at step 0
        self.scale = alpha / r                    ## decouples update size from r

    def forward(self, x):
        ## parallel branch: the base output is never modified in place -> mergeable later
        return self.base(x) + self.scale * (x @ self.A.T) @ self.B.T

    @torch.no_grad()
    def merge(self):
        ## fold into W0 -> zero added latency at inference
        self.base.weight += self.scale * (self.B @ self.A)
        self.B.zero_()   ## the branch is now INSIDE W0; leaving B would double-count it
        return self.base

## Example
layer = LoRALinear(nn.Linear(4, 6), r=2, alpha=4)
x = torch.randn(2, 4)
print(torch.allclose(layer(x), layer.base(x)))    ## True: identical before training
print(sum(p.numel() for p in layer.parameters() if p.requires_grad))  ## 20 vs 24 base weights
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- On GPT-3 175B vs Adam full FT: 10,000× fewer trainable params & 3× less GPU memory.
- ❌Added inference latency after merging, unlike adapters.
- LoRA weights are MBs → many task-specific ones can be swapped, or served concurrently against 1 base.

*Cons?*
- Rank caps how much of the update is expressible → lags full FT when the needed change is large.
- Sensitive to $r$, $\alpha$, target modules, & LR simultaneously.
- ⚠️ Saves optimizer memory, ❌activation memory.

*Why $B=0$ & $A$ random, instead of both zero or both random?*
- Both random → $\Delta W\neq0$ at init → the pretrained function is corrupted before training starts.
- Both zero → $\frac{\partial\mathcal{L}}{\partial A}\propto B^T=0$ & $\frac{\partial\mathcal{L}}{\partial B}\propto A^T=0$ → dead branch, forever.
- $B=0$, $A\neq0$ → output unchanged, yet $\frac{\partial\mathcal{L}}{\partial B}\neq0$ → $B$ moves first, then $A$ receives gradient.

*What is $\alpha$ for?*
- $\frac{\alpha}{r}$ makes the branch's effective magnitude roughly independent of $r$.
- → Change $r$ w/o re-tuning the LR. Common settings: $\alpha=r$ or $\alpha=2r$.

*Which modules should it target?*
- The original work adapted attention projections only (best results from $W_q,W_v$ at a fixed budget).
- Current practice applies it to all linear layers incl. the MLP, which generally helps at equal total params.

*What LR?*
- ~10× the full-FT LR (order $10^{-4}$) ← the branch starts at 0 & has few params to move.
- Reuse the full-FT LR → LoRA barely moves.

*Does the rank need to be large?*
- Style, format, tone, persona → $r=8$–$16$ is usually plenty.
- New knowledge or a real distribution shift → raise $r$, or accept that full FT / CPT is the right tool.

*Why can it be merged when adapters cannot?*
- The branch is linear & parallel → $W_0\mathbf{x}+\Delta W\mathbf{x}=(W_0+\Delta W)\mathbf{x}$.
- Adapters are sequential & nonlinear → no such algebraic collapse exists.
```

&nbsp;

#### QLoRA
- **What**: LoRA over a 4-bit quantized frozen base. {cite:p}`dettmers2023qlora`
- **Why**: LoRA removes optimizer state, ❌the weights themselves.
    - A 65B model at 16-bit is ~130GB of frozen weights before a single activation is stored.
    - → The base weights, not the trainable ones, become the binding constraint.
- **How**: 3 mechanisms stacked on ordinary LoRA.
    1. **NF4**: A 4-bit NormalFloat data type, information-theoretically optimal for normally distributed weights.
    2. **Double quantization**: Quantize the quantization constants as well.
    3. **Paged optimizers**: Unified memory paging to absorb gradient-checkpointing memory spikes.

```{attention} Q&A
:class: dropdown
*Pros?*
- 65B fine-tuned on a single 48GB GPU, matching 16-bit fine-tuning quality.

*Why doesn't 4-bit destroy quality?*
- The base is frozen & used only in the forward pass → quantization error is a fixed perturbation, ❌accumulating noise.
- Weights are dequantized to 16-bit per block for the actual matmul.
- The 16-bit adapters are trained through the quantized base → they absorb its error.

*What does it cost?*
- Dequantization on every forward → slower per step than plain LoRA.
- → Trade throughput for the ability to fit the model at all.

*Why NF4 rather than int4?*
- Pretrained weights are approximately zero-centered normal.
- NF4 places its 16 levels at the quantiles of a normal distribution → equal expected mass per level.
- Int4's uniform levels waste resolution in the tails where almost no weights live.

*Can you merge the adapter?*
- ❌ Cleanly into the 4-bit base: merging then re-quantizing discards the adapter's precision.
- → Serve the adapter separately, or merge into the original 16-bit weights.
```

&nbsp;

#### DoRA
- **Name**: Weight-Decomposed Low-Rank Adaptation {cite:p}`liu2024dora`
- **What**: Split each weight into magnitude & direction; apply LoRA to the direction only.
- **Why**: LoRA's update pattern is structurally unlike full FT's.
    - Decompose both into magnitude & direction changes: correlation is $+0.83$ for LoRA but $-0.62$ for full FT.
    - → LoRA moves magnitude & direction nearly proportionally; full FT trades one against the other.
    - → LoRA cannot express "large directional change, small magnitude change", which full FT does routinely.
- **How**:
    1. Decompose $W_0$ into a magnitude vector & a unit-norm direction matrix.
    2. Train the magnitude vector directly; adapt the direction w/ LoRA.
    3. Re-normalize the adapted direction each step, then rescale by the magnitude.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\mathbf{m}\in\mathbb{R}^{1\times d_\text{in}}$: Trainable magnitude vector, init $\|W_0\|_c$.
    - $A,B$: LoRA factors, as in LoRA.
    - $W_0\in\mathbb{R}^{d_\text{out}\times d_\text{in}}$: Frozen pretrained weight.
- Misc:
    - $\|\cdot\|_c$: Column-wise $L_2$ norm → 1 scalar per column.

Forward:

$$
\mathbf{h}=W'\mathbf{x},\qquad W'=\mathbf{m}\frac{W_0+BA}{\|W_0+BA\|_c}
$$
- $\frac{W_0+BA}{\|W_0+BA\|_c}$: Direction → unit-norm columns.
- $\mathbf{m}$: Rescales column $j$ by $m_j$.
- ⚠️ HF PEFT normalizes each output row of the $(d_\text{out},d_\text{in})$ weight instead → $\mathbf{m}\in\mathbb{R}^{d_\text{out}}$.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8):
        super().__init__()
        self.register_buffer("W0", base.weight.detach().clone())   ## (d_out, d_in), frozen
        self.register_buffer("b0", None if base.bias is None else base.bias.detach().clone())
        d_out, d_in = self.W0.shape
        self.m = nn.Parameter(self.W0.norm(dim=0, keepdim=True))  ## (1, d_in): column norms (paper)
        self.A = nn.Parameter(torch.randn(r, d_in) * 0.01)
        self.B = nn.Parameter(torch.zeros(d_out, r))               ## B=0 -> W' = W0 at step 0

    def forward(self, x):
        V = self.W0 + self.B @ self.A                ## LoRA acts on the direction carrier
        W = self.m * V / V.norm(dim=0, keepdim=True) ## unit columns, rescaled by magnitude
        return F.linear(x, W, self.b0)

## Example
base = nn.Linear(4, 6)
x = torch.randn(2, 4)
print(torch.allclose(DoRALinear(base, r=2)(x), base(x), atol=1e-6))  ## True: W' = W0 at init
```
````

```{attention} Q&A
:class: dropdown
*Why does decoupling help?*
- Under the decomposition, DoRA's magnitude-direction correlation is $-0.31$ vs full FT's $-0.62$ & LoRA's $+0.83$.
- → It recovers full FT's qualitative learning pattern at LoRA's parameter budget.

*What does it cost?*
- The column-norm & renormalization every step → extra training compute & memory over LoRA.
- Paper's fix: treat $\|W_0+BA\|_c$ as a constant in backward → ⬇️gradient-graph memory.
- ✅Still mergeable → ❌inference overhead.

*When is the gain largest?*
- Low rank. DoRA degrades much more gracefully than LoRA as $r$ shrinks.
- → Useful where LoRA's expressivity limit binds.
```

&nbsp;

### Prefix Tuning
- **What**: Trainable "virtual token" vectors prepended to the keys & values at every layer. {cite:p}`li2021prefix`
- **Why**: Prompting is limited to strings the tokenizer can produce.
    - Discrete prompt search ranges over a finite vocabulary; the optimal conditioning vector need not be any word.
    - Full FT stores a whole model per task.
- **How**:
    1. Prepend $p$ trainable key/value vectors per layer.
    2. Every position attends over [prefix; sequence] → the prefix conditions the whole forward pass.
    3. Reparameterize the prefix through an MLP during training, discard the MLP afterward.
    4. Train ~0.1% of params.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $P_K,P_V\in\mathbb{R}^{p\times d}$: Trainable key / value prefixes, 1 pair per layer.
- Hyperparams:
    - $p$: Prefix length.
- Misc:
    - $Q,K,V$: Frozen query / key / value projections of the layer input.
    - $[\cdot;\cdot]$: Concatenation along the sequence axis.

Forward (per layer, single head):

$$
\text{Attn}=\text{softmax}\left(\frac{Q\,[P_K;K]^T}{\sqrt{d}}+M\right)[P_V;V]
$$
- $M\in\mathbb{R}^{L\times(p+L)}$: Causal mask. $M_{t,s}=0$ if $s\le p$ (prefix) or $s-p\le t$; else $-\infty$.
- $L$: Sequence length.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PrefixAttention(nn.Module):
    def __init__(self, d, p):
        super().__init__()
        self.q, self.k, self.v = (nn.Linear(d, d) for _ in range(3))
        self.requires_grad_(False)                       ## freeze the pretrained projections
        self.Pk = nn.Parameter(torch.randn(p, d) * 0.02) ## the ONLY trainable params
        self.Pv = nn.Parameter(torch.randn(p, d) * 0.02)
        ## MLP reparameterization omitted: it only exists during training

    def forward(self, x):                                ## x: (B, L, d)
        B, L, _ = x.shape
        p = self.Pk.size(0)
        k = torch.cat([self.Pk.expand(B, -1, -1), self.k(x)], 1)   ## (B, p+L, d)
        v = torch.cat([self.Pv.expand(B, -1, -1), self.v(x)], 1)
        ## prefix always visible; sequence keys stay causal
        mask = torch.cat([torch.ones(L, p, dtype=torch.bool),
                          torch.ones(L, L, dtype=torch.bool).tril()], 1)  ## (L, p+L)
        return F.scaled_dot_product_attention(self.q(x), k, v, attn_mask=mask)  ## (B, L, d)

## Example
attn = PrefixAttention(d=4, p=2)
print(attn(torch.randn(2, 3, 4)).shape)                          ## torch.Size([2, 3, 4])
print([n for n, t in attn.named_parameters() if t.requires_grad]) ## ['Pk', 'Pv']
```
````

```{attention} Q&A
:class: dropdown
*Why the MLP reparameterization?*
- Optimizing the prefix vectors directly is unstable & highly LR-sensitive.
- Generating them from a smaller matrix through an MLP smooths the optimization; only the generated vectors are kept.

*Why does it beat full FT in low-data settings?*
- Far fewer trainable params → a much stronger implicit prior toward the pretrained function.
- Reported to extrapolate better to topics unseen during training.

*Cons?*
- The prefix permanently occupies context & KV cache → ⬇️usable context window.
- ❌Mergeable → the overhead is per-token, forever.
- More sensitive than LoRA & harder to tune.
```

&nbsp;

#### Prompt Tuning
- **What**: Trainable soft prompt at the input embedding layer only. {cite:p}`lester2021power`
- **Why**: Prefix tuning injects at every layer → more params & more plumbing than the effect requires.
- **How**: Prepend $p$ trainable embedding vectors to the input embeddings; freeze everything else, incl. all attention layers.

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class PromptTuning(nn.Module):
    def __init__(self, embed: nn.Embedding, p):
        super().__init__()
        self.embed = embed.requires_grad_(False)
        ## init from real token embeddings -> converges better than random
        idx = torch.randint(0, embed.num_embeddings, (p,))
        self.P = nn.Parameter(embed.weight[idx].detach().clone())   ## (p, d)

    def forward(self, ids):                                ## ids: (B, L)
        e = self.embed(ids)                                ## (B, L, d)
        ## soft prompt goes in front; the frozen LM consumes (B, p+L, d)
        return torch.cat([self.P.expand(ids.size(0), -1, -1), e], 1)

## Example
pt = PromptTuning(nn.Embedding(10, 4), p=3)
print(pt(torch.randint(0, 10, (2, 5))).shape)   ## torch.Size([2, 8, 4])
```
````

```{attention} Q&A
:class: dropdown
*When does it actually work?*
- Gap to full FT shrinks w/ scale; matched model tuning at T5-XXL (11B), larger gap below.
- → "The power of scale": the bigger the frozen model, the less you need to touch it.

*Pros?*
- The smallest footprint of any PEFT method: a handful of vectors per task.
- Enables **prompt ensembling**: many prompts, 1 frozen model, batched together.
- ⬆️Robustness under domain transfer vs full FT.

*Cons?*
- Slow convergence & init-sensitive (initializing from real token embeddings helps).
- Eats context, cannot be merged.
- Weakest expressivity of the family ← it can only re-condition the input, never change the computation.
```

&nbsp;

````{dropdown} Table: PEFT at a Glance
| Method | Trains | Mergeable | Inference overhead | Note |
|:--|:--|:--|:--|:--|
| Full FT | Everything | — | ❌ | The reference point; max capacity, max cost |
| Adapter | Inserted bottleneck MLPs | ❌ | ✅ Latency at every layer | Nonlinear, sequential |
| LoRA | $A,B$ in parallel w/ $W_0$ | ✅ | ❌ | The default |
| QLoRA | LoRA over a 4-bit base | ⚠️ Not into the 4-bit base | ❌ (⬆️ if unmerged) | Fits the model, ⬇️throughput |
| DoRA | Magnitude vector + LoRA direction | ✅ | ❌ | ⬆️Low-rank quality, ⬆️train cost |
| Prefix Tuning | Per-layer KV prefixes | ❌ | ✅ Context + KV cache | ~0.1% of params |
| Prompt Tuning | Input-embedding prompt | ❌ | ✅ Context | Competitive w/ a large frozen model |
| BitFit | Bias terms only {cite:p}`benzaken2021bitfit` | — (already in $W_0$) | ❌ | Smallest possible change to existing params |
| IA³ | Learned rescaling vectors for K, V, FFN {cite:p}`liu2022fewshot` | ✅ | ❌ | Even fewer params than LoRA |
````

&nbsp;

## Preference Optimization
- **What**: Fit a policy to preference labels w/ an offline supervised loss.
- **Why**: SFT cannot express "this is better than that".
    - CE raises the demonstrated response & drains the rest only through normalization → ❌way to name a specific response as the worse one.
    - Quality is unratable in absolutes but reliably rankable in pairs → the cheap label is a comparison.
    - The [RLHF](rl.md#rlhf) route buys the same signal at the cost of a second model plus a rollout loop.
- **How**:
    1. Collect $(x,y_w,y_l)$ triples, or unpaired good/bad labels (KTO).
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
- The loss pushes $\log\pi_\theta(y_l|x)$ down; if $y_l$ already has negligible probability under $\pi_\theta$, nothing is left to suppress.
- Then the gradient mostly reshapes mass elsewhere, off the data entirely.
- → Sample $y_w,y_l$ from the model being trained (or its close relative), then label.

*Why is offline preference optimization structurally weaker than online RL?*
- The dataset is fixed → the objective constrains the policy only on responses in $\mathcal{D}$.
- Off that support it says nothing → the optimum is free to put mass on responses no one ever ranked.
- Online RL re-samples, so whatever the policy drifts toward gets graded next step.

*Why is a KL anchor to $\pi_\text{ref}$ present in almost all of them?*
- The preference signal is a ranking → it pins down the policy only up to everything it never ranked.
- $\pi_\text{ref}$ supplies the missing prior: stay where you were unless the data says otherwise.
- ORPO replaces it w/ an SFT term; SimPO drops it (❌anchor).
```

&nbsp;

### DPO
- **Name**: Direct Preference Optimization {cite:p}`rafailov2023direct`
- **What**: Preference classification w/ the LM as its own reward model.
- **Why**: The RLHF stack is expensive & fragile.
    - Separate RM → an extra model to train, store, and overfit.
    - PPO loop → rollouts, critic, clip & KL tuning → the dominant cost & the dominant source of instability.
- **How**:
    1. The KL-constrained RL objective has a closed-form optimum.
    2. Invert it → reward = scaled log-ratio of policy to reference.
    3. Substitute into Bradley-Terry → the intractable partition function cancels.
    4. Maximize the likelihood of the observed preferences → binary CE, directly on $\pi_\theta$.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\beta$: Implicit reward scale (= the KL coeff of the RL problem it replaces).

Objective:

$$
\mathcal{L}_\text{DPO}(\theta)=-\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\left[\log\sigma\left(\hat{r}_\theta(x,y_w)-\hat{r}_\theta(x,y_l)\right)\right]
$$
- $\hat{r}_\theta(x,y)=\beta\log\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}$: Implicit reward.

Backward:

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

    - $Z(x)$: Partition function, a sum over all sequences → intractable.

3. Solve for the reward:

    $$
    r(x,y)=\beta\log\frac{\pi^*(y|x)}{\pi_\text{ref}(y|x)}+\beta\log Z(x)
    $$

4. Bradley-Terry depends only on reward differences at the same $x$:

    $$
    p(y_w\succ y_l|x)=\sigma\left(r(x,y_w)-r(x,y_l)\right)
    $$

    → $\beta\log Z(x)$ appears in both terms & cancels. ✅Tractable.

5. Substitute $\pi^*\to\pi_\theta$ & maximize the likelihood of the observed preferences → $\mathcal{L}_\text{DPO}$.

→ The LM is the reward model: any $\pi_\theta$ implicitly defines a reward, & fitting the preferences fits the policy directly.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DPOLoss(nn.Module):
    def __init__(self, beta=0.1):
        super().__init__()
        self.beta = beta

    def forward(self, pol_w, pol_l, ref_w, ref_l):
        ## inputs: (B,) log pi(y|x) from SFTLoss.seq_logp (SFT section); ref_* from the FROZEN model
        r_w = self.beta * (pol_w - ref_w)        ## implicit rewards
        r_l = self.beta * (pol_l - ref_l)
        loss = -F.logsigmoid(r_w - r_l).mean()
        ## diagnostics: accuracy should rise; both rewards typically FALL
        acc = (r_w > r_l).float().mean()
        return loss, acc, r_w.mean(), r_l.mean()

## Example
pw, pl = torch.tensor([-13.0, -21.0]), torch.tensor([-17.0, -24.0])   ## policy log pi(y|x)
rw, rl = torch.tensor([-12.0, -20.0]), torch.tensor([-14.0, -20.0])   ## frozen reference
print([round(t.item(), 4) for t in DPOLoss()(pw, pl, rw, rl)])
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
- Both $\log\pi_\theta(y_w|x)$ & $\log\pi_\theta(y_l|x)$ usually fall during training.
- Length bias → longer responses tend to win.
- Overfits preference pairs quickly; 1 epoch is standard.

*Why does the chosen response's log-prob go down?*
- The loss constrains only the difference $\hat{r}_\theta(x,y_w)-\hat{r}_\theta(x,y_l)$, never either level.
- $y_w$ & $y_l$ share most of their tokens (prefix, style, formatting) → suppressing $y_l$ drags those shared tokens down too.
- The displaced probability mass goes to sequences in neither response → unconstrained by the objective.
- → Track reward accuracy & both reward levels separately; both levels falling fast = mass leaking to off-data sequences.

*Why is $\pi_\text{ref}$ necessary?*
- It is the KL anchor inherited from step 1 of the derivation → the reward becomes relative to $\pi_\text{ref}$.
- ⚠️ It regularizes drift in principle, ❌bound in practice: w/ deterministic preferences the empirical loss can ignore it (see IPO).
- W/o it, the margin can be maximized by degenerate solutions ($\pi_\theta(y_l|x)\to0$ on everything that resembles $y_l$).
- It also cancels prompt difficulty: a prompt where all responses are improbable does not dominate the loss.

*What must $\pi_\text{ref}$ be?*
- The derivation assumes Bradley-Terry preferences, ❌anything about who generated them.
- The practical requirement is distribution match: public preference sets were sampled from some $\pi_\text{SFT}$ → DPO's authors set $\pi_\text{ref}=\pi_\text{SFT}$ whenever it exists.
- W/o an SFT model, they fit $\pi_\text{ref}$ by MLE on the chosen responses → mitigates the shift between the true (unavailable) reference distribution & the one DPO uses.
- → An arbitrary checkpoint is not an error in the algebra; it is an uncontrolled distribution shift.

*What does $\beta$ control?*
- The implicit-reward scale = the KL strength. ⬇️$\beta$ → more drift from $\pi_\text{ref}$; ⬆️$\beta$ → stay put.
- $0.1$ is the common default; Tülu 3 instead used length-normalized DPO w/ $\beta=5$. The log-ratio is divided by $|y|$ → the scale is not comparable. {cite:p}`lambert2024tulu`

*Is DPO really RL-free?*
- As an algorithm, ✅: ❌reward model, ❌sampling, ❌policy gradient → just BCE on a fixed dataset.
- As a problem, ❌: every line of the derivation is the KL-constrained RL objective. DPO removes the loop, not the framing.
```

&nbsp;

### IPO
- **Name**: Identity Preference Optimization {cite:p}`azar2023general`
- **What**: Squared-loss preference matching w/ a finite target margin, ❌Bradley-Terry.
- **Why**: DPO overfits whenever the preferences are (nearly) deterministic.
    - BT maps a preference probability to a reward difference; probability $\to1$ maps to difference $\to\infty$.
    - Real datasets have 1 annotation per pair → the empirical probability is 0 or 1.
    - → The optimum drives $\pi_\theta(y_l|x)\to0$ & the KL term stops binding, no matter how large $\beta$ is.
- **How**: Replace $-\log\sigma(\cdot)$ (always rewards more margin) w/ a squared loss that pins the margin at a finite target.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\tau$: Regularization strength (the role $\beta$ plays in DPO).

Objective:

$$
\mathcal{L}_\text{IPO}(\theta)=\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\left[\left(h_\pi(y_w,y_l,x)-\frac{\tau^{-1}}{2}\right)^2\right]
$$
- $h_\pi(y_w,y_l,x)=\log\frac{\pi_\theta(y_w|x)\,\pi_\text{ref}(y_l|x)}{\pi_\theta(y_l|x)\,\pi_\text{ref}(y_w|x)}$: Log-ratio margin.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class IPOLoss(nn.Module):
    def __init__(self, tau=0.1):
        super().__init__()
        self.target = 1 / (2 * tau)   ## finite target margin

    def forward(self, pol_w, pol_l, ref_w, ref_l):
        ## same inputs as DPOLoss: (B,) seq log-probs
        h = (pol_w - ref_w) - (pol_l - ref_l)          ## log-ratio margin
        return ((h - self.target) ** 2).mean()         ## pulled TO the target, penalized past it

## Example (same tensors as DPOLoss)
pw, pl = torch.tensor([-13.0, -21.0]), torch.tensor([-17.0, -24.0])
rw, rl = torch.tensor([-12.0, -20.0]), torch.tensor([-14.0, -20.0])
print(IPOLoss(tau=0.1)(pw, pl, rw, rl).item())   ## 6.5: margins (2, 3) vs target 5
```
````

```{attention} Q&A
:class: dropdown
*Why does a squared loss fix the overfitting?*
- $-\log\sigma(z)$ is strictly decreasing in $z$ → the optimizer is always paid to grow the margin.
- $(z-c)^2$ penalizes overshoot → the margin is pulled to $c=\frac{\tau^{-1}}{2}$ & held there.
- → $\tau$ controls the distance from $\pi_\text{ref}$, as a regularizer should.

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
- **What**: Preference optimization from unpaired binary labels.
- **Why**: Pairing is the expensive part of preference data.
    - Production feedback arrives as thumbs-up / thumbs-down on single responses, ❌as comparisons.
    - Forcing pairs discards unmatched examples & invents comparisons nobody made.
- **How**: Score each response against a reference point w/ a prospect-theory value function.
    1. Implicit reward = the DPO log-ratio.
    2. Reference point $z_0$ = the KL between policy & reference, estimated from mismatched pairs in the microbatch.
    3. Value = a saturating function of (reward $-$ reference point), flipped in sign for undesirable responses.
    4. Weight the desirable & undesirable classes to handle imbalance.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\beta$: Risk-aversion / reward scale.
    - $\lambda_D$: Weight on desirable examples.
    - $\lambda_U$: Weight on undesirable examples.
- Misc:
    - $\lambda_y$: $\lambda_D$ or $\lambda_U$, whichever class $y$ is in.

Objective:

$$
\mathcal{L}_\text{KTO}(\theta)=\mathbb{E}_{(x,y)\sim\mathcal{D}}\left[\lambda_y-v(x,y)\right]
$$
- $v(x,y)$: Value function.

    $$
    v(x,y)=\begin{cases}\lambda_D\,\sigma\left(\beta\left(\hat{r}_\theta(x,y)-z_0\right)\right) & y\text{ desirable}\\ \lambda_U\,\sigma\left(\beta\left(z_0-\hat{r}_\theta(x,y)\right)\right) & y\text{ undesirable}\end{cases}
    $$

- $\hat{r}_\theta(x,y)=\log\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}$: Implicit reward.
- $z_0=\text{KL}\left(\pi_\theta(y'|x)\ \|\ \pi_\text{ref}(y'|x)\right)$: Reference point.
    - Estimated from mismatched $(x,y')$ pairs within the microbatch.
    - Clamped at 0. ❌Backprop through it.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class KTOLoss(nn.Module):
    def __init__(self, beta=0.1, lam_d=1.0, lam_u=1.0):
        super().__init__()
        self.beta, self.lam_d, self.lam_u = beta, lam_d, lam_u

    def forward(self, pol, ref, desirable, pol_mis, ref_mis):
        ## pol, ref: (B,) log pi(y|x); desirable: (B,) bool label per response
        ## pol_mis, ref_mis: (B,) log pi(y'|x) on MISMATCHED pairs -> KL estimate
        r = pol - ref
        z0 = (pol_mis - ref_mis).mean().clamp(min=0).detach()   ## reference point, no grad
        v = torch.where(desirable,
                        self.lam_d * torch.sigmoid(self.beta * (r - z0)),
                        self.lam_u * torch.sigmoid(self.beta * (z0 - r)))
        d = desirable.float()
        lam = self.lam_d * d + self.lam_u * (1 - d)
        return (lam - v).mean()

## Example
pol, ref = torch.tensor([-10.0, -12.0]), torch.tensor([-11.0, -11.0])   ## r = (1, -1)
desirable = torch.tensor([True, False])
pol_mis, ref_mis = torch.tensor([-30.0, -28.0]), torch.tensor([-31.0, -29.0])   ## z0 = 1
print(round(KTOLoss()(pol, ref, desirable, pol_mis, ref_mis).item(), 4))   ## 0.4751
```
````

```{attention} Q&A
:class: dropdown
*Why a reference point at all?*
- Prospect theory: humans evaluate outcomes as gains/losses relative to a baseline, ❌in absolute terms.
- Mechanically it blocks the cheapest exploit: raising the reward of a desirable response by uniformly inflating the policy also raises $z_0$ → no progress.
- → The model is forced to learn what specifically makes an output desirable.

*Why not backprop through $z_0$?*
- It exists to control where the loss saturates, ❌to be optimized.
- Differentiating it makes training unstable, & $z_0$ is a biased estimate anyway.

*How are $\lambda_D,\lambda_U$ set?*
- Both default to 1, then adjusted for class imbalance: keep $\frac{\lambda_D n_D}{\lambda_U n_U}\in\left[1,\frac{3}{2}\right]$.
- $n_D,n_U$: #desirable & #undesirable examples.

*What $\beta$?*
- Default $0.1$.
- $[0.01,0.10)$ if the model was already fine-tuned on the same data.
- $(0.10,0.50]$ if the reference is already strong, e.g., an aligned instruct model.

*What is the practical payoff?*
- Uses 1-sided feedback → far more usable data in a deployed product.
- Tolerates class imbalance explicitly, which pairwise losses cannot express.

*What is lost vs DPO?*
- The pairwise signal is the cleanest form of preference; splitting a pair into 2 independent labels discards the fact that both answer the same prompt.
```

&nbsp;

### ORPO
- **Name**: Odds Ratio Preference Optimization {cite:p}`hong2024orpo`
- **What**: SFT loss + an odds-ratio penalty, one stage, ❌reference model.
- **Why**: SFT has a side effect that a second stage then has to undo.
    - SFT raises the likelihood of the chosen response & of stylistically similar rejected responses ← ❌repulsive term.
    - Two stages also mean two datasets, two hyperparameter sets, & a frozen copy of the model in memory.
- **How**:
    1. Length-normalize the sequence log-likelihood → $P_\theta(y|x)$.
    2. Odds $=\frac{P}{1-P}$; the log ratio of chosen-to-rejected odds is the preference score.
    3. Loss = NLL on the chosen response $+\ \lambda\times$ the odds-ratio term. Train from the base model in one pass.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\lambda$: Weight on the odds-ratio term.

Objective:

$$
\mathcal{L}_\text{ORPO}=\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\left[\mathcal{L}_\text{SFT}+\lambda\cdot\mathcal{L}_\text{OR}\right],\qquad \mathcal{L}_\text{OR}=-\log\sigma\left(\log\textbf{OR}_\theta(y_w,y_l)\right)
$$
- $\mathcal{L}_\text{SFT}$: Ordinary NLL on $y_w$.
- $\textbf{OR}_\theta(y_w,y_l)=\frac{\textbf{odds}_\theta(y_w|x)}{\textbf{odds}_\theta(y_l|x)}$: Odds ratio.
- $\textbf{odds}_\theta(y|x)=\frac{P_\theta(y|x)}{1-P_\theta(y|x)}$: Odds.
- $\log P_\theta(y|x)=\frac{1}{|y|}\sum_{t=1}^{|y|}\log\pi_\theta(y_t|x,y_{<t})$: Length-normalized log-likelihood.
```

````{important} Code
:class: dropdown
```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ORPOLoss(nn.Module):
    def __init__(self, lam=0.1):
        super().__init__()
        self.lam = lam

    @staticmethod
    def log_odds(logp):
        ## log P/(1-P) from log P; log1p keeps it stable when P is tiny
        return logp - torch.log1p(-torch.exp(logp))

    def forward(self, avg_w, avg_l):
        ## avg_*: (B,) length-normalized log P(y|x) = SFTLoss.seq_logp / |y|
        l_or = -F.logsigmoid(self.log_odds(avg_w) - self.log_odds(avg_l)).mean()
        l_sft = -avg_w.mean()                  ## NLL on the chosen response -> absolute anchor
        return l_sft + self.lam * l_or         ## 1 model, no pi_ref

## Example: P(y_w)=0.5 -> odds 1; P(y_l)=0.2 -> odds 0.25
avg_w, avg_l = torch.tensor([math.log(0.5)]), torch.tensor([math.log(0.2)])
print(round(ORPOLoss()(avg_w, avg_l).item(), 4))   ## 0.7155 = 0.6931 + 0.1 * 0.2231
```
````

```{attention} Q&A
:class: dropdown
*Why odds instead of the plain probability ratio?*
- $\textbf{PR}=\frac{P_\theta(y_w|x)}{P_\theta(y_l|x)}$: Probability ratio.
- ⚠️ Not because the odds ratio is smaller: $\textbf{OR}=\textbf{PR}\cdot\frac{1-P_\theta(y_l|x)}{1-P_\theta(y_w|x)}\geq\textbf{PR}$ whenever $y_w$ leads.
- $\log\textbf{OR}$ has a wider range than $\log\textbf{PR}$ for the same probability pair.
- → Minimizing $-\log\sigma(\cdot)$ on $\log\textbf{PR}$ needs a more extreme probability gap → over-suppresses $y_l$.
- → The odds ratio is the milder, stabler penalty, which a term running alongside SFT needs; an over-extreme penalty would fight the NLL term.

*Why keep the SFT term?*
- The odds-ratio term is purely relative: lowering $y_l$ satisfies it as easily as raising $y_w$.
- The NLL term pins the chosen response's absolute likelihood → the model still learns to produce it.
- → This also replaces the missing $\pi_\text{ref}$ anchor.

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
- **Why**: DPO's implicit reward is not the quantity generation ranks by.
    - Length-normalized beam search & multiple-choice scoring rank by average log-likelihood per token.
    - DPO's reward is a sum of log-ratios against $\pi_\text{ref}$ → a response can win on the training reward yet lose under the model's own generation metric.
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

Objective:

$$
\mathcal{L}_\text{SimPO}(\theta)=-\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\left[\log\sigma\left(\hat{r}_\text{SimPO}(x,y_w)-\hat{r}_\text{SimPO}(x,y_l)-\gamma\right)\right]
$$
- $\hat{r}_\text{SimPO}(x,y)=\frac{\beta}{|y|}\log\pi_\theta(y|x)=\frac{\beta}{|y|}\sum_{t=1}^{|y|}\log\pi_\theta(y_t|x,y_{<t})$: Implicit reward.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimPOLoss(nn.Module):
    def __init__(self, beta=2.0, gamma=1.0):
        super().__init__()
        self.beta, self.gamma = beta, gamma

    def forward(self, logp_w, logp_l, len_w, len_l):
        ## logp_*: (B,) SFTLoss.seq_logp; len_*: (B,) #response tokens
        r_w = self.beta * logp_w / len_w      ## avg log-prob = length-normalized generation metric
        r_l = self.beta * logp_l / len_l      ## no pi_ref -> 1 model in memory
        return -F.logsigmoid(r_w - r_l - self.gamma).mean()   ## must win by >= gamma

## Example: avg log-probs -2 vs -3 -> rewards -4 vs -6 -> margin 2, minus gamma 1
logp_w, logp_l = torch.tensor([-6.0]), torch.tensor([-12.0])
len_w, len_l = torch.tensor([3.0]), torch.tensor([4.0])
print(round(SimPOLoss()(logp_w, logp_l, len_w, len_l).item(), 4))   ## 0.3133 = -log sigmoid(1)
```
````

```{attention} Q&A
:class: dropdown
*Why is length normalization load-bearing?*
- W/o it, the learned reward correlates strongly & positively w/ response length → **length exploitation** (longer, lower-quality output).
- W/ it, the reward difference rises for all pairs regardless of their lengths.
- ⚠️ DPO's log-ratio partly offsets this: it removes the per-token negative drift of raw log-probs, but is not normalized.

*What does $\gamma$ do?*
- Demands a strictly positive gap instead of merely "greater than".
- ⬆️$\gamma$ → reward accuracy rises monotonically, but AlpacaEval 2 win rate rises then falls.
- ← Large $\gamma$ flattens the reward distribution & lowers the log-likelihood of winning sequences → degeneration.
- → Tune $\gamma$ against generation quality, ❌against reward accuracy.

*What is the cost of dropping $\pi_\text{ref}$?*
- ❌KL anchor → nothing bounds the drift from the SFT model.
- The average-log-prob reward is bounded above by 0 → limits the most degenerate solutions, but is not a trust region.
- → An SFT regularization term is a common add-back.

*Why does aligning the reward w/ the generation metric matter?*
- Training optimizes the quantity you write down; serving selects by a different one.
- Any gap between them is a systematic train-inference mismatch, the same failure class as a mismatched chat template.
```

&nbsp;

````{dropdown} Table: Preference Objectives at a Glance
| Method | Implicit reward | Loss shape | $\pi_\text{ref}$ | Data | Length norm | Delta |
|:--|:--|:--|:--|:--|:--|:--|
| DPO | $\beta\log\frac{\pi_\theta}{\pi_\text{ref}}$ | $-\log\sigma(\Delta)$ | ✅ | Pairs | ❌ (partial, via the ratio) | ❌RM, ❌rollouts |
| IPO | Same | $(\Delta-\frac{\tau^{-1}}{2})^2$ | ✅ | Pairs | ❌ | Bounded margin → ❌overfitting |
| KTO | $\log\frac{\pi_\theta}{\pi_\text{ref}}$ | $\lambda_y\left(1-\sigma(s_y\beta(\hat{r}-z_0))\right)$ | ✅ | Unpaired labels | ❌ | ❌Pairing requirement |
| ORPO | $\log\textbf{odds}_\theta$ | $\mathcal{L}_\text{SFT}+\lambda\mathcal{L}_\text{OR}$ | ❌ | Pairs | ✅ | 1 stage, 1 model |
| SimPO | $\frac{\beta}{\lvert y\rvert}\log\pi_\theta$ | $-\log\sigma(\Delta-\gamma)$ | ❌ | Pairs | ✅ | Reward ≈ generation metric |

- $\Delta$: Implicit reward margin between chosen & rejected.
- $s_y$: $+1$ if $y$ desirable, $-1$ if undesirable.
````