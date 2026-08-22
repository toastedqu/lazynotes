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
# PEFT
- **Name**: Parameter-Efficient Fine-Tuning
- **What**: Training a small set of params while the pretrained weights stay frozen.
- **Why**: Full fine-tuning does not scale to many tasks or small budgets.
    - Optimizer state dominates: Adam keeps 2 moments per trainable param → ~12–16 bytes/param **on top of** the weights & gradients.
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
- Hypothesis: adaptation to a downstream task has a low **intrinsic dimension** — the required update lives in a tiny subspace of weight space.
- Support is empirical (matching full FT at 0.01–1% trainable params), ❌proved.
- The pretrained model already contains the capability; the update only has to *select* it.

*Where does the memory actually go?*
- Weights: frozen → can even be quantized.
- Gradients + optimizer state: $\propto$ **trainable** params → this is the entire win.
- Activations: essentially unchanged ← you still backprop through the whole frozen network.
- → PEFT does **not** remove the need for gradient checkpointing on long sequences.

*When is full FT still the right call?*
- Large distribution shift (new language, new modality) → CPT-scale change, ❌a low-rank nudge.
- Abundant data & budget, single deployment target.
- → PEFT's gap over full FT widens exactly as the required change grows.

*Can you stack PEFT on top of preference optimization?*
- ✅ Orthogonal: the adapter is a parameterization, the objective is a loss. LoRA + DPO is routine.
- ⚠️ $\pi_\text{ref}$ becomes free — disable the adapters & the same model **is** the reference.
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

```{attention} Q&A
:class: dropdown
*Why near-identity init?*
- A randomly initialized inserted module corrupts the pretrained function at step 0 → the model has to recover before it can learn.
- Near-zero $U$ makes the branch a no-op initially.

*Why did adapters lose to LoRA?*
- They are **sequential**: the branch must run before the next layer → its cost cannot be folded into $W_0$.
- → Permanent extra depth & inference latency at every layer, every token.
- Worse under model parallelism ← an extra synchronization point per layer.

*What survives of the idea?*
- The bottleneck-plus-residual pattern is everywhere (LoRA is its linear, parallel, mergeable cousin).
- Adapters remain attractive when you want a **nonlinear** task-specific transform.
```

&nbsp;

### LoRA
- **Name**: Low-Rank Adaptation {cite:p}`hu2021lora`
- **What**: Trainable rank-$r$ update added **in parallel** to a frozen weight matrix.
- **Why**: Adapters buy parameter efficiency by paying inference latency.
    - A sequential module cannot be folded into $W_0$ → the cost is permanent.
    - A parallel **linear** branch can be added into the weight matrix after training → free at inference.
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
- Hyperparams:
    - $W_0\in\mathbb{R}^{d_\text{out}\times d_\text{in}}$: Frozen pretrained weight.
    - $r$: Rank.
    - $\alpha$: Scaling numerator.

Forward:

$$
\mathbf{h}=W_0\mathbf{x}+\frac{\alpha}{r}BA\mathbf{x}
$$

Backward (w/ $\mathbf{g}=\frac{\partial\mathcal{L}}{\partial\mathbf{h}}$):

$$
\frac{\partial\mathcal{L}}{\partial B}=\frac{\alpha}{r}\,\mathbf{g}\left(A\mathbf{x}\right)^T,\qquad \frac{\partial\mathcal{L}}{\partial A}=\frac{\alpha}{r}\,B^T\mathbf{g}\,\mathbf{x}^T,\qquad \frac{\partial\mathcal{L}}{\partial W_0}=\varnothing
$$

Trainable params: $r(d_\text{in}+d_\text{out})$ vs $d_\text{in}d_\text{out}$.
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
- ❌Added inference latency after merging — unlike adapters.
- Adapters are MBs → many task-specific ones can be swapped, or served concurrently against 1 base.

*Cons?*
- Rank caps how much of the update is expressible → lags full FT when the needed change is large.
- Sensitive to $r$, $\alpha$, target modules, & LR simultaneously.
- ⚠️ Saves optimizer memory, ❌activation memory.

*Why $B=0$ & $A$ random, instead of both zero or both random?*
- Both random → $\Delta W\neq0$ at init → the pretrained function is corrupted before training starts.
- Both zero → $\frac{\partial\mathcal{L}}{\partial A}\propto B^T=0$ **and** $\frac{\partial\mathcal{L}}{\partial B}\propto A^T=0$ → dead branch, forever.
- $B=0$, $A\neq0$ → output unchanged, yet $\frac{\partial\mathcal{L}}{\partial B}\neq0$ → $B$ moves first, then $A$ receives gradient.

*What is $\alpha$ for?*
- $\frac{\alpha}{r}$ makes the branch's effective magnitude roughly independent of $r$.
- → Change $r$ w/o re-tuning the LR. Common settings: $\alpha=r$ or $\alpha=2r$.

*Which modules should it target?*
- The original work adapted attention projections only (best results from $W_q,W_v$ at a fixed budget).
- Current practice applies it to **all** linear layers incl. the MLP, which generally helps at equal total params.

*What LR?*
- ~10× the full-FT LR (order $10^{-4}$) ← the branch starts at 0 & has few params to move.
- Reusing the full-FT LR is the most common reason "LoRA didn't learn anything".

*Does the rank need to be large?*
- Style, format, tone, persona → $r=8$–$16$ is usually plenty.
- New knowledge or a real distribution shift → raise $r$, or accept that full FT / CPT is the right tool.

*Why can it be merged when adapters cannot?*
- The branch is **linear** & **parallel** → $W_0\mathbf{x}+\Delta W\mathbf{x}=(W_0+\Delta W)\mathbf{x}$.
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
    → 65B fine-tuned on a single 48GB GPU while matching 16-bit fine-tuning quality.

```{attention} Q&A
:class: dropdown
*Why doesn't 4-bit destroy quality?*
- The base is **frozen** & used only in the forward pass → quantization error is a fixed perturbation, ❌accumulating noise.
- Weights are dequantized to 16-bit per block for the actual matmul.
- The 16-bit adapters are trained **through** the quantized base → they absorb its error.

*What does it cost?*
- Dequantization on every forward → slower per step than plain LoRA.
- → Trade throughput for the ability to fit the model at all.

*Why NF4 rather than int4?*
- Pretrained weights are approximately zero-centered normal.
- NF4 places its 16 levels at the quantiles of a normal distribution → equal expected mass per level.
- Int4's uniform levels waste resolution in the tails where almost no weights live.

*Can you merge the adapter?*
- ❌ Cleanly into the 4-bit base — merging then re-quantizing discards the adapter's precision.
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
    - $m\in\mathbb{R}^{1\times d_\text{in}}$: Trainable magnitude vector, init $\|W_0\|_c$.
    - $A,B$: LoRA factors, as in LoRA.
- Hyperparams:
    - $W_0$: Frozen pretrained weight.
- Misc:
    - $\|\cdot\|_c$: Column-wise $L_2$ norm (1 scalar per column).

Decomposition:

$$
W=m\frac{V}{\|V\|_c},\qquad m=\|W\|_c
$$
- $V$: Directional component.

Forward weight:

$$
W'=m\frac{W_0+BA}{\|W_0+BA\|_c}
$$

$m$ & $BA$ are trainable; $W_0$ is frozen. The normalization decouples the two updates, which is the entire point.
```

```{attention} Q&A
:class: dropdown
*Why does decoupling help?*
- Under the decomposition, DoRA's magnitude-direction correlation is $-0.31$ vs full FT's $-0.62$ & LoRA's $+0.83$.
- → It recovers full FT's qualitative learning pattern at LoRA's parameter budget.

*What does it cost?*
- The column-norm & renormalization every step → extra training compute & memory over LoRA.
- ✅Still mergeable → ❌inference overhead.

*When is the gain largest?*
- Low rank. DoRA degrades much more gracefully than LoRA as $r$ shrinks.
- → Useful precisely where LoRA's expressivity limit binds.
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
- **What**: Trainable soft prompt at the **input embedding layer** only. {cite:p}`lester2021power`
- **Why**: Prefix tuning injects at every layer → more params & more plumbing than the effect requires.
- **How**: Prepend $p$ trainable embedding vectors to the input embeddings; freeze everything else, incl. all attention layers.

```{attention} Q&A
:class: dropdown
*When does it actually work?*
- Only at scale: the gap to full FT closes as the model passes ~$10^{10}$ params, and is large below that.
- → "The power of scale": the bigger the frozen model, the less you need to touch it.

*Pros?*
- The smallest footprint of any PEFT method — a handful of vectors per task.
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
| Prompt Tuning | Input-embedding prompt | ❌ | ✅ Context | Needs a very large frozen model |
| BitFit | Bias terms only {cite:p}`benzaken2021bitfit` | — (already in $W_0$) | ❌ | Smallest possible change to existing params |
| IA³ | Learned rescaling vectors for K, V, FFN {cite:p}`liu2022fewshot` | ✅ | ❌ | Even fewer params than LoRA |
````