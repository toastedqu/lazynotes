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
# Issues
This page lists common issues in DL in Alphabetic order.

## Batch-size Pathologies
- **What**: Training behavior depends on batch size.
    - Small: noisy grads, unstable optimization.
    - Large: sharp-minima / generalization drop, fragile training dynamics.
- **Why**:
    1. Gradient noise scale changes with batch size.
        - Mini-batch grad = true grad + noise.
        - ⬆️ Batch size → ⬇️ noise → optimization becomes more "deterministic".
        - BUT what if the noise is beneficial?
        - It helps exploration and can bias toward flatter minima → ⬆️ Generalization.
    2. Batch size interacts with LR & Update Frequency
        - For fixed epochs: ⬆️ Batch size → ⬇️ Param updates per epoch.
        - If fixed LR/schedule, we can under-train (not enough steps) or destabilize (LR too high for the new regime).
    3. Normalization & regularization may depend on batch size.
        - ⬇️ Batch size → BatchNorm stats become noisy → ⬇️ Training stability & perf.
        - Some regularizers (e.g., weight decay) behave differently when we scale LR with batch size.
    4. "Critical batch size" phenomenon.
        - Up to a task-dependent threshold, larger batch improves throughput and keeps convergence similar.
        - Beyond that, returns diminish → More batch doesn’t reduce wall-clock meaningfully and may hurt generalization.
        - Large-batch training can match train loss but lose test accuracy.
- **How to mitigate**:
    - Retune LR + schedule for the new batch size.
    - Stabilize normalization via LayerNorm/GroupNorm/RMSNorm.
    - Preserve "useful noise" with large batch size.
        - Add stochasticity: stronger data augmentation, dropout, stochastic depth, etc.
    - Grad Accumulation.

&nbsp;

## Catastrophic Forgetting
- **What**: When a NN trained on Task A is later trained on Task B, it can lose performance on Task A.
- **Why**: Standard training assumes **i.i.d. + stationary** data. BUT continual learning violates this assumption.
    1. Train on Task A → params settle into a good basin for A.
    2. Switch to Task B (no/limited A data) → optimize only for B’s loss.
    3. Grads for Task B push params into basin B, which isn't meant for basin A.
    4. Result: B improves, A collapses (forgetting).
- **How to mitigate**:
    1. Keep & Mix some old data into new tasks.
    2. Penalize changing "important" params for old tasks.
    3. Separate params per task (e.g., adapters, task-specific heads, routing/mixture-of-experts).

```{attention} Q&A
:class: dropdown
*Is catastrophic forgetting ALWAYS bad?*
- Only if you really need to retain earlier skills → If you only care about the latest data, forgetting can be a feature (fast adaptation).

*What’s the #1 practical cause during fine-tuning?*
- Unconstrained updates + small/new dataset + too-large LR → weight drift.  
```

&nbsp;

## Covariate Shift
- **What**: A learned function suddenly sees its input data distribution change, while the conditional distribution of labels given data is presumed unchanged.
    - Dataset Covariate Shift: Training $\leftrightarrow$ Inference.
    - Internal Covariate Shift: Layer $\leftrightarrow$ Layer.
- **Why**:
    - Dataset: Difference between training & inference data.
    - Internal:
        1. The weights of earlier layer keep changing.
        2. → Later layers must re-adapt to each moving distribution
        3. → BUT the underlying mapping is unchanged
        4. → ⬇️Training stability
- **How to mitigate**:
	- BatchNorm: Re-center & Re-scale each layer's output channel to $N(0,1)$ → Steady distribution → Steady training
        - BUT less effective than LayerNorm ← The normalization is independent between channels.
    - LayerNorm: Re-center & Re-scale each layer's outputs to $N(0,1)$ → Steady distribution → Steady training
	- Proper weight initialization (e.g., Xavier, He, etc.) → Stable gradients
	- Smaller learning rates → Reduce the magnitude of param updates
		- BUT slower convergence.
	- Adaptive learning rates (e.g., Adam) → dynamically control the magnitude of param updates.

## Overfitting
- **What**: The model performs well on the training data but badly on the test/unseen data.
- **Why**: Too high model complexity relative to training data size → Poor generalization.
- **How to mitigate**:
	- Parameter regularization (L1/L2) → reduce the magnitude of params.
    - Dropout → reduce dependency on certain neurons → promote learning of robust features.
    - Data augmentation → broaden training data distribution.
    - Early stopping → halt training before validation performance drops.
    - Cross validation → ensure consistent model performance across different data splits.
	- Reduce model complexity → a simpler model is less likely to capture too many details in the training data.

## Vanishing/Exploding Gradient
- **What**:
    - **Vanishing gradients**: The gradients become smaller and smaller during backprop. They have almost no effect to the front layers.
        - → Slows & potentially stops the learning process for the front layers.
    - **Exploding gradients**: The gradients become larger and larger during backprop. They change the weights of the front layers too much.
        - → Harder convergence & Training instability.
- **Why**:
    - Too many layers.
    - Too small/large gradient on certain layers (e.g., Sigmoid, Tanh, extremely large weight initialization, etc.) → propagates via chain rule.
- **How to mitigate**:
	- Proper activation functions (e.g., ReLU and variants) → Low likelihood of tiny gradients
    - Proper weight initialization (e.g., Xavier, He, etc.) → Maintain stable gradients
    - Batch Normalization → Maintain scaled gradients
    - Residual connection → Shortcut paths for gradient flow
    - Gradient clipping → Caps gradient magnitude

&nbsp;

## Dying ReLU
- **What**: [ReLU](../foundation/activation.md#relu) unit whose pre-activation is $\le0$ for every input → constant 0 output, zero grad.
- **Why**: ReLU's grad is 0 for negative inputs.
    - Dead unit → no grad reaches its weights & bias → no update can revive it from its own loss signal.
- **How**:
    1. Large update (high LR, grad burst) or a very negative bias pushes $z$ below 0 for all inputs.
    2. Output = 0 for every sample → grads w.r.t. its weights & bias = 0.
    3. No new grad signal → plain SGD stops moving those params → the unit stays dead.
        - Momentum/Adam: stored moments keep moving them for a while ($m_t=\beta_1m_{t-1}$), decaying geometrically.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{x}_i$: Input of the unit for sample $i$.
- Params:
    - $\mathbf{w}$: Incoming weights.
    - $b$: Bias.
- Misc:
    - $z_i=\mathbf{w}^\top\mathbf{x}_i+b$: Pre-activation.
    - $\delta_i=\partial\mathcal{L}/\partial\,\text{ReLU}(z_i)$: Upstream grad.

Backward:

$$
\frac{\partial\mathcal{L}}{\partial\mathbf{w}}=\sum_{i=1}^{m}\delta_i\,\mathbb{1}[z_i>0]\,\mathbf{x}_i,\qquad\frac{\partial\mathcal{L}}{\partial b}=\sum_{i=1}^{m}\delta_i\,\mathbb{1}[z_i>0]
$$

Dead:

$$
z_i\le0\ \ \forall i\ \Rightarrow\ \frac{\partial\mathcal{L}}{\partial\mathbf{w}}=\mathbf{0},\ \frac{\partial\mathcal{L}}{\partial b}=0
$$
```

```{note} Example
:class: dropdown
- Inputs in $[0,1]^2$. Unit w/ $\mathbf{w}=(1,1)$, $b=-10$.
- $z\le1+1-10=-8<0$ for every input → output 0 → grads 0 → dead.
- Same unit w/ $b=-1$: $z>0$ only when $x_1+x_2>1$ → sparse but alive.
```

```{attention} Q&A
:class: dropdown
*How to mitigate?*
- ⬇️LR or [warmup](schedule.md#warmup) → fewer oversized updates.
- [He init](train.md#he-initialization) → pre-activations start well scaled.
- [BatchNorm](../foundation/module.md#batchnorm) before ReLU → re-centers $z$ → ~half the inputs positive.
- Non-zero negative slope: [LReLU](../foundation/activation.md#lrelu), [PReLU](../foundation/activation.md#prelu), [ELU](../foundation/activation.md#elu), [GELU](../foundation/activation.md#gelu).
- [Gradient clipping](train.md#gradient-clipping) → bounds grad bursts.
- Small positive bias init: sometimes used. ⚠️ Mixed evidence.

*Dead vs sparse?*
- Sparse: $z<0$ for SOME inputs → a feature, ReLU's selling point.
- Dead: $z\le0$ for ALL inputs → wasted capacity.

*How to detect it?*
- Over a full validation pass, count units w/ zero output on every sample.

*Can a dead unit come back?*
- Hidden layer: ✅ possible. Earlier layers keep training → its inputs $\mathbf{x}_i$ shift → some $z_i$ may turn positive.
- First layer on fixed data, plain SGD w/o weight decay: ❌. Inputs never change & params never move.
- Momentum/Adam residue: moves params, but along the last update direction (often the one that killed the unit). Revival possible, not expected.
- Decoupled weight decay (or L2 under SGD) does not help. It scales $\mathbf{w}$ & $b$ by the same factor → $z_i$ keeps its sign.
- L2 under Adam: per-coordinate normalization shrinks $\mathbf{w}$ & $b$ by different factors → the sign of $z_i$ can change.

*Do residual connections fix it?*
- No. They keep grads flowing to earlier layers around the dead unit. The dead unit itself still gets zero grad.
```

&nbsp;

## Loss Spike
- **What**: Sudden jump in training loss within a few steps, usually followed by recovery.
- **Why**: Each spike wastes steps. A spike that does not recover forces a restart → large compute cost at scale.
    - ⚠️ No established cause. PaLM reports spikes w/o a single identified mechanism.
- **How**: Leading hypotheses, ❌proven:
    1. Curvature or Adam's preconditioned curvature rises during training (progressive sharpening).
    2. The step size crosses the local stability threshold → params overshoot → loss jumps.
    3. Curvature or preconditioner shrinks (self-stabilization) → loss returns to trend.
    - Variant: Adam's $\hat{\mathbf{v}}$ is stale-small on some coordinates → a sudden grad → oversized step. Several reports remove spikes by ⬇️$\beta_2$.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\eta$: Learning rate.
    - $\beta_1,\beta_2$: Adam moment decay rates.
- Misc:
    - $g_t$: Grad of 1 coordinate at step $t$.
    - $m_t,v_t$: Adam 1st & 2nd moment EMAs of that coordinate.

Isolated-gradient ratio (zero grad history, then 1 grad $g_t$; bias correction & $\epsilon$ ignored):

$$
\frac{|\Delta w_t|}{\eta}=\frac{|m_t|}{\sqrt{v_t}}=\frac{1-\beta_1}{\sqrt{1-\beta_2}}
$$

- $\beta_1=0.9,\ \beta_2=0.999$: $\approx3.16$. $\beta_1=0.9,\ \beta_2=0.95$: $\approx0.45$.

Worst case over all grad histories (infinite history, same simplifications):

$$
\sup\frac{|m_t|}{\sqrt{v_t}}=\frac{1-\beta_1}{\sqrt{1-\beta_2}}\cdot\frac{1}{\sqrt{1-\beta_1^2/\beta_2}}\quad(\beta_1^2<\beta_2)
$$

- $\beta_1=0.9,\ \beta_2=0.999$: $\approx7.27$. $\beta_1=0.9,\ \beta_2=0.95$: $\approx1.17$.
- $\beta_1^2\ge\beta_2$: unbounded.
- → Adam's step is ❌bounded by $\eta$ in general.
```

```{tip} Derivation
:class: dropdown
*Why can a sudden grad cause a step $>\eta$?*
1. Before: $m_{t-1}\approx0$, $v_{t-1}\approx0$ (small past grads). Late in training → bias correction ≈ 1.
2. Large grad $g_t$: $m_t=(1-\beta_1)g_t$, $v_t=(1-\beta_2)g_t^2$.
3. $|\Delta w_t|/\eta=|m_t|/\sqrt{v_t}=(1-\beta_1)/\sqrt{1-\beta_2}$.
4. $\beta_2$⬇️ → $v_t$ reacts faster → smaller ratio.
- Grad clipping does not remove this: it scales $g_t$, but the ratio $m_t/\sqrt{v_t}$ is independent of the scale of $g_t$ here.

*Where does the worst case come from?*
1. Unroll: $m_t=(1-\beta_1)\sum_k\beta_1^kg_{t-k}$, $v_t=(1-\beta_2)\sum_k\beta_2^kg_{t-k}^2$.
2. Write $\beta_1^kg_{t-k}=(\beta_1/\sqrt{\beta_2})^k\cdot\sqrt{\beta_2}^{\,k}g_{t-k}$.
3. Cauchy-Schwarz: $\left|\sum_k\beta_1^kg_{t-k}\right|\le\sqrt{\sum_k(\beta_1^2/\beta_2)^k}\sqrt{\sum_k\beta_2^kg_{t-k}^2}$.
4. Geometric sum $=1/(1-\beta_1^2/\beta_2)$ if $\beta_1^2<\beta_2$ → the formula above.
5. Equality: $g_{t-k}\propto(\beta_1/\beta_2)^k$ → grads grow geometrically toward step $t$.
```

```{dropdown} Table: Spike Hypotheses & Mitigations
| Hypothesis | Signal | Mitigation |
|:--|:--|:--|
| Stale Adam 2nd moment | Update/LR ratio jumps on some coordinates | ⬇️$\beta_2$ (0.95 common in LLM pretraining) |
| Step too large for local curvature | Spike after a period of rising sharpness | ⬇️LR, [Warmup](schedule.md#warmup) |
| Full LR before optimizer stats settle | Spikes in the first steps | [Warmup](schedule.md#warmup) |
| Grad-norm burst | Global grad norm ⬆️ | [Gradient clipping](train.md#gradient-clipping). Not sufficient alone. |
| Unlucky data × param state | Spike does not reproduce from an earlier checkpoint | Restart before the spike & skip batches |
```

```{attention} Q&A
:class: dropdown
*Spike vs divergence?*
- Spike: fast jump, then returns to trend.
- [Training divergence](#training-divergence): loss rises & does not recover. Often slow, w/ an identifiable drifting quantity.

*How did PaLM handle spikes?* {cite:p}`chowdhery2022palm`
- ~20 spikes in the 540B run, despite grad clipping.
- Fix: restart from a checkpoint ~100 steps before the spike → skip ~200–500 data batches.
- Re-running the same batches from an earlier checkpoint gave no spike → cause = data × param state, ❌bad data alone.

*Why doesn't grad clipping prevent spikes?*
- Clipping bounds the global grad norm. Adam normalizes per coordinate → the step depends on $m_t/\sqrt{v_t}$, ❌grad scale.

*What to monitor?*
- Global grad norm, update RMS vs param RMS, max attention logit per layer, $\log Z$.
```

&nbsp;

## Training Divergence
- **What**: Loss rises & does not recover.
- **Why**: A hidden quantity drifts over many steps until training breaks.
    - Unlike a [loss spike](#loss-spike), the drift is often measurable before failure.
- **How**:
    1. A quantity with no restoring force grows: attention logits, output-logit level, or param norms.
    2. Training looks normal until it crosses a threshold (softmax saturation, precision loss, update collapse).
    3. → Loss rises, grads become unstable, NaN.

```{dropdown} Table: Divergence Causes & Fixes
| Cause | Signal | Fix |
|:--|:--|:--|
| [Attention logit growth](#attention-logit-growth) | Max attention logit ⬆️, attention entropy → 0 | [QK-norm](../../nlp/attn.md#qk-norm), ⬇️LR |
| [Output logit divergence](#output-logit-divergence) | $\log Z$ drifts far from 0, toward the end of training | [Z-loss](../../nlp/pre/obj.md#z-loss), [weight decay](reg.md#weight-decay) |
| Grad RMS ≈ Adam $\epsilon$ at scale | Update RMS collapses, layers stop learning | ⬇️$\epsilon$ |

- $Z=\sum_k e^{z_k}$: Softmax normalizer of the output logits $z_k$.
```

```{attention} Q&A
:class: dropdown
*What did small-scale proxy studies show?* {cite:p}`wortsman2024small`
- Scope: slow-divergence instabilities, ❌fast loss spikes.
- Attention logit growth & output logit divergence also appear in small models at high LR. The large-scale fixes work there too.
- LR sensitivity ⬆️ w/ model size. Longer warmup & LR-independent weight decay ⬇️ LR sensitivity.
- Adam $\epsilon$: grad RMS ⬇️ w/ scale & LR. Grad RMS ≈ $\epsilon$ → update shrinks. 4.8B model at LR 0.3: $\epsilon=10^{-15}$ improved loss. $\epsilon=10^{-6}$ diverged.

*Always lower Adam $\epsilon$?*
- ⚠️ No. The direction is setup-dependent. Larger $\epsilon$ damps steps on coordinates w/ tiny $\hat v$. RoBERTa used $10^{-6}$.
```

&nbsp;

### Attention Logit Growth
- **What**: Unbounded growth of pre-softmax attention logits during training. {cite:p}`dehghani2023scaling`
- **Why**: Logits are quadratic in the param norms of $W_Q$ & $W_K$ → they grow fast as weights grow.
- **How**:
    1. $\|W_Q\|,\|W_K\|$ ⬆️ → logits ⬆️. ViT-22B study, 8B model w/o normalization: $>50{,}000$ in magnitude.
    2. Softmax saturates → attention ≈ one-hot, entropy ≈ 0.
    3. → Unstable losses & grads → divergence after a few thousand steps.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{x}_i,\mathbf{x}_j$: Token representations (after pre-norm).
- Params:
    - $W_Q,W_K$: Query & key projections.
- Misc:
    - $d_k$: Head dim.
    - $s_{ij}$: Attention logit.
    - $\mathbf{a}$: Attention weights of 1 query row.

Logit & bound:

$$
s_{ij}=\frac{(W_Q\mathbf{x}_i)^\top(W_K\mathbf{x}_j)}{\sqrt{d_k}},\qquad|s_{ij}|\le\frac{\|W_Q\|_2\|W_K\|_2\|\mathbf{x}_i\|\|\mathbf{x}_j\|}{\sqrt{d_k}}
$$

- Scale $W_Q,W_K$ by $c$ → logits scale by $c^2$.

Softmax Jacobian:

$$
\frac{\partial\mathbf{a}}{\partial\mathbf{s}}=\text{diag}(\mathbf{a})-\mathbf{a}\mathbf{a}^\top\ \xrightarrow{\ \mathbf{a}\to\text{one-hot}\ }\ \mathbf{0}
$$

- → Grads through a saturated head vanish.
```

```{attention} Q&A
:class: dropdown
*Why doesn't $1/\sqrt{d_k}$ prevent it?*
- It fixes the logit scale at init only. → [QK-Norm Q&A](../../nlp/attn.md#qk-norm).

*Is it only a large-model problem?*
- No. Small models show it at high LR. W/o QK-norm, the divergence LR drops as size ⬆️.
- Driven by growing query & key norms, ❌growing alignment. In the proxy study, every run w/ max attention logit $>10^4$ diverged. {cite:p}`wortsman2024small`

*Fixes?*
- [QK-norm](../../nlp/attn.md#qk-norm): normalize $\mathbf{q},\mathbf{k}$ → logits bounded by construction.
- ⬇️LR. Original ViT lowered its LR as model size grew.
- Logit soft-capping ([Gemma](../../nlp/models.md#gemma)) or QK-Clip ([Kimi](../../nlp/models.md#kimi)).
```

&nbsp;

### Output Logit Divergence
- **What**: Drift of the output logits away from the log-probs, i.e., $\log Z$ far from 0. {cite:p}`chowdhery2022palm`
- **Why**: Softmax is shift-invariant → cross-entropy has no restoring force on the overall logit level.
- **How**:
    1. Updates for other reasons move all logits together as a side effect.
    2. Nothing pulls them back → logits drift, often to very negative values.
    3. Late in training (unlike attention logit growth) → loss diverges.
    - ⚠️ Why the drift ends in divergence: not settled. Precision loss at large logit magnitudes is a common hypothesis.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{z}\in\mathbb{R}^K$: Output logits.
    - $\mathbf{y}$: One-hot target.
- Misc:
    - $K$: #classes (vocab size).
    - $\mathbf{p}=\text{softmax}(\mathbf{z})$: Predicted probs.
    - $Z=\sum_{k}e^{z_k}$: Softmax normalizer.
    - $\mathbf{1}$: All-ones vector.

Shift invariance:

$$
\text{CE}(\mathbf{z}+c\mathbf{1})=\text{CE}(\mathbf{z})\ \Rightarrow\ \mathbf{1}^\top\nabla_\mathbf{z}\text{CE}=\sum_k(p_k-y_k)=0
$$

- → CE grad has no component along $\mathbf{1}$.
- $\log p_k=z_k-\log Z$ → logits = log-probs only if $\log Z=0$.

Z-loss grad along $\mathbf{1}$ ([Z-Loss](../../nlp/pre/obj.md#z-loss)):

$$
\nabla_\mathbf{z}\left(\log^2Z\right)=2\log Z\cdot\mathbf{p}\ \Rightarrow\ \mathbf{1}^\top\nabla_\mathbf{z}\left(\log^2Z\right)=2\log Z
$$

- → Restoring force that pulls $\log Z$ to 0.
```

```{attention} Q&A
:class: dropdown
*What fixes it?*
- [Z-loss](../../nlp/pre/obj.md#z-loss) resolves it at every scale tested.
- It appears in models w/o weight decay at any scale. [Weight decay](reg.md#weight-decay) also mitigates it in larger models. {cite:p}`wortsman2024small`

*Why does weight decay help?*
- Output logits = output-layer weights × final hidden state → bounded weights → bounded logit drift.

*How does it differ from attention logit growth?*
- Location: output layer vs attention.
- Timing: toward the end of training vs earlier.
- Fix: z-loss vs QK-norm.
```

&nbsp;

## Shortcut Learning
- **What**: Learning a decision rule that passes i.i.d. tests but fails out-of-distribution (o.o.d.). {cite:p}`geirhos2020shortcut`
- **Why**: Many rules fit the training data. Training finds the easiest one, not the intended one.
    - "Easy" depends on architecture, data, loss & optimizer.
- **How**:
    1. Data has a **shortcut opportunity**: a feature that correlates w/ the label in train & i.i.d. test (grass ↔ cow).
    2. The shortcut is easier → learned first → cross-entropy ⬇️ → little pressure to learn more.
    3. i.i.d. test shares the correlation → high benchmark score.
    4. o.o.d. test breaks the correlation → failure.

```{dropdown} Table: Taxonomy of Decision Rules
| Rule uses | Train | i.i.d. test | o.o.d. test |
|:--|:--|:--|:--|
| Uninformative features | ❌ | ❌ | ❌ |
| Overfitting features | ✅ | ❌ | ❌ |
| Shortcut features | ✅ | ✅ | ❌ |
| Intended features | ✅ | ✅ | ✅ |
```

```{note} Example
:class: dropdown
- Cow on a beach → misclassified ← model learned grassland background.
- Pneumonia detector → relies on a hospital-specific metal token in the X-ray.
- Cat shape w/ elephant texture → "elephant" ← ImageNet CNNs rely on texture over global shape.
- BERT on argument reasoning → the cue word "not" alone gives above-chance accuracy.
- Toy: stars vs moons, a fully connected net learns object *location* → chance level once location is controlled.
- RL: a Tetris agent pauses the game forever to avoid losing → [reward hacking](../../nlp/rh.md).
```

```{attention} Q&A
:class: dropdown
*Shortcut learning vs overfitting?*
- [Overfitting](#overfitting): fails on i.i.d. test data.
- Shortcut: passes i.i.d. test data → invisible on standard benchmarks.

*Shortcut learning vs covariate shift?*
- A distribution shift exposes the shortcut. [Covariate shift](#covariate-shift) is one case. $P(y|x)$ can also change.
- The shortcut was learned at train time. The shift only reveals it.

*Does more data fix it?*
- Rarely. Shortcut opportunities persist as long as the correlation exists in the data.

*How to detect it?*
- o.o.d. tests: cue-conflict stimuli, controlled datasets w/ planted correlations, adversarial examples.
- Compare against surprisingly strong simple baselines.

*How to mitigate?*
- Data: block the shortcut ([augmentation](../../cv/image.md#data-augmentation), domain randomization, balanced data).
- Architecture: build in the invariance (a CNN cannot use absolute location as easily as an MLP).
- Training across multiple environments/domains → favor features stable across them.
- Loss: modifications that force the use of all available information.

*What does it say about intelligence/AC?*
- Benchmark success ≠ the ability the benchmark names. Be careful before claiming "object recognition" or "language understanding".
- Animals also show it (unintended cue learning, Clever Hans) → likely a general property of learning systems, ❌DL-specific.
```

&nbsp;
