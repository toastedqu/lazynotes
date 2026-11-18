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
# Regularization
DL-specific regularizers. Related: [L1](../../ml/obj.md#l1), [L2](../../ml/obj.md#l2), [Dropout](../foundation/module.md#dropout), [Label Smoothing](../../ml/obj.md#label-smoothing), [Data Augmentation](../../cv/image.md#data-augmentation).

Notations:
- $\mathbf{w}$: Trainable params, flattened
- $\mathcal{L}(\mathbf{w})$: Training loss
- $t$: Training step
- $\mathbf{g}_t=\nabla\mathcal{L}(\mathbf{w}_t)$: Mini-batch grad at step $t$
- $\eta$: Learning rate
- $\lambda$: Weight decay coeff
- $\sigma$: Noise std

&nbsp;

## Training Control
### Early Stopping
- **What**: Stop when val perf stops improving.
- **Why**: To prevent overfitting.
- **How**:
	1. Choose:
		- Monitored val metric.
		- **Patience**: How many epochs to wait for improvement.
		- **Min Delta**: Minimum threshold to qualify as "improvement".
	2. Train:
		- Every time improvement exceeds Min Delta, reset Patience counter.
		- If Patience runs out, quit & keep best checkpoint.

```{attention} Q&A
:class: dropdown
*Cons?*
- May stop too early → Hyperparam tuning, validation data/metric noise, etc.
```

&nbsp;

## Weight Penalties
### Weight Decay
- **What**: Multiplicative shrink of the weights toward 0 at every step.
- **Why**: The data loss can leave $\|\mathbf{w}\|$ unbounded or weakly constrained.
    - E.g., separable data under cross-entropy ($\|\mathbf{w}\|\to\infty$ keeps ⬇️loss), scale-invariant weights.
    - Plain nets: smaller weights → smoother function → ⬇️overfitting.
    - Weights feeding a normalization layer: $\|\mathbf{w}\|$ does not change the function → WD sets the [effective LR](#effective-learning-rate) instead.
- **How**: Two implementations. Same under SGD. Different under adaptive optimizers.
    - **Coupled (L2)**: Add $\lambda\mathbf{w}$ to the grad → the optimizer rescales it like any other grad.
    - **Decoupled**: Shrink $\mathbf{w}$ directly, outside the optimizer's rescaling → [AdamW](optim.md#adamw). {cite:p}`loshchilov2019decoupled`

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\mathbf{u}_t$: Optimizer's update direction, computed from loss grads only.
    - $\hat{\mathbf{m}}_t$: Bias-corrected 1st moment of the grads fed to Adam.
    - $\hat{\mathbf{v}}_t$: Bias-corrected 2nd moment of the grads fed to Adam.
    - $\epsilon$: Adam's stability constant.

Coupled (L2): penalty $\frac{\lambda}{2}\|\mathbf{w}\|_2^2$ → the optimizer sees $\tilde{\mathbf{g}}_t=\mathbf{g}_t+\lambda\mathbf{w}_t$.

$$
\text{SGD: }\ \mathbf{w}_{t+1}=\mathbf{w}_t-\eta(\mathbf{g}_t+\lambda\mathbf{w}_t)=(1-\eta\lambda)\mathbf{w}_t-\eta\mathbf{g}_t
$$

$$
\text{Adam: }\ \mathbf{w}_{t+1}=\mathbf{w}_t-\eta\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t}+\epsilon},\qquad\hat{\mathbf{m}}_t,\hat{\mathbf{v}}_t\text{ built from }\tilde{\mathbf{g}}
$$

- The decay term $\lambda w_j$ is also divided by $\sqrt{\hat v_{t,j}}+\epsilon$ → per-weight decay rate $\approx\frac{\eta\lambda}{\sqrt{\hat v_{t,j}}+\epsilon}$ (ignoring momentum).

Decoupled:

$$
\mathbf{w}_{t+1}=(1-\eta\lambda)\mathbf{w}_t-\eta\mathbf{u}_t
$$

- SGD ($\mathbf{u}_t=\mathbf{g}_t$): identical to coupled.
- Adam ($\mathbf{u}_t=\hat{\mathbf{m}}_t/(\sqrt{\hat{\mathbf{v}}_t}+\epsilon)$, moments of $\mathbf{g}$ only) = AdamW. Same decay rate $\eta\lambda$ for every weight.
```

````{important} Code
:class: dropdown
```python
import copy
import torch
import torch.nn as nn

class DecoupledWeightDecay:
    def __init__(self, opt, wd):
        ## opt must use weight_decay=0 -> its grads stay pure loss grads
        self.opt, self.wd = opt, wd

    @torch.no_grad()
    def step(self):
        for group in self.opt.param_groups:
            for p in group["params"]:
                ## Shrink OUTSIDE the adaptive rescaling -> same relative decay for every weight
                p.mul_(1 - group["lr"] * self.wd)
        self.opt.step()

## Example: Adam + decoupled decay == torch AdamW; Adam + coupled L2 differs
torch.manual_seed(0)
a = nn.Linear(4, 2)
b, c = copy.deepcopy(a), copy.deepcopy(a)
opt_a = DecoupledWeightDecay(torch.optim.Adam(a.parameters(), lr=0.1), wd=0.1)
opt_b = torch.optim.AdamW(b.parameters(), lr=0.1, weight_decay=0.1)
opt_c = torch.optim.Adam(c.parameters(), lr=0.1, weight_decay=0.1)  ## coupled L2
x = torch.randn(3, 4)
for m, opt in [(a, opt_a), (b, opt_b), (c, opt_c)]:
    m(x).pow(2).sum().backward()
    opt.step()
print(torch.allclose(a.weight, b.weight), torch.allclose(a.weight, c.weight))  ## True False
```
````

```{attention} Q&A
:class: dropdown
*L2 = weight decay?*
- Plain SGD: ✅ identical (penalty $\frac{\lambda}{2}\|\mathbf{w}\|^2$, same $\lambda$).
- SGD + momentum: ≈ only. The L2 term enters the momentum buffer → decay is smeared over later steps.
- Adam: ❌. Weights w/ large grad history (large $\hat v_j$) get the least decay. → [L2](../../ml/obj.md#l2), [AdamW](optim.md#adamw).

*Why does Adam + L2 generalize worse than AdamW?*
- Fast-moving weights (large grads) = least regularized ← decay divided by $\sqrt{\hat v_j}$.
- Decoupling → optimal $\lambda$ depends less on $\eta$. Adam becomes competitive w/ SGD + momentum on image classification.

*What does `weight_decay` mean in PyTorch?*
- `SGD`, `Adam`: coupled L2 (added to the grad).
- `AdamW`: decoupled. Shrink per step = `lr × weight_decay` → still scales w/ the LR & its schedule.
- Loshchilov & Hutter's original: shrink = schedule multiplier × $\lambda$, ❌× base LR → retuning the base LR does not change the decay.

*Which params to exclude?*
- Biases & normalization gains/shifts: standard practice. ← Few params, little capacity. Decaying a norm gain fights the normalizer.
- Embeddings: recipe-dependent.

*Typical values?*
- SGD + momentum on ImageNet CNNs: $10^{-4}$ to $5\times10^{-4}$.
- AdamW for LLM & ViT pretraining: $0.1$ (PyTorch convention: multiplied by LR).

*Why use WD in 1-epoch LLM pretraining, where overfitting is minimal?*
- Not classic capacity control: ~1 pass over the data → train loss ≈ test loss.
- Acts on optimization: bounds $\|\mathbf{w}\|$ → controls the [effective LR](#effective-learning-rate) & helps against [output logit divergence](issues.md#output-logit-divergence).
- ⚠️ The exact mechanism is still debated.
```

&nbsp;

#### Effective Learning Rate
- **What**: Rotation speed of a scale-invariant weight's direction: $\propto\eta/\|\mathbf{w}\|^2$ under SGD. {cite:p}`vanlaarhoven2017l2`
- **Why**: Normalization layers make the loss invariant to $\|\mathbf{w}\|$ → only the direction $\mathbf{w}/\|\mathbf{w}\|$ matters.
    - → WD cannot limit capacity there. It changes how fast the direction moves.
- **How**:
    1. Scale invariance → grad ⊥ $\mathbf{w}$ & grad shrinks as $\|\mathbf{w}\|$ grows.
    2. SGD w/o WD → every step adds an orthogonal vector → $\|\mathbf{w}\|$ grows → effective LR ⬇️ over time.
    3. WD shrinks $\|\mathbf{w}\|$ → effective LR ⬆️.
    4. Growth & shrink balance → equilibrium rotation speed set by $\eta\lambda$ only.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $c>0$: Arbitrary scale factor.
    - $\hat{\mathbf{w}}=\mathbf{w}/\|\mathbf{w}\|$: Weight direction.
    - $\Delta\theta_t$: Angle between $\mathbf{w}_t$ & $\mathbf{w}_{t+1}$.

Override: here $\mathbf{w}$ = only the scale-invariant weights (e.g., one layer that feeds a norm layer).

Definition: $\mathcal{L}$ is **scale-invariant** in $\mathbf{w}$ if

$$
\mathcal{L}(c\mathbf{w})=\mathcal{L}(\mathbf{w})\quad\forall c>0
$$

- Holds for weights directly before BatchNorm, LayerNorm, RMSNorm, or weight normalization (norm's $\epsilon$ ignored).

Properties:
- Orthogonality: $\langle\mathbf{w},\nabla\mathcal{L}(\mathbf{w})\rangle=0$.
- Inverse scaling: $\nabla\mathcal{L}(c\mathbf{w})=\frac{1}{c}\nabla\mathcal{L}(\mathbf{w})$.
- SGD w/o WD → norm grows:

    $$
    \|\mathbf{w}_{t+1}\|^2=\|\mathbf{w}_t\|^2+\eta^2\|\mathbf{g}_t\|^2
    $$

- Angular step:

    $$
    \Delta\theta_t\approx\frac{\eta\|\mathbf{g}_t\|}{\|\mathbf{w}_t\|}=\frac{\eta}{\|\mathbf{w}_t\|^2}\left\|\nabla\mathcal{L}(\hat{\mathbf{w}}_t)\right\|\quad\Rightarrow\quad\eta_\text{eff}=\frac{\eta}{\|\mathbf{w}_t\|^2}
    $$

- SGD + decoupled WD, equilibrium (small $\eta\lambda$):

    $$
    \|\mathbf{w}\|^2\approx\frac{\eta\|\mathbf{g}\|^2}{2\lambda},\qquad\Delta\theta\approx\sqrt{2\eta\lambda}
    $$

- Adam: update size ≈ independent of grad scale → $\Delta\theta\propto\eta/\|\mathbf{w}\|$ → the norm still matters, only less.
```

```{tip} Derivation
:class: dropdown
*Why is the grad ⊥ $\mathbf{w}$?*
1. $\mathcal{L}(c\mathbf{w})$ is constant in $c$.
2. $\frac{d}{dc}\mathcal{L}(c\mathbf{w})\big|_{c=1}=\langle\nabla\mathcal{L}(\mathbf{w}),\mathbf{w}\rangle=0$.

*Why does the grad scale as $1/c$?*
1. Take $\nabla_\mathbf{w}$ of both sides of $\mathcal{L}(c\mathbf{w})=\mathcal{L}(\mathbf{w})$.
2. Chain rule: $c\,\nabla\mathcal{L}(c\mathbf{w})=\nabla\mathcal{L}(\mathbf{w})$.
3. → $\mathbf{g}_t=\nabla\mathcal{L}(\hat{\mathbf{w}}_t)/\|\mathbf{w}_t\|$.

*Why is the equilibrium angle $\sqrt{2\eta\lambda}$?*
1. Update: $\mathbf{w}_{t+1}=(1-\eta\lambda)\mathbf{w}_t-\eta\mathbf{g}_t$, w/ $\mathbf{g}_t\perp\mathbf{w}_t$.
2. Pythagoras: $\|\mathbf{w}_{t+1}\|^2=(1-\eta\lambda)^2\|\mathbf{w}_t\|^2+\eta^2\|\mathbf{g}_t\|^2$.
3. Steady state $\|\mathbf{w}_{t+1}\|=\|\mathbf{w}_t\|$ → $(2\eta\lambda-\eta^2\lambda^2)\|\mathbf{w}\|^2=\eta^2\|\mathbf{g}\|^2$.
4. Drop $\eta^2\lambda^2$ → $\|\mathbf{w}\|^2\approx\eta\|\mathbf{g}\|^2/(2\lambda)$.
5. $\Delta\theta\approx\eta\|\mathbf{g}\|/\|\mathbf{w}\|=\sqrt{2\eta\lambda}$.
- ⚠️ Assumes plain SGD, slowly varying $\|\mathbf{g}\|$, small $\eta\lambda$. Momentum changes the constant.
```

```{note} Example
:class: dropdown
- $\|\mathbf{w}\|$: 1 → 2. Grad halves. Step relative to $\|\mathbf{w}\|$: ×¼. → Same as a 4× LR cut.
- $\eta=0.1$, $\lambda=5\times10^{-4}$ → equilibrium $\Delta\theta\approx\sqrt{10^{-4}}=0.01$ rad per step.
```

```{attention} Q&A
:class: dropdown
*Does WD regularize a network w/ BatchNorm?*
- Not on scale-invariant weights. The function ignores $\|\mathbf{w}\|$ → no capacity limit.
- Effect = effective LR ⬆️ → more grad noise → implicit regularization.
- Empirically, three mechanisms, by setup: {cite:p}`zhang2019three`
    1. Effective LR ⬆️ (SGD/Adam + BN). Almost all of WD's gain came from layers w/ BN.
    2. ≈ input-output Jacobian norm penalty (K-FAC w/ Gauss-Newton metric, ❌BN; exact for linear nets w/ whitened inputs).
    3. Effective damping ⬇️ (K-FAC).

*What happens w/o WD in a normalized net?*
- $\|\mathbf{w}\|$ grows fast early → effective LR collapses → acts like an unplanned extra LR decay.

*Which weights are NOT scale-invariant?*
- Final classifier layer, layers w/o a following norm, the norm gains themselves → classic WD effect there.

*How do LR & WD interact for normalized layers?*
- At equilibrium the rotation speed depends on the product $\eta\lambda$ → LR & WD act as one knob.
- → Changing one requires retuning the other.
```

&nbsp;

## Stochastic Regularization
- **What**: Inject randomness during training. Remove it at test. E.g., [Dropout](../foundation/module.md#dropout).

### Stochastic Depth
- **What**: Skip whole residual blocks at random during training. Keep all blocks at test. {cite:p}`huang2016deep`
- **Why**: Very deep nets → vanishing grads, diminishing feature reuse, slow training.
    - Train a short net (fast, strong grads). Test the deep net (full capacity).
- **How**:
    1. Assign a survival prob $p_\ell$ to each block. Linear decay: early blocks survive more ← later blocks need their low-level features.
    2. Each step: flip a coin per block. Dropped block → output = skip connection only (identity).
    3. Test: all blocks on. Scale each branch by $p_\ell$, or equivalently scale by $1/p_\ell$ at train time.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $f_\ell$: Residual branch of block $\ell$.
- Hyperparams:
    - $L$: #residual blocks.
    - $p_L$: Survival prob of the last block. Default $0.5$.
- Misc:
    - $\mathbf{h}_\ell$: Output of block $\ell$. $\mathbf{h}_0$ = input of the first block.
    - $b_\ell\sim\text{Bernoulli}(p_\ell)$: Survival indicator of block $\ell$.

Survival prob (linear decay):

$$
p_\ell=1-\frac{\ell}{L}(1-p_L),\qquad\ell=1,\dots,L
$$

Training:

$$
\mathbf{h}_\ell=\mathbf{h}_{\ell-1}+b_\ell\,f_\ell(\mathbf{h}_{\ell-1})
$$

Test:

$$
\mathbf{h}_\ell=\mathbf{h}_{\ell-1}+p_\ell\,f_\ell(\mathbf{h}_{\ell-1})
$$

- Original (post-activation ResNet): ReLU after the add.
- Inverted form (DropPath): train w/ $\frac{b_\ell}{p_\ell}f_\ell$, test w/ $f_\ell$. Same expectation.

Expected #active blocks:

$$
\mathbb{E}[\tilde L]=\sum_{\ell=1}^{L}p_\ell=L-\frac{(1-p_L)(L+1)}{2}\ \xrightarrow{p_L=0.5}\ \frac{3L-1}{4}
$$
```

```{tip} Derivation
:class: dropdown
*Why is $\mathbb{E}[\tilde L]=(3L-1)/4$ for $p_L=0.5$?*
1. $\sum_{\ell=1}^{L}p_\ell=L-\frac{1-p_L}{L}\sum_{\ell=1}^{L}\ell=L-\frac{1-p_L}{L}\cdot\frac{L(L+1)}{2}=L-\frac{(1-p_L)(L+1)}{2}$.
2. $p_L=0.5$ → $L-\frac{L+1}{4}=\frac{3L-1}{4}$ → ≈ ¾ of the blocks run per step → ≈25% less training compute.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class StochasticDepth(nn.Module):
    def __init__(self, branch, p_survive):
        super().__init__()
        self.branch, self.p = branch, p_survive

    def forward(self, h):
        if not self.training:
            return h + self.branch(h)  ## all blocks active at test
        ## One coin per SAMPLE (DropPath style); shape (B, 1, ..., 1) broadcasts over features
        b = torch.rand(h.shape[0], *([1] * (h.dim() - 1)), device=h.device) < self.p
        ## Inverted scaling -> E[train output] = test output
        return h + b * self.branch(h) / self.p

def linear_decay(L, p_L=0.5):
    ## p_l = 1 - (l / L)(1 - p_L), l = 1..L -> early blocks survive more often
    return [1 - l / L * (1 - p_L) for l in range(1, L + 1)]

## Example
ps = linear_decay(4)
blocks = nn.Sequential(*[StochasticDepth(nn.Linear(8, 8), p) for p in ps])
h = torch.randn(2, 8)
print(ps, blocks(h).shape)  ## [0.875, 0.75, 0.625, 0.5] torch.Size([2, 8])
```
````

```{attention} Q&A
:class: dropdown
*Why does it need residual connections?*
- A dropped block must leave a valid path → the identity skip.
- Plain net: dropping a layer breaks the chain of transformations.

*Why does it work?*
- Shorter expected depth → shorter grad chains → stronger grads to early layers.
- Implicit ensemble: each step trains 1 of $2^L$ sub-nets of different depths. Test ≈ average of all sub-nets.
    - ❌Exact: blocks are nonlinear → $\mathbb{E}[f_2(\mathbf{h}_1)]\neq f_2(\mathbb{E}[\mathbf{h}_1])$.
    - Exact only per branch: for a fixed input, $\mathbb{E}[b_\ell f_\ell]=p_\ell f_\ell$.

*Pros?*
- ⬇️Training time: ≈25% w/ linear decay & $p_L=0.5$ ← dropped blocks skip forward & backward.
- ⬇️Test error. Made a 1202-layer ResNet trainable: 4.91% error on CIFAR-10.

*Cons?*
- Extra hyperparams ($p_L$, decay rule).
- Per-sample dropping (DropPath) saves ❌compute: the branch still runs for the batch.

*Stochastic Depth vs DropPath?*
- Same idea in modern code: timm's `DropPath` = per-sample stochastic depth.
- Name from FractalNet's **drop-path**: drops paths inside fractal join layers, ❌residual blocks. {cite:p}`larsson2017fractalnet`
- Granularity differs. Original: 1 coin per block per mini-batch. timm: 1 coin per sample. torchvision `StochasticDepth`: `mode="batch"` or `"row"`.

*Stochastic Depth vs Dropout?*
- [Dropout](../foundation/module.md#dropout) zeros single units. Stochastic depth zeros a whole residual branch → structured noise.
- Both: Bernoulli mask + rescale for train/test consistency.

*Where is it used today?*
- Standard in ViT-family & modern CNN recipes (DeiT, Swin, ConvNeXt) as a "drop path rate" hyperparam.
```

&nbsp;

### Noise Injection
- **What**: Random perturbation of inputs, weights, or grads during training only.
- **Why**: Robustness.
    - The fit must hold in a neighborhood of each training point → smoother function.
- **How**: Each step: sample fresh noise → perturb → normal forward & backward. ❌Noise at test.

```{dropdown} Table: Where to Inject Noise
| Where | Perturbation | Implicit effect (small noise) | Related |
|:--|:--|:--|:--|
| Inputs | $\mathbf{x}+\boldsymbol{\xi}$ | Penalize input sensitivity $\Vert\nabla_\mathbf{x}f\Vert^2$ | [Data augmentation](../../cv/image.md#data-augmentation), [Denoising AE](../gen/vae.md#denoising-autoencoder) |
| Hidden units | Bernoulli mask, rescaled | Discourage co-adaptation of units | [Dropout](../foundation/module.md#dropout) |
| Residual branches | Bernoulli skip | Implicit depth ensemble | [Stochastic Depth](#stochastic-depth) |
| Weights | $\mathbf{w}+\boldsymbol{\varepsilon}$ | Penalize curvature $\text{tr}\,\nabla^2\mathcal{L}$ | [SAM](optim.md#sam) (worst case) |
| Grads | $\mathbf{g}_t+\boldsymbol{\zeta}_t$ | Exploration, ❌fixed penalty | Langevin dynamics |

- $f$: Network output.
```

```{attention} Q&A
:class: dropdown
*Noise vs an explicit penalty?*
- Small noise: same objective to 2nd order ([Input Noise](#input-noise), [Weight Noise](#weight-noise)).
- Noise: ❌extra derivatives. Cost = 1 random sample per step. BUT noisier grads.
- Explicit penalty: deterministic. BUT needs Jacobians/Hessians.

*Why ❌noise at test?*
- Noise is a training-time regularizer. Test wants 1 deterministic prediction.
- Noise-free forward ≈ mean prediction over noise. Exact only for linear models.

*Random noise vs adversarial perturbation?*
- Random noise = average case. Most random directions barely change the loss in high dim.
- Adversarial training & SAM = worst case → stronger per sample, ⬆️cost.
```

&nbsp;

#### Input Noise
- **What**: Fresh Gaussian noise added to each training input.
- **Why**: Finite data → the fit is unconstrained between samples.
    - Noise spreads each sample into a small cloud → the model must be flat around it.
- **How**: $\tilde{\mathbf{x}}=\mathbf{x}+\boldsymbol{\xi}$, new $\boldsymbol{\xi}$ every step → train on $\tilde{\mathbf{x}}$ w/ the clean target $y$.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{x}\in\mathbb{R}^n$: Input.
    - $y$: Scalar target.
- Misc:
    - $f(\mathbf{x})$: Scalar network output.
    - $\boldsymbol{\xi}\sim\mathcal{N}(\mathbf{0},\sigma^2I)$: Input noise.
    - $\tilde{\mathcal{L}}$: Expected loss under input noise.

Objective (MSE):

$$
\tilde{\mathcal{L}}=\frac{1}{2}\mathbb{E}_{\mathbf{x},y,\boldsymbol{\xi}}\left[\left(f(\mathbf{x}+\boldsymbol{\xi})-y\right)^2\right]
$$

Small $\sigma$, smooth $f$ (full 2nd-order expansion):

$$
\tilde{\mathcal{L}}=\mathcal{L}+\frac{\sigma^2}{2}\,\mathbb{E}_{\mathbf{x},y}\left[\|\nabla_\mathbf{x}f(\mathbf{x})\|^2+\left(f(\mathbf{x})-y\right)\text{tr}\,\nabla_\mathbf{x}^2f(\mathbf{x})\right]+O(\sigma^4)
$$

- $\mathcal{L}=\frac{1}{2}\mathbb{E}[(f(\mathbf{x})-y)^2]$: Clean loss.

Tikhonov reduction (Bishop): if the model is flexible enough that its minimizer satisfies $f(\mathbf{x})=\mathbb{E}[y|\mathbf{x}]+O(\sigma^2)$:

$$
\tilde{\mathcal{L}}\approx\mathcal{L}+\frac{\sigma^2}{2}\,\mathbb{E}_\mathbf{x}\left[\|\nabla_\mathbf{x}f(\mathbf{x})\|^2\right]
$$

- 2nd term = Tikhonov regularizer on the input gradient. Vector output → $\|J_\mathbf{x}f\|_F^2$.
- Holds near that minimizer only. A restricted model or a point far from it keeps the Hessian term.

Linear model $f=\mathbf{w}^\top\mathbf{x}$ (exact, any $\sigma$):

$$
\tilde{\mathcal{L}}=\mathcal{L}+\frac{\sigma^2}{2}\|\mathbf{w}\|^2
$$

- → Ridge w/ coeff $\sigma^2/2$.
```

```{tip} Derivation
:class: dropdown
*Why is training w/ input noise ≈ Tikhonov regularization?* {cite:p}`bishop1995training`
1. Taylor (smooth $f$): $f(\mathbf{x}+\boldsymbol{\xi})=f+\boldsymbol{\xi}^\top\nabla f+\frac{1}{2}\boldsymbol{\xi}^\top H\boldsymbol{\xi}+O(\|\boldsymbol{\xi}\|^3)$.
    - $H=\nabla^2_\mathbf{x}f$: Input Hessian.
2. Square the residual. Let $r=f(\mathbf{x})-y$:

    $$
    (f(\mathbf{x}+\boldsymbol{\xi})-y)^2=r^2+2r\,\boldsymbol{\xi}^\top\nabla f+(\boldsymbol{\xi}^\top\nabla f)^2+r\,\boldsymbol{\xi}^\top H\boldsymbol{\xi}+O(\|\boldsymbol{\xi}\|^3)
    $$

3. Average over $\boldsymbol{\xi}$: $\mathbb{E}[\boldsymbol{\xi}]=\mathbf{0}$, $\mathbb{E}[\boldsymbol{\xi}\boldsymbol{\xi}^\top]=\sigma^2I$, odd moments vanish:

    $$
    \tilde{\mathcal{L}}=\mathcal{L}+\frac{\sigma^2}{2}\mathbb{E}\left[\|\nabla f\|^2+r\,\text{tr}H\right]+O(\sigma^4)
    $$

4. The $r\,\text{tr}H$ term is not bounded below → unusable as a penalty on its own.
5. Flexible model → minimizer of $\tilde{\mathcal{L}}$: $f(\mathbf{x})=\mathbb{E}[y|\mathbf{x}]+O(\sigma^2)$ → $\mathbb{E}[r|\mathbf{x}]=O(\sigma^2)$ → that term is $O(\sigma^4)$.
6. → Only $\frac{\sigma^2}{2}\mathbb{E}\|\nabla_\mathbf{x}f\|^2$ remains: positive semi-definite, first derivatives only.
- Linear $f$: $H=0$, $\nabla f=\mathbf{w}$ → exact ridge.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class GaussianNoise(nn.Module):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        if not self.training:
            return x  ## noise only in training -> deterministic inference
        return x + self.sigma * torch.randn_like(x)  ## fresh noise on every call

## Example: linear model -> noisy MSE = clean MSE + sigma^2 * ||w||^2 (unhalved MSE)
torch.manual_seed(0)
w, x, y = torch.tensor([1.0, -2.0]), torch.tensor([0.5, 0.5]), torch.tensor(0.0)
xs = GaussianNoise(0.1).train()(x.repeat(100_000, 1))
noisy = ((xs @ w - y) ** 2).mean().item()
clean = ((x @ w - y) ** 2).item()
print(round(noisy, 2), round(clean + 0.1**2 * w.pow(2).sum().item(), 2))  ## 0.3 0.3
```
````

```{attention} Q&A
:class: dropdown
*Assumptions?*
- MSE loss & small $\sigma$. The $O(\sigma^4)$ terms matter for large $\sigma$.
- Tikhonov form only: flexible model, evaluated near its minimizer. Otherwise the Hessian term stays.
- Isotropic noise → equal smoothness in every input direction. Feature scales must be comparable → standardize inputs first.

*Relation to data augmentation?*
- Gaussian noise = the simplest label-preserving augmentation.
- Structured augmentations (crop, flip, color) encode task-specific invariances. Isotropic noise encodes only local smoothness.

*Relation to denoising autoencoders?*
- Same corruption $\tilde{\mathbf{x}}=\mathbf{x}+\boldsymbol{\xi}$. Target = clean $\mathbf{x}$ instead of $y$ → learns to map noisy inputs back to the data.
```

&nbsp;

#### Weight Noise
- **What**: Fresh Gaussian noise added to the weights at each forward pass.
- **Why**: Prefer flat minima.
    - Loss averaged over a weight cloud is low only if the loss is low in the whole neighborhood.
- **How**: $\tilde{\mathbf{w}}=\mathbf{w}+\boldsymbol{\varepsilon}$ → forward & backward at $\tilde{\mathbf{w}}$ → apply that grad to $\mathbf{w}$.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\boldsymbol{\varepsilon}\sim\mathcal{N}(\mathbf{0},\sigma^2I)$: Weight noise.
    - $H_\mathbf{w}=\nabla^2_\mathbf{w}\mathcal{L}(\mathbf{w})$: Loss Hessian.

Objective (small $\sigma$; $\mathcal{L}$ 4× differentiable w/ bounded 4th derivatives):

$$
\mathbb{E}_{\boldsymbol{\varepsilon}}\left[\mathcal{L}(\mathbf{w}+\boldsymbol{\varepsilon})\right]=\mathcal{L}(\mathbf{w})+\frac{\sigma^2}{2}\,\text{tr}\,H_\mathbf{w}+O(\sigma^4)
$$

- $\text{tr}\,H_\mathbf{w}$ = sum of curvatures → penalizes sharp minima.
- Derivation: Taylor-expand $\mathcal{L}$ in $\boldsymbol{\varepsilon}$. The linear & cubic terms average to 0 (symmetric noise).
- Only 2× differentiable → remainder $o(\sigma^2)$, ❌$O(\sigma^4)$.
```

```{attention} Q&A
:class: dropdown
*Weight noise vs SAM?*
- Weight noise: average loss over random perturbations.
- [SAM](optim.md#sam): worst-case loss in a ball → targets the sharpest direction.

*Weight noise vs Dropout?*
- Dropout = multiplicative Bernoulli noise on activations. Dropping weights instead of units = DropConnect.
- Gaussian weight noise = additive & dense.

*Cons?*
- $\|\boldsymbol{\varepsilon}\|\approx\sigma\sqrt{\#\text{params}}$ → 1 global $\sigma$ hits small & large layers alike → scale $\sigma$ per layer.
- Noisier grads → slower convergence.

*Do flat minima generalize better?*
- Often observed. ⚠️ Contested: reparameterization can make a minimum sharp w/o changing the function.

*What if each weight learns its own $\sigma$?*
- → Variational Bayesian NN: learn a Gaussian posterior per weight, trained w/ the ELBO.
```

&nbsp;

#### Gradient Noise
- **What**: Annealed Gaussian noise added to every grad. {cite:p}`neelakantan2015adding`
- **Why**: Hard-to-optimize nets (very deep, memory-augmented) stall from poor init.
    - Early noise pushes params out of flat/poor regions. Late noise vanishes → normal convergence.
- **How**:
    1. Backward → grad $\mathbf{g}_t$.
    2. Add $\mathcal{N}(\mathbf{0},\sigma_t^2I)$ w/ decaying $\sigma_t$.
    3. Pass the noisy grad to any optimizer (SGD, Adam, ...).

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $c$: Initial noise variance. Paper: $c\in\{0.01,0.3,1.0\}$.
    - $\gamma$: Decay exponent. Paper: $0.55$.

Process:

$$
\tilde{\mathbf{g}}_t=\mathbf{g}_t+\mathcal{N}(\mathbf{0},\sigma_t^2I),\qquad\sigma_t^2=\frac{c}{(1+t)^\gamma}
$$
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class GradientNoise:
    def __init__(self, params, c=0.01, gamma=0.55):
        self.params, self.c, self.gamma, self.t = list(params), c, gamma, 0

    @torch.no_grad()
    def __call__(self):
        ## Annealed std: large early (explore), -> 0 later (converge)
        std = (self.c / (1 + self.t) ** self.gamma) ** 0.5
        for p in self.params:
            if p.grad is not None:
                p.grad.add_(torch.randn_like(p.grad) * std)
        self.t += 1

## Example
model = nn.Linear(4, 2)
opt = torch.optim.SGD(model.parameters(), lr=0.1)
noise = GradientNoise(model.parameters(), c=0.3)
model(torch.randn(3, 4)).sum().backward()
noise()     ## perturb AFTER backward, BEFORE step
opt.step()
print(noise.t)  ## 1
```
````

```{attention} Q&A
:class: dropdown
*Results?*
- Trained a 20-layer fully connected net w/ plain SGD from a poor init.
- 72% relative error reduction on a question-answering task.
- Doubled the #accurate binary-multiplication models over 7,000 random restarts.

*SGD is already noisy. Why add more?*
- Mini-batch noise: anisotropic (shaped by the grad covariance) & shrinks as batch size ⬆️.
- Injected noise: isotropic & on an explicit schedule.

*Relation to Langevin dynamics (SGLD)?*
- SGLD ties the noise variance to the LR → iterates sample a posterior.
- Gradient noise uses a free schedule → ❌sampler. Pure optimization aid.

*Cons?*
- Rare in modern recipes ← residuals, normalization, good init & Adam already fix most poor-init stalls.
- Under Adam the noise also inflates $\hat{\mathbf{v}}_t$ → smaller steps on low-grad coordinates.
```

&nbsp;
