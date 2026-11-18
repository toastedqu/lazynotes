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
# Training Toolkit
This page covers training tricks commonly used in all kinds of DL tasks.

This page does NOT cover specialized training methods like transfer learning or post-training. They are discussed separately.

Notations:
- $\theta$: All trainable params (flattened)
- $\theta_t$: Params after optimizer step $t$
- $\eta$: LR
- $B$: Batch size (#samples per optimizer step)
- $\mathcal{L}$: Training loss (expectation over data)
- $\mathbf{g}=\nabla_\theta\mathcal{L}$: True (full-data) grad

Overrides of global notation: $B$ = scalar batch size, ❌upper bound.

&nbsp;

## Weight Initialization
### Zero Initialization
- **What**: All params are 0.
- **Why**: People thought it was simple and unbiased back in the days.

```{attention} Q&A
:class: dropdown
*Pros?*
- Simple, unbiased, good performance with small models.

*Cons?*
- **Failure to break symmetry**: All weights produce the same output and receive the same gradients → No learning.
- Vanishing gradients.
- No non-linearity.
- Dead neurons if activated by the ReLU family.
```

&nbsp;

### Random Initialization
- **What**: All params are random.
- **Why**: To **break symmetry** → Neurons learn diff features.

```{attention} Q&A
:class: dropdown
*Pros?*
- Simple, cheap & breaks symmetry.

*Cons?*
- Require careful selection of scale/distribution.
- Slower convergence.
- Dead neurons with ReLU are still likely.
- Run-to-run variability.
```

&nbsp;

### He Initialization
- **What**: All params in ReLU nets are constrained to a fixed variance.
- **Why**: To mitigate [vanishing/exploding gradients](issues.md) in very deep NNs with ReLU activations.

```{note} Math
:class: dropdown
- Distributions:
	\begin{align*}
	&\text{Normal:}  &&W\sim N(0,\sigma^2),\quad\sigma=\sqrt{\frac{2}{\text{fan\_in}}} \\
	&\text{Uniform:} &&W\sim U(-a,a),\quad a=\sqrt{\frac{6}{\text{fan\_in}}}
	\end{align*}
	- $\text{Var}[U(-a,a)]=\frac{a^2}{3}\rightarrow\frac{a^2}{3}=\frac{2}{\text{fan\_in}}$
- Modes:
	- `fan_in`: Preserves forward activation scale (most common) → Mitigates vanishing/exploding activations.
	- `fan_out`: Preserves backprop gradient scale → Mitigates gradient issues not caused by activation issues.
```

```{attention} Q&A
:class: dropdown
*Pros?*
- Training stability & convergence ⬆️.

*Cons?*
- NOT universally optimal.
- The "right" variant depends on activation, direction, and layer type.
```

&nbsp;

### Xavier/Glorot Initialization
- **What**: "He Init" but preserve variance in **both ways**.
- **Why**: To mitigate [vanishing/exploding gradients](issues.md) in very deep NNs with ReLU activations.

```{note} Math
:class: dropdown
- Distributions:
	\begin{align*}
	&\text{Normal:}  &&W\sim N(0,\sigma^2),\quad\sigma=\sqrt{\frac{2}{n+m}} \\
	&\text{Uniform:} &&W\sim U(-a,a),\quad a=\sqrt{\frac{6}{n+m}}
	\end{align*}
	- $n=\text{fan\_in},\quad m=\text{fan\_out}$.
- Modes:
	- `fan_in`: Preserves forward activation scale (most common) → Mitigates vanishing/exploding activations.
	- `fan_out`: Preserves backprop gradient scale → Mitigates gradient issues not caused by activation issues.
```

```{attention} Q&A
:class: dropdown
*Pros?*
- ✅ for most non-ReLU activations.

*Cons?*
- ❌ for ReLU activations.
- Too many assumptions (independence, no normalization, etc.).
```

&nbsp;

### Orthogonal Initialization
- **What**: Random (semi-)orthogonal weight matrix × gain. {cite:p}`saxe2014exact`
- **Why**: Norm preservation ≠ direction preservation.
    - Gaussian init (He/Xavier) preserves the norm of a typical vector only on average.
    - Product of many Gaussian matrices → singular values spread out w/ depth: most shrink, a few grow → grads vanish in most directions. → [Vanishing/Exploding Gradient](issues.md#vanishing-exploding-gradient)
    - Orthogonal matrix → all singular values = 1 → product of orthogonal matrices is orthogonal → every direction survives at any depth.
- **How**:
    1. Sample a matrix w/ i.i.d. $\mathcal{N}(0,1)$ entries (tall: transpose if $W$ is wide).
    2. QR-decompose it → $Q$ w/ orthonormal columns.
    3. Flip the sign of each column of $Q$ to match the diagonal of $R$ → $Q$ uniform over orthogonal matrices.
    4. Scale by a gain chosen per activation.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $W\in\mathbb{R}^{n_\text{out}\times n_\text{in}}$: Weight matrix.
- Hyperparams:
    - $\gamma$: Gain.
- Misc:
    - $n_\text{in}$: fan_in.
    - $n_\text{out}$: fan_out.

Definition:

$$
W=\gamma Q,\qquad\begin{cases}Q^TQ=I_{n_\text{in}}&n_\text{out}\ge n_\text{in}\ \text{(orthonormal columns)}\\QQ^T=I_{n_\text{out}}&n_\text{out}<n_\text{in}\ \text{(orthonormal rows)}\end{cases}
$$

- All $\min(n_\text{in},n_\text{out})$ singular values of $W$ = $\gamma$.
- Square $Q$ ($n_\text{in}=n_\text{out}=n$): each column has unit norm → $\mathbb{E}[Q_{ij}^2]=1/n$ → same per-entry variance as $\mathcal{N}(0,\gamma^2/n)$.
    - → Same scale as He ($\gamma=\sqrt2$) or Xavier ($\gamma=1$). Only the spectrum differs.
- **Dynamical isometry**: all singular values of the input-output Jacobian $J=\partial\mathbf{h}_L/\partial\mathbf{h}_0$ concentrate near 1.
    - $\mathbf{h}_l$: Activation vector of layer $l$.
    - $L$: #layers.
- Deep linear net, square layers, $\gamma=1$: $J=W_L\cdots W_1$ is orthogonal → $\|J\mathbf{v}\|=\|\mathbf{v}\|$ for every $\mathbf{v}$ → exact dynamical isometry.
```

````{important} Code
:class: dropdown
```python
import torch

class OrthogonalInit:
    def __init__(self, gain=1.0):
        self.gain = gain

    @torch.no_grad()
    def __call__(self, W):
        out_f, in_f = W.shape
        ## QR wants a tall matrix -> build tall, transpose back if W is wide
        A = torch.randn(max(out_f, in_f), min(out_f, in_f))
        Q, R = torch.linalg.qr(A)                       ## Q: orthonormal columns
        Q = Q * torch.sign(torch.diagonal(R))           ## sign fix -> Haar-uniform, not biased by the QR sign convention
        if out_f < in_f:
            Q = Q.T                                     ## wide W -> orthonormal rows
        W.copy_(self.gain * Q)
        return W

## Example
W = OrthogonalInit()(torch.empty(3, 5))
print(torch.allclose(W @ W.T, torch.eye(3), atol=1e-6))   ## True

## 100-layer deep linear net: the norm survives exactly (Gaussian init: only on average)
x0 = torch.randn(4)
x = x0
for _ in range(100):
    x = OrthogonalInit()(torch.empty(4, 4)) @ x
print(x.norm() / x0.norm())                               ## ≈ 1.0, float error only
```
````

```{attention} Q&A
:class: dropdown
*Orthogonal vs He/Xavier?*
- Same per-entry scale for a matching gain.
- Diff = spectrum. Gaussian → singular values spread out (Marchenko-Pastur). The spread of the product grows w/ depth. Orthogonal → all equal.
- Gap: small at ~6 layers. Grows w/ depth.

*Which gain $\gamma$?*
- Linear: 1.
- tanh: $\gamma\approx\gamma_c=1$ → **edge of chaos**. $\gamma<1$ → activity decays w/ depth. $\gamma>1$ → chaotic propagation.
- ReLU: $\sqrt2$ ← ReLU zeroes ~half the units → halves the 2nd moment.
- PyTorch `calculate_gain`: tanh → $5/3$. ReLU → $\sqrt2$.

*Dynamical isometry in nonlinear nets?*
- Sigmoidal nets (e.g., tanh): ✅ approximately, but only w/ orthogonal init.
- ReLU nets: ❌ for any init. {cite:p}`pennington2017resurrecting`

*Where is it used?*
- RNN hidden-to-hidden weights: same matrix applied at every time step → orthogonal → eigenvalues on the unit circle → ⬇️vanishing/exploding over time, at init. → [RNN](../foundation/module.md#rnn)
- Deep RL policy & value nets: common default.

*Does orthogonality persist during training?*
- ❌. GD updates break it.
- To keep it: explicit constraint, or a penalty such as $\|W^TW-I\|_F^2$.

*Cons?*
- QR costs $O(n^3)$ per matrix. 1-time cost → negligible.
- Conv: orthogonal flattened kernel (out-channels × in-channels·kernel-area) ≠ orthogonal conv operator.
- Smaller gain in modern nets ← residual connections & normalization already stabilize signal propagation.
```

&nbsp;

## Gradient Tricks
### Gradient Clipping
- **What**: Cap grads by value/norm.
- **Why**: To mitigate exploding grads.

```{attention} Q&A
:class: dropdown
*Cons?*
- May slow learning & worsen performance if threshold too small.
```

&nbsp;

## Weight Averaging
- **What**: Average params along the training trajectory → use the average for eval.

&nbsp;

### EMA
- **Name**: Exponential Moving Average (of weights)
- **What**: Shadow copy of params, updated as an exponentially decaying average of the training params.
- **Why**: Late iterates bounce around a minimum ← grad noise.
    - Average → noise partly cancels → better eval point than any single iterate, w/o extra grads.
    - Slow & stable copy → consistent target/teacher for self-distillation.
- **How**:
    1. Init the shadow copy = training params.
    2. After every optimizer step: move the shadow a small fraction toward the curr params.
    3. Train w/ $\theta$. Eval/deploy w/ the shadow.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\bar\theta_t$: EMA params after step $t$ (❌grads).
- Hyperparams:
    - $\beta\in[0,1)$: Decay.

Process:

$$
\bar\theta_0=\theta_0,\qquad\bar\theta_t=\beta\,\bar\theta_{t-1}+(1-\beta)\,\theta_t
$$

Unrolled:

$$
\bar\theta_t=(1-\beta)\sum_{s=1}^{t}\beta^{t-s}\theta_s+\beta^t\theta_0
$$

- Weights sum to 1. Weight on random init $\theta_0$ = $\beta^t$ → decays to 0.
- Effective window ≈ $1/(1-\beta)$ steps. $\beta=0.999$ → ~1k steps. $\beta=0.9999$ → ~10k steps.
- Half-life $=\ln2/(-\ln\beta)\approx\ln2/(1-\beta)$ steps.
- Polyak (uniform) averaging $\bar\theta_t=\frac{1}{t}\sum_{s=1}^{t}\theta_s$ = same recursion w/ step-dependent $\beta_t=1-1/t$.
```

```{tip} Derivation
:class: dropdown
*How much noise does EMA remove?*
1. Model: near a minimum $\theta^*$, $\theta_s=\theta^*+\boldsymbol{\xi}_s$. $\boldsymbol{\xi}_s$: zero-mean noise, per-coord variance $\sigma^2$.
2. Assume $\boldsymbol{\xi}_s$ uncorrelated across steps. Let $t\to\infty$ → $\theta_0$ term vanishes.
3. Per-coord variance:

    $$
    \text{Var}[\bar\theta_t]=(1-\beta)^2\sum_{k\ge0}\beta^{2k}\sigma^2=\frac{(1-\beta)^2}{1-\beta^2}\sigma^2=\frac{1-\beta}{1+\beta}\sigma^2
    $$

4. $\beta=0.999$ → variance × ~$5\times10^{-4}$. Mean stays $\theta^*$.
5. Real SGD iterates are strongly correlated across steps → real reduction is much smaller.
```

````{important} Code
:class: dropdown
```python
import copy
import torch
import torch.nn as nn

class EMA:
    def __init__(self, model, beta=0.999):
        self.beta = beta
        self.shadow = copy.deepcopy(model).eval()       ## theta_bar_0 = theta_0
        for p in self.shadow.parameters():
            p.requires_grad_(False)                      ## never trained by backprop

    @torch.no_grad()
    def update(self, model):
        for pe, p in zip(self.shadow.parameters(), model.parameters()):
            ## theta_bar <- beta * theta_bar + (1 - beta) * theta
            pe.mul_(self.beta).add_(p, alpha=1 - self.beta)
        for be, b in zip(self.shadow.buffers(), model.buffers()):
            be.copy_(b)                                  ## BN running stats: copy, not average (simple choice)

## Example
model = nn.Linear(4, 2)
ema = EMA(model, beta=0.9)
opt = torch.optim.SGD(model.parameters(), lr=0.1)
for _ in range(3):
    loss = model(torch.randn(8, 4)).pow(2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    ema.update(model)                                    ## once per optimizer step
print(ema.shadow(torch.randn(2, 4)).shape)               ## torch.Size([2, 2]); eval w/ the EMA copy
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- Cheap: +1 param copy, ❌extra fwd/bwd.
- Smoother eval curve. Often a better final model at the same training cost.
- Eval-only EMA → read-only → training trajectory unchanged.

*Cons?*
- +1 param copy of memory (usually fp32).
- Extra HP $\beta$. Too large → shadow lags far behind → stale weights. Too small → little smoothing.
- Early steps: shadow ≈ random $\theta_0$ → warm up the decay, e.g., $\beta_t=\min\left(\beta,\frac{1+t}{10+t}\right)$.

*How to choose $\beta$?*
- Window $1/(1-\beta)$: ≪ total steps, ≫ correlation time of the iterates.
- DDPM: $\beta=0.9999$. {cite:p}`ho2020denoising`

*Weight EMA vs momentum?*
- Both = EMA.
- [Momentum](optim.md#momentum): EMA of grads → drives each update → changes the trajectory.
- Weight EMA: EMA of params. Eval-only → trajectory unchanged. EMA teacher → sets the targets → changes the online grads.

*Weight EMA vs LR decay?*
- Both reduce the variance of the eval point near a minimum → their gains overlap.
- LR → 0 at the end → iterates stop moving → EMA ≈ final iterate → EMA gain shrinks.

*Why an EMA teacher in SSL & RL?*
- Target from the online net → target moves w/ every step → inconsistent targets → instability or collapse.
- EMA teacher → slow target → consistent targets.
- [MoCo](../repr/ssl.md#moco) key encoder: no averaging → training fails to converge. Slow EMA → works.
- Same pattern: [DINO](../../cv/vit.md#dino) teacher, [Consistency Model](../gen/diffusion.md#consistency-model) target, [DDPG](../../rl/policy.md#ddpg) target nets (Polyak update, $\tau=1-\beta$).

*BatchNorm buffers?*
- Running stats were collected w/ $\theta$, ❌$\bar\theta$. Usually close enough ← $\bar\theta\approx\theta$.
- Large mismatch → recompute the stats w/ $\bar\theta$, as in [SWA](#swa).
```

&nbsp;

### SWA
- **Name**: Stochastic Weight Averaging {cite:p}`izmailov2018averaging`
- **What**: Equal-weight average of SGD iterates, collected late in training under a high constant or cyclical LR.
- **Why**: High-LR SGD stays near the edge of a wide low-loss region. It never reaches the center.
    - Average of points spread around the edge → near the center → wider optimum → ⬆️generalization.
    - ≈ ensemble of the iterates at the cost of 1 model.
- **How**:
    1. Train normally for ~75% of the budget (or start from a pretrained model).
    2. Switch to a high constant or cyclical LR → SGD explores the region.
    3. Every $c$ steps (e.g., once per epoch): add the curr weights to an equal-weight running average.
    4. After training: recompute BatchNorm stats w/ 1 pass over train data under the averaged weights ← stats never collected for them.
    5. Eval w/ the average.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\theta_\text{SWA}$: Averaged params.
    - $\hat\theta$: Pretrained params (start of the SWA phase).
- Hyperparams:
    - $\alpha_1$: Max LR of a cycle.
    - $\alpha_2$: Min LR of a cycle, $\alpha_2\le\alpha_1$.
    - $c$: Cycle length = snapshot interval (steps).
- Misc:
    - $N$: #snapshots in the average.

Process:
1. LR. Constant: $\eta(t)=\alpha_1$ (⇔ $\alpha_2=\alpha_1$). Cyclical: linear decay $\alpha_1\to\alpha_2$ within each cycle, then jump back to $\alpha_1$.

    $$
    \eta(t)=(1-\tau(t))\,\alpha_1+\tau(t)\,\alpha_2,\qquad\tau(t)=\frac{1}{c}\Big(\big((t-1)\bmod c\big)+1\Big)
    $$

    - $\tau(t)\in(0,1]$: Position inside the curr cycle. $\tau=1$ → cycle end → LR $=\alpha_2$.
2. Init $\theta_\text{SWA}=\hat\theta$, $N=1$.
3. If $t\bmod c=0$ (cyclical: end of a cycle, lowest LR):

    $$
    \theta_\text{SWA}\leftarrow\frac{N\,\theta_\text{SWA}+\theta_t}{N+1},\qquad N\leftarrow N+1
    $$

4. Recompute BatchNorm stats for $\theta_\text{SWA}$.
```

```{tip} Derivation
:class: dropdown
*Why does averaging weights ≈ averaging predictions?*
1. Snapshots $\theta_1,\dots,\theta_N$ lie close in weight space. $\theta_\text{SWA}=\frac{1}{N}\sum_i\theta_i$.
2. $\Delta_i=\theta_i-\theta_\text{SWA}$ → $\sum_i\Delta_i=0$.
3. $f(\theta)$: scalar prediction (e.g., 1 class prob), twice differentiable in $\theta$.
4. Linearize at $\theta_\text{SWA}$: $f(\theta_i)=f(\theta_\text{SWA})+\langle\nabla f(\theta_\text{SWA}),\Delta_i\rangle+O(\|\Delta_i\|^2)$.
5. Ensemble − SWA:

    $$
    \frac{1}{N}\sum_if(\theta_i)-f(\theta_\text{SWA})=\Big\langle\nabla f(\theta_\text{SWA}),\frac{1}{N}\sum_i\Delta_i\Big\rangle+O(\Delta^2)=O(\Delta^2)
    $$

    - $\Delta=\max_i\|\Delta_i\|$.
6. Snapshot vs snapshot: $f(\theta_i)-f(\theta_j)=\langle\nabla f(\theta_\text{SWA}),\Delta_i-\Delta_j\rangle+O(\Delta^2)$ → 1st order.
7. → Snapshots disagree at 1st order. Ensemble & SWA agree up to 2nd order. → SWA ≈ ensemble at 1 model's inference cost.
```

````{important} Code
:class: dropdown
```python
import copy
import torch
import torch.nn as nn

class SWA:
    def __init__(self, model):
        self.avg = copy.deepcopy(model)                  ## theta_SWA = theta_hat, N = 1
        self.n = 1

    @torch.no_grad()
    def update(self, model):
        ## equal weights: running mean over all snapshots so far
        for pa, p in zip(self.avg.parameters(), model.parameters()):
            pa.mul_(self.n / (self.n + 1)).add_(p, alpha=1 / (self.n + 1))
        self.n += 1

    @torch.no_grad()
    def update_bn(self, loader):
        ## BN stats belong to the iterates, not to their average -> recompute
        for m in self.avg.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.reset_running_stats()
                m.momentum = None                        ## None -> cumulative average over all batches
        self.avg.train()
        for x in loader:
            self.avg(x)
        self.avg.eval()

## Example
model = nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8), nn.ReLU(), nn.Linear(8, 2))
swa = SWA(model)
opt = torch.optim.SGD(model.parameters(), lr=0.05)      ## constant LR in the SWA phase
for step in range(1, 7):
    loss = model(torch.randn(16, 4)).pow(2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 2 == 0:                                    ## c = 2
        swa.update(model)
swa.update_bn([torch.randn(16, 4) for _ in range(3)])
print(swa.n, swa.avg(torch.randn(2, 4)).shape)           ## 4 torch.Size([2, 2])
```
````

```{dropdown} Table: EMA vs SWA
| | EMA | SWA |
|:--|:--|:--|
| Snapshot weights | Exponential, recent-heavy | Equal |
| Snapshots used | Last ~$1/(1-\beta)$ steps | SWA phase only |
| LR during averaging | Any schedule | High constant or cyclical |
| Update frequency | Every step | Every $c$ steps |
| BatchNorm stats | Copied (approx.) | Recomputed |
| Typical role | Eval model + SSL/RL teacher | Final model |
```

```{attention} Q&A
:class: dropdown
*Pros?*
- ~Free: 1 weighted sum per $c$ steps + 1 BatchNorm pass.
- ⬆️Test accuracy vs standard SGD on CIFAR & ImageNet CNNs.
- 1-model inference cost.

*Cons?*
- Needs the BatchNorm recompute pass. Skip it → stats from other weights → ⬇️accuracy.
- Extra HPs: SWA-phase LR, $c$, start point.
- Valid only within 1 basin. Average of independent runs → high-loss point ← permutation symmetry. → [Mode Connectivity](../theory.md#mode-connectivity)

*Why a high constant/cyclical LR, ❌decayed LR?*
- Decayed LR → iterates collapse onto 1 point → nothing to average.
- High LR → iterates spread over the low-loss region → their mean lands near its center.

*Why does a wider optimum generalize better?*
- Train & test loss surfaces are shifted relative to each other.
- SGD's point: near the edge → small shift → large test-loss increase. Center → robust to the shift.
- Caveat: the flatness ↔ generalization link is contested ← sharpness changes under reparametrization. → [Flat Minima](../theory.md#flat-minima)
```

&nbsp;

## Batch Size
- **What**: #samples averaged into 1 grad estimate per optimizer step.

&nbsp;

### Linear Scaling Rule
- **What**: Batch size ×$\kappa$ → LR ×$\kappa$. {cite:p}`goyal2017accurate`
- **Why**: Scale out data parallelism w/o re-tuning the LR.
    - 1 large-batch step w/ LR ×$\kappa$ ≈ $\kappa$ small-batch steps ← if grads change little over $\kappa$ steps.
    - → Same training curve vs #samples seen.
- **How**:
    1. Tune LR $\eta_0$ at a reference batch size $B_0$.
    2. New batch size $B$ → LR $\eta_0B/B_0$. Keep all other HPs, incl. weight decay.
    3. Add gradual [warmup](schedule.md#warmup) from $\eta_0$ ← the assumption fails early, when weights change fast.
    4. Make the aggregated grad = mean over the global batch. Sum all-reduce → divide each worker's loss by the global batch size. Averaging all-reduce (e.g., DDP) → per-worker mean loss is already correct.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\kappa$: Batch-size multiplier.
    - $\beta_1$: Adam 1st-moment decay.
    - $\beta_2$: Adam 2nd-moment decay.
    - $\epsilon$: Adam denominator constant.

Process:

SGD (linear rule):

$$
B\to\kappa B\quad\Rightarrow\quad\eta\to\kappa\eta
$$

Adam (square-root rule): {cite:p}`malladi2022on`

$$
B\to\kappa B\quad\Rightarrow\quad\eta\to\sqrt{\kappa}\,\eta,\quad1-\beta_1\to\kappa(1-\beta_1),\quad1-\beta_2\to\kappa(1-\beta_2),\quad\epsilon\to\epsilon/\sqrt{\kappa}
$$

ResNet-50 / ImageNet recipe:

$$
\eta=0.1\cdot\frac{B}{256}
$$

- Gradual warmup: LR ramps linearly from 0.1 to $0.1B/256$ over the first 5 epochs.
```

```{tip} Derivation
:class: dropdown
*Why linear for SGD?*
1. $\kappa$ SGD steps, batch $B$, LR $\eta$:

    $$
    \theta_{t+\kappa}=\theta_t-\frac{\eta}{B}\sum_{j<\kappa}\sum_{\mathbf{x}\in\mathcal{B}_j}\nabla\ell(\mathbf{x},\theta_{t+j})
    $$

    - $\ell(\mathbf{x},\theta)$: Per-sample loss.
    - $\mathcal{B}_j$: $j$-th minibatch, $|\mathcal{B}_j|=B$.
2. 1 step on the union $\cup_j\mathcal{B}_j$ (size $\kappa B$), LR $\hat\eta$:

    $$
    \hat\theta_{t+1}=\theta_t-\frac{\hat\eta}{\kappa B}\sum_{j<\kappa}\sum_{\mathbf{x}\in\mathcal{B}_j}\nabla\ell(\mathbf{x},\theta_t)
    $$

3. Assume $\nabla\ell(\mathbf{x},\theta_{t+j})\approx\nabla\ell(\mathbf{x},\theta_t)$ for $j<\kappa$.
4. → $\hat\theta_{t+1}\approx\theta_{t+\kappa}$ only if $\hat\eta=\kappa\eta$.
5. Assumption fails: (a) early training → warmup. (b) very large $\kappa$ → accuracy drops (ResNet-50 / ImageNet: beyond ~8k).

*Why linear for SGD but square root for Adam? (SDE view)*
1. Batch $B$ → grad-noise std $\sigma\propto1/\sqrt{B}$.
2. SGD: SDE time step = $\eta$. Noise coeff ∝ $\sigma\sqrt{\eta}$. Constant only if $\eta/B$ is constant → linear.
3. Adam: update normalized by $\sqrt{v}$ → SDE time step = $\eta^2$. Noise coeff ∝ $\sigma\eta$. Constant only if $\eta/\sqrt{B}$ is constant → square root.
4. $1-\beta_{1,2}\propto\eta^2$ & $\epsilon\propto1/\eta$ → moment windows & $\epsilon$ stay fixed in SDE time.
```

````{important} Code
:class: dropdown
```python
import math

class ScaledLR:
    def __init__(self, base_lr, base_batch, batch, rule="linear", warmup_steps=0):
        k = batch / base_batch
        ## SGD -> linear (eta / B fixed); "sqrt" = LR part of the Adam rule only (see adam_sqrt)
        self.target = base_lr * (k if rule == "linear" else math.sqrt(k))
        self.base_lr, self.warmup = base_lr, warmup_steps

    def __call__(self, step):
        ## gradual warmup base_lr -> target: large LR is unsafe while weights change fast
        if step < self.warmup:
            return self.base_lr + (self.target - self.base_lr) * step / self.warmup
        return self.target

    @staticmethod
    def adam_sqrt(k, lr, beta1, beta2, eps):
        ## full Adam rule: LR x sqrt(k), shorter moment windows, eps / sqrt(k) -> same SDE
        return lr * math.sqrt(k), 1 - k * (1 - beta1), 1 - k * (1 - beta2), eps / math.sqrt(k)

## Example: ResNet-50 recipe, 0.1 @ 256 -> 8192, 5 warmup epochs
sched = ScaledLR(0.1, 256, 8192, warmup_steps=5)
print([round(sched(e), 2) for e in range(7)])            ## [0.1, 0.72, 1.34, 1.96, 2.58, 3.2, 3.2]
print(ScaledLR.adam_sqrt(4, 3e-4, 0.9, 0.999, 1e-8))     ## ≈ (6e-4, 0.6, 0.996, 5e-9): x4 batch, Adam
```
````

```{attention} Q&A
:class: dropdown
*Assumptions?*
- Grads change little over $\kappa$ small steps.
- $B$ below the [critical batch size](#critical-batch-size) → noise-dominated regime.
- SGD-like update (SGD, SGD + momentum). Adam → square-root rule.

*When does it break?*
- Early training → weights change fast → warmup.
- $B$ ≫ critical batch size → LR hits the stability limit set by curvature → extra samples are wasted.
- ResNet-50 / ImageNet: holds up to ~8k.

*Scale the loss instead of the LR?*
- ❌ Not equivalent w/ weight decay. The L2 term is not scaled w/ the loss → effective decay changes.

*Momentum gotcha?*
- LR folded into the velocity ($\mathbf{v}\leftarrow m\mathbf{v}+\eta\,\mathbf{g}_B$) → after an LR change, rescale $\mathbf{v}$ by $\eta_{t+1}/\eta_t$ (**momentum correction**).
    - $m$: Momentum coeff.
    - $\mathbf{g}_B$: Minibatch grad.
- W/o it: LR jumps ⬆️ → velocity too small for the new LR → instability.

*Huge batches?*
- Layer-wise adaptive LR → [LARS](optim.md#lars) (SGD), [LAMB](optim.md#lamb) (Adam).

*Gradient accumulation?*
- Accumulate over $\kappa$ micro-batches ⇔ batch $\kappa B$ → same rule. → [Gradient Accumulation](../sys.md#gradient-accumulation)

*What else changes w/ batch size?*
- Generalization, BatchNorm noise, #updates per epoch. → [Batch-size Pathologies](issues.md#batch-size-pathologies)
```

&nbsp;

### Critical Batch Size
- **What**: Batch size beyond which ⬆️$B$ barely cuts #steps but multiplies #samples. {cite:p}`mccandlish2018an`
- **Why**: Pick $B$ for a time vs compute budget.
    - $B\ll B_\text{crit}$: grad noise ≫ signal → ⬆️$B$ → ~linear ⬇️#steps, ~same #samples → free speedup.
    - $B\gg B_\text{crit}$: grad already accurate → ⬆️$B$ → ~same #steps, ⬆️#samples → wasted compute.
- **How**: $B_\text{crit}\approx$ **gradient noise scale** (grad variance / squared grad norm).
    1. Measure the grad norm at 2 batch sizes. DDP: per-device grad before all-reduce vs global grad after → ~free.
    2. Solve for the squared true-grad norm & the total grad variance.
    3. EMA both over steps. Noise scale = ratio of the 2 EMAs.
    4. Re-measure during training ← it grows as loss ⬇️.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{g}_B$: Minibatch grad estimate at batch size $B$.
- Misc:
    - $\Sigma$: Per-sample grad covariance.
    - $H$: Hessian of $\mathcal{L}$.
    - $S$: #optimizer steps to reach a target loss.
    - $S_\text{min}$: Min #steps ($B\to\infty$).
    - $E$: #training samples to reach the target loss. $E=BS$.
    - $E_\text{min}$: Min #samples ($B\to0$).

Model (i.i.d. samples):

$$
\mathbb{E}[\mathbf{g}_B]=\mathbf{g},\qquad\text{Cov}[\mathbf{g}_B]=\frac{\Sigma}{B}
$$

Objective (1 SGD step, quadratic expansion):

$$
\mathbb{E}\left[\mathcal{L}(\theta-\eta\,\mathbf{g}_B)\right]\approx\mathcal{L}(\theta)-\eta\|\mathbf{g}\|^2+\frac{\eta^2}{2}\left(\mathbf{g}^TH\mathbf{g}+\frac{\text{tr}(H\Sigma)}{B}\right)
$$

Optimal LR & loss drop:

$$
\eta_\text{opt}(B)=\frac{\eta_\text{max}}{1+B_\text{noise}/B},\qquad\Delta\mathcal{L}_\text{opt}(B)=\frac{\Delta\mathcal{L}_\text{max}}{1+B_\text{noise}/B}
$$

- $\eta_\text{max}=\|\mathbf{g}\|^2/\mathbf{g}^TH\mathbf{g}$: Optimal LR w/ the true grad.
- $\Delta\mathcal{L}_\text{max}=\frac{1}{2}\|\mathbf{g}\|^4/\mathbf{g}^TH\mathbf{g}$: Max loss drop per step.

Noise scale:

$$
B_\text{noise}=\frac{\text{tr}(H\Sigma)}{\mathbf{g}^TH\mathbf{g}}\ \xrightarrow{H\propto I}\ B_\text{simple}=\frac{\text{tr}(\Sigma)}{\|\mathbf{g}\|^2}
$$

Steps vs samples (fit over a whole training run):

$$
\left(\frac{S}{S_\text{min}}-1\right)\left(\frac{E}{E_\text{min}}-1\right)=1,\qquad B_\text{crit}=\frac{E_\text{min}}{S_\text{min}}\approx B_\text{noise}
$$

- ⇔ $S=S_\text{min}(1+B_\text{crit}/B)$ & $E=E_\text{min}(1+B/B_\text{crit})$.
- $B=B_\text{crit}$ → $2S_\text{min}$ steps & $2E_\text{min}$ samples.

Estimator (2 batch sizes $B_s<B_b$):

$$
\mathbb{E}\|\mathbf{g}_B\|^2=\|\mathbf{g}\|^2+\frac{\text{tr}(\Sigma)}{B}
$$

$$
\widehat{\|\mathbf{g}\|^2}=\frac{B_b\|\mathbf{g}_{B_b}\|^2-B_s\|\mathbf{g}_{B_s}\|^2}{B_b-B_s},\qquad\widehat{\text{tr}(\Sigma)}=\frac{\|\mathbf{g}_{B_s}\|^2-\|\mathbf{g}_{B_b}\|^2}{1/B_s-1/B_b}
$$

- Both unbiased. Their ratio is biased → average each over many steps first.
```

```{tip} Derivation
:class: dropdown
*Where do $\eta_\text{opt}$ & $\Delta\mathcal{L}_\text{opt}$ come from?*
1. Expected loss after 1 step = quadratic in $\eta$ (Objective above).
2. Set $\partial/\partial\eta=0$: $-\|\mathbf{g}\|^2+\eta\left(\mathbf{g}^TH\mathbf{g}+\text{tr}(H\Sigma)/B\right)=0$.
3. → $\eta_\text{opt}=\dfrac{\|\mathbf{g}\|^2}{\mathbf{g}^TH\mathbf{g}+\text{tr}(H\Sigma)/B}=\dfrac{\eta_\text{max}}{1+B_\text{noise}/B}$.
4. Plug back: $\Delta\mathcal{L}_\text{opt}=\frac{1}{2}\eta_\text{opt}\|\mathbf{g}\|^2=\dfrac{\Delta\mathcal{L}_\text{max}}{1+B_\text{noise}/B}$.
5. $B\ll B_\text{noise}$ → progress per step ∝ $B$ (linear speedup). $B\gg B_\text{noise}$ → progress per step → $\Delta\mathcal{L}_\text{max}$ (saturates).

*Why is the estimator unbiased?*
1. $\mathbb{E}\|\mathbf{g}_B\|^2=\|\mathbb{E}\mathbf{g}_B\|^2+\text{tr}(\text{Cov}\,\mathbf{g}_B)=\|\mathbf{g}\|^2+\text{tr}(\Sigma)/B$.
2. $B_b\,\mathbb{E}\|\mathbf{g}_{B_b}\|^2-B_s\,\mathbb{E}\|\mathbf{g}_{B_s}\|^2=(B_b-B_s)\|\mathbf{g}\|^2$ → $\text{tr}(\Sigma)$ cancels.
3. $\mathbb{E}\|\mathbf{g}_{B_s}\|^2-\mathbb{E}\|\mathbf{g}_{B_b}\|^2=\text{tr}(\Sigma)(1/B_s-1/B_b)$ → $\|\mathbf{g}\|^2$ cancels.
```

````{important} Code
:class: dropdown
```python
import torch

class GradNoiseScale:
    def __init__(self, b_small, b_big, beta=0.99):
        self.bs, self.bb, self.beta = b_small, b_big, beta
        self.g2 = self.tr = None

    def update(self, g_small, g_big):
        ## E|g_B|^2 = |g|^2 + tr(Sigma)/B at 2 batch sizes -> 2 equations, 2 unknowns
        ns, nb = g_small.pow(2).sum(), g_big.pow(2).sum()
        g2 = (self.bb * nb - self.bs * ns) / (self.bb - self.bs)
        tr = (ns - nb) / (1 / self.bs - 1 / self.bb)
        ## each estimate is noisy -> EMA both, then take the ratio (not an EMA of ratios)
        if self.g2 is None:
            self.g2, self.tr = g2, tr
        else:
            self.g2 = self.beta * self.g2 + (1 - self.beta) * g2
            self.tr = self.beta * self.tr + (1 - self.beta) * tr
        return (self.tr / self.g2).item()                ## B_simple ~ B_crit

## Example: |g|^2 = 1, unit noise in 100 dims -> tr(Sigma) = 100 -> B_simple = 100
g = torch.zeros(100); g[0] = 1.0
est = GradNoiseScale(b_small=8, b_big=64)
for _ in range(500):
    per_sample = g + torch.randn(64, 100)                ## 64 per-sample grads
    b = est.update(per_sample[:8].mean(0), per_sample.mean(0))   ## local (8) vs global (64) batch
print(round(b))                                          ## ≈ 100
```
````

```{attention} Q&A
:class: dropdown
*How does it relate to the linear scaling rule?*
- $B\ll B_\text{noise}$: $\eta_\text{opt}\approx\eta_\text{max}B/B_\text{noise}\propto B$ → [Linear Scaling Rule](#linear-scaling-rule).
- $B\gg B_\text{noise}$: $\eta_\text{opt}\to\eta_\text{max}$ → LR saturates → larger batch ≈ same progress per step.

*Why does it grow during training?*
- $\text{tr}(\Sigma)$ ≈ constant. $\|\mathbf{g}\|$ ⬇️ as loss ⬇️ → ratio ⬆️.
- → Small batch early, larger batch later (batch-size ramp).

*Does it depend on model size?*
- Mainly via the loss reached, ❌directly.
- LMs: $B_\text{crit}(\mathcal{L})\approx B_*/\mathcal{L}^{1/\alpha_B}$, $B_*\approx2\times10^8$ tokens, $\alpha_B\approx0.21$ → doubles per ~13% loss ⬇️. {cite:p}`kaplan2020scaling`
    - $\mathcal{L}$: LM cross-entropy loss (nats).
- Across tasks: tens of thousands (ImageNet) → millions (Dota 2 RL).

*$B_\text{simple}$ vs $B_\text{noise}$?*
- $B_\text{simple}$ ignores curvature ($H\propto I$) → cheap.
- Usually within a small constant factor of $B_\text{noise}$. Ratio $B_\text{simple}/B_\text{crit}$ varies up to ~10× across tasks → order-of-magnitude estimate.

*Which $B$ to choose?*
- Compute-limited → $B<B_\text{crit}$.
- Time-limited → $B>B_\text{crit}$.
- $B\approx B_\text{crit}$ → balanced: 2× min steps, 2× min samples.

*Assumptions?*
- Local quadratic loss. i.i.d. samples ($B$ ≪ dataset size). Plain SGD step.
- Empirically still predictive for momentum, Adam, RMSprop.
- Measured noise scale depends on the LR → measure on a run w/ a well-tuned LR.

*What else changes w/ batch size?*
- Generalization, BatchNorm noise, #updates per epoch. → [Batch-size Pathologies](issues.md#batch-size-pathologies)
```

&nbsp;

## Hyperparameter Transfer
- **What**: Tune HPs on a small model → reuse them on a large model.

&nbsp;

### muP
- **Name**: Maximal Update Parametrization (μP) {cite:p}`yang2021tensor`
- **What**: Per-layer width scaling of init variance, LR & multipliers s.t. every layer's features change by $\Theta(1)$ per step as width → ∞.
- **Why**: Standard parametrization (SP) → optimal LR drifts w/ width.
    - SP: logits & attention logits blow up w/ width after 1 step → max stable LR ⬇️ as width ⬆️ → small-model HPs ❌transfer.
    - NTK parametrization: stable, but features freeze as width → ∞ (kernel regime) → ❌feature learning.
    - μP: stable + max feature learning in every layer → dynamics converge to a width-independent limit → optimal HPs stay put.
- **How**: Classify each weight by how its dims grow w/ width → per-class rule (Adam).
    - **Hidden** (width → width): init var $1/\text{fan\_in}$ (= SP). Adam LR $\propto1/\text{fan\_in}$.
        - Init: $W\mathbf{h}$ sums $n$ random-sign terms (CLT) → $1/\sqrt{n}$ entries suffice.
        - Update: $\Delta W$ aligns w/ $\mathbf{h}$ → $\Delta W\mathbf{h}$ sums $n$ same-sign terms (LLN) → Adam entries ($\Theta(\eta)$ each) need $\eta\propto1/n$.
    - **Output** (width → finite, e.g., unembedding): init var $\propto1/\text{fan\_in}^2$. Adam LR $\propto1/\text{fan\_in}$.
        - ← Output sums $n$ features that become correlated w/ the output weights (LLN) → needs a $1/n$ factor, ❌$1/\sqrt{n}$.
    - **Input** (finite → width, e.g., embedding) & all biases: same as SP.
    - **Attention logit**: $\mathbf{q}^T\mathbf{k}/d_k$, ❌$/\sqrt{d_k}$ ← $\mathbf{q}$ & $\mathbf{k}$ become correlated in training (LLN).
    - Write every width factor as $n/n_0$ (attention: $\sqrt{d_{k,0}}/d_k$) → μP = SP at base width $n_0$.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $W_1\in\mathbb{R}^{n\times d_\text{in}}$: Input weights.
    - $W_2\in\mathbb{R}^{n\times n}$: Hidden weights.
    - $W_3\in\mathbb{R}^{d_\text{out}\times n}$: Output weights.
    - $\mathbf{b}_1\in\mathbb{R}^{n}$: Bias of layer 1.
    - $\mathbf{b}_2\in\mathbb{R}^{n}$: Bias of layer 2.
- Hyperparams:
    - $\eta$: Master LR, tuned at base width.
    - $n_0$: Base (proxy) width.
- Misc:
    - $n$: Width. Overrides Tier-1 $n$ (#features).
    - $\tilde{n}=n/n_0$: Width multiplier.
    - $d_\text{in}$: Input dim, fixed.
    - $d_\text{out}$: Output dim, fixed.
    - $\phi$: Activation.
    - $\eta_W$: Adam LR of tensor $W$.

Model (MLP):

$$
f(\mathbf{x})=W_3\,\phi\big(W_2\,\phi(W_1\mathbf{x}+\mathbf{b}_1)+\mathbf{b}_2\big)
$$

μP init:

$$
W_1\sim\mathcal{N}\Big(0,\frac{1}{d_\text{in}}\Big),\quad W_2\sim\mathcal{N}\Big(0,\frac{1}{n}\Big),\quad W_3\sim\mathcal{N}\Big(0,\frac{1}{n\tilde{n}}\Big),\quad\mathbf{b}_1=\mathbf{b}_2=\mathbf{0}
$$

μP Adam LRs:

$$
\eta_{W_1}=\eta_{\mathbf{b}_1}=\eta_{\mathbf{b}_2}=\eta,\qquad\eta_{W_2}=\eta_{W_3}=\frac{\eta}{\tilde{n}}
$$

- $n=n_0$ → $\tilde{n}=1$ → identical to SP ($W_{2,3}\sim\mathcal{N}(0,1/n)$, 1 global LR).
- $n$ ⬆️: $\text{Var}(W_3)=n_0/n^2\propto1/\text{fan\_in}^2$. Hidden & output LR $\propto1/\text{fan\_in}$.

Attention logit (Transformer):

$$
\text{logit}=\frac{\mathbf{q}^T\mathbf{k}}{d_k}\qquad(\text{SP: }\mathbf{q}^T\mathbf{k}/\sqrt{d_k})
$$

- $\mathbf{q},\mathbf{k}\in\mathbb{R}^{d_k}$: Query & key of 1 head.
- Match SP at base head dim $d_{k,0}$: $\sqrt{d_{k,0}}\,\mathbf{q}^T\mathbf{k}/d_k$ (official `mup` code: $8/d_k$ ⇔ $d_{k,0}=64$).
```

```{tip} Derivation
:class: dropdown
*Why does SP blow up w/ width?* (1-hidden-layer linear net, SGD)
1. $f(x)=\mathbf{v}^T\mathbf{u}\,x$. Scalar $x$. $\mathbf{u},\mathbf{v}\in\mathbb{R}^n$.
2. SP: $u_i\sim\mathcal{N}(0,1)$, $v_i\sim\mathcal{N}(0,1/n)$ → $f=\Theta(|x|)$ at init.
3. 1 SGD step, LR 1: $\mathbf{v}'=\mathbf{v}+c\,\mathbf{u}$, $\mathbf{u}'=\mathbf{u}+c\,\mathbf{v}$.
    - $c=\Theta(1)$: scalar set by input, label & loss.
4. New output:

    $$
    f'(x)=\left(\mathbf{v}^T\mathbf{u}+c\,\mathbf{u}^T\mathbf{u}+c\,\mathbf{v}^T\mathbf{v}+c^2\,\mathbf{u}^T\mathbf{v}\right)x
    $$

5. $\mathbf{u}^T\mathbf{u}=\Theta(n)$ (LLN) → $f'$ blows up as $n\to\infty$.
6. μP: $v_i\sim\mathcal{N}(0,1/n^2)$, $\eta_\mathbf{v}=1/n$, $\eta_\mathbf{u}=n$:

    $$
    f'(x)=\left(\mathbf{v}^T\mathbf{u}+\frac{c}{n}\mathbf{u}^T\mathbf{u}+cn\,\mathbf{v}^T\mathbf{v}+c^2\,\mathbf{u}^T\mathbf{v}\right)x
    $$

    - $\frac{c}{n}\mathbf{u}^T\mathbf{u}\to c$. $cn\,\mathbf{v}^T\mathbf{v}\to c$ ← $\mathbf{v}^T\mathbf{v}\approx n\cdot\frac{1}{n^2}$.
    - $\mathbf{v}^T\mathbf{u},\mathbf{u}^T\mathbf{v}=O(1/\sqrt{n})\to0$ ← sum of $n$ random-sign terms of size $1/n$.
7. → $f'=\Theta(1)$ at any width. Both layers contribute $\Theta(1)$ change → feature learning survives.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class MuPMLP(nn.Module):
    def __init__(self, d_in, n, d_out, n0):
        super().__init__()
        self.r = n / n0                                            ## width multiplier n~
        self.inp = nn.Linear(d_in, n)
        self.hid = nn.Linear(n, n)
        self.out = nn.Linear(n, d_out, bias=False)
        nn.init.normal_(self.inp.weight, std=d_in ** -0.5)         ## input: 1/fan_in (= SP)
        nn.init.normal_(self.hid.weight, std=n ** -0.5)            ## hidden: 1/fan_in (= SP)
        nn.init.normal_(self.out.weight, std=(n * self.r) ** -0.5) ## output: 1/(n * n~) ∝ 1/fan_in^2
        nn.init.zeros_(self.inp.bias)
        nn.init.zeros_(self.hid.bias)

    def forward(self, x):
        return self.out(torch.relu(self.hid(torch.relu(self.inp(x)))))

    def param_groups(self, lr):
        ## Adam: width->width & width->finite weights get lr / n~; input weights & biases keep lr
        return [
            {"params": [self.inp.weight, self.inp.bias, self.hid.bias], "lr": lr},
            {"params": [self.hid.weight, self.out.weight], "lr": lr / self.r},
        ]

## Example: lr tuned at n0 = 64, reused unchanged at n = 1024
model = MuPMLP(d_in=4, n=1024, d_out=2, n0=64)
opt = torch.optim.Adam(model.param_groups(lr=1e-3))
print(model(torch.randn(2, 4)).shape)                ## torch.Size([2, 2])
print([g["lr"] for g in opt.param_groups])           ## [0.001, 6.25e-05]
```
````

```{dropdown} Table: μP vs SP
| | Input weights & all biases | Hidden weights | Output weights |
|:--|:--|:--|:--|
| Init var | $1/\text{fan\_in}$ | $1/\text{fan\_in}$ | $1/\text{fan\_in}^2$ (SP: $1/\text{fan\_in}$) |
| SGD LR | $\text{fan\_out}$ (SP: 1) | 1 | $1/\text{fan\_in}$ (SP: 1) |
| Adam LR | 1 | $1/\text{fan\_in}$ (SP: 1) | $1/\text{fan\_in}$ (SP: 1) |

- Width scaling only. Tunable constants omitted.
- Bias: $\text{fan\_in}=1$, $\text{fan\_out}$ = its dim.
- Transformer: also $1/d_k$ attention.
```

```{attention} Q&A
:class: dropdown
*SP vs NTK parametrization vs μP?*
- SP: max stable LR ⬇️ w/ width → optimum drifts.
- NTK parametrization: infinite-width limit = kernel GD → features fixed → ❌pretraining & transfer benefit. → [NTK](../theory.md#ntk)
- Dichotomy: in this parametrization family, a limit has feature learning or kernel dynamics, ❌both. μP = max feature learning.

*Why do HPs transfer?*
- Training dynamics → width-independent limit → loss as a function of HPs converges → its argmin stabilizes.
- Empirically: at fixed HPs, wider = weakly better in μP. ❌ in SP.

*Pitfalls?*
- Reparametrize ALL width-dependent HPs (init, multipliers, input & output LR), ❌only the global LR.
- SGD & Adam rules differ (Table).
- Tied embedding/unembedding: use the equivalent form "output multiplier $1/\text{fan\_in}$, init var 1, Adam LR 1" → same rule as input weights → tying OK.
- Zero-init output layer & query projection → closer HP match between proxy & target.
    - ← μP net at init ≈ Gaussian process w/ variance $\Theta(1/n)$ → negligible at large $n$, NOT at small proxy $n$.

*μP vs linear scaling rule?*
- Both re-scale HPs along 1 scale axis.
- [Linear Scaling Rule](#linear-scaling-rule): batch size → LR. μP: width → per-layer init & LR.
```

&nbsp;

#### muTransfer
- **What**: Tune HPs on a small μP proxy → copy them zero-shot to the large μP target. {cite:p}`yang2021tuning`
- **Why**: Direct HP search on a billion-param model = prohibitive.
- **How**:
    1. Parametrize the target model in μP.
    2. Tune a smaller version (narrower and/or shallower).
    3. Copy the tuned HPs to the target. ❌Re-tune.

```{attention} Q&A
:class: dropdown
*What transfers?*
- ✅ Optimization HPs (LR, schedule, Adam $\beta$), init scale, multipliers.
- ❌ Regularization HPs (dropout, weight decay) ← right amount depends on model size & data size.
- Transfer across: width (theory + empirical). Depth, batch size, seq length, training time (empirical, Transformers only).

*Results?*
- 13M proxy → BERT-large (350M): beats published numbers. Tuning cost ≈ 1 BERT-large pretraining.
- 40M proxy → GPT-3 6.7B: beats published numbers. Tuning cost = 7% of pretraining.

*When does it fail?*
- Proxy too small. Transformer LM thresholds, e.g., width 256, depth 4, batch 32, seq length 128, 5000 steps.
- Init std does not transfer well across depth.
- Depth transfer fails for post-LN Transformers.
- μP theory covers width only. Depth transfer = empirical.
```

&nbsp;
