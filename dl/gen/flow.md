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
# Normalizing Flows
Generative models that map data to a simple base density through invertible layers → exact likelihood. Family overview → [Generative Model](gen.md#generative-model). Simulation-free training of continuous flows → [Flow Matching](diffusion.md#flow-matching).

Notations:
- $x\in\mathbb{R}^d$: Data sample
- $z\in\mathbb{R}^d$: Latent (base) sample
- $d$: Data dim
- $p_X$: Model density of $x$
- $p_Z$: Base density, usually $\mathcal{N}(0,I)$
- $f_\theta:x\mapsto z$: Flow (normalizing direction)
- $g_\theta=f_\theta^{-1}:z\mapsto x$: Generator (sampling direction)
- $J_f(x)=\partial f/\partial x\in\mathbb{R}^{d\times d}$: Jacobian of $f$
- $K$: #flow layers
- $\odot$: Elementwise product

Overrides of global notation: $x$, $z$ = whole vectors (non-bold), ❌scalars. Subscripts $i,j$ = dimension idx ($x_j$), ❌sample/feature idx. Subscript $k$ = layer idx, ❌class idx.

&nbsp;

## Normalizing Flow
- **What**: Invertible, differentiable map from data to a simple base density. Density via change of variables. {cite:p}`tabak2010density`
- **Why**: Exact likelihood + exact sampling (invert the map) + exact latent inference in 1 model.
    - Cost of density vs sampling depends on the architecture → Table: Flow Families.
    - VAE: likelihood only via a bound. GAN: ❌likelihood.
    - Invertible map → each $x$ has exactly 1 $z$ → density of $x$ = density of $z$, corrected for local volume change.
- **How**:
    1. Stack $K$ simple invertible layers: $z=f_K\circ\dots\circ f_1(x)$.
    2. **Density**: push $x$ through $f$ → $\log p_Z(z)$ + sum of per-layer log-det-Jacobians.
    3. **Train**: maximize exact log-likelihood (MLE).
    4. **Sample**: $z\sim p_Z$ → invert the layers in reverse order → $x=g(z)$.
    - Design rule: each layer must be (a) invertible & (b) cheap in log-det. Dense $d\times d$ det = $O(d^3)$ → ❌.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $z_k$: Output of layer $k$. $z_0=x$, $z_K=z$.

Model:

$$
\log p_X(x)=\log p_Z\big(f(x)\big)+\log\left|\det J_f(x)\right|
$$

Composition:

$$
\log p_X(x)=\log p_Z(z_K)+\sum_{k=1}^{K}\log\left|\det\frac{\partial z_k}{\partial z_{k-1}}\right|,\qquad z_k=f_k(z_{k-1})
$$

Sampling direction (same density):

$$
\log p_X\big(g(z)\big)=\log p_Z(z)-\log\left|\det J_g(z)\right|
$$

- ← $J_f(g(z))=J_g(z)^{-1}$ (inverse function theorem).

Training:

$$
\min_\theta\ -\mathbb{E}_{x\sim p_\text{data}}\left[\log p_X(x)\right]
$$

- ⇔ $\min_\theta\text{KL}(p_\text{data}\|p_X)$ (forward KL).
```

```{tip} Derivation
:class: dropdown
*Why does the Jacobian determinant appear?*
1. 1-D: probability mass is conserved → $p_X(x)\,|dx|=p_Z(z)\,|dz|$.
2. → $p_X(x)=p_Z(f(x))\,|f'(x)|$.
3. $d$-D: linearize $f$ near $x$ → a tiny box of volume $\Delta V$ maps to volume $|\det J_f(x)|\,\Delta V$.
4. Mass conserved → $p_X(x)=p_Z(f(x))\,|\det J_f(x)|$.

*Why does the log-det of a stack become a sum?*
1. Chain rule: $J_{f_K\circ\dots\circ f_1}=J_{f_K}\cdots J_{f_1}$.
2. $\det(AB)=\det A\det B$ → $\log|\det|$ of a product = sum of $\log|\det|$.
```

````{important} Code
:class: dropdown
```python
import math
import torch
import torch.nn as nn

class NormalizingFlow(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)    ## each: forward(x) -> (z, log|det|), inverse(z) -> x

    def forward(self, x):                      ## x -> z, summed log|det J_f|
        logdet = torch.zeros(x.shape[:-1])
        for f in self.layers:
            x, ld = f(x)
            logdet = logdet + ld
        return x, logdet

    def log_prob(self, x):
        z, logdet = self(x)
        ## change of variables: log p_X(x) = log N(f(x); 0, I) + log|det J_f(x)|
        log_pz = -0.5 * (z ** 2).sum(-1) - 0.5 * z.shape[-1] * math.log(2 * math.pi)
        return log_pz + logdet

    @torch.no_grad()
    def sample(self, n, d):
        z = torch.randn(n, d)
        for f in reversed(self.layers):        ## generator g = inverse layers in reverse order
            z = f.inverse(z)
        return z

## Example: 1 elementwise affine layer z = (x - mu) / sigma -> model = diagonal Gaussian
class Affine(nn.Module):
    def __init__(self, mu, log_sigma):
        super().__init__()
        self.mu, self.log_sigma = mu, log_sigma

    def forward(self, x):
        return (x - self.mu) * (-self.log_sigma).exp(), (-self.log_sigma.sum()).expand(x.shape[:-1])

    def inverse(self, z):
        return z * self.log_sigma.exp() + self.mu

mu, log_sigma = torch.tensor([1.0, -2.0]), torch.tensor([0.5, 0.0])
flow = NormalizingFlow([Affine(mu, log_sigma)])
x = torch.randn(3, 2)
ref = torch.distributions.Normal(mu, log_sigma.exp()).log_prob(x).sum(-1)
print(torch.allclose(flow.log_prob(x), ref, atol=1e-6))   ## True
print(flow.sample(4, 2).shape)                             ## torch.Size([4, 2])
```
````

```{dropdown} Table: Flow Families
| Family | Example | Jacobian | log-det cost | Density | Sampling |
|:--|:--|:--|:--|:--|:--|
| Planar | Planar flow (Q&A below) | Identity + rank 1 | $O(d)$ | ❌closed-form inverse | 1 pass |
| Coupling | [RealNVP](#realnvp), [Glow](#glow) | Block triangular | $O(d)$ | 1 pass | 1 pass |
| Autoregressive | [MAF](#maf) | Triangular | $O(d)$ | 1 pass | $d$ passes |
| Inverse autoregressive | [IAF](#iaf) | Triangular | $O(d)$ | $d$ passes | 1 pass |
| Invertible linear | [Glow](#glow) 1×1 conv | Dense $c\times c$ | $O(c^3)$, $O(c)$ w/ LU | 1 pass | 1 pass |
| Continuous | [CNF](#cnf), [FFJORD](#ffjord) | Trace of $\partial f/\partial z$ | $O(d)$ per eval w/ Hutchinson | ODE solve | ODE solve |

- $c$: #channels.
```

```{attention} Q&A
:class: dropdown
*Who introduced flows?*
- Composed invertible maps to a Gaussian for density estimation: Tabak & Vanden-Eijnden (2010), Tabak & Turner (2013).
- Popularized in DL by Rezende & Mohamed (VI, 2015) & Dinh et al. (NICE, density estimation).

*Why "normalizing"?*
- $f$ maps data → standard normal. "Flow" = the composition of many layers that the density flows through.

*Pros?*
- Exact log-likelihood → model comparison, bits/dim.
- Exact inference $z=f(x)$ → ❌amortization gap, ❌posterior collapse.
- 1-pass sampling (coupling flows).
- Invertible → activations recomputable from outputs → activation memory need not grow w/ depth.

*Cons?*
- $\dim z=\dim x$ → ❌compression, ❌low-dim latent.
- Invertible + cheap det → restricted layers → many layers & params for a given quality.
- Image sample quality < GAN & diffusion.

*Why can't a flow match every distribution exactly?*
- $g$ = continuous bijection w/ continuous inverse → preserves topology.
- Gaussian base = connected support → ❌exactly disjoint modes. Flow fakes them w/ thin low-density bridges → steep Jacobians.
- Data on a low-dim manifold → density degenerate in $\mathbb{R}^d$ → full-dim flow can only approximate.

*How do you train a flow on discrete pixels?*
- **Dequantization**: add uniform noise $u\in[0,1)^d$ to integer pixel values → continuous data.
- Continuous log-likelihood of $x+u$ lower-bounds the discrete log-likelihood (Jensen) → no density spikes on the integer grid.
- Report bits/dim $=-\log_2p(x)/d$.

*Can flows detect out-of-distribution inputs by likelihood?*
- ❌. Glow trained on CIFAR-10 assigns higher likelihood to SVHN {cite:p}`nalisnick2019do`.
- ← Likelihood is dominated by low-level statistics (smoothness, local variance), ❌semantics.

*Forward vs reverse KL use?*
- Density estimation: MLE = forward KL → mode covering. Needs fast density → [MAF](#maf).
- Variational inference: flow = approximate posterior $q$, reverse KL via the ELBO → needs fast sampling + density of own samples → [IAF](#iaf), planar flows.

*What are planar flows?*
- Rezende & Mohamed's layer; their VI paper popularized flows in DL {cite:p}`rezende2015variational`.
- $g(z)=z+u\,h(w^Tz+b)$, $h$ = smooth nonlinearity.
- $\det J_g=1+u^T\,h'(w^Tz+b)\,w$ (matrix determinant lemma) → $O(d)$.
- $h=\tanh$: invertible if $w^Tu\ge-1$. ❌Closed-form inverse → VI only.
- 1 hidden unit per layer → weak → need many layers.
```

&nbsp;

## Coupling Flows
- **What**: Flow layers that transform part of the dims conditioned on the rest → triangular Jacobian, free inverse.

&nbsp;

### RealNVP
- **Name**: Real-valued Non-Volume Preserving {cite:p}`dinh2017density`
- **What**: Flow of affine coupling layers: copy 1 part of $x$, scale & shift the other part conditioned on the copy.
- **Why**: Need an expressive layer w/ a free inverse & an $O(d)$ log-det.
    - Trick: scale & shift nets $s,t$ are arbitrary NNs → never inverted, never differentiated for the det.
- **How**:
    1. Split $x$ → $(x_A,x_B)$ via a binary mask (checkerboard or channel-wise for images).
    2. Copy: $z_A=x_A$.
    3. Transform: $z_B=x_B\odot\exp(s(x_A))+t(x_A)$.
    4. Jacobian triangular → log-det = sum of $s$.
    5. Alternate the copied part across layers → every dim gets transformed.
    6. Images: **squeeze** ($2\times2\times c\to1\times1\times4c$) + **multi-scale**: factor out half the dims at each scale → model them directly by the base Gaussian.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $x_A\in\mathbb{R}^{d_A}$: Copied part.
    - $x_B\in\mathbb{R}^{d-d_A}$: Transformed part.
- Params:
    - $s,t:\mathbb{R}^{d_A}\to\mathbb{R}^{d-d_A}$: Log-scale & shift nets.

Forward ($x\to z$):

$$
z_A=x_A,\qquad z_B=x_B\odot\exp\big(s(x_A)\big)+t(x_A)
$$

Inverse ($z\to x$):

$$
x_A=z_A,\qquad x_B=\big(z_B-t(z_A)\big)\odot\exp\big(-s(z_A)\big)
$$

Log-det:

$$
\log\left|\det J_f(x)\right|=\sum_{j=1}^{d-d_A}s_j(x_A)
$$
```

```{tip} Derivation
:class: dropdown
*Why is the log-det just $\sum_js_j$?*
1. Order the dims as $(A,B)$. Jacobian:

    $$
    J_f=\begin{bmatrix}I_{d_A}&0\\\dfrac{\partial z_B}{\partial x_A}&\text{diag}\big(\exp(s(x_A))\big)\end{bmatrix}
    $$

2. $z_A$ ignores $x_B$ → upper-right block $=0$ → block lower triangular.
3. Det of a block triangular matrix = product of the diagonal-block dets $=1\cdot\prod_j\exp(s_j)$.
4. $\partial z_B/\partial x_A$ (the complex part, through the NNs) never enters the det.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class AffineCoupling(nn.Module):
    def __init__(self, dim, hidden=64, flip=False):     ## dim must be even
        super().__init__()
        self.flip = flip
        ## s, t: any NN; never inverted -> no architectural constraint
        self.net = nn.Sequential(nn.Linear(dim // 2, hidden), nn.ReLU(), nn.Linear(hidden, dim))

    def _split(self, x):
        a, b = x.chunk(2, dim=-1)
        return (b, a) if self.flip else (a, b)           ## flip -> other half is copied

    def _merge(self, xa, xb):
        return torch.cat((xb, xa) if self.flip else (xa, xb), dim=-1)

    def _st(self, xa):
        s, t = self.net(xa).chunk(2, dim=-1)
        return torch.tanh(s), t                          ## bounded log-scale -> stable training

    def forward(self, x):                                ## x -> z
        xa, xb = self._split(x)
        s, t = self._st(xa)
        return self._merge(xa, xb * s.exp() + t), s.sum(-1)   ## log|det| = sum of log-scales

    def inverse(self, z):                                ## z -> x: same nets, no NN inversion
        za, zb = self._split(z)
        s, t = self._st(za)
        return self._merge(za, (zb - t) * (-s).exp())

## Example
layer = AffineCoupling(4)
x = torch.randn(4)
z, logdet = layer(x)
J = torch.autograd.functional.jacobian(lambda v: layer(v)[0], x)   ## (4, 4), block lower triangular
print(torch.allclose(torch.linalg.slogdet(J)[1], logdet, atol=1e-5))   ## True: matches sum(s)
print(torch.allclose(layer.inverse(z), x, atol=1e-5))                  ## True: exact inverse
```
````

```{attention} Q&A
:class: dropdown
*Why "non-volume preserving"?*
- Predecessor **NICE**: additive coupling $z_B=x_B+t(x_A)$ → $\det=1$ → volume preserving {cite:p}`dinh2015nice`.
    - Needs a final diagonal scaling layer to change volume at all.
- RealNVP: $\exp(s(x_A))$ → volume change depends on the input → more expressive.

*Why $\exp(s)$, ❌a raw scale?*
- $\exp>0$ → never 0 → always invertible.
- log-det $=\sum s$ directly → no $\log$ of the net output.

*Why alternate the copied half?*
- 1 coupling layer = identity on $x_A$.
- Alternate masks or permute between layers → every dim gets transformed & every dim conditions on every other after a few layers.

*Limitation of 1 layer?*
- $z_B$ affine in $x_B$ → 1 layer cannot bend the distribution of $x_B$ given $x_A$ → need deep stacks.

*Coupling vs autoregressive?*
- Coupling = autoregressive flow w/ 2 blocks: $x_B$ conditions on $x_A$ only.
- → Density & sampling both 1 pass. Less expressive per layer than [MAF](#maf).
- MAF paper: RealNVP coupling = special case of MAF.

*Training tricks?*
- Bounded $s$ (e.g., $\tanh$) → avoids $\exp$ overflow.
- Zero-init last layer of $s,t$ → each layer = identity at init.
- RealNVP uses batch norm as an invertible layer w/ its own log-det → [Glow](#glow) replaces it w/ actnorm.
```

&nbsp;

### Glow
- **What**: RealNVP-style flow; each step = actnorm → invertible 1×1 conv → affine coupling. {cite:p}`kingma2018glow`
- **Why**: Fix 2 RealNVP weak points.
    - Fixed channel permutations (reverse, shuffle) → weak mixing between coupling layers.
    - Batch norm → depends on the minibatch → noisy at the tiny batch sizes that large images force.
- **How**:
    1. **ActNorm**: per-channel scale & bias. Init from the 1st batch → zero mean, unit variance. Then plain trainable params → ❌batch dependence.
    2. **Invertible 1×1 conv**: learned $c\times c$ matrix mixes channels at every pixel. Generalizes a permutation.
    3. **Affine coupling** ([RealNVP](#realnvp)) on channel halves, conv nets for $s,t$.
    4. Stack $K$ steps per scale. $L$ scales w/ squeeze & split (multi-scale, as RealNVP).
    - Sampling at reduced temperature ($z\sim\mathcal{N}(0,T^2I)$, $T<1$) → quality ⬆️, diversity ⬇️.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $x\in\mathbb{R}^{c\times h\times w}$: Activation tensor (channels × height × width).
- Params:
    - $\mathbf{s},\mathbf{b}\in\mathbb{R}^{c}$: ActNorm scale & bias.
    - $M\in\mathbb{R}^{c\times c}$: 1×1 conv weight.

ActNorm:

$$
y_{:,u,v}=\mathbf{s}\odot x_{:,u,v}+\mathbf{b},\qquad\log|\det|=h\,w\sum_{j=1}^{c}\log|s_j|
$$

- $(u,v)$: Pixel position.

Invertible 1×1 conv:

$$
y_{:,u,v}=M\,x_{:,u,v},\qquad\log|\det|=h\,w\,\log|\det M|
$$

- Same $M$ at all $hw$ pixels → Jacobian = block diagonal w/ $hw$ copies of $M$.
- $\det M$: $O(c^3)$.

LU parametrization:

$$
M=PL\big(U+\text{diag}(\mathbf{s}_M)\big),\qquad\log|\det M|=\sum_{j=1}^{c}\log|s_{M,j}|
$$

- $P$: Fixed permutation.
- $L$: Lower triangular, unit diagonal.
- $U$: Upper triangular, zero diagonal.
- $\mathbf{s}_M\in\mathbb{R}^c$: Diagonal entries. log-det → $O(c)$.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class ActNorm(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.loc = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.log_scale = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.initialized = False

    def forward(self, x):                                ## x: (B, C, H, W)
        if not self.initialized:                         ## data-dependent init on the 1st batch only
            with torch.no_grad():
                self.loc.copy_(-x.mean((0, 2, 3), keepdim=True))
                self.log_scale.copy_(-x.std((0, 2, 3), keepdim=True).log())
            self.initialized = True
        z = (x + self.loc) * self.log_scale.exp()        ## after init: plain affine, no batch stats
        return z, (x.shape[2] * x.shape[3] * self.log_scale.sum()).expand(x.shape[0])

class InvConv1x1(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.M = nn.Parameter(torch.linalg.qr(torch.randn(c, c))[0])   ## random rotation -> log|det| = 0 at init

    def forward(self, x):
        z = torch.einsum('ij,bjhw->bihw', self.M, x)     ## same C x C mix at every pixel
        logdet = x.shape[2] * x.shape[3] * torch.linalg.slogdet(self.M)[1]
        return z, logdet.expand(x.shape[0])

    def inverse(self, z):
        return torch.einsum('ij,bjhw->bihw', torch.linalg.inv(self.M), z)

## Example: 1 Glow step = ActNorm -> InvConv1x1 -> AffineCoupling on channel halves (coupling omitted)
x = 3 * torch.randn(2, 4, 8, 8) + 1
z, _ = ActNorm(4)(x)
print(z.mean((0, 2, 3)), z.std((0, 2, 3)))          ## ≈ [0, 0, 0, 0], ≈ [1, 1, 1, 1]
conv = InvConv1x1(4)
y, logdet = conv(z)
print(torch.allclose(conv.inverse(y), z, atol=1e-5), logdet)   ## True, ≈ [0, 0] (orthogonal init)
```
````

```{attention} Q&A
:class: dropdown
*Why a learned 1×1 conv, ❌a fixed permutation?*
- Permutation = special case of $M$ (a permutation matrix).
- Learned $M$ → lower NLL & faster convergence than reverse or random permutations (Glow ablation).

*Why ActNorm, ❌BatchNorm?*
- BatchNorm: output depends on other batch items. Noisy at batch size 1 per GPU. Train/test behavior differs.
- ActNorm: batch stats only for init → then a fixed affine layer.

*Additive or affine coupling?*
- Glow ablation: affine → better NLL & faster convergence than additive.

*What does the exact latent space give you?*
- Encode real images exactly → interpolate, or add attribute directions (e.g., "smile" = mean $z$ of smiling − mean $z$ of non-smiling).

*Cons?*
- Very deep & wide models for high-res images → heavy compute & memory.
- Sample quality < GAN & diffusion at equal compute.
```

&nbsp;

## Autoregressive Flows
- **What**: Flow where dim $j$ = affine function of a noise variable, w/ shift & scale conditioned on earlier dims → triangular Jacobian.

&nbsp;

### MAF
- **Name**: Masked Autoregressive Flow {cite:p}`papamakarios2017masked`
- **What**: Stack of Gaussian autoregressive layers; shift & scale of $x_j$ conditioned on $x_{<j}$ by a masked NN.
- **Why**: Gaussian autoregressive model = 1 flow layer → stack layers → more flexible than 1 autoregressive model.
    - Masking → all conditionals in 1 pass → fast density → good for density estimation.
- **How**:
    1. **MADE** conditioner: MLP w/ binary weight masks → output $j$ sees only inputs $<j$.
    2. **Density**: $x$ known → all $\mu_j,\alpha_j$ in 1 pass → all $u_j$ in parallel.
    3. **Sampling**: $x_j$ needs $x_{<j}$ → $d$ sequential passes.
    4. Stack layers. Reverse or permute the dim order between layers.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $u\in\mathbb{R}^d$: Layer's noise variable ($=z$ for a 1-layer flow).
- Params:
    - $h_\theta$: Masked autoregressive conditioner (MADE).
- Misc:
    - $\mu_j,\alpha_j$: Shift & log-scale of dim $j$.

Model (sampling, sequential):

$$
x_j=u_j\exp(\alpha_j)+\mu_j,\qquad(\mu_j,\alpha_j)=h_\theta(x_{<j})
$$

Inference (density, parallel):

$$
u_j=(x_j-\mu_j)\exp(-\alpha_j),\qquad\log p_X(x)=\log p_Z(u)-\sum_{j=1}^{d}\alpha_j
$$
```

```{tip} Derivation
:class: dropdown
*Why is the log-det $-\sum_j\alpha_j$?*
1. $u_j$ depends on $x_j$ & $x_{<j}$ only → $\partial u_j/\partial x_{j'}=0$ for $j'>j$ → $\partial u/\partial x$ lower triangular.
2. Diagonal: $\partial u_j/\partial x_j=\exp(-\alpha_j)$ ← $\mu_j,\alpha_j$ do not depend on $x_j$.
3. Triangular det = product of diagonal → $\log|\det|=-\sum_j\alpha_j$.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class MaskedAffine(nn.Module):
    ## 1-layer stand-in for MADE: strictly lower-triangular weights -> output j sees inputs < j only
    def __init__(self, d):
        super().__init__()
        self.W_mu = nn.Parameter(0.1 * torch.randn(d, d))
        self.W_alpha = nn.Parameter(0.1 * torch.randn(d, d))
        self.register_buffer('mask', torch.tril(torch.ones(d, d), diagonal=-1))

    def forward(self, h):
        return h @ (self.W_mu * self.mask).T, h @ (self.W_alpha * self.mask).T

class MAF(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.cond = MaskedAffine(d)

    def forward(self, x):                      ## density x -> u: 1 parallel pass
        mu, alpha = self.cond(x)               ## conditions on x_{<j}, all known
        return (x - mu) * (-alpha).exp(), -alpha.sum(-1)

    @torch.no_grad()
    def inverse(self, u):                      ## sampling u -> x: d sequential passes
        x = torch.zeros_like(u)
        for j in range(u.shape[-1]):
            mu, alpha = self.cond(x)           ## x_{<j} already filled in
            x[..., j] = u[..., j] * alpha[..., j].exp() + mu[..., j]
        return x

## Example
maf = MAF(3)
x = torch.randn(2, 3)
u, logdet = maf(x)
print(torch.allclose(maf.inverse(u), x, atol=1e-5), logdet.shape)   ## True torch.Size([2])
```
````

```{attention} Q&A
:class: dropdown
*Why stack autoregressive models?*
- 1 Gaussian autoregressive model → each conditional $p(x_j|x_{<j})$ unimodal Gaussian.
- Next layer models the noise $u$ of the previous layer → non-Gaussian conditionals.

*Why does the dim order matter?*
- Each order = a different factorization → different inductive bias.
- Change the order between layers → no dim stays stuck at the start.

*Pros?*
- Density 1 pass → fast MLE training.
- Each dim conditions on all earlier dims → more expressive per layer than coupling.

*Cons?*
- Sampling $d$ sequential passes → slow generation, poor fit for VI.

*MAF vs RealNVP?*
- Coupling = MAF where $x_B$ conditions only on $x_A$ & $x_A$ passes unchanged → MAF generalizes RealNVP.
```

&nbsp;

### IAF
- **Name**: Inverse Autoregressive Flow {cite:p}`kingma2016improved`
- **What**: Autoregressive flow conditioned on the noise $u_{<j}$, ❌the data → parallel sampling.
- **Why**: VAE posterior $q(z|x)$ needs flexibility + fast sampling + density of its own samples.
    - Diagonal Gaussian posterior → loose ELBO.
- **How**:
    1. Sample $u\sim\mathcal{N}(0,I)$ (or the encoder's diagonal Gaussian).
    2. 1 pass: $(\mu,\alpha)=h_\theta(u)$ w/ masking → all $x_j=u_j\exp(\alpha_j)+\mu_j$ in parallel.
    3. Density of own sample: $u$ known → log-det $=\sum_j\alpha_j$ free.
    4. Density of an external $x$: recover $u$ → $u_j$ needs $u_{<j}$ → $d$ sequential passes.
    - In a [VAE](vae.md#vae): the encoder also emits a context vector → extra input to every IAF conditioner.

```{note} Math
:class: dropdown
Model (sampling, parallel):

$$
x_j=u_j\exp(\alpha_j)+\mu_j,\qquad(\mu_j,\alpha_j)=h_\theta(u_{<j})
$$

Density:

$$
\log p_X(x)=\log p_Z(u)-\sum_{j=1}^{d}\alpha_j
$$

- $u,h_\theta,\mu_j,\alpha_j$: as in [MAF](#maf).

Gated form (Kingma et al., numerically stable):

$$
x=\boldsymbol{\gamma}\odot u+(1-\boldsymbol{\gamma})\odot\mathbf{m},\qquad\boldsymbol{\gamma}=\text{sigmoid}(\mathbf{r}),\qquad(\mathbf{m},\mathbf{r})=h_\theta(u)
$$

- $\boldsymbol{\gamma}\in(0,1)^d$: Gate. log-det $=\sum_j\log\gamma_j$.
- $\mathbf{m},\mathbf{r}\in\mathbb{R}^d$: Autoregressive NN outputs; entry $j$ depends on $u_{<j}$ only.
```

````{important} Code
:class: dropdown
```python
import math
import torch
import torch.nn as nn

## Reuses MaskedAffine from MAF above
class IAF(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.cond = MaskedAffine(d)

    def sample(self, u):                       ## u -> x: 1 parallel pass
        mu, alpha = self.cond(u)               ## conditions on u_{<j}, all known up front
        x = u * alpha.exp() + mu
        ## density of OWN sample is free: u is cached, no inversion needed
        log_pu = -0.5 * (u ** 2).sum(-1) - 0.5 * u.shape[-1] * math.log(2 * math.pi)
        return x, log_pu - alpha.sum(-1)

## Example
x, log_q = IAF(3).sample(torch.randn(2, 3))
print(x.shape, log_q.shape)   ## torch.Size([2, 3]) torch.Size([2])
```
````

```{dropdown} Table: MAF vs IAF
| | MAF | IAF |
|:--|:--|:--|
| Conditioner input | Data $x_{<j}$ | Noise $u_{<j}$ |
| Density of any $x$ | ✅ 1 pass | ❌ $d$ passes |
| Sampling | ❌ $d$ passes | ✅ 1 pass |
| Density of own sample | 1 extra pass | ✅ free ($u$ cached) |
| MLE on data | ✅ cheap | ❌ slow |
| Typical use | Density estimation | VI posterior, fast generation |
```

```{attention} Q&A
:class: dropdown
*Why "inverse"?*
- IAF's sampling map = inverse of an autoregressive model's normalizing map.
- Same equations as MAF w/ the roles of $x$ & $u$ swapped → MAF density ⇔ IAF sampling.

*Why does IAF suit VAEs?*
- ELBO needs $\log q(z|x)$ only at samples drawn from $q$ → never an external point.
- Sampling is 1 pass & reparameterizable ([Reparameterization Trick](vae.md#reparameterization-trick)).

*How do you get fast sampling & a good data fit?*
- Train MAF / autoregressive teacher by MLE (fast density).
- Distill into an IAF student by reverse KL on the student's own samples (fast sampling).
- Example: Parallel WaveNet ("probability density distillation").

*Cons?*
- ❌Direct MLE on data → $d$ sequential passes per example.
```

&nbsp;

## Continuous-Time
- **What**: Flows defined by an ODE: layer index → continuous time.

&nbsp;

### Neural ODE
- **Name**: Neural Ordinary Differential Equation {cite:p}`chen2018neural`
- **What**: State evolves by an ODE w/ an NN vector field. Output = ODE solution at the end time.
- **Why**: ResNet update $z_{k+1}=z_k+f(z_k)$ = Euler step of $dz/dt=f(z,t)$ → take the continuous limit.
    - Adaptive ODE solver → compute set by error tolerance, ❌fixed depth.
    - Adjoint backprop → memory $O(1)$ in #solver steps.
    - Continuous time → irregularly sampled time series & continuous flows ([CNF](#cnf)).
- **How**:
    1. **Forward**: black-box ODE solver integrates $dz/dt=f_\theta(z,t)$ from $t_0$ to $t_1$.
    2. **Backward (adjoint)**: solve a 2nd ODE backward in time for $a(t)=\partial\mathcal{L}/\partial z(t)$. Recompute $z(t)$ backward along the way → ❌stored forward activations.
    3. Param grad = integral of $a^T\partial f/\partial\theta$ along the backward solve.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $z(t)\in\mathbb{R}^d$: ODE state. $z(t_0)$ = input, $z(t_1)$ = output.
- Params:
    - $f_\theta(z,t)$: NN vector field.
- Misc:
    - $a(t)=\partial\mathcal{L}/\partial z(t)\in\mathbb{R}^{1\times d}$: Adjoint (row vector).

Forward:

$$
z(t_1)=z(t_0)+\int_{t_0}^{t_1}f_\theta\big(z(t),t\big)\,dt
$$

Backward (adjoint):

$$\begin{align*}
&\frac{da(t)}{dt}=-a(t)\,\frac{\partial f_\theta(z(t),t)}{\partial z},\qquad a(t_1)=\frac{\partial\mathcal{L}}{\partial z(t_1)} \\
&\frac{\partial\mathcal{L}}{\partial z(t_0)}=a(t_0),\qquad\frac{\partial\mathcal{L}}{\partial\theta}=\int_{t_0}^{t_1}a(t)\,\frac{\partial f_\theta(z(t),t)}{\partial\theta}\,dt
\end{align*}$$

- Solve $[z,a,\partial\mathcal{L}/\partial\theta]$ jointly from $t_1$ back to $t_0$.
- Each step needs only vector-Jacobian products $a\,\partial f/\partial z$, $a\,\partial f/\partial\theta$ → 1 reverse-mode call.
```

```{tip} Derivation
:class: dropdown
*Why does the adjoint obey $da/dt=-a\,\partial f/\partial z$?*
1. Small step: $z(t+\varepsilon)=z(t)+\varepsilon f(z(t),t)+O(\varepsilon^2)$.
2. Chain rule: $a(t)=a(t+\varepsilon)\,\dfrac{\partial z(t+\varepsilon)}{\partial z(t)}=a(t+\varepsilon)\left(I+\varepsilon\dfrac{\partial f}{\partial z}\right)+O(\varepsilon^2)$.
3. Rearrange: $\dfrac{a(t+\varepsilon)-a(t)}{\varepsilon}=-a(t+\varepsilon)\dfrac{\partial f}{\partial z}+O(\varepsilon)$.
4. $\varepsilon\to0$ → $\dfrac{da}{dt}=-a\dfrac{\partial f}{\partial z}$.

*Why is the param grad an integral?*
1. Treat $\theta$ as extra state w/ $d\theta/dt=0$.
2. Same rule → its adjoint $a_\theta$ obeys $da_\theta/dt=-a\,\partial f/\partial\theta$, $a_\theta(t_1)=0$.
3. → $a_\theta(t_0)=\int_{t_0}^{t_1}a\,\partial f/\partial\theta\,dt=\partial\mathcal{L}/\partial\theta$.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class NeuralODE(nn.Module):
    def __init__(self, f, n_steps=1000):
        super().__init__()
        self.f, self.n = f, n_steps            ## f(z, t): nn.Module vector field

    def forward(self, z, t0=0.0, t1=1.0):     ## fixed-step Euler; real code: adaptive solver
        h = (t1 - t0) / self.n
        for k in range(self.n):
            z = z + h * self.f(z, t0 + k * h)
        return z

    def adjoint(self, z1, a1, t0=0.0, t1=1.0):
        ## integrate [z, a, dL/dtheta] backward t1 -> t0; nothing stored from the forward pass
        h = (t1 - t0) / self.n
        z, a, t = z1.detach(), a1, t1
        g = [torch.zeros_like(p) for p in self.f.parameters()]
        for _ in range(self.n):
            with torch.enable_grad():
                zr = z.requires_grad_(True)
                fz = self.f(zr, t)
                ## a df/dz and a df/dtheta in 1 reverse-mode call
                vjp = torch.autograd.grad(fz, [zr, *self.f.parameters()], a)
            z = (z - h * fz).detach()                        ## re-solve z backward in time
            a = a + h * vjp[0]                               ## da/dt = -a df/dz, stepping t -> t - h
            g = [gi + h * vi for gi, vi in zip(g, vjp[1:])]  ## dL/dtheta = ∫ a df/dtheta dt
            t = t - h
        return a, g                                          ## dL/dz(t0), dL/dtheta

## Example: f = theta * z -> z(1) = z0 * exp(theta)
class Scale(nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(0.5))

    def forward(self, z, t):
        return self.theta * z

ode = NeuralODE(Scale())
with torch.no_grad():
    z1 = ode(torch.ones(3))                    ## no graph kept -> O(1) memory in #steps
a0, (g,) = ode.adjoint(z1, torch.ones(3))     ## L = z1.sum() -> dL/dz1 = 1
print(a0, g)   ## ≈ [1.649, 1.649, 1.649] = e^0.5, ≈ 4.946 = 3 e^0.5
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- Memory $O(1)$ in #solver steps (adjoint).
- Adaptive compute: tolerance trades accuracy for speed at test time.
- Flow map invertible by construction ← unique ODE solutions.

*Cons?*
- Slow: #function evaluations (NFE) often grows during training as the dynamics get complex.
- Adjoint re-solves $z(t)$ backward → numerical error for unstable/stiff dynamics → inaccurate grads.
- Limited expressivity: trajectories cannot cross.

*Why can't trajectories cross? What breaks?*
- Lipschitz $f$ → unique solution through each point → 2 trajectories that meet must coincide.
- → 1-D flow map is increasing → ❌$x\mapsto-x$.
- Fix: augment the state w/ extra dims → room to go around {cite:p}`dupont2019augmented`.

*Adjoint vs backprop through the solver?*
- Through the solver: exact grad of the discretized forward. Memory $O(\text{NFE})$.
- Adjoint: memory $O(1)$, ~1 extra solve. Grad of the continuous problem → ≈ grad of the discretized forward.

*Relation to ResNet?*
- ResNet = Euler w/ step 1 & separate weights per step.
- Neural ODE = 1 set of weights (optionally $t$-dependent), continuous depth.
```

&nbsp;

### CNF
- **Name**: Continuous Normalizing Flow {cite:p}`chen2018neural`
- **What**: Normalizing flow defined by a Neural ODE. Log-density changes at rate $-\text{tr}(\partial f/\partial z)$.
- **Why**: Discrete flows need special architectures for a cheap det.
    - Continuous time → log-det becomes a trace → any NN vector field works.
- **How**:
    1. $z(0)\sim p_Z$ → integrate $dz/dt=f_\theta(z,t)$ → $x=z(1)$.
    2. Log-density along the path changes by $-\text{tr}(\partial f/\partial z)\,dt$.
    3. Density of $x$: integrate backward $1\to0$ jointly w/ the trace integral.
    4. Train by MLE (ODE solves per step), or simulation-free by [Flow Matching](diffusion.md#flow-matching).

```{note} Math
:class: dropdown
Notations:
- IO:
    - $z(t)\in\mathbb{R}^d$: Flow state. $z(0)\sim p_Z$, $z(1)=x$.
- Params:
    - $f_\theta(z,t)$: NN vector field.
- Override: $t=0$ base, $t=1$ data. [Flow Matching](diffusion.md#flow-matching) uses the reverse: $t=0$ data, $t=1$ noise.

Instantaneous change of variables:

$$
\frac{d\log p\big(z(t)\big)}{dt}=-\text{tr}\left(\frac{\partial f_\theta}{\partial z}\big(z(t),t\big)\right)
$$

Density:

$$
\log p_X(x)=\log p_Z\big(z(0)\big)-\int_0^1\text{tr}\left(\frac{\partial f_\theta}{\partial z}\big(z(t),t\big)\right)dt
$$

- Exact trace: $d$ reverse-mode passes per function evaluation → [FFJORD](#ffjord).
```

```{tip} Derivation
:class: dropdown
*Why a trace, ❌a determinant?*
1. 1 Euler step = discrete flow: $z(t+\varepsilon)=z(t)+\varepsilon f$ → Jacobian $I+\varepsilon\,\partial f/\partial z$.
2. Discrete change of variables: $\log p(z(t+\varepsilon))=\log p(z(t))-\log|\det(I+\varepsilon\,\partial f/\partial z)|$.
3. $\det(I+\varepsilon A)=1+\varepsilon\,\text{tr}(A)+O(\varepsilon^2)$ & $\log(1+u)=u+O(u^2)$.
4. → $\log p(z(t+\varepsilon))-\log p(z(t))=-\varepsilon\,\text{tr}(\partial f/\partial z)+O(\varepsilon^2)$.
5. ÷$\varepsilon$, $\varepsilon\to0$ → the ODE above.
```

````{important} Code
:class: dropdown
```python
import math
import torch
import torch.nn as nn

class CNF(nn.Module):
    def __init__(self, f, n_steps=1000):
        super().__init__()
        self.f, self.n = f, n_steps            ## f(z, t): vector field, no mixing across the batch

    def div(self, fz, z):                      ## exact trace: d reverse-mode passes
        return sum(torch.autograd.grad(fz[:, j].sum(), z, create_graph=True)[0][:, j]
                   for j in range(z.shape[-1]))

    def log_prob(self, x):
        ## data x at t=1 -> base z at t=0, accumulating the trace integral
        h = 1.0 / self.n
        z = x.detach().requires_grad_(True)
        trace = torch.zeros(x.shape[0])
        for k in range(self.n):
            fz = self.f(z, 1.0 - k * h)
            trace = trace + h * self.div(fz, z)
            z = z - h * fz                     ## Euler step t -> t - h
        log_pz = -0.5 * (z ** 2).sum(-1) - 0.5 * z.shape[-1] * math.log(2 * math.pi)
        return log_pz - trace                  ## log p(x) = log p(z(0)) - ∫ tr(df/dz) dt

## Example: linear field f = A z -> x = exp(A) z(0), exact density known
A = torch.tensor([[-1.0, 2.0], [0.0, -0.5]])
x = torch.randn(2, 2)
z0 = x @ torch.linalg.matrix_exp(-A).T
exact = -0.5 * (z0 ** 2).sum(-1) - math.log(2 * math.pi) - A.trace()   ## log|det exp(-A)| = -tr(A)
print(CNF(lambda z, t: z @ A.T).log_prob(x), exact)   ## agree to ~1e-2 (Euler error)
```
````

```{attention} Q&A
:class: dropdown
*CNF vs discrete flow?*
- ✅ Free-form $f$: no coupling, no masking, no triangular structure.
- ✅ Same model computes density & samples (ODE in either direction).
- ❌ Every density or sample = an ODE solve → cost scales w/ NFE.

*Why is MLE training of CNFs slow?*
- Each step solves the ODE + trace integral, then an adjoint solve.
- Learned dynamics get stiffer → NFE ⬆️ during training.
- → [Flow Matching](diffusion.md#flow-matching): regress $f_\theta$ onto a known path velocity → ❌ODE solve during training.

*How does this relate to diffusion?*
- Diffusion's probability-flow ODE = a CNF w/ a score-derived vector field.
- Exact diffusion likelihood = this formula + Hutchinson trace → [Score SDE](diffusion.md#score-sde).
```

&nbsp;

#### FFJORD
- **Name**: Free-form Jacobian of Reversible Dynamics {cite:p}`grathwohl2019ffjord`
- **What**: CNF w/ Hutchinson's unbiased trace estimator → 1 vector-Jacobian product per function eval.
- **Why**: Exact trace = $d$ reverse-mode passes per eval → ❌scales to images.
- **How**:
    1. Sample $\epsilon$ w/ $\mathbb{E}[\epsilon]=0$, $\text{Cov}(\epsilon)=I$ (Gaussian or Rademacher). Fix it for the whole ODE solve.
    2. $\epsilon^T\,\partial f/\partial z$ = 1 vector-Jacobian product = 1 reverse-mode pass.
    3. Dot w/ $\epsilon$ → unbiased trace estimate.
    4. Train w/ the [adjoint](#neural-ode) → memory $O(1)$ in #solver steps.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\epsilon\in\mathbb{R}^d$: Probe vector, $\mathbb{E}[\epsilon]=0$, $\text{Cov}(\epsilon)=I$.

Hutchinson's estimator:

$$
\text{tr}(A)=\mathbb{E}_{\epsilon}\left[\epsilon^TA\,\epsilon\right]
$$

Density estimate:

$$
\log p_X(x)=\log p_Z\big(z(0)\big)-\mathbb{E}_\epsilon\left[\int_0^1\epsilon^T\frac{\partial f_\theta}{\partial z}\big(z(t),t\big)\,\epsilon\;dt\right]
$$

- Unbiased for $\log p_X$ → unbiased MLE gradient.
```

```{tip} Derivation
:class: dropdown
*Why is Hutchinson unbiased?*
1. $\mathbb{E}[\epsilon^TA\epsilon]=\sum_{i,j}A_{ij}\,\mathbb{E}[\epsilon_i\epsilon_j]$.
2. $\text{Cov}(\epsilon)=I$ & zero mean → $\mathbb{E}[\epsilon_i\epsilon_j]=\delta_{ij}$.
3. → $\sum_iA_{ii}=\text{tr}(A)$.

*Why a fixed $\epsilon$ per solve?*
- Fresh noise inside the solver → stochastic ODE → adaptive step control breaks.
- Fixed $\epsilon$ → deterministic ODE. Trace integral linear in the integrand → still unbiased.
```

````{important} Code
:class: dropdown
```python
import math
import torch

## Reuses CNF from CNF above
class FFJORD(CNF):
    def div(self, fz, z):
        ## eps^T (df/dz) in 1 reverse-mode pass, then dot with eps -> unbiased trace
        vjp = torch.autograd.grad(fz, z, self.eps, create_graph=True)[0]
        return (vjp * self.eps).sum(-1)

    def log_prob(self, x):
        self.eps = torch.randn_like(x)         ## fixed for the whole solve
        return super().log_prob(x)

## Example: same linear field as in CNF; tr(A) = -1.5
A = torch.tensor([[-1.0, 2.0], [0.0, -0.5]])
x = torch.randn(1000, 2)
diff = FFJORD(lambda z, t: z @ A.T, n_steps=100).log_prob(x) - CNF(lambda z, t: z @ A.T, n_steps=100).log_prob(x)
print(diff.mean())   ## ≈ 0: unbiased; per-sample spread ~ std of eps^T A eps
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- Free-form $f$ → ⬆️expressivity vs coupling & autoregressive flows.
- Cost per eval $O(d)$ vs $O(d^2)$ for the exact trace.

*Cons?*
- Estimator variance → noisy log-likelihood & grads.
- Many ODE solves → slow training & sampling.

*Gaussian or Rademacher probe?*
- Both unbiased.
- Rademacher ($\pm1$): variance $2(\|A_s\|_F^2-\sum_iA_{ii}^2)$ ≤ Gaussian: $2\|A_s\|_F^2$, $A_s=(A+A^T)/2$.

*How do you cut the variance further?*
- Bottleneck trick: $f=g\circ k$ w/ hidden width $H<d$ → $\text{tr}(\partial g/\partial k\cdot\partial k/\partial z)=\text{tr}(\partial k/\partial z\cdot\partial g/\partial k)$ → probe in $\mathbb{R}^H$ → can lower the variance, ❌guaranteed.
```

&nbsp;
