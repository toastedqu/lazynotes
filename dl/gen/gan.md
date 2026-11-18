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
# GAN
Generative models trained by an adversarial game between a generator and a discriminator. Covers the GAN objective, its failure modes, classic architectures, and Wasserstein GANs.

Notations:
- $x$: Real data sample, $x\sim p_\text{data}$
- $z$: Latent noise, $z\sim p_z$
- $p_\text{data}$: Data distribution
- $p_z$: Latent prior (e.g., $\mathcal{N}(0,I)$)
- $G$: Generator, $z\mapsto G(z)$
- $\tilde x=G(z)$: Generated (fake) sample
- $p_g$: Generator distribution = distribution of $G(z)$, $z\sim p_z$
- $D$: Discriminator, $x\mapsto D(x)\in[0,1]$ = P(real)
- $\theta$: Generator params
- $\phi$: Discriminator params

Overrides of global notation: $x$, $z$ = whole vectors (non-bold), ❌scalar. $D$, $G$ = networks, ❌matrix/random variable.

&nbsp;

## GAN
- **Name**: Generative Adversarial Network {cite:p}`goodfellow2014generative`
- **What**: 2-player minimax game: generator (noise → sample) vs discriminator (sample → P(real)).
    - → [Implicit density](gen.md#implicit-density) model: draws samples, ❌evaluates $p_g(x)$.
- **Why**: Likelihood-free generation.
    - Explicit density models (VAE, flow, autoregressive) need a tractable $p(x)$ or a bound → restricts the architecture.
    - GAN needs only samples from $G$ & a learned judge of realism → any differentiable $G$.
- **How**:
    1. $G$ maps $z\sim p_z$ to $\tilde x=G(z)$.
    2. $D$ = binary classifier: real $x$ → 1, fake $\tilde x$ → 0.
    3. $G$ updates to make $D(\tilde x)\to1$. Gradient flows through $D$ into $G$.
    4. Alternate: $k$ D steps → 1 G step. Paper: $k=1$.
    5. Equilibrium: $p_g=p_\text{data}$, $D\equiv\frac{1}{2}$.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $k$: #D steps per G step.

Inference:

$$
\tilde x=G(z),\quad z\sim p_z
$$

- 1 forward pass per sample.

Training:

$$
\min_G\max_D V(D,G)=\mathbb{E}_{x\sim p_\text{data}}\left[\log D(x)\right]+\mathbb{E}_{z\sim p_z}\left[\log\left(1-D(G(z))\right)\right]
$$

- $V$: Value function.

Optimal discriminator (fixed $G$):

$$
D^*_G(x)=\frac{p_\text{data}(x)}{p_\text{data}(x)+p_g(x)}
$$

Generator criterion at $D^*_G$:

$$
C(G)=\max_DV(D,G)=-\log4+2\,\text{JSD}(p_\text{data}\|p_g)
$$

- $\text{JSD}(P\|Q)=\frac{1}{2}D_\text{KL}(P\|M)+\frac{1}{2}D_\text{KL}(Q\|M)$, $M=\frac{P+Q}{2}$: [Jensen-Shannon divergence](../../nlp/post/kd.md#jsd).
- Global min $C(G)=-\log4$ iff $p_g=p_\text{data}$ → $D^*\equiv\frac{1}{2}$.

Losses (minimize):

$$
\begin{align*}
\mathcal{L}_D&=-\mathbb{E}_{x\sim p_\text{data}}\left[\log D(x)\right]-\mathbb{E}_{z\sim p_z}\left[\log\left(1-D(G(z))\right)\right]\\
\mathcal{L}_G^\text{sat}&=\mathbb{E}_{z\sim p_z}\left[\log\left(1-D(G(z))\right)\right]\\
\mathcal{L}_G^\text{NS}&=-\mathbb{E}_{z\sim p_z}\left[\log D(G(z))\right]
\end{align*}
$$

- $\mathcal{L}_D=-V$ = [BCE](../../ml/obj.md#bce) w/ labels real = 1, fake = 0.
- $\mathcal{L}_G^\text{sat}$: Saturating (minimax) G loss.
- $\mathcal{L}_G^\text{NS}$: Non-saturating G loss = BCE of fakes w/ label 1. Default in practice.

Process (1 iteration):
1. Repeat $k$ times: sample $\{z^{(i)}\}_{i=1}^m\sim p_z$ & $\{x^{(i)}\}_{i=1}^m\sim p_\text{data}$ → step $\phi$ to ⬇️$\mathcal{L}_D$.
2. Sample $\{z^{(i)}\}_{i=1}^m\sim p_z$ → step $\theta$ to ⬇️$\mathcal{L}_G^\text{NS}$.
```

```{tip} Derivation
:class: dropdown
*Why is $D^*_G=\frac{p_\text{data}}{p_\text{data}+p_g}$?*
1. Rewrite the fake term as an expectation over $\tilde x\sim p_g$:

    $$
    V(D,G)=\int p_\text{data}(x)\log D(x)+p_g(x)\log\left(1-D(x)\right)\,dx
    $$

2. Maximize pointwise in $x$: $a\log y+b\log(1-y)$ w/ $a=p_\text{data}(x)$, $b=p_g(x)$, $y=D(x)\in[0,1]$.
3. $\frac{a}{y}-\frac{b}{1-y}=0$ → $y^*=\frac{a}{a+b}$. Second derivative $-\frac{a}{y^2}-\frac{b}{(1-y)^2}<0$ → max.
    - $b=0$ → $y^*=1$. $a=0$ → $y^*=0$. Same formula → $D^*\in\{0,1\}$ on disjoint supports.
- Assumptions: $p_g$ has a density. $D$ = any function (infinite capacity).

*Why is $C(G)=-\log4+2\,\text{JSD}$?*
1. Plug in $D^*_G$:

    $$
    C(G)=\mathbb{E}_{x\sim p_\text{data}}\left[\log\frac{p_\text{data}(x)}{p_\text{data}(x)+p_g(x)}\right]+\mathbb{E}_{x\sim p_g}\left[\log\frac{p_g(x)}{p_\text{data}(x)+p_g(x)}\right]
    $$

2. Write $p_\text{data}+p_g=2M$ → each term $=D_\text{KL}(\cdot\|M)-\log2$.
3. → $C(G)=-\log4+D_\text{KL}(p_\text{data}\|M)+D_\text{KL}(p_g\|M)=-\log4+2\,\text{JSD}(p_\text{data}\|p_g)$.
4. $\text{JSD}\geq0$, $=0$ iff $p_g=p_\text{data}$ → unique global min.

*Why does $\mathcal{L}_G^\text{sat}$ saturate?*
1. Let $D(\tilde x)=\sigma(a)$. $\sigma$: sigmoid. $a$: D's logit on $\tilde x$.
2. Sat: $\frac{\partial}{\partial a}\log\left(1-\sigma(a)\right)=-\sigma(a)=-D(\tilde x)$.
3. NS: $\frac{\partial}{\partial a}\left[-\log\sigma(a)\right]=-\left(1-\sigma(a)\right)=-\left(1-D(\tilde x)\right)$.
4. Early training: fakes are obviously fake → $D(\tilde x)\approx0$.
    - Sat: $|\partial/\partial a|\approx0$ → $G$ learns almost nothing.
    - NS: $|\partial/\partial a|\approx1$ → strong signal.
5. Both push $D(\tilde x)$ up → same fixed point, different gradient scale.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GAN:
    def __init__(self, G, D, z_dim, lr=2e-4):
        self.G, self.D, self.z_dim = G, D, z_dim
        self.opt_g = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
        self.opt_d = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    def step(self, x):
        m = x.size(0)
        real, fake = torch.ones(m, 1), torch.zeros(m, 1)

        ## D step (k = 1): BCE w/ real = 1, fake = 0 <=> maximize V(D, G). D outputs logits
        x_fake = self.G(torch.randn(m, self.z_dim)).detach()  ## detach: D step must not update G
        loss_d = F.binary_cross_entropy_with_logits(self.D(x), real) \
               + F.binary_cross_entropy_with_logits(self.D(x_fake), fake)
        self.opt_d.zero_grad(); loss_d.backward(); self.opt_d.step()

        ## G step, non-saturating: -log D(G(z)) = BCE of fakes against label "real"
        x_fake = self.G(torch.randn(m, self.z_dim))
        loss_g = F.binary_cross_entropy_with_logits(self.D(x_fake), real)
        self.opt_g.zero_grad(); loss_g.backward(); self.opt_g.step()
        ## loss_g.backward() also fills D's grads; opt_d.zero_grad() clears them next step
        return loss_d.item(), loss_g.item()

## Example: 2-D toy data
G = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 2))
D = nn.Sequential(nn.Linear(2, 32), nn.LeakyReLU(0.2), nn.Linear(32, 1))
gan = GAN(G, D, z_dim=8)
x = torch.randn(16, 2) + 3
print(gan.step(x))  ## (loss_d, loss_g); loss_d ≈ 2 log 2 ≈ 1.39 at init
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- 1 forward pass per sample → fast sampling.
- Sharp samples ← $D$ penalizes any detectable artifact. ❌Pixel-wise likelihood that averages modes (VAE blur).
- ❌Explicit density, ❌Markov chain, ❌approximate inference → any differentiable $G$.

*Cons?*
- Unstable: game, ❌single loss → [Non-Convergence](#non-convergence).
- [Mode Collapse](#mode-collapse).
- ❌Likelihood → hard to evaluate & compare models.
- ❌Encoder: $x\to z$ needs a separate inversion.
- D loss ❌tracks sample quality → no reliable learning curve.

*What does $D$ learn?*
- A density ratio: $\frac{D^*}{1-D^*}=\frac{p_\text{data}}{p_g}$.
- → GAN = fit $p_g$ by repeatedly estimating $p_\text{data}/p_g$ w/ a classifier.

*Does alternating SGD provably reach $p_g=p_\text{data}$?*
- ❌. The paper's convergence result assumes infinite capacity, $D=D^*_G$ at every step & updates directly on $p_g$ (convex in $p_g$).
- Real training: updates on $\theta,\phi$ → non-convex. $D\neq D^*$. Few steps → ❌guarantee.

*Why not train $D$ to optimality before each G step?*
- Near-optimal $D$ on (near-)disjoint supports saturates → [Vanishing Generator Gradient](#vanishing-generator-gradient).
- Too weak $D$ → misleading gradient. → Balance via $k$, LRs, capacity.

*Is the NS loss still JSD minimization?*
- ❌. At $D^*$, its gradient = gradient of reverse KL $-\,2\,$JSD → mode-seeking → [Mode Collapse](#mode-collapse).

*Why is $D$'s output a learned loss?*
- $D$ penalizes whatever currently separates fake from real → adaptive perceptual loss.
- Hand-made losses (L2) average plausible outputs → blur. $D$ rejects blur as fake.

*GAN vs VAE?*
- [VAE](vae.md#vae): ELBO, has an encoder, stable, blurry ← Gaussian decoder averages modes.
- GAN: adversarial, ❌encoder, sharp, unstable.

*GAN vs diffusion?*
- [Diffusion](diffusion.md#diffusion-model): regression loss → stable. Likelihood-based → mode coverage. $T$ net evals per sample → slow.
- GAN: 1 net eval → fast. Minimax → unstable, drops modes.
- Diffusion surpassed the best GANs (BigGAN-deep) on ImageNet sample quality {cite:p}`dhariwal2021diffusion`.
- Adversarial losses survive as components: [LDM](diffusion.md#ldm) autoencoder (perceptual + patch-adversarial loss), adversarial distillation of diffusion models into few-step samplers.
```

&nbsp;

### Vanishing Generator Gradient
- **What**: Near-optimal $D$ → ≈0 gradient for $G$ from $\mathcal{L}_G^\text{sat}$. {cite:p}`arjovsky2017towards`
- **Why**: $p_\text{data}$ & $p_g$ have (near-)disjoint supports.
    - Real images lie near a low-dim manifold in pixel space.
    - $p_g$ lies on the image of $G$ → dim ≤ dim($z$) ≪ data dim.
    - 2 low-dim manifolds in general position → their overlap has measure 0.
- **How**:
    1. Disjoint supports → a perfect, smooth $D$ exists: $D=1$ on real, $D=0$ on fake.
    2. → $\text{JSD}=\log2$ for every such $G$ → $C(G)$ is flat → ❌gradient.
    3. $D\to D^*$ → $\nabla_\theta\mathcal{L}_G^\text{sat}\to0$.
    4. NS loss: gradient ❌vanishes, but updates become unstable (high variance) as $D\to D^*$.

```{note} Math
:class: dropdown
Definition:

$$
\text{supp}\,p_\text{data}\cap\text{supp}\,p_g=\varnothing\ \Rightarrow\ \text{JSD}(p_\text{data}\|p_g)=\log2\ \Rightarrow\ C(G)=0
$$

- $\text{supp}$: Support (smallest closed set w/ all probability mass).
- $C(G)$ = const over all $G$ w/ disjoint support → ❌direction toward $p_\text{data}$.

Limit:

$$
\lim_{\|D-D^*\|\to0}\nabla_\theta\,\mathbb{E}_{z\sim p_z}\left[\log\left(1-D(G(z))\right)\right]=0
$$
```

```{attention} Q&A
:class: dropdown
*Why does a classifier-based divergence fail here?*
- JSD only sees *overlap*, ❌*distance*. A fake 1 pixel off & a fake far off both get $D=0$ → same loss.
- → Need a distance that uses sample-space geometry → [WGAN](#wgan).

*Does the NS loss fix it?*
- Partly. Gradient ❌vanishes, but becomes unstable near $D^*$ & is mode-seeking ([Mode Collapse](#mode-collapse)).

*Fixes?*
- Add full-support noise (e.g., Gaussian) to $D$'s inputs → both distributions spread over the whole sample space → supports overlap → JSD informative again.
    - Bounded noise is ❌enough: $\delta_0+U[-1,1]$ & $\delta_{10}+U[-1,1]$ stay disjoint.
- Don't train $D$ to optimality. Limit $D$'s slope ([Spectral Normalization](../foundation/module.md#spectral-normalization), gradient penalty).
- Switch the distance → [WGAN](#wgan).

*Why does D loss → 0 NOT mean success?*
- $D$ separates real & fake perfectly → $D$ won → $G$ gets ❌signal.
```

&nbsp;

### Mode Collapse
- **What**: $G$ covers only a few modes of $p_\text{data}$ → low sample diversity.
    - Complete collapse: many $z$ → ~1 output ("Helvetica scenario").
- **Why**: $G$'s best response to a *fixed* $D$ is a point mass.
    - For fixed $D$: $\min_G\mathbb{E}_z\left[\log\left(1-D(G(z))\right)\right]$ → map every $z$ to $\arg\max_xD(x)$.
    - $D$ scores each sample independently → ❌sees a lack of diversity.
- **How**:
    1. $D$ rates region A as most "real" → $G$ moves many $z$ to A.
    2. $D$ learns that A is fake → $D$⬇️ on A.
    3. $G$ jumps to region B → $D$ follows → ... (mode hopping).
    - Many G steps vs a stale $D$ → faster collapse.

```{note} Math
:class: dropdown
Definition: $p_g$ assigns ≈0 mass to regions where $p_\text{data}$ has mass.

NS-loss gradient at $D^*$ {cite:p}`arjovsky2017towards`:

$$
\mathbb{E}_{z\sim p_z}\left[-\nabla_\theta\log D^*(G_\theta(z))\right]=\nabla_\theta\left[D_\text{KL}(p_g\|p_\text{data})-2\,\text{JSD}(p_g\|p_\text{data})\right]
$$

- $D^*$: Optimal $D$ for the current $\theta$. Gradient evaluated at that $\theta$.
- Reverse KL $D_\text{KL}(p_g\|p_\text{data})=\mathbb{E}_{x\sim p_g}\left[\log\frac{p_g(x)}{p_\text{data}(x)}\right]$:
    - $p_g>0$ where $p_\text{data}\to0$ → cost $\to\infty$ → fake-looking samples punished hard.
    - $p_g\to0$ where $p_\text{data}>0$ → cost $\to0$ → dropped modes nearly free.
    - → **Mode-seeking**.
- $-2\,\text{JSD}$: pushes $p_g$ *away* from $p_\text{data}$ (wrong sign).
```

```{attention} Q&A
:class: dropdown
*Why doesn't the ideal JSD objective prevent it?*
- With $D^*$, JSD does penalize a dropped mode.
- Practice: alternating steps ❌solve the inner max → $G$ best-responds to a stale $D$ → point mass (see **Why**).
- NS loss adds the mode-seeking reverse-KL term (Math).

*How to detect it?*
- Near-duplicate samples for different $z$.
- Within-class collapse can hide behind good per-sample quality → inspect diversity, ❌only realism.

*Fixes?*
- $D$ sees batch statistics → detects low diversity: minibatch discrimination, minibatch std-dev feature.
- $G$ looks ahead: unrolled GAN ($G$ optimizes vs $D$ after several unrolled D steps).
- Change the distance: [WGAN](#wgan).
- Condition on labels: [cGAN](#cgan) → every class must be produced.

*Mode collapse vs representation collapse?*
- Same symptom: constant output for different inputs.
- [Representation collapse](../repr/ssl.md#representation-collapse) (SSL): encoder → 1 embedding ← trivial solution of an invariance loss.
- Mode collapse: generator → few samples ← adversarial best response.
```

&nbsp;

### Non-Convergence
- **What**: $G$ & $D$ params cycle or diverge around the equilibrium, ❌settle.
- **Why**: GAN = game, ❌single objective.
    - Update field $(\nabla_\theta\mathcal{L}_G,\nabla_\phi\mathcal{L}_D)$ ≠ gradient of 1 scalar → rotational part.
    - Each step changes the other player's loss surface → moving target.
- **How**:
    1. Gradient descent ($G$) & ascent ($D$) on a saddle → orbits around the equilibrium.
    2. Finite LR → orbits spiral out (simultaneous updates) or cycle (alternating updates).
    3. Visible as oscillating losses & sample quality.

```{tip} Derivation
:class: dropdown
*Why does gradient descent-ascent fail on $\min_u\max_v\,uv$?*
1. Toy game: $u$ = G's param, $v$ = D's param. Equilibrium $(u,v)=(0,0)$.
2. Simultaneous step, LR $\eta$: $u\leftarrow u-\eta v$, $v\leftarrow v+\eta u$.

    $$
    \begin{bmatrix}u\\v\end{bmatrix}\leftarrow\begin{bmatrix}1&-\eta\\\eta&1\end{bmatrix}\begin{bmatrix}u\\v\end{bmatrix}
    $$

3. Eigenvalues $1\pm i\eta$ → $|\lambda|=\sqrt{1+\eta^2}>1$ → spiral out → diverge.
4. Alternating ($v$ uses the new $u$): $v\leftarrow\eta u+(1-\eta^2)v$.

    $$
    \begin{bmatrix}u\\v\end{bmatrix}\leftarrow\begin{bmatrix}1&-\eta\\\eta&1-\eta^2\end{bmatrix}\begin{bmatrix}u\\v\end{bmatrix}
    $$

5. $\det=1$, $|\text{tr}|=|2-\eta^2|<2$ for $0<\eta<2$ → complex eigenvalues w/ $|\lambda|=1$ → bounded cycle, ❌convergence.
6. Continuous time ($\eta\to0$): $\dot u=-v$, $\dot v=u$ → circles.
```

```{attention} Q&A
:class: dropdown
*Why doesn't this happen in ordinary training?*
- 1 loss → update field = gradient of a scalar → ❌rotation → small steps decrease the loss monotonically.
- Game: $(\nabla_uf,-\nabla_vf)$ is ❌a gradient field → rotation is possible.

*Fixes?*
- ⬇️LR, ⬇️momentum (DCGAN: Adam $\beta_1=0.5$).
- Separate LRs for $G$ & $D$ (two time-scale update rule).
- Damp $D$: gradient penalties ([WGAN-GP](#wgan-gp)), [Spectral Normalization](../foundation/module.md#spectral-normalization).
- Average $G$'s weights over time (EMA) → in the toy game, the average of a cycle = its center = the equilibrium.
```

&nbsp;

## Architectures

### DCGAN
- **Name**: Deep Convolutional Generative Adversarial Network {cite:p}`radford2016unsupervised`
- **What**: All-convolutional $G$ & $D$ + a set of architecture rules for stable GAN training.
- **Why**: Earlier GANs w/ CNNs trained unstably & did not scale to images.
    - Also: test whether $D$ features & $G$'s latent space are useful representations (unsupervised learning).
- **How**:
    - Pooling → strided conv ($D$) & fractional-strided conv ($G$) → [Transposed Convolution](../../cv/cnn.md#transposed-convolution).
    - [BatchNorm](../foundation/module.md#batchnorm) in $G$ & $D$, except the $G$ output layer & the $D$ input layer.
    - ❌FC hidden layers. $z$ (100-d, uniform) → project & reshape → 4×4 feature maps.
    - $G$: ReLU, output Tanh (images scaled to $[-1,1]$). $D$: [LReLU](../foundation/activation.md#lrelu), slope 0.2.
    - Adam, LR $2\times10^{-4}$, $\beta_1=0.5$. Init $\mathcal{N}(0,0.02^2)$. Batch 128.

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class DCGANGenerator(nn.Module):
    def __init__(self, z_dim=100, ch=64, out_ch=3):
        super().__init__()
        def up(i, o):  ## fractional-strided conv doubles H, W -> BN -> ReLU
            return [nn.ConvTranspose2d(i, o, 4, 2, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True)]
        self.net = nn.Sequential(
            ## project z: 1x1 -> 4x4 (the "FC" layer, written as a conv)
            nn.ConvTranspose2d(z_dim, ch * 8, 4, 1, 0, bias=False), nn.BatchNorm2d(ch * 8), nn.ReLU(True),
            *up(ch * 8, ch * 4),  ## 8x8
            *up(ch * 4, ch * 2),  ## 16x16
            *up(ch * 2, ch),      ## 32x32
            ## output layer: no BN, Tanh -> pixels in [-1, 1]
            nn.ConvTranspose2d(ch, out_ch, 4, 2, 1), nn.Tanh(),  ## 64x64
        )
        for mod in self.modules():
            if isinstance(mod, nn.ConvTranspose2d):
                nn.init.normal_(mod.weight, 0.0, 0.02)  ## N(0, 0.02^2) init (paper: all weights)

    def forward(self, z):
        return self.net(z.view(*z.shape, 1, 1))

## Example
G = DCGANGenerator(z_dim=100, ch=8)
z = torch.rand(2, 100) * 2 - 1  ## uniform z in [-1, 1]
print(G(z).shape)  ## torch.Size([2, 3, 64, 64])

## Shallow stand-in D (the real D mirrors all 4 G stages): strided conv, LeakyReLU(0.2),
## no BN on the input layer, no pooling, no FC hidden layer
D = nn.Sequential(
    nn.Conv2d(3, 8, 4, 2, 1), nn.LeakyReLU(0.2, True),                                   ## 32x32
    nn.Conv2d(8, 16, 4, 2, 1, bias=False), nn.BatchNorm2d(16), nn.LeakyReLU(0.2, True),  ## 16x16
    nn.Conv2d(16, 1, 16, 1, 0),                                                           ## 1x1 logit
)
for mod in D.modules():
    if isinstance(mod, nn.Conv2d):
        nn.init.normal_(mod.weight, 0.0, 0.02)  ## same N(0, 0.02^2) init for D
print(D(G(z)).shape)  ## torch.Size([2, 1, 1, 1])
```
````

```{attention} Q&A
:class: dropdown
*Why strided conv, ❌pooling?*
- Fixed pooling → fixed down/upsampling. Strided conv → the net learns its own resampling.

*Why ❌BN on the $G$ output & $D$ input?*
- BN on all layers → sample oscillation & instability (paper).

*Why Tanh at the output?*
- Bounded → $G$ learns faster to saturate & cover the color space (paper). Data scaled to $[-1,1]$ to match.

*Why Adam $\beta_1=0.5$, ❌0.9?*
- $\beta_1=0.9$ → training oscillation & instability (paper).

*Why LReLU in $D$?*
- Paper: works well, esp. at high resolution.
- Common reason: ❌dead units → $G$ gets gradient through all of $D$.

*What did DCGAN show about the latent space?*
- Smooth interpolation between $z$s → evidence against memorization.
- Vector arithmetic on $z$: mean $z$(smiling woman) − mean $z$(neutral woman) + mean $z$(neutral man) → smiling man.
    - Needs the mean of ~3 exemplars per concept. Single samples are unstable.
```

&nbsp;

### cGAN
- **Name**: Conditional Generative Adversarial Network {cite:p}`mirza2014conditional`
- **What**: GAN w/ both $G$ & $D$ conditioned on extra info $c$ (class label, text, image).
- **Why**: Unconditional GAN → ❌control over what is generated.
    - Model $p_\text{data}(x|c)$ instead of $p_\text{data}(x)$.
- **How**:
    1. Feed $c$ as an extra input to both $G$ & $D$ (e.g., concat one-hot $c$ to $z$ & to $x$).
    2. $D$ judges pairs $(x,c)$ → fake if $x$ is unrealistic OR does not match $c$.
    3. Same losses as [GAN](#gan).

```{note} Math
:class: dropdown
Notations:
- IO:
    - $c$: Condition.

Training:

$$
\min_G\max_D\ \mathbb{E}_{(x,c)\sim p_\text{data}}\left[\log D(x,c)\right]+\mathbb{E}_{z\sim p_z,\,c\sim p_\text{data}(c)}\left[\log\left(1-D(G(z,c),c)\right)\right]
$$

- Optimal $D$: $D^*(x,c)=\frac{p_\text{data}(x|c)}{p_\text{data}(x|c)+p_g(x|c)}$ → per-$c$ JSD.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Conditioned(nn.Module):
    ## Wraps G or D: concat one-hot condition c to the input
    def __init__(self, net, n_classes):
        super().__init__()
        self.net, self.K = net, n_classes

    def forward(self, inp, c):
        return self.net(torch.cat([inp, F.one_hot(c, self.K).float()], dim=-1))

## Example: 10 classes, z_dim = 8, 2-D data
G = Conditioned(nn.Sequential(nn.Linear(8 + 10, 32), nn.ReLU(), nn.Linear(32, 2)), 10)
D = Conditioned(nn.Sequential(nn.Linear(2 + 10, 32), nn.LeakyReLU(0.2), nn.Linear(32, 1)), 10)
z, c = torch.randn(4, 8), torch.tensor([0, 3, 3, 9])
x_fake = G(z, c)
print(x_fake.shape, D(x_fake, c).shape)  ## torch.Size([4, 2]) torch.Size([4, 1])
## Training: GAN losses as above; D always sees the same c that G used
```
````

```{attention} Q&A
:class: dropdown
*Why must $D$ also see $c$?*
- $D$ w/o $c$ checks only realism → $G$ can ignore $c$ (realistic sample, wrong class) at no cost.

*Does conditioning reduce mode collapse?*
- Across classes: yes ← $D$ rejects wrong-class samples → every class must be produced.
- Within a class: collapse is still possible.

*Other ways to inject $c$?*
- Concat (original). Auxiliary classifier on $D$ (AC-GAN). Projection: inner product of embedded $c$ w/ $D$'s features. Conditional BatchNorm in $G$.

*How does pix2pix relate?* {cite:p}`isola2017image`
- cGAN w/ $c$ = input image, for paired image-to-image translation.
- Adds an L1 loss to the target → low frequencies correct. PatchGAN $D$ → high-frequency realism per patch.
- $G$ ignored input noise $z$ → noise only via dropout, at train & test time → little output diversity.
- Needs paired data → [CycleGAN](#cyclegan) removes this need.
```

&nbsp;

### CycleGAN
- **What**: Unpaired image-to-image translation w/ 2 GANs ($\mathcal{X}\to\mathcal{Y}$, $\mathcal{Y}\to\mathcal{X}$) + a cycle-consistency loss. {cite:p}`zhu2017unpaired`
- **Why**: Paired data $(x,y)$ is rare (photo ↔ Monet, horse ↔ zebra).
    - Adversarial loss alone is under-constrained: any map that produces realistic $\mathcal{Y}$ images fools $D_\mathcal{Y}$, even if it ignores the content of $x$.
- **How**:
    1. $G:\mathcal{X}\to\mathcal{Y}$ vs $D_\mathcal{Y}$. $F:\mathcal{Y}\to\mathcal{X}$ vs $D_\mathcal{X}$.
    2. Cycle: $x\to G\to F\to\approx x$ & $y\to F\to G\to\approx y$ (L1).
    3. → $G$ must keep the content of $x$ ← $F$ must recover $x$ from $G(x)$.
    - Practice: least-squares GAN loss, $\lambda=10$, ResNet $G$ w/ [InstanceNorm](../foundation/module.md#instancenorm), 70×70 PatchGAN $D$, $D$ trained on a history buffer of 50 generated images.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $x\in\mathcal{X}$: Image from domain $\mathcal{X}$.
    - $y\in\mathcal{Y}$: Image from domain $\mathcal{Y}$.
- Params:
    - $F$: Inverse generator $\mathcal{Y}\to\mathcal{X}$.
    - $D_\mathcal{X},D_\mathcal{Y}$: Discriminators per domain.
- Hyperparams:
    - $\lambda$: Cycle-loss weight.

Overrides: $G:\mathcal{X}\to\mathcal{Y}$ maps images, ❌noise. $y$ = target-domain image, ❌label.

Training:

$$
\begin{align*}
\mathcal{L}_\text{GAN}(G,D_\mathcal{Y})&=\mathbb{E}_{y}\left[\log D_\mathcal{Y}(y)\right]+\mathbb{E}_{x}\left[\log\left(1-D_\mathcal{Y}(G(x))\right)\right]\\
\mathcal{L}_\text{cyc}(G,F)&=\mathbb{E}_{x}\left[\|F(G(x))-x\|_1\right]+\mathbb{E}_{y}\left[\|G(F(y))-y\|_1\right]\\
\mathcal{L}&=\mathcal{L}_\text{GAN}(G,D_\mathcal{Y})+\mathcal{L}_\text{GAN}(F,D_\mathcal{X})+\lambda\,\mathcal{L}_\text{cyc}(G,F)
\end{align*}
$$

$$
G^*,F^*=\arg\min_{G,F}\max_{D_\mathcal{X},D_\mathcal{Y}}\mathcal{L}
$$

- Practice: $\log$ loss → least squares. $G$ minimizes $\mathbb{E}_x\left[(D_\mathcal{Y}(G(x))-1)^2\right]$. $D_\mathcal{Y}$ minimizes $\mathbb{E}_y\left[(D_\mathcal{Y}(y)-1)^2\right]+\mathbb{E}_x\left[D_\mathcal{Y}(G(x))^2\right]$.
- Optional identity loss: $\mathbb{E}_y\left[\|G(y)-y\|_1\right]+\mathbb{E}_x\left[\|F(x)-x\|_1\right]$, weight $0.5\lambda$ → preserves color (painting → photo).
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class CycleGANGenLoss:
    def __init__(self, lam=10.0, lam_id=0.0):
        self.lam, self.lam_id = lam, lam_id

    def __call__(self, G, F, D_X, D_Y, x, y):
        fake_y, fake_x = G(x), F(y)
        ## least-squares adversarial loss: push D's score on fakes toward 1 ("real")
        adv = ((D_Y(fake_y) - 1) ** 2).mean() + ((D_X(fake_x) - 1) ** 2).mean()
        ## cycle consistency (L1): x -> G -> F -> x and y -> F -> G -> y
        cyc = (F(fake_y) - x).abs().mean() + (G(fake_x) - y).abs().mean()
        loss = adv + self.lam * cyc
        if self.lam_id > 0:  ## identity: G(y) ≈ y, F(x) ≈ x -> keeps colors
            loss = loss + self.lam_id * ((G(y) - y).abs().mean() + (F(x) - x).abs().mean())
        return loss

## Example: stand-in 1-layer generators & patch discriminators
G, F = nn.Conv2d(3, 3, 3, padding=1), nn.Conv2d(3, 3, 3, padding=1)
D_X, D_Y = nn.Conv2d(3, 1, 4, 2, 1), nn.Conv2d(3, 1, 4, 2, 1)  ## 8x8 -> 4x4 patch scores
x, y = torch.randn(2, 3, 8, 8), torch.randn(2, 3, 8, 8)
print(CycleGANGenLoss(lam_id=5.0)(G, F, D_X, D_Y, x, y))  ## scalar tensor
```
````

```{attention} Q&A
:class: dropdown
*Why is the cycle loss needed?*
- Adversarial loss matches only the *distribution* $G(\mathcal{X})\approx\mathcal{Y}$.
- Infinitely many maps achieve this. Many ignore $x$ (e.g., collapse to a few good $\mathcal{Y}$ images).
- Cycle loss → $G$ ≈ invertible → keeps the info of $x$.

*Cons?*
- Geometric changes fail (dog → cat: minimal edits). Works best for color & texture changes.
- Deterministic $G$ + cycle loss → ≈bijection → 1-to-many translations unsupported.
- $G$ hides info about $x$ in a near-imperceptible, high-frequency signal in $G(x)$ → $F$ recovers $x$ from it {cite:p}`chu2017cyclegan`.

*Why a buffer of past fakes?*
- $D$ trained only on the latest fakes → oscillation. Past fakes → $D$ remembers old failure modes.

*CycleGAN vs pix2pix?*
- [pix2pix](#cgan): paired data, L1 to the target.
- CycleGAN: unpaired, L1 to the *reconstruction*.
```

&nbsp;

### StyleGAN
- **What**: GAN generator that injects the latent as per-layer "styles" via AdaIN, starting from a learned constant. {cite:p}`karras2019style`
- **Why**: Classic $G$ feeds $z$ only at the input → $z$ must follow the fixed prior $p_z$ → entangled factors, ❌scale-specific control.
    - Intermediate latent $w$ has ❌fixed prior → it can "unwarp" → more disentangled.
- **How**:
    1. **Mapping network**: 8-layer MLP, normalized $z\to w$. Both 512-d.
    2. **Synthesis network**: starts from a learned 4×4×512 constant → upsample + conv blocks → 1024².
    3. **AdaIN** after each conv: learned affine $A(w)$ → per-channel (scale, bias) → normalize each feature map → re-style it.
    4. **Noise inputs**: per-layer Gaussian noise image, learned per-channel scale → stochastic detail (hair, freckles).
    5. **Style mixing**: during training, 90% of images use 2 latents → switch $w_1\to w_2$ at a random layer → decorrelates layers.
        - Coarse styles (4²–8²): pose, face shape, eyeglasses. Middle (16²–32²): facial features, hair style. Fine (64²–1024²): color scheme, microstructure.
    6. **Truncation trick** in $\mathcal{W}$ (sampling only): pull $w$ toward the mean $\bar w$ → fidelity⬆️, diversity⬇️.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $w\in\mathcal{W}=\mathbb{R}^{512}$: Intermediate latent.
- Params:
    - $A$: Learned affine $w\mapsto(s,b)$.
- Hyperparams:
    - $\psi$: Truncation strength.
- Misc:
    - $h_i$: $i$-th channel (feature map) of a conv output.
    - $s_i$: Style scale of channel $i$.
    - $b_i$: Style bias of channel $i$.

Model:

$$
w=\text{MLP}(z),\qquad (s,b)=A(w)
$$

- $z$ is normalized (RMS = 1) before the MLP (official code).

$$
\text{AdaIN}(h_i,w)=s_i\,\frac{h_i-\mu(h_i)}{\sigma(h_i)}+b_i
$$

- $\mu(h_i)$: Mean of $h_i$ over spatial positions, per sample.
- $\sigma(h_i)$: Std of $h_i$ over spatial positions, per sample.
- Normalize → erases the previous layer's style. $(s_i,b_i)$ → sets the new one.
- Official code parameterizes the scale as $1+$ affine output → near-identity style at init.

Noise input:

$$
h_i\leftarrow h_i+\beta_i\,n,\qquad n\sim\mathcal{N}(0,I)
$$

- $n$: 1 single-channel noise image per layer, shared across channels.
- $\beta_i$: Learned per-channel noise scale.

Inference (truncation):

$$
w'=\bar w+\psi\,(w-\bar w),\qquad \bar w=\mathbb{E}_{z\sim p_z}\left[\text{MLP}(z)\right],\quad\psi<1
$$

- $\bar w$: estimated as an EMA of $w$ during training (official code).
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class StyleGANGenerator(nn.Module):
    def __init__(self, z_dim=512, ch=16, n_layers=3, out_ch=3):
        super().__init__()
        ## mapping network: 8-layer MLP z -> w (w_dim = z_dim)
        self.mapping = nn.Sequential(*[mod for _ in range(8) for mod in (nn.Linear(z_dim, z_dim), nn.LeakyReLU(0.2))])
        ## synthesis starts from a learned constant, not from z
        self.const = nn.Parameter(torch.ones(1, ch, 4, 4))
        self.convs = nn.ModuleList(nn.Conv2d(ch, ch, 3, padding=1) for _ in range(n_layers))
        ## per layer: "A" = affine w -> (scale, bias), "B" = per-channel noise scale
        self.A = nn.ModuleList(nn.Linear(z_dim, 2 * ch) for _ in range(n_layers))
        self.B = nn.ParameterList(nn.Parameter(torch.zeros(1, ch, 1, 1)) for _ in range(n_layers))
        self.to_rgb = nn.Conv2d(ch, out_ch, 1)
        self.register_buffer("w_avg", torch.zeros(z_dim))  ## EMA of w during training -> truncation center

    def map(self, z):
        z = z * torch.rsqrt(z.pow(2).mean(dim=1, keepdim=True) + 1e-8)  ## normalize z (RMS = 1)
        return self.mapping(z)

    @staticmethod
    def adain(h, s, b, eps=1e-8):
        ## instance norm: stats per sample & per channel over (H, W) -> erases the old style
        mu = h.mean(dim=(2, 3), keepdim=True)
        sd = (h.var(dim=(2, 3), keepdim=True, unbiased=False) + eps).sqrt()
        ## re-style: new per-channel scale & bias from w
        return s[:, :, None, None] * (h - mu) / sd + b[:, :, None, None]

    def forward(self, z, z2=None, psi=1.0):
        L = len(self.convs)
        ws = [self.map(z)] * L
        if z2 is not None:  ## style mixing: z drives layers before the crossover, z2 the rest
            cross = torch.randint(1, L, ()).item()
            ws = ws[:cross] + [self.map(z2)] * (L - cross)
        ws = [self.w_avg + psi * (w - self.w_avg) for w in ws]  ## truncation; psi = 1 -> off
        h = self.const.expand(z.size(0), -1, -1, -1)
        for l in range(L):
            if l > 0:
                h = F.interpolate(h, scale_factor=2)  ## 4 -> 8 -> 16
            h = self.convs[l](h)
            ## noise: 1 single-channel image per layer, broadcast w/ learned per-channel scale
            h = h + self.B[l] * torch.randn(h.size(0), 1, *h.shape[2:])
            h = F.leaky_relu(h, 0.2)
            s, b = self.A[l](ws[l]).chunk(2, dim=1)
            h = self.adain(h, 1 + s, b)  ## scale = 1 + A(w) -> near-identity style at init
        return self.to_rgb(h)

## Example
G = StyleGANGenerator(z_dim=32, ch=8, n_layers=3)
z, z2 = torch.randn(2, 32), torch.randn(2, 32)
print(G(z).shape)           ## torch.Size([2, 3, 16, 16])
print(G(z, z2).shape)       ## style mixing, same shape
print(G(z, psi=0.7).shape)  ## truncation, same shape
```
````

```{attention} Q&A
:class: dropdown
*Why a mapping network?*
- $z$ must follow $p_z$ (e.g., Gaussian). Real factors of variation (e.g., rare attribute combos) have a different, warped density.
- Fixed $p_z$ → $G$ must warp it → entangled $z$.
- $w=\text{MLP}(z)$ → learned, warped density → $G$ can use a more linear, disentangled $\mathcal{W}$.

*Why start from a constant?*
- All sample-specific variation enters via styles & noise. The constant = shared template → feeding $z$ at the input brought ❌further gain (paper).

*Why does style mixing localize styles?*
- 2 latents per image → $G$ can't assume adjacent layers' styles are correlated → each layer's style controls its own scale.

*What does the noise do?*
- Stochastic detail (hair placement, freckles, pores) w/o changing identity or pose.
- Without it, $G$ must fake randomness from $w$ → wastes capacity, repetitive patterns.

*Truncation trick: pros & cons?*
- ✅ Fidelity⬆️: avoids low-density regions of $\mathcal{W}$.
- ❌ Diversity⬇️. $\psi\to0$ → the "average face".
- Same trade-off as [CFG](diffusion.md#cfg) or low-temperature sampling.

*What did StyleGAN2 change?* {cite:p}`karras2020analyzing`
- AdaIN → "water droplet" artifacts ← $G$ sneaks signal-strength info past instance norm via a strong local spike.
    - Fix: **weight demodulation**. Scale the conv weights by the style, then normalize them by the expected output std. ❌Normalization of actual activations.
- Progressive growing → "phase" artifacts (e.g., teeth stay aligned to the camera, not the pose).
    - Fix: drop progressive growing. Skip-connection $G$, residual $D$.
- Path length regularization → smoother $\mathcal{W}\to$ image map → easier inversion.
- Lazy regularization: compute regularizers less often than the main loss.
```

&nbsp;

## WGAN
- **Name**: Wasserstein Generative Adversarial Network {cite:p}`arjovsky2017wasserstein`
- **What**: GAN whose critic estimates the Wasserstein-1 (Earth Mover's) distance between $p_\text{data}$ & $p_g$.
- **Why**: JSD gives ❌gradient when supports are disjoint → [Vanishing Generator Gradient](#vanishing-generator-gradient).
    - $W_1$ uses sample-space distance → continuous in $\theta$ everywhere & differentiable almost everywhere (if $G$ is locally Lipschitz + mild regularity) → meaningful gradient even w/ disjoint supports.
    - Critic loss correlates w/ sample quality → meaningful learning curve.
- **How**:
    1. $D$ → critic $f$: real-valued score, ❌sigmoid, ❌log.
    2. Critic maximizes $\mathbb{E}_{\text{real}}f-\mathbb{E}_{\text{fake}}f$ over 1-Lipschitz $f$. **Kantorovich–Rubinstein duality** → this max = $W_1$.
    3. Enforce Lipschitz: clip every critic weight to $[-c,c]$ after each step → $f$ is $K$-Lipschitz → estimates $K\cdot W_1$.
    4. $n_\text{critic}=5$ critic steps per G step → critic near optimal. Safe ← its gradient ❌vanishes.
    5. $G$ minimizes $-\mathbb{E}_z\left[f(G(z))\right]$.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $f$: Critic.
- Hyperparams:
    - $c$: Weight-clipping bound.
    - $n_\text{critic}$: #critic steps per G step.
- Misc:
    - $\Pi(p_\text{data},p_g)$: Set of couplings = joint distributions w/ marginals $p_\text{data}$ & $p_g$.
    - $\|f\|_L$: Lipschitz constant of $f$.
    - $K$: Lipschitz bound implied by clipping. Depends only on $c$ & the architecture.

Definition:

$$
W_1(p_\text{data},p_g)=\inf_{\gamma\in\Pi(p_\text{data},p_g)}\mathbb{E}_{(x,\tilde x)\sim\gamma}\left[\|x-\tilde x\|\right]
$$

- $\gamma$: Transport plan. Min cost to move $p_g$'s mass onto $p_\text{data}$.

Kantorovich–Rubinstein duality:

$$
W_1(p_\text{data},p_g)=\sup_{\|f\|_L\leq1}\ \mathbb{E}_{x\sim p_\text{data}}\left[f(x)\right]-\mathbb{E}_{\tilde x\sim p_g}\left[f(\tilde x)\right]
$$

Training:

$$
\min_G\ \max_{f:\,\|f\|_L\leq K}\ \mathbb{E}_{x\sim p_\text{data}}\left[f(x)\right]-\mathbb{E}_{z\sim p_z}\left[f(G(z))\right]
$$

- Inner max $=K\cdot W_1$ if the critic family contains the optimal $f$.
- Generator gradient: $\nabla_\theta W_1=-\mathbb{E}_{z\sim p_z}\left[\nabla_\theta f^*(G_\theta(z))\right]$. $f^*$: optimal critic.
- Defaults: RMSProp, LR $5\times10^{-5}$, $c=0.01$, $m=64$, $n_\text{critic}=5$.
```

```{note} Example
:class: dropdown
Parallel lines in $\mathbb{R}^2$:
- $p_\text{data}$: uniform on $\{0\}\times[0,1]$. $p_g$: uniform on $\{\theta\}\times[0,1]$.
- $W_1=|\theta|$ → continuous, gradient $\text{sign}(\theta)$ → pulls $\theta\to0$.
- JSD $=\log2$ if $\theta\neq0$, $0$ if $\theta=0$ → flat, ❌gradient.
- KL (both directions) $=+\infty$ if $\theta\neq0$.
- → $\theta_t\to0$ converges under $W_1$, but under neither JSD nor KL.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class WGAN:
    def __init__(self, G, f, z_dim, lr=5e-5, c=0.01, n_critic=5):
        self.G, self.f, self.z_dim, self.c, self.n_critic = G, f, z_dim, c, n_critic
        self.opt_g = torch.optim.RMSprop(G.parameters(), lr=lr)
        self.opt_f = torch.optim.RMSprop(f.parameters(), lr=lr)

    def step(self, batches):  ## n_critic real minibatches
        for x in batches:
            x_fake = self.G(torch.randn(x.size(0), self.z_dim)).detach()
            ## critic maximizes E f(x) - E f(G(z)) -> minimize the negative
            loss_f = self.f(x_fake).mean() - self.f(x).mean()
            self.opt_f.zero_grad(); loss_f.backward(); self.opt_f.step()
            ## weight clipping: compact param set -> f is K-Lipschitz
            with torch.no_grad():
                for p in self.f.parameters():
                    p.clamp_(-self.c, self.c)
        loss_g = -self.f(self.G(torch.randn(x.size(0), self.z_dim))).mean()
        self.opt_g.zero_grad(); loss_g.backward(); self.opt_g.step()
        return -loss_f.item(), loss_g.item()  ## -loss_f ≈ K * W1 estimate

## Example: 2-D toy data
G = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 2))
f = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 1))  ## critic: no sigmoid
wgan = WGAN(G, f, z_dim=8)
batches = [torch.randn(16, 2) + 3 for _ in range(wgan.n_critic)]
print(wgan.step(batches))  ## (W1 estimate, G loss)
```
````

```{attention} Q&A
:class: dropdown
*Why "critic", ❌"discriminator"?*
- $f$ is ❌a classifier. It outputs an unbounded score. Only score *differences* matter.

*Why can the critic be trained to optimality, but a GAN discriminator can't?*
- Optimal GAN $D$ → saturated → vanishing gradient.
- Optimal critic → exact $W_1$ gradient. Better critic → better gradient for $G$.

*Why is weight clipping bad?*
- Authors: "a clearly terrible way to enforce a Lipschitz constraint".
- Large $c$ → weights take long to reach the bound → slow critic training.
- Small $c$ → vanishing gradients in deep critics.
- Critic underuses capacity → learns overly simple functions → fixed by [WGAN-GP](#wgan-gp).

*Why RMSProp, ❌Adam?*
- Authors saw instability w/ momentum (Adam, $\beta_1>0$) on the critic ← nonstationary loss.
- [WGAN-GP](#wgan-gp) later trains w/ Adam.

*Is the critic loss a reliable metric?*
- Within 1 run: correlates w/ sample quality.
- Across models: ❌. Scale $K$ & critic capacity differ.

*Does WGAN really minimize $W_1$?*
- Contested: later work argues the critic approximates $W_1$ poorly & WGANs work *because* of this, not despite it {cite:p}`stanczuk2021wasserstein`.
- The Lipschitz constraint on the critic may matter more than the exact distance.

*Does WGAN fix mode collapse?*
- Authors report ❌mode collapse in their experiments.
- ← Critic can train to optimality → $G$ can't exploit a stale critic.
```

&nbsp;

### WGAN-GP
- **Name**: Wasserstein GAN with Gradient Penalty {cite:p}`gulrajani2017improved`
- **What**: WGAN w/ weight clipping → soft penalty on the norm of the critic's input gradient.
- **Why**: Weight clipping → capacity underuse & exploding/vanishing gradients (sensitive to $c$).
    - Optimal critic has gradient norm 1 almost everywhere along lines between coupled real & fake points.
    - → Penalize $\|\nabla f\|\neq1$ on such lines.
- **How**:
    1. Sample real $x$, fake $\tilde x$, $\epsilon\sim U[0,1]$ → $\hat x=\epsilon x+(1-\epsilon)\tilde x$.
    2. Compute $\nabla_{\hat x}f(\hat x)$ w/ autograd. Keep the graph → double backprop.
    3. Add $\lambda\left(\|\nabla_{\hat x}f(\hat x)\|_2-1\right)^2$ to the critic loss. $\lambda=10$, $n_\text{critic}=5$, Adam.
    4. ❌[BatchNorm](../foundation/module.md#batchnorm) in the critic (penalty is per sample) → use [LayerNorm](../foundation/module.md#layernorm).

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\lambda$: Penalty weight. Default 10.
- Misc:
    - $\epsilon\sim U[0,1]$: Interpolation weight, 1 per sample.
    - $\hat x=\epsilon x+(1-\epsilon)\tilde x$: Interpolate between a real & a fake sample.
    - $p_{\hat x}$: Distribution of $\hat x$.

Objective (critic, minimize):

$$
\mathcal{L}_f=\mathbb{E}_{\tilde x\sim p_g}\left[f(\tilde x)\right]-\mathbb{E}_{x\sim p_\text{data}}\left[f(x)\right]+\lambda\,\mathbb{E}_{\hat x\sim p_{\hat x}}\left[\left(\|\nabla_{\hat x}f(\hat x)\|_2-1\right)^2\right]
$$

- Two-sided: pushes the norm *to* 1, ❌only below 1. Converged slightly faster & to better optima (paper).
- $G$ loss unchanged: $-\mathbb{E}_z\left[f(G(z))\right]$.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class GradientPenalty:
    def __init__(self, lam=10.0):
        self.lam = lam

    def __call__(self, f, x, x_fake):
        ## 1 eps per sample, broadcast over all non-batch dims
        eps = torch.rand(x.size(0), *([1] * (x.dim() - 1)))
        x_hat = (eps * x + (1 - eps) * x_fake.detach()).requires_grad_(True)
        ## sum trick: grad of sum = per-sample grads, valid only if f is per-sample (no BatchNorm)
        grad, = torch.autograd.grad(f(x_hat).sum(), x_hat, create_graph=True)  ## keep graph -> penalty trains f
        norm = grad.flatten(1).norm(2, dim=1)
        return self.lam * ((norm - 1) ** 2).mean()

## Example: critic w/ LayerNorm, no BatchNorm, no sigmoid
f = nn.Sequential(nn.Linear(2, 32), nn.LayerNorm(32), nn.LeakyReLU(0.2), nn.Linear(32, 1))
x, x_fake = torch.randn(16, 2), torch.randn(16, 2) + 3
gp = GradientPenalty()(f, x, x_fake)
loss_f = f(x_fake).mean() - f(x).mean() + gp
loss_f.backward()
print(gp.item())  ## scalar >= 0
```
````

```{attention} Q&A
:class: dropdown
*Why penalize on interpolates, ❌everywhere?*
- Enforcing $\|\nabla f\|\leq1$ everywhere is intractable.
- Optimal critic's gradient norm = 1 on lines between coupled points → sample those lines.
- Interpolating random (uncoupled) real-fake pairs is an approximation. It works well in practice.

*Why ❌BatchNorm in the critic?*
- BN maps a *batch* to a batch → $f(\hat x_i)$ depends on other samples.
- Penalty assumes per-sample gradients → BN makes it invalid. LayerNorm normalizes within each sample → OK.

*Cost?*
- Double backprop → ~extra backward pass per critic step.

*Other ways to control the critic's slope?*
- [Spectral Normalization](../foundation/module.md#spectral-normalization): divide each weight by its largest singular value → each layer 1-Lipschitz → $f$ ≤1-Lipschitz (w/ 1-Lipschitz activations). ❌Double backprop.
- R1 penalty: $\frac{\gamma}{2}\mathbb{E}_{x\sim p_\text{data}}\left[\|\nabla_xD(x)\|^2\right]$, $\gamma$ = weight → zero-centered, on real data only. StyleGAN (FFHQ) trains w/ NS loss + R1.

*Pros vs WGAN?*
- Trains many architectures (incl. 101-layer ResNets) w/ almost no hyperparam tuning.
- ❌Clipping → critic uses its capacity.
```

&nbsp;
