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
# Autoencoders & VAE
Encoder–decoder models that learn a compressed latent code. Deterministic (AE, DAE) → probabilistic (VAE & variants) → discrete (VQ-VAE). Family overview → [Generative Models](gen.md).

Notations:
- $x\in\mathbb{R}^d$: Data sample
- $d$: Data dim
- $z\in\mathbb{R}^{d_z}$: Latent code
- $d_z$: Latent dim
- $\hat x$: Reconstruction of $x$
- $f_\phi$: Encoder network w/ params $\phi$
- $g_\theta$: Decoder network w/ params $\theta$
- $p_\text{data}$: Data distribution
- $p(z)=\mathcal{N}(0,I)$: Latent prior
- $q_\phi(z|x)$: Encoder distribution (approx posterior)
- $p_\theta(x|z)$: Decoder distribution (likelihood)
- $\epsilon\sim\mathcal{N}(0,I)$: Standard Gaussian noise
- $I$: Identity matrix
- $D_\text{KL}$: [KL divergence](../../ml/obj.md#kl-divergence)

Overrides of global notation: $x$, $z$ = whole vectors (non-bold). Subscripts on $z$ ($z_e$, $z_q$) name a variant, ❌sample idx.

&nbsp;

## Autoencoder
- **What**: Encoder → low-dim code → decoder, trained to reconstruct its own input.
- **Why**: Learn a compact representation w/o labels.
    - Bottleneck → the code keeps only the information needed to rebuild $x$.
- **How**:
    1. Encode: $z=f_\phi(x)$, $d_z<d$ (undercomplete).
    2. Decode: $\hat x=g_\theta(z)$.
    3. Train: minimize reconstruction loss. [MSE](../../ml/obj.md#mse) for real values, [BCE](../../ml/obj.md#bce) for values in $[0,1]$.

```{note} Math
:class: dropdown
Objective:

$$
\min_{\phi,\theta}\;\mathbb{E}_{x\sim p_\text{data}}\left[\left\|x-g_\theta\left(f_\phi(x)\right)\right\|^2\right]
$$
```

```{tip} Derivation
:class: dropdown
*Why does a linear AE w/ MSE find the PCA subspace?*
1. Linear: $\hat x=W_dW_ex$, $W_e\in\mathbb{R}^{d_z\times d}$, $W_d\in\mathbb{R}^{d\times d_z}$ → $\text{rank}(W_dW_e)\leq d_z$.
2. Centered data matrix $X\in\mathbb{R}^{m\times d}$ (rows = samples) → loss $=\|X-XW_e^TW_d^T\|_F^2$.
3. Eckart–Young: best rank-$d_z$ approx of $X$ in Frobenius norm = truncated SVD $X_{d_z}=XV_{d_z}V_{d_z}^T$.
    - $V_{d_z}\in\mathbb{R}^{d\times d_z}$: top-$d_z$ right singular vectors = top-$d_z$ principal directions.
4. $XV_{d_z}V_{d_z}^T$ has the form $XW_e^TW_d^T$ → it is the optimum.
5. Row space of $X_{d_z}$ = $\text{span}(V_{d_z})$ ⊆ column space of $W_d$. Both have dim $d_z$ → equal (if $\sigma_{d_z}>\sigma_{d_z+1}$).
6. Any invertible $A$: $(W_dA^{-1})(AW_e)$ = same product → same subspace, ❌same axes, ❌orthogonal, ❌ordered by variance.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, d, d_z, h=64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, d_z))   ## f_phi
        self.dec = nn.Sequential(nn.Linear(d_z, h), nn.ReLU(), nn.Linear(h, d))   ## g_theta

    def forward(self, x):
        return self.dec(self.enc(x))          ## d -> d_z (bottleneck) -> d

    def loss(self, x_in, x_target):
        ## target is a separate arg -> DAE (below) reuses this w/ a corrupted x_in
        return (self(x_in) - x_target).pow(2).sum(-1).mean()

## Example
ae = Autoencoder(d=8, d_z=2)
x = torch.randn(4, 8)
print(ae(x).shape, ae.loss(x, x).item() > 0)   ## torch.Size([4, 8]) True
```
````

```{attention} Q&A
:class: dropdown
*AE vs PCA?*
- Linear AE + MSE → same subspace as [PCA](../../ml/unsupervised.md#pca), ❌orthogonal axes (see Derivation).
- Nonlinear AE → curved low-dim manifold → ⬇️reconstruction error at the same $d_z$.

*Is an AE a generative model?*
- ❌ No prior on $z$ → codes fill an irregular region w/ gaps.
- Random $z$ → decoder never saw it → garbage. → Fix: [VAE](#vae).

*What if $d_z\geq d$ (overcomplete)?*
- Can learn the identity → zero loss, useless code.
- Fix: regularize the code.
    - Sparse AE: L1 penalty on $z$ → few active units → [SAE](../repr/interp.md#sae).
    - Denoising AE: corrupt the input → [DAE](#denoising-autoencoder).
    - Contractive AE: penalize $\|\partial f_\phi(x)/\partial x\|_F^2$ → code insensitive to small input changes.

*Why are MSE reconstructions blurry?*
- MSE = Gaussian NLL → optimum = conditional mean of all plausible $x$ → averages fine detail away.
- Fix: perceptual + adversarial losses (e.g., the autoencoder in [LDM](diffusion.md#ldm)).

*Uses?*
- Dimensionality reduction, compression, pretraining.
- Anomaly detection: high reconstruction error → input unlike the training data.
- Latent space for a 2nd-stage generative model → [LDM](diffusion.md#ldm).
```

&nbsp;

### Denoising Autoencoder
- **What**: AE trained to reconstruct the clean $x$ from a corrupted copy $\tilde x$. {cite:p}`vincent2008extracting`
- **Why**: A high-capacity AE can copy its input → useless features.
    - Denoising cannot be solved by copying → the net must learn the structure of the data.
- **How**:
    1. Corrupt: $\tilde x\sim C(\tilde x|x)$.
        - Masking noise: set a random fraction of inputs to 0 (original paper).
        - Additive Gaussian noise. Salt-&-pepper noise.
    2. Encode $\tilde x$ → decode → $\hat x$.
    3. Loss vs the **clean** $x$.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $C(\tilde x|x)$: Corruption distribution.
    - $\sigma$: Gaussian corruption std.
    - $p_\sigma$: Density of $\tilde x$ under Gaussian corruption.
    - $r(\tilde x)=g_\theta(f_\phi(\tilde x))$: Denoiser output.

Objective:

$$
\min_{\phi,\theta}\;\mathbb{E}_{x\sim p_\text{data},\,\tilde x\sim C(\tilde x|x)}\left[\left\|x-r(\tilde x)\right\|^2\right]
$$

Gaussian corruption $\tilde x=x+\sigma\epsilon$ → optimal denoiser (Tweedie's formula):

$$
r^*(\tilde x)=\mathbb{E}[x|\tilde x]=\tilde x+\sigma^2\nabla_{\tilde x}\log p_\sigma(\tilde x)
$$

- $p_\sigma(\tilde x)=\int p_\text{data}(x)\,\mathcal{N}(\tilde x;x,\sigma^2I)\,dx$: data smoothed by noise.
- $r^*(\tilde x)-\tilde x\propto$ score of $p_\sigma$ → DAE = [DSM](diffusion.md#dsm) {cite:p}`vincent2011connection`.
```

```{tip} Derivation
:class: dropdown
*Why does the optimal denoiser point along the score?*
1. MSE optimum = conditional mean: $r^*(\tilde x)=\mathbb{E}[x|\tilde x]$.
2. $\nabla_{\tilde x}\mathcal{N}(\tilde x;x,\sigma^2I)=\mathcal{N}(\tilde x;x,\sigma^2I)\,\frac{x-\tilde x}{\sigma^2}$.
3. $\nabla_{\tilde x}p_\sigma(\tilde x)=\int p_\text{data}(x)\,\mathcal{N}(\tilde x;x,\sigma^2I)\,\frac{x-\tilde x}{\sigma^2}\,dx$.
4. Divide by $p_\sigma(\tilde x)$. Bayes: $p_\text{data}(x)\mathcal{N}(\tilde x;x,\sigma^2I)/p_\sigma(\tilde x)=p(x|\tilde x)$.
5. → $\nabla_{\tilde x}\log p_\sigma(\tilde x)=\frac{\mathbb{E}[x|\tilde x]-\tilde x}{\sigma^2}$ → rearrange.
```

````{important} Code
:class: dropdown
```python
import torch

## Reuses Autoencoder from Autoencoder above
class DenoisingAutoencoder(Autoencoder):
    def __init__(self, d, d_z, mask_p=0.3):
        super().__init__(d, d_z)
        self.mask_p = mask_p

    def corrupt(self, x):
        ## masking noise: zero each input dim w/ prob mask_p
        return x * (torch.rand_like(x) > self.mask_p)

    def denoise_loss(self, x):
        return self.loss(self.corrupt(x), x)   ## input = corrupted, target = CLEAN x

## Example
dae = DenoisingAutoencoder(d=8, d_z=4)
x = torch.randn(4, 8)
print(dae.denoise_loss(x).item() > 0)   ## True
```
````

```{attention} Q&A
:class: dropdown
*Why reconstruct the clean $x$, ❌the corrupted $\tilde x$?*
- Target $\tilde x$ → identity map is optimal again → nothing learned.

*Geometric intuition?*
- Data lies near a low-dim manifold. Corruption pushes points off it.
- DAE learns a vector field that points back to the manifold → learns the manifold's shape.

*How to pick the noise level?*
- Too small → ≈ identity → weak features.
- Too large → only coarse structure survives.
- Diffusion avoids the choice: 1 net trained at all noise levels.

*How does DAE relate to modern models?*
- [Diffusion](diffusion.md#diffusion-model) = DAE at many Gaussian noise levels + iterative sampler.
- [BERT](../../nlp/models.md#bert) masked LM = DAE w/ masking noise on tokens.
- [MAE](../../cv/vit.md#mae) = DAE w/ masking noise on image patches.

*Pros?*
- Robust features. Works overcomplete. No extra penalty term.

*Cons?*
- Corruption type = hand-designed prior. Wrong corruption → wrong invariances.
- Still ❌generative by itself: no prior on $z$.
```

&nbsp;

## VAE
- **Name**: Variational Autoencoder {cite:p}`kingma2014auto`
- **What**: Latent-variable generative model $p_\theta(x|z)p(z)$ + amortized encoder $q_\phi(z|x)$, trained jointly on the ELBO.
- **Why**:
    - *Why do we need it?*
        - AE: no density on $z$ → cannot sample.
        - Latent-variable model: $p_\theta(x)=\int p_\theta(x|z)p(z)dz$ intractable w/ a NN decoder → ❌direct MLE.
        - Posterior $p_\theta(z|x)$ intractable → ❌[EM](../../ml/optim.md#em).
    - *Why does it work?*
        1. ELBO = tractable lower bound of $\log p_\theta(x)$, computed w/ samples from $q_\phi$.
        2. [Reparameterization](#reparameterization-trick) → low-variance grads for $\phi$ & $\theta$ → plain SGD.
        3. KL term pulls every $q_\phi(z|x)$ toward $p(z)$ → codes fill the prior → $z\sim p(z)$ decodes to valid $x$.
- **How**:
    - **Training**:
        1. Encoder outputs mean & log-variance of $q_\phi(z|x)$.
        2. Sample $z=\mu+\sigma\odot\epsilon$.
        3. Decode $z$ → reconstruction loss.
        4. Loss = reconstruction + $D_\text{KL}(q_\phi(z|x)\|p(z))$ = $-$ELBO.
    - **Generation**: $z\sim\mathcal{N}(0,I)$ → decoder → $x$ (usually the decoder mean).

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\mu_\phi(x)\in\mathbb{R}^{d_z}$: Encoder mean.
    - $\sigma_\phi(x)\in\mathbb{R}^{d_z}$: Encoder std (net outputs $\log\sigma_\phi^2$).
- Hyperparams:
    - $\sigma_x^2$: Decoder variance (Gaussian decoder).
- Misc:
    - $\mathcal{L}(\theta,\phi;x)$: ELBO.

Model:

$$
p_\theta(x,z)=p_\theta(x|z)\,p(z),\quad p(z)=\mathcal{N}(0,I),\quad p_\theta(x|z)=\mathcal{N}\left(x;g_\theta(z),\sigma_x^2I\right)
$$

Encoder:

$$
q_\phi(z|x)=\mathcal{N}\left(z;\mu_\phi(x),\text{diag}\left(\sigma_\phi^2(x)\right)\right)
$$

Training (maximize):

$$
\mathcal{L}(\theta,\phi;x)=\underbrace{\mathbb{E}_{q_\phi(z|x)}\left[\log p_\theta(x|z)\right]}_{\text{reconstruction}}-\underbrace{D_\text{KL}\left(q_\phi(z|x)\,\|\,p(z)\right)}_{\text{regularizer}}\;\leq\;\log p_\theta(x)
$$

- Gaussian decoder: $-\log p_\theta(x|z)=\frac{1}{2\sigma_x^2}\|x-g_\theta(z)\|^2+\frac{d}{2}\log(2\pi\sigma_x^2)$ → MSE.
- Bernoulli decoder → BCE.
- Expectation: 1 Monte Carlo sample of $z$ per $x$ is enough for a large batch (paper: batch 100).

Closed-form KL (diagonal Gaussian vs $\mathcal{N}(0,I)$):

$$
D_\text{KL}\left(q_\phi(z|x)\,\|\,\mathcal{N}(0,I)\right)=\frac{1}{2}\sum_{j=1}^{d_z}\left(\mu_j^2+\sigma_j^2-\log\sigma_j^2-1\right)
$$

- $\mu_j,\sigma_j$: $j$-th dims of $\mu_\phi(x),\sigma_\phi(x)$.
```

```{tip} Derivation
:class: dropdown
*Form 1: Jensen. Why is the ELBO a lower bound?*
1. Insert $q_\phi$: $\log p_\theta(x)=\log\int q_\phi(z|x)\frac{p_\theta(x,z)}{q_\phi(z|x)}dz=\log\mathbb{E}_{q_\phi}\left[\frac{p_\theta(x,z)}{q_\phi(z|x)}\right]$.
2. $\log$ concave → Jensen: $\log\mathbb{E}[\cdot]\geq\mathbb{E}[\log\cdot]$.
3. → $\log p_\theta(x)\geq\mathbb{E}_{q_\phi}\left[\log p_\theta(x,z)-\log q_\phi(z|x)\right]$.
4. Split $p_\theta(x,z)=p_\theta(x|z)p(z)$ → $\mathbb{E}_{q_\phi}[\log p_\theta(x|z)]-D_\text{KL}(q_\phi(z|x)\|p(z))$.

*Form 2: KL gap. How loose is the bound?*
1. $\log p_\theta(x)$ does not depend on $z$ → $\log p_\theta(x)=\mathbb{E}_{q_\phi}[\log p_\theta(x)]$.
2. Bayes: $p_\theta(x)=\frac{p_\theta(x,z)}{p_\theta(z|x)}$ → multiply & divide by $q_\phi(z|x)$:

    $$
    \log p_\theta(x)=\mathbb{E}_{q_\phi}\left[\log\frac{p_\theta(x,z)}{q_\phi(z|x)}\right]+\mathbb{E}_{q_\phi}\left[\log\frac{q_\phi(z|x)}{p_\theta(z|x)}\right]=\mathcal{L}+D_\text{KL}\left(q_\phi(z|x)\,\|\,p_\theta(z|x)\right)
    $$

3. KL ≥ 0 → $\mathcal{L}\leq\log p_\theta(x)$. Gap = KL to the true posterior.
4. → Tight iff $q_\phi(z|x)=p_\theta(z|x)$.
5. $\log p_\theta(x)$ independent of $\phi$ → $\max_\phi\mathcal{L}\Leftrightarrow\min_\phi D_\text{KL}(q_\phi\|p_\theta(z|x))$ = variational inference.

*Closed-form KL, 1 dim*:
1. $D_\text{KL}=\mathbb{E}_q[\log q(z)-\log p(z)]$, $q=\mathcal{N}(\mu,\sigma^2)$, $p=\mathcal{N}(0,1)$.
2. $\mathbb{E}_q[\log q]=-\frac{1}{2}\log(2\pi\sigma^2)-\frac{1}{2}$ ← $\mathbb{E}_q[(z-\mu)^2]=\sigma^2$.
3. $\mathbb{E}_q[\log p]=-\frac{1}{2}\log(2\pi)-\frac{1}{2}(\mu^2+\sigma^2)$ ← $\mathbb{E}_q[z^2]=\mu^2+\sigma^2$.
4. Subtract → $\frac{1}{2}(\mu^2+\sigma^2-\log\sigma^2-1)$.
5. Diagonal covariance → dims independent → sum over $j$.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, d, d_z, h=64):
        super().__init__()
        self.d_z = d_z
        self.enc = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, 2 * d_z))  ## -> (mu, log sigma^2)
        self.dec = nn.Sequential(nn.Linear(d_z, h), nn.ReLU(), nn.Linear(h, d))      ## -> decoder mean g(z)

    def forward(self, x):
        mu, logvar = self.enc(x).chunk(2, dim=-1)                ## (B, d_z) each
        ## reparameterization: randomness lives in eps -> grads reach mu & logvar
        z = mu + (0.5 * logvar).exp() * torch.randn_like(mu)
        return self.dec(z), mu, logvar

    def loss(self, x, beta=1.0):
        x_hat, mu, logvar = self(x)
        ## -log p(x|z), Gaussian decoder w/ sigma_x = 1, constant dropped
        rec = 0.5 * (x - x_hat).pow(2).sum(-1)
        ## closed-form KL(N(mu, sigma^2) || N(0, I)), summed over latent dims
        kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).sum(-1)
        return (rec + beta * kl).mean()                          ## beta = 1 -> -ELBO (+ const)

    @torch.no_grad()
    def sample(self, n):
        return self.dec(torch.randn(n, self.d_z))                ## z ~ p(z) -> decoder mean

## Example
vae = VAE(d=8, d_z=2)
x = torch.randn(4, 8)
print(vae.loss(x).item() > 0, vae.sample(3).shape)   ## True torch.Size([3, 8])
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- Principled: maximizes a bound on $\log p_\theta(x)$ → good mode coverage.
- Encoder → $x\to z$ inference in 1 pass. Smooth latent space → interpolation.
- Stable training: no minimax. Fast 1-pass sampling.

*Cons?*
- Blurry samples.
- Bound, ❌exact likelihood.
- Posterior collapse w/ strong decoders.

*Why are VAE samples blurry?*
- Gaussian decoder w/ fixed variance → MSE → decoder outputs the mean of all $x$ consistent w/ $z$.
- KL limits the information in $z$ → many $x$ share 1 code → the mean averages them.

*What is posterior collapse?*
- $q_\phi(z|x)\approx p(z)$ for all $x$ → KL ≈ 0 → decoder ignores $z$.
- Cause: a powerful autoregressive decoder models $p(x)$ alone. Using $z$ costs KL & gives little gain → optimizer drops $z$.
- Fixes:
    - KL annealing ($\beta$: 0 → 1) & word dropout in the decoder {cite:p}`bowman2016generating`.
    - Free bits: no KL penalty below a floor per latent group.
    - Weaker decoder. Discrete latents ([VQ-VAE](#vq-vae)).

*Why does the KL term hurt the code?*
- Average over data: $\mathbb{E}_{p_\text{data}}D_\text{KL}(q_\phi(z|x)\|p(z))=I_q(x;z)+D_\text{KL}(q_\phi(z)\|p(z))$.
    - $q_\phi(z)=\mathbb{E}_{p_\text{data}}[q_\phi(z|x)]$: aggregate posterior.
    - $I_q(x;z)$: mutual information between $x$ & $z$ under $q_\phi$.
- → The KL term penalizes information in $z$ itself. Reconstruction wants more → tension.

*What is the prior hole problem?*
- Aggregate posterior $q_\phi(z)\neq p(z)$ → regions of high $p(z)$ that no $x$ encodes to.
- Sample from such a hole → decoder never trained there → bad sample.
- Fix: learned prior over $z$ (autoregressive prior in [VQ-VAE](#vq-vae), diffusion prior in [LDM](diffusion.md#ldm)).

*Where does the ELBO gap come from?*
- Approximation gap: Gaussian $q$ cannot match the true posterior.
- Amortization gap: 1 shared encoder vs the best $q$ per $x$.

*Why reverse KL $D_\text{KL}(q\|p_\theta(z|x))$ in VI?*
- It needs samples only from $q$ & the unnormalized $p_\theta(x,z)$ → tractable.
- Side effect: mode-seeking → $q$ underestimates posterior variance.

*VAE vs EM?*
- [EM](../../ml/optim.md#em): E-step sets $q$ = exact posterior → gap = 0.
- VAE: posterior intractable → amortized $q_\phi$ → both steps = 1 SGD step on $\mathcal{L}$.

*Deterministic limit?*
- $\sigma_\phi\to0$ → KL ⬆️ w/o bound ($-\log\sigma^2$ term) → VAE forbids a deterministic encoder.
- Drop the KL & the noise → plain [AE](#autoencoder).

*How does it relate to diffusion?*
- [Diffusion](diffusion.md#diffusion-model) = hierarchical VAE w/ a fixed, ❌learned encoder → ❌posterior collapse.

*What does this tell us about intelligence/AC?*
- $-$ELBO = variational free energy → [Predictive Processing](../../ac/theories.md#predictive-processing-pp) says the brain minimizes the same kind of quantity.
- Encoder = amortized perception: 1 pass replaces iterative inference.
```

&nbsp;

### Reparameterization Trick
- **What**: Write $z\sim q_\phi(z|x)$ as a differentiable function of $\phi$ & param-free noise $\epsilon$. {cite:p}`kingma2014auto`
- **Why**: The ELBO needs $\nabla_\phi\mathbb{E}_{q_\phi(z|x)}[f(z)]$. The sampling distribution depends on $\phi$ → cannot backprop through a sample.
    - Score-function estimator ([REINFORCE](../../rl/policy.md#reinforce)) works, but its variance is high.
- **How**:
    1. Sample noise $\epsilon\sim\mathcal{N}(0,I)$. Its distribution does not depend on $\phi$.
    2. Transform: $z=\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon$ → deterministic & differentiable in $\phi$.
    3. Expectation now over fixed $p(\epsilon)$ → move $\nabla_\phi$ inside → backprop.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $f(z)$: Any differentiable function of $z$ (e.g., $\log p_\theta(x|z)$).

Reparameterized gradient:

$$
\nabla_\phi\mathbb{E}_{q_\phi(z|x)}\left[f(z)\right]=\mathbb{E}_{\epsilon\sim\mathcal{N}(0,I)}\left[\nabla_\phi f\left(\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon\right)\right]
$$

Score-function gradient (for comparison):

$$
\nabla_\phi\mathbb{E}_{q_\phi(z|x)}\left[f(z)\right]=\mathbb{E}_{q_\phi(z|x)}\left[f(z)\,\nabla_\phi\log q_\phi(z|x)\right]
$$

- Both unbiased. Reparam uses $\nabla_zf$. Score-function uses only the value $f(z)$.
```

```{tip} Derivation
:class: dropdown
*Why can $\nabla_\phi$ move inside after reparameterization?*
1. $z=\mu_\phi+\sigma_\phi\odot\epsilon$, $\epsilon\sim\mathcal{N}(0,I)$ → $z\sim q_\phi(z|x)$.
2. Change of variables: $\mathbb{E}_{q_\phi(z|x)}[f(z)]=\mathbb{E}_{\epsilon}[f(\mu_\phi+\sigma_\phi\odot\epsilon)]$.
3. $p(\epsilon)$ independent of $\phi$ → swap $\nabla_\phi$ & $\mathbb{E}_\epsilon$ → chain rule through $z$.

*Where does the score-function form come from?*
1. $\nabla_\phi\int q_\phi(z)f(z)dz=\int f(z)\nabla_\phi q_\phi(z)dz$.
2. $\nabla_\phi q_\phi=q_\phi\nabla_\phi\log q_\phi$ → $\mathbb{E}_{q_\phi}[f(z)\nabla_\phi\log q_\phi(z)]$.
```

```{note} Example
:class: dropdown
$f(z)=z^2$, $q=\mathcal{N}(\mu,1)$, $z=\mu+\epsilon$. True grad: $\frac{d}{d\mu}\mathbb{E}[z^2]=\frac{d}{d\mu}(\mu^2+1)=2\mu$.
- Reparam estimator: $2z=2(\mu+\epsilon)$ → variance $=4$ for every $\mu$.
- Score-function estimator: $f(z)\cdot\partial_\mu\log q=(\mu+\epsilon)^2\epsilon$ → variance $=\mu^4+14\mu^2+15$.
    - $\mu=0$: 15 vs 4. $\mu=2$: 87 vs 4.
```

````{important} Code
:class: dropdown
```python
import torch

class GaussianReparam:
    def __call__(self, mu, logvar):
        ## all randomness in eps ~ N(0, I), whose law does not depend on (mu, logvar)
        eps = torch.randn_like(mu)
        return mu + (0.5 * logvar).exp() * eps   ## z: differentiable in mu & logvar

## Example: grad of E[z^2] wrt mu, z ~ N(mu, 1), mu = 2 -> true grad = 4
mu = torch.full((100_000,), 2.0, requires_grad=True)
z = GaussianReparam()(mu, torch.zeros_like(mu))
z.pow(2).sum().backward()
rep = mu.grad                                    ## per-sample reparam estimates: 2z
score = z.detach().pow(2) * (z - mu).detach()   ## REINFORCE: f(z) * d/dmu log q = z^2 * eps
print(rep.mean().item(), rep.var().item())       ## ~4.0 ~4.0
print(score.mean().item(), score.var().item())   ## ~4.0 ~87
```
````

```{attention} Q&A
:class: dropdown
*Why is the reparam gradient usually lower-variance?*
- It uses $\nabla_zf$ → each sample says which direction improves $f$.
- Score-function uses only the scalar $f(z)$ × a random direction $\nabla_\phi\log q$. Large $|f|$ → large variance. Needs baselines.
- Caveat: lower variance is typical for smooth $f$, ❌guaranteed. Counterexamples exist.

*Requirements?*
- $f$ differentiable in $z$.
- $q$ reparameterizable: location-scale families (Gaussian, Laplace) or an invertible CDF transform.

*What if $z$ is discrete?*
- ❌ Reparam: no differentiable path from $\phi$ to a discrete sample.
- Options: score-function + baseline, Gumbel-Softmax relaxation, straight-through estimator ([VQ-VAE](#vq-vae)).

*Why does RL use the score-function estimator?*
- Reward & environment dynamics are not differentiable in the action → no $\nabla_zf$ → [REINFORCE](../../rl/policy.md#reinforce).

*Where else does this trick appear?*
- Diffusion: $x_t=a_tx_0+\sigma_t\epsilon$ ([Forward Process](diffusion.md#forward-process)).
- Bayesian NNs & stochastic policies (SAC): sample params/actions as $\mu+\sigma\epsilon$.
```

&nbsp;

### Beta-VAE
- **What**: VAE w/ the KL term weighted by $\beta$ ($\beta>1$ in the original). {cite:p}`higgins2017beta`
- **Why**: Disentanglement: 1 latent dim ⇔ 1 generative factor (pose, color, size).
    - $\beta>1$ → stronger pressure toward the factorized prior $\mathcal{N}(0,I)$ → less information in $z$ → more efficient, factorized code.
- **How**: Loss = reconstruction + $\beta\cdot$KL.
    - $\beta=1$ → standard VAE.
    - $\beta>1$ → ⬆️disentanglement, ⬆️blur.
    - $\beta<1$ → sharper reconstruction, weaker match to the prior.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\beta\geq0$: KL weight.
    - $\delta$: KL budget (constrained form).

Objective (maximize):

$$
\mathcal{L}_\beta=\mathbb{E}_{q_\phi(z|x)}\left[\log p_\theta(x|z)\right]-\beta\,D_\text{KL}\left(q_\phi(z|x)\,\|\,p(z)\right)
$$

- $\beta\geq1$ → $\mathcal{L}_\beta\leq\mathcal{L}\leq\log p_\theta(x)$ → still a bound.
- $\beta<1$ → ❌bound on $\log p_\theta(x)$.

Constrained origin:

$$
\max_{\phi,\theta}\;\mathbb{E}_{q_\phi(z|x)}\left[\log p_\theta(x|z)\right]\quad\text{s.t.}\quad D_\text{KL}\left(q_\phi(z|x)\,\|\,p(z)\right)<\delta
$$

- Lagrangian (KKT) → $\beta$ = multiplier of the KL constraint.
```

```{attention} Q&A
:class: dropdown
*Why does $\beta>1$ disentangle?*
- Prior $\mathcal{N}(0,I)$ is isotropic & factorized. Heavy KL → each latent dim must pay for every bit it carries.
- → Code uses few dims, each aligned w/ a factor that buys the most reconstruction.
- Caveat: the mechanism is contested. Unsupervised disentanglement is impossible w/o inductive biases on model & data {cite:p}`locatello2019challenging`.

*Cost?*
- ⬇️Reconstruction quality → blurrier samples.
- $\beta$ needs tuning per dataset. Disentanglement varies a lot across random seeds.

*Rate–distortion view?*
- KL = rate (information in $z$). Reconstruction NLL = distortion.
- $\beta$ selects a point on the rate–distortion curve. Same ELBO value can hide very different rate/distortion splits.

*When is $\beta<1$ useful?*
- When $z$ feeds a 2nd-stage model, ❌sampling from $p(z)$.
- E.g., [LDM](diffusion.md#ldm) autoencoders: tiny KL weight → sharp reconstructions; the diffusion prior handles $p(z)$.

*Relation to posterior collapse?*
- $\beta$ annealed 0 → 1 = KL annealing → counters collapse.
- $\beta\gg1$ → pushes toward collapse.
```

&nbsp;

### CVAE
- **Name**: Conditional Variational Autoencoder {cite:p}`sohn2015learning`
- **What**: VAE w/ every distribution conditioned on $c$: prior $p_\psi(z|c)$, encoder $q_\phi(z|x,c)$, decoder $p_\theta(x|z,c)$.
- **Why**: Model a 1-to-many map $c\mapsto x$ & control what is generated.
    - Deterministic regression $\hat x=h(c)$ → averages all valid outputs.
    - Latent $z$ → selects 1 mode → diverse, sharp outputs per $c$.
- **How**:
    - **Training**:
        1. Recognition net sees $(x,c)$ → $q_\phi(z|x,c)$ → sample $z$.
        2. Generation net decodes $(z,c)$ → reconstruction loss.
        3. KL pulls $q_\phi(z|x,c)$ toward the prior net $p_\psi(z|c)$.
    - **Generation**: $z\sim p_\psi(z|c)$ → decode w/ $c$.
    - Simplification: fixed prior $\mathcal{N}(0,I)$, independent of $c$.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $c$: Condition.
- Params:
    - $\psi$: Prior network params.
    - $\mu_\psi(c),\sigma_\psi(c)$: Prior mean & std.

Model:

$$
p_{\theta,\psi}(x|c)=\int p_\theta(x|z,c)\,p_\psi(z|c)\,dz
$$

Training (maximize):

$$
\log p_{\theta,\psi}(x|c)\;\geq\;\mathbb{E}_{q_\phi(z|x,c)}\left[\log p_\theta(x|z,c)\right]-D_\text{KL}\left(q_\phi(z|x,c)\,\|\,p_\psi(z|c)\right)
$$

- Paper notation: input $x$, output $y$. Here: condition $c$, output $x$.

KL of 2 diagonal Gaussians:

$$
D_\text{KL}\left(\mathcal{N}(\mu_1,\sigma_1^2)\,\|\,\mathcal{N}(\mu_2,\sigma_2^2)\right)=\frac{1}{2}\sum_{j=1}^{d_z}\left(\log\frac{\sigma_{2,j}^2}{\sigma_{1,j}^2}+\frac{\sigma_{1,j}^2+(\mu_{1,j}-\mu_{2,j})^2}{\sigma_{2,j}^2}-1\right)
$$

- $(\mu_1,\sigma_1)$: from $q_\phi(z|x,c)$. $(\mu_2,\sigma_2)$: from $p_\psi(z|c)$.
- $\mu_2=0$, $\sigma_2=1$ → VAE's KL.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def mlp(i, o, h=64):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, o))

class CVAE(nn.Module):
    def __init__(self, d, d_c, d_z):
        super().__init__()
        self.enc = mlp(d + d_c, 2 * d_z)     ## q(z|x,c): recognition net
        self.prior = mlp(d_c, 2 * d_z)       ## p(z|c): prior net
        self.dec = mlp(d_z + d_c, d)         ## p(x|z,c): generation net

    def loss(self, x, c):
        mu_q, lv_q = self.enc(torch.cat([x, c], -1)).chunk(2, -1)
        mu_p, lv_p = self.prior(c).chunk(2, -1)
        z = mu_q + (0.5 * lv_q).exp() * torch.randn_like(mu_q)          ## reparam
        rec = 0.5 * (x - self.dec(torch.cat([z, c], -1))).pow(2).sum(-1)
        ## KL of 2 diagonal Gaussians: q(z|x,c) || p(z|c)
        kl = 0.5 * (lv_p - lv_q + (lv_q.exp() + (mu_q - mu_p).pow(2)) / lv_p.exp() - 1).sum(-1)
        return (rec + kl).mean()

    @torch.no_grad()
    def sample(self, c):
        mu_p, lv_p = self.prior(c).chunk(2, -1)
        z = mu_p + (0.5 * lv_p).exp() * torch.randn_like(mu_p)          ## test time: ❌x -> prior net
        return self.dec(torch.cat([z, c], -1))

## Example: 3 classes as one-hot conditions
cvae = CVAE(d=8, d_c=3, d_z=2)
x, c = torch.randn(4, 8), F.one_hot(torch.tensor([0, 1, 2, 0]), 3).float()
print(cvae.loss(x, c).item() > 0, cvae.sample(c).shape)   ## True torch.Size([4, 8])
```
````

```{attention} Q&A
:class: dropdown
*Why a learned prior $p_\psi(z|c)$, ❌fixed $\mathcal{N}(0,I)$?*
- $c$ changes which $z$ are plausible → the prior should depend on $c$.
- Fixed prior still works; it forces the decoder to do all the conditioning.

*Train/test mismatch?*
- Train: $z$ from $q_\phi(z|x,c)$, which sees the target $x$.
- Test: $z$ from $p_\psi(z|c)$ only. Weak KL match → test-time $z$ off the training distribution.
- Paper's fix: also train the "GSNN" path ($z$ from the prior net during training) → hybrid objective.

*Decoder ignores $z$?*
- Strong $c$ → decoder predicts $x$ from $c$ alone → posterior collapse → output ≈ mean over modes again.

*CVAE vs cGAN?*
- CVAE: stable, mode coverage, blurrier.
- [cGAN](gan.md#cgan): sharp, risk of ignoring noise → low diversity per $c$.
```

&nbsp;

## VQ-VAE
- **Name**: Vector Quantised-Variational AutoEncoder {cite:p}`oord2017neural`
- **What**: AE w/ a discrete latent: each encoder output vector snaps to its nearest codebook vector.
- **Why**:
    - Discrete codes → a powerful autoregressive prior (PixelCNN, transformer) models them like tokens.
    - Many modalities are naturally discrete (phonemes, words, symbols).
    - Avoids KL-driven posterior collapse: KL is constant → no KL cost for using $z$.
- **How**:
    - **Training**:
        1. Encode: $z_e=f_\phi(x)$ = grid of $D$-dim vectors.
        2. Quantize: replace each vector by its nearest codebook entry $e_k$ → $z_q$.
        3. Decode: $\hat x=g_\theta(z_q)$.
        4. Backprop: copy the grad at the decoder input to the encoder output (**straight-through**).
        5. Codebook loss (or EMA) moves $e_k$ → $z_e$. **Commitment loss** moves $z_e$ → $e_k$.
    - **Generation** (2 stages):
        1. Freeze the VQ-VAE. Train an autoregressive prior over the code indices.
        2. Sample indices from the prior → look up codes → decode.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $z_e=f_\phi(x)\in\mathbb{R}^{S\times D}$: Encoder output, $S$ vectors (e.g., $32\times32$ grid).
    - $z_q$: Quantized latent, same shape.
- Params:
    - $e_k\in\mathbb{R}^D$: Codebook vector, $k=1..K$.
- Hyperparams:
    - $K$: Codebook size (paper: 512).
    - $\beta$: Commitment weight (paper: 0.25). Unrelated to [Beta-VAE](#beta-vae)'s $\beta$.
    - $\gamma$: EMA decay (paper: 0.99).
- Misc:
    - $\text{sg}[\cdot]$: Stop-gradient. Identity forward, zero grad backward.

Quantization (per vector):

$$
k^*=\arg\min_k\left\|z_e-e_k\right\|_2,\qquad z_q=e_{k^*}
$$

Training (minimize, per vector, averaged over $S$):

$$
L=-\log p_\theta(x|z_q)+\left\|\text{sg}[z_e]-e_{k^*}\right\|_2^2+\beta\left\|z_e-\text{sg}[e_{k^*}]\right\|_2^2
$$

- Term 1 (reconstruction) → decoder & encoder (via straight-through).
- Term 2 (codebook) → codebook only.
- Term 3 (commitment) → encoder only.

Straight-through:

$$
\text{forward: } z_q,\qquad\text{backward: }\nabla_{z_e}L:=\nabla_{z_q}L\quad\Leftrightarrow\quad z_q^\text{st}=z_e+\text{sg}[z_q-z_e]
$$

EMA codebook update (replaces term 2):

$$
N_k\leftarrow\gamma N_k+(1-\gamma)\,n_k,\qquad m_k\leftarrow\gamma m_k+(1-\gamma)\sum_{i:\,k^*_i=k}z_{e,i},\qquad e_k\leftarrow\frac{m_k}{N_k}
$$

- $n_k$: #encoder vectors in the batch assigned to $e_k$.
- $N_k$: EMA of $n_k$.
- $m_k$: EMA of the sum of the assigned vectors.
- $z_{e,i}$: $i$-th encoder vector in the batch.

ELBO view:
- $q(z=k|x)=\mathbb{1}[k=k^*]$ (deterministic one-hot). Uniform prior $p(z=k)=1/K$.
- → $D_\text{KL}=\log K$ per vector → constant → dropped from training.
```

```{tip} Derivation
:class: dropdown
*Why is the EMA update an online k-means?*
1. Codebook loss for $e_k$: $\sum_{i:\,k^*_i=k}\|z_{e,i}-e_k\|^2$.
2. Set grad to 0 → $e_k=\frac{1}{n_k}\sum_{i:\,k^*_i=k}z_{e,i}$ = mean of the assigned vectors = k-means centroid step.
3. Minibatch → the exact mean is unavailable → track numerator ($m_k$) & denominator ($N_k$) by EMA.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, K, D, beta=0.25, ema=False, gamma=0.99):
        super().__init__()
        self.codebook = nn.Embedding(K, D)                    ## e_k in R^D
        self.codebook.weight.data.uniform_(-1 / K, 1 / K)
        self.beta, self.ema, self.gamma = beta, ema, gamma
        if ema:
            self.codebook.weight.requires_grad_(False)        ## EMA replaces the codebook loss
            self.register_buffer("N", torch.ones(K))          ## init 1 -> unused code keeps its e_k
            self.register_buffer("m", self.codebook.weight.data.clone())

    def forward(self, z_e):                                   ## z_e: (S, D); flatten (B, D, H, W) first
        idx = torch.cdist(z_e, self.codebook.weight).argmin(-1)   ## (S,) nearest code
        z_q = self.codebook(idx)                              ## (S, D)
        ## squared L2 summed over D, averaged over S (as in the paper) -> beta keeps its meaning vs recon
        sq = lambda a, b: (a - b).pow(2).sum(-1).mean()
        loss = self.beta * sq(z_e, z_q.detach())              ## commitment: z_e -> sg[e]
        if not self.ema:
            loss = loss + sq(z_q, z_e.detach())               ## codebook: e -> sg[z_e]
        elif self.training:
            self._ema_update(z_e.detach(), idx)
        ## straight-through: value = z_q, grad wrt z_e = grad wrt z_q
        return z_e + (z_q - z_e).detach(), idx, loss

    @torch.no_grad()
    def _ema_update(self, z_e, idx):
        onehot = F.one_hot(idx, self.N.numel()).type_as(z_e)               ## (S, K)
        self.N.mul_(self.gamma).add_(onehot.sum(0), alpha=1 - self.gamma)   ## N_k
        self.m.mul_(self.gamma).add_(onehot.T @ z_e, alpha=1 - self.gamma)  ## m_k: (K, D)
        self.codebook.weight.copy_(self.m / self.N.unsqueeze(1))            ## e_k = m_k / N_k

## Example
vq = VectorQuantizer(K=8, D=4)
z_e = torch.randn(6, 4, requires_grad=True)
z_q, idx, loss = vq(z_e)
(z_q.pow(2).sum() + loss).backward()          ## stand-in reconstruction loss
print(z_q.shape, idx.shape, z_e.grad.shape)   ## torch.Size([6, 4]) torch.Size([6]) torch.Size([6, 4])
print(VectorQuantizer(K=8, D=4, ema=True)(torch.randn(6, 4))[2].item() >= 0)   ## True
```
````

```{attention} Q&A
:class: dropdown
*Why does straight-through work at all?*
- $\arg\min$ is piecewise constant → true grad = 0 almost everywhere → useless.
- STE copies the decoder-input grad to $z_e$. Biased, but $z_e$ & $z_q$ share 1 space → the grad still says how $z_e$ should move.
- Commitment loss keeps $z_e\approx z_q$ → small bias.

*Why the commitment loss?*
- Codebook term moves only $e_k$. Encoder outputs can drift & grow w/o bound if the codebook lags.
- Commitment term ties $z_e$ to its code.
- Paper: results robust for $\beta\in[0.1,2.0]$.

*Why stop-gradients on both terms?*
- Each term trains exactly 1 part: codebook ← term 2, encoder ← term 3, decoder ← term 1.
- No sg → encoder & codebook pull toward each other at the same rate → no anchor.

*What is codebook collapse?*
- Few codes get all assignments → dead codes → ⬇️capacity. Track codebook usage (perplexity).
- Fixes: EMA updates, k-means init, restart dead codes at random encoder outputs, low-dim or $\ell_2$-normalized codes.
- Alternative: FSQ (round each dim to a few fixed levels) → no codebook to collapse.

*Is VQ-VAE really a VAE?*
- Formally: one-hot $q$, uniform prior → ELBO = reconstruction + constant.
- In practice: AE w/ a discrete bottleneck + a separately trained prior.

*Does VQ-VAE prevent posterior collapse?*
- KL is constant → removes the KL-driven pressure toward collapse.
- ❌Guarantee that $z$ stays informative.
    - Codebook collapse (all $x$ → few codes) → near-constant $z_q$ → decoder learns to ignore it.
    - A decoder w/ another input path (e.g., autoregressive on $x_{<j}$) can still ignore $z_q$.

*Why 2 stages?*
- Stage 1: compress → ImageNet 128×128×3 images → 32×32 grid of indices, $K=512$ → ≈42.6× fewer bits.
- Stage 2: autoregressive prior spends capacity on global structure, ❌pixel noise → faster training & sampling.

*Relation to k-means?*
- Codebook term = [K-Means](../../ml/unsupervised.md#k-means) objective. EMA = its online centroid update.

*Where is it used now?*
- Image tokenizers for autoregressive generation (VQGAN = VQ-VAE + perceptual & adversarial losses).
- Neural audio codecs: residual VQ (stacked quantizers on the residual).
- [LDM](diffusion.md#ldm): VQ-regularized autoencoder = 1 of its 2 latent options (other = light KL).
```

&nbsp;
