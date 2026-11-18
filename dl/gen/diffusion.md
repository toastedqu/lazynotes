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
# Diffusion
Generative models that learn to invert a gradual noising process. Covers continuous (Gaussian) diffusion: theory, samplers, guidance, denoiser architectures, flow matching. Discrete text diffusion → [Diffusion LM](../../nlp/pre/obj.md#diffusion-lm).

Notations:
- $x_0$: Clean data sample (image or latent)
- $x_t$: Noisy sample at diffusion time $t$
- $t$: Diffusion time. $t=0$ = clean, $t=T$ = pure noise
- $T$: #diffusion steps
- $d$: Data dim
- $I\in\mathbb{R}^{d\times d}$: Identity matrix
- $\epsilon\sim\mathcal{N}(0,I)$: Standard Gaussian noise
- $\beta_t$: Noise variance added at step $t$
- $\alpha_t=1-\beta_t$: Per-step signal retention
- $\bar\alpha_t=\prod_{s=1}^{t}\alpha_s$: Cumulative signal retention
- $a_t=\sqrt{\bar\alpha_t}$: Signal scale
- $\sigma_t=\sqrt{1-\bar\alpha_t}$: Noise scale
- $q$: Forward (noising) process
- $p_t$: Marginal density of $x_t$
- $p_\theta$: Learned reverse (denoising) process
- $\epsilon_\theta(x_t,t)$: Noise-prediction network
- $c$: Condition (class label, text)
- $\varnothing$: Null condition

Overrides of global notation: $x$ = whole data vector (non-bold), ❌scalar. Subscript on $x$ = diffusion time, ❌sample/feature idx. $\sigma$ w/o subscript = VE noise level ($x=x_0+\sigma\epsilon$).

&nbsp;

## Diffusion Model
- **What**: Generative model that inverts a fixed data → noise process step by step. {cite:p}`sohl2015deep`
- **Why**: 1-shot generation (noise → image in 1 pass) is hard to learn.
    - Diffusion splits it into $T$ small denoising steps.
    - Each step = small Gaussian correction → simple regression loss.
- **How**:
    1. **Forward**: add Gaussian noise over $T$ steps → $x_T\approx\mathcal{N}(0,I)$. Fixed, ❌params.
    2. **Train**: NN predicts the noise in $x_t$ for random $t$ (MSE).
    3. **Sample**: $x_T\sim\mathcal{N}(0,I)$ → denoise step by step → $x_0$.
    - Physics analogy: ink spreads in water (forward). The learned reverse "un-mixes" the ink.

```{dropdown} Table: Generative Model Families
| Family | Train objective | Sampling cost | Likelihood | Typical failure |
|:--|:--|:--|:--|:--|
| GAN | Adversarial minimax | 1 pass | ❌ | Mode collapse, unstable training |
| VAE | ELBO | 1 pass | Lower bound | Blurry samples |
| Normalizing Flow | Exact NLL | 1 pass | ✅ Exact | Invertible arch → ⬇️expressivity |
| Autoregressive | Exact NLL (CE) | $d$ sequential passes | ✅ Exact | Slow for large $d$, fixed order |
| Diffusion | ELBO ≈ denoising MSE | $T$ passes (10–1000) | Bound, or exact via ODE | Slow sampling |
| Flow Matching | Velocity regression | ODE steps (1–50) | Exact via ODE | Same as diffusion |
```

```{attention} Q&A
:class: dropdown
*Pros?*
- Stable training ← plain regression, ❌minimax.
- Mode coverage ⬆️ ← likelihood-based (ELBO) → every training sample needs density → a dropped mode costs loss directly.
    - GAN: the ideal objective (JSD) also penalizes a dropped mode. In practice, weak or mode-seeking generator gradients → mode collapse.
- Test-time compute knob: #steps ⬆️ → quality ⬆️.
- Zero-shot editing (inpainting, img2img, guidance) ← sampling is an iterative process you can intervene in.

*Cons?*
- Slow sampling: $T$ sequential NN evals → [Fast Sampling](#fast-sampling).
- ❌Compact latent: $x_T$ has the same dim as $x_0$.
- Likelihood = bound, or an expensive ODE integral.

*Diffusion vs VAE?*
- Diffusion = hierarchical VAE w/ $T$ latents $x_{1:T}$, each the same dim as $x_0$.
- Encoder $q$ fixed, ❌learned → ❌posterior collapse. ELBO terms = closed-form Gaussian KLs.
- Decoder = 1 shared net $\epsilon_\theta(\cdot,t)$ for all levels.

*Why many small steps, ❌1 big step?*
- Small $\beta_t$ → true reverse step $q(x_{t-1}|x_t)$ ≈ Gaussian → Gaussian $p_\theta$ is a valid model.
- Large step → true reverse = complex multimodal distribution → Gaussian fails.

*What does the denoiser learn at each noise level?*
- High noise: global layout (low frequencies). Low noise: texture (high frequencies).
- ← Natural-image power spectrum decays w/ frequency. White noise is flat. → Noise drowns high frequencies first.
- → Sampling generates coarse → fine.

*Would a perfect model only memorize the training set?*
- Yes. The exact optimal denoiser for a finite dataset = softmax-weighted average of training points.
    - $\mathbb{E}[x_0|x_t]=\sum_nw_nx^{(n)}$, $w_n\propto\exp\left(-\|x_t-a_tx^{(n)}\|^2/2\sigma_t^2\right)$.
    - $t\to0$ → nearest training point → sampling reproduces training data.
- Generalization ← the NN does NOT reach this empirical optimum. Inductive bias & finite capacity → smooth score.
```

&nbsp;

## Forward Process
- **What**: Fixed Markov chain that adds Gaussian noise: $x_0\to x_1\to\dots\to x_T\approx\mathcal{N}(0,I)$.
- **Why**: Need a path from data to a simple prior w/ tractable intermediate distributions.
    - Gives unlimited (noisy input, clean target) training pairs for free.
- **How**:
    1. Each step: shrink the signal by $\sqrt{\alpha_t}$ → add noise of variance $\beta_t$.
    2. Shrink + add keeps unit variance (**variance preserving**) → $x_t$ never blows up.
    3. Gaussians compose → jump to any $t$ in 1 shot.

```{note} Math
:class: dropdown
Process:

$$
q(x_t|x_{t-1})=\mathcal{N}\left(x_t;\sqrt{\alpha_t}\,x_{t-1},\beta_tI\right)
$$

Closed-form marginal:

$$
q(x_t|x_0)=\mathcal{N}\left(x_t;a_tx_0,\sigma_t^2I\right)\quad\Leftrightarrow\quad x_t=a_tx_0+\sigma_t\epsilon
$$

- $a_t^2+\sigma_t^2=1$: variance preserving.
- $\bar\alpha_T\approx0$ → $q(x_T|x_0)\approx\mathcal{N}(0,I)$, independent of $x_0$.
- Marginal $p_t(x_t)=\int q(x_t|x_0)\,p_\text{data}(x_0)\,dx_0$ = data convolved w/ a Gaussian → smooth, but ❌Gaussian for $t<T$.
```

```{tip} Derivation
:class: dropdown
*Why does $x_t=a_tx_0+\sigma_t\epsilon$ hold?*
1. Induction. Assume $x_{t-1}=\sqrt{\bar\alpha_{t-1}}\,x_0+\sqrt{1-\bar\alpha_{t-1}}\,\epsilon'$.
2. One step:

    $$
    x_t=\sqrt{\alpha_t}x_{t-1}+\sqrt{\beta_t}\epsilon''=\sqrt{\bar\alpha_t}x_0+\sqrt{\alpha_t(1-\bar\alpha_{t-1})}\,\epsilon'+\sqrt{1-\alpha_t}\,\epsilon''
    $$

3. Independent zero-mean Gaussians → variances add: $\alpha_t-\bar\alpha_t+1-\alpha_t=1-\bar\alpha_t$.
4. → $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\,\epsilon$.

*Why "variance preserving"?*
- If $\text{Var}(x_{t-1})=I$: $\text{Var}(x_t)=\alpha_tI+\beta_tI=I$.
- Assumes data scaled to ~unit scale (e.g., pixels in $[-1,1]$).
```

````{important} Code
:class: dropdown
```python
import torch

class ForwardProcess:
    def __init__(self, betas):
        self.betas = betas                                ## (T,); index k <-> step k+1
        self.alpha_bar = torch.cumprod(1 - betas, dim=0)  ## (T,)

    def q_sample(self, x0, t, eps):
        ## jump straight to step t: x_t = a_t x0 + sigma_t eps (no loop over steps)
        ab = self.alpha_bar[t].view(-1, *[1] * (x0.dim() - 1))   ## broadcast over (C,H,W)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * eps

## Example
fp = ForwardProcess(torch.linspace(1e-4, 0.02, 1000))   ## DDPM linear schedule
x0 = torch.randn(2, 3, 8, 8)
xt = fp.q_sample(x0, torch.tensor([0, 999]), torch.randn_like(x0))
print(xt.shape, fp.alpha_bar[-1].item())   ## torch.Size([2, 3, 8, 8]) ~4e-5 -> x_T ≈ pure noise
```
````

```{attention} Q&A
:class: dropdown
*Why scale by $\sqrt{\alpha_t}$, ❌only add noise?*
- Only add noise → variance grows w/o bound (variance exploding, VE; used by [NCSN](#ncsn)) → needs a huge max noise level.
- Scale + add → variance fixed at 1 → net input stays in 1 range at every $t$.
- Both work. VP & VE = 2 instances of 1 SDE family → [Score SDE](#score-sde).

*Is the forward process learned?*
- ❌ in DDPM. Fixed → no encoder to train, ❌posterior collapse.
- The schedule can be learned → [Noise Schedule](#noise-schedule).
```

&nbsp;

### Noise Schedule
- **What**: The choice of $\{\beta_t\}$ ⇔ how fast the signal-to-noise ratio (SNR) decays over $t$.
- **Why**: The schedule decides where training capacity & sampling steps go.
    - Too fast → many steps on near-pure noise → wasted.
    - Too slow → $x_T\neq$ pure noise → train/test mismatch at the start of sampling.
- **How**:
    - **Linear** (DDPM): $\beta_t$ linear from $10^{-4}$ to $0.02$, $T=1000$.
    - **Cosine**: $\bar\alpha_t$ follows a $\cos^2$ curve → SNR decays smoothly, ❌long near-zero tail. {cite:p}`nichol2021improved`
    - **Shifted**: move the schedule toward more noise for high resolution.

```{note} Math
:class: dropdown
Definition:

$$
\text{SNR}(t)=\frac{a_t^2}{\sigma_t^2}=\frac{\bar\alpha_t}{1-\bar\alpha_t}
$$

- Monotone ⬇️ in $t$.

Linear:

$$
\beta_t=\beta_1+\frac{t-1}{T-1}(\beta_T-\beta_1),\quad\beta_1=10^{-4},\ \beta_T=0.02
$$

Cosine:

$$
\bar\alpha_t=\frac{f(t)}{f(0)},\quad f(t)=\cos^2\left(\frac{t/T+s}{1+s}\cdot\frac{\pi}{2}\right)
$$

- $s=0.008$: Small offset → $\beta_t$ not too small near $t=0$.
- $\beta_t=1-\bar\alpha_t/\bar\alpha_{t-1}$, clipped to $\leq0.999$ → no singularity near $t=T$.
```

````{important} Code
:class: dropdown
```python
import math
import torch

class CosineSchedule:
    def __init__(self, T, s=0.008):
        t = torch.arange(T + 1) / T
        f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = f / f[0]                                    ## (T+1,), alpha_bar[0] = 1
        ## per-step beta from ratio of cumulative products; clip avoids beta = 1 at t = T
        self.betas = (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(max=0.999)   ## (T,)

## Example
b = CosineSchedule(1000).betas
print(b.shape, b[0].item(), b[-1].item())   ## torch.Size([1000]) ~4e-5 0.999
```
````

```{attention} Q&A
:class: dropdown
*Why did cosine beat linear?*
- Linear: $\bar\alpha_t$ hits ~0 well before $t=T$ → the last steps ≈ pure noise → little information.
- On 32×32 / 64×64 images, skipping up to 20% of the linear schedule's reverse steps barely hurts FID → those steps are wasted.
- Cosine: near-linear drop of $\bar\alpha_t$ in the middle → steps spread over useful SNRs.

*Why does the best schedule depend on resolution?*
- Same per-pixel SNR → higher resolution keeps more information ← neighboring pixels are redundant.
- Average a $k\times k$ patch: signal stays, iid noise variance ÷$k^2$ → effective SNR ×$k^2$.
- → High resolution needs a schedule shifted toward more noise {cite:p}`hoogeboom2023simple`.

*What is zero terminal SNR? Why does it matter?*
- $\bar\alpha_T=0$ exactly → $x_T$ = pure noise.
- Stable Diffusion's schedule: $\sqrt{\bar\alpha_T}\approx0.068$ → training never sees pure noise. Sampling starts from pure noise → mismatch.
- Symptom: SD generates only medium-brightness images. ❌Very bright, ❌very dark.
- Fix: rescale the schedule to $\bar\alpha_T=0$ + use [$v$-prediction](#prediction-target) {cite:p}`lin2024common`.
    - ε-prediction breaks at SNR = 0: $x_T=\epsilon$ → target = input → trivial, & $\hat x_0$ divides by $a_T=0$.

*Does the schedule change the likelihood?*
- Continuous-time ELBO depends only on SNR at the 2 endpoints, ❌the path between {cite:p}`kingma2021variational`.
- The schedule still changes the Monte-Carlo variance of the loss & the weighting of $L_\text{simple}$ → sample quality.
```

&nbsp;

## Reverse Process
- **What**: Learned Markov chain $p_\theta(x_{t-1}|x_t)$ that runs the forward process backward.
- **Why**: True reverse step $q(x_{t-1}|x_t)$ needs $p_\text{data}$ → intractable.
    - Conditioned on $x_0$, the reverse step $q(x_{t-1}|x_t,x_0)$ is a closed-form Gaussian → usable as a training target.
- **How**:
    1. Model each step as a Gaussian w/ learned mean & fixed variance.
    2. Train by maximizing the ELBO on $\log p_\theta(x_0)$.
    3. ELBO = sum of per-step KLs between Gaussians → match $\mu_\theta$ to the posterior mean $\tilde\mu_t$.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\mu_\theta(x_t,t)$: Learned reverse-step mean.
- Hyperparams:
    - $\tilde\sigma_t^2$: Reverse-step variance. Fixed to $\beta_t$ or $\tilde\beta_t$.

Model:

$$
p_\theta(x_{0:T})=p(x_T)\prod_{t=1}^{T}p_\theta(x_{t-1}|x_t),\quad p_\theta(x_{t-1}|x_t)=\mathcal{N}\left(\mu_\theta(x_t,t),\tilde\sigma_t^2I\right),\quad p(x_T)=\mathcal{N}(0,I)
$$

Posterior (training target):

$$
q(x_{t-1}|x_t,x_0)=\mathcal{N}\left(x_{t-1};\tilde\mu_t(x_t,x_0),\tilde\beta_tI\right)
$$

$$
\tilde\mu_t=\frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}x_0+\frac{\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t,\qquad\tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t
$$

- $\tilde\mu_t$: Posterior mean. Weighted mix of $x_0$ & $x_t$.
- $\tilde\beta_t$: Posterior variance.

Objective:

$$
-\log p_\theta(x_0)\leq\mathbb{E}_q\Big[\underbrace{D_\text{KL}\left(q(x_T|x_0)\,\|\,p(x_T)\right)}_{L_T}+\sum_{t=2}^{T}\underbrace{D_\text{KL}\left(q(x_{t-1}|x_t,x_0)\,\|\,p_\theta(x_{t-1}|x_t)\right)}_{L_{t-1}}\underbrace{-\log p_\theta(x_0|x_1)}_{L_0}\Big]
$$

- $L_T$: ❌params → constant.
- $L_{t-1}=\frac{1}{2\tilde\sigma_t^2}\|\tilde\mu_t-\mu_\theta(x_t,t)\|^2+C$ ← KL of 2 Gaussians w/ equal fixed covariance.
```

```{tip} Derivation
:class: dropdown
*Why is the posterior Gaussian?*
1. Bayes: $q(x_{t-1}|x_t,x_0)\propto q(x_t|x_{t-1})\,q(x_{t-1}|x_0)$. Both Gaussian in $x_{t-1}$.
2. Precision: $\frac{\alpha_t}{\beta_t}+\frac{1}{1-\bar\alpha_{t-1}}=\frac{\alpha_t(1-\bar\alpha_{t-1})+\beta_t}{\beta_t(1-\bar\alpha_{t-1})}=\frac{1-\bar\alpha_t}{\beta_t(1-\bar\alpha_{t-1})}$ → variance $\tilde\beta_t$.
3. Mean = variance × (precision-weighted sum): $\tilde\beta_t\left(\frac{\sqrt{\alpha_t}}{\beta_t}x_t+\frac{\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_{t-1}}x_0\right)=\tilde\mu_t$.

*Why is the ELBO a sum of per-step KLs?*
1. Jensen: $-\log p_\theta(x_0)\leq\mathbb{E}_q\left[-\log\frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}\right]$.
2. Markov: $q(x_{1:T}|x_0)=\prod_{t=1}^{T}q(x_t|x_{t-1})$.
3. Bayes, for $t\geq2$: $q(x_t|x_{t-1})=q(x_t|x_{t-1},x_0)=\frac{q(x_{t-1}|x_t,x_0)\,q(x_t|x_0)}{q(x_{t-1}|x_0)}$.
4. The ratios $q(x_t|x_0)/q(x_{t-1}|x_0)$ telescope → $q(x_T|x_0)/q(x_1|x_0)$.
5. Regroup → $L_T+\sum_{t\geq2}L_{t-1}+L_0$.
6. Each $L_{t-1}$ compares 2 Gaussians → closed form → low-variance loss. See [VI](../../math/prob/sampling.md#vi) & [KL Divergence](../../math/info.md#kl-divergence).
```

```{attention} Q&A
:class: dropdown
*Why a Gaussian $p_\theta$?*
- For small $\beta_t$, the true reverse step has the same functional form as the forward step (classic diffusion-process result) → Gaussian.
- Breaks for large steps → few-step samplers need other tools ([Fast Sampling](#fast-sampling)).

*Why fix the variance?*
- $\tilde\sigma_t^2=\beta_t$ & $\tilde\sigma_t^2=\tilde\beta_t$ = optimal for $x_0\sim\mathcal{N}(0,I)$ & for deterministic $x_0$. Real data lies between.
- DDPM: both give similar samples. Learning it naively was unstable.
- Learned interpolation $\exp(v\log\beta_t+(1-v)\log\tilde\beta_t)$ → better likelihood & ~10× fewer sampling steps {cite:p}`nichol2021improved`.
```

&nbsp;

### DDPM
- **Name**: Denoising Diffusion Probabilistic Models {cite:p}`ho2020denoising`
- **What**: Diffusion model trained to predict the added noise $\epsilon$ w/ unweighted MSE.
- **Why**:
    - *Why do we need it?* Earlier diffusion models worked, but samples lagged GANs.
    - *Why does it work?*
        1. Write $\mu_\theta$ via predicted noise → each ELBO term = weighted noise MSE.
        2. Drop the weights → $L_\text{simple}$ → capacity moves to harder high-noise steps → better samples.
        3. Noise prediction = [denoising score matching](#dsm) → links diffusion to score-based models.
- **How**:
    - **Training**: sample $x_0$, $t\sim\mathcal{U}\{1,\dots,T\}$, $\epsilon$ → build $x_t$ → MSE between $\epsilon_\theta(x_t,t)$ & $\epsilon$.
    - **Sampling**: $x_T\sim\mathcal{N}(0,I)$ → for $t=T,\dots,1$: remove predicted noise → add fresh noise (skip at $t=1$).

```{note} Math
:class: dropdown
Model:

$$
\mu_\theta(x_t,t)=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sigma_t}\epsilon_\theta(x_t,t)\right)
$$

Training:

$$
L_\text{simple}=\mathbb{E}_{t,x_0,\epsilon}\left[\left\|\epsilon-\epsilon_\theta(a_tx_0+\sigma_t\epsilon,\,t)\right\|^2\right],\quad t\sim\mathcal{U}\{1,\dots,T\}
$$

- Exact ELBO term: $L_{t-1}=\frac{\beta_t^2}{2\tilde\sigma_t^2\alpha_t\sigma_t^2}\|\epsilon-\epsilon_\theta\|^2+C$. $L_\text{simple}$ sets this weight to 1.

Inference:

$$
x_{t-1}=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sigma_t}\epsilon_\theta(x_t,t)\right)+\tilde\sigma_tz,\quad z\sim\mathcal{N}(0,I)\text{ if }t>1,\text{ else }z=0
$$

- $\tilde\sigma_t^2$: Reverse-step variance ($\beta_t$ or $\tilde\beta_t$).
```

```{tip} Derivation
:class: dropdown
*Why does $\mu_\theta$ take this form?*
1. Invert the forward process: $x_0=(x_t-\sigma_t\epsilon)/a_t$.
2. Plug into $\tilde\mu_t$. Use $\sqrt{\bar\alpha_{t-1}}/a_t=1/\sqrt{\alpha_t}$.
3. $x_t$ coefficient: $\frac{\beta_t+\alpha_t(1-\bar\alpha_{t-1})}{\sqrt{\alpha_t}(1-\bar\alpha_t)}=\frac{1}{\sqrt{\alpha_t}}$ ← $\beta_t+\alpha_t-\bar\alpha_t=1-\bar\alpha_t$.
4. $\epsilon$ coefficient: $-\frac{\beta_t\sigma_t}{\sqrt{\alpha_t}\,\sigma_t^2}=-\frac{\beta_t}{\sqrt{\alpha_t}\,\sigma_t}$.
5. → $\tilde\mu_t=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sigma_t}\epsilon\right)$. Only $\epsilon$ is unknown → predict it.
6. $\tilde\mu_t-\mu_\theta=\frac{\beta_t}{\sqrt{\alpha_t}\,\sigma_t}(\epsilon_\theta-\epsilon)$ → square, divide by $2\tilde\sigma_t^2$ → ELBO weight above.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

## Reuses ForwardProcess from Forward Process above
class DDPM(nn.Module):
    def __init__(self, eps_model, betas):
        super().__init__()
        self.eps_model, self.fp, self.T = eps_model, ForwardProcess(betas), len(betas)

    def loss(self, x0):
        t = torch.randint(0, self.T, (x0.shape[0],))       ## uniform step per sample
        eps = torch.randn_like(x0)
        xt = self.fp.q_sample(x0, t, eps)
        return F.mse_loss(self.eps_model(xt, t), eps)      ## L_simple: unweighted

    @torch.no_grad()
    def sample(self, shape):
        x = torch.randn(shape)                              ## x_T ~ N(0, I)
        for t in reversed(range(self.T)):
            beta, ab = self.fp.betas[t], self.fp.alpha_bar[t]
            eps = self.eps_model(x, torch.full((shape[0],), t))
            x = (x - beta / (1 - ab).sqrt() * eps) / (1 - beta).sqrt()   ## posterior mean
            if t > 0:
                x = x + beta.sqrt() * torch.randn_like(x)  ## sigma_t^2 = beta_t; no noise at last step
        return x

## Example: stand-in eps model that ignores t
net = nn.Conv2d(3, 3, 3, padding=1)
ddpm = DDPM(lambda x, t: net(x), torch.linspace(1e-4, 0.02, 50))
print(ddpm.loss(torch.randn(2, 3, 8, 8)).item() > 0, ddpm.sample((2, 3, 8, 8)).shape)
## True torch.Size([2, 3, 8, 8])
```
````

```{attention} Q&A
:class: dropdown
*Why predict $\epsilon$, ❌$\tilde\mu_t$?*
- DDPM ablation: $\epsilon$-prediction + $L_\text{simple}$ = best FID. $\tilde\mu_t$-prediction worked only w/ the true ELBO.
- $\epsilon$ target has unit variance at every $t$ → well-scaled regression.

*Why drop the ELBO weights?*
- The weight is large at small $t$ (easy, low-noise steps).
- $L_\text{simple}$ down-weights them → capacity on harder high-noise steps → sample quality ⬆️, log-likelihood ⬇️.

*Key numbers?*
- $T=1000$, linear $\beta$ from $10^{-4}$ to $0.02$.
- Unconditional CIFAR-10: FID 3.17, IS 9.46.

*Cons?*
- 1000 sequential NN evals per sample → [DDIM](#ddim) & later samplers.
- Fixed variance → weak log-likelihood → fixed by learned variance.

*How does DDPM relate to score matching?*
- $\epsilon_\theta(x_t,t)\approx-\sigma_t\nabla_{x_t}\log p_t(x_t)$.
- → $L_\text{simple}$ = DSM w/ weight $\sigma_t^2$. DDPM sampling ≈ annealed Langevin dynamics.
```

&nbsp;

### Prediction Target
- **What**: The quantity the net outputs: $\epsilon$, $x_0$, $v$, or score. All interconvertible given $x_t$.
- **Why**: Same information, different numerical conditioning per noise level.
    - $\epsilon$-pred: $\hat x_0=(x_t-\sigma_t\hat\epsilon)/a_t$ → error ×$1/a_t$ → blows up at high noise ($a_t\to0$).
    - $x_0$-pred: $\hat\epsilon=(x_t-a_t\hat x_0)/\sigma_t$ → error ×$1/\sigma_t$ → blows up at low noise ($\sigma_t\to0$).
    - $v$-pred: well-scaled at both ends. Introduced w/ [Progressive Distillation](#progressive-distillation).
- **How**: Train on target $\in\{\epsilon,x_0,v\}$ w/ MSE → convert to what the sampler needs.

```{note} Math
:class: dropdown
Definition:

$$
v=a_t\epsilon-\sigma_tx_0
$$

- $t\to0$: $v\to\epsilon$. SNR $\to0$: $v\to-x_0$.

Conversions (given $x_t=a_tx_0+\sigma_t\epsilon$):

$$
\hat x_0=\frac{x_t-\sigma_t\hat\epsilon}{a_t}=a_tx_t-\sigma_t\hat v,\qquad\hat\epsilon=\frac{x_t-a_t\hat x_0}{\sigma_t}=\sigma_tx_t+a_t\hat v,\qquad\nabla_{x_t}\log p_t(x_t)\approx-\frac{\hat\epsilon}{\sigma_t}
$$

Loss equivalence (same $x_t$):

$$
\|\epsilon-\hat\epsilon\|^2=\text{SNR}(t)\,\|x_0-\hat x_0\|^2,\qquad\|v-\hat v\|^2=\left(1+\text{SNR}(t)\right)\|x_0-\hat x_0\|^2
$$

- → Choosing a target = choosing a loss weight over $t$ (in $x_0$ units).
```

```{tip} Derivation
:class: dropdown
*Where do the SNR weights come from?*
1. Fix $x_t$: $a_tx_0+\sigma_t\epsilon=a_t\hat x_0+\sigma_t\hat\epsilon$ → $\sigma_t(\epsilon-\hat\epsilon)=-a_t(x_0-\hat x_0)$.
2. Square → $\|\epsilon-\hat\epsilon\|^2=\frac{a_t^2}{\sigma_t^2}\|x_0-\hat x_0\|^2$.
3. $v-\hat v=a_t(\epsilon-\hat\epsilon)-\sigma_t(x_0-\hat x_0)=-\frac{a_t^2+\sigma_t^2}{\sigma_t}(x_0-\hat x_0)=-\frac{1}{\sigma_t}(x_0-\hat x_0)$.
4. $\frac{1}{\sigma_t^2}=\frac{a_t^2+\sigma_t^2}{\sigma_t^2}=1+\text{SNR}(t)$.
```

````{important} Code
:class: dropdown
```python
import torch

class PredConvert:
    ## a, s: signal & noise scales a_t, sigma_t (broadcastable), a^2 + s^2 = 1
    def __init__(self, a, s):
        self.a, self.s = a, s
    def v_target(self, x0, eps):  return self.a * eps - self.s * x0
    def x0_from_eps(self, xt, eps): return (xt - self.s * eps) / self.a   ## unstable as a -> 0
    def x0_from_v(self, xt, v):   return self.a * xt - self.s * v          ## stable at both ends
    def eps_from_v(self, xt, v):  return self.s * xt + self.a * v

## Example: round trip recovers x0 & eps exactly
a, s = torch.tensor(0.6), torch.tensor(0.8)
x0, eps = torch.randn(4), torch.randn(4)
xt = a * x0 + s * eps
pc = PredConvert(a, s)
v = pc.v_target(x0, eps)
print(torch.allclose(pc.x0_from_v(xt, v), x0), torch.allclose(pc.eps_from_v(xt, v), eps))  ## True True
```
````

```{attention} Q&A
:class: dropdown
*Which target is best?*
- No universal winner. It interacts w/ the loss weight & schedule.
- $\epsilon$: DDPM, SD 1.x. $v$: SD 2.x 768-px model, progressive distillation. Velocity ([Flow Matching](#flow-matching)): SD3.

*Why does $\epsilon$-pred fail at zero terminal SNR?*
- $a_T=0$ → $x_T=\epsilon$ → target = input → trivial; $\hat x_0$ = division by 0.
- $v$-pred → $v_T=-x_0$ → the net must predict data → meaningful.

*How does this relate to EDM?*
- [EDM](#edm) preconditioning = $\sigma$-dependent mix of skip & net output.
- Under VP scaling w/ $\sigma_\text{data}=1$, EDM's net predicts $-v$.
```

&nbsp;

## Score-Based View
- **What**: Generate by following the **score** $\nabla_x\log p_t(x)$ of noise-smoothed data from noise to data.

&nbsp;

### Score Matching
- **What**: Fit $s_\theta(x)\approx\nabla_x\log p(x)$ w/o knowing $p$. {cite:p}`hyvarinen2005estimation`
- **Why**: Unnormalized model $p_\theta=\tilde p_\theta/Z_\theta$ → $Z_\theta$ intractable → MLE intractable.
    - Score removes $Z$: $\nabla_x\log p_\theta=\nabla_x\log\tilde p_\theta$ ← $Z_\theta$ is constant in $x$.
    - Score suffices to sample (Langevin) → never need $p$ itself.
- **How**: Minimize the Fisher divergence $\mathbb{E}_p\|s_\theta-\nabla\log p\|^2$. Integration by parts removes the unknown $\nabla\log p$.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $s_\theta(x)$: Score model.

Objective:

$$
J(\theta)=\mathbb{E}_{p}\left[\tfrac12\|s_\theta(x)-\nabla_x\log p(x)\|^2\right]=\mathbb{E}_{p}\left[\text{tr}\left(\nabla_xs_\theta(x)\right)+\tfrac12\|s_\theta(x)\|^2\right]+C
$$

- $\nabla_xs_\theta(x)\in\mathbb{R}^{d\times d}$: Jacobian of the score model. Its trace needs $d$ backward passes.
```

```{tip} Derivation
:class: dropdown
*How does the unknown $\nabla\log p$ disappear?*
1. Expand the square: $\tfrac12\|s_\theta\|^2-\langle s_\theta,\nabla\log p\rangle+\tfrac12\|\nabla\log p\|^2$. Last term = $C$.
2. Cross term: $\mathbb{E}_p\langle s_\theta,\nabla\log p\rangle=\int\langle s_\theta,\nabla p\rangle dx$.
3. Integrate by parts (assume $p\,s_\theta\to0$ at infinity): $=-\int p\,\nabla\cdot s_\theta\,dx=-\mathbb{E}_p[\text{tr}(\nabla_xs_\theta)]$.
4. → $J=\mathbb{E}_p[\text{tr}(\nabla_xs_\theta)+\tfrac12\|s_\theta\|^2]+C$.
```

```{attention} Q&A
:class: dropdown
*Cons?*
- $\text{tr}(\nabla_xs_\theta)$ → $O(d)$ backward passes → ❌scales to images → [DSM](#dsm).
- Score is inaccurate in low-density regions ← few training samples there.

*What can the score NOT tell you?*
- Relative weights of well-separated modes. Score is local: inside each mode it is the same for any mixture weight.
- → Plain Langevin samples the wrong mode proportions → fixed by large-noise levels that connect the modes ([NCSN](#ncsn)).
```

&nbsp;

#### DSM
- **Name**: Denoising Score Matching {cite:p}`vincent2011connection`
- **What**: Score matching on noise-perturbed data. Target = score of the known noise kernel.
- **Why**: Plain score matching needs $\text{tr}(\nabla_xs_\theta)$.
    - The conditional score of a Gaussian kernel is closed form → plain regression.
- **How**:
    1. Perturb: $\tilde x=x+\sigma\epsilon$.
    2. Regress $s_\theta(\tilde x)$ onto $-\epsilon/\sigma$.
    3. Optimum = score of the perturbed marginal $p_\sigma$, ❌clean $p$.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $p_\sigma=p*\mathcal{N}(0,\sigma^2I)$: Data density smoothed by noise level $\sigma$.

Objective:

$$
J_\text{DSM}(\theta)=\mathbb{E}_{x\sim p,\,\epsilon}\left[\tfrac12\left\|s_\theta(x+\sigma\epsilon)+\frac{\epsilon}{\sigma}\right\|^2\right]
$$

- $\nabla_{\tilde x}\log q_\sigma(\tilde x|x)=-(\tilde x-x)/\sigma^2=-\epsilon/\sigma$: Score of the Gaussian kernel.
- Optimum: $s^*(\tilde x)=\nabla_{\tilde x}\log p_\sigma(\tilde x)$.

Tweedie's formula:

$$
\mathbb{E}[x\,|\,\tilde x]=\tilde x+\sigma^2\nabla_{\tilde x}\log p_\sigma(\tilde x)
$$

- VP form: $\mathbb{E}[x_0|x_t]=\frac{x_t+\sigma_t^2\nabla\log p_t(x_t)}{a_t}$.
- → Score ⇔ optimal denoiser. Learning one = learning the other.
- DDPM link: $\epsilon_\theta=-\sigma_ts_\theta$ → $\|\epsilon-\epsilon_\theta\|^2=\sigma_t^2\|s_\theta+\epsilon/\sigma_t\|^2$.
```

```{tip} Derivation
:class: dropdown
*Why is the DSM optimum the marginal score?*
1. Only the cross term depends on the target: $\mathbb{E}_{x,\tilde x}\langle s_\theta(\tilde x),\nabla_{\tilde x}\log q_\sigma(\tilde x|x)\rangle$.
2. $\nabla p_\sigma(\tilde x)=\int p(x)\nabla_{\tilde x}q_\sigma(\tilde x|x)\,dx$ → $\mathbb{E}_{x|\tilde x}[\nabla_{\tilde x}\log q_\sigma(\tilde x|x)]=\nabla_{\tilde x}\log p_\sigma(\tilde x)$.
3. → $J_\text{DSM}=$ score matching on $p_\sigma$ $+\,C$. Same minimizer.

*Why does Tweedie's formula hold?*
1. $\nabla_{\tilde x}q_\sigma(\tilde x|x)=q_\sigma(\tilde x|x)\frac{x-\tilde x}{\sigma^2}$.
2. Integrate against $p(x)$: $\nabla p_\sigma(\tilde x)=\frac{1}{\sigma^2}\int p(x)q_\sigma(\tilde x|x)(x-\tilde x)\,dx$.
3. Divide by $p_\sigma(\tilde x)$: $\sigma^2\nabla\log p_\sigma(\tilde x)=\mathbb{E}[x|\tilde x]-\tilde x$.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class DSMLoss(nn.Module):
    def __init__(self, score_model, sigma):
        super().__init__()
        self.s, self.sigma = score_model, sigma

    def forward(self, x):
        eps = torch.randn_like(x)
        x_tilde = x + self.sigma * eps
        ## target = score of the Gaussian kernel = -eps/sigma -> no Jacobian trace
        return 0.5 * (self.s(x_tilde) + eps / self.sigma).pow(2).sum(-1).mean()

## Example: data ~ N(0, I) -> smoothed score is exactly -x / (1 + sigma^2)
sigma = 0.5
loss = DSMLoss(lambda x: -x / (1 + sigma ** 2), sigma)
print(loss(torch.randn(100000, 2)))   ## ≈ 3.2 = d/2 * (1/sigma^2 - 1/(1+sigma^2)): floor > 0 even at optimum
```
````

```{attention} Q&A
:class: dropdown
*Why can't the DSM loss reach 0?*
- Target $-\epsilon/\sigma$ is random given $\tilde x$ → floor = its conditional variance.
- → Raw loss value ❌measures fit quality. Compare against the floor, or compare models.

*Small vs large $\sigma$?*
- Small $\sigma$ → $p_\sigma\approx p$ (low bias). But target variance ∝ $1/\sigma^2$ & ❌coverage off the data manifold.
- Large $\sigma$ → stable target, covers all space. But $p_\sigma$ = blurred data (high bias).
- → Use many levels → [NCSN](#ncsn), diffusion.
```

&nbsp;

### NCSN
- **Name**: Noise Conditional Score Network {cite:p}`song2019generative`
- **What**: 1 score net $s_\theta(x,\sigma)$ for many noise levels, sampled by annealed Langevin dynamics.
- **Why**: Single-level score models fail.
    - Data on a low-dim manifold → score undefined off it.
    - Low-density regions → inaccurate score → Langevin from random init gets lost.
    - Large noise: score good everywhere, data blurred. Small noise: accurate, but only near data. → Use both, anneal.
- **How**:
    1. Geometric noise levels $\sigma_1>\sigma_2>\dots>\sigma_L$.
    2. Train $s_\theta(x,\sigma_i)$ w/ DSM at all levels, weight $\sigma_i^2$.
    3. Sample: Langevin at $\sigma_1$ → output initializes $\sigma_2$ → … → $\sigma_L$.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $L$: #noise levels.
    - $\delta$: Base Langevin step size.
- Misc:
    - $z\sim\mathcal{N}(0,I)$: Fresh Langevin noise.

Objective:

$$
\mathcal{L}(\theta)=\frac{1}{L}\sum_{i=1}^{L}\sigma_i^2\,\mathbb{E}_{x,\epsilon}\left[\tfrac12\left\|s_\theta(x+\sigma_i\epsilon,\sigma_i)+\frac{\epsilon}{\sigma_i}\right\|^2\right]
$$

- $\sigma_i^2$ weight: optimal $\|s\|\propto1/\sigma$ → every level contributes ~equally.

Process (Langevin dynamics at level $i$, repeat $K$ times):

$$
x\leftarrow x+\frac{\delta_i}{2}s_\theta(x,\sigma_i)+\sqrt{\delta_i}\,z,\quad\delta_i=\delta\cdot\sigma_i^2/\sigma_L^2
$$

- $\delta_i$: Step size ∝ $\sigma_i^2$.
- Langevin samples $p$ exactly as $\delta\to0$ & #steps $\to\infty$.
```

````{important} Code
:class: dropdown
```python
import torch

class AnnealedLangevin:
    def __init__(self, score_model, sigmas, n_steps=100, delta=2e-5):
        self.s, self.sigmas, self.K, self.delta = score_model, sigmas, n_steps, delta

    @torch.no_grad()
    def sample(self, x):
        for sigma in self.sigmas:                                   ## large -> small noise
            step = self.delta * (sigma / self.sigmas[-1]) ** 2      ## step ∝ sigma^2
            for _ in range(self.K):
                x = x + 0.5 * step * self.s(x, sigma) + step.sqrt() * torch.randn_like(x)
        return x

## Example: data ~ N(3, I) -> smoothed score is analytic
score = lambda x, sigma: -(x - 3) / (1 + sigma ** 2)
sigmas = torch.logspace(1, -2, 10)                                  ## 10 -> 0.01, geometric
x = AnnealedLangevin(score, sigmas).sample(10 * torch.randn(5000, 2))
print(x.mean(0), x.std(0))   ## ≈ [3, 3], ≈ [1, 1]
```
````

```{attention} Q&A
:class: dropdown
*Why anneal?*
- Large $\sigma$ → modes overlap → Langevin moves between modes → correct mode proportions.
- Small $\sigma$ → refine detail near the data manifold.

*NCSN vs DDPM?*
- Same idea, different noising: NCSN adds noise only (VE). DDPM scales & adds (VP).
- Unified by [Score SDE](#score-sde).
```

&nbsp;

### Score SDE
- **Name**: Score-Based Generative Modeling through Stochastic Differential Equations {cite:p}`song2021score`
- **What**: Continuous-time diffusion: forward SDE data → noise, reverse-time SDE driven by the score.
- **Why**: Unify DDPM & NCSN. Continuous $t$ → any solver, any #steps, exact likelihood.
- **How**:
    1. Pick a forward SDE (VP ⇔ DDPM, VE ⇔ NCSN).
    2. Train $s_\theta(x,t)$ w/ DSM at continuous $t$.
    3. Sample: solve the reverse SDE (stochastic) or the **probability-flow ODE** (deterministic).

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $t\in[0,1]$: Continuous time (override of discrete $t$).
    - $f(x,t)$: Drift.
    - $g(t)$: Diffusion coefficient.
    - $w$: Wiener process.
    - $\bar w$: Reverse-time Wiener process.

Forward SDE:

$$
dx=f(x,t)\,dt+g(t)\,dw
$$

Reverse SDE:

$$
dx=\left[f(x,t)-g(t)^2\nabla_x\log p_t(x)\right]dt+g(t)\,d\bar w,\quad dt<0
$$

Probability-flow ODE:

$$
\frac{dx}{dt}=f(x,t)-\frac12g(t)^2\nabla_x\log p_t(x)
$$

- Same marginals $p_t$ as the SDE. Deterministic.
- Plug in $s_\theta(x,t)\approx\nabla_x\log p_t(x)$ to sample.

Exact likelihood (via the ODE drift $\tilde f$):

$$
\log p_0(x_0)=\log p_1(x_1)+\int_0^1\nabla\cdot\tilde f(x_t,t)\,dt
$$

- $\tilde f$: Right side of the probability-flow ODE. Divergence via Hutchinson's trace estimator.
```

```{dropdown} Table: SDE Instances
| SDE | $f(x,t)$ | $g(t)$ | Discrete counterpart |
|:--|:--|:--|:--|
| VE | $0$ | $\sqrt{d[\sigma^2(t)]/dt}$ | NCSN |
| VP | $-\frac12\beta(t)\,x$ | $\sqrt{\beta(t)}$ | DDPM |

- $\sigma(t)$: VE noise level at time $t$.
- $\beta(t)$: Continuous limit of $T\beta_t$.
```

```{tip} Derivation
:class: dropdown
*Why does the probability-flow ODE have the same marginals as the SDE?*
1. Fokker–Planck for the forward SDE: $\partial_tp_t=-\nabla\cdot(f\,p_t)+\frac12g^2\Delta p_t$.
2. Rewrite diffusion as transport: $\Delta p_t=\nabla\cdot(p_t\nabla\log p_t)$.
3. → $\partial_tp_t=-\nabla\cdot\left[\left(f-\tfrac12g^2\nabla\log p_t\right)p_t\right]$.
4. This is the continuity equation of the ODE $\dot x=f-\frac12g^2\nabla\log p_t$ → same $p_t$ for all $t$.
```

```{attention} Q&A
:class: dropdown
*SDE vs ODE sampling?*
- SDE: fresh noise corrects accumulated errors → strong quality at many steps.
- ODE: deterministic → high-order solvers & few steps, exact likelihood, invertible encoding $x_0\leftrightarrow x_1$.

*What is predictor–corrector sampling?*
- Predictor: 1 reverse-SDE step. Corrector: a few Langevin steps at the same $t$ → fix discretization error.

*Key numbers?*
- Unconditional CIFAR-10: FID 2.20, IS 9.89, 2.99 bits/dim.
- First 1024×1024 samples from a score-based model.
```

&nbsp;

### EDM
- **Name**: Name of the design space in "Elucidating the Design Space of Diffusion-Based Generative Models". The paper gives no expansion. {cite:p}`karras2022elucidating`
- **What**: Diffusion recast in noise level $\sigma$, w/ separate choices for preconditioning, loss weight, training noise, & sampler.
- **Why**: Prior work tied these choices to specific SDE derivations → unclear which choice mattered.
- **How**:
    1. Parametrize by noise level: $x=x_0+\sigma\epsilon$. Any VP model maps here by scaling.
    2. **Precondition**: skip connection + scaled net output → unit-variance input & target at every $\sigma$.
    3. **Train**: log-normal $\sigma$; loss weight cancels the output scale.
    4. **Sample**: Heun (2nd-order) on the probability-flow ODE. Steps dense at small $\sigma$.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $F_\theta$: Raw network.
- Hyperparams:
    - $\sigma_\text{data}=0.5$: Data std.
    - $\sigma_\text{min}=0.002$, $\sigma_\text{max}=80$: Sampling noise range.
    - $\rho=7$: Step-spacing exponent.
    - $N$: #noise levels at sampling.
- Misc:
    - $D_\theta(x;\sigma)$: Denoiser ≈ $\mathbb{E}[x_0|x]$.
- Override: $\sigma$ = VE noise level, $x=x_0+\sigma\epsilon$. VP ↔ VE: $\sigma=\sigma_t/a_t$, $x=x_t/a_t$.

Model:

$$
D_\theta(x;\sigma)=c_\text{skip}(\sigma)\,x+c_\text{out}(\sigma)\,F_\theta\left(c_\text{in}(\sigma)\,x;\,c_\text{noise}(\sigma)\right)
$$

$$
c_\text{skip}=\frac{\sigma_\text{data}^2}{\sigma^2+\sigma_\text{data}^2},\quad c_\text{out}=\frac{\sigma\,\sigma_\text{data}}{\sqrt{\sigma^2+\sigma_\text{data}^2}},\quad c_\text{in}=\frac{1}{\sqrt{\sigma^2+\sigma_\text{data}^2}},\quad c_\text{noise}=\tfrac14\ln\sigma
$$

Training:

$$
\mathbb{E}_{\sigma,x_0,\epsilon}\left[\lambda(\sigma)\left\|D_\theta(x_0+\sigma\epsilon;\sigma)-x_0\right\|^2\right],\quad\lambda(\sigma)=\frac{1}{c_\text{out}(\sigma)^2},\quad\ln\sigma\sim\mathcal{N}(-1.2,1.2^2)
$$

Inference:

$$
\frac{dx}{d\sigma}=\frac{x-D_\theta(x;\sigma)}{\sigma},\qquad\sigma_i=\left(\sigma_\text{max}^{1/\rho}+\frac{i}{N-1}\left(\sigma_\text{min}^{1/\rho}-\sigma_\text{max}^{1/\rho}\right)\right)^\rho,\ i=0,\dots,N-1
$$

- Append $\sigma_N=0$. Heun: 2 evals per step, Euler only on the last step → $2N-1$ NFE. $N=18$ → 35 NFE.
- Score: $\nabla_x\log p(x;\sigma)=(D_\theta(x;\sigma)-x)/\sigma^2$ ← Tweedie.
```

```{tip} Derivation
:class: dropdown
*Why these $c$ coefficients?*
1. Input: require $\text{Var}(c_\text{in}x)=c_\text{in}^2(\sigma_\text{data}^2+\sigma^2)=1$ → $c_\text{in}$.
2. Effective target of $F_\theta$: $(x_0-c_\text{skip}x)/c_\text{out}$. Its variance: $\frac{(1-c_\text{skip})^2\sigma_\text{data}^2+c_\text{skip}^2\sigma^2}{c_\text{out}^2}$.
3. Pick $c_\text{skip}$ to minimize the numerator → $c_\text{skip}=\frac{\sigma_\text{data}^2}{\sigma^2+\sigma_\text{data}^2}$ → $F_\theta$'s errors are amplified the least.
4. Set target variance to 1 → $c_\text{out}^2=\frac{\sigma^2\sigma_\text{data}^2}{\sigma^2+\sigma_\text{data}^2}$.
5. $\lambda=1/c_\text{out}^2$ → loss = unit-weighted MSE on $F_\theta$'s target at every $\sigma$.
- Small $\sigma$: $c_\text{skip}\approx1$ → net predicts a small correction (≈ noise). Large $\sigma$: $c_\text{skip}\approx0$ → net predicts $x_0$.

*Why is the ODE $dx/d\sigma=(x-D)/\sigma$?*
- VE w/ $\sigma(t)=t$: probability-flow drift $=-\sigma\nabla\log p$. Substitute the Tweedie score $(D-x)/\sigma^2$.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class EDM(nn.Module):
    def __init__(self, F_net, sigma_data=0.5):
        super().__init__()
        self.F, self.sd = F_net, sigma_data

    def denoise(self, x, sigma):                       ## sigma: tensor broadcastable to x
        sd2 = self.sd ** 2
        c_skip = sd2 / (sigma ** 2 + sd2)
        c_out = sigma * self.sd / (sigma ** 2 + sd2).sqrt()
        c_in = 1 / (sigma ** 2 + sd2).sqrt()
        ## net sees unit-variance input & regresses a unit-variance target at every sigma
        return c_skip * x + c_out * self.F(c_in * x, sigma.log() / 4)

    def loss(self, x0):
        sigma = (1.2 * torch.randn(x0.shape[0], *[1] * (x0.dim() - 1)) - 1.2).exp()  ## ln sigma ~ N(-1.2, 1.2^2)
        weight = (sigma ** 2 + self.sd ** 2) / (sigma * self.sd) ** 2                ## = 1 / c_out^2
        x = x0 + sigma * torch.randn_like(x0)
        return (weight * (self.denoise(x, sigma) - x0) ** 2).mean()

    @torch.no_grad()
    def sample(self, shape, N=18, s_min=0.002, s_max=80.0, rho=7):
        i = torch.arange(N)
        sig = (s_max ** (1 / rho) + i / (N - 1) * (s_min ** (1 / rho) - s_max ** (1 / rho))) ** rho
        sig = torch.cat([sig, torch.zeros(1)])                       ## end at sigma = 0
        x = sig[0] * torch.randn(shape)
        for s, s_next in zip(sig[:-1], sig[1:]):
            d = (x - self.denoise(x, s)) / s                         ## dx/dsigma
            x_next = x + (s_next - s) * d                            ## Euler
            if s_next > 0:                                           ## Heun correction
                d_next = (x_next - self.denoise(x_next, s_next)) / s_next
                x_next = x + (s_next - s) * (d + d_next) / 2
            x = x_next
        return x

## Example: stand-in net that ignores the noise input
net = nn.Conv2d(3, 3, 3, padding=1)
edm = EDM(lambda x, c: net(x))
print(edm.loss(torch.randn(2, 3, 8, 8)).item() > 0, edm.sample((2, 3, 8, 8)).shape)
## True torch.Size([2, 3, 8, 8])
```
````

```{attention} Q&A
:class: dropdown
*What mattered most?*
- Sampling: Heun + $\rho=7$ spacing → 35 NFE instead of hundreds to thousands.
- Training: preconditioning + log-normal $\sigma$ (focus on mid noise levels).
- Results: CIFAR-10 FID 1.79 (class-conditional), 1.97 (unconditional). ImageNet-64: 1.36 after retraining.

*Why $\rho=7$?*
- $\rho=3$ nearly equalizes the truncation error per step.
- $\rho$ in 5–10 samples better → errors at small $\sigma$ (fine detail) matter most → put more steps there.

*Why Heun?*
- 2nd order → much lower truncation error than Euler per NFE.
- EDM found Heun the best tradeoff between truncation error & NFE among the solvers it tested.
```

&nbsp;

## Fast Sampling

&nbsp;

### DDIM
- **Name**: Denoising Diffusion Implicit Models {cite:p}`song2021denoising`
- **What**: Non-Markovian forward process w/ the same marginals $q(x_t|x_0)$ as DDPM → deterministic, step-skipping sampler for a DDPM-trained net.
- **Why**:
    - *Why do we need it?* DDPM needs all $T$ (1000) sequential steps.
    - *Why does it work?* $L_\text{simple}$ depends only on the marginals $q(x_t|x_0)$, ❌the joint.
        - → Any process w/ the same marginals reuses the trained $\epsilon_\theta$.
        - → Pick a process that skips steps & adds no noise.
- **How**:
    1. Predict $\hat x_0$ from $x_t$ via $\epsilon_\theta$.
    2. Re-noise $\hat x_0$ to an earlier level $t'$ along the *predicted* noise direction (+ optional fresh noise).
    3. Use a sub-sequence of steps (e.g., 50 of 1000).

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\eta\in[0,1]$: Stochasticity.
- Misc:
    - $t'<t$: Next (less noisy) step in the sub-sequence.
    - $\hat x_0=(x_t-\sigma_t\epsilon_\theta(x_t,t))/a_t$: Predicted clean sample.

Process ($t\to t'$):

$$
x_{t'}=a_{t'}\hat x_0+\sqrt{\sigma_{t'}^2-\tilde\sigma_t^2}\;\epsilon_\theta(x_t,t)+\tilde\sigma_tz,\qquad\tilde\sigma_t=\eta\sqrt{\frac{\sigma_{t'}^2}{\sigma_t^2}\left(1-\frac{\bar\alpha_t}{\bar\alpha_{t'}}\right)}
$$

- $z\sim\mathcal{N}(0,I)$.
- $\eta=0$ → deterministic (DDIM).
- $\eta=1$, $t'=t-1$ → $\tilde\sigma_t^2=\tilde\beta_t$ → DDPM.
- ODE view ($\eta=0$): let $\bar x=x/a$, $\bar\sigma=\sigma/a$. Then $\bar x_{t'}=\bar x_t+(\bar\sigma_{t'}-\bar\sigma_t)\,\epsilon_\theta$ = Euler step on $d\bar x=\epsilon_\theta\,d\bar\sigma$.
```

````{important} Code
:class: dropdown
```python
import torch

## Reuses DDPM (and its ForwardProcess) from DDPM above
class DDIMSampler:
    def __init__(self, ddpm, n_steps=10, eta=0.0):
        self.m, self.eta = ddpm, eta
        self.ts = torch.linspace(ddpm.T - 1, 0, n_steps).long()   ## sub-sequence, e.g. 10 of T

    @torch.no_grad()
    def sample(self, shape):
        ab = self.m.fp.alpha_bar
        x = torch.randn(shape)
        for i, t in enumerate(self.ts):
            ab_t = ab[t]
            ab_next = ab[self.ts[i + 1]] if i + 1 < len(self.ts) else torch.tensor(1.0)  ## last target: clean
            eps = self.m.eps_model(x, torch.full((shape[0],), int(t)))
            x0 = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
            sig = self.eta * ((1 - ab_next) / (1 - ab_t) * (1 - ab_t / ab_next)).sqrt()
            ## re-noise x0 to the next level along the SAME predicted noise direction
            x = ab_next.sqrt() * x0 + (1 - ab_next - sig ** 2).sqrt() * eps + sig * torch.randn_like(x)
        return x

## Example: same trained DDPM (from above), 10 steps instead of 50
print(DDIMSampler(ddpm, n_steps=10).sample((2, 3, 8, 8)).shape)   ## torch.Size([2, 3, 8, 8])
```
````

```{attention} Q&A
:class: dropdown
*Does DDIM need retraining?*
- ❌. Same net, same $L_\text{simple}$. Only the sampler changes.

*Pros?*
- 10×–50× faster wall-clock than DDPM at similar quality.
- Deterministic ($\eta=0$) → $x_T$ = latent code → inversion (image → $x_T$) for editing, & interpolation in $x_T$.
- Same $x_T$ → similar high-level image across different step counts.

*Cons?*
- 1st-order (Euler) → quality drops at very few steps → [DPM-Solver](#dpm-solver).

*Why "implicit"?*
- $\eta=0$ → deterministic map $x_T\mapsto x_0$ → implicit generative model, like a GAN generator.
```

&nbsp;

### DPM-Solver
- **Name**: Diffusion Probabilistic Model Solver {cite:p}`lu2022dpm`
- **What**: High-order ODE solver for the probability-flow ODE that solves its linear part exactly.
- **Why**: Generic ODE solvers spend NFE on the linear term → slow. Exploit the semi-linear structure → 10–20 NFE.
- **How**:
    1. Change variable from $t$ to half log-SNR $\lambda$.
    2. Linear part → exact (exponential integrator). Left: an $e^{-\lambda}$-weighted integral of $\epsilon_\theta$.
    3. Taylor-expand $\epsilon_\theta$ in $\lambda$. Order 1 = DDIM. Orders 2, 3 → extra intermediate evals.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $s>t$: Start (noisier) & end time of 1 solver step.
    - $\lambda_t=\log(a_t/\sigma_t)=\frac12\log\text{SNR}(t)$: Half log-SNR.
    - $h=\lambda_t-\lambda_s>0$: Step size in $\lambda$.

Exact solution:

$$
x_t=\frac{a_t}{a_s}x_s-a_t\int_{\lambda_s}^{\lambda_t}e^{-\lambda}\,\hat\epsilon_\theta(\hat x_\lambda,\lambda)\,d\lambda
$$

- $\hat\epsilon_\theta(\hat x_\lambda,\lambda)$: Noise prediction re-indexed by $\lambda$.

1st order (= DDIM, $\eta=0$):

$$
x_t=\frac{a_t}{a_s}x_s-\sigma_t\left(e^{h}-1\right)\epsilon_\theta(x_s,s)
$$

- Check: $\sigma_te^h=a_t\sigma_s/a_s$ → $x_t=\frac{a_t}{a_s}x_s+\left(\sigma_t-\frac{a_t\sigma_s}{a_s}\right)\epsilon_\theta$ = DDIM.
```

```{attention} Q&A
:class: dropdown
*Why change variable to $\lambda$?*
- In $\lambda$, the linear part integrates exactly → the remaining integrand is smooth → low-order Taylor is accurate.

*Key numbers?*
- CIFAR-10: FID 4.70 at 10 NFE, 2.87 at 20 NFE. 4–16× speedup over prior training-free samplers.

*What breaks under guidance?*
- Large [CFG](#cfg) scale → high-order $\epsilon$-solvers become unstable, even slower than DDIM.
- DPM-Solver++ solves in $x_0$-prediction form + thresholding + multistep → 15–20 steps for guided sampling {cite:p}`lu2022dpmpp`.
```

&nbsp;

### Progressive Distillation
- **What**: Distill a $2N$-step deterministic sampler into an $N$-step student. Repeat. {cite:p}`salimans2022progressive`
- **Why**: Good ODE solvers still need ~10+ NFE. Distillation trades training compute for 4–8-step sampling.
- **How**:
    1. Student ← copy of teacher.
    2. Target: 1 student DDIM step $t\to t''$ must land where 2 teacher DDIM steps $t\to t'\to t''$ land.
    3. Train → student becomes the new teacher → halve steps again.
    4. Predict $v$ or $x_0$, ❌$\epsilon$ ← 1 step spans a large noise range → $\hat x_0=(x_t-\sigma_t\hat\epsilon)/a_t$ unstable at $a_t\to0$.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $t>t'>t''$: 3 consecutive teacher time points.
    - $x_{t''}$: Teacher output after 2 DDIM steps from $x_t$.
    - $\tilde x$: $x_0$-target for the student.
    - $\hat x_\phi(x_t)$: Student's $x_0$ prediction.

Objective:

$$
\tilde x=\frac{x_{t''}-\frac{\sigma_{t''}}{\sigma_t}x_t}{a_{t''}-\frac{\sigma_{t''}}{\sigma_t}a_t},\qquad\mathcal{L}=\mathbb{E}\left[w(t)\,\|\tilde x-\hat x_\phi(x_t)\|^2\right]
$$

- $w(t)$: $\max(\text{SNR}(t),1)$ ("truncated SNR") or $1+\text{SNR}(t)$ (= $v$-loss).
```

```{tip} Derivation
:class: dropdown
*Where does $\tilde x$ come from?*
1. 1 DDIM step from $x_t$ w/ predicted $\hat x$: implied noise $\hat\epsilon=(x_t-a_t\hat x)/\sigma_t$ → $x_{t''}=a_{t''}\hat x+\sigma_{t''}\hat\epsilon$.
2. Require it to equal the teacher's 2-step result $x_{t''}$.
3. Solve the linear equation for $\hat x$ → $\tilde x$.
```

````{important} Code
:class: dropdown
```python
import torch

class DistillTarget:
    def __init__(self, teacher_x0):          ## teacher_x0(x, t) -> predicted x0
        self.f = teacher_x0

    @staticmethod
    def ddim(x, x0, a, s, a_n, s_n):         ## deterministic DDIM step: re-noise x0 w/ the implied eps
        return a_n * x0 + s_n * (x - a * x0) / s

    @torch.no_grad()
    def __call__(self, x, t, t1, t2, sched):
        (a, s), (a1, s1), (a2, s2) = sched(t), sched(t1), sched(t2)
        x1 = self.ddim(x, self.f(x, t), a, s, a1, s1)        ## teacher step 1
        x2 = self.ddim(x1, self.f(x1, t1), a1, s1, a2, s2)   ## teacher step 2
        return (x2 - s2 / s * x) / (a2 - s2 / s * a), x2     ## x0-target: 1 student step hits x2

## Example: cosine schedule, stand-in teacher
sched = lambda t: (torch.cos(t * torch.pi / 2), torch.sin(t * torch.pi / 2))   ## a^2 + s^2 = 1
t, t1, t2 = torch.tensor(0.9), torch.tensor(0.6), torch.tensor(0.3)
x = torch.randn(4)
target, x2 = DistillTarget(lambda x, t: 0.5 * x)(x, t, t1, t2, sched)
(a, s), (a2, s2) = sched(t), sched(t2)
print(torch.allclose(DistillTarget.ddim(x, target, a, s, a2, s2), x2))   ## True
```
````

```{attention} Q&A
:class: dropdown
*Key numbers?*
- Start from samplers w/ up to 8192 steps → distill to 4 steps.
- CIFAR-10: FID 3.0 at 4 steps.
- Total distillation cost ≤ cost of training the original model.

*Cons?*
- $\log_2$(#steps) rounds of training.
- Quality drops at 1–2 steps.
- Each round inherits the previous student's errors.
```

&nbsp;

### Consistency Model
- **What**: Net $f_\theta(x_t,t)$ that maps every point on 1 probability-flow ODE trajectory to the trajectory's origin. {cite:p}`song2023consistency`
- **Why**: 1-step generation w/o adversarial training. Keep an optional multi-step mode.
- **How**:
    1. **Boundary**: $f_\theta(x,t_\text{min})=x$ ← skip parametrization.
    2. **Self-consistency**: outputs at adjacent points $t_{n+1},t_n$ of the same trajectory must agree.
    3. Adjacent pair from:
        - **CD** (distillation): 1 ODE-solver step w/ a pretrained diffusion model.
        - **CT** (training): same $x_0$ & same noise at both levels. ❌Teacher.
    4. Target net $\theta^-$ = EMA of $\theta$, stop-grad.
    5. Sample: 1 step $f_\theta(x_T,T)$. Or multistep: denoise → re-noise to a smaller $t$ → denoise.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\theta^-$: EMA target params.
- Hyperparams:
    - $t_\text{min}$: Smallest noise level (0.002).
    - $\mu$: EMA rate.
- Misc:
    - $t_1<\dots<t_N$: Discretized noise levels.
    - $d(\cdot,\cdot)$: Distance (L2, L1, LPIPS).
- Override: $t$ = VE noise level, $x_t=x_0+t\epsilon$ ([EDM](#edm) style).

Model:

$$
f_\theta(x,t)=c_\text{skip}(t)\,x+c_\text{out}(t)\,F_\theta(x,t),\quad c_\text{skip}(t_\text{min})=1,\ c_\text{out}(t_\text{min})=0
$$

Training:

$$
\mathcal{L}=\mathbb{E}_{n,x_0,\epsilon}\left[\lambda(t_n)\,d\left(f_\theta(x_{t_{n+1}},t_{n+1}),\,f_{\theta^-}(\hat x_{t_n},t_n)\right)\right],\qquad\theta^-\leftarrow\mu\theta^-+(1-\mu)\theta
$$

- $\lambda(t_n)$: Per-level weight.
- CD: $\hat x_{t_n}$ = 1 ODE-solver step from $x_{t_{n+1}}$ using the teacher.
- CT: $\hat x_{t_n}=x_0+t_n\epsilon$, same $\epsilon$ as $x_{t_{n+1}}=x_0+t_{n+1}\epsilon$.
```

````{important} Code
:class: dropdown
```python
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConsistencyModel(nn.Module):
    def __init__(self, net, t_min=0.002, sigma_data=0.5):
        super().__init__()
        self.net, self.tmin, self.sd = net, t_min, sigma_data

    def forward(self, x, t):
        c_skip = self.sd ** 2 / ((t - self.tmin) ** 2 + self.sd ** 2)
        c_out = self.sd * (t - self.tmin) / (t ** 2 + self.sd ** 2).sqrt()
        return c_skip * x + c_out * self.net(x, t)            ## f(x, t_min) = x exactly

    def ct_loss(self, x0, t_n, t_n1, ema):                    ## consistency training, no teacher
        eps = torch.randn_like(x0)                            ## SAME noise at both levels
        with torch.no_grad():
            target = ema(x0 + t_n * eps, t_n)                 ## stop-grad EMA target at smaller t
        return F.mse_loss(self(x0 + t_n1 * eps, t_n1), target)

## Example: stand-in net that ignores t
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = nn.Linear(4, 4)
    def forward(self, x, t):
        return self.l(x)

cm = ConsistencyModel(Net())
ema = copy.deepcopy(cm)                                       ## EMA target net
x0 = torch.randn(8, 4)
print(cm.ct_loss(x0, torch.tensor(1.0), torch.tensor(1.5), ema).shape)   ## torch.Size([])
print(torch.allclose(cm(x0, torch.tensor(0.002)), x0))                   ## True: boundary condition
```
````

```{attention} Q&A
:class: dropdown
*Key numbers?*
- 1-step FID (CD): 3.55 CIFAR-10, 6.20 ImageNet 64×64.

*CD vs progressive distillation?*
- CD distills the ODE trajectory directly → 1 training run.
- Progressive distillation distills the sampler → $\log_2$(#steps) rounds.

*Why the boundary condition?*
- Without it, $f_\theta\equiv$ const satisfies self-consistency → collapse.
- $f_\theta(x,t_\text{min})=x$ anchors every trajectory to data.

*Cons?*
- 1-step quality < multi-step diffusion.
- Sensitive to the $t$-discretization & EMA schedule.
```

&nbsp;

## Guidance
- **What**: Steer sampling toward condition $c$ by adding a scaled $\nabla_{x_t}\log p(c|x_t)$ to the score.

&nbsp;

### Classifier Guidance
- **What**: Add a noisy-image classifier's gradient $\nabla_{x_t}\log p_\phi(c|x_t)$, scaled by $w$, to the diffusion score. {cite:p}`dhariwal2021diffusion`
- **Why**: Conditional diffusion alone → diverse but lower-fidelity samples. ❌Knob to trade diversity for fidelity, like GAN truncation.
- **How**:
    1. Train a classifier $p_\phi(c|x_t,t)$ on noisy $x_t$.
    2. Each sampling step: shift the noise prediction by $-w\,\sigma_t\nabla_{x_t}\log p_\phi(c|x_t)$.
    3. $w>1$ → sharper class distribution → fidelity ⬆️, diversity ⬇️.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $p_\phi(c|x_t)$: Noisy-image classifier.
- Hyperparams:
    - $w$: Guidance scale.

Inference:

$$
\nabla\log p_t(x_t|c)=\nabla\log p_t(x_t)+\nabla\log p_t(c|x_t)
$$

$$
\tilde\epsilon(x_t,c)=\epsilon_\theta(x_t,c)-w\,\sigma_t\nabla_{x_t}\log p_\phi(c|x_t)
$$

- ← score $=-\epsilon/\sigma_t$.
- Unconditional base $\epsilon_\theta(x_t)$: implied target ∝ $p_t(x_t)\,p_t(c|x_t)^w$.
- Conditional base $\epsilon_\theta(x_t,c)$ (as in ADM): implied target ∝ $p_t(x_t|c)\,p_t(c|x_t)^w\propto p_t(x_t)\,p_t(c|x_t)^{w+1}$.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class ClassifierGuidance:
    def __init__(self, eps_model, classifier, w):
        self.eps, self.clf, self.w = eps_model, classifier, w

    def __call__(self, x, t, c, sigma_t):
        with torch.enable_grad():
            xg = x.detach().requires_grad_(True)
            logp = self.clf(xg, t).log_softmax(-1)[torch.arange(len(c)), c].sum()   ## log p(c|x_t)
            grad = torch.autograd.grad(logp, xg)[0]
        ## score += w * grad  <=>  eps -= w * sigma_t * grad
        return self.eps(x, t) - self.w * sigma_t * grad

## Example: stand-in eps model & linear classifier over 10 classes
clf_net = nn.Linear(3 * 8 * 8, 10)
g = ClassifierGuidance(lambda x, t: torch.zeros_like(x), lambda x, t: clf_net(x.flatten(1)), w=2.0)
print(g(torch.randn(2, 3, 8, 8), torch.tensor([500, 500]), torch.tensor([1, 7]), 0.8).shape)
## torch.Size([2, 3, 8, 8])
```
````

```{attention} Q&A
:class: dropdown
*Cons?*
- Extra classifier, trained on noisy $x_t$. Off-the-shelf classifiers fail on noisy inputs.
- Gradient can act like an adversarial attack on the classifier → fools it w/o real class features.
- Extra backward pass per step.

*Key result?*
- First diffusion model to beat BigGAN-deep on ImageNet. ImageNet 256×256 FID 4.59.
```

&nbsp;

### CFG
- **Name**: Classifier-Free Guidance {cite:p}`ho2022classifier`
- **What**: Guidance by extrapolating from the unconditional toward the conditional prediction of 1 jointly trained net.
- **Why**: Classifier guidance needs a separate noisy classifier & its gradient can be adversarial.
    - Bayes: $\nabla\log p(c|x_t)=\nabla\log p(x_t|c)-\nabla\log p(x_t)$ → implicit classifier from 2 scores.
- **How**:
    1. **Training**: replace $c$ by $\varnothing$ w/ prob $p_\text{uncond}$ (e.g., 10–20%) → 1 net learns $\epsilon_\theta(x_t,c)$ & $\epsilon_\theta(x_t,\varnothing)$.
    2. **Inference**: 2 predictions per step → extrapolate from unconditional toward conditional.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $s$: Guidance scale.
    - $p_\text{uncond}$: Condition-dropout prob.

Inference:

$$
\tilde\epsilon_\theta(x_t,c)=\epsilon_\theta(x_t,\varnothing)+s\left(\epsilon_\theta(x_t,c)-\epsilon_\theta(x_t,\varnothing)\right)
$$

- $s=0$ → unconditional. $s=1$ → plain conditional. $s>1$ → guided.
- Ho & Salimans write $(1+w)\epsilon_\theta(x_t,c)-w\,\epsilon_\theta(x_t,\varnothing)$ → $s=1+w$.
- Implied score: $\nabla\log p_t(x_t)+s\nabla\log p_t(c|x_t)$ ⇔ $s\nabla\log p_t(x_t|c)+(1-s)\nabla\log p_t(x_t)$.
```

```{tip} Derivation
:class: dropdown
*Why is CFG an implicit classifier?*
1. Bayes: $\log p_t(c|x_t)=\log p_t(x_t|c)-\log p_t(x_t)+\log p(c)$.
2. Gradient in $x_t$: $\nabla\log p_t(c|x_t)=\nabla\log p_t(x_t|c)-\nabla\log p_t(x_t)$.
3. Score $=-\epsilon/\sigma_t$ → $-\sigma_t\nabla\log p_t(c|x_t)\approx\epsilon_\theta(x_t,c)-\epsilon_\theta(x_t,\varnothing)$.
4. Plug into classifier guidance w/ $w=s-1$ on top of the conditional model → CFG formula.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class CFG(nn.Module):
    def __init__(self, eps_model, null_c, p_uncond=0.1):
        super().__init__()
        self.eps, self.null, self.p = eps_model, null_c, p_uncond

    def train_cond(self, c):                          ## training: randomly replace c by the null token
        drop = torch.rand(c.shape[0]) < self.p
        return torch.where(drop, self.null, c)

    def forward(self, x, t, c, s=7.5):
        ## cond & uncond in 1 batched pass -> 2x compute
        e = self.eps(torch.cat([x, x]), torch.cat([t, t]), torch.cat([c, self.null.expand_as(c)]))
        e_c, e_u = e.chunk(2)
        return e_u + s * (e_c - e_u)                  ## extrapolate away from unconditional

## Example: stand-in conditional net; label 0 = null
eps_model = lambda x, t, c: x * c.view(-1, 1, 1, 1)
cfg = CFG(eps_model, null_c=torch.tensor(0))
x, t, c = torch.randn(2, 3, 8, 8), torch.tensor([500, 500]), torch.tensor([3, 5])
print(cfg(x, t, c).shape)   ## torch.Size([2, 3, 8, 8])
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- ❌Classifier. Works for any condition, incl. free-form text.
- 1 extra hyperparam $s$ → quality/diversity knob at inference.

*Cons?*
- 2× NFE per step.
- Large $s$ → oversaturated colors, artifacts, diversity ⬇️.

*Typical values?*
- $p_\text{uncond}$: 0.1–0.2 work well; 0.5 worse.
- Ho & Salimans (ImageNet): best FID at small $w$; IS keeps rising w/ $w$ → quality–diversity tradeoff.
- Stable Diffusion default $s=7.5$.

*What is a negative prompt?*
- Replace $\varnothing$ by the embedding of a negative prompt → extrapolate away from it.

*Does CFG sample exactly from $p(x|c)^s\,p(x)^{1-s}$?*
- ❌. Noising does not commute w/ tilting.
- $p_t(x|c)^s\,p_t(x)^{1-s}$ ≠ Gaussian-smoothed $p_0(x|c)^s\,p_0(x)^{1-s}$.
- → CFG uses a heuristic score. It works empirically, ❌exact sampler of any tilted target.

*Why does guidance improve perceived quality?*
- The learned conditional is too spread out. Guidance sharpens it toward class-typical, high-likelihood samples.
- Same spirit as low-temperature sampling or GAN truncation.
```

&nbsp;

## Architecture
- **What**: Denoiser net $\epsilon_\theta(x_t,t,c)$: output has the same shape as the input; conditioned on $t$ & $c$.

&nbsp;

### Timestep Embedding
- **What**: Map scalar $t$ (or $\sigma$) to a vector that modulates every block. {cite:p}`ho2020denoising`
- **Why**: 1 shared net serves all noise levels → it must know the level ← denoising strength depends on it.
    - Raw scalar $t$ = 1 weak input. Sinusoids → multi-scale features → MLP → rich per-block signal.
- **How**:
    1. $t$ → [sinusoidal](../../nlp/pos.md#sinusoidal) features.
    2. → MLP (Linear–SiLU–Linear).
    3. Inject into each residual block:
        - Add as a per-channel bias (DDPM).
        - Scale & shift after GroupNorm (AdaGN, ADM) {cite:p}`dhariwal2021diffusion`.
        - Scale, shift & gate around LayerNorm (adaLN, [DiT](#dit)).
    - Class label: add a class embedding to the timestep embedding.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $h$: Feature map inside a residual block.
- Hyperparams:
    - $D$: Sinusoid dim.
- Misc:
    - $\mathbf{e}$: Timestep (+ class) embedding.
    - $\mathbf{s},\mathbf{b}$: Per-channel scale & shift, linear in $\mathbf{e}$.

Forward:

$$
\mathbf{e}(t)=\text{MLP}\left(\left[\sin(\omega_kt),\cos(\omega_kt)\right]_{k=0}^{D/2-1}\right),\quad\omega_k=10000^{-2k/D}
$$

AdaGN:

$$
h\leftarrow\mathbf{s}(\mathbf{e})\odot\text{GroupNorm}(h)+\mathbf{b}(\mathbf{e})
$$

- Implementations often use $(1+\mathbf{s})$ → near-identity at init.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class TimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.SiLU(), nn.Linear(4 * dim, 4 * dim))

    def forward(self, t):                                            ## t: (B,) int or float
        k = torch.arange(self.dim // 2)
        freqs = 10000 ** (-2 * k / self.dim)                          ## geometric: fast -> slow
        ang = t.float()[:, None] * freqs[None]                        ## (B, dim/2)
        return self.mlp(torch.cat([ang.sin(), ang.cos()], dim=-1))    ## (B, 4*dim)

## Example
print(TimestepEmbedding(32)(torch.tensor([0, 999])).shape)   ## torch.Size([2, 128])
```
````

```{attention} Q&A
:class: dropdown
*Why sinusoids, ❌raw $t$?*
- Raw $t$ = 1 number → net must build a multi-scale code itself.
- Sinusoids give fine (fast) & coarse (slow) features of $t$ → easy to separate nearby noise levels.

*Add vs scale-shift?*
- Add: shifts features only.
- Scale-shift (AdaGN / adaLN): also rescales features per noise level → better FID in ADM ablations.
```

&nbsp;

### U-Net
- **What**: Conv encoder–decoder w/ skip connections at every resolution. {cite:p}`ronneberger2015u`
- **Why**: Output = same shape as input. Must see global context (high noise) & keep pixel detail (low noise).
    - Encoder downsamples → large receptive field.
    - Skips carry fine detail directly to the decoder.
- **How** (diffusion U-Net):
    1. **Down path**: ResBlocks (+ timestep embedding) → self-attn at low resolution → downsample. Repeat.
    2. **Middle**: ResBlock → self-attn → ResBlock at the lowest resolution.
    3. **Up path**: upsample → concat the skip from the same resolution → ResBlocks (+ attn).
    - DDPM: Wide-ResNet blocks, GroupNorm, self-attn at 16×16 only.
    - ADM: attn at 32, 16, 8; 64 channels per head; BigGAN up/down ResBlocks; AdaGN {cite:p}`dhariwal2021diffusion`.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $h$: Block input feature map.
- Misc:
    - $\text{GN}$: GroupNorm.
    - $\text{Skip}$: Identity, or 1×1 conv if #channels change.

Forward (ResBlock):

$$
u=\text{Conv}\left(\text{SiLU}(\text{GN}(h))\right)+\text{Linear}(\mathbf{e}),\qquad h'=\text{Skip}(h)+\text{Conv}\left(\text{Dropout}(\text{SiLU}(\text{GN}(u)))\right)
$$

- $\mathbf{e}$: [Timestep embedding](#timestep-embedding), broadcast over $H\times W$.
- $u$: Intermediate feature map.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

## Reuses TimestepEmbedding from above
class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, t_dim):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, c_in), nn.Conv2d(c_in, c_out, 3, padding=1)
        self.t = nn.Linear(t_dim, c_out)                                  ## per-channel bias from t
        self.n2, self.c2 = nn.GroupNorm(8, c_out), nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x, e):
        h = self.c1(F.silu(self.n1(x))) + self.t(e)[:, :, None, None]
        return self.skip(x) + self.c2(F.silu(self.n2(h)))

class UNet(nn.Module):
    def __init__(self, c=3, ch=32):
        super().__init__()
        self.temb = TimestepEmbedding(ch)                                 ## -> (B, 4*ch)
        self.inp = nn.Conv2d(c, ch, 3, padding=1)
        self.down = ResBlock(ch, ch, 4 * ch)
        self.pool = nn.Conv2d(ch, 2 * ch, 3, stride=2, padding=1)         ## H -> H/2
        self.mid = ResBlock(2 * ch, 2 * ch, 4 * ch)
        self.attn = nn.MultiheadAttention(2 * ch, 4, batch_first=True)    ## global context, low res only
        self.up = nn.ConvTranspose2d(2 * ch, ch, 2, stride=2)             ## H/2 -> H
        self.dec = ResBlock(2 * ch, ch, 4 * ch)                           ## 2*ch in: concat w/ skip
        self.out = nn.Conv2d(ch, c, 3, padding=1)

    def forward(self, x, t):
        e = self.temb(t)
        h1 = self.down(self.inp(x), e)                                    ## keep for skip: fine detail
        h = self.mid(self.pool(h1), e)
        B, C, H, W = h.shape
        tok = h.flatten(2).transpose(1, 2)                                ## (B, HW, C)
        h = h + self.attn(tok, tok, tok)[0].transpose(1, 2).reshape(B, C, H, W)
        h = self.dec(torch.cat([self.up(h), h1], dim=1), e)               ## U-shape: skip concat
        return self.out(h)                                                ## same shape as x

## Example
print(UNet()(torch.randn(2, 3, 16, 16), torch.tensor([10, 900])).shape)   ## torch.Size([2, 3, 16, 16])
```
````

```{attention} Q&A
:class: dropdown
*Why are skip connections critical?*
- W/o them, detail must pass through the bottleneck → lost.
- Denoising at low noise = pixel-exact correction → needs full-resolution features.

*Why self-attn only at low resolution?*
- Cost $O((HW)^2)$ → affordable only on small feature maps.

*Is the U-shape required?*
- ❌. [DiT](#dit) matches or beats it w/o conv or U-shape.
- U-Net conv bias helps at small scale. Transformers scale more predictably.
```

&nbsp;

### Cross-Attention Conditioning
- **What**: Condition on a token sequence (e.g., text) via [cross-attention](../../nlp/attn.md#cross-attention): queries from image features, keys/values from condition tokens. {cite:p}`rombach2022high`
- **Why**: A class label = 1 vector → add it to the timestep embedding.
    - Text = variable-length token sequence → needs per-token, spatially varying conditioning.
- **How**:
    1. Encode the prompt w/ a frozen text encoder (CLIP, T5) → token features $\tau(c)$.
    2. At each U-Net resolution: self-attn → cross-attn (Q = pixels, K/V = text tokens) → FFN.
    3. Each pixel pulls the text features relevant to it.
    - SD 1.x: frozen CLIP ViT-L/14 text encoder, 77 tokens × 768 dims.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $h\in\mathbb{R}^{HW\times d}$: Flattened feature map.
    - $\tau(c)\in\mathbb{R}^{M\times d_c}$: Condition tokens.
- Params:
    - $W_Q\in\mathbb{R}^{d\times d_k}$: Query projection.
    - $W_K\in\mathbb{R}^{d_c\times d_k}$: Key projection.
    - $W_V\in\mathbb{R}^{d_c\times d_k}$: Value projection.
- Hyperparams:
    - $M$: #condition tokens.

Forward:

$$
\text{Attn}(Q,K,V)=\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V,\quad Q=hW_Q,\ K=\tau(c)W_K,\ V=\tau(c)W_V
$$

- Output $\in\mathbb{R}^{HW\times d_k}$ → projected back to $d$ & added residually.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class CrossAttnBlock(nn.Module):
    def __init__(self, d, d_c, heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, kdim=d_c, vdim=d_c, batch_first=True)

    def forward(self, h, ctx):                       ## h: (B,C,H,W) image feats; ctx: (B,M,d_c) text tokens
        B, C, H, W = h.shape
        q = h.flatten(2).transpose(1, 2)              ## (B, HW, C): 1 query per pixel
        out = self.attn(self.norm(q), ctx, ctx)[0]    ## each pixel reads its relevant text tokens
        return h + out.transpose(1, 2).reshape(B, C, H, W)

## Example: 8x8 feature map, 77 text tokens of width 768 (SD 1.x CLIP)
print(CrossAttnBlock(64, 768)(torch.randn(2, 64, 8, 8), torch.randn(2, 77, 768)).shape)
## torch.Size([2, 64, 8, 8])
```
````

```{attention} Q&A
:class: dropdown
*Why a frozen, pretrained text encoder?*
- Reuses language understanding from far more text than image–caption pairs.
- Imagen: scaling the text encoder (T5) improves fidelity & image–text alignment much more than scaling the diffusion model {cite:p}`saharia2022photorealistic`.

*Cross-attn vs adding a pooled text vector?*
- Pooled vector: 1 global signal, same for every pixel.
- Cross-attn: token-level & spatial → each region binds to its words.
```

&nbsp;

### LDM
- **Name**: Latent Diffusion Model {cite:p}`rombach2022high`
- **What**: Diffusion in the latent space of a pretrained autoencoder.
- **Why**: Pixel-space diffusion spends compute on imperceptible high-frequency detail → hundreds of GPU-days to train, slow sampling.
    - Perceptual compression (autoencoder) ≠ semantic generation (diffusion) → split them.
- **How**:
    1. Train an autoencoder once: encoder $\mathcal{E}$, decoder $\mathcal{D}$, downsampling factor $f$. Loss: perceptual + patch-adversarial + light KL (or VQ) regularization.
    2. Train diffusion on $z_0=\mathcal{E}(x)$.
    3. Sample $z_0$ w/ diffusion → decode $\mathcal{D}(z_0)$.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $z_0=\mathcal{E}(x)$: Clean latent.
- Hyperparams:
    - $f$: Spatial downsampling factor.

Training:

$$
L_\text{LDM}=\mathbb{E}_{\mathcal{E}(x),c,\epsilon,t}\left[\left\|\epsilon-\epsilon_\theta\left(z_t,t,\tau(c)\right)\right\|^2\right],\quad z_t=a_t\mathcal{E}(x)+\sigma_t\epsilon
$$

- $f\in\{4,8,16\}$ = best tradeoff in the paper. SD: $f=8$, 4 channels: $512\times512\times3\to64\times64\times4$ (48× fewer dims).
- Latent rescaled to ~unit variance (SD 1.x: ×0.18215).
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

## Reuses DDPM from above (any diffusion objective works on z)
class LDM(nn.Module):
    def __init__(self, encoder, decoder, diffusion, scale=0.18215):
        super().__init__()
        self.E, self.D, self.diff, self.scale = encoder, decoder, diffusion, scale

    def loss(self, x):
        with torch.no_grad():
            z0 = self.E(x) * self.scale       ## frozen AE; rescale latent to ~unit variance
        return self.diff.loss(z0)             ## diffusion never sees pixels

    @torch.no_grad()
    def sample(self, z_shape):
        return self.D(self.diff.sample(z_shape) / self.scale)

## Example: stand-in f=4 autoencoder on 16x16 images
enc, dec = nn.Conv2d(3, 4, 4, stride=4), nn.ConvTranspose2d(4, 3, 4, stride=4)
ldm = LDM(enc, dec, DDPM(lambda z, t: z, torch.linspace(1e-4, 0.02, 20)))
print(ldm.loss(torch.randn(2, 3, 16, 16)).item() >= 0, ldm.sample((2, 4, 4, 4)).shape)
## True torch.Size([2, 3, 16, 16])
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- Compute ⬇️ by the latent compression ratio. Each step is cheaper.
- Conv backbone → can generate at sizes other than the training size.
- General conditioning via [cross-attn](#cross-attention-conditioning).

*Cons?*
- Autoencoder reconstruction caps quality (small text, small faces, fine texture).
- 2-stage training. A new autoencoder → retrain the diffusion model.

*Why only a light KL on the latent?*
- Strong KL → VAE-style blur.
- Diffusion models the latent prior → the autoencoder needs no Gaussian prior, only a well-scaled latent.
```

&nbsp;

### DiT
- **Name**: Diffusion Transformer {cite:p}`peebles2023scalable`
- **What**: LDM w/ the U-Net replaced by a transformer over latent patches.
- **Why**: Transformers scale predictably. The U-Net's conv bias is not required.
- **How**:
    1. **Patchify**: latent $I\times I\times C$ → $(I/p)^2$ tokens + 2D sinusoidal position embeddings.
    2. $N$ DiT blocks: self-attn + MLP, conditioned via **adaLN-Zero**.
    3. Linear decoder → per-patch noise (& variance) → unpatchify.
    - adaLN-Zero: MLP($t$-emb + $c$-emb) → per-block scale, shift, & residual gate. Gate init 0 → each block = identity at init.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $h$: Token sequence, $(I/p)^2$ tokens.
- Hyperparams:
    - $p$: Patch size.
    - $I$: Latent spatial size.
- Misc:
    - $\mathbf{s}_{1,2},\mathbf{b}_{1,2},\mathbf{g}_{1,2}$: Scale, shift, gate per sub-layer.

Forward:

$$\begin{align*}
h&\leftarrow h+\mathbf{g}_1\odot\text{Attn}\left((1+\mathbf{s}_1)\odot\text{LN}(h)+\mathbf{b}_1\right)\\
h&\leftarrow h+\mathbf{g}_2\odot\text{MLP}\left((1+\mathbf{s}_2)\odot\text{LN}(h)+\mathbf{b}_2\right)
\end{align*}$$

- $(\mathbf{s}_{1,2},\mathbf{b}_{1,2},\mathbf{g}_{1,2})=\text{Linear}\left(\text{SiLU}(\mathbf{e}(t)+\mathbf{e}(c))\right)$, zero-init.
- LN w/o learnable affine params.
- $p$⬇️ → #tokens⬆️ → Gflops⬆️ at ~same #params.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class DiTBlock(nn.Module):
    def __init__(self, d, heads=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d, elementwise_affine=False)   ## affine comes from the condition
        self.ln2 = nn.LayerNorm(d, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        nn.init.zeros_(self.ada[1].weight)                      ## "-Zero": gates = 0 -> block = identity
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, h, cond):                                 ## h: (B,N,d) tokens; cond: (B,d) = e(t) + e(c)
        b1, s1, g1, b2, s2, g2 = self.ada(cond)[:, None].chunk(6, dim=-1)
        x = self.ln1(h) * (1 + s1) + b1
        h = h + g1 * self.attn(x, x, x)[0]
        return h + g2 * self.mlp(self.ln2(h) * (1 + s2) + b2)

## Example: 16x16x4 latent, patch p=2 -> 64 tokens
p, d = 2, 64
z = torch.randn(2, 4, 16, 16)
tokens = nn.Conv2d(4, d, p, stride=p)(z).flatten(2).transpose(1, 2)   ## patchify: (2, 64, d)
print(DiTBlock(d)(tokens, torch.randn(2, d)).shape)                     ## torch.Size([2, 64, 64])
```
````

```{attention} Q&A
:class: dropdown
*Which conditioning won?*
- adaLN-Zero > adaLN > cross-attn > in-context tokens (FID & compute).
- $t$ & class = 1 global vector → modulation is cheap & sufficient.
- Zero-init gates → residual blocks start as identity → stable training.

*What predicts quality?*
- Gflops, ❌#params. More depth/width or more tokens (smaller $p$) → FID ⬇️ consistently.

*Key numbers?*
- DiT-XL/2: 28 layers, width 1152, 16 heads, ~675M params, $p=2$.
- ImageNet 256×256 class-conditional: FID 2.27 (w/ CFG).

*How to condition on text?*
- DiT itself is class-conditional. Text → add cross-attn, or use joint attention ([MM-DiT](#mm-dit)).
```

&nbsp;

#### MM-DiT
- **Name**: Multimodal Diffusion Transformer {cite:p}`esser2024scaling`
- **What**: DiT where text & image tokens keep separate weights but share 1 joint attention.
- **Why**: Cross-attn = 1-way: image reads text, text features never update.
    - Joint attention → bidirectional text ↔ image flow → better prompt following & typography.
- **How**:
    1. 2 token streams: image patches & text tokens. Each has its own LN, QKV, MLP weights.
    2. Concatenate Q, K, V of both streams → 1 attention → split back.
    3. adaLN modulation from $t$ + pooled text embedding.
    - SD3: CLIP-L, CLIP-G & T5-XXL text encoders. Rectified-flow objective ([Flow Matching](#flow-matching)). Up to 8B params.

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MMDiTAttention(nn.Module):
    def __init__(self, d, heads=4):
        super().__init__()
        self.h = heads
        self.qkv_img, self.qkv_txt = nn.Linear(d, 3 * d), nn.Linear(d, 3 * d)   ## separate weights per modality
        self.out_img, self.out_txt = nn.Linear(d, d), nn.Linear(d, d)

    def forward(self, x_img, x_txt):
        N = x_img.shape[1]
        qkv = torch.cat([self.qkv_img(x_img), self.qkv_txt(x_txt)], dim=1)     ## 1 joint sequence
        q, k, v = (u.unflatten(-1, (self.h, -1)).transpose(1, 2) for u in qkv.chunk(3, dim=-1))
        o = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).flatten(2) ## every token sees both
        return self.out_img(o[:, :N]), self.out_txt(o[:, N:])                  ## split back per stream

## Example: 64 image tokens + 77 text tokens
a, b = MMDiTAttention(64)(torch.randn(2, 64, 64), torch.randn(2, 77, 64))
print(a.shape, b.shape)   ## torch.Size([2, 64, 64]) torch.Size([2, 77, 64])
```
````

```{attention} Q&A
:class: dropdown
*Why separate weights, ❌1 shared transformer?*
- Text & image embeddings have different statistics → separate weights fit each modality.
- Joint attention still mixes information → bidirectional.

*What else did SD3 show?*
- Validation loss decreases predictably w/ model size & tracks human-preference metrics.
```

&nbsp;

### ControlNet
- **What**: Trainable copy of a frozen diffusion model's encoder, joined by zero-initialized convs, that adds spatial conditions (edges, depth, pose). {cite:p}`zhang2023adding`
- **Why**: Text cannot specify exact layout.
    - Fine-tuning the whole model on small paired data → overfitting & forgetting.
- **How**:
    1. Freeze the base U-Net.
    2. Clone its encoder & middle block → trainable copy. Input = $x_t$ + encoded condition image.
    3. Copy's outputs → **zero convolutions** (1×1, weights & bias = 0) → added to the base's skip connections & middle block.
    4. At init: zero convs output 0 → model = original exactly. Training grows the control gradually.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\Theta$: Frozen block params.
    - $\Theta_c$: Trainable copy params.
    - $\Theta_{z1},\Theta_{z2}$: Zero-conv params.
- Misc:
    - $\mathcal{F}(\cdot;\Theta)$: Network block.
    - $\mathcal{Z}(\cdot;\cdot)$: Zero convolution.

Forward:

$$
y_c=\mathcal{F}(x;\Theta)+\mathcal{Z}\left(\mathcal{F}\left(x+\mathcal{Z}(c;\Theta_{z1});\Theta_c\right);\Theta_{z2}\right)
$$

- Zero init ≠ zero gradient: $\partial\mathcal{Z}/\partial W$ = its input ≠ 0 → weights move after step 1.
```

````{important} Code
:class: dropdown
```python
import copy
import torch
import torch.nn as nn

class ControlNetBlock(nn.Module):
    def __init__(self, block, ch):
        super().__init__()
        self.base = block.requires_grad_(False)                   ## frozen original
        self.copy = copy.deepcopy(block).requires_grad_(True)     ## trainable clone, same init
        self.z_in, self.z_out = nn.Conv2d(ch, ch, 1), nn.Conv2d(ch, ch, 1)
        for z in (self.z_in, self.z_out):
            nn.init.zeros_(z.weight)                              ## zero conv -> starts as a no-op
            nn.init.zeros_(z.bias)

    def forward(self, x, c):                                      ## c: condition features, same shape as x
        return self.base(x) + self.z_out(self.copy(x + self.z_in(c)))

## Example: at init, output == frozen block's output exactly
blk = nn.Conv2d(8, 8, 3, padding=1)
cn = ControlNetBlock(blk, 8)
x, c = torch.randn(2, 8, 16, 16), torch.randn(2, 8, 16, 16)
print(torch.allclose(cn(x, c), blk(x)))   ## True
```
````

```{attention} Q&A
:class: dropdown
*Why zero convs, ❌random init?*
- Random init → random features injected into a pretrained net → damages it at step 0.
- Zero init → exact original model at start → training only adds control.

*Why a full copy, ❌a small adapter?*
- The copy reuses the pretrained encoder's deep features → robust w/ small (<50k) & large (>1M) datasets.

*Cons?*
- Extra encoder copy → more memory & compute per step.
- 1 ControlNet per condition type.
```

&nbsp;

## Flow Matching
- **What**: Regress a velocity field onto the velocity of a chosen data ↔ noise interpolation; sample by solving the ODE. {cite:p}`lipman2023flow`
- **Why**: Diffusion's probability-flow ODE has curved paths → many solver steps. SDE theory → extra machinery.
    - Choose the path directly (e.g., a straight line) → regress its velocity → simulation-free training, like diffusion.
- **How**:
    1. Sample $x_0\sim p_\text{data}$, $\epsilon\sim\mathcal{N}(0,I)$, $t\sim\mathcal{U}[0,1]$.
    2. Interpolate: $x_t=(1-t)x_0+t\epsilon$.
    3. Regress $v_\theta(x_t,t)$ onto $\epsilon-x_0$ w/ MSE.
    4. Sample: start at $\epsilon$ → integrate $dx/dt=v_\theta$ from $t=1$ to $0$ (Euler, Heun).

```{note} Math
:class: dropdown
Notations:
- Params:
    - $v_\theta(x,t)$: Velocity model.
- Misc:
    - $v^*(x,t)$: Marginal (true) velocity field.
- Override: continuous $t\in[0,1]$, $t=0$ data, $t=1$ noise (SD3 convention). Lipman et al. reverse time: $t=0$ noise, $t=1$ data.

Training:

$$
\mathcal{L}_\text{CFM}=\mathbb{E}_{t,x_0,\epsilon}\left[\left\|v_\theta(x_t,t)-(\epsilon-x_0)\right\|^2\right],\quad x_t=(1-t)x_0+t\epsilon
$$

- Optimum: $v^*(x,t)=\mathbb{E}[\epsilon-x_0\,|\,x_t=x]$.

Inference:

$$
x_{t-\Delta t}=x_t-\Delta t\;v_\theta(x_t,t)
$$

Relation to diffusion:

$$
\hat x_0=x_t-t\,\hat v,\qquad\hat\epsilon=x_t+(1-t)\,\hat v
$$

- Linear path = Gaussian path w/ signal scale $1-t$ & noise scale $t$. ❌Variance preserving.
- Lipman's OT path: mean $tx_1$, std $1-(1-\sigma_\text{min})t$. $\sigma_\text{min}\to0$ → same straight line.
```

```{tip} Derivation
:class: dropdown
*Why regress the conditional velocity $\epsilon-x_0$, ❌the unknown marginal $v^*$?*
1. Target objective (FM): $\mathbb{E}_{t,x_t}\|v_\theta(x_t,t)-v^*(x_t,t)\|^2$. $v^*$ unknown.
2. Expand both FM & CFM: $\|v_\theta\|^2-2\langle v_\theta,\cdot\rangle+\|\cdot\|^2$.
3. $\|v_\theta\|^2$ terms match.
4. Cross terms match by the tower property: $\mathbb{E}[\langle v_\theta(x_t),\epsilon-x_0\rangle]=\mathbb{E}[\langle v_\theta(x_t),\mathbb{E}[\epsilon-x_0|x_t]\rangle]=\mathbb{E}[\langle v_\theta(x_t),v^*(x_t)\rangle]$.
5. Last terms do not depend on $\theta$ → $\mathcal{L}_\text{CFM}=\mathcal{L}_\text{FM}+C$ → same gradients.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FlowMatching(nn.Module):
    def __init__(self, v_model):
        super().__init__()
        self.v = v_model

    def loss(self, x0, eps=None):
        eps = torch.randn_like(x0) if eps is None else eps      ## None -> independent coupling
        t = torch.rand(x0.shape[0], *[1] * (x0.dim() - 1))      ## t ~ U[0,1], broadcastable
        xt = (1 - t) * x0 + t * eps                             ## straight line data -> noise
        return F.mse_loss(self.v(xt, t.flatten()), eps - x0)    ## constant velocity along the line

    @torch.no_grad()
    def sample(self, eps, n_steps=10):
        x, ts = eps, torch.linspace(1, 0, n_steps + 1)          ## t = 1: pure noise
        for t, t_next in zip(ts[:-1], ts[1:]):
            x = x + (t_next - t) * self.v(x, t.expand(x.shape[0]))   ## Euler, dt < 0
        return x

## Example: data = point mass at 3 -> exact v*(x, t) = (x - 3) / t
fm = FlowMatching(lambda x, t: (x - 3) / t.view(-1, 1))
print(fm.loss(torch.full((4, 2), 3.0)).item())        ## ≈ 0 (float error only): v* is exact here
print(fm.sample(torch.randn(4, 2), n_steps=1))        ## all 3.0: straight paths -> 1 Euler step is exact
```
````

```{attention} Q&A
:class: dropdown
*Flow matching vs diffusion?*
- Gaussian-path FM covers diffusion paths as special cases. Same family.
- Differences = path (straight vs VP curve), time parametrization, loss weight.

*Is the learned ODE straight?*
- ❌ in general. Each conditional path is straight. The marginal $v^*$ averages crossing lines → curved.
- → [Rectified Flow](#rectified-flow) reflow straightens it.

*Why is it now the default for large image models?*
- Simple objective, few assumptions.
- SD3: rectified flow + logit-normal $t$ sampling (more weight on mid $t$) beat established diffusion formulations at scale {cite:p}`esser2024scaling`.

*Does the straight conditional path give the optimal-transport map?*
- ❌. Each pair moves along an OT displacement. The marginal flow under independent pairing ≠ the OT map.
```

&nbsp;

### Rectified Flow
- **What**: Flow matching w/ straight-line paths + **reflow**: retrain on the model's own (noise, sample) pairs to straighten paths. {cite:p}`liu2023flow`
- **Why**: Independent (random) pairing → paths cross → marginal flow curved → many steps.
- **How**:
    1. Train 1-rectified flow: FM w/ linear paths on random pairs $(x_0,\epsilon)$.
    2. **Reflow**: sample $\epsilon$ → solve the ODE → $\hat x_0$. New coupling $(\hat x_0,\epsilon)$.
    3. Retrain FM on the new pairs → paths ❌cross → straighter → 1–few steps.
    4. Optional: distill to 1 step.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\hat x_0=\text{ODE}_{v_\theta}(\epsilon;\,1\to0)$: Sample from the current flow.

Process:
1. $v^{(k+1)}=\arg\min_v\mathbb{E}_{t,(\hat x_0,\epsilon)}\left[\|v(x_t,t)-(\epsilon-\hat x_0)\|^2\right]$, $x_t=(1-t)\hat x_0+t\epsilon$, pairs from $v^{(k)}$.
- Each rectification preserves the marginals.
- Each rectification does not increase any convex transport cost.
- Fully straight flow → 1 Euler step is exact.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

## Reuses FlowMatching from above
class Reflow:
    def __init__(self, teacher, student):              ## both FlowMatching
        self.teacher, self.student = teacher, student

    def loss(self, n, shape, n_steps=50):
        eps = torch.randn(n, *shape)
        x0_hat = self.teacher.sample(eps, n_steps)     ## noise -> sample via teacher ODE
        return self.student.loss(x0_hat, eps)          ## same FM loss on coupled pairs -> no crossing

## Example
teacher = FlowMatching(lambda x, t: (x - 3) / t.view(-1, 1))
net = nn.Linear(3, 2)
student = FlowMatching(lambda x, t: net(torch.cat([x, t.view(-1, 1)], dim=-1)))
print(Reflow(teacher, student).loss(8, (2,)).item() >= 0)   ## True
```
````

```{attention} Q&A
:class: dropdown
*Why does straight mean fast?*
- Straight path at constant speed → Euler is exact → 1 step.

*Cons?*
- Reflow needs many ODE solves to build pairs + retraining.
- Each reflow inherits the previous model's sample errors.

*Is SD3 a reflowed model?*
- ❌. SD3 uses the rectified-flow (linear-path FM) objective w/o reflow.
```

&nbsp;
