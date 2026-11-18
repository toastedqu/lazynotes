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
# Generative Models
Map of deep generative model families: what each one models, how it trains, what it trades away. Each family has its own page.

Notations:
- $x\in\mathbb{R}^d$: Data sample
- $d$: Data dim
- $p_\text{data}$: Data distribution
- $p_\theta$: Model distribution
- $\theta$: Model params
- $z$: Latent variable
- $p(z)$: Latent prior, usually $\mathcal{N}(0,I)$
- $c$: Condition (class label, text, image)

Overrides of global notation: $x$, $z$ = whole vectors (non-bold). $x_j$ = $j$-th dim of $x$.

&nbsp;

## Generative Model
- **What**: Model of the data distribution $p(x)$ or $p(x|c)$: a density, a sampler, or both.
- **Why**: Create new data & reason about data, w/o labels.
    - Sample: images, text, audio, molecules.
    - Score: $p_\theta(x)$ → compression, anomaly detection.
    - Complete: fill in missing parts (inpainting, continuation).
    - Represent: latent codes → features for downstream tasks.
- **How**:
    1. Pick a family → decides how $p_\theta$ is represented ([Explicit](#explicit-density) vs [Implicit](#implicit-density)).
    2. Fit $p_\theta\approx p_\text{data}$: minimize a divergence that the family can estimate.
        - KL ⇔ MLE (explicit density), adversarial JSD / Wasserstein (GAN), Fisher divergence (score matching).
    3. Sample: 1 pass (VAE, GAN, flow), $d$ sequential steps (autoregressive), or iterative refinement (diffusion, MCMC).
    - Conditional $p(x|c)$: same families, w/ $c$ fed to the network. Contrast w/ [discriminative](../../ml/misc.md#discriminative-vs-generative) $p(y|x)$.

```{note} Math
:class: dropdown
Objective:

$$
\theta^*=\arg\min_\theta D\left(p_\text{data}\,\|\,p_\theta\right)
$$

- $D$: Divergence. ✅ $D=0\Leftrightarrow p_\theta=p_\text{data}$.

MLE (explicit density):

$$
\theta^*=\arg\max_\theta\,\mathbb{E}_{x\sim p_\text{data}}\left[\log p_\theta(x)\right]=\arg\min_\theta D_\text{KL}\left(p_\text{data}\,\|\,p_\theta\right)
$$
```

```{tip} Derivation
:class: dropdown
*Why is MLE = forward KL minimization?*
1. $D_\text{KL}(p_\text{data}\|p_\theta)=\mathbb{E}_{p_\text{data}}[\log p_\text{data}(x)]-\mathbb{E}_{p_\text{data}}[\log p_\theta(x)]$.
2. 1st term = $-H(p_\text{data})$ → independent of $\theta$.
3. → $\min_\theta D_\text{KL}\Leftrightarrow\max_\theta\mathbb{E}_{p_\text{data}}[\log p_\theta(x)]$.
4. In practice: replace $p_\text{data}$ by the training set → average log-likelihood.
```

```{dropdown} Table: Families
| Family | Likelihood | Sampling speed | Sample quality | Mode coverage | Latent space | Training stability |
|:--|:--|:--|:--|:--|:--|:--|
| [Autoregressive](../../nlp/models.md#gpt) | ✅ Exact | Slow: $d$ sequential steps | High. SOTA for text | ✅ ← MLE | ❌ | ✅ Stable (CE) |
| [VAE](vae.md#vae) | Lower bound (ELBO) | Fast: 1 pass | Low: blurry | ✅ ← MLE-like | ✅ Low-dim + encoder | ✅ Stable. Risk: posterior collapse |
| [GAN](gan.md#gan) | ❌ | Fast: 1 pass | High: sharp | ❌ Mode collapse | ✅ Low-dim, ❌encoder | ❌ Unstable ← minimax |
| [Flow](flow.md#normalizing-flow) | ✅ Exact | Fast: 1 pass (coupling flows) | Medium | ✅ ← MLE | ✅ Same dim as $x$, bijective | ✅ Stable |
| [EBM](ebm.md#ebm) | Unnormalized ($Z_\theta$ intractable) | Slow: MCMC | Medium | Medium ← MCMC mixes poorly | ❌ | ❌ Unstable ← MCMC-based grads |
| [Diffusion](diffusion.md#diffusion-model) | Bound (ELBO), or exact via ODE | Slow: 10–1000 passes | High. SOTA for images | ✅ ← ELBO | Same dim as $x$, ❌compact | ✅ Stable (regression) |

- Qualitative, typical behavior. Individual models can beat their family's row.
- Flow sampling speed: coupling flows (RealNVP, Glow) = 1 pass. Autoregressive flows (MAF) = $d$ sequential steps.
```

```{attention} Q&A
:class: dropdown
*What is the generative learning trilemma?*
- 3 goals: high sample quality, mode coverage, fast sampling. Most families get only 2 {cite:p}`xiao2022tackling`.
    - GAN: quality + speed, ❌coverage.
    - VAE, flow: speed + coverage, ❌quality.
    - Diffusion: quality + coverage, ❌speed.
- Fixes attack the missing corner: diffusion distillation → speed. Their fix (DDGAN): GAN models each large denoising step.

*Does high likelihood mean good samples?*
- ❌ Not in high dim. The 2 are largely independent {cite:p}`theis2016note`.
- Mix a good model at 1% w/ a junk model at 99%: $\log p$ drops by ≤ $\log100\approx4.6$ nats per sample. Per dim → negligible. 99% of samples = junk.
- Reverse also holds: a model that memorizes training images gives great samples & bad test likelihood.

*Why do MLE models blur & GANs drop modes?*
- Unlimited capacity: both reach $p_\theta=p_\text{data}$ → no blur, no dropped modes. The difference appears only w/ a limited model family.
- Forward KL (MLE) + limited family: $p_\theta\approx0$ where $p_\text{data}>0$ → huge loss → covers every mode → mass between modes → blur.
    - E.g., Gaussian/MSE decoder → conditional mean of several plausible outputs.
- GAN: minimax JSD is symmetric → ❌inherently mode-seeking. Mode collapse ← optimization dynamics & limited capacity.
    - Non-saturating generator loss at the optimal critic: grad of $D_\text{KL}(p_\theta\|p_\text{data})-2\,\text{JSD}$ → reverse-KL-like → mode-seeking {cite:p}`arjovsky2017towards`.

*How do you evaluate a generative model?*
- Explicit: test NLL (bits/dim for images, perplexity for text).
- All: sample-based metrics (FID, precision/recall). Compare feature statistics of real vs generated sets.
- Use both. They measure different things (see above).

*Is an LLM a generative model?*
- ✅ It models $p(x)=\prod_tp(x_t|x_{<t})$ → autoregressive family. Prompt = $c$ → $p(x|c)$.

*What does this tell us about intelligence/AC?*
- Predictive processing: the brain = a generative model of its senses. Perception = inference of latent causes → [PP](../../ac/theories.md#predictive-processing-pp).
- VAE-style inference (encoder → latent → reconstruction) is the closest ML analog.
```

&nbsp;

### Explicit Density
- **What**: Model w/ an explicit formula for $p_\theta(x)$. The formula can be intractable or unnormalized.
- **Why**: A density → direct MLE, likelihood scores, principled comparison of models.
- **How**: Split by what you can compute.
    - **Tractable**: exact $\log p_\theta(x)$ → direct MLE.
        - [Autoregressive](../../nlp/models.md#gpt): chain-rule factorization over dims.
        - [Normalizing flow](flow.md#normalizing-flow): invertible map from noise + change of variables.
    - **Approximate**: $p_\theta(x)$ defined, but $\log p_\theta(x)$ too expensive → optimize a surrogate.
        - **Variational**: maximize a lower bound (ELBO). [VAE](vae.md#vae), [diffusion](diffusion.md#diffusion-model) (DDPM view).
        - **Unnormalized**: $p_\theta\propto e^{-E_\theta}$, normalizer $Z_\theta$ intractable → MCMC grads, score matching, or NCE. [EBM](ebm.md#ebm).
    - Diffusion fits more than 1 bucket: ELBO (variational), score $\nabla_x\log p$ (unnormalized, $Z$ cancels), exact $\log p$ via the probability-flow ODE (tractable, costly).

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $f_\theta$: Invertible map $z\mapsto x$, $\dim z=d$.
    - $q_\phi(z|x)$: Approx posterior (encoder).
    - $E_\theta(x)$: Energy.

Autoregressive (tractable):

$$
\log p_\theta(x)=\sum_{j=1}^{d}\log p_\theta(x_j|x_{<j})
$$

Flow (tractable):

$$
\log p_\theta(x)=\log p(z)-\log\left|\det\frac{\partial f_\theta(z)}{\partial z}\right|,\quad z=f_\theta^{-1}(x)
$$

Latent variable + variational bound (approximate):

$$
\log p_\theta(x)=\log\int p_\theta(x|z)\,p(z)\,dz\;\geq\;\mathbb{E}_{q_\phi(z|x)}\left[\log p_\theta(x|z)\right]-D_\text{KL}\left(q_\phi(z|x)\,\|\,p(z)\right)
$$

Energy-based (approximate):

$$
p_\theta(x)=\frac{e^{-E_\theta(x)}}{Z_\theta},\quad Z_\theta=\int e^{-E_\theta(x)}\,dx
$$

- $\nabla_x\log p_\theta(x)=-\nabla_xE_\theta(x)$: score needs no $Z_\theta$.
```

```{attention} Q&A
:class: dropdown
*Why not always use a tractable family?*
- Tractability constrains architecture.
    - Autoregressive: fixed order + $d$ sequential sampling steps.
    - Flow: invertible layers, $\dim z=d$, cheap Jacobian determinant → ⬇️expressivity per param.
- Approximate families buy flexibility w/ a looser objective.

*What breaks if you report likelihood of a continuous model on discrete data (8-bit pixels)?*
- Density can grow w/o bound on discrete points → $\log p\to\infty$ → meaningless.
- Fix: dequantize (add uniform noise in each pixel's bin) → $\log p$ = bound on the discrete likelihood.
- Report bits/dim $=-\log_2p_\theta(x)/d$ → comparable across resolutions.

*Is an ELBO comparable to an exact likelihood?*
- ❌ Directly. ELBO ≤ $\log p$ → a VAE's ELBO understates its true likelihood.
- Tighter estimate: importance sampling w/ many $z$ samples.

*Why is an unnormalized density still useful?*
- Ratios $p_\theta(x)/p_\theta(x')=e^{E_\theta(x')-E_\theta(x)}$ & the score need no $Z_\theta$.
- → Enough for MCMC sampling (Langevin), ranking, & [score matching](diffusion.md#score-matching).
```

&nbsp;

### Implicit Density
- **What**: Model that defines only a sampler $x=G_\theta(z)$, ❌formula for $p_\theta(x)$.
- **Why**: Density constraints (normalization, invertibility, tractable bound) restrict architecture.
    - Drop them → any network $G_\theta$ works → sharp, fast 1-pass samples.
- **How**:
    1. Sample $z\sim p(z)$ → $x=G_\theta(z)$.
    2. Train w/ a divergence estimated from samples only: adversarial critic ([GAN](gan.md#gan)), MMD.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $G_\theta$: Generator $\mathbb{R}^{d_z}\to\mathbb{R}^d$.
- Misc:
    - $d_z$: Latent dim.

Model:

$$
x=G_\theta(z),\quad z\sim p(z)\quad\Leftrightarrow\quad p_\theta=G_{\theta\#}\,p(z)
$$

- $G_{\theta\#}p(z)$: Pushforward of $p(z)$ through $G_\theta$.
- $d_z<d$ → samples lie on a ≤$d_z$-dim manifold in $\mathbb{R}^d$ → $p_\theta$ has no density on $\mathbb{R}^d$.
```

```{attention} Q&A
:class: dropdown
*Why can't a GAN just compute $p_\theta(x)$?*
- Needs change of variables → $G_\theta$ invertible & $d_z=d$. Typical GAN: $d_z\ll d$, ❌invertible.
- $d_z<d$ → $p_\theta$ = measure on a thin manifold → $p_\theta(x)$ undefined → ❌MLE.
    - $D_\text{KL}(p_\text{data}\|p_\theta)=\infty$ unless $p_\text{data}$ lies on that exact manifold.

*Why does that matter for training?*
- $p_\text{data}$ & $p_\theta$ on 2 low-dim manifolds → supports often do not overlap.
- → JSD saturates at $\log2$ → ⬇️useful grads → unstable GANs.
- Fix: a divergence that stays smooth w/o overlap (Wasserstein) → [WGAN](gan.md#wgan).

*Pros?*
- Fast 1-pass sampling. Sharp samples. Free choice of architecture.

*Cons?*
- ❌Likelihood → no NLL evaluation, no density-based anomaly score.
- ❌Encoder → no inference $x\to z$ (needs extra nets, e.g., GAN inversion).
- Mode collapse ← nothing forces coverage of every mode.

*Is a diffusion model implicit?*
- ❌ It defines $p_\theta(x_0)$ via a latent chain → explicit (ELBO).
- Its deterministic sampler (probability-flow ODE, DDIM) = a pushforward of noise like $G_\theta$.
    - Difference: the ODE map is invertible & $\dim x_T=d$ → exact density via instantaneous change of variables.
```

&nbsp;
