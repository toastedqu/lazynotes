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
- **What**:
    - **Forward**: Data → Noise.
        - Thermodynamic Diffusion: Particles gradually spread from high to low concentration.
    - **Reverse**: Noise → Data.
        - Denoising: Gradually denoise pure noise into structured data. (impossible in Physics, but possible in ML)
- **Why**:
    - Stable training ← No adversarial training (> GAN)
    - Full data distribution converage (> VAE)
    - Fine-grained control over iterative generation process.
- **How**: 

```{note} Math
:class: dropdown
Notations:
- IO:
    - $x_0$: The original, clean data at step $0$.
    - $x_t$: The noisy data at step $t$.
    - $x_T$: The noisiest data at step $T$ (i.e., pure noise).
- Hyperparams:
    - $T$: #steps.
- Misc:
    - Noise:
        - $\beta_t$: Variance schedule at step $t$.
            - How much noise is added at each step.
        - $\epsilon\sim N(0,I)$: Standard Gaussian noise.
        - $\alpha_t=1-\beta_t$: Complement of $\beta_t$. 
            - *But why? Why not just use $\beta_t$ alone?*
                - $\beta_t$: **Noise rate**.
                - $\alpha_t$: **Signal retention rate**.
                    - $\alpha_t=1\rightarrow$ Keep all previous signal.
                    - $\alpha_t=0\rightarrow$ Keep no previous signal.
                - This intuition will show its elegance in the subsequent formulas.
        - $\bar{\alpha}_t=\prod_{i=1}^{t}\alpha_i$: Total signal retention rate.
            - How much original signal still remains after $t$ steps.
    - Distribution:
        - $q(\cdot)$: Forward diffusion process.
        - $p(\cdot)$: Reverse diffusion process.
        - $p_\theta(\cdot)$: Learned reverse diffusion process, parametrized by $\theta$.
    - Model:
        - $\epsilon_\theta(\cdot)$: Noise prediction model.

Forward:
- Original Process:

    $$
    q(x_t|x_{t-1})=N(x_t;\sqrt{\alpha_t}x_{t-1},\beta_tI)
    $$

    - Given $x_{t-1}$, the next step $x_t$
        1. Shrinks the signal toward 0 ← $\times\sqrt{\alpha_t}$
        2. Adds Gaussian noise ← $+\sqrt{\beta_t}\epsilon$

- Direct Sampling:

    $$
    x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon
    $$

    - We can directly compute $x_t$ from $x_0$ (significantly faster).
        - $\sqrt{\bar{\alpha}_t}$: How much original signal to keep.
        - $\sqrt{1-\bar{\alpha}_t}$: How much noise to add.


<!-- Reverse:

$$
\hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\hat{\epsilon}}{\sqrt{\bar{\alpha}_t}}
$$ -->
```