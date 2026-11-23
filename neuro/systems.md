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
# Systems
Computational models of perception, action & cognition.

Notations:
- $t$: Time
- $\mathbf{s}$: Stimulus (e.g., image patch as a vector)
- $r$: Response (firing rate)

&nbsp;

## Perception

### V1 Simple & Complex Cells
- **What**: V1 neurons selective for oriented edges.
    - **Simple**: ≈ linear filter w/ separate on / off subregions.
    - **Complex**: same orientation selectivity, but largely insensitive to phase & exact position within the RF.
- **Why**: Oriented edges = basic parts of shapes. Complex cells add local position invariance.
- **How**:
    1. Simple cell ≈ Gabor filter → rectification.
    2. Complex cell ≈ pools simple cells w/ the same orientation & different phases → **energy model**.
    3. Filter (simple) + pool (complex), repeated → Neocognitron → CNN ([Convolution](../cv/cnn.md#convolution), [Pooling](../cv/cnn.md#pooling)).

```{note} Math
:class: dropdown
Energy model {cite:p}`adelson1985spatiotemporal`.

Notations:
- Params:
    - $\mathbf{k}_0,\mathbf{k}_{90}$: Gabor filters, same orientation & frequency, phases 0° & 90° (quadrature pair).

Model:

$$
r_\text{simple}=\left[\mathbf{k}_0^T\mathbf{s}\right]_+,\qquad r_\text{complex}=\left(\mathbf{k}_0^T\mathbf{s}\right)^2+\left(\mathbf{k}_{90}^T\mathbf{s}\right)^2
$$

- $[\cdot]_+$: Rectification.
- Grating at the preferred orientation & frequency, phase $\psi$: $\mathbf{k}_0^T\mathbf{s}\approx c\cos\psi$, $\mathbf{k}_{90}^T\mathbf{s}\approx-c\sin\psi$ → $r_\text{complex}\approx c^2$, independent of $\psi$.
- $c$: Response amplitude (∝ contrast).
```

````{important} Code
:class: dropdown
```python
import torch

class ComplexCell:
    def __init__(self, size=16, freq=0.25, theta=0.0, sigma=3.0):
        u = torch.arange(size) - size / 2
        U1, U2 = torch.meshgrid(u, u, indexing="ij")
        u_rot = U1 * torch.cos(torch.tensor(theta)) + U2 * torch.sin(torch.tensor(theta))
        env = torch.exp(-(U1 ** 2 + U2 ** 2) / (2 * sigma ** 2))
        ## quadrature pair: same orientation & frequency, 90° apart in phase
        self.k0 = env * torch.cos(2 * torch.pi * freq * u_rot)
        self.k90 = env * torch.sin(2 * torch.pi * freq * u_rot)

    def __call__(self, img):
        ## simple cells = linear filters; complex cell = sum of their squares → phase invariant
        return (self.k0 * img).sum() ** 2 + (self.k90 * img).sum() ** 2

## Example: preferred grating at 4 phases
cell = ComplexCell()
u = torch.arange(16) - 8.0
U1, _ = torch.meshgrid(u, u, indexing="ij")
for psi in [0.0, 0.5 * torch.pi, torch.pi, 1.5 * torch.pi]:
    grating = torch.cos(2 * torch.pi * 0.25 * U1 + psi)
    print(round((cell.k0 * grating).sum().item(), 1), round(cell(grating).item(), 1))
## 1st col = simple-cell drive k0·s before rectification: ≈ 27.8, 0, -27.8, 0 → depends on phase
## complex (2nd col): ≈ 771, 776, 771, 776 → ≈ constant
```
````

```{attention} Q&A
:class: dropdown
*Where do Gabor-like filters come from?*

- Learned from natural images by sparse coding or ICA → [Sparse Code](coding.md#sparse-code).
- First-layer filters of trained CNNs look similar.

*2 discrete classes?*

- Roughly. Response modulation by grating phase is bimodal-ish, w/ intermediates → closer to a continuum.

*CNN link?*

- Fukushima's Neocognitron (1980) copied S-cells (simple) & C-cells (complex) → conv + pooling.
```

&nbsp;

## Action

### Internal Model
- **What**: Neural simulation of body & world: **forward model** (command → predicted outcome) & **inverse model** (desired outcome → command). {cite:p}`wolpert1995internal`
- **Why**: Sensory feedback is delayed (tens to 100s of ms, by pathway) & noisy → the controller must predict the consequences of its own commands.
- **How**:
    1. Copy of each motor command (**efference copy**) → forward model → predicted sensory outcome.
    2. Compare predicted vs actual feedback → prediction error → update state estimate & model.
    3. Weight prediction vs feedback by their reliability → Kalman-like state estimate.
    4. Subtract predicted self-generated sensations → self-produced touch feels weaker (hard to tickle yourself; delaying the touch restores ticklishness).

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{u}_t$: Motor command.
    - $\mathbf{o}_t$: Sensory observation.
- Params:
    - $f$: Forward dynamics model.
    - $g$: Observation model.
- Misc:
    - $\hat{\mathbf{z}}_t$: Estimated body / world state.
    - $K$: Gain on the prediction error.

State estimate (observer):

$$
\hat{\mathbf{z}}_{t+1}=\underbrace{f(\hat{\mathbf{z}}_t,\mathbf{u}_t)}_\text{prediction}+K\left(\mathbf{o}_{t+1}-g\left(f(\hat{\mathbf{z}}_t,\mathbf{u}_t)\right)\right)
$$

- Linear-Gaussian case, optimal $K$ → [Kalman Filter](../math/prob/process.md#kalman-filter).
```

```{attention} Q&A
:class: dropdown
*Evidence?*

- Human estimates of unseen hand position: errors match a Kalman-style observer w/ a forward model.

*Where?*

- Cerebellum (forward models), parietal cortex (state estimation).

*AC angle?*

- Sense of agency ↔ predicted & actual consequences match.
- Hypothesis: faulty efference copy → own actions feel externally caused (proposed for some schizophrenia symptoms). Contested.

*ML link?*

- World models in [Model-Based RL](../rl/model.md#model-based-rl).
```

&nbsp;

## Cognition

### Attention
- **What**: Selective prioritization of some inputs or tasks over others under limited capacity.
- **Why**: Cortex cannot fully process everything at once → stimuli compete for representation.
- **How**:
    - **Biased competition**: stimuli in 1 RF suppress each other; top-down signals bias the competition toward the attended one. {cite:p}`desimone1995neural`
    - **Normalization model**: attention multiplies the stimulus drive *before* divisive normalization → [Divisive Normalization](coding.md#divisive-normalization).
    - Neural effects: firing ⬆️ (gain), variability & noise correlations ⬇️.
    - Bias source: a **priority map** = salience (bottom-up) + goal relevance (top-down).
    - **Bottom-up** (salience-driven) vs **top-down** (goal-driven). **Overt** (eye movement) vs **covert** (no eye movement).

```{note} Math
:class: dropdown
Normalization model of attention {cite:p}`reynolds2009normalization`.

Notations:
- IO:
    - $E_a$: Stimulus drive to neuron $a$.
    - $R_a$: Response of neuron $a$.
- Params:
    - $A_a$: Attention gain ($>1$ attended, $=1$ otherwise).
    - $w_{ab}$: Suppressive pool weights.
- Hyperparams:
    - $\sigma$: Semi-saturation constant.

Model:

$$
R_a=\frac{A_aE_a}{\sigma+\sum_bw_{ab}A_bE_b}
$$

Properties:
- Attention field large vs stimulus → scales drive & suppression together → contrast-gain change.
- Attention field small vs stimulus → scales mainly the drive → response-gain change.
- These are limiting regimes; intermediate sizes mix both.
```

````{important} Code
:class: dropdown
```python
## Reuses DivisiveNormalization from neuro/coding.md
import torch

class NormalizationAttention:
    def __init__(self, W, sigma=1.0):
        self.dn = DivisiveNormalization(W, sigma=sigma, q=1.0)

    def __call__(self, E, A):
        ## attention scales the drive BEFORE normalization → biased competition
        return self.dn(A * E)

## Example: 2 stimuli share 1 suppressive pool; attend stimulus 0
att = NormalizationAttention(torch.ones(2, 2))
E = torch.tensor([4.0, 4.0])
print(att(E, torch.tensor([1.0, 1.0])))  ## ≈ [0.444, 0.444] = 4/9 each: no attention
print(att(E, torch.tensor([2.0, 1.0])))  ## ≈ [0.615, 0.308] = 8/13, 4/13: attended ⬆️, competitor ⬇️
```
````

```{attention} Q&A
:class: dropdown
*Brain attention vs transformer attention?*

- Transformer: content-based weighted average of values for every token → routing, no capacity limit per se → [Scaled Dot-Product Attention](../nlp/attn.md#scaled-dot-product-attention).
- Brain: gain modulation & selection under a capacity limit.
- Shared idea: input-dependent weighting + normalization (softmax ≈ divisive normalization).

*Attention = consciousness?*

- Dissociable in experiments: attention can be drawn by unseen cues; some gist is seen w/o focused attention. Extent contested.
- Attention schema theory links the 2 → [AST](../ac/theories.md#attention-schema-theory-ast).
```

&nbsp;

### Working Memory
- **What**: Active maintenance & manipulation of a few items over seconds, w/o the stimulus present.
- **Why**: Reasoning, planning & language need information held "online" across time gaps.
- **How**: Candidate mechanisms.
    1. **Persistent activity**: prefrontal & parietal neurons keep firing during the delay ← recurrent excitation → [Attractor Network](network.md#attractor-network).
    2. **Activity-silent**: short-term synaptic facilitation holds the item; brief bursts refresh it. {cite:p}`mongillo2008synaptic`
    3. **Dynamic coding**: population activity changes during the delay, but the item stays decodable from a stable subspace.

```{attention} Q&A
:class: dropdown
*Capacity?*

- Classic "7 ± 2" → ≈ 3–5 chunks once rehearsal & chunking are controlled. {cite:p}`cowan2001magical`

*Why limited?*

- Hypotheses: interference between attractors, normalization, limited oscillatory slots. Open.

*ML analogy?*

- Persistent activity ≈ RNN hidden state. Activity-silent ≈ [fast weights](learning.md#short-term-plasticity) / KV cache.
- Transformer context: no hard ~4-item limit → human limits are not a law of computation.
```

&nbsp;

### Spatial Navigation
- **What**: Neural codes for position & heading (place, grid, head-direction cells) + their update by self-motion (path integration).
- **Why**: Self-localization & route planning need an internal map (Tolman's "cognitive map").
- **How**:
    - **Place cells**: fire at 1 or a few locations → local, sparse position code.
    - **Grid cells**: fire at the vertices of a triangular (hexagonally symmetric) lattice that tiles the environment. {cite:p}`hafting2005microstructure`
        - Modules w/ different spacings → multi-scale periodic position code.
        - Population activity of 1 module lies on a torus → [Toroidal Grid-Cell Manifold](../math/geo/topology.md#toroidal-grid-cell-manifold).
    - **Head-direction cells**: fire for 1 heading → [Ring Attractor](../math/geo/topology.md#ring-attractor).
    - **Path integration**: velocity input shifts the activity bump of a [Continuous Attractor](network.md#continuous-attractor) → position updated w/o landmarks → landmarks correct accumulated drift.

```{note} Math
:class: dropdown
Idealized grid cell (sum of 3 plane waves).

Notations:
- IO:
    - $\mathbf{p}\in\mathbb{R}^2$: Position.
- Params:
    - $\lambda_g$: Grid spacing (distance between neighboring firing fields).
    - $\mathbf{p}_0$: Grid phase (offset).
    - $\theta_0$: Wave-vector orientation (lattice axes are rotated by 30°).

Model:

$$
f(\mathbf{p})\propto\sum_{k=1}^3\cos\left(\boldsymbol\kappa_k^T(\mathbf{p}-\mathbf{p}_0)\right),\quad\boldsymbol\kappa_k=\frac{4\pi}{\sqrt3\,\lambda_g}\begin{bmatrix}\cos(\theta_0+k\pi/3)\\\sin(\theta_0+k\pi/3)\end{bmatrix}
$$

- 3 wave vectors in 60° steps → interference peaks on a triangular lattice w/ spacing $\lambda_g$.
- Rectify / exponentiate $f$ → nonnegative rate.
- 1 module = cells w/ shared $\lambda_g,\theta_0$ & different $\mathbf{p}_0$.
```

```{attention} Q&A
:class: dropdown
*Why a periodic, multi-module code?*

- Several modules w/ suitable (e.g., incommensurate) periods → like a residue number system → representable range can grow exponentially w/ #modules, if phase precision is high enough.

*Where do hexagons come from?*

- Continuous attractor models produce them.
- RNNs trained to path-integrate can develop grid-like units. Whether this emerges robustly depends on design choices → contested.

*Beyond physical space?*

- Human fMRI shows grid-like (6-fold) signals while navigating an abstract 2D feature space → suggests grid-like coding beyond physical space.

*Relation to ML?*

- Transformers & RNNs trained on spatial or relational prediction tasks can develop place- & grid-like units.
- General idea: a structured, reusable code for relations ("cognitive map") → generalization to new environments.
```

&nbsp;

### Decision Making
- **What**: Choose between options by accumulating noisy evidence until a bound is reached.
- **Why**: 1 noisy sample is unreliable → accumulate → accuracy ⬆️, at the cost of time (**speed–accuracy trade-off**).
- **How**: Drift-diffusion model (DDM).
    1. Decision variable starts at 0 (or at a bias).
    2. Each moment adds momentary evidence (drift + noise).
    3. Hits +bound → choice A. Hits −bound → choice B. Time to bound = decision time $T$.
    4. Reaction time $=T+t_\text{nd}$ (non-decision time: encoding & motor execution).
    5. Neural correlate (random-dot motion): MT = momentary evidence; LIP firing rises w/ accumulated evidence & reaches a common level at choice. {cite:p}`gold2007neural`

```{note} Math
:class: dropdown
Notations:
- IO:
    - $X_t$: Decision variable.
    - $T$: Decision time.
- Params:
    - $v$: Drift rate (evidence strength).
- Hyperparams:
    - $B$: Bound (symmetric, $\pm B$).
    - $\sigma$: Noise std.
- Misc:
    - $W_t$: Standard Brownian motion → [Brownian Motion](../math/prob/process.md#brownian-motion).

Model ($X_0=0$):

$$
dX_t=v\,dt+\sigma\,dW_t,\quad\text{stop when }|X_t|\ge B
$$

Properties ($v>0$ → upper bound correct):

$$
P(\text{correct})=\frac{1}{1+e^{-2vB/\sigma^2}},\qquad\mathbb{E}[T]=\frac{B}{v}\tanh\left(\frac{vB}{\sigma^2}\right)
$$

- $B$⬆️ → accuracy ⬆️, time ⬆️.
- 2 alternatives: DDM = continuous limit of the sequential probability ratio test (SPRT) → fewest expected samples for given error rates.
```

```{tip} Derivation
:class: dropdown
*Why accumulate evidence this way?*
1. Samples $e_\tau\sim\mathcal{N}(+\mu,\varsigma^2)$ (A true) or $\mathcal{N}(-\mu,\varsigma^2)$ (B true).
2. Log-likelihood ratio per sample: $\log\frac{\mathcal{N}(e_\tau;\mu,\varsigma^2)}{\mathcal{N}(e_\tau;-\mu,\varsigma^2)}=\frac{(e_\tau+\mu)^2-(e_\tau-\mu)^2}{2\varsigma^2}=\frac{2\mu e_\tau}{\varsigma^2}$.
3. Total log-likelihood ratio = $\frac{2\mu}{\varsigma^2}\sum_\tau e_\tau$ → random walk w/ drift → continuous limit = DDM.
4. Stop when the evidence sum hits $\pm B$ ⟺ log-LR hits $\pm\frac{2\mu B}{\varsigma^2}$ → Wald's SPRT → for 2 simple hypotheses w/ i.i.d. samples: minimal expected #samples among tests w/ no larger error rates (Wald–Wolfowitz).
5. → [Likelihood Ratio Test](../math/prob/stats.md#likelihood-ratio-test) applied sequentially.
```

````{important} Code
:class: dropdown
```python
import torch

class DDM:
    def __init__(self, v, B, sigma=1.0, dt=1e-3):
        self.v, self.B, self.sigma, self.dt = v, B, sigma, dt

    def __call__(self, n_trials, max_steps=100_000):
        x = torch.zeros(n_trials)
        t = torch.zeros(n_trials)
        done = torch.zeros(n_trials, dtype=torch.bool)
        for _ in range(max_steps):
            ## Euler–Maruyama step; trials that already hit a bound stay frozen
            dx = self.v * self.dt + self.sigma * self.dt ** 0.5 * torch.randn(n_trials)
            x = torch.where(done, x, x + dx)
            t = torch.where(done, t, t + self.dt)
            done = done | (x.abs() >= self.B)
            if done.all():
                break
        ## max_steps x dt = 100 s ≫ typical T → unfinished trials negligible here
        return x >= self.B, t   ## (correct choice since v > 0, decision time)

## Example: v = 1, B = 1, sigma = 1
torch.manual_seed(0)
correct, T = DDM(v=1.0, B=1.0)(n_trials=5000)
print(correct.float().mean(), T.mean())  ## ≈ 0.88 & ≈ 0.76 ← continuous-time values 1/(1 + e^-2) & tanh(1); Euler adds small bias
```
````

```{attention} Q&A
:class: dropdown
*Do LIP neurons really ramp?*

- Trial-averaged: yes. Single trials may show discrete steps instead. {cite:p}`latimer2015single`
- Status: contested.

*Speed–accuracy trade-off in the brain?*

- Bound height ↔ caution. Time pressure → lower or collapsing bounds ("urgency" signals).

*Which params can be fit?*

- Scale $X$, $v$, $B$ & $\sigma$ by the same factor → identical choices & times → fix 1 scale (usually $\sigma$) when fitting.
- Fit $v$, $B$, starting bias & $t_\text{nd}$ to choice proportions & reaction-time distributions.

*ML analogy?*

- Sequential testing, early-exit networks, adaptive computation: stop when confident enough.
```

&nbsp;
