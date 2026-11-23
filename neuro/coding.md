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
# Neural Coding
What spikes mean: encoding (stimulus → spikes), decoding (spikes → stimulus), & the principles that shape codes.

Notations:
- $s$: Stimulus (scalar unless stated)
- $r$: Firing rate (Hz)
- $n$: Spike count in a window
- $T$: Counting window
- $f(s)$: Tuning curve = mean rate given $s$
- $N$: #neurons in a population
- $a,b$: Neuron indices

Overrides of global notation: $n$ = spike count, ❌#features.

&nbsp;

## Spikes

### Spike Train
- **What**: Sequence of spike times of 1 neuron.
- **Why**: Spike shape is roughly stereotyped → standard assumption: all information sits in the spike times.
- **How**: Summarize it at the needed resolution.
    - **Spike-count rate**: $n/T$ over 1 trial.
    - **Time-dependent rate** $r(t)$: average over trials in small bins → **PSTH** (peri-stimulus time histogram).
    - **Inter-spike interval (ISI)** distribution: regularity of firing.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $t_k$: $k$-th spike time.
    - $\langle\cdot\rangle$: Average over trials.
    - $K$: #trials.

Definition:

$$
\rho(t)=\sum_k\delta(t-t_k),\quad n=\int_0^T\rho(t)\,dt,\quad r(t)=\langle\rho(t)\rangle
$$

- $\rho(t)$: Neural response function.

PSTH estimate (bin width $\Delta t$):

$$
\hat r(t)=\frac{\#\text{spikes in }[t,t+\Delta t)\text{ summed over }K\text{ trials}}{K\,\Delta t}
$$
```

```{attention} Q&A
:class: dropdown
*Which rate can the brain use?*

- PSTH needs repeated trials. The brain sees each event once.
- → Brain must average over time (slow) or over many neurons (fast) → [Population Code](#population-code).

*Bin size?*

- Small bins → noisy estimate. Large bins → blurred timing.
```

&nbsp;

### Poisson Spike Model
- **What**: Spikes as a Poisson process w/ rate $r(t)$: counts in disjoint time intervals are independent.
- **Why**: Cortical spike trains look irregular → simplest null model.
    - Gives closed-form likelihoods → decoding & information calculations become tractable.
- **How**:
    1. Divide time into small bins $\Delta t$.
    2. Each bin: spike w.p. ≈ $r(t)\Delta t$ (1st order in $\Delta t$), independent of all other bins.
    3. Homogeneous: $r$ constant. Inhomogeneous: $r(t)$ follows the stimulus.

```{note} Math
:class: dropdown
Model (homogeneous):

$$
P(n)=\frac{(rT)^n}{n!}e^{-rT}
$$

Properties:
- $\mathbb{E}[n]=\text{Var}[n]=rT$ → **Fano factor** $F=\frac{\text{Var}[n]}{\mathbb{E}[n]}=1$.
- ISI $\sim\text{Exponential}(r)$ → **coefficient of variation** $\text{CV}=\frac{\sigma_\text{ISI}}{\mu_\text{ISI}}=1$.
- Inhomogeneous: $n\sim\text{Poisson}\left(\int_0^Tr(t)\,dt\right)$.
- General process → [Poisson Process](../math/prob/process.md#poisson-process).
```

````{important} Code
:class: dropdown
```python
import torch

class PoissonSpikes:
    def __init__(self, dt=1e-3):
        self.dt = dt   ## bin width (s); valid while r * dt << 1

    def __call__(self, rate, n_steps, n_trials=1):
        ## rate (Hz): scalar or (n_steps,) → each bin spikes independently w.p. r * dt
        p = (torch.as_tensor(rate, dtype=torch.float32) * self.dt).expand(n_trials, n_steps)
        return torch.bernoulli(p)

## Example: 20 Hz for 1 s, 1000 trials
spikes = PoissonSpikes()(20.0, n_steps=1000, n_trials=1000)
counts = spikes.sum(dim=1)
print(counts.mean(), counts.var() / counts.mean())  ## ≈ 20, ≈ 0.98 (Bernoulli bins: Fano = 1 - r*dt)
```
````

```{attention} Q&A
:class: dropdown
*Is cortex Poisson?*

- Roughly, on count windows of 100s of ms: CV ≈ 1, Fano ≈ 1 or a bit above.
- ❌ at short timescales: refractoriness → more regular than Poisson.
- ❌ for bursting neurons: CV > 1.
- Stimulus onset lowers trial-to-trial variability in many areas.

*Why irregular if neurons are deterministic?*

- 1 neuron driven by the same fluctuating current → reliable spike times.
- Irregularity comes from the network: balanced excitation & inhibition → fluctuation-driven firing → [E/I Balance](network.md#e-i-balance).

*Better models?*

- Renewal processes (refractoriness), [GLM](#glm) (spike history), doubly stochastic models (fluctuating rate).
```

&nbsp;

## Encoding

### Tuning Curve
- **What**: Mean firing rate as a function of a stimulus feature: $f(s)=\mathbb{E}[r\mid s]$.
- **Why**: Simplest description of what a neuron responds to.
- **How**: Vary 1 feature → average rate per value → fit a curve.
    - **Gaussian** (bell): orientation in V1, position for hippocampal place cells.
    - **Cosine**: movement direction in motor cortex.
    - **Sigmoid** (monotonic): contrast, intensity.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $s_\text{pref}$: Preferred stimulus.
    - $r_\text{max}$: Peak rate.
    - $r_0$: Baseline rate.
    - $\sigma_f$: Tuning width.

Gaussian:

$$
f(s)=r_\text{max}\exp\left(-\frac{(s-s_\text{pref})^2}{2\sigma_f^2}\right)
$$

Cosine (angle $s$; rectify at 0 if negative):

$$
f(s)=r_0+(r_\text{max}-r_0)\cos(s-s_\text{pref})
$$
```

```{attention} Q&A
:class: dropdown
*Where does 1 neuron carry the most information?*

- At the flanks, ❌at the peak. Gaussian tuning: constant-variance noise → max at $|s-s_\text{pref}|=\sigma_f$ (max $|f'|$); Poisson noise → max at $\sqrt2\,\sigma_f$ (max $f'^2/f$).
- Peak: small change in $s$ → ~no change in rate → can't discriminate.
- → [Population Fisher Information](#population-fisher-information): $\propto f'(s)^2/f(s)$.

*Is the tuning curve the whole code?*

- ❌. It ignores variability, correlations, timing & dependence on other features.
```

&nbsp;

### Receptive Field
- **What**: Region of sensory space, & the stimulus pattern within it, that changes a neuron's firing.
- **Why**: Locates & characterizes a sensory neuron's input → building block of sensory hierarchies.
- **How**:
    - **Retina / LGN**: center–surround (e.g., on-center, off-surround) ≈ difference of Gaussians → local contrast.
    - **V1 simple cell**: elongated, oriented on/off subregions ≈ Gabor filter → [V1 Simple & Complex Cells](systems.md#v1-simple-complex-cells).
    - **Classical RF**: stimulus there drives the cell. **Extra-classical** (surround): modulates but cannot drive it alone (e.g., surround suppression).
    - RF size ⬆️ along the hierarchy (V1 → IT), like a deep CNN's [Receptive Field](../cv/cnn.md#receptive-field).

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{u}=(u_1,u_2)$: Position in visual space.
    - $I(\mathbf{u})$: Image intensity (relative to mean).
    - $L$: Linear response.
- Params:
    - $\sigma_c,\sigma_s$: Center / surround width ($\sigma_s>\sigma_c$).
    - $A_c,A_s$: Center / surround strength.
    - $\theta$: Carrier wave-vector angle; the preferred bars run perpendicular to it (along $u_2'$).
    - $\kappa$: Preferred spatial frequency.
    - $\psi$: Phase.
    - $\sigma_1,\sigma_2$: Gabor envelope widths.

Linear response:

$$
L=\int k(\mathbf{u})\,I(\mathbf{u})\,d\mathbf{u}
$$

- $k(\mathbf{u})$: Spatial RF filter.

Center–surround (difference of Gaussians):

$$
k(\mathbf{u})=\frac{A_c}{2\pi\sigma_c^2}e^{-\|\mathbf{u}\|^2/2\sigma_c^2}-\frac{A_s}{2\pi\sigma_s^2}e^{-\|\mathbf{u}\|^2/2\sigma_s^2}
$$

Gabor (V1 simple cell):

$$
k(\mathbf{u})=\exp\left(-\frac{u_1'^2}{2\sigma_1^2}-\frac{u_2'^2}{2\sigma_2^2}\right)\cos(\kappa u_1'-\psi)
$$

- $u_1'=u_1\cos\theta+u_2\sin\theta$, $u_2'=-u_1\sin\theta+u_2\cos\theta$: Coordinates rotated by $\theta$.
```

```{attention} Q&A
:class: dropdown
*Neuro RF vs CNN RF?*

- CNN: input region that can affect a unit, fixed by architecture.
- Neuro: measured region + filter shape + dynamics. Changes w/ stimulus, context, adaptation & attention.

*Why center–surround?*

- Neighboring pixels in natural images are strongly correlated → raw transmission is redundant.
- Center–surround ≈ subtract the local prediction → decorrelate → [Efficient Coding Hypothesis](#efficient-coding-hypothesis).
```

&nbsp;

#### Spike-Triggered Average
- **What**: Mean stimulus segment preceding a spike → estimate of the linear RF filter.
- **Why**: Measure an RF w/o guessing its shape. Play random stimuli; let the spikes select the relevant ones.
- **How**:
    1. Play a long white-noise stimulus.
    2. For each spike, take the stimulus window before it.
    3. Average those windows → STA.
    - Centered Gaussian white noise + [LNP](#lnp-model) neuron → STA ∝ linear filter (up to scale), if $\mathbb{E}[\phi'(\mathbf{k}^T\mathbf{s})]\ne0$.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{s}_t$: Stimulus window before time $t$ (vector).
- Misc:
    - $\mathbf{k}$: True linear filter.
    - $\phi$: Static nonlinearity.

Definition (spikes at $t_1,\dots,t_n$):

$$
\text{STA}=\frac{1}{n}\sum_{k=1}^n\mathbf{s}_{t_k}
$$

Properties:
- $\mathbf{s}\sim\mathcal{N}(0,\sigma^2I)$, $P(\text{spike}\mid\mathbf{s})\propto\phi(\mathbf{k}^T\mathbf{s})$ → $\text{STA}\to\sigma^2\frac{\mathbb{E}[\phi'(\mathbf{k}^T\mathbf{s})]}{\mathbb{E}[\phi(\mathbf{k}^T\mathbf{s})]}\mathbf{k}$.
- Correlated Gaussian stimulus, covariance $\Sigma$ → $\text{STA}\propto\Sigma\mathbf{k}$ → estimate $\mathbf{k}\propto\Sigma^{-1}\text{STA}$.
- Symmetric $\phi$ (e.g., $\phi(x)=x^2$) → $\mathbb{E}[\phi']=0$ → STA = 0 → use spike-triggered covariance.
- Binned Poisson counts: weight each window by its spike count (= $\mathbb{E}[n_t\mid\mathbf{s}_t]\propto\phi$), as in the code.
```

```{tip} Derivation
:class: dropdown
*Why is the STA proportional to the filter?*
1. Bayes: $\text{STA}\to\mathbb{E}[\mathbf{s}\mid\text{spike}]=\frac{\mathbb{E}[\mathbf{s}\,\phi(\mathbf{k}^T\mathbf{s})]}{\mathbb{E}[\phi(\mathbf{k}^T\mathbf{s})]}$.
2. Stein's lemma, $\mathbf{s}\sim\mathcal{N}(0,\sigma^2I)$: $\mathbb{E}[\mathbf{s}\,g(\mathbf{s})]=\sigma^2\,\mathbb{E}[\nabla g(\mathbf{s})]$.
3. $\nabla_\mathbf{s}\phi(\mathbf{k}^T\mathbf{s})=\phi'(\mathbf{k}^T\mathbf{s})\,\mathbf{k}$.
4. → $\text{STA}=\sigma^2\frac{\mathbb{E}[\phi'(\mathbf{k}^T\mathbf{s})]}{\mathbb{E}[\phi(\mathbf{k}^T\mathbf{s})]}\mathbf{k}\propto\mathbf{k}$.
5. $\phi=\exp$ → $\phi'=\phi$ → $\text{STA}=\sigma^2\mathbf{k}$ exactly (in expectation).
```

````{important} Code
:class: dropdown
```python
import torch

class STA:
    def __init__(self, n_lags):
        self.n_lags = n_lags

    def __call__(self, stim, spikes):
        ## stim, spikes: (T,); spike at t uses stim[t - n_lags + 1 : t + 1]
        windows = stim.unfold(0, self.n_lags, 1)   ## (T - n_lags + 1, n_lags), oldest → newest
        w = spikes[self.n_lags - 1:]                ## spike count at the last sample of each window
        assert w.sum() > 0, "no spikes → STA undefined"
        return (w[:, None] * windows).sum(0) / w.sum()

## Example: LNP neuron w/ a known filter, exp nonlinearity
torch.manual_seed(0)
k = torch.tensor([0.0, 0.2, 0.5, 1.0])                                ## weights on the last 4 samples
stim = torch.randn(200_000)                                           ## Gaussian white noise, sigma = 1
rate = 0.1 * torch.exp(stim.unfold(0, 4, 1) @ k)                      ## L → N
spikes = torch.cat([torch.zeros(3), torch.poisson(rate)])             ## → P; pad to align w/ stim
print(STA(4)(stim, spikes))  ## ≈ tensor([0.0, 0.2, 0.5, 1.0]) ← exp → STA = sigma^2 k
```
````

```{attention} Q&A
:class: dropdown
*Why white noise?*

- Uncorrelated stimulus → STA not biased by stimulus correlations.
- Natural stimuli: correct w/ $\Sigma^{-1}$ (+ regularization, since $\Sigma$ is ill-conditioned).

*When does the STA fail?*

- Symmetric nonlinearity (e.g., V1 complex cells respond to both contrast polarities) → STA ≈ 0.
- Several relevant filters → STA = 1 mixed direction → use **spike-triggered covariance** (STC):
    - Whiten the stimulus → eigendecompose the covariance of spike-triggered windows → eigenvalues ≠ prior variance → relevant directions.
    - Misses features that change neither the mean nor the covariance.
```

&nbsp;

### LNP Model
- **Name**: Linear–Nonlinear–Poisson model {cite:p}`chichilnisky2001simple`
- **What**: Encoding model: linear filter → static nonlinearity → Poisson spikes.
- **Why**: Need a fittable stimulus → spike model that captures RF + rectification.
- **How**:
    1. **L**: filter the stimulus w/ the RF $\mathbf{k}$ → drive.
    2. **N**: pass the drive through a nonlinearity $\phi$ → rate ≥ 0.
    3. **P**: draw spikes from a Poisson process at that rate.
    - Fit: [STA](#spike-triggered-average) → $\mathbf{k}$. Mean rate vs drive → $\phi$.

```{note} Math
:class: dropdown
Model:

$$
r(t)=\phi\left(\mathbf{k}^T\mathbf{s}_t\right),\quad n_t\sim\text{Poisson}\left(r(t)\,\Delta t\right)
$$

- $n_t$: Spike count in bin $t$.
```

```{attention} Q&A
:class: dropdown
*Pros?*

- Few params, fast fit, good for early sensory neurons (retina, LGN).
- Fit by STA (Gaussian white noise) or by Poisson maximum likelihood (Paninski, 2004) → any stimulus.

*Cons?*

- No spike history (refractoriness, bursting), no interactions between neurons, 1 filter only.
- Fix → [GLM](#glm).
```

&nbsp;

#### GLM
- **Name**: Generalized Linear Model (neural encoding) {cite:p}`pillow2008spatio`
- **What**: LNP + spike-history & coupling filters, fit by Poisson MLE.
- **Why**: LNP ignores refractoriness, bursting & interactions between neurons.
- **How**: Add the neuron's own past spikes & other neurons' past spikes as extra linear inputs before the exponential.
    - General form → [GLM](../ml/supervised.md#glm).

```{note} Math
:class: dropdown
Notations:
- IO:
    - $y_a(t)$: Spike count of neuron $a$ in bin $t$.
    - $\mathbf{y}_a^\text{hist}(t)$: Recent past counts of neuron $a$.
- Params:
    - $\mathbf{k}_a$: Stimulus filter.
    - $\mathbf{h}_a$: Spike-history filter.
    - $\mathbf{c}_{ab}$: Coupling filter from $b$ to $a$.
    - $\mu_a$: Baseline log-rate.

Model:

$$
\lambda_a(t)=\exp\left(\mathbf{k}_a^T\mathbf{s}_t+\mathbf{h}_a^T\mathbf{y}_a^\text{hist}(t)+\sum_{b\ne a}\mathbf{c}_{ab}^T\mathbf{y}_b^\text{hist}(t)+\mu_a\right)
$$

Training (Poisson log-likelihood, maximize):

$$
\log L=\sum_t\left[y_a(t)\log\lambda_a(t)-\lambda_a(t)\,\Delta t\right]+\text{const}
$$

- $\exp$ link → $\log L$ concave in all params → convex optimization; the max is unique if the design matrix has full column rank & a finite max exists (e.g., ❌ if all counts are 0).
```

```{attention} Q&A
:class: dropdown
*Coupling filter = synapse?*

- ❌. Statistical dependence. Can come from common input, not a direct connection.

*What did it show in retina?*

- Complete population of ON & OFF parasol ganglion cells in a patch of macaque retina: coupling filters capture correlated firing & improve decoding vs an independent model.
```

&nbsp;

## Codes

### Rate Code
- **What**: Information carried by the number of spikes in a time window.
- **Why**: Robust to timing jitter.
    - Classic finding: stimulus intensity ↔ firing rate (stretch receptors, 1920s).
- **How**: Count spikes over a window (tens–hundreds of ms), or across a population in a short window.

```{attention} Q&A
:class: dropdown
*Is a single-neuron rate code fast enough?*

- Measured: humans categorize natural images (animal vs no animal) w/ EEG differences at ~150 ms (Thorpe et al., 1996).
- Estimated: ~10 processing stages → ~10–20 ms per stage → ≤ 1–2 spikes per neuron per stage.
- Estimated, ❌measured: → 1 neuron's count in ~10 ms = 0–2 spikes → rate estimate's relative SD ≈ $1/\sqrt{rT}$ → huge → population rate, or spike timing.

*Rate vs temporal code: a real dichotomy?*

- Partly a matter of timescale: counts in many 5 ms bins keep timing; 1 count over a long window discards it.
- Real question: at which timescale does spike timing carry information beyond the rate?
```

&nbsp;

### Temporal Code
- **What**: Information carried by precise spike times: relative to stimulus onset, to other spikes, or to an oscillation.
- **Why**: Timing adds capacity & speed beyond counts.
- **How**:
    - **Latency code**: first-spike time after onset encodes intensity / identity.
    - **Phase code**: spike phase relative to an ongoing oscillation.
        - Hippocampal **phase precession**: a place cell fires at earlier & earlier theta phases as the animal crosses its field. {cite:p}`okeefe1993phase`
    - **Synchrony**: which neurons fire together.

```{attention} Q&A
:class: dropdown
*Where is timing clearly used?*

- Sound localization: interaural time differences of tens of µs → timing must matter there.
- Phase precession: phase predicts position within the field better than rate alone.

*Why is it hard to prove a temporal code?*

- Must show that downstream neurons read the timing & that behavior depends on it.
- Decodable timing ≠ used timing.
```

&nbsp;

### Population Code
- **What**: Stimulus represented by the joint activity of many broadly tuned neurons.
- **Why**: 1 neuron is noisy, slow to read & ambiguous (same rate for 2 stimuli on opposite flanks).
    - → Many neurons → fast, robust, unambiguous.
- **How**:
    1. Neurons tile stimulus space w/ overlapping tuning curves.
    2. Stimulus → activity pattern $\mathbf{n}$ across the population.
    3. Downstream reads the pattern (population vector, ML, Bayesian) → [Neural Decoding](#neural-decoding).

```{attention} Q&A
:class: dropdown
*Why broad, overlapping tuning?*

- Each stimulus activates many neurons → averaging ⬆️ precision.
- Resolution can be much finer than the tuning width.

*Relation to ANN representations?*

- Distributed codes: features = directions in population space, ❌single units → [Superposition](../math/geo/repr.md#superposition), [Neural Manifold](../math/geo/repr.md#neural-manifold).

*AC angle: the binding problem?*

- Several objects at once → which features belong to which object?
- Binding by synchrony was proposed. Evidence is contested.
```

&nbsp;

#### Population Vector
- **What**: Decoded direction = rate-weighted sum of each neuron's preferred-direction vector. {cite:p}`georgopoulos1986neuronal`
- **Why**: Simple linear readout of cosine-tuned populations (motor-cortex arm direction).
- **How**:
    1. Each neuron $a$ votes for its preferred direction $\mathbf{c}_a$ (unit vector).
    2. Vote weight = its rate above baseline.
    3. Sum the votes → direction of the sum = estimate.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{u}$: True movement direction (unit vector).
    - $r_a$: Rate of neuron $a$.
- Params:
    - $\mathbf{c}_a$: Preferred direction of neuron $a$ (unit vector).
    - $r_0$: Baseline rate.
    - $g=r_\text{max}-r_0$: Tuning gain.

Population vector:

$$
\mathbf{v}_\text{pop}=\sum_{a=1}^N(r_a-r_0)\,\mathbf{c}_a
$$

Properties:
- Cosine tuning + equally spaced $\mathbf{c}_a$ (2D) → $\mathbb{E}[\mathbf{v}_\text{pop}]=\frac{gN}{2}\mathbf{u}$ → mean vector ∥ true direction.
- Non-uniform $\mathbf{c}_a$ → biased → least-squares correction: $\hat{\mathbf{u}}\propto\left(\sum_a\mathbf{c}_a\mathbf{c}_a^T\right)^{-1}\mathbf{v}_\text{pop}$ (best linear unbiased if noise is equal & uncorrelated; else use GLS).
```

```{tip} Derivation
:class: dropdown
1. Cosine tuning: $r_a-r_0=g\cos(\theta-\theta_a)=g\,\mathbf{c}_a^T\mathbf{u}$.
2. → $\mathbf{v}_\text{pop}=g\left(\sum_a\mathbf{c}_a\mathbf{c}_a^T\right)\mathbf{u}$.
3. $N\ge3$ equally spaced directions in 2D: $\sum_a\mathbf{c}_a\mathbf{c}_a^T=\frac{N}{2}I$ (exact).
4. → $\mathbf{v}_\text{pop}=\frac{gN}{2}\mathbf{u}$ ∥ $\mathbf{u}$.
```

````{important} Code
:class: dropdown
```python
import torch

class PopulationVector:
    def __init__(self, pref_angles, r0):
        self.C = torch.stack([pref_angles.cos(), pref_angles.sin()], dim=1)  ## (N, 2) preferred directions
        self.r0 = r0

    def __call__(self, rates):
        v = ((rates - self.r0)[:, None] * self.C).sum(0)  ## each neuron votes for its preferred direction
        return torch.atan2(v[1], v[0])                    ## decoded angle

## Example: 8 cosine-tuned neurons, true direction 60°, no noise
pref = torch.arange(8) * (2 * torch.pi / 8)
theta = torch.tensor(torch.pi / 3)
rates = 30 + 20 * torch.cos(theta - pref)                ## r0 = 30 Hz, gain = 20 Hz
print(PopulationVector(pref, r0=30)(rates).rad2deg())    ## tensor(60.) up to float error
```
````

```{attention} Q&A
:class: dropdown
*Is it optimal?*

- ❌ in general. Ignores noise statistics & non-cosine tuning. ML / Bayesian decoders do better.

*Does motor cortex "encode direction"?*

- Direction is decodable. But motor-cortex population activity is better described as a dynamical system that generates movement → [Trained RNN Models](network.md#trained-rnn-models).

*Practical use?*

- Early brain–computer interfaces decoded arm movement from motor cortex w/ population-vector-style linear decoders.
```

&nbsp;

### Sparse Code
- **What**: Each stimulus activates few neurons strongly (population sparseness). {cite:p}`olshausen1996emergence`
    - Often also: each neuron is active for few stimuli (lifetime sparseness). Not implied by the first.
- **Why**:
    - Energy ← spikes are expensive.
    - Explicit features → easy linear readout & high associative-memory capacity.
    - Natural images have sparse structure (edges) → sparse description is efficient.
- **How**: Learn a dictionary so that each input ≈ a few active dictionary elements.
    1. Infer coefficients: minimize reconstruction error + sparsity penalty.
    2. Update the dictionary to reduce the residual (Hebbian-like).
    3. Train on natural image patches → dictionary = localized, oriented, bandpass filters ≈ V1 simple-cell RFs.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{x}$: Input (image patch).
    - $\mathbf{a}$: Coefficients (neural activity).
- Params:
    - $\Phi$: Dictionary; columns = basis functions (unit norm).
- Hyperparams:
    - $\lambda$: Sparsity weight.
    - $S$: Sparsity cost (e.g., $|\cdot|$).
    - $\eta$: Inference step size.

Objective:

$$
\min_{\Phi,\{\mathbf{a}\}}\sum_\text{patches}\left[\frac{1}{2}\|\mathbf{x}-\Phi\mathbf{a}\|^2+\lambda\sum_jS(a_j)\right]
$$

Inference w/ $S=|\cdot|$ (ISTA):

$$
\mathbf{a}\leftarrow\text{soft}_{\eta\lambda}\left(\mathbf{a}+\eta\,\Phi^T(\mathbf{x}-\Phi\mathbf{a})\right),\quad\text{soft}_\tau(z)=\text{sign}(z)\max(|z|-\tau,0)
$$

- $\eta\le1/\|\Phi\|_2^2$ → converges.

Learning (gradient on $\Phi$, then renormalize columns):

$$
\Delta\Phi\propto(\mathbf{x}-\Phi\mathbf{a})\,\mathbf{a}^T
$$

- = residual × activity → Hebbian-like.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn.functional as F

class SparseCoding:
    def __init__(self, n_pix, n_basis, lam=0.1, lr_dict=0.01):
        self.Phi = F.normalize(torch.randn(n_pix, n_basis), dim=0)   ## unit-norm basis functions
        self.lam, self.lr_dict = lam, lr_dict

    def infer(self, x, n_steps=100):
        ## ISTA: gradient step on reconstruction error, then soft-threshold → exact zeros
        eta = 1.0 / torch.linalg.matrix_norm(self.Phi, ord=2) ** 2
        a = torch.zeros(x.shape[0], self.Phi.shape[1])
        for _ in range(n_steps):
            a = a + eta * (x - a @ self.Phi.T) @ self.Phi
            a = torch.sign(a) * (a.abs() - eta * self.lam).clamp(min=0)
        return a

    def learn(self, x):
        a = self.infer(x)
        ## residual x activity (Hebbian-like), then renormalize so sparsity can't be cheated by scaling
        self.Phi += self.lr_dict * (x - a @ self.Phi.T).T @ a / x.shape[0]
        self.Phi = F.normalize(self.Phi, dim=0)
        return a

## Example: 16 patches of 64 pixels, 128 basis functions (2x overcomplete)
sc = SparseCoding(64, 128)
a = sc.learn(torch.randn(16, 64))
print(a.shape, (a != 0).float().mean())  ## torch.Size([16, 128]), fraction active < 1
```
````

```{attention} Q&A
:class: dropdown
*Local vs sparse vs dense code?*

- **Local** ("grandmother cell"): 1 neuron per item → no generalization, fragile.
- **Dense**: most neurons active → high capacity, but interference & high energy.
- **Sparse**: middle ground.
- Human medial temporal lobe "concept cells": respond to few specific people / objects across modalities → sparse, ❌strictly local.

*Lifetime vs population sparseness?*

- Lifetime: 1 neuron over many stimuli. Population: many neurons for 1 stimulus. Not equivalent.

*Why must $\Phi$'s columns be normalized?*

- Else scale $\Phi$ up & $\mathbf{a}$ down → sparsity cost → 0 w/o real sparsity.

*Relation to ML?*

- [ICA](../ml/unsupervised.md#ica) on natural images also gives Gabor-like filters.
- [SAE](../math/geo/repr.md#sae) = same objective w/ an amortized encoder instead of iterative inference.
```

&nbsp;

### Mixed Selectivity
- **What**: Neurons respond to nonlinear combinations of several task variables, ❌1 variable each. {cite:p}`rigotti2013importance`
- **Why**: Linear readouts only separate linearly separable classes.
    - Nonlinear mixing ⬆️ dimensionality → many context-dependent rules become linearly decodable.
- **How**:
    1. Prefrontal neurons respond to conjunctions (e.g., stimulus × task rule).
    2. → Population activity is high-dimensional across task conditions.
    3. 1 downstream linear readout per rule → many rules from 1 population.
    - Additive codes (pure or linear mixed selectivity) of binary variables → dimensionality ≤ #variables + 1 → XOR-like rules impossible for a linear readout.

```{attention} Q&A
:class: dropdown
*ML analogy?*

- Random nonlinear features / hidden layer → expand dimension → linear separability (Cover's theorem) → [Kernel Trick](../ml/supervised.md#kernel-trick).

*Evidence?*

- Monkey PFC: many neurons show nonlinear mixed selectivity. Dimensionality collapses on error trials.

*Cost?*

- High dimension ⬆️ flexibility but ⬇️ generalization to new conditions.
- Abstraction needs low-dimensional, shared structure → trade-off.
```

&nbsp;

## Decoding

### Neural Decoding
- **What**: Estimate the stimulus (or intention) from observed neural activity.
- **Why**: Test which information a population carries; build brain–computer interfaces.
- **How**:
    1. Fit an encoding model $P(\mathbf{n}\mid s)$ (e.g., independent Poisson w/ tuning curves).
    2. Invert it: **ML** $\hat s=\arg\max_sP(\mathbf{n}\mid s)$, or **Bayesian** $P(s\mid\mathbf{n})\propto P(\mathbf{n}\mid s)P(s)$.
    3. Or fit a direct decoder $\hat s=g(\mathbf{n})$ (linear regression, classifier).

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{n}=(n_1,\dots,n_N)$: Spike counts in window $T$.
- Params:
    - $s_a$: Preferred stimulus of neuron $a$.
    - $\sigma_a$: Tuning width of neuron $a$.

Independent Poisson log-likelihood:

$$
\log P(\mathbf{n}\mid s)=\sum_{a=1}^N\left[n_a\log f_a(s)-Tf_a(s)\right]+\text{const}
$$

Zero-baseline Gaussian tuning, dense uniform tiling ($\sum_af_a(s)$ = const in $s$), ≥ 1 spike, ML estimate:

$$
\hat s=\frac{\sum_an_as_a/\sigma_a^2}{\sum_an_a/\sigma_a^2}
$$

- Equal widths → spike-weighted mean of preferred stimuli (center of mass).
```

```{tip} Derivation
:class: dropdown
1. Dense tiling → $T\sum_af_a(s)$ independent of $s$ → drop it.
2. $\log f_a(s)=\log r_\text{max}-\frac{(s-s_a)^2}{2\sigma_a^2}$.
3. $\frac{\partial}{\partial s}\sum_an_a\log f_a(s)=-\sum_an_a\frac{s-s_a}{\sigma_a^2}=0$.
4. → $\hat s=\frac{\sum_an_as_a/\sigma_a^2}{\sum_an_a/\sigma_a^2}$.
```

````{important} Code
:class: dropdown
```python
import torch

class PoissonMLDecoder:
    def __init__(self, tuning, s_grid, T):
        self.F = tuning(s_grid)      ## (G, N): mean rate of each neuron at each candidate s
        self.s_grid, self.T = s_grid, T

    def __call__(self, n):
        ## independent-Poisson log-likelihood of counts n (N,) for every candidate s
        ll = torch.xlogy(n, self.F).sum(1) - self.T * self.F.sum(1)   ## xlogy: 0 * log 0 = 0
        return self.s_grid[ll.argmax()]

## Example: 20 Gaussian-tuned neurons tiling [-10, 10]
prefs = torch.linspace(-10, 10, 20)
tuning = lambda s: 50 * torch.exp(-(s[:, None] - prefs) ** 2 / (2 * 2.0 ** 2)) + 1e-3
torch.manual_seed(0)
n = torch.poisson(tuning(torch.tensor([3.0]))[0] * 0.2)                 ## counts in T = 0.2 s
print(PoissonMLDecoder(tuning, torch.linspace(-10, 10, 2001), T=0.2)(n))  ## ≈ 3
```
````

```{attention} Q&A
:class: dropdown
*Decodable = used by the brain?*

- ❌. An experimenter's decoder can read information that downstream areas ignore.
- Causal tests needed: perturb the neurons → check that behavior changes as predicted.

*Linear vs nonlinear decoder?*

- Linear decodability = information is "explicit" → a single downstream layer can use it.
- Nonlinear-only decodability = information present but entangled.

*Ignore correlations?*

- Decoder that assumes independence on correlated data can lose information.
```

&nbsp;

### Population Fisher Information
- **What**: Fisher information that a population's responses carry about $s$ → bounds the precision of any unbiased decoder.
- **Why**: Decoder-independent measure of code quality → compare codes, tuning widths, noise correlations.
- **How**:
    1. Cramér–Rao: $\text{Var}[\hat s]\ge1/I_F(s)$ → [Cramér–Rao Bound](../math/prob/stats.md#cram-r-rao-bound).
    2. Independent neurons → contributions add.
    3. Each neuron contributes most on its tuning-curve flanks (Poisson: max $f'^2/f$).
    4. Correlations along the signal direction $\boldsymbol\mu'(s)$ cap the total.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\mathbf{f}(s)$: Vector of tuning curves (rates).
    - $\boldsymbol\mu(s)=T\mathbf{f}(s)$: Mean spike counts; $\boldsymbol\mu'(s)$ its derivative.
    - $\Sigma$: Noise covariance of the counts at fixed $s$ (❌covariance pooled across stimuli, which adds tuning-curve covariance).

Independent Poisson:

$$
I_F(s)=T\sum_{a=1}^N\frac{f_a'(s)^2}{f_a(s)}
$$

Gaussian noise, stimulus-independent $\Sigma$ (linear Fisher information):

$$
I_F(s)=\boldsymbol\mu'(s)^T\Sigma^{-1}\boldsymbol\mu'(s)
$$

Information-limiting correlations {cite:p}`morenobote2014information`:

$$
\Sigma=\Sigma_0+\epsilon\,\boldsymbol\mu'\boldsymbol\mu'^T\ \Rightarrow\ I_F=\frac{I_0}{1+\epsilon I_0}<\frac{1}{\epsilon}
$$

- $I_0=\boldsymbol\mu'^T\Sigma_0^{-1}\boldsymbol\mu'$: Information w/o the differential correlations.
- $\epsilon$: Strength of noise along $\boldsymbol\mu'$ (looks like a stimulus change).
```

```{tip} Derivation
:class: dropdown
*Independent Poisson:*
1. $\log P(n_a\mid s)=n_a\log\left(Tf_a(s)\right)-Tf_a(s)-\log n_a!$.
2. Score: $\partial_s\log P=\frac{f_a'}{f_a}\left(n_a-Tf_a\right)$.
3. $I_F=\mathbb{E}[\text{score}^2]=\frac{f_a'^2}{f_a^2}\text{Var}[n_a]=\frac{f_a'^2}{f_a^2}Tf_a=T\frac{f_a'^2}{f_a}$.
4. Independent neurons → sum over $a$.

*Information-limiting correlations:*
1. Sherman–Morrison: $\Sigma^{-1}=\Sigma_0^{-1}-\frac{\epsilon\,\Sigma_0^{-1}\boldsymbol\mu'\boldsymbol\mu'^T\Sigma_0^{-1}}{1+\epsilon\,\boldsymbol\mu'^T\Sigma_0^{-1}\boldsymbol\mu'}$.
2. $\boldsymbol\mu'^T\Sigma^{-1}\boldsymbol\mu'=I_0-\frac{\epsilon I_0^2}{1+\epsilon I_0}=\frac{I_0}{1+\epsilon I_0}$.
3. If $I_0$ grows w/o bound (e.g., ∝ $N$) → $I_F\to1/\epsilon$.
```

```{note} Example
:class: dropdown
*Where does a Gaussian-tuned neuron carry the most information?*
- $f(s)=r_\text{max}e^{-u^2/2\sigma_f^2}$, $u=s-s_\text{pref}$ → $f'=-\frac{u}{\sigma_f^2}f$ → $\frac{f'^2}{f}=\frac{u^2}{\sigma_f^4}f$.
- $u=0$ (peak) → 0 information.
- Max at $|u|=\sqrt2\,\sigma_f$ ← maximize $u^2e^{-u^2/2\sigma_f^2}$.
```

```{attention} Q&A
:class: dropdown
*Do noise correlations always hurt?*

- ❌. Only the component along $\boldsymbol\mu'$ (differential correlations) limits information.
- Other correlations can leave information unchanged or ⬆️ it.

*How does information scale w/ $N$?*

- Independent: $\propto N$ → unbounded.
- W/ differential correlations: saturates at $1/\epsilon$ → behavior limited by upstream noise (e.g., in the senses), ❌by #neurons.

*Fisher information vs mutual information?*

- Fisher: local discriminability around $s$. Mutual information: global, over the stimulus distribution.
- Large populations: the 2 become linked.
```

&nbsp;

## Coding Principles

### Efficient Coding Hypothesis
- **What**: Sensory neurons maximize information about natural inputs under limited resources (spikes, neurons, energy). {cite:p}`barlow1961possible`
- **Why**: Explain *why* codes look the way they do (normative), ❌only *what* they are.
    - Natural inputs are redundant & non-uniform → naive transmission wastes capacity.
- **How**: Match the code to input statistics.
    - **1 neuron, bounded rate**: make the output distribution uniform → response = CDF of input (**histogram equalization**).
        - Fly large monopolar cells: contrast response ≈ CDF of natural contrasts. {cite:p}`laughlin1981simple`
    - **Population, correlated input**:
        - High SNR → decorrelate (whiten) → center–surround RFs.
        - Low SNR → average neighbors instead (smoothing).
    - **Sparse / independent outputs** → [Sparse Code](#sparse-code) → Gabor-like V1 RFs.
    - **Adaptation**: input statistics change → code re-tunes (e.g., contrast adaptation).

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $F_s$: CDF of the stimulus distribution.
    - $g$: Response function $r=g(s)$ (monotonic).
    - $r_\text{max}$: Max rate.

Objective:

$$
\max_g\ I(s;r)\quad\text{s.t. resource limits}
$$

Single neuron, $r\in[0,r_\text{max}]$, small input-independent output noise:

$$
g(s)=r_\text{max}F_s(s)
$$
```

```{tip} Derivation
:class: dropdown
1. $r=g(s)+z$, input-independent noise $z$ → $I(s;r)=h(r)-h(z)$. Small noise → $h(r)\approx h(g(s))$ → maximize $h(g(s))$.
2. $r\in[0,r_\text{max}]$ → max-entropy distribution = uniform.
3. Monotonic $g$: $P\left(r\le g(s_0)\right)=P(s\le s_0)=F_s(s_0)$.
4. Uniform $r$ needs $P\left(r\le g(s_0)\right)=g(s_0)/r_\text{max}$.
5. → $g(s)=r_\text{max}F_s(s)$: steepest where inputs are most common.
```

```{attention} Q&A
:class: dropdown
*Evidence & limits?*

- Good fits in early sensory systems (retina, LGN, V1).
- Higher areas are shaped by tasks & behavior → efficient coding alone insufficient.

*Other objectives?*

- Min energy for a given information; max predictive information; task-specific accuracy.

*Relation to ML?*

- Infomax → ICA. Whitening → decorrelation layers. Compression trade-offs → [Rate–Distortion Function](../math/info.md#rate-distortion-function), [Information Bottleneck](../math/info.md#information-bottleneck).
```

&nbsp;

### Divisive Normalization
- **What**: Response = own driving input ÷ (constant + pooled drive of a neighborhood). {cite:p}`heeger1992normalization`
- **Why**:
    - Fit a huge input range (contrast, light level) into a limited firing range.
    - Make responses depend on relative, ❌absolute, input → e.g., contrast-invariant orientation tuning.
    - Explains cross-orientation suppression, surround suppression & attention effects.
- **How**:
    1. Each neuron computes its drive (filter → rectify → power).
    2. A pool of neighbors sums their drives.
    3. Divide own drive by (constant + pooled drive).
    - Found from invertebrates to primate cortex → "canonical neural computation". {cite:p}`carandini2012normalization`

```{note} Math
:class: dropdown
Notations:
- IO:
    - $D_a$: Driving input of neuron $a$.
    - $R_a$: Normalized response.
    - $c$: Stimulus contrast.
- Params:
    - $w_{ab}$: Normalization weights.
- Hyperparams:
    - $q$: Exponent (≈ 2).
    - $\sigma$: Semi-saturation constant.
    - $\gamma$: Response scale.

Model:

$$
R_a=\gamma\frac{D_a^{\,q}}{\sigma^q+\sum_bw_{ab}D_b^{\,q}}
$$

Contrast response (pool = the neuron itself, $w_{aa}=1$, $D=c$; max response = $\gamma$):

$$
R(c)=\gamma\frac{c^q}{\sigma^q+c^q}
$$

- = Naka–Rushton (hyperbolic ratio) function. $c=\sigma$ → half max.
```

````{important} Code
:class: dropdown
```python
import torch

class DivisiveNormalization:
    def __init__(self, W, sigma=1.0, q=2.0, gamma=1.0):
        self.W, self.sigma, self.q, self.gamma = W, sigma, q, gamma   ## W: (N, N) pool weights

    def __call__(self, drive):
        d = drive.clamp(min=0) ** self.q                                 ## rectify & exponentiate
        return self.gamma * d / (self.sigma ** self.q + d @ self.W.T)    ## divide by pooled drive

## Example: same pattern at 2 contrasts → ratios kept, magnitude saturates
dn = DivisiveNormalization(torch.ones(3, 3))
x = torch.tensor([1.0, 2.0, 4.0])
print(dn(x))       ## ≈ [0.045, 0.18, 0.73] = [1, 4, 16] / 22
print(dn(10 * x))  ## ≈ [0.048, 0.19, 0.76] = [100, 400, 1600] / 2101 → same 1:4:16, sum < gamma
```
````

```{attention} Q&A
:class: dropdown
*Relation to DL?*

- [Softmax](../dl/basics/activation.md#softmax) = exponentiate + divide by the pooled sum ($\sigma=0$).
- AlexNet's local response normalization was explicitly inspired by lateral inhibition in real neurons.
- [LayerNorm](../dl/basics/module.md#layernorm): subtract mean, divide by pooled std → related, ❌identical.

*Mechanism?*

- Not settled. Candidates: shunting inhibition, synaptic depression, recurrent circuits → may differ by system.
- → [Synapse Model](neuron.md#synapse-model).

*Attention?*

- Normalization model of attention: attention scales the drive before normalization → [Attention](systems.md#attention).
```

&nbsp;
