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
# Theory
Normative frameworks for brain computation, & where neuroscience meets AI (NeuroAI).

Notations:
- $\mathbf{x}$: Sensory input (observation)
- $\mathbf{s}$: Hidden cause (latent state of the world)
- $p(\mathbf{x},\mathbf{s})$: Generative model
- $q(\mathbf{s})$: Approximate posterior (the brain's belief)

&nbsp;

## Frameworks

### Marr's Levels
- **What**: 3 levels of explanation for an information-processing system: computational, algorithmic, implementational.
- **Why**: Data at 1 level do not explain the others.
    - "Which neurons fire" ≠ "which problem is solved" ≠ "which algorithm solves it".
    - → State which level a theory addresses.
- **How**:
    1. **Computational**: what problem is solved & why (goal, constraints).
    2. **Algorithmic / representational**: which representations & which algorithm.
    3. **Implementational**: physical substrate (neurons, synapses, transistors).

```{dropdown} Table: Marr's Levels by Example
| Level | Cash register | Path integration | LLM |
|:--|:--|:--|:--|
| Computational | Addition | Estimate own position from self-motion | Predict the next token |
| Algorithmic | Decimal arithmetic w/ carries | Velocity input shifts a [Continuous Attractor](network.md#continuous-attractor) | Transformer: attention + MLP circuits |
| Implementational | Gears / transistors | Grid-cell & head-direction circuits | GPU kernels |
```

```{attention} Q&A
:class: dropdown
*Are the levels independent?*

- ❌ fully. 1 computation → many algorithms; 1 algorithm → many implementations.
- But constraints flow both ways: energy & noise of neurons constrain which algorithms are feasible.

*Source?*

- 3-level version: Marr's *Vision* (1982). Earlier Marr & Poggio memo (1976) listed 4 levels.

*AC angle?*

- Functionalism: consciousness depends only on the algorithmic level → software can be conscious.
- IIT: depends on the implementational level (physical causal structure) → [IIT](../ac/theories.md#integrated-information-theory-iit).
- → AC theories disagree on *which level* matters.
```

&nbsp;

### Bayesian Brain
- **What**: Hypothesis: the brain represents uncertainty as probability distributions & combines evidence by (approximately) Bayesian inference. {cite:p}`knill2004bayesian`
- **Why**: Senses are noisy & ambiguous → good perception & action must weigh each source by its reliability & use prior knowledge.
- **How**:
    1. Prior $p(\mathbf{s})$ ← experience.
    2. Likelihood $p(\mathbf{x}\mid\mathbf{s})$ ← sensory noise model.
    3. Posterior $\propto$ likelihood × prior → percept & decision.
    4. Several cues → weight each by its reliability (inverse variance).
    - Visual–haptic size judgments: humans match the optimal reliability weights. {cite:p}`ernst2002humans`

```{note} Math
:class: dropdown
Notations:
- IO:
    - $x_i$: Estimate of a scalar $s$ from cue $i$.
    - $\hat s$: Combined estimate.
- Misc:
    - $\sigma_i^2$: Noise variance of cue $i$.

Model: $x_i\mid s\sim\mathcal{N}(s,\sigma_i^2)$, independent; flat prior.

Optimal combination:

$$
\hat s=\sum_iw_ix_i,\quad w_i=\frac{1/\sigma_i^2}{\sum_j1/\sigma_j^2},\qquad\sigma_{\hat s}^2=\left(\sum_j\frac{1}{\sigma_j^2}\right)^{-1}\le\min_j\sigma_j^2
$$

- Gaussian prior $\mathcal{N}(\mu_0,\sigma_0^2)$ = 1 extra cue w/ $x_0=\mu_0$.
```

```{tip} Derivation
:class: dropdown
1. Independent cues + flat prior → $p(s\mid\mathbf{x})\propto\prod_i\exp\left(-\frac{(x_i-s)^2}{2\sigma_i^2}\right)$.
2. Log posterior $=-\frac{1}{2}\sum_i\frac{(s-x_i)^2}{\sigma_i^2}+\text{const}$ → quadratic in $s$ → Gaussian.
3. Precision = coefficient of $-\frac{1}{2}s^2$: $\sum_i\frac{1}{\sigma_i^2}$.
4. Mean: set derivative to 0 → $\sum_i\frac{s-x_i}{\sigma_i^2}=0$ → $\hat s=\frac{\sum_ix_i/\sigma_i^2}{\sum_i1/\sigma_i^2}$.
```

````{important} Code
:class: dropdown
```python
import torch

class CueCombination:
    def __call__(self, means, variances):
        precision = 1 / variances
        w = precision / precision.sum()          ## reliable cue → large weight
        return (w * means).sum(), 1 / precision.sum()

## Example: vision says 10 (sigma = 1), touch says 12 (sigma = 2)
s_hat, var = CueCombination()(torch.tensor([10.0, 12.0]), torch.tensor([1.0, 4.0]))
print(s_hat, var)  ## tensor(10.4) (weights 0.8 / 0.2), tensor(0.8) < min(1, 4)
```
````

```{attention} Q&A
:class: dropdown
*Evidence?*

- Cue combination (visual–haptic, audio–visual), motor control under uncertainty, many illusions as priors (e.g., light-from-above, slow-motion prior).

*Limits?*

- "Bayesian" fits can be post hoc: pick a prior & a likelihood that explain any behavior.
- Many tasks show systematic suboptimality & heuristics.
- Exact inference is intractable in large models → the brain must approximate (sampling? variational?).

*Relation to ML?*

- Same as [Bayes' Rule](../math/prob/bayes.md#bayes-rule) & sensor fusion → [Kalman Filter](../math/prob/process.md#kalman-filter).
```

&nbsp;

#### Probabilistic Population Code
- **What**: Population activity encodes a whole posterior over $s$, ❌only a point estimate. {cite:p}`ma2006bayesian`
- **Why**: Bayesian computation needs uncertainty → where is it in spikes?
- **How**:
    1. Poisson-like variability → log-likelihood of $s$ is linear in the spike counts.
    2. Population gain ⬆️ → more spikes → sharper posterior → **gain encodes certainty**.
    3. 2 cues → 2 populations → add their activity → posterior of the combined evidence.
        - Valid if the 2 populations are conditionally independent given $s$, share the same kernel $\mathbf{h}(s)$, & the prior is counted once.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{n}$: Spike counts of a population.
- Misc:
    - $\mathbf{h}(s)$: Stimulus-dependent kernel.
    - $\varphi(\mathbf{n})$: Stimulus-independent factor.
    - $f_a(s)$: Tuning curve of neuron $a$.

Poisson-like family:

$$
p(\mathbf{n}\mid s)=\varphi(\mathbf{n})\exp\left(\mathbf{h}(s)^T\mathbf{n}\right)
$$

Cue combination by summation (conditionally independent populations, same $\mathbf{h}$, flat prior):

$$
p(s\mid\mathbf{n}_1,\mathbf{n}_2)\propto\exp\left(\mathbf{h}(s)^T(\mathbf{n}_1+\mathbf{n}_2)\right)
$$

- → $\mathbf{n}_3=\mathbf{n}_1+\mathbf{n}_2$ encodes the optimal combined posterior.
- Non-flat prior: multiply the likelihoods, then the prior once.
```

```{tip} Derivation
:class: dropdown
*Independent Poisson neurons are Poisson-like:*
1. $p(\mathbf{n}\mid s)=\prod_a\frac{e^{-f_a(s)}f_a(s)^{n_a}}{n_a!}$.
2. $=\left(\prod_a\frac{1}{n_a!}\right)\exp\left(\sum_an_a\log f_a(s)\right)\exp\left(-\sum_af_a(s)\right)$.
3. Dense, uniform tiling → $\sum_af_a(s)\approx C$, constant in $s$.
4. → $\mathbf{h}(s)=\log\mathbf{f}(s)$, $\varphi(\mathbf{n})=e^{-C}\prod_a1/n_a!$ ($e^{-C}$ cancels in the posterior).

*Gain encodes certainty:*
1. Scale all tuning curves by gain $\gamma$: $\log(\gamma f_a)=\log\gamma+\log f_a$ → $\mathbf{h}$ changes only by a constant.
2. Expected counts $\propto\gamma$ → $\mathbf{h}(s)^T\mathbf{n}$ grows $\propto\gamma$ → log posterior more peaked.
```

```{attention} Q&A
:class: dropdown
*Alternative uncertainty codes?*

- **Sampling**: activity over time = samples from the posterior → variability = uncertainty.
- **PPC**: uncertainty in the gain of 1 snapshot.
- Which the brain uses (if either) = open.

*Relation to decoding?*

- Same likelihood as the Poisson decoder → [Neural Decoding](coding.md#neural-decoding).
```

&nbsp;

### Predictive Coding
- **What**: Hierarchical generative model in cortex: feedback carries predictions of lower-level activity; feedforward carries prediction errors. {cite:p}`rao1999predictive`
- **Why**: 2 goals in 1 scheme.
    - Efficient coding: send only what was not predicted.
    - Hierarchical Bayesian inference: higher levels explain lower levels.
- **How**:
    1. Each level predicts the activity of the level below through generative weights.
    2. Error = actual − predicted → sent up.
    3. **Inference** (fast): each level updates its causes to reduce the error below & to agree w/ the prediction from above.
    4. **Learning** (slow): weights change by local Hebbian rule (error × cause).
    5. Extra-classical effects (e.g., **end-stopping**): a long bar is predictable from higher-level context → error ⬇️ → response ⬇️.
    - 1-level inference w/ precision → [Predictive Processing](../ac/theories.md#predictive-processing-pp).

```{note} Math
:class: dropdown
Linear 2-level version.

Notations:
- IO:
    - $\mathbf{r}_1,\mathbf{r}_2$: Causes at levels 1 & 2.
- Params:
    - $U_1$: Generative weights level 1 → input.
    - $U_2$: Generative weights level 2 → level 1.
- Hyperparams:
    - $\sigma_0^2,\sigma_1^2$: Error variances (inverse precisions).
    - $\kappa$: Inference rate.
- Misc:
    - $\boldsymbol\varepsilon_0=\mathbf{x}-U_1\mathbf{r}_1$: Input error.
    - $\boldsymbol\varepsilon_1=\mathbf{r}_1-U_2\mathbf{r}_2$: Level-1 error.

Objective ($=-\log p(\mathbf{x},\mathbf{r}_1\mid\mathbf{r}_2)$ + const, isotropic Gaussian errors; a prior on $\mathbf{r}_2$ adds a term):

$$
E=\frac{1}{2\sigma_0^2}\|\boldsymbol\varepsilon_0\|^2+\frac{1}{2\sigma_1^2}\|\boldsymbol\varepsilon_1\|^2
$$

Inference (gradient descent on $E$):

$$
\Delta\mathbf{r}_1=\kappa\left(\frac{U_1^T\boldsymbol\varepsilon_0}{\sigma_0^2}-\frac{\boldsymbol\varepsilon_1}{\sigma_1^2}\right),\qquad\Delta\mathbf{r}_2=\kappa\,\frac{U_2^T\boldsymbol\varepsilon_1}{\sigma_1^2}
$$

- $\mathbf{r}_1$: pushed by the error below (bottom-up) & pulled by the prediction from above (top-down).

Learning:

$$
\Delta U_1\propto\boldsymbol\varepsilon_0\mathbf{r}_1^T,\qquad\Delta U_2\propto\boldsymbol\varepsilon_1\mathbf{r}_2^T
$$

- = error × presynaptic cause → local & Hebbian-like.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn.functional as F

class HierarchicalPC:
    def __init__(self, sizes=(16, 8, 4), kappa=0.05, lr=0.01):
        self.U1 = F.normalize(torch.randn(sizes[0], sizes[1]), dim=0)   ## level 1 → input
        self.U2 = F.normalize(torch.randn(sizes[1], sizes[2]), dim=0)   ## level 2 → level 1
        self.sizes, self.kappa, self.lr = sizes, kappa, lr

    def infer(self, x, n_steps=100):
        r1, r2 = torch.zeros(self.sizes[1]), torch.zeros(self.sizes[2])
        for _ in range(n_steps):
            e0 = x - self.U1 @ r1              ## bottom-up error
            e1 = r1 - self.U2 @ r2             ## mismatch w/ top-down prediction
            r1 = r1 + self.kappa * (self.U1.T @ e0 - e1)
            r2 = r2 + self.kappa * (self.U2.T @ e1)
        ## errors at the final state → learning uses 1 consistent state
        return r1, r2, x - self.U1 @ r1, r1 - self.U2 @ r2

    def learn(self, x):
        r1, r2, e0, e1 = self.infer(x)
        ## local learning: error x cause; renormalize columns (scale is otherwise free)
        self.U1 = F.normalize(self.U1 + self.lr * torch.outer(e0, r1), dim=0)
        self.U2 = F.normalize(self.U2 + self.lr * torch.outer(e1, r2), dim=0)
        return e0.pow(2).sum()

## Example: inputs generated from 4 hidden causes
torch.manual_seed(0)
A = torch.randn(16, 4) / 2
pc = HierarchicalPC()
errs = torch.stack([pc.learn(A @ torch.randn(4)) for _ in range(3000)])
print(errs[:200].mean(), errs[-200:].mean())  ## input error typically drops as U1 learns span(A)
```
````

```{attention} Q&A
:class: dropdown
*Evidence?*

- Mismatch & omission responses, repetition suppression, end-stopping & surround suppression.
- Dedicated error-neuron populations: debated.

*Relation to sparse coding?*

- 1 level + sparse prior on causes = [Sparse Code](coding.md#sparse-code).

*Relation to backprop?*

- Under specific conditions, predictive-coding networks w/ local learning approximate backprop gradients → [Backprop in the Brain](learning.md#backprop-in-the-brain).

*Relation to ML?*

- Same objective type as a hierarchical VAE: amortized encoder replaces iterative inference → [VAE](../dl/gen/vae.md#vae).
- PredNet-style video predictors use explicit error units.
```

&nbsp;

### Free Energy Principle
- **What**: Claim: a self-organizing system that persists (e.g., a brain) must minimize variational free energy = an upper bound on sensory surprise. {cite:p}`friston2010free`
- **Why**: Unify perception, learning, attention & action under 1 objective.
- **How**:
    1. Brain holds a generative model $p(\mathbf{x},\mathbf{s})$ & a belief $q(\mathbf{s})$.
    2. **Perception**: update $q$ → minimize $F$ → approximate posterior.
    3. **Learning**: update the model's params → minimize $F$ over time.
    4. **Attention**: optimize precision (expected reliability of errors).
    5. **Action** (**active inference**): change the input $\mathbf{x}$ by acting → minimize $F$ → make sensations match predictions.
    - [Predictive Coding](#predictive-coding) = FEP perception w/ Gaussian model & a point estimate (fixed posterior covariance) → the free-energy mean update = the predictive-coding update.

```{note} Math
:class: dropdown
Definition:

$$
F[q]=\mathbb{E}_q\left[\ln q(\mathbf{s})-\ln p(\mathbf{x},\mathbf{s})\right]=\text{KL}\left[q(\mathbf{s})\,\|\,p(\mathbf{s}\mid\mathbf{x})\right]-\ln p(\mathbf{x})\ \ge\ -\ln p(\mathbf{x})
$$

- $-\ln p(\mathbf{x})$: Surprise.
- $F$ = negative ELBO.

Accuracy vs complexity:

$$
F=\underbrace{-\mathbb{E}_q\left[\ln p(\mathbf{x}\mid\mathbf{s})\right]}_\text{inaccuracy}+\underbrace{\text{KL}\left[q(\mathbf{s})\,\|\,p(\mathbf{s})\right]}_\text{complexity}
$$

Active inference, immediate action ($a$ changes the input):

$$
a^*=\arg\min_aF\left[q,\mathbf{x}(a)\right]
$$

- Planning (policy selection) instead scores predicted futures w/ **expected free energy** (expected cost + information gain).
```

```{tip} Derivation
:class: dropdown
1. $p(\mathbf{x},\mathbf{s})=p(\mathbf{s}\mid\mathbf{x})\,p(\mathbf{x})$.
2. $F=\mathbb{E}_q\left[\ln q(\mathbf{s})-\ln p(\mathbf{s}\mid\mathbf{x})\right]-\ln p(\mathbf{x})=\text{KL}[q\,\|\,p(\mathbf{s}\mid\mathbf{x})]-\ln p(\mathbf{x})$.
3. KL ≥ 0 → $F\ge-\ln p(\mathbf{x})$; equality iff $q=p(\mathbf{s}\mid\mathbf{x})$.
4. Alternatively $p(\mathbf{x},\mathbf{s})=p(\mathbf{x}\mid\mathbf{s})\,p(\mathbf{s})$ → accuracy–complexity form.
```

```{attention} Q&A
:class: dropdown
*Is it falsifiable?*

- Contested. Critics: a principle (like least action), ❌a testable theory → any behavior fits some generative model.
- Specific process theories (predictive coding, active inference models) make testable predictions.

*Dark-room problem?*

- Why not hide in a dark room to minimize surprise?
- Answer given: the organism's model expects food, warmth, light → a dark room is surprising under those priors. Critics: this moves the work into the priors.

*Relation to RL?*

- Reward → prior preference over observations. Exploration → minimize expected free energy (includes information gain).

*Relation to ML?*

- Perception = [VI](../math/prob/sampling.md#vi); $F$ = negative ELBO of a [VAE](../dl/gen/vae.md#vae).

*AC angle?*

- Basis of predictive processing accounts of consciousness & self → [Predictive Processing](../ac/theories.md#predictive-processing-pp).
```

&nbsp;

## NeuroAI

### Goal-Driven Modeling
- **What**: Model a brain area w/ a deep net optimized for the area's behavioral task; then compare model units to neurons. {cite:p}`yamins2016using`
- **Why**: Hand-designed models fail on natural stimuli. Models fit directly to neural data overfit & explain little.
    - Task optimization → features emerge w/o fitting neural data → fair test.
- **How**:
    1. Pick task (e.g., object categorization), architecture (deep CNN) & training data.
    2. Train on the task only.
    3. Compare layers to recordings: **neural predictivity** (regress neurons on features), [RSA](../math/geo/repr.md#rsa), [CKA](../math/geo/repr.md#cka).
    4. Find layer ↔ area correspondence.
    - Ventral visual stream: intermediate layers predict V4; top layers predict IT; better categorization ↔ better IT predictivity (within the studied model class). {cite:p}`yamins2014performance`
    - IT population supports linear object decoding ~100–150 ms after image onset → fast enough for a largely feedforward sweep.
        - Harder images get solved ~30 ms later in IT & are poorly predicted by feedforward CNNs → recurrence adds computation.

```{note} Math
:class: dropdown
Neural predictivity.

Notations:
- IO:
    - $\Phi_\ell\in\mathbb{R}^{m\times d}$: Layer-$\ell$ features for $m$ stimuli.
    - $\mathbf{y}_a\in\mathbb{R}^m$: Responses of neuron $a$.
- Params:
    - $\mathbf{w}_a$: Linear map from features to neuron $a$.
- Hyperparams:
    - $\lambda$: Ridge strength.

Fit on training stimuli (intercept $b_a$ unpenalized):

$$
\hat{\mathbf{w}}_a,\hat b_a=\arg\min_{\mathbf{w},b}\|\mathbf{y}_a-\Phi_\ell\mathbf{w}-b\mathbf{1}\|^2+\lambda\|\mathbf{w}\|^2
$$

- Equivalent: center $\Phi_\ell$ & $\mathbf{y}_a$ on their training means, fit $\mathbf{w}$ w/o intercept.

Score on held-out stimuli: $R^2$ (or correlation) between $\Phi_\ell^\text{test}\hat{\mathbf{w}}_a+\hat b_a$ & $\mathbf{y}_a^\text{test}$, often divided by the neuron's noise ceiling.
```

````{important} Code
:class: dropdown
```python
import torch

class NeuralPredictivity:
    def __init__(self, ridge=1.0):
        self.ridge = ridge

    def __call__(self, F, R, n_train):
        ## F: (stimuli, features) from 1 model layer; R: (stimuli, neurons) recordings
        Ftr, Fte, Rtr, Rte = F[:n_train], F[n_train:], R[:n_train], R[n_train:]
        mF, mR = Ftr.mean(0), Rtr.mean(0)
        A = (Ftr - mF).T @ (Ftr - mF) + self.ridge * torch.eye(F.shape[1])
        W = torch.linalg.solve(A, (Ftr - mF).T @ (Rtr - mR))     ## ridge: features → neurons
        pred = (Fte - mF) @ W + mR
        ## held-out R^2 per neuron
        return 1 - ((Rte - pred) ** 2).sum(0) / ((Rte - Rte.mean(0)) ** 2).sum(0)

## Example: "neurons" = noisy linear readout of layer 2 of a toy 2-layer net
torch.manual_seed(0)
S = torch.randn(600, 10)                                          ## stimuli
layer1 = torch.tanh(S @ torch.randn(10, 50))
layer2 = torch.relu(layer1 @ torch.randn(50, 50) / 50 ** 0.5)
R = layer2 @ torch.randn(50, 20) + 0.5 * torch.randn(600, 20)
for name, Fl in [("layer1", layer1), ("layer2", layer2)]:
    print(name, NeuralPredictivity()(Fl, R, n_train=400).median())
## layer2 → high (near the noise ceiling); layer1 → lower ← R is nonlinear in layer1
```
````

```{attention} Q&A
:class: dropdown
*Pros?*

- Best current quantitative models of ventral visual stream & of some auditory & language areas.
- Hypotheses at Marr's computational level (task) + algorithmic level (architecture).

*Cons?*

- Many different models reach similar predictivity → weak discrimination between hypotheses.
- Predictive ≠ explanatory: a 100M-param model of the brain is another black box.
- Systematic gaps: adversarial vulnerability, shape vs texture bias, missing recurrence (images solved late in IT).
- Emergence of brain-like units can depend on hand-picked design choices (e.g., grid-like units in path-integrating RNNs).

*LLMs & the brain?*

- LLM activations linearly predict fMRI responses in human language areas. Across the models tested (Schrimpf et al., 2021), better next-word prediction ↔ higher predictivity.

*Relation to representation convergence?*

- Different models & brains may converge on similar representations → [Platonic Representation Hypothesis](../math/geo/repr.md#platonic-representation-hypothesis).
```

&nbsp;

### Spiking Neural Network
- **What**: Network of spiking neurons (usually [LIF](neuron.md#leaky-integrate-and-fire)); signals = binary spike events over time; trained mostly by surrogate-gradient BPTT. {cite:p}`neftci2019surrogate`
- **Why**:
    - Energy: sparse, event-driven computation on neuromorphic hardware.
    - Biological realism & temporal coding.
    - Problem: spike = step function → derivative 0 almost everywhere → no gradient.
- **How**:
    1. Discrete-time LIF: leak, integrate input, spike at threshold, reset.
    2. Unroll over time like an RNN → backprop through time.
    3. **Surrogate gradient**: forward pass uses the true step; backward pass replaces its derivative w/ a smooth function (e.g., fast-sigmoid derivative). {cite:p}`zenke2018superspike`

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{x}_t\in\mathbb{R}^{n_\text{in}}$: Input spikes / currents at step $t$.
    - $\mathbf{y}_t\in\{0,1\}^{n_\text{out}}$: Output spikes.
- Params:
    - $W\in\mathbb{R}^{n_\text{out}\times n_\text{in}}$: Synaptic weights.
- Hyperparams:
    - $\beta_\text{m}\in(0,1)$: Membrane decay per step.
    - $\vartheta$: Threshold.
    - $\beta_\text{sg}$: Surrogate steepness.
- Misc:
    - $\mathbf{u}_t$: Membrane potential.
    - $\Theta$: Heaviside step.

Forward (soft reset: subtract $\vartheta$ after a spike, then leak):

$$
\mathbf{u}_t=\beta_\text{m}\left(\mathbf{u}_{t-1}-\vartheta\,\mathbf{y}_{t-1}\right)+W\mathbf{x}_t,\qquad\mathbf{y}_t=\Theta(\mathbf{u}_t-\vartheta)
$$

Backward (fast-sigmoid surrogate):

$$
\frac{\partial y}{\partial u}\approx\frac{1}{\left(1+\beta_\text{sg}|u-\vartheta|\right)^2}
$$

- True derivative = Dirac delta → 0 almost everywhere.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, beta=10.0):                  ## v = u - threshold
        ctx.save_for_backward(v)
        ctx.beta = beta
        return (v > 0).float()                       ## forward: true binary spikes

    @staticmethod
    def backward(ctx, grad_out):
        (v,) = ctx.saved_tensors
        ## backward: smooth stand-in for the Dirac delta (fast-sigmoid derivative)
        return grad_out / (1 + ctx.beta * v.abs()) ** 2, None

class SpikingLayer(nn.Module):
    def __init__(self, n_in, n_out, decay=0.9, threshold=1.0):
        super().__init__()
        self.fc = nn.Linear(n_in, n_out, bias=False) ## synaptic weights W (scaffolding)
        self.decay, self.threshold = decay, threshold

    def forward(self, x):                            ## x: (T, B, n_in)
        u = torch.zeros(x.shape[1], self.fc.out_features)
        spikes = []
        for x_t in x:
            u = self.decay * u + self.fc(x_t)        ## leaky integration
            y = SurrogateSpike.apply(u - self.threshold)
            u = u - self.threshold * y               ## soft reset (leak applies next step)
            spikes.append(y)
        return torch.stack(spikes)                   ## (T, B, n_out)

## Example: T = 20 steps, batch 2, 4 inputs → 3 spiking units
torch.manual_seed(0)
layer = SpikingLayer(4, 3)
y = layer(torch.rand(20, 2, 4))
y.sum().backward()
print(y.shape, y.unique(), layer.fc.weight.grad.abs().sum() > 0)
## torch.Size([20, 2, 3]), values ⊆ {0., 1.}, tensor(True) ← surrogate passes gradient
```
````

```{attention} Q&A
:class: dropdown
*Pros?*

- Energy efficiency on neuromorphic chips (event-driven, sparse).
- Native handling of event-based sensors (e.g., event cameras).

*Cons?*

- BPTT memory ∝ #time steps. Accuracy still often below equivalent ANNs on standard benchmarks.
- On GPUs, SNNs are usually *less* efficient than ANNs → benefit needs neuromorphic hardware.

*Alternatives to surrogate gradients?*

- ANN → SNN conversion: train an ANN, map ReLU activations to firing rates → many time steps needed.
- Local / online rules: eligibility traces × learning signal → [Three-Factor Learning Rule](learning.md#three-factor-learning-rule).

*Is the surrogate "cheating"?*

- Yes in the strict sense: forward uses hard spikes; backward swaps in a local pseudo-derivative.
- → The update ≠ the true gradient of the spiking model (0 almost everywhere) & need not be the gradient of any loss. Works well in practice.
- Same idea as the straight-through estimator for quantized nets.
```

&nbsp;
