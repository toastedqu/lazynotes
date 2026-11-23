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
# Learning
Learning rules for neural circuits: synaptic plasticity, reward learning, credit assignment, memory systems. Synapse models → [Neuron Models](neuron.md#synapse-model).

Notations:
- $w$: Synaptic weight (pre → post)
- $\mathbf{w}$: Weight vector onto 1 postsynaptic neuron
- $x$: Presynaptic activity (rate)
- $\mathbf{x}$: Presynaptic activity vector
- $y$: Postsynaptic activity (rate)
- $\eta$: Learning rate
- $t$: Time

&nbsp;

## Synaptic Plasticity

### Short-Term Plasticity
- **What**: Synaptic strength changes over ms–s w/ recent presynaptic spikes: **depression** or **facilitation**.
- **Why**: Turns a synapse into a dynamic filter → it transmits temporal context, ❌only a fixed weight.
- **How**:
    - **Depression**: each spike uses up ready vesicles → next spike releases less → recovery over ~100s of ms – s.
    - **Facilitation**: residual presynaptic calcium ⬆️ release probability for the next spike → decays over ~100s of ms.
    - Depressing synapse at high rates: amplitude ∝ 1/rate → total drive ≈ constant → transmits rate *changes*, ❌steady rate.

```{note} Math
:class: dropdown
Tsodyks–Markram model, depression only {cite:p}`tsodyks1997neural`.

Notations:
- Params:
    - $A_\text{max}$: Max response (all resources released).
- Hyperparams:
    - $U$: Fraction of available resources used per spike.
    - $\tau_\text{rec}$: Recovery time constant.
- Misc:
    - $R\in[0,1]$: Fraction of available resources (vesicles).
    - $t_k$: $k$-th presynaptic spike time.

Model:

$$
\frac{dR}{dt}=\frac{1-R}{\tau_\text{rec}}-U\sum_kR(t_k^-)\,\delta(t-t_k),\qquad A_k=A_\text{max}\,U\,R(t_k^-)
$$

- $A_k$: Response to spike $k$.
- $R(t_k^-)$: $R$ just before spike $k$.
- Facilitation (full model): utilization $u$ jumps by $U(1-u^-)$ at each spike & decays to 0 w/ time constant $\tau_F$; release uses $u^+R^-$ instead of $UR^-$.

Steady state just before each spike, regular rate $r$ (inter-spike interval $\Delta=1/r$):

$$
R^*=\frac{1-e^{-\Delta/\tau_\text{rec}}}{1-(1-U)\,e^{-\Delta/\tau_\text{rec}}}
$$

- High rate ($\Delta\ll U\tau_\text{rec}$): $A^*\approx\frac{A_\text{max}}{r\,\tau_\text{rec}}$ → $A^*\cdot r\approx A_\text{max}/\tau_\text{rec}$, independent of $r$.
```

```{tip} Derivation
:class: dropdown
1. Spike: $R\to(1-U)R$.
2. Between spikes: $1-R$ decays w/ $\tau_\text{rec}$ → $R(\Delta^-)=1-\left(1-R_\text{after}\right)e^{-\Delta/\tau_\text{rec}}$.
3. Steady state: $R^*=1-\left(1-(1-U)R^*\right)e^{-\Delta/\tau_\text{rec}}$ → solve for $R^*$ → formula above.
4. $\Delta\ll\tau_\text{rec}$: numerator ≈ $\Delta/\tau_\text{rec}$. Denominator ≈ $U$ (if $\Delta/\tau_\text{rec}\ll U$).
5. → $R^*\approx\frac{\Delta}{U\tau_\text{rec}}$ → $A^*=A_\text{max}UR^*\approx\frac{A_\text{max}}{r\tau_\text{rec}}$.
```

```{attention} Q&A
:class: dropdown
*Function?*

- Depression: gain control & novelty detection → transmits rate changes.
- Facilitation: favors bursts & sustained high rates.

*Working memory?*

- Model prediction: facilitation can hold an item for ~1 s w/o persistent spiking → [Working Memory](systems.md#working-memory).

*ML analogy?*

- "Fast weights": a 2nd, quickly decaying set of weights that stores recent context.
```

&nbsp;

### Hebbian Learning
- **What**: Strengthen a synapse when pre & post neurons are active together: $\Delta w\propto x\,y$.
- **Why**: Local, unsupervised rule → learns input correlations → associations & features w/o a teacher.
    - Hebb (1949): a cell that repeatedly takes part in firing another → their connection grows ("fire together, wire together").
- **How**:
    1. Pre active & post active → $w$⬆️.
    2. Repeated co-activation → connected groups (cell assemblies); neuron becomes selective to correlated inputs.
    3. Plain rule is unstable → add decay, normalization, or a threshold ([Oja's Rule](#ojas-rule), [BCM Rule](#bcm-rule), [Homeostatic Plasticity](#homeostatic-plasticity)).

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $C=\mathbb{E}[\mathbf{x}\mathbf{x}^T]$: Input correlation matrix ($\mathbf{x}$ zero-mean).
    - $\lambda_j,\mathbf{e}_j$: Eigenvalues & eigenvectors of $C$.
    - $\bar x,\bar y$: Mean pre / post activity.

Linear neuron $y=\mathbf{w}^T\mathbf{x}$, plain Hebb:

$$
\Delta\mathbf{w}=\eta\,y\,\mathbf{x}
$$

Average dynamics:

$$
\frac{d\mathbf{w}}{dt}\propto C\mathbf{w}\ \Rightarrow\ \mathbf{w}(t)=\sum_jc_je^{\lambda_jt}\mathbf{e}_j
$$

- $\|\mathbf{w}\|\to\infty$. Direction → $\pm\mathbf{e}_1$ (top eigenvector), if $\lambda_1>0$ is simple & $\mathbf{w}(0)$ has a nonzero $\mathbf{e}_1$ component.
- $\mathbf{x}$ = activity deviations from the mean (raw rates ≥ 0 → mean direction dominates).

Covariance rule (allows depression):

$$
\Delta w=\eta\,(x-\bar x)(y-\bar y)
$$
```

```{tip} Derivation
:class: dropdown
*Why does plain Hebb blow up along the top principal component?*
1. $\Delta\mathbf{w}=\eta\,\mathbf{x}\mathbf{x}^T\mathbf{w}$.
2. Average over inputs (slow learning): $\mathbb{E}[\Delta\mathbf{w}]=\eta\,C\mathbf{w}$ → linear ODE $\dot{\mathbf{w}}\propto C\mathbf{w}$.
3. Expand in eigenvectors of $C$: each component grows as $e^{\lambda_jt}$.
4. $C$ positive semidefinite → $\lambda_j\ge0$ → no component shrinks → $\|\mathbf{w}\|$ grows.
5. Largest $\lambda_1$ grows fastest → direction → $\mathbf{e}_1$.
```

```{attention} Q&A
:class: dropdown
*Why unstable?*

- Positive feedback: larger $w$ → larger $y$ → larger $\Delta w$.

*What does it learn?*

- The direction of max input variance → [PCA](../ml/unsupervised.md#pca) (w/ a stabilizer).
- Outer-product storage of patterns → [Hopfield Network](../dl/gen/ebm.md#hopfield-network).

*vs backprop?*

- No error signal → unsupervised. Cannot assign credit for a task objective → [Backprop in the Brain](#backprop-in-the-brain).

*Experimental counterpart?*

- **LTP / LTD** (long-term potentiation / depression): lasting ⬆️/⬇️ of synaptic strength after paired pre & post activity.
- NMDA-dependent plasticity: the NMDA receptor opens only w/ presynaptic input **AND** postsynaptic depolarization → a physical AND gate = Hebb's condition. (Not all LTP is NMDA-dependent.)
```

&nbsp;

#### Oja's Rule
- **What**: Hebbian rule + activity-dependent decay that keeps $\|\mathbf{w}\|\to1$ → neuron learns the first principal component. {cite:p}`oja1982simplified`
- **Why**: Plain Hebb blows up. Explicit normalization $\mathbf{w}/\|\mathbf{w}\|$ is non-local: each synapse would need all other weights.
- **How**: Subtract a decay term $\propto y^2w$ → each synapse needs only its own $w$, its input $x$ & the post activity $y$ → local.

```{note} Math
:class: dropdown
Rule:

$$
\Delta\mathbf{w}=\eta\,y\left(\mathbf{x}-y\,\mathbf{w}\right),\quad y=\mathbf{w}^T\mathbf{x}
$$

Properties:
- Average fixed points: $C\mathbf{w}=(\mathbf{w}^TC\mathbf{w})\,\mathbf{w}$ → unit-norm eigenvectors w/ $\lambda>0$, or any $\mathbf{w}\in\ker C$ (incl. 0).
- Only $\pm\mathbf{e}_1$ is attracting, if $\lambda_1>0$ is simple (averaged dynamics; a constant-step online rule fluctuates around it).
```

```{tip} Derivation
:class: dropdown
*Oja = normalized Hebb to 1st order:*
1. Normalized Hebb: $\mathbf{w}'=\frac{\mathbf{w}+\eta y\mathbf{x}}{\|\mathbf{w}+\eta y\mathbf{x}\|}$, w/ $\|\mathbf{w}\|=1$.
2. $\|\mathbf{w}+\eta y\mathbf{x}\|^{-1}\approx1-\eta y\,\mathbf{w}^T\mathbf{x}=1-\eta y^2$.
3. $\mathbf{w}'\approx(\mathbf{w}+\eta y\mathbf{x})(1-\eta y^2)=\mathbf{w}+\eta y(\mathbf{x}-y\mathbf{w})+O(\eta^2)$.

*Norm converges to 1:*
1. Average: $\dot{\mathbf{w}}\propto C\mathbf{w}-(\mathbf{w}^TC\mathbf{w})\mathbf{w}$.
2. $\frac{d}{dt}\|\mathbf{w}\|^2\propto2\,\mathbf{w}^TC\mathbf{w}\left(1-\|\mathbf{w}\|^2\right)$ → $\|\mathbf{w}\|\to1$.

*Only the top eigenvector is stable:*
1. Perturb $\mathbf{w}=\mathbf{e}_i+\epsilon\mathbf{e}_j$.
2. $\dot{\mathbf{w}}\approx\epsilon(\lambda_j-\lambda_i)\,\mathbf{e}_j+O(\epsilon^2)$.
3. Grows if $\lambda_j>\lambda_i$ → only $i=1$ (largest $\lambda$) is stable.
```

````{important} Code
:class: dropdown
```python
import torch

class Oja:
    def __init__(self, n_in, lr=0.01):
        self.w = torch.nn.functional.normalize(torch.randn(n_in), dim=0)
        self.lr = lr

    def step(self, x):
        y = self.w @ x                               ## linear neuron
        self.w += self.lr * y * (x - y * self.w)     ## Hebb (y x) minus local decay (y^2 w)
        return y

## Example: 2D data stretched along the 45° diagonal
torch.manual_seed(0)
A = torch.tensor([[2.0, 0.0], [0.0, 0.5]])               ## std 2 vs 0.5
R = torch.tensor([[0.7071, -0.7071], [0.7071, 0.7071]])  ## rotate by 45°
X = torch.randn(5000, 2) @ A @ R.T                       ## top eigenvector = [0.707, 0.707]
oja = Oja(2)
for x in X:
    oja.step(x)
print(oja.w, oja.w.norm())  ## ≈ ±[0.707, 0.707], norm ≈ 1
```
````

```{attention} Q&A
:class: dropdown
*More than 1 component?*

- Sanger's rule / Oja's subspace rule: several neurons + decorrelating terms → top-$k$ principal subspace.

*Biological?*

- Decay ∝ $y^2w$ ≈ heterosynaptic depression. Exact form not established.
```

&nbsp;

#### BCM Rule
- **What**: Post activity above a sliding threshold → potentiation; below → depression. Threshold tracks recent activity. {cite:p}`bienenstock1982theory`
- **Why**: Fixed-threshold Hebbian rules saturate or die → need stability **and** selectivity.
    - Target: development of orientation selectivity & ocular dominance in visual cortex.
- **How**:
    1. $\Delta w\propto x\cdot\phi(y,\theta_M)$: $\phi<0$ if $0<y<\theta_M$, $\phi>0$ if $y>\theta_M$.
    2. Neuron very active → $\theta_M$⬆️ → harder to potentiate. Neuron silent → $\theta_M$⬇️.
    3. → Stable, selective neuron: responds strongly to few input patterns.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\tau_\theta$: Threshold time constant.
- Misc:
    - $\theta_M$: Modification threshold.

Common form ($y$ dimensionless; w/ rates in Hz use $y^2/y_0$):

$$
\frac{d\mathbf{w}}{dt}=\eta\,\mathbf{x}\,y\,(y-\theta_M),\qquad\tau_\theta\frac{d\theta_M}{dt}=y^2-\theta_M
$$

- → $\theta_M\approx\mathbb{E}[y^2]$: grows faster than mean activity → stability.
- Needs $\theta_M$ to adapt fast enough vs learning (small $\tau_\theta$); a slow threshold → oscillation / instability.
- Forms of $\theta_M$ vary across papers. Required: superlinear in mean activity.
```

```{attention} Q&A
:class: dropdown
*Evidence for a sliding threshold?*

- Visual deprivation (dark rearing) lowers the potentiation threshold in visual cortex → sliding threshold.

*BCM vs Oja?*

- Oja → 2nd-order statistics (variance) → PCA.
- BCM → sensitive to higher-order statistics → selective, bimodal responses.

*Relation to STDP?*

- Spike-triplet STDP models reduce to BCM-like rate rules.
```

&nbsp;

### STDP
- **Name**: Spike-Timing-Dependent Plasticity {cite:p}`markram1997regulation,bi1998synaptic`
- **What**: Weight change set by the relative timing of pre & post spikes: pre → post → potentiation; post → pre → depression.
- **Why**: Rate-based Hebb ignores causality. STDP strengthens inputs that *predict* the post spike.
- **How**:
    1. Pair spikes: $\Delta t=t_\text{post}-t_\text{pre}$.
    2. $\Delta t>0$ (pre first, within ~20 ms) → potentiation; shrinks as $|\Delta t|$⬆️.
    3. $\Delta t<0$ (post first) → depression.
    4. Online: keep decaying traces of recent spikes → update at each spike.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $A_+,A_-$: Max potentiation / depression.
    - $\tau_+,\tau_-$: Window time constants.
- Misc:
    - $\Delta t=t_\text{post}-t_\text{pre}$: Spike-time difference.
    - $z_\text{pre},z_\text{post}$: Spike traces.

Pair-based window:

$$
\Delta w=\begin{cases}
A_+\,e^{-\Delta t/\tau_+}, & \Delta t>0\\
-A_-\,e^{\Delta t/\tau_-}, & \Delta t<0
\end{cases}
$$

- Common modeling choice: $\tau_+=\tau_-=20$ ms; $A_-/A_+$ slightly > 1 (e.g., 1.05).

Online (trace) form:

$$
\tau_+\dot z_\text{pre}=-z_\text{pre}+\tau_+\textstyle\sum\delta_\text{pre},\quad\tau_-\dot z_\text{post}=-z_\text{post}+\tau_-\textstyle\sum\delta_\text{post}
$$

- Post spike → $w\mathrel{+}=A_+z_\text{pre}$. Pre spike → $w\mathrel{-}=A_-z_\text{post}$.
- $\delta_\text{pre},\delta_\text{post}$: Spike impulses → each spike adds 1 to its trace.
```

````{important} Code
:class: dropdown
```python
import torch

class STDP:
    def __init__(self, n_pre, A_plus=0.01, A_minus=0.0105, tau_plus=20.0, tau_minus=20.0, dt=1.0):
        self.w = torch.full((n_pre,), 0.5)
        self.z_pre = torch.zeros(n_pre)       ## decaying memory of each input's recent spikes
        self.z_post = torch.tensor(0.0)       ## decaying memory of the post neuron's recent spikes
        self.A_plus, self.A_minus = A_plus, A_minus
        self.d_plus = torch.exp(torch.tensor(-dt / tau_plus))
        self.d_minus = torch.exp(torch.tensor(-dt / tau_minus))

    def step(self, pre, post):
        ## pre: (n_pre,) 0/1 spikes; post: 0/1 spike in this time step
        ## decay traces, update w from PAST spikes only, then register current spikes
        self.z_pre = self.z_pre * self.d_plus
        self.z_post = self.z_post * self.d_minus
        ## post spike → potentiate recently active inputs (pre before post)
        ## pre spike → depress if the post neuron fired recently (post before pre)
        self.w += self.A_plus * self.z_pre * post - self.A_minus * self.z_post * pre
        self.w.clamp_(0.0, 1.0)
        self.z_pre = self.z_pre + pre
        self.z_post = self.z_post + post
        return self.w

## Example: input 0 fires 5 ms before the post spike, input 1 fires 5 ms after; 100 pairings
stdp = STDP(2)
for _ in range(100):
    for t in range(100):                                          ## 100 ms per pairing
        pre = torch.tensor([float(t == 10), float(t == 20)])
        stdp.step(pre, float(t == 15))
print(stdp.w)  ## tensor([1., 0.]) ← per pairing ≈ +0.01 e^(-5/20) vs -0.0105 e^(-5/20) (+ tiny cross-pairing terms), clipped
```
````

```{attention} Q&A
:class: dropdown
*Why $A_-\tau_->A_+\tau_+$?*

- Uncorrelated pre & post → both orders equally likely → net change ∝ $A_+\tau_+-A_-\tau_-<0$ → depression.
- → Only inputs whose spikes reliably precede the post spike survive (predictive; causal or via common input) → competition & stability.

*Relation to rate-based Hebb?*

- Average pair STDP over Poisson spike trains → rate-based Hebbian rule.
- Triplet STDP → BCM-like rule.

*What does a neuron learn w/ STDP?*

- Strengthens the earliest predictive inputs → fires earlier → learns sequences & predicts its inputs.

*Limits?*

- Pair-based rule fails at high rates (triplet effects).
- Window shape varies w/ synapse type, dendritic location & neuromodulation (some windows are inverted).
- Mostly measured in slices. In vivo evidence is weaker.
```

&nbsp;

### Homeostatic Plasticity
- **What**: Slow negative feedback that keeps a neuron's average activity near a set point by rescaling its synapses or excitability. {cite:p}`turrigiano1998activity`
- **Why**: Hebbian plasticity = positive feedback → runaway excitation or silence.
- **How**:
    - **Synaptic scaling**: activity too low for hours–days → scale up all excitatory synapses by the same factor; too high → scale down.
        - Multiplicative → relative weights (stored information) preserved.
    - **Intrinsic plasticity**: adjust ion channels → shift the f–I curve.
    - Evidence (cultured neocortical neurons): block activity (TTX) → miniature EPSC amplitudes scale up multiplicatively. Block inhibition (bicuculline) → scale down.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $y^*$: Target activity (set point).
    - $\tau_H$: Homeostatic time constant (hours–days).
- Misc:
    - $\bar y$: Running average of $y$.

Multiplicative synaptic scaling:

$$
\tau_H\frac{d\mathbf{w}}{dt}=\mathbf{w}\,\frac{y^*-\bar y}{y^*}
$$

- All weights scale by the same factor → ratios $w_j/w_{j'}$ unchanged.
```

```{attention} Q&A
:class: dropdown
*Why multiplicative?*

- Preserves relative weights → memories survive the rescaling.

*Open problem?*

- Measured homeostasis (hours–days) looks too slow to stop fast Hebbian runaway (seconds–minutes).
- → Faster compensatory mechanisms likely exist (e.g., heterosynaptic plasticity, inhibitory plasticity).

*ML analogy?*

- Normalization layers & activity regularizers keep activation statistics in range → [Normalization](../dl/basics/module.md#normalization).
```

&nbsp;

## Reward Learning

### Rescorla–Wagner Model
- **What**: Each present cue's associative strength changes ∝ the shared prediction error: outcome − summed prediction.
- **Why**: Co-occurrence (Hebbian contiguity) fails to explain **blocking**.
    - Cue A already predicts reward → add cue B, still rewarded → B learns ~nothing.
    - → Learning is driven by surprise, ❌co-occurrence.
- **How**:
    1. Trial: prediction = sum of strengths of all present cues.
    2. Error = outcome $\lambda$ − prediction.
    3. Each present cue's strength moves ∝ error.
    - = delta rule (LMS) on binary cue features.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{u}\in\{0,1\}^K$: Cue-presence vector ($K$ cues).
    - $\lambda$: Outcome (reward magnitude; 0 if none).
- Params:
    - $\mathbf{v}\in\mathbb{R}^K$: Associative strengths.
- Hyperparams:
    - $\alpha_X$: Salience of cue $X$.
    - $\beta$: Outcome learning rate.

Update for each present cue $X$:

$$
\Delta v_X=\alpha_X\,\beta\left(\lambda-\sum_{i\in\text{present}}v_i\right)
$$

Vector form (equal saliences, $\eta=\alpha\beta$):

$$
\mathbf{v}\leftarrow\mathbf{v}+\eta\left(\lambda-\mathbf{v}^T\mathbf{u}\right)\mathbf{u}
$$

- = SGD on $\frac{1}{2}(\lambda-\mathbf{v}^T\mathbf{u})^2$.
```

````{important} Code
:class: dropdown
```python
import torch

class RescorlaWagner:
    def __init__(self, n_cues, lr=0.2):
        self.v = torch.zeros(n_cues)    ## associative strength per cue
        self.lr = lr

    def update(self, u, lam):
        delta = lam - self.v @ u         ## shared prediction error of all present cues
        self.v += self.lr * delta * u    ## only present cues (u = 1) change
        return delta

## Example: blocking. Phase 1: A → reward. Phase 2: A+B → reward.
rw = RescorlaWagner(2)
for _ in range(100):
    rw.update(torch.tensor([1.0, 0.0]), lam=1.0)
for _ in range(100):
    rw.update(torch.tensor([1.0, 1.0]), lam=1.0)
print(rw.v)  ## ≈ tensor([1., 0.]) → B blocked (control: A+B from scratch → [0.5, 0.5])
```
````

```{attention} Q&A
:class: dropdown
*What does it miss?*

- Time inside a trial: when the cue & reward occur, 2nd-order conditioning → TD extends it → [Reward Prediction Error](#reward-prediction-error).
- Latent inhibition: pre-exposure to a cue w/o outcome slows later learning; RW predicts no effect.
- Spontaneous recovery after extinction: RW treats extinction as unlearning → recovery impossible.

*Relation to ML?*

- = Widrow–Hoff delta rule = LMS = online linear regression.
- Error-driven learning = the same principle as gradient descent on squared error.
```

&nbsp;

### Reward Prediction Error
- **What**: Phasic dopamine firing ≈ temporal-difference (TD) error: (reward + predicted future value) − current predicted value. {cite:p}`schultz1997neural`
- **Why**: Links a learning algorithm ([TD Learning](../rl/value.md#td-learning)) to a measured neural signal → explains how animals learn to predict & seek reward.
- **How**: Midbrain dopamine neurons (VTA, SNc):
    1. Unexpected reward → burst.
    2. After learning: cue predicts reward → burst moves to the cue; reward itself → no response.
    3. Predicted reward omitted → dip below baseline at the expected time.
    4. Dopamine broadcast to striatum & cortex → gates plasticity of recently active synapses → [Three-Factor Learning Rule](#three-factor-learning-rule).

```{note} Math
:class: dropdown
Notations:
- IO:
    - $r_t$: Reward at step $t$.
    - $s_t$: State at step $t$.
- Params:
    - $V(s)$: Predicted value.
- Hyperparams:
    - $\gamma$: Discount factor.

TD error:

$$
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)
$$

Value update:

$$
V(s_t)\leftarrow V(s_t)+\eta\,\delta_t
$$
```

```{note} Example
:class: dropdown
Cue, then reward 1 s later. Reward = 1. $\gamma\approx1$. Assume $V$(pre-cue) ≈ 0 (e.g., long, variable gaps between trials).

| Condition | $\delta$ at cue | $\delta$ at reward time | Dopamine |
|:--|:--|:--|:--|
| Before learning ($V$ = 0 everywhere) | 0 | $1+0-0=1$ | Burst at reward |
| After learning ($V$ = 1 from cue to reward) | $0+1-0=1$ | $1+0-1=0$ | Burst at cue only |
| After learning, reward omitted | 1 | $0+0-1=-1$ | Burst at cue, dip at expected reward |
```

```{attention} Q&A
:class: dropdown
*Correlation or cause?*

- Optogenetically timed dopamine pulses can substitute for natural prediction errors during conditioning → causal role.

*Distributional code?*

- Dopamine neurons differ in how strongly they scale positive vs negative errors → different "optimism".
- → Consistent w/ a population code for the reward distribution: asymmetric TD learning → expectiles. {cite:p}`dabney2020distributional`
- Related: [C51](../rl/value.md#c51) (categorical atoms, a different parameterization).

*Is all dopamine RPE?*

- ❌. Dopamine also tracks novelty, salience, movement & value ramps. Interpretation of ramps is contested.

*Model-free only?*

- ❌. Dopamine errors also reflect inferred hidden states & model-based expectations.

*Relation to Rescorla–Wagner?*

- RW: 1 update per trial, target = outcome.
- TD: 1 update per time step, target = $r_t+\gamma V(s_{t+1})$ (bootstrapped) → explains the shift of the error to the earliest predictive cue.

*Where are the actor & the critic?*

- Proposal: ventral striatum ≈ critic (learns $V$), dorsal striatum ≈ actor (learns the policy); dopamine $\delta$ trains both → [Actor–critic](../rl/policy.md#actor-critic).
- Go/NoGo models: dopamine bursts strengthen the "Go" (D1) pathway; dips strengthen the "NoGo" (D2) pathway → separate learning from + & − errors.
```

&nbsp;

### Three-Factor Learning Rule
- **What**: $\Delta w$ = (pre–post coincidence, held as an eligibility trace) × (neuromodulatory 3rd factor). {cite:p}`fremaux2016neuromodulated`
- **Why**: Reward arrives seconds after the action → the Hebbian coincidence is long gone → **distal reward problem** (temporal credit assignment).
- **How**:
    1. Pre–post coincidence → no weight change yet; tags the synapse w/ an **eligibility trace** $e$.
    2. $e$ decays over ~seconds.
    3. Neuromodulator $M$ arrives (dopamine = RPE; ACh / NE = surprise, novelty) → $\Delta w=\eta\,M\,e$.
    4. → Only synapses active shortly before the outcome change.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\tau_e$: Eligibility time constant.
- Misc:
    - $e$: Eligibility trace.
    - $M(t)$: Third factor (e.g., RPE $\delta$).
    - $H(t)$: Hebbian / STDP coincidence events (impulses; each tags the synapse).
    - $\bar y=\mathbb{E}[y\mid\mathbf{x}]$: Noise-free post activity.
    - $R$: Reward. $b$: Reward baseline (independent of the noise $\xi$ given $\mathbf{x}$).

Rule:

$$
\frac{de}{dt}=-\frac{e}{\tau_e}+H(t),\qquad\frac{dw}{dt}=\eta\,M(t)\,e(t)
$$

Link to REINFORCE (node perturbation, 1-step reward): $y=\mathbf{w}^T\mathbf{x}+\xi$, $\xi\sim\mathcal{N}(0,\sigma^2)$, $H=(y-\bar y)\,\mathbf{x}$, $M=R-b$:

$$
\mathbb{E}[\Delta\mathbf{w}]\propto\nabla_\mathbf{w}\,\mathbb{E}[R]
$$
```

```{tip} Derivation
:class: dropdown
1. Policy: $y\sim\mathcal{N}(\mathbf{w}^T\mathbf{x},\sigma^2)$ → $\nabla_\mathbf{w}\log p(y\mid\mathbf{x})=\frac{(y-\mathbf{w}^T\mathbf{x})}{\sigma^2}\mathbf{x}=\frac{(y-\bar y)}{\sigma^2}\mathbf{x}$.
2. [REINFORCE](../rl/policy.md#reinforce): $\nabla_\mathbf{w}\mathbb{E}[R]=\mathbb{E}\left[(R-b)\,\nabla_\mathbf{w}\log p(y\mid\mathbf{x})\right]$.
3. → $\mathbb{E}\left[(R-b)(y-\bar y)\mathbf{x}\right]\propto\nabla_\mathbf{w}\mathbb{E}[R]$.
4. $(y-\bar y)\mathbf{x}$ = Hebbian term (pre × post fluctuation). $R-b$ = third factor.
```

````{important} Code
:class: dropdown
```python
import torch

class ThreeFactor:
    def __init__(self, n_in, tau_e=1000.0, lr=0.1, dt=1.0):
        self.w = torch.zeros(n_in)
        self.e = torch.zeros(n_in)                                   ## eligibility trace = synaptic "tag"
        self.decay = torch.exp(torch.tensor(-dt / tau_e))
        self.lr = lr

    def step(self, x, y, y_mean, M=0.0):
        ## Hebbian coincidence (pre x post fluctuation) only tags the synapse (impulse into the trace)
        self.e = self.e * self.decay + (y - y_mean) * x
        ## 3rd factor (e.g., dopamine RPE) turns tags into weight changes
        self.w += self.lr * M * self.e
        return self.w

## Example: input 0 co-active w/ a post burst at t = 0; reward (M = 1) arrives 500 ms later
tf = ThreeFactor(2)
tf.step(torch.tensor([1.0, 0.0]), y=1.0, y_mean=0.0)   ## coincidence on synapse 0 only
for _ in range(499):
    tf.step(torch.zeros(2), y=0.0, y_mean=0.0)          ## tag decays, w unchanged
print(tf.step(torch.zeros(2), y=0.0, y_mean=0.0, M=1.0))  ## ≈ tensor([0.0607, 0.]) = 0.1 * exp(-0.5)
```
````

```{attention} Q&A
:class: dropdown
*Biological evidence?*

- Eligibility traces lasting ~seconds, converted into plasticity by neuromodulators, are reported in striatum, cortex & hippocampus.

*ML equivalents?*

- [Eligibility Traces](../rl/value.md#eligibility-traces) in TD($\lambda$).
- REINFORCE (Derivation).
- Online training of recurrent spiking networks w/ local traces × broadcast learning signals.

*Limits?*

- 1 scalar $M$ for millions of synapses → gradient estimate variance ⬆️ w/ #neurons → slow learning.
- → Motivates vector-valued error signals → [Backprop in the Brain](#backprop-in-the-brain).
```

&nbsp;

### Neuromodulation as Metaparameters
- **What**: Hypothesis: dopamine carries the TD error; serotonin, noradrenaline & acetylcholine each set 1 global RL metaparameter. {cite:p}`doya2002metalearning`
- **Why**: RL needs global settings (learning rate, exploration, horizon) that must change w/ context.
    - Neuromodulators reach large brain areas at once & change slowly → natural carriers for global settings.
- **How**: Doya's mapping.
    - **Dopamine** → TD error $\delta$.
    - **Serotonin** → discount factor $\gamma$ (patience).
    - **Noradrenaline** → inverse temperature $\beta$ (exploration vs exploitation).
    - **Acetylcholine** → learning rate $\alpha$ (memory update speed).

```{note} Math
:class: dropdown
Notations:
- IO:
    - $r_t$: Reward.
    - $s_t,a_t$: State & action.
- Params:
    - $Q(s,a)$: Action values.
- Hyperparams:
    - $\alpha$: Learning rate (ACh).
    - $\gamma$: Discount factor (5-HT).
    - $\beta$: Inverse temperature (NE).

Where each metaparameter enters:

$$
\begin{align*}
\delta_t&=r_t+\gamma\max_{a'}Q(s_{t+1},a')-Q(s_t,a_t) && \text{(DA signals }\delta\text{)}\\
Q(s_t,a_t)&\leftarrow Q(s_t,a_t)+\alpha\,\delta_t\\
\pi(a\mid s)&=\frac{e^{\beta Q(s,a)}}{\sum_{a'}e^{\beta Q(s,a')}}
\end{align*}
$$

- Policy → [Boltzmann Exploration](../rl/intro.md#boltzmann-exploration).
```

```{attention} Q&A
:class: dropdown
*Status?*

- Proposal, ❌established. Dopamine ↔ $\delta$ has the strongest support. Serotonin ↔ $\gamma$ is the most contested.

*Alternative: uncertainty signals?*

- ACh signals **expected** uncertainty (known unreliability of cues) → rely less on top-down predictions.
- NE signals **unexpected** uncertainty (the world changed) → reset & relearn. {cite:p}`yu2005uncertainty`

*ML analogy?*

- Meta-learning & adaptive hyperparameters: LR schedules, adaptive optimizers, temperature annealing, discount tuning.
```

&nbsp;

## Credit Assignment

### Marr–Albus Model
- **What**: Cerebellum as a supervised learner: sparse expansion layer (granule cells) + perceptron-like readout (Purkinje cells) trained by an error signal (climbing fibers).
- **Why**: Fast, precise movement needs learned sensorimotor maps → needs a teacher signal.
    - The cerebellum's very regular circuit suggested 1 learning algorithm; theory predicted the plasticity before it was measured.
- **How**:
    1. Context input (mossy fibers) → expanded into a huge, sparse granule-cell code → many features, easier linear separation.
    2. Each Purkinje cell sums ~10⁵ parallel-fiber synapses → linear readout.
    3. Typically 1 climbing fiber per adult Purkinje cell → teaching / error signal.
    4. Parallel fiber active + climbing-fiber signal → that synapse changes → output corrected.
    - Marr (1969) proposed potentiation. Albus (1971) proposed depression. Experiments (Ito, 1982) found long-term depression.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{m}$: Mossy-fiber input.
    - $\mathbf{g}$: Granule-cell activity (expanded).
    - $y$: Purkinje output.
    - $y^*$: Desired output.
- Params:
    - $J$: Fixed expansion weights ($\dim\mathbf{g}\gg\dim\mathbf{m}$).
    - $\mathbf{w}$: Parallel fiber → Purkinje weights.
- Hyperparams:
    - $\theta$: Granule threshold (sets sparsity).

Model:

$$
\mathbf{g}=\left[J\mathbf{m}-\theta\right]_+,\qquad y=\mathbf{w}^T\mathbf{g}
$$

Training (LMS; computational surrogate: climbing fiber ≈ signed error $e=y-y^*$):

$$
\Delta\mathbf{w}=-\eta\,e\,\mathbf{g}
$$

- $e>0$ (output too large) → active synapses depress.
```

````{important} Code
:class: dropdown
```python
import torch

class MarrAlbus:
    def __init__(self, n_in, n_granule=500, theta=1.0, lr=0.01):
        self.J = torch.randn(n_granule, n_in)     ## fixed random expansion (mossy → granule)
        self.w = torch.zeros(n_granule)           ## plastic parallel-fiber → Purkinje weights
        self.theta, self.lr = theta, lr

    def step(self, m, y_target):
        g = (self.J @ m - self.theta).clamp(min=0)   ## sparse, high-dim granule code
        e = self.w @ g - y_target                    ## climbing-fiber error signal
        self.w -= self.lr * e * g                    ## LMS: only active synapses change
        return e ** 2

## Example: learn y = m1 * m2 (not linear in m) on [-1, 1]^2
torch.manual_seed(0)
model = MarrAlbus(2)
M = torch.rand(20_000, 2) * 2 - 1
losses = torch.stack([model.step(m, m[0] * m[1]) for m in M])
print(losses[:1000].mean(), losses[-1000:].mean())  ## untrained MSE = E[(m1 m2)^2] = 1/9 ≈ 0.11; late average falls well below
```
````

```{attention} Q&A
:class: dropdown
*Why expand?*

- More dimensions → more input patterns become linearly separable (Cover's theorem) → 1 linear readout learns more maps.
- Sparse code → fewer overlaps between patterns → less interference.

*ML analogy?*

- Random features / kernel machine / extreme learning machine: fixed random hidden layer + trained linear readout.
- Same design as reservoir computing → [Reservoir Computing](network.md#reservoir-computing).

*Limit?*

- Only 1 plastic layer → credit assignment is easy (the error is local). The hard problem = many plastic layers → [Backprop in the Brain](#backprop-in-the-brain).
```

&nbsp;

### Backprop in the Brain
- **What**: The question whether cortex approximates backprop-like credit assignment, & the proposed mechanisms. {cite:p}`lillicrap2020backpropagation`
- **Why**:
    - Deep nets trained by backprop are the best current models of sensory cortex → [Goal-Driven Modeling](theory.md#goal-driven-modeling).
    - Hebbian & scalar 3-factor rules → slow for deep, wide networks.
- **How**: List backprop's biologically implausible requirements → find local substitutes.
    1. **Weight transport**: backward pass uses $W^T$ → feedback synapses must mirror forward weights. → [Feedback Alignment](#feedback-alignment).
    2. **Separate error signals**: errors must not corrupt activities → segregated compartments (apical = feedback, basal = feedforward) → [Dendritic Computation](neuron.md#dendritic-computation).
    3. **Signed, precise errors**: rates are ≥ 0 → encode errors as **differences of activity** (across time, phases, or compartments).
    4. **Separate phases**: frozen forward activity during the backward sweep → continuous-time dynamics that settle.
    - Lillicrap et al. name the shared idea **NGRAD**: neural gradient representation by activity differences.

```{dropdown} Table: Backprop Problems & Proposed Solutions
| Backprop requirement | Biological problem | Proposed solutions |
|:--|:--|:--|
| Backward weights = $W^T$ | Weight transport | Feedback alignment, sign symmetry, learned feedback weights |
| Error separate from activity | Same neurons carry both | Apical vs basal dendrites, bursts vs single spikes, separate error neurons |
| Signed errors | Rates ≥ 0 | Activity differences over time / phases; error neurons w/ baseline |
| Forward & backward phases | No global clock | Equilibrium-based methods, predictive coding dynamics |
| Exact derivatives | Noisy, spiking units | Surrogate / approximate derivatives |
```

```{attention} Q&A
:class: dropdown
*Does the brain do backprop?*

- Unknown. No direct evidence of exact backprop.
- Several local approximations exist. Which one (if any) the brain uses = open & contested.

*Predictive coding & backprop?*

- Predictive coding networks w/ local Hebbian plasticity can approximate backprop gradients under specific conditions. {cite:p}`whittington2017approximation`
- → [Predictive Coding](theory.md#predictive-coding).

*Why should ML care?*

- Local learning rules → on-chip learning for neuromorphic & analog hardware; no global backward pass.
```

&nbsp;

#### Feedback Alignment
- **What**: Backprop w/ fixed random feedback matrices $B$ instead of $W^T$. {cite:p}`lillicrap2016random`
- **Why**: Removes weight transport: the feedback path needs no copy of the forward weights.
- **How**:
    1. Forward pass as usual.
    2. Send the output error backward through fixed random $B$.
    3. Update each forward weight w/ (local error × presynaptic activity).
    4. Forward weights come to **align** w/ $B^T$ → random feedback becomes a useful error signal.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{h}$: Hidden activity.
    - $\hat{\mathbf{y}}$: Output.
    - $\mathbf{y}^*$: Target.
    - $\mathbf{e}=\hat{\mathbf{y}}-\mathbf{y}^*$: Output error.
- Params:
    - $W_1,W_2$: Forward weights.
    - $B$: Fixed random feedback matrix (shape of $W_2^T$).
- Hyperparams:
    - $\phi$: Hidden nonlinearity.

Forward:

$$
\mathbf{h}=\phi(W_1\mathbf{x}),\quad\hat{\mathbf{y}}=W_2\mathbf{h}
$$

Backward:

$$
\boldsymbol{\delta}_h^\text{BP}=\left(W_2^T\mathbf{e}\right)\odot\phi'(W_1\mathbf{x}),\qquad\boldsymbol{\delta}_h^\text{FA}=\left(B\,\mathbf{e}\right)\odot\phi'(W_1\mathbf{x})
$$

Updates:

$$
\Delta W_2=-\eta\,\mathbf{e}\,\mathbf{h}^T,\quad\Delta W_1=-\eta\,\boldsymbol{\delta}_h^\text{FA}\,\mathbf{x}^T
$$

- Hidden-layer FA update within 90° of backprop's iff $\mathbf{e}^TW_2D^2B\,\mathbf{e}>0$, $D=\text{diag}\left(\phi'(W_1\mathbf{x})\right)$. Output layer uses the exact gradient.
- **Direct feedback alignment**: send $\mathbf{e}$ to every layer through its own random matrix.
```

````{important} Code
:class: dropdown
```python
import torch

class FeedbackAlignmentMLP:
    def __init__(self, n_in, n_hid, n_out, lr=0.005):
        self.W1 = torch.randn(n_hid, n_in) / n_in ** 0.5
        self.W2 = torch.randn(n_out, n_hid) / n_hid ** 0.5
        self.B = torch.randn(n_hid, n_out) / n_out ** 0.5    ## fixed random feedback, replaces W2.T
        self.lr = lr

    def step(self, x, y_target):
        h = torch.tanh(self.W1 @ x)
        e = self.W2 @ h - y_target                           ## output error
        delta_h = (self.B @ e) * (1 - h ** 2)                ## backprop would use W2.T @ e
        self.W2 -= self.lr * torch.outer(e, h)               ## local: error x presynaptic activity
        self.W1 -= self.lr * torch.outer(delta_h, x)
        return (e ** 2).sum()

## Example: learn a random linear map 10 → 2
torch.manual_seed(0)
T = torch.randn(2, 10)
net = FeedbackAlignmentMLP(10, 32, 2)
losses = torch.stack([net.step(x, T @ x) for x in torch.randn(5000, 10)])
print(losses[:200].mean(), losses[-200:].mean())   ## loss typically drops a lot
print(torch.nn.functional.cosine_similarity(net.W2.flatten(), net.B.T.flatten(), dim=0))  ## typically > 0: proxy for W2 aligning w/ B.T
```
````

```{attention} Q&A
:class: dropdown
*Why does random feedback work?*

- $W_1$ updates push hidden activity in directions that $B\mathbf{e}$ can exploit.
- $W_2$ then aligns w/ $B^T$ → FA updates fall within 90° of true gradients.

*Limits?*

- Matches backprop on small tasks. Degrades on deep conv nets & large-scale vision (e.g., ImageNet).
- Sign-symmetric feedback (same signs as $W^T$, random magnitudes) closes much of the gap.

*Biologically complete?*

- ❌. Removes weight transport only. Still needs a separate, signed error pathway.
```

&nbsp;

## Memory

### Complementary Learning Systems
- **What**: 2 learning systems: fast hippocampus stores episodes; slow neocortex extracts structure from interleaved replay. {cite:p}`mcclelland1995why`
- **Why**: Trade-off in 1 network.
    - Fast learning of new items in a distributed network → **catastrophic interference** w/ old knowledge → [Catastrophic Forgetting](../dl/train/issues.md#catastrophic-forgetting).
    - Slow interleaved learning → good generalization, but no one-shot learning.
- **How**:
    1. Hippocampus: sparse, separated codes → fast one-shot storage w/o interference.
    2. Offline (sleep, rest): hippocampus **replays** episodes.
        - Cells co-active during experience reactivate together in later sleep. {cite:p}`wilson1994reactivation`
        - Replayed sequences are often time-compressed (Lee & Wilson, 2002).
    3. Neocortex learns slowly from replay interleaved w/ old knowledge → integrates w/o forgetting.
    4. Over time, some memories become supported by neocortex (**systems consolidation**). Detailed episodic memories may stay hippocampus-dependent (contested).

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\theta$: Slow (neocortical) network params.
- Misc:
    - $\mathcal{D}_\text{old}$: Previously learned data.
    - $\mathcal{D}_\text{new}$: New episodes.
    - $\mathcal{L}$: Loss.

Focused learning (new data only):

$$
\theta\leftarrow\theta-\eta\,\nabla_\theta\,\mathbb{E}_{\mathcal{D}_\text{new}}[\mathcal{L}]
$$

- Steps ignore $\mathcal{D}_\text{old}$ → overwrite shared weights → interference.

Interleaved learning (replay):

$$
\theta\leftarrow\theta-\eta\,\nabla_\theta\,\mathbb{E}_{\mathcal{D}_\text{old}\cup\mathcal{D}_\text{new}}[\mathcal{L}]
$$

- Hippocampal replay supplies samples of $\mathcal{D}_\text{new}$ (& reactivated old memories) → SGD on a replay-weighted mix of old & new data (= uniform over the union only if the sampling matches).
```

```{attention} Q&A
:class: dropdown
*Why sparse codes in the hippocampus?*

- **Pattern separation**: similar episodes → dissimilar codes → less interference.
- **Pattern completion**: recurrent network → partial cue → full memory → [Attractor Network](network.md#attractor-network).

*Is replay causal?*

- Disrupting sharp-wave ripples (when most replay occurs) during post-training sleep impairs spatial learning in rats → ripple-associated activity is needed.

*Forward or reverse?*

- Both. Reverse replay occurs at reward sites → resembles propagating value backward along a path.

*ML analogs?*

- Experience replay in [DQN](../rl/value.md#dqn) & [Prioritized Replay](../rl/value.md#prioritized-replay); generative replay in continual learning.
- Retrieval-augmented LMs: fast external memory + slow weights.

*Revisions?*

- Information consistent w/ an existing schema can enter neocortex quickly → "slow cortex" is not absolute.
```

&nbsp;
