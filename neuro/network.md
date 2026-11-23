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
# Network Dynamics
Dynamics & computation in recurrent networks of model neurons. Single units → [Neuron Models](neuron.md).

Notations:
- $N$: #units
- $\mathbf{r}\in\mathbb{R}^N$: Firing rates (activations)
- $W\in\mathbb{R}^{N\times N}$: Recurrent weight matrix
- $\mathbf{h}$: External input
- $\phi$: Rate nonlinearity
- $\tau$: Time constant

Overrides of global notation: $i,j$ = unit indices; $k$ = eigenmode index. $I$ = inhibitory activity (Wilson–Cowan); identity matrix = $I_N$.

&nbsp;

## Population Models

### Wilson–Cowan Model
- **What**: 2 coupled rate equations for the mean activity of an excitatory (E) & an inhibitory (I) population.
- **Why**: Simplest mean-field model of a local circuit. Explains steady states, bistability & oscillations from E–I interaction.
- **How**:
    1. E excites itself & I. I inhibits E (& itself).
    2. Each population relaxes toward a sigmoid of its net input.
    3. Strong E self-excitation + slow inhibitory feedback → E rises, I catches up, E falls, I decays → **limit cycle** (oscillation).
    4. Other params → 1 stable state, or 2 (bistability, hysteresis).

```{note} Math
:class: dropdown
Notations:
- IO:
    - $E,I$: Mean activity of the E & I populations.
    - $h_E,h_I$: External inputs.
- Params:
    - $w_{EE},w_{EI},w_{IE},w_{II}\ge0$: Coupling strengths ($w_{XY}$: from $Y$ to $X$).
- Hyperparams:
    - $\tau_E,\tau_I$: Time constants.
    - $f$: Sigmoid.

Model (common simplified form):

$$
\begin{align*}
\tau_E\frac{dE}{dt}&=-E+f\left(w_{EE}E-w_{EI}I+h_E\right)\\
\tau_I\frac{dI}{dt}&=-I+f\left(w_{IE}E-w_{II}I+h_I\right)
\end{align*}
$$

- Original 1972 form also has refractory factors, e.g., $(1-r_EE)$ multiplying $f$.

Jacobian at a fixed point:

$$
J=\begin{bmatrix}\dfrac{-1+w_{EE}f'_E}{\tau_E} & -\dfrac{w_{EI}f'_E}{\tau_E}\\[3mm]\dfrac{w_{IE}f'_I}{\tau_I} & \dfrac{-1-w_{II}f'_I}{\tau_I}\end{bmatrix}
$$

- $f'_E,f'_I$: Sigmoid slope at the fixed point.
- Locally (exponentially) stable iff $\text{tr}\,J<0$ & $\det J>0$.
- Eigenvalues cross the imaginary axis at $\text{tr}\,J=0$, $\det J>0$ → Hopf bifurcation (oscillation onset), given generic transversality & nonzero 1st Lyapunov coefficient.
- $\text{tr}\,J>0$ needs $w_{EE}f'_E>1$: E alone must be unstable.
```

````{important} Code
:class: dropdown
```python
import torch

class WilsonCowan:
    def __init__(self, wEE=10.0, wEI=10.0, wIE=10.0, wII=0.0, hE=0.0, hI=-5.0, tauE=1.0, tauI=2.0):
        self.w = (wEE, wEI, wIE, wII)
        self.h = (hE, hI)
        self.tau = (tauE, tauI)

    def run(self, E=0.1, I=0.1, T=100.0, dt=0.01):
        wEE, wEI, wIE, wII = self.w
        trace = []
        for _ in range(int(T / dt)):
            ## each population relaxes toward a sigmoid of its net input (Euler)
            dE = (-E + torch.sigmoid(torch.tensor(wEE * E - wEI * I + self.h[0]))) / self.tau[0]
            dI = (-I + torch.sigmoid(torch.tensor(wIE * E - wII * I + self.h[1]))) / self.tau[1]
            E, I = E + dt * dE.item(), I + dt * dI.item()
            trace.append(E)
        return torch.tensor(trace)

## Example: same circuit, slow vs fast inhibition; unique fixed point E = I = 0.5
for tauI in [2.0, 0.5]:
    E = WilsonCowan(tauI=tauI).run()
    print(tauI, E[-5000:].min(), E[-5000:].max())
## tauI = 2.0: tr J = 1.5 - 0.5 > 0 → unstable spiral → limit cycle (min < 0.5 < max)
## tauI = 0.5: tr J = 1.5 - 2.0 < 0 → stable spiral → min ≈ max ≈ 0.5
```
````

```{attention} Q&A
:class: dropdown
*Why must inhibition be slow for oscillation?*

- Fast I tracks E instantly → acts like extra self-inhibition of E → $\text{tr}\,J<0$ → stable, if inhibitory feedback is strong enough ($\det J>0$). Weak inhibition → saddle; speed cannot fix it.
- Slow I → delayed negative feedback → overshoot → oscillation.
- E–I loops = a standard model of gamma rhythms.

*Inhibition-stabilized network (ISN)?*

- E alone unstable ($w_{EE}f'_E>1$), stabilized by inhibition.
- Paradoxical effect: ⬆️ external drive to I → steady-state I rate ⬇️ ← E drops, I loses excitation.
- Reported in several cortical areas (e.g., Sanzeni et al., 2020) → test for strong recurrent excitation.

*Pros & cons?*

- ✅ 2D → full phase-plane analysis.
- ❌ No spikes, no spatial structure (spatial extension: neural field models).
```

&nbsp;

## Attractors

### Attractor Network
- **What**: Recurrent network whose stable activity states (attractors) store memories or variables; activity flows to the attractor of its starting basin.
- **Why**: Memory w/o a buffer: persistent activity & pattern completion emerge from recurrent dynamics.
- **How**:
    1. Recurrent weights shape a landscape of basins.
    2. Input sets the initial state.
    3. Dynamics flow to the attractor of that basin.
    4. Attractor persists after input stops → memory. Partial cue → full pattern → **pattern completion**.
    - Discrete memories → point attractors → [Hopfield Network](../dl/gen/ebm.md#hopfield-network).
    - Continuous variables → [Continuous Attractor](#continuous-attractor).
    - General theory → [Attractor](../math/diffeq.md#attractor).

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $x_i$: Input current of unit $i$; $r_i=\phi(x_i)$.
    - $L$: Lyapunov (energy) function.

Continuous Hopfield network:

$$
\tau\frac{dx_i}{dt}=-x_i+\sum_jW_{ij}\phi(x_j)+h_i
$$

Symmetric $W$, constant $\mathbf{h}$, $\phi'>0$ → Lyapunov function:

$$
L=-\frac{1}{2}\sum_{i,j}W_{ij}r_ir_j-\sum_ih_ir_i+\sum_i\int_{\rho_0}^{r_i}\phi^{-1}(\rho)\,d\rho
$$

- $\frac{dL}{dt}\le0$ → only fixed-point attractors, no cycles or chaos.
- $\rho_0$: Any reference point in the range of $\phi$.
- Binary Hopfield network, Hebbian storage of random patterns: capacity ≈ $0.138N$ (approximate retrieval, $N\to\infty$).
```

```{tip} Derivation
:class: dropdown
*Why does $L$ never increase?*
1. $\frac{\partial L}{\partial r_i}=-\sum_jW_{ij}r_j-h_i+\phi^{-1}(r_i)=-\sum_jW_{ij}r_j-h_i+x_i$ (uses $W$ symmetric).
2. → $\frac{\partial L}{\partial r_i}=-\tau\frac{dx_i}{dt}$.
3. $\frac{dL}{dt}=\sum_i\frac{\partial L}{\partial r_i}\frac{dr_i}{dt}=-\tau\sum_i\frac{dx_i}{dt}\phi'(x_i)\frac{dx_i}{dt}=-\tau\sum_i\phi'(x_i)\left(\frac{dx_i}{dt}\right)^2\le0$.
4. $\phi'>0$ → equality only at fixed points.
```

```{dropdown} Table: Attractor Types
| Attractor | Stored quantity | Example model / system |
|:--|:--|:--|
| Point (discrete set) | Discrete memories, decisions | Hopfield network, winner-take-all choice |
| Line | 1 analog value | Oculomotor integrator (eye position) |
| Ring | 1 angle | Head direction |
| Plane / torus | 2D position | Grid cells (torus) |
| Limit cycle | Rhythm | Locomotion pattern generators, Wilson–Cowan oscillation |
| Chaotic | ❌stable memory; rich dynamics | Random networks w/ $g>1$ |
```

```{attention} Q&A
:class: dropdown
*Evidence?*

- Persistent delay-period activity in prefrontal & parietal cortex → [Working Memory](systems.md#working-memory).
- Fly head-direction circuit: a bump of activity moves around a ring-shaped structure, as a ring attractor predicts (Kim et al., 2017).

*Symmetric weights are unrealistic?*

- Yes (Dale's principle, asymmetric connections). Asymmetric networks can still have attractors, but no energy guarantee → cycles & chaos possible.

*Relation to transformers?*

- Modern Hopfield networks: exponential capacity; 1 update step = attention → [Modern Hopfield Network](../dl/gen/ebm.md#modern-hopfield-network).

*Attractors in AC theories?*

- Stable, self-sustaining states ≈ "ignition" in global workspace theory → [GNWT](../ac/theories.md#global-neuronal-workspace-theory-gnwt).
```

&nbsp;

### Continuous Attractor
- **What**: Attractor network w/ a continuum of stable states (line, ring, torus) → stores & integrates a continuous variable.
- **Why**: Hold analog values (eye position, heading, location) for seconds w/o input.
    - Add a velocity input → the network integrates it (e.g., path integration).
- **How**:
    1. Symmetric, translation-invariant connections (e.g., on a ring: local excitation + broad inhibition) → if 1 stable bump exists, every shifted copy is too (exact only for a continuum; finite rings approximate it).
    2. Along the manifold: no restoring force (neutral direction). Off the manifold: perturbations decay.
    3. Velocity input pushes the bump along → integration.
    - Linear version: 1 eigenvalue of symmetric $W$ = 1, all others < 1 → **line attractor** = perfect integrator. Proposed for the oculomotor integrator. {cite:p}`seung1996brain`
    - Ring → [Ring Attractor](../math/geo/topology.md#ring-attractor). Torus → [Toroidal Grid-Cell Manifold](../math/geo/topology.md#toroidal-grid-cell-manifold).

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\lambda_k,\mathbf{e}_k$: Eigenvalues & orthonormal eigenvectors of symmetric $W$.
    - $c_k=\mathbf{e}_k^T\mathbf{r}$: Activity along mode $k$.

Linear network:

$$
\tau\frac{d\mathbf{r}}{dt}=-\mathbf{r}+W\mathbf{r}+\mathbf{h}
$$

Per mode:

$$
\tau\frac{dc_k}{dt}=-(1-\lambda_k)\,c_k+\mathbf{e}_k^T\mathbf{h}\ \Rightarrow\ \tau_k^\text{eff}=\frac{\tau}{1-\lambda_k}
$$

- $\lambda_k<1$: leaky; memory ≈ $\tau/(1-\lambda_k)$.
- $\lambda_k=1$: $\tau\dot c_k=\mathbf{e}_k^T\mathbf{h}$ → perfect integrator; holds $c_k$ forever when $\mathbf{h}=0$.
- $\lambda_k>1$: unstable → runaway.
```

```{tip} Derivation
:class: dropdown
1. Symmetric $W=\sum_k\lambda_k\mathbf{e}_k\mathbf{e}_k^T$, $\mathbf{e}_k$ orthonormal.
2. Project the model onto $\mathbf{e}_k$: $\tau\,\mathbf{e}_k^T\dot{\mathbf{r}}=-\mathbf{e}_k^T\mathbf{r}+\mathbf{e}_k^TW\mathbf{r}+\mathbf{e}_k^T\mathbf{h}$.
3. $\mathbf{e}_k^TW=\lambda_k\mathbf{e}_k^T$ → $\tau\dot c_k=-(1-\lambda_k)c_k+\mathbf{e}_k^T\mathbf{h}$.
4. Modes decouple → each = membrane-like leaky integrator w/ time constant $\tau/(1-\lambda_k)$.
```

````{important} Code
:class: dropdown
```python
import torch

class LineAttractor:
    def __init__(self, n, lam, tau=0.1):
        e = torch.ones(n) / n ** 0.5             ## the integrating mode (uniform pattern)
        self.W = lam * torch.outer(e, e)         ## rank-1: eigenvalue lam on e, 0 elsewhere
        self.e, self.tau = e, tau

    def run(self, h_seq, dt=0.001):
        r = torch.zeros_like(self.e)
        for h in h_seq:
            r = r + dt / self.tau * (-r + self.W @ r + h)
        return self.e @ r                        ## activity along the mode

## Example: 50 ms input pulse along e, then 2 s of silence; tau = 100 ms
n = 10
pulse = [torch.ones(n) / n ** 0.5] * 50 + [torch.zeros(n)] * 2000
print(LineAttractor(n, lam=1.0).run(pulse))   ## ≈ 0.5 = pulse area / tau, held
print(LineAttractor(n, lam=0.9).run(pulse))   ## ≈ 0.07 ← tau_eff = 1 s → 0.49 e^-2 after 2 s of leak
```
````

```{attention} Q&A
:class: dropdown
*Fine-tuning problem?*

- Integrator must hold eye position for ~tens of s, while neuronal & synaptic time constants are ms–100s of ms.
- $\tau=100$ ms, target $\tau^\text{eff}=20$ s → $|1-\lambda|\le0.005$ → 0.5% precision.
- Mistuned → drift (leaky or unstable). Proposed fixes: feedback-driven tuning, bistable cellular / dendritic elements, many discrete attractors.

*Noise?*

- Neutral direction → noise accumulates → bump diffuses along the manifold → memory error ⬆️ w/ delay.

*Relation to ML?*

- LSTM cell state w/ forget gate ≈ 1 → line-attractor-like memory → [LSTM](../dl/basics/module.md#lstm).
- Linear recurrences w/ eigenvalues near 1 → long memory in [SSMs](../nlp/ssm.md).
```

&nbsp;

## Balanced & Chaotic Regimes

### E/I Balance
- **What**: Large excitatory & inhibitory inputs to each neuron nearly cancel → net input of order 1 → irregular, fluctuation-driven firing. {cite:p}`vanvreeswijk1996chaos`
- **Why**: Each cortical neuron gets thousands of inputs.
    - Averaging many inputs → smooth input → regular firing expected.
    - Observed: irregular, Poisson-like firing → [Poisson Spike Model](coding.md#poisson-spike-model).
- **How**:
    1. Each neuron gets $K$ inputs w/ strength $\propto1/\sqrt{K}$.
    2. Total E & total I input are each $O(\sqrt{K})$ → large.
    3. Network dynamics set the rates so that they cancel to $O(1)$ → no fine tuning needed.
    4. Remaining fluctuations are $O(1)$ → threshold crossings driven by fluctuations → irregular spikes.
    5. Population rates respond linearly & fast to external input.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $r_E,r_I$: Mean E / I population rates.
    - $r_0$: External input rate.
- Params:
    - $j_{XY}$: Rescaled coupling from population $Y$ to $X$ ($J_{XY}=j_{XY}/\sqrt{K}$).
- Misc:
    - $K$: #inputs per neuron from each population.
    - $\mu_X$: Mean input to a neuron in population $X$.

Mean input (E neuron; I analogous):

$$
\mu_E=K\left(J_{EE}r_E-J_{EI}r_I+J_{E0}r_0\right)=\sqrt{K}\left(j_{EE}r_E-j_{EI}r_I+j_{E0}r_0\right)
$$

Balance ($K\to\infty$, $\mu_E$ finite):

$$
\begin{align*}
j_{EE}r_E-j_{EI}r_I+j_{E0}r_0&=0\\
j_{IE}r_E-j_{II}r_I+j_{I0}r_0&=0
\end{align*}
$$

- Linear in $r_E,r_I,r_0$ → rates $\propto r_0$ → linear population response.
- Needs an invertible coupling matrix & a positive solution → constraints on the $j$'s.

Input variance (independent Poisson inputs):

$$
\sigma^2\propto KJ^2r=j^2r=O(1)
$$

- Per population, w/ a fixed synaptic filter or counting window. Input correlations add covariance terms.
```

```{attention} Q&A
:class: dropdown
*Why $1/\sqrt{K}$ scaling?*

- Mean ∝ $KJ$ & std ∝ $\sqrt{K}J$. $J\propto1/\sqrt{K}$ → std = $O(1)$, mean = $O(\sqrt{K})$ unless cancelled.
- Same logic as $1/\sqrt{\text{fan-in}}$ weight init in DL → [Xavier/Glorot Initialization](../dl/train/train.md#xavier-glorot-initialization).

*Evidence?*

- In vivo intracellular recordings: E & I conductances rise & fall together.
- Balance is tighter in some regimes ("tight" vs "loose" balance) → degree contested.

*Where do E & I populations come from?*

- Dale's principle: each neuron is E or I → [Synapse Model](neuron.md#synapse-model).

*Functional benefit?*

- Fast tracking of input changes; linear population gain; a substrate for chaos & rich dynamics.
```

&nbsp;

### Chaos in Random Networks
- **What**: Random recurrent rate network: coupling gain $g<1$ → activity decays to 0; $g>1$ → self-sustained chaotic activity ($N\to\infty$). {cite:p}`sompolinsky1988chaos`
- **Why**: Explain spontaneous, irregular activity w/o external noise. Locate the regime where recurrent networks are most flexible.
- **How**:
    1. Random weights $W_{ij}\sim\mathcal{N}(0,g^2/N)$.
    2. Linearize at 0: eigenvalues of $W$ fill a disk of radius $g$ (circular law).
    3. $g<1$: all modes decay → fixed point 0.
    4. $g>1$: modes grow, saturation bounds them → chaotic activity for large random networks (dynamical mean-field theory).
        - Bounded instability alone ≠ chaos: 1 unit w/ $\dot x=-x+g\tanh x$, $g>1$, just settles at a nonzero fixed point.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\mathbf{x}$: Input currents; $\mathbf{r}=\tanh(\mathbf{x})$.
    - $g$: Coupling gain.

Model:

$$
\tau\frac{d\mathbf{x}}{dt}=-\mathbf{x}+W\tanh(\mathbf{x}),\quad W_{ij}\sim\mathcal{N}\left(0,\frac{g^2}{N}\right)
$$

Properties:
- Jacobian at 0: $(-I_N+W)/\tau$ → stable iff all eigenvalues of $W$ have real part $<1$ → $g<1$ ($N\to\infty$).
- $g>1$: fixed point 0 unstable; dynamics chaotic (positive Lyapunov exponent).
- Mean-field theory: each unit sees Gaussian input whose autocorrelation is solved self-consistently.
```

````{important} Code
:class: dropdown
```python
import torch

class RandomRNN:
    def __init__(self, n, g, tau=1.0):
        self.W = g * torch.randn(n, n) / n ** 0.5   ## W_ij ~ N(0, g^2 / N)
        self.tau = tau

    def run(self, x, T=200.0, dt=0.1):
        traj = []
        for _ in range(int(T / dt)):
            x = x + dt / self.tau * (-x + self.W @ torch.tanh(x))
            traj.append(x)
        return torch.stack(traj)                    ## (steps, n)

## Example: same random start, below vs above g = 1
torch.manual_seed(0)
x0 = torch.randn(500)
for g in [0.5, 1.5]:
    late = RandomRNN(500, g).run(x0)[-1000:]       ## last 100 time units
    print(g, late.std(dim=0).mean())               ## temporal variation per unit, averaged
## g = 0.5 → ≈ 0 (settled at 0); g = 1.5 → O(1): keeps changing (proving chaos needs a Lyapunov exponent)
```
````

```{attention} Q&A
:class: dropdown
*Why is "near $g=1$" useful?*

- $g\ll1$: fast forgetting. $g\gg1$: chaos amplifies noise → no reliable memory.
- Near the transition: long memory & rich dynamics → often best for [Reservoir Computing](#reservoir-computing) (task-dependent).

*DL link?*

- Same transition in deep nets at init: weight variance $\sigma_w^2$ sets ordered vs chaotic signal propagation → init near the "edge of chaos" trains deepest.
- Orthogonal init: all singular values of each weight matrix = 1; activation derivatives still distort the Jacobian → [Orthogonal Initialization](../dl/train/train.md#orthogonal-initialization).

*Is the brain critical?*

- Criticality hypothesis: cortex operates near a phase transition.
- Evidence: neuronal avalanches w/ power-law sizes (exponent ≈ −3/2) in cortical tissue (Beggs & Plenz, 2003).
- Contested: power laws also arise from non-critical mechanisms.
```

&nbsp;

#### Reservoir Computing
- **What**: Fixed (usually random) recurrent network (reservoir) + trained readout (usually linear). {cite:p}`jaeger2001echo,maass2002real`
    - **Echo state network** (rate units). **Liquid state machine** (spiking units).
- **Why**: Training recurrent weights by BPTT is slow & unstable (vanishing / exploding gradients).
    - A random reservoir already holds a rich, nonlinear memory of input history → a linear readout can compute many temporal functions.
    - Brain angle: local cortical circuits may act as generic reservoirs.
- **How**:
    1. Random recurrent $W$, scaled near the edge of stability.
    2. Drive the reservoir w/ the input; record its states.
    3. Fit only the readout by ridge regression.
    - **Echo state property**: state = function of input history; initial state forgotten.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{u}_t$: Input.
    - $\mathbf{x}_t$: Reservoir state.
    - $\mathbf{y}_t$: Output.
- Params:
    - $W_\text{in}$: Fixed input weights.
    - $W_\text{out}$: Trained readout.
- Hyperparams:
    - $\lambda$: Ridge strength.
- Misc:
    - $X$: States as columns. $Y$: Targets as columns.

Model:

$$
\mathbf{x}_{t+1}=\tanh\left(W\mathbf{x}_t+W_\text{in}\mathbf{u}_{t+1}\right),\qquad\mathbf{y}_t=W_\text{out}\mathbf{x}_t
$$

Training (ridge regression):

$$
W_\text{out}=YX^T\left(XX^T+\lambda I_N\right)^{-1}
$$

Echo state property (standard tanh ESN):
- Sufficient: largest singular value of $W$ < 1.
- Common heuristic: spectral radius < 1. Neither necessary nor sufficient in general.
```

````{important} Code
:class: dropdown
```python
import torch

class EchoStateNetwork:
    def __init__(self, n, n_in=1, rho=0.9, in_scale=0.5, ridge=1e-4):
        W = torch.randn(n, n)
        self.W = rho * W / torch.linalg.eigvals(W).abs().max()   ## set spectral radius to rho
        self.W_in = in_scale * torch.randn(n, n_in)
        self.ridge = ridge

    def states(self, u):                                         ## u: (T, n_in)
        x, X = torch.zeros(self.W.shape[0]), []
        for u_t in u:
            x = torch.tanh(self.W @ x + self.W_in @ u_t)         ## fixed, untrained dynamics
            X.append(x)
        return torch.stack(X)                                    ## (T, n)

    def fit(self, u, y, washout=100):
        X, y = self.states(u)[washout:], y[washout:]             ## drop initial transient
        ## only the linear readout is trained (ridge regression)
        A = X.T @ X + self.ridge * torch.eye(X.shape[1])
        self.W_out = torch.linalg.solve(A, X.T @ y)

    def predict(self, u, washout=100):
        return self.states(u)[washout:] @ self.W_out             ## frozen readout

## Example: memory task y_t = u_{t-5}; train, then test on a fresh sequence
torch.manual_seed(0)
u, u_test = torch.rand(3000, 1) * 2 - 1, torch.rand(1000, 1) * 2 - 1
y, y_test = torch.roll(u, 5, dims=0), torch.roll(u_test, 5, dims=0)   ## washout drops roll's wrap-around
esn = EchoStateNetwork(200)
esn.fit(u, y)
y_hat = esn.predict(u_test)
print(torch.corrcoef(torch.stack([y_hat[:, 0], y_test[100:, 0]]))[0, 1])  ## close to 1 on held-out data
```
````

```{attention} Q&A
:class: dropdown
*Pros?*

- Training = 1 linear solve → fast, no BPTT.
- Many physical dynamical systems can serve as the reservoir (photonic, spintronic, analog chips).

*Cons?*

- Task-agnostic reservoir → needs many units; trained RNNs are usually more efficient.

*FORCE learning?*

- Feed the readout back into a chaotic reservoir; update the readout online w/ recursive least squares → keeps the error small from the start → network learns to generate target patterns. {cite:p}`sussillo2009generating`

*Relation to other models?*

- Same design as the cerebellar [Marr–Albus Model](learning.md#marr-albus-model) & random-feature models: fixed random expansion + trained linear readout.
```

&nbsp;

## Computation Through Dynamics

### Trained RNN Models
- **What**: Train an RNN on the animal's task → reverse-engineer its dynamics → mechanistic hypotheses for the brain. {cite:p}`mante2013context`
- **Why**: Recorded populations are high-dimensional & heterogeneous → single-neuron tuning does not reveal the mechanism.
    - Trained RNN = fully observable model system w/ similar population activity.
- **How**:
    1. Train an RNN on the task (BPTT).
    2. Compare its population activity w/ recordings (e.g., in a PCA subspace).
    3. Find fixed & slow points: minimize $q(\mathbf{x})=\frac{1}{2}\|F(\mathbf{x})\|^2$.
    4. Linearize around them → eigenvalues & eigenvectors explain the computation.
    - Context-dependent choice (motion vs color, same stimuli):
        - Both inputs reach prefrontal cortex. Irrelevant input is not filtered out before it.
        - RNN: context changes the local dynamics → a **selection vector** picks which input moves the state along an approximate line attractor.
    - Motor cortex: preparatory state sets the initial condition → rotational population dynamics during reaching, consistent w/ a dynamical system that generates the movement.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{x}$: Hidden state (currents).
    - $\mathbf{u}$: Input (held constant during the analysis).
- Params:
    - $B$: Input weights.
- Misc:
    - $F$: Vector field.
    - $\mathbf{x}^*$: Fixed point.

Dynamics:

$$
\tau\frac{d\mathbf{x}}{dt}=F(\mathbf{x})=-\mathbf{x}+W\phi(\mathbf{x})+B\mathbf{u}
$$

Fixed & slow points {cite:p}`sussillo2013opening`:

$$
q(\mathbf{x})=\frac{1}{2}\|F(\mathbf{x})\|^2,\qquad\mathbf{x}^*\in\text{local minima of }q\text{ (many random starts)}
$$

- $q(\mathbf{x}^*)=0$: fixed point. Small $q>0$: slow point.

Linearization (at a fixed point; at a slow point $\mathbf{x}_c$, add the small constant drift $F(\mathbf{x}_c)$):

$$
\tau\frac{d\,\delta\mathbf{x}}{dt}\approx J\,\delta\mathbf{x},\quad J=-I_N+W\,\text{diag}\left(\phi'(\mathbf{x}^*)\right)
$$

- All eigenvalues of $J$ w/ real part < 0 → stable. Eigenvalue ≈ 0 → slow / integrating direction.
```

````{important} Code
:class: dropdown
```python
import torch

class FixedPointFinder:
    def __init__(self, W, B, u):
        self.W, self.bu = W, B @ u                   ## input held constant during analysis

    def F(self, x):
        return -x + self.W @ torch.tanh(x) + self.bu

    def find(self, x0, steps=2000, lr=0.5):
        x = x0.clone().requires_grad_()
        opt = torch.optim.SGD([x], lr=lr)              ## plain GD: converges to q ≈ 0 (Adam hovers)
        for _ in range(steps):
            q = 0.5 * self.F(x).pow(2).sum()         ## q(x) = 0 exactly at fixed points
            opt.zero_grad(); q.backward(); opt.step()
        x = x.detach()
        q = 0.5 * self.F(x).pow(2).sum().item()      ## residual at the returned point
        J = -torch.eye(len(x)) + self.W * (1 - torch.tanh(x) ** 2)   ## W diag(phi'(x))
        return x, q, torch.linalg.eigvals(J).real.max().item()

## Example: 2 self-exciting units (W = 2I) → each unit bistable: x = 2 tanh(x) → x ∈ {-1.915, 0, 1.915}
fpf = FixedPointFinder(2 * torch.eye(2), torch.zeros(2, 1), torch.zeros(1))
torch.manual_seed(0)
for x0 in torch.rand(4, 2) * 6 - 3:
    x, q, max_re = fpf.find(x0)
    if q > 1e-8:
        print(x, q, "slow point / not converged")    ## classify only true fixed points
    else:
        print(x, "stable" if max_re < -1e-3 else "unstable" if max_re > 1e-3 else "marginal (linearization inconclusive)")
## stable iff both coordinates ≈ ±1.915 (eigenvalue -1 + 2 sech^2 ≈ -0.83); a 0 coordinate → eigenvalue +1 → unstable
```
````

```{attention} Q&A
:class: dropdown
*Pros?*

- Turns "the network works" into a mechanism: attractors, saddles, rotations, selection vectors.
- Generates testable predictions for recordings & perturbations.

*Cons?*

- Many RNN solutions fit the same task. Architecture, regularization & training details change the mechanism.
- → A trained RNN = 1 hypothesis, ❌the brain's answer. Constrain it w/ neural data.

*Relation to ML interpretability?*

- Same goal as circuit-level analysis of trained transformers → [Circuit](../dl/repr/interp.md#circuit).
- Population geometry → [Neural Manifold](../math/geo/repr.md#neural-manifold).

*Low-rank RNNs?*

- Constrain $W$ to low rank → recurrent drive confined to a low-dimensional subspace (inputs can still drive other directions) → easier to analyze & often enough to fit tasks.
```

&nbsp;
