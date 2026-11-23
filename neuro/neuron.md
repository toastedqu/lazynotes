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
# Neuron Models
Mathematical models of single neurons & synapses, from biophysical to abstract. Networks of them → [Network Dynamics](network.md).

Notations:
- $V$: Membrane potential (inside − outside), mV
- $t$: Time, ms
- $C_m$: Membrane capacitance
- $I_\text{ext}$: Injected (external) current
- $g_X$: Conductance of channel type $X$
- $E_X$: Reversal potential of $X$ (the $V$ at which its current changes sign)
- $\tau_m$: Membrane time constant

Overrides of global notation: $E$ = reversal potential, ❌expectation. $C$ = capacitance. $R$ = resistance. $m,n$ (HH) = gating variables, ❌#samples / #features. $x$ (cable) = position along the cable.

&nbsp;

## Membrane

### Membrane Equation
- **What**: Membrane as an RC circuit: capacitor in parallel w/ a leak conductance & a battery $E_L$.
- **Why**: Base of every neuron model. Predicts how a neuron sums inputs below threshold.
- **How**:
    1. Input current charges the capacitor & leaks out through the conductance.
    2. → $V$ relaxes exponentially to $E_L+I_\text{ext}/g_L$ w/ time constant $\tau_m$.
    3. $\tau_m$ = memory of the neuron: inputs within ~$\tau_m$ add up (**temporal summation**).

```{note} Math
:class: dropdown
Notations:
- Params:
    - $g_L$: Leak conductance.
    - $E_L$: Leak reversal potential (≈ rest).

Model (per unit area; $I_\text{ext}$ = current density):

$$
C_m\frac{dV}{dt}=-g_L(V-E_L)+I_\text{ext}
$$

Solution (constant $I_\text{ext}$):

$$
V(t)=V_\infty+\left(V(0)-V_\infty\right)e^{-t/\tau_m},\quad V_\infty=E_L+\frac{I_\text{ext}}{g_L},\quad\tau_m=\frac{C_m}{g_L}=R_mC_m
$$

- $R_m=1/g_L$: Specific membrane resistance.
- Typical values: $C_m\approx1$ µF/cm², $\tau_m$ ≈ 10–20 ms (cortical neurons at rest).
```

```{attention} Q&A
:class: dropdown
*Why does $\tau_m$ not depend on cell size?*

- $C$ & $g_L$ both scale w/ membrane area → ratio cancels.

*$\tau_m$ in an active network?*

- Background synaptic **conductance** $g_\text{bg}$ adds to the leak → $\tau_\text{eff}=C_m/(g_L+g_\text{bg})$ ⬇️ (high-conductance state).
- Injected current alone only shifts $V_\infty$; it does not change $\tau_m$.
- → Shorter summation window → sharper coincidence detection.

*Relation to ML?*

- Linear 1st-order ODE = leaky integrator = exponential moving average of the input.
- Same structure as a diagonal linear state-space layer → [SSM](../nlp/ssm.md).
```

&nbsp;

#### Cable Theory
- **What**: Membrane equation extended in space: $V$ along a thin cylinder (dendrite / axon).
- **Why**: Dendrites are long → input far from the soma decays & slows on the way → synapse location matters.
- **How**: At each point, current either leaks out (membrane) or flows along the core (axial resistance).
    - → Steady-state $V$ decays exponentially w/ distance; length constant $\lambda$.
    - Thicker cable → less axial resistance → $\lambda$⬆️.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $x$: Position along the cable.
- Params:
    - $d$: Cable diameter.
    - $R_i$: Axial resistivity (Ω·cm).
    - $R_m$: Specific membrane resistance (Ω·cm²).

Cable equation:

$$
\lambda^2\frac{\partial^2V}{\partial x^2}=\tau_m\frac{\partial V}{\partial t}+(V-E_L),\quad\lambda=\sqrt{\frac{R_md}{4R_i}}
$$

- $\lambda$: Length constant.

Steady state, semi-infinite cable, $V(0)$ held fixed:

$$
V(x)-E_L=\left(V(0)-E_L\right)e^{-x/\lambda}
$$
```

```{tip} Derivation
:class: dropdown
1. Per unit length: axial resistance $r_i=\frac{4R_i}{\pi d^2}$ (Ω/cm), membrane resistance × length $r_m=\frac{R_m}{\pi d}$ (Ω·cm), capacitance $c_m=C_m\pi d$. Segment $\Delta x$: axial $r_i\Delta x$, membrane $r_m/\Delta x$, capacitance $c_m\Delta x$.
2. Axial current (Ohm): $I_a=-\frac{1}{r_i}\frac{\partial V}{\partial x}$.
3. Charge conservation: current lost from the core = membrane current → $-\frac{\partial I_a}{\partial x}=c_m\frac{\partial V}{\partial t}+\frac{V-E_L}{r_m}$.
4. Substitute 2 into 3, multiply by $r_m$: $\frac{r_m}{r_i}\frac{\partial^2V}{\partial x^2}=r_mc_m\frac{\partial V}{\partial t}+(V-E_L)$.
5. $\frac{r_m}{r_i}=\frac{R_m/(\pi d)}{4R_i/(\pi d^2)}=\frac{R_md}{4R_i}=\lambda^2$. $r_mc_m=R_mC_m=\tau_m$.
6. Steady state ($\partial_tV=0$) → $\lambda^2V''=V-E_L$ → bounded solution $\propto e^{-x/\lambda}$.
```

```{attention} Q&A
:class: dropdown
*Typical $\lambda$?*

- $d=2$ µm, $R_m=20$ kΩ·cm², $R_i=200$ Ω·cm → $\lambda=\sqrt{\frac{2\times10^4\cdot2\times10^{-4}}{800}}$ cm ≈ 0.7 mm.
- Same order as dendrite length → distal inputs attenuate a lot.

*Transient vs steady-state attenuation?*

- Fast signals attenuate more than $e^{-x/\lambda}$ ← capacitance low-pass filters them.

*How is it simulated?*

- Discretize the cable into compartments → coupled membrane equations → **multi-compartment model**.
- Cable equation = heat equation + decay term → [Heat Equation](../math/diffeq.md#heat-equation).
```

&nbsp;

## Spiking Models

### Hodgkin–Huxley Model
- **What**: Conductance-based ODE model of the action potential: membrane capacitor + voltage-gated Na⁺, K⁺ & leak conductances.
- **Why**: Explain the spike (brief, all-or-none, ~100 mV voltage pulse) quantitatively from measured currents.
    - Fit to squid giant-axon data (1952) → predicted spike shape & conduction speed.
- **How**:
    1. Membrane = circuit: $C_m\dot V$ = −(sum of ionic currents) + input.
    2. Each ionic current = conductance × driving force $(V-E_X)$.
    3. Conductances set by **gating variables** $m,h,n\in[0,1]$ = fraction of open gates.
    4. Each gate relaxes toward a voltage-dependent steady state w/ a voltage-dependent time constant.
    5. Spike:
        1. $m$ (Na⁺ activation) is fast → depolarization opens Na⁺ → more depolarization → upstroke (positive feedback).
        2. $h$ (Na⁺ inactivation) ⬇️ & $n$ (K⁺ activation) ⬆️ are slow → repolarization (negative feedback).
        3. $h$ recovers slowly → **refractory period** → caps max firing rate.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\bar g_\text{Na},\bar g_K$: Max Na⁺ / K⁺ conductance.
    - $g_L$: Leak conductance.
- Misc:
    - $m$: Na⁺ activation gate.
    - $h$: Na⁺ inactivation gate.
    - $n$: K⁺ activation gate.
    - $\alpha_x(V),\beta_x(V)$: Opening / closing rate of gate $x$.

Model:

$$
C_m\frac{dV}{dt}=-\bar g_\text{Na}m^3h(V-E_\text{Na})-\bar g_Kn^4(V-E_K)-g_L(V-E_L)+I_\text{ext}
$$

$$
\frac{dx}{dt}=\alpha_x(V)(1-x)-\beta_x(V)\,x=\frac{x_\infty(V)-x}{\tau_x(V)},\quad x\in\{m,h,n\}
$$

- $x_\infty=\frac{\alpha_x}{\alpha_x+\beta_x}$: Steady-state open fraction.
- $\tau_x=\frac{1}{\alpha_x+\beta_x}$: Gate time constant.
- $m^3h$: 3 activation gates & 1 inactivation gate must all be open.
- $n^4$: 4 K⁺ gates must all be open.

Squid axon params (rest ≈ −65 mV; mV, ms, µF/cm², mS/cm²):
- $C_m=1$, $\bar g_\text{Na}=120$, $\bar g_K=36$, $g_L=0.3$.
- $E_\text{Na}=50$, $E_K=-77$, $E_L=-54.387$.

Rates (1/ms):

$$
\begin{align*}
\alpha_n&=\frac{0.01(V+55)}{1-e^{-(V+55)/10}}, & \beta_n&=0.125\,e^{-(V+65)/80}\\
\alpha_m&=\frac{0.1(V+40)}{1-e^{-(V+40)/10}}, & \beta_m&=4\,e^{-(V+65)/18}\\
\alpha_h&=0.07\,e^{-(V+65)/20}, & \beta_h&=\frac{1}{1+e^{-(V+35)/10}}
\end{align*}
$$

- $\alpha_n$ at $V=-55$ & $\alpha_m$ at $V=-40$: 0/0 → use the limit (0.1 & 1).
```

````{important} Code
:class: dropdown
```python
import numpy as np

class HodgkinHuxley:
    ## squid params, rest ≈ -65 mV; units: mV, ms, uF/cm^2, mS/cm^2, uA/cm^2
    C, gNa, gK, gL = 1.0, 120.0, 36.0, 0.3
    ENa, EK, EL = 50.0, -77.0, -54.387

    @staticmethod
    def vtrap(z):
        ## z / (1 - e^-z); limit 1 at z = 0 (avoids the 0/0 in alpha_n, alpha_m)
        return 1.0 if abs(z) < 1e-7 else z / -np.expm1(-z)

    @staticmethod
    def rates(V):
        an = 0.1 * HodgkinHuxley.vtrap((V + 55) / 10)
        bn = 0.125 * np.exp(-(V + 65) / 80)
        am = 1.0 * HodgkinHuxley.vtrap((V + 40) / 10)
        bm = 4.0 * np.exp(-(V + 65) / 18)
        ah = 0.07 * np.exp(-(V + 65) / 20)
        bh = 1 / (1 + np.exp(-(V + 35) / 10))
        return (an, bn), (am, bm), (ah, bh)

    def __init__(self, V0=-65.0):
        self.V = V0
        ## start every gate at its steady state for V0
        (an, bn), (am, bm), (ah, bh) = self.rates(V0)
        self.n, self.m, self.h = an / (an + bn), am / (am + bm), ah / (ah + bh)

    def step(self, I_ext, dt=0.01):
        (an, bn), (am, bm), (ah, bh) = self.rates(self.V)
        ## ionic current = conductance x driving force
        I_Na = self.gNa * self.m**3 * self.h * (self.V - self.ENa)
        I_K = self.gK * self.n**4 * (self.V - self.EK)
        I_L = self.gL * (self.V - self.EL)
        ## each gate relaxes toward its voltage-dependent steady state (forward Euler)
        self.n += dt * (an * (1 - self.n) - bn * self.n)
        self.m += dt * (am * (1 - self.m) - bm * self.m)
        self.h += dt * (ah * (1 - self.h) - bh * self.h)
        self.V += dt * (I_ext - I_Na - I_K - I_L) / self.C
        return self.V

## Example: 10 uA/cm^2 step for 50 ms
neuron = HodgkinHuxley()
V = np.array([neuron.step(10.0) for _ in range(5000)])
print(V.max() > 0)                           ## True: spikes overshoot 0 mV
print(((V[1:] > 0) & (V[:-1] <= 0)).sum())   ## #spikes = #upward crossings of 0 mV
```
````

```{attention} Q&A
:class: dropdown
*Why $m^3h$ & $n^4$?*

- Empirical fit to the sigmoidal onset of the measured conductances.
- Later reading: channel opens only if several independent subunits are all open → product of probabilities.
- Real channel gating is more complex → HH = phenomenological, but quantitatively accurate.

*Is there a fixed threshold voltage?*

- ❌. Threshold depends on how fast $V$ rises & on recent spikes ($h$ still low).
- HH has no exact threshold voltage, only a steep boundary in the 4D state space.

*Pros?*

- Every param = a measurable quantity.
- Template for all conductance-based models (add more channel types → bursting, adaptation).

*Cons?*

- 4 stiff ODEs per compartment → small $dt$ → slow for large networks.
- Hard to analyze → reduce to 2D ([FitzHugh–Nagumo](#fitzhugh-nagumo-model)) or to threshold models ([LIF](#leaky-integrate-and-fire)).

*What kind of excitability?*

- Onset of repetitive firing = bifurcation of the 4D dynamics → [Bifurcation](../math/diffeq.md#bifurcation).
- HH: firing starts at a nonzero rate (**class 2** excitability).
```

&nbsp;

#### FitzHugh–Nagumo Model
- **What**: 2D reduction of HH: fast voltage-like variable w/ cubic nonlinearity + slow linear recovery variable.
- **Why**: HH = 4D → no phase-plane picture. 2D → spikes, threshold & oscillation onset become visible geometry.
- **How**: Phase plane.
    1. Qualitatively: $v$ ↔ HH's fast variables ($V$, $m$); $w$ ↔ slow variables ($h$, $n$). (Built as a generalized van der Pol oscillator, ❌a formal reduction.)
    2. $v$-nullcline = cubic (N-shape). $w$-nullcline = line.
    3. Rest = their intersection on the left branch.
    4. Kick $v$ past the middle branch → $v$ jumps to the right branch (spike) → $w$ slowly rises → $v$ falls back → refractory return.
    5. $I$⬆️ → intersection moves onto the middle branch → rest loses stability (subcritical Hopf) → large limit cycle → periodic spiking.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $v$: Fast, voltage-like variable.
    - $w$: Slow recovery variable.
    - $I$: Input (model units).
- Hyperparams:
    - $\varepsilon$: Time-scale ratio ($\ll1$).
    - $a,b$: Shape the recovery nullcline $w=(v+a)/b$ (intercept $a/b$, slope $1/b$).

Model ($t$ in model time units):

$$
\frac{dv}{dt}=v-\frac{v^3}{3}-w+I,\qquad\frac{dw}{dt}=\varepsilon\,(v+a-bw)
$$

- Common values: $a=0.7$, $b=0.8$, $\varepsilon=0.08$.

Nullclines:

$$
w=v-\frac{v^3}{3}+I,\qquad w=\frac{v+a}{b}
$$

Jacobian at a fixed point $(v^*,w^*)$:

$$
J=\begin{bmatrix}1-v^{*2} & -1\\ \varepsilon & -\varepsilon b\end{bmatrix}
$$

- Hopf (trace = 0, $\det J=\varepsilon(1-\varepsilon b^2)>0$): $v^{*2}=1-\varepsilon b$ → $v^*\approx\pm0.968$ → $I\approx0.331$ & $I\approx1.419$ for these values.
- Both Hopf points are subcritical → near $I\approx0.331$, rest & spiking can coexist (bistability). Above $I\approx1.419$: depolarized steady state, no spiking.
```

````{important} Code
:class: dropdown
```python
import numpy as np

class FitzHughNagumo:
    def __init__(self, a=0.7, b=0.8, eps=0.08):
        self.a, self.b, self.eps = a, b, eps
        self.v, self.w = -1.1994, -0.6243           ## ≈ rest point for I = 0 & the default a, b

    def step(self, I, dt=0.01):
        dv = self.v - self.v**3 / 3 - self.w + I    ## fast: cubic → bistable branches
        dw = self.eps * (self.v + self.a - self.b * self.w)  ## slow: recovery
        self.v += dt * dv
        self.w += dt * dw
        return self.v

## Example: 200 time units below & above the Hopf onset (I ≈ 0.33)
for I in [0.0, 0.5]:
    fhn = FitzHughNagumo()
    v = np.array([fhn.step(I) for _ in range(20_000)])
    print(I, ((v[1:] > 1) & (v[:-1] <= 1)).sum())  ## 0 → rest; I = 0.5 → periodic spikes (count > 0)
```
````

```{attention} Q&A
:class: dropdown
*Why does a 2D model capture so much?*

- Time-scale separation ($\varepsilon\ll1$) → fast jumps between branches & slow crawls along them → relaxation oscillator.

*What does it miss?*

- Spike shape, multiple channel types, class 1 excitability (FHN onset via Hopf → nonzero frequency).

*Relation to other models?*

- Van der Pol oscillator + recovery offset.
- Morris–Lecar: biophysical 2D model that can show class 1 or class 2.
```

&nbsp;

### Leaky Integrate-and-Fire
- **What**: Membrane equation + hard threshold & reset. Spike = event, ❌waveform.
- **Why**: [HH](#hodgkin-huxley-model) = 4 stiff ODEs per neuron → slow & hard to analyze.
    - Spike shape is stereotyped → keep only spike times.
- **How**:
    1. Integrate input w/ leak ([Membrane Equation](#membrane-equation)).
    2. $V$ reaches $V_\text{th}$ → emit spike.
    3. Reset $V$ to $V_\text{reset}$. Hold it there for the refractory time $t_\text{ref}$.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $R$: Input resistance (whole cell).
    - $E_L$: Leak reversal (rest).
- Hyperparams:
    - $V_\text{th}$: Spike threshold.
    - $V_\text{reset}$: Reset potential.
    - $t_\text{ref}$: Refractory period.
- Misc:
    - $r$: Firing rate.

Model:

$$
\tau_m\frac{dV}{dt}=-(V-E_L)+RI_\text{ext}
$$

- $V\ge V_\text{th}$ → spike at $t$; $V\leftarrow V_\text{reset}$; hold for $t_\text{ref}$.

f–I curve (constant $I_\text{ext}$):

$$
r=\begin{cases}
0, & RI_\text{ext}\le V_\text{th}-E_L\\[2mm]
\left[t_\text{ref}+\tau_m\ln\dfrac{E_L+RI_\text{ext}-V_\text{reset}}{E_L+RI_\text{ext}-V_\text{th}}\right]^{-1}, & \text{otherwise}
\end{cases}
$$

- $I_\text{ext}$: whole-cell current here; $RI_\text{ext}$ in mV.
- **Rheobase** $I_\text{rh}=(V_\text{th}-E_L)/R$: fires iff $I_\text{ext}>I_\text{rh}$ (at equality, $V$ only approaches threshold).
- $I_\text{ext}\to\infty$ → $r\to1/t_\text{ref}$.
```

```{tip} Derivation
:class: dropdown
1. Between spikes ([Membrane Equation](#membrane-equation)): $V(t)=V_\infty+(V_\text{reset}-V_\infty)e^{-t/\tau_m}$, $V_\infty=E_L+RI_\text{ext}$.
2. Next spike when $V(T)=V_\text{th}$ → $e^{-T/\tau_m}=\frac{V_\infty-V_\text{th}}{V_\infty-V_\text{reset}}$.
3. → $T=\tau_m\ln\frac{V_\infty-V_\text{reset}}{V_\infty-V_\text{th}}$.
4. Needs $V_\infty>V_\text{th}$ → else $V$ never reaches threshold → rheobase.
5. Inter-spike interval $=t_\text{ref}+T$ → $r=1/(t_\text{ref}+T)$.
6. Large $I_\text{ext}$: $\ln\left(1+\frac{V_\text{th}-V_\text{reset}}{V_\infty-V_\text{th}}\right)\approx\frac{V_\text{th}-V_\text{reset}}{RI_\text{ext}}$ → w/o refractoriness, $r$ grows ≈ linearly in $I_\text{ext}$.
```

````{important} Code
:class: dropdown
```python
import torch

class LIF:
    ## units: mV, ms, MOhm, nA (MOhm x nA = mV)
    def __init__(self, n, tau=20.0, E_L=-65.0, V_th=-50.0, V_reset=-65.0, R=10.0, t_ref=2.0, dt=0.1):
        self.tau, self.E_L, self.V_th, self.V_reset, self.R, self.dt = tau, E_L, V_th, V_reset, R, dt
        self.ref_steps = round(t_ref / dt)
        self.V = torch.full((n,), E_L)
        self.ref = torch.zeros(n, dtype=torch.long)   ## remaining refractory steps per neuron

    def step(self, I):
        ## leaky integration (Euler); V frozen while refractory
        dV = (-(self.V - self.E_L) + self.R * I) / self.tau * self.dt
        self.V = torch.where(self.ref > 0, self.V, self.V + dV)
        self.ref = (self.ref - 1).clamp(min=0)
        ## threshold -> spike -> reset & start the refractory clock
        spike = self.V >= self.V_th
        self.V = torch.where(spike, torch.full_like(self.V, self.V_reset), self.V)
        self.ref = torch.where(spike, torch.full_like(self.ref, self.ref_steps), self.ref)
        return spike

## Example: 3 neurons, constant input, 1 s
lif = LIF(3)
I = torch.tensor([1.0, 2.0, 4.0])                         ## nA; rheobase = 15 mV / 10 MOhm = 1.5 nA
counts = sum(lif.step(I).long() for _ in range(10_000))   ## 10,000 steps x 0.1 ms = 1 s
print(counts)  ## ≈ tensor([0, 33, 87]) ← f–I formula: 0, 33.6, 87.7 Hz
```
````

```{attention} Q&A
:class: dropdown
*What does LIF lose?*

- Spike shape, adaptation, bursting, dendrites, dynamic threshold.
- Fix: add variables → adaptive exponential IF (AdEx), [Izhikevich](#izhikevich-model).

*Why keep the leak?*

- No leak (perfect integrator): any positive input fires eventually; $r=\left[t_\text{ref}+C(V_\text{th}-V_\text{reset})/I_\text{ext}\right]^{-1}$ → $r\propto I_\text{ext}$ if $t_\text{ref}=0$.
- Leak → rheobase + forgetting window ~$\tau_m$ → coincidence detection.

*How does LIF relate to ReLU?*

- f–I curve: zero below rheobase; just above it, $r$ rises steeply (∝ $1/\log\frac{1}{I-I_\text{rh}}$); then ≈ linear; then saturates at $1/t_\text{ref}$.
- Input noise smooths the corner → softplus-like.
- → ReLU ≈ crude rate approximation of an integrate-and-fire neuron.

*Where is LIF used?*

- Spiking network theory & simulations, [Spiking Neural Networks](theory.md#spiking-neural-network), neuromorphic chips.
```

&nbsp;

### Izhikevich Model
- **What**: 2-variable spiking model (quadratic voltage + slow recovery + reset) that reproduces many cortical firing patterns. {cite:p}`izhikevich2003simple`
- **Why**: LIF: cheap but 1 firing pattern. HH: rich but costly. → Want both.
- **How**:
    1. Quadratic term in $V$ → fast positive feedback → upstroke.
    2. Recovery $u$ (lumps K⁺ activation & Na⁺ inactivation) → slow negative feedback.
    3. $V\ge30$ mV → reset $V\leftarrow c$, $u\leftarrow u+d$.
    4. 4 params $(a,b,c,d)$ select the firing type.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $u$: Recovery variable.
- Hyperparams:
    - $a$: Recovery speed.
    - $b$: Sensitivity of $u$ to $V$.
    - $c$: Reset value of $V$ (mV).
    - $d$: Reset jump of $u$.

Model ($V$ in mV, $t$ in ms):

$$
\frac{dV}{dt}=0.04V^2+5V+140-u+I_\text{ext},\quad\frac{du}{dt}=a(bV-u)
$$

- $V\ge30$ → $V\leftarrow c$, $u\leftarrow u+d$.
- $I_\text{ext}$ in model units (absorbs resistance & capacitance).
```

```{dropdown} Table: Izhikevich Param Sets
| Firing type | $a$ | $b$ | $c$ | $d$ | Typical cell |
|:--|:--|:--|:--|:--|:--|
| Regular spiking (RS) | 0.02 | 0.2 | −65 | 8 | Excitatory cortical, adapting |
| Intrinsically bursting (IB) | 0.02 | 0.2 | −55 | 4 | Excitatory, initial burst |
| Chattering (CH) | 0.02 | 0.2 | −50 | 2 | Excitatory, repetitive bursts |
| Fast spiking (FS) | 0.1 | 0.2 | −65 | 2 | Inhibitory interneuron |
| Low-threshold spiking (LTS) | 0.02 | 0.25 | −65 | 2 | Inhibitory interneuron |
```

```{attention} Q&A
:class: dropdown
*Pros?*

- ~LIF cost, ~HH variety of firing patterns.

*Cons?*

- Phenomenological: $(a,b,c,d)$ ≠ measurable biophysical quantities.
- Quadratic term blows up fast → needs small $dt$ & the reset rule.

*Why a quadratic?*

- Quadratic integrate-and-fire = normal form of a saddle-node on invariant circle bifurcation → **class 1** excitability: firing can start at arbitrarily low rates.
- → [Bifurcation](../math/diffeq.md#bifurcation).
```

&nbsp;

## Synapses

### Synapse Model
- **What**: Presynaptic spike → transient postsynaptic conductance → current $\propto(V-E_\text{syn})$.
- **Why**: Couple neuron models into networks. Synaptic sign & time course set network dynamics.
- **How**:
    1. Spike → conductance $g_\text{syn}(t)$ jumps by a fixed amount, then decays (ms).
        - Fixed jump = no history dependence. Real synapses depress / facilitate → [Short-Term Plasticity](learning.md#short-term-plasticity).
    2. Current drives $V$ toward $E_\text{syn}$.
    3. $E_\text{syn}$ above threshold → excitatory (EPSP). $E_\text{syn}$ below threshold → inhibitory: hyperpolarizing IPSP, or shunting w/o voltage change.
    4. Postsynaptic potentials sum in time (within ~$\tau_m$) & space.
    - 1 EPSP ≈ 0.1–1 mV ≪ threshold gap (~15 mV) → needs tens of near-coincident inputs.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\bar g_\text{syn}$: Conductance jump per spike (weight).
    - $E_\text{syn}$: Synaptic reversal potential.
    - $\tau_s$: Synaptic decay time constant.
- Misc:
    - $t_k$: $k$-th presynaptic spike time.
    - $\Theta$: Heaviside step.

Conductance-based synapse:

$$
I_\text{syn}(t)=g_\text{syn}(t)\left(V-E_\text{syn}\right),\quad g_\text{syn}(t)=\bar g_\text{syn}\sum_ke^{-(t-t_k)/\tau_s}\,\Theta(t-t_k)
$$

- Effect depends on $V$ ← driving force.
- **Current-based** approximation: replace $(V-E_\text{syn})$ by a constant → linear in input (common in theory).

Shunting inhibition ($E_i=E_L$), steady state:

$$
V-E_L=\frac{g_e\left(E_e-E_L\right)}{g_L+g_e+g_i}
$$

- $g_e,E_e$: Excitatory conductance & reversal.
- $g_i$: Inhibitory conductance.
- No IPSP at rest, but $g_i$ divides the excitatory effect.
```

```{tip} Derivation
:class: dropdown
*Why does shunting inhibition divide?*
1. Steady state: $0=-g_L(V-E_L)-g_e(V-E_e)-g_i(V-E_i)$.
2. Set $E_i=E_L$, let $u=V-E_L$: $0=-(g_L+g_i)u-g_e\left(u-(E_e-E_L)\right)$.
3. Solve: $u=\frac{g_e(E_e-E_L)}{g_L+g_e+g_i}$ → $g_i$ in the denominator only.
```

```{attention} Q&A
:class: dropdown
*Conductance-based vs current-based?*

- Conductance: inputs interact (shunting, saturation near $E_\text{syn}$) → realistic.
- Current: inputs add linearly → analyzable → most theory & ANN analogies.

*Does shunting divide firing rates?*

- Divides subthreshold $V$ (Derivation).
- On firing rate, shunting acts mostly subtractively → divisive gain needs extra mechanisms (e.g., balanced background noise).
- → [Divisive Normalization](coding.md#divisive-normalization).

*Why are real synapses unreliable?*

- Transmitter release is probabilistic: per-site release probability is often well below 1.
- Function unresolved. Hypotheses: save energy, regularize (dropout-like), sample from uncertainty.

*Dale's principle?*

- 1 neuron's outgoing synapses are all excitatory or all inhibitory. Holds for most neurons; co-release exceptions exist.
- ANN: 1 unit can have + & − outgoing weights. Network models often impose separate E & I populations → [E/I Balance](network.md#e-i-balance).
```

&nbsp;

### Dendritic Computation
- **What**: Nonlinear integration of inputs inside the dendritic tree → 1 neuron ≈ multi-layer network. {cite:p}`poirazi2003pyramidal`
- **Why**: Point-neuron model (weighted sum → threshold) ignores where inputs land → underestimates 1 neuron's power.
- **How**: Active dendrites add local nonlinearities.
    - **Branch spikes**: clustered inputs on 1 thin branch → local regenerative event → branch = sigmoidal subunit.
    - → 2-layer model: branches = hidden units; soma sums them.
    - **Apical–basal coupling** (layer-5 pyramidal cells): coincident top-down (apical) & bottom-up (basal) input → burst.
        - → Multiplicative gating of feedforward input by context.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{x}_b$: Inputs onto branch $b$.
    - $y$: Output rate.
- Params:
    - $\mathbf{w}_b$: Synaptic weights on branch $b$.
    - $\alpha_b$: Branch-to-soma coupling.
- Hyperparams:
    - $\sigma_b$: Branch nonlinearity (sigmoid).
    - $g$: Somatic output nonlinearity.

2-layer model:

$$
y=g\left(\sum_b\alpha_b\,\sigma_b\left(\mathbf{w}_b^T\mathbf{x}_b\right)\right)
$$

- Point neuron = special case: $\sigma_b$ linear.
- Same inputs on 1 branch vs spread over branches → different outputs → location matters.
```

```{attention} Q&A
:class: dropdown
*How deep is 1 neuron?*

- Detailed layer-5 pyramidal neuron model, ms-precise I/O → needs a temporally convolutional DNN w/ 5–8 layers. {cite:p}`beniaguev2021single`
- Remove NMDA receptors from the model → 1 hidden layer suffices → NMDA-based dendritic nonlinearity = main source of depth.

*Can 1 neuron compute XOR?*

- Point neuron: ❌ (linearly separable functions only).
- Human layer-2/3 dendrites: dendritic spikes largest near threshold & smaller for stronger input → non-monotonic → XOR-like in a constrained model (Gidon et al., 2020).

*ML angle?*

- Apical × basal coupling = proposed substrate for top-down credit assignment → [Backprop in the Brain](learning.md#backprop-in-the-brain).
- Implication: #units in an ANN ≠ #neurons in a brain.

*AC angle?*

- Dendritic integration theory: conscious processing needs apical–basal coupling in layer-5 pyramidal cells.
- General anesthesia blocks apical-to-somatic signaling in mouse layer-5 cells (Suzuki & Larkum, 2020). Link to consciousness = hypothesis.
```

&nbsp;

## Rate Models

### Firing-Rate Model
- **What**: Neuron or population described by a continuous firing rate, ❌spikes.
- **Why**: Many questions (memory, decisions, circuit dynamics) depend on average activity → spike times = unneeded detail.
    - Rate models are analyzable & map directly onto ANNs.
- **How**:
    1. Average spikes over time, trials, or a population → rate $r$.
    2. Rate relaxes toward a nonlinear function of its input w/ time constant $\tau_r$.
    3. Network: input = weighted sum of other rates + external input.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{r}$: Rate vector.
    - $\mathbf{h}$: External input.
- Params:
    - $W$: Recurrent weight matrix.
- Hyperparams:
    - $\tau_r$: Rate time constant.
    - $\phi$: Rate nonlinearity (f–I curve, e.g., ReLU, sigmoid).

Model:

$$
\tau_r\frac{d\mathbf{r}}{dt}=-\mathbf{r}+\phi\left(W\mathbf{r}+\mathbf{h}\right)
$$

Properties:
- Fixed point: $\mathbf{r}^*=\phi(W\mathbf{r}^*+\mathbf{h})$.
- No recurrence ($W=0$), $\mathbf{h}=W_\text{in}\mathbf{x}$: $\mathbf{r}^*=\phi(W_\text{in}\mathbf{x})$ = 1 ANN layer. $W_\text{in}$: feedforward weights.
- Euler step $\Delta t=\tau_r$: $\mathbf{r}_{t+1}=\phi(W\mathbf{r}_t+\mathbf{h}_t)$ = vanilla [RNN](../dl/basics/module.md#rnn).
```

```{dropdown} Table: Neuron Models
| Model | State per neuron | Spikes | Biophysical | Cost | Use |
|:--|:--|:--|:--|:--|:--|
| Hodgkin–Huxley | $V,m,h,n$ (+ more channels) | Waveform | ✅ | High | Spike mechanisms |
| Multi-compartment | HH per compartment | Waveform | ✅ | Very high | Dendritic computation |
| FitzHugh–Nagumo | $v,w$ | Waveform (abstract) | ❌ | Low | Phase-plane analysis |
| Izhikevich | $V,u$ | Events | ❌ | Low | Large networks w/ diverse firing |
| LIF | $V$ | Events | Partly | Very low | Spiking network theory, SNNs |
| Firing-rate | $r$ | ❌ | ❌ | Lowest | Circuit dynamics, ANN link |
```

```{dropdown} Table: Biological Neuron vs ANN Unit
| Property | Biological neuron | ANN unit |
|:--|:--|:--|
| Output | Spikes (events) | Real number |
| Time | Continuous dynamics, ms–s memory | Static (or discrete RNN steps) |
| Sign of outgoing weights | Fixed per neuron (Dale) | Any |
| Input integration | Nonlinear dendrites | Weighted sum |
| Reliability | Stochastic release | Deterministic |
| Learning signal | Local + neuromodulators | Global backprop gradient |
```

```{attention} Q&A
:class: dropdown
*When is a rate model valid?*

- Many neurons firing asynchronously, or dynamics slow vs spike timing.
- Fails when precise timing or synchrony carries information.

*Which time constant is $\tau_r$?*

- Depends on the regime: membrane $\tau_m$ or synaptic $\tau_s$. Different derivations pick different ones → treat $\tau_r$ as a modeling choice.

*Is an ANN a brain model?*

- ANN unit ≈ steady-state rate neuron. That is all the mapping guarantees.
- Whether trained ANNs model brains → [Goal-Driven Modeling](theory.md#goal-driven-modeling).
```

&nbsp;
