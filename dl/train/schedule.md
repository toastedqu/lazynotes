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
# LR Schedule
Rules that change the LR during training.

Notations:
- $t$: Training step.
- $\eta_t$: LR at step $t$.
- $\eta_0$: Initial (peak) LR.

&nbsp;

## LR Scheduler
- **What**: Dynamically adjust LR during training.
- **Why**:
	- ⬇️LR → ⬇️Convergence speed → Training takes too long → We want fast
	- ⬆️LR → ⬆️Convergence speed → Overshoot global minima → We want fast initially but slow later
	- LR Scheduler: How about both?

### Decay
- **How**:
	1. Start with high LR.
	2. Gradually ⬇️ LR.

#### Step Decay
- **How**:
	1. Init: LR, Decay rate & Step interval.
	2. Step function:
		1. Within each step interval, LR stays the same.
		2. When jumping to the next interval, LR drops.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\eta_t$: LR at step $t$.
- Hyperparams:
	- $\eta_0$: Initial LR.
	- $\gamma$: Decay rate.
	- $s$: Step interval.

Scheduler:

$$
\eta_t\leftarrow\eta_0\cdot\gamma^{\lfloor\frac{t}{s}\rfloor}
$$

e.g., if $s=10$
- $\eta_5=\eta_0$
- $\eta_{10}=\eta_0\cdot\gamma$
- $\eta_{15}=\eta_0\cdot\gamma$
- $\eta_{20}=\eta_0\cdot\gamma^2$
- ...
```

&nbsp;

#### Exponential Decay
- **Why**: Step Decay makes abrupt changes
	- → Unstable convergence & potential disruptions in training
	- → Need **smoother transition**
- **How**:
	1. Init: LR, Decay rate.
	2. At each step, multiply curr LR with decay rate.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\eta_t$: LR at step $t$.
- Hyperparams:
	- $\eta_0$: Initial LR.
	- $\gamma$: Decay rate.

Scheduler:

$$
\eta_t\leftarrow\eta_{t-1}\cdot\gamma
$$
```

&nbsp;

#### Polynomial Decay
- **Why**: Step & Exponential decay can decay FOREVER & have no control over decay shape.
- **How**:
	1. Init: Initial LR, **Final LR**, Decay rate, #Decay steps, Power $p$.
		- $p=1$: Linear decay.
		- $p>1$: LR ⬇️ fast initially & slow when approaching final LR.
		- $p<1$ (uncommon): LR ⬇️ slow initially & fast when approaching final LR.
	2. At each step, compute curr LR.
	3. If decay process ends, LR stays at final LR.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\eta_t$: LR at step $t$.
- Hyperparams:
	- $\eta_0$: Initial LR.
	- $\eta_\text{end}$: Final LR.
	- $\gamma$: Decay rate.
	- $p$: Power.
	- $T$: #Decay steps.

Scheduler:

$$
\eta_t\leftarrow(\eta_0-\eta_\text{end})\cdot\left(1-\frac{t}{T}\right)^p+\eta_\text{end}
$$
```

&nbsp;

### Cycle
- **What**: Cycle the LR between lower & upper bounds.
- **Why**: Decays are good, BUT they are heavily sensitive to hyperparams.
	- If LR ⬇️ too slow → Overshoot global minima
	- If LR ⬇️ too fast → No chance to explore further regions → Stuck in local minima & saddle points
	- Periodically increasing LR can help escape local minima & saddle points.

#### Cyclical LR
- **How**:
	1. Init: Min LR, Max LR, Step size for half cycle.
	2. Choose a cycle policy (See [bckenstler/CLR](https://github.com/bckenstler/CLR)): 
		- **triangle**: In each cycle:
			1. LR linearly ⬆️ from min to max over step size.
			2. LR linearly ⬇️ from max to min over step size.
		- **triangle2**: Triangle BUT max LR is halved at the each of each cycle.
		- **exp_range**: Triangle BUT max LR decays exponentially.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\eta_t$: LR at step $t$.
- Hyperparams:
	- $\eta_{\min}$: Min LR.
	- $\eta_{\max}$: Max LR.
	- $s$: Step size for half cycle.
	- $\gamma$: Decay rate.
- Misc:
	- $c=\lfloor 1+\frac{t}{2s} \rfloor$: Cycle number.
	- $x=\left|\frac{t}{s}-2c+1\right|$: Fractional distance into curr half-cycle.

Scheduler:

$$\begin{align*}
&\text{triangle:}  &&\eta_t\leftarrow\eta_{\min}+(\eta_{\max}-\eta_{\min})\cdot\max(0,1-x) \\
&\text{triangle2:} &&\eta_{\max,c}=\frac{\eta_{\max,0}}{2^{c-1}} \\
&\text{exp range:} &&\eta_{\max,t}=\eta_{\max,0}\cdot\gamma^t \\
\end{align*}$$
```

&nbsp;

#### Cosine Annealing
- **Why**: Cyclical LR starts with low LR, which slows the initial learning process.
	- We still want to start fast & also keep the advantage of cycles & smooth curves.
- **How**:
	1. Init: Min LR, Max LR, Step size for **full** cycle.
	2. For each cycle, anneal:
		1. $t=0\rightarrow\eta_t=\eta_{\max}$
		2. $t\in(0,S)\rightarrow\eta_t$⬇️ via cosine curve
		3. $t=S\rightarrow\eta_t=\eta_{\min}$

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\eta_t$: LR at step $t$.
- Hyperparams:
	- $\eta_{\min}$: Min LR.
	- $\eta_{\max}$: Max LR.
	- $S$: Step size for full cycle.

Scheduler:

$$
\eta_t = \eta_{\min} + \frac{1}{2} (\eta_{\max} - \eta_{\min}) \left(1 + \cos\left(\frac{t}{S}\pi\right)\right)
$$
```

```{attention} Q&A
:class: dropdown
*But why cosine?*
- First, cosine is NOT the only option that works.
- Cosine is smooth, and its 1st derivative (i.e., sine) is also smooth, and its 2nd derivative (i.e., cosine) is also smooth... → Cosine is infinitely smooth.
	- Smoothness ⬆️ → Training stability ⬆️
- Cosine follows the **slow-fast-slow** decay pattern:
	- Initial lingering high LR allows for broad exploration.
	- Rapid drop in LR allows for faster convergence.
	- Final lingering low LR allows for fine-tuning into a good minimum.
- Cosine works incredibly well with SGDR because of smoothness & slow-fast-slow.

*What's SGDR (SGD with Warm Restarts)?*
- The length of each cycle is no longer fixed but progressively longer.
- $T_i=T_0\times\lambda_\text{mul}^i$, where $\lambda_\text{mul}$ is a factor by which the cycle length is increased after each restart.
```

&nbsp;

#### One Cycle Policy
- **What**: Use ONLY 1 cycle for LR scheduling throughout training.
- **Why**: Super-convergence: training models to high accuracy in ⬇️⬇️ epochs.
- **How**:
	1. Warm-up: LR ⬆️ from min to max (e.g., Linear).
	2. Cool-down: LR ⬇️ from max to min (e.g., Cosine).
	3. Annihilation: LR ⬇️ below min.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\eta_t$: LR at step $t$.
- Hyperparams:
	- $\eta_{\min}$: Min/Initial LR.
	- $\eta_{\max}$: Max LR, higher than Cyclical LR.
	- $\eta_\text{final}$: Final LR, smaller than Cyclical LR.
	- $T$: #Epochs.
	- $\%_\text{up}$: Percentage of up iterations.
	- $\%_\text{anni}$: Percentage of annihilation iterations.
- Misc:
	- $T_\text{up}=\lfloor \%_\text{up}\cdot T \rfloor$: #Warm-up iterations.
	- $T_\text{anni}=\lfloor \%_\text{anni}\cdot T \rfloor$: #Annihilation iterations.
	- $T_\text{down}=T-T_\text{up}-T_\text{anni}$: #Cool-down iterations.

Scheduler:

$$\begin{align*}
&\text{Warm-up (Linear):} &&\eta_t = \eta_{\min} + (\eta_{\max} - \eta_{\min}) \frac{t}{T_\text{up}} \\
&\text{Cool-down (Cosine):} &&\eta_t = \eta_{\min} + \frac{1}{2} (\eta_{\max} - \eta_{\min}) \left(1 + \cos\left(\frac{t-T_\text{up}}{T_\text{down}}\pi\right)\right) \\
&\text{Annihilation (Cosine):} &&\eta_t = \eta_{final} + \frac{1}{2}(\eta_{min} - \eta_{final}) \left(1 + \cos\left(\frac{t-T_\text{up}-T_\text{down}}{T_{anni}}\pi\right)\right)
\end{align*}$$
```

```{attention} Q&A
:class: dropdown
*Why does the OG paper schedule momentum together with LR?*
- Pure empirical findings:
	- ⬆️LR → ⬇️momentum helps prevent divergence.
	- ⬇️LR → ⬆️momentum helps continue making progress.
```

&nbsp;

### ReduceLROnPlateau
- **What**: Reduce LR when the model is on a plateau of **performance** (i.e., stuck).
- **Why**: No scheduler above cares about actual metrics, BUT we do.
- **How**:
	1. Init:
		- Metric (e.g., validation loss).
		- Improvement: Whether to go up/down on the monitored metric, and how much counts as a significant improvement.
		- Patience: #Epochs to wait for an improvement in the monitored metric before concluding the training hits a plateau.
		- Min LR.
	2. Monitor:
		- If the metric never improves during patience, LR is reduced by a multiplicative factor ($\eta_\text{new}=\eta_\text{old}\cdot\lambda_\text{mul}$).
		- Else, recount patience from improvement event.

```{attention} Q&A
:class: dropdown
*Seems promising. Why use other schedulers then?*
- Proactive vs Reactive:
	- ReduceLROnPlateau is **reactive**: It only works when a problem occurs.
	- Other schedulers are **proactive**: They strictly follow a pre-defined schedule regardless of metric performance.
- Hyperparameter sensitivity:
	- Other schedulers also suffer from this, BUT ReduceLROnPlateau suffers even more:
		- ❌Metric? GG.
		- ❌Improvement? GG.
		- ❌Patience? GG.
		- We could waste epochs being stuck without doing anything.
```

&nbsp;

## Warmup
- **What**: Start with "small LR → large LR".
- **Why**: Large initial LR can go wrong:
	- Large batches → Accurate initial grads $\xrightarrow{+\text{large LR}}$ Overshooting
	- Adaptive Optimizers → Super noisy average grads due to few samples in history $\xrightarrow{+\text{large LR}}$ Wild fluctiations
	- Early local minima/saddle points.

### Linear Warmup
- **How**:
	1. Init: initial LR, target LR, #Epochs for warmup.
	2. For each step, increase initial LR toward target LR by a fixed amount (i.e., linearly).

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\eta_t$: LR at step $t$.
- Hyperparams:
	- $\eta_0$: Initial LR.
	- $\eta_{\text{tgt}}$: Target LR.
	- $T$: #Epochs for warmup.

Warmup:

$$
\eta_t = \eta_0 + (\eta_{\text{tgt}} - \eta_0) \times \frac{t}{T}
$$
```

&nbsp;

### Exponential Warmup
- **Why**: Exponential function offers an even gentler start & a faster ramp-up towards the target → Amplifies the reasons for using warmups
	- However, empirically, it's not necessarily better than Linear.
- **How**:
	1. Init: initial LR, target LR, #Epochs for warmup.
	2. For each step, increase initial LR toward target LR at a fixed rate (i.e., exponentially).

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\eta_t$: LR at step $t$.
- Hyperparams:
	- $\eta_0$: Initial LR.
	- $\eta_{\text{tgt}}$: Target LR.
	- $T$: #Epochs for warmup.

Warmup:

$$
\eta_t = \eta_0 * \left(\frac{\eta_{\text{tgt}}}{\eta_0}\right)^{\frac{t}{T}}
$$
```

&nbsp;

## WSD
- **Name**: Warmup-Stable-Decay {cite:p}`hu2024minicpm`
    - ⚠️ Name & LLM popularity: MiniCPM (2024). The shape is older: constant LR + final cooldown was earlier called a "trapezoidal" schedule in vision work.
- **What**: Warmup → constant peak LR → short final decay.
- **Why**: [Cosine](#cosine-annealing) ties the LR at every step to the total steps $T$.
    - Mid-run cosine checkpoints are ❌finished & ❌reusable for a longer run.
    - Extend training → restart the schedule → pay again.
    - WSD: stable phase is independent of $T$ → branch a decay from any checkpoint.
- **How**:
    1. **Warmup**: LR ⬆️ to $\eta_{\max}$ (e.g., [linear](#linear-warmup)).
    2. **Stable**: hold $\eta_{\max}$ for most of training. Save checkpoints.
    3. **Decay**: over the last ~10–20% of steps, LR ⬇️ to ~0 → loss drops sharply.
    4. Need a longer run? Resume from a stable-phase (pre-decay) checkpoint at $\eta_{\max}$.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\eta_{\max}$: Peak LR.
    - $\eta_{\min}$: Final LR (often 0).
    - $T_\text{w}$: Warmup end step.
    - $T_\text{s}$: Stable end step (decay start).
    - $T$: Total steps.
- Misc:
    - $T_\text{d}=T-T_\text{s}$: #Decay steps.
    - $\tau=\frac{t-T_\text{s}}{T_\text{d}}\in[0,1]$: Decay progress.
    - $f(\tau)$: Decay shape. Decreasing, $f(0)=1$, $f(1)=0$.

Scheduler:

$$
\eta_t=\begin{cases}
\eta_{\max}\frac{t}{T_\text{w}} & t<T_\text{w}\\
\eta_{\max} & T_\text{w}\le t<T_\text{s}\\
\eta_{\min}+(\eta_{\max}-\eta_{\min})\,f(\tau) & T_\text{s}\le t\le T
\end{cases}
$$

Decay shapes:

$$\begin{align*}
&\text{Linear:} &&f(\tau)=1-\tau\\
&\text{Cosine:} &&f(\tau)=\tfrac{1}{2}\left(1+\cos\pi\tau\right)\\
&\text{1-sqrt:} &&f(\tau)=1-\sqrt{\tau}
\end{align*}$$

- MiniCPM used an exponential decay.
```

````{important} Code
:class: dropdown
```python
import math

class WSD:
    def __init__(self, lr_max, T_warmup, T_stable, T_total, lr_min=0.0, shape="1-sqrt"):
        self.lr_max, self.lr_min, self.shape = lr_max, lr_min, shape
        self.T_warmup, self.T_stable, self.T_total = T_warmup, T_stable, T_total

    def __call__(self, t):
        if t < self.T_warmup:
            return self.lr_max * t / self.T_warmup
        if t < self.T_stable:
            return self.lr_max  ## independent of T_total -> reusable checkpoints
        tau = min(1.0, (t - self.T_stable) / (self.T_total - self.T_stable))
        f = {"linear": 1 - tau,
             "cosine": 0.5 * (1 + math.cos(math.pi * tau)),
             "1-sqrt": 1 - math.sqrt(tau)}[self.shape]
        return self.lr_min + (self.lr_max - self.lr_min) * f

## Example: 100 steps, 10 warmup, decay over the last 20
sched = WSD(lr_max=1e-3, T_warmup=10, T_stable=80, T_total=100)
print([round(sched(t), 6) for t in (5, 50, 80, 95, 100)])
## [0.0005, 0.001, 0.001, 0.000134, 0.0]

## Plug into torch: LambdaLR multiplies the optimizer's base LR (set it to lr_max)
## torch.optim.lr_scheduler.LambdaLR(opt, lambda t: sched(t) / sched.lr_max)
```
````

```{attention} Q&A
:class: dropdown
*WSD vs cosine?*
- Cosine: LR depends on $T$ at every step → 1 run = 1 budget.
- WSD: only the decay depends on $T$ → 1 stable run + short decays = many budgets.
- Final loss: ≈ cosine w/ a cooldown of ≤20% of steps. Longer cooldown or the (1-sqrt) shape can beat cosine. {cite:p}`hagele2024scaling`

*Why does it allow continual training & checkpoint reuse?*
- Stable-phase checkpoints sit at $\eta_{\max}$ → resume w/o re-warmup or schedule mismatch.
- Keep the pre-decay checkpoint. A decayed checkpoint needs the LR raised again to continue → loses part of the decay gain.
- Scaling-law studies: 1 stable run per model size + decay branches at several token counts → a data-scaling curve for ≈ the cost of the longest run plus the decays (MiniCPM).

*Why does the loss drop sharply during the decay?*
- High LR → iterate bounces between the steep walls of the loss valley (noise-dominated loss) while it still moves along the valley floor.
- Decay → bounce amplitude ⬇️ → iterate settles toward the floor → the hidden progress becomes visible.
- ⚠️ This "river valley" picture is a conjecture, ❌proven for real LLM losses.

*How long & what shape for the decay phase?*
- Length: MiniCPM: ~10% of tokens is enough, 2.5% is too short.
- Shape:
    - (1-sqrt) > linear in Hägele et al.'s runs.
    - (1-sqrt): infinite slope at $\tau=0$ → LR drops fast at the start of the decay, then lingers near $\eta_{\min}$.
    - MiniCPM: exponential.

*What data goes into the decay phase?*
- MiniCPM: mix high-quality & SFT data into the decay phase → the final low-LR phase trains on the best data.

*WSD vs weight averaging?*
- [SWA](train.md#swa) over the constant-LR phase recovers much of the decay gain, but a gap to a real cooldown remains (Hägele et al.).
- [Schedule-Free](optim.md#schedule-free): ❌decay at all; averaging replaces it.

*Cons?*
- Extra hyperparams: decay start, length, shape.
- Stable-phase loss looks worse than cosine at the same step → misleading mid-run comparisons.
- Must store pre-decay checkpoints.
```

&nbsp;
