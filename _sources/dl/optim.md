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
# Parameter Optimization
- **What**: Find the optimal params of the given model for the given task.
- **Why**: To best solve the task.

<br/>

## Gradient Descent
- **What**: Update the params based on the grad's size and direction.
	- **Gradient**: First-order derivative of the loss w.r.t. the corresponding param.
	- **Descent**: Subtract the grad.
		- Objective: **Loss minimization**.
		- Assumption: Loss function is **convex**.
		- Descent in both cases:
			- $\frac{\partial\mathcal{L}}{\partial\mathcal{w}}>0\rightarrow \mathcal{L}\propto w\rightarrow$ Reduce $w$ to reduce $\mathcal{L}$ $\rightarrow$ Subtract the positive grad.
			- $\frac{\partial\mathcal{L}}{\partial\mathcal{w}}<0\rightarrow \mathcal{L}\propto -w\rightarrow$ Raise $w$ to reduce $\mathcal{L}$ $\rightarrow$ Subtract the negative grad.
- **Why**: Practical.
	- *Analytical solutions?* Often impossible.
	- *Search?* Impractical.
	- *Other optimization algorithms?* High computational cost.
	- *Second-order derivative?* High computational cost.
	- Gradient descent: practical, simple, intuitive, iterative, scalable, empirically effective, ...
- **How**: For each param:
	1. Calculate grad.
	2. Subtract grad.
	3. Use new params to calculate loss.
	4. Repeat Steps 1-3 till training ends.

```{admonition} Math
:class: note, dropdown
Notations:
- IO:
	- $(x_i,y_i)$: Sample.
	- $\mathcal{D}_{\mathrm{mini}}$: Mini-batch.
	- $\mathcal{D}$: Dataset.
- Params:
    - $w_t$: Param at step $t$.
- Hyperparams:
    - $\eta$: Learning rate.

Types:
- **Stochastic (SGD)**:

$$
w_{t+1} \leftarrow w_t - \eta \frac{\partial \mathcal{L}(w_t; x_i,y_i)}{\partial w_t}
$$

- **Mini-Batch**:

$$
w_{t+1} \leftarrow w_t - \eta \frac{\partial \mathcal{L}(w_t; \mathcal{D}_{\mathrm{mini}})}{\partial w_t}
$$

- **Batch**:

$$
w_{t+1} \leftarrow w_t - \eta \frac{\partial L(w_t; \mathcal{D})}{\partial w_t}
$$
```

<br/>

## Momentum
- **What**: GD + Cache of past movements.
- **Why**: GD is:
	- too slow in flat regions.
	- oscillates too much in steep valley regions.
	- stuck in local minima.
- **How**: **Velocity**: Exponentially decaying moving average of past grads.
	- At each param update step, add a fraction of the previous update to the curr grad.
		- Flat region: Prev grad & Curr grad same direction $\rightarrow$ Velocity builds up $\rightarrow$ Faster convergence
		- Valley region: Prev grad & Curr grad diff direction $\rightarrow$ Velocity cancels out $\rightarrow$ Oscillations are dampened
	
```{admonition} Math
:class: note, dropdown
Notations:
- Params:
	- $w_t$: Param at step $t$.
- Hyperparams:
	- $\beta$: Momentum coeff.
	- $\eta$: Learning rate.
- Misc:
	- $g_t$: Gradient $\frac{\partial\mathcal{L}}{\partial w_{t-1}}$.
	- $v_t$: Velocity at step $t$.

Process:
1. Velocity update:

$$\begin{align*}
&\text{Init}: &&v_0 = 0\\
&\text{EWMA}: &&v_t = \beta v_{t-1} + (1 - \beta) g_t \\
&\text{Accumulation}: &&v_t= \beta v_{t-1} + g_t \\
&\text{Direct}: &&v_t= \beta v_{t-1} + \eta g_t
\end{align*}$$

2. Param update:

$$\begin{align*}
&\text{EWMA & Accumulation} &&w_t \leftarrow w_{t-1} - \eta v_t \\
&\text{Direct} &&w_t \leftarrow w_{t-1} - v_t 
\end{align*}$$
```

### NAG (Nesterov Accelerated Gradient)
- **What**: Momentum + "Look-ahead"
- **Why**: Momentum can sometimes jump over the global minimum.
	- It adds the accumulated momentum, THEN considers curr grad.
- **How**:
	1. Look ahead.
	2. Compute grad from the preliminary position (i.e., preliminary jump).
	3. Update param with this preliminary grad + accumulated momentum.

```{admonition} Math
:class: note, dropdown
Notations:
- Params:
	- $w_t$: Param at step $t$.
	- $\tilde{w}_t$: Preliminary param at step $t$.
- Hyperparams:
	- $\beta$: Momentum coeff.
	- $\eta$: Learning rate.
- Misc:
	- $g_t$: Gradient $\frac{\partial\mathcal{L}}{\partial w_{t-1}}$.
	- $\tilde{g}_t$: Preliminary grad $\frac{\partial\mathcal{L}}{\partial \tilde{w}_t}$ at step $t$.
	- $v_t$: Velocity at step $t$.

Process:
1. Look-ahead position:

$$
\tilde{w}_t = w_t - \beta v_{t-1}
$$

2. Velocity update (accumulation):

$$
v_t = \beta v_{t-1} + \tilde{g}_t
$$

3. Param update:

$$
w_t \leftarrow w_{t-1} - \eta v_t 
$$
```

<br/>

## Adaptive LR
### Adagrad (Adaptive Gradient)
- **What**: GD + Adaptive learning rate for each param.
- **Why**: A single, fixed learning rate is problematic:
	- Diff feature frequencies: Rare/Common features $\rightarrow$ Rarely/Commonly update their corresponding params $\rightarrow$ Need to take larger/smaller steps.
	- Diff param sizes: Large/Small params $\rightarrow$ Large/Small grads in nature
- **How**:
	1. Track past grads for each param.
	2. Scale learning rate by $\sqrt{\sum\text{grads}^2}$.
		- Large/Small past grads $\rightarrow$ Small/Large learning rate

```{admonition} Math
:class: note, dropdown
Notations:
- Params:
	- $w_t$: Param at step $t$.
- Hyperparams:
	- $\eta$: Global learning rate.
	- $\epsilon$: Small constant (prevent division by 0).
- Misc:
	- $g_t$: Gradient $\frac{\partial\mathcal{L}}{\partial w_{t-1}}$.
	- $G_t$: Accumulated squared gradients.

Process:
1. Grad accumulation:

$$\begin{align*}
&\text{Matrix form}: &&G_{t,ii}=G_{t-1,ii}+(g_{t,i})^2 \\
&\text{Vector form}: &&G_t=G_{t-1}+g_t\odot g_t
\end{align*}$$

2. Param update:

$$
w_t \leftarrow w_{t-1} - \frac{\eta}{\sqrt{G_t+\epsilon}}\odot g_t
$$
```

```{admonition} Q&A
:class: tip, dropdown
*Why square?*
- Positive + Penalizes larger grads more.

*Why sum?*
- Memory of how active the param has been throughout the entire training process.
```

### RMSprop (Root Mean Square Propagation)
- **What**: Adagrad w EWMA (Exponentially Weighted Moving Average) of squared grads instead of sum.
- **Why**:
	- Learning rate decays too much $\leftarrow$ Grad sum grows till training ends.
- **How**: ❌Grad sum, ✅EWMA of past squared grads $\rightarrow$ No infinite growth

```{admonition} Math
:class: note, dropdown
Notations:
- Params:
    - $w_t$: Param at step $t$.
- Hyperparams:
    - $\beta$: Decay rate for the EWMAs.
	- $\eta$: Learning rate.
    - $\epsilon$: Small constant (prevent division by 0).
- Misc:
    - $g_t$: Grad $\frac{\partial\mathcal{L}}{\partial w_{t-1}}$.
    - $G_t$: EWMA of squared grads.

Process:
1. Accumulate squared grads (EWMA):

$$
G_t=\beta G_{t-1}+(1-\beta)g_t\odot g_t
$$

2. Update param:

$$
w_t \leftarrow w_{t-1} - \frac{\eta}{\sqrt{G_t+\epsilon}}\odot g_t
$$
```

```{admonition} Q&A
:class: tip, dropdown
*Why name it RMS?*
- Square: $g^2_t$.
- Mean: EWMA.
- Root: $\frac{1}{\sqrt{G_t+\epsilon}}$.

*Why RMS?*
- Same unit.

*Why EWMA?*
- The older/newer the grad, the less/more influence it has on curr param update.
```

### Adadelta
- **What**: RMSprop w/o manually set global learning rate + EWMA on param updates.
- **Why**:
	- Performance is highly sensitive to learning rate.
- **How**:
    1.  ❌Grad sum, ✅EWMA of past squared grads ($G_t$) $\rightarrow$ No infinite growth
	2.  ❌Global LR, ✅EWMA of past squared param updates ($\Delta W_{t}$) $\rightarrow$ No LR decay
    3.  ❌Fixed LR, ✅Adaptive ratio between RMS of prev param updates & RMS of curr accumulated squared grads ($\frac{\text{RMS}[\Delta w]_{\text{prev}}}{\text{RMS}[g]_{\text{curr}}}$)

```{admonition} Math
:class: note, dropdown
Notations:
- Params:
    - $w_t$: Param at step $t$.
- Hyperparams:
    - $\beta$: Decay rate for the EWMAs.
    - $\epsilon$: Small constant (prevent division by 0).
- Misc:
    - $g_t$: Grad $\frac{\partial\mathcal{L}}{\partial w_t}$.
	- $\Delta w_t$: param update at step $t$.
    - $G_t$: EWMA of squared grads.
    - $\Delta W_{t}$: EWMA of squared param updates.

Process:
1.  Accumulate squared grads (EWMA):

$$
G_t = \beta G_{t-1} + (1-\beta) (g_t \odot g_t)
$$

2.  Calculate param update:

$$\begin{align*}
\Delta w_t &=-\frac{\sqrt{\Delta W_{t-1} + \epsilon}}{\sqrt{G_t + \epsilon}} \odot g_t \\
&=-\frac{\text{RMS}[\Delta w]_{t-1}}{\text{RMS}[g]_t}\odot g_t
\end{align*}$$

3.  Accumulate squared param updates (EWMA):**

$$
\Delta W_{t} = \beta \Delta W_{t-1} + (1-\beta) (\Delta w_t \odot \Delta w_t)
$$

4.  Update param:

$$
w_{t+1} = w_t + \Delta w_t
$$
```

```{admonition} Q&A
:class: tip, dropdown
*Why RMS ratio?*
1. It's in the exact **same unit** as the value it applies to.
2. $\frac{\text{RMS}[\Delta w]_{t-1}}{\text{RMS}[g]_t}$ has units "$\Delta$param/grad".
3. Multiply this by grad gives $\Delta$param.

*Why RMSprop instead of Adadelta?*
- ⬆️Empirical performance & convergence speed.
- ⬆️Simplicity & Interpretability.
- Adam is built upon RMSprop & outperforms everything above.
```

<br/>

## Adam (Adaptive Moment Estimation)
- **What**: Momentum + RMSprop.
- **Why**: Combine benefits from Momentum & RMSprop:
	- **Momentum = 1st moment (mean) of grads**
		- Momentum keeps track of the **direction** of recent grads.
		- Momentum speeds up optimization in that direction.
	- **RMSprop = 2nd moment (uncentered variance) of grads**
		- RMSprop keep track of the **magnitude** of recent grads.
		- RMSprop adapts LR for each param based on its grad size.
- **How**:
	1. Update 1st moment using new grads.
	2. Update 2nd moment using new squared grads.
	3. Bias Correction: EWMAs are init to 0 $\rightarrow$ They are biased toward 0 early in training $\rightarrow$ Need to correct it
	4. Update params.

```{admonition} Math
:class: note, dropdown
Notations:
- Params:
    - $w_t$: Param at step $t$.
- Hyperparams:
    - $\beta_1$: Decay rate for 1st moment. Default 0.9.
	- $\beta_2$: Decay rate for 2nd moment. Default 0.999.
	- $\eta$: Learning rate.
    - $\epsilon$: Small constant (prevent division by 0).
- Misc:
	- $g_t$: Grad $\frac{\partial\mathcal{L}}{\partial w_{t-1}}$.
    - $v_t$: 1st moment estimate at step $t$.
	- $G_t$: 2nd moment estimate at step $t$.

Process:
1.  Update 1st moment:

$$
v_t = \beta_1 v_{t-1} + (1 - \beta_1) g_t
$$

2. Update 2nd moment:

$$
G_t = \beta_2 G_{t-1} + (1-\beta_2) (g_t \odot g_t)
$$

3. Correct bias:

$$\begin{align*}
&\hat{v}_t=\frac{v_t}{1-\beta_1^t} \\
&\hat{G}_t=\frac{G_t}{1-\beta_2^t}
\end{align*}$$

4. Update params:

$$
w_t \leftarrow w_{t-1} - \frac{\eta}{\sqrt{\hat{G}_t+\epsilon}} \hat{v}_t
$$
```

```{admonition} Q&A
:class: tip, dropdown
*Pros?*
- ✅Best empirical performance & convergence speed.

*Cons?*
- ⬆️Memory cost.
- May still get stuck in local optima.
```

### AdamW (Adam with Weight Decay)
- **What**: Adam + Weight Decay.
- **Why**: L2 regularization.
	- L2 adds an additional change to the original grad: $\lambda w$.
	- $\rightarrow$ Designed to decay all weights toward 0, at a rate proportional to their magnitudes, regardless of grad history.
	- BUT in Adam, weights with large/small historical grads get a small/large learning rate, thus a small/large weight decay.
	- Adam & L2 conflict by design.
- **How**: Add an extra weight decay step after Adam.

```{admonition} Math
:class: note, dropdown
Notations:
- Params:
    - $w_t$: Param at step $t$.
- Hyperparams:
    - $\lambda$: Decay rate.
	- $\eta$: Learning rate.
- Misc:
	- $\Delta w_t^{\text{Adam}}$: Weight decay via Adam.

Final param update:

$$
w_t \leftarrow w_{t-1} - \Delta w_t^{\text{Adam}} - \eta\lambda w_{t-1}
$$
```

```{admonition} Q&A
:class: tip, dropdown
*Why not replace Adam?*
- ONLY applicable when L2 regularization is used (or set $\lambda=0$ for other cases for AdamW).
```

<!-- ## Nadam (Nesterov-accelerated Adaptive Moment Estimation)
## AdaMax
## AMSGrad -->

<!-- ## Second-Order
### L-BFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno)
### Hessian-Free Optimization -->

<!-- 
## Evolutionary
### Genetic Algorithms
### Particle Swarm Optimization (PSO)
### Differential Evolution (DE)
### Covariance Matrix Adaptation Evolution Strategy (CMA-ES)

## Misc
### FTRL (Follow The Regularized Leader)
### Yogi Optimizer
### RAdam (Rectified Adam)
### Lookahead Optimizer -->

<br/><br/>

# LR Scheduler
- **What**: Dynamically adjust LR during training.
- **Why**:
	- ⬇️LR $\rightarrow$ ⬇️Convergence speed $\rightarrow$ Training takes too long $\rightarrow$ We want fast
	- ⬆️LR $\rightarrow$ ⬆️Convergence speed $\rightarrow$ Overshoot global minima $\rightarrow$ We want fast initially but slow later
	- LR Scheduler: How about both?

## Decay
- **How**:
	1. Start with high LR.
	2. Gradually ⬇️ LR.

### Step Decay
- **How**:
	1. Init: LR, Decay rate & Step interval.
	2. Step function:
		1. Within each step interval, LR stays the same.
		2. When jumping to the next interval, LR drops.

```{admonition} Math
:class: note, dropdown
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

### Exponential Decay
- **Why**: Step Decay makes abrupt changes
	- $\rightarrow$ Unstable convergence & potential disruptions in training
	- $\rightarrow$ Need **smoother transition**
- **How**:
	1. Init: LR, Decay rate.
	2. At each step, multiply curr LR with decay rate.

```{admonition} Math
:class: note, dropdown
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

### Polynomial Decay
- **Why**: Step & Exponential decay can decay FOREVER & have no control over decay shape.
- **How**:
	1. Init: Initial LR, **Final LR**, Decay rate, #Decay steps, Power $p$.
		- $p=1$: Linear decay.
		- $p>1$: LR ⬇️ fast initially & slow when approaching final LR.
		- $p<1$ (uncommon): LR ⬇️ slow initially & fast when approaching final LR.
	2. At each step, compute curr LR.
	3. If decay process ends, LR stays at final LR.

```{admonition} Math
:class: note, dropdown
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

<br/>

## Cycle
- **What**: Cycle the LR between lower & upper bounds.
- **Why**: Decays are good, BUT they are heavily sensitive to hyperparams.
	- If LR ⬇️ too slow $\rightarrow$ Overshoot global minima
	- If LR ⬇️ too fast $\rightarrow$ No chance to explore further regions $\rightarrow$ Stuck in local minima & saddle points
	- Periodically increasing LR can help escape local minima & saddle points.

### Cyclical LR
- **How**:
	1. Init: Min LR, Max LR, Step size for half cycle.
	2. Choose a cycle policy (See [bckenstler/CLR](https://github.com/bckenstler/CLR)): 
		- **triangle**: In each cycle:
			1. LR linearly ⬆️ from min to max over step size.
			2. LR linearly ⬇️ from max to min over step size.
		- **triangle2**: Triangle BUT max LR is halved at the each of each cycle.
		- **exp_range**: Triangle BUT max LR decays exponentially.

```{admonition} Math
:class: note, dropdown
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

### Cosine Annealing
- **Why**: Cyclical LR starts with low LR, which slows the initial learning process.
	- We still want to start fast & also keep the advantage of cycles & smooth curves.
- **How**:
	1. Init: Min LR, Max LR, Step size for **full** cycle.
	2. For each cycle, anneal:
		1. $t=0\rightarrow\eta_t=\eta_{\max}$
		2. $t\in(0,S)\rightarrow\eta_t$⬇️ via cosine curve
		3. $t=S\rightarrow\eta_t=\eta_{\min}$

```{admonition} Math
:class: note, dropdown
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

```{admonition} Q&A
:class: tip, dropdown
*But why cosine?*
- First, cosine is NOT the only option that works.
- Cosine is smooth, and its 1st derivative (i.e., sine) is also smooth, and its 2nd derivative (i.e., cosine) is also smooth... $\rightarrow$ Cosine is infinitely smooth.
	- Smoothness ⬆️ $\rightarrow$ Training stability ⬆️
- Cosine follows the **slow-fast-slow** decay pattern:
	- Initial lingering high LR allows for broad exploration.
	- Rapid drop in LR allows for faster convergence.
	- Final lingering low LR allows for fine-tuning into a good minimum.
- Cosine works incredibly well with SGDR because of smoothness & slow-fast-slow.

*What's SGDR (SGD with Warm Restarts)?*
- The length of each cycle is no longer fixed but progressively longer.
- $T_i=T_0\times\lambda_\text{mul}^i$, where $\lambda_\text{mul}$ is a factor by which the cycle length is increased after each restart.
```

### One Cycle Policy
- **What**: Use ONLY 1 cycle for LR scheduling throughout training.
- **Why**: Super-convergence: training models to high accuracy in ⬇️⬇️ epochs.
- **How**:
	1. Warm-up: LR ⬆️ from min to max (e.g., Linear).
	2. Cool-down: LR ⬇️ from max to min (e.g., Cosine).
	3. Annihilation: LR ⬇️ below min.

```{admonition} Math
:class: note, dropdown
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

```{admonition} Q&A
:class: tip, dropdown
*Why does the OG paper schedule momentum together with LR?*
- Pure empirical findings:
	- ⬆️LR $\rightarrow$ ⬇️momentum helps prevent divergence.
	- ⬇️LR $\rightarrow$ ⬆️momentum helps continue making progress.
```

<br/>

## ReduceLROnPlateau
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

```{admonition} Q&A
:class: tip, dropdown
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

## Warmup
### Linear Warmup
### Exponential Warmup
### Gradual Warmup