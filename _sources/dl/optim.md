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

# Gradient Descent
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

# Momentum
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

## Nesterov Accelerated Gradient (NAG)
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

# Adaptive Learning Rate
## Adagrad (Adaptive Gradient)
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
	- $v_t$: Velocity at step $t$.

Process:
1. Grad accumulation:

$$\begin{align*}
&\text{Matrix form}: &&G_{t,ii}=G_{t-1,ii}+(g_{t,i})^2 \\
&\text{Vector form}: &&G_t=G_{t-1}+g_t\odot g_t
\end{align*}$$

2. Param update:

$$
w_t \leftarrow w_{t-1} - \frac{\eta}{\sqrt{G_t+\epsilon}} v_t
$$
```

```{admonition} Q&A
:class: tip, dropdown
*Why square?*
- Positive + Penalizes larger grads more.

*Why sum?*
- Memory of how active the param has been throughout the entire training process.
```

## Adadelta
- **What**: Adagrad w/o manually set global learning rate.
- **Why**:
	- Performance is highly sensitive to learning rate.
	- Learning rate decays too much $\leftarrow$ Grad sum grows till training ends.
- **How**:
    1.  ~~Grad sum~~ EMA (Exponentially Moving Average) of past squared grads ($E[g^2]_t$) $\rightarrow$ No infinite growth
	2.  ~~Global LR~~ EMA of past squared param updates ($E[\Delta w^2]_t$) $\rightarrow$ No LR decay
    3.  ~~Fixed LR~~ Adaptive ratio between RMS of prev param updates & RMS of curr accumulated squared grads ($\frac{\text{RMS}[\Delta w]_{\text{prev}}}{\text{RMS}[g]_{\text{curr}}}$)

```{admonition} Math
:class: note, dropdown
Notations:
- Params:
    - $w_t$: Param at step $t$.
- Hyperparams:
    - $\rho$: Decay rate for the EMAs.
    - $\epsilon$: Small constant (prevent division by 0).
- Misc:
    - $g_t$: Grad $\frac{\partial\mathcal{L}}{\partial w_t}$.
	- $\Delta w_t$: param update at step $t$.
    - $E[g^2]_t$: EMA of squared grads.
    - $E[\Delta w^2]_t$: EMA of squared param updates.

Process:
1. Init:
	- Accumulated squared gradients: $E[g^2]_0 = 0$
	- Accumulated squared updates: $E[\Delta w^2]_0 = 0$
2. For each step $t$:
	1.  Accumulate squared grads (EMA):

		$$
		E[g^2]_t = \rho E[g^2]_{t-1} + (1-\rho) (g_t \odot g_t)
		$$

	2.  Calculate param update:

		$$\begin{align*}
		\Delta w_t &=-\frac{\sqrt{E[\Delta w^2]_{t-1} + \epsilon}}{\sqrt{E[g^2]_t + \epsilon}} \odot g_t \\
		&=-\frac{\text{RMS}[\Delta w]_{t-1}}{$\text{RMS}[g]_t$}\odot g_t
		\end{align*}$$

	3.  Accumulate squared param updates (EMA):**

		$$
		E[\Delta w^2]_t = \rho E[\Delta w^2]_{t-1} + (1-\rho) (\Delta w_t \odot \Delta w_t)
		$$

	4.  Apply param update:

		$$
		w_{t+1} = w_t + \Delta w_t
		$$
```

```{admonition} Q&A
:class: tip, dropdown
*Why RMS?*
1. It's in the exact **same unit** as the value it applies to.
2. $\frac{\text{RMS}[\Delta w]_{t-1}}{$\text{RMS}[g]_t$}$ has units "$\Delta$param/grad".
3. Multiply this by grad gives $\Delta$param.

*Why sum?*
- Memory of how active the param has been throughout the entire training process.
```


## RMSprop (Root Mean Square Propagation)
## Adam (Adaptive Moment Estimation)
## AdamW (Adam with Weight Decay)
## Nadam (Nesterov-accelerated Adaptive Moment Estimation)
## AdaMax
## AMSGrad

## Second-Order
### L-BFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno)
### Hessian-Free Optimization

## Evolutionary
### Genetic Algorithms
### Particle Swarm Optimization (PSO)
### Differential Evolution (DE)
### Covariance Matrix Adaptation Evolution Strategy (CMA-ES)

## Misc
### FTRL (Follow The Regularized Leader)
### Yogi Optimizer
### RAdam (Rectified Adam)
### Lookahead Optimizer

# Scheduler
## Basic
### Step Decay
### Exponential Decay
### Polynomial Decay

## Advanced
### Cyclical Learning Rate (CLR)
### One Cycle Policy
### Cosine Annealing
### ReduceLROnPlateau

## Warmup
### Linear Warmup
### Exponential Warmup
### Gradual Warmup