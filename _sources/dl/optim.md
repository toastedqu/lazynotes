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

## Gradient Descent
- **What**: Update the params based on the gradient's size and direction.
	- **Gradient**: First-order derivative of the loss w.r.t. the corresponding param.
	- **Descent**: Subtract the gradient.
		- Objective: **Loss minimization**.
		- Assumption: Loss function is **convex**.
		- Descent in both cases:
			- $\frac{\partial\mathcal{L}}{\partial\mathcal{w}}>0\rightarrow \mathcal{L}\propto w\rightarrow$ Reduce $w$ to reduce $\mathcal{L}$ $\rightarrow$ Subtract the positive gradient.
			- $\frac{\partial\mathcal{L}}{\partial\mathcal{w}}<0\rightarrow \mathcal{L}\propto -w\rightarrow$ Raise $w$ to reduce $\mathcal{L}$ $\rightarrow$ Subtract the negative gradient.
- **Why**: Practical.
	- *Analytical solutions?* Often impossible.
	- *Search?* Impractical.
	- *Other optimization algorithms?* High computational cost.
	- *Second-order derivative?* High computational cost.
	- Gradient descent: practical, simple, intuitive, iterative, scalable, empirically effective, ...
- **How**: For each param:
	1. Calculate gradient.
	2. Subtract gradient.
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

## Momentum
- **What**: SGD + Cache of past movements.
- **Why**: SGD is:
	- too slow in flat regions.
	- oscillates too much in steep valley regions.
	- stuck in local minima.
- **How**: **Velocity**: Exponentially decaying moving average of past gradients.
	- At each param update step, add a fraction of the previous update to the curr gradient.
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
&\text{Init}: \quad v_0 &= 0  \quad \text{ Init} \\
&\text{EWMA (Exponentially Weighted MA)}: \quad v_t &= \beta v_{t-1} + (1 - \beta) g_t \\
&\text{Accumulation}: \quad v_t &= \beta v_{t-1} + g_t \\
&\text{Direct}: \quad v_t &= \beta v_{t-1} + \eta g_t
\end{align*}$$

2. Param update:

$$\begin{align*}
&w_t \leftarrow w_{t-1} - \eta v_t \quad &\text{EWMA & Accumulation} \\
&w_t \leftarrow w_{t-1} - v_t \quad &\text{Direct} \\
\end{align*}$$
```

### Nesterov Accelerated Gradient (NAG)

## Adaptive Learning Rate
### Adagrad (Adaptive Gradient Algorithm)
### Adadelta
### RMSprop (Root Mean Square Propagation)
### Adam (Adaptive Moment Estimation)
### AdamW (Adam with Weight Decay)
### Nadam (Nesterov-accelerated Adaptive Moment Estimation)
### AdaMax
### AMSGrad

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