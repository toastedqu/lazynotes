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
    - $w$: Param.
- Hyperparams:
    - $\alpha$: Learning rate.

Types:
- **Stochastic (SGD)**:

$$
w \leftarrow w - \alpha \frac{\partial \mathcal{L}(w; x_i,y_i)}{\partial w}
$$

- **Mini-Batch**:

$$
w \leftarrow w - \alpha \frac{\partial \mathcal{L}(w; \mathcal{D}_{\mathrm{mini}})}{\partial w}
$$

- **Batch**:

$$
w \leftarrow w - \alpha \frac{\partial L(w; \mathcal{D})}{\partial w}
$$
```

## Momentum
### Momentum
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