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
# Optimization
- **What**: Adjust a model's params and/or hyperparams to achieve the best performance according to some criterion (i.e., Machine Learning!)
- **Why**: **ML = Function Approximation**.
	- There is an underlying function which maps features to targets, but we don't know what it is.
	- Optimization aims to approximate this function as accurately as possible.

This page covers methods for params AND hyperparams (because at the end of the day it's just optimization, duh?)

This page ONLY covers methods applicable to multiple models and/or objectives. For specific optimization methods, please refer to their corresponding model page.

This page does NOT cover [Gradient Descent & family](../dl/optim.md).

## Second-Order
### Newton's Method
### BFGS (Broyden–Fletcher–Goldfarb–Shanno)
### L-BFGS (Limited-Memory BFGS)
### DFP (Davidon–Fletcher–Powell)


## EM (Expectation-Maximization)
- **What**: Compute ML/MAP estimates with **unobserved latent variables** and/or **missing data**.
- **Why**:
	- If data were fully observed, then ML/MAP estimates would be easy to compute.
	- If we observe the latent vars, we observe the full data.
	- BUT, direct MLE/MAP is hard $\leftarrow$ Sum/Integrate over all hidden configs $\leftarrow$ Bayes' Theorem
	- $\rightarrow$ Iteratively observe & optimize.
- **How**: Iterate between Expectation & Maximization.
	- **Expectation**: Given curr params, estimate latent vars.
	- **Maximization**: Given curr data, optimize params to maximize data prob.

```{note} Math
:class: dropdown
Notation:
- IO:
	- $X={\mathbf{x}_1,\cdots,\mathbf{x}_m}$: Observed data.
- Params:
	- $\bm{\theta}$: Model params.
- Misc:
	- $\mathbf{z}_i\in\mathcal{Z}$: Latent variable for sample $i$.
	- $q_i(\mathbf{z}_i)$: Prob distribution of $z_i$.

Goal: Maximize (log) likelihood of observed data:

$$
L(\theta)=\log P(X|\theta)=\sum_{i=1}^{m}\log\left[\sum_{\mathbf{z}_i\in\mathcal{Z}}q_i(\mathbf{z}_i)\frac{p(\mathbf{x}_i,\mathbf{z}_i|\bm{\theta})}{}\right]
$$
```

## Search
### Grid Search
### Random Search

## SMBO (Sequential Model-Based Optimization)
### GP–Based Bayesian Optimization
### TPE (Tree‑structured Parzen Estimator)
### SMAC (Sequential Model-based Algorithm Configuration)
### Multi‑objective SMBO
### Constrained SMBO
### BOHB (Bayesian Optimization with Hyperband)

## Evolution
### GA (Genetic Algorithms)
- **What**: Population-based metaheuristic inspired by **natural selection**.
- **Why**: In the following scenarios, it's hard to estimate value functions BUT easy to optimize policies alone:
	- Partially observable environments.
	- Non-differentiable objectives (e.g., black-box classifiers).
	- Multi-modal search spaces with many local optima.
	- High-dimensional, mixed (continuous/discrete) action spaces.
- **How**:
	1. **Initialization**: Randomly generate an initial population of hyperparam configs.
	2. **Evaluation**: Compute **fitness** for each config.
	3. **Selection**: Choose parents based on fitness.
		- Selection probability ∝ Fitness.
	4. **Crossover**: For each new offspring, select 2 parents and blend their hyperparameters.
	5. **Mutation**: Perturb offspring hyperparameters at a small probability.
	6. **Replacement**: Form next generation based on replacement rules:
		- **Generational**: Replace entire population with new offspring.
		- **Steady-State**: Replace worst individuals gradually.
		- **Elitism**: Retain top-$k$ individuals from current generation.
	7. **Iteration**: Repeat Step 3-5.
	8. **Termination**: Stop when reaching #iterations / convergence.

```{attention} Q&A
:class: dropdown
*Pros?*
- ✅Parallelism.
- ✅Flexible $\leftarrow$ Tolerate arbitrary hyperparam types
- ✅Escapes local optima $\leftarrow$ Randomness
- ❌Gradient.

*Cons?*
- ❌Sample efficiency $\leftarrow$ Requires many samples to work well.
- ❌Stability $\leftarrow$ Randomness
- ⬆️Computational cost (despite parallelism).
- ✅Hyperparameter Tuning.
```

```{note} Math
:class: dropdown
Example: Basic Evolution Strategy (ES)

Notation:
- IO:
	- $\mathcal{P}={\mathbf{x}_i,\cdots,\mathbf{x}_i|\mathbf{x}_i\in\mathbb{R}^n}$: Population.
- Hyperparams:
	- $f:\mathbb{R}^n\to\mathbb{R}$: Fitness function.
	- $\alpha\in\text{Uniform(0,1)}$: Crossover rate (i.e., Proportion of genes from Parent $i$ passed to its offspring).
		- Typically 0.6–1.0 for real-coded GAs.
	- $p_\mathrm{mut}$: Mutation rate (i.e., Probability of perturbing the offspring).
		- Typically 0.01-0.1 for stability.
- Misc:
	- $p_i$: Probability of $\mathbf{x}^{(i)}$ being selected as a parent.
	- $\mathbf{o}$: Offspring between Parents $i$ & $j$.


Procedure:
1. Selection (Roulette Wheel / Fitness Proportional):

   $$
   p_i = \frac{f(\mathbf{x}_i)}{\sum_{j=1}^m f(\mathbf{x}_j)}
   $$

2. Crossover: Generate offspring for parents $\mathbf{x}_i\ \&\ \mathbf{x}_j$:

   $$
   \mathbf{o}=\alpha\mathbf{x}_i+(1-\alpha)\mathbf{x}_j
   $$

3. Mutation (Gaussian): Each gene of offspring is perturbed with probability $p_{\mathrm{mut}}$:

   $$
   \mathbf{x}_k\leftarrow\begin{cases}
     \mathbf{o}_k+\epsilon & \epsilon\sim\mathcal{N}(0,\sigma^2) \\
     \mathbf{o}_k & \text{otherwise}
   \end{cases}
   $$
```

### GP (Genetic Programming)
### CMA‑ES (Covariance Matrix Adaptation Evolution Strategy)
### DE (Differential Evolution)