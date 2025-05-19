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
# Hyperparameter Optimization

# Search
## Grid Search
## Random Search

# Sequential Model (SMBO)
## GP–Based Bayesian Optimization
## Tree‑Structured Parzen Estimator (TPE)
## Sequential Model-based Algorithm Configuration (SMAC)
## Bayesian Neural Network Surrogates
## Multi‑objective SMBO
## Constrained SMBO
## BOHB (Bayesian Optimization with Hyperband)
## FABOLAS (Fast Bayesian Optimization of Machine Learning Hyperparameters on Large Datasets)

# Evolution
## Genetic Algorithms (GA)
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

```{admonition} Q&A
:class: tip, dropdown
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

```{admonition} Math
:class: note, dropdown
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

## Genetic Programming (GP)
## Covariance Matrix Adaptation Evolution Strategy (CMA‑ES)
## Differential Evolution (DE)
## Particle Swarm Optimization (PSO)
## Population‑Based Training (PBT)

# Gradient
## Hypergradient Descent
## Implicit Differentiation (Reverse‑Mode)
## Differentiable Architecture Search (DARTS‑style)

<!-- # Multi‑Fidelity & Bandit‑Based Methods
4.1. Successive Halving
4.2. Hyperband
4.3. Asynchronous Successive Halving (ASHA)
4.4. Population‑Based Bandits -->