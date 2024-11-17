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
# Genetic Algorithm
- **What**: Methods that are analogous to biological evolution, consisting of:
	- **Population**: A set of candidate solutions.
	- **Fitness**: How good each candidate is.
	- **Selection**: How to choose the best candidates to reproduce.
	- **Crossover**: How to combine parts of two candidates to create offspring.
	- **Mutation**: How to randomly alter candidates to introduce variability.
	- **Iteration**: Repetition.
- **Why**: In the following scenarios, it's hard to estimate value functions and easy to optimize policies alone:
	- Partially observable environments.
	- High-dimensional, continuous action spaces.
- **How**:
	1. Apply multiple static policies to multiple instances of the environment over a certain period of time.
	2. Policies that obtain max rewards, together with random variations, are carried over to the next generation of policies.
	3. Repeat.
- **Pros**:
	- Parallelism.
	- Scalability.
	- Can escape local optima <- randomness.
	- Direct Policy Optimization -> efficient learning.
	- No differentiation -> efficient learning.
	- Suitable for sparse rewards <- evaluate policies by total reward.
- **Cons**:
	- Sample inefficiency <- requires many samples to work well.
	- Instability <- randomness.

```{admonition} Math
:class: note, dropdown
Example of the most basic evolutionary algorithm.

Notations:
- IO:
	- $\mathcal{X}=\{\mathbf{x}_{1},\cdots,\mathbf{x}_{m}\}$: Population.
		- $\mathbf{x}_{i}\in\mathbb{R}^n$: Sample.
	- $\mathbf{x}^*\in\mathbb{R}^n$: Optimal sample.

Procedure:
- Fitness:

	$$
	f:\mathbb{R}^n\to\mathbb{R}
	$$

- Selection: Roulette Wheel

	$$\begin{align}
	&p_{i}=\frac{f(\mathbf{x}_{i})}{\sum_{j=1}^mf(\mathbf{x}_{j})} \\ \\
	&P_{t}^{(i)}=\mathbf{x}_{i}
	\end{align}$$
	- $p_{i}$: Probability of $\mathbf{x}_{i}$ being selected as a parent.
	- $P_{t}^{(i)}$: The $i$th sample as a parent at iteration $t$.

- Crossover: Single, real-valued

	$$
	\text{O}_{t}=\alpha P_{t}^{(i)}+(1-\alpha)P_{t}^{(j)}
	$$
	- $O_{t}$: The single offspring between parents $i$ and $j$ at iteration $t$.
	- $\alpha$: The proportion of genes of parent $i$ passed to its offspring.

- Mutation: Gaussian, at a low probability $p_\text{mut}$

	$$
	P_{t+1}=O_{t}+\epsilon,\ \ \ \epsilon\sim N(0,\sigma^2)
	$$
```