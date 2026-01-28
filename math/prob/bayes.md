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
# Bayesian
Study notes from {cite:t}`bayes` and {cite:t}`bayes2`.

## Bayes' Rule
- **What**: Posterior $\propto$ Prior $\times$ Likelihood.
    - **Prior**: The **believed** probability distribution for the unknown, before seeing any data.
    - **Likelihood**: How strongly each possible unknown value would make the observed data **unsurprising**.
    - **Posterior**: The **updated** probability distribution for the unknown after seeing the data.
- **Why**:
    - Math: Joint probability can be factored in 2 directions of conditional probabilities.
    - Intuition: The unique way to update beliefs coherently.
- **How**:
$$\begin{align*}
&\text{Discrete}:    &&P(A|B)=\frac{P(B|A)P(A)}{P(B)} \\
&\text{Parametric}: &&\pi(\theta|x)=\frac{p(x|\theta)\pi(\theta)}{\int p(x|\vartheta)\pi(\vartheta)d\vartheta}
\end{align*}$$

```{attention} Q&A
:class: dropdown
*Why $\pi$ instead of $p$?*
- $p(\cdot)$: Density on the **data** space.
- $\pi(\cdot)$: Density on the **parameter** space.
```

&nbsp;

### Likelihood Principle
- **What**: Posterior depends on the data ONLY via likelihood.
	- If 2 datasets induce proportional likelihoods, thjey induce the SAME posterior up to the SAME normalizing constant.
- **Why**: Math.
- **How**:
$$
\frac{\pi(\theta_1|x)}{\pi(\theta_2|x)}=\frac{p(x|\theta_1)}{p(x|\theta_2)}\frac{\pi(\theta_1)}{\pi(\theta_2)}
$$
	- Renormalization does NOT depend on $\theta$.
	- Posterior odds = Prior odds $\times$ Likelihood ratio.

&nbsp;

### Sufficient Statistics
- **What**: A statistic $T(X)$ is sufficient for $\theta$ if, once we know $T(X)$, the rest of the data contains NO additional info about $\theta$.
	- Definitions:
		- Conditional distribution: Given $T=t$, the shape of the remaining randomness in $X$ is param-free.
		$$
		P(X|T(X)=t)\text{ is the same for all }\theta
		$$
		- Conditional independence: Posterior depends on $X$ only via $T(X)$
		$$
		\pi(\theta|X)=\pi(\theta|T(X))
		$$
		- Likelihood preservation: 2 datasets with the same $T$ must have proportional likelihoods in $\theta$, just not dependent on $\theta$.
		$$
		T(x)=T(y)\Rightarrow\frac{p(x|\theta)}{p(y|\theta)}\text{ is the same for all }\theta
		$$
- **Why**:
	- *Why does sufficiency matter?*
		- Bayes automatically compresses data down to anything that ONLY preserves the likelihood.
- **How**:
	- Fisher-Neyma Factorization Theorem:
	$$
	T(X)\text{ is sufficient for }\theta\Leftrightarrow p_\theta(x)=h(x)g_\theta(T(x))
	$$
		- $h(x)\ge0$ does NOT depend on $\theta$.
		- $g_\theta(\cdot)\ge0$ depends on $x$ ONLY via $T(x)$.
	- In practice:
		1. Write down the jointly likelihood $p_\theta(x)$.
		2. Separate $\theta$-free factors from $\theta$-dependent part.
		3. Identify the smallest function of the data that captures $\theta$-dependence.
		4. That's a sufficient statistic.

&nbsp;

### Identifiability
- **What**: Whether the mapping $\theta \mapsto p(\cdot | \theta)$ is injective.
	- **Injective**: $f(\theta_1)=f(\theta_2)\Rightarrow\theta_1=\theta_2$
	- **Identifiable**: If 2 params generate the SAME distribution over data, then they MUST be the same.
- **Why**: In Bayesian,
	- If the likelihood mapping is NOT injective, then diff params can produce the same likelihood func for all data.
	- → NO data can distinguish them.
	- Bayesian can ONLY learn identifiable models.
- **How**:
$$
p(\cdot|\theta_1)=p(\cdot|\theta_2)\Rightarrow\theta_1=\theta_2
$$