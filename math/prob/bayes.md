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