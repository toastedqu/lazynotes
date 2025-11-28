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
# Information Theory
## Information
- **What**: Surprise.
    - Less/More likely events $\rightarrow$ More/Less info
- **Why**: To mathematically analyze, optimize, & understand the fundamental limits of systems that store, process, & transmit data.
    - Data compression
    - Communication
    - Systems
    - Cryptography
    - ML
    - ...
- **How**: To quantify surprise (i.e., self-information), we want
    1. Higher surprise for less probable events.
    2. Zero surprise for deterministic events.
    3. If 2 independent events happen, their surprises should add up.
    - 2 & 3 $\rightarrow$ $\log P$
    - 1 $\rightarrow$ $-\log P$

```{dropdown} ELI5
*What is information?*
- You live in the middle of Egypt.
- Weather forecast: "It will be sunny tmrw."
- You: "lol as expected." (You didn't get much info)
- Weather forecast: "It will snow tmrw."
- You: "Hol up WTF???" (You got huge info)
```

```{note} Math
:class: dropdown
**Information**:

$$
I(x)=-\log P(x)
$$
- $x$: Event outcome.
- $P$: Probability.
- $I$: Information.
- Units of $I(x)$:
	- **Bits**: $\log_2$, mainly used in EECS.
	- **Nats**: $\ln$, mainly used in Math/Stats for convenience in calculus.
	- $1\text{ nat}=\frac{1}{\ln 2}\text{ bits}$
```

## Entropy
- **What**: Average surprise.
	- More/Less randomness $\rightarrow$ High/Low entropy
- **Why**: To quantitatively measure the inherent randomness in a system/process.
- **How**: Take the expectation of info across the probability distribution. 

```{dropdown} ELI5
*What is entropy?*
- So somehow it wasn't sunny in Egypt last time.
- But then it's been sunny all the time.
- How much surprise do we expect on average?
- Not much. We need a measure of the predictability.
```

```{note} Math
:class: dropdown
**Entropy**:

$$
H(X)=E_{x\sim P}[I(x)]=-\sum_xP(x)\log P(x)=-\int_{-\infty}^\infty p(x)\log p(x)dx
$$
- $X$: Random variable.
- $p(x)$: PDF.
- Properties:
	- $H(X)\geq 0$
	- $H(X)\leq \log N$, where $N$ is # of possible outcomes.
		- $H(X)=\log N\text{ iff }\forall x\sim P: P(x)=\frac{1}{N}$
	- $H(X)$ is concave.

**Joint Entropy**:

$$
H(X,Y)=-\sum_x\sum_yP(x,y)\log P(x,y)
$$

**Conditional Entropy**:

$$\begin{align*}
H(Y|X)&=\sum_xP(x)H(Y|X=x) \\
H(Y|X=x)&=-\sum_yP(y|x)\log P(y|x)
\end{align*}$$

**Chain Rule**:

$$
H(X,Y)=H(X)+H(Y|X)
$$

**Mutual Info**:

$$
I(X,Y)=H(X)-H(X|Y)=H(Y)-H(Y|X)=H(X)+H(Y)-H(X,Y)
$$
```

## KL Divergence
- **What**

```{note} Math
:class: dropdown
KL Divergence

$$
D_\text{KL}(P||Q)=\sum P(x)\log\frac{P(x)}{Q(x)}
$$
```