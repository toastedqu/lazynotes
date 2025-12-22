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
Consciousness depends on how internal states are organized, differentiated, integrated, and self-related. 

Information theory provides the only rigorous, substrate-independent way to quantify things like how many meaningful states a system can occupy, how unified those states are, how much the system models itself, and how its internal causal structure works, all of which are plausibly necessary for subjective experience.

Information theory doesn’t guarantee consciousness by itself, but without it, there’s no clear way to even define, measure, or compare the kinds of internal organization consciousness seems to require.

Here are my study notes from {cite:t}`info`.

&nbsp;

## Basics
### Information
- **What**: Surprise.
    - Less likely events → More info.
	- e.g.,
		- You are in the Sahara Desert. 
		- Weather forecast: "It will be sunny tmrw."
		- You: "As expected." (You didn't get much info)
		- Weather forecast: "It will snow tmrw."
		- You: "WTF?!" (You got huge info)
- **How**: To quantify surprise, we want
    1. Higher surprise for less probable events.
    2. Zero surprise for deterministic events.
    3. If two independent events happen, their surprises should add up.

	Point 2 & 3 lead to "$\log P$" as the best basis. 
	
	Point 1 refines it to "$-\log P$".

```{note} Math
:class: dropdown
Notations:
- $x$: Event outcome.
- $P$: Probability.

Information:

$$
I(x):=-\log P(x)
$$
- Units of $I(x)$:
	- **Bits**: $\log_2$, mainly used in EECS.
	- **Nats**: $\ln$, mainly used in Math/Stats for convenience in calculus.
	- $1\text{ nat}=\frac{1}{\ln 2}\text{ bits}$
```

&nbsp;

### Entropy
- **What**: Average surprise.
	- More uncertainty → High entropy.
	- e.g.,
		- You have a dice of 6 numbers. Each number is equally likely. You have no idea which number it will be. You will be more surprised on average.
		- You have a dice of five 1s and one 6. You always have a strong guess on the number being 1. You will be less surprised on average.
- **Why**: To quantitatively measure the **inherent uncertainty** in a system/process.
- **How**: The expectation of information across the whole probability distribution.

```{note} Math
:class: dropdown
Notations:
- $X$: Random variable.
- $p(\cdot)$: PDF.
- $N$: # possible outcomes.

Entropy:

$$\begin{align*}
H(X)&:=E_{x\sim P}[I(x)] \\
	&=-\sum_xP(x)\log P(x) \\
	&=-\int_{-\infty}^\infty p(x)\log p(x)dx
\end{align*}$$
- $H(X)\in [0,\log N]$
- $H(X)=\log N\ \ \ \text{iff}\ \ \ \forall x\sim P: P(x)=\frac{1}{N}$
- $H(X)$ is concave.
```

```{tip} Derivation
:class: dropdown
*Why is entropy concave?*

1. Concavity means:

$$
\forall x:\ \ f''(x)\leq 0
$$

2. Let $H(p)=-\sum_{i=1}^{m}p_i\log p_i$, where $p_i\in[0,1]$. 

	Partial derivative:

	$$
	\frac{\partial H}{\partial p_i}=-(\log p_i+1)
	$$

	Second derivative:

	$$
	\frac{\partial^2 H}{\partial p_i \partial p_j}&=\begin{cases}
	-\frac{1}{p_i} & i=j \\
	0 & i\neq j
	\end{cases}\leq 0
	$$
```

&nbsp;

#### Joint Entropy
- **What**: Average surprise (i.e., inherent uncertainty) in the outcome of both random vars at once.
	- If $(X,Y)$ can be lots of different combinations & they are all fairly likely, high joint entropy.
	- If $(X,Y)$ are only likely to be a few pairs, low joint entropy.
- **Why**: A system can have multiple random parts → A measure of how uncertain the full system is.
- **How**: The expectation of information across the joint probability distribution.

```{note} Math
:class: dropdown
Joint Entropy:

$$
H(X,Y)=-\sum_x\sum_yP(x,y)\log P(x,y)
$$
- If $X\ \& \ Y$ are independent: $H(X,Y)=H(X)+H(Y)$.
- If $Y$ is FULLY determined by $X$: $H(X,Y)=H(X)=H(Y)$.
```

&nbsp;

#### Conditional Entropy
- **What**: Average surprise (i.e., inherent uncertainty) in the outcome of both random vars at once.
	- If $(X,Y)$ can be lots of different combinations & they are all fairly likely, high joint entropy.
	- If $(X,Y)$ are only likely to be a few pairs, low joint entropy.
- **Why**: A system can have multiple random parts → A measure of how uncertain the full system is.
- **How**: The expectation of information across the joint probability distribution.

```{note} Math
:class: dropdown
Conditional Entropy:

$$\begin{align*}
H(Y|X)&=\sum_xP(x)H(Y|X=x) \\
H(Y|X=x)&=-\sum_yP(y|x)\log P(y|x)
\end{align*}$$

Chain Rule:

$$
H(X,Y)=H(X)+H(Y|X)
$$
```

&nbsp;

### KL Divergence
- **What**: How wasteful it is to use distribution $Q$ to approximate the true distribution $P$ of info.
	- If $Q$ matches $P$, we are using the optimal #bits.
	- If $Q$ does NOT match $P$, we are spending extra bits per symbol on average.
	- KLD = Expected # of extra bits per sample to approximate $P$ with $Q$.
- **Why**: Distribution approximation is EXTREMELY useful in various cases (e.g., Loss functions in ML, measuring info gain, etc.)
- **How**: Cross Entropy minus True Entropy.

```{note} Math
:class: dropdown
Notations:
- $P$: True distribution.
- $Q$: Approximated distribution.

KLD:

$$\begin{align*}
D_{KL}(P||Q)&:=\sum_xP(x)\log\frac{P(x)}{Q(x)} \\
			&:=\int p(x)\log\frac{p(x)}{q(x)}dx
\end{align*}$$
- $D_{KL}(P||Q)\geq 0$
- $D_{KL}(P||Q)=0\ \ \ \text{iff}\ \ \ P=Q$
- $D_{KL}(P||Q)\neq D_{KL}(Q||P)$
- $D_{KL}(P||Q)$ is convex.

KLD (Entropy ver.):

$$
D_{KL}(P||Q)=H(P,Q)-H(P)
$$
- $H(P)$ is model-independent → Minimize CE = Minimize KLD.
- $H(P,Q)$ is convex for fixed $P$.
```

```{tip} Derivation
:class: dropdown
*Why is KLD convex?*

1. Jensen's Inequality: Let $f$ be convex and $X$ be a random variable,

$$
f(E[X])\leq E[f(X)]
$$

2. Apply it to KLD:

$$\begin{align*}
D_{KL}(P||Q)&=-E\left[\log\frac{Q(X)}{P(X)}\right] \\
			&\geq -\log E\left[\frac{Q(X)}{P(X)}\right] \\
			&=-\log\left(\sum_xP(x)\frac{Q(x)}{P(x)}\right) \\
			&=-\log 1 \\
			&=0
\end{align*}$$
```

&nbsp;

### Mutual Information

```{note} Math
:class: dropdown
Mutual Info:

$$
I(X,Y):=\sum_{x,y}P(x,y)\log\frac{P(x,y)}{P(x)P(y)}
$$

Mutual Info & Entropy:

$$\begin{align*}
I(X,Y)&=H(X)-H(X|Y) \\
	  &=H(Y)-H(Y|X) \\
	  &=H(X)+H(Y)-H(X,Y)
\end{align*}$$
```