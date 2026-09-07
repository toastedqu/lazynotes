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
Study notes from {cite:t}`info`.
Additional coding-theorem notes from {cite:t}`info_coding`.

Notations:
- $X$: Random variable; $x$ denotes a realization.
- $Y$: Second random variable.
- $Z$: Conditioning variable or channel output.
- $T$: Representation of an observed variable.
- $P$: Source or reference probability distribution; $P(x)$ abbreviates $P(X=x)$.
- $Q$: Alternative or model probability distribution.
- $p$: Probability density when a variable is continuous.
- $q$: Alternative probability density.
- $\mathcal{X}$: Alphabet of $X$; analogous calligraphic letters denote other alphabets.
- $H$: Discrete Shannon entropy.
- $h$: Differential entropy.
- $I(X;Y)$: Mutual information; $I(X,Y)$ in the original formulas means the same quantity.
- $D_{KL}$: Kullback–Leibler divergence.
- $L$: Block length.
- $X_{1:L}$: Sequence $(X_1,\ldots,X_L)$.
- $\hat{X}$: Reconstructed variable in coding sections; here the hat denotes reconstruction rather than parameter estimation.
- $d(x,\hat{x})$: Nonnegative distortion assigned to reconstructing $x$ as $\hat{x}$.
- $D$: Allowed expected distortion per source symbol.
- $R$: Coding rate in bits per source symbol unless explicitly stated otherwise.
- $H_2(u)=-u\log_2u-(1-u)\log_2(1-u)$: Binary entropy, with endpoint values defined by continuity.

- **Conventions**: $\log=\log_2$ unless $\ln$ is specified; information quantities are in bits. Discrete alphabets are finite unless stated otherwise; entropy-difference identities require finite terms.
- **Zero probabilities**: $0\log0=0$; a positive-probability outcome assigned probability $0$ by a model contributes $+\infty$ to its expected log loss.
- **Reading order**: Uncertainty → dependence → distribution mismatch → coding limits → prediction limits.

&nbsp;

## Entropy and Information Measures

### Information (Surprisal)
- **What**: Surprise.
    - Less likely events → More info.
    - e.g.,
        - You are in the Sahara Desert.
        - Weather forecast: "It will be sunny tmrw."
        - You: "As expected." (You didn't get much info)
        - Weather forecast: "It will snow tmrw."
        - You: "WTF?!" (You got huge info)
- **Why**: A common numerical scale for the information carried by individual outcomes.
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

```{tip} Derivation
:class: dropdown
*Why a logarithm rather than another decreasing function?*

1. Write surprise as a function $s(u)$ of an event probability $u\in(0,1]$.
2. Independent events impose $s(uv)=s(u)+s(v)$; monotonicity rules out pathological additive solutions after taking logarithms.
3. Therefore $s(u)=-c\ln u$ for a constant $c>0$. Choosing $s(1/2)=1$ gives $c=1/\ln2$.
4. **Clarification**: Zero surprise & additivity alone are not the full characterization; monotonicity, continuity, or another suitable regularity condition is needed.
```

```{attention} Q&A
:class: dropdown
*Does surprise measure meaning or usefulness?*

- No. An irrelevant random bit can be surprising; a useful fact can already be expected.
- The reference distribution matters: the same outcome can surprise one model but not another.

*What happens at probabilities $0$ & $1$?*

- $P(x)=1$ → $I(x)=0$.
- $P(x)\downarrow0$ → $I(x)\uparrow\infty$. An exactly impossible outcome is never sampled under $P$.
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

- **Discrete/continuous clarification**: The sum defines discrete $H(X)$; the integral defines a different quantity, differential $h(X)$. They are not equal expressions for one entropy. The bounds immediately below apply only to discrete $X$ with $N$ possible outcomes.
- $H(X)\in [0,\log N]$
- $H(X)=\log N\ \ \ \text{iff}\ \ \ \forall x\sim P: P(x)=\frac{1}{N}$
- $H(X)$ is concave.
- $H(X)=0$ iff $X$ is constant almost surely.
- **Effective alphabet size**: $2^{H(X)}$; equals the actual alphabet size for a uniform distribution.
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
    \begin{aligned}
    \frac{\partial^2 H}{\partial p_i \partial p_j}&=\begin{cases}
    -\frac{1}{p_i} & i=j \\
    0 & i\neq j
    \end{cases}\leq 0
    \end{aligned}
    $$

3. **Clarification**: Step 1 is the twice-differentiable scalar criterion. A multivariate function requires a negative-semidefinite Hessian, not merely nonpositive entries. In step 2, $p$ is the probability vector, $m$ counts outcomes locally, $\sum_i p_i=1$, and the displayed derivatives use natural logarithms at $p_i>0$.
4. For base-2 entropy, the corresponding derivatives & directional curvature are:

    $$
    \begin{align*}
    \frac{\partial H}{\partial p_i}&=-\frac{\ln p_i+1}{\ln2},\\
    \mathbf{v}^{T}\nabla^2H(p)\mathbf{v}
    &=-\frac{1}{\ln2}\sum_i\frac{v_i^2}{p_i}\leq0.
    \end{align*}
    $$

    - $\mathbf{v}$: Any perturbation direction; $\sum_i v_i=0$ for directions tangent to the probability simplex.

5. The continuous extension $-u\log u\to0$ as $u\downarrow0$ includes boundary distributions.
```

```{attention} Q&A
:class: dropdown
*What does concavity mean operationally?*

- Mixing source distributions without revealing the mixture label cannot reduce entropy below their weighted average.
- Revealing that label removes the uncertainty contributed by not knowing which source was selected.

*Why does the uniform distribution maximize entropy?*

- For uniform $U$ on $N$ outcomes, $D_{KL}(P\|U)=\log N-H(P)\geq0$.
- Equality iff $P=U$.

*Is high entropy evidence of intelligence or structure?*

- No. Independent fair bits have maximal entropy but no predictive dependence.
- Structure requires dependence measures, not uncertainty alone.
```

&nbsp;

#### Differential Entropy
- **What**: Expected negative log density relative to a continuous reference measure.
- **Why**: Continuous distributions need a density-based counterpart to discrete entropy.
- **How**: Average density-level surprise; retain the coordinate system & measurement resolution.

```{note} Math
:class: dropdown
For an absolutely continuous scalar variable, when the integral is defined:

$$
h(X)=-\int p(x)\log_2p(x)\,dx.
$$

- Density is not point probability: $p(x)$ can exceed $1$, while $P(X=x)=0$.
- $h(X)$ can be negative, infinite, or undefined; there is no universal discrete-style lower bound.

For a differentiable bijection $g$ with nonzero derivative, assuming the expectations exist:

$$
h(g(X))=h(X)+\mathbb{E}\log_2|g'(X)|.
$$

- $g$: Invertible coordinate transformation.
- In particular, $h(aX+b)=h(X)+\log_2|a|$ for $a\neq0$.
- For vectors, replace $|g'(X)|$ by the absolute Jacobian determinant.

For a scalar density continuous on a compact interval, quantized into bins of width $\Delta$:

$$
H(X_\Delta)=h(X)+\log_2(1/\Delta)+o(1),\qquad\Delta\downarrow0.
$$

- $X_\Delta$: Discrete bin index.
- $\Delta$: Quantization resolution in the chosen coordinates.
- $o(1)$: Term tending to $0$ as resolution becomes arbitrarily fine.
```

```{tip} Derivation
:class: dropdown
*Where does the resolution term come from?*

1. A small bin near $x$ has probability approximately $p(x)\Delta$.
2. Its discrete surprise is approximately $-\log_2p(x)+\log_2(1/\Delta)$.
3. Averaging the first term gives $h(X)$; averaging the constant second term leaves $\log_2(1/\Delta)$.
```

```{note} Example
:class: dropdown
- $X\sim\operatorname{Uniform}(0,1/4)$ → $p(x)=4$ on its support → $h(X)=-2$ bits.
- Multiplying the coordinate by $4$ gives a uniform variable on $(0,1)$ with differential entropy $0$.
- No information was destroyed: the transformation is invertible.
```

```{attention} Q&A
:class: dropdown
*Which discrete identities survive?*

- Chain rules & $I(X;Y)=h(X)-h(X|Y)$ survive when the required densities exist & the terms are finite.
- Nonnegativity of entropy does not. Deterministic continuous relationships can be singular, so the entropy-difference expression may be undefined.
- Define continuous mutual information directly through KL; it remains nonnegative & invariant under invertible changes of coordinates.

*What maximizes differential entropy?*

- Fixed support interval of length $a$: uniform density, $h(X)\leq\log_2a$.
- Fixed finite variance $\sigma^2>0$: Gaussian density,

    $$
    h(X)\leq\frac12\log_2(2\pi e\sigma^2).
    $$

- $\sigma^2$: Variance constraint; equality in the second bound iff the density is Gaussian.
- Without a scale or support constraint, there is no finite maximum.
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
- **Deterministic-dependence clarification**: $Y=f(X)$ guarantees $H(X,Y)=H(X)$ & $H(Y)\leq H(X)$, not generally $H(X)=H(Y)$. The last equality requires that $X$ also be recoverable from $Y$ almost surely.
- $\max\{H(X),H(Y)\}\leq H(X,Y)\leq H(X)+H(Y)$.
```

```{note} Example
:class: dropdown
- $X$ uniform on $\{0,1,2,3\}$; $Y=X\bmod2$.
- $Y$ is fully determined by $X$, but $H(X)=2$, $H(Y)=1$, $H(X,Y)=2$ bits.
- Knowing parity discards one bit; knowing the full value determines both variables.
```

```{attention} Q&A
:class: dropdown
*When does observing a pair add no uncertainty beyond observing either variable?*

- $H(X,Y)=H(X)$ iff $H(Y|X)=0$.
- Equality to both marginal entropies requires mutual deterministic recoverability.
```

&nbsp;

#### Conditional Entropy
- **What**: How much uncertainty remains about a random var after learning about another random var.
    - If $X$ tells us EVERYTHING about $Y$, $H(Y|X)=0$.
    - If $X$ tells us NOTHING about $Y$, $H(Y|X)=H(Y)$.
- **Why**: The measure of the "remaining uncertainty" → Super important in various topics: Compression, Learning, Causality, etc.
- **How**: The expectation of information across the conditional probability distribution.

```{note} Math
:class: dropdown
Conditional Entropy:

$$\begin{align*}
H(Y|X)&=\sum_xP(x)H(Y|X=x)=-\sum_{x,y}P(x,y)\log P(y|x) \\
H(Y|X=x)&=-\sum_yP(y|x)\log P(y|x)
\end{align*}$$

- Conditioning ALWAYS reduces entropy: $H(Y|X)\leq H(Y)$.
- **Averaging clarification**: This compares expected conditional entropy with marginal entropy. It does not assert $H(Y|X=x)\leq H(Y)$ for every observed $x$.
- Independence: $H(Y|X)=H(Y)$.
- Deterministic dependence: $Y=f(X)\rightarrow H(Y|X)=0$.
- $0\leq H(Y|X)\leq H(Y)$; equality at the upper bound iff $X$ & $Y$ are independent.

Chain Rule:

$$
H(X,Y)=H(X)+H(Y|X)=H(Y)+H(X|Y)
$$
- Joint uncertainty of $X,Y$ = Uncertainty of $X$ + Uncertainty of $Y$ once $X$ is known.
```

```{note} Example
:class: dropdown
- $P(X=1)=1/4$; if $X=1$, $Y$ is a fair bit; if $X=0$, $Y=0$.
- $P(Y=1)=1/8$ → $H(Y)=H_2(1/8)<1$.
- Observing $X=1$ raises uncertainty to $H(Y|X=1)=1$ bit.
- Averaging both cases gives $H(Y|X)=1/4$ bit, still below $H(Y)$.
```

```{attention} Q&A
:class: dropdown
*Why can conditioning reduce uncertainty only on average?*

- Some observations reveal an unusually ambiguous subpopulation; others reveal an unusually predictable one.
- Averaging conditional distributions reconstructs the marginal distribution; entropy concavity gives the inequality.
```

&nbsp;

### Mutual Information
- **What**: Expected reduction in uncertainty about one variable from observing another.
- **Why**: Marginal entropies do not distinguish shared information from independent randomness.
- **How**: Compare the joint distribution with the distribution that would hold under independence.

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

$$
\begin{align*}
I(X;Y)
&=D_{KL}(P_{XY}\|P_XP_Y)\\
&=\mathbb{E}_{X}D_{KL}(P_{Y|X}\|P_Y).
\end{align*}
$$

- $P_{XY}$: Joint law.
- $P_XP_Y$: Product of the marginal laws, not the actual joint law unless independent.
- $P_{Y|X}$: Conditional law of $Y$ at the sampled value of $X$.
- $0\leq I(X;Y)\leq\min\{H(X),H(Y)\}$ for discrete variables.
- $I(X;Y)=I(Y;X)$.
- $I(X;Y)=0$ iff $X$ & $Y$ are independent.
- **Pointwise mutual information**: $\log_2[P(x,y)/(P(x)P(y))]$; individual values may be negative even though their expectation is nonnegative.
```

```{tip} Derivation
:class: dropdown
*Why does dependence equal an entropy reduction?*

1. Substitute $P(x,y)=P(x)P(y|x)$ into the log ratio.
2. Split $\log[P(y|x)/P(y)]$ into $\log P(y|x)-\log P(y)$.
3. Average under the joint law: the two terms become $-H(Y|X)$ & $H(Y)$.
4. Nonnegativity follows from KL nonnegativity; equality means joint law = product law.
```

```{note} Example
:class: dropdown
- Let $X$ be a fair bit & independently flip it with probability $\epsilon$ to obtain $Y$.
- $H(X)=H(Y)=1$, $H(Y|X)=H_2(\epsilon)$, so $I(X;Y)=1-H_2(\epsilon)$.
- $\epsilon=0$ or $1$ → perfect recoverability → $1$ bit shared.
- $\epsilon=1/2$ → independent output → $0$ bits shared.
```

```{attention} Q&A
:class: dropdown
*Does zero correlation imply zero mutual information?*

- No. Correlation captures a particular second-order relationship; independence removes every statistical relationship.
- For $X$ uniform on $\{-1,0,1\}$ & $Y=X^2$, covariance is zero but $I(X;Y)=H(Y)>0$.

*Is information gain the same as entropy reduction for one observation?*

- Posterior-to-prior KL, $D_{KL}(P_{Y|X=x}\|P_Y)$, is nonnegative for each observation.
- $H(Y)-H(Y|X=x)$ can be negative. Their averages over $X$ both equal $I(X;Y)$.

*Does mutual information imply causation or accessible knowledge?*

- Neither. Dependence may come from a common cause; exploiting it may require a computationally difficult decoder.
- Mutual information measures what is statistically available, not what a particular learner can extract.

*Can continuous mutual information be infinite?*

- Yes. For a non-atomic continuous $X$, exact observation gives $I(X;X)=+\infty$.
- Finite measurement resolution or observation noise changes the joint law & can make the quantity finite.
```

&nbsp;

#### Conditional Mutual Information
- **What**: Shared information remaining after a third variable is observed.
- **Why**: Separate additional information from information already supplied by context.
- **How**: Measure dependence within each conditional distribution, then average over contexts.

```{note} Math
:class: dropdown
$$
\begin{align*}
I(X;Y|Z)
&=\sum_zP(z)D_{KL}(P_{XY|z}\|P_{X|z}P_{Y|z})\\
&=\sum_{x,y,z}P(x,y,z)\log_2\frac{P(x,y|z)}{P(x|z)P(y|z)}\\
&=H(X|Z)-H(X|Y,Z)\\
&=H(X|Z)+H(Y|Z)-H(X,Y|Z).
\end{align*}
$$

- Conditional laws need only be specified for $z$ with $P(z)>0$.
- $I(X;Y|Z)\geq0$, with equality iff $X\perp Y\mid Z$ almost surely.
- $X\perp Y\mid Z$: Conditional independence.
- No general ordering between $I(X;Y)$ & $I(X;Y|Z)$.
```

```{note} Example
:class: dropdown
- **Conditioning creates dependence**: Independent fair bits $X,Y$; $Z=X\oplus Y$.
- $\oplus$: Exclusive OR.
- $I(X;Y)=0$, but observing parity makes either bit determine the other: $I(X;Y|Z)=1$.
- **Conditioning removes dependence**: $X=Y=Z$ is a fair bit → $I(X;Y)=1$, $I(X;Y|Z)=0$.
```

```{attention} Q&A
:class: dropdown
*Does a nonnegative conditional quantity mean conditioning always increases mutual information?*

- No. Nonnegativity compares $I(X;Y|Z)$ with zero, not with $I(X;Y)$.
- Conditioning can expose a hidden relation or explain away a shared one.
```

&nbsp;

### Chain Rules
- **What**: Decompositions of joint uncertainty & information into sequential contributions.
- **Why**: Reason about sequences & incremental evidence without assuming independence.
- **How**: Factor joint probabilities into conditionals; logarithms turn products into sums.

```{note} Math
:class: dropdown
$$
\begin{align*}
H(X_{1:L})&=\sum_{t=1}^{L}H(X_t|X_{1:t-1}),\\
H(X_{1:L}|Z)&=\sum_{t=1}^{L}H(X_t|X_{1:t-1},Z),\\
I(X;Y,Z)&=I(X;Y)+I(X;Z|Y),\\
I(X_{1:L};Y|Z)&=\sum_{t=1}^{L}I(X_t;Y|X_{1:t-1},Z).
\end{align*}
$$

- $t$: Position in a sequence.
- $X_{1:0}$: Empty history; conditioning on it has no effect.
- Reordering variables changes individual terms but not the joint total.

$$
H(X_{1:L})\leq\sum_{t=1}^{L}H(X_t).
$$

- Equality iff all variables are mutually independent.
```

```{tip} Derivation
:class: dropdown
*Why does the probability chain rule become an entropy chain rule?*

1. Factor $P(x_{1:L})=\prod_{t=1}^{L}P(x_t|x_{1:t-1})$ on positive-probability sequences.
2. Apply $-\log_2$ to obtain a sum of conditional surprises.
3. Take expectation under the true joint law; each term is a conditional entropy.
4. Subtract conditional from unconditional versions to obtain the mutual-information rules.
```

```{attention} Q&A
:class: dropdown
*Does an autoregressive factorization impose an independence assumption?*

- No. Keeping the full history is an exact factorization of any joint distribution.
- Truncating history or restricting the conditional model introduces assumptions.

*Can evidence add less than its marginal information suggests?*

- Yes. Its incremental contribution is conditional mutual information, not its marginal mutual information.
- Duplicated evidence contributes zero after its copy is known.
```

&nbsp;

### Data Processing Inequality
- **What**: Information cannot increase under processing without new side information.
- **Why**: Bound what any representation or downstream decoder can retain from its input.
- **How**: A processor sees only its input; discarded distinctions cannot be recovered from its output alone.

```{note} Math
:class: dropdown
For a Markov chain $X\to Y\to Z$:

$$
P(x,y,z)=P(x)P(y|x)P(z|y),\qquad X\perp Z\mid Y.
$$

$$
\begin{align*}
I(X;Z)&\leq I(X;Y),\\
I(X;Z)&\leq I(Y;Z),\\
I(X;Y)-I(X;Z)&=I(X;Y|Z).
\end{align*}
$$

- Equality in the first bound iff $I(X;Y|Z)=0$.
- The loss identity assumes finite mutual informations; the inequality also applies to extended values.
- $X\to Y\to Z$ denotes a factorization, not by itself a causal assertion.
```

```{tip} Derivation
:class: dropdown
*Why can processing not create mutual information?*

1. Expand the same quantity in two orders:

    $$
    I(X;Y,Z)=I(X;Y)+I(X;Z|Y)
             =I(X;Z)+I(X;Y|Z).
    $$

2. The Markov condition sets $I(X;Z|Y)=0$.
3. Rearranging leaves the nonnegative loss $I(X;Y|Z)$.
4. The conditional independence also holds in the reverse chain $Z\to Y\to X$, giving the second bound.
```

```{attention} Q&A
:class: dropdown
*When is a representation sufficient for a target?*

- For $T=f(X)$, $I(T;Y)\leq I(X;Y)$.
- Equality iff $Y\perp X\mid T$, equivalently $P(Y|X)=P(Y|T)$ almost surely.
- This is target-specific sufficiency: $T$ may discard information about $X$ while preserving everything in $X$ relevant to $Y$.

*Why can processing improve prediction if it cannot add information?*

- A representation can make existing information accessible to a restricted decoder.
- Extra training data, memory, or observations must be included as inputs before applying the inequality; an omitted input can invalidate the claimed Markov chain.

*Does injecting independent randomness help retain information?*

- It can aid a constrained procedure, but cannot increase mutual information about the upstream variable beyond that in the processor's input.
```

&nbsp;

### Total Correlation
- **What**: Joint dependence among multiple variables.
- **Why**: Pairwise mutual information can miss relationships visible only jointly.
- **How**: Compare the joint distribution with the product of all its marginals.

```{note} Math
:class: dropdown
$$
\begin{align*}
\operatorname{TC}(X_{1:L})
&=D_{KL}\left(P_{X_{1:L}}\middle\|\prod_{t=1}^{L}P_{X_t}\right)\\
&=\sum_{t=1}^{L}H(X_t)-H(X_{1:L})\\
&=\sum_{t=2}^{L}I(X_t;X_{1:t-1})\geq0.
\end{align*}
$$

- $\operatorname{TC}$: Total correlation, also called multi-information.
- Zero iff the variables are mutually independent.
- For two variables, equals mutual information.
```

```{note} Example
:class: dropdown
- Independent fair bits $X,Y$; $Z=X\oplus Y$.
- Every pair is independent, but $H(X,Y,Z)=2$ while the sum of marginal entropies is $3$.
- $\operatorname{TC}(X,Y,Z)=1$ bit: one joint constraint invisible to pairwise tests.
```

```{attention} Q&A
:class: dropdown
*Does total correlation isolate redundant versus synergistic information?*

- No. It measures departure from mutual independence, not a unique decomposition of the roles of individual variables.
- The three-variable quantity $I(X;Y)-I(X;Y|Z)$ is different & may be negative; it is not a nonnegative total-dependence measure.
```

&nbsp;

### Entropy Rate
- **What**: Asymptotic new uncertainty per symbol in a stochastic process.
- **Why**: Temporal dependence makes single-symbol entropy overstate the information in a long sequence.
- **How**: Condition each symbol on its history, then take the long-run limit.

```{note} Math
:class: dropdown
For a stationary finite-alphabet process:

$$
\begin{align*}
\overline{H}(X)
&=\lim_{L\to\infty}\frac{H(X_{1:L})}{L}\\
&=\lim_{L\to\infty}H(X_L|X_{1:L-1})\\
&=\inf_{L\geq1}H(X_L|X_{1:L-1}).
\end{align*}
$$

- $\overline{H}(X)$: Entropy rate of the process.
- **Stationary**: Finite-block distributions are invariant to shifts in time.
- $0\leq\overline{H}(X)\leq H(X_1)$.
- Independent identically distributed source: $\overline{H}(X)=H(X_1)$.
- Stationary first-order Markov source: $\overline{H}(X)=H(X_2|X_1)$.
```

```{tip} Derivation
:class: dropdown
*Why do the block-average & conditional limits agree?*

1. Stationarity & conditioning imply the sequence $H(X_L|X_{1:L-1})$ is nonincreasing.
2. It is bounded below by $0$, so it converges.
3. The entropy chain rule writes $H(X_{1:L})/L$ as the arithmetic mean of its first $L$ terms.
4. Arithmetic means of a convergent sequence have the same limit.
```

```{note} Example
:class: dropdown
- A stationary fair binary Markov chain flips its current bit with probability $\epsilon$.
- Each symbol has entropy $1$ bit, but

    $$
    H(X_{1:L})=1+(L-1)H_2(\epsilon),\qquad
    \overline{H}(X)=H_2(\epsilon).
    $$

- For $\epsilon=0$, the first bit determines the whole sequence; block entropy stays $1$, while entropy per symbol tends to $0$.
```

```{attention} Q&A
:class: dropdown
*Is stationarity enough for a typical-set theorem?*

- It gives the entropy-rate identities above, but not convergence of every sample path's normalized surprise to a single deterministic rate.
- The usual stationary extension of typicality also assumes ergodicity.

*Does low entropy rate imply a simple predictive state?*

- No. It bounds asymptotic unpredictability, not the memory or computation required for optimal prediction.
```

&nbsp;

## Divergences Between Distributions

### KL Divergence
- **Name**: Kullback–Leibler divergence, also called relative entropy.
- **What**: Extra cost to use distribution $Q$ to approximate true distribution $P$.
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

- **Domain clarification**: The sum is for probability masses; the integral is for densities relative to the same measure. These are alternatives for different types of distributions.
- $D_{KL}(P||Q)\geq 0$
- $D_{KL}(P||Q)=0\ \ \ \text{iff}\ \ \ P=Q$
- $D_{KL}(P||Q)\neq D_{KL}(Q||P)$
- **Asymmetry clarification**: The two directions need not be equal, rather than being unequal for every pair. Equality can occur, including when $P=Q$.
- $D_{KL}(P||Q)$ is convex.
- **Convexity clarification**: Jointly convex in the pair of distributions $(P,Q)$, hence convex in either one with the other fixed. This does not imply convexity in parameters of a nonlinear model.
- $P\not\ll Q$ → $D_{KL}(P\|Q)=+\infty$.
- $P\ll Q$: Every event with zero $Q$ probability also has zero $P$ probability.
- For finite alphabets, finiteness is equivalent to $Q(x)>0$ wherever $P(x)>0$; in continuous or countably infinite spaces, absolute continuity alone does not ensure a finite integral.

KLD (Entropy ver.):

$$
D_{KL}(P||Q)=H(P,Q)-H(P)
$$
- $H(P)$ is model-independent → Minimize CE = Minimize KLD.
- $H(P,Q)$ is convex for fixed $P$.
- **Coding clarification**: The extra-bits interpretation uses ideal log code lengths, or asymptotic coding. Finite prefix-code lengths must be integers.
- **Entropy-difference clarification**: Use this subtraction only when defined; the direct KL expression remains meaningful when subtracting entropies would give $\infty-\infty$.
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

3. **Proof clarification**: Step 2 proves nonnegativity, not convexity. The expectation is under $P$; writing the sum as $1$ requires that $Q$ put all its mass on the support of $P$. In general:

    $$
    D_{KL}(P\|Q)\geq-\log_2\sum_{x:P(x)>0}Q(x)\geq0.
    $$

4. **Joint-convexity proof**: For nonnegative sequences, the log-sum inequality is

    $$
    \sum_r a_r\log_2\frac{a_r}{b_r}
    \geq
    \left(\sum_r a_r\right)
    \log_2\frac{\sum_r a_r}{\sum_r b_r}.
    $$

    - $a_r$: Nonnegative numerator weights.
    - $b_r$: Nonnegative denominator weights.
    - $r$: Term index; zero terms follow the KL conventions.

5. Apply log-sum at each outcome with weights $\lambda$ & $1-\lambda$, then sum:

    $$
    D_{KL}\bigl(\lambda P_1+(1-\lambda)P_2\|
                    \lambda Q_1+(1-\lambda)Q_2\bigr)
    \leq
    \lambda D_{KL}(P_1\|Q_1)+(1-\lambda)D_{KL}(P_2\|Q_2).
    $$

    - $\lambda\in[0,1]$: Mixture weight.
    - $P_1,P_2$: Two source distributions.
    - $Q_1,Q_2$: Their corresponding comparison distributions.
```

```{note} Example
:class: dropdown
- $P=(1/2,1/2)$ & $Q=(3/4,1/4)$:

    $$
    \begin{align*}
    D_{KL}(P\|Q)&=\tfrac12\log_2(4/3),\\
    D_{KL}(Q\|P)&=\tfrac34\log_2(3/2)-\tfrac14.
    \end{align*}
    $$

- Replacing $Q$ by $(1,0)$ gives $D_{KL}(P\|Q)=+\infty$ but $D_{KL}(Q\|P)=1$ bit.
```

```{attention} Q&A
:class: dropdown
*Is KL a distance metric?*

- No symmetry guarantee & no triangle inequality.
- It is an expected log-likelihood ratio under its first argument, not a geometric distance between parameter vectors.

*Why do forward & reverse KL behave differently under model restrictions?*

- Minimizing $D_{KL}(P\|Q)$ averages over $P$: missing any positive-$P$ region with $Q=0$ is infinitely costly.
- Minimizing $D_{KL}(Q\|P)$ averages over $Q$: assigning mass where $P=0$ is infinitely costly, but omitted modes need not be directly penalized.
- “Mass covering” & “mode seeking” describe tendencies of restricted approximating families, not universal optimization theorems.

*Does changing coordinates change KL?*

- A common invertible transformation preserves it: the density Jacobians cancel in the ratio.
- A lossy common transformation can only decrease it.
```

&nbsp;

### Cross Entropy
- **What**: Total average surprise to use distribution $Q$ to approximate true distribution $P$.
- **Why**: Measure the whole prediction or coding cost, not only the penalty for mismatch.
- **How**: Draw outcomes from $P$, but score them using the probabilities assigned by $Q$.

```{note} Math
:class: dropdown
$$
\begin{align*}
H(P,Q)&=-\sum_xP(x)\log_2Q(x)\\
&=H(P)+D_{KL}(P\|Q)\geq H(P).
\end{align*}
$$

- Equality iff $P=Q$.
- Linear in $P$ for fixed $Q$ & convex in $Q$ for fixed $P$; joint convexity in $(P,Q)$ is not implied.
- For densities, $h(p,q)=-\int p\log_2q$ can be negative; $h(p,q)=h(p)+D_{KL}(P\|Q)$ requires well-defined terms.

Conditional prediction:

$$
\mathbb{E}_{P_{XY}}[-\log_2Q(Y|X)]
=H(Y|X)+\mathbb{E}_{P_X}D_{KL}(P_{Y|X}\|Q_{Y|X}).
$$

- $Q_{Y|X}$: Model's conditional prediction law.
- The irreducible cost is conditional entropy; excess cost is expected conditional KL.
```

```{tip} Derivation
:class: dropdown
*Why does minimizing log loss recover the true distribution?*

1. Insert $\log Q(x)=\log P(x)-\log[P(x)/Q(x)]$ into the cross-entropy sum.
2. The source entropy is fixed while the model varies.
3. KL is nonnegative & vanishes exactly when the distributions agree.
4. For observations sampled from $P$, empirical average negative log likelihood estimates the population cross entropy; minimizing it is not a guarantee of population optimality with finite data or a restricted model family.
```

```{attention} Q&A
:class: dropdown
*Is cross entropy a divergence?*

- Not by itself: $H(P,P)=H(P)$ is generally nonzero.
- Subtract the source entropy to obtain KL.

*What changes for sequences?*

- Factor $Q(x_{1:L})=\prod_tQ(x_t|x_{1:t-1})$.
- Expected sequence log loss is a sum of conditional cross entropies; no independence assumption is needed.
- Ideal per-symbol coding cost & log loss coincide when measured under the same source, model, and logarithm base.

*What is perplexity?*

- $2^{\text{cross entropy per symbol}}$; the effective branching factor implied by the model's average log loss.
- Compare only with the same symbolization & evaluation distribution; changing token units changes the scale.
```

&nbsp;

### Jensen–Shannon Divergence
- **What**: Average KL to a mixture of the compared distributions. {cite:p}`lin1991divergence`
- **Why**: Compare distributions symmetrically without infinite penalties for disjoint supports.
- **How**: Mix the two sources equally, then measure how much identifying each source changes the distribution.

```{note} Math
:class: dropdown
$$
\begin{align*}
M&=\tfrac12P+\tfrac12Q,\\
D_{JS}(P,Q)
&=\tfrac12D_{KL}(P\|M)+\tfrac12D_{KL}(Q\|M)\\
&=H(M)-\tfrac12H(P)-\tfrac12H(Q).
\end{align*}
$$

- $M$: Equal-weight mixture law.
- $D_{JS}$: Jensen–Shannon divergence.
- $0\leq D_{JS}(P,Q)\leq1$ bit.
- Zero iff $P=Q$; one bit iff the laws are mutually singular.
- Symmetric & jointly convex; $\sqrt{D_{JS}}$ is a metric, while $D_{JS}$ itself is not.
- The KL-mixture definition applies to continuous laws too; use the entropy difference only when its terms are finite.

For a source label $B$ with equal probabilities, $X|B=0\sim P$, $X|B=1\sim Q$:

$$
D_{JS}(P,Q)=I(B;X).
$$

- $B$: Label indicating which distribution generated the observation.
- For mixture weight $\alpha\in(0,1)$, replace $1/2$ by $\alpha$ & $1-\alpha$; the corresponding weighted divergence is bounded by $H_2(\alpha)$.
```

```{tip} Derivation
:class: dropdown
*Why is the divergence bounded by one bit?*

1. The mixture law is the marginal distribution of $X$ when the source label is hidden.
2. Average posterior information about the label equals $\tfrac12D_{KL}(P\|M)+\tfrac12D_{KL}(Q\|M)$.
3. $I(B;X)\leq H(B)=1$ bit.
4. Equality requires that the observation reveal the label perfectly, i.e., mutually singular source laws.
```

```{note} Example
:class: dropdown
- $P=(1,0)$, $Q=(0,1)$ → both directed KLs are infinite.
- Their mixture is $(1/2,1/2)$ → each KL to the mixture is $1$ bit → $D_{JS}=1$ bit.
```

```{dropdown} Table: Distribution Comparison
| Quantity | Expectation under | Zero condition | Symmetric? | Range in bits |
|:--|:--|:--|:--|:--|
| $H(P,Q)$ | $P$ | Both laws concentrate on the same outcome | No | $[0,\infty]$ for discrete laws |
| $D_{KL}(P\|Q)$ | $P$ | $P=Q$ | No | $[0,\infty]$ |
| $D_{JS}(P,Q)$ | Equal mixture of the two KL costs | $P=Q$ | Yes | $[0,1]$ |
```

```{attention} Q&A
:class: dropdown
*Why not average forward & reverse KL instead?*

- That symmetric sum or average still diverges for support mismatch.
- Jensen–Shannon compares each distribution with a mixture that contains its support.

*What information does the upper bound discard?*

- All disjoint-support pairs reach the same maximum, regardless of the geometric separation between their supports.
- It measures source distinguishability, not transport distance.
```

&nbsp;

### Divergence Chain Rule and Contraction
- **What**: Decomposition of distribution mismatch into marginal & conditional mismatch.
- **Why**: Track where a joint model is wrong & what remains detectable after processing.
- **How**: Factor both distributions in the same order; use the same observation channel for contraction.

```{note} Math
:class: dropdown
$$
D_{KL}(P_{XY}\|Q_{XY})
=D_{KL}(P_X\|Q_X)
+\mathbb{E}_{P_X}D_{KL}(P_{Y|X}\|Q_{Y|X}).
$$

- The conditional term is averaged under $P_X$, not $Q_X$.
- Assume $P_X\ll Q_X$ so the conditional comparison is specified where needed; support violations give infinite divergence.

For a shared channel $K(z|x)$:

$$
D_{KL}(PK\|QK)\leq D_{KL}(P\|Q).
$$

- $K(z|x)$: Same conditional output law applied to either input distribution.
- $PK$: Output law $\sum_xP(x)K(z|x)$.
- $QK$: Output law $\sum_xQ(x)K(z|x)$.
- Applying different channels need not contract divergence.
```

```{tip} Derivation
:class: dropdown
*Why does a shared channel contract KL?*

1. Form joint laws $P(x)K(z|x)$ & $Q(x)K(z|x)$.
2. The forward chain rule gives joint KL equal to $D_{KL}(P\|Q)$: the channel mismatch term is zero.
3. The reverse chain rule gives joint KL equal to output KL plus a nonnegative expected KL between conditional input laws.
4. Dropping that nonnegative term proves contraction.
```

```{attention} Q&A
:class: dropdown
*What is the cost of using independent models for independent samples?*

- $D_{KL}(P^{\otimes L}\|Q^{\otimes L})=L\,D_{KL}(P\|Q)$.
- $P^{\otimes L}$ denotes the law of $L$ independent draws from $P$.
- For dependent sequences, use the chain rule's history-conditioned mismatch terms instead.

*How does this relate to mutual-information data processing?*

- Write mutual information as KL between a joint law & the product of its marginals.
- Apply a common channel to the processed coordinate in both laws.
```

&nbsp;

### Total Variation and Pinsker's Inequality
- **What**: Event-probability discrepancy controlled by relative entropy.
- **Why**: Translate small KL into a bound on how differently two distributions can predict events.
- **How**: Maximize the probability gap over events; KL lower-bounds the squared gap.

```{note} Math
:class: dropdown
$$
\operatorname{TV}(P,Q)
=\sup_{\mathcal{A}}|P(\mathcal{A})-Q(\mathcal{A})|
=\frac12\sum_x|P(x)-Q(x)|.
$$

- $\operatorname{TV}$: Total variation distance.
- $\mathcal{A}$: Measurable event; the last expression is the discrete form.

Pinsker's inequality, with KL measured in bits:

$$
D_{KL}(P\|Q)\geq\frac{2}{\ln2}\operatorname{TV}(P,Q)^2.
$$

For equal prior probabilities in the binary test $P$ versus $Q$:

$$
P_e^*=\frac{1-\operatorname{TV}(P,Q)}{2}.
$$

- $P_e^*$: Minimum achievable probability of choosing the wrong source distribution from one observation.
```

```{attention} Q&A
:class: dropdown
*Does small total variation imply small KL?*

- Not without further assumptions. $P=(1-\epsilon,\epsilon)$ & $Q=(1,0)$ have total variation $\epsilon$ but infinite $D_{KL}(P\|Q)$ for every $\epsilon>0$.
- Pinsker controls event discrepancies from KL, not KL from event discrepancies.
```

&nbsp;

## Compression and Rate–Distortion Theory

### Source Coding
- **What**: Representation of source outcomes by codewords.
- **Why**: Storage & communication costs depend on code length, not the names of outcomes.
- **How**: Assign shorter codewords to more probable outcomes while keeping decoding unambiguous.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $c(x)$: Binary codeword assigned to outcome $x$.
- Misc:
    - $\ell(x)$: Integer length of $c(x)$.
    - $\bar{\ell}$: Expected code length $\sum_xP(x)\ell(x)$.

- **Uniquely decodable**: Every finite concatenation of codewords determines exactly one source sequence.
- **Prefix-free**: No codeword is a prefix of another; decoding can finish as soon as a codeword ends.
- Prefix-free implies uniquely decodable, but the reverse need not hold.

Kraft–McMillan inequality:

$$
\sum_{x\in\mathcal{X}}2^{-\ell(x)}\leq1.
$$

- Necessary for lengths of a uniquely decodable binary code.
- Conversely, nonnegative integer lengths satisfying it admit a prefix-free binary code.
- The length condition guarantees existence of a code, not unique decodability of arbitrary bit strings with those lengths.

For an optimal binary prefix code:

$$
H(X)\leq\bar{\ell}_{\mathrm{opt}}<H(X)+1.
$$

- $\bar{\ell}_{\mathrm{opt}}$: Minimum expected length among binary prefix codes.
- $P$ is known & only positive-probability outcomes require codewords.
- A deterministic source can use an empty codeword when the source-symbol count is known.
```

```{tip} Derivation
:class: dropdown
*Why does entropy bound expected prefix-code length?*

1. Let $s=\sum_x2^{-\ell(x)}\leq1$ & normalize the code lengths into a distribution $Q_\ell(x)=2^{-\ell(x)}/s$.
2. Substitute into KL:

    $$
    D_{KL}(P\|Q_\ell)=-H(X)+\bar{\ell}+\log_2s.
    $$

3. Thus $\bar{\ell}=H(X)+D_{KL}(P\|Q_\ell)-\log_2s\geq H(X)$.
4. For an upper bound, choose Shannon lengths $\ell(x)=\lceil-\log_2P(x)\rceil$. They satisfy Kraft because $2^{-\ell(x)}\leq P(x)$.
5. Each length is less than $-\log_2P(x)+1$; averaging proves the upper bound.
```

```{note} Example
:class: dropdown
- Probabilities $(1/2,1/4,1/8,1/8)$ admit codewords $(0,10,110,111)$.
- Lengths $(1,2,3,3)$ satisfy Kraft with equality.
- $\bar{\ell}=H(X)=7/4$ bits per symbol; a fixed two-bit code spends $1/4$ extra bit per symbol.
```

```{attention} Q&A
:class: dropdown
*When can expected length equal entropy exactly?*

- For a prefix code on the support, equality requires $P(x)=2^{-\ell(x)}$ for every outcome.
- Otherwise integer lengths generally cause a gap; block coding amortizes it.

*Does every lossless code obey $\bar{\ell}\geq H(X)$?*

- The bound above needs unique decodability of concatenations or a prefix constraint.
- An injective code for one message with externally supplied message boundaries has a different problem definition; omitting the boundary information can invalidate this bound.

*Where do familiar compressors fit?*

- Huffman coding optimizes expected length among symbol-by-symbol prefix codes for a known finite distribution.
- Arithmetic coding works on whole-sequence probabilities, avoiding a separate integer-length penalty for every source symbol.
- These construct codes; entropy states a limit independent of a particular construction.
```

&nbsp;

### Typical Sets and the AEP
- **Name**: Asymptotic Equipartition Property.
- **What**: Concentration of per-symbol surprise around the entropy rate.
- **Why**: Most long source sequences occupy an exponentially smaller set than all possible sequences.
- **How**: Average many log probabilities; concentration turns unequal symbol probabilities into nearly equal exponential-scale block probabilities.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\epsilon>0$: Tolerance in bits per symbol.
    - $\delta\in(0,1)$: Allowed atypical probability for sufficiently long blocks.
- Misc:
    - $\mathcal{A}^{(L)}_\epsilon$: Weakly typical set.

For an independent identically distributed finite-alphabet source:

$$
-\frac1L\log_2P(X_{1:L})
=\frac1L\sum_{t=1}^{L}-\log_2P(X_t)
\xrightarrow{\mathrm{a.s.}}H(X_1).
$$

- $\xrightarrow{\mathrm{a.s.}}$: Almost-sure convergence.

$$
\mathcal{A}^{(L)}_\epsilon
=\left\{x_{1:L}:
\left|-\frac1L\log_2P(x_{1:L})-H(X_1)\right|\leq\epsilon\right\}.
$$

For every typical sequence:

$$
2^{-L(H(X_1)+\epsilon)}
\leq P(x_{1:L})
\leq2^{-L(H(X_1)-\epsilon)}.
$$

For sufficiently large $L$:

$$
\begin{align*}
P(X_{1:L}\in\mathcal{A}^{(L)}_\epsilon)&\geq1-\delta,\\
(1-\delta)2^{L(H(X_1)-\epsilon)}
\leq|\mathcal{A}^{(L)}_\epsilon|
&\leq2^{L(H(X_1)+\epsilon)}.
\end{align*}
$$

- **Stationary ergodic extension**: Replace $H(X_1)$ by $\overline{H}(X)$; normalized block surprise converges almost surely to the entropy rate.
- **Ergodic**: Shift-invariant events have probability $0$ or $1$; the process does not select a persistent hidden stationary component with its own long-run statistics.
```

```{tip} Derivation
:class: dropdown
*Why are there about $2^{LH}$ typical sequences?*

1. Independence expresses block surprise as a sum of identically distributed individual surprises.
2. The law of large numbers gives the concentration statement.
3. Every typical sequence has probability at least $2^{-L(H+\epsilon)}$; total probability is at most $1$, giving the cardinality upper bound.
4. Typical probability is at least $1-\delta$ & each typical sequence has probability at most $2^{-L(H-\epsilon)}$, giving the lower bound.
5. Here $H$ abbreviates $H(X_1)$; the approximation is exponential-scale, not equality of individual probabilities.
```

```{note} Example
:class: dropdown
- For independent bits with $P(X_t=1)=1/4$, there are $2^L$ possible blocks.
- The typical set has exponential size about $2^{L H_2(1/4)}$, with $H_2(1/4)<1$.
- The all-zero block is individually most probable, but typical blocks have about one-quarter ones.
- A most probable sequence & a high-total-probability set answer different questions.
```

```{attention} Q&A
:class: dropdown
*Does “equipartition” mean exactly uniform?*

- No. Typical log probabilities differ by at most $2L\epsilon$; their ordinary probability ratio can still be exponentially large.
- What agrees asymptotically is per-symbol surprise.

*Can the theorem be used unchanged on a nonstationary source?*

- No. Distribution drift or nonergodic mixtures can prevent concentration at one deterministic entropy rate.
- The relevant coding limit then needs assumptions beyond the stationary ergodic formulation.
```

&nbsp;

### Lossless Compression Limits
- **What**: Fundamental rates for exact or asymptotically exact source reconstruction.
- **Why**: Separate entropy limits from finite-block overhead & the chosen error requirement.
- **How**: Use variable-length codes for every block, or fixed-length indices for a high-probability subset.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\ell_L$: Length of a prefix codeword for the entire block.
    - $P_e^{(L)}$: Block reconstruction error probability.

For any finite-alphabet block distribution, an optimal prefix code satisfies:

$$
\frac{H(X_{1:L})}{L}
\leq\frac{\mathbb{E}\ell_L}{L}
<\frac{H(X_{1:L})}{L}+\frac1L.
$$

- Every possible block is reconstructed exactly.
- For an independent identically distributed source, $H(X_{1:L})=LH(X_1)$.
- For a stationary finite-alphabet source, the optimal expected rate tends to $\overline{H}(X)$.

For fixed-length coding of an independent identically distributed finite-alphabet source:

- $R>H(X_1)$ → codes exist with $P_e^{(L)}\to0$.
- $R<H(X_1)$ → every sequence of codes at that rate has $P_e^{(L)}\to1$.
- $R=H(X_1)$ is a boundary; the strict-inequality statements alone do not determine its finite-block behavior.
- **Zero error at fixed length**: Every positive-probability block needs its own index, requiring at least $\lceil\log_2|\operatorname{supp}(P_{X_{1:L}})|\rceil$ bits.
```

```{tip} Derivation
:class: dropdown
*Why is entropy the threshold for fixed-length almost-lossless coding?*

1. **Achievability**: Pick $\epsilon>0$ with $H(X_1)+\epsilon<R$. Index all typical blocks; send a failure index for the rest.
2. Typical-set cardinality fits within the rate for large $L$; atypical probability tends to zero.
3. **Converse**: A rate-$R$ code can correctly reconstruct at most $2^{LR}$ distinct blocks.
4. For $R<H(X_1)-\epsilon$, the total probability of correctly decoded typical blocks is at most

    $$
    2^{LR}2^{-L(H(X_1)-\epsilon)}\longrightarrow0.
    $$

5. The atypical set also has vanishing probability, so correct reconstruction probability tends to zero.
```

```{attention} Q&A
:class: dropdown
*Does an entropy of less than one bit let every binary message be shortened?*

- No. Shortening likely strings requires longer descriptions for others, or permitting errors on some strings.
- Counting forbids an injective compressor from making every finite binary string shorter.

*What costs are excluded from the source-coding theorem?*

- A shared source model & codebook are assumed known.
- Model transmission, finite precision, headers, computation & latency can add practical costs.
```

&nbsp;

#### Compression with Side Information
- **What**: Lossless coding when correlated information is already available to the decoder.
- **Why**: Information the decoder already has need not be sent again.
- **How**: Identify the source sequence among those compatible with the decoder's side information.

```{note} Math
:class: dropdown
For independent identically distributed pairs $(X_t,Y_t)$ on finite alphabets:

- With $Y_{1:L}$ at the decoder, the infimum almost-lossless rate for $X_{1:L}$ is $H(X|Y)$.
- This rate is attainable asymptotically even if the encoder observes only $X_{1:L}$.

For separate encoders of both sources & a joint decoder, the closure of the achievable rate region is:

$$
\begin{align*}
R_X&\geq H(X|Y),\\
R_Y&\geq H(Y|X),\\
R_X+R_Y&\geq H(X,Y).
\end{align*}
$$

- $R_X$: Rate of the encoder observing $X$ only.
- $R_Y$: Rate of the encoder observing $Y$ only.
- **Slepian–Wolf region**: Joint reconstruction with block error tending to zero; not a finite-block zero-error guarantee.
```

```{tip} Derivation
:class: dropdown
*How can an encoder exploit side information it never sees?*

1. Randomly assign source blocks to bins; encoder sends only the bin index.
2. The decoder searches that bin for a block jointly typical with its side information.
3. There are exponentially about $2^{LH(X|Y)}$ conditional candidates, so a rate just above $H(X|Y)$ makes ambiguity unlikely.
4. The construction relies on shared binning rules & joint source statistics, not encoder access to the realized side information.
```

```{attention} Q&A
:class: dropdown
*Does the same rate hold if the side-information distribution changes?*

- Not automatically. The limit assumes the specified joint law & its conditional entropy.
- Side information independent of $X$ gives no gain; side information determining $X$ allows zero asymptotic rate.
```

&nbsp;

### Universal Coding and Description Length
- **What**: Source coding without knowing the exact generating distribution.
- **Why**: A compression claim should not hide the cost of selecting or communicating its model.
- **How**: Encode a model plus its residual data, or share a mixture of candidate source models.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\theta$: Index in a finite or countable collection of candidate source laws.
- Misc:
    - $P_\theta$: Candidate law for a complete source block.
    - $w_\theta$: Positive shared model weight, with $\sum_\theta w_\theta=1$.
    - $\ell(\theta)$: Prefix-code length describing a selected model.
    - $Q_{\mathrm{mix}}$: Mixture distribution over source blocks.

$$
Q_{\mathrm{mix}}(x_{1:L})=\sum_\theta w_\theta P_\theta(x_{1:L}).
$$

For every candidate with $w_\theta>0$:

$$
-\log_2Q_{\mathrm{mix}}(x_{1:L})
\leq-\log_2P_\theta(x_{1:L})-\log_2w_\theta.
$$

- The mixture's ideal code length is within the model's weight penalty of each candidate's ideal code length, pointwise in the observed block.
- Rounding to a prefix code adds less than one bit per block.

Two-part description length:

$$
\ell(\theta)+\left\lceil-\log_2P_\theta(x_{1:L})\right\rceil.
$$

- First describe the model; then describe the data using that model.
- Minimizing the combined length is the two-part minimum-description-length principle.
- Continuous model parameters require a coding or discretization scheme; an exact real-valued parameter is not a free finite message.
```

```{tip} Derivation
:class: dropdown
*Why does mixing bound regret against each candidate?*

1. The mixture contains every nonnegative summand, so $Q_{\mathrm{mix}}(x_{1:L})\geq w_\theta P_\theta(x_{1:L})$.
2. Apply the decreasing function $-\log_2$.
3. Divide by $L$: a fixed model penalty becomes negligible per symbol as $L$ grows.
```

```{attention} Q&A
:class: dropdown
*Does this give a universal compressor for every possible source?*

- No. The pointwise guarantee is relative to the specified model collection & weights.
- Vanishing per-symbol redundancy for a source class requires appropriate coverage or approximation properties.

*Why not select the best-fitting model without paying for it?*

- The decoder must know which model to use; ignoring that message overstates compression.
- Description length balances data fit against a specified coding cost, not an unspecified preference for “simplicity.”
```

&nbsp;

### Lossy Compression
- **What**: Source representation permitting controlled reconstruction error.
- **Why**: Exact recovery may spend bits on distinctions irrelevant to the reconstruction task.
- **How**: Map many source blocks to one reproduction block & quantify the lost distinctions with a distortion function.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\widehat{\mathcal{X}}$: Reconstruction alphabet, not necessarily equal to $\mathcal{X}$.
- Params:
    - $f_L$: Encoder mapping source blocks to a finite index set.
    - $g_L$: Decoder mapping indices to reconstruction blocks.
- Misc:
    - $M_L$: Number of available indices.
    - $J$: Transmitted index.
    - $d_L$: Mean distortion within a block.

$$
J=f_L(X_{1:L})\in\{1,\ldots,M_L\},\qquad
\hat{X}_{1:L}=g_L(J).
$$

$$
\begin{align*}
R_L&=\frac{\log_2M_L}{L},\\
d_L(x_{1:L},\hat{x}_{1:L})
&=\frac1L\sum_{t=1}^{L}d(x_t,\hat{x}_t),\\
\mathbb{E}d_L(X_{1:L},\hat{X}_{1:L})&\leq D.
\end{align*}
$$

- $R_L$: Block code rate in bits per source symbol; an integer binary index uses $\lceil\log_2M_L\rceil$ bits.
- **Hamming distortion**: $d(x,\hat{x})=\mathbf{1}\{x\neq\hat{x}\}$ → expected symbol error fraction.
- **Squared-error distortion**: $d(x,\hat{x})=(x-\hat{x})^2$ → expected squared reconstruction error.
- $\mathbf{1}\{\cdot\}$: Indicator, equal to $1$ when its condition holds & $0$ otherwise.
```

```{attention} Q&A
:class: dropdown
*Does a small expected distortion protect every example?*

- No. An expectation can hide large errors on rare outcomes.
- Worst-case distortion, excess-distortion probability & average distortion are different requirements.

*Is distortion inherently a distance metric?*

- No. It need not be symmetric or satisfy the triangle inequality.
- It specifies which reconstruction mistakes matter; information theory does not choose that criterion automatically.

*Is zero per-symbol error the same as zero block error asymptotically?*

- No. A vanishing expected fraction of incorrect symbols can coexist with a high probability that at least one symbol is wrong.
- The lossless theorem's block-error criterion is stronger.
```

&nbsp;

### Rate–Distortion Function
- **What**: Minimum asymptotic information rate at a prescribed expected distortion.
- **Why**: Quantify the best possible compression–fidelity tradeoff before choosing a compressor.
- **How**: Find the least informative reconstruction channel that meets the distortion constraint.

```{note} Math
:class: dropdown
For an independent identically distributed finite-alphabet source & a finite reconstruction alphabet with bounded distortion:

$$
R(D)=\min_{P_{\hat{X}|X}:\ \mathbb{E}d(X,\hat{X})\leq D}I(X;\hat{X}).
$$

- $P_{\hat{X}|X}$: Test channel, a conditional reconstruction law.
- The joint law in both the objective & constraint is $P_XP_{\hat{X}|X}$; $P_X$ is fixed.
- $R(D)=+\infty$ when the constraint is infeasible.
- Operationally, $R(D)$ is the infimum limiting rate of block codes with asymptotic expected distortion at most $D$.
- Rates strictly above $R(D)$ are achievable with distortion approaching at most $D$; lower rates cannot attain that asymptotic distortion.

$$
D_{\min}=\sum_xP(x)\min_{\hat{x}}d(x,\hat{x}),\qquad
D_0=\min_{\hat{x}}\sum_xP(x)d(x,\hat{x}).
$$

- $D_{\min}$: Smallest distortion possible when the reconstruction can depend freely on the input.
- $D_0$: Smallest distortion achievable with a constant reconstruction & no message.
- $R(D)$ is nonnegative, nonincreasing & convex on its feasible domain.
- $D\geq D_0$ → $R(D)=0$.
- With $\widehat{\mathcal{X}}=\mathcal{X}$ & $d(x,\hat{x})=0$ iff $x=\hat{x}$, $R(0)=H(X)$.

An equivalent supporting-line optimization at the corresponding tradeoff point:

$$
\min_{P_{\hat{X}|X}}\left\{I(X;\hat{X})+\beta\,\mathbb{E}d(X,\hat{X})\right\}.
$$

- $\beta\geq0$: Penalty per unit distortion; when $R$ is differentiable at an interior optimum, $\beta=-R'(D)$.
```

```{tip} Derivation
:class: dropdown
*Why mutual information rather than reconstruction entropy?*

1. A block code transmits only $J$. Data processing & the finite index set imply

    $$
    I(X_{1:L};\hat{X}_{1:L})\leq H(J)\leq\log_2M_L.
    $$

2. For an independent source, expand entropy & remove conditioning:

    $$
    \begin{align*}
    I(X_{1:L};\hat{X}_{1:L})
    &=\sum_{t=1}^{L}H(X_t)
      -\sum_{t=1}^{L}H(X_t|X_{1:t-1},\hat{X}_{1:L})\\
    &\geq\sum_{t=1}^{L}I(X_t;\hat{X}_t)\\
    &\geq\sum_{t=1}^{L}R(D_t)
    \geq L R\left(\frac1L\sum_{t=1}^{L}D_t\right).
    \end{align*}
    $$

    - $D_t=\mathbb{E}d(X_t,\hat{X}_t)$: Distortion at position $t$.
    - The last step uses convexity.

3. Combine the bounds to get the converse: rate must be at least the rate–distortion function of the achieved average distortion.
4. **Achievability mechanism**: Generate reproduction blocks from the test channel's marginal $P_{\hat{X}}$. A source-compatible reproduction has exponential-scale probability $2^{-LI(X;\hat{X})}$.
5. A codebook with slightly more than $2^{LI(X;\hat{X})}$ entries covers typical source blocks with high probability; joint typicality controls distortion. A fallback reconstruction controls atypical blocks because distortion is bounded.

*Why is $R(D)$ convex?*

1. Use two admissible test channels with an input-independent selector of probabilities $\lambda$ & $1-\lambda$.
2. Distortion averages linearly to $\lambda D_1+(1-\lambda)D_2$.
3. Mutual information without the selector is at most that with the selector; the latter is the weighted average of the two channel mutual informations.
4. Thus $R(\lambda D_1+(1-\lambda)D_2)\leq\lambda R(D_1)+(1-\lambda)R(D_2)$.
```

```{attention} Q&A
:class: dropdown
*Why optimize a stochastic channel if a deployed block encoder is deterministic?*

- The test channel describes the joint statistics that a large codebook should reproduce.
- It is not a requirement to send an independently randomized reconstruction for each symbol; that would generally waste rate.

*Does the same single-letter formula hold for sources with memory?*

- Not automatically. The source model, block distortion & asymptotic optimization must incorporate temporal dependence.
- Replacing $H(X)$ by an entropy rate does not by itself convert every rate–distortion formula to a dependent-source formula.

*Does low distortion preserve semantic content?*

- Only to the extent that the chosen distortion captures the distinctions of interest.
- Two distortion functions on the same source generally produce different rate–distortion curves.
```

&nbsp;

#### Binary Source with Hamming Distortion
- **What**: Exact compression–error curve for independent fair bits.
- **Why**: A discrete case where the entropy cost of tolerated mistakes is explicit.
- **How**: Allow symmetric reconstruction flips; their uncertainty reduces the required information rate.

```{note} Math
:class: dropdown
For $X\sim\operatorname{Bernoulli}(1/2)$ & $d(x,\hat{x})=\mathbf{1}\{x\neq\hat{x}\}$:

$$
R(D)=
\begin{cases}
1-H_2(D),&0\leq D\leq1/2,\\
0,&D\geq1/2.
\end{cases}
$$

- At $D=0$, one bit per source bit is necessary.
- At $D=1/2$, a constant reconstruction needs no message.
- The curve is asymptotic; it does not prescribe a one-symbol code of fractional length.
```

```{tip} Derivation
:class: dropdown
*Why subtract binary entropy?*

1. Let $E=\mathbf{1}\{X\neq\hat{X}\}$, with actual error probability $\delta\leq D\leq1/2$.
2. Given $\hat{X}$, the error bit determines $X$, so

    $$
    H(X|\hat{X})=H(E|\hat{X})\leq H(E)=H_2(\delta)\leq H_2(D).
    $$

3. Therefore $I(X;\hat{X})\geq1-H_2(D)$.
4. Take a fair $\hat{X}$ & an independent $E\sim\operatorname{Bernoulli}(D)$; set $X=\hat{X}\oplus E$. This joint law attains equality & has the required fair source marginal.
```

```{attention} Q&A
:class: dropdown
*What changes for biased bits?*

- The zero-rate reconstruction is the more probable symbol, not a fair random guess.
- The zero-rate threshold becomes $\min\{P(X=0),P(X=1)\}$; the fair-bit formula must not be applied unchanged.
```

&nbsp;

#### Gaussian Source with Squared-Error Distortion
- **What**: Exact compression–error curve for independent Gaussian samples.
- **Why**: Exposes the cost of preserving continuous precision.
- **How**: Allocate the allowed error to independent Gaussian residual uncertainty.

```{note} Math
:class: dropdown
For $X\sim\mathcal{N}(0,\sigma^2)$, $\sigma^2>0$, & $d(x,\hat{x})=(x-\hat{x})^2$:

$$
R(D)=
\begin{cases}
+\infty,&D=0,\\
\frac12\log_2\frac{\sigma^2}{D},&0<D<\sigma^2,\\
0,&D\geq\sigma^2.
\end{cases}
$$

- $\sigma^2$: Source variance.
- Rate is bits per scalar source sample.
- This is the continuous Gaussian theorem; it is not an application of the earlier finite-alphabet, bounded-distortion assumptions.
```

```{tip} Derivation
:class: dropdown
*Why does the variance ratio determine the rate?*

1. Let $E=X-\hat{X}$; its second moment is at most $D$.
2. Conditional translation, conditioning & the Gaussian maximum-entropy bound give

    $$
    h(X|\hat{X})=h(E|\hat{X})\leq h(E)
    \leq\frac12\log_2(2\pi eD).
    $$

3. Subtract from $h(X)=\frac12\log_2(2\pi e\sigma^2)$ to get $I(X;\hat{X})\geq\frac12\log_2(\sigma^2/D)$.
4. Attain the bound for $0<D<\sigma^2$ with the backward test channel

    $$
    X=\hat{X}+E,\qquad
    \hat{X}\sim\mathcal{N}(0,\sigma^2-D),\qquad
    E\sim\mathcal{N}(0,D),\qquad E\perp\hat{X}.
    $$

5. For $D\geq\sigma^2$, reconstruct $0$. At $D=0$, exact real-valued recovery requires infinite information.
```

```{attention} Q&A
:class: dropdown
*Can the optimal test channel be written as $X$ plus independent noise?*

- The independent residual in the construction is in $X=\hat{X}+E$, not generally in $\hat{X}=X+E$.
- The forward conditional mean is $\mathbb{E}[\hat{X}|X]=(1-D/\sigma^2)X$ for $0<D<\sigma^2$.

*How much does one extra bit per sample improve distortion?*

- In the positive-rate regime, $D=\sigma^2\,2^{-2R}$.
- Increasing $R$ by one bit divides distortion by four.
```

&nbsp;

### Information Bottleneck
- **What**: Compression of an observation while retaining information about a specified target. {cite:p}`tishby1999information`
- **Why**: Preserving every input detail wastes capacity when only target-relevant distinctions matter.
- **How**: Trade input information retained by a representation against target information lost through it.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $P_{XY}$: Fixed joint law of observation & relevance target.
- Params:
    - $P_{T|X}$: Stochastic encoder, with no direct access to $Y$.
- Hyperparams:
    - $\beta\geq0$: Relative weight on target information.

The encoder imposes $Y\to X\to T$:

$$
P(x,y,t)=P(x,y)P(t|x).
$$

Objective:

$$
\min_{P_{T|X}}\mathcal{L}_{\mathrm{IB}}
=\min_{P_{T|X}}\left\{I(X;T)-\beta I(T;Y)\right\}.
$$

- $\mathcal{L}_{\mathrm{IB}}$: Information-bottleneck objective.
- The representation alphabet is fixed or constrained as part of the optimization problem.

Define the representation-induced distortion:

$$
d_{\mathrm{IB}}(x,t)=D_{KL}(P_{Y|x}\|P_{Y|t}).
$$

- $d_{\mathrm{IB}}$: Cost of replacing the input's predictive distribution by its representation's predictive distribution.

$$
\begin{align*}
\mathbb{E}_{X,T}d_{\mathrm{IB}}(X,T)
&=I(X;Y)-I(T;Y)=I(X;Y|T),\\
\mathcal{L}_{\mathrm{IB}}
&=I(X;T)+\beta\mathbb{E}d_{\mathrm{IB}}(X,T)-\beta I(X;Y).
\end{align*}
$$

- The last term is fixed; the first two reveal a rate–distortion tradeoff.
- Unlike an externally specified distortion matrix, $d_{\mathrm{IB}}$ depends on the encoder through $P(Y|T)$.

For finite alphabets, interior stationary solutions satisfy:

$$
\begin{align*}
P(t|x)&=\frac{P(t)}{\mathcal{Z}_\beta(x)}
        2^{-\beta D_{KL}(P_{Y|x}\|P_{Y|t})},\\
P(t)&=\sum_xP(x)P(t|x),\\
P(y|t)&=\frac{\sum_xP(x,y)P(t|x)}{P(t)}.
\end{align*}
$$

- $\mathcal{Z}_\beta(x)$: Normalizer making $\sum_tP(t|x)=1$.
- Conditional target laws are needed only for active representations with $P(t)>0$.
- The base-$2$ exponential matches KL measured in bits.
- Self-consistency is necessary at an interior optimum, not a guarantee of the global minimum.
```

```{tip} Derivation
:class: dropdown
*Why does discarded target information become a KL distortion?*

1. Under $Y\to X\to T$, $P(Y|X,T)=P(Y|X)$.
2. Expand the expected predictive KL:

    $$
    \begin{align*}
    \mathbb{E}_{X,T}D_{KL}(P_{Y|X}\|P_{Y|T})
    &=\mathbb{E}\log_2\frac{P(Y|X)}{P(Y|T)}\\
    &=H(Y|T)-H(Y|X)\\
    &=I(X;Y)-I(T;Y).
    \end{align*}
    $$

3. Substitute this identity into the objective; $I(X;Y)$ is independent of the encoder.
4. Zero distortion means $P(Y|X)=P(Y|T)$ almost surely: the representation is sufficient for the chosen target.
```

```{note} Example
:class: dropdown
- $X=(Y,U)$, with independent fair bits $Y$ & $U$.
- $U$: Nuisance bit unrelated to the target.
- $T=X$ retains $2$ input bits & $1$ target bit.
- $T=Y$ retains $1$ input bit & the same $1$ target bit.
- A constant $T$ retains neither; the tradeoff decides whether target retention justifies its rate.
```

```{attention} Q&A
:class: dropdown
*What happens as the tradeoff weight changes?*

- At $\beta=0$, any input-independent representation minimizes the objective.
- In fact, $I(T;Y)\leq I(T;X)$ implies a constant representation is globally optimal for every $0\leq\beta\leq1$.
- Larger $\beta$ prioritizes target information; perfect retention still requires enough representation capacity.

*Why not use a deterministic continuous encoder without qualification?*

- For a finite discrete deterministic representation, $I(X;T)=H(T)$.
- If $T=f(X)$ has a non-atomic continuous law, $I(X;T)=+\infty$; the naive objective is not a finite compression measure.
- Quantization, a stochastic encoder, or an explicitly chosen finite-resolution problem changes what is being optimized.

*Does the objective explain every successful neural representation?*

- No. It defines a particular information-theoretic optimization problem; it does not establish that an arbitrary training procedure solves it.
- Relevance is specified by $Y$. Preserving information about one target does not guarantee usefulness for every future task.
```

&nbsp;

## Information Limits of Communication and Prediction

### Channel Capacity
- **What**: Maximum asymptotically reliable communication rate through a specified channel.
- **Why**: Noise limits the information any encoder–decoder pair can transmit.
- **How**: Choose input statistics & long codewords that remain distinguishable after channel corruption.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $K(y|x)$: Fixed discrete memoryless channel.
- Misc:
    - $C$: Capacity in bits per channel use.
    - $R_{\mathrm{ch}}$: Communication rate in bits per channel use.

$$
C=\max_{P_X}I(X;Y),\qquad P(x,y)=P_X(x)K(y|x).
$$

- Finite channel alphabets; no feedback or additional input constraint in this formula.
- **Memoryless**: $K(y_{1:L}|x_{1:L})=\prod_{t=1}^{L}K(y_t|x_t)$.
- With a uniform message encoded into $L$ channel inputs, $R_{\mathrm{ch}}<C$ permits block error tending to zero as $L\to\infty$.
- $R_{\mathrm{ch}}>C$ cannot have vanishing block error.
- An input cost constraint restricts the maximizing distributions; without such a constraint, some continuous channels have infinite capacity.

For a binary symmetric channel with flip probability $\epsilon$:

$$
C=1-H_2(\epsilon).
$$

- $\epsilon\in[0,1]$: Probability that the channel flips an input bit independently.
- A uniform input attains capacity.
```

```{tip} Derivation
:class: dropdown
*Why does the binary symmetric channel have this capacity?*

1. $H(Y|X)=H_2(\epsilon)$ for every input distribution.
2. Binary output implies $H(Y)\leq1$.
3. Thus $I(X;Y)\leq1-H_2(\epsilon)$.
4. A uniform input makes the output uniform, attaining the bound.
```

```{attention} Q&A
:class: dropdown
*How is channel capacity different from rate–distortion?*

- Capacity fixes a physical channel & maximizes over input laws.
- Rate–distortion fixes a source law & minimizes over admissible reconstruction channels.
- The former asks how much information can pass; the latter asks how much must pass.

*How do compression & communication combine?*

- For an independent identically distributed finite-alphabet source, a discrete memoryless channel & bounded distortion, let $\rho$ be channel uses per source symbol.
- $R(D)<\rho C$ is sufficient asymptotically; $R(D)>\rho C$ is impossible.
- Separate source coding & channel coding attain this asymptotic tradeoff; finite delay or different source/channel assumptions need separate analysis.

*Does a larger capacity guarantee better reasoning?*

- No. Capacity bounds reliable transmission, not the semantic value of the transmitted content or the computation needed to use it.
```

&nbsp;

### Fano's Inequality
- **What**: A lower bound on prediction error from remaining conditional uncertainty.
- **Why**: Distinguish an information-limited prediction task from a poor choice of predictor.
- **How**: Bound the bits needed to describe the target by whether the predictor erred & which alternative was correct.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\hat{Y}=g(X)$: Deterministic prediction from the observation.
- Misc:
    - $N=|\mathcal{Y}|\geq2$: Number of target classes.
    - $P_e=P(\hat{Y}\neq Y)$: Prediction error probability.

$$
H(Y|X)\leq H_2(P_e)+P_e\log_2(N-1)
\leq1+P_e\log_2N.
$$

- Holds for every predictor $g$, not only the optimal one.

When $Y$ is uniform over its $N$ classes:

$$
P_e\geq1-\frac{I(X;Y)+1}{\log_2N}.
$$

- Uniformity is needed to substitute $H(Y)=\log_2N$ in this simplified form.
- A negative lower bound is vacuous; probabilities are already bounded below by zero.
```

```{tip} Derivation
:class: dropdown
*How does uncertainty force errors?*

1. Let $E=\mathbf{1}\{\hat{Y}\neq Y\}$. Given $(X,Y)$, the error indicator is determined.
2. Expand the joint conditional entropy in two orders:

    $$
    H(Y|X)=H(E,Y|X)=H(E|X)+H(Y|E,X).
    $$

3. $H(E|X)\leq H(E)=H_2(P_e)$.
4. If $E=0$, the target is known from $X$; if $E=1$, at most $N-1$ target values remain.
5. Average those cases to get $H(Y|E,X)\leq P_e\log_2(N-1)$.
6. Use $H(Y|X)=H(Y)-I(X;Y)$ for the information-based error bound.
```

```{note} Example
:class: dropdown
- A uniformly distributed target has $16$ classes.
- Every available representation retains at most $1$ bit about the target.
- Fano gives $P_e\geq1-(1+1)/4=1/2$: no downstream classifier using only that representation can achieve error below one-half.
```

```{attention} Q&A
:class: dropdown
*Does high mutual information guarantee a computationally practical accurate predictor?*

- No. Fano is a converse: too little information forces errors.
- The inequality does not construct a predictor or bound the computation needed to decode the information.

*How does representation loss enter the bound?*

- Replace $X$ with the available representation $T$.
- For $Y\to X\to T$, data processing gives $I(T;Y)\leq I(X;Y)$; preprocessing cannot evade the information limit by itself.
```

&nbsp;

### Predictive Information
- **What**: Mutual information between an observed past & an unobserved future. {cite:p}`bialek2001predictability`
- **Why**: Separate reusable temporal structure from irreducible sequence randomness.
- **How**: Measure how much observing the past reduces uncertainty about a future block.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $F$: Number of future symbols considered.
- Misc:
    - $H_r=H(X_{1:r})$: Entropy of a length-$r$ block.
    - $\mathcal{E}$: Excess entropy, the infinite-past/infinite-future information.

For a stationary finite-alphabet process:

$$
\begin{align*}
I_{\mathrm{pred}}(L,F)
&=I(X_{1:L};X_{L+1:L+F})\\
&=H_L+H_F-H_{L+F}.
\end{align*}
$$

- $I_{\mathrm{pred}}$: Predictive information at the specified past & future horizons.
- Stationarity identifies the future block's entropy with $H_F$.

$$
\begin{align*}
\lim_{F\to\infty}I_{\mathrm{pred}}(L,F)
&=H_L-L\overline{H}(X),\\
\mathcal{E}
&=\lim_{L\to\infty}\left[H_L-L\overline{H}(X)\right]\\
&=\sum_{t=1}^{\infty}
 \left[H(X_t|X_{1:t-1})-\overline{H}(X)\right].
\end{align*}
$$

- The limits can be $+\infty$; the series has nonnegative terms.
- $\mathcal{E}$ measures the sublinear part of block entropy beyond the extensive entropy-rate term.
```

```{tip} Derivation
:class: dropdown
*Why subtract the entropy-rate contribution?*

1. For fixed $L$, the block-entropy increment is

    $$
    H_{F+L}-H_F
    =\sum_{t=F+1}^{F+L}H(X_t|X_{1:t-1}).
    $$

2. Each of these $L$ conditional entropies tends to $\overline{H}(X)$ as $F\to\infty$.
3. Substitute into $I_{\mathrm{pred}}(L,F)=H_L-(H_{F+L}-H_F)$.
4. Expand $H_L$ with the entropy chain rule & then let $L$ grow to obtain the nonnegative series.
```

```{note} Example
:class: dropdown
- **Independent fair bits**: $\overline{H}(X)=1$, $H_L=L$, $\mathcal{E}=0$. Maximal randomness, no predictive information.
- **One fair bit repeated forever**: $\overline{H}(X)=0$, $H_L=1$, $\mathcal{E}=1$. One persistent bit, not continuing novelty.
- **Stationary fair binary Markov chain** with flip probability $\epsilon$: $H_L=1+(L-1)H_2(\epsilon)$, hence $\mathcal{E}=1-H_2(\epsilon)$.
```

```{attention} Q&A
:class: dropdown
*How does predictive information guide a memory representation?*

- Let $T$ depend only on the observed past. Data processing bounds its future information by that of the full past.
- An information bottleneck with the future as relevance target asks which past distinctions should survive a memory constraint.
- The chosen future horizon & observable variables determine which distinctions count as relevant.

*Is predictive information a definition of intelligence or consciousness?*

- No. Simple processes can carry persistent predictive information.
- The quantity captures statistical predictability, not agency, semantic understanding, or subjective experience.
```

&nbsp;