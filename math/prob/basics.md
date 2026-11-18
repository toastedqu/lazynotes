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
# Basics
Study notes from {cite:t}`math` and {cite:t}`prob_lifesaver`.
Probability space → random variables → joint behavior → summaries → transforms.
Named distributions live in [Distributions](dist.md).

Notations:
- $\Omega$: Sample space.
- $\mathcal{F}$: σ-algebra of events.
- $P$: Probability measure.
- $A$: Event.
- $B$: Second event.
- $X$: Random variable.
- $Y$: Second random variable.
- $F$: CDF.
- $f$: PDF (continuous) or PMF (discrete).
- $\mu=E[X]$: Mean.
- $\sigma^2=Var[X]$: Variance.

&nbsp;

## Probability
### Sample Space 
- **What**: Set of all possible outcomes of a random experiment ($\Omega$).
    - e.g., Coin flip: $\Omega=\{\text{head}, \text{tail}\}$.
    - e.g., Time till the universe ends: $\Omega=[0,\infty)$.

### Event
- **What**: A subset of the sample space ($A\subseteq\Omega$).
    - e.g., Coin flip is heads: $A=\{\text{head}\}$.
    - e.g., The universe ends within 60 seconds: $A=[0,60]$.

### σ-algebra
- **What**: A menu of allowed events in the sample space w/ 3 conditions ($\mathcal{F}$):
    1. **It contains the whole space**: $\Omega\in\mathcal{F}$.
    2. **It is closed under complements**: $A\in\mathcal{F}\rightarrow A^{c}=\Omega\backslash A \in\mathcal{F}$.
    3. **It is closed under countable unions**: $A_1,A_2,\cdots\in\mathcal{F}\rightarrow \bigcup_{i=1}^\infty A_i\in\mathcal{F}$.
- **Why**:
    - *Why these 3 specific conditions?*
        - These 3 closure rules are the minimum for events to behave like sensible boolean (yes/no) questions & for probability to work reliably in infinite processes.
        - Rule 1 ← "Something happened for sure" is an event.
        - Rule 2 ← "$A$ happened" and "$A$ didn't happen" are both events.
        - Rule 3 ← "At least one of these events happened" is an event.
    - *Why countable unions?*
        - **Countable union**: An OR-list we can list one-by-one in a sequence.
            - e.g., Set of rational numbers $\mathbb{Q}$.
        - **Uncountable union**: An OR-list we cannot list one-by-one in a sequence.
            - e.g., Set of every real number between 0 and 1 $[0,1]$.
        - We cannot assign probabilities to EVERY subset in an uncountable union.
            - e.g., It's impossible to assign a probability to a Vitali set in $[0,1]$.
            - We can still have σ-algebra on $[0,1]$, but it won't work on EVERY subset of it.
- **How**: Allowed events if $A_1,A_2,\cdots\in\mathcal{F}$:

| Operation              | Notation                   |
|:---------------------- |:-------------------------- |
| Complement             | $A^{c}$                    |
| Countable union        | $\bigcup_{i=1}^\infty A_i$ |
| Finite union           | $\bigcup_{i=1}^n A_i$      |
| Countable intersection | $\bigcap_{i=1}^\infty A_i$ |
| Finite intersection    | $\bigcap_{i=1}^n A_i$      |
| Difference             | $A\backslash B$            |
| Empty set              | $\varnothing$              |

&nbsp;

### Probability
- **What**: A real function $P:\mathcal{F}\rightarrow [0,1]$ satisfying the 3 Kolmogorov axioms:
    - **Non-negativity**: $\forall A\in\mathcal{F}: P(A)\geq 0$
    - **Normalization**: $P(\Omega)=1$ (and $P(\varnothing)=0$)
    - **Countable additivity**: If $A_i$ are mutually exclusive events, then $P(\bigcup_{i=1}^\infty A_i)=\sum_{i=1}^\infty P(A_i)$
- **Why**:
    - *Why these 3 specific axioms?*
        - These 3 axioms are the minimal rules for "measuring uncertainty as a number" to behave consistently w/ ordinary logic about events AND to work for infinite processes.
        - Axiom 1 ← A negative probability means "less than impossible". What?
        - Axiom 2 ← "Anything in the sample space can happen" is a certainty, so 1.
        - Axiom 3 ← If events cannot happen together, then the chance of "one of them happens" should be the sum of their chances.
- **How**: Rules for probability:

| Rule                  | Formula                                                                                                            |
|:-------------------------------- |:-------------------------------------------------------------------------------------------------------------------- |
| Complement                       | $P(A^c)=1-P(A)$                                                                                                      |
| Monotonicity                     | $A\subseteq B\rightarrow P(A)\leq P(B)$                                                                              |
| Difference                       | $A\subseteq B\rightarrow P(B\backslash A)=P(B)-P(A)$                                                                 |
| Inclusion-Exclusion (2 events)   | $P(A\cup B)=P(A)+P(B)-P(A\cap B)$                                                                                    |
| Boole's Inequality (union bound) | $P\left(\bigcup_i A_i\right)\leq \sum_i P(A_i)$                                                                      |
| Mutual exclusivity (disjoint)    | $A\cap B=\varnothing\rightarrow P(A\cap B)=0$                                                                        |
| Independence                     | $P(A\cap B)=P(A)P(B)$                                                                                                |
| Continuity from below            | $A_1\subseteq A_2\subseteq\cdots\rightarrow P\left(\bigcup_{i=1}^\infty A_i\right)=\lim_{i\rightarrow\infty}P(A_i)$ |
| Continuity from above            | $A_1\supseteq A_2\supseteq\cdots\rightarrow P\left(\bigcap_{i=1}^\infty A_i\right)=\lim_{i\rightarrow\infty}P(A_i)$ |

&nbsp;

### Conditional Probability
- **What**: Probability of event $B$ when event $A$ has already occurred.
$$
P(B|A)=\frac{P(A\cap B)}{P(A)}
$$
- **Why**: Conditioning updates probabilities when we learn new info.
- **How**: Rules for conditional probability:

| Rule           | Formula                                                                                         |
|:------------------------|:-----------------------------------------------------------------------------------------------|
| Multiplication           | $P(A\cap B)=P(A\mid B)P(B)=P(B\mid A)P(A)$                                                      |
| Chain rule               | $P\left(\bigcap_{i=1}^n A_i\right)=\prod_{i=1}^n P\left(A_i\mid \bigcap_{j=1}^{i-1} A_j\right)$ |
| Law of total probability | $P(B)=\sum_i P(B\mid A_i)P(A_i)$                                                                |
| Bayes' theorem           | $P(A\mid B)=\frac{P(B\mid A)P(A)}{P(B)}$                                                        |
| Independence             | $P(A\mid B)=P(A)$                                                                               |

&nbsp;

### Independence
- **What**: Joint probability = product of individual probabilities.
    - Events: $P(A\cap B)=P(A)P(B)$.
    - RVs: joint CDF factorizes, $F_{X,Y}(x,y)=F_X(x)F_Y(y)\ \forall x,y$.
- **Why**: Factorization.
    - Knowing $B$ does not change belief in $A$: $P(A\mid B)=P(A)$ if $P(B)>0$.
    - Joint → product of marginals → iid likelihood = product → log-likelihood = sum.
- **How**: 4 levels:
    - **Pairwise**: every pair of events factorizes.
    - **Mutual**: every finite subfamily factorizes. Mutual → pairwise. ❌ Converse.
    - **Conditional** ($A\perp B\mid C$): factorization under $P(\cdot\mid C)$.
    - **RVs**: every event about $X$ is independent of every event about $Y$ ⇔ joint CDF/PDF/PMF factorizes.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $A_i$: $i$-th event; $i$ = event idx (overrides sample idx).
    - $n$: #events (overrides #features).
    - $\mathcal{I}$: Subset of event indices.
    - $C$: Conditioning event w/ $P(C)>0$.
    - $Z$: Third RV.
    - $\mathcal{S}$: Borel set for $X$.
    - $\mathcal{T}$: Borel set for $Y$.
    - $\perp$: "is independent of".

Events:

$$
\begin{align*}
&\text{Pairwise:} && P(A_i\cap A_{i'})=P(A_i)P(A_{i'})\quad\forall i\neq i' \\
&\text{Mutual:} && P\left(\bigcap_{i\in\mathcal{I}}A_i\right)=\prod_{i\in\mathcal{I}}P(A_i)\quad\forall\mathcal{I}\subseteq\{1,\ldots,n\},\ |\mathcal{I}|\geq 2 \\
&\text{Conditional:} && P(A\cap B\mid C)=P(A\mid C)P(B\mid C)
\end{align*}
$$

- Mutual: $2^n-n-1$ equations. Pairwise: $\binom{n}{2}$ equations.

RVs:

$$
\begin{align*}
X\perp Y
&\Leftrightarrow P(X\in\mathcal{S},Y\in\mathcal{T})=P(X\in\mathcal{S})P(Y\in\mathcal{T})\quad\forall\mathcal{S},\mathcal{T} \\
&\Leftrightarrow F_{X,Y}(x,y)=F_X(x)F_Y(y)\quad\forall x,y \\
&\Leftrightarrow f_{X,Y}(x,y)=f_X(x)f_Y(y)\quad\text{(PMF: }\forall x,y\text{; PDF: almost everywhere)}
\end{align*}
$$

$$
X\perp Y\mid Z\Leftrightarrow f(x,y\mid z)=f(x\mid z)f(y\mid z)
$$

- Joint & conditional densities: [Joint, Marginal & Conditional Distributions](#joint-marginal-conditional-distributions).
- $X\perp Y$ → $g(X)\perp h(Y)$ for any measurable $g,h$.
- $X\perp Y$ (w/ finite means) → $E[XY]=E[X]E[Y]$; w/ finite variances → $Var[X+Y]=Var[X]+Var[Y]$.
```

```{tip} Derivation
:class: dropdown
*Why does $A\perp B$ give $A\perp B^c$?*

1. $A$ = disjoint union of $A\cap B$ & $A\cap B^c$:

$$
P(A\cap B^c)=P(A)-P(A\cap B)
$$

2. Use $A\perp B$:

$$
P(A\cap B^c)=P(A)-P(A)P(B)=P(A)(1-P(B))=P(A)P(B^c)
$$

*Why does a factorized CDF give a factorized PDF?*

1. $f_{X,Y}=\frac{\partial^2F_{X,Y}}{\partial x\,\partial y}$.
2. $F_{X,Y}(x,y)=F_X(x)F_Y(y)$ → $\frac{\partial^2}{\partial x\,\partial y}\left[F_X(x)F_Y(y)\right]=f_X(x)f_Y(y)$.
```

```{note} Example
:class: dropdown
Setup: 2 fair coin flips.
- $A$: 1st flip = H.
- $B$: 2nd flip = H.
- $C$: Both flips equal.
- $P(A)=P(B)=P(C)=\frac{1}{2}$.

**Pairwise ✅, mutual ❌:**
- $P(A\cap B)=P(A\cap C)=P(B\cap C)=P(HH)=\frac{1}{4}$ → every pair independent.
- $P(A\cap B\cap C)=P(HH)=\frac{1}{4}\neq\frac{1}{8}$ → not mutually independent.

**Independent ✅, conditionally independent ❌** (explaining away):
- $A\perp B$, but given $C$: $P(A\cap B\mid C)=\frac{1/4}{1/2}=\frac{1}{2}$.
- $P(A\mid C)P(B\mid C)=\frac{1}{2}\cdot\frac{1}{2}=\frac{1}{4}\neq\frac{1}{2}$.
- Given "both equal", the 1st flip determines the 2nd.

**Conditionally independent ✅, independent ❌** (common cause):
- Pick coin 1 ($P(H)=0.9$) or coin 2 ($P(H)=0.1$) w/ prob $\frac{1}{2}$ each. Flip it twice.
- Given the coin: flips independent.
- Unconditionally: $P(A)=P(B)=0.5$, but $P(A\cap B)=\frac{1}{2}(0.81)+\frac{1}{2}(0.01)=0.41\neq 0.25$.
- 1st H → coin 1 more likely → 2nd H more likely.
```

```{attention} Q&A
:class: dropdown
*Independent vs mutually exclusive?*

- Disjoint w/ $P(A),P(B)>0$ → $P(A\cap B)=0\neq P(A)P(B)$ → dependent.
- Knowing $A$ happened → $B$ did not happen. Maximal dependence.

*Can an event be independent of itself?*

- $P(A)=P(A)^2$ → only if $P(A)\in\{0,1\}$.

*Is pairwise independence enough?*

- No. 2 coins + "both equal" (Example) → pairwise ✅, mutual ❌.
- Pairwise independence still gives $Var[\sum_iX_i]=\sum_iVar[X_i]$ ← only pairwise covariances appear.

*Does independence imply conditional independence, or the reverse?*

- Neither.
- Independent causes + observed common effect → dependent given the effect (explaining away).
- Common cause → dependent unconditionally, independent given the cause.

*Uncorrelated vs independent?*

- Independent → uncorrelated (finite 2nd moments). ❌ Converse. See [Covariance & Correlation](#covariance-correlation).

*Where does ML use independence?*

- iid data → likelihood factorizes → sum of per-sample losses.
- [Naive Bayes](../../ml/supervised.md#naive-bayes): features conditionally independent given the class.
- [Bayesian networks](bayes.md#bayesian-networks): graph structure encodes conditional independences.
```

&nbsp;

### Probability Space
- **What**: $(\Omega,\mathcal{F},P)$
- **Why**:
    - We need a container that packages everything we need to do probability in a mathematically consistent way.
    - It specifies what random outcomes are allowed.
    - It specifies how likely each event is.
    - It provides the base for random variables ← ANY random variable is defined on this space.
    - It functions as the ground truth model for a random event. We can derive everything from it.

&nbsp;

## Random Variables & Distributions
### Random Variable
- **What**: A variable which takes its value randomly from a subset of $\mathbb{R}$.

&nbsp;

### Distribution
- **What**: The probability that the random variable $X$ takes a value at most $x$:
$$\begin{align*}
&\text{Discrete:}   && F(x)=P(X\leq x)=\sum_{x_i\leq x}P(X=x_i) \\
&\text{Continuous:} && F(x)=P(X\leq x)=\int_{-\infty}^xf(t)dt
\end{align*}$$
- **Why**: Distribution tells us everything probabilistic about a random variable.
- **How**: Properties:
    - $F(-\infty)=0, F(\infty)=1$
    - $F(x)$ is non-decreasing.
    - $F(x)$ is continuous on the right (left if defined as $P(X<x)$).
    - $F'(x)=f(x)$ if $f(x)$ is continuous.
    - No probability at any point for continuous distributions: $\forall c\in\mathbb{R}: P(X=c)=0$

&nbsp;

### Joint, Marginal & Conditional Distributions
- **What**: Distributions of ≥2 RVs.
    - **Joint** $f(x,y)$: all RVs together.
    - **Marginal** $f(x)$: a subset; other RVs summed/integrated out.
    - **Conditional** $f(y\mid x)$: one RV w/ the other fixed.
- **Why**: 1 RV at a time misses dependence.
    - Joint = full model. Marginalize & condition = the 2 basic operations of inference.
- **How**:
    1. **Marginalize**: sum/integrate the joint over unwanted RVs.
    2. **Condition**: slice the joint at the observed value → renormalize by the marginal.
    3. **Flip** w/ [Bayes' rule](bayes.md#bayes-rule): $f(x\mid y)\propto f(y\mid x)f(x)$.
    4. **Factorize** w/ the chain rule: joint = product of conditionals.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $F_{X,Y}$: Joint CDF.
    - $f_{X,Y}$: Joint PDF/PMF.
    - $f_X$: Marginal PDF/PMF of $X$.
    - $f_{Y\mid X}$: Conditional PDF/PMF of $Y$ given $X$.
    - $X_j$: $j$-th of $n$ RVs.

Joint:

$$
F_{X,Y}(x,y)=P(X\leq x,Y\leq y),\qquad
f_{X,Y}(x,y)=\begin{cases}
P(X=x,Y=y) & \text{discrete}\\
\frac{\partial^2F_{X,Y}(x,y)}{\partial x\,\partial y} & \text{continuous}
\end{cases}
$$

- Continuous: $P((X,Y)\in\mathcal{S})=\iint_{\mathcal{S}}f_{X,Y}(x,y)\,dx\,dy$ for a region $\mathcal{S}\subseteq\mathbb{R}^2$.
- Subscripts dropped when the args identify the RVs: $f(y\mid x)=f_{Y\mid X}(y\mid x)$.

Marginal:

$$
\begin{align*}
&\text{Discrete:}   && f_X(x)=\sum_yf_{X,Y}(x,y) \\
&\text{Continuous:} && f_X(x)=\int_{-\infty}^{\infty}f_{X,Y}(x,y)\,dy
\end{align*}
$$

Conditional:

$$
f_{Y\mid X}(y\mid x)=\frac{f_{X,Y}(x,y)}{f_X(x)},\qquad f_X(x)>0
$$

Bayes' rule:

$$
f_{X\mid Y}(x\mid y)=\frac{f_{Y\mid X}(y\mid x)f_X(x)}{\int f_{Y\mid X}(y\mid x')f_X(x')\,dx'}
$$

- $x'$: Integration dummy. Discrete: sum instead of integral.

Chain rule:

$$
f(x_1,\ldots,x_n)=\prod_{j=1}^{n}f(x_j\mid x_1,\ldots,x_{j-1})
$$

- Every ordering of the $X_j$ gives a valid factorization.
```

```{tip} Derivation
:class: dropdown
*Why is $f(y\mid x)$ a valid conditional density if $P(X=x)=0$?*

1. Condition on a thin slab w/ positive prob:

$$
P(Y\leq y\mid x\leq X\leq x+h)=\frac{\int_x^{x+h}\int_{-\infty}^{y}f(s,t)\,dt\,ds}{\int_x^{x+h}f_X(s)\,ds}
$$

- $s,t$: Integration dummies.
- $h>0$: Slab width.

2. $h\to 0$, $f$ continuous → numerator ≈ $h\int_{-\infty}^{y}f(x,t)\,dt$, denominator ≈ $hf_X(x)$.
3. → $\int_{-\infty}^{y}\frac{f(x,t)}{f_X(x)}\,dt$ → differentiate in $y$ → $\frac{f(x,y)}{f_X(x)}$.

- Caveat: on a prob-0 event, the answer depends on the limit procedure (Borel–Kolmogorov paradox). Different slab shapes can give different conditionals.
```

```{note} Example
:class: dropdown
Joint PMF of $X,Y\in\{0,1\}$:

| | $Y=0$ | $Y=1$ | $f_X$ |
|:--|:--|:--|:--|
| $X=0$ | 0.4 | 0.1 | 0.5 |
| $X=1$ | 0.2 | 0.3 | 0.5 |
| $f_Y$ | 0.6 | 0.4 | 1 |

- Marginals = row/column sums.
- Conditional: $f(Y=1\mid X=1)=\frac{0.3}{0.5}=0.6$.
- Bayes: $f(X=1\mid Y=1)=\frac{0.6\cdot 0.5}{0.4}=0.75=\frac{0.3}{0.4}$ ✅.
- Dependent: $f(1,1)=0.3\neq f_X(1)f_Y(1)=0.2$.
```

```{attention} Q&A
:class: dropdown
*Do the marginals determine the joint?*

- No. $X,Y\sim\text{Bernoulli}(\frac{1}{2})$: $Y=X$ & $X\perp Y$ have the same marginals.
- Missing piece = dependence structure → **copula** (Sklar's theorem).
- Exception: [independence](#independence) → joint = product of marginals.

*Is $f(y\mid x)$ a density in $x$?*

- In $y$: yes, $\int f(y\mid x)\,dy=1$.
- In $x$: no. As a function of $x$ (or a parameter) it is a **likelihood**. Its integral need not be 1.

*Why is marginalization expensive?*

- Sum over all joint values of the removed RVs → exponential in #discrete RVs.
- Fixes: exploit factorization (dynamic programming, e.g., [HMM](process.md#hmm) forward algorithm), sampling, variational bounds.

*Which chain-rule order to use?*

- All orders give the same joint.
- Order matters for modeling: autoregressive LMs use $p(y)=\prod_tp(y_t\mid y_{<t})$ for token sequence $y$ → each factor = one softmax.

*Mixed discrete & continuous RVs?*

- PMF in the discrete arg, PDF in the continuous arg.
- e.g., mixture model: $f(x)=\sum_kP(k)f(x\mid k)$ w/ discrete component $k$.
```

&nbsp;

### Change of Variables
- **What**: Density of $Y=g(X)$ from the density of $X$.
- **Why**: Mass conservation.
    - Density = probability per unit length/volume.
    - $g$ stretches/compresses space → density must shrink/grow by the same factor.
- **How**:
    1. Invert: $x=g^{-1}(y)$.
    2. Evaluate $f_X$ at $x$.
    3. Divide by the local volume stretch $|\det J_g(x)|$ (1-D: $|g'(x)|$).
    4. $g$ not injective → sum steps 1–3 over all preimages of $y$.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{X}\in\mathbb{R}^n$: Random vector.
    - $\mathbf{Y}=g(\mathbf{X})\in\mathbb{R}^n$: Transformed random vector.
- Misc:
    - $g$: Transformation.
    - $J_g(\mathbf{x})=\frac{\partial g(\mathbf{x})}{\partial\mathbf{x}}\in\mathbb{R}^{n\times n}$: Jacobian of $g$ at $\mathbf{x}$.

1-D, $g$ strictly monotone & differentiable:

$$
f_Y(y)=f_X(g^{-1}(y))\left|\frac{dg^{-1}(y)}{dy}\right|=\frac{f_X(x)}{|g'(x)|},\qquad x=g^{-1}(y)
$$

Multivariate, $g$ bijective & differentiable w/ differentiable inverse:

$$
f_{\mathbf{Y}}(\mathbf{y})=f_{\mathbf{X}}(g^{-1}(\mathbf{y}))\left|\det J_{g^{-1}}(\mathbf{y})\right|=\frac{f_{\mathbf{X}}(\mathbf{x})}{\left|\det J_g(\mathbf{x})\right|},\qquad\mathbf{x}=g^{-1}(\mathbf{y})
$$

Non-injective, 1-D, finitely many preimages w/ $g'(x)\neq 0$:

$$
f_Y(y)=\sum_{x:\,g(x)=y}\frac{f_X(x)}{|g'(x)|}
$$

Log form (normalizing flows), $g=g_L\circ\cdots\circ g_1$:

$$
\log f_{\mathbf{Y}}(\mathbf{y})=\log f_{\mathbf{X}}(\mathbf{x}_0)-\sum_{\ell=1}^{L}\log\left|\det J_{g_\ell}(\mathbf{x}_{\ell-1})\right|
$$

- $\mathbf{x}_\ell=g_\ell(\mathbf{x}_{\ell-1})$: Output of layer $\ell$; $\mathbf{x}_0=\mathbf{x}$, $\mathbf{x}_L=\mathbf{y}$.
- $L$: #layers.
```

```{tip} Derivation
:class: dropdown
*1-D:*

1. $g$ increasing: $F_Y(y)=P(g(X)\leq y)=P(X\leq g^{-1}(y))=F_X(g^{-1}(y))$.
2. Differentiate: $f_Y(y)=f_X(g^{-1}(y))\frac{dg^{-1}(y)}{dy}$.
3. $g$ decreasing: $F_Y(y)=P(X\geq g^{-1}(y))=1-F_X(g^{-1}(y))$ → $f_Y(y)=-f_X(g^{-1}(y))\frac{dg^{-1}(y)}{dy}$.
4. Decreasing → $\frac{dg^{-1}}{dy}<0$ → both cases = $f_X(g^{-1}(y))\left|\frac{dg^{-1}(y)}{dy}\right|$.
5. Inverse function theorem: $\frac{dg^{-1}(y)}{dy}=\frac{1}{g'(x)}$.

*Multivariate:*

1. For any region $\mathcal{S}$:

$$
P(\mathbf{Y}\in\mathcal{S})=P(\mathbf{X}\in g^{-1}(\mathcal{S}))=\int_{g^{-1}(\mathcal{S})}f_{\mathbf{X}}(\mathbf{x})\,d\mathbf{x}
$$

2. Substitute $\mathbf{x}=g^{-1}(\mathbf{y})$, $d\mathbf{x}=\left|\det J_{g^{-1}}(\mathbf{y})\right|d\mathbf{y}$:

$$
P(\mathbf{Y}\in\mathcal{S})=\int_{\mathcal{S}}f_{\mathbf{X}}(g^{-1}(\mathbf{y}))\left|\det J_{g^{-1}}(\mathbf{y})\right|d\mathbf{y}
$$

3. True for every $\mathcal{S}$ → the integrand is $f_{\mathbf{Y}}$.
4. $J_{g^{-1}}(\mathbf{y})=J_g(\mathbf{x})^{-1}$ → $\det J_{g^{-1}}(\mathbf{y})=\frac{1}{\det J_g(\mathbf{x})}$.

*Non-injective:* split the domain into pieces where $g$ is monotone → apply the 1-D formula per piece → add the contributions.
```

```{note} Example
:class: dropdown
**Affine**: $Y=aX+b$, $a\neq 0$ → $f_Y(y)=\frac{1}{|a|}f_X\left(\frac{y-b}{a}\right)$.

**Square** (non-injective): $X\sim\mathcal{N}(0,1)$, $Y=X^2$.
- Preimages of $y>0$: $\pm\sqrt{y}$. $|g'(\pm\sqrt{y})|=2\sqrt{y}$.
- $f_Y(y)=2\cdot\frac{1}{\sqrt{2\pi}}e^{-y/2}\cdot\frac{1}{2\sqrt{y}}=\frac{e^{-y/2}}{\sqrt{2\pi y}}$ → $\chi^2_1$ ✅.

**Exp** (mode moves): $X\sim\mathcal{N}(0,1)$, $Y=e^X$.
- $f_Y(y)=\frac{1}{y}\cdot\frac{1}{\sqrt{2\pi}}e^{-(\log y)^2/2}$ → lognormal.
- Mode of $X$ = 0, but mode of $Y$ = $e^{-1}\neq e^0$ ← the $\frac{1}{y}$ Jacobian factor.
```

````{important} Code
:class: dropdown
```python
import torch
from torch.autograd.functional import jacobian
from torch.distributions import MultivariateNormal

class TransformedDensity:
    def __init__(self, base, g, g_inv):
        ## base: distribution of X; Y = g(X), g bijective
        self.base, self.g, self.g_inv = base, g, g_inv

    def log_prob(self, y):
        ## y: (n,) one point; pull back to its unique preimage
        x = self.g_inv(y)
        J = jacobian(self.g, x)                          ## (n, n) = dg/dx at x
        ## mass conservation: divide density by local volume stretch |det J|
        return self.base.log_prob(x) - torch.linalg.slogdet(J).logabsdet

## Example: X ~ N(0, I), Y = W X + b  ->  Y ~ N(b, W W^T); det W = 2
W = torch.tensor([[2.0, 0.0], [1.0, 1.0]])
b = torch.tensor([1.0, -1.0])
base = MultivariateNormal(torch.zeros(2), torch.eye(2))
T = TransformedDensity(base, lambda x: W @ x + b, lambda y: torch.linalg.solve(W, y - b))
y = torch.tensor([0.3, 0.7])
print(T.log_prob(y))                                 ## tensor(-4.6935)
print(MultivariateNormal(b, W @ W.T).log_prob(y))    ## tensor(-4.6935), same
```
````

```{attention} Q&A
:class: dropdown
*Why $|\det J|$, not $\det J$?*

- Density ≥ 0. A negative determinant only means $g$ flips orientation.

*Why does a PMF need no Jacobian?*

- Discrete mass sits on points, not volumes → no stretching.
- $P(Y=y)=\sum_{x:\,g(x)=y}P(X=x)$.

*Samples vs densities?*

- Samples transform directly: $y=g(x)$. No Jacobian.
- Expectations need no Jacobian either: $E[h(Y)]=E[h(g(X))]$.
- Only the density of $Y$ needs the Jacobian. See [Marginalization & Reparameterization](bayes.md#marginalization-reparameterization).

*Is the mode invariant under reparameterization?*

- No. Density = per-volume quantity, not a function value → nonlinear $g$ moves the mode (Example: $e^{-1}\neq e^0$).
- → MAP estimates depend on the parameterization.
- Mean also moves: $E[g(X)]\neq g(E[X])$ in general; equality for affine $g$.

*What if $g$ changes dimension, e.g., $Y=X_1+X_2$?*

- Augment to a bijection $(X_1,X_1+X_2)$ → Jacobian det = 1 → marginalize the extra coordinate.
- $X_1\perp X_2$ → convolution: $f_Y(y)=\int f_{X_1}(x)f_{X_2}(y-x)\,dx$.

*How do normalizing flows use it?*

- Stack invertible layers $g_\ell$ w/ cheap $\log|\det J_{g_\ell}|$ (e.g., triangular Jacobian → det = product of the diagonal).
- → exact log-likelihood → train by MLE.
- Continuous-time limit: log-det sum → integral of a trace. See [Continuous Normalizing Flow](../diffeq.md#continuous-normalizing-flow).

*How does the reparameterization trick relate?*

- Write $z=g_\phi(\varepsilon)$, e.g., $z=m_\phi+s_\phi\varepsilon$ w/ $\varepsilon\sim\mathcal{N}(0,1)$.
- → $\nabla_\phi E_{z\sim q_\phi}[h(z)]=E_\varepsilon[\nabla_\phi h(g_\phi(\varepsilon))]$ → low-variance pathwise gradients.
- Uses only the pushforward of samples. The Jacobian appears only if $\log q_\phi(z)$ itself is needed.
- $m_\phi$, $s_\phi$: mean & std output by a network w/ params $\phi$.

*How to sample any 1-D distribution from a uniform?*

- Inverse CDF: $U\sim\text{Uniform}(0,1)$ → $X=F^{-1}(U)$ has CDF $F$.
    - $F^{-1}(u)=\inf\{x:F(x)\geq u\}$: generalized inverse → works for discrete $F$ too.
- Reverse: continuous $F$ → $F(X)\sim\text{Uniform}(0,1)$ (probability integral transform).
```

&nbsp;

### Expected Value / Mean
- **What**: Average outcome if repeating a random process multiple times (i.e., weigh each possible outcome by its likelihood).
$$\begin{align*}
&\text{Discrete:}   && E[X]=\sum_i x_ip_i \\
&\text{Continuous:} && E[X]=\int_{-\infty}^{\infty}xf(x)dx
\end{align*}$$
- **How**:

| Property                             | Formula                          |
| ------------------------------------ | ---------------------------------- |
| Linearity *(no independence needed)* | $E[aX+bY+c]=aE[X]+bE[Y]+c$         |
| Monotonicity                         | $X\leq Y\rightarrow E[X]\leq E[Y]$ |
| Product rule *(independence)*        | $E[XY]=E[X]E[Y]$                   |
| Law of total expectation             | $E[X]=E[E[X\|Y]]$                   |

&nbsp;

### Variance
- **What**: Average squared deviation between outcomes and mean.
$$\begin{align*}
&\text{Discrete:}   && Var[X]=E[(X-\mu)^2]=\sum_i (x_i-\mu)^2p_i \\
&\text{Continuous:} && Var[X]=E[(X-\mu)^2]=\int_{-\infty}^{\infty}(x-\mu)^2f(x)dx
\end{align*}$$
- **Why**:
    - *Why squared?*
        - Mathematically convenient (see **How**).
        - Smooth/Differentiable.
        - Punishes big surprises more.
            - BUT WHY CARE?
            - 1) Big misses are often disproportionately bad.
            - 2) Big surprises dominate what we care about, in various contexts.
    - *Why care?*
        - It tells us how **unreliable** the mean is.
- **How**:

| Property                         | Formula                          |
| -------------------------------- | ---------------------------------- |
| Scale                            | $Var[aX]=a^2 Var[X]$               |
| Shift                            | $Var[X+c]=Var[X]$                  |
| Sum                              | $Var[X+Y]=Var[X]+Var[Y]+2Cov[X,Y]$ |
| Relationship w/ Mean | $Var[X]=E[X^2]-E[X]^2$             |
| Law of total variance            | $Var[X]=E[Var[X\|Y]]+Var[E[X\|Y]]$   |
| Chebyshev Inequality             | $\forall\lambda>0: P(\|X-\mu\|\geq \lambda\sigma)\leq\frac{1}{\lambda^2}$<br>It's very unlikely for $X$ to be farther from $\mu$ than a multiple of $\sigma$. |

&nbsp;

### Covariance & Correlation
- **What**: Expected product of 2 RVs' deviations from their means.
    - **Correlation** $\rho$: covariance / product of std devs → unit-free, in $[-1,1]$.
- **Why**: Linear co-movement in one number.
    - Variance of sums & linear combinations needs it.
- **How**:
    - $X$ & $Y$ on the same side of their means → product > 0 → $Cov>0$.
    - Opposite sides → product < 0 → $Cov<0$.
    - Divide by $\sigma_X\sigma_Y$ → remove units & scale → $\rho$.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{X}=(X_1,\ldots,X_n)^T$: Random vector.
- Misc:
    - $\sigma_X$: Std dev of $X$.
    - $\sigma_Y$: Std dev of $Y$.
    - $\rho_{X,Y}$: Pearson correlation.
    - $\Sigma\in\mathbb{R}^{n\times n}$: Covariance matrix of $\mathbf{X}$.
    - $Z$: Third RV.
    - $a$: Scalar constant.
    - $b$: Scalar constant.
    - $c$: Scalar constant.
    - $d$: Scalar constant.
    - $\mathbf{a}\in\mathbb{R}^n$: Constant vector.
    - $W\in\mathbb{R}^{k\times n}$: Constant matrix; $k$ = #outputs (overrides class idx).
    - $\mathbf{b}\in\mathbb{R}^k$: Constant vector.

Definition:

$$
\begin{align*}
Cov[X,Y]&=E[(X-E[X])(Y-E[Y])]=E[XY]-E[X]E[Y] \\
\rho_{X,Y}&=\frac{Cov[X,Y]}{\sigma_X\sigma_Y}\in[-1,1],\qquad\sigma_X,\sigma_Y>0 \\
\Sigma&=E\left[(\mathbf{X}-E[\mathbf{X}])(\mathbf{X}-E[\mathbf{X}])^T\right],\qquad\Sigma_{jj'}=Cov[X_j,X_{j'}]
\end{align*}
$$

Properties:
- Symmetric: $Cov[X,Y]=Cov[Y,X]$. $Cov[X,X]=Var[X]$.
- Bilinear: $Cov[aX+bY,Z]=aCov[X,Z]+bCov[Y,Z]$.
- Shift-invariant: $Cov[X+c,Y]=Cov[X,Y]$.
- Linear map: $Cov[W\mathbf{X}+\mathbf{b}]=W\Sigma W^T$.
- $\rho_{aX+b,cY+d}=\text{sign}(ac)\,\rho_{X,Y}$ for $a,c\neq 0$.
- $|\rho_{X,Y}|=1\Leftrightarrow Y=aX+b$ almost surely for some $a\neq 0$.
- $X\perp Y$ → $Cov[X,Y]=0$. ❌ Converse.

Linear combination:

$$
Var[\mathbf{a}^T\mathbf{X}]=\sum_{j,j'}a_ja_{j'}Cov[X_j,X_{j'}]=\mathbf{a}^T\Sigma\mathbf{a}\geq 0
$$

- → $\Sigma$ symmetric PSD.
```

```{tip} Derivation
:class: dropdown
*Why $E[XY]-E[X]E[Y]$?*

1. Expand: $E[XY-XE[Y]-E[X]Y+E[X]E[Y]]$.
2. Linearity: $E[XY]-E[X]E[Y]-E[X]E[Y]+E[X]E[Y]=E[XY]-E[X]E[Y]$.

*Why $|\rho|\leq 1$?*

1. Apply [Cauchy–Schwarz](limits.md#cauchy-schwarz-inequality) to the centered RVs $X-E[X]$ & $Y-E[Y]$:

$$
|Cov[X,Y]|=\left|E[(X-E[X])(Y-E[Y])]\right|\leq\sqrt{Var[X]}\sqrt{Var[Y]}=\sigma_X\sigma_Y
$$

2. Divide by $\sigma_X\sigma_Y$.
3. Equality ⇔ centered RVs linearly dependent ⇔ $Y=aX+b$ almost surely.

*Why is $\Sigma$ PSD?*

1. $\mathbf{a}^T\mathbf{X}-E[\mathbf{a}^T\mathbf{X}]=\mathbf{a}^T(\mathbf{X}-E[\mathbf{X}])$.
2. $Var[\mathbf{a}^T\mathbf{X}]=E\left[\mathbf{a}^T(\mathbf{X}-E[\mathbf{X}])(\mathbf{X}-E[\mathbf{X}])^T\mathbf{a}\right]=\mathbf{a}^T\Sigma\mathbf{a}$.
3. Variance ≥ 0 for every $\mathbf{a}$ → PSD.
```

```{note} Example
:class: dropdown
**Uncorrelated ≠ independent**: $X$ uniform on $\{-1,0,1\}$, $Y=X^2$.
- $E[X]=0$, $E[XY]=E[X^3]=0$ → $Cov[X,Y]=0$.
- $P(Y=0\mid X=0)=1\neq P(Y=0)=\frac{1}{3}$ → dependent. $Y$ is even a function of $X$.
- Same w/ $X\sim\mathcal{N}(0,1)$: $Cov[X,X^2]=E[X^3]=0$.
```

````{important} Code
:class: dropdown
```python
import numpy as np

class Covariance:
    def __call__(self, X):
        ## X: (m, n) samples; center each feature, then average outer products
        Xc = X - X.mean(axis=0)
        return Xc.T @ Xc / (len(X) - 1)        ## (n, n); m-1 -> unbiased (mean is estimated)

    def corr(self, X):
        S = self(X)
        s = np.sqrt(np.diag(S))
        return S / np.outer(s, s)              ## rho_jj' = Sigma_jj' / (sigma_j sigma_j')

## Example: x^2 is a function of x, yet uncorrelated w/ x
rng = np.random.default_rng(0)
x = rng.standard_normal(100_000)
X = np.stack([x, x**2, 2 * x + 1], axis=1)     ## (m, 3)
print(Covariance().corr(X).round(2))           ## [0,1] ~ 0; [0,2] = 1
## PSD; col 3 is linear in col 1 -> min eigenvalue ~ 0 (singular)
print(np.linalg.eigvalsh(Covariance()(X)).min() > -1e-9)   ## True
```
````

```{attention} Q&A
:class: dropdown
*Does uncorrelated imply independent?*

- No. $X$ & $X^2$ (Example).
- Exception: **jointly** Gaussian → uncorrelated ⇔ independent.
- Marginally Gaussian is not enough: $X\sim\mathcal{N}(0,1)$, random sign $S=\pm 1$ indep. of $X$, $Y=SX$ → $Y\sim\mathcal{N}(0,1)$, $Cov[X,Y]=E[S]E[X^2]=0$, but $|Y|=|X|$.

*What does $\rho$ measure?*

- Strength of the **linear** relation only.
- $\rho^2$ = fraction of $Var[Y]$ explained by the best linear predictor $aX+b$.
- Nonlinear dependence can give $\rho=0$.

*Covariance vs correlation?*

- $Cov$: units of $X$ × units of $Y$, scale-dependent.
- $\rho$: unit-free, invariant to positive affine rescaling of either RV.

*When is $\Sigma$ singular?*

- $\exists\mathbf{a}\neq\mathbf{0}$: $\mathbf{a}^T\Sigma\mathbf{a}=Var[\mathbf{a}^T\mathbf{X}]=0$ → $\mathbf{a}^T\mathbf{X}$ constant almost surely.
- → $\mathbf{X}$ lives on an affine subspace → no density on $\mathbb{R}^n$.

*Is correlation transitive?*

- No. $\rho_{X,Y}=\rho_{Y,Z}=0.5$ allows $\rho_{X,Z}=-0.5$ (correlation matrix still PSD).

*Pearson vs Spearman?*

- Spearman = Pearson on ranks → measures monotone (not only linear) association, robust to outliers.

*Does $Cov[X,Y]=0$ for independent RVs always hold?*

- Only w/ finite 2nd moments. e.g., Cauchy RVs: covariance undefined.

*Correlation vs causation?*

- A common cause $Z$ can correlate $X$ & $Y$ w/o any causal link between them.

*Where does ML use $\Sigma$?*

- [PCA](../../ml/unsupervised.md#pca): eigenvectors of $\Sigma$ = directions of max variance.
- Whitening ($\Sigma\succ0$): $\Sigma^{-1/2}(\mathbf{X}-E[\mathbf{X}])$ → identity covariance.
- Mahalanobis distance ($\Sigma\succ0$): $\sqrt{(\mathbf{x}-E[\mathbf{X}])^T\Sigma^{-1}(\mathbf{x}-E[\mathbf{X}])}$.
```

&nbsp;

### Conditional Expectation
- **What**: Mean of $Y$ under $f(y\mid x)$, as a function of $X$.
    - $E[Y\mid X=x]=h(x)$: a number.
    - $E[Y\mid X]=h(X)$: a RV.
- **Why**: Best prediction of $Y$ from $X$.
    - Also: compute hard expectations in stages (condition first, then average).
- **How**:
    1. Fix $X=x$ → conditional distribution $f(y\mid x)$.
    2. Take its mean → $h(x)$.
    3. Plug the RV back in → $E[Y\mid X]=h(X)$.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $h(x)=E[Y\mid X=x]$: Regression function.
    - $g$: Any function w/ $E[g(X)^2]<\infty$.
    - $\mathcal{G}\subseteq\mathcal{F}$: Sub-σ-algebra (partial info).

Definition:

$$
\begin{align*}
&\text{Discrete:}   && E[Y\mid X=x]=\sum_yyf(y\mid x) \\
&\text{Continuous:} && E[Y\mid X=x]=\int_{-\infty}^{\infty}yf(y\mid x)\,dy
\end{align*}
$$

- General ($E[|Y|]<\infty$): $E[Y\mid\mathcal{G}]$ = the $\mathcal{G}$-measurable RV w/ $E[E[Y\mid\mathcal{G}]\mathbb{1}_G]=E[Y\mathbb{1}_G]$ $\forall G\in\mathcal{G}$. Unique almost surely.
- $E[Y\mid X]=E[Y\mid\sigma(X)]$; $\sigma(X)$: σ-algebra generated by $X$.

Properties:
- Tower: $E[E[Y\mid X]]=E[Y]$.
- Take out what is known: $E[g(X)Y\mid X]=g(X)E[Y\mid X]$.
- Independence: $X\perp Y$ → $E[Y\mid X]=E[Y]$.
- Orthogonality: $E[(Y-E[Y\mid X])g(X)]=0$ $\forall g$ w/ $E[g(X)^2]<\infty$ ($E[Y^2]<\infty$).

MSE-optimal ($E[Y^2]<\infty$):

$$
E[Y\mid X]=\arg\min_gE[(Y-g(X))^2]
$$

Law of total variance:

$$
Var[Y]=E[Var[Y\mid X]]+Var[E[Y\mid X]],\qquad Var[Y\mid X]=E[(Y-E[Y\mid X])^2\mid X]
$$
```

```{tip} Derivation
:class: dropdown
*Tower (continuous):*

$$
E[h(X)]=\int h(x)f_X(x)\,dx=\iint yf(y\mid x)f_X(x)\,dy\,dx=\iint yf(x,y)\,dx\,dy=E[Y]
$$

*Orthogonality:*

1. Tower: $E[(Y-h(X))g(X)]=E\left[E[(Y-h(X))g(X)\mid X]\right]$.
2. Take out $g(X)$: $=E\left[g(X)\left(E[Y\mid X]-h(X)\right)\right]=E[g(X)\cdot 0]=0$.

*MSE-optimal:*

1. Add & subtract $h(X)$:

$$
E[(Y-g(X))^2]=E[(Y-h(X))^2]+2E[(Y-h(X))(h(X)-g(X))]+E[(h(X)-g(X))^2]
$$

2. $h(X)-g(X)$ is a function of $X$ → cross term = 0 (orthogonality).
3. → $E[(Y-g(X))^2]=E[(Y-h(X))^2]+E[(h(X)-g(X))^2]\geq E[(Y-h(X))^2]$.
4. Equality ⇔ $g(X)=h(X)$ almost surely.

*Law of total variance:*

1. Add & subtract $h(X)$; $E[h(X)]=E[Y]$ by tower:

$$
Var[Y]=E[(Y-h(X))^2]+E[(h(X)-E[Y])^2]+2E[(Y-h(X))(h(X)-E[Y])]
$$

2. $h(X)-E[Y]$ is a function of $X$ → cross term = 0 (orthogonality).
3. 1st term: tower → $E\left[E[(Y-h(X))^2\mid X]\right]=E[Var[Y\mid X]]$.
4. 2nd term: $E[(h(X)-E[h(X)])^2]=Var[E[Y\mid X]]$.
```

```{note} Example
:class: dropdown
$X\in\{0,1\}$ w/ prob $\frac{1}{2}$ each. $Y\mid X=0\sim\mathcal{N}(0,1)$. $Y\mid X=1\sim\mathcal{N}(2,1)$.
- $h(0)=0$, $h(1)=2$ → $E[Y\mid X]=2X$.
- Tower: $E[Y]=E[2X]=1$.
- Within-group: $E[Var[Y\mid X]]=1$.
- Between-group: $Var[E[Y\mid X]]=Var[2X]=4\cdot\frac{1}{4}=1$.
- → $Var[Y]=2$. Check: $E[Y^2]=\frac{1}{2}(1)+\frac{1}{2}(1+4)=3$ → $Var[Y]=3-1^2=2$ ✅.
```

```{attention} Q&A
:class: dropdown
*$E[Y\mid X=x]$ vs $E[Y\mid X]$?*

- $E[Y\mid X=x]$: a number for each $x$.
- $E[Y\mid X]$: a RV, the function $h$ applied to $X$.

*Why does MSE training learn $E[Y\mid X]$?*

- Population [MSE](../../ml/obj.md#mse) is minimized by $h(X)=E[Y\mid X]$ (Derivation).
- → Flexible model + enough data + MSE → approximates the conditional mean.
- Other losses → other targets: absolute error → conditional median; cross-entropy → full $p(y\mid x)$.

*What is the geometric picture?*

- $E[Y\mid X]$ = orthogonal projection of $Y$ onto the space of square-integrable functions of $X$.
- Residual ⊥ every $g(X)$ → orthogonality.
- Linear regression projects onto a smaller subspace (linear functions) → never beats $E[Y\mid X]$; ties if $E[Y\mid X]$ is affine (e.g., jointly Gaussian).

*Residual orthogonal to $X$ → residual independent of $X$?*

- No. Only $E[Y-h(X)\mid X]=0$.
- Residual spread can depend on $X$ (heteroskedastic noise).

*What does the law of total variance say?*

- Total = unexplained (avg within-group) + explained (between-group).
- → $E[Var[Y\mid X]]\leq Var[Y]$: conditioning reduces variance on average.
- Basis of [Rao–Blackwellization](sampling.md#rao-blackwellization): replace an estimator by its conditional expectation → same mean, ≤ variance.

*Where else does it appear?*

- [Martingale](process.md#martingale): $E[X_{t+1}\mid\mathcal{F}_t]=X_t$ for process $X_t$ & filtration $\mathcal{F}_t$.
- Bellman equation: value = conditional expectation of return given the state.
- EM E-step: expected complete-data log-likelihood given the observed data.
```

&nbsp;

### Moment
- **What**: Mean of a power of a random variable.
- **Why**:
    - *Why was moment created?*
        - Moments let us compress an entire probability distribution into a few meaningful numbers
        - They can summarize the shape quickly:
            - 1st moment = average value
            - 2nd moment = spread
            - 2nd central moment = variance
            - 3rd central moment = skewness (left/right)
            - 4th central moment = kurtosis (tail/heaviness)
        - They are super useful for approximations for complicated distributions.
        - They connect directly to prediction & risk.
    - *Why the name "moment"?*
        - In physics, "moment" measures how mass is distributed.
        - In probability, "moment" measures how probability mass is distributed.
- **How**:
    - $k$th Moment: $\mu_k=E[X^k]$.
    - $k$th Central moment: $\mu_k=E[(X-E[X])^k]$.

&nbsp;

### Moment Generating Function
- **What**: Avg exponential weighting of a random variable.
    - → A tail-sensitive fingerprint that lets us read off moments by differentiating at 0.
$$
M_X(t)=E[e^{tX}]
$$
    - $t$: A knob that changes what values of $X$ we emphasize.
    - $t>0$ → $M_X(t)$ grows fast for large $X$ → Highly sensitive to the **right tail**.
    - $t<0$ → $M_X(t)$ shrinks large $X$ → Highly sensitive to the **left tail**.
    - "If we weigh outcomes exponentially toward big/small values, what avg weight do we get?"
- **Why**: A compression of a random variable's distribution into one smooth function.
    - *Why exponential?*
        - Perfect sum of independent vars: $e^{t(X+Y)}=e^{tX}e^{tY}$
            - Who cares?
            - ← Adding random vars is usually very hard.
            - ← Multiplying functions is usually much easier.
            - "Hard convolution" → "Easy multiplication".
        - Derivatives create powers of $X$:
            - $\frac{d}{dt}e^{tX}=Xe^{tX}$
            - $\frac{d^2}{dt^2}e^{tX}=X^2e^{tX}$
            - ...
    - *Why "moment generating"?*
        - Try differentiate at $t=0$:
            - $M'_X(0)=E[X]$
            - $M''_X(0)=E[X^2]$
            - ...
    - *Why MGF sometimes doesn't exist?*
        - If a distribution has heavy tails, $e^{tX}$ can explode.
        - → $E[e^{tX}]\rightarrow\infty$
- **How**:

| Property             | Formula                                  |
| -------------------- | ---------------------------------------- |
| Normalization        | $M_X(0)=1$                               |
| Moment generation    | $M_X^{(k)}(0)=E[X^k]$                    |
| Linearity            | $Y=aX+b\rightarrow M_Y(t)=e^{bt}M_X(at)$ |
| Sum (*independence*) | $M_{\sum X_i}(t)=\prod_iM_{X_i}(t)$      |
| Convexity            | $M_X(t)$ is convex in $t$                |

&nbsp;

### Characteristic Function
- **What**: Avg of a complex rotation of a random variable.
    - → A frequency fingerprint of the distribution that always exists.
$$
\varphi_X(t)=E[e^{itX}]
$$
    - $|e^{itX}|=1$ → CF spins around the unit circle in the complex plane.
        - Outcome $X=x$ is an arrow of length 1 at angle $tx$.
        - Expectation averages all arrows.
    - If distribution is very spread out → Arrows **point in many directions & cancel out** → $\varphi_X(t)$ small.
    - If distribution is very concentrated → Arrows **line up & don't cancel** → $\varphi_X(t)$ large.
    - "For each frequency $t$, how much does the distribution 'line up'?" (i.e., Fourier transform)
- **Why**:
    - *Why does a unit circle spin uniquely determine a distribution?*
        - CF = Fourier Transform:
$$
\varphi(t)=\int_{-\infty}^{\infty}e^{itx}f(x)dx
$$
        - Fourier Transform: "How much of each wave $e^{itx}$ is inside this function?"
        - Fourier Transform is **one-to-one** → Knowing all freq info is enough to reconstruct the original signal.
            - If we know the coefficient for EVERY freq $t$, we know the function's expansion in that wave-basis.
        - → Two diff distributions can't have the exact same coordinates in a basis.
    - *Why CF always exists?*
        - It's just an arrow on a unit circle. It never explodes or vanishes.
- **How**:

| Property                | Formula                                                                                    |
| ----------------------- | ------------------------------------------------------------------------------------------ |
| Normalization           | $\varphi_X(0)=1$                                                                               |
| Moment generation       | $\varphi_X^{(k)}(0)=i^kE[X^k]$                                                             |
| Linearity               | $Y=aX+b\rightarrow \varphi_Y(t)=e^{itb}\varphi_X(at)$                                      |
| Sum (*independence*)    | $\varphi_{\sum X_i}(t)=\prod_i\varphi_{X_i}(t)$                                            |
| Lévy's continuity theorem | $\forall t,\varphi_{X_n}(t)\rightarrow\varphi_X(t)\ \Longleftrightarrow\ X_n\Rightarrow X$ |
| Relationship w/ MGF   | $M_X(t)=\varphi_X(-it)$                                                                    |


&nbsp;
