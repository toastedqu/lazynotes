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
# Probability Theory
Study notes from {cite:t}`handbook_math` and {cite:t}`prob_lifesaver`.

## Basics
### Sample Space 
- **What**: Set of all possible outcomes of a random experiment ($\Omega$).
    - e.g., Coin flip: $\Omega=\{\text{head}, \text{tail}\}$.
    - e.g., Time till the universe ends: $\Omega=[0,\infty)$.

### Event
- **What**: A subset of the sample space ($A\subseteq\Omega$).
    - e.g., Coin flip is heads: $A=\{\text{head}\}$.
    - e.g., The universe ends within 60 seconds: $A=[0,60]$.

### σ-algebra
- **What**: A menu of allowed events in the sample space with 3 conditions ($\mathcal{F}$):
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
        - These 3 axioms are the minimal rules for "measuring uncertainty as a number" to behave consistently with ordinary logic about events AND to work for infinite processes.
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
| Continuity from above            | $A_1\subseteq A_2\subseteq\cdots\rightarrow P!\left(\bigcup_{i=1}^\infty A_i\right)=\lim_{i\rightarrow\infty}P(A_i)$ |
| Continuity from below            | $A_1\supseteq A_2\supseteq\cdots\rightarrow P!\left(\bigcap_{i=1}^\infty A_i\right)=\lim_{i\rightarrow\infty}P(A_i)$ |

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

### Probability Space
- **What**: $(\Omega,\mathcal{F},P)$
- **Why**:
    - We need a container that packages everything we need to do probability in a mathematically consistent way.
    - It specifies what random outcomes are allowed.
    - It specifies how likely each event is.
    - It provides the base for random variables ← ANY random variable is defined on this space.
    - It functions as the ground truth model for a random event. We can derive everything from it.