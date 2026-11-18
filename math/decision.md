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
# Decision & Game Theory
One rational agent → many interacting agents → designing the game → agents that others can predict.
Single-agent loss minimization under a posterior lives in [Bayesian decision theory](prob/bayes.md#bayesian-decision-theory).

Notations:
- $\mathcal{O}$: Outcome set.
- $o$: Outcome.
- $\Delta(\cdot)$: Set of probability distributions over a set; finite support unless stated.
- $p\in\Delta(\mathcal{O})$: Lottery.
- $\succsim$: Weak preference ("at least as good as").
- $\succ$: Strict preference.
- $\sim$: Indifference.
- $u$: Utility function.
- $a$: Action of a single decision-maker.
- $\mathcal{A}$: Action set of a single decision-maker.
- $V$: Realized utility, as a random variable.
- $e$: Observation history of a single decision-maker.
- $\mathcal{N}=\{1,\ldots,N\}$: Player set.
- $N$: #players.
- $i$: Player index.
- $-i$: All players except $i$.
- $\mathcal{S}_i$: Pure-strategy set of player $i$.
- $\mathcal{S}=\prod_{i}\mathcal{S}_i$: Set of pure-strategy profiles.
- $s=(s_i,s_{-i})\in\mathcal{S}$: Pure-strategy profile.
- $\sigma_i\in\Delta(\mathcal{S}_i)$: Mixed strategy of player $i$.
- $u_i$: vNM payoff of player $i$; extended to mixed profiles by expectation.
- $\text{BR}_i(\sigma_{-i})$: Best-response set of player $i$.
- $A$: Row player's payoff matrix in a 2-player game.
- $B$: Column player's payoff matrix in a 2-player game.
- $x$: Row player's mixed strategy as a probability vector.
- $y$: Column player's mixed strategy as a probability vector.
- $\theta_i\in\Theta_i$: Private type of player $i$.
- $\mathcal{C}$: Set of alternatives (candidates) in collective choice.
- $\succsim_i$: Ranking of voter/player $i$.

Overrides of global notation: $i$ indexes players, not samples. $x$, $y$ are mixed-strategy vectors, not input/output; they stay unbolded.

&nbsp;

## Rational Choice

### Preference Relation
- **What**: Binary relation on options; **rational** = complete & transitive.
- **Why**: Minimal consistency for choice: a best option exists on every finite set.
- **How**:
    1. Rank every pair (completeness).
    2. Chain rankings without cycles (transitivity).
    3. Represent by a utility function: higher utility ⟺ preferred.

```{note} Math
:class: dropdown
Definition (for all $o_1,o_2,o_3\in\mathcal{O}$):
- Completeness: $o_1\succsim o_2$ or $o_2\succsim o_1$.
- Transitivity: $o_1\succsim o_2$ & $o_2\succsim o_3$ → $o_1\succsim o_3$.
- $o_1\succ o_2$ ⟺ $o_1\succsim o_2$ & not $o_2\succsim o_1$.
- $o_1\sim o_2$ ⟺ $o_1\succsim o_2$ & $o_2\succsim o_1$.

Ordinal representation:

$$
o_1\succsim o_2\iff u(o_1)\geq u(o_2)
$$

- Finite or countable $\mathcal{O}$ + rational → $u$ exists.
- Uncountable $\mathcal{O}\subseteq\mathbb{R}^n$: rational + continuous (closed upper & lower contour sets) → continuous $u$ exists (Debreu).
- $u$ is unique up to strictly increasing transforms → only the order carries meaning.
```

```{attention} Q&A
:class: dropdown
*Why transitivity?*

- Money pump: cycle $o_1\succ o_2\succ o_3\succ o_1$ → agent pays a small fee for each "upgrade" → loops forever → sure loss.

*Is completeness innocent?*

- No. It is the most contested axiom: real agents often cannot rank some pairs.
- Incomplete preferences → no single $u$; under regularity conditions → a *set* of utility functions that must all agree.

*Does every rational preference have a utility function?*

- No. Lexicographic preferences on $\mathbb{R}^2$ are complete & transitive but have no real-valued $u$ ← not continuous.

*Does "rational" mean selfish?*

- No. Rationality = consistency only. Altruistic preferences can be rational.
```

&nbsp;

### Expected Utility
- **What**: Probability-weighted average of outcome utilities.
- **Why**: Rank risky prospects (lotteries), not only sure outcomes.
    - Expected *money* fails ← St. Petersburg paradox: infinite expected payoff, but people pay only a small finite price to play.
- **How**: Assign utility to each outcome → weight by probability → sum → choose the max.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\mathcal{P}\subseteq\Delta(\mathcal{O})$: Feasible lotteries.
    - $\alpha\in[0,1]$: Mixing weight.

Definition:

$$
\mathbb{E}_p[u]=\sum_{o\in\mathcal{O}}p(o)\,u(o),
\qquad
p^*\in\arg\max_{p\in\mathcal{P}}\mathbb{E}_p[u]
$$

- Linear in probabilities: $\mathbb{E}_{\alpha p+(1-\alpha)p'}[u]=\alpha\mathbb{E}_p[u]+(1-\alpha)\mathbb{E}_{p'}[u]$.
    - $\alpha p+(1-\alpha)p'$: Mixture lottery; outcome $o$ has probability $\alpha p(o)+(1-\alpha)p'(o)$.
```

```{note} Example
:class: dropdown
St. Petersburg game: flip a fair coin until heads; heads on flip $t$ pays $2^t$.

- Expected money: $\sum_{t\geq1}2^{-t}\cdot2^t=\sum_{t\geq1}1=\infty$.
- Log utility: $\sum_{t\geq1}2^{-t}\cdot t\ln2=2\ln2$ → certainty equivalent $e^{2\ln2}=4$.
- → Concave $u$ turns an infinite-value game into a game worth $4$.
```

```{attention} Q&A
:class: dropdown
*Expected utility vs expected value?*

- Expected value = expected utility with linear $u$ → risk neutral.

*Does log utility fully fix St. Petersburg?*

- No. Raise payoffs to $e^{2^t}$ → log utility diverges again.
- Only bounded $u$ blocks all such games. Savage's axioms imply bounded $u$.

*What breaks with unbounded utility?*

- Tiny probability × huge utility dominates every decision ("Pascal's mugging") → choices driven by the least reliable estimates.
- AI angle: an agent with unbounded $u$ can be steered by implausible high-stakes claims.

*How does RL relate?*

- Expected return = expected utility with utility additive over time.
- [MDP](../rl/intro.md#mdp) = sequential EU maximization with Markov state.
- [Bandits](../rl/intro.md#bandits) = EU maximization when outcome distributions must be learned.
```

&nbsp;

### vNM Utility Theorem
- **Name**: von Neumann–Morgenstern Utility Theorem {cite:p}`vonneumann1947theory`
- **What**: Preferences over lotteries satisfy 4 axioms ⟺ they maximize expected utility.
- **Why**: Justifies EU as *the* consistent rule for choice under known risk.
    - Utility becomes **cardinal**: unique up to positive affine transforms.
- **How**:
    1. Fix the best outcome $\bar o$ & worst outcome $\underline o$.
    2. For each outcome $o$, find $\alpha_o$ s.t. $o\sim$ ($\bar o$ w.p. $\alpha_o$, $\underline o$ else).
    3. Set $u(o)=\alpha_o$.
    4. Independence → any lottery is worth its expected $\alpha$.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $q$: Second lottery.
    - $r$: Third lottery.

Axioms on $\succsim$ over $\Delta(\mathcal{O})$, $\mathcal{O}$ finite:
1. **Completeness**.
2. **Transitivity**.
3. **Continuity**: $p\succsim q\succsim r$ → $\exists\alpha\in[0,1]$: $\alpha p+(1-\alpha)r\sim q$.
4. **Independence**: $\forall r,\ \forall\alpha\in(0,1)$: $p\succsim q\iff\alpha p+(1-\alpha)r\succsim\alpha q+(1-\alpha)r$.

Theorem:

$$
\text{1–4 hold}\iff\exists u:\mathcal{O}\to\mathbb{R}\ \text{s.t.}\ p\succsim q\iff\mathbb{E}_p[u]\geq\mathbb{E}_q[u]
$$

Uniqueness: $u'$ also represents $\succsim$ ⟺ $u'=cu+d$, $c>0$.
```

```{tip} Derivation
:class: dropdown
*Why do the axioms force EU? (sketch, finite $\mathcal{O}$, ⟹ direction)*

1. Let $\bar o$ / $\underline o$ be a best / worst sure outcome. If $\bar o\sim\underline o$, every lottery is indifferent → constant $u$.
2. Write $L_\alpha=\alpha\bar o+(1-\alpha)\underline o$. Independence → $\alpha>\beta$ ⟹ $L_\alpha\succ L_\beta$.
3. Every $p$ satisfies $\bar o\succsim p\succsim\underline o$. Continuity → some $\alpha_p$ with $p\sim L_{\alpha_p}$. Step 2 → $\alpha_p$ is unique.
4. Define $U(p)=\alpha_p$. Then $p\succsim q\iff U(p)\geq U(q)$.
5. Independence (applied twice, then reduce the compound lottery) → $U(\beta p+(1-\beta)q)=\beta U(p)+(1-\beta)U(q)$.
6. $p=\sum_o p(o)\delta_o$ → linearity gives $U(p)=\sum_o p(o)u(o)$ with $u(o)=U(\delta_o)$.
    - $\delta_o$: Sure lottery on $o$.
```

```{attention} Q&A
:class: dropdown
*Ordinal vs cardinal utility?*

- Ordinal ($u$ up to monotone transforms): only order.
- vNM ($u$ up to affine transforms): ratios of utility differences carry meaning.
- Neither allows comparing utility *across* people.

*Is vNM utility "happiness"?*

- No. $u$ only encodes choices over lotteries. It says nothing about felt intensity.

*Why independence?*

- A common component $r$ should not change how $p$ & $q$ compare.
- Violations + standard sequential-choice assumptions (consequentialism, compound-lottery reduction) → dynamically inconsistent plans → exploitable. Contested: dropping either assumption blocks the money pump.
- Most-violated axiom in experiments → see [Allais](#allais-paradox).

*What does vNM NOT give?*

- Probabilities: vNM assumes them → [Savage](#savages-framework) derives them.
- Values: any $u$ is allowed; the theorem constrains consistency, not content.

*Does vNM imply that a capable AI is a goal-directed EU maximizer?*

- Contested.
- For: an agent that cannot be money-pumped over lotteries acts *as if* it maximizes some $u$.
- Against: over full trajectories, *any* behavior maximizes some $u$ → the claim is empty unless $\mathcal{O}$ is restricted to coarse world states.
```

&nbsp;

#### Allais Paradox
- **What**: Common choice pattern that violates the independence axiom {cite:p}`allais1953comportement`.
- **Why**: Shows the **certainty effect**: people overweight sure outcomes.
- **How**: Offer two choice pairs that differ only by a common consequence → many people reverse their preference.

```{note} Example
:class: dropdown
Classic version, prizes in \$M:

| Choice | Option A | Option B |
|:--|:--|:--|
| 1 | $1$ w.p. $1$ | $1$ w.p. $0.89$; $5$ w.p. $0.10$; $0$ w.p. $0.01$ |
| 2 | $1$ w.p. $0.11$; $0$ w.p. $0.89$ | $5$ w.p. $0.10$; $0$ w.p. $0.90$ |

- Common pattern: 1A ≻ 1B & 2B ≻ 2A.
- Pair 2 = pair 1 with the common "$1$ w.p. $0.89$" replaced by "$0$ w.p. $0.89$".
```

```{tip} Derivation
:class: dropdown
*Why can no $u$ fit the common pattern?*

1. 1A ≻ 1B ⟺ $u(1)>0.89u(1)+0.10u(5)+0.01u(0)$ ⟺ $0.11u(1)>0.10u(5)+0.01u(0)$.
2. 2B ≻ 2A ⟺ $0.10u(5)+0.90u(0)>0.11u(1)+0.89u(0)$ ⟺ $0.10u(5)+0.01u(0)>0.11u(1)$.
3. Lines 1 & 2 contradict for every $u$.
```

```{attention} Q&A
:class: dropdown
*Irrational choice or wrong axiom?*

- Contested. Savage reported that he first chose the Allais pattern, then revised it after reflection.
- Normative view: independence is right; people err.
- Descriptive view: models with probability weighting (rank-dependent utility, prospect theory) fit the data better.
```

&nbsp;

### Risk Attitudes
- **What**: Curvature of utility over money: concave = risk-averse, linear = risk-neutral, convex = risk-seeking.
- **Why**: Lotteries with equal mean but different spread → agents differ in how much they pay to avoid spread.
- **How**: Compare a lottery with its mean.
    - Concave $u$ + Jensen → $\mathbb{E}[u(W)]\leq u(\mathbb{E}[W])$ → sure mean preferred.
    - Measure by **certainty equivalent** & **risk premium**.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $W$: Random monetary wealth.
    - $w$: Wealth level.
    - $\bar w=\mathbb{E}[W]$: Mean wealth.

Definition:

$$
u(\text{CE})=\mathbb{E}[u(W)],
\qquad
\text{RP}=\bar w-\text{CE}
$$

- $\text{CE}$: Certainty equivalent.
- $\text{RP}$: Risk premium. $\text{RP}>0$ ⟺ risk-averse for that lottery.

Arrow–Pratt measures:

$$
r_A(w)=-\frac{u''(w)}{u'(w)},
\qquad
r_R(w)=w\,r_A(w)
$$

- $r_A$: Absolute risk aversion; invariant to affine transforms of $u$.
- $r_R$: Relative risk aversion.

Small risks: $\text{RP}\approx\tfrac12 r_A(\bar w)\operatorname{Var}(W)$.
```

```{tip} Derivation
:class: dropdown
*Why $\text{RP}\approx\tfrac12 r_A\operatorname{Var}(W)$?*

1. Write $W=\bar w+\epsilon$, $\mathbb{E}[\epsilon]=0$, $\operatorname{Var}(\epsilon)=v$ small.
2. Left side: $u(\bar w-\text{RP})\approx u(\bar w)-\text{RP}\,u'(\bar w)$.
3. Right side: $\mathbb{E}[u(\bar w+\epsilon)]\approx u(\bar w)+\tfrac12 v\,u''(\bar w)$.
4. Equate → $\text{RP}\approx-\tfrac12 v\,u''(\bar w)/u'(\bar w)$.
```

````{important} Code
:class: dropdown
```python
import numpy as np

class CertaintyEquivalent:
    ## CRRA utility; gamma = 1 -> log utility
    def __init__(self, gamma):
        self.gamma = gamma

    def u(self, w):
        return np.log(w) if self.gamma == 1 else w ** (1 - self.gamma) / (1 - self.gamma)

    def u_inv(self, v):
        return np.exp(v) if self.gamma == 1 else ((1 - self.gamma) * v) ** (1 / (1 - self.gamma))

    def __call__(self, outcomes, probs):
        w, p = np.asarray(outcomes, float), np.asarray(probs, float)
        eu = p @ self.u(w)
        ce = self.u_inv(eu)        ## sure amount with the same utility as the lottery
        return ce, p @ w - ce      ## (certainty equivalent, risk premium)

## Example: 50/50 lottery over 50 or 150
print(CertaintyEquivalent(gamma=1)([50, 150], [0.5, 0.5]))  ## CE ≈ 86.60, RP ≈ 13.40
```
````

```{dropdown} Table: Standard Utility Families
| Family | $u(w)$ | $r_A(w)$ | $r_R(w)$ | Note |
|:--|:--|:--|:--|:--|
| Linear | $w$ | $0$ | $0$ | Risk-neutral |
| CARA | $-e^{-cw}$ | $c$ | $cw$ | Gaussian $W$ → $\text{CE}=\bar w-\tfrac{c}{2}\operatorname{Var}(W)$ exactly |
| CRRA | $\frac{w^{1-\gamma}}{1-\gamma}$ ($\gamma\neq1$) | $\gamma/w$ | $\gamma$ | $w>0$ |
| Log | $\ln w$ | $1/w$ | $1$ | CRRA at $\gamma=1$ |

- $c>0$: CARA coefficient.
- $\gamma>0$: CRRA coefficient.
```

```{attention} Q&A
:class: dropdown
*Is risk aversion irrational?*

- No. vNM allows any $u$; concave $u$ is consistent.

*Risk vs uncertainty?*

- Risk: known probabilities → vNM.
- Uncertainty (ambiguity): unknown probabilities → [Savage](#savages-framework), [Ellsberg](#ellsberg-paradox).

*Why log utility in betting?*

- Maximizing $\mathbb{E}[\ln W]$ per bet = Kelly criterion → maximizes the long-run growth rate of wealth.

*Is a CVaR objective a risk-averse EU?*

- No. CVaR weights outcomes by their rank, not through $u$ → violates independence → outside EU.
```

&nbsp;

### Savage's Framework
- **What**: Axioms on preferences over acts → subjective probability *and* utility, derived jointly {cite:p}`savage1954foundations`.
- **Why**: vNM assumes known probabilities; real decisions rarely come with them.
    - → Probability = summary of betting behavior, not an assumption.
- **How**:
    1. Model states $\omega$ (unknown), consequences $o$, acts $f:\Omega\to\mathcal{O}$.
    2. Impose axioms P1–P7 on $\succsim$ over acts.
    3. Read off a unique probability $P$ & a utility $u$ (unique up to affine transforms).
    4. Rank acts by subjective expected utility.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\Omega$: State space.
    - $\omega$: State.
    - $f,g:\Omega\to\mathcal{O}$: Acts.
    - $E\subseteq\Omega$: Event.
    - $fEh$: Act equal to $f$ on $E$ & to $h$ on $\Omega\setminus E$.
    - $P$: Subjective probability on events.

Representation:

$$
f\succsim g\iff\int_\Omega u(f(\omega))\,dP(\omega)\geq\int_\Omega u(g(\omega))\,dP(\omega)
$$

- $P$: Unique, finitely additive, can split any event into parts of every smaller probability.
- $u$: Bounded, unique up to positive affine transforms.

**Sure-thing principle** (P2):

$$
fEh\succsim gEh\iff fEh'\succsim gEh'
$$

- Preference between $f$ & $g$ on $E$ ignores what both give outside $E$.
```

```{dropdown} Table: Savage's Axioms
| Axiom | Content |
|:--|:--|
| P1 | $\succsim$ on acts is complete & transitive. |
| P2 | Sure-thing principle. |
| P3 | On a non-null event, ranking of constant consequences does not depend on the event. |
| P4 | Which of two events you bet on does not depend on the prizes used. → defines "more probable". |
| P5 | Some consequence is strictly preferred to another. |
| P6 | $f\succ g$ & any consequence $o$ → some finite partition exists s.t. setting either act to $o$ on any one cell keeps $f\succ g$. |
| P7 | Technical dominance condition for acts with infinitely many consequences. |
```

```{attention} Q&A
:class: dropdown
*vNM vs Savage?*

- vNM: probabilities given → derive $u$.
- Savage: derive $P$ & $u$ together from choices over acts.

*Hidden assumption?*

- States do not depend on the act. Acts are maps from states, so act choice cannot shift the state.
- Newcomb-type problems break this → [CDT](#cdt) vs [EDT](#edt) disagree.

*Why must $\Omega$ be infinite?*

- P6 forces fine partitions → $P$ takes every value in $[0,1]$ → $\Omega$ is infinite.

*Link to ML?*

- Bayesian agents = Savage agents with a likelihood model → [Bayesian decision theory](prob/bayes.md#bayesian-decision-theory).
```

&nbsp;

#### Ellsberg Paradox
- **What**: Ambiguity-averse choice pattern that violates the sure-thing principle {cite:p}`ellsberg1961risk`.
- **Why**: Shows that people treat unknown probabilities (ambiguity) differently from known ones (risk).
- **How**: Bet on an urn with partly unknown composition → choices fit no single probability.

```{note} Example
:class: dropdown
Urn: 90 balls. 30 red. 60 black or yellow, mix unknown. Prize \$100 if the named event occurs.

| Bet | Wins on | Typical choice |
|:--|:--|:--|
| I | Red | I ≻ II |
| II | Black | |
| III | Red or Yellow | IV ≻ III |
| IV | Black or Yellow | |

- I ≻ II → $P(\text{R})>P(\text{B})$.
- IV ≻ III → $P(\text{B})+P(\text{Y})>P(\text{R})+P(\text{Y})$ → $P(\text{B})>P(\text{R})$.
- Contradiction → no $P$ fits.
- Sure-thing view: I/II and III/IV differ only in the common payoff on Yellow ($0$ vs $100$).
```

```{attention} Q&A
:class: dropdown
*How to model ambiguity aversion?*

- Maxmin EU (Gilboa–Schmeidler): hold a *set* of priors → score each act by its worst-case EU.

*ML analogue?*

- Distributionally robust optimization = maxmin EU over an uncertainty set of data distributions.
```

&nbsp;

### Dutch Book Argument
- **What**: Credences that violate the probability axioms → a set of individually acceptable bets with a sure loss {cite:p}`ramsey1926truth,definetti1937prevision`.
- **Why**: Justifies subjective degrees of belief as probabilities without frequencies.
- **How**:
    1. Credence $c(E)$ = fair price of a ticket that pays $1$ if $E$.
    2. Agent buys or sells at that price.
    3. Incoherent $c$ → bookie picks the side of each ticket → net payoff $<0$ in every state.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\Omega$: Finite state space.
    - $E,F\subseteq\Omega$: Events.
    - $c(E)\in\mathbb{R}$: Credence = fair price for a ticket paying $1$ if $E$.
    - $\lambda_E\in\mathbb{R}$: Stake on $E$; $\lambda_E>0$ = agent buys, $\lambda_E<0$ = agent sells.

Agent's net payoff in state $\omega$:

$$
\sum_E\lambda_E\left(\mathbb{1}[\omega\in E]-c(E)\right)
$$

Theorem: no stakes give a negative payoff in every state ⟺ $c$ is a finitely additive probability:
- $c(E)\geq0$.
- $c(\Omega)=1$.
- $c(E\cup F)=c(E)+c(F)$ for disjoint $E,F$.
```

```{tip} Derivation
:class: dropdown
*Why must additivity hold? (⟸ direction for one axiom)*

1. Assume disjoint $E,F$ with $c(E\cup F)>c(E)+c(F)$.
2. Agent buys a ticket on $E\cup F$ & sells tickets on $E$ and on $F$.
3. Net: $\left(\mathbb{1}_{E\cup F}-c(E\cup F)\right)-\left(\mathbb{1}_E-c(E)\right)-\left(\mathbb{1}_F-c(F)\right)$.
4. Disjoint → $\mathbb{1}_{E\cup F}=\mathbb{1}_E+\mathbb{1}_F$ → indicators cancel.
5. Net $=c(E)+c(F)-c(E\cup F)<0$ in every state.
6. Reverse the inequality → reverse every side.
```

```{attention} Q&A
:class: dropdown
*Assumptions?*

- Utility linear in money at the stakes used.
- Agent accepts either side at the fair price.
- Bookie knows nothing the agent does not.

*Countable additivity?*

- Finite books → finite additivity only. de Finetti accepted only finite additivity.

*Updating?*

- A diachronic Dutch book supports updating by conditionalization. Whether it binds a rational agent is contested.

*Link to preferences?*

- Same logic as the money pump: coherence = no guaranteed exploitation.
```

&nbsp;

## Strategic Games

### Normal-Form Game
- **What**: Players, strategy sets & payoff functions; each player picks one complete strategy, timing & information suppressed.
- **Why**: Each payoff depends on others' choices → single-agent EU alone cannot pick an action.
- **How**:
    1. List each player's strategies.
    2. Assign a payoff vector to every profile (a matrix pair for 2 players).
    3. Allow randomization (mixed strategies); payoffs become EU.

```{note} Math
:class: dropdown
Definition:

$$
G=\left(\mathcal{N},(\mathcal{S}_i)_{i\in\mathcal{N}},(u_i)_{i\in\mathcal{N}}\right)
$$

Mixed payoff:

$$
u_i(\sigma)=\sum_{s\in\mathcal{S}}\Big(\prod_{j\in\mathcal{N}}\sigma_j(s_j)\Big)u_i(s)
$$

2 players ($A,B\in\mathbb{R}^{|\mathcal{S}_1|\times|\mathcal{S}_2|}$):

$$
u_1(x,y)=x^\top Ay,
\qquad
u_2(x,y)=x^\top By
$$

Best response:

$$
\text{BR}_i(\sigma_{-i})=\arg\max_{\sigma_i\in\Delta(\mathcal{S}_i)}u_i(\sigma_i,\sigma_{-i})
$$

- $u_i$ is linear in $\sigma_i$ → mixed best responses = mixtures of pure best responses.
```

````{important} Code
:class: dropdown
```python
import numpy as np

class NormalFormGame:
    def __init__(self, A, B):
        self.A = np.asarray(A, float)  ## row player's payoffs [m1, m2]
        self.B = np.asarray(B, float)  ## column player's payoffs [m1, m2]

    def payoffs(self, x, y):
        ## mixed strategies are probability vectors; payoffs are bilinear
        return float(x @ self.A @ y), float(x @ self.B @ y)

    def best_responses(self, x, y):
        ## pure best responses suffice: payoff is linear in one's own mix
        r, c = self.A @ y, x @ self.B
        return np.flatnonzero(np.isclose(r, r.max())), np.flatnonzero(np.isclose(c, c.max()))

## Example: Matching Pennies
mp = NormalFormGame([[1, -1], [-1, 1]], [[-1, 1], [1, -1]])
x = y = np.array([0.5, 0.5])
print(mp.payoffs(x, y))         ## (0.0, 0.0)
print(mp.best_responses(x, y))  ## (array([0, 1]), array([0, 1]))
```
````

```{attention} Q&A
:class: dropdown
*Payoff = money?*

- No. Payoffs are vNM utilities. Money payoffs need a stated risk attitude.

*Why mixed strategies?*

- Matching Pennies has no pure equilibrium → unpredictability has value.
- Interpretations: deliberate randomization; population frequencies; others' beliefs about you.

*What does the normal form lose?*

- Timing & information. Each extensive game has a normal form (strategy = complete plan), but credibility checks need the tree → [Extensive-Form Game](#extensive-form-game).
```

&nbsp;

### Dominance
- **What**: $s_i$ **strictly dominates** $s_i'$: $s_i$ is better against *every* opponent profile.
- **Why**: Weakest rationality test: a rational player never plays a strictly dominated strategy, whatever its beliefs.
- **How**:
    1. Delete strictly dominated strategies (IESDS: iterated elimination of strictly dominated strategies).
    2. Repeat on the reduced game ← common knowledge of rationality.
    3. One profile survives → **dominance-solvable**.

```{note} Math
:class: dropdown
Definitions:
- Strict: $u_i(s_i,s_{-i})>u_i(s_i',s_{-i})\ \ \forall s_{-i}$.
- Weak: $\geq$ for all $s_{-i}$, $>$ for some.
- A mixed $\sigma_i$ can dominate a pure $s_i'$ even when no pure strategy does.
- Dominant strategy: dominates every other own strategy.

Properties:
- IESDS result is order-independent in finite games.
- Iterated weak elimination is order-dependent & can delete Nash equilibria.
- Every Nash equilibrium survives IESDS.
- 2 players: never a best response ⟺ strictly dominated by a mixed strategy → IESDS (with mixed dominators) = **rationalizability**.
```

```{note} Example
:class: dropdown
Row payoffs against two column strategies: $s^1=(3,0)$, $s^2=(0,3)$, $s^3=(1,1)$.

- No pure strategy dominates $s^3$.
- 50/50 mix of $s^1,s^2$ yields $(1.5,1.5)>(1,1)$ → $s^3$ is strictly dominated.
```

````{important} Code
:class: dropdown
```python
import numpy as np

## Reuses NormalFormGame from Normal-Form Game above
class IESDS:
    ## pure-strategy dominators only; mixed dominators need an LP
    def __call__(self, game):
        rows, cols = list(range(game.A.shape[0])), list(range(game.A.shape[1]))
        while True:
            A, B = game.A[np.ix_(rows, cols)], game.B[np.ix_(rows, cols)]
            bad_r = [r for r in range(len(rows))
                     if any(np.all(A[k] > A[r]) for k in range(len(rows)) if k != r)]
            bad_c = [c for c in range(len(cols))
                     if any(np.all(B[:, k] > B[:, c]) for k in range(len(cols)) if k != c)]
            if not bad_r and not bad_c:
                return rows, cols
            ## remove all at once: strict-dominance elimination is order-independent
            rows = [s for k, s in enumerate(rows) if k not in bad_r]
            cols = [s for k, s in enumerate(cols) if k not in bad_c]

## Example: Prisoner's Dilemma; 0 = Cooperate, 1 = Defect
pd = NormalFormGame([[3, 0], [5, 1]], [[3, 5], [0, 1]])
print(IESDS()(pd))  ## ([1], [1])
```
````

```{attention} Q&A
:class: dropdown
*Dominant-strategy equilibrium vs Nash equilibrium?*

- Dominant: best whatever others do → no beliefs needed. Rarely exists.
- Nash: best given correct beliefs about others.
- Mechanism design targets dominant strategies → strategy-proofness.

*Why iterate only strict dominance?*

- Weak elimination depends on the order & can remove equilibria.

*Assumption behind iteration?*

- Common knowledge of rationality: I know you are rational, you know that I know, etc.
- Each extra round of deletion needs one more level of this knowledge.
```

&nbsp;

### Nash Equilibrium
- **What**: Profile where every player's strategy is a best response to the others' {cite:p}`nash1950equilibrium`.
- **Why**: Stable prediction: no player gains by a unilateral deviation.
    - = fixed point of mutual best responses → self-enforcing agreement.
- **How**:
    - **Pure**: find cells where the row payoff is a column max & the column payoff is a row max.
    - **Mixed**: each player mixes to make the *opponent* indifferent over the opponent's support.
    - **Existence**: every finite game has ≥1 mixed equilibrium.

```{note} Math
:class: dropdown
Definition:

$$
u_i(\sigma_i^*,\sigma_{-i}^*)\geq u_i(s_i,\sigma_{-i}^*)
\quad\forall i\in\mathcal{N},\ \forall s_i\in\mathcal{S}_i
$$

- Pure deviations suffice ← linearity in $\sigma_i$.

**Indifference principle**:

$$
s_i\in\operatorname{supp}(\sigma_i^*)
\implies
u_i(s_i,\sigma_{-i}^*)=\max_{s_i'\in\mathcal{S}_i}u_i(s_i',\sigma_{-i}^*)
$$

**Existence**: $\text{BR}(\sigma)=\prod_i\text{BR}_i(\sigma_{-i})$ on compact convex $\prod_i\Delta(\mathcal{S}_i)$.
- Nonempty, convex-valued, closed graph → Kakutani → $\exists\sigma^*\in\text{BR}(\sigma^*)$.
```

```{tip} Derivation
:class: dropdown
*Mixed equilibrium of a 2×2 game with full support:*

1. Let $y_0$ = probability column plays column 0. Row indifference:

    $$
    A_{00}y_0+A_{01}(1-y_0)=A_{10}y_0+A_{11}(1-y_0)
    \implies
    y_0=\frac{A_{11}-A_{01}}{A_{00}-A_{01}-A_{10}+A_{11}}
    $$

2. Let $x_0$ = probability row plays row 0. Column indifference:

    $$
    B_{00}x_0+B_{10}(1-x_0)=B_{01}x_0+B_{11}(1-x_0)
    \implies
    x_0=\frac{B_{11}-B_{10}}{B_{00}-B_{10}-B_{01}+B_{11}}
    $$

3. → The row's mix depends only on the *column's* payoffs, and vice versa.
4. Valid only if both denominators are nonzero & both values lie in $(0,1)$.
```

```{note} Example
:class: dropdown
Battle of the Sexes: $A=\begin{bmatrix}2&0\\0&1\end{bmatrix}$, $B=\begin{bmatrix}1&0\\0&2\end{bmatrix}$.

- Pure equilibria: (0,0) & (1,1).
- Mixed: $x_0=\frac{2-0}{1-0-0+2}=\frac23$, $y_0=\frac{1-0}{2-0-0+1}=\frac13$.
- Mixed payoffs: $\frac23$ each → worse than either pure equilibrium for both players.
```

````{important} Code
:class: dropdown
```python
import itertools
import numpy as np

## Reuses NormalFormGame from Normal-Form Game above
class SupportEnumeration:
    ## All Nash equilibria of a nondegenerate 2-player game (equal-size supports)
    def __init__(self, game, tol=1e-9):
        self.A, self.B, self.tol = game.A, game.B, tol

    def _mix(self, M, own, opp, size):
        ## opponent mix q on 'opp' that makes rows 'own' of M equally good: M q = v, sum q = 1
        k = len(own)
        lhs, rhs = np.zeros((k + 1, k + 1)), np.zeros(k + 1)
        lhs[:k, :k], lhs[:k, k] = M[np.ix_(own, opp)], -1.0
        lhs[k, :k], rhs[k] = 1.0, 1.0
        try:
            sol = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            return None
        q = np.zeros(size)
        q[list(opp)] = sol[:k]
        return q if np.all(q >= -self.tol) else None

    def __call__(self):
        m, n = self.A.shape
        for k in range(1, min(m, n) + 1):
            for I in itertools.combinations(range(m), k):
                for J in itertools.combinations(range(n), k):
                    y = self._mix(self.A, I, J, n)    ## column mix: row player indifferent on I
                    x = self._mix(self.B.T, J, I, m)  ## row mix: column player indifferent on J
                    if x is None or y is None:
                        continue
                    ## no profitable deviation outside the supports
                    if (self.A @ y).max() <= (self.A @ y)[list(I)].max() + self.tol and \
                       (x @ self.B).max() <= (x @ self.B)[list(J)].max() + self.tol:
                        yield x, y

## Example: Battle of the Sexes
bos = NormalFormGame([[2, 0], [0, 1]], [[1, 0], [0, 2]])
for x, y in SupportEnumeration(bos)():
    print(np.round(x, 3), np.round(y, 3))
## [1. 0.] [1. 0.]
## [0. 1.] [0. 1.]
## [0.667 0.333] [0.333 0.667]
```
````

```{attention} Q&A
:class: dropdown
*Pros?*

- Always exists in finite games.
- Necessary for any self-enforcing prediction.

*Cons?*

- Multiple equilibria → selection problem.
- Can be Pareto-inefficient → [Prisoner's Dilemma](#prisoners-dilemma).
- Assumes correct beliefs about others' strategies.
- Hard to compute: finding an (approximate) NE is PPAD-complete, even for 2 players {cite:p}`daskalakis2009complexity,chen2009settling`.
    - Exact NE with ≥3 players can be irrational → harder class (FIXP).

*Does play converge to Nash equilibrium?*

- Not in general. Simple learning dynamics (e.g., fictitious play, gradient play) can cycle.
- GAN training = search for an equilibrium of a 2-player game → oscillation is a known failure.

*Why does my mix depend on the opponent's payoffs?*

- My mix must make *the opponent* indifferent; otherwise the opponent plays a pure best response.

*Nash vs minimax?*

- Coincide in 2-player zero-sum games → [Minimax Theorem](#minimax-theorem).
```

&nbsp;

### Prisoner's Dilemma
- **What**: 2-player game where defection strictly dominates, but mutual cooperation is better for both than mutual defection.
- **Why**: Canonical gap between individual & collective rationality.
- **How**: Payoffs $T>R>P>S$ → Defect dominates → (D, D) is the unique equilibrium → both get $P<R$.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $T$: Temptation payoff (defect vs cooperator).
    - $R$: Reward payoff (mutual cooperation).
    - $P$: Punishment payoff (mutual defection).
    - $S$: Sucker payoff (cooperate vs defector).

Row payoffs (C = Cooperate, D = Defect):

$$
A=\begin{bmatrix}R&S\\T&P\end{bmatrix},
\qquad B=A^\top,
\qquad T>R>P>S
$$

- Repeated version often also assumes $2R>T+S$ → alternating exploitation is worse than steady cooperation.

**Pareto dominance**: $s$ Pareto-dominates $s'$ ⟺ $u_i(s)\geq u_i(s')\ \forall i$, strict for some $i$.
- (C, C) Pareto-dominates the unique equilibrium (D, D).
```

```{dropdown} Table: Canonical 2×2 Games
| Game | Structure | Pure equilibria | Lesson |
|:--|:--|:--|:--|
| Prisoner's Dilemma | $T>R>P>S$ | (D, D) | Dominance → inefficiency |
| Stag Hunt | $R>T\geq P>S$ | (C, C), (D, D) | Coordination: payoff-best vs safe |
| Chicken (Hawk–Dove) | $T>R>S>P$ | (C, D), (D, C) + 1 mixed | Anti-coordination; conflict over who yields |
| Battle of the Sexes | Both prefer agreeing; disagree on where | 2 asymmetric + 1 mixed | Coordination with conflict |
| Matching Pennies | Zero-sum | None; unique mixed $(\frac12,\frac12)$ | Unpredictability |

- Letters follow the PD notation: C = cooperate / yield, D = defect / dare.
```

```{attention} Q&A
:class: dropdown
*Is defecting rational?*

- One-shot, causally independent choices: yes, under dominance & [CDT](#cdt).
- Twin PD (opponent runs the same reasoning): [EDT](#edt) & [FDT](#fdt) cooperate. Lewis showed this PD is a Newcomb problem.

*How do agents escape?*

- Repetition → [Folk Theorem](#folk-theorem).
- Contracts & mechanisms that change payoffs.
- Changed preferences (altruism, reputation).
- Readable source code: programs that condition on each other's code reach mutual cooperation in one shot ("program equilibrium") {cite:p}`tennenholtz2004program`.

*Why does it matter for AI?*

- Multi-agent AI systems face PD-like races (e.g., safety vs speed). Copyable, inspectable agents make the program-equilibrium route realistic.
```

&nbsp;

### Zero-Sum Game
- **What**: Game where the payoffs sum to zero in every outcome.
- **Why**: Pure conflict → no coordination problem → strongest solution concept: a unique game **value**.
- **How**: Each player maximizes its worst-case payoff; the [Minimax Theorem](#minimax-theorem) makes both guarantees equal.

```{note} Math
:class: dropdown
Definition:

$$
u_2(s)=-u_1(s)\ \ \forall s\in\mathcal{S}
\qquad(B=-A)
$$

- Constant-sum games are equivalent (subtract the constant).
- Row = maximizer of $x^\top Ay$; column = minimizer.
```

```{attention} Q&A
:class: dropdown
*Why are zero-sum games easy compared with general-sum games?*

- Every equilibrium gives the same value.
- **Interchangeability**: any maximin $x$ & any minimax $y$ form an equilibrium → no selection problem.
- Equilibrium = LP → polynomial time.

*ML examples?*

- Original GAN objective, adversarial training ($\min_\theta\max_\delta$), self-play in Go/chess → [MCTS](../rl/model.md#mcts-with-uct), [AlphaZero](../rl/model.md#alphazero).
```

&nbsp;

#### Minimax Theorem
- **What**: Finite 2-player zero-sum game: $\max_x\min_y x^\top Ay=\min_y\max_x x^\top Ay$ over mixed strategies {cite:p}`vonneumann1928theorie`.
- **Why**: Revealing your *mixed* strategy (not its realized draw) costs nothing → the game has a well-defined value.
- **How**:
    1. Row picks $x$ to maximize its guarantee $\min_j(x^\top A)_j$; pure replies suffice.
    2. Column picks $y$ to minimize its worst loss $\max_k(Ay)_k$.
    3. The two problems are dual LPs → strong [LP duality](../ml/optim.md#duality) → equal optimal values.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $m$: #row strategies (overrides #samples).
    - $n$: #column strategies (overrides #features).
    - $\Delta_m$: Probability simplex in $\mathbb{R}^m$.
    - $v$: Game value.

Theorem:

$$
v=\max_{x\in\Delta_m}\min_{y\in\Delta_n}x^\top Ay=\min_{y\in\Delta_n}\max_{x\in\Delta_m}x^\top Ay
$$

Row LP & its dual (column LP):

$$
\begin{align*}
&\max_{x,v}\ v\quad\text{s.t.}\ A^\top x\geq v\mathbf{1},\ \mathbf{1}^\top x=1,\ x\geq0\\
&\min_{y,v}\ v\quad\text{s.t.}\ Ay\leq v\mathbf{1},\ \mathbf{1}^\top y=1,\ y\geq0
\end{align*}
$$

- Pure saddle point exists ⟺ $\max_k\min_j A_{kj}=\min_j\max_k A_{kj}$.
- Sion: equality also holds for $f(x,y)$ quasi-concave & upper semicontinuous in $x$, quasi-convex & lower semicontinuous in $y$, on convex sets with one of them compact.
```

```{tip} Derivation
:class: dropdown
*Why is $\max\min\leq\min\max$ always true?*

1. For any $x,y'$: $\min_y x^\top Ay\leq x^\top Ay'\leq\max_{x'}x'^\top Ay'$.
2. Left side ignores $y'$; right side ignores $x$.
3. Max over $x$ on the left, min over $y'$ on the right → $\max_x\min_y\leq\min_{y'}\max_{x'}$.
4. Equality needs convexity of the mixed-strategy sets + bilinearity → LP strong duality.
```

````{important} Code
:class: dropdown
```python
import numpy as np
from scipy.optimize import linprog

class ZeroSumSolver:
    ## row player's maximin strategy & game value via LP
    def __call__(self, A):
        A = np.asarray(A, float)
        m, n = A.shape
        c = np.r_[np.zeros(m), -1.0]               ## variables [x, v]; maximize v
        A_ub = np.c_[-A.T, np.ones(n)]             ## v <= (A^T x)_j: guarantee vs every pure reply
        A_eq = np.r_[np.ones(m), 0.0][None, :]     ## x sums to 1
        bounds = [(0, None)] * m + [(None, None)]
        res = linprog(c, A_ub=A_ub, b_ub=np.zeros(n), A_eq=A_eq, b_eq=[1.0], bounds=bounds)
        return res.x[:m], res.x[m]

## Example: Rock-Paper-Scissors
rps = [[0, -1, 1], [1, 0, -1], [-1, 1, 0]]
x, v = ZeroSumSolver()(rps)
print(np.round(x, 3), round(v, 3))  ## [0.333 0.333 0.333] 0.0 (up to solver round-off)
## column player's strategy: ZeroSumSolver()(-np.asarray(rps).T)
```
````

```{attention} Q&A
:class: dropdown
*Does minimax hold in general-sum games?*

- The guarantee $\max_x\min_y u_1$ exists, but maximin strategies are generally not equilibria.

*Does minimax hold for GANs?*

- Not guaranteed. NN objectives are not convex–concave → $\max\min\neq\min\max$ is possible → order of play matters.

*Game-tree link?*

- Perfect-information trees: pure minimax by backward induction (alpha-beta search). [MCTS](../rl/model.md#mcts-with-uct) approximates it by sampling.
```

&nbsp;

### Correlated Equilibrium
- **What**: Distribution over profiles s.t. obeying a private recommendation drawn from it is a best response {cite:p}`aumann1974subjectivity`.
- **Why**: Shared randomness (e.g., a traffic light) → in some games, welfare above every Nash equilibrium; always computable by LP.
- **How**:
    1. Mediator samples $s\sim\mu$.
    2. Mediator tells player $i$ only $s_i$.
    3. Given $s_i$, obeying beats every deviation $s_i'$ in expectation.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\mu\in\Delta(\mathcal{S})$: Joint distribution over profiles.

Correlated equilibrium (CE):

$$
\sum_{s_{-i}}\mu(s_i,s_{-i})\left[u_i(s_i,s_{-i})-u_i(s_i',s_{-i})\right]\geq0
\quad\forall i,\ \forall s_i,s_i'\in\mathcal{S}_i
$$

Coarse correlated equilibrium (CCE): deviation chosen *before* seeing the recommendation:

$$
\sum_{s}\mu(s)\left[u_i(s)-u_i(s_i',s_{-i})\right]\geq0
\quad\forall i,\ \forall s_i'\in\mathcal{S}_i
$$

Properties:
- Linear constraints in $\mu$ → CE set = convex polytope → welfare max = LP.
- Nash equilibrium = CE with $\mu=\prod_i\sigma_i$ → NE ⊆ CE ⊆ CCE.
```

````{important} Code
:class: dropdown
```python
import itertools
import numpy as np
from scipy.optimize import linprog

## Reuses NormalFormGame from Normal-Form Game above
class CorrelatedEquilibrium:
    ## welfare-maximizing CE of a 2-player game via LP over mu[r, c]
    def __init__(self, game):
        self.A, self.B = game.A, game.B

    def __call__(self):
        m, n = self.A.shape
        cons = []  ## each row g encodes g . mu >= 0
        for r, r2 in itertools.permutations(range(m), 2):
            g = np.zeros((m, n))
            g[r] = self.A[r] - self.A[r2]          ## row told r, considers r2
            cons.append(g.ravel())
        for c, c2 in itertools.permutations(range(n), 2):
            g = np.zeros((m, n))
            g[:, c] = self.B[:, c] - self.B[:, c2]  ## column told c, considers c2
            cons.append(g.ravel())
        welfare = (self.A + self.B).ravel()
        res = linprog(-welfare, A_ub=-np.array(cons), b_ub=np.zeros(len(cons)),
                      A_eq=np.ones((1, m * n)), b_eq=[1.0], bounds=[(0, None)] * (m * n))
        return res.x.reshape(m, n)

## Example: Chicken; 0 = Dare, 1 = Yield
chicken = NormalFormGame([[0, 7], [2, 6]], [[0, 2], [7, 6]])
print(np.round(CorrelatedEquilibrium(chicken)(), 3))  ## [[0, 0.25], [0.25, 0.5]] -> 5.25 each
```
````

```{note} Example
:class: dropdown
Chicken above (0 = Dare, 1 = Yield):

- Mixed NE: each dares w.p. $\frac13$ → payoff $\frac{14}{3}\approx4.67$ each.
- Best CE: (Dare, Yield) $\frac14$, (Yield, Dare) $\frac14$, (Yield, Yield) $\frac12$ → $5.25$ each.
- Told "Yield": opponent dares w.p. $\frac13$ → Yield $=\frac13\cdot2+\frac23\cdot6=\frac{14}{3}$ = Dare $=\frac23\cdot7$ → obeying is optimal.
```

```{attention} Q&A
:class: dropdown
*Why can CE beat every NE?*

- Correlation avoids the (Dare, Dare) crash.
- Private recommendations hide enough to keep deviation unprofitable.

*Can learning reach it?*

- Yes. No-internal-regret learning (e.g., regret matching) → empirical joint play converges to the CE set {cite:p}`hart2000simple`.
- No-external-regret learning → CCE set.
- Contrast: no comparable general guarantee for Nash equilibrium.

*Computation?*

- CE: LP, polynomial in the size of the normal form. Nash: PPAD-complete even for 2 players.
```

&nbsp;

## Dynamic & Incomplete-Information Games

### Extensive-Form Game
- **What**: Game tree: who moves when, what each knows, payoffs at leaves {cite:p}`kuhn1953extensive`.
- **Why**: The normal form hides timing & information → cannot separate credible from non-credible threats.
- **How**:
    - Nodes = decision points; edges = actions; chance nodes = Nature.
    - **Information set** = nodes a player cannot tell apart → same action at all of them.
    - Strategy = complete plan: an action at *every* own information set, even unreached ones.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\mathcal{H}_i$: Information sets of player $i$.
    - $h$: Information set.
    - $\mathcal{A}(h)$: Actions available at $h$.

Pure strategy: $s_i(h)\in\mathcal{A}(h)\ \ \forall h\in\mathcal{H}_i$.

$$
|\mathcal{S}_i|=\prod_{h\in\mathcal{H}_i}|\mathcal{A}(h)|
$$

- → Exponential in the tree size.

Behavior strategy: independent $b_i(h)\in\Delta(\mathcal{A}(h))$ at each $h$.

**Kuhn's theorem**: perfect recall → every mixed strategy has an outcome-equivalent behavior strategy, and vice versa.
- Perfect recall: a player never forgets its own past actions or information.
```

```{attention} Q&A
:class: dropdown
*Perfect vs complete information?*

- Perfect: every player sees all past moves (all information sets are singletons).
- Complete: every player knows all payoffs.
- Chess: perfect & complete. Poker: imperfect (hidden cards). Auctions: incomplete (hidden values) → [Bayesian Game](#bayesian-game).

*What does perfect information buy?*

- A pure-strategy equilibrium found by backward induction (Zermelo; Kuhn) → [SPE](#subgame-perfect-equilibrium).

*What breaks without perfect recall?*

- Kuhn's equivalence fails. Example: the absent-minded driver cannot tell two intersections apart → planning & in-the-moment choice can disagree.
```

&nbsp;

### Subgame Perfect Equilibrium
- **What**: Profile that is a Nash equilibrium in every subgame {cite:p}`selten1965spieltheoretische`.
- **Why**: Nash equilibrium allows **non-credible threats** off the equilibrium path.
    - Incumbent threatens a price war; after entry, fighting hurts the incumbent → the threat is empty.
- **How**: Backward induction (finite, perfect information):
    1. Solve the last movers' choices.
    2. Replace each solved subtree by its payoff.
    3. Move up; repeat to the root.

```{note} Math
:class: dropdown
Definitions:
- **Subgame**: subtree that starts at a singleton information set, contains all its successors & cuts no information set.
- **SPE**: $\sigma^*$ restricted to every subgame $\Gamma$ is a Nash equilibrium of $\Gamma$.

Properties:
- Finite perfect-information game → pure SPE exists (backward induction); unique if no player has ties.
- Every SPE is a Nash equilibrium; not conversely.
```

```{note} Example
:class: dropdown
Entry game. Entrant: Out → $(0,2)$; In → Incumbent picks Fight $(-1,-1)$ or Accommodate $(1,1)$.

- Nash equilibria: (Out, Fight) & (In, Accommodate).
- (Out, Fight) relies on Fight, which the Incumbent would never choose after entry.
- SPE: (In, Accommodate) only.
```

````{important} Code
:class: dropdown
```python
class BackwardInduction:
    ## node: ('leaf', payoffs) or (player, {action: child})
    def __call__(self, node, path=()):
        if node[0] == 'leaf':
            return node[1], {}
        player, children = node
        plan, best = {}, None
        for a, child in children.items():
            value, sub = self(child, path + (a,))
            plan.update(sub)  ## keep optimal moves at unreached nodes too: SPE needs them
            if best is None or value[player] > best[1][player]:
                best = (a, value)
        plan[path] = best[0]
        return best[1], plan

## Example: entry game; player 0 = Entrant, 1 = Incumbent
entry = (0, {'Out': ('leaf', (0, 2)),
             'In': (1, {'Fight': ('leaf', (-1, -1)), 'Accommodate': ('leaf', (1, 1))})})
print(BackwardInduction()(entry))  ## ((1, 1), {('In',): 'Accommodate', (): 'In'})
```
````

```{attention} Q&A
:class: dropdown
*Why is commitment valuable?*

- A leader that can commit (Stackelberg) turns a threat into a credible one → better SPE payoff for the leader.
- Precommitment is the CDT fix for Newcomb-type problems; [FDT](#fdt) claims to make it unnecessary.

*Centipede & ultimatum games?*

- Backward induction: take the pot at once (centipede); offer the minimum & accept it (ultimatum).
- Lab subjects deviate in both → rationality is not common knowledge, or payoffs include fairness.

*Imperfect information?*

- Few proper subgames → SPE barely bites → refine with beliefs at information sets (perfect Bayesian, sequential equilibrium).
```

&nbsp;

### Bayesian Game
- **What**: Game with incomplete information: each player has a private type drawn from a common prior {cite:p}`harsanyi1967games`.
- **Why**: Players often do not know others' payoffs (values, costs).
    - **Harsanyi transformation**: unknown payoffs → Nature moves first → imperfect-information game → standard tools apply.
- **How**:
    1. Nature draws the type profile $\theta\sim P$.
    2. Each player sees only its own type $\theta_i$.
    3. Strategy = map from own type to action.
    4. **Bayesian Nash equilibrium (BNE)**: every type best-responds in expectation over others' types.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\theta=(\theta_i,\theta_{-i})$: Type profile.
    - $P$: Common prior over type profiles.
    - $\beta_i:\Theta_i\to\mathcal{S}_i$: Strategy of player $i$.
    - $u_i(s;\theta)$: Payoff; may depend on all types.

BNE:

$$
\beta_i^*(\theta_i)\in\arg\max_{s_i\in\mathcal{S}_i}
\mathbb{E}_{\theta_{-i}\sim P(\cdot\mid\theta_i)}
\left[u_i\big(s_i,\beta_{-i}^*(\theta_{-i});\theta\big)\right]
\quad\forall i,\ \forall\theta_i
$$
```

```{tip} Derivation
:class: dropdown
*First-price auction, 2 bidders, values iid $\mathcal{U}[0,1]$: why bid half your value?*

1. Guess the opponent bids $b(\theta')=k\theta'$, $k>0$.
2. Bid $b\leq k$ wins w.p. $P(k\theta'<b)=b/k$.
3. Expected payoff: $(\theta_i-b)\,b/k$ → maximize → $b=\theta_i/2$.
4. Consistent with step 1 at $k=\frac12$ → BNE $b(\theta_i)=\theta_i/2$.
5. Revenue $=\mathbb{E}[\max/2]=\frac13$ = second-price revenue $\mathbb{E}[\min]=\frac13$ → revenue equivalence.
```

```{attention} Q&A
:class: dropdown
*Why a common prior?*

- Keeps beliefs consistent → equilibrium well defined.
- Contested: with a common prior & common knowledge of posteriors, agents cannot agree to disagree (Aumann).

*Bayesian game vs Bayesian decision theory?*

- [Bayesian decision theory](prob/bayes.md#bayesian-decision-theory): one agent vs Nature.
- Bayesian game: uncertainty about *strategic* agents → beliefs about beliefs.

*Where is it used?*

- Auctions, signaling, contract design, [Mechanism Design](#mechanism-design).
```

&nbsp;

### Repeated Game
- **What**: Same stage game played many times; payoff = discounted sum.
- **Why**: Future retaliation → cooperation without external contracts.
- **How**:
    1. Strategies condition on the full history.
    2. Deviation today → punishment later.
    3. Cooperation holds iff the one-shot gain < the discounted future loss.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\delta\in[0,1)$: Discount factor; equivalently, probability the game continues.
- Misc:
    - $s^t$: Profile played in round $t$.

Average discounted payoff:

$$
(1-\delta)\sum_{t=0}^{\infty}\delta^t u_i(s^t)
$$

Grim trigger in PD (cooperate until anyone defects, then defect forever) is an SPE iff

$$
\frac{R}{1-\delta}\geq T+\frac{\delta P}{1-\delta}
\iff
\delta\geq\frac{T-R}{T-P}
$$

- $T,R,P$: [PD](#prisoners-dilemma) payoffs. With $(T,R,P)=(5,3,1)$ → $\delta\geq\frac12$.

**One-shot deviation principle**: in discounted games, a profile is an SPE ⟺ no single-round deviation in any subgame pays.
```

```{attention} Q&A
:class: dropdown
*Finite horizon?*

- Unique stage equilibrium + known last round → backward induction → defect in every round (unraveling).

*Is Tit-for-Tat an equilibrium?*

- Nash for patient players; generally not subgame perfect (punishment echoes cost the punisher).
- Did well in Axelrod's tournaments: nice, retaliatory, forgiving.

*Why is discounting the same as uncertain end?*

- Continuation probability $\delta$ multiplies future payoffs exactly like a discount factor.
```

&nbsp;

#### Folk Theorem
- **What**: Patient players in an infinitely repeated game: any feasible, strictly individually rational payoff vector is an SPE payoff {cite:p}`fudenberg1986folk`.
- **Why**: Explains cooperation without contracts.
    - Flip side: repetition alone predicts almost anything → equilibrium selection problem.
- **How**:
    1. Prescribe a target path of play.
    2. A deviator is pushed down to its minmax payoff.
    3. Punishers are rewarded for punishing → punishment is credible.
    4. $\delta\to1$ → future losses outweigh any one-shot gain.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\underline{v}_i$: Minmax payoff of player $i$.
    - $\mathcal{V}$: Feasible payoff set.
    - $\mathcal{V}^*$: Feasible, strictly individually rational payoffs.

Definitions:

$$
\underline{v}_i=\min_{\sigma_{-i}}\max_{s_i}u_i(s_i,\sigma_{-i}),
\qquad
\mathcal{V}=\operatorname{conv}\{u(s):s\in\mathcal{S}\},
\qquad
\mathcal{V}^*=\{v\in\mathcal{V}:v_i>\underline{v}_i\ \forall i\}
$$

- $\sigma_{-i}$: Independent mixed strategies of the others.

Theorem: $N=2$, or $\dim\mathcal{V}^*=N$ (full dimensionality) →

$$
\forall v\in\mathcal{V}^*\ \exists\underline\delta<1:\ \forall\delta\in(\underline\delta,1),\ \exists\text{ SPE with average payoff }v
$$
```

```{attention} Q&A
:class: dropdown
*Why full dimensionality for $N\geq3$?*

- Punishers need private rewards. If payoffs move together, rewarding a punisher also rewards the deviator.

*What does it say about AI cooperation?*

- Long-lived, patient agents that observe each other can sustain cooperation.
- Bad equilibria (collusion, mutual punishment) are equally sustainable.
```

&nbsp;

## Collective Choice & Mechanism Design

### Social Choice
- **What**: Aggregation of individual rankings into one collective ranking or choice.
- **Why**: Groups (voters, committees, annotators) must act as one.
- **How**: Pick an aggregation rule.
    - **Plurality**: most first places.
    - **Borda**: $|\mathcal{C}|-1$ points for 1st, …, $0$ for last.
    - **Condorcet winner**: beats every rival in pairwise majority; may not exist.

```{note} Math
:class: dropdown
Definitions:
- Social welfare function (SWF): $F:(\succsim_1,\ldots,\succsim_N)\mapsto\succsim$ (collective ranking).
- Social choice function (SCF): $f:(\succsim_1,\ldots,\succsim_N)\mapsto c\in\mathcal{C}$ (single winner).

Borda score:

$$
\text{Borda}(c)=\sum_{i\in\mathcal{N}}\left|\{c'\in\mathcal{C}:c\succ_i c'\}\right|
$$
```

```{note} Example
:class: dropdown
Condorcet paradox: 3 voters rank $a\succ b\succ c$, $b\succ c\succ a$, $c\succ a\succ b$.

- Majority: $a$ beats $b$ (2:1), $b$ beats $c$ (2:1), $c$ beats $a$ (2:1) → cycle.
- → Majority rule can output an intransitive "ranking".
```

````{important} Code
:class: dropdown
```python
class SocialChoice:
    def __init__(self, ballots):
        self.ballots = [list(b) for b in ballots]  ## each ballot: best -> worst
        self.cands = sorted(self.ballots[0])

    def plurality(self):
        return {c: sum(b[0] == c for b in self.ballots) for c in self.cands}

    def borda(self):
        K = len(self.cands)
        return {c: sum(K - 1 - b.index(c) for b in self.ballots) for c in self.cands}

    def condorcet_winner(self):
        ## must win every head-to-head majority vote
        for c in self.cands:
            if all(2 * sum(b.index(c) < b.index(d) for b in self.ballots) > len(self.ballots)
                   for d in self.cands if d != c):
                return c
        return None

## Example: Condorcet paradox
cycle = SocialChoice(["abc", "bca", "cab"])
print(cycle.borda(), cycle.condorcet_winner())  ## {'a': 3, 'b': 3, 'c': 3} None
```
````

```{attention} Q&A
:class: dropdown
*Two alternatives?*

- Simple majority works: it is the only rule that is anonymous, neutral & positively responsive (May's theorem).

*Single-peaked preferences?*

- Options on a line, each voter has one peak, odd #voters → a Condorcet winner exists = the median voter's peak.
- Even #voters → pairwise ties can occur.

*Why does it matter for AI?*

- Aligning to many humans = aggregating their preferences.
- Bradley–Terry reward learning over annotators with hidden differences aggregates like Borda count {cite:p}`siththaranjan2024distributional` → [RLHF](../nlp/post/rl.md#rlhf) picks a voting rule implicitly.
```

&nbsp;

#### Arrow's Impossibility Theorem
- **What**: ≥3 alternatives: no SWF satisfies unrestricted domain, Pareto, IIA & non-dictatorship {cite:p}`arrow1950difficulty`.
- **Why**: No ranking aggregation is "fair" in every sense → every rule gives up some axiom.
- **How**: Pareto + IIA → some coalition is decisive → decisive coalitions shrink → one voter is decisive → dictator.

```{note} Math
:class: dropdown
Conditions on $F$ (output must be a complete, transitive ranking):
1. **Unrestricted domain (U)**: $F$ accepts every profile of complete, transitive rankings.
2. **Weak Pareto (P)**: $c\succ_i c'\ \forall i$ → $c\succ c'$.
3. **Independence of irrelevant alternatives (IIA)**: the social ranking of $c$ vs $c'$ depends only on individual rankings of $c$ vs $c'$.
4. **Non-dictatorship (D)**: no $i$ s.t. $c\succ_i c'$ → $c\succ c'$ for all profiles & all pairs.

Theorem: $|\mathcal{C}|\geq3$, $2\leq N<\infty$ → no $F$ satisfies U, P, IIA & D.
```

```{attention} Q&A
:class: dropdown
*Which axiom do common rules break?*

- Borda & plurality: IIA.
- Pairwise majority: transitive output (Condorcet cycles).

*Escape routes?*

- Restrict the domain (single-peaked).
- Use cardinal or interpersonally comparable utilities (score voting, utilitarian sums). Arrow uses ordinal rankings only.

*Implication for alignment?*

- "Humanity's preferences" has no axiom-complete aggregate → choosing an aggregation rule is a value judgment, not a technical detail.
```

&nbsp;

#### Gibbard–Satterthwaite Theorem
- **What**: ≥3 possible winners: every strategy-proof, onto, deterministic SCF is dictatorial {cite:p}`gibbard1973manipulation,satterthwaite1975strategy`.
- **Why**: Strategic misreporting is unavoidable for reasonable voting rules → mechanism design must restrict preferences, use money, or randomize.
- **How**: Strategy-proofness → monotonicity → induces an SWF that satisfies Arrow's conditions → dictator.

```{note} Math
:class: dropdown
Setting: unrestricted strict rankings over $\mathcal{C}$; $f$ onto $\mathcal{C}$; $|\mathcal{C}|\geq3$.

Strategy-proof:

$$
f(\succsim_i,\succsim_{-i})\succsim_i f(\succsim_i',\succsim_{-i})
\quad\forall i,\ \forall\succsim_i,\succsim_i',\succsim_{-i}
$$

- Truthful report is a dominant strategy.

Conclusion: $\exists i$ s.t. $f$ always picks $i$'s top choice.
```

```{attention} Q&A
:class: dropdown
*Escape routes?*

- Money + quasi-linear utility → [VCG](#vcg).
- Single-peaked domain → median-voter rule is strategy-proof.
- Randomized rules (e.g., random dictator) → not covered by the deterministic theorem.
```

&nbsp;

### Mechanism Design
- **What**: Design of game rules whose equilibrium play produces a desired outcome {cite:p}`hurwicz1973design`.
- **Why**: The best outcome depends on private information that agents may misreport.
    - "Reverse game theory": fix the target equilibrium outcome, then choose the game.
- **How**:
    1. State the target SCF $f(\theta)$.
    2. Design message spaces & an outcome rule $g$.
    3. Require that equilibrium messages produce $f(\theta)$ for every type profile.
    4. Constraints: incentive compatibility (IC), individual rationality (IR), budget balance.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\mathcal{M}_i$: Message space of player $i$.
    - $g:\prod_i\mathcal{M}_i\to\mathcal{O}$: Outcome rule.
    - $\beta_i:\Theta_i\to\mathcal{M}_i$: Reporting strategy.
    - $t_i$: Payment by player $i$.
    - $v_i(o;\theta_i)$: Value of outcome $o$ to type $\theta_i$.

Mechanism $\mathcal{M}=((\mathcal{M}_i)_i,g)$ implements $f$ if equilibrium $\beta^*$ satisfies

$$
g(\beta^*(\theta))=f(\theta)\quad\forall\theta
$$

- **Direct** mechanism: $\mathcal{M}_i=\Theta_i$.
- Quasi-linear utility: $u_i=v_i(o;\theta_i)-t_i$.
- **DSIC**: truth-telling is a dominant strategy.
- **BIC**: truth-telling is a BNE.
- **IR**: participation is at least as good as opting out.
```

```{attention} Q&A
:class: dropdown
*Hard limits?*

- Myerson–Satterthwaite: bilateral trade, independent private values w/ continuous positive densities, overlapping value ranges → no mechanism is ex-post efficient, BIC, interim IR & budget-balanced together.

*AI relevance?*

- Reward design for learning agents = mechanism design with a learner as the agent → [Goodhart's Law](#goodharts-law) is the failure mode.
- Ad auctions, compute markets, multi-agent negotiation.
```

&nbsp;

#### Revelation Principle
- **What**: Any outcome that some mechanism implements in equilibrium is also implemented by a truthful direct mechanism {cite:p}`myerson1979incentive`.
- **Why**: The designer can restrict the search to incentive-compatible direct mechanisms.
- **How**:
    1. Take a mechanism $\mathcal{M}$ with equilibrium $\beta^*$.
    2. Build a new mechanism: ask for types → simulate $\beta^*$ on the reports → apply $g$.
    3. Lying to the simulator = deviating in $\mathcal{M}$ → unprofitable → truth is an equilibrium.

```{note} Math
:class: dropdown
Construction:

$$
g'(\hat\theta)=g\big(\beta^*(\hat\theta)\big)
$$

- $\hat\theta$: Reported type profile.
- $\beta^*$ dominant-strategy (or Bayes–Nash) equilibrium of $\mathcal{M}$ → truth-telling is dominant (or Bayes–Nash) in $g'$.
```

```{attention} Q&A
:class: dropdown
*Limits?*

- The direct mechanism can have extra, bad equilibria → full implementation is a separate problem.
- Assumes the designer can commit to $g'$.
- Ignores communication & computation cost: reporting a full combinatorial valuation can be infeasible.
```

&nbsp;

#### VCG
- **Name**: Vickrey–Clarke–Groves Mechanism {cite:p}`vickrey1961counterspeculation,clarke1971multipart,groves1973incentives`
- **What**: Pick the welfare-maximizing outcome; charge each agent the externality it imposes on the others.
- **Why**: Truthful reporting becomes dominant → efficient outcome under quasi-linear utility.
- **How**:
    1. Agents report valuations.
    2. Choose $o^*$ that maximizes total reported value.
    3. Agent $i$ pays: others' best welfare without $i$ − others' welfare at $o^*$ (Clarke pivot).
    4. Single item → second-price (Vickrey) auction.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\hat v_j$: Reported valuation of agent $j$.

Allocation & payment:

$$
o^*\in\arg\max_{o}\sum_{j}\hat v_j(o),
\qquad
t_i=\max_{o}\sum_{j\neq i}\hat v_j(o)-\sum_{j\neq i}\hat v_j(o^*)
$$
```

```{tip} Derivation
:class: dropdown
*Why is truth dominant?*

1. Utility of $i$: $v_i(o^*)-t_i=\left[v_i(o^*)+\sum_{j\neq i}\hat v_j(o^*)\right]-\max_o\sum_{j\neq i}\hat v_j(o)$.
2. The last term does not depend on $i$'s report.
3. The bracket is total welfare with $i$'s *true* value.
4. Truthful report → mechanism maximizes exactly that bracket → no misreport does better.
```

````{important} Code
:class: dropdown
```python
import numpy as np

class VCG:
    def __call__(self, values):
        V = np.asarray(values, float)  ## [agents, outcomes] reported values
        welfare = V.sum(0)
        o = int(np.argmax(welfare))    ## efficient outcome
        pays = []
        for i in range(V.shape[0]):
            others = welfare - V[i]    ## others' welfare for each outcome
            ## Clarke pivot: others' best welfare w/o i minus their welfare at the chosen outcome
            pays.append(others.max() - others[o])
        return o, np.array(pays)

## Example: one item, 3 bidders; outcome k = "bidder k wins"
print(VCG()(np.diag([10.0, 7.0, 4.0])))  ## (0, array([7., 0., 0.])) -> second-price auction
```
````

```{attention} Q&A
:class: dropdown
*Cons?*

- Not budget-balanced in general.
- Low or zero revenue in some combinatorial auctions.
- Vulnerable to collusion & fake bidders (shills).
- Winner determination is NP-hard in combinatorial auctions.

*Why do real ad auctions differ?*

- Sponsored search historically used generalized second-price (GSP) auctions, which are not truthful.
```

&nbsp;

## Cooperative Games

### Shapley Value
- **What**: Average marginal contribution of a player over all join orders {cite:p}`shapley1953value`.
- **Why**: Fair split of a coalition's total value.
    - Only rule that satisfies efficiency, symmetry, null player & additivity.
- **How**:
    1. Order the players at random.
    2. Credit each player with the value it adds when it joins.
    3. Average over all $N!$ orders.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\nu:2^{\mathcal{N}}\to\mathbb{R}$: Characteristic function, $\nu(\emptyset)=0$.
    - $\mathcal{T}$: Coalition.
    - $\phi_i(\nu)$: Shapley value of player $i$.

$$
\phi_i(\nu)=\sum_{\mathcal{T}\subseteq\mathcal{N}\setminus\{i\}}\frac{|\mathcal{T}|!\,(N-|\mathcal{T}|-1)!}{N!}\left[\nu(\mathcal{T}\cup\{i\})-\nu(\mathcal{T})\right]
$$

Axioms:
- Efficiency: $\sum_i\phi_i=\nu(\mathcal{N})$.
- Symmetry: interchangeable players get equal shares.
- Null player: zero marginal contribution everywhere → $\phi_i=0$.
- Additivity: $\phi(\nu+\nu')=\phi(\nu)+\phi(\nu')$.
```

````{important} Code
:class: dropdown
```python
import itertools

class ShapleyValue:
    def __init__(self, players, value_fn):
        self.players, self.v = list(players), value_fn  ## value_fn: frozenset -> float

    def __call__(self):
        phi = {p: 0.0 for p in self.players}
        orders = list(itertools.permutations(self.players))
        for order in orders:
            coalition = frozenset()
            for p in order:
                ## marginal contribution of p when joining in this order
                phi[p] += self.v(coalition | {p}) - self.v(coalition)
                coalition = coalition | {p}
        return {p: s / len(orders) for p, s in phi.items()}

## Example: glove game; L holds a left glove, R1 & R2 right gloves; a pair is worth 1
v = lambda T: float('L' in T and ('R1' in T or 'R2' in T))
print(ShapleyValue(['L', 'R1', 'R2'], v)())  ## {'L': 0.667, 'R1': 0.167, 'R2': 0.167} (rounded)
```
````

```{attention} Q&A
:class: dropdown
*Link to ML?*

- SHAP: players = features; $\nu(\mathcal{T})$ = model output with only features in $\mathcal{T}$ known → feature attribution {cite:p}`lundberg2017unified`.
- Cost: $N!$ orders → Monte Carlo over random orders, or model-specific shortcuts.

*Shapley value vs core?*

- Core: allocations that no coalition can improve on by leaving. It can be empty.
- Glove game: core = $(1,0,0)$ ← R1 & R2 compete. Shapley $(\frac23,\frac16,\frac16)$ lies outside: $\{L,R1\}$ can get $1>\frac56$.

*Does "fair" mean causal?*

- No. Shapley credit depends on how $\nu$ handles missing players (e.g., which baseline replaces absent features).
```

&nbsp;

## Decision Theories for Predictable Agents

### Newcomb's Problem
- **What**: Take one opaque box or both boxes; a reliable predictor filled the opaque box iff it predicted one-boxing {cite:p}`nozick1969newcomb`.
- **Why**: Splits two principles that usually agree: dominance (take both) vs expected utility conditional on the act (take one).
    - Savage's act–state independence fails: the act correlates with the state (the prediction) without causing it.
- **How**:
    - Box A (transparent): \$1,000.
    - Box B (opaque): \$1,000,000 if the predictor predicted one-boxing, else \$0.
    - Two-box argument: contents are already fixed → taking A adds \$1,000 in either state.
    - One-box argument: one-boxers almost always leave rich.

```{tip} Derivation
:class: dropdown
*When does each argument win? (utility linear in money)*

1. Let $\alpha$ = predictor accuracy, independent of the act.
2. Conditional on the act: $\mathbb{E}[V\mid\text{one}]=\alpha\cdot10^6$, $\mathbb{E}[V\mid\text{two}]=(1-\alpha)\cdot10^6+10^3$.
3. → One-box iff $\alpha>0.5005$.
4. Holding $q=P(\text{B full})$ fixed: $\mathbb{E}[V\mid do(\text{one})]=q\cdot10^6<q\cdot10^6+10^3=\mathbb{E}[V\mid do(\text{two})]$.
5. → Two-box for every $q$.
```

```{attention} Q&A
:class: dropdown
*Who is right?*

- Unresolved. Philosophers remain split.
- Two-boxers: the predictor rewards irrationality; you cannot cause money into a sealed box.
- One-boxers: a theory that predictably leaves you poorer is the wrong theory.

*Why does AI care?*

- AI agents can be copied, simulated & inspected → accurate predictors of an AI are realistic.
- Newcomb-like structure appears in cooperation between copies, negotiation & blackmail.

*Is a perfect predictor needed?*

- No. Under these payoffs & linear utility, any accuracy above $0.5005$ favors one-boxing in step 3.
```

&nbsp;

### CDT
- **Name**: Causal Decision Theory {cite:p}`gibbard1978counterfactuals`
- **What**: Choose the action with the highest expected utility under an *intervention* on the action.
- **Why**: Evidential reasoning endorses acts that only correlate with good outcomes, without causing them.
- **How**: Set Act $=a$ by intervention → update only its causal descendants → hold everything else at prior beliefs → compute EU.

```{note} Math
:class: dropdown
$$
\text{CDT}(e)\in\arg\max_{a\in\mathcal{A}}\mathbb{E}\left[V\mid do(\text{Act}=a),\ \text{Obs}=e\right]
$$

- $do(\cdot)$: Causal intervention (Pearl).
- Gibbard–Harper form: weight outcomes by probabilities of counterfactuals "if I did $a$, $o$ would occur".
```

```{attention} Q&A
:class: dropdown
*Where does CDT lose utility?*

- Newcomb: two-boxes → \$1,000.
- Twin PD: defects against its own copy.
- Death in Damascus: no stable choice.
- Parfit's hitchhiker: cannot credibly promise to pay → left in the desert.

*Why is CDT the mainstream view?*

- Matches dominance & causal intuitions.
- Equals EDT when acts carry no evidence about states (Savage's setting).

*Standard fix?*

- Precommitment: a CDT agent pays to bind itself before the prediction. FDT asks why the binding device is needed.
```

&nbsp;

### EDT
- **Name**: Evidential Decision Theory {cite:p}`jeffrey1965logic`
- **What**: Choose the action with the highest expected utility *conditional* on taking it.
- **Why**: Plain Bayesian conditioning; no causal model needed. Drops Savage's act–state independence.
- **How**: Treat your own action as evidence → condition beliefs on Act $=a$ → compute EU.

```{note} Math
:class: dropdown
$$
\text{EDT}(e)\in\arg\max_{a\in\mathcal{A}}\mathbb{E}\left[V\mid\text{Obs}=e,\ \text{Act}=a\right]
$$

- Jeffrey's "desirability" of an act = its value as news.
- Actions with probability $0$ have undefined conditionals → excluded.
```

```{attention} Q&A
:class: dropdown
*Where does EDT lose utility?*

- Smoking lesion: a gene causes both smoking & cancer; smoking itself is harmless → EDT refuses to smoke. ("Tickle defense": self-knowledge can screen off the evidence; contested.)
- XOR blackmail: pays to receive good news about something its payment cannot change ("managing the news").
- Transparent Newcomb & Parfit's hitchhiker: after observing, its act carries no more evidence → loses like CDT.

*Where does it win?*

- One-boxes in Newcomb; cooperates in Twin PD.
```

&nbsp;

### FDT
- **Name**: Functional Decision Theory {cite:p}`yudkowsky2017functional`
- **What**: Choose the output of your *decision function* that yields the best outcome, varying every process that computes the same function (copies, predictor models).
- **Why**: CDT & EDT each lose utility in some Newcomb-like problem. FDT claims one rule that does well in all of them.
    - Contested; see Q&A.
- **How**:
    1. Model which variables *subjunctively depend* on your decision function (twins, predictors' simulations).
    2. Intervene on the function's output, not on the physical act.
    3. Propagate to all dependents → compute EU → choose the max.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\text{fdt}(e)$: Node for the output of the agent's decision function on input $e$; logical dependencies are inputs to the model, like causal ones for CDT.

$$
\text{FDT}(e)\in\arg\max_{a\in\mathcal{A}}\mathbb{E}\left[V\mid do(\text{fdt}(e)=a)\right]
$$

- vs CDT: intervene on the function node, not on Act. Every node that computes the same function changes too.
- Authors' preferred formalization selects a *policy* (observation → action map), not an action → close to [UDT](#udt).
```

```{dropdown} Table: Decision Theories on Newcomb-Like Problems
| Problem | CDT | EDT | FDT |
|:--|:--|:--|:--|
| Newcomb | Two-box | One-box | One-box |
| Smoking lesion | Smoke | Refrain | Smoke |
| Twin PD | Defect | Cooperate | Cooperate |
| Parfit's hitchhiker | Won't pay → stranded | Won't pay → stranded | Pays → rescued |
| Transparent Newcomb | Two-box | Two-box | One-box |
| XOR blackmail | Refuse | Pay | Refuse |
| Mechanical blackmail | Pay | Pay | Refuse → not blackmailed |
| Death in Damascus | Unstable | Stay | Stay |

- Verdicts as analyzed in the FDT paper; the EDT entry for Death in Damascus follows from conditioning (death is certain either way → save the fee).
```

```{attention} Q&A
:class: dropdown
*Main open problem?*

- FDT must evaluate **counterpossibles**: "what if my function output $a$", when logic fixes a different output.
- No satisfactory formal account exists. The authors call this FDT's main drawback & take the dependency structure as a given input.

*Other objections?*

- Contested: critics argue the test problems are chosen to reward FDT-like dispositions.
- "Same function" is underspecified: how similar must another computation be to count?
- In the realized branch FDT can do worse (e.g., refuses mechanical blackmail even after the letter arrives).

*Why does it matter for AI?*

- Copies & simulations of AIs make Newcomb-like problems routine.
- Claimed benefits: cooperation with copies, resistance to blackmail, commitment without external devices.
- Contested ≠ refuted: treat it as a research program, not settled theory.
```

&nbsp;

#### UDT
- **Name**: Updateless Decision Theory {cite:p}`dai2009towards`
- **What**: Choose the best *policy* (observation → action map) from the prior, without updating on observations; then execute it.
- **Why**: Updating can break dynamic consistency: after an observation, the agent may abandon the plan it endorsed before.
- **How**:
    1. Score each policy by EU under the prior over all worlds, including worlds that do not match the current observation.
    2. Pick the best policy $\pi^*$.
    3. Act $\pi^*(e)$ on the actual observation $e$.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\pi:\mathcal{E}\to\mathcal{A}$: Policy.
    - $\mathcal{E}$: Set of possible observations.

$$
\pi^*\in\arg\max_{\pi}\mathbb{E}_{\text{prior}}\left[V\mid\text{policy}=\pi\right],
\qquad a=\pi^*(e)
$$

- "$\text{policy}=\pi$" is set the FDT way: every model or copy of the agent's policy changes with it.
```

```{note} Example
:class: dropdown
Counterfactual mugging: a perfect predictor flips a fair coin.

- Tails: it asks you for \$100.
- Heads: it pays you \$10,000 iff it predicts you would pay on tails.
- The coin lands tails.
- Updated view: paying = $-100$ → refuse.
- UDT: policy "pay" scores $0.5\cdot10{,}000-0.5\cdot100=4{,}950>0$ → pay.
- Imperfect predictor w/ accuracy $\alpha$: pay iff $5000\alpha-50>5000(1-\alpha)$ ⟺ $\alpha>0.505$.
```

```{attention} Q&A
:class: dropdown
*UDT vs ordinary Bayesian planning?*

- No predictors (e.g., a standard [POMDP](../rl/intro.md#pomdp)): the best policy from the prior = acting optimally after each update → UDT changes nothing.
- They differ only when the environment depends on what you *would* do in branches you are not in.

*UDT vs FDT?*

- FDT's preferred policy form ≈ UDT. UDT came first as an informal blog proposal.

*Open problems?*

- No canonical formal definition; needs logical uncertainty & a choice of prior.
- Multi-agent: early updateless commitments can escalate ("commitment races").
```

&nbsp;

## Goodhart's Law
- **What**: Proxy–goal correlation collapses under optimization pressure on the proxy {cite:p}`goodhart1975problems,manheim2018categorizing`.
    - Goodhart's wording: "Any observed statistical regularity will tend to collapse once pressure is placed upon it for control purposes."
- **Why**: Every objective we can write (reward model, benchmark, loss) is a proxy for the intended goal → hard optimization exploits the gap.
- **How**: Four mechanisms:
    - **Regressional**: proxy = goal + noise → selecting on the proxy also selects noise → goal < proxy.
    - **Extremal**: optimization moves into regions where the proxy–goal relation never held.
    - **Causal**: proxy correlates with the goal without causing it → pushing the proxy leaves the goal unchanged.
    - **Adversarial**: another agent games the proxy for its own goal.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $G$: Goal value.
    - $M$: Proxy (measured) value.
    - $\epsilon$: Proxy noise.
    - $\tau^2$: Noise variance.

Regressional model:

$$
G\sim\mathcal{N}(0,1),
\qquad
M=G+\epsilon,\ \ \epsilon\sim\mathcal{N}(0,\tau^2)\ \text{indep.}
$$

$$
\mathbb{E}[G\mid M=m]=\frac{m}{1+\tau^2}
$$

- → The winner's *expected* goal value is only a fraction $\frac{1}{1+\tau^2}$ of its proxy score.
```

```{tip} Derivation
:class: dropdown
*Why the shrink factor $\frac{1}{1+\tau^2}$?*

1. $(G,M)$ is jointly Gaussian with $\operatorname{Cov}(G,M)=\operatorname{Var}(G)=1$, $\operatorname{Var}(M)=1+\tau^2$.
2. Gaussian conditioning: $\mathbb{E}[G\mid M=m]=\frac{\operatorname{Cov}(G,M)}{\operatorname{Var}(M)}m=\frac{m}{1+\tau^2}$.
3. Selection picks large $m$ → the absolute gap $m-\mathbb{E}[G\mid m]=\frac{\tau^2}{1+\tau^2}m$ grows with optimization pressure.
```

````{important} Code
:class: dropdown
```python
import numpy as np

class RegressionalGoodhart:
    def __init__(self, noise_std, seed=0):
        self.noise_std, self.rng = noise_std, np.random.default_rng(seed)

    def __call__(self, n, trials=2000):
        G = self.rng.standard_normal((trials, n))                        ## true goal
        M = G + self.noise_std * self.rng.standard_normal((trials, n))  ## proxy
        best = M.argmax(1)                                               ## best-of-n on the proxy
        rows = np.arange(trials)
        return M[rows, best].mean(), G[rows, best].mean()                ## winner's proxy vs goal

## Example: more pressure (larger n) -> wider proxy-goal gap; goal ≈ proxy / (1 + noise_std**2)
for n in [1, 10, 100]:
    print(n, np.round(RegressionalGoodhart(noise_std=1.0)(n), 2))
```
````

```{attention} Q&A
:class: dropdown
*Where does it appear in ML?*

- Reward hacking in [RLHF](../nlp/post/rl.md#rlhf): policy exploits reward-model errors.
- Benchmark overfitting; specification gaming in RL.

*Mitigations?*

- Limit optimization pressure: KL penalty to a reference policy; quantilizers (sample from the top fraction of a base distribution instead of the argmax).
- Better proxies: ensembles, held-out & adversarial evaluation.
- None removes the gap; they bound how far optimization exploits it.

*Is it a law?*

- An empirical tendency, not a theorem. The regressional variant is a theorem under the stated noise model.

*What does it tell us about AC/ASI?*

- A stronger optimizer pushed on a fixed proxy drifts further from intent → capability amplifies misspecification.
```

&nbsp;
