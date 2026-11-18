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
# Computability & Algorithmic Information
What machines can compute at all, & how much information one individual object contains.
Standard references: {cite:t}`toc` (computability) & {cite:t}`ait` (algorithmic information).
Running-time complexity classes (P, NP) are out of scope.

Notations:
- $\{0,1\}^*$: Set of finite binary strings.
- $\epsilon$: Empty string.
- $x,y,w$: Binary strings.
- $|x|$: Length of $x$ in bits.
- $\langle\cdot\rangle$: Fixed computable encoding of machines, tuples & numbers as binary strings.
- $M$: Turing machine.
- $M(w)\downarrow$: $M$ halts on input $w$.
- $M(w)\uparrow$: $M$ runs forever on input $w$.
- $\varphi_e$: Partial computable function computed by the machine with code $e$.
- $\mathrm{HALT}=\{\langle M,w\rangle:M(w)\downarrow\}$: Halting set.
- $U$: Fixed reference universal machine; prefix-free in the algorithmic-information sections.
- $p$: Program for $U$.
- $K(x)$: Prefix Kolmogorov complexity of $x$ w.r.t. $U$.
- $T$: Effectively axiomatized formal theory (axioms are c.e.).
- $\mu$: Computable probability measure (data source or environment).
- $\xi$: Solomonoff universal mixture.

- **Overrides**: $x,y$ are data strings, not model input/output. $T$ & $M$ are a theory & a machine, not matrices.
- **Conventions**: $\log=\log_2$. $O(1)$: additive constant that depends on $U$ & on the fixed objects named in the statement, never on $x$.

&nbsp;

## Models of Computation

### Turing Machine
- **What**: Finite-state controller + unbounded read/write tape + moving head.
- **Why**: "Effective procedure" is informal → need a precise object to prove what algorithms can & cannot do.
- **How**:
    1. Read the symbol under the head.
    2. Look up (curr state, symbol) in a finite transition table.
    3. Write a symbol, move the head L/R, switch state.
    4. Repeat until a halt state. Output = tape content.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $Q$: Finite set of states.
    - $\Gamma$: Finite tape alphabet.
    - $\sqcup\in\Gamma$: Blank symbol.
    - $\delta$: Transition function.
    - $q_0\in Q$: Start state.
    - $F\subseteq Q$: Halt states.

Definition:

$$
M=(Q,\Gamma,\sqcup,\delta,q_0,F),\qquad \delta:(Q\setminus F)\times\Gamma\to Q\times\Gamma\times\{L,R\}
$$

- Configuration: curr state + tape content + head position.
- Step: apply $\delta$ once to the curr configuration.
- Tape: unbounded, but only finitely many non-blank cells at every step.
- Output: tape content when a state in $F$ is reached. ❌Halt → output undefined → $M$ computes a **partial** function.
- Multi-tape, nondeterministic, or any $|\Gamma|\ge2$ variant → same class of computable functions.
- Multi-tape → single-tape simulation: $t$ steps → $O(t^2)$ steps.
```

````{important} Code
:class: dropdown
```python
from collections import defaultdict

class TuringMachine:
    def __init__(self, delta, start="A", halt="H", blank=0):
        self.delta = delta  ## (state, symbol) -> (write, move, next_state)
        self.start, self.halt, self.blank = start, halt, blank

    def run(self, max_steps):
        ## sparse tape: unbounded both ways, finitely many non-blank cells
        tape = defaultdict(lambda: self.blank)
        state, head, steps = self.start, 0, 0
        while state != self.halt:
            if steps == max_steps:
                return None  ## halting is undecidable -> a step cap is the only safe stop
            write, move, state = self.delta[(state, tape[head])]
            tape[head] = write
            head += 1 if move == "R" else -1
            steps += 1
        return steps, sum(v != self.blank for v in tape.values())

## Example: 2-state busy beaver champion
bb2 = {("A", 0): (1, "R", "B"), ("A", 1): (1, "L", "B"),
       ("B", 0): (1, "L", "A"), ("B", 1): (1, "R", "H")}
print(TuringMachine(bb2).run(max_steps=100))  ## (6, 4): halts after 6 steps with 4 ones
```
````

```{attention} Q&A
:class: dropdown
*Why a tape & not RAM?*

- Simplest model that is still universal → short proofs.
- RAM machines, λ-calculus, μ-recursive functions → same computable functions. They differ only in running time.

*Deterministic vs nondeterministic?*

- Same computability ← simulate all branches breadth-first.
- Running time can differ: that gap is the P vs NP question.
```

&nbsp;

#### Universal Turing Machine
- **What**: TM that runs any TM from its description: $U(\langle M,w\rangle)=M(w)$.
- **Why**: One fixed machine runs every program → stored-program computer; reference machine for $K$.
- **How**:
    1. Tape holds $\langle M\rangle$ (transition table), $M$'s simulated tape, & $M$'s simulated state.
    2. Per simulated step: find the table row for (state, symbol) → apply write, move, next state.
    3. Simulated $M$ halts → $U$ halts with the same output.

```{attention} Q&A
:class: dropdown
*What is the simulation overhead?*

- Multi-tape $U$ simulates $t$ steps of $M$ in $O(t\log t)$ steps (Hennie–Stearns).
- Constant depends on $M$, not on the input.

*Why does universality matter for algorithmic information?*

- $U$ simulates any machine $V$ given the prefix $\langle V\rangle$ → [Invariance Theorem](#invariance-theorem).

*Real-world instance?*

- An interpreter (Python) or a CPU = universal machine. Code = data.
```

&nbsp;

### Church–Turing Thesis
- **What**: Effectively calculable function = Turing-computable function.
- **Why**: "Algorithm" is informal → the thesis ties the intuition to one formal class.
    - Not a theorem: one side is informal → only supported by evidence, never proved.
- **How**: Evidence by convergence:
    1. Independent formalisms (λ-calculus, μ-recursive functions, TMs, Post systems, register machines) → provably the same class.
    2. No counterexample since 1936.

```{attention} Q&A
:class: dropdown
*Which variants exist?*

- **Physical CT**: every function a physical system can compute is Turing-computable. Empirical claim; open.
- **Extended CT**: every reasonable model is simulable by a probabilistic TM w/ polynomial overhead. Quantum computing challenges it (Shor's factoring), if factoring has no classical polynomial algorithm.
- Quantum computing does not challenge computability: a TM simulates a quantum computer w/ exponential overhead.

*Is hypercomputation possible?*

- Oracle machines, infinite-time TMs & exact real-number machines are mathematically well defined.
- No known physical realization.

*Does the brain violate it?*

- No accepted evidence.
- Penrose–Lucas claim human mathematical insight is non-computable → [Orch OR](../ac/theories.md#orchestrated-objective-reduction-theory-orch-or). Most logicians reject the argument ([Gödel](#godel)).

*What does it imply for AC?*

- Physical CT + consciousness depends only on physical processes → a TM can reproduce a conscious system's input–output behavior.
- Same behavior ≠ same experience. CT is silent on whether a simulation is conscious.
- [IIT](../ac/theories.md#integrated-information-theory-iit) denies it: consciousness depends on causal structure, not on function.
```

&nbsp;

### Turing Completeness
- **What**: Ability to simulate any TM, given unbounded memory & time.
- **Why**: Universality test for languages, physical systems & NN architectures.
- **How**: Emulate a known universal model: encode tape, head, state & transition table.

```{attention} Q&A
:class: dropdown
*Examples?*

- ✅ Python, C, λ-calculus, Rule 110 cellular automaton, Conway's Game of Life.
- ❌ Finite automata, true regular expressions, any language in which every program halts.

*Why can no language in which every program halts be Turing complete?*

- Its interpreter $\mathrm{run}(e,z)$ is total & computable.
- Diagonal: $d(e)=\mathrm{run}(e,e)+1$ is total & computable.
- $d$ = program $e_d$ → $d(e_d)=d(e_d)+1$. Contradiction → the language misses a computable function.

*Is a physical computer Turing complete?*

- Strictly ❌: finite memory → finite automaton.
- "Turing complete" = idealization w/ memory that can always be extended.

*Are transformers Turing complete?*

- ✅ in theory: hard attention + arbitrary-precision rationals + unbounded decoding steps {cite:p}`perez2021attention`.
- ❌ w/ fixed precision (same paper).
- Decoding steps = compute: polynomially many chain-of-thought steps → exactly P (Merrill & Sabharwal, ICLR 2024; log precision, saturated attention, generalized pre-norm).
- RNNs: rational weights + unbounded precision → Turing complete (Siegelmann & Sontag, 1995).

*Does Turing completeness matter for ML?*

- It measures expressivity in the limit only.
- ❌ Says nothing about learnability, sample efficiency, or what SGD finds.
```

&nbsp;

## Computability

### Decidable & Semi-Decidable Sets
- **What**: Membership answerable by a TM that always halts (decidable) / halts exactly on members (semi-decidable).
    - Decidable = recursive = computable set.
    - Semi-decidable = computably enumerable (c.e.) = recursively enumerable (r.e.).
- **Why**: Separate "solvable" from "confirmable when true" → exact boundary of algorithms.
- **How**:
    - Decidable: run the decider → an answer always arrives.
    - Semi-decidable: search for a finite certificate (halting run, proof). Found → "yes". Else search forever.
    - Dovetail: phase $s$ runs inputs $1..s$ for $s$ steps each → enumerates all members, although some runs never stop.

```{note} Math
:class: dropdown
Notations:
- $A,B\subseteq\{0,1\}^*$: Sets (decision problems).
- $\bar{A}$: Complement of $A$.
- $\mathbf{1}_A$: Indicator function of $A$.

Definitions:
- $A$ decidable ⇔ $\mathbf{1}_A$ is total computable.
- $A$ c.e. ⇔ $A=\mathrm{dom}(\varphi_e)$ for some $e$
    - ⇔ $A=\emptyset$ or $A$ = range of a total computable function
    - ⇔ $A=\{x:\exists t\ R(x,t)\}$ for a decidable relation $R$.

Properties:
- **Complement theorem**: $A$ decidable ⇔ $A$ & $\bar{A}$ both c.e.
- Decidable sets: closed under complement, union, intersection.
- C.e. sets: closed under union & intersection; ❌ closed under complement.
```

```{tip} Derivation
:class: dropdown
*Why does "$A$ & $\bar{A}$ both c.e." imply decidable?*

1. Take recognizers $M_A$ & $M_{\bar{A}}$.
2. On input $x$, alternate single steps of $M_A(x)$ & $M_{\bar{A}}(x)$.
3. $x$ lies in exactly one of $A$, $\bar{A}$ → exactly one recognizer halts → output its answer.
4. Converse: a decider for $A$ is a recognizer for both $A$ & $\bar{A}$.
```

```{dropdown} Table: Classes of Sets
| Class | Example |
|:--|:--|
| Decidable | Primes; "$M$ halts within $t$ steps" |
| C.e., not decidable | $\mathrm{HALT}$; theorems of Peano arithmetic |
| Co-c.e., not decidable | $\overline{\mathrm{HALT}}$ |
| Neither | $\{e:\varphi_e\text{ is total}\}$ |
```

```{attention} Q&A
:class: dropdown
*Why are almost all sets undecidable?*

- #TMs is countable. #subsets of $\{0,1\}^*$ is uncountable.
- → Almost all sets have no decider. Non-constructive.

*What does the $\exists t\ R(x,t)$ form mean?*

- C.e. = "a finite witness exists & a machine can check it".
- NP = same idea w/ polynomial-size witness & polynomial-time check.

*Is semi-decidability useful in practice?*

- ✅ Theorem provers, counterexample search, fuzzers: "yes" arrives in finite time.
- "No" may never arrive → practical systems use timeouts.
```

&nbsp;

### Halting Problem
- **What**: Undecidable problem: does TM $M$ halt on input $w$?
- **Why**: Canonical undecidable problem → base of most undecidability proofs via [reduction](#reduction).
- **How**: Diagonalization: feed a hypothetical decider a program built to contradict the decider on its own code.

```{note} Math
:class: dropdown
- $\mathrm{HALT}$ is c.e. ← run $M$ on $w$; accept on halt.
- $\mathrm{HALT}$ is not decidable ← Derivation.
- $\overline{\mathrm{HALT}}$ is not c.e. ← complement theorem.
```

```{tip} Derivation
:class: dropdown
*Why is $\mathrm{HALT}$ undecidable?*

1. Assume a total computable $h$ with $h(\langle M,w\rangle)=1$ iff $M(w)\downarrow$.
2. Build machine $D$. On input $\langle M\rangle$: if $h(\langle M,\langle M\rangle\rangle)=1$, loop forever; else halt.
3. Run $D$ on its own code: $D(\langle D\rangle)\downarrow$ ⇔ $h(\langle D,\langle D\rangle\rangle)=0$ ⇔ $D(\langle D\rangle)\uparrow$.
4. Contradiction → $h$ does not exist.
5. Same pattern as Cantor's diagonal: $D$ differs from every machine $M$ on input $\langle M\rangle$.
```

```{attention} Q&A
:class: dropdown
*Why not run the program w/ a timeout?*

- Timeout → semi-decision only.
- No computable bound on running time exists ← else run that long & decide $\mathrm{HALT}$.
- [Busy Beaver](#busy-beaver) makes the missing bound concrete.

*Does "undecidable" mean no instance can be solved?*

- ❌. Many instances are easy (no loops; a termination proof exists).
- Undecidable = no single algorithm is correct on ALL inputs.

*Does it apply to finite-memory machines?*

- Finite memory → finite #configurations → a repeated configuration means an infinite loop → decidable in principle.
- #configurations is exponential in memory size → infeasible.

*Does it show humans beat machines?*

- ❌. The proof applies to any TM-equivalent reasoner.
- No evidence shows that humans decide halting in general.
```

&nbsp;

### Reduction
- **What**: Computable translation of instances of $A$ into instances of $B$ that preserves the yes/no answer.
- **Why**: Transfer hardness: $A$ undecidable & $A$ reduces to $B$ → $B$ undecidable. No new diagonal argument per problem.
- **How**:
    1. Pick a known undecidable $A$ (usually $\mathrm{HALT}$).
    2. Build a computable $f$ with $x\in A\Leftrightarrow f(x)\in B$.
    3. A decider for $B$ composed w/ $f$ → a decider for $A$ → contradiction.

```{note} Math
:class: dropdown
Notations:
- $f$: Total computable function.

Many-one reduction:

$$
A\le_m B\iff\exists f\ \forall x:\ x\in A\Leftrightarrow f(x)\in B
$$

Turing reduction:
- $A\le_T B$ ⇔ an oracle TM w/ an oracle for $B$ decides $A$.

Properties:
- $A\le_m B$ & $B$ decidable → $A$ decidable.
- $A\le_m B$ & $B$ c.e. → $A$ c.e.
- $A\le_m B$ → $A\le_T B$. Converse ❌: $\mathrm{HALT}\le_T\overline{\mathrm{HALT}}$, but $\mathrm{HALT}\not\le_m\overline{\mathrm{HALT}}$ ← else $\mathrm{HALT}$ would be co-c.e.
```

```{note} Example
:class: dropdown
*Is "halts on empty input" undecidable?*

- Target: $B=\{\langle M\rangle:M(\epsilon)\downarrow\}$.
- $f(\langle M,w\rangle)=\langle M_w\rangle$. $M_w$ ignores its input & runs $M$ on $w$.
- $M(w)\downarrow\Leftrightarrow M_w(\epsilon)\downarrow$ → $\mathrm{HALT}\le_m B$ → $B$ undecidable.
```

```{attention} Q&A
:class: dropdown
*Which direction is correct?*

- Reduce FROM the known-hard problem TO the new problem: $\mathrm{HALT}\le B$.
- $B\le\mathrm{HALT}$ proves nothing about the hardness of $B$.

*What is an oracle?*

- Black box that answers membership in $B$ in one step → relative computability.
- A $\mathrm{HALT}$-oracle machine cannot decide its own halting problem ← same diagonal argument, relativized.
- → Infinite hierarchy of harder problems (Turing jumps).
```

&nbsp;

### Rice's Theorem
- **What**: Every nontrivial semantic property of programs is undecidable.
    - Semantic: depends only on the computed function, not on the code text.
    - Nontrivial: some programs have it, some do not.
- **Why**: One theorem rules out whole classes of perfect tools: "computes $f$", "never outputs X", "is equivalent to program P".
- **How**: Reduce $\mathrm{HALT}$ to the property:
    1. WLOG the nowhere-defined function lacks property $\mathcal{P}$. Pick $e_1$ with $\varphi_{e_1}\in\mathcal{P}$.
    2. Given $\langle M,w\rangle$, build program $q$: on input $z$, run $M(w)$; if it halts, return $\varphi_{e_1}(z)$.
    3. $M(w)\downarrow$ → $\varphi_q=\varphi_{e_1}\in\mathcal{P}$. $M(w)\uparrow$ → $\varphi_q$ nowhere defined $\notin\mathcal{P}$.
    4. A decider for $\mathcal{P}$ → a decider for $\mathrm{HALT}$. Contradiction.

```{note} Math
:class: dropdown
Notations:
- $\mathcal{P}$: Class of partial computable functions, $\emptyset\neq\mathcal{P}\neq$ all partial computable functions.

Theorem:

$$
\{e:\varphi_e\in\mathcal{P}\}\ \text{is undecidable}
$$

- Nowhere-defined function $\in\mathcal{P}$ → apply the theorem to the complement class ← decidability is closed under complement.
```

```{attention} Q&A
:class: dropdown
*Are syntactic properties also undecidable?*

- ❌. "Has < 100 lines" or "contains a loop" are decidable. Rice covers behavior only.

*Does Rice make program verification impossible?*

- ❌. Practical tools give up one guarantee:
    - Sound over-approximation (type checkers, abstract interpretation) → false alarms.
    - Restricted languages → only some programs are expressible.
    - Human-written proofs → the machine checks proofs, it does not find all of them.

*What does it mean for AI safety?*

- No algorithm decides, for every program, a behavioral property defined over all inputs ("never takes unsafe action").
- Verification must restrict the class: e.g., a fixed ReLU network on a bounded input box → decidable.
    - "An unsafe input exists" → NP-complete (Reluplex, 2017). "No unsafe input exists" → coNP-complete.
```

&nbsp;

### Recursion Theorem
- **What**: Every computable program transformation has a fixed point: a program that behaves like its own transform.
- **Why**: Self-reference is legal in computation → a program can obtain & use its own source code.
- **How**: Quine construction:
    1. Part B: given text $t$, build the program "print $t$" → output that program followed by $t$.
    2. Part A = "print the text of B".
    3. Run AB: A writes B's text; B outputs "print B's text" + B's text = A + B = own source.
    4. Replace "output own source" with "run $f$ on own source" → fixed point of $f$.

```{note} Math
:class: dropdown
Notations:
- $f$: Total computable function on program codes.

Kleene's second recursion theorem:

$$
\forall f\ \exists e:\ \varphi_e=\varphi_{f(e)}
$$

Parametrized form:
- For every partial computable $\psi(e,z)$ there is $e$ with $\varphi_e(z)=\psi(e,z)$ for all $z$.
- → A program may use its own code $e$ as an input.
```

````{note} Example
:class: dropdown
Python quine: the program prints its own source.

```python
s = 's = %r\nprint(s %% s)'
print(s % s)
```

- `%r` inserts the quoted text of `s` → line 1.
- `%%` becomes `%` → line 2.
````

```{attention} Q&A
:class: dropdown
*How does Rice's theorem follow?*

- Assume a decider for $\mathcal{P}$. Fix $e_0$ with $\varphi_{e_0}\notin\mathcal{P}$ & $e_1$ with $\varphi_{e_1}\in\mathcal{P}$.
- Program $e$: get own code; if the decider says $\varphi_e\in\mathcal{P}$, behave like $e_0$; else like $e_1$.
- Either answer is wrong → contradiction.

*Where does it appear in practice?*

- Self-replicating programs: von Neumann's self-reproducing automata, computer viruses.

*What does it tell us about AC?*

- A computable system can hold & operate on a complete description of itself.
- → Self-reference does not require non-computable physics.
- A self-model is not consciousness. The theorem removes one claimed barrier only.
```

&nbsp;

(godel)=
### Gödel's Incompleteness Theorems
- **What**: Consistent, effectively axiomatized theories that contain basic arithmetic are incomplete & cannot prove their own consistency.
- **Why**: Hilbert's program sought a complete, consistent, mechanically checkable foundation for mathematics → impossible.
- **How**:
    1. Arithmetize syntax: formulas & proofs → numbers. "$d$ is a proof of $\phi$" → decidable relation.
    2. Diagonal lemma → sentence $G$ ⇔ "$G$ is not provable in $T$".
    3. $T$ consistent → $T\nvdash G$ → $G$ is true in $\mathbb{N}$.
    4. Formalize step 3 inside $T$: $T\vdash\mathrm{Con}(T)\to G$. If $T\vdash\mathrm{Con}(T)$ → $T\vdash G$. Contradiction → second theorem.

```{note} Math
:class: dropdown
Notations:
- $T\vdash\phi$: $T$ proves sentence $\phi$.
- $\mathrm{Con}(T)$: Arithmetic sentence "$T$ proves no contradiction".
- $Q$: Robinson arithmetic (weak finite fragment of arithmetic).
- $\mathrm{PA}$: Peano arithmetic.

First theorem (Gödel–Rosser):
- $T$ consistent, effectively axiomatized, interprets $Q$ → there is a sentence $\phi$ with $T\nvdash\phi$ & $T\nvdash\neg\phi$.
- Gödel's 1931 version needs ω-consistency. Rosser (1936) → plain consistency suffices.

Second theorem:
- $T$ consistent, effectively axiomatized, extends $\mathrm{PA}$ → $T\nvdash\mathrm{Con}(T)$.
- Holds more generally for theories that satisfy the Hilbert–Bernays–Löb derivability conditions.
```

```{tip} Derivation
:class: dropdown
*Why does the undecidability of $\mathrm{HALT}$ imply incompleteness?*

1. Assume $T$ is complete, consistent, effectively axiomatized, interprets $Q$, & proves only true halting statements.
2. The theorems of $T$ are c.e. → enumerate them.
3. For $\langle M,w\rangle$, search for a proof of "$M(w)\downarrow$" or of "$M(w)\uparrow$".
4. Completeness → one proof appears. Truth of halting theorems → that answer is correct.
5. → $T$ decides $\mathrm{HALT}$. Contradiction.
6. **Clarification**: This short proof needs truth for halting statements ($\Sigma_1$-soundness). The Gödel–Rosser theorem needs only consistency.
```

```{attention} Q&A
:class: dropdown
*Does it mean some truths are unknowable?*

- ❌. Incompleteness is relative to a fixed $T$. $T+\mathrm{Con}(T)$ proves $G_T$.
- $T+\mathrm{Con}(T)$ can be inconsistent: $T=\mathrm{PA}+\neg\mathrm{Con}(\mathrm{PA})$ is consistent, but $T\vdash\neg\mathrm{Con}(T)$. $T$ sound → $T+\mathrm{Con}(T)$ consistent.
- The stronger consistent theory has its own unprovable sentence. No single consistent, effectively axiomatized theory proves all arithmetic truths.

*Which theories escape?*

- Not effectively axiomatized: true arithmetic (complete, but its theorems are not c.e.).
- Too weak: Presburger arithmetic (addition only) → complete & decidable.
- Not arithmetic: real closed fields (Tarski) → complete & decidable.

*Natural examples of unprovable statements?*

- Goodstein's theorem: true, unprovable in PA.
- Busy beaver: ZFC cannot determine $S(n)$ for some explicit $n$, if ZFC is consistent → [Busy Beaver](#busy-beaver).

*Does Gödel show that minds are non-computable (Penrose–Lucas)?*

- Claim: a human sees that $G_T$ is true for every $T$ → the human is no $T$ → the mind is non-computable.
- Rejection: seeing $G_T$ true requires knowing that $T$ is consistent. Humans cannot verify their own consistency; humans can be inconsistent.
- Link: [Orch OR](../ac/theories.md#orchestrated-objective-reduction-theory-orch-or) builds on this argument.

*What does it tell us about AC?*

- The limits apply to any reasoner that works inside one fixed sound theory, human or machine.
- An AI can adopt stronger axioms, as humans do. No theorem shows that humans escape the limits.
```

&nbsp;

### Busy Beaver
- **What**: Max #steps of a halting $n$-state, 2-symbol TM started on a blank tape {cite:p}`rado1962noncomputable`.
- **Why**: Concrete uncomputable function → shows how fast "uncomputable" grows & measures halting difficulty per machine size.
- **How**:
    - Among all $n$-state machines, take those that halt on a blank tape → record the max step count.
    - Known $S(n)$ → decides blank-tape halting for every $n$-state machine: run $S(n)$ steps; still running → never halts.

```{note} Math
:class: dropdown
Notations:
- $S(n)$: Max step count.
- $\Sigma(n)$: Max #1s on the final tape.

Definition:

$$
S(n)=\max\{\mathrm{steps}(M,\epsilon): M\text{ has }n\text{ states \& 2 symbols},\ M(\epsilon)\downarrow\}
$$

- $\Sigma(n)\le S(n)$ ← each step writes at most one 1.
- For every computable $g$: $\Sigma(n)>g(n)$ for all sufficiently large $n$ (Radó). $S$ inherits this.
- $S(5)=47{,}176{,}870$ & $\Sigma(5)=4{,}098$, proved in Coq; $S(6)>2\uparrow\uparrow\uparrow5$ {cite:p}`bbchallenge2025determination`.
```

```{tip} Derivation
:class: dropdown
*Why can no computable function bound $S$?*

1. Assume a computable $g$ with $S(n)\le g(n)$ for all $n$.
2. Decide blank-tape halting: given $n$-state $M$, run $g(n)$ steps. Halted → yes. Else → never halts.
3. Blank-tape halting is undecidable ← $\mathrm{HALT}$ reduces to it (machine first writes $w$, then runs $M$).
4. Contradiction → $S(n)>g(n)$ for infinitely many $n$.
5. **Clarification**: Radó's theorem is stronger: $\Sigma$ exceeds every computable $g$ for ALL sufficiently large $n$.
```

```{note} Example
:class: dropdown
The 2-state champion in the [Turing Machine](#turing-machine) Code halts after 6 steps w/ 4 ones → $S(2)=6$, $\Sigma(2)=4$.
```

```{dropdown} Table: Known Values
| $n$ | $S(n)$ | $\Sigma(n)$ |
|:--|:--|:--|
| 1 | 1 | 1 |
| 2 | 6 | 4 |
| 3 | 21 | 6 |
| 4 | 107 | 13 |
| 5 | 47,176,870 | 4,098 |
| 6 | $>2\uparrow\uparrow\uparrow5$ | unknown |
```

```{attention} Q&A
:class: dropdown
*Why is $S(5)$ known but $S(6)$ far away?*

- An exact value needs a halting decision for every $n$-state machine.
- Some 6-state machines behave like Collatz-type iterations → a decision may need new mathematics.

*Can a formal theory determine every $S(n)$?*

- ❌. TM $Z$ searches for a contradiction in ZFC. ZFC consistent → $Z$ never halts, but ZFC cannot prove it (second incompleteness theorem).
- ZFC $\vdash S(n)=s$ for $n$ = #states of $Z$ → ZFC proves "$Z$ runs $s$ steps w/o halting → never halts" → ZFC $\vdash\mathrm{Con}(\mathrm{ZFC})$. Contradiction.
- First explicit size: 7,918 states (Yedidia & Aaronson, 2016), under an assumption stronger than $\mathrm{Con}(\mathrm{ZFC})$. Later work lowered it.

*How does it relate to Kolmogorov complexity?*

- Both measure what short programs can do: $S$ → max running time; $K$ → shortest description.
- Knowing $S(n)$ ⇔ knowing which $n$-state machines halt on a blank tape.
```

&nbsp;

## Algorithmic Information
- **What**: Information theory of individual objects, measured by the length of their shortest programs.

### Kolmogorov Complexity
- **What**: Length of the shortest program that outputs $x$ & halts {cite:p}`kolmogorov1965three`.
- **Why**: Shannon entropy measures distributions, not single objects.
    - Under a uniform source, "01" × 500 & a random 1000-bit string have the same probability $2^{-1000}$.
    - → Need the complexity of one object.
- **How**:
    1. Fix a universal machine $U$.
    2. Consider all programs that print $x$; take the shortest length.
    3. Regular $x$ → short loop program → low $K$. Patternless $x$ → only "print $x$" → $K\approx|x|$.

```{note} Math
:class: dropdown
Notations:
- $C(x)$: Plain Kolmogorov complexity.
- $U_{\mathrm{plain}}$: Universal machine w/o the prefix-free restriction.
- $U(p,y)$: Output of $U$ on program $p$ w/ auxiliary input $y$.

Plain complexity:

$$
C(x)=\min\{|p|:U_{\mathrm{plain}}(p)=x\}
$$

Prefix complexity (default on this page):

$$
K(x)=\min\{|p|:U(p)=x\}
$$

- $U$ is a prefix machine: no halting program is a proper prefix of another → programs are self-delimiting.
- Conditional: $K(x\mid y)=\min\{|p|:U(p,y)=x\}$.
- Pair: $K(x,y)=K(\langle x,y\rangle)$.

Properties:
- $C(x)\le|x|+O(1)$.
- $K(x)\le|x|+K(|x|)+O(1)\le|x|+2\log|x|+O(1)$.
- $C(x)\le K(x)+O(1)$ & $K(x)\le C(x)+2\log C(x)+O(1)$.
- Subadditivity: $K(x,y)\le K(x)+K(y)+O(1)$.
- Kraft: $\sum_x2^{-K(x)}\le1$.
- Upper semicomputable: a computable $g(x,t)$, nonincreasing in $t$, satisfies $\lim_{t\to\infty}g(x,t)=K(x)$.
```

```{tip} Derivation
:class: dropdown
*Why $K(x)\le|x|+2\log|x|+O(1)$?*

1. Program = self-delimiting code for $n=|x|$, then the $n$ bits of $x$, read by a fixed routine.
2. Self-delimiting code for $n$: double each bit of binary $n$, then append `01` → $2\lceil\log(n+1)\rceil+2$ bits.
3. The routine reads $n$, then exactly $n$ more bits → it knows where the program ends → prefix-free.
4. Replace step 2 by a shortest program for $n$ → $K(x)\le|x|+K(|x|)+O(1)$.

*Why is $K$ upper semicomputable?*

1. Stage $t$: run every program $p$ w/ $|p|\le t$ for $t$ steps.
2. $g(x,t)$ = min of $|x|+2\log|x|+c$ & the lengths of all programs seen to output $x$.
3. More stages → more halting programs found → $g$ only decreases → reaches $K(x)$ at some unknown stage.
```

```{note} Example
:class: dropdown
Strings of length $n=10^6$:
- $0^n$: $K\le K(n)+O(1)\le2\log n+O(1)\approx40+O(1)$ bits.
- First $n$ bits of $\pi$: $K\le K(n)+O(1)$ → low, although the bits look statistically random.
- $n$ fair coin flips: $K\ge n-c$ w/ probability $\ge1-2^{-c}$ → [Algorithmic Randomness](#algorithmic-randomness).
```

```{attention} Q&A
:class: dropdown
*Why "$+O(1)$" everywhere?*

- The choice of $U$ shifts $K$ by a bounded constant → [Invariance Theorem](#invariance-theorem).
- → Only statements up to $O(1)$ are machine-independent.

*Plain $C$ vs prefix $K$?*

- $C$: subadditivity fails; only $C(x,y)\le C(x)+C(y)+O(\log\min\{C(x),C(y)\})$ ← a plain program does not mark where $x$ ends inside $xy$.
- $K$: self-delimiting → Kraft holds → $2^{-K(x)}$ behaves like a probability → $K$ behaves like entropy.
- Prefix $K$ is the standard choice in algorithmic information theory.

*Why does simplicity help generalization (Occam bound)?*

- Countable hypothesis class, prefix-code lengths $\ell(h)$ fixed before seeing data, realizable target, $m$ i.i.d. samples.
- W/ probability $\ge1-\delta$, every $h$ consistent w/ the data has true error $\le\frac{\ell(h)\ln2+\ln(1/\delta)}{m}$.
- ← Union bound w/ budget $\delta2^{-\ell(h)}$ per $h$: a fixed $h$ w/ error $>\varepsilon$ fits $m$ samples w/ probability $\le(1-\varepsilon)^m\le e^{-\varepsilon m}$.
- $K(h)$ satisfies Kraft → a valid $\ell(h)$ in principle, but uncomputable.

*Does low $K$ mean "meaningful"?*

- ❌. $0^n$ (low $K$) & coin flips (high $K$) both carry no interesting structure.
- Meaningful structure lies between the extremes → logical depth (Bennett), sophistication.
```

&nbsp;

#### Invariance Theorem
- **What**: Changing the universal machine changes $K$ by at most a constant that does not depend on $x$.
- **Why**: W/o it, $K$ would depend arbitrarily on the programming language → no objective complexity.
- **How**: $U$ simulates any machine $V$ given the prefix $\langle V\rangle$ → every $V$-program becomes a $U$-program that is $|\langle V\rangle|$ bits longer.

```{note} Math
:class: dropdown
Notations:
- $V$: Any prefix machine.
- $K_V(x)$: Complexity of $x$ w.r.t. $V$.

$$
\forall V\ \exists c_V\ \forall x:\ K_U(x)\le K_V(x)+c_V
$$

- $U$, $V$ both universal → $|K_U(x)-K_V(x)|\le c_{U,V}$.
```

```{attention} Q&A
:class: dropdown
*Is the constant small?*

- No guarantee. $V$ can hard-code a 1 GB string as a 1-bit program → $c_{U,V}$ is huge.
- → $K$ is meaningful asymptotically & for long strings. For short strings, the choice of $U$ dominates.
- The same issue affects the Solomonoff prior & AIXI ([AIXI](#aixi)).

*Python vs Lisp?*

- An interpreter for one written in the other → constant offset.
```

&nbsp;

### Uncomputability of Kolmogorov Complexity
- **What**: No algorithm computes $K(x)$ for every $x$.
- **Why**: Limits practical use → only upper bounds via real compressors; the true complexity of data is unknowable in general.
- **How**: Berry paradox ("the smallest number not definable in under sixty letters"):
    1. Assume $K$ computable.
    2. Program $P_n$: enumerate strings in order; output the first $x$ w/ $K(x)\ge n$.
    3. $|P_n|\le K(n)+c\le2\log n+c'$ → $K(x)\le2\log n+c'<n$ for large $n$. But $K(x)\ge n$. Contradiction.

```{note} Math
:class: dropdown
- $K$ is not computable.
- Stronger: every computable $g$ w/ $g(x)\le K(x)$ for all $x$ is bounded → no useful computable lower bound.
- $K$ is computable relative to a $\mathrm{HALT}$ oracle ← the oracle tells which short programs halt.
- $K$ is upper semicomputable → compressors give valid upper bounds, never certified lower bounds.
```

```{attention} Q&A
:class: dropdown
*Does a better compressor converge to $K$?*

- Every real compressor is a fixed machine → $K(x)\le Z(x)+O(1)$ for compressed length $Z$.
- ❌ Guarantee of closeness: no computable method certifies how far $Z(x)$ is from $K(x)$.

*Why does it matter for ML?*

- MDL & compression-based model selection use computable code lengths, not $K$.
- → Results depend on the chosen code (model class), like a prior.
```

&nbsp;

#### Chaitin's Incompleteness Theorem
- **What**: A consistent, effectively axiomatized theory w/ basic arithmetic proves "$K(x)>L$" for no $x$, once $L\ge$ a constant $L_T$.
- **Why**: Quantitative incompleteness: almost all strings are complex, but $T$ certifies the complexity of none beyond $L_T$.
- **How**: Berry paradox inside $T$:
    1. Program $P$: search all $T$-proofs; output the first $x$ w/ a proof of "$K(x)>L$".
    2. $K(x)\le|P|\le K(T)+K(L)+c<L$ for large $L$ → the proved statement is false.
    3. "$K(x)>L$" false → "$K(x)\le L$" is a true halting fact → $T$ proves it (interprets $Q$) → $T$ inconsistent.

```{note} Math
:class: dropdown
Notations:
- $K(T)$: Length of a shortest program that enumerates the axioms of $T$.

For consistent $T$ that interprets $Q$:

$$
\exists L_T\ \forall x:\ T\nvdash\text{"}K(x)>L_T\text{"}
$$

- From step 2: $L_T\le K(T)+O(\log K(T))$.
- All but finitely many $x$ satisfy $K(x)>L_T$.
```

```{attention} Q&A
:class: dropdown
*Does it mean "a theory cannot prove more than its own complexity"?*

- This reading is contested. $L_T$ depends on the encoding & is not a measure of the strength of $T$ (Raatikainen).

*Can we certify that a string is random?*

- Only up to complexity $L_T$. Statistical tests can reject randomness, never certify it.
```

&nbsp;

### Algorithmic Randomness
- **What**: Randomness = incompressibility {cite:p}`martinlof1966definition`.
    - Finite $x$: $c$-incompressible iff $C(x)\ge|x|-c$.
    - Infinite sequence: Martin-Löf random iff it passes every effective statistical test.
- **Why**: Probability theory cannot call one sequence random: every $n$-bit string has probability $2^{-n}$ under fair coins.
- **How**:
    - Counting: few short programs exist → most strings are incompressible.
    - Every computable regularity (bias, pattern) → compression → a test detects it.
    - Levin–Schnorr: the test definition & the compression definition coincide.

```{note} Math
:class: dropdown
Notations:
- $\omega\in\{0,1\}^\infty$: Infinite binary sequence.
- $\omega_{1:n}$: First $n$ bits of $\omega$.
- $\lambda$: Uniform (fair-coin) measure on infinite sequences.
- $V_k$: $k$-th open set of a test.

Counting:

$$
\#\{x\in\{0,1\}^n:C(x)<n-c\}\le2^{n-c}-1
$$

- ← Fewer than $2^{n-c}$ programs are shorter than $n-c$.
- → At least $2^n-2^{n-c}+1$ strings of length $n$ are $c$-incompressible. Fraction $>1-2^{-c}$.

Martin-Löf test:
- Uniformly c.e. sequence of open sets $(V_k)$ w/ $\lambda(V_k)\le2^{-k}$.
- $\omega$ passes ⇔ $\omega\notin\bigcap_kV_k$.

Levin–Schnorr theorem:

$$
\omega\text{ is Martin-Löf random}\iff\exists c\ \forall n:\ K(\omega_{1:n})\ge n-c
$$

Properties:
- Random sequences have $\lambda$-measure 1.
- They satisfy every effective probability law (strong law of large numbers, law of the iterated logarithm).
- No computable sequence is random.
```

```{attention} Q&A
:class: dropdown
*Why prefix $K$, not plain $C$, for infinite sequences?*

- Every $\omega$ has $C(\omega_{1:n})\le n-\log n+O(1)$ for infinitely many $n$ (Martin-Löf oscillations).
- → No sequence satisfies the plain version. Prefix $K$ absorbs the length information.

*Are the digits of $\pi$ random?*

- Statistically they look random (normality is conjectured).
- Algorithmically ❌: $K(\pi_{1:n})\le K(n)+O(1)$.

*Is PRNG output random?*

- ❌ Algorithmically: $K\le K(\text{seed})+K(n)+O(1)$.
- Cryptographic pseudorandomness is a different notion: no efficient test distinguishes it from random.

*Can a fair coin produce compressible output?*

- ✅, rarely: probability of more than $c$ bits of compression $<2^{-c}$.
```

&nbsp;

### Algorithmic Probability
- **What**: Probability that $U$ outputs $x$ when its program bits are fair coin flips {cite:p}`solomonoff1964formal`.
- **Why**: Universal prior: Occam (short programs → high weight) + Epicurus (keep every explanation consistent w/ the data).
- **How**: Sum over ALL programs that print $x$, weight $2^{-|p|}$ each. The shortest program dominates (coding theorem).

```{note} Math
:class: dropdown
Notations:
- $P_U(x)$: Algorithmic probability of $x$.
- $\nu$: Lower-semicomputable semimeasure on $\{0,1\}^*$.
- $K(\nu)$: Length of a shortest program that lower-semicomputes $\nu$.

Definition:

$$
P_U(x)=\sum_{p:\,U(p)=x}2^{-|p|}
$$

- Semimeasure: $\sum_xP_U(x)=\Omega_U<1$ ← some programs never halt ([Chaitin's Omega](#chaitins-omega)).

Coding theorem (Levin; Chaitin):

$$
K(x)=-\log P_U(x)+O(1)
$$

Universality:

$$
P_U(x)\ge2^{-K(\nu)-O(1)}\,\nu(x)\quad\forall x
$$

- Every computable distribution is a special case of $\nu$.
```

```{tip} Derivation
:class: dropdown
*Why does the shortest program dominate the sum?*

1. Easy direction: the shortest program alone contributes $2^{-K(x)}$ → $P_U(x)\ge2^{-K(x)}$ → $-\log P_U(x)\le K(x)$.
2. Hard direction: $P_U$ is lower semicomputable → enumerate increasing approximations.
3. The Kraft–Chaitin theorem builds, effectively, a prefix code w/ codeword length $\lceil-\log P_U(x)\rceil+1$ for each $x$.
4. $U$ can decode that code w/ a fixed program → $K(x)\le-\log P_U(x)+O(1)$.
```

```{attention} Q&A
:class: dropdown
*Why a semimeasure & not a measure?*

- Non-halting programs leak mass.
- Normalizing by $\Omega_U$ breaks lower semicomputability ← $\Omega_U$ is uncomputable.

*Is this Bayesian?*

- ✅. $P_U$ equals, up to a multiplicative constant, a mixture of all lower-semicomputable semimeasures w/ prior weight $2^{-K(\nu)}$.
- Same regret logic as [mixture coding](info.md#universal-coding-and-description-length), w/ all computable models as candidates.

*Does the most probable explanation = the simplest one?*

- Approximately: $-\log P_U\approx K$. But $P_U$ also counts the many longer programs.
```

&nbsp;

### Chaitin's Omega
- **What**: Probability that $U$ halts when its program bits are fair coin flips {cite:p}`chaitin1975theory`.
- **Why**: Packs the halting problem into one real number → explicit random, uncomputable, but approximable real.
- **How**:
    1. Feed $U$ coin flips until it has read a whole program (prefix-free → well-defined).
    2. $\Omega_U$ = probability that this program halts.
    3. Approximate from below: dovetail all programs; add $2^{-|p|}$ each time a program $p$ halts.

```{note} Math
:class: dropdown
Definition:

$$
\Omega_U=\sum_{p:\,U(p)\downarrow}2^{-|p|}\in(0,1)
$$

- Left-c.e.: a computable increasing sequence of rationals converges to $\Omega_U$. Its convergence rate is uncomputable.
- First $n$ bits of $\Omega_U$ → decide halting for every program w/ $|p|\le n$.
- Martin-Löf random: $K(\Omega_{1:n})\ge n-O(1)$.
- A sound, effectively axiomatized theory determines only finitely many bits of $\Omega_U$.
```

```{tip} Derivation
:class: dropdown
*Why do $n$ bits of $\Omega_U$ decide halting for $|p|\le n$?*

1. Let $\omega_n=0.\Omega_1\cdots\Omega_n$ (binary truncation) → $\omega_n<\Omega_U<\omega_n+2^{-n}$. Both inequalities are strict ← $\Omega_U$ is irrational.
2. Dovetail all programs & sum $2^{-|p|}$ over the halted ones until the sum exceeds $\omega_n$. This happens in finite time.
3. Take any $|p|\le n$ that has not halted yet. If it halted later → total $\ge$ sum $+2^{-n}>\omega_n+2^{-n}>\Omega_U$. Contradiction.
4. → Every such $p$ never halts.
```

```{attention} Q&A
:class: dropdown
*Is $\Omega$ one specific number?*

- ❌. It depends on $U$.
- Every left-c.e. Martin-Löf random real is $\Omega_U$ for some universal $U$ (Kučera–Slaman).

*Why is it called an oracle?*

- Many open problems are halting questions. Goldbach: a program that searches for a counterexample halts iff Goldbach is false.
- → A fixed number of bits of $\Omega_U$ (depending on $U$) decides Goldbach.

*Is it useful in practice?*

- ❌. Computing $n$ bits needs time that grows like the busy beaver function of $n$.
```

&nbsp;

### Kolmogorov Complexity vs. Shannon Entropy
- **What**: Expected prefix complexity = Shannon entropy, up to the complexity of the distribution.
- **Why**: Bridge between per-object information ($K$) & per-distribution information ($H$) → algorithmic information generalizes Shannon.
- **How**:
    - $K$ lengths form a prefix code → on average they cannot beat entropy.
    - Coding theorem → $K(x)\lesssim-\log P(x)+K(P)$ → Shannon-optimal up to the cost of describing $P$.

```{note} Math
:class: dropdown
Notations:
- $P$: Computable distribution on $\{0,1\}^*$ w/ finite entropy.
- $H(P)$: Shannon entropy of $P$ in bits.
- $K(P)$: Length of a shortest program that computes $P$.
- $X_{1:n}$: First $n$ symbols of a stationary ergodic source.

$$
0\le\sum_xP(x)K(x)-H(P)\le K(P)+O(1)
$$

- Stationary ergodic source → $\lim_{n\to\infty}K(X_{1:n})/n$ = [entropy rate](info.md#entropy-rate), almost surely (Brudno).
```

```{tip} Derivation
:class: dropdown
1. Lower bound: $\sum_x2^{-K(x)}\le1$ → $K$ is a valid prefix-code length → expected length $\ge H(P)$ ([source coding](info.md#source-coding)).
2. Upper bound: universality → $P_U(x)\ge2^{-K(P)-O(1)}P(x)$.
3. Coding theorem → $K(x)=-\log P_U(x)+O(1)\le-\log P(x)+K(P)+O(1)$.
4. Take the expectation under $P$.
```

```{dropdown} Table: Shannon vs. Kolmogorov
| | Shannon entropy $H$ | Kolmogorov complexity $K$ |
|:--|:--|:--|
| Object | Distribution | Individual string |
| Needs a probability model | ✅ | ❌ |
| Computable | ✅ (given $P$ w/ finite support); else only lower semicomputable | ❌ (upper semicomputable) |
| Chain rule | $H(X,Y)=H(X)+H(Y\mid X)$ | $K(x,y)=K(x)+K(y\mid x,K(x))+O(1)$ |
| Mutual information | $I(X;Y)$ | $I(x:y)$ |
| Exact up to | Nothing | $O(1)$ (choice of $U$) |
```

```{attention} Q&A
:class: dropdown
*When does the gap $K(P)$ matter?*

- Complex distribution & short data → $K(P)$ dominates.
- $n$-blocks of an i.i.d. source: $K(P_n)\le K(\text{source})+K(n)+O(1)$ grows $\propto\log n$, $H(P_n)=nH(X_1)$ → per-symbol gap → 0.

*Which notion should an ML engineer use?*

- Data from a model you can write down → $H$, cross-entropy, KL.
- Single objects w/o a model (one genome, one program) → $K$ concepts, approximated by compressors ([NCD](#normalized-compression-distance)).
```

&nbsp;

### Symmetry of Information
- **What**: Chain rule for prefix Kolmogorov complexity {cite:p}`chaitin1975theory`.
- **Why**: Defines algorithmic mutual information → a symmetric measure of the information one object holds about another.
- **How**: Describe $x$, then $y$ given $x$ → the lengths add. The reverse inequality: a joint description cannot be much shorter.

```{note} Math
:class: dropdown
Chain rule:

$$
K(x,y)=K(x)+K(y\mid x,K(x))+O(1)
$$

- W/o $K(x)$ in the condition: $K(x,y)=K(x)+K(y\mid x)+O(\log K(x,y))$.

Algorithmic mutual information:

$$
I(x:y)=K(x)+K(y)-K(x,y)
$$

- $I(x:y)=K(y)-K(y\mid x,K(x))+O(1)$ → symmetric up to $O(1)$.
- Information conservation: for computable $f$, $I(f(x):y)\le I(x:y)+K(f)+O(1)$.
```

```{attention} Q&A
:class: dropdown
*Why the extra $K(x)$ in the condition?*

- $x$ alone does not reveal the length of its shortest program. That length can cost up to $O(\log|x|)$ bits.

*How does it relate to Shannon?*

- Same shape as $H(X,Y)=H(X)+H(Y\mid X)$.
- For a computable joint distribution, the expected $I(x:y)$ equals $I(X;Y)$ up to a term that depends on the complexity of the distribution.

*What does information conservation say?*

- Algorithmic analog of the [data processing inequality](info.md#data-processing-inequality).
- Computation cannot create information about $y$, beyond the bits of the procedure $f$ itself.
```

&nbsp;

### Normalized Compression Distance
- **What**: Similarity metric from compressed lengths of $x$, $y$ & their concatenation $xy$ {cite:p}`cilibrasi2005clustering`.
- **Why**: The normalized information distance (NID) is a universal similarity but uncomputable → a real compressor approximates it, w/o features.
- **How**:
    1. Compress $x$, $y$, & $xy$.
    2. Shared structure → $xy$ compresses almost as well as the larger part alone → distance ≈ 0.
    3. Unrelated → compressed $xy$ ≈ compressed $x$ + compressed $y$ → distance ≈ 1.

```{note} Math
:class: dropdown
Notations:
- $Z(x)$: Compressed length of $x$ under a real compressor (gzip, bzip2, LZMA). The paper writes $C$.

Normalized information distance:

$$
\mathrm{NID}(x,y)=\frac{\max\{K(x\mid y),K(y\mid x)\}}{\max\{K(x),K(y)\}}
$$

Normalized compression distance:

$$
\mathrm{NCD}(x,y)=\frac{Z(xy)-\min\{Z(x),Z(y)\}}{\max\{Z(x),Z(y)\}}
$$

- Numerator ← symmetry of information: $Z(xy)-Z(x)$ approximates $K(y\mid x)$.
- Range ≈ $[0,1+\varepsilon]$; $\varepsilon$ ← compressor imperfection.
```

````{important} Code
:class: dropdown
```python
import zlib

class NCD:
    def __init__(self, level=9):
        self.level = level

    def Z(self, b):
        ## compressed length = computable upper bound on K (plus format overhead)
        return len(zlib.compress(b, self.level))

    def __call__(self, x, y):
        zx, zy, zxy = self.Z(x), self.Z(y), self.Z(x + y)
        ## shared structure -> x+y compresses almost as well as the larger part alone
        return (zxy - min(zx, zy)) / max(zx, zy)

## Example
a = b"the quick brown fox jumps over the lazy dog. " * 20
b = b"the quick brown fox jumps over the lazy cat. " * 20
c = bytes(range(256)) * 4
ncd = NCD()
print(ncd(a, b) < ncd(a, c))  ## True: a & b share structure, a & c do not
```
````

```{attention} Q&A
:class: dropdown
*Pros?*

- Parameter-free & feature-free.
- Works on any data type: DNA, text, music, malware binaries.

*Cons?*

- Compressor window limits: zlib sees only 32 KB back → longer related files look unrelated.
- $O(N^2)$ compressions for $N$ objects.
- $\mathrm{NCD}(x,x)>0$ in practice ← compressor overhead.
- Slow & usually weaker than learned embeddings when training data exists.

*How does it relate to language models?*

- Better predictor → shorter arithmetic code → better compressor.
- → An LM can serve as the compressor $Z$ ([Solomonoff Induction](#solomonoff-induction)).
```

&nbsp;

## Universal Induction & Agents

### Solomonoff Induction
- **What**: Next-symbol prediction by a Bayes mixture over all programs, weighted by $2^{-|p|}$ {cite:p}`solomonoff1964formal`.
- **Why**: Formal solution to induction.
    - No hand-picked hypothesis class.
    - Fast convergence for ANY computable data source.
- **How**:
    1. Prior: every program $p$ for a monotone universal machine gets weight $2^{-|p|}$.
    2. Keep the programs whose output starts w/ the observed data.
    3. Predict the next symbol by the weighted vote of the survivors.

```{note} Math
:class: dropdown
Notations:
- $x_{<t}$: Observed bits $x_1\cdots x_{t-1}$.
- $U(p)=x*$: Monotone $U$ on program $p$ outputs a sequence that starts w/ $x$.
- $K(\mu)$: Length of a shortest program that computes $\mu$.

Universal mixture:

$$
\xi(x)=\sum_{p:\,U(p)=x*}2^{-|p|}
$$

- Sum over minimal programs: no proper prefix of $p$ also outputs $x*$.
- Monotone machine: reads program bits & writes output bits one by one; never erases output.

Prediction:

$$
\xi(b\mid x_{<t})=\frac{\xi(x_{<t}b)}{\xi(x_{<t})},\qquad b\in\{0,1\}
$$

Dominance:

$$
\xi(x)\ge2^{-K(\mu)-O(1)}\,\mu(x)\quad\forall x
$$

Convergence (Solomonoff, 1978; this form: Hutter):

$$
\sum_{t=1}^{\infty}\mathbb{E}_\mu\Big[\sum_{b\in\{0,1\}}\big(\xi(b\mid x_{<t})-\mu(b\mid x_{<t})\big)^2\Big]\le K(\mu)\ln2+O(1)
$$

- Finite total error → $\xi(\cdot\mid x_{<t})\to\mu(\cdot\mid x_{<t})$ w/ $\mu$-probability 1.
```

```{tip} Derivation
:class: dropdown
*Why is the total prediction error bounded by $K(\mu)$?*

1. Dominance → $\ln\frac{\mu(x_{1:n})}{\xi(x_{1:n})}\le(K(\mu)+c)\ln2$ for every $x_{1:n}$.
2. Take $\mathbb{E}_\mu$. Write both joint probabilities as products of conditionals.
3. → $\sum_{t=1}^{n}\mathbb{E}_\mu\big[D_t\big]\le(K(\mu)+c)\ln2$, where $D_t=\sum_b\mu(b\mid x_{<t})\ln\frac{\mu(b\mid x_{<t})}{\xi(b\mid x_{<t})}\ge0$.
4. $D_t\ge0$ also for the defective $\xi(\cdot\mid x_{<t})$ ← its total mass is $\le1$.
5. Per step: squared distance $\le D_t$ (Pinsker-type bound, valid for semimeasures).
6. Bound holds for all $n$ → the infinite sum is finite.
```

````{important} Code
:class: dropdown
```python
from itertools import product

class SolomonoffToy:
    ## Hypotheses: "repeat pattern q forever", 1 <= |q| <= max_len
    ## Prefix-free code for q: |q| in unary (|q|+1 bits) + q itself -> 2|q|+1 bits
    def __init__(self, max_len=6):
        self.hyps = ["".join(q) for n in range(1, max_len + 1) for q in product("01", repeat=n)]

    def weight(self, q):
        return 2.0 ** -(2 * len(q) + 1)  ## Occam prior 2^{-code length}

    def predict(self, x):
        ## P(next bit = 1 | x): weighted vote of hypotheses consistent with x
        votes = {"0": 0.0, "1": 0.0}
        for q in self.hyps:
            out = (q * (len(x) // len(q) + 2))[: len(x) + 1]
            if out[:-1] == x:  ## deterministic hypothesis: likelihood 1 if consistent, else 0
                votes[out[-1]] += self.weight(q)
        return votes["1"] / (votes["0"] + votes["1"])

## Example
print(round(SolomonoffToy().predict("0101"), 3))  ## 0.021: short "01" dominates, longer patterns keep a small vote
```
````

```{attention} Q&A
:class: dropdown
*Why is it uncomputable?*

- $\xi$ needs to know which programs output $x*$ → halting-type question.
- $\xi$ is only lower semicomputable. Practical stand-ins: MDL, context-tree weighting, trained sequence models.

*Does it escape the No Free Lunch theorem?*

- NFL averages uniformly over all targets.
- Solomonoff bets on computable source distributions, w/ more weight on low-$K$ ones.
- Sample paths may still be algorithmically random (fair coin: simple source, incompressible paths). The convergence bound still holds.
- ❌ Distribution-free guarantee over all targets.

*How does it relate to LLM pretraining?*

- Next-token log loss = code length under arithmetic coding → minimizing log loss = compression.
- Evidence: Chinchilla 70B compresses ImageNet patches to 43.4% & LibriSpeech samples to 16.4% of raw size; PNG gets 58.5% & FLAC 30.3% {cite:p}`deletang2024language`.
- These rates exclude the model's own size. Counting parameters → large LMs compress small datasets poorly.
- "LLM ≈ Solomonoff" is an analogy, not a theorem: finite context, finite compute, a learned (not universal) prior.

*What breaks w/ a different $U$?*

- Bounds stay valid; constants change by $O(1)$ in $K(\mu)$.
- Finite-data predictions depend on $U$ → the choice of $U$ is a prior choice.
```

&nbsp;

### Levin Search
- **What**: Run all programs in parallel, program $p$ w/ time share $\propto2^{-|p|}$; stop at the first verified solution {cite:p}`levin1973universal`.
- **Why**: Inversion problems (find $w$ w/ $\mathrm{check}(w)=1$): optimal up to a constant factor $2^{|p|}$ against every solver $p$.
- **How**:
    1. Phase $i=1,2,\ldots$: run each program $p$ w/ $|p|\le i$ for $2^{i-|p|}$ steps.
    2. Verify every output. Correct → stop.
    3. Else $i\leftarrow i+1$ & restart. Doubling budgets → restarts cost only a constant factor.

```{note} Math
:class: dropdown
Notations:
- $t_p$: Steps that program $p$ needs to output a valid $w$, verification included.

Phase cost (prefix-free programs, Kraft):

$$
\sum_{|p|\le i}2^{i-|p|}=2^i\sum_{|p|\le i}2^{-|p|}\le2^i
$$

- Solver $p$ succeeds by phase $i^*=|p|+\lceil\log t_p\rceil$.
- Total time $\le\sum_{i\le i^*}2^i<2^{i^*+1}\le4\cdot2^{|p|}t_p$.

$$
T_{\mathrm{Levin}}\le4\min_p2^{|p|}\,t_p
$$

- Ignores the constant overhead of simulating programs on $U$.
- Kt complexity: $Kt(x)=\min_p\{|p|+\log t(p,x)\}$ = the cost that Levin search optimizes. $t(p,x)$: steps for $p$ to output $x$.
```

````{important} Code
:class: dropdown
```python
from itertools import product

class LevinSearch:
    def __init__(self, run, check):
        self.run, self.check = run, check  ## run(p, budget) -> output, or None if no halt within budget

    def __call__(self, max_phase=20):
        for i in range(1, max_phase + 1):
            for n in range(1, i + 1):
                for bits in product("01", repeat=n):
                    p = "".join(bits)
                    out = self.run(p, budget=2 ** (i - n))  ## time share 2^{i-|p|}
                    if out is not None and self.check(out):
                        return p, out, i
        return None

## Toy language: '0' -> x+=1, '10' -> x*=2, '11' -> halt; 1 instruction = 1 step
## Valid programs match (0|10)*11 -> prefix-free; other strings are rejected
def run(p, budget):
    x, k, steps = 0, 0, 0
    while k < len(p):
        if steps == budget:
            return None
        if p[k] == "0":
            x, k = x + 1, k + 1
        elif p[k:k + 2] == "10":
            x, k = 2 * x, k + 2
        elif p[k:k + 2] == "11":
            return x if k + 2 == len(p) else None  ## halt must be the last instruction
        else:
            return None
        steps += 1
    return None

## Example: find a program that outputs 6
print(LevinSearch(run, check=lambda x: x == 6)())  ## ('0001011', 6, 10): 1+1+1, *2, halt
```
````

```{attention} Q&A
:class: dropdown
*Why is it not used in practice?*

- The constant $2^{|p|}$ is astronomical for real solvers ($|p|$ ≈ thousands of bits).

*How does it relate to P vs NP?*

- For NP search problems w/ a polynomial-time verifier: if any program solves them in polynomial time, Levin search does too, w/ a huge constant.
- → An optimal algorithm exists. Its running time is unknown.

*Successors?*

- Hutter's "fastest & shortest algorithm": $T(x)\le5t_p(x)+d_p\,\mathrm{time}_{t_p}(x)+c_p$ for every $p$ w/ provable correctness & a provable time bound $t_p$.
    - $\mathrm{time}_{t_p}(x)$: time to compute the bound $t_p(x)$. $d_p,c_p$: constants, $c_p$ huge.
- Optimal Ordered Problem Solver (Schmidhuber): reuses earlier solutions as cheap prefixes.
```

&nbsp;

### AIXI
- **Name**: AI$\xi$: AI w/ the universal mixture $\xi$ as environment model {cite:p}`hutter2000theory`
- **What**: Expectimax planning over a Solomonoff mixture of all computable environments.
- **Why**: Formal definition of an optimal general agent w/o domain knowledge → reference point for general RL & ASI theory.
- **How**:
    1. Belief: weight every environment program $q$ by $2^{-|q|}$; keep those consistent w/ the history.
    2. Plan: expectimax over future actions & percepts up to the horizon.
    3. Act: take the first action of the best plan; observe; update; repeat.

```{note} Math
:class: dropdown
Notations:
- $a_t$: Action at step $t$.
- $o_t$: Observation at step $t$.
- $r_t\in[0,1]$: Reward at step $t$.
- $N$: Horizon (last step).
- $q$: Environment program.
- $U(q,a_{1:N})$: Percept sequence that $q$ outputs, given the action sequence.
    - $q$ is chronological: outputs $o_tr_t$ before reading $a_{t+1}$ → percepts never depend on future actions.

Action selection at step $t$:

$$
a_t=\arg\max_{a_t}\sum_{o_tr_t}\cdots\max_{a_N}\sum_{o_Nr_N}\big[r_t+\cdots+r_N\big]\sum_{q:\,U(q,a_{1:N})=o_1r_1\cdots o_Nr_N}2^{-|q|}
$$

- Past actions & percepts are fixed by the history → the inner sum counts only consistent $q$.
- Hutter writes $\ell(q)$ for $|q|$ & $m$ for $N$.
- AIXI$tl$: computable variant w/ bounds $t$ (time) & $l$ (program length); time per cycle $O(t\cdot2^l)$.
```

```{attention} Q&A
:class: dropdown
*Pros?*

- No parameters beyond the choice of $U$ & the horizon (or discount).
- Bayes-optimal w.r.t. $\xi$ by construction.
- Clean decomposition: intelligence = universal prediction + planning.

*Cons?*

- Incomputable ← $\xi$ & expectimax over infinitely many programs.
- Depends on $U$: a "dogmatic" universal prior makes AIXI follow any fixed computable policy; every policy is Pareto optimal over all computable environments {cite:p}`leike2015bad`.
- Exploration: traps can stop learning → ❌ asymptotic optimality in general environments.
- Dualistic: the agent sits outside the environment → ❌ models of its own computation, damage, or self-modification.

*Is AIXI how LLM agents work?*

- Partial analogy: LM ≈ learned approximate prior over sequences; agent scaffold ≈ shallow planning.
- No theorem links them.

*What does it tell us about intelligence & AC?*

- AIXI defines intelligence purely by behavior (reward).
- It says nothing about experience → contrast w/ AC theories based on structure ([IIT](../ac/theories.md#integrated-information-theory-iit)).
```

&nbsp;

#### Universal Intelligence
- **What**: Expected total reward of an agent over all computable environments, weighted by $2^{-K(\text{environment})}$ {cite:p}`legg2007universal`.
- **Why**: Formal definition of machine intelligence: the ability to achieve goals in a wide range of environments.
- **How**: Score policy $\pi$ in every environment → weight simple environments more → sum.

```{note} Math
:class: dropdown
Notations:
- $\pi$: Agent policy.
- $\mathcal{E}$: Computable environments w/ total reward $\le1$.
- $V_\mu^\pi$: Expected total reward of $\pi$ in environment $\mu$.

$$
\Upsilon(\pi)=\sum_{\mu\in\mathcal{E}}2^{-K(\mu)}V_\mu^\pi
$$

- AIXI maximizes $\Upsilon$ when both use the same $U$ & the same lifetime-reward objective (Legg & Hutter). Finite horizon $N$ → objectives differ.
```

```{attention} Q&A
:class: dropdown
*Cons?*

- Uncomputable ← $K$ & a sum over all environments.
- Depends on $U$: a different universal prior can rank AIXI near the minimum score.
- $2^{-K}$ weighting favors simple environments → a fixed choice of task distribution.

*How does it relate to benchmarks?*

- A benchmark = a finite, hand-picked sample of environments w/ hand-picked weights.
- $\Upsilon$ = the idealized limit w/ a universal weighting.
```

&nbsp;
