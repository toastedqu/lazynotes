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
# Theories
This page is my interpretation of computational theories of consciousness from {cite:t}`ding2023survey`.
RPT & PP come from the theory families in {cite:t}`seth2022theories`. AI indicator properties come from {cite:t}`butlin2023consciousness`.

Notations:
- $\mathcal{X}$: Whole system (set of units, e.g., neurons).
- $\mathcal{S}\subseteq\mathcal{X}$: Candidate subsystem.
- $\{\mathcal{A},\mathcal{B}\}$: Bipartition of $\mathcal{S}$ ($\mathcal{B}=\mathcal{S}\setminus\mathcal{A}$).
- $X_\mathcal{A}$: State of the units in $\mathcal{A}$ (random vector).
- $H(\cdot)$: Shannon entropy (bits).
- $I(\cdot;\cdot)$: Mutual information (bits).
- $\Phi$: Integrated information.
- $\hbar$: Reduced Planck constant.

&nbsp;

## Integrated Information Theory (IIT)
- **What**: Consciousness = a system's maximally irreducible intrinsic cause–effect structure {cite:p}`tononi2004information,albantakis2023integrated`.
    - **Intrinsic**: Cause–effect power of the system on itself, not as seen by an outside observer.
    - **Irreducible**: Any cut into parts loses cause–effect power → whole > sum of parts.
    - **Maximally**: Of overlapping candidate subsystems, only the one w/ max integrated information ($\varphi_s$ in IIT 4.0) is conscious (the **complex**).
    - Quantity of consciousness = $\Phi$. Quality = shape of the cause–effect structure.
- **Why**: Start from experience, not from the brain.
    - Neural-correlate search finds *where* consciousness is, not *why* ← no principle that says which physical systems are conscious.
    - IIT: properties every experience has (axioms) → properties its substrate must have (postulates) → measure them in any system.
- **How**:
    1. Axioms: Experience exists & is intrinsic, specific, unitary, definite, structured.
    2. Postulates: Translate each axiom into a property of physical cause–effect power.
    3. Measure: Perturb the system → find the cut that loses the least cause–effect power → $\Phi$ = power lost at that cut.
    4. Exclude: Of overlapping candidates, max integrated information wins → complex.
        - Repeat on the remaining units → a set of disjoint complexes (IIT 3.0/4.0).

```{note} Math
:class: dropdown
IIT 1.0 version (computable intuition).

Notations:
- Misc:
    - $\mathcal{A}^{H^\text{max}}$: $\mathcal{A}$ w/ its outputs replaced by independent max-entropy noise sources.
    - $H^\text{max}(\mathcal{A})$: Max possible entropy of $X_\mathcal{A}$.

Process:
1. Effective information (directed):

    $$
    \text{EI}(\mathcal{A}\to\mathcal{B})=I(X_{\mathcal{A}^{H^\text{max}}};X_\mathcal{B})
    $$

2. Effective information across the cut:

    $$
    \text{EI}(\mathcal{A}\rightleftarrows\mathcal{B})=\text{EI}(\mathcal{A}\to\mathcal{B})+\text{EI}(\mathcal{B}\to\mathcal{A})
    $$

3. Minimum information bipartition (weakest link):

    $$
    \text{MIB}(\mathcal{S})=\arg\min_{\{\mathcal{A},\mathcal{B}\}}\frac{\text{EI}(\mathcal{A}\rightleftarrows\mathcal{B})}{\min\{H^\text{max}(\mathcal{A}),H^\text{max}(\mathcal{B})\}}
    $$

4. Integrated information = **non-normalized** EI at the MIB:

    $$
    \Phi(\mathcal{S})=\text{EI}(\text{MIB}(\mathcal{S}))
    $$

5. Complex: $\mathcal{S}$ w/ $\Phi(\mathcal{S})>0$ that is not inside a larger subset w/ higher $\Phi$. Main complex = complex w/ max $\Phi$.

Properties:
- $\text{EI}(\mathcal{A}\to\mathcal{B})\neq\text{EI}(\mathcal{B}\to\mathcal{A})$ in general, but $I(X_\mathcal{A};X_\mathcal{B})=I(X_\mathcal{B};X_\mathcal{A})$.
- $\text{EI}(\mathcal{A}\to\mathcal{B})\leq\min\{H^\text{max}(\mathcal{A}),H^\text{max}(\mathcal{B})\}$.
- #bipartitions of $\mathcal{S}$: $2^{|\mathcal{S}|-1}-1$.
```

```{note} Example
:class: dropdown
*Why is a brain conscious but a camera is not?*

- **Photodiode**: 2 states (light/dark) → 1 bit → tiny repertoire → $\Phi$ ≈ 0.
- **1-megapixel camera**: $2^{1{,}000{,}000}$ states → huge information.
    - BUT pixels do not interact → cut the chip into single pixels → nothing lost → $\Phi=0$.
- **Brain**: Huge repertoire **AND** parts causally interdependent → cut it anywhere → information lost → $\Phi$ high.
- → Consciousness needs differentiation (many states) **AND** integration (one whole).
```

```{dropdown} Table: Axioms → Postulates (IIT 4.0)
| Axiom (experience is …) | Postulate (substrate's cause–effect power must be …) |
|:--|:--|
| Exists (0th axiom) | Physical: it takes & makes a difference |
| Intrinsic: for itself | Intrinsic: takes & makes a difference within itself |
| Specific: this one | Specific: selects 1 cause–effect state (max intrinsic information) |
| Unitary: one whole | Unitary: irreducible ($\varphi>0$ over its minimum partition) |
| Definite: this whole | Definite: the unit set w/ max $\varphi$ (= complex) |
| Structured: distinctions & relations | Structured: subsets of units specify distinctions & relations |

- $\varphi$: Integrated information of a system or of one distinction/relation.
- In IIT 4.0, $\Phi$ = sum of $\varphi$ over all distinctions & relations of the complex ("structure integrated information").
```

```{attention} Q&A
:class: dropdown
*Pros?*

- Only major theory w/ a quantitative measure → graded consciousness & predictions for any system.
- Explains clinical facts:
    - Cerebellum has more neurons than cortex, but its loss does not remove consciousness ← many quasi-independent modules → low $\Phi$.
    - Dreamless sleep & anesthesia → cortical integration breaks down → $\Phi$⬇️.

*Cons?*

- Intractable: search over all subsets & partitions → exact $\Phi$ only for toy systems.
- Hard to falsify: axioms are claimed self-evident; $\Phi$ of a real brain cannot be computed.
    - 2023 open letter (100+ researchers) called IIT "pseudoscience". Others defend it. Status: contested.
- Counterintuitive: a large, regular grid of simple logic gates can have high $\Phi$ → "highly conscious". IIT accepts this.

*Which $\Phi$?*

- IIT 1.0 (2004): EI at the MIB (Math block).
- IIT 2.0 (2008): state-dependent; discrete systems.
- IIT 3.0 (2014): cause–effect repertoires; $\Phi$ over the conceptual structure.
- IIT 4.0 (2023): intrinsic information; system $\varphi_s$; $\Phi$ = sum over distinctions & relations.
- → $\Phi$ values are ❌ comparable across versions.

*Why normalize in the MIB search, but report the raw EI?*

- EI across a cut ≤ max entropy of the smaller part → raw minimum always cuts off 1 unit → trivial weakest link.
- Normalization → compare cuts fairly → find the true weakest link.
- Raw EI at that cut → $\Phi$ in bits.

*Why inject max-entropy noise instead of using observed correlations?*

- Correlation ≠ causation: 2 units w/ a common input correlate w/o affecting each other.
- Replace $\mathcal{A}$'s outputs w/ independent noise → $\mathcal{B}$ cannot affect $\mathcal{A}$ → any shared information must flow $\mathcal{A}\to\mathcal{B}$.

*Is a feedforward network conscious under IIT?*

- ❌. Feedforward → the input layer gets no input from the rest → cut there loses nothing → $\varphi_s=0$.
- Any recurrent system's I/O behavior can be unrolled into a feedforward one → same behavior, no consciousness.
- → IIT is **not** functionalist. Behavior proves nothing.

*Can an LLM be conscious under IIT?*

- $\Phi$ depends on the physical causal structure of the hardware, not on the simulated computation.
- Standard digital computers → ≈ no consciousness, even if they simulate a brain perfectly (IIT 4.0 claim).
- → AC under IIT = build hardware w/ the right intrinsic causal structure (e.g., neuromorphic), not better software.

*How is IIT tested?*

- PCI (perturbational complexity index): TMS pulse → EEG response → compressibility.
    - Integrated & differentiated response → high PCI. Proxy inspired by IIT, not $\Phi$ itself.
- Adversarial test vs GNWT → see [GNWT](#gnwt).
```

&nbsp;

## Consciousness as a State of Matter
- **What**: Consciousness = a phase of matter ("perceptronium") defined by its information-processing properties {cite:p}`tegmark2015consciousness`.
- **Why**: Pattern, not substance.
    - Solid vs liquid vs gas ← arrangement of atoms, not type of atoms → consciousness may also be a pattern.
    - Physics needs it: **quantum factorization problem** → why do observers split Hilbert space into the objects & basis we perceive?
- **How**: State necessary (not sufficient) principles → test each in classical & quantum systems → find conflicts.
    - **Information**: Substantial information storage capacity.
    - **Dynamics**: Substantial information processing capacity.
    - **Independence**: Substantial independence from the rest of the world.
    - **Integration**: Cannot consist of nearly independent parts.
    - **Autonomy**: Dynamics + independence together.
    - **Utility**: Evolved system records mainly useful information (evolutionary explanation, not a necessary condition).

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\rho$: Density matrix of the system.
    - $\rho_1,\rho_2$: Reduced density matrices of the 2 parts ($\rho_1=\text{tr}_2\,\rho$).
    - $S(\rho)=-\text{tr}\,\rho\log_2\rho$: Von Neumann entropy.
    - $\mathbf{U}$: Unitary (change of basis).
    - $\mathbf{H}$: Hamiltonian (bold ≠ entropy $H$).
    - $\mathbf{I}$: Identity operator.

Integration (cruelest cut = the cut that minimizes MI):
- Classical:

    $$
    \Phi=\min_{\{\mathcal{A},\mathcal{B}\}}I(X_\mathcal{A};X_\mathcal{B})
    $$

- Quantum (all factorizations allowed):

    $$
    \Phi=\min_{\mathbf{U}}I(\mathbf{U}\rho\mathbf{U}^\dagger),\quad I(\rho)=S(\rho_1)+S(\rho_2)-S(\rho)
    $$

    - $I(\rho)$: Quantum MI across a fixed bipartition.

Independence:

$$
\mathbf{H}=\mathbf{H}_1\otimes\mathbf{I}+\mathbf{I}\otimes\mathbf{H}_2+\mathbf{H}_3
$$
- $\mathbf{H}_1,\mathbf{H}_2$: Internal Hamiltonians of the 2 parts.
- $\mathbf{H}_3$: Interaction. $\mathbf{H}_3=0$ → perfectly independent parts.

Autonomy:

$$
\begin{align*}
\delta H&=\sqrt{\text{tr}\left[\mathbf{H}^2\rho^2-\mathbf{H}\rho\mathbf{H}\rho\right]}\\
\tau_\text{dyn}&=\frac{\hbar}{\delta H},\quad\tau_\text{ind}=\left[\ddot{S}^\text{lin}_1(0)\right]^{-1/2},\quad\text{Autonomy}=\frac{\tau_\text{ind}}{\tau_\text{dyn}}
\end{align*}
$$
- $\delta H$: Energy coherence (= energy uncertainty for a pure state).
- $\tau_\text{dyn}$: Time for 1 elementary operation (e.g., a bit flip).
- $S^\text{lin}_1=1-\text{tr}\,\rho_1^2$: Linear entropy of part 1 → measures information exchange w/ the rest.
- $\tau_\text{ind}$: Time until significant information exchange w/ the environment.
```

````{important} Code
:class: dropdown
```python
import itertools
import torch

class CruelestCutPhi:
    """Classical Phi for a Gaussian system: MI across the bipartition that minimizes it."""
    def __call__(self, cov):
        n = cov.shape[0]
        best, best_cut = float("inf"), None
        ## fix unit 0 in part A → each unordered bipartition counted once (2^(n-1) - 1 cuts)
        for k in range(n - 1):
            for rest in itertools.combinations(range(1, n), k):
                a = [0, *rest]
                b = [j for j in range(n) if j not in a]
                ## Gaussian MI: 1/2 log2(|Σ_A||Σ_B| / |Σ|)
                mi = 0.5 * (torch.logdet(cov[a][:, a]) + torch.logdet(cov[b][:, b])
                            - torch.logdet(cov)) / torch.log(torch.tensor(2.0))
                if mi.item() < best:
                    best, best_cut = mi.item(), (a, b)
        return best, best_cut

## Example: 2 tightly coupled pairs {0,1} & {2,3}, weak link 1–2
cov = torch.eye(4)
cov[0, 1] = cov[1, 0] = 0.9
cov[2, 3] = cov[3, 2] = 0.9
cov[1, 2] = cov[2, 1] = 0.1
phi, cut = CruelestCutPhi()(cov)
print(round(phi, 3), cut)  ## 0.234 ([0, 1], [2, 3]) → the cruelest cut is the weak link
```
````

```{attention} Q&A
:class: dropdown
*What is the integration paradox?*

- Hopfield net w/ Hebbian learning: #attractors ≤ $0.14n$ → capacity ≈ $\log_2 n$ ≈ 37 bits for $n=10^{11}$ neurons.
- → $\Phi$ ≤ ~37 bits. BUT a conscious experience seems to hold far more information.
- Error-correcting (random) codes reach ~$n/2$ bits → maybe the brain uses better codes than Hopfield nets.

*What is the quantum integration paradox?*

- Minimize over **all** factorizations → $\Phi$ of any quantum state ≤ ~0.25 bit, for any system size.
- e.g., Bell pair: maximally entangled, but rotate the basis so it becomes a product state → $\Phi=0$.
- → Integration of a static state cannot be enough → need dynamics (information processing, not storage).

*What is the quantum Zeno paradox?*

- Maximize independence (minimize $\|\mathbf{H}_3\|$) → best split is in the energy eigenbasis.
- In that split, the parts do not evolve → "all change grinds to a halt".
- → Independence alone fails → balance it w/ dynamics → **autonomy** principle.
- In Tegmark's toy model, autonomy grows exponentially w/ #qubits.

*How does this $\Phi$ differ from IIT's?*

- MI (correlation), not EI (causation via perturbation).
- No normalization.
- Defined for any quantum system.

*Pros?*

- Physics-grounded & quantitative.
- Substrate-independent: "consciousness is the way information feels when being processed in certain complex ways" (Tegmark).

*Cons?*

- Only necessary conditions → cannot say what **is** conscious.
- Does not address why the pattern feels like anything.

*AC angle?*

- Pattern, not substrate → silicon can qualify in principle.
- Sufficiency unknown → no verdict on any specific AI system.
```

&nbsp;

## Orchestrated Objective Reduction Theory (Orch OR)
- **What**: Consciousness = discrete quantum-state collapses in neuronal microtubules, orchestrated by biology {cite:p}`hameroff1996orchestrated`.
- **Why**:
    - *Why do we need it?* Penrose: human mathematical understanding is non-computable (Gödel-based argument).
        - → No algorithm can be conscious → need non-computable physics.
    - *Why does it work?* Objective reduction (OR) = Penrose's candidate for a physical process that is neither algorithmic nor random.
        - → Can host non-computable understanding & free will (claim).
- **How**:
    1. Tubulin proteins in microtubules enter quantum superposition → quantum computation.
    2. Synaptic inputs & microtubule-associated proteins "orchestrate" (tune) the superposition.
    3. Superposition grows until it reaches the OR threshold → collapse after $\tau\approx\hbar/E_G$ (see [Diósi–Penrose OR](#dp-or)).
    4. Each OR = 1 conscious moment. Its outcome sets tubulin states → regulates neuron firing → behavior.
    5. Stream of consciousness = sequence of ORs. Hameroff & Penrose link it to gamma-band (~40 Hz) EEG.

```{attention} Q&A
:class: dropdown
*Pros?*

- Addresses non-computability & free will directly.
- Makes specific physical claims → falsifiable.

*Cons?*

- Decoherence: brain = warm, wet, noisy.
    - Estimated decoherence ~$10^{-13}$–$10^{-20}$ s ≪ neural dynamics ~$10^{-3}$–$10^{-1}$ s {cite:p}`tegmark2000importance`.
    - Orch OR proponents recomputed longer times. Contested.
- Gödel argument: most logicians & philosophers reject it ← it assumes humans can know their own reasoning is consistent.
- No direct evidence of orchestrated quantum coherence in microtubules at the required scale & duration.
- The parameter-free version of the underlying collapse model is ruled out → see [Diósi–Penrose OR](#dp-or).

*AC angle?*

- Any classical computer (incl. LLMs) ❌ ← it runs an algorithm.
- AC would need hardware that hosts orchestrated OR events.
```

&nbsp;

(dp-or)=
### Diósi–Penrose Objective Reduction
- **What**: Gravity-induced collapse: a superposition of 2 mass distributions decays spontaneously after $\tau\approx\hbar/E_G$ {cite:p}`diosi1989models,penrose1996gravity`.
- **Why**: Superposed masses → superposed spacetime geometries → ill-defined → nature must select one.
    - Also solves the measurement problem w/o a conscious observer.
- **How**: Larger mass separation → larger $E_G$ → faster collapse.
    - Electron → ~never collapses on its own. Macroscopic object → collapses almost instantly.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $E_G$: Gravitational self-energy of the difference between the 2 superposed mass distributions.
    - $\tau$: Mean lifetime of the superposition.

Definition:

$$
\tau\approx\frac{\hbar}{E_G}
$$

- Order of magnitude: $E_G\sim Gm^2/a$ for mass $m$ of size $a$, displaced by ≳ its own size.
    - $G$: Gravitational constant.
- Collapse is random in time (mean $\tau$).
```

```{attention} Q&A
:class: dropdown
*Has it been tested?*

- Collapse → random diffusion of particles → charged particles emit faint radiation.
- Gran Sasso underground experiment found no such excess → parameter-free version ❌ {cite:p}`donadi2021underground`.
- Versions w/ a larger free mass-density size parameter survive.
- → Orch OR is not refuted, but its quantitative base is weaker.
```

&nbsp;

## Global Workspace Theory (GWT)
- **What**: Consciousness = global broadcast of 1 selected content from a limited-capacity workspace to many unconscious specialist processors {cite:p}`baars1988cognitive`.
- **Why**: Brain = many parallel, unconscious specialists. Conscious experience = serial, limited, unified.
    - → Need a mechanism that selects 1 content & shares it w/ all specialists.
- **How**: Theater metaphor.
    - **Stage**: Working memory (limited capacity).
    - **Spotlight**: Attention → selects the conscious content on stage.
    - **Actors**: Candidate contents (percepts, thoughts) → compete for the spotlight.
    - **Director & context**: Goals & unconscious contexts → bias the competition.
    - **Audience**: Unconscious specialists → receive the broadcast → respond.
    - Cycle: Specialists compete → winner enters the workspace (= conscious) → broadcast → specialists respond → next competition.

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class GlobalWorkspace(nn.Module):
    def __init__(self, n_modules, d):
        super().__init__()
        ## specialists = scaffolding; the concept is competition + broadcast
        self.specialists = nn.ModuleList([nn.Linear(d, d) for _ in range(n_modules)])
        self.salience = nn.Linear(d, 1)

    def forward(self, h):                                   ## h: (n_modules, d) private states
        proposals = torch.stack([f(x) for f, x in zip(self.specialists, h)])  ## (n_modules, d)
        scores = self.salience(proposals).squeeze(-1)       ## (n_modules,)
        ## competition: limited capacity → exactly 1 content wins the stage
        winner = scores.argmax()
        content = proposals[winner]                         ## (d,)
        ## broadcast: every specialist receives the same content
        h = h + content
        return h, winner.item()

## Example
gw = GlobalWorkspace(n_modules=3, d=4)
h, winner = gw(torch.randn(3, 4))
print(h.shape, winner)  ## torch.Size([3, 4]) <index of the winning module>
```
````

```{attention} Q&A
:class: dropdown
*Pros?*

- Explains limited capacity, seriality, & why conscious content is reportable & usable by any system.
- Functionalist & computational → directly implementable (e.g., LIDA cognitive architecture).

*Cons?*

- Explains **access**, not **phenomenal** experience → the hard problem stays.
- "Workspace" & "broadcast" are metaphors in the original theory → GNWT adds neural details.

*Conscious vs unconscious processing?*

- Unconscious = local processing inside specialists.
- Conscious = the same content made globally available.
- e.g., Subliminal word → processed & primes behavior, but never broadcast → not reported.

*Does a transformer have a global workspace?*

- Pro: residual stream ≈ broadcast ← every layer reads from & writes to it.
- Con (Butlin et al.):
    - Residual width = input width → no real limited-capacity bottleneck.
    - No recurrence → modules do not get the broadcast back.
    - No distinct workspace separate from the modules.
- → Weak case. Perceiver (latent bottleneck + cross-attention) is closer, but still short.

*AC angle?*

- Functionalist → if GWT is correct, any system w/ the architecture is conscious → AC = engineering problem.
- Butlin indicators: parallel specialized modules (GWT-1), limited-capacity workspace w/ selective attention (GWT-2), global broadcast (GWT-3), state-dependent attention to query modules in sequence (GWT-4).
```

&nbsp;

(gnwt)=
### Global Neuronal Workspace Theory (GNWT)
- **What**: Neural version of GWT: workspace = heavily interconnected cortical neurons w/ long-range axons (esp. prefrontal) that ignite & broadcast {cite:p}`dehaene1998neuronal`.
- **Why**: GWT is a metaphor → need falsifiable neural predictions.
- **How**:
    1. Stimulus climbs the cortical hierarchy bottom-up (~100–300 ms) → unconscious.
    2. If strong enough & relevant to goals → **ignition**: sudden, nonlinear, self-sustaining reverberation in workspace neurons.
    3. Ignited content → broadcast to all processors → reportable.
    4. Only 1 representation active at a time; competitors are inhibited.

```{attention} Q&A
:class: dropdown
*Neural signatures of conscious access?*

- Late (≳300 ms), all-or-none amplification (ignition).
- P3b wave in EEG.
- Long-range synchronization between distant cortical areas.

*Evidence?*

- Masking & attentional blink: same stimulus, seen vs unseen → only seen trials show ignition.

*How did GNWT & IIT do in a direct test?*

- Adversarial collaboration: preregistered divergent predictions, $n=256$, fMRI + MEG + iEEG {cite:p}`cogitate2025adversarial`.
- Both theories challenged:
    - GNWT: general lack of ignition at stimulus offset; some conscious dimensions poorly represented in prefrontal cortex.
    - IIT: no sustained synchronization within posterior cortex.

*Cons?*

- Prefrontal activity may reflect the **report**, not the experience → no-report paradigms weaken it.
- Like GWT: explains access, not phenomenal experience.
```

&nbsp;

## Higher-Order Theory (HOT)
- **What**: A mental state is conscious iff one is suitably aware of being in it, via a higher-order representation {cite:p}`carruthers2026higher`.
    - **First-order state**: Represents the world (e.g., seeing red).
    - **Higher-order state**: Represents a first-order state (e.g., "I am seeing red").
- **Why**: Unconscious perception exists (blindsight, masked priming).
    - → First-order representation alone ≠ consciousness.
    - → Extra ingredient = the mind represents its own state.
- **How**: Variants differ in the form of the higher-order state & its relation to the first-order state.
    - **HOPT**: Perception-like (inner sense).
    - **HOTT**: Thought-like.
    - **SRT**: The state represents itself.

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class HigherOrderMonitor(nn.Module):
    """Metacognitive monitoring (Butlin HOT-2) only; no belief/action update from it (HOT-3)."""
    def __init__(self, d_in, d_hidden, n_classes):
        super().__init__()
        ## first-order network: represents the world
        self.first = nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU())
        self.head = nn.Linear(d_hidden, n_classes)
        ## higher-order network: input = the first-order REPRESENTATION, not the world
        self.second = nn.Linear(d_hidden, 1)

    def forward(self, x):
        h = self.first(x)                                    ## (B, d_hidden) first-order state
        logits = self.head(h)                                ## (B, n_classes) first-order judgment
        ## detach: the monitor reads the state but does not reshape it
        p_reliable = torch.sigmoid(self.second(h.detach())).squeeze(-1)  ## (B,)
        return logits, p_reliable

## Example
m = HigherOrderMonitor(d_in=4, d_hidden=8, n_classes=3)
x, y = torch.randn(2, 4), torch.tensor([0, 2])
logits, p = m(x)
## train the monitor to predict whether the first-order judgment is correct (confidence)
target = (logits.argmax(-1) == y).float()
loss = F.binary_cross_entropy(p, target)
print(logits.shape, p.shape)  ## torch.Size([2, 3]) torch.Size([2])
```
````

```{attention} Q&A
:class: dropdown
*Pros?*

- Explains conscious vs unconscious versions of the same content (e.g., blindsight).
- Links consciousness to metacognition → measurable (confidence ratings, wagering).

*Cons?*

- **Rock problem**: Thinking about a rock does not make the rock conscious → why does thinking about a mental state?
    - Reply: HOT applies only to mental states.
- **Empty HOT**: Higher-order state w/o a real target → what is experienced?
    - Rosenthal: the higher-order content fixes what it is like → you can seem to be in a state you are not in.
- **Too demanding**: Infants & many animals may lack higher-order states → not conscious?

*How does HOT differ from GWT?*

- GWT: conscious = globally available first-order content.
- HOT: conscious = meta-represented content.
- → A broadcast w/o meta-representation is conscious under GWT, not under HOT.

*AC angle?*

- Functionalist → an AI w/ the right metacognitive monitoring can qualify.
- Butlin indicators: generative/noisy perception (HOT-1), metacognitive monitoring that separates reliable percepts from noise (HOT-2), agency that updates beliefs from that monitoring (HOT-3), sparse & smooth "quality space" coding (HOT-4).
- LLM verbal self-reports ≠ higher-order representations ← they can copy human text about experience.
```

&nbsp;

### Higher-Order Perception Theory (HOPT)
- **What**: Conscious state = first-order state that is the target of a perception-like higher-order state from an "inner sense" {cite:p}`armstrong1968materialist`.
- **Why**: Experience is fine-grained & non-conceptual → a perception-like (non-conceptual) higher-order state can match that richness.
- **How**: Inner sense scans first-order states → produces equally fine-grained higher-order representations of them.

```{attention} Q&A
:class: dropdown
*Cons?*

- No known organ or mechanism for an "inner sense".
- Outer senses have their own phenomenal character; inner sense seems to have none.
- Attention to first-order states may do the same job → higher-order perception unnecessary.
```

&nbsp;

### Higher-Order Thought Theory (HOTT)
- **What**: Conscious state = state that causes (or is disposed to cause) a non-inferential thought that one is in it {cite:p}`rosenthal1986two`.
- **Why**: Thought-like = conceptual → experience tracks the concepts we have.
    - e.g., A wine taster learns new concepts → notices more qualities in the same wine.
- **How**:
    - **Actualist** (Rosenthal): an actual, occurrent higher-order thought is required.
    - **Dispositionalist** (Carruthers): availability to a higher-order thought is enough → no need for 1 thought per conscious state.

```{attention} Q&A
:class: dropdown
*Why must the higher-order thought be non-inferential?*

- You can infer your own anger from your behavior w/o feeling it.
- → Inferred self-knowledge ≠ consciousness → HOT must arise directly.

*Must the higher-order thought itself be conscious?*

- ❌. Else infinite regress.
- It becomes conscious only if a 3rd-order thought targets it → introspection.
```

&nbsp;

### Self-Representational Theory (SRT)
- **What**: Conscious state = state that represents itself; the higher-order content is part of the same state {cite:p}`kriegel2009subjective`.
- **Why**: HOPT & HOTT use 2 separate states → misrepresentation & rock problems.
    - Same state → no gap between the state & its representation.
- **How**:
    - **Part–whole**: First-order & higher-order contents = 2 parts of 1 complex state.
    - **Identity**: 1 state plays both roles.

```{attention} Q&A
:class: dropdown
*Cons?*

- Unclear how a state can represent itself in a non-trivial way.
- No concrete neural or computational mechanism.
```

&nbsp;

## Attention Schema Theory (AST)
- **What**: Awareness = the brain's simplified internal model of its own attention {cite:p}`graziano2011human`.
- **Why**:
    - *Why do we need it?* Control. Good control needs a model of the controlled thing.
        - Body schema → arm control. Attention schema → attention control.
    - *Why does it work?* The brain reports from the schema.
        - Schema = simplified → no neurons, no signal competition.
        - → It depicts attention as a non-physical inner "awareness" → we claim to have ineffable subjective experience.
- **How**:
    1. **Attention**: Signals compete; winners get deeper processing.
    2. **Attention schema**: A simplified, continuously updated model of the current attention state.
    3. **Control**: The schema predicts & steers attention.
    4. **Report**: Cognition & speech read the schema → "I am aware of X".
    5. **Social**: The same machinery models other people's attention → we attribute awareness to them.

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionSchemaAgent(nn.Module):
    def __init__(self, d, n_items, d_schema=2):
        super().__init__()
        self.salience = nn.Linear(d, 1)               ## attention: bottom-up competition
        ## attention schema: low-dim, lossy model of the attention state
        self.encode = nn.Linear(n_items, d_schema)
        self.decode = nn.Linear(d_schema, n_items)

    def forward(self, items, prev_schema=None):       ## items: (n_items, d)
        logits = self.salience(items).squeeze(-1)     ## (n_items,)
        if prev_schema is not None:
            ## control: top-down bias comes from the MODEL of attention
            logits = logits + prev_schema
        attn = F.softmax(logits, dim=-1)              ## (n_items,) actual attention
        schema = self.decode(torch.tanh(self.encode(attn.detach())))  ## (n_items,) simplified description
        ## report: "I am aware of item k" is read from the schema, NOT from attention itself
        report = schema.argmax().item()
        ## train the schema to track attention: KL(attn || softmax(schema))
        schema_loss = F.kl_div(F.log_softmax(schema, dim=-1), attn.detach(), reduction="sum")
        return attn, schema, report, schema_loss

## Example: 2 steps; step 2 uses the schema from step 1 for control
agent = AttentionSchemaAgent(d=4, n_items=3)
items = torch.randn(3, 4)
attn, schema, report, loss = agent(items)
attn, schema, report, loss = agent(items, prev_schema=schema.detach())
print(attn.shape, report)  ## torch.Size([3]) <index of the reported item>
```
````

```{attention} Q&A
:class: dropdown
*Attention vs awareness?*

- Attention = mechanistic process that selects data.
- Awareness = the brain's simplified, partly wrong description of that process.
- Usually correlated, but they can dissociate (e.g., an unseen cue still captures attention).

*Key prediction?*

- No awareness of a stimulus → attention to it can still occur, but **control** of that attention degrades.

*Evidence in AI?*

- RL agent w/ an attention spotlight learns the task w/ an attention schema.
- Disable the schema → performance drops sharply {cite:p}`wilterson2021attention`.

*Does AST solve the hard problem?*

- It deflates it: it explains why we **believe & claim** to have non-physical experience.
- No extra "magic" is produced → close to illusionism.
- Critics: it explains the report, not the experience → it changes the subject.

*How does AST relate to GWT & HOT?*

- Attention that wins the competition = GWT's workspace content ("I-consciousness").
- Schema = higher-order model of that attention ("M-consciousness") → HOT-like.
- → AST ≈ GWT + a HOT-style self-model.

*Is transformer attention an attention schema?*

- ❌. Transformer attention = attention only.
- No separate model of its own attention that is used for control → Butlin indicator AST-1 not met.

*AC angle?*

- Functionalist & concrete → AC = build an attention schema that controls attention & drives self-report.
```

&nbsp;

## Conscious Turing Machine (CTM)
- **What**: Formal TCS model of GWT: a 7-tuple ⟨STM, LTM, Up Tree, Down Tree, Links, Input, Output⟩ {cite:p}`blum2022theory`.
- **Why**: The Turing machine formalizes computation, not consciousness.
    - → Simplest formal model where consciousness can be defined & reasoned about w/ theorems & resource limits.
- **How**:
    1. **LTM**: $N>10^7$ unconscious processors. Each submits 1 chunk per clock tick.
    2. **Up Tree**: Binary tree. Each node keeps 1 of its 2 children's chunks w/ probability ∝ $f(\text{chunk})$ (coin-flip neuron).
    3. **STM**: Holds 1 chunk = **conscious content**.
    4. **Down Tree**: Broadcasts the STM chunk to all LTM processors in 1 tick.
        - **Conscious awareness** = reception of the broadcast by all LTM processors.
    5. **Links**: Processors that often talk via STM form direct links → conscious communication becomes unconscious.
    6. **Feeling of consciousness**: Model-of-the-World processor (world model incl. "self") + Brainish inner language + predictive dynamics.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $c=\langle\text{address},t,\text{gist},\text{weight},\text{intensity},\text{mood}\rangle$: Chunk.
    - $f$: Competition function (chunk → $\mathbb{R}_{\geq0}$).
    - $c_L,c_R$: Chunks in the left & right child of a node.
    - $h$: Height of the Up Tree ($N=2^h$).

Process:
1. Leaf: processor $p$ submits $c_p$ w/ $\text{intensity}=|\text{weight}|$, $\text{mood}=\text{weight}$.
2. At each node: the left chunk wins w/ probability

    $$
    P(c_L\text{ wins})=\begin{cases}\dfrac{f(c_L)}{f(c_L)+f(c_R)}&f(c_L)+f(c_R)>0\\[4pt]\dfrac{1}{2}&\text{otherwise}\end{cases}
    $$

3. Winner moves up w/ address, gist, weight unchanged; intensity & mood = **sums** of the 2 children's values.
4. After $h$ ticks: root chunk enters STM → broadcast at the next tick.

Properties:
- $f$ is **additive** if $f(\text{parent chunk})=f(c_L)+f(c_R)$ (e.g., $f=\text{intensity}$).
- Theorem (additive $f$, $\sum_{p'}f(c_{p'})>0$):

    $$
    P(c_p\text{ reaches STM})=\frac{f(c_p)}{\sum_{p'=1}^{N}f(c_{p'})}
    $$

    - All $f=0$ → every contest is a fair coin → each chunk wins w/ probability $1/N$.

- $\text{mood}_t/N$ = average mood of all chunks submitted at $t-h$.
- Delay: $h=\log_2 N$ ticks up + 1 tick down. e.g., 1 tick = 100 ms, $N=10^7$ → ~2.3 s + 0.1 s ≈ 2.4 s.
```

```{tip} Derivation
:class: dropdown
*Why does additive $f$ give a winning probability proportional to $f$?*

1. Let $F(v)=\sum_{p\in\text{leaves}(v)}f(c_p)$ for node $v$.
2. Additive $f$ → the chunk at node $v$ has $f=F(v)$, whichever leaf won.
3. Claim: $P(c_p\text{ at }v)=f(c_p)/F(v)$ for every leaf $p$ under $v$ w/ $F(v)>0$.
4. Base: leaf $v=p$ w/ $f(c_p)>0$ → $f(c_p)/f(c_p)=1$. ✓
5. Step: $p$ under left child $L$ of $v$ (right child $R$), $F(L)>0$:

    $$
    P(c_p\text{ at }v)=\frac{f(c_p)}{F(L)}\cdot\frac{F(L)}{F(L)+F(R)}=\frac{f(c_p)}{F(v)}
    $$

6. If $F(L)=0$ (so $f(c_p)=0$) & $F(R)>0$ → $L$ loses w/ probability 1 → $P=0=f(c_p)/F(v)$. ✓
7. Root: $F(\text{root})=\sum_{p'}f(c_{p'})$. ∎
8. → Leaf placement does not matter. Low-$f$ chunks still have a chance.
```

````{important} Code
:class: dropdown
```python
import torch

class ConsciousTuringMachine:
    """Probabilistic Up Tree competition + Down Tree broadcast. N = 2^h LTM processors."""
    def __init__(self, n_processors):
        assert n_processors > 0 and n_processors & (n_processors - 1) == 0  ## perfect binary Up Tree
        self.n = n_processors

    def up_tree(self, weight):                          ## weight: (N,) valenced weights
        idx = torch.arange(self.n)                      ## address of the chunk at each node
        intensity, mood = weight.abs(), weight.clone()  ## leaf values; f = intensity (additive)
        while idx.numel() > 1:
            fl, fr = intensity[0::2], intensity[1::2]
            denom = fl + fr
            ## coin-flip neuron: left wins w.p. fl/(fl+fr); 1/2 if both are 0
            p_left = torch.where(denom > 0, fl / denom.clamp_min(1e-12), torch.full_like(denom, 0.5))
            left_wins = torch.rand_like(p_left) < p_left
            idx = torch.where(left_wins, idx[0::2], idx[1::2])
            ## winner carries the SUM of its children's intensity & mood up the tree
            intensity, mood = fl + fr, mood[0::2] + mood[1::2]
        return idx.item(), mood.item()                  ## STM: winning processor, CTM mood

    def __call__(self, gists, weight):
        winner, mood = self.up_tree(weight)
        ## Down Tree: broadcast the winning gist to ALL processors in 1 tick
        return [gists[winner]] * self.n, mood

## Example
ctm = ConsciousTuringMachine(4)
weight = torch.tensor([4.0, -2.0, 1.0, 1.0])
wins = torch.zeros(4)
for _ in range(10_000):
    wins[ctm.up_tree(weight)[0]] += 1
print(wins / 10_000)   ## ≈ |w| / Σ|w| = [0.5, 0.25, 0.125, 0.125]
print(ctm(["food", "pain", "noise", "itch"], weight))  ## e.g. (['food', 'food', 'food', 'food'], 4.0)
```
````

```{attention} Q&A
:class: dropdown
*Pros?*

- Precise → theorems & resource bounds.
- Explains blindsight (information reaches LTM processors, not STM), inattentional & change blindness, dreams.

*Cons?*

- Assumes computational functionalism.
- "Feeling of consciousness" is asserted from the architecture, not derived → hard problem stays.
- Simplified: all processors compete & receive all broadcasts. In humans, e.g., the dorsal visual stream never becomes conscious.

*How does CTM differ from GWT?*

- 1 fixed "actor" in STM that only broadcasts. All processors are in LTM.
- Competition is a formal, probabilistic tournament.
- Chunks carry intensity & mood → CTM has a global mood.

*Why must $f$ be additive?*

- Fairness: winning probability does not depend on leaf placement.
- Counterexample: $f=|\text{mood}|$. Siblings w/ weights $+100$ & $-100$ → mood sums to 0 → neither reaches STM, even though both are intense.

*Free will?*

- The coin-flip neurons add randomness.
- The Blums argue the **feeling** of free will arises even in a fully deterministic CTM (e.g., pseudorandom coins).

*AC angle?*

- By construction: a machine w/ the CTM architecture & its special processors has the feeling of consciousness (if CTM is right).
```

&nbsp;

## Recurrent Processing Theory (RPT)
- **What**: Consciousness = recurrent (feedback) processing within sensory cortex {cite:p}`lamme2006towards`.
- **Why**:
    - The fast feedforward sweep reaches high-level areas but stays unconscious → recurrence is the extra ingredient.
    - Report-based definitions mix consciousness w/ access → define consciousness by neural markers ("neural stance").
- **How**: Visual processing stages.
    1. Feedforward sweep (~100 ms) → features extracted → unconscious.
    2. Local recurrence in visual areas → features bind into an organized scene (figure–ground) → **phenomenal** consciousness.
    3. Widespread recurrence incl. fronto-parietal areas → **access** (report, working memory).

```{attention} Q&A
:class: dropdown
*Pros?*

- Neural & falsifiable.
- Explains masking: a backward mask disrupts recurrent, not feedforward, processing → stimulus unseen.
- Separates phenomenal from access consciousness → allows "overflow" (seeing more than you can report).

*Cons?*

- Phenomenal-without-access cannot be confirmed by report → hard to test.
- Recurrence also occurs in unconscious processing → not sufficient alone?

*How does RPT differ from GNWT?*

- GNWT: conscious = global access via fronto-parietal ignition.
- RPT: local recurrence in sensory cortex is enough. Global ignition = access & report only.
- → They disagree about the role of prefrontal cortex.

*Do transformers have recurrence?*

- 1 forward pass → ❌ recurrence.
- Autoregressive generation feeds outputs back → a loop through text, not recurrence within perception.
- Butlin indicators: input modules w/ algorithmic recurrence (RPT-1) → many AI systems (e.g., RNNs) meet it. Organized, integrated perceptual representations (RPT-2).
- **Algorithmic** recurrence (same weights reused over time) vs **implementational** recurrence (same physical neurons reused) → the stricter reading excludes simulations.
```

&nbsp;

## Predictive Processing (PP)
- **What**: Perception = the brain's top-down best guess of the hidden causes of its sensory input; conscious content = that guess {cite:p}`seth2021predictive`.
    - Seth: perception = "controlled hallucination".
    - PP = a theory **for** consciousness science (explain, predict & control properties of experience), not a complete theory **of** consciousness.
- **Why**: Senses are noisy & ambiguous → the brain must infer their causes → Bayesian inference.
    - Hard problem → "real problem": explain specific properties of experience via predictions.
- **How**:
    1. A generative model at each level predicts the activity of the level below.
    2. Mismatch → prediction error → sent up.
    3. Error, weighted by **precision** (confidence), updates beliefs (perception) or drives action (active inference).
    4. Conscious content = current top-down predictions, not the raw signal.
    5. Self & emotion = predictions about interoceptive body signals ("beast machine").

```{note} Math
:class: dropdown
1-level linear-Gaussian predictive coding.

Notations:
- IO:
    - $\mathbf{x}$: Sensory input.
    - $\boldsymbol{\mu}$: Belief about the hidden cause (inferred).
- Params:
    - $W$: Generative weights (predict $\mathbf{x}$ from $\boldsymbol{\mu}$).
    - $\boldsymbol{\eta}$: Prior mean of the cause.
- Hyperparams:
    - $\pi_x$: Sensory precision (inverse noise variance).
    - $\pi_p$: Prior precision.
    - $\kappa$: Update rate.
- Misc:
    - $I$: Identity matrix.

Model:

$$
\mathbf{x}\sim\mathcal{N}(W\boldsymbol{\mu},\pi_x^{-1}I),\quad\boldsymbol{\mu}\sim\mathcal{N}(\boldsymbol{\eta},\pi_p^{-1}I)
$$

Objective (free energy, up to a constant):

$$
F(\boldsymbol{\mu})=\frac{\pi_x}{2}\|\mathbf{x}-W\boldsymbol{\mu}\|^2+\frac{\pi_p}{2}\|\boldsymbol{\mu}-\boldsymbol{\eta}\|^2=-\log p(\mathbf{x},\boldsymbol{\mu})+\text{const}
$$

Inference (gradient descent on $F$):

$$
\boldsymbol{\mu}\leftarrow\boldsymbol{\mu}+\kappa\left(\pi_x W^T\boldsymbol{\varepsilon}_x-\pi_p\boldsymbol{\varepsilon}_p\right)
$$
- $\boldsymbol{\varepsilon}_x=\mathbf{x}-W\boldsymbol{\mu}$: Sensory prediction error (bottom-up).
- $\boldsymbol{\varepsilon}_p=\boldsymbol{\mu}-\boldsymbol{\eta}$: Deviation from the prior.
- Converges iff $0<\kappa<2/\lambda_\text{max}\left(\pi_x W^TW+\pi_p I\right)$ ← $F$ is a quadratic w/ that Hessian.

Fixed point (= posterior mean, MAP):

$$
\boldsymbol{\mu}^*=\left(\pi_x W^TW+\pi_p I\right)^{-1}\left(\pi_x W^T\mathbf{x}+\pi_p\boldsymbol{\eta}\right)
$$
```

````{important} Code
:class: dropdown
```python
import torch

class PredictiveCoding:
    """1-level linear-Gaussian predictive coding: infer the hidden cause mu of input x."""
    def __init__(self, W, eta, pi_x=1.0, pi_p=1.0, kappa=0.1):
        self.W, self.eta = W, eta                    ## generative model: x ≈ W mu, prior mean eta
        self.pi_x, self.pi_p, self.kappa = pi_x, pi_p, kappa

    def __call__(self, x, n_steps=100):
        mu = self.eta.clone()                        ## start from the prior belief
        for _ in range(n_steps):
            eps_x = x - self.W @ mu                  ## bottom-up prediction error
            eps_p = mu - self.eta                    ## pull back toward the prior
            ## precision-weighted errors drive the belief update (gradient descent on F)
            mu = mu + self.kappa * (self.pi_x * self.W.T @ eps_x - self.pi_p * eps_p)
        return mu

## Example: same input, different sensory precision
W, eta, x = torch.eye(2), torch.zeros(2), torch.tensor([2.0, 0.0])
print(PredictiveCoding(W, eta, pi_x=1.0)(x))  ## ≈ [1.0, 0.0] → belief halfway between prior & data
print(PredictiveCoding(W, eta, pi_x=9.0)(x))  ## ≈ [1.8, 0.0] → high precision → belief follows data
```
````

```{attention} Q&A
:class: dropdown
*Pros?*

- Unifies perception, action, attention & learning under 1 objective.
- Explains illusions & hallucinations: strong priors override weak input.

*Cons?*

- Very general → hard to falsify.
- PP also describes unconscious processing → needs extra criteria for which predictions are conscious.
- Describes what perception does, not why it feels like anything.

*What is attention in PP?*

- Precision optimization: up-weight prediction errors from reliable sources.

*How does PP relate to ML?*

- Variational free energy = negative ELBO → a VAE minimizes the same kind of objective.
- VAE encoder = amortized inference: 1 forward pass replaces the iterative $\boldsymbol{\mu}$ updates.

*AC angle?*

- Butlin indicator: input modules that use predictive coding (PP-1) → easy to build in ML.
- Seth argues consciousness may depend on being alive (self-regulation of a living body) → skeptical about LLM consciousness.
```

&nbsp;

## Comparison

```{dropdown} Table: Theories at a glance
| Theory | Conscious iff … | Proposed locus | Functionalist? | AC under the theory's own logic |
|:--|:--|:--|:--|:--|
| IIT | Max irreducible intrinsic cause–effect structure | Posterior cortex | ❌ Physical causal structure | ❌ on standard digital hardware |
| State of matter | Information + integration + independence + dynamics | Any matter | ✅ Pattern, not substrate | Open: only necessary conditions |
| Orch OR | Orchestrated OR in microtubules | Neuronal microtubules | ❌ Needs non-computable physics | ❌ for any classical computer |
| GWT / GNWT | Content broadcast from a limited workspace | Prefrontal–parietal network | ✅ | ✅ if built; LLMs: weak case |
| HOT | Higher-order representation of the state | Prefrontal cortex | ✅ | ✅ w/ metacognitive monitoring |
| AST | Brain's model of its own attention | Temporoparietal junction | ✅ | ✅ w/ an attention schema |
| CTM | STM content received by all LTM processors | Formal model | ✅ | ✅ by construction |
| RPT | Recurrent processing in sensory cortex | Sensory cortex | ✅ if algorithmic recurrence counts | ✅ w/ recurrent perception |
| PP | Top-down best guess of sensory causes | Whole cortical hierarchy | ✅ in form; Seth leans biological | Open |
```