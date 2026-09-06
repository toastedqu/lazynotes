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
# Decoding
- **What**: Token probabilities → Output token
- **Why**:
	- In generation tasks, the transformer model only estimates the probabilities of outputting each token via logits.
	- We need to select which token to output at each step, and we need to find the most probable output sequence given the input sequence.

```{note} Math
:class: dropdown
Notations:
- IO:
	- $X$: Input sequence.
	- $Y$: Output sequence.
	- $y_t$: Output token at step $t$.
	- $Y_{<t}$: Output sequence up till step $t$.
- Hyperparam:
	- $\tau$: Temperature.
- Misc:
	- $\mathcal{V}$: Vocab set.
	- $v_i$: $i$th token.
	- $z_{ti}$: Logit of $v_i$ at step $t$.
	- $T$: Total output sequence length.

Procedure:
1. Probability of outputting token $v_i$ at step $t$:

$$
P(y_t=v_i|Y_{<t},X)=\text{softmax}(z_{ti})=\frac{\exp(\frac{z_{ti}}{\tau})}{\sum_{j=1}^{|\mathcal{V}|}\exp(\frac{z_{tj}}{\tau})}
$$

2. Probability of outputting sequence $Y$:

$$
P(Y|X)=\prod_{t=1}^{T}P(y_t|Y_{<t},X)
$$

3. Objective - Find the optimal sequence:

$$
Y_*=\arg\max_YP(Y|X)
$$
```

## Temperature
- **What**: Randomness control.
- **Why**: **Boltzmann distribution** from statistical mechanics: $p_i\propto\exp\left(-\frac{E_i}{kT}\right)$
	- This describes the probability of a system being in a particular state $i$ given the state's energy $E_i$ and the system's temperature $T$.
	- Temperature $T$ controls the randomness of physical systems:
		- $T$⬆️ → Difference in $p$ between low-energy and high-energy states⬇️ → Can't stick to low-energy states → Randomness⬆️
		- $T$⬇️ → Difference in $p$ between low-energy and high-energy states⬆️ → Stick to low-energy states → Randomness⬇️
	- Identical problem setting:
		- States $\Leftrightarrow$ Tokens
		- Energy $\Leftrightarrow$ Logits
		- $T$⬆️ → Less probable tokens become more probable → Randomness⬆️
		- $T$⬇️ → Less probable tokens become even less probable → Randomness⬇️
- **How**:
	- $\tau\rightarrow 0$: More deterministic.
	- $\tau\rightarrow \infty$: More random/uniform.
	- $\tau=1$: Standard softmax. Probabilities reflect differences in logits.
	- $\tau>1$: Generally NOT recommended ← Randomness goes BEYOND what the model has learnt.

## Penalty
- **What**: Penalize the logits of tokens present in current token sequence.
- **Why**: Autoregressive LMs can fall into repetition loops. (Occurs much more often with Greedy/Beam Search)
    - Autoregressive LMs are trained to predicted the most probable next token given current context.
    - If a particular token/phrase is highly probable given the current context & gets picked, it becomes part of the new context.
    - If the same token/phrase is still the most probable given this new slightly longer context, it's likely to get picked again.
    - ...
    - Infinite "positive" feedback loop.

```{dropdown} Table: Penalty Types
| Type | What | Math | Cons |
|:-----|:-----|:-----|:-----|
| **Frequency** | Subtraction based on how many times the token occurred in the output sequence | $z_{ti} \leftarrow z_{ti}-\alpha n_{v_i}$ | Suppresses important keywords |
| **Presence** | Subtraction based on the **existence** of the token in the output sequence | $z_{ti} \leftarrow z_{ti}-\beta \mathbf{1}_{v_i}[Y_{<t}]$ | Incoherence ← Too harsh compared to frequency penalty |
| **Repetition** | Multiplication based on the **existence** of the token in the **entire sequence** | $z_{ti} \leftarrow \begin{cases} z_{ti} / \rho & v_i \in [X, Y_{<t}]\ \& \ z_{ti} > 0 \\ z_{ti} \cdot \rho & v_i \in [X, Y_{<t}]\ \& \ z_{ti} < 0 \\ z_{ti} & v_i \notin [X, Y_{<t}] \end{cases}$ | Suppresses references to important keywords in the input | 

Notations:
- Hyperparams:
	- $\alpha$: Frequency penalty hyperparam.
	- $\beta$: Presence penalty hyperparam.
	- $\rho$: Repetition penalty hyperparam.
- Misc:
	- $\mathbf{1}_{v_i}[Y_{<t}]$: 1 if token $v_i$ is present in $Y_{<t}$, else 0.
```

## Greedy Search
- **What**: Always take the most probable token at each step.
- **Why**: Simplest.

```{note} Math
:class: dropdown
Greedy Search:

$$
y_t=\arg\max_{v}P(y_t=v|Y_{*,<t},X)
$$
```

## Beam Search
- **What**: Iteratively explore & evaluate multiple hypotheses (i.e., beams). 
- **Why**: Greedy search focuses on local optima → May not lead to globally optimal sequence.
- **How**:
	1. Initialize $k$ beams with top $k$ most probable tokens.
	2. For each step:
		1. For each beam: Compute probability distribution for next token.
		2. Consider all possible next tokens in vocab → Form candidate beams.
		3. For each candidate beam: Compute a **score** based on the log probability of the beam sequence.
		4. Select top $k$ beams with the highest scores.
	3. Stop when
		- Max length.
		- All $k$ beams have generated the EOS token.
	4. Output the beam with the highest score.

```{note} Math
:class: dropdown
Beam Search:
1. At $t=1$, select top $k$ most probable tokens via $P(y_1|X)$. Each forms an initial beam $Y_{1,i}=[y_{1,i}]$.
2. $\forall t>1$:
	1. $\forall Y_{t-1,i}=[y_{1,i},\cdots,y_{t-1,i}]$: Compute $P(y_t|Y_{t-1,i},X)$.
	2. Consider all $|\mathcal{V}|$ possible tokens → Form $k\times|\mathcal{V}|$ candidate beams.
	3. $\forall Y_{t,i}=[y_{1,i},\cdots,y_{t,i}]$: Compute $S(Y_t)=\log P(Y_t|X)=\sum_{t'=1}^{t}\log P(y_{t'}|Y_{<t'},X)$.
	4. Select top $k$ beams with the highest scores.
3. Termination.
4. Output.
```

```{attention} Q&A
:class: dropdown
*Cons?*
- Beam Search naturally favors shorter sequences ← Adding more log probabilities reduces the score.

*Solution?*
- **Length normalization**:

	$$
	S(Y_t)=\frac{1}{t^\alpha}\sum_{t'=1}^{t}\log P(y_{t'}|Y_{<t'},X)
	$$
	- $\alpha$: Length normalization hyperparameter.
```

## Sampling
- **What**: Random selection from a set/distribution.
- **Why**: Controlled randomness.
	- Greedy or Beam Search lead to **deterministic** outcomes.
		- Greedy: Most probable tokens at each step.
		- Beam: Most probable sequence at the end.
	- Deterministic outcomes = Generic, repetitive, lacking creativity.

### Multinomial
- **What**: Output token $\sim$ Full vocab distribution.
    - a.k.a. **ancestral**/**pure** sampling. Combined w/ $\tau$ → **temperature sampling**.
- **Why**: Unbiased — the output distribution IS the model's distribution.
    - → Every **truncation** or **search** strategy below is a deliberate bias, traded for coherence.
- **How**: Draw $y_t\sim P(\cdot|Y_{<t},X)$ at each step. ❌Truncation, ❌Search.

```{attention} Q&A
:class: dropdown
*Why is it almost never used raw?*
- $|\mathcal{V}|\sim10^5$ → each tail token is tiny, but the tail's **total** mass is not.
- One junk token → enters the context → model conditions on its own mistake → derailment.
- → Top-k/Top-p/Min-p all exist to cut that tail.
```

### Top-k
- **What**: Output token $\sim$ Top-$k$ most probable tokens.
- **Why**: Temperature sampling → Too much randomness if large temperature → Incoherent sequence
	- Any token may be selected based on its probability, including less probable ones.
	- We don't want that, so we limit the vocab options and resample.
- **How**:
	1. Compute probabilities of all tokens in vocab.
	2. Select top $k$ tokens.
	3. Re-normalize probabilities of top $k$ tokens.
	4. Sample.

### Top-p (Nucleus)
- **What**: Output token $\sim$ Smallest possible set of tokens whose cumulative probability exceeds $p$.
	- **Nucleus**: the set.
- **Why**: In Top-k,
	- If the model is very certain about the next word, large $k$ → too random.
	- If the model is very uncertain about the next word, small $k$ → too deterministic.
- **How**:
	1. Compute probabilities of all tokens in vocab.
	2. Sort tokens by probability.
	3. Form nucleus.
	4. Re-normalize probabilities of top $k$ tokens.
	5. Sample.

### Min-p
- **What**: Output token $\sim$ Tokens whose probability is $\geq$ a fixed **fraction of the top token's** probability. {cite:p}`nguyen2024turning`
- **Why**: Top-k & Top-p cutoffs are **absolute** → the same setting means different things at different confidence levels.
    - Peaked (model certain) → top token alone may exceed $p$ → effectively greedy → Diversity⬇️
    - Flat (model uncertain) → $p=0.9$ still sweeps in a long junk tail → Coherence⬇️
    - → Scale the threshold by the model's own confidence $p_{\max}$ → **relative** truncation.
- **How**:
    1. Compute probabilities of all tokens in vocab.
    2. Threshold $\leftarrow p_\text{base}\cdot p_{\max}$.
    3. Keep tokens above threshold.
    4. Re-normalize probabilities of kept tokens.
    5. Sample.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $p_\text{base}\in(0,1]$: Base probability floor.
- Misc:
    - $p_{\max}=\max_jP(y_t=v_j|Y_{<t},X)$: Probability of the most likely token at step $t$.

Truncation set:

$$
\mathcal{V}_\text{min-p}=\left\{v_i\in\mathcal{V}:P(y_t=v_i|Y_{<t},X)\geq p_\text{base}\cdot p_{\max}\right\}
$$

Then re-normalize over $\mathcal{V}_\text{min-p}$ & sample.
```

```{dropdown} Table: Truncation Samplers
| Sampler | Cutoff | Set size | Adapts to confidence? | Failure mode |
|:--|:--|:--|:--|:--|
| **Top-k** | Fixed **count** $k$ | Fixed | ❌ | Peaked → junk admitted; Flat → over-truncated |
| **Top-p** | Fixed **cumulative mass** $p$ | Varies | Partially | Flat → long junk tail still fits under $p$ |
| **Min-p** | Fixed **ratio** to $p_{\max}$ | Varies | ✅ | Near-uniform → almost nothing filtered ($p_{\max}\approx$ every $p_i$) |
```

```{attention} Q&A
:class: dropdown
*Pros?*
- Threshold tracks per-step confidence → Diversity⬆️ when unsure, Precision⬆️ when sure.
- Holds up at **high temperature**, where Top-p turns incoherent (GPQA, GSM8K, AlpacaEval Creative Writing; Mistral & Llama 3, 1B-123B).
- Adopted in HF Transformers, vLLM, llama.cpp → de-facto standard knob.

*Cons?*
- ❌Bounds the candidate count → a near-uniform distribution passes almost everything through.
- Redundant w/ Top-p → normally one or the other, not both.

*Why is it "argmax invariant"?*
- $p_\text{base}\leq1$ → $p_{\max}\geq p_\text{base}\cdot p_{\max}$ → the top token is NEVER filtered.
- → Min-p can't change a greedy decode → free to skip when $\tau=0$.
```

## Sampler Pipeline
- **What**: The order in which logit modifiers & truncators are applied at each step.
- **Why**: They do NOT commute → same hyperparams, different order, different distribution.
    - $\tau$ rescales logits → changes the very probability mass that Top-p/Min-p measure.
    - Truncate-then-temper $\neq$ temper-then-truncate.
- **How**: Penalties → Temperature → Truncation → Re-normalize → Sample.
    - Penalties act on **raw logits** ← defined as logit offsets.
    - Truncators act on **tempered probabilities** ← their thresholds are probability-space quantities.

```{dropdown} Table: Applied Order
| # | HF Transformers `generate` | vLLM v1 `Sampler` |
|:--|:--|:--|
| 1 | Penalties, logit bias, bad-word masks | Penalties, logit bias, bad-word masks |
| 2 | Temperature | Temperature |
| 3 | Top-k | **Min-p** |
| 4 | Top-p | Top-k |
| 5 | **Min-p** | Top-p |
| 6 | Sample | Sample |

⚠️ Stages 1-2 agree; the **truncator order differs**. Same `(top_p, min_p)` → different distribution across engines.
```

````{important} Code
:class: dropdown
```python
import torch

def sample(logits, prev_ids, temp=1.0, top_k=0, top_p=1.0, min_p=0.0,
           freq_pen=0.0, pres_pen=0.0):
    ## 1. penalties act on RAW logits <- they are defined as logit offsets
    if freq_pen or pres_pen:
        counts = torch.bincount(prev_ids, minlength=logits.size(-1))
        logits = logits - freq_pen * counts - pres_pen * (counts > 0)

    ## 2. temperature BEFORE truncation -> it reshapes the mass top-p/min-p measure
    logits = logits / max(temp, 1e-5)
    probs = torch.softmax(logits, -1)

    ## 3a. top-k: fixed COUNT
    if top_k:
        kth = probs.topk(top_k).values[-1]
        probs = probs.masked_fill(probs < kth, 0.0)
        probs = probs / probs.sum()             ## re-normalize -> top-p measures the SURVIVING mass

    ## 3b. top-p: fixed CUMULATIVE MASS
    if top_p < 1.0:
        s, idx = probs.sort(descending=True)
        drop = (s.cumsum(-1) - s) >= top_p      ## exclusive cumsum -> top token always kept
        probs = probs.masked_fill(torch.zeros_like(drop).scatter(0, idx, drop), 0.0)

    ## 3c. min-p: fixed RATIO to the peak (scale-free -> re-normalization irrelevant here)
    if min_p:
        probs = probs.masked_fill(probs < min_p * probs.max(), 0.0)

    ## 4. re-normalize, then sample
    return torch.multinomial(probs / probs.sum(), 1)

## Example
torch.manual_seed(0)
print(sample(torch.randn(10), torch.tensor([3, 3, 7]), temp=0.8, top_p=0.9, min_p=0.05))
```
````

## Speculative Decoding
- **What**: Cheap draft model proposes $\gamma$ tokens → target model verifies them all in ONE forward pass. {cite:p}`leviathan2023fast,chen2023accelerating`
- **Why**: Autoregressive decoding is **memory-bandwidth-bound at low batch**, ❌compute-bound.
    - Each step drags ALL params HBM → on-chip cache just to emit ONE token → GPU mostly idle.
    - → Scoring 1 token and scoring $\gamma+1$ tokens cost nearly the same wall-clock.
    - → Spend the wasted parallelism **verifying a guess** instead of generating one token.
- **How**:
    1. Draft model autoregressively generates $\gamma$ tokens (cheap, sequential).
    2. Target model scores all $\gamma+1$ positions in one parallel pass.
    3. Accept each drafted token w.p. $\min(1,p/q)$; stop at the first rejection.
    4. On rejection, resample that token from the **residual** $\max(0,p-q)$.
    5. On full acceptance, sample a **bonus** token from the target → $\gamma+1$ tokens in one iteration.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $q$: Draft (small) model distribution.
    - $p$: Target (large) model distribution.
- Hyperparams:
    - $\gamma$: #Tokens drafted per iteration.
- Misc:
    - $\tilde{y}_i$: $i$th drafted token.
    - $\alpha$: Expected acceptance rate.
    - $D_{TV}$: Total variation distance.

Process:
1. Draft $\tilde{y}_i\sim q(\cdot|Y_{<t},\tilde{y}_{<i})$ for $i=1,\cdots,\gamma$.
2. One target pass → $p(\cdot|Y_{<t},\tilde{y}_{<i})$ for $i=1,\cdots,\gamma+1$.
3. Accept $\tilde{y}_i$ with probability

    $$
    \min\left(1,\frac{p(\tilde{y}_i|Y_{<t},\tilde{y}_{<i})}{q(\tilde{y}_i|Y_{<t},\tilde{y}_{<i})}\right)
    $$

4. On the first rejection, resample that position from the normalized residual

    $$
    y_i\sim\frac{\max(0,p(\cdot)-q(\cdot))}{\sum_v\max(0,p(v)-q(v))}
    $$

Guarantee: the emitted tokens are distributed **exactly** as $p$, for ANY $q$.

Acceptance rate:

$$
\alpha=\mathbb{E}\left[1-D_{TV}(p,q)\right]=\mathbb{E}\left[\sum_v\min(p(v),q(v))\right]
$$
- Written $D_{LK}$ in the original paper; it equals $D_{TV}$.

Yield (assuming i.i.d. per-step acceptance) — a capped geometric over $1,\cdots,\gamma+1$:

$$
\mathbb{E}[\#\text{tokens per iter}]=\frac{1-\alpha^{\gamma+1}}{1-\alpha}
$$
- $\alpha\rightarrow1$ (perfect draft): removable singularity → $\gamma+1$, the cap.

Walltime improvement:

$$
\frac{1-\alpha^{\gamma+1}}{(1-\alpha)(\gamma c+1)}
$$
- $c$: Cost ratio of one draft run to one target run. ($c<0.05$ when $q$ is orders of magnitude smaller than $p$.)
- → The $\gamma c$ term is exactly why $\gamma$ can't grow freely: drafting cost is linear, token yield saturates.
```

````{important} Code
:class: dropdown
```python
import torch

@torch.no_grad()
def speculative_step(p_fn, q_fn, prefix, gamma=4):
    """p_fn(ctx, n) -> (n, V): target scores the last n positions in ONE pass.
       q_fn(ctx)    -> (V,):   draft scores the next token."""
    ## 1. draft: gamma cheap autoregressive steps
    draft, q_probs, ctx = [], [], prefix
    for _ in range(gamma):
        q = q_fn(ctx)
        tok = torch.multinomial(q, 1)
        q_probs.append(q); draft.append(tok)
        ctx = torch.cat([ctx, tok])

    ## 2. verify: ONE target pass covers all gamma+1 positions
    p_all = p_fn(ctx, gamma + 1)

    ## 3. accept/reject left to right
    out = []
    for i, tok in enumerate(draft):
        p, q = p_all[i], q_probs[i]
        if torch.rand(1) < (p[tok] / q[tok]).clamp(max=1.0):
            out.append(tok)
        else:
            ## 4. residual resample -> output law stays EXACTLY p, for any draft q
            resid = (p - q).clamp(min=0)
            return out + [torch.multinomial(resid / resid.sum(), 1)]

    ## 5. all accepted -> free bonus token from the target
    return out + [torch.multinomial(p_all[gamma], 1)]

## Example: toy models over a vocab of 8 (draft = a flattened copy of the target)
torch.manual_seed(0)
z = torch.randn(8)
p_fn = lambda ctx, n: torch.softmax(z, -1).expand(n, 8)
q_fn = lambda ctx: torch.softmax(z * 0.5, -1)
print(len(speculative_step(p_fn, q_fn, torch.tensor([0]), gamma=4)))  ## 1..5
```
````

```{attention} Q&A
:class: dropdown
*Why is it lossless?*
- Accept-reject + residual resampling = **modified rejection sampling**, i.e. a **maximal coupling** of $p$ & $q$.
    - ❌Textbook rejection sampling ← on rejection it draws from the residual & STOPS, it does NOT retry from $q$.
- Accepted mass $\min(p,q)$ + residual mass $\max(0,p-q)$ = $p$ for every token.
- → Output distribution is $p$ regardless of how bad $q$ is. $q$ affects SPEED only, ❌quality.

*Then why not make $\gamma$ huge?*
- Acceptance is a prefix: one rejection discards ALL later drafted tokens.
- Accepted run length decays geometrically as $\alpha^i$ → drafting cost grows linearly, payoff saturates.
- → $\gamma$ is tuned to the draft/target cost ratio; typically single digits.

*When does it NOT help?*
- Large batch / high throughput regime → decoding is already compute-bound, ❌spare parallelism.
- Low $\alpha$ (draft poorly aligned) → drafting cost outweighs accepted tokens.

*Greedy target?*
- Accept iff the drafted token equals the target's argmax → verification becomes exact match.
```

### Medusa
- **What**: Replace the draft model w/ $K$ extra decoding heads on the target's own last hidden state. {cite:p}`cai2024medusa`
- **Why**: A separate draft model must be obtained, aligned, served & memory-resident.
    - → Predict $t+2,\cdots,t+K+1$ directly from the backbone → ❌Second model.
- **How**:
    1. Attach $K$ lightweight heads to the last hidden state $h_t$; head $k$ predicts position $t+k+1$.
        - The **original** LM head still predicts $t+1$ → $K+1$ positions covered per step.
    2. Take top candidates per head → Cartesian product → many candidate continuations.
    3. Verify them **all at once** via **tree attention** (one pass, a mask encodes the tree).
    4. Accept the longest valid prefix.

```{attention} Q&A
:class: dropdown
*Two recipes?*
- **Medusa-1**: Heads on a **frozen** backbone → >2.2x speedup.
- **Medusa-2**: Heads + backbone trained jointly → 2.3-2.8x, needs a recipe that preserves backbone quality.
    - ⚠️ The arXiv v1 preprint reported 2.3-3.6x; the published version revises it down to 2.3-2.8x.

*Is it really lossless?*
- Only for Medusa-1 (backbone untouched) AND only under the **standard** accept/reject rule.
- Medusa's proposed **typical acceptance** relaxes that rule to raise the acceptance rate → ❌Distribution-preserving.

*Why tree attention?*
- Independent heads → the top-1 of each rarely forms one good sequence.
- A tree evaluates many continuations for roughly the cost of one → Acceptance rate⬆️
```

### EAGLE
- **Name**: Extrapolation Algorithm for Greater Language-model Efficiency {cite:p}`li2024eagle`
- **What**: Autoregress at the **feature** level (second-to-top hidden state) instead of the token level.
- **Why**: Medusa's heads predict each position **independently** → ❌conditioning on the previous draft token → Acceptance rate⬇️
    - Feature sequences are smoother & more predictable than token sequences.
    - But features alone are ambiguous ← the sampled token is unknown.
    - → Feed the token sequence **shifted one step ahead** alongside the features → uncertainty resolved.
- **How**: A one-layer autoregressive head consumes (features, shifted tokens) → predicts the next feature → reuse the frozen LM head → draft tokens → verify as usual.

```{attention} Q&A
:class: dropdown
*Gains?*
- 2.7-3.5x latency speedup on LLaMA2-Chat 70B, throughput doubled, generated distribution preserved.

*Why is it lossless?*
- The draft mechanism changed; the accept/reject rule did NOT → still exact rejection sampling on $p$.
```

### Prompt Lookup
- **What**: Draft by **copying** an n-gram continuation from the existing context. {cite:p}`saxena2023prompt`
- **Why**: ❌Draft model, ❌training, ❌extra memory.
    - Many workloads are heavily **extractive** → the output largely repeats the input.
- **How**:
    1. Match the last $n$ generated tokens against the prompt/context.
    2. On a hit, take the following few tokens as the draft.
    3. Verify with the standard accept/reject rule.

```{attention} Q&A
:class: dropdown
*When does it shine?*
- ✅RAG, summarization, code editing, multi-turn edits → high verbatim overlap w/ input.
- ❌Open-ended generation → almost no matches → falls back to plain decoding, near-zero overhead.
```

## Constrained Decoding
- **What**: Mask the logits of every token that would violate a formal grammar. {cite:p}`willard2023efficient`
- **Why**: Prompting for JSON/tool calls is a **soft** request → non-zero failure rate at any model scale.
    - Downstream parsers are all-or-nothing → 1 stray token breaks the pipeline.
    - Validity is a property of the **output space**, ❌the weights → enforce at decode time, ❌train time.
- **How**:
    1. Compile the schema (regex / JSON Schema / CFG) → FSM or pushdown automaton.
    2. Index it against the vocab → each state maps to its set of **allowed** next tokens.
    3. $z_{ti}\leftarrow-\infty$ for every disallowed token → re-normalize → sample as usual.
    4. Advance the automaton with the chosen token.

````{important} Code
:class: dropdown
```python
import torch

class GrammarMask:
    """Compiled schema -> automaton state gives the allowed token set at each step."""
    def __init__(self, transitions, start=0):
        self.transitions = transitions   ## {state: {token_id: next_state}}
        self.state = start

    def apply(self, logits):
        ## -inf everywhere EXCEPT the grammar-legal tokens -> softmax zeroes them out
        mask = torch.full_like(logits, float("-inf"))
        mask[list(self.transitions[self.state])] = 0.0
        return logits + mask

    def advance(self, token_id):
        self.state = self.transitions[self.state][int(token_id)]

## Example: vocab of 5, grammar accepts only 1 -> (2|3) -> 4 -> EOS(0)
g = GrammarMask({0: {1: 1}, 1: {2: 2, 3: 2}, 2: {4: 3}, 3: {0: 3}})
print(torch.softmax(g.apply(torch.zeros(5)), -1))  ## [0., 1., 0., 0., 0.] -> only token 1
```
````

```{attention} Q&A
:class: dropdown
*Is it lossless like speculative decoding?*
- ❌ Speculative decoding preserves $p$ exactly; constrained decoding **deliberately changes** it.
- It **locally normalizes** $p$ over the legal set at each step ← ❌Conditioning on "the FULL sequence is valid".
- → Mass is redistributed myopically → can steer into a locally-legal path the model would never pick globally.
- → Can force a token the model finds unlikely → Quality⬇️ if the grammar fights the model.

*Cost?*
- Naive: walk the automaton over all $|\mathcal{V}|$ tokens EVERY step → non-negligible overhead.
- Outlines: precompute a state → allowed-token index → $O(1)$ mask lookup, but only for regex/FSM.
- XGrammar: splits **context-independent** tokens (prechecked) from **context-dependent** ones (runtime), + persistent stack for CFGs → up to 100x over prior solutions, near-zero end-to-end overhead. {cite:p}`dong2024xgrammar`

*Gotcha?*
- Masks are over **tokens**, constraints are over **characters** → a single token may straddle a grammar boundary.
- → Tokenizer-aware compilation is mandatory; naive char-level FSMs are wrong.
```

## Test-Time Search
- **What**: Sample $N$ complete sequences → select/aggregate one.
- **Why**: Trade inference compute for accuracy, ❌Retraining.
    - One sample = one draw from a distribution that is right *often*, ❌always.
    - $N$ draws surface correct answers the top-1 misses → 2 ways to cash that in:
        - **Score** them → keep the best (Best-of-N).
        - **Count** them → keep the most agreed-upon (Self-Consistency).
- **How**: Orthogonal to everything above — those pick a **token**, these pick a **sequence**.

### Best-of-N
- **What**: Sample $N$ → score each w/ a verifier/RM → return the argmax. {cite:p}`cobbe2021training`
    - a.k.a. **rejection sampling**, **reranking**.
- **Why**: Verifying a solution is easier than generating one.
    - The generator's top-1 is often wrong while a correct answer sits somewhere in its top-$N$.
- **How**:
    1. Sample $N$ i.i.d. completions at $\tau>0$.
    2. Score each w/ a verifier (programmatic) or RM (learned).
    3. Return the highest-scoring one.

```{note} Math
:class: dropdown
Notations:
- $\pi_\text{ref}$: Base sampling policy.
- $\pi_{\text{BoN}}$: Induced best-of-$N$ policy.
- $N$: #Samples.

KL cost of the induced policy:

$$
D_{KL}(\pi_\text{BoN}\|\pi_\text{ref})\leq\log N-\frac{N-1}{N}
$$
- Long quoted as exact; proven to be an **upper bound** only. {cite:p}`beirami2024theoretical`
- → BoN buys reward at a KL cost growing only **logarithmically** in $N$.
```

```{attention} Q&A
:class: dropdown
*Why does accuracy saturate, or fall, as $N$⬆️?*
- A learned RM is a **proxy** → argmax over more samples finds its holes faster → [reward hacking](../rh.md).
- ✅Exact verifier: keeps improving. ❌Learned RM: peaks, then degrades.

*BoN vs. RLHF?*
- BoN: inference-time, ❌weight change, cost $\propto N$ on EVERY query.
- RLHF: train-time, cost paid once, free at inference.
- Both trace a reward-vs-KL frontier → BoN is the baseline RLHF must beat.

*Why is it also a training tool?*
- BoN outputs → SFT data = **rejection sampling fine-tuning** → distills the search back into the weights.
```

### Self-Consistency
- **What**: Sample $N$ CoT paths → **majority-vote** the final answer. {cite:p}`wang2022self`
- **Why**: ❌Verifier, ❌RM, ❌Extra training.
    - One question, many valid reasoning paths → they should converge on the same answer.
    - Agreement across independent paths is itself evidence of correctness.
- **How**:
    1. Sample $N$ chains-of-thought at $\tau>0$.
    2. Extract the final answer from each.
    3. Return the most frequent answer.

```{attention} Q&A
:class: dropdown
*Why not greedy-decode the single most likely chain?*
- Greedy commits to one path → a single early misstep is unrecoverable.
- Most likely **chain** $\neq$ most likely **answer**.
- → Voting marginalizes over chains → approximates $\arg\max_a\sum_{\text{chains}\rightarrow a}P(\text{chain}|X)$.

*Limits?*
- Needs a **discrete, extractable** answer to vote on → ❌Open-ended generation.
- Cost $\propto N$; gains flatten as $N$⬆️
- ❌Detects a **consistently** wrong model → majority of garbage is still garbage.
```
