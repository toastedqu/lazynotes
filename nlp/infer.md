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
# Inference
## Decoding
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

### Temperature
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

### Penalty
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

### Greedy Search
- **What**: Always take the most probable token at each step.
- **Why**: Simplest.

```{note} Math
:class: dropdown
Greedy Search:

$$
y_t=\arg\max_{v}P(y_t=v|Y_{*,<t},X)
$$
```

### Beam Search
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

### Sampling
- **What**: Random selection from a set/distribution.
- **Why**: Controlled randomness.
	- Greedy or Beam Search lead to **deterministic** outcomes.
		- Greedy: Most probable tokens at each step.
		- Beam: Most probable sequence at the end.
	- Deterministic outcomes = Generic, repetitive, lacking creativity.

#### Top-k
- **What**: Output token $\sim$ Top-$k$ most probable tokens.
- **Why**: Temperature sampling → Too much randomness if large temperature → Incoherent sequence
	- Any token may be selected based on its probability, including less probable ones.
	- We don't want that, so we limit the vocab options and resample.
- **How**:
	1. Compute probabilities of all tokens in vocab.
	2. Select top $k$ tokens.
	3. Re-normalize probabilities of top $k$ tokens.
	4. Sample.

#### Top-p (Nucleus)
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

&nbsp;

## KV Cache
- **What**: Cache the $K$ & $V$ vectors for previous tokens.
- **Why**: They don't change during inference.
    1. ONLY the last hidden state is used to predict next token.
    2. What's the last hidden state?
        - Recall attention formula: $\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$.
        - $Q$: Query vector for LAST token.
        - $K$: Key vectors for ALL tokens.
        - $V$: Value vectors for ALL tokens.
    3. The causal masking in transformer decoders prevents later tokens from affecting earlier tokens.
    4. → At each autoregressive step, $K$ & $V$ of input tokens never change.
    5. $\xrightarrow{\text{cache}}$ Avoid recomputation.
- **How**: Cache.

```{attention} Q&A
:class: dropdown
*Pro Tip:* The query vectors for previous tokens are NEVER needed during inference.

*If it's so good, any cons?*
- ⬆️Memory cost: $O(m\cdot h\cdot d_k\cdot \#\mathrm{layers})$

*When should you turn it off?*
- Very short prompts: Recomputation is cheaper than caching.
- Parallelism & Hardware Acceleration: Caching adds an ALL-layer ALL-head copy for each step for each GPU.
```