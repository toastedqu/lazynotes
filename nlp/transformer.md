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
# Transformer
Transformer uses **self-attention** to model relationships between all tokens in input sequence. It excels in NLP because of **long-range dependencies** & **parallel processing** {cite:p}`vaswani2017attention`.

```{image} ../images/nlp-transformer/transformer.png
:align: center
:width: 500px
```


<!-- ````{dropdown} Input
```{image} ../images/transformer/input.png
:align: center
:width: 400px
```
````

````{dropdown} MHA
```{image} ../images/transformer/attention.png
:align: center
:width: 400px
```
```` -->

## Tokenization
- **What**: Text sequence → Token sequence
- **Why**: Machines can only read numbers.

```{attention} Q&A
:class: dropdown
*Why subword-level vocab? Why not whole words? Why not chars?*
- Word-level vocab explode with out-of-vocab words.
- Char-level vocab misses morphology.
- Subword offers a fixed-size, open-vocab symbol set which can handle rare words while maintaining morphology.
```

### BPE
- **What**: Byte-Pair Encoding (**Frequency-based** subword tokenization).
- **How**:
	1. Start with single chars as tokens.
	2. Count each pair of adjacent tokens.
	3. Merge the **most frequent** pair into one token.
	4. Repeat Step 2-3 till reaching vocab size.

### WordPiece
- **What**: **Likelihood-based** subword tokenization.
- **Why**: 
	- BPE merges by frequency → Greedy → Doesn't care about the merge's impact on the overall LM probability
	- WordPiece aims to **maximize corpus likelihood** under a unigram LM over subword tokens.
- **How**:
	1. Start with single chars as tokens.
	2. Compute **corpus likelihood gain** for each pair of adjacent tokens.
	3. Merge the pair with the **highest likelihood gain** into one token.
	4. Repeat Step 2-3 till reaching vocab size.

```{note} Math
:class: dropdown
Notations:
- IO:
	- $w_i$: $i$th word (i.e., space-separated tokens) in training corpus.
	- $t_{i,j}$: $j$th subtoken in word $w_i$.
- Hyperparams:
	- $M$: #words in training corpus.
	- $m_{w_i}$: #subtokens in word $w_i$.
- Params:
	- $\mathcal{V}$: Vocab of subword tokens.
- Misc:
	- $L(\mathcal{V})$: Corpus likelihood given curr vocab.
	- $a,b$: 2 selected tokens.
	- $c$: New token if merging $ab$.

Objective: Maximize Corpus Likelihood

$$
L(\mathcal{V})=\prod_{i=1}^{M}P(w_i|\mathcal{V})=\prod_{i=1}^{M}\prod_{j=1}^{m_{w_i}}t_{i,j}
$$

- Likelihood: #Counts of curr token over #Counts of all tokens in corpus from vocab:

$$
P(a)=\frac{\# a}{\sum_{b\in\mathcal{V}}\# b}
$$

Merging $a$&$b$ into $c$:

$$\begin{align*}
\# c\leftarrow \# c + \# ab \\
\# a\leftarrow \# a - \# ab	\\
\# b\leftarrow \# b - \# ab
\end{align*}$$

Likelihood Gain:

$$
\Delta(a,b)=\sum_{ab}\left[\log P_\mathrm{postmerge}(c)-\log P(a)-\log P(b)\right]
$$
```

### Unigram
- **What**: Reverse of BPE/Wordpiece → Start with a **large candidate vocabulary** & Repeatedly **prune tokens** while fitting a **unigram LM** over tokens
- **Why**: 
	- Pruning avoids committing to early irreversible merges.
	- Unigram explicitly models segmentation uncertainty → Produce more natural/diverse segmentations
- **How**:
	1. Build a large candidate vocab from training corpus.
		- **Candidate**: Every substring up to a max length (e.g., ~10–12 for alphabetic scripts).
		- **Large**: 10×~30× final vocab size.
	2. Assign token probabilities via a unigram LM.
		- Treat each token $t$ as a symbol with probability $p(t)$ ($\sum_t p(t)=1$).
		- A string $x$ has many valid segmentations $s=(t_1,\dots,t_k)$.
		- Each segmentation has probability $P(s)=\prod_i p(t_i)$.
		- Use DP to compute:
			- Training: **Forward–Backward** marginals / expected counts.
			- Inference: **Viterbi** best segmentation.
	3. Fit $p(t)$ with EM.
		- **E-step**: Compute expected token counts under $P(s|x)$ across the corpus.
		- **M-step**: Update probabilities from expected counts.
	4. Prune vocabulary iteratively.
		1. Score each token by how costly it is to remove (e.g., increase in NLL).
		2. Remove a fraction of **lowest-utility** tokens (e.g., 10–20%), keep required coverage tokens (special tokens), and retain an **UNK/fallback** mechanism
		3. Re-run EM on the reduced vocab.
		4. Repeat until reaching target vocab size.

```{note} Math
:class: dropdown
Notations:
- Data:
  - $x$: Input string.
  - $\mathcal{V}$: Curr vocab.
  - $\mathrm{Seg}(x)$: Set of all valid segmentations of $x$ using tokens in $\mathcal{V}$.
- Params:
  - $p(t)$: Unigram probability of token $t$, with $\sum_{t\in\mathcal{V}} p(t)=1$.
- Segmentation:
  - $s=(t_1,\dots,t_k)\in \mathrm{Seg}(x)$: A segmentation of $x$.
  - $\text{count}(t;s)$: #times token $t$ appears in segmentation $s$.

Model:
- Segmentation probability:
  $$
  P(s)=\prod_{i=1}^k p(t_i)
  $$
- String probability (sum over all segmentations):
  $$
  P(x)=\sum_{s\in \mathrm{Seg}(x)} P(s)
  =\sum_{s\in \mathrm{Seg}(x)} \prod_{i=1}^k p(t_i)
  $$
- Posterior over segmentations:
  $$
  P(s|x)=\frac{P(s)}{P(x)}
  $$
- Expected count of token $t$ under the posterior:
  $$
  \mathbb{E}[\text{count}(t)|x]
  =\sum_{s\in \mathrm{Seg}(x)} \text{count}(t;s)\,P(s|x)
  $$

EM updates (corpus-level):
- Let $C(t)=\sum_x \mathbb{E}[\text{count}(t)|x]$. Then:
  $$
  p(t)\leftarrow \frac{C(t)}{\sum_{t'\in\mathcal{V}} C(t')}
  $$

Inference (Viterbi):
- Find the best segmentation:
  $$
  s^*=\arg\max_{s\in \mathrm{Seg}(x)} P(s)
  =\arg\max_{s\in \mathrm{Seg}(x)} \sum_{i=1}^k \log p(t_i)
  $$
```

```{attention} Q&A
:class: dropdown
*How is this “reverse of BPE/WordPiece,” exactly?*  
- BPE/WordPiece: start from small units (chars/bytes) and **merge** to grow vocab.  
- Unigram: start from huge vocab of substrings and **prune** down while optimizing likelihood.

*Why do we need Forward–Backward instead of only Viterbi during training?*  
- Viterbi uses only the single best segmentation, which can over-commit early.  
- Forward–Backward spreads probability mass across *all* segmentations, giving smoother, more robust expected counts for EM.

*What guarantees coverage at inference time?*  
- In practice you keep a base set (often all single characters or bytes) and/or provide an UNK/fallback path, so any string can be segmented even if longer substrings were pruned.

*When does Unigram tend to shine?*  
- When you want **multiple plausible segmentations** during training (morphology-rich languages, mixed scripts, noisy text), and you’d rather not lock in merges that looked good early but age poorly as vocab grows.
```

&nbsp;

## Token Embedding
- **What**: Tokens → Semantic vectors.
- **Why**: 
	- Discrete tokens $\xrightarrow{\text{map to}}$ Continuous vectors
	- Vocab index $\xrightarrow{\text{map to}}$ Semantic meaning
	- Vocab size $\xrightarrow{\text{reduced to}}$ hidden size
- **How**: [Linear](../dl/module.md#linear).

```{note} Math
:class: dropdown
Notations:
- IO:
	- $T=(t_1,\cdots,t_m)$: Input token sequence (after tokenization).
    - $X\in\mathbb{R}^{m\times d_{\text{model}}}$: Output semantic vectors.
- Params:
    - $E\in\mathbb{R}^{V\times d_{\text{model}}}$: Embedding matrix/look-up table.
- Hyperparams:
    - $m$: #Tokens.
    - $d_{\text{model}}$: Embedding dim for the model.
    - $V$: Vocab size.

Token Embedding:

$$
X=\begin{bmatrix}
E_{t_1} \\
\vdots \\
E_{t_m}
\end{bmatrix}
$$
```

&nbsp;

## Positional Encoding
- **What**: Semantic vectors + Positional vectors → Position-aware vectors
- **Why**:
	- Transformers don't know positions.
	- BUT positions matter!
		- No PE → self-attention scores remain unchanged regardless of token orders {cite:p}`wang_positional_encoding`.

### Sinusoidal PE
- **What**: Positional vectors → Sine waves
- **Why**:
	- Continuous & multi-scale → Generalize to sequences of arbitrary lengths
	- No params → Low computational cost
	- Empirically performed as well as learned PE

```{note} Math
:class: dropdown
Notations:
- IO:
	- $pos\in\mathbb{R}$: Input token position.
- Hyperparams:
	- $i$: Embedding dimension index.
	- $d_{\text{model}}$: Embedding dimension.

Sinusoidal PE:

$$\begin{align*}
&PE_{(pos, 2i)}=\sin\left(\frac{pos}{10000^{\frac{2i}{d_{\text{model}}}}}\right) \\
&PE_{(pos, 2i+1)}=\cos\left(\frac{pos}{10000^{\frac{2i}{d_{\text{model}}}}}\right)
\end{align*}$$
```

```{attention} Q&A
:class: dropdown
*Cons?*
- No params → No learning of task-specific position patterns.
- Requires uniform token importance across the sequence. {cite:p}`vaswani2017attention`
- Cannot capture complex, relative, or local positional relationships.
```

### RoPE
- **What**: Rotary Postion Embedding.
	- Encode relative positions ← Rotate each QKV pair.
- **Why**:
	- Absolute PE tie each token to a fixed index → NO generalization to longer sequences
	- Learned relative PE learn one weight per distance bucket → Limit distance ranges & Add params
	- RoPE: **Param-free, Continuous, Relative, Generalizable**.
- **How**:
	1. Project each token embedding at position index $p$ to $\mathbf{q}_p$ & $\mathbf{k}_p$.
	2. Each query/key vector of hidden dim $d$ = $\frac{d}{2}$ two-component planes formed by even-odd feature pairs.
	3. Assign a rotation angle $\theta_n$, which grows smoothly with two-component pair index $n$, to each token.
	4. Define a fixed 2D rotation operator $R(\theta_n)$, which rotates every plane counterclockwise by the angle $\theta_n$.
	5. Rotate query & key vectors.
	6. Compute attention scores with rotated query & key vectors.
		- The paired rotation ONLY depends on the relative offset between their positions.

```{note} Math
:class: dropdown
I forgot everything about Complex Analysis, so I will write in the layman's way.

Notations:
- IO:
	- $\mathbf{x}\in\mathbb{R}^d$: Input token embedding.
- Hyperparam:
	- $d$: Hidden dim.
- Misc:
	- $i,j$: Token position indices.
	- $n\in[1,\cdots,\frac{d}{2}]$: 2D plane index.
	- $\theta_n$: Angle assigned to plane $n$.
	- $R_\theta$: Rotation matrix for angle $\theta$.
	- $\mathbf{q}_i$: Query vector for $i$th token.
	- $\mathbf{k}_j$: Key vector for $j$th token.

2D plane:
1. The input token vector can be viewed as a sequence of consecutive pairs:

$$
\mathbf{x}=\left[(x_1,x_2),\cdots,(x_{d-1},x_d)\right]
$$

2. Each pair $(x_{2n-1},x_{2n})$ = A point in a 2D Cartesian plane.
3. For each pair, define a rotation angle, which is smoothly dependent on the plane index:

$$
\theta_n=10000^{-\frac{2(n-1)}{d}}
$$

4. For each pair, define a rotation matrix:

$$
R_{\theta_n}=\begin{bmatrix}
\cos\theta_n & -\sin\theta_n \\
\sin\theta_n & \cos\theta_n
\end{bmatrix}
$$

5. Note the core property of rotation matrices:

$$
R_\theta^TR_\phi=R_{\phi-\theta}
$$

6. For each pair, rotate counterclockwise (i.e., Cartesian → Polar → Cartesian):

$$
\begin{bmatrix}
x'_{2n-1} \\ x'_{2n}
\end{bmatrix}=R_{\theta_n}\begin{bmatrix}
x_{2n-1} \\ x_{2n}
\end{bmatrix}=\begin{bmatrix}
\cos\theta_n & -\sin\theta_n \\
\sin\theta_n & \cos\theta_n
\end{bmatrix}\begin{bmatrix}
x_{2n-1} \\ x_{2n}
\end{bmatrix}
$$

RoPE:
1. For a given query vector $\mathbf{q}_i$ and key vector $\mathbf{k}_j$, apply RoPE to each 2D plane.
	- NOTE: The angle fed into the rotation matrix is ALSO dependent on the **position**.

$$\begin{align*}
{\mathbf{q}'}_i^{(n)}&=R_{i\theta_n}\mathbf{q}_i^{(n)} \\
{\mathbf{k}'}_j^{(n)}&=R_{j\theta_n}\mathbf{k}_j^{(n)}
\end{align*}$$

2. For each pair, compute dot product, which is ONLY dependent on the vector values and the relative distance between $i$ & $j$:

$$\begin{align*}
\left({\mathbf{q}'}_i^{(k)}\right)^T{\mathbf{k}'}_j^{(k)}&=\left(R_{i\theta_n}\mathbf{q}_i^{(n)}\right)^TR_{j\theta_n}\mathbf{k}_j^{(n)} \\
&=\left(\mathbf{q}_i^{(n)}\right)^TR_{i\theta_n}^TR_{j\theta_n}\mathbf{k}_j^{(n)} \\
&=\left(\mathbf{q}_i^{(n)}\right)^TR_{(j-i)\theta_n}\mathbf{k}_j^{(n)}
\end{align*}$$
```

```{attention} Q&A
:class: dropdown
*That's genius. How did they come up with this?*
- They appreciate relative PE, but they don't appreciate the additional trainable params.
- Let $\mathbf{q}_i$ & $\mathbf{k}_j$ be query & key vectors at positions $i$ and $j$ respectively.
- They want their dot product to ONLY depend on the input vectors and the relative distance $i-j$:

$$
f(\mathbf{q},i)\cdot f(\mathbf{k},j) = g(\mathbf{q},\mathbf{k},i-j)
$$

- This is best achieved by rotation as explained in the Math section.

*Why define the angle that way?*
- Well, they didn't define it that way, but the OG [Sinusoidal PE](#sinusoidal-pe) did.
- Long-term decay: This function ensures that the inner-product decays as the relative distance increases.
- Varying granularity:
	- Early dimensions have large angles → Rotation changes significantly even between nearby tokens → Effectively capture fine-grained relationship
	- Late dimensions have small angles → Rotation changes very slowly, almost identical between nearby tokens → They ONLY make a significant difference with distant tokens → Effectively capture coarse-grained relationship
```

&nbsp;

## Attention
- **What**: Different weights for different parts.
    - Focus on important parts & Diffuse on trivial parts.
- **Why**: Enable models to dynamically align & retrieve most relevant parts of input sequence when generating each output token.

### Self-Attention
- **What**: Each token in the sequence pays attention to all other tokens in the same sequence, to produce **context-aware representations**.
- **Why**: **Long-range dependencies** + **Parallel processing**
- **How**: Information Retrieval.
	1. All tokens $\xrightarrow{\text{convert to}}$ $Q$,$K$,$V$
		- $Q$: What are you looking for?
		- $K$: What are the keywords for identification?
		- $V$: What is the content?
	2. For each token $t$:
        1. Multiply its query & all tokens' keys → Relevance scores
		2. Scale & Softmax the scores → Attention weights
		3. Weighted sum of all tokens' values → $t$'s contextual representation

```{note} Math
:class: dropdown
Notations:
- Input:
	- $X\in\mathbb{R}^{m\times d_{\text{model}}}$: Input sequence
- Params:
	- $W_Q\in\mathbb{R}^{d_{\text{model}}\times d_K}$: Weight matrix for Q.
	- $W_K\in\mathbb{R}^{d_{\text{model}}\times d_K}$: Weight matrix for K.
	- $W_V\in\mathbb{R}^{d_{\text{model}}\times d_V}$: Weight matrix for V.
- Hyperparams:
	- $m$: #Tokens.
	- $d_{\text{model}}$: Embedding dim of input sequence.
	- $d_K$: Embedding dim of Q & K.
		- Q & K share the same embedding dim for matrix multiplication.
	- $d_V$: Embedding dim of V (practically the same as $d_K$).
- Misc:
	- $Q=XW_Q\in\mathbb{R}^{m\times d_K}$: Q vectors for all tokens.
	- $K=XW_K\in\mathbb{R}^{m\times d_K}$: K vectors for all tokens.
	- $V=XW_V\in\mathbb{R}^{m\times d_V}$: V vectors for all tokens.

Scaled Dot-Product Attention:

$$
\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^T}{\sqrt{d_K}}\right)V
$$
```

```{tip} Derivation (Backprop)
:class: dropdown
Notations:
- $S=\frac{QK^T}{\sqrt{d_K}}$
- $A=\text{softmax}(S)$
- $Y=AV$

Process:
1. V:

$$
\frac{\partial\mathcal{L}}{\partial V}=A^T\frac{\partial\mathcal{L}}{\partial Y}
$$

2. A:

$$
\frac{\partial\mathcal{L}}{\partial A}=\frac{\partial\mathcal{L}}{\partial Y}V^T
$$

3. S:

- Recall that for $\mathbf{a}=\text{softmax}(\mathbf{s})$:

	$$
	\frac{\partial a_i}{\partial s_j}=a_i(\delta_{ij}-a_j)
	$$

	where $\delta_{ij}=1\text{ if }i=j\text{ else }0$.

- For each row $i$ of $S$:

	$$\begin{align*}
	\frac{\partial\mathcal{L}}{\partial S_{ij}}&=\sum_{k=1}^{m}\frac{\partial\mathcal{L}}{\partial A_{ik}}\frac{\partial A_{ik}}{\partial S_{ij}} \\
	&=\frac{\partial\mathcal{L}}{\partial A_{ij}}A_{ij}-A_{ij}\frac{\partial\mathcal{L}}{\partial A_{ij}}A_{ij}-A_{ij}\sum_{k\neq j}\frac{\partial\mathcal{L}}{\partial A_{ik}}A_{ik} \\
	&=\frac{\partial\mathcal{L}}{\partial A_{ij}}A_{ij}-A_{ij}\sum_{k=1}^{m}\frac{\partial\mathcal{L}}{\partial A_{ik}}A_{ik}
	\end{align*}$$

4. Q&K:

$$\begin{align*}
&\frac{\partial\mathcal{L}}{\partial Q}=\frac{\partial\mathcal{L}}{\partial S}\frac{K}{\sqrt{d_K}} \\
&\frac{\partial\mathcal{L}}{\partial K}=\frac{\partial\mathcal{L}}{\partial S}^T\frac{Q}{\sqrt{d_K}}
\end{align*}$$

5. Ws:

$$\begin{align*}
&\frac{\partial\mathcal{L}}{\partial W_Q}=X^T\frac{\partial\mathcal{L}}{\partial Q}\\
&\frac{\partial\mathcal{L}}{\partial W_K}=X^T\frac{\partial\mathcal{L}}{\partial K}\\
&\frac{\partial\mathcal{L}}{\partial W_V}=X^T\frac{\partial\mathcal{L}}{\partial V}
\end{align*}$$

6. X:

$$
\frac{\partial\mathcal{L}}{\partial X}=\frac{\partial\mathcal{L}}{\partial Q}W_Q^T+\frac{\partial\mathcal{L}}{\partial K}W_K^T+\frac{\partial\mathcal{L}}{\partial V}W_V^T
$$
```

```{attention} Q&A
:class: dropdown
*Cons?*
- ⬆️ Computational cost ← $O(n^2)$ (?)
- Fixed sequence length.

*Why scale?*
1. Dot product scales with dimension size.
2. Assume elements follow $\mathcal{N}(0,1)$, then dot product follows $\mathcal{N}(0,d_K)$.
3. Scaling normalizes this variance.
	
*Why softmax?*
- Scores → Probability distribution
	- All weights > 0.
	- All weights sum to 1.
- Score margins are amplified → More attention to relevant elements
```

### Masked/Causal Attention
- **What**: Self-attention BUT each token can only see its previous tokens (and itself).
- **Why**: Autoregressive generation.
- **How**: For each token, mask attention scores of all future tokens to $-\infty$ before softmax.
	- $\text{softmax}(-\infty)$=0

```{note} Math
:class: dropdown
Causal Attention:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
$$
- $M\in\mathbb{R}^{m\times m}$: Mask matrix
	- $M_{ij}=0\text{ if }i\geq j$ else $-\infty$
```

```{attention} Q&A
:class: dropdown
*Conditions?*
- Only applicable in decoder.
	- Encoder's goal: Convert sequence into a meaningful representation.
	- Decoder's goal: predict next token.

*Cons?*
- Unidirectional context.
- Limited context for early tokens.
	- Token 1 only sees 1 token.
	- Token 2 only sees 2 tokens.
	- ...
```

### Cross Attention
- **What**: Scaled Dot-Product Attention BUT
	- K&V ← Source (e.g., Encoder)
	- Q ← Curr sequence (i.e., Decoder)
- **Why**: Additional source info may be helpful for predicting next token for curr sequence.
- **How**: See [Self-Attention](#self-attention) but $K \& V$ are from Encoder.

### Multi-Head Attention
- **What**: Multiple self-attention modules running in parallel.
- **Why**:
	- 1 attention module $\xrightarrow{\text{monitor}}$ 1 representation subspace
	- Language is complex: morphology, syntax, semantics, context, ...
	- $h$ attention modules $\xrightarrow{\text{monitor}}$ $h$ representation subspaces
- **How**:
    1. Each head takes the input token sequence and generates an output via self-attention.
    2. Concatenate all outputs.
    3. Linear transform to match embedding dim.

```{note} Math
:class: dropdown
Notations:
- Params:
	- $W_O\in\mathbb{R}^{(h\cdot d_v)\times d_{\text{model}}}$: Weight matrix to transform concatenated head outputs.
- Hyperparams:
	- $h$: #Heads.
- Intermediate values:
	- $\text{head}_i\in\mathbb{R}^{m\times d_V}$: Weighted contextual representation of input sequence from head $i$.

MHA:

$$
\text{MultiHead}(Q,K,V)=\text{Concat}(\text{head}_1,\cdots,\text{head}_h)W_O
$$
```

```{attention} Q&A
:class: dropdown
*Cons?*
- ⬆️ Computational cost
- ⬇️ Interpretability
- Redundancy ← Some heads may learn similar patterns
```

&nbsp;

## Encoder
- **What**:
    1. MHA + Residual Connection + LayerNorm
    2. Position-wise FFN + Residual Connection + LayerNorm
- **Why**: MHA merely forms a softmax-weighted linear blend of other tokens' value vectors → Where is non-liearity & info complexity?
    - **Non-linearity**: Two-layer FFN with ReLU in between.
    - **Info complexity**: Each token gets its own info processing stage: "Diffuse → Activate → Compress".
        - A mere weighted sum of value vectors → Higher-order feature mixes.
- **How**:
    1. **MHA**: Immediately establish global context from input sequence BEFORE per-token processing.
        1. **Residual Connection**: Preserve the original signal & Ensure training stability for deep stacks.
        2. **LayerNorm**: Re-center & Rescale outputs to curb [covariate shift](../dl/issues.md#covariate-shift).
    2. **Position-wise FFN**: Apply FFN independently **to each token vector** → Transform features within each position's channel dimension, w/o exchanging info across positions.
        1. **Residual Connection**: Preserve the original signal & Ensure training stability for deep stacks.
        2. **LayerNorm**: Re-center & Rescale outputs to curb [covariate shift](../dl/issues.md#covariate-shift).

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{x}\in\mathbb{R^{d_{\text{model}}}}$: Input token vector.
- Params:
    - $W_1\in\mathbb{R}^{d_{\text{model}}\times d_{\text{FFN}}}, \mathbf{b}_1\in\mathbb{R}^{d_{\text{FFN}}}$: Diffuse weights & biases.
    - $W_2\in\mathbb{R}^{d_{\text{FFN}}\times d_{\text{model}}}, \mathbf{b}_2\in\mathbb{R}^{d_{\text{model}}}$: Compress weights & biases.

Position-wise FFN:

$$
FFN(\mathbf{x})=\max(0,\mathbf{x}W_1+\mathbf{b}_1)W_2+\mathbf{b}_2
$$
```

## Decoder
- **What**:
    1. Masked MHA + Residual Connection + LayerNorm
    2. (Optional) Cross MHA + Residual Connection + LayerNorm
    3. Position-wise FFN + Residual Connection + LayerNorm
- **Why**: See [Masked MHA](#maskedcausal-attention), [Cross MHA](#cross-attention) and [Position-wise FFN](#encoder).
    - BUT why cross after mask?
        - The goal of decoder is Next Token Prediction.
        - In order to predict anything, we need the context first.
        - Masked MHA provides AR context for the target token.
        - THEN, we can add supplementary info from Cross MHA.
        - THEN, we do per-token processing.

## Output
- **What**: Embeddings → Token probability distribution.
- **Why**: Next Token Prediction.
- **How**: Linear + Softmax.
    - **Linear**: Shape conversion: Embedding dim → Vocab size.
    - **Softmax**: Logits → Probs

## References
- [Jay Alammar's Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [The OG transformer paper](https://arxiv.org/pdf/1706.03762)
- [HF's LLM course](https://huggingface.co/learn/llm-course/chapter1/1?fw=pt)
- [Yi Wang's Positional Encoding](https://wangkuiyi.github.io/positional_encoding.html)