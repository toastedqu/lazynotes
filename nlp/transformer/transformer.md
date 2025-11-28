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
- **What**: **Self-attention** for modeling relationships between all tokens in input sequence.
- **Why**: **Long-range dependencies** + **Parallel processing**.
- **How**: {cite:p}`vaswani2017attention`

```{image} ../../images/transformer/transformer.png
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
- **What**: Sequence $\rightarrow$ Tokens
- **Why**: Machines can only read numbers.

```{attention} Q&A
:class: dropdown
*Why subword-level vocab? Why not whole words? Why not characters?*
- Word-level vocab explode with out-of-vocab words.
- Char-level vocab misses morphology.
- Subword offers a fixed-size, open-vocab symbol set which can handle rare words while maintaining morphology.
```

<br/>

## Token Embedding
- **What**: Tokens $\rightarrow$ Semantic vectors.
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

<br/>

## Positional Encoding
- **What**: Semantic vectors + Positional vectors $\rightarrow$ Position-aware vectors
- **Why**: Transformers don't know positions due to parallelism, BUT positions matter.
    - No PE $\rightarrow$ Self-attention scores remain unchanged regardless of token orders
- **How**: Add positional vectors onto semantic vectors.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $X\in\mathbb{R}^{m\times d_{\text{model}}}$: Input/Output semantic vectors.
- Params:
    - $P\in\mathbb{R}^{m\times d_{\text{model}}}$: Positional embedding vectors.

Positional Encoding:

$$
X\leftarrow X+P
$$
```

<br/>

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
        1. Multiply its query & all tokens' keys $\rightarrow$ Relevance scores
		2. Scale & Softmax the scores $\rightarrow$ Attention weights
		3. Weighted sum of all tokens' values $\rightarrow$ $t$'s contextual representation

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
- ⬆️ Computational cost $\leftarrow$ $O(n^2)$ (?)
- Fixed sequence length.

*Why scale?*
1. Dot product scales with dimension size.
2. Assume elements follow $\mathcal{N}(0,1)$, then dot product follows $\mathcal{N}(0,d_K)$.
3. Scaling normalizes this variance.
	
*Why softmax?*
- Scores $\rightarrow$ Probability distribution
	- All weights > 0.
	- All weights sum to 1.
- Score margins are amplified $\rightarrow$ More attention to relevant elements
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
	- K&V $\leftarrow$ Source (e.g., Encoder)
	- Q $\leftarrow$ Curr sequence (i.e., Decoder)
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
- Redundancy $\leftarrow$ Some heads may learn similar patterns
```

<br>

## Encoder
- **What**:
    1. MHA + Residual Connection + LayerNorm
    2. Position-wise FFN + Residual Connection + LayerNorm
- **Why**: MHA merely forms a softmax-weighted linear blend of other tokens' value vectors $\rightarrow$ Where is non-liearity & info complexity?
    - **Non-linearity**: Two-layer FFN with ReLU in between.
    - **Info complexity**: Each token gets its own info processing stage: "Diffuse $\rightarrow$ Activate $\rightarrow$ Compress".
        - A mere weighted sum of value vectors $\rightarrow$ Higher-order feature mixes.
- **How**:
    1. **MHA**: Immediately establish global context from input sequence BEFORE per-token processing.
        1. **Residual Connection**: Preserve the original signal & Ensure training stability for deep stacks.
        2. **LayerNorm**: Re-center & Rescale outputs to curb [covariate shift](../dl/issues.md#internal-covariate-shift).
    2. **Position-wise FFN**: Apply FFN independently **to each token vector** $\rightarrow$ Transform features within each position's channel dimension, w/o exchanging info across positions.
        1. **Residual Connection**: Preserve the original signal & Ensure training stability for deep stacks.
        2. **LayerNorm**: Re-center & Rescale outputs to curb [covariate shift](../dl/issues.md#internal-covariate-shift).

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
- **What**: Embeddings $\rightarrow$ Token probability distribution.
- **Why**: Next Token Prediction.
- **How**: Linear + Softmax.
    - **Linear**: Shape conversion: Embedding dim $\rightarrow$ Vocab size.
    - **Softmax**: Logits $\rightarrow$ Probs

## References
- [Jay Alammar's Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [The OG transformer paper](https://arxiv.org/pdf/1706.03762)
- [HF's LLM course](https://huggingface.co/learn/llm-course/chapter1/1?fw=pt)
- [Yi Wang's Positional Encoding](https://wangkuiyi.github.io/positional_encoding.html)