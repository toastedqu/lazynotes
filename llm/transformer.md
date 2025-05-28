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
- **What**: **Self-attention** for sequential data.
- **Why**: **Long-range dependencies** + **Parallel processing**.
- **How**: {cite:p}`vaswani2017attention`
```{image} ../images/transformer/input.png
:align: center
:width: 500px
```

# Input
- **How**:
```{image} ../images/transformer/input.png
:align: center
:width: 400px
```

## Tokenization
- **What**: Sequence $\rightarrow$ Tokens
- **Why**: Machines can only read numbers.
- **How**: (tbd)

<br/>

## Token Embedding
- **What**: Tokens $\rightarrow$ Semantic vectors.
- **Why**:
	- Discrete $\rightarrow$ Continuous
	- Vocab index $\rightarrow$ Semantic meaning
	- Vocab size $\xrightarrow{\text{reduced to}}$ hidden size
- **How**: Look-up table / [Linear](../basics.md#linear).

<br/>

## Positional Encoding
- **What**: Semantic vectors + Positional info $\rightarrow$ Position-aware vectors
- **Why**:
	- Transformers don't know positions.
	- BUT positions matter!
		- No PE $\rightarrow$ self-attention scores remain unchanged regardless of token orders {cite:p}`wang_positional_encoding`.

### Sinusoidal PE
- **What**: Positional info $\rightarrow$ Sine waves
- **Why**:
	- Continuous & multi-scale $\rightarrow$ Generalize to sequences of arbitrary lengths
	- No params $\rightarrow$ Low computational cost
	- Empirically performed as well as learned PE

```{admonition} Math
:class: note, dropdown
Sinusoidal PE:

$$\begin{align*}
&PE_{(pos, 2i)}=\sin\left(\frac{pos}{10000^{\frac{2i}{d_{\text{model}}}}}\right) \\
&PE_{(pos, 2i+1)}=\cos\left(\frac{pos}{10000^{\frac{2i}{d_{\text{model}}}}}\right)
\end{align*}$$
- Input:
	- $pos\in\mathbb{R}$: Token position.
- Hyperparams:
	- $i$: Embedding dimension index.
	- $d_{\text{model}}$: Embedding dimension.
```

```{admonition} Q&A
:class: tip, dropdown
*Cons?*
- No params $\rightarrow$ No learning of task-specific position patterns.
- Requires uniform token importance across the sequence. {cite:p}`vaswani2017attention`
- Cannot capture complex, relative, or local positional relationships.
```

### RoPE
- **What**: Rotation matrix $\times$ Token embeddings $\xrightarrow{\text{encode}}$ Relative Position.
- **Why**: 


<br/>

## Attention
### Self-Attention
- **What**: Each element in the sequence pays attention to each other.
- **Why**: **Long-range dependencies** + **Parallel processing**
- **How**:
	1. All elements $\rightarrow$ QKV
		- Q: What are you looking for?
		- K: What are your keywords for search?
		- V: What info do you have?
	2. For each token T:
		1. T's Query & All Keys $\rightarrow$ Relevance scores
		2. $\rightarrow$ Attention weights
		3. $\rightarrow$ Weighted sum of T's Value (i.e., T's contextual representation)

```{dropdown} ELI5
You are in a top AI conference.

Each guy is an element.

You have some dumb question in mind. (Q)

Each guy has their badges and posters with titles and metadata. (K)

Each guy knows the details of their projects. (V)

You walk around & check out the whole venue.

You see topics that you don't really care. You skim & skip.

You see topics that are related to your question. You talk to the guys to learn more.

You see topics that you are obsessed with. You ask the guys a billion follow-up questions and memorize every single technical detail of their Github implementation.

The conference ends.

You have learnt something about everything, but not everything weighs the same in your heart.
```

```{admonition} Math
:class: note, dropdown
Scaled Dot-Product Attention:

$$
\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^T}{\sqrt{d_K}}\right)V
$$
- Input:
	- $X\in\mathbb{R}^{m\times n}$: Input sequence
- Params:
	- $W_Q\in\mathbb{R}^{n\times d_K}$: Weight matrix for Q.
	- $W_K\in\mathbb{R}^{n\times d_K}$: Weight matrix for K.
	- $W_V\in\mathbb{R}^{n\times d_V}$: Weight matrix for V.
- Hyperparams:
	- $m$: #Tokens.
	- $n$: Hidden size.
	- $d_K$: Hidden size of Q & K.
		- Q & K share the same hidden size for matrix multiplication.
	- $d_V$: Hidden size of V.
- Intermediate values:
	- $Q=XW_Q\in\mathbb{R}^{m\times d_K}$: Q vectors for all elems.
	- $K=XW_K\in\mathbb{R}^{m\times d_K}$: K vectors for all elems.
	- $V=XW_V\in\mathbb{R}^{m\times d_V}$: V vectors for all elems.
```

```{admonition} Derivation (Backprop)
:class: important, dropdown
Notations:
- $S=\frac{QK^T}{\sqrt{d_K}}$
- $A=\text{softmax}(S)$
- $Y=AV$

Process:
1. V:

$$
\frac{\partial L}{\partial V}=A^T\frac{\partial L}{\partial Y}
$$

2. A:

$$
\frac{\partial L}{\partial A}=\frac{\partial L}{\partial Y}V^T
$$

3. S:

- Recall that for $\mathbf{a}=\text{softmax}(\mathbf{s})$:

	$$
	\frac{\partial a_i}{\partial s_j}=a_i(\delta_{ij}-a_j)
	$$

	where $\delta_{ij}=1\text{ if }i=j\text{ else }0$.

- For each row $i$ of $S$:

	$$\begin{align*}
	\frac{\partial L}{\partial S_{ij}}&=\sum_{k=1}^{m}\frac{\partial L}{\partial A_{ik}}\frac{\partial A_{ik}}{\partial S_{ij}} \\
	&=\frac{\partial L}{\partial A_{ij}}A_{ij}-A_{ij}\frac{\partial L}{\partial A_{ij}}A_{ij}-A_{ij}\sum_{k\neq j}\frac{\partial L}{\partial A_{ik}}A_{ik} \\
	&=\frac{\partial L}{\partial A_{ij}}A_{ij}-A_{ij}\sum_{k=1}^{m}\frac{\partial L}{\partial A_{ik}}A_{ik}
	\end{align*}$$

4. Q&K:

$$\begin{align*}
&\frac{\partial L}{\partial Q}=\frac{\partial L}{\partial S}\frac{K}{\sqrt{d_K}} \\
&\frac{\partial L}{\partial K}=\frac{\partial L}{\partial S}^T\frac{Q}{\sqrt{d_K}}
\end{align*}$$

5. Ws:

$$\begin{align*}
&\frac{\partial L}{\partial W_Q}=X^T\frac{\partial L}{\partial Q}\\
&\frac{\partial L}{\partial W_K}=X^T\frac{\partial L}{\partial K}\\
&\frac{\partial L}{\partial W_V}=X^T\frac{\partial L}{\partial V}
\end{align*}$$

6. X:

$$
\frac{\partial L}{\partial X}=\frac{\partial L}{\partial Q}W_Q^T+\frac{\partial L}{\partial K}W_K^T+\frac{\partial L}{\partial V}W_V^T
$$
```

```{admonition} Q&A
:class: tip, dropdown
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

```{admonition} Math
:class: note, dropdown
Causal Attention:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
$$
- $M\in\mathbb{R}^{m\times m}$: Mask matrix
	- $M_{ij}=0\text{ if }i\geq j$ else $-\infty$
```

```{admonition} Q&A
:class: tip, dropdown
*Conditions?*
- Only applicable in decoder.
	- Main goal of encoder: convert sequence into a meaningful representation.
	- Main goal of decoder: predict next token.

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
	- Q $\leftarrow$ Current sequence (i.e., Decoder)
- **Why**: Additional source info may be helpful for predicting next token for current sequence.
- **How**: See [Self-Attention](#self-attention).

### Multi-Head Attention
- **What**: Multiple self-attention modules running in parallel.
- **Why**:
	- $1$ attention module $\xrightarrow{\text{monitor}}$ $1$ representation subspace
	- Language is complex: morphology, syntax, semantics, context, ...
	- $h$ attention modules $\xrightarrow{\text{monitor}}$ $h$ representation subspaces
- **How**: Each head $\xrightarrow{\text{self-attention}}$ Each output $\xrightarrow{\text{concatenate}}$ All outputs $\xrightarrow{\text{linear transform}}$ Final output

```{admonition} Math
:class: note, dropdown
MHA:

$$
\text{MultiHead}(Q,K,V)=\text{Concat}(\text{head}_1,\cdots,\text{head}_h)W_O
$$
- Params:
	- $W_O$: Weight matrix to transform concatenated head outputs.
- Hyperparams:
	- $h$: #Heads.
- Intermediate values:
	- $\text{head}_i\in\mathbb{R}^{m\times d_V}$: Weighted representation of input sequence from the $i$th head.
```

```{admonition} Q&A
:class: tip, dropdown
*Cons?*
- ⬆️ Computational cost
- ⬇️ Interpretability
- Redundancy $\leftarrow$ some heads may learn similar patterns
```

<br/>

<!-- # Encoder
- **What**: Sequence -> **Contextual representation**.
- **Why**: To produce more meaningful representations (context + semantics + position).
- **How**:
	1. [Multi-Head Attention](#multi-head-attention): Applies attention across all tokens.
	2. [Residual Connection](#residual-connection): Adds contextual info to the original input, to **prevent info loss** and **make gradients smooth**.
	3. [Layer Normalization](#layer-normalization): Enhances training stability & speed.
	4. [Feed-Forward Network](#feed-forward-network): Refines the representation & Captures additional complex patterns.
- **Where**: Inference models (e.g., BERT family).

## Multi-Head Attention

## Scale Dot-Product Attention
```{image} ../../images/scaled_dot_product_attention.png
:align: center
:width: 250px
```

## Residual Connection

## Layer Normalization

## Feed-Forward Network

# Decoder
- **What**: Encoded representation -> **Sequence**.
- **Why**: To produce context-aware outputs via encoder's info & previously generated tokens.
- **How**:
	1. [Masked Multi-Head Attention](#masked-multi-head-attention): Attends ONLY to past tokens in the target sequence, ensuring autoregressive output.
	2. [Encoder-Decoder Cross-Attention](#encoder-decoder-cross-attention): Integrates input sequence context into output generation.
	3. [Residual Connection](#residual-connection): Adds contextual info to the original input, to **prevent info loss** and **make gradients smooth**.
	4. [Layer Normalization](#layer-normalization): Enhances training stability & speed.
	5. [Feed-Forward Network](#feed-forward-network): Refines the representation & Captures additional complex patterns.
- **Where**: Generative models (e.g., GPT family)

## Masked Multi-Head Attention

## Encoder-Decoder Cross-Attention

# Output

## Linear
See [Linear](../modules/basics.md#linear)

## Softmax
See [Softmax](../modules/activations.md#softmax) -->