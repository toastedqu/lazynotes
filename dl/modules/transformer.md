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
```{image} ../../images/transformer.png
:align: center
:width: 500px
```
- **What**: **Self-attention** for sequential data.
- **Why**: **Long-range dependencies** + **Parallel processing**.
- **How**:
	- **Input**:
		- [Token Embedding](#token-embedding): Converts each word/token into a high-dimensional vector for **semantic information**.
		- [Positional Encoding](#positional-encoding): Adds **positional information** on top of embeddings to preserve token order.
	- **Middle**:
		- [Encoder](#encoder): Encodes the input into a **contextual representation**.
		- [Decoder](#decoder): Decodes the contextual representation **one token at a time** to form output.
	- **Output**:
		- [Linear](#linear): Transforms representation for next-token prediction.
		- [Softmax](#softmax): Outputs probability for each token as the next token.
- **Conditions**:
	- Discrete inputs (in the form of tokens/patches).
	- Sequential input + Global dependencies.
- **Applications**: Sequential data.
- **Pros**:
	- Outperforms RNNs.
	- Higher computational efficiency than RNNs.
	- Reduces Vanishing Gradients.
	- Scalable.
	- Flexible for extensive applications -> Enables transfer learning.
- **Cons**:
	- **Quadratic complexity** w.r.t. sequence length.
	- Slower for autoregressive tasks (i.e., generation).
	- High data requirement.
	- High computation requirement.
	- Low interpretability.
	- RIP environment.

## Input
### Token Embedding
- **What**: Discrete tokens -> Continuous **semantic representations**.
- **Why**:
	- Enables transformer to process discrete tokens.
	- Captures tokens' intrinsic semantic meanings.
	- Reduces **vocabulary** **dimensionality** (vocab size -> hidden size).
- **How**: Look-up table or [Linear](../basics.md#linear).

### Positional Encoding
- **What**: Adds **positional information** to each token in the sequence.
- **Why**: Transformers lack inherent position awareness AND positions matter.
	- Without positional encoding, self-attention scores remain unchanged regardless of token orders (See [Yi Wang's blog post](https://wangkuiyi.github.io/positional_encoding.html)).
- **When**: Positions matter.

#### Sinusoidal PE
- **What**: Sin & Cos at varying frequencies to encode positions.
- **Why**:
	- Provides **continuous, multi-scale position representations** that can generalize to sequences of any length.
	- Empirically performed well (same as learned PE).
- **Conditions**:
	- Positional information is expressed by multi-frequency sine/cosine waves.
	- Uniform token importance across the sequence.
- **Applications**: The OG Transformer.
- **Pros**:
	- High computational efficiency <- No param.
	- Extrapolates to unseen sequence lengths.
	- Continuous positions (discrete)
	- Multi-scale positional representations (as an embedding vector).
- **Cons**:
	- No param -> No learning of task-specific position patterns.
	- Cannot capture complex, relative, or local positional relationships.

```{admonition} Math
:class: note, dropdown
Notations:
- IO:
	- $pos\in\mathbb{R}$: Token position.
- Hyperparams:
	- $i$: Embedding dimension index.
	- $d_{\text{model}}$: Embedding dimension.

Forward:

$$\begin{align}
&PE_{(pos, 2i)}=\sin\left(\frac{pos}{10000^{\frac{2i}{d_{\text{model}}}}}\right) \\
&PE_{(pos, 2i+1)}=\cos\left(\frac{pos}{10000^{\frac{2i}{d_{\text{model}}}}}\right)
\end{align}$$
```

## Encoder
- **What**: Sequence -> **Contextual representation**.
- **Why**: To produce more meaningful representations (context + semantics + position).
- **How**:
	1. [Multi-Head Attention](#multi-head-attention): Applies attention across all tokens.
	2. [Residual Connection](#residual-connection): Adds contextual info to the original input, to **prevent info loss** and **make gradients smooth**.
	3. [Layer Normalization](#layer-normalization): Enhances training stability & speed.
	4. [Feed-Forward Network](#feed-forward-network): Refines the representation & Captures additional complex patterns.
- **Where**: Inference models (e.g., BERT family).

### Multi-Head Attention

#### Self-Attention

#### Scale Dot-Product Attention
```{image} ../../images/scaled_dot_product_attention.png
:align: center
:width: 250px
```

### Residual Connection

### Layer Normalization

### Feed-Forward Network

## Decoder
- **What**: Encoded representation -> **Sequence**.
- **Why**: To produce context-aware outputs via encoder's info & previously generated tokens.
- **How**:
	1. [Masked Multi-Head Attention](#masked-multi-head-attention): Attends ONLY to past tokens in the target sequence, ensuring autoregressive output.
	2. [Encoder-Decoder Cross-Attention](#encoder-decoder-cross-attention): Integrates input sequence context into output generation.
	3. [Residual Connection](#residual-connection): Adds contextual info to the original input, to **prevent info loss** and **make gradients smooth**.
	4. [Layer Normalization](#layer-normalization): Enhances training stability & speed.
	5. [Feed-Forward Network](#feed-forward-network): Refines the representation & Captures additional complex patterns.
- **Where**: Generative models (e.g., GPT family)

### Masked Multi-Head Attention

### Encoder-Decoder Cross-Attention

## Output

### Linear
See [Linear](../modules/basics.md#linear)

### Softmax
See [Softmax](../modules/activations.md#softmax)