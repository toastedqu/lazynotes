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


<!-- ## Encoder
- **What**: Sequence -> **Contextual representation**.
- **Why**: To produce more meaningful representations (context + semantics + position).
- **How**:
	1. [Multi-Head Attention](#multi-head-attention): Applies attention across all tokens.
	2. [Residual Connection](#residual-connection): Adds contextual info to the original input, to **prevent info loss** and **make gradients smooth**.
	3. [Layer Normalization](#layer-normalization): Enhances training stability & speed.
	4. [Feed-Forward Network](#feed-forward-network): Refines the representation & Captures additional complex patterns.
- **Where**: Inference models (e.g., BERT family).

### Multi-Head Attention

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
See [Softmax](../modules/activations.md#softmax) -->