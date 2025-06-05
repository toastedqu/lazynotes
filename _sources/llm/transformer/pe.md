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
# Positional Encoding
- **What**: Semantic vectors + Positional vectors $\rightarrow$ Position-aware vectors
- **Why**:
	- Transformers don't know positions.
	- BUT positions matter!
		- No PE $\rightarrow$ self-attention scores remain unchanged regardless of token orders {cite:p}`wang_positional_encoding`.

## Sinusoidal PE
- **What**: Positional vectors $\rightarrow$ Sine waves
- **Why**:
	- Continuous & multi-scale $\rightarrow$ Generalize to sequences of arbitrary lengths
	- No params $\rightarrow$ Low computational cost
	- Empirically performed as well as learned PE

```{admonition} Math
:class: note, dropdown
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

```{admonition} Q&A
:class: tip, dropdown
*Cons?*
- No params $\rightarrow$ No learning of task-specific position patterns.
- Requires uniform token importance across the sequence. {cite:p}`vaswani2017attention`
- Cannot capture complex, relative, or local positional relationships.
```

## RoPE (Rotary Postion Embedding)
- **What**: Rotation matrix $\times$ Token embeddings $\xrightarrow{\text{encode}}$ Relative Position.
- **Why**:
	- Cons of Absolute PE: Cannot generalize to unseen sequence length.