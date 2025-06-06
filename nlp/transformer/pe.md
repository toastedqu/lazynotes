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
- **What**: Encode relative positions $\leftarrow$ Rotate each QKV pair.
- **Why**:
	- Absolute PE tie each token to a fixed index $\rightarrow$ NO generalization to longer sequences
	- Learned relative PE learn one weight per distance bucket $\rightarrow$ Limit distance ranges & Add params
	- RoPE: **Param-free, Continuous, Relative, Generalizable**.
- **How**:
	1. Project each token embedding at position index $p$ to $\mathbf{q}_p$ & $\mathbf{k}_p$.
	2. Each query/key vector of hidden dim $d$ = $\frac{d}{2}$ two-component planes formed by even-odd feature pairs.
	3. Assign a rotation angle $\theta_n$, which grows smoothly with two-component pair index $n$, to each token.
	4. Define a fixed 2D rotation operator $R(\theta_n)$, which rotates every plane counterclockwise by the angle $\theta_n$.
	5. Rotate query & key vectors.
	6. Compute attention scores with rotated query & key vectors.
		- The paired rotation ONLY depends on the relative offset between their positions.

```{admonition} Math
:class: note, dropdown
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

6. For each pair, rotate counterclockwise (i.e., Cartesian $\rightarrow$ Polar $\rightarrow$ Cartesian):

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

```{admonition} Q&A
:class: tip, dropdown
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
	- Early dimensions have large angles $\rightarrow$ Rotation changes significantly even between nearby tokens $\rightarrow$ Effectively capture fine-grained relationship
	- Late dimensions have small angles $\rightarrow$ Rotation changes very slowly, almost identical between nearby tokens $\rightarrow$ They ONLY make a significant difference with distant tokens $\rightarrow$ Effectively capture coarse-grained relationship
```