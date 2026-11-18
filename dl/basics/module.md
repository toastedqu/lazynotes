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
# Module
A module/layer is a function mapping an input tensor $X$ to an output tensor $Y$.
- ML = Function Approximation.
- DL = Function Approximation with a deep NN.
- Deep NN = A bunch of modules stacked together.

Let $g$ denote the gradient $\frac{\partial\mathcal{L}}{\partial y}$ for readability.

This page does NOT cover attention & transformer modules, or convolution & pooling ([CNN Architectures](../../cv/cnn.md)).

&nbsp;

## Linear
- **What**: Linear transform.
- **Why**: Simplest way to transform data & learn patterns.
- **How**: Sum of weighted input features (+ bias).

`````{note} Math
:class: dropdown
````{tab-set}
```{tab} Vector
Notations:
- IO:
    - $\mathbf{x}\in\mathbb{R}^{H_{in}}$: Input vector.
    - $\mathbf{y}\in\mathbb{R}^{H_{out}}$: Output vector.
- Params:
    - $W\in\mathbb{R}^{H_{out}\times H_{in}}$: Weight matrix.
    - $\textbf{b}\in\mathbb{R}^{H_{out}}$: Bias vector.
- Hyperparams:
    - $H_{in}$: Input feature dimension.
    - $H_{out}$: Output feature dimension.

Forward:

$$
\textbf{y}=W\textbf{x}+\textbf{b}
$$

Backward:

$$\begin{align*}
&\frac{\partial\mathcal{L}}{\partial W}=\textbf{g}\textbf{x}^T \\
&\frac{\partial\mathcal{L}}{\partial\textbf{b}}=\textbf{g}\\
&\frac{\partial\mathcal{L}}{\partial\textbf{x}}=W^T\textbf{g}
\end{align*}$$
```
```{tab} Tensor
Notations:
- IO:
    - $\mathbf{X}\in\mathbb{R}^{*\times H_{in}}$: Input tensor.
    - $\mathbf{Y}\in\mathbb{R}^{*\times H_{out}}$: Output tensor.
- Params:
    - $W\in\mathbb{R}^{H_{out}\times H_{in}}$: Weight matrix.
    - $\textbf{b}\in\mathbb{R}^{H_{out}}$: Bias vector.
- Hyperparams:
    - $H_{in}$: Input feature dimension.
    - $H_{out}$: Output feature dimension.

Forward:

$$
\textbf{Y}=\textbf{X}W^T+\textbf{b}
$$

Backward:

$$\begin{align*}
&\frac{\partial\mathcal{L}}{\partial W}=\textbf{g}^T\textbf{X} \\
&\frac{\partial\mathcal{L}}{\partial\textbf{b}}=\sum_*\textbf{g}_*\\
&\frac{\partial\mathcal{L}}{\partial\textbf{x}}=\textbf{g}W
\end{align*}$$
```
````
`````

&nbsp;

## Embedding
- **What**: Trainable lookup table: categorical ID → dense vector.
- **Why**: Categorical IDs (tokens, users, items) have no numeric order or scale → NN needs a learned dense vector per ID.
- **How**:
    1. Keep a matrix $E$ w/ 1 row per ID.
    2. **Forward**: ID $t$ → row $E_t$.
    3. **Backward**: grad reaches only the looked-up rows. Repeated IDs → grads add up.
    - Token-specific use ($\times\sqrt{d}$, shared output layer) → [Token Embedding](../../nlp/transformer.md#token-embedding), [Weight Tying](../../nlp/transformer.md#weight-tying).

```{note} Math
:class: dropdown
Notations:
- IO:
    - $t\in\{0,\dots,V-1\}$: Input ID (0-based, as in code).
    - $\mathbf{y}\in\mathbb{R}^{d}$: Output vector.
- Params:
    - $E\in\mathbb{R}^{V\times d}$: Embedding matrix. Row $E_t$ = vector of ID $t$.
- Hyperparams:
    - $V$: #IDs (vocab size).
    - $d$: Embedding dim.
- Misc:
    - $\mathbf{e}_t\in\{0,1\}^{V}$: One-hot vector of $t$.

Forward:

$$
\mathbf{y}=E^T\mathbf{e}_t=E_t^T
$$

- = [Linear](#linear) w/ $W=E^T$, $\mathbf{b}=\mathbf{0}$, $\mathbf{x}=\mathbf{e}_t$.

Backward:

$$
\frac{\partial\mathcal{L}}{\partial E}=\mathbf{e}_t\mathbf{g}^T
$$

- Only row $t$ is non-zero.
- Batch of IDs $t_1,\dots,t_m$:

    $$
    \frac{\partial\mathcal{L}}{\partial E_r}=\sum_{i:\,t_i=r}\mathbf{g}_i^T
    $$

    - $E_r$: Row $r$ of $E$.
    - ID $r$ absent from the batch → grad $=\mathbf{0}$.
- ❌Grad w.r.t. $t$ ← discrete input.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Embedding(nn.Module):
    def __init__(self, V, d):
        super().__init__()
        self.E = nn.Parameter(torch.randn(V, d))  ## N(0,1), PyTorch's default init

    def forward(self, ids):
        ## row lookup = one_hot(ids) @ E, w/o building the V-dim one-hot or the O(Vd) matmul
        return self.E[ids]

## Example
emb = Embedding(V=10, d=4)
ids = torch.tensor([[1, 3, 3]])                                    ## ID 3 appears twice
print(torch.allclose(emb(ids), F.one_hot(ids, 10).float() @ emb.E))  ## True
emb(ids).sum().backward()
print(emb.E.grad.abs().sum(dim=1))  ## tensor([0., 4., 0., 8., 0., 0., 0., 0., 0., 0.]) -> sparse rows, repeats add up
```
````

```{attention} Q&A
:class: dropdown
*Why not one-hot × Linear?*
- Same function.
- Lookup: $O(d)$ per ID. One-hot matmul: $O(Vd)$ per ID + a $V$-dim input.

*Why does the sparse gradient matter?*
- Only rows in the batch get non-zero grads.
- Huge tables (recsys IDs, large vocabs) → sparse grads + sparse optimizers (e.g., PyTorch `sparse=True` + `SparseAdam`) → memory & compute ⬇️.
- Gotcha: dense Adam/momentum → rows w/ zero grad still move ← stale momentum. Decoupled weight decay shrinks every row each step.

*Rare IDs?*
- Few grad updates → rows stay near init → poor vectors.
- e.g., "glitch tokens": tokens in the vocab but rare in training data → near-untrained rows → odd LLM behavior.

*Unseen IDs at test time?*
- ❌Row exists → reserve an UNK row, or hash IDs into a fixed #buckets (hashing trick; collisions share a row).

*How large is the table?*
- $V\cdot d$ params. GPT-2 small: $50257\times768\approx38.6$M of $\approx124$M params.
- Large share in small models → [Weight Tying](../../nlp/transformer.md#weight-tying) saves it.

*Padding?*
- `padding_idx` row: zero grad → fixed during training. Defaults to a zero vector at construction → pad tokens add nothing.

*Is an embedding just a learned low-rank one-hot code?*
- Yes: $E\in\mathbb{R}^{V\times d}$, $d\ll V$ → each ID gets a point in a shared $d$-dim space.
- Similar IDs end up close only if the loss rewards it (shared contexts, co-occurrence). ❌Built-in similarity.
```

&nbsp;

## GLU
- **Name**: Gated Linear Unit {cite:p}`dauphin2017language`
- **What**: Linear projection ⊙ sigmoid of a 2nd linear projection.
- **Why**: Gating w/ a linear gradient path.
    - Gate → each unit decides, per input, how much of its value passes.
    - Value path has no activation → grad not shrunk by an activation derivative → ⬇️vanishing gradient in deep stacks.
- **How**:
    1. 2 parallel projections of the same input: **value** & **gate**.
    2. Sigmoid on the gate → $(0,1)$.
    3. Multiply elementwise.
    - Replace the sigmoid w/ ReLU/GELU/Swish → ReGLU/GEGLU/SwiGLU. Transformer FFN use → [SwiGLU](../../nlp/models.md#swiglu).

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{x}\in\mathbb{R}^{H_{in}}$: Input vector.
    - $\mathbf{y}\in\mathbb{R}^{H_{out}}$: Output vector.
- Params:
    - $W\in\mathbb{R}^{H_{out}\times H_{in}}$: Value weight.
    - $\mathbf{b}\in\mathbb{R}^{H_{out}}$: Value bias.
    - $V\in\mathbb{R}^{H_{out}\times H_{in}}$: Gate weight.
    - $\mathbf{c}\in\mathbb{R}^{H_{out}}$: Gate bias.
- Intermediate values:
    - $\mathbf{a}=W\mathbf{x}+\mathbf{b}$: Value.
    - $\mathbf{s}=V\mathbf{x}+\mathbf{c}$: Gate pre-activation.

Forward:

$$
\mathbf{y}=\mathbf{a}\odot\sigma(\mathbf{s})
$$

- $\sigma$: Sigmoid.

Backward:

$$\begin{align*}
&\boldsymbol{\delta}_a=\frac{\partial\mathcal{L}}{\partial\mathbf{a}}=\mathbf{g}\odot\sigma(\mathbf{s})\\
&\boldsymbol{\delta}_s=\frac{\partial\mathcal{L}}{\partial\mathbf{s}}=\mathbf{g}\odot\mathbf{a}\odot\sigma(\mathbf{s})\odot(1-\sigma(\mathbf{s}))\\
&\frac{\partial\mathcal{L}}{\partial W}=\boldsymbol{\delta}_a\mathbf{x}^T,\quad\frac{\partial\mathcal{L}}{\partial\mathbf{b}}=\boldsymbol{\delta}_a\\
&\frac{\partial\mathcal{L}}{\partial V}=\boldsymbol{\delta}_s\mathbf{x}^T,\quad\frac{\partial\mathcal{L}}{\partial\mathbf{c}}=\boldsymbol{\delta}_s\\
&\frac{\partial\mathcal{L}}{\partial\mathbf{x}}=W^T\boldsymbol{\delta}_a+V^T\boldsymbol{\delta}_s
\end{align*}$$
```

```{tip} Derivation
:class: dropdown
*Why does the linear path reduce vanishing gradients?*
1. GLU: $\mathbf{y}=\mathbf{a}\odot\sigma(\mathbf{s})$ → $\frac{\partial\mathbf{y}}{\partial\mathbf{a}}=\text{diag}(\sigma(\mathbf{s}))$.
    - Only the gate value scales the grad. ❌Activation derivative.
    - Open gate ($\sigma\approx1$) → grad passes unchanged.
2. GTU (gated tanh unit, LSTM-style gate used before GLU): $\mathbf{y}=\tanh(\mathbf{a})\odot\sigma(\mathbf{s})$ → $\frac{\partial\mathbf{y}}{\partial\mathbf{a}}=\text{diag}(\tanh'(\mathbf{a})\odot\sigma(\mathbf{s}))$.
    - 2 factors $\le1$. $\tanh'\to0$ when $\mathbf{a}$ saturates.
3. Stack $L$ layers → product of $L$ such factors → GTU grads vanish faster.
    - Dauphin et al.: GLU = "multiplicative skip connection".
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GLU(nn.Module):
    def __init__(self, d_in, d_out, act=torch.sigmoid):
        super().__init__()
        self.W = nn.Linear(d_in, d_out)  ## value path
        self.V = nn.Linear(d_in, d_out)  ## gate path
        self.act = act                   ## sigmoid -> GLU; relu / gelu / silu -> ReGLU / GEGLU / SwiGLU

    def forward(self, x):
        ## value stays linear; only the gate goes through act -> grad to W skips any act derivative
        return self.W(x) * self.act(self.V(x))

## Example
x = torch.randn(2, 4)
print(GLU(4, 6)(x).shape)              ## torch.Size([2, 6])
print(GLU(4, 6, act=F.gelu)(x).shape)  ## GEGLU: torch.Size([2, 6])
```
````

```{dropdown} Table: GLU Variants
| Variant | Gate $\phi$ | Layer (biases omitted) | Used in |
|:--|:--|:--|:--|
| GLU | Sigmoid | $W\mathbf{x}\odot\sigma(V\mathbf{x})$ | Gated CNN LM |
| Bilinear | Identity | $W\mathbf{x}\odot V\mathbf{x}$ | — |
| ReGLU | [ReLU](activation.md#relu) | $W\mathbf{x}\odot\max(0,V\mathbf{x})$ | — |
| GEGLU | [GELU](activation.md#gelu) | $W\mathbf{x}\odot\mathrm{GELU}(V\mathbf{x})$ | T5 v1.1, Gemma |
| SwiGLU | Swish$_\beta$ | $W\mathbf{x}\odot\mathrm{Swish}_\beta(V\mathbf{x})$ | PaLM, Llama |

- $\mathrm{Swish}_\beta(z)=z\,\sigma(\beta z)$. $\beta=1$ → [SiLU](activation.md#silu).
- Shazeer {cite:p}`shazeer2020glu`: each variant as a Transformer FFN $W_2(\phi(V\mathbf{x})\odot W\mathbf{x})$ → GEGLU & SwiGLU best. Details → [SwiGLU](../../nlp/models.md#swiglu).
```

```{attention} Q&A
:class: dropdown
*Pros?*
- Linear value path → ⬇️vanishing gradient in deep stacks.
- Multiplicative → each unit = product of 2 linear features of $\mathbf{x}$ → 2nd-order interactions.
- Input-dependent feature selection, per unit.

*Cons?*
- 2 projections → 2× params & FLOPs at the same $H_{out}$.
    - FFN fix: shrink the hidden dim to $\frac{2}{3}$ → [SwiGLU](../../nlp/models.md#swiglu).
- 1 extra activation tensor in memory.

*GLU vs GTU vs Bilinear vs Linear?*
- Dauphin et al.: GLU > GTU → faster convergence, better perplexity.
- Dauphin et al., Google Billion Word: GLU > Bilinear > Linear → the sigmoid's non-linearity helps, beyond the product alone.

*Is GLU an activation function?*
- The full GLU has params ($W$, $V$) → a layer.
- PyTorch `F.glu` / `nn.GLU` = param-free part only: split the input in half along a dim → $\mathbf{a}\odot\sigma(\mathbf{s})$.
- → Fused implementation: 1 matmul to $2H_{out}$ → split → gate.

*Sigmoid gate vs GELU/Swish gate?*
- Sigmoid: $(0,1)$ → a true gate.
- GELU/Swish: unbounded & can be negative → "gate" = 2nd linear feature, softly rectified. Closer to Bilinear.

*How does it relate to LSTM gates?*
- Same idea: sigmoid gate × content.
- LSTM gates act over time on a memory. GLU gates features within 1 feedforward layer.
- LSTM's $\mathbf{o}_t\odot\tanh(\mathbf{c}_t)$ = GTU-like → saturating path. GLU drops the tanh.
```

&nbsp;

## Dropout
- **What**: Randomly ignore some neurons during training.
- **Why**: To reduce overfitting.
- **How**: During training:
    1. Randomly set a fraction of neurons to 0.
    2. Scale the outputs/gradients on active neurons by the keep probability.

`````{note} Math
:class: dropdown
````{tab-set}
```{tab} Vector
Notations:
- IO:
    - $\mathbf{x}\in\mathbb{R}^{H_{in}}$: Input vector.
    - $\mathbf{y}\in\mathbb{R}^{H_{in}}$: Output vector.
- Hyperparams:
    - $p$: Keep probability.
- Intermediate values:
    - $\textbf{m}\in\mathbb{R}^{H_{in}}$: binary mask, where each element $m\sim\text{Bernoulli}(p)$.

Forward:

$$
\textbf{y}=\frac{\textbf{m}\odot\textbf{x}}{p}
$$

Backward:

$$
\frac{\partial\mathcal{L}}{\partial\textbf{x}} = \frac{\textbf{m}\odot\textbf{g}}{p}
$$
```
```{tab} Tensor
Notations:
- IO:
    - $\mathbf{X}\in\mathbb{R}^{*\times H_{in}}$: Input tensor.
    - $\mathbf{Y}\in\mathbb{R}^{*\times H_{in}}$: Output tensor.
- Hyperparams:
    - $p$: Keep probability.
- Intermediate values:
    - $\textbf{M}\in\mathbb{R}^{*\times H_{in}}$: binary mask, where each element $m\sim\text{Bernoulli}(p)$.

Forward:

$$
\textbf{Y}=\frac{\textbf{M}\odot\textbf{X}}{p}
$$

Backward:

$$
\frac{\partial\mathcal{L}}{\partial\textbf{X}} = \frac{\textbf{M}\odot\textbf{g}}{p}
$$
```
````
`````

```{attention} Q&A
:class: dropdown
*Cons?*
- ⬆️Training time ← ⬇️Convergence speed ← Sparsity
- ✅Hyperparameter Tuning.
```

&nbsp;

## Residual Connection
- **What**: Model the residual ($Y-X$) instead of the output ($Y$).
- **Why**: To mitigate [vanishing/exploding gradients](../train/issues.md).
- **How**: Add input $X$ to block output $F(X)$.
    - If the feature dimension of $X$ and $F(X)$ doesn't match, use a shortcut linear layer on $X$ to change its feature dimension.

`````{note} Math
:class: dropdown
````{tab-set}
```{tab} Vector
Notation:
- IO:
    - $\mathbf{x}\in\mathbb{R}^{H_{in}}$: Input vector.
    - $\mathbf{y}\in\mathbb{R}^{H_{out}}$: Output vector.
- Hyperparams:
    - $F(\cdot)\in\mathbb{R}^{H_{out}}$: The aggregate function of all layers within the residual block.

Forward:

$$
\textbf{y}=F(\textbf{x})+\textbf{x}
$$

Backward:

$$
\frac{\partial\mathcal{L}}{\partial\textbf{x}}=\mathbf{g}(1+\frac{\partial F(\textbf{x})}{\partial\textbf{x}})
$$
```
```{tab} Tensor
Notation:
- IO:
    - $\mathbf{x}\in\mathbb{R}^{*\times H_{in}}$: Input tensor.
    - $\mathbf{y}\in\mathbb{R}^{*\times H_{out}}$: Output tensor.
- Hyperparams:
    - $F(\cdot)\in\mathbb{R}^{H_{out}}$: The aggregate function of all layers within the residual block.

Forward:

$$
\textbf{Y}=F(\textbf{X})+\textbf{X}
$$

Backward:

$$
\frac{\partial\mathcal{L}}{\partial\textbf{X}}=\mathbf{g}(1+\frac{\partial F(\textbf{X})}{\partial\textbf{X}})
$$
```
````
`````

&nbsp;

## Normalization
### BatchNorm
- **What**: Normalize each feature across input samples to zero mean & unit variance.
- **Why**: To mitigate [internal covariate shift](../train/issues.md#vanishing/internal-covariate-shift).
- **How**:
    1. Calculate the mean and variance for each batch.
    2. Normalize the batch.
    3. Scale and shift the normalized output using learnable params.


```{note} Math
:class: dropdown
Notation:
- IO:
    - $\mathbf{X}\in\mathbb{R}^{m\times n}$: Input matrix.
    - $\mathbf{Y}\in\mathbb{R}^{m\times n}$: Output matrix.
- Params:
    - $\gamma\in\mathbb{R}$: Scale param.
    - $\beta\in\mathbb{R}$: Shift param.

Forward:
1. Calculate the mean and variance for each batch.

    $$\begin{align*}
    \boldsymbol{\mu}_B&=\frac{1}{m}\sum_{i=1}^{m}\textbf{x}_i\\
    \boldsymbol{\sigma}_B^2&=\frac{1}{m}\sum_{i=1}^{m}(\textbf{x}_i-\boldsymbol{\mu}_B)^2
    \end{align*}$$

2. Normalize each batch.

    $$
    \textbf{z}_i=\frac{\textbf{x}_i-\boldsymbol{\mu}_B}{\sqrt{\boldsymbol{\sigma}_B^2+\epsilon}}
    $$

    where $\epsilon$ is a small constant to avoid dividing by 0.

3. Scale and shift the normalized output.

    $$
    \textbf{y}_i=\gamma\textbf{z}_i+\beta
    $$

Backward:
1. Gradient w.r.t. params:

    $$\begin{align*}
    &\frac{\partial\mathcal{L}}{\partial\gamma}=\sum_{i=1}^{m}\textbf{g}_i\textbf{z}_i\\
    &\frac{\partial\mathcal{L}}{\partial\beta}=\sum_{i=1}^{m}\textbf{g}_i
    \end{align*}$$
2. Gradient w.r.t. input:

    $$\begin{align*}
    &\frac{\partial\mathcal{L}}{\partial\textbf{z}_i}=\gamma\textbf{g}_i\\
    &\frac{\partial\mathcal{L}}{\partial\boldsymbol{\sigma}_B^2}=\sum_{i=1}^{m}\frac{\partial\mathcal{L}}{\partial\textbf{z}_i}(\textbf{x}_i-\boldsymbol{\mu}_B)\left(-\frac{1}{2}(\boldsymbol{\sigma}_B^2+\epsilon)^{-\frac{3}{2}}\right)\\
    &\frac{\partial\mathcal{L}}{\partial\boldsymbol{\mu}_B}=\sum_{i=1}^{m}\frac{\partial\mathcal{L}}{\partial\textbf{z}_i}\cdot\left(-\frac{1}{\sqrt{\boldsymbol{\sigma}_B^2+\epsilon}}\right)+\frac{\partial\mathcal{L}}{\partial\boldsymbol{\sigma}_B^2}\cdot\left(-\frac{2}{m}\sum_{i=1}^{m}(\textbf{x}_i-\boldsymbol{\mu}_B)\right)\\
    &\frac{\partial\mathcal{L}}{\partial\textbf{x}_i}=\frac{1}{\sqrt{\boldsymbol{\sigma}_B^2+\epsilon}}\left(\frac{\partial\mathcal{L}}{\partial\textbf{z}_i}+\frac{2}{m}\frac{\partial\ma                                                                         thcal{L}}{\partial\boldsymbol{\sigma}_B^2}(\textbf{x}_i-\boldsymbol{\mu}_B)+\frac{1}{m}\frac{\partial\mathcal{L}}{\partial\boldsymbol{\mu}_B}\right)
    \end{align*}$$
```

```{attention} Q&A
:class: dropdown
*Pros?*
- Accelerates training with higher learning rates.
- Reduces sensitivity to weight initialization.
- Mitigates [vanishing/exploding gradients](../train/issues.md).

*Cons?*
- Adds computation overhead and complexity.
- Works best when each mini-batch is representative of the overall input distribution to accurately estimate the mean and variance.
- Causes potential issues in certain cases like small mini-batches or when batch statistics differ from overall dataset statistics.
```

&nbsp;

### LayerNorm
- **What**: Normalize each sample across input features to zero mean and unit variance.
- **Why**: BatchNorm depends on the batch size.
    - When it's too big, high computational cost.
    - When it's too small, the batch may not be representative of the underlying data distribution.
    - Hyperparam tuning is required to find the optimal batch size, leading to high computational cost.
- **How**:
    1. Calculate the mean and variance for each feature.
    2. Normalize the feature.
    3. Scale and shift the normalized output using learnable params.

```{note} Math
:class: dropdown
It's easy to explain with the vector form for BatchNorm, but it's more intuitive to explain with the scalar form for LayerNorm.

Notations:
- IO:
    - $x_{ij}\in\mathbb{R}$: $j$th feature value for $i$th input sample.
    - $y_{ij}\in\mathbb{R}$: $j$th feature value for $i$th output sample.
- Params:
    - $\boldsymbol{\gamma}\in\mathbb{R}^n$: Scale param.
    - $\boldsymbol{\beta}\in\mathbb{R}^n$: Shift param.

Forward:
1. Calculate the mean and variance for each feature.

    $$\begin{align*}
    \mu_i&=\frac{1}{n}\sum_{j=1}^{n}x_{ij}\\
    \sigma_i^2&=\frac{1}{n}\sum_{j=1}^{n}(x_{ij}-\mu_i)^2
    \end{align*}$$

2. Normalize each feature.

    $$
    z_{ij}=\frac{x_{ij}-\mu_i}{\sqrt{\sigma_i^2+\epsilon}}
    $$

    where $\epsilon$ is a small constant to avoid dividing by 0.

3. Scale and shift the normalized output.

    $$
    y_{ij}=\gamma_jz_{ij}+\beta_j
    $$

Backward:
1. Gradient w.r.t. params:

    $$\begin{align*}
    &\frac{\partial\mathcal{L}}{\partial\gamma_j}=\sum_{i=1}^{m}g_{ij}z_{ij}\\
    &\frac{\partial\mathcal{L}}{\partial\beta_j}=\sum_{i=1}^{m}g_{ij}
    \end{align*}$$
2. Gradient w.r.t. input:

    $$\begin{align*}
    &\frac{\partial\mathcal{L}}{\partial z_{ij}}=\gamma_jg_{ij}\\
    &\frac{\partial\mathcal{L}}{\partial\sigma_i^2}=\sum_{j=1}^{n}\frac{\partial\mathcal{L}}{\partial z_{ij}}(x_{ij}-\mu_i)\left(-\frac{1}{2}(\sigma_i^2+\epsilon)^{-\frac{3}{2}}\right)\\
    &\frac{\partial\mathcal{L}}{\partial\mu_i}=\sum_{j=1}^{n}\frac{\partial\mathcal{L}}{\partial z_{ij}}\cdot\left(-\frac{1}{\sqrt{\sigma_i^2+\epsilon}}\right)+\frac{\partial\mathcal{L}}{\partial\sigma_i^2}\cdot\left(-\frac{2}{n}\sum_{j=1}^{n}(x_{ij}-\mu_i)\right)\\
    &\frac{\partial\mathcal{L}}{x_{ij}}=\frac{1}{\sqrt{\sigma_i^2+\epsilon}}\left(\frac{\partial\mathcal{L}}{\partial z_{ij}}+\frac{2}{n}\frac{\partial\mathcal{L}}{\partial\sigma_i^2}(x_{ij}-\mu_i)+\frac{1}{n}\frac{\partial\mathcal{L}}{\partial\mu_i}\right)
    \end{align*}$$
```

```{attention} Q&A
:class: dropdown
*Pros?*
- Reduces hyperparam tuning effort.
- High consistency during training and inference.
- Mitigates [vanishing/exploding gradients](../train/issues.md).

*Cons?*
- Adds computation overhead and complexity.
- Inapplicable in CNNs due to varied statistics of spatial features.
```

&nbsp;

### RMSNorm
Normalize each sample across input features to zero mean and unit variance.
- **What**: Normalize each sample's input features using its RMS, then apply a learnable per-feature gain.
- **Why**: Cheaper & More stable than LayerNorm ← Skip mean-centering.
- **How**:
    1. For each token vector $\mathbf{x}$, compute RMS over hidden dimension.
    2. Divide $\mathbf{x}$ by the RMS (plus $\epsilon$).
    3. Apply learnable gain $\boldsymbol{\gamma}$ elementwise.

```{note} Math
:class: dropdown
Notations:
- IO:
  - $\mathbf{x}\in\mathbb{R}^{d}$: input hidden vector for one token (one position).
  - $\mathbf{y}\in\mathbb{R}^{d}$: output vector.
- Params:
  - $\boldsymbol{\gamma}\in\mathbb{R}^{d}$: learnable gain (per feature).
- Hyperparams / misc:
  - $d$: hidden size.
  - $\epsilon>0$: numerical stability constant.

Forward:
1. RMS over features:
$$
\operatorname{rms}(\mathbf{x})=\sqrt{\frac{1}{d}\sum_{j=1}^{d}x_j^2}
$$

2. Normalize + scale:
$$
\mathbf{y}=\boldsymbol{\gamma}\odot \frac{\mathbf{x}}{\operatorname{rms}(\mathbf{x})+\epsilon}
$$

Backward: Let $s=\operatorname{rms}+\epsilon$
- Gradient w.r.t. gain:
$$
\frac{\partial\mathcal{L}}{\partial \boldsymbol{\gamma}}=\mathbf{g}\odot\frac{\mathbf{x}}{s}
$$

- Gradient w.r.t. input:
    1. Derivative of RMS scalar:
    $$
    \frac{\partial \operatorname{rms}}{\partial \mathbf{x}}=\frac{\mathbf{x}}{d\operatorname{rms}}
    $$

    2. Final grad:
    $$
    \frac{\partial\mathcal{L}}{\partial \mathbf{x}}=\frac{\mathbf{g}\odot\boldsymbol{\gamma}}{s}-\frac{\mathbf{x}}{d\,\operatorname{rms}\,s^{2}}\left((\mathbf{g}\odot\boldsymbol{\gamma})^{T}\mathbf{x}\right)
    $$
```

```{attention} Q&A
:class: dropdown
*Why does skipping mean-centering work?*
- In many Transformer setups, the key stability issue is controlling activation scale. 
- Residual connections + linear layers can absorb offsets, while RMS-based scaling keeps magnitudes from drifting.
```

&nbsp;

### InstanceNorm
- **What**: Normalize each channel of each sample over its spatial positions. {cite:p}`ulyanov2016instance`
- **Why**: Style transfer: output contrast should not depend on the content image's contrast.
    - Per-sample, per-channel mean & var ≈ contrast/style info → remove it.
    - Ulyanov et al.: swap BN → IN in the stylization generator & keep it at test time → large quality gain.
- **How**:
    1. Per (sample, channel): mean & var over all spatial positions.
    2. Normalize.
    3. Optional per-channel scale & shift.
    4. Same computation at train & test ← ❌batch statistics, ❌running averages.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{X}\in\mathbb{R}^{m\times C\times P}$: Input feature maps, spatial dims flattened.
    - $\mathbf{Y}\in\mathbb{R}^{m\times C\times P}$: Output feature maps.
    - $x_{icp}$: Value at sample $i$, channel $c$, spatial position $p$.
- Params:
    - $\boldsymbol{\gamma}\in\mathbb{R}^{C}$: Per-channel scale.
    - $\boldsymbol{\beta}\in\mathbb{R}^{C}$: Per-channel shift.
- Hyperparams:
    - $C$: #channels.
    - $P$: #spatial positions (height × width).
    - $\epsilon$: Small constant to avoid dividing by 0.

Forward:

$$\begin{align*}
\mu_{ic}&=\frac{1}{P}\sum_{p=1}^{P}x_{icp}\\
\sigma_{ic}^2&=\frac{1}{P}\sum_{p=1}^{P}(x_{icp}-\mu_{ic})^2\\
z_{icp}&=\frac{x_{icp}-\mu_{ic}}{\sqrt{\sigma_{ic}^2+\epsilon}}\\
y_{icp}&=\gamma_cz_{icp}+\beta_c
\end{align*}$$

Backward:

$$\begin{align*}
&\frac{\partial\mathcal{L}}{\partial\gamma_c}=\sum_{i=1}^{m}\sum_{p=1}^{P}g_{icp}z_{icp},\quad\frac{\partial\mathcal{L}}{\partial\beta_c}=\sum_{i=1}^{m}\sum_{p=1}^{P}g_{icp}\\
&\frac{\partial\mathcal{L}}{\partial x_{icp}}=\frac{1}{\sqrt{\sigma_{ic}^2+\epsilon}}\left(\hat{g}_{icp}-\frac{1}{P}\sum_{p'=1}^{P}\hat{g}_{icp'}-z_{icp}\frac{1}{P}\sum_{p'=1}^{P}\hat{g}_{icp'}z_{icp'}\right)
\end{align*}$$

- $\hat{g}_{icp}=\gamma_cg_{icp}$: Grad w.r.t. $z_{icp}$.
- Same closed form as BatchNorm & LayerNorm backward; only the set being averaged changes.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class InstanceNorm(nn.Module):
    def __init__(self, C, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(C))
        self.beta = nn.Parameter(torch.zeros(C))
        self.eps = eps

    def forward(self, x):  ## x: (m, C, H, W)
        ## stats per (sample, channel) over H, W only -> each image's own contrast is removed
        mu = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), unbiased=False, keepdim=True)
        z = (x - mu) / torch.sqrt(var + self.eps)
        return self.gamma.view(1, -1, 1, 1) * z + self.beta.view(1, -1, 1, 1)

## Example
x = torch.randn(2, 3, 8, 8) * 5 + 2
y = InstanceNorm(3)(x)
print(y.shape)                         ## torch.Size([2, 3, 8, 8])
print(y.mean(dim=(2, 3)).abs().max())  ## ~0: every (sample, channel) map is centered
```
````

```{attention} Q&A
:class: dropdown
*IN vs BN at test time?*
- IN: uses the test image's own stats. Train = test.
- BN: uses running averages from training.

*Why does IN suit style transfer & image generation?*
- Feature-map mean & std per channel ≈ style (contrast, color, texture strength).
- IN removes them → content stays → a style can be injected.
- AdaIN: replace $\gamma_c,\beta_c$ w/ the style image's per-channel std & mean → arbitrary style transfer. Same idea drives [StyleGAN](../gen/gan.md#stylegan)'s style injection.

*Cons?*
- Discards per-channel global intensity → hurts recognition when that intensity is informative.
- e.g., GroupNorm paper (ResNet-50, ImageNet): IN error > BN, GN, LN.

*Degenerate case?*
- $P=1$ (1×1 feature map) → $x=\mu$ → $z=0$ → $y=\beta_c$ → all input info lost.

*How does it relate to GroupNorm & LayerNorm?*
- IN = [GroupNorm](#groupnorm) w/ $G=C$ (1 channel per group).
- LN = GroupNorm w/ $G=1$.

*PyTorch default?*
- `InstanceNorm2d`: `affine=False`, `track_running_stats=False` → no $\gamma,\beta$, no running stats.
```

&nbsp;

### GroupNorm
- **What**: Normalize each sample over groups of channels & all spatial positions. {cite:p}`wu2018group`
- **Why**: BN error ⬆️ fast as batch size ⬇️ ← noisy batch statistics.
    - Detection, segmentation, video, high-res generation → memory caps the batch at 1–2 per GPU.
    - GN: ❌batch dim → accuracy stable across batch sizes.
- **How**:
    1. Split $C$ channels into $G$ groups of $C/G$ (paper default $G=32$).
    2. Per (sample, group): mean & var over the group's channels × spatial positions.
    3. Normalize. Per-channel scale & shift.
    - $G=1$ → [LayerNorm](#layernorm). $G=C$ → [InstanceNorm](#instancenorm).

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{X}\in\mathbb{R}^{m\times C\times P}$: Input feature maps, spatial dims flattened.
    - $\mathbf{Y}\in\mathbb{R}^{m\times C\times P}$: Output feature maps.
    - $x_{icp}$: Value at sample $i$, channel $c$, spatial position $p$.
- Params:
    - $\boldsymbol{\gamma}\in\mathbb{R}^{C}$: Per-channel scale.
    - $\boldsymbol{\beta}\in\mathbb{R}^{C}$: Per-channel shift.
- Hyperparams:
    - $G$: #groups. $C$ divisible by $G$.
    - $C$: #channels.
    - $P$: #spatial positions (height × width).
    - $\epsilon$: Small constant to avoid dividing by 0.
- Misc:
    - $\mathcal{G}$: Set of channels in 1 group, $|\mathcal{G}|=C/G$.
    - $\mathcal{G}(c)$: Group that contains channel $c$.
    - $M=PC/G$: #values per (sample, group).

Forward:

$$\begin{align*}
\mu_{i\mathcal{G}}&=\frac{1}{M}\sum_{c\in\mathcal{G}}\sum_{p=1}^{P}x_{icp}\\
\sigma_{i\mathcal{G}}^2&=\frac{1}{M}\sum_{c\in\mathcal{G}}\sum_{p=1}^{P}(x_{icp}-\mu_{i\mathcal{G}})^2\\
z_{icp}&=\frac{x_{icp}-\mu_{i\mathcal{G}(c)}}{\sqrt{\sigma_{i\mathcal{G}(c)}^2+\epsilon}}\\
y_{icp}&=\gamma_cz_{icp}+\beta_c
\end{align*}$$

Backward:
- $\frac{\partial\mathcal{L}}{\partial\gamma_c}$, $\frac{\partial\mathcal{L}}{\partial\beta_c}$: same as [InstanceNorm](#instancenorm).
- Input, w/ $\hat{g}_{icp}=\gamma_cg_{icp}$ & $\mathcal{G}=\mathcal{G}(c)$:

    $$
    \frac{\partial\mathcal{L}}{\partial x_{icp}}=\frac{1}{\sqrt{\sigma_{i\mathcal{G}}^2+\epsilon}}\left(\hat{g}_{icp}-\frac{1}{M}\sum_{c'\in\mathcal{G}}\sum_{p'=1}^{P}\hat{g}_{ic'p'}-z_{icp}\frac{1}{M}\sum_{c'\in\mathcal{G}}\sum_{p'=1}^{P}\hat{g}_{ic'p'}z_{ic'p'}\right)
    $$
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupNorm(nn.Module):
    def __init__(self, G, C, eps=1e-5):
        super().__init__()
        assert C % G == 0
        self.G, self.eps = G, eps
        self.gamma = nn.Parameter(torch.ones(C))  ## per-channel affine, as in BN
        self.beta = nn.Parameter(torch.zeros(C))

    def forward(self, x):  ## x: (m, C, H, W)
        m, C, H, W = x.shape
        ## channels of 1 group are contiguous -> (m, G, C/G * H * W)
        xg = x.reshape(m, self.G, -1)
        ## stats per (sample, group): no batch dim -> batch size 1 works
        mu = xg.mean(dim=-1, keepdim=True)
        var = xg.var(dim=-1, unbiased=False, keepdim=True)
        z = ((xg - mu) / torch.sqrt(var + self.eps)).reshape(m, C, H, W)
        return self.gamma.view(1, C, 1, 1) * z + self.beta.view(1, C, 1, 1)

## Example
x = torch.randn(2, 6, 4, 4)
gn = GroupNorm(G=3, C=6)
print(gn(x).shape)                                           ## torch.Size([2, 6, 4, 4])
print(torch.allclose(gn(x), F.group_norm(x, 3), atol=1e-5))  ## True
## G=1 -> LayerNorm over (C, H, W) w/ per-channel affine; G=C -> InstanceNorm
```
````

```{dropdown} Table: Normalization Axes
| Method | Stats over (image $m\times C\times H\times W$) | Stats over (tokens $m\times L\times d$) | Batch-dependent | Mean-centered | Typical use |
|:--|:--|:--|:--|:--|:--|
| [BatchNorm](#batchnorm) | $m,H,W$ per channel | $m,L$ per feature | ✅ → running stats at test | ✅ | CNNs w/ large batches |
| [LayerNorm](#layernorm) | $C,H,W$ per sample | $d$ per token | ❌ | ✅ | Transformers, RNNs |
| [RMSNorm](#rmsnorm) | rare | $d$ per token | ❌ | ❌ | Modern LLMs |
| GroupNorm | $C/G,H,W$ per (sample, group) | rare | ❌ | ✅ | Detection, segmentation, diffusion U-Nets |
| [InstanceNorm](#instancenorm) | $H,W$ per (sample, channel) | rare | ❌ | ✅ | Style transfer, image generation |

- $H$: Height.
- $W$: Width.
- $L$: Sequence length.
- $d$: Hidden size.
- After normalizing: learnable per-channel/feature affine (RMSNorm: gain only. InstanceNorm: optional, PyTorch default `affine=False`).
```

```{attention} Q&A
:class: dropdown
*Why groups of channels?*
- Channels are not independent: filters come in related families (orientation, frequency, texture).
- Classical features (SIFT, HOG) already normalize per group of histogram channels.
- → Normalize related channels together. Each group keeps its own scale.

*Pros?*
- Batch-independent → batch 1–2 OK. Train = test. ❌Running stats.
- Paper (ResNet-50, ImageNet): batch 2 → GN error 10.6 pp lower than BN (24.1% vs 34.7%).
- Pretrain → fine-tune w/ a different batch size: ❌stat mismatch.

*Cons?*
- Normal batch sizes → slightly worse than BN (paper: 0.5 pp) ← loses BN's batch-noise regularization.
- Extra hyperparam $G$.
- ❌Fold into the previous conv at inference. BN can ← fixed running stats = fixed affine map.

*Why not LN or IN for CNNs?*
- LN ($G=1$): 1 mean/var for all channels → forces every channel to share stats.
- IN ($G=C$): each channel normalized alone → ignores channel dependence.
- Paper: $G=32$ beats both extremes.

*Where is it used?*
- Detection & segmentation (small per-GPU batch).
- Diffusion [U-Net](../gen/diffusion.md#u-net)s. AdaGN = GN + timestep/class-dependent scale & shift.
```

&nbsp;

### Weight Normalization
- **What**: Reparameterize each weight vector as length × unit direction. {cite:p}`salimans2016weight`
- **Why**: BN speeds up training, but its batch statistics add noise & batch dependence → poor fit for RNNs, RL, noise-sensitive generative models.
    - WN: much of BN's speedup, w/o batch statistics.
- **How**:
    1. Per output neuron: train a direction $\mathbf{v}$ & a length $\alpha$ instead of $\mathbf{w}$.
    2. $\mathbf{w}=\alpha\,\mathbf{v}/\|\mathbf{v}\|$ → length & direction learn separately → better-conditioned grads.
    3. **Init** (data-dependent): 1 minibatch → set $\alpha$ & $b$ so each pre-activation starts at mean 0, var 1.
        - BN-like start, then no more data-dependent stats.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{x}\in\mathbb{R}^{H_{in}}$: Input vector.
    - $y\in\mathbb{R}$: Pre-activation of 1 output neuron.
- Params:
    - $\mathbf{v}\in\mathbb{R}^{H_{in}}$: Direction param.
    - $\alpha\in\mathbb{R}$: Length param. Paper's $g$, renamed ← $g$ = output grad on this page.
    - $b\in\mathbb{R}$: Bias.
- Misc:
    - $\mathbf{w}\in\mathbb{R}^{H_{in}}$: Effective weight vector.
    - $\nabla_{\mathbf{w}}\mathcal{L}$: Grad w.r.t. $\mathbf{w}$ as if $\mathbf{w}$ were the param. $=g\,\mathbf{x}$ for 1 neuron.

Forward:

$$
\mathbf{w}=\frac{\alpha}{\|\mathbf{v}\|}\mathbf{v},\qquad y=\mathbf{w}^T\mathbf{x}+b
$$

- $\|\mathbf{w}\|=|\alpha|$, independent of $\|\mathbf{v}\|$.
- Layer: 1 $(\alpha,\mathbf{v})$ pair per row of $W$.

Backward:

$$\begin{align*}
&\frac{\partial\mathcal{L}}{\partial\alpha}=\frac{\mathbf{v}^T\nabla_{\mathbf{w}}\mathcal{L}}{\|\mathbf{v}\|}\\
&\nabla_{\mathbf{v}}\mathcal{L}=\frac{\alpha}{\|\mathbf{v}\|}M_{\mathbf{w}}\nabla_{\mathbf{w}}\mathcal{L}
\end{align*}$$

- $M_{\mathbf{w}}=I-\frac{\mathbf{w}\mathbf{w}^T}{\|\mathbf{w}\|^2}$: Projection onto the complement of $\mathbf{w}$.
- 2 effects: grad scaled by $\alpha/\|\mathbf{v}\|$, grad component along $\mathbf{w}$ removed.
```

```{tip} Derivation
:class: dropdown
*Why does $\|\mathbf{v}\|$ grow & self-tune the LR?*
1. $M_{\mathbf{w}}$ removes the component along $\mathbf{w}\parallel\mathbf{v}$ → $\nabla_{\mathbf{v}}\mathcal{L}\perp\mathbf{v}$.
2. Plain GD step $\mathbf{v}'=\mathbf{v}-\eta\nabla_{\mathbf{v}}\mathcal{L}$ → Pythagoras:

    $$
    \|\mathbf{v}'\|^2=\|\mathbf{v}\|^2+\eta^2\|\nabla_{\mathbf{v}}\mathcal{L}\|^2\ge\|\mathbf{v}\|^2
    $$

3. Direction change per step $\approx\eta\|\nabla_{\mathbf{v}}\mathcal{L}\|/\|\mathbf{v}\|$. And $\|\nabla_{\mathbf{v}}\mathcal{L}\|\propto1/\|\mathbf{v}\|$ → effective LR on the direction $\propto\eta/\|\mathbf{v}\|^2$.
4. $\eta$ too large → big steps → $\|\mathbf{v}\|$ grows fast → effective LR ⬇️ → self-stabilizing.
- ⚠️ Exact only for GD w/o momentum (paper's assumption). Momentum/Adam → approximate.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class WeightNormLinear(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.v = nn.Parameter(torch.randn(d_out, d_in) * 0.05)  ## direction, 1 row per output neuron
        self.alpha = nn.Parameter(torch.ones(d_out))             ## length (paper's g)
        self.b = nn.Parameter(torch.zeros(d_out))

    def forward(self, x):
        ## w = alpha * v / ||v|| row-wise -> ||w_row|| = |alpha|, whatever the scale of v
        w = self.alpha[:, None] * self.v / self.v.norm(dim=1, keepdim=True)
        return x @ w.T + self.b

    @torch.no_grad()
    def data_init(self, x):
        ## data-dependent init: first-batch pre-activations -> mean 0, std 1 per neuron
        t = x @ (self.v / self.v.norm(dim=1, keepdim=True)).T
        mu, std = t.mean(dim=0), t.std(dim=0)
        self.alpha.copy_(1 / std)
        self.b.copy_(-mu / std)

## Example
x = torch.randn(64, 4) * 3 + 1
layer = WeightNormLinear(4, 3)
layer.data_init(x)
y = layer(x)
print(y.mean(dim=0), y.std(dim=0))  ## ~[0, 0, 0], ~[1, 1, 1]
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- ❌Batch dependence → works for RNNs, RL (DQN), generative models, batch size 1.
- Cheap ← normalizes weights, ❌activations. #weights ≪ #activations in CNNs.
- Deterministic. Train = test.

*Cons?*
- ❌Controls activation scale during training → needs data-dependent init.
    - Data-dependent init ❌ for RNNs (paper) → standard init there.
- Deep CNNs: usually weaker than BN. Transformers: LN/RMSNorm dominate.

*WN vs BN?*
- 1 layer, whitened input (independent features, mean 0, var 1) → pre-activation $t=\mathbf{v}^T\mathbf{x}$ has mean 0, std $\|\mathbf{v}\|$ → BN's normalization = dividing by $\|\mathbf{v}\|$ = WN.
- Deeper layers / correlated inputs → they differ: BN uses actual batch stats.
- Paper also proposes mean-only BN + WN: subtract batch mean, ❌divide by batch std.

*Why does it speed up training?*
- Grad scaled by $\alpha/\|\mathbf{v}\|$ & projected away from $\mathbf{w}$ → grad covariance closer to identity → better conditioning.

*Interaction w/ weight decay?*
- Output invariant to $\|\mathbf{v}\|$ → weight decay on $\mathbf{v}$ only shrinks $\|\mathbf{v}\|$ → effective LR on the direction ⬆️. Same effect as weight decay before BN.

*WN vs Spectral Normalization?*
- WN: fixes each row's norm. Learned $\alpha$ → scale stays free.
- SN: fixes the whole matrix's largest singular value to 1 → Lipschitz bound.
```

&nbsp;

### Spectral Normalization
- **What**: Divide each weight matrix by its largest singular value. {cite:p}`miyato2018spectral`
- **Why**: GAN training unstable ← discriminator too sharp.
    - Sharp $D$ → unbounded/erratic grads to $G$, or vanishing grads once $D$ separates real & fake perfectly.
    - Every layer's Lipschitz constant $\approx1$ → whole $D$ $\approx1$-Lipschitz bound (w/ 1-Lipschitz activations, e.g., ReLU, LReLU).
- **How**:
    1. Estimate the largest singular value w/ power iteration: 1 step per training step. Reuse the singular-vector estimate from the prev step.
    2. Forward w/ $W/\sigma(W)$.
    3. Backprop through $\sigma(W)$. Treat the singular-vector estimates as constants.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{x}\in\mathbb{R}^{H_{in}}$: Input vector.
    - $\mathbf{y}\in\mathbb{R}^{H_{out}}$: Output vector.
- Params:
    - $W\in\mathbb{R}^{H_{out}\times H_{in}}$: Unnormalized weight.
    - $\mathbf{b}\in\mathbb{R}^{H_{out}}$: Bias.
- Misc:
    - $\sigma(W)=\max_{\mathbf{h}\neq\mathbf{0}}\frac{\|W\mathbf{h}\|_2}{\|\mathbf{h}\|_2}$: Spectral norm = largest singular value of $W$.
    - $\mathbf{u}\in\mathbb{R}^{H_{out}}$: Estimate of the top left singular vector. Persists across steps.
    - $\mathbf{v}\in\mathbb{R}^{H_{in}}$: Estimate of the top right singular vector.

Forward:
1. 1 power-iteration step:

    $$
    \mathbf{v}\leftarrow\frac{W^T\mathbf{u}}{\|W^T\mathbf{u}\|_2},\qquad\mathbf{u}\leftarrow\frac{W\mathbf{v}}{\|W\mathbf{v}\|_2}
    $$

2. Estimate & normalize:

    $$
    \hat{\sigma}=\mathbf{u}^TW\mathbf{v},\qquad\bar{W}=\frac{W}{\hat{\sigma}},\qquad\mathbf{y}=\bar{W}\mathbf{x}+\mathbf{b}
    $$

Backward ($\mathbf{u},\mathbf{v}$ constant):

$$
\frac{\partial\mathcal{L}}{\partial W}=\frac{1}{\hat{\sigma}}\left(G-\langle G,\bar{W}\rangle_F\,\mathbf{u}\mathbf{v}^T\right)
$$

- $G=\frac{\partial\mathcal{L}}{\partial\bar{W}}=\mathbf{g}\mathbf{x}^T$: [Linear](#linear) weight grad.
- $\langle A,B\rangle_F=\sum_{ij}A_{ij}B_{ij}$: Frobenius inner product.
- 2nd term: removes the grad component that would only rescale $W$ along its top singular direction.
```

```{tip} Derivation
:class: dropdown
*Why does normalizing each layer bound the whole network?*
1. Linear layer: $\|W\mathbf{x}_1-W\mathbf{x}_2\|\le\sigma(W)\|\mathbf{x}_1-\mathbf{x}_2\|$ → Lipschitz constant $=\sigma(W)$.
2. ReLU, LReLU: Lipschitz constant $=1$.
3. Composition: $\|f\circ h\|_\text{Lip}\le\|f\|_\text{Lip}\|h\|_\text{Lip}$.
4. → $\|D\|_\text{Lip}\le\prod_l\sigma(\bar{W}_l)=1$ w/ exact $\sigma(W_l)$.
    - Power-iteration estimate: $\hat{\sigma}=\mathbf{u}^TW\mathbf{v}\le\sigma(W)$ for unit $\mathbf{u},\mathbf{v}$ → $\sigma(\bar{W})=\sigma(W)/\hat{\sigma}\ge1$, close to 1 once the estimate converges → bound holds only approximately.

*Where does the backward come from?*
1. $\hat{\sigma}=\mathbf{u}^TW\mathbf{v}$ → $\frac{\partial\hat{\sigma}}{\partial W}=\mathbf{u}\mathbf{v}^T$.
2. $d\bar{W}=\frac{dW}{\hat{\sigma}}-\frac{W}{\hat{\sigma}^2}d\hat{\sigma}=\frac{1}{\hat{\sigma}}\left(dW-\bar{W}\langle\mathbf{u}\mathbf{v}^T,dW\rangle_F\right)$.
3. $d\mathcal{L}=\langle G,d\bar{W}\rangle_F=\frac{1}{\hat{\sigma}}\left(\langle G,dW\rangle_F-\langle G,\bar{W}\rangle_F\langle\mathbf{u}\mathbf{v}^T,dW\rangle_F\right)$ → result.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralNormLinear(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W = nn.Parameter(torch.randn(d_out, d_in))
        self.b = nn.Parameter(torch.zeros(d_out))
        ## u, v persist across steps -> warm start, so 1 power-iteration step per call suffices
        self.register_buffer("u", F.normalize(torch.randn(d_out), dim=0))
        self.register_buffer("v", F.normalize(torch.randn(d_in), dim=0))

    def forward(self, x):
        if self.training:
            with torch.no_grad():  ## u, v = constants for autograd
                self.v.copy_(F.normalize(self.W.T @ self.u, dim=0))
                self.u.copy_(F.normalize(self.W @ self.v, dim=0))
        u, v = self.u.clone(), self.v.clone()
        sigma = u @ self.W @ v     ## grad flows into W: d sigma / dW = u v^T
        return x @ (self.W / sigma).T + self.b

## Example
layer = SpectralNormLinear(4, 3)
for _ in range(100):  ## each call = 1 more power-iteration step
    layer(torch.randn(2, 4))
W_sn = layer.W / (layer.u @ layer.W @ layer.v)
print(torch.linalg.matrix_norm(W_sn, ord=2))  ## ~1.0
```
````

```{attention} Q&A
:class: dropdown
*Why bound the discriminator's Lipschitz constant?*
- Bounded $\|\nabla_\mathbf{x}D\|$ → $G$ gets smooth, bounded grads.
- [WGAN](../gen/gan.md#wgan) critic must be 1-Lipschitz by theory → SN = 1 way to enforce it.

*SN vs weight clipping vs gradient penalty?*
- Weight clipping: pushes weights toward low rank → ⬇️$D$ capacity (paper: singular values concentrate on few components).
- Gradient penalty (WGAN-GP): extra forward & backward pass. Enforced only at sampled points.
- SN: 1 power-iteration step per layer. Constraint on the weights → applies to every input, not only sampled points.
    - Approximate: $\hat{\sigma}\le\sigma(W)$ → $\sigma(\bar{W})$ slightly $\ge1$ until the estimate converges.

*SN vs Weight Normalization in $D$?*
- WN fixes each row's norm → to maximize $\|\bar{W}\mathbf{h}\|$, $D$ concentrates on 1 singular direction → low rank.
- SN fixes only the largest singular value → other singular values free → rank not penalized.

*Why is 1 power-iteration step enough?*
- $W$ changes little per step → $\mathbf{u}$ from the prev step ≈ top singular vector → 1 step refines it.

*Is the bound tight?*
- ❌. Product of per-layer norms = upper bound → true Lipschitz constant can be much smaller → SN can over-constrain.
- Conv layers: paper reshapes the kernel to a $d_{out}\times(d_{in}hw)$ matrix → its spectral norm ≠ the conv operator's exact norm (depends on stride & padding).

*Where else is it used?*
- SAGAN, BigGAN: SN in $G$ too.
- Lipschitz-constrained nets, e.g., invertible ResNets need each residual branch's Lipschitz constant $<1$.
```

&nbsp;

## RNN

```{image} ../../images/dl-module/rnn.png
:align: center
:width: 400px
```

- **What**: Keep & Update a hidden state while scanning a sequence.
- **Why**: To model temporal dependencies with a parameter-sharing cell (same weights at every timestep).
- **How**: 
    1. Keep track of a hidden state.
    2. At each step, combine current input with previous hidden state, then activate.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{x}_t\in\mathbb{R}^{H_{in}}$: Input at time $t$
    - $\mathbf{h}_t\in\mathbb{R}^{H}$: Hidden state at time $t$
    - $\mathbf{y}_t\in\mathbb{R}^{H_{out}}$: Output at time $t$ (optional)
- Params:
    - $W_{xh}\in\mathbb{R}^{H\times H_{in}},\; W_{hh}\in\mathbb{R}^{H\times H},\; \mathbf{b}_h\in\mathbb{R}^{H}$
    - (optional) $W_{hy}\in\mathbb{R}^{H_{out}\times H},\; \mathbf{b}_y\in\mathbb{R}^{H_{out}}$
- Hyperparams:
    - $H_{in}, H_{out}, H$: Input size, output size, and hidden size.

Forward:
\begin{align*}
&\text{Activation:} &&\mathbf{a}_t = W_{xh}\mathbf{x}_t + W_{hh}\mathbf{h}_{t-1} + \mathbf{b}_h \\
&\text{Hidden:} &&\mathbf{h}_t=\tanh(\mathbf{a}_t) \\
&\text{Output (optional):} &&\mathbf{y}_t = W_{hy}\mathbf{h}_t + \mathbf{b}_y
\end{align*}

Backward:
- Let $\mathbf{g}_{h_t}=\frac{\partial\mathcal{L}}{\partial \mathbf{h}_t}$. Then
$$
\delta_t \equiv \frac{\partial\mathcal{L}}{\partial \mathbf{a}_t}=\mathbf{g}_{h_t}\odot(1-\mathbf{h}_t^2)
$$
- Gradients:
\begin{align*}
&\frac{\partial\mathcal{L}}{\partial W_{xh}} \mathrel{+}= \delta_t \mathbf{x}_t^T \\
&\frac{\partial\mathcal{L}}{\partial W_{hh}} \mathrel{+}= \delta_t \mathbf{h}_{t-1}^T \\
&\frac{\partial\mathcal{L}}{\partial \mathbf{b}_h} \mathrel{+}= \delta_t \\
&\frac{\partial\mathcal{L}}{\partial \mathbf{h}_{t-1}} \mathrel{+}= W_{hh}^T\delta_t
\end{align*}
```

```{attention} Q&A
:class: dropdown
*Cons?*
- **Vanishing/exploding gradients** in backprop-through-time (BPTT), esp. for long sequences.
- Hard to preserve information for many timesteps.
```

&nbsp; 

### GRU

```{image} ../../images/dl-module/gru.png
:align: center
:width: 400px
```

- **What**: Gated Recurrent Unit.
    - **Gate**: controls how much past information to keep vs overwrite.
- **Why**:
    - Stable gradients.
    - Selective memory.
- **How**:
    - **Update gate**: Keep info nearly unchanged for many steps.
    - **Reset gate**: Ignore irrelevant history when forming candidate.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{x}_t\in\mathbb{R}^{H_{in}}$
    - $\mathbf{h}_t\in\mathbb{R}^{H}$
- Intermediate:
    - $\mathbf{z}_t\in\mathbb{R}^{H}$: Update gate.
    - $\mathbf{r}_t\in\mathbb{R}^{H}$: Reset gate.
    - $\tilde{\mathbf{h}}_t\in\mathbb{R}^{H}$: Candidate state.

Forward:
\begin{align*}
&\text{Update:}  &&\mathbf{z}_t=\sigma(W_{xz}\mathbf{x}_t+W_{hz}\mathbf{h}_{t-1}+\mathbf{b}_z) \\
&\text{Reset:}  &&\mathbf{r}_t=\sigma(W_{xr}\mathbf{x}_t+W_{hr}\mathbf{h}_{t-1}+\mathbf{b}_r) \\
&\text{Candidate:} &&\tilde{\mathbf{h}}_t=\tanh(W_{xh}\mathbf{x}_t+W_{hh}(\mathbf{r}_t\odot \mathbf{h}_{t-1})+\mathbf{b}_h) \\
&\text{Hidden:}  &&\mathbf{h}_t=(1-\mathbf{z}_t)\odot \mathbf{h}_{t-1}+\mathbf{z}_t\odot \tilde{\mathbf{h}}_t
\end{align*}
```

```{attention} Q&A
:class: dropdown
*Cons?*
- Still struggle with long dependencies.
- Computational cost ⬆️.
- Interpretability ⬇️.
```

&nbsp;

### LSTM

```{image} ../../images/dl-module/lstm.png
:align: center
:width: 400px
```

- **What**: Long Short-Term Memory.
    - GRU + cell state (i.e., memory).
- **Why**:
    - Memory cell mitigates vanishing gradients.
    - Finer control over write/keep/expose operations → More robust on longer sequences.
- **How**:
    - **Forget gate**: How much previous memory to keep/erase.
    - **Input gate**: How much new info to write into memory.
    - **Candidate**: New content that could be stored in memory.
    - **Cell state**: Running memory that carries info forward through time with minimal distortion.
    - **Output gate**: How much memory to reveal as the hidden/output state.


```{note} Math
:class: dropdown
Notations:
- IO:
  - $\mathbf{x}_t\in\mathbb{R}^{H_{in}}$
  - $\mathbf{h}_t\in\mathbb{R}^{H}$
  - $\mathbf{c}_t\in\mathbb{R}^{H}$: Cell state.
- Intermediate:
  - $\mathbf{i}_t\in\mathbb{R}^{H}$: Input.
  - $\mathbf{f}_t\in\mathbb{R}^{H}$: Forget.
  - $\mathbf{o}_t\in\mathbb{R}^{H}$: Output.
  - $\tilde{\mathbf{c}}_t\in\mathbb{R}^{H}$: Candidate.

Forward:
\begin{align*}
&\text{Forget:} &&\mathbf{f}_t=\sigma\left(W_{xf}\mathbf{x}_t+W_{hf}\mathbf{h}_{t-1}+\mathbf{b}_f\right) \\
&\text{Input:} &&\mathbf{i}_t=\sigma\left(W_{xi}\mathbf{x}_t+W_{hi}\mathbf{h}_{t-1}+\mathbf{b}_i\right) \\
&\text{Candidate:} &&\tilde{\mathbf{c}}_t=\tanh\left(W_{xc}\mathbf{x}_t+W_{hc}\mathbf{h}_{t-1}+\mathbf{b}_c\right) \\
&\text{Cell state:} &&\mathbf{c}_t=\mathbf{f}_t\odot \mathbf{c}_{t-1}+\mathbf{i}_t\odot \tilde{\mathbf{c}}_t \\
&\text{Output:} &&\mathbf{o}_t=\sigma\left(W_{xo}\mathbf{x}_t+W_{ho}\mathbf{h}_{t-1}+\mathbf{b}_o\right) \\
&\text{Hidden:} &&\mathbf{h}_t=\mathbf{o}_t\odot \tanh(\mathbf{c}_t)
\end{align*}
```

```{attention} Q&A
:class: dropdown
**Pros?**
- Strong long-range memory in the RNN era.
- Very stable training in many sequence tasks.

**Cons?**
- SO SLOW.
- No one cares. It's the transformer era.
```

&nbsp;
