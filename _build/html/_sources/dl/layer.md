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
# Layer
A layer is a function that maps input tensor $X$ to output tensor $Y$.

Let $g$ denote the gradient $\frac{\partial\mathcal{L}}{\partial y}$ for readability.

<br/>

# Basic
## Linear
- **What**: Linear transformation.
- **Why**: The simplest way to transform data & learn patterns.
- **How**: input features * weights (+ bias).

````{admonition} Math
:class: note, dropdown
:class: note, dropdown
```{tab} Vector
**Notations**:
- IO:
    - $\mathbf{x}\in\mathbb{R}^{H_{in}}$: Input vector.
    - $\mathbf{y}\in\mathbb{R}^{H_{out}}$: Output vector.
- Params:
    - $W\in\mathbb{R}^{H_{out}\times H_{in}}$: Weight matrix.
    - $\textbf{b}\in\mathbb{R}^{H_{out}}$: Bias vector.
- Hyperparams:
    - $H_{in}$: Input feature dimension.
    - $H_{out}$: Output feature dimension.

**Forward**:

$$
\textbf{y}=W\textbf{x}+\textbf{b}
$$

**Backward**:

$$\begin{align*}
&\frac{\partial\mathcal{L}}{\partial W}=\textbf{g}\textbf{x}^T \\
&\frac{\partial\mathcal{L}}{\partial\textbf{b}}=\textbf{g}\\
&\frac{\partial\mathcal{L}}{\partial\textbf{x}}=W^T\textbf{g}
\end{align*}$$
```
```{tab} Tensor
**Notations**:
- IO:
    - $\mathbf{X}\in\mathbb{R}^{*\times H_{in}}$: Input tensor.
    - $\mathbf{Y}\in\mathbb{R}^{*\times H_{out}}$: Output tensor.
- Params:
    - $W\in\mathbb{R}^{H_{out}\times H_{in}}$: Weight matrix.
    - $\textbf{b}\in\mathbb{R}^{H_{out}}$: Bias vector.
- Hyperparams:
    - $H_{in}$: Input feature dimension.
    - $H_{out}$: Output feature dimension.

**Forward**:

$$
\textbf{Y}=\textbf{X}W^T+\textbf{b}
$$

**Backward**:

$$\begin{align*}
&\frac{\partial\mathcal{L}}{\partial W}=\textbf{g}^T\textbf{X} \\
&\frac{\partial\mathcal{L}}{\partial\textbf{b}}=\sum_*\textbf{g}_*\\
&\frac{\partial\mathcal{L}}{\partial\textbf{x}}=\textbf{g}W
\end{align*}$$
````


## Dropout
- **What**: Randomly ignore some neurons during training.
- **Why**: To reduce overfitting.
- **How**: During training:
    1. Randomly set a fraction of neurons to 0.
    2. Scale the outputs/gradients on active neurons by the keep probability.

````{admonition} Math
:class: note, dropdown
```{tab} Vector
**Notations**:
- IO:
    - $\mathbf{x}\in\mathbb{R}^{H_{in}}$: Input vector.
    - $\mathbf{y}\in\mathbb{R}^{H_{in}}$: Output vector.
- Hyperparams:
    - $p$: Keep probability.
- Intermediate values:
    - $\textbf{m}\in\mathbb{R}^{H_{in}}$: binary mask, where each element $m\sim\text{Bernoulli}(p)$.

**Forward**:

$$
\textbf{y}=\frac{\textbf{m}\odot\textbf{x}}{p}
$$

**Backward**:

$$
\frac{\partial\mathcal{L}}{\partial\textbf{x}} = \frac{\textbf{m}\odot\textbf{g}}{p}
$$
```
```{tab} Tensor
**Notations**:
- IO:
    - $\mathbf{X}\in\mathbb{R}^{*\times H_{in}}$: Input tensor.
    - $\mathbf{Y}\in\mathbb{R}^{*\times H_{in}}$: Output tensor.
- Hyperparams:
    - $p$: Keep probability.
- Intermediate values:
    - $\textbf{M}\in\mathbb{R}^{*\times H_{in}}$: binary mask, where each element $m\sim\text{Bernoulli}(p)$.

**Forward**:

$$
\textbf{Y}=\frac{\textbf{M}\odot\textbf{X}}{p}
$$

**Backward**:

$$
\frac{\partial\mathcal{L}}{\partial\textbf{X}} = \frac{\textbf{M}\odot\textbf{g}}{p}
$$
```
````

```{admonition} Q&A
:class: tip, dropdown
*Cons?*
- ⬆️Training time $\leftarrow$ Longer convergence
- Needs Hyperparameter Tuning
```

## Residual Connection
- **What**: Model the residual ($Y-X$) instead of the output ($Y$).
- **Why**: To mitigate [vanishing/exploding gradients](../dl/issues.md/#vanishing/exploding-gradient).
- **How**: Add input $X$ to block output $F(X)$.
    - If the feature dimension of $X$ and $F(X)$ doesn't match, use a shortcut linear layer on $X$ to change its feature dimension.

````{admonition} Math
:class: note, dropdown
```{tab} Vector
**Notation**:
- IO:
    - $\mathbf{x}\in\mathbb{R}^{H_{in}}$: Input vector.
    - $\mathbf{y}\in\mathbb{R}^{H_{out}}$: Output vector.
- Hyperparams:
    - $F(\cdot)\in\mathbb{R}^{H_{out}}$: The aggregate function of all layers within the residual block.

**Forward**:

$$
\textbf{y}=F(\textbf{x})+\textbf{x}
$$

**Backward**:

$$
\frac{\partial\mathcal{L}}{\partial\textbf{x}}=\mathbf{g}(1+\frac{\partial F(\textbf{x})}{\partial\textbf{x}})
$$
```
```{tab} Tensor
**Notation**:
- IO:
    - $\mathbf{x}\in\mathbb{R}^{*\times H_{in}}$: Input tensor.
    - $\mathbf{y}\in\mathbb{R}^{*\times H_{out}}$: Output tensor.
- Hyperparams:
    - $F(\cdot)\in\mathbb{R}^{H_{out}}$: The aggregate function of all layers within the residual block.

**Forward**:

$$
\textbf{Y}=F(\textbf{X})+\textbf{X}
$$

**Backward**:

$$
\frac{\partial\mathcal{L}}{\partial\textbf{X}}=\mathbf{g}(1+\frac{\partial F(\textbf{X})}{\partial\textbf{X}})
$$
```
````


## Normalization
### Batch Normalization
- **What**: Normalize each feature across input samples to zero mean & unit variance.
- **Why**: To mitigate [internal covariate shift](../dl/issues.md/#vanishing/internal-covariate-shift).
- **How**:
    1. Calculate the mean and variance for each batch.
    2. Normalize the batch.
    3. Scale and shift the normalized output using learnable params.


```{admonition} Math
:class: note, dropdown
**Notation**:
- IO:
    - $\mathbf{X}\in\mathbb{R}^{m\times n}$: Input matrix.
    - $\mathbf{Y}\in\mathbb{R}^{m\times n}$: Output matrix.
- Params:
    - $\gamma\in\mathbb{R}$: Scale param.
    - $\beta\in\mathbb{R}$: Shift param.

**Forward**:
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

**Backward**:
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

```{admonition} Q&A
:class: tip, dropdown
*Pros?*
- Accelerates training with higher learning rates.
- Reduces sensitivity to weight initialization.
- Mitigates [vanishing/exploding gradients](../dl/issues.md/#vanishing/exploding-gradient).

*Cons?*
- Adds computation overhead and complexity.
- Works best when each mini-batch is representative of the overall input distribution to accurately estimate the mean and variance.
- Causes potential issues in certain cases like small mini-batches or when batch statistics differ from overall dataset statistics.
```

### Layer Normalization
- **What**: Normalize each sample across the input features to zero mean and unit variance.
- **Why**: Batch normalization depends on the batch size.
    - When it's too big, high computational cost.
    - When it's too small, the batch may not be representative of the underlying data distribution.
    - Hyperparam tuning is required to find the optimal batch size, leading to high computational cost.
- **How**:
    1. Calculate the mean and variance for each feature.
    2. Normalize the feature.
    3. Scale and shift the normalized output using learnable params.

```{admonition} Math
:class: note, dropdown
It's easy to explain with the vector form for batch normalization, but it's more intuitive to explain with the scalar form for layer normalization.

**Notations**:
- IO:
    - $x_{ij}\in\mathbb{R}$: $j$th feature value for $i$th input sample.
    - $y_{ij}\in\mathbb{R}$: $j$th feature value for $i$th output sample.
- Params:
    - $\boldsymbol{\gamma}\in\mathbb{R}^n$: Scale param.
    - $\boldsymbol{\beta}\in\mathbb{R}^n$: Shift param.

**Forward**:
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

**Backward**:
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

```{admonition} Q&A
:class: tip, dropdown
*Pros*:
- Reduces hyperparam tuning effort.
- High consistency during training and inference.
- Mitigates [vanishing/exploding gradients](../dl/issues.md/#vanishing/exploding-gradient)..

*Cons*:
- Adds computation overhead and complexity.
- Inapplicable in CNNs due to varied statistics of spatial features.
```

<br/>

# Transformer
<!-- ```{image} ../../images/transformer.png
:align: center
:width: 500px
``` -->
- **What**: **Self-attention** for sequential data.
- **Why**: **Long-range dependencies** + **Parallel processing**
- **How**:
    - Input:
        1. **Tokenization**: Sequence $\xrightarrow{\text{split}}$ Tokens
        2. **Token Embedding**: Tokens $\rightarrow$ Semantic vectors
        3. **Positional Encoding**: Semantic vectors $\xrightarrow{+\text{positional info}}$ Position-aware vectors
    - Attention:
        1. **Encoder**: (Input) Position-aware vectors $\rightarrow$ (Input) Context-aware vectors
        2. **Decoder**:
            - **Encoder-Decoder Decoder**: (Input) Context-aware vectors + (Output) Position-aware vectors $\rightarrow$ (Output) Context-aware vectors
            - **Decoder-Only Decoder**: (Input) Position-aware vectors $\rightarrow$ (Input) Masked context-aware vectors
    - Output:
        1. **Output Layer**: (Output) Context-aware vectors $\xrightarrow{\text{predict}}$ Next token

<br><br>

## Input
### Tokenization
- **What**: Sequence $\xrightarrow{\text{split}}$ Tokens
- **Why**: Machines can only read numbers.
- **How**: (tbd)

### Token Embedding
- **What**: Tokens $\rightarrow$ Semantic vectors.
- **Why**:
    - Discrete $\rightarrow$ Continuous
    - Vocab index $\rightarrow$ Semantic meaning
    - Vocab size $\xrightarrow{\text{reduced to}}$ hidden size
- **How**: Look-up table / [Linear](../basics.md#linear).

### Positional Encoding
- **What**: Semantic vectors $\xrightarrow{+\text{positional info}}$ Position-aware vectors
- **Why**:
    - Transformers don't know positions.
    - BUT positions matter!
        - No PE $\rightarrow$ self-attention scores remain unchanged regardless of token orders {cite:p}`wang_positional_encoding`.

#### Sinusoidal PE
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

<br><br>

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
    - $n$: #Features/Hidden size.
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

STEP 1 - V:

$$
\frac{\partial L}{\partial V}=A^T\frac{\partial L}{\partial Y}
$$

STEP 2 - A:

$$
\frac{\partial L}{\partial A}=\frac{\partial L}{\partial Y}V^T
$$

STEP 3 - S:

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

STEP 4 - Q&K:

$$\begin{align*}
&\frac{\partial L}{\partial Q}=\frac{\partial L}{\partial S}\frac{K}{\sqrt{d_K}} \\
&\frac{\partial L}{\partial K}=\frac{\partial L}{\partial S}^T\frac{Q}{\sqrt{d_K}}
\end{align*}$$

STEP 5 - Ws:

$$\begin{align*}
&\frac{\partial L}{\partial W_Q}=X^T\frac{\partial L}{\partial Q}\\
&\frac{\partial L}{\partial W_K}=X^T\frac{\partial L}{\partial K}\\
&\frac{\partial L}{\partial W_V}=X^T\frac{\partial L}{\partial V}
\end{align*}$$

STEP 6 - X:

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

# Convolutional
- **What**: Apply a set of filters to input data to extract local features. ([paper](https://proceedings.neurips.cc/paper_files/paper/1989/file/53c3bce66e43be4f209556518c2fcb54-Paper.pdf))
- **Why**: To learn spatial hierarchies of features.
- **How**: Slide multiple filters/kernel (i.e., small matrices) over the input data.
    - At each step, perform element-wise multiplication and summation between each filter and the scanned area, producing a feature map.
- **When**: Used with grid-like data such as images and video frames.
- **Where**: Computer Vision.
- **Pros**:
    - Translation invariance.
    - Efficiently captures spatial hierarchies.
- **Cons**:
    - High computational cost for big data
    - Requires big data to be performant.
    - Requires extensive hyperparam tuning.

<!-- ```{admonition} Math
:class: note, dropdown
**Notations**:
    - IO:
        - $\mathbf{X}\in\mathbb{R}^{H_{in}\times W_{in}\times C_{in}}$: Input volume.
        - $\mathbf{Y}\in\mathbb{R}^{H_{out}\times W_{out}\times C_{out}}$: Output volume.
    - Params:
        - $\mathbf{W}\in\mathbb{R}^{F_{H}\times F_{W}\times C_{out}\times C_{in}}$: Filters.
        - $\mathbf{b}\in\mathbb{R}^{C_{out}}$: Biases.
    - Hyperparams:
        - $H_{in}, W_{in}$: Input height & width.
        - $C_{in}$: #Input channels.
        - $C_{out}$: #Filters (i.e., #Output channels).
        - $f_h, f_w$: Filter height & width.
        - $s$: Stride size.
        - $p$: Padding size.
**Forward**:

    $$
    Y_{h,w,c_{out}}=\sum_{c_{in}=1}^{C_{in}}\sum_{i=1}^{f_h}\sum_{j=1}^{f_w}W_{i,j,c_{out},c_{in}}\cdot X_{sh+i-p,sw+j-p,c_{in}}+b_{c_{out}}
    $$

    where

    $$\begin{align*}
    H_{out}&=\left\lfloor\frac{H_{in}+2p-f_h}{s}\right\rfloor+1\\
    W_{out}&=\left\lfloor\frac{W_{in}+2p-f_w}{s}\right\rfloor+1
    \end{align*}$$

**Backward**:

    $$\begin{align*}
    &\frac{\partial\mathcal{L}}{\partial W_{i,j,c_{out},c_{in}}}=\sum_{h=1}^{H_{out}}\sum_{w=1}^{W_{out}}g_{h,w,c_{out}}\cdot X_{sh+i-p, sw+j-p, c_{in}}\\
    &\frac{\partial\mathcal{L}}{\partial b_{c_{out}}}=\sum_{h=1}^{H_{out}}\sum_{w=1}^{W_{out}}g_{h,w,c_{out}}\\
    &\frac{\partial\mathcal{L}}{\partial X_{i,j,c_{in}}}=\sum_{c_{out}=1}^{C_{out}}\sum_{h=1}^{f_h}\sum_{w=1}^{f_w}g_{h,w,c_{out}}\cdot W_{i-sh+p,j-sw+p,c_{out},c_{in}}
    \end{align*}$$

    Notice it is similar to backprop of linear layer except it sums over the scanned area and removes padding.
``` -->

## Depthwise Separable Convolution
- **What**: Depthwise convolution + Pointwise convolution. ([paper](https://arxiv.org/pdf/1610.02357))
- **Why**: To significantly reduce computational cost and #params.
- **How**:
    - **Depthwise**: Use a single filter independently per channel.
    - **Pointwise**: Use Conv1d to combine the outputs of depthwise convolution.
- **When**: When computational efficiency and model size are crucial.
- **Where**: [MobileNets](https://arxiv.org/pdf/1704.04861), [Xception](https://openaccess.thecvf.com/content_cvpr_2017/papers/Chollet_Xception_Deep_Learning_CVPR_2017_paper.pdf), etc.
- **Pros**: Significantly higher computational efficiency (time & space).
- **Cons**: Lower accuracy.

<!-- ```{admonition} Math
:class: note, dropdown
**Notations**:
    - IO:
        - $\mathbf{X} \in \mathbb{R}^{H_{in} \times W_{in} \times C_{in}}$: Input volume.
        - $\mathbf{Y} \in \mathbb{R}^{H_{out} \times W_{out} \times C_{out}}$: Output volume.
    - Params:
        - $\mathbf{W^d} \in \mathbb{R}^{f_h \times f_w \times C_{in}}$: Depthwise filters.
        - $\mathbf{b^d} \in \mathbb{R}^{C_{in}}$: Depthwise biases.
        - $\mathbf{W^p} \in \mathbb{R}^{1 \times 1 \times C_{in} \times C_{out}}$: Pointwise filters.
        - $\mathbf{b^p} \in \mathbb{R}^{C_{out}}$: Pointwise biases.
    - Hyperparams:
        - $H_{in}, W_{in}$: Input height & width.
        - $C_{in}$: #Input channels.
        - $C_{out}$: #Output channels.
        - $f_h, f_w$: Filter height & width.
        - $s$: Stride size.
        - $p$: Padding size.
**Forward**:
    1. Depthwise convolution: Calculate $\mathbf{Z} \in \mathbb{R}^{H_{out} \times W_{out} \times C_{in}}$:

        $$
        Z_{h,w,c_{in}} = \sum_{i=1}^{f_h} \sum_{j=1}^{f_w} W^d_{i,j,c_{in}} \cdot X_{sh+i-p, sw+j-p, c_{in}} + b^d_{c_{in}}
        $$

    2. Pointwise convolution:

        $$
        Y_{h,w,c_{out}} = \sum_{c_{in}=1}^{C_{in}} W^p_{1,1,c_{in},c_{out}} \cdot Z_{h,w,c_{in}} + b^p_{c_{out}}
        $$
        where
        $$\begin{align*}
        H_{out} &= \left\lfloor \frac{H_{in} + 2p - f_h}{s} \right\rfloor + 1 \\
        W_{out} &= \left\lfloor \frac{W_{in} + 2p - f_w}{s} \right\rfloor + 1
        \end{align*}$$

**Backward**:
    1. Pointwise convolution: Let $g^{p}\in\mathbb{R}^{H_{out}\times W_{out}\times C_{out}}$ be $\frac{\partial\mathcal{L}}{\partial\mathbf{Y}}$.

        $$\begin{align*}
        &\frac{\partial \mathcal{L}}{\partial W^p_{1,1,c_{in},c_{out}}} = \sum_{h=1}^{H_{out}} \sum_{w=1}^{W_{out}} g^{p}_{h,w,c_{out}} \cdot Z_{h,w,c_{in}}\\
        &\frac{\partial \mathcal{L}}{\partial b^p_{c_{out}}} = \sum_{h=1}^{H_{out}} \sum_{w=1}^{W_{out}} g^{p}_{h,w,c_{out}}\\
        &\frac{\partial \mathcal{L}}{\partial Z_{h,w,c_{in}}} = \sum_{c_{out}=1}^{C_{out}} g^{p}_{h,w,c_{out}} \cdot W^p_{1,1,c_{in},c_{out}}
        \end{align*}$$
    2. Depthwise convolution: Let $g^{d}\in\mathbb{R}^{H_{out}\times W_{out}\times C_{in}}$ be $\frac{\partial\mathcal{L}}{\partial\mathbf{Z}}$.

        $$\begin{align*}
        &\frac{\partial \mathcal{L}}{\partial W^d_{i,j,c_{in}}} = \sum_{h=1}^{H_{out}} \sum_{w=1}^{W_{out}} g^d_{h,w,c_{in}} \cdot X_{sh+i-p, sw+j-p, c_{in}}\\
        &\frac{\partial \mathcal{L}}{\partial b_{d,c_{in}}} = \sum_{h=1}^{H_{out}} \sum_{w=1}^{W_{out}} g^d_{h,w,c_{in}}\\
        &\frac{\partial \mathcal{L}}{\partial X_{i,j,c_{in}}} = \sum_{h=1}^{f_h} \sum_{w=1}^{f_w} g^d_{h,w,c_{in}} \cdot W^d_{i-sh+p,j-sw+p,c_{in}}
        \end{align*}$$
``` -->

## Atrous/Dilated Convolution
- **What**: Add holes between filter elements (i.e., dilation). ([paper](https://arxiv.org/pdf/1511.07122))
- **Why**: The filters can capture larger contextual info without increasing #params.
- **How**: Introduce a dilation rate $r$ to determine the space between the filter elements. Then compute convolution accordingly.
- **When**: When understanding the broader context is important.
- **Where**: Semantic image segmentation, object detection, depth estimation, optical flow estimation, etc.
- **Pros**:
    - Larger receptive fields without increasing #params.
    - Captures multi-scale info without upsampling layers.
- **Cons**:
    - Requires very careful hyperparam tuning, or info loss.

<!-- ```{admonition} Math
:class: note, dropdown
**Notations**:
    - IO:
        - $\mathbf{X}\in\mathbb{R}^{H_{in}\times W_{in}\times C_{in}}$: Input volume.
        - $\mathbf{Y}\in\mathbb{R}^{H_{out}\times W_{out}\times C_{out}}$: Output volume.
    - Params:
        - $\mathbf{W}\in\mathbb{R}^{F_{H}\times F_{W}\times C_{out}\times C_{in}}$: Filters.
        - $\mathbf{b}\in\mathbb{R}^{C_{out}}$: Biases.
    - Hyperparams:
        - $H_{in}, W_{in}$: Input height & width.
        - $C_{in}$: #Input channels.
        - $C_{out}$: #Filters (i.e., #Output channels).
        - $f_h, f_w$: Filter height & width.
        - $s$: Stride size.
        - $p$: Padding size.
        - $r$: Dilation rate.
**Forward**:
    1. (optional) Pad input tensor: $\mathbf{X}^\text{pad}\in\mathbb{R}^{(H_{in}+2p)\times (W_{in}+2p)\times C_{in}}$
    2. Perform element-wise multiplication (i.e., convolution):

        $$
        Y_{h,w,c_{out}}=\sum_{c_{in}=1}^{C_{in}}\sum_{i=1}^{f_h}\sum_{j=1}^{f_w}W_{i,j,c_{out},c_{in}}\cdot X_{sh+r(i-1)-p,sw+r(j-1)-p,c_{in}}+b_{c_{out}}
        $$

        where

        $$\begin{align*}
        H_{out}&=\left\lfloor\frac{H_{in}+2p-r(f_h-1)-1}{s}\right\rfloor+1\\
        W_{out}&=\left\lfloor\frac{W_{in}+2p-r(f_w-1)-1}{s}\right\rfloor+1
        \end{align*}$$

**Backward**:

    $$\begin{align*}
    &\frac{\partial\mathcal{L}}{\partial W_{i,j,c_{out},c_{in}}}=\sum_{h=1}^{H_{out}}\sum_{w=1}^{W_{out}}g_{h,w,c_{out}}\cdot X_{sh+r(i-1)-p, sw+r(j-1)-p, c_{in}}\\
    &\frac{\partial\mathcal{L}}{\partial b_{c_{out}}}=\sum_{h=1}^{H_{out}}\sum_{w=1}^{W_{out}}g_{h,w,c_{out}}\\
    &\frac{\partial\mathcal{L}}{\partial X_{i,j,c_{in}}}=\sum_{c_{out}=1}^{C_{out}}\sum_{h=1}^{f_h}\sum_{w=1}^{f_w}g_{h,w,c_{out}}\cdot W_{r(i-1)-sh+p,r(j-1)-sw+p,c_{out},c_{in}}
    \end{align*}$$
``` -->

## Pooling
- **What**: Convolution but ([paper](https://proceedings.neurips.cc/paper_files/paper/1989/file/53c3bce66e43be4f209556518c2fcb54-Paper.pdf))
    - computes a heuristic per scanned patch.
    - uses the same #channels.
- **Why**: Dimensionality reduction while preserving dominant features.
- **How**: Slide the pooling window over the input & apply the heuristic within the scanned patch.
    - **Max**: Output the maximum value from each patch.
    - **Average**: Output the average value of each patch.
- **When**: When downsampling is necessary.
- **Where**: After convolutional layer.
- **Pros**:
    - Significantly higher computational efficiency (time & space).
    - No params to train.
    - Reduces overfitting.
    - Preserves translation invariance without losing too much info.
    - High robustness.
- **Cons**:
    - Slight spatial info loss.
    - Requires hyperparam tuning.
        - Large filter or stride results in coarse features.
- **Max vs Average**:
    - **Max**: Captures most dominant features; higher robustness.
    - **Avg**: Preserves more info; provides smoother features; dilutes the importance of dominant features.

<!-- ```{admonition} Math
:class: note, dropdown
**Notations**:
    - IO:
        - $\mathbf{X}\in\mathbb{R}^{H_{in}\times W_{in}\times C_{in}}$: Input volume.
        - $\mathbf{Y}\in\mathbb{R}^{H_{out}\times W_{out}\times C_{in}}$: Output volume.
    - Hyperparams:
        - $H_{in}, W_{in}$: Input height & width.
        - $C_{in}$: #Input channels.
        - $f_h, f_w$: Filter height & width.
        - $s$: Stride size.
**Forward**:

    $$\begin{array}{ll}
    \text{Max:} & Y_{h,w,c}=\max_{i=1,\cdots,f_h\ |\ j=1,\cdots,f_w}X_{sh+i,sw+j,c}\\
    \text{Avg:} & Y_{h,w,c}=\frac{1}{f_hf_w}\sum_{i=1}^{f_h}\sum_{j=1}^{f_w}X_{sh+i,sw+j,c}
    \end{array}$$

    where

    $$\begin{align*}
    H_{out}&=\left\lfloor\frac{H_{in}-f_h}{s}\right\rfloor+1\\
    W_{out}&=\left\lfloor\frac{W_{in}-f_h}{s}\right\rfloor+1
    \end{align*}$$

**Backward**:

    $$\begin{array}{ll}
    \text{Max:} & \frac{\partial\mathcal{L}}{\partial X_{sh+i,sw+j,c}}=g_{h,w,c}\text{ if }X_{sh+i,sw+j,c}=Y_{h,w,c}\\
    \text{Avg:} & \frac{\partial\mathcal{L}}{\partial X_{sh+i,sw+j,c}}=\frac{g_{h,w,c}}{f_hf_w}
    \end{array}$$

    - Max: Gradients only propagate to the max element of each window.
    - Avg: Gradients are equally distributed among all elements in each window.
``` -->

<br/>

# Recurrent
```{image} ../images/RNN.png
:width: 400
:align: center
```

$$
h_t=\tanh(x_tW_{xh}^T+h_{t-1}W_{hh}^T)
$$

Idea: **recurrence** - maintain a hidden state that captures information about previous inputs in the sequence

Notations:
- $ x_t$: input at time $t$ of shape $(m,H_{in}) $
- $ h_t$: hidden state at time $t$ of shape $(D,m,H_{out}) $
- $ W_{xh}$: weight matrix of shape $(H_{out},H_{in})$ if initial layer, else $(H_{out},DH_{out}) $
- $ W_{hh}$: weight matrix of shape $(H_{out},H_{out}) $
- $ H_{in}$: input size, #features in $x_t $
- $ H_{out}$: hidden size, #features in $h_t $
- $ m $: batch size
- $ D$: $=2$ if bi-directional else $1 $

Cons:
- Short-term memory: hard to carry info from earlier steps to later ones if long seq
- Vanishing gradient: gradients in earlier parts become extremely small if long seq

## GRU

```{image} ../images/GRU.png
:width: 400
:align: center
```

$$\begin{align*}
&r_t=\sigma(x_tW_{xr}^T+h_{t-1}W_{hr}^T) \\\\
&z_t=\sigma(x_tW_{xz}^T+h_{t-1}W_{hz}^T) \\\\
&\tilde{h}\_t=\tanh(x_tW_{xn}^T+r_t\odot(h_{t-1}W_{hn}^T)) \\\\
&h_t=(1-z_t)\odot\tilde{h}\_t+z_t\odot h_{t-1}
\end{align*}$$

Idea: Gated Recurrent Unit - use 2 gates to address long-term info propagation issue in RNN:
1. **Reset gate**: determine how much of $ h_{t-1}$ should be ignored when computing $\tilde{h}\_t $.
2. **Update gate**: determine how much of $ h_{t-1}$ should be retained for $h_t $.
3. **Candidate**: calculate candidate $ \tilde{h}\_t$ with reset $h_{t-1} $.
4. **Final**: calculate weighted average between candidate $ \tilde{h}\_t$ and prev state $h_{t-1} $ with the retain ratio.

Notations:
- $ r_t$: reset gate at time $t$ of shape $(m,H_{out}) $
- $ z_t$: update gate at time $t$ of shape $(m,H_{out}) $
- $ \tilde{h}\_t$: candidate hidden state at time $t$ of shape $(m,H_{out}) $
- $ \odot $: element-wise product

## LSTM

```{image} ../images/LSTM.png
:width: 400
:align: center
```

$$\begin{align*}
&i_t=\sigma(x_tW_{xi}^T+h_{t-1}W_{hi}^T) \\\\
&f_t=\sigma(x_tW_{xf}^T+h_{t-1}W_{hf}^T) \\\\
&\tilde{c}\_t=\tanh(x_tW_{xc}^T+h_{t-1}W_{hc}^T) \\\\
&c_t=f_t\odot c_{t-1}+i_t\odot \tilde{c}\_t \\\\
&o_t=\sigma(x_tW_{xo}^T+h_{t-1}W_{ho}^T) \\\\
&h_t=o_t\odot\tanh(c_t)
\end{align*}$$

Idea: Long Short-Term Memory - use 3 gates:
1. **Input gate**: determine what new info from $ x_t$ should be added to cell state $c_t $.
2. **Forget gate**: determine what info from prev cell $ c_{t-1} $ should be forgotten.
3. **Candidate cell**: create a new candidate cell from $ x_t$ and $h_{t-1} $.
4. **Update cell**: use $ i_t$ and $f_t $ to combine prev and new candidate cells.
5. **Output gate**: determine what info from curr cell $ c_t$ should be added to output $h_t $.
6. **Final**: simply apply $ o_t$ to activated cell $c_t $.

Notations:
- $ i_t$: input gate at time $t$ of shape $(m,H_{out}) $
- $ f_t$: forget gate at time $t$ of shape $(m,H_{out}) $
- $ c_t$: cell state at time $t$ of shape $(m,H_{cell}) $
- $ o_t$: output gate at time $t$ of shape $(m,H_{out}) $
- $ H_{cell}$: cell hidden size (in most cases same as $H_{out} $)

## Bidirectional
## Stacked

<br/>

# Activation
An activation function adds nonlinearity to the output of a layer to enhance complexity. [ReLU](#relu) and [Softmax](#softmax) are SOTA.

Notations:
- $ z $: input (element-wise)

## Binary-like

### Sigmoid

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

Idea:
-  $ \sigma(z)\in(0,1)$ and $\sigma(0)=0.5 $.

Pros:
-  imitation of the firing rate of a neuron, 0 if too negative and 1 if too positive.
-  smooth gradient.

Cons:
-  vanishing gradient: gradients rapidly shrink to 0 along backprop as long as any input is too positive or too negative.
-  non-zero centric bias $ \rightarrow $ non-zero mean activations.
-  computationally expensive.

### Tanh

$$
\tanh(z)=\frac{e^z-e^{-z}}{e^z+e^{-z}}
$$

Idea:
- $ \tanh(z)\in(-1,1)$ and $\tanh(0)=0 $.

Pros:
- zero-centered
- imitation of the firing rate of a neuron, -1 if too negative and 1 if too positive.
- smooth gradient.

Cons:
- vanishing gradient.
- computationally expensive.

## Linear Units (Rectified)

### ReLU

$$
\mathrm{ReLU}(z)=\max{(0,z)}
$$

Name: Rectified Linear Unit

Idea:
- convert negative linear outputs to 0.

Pros:
- no vanishing gradient
- activate fewer neurons
- much less computationally expensive compared to sigmoid and tanh.

Cons:
- dying ReLU: if most inputs are negative, then most neurons output 0 $ \rightarrow$ no gradient for such neurons $\rightarrow$ no param update $\rightarrow $ they die. (NOTE: A SOLVABLE DISADVANTAGE)

    - Cause 1: high learning rate $ \rightarrow$ too much subtraction in param update $\rightarrow$ weight too negative $\rightarrow $ input for neuron too negative.
    - Cause 2: bias too negative $ \rightarrow $ input for neuron too negative.

-  activation explosion as $ z\rightarrow\infty $. (NOTE: NOT A SEVERE DISADVANTAGE SO FAR)


### LReLU

$$
\mathrm{LReLU}(z)=\max{(\alpha z,z)}
$$


Name: Leaky Rectified Linear Unit

Params:
- $ \alpha\in(0,1) $: hyperparam (negative slope), default 0.01.

Idea:
- scale negative linear outputs by $ \alpha $.

Pros:
- no dying ReLU.

Cons:
- slightly more computationally expensive than ReLU.
- activation explosion as $ z\rightarrow\infty $.


### PReLU

$$
\mathrm{PReLU}(z)=\max{(\alpha z,z)}
$$

Name: Parametric Rectified Linear Unit

Params:
- $ \alpha\in(0,1) $: learnable param (negative slope), default 0.25.

Idea:
- scale negative linear outputs by a learnable $ \alpha $.

Pros:
- a variable, adaptive param learned from data.

Cons:
- slightly more computationally expensive than LReLU.
- activation explosion as $ z\rightarrow\infty $.



### RReLU

$$
\mathrm{RReLU}(z)=\max{(\alpha z,z)}
$$


Name: Randomized Rectified Linear Unit

Params:
- $ \alpha\sim\mathrm{Uniform}(l,u) $: a random number sampled from a uniform distribution.
- $ l,u $: hyperparams (lower bound, upper bound)

Idea:
- scale negative linear outputs by a random $ \alpha $.

Pros:
- reduce overfitting by randomization.

Cons:
- slightly more computationally expensive than LReLU.
- activation explosion as $ z\rightarrow\infty $.

## Linear Units (Exponential)

### ELU

$$
\mathrm{ELU}(z)=\begin{cases}
z & \mathrm{if}\ z\geq0 \\\\
\alpha(e^z-1) & \mathrm{if}\ z<0
\end{cases}
$$


Name: Exponential Linear Unit

Params:
- $ \alpha $: hyperparam, default 1.

Idea:
- convert negative linear outputs to the non-linear exponential function above.

Pros:
- mean unit activation is closer to 0 $ \rightarrow $ reduce bias shift (i.e., non-zero mean activation is intrinsically a bias for the next layer.)
- lower computational complexity compared to batch normalization.
- smooth to $ -\alpha $ slowly with smaller derivatives that decrease forwardprop variation.
- faster learning and higher accuracy for image classification in practice.

Cons:
- slightly more computationally expensive than ReLU.
- activation explosion as $ z\rightarrow\infty $.



### SELU

$$
\mathrm{SELU}(z)=\lambda\begin{cases}
z & \mathrm{if}\ z\geq0 \\
\alpha(e^z-1) & \mathrm{if}\ z<0
\end{cases}
$$


Name: Scaled Exponential Linear Unit

Params:
- $ \alpha $: hyperparam, default 1.67326.
- $ \lambda $: hyperparam (scale), default 1.05070.

Idea:
- scale ELU.

Pros:
- self-normalization $ \rightarrow $ activations close to zero mean and unit variance that are propagated through many network layers will converge towards zero mean and unit variance.

Cons:
- more computationally expensive than ReLU.
- activation explosion as $ z\rightarrow\infty $.



### CELU

$$
\mathrm{CELU}(z)=\begin{cases}
z & \mathrm{if}\ z\geq0\\
\alpha(e^{\frac{z}{\alpha}}-1) & \mathrm{if}\ z<0
\end{cases}
$$


Name: Continuously Differentiable Exponential Linear Unit

Params:
- $ \alpha $: hyperparam, default 1.

Idea:
- scale the exponential part of ELU with $ \frac{1}{\alpha} $ to make it continuously differentiable.

Pros:
- smooth gradient due to continuous differentiability (i.e., $ \mathrm{CELU}'(0)=1 $).

Cons:
- slightly more computationally expensive than ELU.
- activation explosion as $ z\rightarrow\infty $.

## Linear Units (Others)

### GELU

$$
\mathrm{GELU}(z)=z*\Phi(z)=0.5z(1+\tanh{[\sqrt{\frac{2}{\pi}}(z+0.044715z^3)]})
$$


Name: Gaussian Error Linear Unit

Idea:
- weigh each output value by its Gaussian cdf.

Pros:
- throw away gate structure and add probabilistic-ish feature to neuron outputs.
- seemingly better performance than the ReLU and ELU families, SOTA in transformers.

Cons:
- slightly more computationally expensive than ReLU.
- lack of practical testing at the moment.



### SiLU

$$
\mathrm{SiLU}(z)=z*\sigma(z)
$$

Name: Sigmoid Linear Unit

Idea:
- weigh each output value by its sigmoid value.

Pros:
- throw away gate structure.
- seemingly better performance than the ReLU and ELU families.

Cons:
- worse than GELU.



### Softplus

$$
\mathrm{softplus}(z)=\frac{1}{\beta}\log{(1+e^{\beta z})}
$$


Idea:
- smooth approximation of ReLU.

Pros:
- differentiable and thus theoretically better than ReLU.

Cons:
- empirically far worse than ReLU in terms of computation and performance.



## Multiclass

### Softmax

$$
\mathrm{softmax}(z_i)=\frac{\exp{(z_i)}}{\sum_j{\exp{(z_j)}}}
$$


Idea:
- convert each value $ z_i$ in the output tensor $\mathbf{z}$ into its corresponding exponential probability s.t. $\sum_i{\mathrm{softmax}(z_i)}=1 $.

Pros:
- your single best choice for multiclass classification.

Cons:
- mutually exclusive classes (i.e., one input can only be classified into one class.)

### Softmin

$$
\mathrm{softmin}(z_i)=\mathrm{softmax}(-z_i)=\frac{\exp{(-z_i)}}{\sum_j{\exp{(-z_j)}}}
$$

Idea:
- reverse softmax.

Pros:
- suitable for multiclass classification.

Cons:
- why not softmax.