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

This page does NOT cover attention & transformer modules.

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
- **Why**: To mitigate [vanishing/exploding gradients](../dl/issues.md/).
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
- **Why**: To mitigate [internal covariate shift](../dl/issues.md/#vanishing/internal-covariate-shift).
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
- Mitigates [vanishing/exploding gradients](../dl/issues.md/).

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
- Mitigates [vanishing/exploding gradients](../dl/issues.md/).

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

## Convolution
- **What**: Slide a set of filters over input data to extract local features.
- **Why**: To learn spatial hierarchies of local features.
- **How**: 
    1. Initialize multiple small matrices (i.e., **filters/kernels**).
    2. From top-left to Bottom-right: 
        1. Perform element-wise multiplication and summation between each filter & scanned area of input data
        2. Store the output in the corresponding position as a feature map.

```{note} Math
:class: dropdown
Notations:
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

Forward:

$$
Y_{h,w,c_{out}}=\sum_{c_{in}=1}^{C_{in}}\sum_{i=1}^{f_h}\sum_{j=1}^{f_w}W_{i,j,c_{out},c_{in}}\cdot X_{s(h-1)+i-p,s(w-1)+j-p,c_{in}}+b_{c_{out}}
$$

where

$$\begin{align*}
H_{out}&=\left\lfloor\frac{H_{in}+2p-f_h}{s}\right\rfloor+1\\
W_{out}&=\left\lfloor\frac{W_{in}+2p-f_w}{s}\right\rfloor+1
\end{align*}$$

Backward:

$$\begin{align*}
&\frac{\partial\mathcal{L}}{\partial W_{i,j,c_{out},c_{in}}}=\sum_{h=1}^{H_{out}}\sum_{w=1}^{W_{out}}g_{h,w,c_{out}}\cdot X_{s(h-1)+i-p, s(w-1)+j-p, c_{in}}\\
&\frac{\partial\mathcal{L}}{\partial b_{c_{out}}}=\sum_{h=1}^{H_{out}}\sum_{w=1}^{W_{out}}g_{h,w,c_{out}}\\
&\frac{\partial\mathcal{L}}{\partial X_{i,j,c_{in}}}=\sum_{c_{out}=1}^{C_{out}}\sum_{h=1}^{f_h}\sum_{w=1}^{f_w}g_{h,w,c_{out}}\cdot W_{i-s(h-1)+p,j-s(w-1)+p,c_{out},c_{in}}
\end{align*}$$

Notice it is similar to backprop of linear layer except it sums over the scanned area and removes padding.
```

```{attention} Q&A
:class: dropdown
*Pros?*
- Translation invariance.
- Efficiently captures spatial hierarchies.

*Cons?*
- High computational cost.
- Requires big data to be performant.
- Requires extensive hyperparam tuning.
```

&nbsp;

### Depthwise Convolution
- **What**: Apply a single convolutional filter to each input channel independently.
- **Why**:
    - To learn spatial features within each channel separately.
    - → Significantly reduce computational cost and model params compared to standard convolution.
- **How**:
    1. Initialize a set of filters, one for each input channel.
    2. For each input channel, from top-left to bottom-right:
        1. Perform element-wise multiplication & summation between its dedicated filter & the scanned area.
        2. Store the output in the corresponding position in the respective output feature map.
    3. The resulting feature maps (one for each input channel) are typically stacked together.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{X}\in\mathbb{R}^{H_{in}\times W_{in}\times C_{in}}$: Input volume.
    - $\mathbf{Y}\in\mathbb{R}^{H_{out}\times W_{out}\times C_{in}}$: Output volume. (Note: $C_{out} = C_{in}$ for a pure depthwise convolution layer)
- Params:
    - $\mathbf{W}\in\mathbb{R}^{F_{H}\times F_{W}\times C_{in}}$: Filters (one $F_H \times F_W$ filter per input channel).
    - $\mathbf{b}\in\mathbb{R}^{C_{in}}$: Biases (one bias per input channel).
- Hyperparams:
    - $H_{in}, W_{in}$: Input height & width.
    - $C_{in}$: #Input channels (and also #Output channels).
    - $f_h, f_w$: Filter height & width.
    - $s$: Stride size.
    - $p$: Padding size.

Forward:

$$
Y_{h,w,c}=\sum_{i=1}^{f_h}\sum_{j=1}^{f_w}W_{i,j,c}\cdot X_{s(h-1)+i-p,s(w-1)+j-p,c}+b_{c}
$$

Backward:

$$\begin{align*}
&\frac{\partial\mathcal{L}}{\partial W_{i,j,c}}=\sum_{h=1}^{H_{out}}\sum_{w=1}^{W_{out}}g_{h,w,c}\cdot X_{s(h-1)+i-p, s(w-1)+j-p, c}\\
&\frac{\partial\mathcal{L}}{\partial b_{c}}=\sum_{h=1}^{H_{out}}\sum_{w=1}^{W_{out}}g_{h,w,c}\\
&\frac{\partial\mathcal{L}}{\partial X_{i',j',c}}=\sum_{h=1}^{H_{out}}\sum_{w=1}^{W_{out}}\sum_{k_h=1}^{f_h}\sum_{k_w=1}^{f_w} \left( g_{h,w,c} \cdot W_{k_h,k_w,c} \cdot \mathbb{1}(i' = s(h-1)+k_h-p \land j' = s(w-1)+k_w-p) \right)
\end{align*}$$

- More practically, the gradient with respect to the input $\mathbf{X}$ involves a "full" convolution of the gradients $g_c$ (padded appropriately) with the corresponding flipped filter $W_c$.

Notice the similarity to the standard convolution's backpropagation but applied independently for each channel.
```

```{attention} Q&A
:class: dropdown
*Pros?*
- Computational cost⬇️⬇️ ← #Params⬇️⬇️ & #Multiplications⬇️⬇️
- Learns per-channel spatial features.

*Cons?*
- ❌Cross-channel info.
```

&nbsp;

### Pooling
- **What**: Convolution but
    - computes a heuristic per scanned patch.
    - uses the same #channels.
- **Why**: Dimensionality reduction while preserving dominant features.
- **How**: Slide the pooling window over the input & apply the heuristic (max/avg) within the scanned patch.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{X}\in\mathbb{R}^{H_{in}\times W_{in}\times C_{in}}$: Input volume.
    - $\mathbf{Y}\in\mathbb{R}^{H_{out}\times W_{out}\times C_{in}}$: Output volume.
- Hyperparams:
    - $H_{in}, W_{in}$: Input height & width.
    - $C_{in}$: #Input channels.
    - $f_h, f_w$: Filter height & width.
    - $s$: Stride size.

Forward:

$$\begin{array}{ll}
\text{Max:} & Y_{h,w,c}=\max_{i=1,\cdots,f_h\ |\ j=1,\cdots,f_w}X_{sh+i,sw+j,c}\\
\text{Avg:} & Y_{h,w,c}=\frac{1}{f_hf_w}\sum_{i=1}^{f_h}\sum_{j=1}^{f_w}X_{sh+i,sw+j,c}
\end{array}$$

where

$$\begin{align*}
H_{out}&=\left\lfloor\frac{H_{in}-f_h}{s}\right\rfloor+1\\
W_{out}&=\left\lfloor\frac{W_{in}-f_h}{s}\right\rfloor+1
\end{align*}$$

Backward:

$$\begin{array}{ll}
\text{Max:} & \frac{\partial\mathcal{L}}{\partial X_{sh+i,sw+j,c}}=g_{h,w,c}\text{ if }X_{sh+i,sw+j,c}=Y_{h,w,c}\\
\text{Avg:} & \frac{\partial\mathcal{L}}{\partial X_{sh+i,sw+j,c}}=\frac{g_{h,w,c}}{f_hf_w}
\end{array}$$

- Max: Gradients only propagate to the max element of each window.
- Avg: Gradients are equally distributed among all elements in each window.
```

```{attention} Q&A
:class: dropdown
*Max vs Average*
- **Max**: Captures most dominant features; higher robustness.
- **Avg**: Preserves more info; provides smoother features; dilutes the importance of dominant features.

*Pros?*
- Computational cost⬇️⬇️.
- ❌ params.
- Preserves translation invariance w/o losing too much info.
- Overfitting ⬇️.
- Robustness ⬆️.

*Cons?*
- Slight spatial info loss.
- Requires hyperparam tuning.
    - Large filter or stride results in coarse features.
```

&nbsp;

## RNN

```{image} ../images/dl-module/rnn.png
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

```{image} ../images/dl-module/gru.png
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

```{image} ../images/dl-module/lstm.png
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

## Activation
- **What**: An element-wise non-linear function over a layer's output.
- **Why**: Non-linearity.
    - No activation → NN = LinReg.

### Binary-like
- **What**: Functions with near-binary outputs.
- **Why**: Biological neurons generally:
    - react little to small inputs.
    - react rapidly after input stimulus passes a threshold.
    - converge to a max as stimulus increases.

&nbsp;

#### Sigmoid
- **What**: Sigmoid function.
- **Why**: Mathematically convenient ← Smooth gradient.

```{note} Math
:class: dropdown
Forward:

$$
y=\sigma(z)=\frac{1}{1+e^{-z}}
$$
- $\sigma(z)\in(0,1)$
- $\sigma(0)=0.5$


Backward:

$$
\frac{\partial\mathcal{L}}{\partial z}=\frac{\partial\mathcal{L}}{\partial y}y(1-y)
$$
```

```{attention} Q&A
:class: dropdown
*Cons?*
-  Vanishing gradient.
-  Non-zero centric bias → Non-zero mean activations
```

&nbsp;

#### Tanh
- **What**: Tanh function.
- **Why**: Mathematically convenient ← Smooth gradient.

```{note} Math
:class: dropdown
Forward:

$$
y=\tanh(z)=\frac{e^z-e^{-z}}{e^z+e^{-z}}
$$
- $\tanh(z)\in(-1,1)$
- $\tanh(0)=0$

Backward:

$$
\frac{\partial\mathcal{L}}{\partial z}=\frac{\partial\mathcal{L}}{\partial y}(1-y^2)
$$
```

```{attention} Q&A
:class: dropdown
*Pros?*
- Zero-centered.

*Cons?*
-  Vanishing gradient.
```

&nbsp;

### ReLU
- **What**: Rectified Linear Unit
- **Why**:
    - Binary-like activation functions suffered from vanishing gradients.
    - Biological neurons either fire or remain inactive.
    - ReLU-like functions existed long ago ([Householder, 1941](https://link.springer.com/article/10.1007/BF02478220)).
- **How**: Linear for positive, 0 for negative.

```{note} Math
:class: dropdown
Forward:

$$
y=\text{ReLU}(z)=\max{(0,z)}
$$

Backward:

$$
\frac{\partial\mathcal{L}}{\partial z}=\begin{cases}
\frac{\partial\mathcal{L}}{\partial y} & z\geq0 \\
0 & z<0
\end{cases}
$$
```

```{attention} Q&A
:class: dropdown
*Pros?*
- ❌Vanishing gradient.
- ✅Sparsity.
- ✅Computational efficiency.

*Cons?*
- **Dying ReLU**: If most inputs are negative, then most neurons output 0 → No gradient → No param update $\rightarrow $ Dead. (NOTE: A SOLVABLE DISADVANTAGE)
    - Cause 1: High learning rate $ \rightarrow$ Too much subtraction in param update → Weight⬇️⬇️ → Input for neuron⬇️⬇️.
    - Cause 2: Bias too negative → Input for neuron⬇️⬇️.
- Activation explosion $\longleftarrow$ $z\rightarrow\infty$. (NOTE: NOT A SEVERE DISADVANTAGE SO FAR)
```

&nbsp;

#### LReLU
- **What**: Leaky ReLU.
- **Why**: Dying ReLU.
- **How**: Linear for positive, tiny linear for negative.

```{note} Math
:class: dropdown
Forward:

$$
y=\text{LReLU}(z)=\max{(\alpha z,z)}
$$
- $\alpha\in(0,1)$: Negative slope hyperparam, default 0.01.

Backward:

$$
\frac{\partial\mathcal{L}}{\partial z}=\begin{cases}
\frac{\partial\mathcal{L}}{\partial y} & z\geq0 \\
\alpha\frac{\partial\mathcal{L}}{\partial y} & z<0
\end{cases}
$$
```

```{attention} Q&A
:class: dropdown
*Why aren't we using LReLU in place of ReLU?*

Because Dying ReLU became insignificant.
- Empirical performance: ReLU >> LReLU ← Sparsity
- Dying ReLU is solvable with other structural changes:
    - Weight Init → Ensure sufficient initial positive weights.
    - Batch Norm → Ensure $\sim$50% input values are positive.
    - Residual Connection → Gradients can flow directly back to input even if ReLU is dead.
```

&nbsp;

#### PReLU
- **What**: Parametric ReLU.
- **Why**: Fixed slope in LReLU.
- **How**: Scale negative linear outputs by a learnable $\alpha$.

```{note} Math
:class: dropdown
Forward:

$$
y=\mathrm{PReLU}(z)=\max{(\alpha z,z)}
$$
- $\alpha\in(0,1)$: Learnable negative slope param, default 0.25.

Backward:

$$
\frac{\partial\mathcal{L}}{\partial z}=\begin{cases}
\frac{\partial\mathcal{L}}{\partial y} & z\geq0 \\
\alpha\frac{\partial\mathcal{L}}{\partial y} & z<0
\end{cases}
$$
```

```{attention} Q&A
:class: dropdown
*Pros?*
- Adaptive param learned from data.

*Cons?*
- Computational cost ⬆️.
```

&nbsp;

#### RReLU
- **What**: Randomized ReLU.
- **Why**: Dying ReLU.
- **How**: Scale negative linear outputs by a random $ \alpha $.

```{note} Math
:class: dropdown
Forward:

$$
y=\mathrm{RReLU}(z)=\max{(\alpha z,z)}
$$
- $\alpha\sim\mathrm{Uniform}(l,u)$ in training.
- $\alpha=\frac{l+u}{2}$ in inference.
- $ l,u $: hyperparams (lower bound, upper bound).

Backward:

$$
\frac{\partial\mathcal{L}}{\partial z}=\begin{cases}
\frac{\partial\mathcal{L}}{\partial y} & z\geq0 \\
\alpha\frac{\partial\mathcal{L}}{\partial y} & z<0
\end{cases}
$$
```

```{attention} Q&A
:class: dropdown
*Pros?*
- Reduce overfitting by randomization.

*Cons?*
- Computational cost ⬆️.
```

&nbsp;

### ELU
- **What**: Exponential Linear Units.
- **Why**:
    - Dying ReLU.
    - Negative outputs push mean activations to 0 → bias shift ⬇️.
    - Smooth.
- **How**: Linear for positive, exponential saturation for negative.

```{note} Math
:class: dropdown
Forward:

$$
y=\text{ELU}(z)=\begin{cases}
z & z\ge 0\\
\alpha(e^z-1) & z<0
\end{cases}
$$
- $\alpha>0$: controls negative saturation level ($y\to -\alpha$ as $z\to -\infty$).

Backward:

$$
\frac{\partial\mathcal{L}}{\partial z}=\begin{cases}
\frac{\partial\mathcal{L}}{\partial y} & z\ge 0 \\
\frac{\partial\mathcal{L}}{\partial y}\cdot \alpha e^z & z<0
\end{cases}
$$
```

```{attention} Q&A
:class: dropdown
*Pros?*
- Non-zero gradient for $z<0$ → fewer “dead” neurons than ReLU.
- Negative outputs → activations closer to zero-mean.
- Smooth for $z<0$ (and continuous at $0$).

*Cons?*
- Saturation for very negative $z$: $\alpha e^z \to 0$ → gradients can still vanish on far negative side.
- Computational cost ⬆️ (exp) vs ReLU.
- Can be less GPU-friendly than piecewise-linear activations.
```

&nbsp;

#### SELU
- **What**: Scaled ELU.
- **Why**: Self-normalizing to $N(0,1)$ → Vanishing/Exploding activations ⬇️⬇️
- **How**: ELU scaled by fixed constants.

```{note} Math
:class: dropdown
Forward:

$$
y=\text{SELU}(z)=\lambda\begin{cases}
z & z\ge 0\\
\alpha(e^z-1) & z<0
\end{cases}
$$
- $\alpha \approx 1.6733, \lambda \approx 1.0507$


Backward:

$$
\frac{\partial\mathcal{L}}{\partial z}=\begin{cases}
\lambda\frac{\partial\mathcal{L}}{\partial y} & z\ge 0 \\
\lambda\frac{\partial\mathcal{L}}{\partial y}\cdot \alpha e^z & z<0
\end{cases}
$$
```

```{attention} Q&A
:class: dropdown
*Pros?*
- Self-normalization.
- Reduce reliance on explicit normalization layers in some settings.

*Cons?*
- Useless with Normalization layers.
```

&nbsp;

#### CELU
- **What**: Continuously Differentiable Exponential Linear Unit.
- **Why**: ELU is NOT continuously differentiable at $0$ unless $\alpha=1$.
- **How**: Linear for positive, scaled exponential for negative with $z/\alpha$ inside the exp.

```{note} Math
:class: dropdown
Forward:

$$
y=\text{CELU}(z)=\begin{cases}
z & z\ge 0\\
\alpha\left(e^{z/\alpha}-1\right) & z<0
\end{cases}
$$
- $\alpha>0$: controls negative saturation level ($y\to -\alpha$ as $z\to -\infty$).

Backward:

$$
\frac{\partial\mathcal{L}}{\partial z}=\begin{cases}
\frac{\partial\mathcal{L}}{\partial y} & z\ge 0 \\
\frac{\partial\mathcal{L}}{\partial y}\cdot e^{z/\alpha} & z<0
\end{cases}
$$

- $\frac{d}{dz}\text{CELU}(0^-)=e^{0}=1=\frac{d}{dz}\text{CELU}(0^+)$
```

```{attention} Q&A
:class: dropdown
*Pros?*
- Continuously differentiable at $0$ for any $\alpha$ (smooth gradient through the transition).

*Cons?*
- Saturates for very negative $z$ ($e^{z/\alpha}\to 0$) → vanishing gradients far left.
- Computational cost ⬆️ (exp).
```

&nbsp;

### Other LUs
#### Softplus
- **What**: Smooth approximation to ReLU.
- **Why**: Differentiable everywhere (no kink at 0).
- **How**: Log-sum-exp smooth ramp.

```{note} Math
:class: dropdown
Forward:

$$
y=\text{Softplus}(z)=\ln(1+e^z)
$$

Backward:

$$
\frac{\partial\mathcal{L}}{\partial z}
=\frac{\partial\mathcal{L}}{\partial y}\cdot \sigma(z)
=\frac{\partial\mathcal{L}}{\partial y}\cdot \frac{1}{1+e^{-z}}
$$
```

```{attention} Q&A
:class: dropdown
*Pros?*
- Smooth everywhere.
- No 0 gradient (unlike ReLU for $z<0$).

*Cons?*
- ❌ Sparsity.
- Computational cost ⬆️ (log/exp).
- Smaller gradients for large negative inputs, vs ReLU.
```

&nbsp;

#### SiLU
- **What**: Swish / Sigmoid Linear Unit.
- **Why**: Smooth and empirically competitive in CNN family.
- **How**: Gate input by its sigmoid.

```{note} Math
:class: dropdown
Forward:

$$
y=\text{SiLU}(z)=z\,\sigma(z)
$$

Backward:

$$
\frac{\partial\mathcal{L}}{\partial z}
=\frac{\partial\mathcal{L}}{\partial y}\cdot \left(\sigma(z)+z\,\sigma(z)(1-\sigma(z))\right)
$$
```

```{attention} Q&A
:class: dropdown
*Pros?*
- Smooth.

*Cons?*
- Computational cost ⬆️ (sigmoid).
```

&nbsp;

#### GELU
- **What**: Gaussian Error Linear Unit.
- **Why**: Smooth, non-linear gating.
- **How**: Multiply input by probability it’s “kept” under a Gaussian.

```{note} Math
:class: dropdown
Forward (definition):

$$
y=\text{GELU}(z)=z\,\Phi(z)
$$

Backward (exact):

$$
\frac{\partial\mathcal{L}}{\partial z}
=\frac{\partial\mathcal{L}}{\partial y}\left(\Phi(z)+z\phi(z)\right)
$$
```

```{attention} Q&A
:class: dropdown
*Pros?*
- Empirically best in **transformers**.

*Cons?*
- Computational cost ⬆️ vs ReLU (though approximations help).
```

&nbsp;

### Softmax
- **What**: Numbers → Probabilities
- **Why**: Multiclass classification.
- **How**:
    1. Exponentiation: Larger/Smaller numbers → even larger/smaller numbers.
    2. Normalization: Numbers → Probabilities

```{note} Math
:class: dropdown
Forward:

$$
y_i=\text{softmax}(z_i)=\frac{\exp{(z_i)}}{\sum_j{\exp{(z_j)}}}
$$
- $i \& j$: Class indices. 

Backward:

$$
\frac{\partial\mathcal{L}}{\partial z_i}=\begin{cases}
\frac{\partial\mathcal{L}}{\partial y_i}y_i(1-y_i) & i=j \\
-\frac{\partial\mathcal{L}}{\partial y_i}y_iy_j & i\neq j
\end{cases}
$$
```