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
# CNN Architectures
Convolutional networks for vision: building blocks, design principles, & the major classification-backbone families. Generic modules → [Module](../dl/foundation/module.md). U-Net → [Diffusion](../dl/gen/diffusion.md#u-net).

Notations:
- $\mathbf{X}\in\mathbb{R}^{C_{in}\times H_{in}\times W_{in}}$: Input feature map
- $\mathbf{Y}\in\mathbb{R}^{C_{out}\times H_{out}\times W_{out}}$: Output feature map
- $C_{in}$: #input channels
- $C_{out}$: #output channels
- $H$: Feature-map height
- $W$: Feature-map width
- $c$: Channel idx
- $h$: Row idx
- $w$: Column idx
- $\mathbf{K}$: Kernel (filter) weights. Shape set per section.
- $k$: Kernel size ($k\times k$)
- $s$: Stride
- $p$: Zero padding per side
- $d$: Dilation
- $G$: #groups
- $F(\cdot)$: Residual branch of a block

Overrides of global notation: $k$ = kernel size, ❌class idx. $W$ = width, ❌weight matrix. Batch dim omitted in math. Code uses `(B, C, H, W)`. Indices are 0-based.

&nbsp;

## CNN
- **Name**: Convolutional Neural Network
- **What**: NN whose main layers are convolutions: local, weight-shared filters over a grid.
- **Why**: Images have local, position-independent structure. An MLP ignores it.
    - 224×224×3 input → 1 FC layer w/ 4096 units ≈ 617M params.
    - Conv: locality + weight sharing → #params independent of image size.
- **How**:
    1. **Stem**: Strided conv (+ pool) → fast downsample.
    2. **Stages**: Repeat [conv → norm → activation] blocks. Between stages: $H,W$ ÷2 & $C$ ×2.
    3. **Head**: [Global average pooling](#global-average-pooling) → linear → class logits.
    - Early layers → edges & colors. Later layers → parts & objects.

````{dropdown} Table 1: CNN Families
| Year | Model | Key idea | Params |
|:--|:--|:--|:--|
| 1998 | [LeNet-5](#lenet-5) | Conv + subsample + FC, end-to-end backprop | ~60K |
| 2012 | [AlexNet](#alexnet) | ReLU, dropout, GPUs, ImageNet scale | ~60M |
| 2014 | [VGG-16](#vgg) | Only 3×3 convs, depth | ~138M |
| 2014 | [GoogLeNet](#googlenet) | Inception module, 1×1 reduce, GAP | ~5–7M |
| 2015 | [ResNet-50](#resnet) | Residual learning, bottleneck | ~25.6M |
| 2016 | [DenseNet-121](#densenet) | Concat all previous features | ~8.0M |
| 2016 | [ResNeXt-50](#resnext) | Cardinality (grouped conv) | ~25.0M |
| 2017 | [SENet](#senet) | Channel gating | Backbone + ~10% |
| 2017 | [MobileNet](#mobilenet) | Depthwise separable conv | 4.2M |
| 2017 | [ShuffleNet](#shufflenet) | Grouped 1×1 + channel shuffle | — |
| 2018 | [MobileNetV2](#mobilenetv2) | Inverted residual, linear bottleneck | 3.4M |
| 2019 | [MobileNetV3](#mobilenetv3) | NAS + SE + h-swish | — |
| 2019 | [EfficientNet-B0](#efficientnet) | NAS baseline + compound scaling | 5.3M |
| 2022 | [ConvNeXt-T](#convnext) | Transformer-era design in a pure CNN | ~28M |

- Year: first public release (arXiv or conference).
- GoogLeNet params: ~6.8M from the paper's layer table. The Inception-v3 paper quotes 5M.
````

```{dropdown} Table 2: Design Principles
| Principle | Origin | Why |
|:--|:--|:--|
| Stack 3×3 kernels | VGG | Same RF as a large kernel, fewer params, more nonlinearity |
| 1×1 conv to reduce channels | NIN, GoogLeNet | Cheap channel mixing before expensive spatial convs |
| $H,W$ ÷2 & $C$ ×2 per stage | VGG, ResNet | ≈ constant FLOPs per layer |
| BatchNorm after conv | BN-Inception, ResNet | Stable, fast training |
| Residual shortcut | ResNet | Trainable 100+ layers |
| GAP head | NIN, GoogLeNet | ❌FC params, any input size |
| Depthwise separable conv | Xception, MobileNet | ~$k^2$× fewer FLOPs |
| Inverted bottleneck | MobileNetV2 | Capacity for depthwise conv, small residual tensors |
| Channel attention | SENet | Input-dependent channel weights |
| Compound scaling | EfficientNet | Balanced depth, width, & resolution |
| 7×7 depthwise, LN, GELU, patchify stem | ConvNeXt | Transformer-style design w/o attention |
```

```{attention} Q&A
:class: dropdown
*Equivariance vs invariance?*
- **Equivariant**: shift input → output shifts the same way. Conv (stride 1) is shift-equivariant.
- **Invariant**: shift input → output unchanged. Pooling & GAP → approximate shift invariance.
- Classifier wants invariance. Detector & segmenter want equivariance.

*Is a CNN truly shift-invariant?*
- ❌. Stride & pooling subsample w/o low-pass filtering → aliasing → a 1-pixel shift can change the output.
- Zero padding breaks equivariance at borders → CNNs can encode absolute position.
- ❌Rotation or scale equivariance. Get it from data augmentation or special architectures.

*What is a conv layer in MLP terms?*
- Linear layer w/ a sparse, weight-tied (doubly block-Toeplitz) matrix.
    - Locality → zeros outside the $k\times k$ window.
    - Weight sharing → same values in every row, shifted.
- → CNN = MLP w/ a hard prior. Prior right → better sample efficiency. Prior wrong → capped accuracy.

*Why halve $H,W$ & double $C$ per stage?*
- Conv MACs $\propto HW\,C_{in}C_{out}$ → $H,W$ ÷2 & $C$ ×2 → per-layer cost ≈ constant.
- Deeper → larger RF → more abstract features → more channels to represent them.

*Pros?*
- Param-efficient ← weight sharing.
- Sample-efficient ← locality & equivariance priors match images.
- Hierarchical features ← stacking grows the RF.

*Cons?*
- Global context needs depth or pooling.
- Fixed weights after training → mixing is position-based, ❌content-based (vs [attention](../nlp/attn.md)).

*How does this relate to the visual cortex?*
- Hubel & Wiesel: simple cells (local oriented edge detectors) → complex cells (pool over position). Conv & pooling copy this.
- Fukushima's Neocognitron (1980): S-cells & C-cells = conv & pooling, trained w/o backprop.
- Task-optimized CNNs predict neural responses in macaque V4 & IT {cite:p}`yamins2014performance`.
- → Optimizing for a task can reproduce brain-like representations.
- Caveat: shared representations ≠ shared mechanism. Standard CNNs are feedforward. Cortex has heavy recurrence & feedback.
```

&nbsp;

## Building Blocks

&nbsp;

### Convolution
- **What**: Dot product of each learnable $k\times k\times C_{in}$ filter with every input patch.
- **Why**: Locality & weight sharing.
    - Locality: a pixel correlates mostly w/ its neighbors → each output sees a $k\times k$ patch only.
    - Weight sharing: a pattern can appear anywhere → same filter at every position → #params independent of $H,W$.
- **How**:
    1. Pad the input w/ $p$ zeros on each side.
    2. Slide each filter over the input in steps of $s$, row by row.
    3. At each position: multiply filter & patch elementwise → sum over $k\times k\times C_{in}$ → add bias.
    4. 1 filter → 1 output channel. $C_{out}$ filters → $C_{out}$ channels.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\mathbf{K}\in\mathbb{R}^{C_{out}\times C_{in}\times k\times k}$: Filters.
    - $\mathbf{b}\in\mathbb{R}^{C_{out}}$: Biases.
- Misc:
    - $\tilde{\mathbf{X}}$: $\mathbf{X}$ zero-padded by $p$ on each spatial side.
    - $o$: Output channel idx.
    - $u,v$: Kernel row & column offsets.
    - $\mathbf{G}=\partial\mathcal{L}/\partial\mathbf{Y}$: Upstream gradient.

Forward:

$$
Y_{o,h,w}=b_o+\sum_{c=0}^{C_{in}-1}\sum_{u=0}^{k-1}\sum_{v=0}^{k-1}K_{o,c,u,v}\,\tilde{X}_{c,\,sh+u,\,sw+v}
$$

Output size (same form for $W_{out}$):

$$
H_{out}=\left\lfloor\frac{H_{in}+2p-k}{s}\right\rfloor+1
$$

Cost:

$$
\#\text{params}=C_{out}C_{in}k^2+C_{out},\qquad\#\text{MACs}=C_{out}C_{in}k^2H_{out}W_{out}
$$

- MAC: Multiply-accumulate. FLOPs ≈ 2 × MACs. Many papers report MACs as "FLOPs".

Backward:

$$\begin{align*}
\frac{\partial\mathcal{L}}{\partial K_{o,c,u,v}}&=\sum_{h,w}G_{o,h,w}\,\tilde{X}_{c,\,sh+u,\,sw+v}\\
\frac{\partial\mathcal{L}}{\partial b_o}&=\sum_{h,w}G_{o,h,w}\\
\frac{\partial\mathcal{L}}{\partial\tilde{X}_{c,h',w'}}&=\sum_{o}\sum_{\substack{h,w,u,v:\\ sh+u=h',\ sw+v=w'}}G_{o,h,w}\,K_{o,c,u,v}
\end{align*}$$

- $h',w'$: Padded-input row & column idx.
- $\partial\mathcal{L}/\partial\mathbf{X}$: Crop the $p$-wide border from $\partial\mathcal{L}/\partial\tilde{\mathbf{X}}$.
- Filter grad = correlation of the input w/ the upstream grad.
- Input grad = [transposed convolution](#transposed-convolution) of $\mathbf{G}$ w/ $\mathbf{K}$. At $s=1$: full cross-correlation of $\mathbf{G}$ w/ the 180°-flipped kernel (= true convolution w/ $\mathbf{K}$).
```

```{tip} Derivation
:class: dropdown
*Why is the input grad a transposed convolution?*
1. Conv is linear in $\tilde{\mathbf{X}}$: $\text{vec}(\mathbf{Y})=M\,\text{vec}(\tilde{\mathbf{X}})+\text{bias}$ for a sparse matrix $M$.
2. Chain rule: $\text{vec}(\partial\mathcal{L}/\partial\tilde{\mathbf{X}})=M^T\,\text{vec}(\mathbf{G})$.
3. Transposed convolution is defined as multiplication by $M^T$.
4. Entry-wise: $M_{(o,h,w),(c,h',w')}=K_{o,c,h'-sh,w'-sw}$ if $0\le h'-sh<k$ & $0\le w'-sw<k$, else 0. Summing $G_{o,h,w}M_{(o,h,w),(c,h',w')}$ over $(o,h,w)$ gives the Backward formula.
```

```{note} Example
:class: dropdown
ResNet stem: $224\times224\times3$ input, 64 filters, $k=7$, $s=2$, $p=3$, ❌bias.
- $H_{out}=\lfloor(224+6-7)/2\rfloor+1=112$.
- #params $=64\cdot3\cdot49=9{,}408$.
- #MACs $=9{,}408\cdot112^2\approx118$M.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2d(nn.Module):
    def __init__(self, c_in, c_out, k, s=1, p=0):
        super().__init__()
        self.K = nn.Parameter(torch.randn(c_out, c_in, k, k) * (c_in * k * k) ** -0.5)
        self.b = nn.Parameter(torch.zeros(c_out))
        self.k, self.s, self.p = k, s, p

    def forward(self, x):
        B, _, H, W = x.shape
        Ho = (H + 2 * self.p - self.k) // self.s + 1
        Wo = (W + 2 * self.p - self.k) // self.s + 1
        ## im2col: each k x k x C_in patch -> 1 column, ordered (c, u, v)
        cols = F.unfold(x, self.k, padding=self.p, stride=self.s)   ## (B, C_in*k*k, Ho*Wo)
        ## same filters dotted w/ every patch = weight sharing
        y = self.K.flatten(1) @ cols + self.b[:, None]               ## (B, C_out, Ho*Wo)
        return y.view(B, -1, Ho, Wo)

## Example
x = torch.randn(2, 3, 8, 8)
print(Conv2d(3, 4, k=3, s=2, p=1)(x).shape)  ## torch.Size([2, 4, 4, 4])
```
````

```{attention} Q&A
:class: dropdown
*Convolution or cross-correlation?*
- DL "convolution" = cross-correlation (❌kernel flip).
- Kernel is learned → the flip does not matter.

*Why odd $k$?*
- Has a center pixel → symmetric "same" padding $p=(k-1)/2$ at $s=1$.

*Output-size gotchas?*
- "Same" ($s=1$): $p=(k-1)/2$ → $H_{out}=H_{in}$.
- "Valid": $p=0$ → shrinks by $k-1$.
- Floor → last rows & columns are skipped if $H_{in}+2p-k$ is not divisible by $s$.

*Where do params vs compute live?*
- Params independent of $H,W$. MACs $\propto H_{out}W_{out}$.
- → Early high-res layers dominate compute. Late wide layers dominate params.

*Why `bias=False` before BatchNorm?*
- BN subtracts the per-channel mean → a constant bias cancels → wasted params.

*Pros?*
- Param-efficient ← weight sharing.
- Shift-equivariant (at $s=1$, away from borders).

*Cons?*
- Local only: 1 layer sees $k\times k$.
- Input-independent weights at inference.

*How does it relate to attention?*
- Conv: fixed weights by relative position, local window.
- Attention: weights computed from content, global window.
- 1 conv layer = attention w/ a fixed, local, position-only pattern.
```

&nbsp;

#### Dilated Convolution
- **What**: Convolution w/ $d-1$ gaps between kernel taps. {cite:p}`yu2016multi`
- **Why**: Large RF w/o downsampling or extra params.
    - Dense prediction (segmentation) needs full resolution & wide context.
- **How**:
    1. Read the input at step $d$ inside the window → $k\times k$ taps span $(d(k-1)+1)^2$ pixels.
    2. Stack layers w/ growing $d$ (1, 2, 4, …) → RF grows exponentially w/ depth.

```{note} Math
:class: dropdown
Forward:

$$
Y_{o,h,w}=b_o+\sum_{c,u,v}K_{o,c,u,v}\,\tilde{X}_{c,\,sh+du,\,sw+dv}
$$

Effective kernel & output size:

$$
k_\text{eff}=d(k-1)+1,\qquad H_{out}=\left\lfloor\frac{H_{in}+2p-d(k-1)-1}{s}\right\rfloor+1
$$

- $d=1$ → standard conv.
```

```{note} Example
:class: dropdown
3 layers of 3×3, $s=1$, $d=1,2,4$:
- RF: $1\to3\to7\to15$.
- Plain 3×3 ($d=1$) ×3: RF $=7$.
- Same 27 weights per in-out channel pair in both cases.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DilatedConv2d(nn.Module):
    def __init__(self, c_in, c_out, k, d):
        super().__init__()
        self.K = nn.Parameter(torch.randn(c_out, c_in, k, k) * (c_in * k * k) ** -0.5)
        self.d = d

    def forward(self, x):
        k, d = self.K.shape[-1], self.d
        k_eff = d * (k - 1) + 1
        ## insert d-1 zeros between taps -> wider kernel, same #params
        K = self.K.new_zeros(*self.K.shape[:2], k_eff, k_eff)
        K[..., ::d, ::d] = self.K
        return F.conv2d(x, K, padding=k_eff // 2)  ## "same" size for odd k

## Example
x = torch.randn(1, 2, 9, 9)
print(DilatedConv2d(2, 4, k=3, d=2)(x).shape)  ## torch.Size([1, 4, 9, 9]), 5x5 span
```
````

```{attention} Q&A
:class: dropdown
*Dilation vs stride vs pooling for a larger RF?*
- Stride & pooling: RF ⬆️ but resolution ⬇️.
- Dilation: RF ⬆️ & resolution kept. Compute stays at full resolution.

*Cons?*
- **Gridding**: large $d$ → taps skip pixels → stacked layers sample a sparse grid → checkerboard holes.
    - Fix: mix rates that do not share a common factor (e.g., 1, 2, 3).
- Full-res feature maps → memory ⬆️.

*Other names?*
- "Atrous convolution" (DeepLab). The idea comes from the "algorithme à trous" in wavelet analysis. Yu & Koltun named it "dilated" for CNNs.
```

&nbsp;

#### Pointwise Convolution
- **What**: $1\times1$ convolution: same linear map $C_{in}\to C_{out}$ at every pixel. {cite:p}`lin2014network`
- **Why**: Mix channels & change #channels cheaply, w/o spatial mixing.
- **How**: Each pixel's channel vector $\mathbf{X}_{:,h,w}$ → $K\mathbf{X}_{:,h,w}$.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $K\in\mathbb{R}^{C_{out}\times C_{in}}$: Channel-mixing matrix.

Forward:

$$
Y_{o,h,w}=\sum_{c}K_{o,c}X_{c,h,w}
$$

- $o$: Output channel idx.

Cost: $C_{in}C_{out}HW$ MACs → $k^2$× cheaper than a $k\times k$ conv w/ the same channels.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class PointwiseConv(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.K = nn.Parameter(torch.randn(c_out, c_in) * c_in ** -0.5)

    def forward(self, x):
        ## same C_in -> C_out linear map at every (h, w); no spatial mixing
        return torch.einsum("oc,bchw->bohw", self.K, x)

## Example
x = torch.randn(2, 8, 5, 5)
print(PointwiseConv(8, 3)(x).shape)  ## torch.Size([2, 3, 5, 5])
```
````

```{attention} Q&A
:class: dropdown
*Why is it everywhere?*
- Reduce channels before an expensive $k\times k$ conv ([GoogLeNet](#googlenet), [bottleneck](#bottleneck-block)).
- Expand or project channels ([MobileNetV2](#mobilenetv2)).
- Match channels on a residual shortcut ([ResNet](#resnet)).

*Is it the same as a Linear layer?*
- ✅. = `nn.Linear` on a channels-last tensor. [ConvNeXt](#convnext) implements it that way.

*Does it add nonlinearity?*
- ❌ by itself. W/ an activation after it: stack of 1×1 + ReLU = per-pixel MLP (NIN's "mlpconv").
```

&nbsp;

#### Grouped Convolution
- **What**: Split channels into $G$ groups; convolve each group independently. {cite:p}`krizhevsky2012imagenet`
- **Why**: Params & MACs ÷ $G$.
    - Origin: AlexNet split its model across 2 GPUs w/ 3GB memory each.
- **How**:
    1. Split input channels into $G$ slices of $C_{in}/G$.
    2. Split filters into $G$ banks of $C_{out}/G$.
    3. Bank $g$ convolves slice $g$ only.
    4. Concat the $G$ outputs on channels.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{X}^{(g)}\in\mathbb{R}^{(C_{in}/G)\times H_{in}\times W_{in}}$: Input slice $g$.
- Params:
    - $\mathbf{K}^{(g)}\in\mathbb{R}^{(C_{out}/G)\times(C_{in}/G)\times k\times k}$: Filter bank $g$.

Forward:

$$
\mathbf{Y}=\text{concat}_{g=1}^{G}\ \text{Conv}\left(\mathbf{X}^{(g)};\mathbf{K}^{(g)}\right)
$$

Cost:

$$
\#\text{params}=\frac{C_{out}C_{in}k^2}{G},\qquad\#\text{MACs}=\frac{C_{out}C_{in}k^2H_{out}W_{out}}{G}
$$

- $G=1$ → standard conv.
- $G=C_{in}=C_{out}$ → depthwise conv.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupedConv2d(nn.Module):
    def __init__(self, c_in, c_out, k, G):
        super().__init__()
        assert c_in % G == 0 and c_out % G == 0
        self.K = nn.Parameter(torch.randn(c_out, c_in // G, k, k) * (c_in // G * k * k) ** -0.5)
        self.G = G

    def forward(self, x):
        xs = x.chunk(self.G, dim=1)       ## G input slices of C_in/G channels
        Ks = self.K.chunk(self.G, dim=0)  ## G filter banks of C_out/G filters
        ## group g sees only its own slice -> no cross-group mixing
        ys = [F.conv2d(xg, Kg, padding=Kg.shape[-1] // 2) for xg, Kg in zip(xs, Ks)]
        return torch.cat(ys, dim=1)

## Example
x = torch.randn(2, 8, 6, 6)
print(GroupedConv2d(8, 4, k=3, G=2)(x).shape)  ## torch.Size([2, 4, 6, 6])
```
````

```{attention} Q&A
:class: dropdown
*Cons?*
- ❌Cross-group information. Stacked grouped convs keep groups isolated.
    - Fix: [channel shuffle](#shufflenet) or a dense $1\times1$ conv after.
- MACs ÷$G$, but memory access does not shrink as much → wall-clock gain < MAC gain.

*Where is it used?*
- [ResNeXt](#resnext): cardinality = $G$ in the 3×3 conv.
- [ShuffleNet](#shufflenet): grouped $1\times1$.
- Depthwise conv: $G=C_{in}$.
```

&nbsp;

#### Depthwise Separable Convolution
- **What**: Depthwise conv (1 $k\times k$ filter per channel) → pointwise $1\times1$ conv. {cite:p}`sifre2014rigid`
- **Why**: Standard conv learns spatial & cross-channel patterns jointly → $k^2C_{in}C_{out}$ cost.
    - Factorize into spatial-only + channel-only → ~$k^2$× cheaper.
    - Assumption: spatial & channel correlations can be decoupled.
- **How**:
    1. **Depthwise**: each channel → its own $k\times k$ filter. Spatial mixing only.
    2. **Pointwise**: $1\times1$ conv. Channel mixing only.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\mathbf{K}^{dw}\in\mathbb{R}^{C_{in}\times k\times k}$: Depthwise filters.
    - $K^{pw}\in\mathbb{R}^{C_{out}\times C_{in}}$: Pointwise matrix.
- Intermediate values:
    - $\mathbf{Z}\in\mathbb{R}^{C_{in}\times H_{out}\times W_{out}}$: Depthwise output.

Forward:

$$\begin{align*}
Z_{c,h,w}&=\sum_{u,v}K^{dw}_{c,u,v}\,\tilde{X}_{c,\,sh+u,\,sw+v}\\
Y_{o,h,w}&=\sum_{c}K^{pw}_{o,c}\,Z_{c,h,w}
\end{align*}$$

- $o$: Output channel idx.
- $u,v$: Kernel offsets.

Cost ratio vs standard conv:

$$
\frac{k^2C_{in}H_{out}W_{out}+C_{in}C_{out}H_{out}W_{out}}{k^2C_{in}C_{out}H_{out}W_{out}}=\frac{1}{C_{out}}+\frac{1}{k^2}
$$
```

```{tip} Derivation
:class: dropdown
*What standard conv does it equal?*
1. Substitute $Z$ into $Y$: $Y_{o,h,w}=\sum_c\sum_{u,v}K^{pw}_{o,c}K^{dw}_{c,u,v}\,\tilde{X}_{c,\,sh+u,\,sw+v}$.
2. → Standard conv w/ kernel $K_{o,c,u,v}=K^{pw}_{o,c}\,K^{dw}_{c,u,v}$.
3. → Each $k\times k$ slice $K_{o,c,:,:}$ = a scaled copy of channel $c$'s single filter.
4. → Rank-constrained factorization: $k^2C_{in}+C_{in}C_{out}$ free params instead of $k^2C_{in}C_{out}$.
```

```{note} Example
:class: dropdown
$k=3$, $C_{out}=256$: ratio $=1/256+1/9\approx0.115$ → ~8.7× fewer MACs.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

## Reuses GroupedConv2d & PointwiseConv from above
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, c_in, c_out, k=3):
        super().__init__()
        self.dw = GroupedConv2d(c_in, c_in, k, G=c_in)  ## G = C_in -> 1 filter per channel
        self.pw = PointwiseConv(c_in, c_out)            ## all channel mixing happens here

    def forward(self, x):
        return self.pw(self.dw(x))

## Example
x = torch.randn(2, 8, 6, 6)
print(DepthwiseSeparableConv(8, 16)(x).shape)  ## torch.Size([2, 16, 6, 6])
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- ~$k^2$× fewer MACs & params.
- Core of [MobileNet](#mobilenet), [EfficientNet](#efficientnet), [ConvNeXt](#convnext).

*Cons?*
- Less capacity per layer → needs more width or depth.
- Depthwise conv has low arithmetic intensity → memory-bound on GPUs → speedup < MAC reduction.

*How does it relate to Inception?*
- Inception splits channels into a few branches. Push it to 1 branch per channel → depthwise separable conv.
- Xception ("Extreme Inception") replaces Inception modules w/ depthwise separable convs.

*Origin?*
- Sifre developed it in 2013. His 2014 thesis reports the detailed results.
```

&nbsp;

#### Transposed Convolution
- **What**: Adjoint (transpose) of a convolution: learnable upsampling.
- **Why**: Decoders (segmentation, GANs, autoencoders) must upsample w/ learned weights.
    - Also called "deconvolution". Misnomer: ❌inverse of conv.
- **How**:
    1. Each input pixel scales the kernel → one $k\times k\times C_{out}$ patch.
    2. Place the patch at $(sh,sw)$ in the output.
    3. Sum overlapping patches.
    4. Crop $p$ from each border.
    - Equivalent: insert $s-1$ zeros between input pixels → pad by $k-1-p$ → stride-1 conv w/ the flipped kernel.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\mathbf{K}\in\mathbb{R}^{C_{in}\times C_{out}\times k\times k}$: Filters (PyTorch layout).
- Intermediate values:
    - $\tilde{\mathbf{Y}}$: Output before cropping.

Forward:

$$
\tilde{Y}_{o,h',w'}=\sum_{c}\sum_{\substack{h,w,u,v:\\ sh+u=h',\ sw+v=w'}}X_{c,h,w}\,K_{c,o,u,v}
$$

- $o$: Output channel idx.
- $h',w'$: Uncropped output idx.
- $\mathbf{Y}$: $\tilde{\mathbf{Y}}$ w/ $p$ rows & columns cropped from each border (+ bias).

Output size:

$$
H_{out}=(H_{in}-1)s-2p+k
$$

- Same form as the conv input-grad formula ← transposed conv = backward pass of conv w.r.t. its input.
- Conv w/ matrix $M$ maps $\mathbb{R}^a\to\mathbb{R}^b$. Transposed conv uses $M^T$: $\mathbb{R}^b\to\mathbb{R}^a$. It restores the shape, ❌the values.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvTranspose2d(nn.Module):
    def __init__(self, c_in, c_out, k, s=1, p=0):
        super().__init__()
        self.K = nn.Parameter(torch.randn(c_in, c_out, k, k) * (c_in * k * k) ** -0.5)
        self.b = nn.Parameter(torch.zeros(c_out))
        self.k, self.s, self.p = k, s, p

    def forward(self, x):
        B, _, H, W = x.shape
        k, s, p = self.k, self.s, self.p
        ## each input pixel paints a (C_out, k, k) patch scaled by its values
        cols = torch.einsum("bchw,coij->boijhw", x, self.K)  ## (B, C_out, k, k, H, W)
        cols = cols.reshape(B, -1, H * W)                   ## (B, C_out*k*k, H*W)
        Ho, Wo = (H - 1) * s + k, (W - 1) * s + k           ## size before cropping
        y = F.fold(cols, (Ho, Wo), k, stride=s)             ## overlap-add patches at (s*h, s*w)
        return y[..., p:Ho - p, p:Wo - p] + self.b[:, None, None]  ## crop p per border, add bias

## Example
x = torch.randn(1, 4, 4, 4)
print(ConvTranspose2d(4, 2, k=4, s=2, p=1)(x).shape)  ## torch.Size([1, 2, 8, 8]): 2x upsample
```
````

```{attention} Q&A
:class: dropdown
*Why checkerboard artifacts?*
- $k$ not divisible by $s$ → output pixels get unequal #overlapping patches → periodic grid pattern {cite:p}`odena2016deconvolution`.
- Fixes:
    - Pick $k$ divisible by $s$ (e.g., $k=4$, $s=2$). Reduces the risk, does not remove it.
    - Resize (nearest or bilinear) → regular conv.

*Is it the inverse of conv?*
- ❌. It inverts the shape map only. $M^TM\ne I$ in general.

*Why `output_padding`?*
- Conv floors the output size → several $H_{in}$ map to the same $H_{out}$.
- → Transposed conv's output size is ambiguous. `output_padding` picks one.
```

&nbsp;

### Pooling
- **What**: Fixed per-channel summary (max or mean) of each $k\times k$ window.
- **Why**: Downsample → compute ⬇️ & RF ⬆️.
    - Max ignores where in the window the peak sits → partial local invariance.
    - ❌Guaranteed: a shift across a window boundary can change the output.
- **How**: Slide the window w/ stride $s$ (usually $k=s=2$) over each channel separately → take max or mean. Channels never mix.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $u,v$: Window offsets.
    - $\mathbf{G}=\partial\mathcal{L}/\partial\mathbf{Y}$: Upstream gradient.
    - $\mathcal{W}(h,w)$: Input positions in the window of output $(h,w)$.

Forward ($p=0$):

$$\begin{align*}
\text{Max: }Y_{c,h,w}&=\max_{0\le u,v<k}X_{c,\,sh+u,\,sw+v}\\
\text{Avg: }Y_{c,h,w}&=\frac{1}{k^2}\sum_{u=0}^{k-1}\sum_{v=0}^{k-1}X_{c,\,sh+u,\,sw+v}
\end{align*}$$

Output size: $H_{out}=\lfloor(H_{in}-k)/s\rfloor+1$. $C_{out}=C_{in}$.

Backward:

$$\begin{align*}
\text{Max: }\frac{\partial\mathcal{L}}{\partial X_{c,h',w'}}&=\sum_{h,w}G_{c,h,w}\,\mathbb{1}\left[(h',w')=\arg\max_{\mathcal{W}(h,w)}X_{c,\cdot,\cdot}\right]\\
\text{Avg: }\frac{\partial\mathcal{L}}{\partial X_{c,h',w'}}&=\frac{1}{k^2}\sum_{(h,w):\,(h',w')\in\mathcal{W}(h,w)}G_{c,h,w}
\end{align*}$$

- $h',w'$: Input idx.
- Max: grad goes only to the argmax of each window (subgradient of max).
- Avg: grad spreads equally over the window.
- Overlapping windows ($s<k$) → contributions sum.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class Pool2d(nn.Module):
    def __init__(self, k, s=None, mode="max"):
        super().__init__()
        self.k, self.s, self.mode = k, s or k, mode

    def forward(self, x):
        ## (B, C, Ho, Wo, k, k): windows taken per channel -> channels never mix
        win = x.unfold(2, self.k, self.s).unfold(3, self.k, self.s)
        return win.amax(dim=(-2, -1)) if self.mode == "max" else win.mean(dim=(-2, -1))

## Example
x = torch.randn(2, 3, 8, 8)
print(Pool2d(2)(x).shape)  ## torch.Size([2, 3, 4, 4])
```
````

```{attention} Q&A
:class: dropdown
*Max vs Avg?*
- **Max**: keeps the strongest activation → "is the feature present anywhere here?" Sparse grads.
- **Avg**: keeps the mean → smoother, keeps weak signals, dilutes strong ones. Dense grads.

*Pros?*
- 0 params.
- Compute ⬇️ for later layers.
- RF ⬆️ fast.

*Cons?*
- Discards spatial detail → hurts dense prediction.
- Fixed, not learned.
- Subsampling w/o low-pass filter → aliasing.

*Pooling or strided conv?*
- Strided conv learns its downsampling. An all-conv net matches pooled nets {cite:p}`springenberg2015striving`.
- Modern backbones mostly use strided conv. ResNet keeps 1 max pool in the stem.
```

&nbsp;

#### Global Average Pooling
- **What**: Mean of each channel over all $H\times W$ positions. {cite:p}`lin2014network`
- **Why**: Flatten + FC head → huge params & a fixed input size.
    - VGG-16: ~124M of its ~138M params are in the FC layers.
- **How**: $\mathbf{X}\in\mathbb{R}^{C\times H\times W}$ → mean over $h,w$ → $\mathbf{z}\in\mathbb{R}^{C}$ → linear classifier.

```{note} Math
:class: dropdown
Forward:

$$
z_c=\frac{1}{HW}\sum_{h=0}^{H-1}\sum_{w=0}^{W-1}X_{c,h,w}
$$

- $\mathbf{z}\in\mathbb{R}^{C}$: Pooled feature vector.
```

```{attention} Q&A
:class: dropdown
*Pros?*
- 0 params → less overfitting than an FC head.
- Any input size → same output size.

*Cons?*
- Discards spatial layout → ❌ for tasks that need position.

*What does it enable?*
- **CAM** (class activation map, Zhou et al. 2016):
    1. Logit: $\text{logit}_y=\sum_c a_{y,c}z_c=\frac{1}{HW}\sum_{h,w}\sum_c a_{y,c}X_{c,h,w}$.
    2. → Map $M_y(h,w)=\sum_c a_{y,c}X_{c,h,w}$ shows where class $y$'s evidence is.
    - $a_{y,c}$: Classifier weight of class $y$ on channel $c$.
- Works because GAP & the linear head commute.
```

&nbsp;

### Receptive Field
- **What**: Input region that can affect one output unit.
- **Why**: A unit can only use its RF → RF must cover the object or context the task needs.
- **How**: Each layer grows the RF by $(k_\text{eff}-1)$ × (product of all earlier strides).
    - → Stride compounds: layers after downsampling grow the RF faster.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $l$: Layer idx.
    - $r_l$: RF size of a unit at layer $l$ (in input pixels).
    - $j_l$: Jump: input-pixel distance between adjacent units at layer $l$.

Process:

$$\begin{align*}
r_0&=1,\qquad j_0=1\\
r_l&=r_{l-1}+\left(k_l^\text{eff}-1\right)j_{l-1}\\
j_l&=j_{l-1}\,s_l
\end{align*}$$

- $k_l^\text{eff}=d_l(k_l-1)+1$: Effective kernel size of layer $l$.
- $L$ layers of 3×3, $s=1$: $r_L=2L+1$.
```

```{tip} Derivation
:class: dropdown
*Why this recurrence?*
1. A unit at layer $l$ reads $k_l^\text{eff}$ adjacent positions of layer $l-1$ per axis.
2. Adjacent layer-$(l-1)$ units are $j_{l-1}$ input pixels apart.
3. → The window spans $(k_l^\text{eff}-1)j_{l-1}$ extra input pixels beyond one layer-$(l-1)$ RF.
4. Stride $s_l$ → adjacent layer-$l$ units are $s_l$ layer-$(l-1)$ positions apart → $j_l=j_{l-1}s_l$.
```

````{important} Code
:class: dropdown
```python
class ReceptiveField:
    def __call__(self, layers):
        ## layers: list of (k, s, d)
        r, j = 1, 1  ## RF size, jump
        for k, s, d in layers:
            r += d * (k - 1) * j  ## grows by effective kernel span x current jump
            j *= s                ## stride compounds for all later layers
        return r, j

## Example
rf = ReceptiveField()
print(rf([(3, 1, 1)] * 3))                     ## (7, 1): three 3x3 = one 7x7
print(rf([(7, 2, 1), (3, 2, 1), (3, 1, 1)]))   ## (19, 4): ResNet-like stem + 1 conv
```
````

```{attention} Q&A
:class: dropdown
*Theoretical vs effective RF?*
- Center pixels reach the output through many more paths than border pixels.
- → Effective RF ≈ Gaussian. It fills only a fraction of the theoretical RF {cite:p}`luo2016understanding`.
- In that paper's simplified analysis, effective RF grows ~$O(\sqrt{L})$ w/ depth $L$, theoretical RF grows $O(L)$.

*How to grow the RF?*
- Depth: linear growth.
- Stride or pooling: multiplies later growth.
- Dilation: growth w/o resolution loss.
- Larger kernels ([ConvNeXt](#convnext): 7×7).
- Attention: global in 1 layer.

*Theoretical RF > image size → global context?*
- ❌. Effective RF can still be much smaller.
```

&nbsp;

## Classic CNNs

&nbsp;

### LeNet-5
- **What**: 7-layer CNN (conv → subsample → conv → subsample → FC) for 32×32 handwritten characters. {cite:p}`lecun1998gradient`
- **Why**: Hand-crafted features + classifier → brittle & labor-intensive.
    - Learn features end-to-end from pixels w/ backprop.
- **How**:
    1. C1: 6 maps, 5×5 conv → 28×28.
    2. S2: 2×2 subsample → 14×14.
    3. C3: 16 maps, 5×5 conv → 10×10. Each map reads a subset of S2 maps.
    4. S4: 2×2 subsample → 5×5.
    5. C5: 120 maps, 5×5 conv → 1×1.
    6. F6: 84 units → 10 RBF output units.
    - Scaled tanh activations. Trainable subsampling: sum × coeff + bias → sigmoid.

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class LeNet5(nn.Module):
    ## modern simplification: avg pool, full C3, linear output (paper: trainable subsampling, partial C3, RBF)
    def __init__(self, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5), nn.Tanh(), nn.AvgPool2d(2),   ## 32 -> 28 -> 14
            nn.Conv2d(6, 16, 5), nn.Tanh(), nn.AvgPool2d(2),  ## 14 -> 10 -> 5
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120), nn.Tanh(),            ## = C5 (5x5 conv on a 5x5 map)
            nn.Linear(120, 84), nn.Tanh(),
            nn.Linear(84, n_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))

## Example
print(LeNet5()(torch.randn(2, 1, 32, 32)).shape)  ## torch.Size([2, 10]), ~61.7K params
```
````

```{attention} Q&A
:class: dropdown
*What carried over to modern CNNs?*
- Conv → downsample → FC template.
- Weight sharing & local receptive fields.
- End-to-end gradient training.

*What did not?*
- tanh → ReLU.
- Trainable average subsampling → max pool or strided conv.
- Partial C3 connectivity → full or grouped.
- RBF output → softmax.

*Why did CNNs stall until 2012?*
- Data: ❌large labeled image sets until ImageNet.
- Compute: ❌GPU training.
- Training tricks: ReLU, dropout, good init.
```

&nbsp;

### AlexNet
- **What**: 8-layer CNN (5 conv + 3 FC) w/ ReLU, dropout, & 2-GPU training. Won ILSVRC-2012. {cite:p}`krizhevsky2012imagenet`
- **Why**: Show that a large CNN trained on large data beats hand-crafted pipelines.
    - ImageNet: 1.2M training images, 1000 classes.
    - Top-5 test error: 15.3% (ensemble) vs 26.2% for the 2nd-place entry.
- **How**:
    1. Conv1: 96 filters, 11×11, stride 4 → conv2: 5×5 → conv3–5: 3×3.
    2. Overlapping max pool (3×3, $s=2$). Local response normalization (LRN) after conv1 & conv2.
    3. FC 4096 → FC 4096 → 1000-way softmax. Dropout 0.5 on the first 2 FCs.
    4. Train: SGD w/ momentum 0.9, weight decay 5e-4, batch 128. Augment w/ random crops, flips, & PCA color jitter.
    5. Split channels across 2 GTX 580 (3GB) GPUs. Some layers read only the same-GPU half → [grouped conv](#grouped-convolution).

```{attention} Q&A
:class: dropdown
*Which ideas mattered most?*
- ReLU: non-saturating → reaches 25% CIFAR-10 training error 6× faster than tanh.
- Dropout: controls overfitting of ~60M params.
- Data augmentation.
- GPU training: 5–6 days on 2 GPUs.

*What did not last?*
- LRN: VGG found no gain on ILSVRC {cite:p}`simonyan2015very`. BatchNorm replaced it later.
- Large 11×11 & 5×5 kernels → stacks of 3×3.
- Huge FC layers → GAP.

*224 or 227?*
- Paper states 224×224 input & 55×55 conv1 output.
- $(224-11)/4+1$ is not an integer. 55 needs 227, or padding 2.
```

&nbsp;

### VGG
- **Name**: Visual Geometry Group (Oxford) network {cite:p}`simonyan2015very`
- **What**: Deep plain CNN (16–19 layers) of only 3×3 convs & 2×2 max pools.
- **Why**: Test depth as the main design axis.
    - Uniform 3×3 kernels → depth ⬆️ at the same RF & fewer params.
- **How**:
    1. 5 stages. Each: [3×3 conv, $s=1$, $p=1$, ReLU] × {2, 2, 3, 3, 3} (VGG-16) → 2×2 max pool, $s=2$.
    2. Channels: 64 → 128 → 256 → 512 → 512.
    3. Head: FC 4096 → FC 4096 → FC 1000.
    - ILSVRC-2014: 2nd in classification, 1st in localization.

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class VGG(nn.Module):
    CFG16 = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]

    def __init__(self, cfg, n_classes=1000):
        super().__init__()
        layers, c = [], 3
        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(2))                         ## halve H, W
            else:
                layers += [nn.Conv2d(c, v, 3, padding=1), nn.ReLU()]  ## 3x3 only, size-preserving
                c = v
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(7), nn.Flatten(),                   ## 7x7 at 224 input
            nn.Linear(c * 7 * 7, 4096), nn.ReLU(), nn.Dropout(),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(),
            nn.Linear(4096, n_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))

## Example: tiny config; VGG(VGG.CFG16) is VGG-16
print(VGG([8, "M", 16, "M"], n_classes=10)(torch.randn(2, 3, 32, 32)).shape)  ## torch.Size([2, 10])
```
````

```{attention} Q&A
:class: dropdown
*Why stack 3×3 instead of one 7×7?*
- 3 stacked 3×3 ($s=1$) → RF 7. Same as one 7×7.
- Params ($C$ in & out): $3\cdot9C^2=27C^2$ vs $49C^2$ → 45% fewer.
- 3 ReLUs vs 1 → more nonlinear, more discriminative.

*Cons?*
- ~138M params. ~124M are in the FC head.
- Heavy compute: VGG-19 ≈ 19.6G vs ResNet-152 ≈ 11.3G multiply-adds (the ResNet paper calls them "FLOPs").
- Plain stacking stops helping at ~20 layers → [ResNet](#resnet).

*How were the deep configs trained?*
- Paper: init deeper nets from a trained shallow net.
- After submission, the authors found that Glorot init alone works.

*Still used?*
- As a fixed feature extractor: perceptual & style losses.
```

&nbsp;

### GoogLeNet
- **What**: 22-layer CNN of stacked Inception modules: parallel multi-scale conv & pool branches. {cite:p}`szegedy2015going`
- **Why**: Objects appear at many scales → no single best kernel size → use several in parallel.
    - Naive parallel 3×3 & 5×5 on wide inputs → compute explodes → reduce channels first w/ 1×1.
- **How**:
    1. **Inception module**: 4 branches on the same input: 1×1 | 1×1 → 3×3 | 1×1 → 5×5 | 3×3 max pool → 1×1. Concat on channels.
    2. 1×1 "reduce" before 3×3 & 5×5 → channels ⬇️ → MACs ⬇️.
    3. Stack 9 modules, w/ max pools between groups. GAP → dropout 0.4 → linear.
    4. 2 auxiliary classifiers on middle layers. Loss weight 0.3. Removed at inference.
    - ILSVRC-2014 classification winner: top-5 6.67%. Budget ~1.5B multiply-adds per inference.

```{note} Example
:class: dropdown
5×5 branch of inception (3a): input $28\times28\times192$ → 32 maps.
- Direct: $25\cdot192\cdot32\cdot784\approx120.4$M MACs.
- 1×1 reduce to 16 first: $192\cdot16\cdot784+25\cdot16\cdot32\cdot784\approx2.4\text{M}+10.0\text{M}=12.4$M MACs.
- → ~10× fewer.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class InceptionModule(nn.Module):
    def __init__(self, c_in, c1, c3r, c3, c5r, c5, cp):
        super().__init__()
        conv = lambda i, o, k: nn.Sequential(nn.Conv2d(i, o, k, padding=k // 2), nn.ReLU())
        self.b1 = conv(c_in, c1, 1)
        self.b3 = nn.Sequential(conv(c_in, c3r, 1), conv(c3r, c3, 3))  ## 1x1 reduce -> cheap 3x3
        self.b5 = nn.Sequential(conv(c_in, c5r, 1), conv(c5r, c5, 5))  ## 1x1 reduce -> cheap 5x5
        self.bp = nn.Sequential(nn.MaxPool2d(3, 1, 1), conv(c_in, cp, 1))

    def forward(self, x):
        ## parallel multi-scale branches, same H, W -> concat on channels
        return torch.cat([self.b1(x), self.b3(x), self.b5(x), self.bp(x)], dim=1)

## Example: inception (3a)
x = torch.randn(1, 192, 28, 28)
print(InceptionModule(192, 64, 96, 128, 16, 32, 32)(x).shape)  ## torch.Size([1, 256, 28, 28])
```
````

```{attention} Q&A
:class: dropdown
*Why auxiliary classifiers?*
- Inject grads into middle layers & regularize.
- Inception-v3 found they act mainly as regularizers, w/ little effect on early convergence {cite:p}`szegedy2016rethinking`.

*Pros?*
- Far fewer params than AlexNet (~5–7M vs ~60M) at much better accuracy.
- Multi-scale features per layer.

*Cons?*
- Hand-designed: many per-module hyperparams → hard to adapt or scale.
- Multi-branch → fragmented ops → lower GPU efficiency.

*What came after?*
- Inception-v2/v3 (same paper as above): BatchNorm, 5×5 → two 3×3, $n\times n$ → $1\times n$ + $n\times1$, label smoothing.
- Inception-v4 & Inception-ResNet: add residual shortcuts.
- Xception: branches → [depthwise separable conv](#depthwise-separable-convolution).
```

&nbsp;

### ResNet
- **What**: CNN of stacked residual blocks $\mathbf{y}=\text{ReLU}(F(\mathbf{x})+\mathbf{x})$; trains 100+ layers. {cite:p}`he2016deep`
- **Why**: **Degradation**: deeper plain nets get higher *training* error.
    - ❌Overfitting: training error is worse, not only test error.
    - ❌Vanishing grads alone: plain nets w/ BatchNorm still degrade.
    - Hypothesis: stacked nonlinear layers struggle to learn the identity. Learn $F=H-\mathbf{x}$ instead → identity = push $F\to0$ → easy.
- **How**:
    1. Stem: 7×7 conv, $s=2$, 64 ch → 3×3 max pool, $s=2$ → $H/4$.
    2. 4 stages, inner widths 64, 128, 256, 512. Block output = width × expansion (basic: ×1; [bottleneck](#bottleneck-block): ×4 → 256…2048). First block of stages 2–4 uses $s=2$ & doubles the width.
    3. Basic block: conv-BN-ReLU-conv-BN → + shortcut → ReLU.
    4. Shortcut: identity. If the shape changes: 1×1 conv ($s=2$) + BN (projection).
    5. GAP → FC.
    - Generic residual connection → [Residual Connection](../dl/foundation/module.md#residual-connection).
    - ILSVRC-2015 classification winner: top-5 3.57% (ensemble).

```{note} Math
:class: dropdown
Notations:
- Params:
    - $W_s$: Shortcut map: identity, or 1×1 conv + BN when the shape changes.

Model:

$$
\mathbf{y}=\text{ReLU}\left(F(\mathbf{x})+W_s\mathbf{x}\right),\qquad F=\text{BN}\circ\text{Conv}_{3\times3}\circ\text{ReLU}\circ\text{BN}\circ\text{Conv}_{3\times3}
$$

Depth configs (blocks per stage):
- ResNet-18: basic, [2, 2, 2, 2].
- ResNet-34: basic, [3, 4, 6, 3].
- ResNet-50: [bottleneck](#bottleneck-block), [3, 4, 6, 3].
- ResNet-101: bottleneck, [3, 4, 23, 3].
- ResNet-152: bottleneck, [3, 8, 36, 3].
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, c_in, c, s=1):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(c_in, c, 3, s, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(),
            nn.Conv2d(c, c, 3, 1, 1, bias=False), nn.BatchNorm2d(c),
        )
        ## projection only when the shape changes; identity otherwise
        self.short = nn.Identity() if s == 1 and c_in == c else nn.Sequential(
            nn.Conv2d(c_in, c, 1, s, bias=False), nn.BatchNorm2d(c))

    def forward(self, x):
        return F.relu(self.f(x) + self.short(x))  ## ReLU after the add (original ResNet)

class ResNet(nn.Module):
    def __init__(self, block, layers, n_classes=1000):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(3, 2, 1))                                  ## H -> H/4
        blocks, c_in = [], 64
        for i, n in enumerate(layers):
            c = 64 * 2 ** i
            for b in range(n):
                s = 2 if i > 0 and b == 0 else 1                    ## downsample at stage entry (not stage 1)
                blocks.append(block(c_in, c, s))
                c_in = c * block.expansion
        self.stages = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c_in, n_classes))

    def forward(self, x):
        return self.head(self.stages(self.stem(x)))

## Example: ResNet-18
print(ResNet(BasicBlock, [2, 2, 2, 2], 10)(torch.randn(2, 3, 64, 64)).shape)  ## torch.Size([2, 10])
```
````

```{attention} Q&A
:class: dropdown
*Why does it work?*
- Default behavior = identity → adding a block cannot easily hurt.
- Grad path: $\frac{\partial\mathcal{L}}{\partial\mathbf{x}}$ has a direct term through the shortcut → reaches early layers.
- Original block has ReLU after the add → path not exactly identity → [pre-activation](#pre-activation-resnet) fixes this.

*Degradation = overfitting?*
- ❌. Deeper plain nets have higher *training* error. Overfitting = low train, high test error.

*Identity or projection shortcut?*
- Paper options: (A) zero-pad when channels grow, (B) projection only when the shape changes, (C) projection everywhere.
- C is slightly best. B is the default: near-equal accuracy, fewer params.

*Training trick?*
- Init $\gamma=0$ in the last BN of each block → each block starts as identity → helps large-batch training {cite:p}`goyal2017accurate`.

*Pros?*
- Trains 100+ layers. Simple, uniform design.
- ResNet-50 (3.8G multiply-adds) is a default backbone.

*Cons?*
- Many blocks contribute little in very deep variants → [Wide ResNet](#wide-resnet).
```

&nbsp;

#### Bottleneck Block
- **What**: Residual block 1×1 (reduce to $C$) → 3×3 ($C$) → 1×1 (expand to $4C$). {cite:p}`he2016deep`
- **Why**: 3×3-only blocks at 256+ channels → too expensive for 50+ layers.
- **How**:
    1. 1×1 shrinks channels 4×.
    2. 3×3 runs on the narrow tensor.
    3. 1×1 restores width → add shortcut → ReLU.

```{note} Example
:class: dropdown
Params per block (in & out width shown):
- Basic, 64-ch: $2\cdot9\cdot64^2=73{,}728$.
- Bottleneck, 256-ch ($256\to64\to64\to256$): $256\cdot64+9\cdot64^2+64\cdot256=69{,}632$.
- → Same cost, 4× wider in/out.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

## Reuses ResNet from above
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, c_in, c, s=1):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(c_in, c, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(),            ## reduce
            nn.Conv2d(c, c, 3, s, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(),         ## stride here = "v1.5"
            nn.Conv2d(c, 4 * c, 1, bias=False), nn.BatchNorm2d(4 * c),                  ## expand
        )
        self.short = nn.Identity() if s == 1 and c_in == 4 * c else nn.Sequential(
            nn.Conv2d(c_in, 4 * c, 1, s, bias=False), nn.BatchNorm2d(4 * c))

    def forward(self, x):
        return F.relu(self.f(x) + self.short(x))

## Example: ResNet-50
print(ResNet(Bottleneck, [3, 4, 6, 3], 10)(torch.randn(2, 3, 64, 64)).shape)  ## torch.Size([2, 10])
```
````

```{attention} Q&A
:class: dropdown
*Where does the stride go?*
- Original paper: first 1×1 → the 1×1 skips 3/4 of positions.
- torchvision "v1.5": 3×3 → slightly ⬆️accuracy, slightly ⬆️FLOPs.

*How does it relate to MobileNetV2?*
- Bottleneck: wide → narrow → wide. Shortcut joins the wide ends.
- [Inverted residual](#mobilenetv2): narrow → wide → narrow. Shortcut joins the narrow ends.
```

&nbsp;

#### Pre-activation ResNet
- **What**: Residual block w/ BN → ReLU → conv order & nothing after the add. {cite:p}`he2016identity`
- **Why**: Original block: ReLU after the add → shortcut is not a pure identity → signal & grads distort over 100s of blocks.
- **How**: $\mathbf{x}_{l+1}=\mathbf{x}_l+F(\mathbf{x}_l)$, w/ $F$ = BN → ReLU → conv → BN → ReLU → conv. Add a final BN + ReLU before GAP.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\mathbf{x}_l$: Input of block $l$.
    - $L$: Idx of any deeper block.

Forward (unrolled):

$$
\mathbf{x}_L=\mathbf{x}_l+\sum_{i=l}^{L-1}F(\mathbf{x}_i)
$$

Backward:

$$
\frac{\partial\mathcal{L}}{\partial\mathbf{x}_l}=\frac{\partial\mathcal{L}}{\partial\mathbf{x}_L}\left(I+\frac{\partial}{\partial\mathbf{x}_l}\sum_{i=l}^{L-1}F(\mathbf{x}_i)\right)
$$

- The $\frac{\partial\mathcal{L}}{\partial\mathbf{x}_L}\cdot I$ term reaches every earlier block directly.
- The second term is unlikely to be exactly $-I$ for all samples → grad rarely cancels.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class PreActBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.f = nn.Sequential(
            nn.BatchNorm2d(c), nn.ReLU(), nn.Conv2d(c, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(), nn.Conv2d(c, c, 3, padding=1, bias=False),
        )

    def forward(self, x):
        return x + self.f(x)  ## nothing after the add -> clean identity path

## Example
print(PreActBlock(8)(torch.randn(2, 8, 6, 6)).shape)  ## torch.Size([2, 8, 6, 6])
```
````

```{attention} Q&A
:class: dropdown
*Why not scale the shortcut (e.g., $0.5\mathbf{x}$)?*
- Scale $\lambda$ per block → $\prod\lambda_i$ across blocks → grads explode ($\lambda>1$) or vanish ($\lambda<1$).
- Gating & dropout on the shortcut also hurt in the paper.

*Pros?*
- Trains 1000+ layer ResNets. Better accuracy than post-activation at large depth.
- BN on each block's input → regularization.

*How does it relate to Transformers?*
- Same idea as Pre-LN: norm inside the branch, untouched residual stream → see [Add & Norm](../nlp/transformer.md#add-norm).
```

&nbsp;

#### Wide ResNet
- **What**: ResNet w/ fewer layers & $\omega\times$ more channels per block. {cite:p}`zagoruyko2016wide`
- **Why**: Very deep, thin ResNets → diminishing returns. Many blocks learn little. Depth is sequential → slow.
- **How**:
    1. Multiply each basic block's channels by widening factor $\omega$. Name: WRN-$n$-$\omega$ ($n$ = depth).
    2. Optional dropout between the 2 convs of a block.
    - $\omega$ = the paper's $k$.

```{attention} Q&A
:class: dropdown
*Pros?*
- A 16-layer WRN beats 1000-layer thin ResNets on CIFAR in accuracy & speed.
- Width parallelizes on GPUs better than depth.

*Cons?*
- Params $\propto\omega^2$ → memory ⬆️.

*Depth or width?*
- Both help. Past a point, width gives more accuracy per unit of training time.
- Used in DDPM's U-Net blocks → [U-Net](../dl/gen/diffusion.md#u-net).
```

&nbsp;

#### ResNeXt
- **What**: Bottleneck ResNet whose 3×3 conv is split into $G$ parallel paths (cardinality). {cite:p}`xie2017aggregated`
- **Why**: Add a design axis beyond depth & width.
    - Cardinality ⬆️ at fixed FLOPs → accuracy ⬆️ more than depth ⬆️ or width ⬆️.
- **How**:
    1. Split-transform-merge (Inception idea) w/ $G$ identical paths.
    2. Implement as 1×1 → 3×3 [grouped conv](#grouped-convolution) ($G$ groups) → 1×1.
    - ResNeXt-50 (32×4d): $G=32$, 4 channels per group in stage 1 → 1×1 128 → 3×3 grouped 128 → 1×1 256.
    - $G$ = the paper's $C$.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\mathcal{T}_g$: Path $g$: 1×1 → 3×3 → 1×1, same topology for all $g$.

Model:

$$
\mathbf{y}=\text{ReLU}\left(\mathbf{x}+\sum_{g=1}^{G}\mathcal{T}_g(\mathbf{x})\right)
$$
```

```{tip} Derivation
:class: dropdown
*Why is the sum of paths a grouped conv?*
1. Each path's last 1×1 maps its $C/G$ channels to the output. Summing $G$ such maps = one 1×1 over the concat of all paths.
2. Each path's first 1×1 reads the same input. $G$ such maps = one 1×1 w/ $C$ outputs, split into $G$ slices.
3. The middle 3×3 of path $g$ reads only slice $g$ → grouped conv w/ $G$ groups.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNeXtBlock(nn.Module):
    def __init__(self, c_in, c_mid, c_out, G=32):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(c_in, c_mid, 1, bias=False), nn.BatchNorm2d(c_mid), nn.ReLU(),
            ## G parallel paths == grouped conv (see GroupedConv2d above)
            nn.Conv2d(c_mid, c_mid, 3, padding=1, groups=G, bias=False), nn.BatchNorm2d(c_mid), nn.ReLU(),
            nn.Conv2d(c_mid, c_out, 1, bias=False), nn.BatchNorm2d(c_out),
        )

    def forward(self, x):
        return F.relu(x + self.f(x))

## Example: ResNeXt-50 (32x4d) stage-1 block
print(ResNeXtBlock(256, 128, 256)(torch.randn(1, 256, 8, 8)).shape)  ## torch.Size([1, 256, 8, 8])
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- Better accuracy than ResNet at equal FLOPs & params.
- Uniform paths → 2 hyperparams (cardinality, width per group), unlike Inception.

*Cons?*
- Grouped conv runs less efficiently on GPUs than dense conv at equal FLOPs.
```

&nbsp;

### DenseNet
- **What**: CNN where each layer in a block takes the concat of all earlier layers' outputs. {cite:p}`huang2017densely`
- **Why**: Feature reuse & short grad paths w/ fewer params.
    - ResNet sums → features merge. Concat keeps them separate & reusable.
- **How**:
    1. **Dense block**: layer $l$ input = concat of $\mathbf{x}_0,\dots,\mathbf{x}_{l-1}$ → BN-ReLU-1×1 ($4\kappa$) → BN-ReLU-3×3 ($\kappa$) → append $\kappa$ new maps.
    2. **Transition** between blocks: BN → 1×1 conv (keep $\theta$ of the channels) → 2×2 avg pool.
    - ImageNet models: $\kappa=32$, $\theta=0.5$ (DenseNet-BC).
    - $\kappa$ = the paper's growth rate $k$.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\kappa$: Growth rate: #new maps per layer.
    - $\theta\in(0,1]$: Transition compression.
- Misc:
    - $\kappa_0$: #channels entering the block.
    - $l\ge1$: Layer idx within the block. $\mathbf{x}_0$ = block input.
    - $H_l$: Layer $l$'s BN-ReLU-conv composite.

Forward:

$$
\mathbf{x}_l=H_l\left([\mathbf{x}_0,\mathbf{x}_1,\dots,\mathbf{x}_{l-1}]\right)
$$

- $[\cdot]$: Channel concat.
- Input channels of layer $l$: $\kappa_0+\kappa(l-1)$.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class DenseLayer(nn.Module):
    def __init__(self, c_in, kappa):
        super().__init__()
        self.f = nn.Sequential(
            nn.BatchNorm2d(c_in), nn.ReLU(), nn.Conv2d(c_in, 4 * kappa, 1, bias=False),         ## bottleneck
            nn.BatchNorm2d(4 * kappa), nn.ReLU(), nn.Conv2d(4 * kappa, kappa, 3, padding=1, bias=False),
        )

    def forward(self, x):
        return self.f(x)

class DenseBlock(nn.Module):
    def __init__(self, c0, kappa, n_layers):
        super().__init__()
        self.layers = nn.ModuleList(DenseLayer(c0 + i * kappa, kappa) for i in range(n_layers))

    def forward(self, x):
        for layer in self.layers:
            x = torch.cat([x, layer(x)], dim=1)  ## append kappa new maps; keep all old ones
        return x

## Example
print(DenseBlock(16, kappa=8, n_layers=4)(torch.randn(2, 16, 6, 6)).shape)  ## torch.Size([2, 48, 6, 6])
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- Param-efficient: DenseNet-121 ≈ 8.0M params.
- Every layer gets a direct grad from the loss → implicit deep supervision.
- Small $\kappa$ suffices ← each layer adds little & reuses the rest.

*Cons?*
- Memory-heavy: naive concat copies all earlier maps per layer.
- Many small, narrow convs → slower on GPUs than FLOPs suggest.

*Sum or concat?*
- Sum (ResNet): fixed width, features blend.
- Concat (DenseNet): width grows linearly, features stay separate.
```

&nbsp;

### SENet
- **Name**: Squeeze-and-Excitation Network {cite:p}`hu2018squeeze`
- **What**: Block that rescales each channel by a gate computed from global context.
- **Why**: Conv mixes channels w/ fixed weights & a local view → ❌input-dependent channel importance.
- **How**:
    1. **Squeeze**: GAP → $\mathbf{z}\in\mathbb{R}^{C}$.
    2. **Excitation**: FC ($C\to C/r$) → ReLU → FC ($C/r\to C$) → sigmoid → gates $\mathbf{a}\in(0,1)^C$.
    3. **Scale**: multiply channel $c$ by $a_c$.
    4. Insert in each residual branch, before the add.
    - $r=16$ default. ILSVRC-2017 winner: top-5 2.251% (ensemble).

```{note} Math
:class: dropdown
Notations:
- Params:
    - $W_1\in\mathbb{R}^{(C/r)\times C}$: Squeeze FC.
    - $W_2\in\mathbb{R}^{C\times(C/r)}$: Excitation FC.
- Hyperparams:
    - $r$: Reduction ratio.

Forward:

$$\begin{align*}
z_c&=\frac{1}{HW}\sum_{h,w}X_{c,h,w}\\
\mathbf{a}&=\sigma\left(W_2\,\text{ReLU}(W_1\mathbf{z})\right)\\
Y_{c,h,w}&=a_c\,X_{c,h,w}
\end{align*}$$

- $\sigma$: Sigmoid.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(c, c // r), nn.ReLU(), nn.Linear(c // r, c), nn.Sigmoid())

    def forward(self, x):
        z = x.mean(dim=(2, 3))             ## squeeze: global context per channel
        a = self.fc(z)[:, :, None, None]   ## excite: per-channel gate in (0, 1)
        return x * a                       ## rescale channels

## Example
print(SEBlock(32)(torch.randn(2, 32, 6, 6)).shape)  ## torch.Size([2, 32, 6, 6])
```
````

```{attention} Q&A
:class: dropdown
*Why sigmoid, not softmax?*
- Several channels can matter at once → independent gates, ❌competition.

*Pros?*
- Plug-in for any backbone.
- Cheap: SE-ResNet-50 ≈ 28.1M params vs 25.6M; ≈ 3.87G vs 3.86G multiply-adds.

*Cons?*
- FC params grow w/ $C^2/r$ → concentrated in late, wide stages.
- GAP → sequential dependency → extra latency on hardware.

*How does it relate to attention?*
- Channel attention: input-dependent weights over channels, ❌over positions.
- Reused in [MobileNetV3](#mobilenetv3) & [EfficientNet](#efficientnet).
```

&nbsp;

## Efficient CNNs

&nbsp;

### MobileNet
- **What**: CNN of [depthwise separable convs](#depthwise-separable-convolution), w/ width & resolution multipliers. {cite:p}`howard2017mobilenets`
- **Why**: Mobile & embedded devices: tight latency, memory, & power budgets.
- **How**:
    1. Replace each 3×3 conv w/ 3×3 depthwise → BN → ReLU → 1×1 → BN → ReLU.
    2. Downsample w/ stride-2 depthwise convs.
    3. Width multiplier $\alpha\in(0,1]$: scale all channels by $\alpha$.
    4. Resolution multiplier $\rho\in(0,1]$: scale the input size by $\rho$.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\alpha$: Width multiplier.
    - $\rho$: Resolution multiplier.

Cost of one layer (MACs):

$$
k^2\cdot\alpha C_{in}\cdot\rho H_{out}\cdot\rho W_{out}+\alpha C_{in}\cdot\alpha C_{out}\cdot\rho H_{out}\cdot\rho W_{out}
$$

- Dominated by the pointwise term → cost ≈ $\propto\alpha^2\rho^2$.
```

```{note} Example
:class: dropdown
1.0 MobileNet-224: 569M MACs, 4.2M params, 70.6% ImageNet top-1.
- 3×3 depthwise separable conv → 8–9× fewer MACs than standard 3×3.
```

```{attention} Q&A
:class: dropdown
*Where does the compute go?*
- ~95% of compute & ~75% of params are in 1×1 convs.
- → 1×1 = dense matmul (GEMM) → runs efficiently.

*Cons?*
- No residuals → harder to go deep → [V2](#mobilenetv2).
- Small $\alpha$ → accuracy drops fast.
- Depthwise layers are memory-bound → measured speedup < MAC reduction.
```

&nbsp;

#### MobileNetV2
- **What**: MobileNet w/ inverted residual blocks & linear bottlenecks. {cite:p}`sandler2018mobilenetv2`
- **Why**:
    - V1 has no shortcuts → deep training harder.
    - ReLU on low-dim features destroys information.
- **How**:
    1. 1×1 expand ($C\to tC$; $t=6$, except $t=1$ in the first bottleneck) → BN → ReLU6.
    2. 3×3 depthwise → BN → ReLU6.
    3. 1×1 project ($tC\to C'$) → BN. **Linear**: ❌activation.
    4. Residual add when $s=1$ & $C=C'$.
    - Inverted: narrow → wide → narrow. Shortcut joins the narrow ends.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $t$: Expansion factor.
- Misc:
    - $\text{PW}$: Pointwise conv + BN.
    - $\text{DW}$: Depthwise 3×3 conv + BN.
    - $\text{ReLU6}(x)=\min(\max(x,0),6)$.

Forward ($s=1$, $C=C'$):

$$
\mathbf{y}=\mathbf{x}+\text{PW}_{tC\to C}\left(\text{ReLU6}\left(\text{DW}\left(\text{ReLU6}\left(\text{PW}_{C\to tC}(\mathbf{x})\right)\right)\right)\right)
$$
```

```{note} Example
:class: dropdown
MobileNetV2 (1.0, 224): 3.4M params, 300M MACs, 72.0% ImageNet top-1.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class InvertedResidual(nn.Module):
    def __init__(self, c_in, c_out, s=1, t=6):
        super().__init__()
        c_mid = t * c_in
        self.use_res = s == 1 and c_in == c_out
        self.f = nn.Sequential(
            nn.Conv2d(c_in, c_mid, 1, bias=False), nn.BatchNorm2d(c_mid), nn.ReLU6(),            ## expand
            nn.Conv2d(c_mid, c_mid, 3, s, 1, groups=c_mid, bias=False), nn.BatchNorm2d(c_mid), nn.ReLU6(),  ## depthwise
            nn.Conv2d(c_mid, c_out, 1, bias=False), nn.BatchNorm2d(c_out),                        ## linear projection
        )

    def forward(self, x):
        return x + self.f(x) if self.use_res else self.f(x)

## Example
print(InvertedResidual(16, 16)(torch.randn(2, 16, 8, 8)).shape)       ## torch.Size([2, 16, 8, 8])
print(InvertedResidual(16, 24, s=2)(torch.randn(2, 16, 8, 8)).shape)  ## torch.Size([2, 24, 4, 4])
```
````

```{attention} Q&A
:class: dropdown
*Why expand before the depthwise conv?*
- Depthwise conv ❌mixes channels & has few params → weak in low dims.
- Its cost $\propto C$ (❌$C^2$) → running it at $6C$ is cheap.

*Why a linear bottleneck?*
- ReLU zeros a half-space → information loss. In a low-dim space, that loss is large.
- The paper argues the useful features lie on a low-dim manifold. Keep it intact → no ReLU in the narrow layer.
- Empirically, adding ReLU to the bottleneck lowers accuracy.

*Why ReLU6?*
- Bounded range → robust in low-precision (fixed-point) inference.

*Why put the shortcut on the narrow ends?*
- Smaller tensors to keep in memory between blocks.
```

&nbsp;

#### MobileNetV3
- **What**: MobileNetV2 blocks + SE + h-swish; layout found by NAS & NetAdapt. {cite:p}`howard2019searching`
- **Why**: Hand design hits limits. Optimize measured phone latency, ❌FLOPs.
- **How**:
    1. Platform-aware NAS (MnasNet-style) → block-level structure.
    2. NetAdapt → per-layer channel counts.
    3. [SE](#senet) in some blocks. Gate = hard sigmoid. Squeeze width = 1/4 of the expansion width.
    4. h-swish in the deeper layers.
    5. Hand-redesign the costly first & last layers.
    - Large: +3.2 pp ImageNet top-1 & ~20% lower latency vs V2 (on the paper's phone).

```{note} Math
:class: dropdown
Forward:

$$\begin{align*}
\text{h-sigmoid}(x)&=\frac{\text{ReLU6}(x+3)}{6}\\
\text{h-swish}(x)&=x\cdot\text{h-sigmoid}(x)
\end{align*}$$

- Piecewise-linear approximations of sigmoid & [SiLU](../dl/foundation/activation.md#silu) (swish).
```

```{attention} Q&A
:class: dropdown
*Why h-swish instead of swish?*
- Sigmoid is costly on mobile hardware & hard to quantize.
- ReLU6-based version ≈ same accuracy, cheaper & quantization-friendly.

*Why h-swish only in deeper layers?*
- Cost of a nonlinearity $\propto$ feature-map size → early, high-res layers pay the most.

*Cons?*
- Search-found → irregular layout, hard to modify by hand.
- Optimized for a specific mobile CPU → not optimal on GPUs.
```

&nbsp;

### ShuffleNet
- **What**: CNN using grouped 1×1 convs + channel shuffle. {cite:p}`zhang2018shufflenet`
- **Why**: In tiny nets, dense 1×1 convs dominate cost.
    - Grouped 1×1 is cheap, but stacked groups never exchange information.
- **How**:
    1. Grouped 1×1 conv.
    2. **Channel shuffle**: reshape channels to $(G, C/G)$ → transpose → flatten. Each next group gets channels from every earlier group.
    3. 3×3 depthwise conv.
    4. Grouped 1×1 conv.
    5. Residual add. For $s=2$: concat w/ an avg-pooled shortcut.

```{note} Math
:class: dropdown
Channel shuffle (permutation):

$$
c=g\cdot\frac{C}{G}+q\ \longmapsto\ q\cdot G+g
$$

- $g\in\{0,\dots,G-1\}$: Group idx.
- $q\in\{0,\dots,C/G-1\}$: Idx within the group.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class ChannelShuffle(nn.Module):
    def __init__(self, G):
        super().__init__()
        self.G = G

    def forward(self, x):
        B, C, H, W = x.shape
        ## (G, C/G) -> (C/G, G): each next-layer group now mixes channels from every old group
        return x.view(B, self.G, C // self.G, H, W).transpose(1, 2).reshape(B, C, H, W)

## Example: 6 channels, 2 groups
x = torch.arange(6.0).view(1, 6, 1, 1)
print(ChannelShuffle(2)(x).flatten())  ## tensor([0., 3., 1., 4., 2., 5.])
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- Cross-group flow at 0 params & near-0 FLOPs.

*FLOPs = speed?*
- ❌. ShuffleNet V2 gives 4 guidelines for real speed {cite:p}`ma2018shufflenet`:
    1. Equal input & output channel width minimizes memory access cost.
    2. Excessive group conv increases memory access cost.
    3. Network fragmentation (many small branches) reduces parallelism.
    4. Element-wise ops (ReLU, add) are non-negligible.
- → V2 replaces grouped 1×1 w/ a channel split + concat.
```

&nbsp;

### EfficientNet
- **What**: CNN family scaled from one NAS-found baseline (B0) by compound scaling of depth, width, & resolution. {cite:p}`tan2019efficientnet`
- **Why**: Scaling 1 dimension alone saturates fast.
    - Dimensions interact: higher resolution needs more depth (RF) & more width (finer patterns).
- **How**:
    1. NAS → B0: MBConv blocks ([MobileNetV2](#mobilenetv2) inverted residual + [SE](#senet)) w/ SiLU (swish).
    2. Fix $\phi=1$. Grid search $\alpha,\beta,\gamma$ under total FLOPs ≈ 2×.
    3. Fix $\alpha,\beta,\gamma$. Raise $\phi$ → B1–B7.
    - B7: 84.4% ImageNet top-1, 66M params. 8.4× smaller & 6.1× faster than GPipe (ICML version).

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\phi$: Compound coefficient.
    - $\alpha,\beta,\gamma\ge1$: Per-dimension bases.
- Misc:
    - $m_d$: Depth multiplier (paper: $d$).
    - $m_w$: Width multiplier (paper: $w$).
    - $m_r$: Resolution multiplier (paper: $r$).

Objective:

$$
m_d=\alpha^\phi,\quad m_w=\beta^\phi,\quad m_r=\gamma^\phi,\qquad\text{s.t. }\alpha\beta^2\gamma^2\approx2
$$

- Found: $\alpha=1.2$, $\beta=1.1$, $\gamma=1.15$.
- → FLOPs ≈ $2^\phi$.
```

```{tip} Derivation
:class: dropdown
*Why is FLOPs $\propto m_dm_w^2m_r^2$?*
1. Conv MACs per layer $\propto C_{in}C_{out}HW$.
2. Width scales both $C_{in}$ & $C_{out}$ → $m_w^2$.
3. Resolution scales both $H$ & $W$ → $m_r^2$.
4. Depth scales #layers → $m_d$.
5. → $(\alpha\beta^2\gamma^2)^\phi\approx2^\phi$.
```

````{important} Code
:class: dropdown
```python
class CompoundScaling:
    def __init__(self, alpha=1.2, beta=1.1, gamma=1.15, base_res=224):
        self.a, self.b, self.g, self.res = alpha, beta, gamma, base_res

    def __call__(self, phi):
        m_d, m_w, m_r = self.a ** phi, self.b ** phi, self.g ** phi
        flops = m_d * m_w ** 2 * m_r ** 2  ## ~= 2 ** phi by construction
        return round(m_d, 2), round(m_w, 2), round(self.res * m_r), round(flops, 2)

## Example
print(CompoundScaling()(1))  ## (1.2, 1.1, 258, 1.92)
print(CompoundScaling()(3))  ## (1.73, 1.33, 341, 7.08)
## released B1-B7 use rounded, hand-set resolutions (240, 260, 300, 380, 456, 528, 600)
```
````

```{attention} Q&A
:class: dropdown
*Why the constraint $\alpha\beta^2\gamma^2\approx2$?*
- Each +1 in $\phi$ → ~2× FLOPs → predictable compute ladder.
- Search $\alpha,\beta,\gamma$ once on the small model → cheap. Reuse for all sizes.

*Pros?*
- Best accuracy per FLOP & per param at release.
- One rule for any compute budget.

*Cons?*
- High resolution + depthwise convs → memory-heavy, slow GPU training.
- FLOP-efficient ≠ GPU-latency-efficient → EfficientNetV2 targets training speed.
- Reported B0 accuracy differs by version: 76.3% (ICML) vs 77.1% (later arXiv).
```

&nbsp;

## Modern CNNs

&nbsp;

### ConvNeXt
- **What**: Pure CNN made by modernizing ResNet-50 step by step toward Swin Transformer design. {cite:p}`liu2022convnet`
- **Why**: Did ViTs win because of attention, or because of better training & design?
    - Test: port Transformer design choices into a CNN, keep only conv.
- **How** (ResNet-50 → ConvNeXt-T; keep each step that helps):
    1. Modern training recipe (AdamW, 300 epochs, Mixup, CutMix, RandAugment, stochastic depth, label smoothing): 76.1% → 78.8%.
    2. Macro: stage ratio 3:3:9:3 (T; larger models 3:3:27:3). Patchify stem: 4×4 conv, $s=4$.
    3. ResNeXt-ify: depthwise conv. Width 64 → 96.
    4. Inverted bottleneck: expand 4×.
    5. Large kernel: move the depthwise conv to the block start. 7×7.
    6. Micro: ReLU → [GELU](../dl/foundation/activation.md#gelu). 1 activation & 1 norm per block. BN → [LN](../dl/foundation/module.md#layernorm). Separate downsampling layers (LN + 2×2 conv, $s=2$).
    - Result: ConvNeXt-T 82.1% ImageNet top-1, ~28M params, 4.5G multiply-adds (reported as "FLOPs").

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\boldsymbol{\gamma}\in\mathbb{R}^{C}$: Layer scale, init $10^{-6}$.
- Misc:
    - $\text{DW}_{7\times7}$: 7×7 depthwise conv.
    - $\text{PW}_{a\to b}$: Pointwise conv $a\to b$ channels.

Forward:

$$
\mathbf{y}=\mathbf{x}+\boldsymbol{\gamma}\odot\text{PW}_{4C\to C}\left(\text{GELU}\left(\text{PW}_{C\to4C}\left(\text{LN}\left(\text{DW}_{7\times7}(\mathbf{x})\right)\right)\right)\right)
$$

- Stochastic depth on the branch during training.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNeXtBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 7, padding=3, groups=c)  ## 7x7 depthwise: spatial mixing only
        self.norm = nn.LayerNorm(c)                        ## LN over channels (channels-last)
        self.pw1 = nn.Linear(c, 4 * c)                     ## 1x1 conv == Linear on channels-last
        self.pw2 = nn.Linear(4 * c, c)
        self.gamma = nn.Parameter(1e-6 * torch.ones(c))    ## layer scale: branch starts ~0

    def forward(self, x):                                  ## x: (B, C, H, W)
        h = self.dw(x).permute(0, 2, 3, 1)                 ## -> (B, H, W, C)
        h = self.pw2(F.gelu(self.pw1(self.norm(h))))       ## inverted bottleneck = Transformer FFN
        return x + (self.gamma * h).permute(0, 3, 1, 2)  ## stochastic depth (DropPath) omitted

## Example
print(ConvNeXtBlock(16)(torch.randn(2, 16, 8, 8)).shape)  ## torch.Size([2, 16, 8, 8])
```
````

```{attention} Q&A
:class: dropdown
*CNN or ViT?*
- ViT < comparable ResNets on ImageNet-1k alone. ViT > ResNets when pretrained on ImageNet-21k or JFT-300M {cite:p}`dosovitskiy2021image`.
- ConvNeXt: most of the gap comes from training recipe & design, ❌attention itself.
- CNN priors → better at small data. ViT → scales w/ data, handles any token type (multimodal).

*How does the block map to a Transformer block?*
- Depthwise 7×7 ↔ attention: spatial mixing only.
- 1×1 → GELU → 1×1 ↔ FFN: channel mixing only, 4× expansion.
- LN & 1 activation per block ↔ Transformer micro-design.

*Why 7×7?*
- Accuracy saturates at 7×7 in the paper.
- Depthwise → large kernel stays cheap.

*Cons?*
- Depthwise 7×7 → lower GPU throughput than dense 3×3 at equal FLOPs.
- Still local per layer → global context only via depth.
```

&nbsp;
