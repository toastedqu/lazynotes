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
# Image Input & Augmentation
Raw image → model input tensor, & data augmentation. Architectures live elsewhere: [Convolution](cnn.md#convolution), [DiT](../dl/gen/diffusion.md#dit).

Notations:
- $\mathbf{X}\in\mathbb{R}^{C\times H\times W}$: One image, channel-first
- $C$: #channels (3 for RGB)
- $H$: Height (pixels)
- $W$: Width (pixels)
- $u$: Pixel row idx
- $v$: Pixel column idx
- $K$: #classes
- $\mathbf{y}\in[0,1]^K$: Target distribution (one-hot, or soft after mixing)
- $f_\theta$: Model (outputs logits)
- $t$: Random transform (1 augmentation draw)
- $\mathcal{T}$: Distribution over transforms
- $\lambda$: Mixing ratio (share of the first image)

&nbsp;

## Representation

### Digital Image
- **What**: $H\times W$ grid of pixels. Each pixel = $C$ quantized intensities.
- **Why**: Computer stores only finite arrays → sample continuous light at discrete positions & levels.
- **How**:
    1. Sensor: 1 color filter per photosite (Bayer mosaic) → demosaic → RGB per pixel.
    2. Camera processing: white balance → sRGB gamma encoding → quantize to 8 bits/channel → `uint8` in $\{0,\dots,255\}$.
    3. Compress (JPEG/PNG) → file. Decode → `uint8` array, HWC layout.

```{attention} Q&A
:class: dropdown
*HWC or CHW?*
- Files, PIL, NumPy, OpenCV: HWC. PyTorch: CHW (batch: NCHW). TF & Flax: NHWC. Core JAX `lax` conv: NCHW by default.
- Convert w/ `permute(2, 0, 1)`. `reshape`/`view` instead of `permute` → no error, scrambled pixels.

*RGB or BGR?*
- OpenCV `imread` → BGR. PIL → RGB.
- Mix-up → red & blue swapped → no error, silent accuracy drop on a pretrained model.

*Are pixel values proportional to light intensity?*
- ❌ sRGB values are gamma-encoded: linear light ≈ $(\text{value}/255)^{2.2}$.
- → Averaging pixels (resize, blend, Mixup) in sRGB ≠ physically correct light mixing.
- Standard practice ignores this. Models train & test on sRGB values → consistent.

*Other loading gotchas?*
- EXIF orientation: phones store rotated pixels + an orientation tag. Loader ignores tag → rotated image (fix: PIL `ImageOps.exif_transpose`).
- RGBA (4 ch), grayscale (1 ch), CMYK → convert to RGB (PIL `.convert("RGB")`).
- 16-bit images (medical, raw, HDR): max ≠ 255. Do not hard-code ÷255.

*How large is one decoded image?*
- 224×224×3 `uint8` = 150,528 B ≈ 147 KiB. As `float32`: ×4 ≈ 588 KiB.
- JPEG on disk ≈ 10× smaller → decode is CPU work on every load.
```

&nbsp;

#### Color Space
- **What**: Coordinate system for pixel color.
- **Why**: Some tasks & operations need different axes, e.g., brightness separated from color.
- **How**: Linear or nonlinear map from RGB to new axes.

```{dropdown} Table: Color Spaces
| Space | Axes | Map from RGB | Typical use |
|:--|:--|:--|:--|
| RGB | Red, green, blue | — | Default model input |
| Grayscale | Luma $Y$ | $Y=0.299R+0.587G+0.114B$ (BT.601) | 1-ch models, grayscale augmentation |
| HSV | Hue, saturation, value | Nonlinear | Hue & saturation jitter |
| YCbCr | Luma + 2 chroma | Affine | JPEG compression, video |
| Lab | Lightness + 2 opponent axes | Nonlinear | Perceptual color distances |
```

```{attention} Q&A
:class: dropdown
*Why unequal grayscale weights?*
- Human eye is most sensitive to green, least to blue → weights match perceived brightness.

*Does the input color space matter for a CNN?*
- First conv layer is linear in its input.
- RGB → YCbCr is affine → first layer can absorb it → ~no difference.
- Nonlinear spaces (HSV, Lab) change what a linear first layer can represent. Pretrained models expect the exact space they saw (usually RGB).

*Why does JPEG use YCbCr?*
- Eye is less sensitive to color detail than to brightness detail.
- → Store chroma at ¼ resolution (4:2:0) → samples per pixel 3 → 1.5.
- → Color edges blur first under heavy compression.
```

&nbsp;

## Preprocessing
- **What**: Bytes $\xrightarrow{\text{decode}}$ `uint8` HWC $\xrightarrow{\text{resize, crop}}$ fixed size $\xrightarrow{\text{to tensor}}$ `float` CHW in $[0,1]$ $\xrightarrow{\text{normalize}}$ $\mathbf{X}$ $\xrightarrow{\text{collate}}$ batch $(B,C,H,W)$
- **Why**: A batch is one tensor → every image needs the same shape. Nets train best on small, centered inputs.
- **How**:
    - **Train**: random geometry instead of a fixed crop → [Data Augmentation](#data-augmentation).
    - **Eval**: deterministic. Resize shorter side to $S/0.875$ → center crop $S\times S$ (e.g., 256 → 224).
    - Run in DataLoader worker processes (CPU) → overlaps w/ GPU compute.

````{important} Code
:class: dropdown
```python
import torch
from torchvision.transforms import v2

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]   ## ImageNet stats, RGB order

class ImagePipeline:
    def __init__(self, S=224, train=True):
        if train:
            ## random geometry -> a new view of the image every epoch
            geo = [v2.RandomResizedCrop(S, antialias=True), v2.RandomHorizontalFlip()]
        else:
            ## deterministic: shorter side -> S / 0.875, then center crop
            geo = [v2.Resize(int(S / 0.875), antialias=True), v2.CenterCrop(S)]
        self.t = v2.Compose(geo + [
            v2.ToDtype(torch.float32, scale=True),   ## uint8 [0, 255] -> float [0, 1]
            v2.Normalize(MEAN, STD),                 ## per-channel standardization
        ])

    def __call__(self, img):                         ## img: uint8 (C, H, W), RGB
        return self.t(img)

## Example: one decoded 300x400 RGB image
img = torch.randint(0, 256, (3, 300, 400), dtype=torch.uint8)
print(ImagePipeline(train=True)(img).shape, ImagePipeline(train=False)(img).shape)
## torch.Size([3, 224, 224]) torch.Size([3, 224, 224])
```
````

```{attention} Q&A
:class: dropdown
*Why resize the shorter side, then center crop?*
- Keeps the aspect ratio → shapes are not stretched.
- Crop drops borders. Object of interest is usually near the center.
- 224/256 = 0.875 = common "crop fraction".

*Why not resize straight to $S\times S$?*
- Non-square image → aspect distortion. Train & eval must then distort in the same way.
- Detection: **letterbox** instead → resize the longer side to $S$, pad the rest → no content lost, no distortion.

*Must eval preprocessing match training?*
- Yes. Mismatch (interpolation, crop fraction, mean/std, RGB/BGR) = [covariate shift](../dl/train/issues.md#covariate-shift).
- Pretrained model → copy its exact preprocessing config, ❌your own defaults.

*GPU sits idle during training. Why?*
- JPEG decode + augmentation on CPU cannot keep up → input-bound.
- Fixes: more workers, pre-resized dataset, GPU decode & augmentation (e.g., NVIDIA DALI), cheaper augmentations.
```

&nbsp;

### Resize
- **What**: Resample the pixel grid to a new size $H'\times W'$ by interpolation.
- **Why**: Model needs a fixed size. Compute ∝ #pixels.
- **How**: Each output pixel = weighted avg of input pixels near its source position.
    - **Nearest**: copy the closest pixel. Use for label masks (values must stay class IDs).
    - **Bilinear**: 2×2 neighbors, linear weights.
    - **Bicubic**: 4×4 neighbors, cubic weights → sharper.
    - **Antialias / area**: low-pass filter w/ width ∝ downsampling factor, then sample.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{X}'\in\mathbb{R}^{C\times H'\times W'}$: Resized image.
- Misc:
    - $u_s,v_s$: Source coordinates of output pixel $(u,v)$ in the input grid.

Definition (bilinear, per channel, 0-indexed):

$$
u_s=\left(u+\tfrac{1}{2}\right)\frac{H}{H'}-\tfrac{1}{2},\qquad v_s=\left(v+\tfrac{1}{2}\right)\frac{W}{W'}-\tfrac{1}{2}
$$

- Half-pixel centers (`align_corners=False`).
- `align_corners=True`: $u_s=u\,\frac{H-1}{H'-1}$ → corner pixel centers map exactly.

$$
\begin{align*}
X'_{u,v}=\;&(1-\delta_u)(1-\delta_v)\,X_{u_0,v_0}+(1-\delta_u)\,\delta_v\,X_{u_0,v_0+1}\\
+\;&\delta_u(1-\delta_v)\,X_{u_0+1,v_0}+\delta_u\,\delta_v\,X_{u_0+1,v_0+1}
\end{align*}
$$

- $u_0=\lfloor u_s\rfloor$: Upper neighbor row.
- $v_0=\lfloor v_s\rfloor$: Left neighbor column.
- $\delta_u=u_s-u_0$: Vertical weight.
- $\delta_v=v_s-v_0$: Horizontal weight.
- Borders: $u_s,v_s$ can fall outside the grid (e.g., $u_s<0$ when upsampling) → clamp indices to $[0,H-1]$ & $[0,W-1]$ (edge replicate).
```

```{note} Example
:class: dropdown
Downsample a 1-pixel checkerboard (alternating 0 & 1) by 2×:
- Nearest: keeps pixels $(2u,2v)$ → all have the same parity → solid 0 (or solid 1). ❌ Wrong: true local mean = 0.5.
- Area (2×2 average): every output = 0.5 → uniform gray. ✅ Correct.
- → Detail finer than the new pixel spacing must be averaged away, ❌sampled.
```

```{attention} Q&A
:class: dropdown
*Why antialias when downsampling?*
- Downsample by $s$ → new grid represents only frequencies below its Nyquist limit.
- Higher frequencies do not vanish. They fold into false low frequencies (**aliasing**) → moiré, jagged edges.
- Antialias = blur first (low-pass), kernel width ∝ $s$.
- Fixed-width bilinear (2 taps per axis) at large $s$ ignores most input pixels → ≈ nearest → aliasing.

*Do resize implementations really differ?*
- Yes. PIL scales the filter w/ the factor. Fixed-width bilinear (as in the PyTorch-FID & TF-FID resizers) aliases.
- Same images, different resizer → different FID {cite:p}`parmar2022aliased`.
- → Match the resizer of training at inference.

*What is the `align_corners` gotcha?*
- `True` vs `False` → sub-pixel shift between grids.
- Train in one convention, deploy in another → small, systematic misalignment. Visible in dense tasks (segmentation, keypoints).

*Does upsampling add information?*
- ❌ It interpolates existing samples. Larger input → more compute, maybe a better scale match for the model, no new detail.
```

&nbsp;

### Pixel Normalization
- **What**: Per-channel standardization w/ fixed dataset statistics.
- **Why**: Conditioning.
    - Raw inputs in $[0,255]$, all positive → large, same-sign weight grads per neuron → zig-zag GD, slow.
    - Zero mean & unit variance → first-layer activations start at the scale [He init](../dl/train/train.md#he-initialization) is designed to preserve.
- **How**:
    1. Scale `uint8` → $[0,1]$ (÷255).
    2. Once: compute mean & std per channel over all train pixels.
    3. Every load (train & eval): subtract mean, divide by std.
    - Generative models (diffusion, GAN) map to $[-1,1]$ instead: $2\mathbf{X}-1$.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{X}^{(i)}$: $i$-th train image, values in $[0,1]$.
    - $\tilde{\mathbf{X}}$: Normalized image.
- Misc:
    - $\mu_c$: Mean of channel $c$.
    - $\sigma_c$: Std of channel $c$.

Definition:

$$
\begin{align*}
\mu_c&=\frac{1}{mHW}\sum_{i=1}^{m}\sum_{u,v}X^{(i)}_{c,u,v}\\
\sigma_c^2&=\frac{1}{mHW}\sum_{i=1}^{m}\sum_{u,v}\left(X^{(i)}_{c,u,v}-\mu_c\right)^2\\
\tilde{X}_{c,u,v}&=\frac{X_{c,u,v}-\mu_c}{\sigma_c}
\end{align*}
$$

- $2C$ constants total. Same constants for every image & every position.
- ImageNet: $\mu=(0.485,0.456,0.406)$, $\sigma=(0.229,0.224,0.225)$.
- CLIP: $\mu=(0.4815,0.4578,0.4082)$, $\sigma=(0.2686,0.2613,0.2758)$.
```

````{important} Code
:class: dropdown
```python
import torch

class PixelNorm:
    def fit(self, X):                                    ## X: (m, C, H, W) in [0, 1], TRAIN split only
        ## pool over samples AND positions -> one number per channel
        self.mu = X.mean(dim=(0, 2, 3)).view(-1, 1, 1)                  ## (C, 1, 1)
        self.sigma = X.std(dim=(0, 2, 3), correction=0).view(-1, 1, 1)  ## (C, 1, 1)
        return self

    def __call__(self, X):
        return (X - self.mu) / self.sigma                ## broadcasts over (H, W)

## Example: 3 channels on very different scales
X = torch.rand(8, 3, 4, 4) * torch.tensor([1.0, 0.5, 0.2]).view(1, 3, 1, 1)
Z = PixelNorm().fit(X)(X)
print(Z.mean(dim=(0, 2, 3)), Z.std(dim=(0, 2, 3), correction=0))   ## ~[0, 0, 0] ~[1, 1, 1]
```
````

```{attention} Q&A
:class: dropdown
*Why per channel, ❌per pixel?*
- Per-pixel mean = a "mean image" → assumes objects sit at fixed positions.
- Random crops shift content → a position-specific offset becomes noise.
- Per-channel: $2C$ numbers, same at every position → fits [convolution](cnn.md#convolution)'s translation equivariance.

*Fine-tuning a pretrained model: whose statistics?*
- The pretrained model's. Its first layer learned weights for that input distribution.
- Small mismatch (ImageNet vs CLIP constants) → minor shift.
- Large mismatch ($[0,255]$ vs normalized) → garbage outputs, no error.

*Compute the statistics on which split?*
- Train only → avoids [data leakage](../ml/misc.md#data-leakage). Effect is tiny here. The rule still applies.

*How is it different from [BatchNorm](../dl/foundation/module.md#batchnorm)?*
- Pixel normalization: fixed constants, input only, no params.
- BatchNorm: batch statistics at every layer + learnable scale & shift.
- Both fix scale. Only BatchNorm keeps fixing it as activations drift during training.

*Per-image standardization (each image to mean 0, std 1)?*
- Removes global brightness & contrast → invariance for free.
- Also deletes signal when brightness matters (day vs night, exposure-dependent classes).

*Why $[-1,1]$ for diffusion models?*
- Noise prior $\mathcal{N}(0,I)$ is zero-centered, unit scale → $[-1,1]$ puts data in a zero-centered, order-1 range → comparable to the noise scale.
- ❌Exactly unit variance. Exact variance preservation in the [forward process](../dl/gen/diffusion.md#forward-process) needs unit-variance data.
```

&nbsp;

## Model Input

### Conv Stem
- **What**: First conv & pool layers. They map the pixel grid to a smaller, wider feature map.
- **Why**: Pixels are redundant & high-resolution. Downsample early → compute ⬇️ in all later layers.
- **How**:
    1. Feed $\mathbf{X}$ as a grid. ❌Flattening → keeps locality for [convolution](cnn.md#convolution).
    2. Strided conv + [pooling](cnn.md#pooling) → spatial size ⬇️, #channels ⬆️.
    - ResNet stem: 7×7 conv, stride 2, 64 ch → 3×3 max-pool, stride 2 → 224² → 56² × 64.

```{attention} Q&A
:class: dropdown
*Does a CNN need a fixed input size?*
- Conv & pool layers: ❌ They slide over any size.
- FC head: ✅ Fixed input dim → global average pooling before the head removes the constraint.
- Accuracy still peaks near the training object scale → [train-test resolution discrepancy](#random-resized-crop).

*What does the first layer learn?*
- Oriented edge detectors (Gabor-like) & color blobs.
- Similar across datasets & architectures → reason early layers transfer well.
```

&nbsp;

### Patch Embedding
- **What**: Cut the image into non-overlapping $P\times P$ patches → flatten each → shared linear projection → token sequence. {cite:p}`dosovitskiy2021image`
- **Why**: A transformer needs a token sequence.
    - 1 token per pixel: 224² = 50,176 tokens → $\approx2.5\times10^9$ [attention](../nlp/attn.md#scaled-dot-product-attention) scores per head per layer → infeasible.
    - $P=16$ → 196 tokens.
- **How**:
    1. Split $\mathbf{X}$ → $N=HW/P^2$ patches. Each patch = $P^2C$ values.
    2. Shared linear map → $D$-dim token per patch.
    3. Prepend a learnable [CLS] token. Add [learned position embeddings](../nlp/pos.md#learned).

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{x}_p^r\in\mathbb{R}^{P^2C}$: Flattened $r$-th patch, raster order.
    - $Z_0\in\mathbb{R}^{(N+1)\times D}$: Token sequence fed to the transformer.
- Params:
    - $E\in\mathbb{R}^{P^2C\times D}$: Patch projection.
    - $\mathbf{x}_\text{cls}\in\mathbb{R}^{D}$: [CLS] token.
    - $E_\text{pos}\in\mathbb{R}^{(N+1)\times D}$: Position embeddings.
- Hyperparams:
    - $P$: Patch size.
    - $D$: Token width.

Forward:

$$
Z_0=\left[\mathbf{x}_\text{cls};\;\mathbf{x}_p^1E;\;\dots;\;\mathbf{x}_p^NE\right]+E_\text{pos},\qquad N=\frac{HW}{P^2}
$$

- ViT-B/16 at 224²: $N=196$, sequence length 197. $P^2C=768=D$.
- Attention cost ∝ $N^2\propto P^{-4}$ at fixed resolution.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, P=16, C=3, D=768):
        super().__init__()
        self.P = P
        N = (img_size // P) ** 2
        self.proj = nn.Linear(P * P * C, D)                       ## E: one map shared by all patches
        self.cls = nn.Parameter(torch.zeros(1, 1, D))
        self.pos = nn.Parameter(torch.randn(1, N + 1, D) * 0.02)  ## learned 1D position embeddings

    def patchify(self, X):                                        ## X: (B, C, H, W)
        B, C, H, W = X.shape
        P = self.P
        ## split H & W into (grid, within-patch) -> grid axes first -> each patch flattened as (C, P, P)
        X = X.reshape(B, C, H // P, P, W // P, P).permute(0, 2, 4, 1, 3, 5)
        return X.reshape(B, (H // P) * (W // P), C * P * P)      ## (B, N, P*P*C), raster order

    def forward(self, X):
        z = self.proj(self.patchify(X))                           ## (B, N, D)
        z = torch.cat([self.cls.expand(len(z), -1, -1), z], dim=1)
        return z + self.pos                                       ## (B, N+1, D)

## Example: 32x32 image, P=8 -> 16 patches + CLS
pe = PatchEmbed(img_size=32, P=8, C=3, D=64)
X = torch.randn(2, 3, 32, 32)
print(pe(X).shape)                                                ## torch.Size([2, 17, 64])

## Same map as a conv w/ kernel = stride = P (how ViT code implements it)
conv = nn.Conv2d(3, 64, kernel_size=8, stride=8)
conv.weight.data = pe.proj.weight.data.view(64, 3, 8, 8)
conv.bias.data = pe.proj.bias.data
z_conv = conv(X).flatten(2).transpose(1, 2)                       ## (2, 16, 64)
print(torch.allclose(z_conv, pe.proj(pe.patchify(X)), atol=1e-5)) ## True
```
````

```{attention} Q&A
:class: dropdown
*Why is patch embedding the same as a conv?*
- Each token = linear fn of 1 patch, same weights at every position.
- = conv w/ kernel size = stride = $P$, $D$ output channels. Implementations use `Conv2d`.

*Patch size tradeoff?*
- $P$⬇️ → $N\propto P^{-2}$⬆️ → finer detail, accuracy ⬆️, attention cost ∝ $P^{-4}$⬆️.
- Name encodes it: ViT-B/16 vs ViT-B/32.

*Why does ViT need more data than a CNN?*
- Less [inductive bias](../ml/misc.md#inductive-bias): 2D structure enters only via patchify & position embeddings. Attention is global from layer 1.
- ImageNet-1k only → ViT < comparable ResNets. Large pretraining (JFT-300M) → ViT ≥ CNNs.
- Strong augmentation + distillation close most of the gap on ImageNet-1k alone {cite:p}`touvron2021training`.

*Fine-tune at a higher resolution?*
- Keep $P$ → more patches → old position embeddings do not cover the new grid.
- Fix: reshape $E_\text{pos}$ (w/o [CLS]) to its $\sqrt{N}\times\sqrt{N}$ grid → 2D interpolate → new grid size.

*1D or 2D position embeddings?*
- ViT: learned 1D ≈ 2D-aware variants → no significant gain. The model learns the 2D grid from data.
- Fixed 2D sin-cos also works → used by [DiT](../dl/gen/diffusion.md#dit).

*[CLS] token or global average pooling?*
- Both work in ViT once the LR is tuned for each.

*Continuous patches vs discrete image tokens?*
- Patch embedding → continuous vectors. Multimodal LMs usually project vision-encoder patch features into the LM embedding space.
- Discrete: a VQ autoencoder maps patches to codebook IDs → same next-token CE as text.
```

&nbsp;

## Data Augmentation
- **What**: Train on random, label-preserving transforms of each sample.
- **Why**:
    - *Why do we need it?*
        - Labeled data is finite → [overfitting](../dl/train/issues.md#overfitting).
        - Model must ignore nuisances (position, scale, lighting). Data shows each object under few of them.
    - *Why does it work?*
        - Encodes a prior: "label does not change under $t$".
        - → Fills the neighborhood of each sample w/ new valid samples → effective #samples ⬆️.
- **How**:
    1. Each load of a sample: draw $t\sim\mathcal{T}$ → train on $(t(\mathbf{X}),\mathbf{y})$.
    2. New draw every epoch → the model almost never sees the same input twice.
    3. Eval: no random transforms (except [TTA](#test-time-augmentation)).

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\ell$: Per-sample loss.
    - $\mathcal{D}$: Training set.

Objective:

$$
\min_\theta\;\mathbb{E}_{(\mathbf{X},\mathbf{y})\sim\mathcal{D}}\;\mathbb{E}_{t\sim\mathcal{T}}\left[\ell\left(f_\theta(t(\mathbf{X})),\mathbf{y}\right)\right]
$$

- Each SGD step = 1-sample Monte-Carlo estimate of the inner expectation.
- $\mathcal{T}=\{\text{identity}\}$ → plain ERM.
```

```{tip} Derivation
:class: dropdown
*Why is small-noise augmentation a regularizer?*

Setup: $\mathbf{x}$ = flattened image, scalar output $f$, scalar target $y$, squared loss, noise $\boldsymbol{\epsilon}\sim\mathcal{N}(0,\sigma^2I)$.

1. Taylor: $f(\mathbf{x}+\boldsymbol{\epsilon})\approx f+\boldsymbol{\epsilon}^T\nabla f+\frac{1}{2}\boldsymbol{\epsilon}^T\nabla^2f\,\boldsymbol{\epsilon}$.
2. Odd moments of $\boldsymbol{\epsilon}$ = 0. $\mathbb{E}[\boldsymbol{\epsilon}^T\nabla^2f\,\boldsymbol{\epsilon}]=\sigma^2\operatorname{tr}\nabla^2f$.
3. → Expected loss:

    $$
    \mathbb{E}_{\boldsymbol{\epsilon}}\left[(f(\mathbf{x}+\boldsymbol{\epsilon})-y)^2\right]=(f-y)^2+\sigma^2\|\nabla_\mathbf{x}f\|^2+\sigma^2(f-y)\operatorname{tr}\nabla^2_\mathbf{x}f+O(\sigma^4)
    $$

4. Near a good fit, $f-y\approx0$ → 3rd term small.
5. → Augmented loss ≈ clean loss + $\sigma^2\|\nabla_\mathbf{x}f\|^2$: penalty on input sensitivity (Tikhonov regularization).
6. Other transforms: same idea along their own directions → penalize sensitivity of $f$ to that transform.
```

```{dropdown} Table: Augmentation Families
| Family | Changes | Label | Teaches invariance to | Examples |
|:--|:--|:--|:--|:--|
| [Geometric](#geometric) | Pixel positions | Same | Position, scale, orientation | Flip, crop, rotate |
| [Photometric](#photometric) | Pixel values | Same | Lighting, camera, compression | Color jitter, blur |
| [Occlusion](#cutout) | Region → constant/noise | Same | Missing parts | Cutout, Random Erasing |
| [Mixing](#mixup) | 2 images combined | Mixed | — (smoothness between classes) | Mixup, CutMix |
| [Learned/random policy](#autoaugment) | Sequence of ops | Same | Op set | AutoAugment, RandAugment |
```

```{attention} Q&A
:class: dropdown
*When does "label-preserving" fail?*
- Flips & rotations on text & digits: 'b' ↔ 'd', 6 ↔ 9.
- Vertical flip on natural scenes: sky at the bottom never occurs at test time.
- Color jitter when color is the label: fruit ripeness, skin lesions.
- Aggressive crop removes the object → label noise.
- Rule: $\mathcal{T}$ = transforms the task is truly invariant to.

*Augment before or after the train/test split?*
- After. Train split only. Augmented copies of a test image in train = [data leakage](../ml/misc.md#data-leakage).

*Offline or online?*
- Offline: precompute $k$ copies → disk ×$k$, fixed variety.
- Online: fresh draw each load → unlimited variety, CPU cost. Standard.

*Augmentation vs architecture for invariance?*
- Augmentation: learned, approximate, only for the transforms seen. Costs data & compute.
- Architecture (e.g., conv → translation equivariance): exact & free, but must be designed in.
- Less [inductive bias](../ml/misc.md#inductive-bias) in the architecture → more augmentation needed (ViT vs CNN).

*How strong?*
- Too strong → train distribution far from test → underfitting.
- Larger models & longer schedules tolerate & gain more from strong augmentation.
- Smaller datasets gain more from augmentation. Web scale needs little: CLIP used only random square crops.

*Detection & segmentation?*
- Geometric transforms → apply the same transform to labels (boxes, masks).
- Masks → nearest interpolation. Boxes cut by a crop → clip or drop.
- Photometric transforms → image only.

*Does augmentation fix [class imbalance](../ml/misc.md#class-imbalance)?*
- Partly. Heavier augmentation of minority-class oversamples ⬇️ exact duplicates. It does not add new modes of the minority class.
```

&nbsp;

### Geometric
- **What**: Transforms that move pixels. Values change only by interpolation.
- **Why**: Object identity does not depend on position, scale, orientation, or small viewpoint changes.
- **How**:
    - **Horizontal flip**: mirror w/ $p=0.5$.
    - **Crop / translate**: random sub-window. CIFAR standard: pad 4 → random 32×32 crop + flip.
    - **Rotate / scale / shear**: affine map + interpolation. Fill exposed corners (constant or reflect).
    - **Perspective / elastic**: rarer. Elastic is common for handwriting & medical images.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\mathbf{q}=(u,v)^T$: Output pixel coordinates (centered).
    - $A\in\mathbb{R}^{2\times2}$: Linear part (rotate, scale, shear).
    - $\mathbf{b}\in\mathbb{R}^{2}$: Translation.
    - $\phi$: Rotation angle.

Definition (inverse warp):

$$
X'_{:,\mathbf{q}}=X_{:,\,A^{-1}(\mathbf{q}-\mathbf{b})}
$$

- Non-integer source position → [interpolate](#resize).
- Rotation: $A=\begin{bmatrix}\cos\phi&-\sin\phi\\\sin\phi&\cos\phi\end{bmatrix}$.
- Horizontal flip: $X'_{:,u,v}=X_{:,u,W-1-v}$ (0-indexed).
```

```{attention} Q&A
:class: dropdown
*Why inverse warp, ❌forward warp?*
- Forward: push each input pixel to its new position → some output pixels get no value (holes), others get several.
- Inverse: pull a value for each output pixel → every output pixel is defined.

*Why horizontal but ❌vertical flip on ImageNet?*
- Gravity → natural scenes are not up/down symmetric. They are ~left/right symmetric.
- Aerial, satellite, microscopy images: no canonical orientation → vertical flips & 90° rotations are valid.

*Rotation side effects?*
- Exposed corners → constant fill creates artificial borders. Model can learn "border = augmented".
- Reflect padding or a crop after rotation reduces this.
```

&nbsp;

#### Random Resized Crop
- **What**: Crop w/ random area & aspect ratio → resize to fixed $S\times S$. {cite:p}`szegedy2015going`
- **Why**: Scale, translation & aspect invariance in 1 op. Default ImageNet training augmentation.
- **How**:
    1. Sample area fraction $s\sim U(0.08,1)$ & aspect ratio $r\in[\frac{3}{4},\frac{4}{3}]$ (log-uniform in torchvision).
    2. Box fits → place it uniformly at random. Else retry (10 tries) → fallback: center crop.
    3. Resize box → $S\times S$.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $S$: Output side.
    - $[s_\text{min},s_\text{max}]$: Area-fraction range, default $[0.08,1]$.
    - $[r_\text{min},r_\text{max}]$: Aspect-ratio range, default $[3/4,4/3]$.
- Misc:
    - $w,h$: Crop width & height.

Process:
1. $s\sim U(s_\text{min},s_\text{max})$, $\log r\sim U(\log r_\text{min},\log r_\text{max})$.
2. Box size:

    $$
    w=\sqrt{sHW\,r},\qquad h=\sqrt{sHW/r}
    $$

    - $wh=sHW$: area fraction $s$. $w/h=r$: aspect $r$.
3. Accept if $w\le W$ & $h\le H$ → top-left corner uniform over valid positions.
```

````{important} Code
:class: dropdown
```python
import math
import random
import torch
import torch.nn.functional as F

class RandomResizedCrop:
    def __init__(self, S, scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3), tries=10):
        self.S, self.scale, self.ratio, self.tries = S, scale, ratio, tries

    def box(self, H, W):
        log_r = (math.log(self.ratio[0]), math.log(self.ratio[1]))
        for _ in range(self.tries):
            area = random.uniform(*self.scale) * H * W
            r = math.exp(random.uniform(*log_r))          ## log-uniform -> r and 1/r equally likely
            w, h = round(math.sqrt(area * r)), round(math.sqrt(area / r))
            if 0 < w <= W and 0 < h <= H:                 ## must fit, else resample
                return random.randint(0, H - h), random.randint(0, W - w), h, w
        ## fallback (as torchvision): center crop, aspect clamped into the allowed range
        in_r = W / H
        if in_r < self.ratio[0]:
            w, h = W, round(W / self.ratio[0])
        elif in_r > self.ratio[1]:
            w, h = round(H * self.ratio[1]), H
        else:
            w, h = W, H
        return (H - h) // 2, (W - w) // 2, h, w

    def __call__(self, X):                                ## X: (C, H, W) float
        y0, x0, h, w = self.box(*X.shape[-2:])
        crop = X[:, y0:y0 + h, x0:x0 + w]
        ## any crop shape -> S x S: aspect distortion is part of the augmentation
        return F.interpolate(crop[None], size=(self.S, self.S), mode="bilinear",
                             align_corners=False, antialias=True)[0]

## Example
print(RandomResizedCrop(224)(torch.rand(3, 300, 400)).shape)   ## torch.Size([3, 224, 224])
```
````

```{attention} Q&A
:class: dropdown
*Why a minimum area of 8%?*
- Strong zoom → model must recognize from parts.
- Lower bound = strength knob. Raise it when the model underfits.

*What is the train-test resolution discrepancy?*
- Train: crop of avg area < 1 → resized to $S$ → objects look larger.
- Test: center crop of a resized image at the same $S$ → objects look smaller.
- → Train & test object scales differ. Fixes: test at a higher resolution, or fine-tune at the test resolution {cite:p}`touvron2019fixing`.
    - ResNet-50 trained at 224: test at 224 → 77.0% top-1. Test at 288, no fine-tuning → 78.4%.

*RRC vs CIFAR's pad-and-crop?*
- Pad-and-crop: translation only, scale fixed.
- RRC: translation + scale + aspect. Too aggressive for 32×32 images: 8% of 32² ≈ 9×9 px.
```

&nbsp;

### Photometric
- **What**: Transforms that change pixel values, ❌positions.
- **Why**: Identity does not depend on lighting, white balance, camera, sensor noise, or compression.
- **How**:
    - **Color jitter**: random brightness, contrast, saturation, hue.
    - **Random grayscale**: drop color w/ prob $p$.
    - **Blur, noise, JPEG artifacts**: simulate optics, sensor & compression.
    - **Solarize, posterize, equalize, autocontrast**: image-editing ops used by [policy-based methods](#autoaugment).

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $g(\mathbf{X})$: Grayscale image ($0.299R+0.587G+0.114B$), broadcast to $C$ channels.
    - $\bar{g}$: Mean of $g(\mathbf{X})$ over all pixels.
    - $\beta_b,\beta_c,\beta_s,\beta_h$: Jitter strengths (brightness, contrast, saturation, hue).
    - $b,c,s$: Sampled factors.
    - $\Delta$: Sampled hue shift (fraction of the hue circle).

Definition (as in torchvision `ColorJitter`; clip result to $[0,1]$):

$$
\begin{align*}
\text{Brightness: }&\mathbf{X}'=b\,\mathbf{X}, & b&\sim U(\max(0,1-\beta_b),\,1+\beta_b)\\
\text{Contrast: }&\mathbf{X}'=c\,\mathbf{X}+(1-c)\,\bar{g}, & c&\sim U(\max(0,1-\beta_c),\,1+\beta_c)\\
\text{Saturation: }&\mathbf{X}'=s\,\mathbf{X}+(1-s)\,g(\mathbf{X}), & s&\sim U(\max(0,1-\beta_s),\,1+\beta_s)\\
\text{Hue: }&H\text{-channel (HSV)}\leftarrow H+\Delta \pmod 1, & \Delta&\sim U(-\beta_h,\beta_h),\ \beta_h\le0.5
\end{align*}
$$

- Contrast & saturation = blend toward a gray target. Factor > 1 → extrapolate away from it.
```

```{attention} Q&A
:class: dropdown
*Why apply jitter ops in random order?*
- Ops do not commute (e.g., clip after brightness changes contrast's input).
- Random order → more diverse outputs. torchvision `ColorJitter` permutes the 4 ops per call.

*When does photometric augmentation hurt?*
- Color/brightness carries the label (medical stains, ripeness, day vs night).
- Pretrained features already invariant → small gain, extra CPU.

*Supervised vs self-supervised: same strength?*
- ❌ Contrastive SSL gains from much stronger color distortion than supervised training → [Contrastive Views](#contrastive-views).
```

&nbsp;

#### PCA Color Augmentation
- **What**: Add random multiples of the RGB principal components to every pixel of an image. {cite:p}`krizhevsky2012imagenet`
- **Why**: Identity is invariant to illumination intensity & color. Shift along the data's main color axes, ❌arbitrary axes.
- **How**:
    1. Once: PCA on the 3×3 covariance of all train RGB pixels → eigvecs & eigvals.
    2. Per image load: draw 3 Gaussian coefficients → 1 RGB offset → add it to every pixel.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\mathbf{p}_q\in\mathbb{R}^3$: $q$-th eigenvector of the RGB pixel covariance.
    - $\nu_q$: $q$-th eigenvalue.
    - $a_q$: Random coefficient.

Definition (for every pixel $(u,v)$):

$$
X'_{:,u,v}=X_{:,u,v}+\left[\mathbf{p}_1,\mathbf{p}_2,\mathbf{p}_3\right]\left[a_1\nu_1,\;a_2\nu_2,\;a_3\nu_3\right]^T,\qquad a_q\sim\mathcal{N}(0,0.1^2)
$$

- $a_q$: drawn once per image per load → same offset for all pixels.
- Original paper scales by $\nu_q$ (variance), ❌$\sqrt{\nu_q}$ (std) → offset magnitude depends on the pixel scale.
```

````{important} Code
:class: dropdown
```python
import torch

class PCALighting:
    def __init__(self, std=0.1):
        self.std = std

    def fit(self, X):                                 ## X: (m, 3, H, W) train images
        pix = X.permute(0, 2, 3, 1).reshape(-1, 3)    ## every pixel = one RGB sample
        self.nu, self.p = torch.linalg.eigh(torch.cov(pix.T))   ## eigvals (3,), eigvecs as columns (3, 3)
        return self

    def __call__(self, X):                            ## X: (3, H, W), one image
        a = torch.randn(3) * self.std                 ## new draw per image, per load
        shift = self.p @ (a * self.nu)                ## (3,): ONE RGB offset for the whole image
        return X + shift.view(3, 1, 1)

## Example
X = torch.rand(16, 3, 8, 8)
out = PCALighting().fit(X)(X[0])
print(out.shape, (out - X[0]).std(dim=(1, 2)))       ## torch.Size([3, 8, 8]) ~[0, 0, 0] -> constant shift per channel
```
````

```{attention} Q&A
:class: dropdown
*Did it help?*
- AlexNet: top-1 error ⬇️ by > 1%.

*PCA lighting vs color jitter?*
- PCA: data-adaptive axes, 1 global offset.
- Jitter: hand-set axes (brightness, saturation, hue), multiplicative.
- Modern recipes use jitter or policy-based methods. PCA lighting is mostly historical.
```

&nbsp;

### Cutout
- **What**: Set 1 random square patch of the input to 0. {cite:p}`devries2017improved`
- **Why**: Model over-relies on the most discriminative region → fails when that region is occluded.
    - Mask it at random → model must use the full context.
- **How**:
    1. Sample the patch center uniformly over the image.
    2. Mask an $L\times L$ square, clipped at the border → patch may lie partly outside.
    3. Apply after normalization → 0 = dataset mean.
    - $L$: 16 (CIFAR-10), 8 (CIFAR-100), 20 (SVHN).

````{important} Code
:class: dropdown
```python
import torch

def rand_box(H, W, h, w):
    ## center uniform over the image; box clipped at the border -> can be smaller than h x w
    cy, cx = torch.randint(H, (1,)).item(), torch.randint(W, (1,)).item()
    y0, x0 = cy - h // 2, cx - w // 2                     ## unclipped corner; far edge = corner + size
    return max(y0, 0), min(y0 + h, H), max(x0, 0), min(x0 + w, W)

class Cutout:
    def __init__(self, L):
        self.L = L

    def __call__(self, X):                        ## X: (C, H, W), already normalized -> 0 = mean
        y0, y1, x0, x1 = rand_box(*X.shape[-2:], self.L, self.L)
        X = X.clone()
        X[:, y0:y1, x0:x1] = 0                    ## same hole in all channels
        return X

## Example: 8x8 image, L=4
out = Cutout(4)(torch.ones(3, 8, 8))
print((out[0] == 0).sum().item())                ## 16 if the box is fully inside, fewer if clipped
```
````

```{attention} Q&A
:class: dropdown
*Cutout vs [Dropout](../dl/foundation/module.md#dropout)?*
- Dropout: independent units → neighbors carry nearly the same info (spatial correlation) → info survives.
- Cutout: 1 contiguous region → info in that region is gone → real occlusion.

*Effect of letting the patch fall partly outside?*
- Masked area varies → some images stay ~intact → train distribution stays closer to clean test images.

*Cons?*
- Masked pixels carry no information → part of each image is wasted → [CutMix](#cutmix) fills them.
- Patch can cover the whole object on small images → label noise.
```

&nbsp;

#### Random Erasing
- **What**: Cutout w/ random probability, area, aspect ratio, & random-value fill. {cite:p}`zhong2020random`
- **Why**: Same goal as Cutout (occlusion robustness). More diverse masks.
- **How**: W/ prob $p=0.5$:
    1. Area fraction $\sim U(0.02,0.4)$. Aspect ratio $\sim U(0.3,1/0.3)$.
    2. Resample the box until it lies fully inside the image.
    3. Fill each erased pixel w/ random values.

```{attention} Q&A
:class: dropdown
*Random Erasing vs Cutout?*
- Developed concurrently (both 2017).
- Random Erasing: random apply prob, random rectangle shape, box always inside, noise fill.
- Cutout: always applied, fixed square, may clip at border, mean fill.
- torchvision `RandomErasing` default fill = 0, ❌random values.
```

&nbsp;

### Mixup
- **What**: Train on convex combinations of 2 samples & their targets. {cite:p}`zhang2018mixup`
- **Why**:
    - *Why do we need it?*
        - ERM memorizes → overconfident, sharp transitions between classes.
        - Label-preserving transforms explore only around a single sample.
    - *Why does it work?*
        - Forces ~linear behavior between training samples → smoother decision boundary.
        - Soft targets → confidence ⬇️ → regularization & better calibration.
- **How**:
    1. $\lambda\sim\text{Beta}(\alpha,\alpha)$.
    2. Pair each sample w/ a partner from a shuffled copy of the same batch.
    3. Mix inputs & targets w/ the same $\lambda$ → CE w/ the soft target.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\alpha$: Beta concentration.
- Misc:
    - $\mathbf{p}$: Model's predicted class probabilities on $\tilde{\mathbf{X}}$.

Definition:

$$
\tilde{\mathbf{X}}=\lambda\mathbf{X}_i+(1-\lambda)\mathbf{X}_{i'},\qquad\tilde{\mathbf{y}}=\lambda\mathbf{y}_i+(1-\lambda)\mathbf{y}_{i'},\qquad\lambda\sim\text{Beta}(\alpha,\alpha)
$$

Objective:

$$
-\sum_{k=1}^{K}\tilde{y}_k\log p_k=\lambda\,\text{CE}(\mathbf{y}_i,\mathbf{p})+(1-\lambda)\,\text{CE}(\mathbf{y}_{i'},\mathbf{p})
$$

- $\alpha\to0$ → $\lambda\in\{0,1\}$ → plain ERM.
- $\alpha=1$ → $\lambda\sim U(0,1)$.
```

```{tip} Derivation
:class: dropdown
*Why does mixing labels equal mixing losses?*
1. CE is linear in the target: $\text{CE}(\mathbf{y},\mathbf{p})=-\sum_ky_k\log p_k$.
2. Substitute $\tilde{\mathbf{y}}=\lambda\mathbf{y}_i+(1-\lambda)\mathbf{y}_{i'}$:

    $$
    -\sum_k\left(\lambda y_{i,k}+(1-\lambda)y_{i',k}\right)\log p_k=\lambda\,\text{CE}(\mathbf{y}_i,\mathbf{p})+(1-\lambda)\,\text{CE}(\mathbf{y}_{i'},\mathbf{p})
    $$

3. → Implement either way: soft target, or 2 hard-label losses weighted by $\lambda$.
```

````{important} Code
:class: dropdown
```python
import torch

class Mixup:
    def __init__(self, alpha=0.2):
        self.beta = torch.distributions.Beta(alpha, alpha)

    def __call__(self, X, Y):                    ## X: (B, C, H, W); Y: (B, K) one-hot or soft
        lam = self.beta.sample().item()          ## 1 lambda per batch, as in the official code
        perm = torch.randperm(len(X))            ## partner = shuffled copy of the SAME batch
        X_mix = lam * X + (1 - lam) * X[perm]
        Y_mix = lam * Y + (1 - lam) * Y[perm]    ## SAME lambda for inputs & targets
        return X_mix, Y_mix

## Example
X = torch.randn(4, 3, 8, 8)
Y = torch.eye(3)[torch.tensor([0, 1, 2, 0])]    ## one-hot, K=3
X_mix, Y_mix = Mixup(alpha=1.0)(X, Y)
print(X_mix.shape, Y_mix.sum(dim=1))             ## torch.Size([4, 3, 8, 8]) tensor([1., 1., 1., 1.])
## loss: F.cross_entropy(logits, Y_mix) accepts class probabilities as targets
```
````

```{attention} Q&A
:class: dropdown
*How to set $\alpha$?*
- ImageNet: $\alpha\in[0.1,0.4]$ improves over ERM. Large $\alpha$ → underfitting.
- $\alpha$⬆️ → training error ⬆️, generalization gap ⬇️.
- Larger models & longer training gain the most.

*Mixed images are unrealistic. Why does it help?*
- Goal is not realism. Goal = constrain $f_\theta$ between samples.
- Target mixes w/ the input → ❌same-label (invariance) assumption → ✅linear target-interpolation assumption instead.

*Effect on calibration?*
- Mixup-trained nets are better calibrated {cite:p}`thulasidasan2019mixup` → lower [ECE](../ml/eval.md#ece).

*Mixup vs [label smoothing](../ml/obj.md#label-smoothing)?*
- Both soften targets.
- Label smoothing: mix w/ the uniform distribution, input unchanged.
- Mixup: mix w/ 1 specific other sample, input mixed too → data-dependent softening.

*When does it hurt?*
- Target does not mix linearly w/ the input. Counting: blend images w/ 3 & 5 objects → 8 faint objects visible, target says 4.
- Large $\alpha$ on small models / short schedules → underfitting.
```

&nbsp;

#### CutMix
- **What**: Paste a random box from image $i'$ into image $i$. Mix targets by box area. {cite:p}`yun2019cutmix`
- **Why**:
    - Mixup → ghost overlays, unnatural local statistics.
    - Cutout → masked pixels carry no information.
    - CutMix → every pixel is real + occlusion robustness + better localization.
- **How**:
    1. $\lambda\sim\text{Beta}(1,1)=U(0,1)$.
    2. Box w/ area fraction $1-\lambda$: sides $W\sqrt{1-\lambda}$ & $H\sqrt{1-\lambda}$, center uniform, clipped at the border.
    3. Recompute $\lambda$ from the clipped box area → mix targets.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $M\in\{0,1\}^{H\times W}$: Binary mask. 0 inside the box.
    - $\odot$: Element-wise product, broadcast over channels.

Definition:

$$
\tilde{\mathbf{X}}=M\odot\mathbf{X}_i+(1-M)\odot\mathbf{X}_{i'},\qquad\tilde{\mathbf{y}}=\lambda\mathbf{y}_i+(1-\lambda)\mathbf{y}_{i'},\qquad\lambda=\frac{\sum_{u,v}M_{u,v}}{HW}
$$

- $\lambda$ = share of pixels from $\mathbf{X}_i$, after clipping.
```

````{important} Code
:class: dropdown
```python
import torch

## Reuses rand_box from Cutout above
class CutMix:
    def __init__(self, alpha=1.0):
        self.beta = torch.distributions.Beta(alpha, alpha)

    def __call__(self, X, Y):                            ## X: (B, C, H, W); Y: (B, K)
        B, _, H, W = X.shape
        lam = self.beta.sample().item()
        perm = torch.randperm(B)
        r = (1 - lam) ** 0.5                             ## side ratio -> box area = (1 - lam) * H * W
        y0, y1, x0, x1 = rand_box(H, W, int(H * r), int(W * r))
        X = X.clone()
        X[:, :, y0:y1, x0:x1] = X[perm, :, y0:y1, x0:x1] ## paste the partner's pixels into the box
        lam = 1 - (y1 - y0) * (x1 - x0) / (H * W)        ## recompute from the CLIPPED box
        return X, lam * Y + (1 - lam) * Y[perm]

## Example
X = torch.randn(4, 3, 8, 8)
Y = torch.eye(3)[torch.tensor([0, 1, 2, 0])]
X_cm, Y_cm = CutMix()(X, Y)
print(X_cm.shape, Y_cm.sum(dim=1))                   ## torch.Size([4, 3, 8, 8]) tensor([1., 1., 1., 1.])
```
````

```{dropdown} Table: Occlusion & Mixing Methods
| | Region | Fill | Target | Pixels wasted |
|:--|:--|:--|:--|:--|
| Cutout | Fixed square, may clip | Mean (0) | Unchanged | ✅ |
| Random Erasing | Random rectangle, inside | Random values | Unchanged | ✅ |
| Mixup | Whole image | Blend of 2 images | Mixed by $\lambda$ | ❌ |
| CutMix | Random box | Other image's pixels | Mixed by area | ❌ |
```

```{attention} Q&A
:class: dropdown
*Why recompute $\lambda$ after clipping?*
- Clipping shrinks the box → target must match the true pixel share.

*Failure mode?*
- Box lands on background of $\mathbf{X}_{i'}$ → target says "30% class of $i'$", no such object visible → label noise.
- Box covers the whole object of $\mathbf{X}_i$ → same problem, reversed.

*Mixup or CutMix?*
- CutMix paper: CutMix > Mixup on ImageNet classification & weakly-supervised localization ← natural local statistics.
- Modern ViT recipes (timm, DeiT) pick 1 of the 2 at random per batch.
```

&nbsp;

### AutoAugment
- **What**: Search for the augmentation policy w/ RL on a proxy task. {cite:p}`cubuk2019autoaugment`
- **Why**: Best augmentations are dataset-specific. Manual design = slow trial & error.
- **How**:
    1. Policy = 5 sub-policies. Sub-policy = 2 ops in sequence. Op = (type, apply prob, magnitude).
    2. RNN controller samples a policy → train a small child model on a reduced dataset → val accuracy = reward → PPO update.
    3. Final policy = concatenation of the best 5 policies → 25 sub-policies.
    4. Train the real model: per image, pick 1 sub-policy uniformly → apply its 2 ops w/ their probs.

```{note} Math
:class: dropdown
Search space:
- 16 op types × 10 magnitudes × 11 probabilities per op.
- Sub-policy (2 ops): $(16\cdot10\cdot11)^2$ choices.
- Policy (5 sub-policies): $(16\cdot10\cdot11)^{10}\approx2.9\times10^{32}$ choices.
```

```{attention} Q&A
:class: dropdown
*Cons?*
- Search cost: ~15K GPU hours for ImageNet.
- Proxy mismatch: policy tuned on a small model & dataset. Best strength depends on model & data size → suboptimal on the real task.

*Pros?*
- Policies transfer to other datasets of the same domain.
- Showed that augmentation choice alone moves SOTA.
```

&nbsp;

#### RandAugment
- **What**: Apply $N$ ops sampled uniformly from 14, all at 1 shared magnitude $M$. {cite:p}`cubuk2020randaugment`
- **Why**: AutoAugment search is expensive & runs on a proxy.
    - Best strength grows w/ model size & dataset size → tune on the target task directly.
- **How**:
    1. Per image: sample $N$ ops uniformly w/ replacement → apply each at magnitude $M$.
    2. Grid-search 2 integers $(N,M)$ on the target task → search space ~$10^2$.
    - Ops: Identity, AutoContrast, Equalize, Rotate, Solarize, Color, Posterize, Contrast, Brightness, Sharpness, ShearX/Y, TranslateX/Y.
    - torchvision default: $N=2$, $M=9$ on a 0–30 scale.

````{important} Code
:class: dropdown
```python
import math
import random
import torch
from torchvision.transforms.v2 import functional as F

def _sign():
    return random.choice([-1, 1])                 ## geometric & enhance ops go either direction

## Subset of the 14 ops. f = M / 30 in [0, 1]; ranges follow torchvision's RandAugment
OPS = {
    "Identity":     lambda x, f: x,
    "AutoContrast": lambda x, f: F.autocontrast(x),
    "Equalize":     lambda x, f: F.equalize(x),
    "Rotate":       lambda x, f: F.rotate(x, _sign() * 30.0 * f),
    "ShearX":       lambda x, f: F.affine(x, angle=0.0, translate=[0, 0], scale=1.0,
                                          shear=[_sign() * math.degrees(math.atan(0.3 * f)), 0.0]),
    "TranslateY":   lambda x, f: F.affine(x, angle=0.0, translate=[0, int(_sign() * 0.45 * f * x.shape[-2])],
                                          scale=1.0, shear=[0.0, 0.0]),
    "Brightness":   lambda x, f: F.adjust_brightness(x, 1 + _sign() * 0.9 * f),
    "Contrast":     lambda x, f: F.adjust_contrast(x, 1 + _sign() * 0.9 * f),
    "Solarize":     lambda x, f: F.solarize(x, 255.0 * (1 - f)),
    "Posterize":    lambda x, f: F.posterize(x, 8 - round(4 * f)),
}

class RandAugment:
    def __init__(self, N=2, M=9):
        self.N, self.f = N, M / 30                ## ONE magnitude shared by every op

    def __call__(self, x):                        ## x: uint8 (C, H, W)
        for name in random.choices(list(OPS), k=self.N):   ## uniform, with replacement
            x = OPS[name](x, self.f)
        return x

## Example
img = torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8)
print(RandAugment()(img).shape)                   ## torch.Size([3, 32, 32])
```
````

```{attention} Q&A
:class: dropdown
*Why can 2 integers replace a $10^{32}$ search?*
- Uniform op choice matches AutoAugment accuracy → the exact op/prob combination matters little.
- The knob that matters = overall strength → $M$ (& $N$).

*How does the best $M$ scale?*
- Larger model → larger optimal $M$ → bigger nets need more regularization.
- Larger training set → larger optimal $M$. Counterintuitive. One hypothesis: strong distortion on a small dataset → too little clean signal.
- → A strength tuned on a small proxy (AutoAugment) is too weak for the full task.
```

&nbsp;

#### TrivialAugment
- **What**: Apply 1 op, sampled uniformly, at a strength sampled uniformly from $\{0,\dots,30\}$. {cite:p}`muller2021trivialaugment`
- **Why**: RandAugment still needs a grid search over $(N,M)$. TrivialAugment has 0 hyperparams & matches or beats it.
- **How**: Per image: op $\sim U(\text{ops})$, strength $\sim U\{0,\dots,30\}$ → apply → done.

````{important} Code
:class: dropdown
```python
import random

## Reuses OPS from RandAugment above (torchvision's TrivialAugmentWide uses wider op ranges)
class TrivialAugment:
    def __call__(self, x):                        ## x: uint8 (C, H, W)
        name = random.choice(list(OPS))           ## exactly 1 op
        m = random.randint(0, 30)                 ## fresh strength per image -> nothing to tune
        return OPS[name](x, m / 30)
```
````

```{attention} Q&A
:class: dropdown
*Why does random strength work?*
- Uniform strength covers the whole range → right strength is in the mix for every model size.
- Training distribution = unweighted average of the $|\text{ops}|$ single-op distributions → no stacked, compounding distortions.

*Is it a special case of RandAugment?*
- ❌ RandAugment fixes $M$ for all images. TrivialAugment resamples strength per image → different training distribution.
```

&nbsp;

### Test-Time Augmentation
- **What**: Average predictions over several augmented views of 1 test input.
- **Why**: 1 view = 1 noisy draw of the model's output under nuisance → averaging ⬇️ variance. Ensemble w/o extra training.
- **How**:
    1. Build $R$ views w/ train-time transforms (flips, crops, scales).
    2. Average the predicted probabilities → predict.
    - AlexNet 10-crop: 4 corner + 1 center 224² crops, + their horizontal flips.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $R$: #views.
- Misc:
    - $t_r$: $r$-th test-time transform.

Inference:

$$
\hat{\mathbf{p}}(\mathbf{X})=\frac{1}{R}\sum_{r=1}^{R}\operatorname{softmax}\left(f_\theta(t_r(\mathbf{X}))\right)
$$
```

````{important} Code
:class: dropdown
```python
import torch

class FlipTTA:
    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def __call__(self, X):                                  ## X: (B, C, H, W)
        views = [X, X.flip(-1)]                             ## identity + horizontal flip (seen in training)
        probs = [self.model(v).softmax(-1) for v in views]
        return torch.stack(probs).mean(0)                   ## average PROBABILITIES over views

## Example
model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 5))
print(FlipTTA(model)(torch.randn(2, 3, 8, 8)).sum(-1))     ## tensor([1., 1.])
```
````

```{attention} Q&A
:class: dropdown
*Cost?*
- $R\times$ inference compute & latency.

*Is the averaged prediction better than a single view?*
- $p_r$: probability of the true class from view $r$.
- $-\log$ is convex → Jensen: $-\log\left(\frac{1}{R}\sum_rp_r\right)\le\frac{1}{R}\sum_r(-\log p_r)$.
- → Log loss of the averaged prediction ≤ MEAN log loss of the views.
- ❌Guarantee vs the best view (e.g., the clean identity view). Bad extra views can make TTA worse than no TTA.

*Which transforms?*
- Only those used in training. Unseen transforms → off-distribution views → worse.

*Dense tasks (segmentation)?*
- Invert each geometric transform on the output (e.g., flip the mask back) → average per pixel.
```

&nbsp;

### Contrastive Views
- **What**: 2 random augmentations of 1 image = a positive pair. Views of other images = negatives. {cite:p}`chen2020simple`
- **Why**: No labels → the augmentation set $\mathcal{T}$ defines the task.
    - Representation becomes invariant to exactly what $\mathcal{T}$ changes.
- **How**:
    1. Per image: 2 independent draws $t,t'\sim\mathcal{T}$ → 2 views.
    2. Encoder → embeddings → [InfoNCE](../ml/obj.md#infonce) pulls the pair together, pushes other images apart.
    - SimCLR $\mathcal{T}$: random resized crop + flip → color jitter ($p=0.8$) → grayscale ($p=0.2$) → Gaussian blur ($p=0.5$).

````{important} Code
:class: dropdown
```python
import torch
from torchvision.transforms import v2

def simclr_transform(S, s=1.0):                   ## s: color-distortion strength
    k = int(0.1 * S) // 2 * 2 + 1                 ## blur kernel ~10% of the side, forced odd
    return v2.Compose([
        v2.RandomResizedCrop(S, antialias=True),
        v2.RandomHorizontalFlip(),
        v2.RandomApply([v2.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)], p=0.8),
        v2.RandomGrayscale(p=0.2),
        v2.RandomApply([v2.GaussianBlur(k, sigma=(0.1, 2.0))], p=0.5),
    ])

class TwoViews:
    def __init__(self, transform):
        self.t = transform

    def __call__(self, img):
        ## 2 INDEPENDENT draws -> positive pair; their differences = what the encoder must ignore
        return self.t(img), self.t(img)

## Example
img = torch.randint(0, 256, (3, 96, 96), dtype=torch.uint8)
a, b = TwoViews(simclr_transform(64))(img)
print(a.shape, b.shape)                           ## torch.Size([3, 64, 64]) torch.Size([3, 64, 64])
```
````

```{attention} Q&A
:class: dropdown
*Which augmentations matter most?*
- Crop + color distortion composed. Neither alone is enough.
- Crop alone → crops of 1 image share a color histogram → the net matches histograms (shortcut) → weak features.

*Same augmentation strength as supervised training?*
- ❌ Stronger color distortion improves contrastive features. It does not help, or hurts, supervised training.
- ← Supervised labels already define the task. In SSL, augmentation is the only task signal.

*Risk?*
- Invariance to a label-relevant factor → features lose it. Color-invariant features → bad for flower species or bird plumage.

*Is there a biological analog?*
- Successive retinal views of 1 object (eye, head & object motion) act as natural positive pairs.
- Temporal continuity gives invariance signal w/o labels. Slow feature analysis formalizes this idea.
```

&nbsp;
