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
# Vision Transformers
How the transformer is adapted to images, & the major ViT families. Image → tokens: [Patch Embedding](image.md#patch-embedding). [Attention](../nlp/attn.md), [position encoding](../nlp/pos.md), [distillation](../nlp/post/kd.md) & the [transformer block](../nlp/transformer.md) have their own pages.

Notations:
- $\mathbf{X}\in\mathbb{R}^{C\times H\times W}$: One image, channel-first
- $C$: #channels
- $H$: Image height (pixels)
- $W$: Image width (pixels)
- $P$: Patch size
- $N=HW/P^2$: #patch tokens
- $D$: Token width (model dim)
- $L$: #encoder layers
- $\mathbf{z}_\ell$: Token sequence after layer $\ell$. Row 0 = [CLS].
- $B$: Batch size
- $K$: #classes

Override: $W$ = image width (❌weight matrix, ❌window). $D$ = model dim (NLP pages: $d$). Both match [Image Input](image.md) & the ViT paper.

&nbsp;

## Plain ViT
- **What**: Single-scale encoder: same #tokens & width in every layer.

&nbsp;

### ViT
- **Name**: Vision Transformer {cite:p}`dosovitskiy2021image`
- **What**: Standard transformer encoder over a sequence of image patches.
- **Why**: Test whether CNN priors are necessary.
    - 2020: attention in vision = add-on to CNNs, or a replacement of parts inside a CNN layout.
    - Transformers scale well in NLP → reuse one on images w/ minimal changes.
- **How**:
    1. [Patch embedding](image.md#patch-embedding): image → $N$ patch tokens + [CLS] + learned 1D positions.
    2. $L$ [pre-norm](../nlp/models.md#pre-norm) encoder layers. ❌Mask → every patch attends to every patch.
    3. LN on the final [CLS] row → classification head.
        - **Pretraining**: MLP head w/ 1 hidden layer.
        - **Fine-tuning**: replace it w/ a zero-initialized $D\times K$ linear layer.
    4. Pretrain on a large dataset (ImageNet-21k, JFT-300M) → fine-tune on a small one, often at higher resolution.

```{note} Math
:class: dropdown
Model:

$$\begin{align*}
\mathbf{z}_0&=\text{PatchEmbed}(\mathbf{X})\in\mathbb{R}^{(N+1)\times D}\\
\mathbf{z}_\ell&=\text{EncoderLayer}_\ell(\mathbf{z}_{\ell-1}),\qquad\ell=1,\dots,L\\
\hat{\mathbf{y}}&=\text{softmax}\left(\text{Head}\left(\text{LN}(\mathbf{z}_L^0)\right)\right)\in\mathbb{R}^{K}
\end{align*}$$

- $\mathbf{z}_L^0$: [CLS] row of the last layer.
- Cost per layer: $\mathcal{O}(N^2D)$ attention + $\mathcal{O}(ND^2)$ projections & MLP.
- $N\propto HW$ → attention cost $\propto(HW)^2$ → quadratic in #pixels.

Training: [cross entropy](../ml/obj.md#cross-entropy) on image labels.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

## Reuses PatchEmbed from cv/image.md
class ViT(nn.Module):
    def __init__(self, img_size=224, P=16, C=3, D=768, L=12, heads=12, K=1000):
        super().__init__()
        self.embed = PatchEmbed(img_size, P, C, D)                  ## (B, N+1, D): [CLS] + patches + positions
        layer = nn.TransformerEncoderLayer(D, heads, 4 * D, activation="gelu",
                                           batch_first=True, norm_first=True)   ## the NLP block, unchanged
        self.encoder = nn.TransformerEncoder(layer, L)
        self.norm = nn.LayerNorm(D)
        self.head = nn.Linear(D, K)

    def forward(self, X):
        z = self.encoder(self.embed(X))                             ## no mask: all patches see all patches
        return self.head(self.norm(z[:, 0]))                        ## classify from the [CLS] row only

## Example: 32x32 image, P=8 -> 16 patches + [CLS]
model = ViT(img_size=32, P=8, D=64, L=2, heads=4, K=10)
print(model(torch.randn(2, 3, 32, 32)).shape)                       ## torch.Size([2, 10])
```
````

```{dropdown} Table: ViT sizes
| | $L$ | $D$ | MLP size | Heads | #Params |
|:--|:--|:--|:--|:--|:--|
| ViT-B | 12 | 768 | 3072 | 12 | 86M |
| ViT-L | 24 | 1024 | 4096 | 16 | 307M |
| ViT-H | 32 | 1280 | 5120 | 16 | 632M |

- Name suffix = $P$: ViT-L/16 → $16\times16$ patches → 196 tokens at 224².
- Pretraining: Adam ($\beta_1=0.9$, $\beta_2=0.999$), batch 4096, weight decay 0.1.
- Best: ViT-H/14 pretrained on JFT-300M → 88.55% ImageNet top-1.
```

```{attention} Q&A
:class: dropdown
*Why does ViT need large pretraining data?*
- Weak [inductive bias](../ml/misc.md#inductive-bias) → it must learn locality from data. Details: [Patch Embedding](image.md#patch-embedding) Q&A.

*What does ViT learn in place of conv priors?*
- Patch projection $E$: top principal components look like basis functions for fine structure inside a patch.
- Position embeddings: learned as 1D, but close patches get similar embeddings → row & column structure is learned from data.
- Attention distance: some heads attend to most of the image already in the lowest layers. Other heads stay local there. Mean distance ⬆️ w/ depth.

*ViT vs ResNet at equal compute?*
- Pretrained on JFT-300M: ViT needs ~2–4× less compute for the same transfer performance.
- Hybrid (CNN feature map → tokens) beats pure ViT at small budgets. The gap vanishes for larger models.

*Why is plain ViT a weak backbone for detection & segmentation?*
- 1 scale ($1/P$) in every layer → ❌feature pyramid.
- Dense tasks need high resolution → attention cost $\propto(HW)^2$.
- → [Swin](#swin), or windowed attention as in [SAM](#sam).

*Are patches really "words"?*
- ❌Fixed vocabulary: a patch token is a continuous vector.
- Neighboring patches are highly redundant; words are not → a very high mask ratio still works in [MAE](#mae).
```

&nbsp;

### DeiT
- **Name**: Data-efficient image Transformers {cite:p}`touvron2021training`
- **What**: ViT trained on ImageNet-1k only, w/ a strong training recipe & a distillation token.
- **Why**: ViT needed 14M–300M images & large compute → out of reach for most labs.
- **How**:
    1. Architecture: DeiT-B = ViT-B. DeiT-S & DeiT-Ti change only width & #heads. Each head stays 64-dim.
    2. Recipe: AdamW, [RandAugment](image.md#randaugment), [Mixup](image.md#mixup), [CutMix](image.md#cutmix), [Random Erasing](image.md#random-erasing), repeated augmentation, stochastic depth. 300 epochs.
    3. **Distillation token**: extra learnable token next to [CLS]. Its own head learns the teacher's top-1 label.
    4. Inference: add the softmax outputs of the 2 heads (late fusion).

```{note} Math
:class: dropdown
Notations:
- IO:
    - $y$: True label.
    - $y_t=\arg\max_c Z_t(c)$: Teacher's hard label.
- Misc:
    - $Z_\text{cls}$: Logits of the [CLS] head.
    - $Z_\text{dist}$: Logits of the distillation-token head.
    - $Z_t$: Teacher logits.
    - $\psi$: Softmax.

Objective (hard-label distillation):

$$
\mathcal{L}=\tfrac{1}{2}\,\text{CE}\big(\psi(Z_\text{cls}),y\big)+\tfrac{1}{2}\,\text{CE}\big(\psi(Z_\text{dist}),y_t\big)
$$

- Soft alternative: [token-level KD](../nlp/post/kd.md#token-level-kd) on $Z_\text{dist}$. Hard works better for DeiT.

Inference:

$$
\hat{y}=\arg\max_c\left[\psi(Z_\text{cls})+\psi(Z_\text{dist})\right]_c
$$
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

## Reuses ViT from #vit above
class DeiT(ViT):
    def __init__(self, **kw):
        super().__init__(**kw)
        D, K = self.head.in_features, self.head.out_features
        self.dist = nn.Parameter(torch.randn(1, 1, D) * 0.02)       ## distillation token; its position folded in
        self.head_dist = nn.Linear(D, K)                            ## 2nd head, own weights

    def forward(self, X):
        z = self.embed(X)                                           ## [CLS] + patches, w/ positions
        z = torch.cat([z[:, :1], self.dist.expand(len(z), -1, -1), z[:, 1:]], dim=1)   ## [CLS, DIST, patches]
        z = self.norm(self.encoder(z))
        return self.head(z[:, 0]), self.head_dist(z[:, 1])

    def loss(self, X, y, teacher_logits):
        z_cls, z_dist = self(X)
        y_t = teacher_logits.argmax(-1)                             ## hard distillation: teacher's top-1 = a label
        return 0.5 * F.cross_entropy(z_cls, y) + 0.5 * F.cross_entropy(z_dist, y_t)

    @torch.no_grad()
    def predict(self, X):
        z_cls, z_dist = self(X)
        return (z_cls.softmax(-1) + z_dist.softmax(-1)).argmax(-1)  ## late fusion of both heads

## Example: 32x32 images, 10 classes, stand-in teacher logits
model = DeiT(img_size=32, P=8, D=64, L=2, heads=4, K=10)
X, y = torch.randn(2, 3, 32, 32), torch.tensor([3, 7])
print(model.loss(X, y, torch.randn(2, 10)).item() > 0, model.predict(X).shape)   ## True torch.Size([2])
```
````

```{dropdown} Table: DeiT results
| | $D$ | Heads | #Params | No distill | ⚗ Distilled |
|:--|:--|:--|:--|:--|:--|
| DeiT-Ti | 192 | 3 | 5M | 72.2 | 74.5 |
| DeiT-S | 384 | 6 | 22M | 79.8 | 81.2 |
| DeiT-B | 768 | 12 | 86M | 81.8 | 83.4 |

- ImageNet-1k top-1 (%) at 224², 300 epochs, $L=12$ for all.
- ⚗: distillation token + hard-label distillation. Teacher: RegNetY-16GF (84M params, 82.9%).
- DeiT-B fine-tuned at 384²: 83.1 → 84.5 w/ ⚗. Longer schedule: up to 85.2.
- DeiT-B: 53 h of pretraining on 1 node w/ 8 GPUs.
```

```{attention} Q&A
:class: dropdown
*Why a separate distillation token, ❌1 head w/ 2 losses?*
- $y$ & $y_t$ can disagree → each token specializes in 1 target.
- The learned tokens differ: cosine 0.06 at input → 0.93 at the last layer (similar, not identical).
- DeiT-B: class token only + hard distillation 83.0 → both tokens 83.4.

*Why hard labels > soft distillation here?*
- Random crops can remove the object: a "cat" image cropped to grass only.
- Teacher sees the same crop → its top-1 label fits the crop better than the dataset label.

*Why a CNN teacher?*
- CNN teacher > transformer teacher for the student, even at similar teacher accuracy.
- Paper's explanation: distillation transfers the CNN's inductive bias. Hypothesis, not proven.
- Student surpasses the teacher: 83.4 (DeiT-B ⚗) vs 82.9 (RegNetY-16GF).

*Is the recipe or the distillation the main gain?*
- Recipe alone: ViT-B from scratch on ImageNet-1k → 81.8%. ViT paper's ImageNet-only ViT was a few % below comparable ResNets.
- Optimizer matters: SGD instead of AdamW → 74.5%.
```

&nbsp;

### Registers
- **What**: Extra learnable tokens appended to the ViT input & discarded at the output. {cite:p}`darcet2024vision`
- **Why**: Large ViTs take over background patch tokens as scratch space.
    - Symptom: ~2% of tokens w/ ~10× norm, in low-information background regions → artifacts in attention & patch feature maps.
    - These tokens drop local info (position, pixels) & store global image info.
    - Seen in supervised (DeiT-III), text-supervised (OpenCLIP) & self-supervised (DINOv2) ViTs. Original DINO is the exception.
- **How**:
    1. After patch embedding, append $R$ learnable tokens. ❌Position embedding.
    2. Run the encoder unchanged.
    3. Drop register outputs. Use [CLS] & patch outputs as before.
    - Default $R=4$ → FLOPs <+2%, params negligible.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\mathbf{r}_1,\dots,\mathbf{r}_R\in\mathbb{R}^{D}$: Register tokens.
- Hyperparams:
    - $R$: #registers.

Forward:

$$
\mathbf{z}_0=\left[\,\text{PatchEmbed}(\mathbf{X});\ \mathbf{r}_1;\ \dots;\ \mathbf{r}_R\,\right]\in\mathbb{R}^{(N+1+R)\times D}
$$

- Output: keep rows $0..N$ of $\mathbf{z}_L$ ([CLS] + patches). Discard the last $R$ rows.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

## Reuses ViT from #vit above
class ViTWithRegisters(ViT):
    def __init__(self, R=4, **kw):
        super().__init__(**kw)
        D = self.head.in_features
        ## random init: identical tokens w/o positions would get identical updates & stay identical
        self.reg = nn.Parameter(torch.randn(1, R, D) * 0.02)

    def forward(self, X):
        z = self.embed(X)                                           ## (B, N+1, D) w/ positions
        z = torch.cat([z, self.reg.expand(len(z), -1, -1)], dim=1)  ## registers: no position embedding
        z = self.norm(self.encoder(z))[:, : -self.reg.shape[1]]     ## drop registers at the output
        return self.head(z[:, 0]), z[:, 1:]                         ## class logits, clean patch features

## Example
model = ViTWithRegisters(R=4, img_size=32, P=8, D=64, L=2, heads=4, K=10)
logits, feats = model(torch.randn(2, 3, 32, 32))
print(logits.shape, feats.shape)                                    ## torch.Size([2, 10]) torch.Size([2, 16, 64])
```
````

```{attention} Q&A
:class: dropdown
*When do the artifacts appear?*
- Only in large models: of Tiny → giant, only the 3 largest (ViT-L & up).
- Only after ~1/3 of training. They separate from other tokens around the middle layers.
- → Spare capacity + long training → model recycles redundant patches for global computation.

*Why doesn't [CLS] already fill this role?*
- [CLS] is present in all affected models, yet the artifacts still appear → 1 extra slot is not enough.

*Pros?*
- Smooth attention & feature maps → better dense prediction & unsupervised object discovery.
- Linear-probe classification: unchanged or slightly ⬆️. OpenCLIP zero-shot ImageNet: unchanged.

*How does this relate to LLM [attention sinks](../nlp/attn.md#attention-sink)?*
- Both: low-content tokens take on a role unrelated to their content. Fix in both: give the model explicit slots.
- Sinks absorb attention mass. Registers hold global information. Related phenomena, not shown to be 1 mechanism.
```

&nbsp;

## Hierarchical ViT
- **What**: Multi-stage ViT: #tokens ⬇️ & width ⬆️ stage by stage, like a CNN feature pyramid.

&nbsp;

### Swin
- **What**: Hierarchical ViT w/ self-attention inside local windows that shift between consecutive layers. {cite:p}`liu2021swin`
- **Why**: Plain ViT is a poor general backbone.
    - Global attention cost $\propto(\#\text{tokens})^2$ → infeasible at detection & segmentation resolutions.
    - 1 scale → ❌multi-scale feature maps for FPN / U-Net-style heads.
    - Visual objects vary widely in scale. Words do not.
- **How**:
    1. Patchify w/ $P=4$ → $\frac{H}{4}\times\frac{W}{4}$ tokens, width $D$.
    2. Each stage = pairs of blocks. Block 1: **W-MSA** inside non-overlapping $M\times M$ windows. Block 2: **SW-MSA**: windows shifted by $\lfloor M/2\rfloor$ → links neighboring windows.
    3. **Patch merging** between stages: concat each 2×2 group of tokens → linear → resolution ½, width ×2.
    4. 4 stages → strides 4, 8, 16, 32 → same pyramid as a ResNet.
    - **Relative position bias** inside each window. ❌Absolute position embedding.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $M$: Window side (#tokens). Default 7.
- Misc:
    - $h\times w$: Token grid of the current stage.
    - $d_k$: Dim per head.
    - $G\in\mathbb{R}^{M^2\times M^2}$: Relative position bias per head (paper's $B$; renamed ← $B$ = batch size here).
    - $\hat{G}\in\mathbb{R}^{(2M-1)\times(2M-1)}$: Learned bias table per head.
    - $U\in\mathbb{R}^{4D'\times2D'}$: Patch-merging projection. $D'$ = width of the current stage.
- Paper's $C$ (stage-1 width) = $D$ here.

Forward (1 window, 1 head):

$$
\text{Attn}(Q,K,V)=\text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+G\right)V,\qquad Q,K,V\in\mathbb{R}^{M^2\times d_k}
$$

- $G_{ab}=\hat{G}\left[\Delta u_{ab}+M-1,\ \Delta v_{ab}+M-1\right]$.
    - $\Delta u_{ab},\Delta v_{ab}\in[-M+1,M-1]$: Row & column offset between window slots $a$ & $b$.
- Blocks alternate W-MSA & SW-MSA. The rest of each block is the standard pre-norm block.

Patch merging:

$$
\mathbf{z}'_{(u,v)}=\left[\mathbf{z}_{(2u,2v)};\ \mathbf{z}_{(2u+1,2v)};\ \mathbf{z}_{(2u,2v+1)};\ \mathbf{z}_{(2u+1,2v+1)}\right]U
$$

- $(u,v)$: Token grid row & column.

Cost (softmax omitted):

$$
\Omega(\text{MSA})=4hwD'^2+2(hw)^2D',\qquad \Omega(\text{W-MSA})=4hwD'^2+2M^2hwD'
$$

- $4hwD'^2$: Q, K, V & output projections.
- $2(hw)^2D'$ vs $2M^2hwD'$: each token attends to $M^2$ tokens, ❌all $hw$ → linear in #tokens for fixed $M$.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwinAttention(nn.Module):
    def __init__(self, D, heads, M=7, shift=False):
        super().__init__()
        self.h, self.M, self.s = heads, M, (M // 2 if shift else 0)
        self.qkv, self.out = nn.Linear(D, 3 * D), nn.Linear(D, D)
        self.G_hat = nn.Parameter(torch.zeros(heads, 2 * M - 1, 2 * M - 1))   ## bias table per head
        u, v = torch.meshgrid(torch.arange(M), torch.arange(M), indexing="ij")
        u, v = u.flatten(), v.flatten()                                        ## row/col of each window slot
        self.register_buffer("du", u[:, None] - u[None] + M - 1)              ## offsets shifted to >= 0
        self.register_buffer("dv", v[:, None] - v[None] + M - 1)

    def windows(self, x):                                                      ## (B, h, w, c) -> (B*#win, M*M, c)
        B, h, w, c = x.shape
        M = self.M
        x = x.view(B, h // M, M, w // M, M, c).transpose(2, 3)
        return x.reshape(-1, M * M, c)

    def forward(self, x):                                                      ## x: (B, h, w, D), h & w divisible by M
        B, h, w, D = x.shape
        M, s = self.M, self.s
        if s:                                                                  ## cyclic shift: new windows straddle old borders
            x = x.roll((-s, -s), dims=(1, 2))
        q, k, v = (t.unflatten(-1, (self.h, -1)).transpose(1, 2)
                   for t in self.qkv(self.windows(x)).chunk(3, dim=-1))        ## (B*#win, heads, M*M, d_k)
        bias = self.G_hat[:, self.du, self.dv]                                 ## (heads, M*M, M*M)
        if s:                                                                  ## mask pairs that were NOT neighbors before the roll
            reg = torch.zeros(h, w)
            cuts = (slice(0, -M), slice(-M, -s), slice(-s, None))
            for a, rows in enumerate(cuts):
                for b, cols in enumerate(cuts):
                    reg[rows, cols] = 3 * a + b                                ## region id in the shifted frame
            reg = self.windows(reg[None, :, :, None]).squeeze(-1)              ## (#win, M*M)
            mask = (reg[:, :, None] != reg[:, None, :]).float() * -1e9
            bias = (bias[None] + mask[:, None]).repeat(B, 1, 1, 1)             ## (B*#win, heads, M*M, M*M)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        o = self.out(o.transpose(1, 2).flatten(2))                             ## (B*#win, M*M, D)
        o = o.view(B, h // M, w // M, M, M, D).transpose(2, 3).reshape(B, h, w, D)   ## undo windows
        return o.roll((s, s), dims=(1, 2)) if s else o                         ## undo the shift

## Example: 14x14 token grid, M=7 -> 4 windows per image. W-MSA then SW-MSA (LN, MLP, residual omitted)
x = torch.randn(2, 14, 14, 32)
w_msa, sw_msa = SwinAttention(32, 4, M=7), SwinAttention(32, 4, M=7, shift=True)
print(sw_msa(w_msa(x)).shape)                                                  ## torch.Size([2, 14, 14, 32])
```
````

```{dropdown} Table: Swin variants
| | $D$ | Blocks per stage | #Params | ImageNet-1k top-1 |
|:--|:--|:--|:--|:--|
| Swin-T | 96 | 2, 2, 6, 2 | 29M | 81.3 |
| Swin-S | 96 | 2, 2, 18, 2 | 50M | 83.0 |
| Swin-B | 128 | 2, 2, 18, 2 | 88M | 83.5 |
| Swin-L | 192 | 2, 2, 18, 2 | 197M | 87.3 |

- T/S/B: trained on ImageNet-1k, 224². L: pretrained on ImageNet-22k, evaluated at 384².
- $M=7$, 32 dims per head, MLP ratio 4 in all variants.
- Swin-T ≈ ResNet-50 compute. Swin-S ≈ ResNet-101.
- Swin-L: COCO test-dev 58.7 box AP / 51.1 mask AP. ADE20K val 53.5 mIoU.
```

```{attention} Q&A
:class: dropdown
*Why shift the windows?*
- W-MSA alone → ❌information across window borders → each window acts as an isolated image.
- Shift by $\lfloor M/2\rfloor$ → next layer's windows straddle old borders → receptive field grows across layers.
- Swin-T ablation: shift → +1.1% ImageNet top-1, +2.8 box AP / +2.2 mask AP on COCO, +2.8 mIoU on ADE20K.

*Why cyclic shift + mask, ❌padding?*
- Plain shift → more, partial windows: 2×2 → 3×3 windows = 2.25× compute.
- Cyclic shift keeps the #windows. The mask blocks attention between sub-windows that were not adjacent.

*Relative bias or absolute position embedding?*
- Relative bias → clearly better. Adding absolute embedding on top → slightly worse.

*How does Swin relate to CNNs?*
- It brings back CNN priors: locality (windows), hierarchy (merging), translation-invariant position terms (relative bias).
- Reverse direction: a ResNet modernized step by step toward ViT design (ConvNeXt) matches or beats Swin on ImageNet, COCO & ADE20K {cite:p}`liu2022convnet`.
- → Much of Swin's gain comes from design & training choices, not from attention alone.

*Cons?*
- Shift, roll & masking → more complex code; harder to use w/ fused attention kernels.
- Grid must be divisible by $M$ → padding at arbitrary input sizes.
- Global context only builds up over layers.
```

&nbsp;

## Resolution
- **What**: Make a ViT work at input sizes & aspect ratios other than the pretraining ones.

&nbsp;

(vit-2d-rope)=
### 2D RoPE
- **What**: [RoPE](../nlp/pos.md#rope) w/ half the rotary pairs rotated by the patch's row idx & half by its column idx.
- **Why**: Learned absolute position embeddings are tied to 1 grid size.
    - New resolution → interpolate the table ([Patch Embedding](image.md#patch-embedding)) → approximate, usually needs fine-tuning.
    - RoPE → attention logit depends on the 2D offset only → any grid, ❌table.
- **How**:
    1. Patch at grid cell $(u,v)$. Split each head's $d_k$ dims into 2 halves.
    2. First half: 1D RoPE w/ position $u$. Second half: 1D RoPE w/ position $v$.
    3. → $\tilde{\mathbf{q}}^\top\tilde{\mathbf{k}}$ depends on content & $(\Delta u,\Delta v)$ only.
    - **Axial** (above): every pair sees only 1 axis. Used by DINOv3.
    - **RoPE-Mixed**: learnable per-pair frequencies on both axes → also diagonal directions {cite:p}`heo2024rotary`.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $(u,v)$: Grid row & column of a patch.
    - $d_k$: Dim per head.
    - $\mathbf{q}_{[a:b]}$: Dims $a$ to $b-1$ of $\mathbf{q}$.

Forward (axial):

$$
\tilde{\mathbf{q}}_{(u,v)}=\left[\,\text{RoPE}\left(\mathbf{q}_{[0:d_k/2]},\,u\right);\ \text{RoPE}\left(\mathbf{q}_{[d_k/2:d_k]},\,v\right)\right]
$$

- Same for $\mathbf{k}$. Each half = 1D RoPE → its dot product depends only on that axis's offset ([derivation](../nlp/pos.md#rope)).
- Sum of the 2 halves:

    $$
    \tilde{\mathbf{q}}_{(u,v)}^\top\tilde{\mathbf{k}}_{(u',v')}=g\left(\mathbf{q},\mathbf{k},\,u'-u,\,v'-v\right)
    $$

- RoPE-Mixed: rotate pair $n$ by angle $u\,\theta^u_n+v\,\theta^v_n$, w/ $\theta^u_n,\theta^v_n$ learnable → still relative.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

## Reuses RoPE from nlp/pos.md
class AxialRoPE2D(nn.Module):
    def __init__(self, d_k, base=10000.0):
        super().__init__()
        self.rope = RoPE(d_k // 2, base)                            ## 1D RoPE on each half, same frequencies

    def forward(self, x, u, v):                                     ## x: (..., N, d_k); u, v: (N,) row & column idx
        h = x.shape[-1] // 2
        return torch.cat([self.rope(x[..., :h], u), self.rope(x[..., h:], v)], dim=-1)

## Example: logit depends only on the 2D offset (1, 2)
rope2d = AxialRoPE2D(8)
q, k = torch.randn(1, 8), torch.randn(1, 8)
t = lambda a: torch.tensor([a])
s1 = (rope2d(q, t(0), t(0)) * rope2d(k, t(1), t(2))).sum()
s2 = (rope2d(q, t(2), t(1)) * rope2d(k, t(3), t(3))).sum()
print(torch.allclose(s1, s2, atol=1e-5))                            ## True
```
````

```{attention} Q&A
:class: dropdown
*Why not 1D RoPE on the raster idx $u\cdot(W/P)+v$?*
- Vertical neighbor offset = $W/P$ → depends on grid width → changes w/ resolution & aspect ratio.
- 2 axes → offsets are the same at any grid size.

*Why does it transfer across resolutions better than interpolated absolute embeddings?*
- Interpolation changes the position vector of every token. RoPE keeps local offsets exact.
- Offsets larger than the training grid are still unseen → same issue as LLM [context extension](../nlp/pos.md#context-extension).

*Axial vs Mixed?*
- Axial: each pair sees 1 axis → ❌diagonal frequencies.
- Mixed: learnable frequencies combine both axes → handles diagonals, at a small param cost.
```

&nbsp;

### NaViT
- **Name**: Native Resolution ViT {cite:p}`dehghani2023patch`
- **What**: ViT trained on images at their native resolution & aspect ratio, packed into fixed-length sequences.
- **Why**: Resize to a fixed square = CNN-era habit.
    - → distorts aspect ratio, & wastes or loses pixels.
    - A ViT is a sequence model → a variable #tokens per image is natural.
- **How**:
    1. Patchify each image at its own size → variable #tokens.
    2. **Pack** several images into 1 sequence, as in LLM [document packing](../nlp/pre/obj.md#document-packing).
    3. **Masked self-attention**: tokens attend only within their own image ([document mask](../nlp/attn.md#document)). **Masked pooling** → 1 vector per image.
    4. **Factorized position embeddings**: separate row & column embeddings, summed. Absolute (patch idx) or fractional (idx ÷ grid size) coordinates.
    5. **Continuous token dropping**: random drop rate per image → faster training. **Resolution sampling**: random resolution per image.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\phi_u,\phi_v$: Row & column position embeddings.
- Misc:
    - $(u,v)$: Grid row & column of a patch.
    - $\mathcal{S}_i$: Token positions of image $i$ in the packed sequence.

Forward:

$$
\mathbf{z}_0[t]=\mathbf{x}_p^tE+\phi_u(u_t)+\phi_v(v_t)
$$

$$
\text{Mask}_{st}=\begin{cases}0 & s,t\in\mathcal{S}_i\text{ for some }i\\ -\infty & \text{otherwise}\end{cases}
$$

$$
\mathbf{h}_i=\frac{1}{|\mathcal{S}_i|}\sum_{t\in\mathcal{S}_i}\mathbf{z}_L[t]
$$

- $\mathbf{x}_p^t E$: Patch projection of token $t$, as in [Patch Embedding](image.md#patch-embedding).
- $\mathbf{h}_i$: Pooled vector of image $i$. Mean pooling shown. Other masked poolings work the same way.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class NaViT(nn.Module):
    def __init__(self, P=16, C=3, D=768, max_side=64, heads=12, L=12):
        super().__init__()
        self.P = P
        self.proj = nn.Linear(P * P * C, D)
        self.pos_u = nn.Embedding(max_side, D)                      ## factorized: 1 row table + 1 column table
        self.pos_v = nn.Embedding(max_side, D)
        layer = nn.TransformerEncoderLayer(D, heads, 4 * D, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, L)

    def tokens(self, X):                                            ## X: (C, H, W); H, W divisible by P, any ratio
        C, H, W = X.shape
        P, gh, gw = self.P, H // self.P, W // self.P
        patches = X.reshape(C, gh, P, gw, P).permute(1, 3, 0, 2, 4).reshape(gh * gw, C * P * P)
        u = torch.arange(gh).repeat_interleave(gw)                  ## row idx of each patch
        v = torch.arange(gw).repeat(gh)                             ## column idx
        return self.proj(patches) + self.pos_u(u) + self.pos_v(v)

    def forward(self, images):                                      ## list of (C, H_i, W_i)
        toks = [self.tokens(X) for X in images]
        ids = torch.cat([torch.full((len(t),), i) for i, t in enumerate(toks)])   ## image id per token
        z = torch.cat(toks)[None]                                   ## (1, sum N_i, D): 1 packed sequence
        z = self.encoder(z, mask=ids[:, None] != ids[None, :])[0]   ## True = blocked: other image
        onehot = (ids[None, :] == torch.arange(len(images))[:, None]).float()      ## (#img, sum N_i)
        return onehot @ z / onehot.sum(1, keepdim=True)             ## masked mean pooling -> (#img, D)

## Example: a 32x48 & a 16x16 image, P=8 -> 24 + 4 = 28 tokens in 1 sequence
model = NaViT(P=8, D=32, heads=4, L=2)
print(model([torch.randn(3, 32, 48), torch.randn(3, 16, 16)]).shape)   ## torch.Size([2, 32])
```
````

```{attention} Q&A
:class: dropdown
*Why does packing help?*
- Fixed sequence length → batches stay dense w/ real tokens → ❌padding waste.
- Same compute → better models: the best ViT is matched by a NaViT w/ 4× less compute.

*Why factorized position embeddings?*
- A joint 2D table must see every $(u,v)$ pair in training.
- Separate row & column tables → any aspect ratio, & easier extrapolation to unseen sizes.

*Inference benefit?*
- Choose the resolution at test time → a direct cost-vs-accuracy knob, w/o retraining.

*How does it relate to MAE masking?*
- Token dropping = random token removal, like [MAE](#mae). Here only for speed, ❌reconstruction target.
```

&nbsp;

## Self-Supervised Pretraining
- **What**: Pretrain a ViT w/o labels. The image itself supplies the target.

&nbsp;

### BEiT
- **Name**: Bidirectional Encoder representation from Image Transformers {cite:p}`bao2022beit`
- **What**: [MLM](../nlp/pre/obj.md#mlm) for images: predict the discrete visual token of each masked patch.
- **Why**: Port BERT pretraining to ViT.
    - Discrete targets → CE over a vocabulary, as in BERT.
    - Paper's argument: pixel regression spends capacity on short-range, high-frequency detail.
- **How**:
    1. Tokenizer: pretrained dVAE (from DALL·E) → each patch → 1 of 8192 visual tokens. 224² → 14×14 tokens.
    2. **Blockwise masking**: mask rectangular blocks (≥16 patches each) until ~40% of patches are masked.
    3. Replace masked patch embeddings w/ 1 learnable [MASK] embedding → ViT.
    4. Softmax head at masked positions → CE vs the visual token.
    5. Fine-tune w/ a task head.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $t_i\in\{1,\dots,8192\}$: Visual token of patch $i$.
    - $\mathcal{M}$: Masked positions.
    - $\mathbf{X}^\mathcal{M}$: Image w/ the patches in $\mathcal{M}$ replaced by [MASK].
- Misc:
    - $\mathcal{D}$: Training images.

Objective:

$$
\max\ \sum_{\mathbf{X}\in\mathcal{D}}\mathbb{E}_{\mathcal{M}}\left[\sum_{i\in\mathcal{M}}\log p_\text{MIM}\left(t_i\mid\mathbf{X}^\mathcal{M}\right)\right]
$$
```

```{attention} Q&A
:class: dropdown
*Results?*
- BEiT-B: 83.2% ImageNet-1k vs 81.8% for DeiT-B from scratch, same setup.
- BEiT-L: 86.3% w/ ImageNet-1k only > supervised ViT-L on ImageNet-22k (85.2%).

*Which parts matter?*
- Ablation (300 epochs): visual tokens → raw-pixel regression: ImageNet 82.9 → 81.0, ADE20K 44.7 → 41.4.
- ❌Blockwise masking: ImageNet ≈ same, ADE20K 44.7 → 42.9.

*Cons?*
- Extra tokenizer stage: dVAE trained separately, on extra data.
- [MAE](#mae) later matched it w/ normalized pixels & no tokenizer.
```

&nbsp;

### MAE
- **Name**: Masked Autoencoder {cite:p}`he2022masked`
- **What**: Mask most patches → ViT encoder on the visible patches only → light decoder reconstructs the masked pixels.
- **Why**: Images are spatially redundant.
    - Low mask ratio → solvable by interpolation from neighbors → weak signal.
    - Mask 75% → model must understand objects & scenes.
    - Encoder skips masked tokens → 4× fewer tokens → ≥3× faster pretraining → scales to ViT-H.
- **How**:
    1. Patchify. Random shuffle per image → keep 25% (uniform random, ❌blocks).
    2. Encoder (ViT) on visible tokens only. ❌Mask tokens in the encoder.
    3. Decoder input: encoded visible tokens + 1 shared learnable mask token at every masked position + position embeddings for all.
    4. Decoder: 8 blocks, width 512 → pixel values per patch.
    5. MSE on masked patches only. Target = per-patch normalized pixels.
    6. Discard the decoder. Fine-tune or probe the encoder.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{x}_p^i\in\mathbb{R}^{P^2C}$: Pixels of patch $i$.
    - $\hat{\mathbf{x}}_p^i$: Decoder output for patch $i$.
- Hyperparams:
    - Mask ratio: 0.75.
- Misc:
    - $\mathcal{M}$: Masked patches, $|\mathcal{M}|=0.75N$.
    - $\mu_i$: Mean of patch $i$'s pixels.
    - $\sigma_i$: Std of patch $i$'s pixels.

Objective:

$$
\mathcal{L}=\frac{1}{|\mathcal{M}|\,P^2C}\sum_{i\in\mathcal{M}}\left\|\hat{\mathbf{x}}_p^i-\frac{\mathbf{x}_p^i-\mu_i}{\sigma_i}\right\|_2^2
$$
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

## Reuses PatchEmbed from cv/image.md
class MAE(nn.Module):
    def __init__(self, img_size=224, P=16, C=3, D=768, L=12, heads=12,
                 D_dec=512, L_dec=8, heads_dec=16, ratio=0.75):
        super().__init__()
        self.pe, self.ratio = PatchEmbed(img_size, P, C, D), ratio
        N = (img_size // P) ** 2
        layer = lambda d, h: nn.TransformerEncoderLayer(d, h, 4 * d, activation="gelu",
                                                        batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer(D, heads), L)
        self.to_dec = nn.Linear(D, D_dec)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, D_dec))    ## 1 shared learned vector
        self.dec_pos = nn.Parameter(torch.randn(1, N, D_dec) * 0.02)
        self.decoder = nn.TransformerEncoder(layer(D_dec, heads_dec), L_dec)   ## narrow & shallow
        self.to_pix = nn.Linear(D_dec, P * P * C)

    def forward(self, X):
        target = self.pe.patchify(X)                                ## (B, N, P*P*C)
        z = self.pe.proj(target) + self.pe.pos[:, 1:]               ## patch tokens + positions, no [CLS]
        B, N, D = z.shape
        vis = torch.rand(B, N).argsort(dim=1)[:, : int(N * (1 - self.ratio))]   ## random 25% kept per image
        h = self.encoder(z.gather(1, vis[..., None].expand(-1, -1, D)))         ## encoder sees ONLY visible tokens
        h = self.to_dec(h)
        full = self.mask_token.expand(B, N, -1).clone()
        full.scatter_(1, vis[..., None].expand(-1, -1, h.shape[-1]), h)         ## visible back in place, mask token elsewhere
        pred = self.to_pix(self.decoder(full + self.dec_pos))      ## (B, N, P*P*C)
        mu, var = target.mean(-1, keepdim=True), target.var(-1, keepdim=True)
        target = (target - mu) / (var + 1e-6).sqrt()                ## per-patch normalized pixels
        masked = torch.ones(B, N, dtype=torch.bool)
        masked[torch.arange(B)[:, None], vis] = False
        return ((pred - target) ** 2).mean(-1)[masked].mean()       ## MSE on masked patches only

## Example: 32x32 images, P=8 -> 16 patches, 4 visible
mae = MAE(img_size=32, P=8, D=64, L=2, heads=4, D_dec=32, L_dec=1, heads_dec=4)
print(mae(torch.randn(2, 3, 32, 32)).item() > 0)                    ## True
```
````

```{attention} Q&A
:class: dropdown
*Why 75% for images vs 15% in BERT?*
- A word carries dense meaning. A patch is mostly predictable from its neighbors.
- → High ratio needed for a nontrivial task. 75% works well for both fine-tuning & linear probing.

*Why keep mask tokens out of the encoder?*
- ViT-L ablation: encoder w/ mask tokens → 84.2% fine-tune, 59.6% linear, 3.3× FLOPs. W/o → 84.9%, 73.5%, 1×.
- Mask tokens never enter the encoder → ❌train/fine-tune mismatch: fine-tuning has no masks either.

*Random vs block masking?*
- MAE: random 75% > block masking (block 50%: 83.9, block 75%: 82.8 vs random 75%: 84.9).
- Opposite of [BEiT](#beit), where blockwise helps. Different targets & mask ratios → different best masks.

*Pixels vs tokens?*
- Normalized pixels (85.4) ≈ dVAE tokens (85.3) in fine-tuning. Tokens → lower linear probe.
- Normalization removes each patch's mean & scale → loss weights local structure, not absolute intensity.
- Loss on all pixels, ❌masked only → ~0.5% worse.

*Linear probe vs fine-tune?*
- MAE features are less linearly separable than contrastive features (e.g., MoCo v3).
- Tune ≥1 block → MAE better. → Linear probing & fine-tuning accuracy are largely uncorrelated.

*Key numbers?*
- ViT-H: 87.8% ImageNet-1k, best among methods using only ImageNet-1k data.
- Decoder: ~9% of ViT-L's FLOPs per token.
```

&nbsp;

### DINO
- **Name**: self-DIstillation with NO labels {cite:p}`caron2021emerging`
- **What**: Student matches an EMA teacher's softmax output, across different crops of the same image.
- **Why**: Learn features w/o labels, w/o negatives & w/o reconstruction.
    - Crops of 1 image should map to the same distribution over learned "pseudo-classes" → invariance.
- **How**:
    1. **Multi-crop**: 2 global crops (224², >50% of the image) + several local crops (96², <50%).
    2. Student sees all crops. Teacher sees global crops only → local-to-global correspondence.
    3. Both: ViT + projection head → $K_o$ logits.
    4. Teacher output: **center** (subtract a running mean $\mathbf{c}$) → **sharpen** (low temperature $\tau_t$).
    5. Loss: CE (teacher probs, student probs) over all (teacher crop, different student crop) pairs.
    6. Grads → student only. Teacher ← EMA of the student.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{X}_1^g,\mathbf{X}_2^g$: Global crops.
    - $\mathcal{X}$: All crops of 1 image (global + local).
- Params:
    - $\theta_s$: Student params.
    - $\theta_t$: Teacher params.
- Hyperparams:
    - $K_o$: Output dim, 65,536 (paper's $K$; renamed ← $K$ = #classes here).
    - $\tau_s$: Student temperature, 0.1.
    - $\tau_t$: Teacher temperature, linear warm-up 0.04 → 0.07 over the first 30 epochs.
    - $\lambda$: EMA rate, cosine schedule 0.996 → 1.
    - $m$: Center momentum, 0.9.
- Misc:
    - $g_\theta$: ViT + projection head → logits in $\mathbb{R}^{K_o}$.
    - $\mathbf{c}\in\mathbb{R}^{K_o}$: Center.

Objective:

$$
P_s(\mathbf{X})=\text{softmax}\left(\frac{g_{\theta_s}(\mathbf{X})}{\tau_s}\right),\qquad P_t(\mathbf{X})=\text{softmax}\left(\frac{g_{\theta_t}(\mathbf{X})-\mathbf{c}}{\tau_t}\right)
$$

$$
\min_{\theta_s}\sum_{\mathbf{X}\in\{\mathbf{X}_1^g,\mathbf{X}_2^g\}}\ \sum_{\substack{\mathbf{X}'\in\mathcal{X}\\ \mathbf{X}'\neq\mathbf{X}}}H\big(P_t(\mathbf{X}),P_s(\mathbf{X}')\big),\qquad H(\mathbf{a},\mathbf{b})=-\mathbf{a}^\top\log\mathbf{b}
$$

Updates (no grads):

$$
\theta_t\leftarrow\lambda\theta_t+(1-\lambda)\theta_s,\qquad \mathbf{c}\leftarrow m\,\mathbf{c}+(1-m)\frac{1}{B}\sum_{i=1}^{B}g_{\theta_t}(\mathbf{X}_i)
$$
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DINOLoss(nn.Module):
    def __init__(self, K_o, tau_s=0.1, tau_t=0.04, m=0.9):
        super().__init__()
        self.tau_s, self.tau_t, self.m = tau_s, tau_t, m
        self.register_buffer("c", torch.zeros(K_o))

    def forward(self, s_out, t_out):
        ## s_out: list of (B, K_o) student logits for ALL crops; first 2 = global crops
        ## t_out: list of 2 (B, K_o) teacher logits for the global crops, computed under no_grad
        t_prob = [F.softmax((t - self.c) / self.tau_t, dim=-1) for t in t_out]   ## center, then sharpen
        loss, n = 0.0, 0
        for i, p_t in enumerate(t_prob):
            for j, s in enumerate(s_out):
                if j == i:                                          ## skip: same crop on both sides
                    continue
                loss = loss + (-p_t * F.log_softmax(s / self.tau_s, dim=-1)).sum(-1).mean()
                n += 1
        with torch.no_grad():                                       ## center = EMA of the teacher batch mean
            self.c.mul_(self.m).add_((1 - self.m) * torch.cat(t_out).mean(0))
        return loss / n

    @staticmethod
    @torch.no_grad()
    def update_teacher(teacher, student, lam=0.996):
        for pt, ps in zip(teacher.parameters(), student.parameters()):
            pt.mul_(lam).add_((1 - lam) * ps)                       ## teacher = EMA of student, never backprop

## Example: stand-in backbone + head (any input size); 2 global + 4 local crops, K_o = 16
net = lambda: nn.Sequential(nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(3 * 16, 16))
student, teacher = net(), net()
teacher.load_state_dict(student.state_dict())
crops = [torch.randn(2, 3, 32, 32) for _ in range(2)] + [torch.randn(2, 3, 12, 12) for _ in range(4)]
crit = DINOLoss(K_o=16)
with torch.no_grad():
    t_out = [teacher(x) for x in crops[:2]]                         ## teacher: global crops only
loss = crit([student(x) for x in crops], t_out)
loss.backward()
DINOLoss.update_teacher(teacher, student)
print(loss.item() > 0)                                              ## True
```
````

```{attention} Q&A
:class: dropdown
*How does DINO avoid collapse w/o negatives?*
- 2 collapse modes: 1 dim dominates every output, or every output is uniform.
- Centering stops 1 dim from dominating, but pushes toward uniform. Sharpening does the opposite. Both together → balance.
- Fixed $\tau_t>0.06$ → collapse: loss stays at $\ln K_o$.
- Warm-up 0.04 → 0.07 does not collapse ← training starts sharp, then relaxes.

*Why an EMA teacher?*
- Teacher ≈ ensemble of past students (Polyak-Ruppert averaging) → better than the student during all of training.
- → Teacher gives higher-quality targets than the student itself.

*Why is it called distillation?*
- Same CE as [token-level KD](../nlp/post/kd.md#token-level-kd). But teacher = EMA of the student, ❌pretrained teacher, ❌labels.

*How does it differ from contrastive learning?*
- [Contrastive views](image.md#contrastive-views) + [InfoNCE](../ml/obj.md#infonce) need negatives → large batches or queues.
- DINO: ❌negatives. Centering uses only first-order batch statistics → works across batch sizes.

*What emerged?*
- Last-layer [CLS] attention maps segment objects w/o supervision. Much less clear in supervised ViTs & CNNs.
- Features are strong k-NN classifiers: 78.3% ImageNet top-1 w/ ViT-S/8.
- Linear probe: 80.1% w/ ViT-B/8.

*Why small patches & multi-crop?*
- $P$ 16 → 8: accuracy ⬆️, at much higher cost (4× tokens).
- Multi-crop (ViT-S/16, 100 epochs): 2×224² → 67.8%; + 10×96² → 74.6%. Local crops are cheap.
```

&nbsp;

#### DINOv2
- **What**: DINO + iBOT patch loss, scaled to a 1.1B ViT-g on 142M curated images, then distilled into smaller ViTs. {cite:p}`oquab2024dinov2`
- **Why**: Goal = all-purpose frozen features: work for image- & pixel-level tasks w/o fine-tuning.
    - SSL on uncurated web data → lower feature quality. Curated datasets alone → too small.
- **How**:
    1. Data: LVD-142M ← retrieve uncurated web images close to curated datasets, & deduplicate.
    2. Loss = DINO image-level loss ([CLS]) + iBOT patch-level loss + KoLeo regularizer.
        - **iBOT**: mask some student patches → student predicts the teacher's tokens at those positions.
    3. Teacher centering: Sinkhorn-Knopp (3 iterations) replaces the running mean.
    4. Short final phase at 518² resolution.
    5. Distill ViT-g → ViT-S/B/L.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\mathbf{x}_1,\dots,\mathbf{x}_B$: $\ell_2$-normalized features in a batch.
    - $p_{t}$, $p_{s}$: Teacher & student probabilities, as in [DINO](#dino).

Objective terms:

$$
\mathcal{L}_\text{DINO}=-\sum p_t\log p_s,\qquad \mathcal{L}_\text{KoLeo}=-\frac{1}{B}\sum_{i=1}^{B}\log d_i,\qquad d_i=\min_{j\neq i}\|\mathbf{x}_i-\mathbf{x}_j\|
$$

- iBOT term: same CE form, at masked patch positions, w/ a separate head.
- KoLeo ⬇️ ← nearest neighbors far apart → features spread uniformly over the batch.
- Paper's $n$ & $d_{n,i}$ → $B$ & $d_i$ here ← $n$ = #features globally.
```

```{attention} Q&A
:class: dropdown
*What does each loss term add?*
- KoLeo → nearest-neighbor tasks ⬆️ (retrieval).
- iBOT patch loss → patch-level tasks ⬆️ (segmentation).

*Why distill small models, ❌train them directly?*
- ViT-L/14 distilled from ViT-g/14 > ViT-L/14 trained from scratch on all 12 benchmarks.

*Result?*
- Beats OpenCLIP on most image- & pixel-level benchmarks w/ frozen features.

*Known flaw?*
- High-norm artifact tokens → fixed by [Registers](#registers) ("DINOv2 + reg" checkpoints).
```

&nbsp;

#### DINOv3
- **What**: DINOv2 recipe scaled to a 7B ViT on LVD-1689M, + Gram anchoring to keep patch features sharp. {cite:p}`simeoni2026dinov3`
- **Why**: Long training of large SSL models → dense (patch) features degrade, even while global metrics keep rising.
    - Symptom: cosine similarity between [CLS] & patch outputs ⬆️ during training → patch features lose locality.
    - Different from the high-norm artifacts that [Registers](#registers) fix.
- **How**:
    1. Data: LVD-1689M, curated from a pool of public Instagram images.
    2. Model: custom 7B ViT, axial [2D RoPE](#vit-2d-rope), register tokens. Constant hyperparam schedules, 1M iterations.
    3. **Gram anchoring** (refinement phase after the 1M iterations): match the student's patch-similarity matrix to that of the **Gram teacher**. Global crops only.
        - Gram teacher: starts as an early-iteration teacher. Every 10k iterations → reset to the current EMA teacher.
    4. Post-hoc: high-resolution adaptation, distillation into ViT-S/B/L & ConvNeXt, text alignment.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $X_S\in\mathbb{R}^{N\times D}$: $\ell_2$-normalized patch features of the student.
    - $X_G\in\mathbb{R}^{N\times D}$: Same, from the Gram teacher.

Objective:

$$
\mathcal{L}_\text{Gram}=\left\|X_SX_S^\top-X_GX_G^\top\right\|_F^2
$$

- Constrains only pairwise patch similarities → features can still change if the similarity structure is kept.
- Paper notation: $P$ patches, dim $d$ → here $N$, $D$.
```

```{attention} Q&A
:class: dropdown
*Why anchor the Gram matrix, ❌the features?*
- Early-iteration features have good locality but weaker semantics.
- Matching features → copies the weak semantics. Matching similarities → keeps only the local structure.

*Why RoPE?*
- Relative positions → runs at varying resolutions w/o adaptation.

*Result?*
- Strongest gains on dense tasks (segmentation, depth) w/ frozen features. Distilled ViT-L is close to the 7B teacher on many tasks.
```

&nbsp;

## Vision-Language
- **What**: Supervise the image encoder w/ paired text, ❌fixed label set.

&nbsp;

### CLIP
- **Name**: Contrastive Language-Image Pre-training {cite:p}`radford2021learning`
- **What**: Image & text encoders trained so that matching image-caption pairs have the highest cosine similarity in the batch.
- **Why**: Fixed label sets limit vision models: new concept → new labeled data.
    - Web captions → free, open-vocabulary supervision.
    - Text encoder at test time → classify w/ any label set (zero-shot).
- **How**:
    1. Data: 400M (image, text) pairs from the internet.
    2. Image encoder (ResNet or ViT) & text encoder (transformer) → linear projection each → $\ell_2$-normalize.
    3. $B\times B$ cosine-similarity matrix × learnable logit scale $t$.
    4. Symmetric [InfoNCE](../ml/obj.md#infonce): CE over rows (image → text) + CE over columns (text → image).
    5. **Zero-shot**: embed the prompt "A photo of a {label}." for every class → predict the most similar.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{e}^I_i$: Normalized image embedding of pair $i$.
    - $\mathbf{e}^T_j$: Normalized text embedding of pair $j$.
- Params:
    - $t=1/\tau$: Learnable logit scale. Init $1/0.07$, clipped at 100.

Objective:

$$
S_{ij}=t\,\mathbf{e}^{I\top}_i\mathbf{e}^T_j
$$

$$
\mathcal{L}=\frac{1}{2B}\sum_{i=1}^{B}\left[-\log\frac{e^{S_{ii}}}{\sum_{j=1}^{B}e^{S_{ij}}}-\log\frac{e^{S_{ii}}}{\sum_{j=1}^{B}e^{S_{ji}}}\right]
$$

Inference (zero-shot):

$$
\hat{k}=\arg\max_{k}\ \mathbf{e}^{I\top}\mathbf{e}^T\big(\text{prompt}_k\big)
$$
```

````{important} Code
:class: dropdown
```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_t = nn.Parameter(torch.tensor(math.log(1 / 0.07)))   ## learnable logit scale t = 1/tau

    def forward(self, img_emb, txt_emb):                               ## (B, D_e) each; row i = pair i
        e_i, e_t = F.normalize(img_emb, dim=-1), F.normalize(txt_emb, dim=-1)
        t = self.log_t.exp().clamp(max=100)                            ## logits scaled by at most 100
        S = t * e_i @ e_t.T                                            ## (B, B): image i vs text j
        labels = torch.arange(len(S), device=S.device)                 ## positives on the diagonal
        ## InfoNCE (ml/obj.md) in both directions: image->text over rows, text->image over columns
        return (F.cross_entropy(S, labels) + F.cross_entropy(S.T, labels)) / 2

## Example: 4 pairs, 8-dim joint space
img, txt = torch.randn(4, 8), torch.randn(4, 8)
print(CLIPLoss()(img, txt).item() > 0)                                 ## True
## zero-shot: 3 class-prompt embeddings -> nearest per image
prompts = F.normalize(torch.randn(3, 8), dim=-1)
print((F.normalize(img, dim=-1) @ prompts.T).argmax(-1))               ## 4 predicted class ids
```
````

```{attention} Q&A
:class: dropdown
*Why contrastive, ❌caption prediction?*
- Transformer LM captioning learns zero-shot ImageNet 3× slower than predicting a bag of words.
- Contrastive objective → another 4× more efficient than bag of words.
- Exact words are hard to predict. "Which caption matches?" is an easier, sufficient task.

*Why a learnable, clipped temperature?*
- $\tau$ is very sensitive ([InfoNCE](../ml/obj.md#infonce) Q&A) → learn it.
- Clip at 100 → prevents training instability.

*Why a huge batch (32,768)?*
- $B-1$ negatives per pair. The InfoNCE bound is capped at $\log B$.
- Cost: softmax needs all $B\times B$ similarities → embeddings all-gathered across devices.

*Prompt engineering?*
- "A photo of a {label}." → +1.3% ImageNet vs the bare label. Ensemble of 80 prompts → +3.5% more. Together ≈ +5%.

*Key numbers?*
- Best zero-shot ImageNet: 76.2% → matches the original supervised ResNet-50, w/o its 1.28M labeled images.
- Text encoder: 63M params, 12 layers, width 512, 8 heads, BPE vocab 49,152, max 76 tokens.

*Cons?*
- Weak zero-shot on specialized or abstract tasks: satellite images, tumor detection, counting objects, distance estimation.
- In-batch false negatives (near-duplicate captions) → pushed apart.

*Where is it used now?*
- CLIP- & SigLIP-trained ViTs are common vision encoders in VLMs (e.g., LLaVA uses CLIP ViT-L/14). Text encoders condition [diffusion](../dl/gen/diffusion.md#cross-attention-conditioning).
```

&nbsp;

#### SigLIP
- **Name**: Sigmoid Loss for Language-Image Pre-training {cite:p}`zhai2023sigmoid`
- **What**: CLIP w/ the softmax contrastive loss replaced by 1 independent binary sigmoid loss per image-text pair.
- **Why**: Softmax normalizes each row & column over the whole batch → needs a global view of all $B\times B$ similarities.
    - → all-gather, memory ⬆️, tied to batch size.
    - Sigmoid: each pair alone → ❌batch-wide normalizer.
- **How**:
    1. Same 2 encoders. Normalize. $B\times B$ similarities.
    2. Logit $=t\cdot\text{sim}+b$. Learnable $t$ (init 10) & bias $b$ (init −10).
    3. Label +1 for the $B$ matched pairs, −1 for the $B^2-B$ others → binary log-sigmoid loss. Sum, ÷$B$.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $t=e^{t'}$: Logit scale. $t'$ learnable, init $\log10$.
    - $b$: Logit bias, init $-10$.
- Misc:
    - $\sigma$: Sigmoid.
    - $\mathbf{e}^I_i,\mathbf{e}^T_j$: As in [CLIP](#clip).

Objective:

$$
\mathcal{L}=-\frac{1}{B}\sum_{i=1}^{B}\sum_{j=1}^{B}\log\sigma\Big(z_{ij}\left(t\,\mathbf{e}^{I\top}_i\mathbf{e}^T_j+b\right)\Big),\qquad z_{ij}=\begin{cases}1 & i=j\\-1 & i\neq j\end{cases}
$$
```

````{important} Code
:class: dropdown
```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SigLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.t_prime = nn.Parameter(torch.tensor(math.log(10.0)))      ## t = exp(t'), init t = 10
        self.b = nn.Parameter(torch.tensor(-10.0))                      ## init: every pair looks like a non-match

    def forward(self, img_emb, txt_emb):
        e_i, e_t = F.normalize(img_emb, dim=-1), F.normalize(txt_emb, dim=-1)
        logits = self.t_prime.exp() * e_i @ e_t.T + self.b              ## (B, B): each pair scored alone
        z = 2 * torch.eye(len(logits), device=logits.device) - 1        ## +1 on the diagonal, -1 elsewhere
        return -F.logsigmoid(z * logits).sum() / len(logits)            ## no softmax -> no batch-wide normalizer

## Example: 4 pairs, 8-dim joint space
print(SigLIPLoss()(torch.randn(4, 8), torch.randn(4, 8)).item() > 0)   ## True
```
````

```{attention} Q&A
:class: dropdown
*Why init $b=-10$?*
- $B$ positives vs $B^2-B$ negatives → most pairs are non-matches.
- Start near that prior → ❌large early gradients spent only on pushing negatives down.

*Why does it scale?*
- Loss = sum over independent pairs → compute it chunk by chunk across devices, ❌full $B\times B$ matrix in 1 place.
- Better than softmax at small batches. Also scales to larger ones.

*How large a batch?*
- Tested up to 1M. Gains saturate quickly. 32k is sufficient.

*Key result?*
- SigLiT (frozen pretrained image encoder, train the text side): 84.5% zero-shot ImageNet in 2 days on 4 TPUv4 chips.
```

&nbsp;

## Dense Prediction
- **What**: Predict per object or per pixel, ❌per image.

&nbsp;

### DETR
- **Name**: DEtection TRansformer {cite:p}`carion2020end`
- **What**: Object detection as direct set prediction: CNN + transformer encoder-decoder → fixed set of boxes, matched 1-to-1 to ground truth.
- **Why**: Classic detectors depend on hand-designed parts: anchors, & NMS to remove duplicate boxes.
    - Set loss w/ unique matching → the model learns not to duplicate → ❌NMS, ❌anchors.
- **How**:
    1. CNN backbone (ResNet-50) → feature map → 1×1 conv → flatten → tokens + 2D positional encodings.
    2. Transformer encoder over these tokens → global context.
    3. Decoder: $N_q=100$ learned **object queries** → self-attention among queries + [cross-attention](../nlp/attn.md#cross-attention) to the encoder output. All in parallel, ❌autoregressive.
    4. FFN per query → class (incl. "no object" $\varnothing$) + normalized box.
    5. **Hungarian matching**: min-cost 1-to-1 assignment between predictions & ground truth (padded w/ $\varnothing$).
    6. Loss on the matched pairs: CE on class + L1 & GIoU on box.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $y_i=(c_i,\mathbf{b}_i)$: Ground truth $i$: class & box. Padded w/ $c_i=\varnothing$ to $N_q$ entries.
    - $\hat{p}_j(c)$: Predicted probability of class $c$ for query $j$.
    - $\hat{\mathbf{b}}_j\in[0,1]^4$: Predicted box of query $j$.
- Hyperparams:
    - $N_q$: #object queries, 100.
    - $\lambda_\text{L1}=5$, $\lambda_\text{iou}=2$: Box-loss weights.
- Misc:
    - $\pi$: Permutation of the $N_q$ queries.
    - $\mathfrak{S}_{N_q}$: All such permutations.

Process:
1. Matching:

    $$
    \hat{\pi}=\arg\min_{\pi\in\mathfrak{S}_{N_q}}\sum_{i=1}^{N_q}\mathbb{1}_{\{c_i\neq\varnothing\}}\left[-\hat{p}_{\pi(i)}(c_i)+\mathcal{L}_\text{box}\big(\mathbf{b}_i,\hat{\mathbf{b}}_{\pi(i)}\big)\right]
    $$

2. Loss:

    $$
    \mathcal{L}=\sum_{i=1}^{N_q}\left[-\log\hat{p}_{\hat{\pi}(i)}(c_i)+\mathbb{1}_{\{c_i\neq\varnothing\}}\,\mathcal{L}_\text{box}\big(\mathbf{b}_i,\hat{\mathbf{b}}_{\hat{\pi}(i)}\big)\right]
    $$

    $$
    \mathcal{L}_\text{box}(\mathbf{b},\hat{\mathbf{b}})=\lambda_\text{iou}\,\mathcal{L}_\text{GIoU}(\mathbf{b},\hat{\mathbf{b}})+\lambda_\text{L1}\|\mathbf{b}-\hat{\mathbf{b}}\|_1
    $$

- Matching uses $\hat{p}$, ❌$\log\hat{p}$ → class term on the same scale as $\mathcal{L}_\text{box}$. Better empirically.
- $c_i=\varnothing$ terms: log-prob down-weighted by 10 ← class imbalance (most queries match $\varnothing$).
- Loss terms normalized by #objects in the batch.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from torchvision.ops import generalized_box_iou

class HungarianMatcher(nn.Module):
    def __init__(self, w_cls=1.0, w_l1=5.0, w_giou=2.0):
        super().__init__()
        self.w_cls, self.w_l1, self.w_giou = w_cls, w_l1, w_giou

    @torch.no_grad()
    def forward(self, logits, boxes, gt_cls, gt_boxes):
        ## logits: (N_q, K+1) incl. "no object"; boxes: (N_q, 4) xyxy in [0, 1]
        ## gt_cls: (G,), gt_boxes: (G, 4); G <= N_q real objects, the rest of the queries -> no object
        p = logits.softmax(-1)[:, gt_cls]                           ## (N_q, G): prob. of each GT's class
        cost = (-self.w_cls * p                                     ## probability, not log-prob
                + self.w_l1 * torch.cdist(boxes, gt_boxes, p=1)
                - self.w_giou * generalized_box_iou(boxes, gt_boxes))   ## L_GIoU = 1 - GIoU; constant dropped
        q_idx, g_idx = linear_sum_assignment(cost.numpy())          ## Hungarian: min total cost, 1-to-1
        return torch.as_tensor(q_idx), torch.as_tensor(g_idx)       ## loss (Math) is then computed on these pairs

## Example: 5 queries, 3 classes + no-object, 2 GT objects
xy = torch.rand(5, 2) * 0.5
boxes = torch.cat([xy, xy + 0.3], dim=1)                            ## valid xyxy boxes
gt = torch.tensor([[0.1, 0.1, 0.4, 0.4], [0.5, 0.5, 0.9, 0.9]])
print(HungarianMatcher()(torch.randn(5, 4), boxes, torch.tensor([2, 0]), gt))   ## 2 (query, GT) pairs
```
````

```{attention} Q&A
:class: dropdown
*Why no duplicates w/o NMS?*
- 1-to-1 matching → only 1 query per object gets the box target. All others → $\varnothing$.
- Query self-attention → queries can coordinate which object each one takes.

*Why Hungarian, ❌greedy matching?*
- Optimal assignment for the whole set → loss is invariant to the order of predictions & each object gets exactly 1 query.

*Results?*
- COCO val, ResNet-50: 42.0 AP, same as a tuned Faster R-CNN-FPN w/ similar #params (41M vs 42M).
- Large objects: 61.1 vs 53.4 AP. Small objects: 20.5 vs 26.6 AP.

*Cons?*
- Slow convergence: 500 epochs to match Faster R-CNN.
- Weak on small objects ← global attention over a high-resolution feature map is too costly → only 1 low-resolution scale.
- Later fixes (e.g., Deformable DETR) use sparse attention over multi-scale features.
```

&nbsp;

### SAM
- **Name**: Segment Anything Model {cite:p}`kirillov2023segment`
- **What**: Promptable segmentation: heavy ViT image encoder + light prompt encoder & mask decoder → mask for any prompt.
- **Why**: Goal = segmentation foundation model w/ zero-shot transfer via prompts, as in LLMs.
    - No web-scale mask dataset existed → build SA-1B w/ the model in the annotation loop.
- **How**:
    1. **Image encoder**: MAE-pretrained ViT-H/16. Input 1024×1024. 14×14 windowed attention + 4 global-attention blocks → 64×64 embedding. Runs once per image.
    2. **Prompt encoder**:
        - Points, boxes → positional encodings + learned embedding per prompt type.
        - Masks → convs, added to the image embedding.
        - Free-form text → CLIP text encoder (exploratory).
    3. **Mask decoder**: 2 modified transformer decoder blocks. Self-attention over prompt tokens + cross-attention in both directions (prompt ↔ image) → upsample → mask from an output token.
    4. **Ambiguity**: 3 masks per prompt + predicted IoU per mask. Backprop only the lowest-loss mask.
    5. **Data engine**: assisted-manual → semi-automatic → fully automatic → SA-1B: 11M images, 1.1B masks.
    - Prompt encoder + mask decoder: ~50 ms in a web browser.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $m$: Ground-truth mask.
    - $\hat{m}_r$: Predicted mask $r\in\{1,2,3\}$.

Objective (per prompt):

$$
\mathcal{L}=\min_{r\in\{1,2,3\}}\left[20\,\mathcal{L}_\text{focal}(\hat{m}_r,m)+\mathcal{L}_\text{dice}(\hat{m}_r,m)\right]
$$

- [Focal loss](../ml/obj.md#focal-loss) : dice = 20 : 1.
- IoU head: MSE between predicted IoU & the mask's true IoU w/ $m$.
```

```{attention} Q&A
:class: dropdown
*Why split heavy encoder vs light decoder?*
- Image encoder cost is paid once per image. Each new prompt costs only the light decoder → interactive use.

*Why 3 masks?*
- 1 point is ambiguous: e.g., a point on a shirt → shirt or whole person.
- Nested masks are mostly ≤3 levels deep: whole, part, subpart.
- Multiple prompts → less ambiguity → a 4th output mask is returned instead.

*Why windowed attention in the encoder?*
- 1024² / 16 → 4096 tokens → global attention in every block is too costly.
- Windows (cf. [Swin](#swin)) + 4 global blocks → most layers stay cheap, global context still flows.

*Limits?*
- Output = masks, ❌class labels.
- Text prompts are exploratory only.
```

&nbsp;
