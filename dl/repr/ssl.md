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
# Self-Supervised Learning
Representation learning w/o labels: the input supplies the training signal. Joint-embedding & latent-prediction methods. Masked image modeling → [MAE](../../cv/vit.md#mae), [BEiT](../../cv/vit.md#beit). ViT self-distillation → [DINO](../../cv/vit.md#dino). Image-text → [CLIP](../../cv/vit.md#clip). Text → [MLM](../../nlp/pre/obj.md#mlm). Loss primitive → [InfoNCE](../../ml/obj.md#infonce). View construction → [Contrastive Views](../../cv/image.md#contrastive-views).

Notations:
- $\mathbf{x}$: Input sample (image)
- $\mathcal{T}$: Augmentation distribution
- $\mathbf{v},\mathbf{v}'$: 2 views of $\mathbf{x}$ ← 2 independent draws from $\mathcal{T}$
- $f$: Encoder (backbone)
- $g$: Projection head (projector)
- $q$: Predictor
- $\mathbf{h}=f(\mathbf{v})$: Representation → used downstream
- $\mathbf{z}=g(\mathbf{h})\in\mathbb{R}^d$: Embedding → fed to the loss
- $d$: Embedding dim
- $B$: Batch size (#images)
- $Z,Z'\in\mathbb{R}^{B\times d}$: Batch embeddings of the 2 views. Row $i$ = sample $i$.
- $\theta$: Online params (updated by grads)
- $\xi$: Target params (❌grads)
- $\lambda$: EMA rate: $\xi\leftarrow\lambda\xi+(1-\lambda)\theta$
- $\tau$: Temperature
- $\text{sim}(\mathbf{a},\mathbf{b})=\frac{\mathbf{a}^\top\mathbf{b}}{\|\mathbf{a}\|\|\mathbf{b}\|}$: Cosine similarity
- $\text{sg}(\cdot)$: Stop-gradient (treat as constant in backprop)

Overrides of global notation: $B$ = batch size (❌$m$). $d$ = embedding dim (❌$n$). $\lambda$ = EMA rate, as in [DINO](../../cv/vit.md#dino) (MoCo paper: $m$. BYOL paper: $\tau$).

&nbsp;

## SSL
- **Name**: Self-Supervised Learning
- **What**: Learning w/ targets built from the unlabeled input itself.
- **Why**: Labels are scarce. Unlabeled data is abundant.
    - Pretrain once w/o labels → transfer to many tasks w/ few labels.
- **How**:
    1. Pick a pretext task w/ a free target: another view, a masked part, the next token.
    2. Pretrain $f$ on it.
    3. Discard pretext-only parts ($g$, $q$, decoder).
    4. Use $\mathbf{h}$: linear probe, k-NN, or fine-tune.

```{dropdown} Table: SSL Families
| Family | Training signal | Needs hand-crafted views | Examples |
|:--|:--|:--|:--|
| Contrastive | Positive pairs together, negatives apart | ✅ (CLIP: ❌, image-caption pairs) | [SimCLR](#simclr), [MoCo](#moco), [CLIP](../../cv/vit.md#clip) |
| Non-contrastive: distillation | Online net predicts a target net's output on another view | ✅ | [BYOL](#byol), [SimSiam](#simsiam), [DINO](../../cv/vit.md#dino) |
| Non-contrastive: redundancy reduction | Views agree + embedding dims decorrelated | ✅ | [Barlow Twins](#barlow-twins), [VICReg](#vicreg) |
| Masked predictive (generative) | Reconstruct masked input (pixels / tokens) | ❌ masking only | [MAE](../../cv/vit.md#mae), [BEiT](../../cv/vit.md#beit), [MLM](../../nlp/pre/obj.md#mlm) |
| Joint-embedding predictive | Predict the embedding of a masked region | ❌ masking only | [JEPA](#jepa), [I-JEPA](#i-jepa) |

- Collapse guard of each family → [Representation Collapse](#representation-collapse).
```

```{attention} Q&A
:class: dropdown
*SSL vs unsupervised learning?*
- Both: ❌human labels.
- SSL builds an explicit target per sample → trains w/ supervised losses (CE, MSE).
- Classic unsupervised (clustering, PCA, density estimation): ❌per-sample target.

*Is next-token prediction SSL?*
- ✅ Target = next token of the same text → ❌labels. LLM pretraining = SSL at scale → [CLM](../../nlp/pre/obj.md#clm).

*How is an SSL representation evaluated?*
- [Linear probe](interp.md#linear-probe): freeze $f$ → train a linear classifier on $\mathbf{h}$. Standard ImageNet protocol.
- k-NN on frozen $\mathbf{h}$: ❌training, ❌hyperparams.
- Fine-tune: all layers, often w/ 1% or 10% of labels (semi-supervised).
- Transfer: detection, segmentation, other datasets.
- Linear probe & fine-tune rankings can disagree → [MAE](../../cv/vit.md#mae) Q&A.

*Joint-embedding vs reconstruction?*
- Reconstruction in input space → must model every detail, incl. unpredictable texture & noise → capacity spent on low-level detail.
- Joint embedding → compare in embedding space → can drop nuisance detail.
- Cost: joint embedding has a trivial solution (constant output) → needs a collapse guard. Reconstruction target is fixed → ❌collapse.

*What does invariance to $\mathcal{T}$ teach?*
- $\mathbf{h}$ keeps what survives $\mathcal{T}$ (object identity), drops what $\mathcal{T}$ changes (color, crop position).
- $\mathcal{T}$ = the prior. Wrong $\mathcal{T}$ → wrong invariance → [Contrastive Views](../../cv/image.md#contrastive-views) Q&A.
```

&nbsp;

### Representation Collapse
- **What**: Embeddings that lose input information: all inputs → 1 point (**complete**), or → a subspace (**dimensional**). {cite:p}`jing2022understanding`
- **Why**: Joint-embedding losses reward agreement between views.
    - Constant output → perfect agreement → minimal loss, zero information.
- **How**:
    - **Complete**: pull term w/o any push → all embeddings converge to 1 point.
    - **Dimensional**: embeddings vary only along $r<d$ directions. 2 causes in contrastive learning:
        1. **Strong augmentation**: augmentation variance > data variance along a direction → weights along it decay to 0.
        2. **Implicit regularization**: in ≥2-layer nets, adjacent weight matrices align → product drifts to low rank.
    - **Detect**: singular values of the embedding covariance, log scale → collapsed dims ≈ 0.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $C_Z\in\mathbb{R}^{d\times d}$: Embedding covariance over $B$ samples (a batch or a whole eval set).
    - $s_1\ge\dots\ge s_d\ge0$: Singular values of $C_Z$.

Definition:

$$
C_Z=\frac{1}{B}\sum_{i=1}^{B}(\mathbf{z}_i-\bar{\mathbf{z}})(\mathbf{z}_i-\bar{\mathbf{z}})^\top,\qquad\bar{\mathbf{z}}=\frac{1}{B}\sum_{i=1}^{B}\mathbf{z}_i
$$

- Complete collapse: $C_Z=0$ ⇔ $s_1=0$.
- Dimensional collapse: $\text{rank}(C_Z)=r<d$ ⇔ $s_{r+1}=\dots=s_d=0$.
- Centered → $\text{rank}(C_Z)\le\min(d,B-1)$ → $B\le d$ gives zero $s_j$ w/o any collapse → estimate on $B>d$ samples.
- Healthy ($B>d$): all $s_j>0$, spectrum decays slowly.
```

```{tip} Derivation
:class: dropdown
*Why does strong augmentation cause dimensional collapse?* Linear model, InfoNCE, gradient flow, as in the cited paper.
1. Encoder $\mathbf{z}=W\mathbf{v}$. Views $\mathbf{v}_i,\mathbf{v}_i'$ of sample $i$.
2. Paper's Lemma 1–2: gradient flow gives $\dot{W}=WX$ w/ $X=\hat{\Sigma}_0-\hat{\Sigma}_1$.
    - $\hat{\Sigma}_0=\sum_{i,j}\alpha_{ij}(\mathbf{v}_i-\mathbf{v}_j)(\mathbf{v}_i-\mathbf{v}_j)^\top$: softmax-weighted data covariance ← negatives push apart.
    - $\hat{\Sigma}_1=\sum_i(1-\alpha_{ii})(\mathbf{v}_i'-\mathbf{v}_i)(\mathbf{v}_i'-\mathbf{v}_i)^\top$: augmentation covariance ← positives pull together.
    - $\alpha_{ij}$: softmax weight of pair $(i,j)$ in sample $i$'s InfoNCE term. $\alpha_{ii}$: weight of the positive.
3. Treat $X$ as fixed. Solve the linear ODE: $W(t)=W(0)\,e^{Xt}=W(0)\,Ue^{\Lambda t}U^\top$ ← $X=U\Lambda U^\top$.
4. Augmentation stronger than data along eigvec $\mathbf{u}_k$ → $\Lambda_{kk}<0$ → $e^{\Lambda_{kk}t}\to0$.
5. → $W\mathbf{u}_k\to\mathbf{0}$ → $W$ loses rank → $C_Z=W\,\text{Cov}(\mathbf{v})\,W^\top$ loses rank.
- Weak augmentation → $X\succ0$ → a 1-layer linear model does not collapse. Deeper nets still collapse ← implicit regularization.
```

```{dropdown} Table: Collapse Guards
| Method | Guard against complete collapse | Guard against dimensional collapse |
|:--|:--|:--|
| [SimCLR](#simclr), [MoCo](#moco) | Negatives repel | ❌ explicit guard → still occurs |
| [BYOL](#byol), [SimSiam](#simsiam) | Asymmetry: predictor + stop-grad (+ EMA target) | ❌ explicit guard |
| [DINO](../../cv/vit.md#dino) | Centering + sharpening of teacher output | ❌ explicit guard |
| [Barlow Twins](#barlow-twins) | Cross-correlation diagonal → 1 | Off-diagonal → 0 |
| [VICReg](#vicreg) | Variance hinge per dim | Covariance off-diagonal → 0 |
| [MAE](../../cv/vit.md#mae) | Fixed pixel target | Fixed pixel target |
| [I-JEPA](#i-jepa) | EMA target + predictor + masking asymmetry | ❌ explicit guard |
```

```{attention} Q&A
:class: dropdown
*Why can't the loss curve reveal collapse?*
- For BYOL/SimSiam, collapse IS a minimizer → loss looks great.
- Monitor instead:
    - Per-dim std of $\ell_2$-normalized $\mathbf{z}$: ≈0 when collapsed, ≈$1/\sqrt{d}$ when healthy.
    - k-NN accuracy during training.
    - Singular-value spectrum / rank of $C_Z$.

*Does contrastive learning prevent all collapse?*
- ✅ Complete collapse ← negatives.
- ❌ Dimensional collapse. Pretrained SimCLR, $d=128$: many singular values of $C_Z$ ≈ 0.

*How does the projector relate to collapse?*
- Without a projector, SimCLR's representation $\mathbf{h}$ collapses dimensionally. The projector absorbs the collapse → $\mathbf{h}$ keeps more dims.
- ImageNet linear probe, ResNet-50, 100 epochs: ❌projector 51.5 → linear 61.1 → 2-layer MLP 66.5.
- **DirectCLR**: InfoNCE on the first $d_0$ dims of $\mathbf{h}$, ❌trainable projector → 62.7 > linear projector.

*Is it the same as neural collapse or mode collapse?*
- ❌ [Neural collapse](../theory.md#neural-collapse): supervised last-layer features converge to class means. Expected & label-driven.
- ❌ [Mode collapse](../gen/gan.md#mode-collapse): a generator covers few data modes. Output-distribution failure, ❌embedding failure.
```

&nbsp;

## Contrastive
- **What**: Pull positive pairs together, push negatives apart in embedding space.

&nbsp;

### SimCLR
- **Name**: Simple framework for Contrastive Learning of visual Representations {cite:p}`chen2020simple`
- **What**: Contrastive learning w/ 2 augmented views per image, an MLP projection head & in-batch negatives.
- **Why**: Earlier contrastive methods needed special architectures or memory banks.
    - SimCLR: standard ResNet + 3 ingredients → SOTA: composed augmentations, nonlinear projection head, large batch.
- **How**:
    1. Each image → 2 views ([Contrastive Views](../../cv/image.md#contrastive-views)).
    2. $\mathbf{h}=f(\mathbf{v})$ (ResNet-50) → $\mathbf{z}=g(\mathbf{h})$ (2-layer MLP, 128-d).
    3. $B$ images → $2B$ views. Each view: 1 positive, $2(B-1)$ negatives.
    4. **NT-Xent** (Normalized Temperature-scaled Cross Entropy) = [InfoNCE](../../ml/obj.md#infonce) on cosine similarity ÷ $\tau$.
    5. Batch 4096 → [LARS](../train/optim.md#lars) for stability.
    6. Discard $g$. Use $\mathbf{h}$.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $W^{(1)},W^{(2)}$: Projector weights.
- Misc:
    - $a\in\{1,\dots,2B\}$: View idx in the batch.
    - $p(a)$: Idx of the other view of the same image.
    - $\sigma$: ReLU.

Model:

$$
\mathbf{z}=g(\mathbf{h})=W^{(2)}\sigma\left(W^{(1)}\mathbf{h}\right)
$$

Objective:

$$
\ell_a=-\log\frac{\exp\left(\text{sim}(\mathbf{z}_a,\mathbf{z}_{p(a)})/\tau\right)}{\sum_{c=1}^{2B}\mathbf{1}[c\neq a]\exp\left(\text{sim}(\mathbf{z}_a,\mathbf{z}_c)/\tau\right)},\qquad\mathcal{L}_\text{NT-Xent}=\frac{1}{2B}\sum_{a=1}^{2B}\ell_a
$$

- Denominator: $2B-1$ terms = 1 positive + $2(B-1)$ negatives.
- Negatives come from both views. The [InfoNCE](../../ml/obj.md#infonce) snippet uses only the other view's $B-1$ items.
- Paper ablation ($\ell_2$-norm, batch 4096): $\tau=0.1$ best.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn.functional as F

class NTXent:
    def __init__(self, tau=0.1):
        self.tau = tau

    def __call__(self, z1, z2):                          ## (B, d) each: view 1 & view 2 of the same B images
        B = z1.size(0)
        z = F.normalize(torch.cat([z1, z2]), dim=-1)     ## (2B, d) -> dot product = cosine sim
        S = z @ z.T / self.tau                           ## (2B, 2B)
        self_mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        S = S.masked_fill(self_mask, float("-inf"))      ## 1[c != a]: a view is never its own negative
        ## positive of view a = other view of the same image: a <-> a + B
        pos = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
        return F.cross_entropy(S, pos)                   ## CE over 2B-1 candidates, mean over 2B anchors

## Example: 4 images, 8-dim embeddings
torch.manual_seed(0)
z1 = torch.randn(4, 8)
close = NTXent()(z1, z1 + 0.01 * torch.randn(4, 8))     ## views agree
far = NTXent()(z1, torch.randn(4, 8))                    ## views unrelated
print(close.item() < far.item())                         ## True
```
````

```{attention} Q&A
:class: dropdown
*Why a projection head, then throw it away?*
- Linear eval on $\mathbf{h}$: nonlinear head +3% vs linear head, >10% vs no head.
- $\mathbf{h}$ (before $g$) beats $\mathbf{z}$ (after $g$) by >10%.
- ← NT-Xent makes $\mathbf{z}$ invariant to $\mathcal{T}$ → $\mathbf{z}$ drops info (color, orientation) that downstream tasks may need. $g$ absorbs this loss. $\mathbf{h}$ keeps the info.
- Collapse angle → [Representation Collapse](#representation-collapse) Q&A.

*Why large batches?*
- Negatives = in-batch → $2(B-1)$ per anchor. Small $B$ → few negatives → weak signal.
- 100 epochs: large batches win clearly. Longer training shrinks the gap.

*Why $\ell_2$-norm & temperature?*
- Ablation: $\ell_2$ + $\tau=0.1$ → 64.4%. $\tau=1$ → 58.0%. ❌$\ell_2$ → ≈57%.
- ❌$\ell_2$: contrastive task accuracy ⬆️, representation quality ⬇️.
- Low $\tau$ → softmax weights hard negatives more → [InfoNCE](../../ml/obj.md#infonce) Q&A.

*Why global BN?*
- Positive pair on the same device + per-device BN stats → leak → model matches pairs via batch stats, ❌via content.
- Fix: aggregate BN mean & var over all devices.

*Which augmentations matter?*
- Crop + color distortion, composed → [Contrastive Views](../../cv/image.md#contrastive-views) Q&A.

*What does the loss optimize, geometrically?*
- As #negatives → ∞, InfoNCE optimizes **alignment** + **uniformity** on the unit hypersphere {cite:p}`wang2020understanding`.
    - Alignment: $\mathcal{L}_\text{align}=\mathbb{E}_\text{pos}\|\mathbf{z}-\mathbf{z}'\|_2^2$ → positives close.
    - Uniformity: $\mathcal{L}_\text{uniform}=\log\mathbb{E}_{i,j\text{ i.i.d.}}e^{-t\|\mathbf{z}_i-\mathbf{z}_j\|_2^2}$, $t>0$ → embeddings spread over the sphere → ❌complete collapse.

*Key numbers?*
- ResNet-50 (4×): 76.5% ImageNet linear eval = supervised ResNet-50.
- Same model, 1% labels, fine-tuned: 85.8% top-5.

*Cons?*
- Compute & memory ⬆️ ← large batch.
- False negatives → [InfoNCE](../../ml/obj.md#infonce) Q&A.
- Quality depends on hand-tuned $\mathcal{T}$.

*SimCLR vs CLIP?*
- Same symmetric InfoNCE over a $B\times B$ similarity matrix.
- [CLIP](../../cv/vit.md#clip): positive = (image, caption). 2 encoders, 2 modalities. Negatives only across modalities.
```

&nbsp;

### MoCo
- **Name**: Momentum Contrast {cite:p}`he2020momentum`
- **What**: Contrastive learning w/ a queue of past keys as negatives & an EMA key encoder.
- **Why**: Need many negatives that are also consistent.
    - End-to-end (SimCLR-style): #negatives tied to batch size → memory-bound.
    - Memory bank: many negatives, but keys from stale encoders all over the past epoch → inconsistent.
- **How**:
    1. Query $\mathbf{z}$ from encoder $\theta$ on view $\mathbf{v}$. Positive key $\mathbf{k}^+$ from encoder $\xi$ on view $\mathbf{v}'$, ❌grad.
    2. Negatives: queue of $K$ keys from previous batches.
    3. InfoNCE over $1+K$ logits, $\tau=0.07$.
    4. Backprop → $\theta$. $\xi$ ← EMA of $\theta$, $\lambda=0.999$.
    5. Enqueue the current keys. Dequeue the oldest batch.
    6. **Shuffling BN**: shuffle sample order across GPUs for the key encoder only.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $K$: Queue size, 65,536.
- Misc:
    - $\mathbf{z}$: Query (paper's $q$), $\ell_2$-normalized.
    - $\mathbf{k}^+$: Positive key, $\ell_2$-normalized.
    - $\mathbf{k}_1,\dots,\mathbf{k}_K$: Queued negative keys.

Objective:

$$
\mathcal{L}=-\log\frac{\exp(\mathbf{z}^\top\mathbf{k}^+/\tau)}{\exp(\mathbf{z}^\top\mathbf{k}^+/\tau)+\sum_{j=1}^{K}\exp(\mathbf{z}^\top\mathbf{k}_j/\tau)}
$$

Momentum update (no grads):

$$
\xi\leftarrow\lambda\xi+(1-\lambda)\theta,\qquad\lambda=0.999
$$

- Paper's $\theta_q,\theta_k,m$ → $\theta,\xi,\lambda$ here.
```

````{important} Code
:class: dropdown
```python
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

class MoCo(nn.Module):
    def __init__(self, encoder, d=128, K=65536, lam=0.999, tau=0.07):
        super().__init__()
        self.enc_q = encoder                                              ## theta: trained by grads
        self.enc_k = copy.deepcopy(encoder).requires_grad_(False)         ## xi: EMA copy, never backprop
        self.K, self.lam, self.tau = K, lam, tau
        self.register_buffer("queue", F.normalize(torch.randn(K, d), dim=-1))   ## random keys at start
        self.register_buffer("ptr", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def momentum_update(self):
        for pk, pq in zip(self.enc_k.parameters(), self.enc_q.parameters()):
            pk.mul_(self.lam).add_((1 - self.lam) * pq)                   ## slow drift -> keys stay consistent

    def forward(self, v1, v2):
        z = F.normalize(self.enc_q(v1), dim=-1)                           ## (B, d) queries
        with torch.no_grad():
            self.momentum_update()
            k = F.normalize(self.enc_k(v2), dim=-1)                       ## (B, d) positive keys
        l_pos = (z * k).sum(-1, keepdim=True)                             ## (B, 1)
        l_neg = z @ self.queue.clone().T                                  ## (B, K); clone: queue is overwritten below
        logits = torch.cat([l_pos, l_neg], dim=1) / self.tau
        labels = torch.zeros(len(z), dtype=torch.long, device=z.device)   ## positive sits at index 0
        loss = F.cross_entropy(logits, labels)
        ## FIFO: newest keys overwrite the oldest batch (assumes K % B == 0). Shuffling BN omitted (1 device).
        B, p = len(k), int(self.ptr)
        self.queue[p:p + B] = k
        self.ptr.fill_((p + B) % self.K)
        return loss

## Example: linear stand-in encoder, K = 32, B = 4
moco = MoCo(nn.Linear(16, 8), d=8, K=32)
loss = moco(torch.randn(4, 16), torch.randn(4, 16))
loss.backward()
print(loss.item() > 0, int(moco.ptr))                                    ## True 4
```
````

```{attention} Q&A
:class: dropdown
*Why a momentum encoder, ❌a copy of $\theta$ each step?*
- Queue holds keys from many past steps. Fast-changing key encoder → keys inconsistent w/ each other.
- Ablation (ResNet-50, linear eval): $\lambda=0$ → loss oscillates, fails. 0.9 → 55.2%. 0.99 → 57.8%. 0.999 → 59.0%. 0.9999 → 58.9%.

*Why a queue?*
- #negatives decoupled from batch size: $K=65{,}536$ w/ batch 256.
- Cheap: stores only $d$-dim keys, ❌their activations.

*Why shuffling BN?*
- Query & positive key in the same per-GPU batch → BN stats leak information → model "cheats" → low loss, poor features.
- Shuffle the key encoder's samples across GPUs → query & key see different BN stats.

*MoCo vs SimCLR?*
- Negatives: queue (MoCo) vs current batch (SimCLR).
- MoCo v2 adds SimCLR's MLP head, blur aug & cosine LR {cite:p}`chen2020improved`.
    - 800 epochs, batch 256: 71.1% > SimCLR 69.3% (1000 epochs, batch 4096).
- MoCo v2 paper: head & augmentation are orthogonal to the negative source. Queue → small batches suffice.

*Key number?*
- MoCo v1, ResNet-50: 60.6% ImageNet linear eval.

*Cons?*
- Queue keys are slightly stale ← older $\xi$. $\lambda$ trades consistency vs freshness.
- False negatives, as in all InfoNCE methods.
- Extra forward pass for keys.

*Legacy?*
- EMA target encoder → reused w/o negatives in [BYOL](#byol), [DINO](../../cv/vit.md#dino), [I-JEPA](#i-jepa).
```

&nbsp;

## Non-Contrastive
- **What**: Joint-embedding SSL w/o negative pairs.

&nbsp;

### BYOL
- **Name**: Bootstrap Your Own Latent {cite:p}`grill2020bootstrap`
- **What**: Online network predicts an EMA target network's projection of another view.
- **Why**: Contrastive methods need many negatives (large batch or queue) & strong color augmentation.
    - Drop negatives → need another collapse guard.
- **How**:
    1. 2 views. Online: $f_\theta\to g_\theta\to q_\theta$. Target: $f_\xi\to g_\xi$. $q$ only on the online side.
    2. Loss: squared $\ell_2$ distance between normalized prediction & normalized target projection = $2-2\cos$.
    3. Symmetrize: swap views, add.
    4. Grads → $\theta$ only. $\xi$ ← EMA of $\theta$.
    5. $\lambda$: 0.996 → 1, cosine schedule.
    6. Projector & predictor: MLP 4096 hidden → 256 out. Keep $f_\theta$.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\lambda_\text{base}$: Initial EMA rate, 0.996.
- Misc:
    - $\mathbf{z}_\theta=g_\theta(f_\theta(\mathbf{v}))$: Online projection.
    - $\mathbf{p}=q_\theta(\mathbf{z}_\theta)$: Online prediction.
    - $\mathbf{z}'_\xi=g_\xi(f_\xi(\mathbf{v}'))$: Target projection.
    - $k$: Training step.
    - $k_\text{max}$: #training steps.

Objective:

$$
\mathcal{L}_{\theta,\xi}=\left\|\frac{\mathbf{p}}{\|\mathbf{p}\|_2}-\frac{\mathbf{z}'_\xi}{\|\mathbf{z}'_\xi\|_2}\right\|_2^2=2-2\,\text{sim}(\mathbf{p},\mathbf{z}'_\xi)
$$

$$
\mathcal{L}^\text{BYOL}_{\theta,\xi}=\mathcal{L}_{\theta,\xi}+\tilde{\mathcal{L}}_{\theta,\xi}
$$

- $\tilde{\mathcal{L}}_{\theta,\xi}$: same loss w/ $\mathbf{v}'$ through the online net & $\mathbf{v}$ through the target net.

Updates:

$$
\theta\leftarrow\text{optimizer}\left(\theta,\nabla_\theta\mathcal{L}^\text{BYOL}_{\theta,\xi}\right),\qquad\xi\leftarrow\lambda\xi+(1-\lambda)\theta
$$

$$
\lambda=1-(1-\lambda_\text{base})\cdot\frac{\cos(\pi k/k_\text{max})+1}{2}
$$

- ❌$\nabla_\xi$ → BYOL does not minimize one loss jointly over $(\theta,\xi)$.
```

````{important} Code
:class: dropdown
```python
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class BYOL(nn.Module):
    def __init__(self, encoder, projector, predictor, lam_base=0.996):
        super().__init__()
        self.online = nn.Sequential(encoder, projector)                    ## f_theta -> g_theta
        self.predictor = predictor                                         ## q_theta: online branch ONLY
        self.target = copy.deepcopy(self.online).requires_grad_(False)     ## f_xi -> g_xi
        self.lam_base = lam_base

    @staticmethod
    def loss_fn(p, z):
        ## squared L2 between unit vectors = 2 - 2 cos; detach = stop-grad on the target
        return 2 - 2 * F.cosine_similarity(p, z.detach(), dim=-1)

    def forward(self, v1, v2):
        p1, p2 = self.predictor(self.online(v1)), self.predictor(self.online(v2))
        with torch.no_grad():
            z1, z2 = self.target(v1), self.target(v2)
        ## symmetrized: each view predicts the target projection of the OTHER view
        return (self.loss_fn(p1, z2) + self.loss_fn(p2, z1)).mean()

    @torch.no_grad()
    def update_target(self, k, k_max):
        lam = 1 - (1 - self.lam_base) * (math.cos(math.pi * k / k_max) + 1) / 2   ## 0.996 -> 1
        for pt, po in zip(self.target.parameters(), self.online.parameters()):
            pt.mul_(lam).add_((1 - lam) * po)

## Example: linear stand-in encoder, MLP projector & predictor, d = 4
mlp = lambda i, o: nn.Sequential(nn.Linear(i, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Linear(16, o))
byol = BYOL(nn.Linear(10, 8), mlp(8, 4), mlp(4, 4))
loss = byol(torch.randn(6, 10), torch.randn(6, 10))
loss.backward()
byol.update_target(k=0, k_max=100)
print(0 <= loss.item() <= 8)                                              ## True: each term in [0, 4]
```
````

```{attention} Q&A
:class: dropdown
*Why doesn't it collapse w/o negatives?*
- Collapse IS a minimizer of the loss. Nothing in the objective forbids it.
- Ablations (300 epochs, linear eval):
    - ❌Predictor, or ❌EMA target ($\lambda=0$) → collapse (<1%).
    - Fixed random target ($\lambda=1$) → 18.8%. ❌collapse, but weak.
    - EMA target, $\lambda=0.99$ → 72.5%.
- Paper's hypothesis: optimal predictor → online update follows the grad of $\mathbb{E}\left[\sum_j\text{Var}(z'_{\xi,j}\mid\mathbf{z}_\theta)\right]$.
    - Constant $\mathbf{z}_\theta$ maximizes this conditional variance → collapsed equilibria are unstable.
    - EMA target → changes slowly → predictor stays near-optimal.
    - Support: ❌target net + 10× predictor LR → 66.9% (❌collapse).
- Caveat: mechanism contested. No complete theory for deep nets.

*Does BN secretly supply negatives?*
- Claim: BN batch stats mix samples → implicit contrastive term.
- Test: replace BN w/ GroupNorm + Weight Standardization (❌batch stats) → 73.9% vs 74.3% {cite:p}`richemond2020byol`.
- → BN is not essential. Its main role ≈ good conditioning at init.

*Pros?*
- Robust to batch size: stable 256–4096. Drops below 256 only ← BN in the encoder.
- Less sensitive to removing color augmentation than SimCLR ← must keep any info the target encodes, ❌only enough to beat negatives.

*Key numbers?*
- ResNet-50: 74.3% ImageNet linear eval. Larger ResNet: 79.6%.

*Cons?*
- 2 networks → 2× forward cost.
- Interacting hyperparams: $\lambda$ schedule, predictor LR, weight decay.
- Silent failure mode: collapse gives a low loss.

*BYOL vs DINO?*
- Shared: EMA teacher, ❌negatives.
- BYOL: regression on continuous projections + predictor.
- [DINO](../../cv/vit.md#dino): CE over softmax "prototypes" + centering & sharpening, ❌predictor.

*Why "bootstrap"?*
- Idiomatic sense, ❌statistical bootstrap: the net's own past outputs serve as targets for its improved version.
```

&nbsp;

#### SimSiam
- **What**: BYOL w/o the EMA target: 1 shared encoder, predictor & stop-gradient. {cite:p}`chen2021exploring`
- **Why**: Find the minimal collapse guard.
    - Result: ❌negatives, ❌large batch, ❌momentum encoder. Stop-gradient is essential.
- **How**:
    1. 2 views → same $g_\theta\circ f_\theta$ → $\mathbf{z}_1,\mathbf{z}_2$. Projector: 3-layer MLP, 2048-d.
    2. Predictor $q_\theta$: bottleneck MLP 2048 → 512 → 2048 → $\mathbf{p}_1,\mathbf{p}_2$.
    3. Loss: negative cosine between $\mathbf{p}_1$ & $\text{sg}(\mathbf{z}_2)$. Symmetrize.
    4. SGD, batch 512 default.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\mathbf{z}_1,\mathbf{z}_2$: Embeddings of $\mathbf{v},\mathbf{v}'$.
    - $\mathbf{p}_1=q_\theta(\mathbf{z}_1)$, $\mathbf{p}_2=q_\theta(\mathbf{z}_2)$: Predictions.

Objective:

$$
\mathcal{D}(\mathbf{p},\mathbf{z})=-\text{sim}(\mathbf{p},\mathbf{z})
$$

$$
\mathcal{L}=\tfrac{1}{2}\mathcal{D}\left(\mathbf{p}_1,\text{sg}(\mathbf{z}_2)\right)+\tfrac{1}{2}\mathcal{D}\left(\mathbf{p}_2,\text{sg}(\mathbf{z}_1)\right)
$$

- $\mathcal{L}\in[-1,1]$. Collapse reaches $-1$.
- = BYOL loss w/ $\lambda=0$ (target = online), up to the affine map $2-2\cos$.
```

```{tip} Derivation
:class: dropdown
*Why does stop-grad make sense?* Paper's EM-like hypothesis (not a proof).
1. Add 1 free variable $\eta_\mathbf{x}$ per image. $F_\theta=g_\theta\circ f_\theta$.

    $$
    \mathcal{L}(\theta,\eta)=\mathbb{E}_{\mathbf{x},t\sim\mathcal{T}}\left[\left\|F_\theta(t(\mathbf{x}))-\eta_\mathbf{x}\right\|_2^2\right]
    $$

2. Solve $\min_{\theta,\eta}\mathcal{L}$ by alternation, as in k-means.
3. $\theta$-step: $\eta$ fixed → $\eta$ is a constant → ❌grad into it → **stop-gradient**.
4. $\eta$-step: MSE → $\eta_\mathbf{x}\leftarrow\mathbb{E}_{t}\left[F_\theta(t(\mathbf{x}))\right]$: mean embedding over augmentations.
5. One-step approximation: estimate $\eta_\mathbf{x}$ w/ 1 sampled view $t'$ → $\eta_\mathbf{x}\approx F_\theta(t'(\mathbf{x}))$.
6. Plug into step 3 → $\left\|F_\theta(t(\mathbf{x}))-\text{sg}\left(F_\theta(t'(\mathbf{x}))\right)\right\|^2$ = SimSiam w/o predictor.
7. Predictor $q$: trained to map $\mathbf{z}_1$ to $\mathbf{z}_2$ → optimum $q(\mathbf{z}_1)=\mathbb{E}_t[\mathbf{z}_2]$ → fills the gap left by the 1-sample estimate.
- MSE on $\ell_2$-normalized vectors ≡ cosine loss → same analysis.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimSiam(nn.Module):
    def __init__(self, encoder, predictor):
        super().__init__()
        self.encoder = encoder                        ## g_theta o f_theta, SHARED by both views
        self.predictor = predictor                    ## q_theta

    @staticmethod
    def D(p, z):
        return -F.cosine_similarity(p, z.detach(), dim=-1).mean()   ## detach = stop-grad

    def forward(self, v1, v2):
        z1, z2 = self.encoder(v1), self.encoder(v2)
        p1, p2 = self.predictor(z1), self.predictor(z2)
        return 0.5 * self.D(p1, z2) + 0.5 * self.D(p2, z1)

## Example: linear stand-ins
model = SimSiam(nn.Linear(10, 8), nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 8)))
loss = model(torch.randn(6, 10), torch.randn(6, 10))
loss.backward()
print(-1 <= loss.item() <= 1)                         ## True
```
````

```{attention} Q&A
:class: dropdown
*What happens w/o stop-grad?*
- Loss → $-1$ immediately. Per-dim std of $\ell_2$-normalized output → 0. Linear eval: 0.1% (chance).
- W/ stop-grad: 67.7% (100 epochs). Std ≈ $1/\sqrt{d}$.

*Why is the predictor needed?*
- ❌Predictor → 0.1%.
- ← Symmetric loss w/ $q=\text{id}$: grad has the same direction as the grad of $\mathcal{D}(\mathbf{z}_1,\mathbf{z}_2)$, w/o sg, scaled by ½ → sg becomes a no-op → collapse.
- Fixed random predictor → 1.5%. Predictor w/ constant LR → 68.1% > 67.7% w/ decayed LR.

*BYOL reports collapse at $\lambda=0$. Why does SimSiam work?*
- Predictor optimization decides it. BYOL w/o target net works too once its predictor LR is ⬆️ → [BYOL](#byol) Q&A.

*Batch size?*
- 64 → 66.1%. 256–2048 → ≈68%. 4096 → 64.0%.
- Same plain SGD at every batch size, ❌LARS.

*Key numbers?*
- ResNet-50, batch 256: 68.1% (100 epochs, best among reproduced baselines at 100 epochs) → 71.3% (800 epochs).

*Is there a theory?*
- Linear-network analysis {cite:p}`tian2021understanding`:
    - Predictor eigenspace aligns w/ the eigenspace of its input's correlation matrix.
    - **DirectPred**: set the linear predictor from that eigendecomposition, ❌grads → 72.4% vs 72.5% for BYOL's 2-layer predictor (300 epochs).
    - EMA, weight decay & predictor LR all shape the dynamics.
```

&nbsp;

### Barlow Twins
- **What**: Push the cross-correlation matrix of the 2 views' embeddings toward identity. {cite:p}`zbontar2021barlow`
- **Why**: Collapse guard by construction, ❌negatives, ❌asymmetry.
    - Inspired by H. Barlow's **redundancy reduction** principle: good sensory codes have statistically independent components.
- **How**:
    1. 2 views → same net → $Z,Z'$. Projector: 3 linear layers × 8192.
    2. Standardize each dim over the batch: mean 0, std 1.
    3. Cross-correlation $C=\frac{1}{B}Z^\top Z'$ ($d\times d$).
    4. Diagonal → 1: **invariance** ← dim $j$ agrees across views.
    5. Off-diagonal → 0: **redundancy reduction** ← dims $j\neq j'$ carry different information.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\alpha_\text{off}$: Off-diagonal weight, $5\cdot10^{-3}$ (paper's $\lambda$).
- Misc:
    - $z_{ij}$: Entry $(i,j)$ of $Z$, mean-centered over the batch.
    - $j,j'$: Embedding dim idx.
    - $C\in[-1,1]^{d\times d}$: Cross-correlation matrix.

Objective:

$$
C_{jj'}=\frac{\sum_{i=1}^{B}z_{ij}\,z'_{ij'}}{\sqrt{\sum_{i=1}^{B}z_{ij}^2}\sqrt{\sum_{i=1}^{B}{z'_{ij'}}^2}}
$$

$$
\mathcal{L}_\text{BT}=\underbrace{\sum_{j}(1-C_{jj})^2}_\text{invariance}+\alpha_\text{off}\underbrace{\sum_{j}\sum_{j'\neq j}C_{jj'}^2}_\text{redundancy reduction}
$$

- Paper indexes batch by $b$ & dims by $i,j$ → $i$ & $j,j'$ here.
- Mean-centered columns ⊥ $\mathbf{1}$ → $\text{rank}(C)\le\min(B-1,d)$ → $C=I$ reachable only if $B>d$.
- Constant dim → 0/0. Implementations add $\epsilon$ to the std (or use affine-free BN) → constant dim standardizes to 0 → $C_{jj}=0$.
```

````{important} Code
:class: dropdown
```python
import torch

class BarlowTwinsLoss:
    def __init__(self, alpha_off=5e-3, eps=1e-5):
        self.alpha_off, self.eps = alpha_off, eps

    def __call__(self, z1, z2):                                   ## (B, d) each
        B = z1.size(0)
        ## standardize each dim over the BATCH -> C entries are correlations in [-1, 1]
        z1 = (z1 - z1.mean(0)) / (z1.std(0, unbiased=False) + self.eps)
        z2 = (z2 - z2.mean(0)) / (z2.std(0, unbiased=False) + self.eps)
        C = z1.T @ z2 / B                                         ## (d, d) cross-correlation
        diag = torch.diagonal(C)
        on = (1 - diag).pow(2).sum()                              ## invariance: dim j agrees across views
        off = C.pow(2).sum() - diag.pow(2).sum()                  ## redundancy: dims j != j' decorrelate
        return on + self.alpha_off * off

## Example
torch.manual_seed(0)
crit, z = BarlowTwinsLoss(), torch.randn(16, 8)
print(crit(z, z).item() < crit(z, torch.randn(16, 8)).item())   ## True: identical views -> diag ~ 1 (eps)
print(crit(torch.ones(16, 8), torch.ones(16, 8)))               ## tensor(8.) -> complete collapse costs d
```
````

```{attention} Q&A
:class: dropdown
*Why can't it collapse?*
- Complete: constant dim → zero variance → standardized to 0 (w/ $\epsilon$) → $C_{jj}=0$ → invariance term $=d$, far from its min 0.
- Dimensional: copied or correlated dims → off-diagonal $C_{jj'}\neq0$ → penalized.

*Pros?*
- ❌Negatives, ❌predictor, ❌EMA, ❌stop-grad. Symmetric.
- Robust to batch size: ≈ unaffected at batch 256. SimCLR drops ≈4 points there.
- Gains from a very wide projector: keeps improving at every tested output dim. SimCLR & BYOL saturate.

*Can $C$ actually reach $I$?*
- Only if $B>d$ ← $\text{rank}(C)\le B-1$ after mean-centering.
- Default: $B=2048$, $d=8192$ → $C=I$ unreachable. Training still works ← the loss only needs a useful descent direction.

*Cons?*
- $d\times d$ matrix → $O(d^2)$ memory & compute. $d=8192$ → 67M entries.
- Needs batch statistics → per-dim std from small batches is noisy.

*Breaking symmetry?*
- + Predictor and/or stop-grad, as in BYOL/SimSiam → slightly worse.

*Why does it work, information-theoretically?*
- Paper: instance of the **Information Bottleneck**: $\mathbf{z}$ informative about the image, uninformative about the distortion.
- Loss = proxy entropy estimator under a Gaussian parametrization → cheap from few samples, even at large $d$.

*Key number?*
- ResNet-50, 1000 epochs, batch 2048: 73.2% ImageNet linear eval.
```

&nbsp;

### VICReg
- **Name**: Variance-Invariance-Covariance Regularization {cite:p}`bardes2022vicreg`
- **What**: View-matching MSE + per-dim std hinge + off-diagonal covariance penalty, regularized per branch.
- **Why**: 1 explicit loss term per anti-collapse job.
    - Barlow Twins mixes both branches in 1 standardized matrix. VICReg regularizes each branch alone.
    - → Branches may differ: separate weights, architectures, modalities.
- **How**:
    1. 2 views → $Z,Z'$. Expander (projector): 3 layers × 8192.
    2. **Invariance**: MSE between $Z$ & $Z'$.
    3. **Variance**: std of each dim over the batch ≥ $\gamma=1$ (hinge), per branch.
    4. **Covariance**: off-diagonal covariance → 0, per branch.
    5. Weighted sum. ❌Weight sharing required, ❌BN on embeddings, ❌stop-grad, ❌predictor, ❌EMA.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\alpha_\text{inv}=25$, $\alpha_\text{var}=25$, $\alpha_\text{cov}=1$: Term weights (paper's $\lambda,\mu,\nu$).
    - $\gamma=1$: Target std.
    - $\epsilon=10^{-4}$: Numerical stabilizer.
- Misc:
    - $Z_{:,j}\in\mathbb{R}^{B}$: Dim $j$ over the batch.
    - $C(Z)\in\mathbb{R}^{d\times d}$: Covariance of $Z$.

Objective:

$$
s(Z,Z')=\frac{1}{Bd}\sum_{i=1}^{B}\|\mathbf{z}_i-\mathbf{z}'_i\|_2^2
$$

- Paper eq. (5): $\frac{1}{B}$. Paper pseudocode (`mse_loss`): $\frac{1}{Bd}$. Form here = pseudocode.

$$
v(Z)=\frac{1}{d}\sum_{j=1}^{d}\max\left(0,\ \gamma-\sqrt{\text{Var}(Z_{:,j})+\epsilon}\right)
$$

$$
C(Z)=\frac{1}{B-1}\sum_{i=1}^{B}(\mathbf{z}_i-\bar{\mathbf{z}})(\mathbf{z}_i-\bar{\mathbf{z}})^\top,\qquad c(Z)=\frac{1}{d}\sum_{j\neq j'}C(Z)_{jj'}^2
$$

$$
\ell(Z,Z')=\alpha_\text{inv}\,s(Z,Z')+\alpha_\text{var}\left[v(Z)+v(Z')\right]+\alpha_\text{cov}\left[c(Z)+c(Z')\right]
$$
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn.functional as F

class VICRegLoss:
    def __init__(self, a_inv=25.0, a_var=25.0, a_cov=1.0, gamma=1.0, eps=1e-4):
        self.a_inv, self.a_var, self.a_cov, self.gamma, self.eps = a_inv, a_var, a_cov, gamma, eps

    def var(self, z):                                             ## (B, d)
        std = torch.sqrt(z.var(dim=0) + self.eps)                 ## per-dim std over the batch
        return F.relu(self.gamma - std).mean()                    ## hinge: only dims w/ std < gamma pay

    def cov(self, z):
        B, d = z.shape
        z = z - z.mean(0)
        C = z.T @ z / (B - 1)                                     ## (d, d) covariance
        return (C.pow(2).sum() - torch.diagonal(C).pow(2).sum()) / d   ## off-diagonal only

    def __call__(self, z1, z2):
        inv = F.mse_loss(z1, z2)                                  ## mean over B*d (paper pseudocode)
        return (self.a_inv * inv
                + self.a_var * (self.var(z1) + self.var(z2))      ## each branch on its own
                + self.a_cov * (self.cov(z1) + self.cov(z2)))

## Example: same views, healthy vs collapsed embeddings
torch.manual_seed(0)
crit, z = VICRegLoss(), torch.randn(32, 8)
zero = torch.zeros(32, 8)
print(crit(z, z).item() < crit(zero, zero).item())                ## True: collapse -> variance hinge ~ 2 * 25 * 0.99
```
````

```{attention} Q&A
:class: dropdown
*What does each term prevent?*
- Invariance alone → complete collapse is optimal.
- + Variance → ❌complete collapse. Dims can still be copies of each other → dimensional collapse.
- + Covariance → dims decorrelated → ❌dimensional collapse.

*Why std, ❌variance, inside the hinge?*
- Var: $\frac{\partial\,\text{Var}}{\partial z_{ij}}=\frac{2(z_{ij}-\bar{z}_j)}{B-1}$ → shrinks w/ the spread $\sqrt{\text{Var}}$ → restoring force fades as dims shrink → collapse.
- Std: $\frac{\partial\sqrt{\text{Var}+\epsilon}}{\partial z_{ij}}=\frac{z_{ij}-\bar{z}_j}{(B-1)\sqrt{\text{Var}+\epsilon}}$ → $|z_{ij}-\bar{z}_j|\sim\sqrt{\text{Var}}$ → grad ≈ independent of the spread while $\text{Var}\gg\epsilon$.
- $\text{Var}\ll\epsilon$ (exact collapse) → grad → 0 for std too. $\epsilon$ only blocks division by 0.

*Why a hinge, ❌exact std $=\gamma$?*
- Only a floor is needed to block collapse. Above $\gamma$ → free.

*VICReg vs Barlow Twins?*
- Barlow: 1 cross-correlation matrix → invariance & decorrelation coupled through standardization.
- VICReg: invariance = raw MSE. Variance & covariance per branch → ❌standardization, ❌shared weights needed.
- Same score: 73.2% ResNet-50 linear eval. Both $O(d^2)$.

*Pros?*
- Branches can use different architectures or modalities.
- Variance term = plug-in regularizer: stabilizes & improves other methods.

*Cons?*
- 3 weights + $\gamma$ to tune.
- Batch statistics → covariance estimate noisy for small $B$.

*Relation to JEPA?*
- [JEPA](#jepa) proposal: VICReg-style terms keep $\mathbf{s}_x$, $\mathbf{s}_y$ informative.
```

&nbsp;

## Predictive
- **What**: Predict a hidden part of the input from the visible part.

&nbsp;

### JEPA
- **Name**: Joint-Embedding Predictive Architecture {cite:p}`lecun2022path`
- **What**: Predict the embedding of a target signal $y$ from the embedding of a context signal $x$.
- **Why**: Generative models predict every detail of $y$ in input space.
    - Many details are unpredictable (texture, noise) → capacity wasted, averaged predictions.
    - Embedding space can drop unpredictable detail → predict only the predictable, abstract part.
    - Invariance-based joint embedding (SimCLR, BYOL, VICReg) → views of 1 image → same embedding → forced invariance to hand-crafted $\mathcal{T}$. BYOL's predictor does not change this: it gets ❌target location or latent.
    - JEPA: predictor conditioned on $\mathbf{r}$ (e.g., target position) models the $x\to y$ relation → ❌hand-crafted invariance.
- **How**:
    1. Encode: $\mathbf{s}_x=\text{Enc}_x(x)$, $\mathbf{s}_y=\text{Enc}_y(y)$.
    2. Predict: $\hat{\mathbf{s}}_y=\text{Pred}(\mathbf{s}_x,\mathbf{r})$. Latent $\mathbf{r}$ carries what $x$ cannot determine about $y$.
    3. Energy = prediction error in embedding space. Low on compatible $(x,y)$.
    4. Train non-contrastively. 4 criteria:
        - Max information in $\mathbf{s}_x$ about $x$.
        - Max information in $\mathbf{s}_y$ about $y$.
        - Min prediction error.
        - Min information in $\mathbf{r}$.
    5. **H-JEPA**: stack JEPAs → higher levels = more abstract, longer-horizon prediction.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $x$, $y$: Context & target signal (override: page $\mathbf{x}$ = whole input, Tier-1 $y$ = output).
- Misc:
    - $\mathbf{s}_x,\mathbf{s}_y$: Embeddings of $x$, $y$.
    - $\mathbf{r}\in\mathcal{R}$: Latent variable (paper's $z$ ← page $\mathbf{z}$ = embedding).
    - $D$: Distance in embedding space.

Model:

$$
E(x,y,\mathbf{r})=D\left(\mathbf{s}_y,\ \text{Pred}(\mathbf{s}_x,\mathbf{r})\right),\qquad F(x,y)=\min_{\mathbf{r}\in\mathcal{R}}E(x,y,\mathbf{r})
$$

- $F$: Free energy. Low → $y$ compatible w/ $x$.
- Info in $\mathbf{r}$ limited (low dim, sparse, discrete, or noisy) → $\mathbf{r}$ cannot encode $y$ on its own.
```

```{dropdown} Table: Architectures
| Architecture | Loss computed on | Collapse risk | Needs hand-crafted views |
|:--|:--|:--|:--|
| Joint-Embedding (SimCLR, BYOL, VICReg) | $D(\text{Enc}(x),\text{Enc}(y))$ | ✅ | ✅ |
| Generative (MAE, BEiT, MLM) | $D(y,\text{Dec}(\text{Enc}(x),\mathbf{r}))$ in input space | ❌ if $\mathbf{r}$ low-capacity | ❌ |
| JEPA (I-JEPA) | $D(\mathbf{s}_y,\text{Pred}(\mathbf{s}_x,\mathbf{r}))$ in embedding space | ✅ | ❌ |
```

```{attention} Q&A
:class: dropdown
*Is JEPA empirically validated?*
- The source is a position paper: a research agenda, ❌experiments.
- [I-JEPA](#i-jepa) = a later image instantiation w/ results.

*Why is JEPA collapse-prone?*
- Both encoders trainable → constant $\mathbf{s}_x=\mathbf{s}_y$ → zero error.
- Generative: target $y$ is fixed data → ❌collapse.
- Guards: info-maximizing regularizer on $\mathbf{s}_x$, $\mathbf{s}_y$ ([VICReg](#vicreg) style), or EMA target + asymmetry ([I-JEPA](#i-jepa)).

*Why the latent $\mathbf{r}$?*
- $y$ not fully determined by $x$ → many valid $y$. Deterministic predictor → predicts their average.
- $\mathbf{r}$ indexes the alternatives.
- Too much info in $\mathbf{r}$ → $\mathbf{r}$ encodes $y$ itself → predictor ignores $\mathbf{s}_x$ → energy low everywhere → "min info in $\mathbf{r}$".
- I-JEPA: $\mathbf{r}$ = mask tokens + positions of the target block → low capacity.

*JEPA vs generative world models?*
- Generative: predict future frames pixel by pixel.
- JEPA: predict future embeddings → plan by optimizing actions in embedding space.

*What does it say about intelligence?*
- Claim: humans & animals learn most world knowledge by observation, w/o labels or rewards.
- → SSL world models = core module of autonomous intelligence.
```

&nbsp;

#### I-JEPA
- **Name**: Image-based Joint-Embedding Predictive Architecture {cite:p}`assran2023self`
- **What**: JEPA for images: from 1 context block, predict the patch embeddings of several target blocks.
- **Why**: Semantic features w/o hand-crafted view augmentations.
    - Invariance methods: semantic, but biased by $\mathcal{T}$.
    - Pixel reconstruction ([MAE](../../cv/vit.md#mae)): ❌views, but lower semantic level → weak linear probe.
- **How**:
    1. Patchify. Target encoder $f_\xi$ (ViT) on the **full** image → patch embeddings $\mathbf{s}_y$.
    2. Sample $M=4$ target blocks: scale 0.15–0.2, aspect ratio 0.75–1.5. Mask the target encoder's **output**, ❌input.
    3. Sample 1 context block: scale 0.85–1.0. Remove its overlap w/ targets. Context encoder $f_\theta$ (ViT) on these patches only.
    4. Predictor $q_\phi$ (narrow ViT, width 384): context embeddings + 1 mask token per target patch (shared learnable vector + position embedding) → predicted embeddings. 1 pass per target block.
    5. Squared $\ell_2$ loss on target patches. Grads → $\theta,\phi$. $\xi$ ← EMA of $\theta$, $\lambda$: 0.996 → 1 (linear).
    6. Eval: target encoder, average-pooled patch embeddings. ❌[CLS].

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathcal{B}_i$: Patch idx set of target block $i$.
- Params:
    - $\phi$: Predictor params.
- Hyperparams:
    - $M$: #target blocks, 4.
- Misc:
    - $\mathbf{s}_{y_j}$: Target-encoder embedding of patch $j$, full image as input.
    - $\hat{\mathbf{s}}_{y_j}$: Predicted embedding of patch $j$.
    - $\mathbf{m}_j$: Mask token for patch $j$ (shared vector + position $j$).

Objective:

$$
\hat{\mathbf{s}}_y(i)=q_\phi\left(f_\theta(x),\{\mathbf{m}_j\}_{j\in\mathcal{B}_i}\right),\qquad\mathbf{s}_y=\text{sg}\left(f_\xi(\text{full image})\right)
$$

$$
\mathcal{L}=\frac{1}{M}\sum_{i=1}^{M}\sum_{j\in\mathcal{B}_i}\left\|\hat{\mathbf{s}}_{y_j}-\mathbf{s}_{y_j}\right\|_2^2
$$

- $x$: context block patches.
```

````{important} Code
:class: dropdown
```python
import copy
import torch
import torch.nn as nn

class IJEPA(nn.Module):
    def __init__(self, N=16, D=32, D_pred=16, heads=4, L=2, L_pred=1):
        super().__init__()
        layer = lambda d: nn.TransformerEncoderLayer(d, heads, 4 * d, batch_first=True, norm_first=True)
        ## FIXED positions (official: 2D sin-cos, requires_grad=False) -> target input changes only via EMA
        self.pos = nn.Parameter(torch.randn(1, N, D) * 0.02, requires_grad=False)
        self.context_enc = nn.TransformerEncoder(layer(D), L)                        ## f_theta
        self.target_enc = copy.deepcopy(self.context_enc).requires_grad_(False)     ## f_xi: EMA only
        self.to_pred, self.from_pred = nn.Linear(D, D_pred), nn.Linear(D_pred, D)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, D_pred))                    ## 1 shared vector
        self.pred_pos = nn.Parameter(torch.randn(1, N, D_pred) * 0.02)
        self.predictor = nn.TransformerEncoder(layer(D_pred), L_pred)                ## q_phi: NARROW

    def forward(self, tokens, ctx, targets):
        ## tokens: (B, N, D) patch embeddings; ctx: context idx; targets: list of target-block idx
        x = tokens + self.pos
        with torch.no_grad():
            s_y = self.target_enc(x)                                     ## FULL image, then mask the OUTPUT
        s_x = self.to_pred(self.context_enc(x[:, ctx]))                  ## encoder sees context patches only
        loss = 0.0
        for t in targets:                                                ## 1 predictor pass per target block
            m = (self.mask_token + self.pred_pos[:, t]).expand(len(x), -1, -1)   ## "predict at these positions"
            h = self.predictor(torch.cat([s_x + self.pred_pos[:, ctx], m], dim=1))
            pred = self.from_pred(h[:, len(ctx):])                       ## outputs at the mask tokens
            loss = loss + (pred - s_y[:, t]).pow(2).sum(dim=(1, 2)).mean()   ## sum over block, mean over batch
        return loss / len(targets)

    @torch.no_grad()
    def update_target(self, lam=0.996):
        for pt, pc in zip(self.target_enc.parameters(), self.context_enc.parameters()):
            pt.mul_(lam).add_((1 - lam) * pc)

## Example: 4x4 patch grid (N = 16); 2 target blocks; context = remaining patches
model = IJEPA()
targets = [torch.tensor([0, 1, 4, 5]), torch.tensor([10, 11, 14, 15])]
ctx = torch.tensor([2, 3, 6, 7, 8, 9, 12, 13])
loss = model(torch.randn(2, 16, 32), ctx, targets)
loss.backward()
model.update_target()
print(loss.item() > 0)                                                   ## True
```
````

```{attention} Q&A
:class: dropdown
*Why mask the target encoder's output, ❌its input?*
- Target embeddings then see the full image → higher semantic level. Paper: crucial.

*Why large target blocks & a big context?*
- Small targets → local, low-level prediction (texture).
- Targets 15–20% of the image → semantic content. Context must be informative & spatially spread.

*Embedding vs pixel targets?*
- ViT-L/16, 1% ImageNet labels, linear eval: target-encoder embeddings 66.9% vs pixels 40.7%.

*Why a narrow predictor?*
- Width 384 vs 1024 (ViT-L encoder): 70.7% vs 68.4% (1% labels, fine-tuned). Bottleneck helps.

*Key numbers?*
- ImageNet linear eval: ViT-H/14 (300 epochs) 79.3%. ViT-H/16 at 448² → 81.1%.
- MAE ViT-H/14 (1600 epochs): 77.2%.
- Cost: ViT-H/14 on ImageNet, 16 A100 GPUs, <72 h.

*How does it avoid collapse?*
- EMA target encoder (essential for ViT joint-embedding training, per paper) + predictor + masking asymmetry.
- ❌Explicit variance term → open questions of [BYOL](#byol) apply.

*I-JEPA vs MAE?*
- Shared: mask, predict missing patches, ViT encoder on visible patches only.
- MAE: decoder → pixels. I-JEPA: predictor → EMA-target embeddings.
```

&nbsp;
