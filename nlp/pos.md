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
How a Transformer learns token order. Covers absolute, relative & rotary schemes, RoPE context extension, and models with no explicit position signal.

Notations:
- $m$: #Tokens in sequence
- $d$: Model dim
- $h$: #Heads
- $d_k$: Dim per query/key head
- $i$: Query position
- $j$: Key position
- $\mathbf{x}_i\in\mathbb{R}^d$: Token embedding at position $i$
- $\mathbf{q}_i,\mathbf{k}_j,\mathbf{v}_j\in\mathbb{R}^{d_k}$: Query/Key/Value vector at position $i$/$j$
- $s_{ij}$: Attention logit of query $i$ on key $j$
- $L$: Training context length
- $L'$: Target (extended) context length
- $s=L'/L$: Context scale factor
- $b$: RoPE base

Override: $m$ = #tokens (global $m$ = #samples). $i,j$ = positions (global $i,j$ = sample/feature idx).

&nbsp;

## Permutation Equivariance
- **What**: Permute the input tokens → the output rows permute the same way. Nothing else changes.
- **Why**: Self-attention, FFN & LayerNorm have this property → a Transformer w/o position info sees a **bag of tokens**.
    - "dog bites man" & "man bites dog" → same set of output vectors.
- **How**: Break the symmetry. Inject position at one of 3 points:
    1. **Input**: add a position vector to $\mathbf{x}_i$ → [Absolute](#absolute).
    2. **Logits**: add a bias that depends on $j-i$ to $s_{ij}$ → [Relative](#relative).
    3. **Query/Key**: rotate $\mathbf{q}_i,\mathbf{k}_j$ by position-dependent angles → [RoPE](#rope).
    - Or inject nothing & rely on the causal mask → [NoPE](#nope).

```{tip} Derivation
:class: dropdown
*Why is self-attention permutation-equivariant?*

1. Let $P\in\{0,1\}^{m\times m}$ be a permutation matrix. $P^\top P=I$.
2. Permuted input $PX$ → $Q'=PQ$, $K'=PK$, $V'=PV$.
3. Logits: $Q'K'^\top=PQK^\top P^\top$ → rows & columns permuted the same way.
4. Row-wise softmax commutes with this: $\mathrm{softmax}(PSP^\top)=P\,\mathrm{softmax}(S)P^\top$.
5. Output: $P\,\mathrm{softmax}(S)P^\top PV=P\,\mathrm{softmax}(S)V$ → $\mathrm{Attn}(PX)=P\,\mathrm{Attn}(X)$.
6. FFN & LayerNorm act on each row independently → also equivariant.
7. → A stack of these layers is equivariant.
8. A causal mask $M$ breaks step 4: $PMP^\top\neq M$ for a non-identity $P$ → [NoPE](#nope) works only in decoders.
```

```{dropdown} Table: PE Schemes
| Scheme | Injected at | Learned params | Relative? | Extrapolates past $L$? | Used by |
|:--|:--|:--|:--|:--|:--|
| [Sinusoidal](#sinusoidal) | Input | ❌ | Partly | ❌ (a few dozen tokens) | Original Transformer |
| [Learned](#learned) | Input | $L\times d$ | ❌ | ❌ (no row for $i\geq L$) | BERT, GPT-2 |
| [RPR](#relative-position-representations) | Logits & values | $(2c+1)\times d_k\times2$ | ✅ (clipped) | ✅ beyond clip | Shaw et al. |
| [T5 Bias](#t5-bias) | Logits | $32\times h$ | ✅ (bucketed) | Partly | T5 |
| [ALiBi](#alibi) | Logits | ❌ | ✅ | ✅ (PPL) | BLOOM |
| [RoPE](#rope) | $\mathbf{q},\mathbf{k}$ | ❌ | ✅ | ❌ w/o [extension](#context-extension) | Llama, Qwen, Mistral, DeepSeek |
| [NoPE](#nope) | Nowhere | ❌ | Implicit | Task-dependent | Global layers of Llama 4 |
```

```{attention} Q&A
:class: dropdown
*Is position info needed at all?*
- Encoder (bidirectional): yes. W/o it, the model cannot tell word order.
- Decoder (causal): not strictly. The causal mask leaks position → [NoPE](#nope).

*Absolute vs relative position — which does language need?*
- Mostly relative: "the word 2 tokens back" means the same thing at position 5 & 5000.
- Relative schemes → translation-invariant → better length generalization.
```

&nbsp;

## Absolute
- **What**: Position $i$ → vector $\mathbf{p}_i\in\mathbb{R}^d$, added to $\mathbf{x}_i$ before layer 1.

### Sinusoidal
- **What**: Fixed sine/cosine waves of geometrically spaced frequencies, one frequency per dim pair. {cite:p}`vaswani2017attention`
- **Why**: Unique, smooth, param-free position code.
    - Each offset $\Delta$ is a fixed linear map → model can learn to attend by relative position.
    - Defined for any $i$ → hoped to extrapolate past $L$.
- **How**:
    1. Each dim pair $(2n,2n+1)$ gets a frequency $\omega_n$. Pair 0 = fastest. Last pair = slowest.
    2. Encode $i$ as $(\sin\omega_n i,\cos\omega_n i)$ for every pair.
    3. Add $\mathbf{p}_i$ to $\mathbf{x}_i$.
    - Clock analogy: fast pairs = second hand (fine position). Slow pairs = hour hand (coarse position).

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $n\in\{0,\dots,d/2-1\}$: Dim-pair index.
    - $\omega_n$: Angular frequency of pair $n$.
    - $\Delta$: Position offset.

Definition:

$$
p_{i,2n}=\sin(\omega_n i),\qquad p_{i,2n+1}=\cos(\omega_n i),\qquad \omega_n=10000^{-2n/d}
$$

- Wavelengths $2\pi/\omega_n$: geometric progression from $2\pi$ to $<10000\cdot2\pi$.

Properties:
- Shift by $\Delta$ = rotation of each pair (independent of $i$):

    $$
    \begin{bmatrix}\sin\omega_n(i+\Delta)\\ \cos\omega_n(i+\Delta)\end{bmatrix}=\begin{bmatrix}\cos\omega_n\Delta & \sin\omega_n\Delta\\ -\sin\omega_n\Delta & \cos\omega_n\Delta\end{bmatrix}\begin{bmatrix}\sin\omega_n i\\ \cos\omega_n i\end{bmatrix}
    $$

- Dot product depends only on the offset:

    $$
    \mathbf{p}_i^\top\mathbf{p}_j=\sum_{n=0}^{d/2-1}\cos\big(\omega_n(i-j)\big)
    $$
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class SinusoidalPE(nn.Module):
    def __init__(self, d, max_len=4096):
        super().__init__()
        i = torch.arange(max_len)[:, None]                     ## (max_len, 1)
        omega = 10000 ** (-torch.arange(0, d, 2) / d)          ## (d/2,): fast -> slow
        pe = torch.zeros(max_len, d)
        pe[:, 0::2] = torch.sin(i * omega)                     ## even dims
        pe[:, 1::2] = torch.cos(i * omega)                     ## odd dims
        self.register_buffer("pe", pe)                         ## fixed, not a parameter

    def forward(self, x):
        ## x: (B, m, d) -> add the first m position vectors
        return x + self.pe[: x.shape[1]]

## Example
x = torch.randn(2, 5, 8)
print(SinusoidalPE(8)(x).shape)  ## torch.Size([2, 5, 8])
```
````

```{attention} Q&A
:class: dropdown
*Why both sin & cos?*
- A shift by $\Delta$ is a rotation of $(\sin,\cos)$. Sin alone → $\sin\omega(i+\Delta)$ is not a linear function of $\sin\omega i$ alone.

*Why base 10000?*
- Sets the slowest wavelength ≈ $10000\cdot2\pi$ → far longer than training sequences → slow pairs never wrap → no two positions collide.
- An empirical choice. No derivation in the paper.

*Does it extrapolate past $L$?*
- The paper hoped so. Measured: ❌. LM PPL degrades a few dozen tokens beyond $L$. {cite:p}`press2022train`

*Is it truly relative?*
- Only before projection. $\mathbf{p}_i^\top\mathbf{p}_j$ depends on $i-j$. But attention sees $(\mathbf{x}_i+\mathbf{p}_i)^\top W_QW_K^\top(\mathbf{x}_j+\mathbf{p}_j)$ → cross terms $\mathbf{x}_i^\top W_QW_K^\top\mathbf{p}_j$ depend on absolute $j$.
- $\mathbf{p}_i^\top\mathbf{p}_j$ is symmetric in $i-j$ → cannot tell left from right. $W_QW_K^\top$ is needed for direction.

*Why add instead of concatenate?*
- Add keeps $d$ fixed. Concat costs extra dims in every layer.
- The model can learn to keep content & position in near-orthogonal subspaces. This is a common rationale, not an ablation from the paper.
```

&nbsp;

### Learned
- **What**: Trainable table $P\in\mathbb{R}^{L\times d}$. Row $i$ = position vector. {cite:p}`gehring2017convolutional`
- **Why**: Let data decide the position code. ❌Hand-designed prior.
- **How**: $\mathbf{x}_i\leftarrow\mathbf{x}_i+P_i$. Train $P$ with the rest of the model.

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class LearnedPE(nn.Module):
    def __init__(self, d, L):
        super().__init__()
        self.P = nn.Parameter(torch.randn(L, d) * 0.02)  ## one trainable row per position

    def forward(self, x):
        m = x.shape[1]
        assert m <= self.P.shape[0], "no row exists for positions >= L"
        return x + self.P[:m]

## Example
x = torch.randn(2, 5, 8)
print(LearnedPE(8, L=16)(x).shape)  ## torch.Size([2, 5, 8])
```
````

```{attention} Q&A
:class: dropdown
*Learned vs sinusoidal — which is better?*
- Original Transformer: nearly identical (EN-DE dev BLEU 25.7 vs 25.8). {cite:p}`vaswani2017attention`

*Cons?*
- Hard length limit: no row for $i\geq L$ (BERT: 512, GPT-2: 1024).
- Rows for rare large $i$ get few updates → undertrained tail.
- Absolute → ❌translation invariance.
```

&nbsp;

## Relative
- **What**: Make $s_{ij}$ depend on the offset $j-i$, not on $i$ or $j$ alone.
- **Why**: Same offset → same relation at any absolute position → translation invariance & better length generalization.

### Relative Position Representations
- **What**: Learned vectors per clipped offset, added to keys (and values) inside attention. {cite:p}`shaw2018self`
- **Why**: Absolute PE must learn relative patterns indirectly. RPR encodes the offset directly.
- **How**:
    1. Clip offset $j-i$ to $[-c,c]$ → $2c+1$ labels.
    2. Look up learned $\mathbf{a}^K_{ij},\mathbf{a}^V_{ij}$ for that label.
    3. Logit: use $\mathbf{k}_j+\mathbf{a}^K_{ij}$. Output: use $\mathbf{v}_j+\mathbf{a}^V_{ij}$.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\mathbf{w}^K_r,\mathbf{w}^V_r\in\mathbb{R}^{d_k}$: Learned vectors for clipped offset $r\in\{-c,\dots,c\}$.
- Hyperparams:
    - $c$: Clipping distance (paper: $k$).
- Misc:
    - $\alpha_{ij}$: Attention weight, $\mathrm{softmax}_j(s_{ij})$.

Forward:

$$\begin{align*}
&\mathbf{a}^K_{ij}=\mathbf{w}^K_{\mathrm{clip}(j-i,c)},\qquad \mathbf{a}^V_{ij}=\mathbf{w}^V_{\mathrm{clip}(j-i,c)},\qquad \mathrm{clip}(r,c)=\max(-c,\min(c,r)) \\
&s_{ij}=\frac{\mathbf{q}_i^\top(\mathbf{k}_j+\mathbf{a}^K_{ij})}{\sqrt{d_k}},\qquad \mathbf{z}_i=\sum_j\alpha_{ij}(\mathbf{v}_j+\mathbf{a}^V_{ij})
\end{align*}$$
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class RelativePositionAttention(nn.Module):
    def __init__(self, d_k, c):
        super().__init__()
        self.c = c
        self.wK = nn.Embedding(2 * c + 1, d_k)  ## one vector per clipped offset
        self.wV = nn.Embedding(2 * c + 1, d_k)

    def forward(self, q, k, v):
        ## q, k, v: (m, d_k), single head
        m, d_k = q.shape
        i = torch.arange(m)[:, None]
        j = torch.arange(m)[None, :]
        r = (j - i).clamp(-self.c, self.c) + self.c       ## (m, m) offset label in [0, 2c]
        aK, aV = self.wK(r), self.wV(r)                   ## (m, m, d_k)
        ## content term + query-offset term
        s = (q @ k.T + torch.einsum("id,ijd->ij", q, aK)) / d_k ** 0.5
        a = s.softmax(-1)
        return a @ v + torch.einsum("ij,ijd->id", a, aV)

## Example
q, k, v = (torch.randn(5, 8) for _ in range(3))
print(RelativePositionAttention(8, c=2)(q, k, v).shape)  ## torch.Size([5, 8])
```
````

```{attention} Q&A
:class: dropdown
*Results?*
- WMT14: +1.3 BLEU EN-DE, +0.3 BLEU EN-FR over absolute PE. Adding absolute PE on top: no further gain.

*Why clip?*
- Precise offset matters little beyond some distance (paper hypothesis).
- Clip → finite table → any length works at inference.

*Cons?*
- $\mathcal{O}(m^2d_k)$ extra memory for $\mathbf{a}_{ij}$ → ❌fused attention kernels.
- The value term $\mathbf{a}^V$ was not needed for MT in the paper's ablation → later schemes drop it.
```

&nbsp;

### T5 Bias
- **What**: One learned scalar per (offset bucket, head), added to the logit. {cite:p}`raffel2020exploring`
- **Why**: Shaw-style vectors are costly. A scalar per offset is enough to steer attention.
- **How**:
    1. Map $j-i$ to one of 32 buckets.
        - Small offsets → own bucket. Larger offsets → log-spaced buckets up to 128. Beyond 128 → one shared bucket.
    2. Look up scalar $\beta_{\text{bucket},u}$ for head $u$.
    3. $s_{ij}\leftarrow s_{ij}+\beta_{\text{bucket}(j-i),u}$.
    - Shared across layers. Each head has its own scalars.

````{important} Code
:class: dropdown
```python
import math
import torch
import torch.nn as nn

class T5Bias(nn.Module):
    def __init__(self, h, n_buckets=32, max_dist=128, bidirectional=True):
        super().__init__()
        self.nb, self.md, self.bi = n_buckets, max_dist, bidirectional
        self.emb = nn.Embedding(n_buckets, h)  ## one scalar per (bucket, head)

    def bucket(self, rel):  ## rel = j - i
        nb, ret = self.nb, torch.zeros_like(rel)
        if self.bi:
            nb //= 2
            ret += (rel > 0).long() * nb       ## keys after the query get their own half
            rel = rel.abs()
        else:
            rel = (-rel).clamp(min=0)          ## causal: only j <= i matters
        exact = nb // 2
        ## offsets >= exact share log-spaced buckets; everything past max_dist hits the last one
        large = exact + (torch.log(rel.float().clamp(min=1) / exact)
                         / math.log(self.md / exact) * (nb - exact)).long()
        return ret + torch.where(rel < exact, rel, large.clamp(max=nb - 1))

    def forward(self, m):
        i = torch.arange(m)[:, None]
        j = torch.arange(m)[None, :]
        return self.emb(self.bucket(j - i)).permute(2, 0, 1)  ## (h, m, m), add to logits

## Example
b = T5Bias(h=4)
print(b.bucket(torch.tensor([-200, -128, -20, -3, 0, 3, 20, 128, 200])))
## tensor([15, 15, 10,  3,  0, 19, 26, 31, 31])
print(b(5).shape)  ## torch.Size([4, 5, 5])
```
````

```{attention} Q&A
:class: dropdown
*Is a layer blind beyond offset 128?*
- One layer: yes, all offsets >128 share one scalar.
- Stack: no. Later layers combine local info from earlier layers → longer effective range.

*Cons?*
- Bias is an $m\times m$ tensor per head → slower & more memory than sinusoidal. {cite:p}`press2022train`
- Learned scalars → extrapolation not guaranteed.
```

&nbsp;

### ALiBi
- **Name**: Attention with Linear Biases {cite:p}`press2022train`
- **What**: Fixed penalty $\propto$ distance, added to the logit. Different slope per head.
- **Why**: Train short, test long.
    - Sinusoidal fails a few dozen tokens past $L$. T5 bias extrapolates better but costs time & memory.
    - ALiBi: ❌params, ❌position embeddings, ~0 overhead.
- **How**:
    1. Do not add any position vector to the input.
    2. $s_{ij}\leftarrow s_{ij}-\mu_u(i-j)$ for $j\leq i$.
    3. Slopes $\mu_u$: geometric sequence, fixed before training.
        - Steep head → local focus. Shallow head → long range.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $u\in\{1,\dots,h\}$: Head index.
    - $\mu_u$: Slope of head $u$.

Forward:

$$
s_{ij}=\frac{\mathbf{q}_i^\top\mathbf{k}_j}{\sqrt{d_k}}-\mu_u\,(i-j),\qquad j\leq i
$$

$$
\mu_u=2^{-8u/h}
$$

- $h=8$ → $\mu=\frac{1}{2},\frac{1}{4},\dots,\frac{1}{256}$.
- The bias is not divided by $\sqrt{d_k}$.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class ALiBi(nn.Module):
    def __init__(self, h):
        super().__init__()
        u = torch.arange(1, h + 1)
        self.register_buffer("mu", 2.0 ** (-8.0 * u / h))  ## fixed slopes, not learned

    def forward(self, m):
        i = torch.arange(m)[:, None]
        j = torch.arange(m)[None, :]
        dist = (i - j).clamp(min=0)                        ## future keys are masked anyway
        return -self.mu[:, None, None] * dist              ## (h, m, m), add to logits

## Example
bias = ALiBi(h=8)(4)
print(bias.shape)  ## torch.Size([8, 4, 4])
print(bias[0, 3])  ## tensor([-1.5000, -1.0000, -0.5000, -0.0000]) -> slope 1/2
```
````

```{attention} Q&A
:class: dropdown
*Results?*
- 1.3B LM trained on $L=1024$, tested on 2048 → same PPL as sinusoidal trained on 2048, with 11% faster training & 11% less memory.

*Does flat PPL past $L$ mean it uses the long context?*
- Not necessarily. The penalty grows linearly → far tokens get ~0 weight → behaves like a soft local window.
- Flat PPL can mean distant tokens are ignored, not used.
- On reasoning tasks, ALiBi length generalization is poor. {cite:p}`kazemnejad2023impact`

*Why fixed slopes, not learned?*
- Trainable slopes did not give strong extrapolation & slowed training by 3%. The same geometric set works across domains & sizes.

*Cons?*
- Strong recency bias → hurts long-range retrieval.
- Bias added to logits → needs kernel support for the bias term.
```

&nbsp;

## Rotary
### RoPE
- **Name**: Rotary Position Embedding {cite:p}`su2024roformer`
- **What**: Rotate each 2D pair of $\mathbf{q}_i$ & $\mathbf{k}_j$ by angle $\propto$ own position.
- **Why**: Absolute encoding, relative effect.
    - Rotation by $i$ & rotation by $j$ → dot product sees only $j-i$.
    - ❌Params. ❌Bias tensor. Each vector is rotated once by its own position → works with KV cache.
- **How**:
    1. Split $\mathbf{q}_i$ into $d_k/2$ 2D pairs. Pair $n$ gets base frequency $\theta_n$.
    2. Rotate pair $n$ by angle $i\theta_n$. Do the same to $\mathbf{k}_j$ with angle $j\theta_n$.
    3. Compute $\mathbf{q}_i^\top\mathbf{k}_j$ as usual.
        - Rotations cancel to a net rotation of $(j-i)\theta_n$.
    - $\mathbf{v}$ is not rotated.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $n\in\{0,\dots,d_k/2-1\}$: Pair index.
    - $\theta_n$: Base frequency of pair $n$.
    - $R(\phi)\in\mathbb{R}^{2\times2}$: 2D rotation by angle $\phi$.
    - $\mathbf{q}^{(n)}\in\mathbb{R}^2$: Pair $n$ of $\mathbf{q}$, i.e., dims $(2n,2n+1)$.

Definition:

$$
\theta_n=b^{-2n/d_k},\qquad b=10000,\qquad R(\phi)=\begin{bmatrix}\cos\phi & -\sin\phi\\ \sin\phi & \cos\phi\end{bmatrix}
$$

- Same frequencies as [Sinusoidal](#sinusoidal), with $d_k$ in place of $d$. The paper indexes pairs from 1: $\theta=10000^{-2(n-1)/d}$.

Forward:

$$
\tilde{\mathbf{q}}_i^{(n)}=R(i\theta_n)\,\mathbf{q}_i^{(n)},\qquad \tilde{\mathbf{k}}_j^{(n)}=R(j\theta_n)\,\mathbf{k}_j^{(n)}
$$

$$
s_{ij}=\frac{1}{\sqrt{d_k}}\sum_{n=0}^{d_k/2-1}\mathbf{q}_i^{(n)\top}R\big((j-i)\theta_n\big)\,\mathbf{k}_j^{(n)}
$$

Equivalent complex form: pair $n$ = $z=q_{2n}+\mathrm{i}\,q_{2n+1}\in\mathbb{C}$, rotation = $z\,e^{\mathrm{i}\,i\theta_n}$.
```

```{tip} Derivation
:class: dropdown
*Why does the logit depend only on $j-i$?*

1. Rotations are orthogonal: $R(\phi)^\top=R(-\phi)$.
2. Rotations compose by adding angles: $R(\alpha)R(\beta)=R(\alpha+\beta)$.
3. Per pair: $\tilde{\mathbf{q}}_i^{(n)\top}\tilde{\mathbf{k}}_j^{(n)}=\mathbf{q}_i^{(n)\top}R(i\theta_n)^\top R(j\theta_n)\mathbf{k}_j^{(n)}=\mathbf{q}_i^{(n)\top}R\big((j-i)\theta_n\big)\mathbf{k}_j^{(n)}$.
4. Sum over pairs → $s_{ij}$ depends on content & $j-i$ only.

*Where does the design come from?*

1. Goal: find $f$ such that $f(\mathbf{q},i)^\top f(\mathbf{k},j)=g(\mathbf{q},\mathbf{k},j-i)$ with ❌params.
2. In 2D, multiply by $e^{\mathrm{i}\,i\theta}$: $\mathrm{Re}\big[(q e^{\mathrm{i}\,i\theta})^*(k e^{\mathrm{i}\,j\theta})\big]=\mathrm{Re}\big[q^*k\,e^{\mathrm{i}(j-i)\theta}\big]$ → depends on $j-i$.
3. Generalize to $d_k$ dims: apply independent 2D rotations to $d_k/2$ pairs, each with its own $\theta_n$.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class RoPE(nn.Module):
    def __init__(self, d_k, base=10000.0):
        super().__init__()
        n = torch.arange(d_k // 2)
        self.register_buffer("theta", base ** (-2 * n / d_k))  ## theta_n: fast -> slow

    def forward(self, x, pos):
        ## x: (..., m, d_k), pos: (m,) positions (float OK -> used by interpolation)
        ang = pos[:, None].float() * self.theta                ## (m, d_k/2): angle i * theta_n
        cos, sin = ang.cos(), ang.sin()
        x1, x2 = x[..., 0::2], x[..., 1::2]                    ## pair n = dims (2n, 2n+1)
        out = torch.empty_like(x)
        out[..., 0::2] = x1 * cos - x2 * sin                   ## 2D rotation per pair
        out[..., 1::2] = x1 * sin + x2 * cos
        return out

## Example: logit depends only on the offset j - i
rope = RoPE(8)
q, k = torch.randn(1, 8), torch.randn(1, 8)
p = lambda i: torch.tensor([i])
s1 = (rope(q, p(3)) * rope(k, p(5))).sum()    ## offset 2
s2 = (rope(q, p(10)) * rope(k, p(12))).sum()  ## offset 2
print(torch.allclose(s1, s2, atol=1e-5))      ## True
```
````

```{attention} Q&A
:class: dropdown
*Why rotate $\mathbf{q},\mathbf{k}$ but not $\mathbf{v}$?*
- Position must shape *who attends to whom* → the logit. $\mathbf{v}$ is the payload. Rotating it would mix position into content.

*Why does it work with the KV cache?*
- $\tilde{\mathbf{k}}_j$ depends only on $\mathbf{k}_j$ & $j$ → rotate once, cache, reuse. A new token never changes old keys. → [KV Cache](infer/cache.md#kv-cache)

*Interleaved vs half-split pairs?*
- Paper: pairs $(2n,2n+1)$. Llama-style code: pairs $(n,n+d_k/2)$ ("rotate_half").
- Same math up to a fixed permutation of $W_Q,W_K$ columns. Checkpoints are not interchangeable w/o permuting weights.

*Long-term decay?*
- The paper bounds $|s_{ij}|$ via Abel summation. The distance term of that bound trends down with $|j-i|$ (shown in a plot, not monotonic, not proved to decay). No guarantee that any single logit decays.

*Why does RoPE fail past $L$ if it is relative?*
- Slow pairs ($\theta_n$ small) do not complete one turn within $L$. Their angles past $L$ are unseen → out of distribution.
- → These dims act like absolute position → fix with [Context Extension](#context-extension).

*What does base $b$ control?*
- $b$⬆️ → $\theta_n$⬇️ for all $n>0$ ($\theta_0=1$ always) → slower rotation → longer wavelengths → longer usable context.
- Code Llama raises $b$ to $10^6$ for long context ("adjusted base frequency"). {cite:p}`peng2024yarn`

*Cons?*
- Extrapolation past $L$ ❌ w/o scaling tricks.
- Mixes with low-rank KV compression poorly → [decoupled RoPE in MLA](infer/cache.md#mla).
```

&nbsp;

## Context Extension
- **What**: Adapt a RoPE model trained on $L$ to a longer window $L'=sL$, $s>1$.
- **Why**: Pretraining at long $m$ costs $\mathcal{O}(m^2)$ attention. Plain extrapolation past $L$ → unseen angles → logits blow up.
- **How**: Remap positions or frequencies → fewer out-of-distribution rotations, local detail kept as far as possible → short fine-tune.

### Position Interpolation
- **What**: Scale positions by $L/L'$ → all positions fit in $[0,L)$. {cite:p}`chen2023extending`
- **Why**: Interpolation between seen angles is stable. Extrapolation to unseen angles is not.
    - Upper bound of interpolation error ≈ $600\times$ smaller than for extrapolation.
- **How**: Feed position $iL/L'$ to RoPE instead of $i$. Fine-tune briefly.

```{note} Math
:class: dropdown
Forward:

$$
\tilde{\mathbf{q}}_i^{(n)}=R\left(\frac{i}{s}\,\theta_n\right)\mathbf{q}_i^{(n)}\quad\Leftrightarrow\quad\theta_n\to\frac{\theta_n}{s}\ \text{for every pair}
$$
```

````{important} Code
:class: dropdown
```python
import torch

## Reuses RoPE from above
class PIRoPE(RoPE):
    def __init__(self, d_k, L, L_new, base=10000.0):
        super().__init__(d_k, base)
        self.ratio = L / L_new

    def forward(self, x, pos):
        return super().forward(x, pos * self.ratio)  ## fractional positions, all < L

## Example
pi = PIRoPE(8, L=2048, L_new=8192)
print(pi(torch.randn(10, 8), torch.arange(10)).shape)  ## torch.Size([10, 8])
```
````

```{attention} Q&A
:class: dropdown
*Results?*
- LLaMA 7B–65B extended with ≤1000 fine-tuning steps. Max window shown: 32768 (7B, 13B), 16384 (33B), 8192 (65B). Quality within the original window is mostly kept.

*Cons?*
- Shrinks every frequency by $s$, including fast pairs → nearby tokens become harder to tell apart.
- In practice, PI fine-tunes degrade past roughly $s=8$. {cite:p}`peng2024yarn`
```

&nbsp;

### NTK-Aware
- **What**: Raise base $b$ instead of scaling positions → fast pairs barely change, slow pairs are interpolated. {cite:p}`bloc972023ntkaware`
- **Why**: PI loses high-frequency (local) detail. Spread the scaling unevenly: high frequencies less, low frequencies more.
- **How**: Replace $b$ by $b'$ such that the slowest pair is scaled by exactly $1/s$ & the fastest pair is unchanged.
    - **Dynamic NTK**: at inference, set $s=\max(1,m/L)$ from the current length → no change for $m\leq L$.
        - ⚠️ KV cache: cached keys were rotated with the old $s$. Cache keys before RoPE, or freeze $s$ per request.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $b'$: New base.

Forward:

$$
b'=b\cdot s^{\frac{d_k}{d_k-2}},\qquad \theta'_n=b'^{-2n/d_k}=\theta_n\cdot s^{-\frac{2n}{d_k-2}}
$$

- $n=0$ → $\theta'_0=\theta_0$ (fastest pair unchanged).
- $n=d_k/2-1$ → $\theta'_n=\theta_n/s$ (slowest pair = PI).
```

```{tip} Derivation
:class: dropdown
1. Want: slowest pair $n=d_k/2-1$ scaled like PI: $b'^{-(d_k-2)/d_k}=b^{-(d_k-2)/d_k}/s$.
2. Raise both sides to $-d_k/(d_k-2)$: $b'=b\cdot s^{d_k/(d_k-2)}$.
3. Ratio for pair $n$: $\theta'_n/\theta_n=(b'/b)^{-2n/d_k}=s^{-2n/(d_k-2)}$ → smooth from $1$ to $1/s$.
```

````{important} Code
:class: dropdown
```python
## Reuses RoPE from above
class NTKRoPE(RoPE):
    def __init__(self, d_k, s, base=10000.0):
        super().__init__(d_k, base * s ** (d_k / (d_k - 2)))  ## base change only

## Example: per-pair frequency ratio vs plain RoPE
print(NTKRoPE(8, s=4).theta / RoPE(8).theta)
## tensor([1.0000, 0.6300, 0.3969, 0.2500]) -> fast pair kept, slowest = 1/s
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- Works w/o fine-tuning far better than PI.

*Cons?*
- Middle pairs are slightly extrapolated → fine-tuned NTK-aware is worse than fine-tuned PI.
- Effective extension < $s$ → $s$ must be set higher than the target ratio. {cite:p}`peng2024yarn`
```

&nbsp;

### YaRN
- **Name**: Yet another RoPE extensioN {cite:p}`peng2024yarn`
- **What**: Per-pair interpolation by wavelength (**NTK-by-parts**) + attention temperature.
- **Why**: Treat each pair by what it encodes.
    - Pairs with many turns within $L$ encode relative position → do not touch.
    - Pairs with <1 turn within $L$ encode absolute position → interpolate fully (no extrapolation).
    - Longer context → flatter softmax → sharpen logits with a temperature.
- **How**:
    1. For each pair, count turns within $L$: $r_n=L/\lambda_n$.
    2. $r_n>\beta$ → keep $\theta_n$. $r_n<\alpha$ → $\theta_n/s$. In between → linear ramp.
    3. Scale logits by $1/t$.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\alpha,\beta$: Ramp bounds (Llama: $\alpha=1,\beta=32$).
- Misc:
    - $\lambda_n=2\pi/\theta_n$: Wavelength of pair $n$.
    - $r_n$: #Turns of pair $n$ within $L$.
    - $\gamma$: Ramp function.
    - $t$: Attention temperature.

Forward:

$$
r_n=\frac{L}{\lambda_n},\qquad \gamma(r)=\begin{cases}0 & r<\alpha\\ 1 & r>\beta\\ \frac{r-\alpha}{\beta-\alpha} & \text{otherwise}\end{cases}
$$

$$
\theta'_n=\big(1-\gamma(r_n)\big)\frac{\theta_n}{s}+\gamma(r_n)\,\theta_n
$$

$$
s_{ij}=\frac{\tilde{\mathbf{q}}_i^\top\tilde{\mathbf{k}}_j}{t\sqrt{d_k}},\qquad \sqrt{1/t}=0.1\ln s+1
$$

- $\sqrt{1/t}$ is fit on LLaMA 7B–65B. Implementation: multiply both $\tilde{\mathbf{q}}$ & $\tilde{\mathbf{k}}$ by $\sqrt{1/t}$.
```

````{important} Code
:class: dropdown
```python
import math
import torch

## Reuses RoPE from above
class YaRN(RoPE):
    def __init__(self, d_k, L, s, base=10000.0, alpha=1.0, beta=32.0):
        super().__init__(d_k, base)
        r = L * self.theta / (2 * math.pi)                ## r = L / wavelength = #turns within L
        gamma = ((r - alpha) / (beta - alpha)).clamp(0, 1)
        ## gamma = 0 -> interpolate (theta / s), gamma = 1 -> keep theta
        self.theta = (1 - gamma) * self.theta / s + gamma * self.theta
        self.mscale = 0.1 * math.log(s) + 1               ## sqrt(1/t), applied to q AND k

    def forward(self, x, pos):
        return super().forward(x, pos) * self.mscale

## Example
yarn = YaRN(8, L=4096, s=4)
print(yarn.theta / RoPE(8).theta)  ## tensor([1.0000, 1.0000, 0.3835, 0.2500])
print(yarn(torch.randn(10, 8), torch.arange(10)).shape)  ## torch.Size([10, 8])
```
````

```{attention} Q&A
:class: dropdown
*Results?*
- Needs 10× fewer tokens & 2.5× fewer training steps than previous extension methods.
- Also extrapolates beyond the length of its own fine-tuning data.

*Why a temperature?*
- More keys in the softmax → weights spread out → entropy⬆️. $1/t>1$ sharpens logits back.
- The fitted $t$ transfers across LLaMA & Llama 2 sizes.

*PI vs NTK-aware vs YaRN?*
- PI: every pair $\div s$.
- NTK-aware: smooth per-pair scaling through a base change.
- YaRN: per-pair scaling by wavelength with a hard keep/interpolate split + temperature.
```

&nbsp;

## NoPE
- **What**: Causal decoder with no explicit position encoding. {cite:p}`haviv2022transformer`
- **Why**: The causal mask already leaks position.
    - Token $i$ attends to exactly $i+1$ tokens. With a distinct first token (e.g., BOS), uniform attention gives it weight $\frac{1}{i+1}$ → position is recoverable.
    - ❌Anchor (all tokens identical) → every position averages the same values → no position signal.
    - → Competitive LM PPL w/o any PE. Probes find absolute position inside the network.
- **How**: Remove the PE. Keep the causal mask.

```{attention} Q&A
:class: dropdown
*Does it work in an encoder?*
- ❌. Bidirectional attention is permutation-equivariant → no position at all. See [Permutation Equivariance](#permutation-equivariance).

*Length generalization?*
- On reasoning & math tasks, NoPE beats APE, T5 bias, ALiBi & RoPE in small decoder-only models. {cite:p}`kazemnejad2023impact`
- Learned NoPE attention resembles T5 relative bias patterns.

*Where is it used?*
- Hybrid stacks interleave the two. Llama 4: local (chunked) attention layers use RoPE, sparse global attention layers use NoPE.
    - → RoPE layers carry local order. NoPE layers carry long-range retrieval w/o out-of-range angles.

*Cons?*
- Position must be computed by the network → uses capacity.
- Results at frontier scale are less settled than for RoPE.
```

&nbsp;
