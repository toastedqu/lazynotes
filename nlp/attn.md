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
# Attention
The attention structures that ship in modern LLMs. Cache economics & serving of the same variants: [Cache](infer/cache.md).

Notations:
- $m$: #Tokens in sequence
- $d$: Model dim
- $h$: #Query heads
- $n_g$: #KV heads (i.e., #groups)
- $d_k$: Dim per query/key head
- $d_v$: Dim per value head
- $l$: #Layers
- $W$: Sliding window size (subscripted $W_Q,W_K,W_V,W_O$ are projection matrices)
- $X\in\mathbb{R}^{m\times d}$: Input token representations
- $Q,K,V$: Query/Key/Value matrices
- $\mathbf{q}_i,\mathbf{k}_j,\mathbf{v}_j$: Query/Key/Value vector of token $i$/$j$
- $S=QK^\top/\sqrt{d_k}$: Attention logits
- $A=\mathrm{softmax}(S+M)$: Attention weights
- $M$: Additive mask matrix

Override: $m$ = #tokens (global $m$ = #samples), matching [Cache](infer/cache.md).

&nbsp;

## Scaled Dot-Product Attention
- **What**: Softmax-weighted sum of value vectors, weighted by scaled query-key dot products. {cite:p}`vaswani2017attention`
- **Why**: Token mixing must be **content-addressed**, not position-addressed.
    - Recurrence/convolution mix by *position* → fixed pattern, distance-limited, ❌parallel over time.
    - Attention mixes by *similarity* → any token reaches any token in 1 hop, pattern chosen at runtime per input.
- **How**: Soft dictionary lookup.
    1. Each token emits $\mathbf{q}$ (what I want), $\mathbf{k}$ (what I offer), $\mathbf{v}$ (what I carry).
    2. $\mathbf{q}_i\cdot\mathbf{k}_j$ → relevance of $j$ to $i$.
    3. Scale by $1/\sqrt{d_k}$ → softmax → non-negative weights summing to 1.
    4. Weighted sum of $\mathbf{v}$ → new representation of $i$.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $O\in\mathbb{R}^{m\times d_v}$: Output.
- Params:
    - $W_Q,W_K\in\mathbb{R}^{d\times d_k}$: Query/Key projections.
    - $W_V\in\mathbb{R}^{d\times d_v}$: Value projection.
- Misc:
    - $\mathcal{A}_i$: Keys visible to query $i$ (set by $M$).

Forward:

$$
O=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V,\qquad Q=XW_Q,\quad K=XW_K,\quad V=XW_V
$$

Row-wise:

$$
\mathbf{o}_i=\sum_{j\in\mathcal{A}_i}a_{ij}\mathbf{v}_j,\qquad a_{ij}=\frac{\exp(s_{ij})}{\sum_{j'\in\mathcal{A}_i}\exp(s_{ij'})},\qquad s_{ij}=\frac{\mathbf{q}_i\cdot\mathbf{k}_j}{\sqrt{d_k}}
$$

Backward:

$$\begin{align*}
&\frac{\partial\mathcal{L}}{\partial V}=A^\top\frac{\partial\mathcal{L}}{\partial O} \\
&\frac{\partial\mathcal{L}}{\partial S}=A\odot\left(G-\left(\left(A\odot G\right)\mathbf{1}\right)\mathbf{1}^\top\right),\qquad G=\frac{\partial\mathcal{L}}{\partial O}V^\top \\
&\frac{\partial\mathcal{L}}{\partial Q}=\frac{1}{\sqrt{d_k}}\frac{\partial\mathcal{L}}{\partial S}K,\qquad \frac{\partial\mathcal{L}}{\partial K}=\frac{1}{\sqrt{d_k}}\left(\frac{\partial\mathcal{L}}{\partial S}\right)^\top Q
\end{align*}$$

Cost: $\mathcal{O}(m^2d_k)$ FLOPs, $\mathcal{O}(m^2)$ activation memory if $A$ is materialized.
```

```{tip} Derivation
:class: dropdown
*Why divide by $\sqrt{d_k}$?*

1. At init, treat entries of $\mathbf{q}_i,\mathbf{k}_j$ as iid, mean 0, variance 1.
2. $\mathbf{q}_i\cdot\mathbf{k}_j=\sum_{c=1}^{d_k}q_ck_c$ → $\mathbb{E}=0$, $\mathrm{Var}=d_k$ → std $=\sqrt{d_k}$.
3. → Logit spread grows with head dim → softmax saturates → $\mathbf{a}\to$ one-hot.
4. Softmax Jacobian $\mathrm{diag}(\mathbf{a})-\mathbf{a}\mathbf{a}^\top\to0$ as $\mathbf{a}\to$ one-hot → grads vanish.
5. $\div\sqrt{d_k}$ → $\mathrm{Var}(s_{ij})=1$, independent of $d_k$ → same softmax sharpness at any head width.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn.functional as F

def sdpa(q, k, v, mask=None):
    ## q: (..., m_q, d_k) | k: (..., m_kv, d_k) | v: (..., m_kv, d_v)
    s = q @ k.transpose(-2, -1) / q.shape[-1] ** 0.5
    if mask is not None:
        s = s + mask  ## 0 = allowed, -inf = blocked -> exp(-inf) = 0 after softmax
    return F.softmax(s, dim=-1) @ v

## Example
q, k, v = (torch.randn(2, 5, 8) for _ in range(3))
print(sdpa(q, k, v).shape)  ## torch.Size([2, 5, 8])
```
````

```{attention} Q&A
:class: dropdown
*Why a dot product instead of a learned scoring MLP (additive/Bahdanau attention)?*
- Dot product = one GEMM → tensor cores. Additive scoring needs an MLP evaluated on all $m^2$ pairs → ❌matmul, ❌speed.
- Comparable quality at equal params → additive attention is extinct in LLMs.

*Where does position enter?*
- Nowhere. SDPA is **permutation-equivariant** over keys → position must be injected, in modern LLMs by **RoPE** rotating $\mathbf{q},\mathbf{k}$ before the dot product.

*Why can $d_v\neq d_k$?*
- $QK^\top$ forces $\mathbf{q},\mathbf{k}$ to share a space; $\mathbf{v}$ only rides the weights → $d_v$ is free. Usually $d_v=d_k$; DeepSeek's [MLA](infer/cache.md#mla) is a live counterexample.

*Cons?*
- $\mathcal{O}(m^2)$ compute → the long-context bottleneck.
- Softmax sums to 1 → a head cannot attend to nothing → [Attention Sink](#attention-sink).
- Uniform cost per pair, regardless of how irrelevant → [Sparse Attention](#sparse-attention).
```

&nbsp;

## Masking
- **What**: Additive $-\infty$ pattern deciding which (query, key) pairs may interact.
- **Why**: All-to-all is wrong for two independent reasons.
    - **Causality**: next-token prediction is trivially solvable if a token sees its own future.
    - **Batching**: sequences are packed/padded together and must not leak into each other.
- **How**: $M_{ij}=0$ (allow) or $-\infty$ (block), added to $S$ before softmax.

### Causal
- **What**: Query $i$ sees keys $j\leq i$ only.
- **Why**: Turns ONE forward pass over $m$ tokens into $m$ next-token training signals.
    - ❌Mask → position $i$ sees token $i{+}1$, which IS its own label → 0 loss, 0 learning.
- **How**: Lower-triangular allow-pattern, $M_{ij}=0$ if $j\leq i$ else $-\infty$.

&nbsp;

### Prefix
- **What**: Bidirectional within a prefix of length $p$, causal after it.
- **Why**: The prompt is known in full before anything is generated → causal masking on it buys nothing at inference.
    - Trade: ❌next-token training signal on the prefix, ✅bidirectional encoding of it.
- **How**: $M_{ij}=0$ if $(i<p\ \text{and}\ j<p)$ or $j\leq i$.

&nbsp;

### Document
- **What**: Block-diagonal ∩ causal — query & key must belong to the same packed document.
- **Why**: Training packs many short docs into one $m$-length sequence to kill padding waste.
    - ❌Mask → doc $B$ attends into doc $A$ → cross-document contamination.
- **How**: $M_{ij}=0$ if $\mathrm{doc}(i)=\mathrm{doc}(j)$ and $j\leq i$. Position ids reset per document.

```{dropdown} Table: Masks
| Mask | Allowed $(i,j)$ | Where |
|:--|:--|:--|
| Full (bidirectional) | all | Encoders, embedding/rerank models |
| Causal | $j\leq i$ | Every decoder-only LLM |
| Prefix | $(i<p\wedge j<p)\vee j\leq i$ | Prefix-LM, prompt-conditioned VLMs |
| Document | $\mathrm{doc}(i)=\mathrm{doc}(j)\wedge j\leq i$ | Packed pretraining batches |
| Sliding window | $i-W<j\leq i$ | [Local layers](#sliding-window) |
```

````{important} Code
:class: dropdown
```python
import torch

def build_mask(m, kind="causal", p=None, doc_id=None, w=None):
    i = torch.arange(m)[:, None]
    j = torch.arange(m)[None, :]
    allow = j <= i                                  ## causal base
    if kind == "full":
        allow = torch.ones(m, m, dtype=torch.bool)
    elif kind == "prefix":
        allow = allow | ((i < p) & (j < p))         ## prefix block is bidirectional
    elif kind == "document":
        allow = allow & (doc_id[:, None] == doc_id[None, :])
    elif kind == "window":
        allow = allow & (j > i - w)                 ## keep the last w keys only
    ## -inf where blocked -> softmax sends those weights to exactly 0
    return torch.where(allow, 0.0, float("-inf"))

## Example: 2 docs packed into one length-6 sequence
doc = torch.tensor([0, 0, 0, 1, 1, 1])
print(build_mask(6, "document", doc_id=doc)[4])
## tensor([-inf, -inf, -inf, 0., 0., -inf]) -> token 4 cannot see doc 0
```
````

```{attention} Q&A
:class: dropdown
*Why isn't the mask an actual matrix in production?*
- $m^2$ entries at $m=128$K is ≈17B. Causal & window masks are implied by index arithmetic inside the [FlashAttention](#flashattention) kernel — fully-masked tiles are skipped, never computed.
- Corollary: passing an arbitrary dense `attn_mask` to `F.scaled_dot_product_attention` silently disables the flash backend, which accepts only `is_causal`.

*Is bidirectional attention dead?*
- For generation, yes. Alive wherever the whole input is known and nothing is generated: embedding, reranking, classification encoders.

*Does causal masking waste compute?*
- Naively yes — half the scores are computed then discarded. Real kernels skip the fully-masked tiles → ≈2× less work.

*What breaks if position ids are NOT reset per packed document?*
- **Absolute/learned PE**: the same document encodes differently depending on where it landed in the pack.
- **RoPE**: nothing, strictly — it depends only on $i-j$, so relative distances *inside* a document are unaffected. Resetting is consistency, not correctness.
- → The **mask** is what prevents contamination; position ids alone never would.
```

&nbsp;

## Multi-Head Attention
- **What**: $h$ parallel SDPA heads on $d_k=d/h$-wide slices, concatenated, mixed by $W_O$. {cite:p}`vaswani2017attention`
- **Why**: One softmax = ONE weighted average = one relation per token pair.
    - A token needs several at once: its syntactic head, its coreferent, the last mention of the same entity.
    - $h$ heads → $h$ relations for the **same total FLOPs**, since $d_k$ shrinks by $h$.
- **How**:
    1. Project $X$ → $h$ triples $(Q^{(i)},K^{(i)},V^{(i)})$ of width $d_k$.
    2. SDPA per head, in parallel.
    3. Concat → $\mathbb{R}^{m\times hd_v}$ → $W_O$ → back to $d$.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $W_Q^{(i)},W_K^{(i)}\in\mathbb{R}^{d\times d_k}$, $W_V^{(i)}\in\mathbb{R}^{d\times d_v}$: Per-head projections.
    - $W_O\in\mathbb{R}^{hd_v\times d}$: Output projection.

Forward:

$$
\mathrm{MHA}(X)=\mathrm{Concat}\left(\mathrm{head}_1,\cdots,\mathrm{head}_h\right)W_O,\qquad \mathrm{head}_i=\mathrm{Attn}\left(XW_Q^{(i)},XW_K^{(i)},XW_V^{(i)}\right)
$$

Equivalently, as a sum of per-head writes into the residual stream:

$$
\mathrm{MHA}(X)=\sum_{i=1}^{h}\mathrm{head}_i\,W_O^{(i)},\qquad W_O^{(i)}\in\mathbb{R}^{d_v\times d}
$$

Cost, with $d_k=d_v=d/h$:

$$
\underbrace{4md^2}_{\text{projections}}+\underbrace{2m^2d}_{\text{scores \& weighted sums}}\quad\text{MACs, independent of }h
$$
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        assert d % h == 0
        self.h, self.d_k = h, d // h
        ## One fused projection per role, split into heads afterwards
        self.W_q, self.W_k, self.W_v = (nn.Linear(d, d, bias=False) for _ in range(3))
        self.W_o = nn.Linear(d, d, bias=False)

    def forward(self, x, causal=True):
        B, T, _ = x.shape
        ## (B, T, d) -> (B, h, T, d_k): heads become a batch dim, so all h run as one matmul
        split = lambda t: t.view(B, T, self.h, self.d_k).transpose(1, 2)
        q, k, v = split(self.W_q(x)), split(self.W_k(x)), split(self.W_v(x))
        o = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        ## Concat heads back, then W_o mixes across them
        return self.W_o(o.transpose(1, 2).reshape(B, T, -1))

## Example
attn = MultiHeadAttention(d=64, h=8)
print(attn(torch.randn(2, 5, 64)).shape)  ## torch.Size([2, 5, 64])
```
````

```{attention} Q&A
:class: dropdown
*Why is total FLOPs independent of $h$?*
- Scores cost $m^2d_kh=m^2d$; projections cost $md^2$ either way. $h$ only decides how $d$ is partitioned, not how much work there is.

*Then why not $h=d$ (one dim per head)?*
- $d_k$ is the rank of each head's query-key bilinear form. $d_k$⬇️ → each head can compare on fewer directions → scores collapse. Practice settles at $d_k=64$–$128$, up to 256 (Gemma 3).

*What does $W_O$ actually do?*
- Concat alone leaves each head writing into a fixed disjoint slice. $W_O$ lets every head write into any direction of the residual stream, and lets heads cancel each other.

*Cons?*
- Redundancy: many heads are prunable post-hoc with little loss.
- Cache size $\propto h$ → the decode bandwidth wall → [Head Sharing](#head-sharing).
```

&nbsp;

## Head Sharing
- **What**: Shrink #KV heads below #query heads, or replace the KV heads with a shared low-rank latent.
- **Why**: [Decode](infer/cache.md#decode) is bandwidth-bound, and the bytes it moves scale with $n_g$ while the FLOPs scale with $h$.
    - MHA reloads $2hd_k$ cached elements per cached token per layer to do $\mathcal{O}(hd_k)$ FLOPs on them → arithmetic intensity ≈1 FLOP/byte in FP16 → the GPU idles.
    - Shrinking $h$ instead would cost representation subspaces; shrinking $n_g$ costs only *what heads look at*, not *what they ask*.
- **How**: One axis, three points on it.
    - **MQA**: $n_g=1$ — all queries share one K/V head.
    - **GQA**: $1<n_g<h$ — one K/V head per group of $h/n_g$ queries.
    - **MLA**: cache one low-rank latent $\mathbf{c}^{KV}$ per token, up-project to all $h$ heads on the fly.

```{dropdown} Table: Head-Sharing Variants
| Variant | KV heads | Cache elements / token / layer | Ships in |
|:--|:--|:--|:--|
| MHA | $h$ | $2hd_k$ | GPT-2/3, Llama-2 7B/13B |
| [MQA](infer/cache.md#mqa) | $1$ | $2d_k$ | PaLM, Falcon |
| [GQA](infer/cache.md#gqa) | $n_g$ | $2n_gd_k$ | Llama-3/4, Qwen, Mistral, Gemma, gpt-oss |
| [MLA](infer/cache.md#mla) | — (latent) | $d_c+d_h^R$ | DeepSeek-V2/V3, Kimi K2 |

- $d_h$: Content head dim. $d_c$: KV latent dim. $d_h^R$: Decoupled RoPE key dim.
- DeepSeek-V2 sets $d_c=4d_h$, $d_h^R=d_h/2$ → $4.5d_h$ ≡ GQA at $n_g=2.25$ groups of width $d_h$.
- Full mechanism, math & code for each: [Architectural Compression](infer/cache.md#architectural-compression).
```

```{attention} Q&A
:class: dropdown
*Which one do you pick?*
- **GQA, $n_g=8$** is the default ← 8 KV heads shard cleanly across an 8-GPU tensor-parallel node, so no KV head has to be replicated.
- **MLA** if you control pretraining: smaller cache than GQA at equal-or-better quality, at the cost of a more complex kernel.
- **MQA** if the cache must be as small as physically possible, accepting the quality & stability cost. It ran at frontier scale (PaLM-540B, Falcon) before GQA superseded it.

*Does head sharing reduce FLOPs?*
- Barely. Only $W_K,W_V$ shrink; attention still runs $h$ query heads. The win is **memory traffic**, which is what decode is bound by.

*Does it change prefill?*
- Almost not at all. Prefill is compute-bound and amortizes one weight load over $m$ tokens → head sharing is a decode-side optimization.

*Why does MHA survive at all?*
- Small models & encoders, where the cache is not the bottleneck, and MHA is the quality ceiling of this axis.
```

&nbsp;

## QK-Norm
- **What**: Normalize $\mathbf{q}$ & $\mathbf{k}$ per head before the dot product. {cite:p}`henry2020querykey`
- **Why**: Attention logits have no bound, and they grow during training.
    - $|s_{ij}|\leq\|\mathbf{q}_i\|\|\mathbf{k}_j\|/\sqrt{d_k}$, and nothing constrains $W_Q,W_K$ from growing.
    - → Logits⬆️ → softmax saturates → $A$ one-hot → Jacobian ≈0 → **dead head + loss spike**.
    - Worse with depth⬆️ & LR⬆️ → a standard failure mode of large pretraining runs.
- **How**: RMSNorm with a learned per-channel gain over $d_k$, applied to $\mathbf{q}$ and $\mathbf{k}$ right after projection.
    - → $\|\hat{\mathbf{q}}\|\leq\sqrt{d_k}\|\boldsymbol{\gamma}_q\|_\infty$, no matter how large $X$ or $W_Q$ grow → logits can no longer drift with the projections.
    - Applied **before** RoPE in practice; RoPE is a rotation, so it preserves the bound either way.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\boldsymbol{\gamma}_q,\boldsymbol{\gamma}_k\in\mathbb{R}^{d_k}$: Learned per-head gains.
    - $g$: Learned scalar scale (original form).

Modern form (RMSNorm, per head):

$$
\hat{\mathbf{q}}_i=\frac{\mathbf{q}_i}{\sqrt{\frac{1}{d_k}\sum_{c=1}^{d_k}q_{ic}^2}}\odot\boldsymbol{\gamma}_q,\qquad s_{ij}=\frac{\hat{\mathbf{q}}_i\cdot\hat{\mathbf{k}}_j}{\sqrt{d_k}}
$$

Original form ($\ell_2$-norm, learned scale replacing $1/\sqrt{d_k}$):

$$
s_{ij}=g\cdot\frac{\mathbf{q}_i}{\|\mathbf{q}_i\|_2}\cdot\frac{\mathbf{k}_j}{\|\mathbf{k}_j\|_2}\in[-|g|,|g|]
$$
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class QKNorm(nn.Module):
    def __init__(self, d_k, eps=1e-6):
        super().__init__()
        ## One gain vector per role, shared across heads (per-head dim d_k)
        self.g_q = nn.Parameter(torch.ones(d_k))
        self.g_k = nn.Parameter(torch.ones(d_k))
        self.eps = eps

    def _rms(self, x, g):
        ## Normalize along the HEAD dim only -> bounds ||q||, ||k|| -> bounds the logits
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * g

    def forward(self, q, k):
        ## q, k: (B, h, T, d_k), straight out of the projections and before RoPE
        return self._rms(q, self.g_q), self._rms(k, self.g_k)

## Example: a head whose projections have blown up
torch.manual_seed(0)
q = torch.randn(1, 1, 3, 4) * 50
k = torch.randn(1, 1, 3, 4) * 50
raw = (q @ k.transpose(-2, -1) / 2).abs().max()
qn, kn = QKNorm(4)(q, k)
print(raw.item() > 100, (qn @ kn.transpose(-2, -1) / 2).abs().max().item() < 10)  ## True True
```
````

```{attention} Q&A
:class: dropdown
*Doesn't $1/\sqrt{d_k}$ already handle this?*
- No. $1/\sqrt{d_k}$ fixes logit variance **at init**, under the iid unit-variance assumption. Training breaks that assumption by growing $W_Q,W_K$. QK-Norm re-imposes the bound at every step.

*Why per-head, and not just rely on pre-norm on the residual stream?*
- Pre-norm bounds the layer *input*, not $W_Q,W_K$. Blow-up is head-local: one saturating head is enough to spike the loss.

*Cost?*
- 2 RMSNorms over $d_k$ per token per head → negligible against $\mathcal{O}(m^2d)$ matmuls, and it does not touch the cache size.

*Any downside?*
- Discards $\|\mathbf{q}\|$ as a content signal: a head can no longer sharpen its own distribution by growing its query norm — only $\boldsymbol{\gamma}_q$ can, and that is content-independent.

*Who ships it?*
- Gemma 3, Qwen3, OLMo 2 → effectively standard in post-2024 pretraining recipes.
```

&nbsp;

## Attention Sink
- **What**: Large, content-independent attention mass parked on the first few tokens. {cite:p}`xiao2023efficient`
- **Why**: Softmax weights must sum to 1 → a head with nothing to retrieve still has to park its mass **somewhere**.
    - → Training converges on a no-op target: **token 0**, the one position every causal query can see (the next few inherit the role for almost all queries).
- **How**: Two forms.
    - **Emergent**: the first few tokens absorb the mass regardless of their content (StreamingLLM: keeping **4** is enough).
        - → Evicting them (naive sliding window) redistributes that mass onto real tokens → perplexity explodes. Fix: pin them → [StreamingLLM](infer/cache.md#streamingllm).
    - **Learned**: add a trainable per-head logit $\sigma$ to the softmax denominator → an explicit "attend to nothing" slot, costing 0 context. {cite:p}`openai2025gptoss`

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\sigma^{(h)}$: Learned sink logit of head $h$.

Learned sink:

$$
a_{ij}=\frac{\exp(s_{ij})}{\exp(\sigma^{(h)})+\sum_{j'\in\mathcal{A}_i}\exp(s_{ij'})}\qquad\Rightarrow\qquad\sum_{j}a_{ij}=1-\frac{\exp(\sigma^{(h)})}{\exp(\sigma^{(h)})+\sum_{j'}\exp(s_{ij'})}<1
$$

Equivalent to appending one virtual key with logit $\sigma^{(h)}$ and value $\mathbf{0}$:

$$
\mathbf{o}_i=\mathrm{softmax}\left(\left[\mathbf{s}_i,\ \sigma^{(h)}\right]\right)\cdot\left[V;\ \mathbf{0}^\top\right]
$$
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class SinkAttention(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.sink = nn.Parameter(torch.zeros(h))  ## one learned logit per head

    def forward(self, q, k, v, causal=True):
        s = q @ k.transpose(-2, -1) / q.shape[-1] ** 0.5
        if causal:
            T = q.shape[-2]
            causal_mask = torch.ones(T, T, dtype=torch.bool, device=s.device).triu(1)
            s = s.masked_fill(causal_mask, float("-inf"))
        ## Append the sink logit as an extra column -> it enters the denominator only
        sink = self.sink.view(1, -1, 1, 1).expand(*s.shape[:-1], 1)
        a = torch.softmax(torch.cat([s, sink], dim=-1), dim=-1)
        ## Drop the sink column: its value is 0, so it contributes nothing to the output
        return a[..., :-1] @ v

## Example: rows no longer sum to 1 -> the head can output ~nothing
q, k, v = (torch.randn(1, 2, 4, 8) for _ in range(3))
attn = SinkAttention(h=2)
print(attn(q, k, v).shape)  ## torch.Size([1, 2, 4, 8])
```
````

```{attention} Q&A
:class: dropdown
*Why does the sink land on the FIRST tokens specifically?*
- Causal masking → token 0 is the ONLY key visible to every query, so it is the one dump every head can share. Later positions are invisible to earlier queries, so they can only serve as partial sinks.

*How do you spot one?*
- A vertical stripe at the leftmost columns across nearly all heads & layers, persisting at any sequence length.

*Why is a learned sink better than pinning 4 real tokens?*
- ✅0 context slots consumed, ✅composes with sliding windows for free, ✅the head can express "no match" instead of faking one.
- Numerically it is just an extra term in the denominator — no extra key/value bytes in the cache.

*What does this say about softmax as a design choice?*
- Softmax hard-codes "you must attend to exactly 1.0 of something". That is an assumption, not a requirement, and models spend real capacity working around it.
```

&nbsp;

## FlashAttention
- **What**: Exact attention computed in SRAM-resident tiles; the $m\times m$ score matrix is never written to HBM. {cite:p}`dao2022flashattention`
- **Why**: Attention is **memory-bound**, not compute-bound.
    - Naive impl round-trips an $m\times m$ matrix through HBM several times (scores → mask → softmax → dropout → $AV$).
    - The matmuls are cheap; moving $\mathcal{O}(m^2)$ elements is not.
    - → Approximate attention (Linformer/Performer/Reformer) cut FLOPs but ignored memory traffic → ❌wall-clock win → extinct.
- **How**:
    1. Tile $Q$ into row blocks, $K,V$ into column blocks.
    2. For each $Q$ block, stream $K,V$ blocks through SRAM.
    3. **Online softmax**: keep a running row max $\mu$, running sum $\ell$, running output $\tilde{\mathbf{o}}$; rescale all three by one scalar whenever a new block raises the max.
    4. Backward: save only $O$ and per-row $(\mu,\ell)$ → **recompute** $S,A$ in SRAM instead of reading them from HBM.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\mathcal{M}$: SRAM size in elements. (Overrides $M$ = mask; the mask plays no role here.)
    - $T$: #Key blocks.
    - $\mathbf{s}^{(t)}$: Logits of query row against key block $t$.

Online softmax, for $t=1,\dots,T$ with $\mu^{(0)}=-\infty,\ \ell^{(0)}=0,\ \tilde{\mathbf{o}}^{(0)}=\mathbf{0}$:

$$\begin{align*}
&\mu^{(t)}=\max\left(\mu^{(t-1)},\ \max_j s^{(t)}_j\right) \\
&\ell^{(t)}=e^{\mu^{(t-1)}-\mu^{(t)}}\ell^{(t-1)}+\sum_je^{s^{(t)}_j-\mu^{(t)}} \\
&\tilde{\mathbf{o}}^{(t)}=e^{\mu^{(t-1)}-\mu^{(t)}}\tilde{\mathbf{o}}^{(t-1)}+\sum_je^{s^{(t)}_j-\mu^{(t)}}\mathbf{v}^{(t)}_j
\end{align*}$$

Output: $\mathbf{o}=\tilde{\mathbf{o}}^{(T)}/\ell^{(T)}$.

HBM accesses:

$$
\underbrace{\Theta\left(md_k+m^2\right)}_{\text{standard}}\qquad\text{vs}\qquad\underbrace{\Theta\left(\frac{m^2d_k^2}{\mathcal{M}}\right)}_{\text{FlashAttention}}
$$
- $d_k^2/\mathcal{M}\ll1$ at typical head dims ($d_k=64$–$128$, $\mathcal{M}\sim10^5$ elements) → many-fold fewer accesses.

Activation memory: $\mathcal{O}(m)$ ← only $(\mu,\ell)$ per row survive, not $A$.
```

```{tip} Derivation
:class: dropdown
*Why is the running rescale exact, not an approximation?*

1. Softmax is shift-invariant: $\mathrm{softmax}(\mathbf{s})=\mathrm{softmax}(\mathbf{s}-c)$ for any scalar $c$.
2. → Accumulate everything relative to the max **seen so far**, $\mu^{(t)}$, purely for fp stability.
3. When block $t$ raises the max, every stored term must be re-based:

    $$
    e^{s_j-\mu^{(t)}}=e^{s_j-\mu^{(t-1)}}\cdot e^{\mu^{(t-1)}-\mu^{(t)}}
    $$

4. → The whole re-basing is ONE scalar multiply applied to $\ell$ and $\tilde{\mathbf{o}}$. No term is dropped.
5. At $t=T$: $\ \tilde{\mathbf{o}}^{(T)}/\ell^{(T)}=\sum_j\frac{e^{s_j-\mu}}{\sum_{j'}e^{s_{j'}-\mu}}\mathbf{v}_j$ = exact softmax attention.
6. Backward: $A$ is regenerable from $Q,K$ and the saved per-row $(\mu,\ell)$ → recomputation costs FLOPs but removes an $\mathcal{O}(m^2)$ HBM read. On a memory-bound op that trade is strictly positive.
```

````{important} Code
:class: dropdown
```python
import torch

def flash_attention(q, k, v, block=2):
    ## q: (m_q, d_k) | k: (m_kv, d_k) | v: (m_kv, d_v). One head -> the algorithm, not the kernel.
    m_q, d_k = q.shape
    o = q.new_zeros(m_q, v.shape[-1])
    mu = q.new_full((m_q, 1), float("-inf"))  ## running row max
    l = q.new_zeros(m_q, 1)                   ## running row sum of exp
    for j in range(0, k.shape[0], block):
        ## Exactly ONE K/V tile is resident at a time -> the m x m score matrix never exists
        kb, vb = k[j:j + block], v[j:j + block]
        s = q @ kb.T / d_k ** 0.5
        mu_new = torch.maximum(mu, s.max(dim=-1, keepdim=True).values)
        rescale = (mu - mu_new).exp()   ## one scalar per row re-bases both accumulators
        p = (s - mu_new).exp()
        l = rescale * l + p.sum(dim=-1, keepdim=True)
        o = rescale * o + p @ vb
        mu = mu_new
    return o / l                        ## normalize once at the end (the FA-2 ordering)

## Example: exact, not approximate
q, k, v = (torch.randn(4, 8, dtype=torch.float64) for _ in range(3))
ref = torch.softmax(q @ k.T / 8 ** 0.5, dim=-1) @ v
print(torch.allclose(flash_attention(q, k, v), ref))  ## True
```
````

```{attention} Q&A
:class: dropdown
*Is it an approximation?*
- ❌ Identical math. Only the accumulation order differs → fp rounding differs, results do not.

*Does it reduce FLOPs?*
- ❌ It slightly **increases** them (backward recomputation). It reduces HBM traffic, which is what the wall clock measures.

*What does it actually unlock?*
- $\mathcal{O}(m)$ instead of $\mathcal{O}(m^2)$ activation memory → long-context training becomes possible at all, independent of the speedup.

*When does it NOT fire?*
- Arbitrary dense additive masks, exotic head dims, fp32, or anything needing the materialized $A$ (attention-map analysis, some interpretability probes). PyTorch then silently falls back to a slower backend.

*Why did approximate attention lose to an exact method?*
- It optimized the wrong quantity. FLOPs were never the binding constraint — IO was.
```

### FlashAttention-2
- **What**: Same algorithm, re-partitioned for the GPU. {cite:p}`dao2023flashattention2`
- **Why**: FA-1 hit only 25-40% of peak FLOPs/s — now bound by non-matmul work and idle SMs, not by IO.
- **How**:
    1. ⬇️Non-matmul FLOPs: rescale $\tilde{\mathbf{o}}$ **once at the end** instead of every block.
    2. Parallelize over the **sequence** dim, not just batch×head → occupancy holds at small batch / long context.
    3. $Q$ outer / $K,V$ inner loop with $Q$ split across warps → ❌inter-warp shared-memory traffic.
- ≈2× over FA-1, 50-73% of theoretical max FLOPs/s on A100; 225 TFLOPs/s end-to-end GPT training (72% MFU).

### FlashAttention-3
- **What**: FA-2 restructured around Hopper's asynchronous units + FP8. {cite:p}`shah2024flashattention3`
- **Why**: FA-2 reaches only 35% utilization on H100 ← it ignores TMA and warp-group async matmuls.
- **How**:
    1. **Warp specialization**: producer warps issue TMA loads while consumer warps run matmuls → overlap compute & data movement.
    2. **Interleave** block-wise GEMM with softmax → hide the non-tensor-core exponentials under the matmuls.
    3. **FP8** via block quantization + incoherent processing (random orthogonal rotation) so outliers don't destroy accuracy.
- 1.5-2.0× over FA-2 on H100; FP16 up to 740 TFLOPs/s (75% util), FP8 close to 1.2 PFLOPs/s, 2.6× lower numerical error than baseline FP8 attention.

&nbsp;

## Sparse Attention
- **What**: Attend over a **subset** of keys per query.
- **Why**: FlashAttention makes attention IO-optimal but still $\mathcal{O}(m^2)$.
    - Attention is $\mathcal{O}(m^2)$ while everything else is $\mathcal{O}(m)$ → it crosses over and dominates prefill at long context.
    - Attention maps are empirically near-sparse → most of that quadratic work barely moves the output.

### Sliding Window
- **What**: Query $i$ attends to the last $W$ keys only; global reach is recovered by depth ($l$ layers → span $l(W-1)+1\approx l\cdot W$).
- **Why**: Static (❌selection cost) + contiguous (✅coalesced reads) + cache-bounding ($W$ instead of $m$) — a combination no dynamic scheme offers.
- **How**: Interleave with full-attention layers so exact long-range recall survives. Ratios, span math & cache accounting: [Sliding Window Attention](infer/cache.md#sliding-window-attention) & [Hybrid Attention](infer/cache.md#hybrid-attention).

### NSA
- **Name**: Native Sparse Attention {cite:p}`yuan2025native`
- **What**: 3 parallel branches per query — compressed blocks, selected blocks, local window — mixed by a learned gate, trained end-to-end.
- **Why**:
    - *Why do we need it?*
        - Post-hoc sparsity ([H2O](infer/cache.md#h2o), [SnapKV](infer/cache.md#snapkv)) prunes a model that was **trained dense** → it never learned to live without those tokens.
        - Most patterns accelerate ONE phase → the other stays dense → little end-to-end win.
        - Per-head token selection breaks GQA: a group must load the **union** of its heads' picks → traffic barely drops.
    - *Why does it work?*
        - **Blockwise**, not per-token, selection → contiguous reads, balanced arithmetic intensity.
        - Selection **shared within a GQA group** (importance scores summed over the group's heads) → traffic actually drops.
        - End-to-end trainable despite the discrete top-$n$ routing → the model learns what to drop instead of being told.
- **How**:
    1. **Compress**: each block of tokens → 1 K/V pair via a learned MLP → coarse global view.
    2. **Select**: rank blocks by the compressed branch's softmax scores → keep top-$n$ → attend at full granularity.
    3. **Window**: last $W$ tokens verbatim → the local detail the other two branches smear.
    4. **Gate**: sigmoid gates from the layer input blend the 3 branch outputs.
- Matches or beats full attention on a 27B pretrained backbone, with speedup growing in $m$.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $c\in\{\mathrm{cmp},\mathrm{slc},\mathrm{win}\}$: Branch.
    - $\tilde{K}^c_t,\tilde{V}^c_t$: Keys/values kept by branch $c$ for query $t$.
    - $g^c_t\in[0,1]$: Gate of branch $c$.
    - $\mathbf{h}_t$: Layer input at position $t$.

$$
\mathbf{o}_t=\sum_{c}g^c_t\cdot\mathrm{Attn}\left(\mathbf{q}_t,\tilde{K}^c_t,\tilde{V}^c_t\right),\qquad g^c_t=\sigma\left(\mathrm{MLP}\left(\mathbf{h}_t\right)\right)
$$
```

### DSA
- **Name**: DeepSeek Sparse Attention {cite:p}`deepseekai2025v32`
- **What**: A cheap **indexer** scores every past token; attention then runs over the top-$k$ only.
- **Why**: NSA selects at block granularity and is trained natively from scratch.
    - DSA selects at **token** granularity and is grafted onto an existing [MLA](infer/cache.md#mla) model by continued pretraining.
- **How**:
    1. **Lightning indexer**: a few small heads scoring $\mathbf{q}^I$ against a single shared index key per token, in FP8 → cost ≪ main attention.
    2. **Top-$k$** ($k=2048$) indices per query.
    3. MLA attends over those $k$ tokens only → $\mathcal{O}(mk)$ instead of $\mathcal{O}(m^2)$.
- V3.2-Exp vs V3.1-Terminus under matched training: benchmark parity (MMLU-Pro 85.0 vs 85.0) at much lower long-context cost.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\mathbf{q}^I_{tj}$: Indexer query of head $j$ at position $t$.
    - $\mathbf{k}^I_s$: Indexer key of token $s$ (ONE head, shared by all indexer heads).
    - $w^I_{tj}$: Per-token, per-indexer-head weight.
- Hyperparams:
    - $H^I$: #Indexer heads.
    - $k$: #Selected tokens.

Index score & selection:

$$
I_{ts}=\sum_{j=1}^{H^I}w^I_{tj}\cdot\mathrm{ReLU}\left(\mathbf{q}^I_{tj}\cdot\mathbf{k}^I_s\right),\qquad \mathcal{A}_t=\mathrm{Top}\text{-}k_{\,s\leq t}\left(I_{ts}\right)
$$

Main attention is then plain MLA restricted to $\mathcal{A}_t$.
```

```{attention} Q&A
:class: dropdown
*Why ReLU instead of softmax in the indexer?*
- Only the **ranking** matters, not a distribution → normalization is wasted work. ReLU is chosen for throughput and zeroes out irrelevant tokens outright.

*Doesn't the indexer itself cost $\mathcal{O}(m^2)$?*
- Yes. Tiny head dim, few heads, one shared key head and FP8 make its constant small enough that the $\mathcal{O}(mk)$ main attention dominates. Sparse attention buys asymptotics in the **expensive** term only.

*Static vs dynamic sparsity — what's the real difference?*
- **Static** (sliding window): free to compute, bounds the **cache** at $W$, provably loses distant tokens.
- **Dynamic** (NSA/DSA): must retain the **full** cache (any token might be selected) → saves **compute**, not memory.
- → They are complementary, not competing; NSA runs a window branch for exactly this reason.

*Why not just prune at inference (H2O/SnapKV)?*
- The top 20% of tokens by attention score account for only ~70% of total attention mass → post-hoc pruning removes tokens the dense model was relying on. Native training lets the model route around the loss instead.
```

&nbsp;

## Linear Attention
- **What**: Replace $\exp(\mathbf{q}\cdot\mathbf{k})$ with a factorizable kernel $\phi(\mathbf{q})^\top\phi(\mathbf{k})$ → attention becomes an RNN with a matrix state. {cite:p}`katharopoulos2020transformers`
- **Why**: Softmax is precisely what forces $\mathcal{O}(m^2)$ — it couples every pair before normalizing.
    - Drop it → associativity applies: $\left(\phi(Q)\phi(K)^\top\right)V=\phi(Q)\left(\phi(K)^\top V\right)$.
    - → $\mathcal{O}(md_kd_v)$ training, $\mathcal{O}(1)$ per decode step, and the KV cache stops growing entirely.
- **How**:
    1. Feature map $\phi(\cdot)$, e.g. $\mathrm{elu}(x)+1$ → non-negative scores.
    2. Carry a state $S_t\in\mathbb{R}^{d_v\times d_k}$ (all past associations) and a normalizer $\mathbf{z}_t$.
    3. Read the state with $\phi(\mathbf{q}_t)$ instead of scanning history.
    4. Train with a **chunkwise** form: quadratic inside a chunk, recurrent across chunks → keeps the tensor cores fed.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\phi$: Feature map.
    - $S_t\in\mathbb{R}^{d_v\times d_k}$: Matrix state.
    - $\mathbf{z}_t\in\mathbb{R}^{d_k}$: Normalizer state.

Causal linear attention:

$$
\mathbf{o}_t=\frac{\sum_{j\leq t}\left(\phi(\mathbf{k}_j)\cdot\phi(\mathbf{q}_t)\right)\mathbf{v}_j}{\sum_{j\leq t}\phi(\mathbf{k}_j)\cdot\phi(\mathbf{q}_t)}=\frac{S_t\,\phi(\mathbf{q}_t)}{\mathbf{z}_t\cdot\phi(\mathbf{q}_t)}
$$

Recurrence:

$$
S_t=S_{t-1}+\mathbf{v}_t\phi(\mathbf{k}_t)^\top,\qquad \mathbf{z}_t=\mathbf{z}_{t-1}+\phi(\mathbf{k}_t)
$$

State size $d_kd_v$ per head per layer, **independent of $m$**.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn.functional as F

def linear_attention(q, k, v):
    ## q, k: (m, d_k) | v: (m, d_v)
    phi = lambda x: F.elu(x) + 1        ## non-negative features -> weights stay non-negative
    q, k = phi(q), phi(k)
    S = q.new_zeros(v.shape[-1], q.shape[-1])  ## (d_v, d_k) state
    z = q.new_zeros(q.shape[-1])               ## (d_k,) normalizer
    out = []
    for t in range(q.shape[0]):
        ## Write then read -> causal. Each step costs O(d_k * d_v), NOT O(t): no history scan.
        S = S + torch.outer(v[t], k[t])
        z = z + k[t]
        out.append(S @ q[t] / (z @ q[t]))
    return torch.stack(out)

## Example: identical to the quadratic form, in O(m) with O(d_k*d_v) memory
q, k, v = (torch.randn(6, 4, dtype=torch.float64) for _ in range(3))
phi = lambda x: F.elu(x) + 1
A = (phi(q) @ phi(k).T).tril()
ref = (A / A.sum(-1, keepdim=True)) @ v
print(torch.allclose(linear_attention(q, k, v), ref))  ## True
```
````

```{attention} Q&A
:class: dropdown
*Where does the quality go?*
- $S_t$ is a **fixed-capacity** $d_k\times d_v$ compression of unbounded history → exact recall of an arbitrary distant token is information-theoretically impossible. Softmax attention keeps every token verbatim.

*So why did it come back after failing in 2020?*
- Not for the FLOPs — for the **cache**. Decode is bandwidth-bound and a constant-size state is the strongest possible answer to that.
- And the state update got smarter: gating + delta rule fixed the "never forgets, never corrects" failure.

*Is it really $\mathcal{O}(1)$ per step?*
- Per step yes, but the constant is $\approx2d_kd_v$ MACs (one state write + one state read).
- Softmax decode over $t$ cached tokens costs $t(d_k+d_v)$ → they cross at $t=\frac{2d_kd_v}{d_k+d_v}\approx128$ for $d_k=d_v=128$.
- → The win starts around a hundred tokens and grows linearly from there; at short context linear attention is simply slower.
```

### Gated DeltaNet
- **Name**: Gated Delta Networks {cite:p}`yang2024gated`
- **What**: Linear attention whose state **erases before writing**, under a decay gate.
- **Why**: A purely additive state never forgets and never corrects.
    - → Fixed capacity fills with stale associations → the retrieval collapse that killed 2020-era linear attention.
    - **Gate** $\alpha_t$: wipe everything fast (needed at document boundaries / topic switches).
    - **Delta rule** $\beta_t$: overwrite ONE key's slot, leave the rest untouched (needed for "X is now 7, not 3").
    - The two are complementary → combine.
- **How**: $\alpha_t$ decays the whole state; $(I-\beta_t\mathbf{k}_t\mathbf{k}_t^\top)$ projects out the incoming key's old content (keys are $\ell_2$-normalized, so $\mathbf{k}\mathbf{k}^\top$ IS a projector); then write the new value.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\alpha_t\in(0,1)$: Decay gate.
    - $\beta_t\in(0,1)$: Write strength.
    - $\|\mathbf{q}_t\|_2=\|\mathbf{k}_t\|_2=1$: Queries & keys are $\ell_2$-normalized.

State update & read:

$$
S_t=S_{t-1}\left(\alpha_t\left(I-\beta_t\mathbf{k}_t\mathbf{k}_t^\top\right)\right)+\beta_t\mathbf{v}_t\mathbf{k}_t^\top,\qquad \mathbf{o}_t=S_t\mathbf{q}_t
$$

Special cases:
- $\alpha_t=1$ → DeltaNet (erase-then-write, no decay).
- Drop the $\mathbf{k}\mathbf{k}^\top$ term (absorbing $\beta_t$ into $\mathbf{v}_t$) → Mamba2's gated additive state.
```

### Hybrid Stack
- **What**: A mostly-linear stack with a few full-attention layers interleaved.
- **Why**: The two failure modes are exactly complementary.
    - Linear: state size constant in $m$, ❌exact long-range recall.
    - Full: exact recall, $\mathcal{O}(m)$ cache.
    - → A handful of full layers restores retrieval while the cache shrinks by the layer ratio.
- **How**: Interleave at a fixed ratio. In production:
    - MiniMax-01: **7:1** lightning attention : softmax attention. {cite:p}`minimax2025minimax01`
    - Qwen3-Next: **3:1** Gated DeltaNet : gated attention (12 blocks × 4 layers). {cite:p}`qwenteam2025qwen3next`
        - *Gated attention* = SDPA whose output is scaled by an input-dependent sigmoid gate before $W_O$.
    - Kimi Linear: **3:1** KDA : MLA → KV cache ⬇️ up to 75%, decode up to 6× faster at 1M context. {cite:p}`team2025kimi`

```{attention} Q&A
:class: dropdown
*Is attention being replaced?*
- No. Every shipped "linear" model is a hybrid — the full-attention layers are load-bearing, not legacy.

*Where does the cost actually land?*
- **Prefill**: the few full layers are $\mathcal{O}(m^2)$ and dominate at long $m$; the linear layers are noise.
- **Decode**: only the full layers hold a cache → the layer ratio **is** the cache reduction factor.

*Why not just enlarge the state instead of adding full layers?*
- The state is read & written every step → compute and bandwidth scale with $d_kd_v$. Sized to match exact recall, you have rebuilt a KV cache with extra steps.

*Why does 1 full layer per 3-7 linear layers suffice?*
- The residual stream is shared: a full layer can retrieve a distant token once and write it into the stream, after which the linear layers process it locally. Exact recall is needed at a few points, not everywhere.
```

&nbsp;

## Cross Attention
- **What**: $Q$ from the target stream, $K,V$ from a separate source (encoder output / another modality).
- **Why**: Fuse a source that is fully known upfront, has its own length & modality, and must not be re-encoded at every decode step.
- **How**: Same SDPA, ❌causal mask (the source is fully visible). $K,V$ are computed ONCE over the source and reused for the whole generation.

```{attention} Q&A
:class: dropdown
*Why did decoder-only LLMs drop it?*
- Concatenating source + target into one sequence under causal self-attention does the same job with ONE stack, one set of weights, one scaling law.

*So where is it still the right answer?*
- **Encoder-decoder ASR/translation** (e.g. Whisper): source is a fixed-length audio encoding, never generated.
- **VLMs that inject vision via cross-attention layers**: image K/V stay outside the causal cache instead of inflating it for the whole generation.
- **Frozen-backbone adaptation**: new cross-attention layers can be added without touching the pretrained self-attention path.

*Prefixing source tokens vs cross-attention — the tradeoff?*
- Prefixing: source tokens join the causal sequence → prefill becomes $\mathcal{O}((m_\text{src}+m)^2)$ and EVERY self-attention layer caches them for the whole generation.
- Cross-attention: source K/V are built once and live only in the cross-attention layers → self-attention prefill stays $\mathcal{O}(m^2)$, and the causal cache never carries the source.
- ❌Free: every decode step still attends over all $m_\text{src}$ source positions, so total cross-attention work is $\Theta(m_\text{src}m)$. The saving is in the quadratic term and the cache, not in the source scan.
```
