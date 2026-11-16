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
- $r$: Head idx
- $d_k$: Dim per query/key head
- $d_v$: Dim per value head
- $W$: Sliding window size
- $X\in\mathbb{R}^{m\times d}$: Input token representations
- $Q$: Query matrix
- $K$: Key matrix
- $V$: Value matrix
- $\mathbf{q}_i$: Query vector of token $i$
- $\mathbf{k}_j$: Key vector of token $j$
- $\mathbf{v}_j$: Value vector of token $j$
- $S=QK^\top/\sqrt{d_k}$: Attention logits
- $M$: Additive mask matrix
- $A=\mathrm{softmax}(S+M)$: Attention weights
- $\mathcal{A}_i$: Keys visible to query $i$

Override: $m$ = #tokens (global $m$ = #samples), matching [Cache](infer/cache.md).
Override: $i$ = query position, $j$ = key position (global: sample idx, feature idx).
Subscripted $W_Q,W_K,W_V,W_O$ are projection matrices, not window sizes.

&nbsp;

## Scaled Dot-Product Attention
- **What**: Value average weighted by softmax of scaled query-key dot products. {cite:p}`vaswani2017attention`
- **Why**: Content-addressed token mixing.
    - Recurrence mixes by position → sequential over time, distant info decays through many steps.
    - Convolution mixes by position → fixed local kernel, reach grows only with depth.
    - Attention mixes by similarity → any token reaches any token in 1 hop; pattern set at runtime per input.
- **How**: Soft dictionary lookup.
    1. Each token emits $\mathbf{q}$ (what it wants), $\mathbf{k}$ (what it offers), $\mathbf{v}$ (what it carries).
    2. $\mathbf{q}_i\cdot\mathbf{k}_j$ → relevance of $j$ to $i$.
    3. Scale by $1/\sqrt{d_k}$ → softmax → non-negative weights, sum 1.
    4. Weighted sum of $\mathbf{v}$ → new representation of $i$.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $O\in\mathbb{R}^{m\times d_v}$: Output.
- Params:
    - $W_Q\in\mathbb{R}^{d\times d_k}$: Query projection.
    - $W_K\in\mathbb{R}^{d\times d_k}$: Key projection.
    - $W_V\in\mathbb{R}^{d\times d_v}$: Value projection.

Forward:

$$
O=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V,\qquad Q=XW_Q,\quad K=XW_K,\quad V=XW_V
$$

$$
\mathbf{o}_i=\sum_{j\in\mathcal{A}_i}a_{ij}\mathbf{v}_j,\qquad a_{ij}=\frac{\exp(s_{ij})}{\sum_{j'\in\mathcal{A}_i}\exp(s_{ij'})},\qquad s_{ij}=\frac{\mathbf{q}_i\cdot\mathbf{k}_j}{\sqrt{d_k}}
$$

- $\mathbf{o}_i$: Row $i$ of $O$.
- $a_{ij}$: Entry $(i,j)$ of $A$.
- $s_{ij}$: Entry $(i,j)$ of $S$.

Backward:

$$\begin{align*}
&\frac{\partial\mathcal{L}}{\partial V}=A^\top\frac{\partial\mathcal{L}}{\partial O} \\
&\frac{\partial\mathcal{L}}{\partial S}=A\odot\left(G-\left(\left(A\odot G\right)\mathbf{1}\right)\mathbf{1}^\top\right),\qquad G=\frac{\partial\mathcal{L}}{\partial O}V^\top \\
&\frac{\partial\mathcal{L}}{\partial Q}=\frac{1}{\sqrt{d_k}}\frac{\partial\mathcal{L}}{\partial S}K,\qquad \frac{\partial\mathcal{L}}{\partial K}=\frac{1}{\sqrt{d_k}}\left(\frac{\partial\mathcal{L}}{\partial S}\right)^\top Q
\end{align*}$$

- $G$: $\partial\mathcal{L}/\partial A$.
- $\mathbf{1}\in\mathbb{R}^{m}$: All-ones vector.
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
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def forward(self, q, k, v, mask=None):
        ## q: (..., m_q, d_k) | k: (..., m_kv, d_k) | v: (..., m_kv, d_v)
        s = q @ k.transpose(-2, -1) / q.shape[-1] ** 0.5
        if mask is not None:
            s = s + mask.to(s)  ## 0 = allowed, -inf = blocked -> exp(-inf) = 0 after softmax
        return F.softmax(s, dim=-1) @ v

## Example
q, k, v = (torch.randn(2, 5, 8) for _ in range(3))
print(ScaledDotProductAttention()(q, k, v).shape)  ## torch.Size([2, 5, 8])
```
````

```{attention} Q&A
:class: dropdown
*Why a dot product instead of a learned scoring MLP (additive/Bahdanau attention)?*
- Dot product = 1 GEMM → tensor cores.
- Additive scoring runs an MLP on all $m^2$ pairs → slower, more memory.
- Additive beats unscaled dot product at large $d_k$. The $1/\sqrt{d_k}$ scale targets exactly that gap.

*Where does position enter?*
- Nowhere in SDPA itself.
    - Permute keys & values jointly → output unchanged.
    - Permute queries → outputs permute the same way.
- → Inject position externally. Modern LLMs use **RoPE**: rotate $\mathbf{q},\mathbf{k}$ before the dot product.
- A causal mask breaks the symmetry → decoder LMs w/o positional encoding still learn some position info.

*Why can $d_v\neq d_k$?*
- $QK^\top$ forces $\mathbf{q},\mathbf{k}$ into one shared space. $\mathbf{v}$ only gets weighted → $d_v$ is free.
- Usually $d_v=d_k$. DeepSeek's [MLA](infer/cache.md#mla) is a live counterexample.

*Cost?*
- $\mathcal{O}(m^2d_k)$ FLOPs per head.
- $\mathcal{O}(m^2)$ activation memory if $A$ is materialized → [FlashAttention](#flashattention) removes this.

*Cons?*
- Quadratic in $m$ → the long-context bottleneck.
- Softmax sums to 1 → a head cannot attend to nothing → [Attention Sink](#attention-sink).
- Same cost per pair, however irrelevant → [Sparse Attention](#sparse-attention).
```

&nbsp;

## Masking
- **What**: Additive $-\infty$ pattern that sets which (query, key) pairs interact.
- **Why**: All-to-all attention is wrong for 2 independent reasons.
    - **Causality**: next-token prediction is trivial if a token sees its own future.
    - **Batching**: packed/padded sequences must not leak into each other.
- **How**: $M_{ij}=0$ (allow) or $-\infty$ (block), added to $S$ before softmax → blocked weights become exactly 0.

```{dropdown} Table: Masks
| Mask | Allowed $(i,j)$ | Where |
|:--|:--|:--|
| Full (bidirectional) | all | Encoders, embedding/rerank models |
| [Causal](#causal) | $j\leq i$ | Autoregressive decoder-only LMs |
| [Prefix](#prefix) | $(i<p\wedge j<p)\vee j\leq i$ | Prefix-LM, prompt-conditioned VLMs |
| [Document](#document) | $\mathrm{doc}(i)=\mathrm{doc}(j)\wedge j\leq i$ | Packed pretraining batches |
| [Sliding window](#sliding-window) | $i-W<j\leq i$ | Local layers |

- $p$: Prefix length.
- $\mathrm{doc}(i)$: Document id of token $i$.
```

````{important} Code
:class: dropdown
```python
import torch

class AttentionMask:
    def __init__(self, kind="causal", p=None, w=None):
        self.kind, self.p, self.w = kind, p, w

    def __call__(self, m, doc_id=None):
        i = torch.arange(m)[:, None]
        j = torch.arange(m)[None, :]
        allow = j <= i                                         ## causal base
        if self.kind == "full":
            allow = torch.ones(m, m, dtype=torch.bool)
        elif self.kind == "prefix":
            allow = allow | ((i < self.p) & (j < self.p))      ## prefix block is bidirectional
        elif self.kind == "document":
            allow = allow & (doc_id[:, None] == doc_id[None, :])
        elif self.kind == "window":
            allow = allow & (j > i - self.w)                   ## keep the last w keys only
        ## -inf where blocked -> softmax sends those weights to exactly 0
        return torch.where(allow, 0.0, float("-inf"))

## Example: 2 docs packed into one length-6 sequence
doc = torch.tensor([0, 0, 0, 1, 1, 1])
print(AttentionMask("document")(6, doc_id=doc)[4])
## tensor([-inf, -inf, -inf, 0., 0., -inf]) -> token 4 cannot see doc 0
```
````

```{attention} Q&A
:class: dropdown
*Why is the mask not a real matrix in production?*
- $m^2$ entries: $m=128$K → ≈17B.
- Tiled kernels such as [FlashAttention](#flashattention) derive causal & window masks from index arithmetic → fully-masked tiles are skipped, never computed.
- A dense `attn_mask` passed to `F.scaled_dot_product_attention` disables the flash backend, which accepts only `is_causal` → PyTorch falls back to a slower backend.

*Is bidirectional attention dead?*
- For generation, yes.
- Still used where the whole input is known & nothing is generated: embedding, reranking, classification encoders.
```

&nbsp;

### Causal
- **What**: Query $i$ sees keys $j\leq i$ only.
- **Why**: 1 forward pass over $m$ tokens → $m$ next-token training signals.
    - ❌Mask → position $i$ sees token $i{+}1$ = its own label → copies it → loss ≈0, nothing learned.
- **How**: Lower-triangular allow pattern: $M_{ij}=0$ if $j\leq i$, else $-\infty$.

```{attention} Q&A
:class: dropdown
*Does causal masking waste compute?*
- Naive kernel: yes. It computes all $m^2$ scores, then discards ≈half.
- Tiled kernels skip fully-masked tiles → ≈2× less work.
```

&nbsp;

### Prefix
- **What**: Bidirectional inside a prefix of length $p$, causal after it.
- **Why**: The prompt is fully known before generation → causal masking on it gains nothing at inference.
    - Trade: ❌next-token signal on the prefix, ✅bidirectional encoding of it.
- **How**: $M_{ij}=0$ if $(i<p\wedge j<p)$ or $j\leq i$, else $-\infty$.

&nbsp;

### Document
- **What**: Block-diagonal ∩ causal: query & key must be in the same packed document.
- **Why**: Packing without leakage.
    - Pretraining packs many short docs into 1 length-$m$ sequence → ⬇️padding waste.
    - ❌Mask → doc B attends into doc A → cross-document contamination.
- **How**: $M_{ij}=0$ if $\mathrm{doc}(i)=\mathrm{doc}(j)$ and $j\leq i$, else $-\infty$. Reset position ids per document.

```{attention} Q&A
:class: dropdown
*What breaks if position ids are not reset per packed document?*
- **Absolute/learned PE**: the same document encodes differently, depending on where it lands in the pack.
- **RoPE**: nothing, strictly. RoPE depends only on $i-j$ → distances inside a document are unchanged. Reset is for consistency, not correctness.
- → The mask prevents contamination. Position ids alone never do.
```

&nbsp;

## Multi-Head Attention
- **What**: $h$ parallel SDPA heads of width $d_k=d/h$, concatenated, mixed by $W_O$. {cite:p}`vaswani2017attention`
- **Why**: Several relations per token pair.
    - 1 softmax = 1 weighted average = 1 relation.
    - A token needs several at once: its syntactic head, its coreferent, the last mention of the same entity.
    - $h$ heads → $h$ relations at the same total FLOPs ← $d_k$ shrinks by $h$.
- **How**:
    1. Project $X$ → $h$ triples $(Q^{(r)},K^{(r)},V^{(r)})$ of width $d_k$.
    2. SDPA per head, in parallel.
    3. Concat → $\mathbb{R}^{m\times hd_v}$ → $W_O$ → back to $d$.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $W_Q^{(r)}\in\mathbb{R}^{d\times d_k}$: Query projection of head $r$.
    - $W_K^{(r)}\in\mathbb{R}^{d\times d_k}$: Key projection of head $r$.
    - $W_V^{(r)}\in\mathbb{R}^{d\times d_v}$: Value projection of head $r$.
    - $W_O\in\mathbb{R}^{hd_v\times d}$: Output projection.

Forward:

$$
\mathrm{MHA}(X)=\mathrm{Concat}\left(\mathrm{head}_1,\cdots,\mathrm{head}_h\right)W_O,\qquad \mathrm{head}_r=\mathrm{Attn}\left(XW_Q^{(r)},XW_K^{(r)},XW_V^{(r)}\right)
$$

- $\mathrm{Attn}$: SDPA with mask $M$.

Equivalently, a sum of per-head writes into the residual stream:

$$
\mathrm{MHA}(X)=\sum_{r=1}^{h}\mathrm{head}_r\,W_O^{(r)}
$$

- $W_O^{(r)}\in\mathbb{R}^{d_v\times d}$: Rows $(r-1)d_v+1$ to $rd_v$ of $W_O$.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

## Reuses ScaledDotProductAttention & AttentionMask from this page
class MultiHeadAttention(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        assert d % h == 0
        self.h, self.d_k = h, d // h
        ## One fused projection per role, split into heads afterwards
        self.W_q, self.W_k, self.W_v = (nn.Linear(d, d, bias=False) for _ in range(3))
        self.W_o = nn.Linear(d, d, bias=False)
        self.attn = ScaledDotProductAttention()

    def forward(self, x, causal=True):
        B, T, _ = x.shape
        ## (B, T, d) -> (B, h, T, d_k): heads become a batch dim, so all h run as one matmul
        split = lambda t: t.view(B, T, self.h, self.d_k).transpose(1, 2)
        q, k, v = split(self.W_q(x)), split(self.W_k(x)), split(self.W_v(x))
        mask = AttentionMask("causal")(T) if causal else None  ## (T, T), broadcasts over B & h
        o = self.attn(q, k, v, mask)
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
- With $d_k=d_v=d/h$:

    $$
    \underbrace{4md^2}_{\text{projections}}+\underbrace{2m^2d}_{\text{scores \& weighted sums}}\quad\text{MACs}
    $$

- $h$ only sets how $d$ is split, not how much work there is.

*Then why not $h=d$ (one dim per head)?*
- $d_k$ = rank of each head's query-key bilinear form.
- $d_k$⬇️ → each head compares along fewer directions → scores collapse.
- Practice: $d_k=64$–$128$, up to 256 (Gemma 3).

*What does $W_O$ do?*
- Concat alone → each head writes into a fixed, disjoint slice.
- $W_O$ → every head can write into any direction of the residual stream, and heads can cancel each other.

*Cons?*
- Redundancy: many heads are prunable post-hoc with little loss.
- Cache size $\propto h$ → the decode bandwidth wall → [Head Sharing](#head-sharing).
```

&nbsp;

## Head Sharing
- **What**: Fewer KV heads $n_g$ than query heads $h$, or one shared low-rank KV latent.
- **Why**: [Decode](infer/cache.md#decode) is bandwidth-bound. Bytes moved scale with $n_g$; FLOPs scale with $h$.
    - MHA loads $2hd_k$ cached elements per cached token per layer for $\mathcal{O}(hd_k)$ FLOPs → ≈1 FLOP/byte in FP16 → GPU idles.
    - Cut $h$ → lose query subspaces. Cut $n_g$ → lose only *what heads look at*, not *what they ask*.
- **How**: One axis, 3 points.
    - **MQA**: $n_g=1$. All query heads share 1 K/V head.
    - **GQA**: $1<n_g<h$. 1 K/V head per group of $h/n_g$ query heads.
    - **MLA**: cache 1 low-rank latent per token; up-project it to all $h$ heads on the fly.

```{dropdown} Table: Head-Sharing Variants
| Variant | KV heads | Cache elements / token / layer | Ships in |
|:--|:--|:--|:--|
| MHA | $h$ | $2hd_k$ | GPT-2/3, Llama-2 7B/13B |
| [MQA](infer/cache.md#mqa) | $1$ | $2d_k$ | PaLM, Falcon-7B |
| [GQA](infer/cache.md#gqa) | $n_g$ | $2n_gd_k$ | Llama-3/4, Qwen, Mistral, Gemma, gpt-oss |
| [MLA](infer/cache.md#mla) | — (latent) | $d_c+d_h^R$ | DeepSeek-V2/V3, Kimi K2 |

- $d_h$: Content head dim.
- $d_c$: KV latent dim.
- $d_h^R$: Decoupled RoPE key dim.
- DeepSeek-V2: $d_c=4d_h$, $d_h^R=d_h/2$ → $4.5d_h$ ≡ GQA with $n_g=2.25$ groups of width $d_h$.
- Full mechanism, math & code for each: [Architectural Compression](infer/cache.md#architectural-compression).
```

```{attention} Q&A
:class: dropdown
*Which one do you pick?*
- **GQA, $n_g=8$**: default ← 8 KV heads shard cleanly across an 8-GPU tensor-parallel node → no KV head is replicated.
- **MLA**: if you control pretraining. Smaller cache than GQA at equal or better quality; cost = more complex kernel.
- **MQA**: if the cache must be as small as possible, at a quality & stability cost. Ran at frontier scale (PaLM-540B) before GQA superseded it.

*Does head sharing reduce FLOPs?*
- Barely. Only $W_K,W_V$ shrink; attention still runs $h$ query heads.
- The win is memory traffic, which bounds decode.

*Does it change prefill?*
- Almost not. Prefill is compute-bound & amortizes 1 weight load over $m$ tokens → head sharing is a decode-side optimization.

*Why does MHA survive at all?*
- Small models & encoders: cache is not the bottleneck, and MHA is the quality ceiling on this axis.
```

&nbsp;

## QK-Norm
- **What**: Normalize $\mathbf{q}$ & $\mathbf{k}$ before the dot product. {cite:p}`henry2020querykey`
- **Why**: Unbounded logits grow during training.
    - $|s_{ij}|\leq\|\mathbf{q}_i\|\|\mathbf{k}_j\|/\sqrt{d_k}$. Nothing stops $W_Q,W_K$ from growing.
    - → Logits⬆️ → softmax saturates → $A$ one-hot → Jacobian ≈0 → dead head + loss spike.
    - Worse w/ depth⬆️ & LR⬆️ → a standard failure mode of large pretraining runs.
- **How**: RMSNorm w/ learned per-channel gain on $\mathbf{q}$ & $\mathbf{k}$, right after projection.
    - Per-head form: $\|\hat{\mathbf{q}}\|\leq\sqrt{d_k}\|\boldsymbol{\gamma}_q\|_\infty$ for any $X$, $W_Q$ → logits cannot drift with the projections.
    - Scope: per head over $d_k$ (Gemma 3, Qwen3) or over all heads jointly (OLMo 2).
    - Applied before RoPE in practice. RoPE is a rotation → the bound holds either way.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\boldsymbol{\gamma}_q\in\mathbb{R}^{d_k}$: Learned query gain, shared across heads.
    - $\boldsymbol{\gamma}_k\in\mathbb{R}^{d_k}$: Learned key gain, shared across heads.
    - $g$: Learned scalar scale (original form).

Forward, modern form (RMSNorm, per head):

$$
\hat{\mathbf{q}}_i=\frac{\mathbf{q}_i}{\sqrt{\frac{1}{d_k}\sum_{c=1}^{d_k}q_{ic}^2}}\odot\boldsymbol{\gamma}_q,\qquad s_{ij}=\frac{\hat{\mathbf{q}}_i\cdot\hat{\mathbf{k}}_j}{\sqrt{d_k}}
$$

- $\hat{\mathbf{k}}_j$: Same op on $\mathbf{k}_j$ with $\boldsymbol{\gamma}_k$.

Forward, original form ($\ell_2$-norm; learned $g$ replaces $1/\sqrt{d_k}$):

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
- No. $1/\sqrt{d_k}$ fixes logit variance at init, under the iid unit-variance assumption.
- Training breaks that assumption by growing $W_Q,W_K$. QK-Norm re-imposes the bound at every step.

*Why not rely on pre-norm on the residual stream?*
- Pre-norm bounds the layer input $X$, not $W_Q,W_K$.
- Blow-up is head-local: 1 saturating head is enough to spike the loss.

*Cost?*
- 2 RMSNorms per token per head → negligible vs. $\mathcal{O}(m^2d)$ matmuls. Cache size unchanged.

*Any downside?*
- Discards $\|\mathbf{q}\|$ as a content signal.
- → A head cannot sharpen its distribution by growing its query norm. Only $\boldsymbol{\gamma}_q$ can, and it is content-independent.

*Who ships it?*
- Gemma 3, Qwen3, OLMo 2 → common in post-2024 pretraining recipes.
```

&nbsp;

## Attention Sink
- **What**: Large, content-independent attention mass on the first few tokens. {cite:p}`xiao2023efficient`
- **Why**: Softmax weights sum to 1 → a head with nothing to retrieve must still put its mass somewhere.
    - → Training picks a no-op target: token 0, the only key every causal query sees. The next few tokens take the role for almost all queries.
- **How**: 2 forms.
    - **Emergent**: first few tokens absorb the mass regardless of content. StreamingLLM: keeping 4 is enough.
        - Evict them (naive sliding window) → mass moves onto real tokens → perplexity explodes. Fix: pin them → [StreamingLLM](infer/cache.md#streamingllm).
    - **Learned**: trainable per-head logit $\sigma^{(r)}$ in the softmax denominator → explicit "attend to nothing" slot, 0 context cost.
        - Ships in gpt-oss. {cite:p}`openai2025gptoss`

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\sigma^{(r)}$: Learned sink logit of head $r$.

Forward:

$$
a_{ij}=\frac{\exp(s_{ij})}{\exp(\sigma^{(r)})+\sum_{j'\in\mathcal{A}_i}\exp(s_{ij'})}\qquad\Rightarrow\qquad\sum_{j\in\mathcal{A}_i}a_{ij}=1-\frac{\exp(\sigma^{(r)})}{\exp(\sigma^{(r)})+\sum_{j'\in\mathcal{A}_i}\exp(s_{ij'})}<1
$$

Equivalently, append 1 virtual key with logit $\sigma^{(r)}$ and value $\mathbf{0}$:

$$
\mathbf{o}_i=\mathrm{softmax}\left(\left[\mathbf{s}_i,\ \sigma^{(r)}\right]\right)\left[V;\ \mathbf{0}^\top\right]
$$

- $\mathbf{s}_i$: Row $i$ of $S+M$.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

## Reuses AttentionMask from this page
class SinkAttention(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.sink = nn.Parameter(torch.zeros(h))  ## one learned logit per head

    def forward(self, q, k, v, causal=True):
        ## q, k, v: (B, h, T, d_k)
        s = q @ k.transpose(-2, -1) / q.shape[-1] ** 0.5
        if causal:
            s = s + AttentionMask("causal")(q.shape[-2]).to(s)
        ## Append the sink logit as an extra column -> it enters the denominator only
        sink = self.sink.view(1, -1, 1, 1).expand(*s.shape[:-1], 1)
        a = torch.softmax(torch.cat([s, sink], dim=-1), dim=-1)
        ## Drop the sink column: its value is 0, so it contributes nothing to the output
        return a[..., :-1] @ v

## Example
q, k, v = (torch.randn(1, 2, 4, 8) for _ in range(3))
print(SinkAttention(h=2)(q, k, v).shape)  ## torch.Size([1, 2, 4, 8])
```
````

```{attention} Q&A
:class: dropdown
*Why does the sink land on the first tokens?*
- Causal masking → token 0 is the only key visible to every query → the one dump every head can share.
- Later positions are invisible to earlier queries → only partial sinks.

*How do you spot one?*
- Vertical stripe at the leftmost columns, across nearly all heads & layers, at any sequence length.

*Why is a learned sink better than pinning 4 real tokens?*
- ✅0 context slots used, ✅composes with sliding windows for free, ✅head can express "no match" instead of faking one.
- It is 1 extra term in the denominator → no extra key/value bytes in the cache.

*What does this say about softmax as a design choice?*
- Softmax hard-codes "attend to exactly 1.0 of something". That is an assumption, not a requirement.
- Models spend real capacity to work around it.
```

&nbsp;

## FlashAttention
- **What**: Exact attention computed tile by tile in SRAM. {cite:p}`dao2022flashattention`
    - The $m\times m$ score matrix is never written to HBM.
- **Why**: Attention is memory-bound, not compute-bound.
    - Naive kernel round-trips an $m\times m$ matrix through HBM several times (scores → mask → softmax → dropout → $AV$).
    - Matmuls are cheap; moving $\mathcal{O}(m^2)$ elements is not.
    - → Approximate attention (Linformer/Performer/Reformer) cut FLOPs but ignored IO → often ❌wall-clock win.
- **How**:
    1. Tile $Q$ into row blocks, $K,V$ into column blocks.
    2. For each $Q$ block, stream $K,V$ blocks through SRAM.
    3. **Online softmax**: keep a running row max, running sum & running output. Rescale all three by 1 scalar when a new block raises the max.
    4. **Recompute** in backward: save only $O$ & per-row softmax stats → rebuild $S,A$ in SRAM instead of reading them from HBM.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $T$: #Key blocks.
    - $\mathbf{s}^{(t)}$: Logits of 1 query row vs. key block $t$.
    - $\mathbf{v}^{(t)}_j$: Value $j$ of key block $t$.

Process (1 query row; init $\mu^{(0)}=-\infty$, $\ell^{(0)}=0$, $\tilde{\mathbf{o}}^{(0)}=\mathbf{0}$):

1. For $t=1,\dots,T$:

    $$\begin{align*}
    &\mu^{(t)}=\max\left(\mu^{(t-1)},\ \max_j s^{(t)}_j\right) \\
    &\ell^{(t)}=e^{\mu^{(t-1)}-\mu^{(t)}}\ell^{(t-1)}+\sum_je^{s^{(t)}_j-\mu^{(t)}} \\
    &\tilde{\mathbf{o}}^{(t)}=e^{\mu^{(t-1)}-\mu^{(t)}}\tilde{\mathbf{o}}^{(t-1)}+\sum_je^{s^{(t)}_j-\mu^{(t)}}\mathbf{v}^{(t)}_j
    \end{align*}$$

    - $\mu^{(t)}$: Running row max.
    - $\ell^{(t)}$: Running softmax denominator.
    - $\tilde{\mathbf{o}}^{(t)}$: Running unnormalized output.
2. Output $\mathbf{o}=\tilde{\mathbf{o}}^{(T)}/\ell^{(T)}$.
```

```{tip} Derivation
:class: dropdown
*Why is the running rescale exact, not an approximation?*

1. Softmax is shift-invariant: $\mathrm{softmax}(\mathbf{s})=\mathrm{softmax}(\mathbf{s}-c)$ for any scalar $c$.
2. → Accumulate everything relative to the max seen so far, $\mu^{(t)}$, only for fp stability.
3. When block $t$ raises the max, re-base every stored term:

    $$
    e^{s_j-\mu^{(t)}}=e^{s_j-\mu^{(t-1)}}\cdot e^{\mu^{(t-1)}-\mu^{(t)}}
    $$

4. → The whole re-basing is 1 scalar multiply on $\ell$ and $\tilde{\mathbf{o}}$. No term is dropped.
5. At $t=T$: $\ \tilde{\mathbf{o}}^{(T)}/\ell^{(T)}=\sum_j\frac{e^{s_j-\mu}}{\sum_{j'}e^{s_{j'}-\mu}}\mathbf{v}_j$ = exact softmax attention.
6. Backward: $A$ is rebuilt from $Q,K$ & the saved per-row stats → extra FLOPs, but no $\mathcal{O}(m^2)$ HBM read. On a memory-bound op, that trade wins.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class FlashAttention(nn.Module):
    def __init__(self, block=2):
        super().__init__()
        self.block = block  ## #keys per K/V tile (SRAM-sized in a real kernel)

    def forward(self, q, k, v):
        ## q: (m_q, d_k) | k: (m_kv, d_k) | v: (m_kv, d_v). One head -> the algorithm, not the kernel.
        m_q, d_k = q.shape
        o = q.new_zeros(m_q, v.shape[-1])         ## running unnormalized output
        mu = q.new_full((m_q, 1), float("-inf"))  ## running row max
        l = q.new_zeros(m_q, 1)                   ## running row sum of exp
        for j in range(0, k.shape[0], self.block):
            ## Only one K/V tile is live at a time -> the m x m score matrix never exists
            kb, vb = k[j:j + self.block], v[j:j + self.block]
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
print(torch.allclose(FlashAttention()(q, k, v), ref))  ## True
```
````

```{attention} Q&A
:class: dropdown
*Is it an approximation?*
- ❌ Same math. Only the accumulation order differs → fp rounding differs, results do not.

*Does it reduce FLOPs?*
- ❌ It slightly increases them (backward recompute). It reduces HBM traffic, which sets the wall clock.

*How much IO does it save?*
- HBM accesses: $\Theta(md_k+m^2)$ standard vs. $\Theta(m^2d_k^2/\mathcal{M})$ FlashAttention.
    - $\mathcal{M}$: SRAM size.
- Typical $d_k=64$–$128$ & SRAM ≈100KB → $d_k^2$ is many times smaller than $\mathcal{M}$ → many-fold fewer accesses.

*What does it unlock?*
- $\mathcal{O}(m)$ instead of $\mathcal{O}(m^2)$ activation memory ← only per-row stats survive, not $A$.
- → Long-context training becomes possible at all, independent of the speedup.

*When does it NOT fire?*
- Arbitrary dense additive masks, unsupported head dims, fp32, or anything that needs the materialized $A$ (attention-map analysis, some interpretability probes).
- PyTorch then falls back to a slower backend without an error.

*Why did approximate attention lose to an exact method?*
- It optimized FLOPs. IO was the binding constraint.
```

&nbsp;

### FlashAttention-2
- **What**: FlashAttention re-partitioned across GPU thread blocks & warps. {cite:p}`dao2023flashattention2`
- **Why**: FA-1 reaches only 25–40% of peak FLOPs/s ← non-matmul work + poor work partitioning, not IO.
- **How**:
    1. ⬇️Non-matmul FLOPs: rescale $\tilde{\mathbf{o}}$ once at the end, not every block.
    2. Parallelize over the sequence dim too, not only batch × head → occupancy holds at small batch / long context.
    3. $Q$ outer loop, $K,V$ inner loop; split $Q$ across warps → ❌inter-warp shared-memory traffic.

```{attention} Q&A
:class: dropdown
*Speedup?*
- ≈2× over FA-1.
- 50–73% of theoretical max FLOPs/s on A100.
- End-to-end GPT training: up to 225 TFLOPs/s per A100 (72% MFU).
```

&nbsp;

### FlashAttention-3
- **What**: FA-2 restructured for Hopper async hardware + FP8. {cite:p}`shah2024flashattention3`
- **Why**: FA-2 reaches only 35% utilization on H100 ← it ignores TMA & async warp-group matmuls.
- **How**:
    1. **Warp specialization**: producer warps issue TMA loads; consumer warps run matmuls → overlap data movement & compute.
    2. **Interleave** block-wise GEMM & softmax → hide the slow exponentials under tensor-core matmuls.
    3. **FP8** w/ block quantization + incoherent processing (random orthogonal rotation) → outliers do not destroy accuracy.

```{attention} Q&A
:class: dropdown
*Speedup?*
- 1.5–2.0× over FA-2 on H100.
- FP16: up to 740 TFLOPs/s (75% utilization).
- FP8: close to 1.2 PFLOPs/s.
- FP8 numerical error: 2.6× lower than baseline FP8 attention.
```

&nbsp;

## Sparse Attention
- **What**: Attention over a subset of keys per query.
- **Why**: FlashAttention is IO-optimal but still $\mathcal{O}(m^2)$.
    - Attention is $\mathcal{O}(m^2)$; the rest of the layer is $\mathcal{O}(m)$ → attention dominates prefill at long context.
    - Attention maps are near-sparse in practice → most of the quadratic work barely moves the output.

```{attention} Q&A
:class: dropdown
*Static vs. dynamic sparsity?*
- **Static** (sliding window): free to compute, bounds the cache at $W$ per layer, provably loses distant tokens within that layer.
- **Dynamic** ([NSA](#nsa), [DSA](#dsa)): must keep the full cache (any token can be selected) → saves compute, not memory.
- → Complementary, not competing. NSA runs a window branch for exactly this reason.
```

&nbsp;

### Sliding Window
- **What**: Query $i$ attends to the last $W$ keys only.
    - Depth recovers global reach: $l$ layers → receptive span $l(W-1)+1$.
- **Why**: Static + contiguous + cache-bounding, all at once.
    - Static → ❌selection cost.
    - Contiguous → coalesced reads.
    - Cache-bounding → $W$ entries instead of $m$, per sliding-window layer.
- **How**: Interleave with full-attention layers → direct access to any past token survives in those layers. Ratios, span math & cache accounting: [Sliding Window Attention](infer/cache.md#sliding-window-attention) & [Hybrid Attention](infer/cache.md#hybrid-attention).

&nbsp;

### NSA
- **Name**: Native Sparse Attention {cite:p}`yuan2025native`
- **What**: 3 parallel branches per query (compressed blocks, selected blocks, local window), mixed by a learned gate, trained end-to-end.
- **Why**:
    - *Why do we need it?*
        - Post-hoc sparsity ([H2O](infer/cache.md#h2o), [SnapKV](infer/cache.md#snapkv)) prunes a model trained dense → the model never learned to work without those tokens.
        - Most sparse patterns speed up 1 phase only → the other stays dense → small end-to-end win.
        - Per-head token selection breaks GQA: a group must load the union of its heads' picks → traffic barely drops.
    - *Why does it work?*
        - **Blockwise** selection, not per-token → contiguous reads, balanced arithmetic intensity.
        - Selection shared within a GQA group (importance scores summed over the group's heads) → traffic drops.
        - End-to-end trainable despite discrete top-$n$ routing → the model learns what to drop.
- **How**:
    1. **Compress**: each block of tokens → 1 K/V pair via a learned MLP → coarse global view.
    2. **Select**: rank blocks by the compressed branch's attention scores → keep top-$n$ → attend at full granularity.
    3. **Window**: last $W$ tokens verbatim → local detail the other 2 branches blur.
    4. **Gate**: sigmoid gates from the layer input blend the 3 branch outputs.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $c\in\{\mathrm{cmp},\mathrm{slc},\mathrm{win}\}$: Branch.
    - $\tilde{K}^c_i$: Keys kept by branch $c$ for query $i$.
    - $\tilde{V}^c_i$: Values kept by branch $c$ for query $i$.
    - $g^c_i\in[0,1]$: Gate of branch $c$ for query $i$.

Forward:

$$
\mathbf{o}_i=\sum_{c}g^c_i\cdot\mathrm{Attn}\left(\mathbf{q}_i,\tilde{K}^c_i,\tilde{V}^c_i\right),\qquad g^c_i=\mathrm{sigmoid}\left(\mathrm{MLP}\left(\mathbf{x}_i\right)\right)
$$

- $\mathbf{x}_i$: Row $i$ of $X$.
```

```{attention} Q&A
:class: dropdown
*Results?*
- 27B-param MoE (3B active), pretrained with NSA → matches or beats full attention.
    - Token count: the paper states both 260B and 270B.
- Faster than full attention in decode, forward & backward. Speedup grows with $m$.

*Why not prune a dense model at inference (H2O/SnapKV)?*
- Top 20% of attention covers only ≈70% of total attention mass (Chen et al., as cited in NSA).
- → Pruning removes tokens the dense model relies on, e.g., retrieval heads.
- Native training → the model learns to route around the missing tokens.
```

&nbsp;

### DSA
- **Name**: DeepSeek Sparse Attention {cite:p}`deepseekai2025v32`
- **What**: Cheap indexer scores all past tokens → main attention over the top-$k$ only.
- **Why**: Main attention $\mathcal{O}(m^2)$ dominates long-context cost.
    - Goal: add sparsity to an existing dense [MLA](infer/cache.md#mla) model by continued pretraining, not from scratch.
- **How**:
    1. **Lightning indexer**: few small FP8 heads score each query against 1 shared index key per token → cost ≪ main attention.
    2. **Top-$k$** ($k=2048$) tokens per query.
    3. MLA attends over those $k$ tokens only → $\mathcal{O}(mk)$ instead of $\mathcal{O}(m^2)$.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $H^I$: #Indexer heads.
    - $k$: #Selected tokens (overrides global $k$ = class idx).
- Misc:
    - $\mathbf{q}^I_{ir}$: Indexer query of head $r$ at position $i$.
    - $\mathbf{k}^I_j$: Indexer key of token $j$ (1 head, shared by all indexer heads).
    - $w^I_{ir}$: Weight of indexer head $r$ at position $i$.
    - $I_{ij}$: Index score of key $j$ for query $i$.

Forward:

$$
I_{ij}=\sum_{r=1}^{H^I}w^I_{ir}\cdot\mathrm{ReLU}\left(\mathbf{q}^I_{ir}\cdot\mathbf{k}^I_j\right),\qquad \mathcal{A}_i=\mathrm{Top}\text{-}k_{\,j\leq i}\left(I_{ij}\right)
$$

- Paper indices: query $t$, key $s$, indexer head $j$.
- Main attention = MLA restricted to $\mathcal{A}_i$.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

## Reuses AttentionMask from this page
class LightningIndexer(nn.Module):
    def __init__(self, d, n_heads=4, d_i=16, top_k=2):
        super().__init__()
        self.n, self.d_i, self.top_k = n_heads, d_i, top_k
        self.W_q = nn.Linear(d, n_heads * d_i, bias=False)  ## per-head indexer queries
        self.W_k = nn.Linear(d, d_i, bias=False)            ## ONE shared indexer key per token
        self.W_w = nn.Linear(d, n_heads, bias=False)        ## per-token head weights

    def forward(self, x):
        ## x: (m, d) -> additive mask (m, m) that keeps the top-k past keys per query
        m = x.shape[0]
        ## Real DSA also applies RoPE to part of the indexer q/k; omitted here
        q = self.W_q(x).view(m, self.n, self.d_i)            ## (m, H, d_i)
        k = self.W_k(x)                                      ## (m, d_i)
        w = self.W_w(x)                                      ## (m, H)
        ## ReLU, no softmax: only the ranking matters
        scores = torch.einsum("ihd,jd->ihj", q, k).relu()    ## (m, H, m)
        I = (w.unsqueeze(-1) * scores).sum(1)                ## (m, m)
        I = I + AttentionMask("causal")(m)                   ## future keys score -inf
        idx = I.topk(min(self.top_k, m), dim=-1).indices
        ## Early queries have < k past keys -> topk also returns -inf slots; drop them
        keep = torch.zeros_like(I, dtype=torch.bool).scatter(-1, idx, True) & I.isfinite()
        return torch.where(keep, 0.0, float("-inf"))

## Example: 6 tokens, each query keeps <= 2 past keys
mask = LightningIndexer(d=32)(torch.randn(6, 32))
print((mask == 0).sum(-1))  ## tensor([1, 2, 2, 2, 2, 2])
```
````

```{attention} Q&A
:class: dropdown
*Why ReLU instead of softmax in the indexer?*
- Only the ranking matters, not a distribution → normalization is wasted work.
- ReLU is chosen for throughput.

*Doesn't the indexer itself cost $\mathcal{O}(m^2)$?*
- Yes. Few heads, small head dim, 1 shared key head & FP8 → small constant → the $\mathcal{O}(mk)$ main attention dominates.
- Sparse attention buys asymptotics in the expensive term only.

*DSA vs. NSA?*
- NSA: block-level selection, 3 branches, trained natively from scratch.
- DSA: token-level selection, 1 branch, retrofitted onto a dense model by continued pretraining.

*Results?*
- V3.2-Exp vs. V3.1-Terminus under matched training: benchmark parity (MMLU-Pro 85.0 vs. 85.0) at much lower long-context cost.
```

&nbsp;

## Linear Attention
- **What**: Attention with a factorizable kernel $\phi(\mathbf{q})^\top\phi(\mathbf{k})$ in place of $\exp(\mathbf{q}\cdot\mathbf{k})$. {cite:p}`katharopoulos2020transformers`
- **Why**: Softmax forces $\mathcal{O}(m^2)$: it normalizes over all pairs.
    - Drop it → associativity applies: $\left(\phi(Q)\phi(K)^\top\right)V=\phi(Q)\left(\phi(K)^\top V\right)$.
    - → $\mathcal{O}(md_kd_v)$ training, $\mathcal{O}(1)$ per decode step, constant-size state instead of a growing KV cache.
- **How**: Attention becomes an RNN with a matrix state.
    1. Feature map $\phi$, e.g., $\mathrm{elu}(x)+1$ → non-negative scores.
    2. Carry a state (sum of all past value-key outer products) & a normalizer.
    3. Read the state with $\phi(\mathbf{q}_t)$ instead of scanning history.
    4. Train with a chunkwise form: quadratic inside a chunk, recurrent across chunks → tensor cores stay busy.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $t$: Curr position.
    - $\phi$: Feature map.
    - $S_t\in\mathbb{R}^{d_v\times d_k}$: Matrix state (overrides page $S$ = logits).
    - $\mathbf{z}_t\in\mathbb{R}^{d_k}$: Normalizer state.

Forward:

$$
\mathbf{o}_t=\frac{\sum_{j\leq t}\left(\phi(\mathbf{k}_j)\cdot\phi(\mathbf{q}_t)\right)\mathbf{v}_j}{\sum_{j\leq t}\phi(\mathbf{k}_j)\cdot\phi(\mathbf{q}_t)}=\frac{S_t\,\phi(\mathbf{q}_t)}{\mathbf{z}_t\cdot\phi(\mathbf{q}_t)}
$$

$$
S_t=S_{t-1}+\mathbf{v}_t\phi(\mathbf{k}_t)^\top,\qquad \mathbf{z}_t=\mathbf{z}_{t-1}+\phi(\mathbf{k}_t)
$$
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearAttention(nn.Module):
    def phi(self, x):
        return F.elu(x) + 1  ## non-negative features -> weights stay non-negative

    def forward(self, q, k, v):
        ## q, k: (m, d_k) | v: (m, d_v)
        q, k = self.phi(q), self.phi(k)
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
attn = LinearAttention()
q, k, v = (torch.randn(6, 4, dtype=torch.float64) for _ in range(3))
A = (attn.phi(q) @ attn.phi(k).T).tril()
ref = (A / A.sum(-1, keepdim=True)) @ v
print(torch.allclose(attn(q, k, v), ref))  ## True
```
````

```{attention} Q&A
:class: dropdown
*Where does the quality go?*
- $S_t$ is a fixed-capacity $d_v\times d_k$ compression of unbounded history → exact recall of an arbitrary distant token is impossible.
- Softmax attention keeps every token verbatim.

*Why did it come back after losing favor in 2020?*
- For the cache, not the FLOPs. Decode is bandwidth-bound; a constant-size state is the strongest answer to that.
- Smarter state updates (gating + delta rule) fixed the "never forgets, never corrects" failure → [Gated DeltaNet](#gated-deltanet).

*Is it really $\mathcal{O}(1)$ per step?*
- Per step, yes. State = $d_kd_v$ per head per layer, independent of $m$.
- Constant ≈$2d_kd_v$ MACs (1 state write + 1 state read).
- Softmax decode over $t$ cached tokens costs $t(d_k+d_v)$ → arithmetic crossover at $t=\frac{2d_kd_v}{d_k+d_v}=128$ for $d_k=d_v=128$.
- → Arithmetic win grows linearly past that point. Latency crossover depends on hardware, kernels & batching.
```

&nbsp;

### Gated DeltaNet
- **What**: Linear attention with a decay gate & a delta-rule state update. {cite:p}`yang2024gated`
- **Why**: A purely additive state never forgets & never corrects.
    - → Fixed capacity fills with stale associations → retrieval collapse.
    - **Gate** $\alpha_t$: fast global wipe (document boundary, topic switch).
    - **Delta rule** $\beta_t$: correct the value stored along $\mathbf{k}_t$; directions orthogonal to $\mathbf{k}_t$ stay untouched ("X is now 7, not 3").
    - The two are complementary → combine.
- **How**:
    1. Decay the whole state by $\alpha_t$.
    2. Erase the old value read along $\mathbf{k}_t$. Unit keys → $\mathbf{k}_t\mathbf{k}_t^\top$ is the projector onto the $\mathbf{k}_t$ direction.
    3. Write $\mathbf{v}_t$ at $\mathbf{k}_t$ with strength $\beta_t$.
    4. Read with $\mathbf{q}_t$.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $t$: Curr position.
    - $\alpha_t\in(0,1)$: Decay gate.
    - $\beta_t\in(0,1)$: Write strength.
    - $S_t\in\mathbb{R}^{d_v\times d_k}$: Matrix state (overrides page $S$ = logits).

Forward:

$$
S_t=S_{t-1}\left(\alpha_t\left(I-\beta_t\mathbf{k}_t\mathbf{k}_t^\top\right)\right)+\beta_t\mathbf{v}_t\mathbf{k}_t^\top,\qquad \mathbf{o}_t=S_t\mathbf{q}_t
$$

- $I\in\mathbb{R}^{d_k\times d_k}$: Identity.
- Constraint: $\|\mathbf{q}_t\|_2=\|\mathbf{k}_t\|_2=1$.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedDeltaNet(nn.Module):
    def forward(self, q, k, v, alpha, beta):
        ## q, k: (m, d_k) | v: (m, d_v) | alpha, beta: (m,), in (0, 1]
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)  ## unit keys -> k k^T is a projector
        S = q.new_zeros(v.shape[-1], q.shape[-1])               ## (d_v, d_k) state
        out = []
        for t in range(q.shape[0]):
            ## Decay & erase the old value at k_t: S (I - beta k k^T) = S - beta (S k) k^T
            S = alpha[t] * (S - beta[t] * torch.outer(S @ k[t], k[t]))
            ## Write the new value at k_t
            S = S + beta[t] * torch.outer(v[t], k[t])
            out.append(S @ q[t])
        return torch.stack(out)

## Example: write the same key twice (3, then 7); alpha = beta = 1 -> read 7, not 3 + 7
k = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
v = torch.tensor([[3.0], [7.0]])
ones = torch.ones(2)
print(GatedDeltaNet()(k, k, v, alpha=ones, beta=ones)[-1])  ## tensor([7.])
```
````

```{attention} Q&A
:class: dropdown
*How does it relate to DeltaNet & Mamba2?*
- $\alpha_t=1$ → DeltaNet (erase-then-write, no decay).
- Drop the $\mathbf{k}_t\mathbf{k}_t^\top$ term, absorb $\beta_t$ into $\mathbf{v}_t$ → Mamba2's gated additive state.

*Why normalize the keys?*
- Along $\mathbf{k}_t$, the transition $I-\beta_t\mathbf{k}_t\mathbf{k}_t^\top$ scales by $1-\beta_t\|\mathbf{k}_t\|^2$.
- Unit keys & $\beta_t\in(0,1)$ → factor in $(0,1)$; orthogonal directions keep factor 1 → transition is non-expansive → no step amplifies the state.
- Unnormalized keys → factor can leave $[-1,1]$ → repeated steps can amplify the state.
```

&nbsp;

### Hybrid Stack
- **What**: Mostly linear-attention layers + a few full-attention layers, interleaved.
- **Why**: The two failure modes are complementary.
    - Linear: constant state, ❌exact long-range recall.
    - Full: exact recall, $\mathcal{O}(m)$ cache.
    - → A few full layers restore retrieval; the cache shrinks by the layer ratio.
- **How**: Interleave at a fixed ratio, linear : full ≈ 3:1 to 7:1.

```{dropdown} Table: Production Hybrids
| Model | Linear layer | Full layer | Ratio |
|:--|:--|:--|:--|
| MiniMax-01 {cite:p}`minimax2025minimax01` | Lightning attention | Softmax attention | 7:1 |
| Qwen3-Next {cite:p}`qwenteam2025qwen3next` | Gated DeltaNet | Gated attention | 3:1 |
| Kimi Linear {cite:p}`team2025kimi` | KDA | MLA | 3:1 |

- Gated attention: SDPA output × input-dependent sigmoid gate, before $W_O$.
- KDA: Kimi Delta Attention.
- Qwen3-Next: 48 layers = 12 blocks × (3 Gated DeltaNet + 1 gated attention).
- Kimi Linear: KV cache ⬇️ up to 75%, decode throughput up to 6× at 1M context.
```

```{attention} Q&A
:class: dropdown
*Is attention being replaced?*
- No. Frontier-scale linear models ship as hybrids (MiniMax-01, Qwen3-Next, Kimi Linear). The full-attention layers are load-bearing, not legacy.

*Where does the cost land?*
- **Prefill**: the few full layers are $\mathcal{O}(m^2)$ & dominate at long $m$. Linear layers are noise.
- **Decode**: only full layers hold a growing cache → the layer ratio sets the cache reduction.

*Why not enlarge the state instead of adding full layers?*
- The state is read & written every step → compute & bandwidth scale with $d_kd_v$.
- Sized for exact recall → you rebuild a KV cache with extra steps.

*Why does 1 full layer per 3–7 linear layers suffice?*
- The residual stream is shared. A full layer retrieves a distant token once & writes it into the stream.
- Linear layers then process it locally → exact recall is needed at a few points, not everywhere.
```

&nbsp;

## Cross Attention
- **What**: SDPA with $Q$ from the target stream and $K,V$ from a separate source.
- **Why**: Fuse a source that is fully known upfront, has its own length & modality, and must not be re-encoded per decode step.
- **How**: Same SDPA, ❌causal mask (source fully visible). Compute source $K,V$ once; reuse them for the whole generation.

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

## Reuses ScaledDotProductAttention from this page
class CrossAttention(nn.Module):
    def __init__(self, d, d_src, d_k):
        super().__init__()
        self.W_q = nn.Linear(d, d_k, bias=False)      ## queries from the target stream
        self.W_k = nn.Linear(d_src, d_k, bias=False)  ## keys & values from the source
        self.W_v = nn.Linear(d_src, d_k, bias=False)
        self.W_o = nn.Linear(d_k, d, bias=False)
        self.attn = ScaledDotProductAttention()

    def forward(self, x, src, src_mask=None):
        ## x: (B, m, d) target | src: (B, m_src, d_src). No causal mask: the whole source is visible.
        ## src_mask (optional, additive, broadcastable to (B, m, m_src)) only blocks source padding
        return self.W_o(self.attn(self.W_q(x), self.W_k(src), self.W_v(src), src_mask))

## Example: 3 target tokens attend over 7 source positions of another width
ca = CrossAttention(d=16, d_src=24, d_k=8)
print(ca(torch.randn(2, 3, 16), torch.randn(2, 7, 24)).shape)  ## torch.Size([2, 3, 16])
```
````

```{attention} Q&A
:class: dropdown
*Why did decoder-only LLMs drop it?*
- Concatenate source + target into 1 sequence under causal self-attention → equivalent conditioning with 1 stack, 1 set of weights, 1 scaling law.
- Not identical representations: a causal source prefix lacks the bidirectional source encoding of an encoder.

*Where is it still the right answer?*
- **Encoder-decoder ASR/translation** (e.g., Whisper): source is a fixed-length audio encoding, never generated.
- **VLMs that inject vision via cross-attention layers**: image K/V stay outside the causal cache.
- **Frozen-backbone adaptation**: add new cross-attention layers without touching the pretrained self-attention path.

*Prefixing source tokens vs. cross-attention?*
- $m_\text{src}$: #Source tokens.
- Prefixing: source tokens join the causal sequence → prefill is $\mathcal{O}((m_\text{src}+m)^2)$, and every self-attention layer caches them for the whole generation.
- Cross-attention: source K/V are built once & live only in cross-attention layers → self-attention prefill stays $\mathcal{O}(m^2)$; the causal cache never holds the source.
- ❌Free: each decode step still attends over all $m_\text{src}$ source positions → total cross-attention work $\Theta(m_\text{src}m)$. The saving is in the quadratic term & the cache, not the source scan.
```
