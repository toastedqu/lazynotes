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
# Cache
KV cache mechanics, the architectures that shrink it, and the serving systems that manage it. Decoding strategies: [Decoding](dec.md).

Notations:
- $m$: #Tokens in context (i.e., sequence length)
- $b$: #Sequences in batch
- $l$: #Layers
- $h$: #Query heads
- $n_g$: #KV heads (i.e., #groups)
- $d_k$: Dim per head
- $d$: Model dim
- $p$: #Bytes per cached element (FP16/BF16: 2, FP8: 1)
- $W$: Attention window size
- $P$: #Params
- $\mathbf{q}_t,\mathbf{k}_t,\mathbf{v}_t$: Query/Key/Value vectors of token $t$

Override: $m$ = #tokens (global $m$ = #samples), matching `## KV Cache` below.

&nbsp;

## KV Cache
- **What**: Cache the $K$ & $V$ vectors for previous tokens.
- **Why**: They don't change during inference.
    1. ONLY the last hidden state is used to predict next token.
    2. What's the last hidden state?
        - Recall attention formula: $\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$.
        - $Q$: Query vector for LAST token.
        - $K$: Key vectors for ALL tokens.
        - $V$: Value vectors for ALL tokens.
    3. The causal masking in transformer decoders prevents later tokens from affecting earlier tokens.
    4. → At each autoregressive step, $K$ & $V$ of input tokens never change.
    5. $\xrightarrow{\text{cache}}$ Avoid recomputation.
- **How**: Cache.

```{attention} Q&A
:class: dropdown
*Pro Tip:* The query vectors for previous tokens are NEVER needed during inference.

*If it's so good, any cons?*
- ⬆️Memory cost: $O(m\cdot h\cdot d_k\cdot \#\mathrm{layers})$

*When should you turn it off?*
- Very short prompts: Recomputation is cheaper than caching.
- Parallelism & Hardware Acceleration: Caching adds an ALL-layer ALL-head copy for each step for each GPU.
```

&nbsp;

## Prefill
- **What**: One parallel forward pass over the whole prompt → 1st output token + fully populated cache.
- **Why**: Prompt tokens are ALL known upfront → ❌autoregression, ✅one batched matmul over $m$ positions.
    - Causal masking makes 1 parallel pass numerically identical to $m$ sequential steps.
- **How**:
    1. Embed all $m$ prompt tokens.
    2. Per layer: compute $Q,K,V$ for all positions at once → write $K,V$ into cache.
    3. Read logits at the LAST position ONLY → sample 1st output token.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $I$: Arithmetic intensity (FLOPs per byte moved).

Cost:

$$
\underbrace{\mathcal{O}(b\,m^2\,l\,h\,d_k)}_{\text{attention}}+\underbrace{\mathcal{O}(b\,m\,P)}_{\text{weights}}
$$

Weight traffic is $\mathcal{O}(P)$ regardless of $m$:

$$
I_\text{weights}=\frac{2Pbm}{pP}=\frac{2bm}{p}
$$

Compute-bound iff $\frac{2bm}{p}>I_\text{ridge}$, i.e. $bm\gtrsim150$ tokens at FP16 on an A100.
```

```{attention} Q&A
:class: dropdown
*Latency metric?*
- **TTFT** (Time To First Token).

*Compute- or memory-bound?*
- **Compute-bound** whenever $bm$ is large: those tokens share ONE weight load → $I\propto bm$.
- Exception: a very short prompt ($bm\lesssim150$ at FP16 on an A100) stays memory-bound → batching short prefills does help.
- → Batching prefills buys little; the GPU is already saturated by one long prompt.

*What dominates as $m$⬆️?*
- Attention $\mathcal{O}(m^2)$ vs everything else $\mathcal{O}(m)$ → attention crosses over at long context.

*Why does prefill hurt a live server?*
- One long prefill occupies the GPU for many ms → all in-flight decodes stall → ITL spike. Fix: [Chunked Prefill](#chunked-prefill).
```

&nbsp;

## Decode
- **What**: One forward pass per generated token: append 1 row to the cache, attend over all of it.
- **Why**: Token $t$ is unknown until $t-1$ is sampled → strictly sequential, ❌parallelizable over time.
- **How**:
    1. Embed ONLY the newest token → $\mathbf{q}_t$ (1 row).
    2. Per layer: compute $\mathbf{k}_t,\mathbf{v}_t$ → append to cache.
    3. Attend $\mathbf{q}_t$ over ALL $m$ cached keys & values.
    4. Sample → repeat.

```{note} Math
:class: dropdown
Per step, per layer, per sequence:

$$
\mathbf{o}_t=\mathrm{softmax}\left(\frac{\mathbf{q}_t K_{\leq t}^T}{\sqrt{d_k}}\right)V_{\leq t}
$$
- $K_{\leq t},V_{\leq t}\in\mathbb{R}^{t\times d_k}$: Cached keys & values (per KV head).

Bytes moved per step:

$$
\underbrace{pP}_{\text{weights}}+\underbrace{2\,p\,b\,m\,l\,n_g\,d_k}_{\text{KV cache}}
$$
```

````{tip} Derivation
:class: dropdown
*Why is decode memory-bound, and why doesn't batching fix attention?*

1. **Weights**: bytes $=pP$, FLOPs $=2Pb$ → $I_\text{weights}=\dfrac{2b}{p}$.
    - Grows with $b$ → batching DOES fix the GEMMs.
2. **KV cache**: bytes $=2pbmln_gd_k$.
3. Attention FLOPs $=\underbrace{2bmlhd_k}_{\mathbf{q}K^T}+\underbrace{2bmlhd_k}_{AV}=4bmlhd_k$.
4. → 

    $$
    I_\text{attn}=\frac{4bmlhd_k}{2pbmln_gd_k}=\frac{2h}{p\,n_g}
    $$

5. **$b$ and $m$ cancel.** Each sequence owns a private cache → nothing to amortize.
6. FP16 ($p=2$) → $I_\text{attn}=h/n_g$: MHA **1**, GQA-8 **8**, MQA ($h{=}32$) **32**.
7. Ridge point $\approx\frac{312\text{ TFLOP/s}}{2.0\text{ TB/s}}\approx150$ FLOP/byte (A100-80GB, BF16).
8. → Every deployed variant sits **far** below the ridge → decode attention is bandwidth-bound in practice.
    - $I_\text{attn}$ only reaches the ridge at implausible extremes (e.g. FP8 MQA with $h=128$ → 256).
9. → Halving cache bytes ≈ halving decode attention time. Every technique below is this one lever.
````

````{important} Code
:class: dropdown
```python
import torch

class KVCache:
    def __init__(self, b, max_m, l, n_g, d_k, dtype=torch.float16):
        ## Preallocate the full (b, n_g, max_m, d_k) buffer per layer -> no realloc per step
        self.k = [torch.zeros(b, n_g, max_m, d_k, dtype=dtype) for _ in range(l)]
        self.v = [torch.zeros(b, n_g, max_m, d_k, dtype=dtype) for _ in range(l)]
        self.m = 0  ## #tokens currently cached

    def update(self, layer, k_new, v_new):
        ## k_new/v_new: (b, n_g, t, d_k). t = m in prefill, t = 1 in decode
        t = k_new.shape[2]
        s = slice(self.m, self.m + t)
        self.k[layer][:, :, s] = k_new
        self.v[layer][:, :, s] = v_new
        ## Return the WHOLE prefix, not just the new rows -> the query attends over all of it
        return self.k[layer][:, :, :self.m + t], self.v[layer][:, :, :self.m + t]

    def advance(self, t):
        self.m += t  ## once per step, AFTER every layer has written

## Example: 2 layers, b=1, n_g=4 KV heads, d_k=8
cache = KVCache(b=1, max_m=16, l=2, n_g=4, d_k=8)
k, _ = cache.update(0, torch.randn(1, 4, 5, 8), torch.randn(1, 4, 5, 8))
print(k.shape)   ## prefill 5 tokens -> torch.Size([1, 4, 5, 8])
cache.advance(5)
k, _ = cache.update(0, torch.randn(1, 4, 1, 8), torch.randn(1, 4, 1, 8))
print(k.shape)   ## decode 1 token  -> torch.Size([1, 4, 6, 8])
```
````

```{dropdown} Table: Prefill vs Decode
| | **Prefill** | **Decode** |
|:--|:--|:--|
| Tokens per pass | $m$ | 1 |
| Parallel over time | ✅ | ❌ |
| Bottleneck | Compute | Memory bandwidth |
| Arithmetic intensity | $\propto bm$ | $2h/(p\,n_g)$ |
| Latency metric | TTFT | TPOT / ITL |
| Cache op | Bulk write | Append 1 row + read all |
| Batching helps? | Barely (already saturated) | ✅ weights, ❌ attention |
| Attention cost | $\mathcal{O}(m^2)$ | $\mathcal{O}(m)$ |
```

```{attention} Q&A
:class: dropdown
*Why is decode so much less efficient than training?*
- Training: every token in the batch contributes gradients → $I\propto$ #tokens.
- Decode: 1 token per sequence per step → the same weights are re-read for ~1 token of math.

*Where does the time actually go at long context?*
- Short $m$: weight loading dominates → batch harder.
- Long $m$: KV cache traffic dominates → shrink the cache. The crossover is where $2pbmln_gd_k>pP$.

*Does the cache change any past entry?*
- ❌ Never ← causal masking. Cached rows are append-only & immutable, which is exactly what makes them shareable across requests ([Prefix Caching](#prefix-caching)).

*Is the Q vector cached?*
- ❌ Only $K$ & $V$ are reused across steps; $\mathbf{q}_t$ is consumed and discarded.
```

&nbsp;

## Memory Footprint
- **What**: Cache bytes grow **linearly** in $b$, $m$, $l$, $n_g$, $d_k$, $p$.
- **Why**: It sets BOTH serving limits at once.
    - **Capacity**: HBM left after weights → max batch size → throughput ceiling.
    - **Bandwidth**: bytes re-read every decode step → per-token latency.
- **How**: 2 tensors ($K$&$V$) × $n_g$ heads × $d_k$ dims × $l$ layers × $p$ bytes, per token per sequence.

```{note} Math
:class: dropdown
Bytes:

$$
\text{KV bytes}=2\cdot b\cdot m\cdot l\cdot n_g\cdot d_k\cdot p
$$
- $2$: One tensor for $K$, one for $V$.

Per token per sequence:

$$
\text{bytes/token}=2\,l\,n_g\,d_k\,p
$$
```

```{dropdown} Table: KV Cache Footprint (FP16, $b=1$)
| Model | $l$ | $h$ | $n_g$ | $d_k$ | Bytes/token | 8K ctx | 128K ctx |
|:--|:--|:--|:--|:--|:--|:--|:--|
| Llama-3-8B (GQA) | 32 | 32 | 8 | 128 | 128 KiB | 1 GiB | 16 GiB |
| Llama-3-70B (GQA) | 80 | 64 | 8 | 128 | 320 KiB | 2.5 GiB | 40 GiB |
| Llama-3-70B *if MHA* | 80 | 64 | 64 | 128 | 2.5 MiB | 20 GiB | 320 GiB |

The hypothetical MHA row is why GQA is universal: without it a single 128K request would not fit on one 80GB GPU.
```

```{attention} Q&A
:class: dropdown
*Why is $h$ absent from the formula?*
- Only **KV** heads are stored. Query heads cost FLOPs, not cache. That asymmetry is the entire premise of [MQA](#mqa)/[GQA](#gqa).

*Weights are shared across a batch, so why isn't the cache?*
- Weights are static & request-independent. The cache is a function of the request's own tokens → private, unless prefixes literally match.

*Which knob is cheapest to turn?*
- $p$ (quantization): no retraining, immediate 2× at FP8.
- $n_g$ (GQA/MLA): biggest win, but needs (up)training.
- $m$ (eviction/windows): unbounded win, but loses information.

*Does MoE change the cache?*
- ❌ Sparsity is in the FFN. Attention is dense → cache depends on $l,n_g,d_k$ only. A 236B MoE can have a smaller cache than a 70B dense model.
```

&nbsp;

## Architectural Compression
- **What**: Shrink bytes/token by changing the attention module itself.
- **Why**: Lossless at inference time — the model was *trained* with the smaller cache, so nothing is discarded at serving.
    - Cost is moved to (pre)training, which is paid once.

### MQA
- **Name**: Multi-Query Attention {cite:p}`shazeer2019fast`
- **What**: $h$ query heads, **1** shared $K$/$V$ head.
- **Why**: Incremental decoding is bottlenecked by repeatedly loading the $K$/$V$ tensors, not by FLOPs.
- **How**: Keep $W_Q$ per-head; collapse $W_K,W_V$ to a single head broadcast to all queries.

```{note} Math
:class: dropdown
$$
\text{Cache}_\text{MQA}=2\,d_k\,l\quad\text{elements/token}\qquad(n_g=1)
$$

Compression vs MHA: $h\times$.
```

```{attention} Q&A
:class: dropdown
*Pros?*
- $h\times$ ⬇️cache → $h\times$ ⬆️decode arithmetic intensity.

*Cons?*
- Quality degradation & training instability ← all heads retrieve from ONE $K$/$V$ subspace; heads can differ only in what they *ask*, not in what they *look at*.

*Why not just use fewer heads instead?*
- Shrinking $h$ shrinks $Q$ too → fewer representation subspaces. MQA keeps all $h$ query heads and cuts only the cached tensors, i.e., only the thing that is actually the bottleneck.
```

&nbsp;

### GQA
- **Name**: Grouped-Query Attention {cite:p}`ainslie2023gqa`
- **What**: $h$ query heads split into $n_g$ groups; 1 shared $K$/$V$ head per group.
- **Why**: MHA & MQA are the two endpoints of one axis; the interesting models are in between.
    - MHA ($n_g=h$): best quality, biggest cache.
    - MQA ($n_g=1$): smallest cache, degraded & unstable.
    - → Interpolate: near-MHA quality at near-MQA speed.
- **How**:
    1. **Uptrain**, don't retrain: mean-pool the $K$/$V$ projections within each group of an existing MHA checkpoint.
    2. Continue pre-training for **5%** of the original compute.

```{note} Math
:class: dropdown
$$
\text{Cache}_\text{GQA}=2\,n_g\,d_k\,l\quad\text{elements/token},\qquad 1\leq n_g\leq h
$$

Uptraining init, for group $g$ of size $h/n_g$:

$$
W_K^{(g)}=\frac{n_g}{h}\sum_{i\in g}W_K^{(i)},\qquad W_V^{(g)}=\frac{n_g}{h}\sum_{i\in g}W_V^{(i)}
$$
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupedQueryAttention(nn.Module):
    def __init__(self, d, h, n_g):
        super().__init__()
        assert h % n_g == 0
        self.h, self.n_g, self.d_k, self.r = h, n_g, d // h, h // n_g
        self.W_q = nn.Linear(d, h * self.d_k, bias=False)
        ## K/V project to n_g heads only -> this single change IS the cache saving
        self.W_k = nn.Linear(d, n_g * self.d_k, bias=False)
        self.W_v = nn.Linear(d, n_g * self.d_k, bias=False)
        self.W_o = nn.Linear(h * self.d_k, d, bias=False)

    def forward(self, x, cache=None, causal=True):
        B, T, _ = x.shape
        q = self.W_q(x).view(B, T, self.h,   self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.n_g, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_g, self.d_k).transpose(1, 2)
        if cache is not None:
            k, v = cache(k, v)                    ## stores n_g heads, NOT h
        ## Expand n_g -> h only for the matmul. Fused kernels do this in SRAM,
        ## so HBM traffic stays proportional to n_g -- that is the whole point.
        k = k.repeat_interleave(self.r, dim=1)
        v = v.repeat_interleave(self.r, dim=1)
        ## `is_causal` is only valid for a SQUARE score matrix (uncached prefill).
        ## With a cache, T=1 vs M>1 is non-square and torch aligns the mask UPPER-LEFT,
        ## which would expose key 0 only -> pass causal=False for single-token decode.
        o = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        return self.W_o(o.transpose(1, 2).reshape(B, T, -1))

## Example: 8 query heads sharing 2 KV heads (r=4), same ratio family as Llama-3
attn = GroupedQueryAttention(d=512, h=8, n_g=2)
print(attn(torch.randn(2, 6, 512)).shape)   ## prefill -> torch.Size([2, 6, 512])
print(attn(torch.randn(2, 1, 512), causal=False).shape)  ## decode -> torch.Size([2, 1, 512])
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- $h/n_g$× ⬇️cache with near-MHA quality.
- Retrofits an existing MHA checkpoint for 5% of pretraining compute → ❌from-scratch run.
- More stable than MQA ← each group keeps its own $K$/$V$ subspace.

*Does it reduce FLOPs?*
- Marginally: only the $W_K,W_V$ projections shrink; attention still runs over $h$ query heads.
- The win is **memory traffic**, which is what decode is bound by.

*How is $n_g$ chosen in practice?*
- Tensor parallelism: with TP degree $t$, KV heads are sharded across GPUs → want $n_g\geq t$, else KV heads must be **replicated** and the saving partially evaporates.
- Llama-3 uses $n_g=8$ at every size, matching 8-GPU nodes.

*Why does mean-pooling work as an init?*
- It preserves the most information from the pretrained checkpoint. Empirically ordered: mean-pool > select one head > random init — exactly the order of how much of the original weights survive.

*What breaks if $n_g\to1$?*
- Reduces to MQA → the quality/stability cliff it was designed to avoid.
```

&nbsp;

### MLA
- **Name**: Multi-head Latent Attention {cite:p}`deepseekai2024deepseek`
- **What**: Cache ONE low-rank latent $\mathbf{c}_t^{KV}$ per token; reconstruct all $h$ keys & values from it.
- **Why**: GQA saves memory by making heads *share* $K$/$V$ — a hard capacity cut. Low-rank projection saves memory *without* forcing heads to be identical.
    - → Cache like GQA with 2.25 groups, quality **stronger** than MHA.
- **How**:
    1. **Down-project** $\mathbf{h}_t$ → latent $\mathbf{c}_t^{KV}\in\mathbb{R}^{d_c}$, $d_c\ll h\,d_k$. Cache it.
    2. **Up-project** $\mathbf{c}_t^{KV}$ → full per-head $K$&$V$ on the fly.
    3. **Absorb** $W^{UK}$ into $W^{Q}$ and $W^{UV}$ into $W^{O}$ → keys & values are never materialized at all.
    4. Carry RoPE on a **separate, decoupled** key of dim $d_h^R$, shared by all heads, cached alongside.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $W^{DKV}\in\mathbb{R}^{d_c\times d}$: KV down-projection.
    - $W^{UK},W^{UV}\in\mathbb{R}^{h d_k\times d_c}$: KV up-projections.
    - $W^{DQ}\in\mathbb{R}^{d_c'\times d}$, $W^{UQ}\in\mathbb{R}^{h d_k\times d_c'}$: Query down/up-projections.
    - $W^{KR}\in\mathbb{R}^{d_h^R\times d}$: Decoupled key projection (single head).
    - $W^{QR}\in\mathbb{R}^{d_h^R h\times d_c'}$: Decoupled query projection.
- Hyperparams:
    - $d_c$: KV compression dim.
    - $d_c'$: Query compression dim.
    - $d_h^R$: Per-head dim of the decoupled query & key.
- Intermediate values:
    - $\mathbf{h}_t\in\mathbb{R}^{d}$: Layer input at position $t$.

Compression:

$$
\mathbf{c}_t^{KV}=W^{DKV}\mathbf{h}_t,\qquad \mathbf{k}_t^C=W^{UK}\mathbf{c}_t^{KV},\qquad \mathbf{v}_t^C=W^{UV}\mathbf{c}_t^{KV}
$$

$$
\mathbf{c}_t^{Q}=W^{DQ}\mathbf{h}_t,\qquad \mathbf{q}_t^C=W^{UQ}\mathbf{c}_t^{Q}
$$
- Query compression is for training memory only — $\mathbf{c}_t^Q$ is NOT cached.

Decoupled RoPE:

$$
\mathbf{q}_t^R=\mathrm{RoPE}(W^{QR}\mathbf{c}_t^{Q}),\qquad \mathbf{k}_t^R=\mathrm{RoPE}(W^{KR}\mathbf{h}_t)
$$

$$
\mathbf{q}_{t,i}=[\mathbf{q}_{t,i}^C;\mathbf{q}_{t,i}^R],\qquad \mathbf{k}_{t,i}=[\mathbf{k}_{t,i}^C;\mathbf{k}_t^R]
$$
- $\mathbf{k}_t^R$ carries no head index → ONE copy shared by all $h$ heads.

Attention:

$$
\mathbf{o}_{t,i}=\sum_{j=1}^{t}\mathrm{softmax}_j\left(\frac{\mathbf{q}_{t,i}^T\mathbf{k}_{j,i}}{\sqrt{d_k+d_h^R}}\right)\mathbf{v}_{j,i}^C,\qquad \mathbf{u}_t=W^O[\mathbf{o}_{t,1};\cdots;\mathbf{o}_{t,h}]
$$

Cache:

$$
\text{Cache}_\text{MLA}=(d_c+d_h^R)\,l\quad\text{elements/token}
$$

DeepSeek-V2: $d_c=4d_k$, $d_h^R=d_k/2$ → $\frac{9}{2}d_kl$ = GQA with **2.25** groups.
```

````{tip} Derivation
:class: dropdown
*Why must RoPE be decoupled instead of applied to $\mathbf{k}_t^C$?*

1. The absorption trick is what makes MLA cheap: $\mathbf{q}_t^T\mathbf{k}_j=(W^{UQ}\mathbf{c}_t^Q)^TW^{UK}\mathbf{c}_j^{KV}$.
2. Regroup: $\mathbf{c}_t^{Q^T}\underbrace{(W^{UQ^T}W^{UK})}_{\text{precomputable}}\mathbf{c}_j^{KV}$ → $W^{UK}$ folds into $W^{UQ}$ **once, offline**.
3. → Keys are never reconstructed; the latent is queried directly.
4. Now insert RoPE on the key: $\mathbf{k}_j\leftarrow R_jW^{UK}\mathbf{c}_j^{KV}$, with $R_j$ the rotation for position $j$.
5. The product becomes $\mathbf{c}_t^{Q^T}W^{UQ^T}R_{t}^TR_jW^{UK}\mathbf{c}_j^{KV}$, and $R_t^TR_j=R_{j-t}$ sits **between** $W^{UQ^T}$ and $W^{UK}$.
6. Matrix multiplication doesn't commute → the folded matrix would depend on the **relative position** $j-t$ → a different matrix per token pair → ❌precomputable.
7. → Every key would have to be reconstructed each step, restoring MHA's traffic.
8. **Fix**: give RoPE its own dimensions. $[\mathbf{k}^C;\mathbf{k}^R]$ splits the dot product into an absorbable content term and a small $d_h^R$ positional term computed directly.
````

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def rope(x, start=0):
    ## Minimal RoPE over positions in dim -2 (see nlp/transformer.md#rope).
    ## `start` = #tokens already cached, else a decode step would rotate at position 0.
    T, D = x.shape[-2], x.shape[-1]
    pos = torch.arange(start, start + T, device=x.device).unsqueeze(-1)
    freq = 10000.0 ** (-torch.arange(0, D, 2, device=x.device) / D)
    a, b = x[..., 0::2], x[..., 1::2]
    c, s = torch.cos(pos * freq), torch.sin(pos * freq)
    return torch.stack([a * c - b * s, a * s + b * c], dim=-1).flatten(-2)

class MLA(nn.Module):
    def __init__(self, d, h, d_k, d_c, d_r):
        super().__init__()
        self.h, self.d_k, self.d_r = h, d_k, d_r
        self.W_dkv = nn.Linear(d, d_c, bias=False)          ## -> c^{KV}, CACHED (d_c)
        self.W_kr = nn.Linear(d, d_r, bias=False)           ## -> k^R,  CACHED (d_r), 1 head for all
        self.W_uk = nn.Linear(d_c, h * d_k, bias=False)     ## c^{KV} -> content keys
        self.W_uv = nn.Linear(d_c, h * d_k, bias=False)     ## c^{KV} -> values
        ## Fused W^{DQ}/W^{UQ}/W^{QR}: query compression is a TRAINING-memory trick and
        ## has no effect on the cache, so it is collapsed here into one projection.
        self.W_q = nn.Linear(d, h * (d_k + d_r), bias=False)
        self.W_o = nn.Linear(h * d_k, d, bias=False)

    def forward(self, x, cache=None, causal=True, start=0):
        B, T, _ = x.shape
        ## The ONLY two tensors ever stored: d_c + d_r elements per token per layer
        c_kv = self.W_dkv(x)                                       ## (B, T, d_c)
        k_r = rope(self.W_kr(x), start)                            ## (B, T, d_r)
        if cache is not None:
            c_kv, k_r = cache(c_kv, k_r)
        M = c_kv.shape[1]
        ## Reconstruct full per-head K/V from the latent. Serving kernels skip this
        ## entirely by folding W_uk into W_q and W_uv into W_o.
        k_c = self.W_uk(c_kv).view(B, M, self.h, self.d_k).transpose(1, 2)
        v = self.W_uv(c_kv).view(B, M, self.h, self.d_k).transpose(1, 2)
        k = torch.cat([k_c, k_r[:, None].expand(-1, self.h, -1, -1)], dim=-1)
        q = self.W_q(x).view(B, T, self.h, self.d_k + self.d_r).transpose(1, 2)
        ## RoPE touches ONLY the last d_r dims -> content dims stay absorbable
        q = torch.cat([q[..., :self.d_k], rope(q[..., self.d_k:], start)], dim=-1)
        ## SDPA scale = 1/sqrt(q.size(-1)) = 1/sqrt(d_k + d_r), matching the Math block.
        ## As in GQA, causal=True is only valid uncached (square scores); with a cache
        ## and T=1 pass causal=False.
        o = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        return self.W_o(o.transpose(1, 2).reshape(B, T, -1))

## Example: DeepSeek-V2 ratios shrunk -> d_c = 4*d_k, d_r = d_k/2
mla = MLA(d=256, h=4, d_k=16, d_c=64, d_r=8)
print(mla(torch.randn(2, 6, 256)).shape)   ## torch.Size([2, 6, 256])
print(64 + 8, 2 * 4 * 16)                  ## 72 cached vs 128 for MHA, per token per layer
```
````

```{dropdown} Table: KV Cache per Token (# elements)
| Variant | Cache/token | Capability |
|:--|:--|:--|
| MHA | $2\,h\,d_k\,l$ | Strong |
| GQA | $2\,n_g\,d_k\,l$ | Moderate |
| MQA | $2\,d_k\,l$ | Weak |
| MLA | $(d_c+d_h^R)\,l\approx\frac{9}{2}d_k\,l$ | Stronger |

Reproduced from {cite:t}`deepseekai2024deepseek`. Measured in elements, independent of storage precision.
```

```{attention} Q&A
:class: dropdown
*Pros?*
- 93.3% ⬇️cache & 5.76× ⬆️max generation throughput vs DeepSeek 67B (MHA).
- Quality **above** MHA, not merely near it → the low-rank bottleneck acts as a regularizer rather than a capacity cut.

*Cons?*
- Complex: 8 projection matrices vs MHA's 4, + a bespoke RoPE path.
- Absorption reshapes the weights → needs dedicated kernels; a naive implementation materializes $K$/$V$ and loses the win.
- ⬆️Compute at short context ← up-projections run every step. Only pays off when cache traffic dominates.

*Why is it stronger than MHA if it stores less?*
- MHA's $K$/$V$ are already near low-rank across heads. MLA parameterizes that structure explicitly, so capacity goes into the shared latent instead of $h$ redundant copies.

*Why compress $Q$ too if queries aren't cached?*
- Purely to cut **activation memory & params** during training. It is irrelevant to the cache — and it is the reason a separate $d_c'$ exists.

*Is the decoupled key per-head?*
- ❌ $\mathbf{k}_t^R$ is ONE vector shared by all heads (only $\mathbf{q}^R$ is per-head), so it adds $d_h^R$, not $h\cdot d_h^R$, to the cache.
```

&nbsp;

### Sliding Window Attention
- **Name**: SWA {cite:p}`beltagy2020longformer`
- **What**: Each token attends only to the previous $W$ tokens.
- **Why**: Cache $\mathcal{O}(m)\to\mathcal{O}(W)$ and attention $\mathcal{O}(m^2)\to\mathcal{O}(mW)$ — the cache stops growing entirely.
    - Stacking recovers reach: layer $k$ sees $\approx kW$ tokens even though each layer sees $W$.
- **How**: **Rolling buffer** of fixed size $W$; token $i$ overwrites slot $i\bmod W$.

```{note} Math
:class: dropdown
Mask:

$$
M_{ij}=\begin{cases}0 & i-W<j\leq i\\ -\infty & \text{otherwise}\end{cases}
$$

Rolling buffer write:

$$
\text{slot}(i)=i\bmod W\quad\Rightarrow\quad \text{Cache}_\text{SWA}=2\min(m,W)\,n_g\,d_k\,l\,p\ \text{bytes}
$$

Receptive field after $k$ layers: $\approx kW$.
```

```{attention} Q&A
:class: dropdown
*Pros?*
- Constant memory in $m$ → context length limited by quality, not by HBM.
- Mistral-7B: $W=4096$, $l=32$ → theoretical span $\approx131$K. {cite:p}`jiang2023mistral`

*Cons?*
- Information beyond the effective span is genuinely unrecoverable — this is lossy, unlike GQA/MLA.
- Exact retrieval (needle-in-a-haystack) degrades if every layer is local.

*How is it used in modern models?*
- **Hybrid local/global**: interleave many SWA layers with a few full-attention layers, so only the global layers pay $\mathcal{O}(m)$.
- Gemma 2: 1:1 local:global, $W=4096$. Gemma 3: **5:1**, $W=1024$ → KV overhead at 32K drops from 60% (global-only) to <15%. {cite:p}`gemmateam2025gemma`

*Why doesn't a smaller window hurt more?*
- Perplexity is dominated by local context; the sparse global layers carry the long-range dependencies. Gemma 3 found window size can shrink substantially at near-zero perplexity cost.
```

&nbsp;

### Hybrid Attention
- **What**: Interleave attention layers with recurrent layers (SSM / linear attention) that keep a **fixed-size** state.
- **Why**: Every technique above shrinks the *constant* in $\mathcal{O}(m)$. A recurrent layer removes the $m$ entirely.
    - → At 256K context the cache, not the weights, is the binding constraint.
- **How**: Blocks of $a$ attention layers per $m'$ recurrent layers; only the attention layers hold a KV cache.

```{attention} Q&A
:class: dropdown
*Pros?*
- Jamba: $a{:}m'=1{:}7$ → **8×** smaller cache than a vanilla Transformer, 256K context on a single 80GB GPU. {cite:p}`lieber2024jamba`
- Decode throughput ⬆️ ← the recurrent layers are $\mathcal{O}(1)$ per token in both time & memory.

*Cons?*
- Recurrent layers compress history into a fixed state → lossy recall. Pure SSMs underperform on in-context copy/retrieval.
- → Keep a minority of full-attention layers to restore exact lookup. The ratio is the quality/memory dial.

*Why keep any attention at all?*
- Attention's cache is a lossless, addressable record of every token. That is exactly what a fixed-size state cannot provide, and exactly what retrieval needs.
```

&nbsp;

## Runtime Compression
- **What**: Shrink an already-trained model's cache at serving time.
- **Why**: Architecture is fixed once weights ship. Everything left must be done to the cache **contents**.
- **How**: Two orthogonal axes — fewer bytes per entry (quantize) or fewer entries (evict/offload).

### Quantization
- **What**: Store $K$/$V$ in INT8 / FP8 / INT4 instead of FP16.
- **Why**: Cache bytes $\propto p$ → the only knob that needs neither retraining nor information loss in *which* tokens are kept.
- **How**: Per-group affine min-max scaling, dequantized inside the attention kernel.
    - **$K$: per-channel** ← outliers concentrate in a few **fixed** channels; a per-token scale would be hijacked by them and destroy every other channel of that token.
    - **$V$: per-token** ← no channel-outlier structure, and attention reduces $V$ **along the token axis**, so per-token scales factor cleanly out of the weighted sum. {cite:p}`liu2024kivi`

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $n_\text{bit}$: Bit width.
- Misc:
    - $\mathcal{G}$: Quantization group (a channel for $K$, a token for $V$).

Affine quantization over group $\mathcal{G}$:

$$
s=\frac{\max_\mathcal{G}x-\min_\mathcal{G}x}{2^{n_\text{bit}}-1},\qquad q=\left\lfloor\frac{x-\min_\mathcal{G}x}{s}\right\rceil,\qquad \hat{x}=qs+\min_\mathcal{G}x
$$

Effective bytes/element: $p=\frac{n_\text{bit}}{8}+\frac{2\cdot 2}{|\mathcal{G}|}$
- 2nd term: FP16 scale + zero-point amortized over the group → small groups erode the saving.
```

````{important} Code
:class: dropdown
```python
import torch

def quantize(x, dim, n_bit=8):
    ## Asymmetric min-max over `dim`. Codes are kept one-per-uint8 for readability;
    ## real kernels bit-pack sub-byte codes and store s/lo in FP16, which is what the
    ## bytes/element formula above accounts for.
    lo, hi = x.amin(dim, keepdim=True), x.amax(dim, keepdim=True)
    s = (hi - lo).clamp(min=1e-8) / (2 ** n_bit - 1)
    return ((x - lo) / s).round().to(torch.uint8), s, lo

def dequantize(q, s, lo):
    return q.to(s.dtype) * s + lo

## (b, n_g, m, d_k)
k = torch.randn(1, 4, 128, 64)
k[..., 7] *= 20          ## a persistent outlier CHANNEL -- why K cannot be per-token
v = torch.randn(1, 4, 128, 64)

qk, sk, ok = quantize(k, dim=-2)   ## K: reduce over TOKENS  -> one scale per channel
qv, sv, ov = quantize(v, dim=-1)   ## V: reduce over CHANNELS -> one scale per token

## Example: the axis is the entire concept -- note where the 1 lands
print(sk.shape, sv.shape)  ## torch.Size([1, 4, 1, 64]) torch.Size([1, 4, 128, 1])
## Per-token on K would put channel 7's outlier in every token's scale:
bad = dequantize(*quantize(k, dim=-1))
print(f"{(bad - k).abs().mean():.3f} vs {(dequantize(qk, sk, ok) - k).abs().mean():.3f}")
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- FP8 halves the cache with essentially no quality loss → widely supported in production stacks, though still opt-in (vLLM defaults `kv_cache_dtype=auto`, i.e. the model dtype).
- KIVI (2-bit): 2.6× ⬇️peak memory → up to 4× ⬆️batch → 2.35–3.47× ⬆️throughput.

*Cons?*
- Dequantization overhead in the attention kernel; only wins because decode is bandwidth-bound, not compute-bound.
- Sub-4-bit needs a **full-precision residual window** for the most recent tokens, which the current query attends to most sharply.

*Why quantize the cache rather than the weights?*
- Orthogonal: weights are $\mathcal{O}(P)$ and fixed; cache is $\mathcal{O}(bml)$ and unbounded. At long context the cache exceeds the weights → it becomes the larger target. Do both.

*Is this the same as [QLoRA](../post/peft.md#qlora)-style weight quantization?*
- Same arithmetic, different tensor & different constraint: weights are quantized once offline, the cache is quantized continuously on the decode critical path.
```

&nbsp;

### Eviction
- **What**: Drop cache entries for tokens judged unimportant.
- **Why**: Attention is empirically **sparse** — a small subset of positions absorbs most of the mass, so most entries pay full bandwidth for near-zero contribution.
- **How**: Score positions → keep a fixed budget → gather the survivors.

```{dropdown} Table: Eviction Policies
| | **StreamingLLM** | **H2O** | **SnapKV** |
|:--|:--|:--|:--|
| Signal | Position only | Accumulated attention | Attention from an observation window |
| When | Every step | Every step | Once, at end of prefill |
| Keeps | 4 sinks + recent $W$ | Heavy hitters + recent | Clustered per-head winners + recent |
| Granularity | Model-wide | Per head | Per head |
| Target | Infinite streaming | Long generation | Long **prompts** |
| Cost | ~0 | Running score per position | One pooled top-$k$ |
```

#### StreamingLLM
- **What**: Window attention + the first **4** tokens, kept forever. {cite:p}`xiao2023efficient`
- **Why**: Plain window attention collapses the instant token 0 is evicted.
    - Softmax must sum to 1 → when no key matches, the model dumps surplus mass on the earliest tokens (**attention sink**).
    - Initial tokens are visible to every later token → training makes them the natural dumping ground.
    - Evict them → that mass is forced onto real tokens → score distribution shifts → PPL explodes.
- **How**: Cache = [4 sinks] + [rolling recent]. Assign positions by **index within the cache**, not by position in the text.

````{important} Code
:class: dropdown
```python
import torch

class StreamingKVCache:
    def __init__(self, n_sink=4, n_recent=1020):
        self.n_sink, self.n_recent, self.k, self.v = n_sink, n_recent, None, None

    def __call__(self, k_new, v_new):
        ## k_new/v_new: (b, n_g, t, d_k)
        cat = lambda a, b: b if a is None else torch.cat([a, b], dim=2)
        self.k, self.v = cat(self.k, k_new), cat(self.v, v_new)
        if self.k.shape[2] > self.n_sink + self.n_recent:
            ## Sinks are pinned; everything else rolls
            keep = lambda t: torch.cat([t[:, :, :self.n_sink], t[:, :, -self.n_recent:]], dim=2)
            self.k, self.v = keep(self.k), keep(self.v)
        ## RoPE must be applied downstream by CACHE index, not text index:
        ## after eviction the sinks sit adjacent to tokens thousands of steps away.
        return self.k, self.v

## Example: 2 sinks + 4 recent, fed 10 tokens
c = StreamingKVCache(n_sink=2, n_recent=4)
for _ in range(10):
    k, v = c(torch.randn(1, 2, 1, 8), torch.randn(1, 2, 1, 8))
print(k.shape)   ## torch.Size([1, 2, 6, 8])
```
````

```{attention} Q&A
:class: dropdown
*Why exactly 4?*
- Ablation: 1 or 2 sinks do NOT restore perplexity; 4 suffices; beyond 4 is marginal. The models weren't pretrained with a fixed BOS, so the sink role is spread over several early tokens.

*Pros?*
- 4M-token streaming at constant memory, ❌finetuning.
- 22.2× faster than sliding window with recomputation.

*Cons?*
- ❌ It does NOT extend the context window. Tokens outside the rolling window are gone — the model stays *stable*, not *informed*.

*Is the sink semantic?*
- ❌ Replacing the 4 initial tokens with linebreak tokens recovers perplexity just as well → the sink is a **positional** artifact of softmax normalization, not content.

*Consequence for architecture?*
- A dedicated learnable sink token can be added at pretraining, which several modern models now do.
```

&nbsp;

#### H2O
- **Name**: Heavy-Hitter Oracle {cite:p}`zhang2023h2o`
- **What**: Evict to a budget of {heavy hitters} ∪ {recent tokens}, scored by **accumulated** attention.
- **Why**: A small set of tokens ("heavy hitters") absorbs most attention mass across the whole generation, and removing them degrades quality sharply — so importance is measurable, not just positional.
- **How**:
    1. Maintain a running sum of attention received, per position, per head.
    2. At each step, greedily keep top-scoring positions + the most recent ones.
    3. Evict the rest.

```{attention} Q&A
:class: dropdown
*Pros?*
- 20% budget → up to 29× ⬆️throughput vs DeepSpeed Zero-Inference & HF Accelerate, 3× vs FlexGen; up to 1.9× ⬇️latency at equal batch size.
- Eviction posed as dynamic submodular maximization → a theoretical guarantee for the greedy policy.

*Cons?*
- **Irreversible**: an evicted token cannot come back if a later query needs it. Accumulated past attention is only a proxy for future relevance.
- Score bookkeeping runs on the decode critical path and fights fused kernels that never materialize the attention matrix.

*Why include recent tokens unconditionally?*
- Recency is the strongest predictor of the next query's focus, and new tokens have had no chance to accumulate score → pure top-$k$ would starve them.
```

&nbsp;

#### SnapKV
- **What**: Compress the **prompt's** cache once, at the end of prefill, using the last few prompt tokens as voters. {cite:p}`li2024snapkv`
- **Why**: Long-prompt workloads (RAG, long docs) are dominated by prompt KV, not generated KV — and per-head attention patterns are already decided **before** generation starts.
    - → One-shot selection, ❌per-step bookkeeping.
- **How**:
    1. **Vote**: aggregate attention from an observation window at the prompt's end over all prompt positions.
    2. **Pool**: max-pool the scores so neighbours of a winner survive — else selected tokens are torn out of their phrases.
    3. **Select**: per-head top-$k$ to the budget, plus the observation window itself.

````{important} Code
:class: dropdown
```python
import torch
import torch.nn.functional as F

def snapkv(k, v, q_obs, budget, kernel=7):
    ## k, v: (b, h, m, d_k) prompt cache. q_obs: (b, h, w, d_k) = last w prompt queries.
    b, h, m, d = k.shape
    w = q_obs.shape[2]
    if m <= budget:
        return k, v
    ## 1. Vote: scaled, CAUSALLY MASKED attention from the observation window.
    ##    Row r of the window sits at absolute position m-w+r, so it may not see beyond it.
    logits = (q_obs @ k.transpose(-1, -2)) / d ** 0.5                      ## (b, h, w, m)
    pos = torch.arange(m, device=k.device)
    mask = pos > (m - w + torch.arange(w, device=k.device))[:, None]       ## (w, m)
    scores = logits.masked_fill(mask, float("-inf")).softmax(-1).sum(dim=2)  ## (b, h, m)
    ## 2. Pool: keep a winner's neighbours too, else we slice tokens out of phrases
    scores = F.max_pool1d(scores, kernel, stride=1, padding=kernel // 2)
    ## 3. Select per HEAD -- heads disagree about what matters
    keep = scores[..., :m - w].topk(budget - w, dim=-1).indices
    tail = torch.arange(m - w, m, device=k.device).expand(b, h, w)         ## always keep the window
    idx = torch.cat([keep, tail], -1).sort(-1).values[..., None].expand(-1, -1, -1, d)
    return k.gather(2, idx), v.gather(2, idx)

## Example: 128-token prompt -> 32-token budget, 8-token observation window
k, v = torch.randn(1, 4, 128, 16), torch.randn(1, 4, 128, 16)
print(snapkv(k, v, torch.randn(1, 4, 8, 16), budget=32)[0].shape)  ## torch.Size([1, 4, 32, 16])
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- 3.6× ⬆️generation speed & 8.2× ⬆️memory efficiency at 16K input, ❌finetuning.
- 380K context on a single A100-80GB with negligible needle-in-a-haystack loss.
- Compression cost is paid **once**, so decode stays untouched.

*Cons?*
- Assumes the prompt's relevant regions are fixed before generation → fails when the answer's focus shifts mid-generation.
- Prompt-only: generated tokens still accumulate uncompressed.

*Why pool?*
- Attention scores are spiky but information is contiguous. Keeping isolated peak tokens preserves the score and destroys the phrase.

*Why per-head rather than per-layer?*
- Heads specialize (positional, syntactic, retrieval). A shared budget would let one head's pattern evict another's evidence.
```

&nbsp;

### Offloading
- **What**: Spill cache (and weights) to CPU DRAM / NVMe, stream back on demand. {cite:p}`sheng2023flexgen`
- **Why**: Host memory is $10$–$100\times$ larger than HBM → capacity ceiling lifted.
- **How**: Overlap transfer with compute; schedule the traversal order to maximize reuse per transfer.

```{attention} Q&A
:class: dropdown
*Cons?*
- PCIe ≈ $\mathcal{O}(10)$ GB/s vs HBM ≈ $\mathcal{O}(10^3)$ GB/s → ~50× slower. Directly on the critical path of a bandwidth-bound phase.

*So when is it actually worth it?*
- **Throughput-oriented offline** batch jobs where per-token latency is irrelevant.
- **Cold storage for reuse**: keep an evicted prefix in DRAM so a later request can [prefix-cache](#prefix-caching) it instead of re-prefilling. Re-loading is cheaper than recomputing when the prompt is long.
- ❌ Interactive serving of a single stream.
```

&nbsp;

## Serving
- **What**: Manage the cache across many concurrent requests.
- **Why**: The techniques above shrink one request's cache. Throughput is decided by how many requests share the GPU, which is an **allocation & scheduling** problem.
- **How**: Treat the cache as a paged, shareable, schedulable resource rather than a per-request tensor.

### PagedAttention
- **What**: Store the cache in fixed-size non-contiguous **blocks** + a per-sequence **block table**. {cite:p}`kwon2023efficient`
- **Why**: Output length is unknown a priori, so naive systems pre-allocate a contiguous chunk for `max_len`.
    - **Reserved**: allocated for tokens not yet generated.
    - **Internal frag**: the request finished early; the tail is never used.
    - **External frag**: leftover gaps between differently-sized chunks.
    - → Only **20.4–38.2%** of KV memory held real tokens in prior systems.
- **How**: OS virtual memory, applied to the cache.
    1. Carve HBM into uniform physical blocks (typically 16 tokens).
    2. Per sequence, keep a block table: logical index → physical block.
    3. Allocate one block **on demand**, only when the current block fills.
    4. Attention kernel gathers by block table → blocks need not be adjacent.
    5. Share blocks across sequences by refcount; **copy-on-write** when one diverges.

````{important} Code
:class: dropdown
```python
import torch

class PagedKVCache:
    def __init__(self, n_blocks, block_size, n_g, d_k, dtype=torch.float16):
        ## ONE physical pool shared by every sequence -> no per-request reservation
        self.pool_k = torch.zeros(n_blocks, block_size, n_g, d_k, dtype=dtype)
        self.pool_v = torch.zeros(n_blocks, block_size, n_g, d_k, dtype=dtype)
        self.block_size, self.free = block_size, list(range(n_blocks))
        self.tables, self.lens = {}, {}     ## seq_id -> [physical block ids] / length

    def append(self, sid, k, v):
        ## k, v: (n_g, d_k) for ONE new token
        t, n = self.tables.setdefault(sid, []), self.lens.get(sid, 0)
        if n % self.block_size == 0:
            t.append(self.free.pop())       ## allocate lazily, one block at a time
        blk, off = t[n // self.block_size], n % self.block_size
        self.pool_k[blk, off], self.pool_v[blk, off] = k, v
        self.lens[sid] = n + 1

    def gather(self, sid):
        ## Real kernels gather block-by-block inside the attention loop; materialized here
        n, t = self.lens[sid], self.tables[sid]
        return self.pool_k[t].flatten(0, 1)[:n], self.pool_v[t].flatten(0, 1)[:n]

## Example: block_size=4, 6 tokens -> 2 blocks, 2 wasted slots (internal frag ONLY)
cache = PagedKVCache(n_blocks=8, block_size=4, n_g=2, d_k=16)
for _ in range(6):
    cache.append("a", torch.randn(2, 16), torch.randn(2, 16))
k, v = cache.gather("a")
print(len(cache.tables["a"]), k.shape)   ## 2 torch.Size([6, 2, 16])
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- Near-zero waste → far larger batches → **2–4×** ⬆️throughput at equal latency vs FasterTransformer & Orca.
- Enables sharing: parallel sampling and beam search share the prompt's blocks instead of duplicating them.

*Cons?*
- Attention must be a gather over blocks → ❌stock `scaled_dot_product_attention`, ✅custom kernel.
- Block table indirection costs a little latency per step.

*Which fragmentation survives?*
- Internal only, and bounded by (block_size − 1) slots in the **last** block of each sequence. External fragmentation disappears because all blocks are the same size.

*Trade-off in block size?*
- Small: less waste, more indirection & worse memory coalescing.
- Large: better kernel efficiency, more waste and coarser prefix sharing.

*What happens under memory pressure?*
- Preempt a sequence: either **swap** its blocks to CPU or **discard & recompute** them later. Recompute is often cheaper ← prefill is compute-bound and parallel, while swapping is bandwidth-bound.
```

&nbsp;

### Prefix Caching
- **What**: Reuse cached blocks across **requests** that share a token prefix. {cite:p}`kwon2023efficient`
- **Why**: Real traffic is enormously redundant — system prompts, few-shot exemplars, multi-turn history, agent scratchpads, document QA over one document.
    - Cached rows are immutable & position-dependent only through the prefix → identical prefix ⇒ identical KV. Reuse is **exact**, not approximate.
    - vLLM reserved blocks for *predefined* prefixes; the automatic, any-prefix version is **RadixAttention**. {cite:p}`zheng2024sglang`
- **How**:
    1. Hash each block by (its tokens + the hash of all preceding blocks) → the hash identifies a whole prefix.
    2. On arrival, match the longest run of hashes already resident.
    3. Skip prefill for matched blocks; bump their refcount.
    4. Evict unreferenced blocks by LRU. SGLang organizes them as a **radix tree** over token sequences, matching any prefix rather than a preregistered one.

````{important} Code
:class: dropdown
```python
import hashlib

def block_hashes(tokens, block_size=16):
    ## Chain each block's hash into the next -> equal hash <=> identical FULL prefix,
    ## so a match at block i implies every earlier block matched too.
    out, parent = [], b""
    for i in range(0, len(tokens) - block_size + 1, block_size):
        parent = hashlib.sha256(parent + bytes(str(tokens[i:i + block_size]), "utf8")).digest()
        out.append(parent)
    return out

## Example: two requests sharing a 32-token system prompt
sys_prompt = list(range(32))
a, b = sys_prompt + [100, 101] * 8, sys_prompt + [200, 201] * 8
ha, hb = block_hashes(a), block_hashes(b)
hit = sum(1 for x, y in zip(ha, hb) if x == y)
print(len(ha), hit)   ## 3 2 -> 2 of 3 blocks reused, prefill skipped for 32 of 48 tokens
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- ⬇️TTFT & ⬇️prefill FLOPs proportional to the shared prefix — often the majority of the prompt.
- SGLang: up to 6.4× ⬆️throughput on agent, few-shot, and multi-turn workloads.
- This is what commercial "prompt caching" APIs expose.

*Cons?*
- **Exact prefix match only**, at block granularity. One differing token at position 0 invalidates everything after it.
- Cached blocks occupy HBM that could hold running requests → eviction policy becomes a throughput/hit-rate trade-off.

*Practical consequence for prompt design?*
- Put **static content first** (system prompt → tools → few-shot → retrieved docs → user turn). Anything volatile early (a timestamp, a request ID) destroys the whole cache.

*Why a radix tree instead of a flat hash map?*
- Requests form a prefix hierarchy; a radix tree matches the longest shared prefix in one traversal and makes LRU eviction operate on shared subtrees.

*Interaction with PagedAttention?*
- Prefix caching is only practical **because** of paging: sharing needs blocks that multiple block tables can point at, with refcounts and copy-on-write.
```

&nbsp;

### Continuous Batching
- **What**: Schedule at **iteration** granularity — admit and retire requests between decode steps. {cite:p}`yu2022orca`
- **Why**: Static batching runs the batch until the **longest** sequence finishes.
    - Finished slots keep occupying memory & compute → GPU idles on padding.
    - New requests wait for the whole batch → queueing delay ⬆️.
- **How**:
    1. **Iteration-level scheduling**: after every step, evict finished sequences and admit queued ones.
    2. **Selective batching**: batch the token-wise ops (FFN, projections) across all sequences; in Orca, run attention **per sequence** ← different lengths ⇒ ❌single matmul.

```{attention} Q&A
:class: dropdown
*Pros?*
- ⬆️GPU occupancy & ⬇️queueing delay, largest when output lengths are highly variable — which is the normal case.

*Why must attention be excluded from the batched ops?*
- Token-wise ops see each token independently → a ragged batch flattens to one matrix.
- Attention couples a token to its **own** sequence's cache, whose length differs per request → no shared shape, so Orca split it out.
- Later ragged/paged attention kernels do batch variable lengths directly, so this is an implementation constraint, not a law.

*Relation to PagedAttention?*
- Complementary: continuous batching decides **which** requests run each step; paging decides **where** their cache lives. Continuous batching without paging just fragments memory faster.
```

&nbsp;

### Chunked Prefill
- **What**: Split a prefill into fixed-size token chunks and co-schedule each chunk with ongoing decodes. {cite:p}`agrawal2024taming`
- **Why**: Prefill and decode have opposite profiles, and mixing them naively is bad for both.
    - Prefill-only iteration → all in-flight decodes **stall** → ITL spike.
    - Decode-only iteration → ~1 token per sequence → GPU compute idle.
- **How**: **Stall-free** schedule.
    1. Fill the batch with all pending decodes first.
    2. Top up to a fixed token budget with a slice of some prefill.
    3. → Every iteration has a uniform token count → ⬇️pipeline bubbles.

```{attention} Q&A
:class: dropdown
*Pros?*
- Decodes never pause → tail ITL bounded under load.
- Piggybacked prefill tokens raise arithmetic intensity of otherwise memory-bound decode iterations → both metrics improve at once.
- 2.6× ⬆️serving capacity (Mistral-7B, 1×A100) & 3.7× (Yi-34B, 2×A100) vs vLLM; 5.6× on Falcon-180B with pipeline parallelism.

*Cons?*
- Chunk $i$ must attend to the KV of chunks $<i$ → the prompt's cache is re-read once per chunk → extra attention memory traffic, growing as chunk size ⬇️.

*How is chunk size chosen?*
- Large: less re-read, but a chunk long enough to stall decodes again.
- Small: tighter ITL, more redundant cache reads. Tuned to the ITL SLO.
```

&nbsp;

### Disaggregation
- **What**: Run prefill and decode on **separate** GPU pools; ship the cache between them. {cite:p}`zhong2024distserve`
- **Why**: One pool cannot be optimal for two workloads with opposite bottlenecks and separate SLOs.
    - Colocating → prefill/decode interference.
    - Couples the parallelism plan: prefill wants tensor parallelism for latency, decode wants larger batches for throughput.
    - TTFT and TPOT must otherwise be traded against each other.
- **How**:
    1. Prefill node computes the prompt's cache.
    2. Transfer over NVLink / RDMA.
    3. Decode node resumes generation.
    4. Size & parallelize each pool independently against its own SLO.

```{attention} Q&A
:class: dropdown
*Pros?*
- 7.4× more requests **or** 12.6× tighter SLO vs colocated systems, at >90% SLO attainment.
- Each phase gets its own batch policy & parallelism degree.

*Cons?*
- The whole cache crosses the interconnect per request → only viable when transfer time ≪ prefill time, i.e., high-bandwidth links and long prompts.
- More moving parts: two pools to provision, and the ratio must track live traffic mix.

*Why is the transfer affordable at all?*
- Cache size ($2mln_gd_kp$ bytes) is linear in $m$ while prefill compute is quadratic → the ratio improves as prompts get longer.

*Where does it lead?*
- A cluster-wide **KV store**: pool DRAM/SSD across nodes so a cache can be produced once and reused by any decode node, merging disaggregation with prefix caching. {cite:p}`qin2024mooncake`
```

&nbsp;