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
# Transformer
The original encoder-decoder Transformer for machine translation, component by component. Study notes from {cite:t}`vaswani2017attention`.

Notations:
- $m_\text{src}$: #Source tokens
- $m$: #Target tokens
- $d$: Model dim (paper: $d_\text{model}$)
- $d_{ff}$: FFN inner dim
- $h$: #Heads
- $d_k$: Dim per query/key head
- $d_v$: Dim per value head
- $N$: #Layers per stack
- $\mathcal{V}$: Shared source-target vocab
- $E\in\mathbb{R}^{|\mathcal{V}|\times d}$: Shared embedding matrix
- $H$: Hidden states entering a sub-layer, one $d$-dim row per token
- $Z\in\mathbb{R}^{m_\text{src}\times d}$: Encoder output (memory)
- $x$: Source sequence
- $y_t$: $t$-th target token
- $P_\text{drop}$: Dropout rate

Override: $m$ = #target tokens (global $m$ = #samples).

&nbsp;

## Architecture
- **What**: Encoder-decoder seq2seq model built only from attention, position-wise FFNs, residuals & LayerNorm. {cite:p}`vaswani2017attention`
- **Why**: RNN seq2seq is sequential.
    - RNN: $\mathcal{O}(m)$ sequential steps per layer → ❌parallel over time → slow training.
    - RNN/CNN: path between 2 distant tokens grows with distance → long-range deps hard to learn.
    - Self-attention: $\mathcal{O}(1)$ sequential steps & $\mathcal{O}(1)$ path between any 2 tokens.
- **How**:
    1. **Encoder**: Source tokens → embedding + PE → $N$ encoder layers → memory $Z$.
    2. **Decoder**: Target tokens shifted right → embedding + PE → $N$ decoder layers. Every layer reads $Z$ via cross-attention.
    3. **Output**: Linear + softmax → next-token distribution.
    - **Training**: all target positions in 1 pass. **Inference**: 1 token per step.

```{image} ../images/nlp-transformer/transformer.png
:align: center
:width: 500px
```

```{note} Example
:class: dropdown
*Where do the base model's 65M params live?* (weights only; $d=512$, $d_{ff}=2048$, $|\mathcal{V}|\approx37000$)

| Block | Formula | #Params |
|:--|:--|:--|
| MHA | $4d^2$ | 1.05M |
| FFN | $2dd_{ff}$ | 2.10M |
| Encoder layer | MHA + FFN | 3.15M |
| Decoder layer | 2 MHA + FFN | 4.19M |
| Encoder ($\times6$) | | 18.9M |
| Decoder ($\times6$) | | 25.2M |
| $E$ (shared $\times3$) | $\lvert\mathcal{V}\rvert d$ | 18.9M |
| **Total** | | **≈63.0M** |

- Biases & LayerNorm add <0.1M.
- Gap to the reported 65M ← vocab is only "about 37000" & unlisted implementation details.
- FFN = 2/3 of every encoder layer.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

## Reuses SinusoidalPE from nlp/pos.md
## Reuses TokenEmbedding, EncoderLayer, DecoderLayer from the sections below
class Transformer(nn.Module):
    def __init__(self, V, d=512, h=8, d_ff=2048, N=6, p=0.1):
        super().__init__()
        self.emb = TokenEmbedding(V, d)              ## ONE matrix: source, target & output
        self.pe = SinusoidalPE(d)
        self.drop = nn.Dropout(p)                    ## dropout on embedding + PE sums
        self.enc = nn.ModuleList(EncoderLayer(d, h, d_ff, p) for _ in range(N))
        self.dec = nn.ModuleList(DecoderLayer(d, h, d_ff, p) for _ in range(N))

    def encode(self, src):
        z = self.drop(self.pe(self.emb(src)))
        for layer in self.enc:
            z = layer(z)
        return z                                     ## (B, m_src, d) memory

    def decode(self, tgt_in, z):
        x = self.drop(self.pe(self.emb(tgt_in)))
        for layer in self.dec:
            x = layer(x, z)                          ## every layer reads the SAME top-encoder z
        return self.emb.logits(x)                    ## (B, m, |V|)

    def forward(self, src, tgt_in):
        ## padding masks omitted for brevity
        return self.decode(tgt_in, self.encode(src))

## Example
model = Transformer(V=100, d=32, h=4, d_ff=64, N=2)
src, tgt_in = torch.randint(0, 100, (2, 7)), torch.randint(0, 100, (2, 5))
print(model(src, tgt_in).shape)  ## torch.Size([2, 5, 100])
```
````

```{dropdown} Table 1: Layer Types
| Layer | Compute per layer | Sequential ops | Max path length |
|:--|:--|:--|:--|
| Self-attention | $\mathcal{O}(m^2d)$ | $\mathcal{O}(1)$ | $\mathcal{O}(1)$ |
| Recurrent | $\mathcal{O}(md^2)$ | $\mathcal{O}(m)$ | $\mathcal{O}(m)$ |
| Convolution | $\mathcal{O}(cmd^2)$ | $\mathcal{O}(1)$ | $\mathcal{O}(\log_c m)$ |
| Self-attention (restricted) | $\mathcal{O}(rmd)$ | $\mathcal{O}(1)$ | $\mathcal{O}(m/r)$ |

- $c$: Kernel size. Path $\mathcal{O}(\log_c m)$ needs dilated convolutions. Contiguous kernels need $\mathcal{O}(m/c)$ layers.
- $r$: Neighborhood size of restricted self-attention.
```

```{dropdown} Table 2: Hyperparams
| | $N$ | $d$ | $d_{ff}$ | $h$ | $d_k=d_v$ | $P_\text{drop}$ | $\epsilon_{ls}$ | Steps | #Params |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| Base | 6 | 512 | 2048 | 8 | 64 | 0.1 | 0.1 | 100K | 65M |
| Big | 6 | 1024 | 4096 | 16 | 64 | 0.3 | 0.1 | 300K | 213M |

- Big EN-FR used $P_\text{drop}=0.1$.
- Hardware: 1 machine, 8 P100 GPUs. Base: 12 hours. Big: 3.5 days.
```

```{dropdown} Table 3: Original vs Llama-style LLM
| Component | Original (2017) | Typical modern decoder-only LLM |
|:--|:--|:--|
| Stack | Encoder-decoder | Decoder-only |
| Norm placement | Post-LN | Pre-norm |
| Norm | [LayerNorm](../dl/foundation/module.md#layernorm) | [RMSNorm](../dl/foundation/module.md#rmsnorm) |
| Position | Sinusoidal, added at input | [RoPE](pos.md#rope) on $\mathbf{q},\mathbf{k}$ |
| FFN | ReLU, $d_{ff}=4d$ | Gated (SwiGLU), 3 matrices. $d_{ff}=\frac{8}{3}d$ keeps the param count of $4d$ ReLU; actual ratio is model-specific (Llama 3: 3.25–3.5) |
| Attention | MHA | [GQA](infer/cache.md#gqa) |
| Bias terms | Yes | Usually none |
| Embedding | Tied ×3 | Often untied |
| Dropout | 0.1 | Usually 0 in pretraining |
```

```{attention} Q&A
:class: dropdown
*When is self-attention cheaper than recurrence?*
- Table 1 compares only the mixing step: $m^2d<md^2$ ⟺ $m<d$.
- Full MHA layer also pays $4md^2$ for the $Q,K,V,O$ projections → total ≈ $4md^2+2m^2d$. The $m^2$ term dominates only when $m\gtrsim2d$.
- 2017 MT: subword sentences ≪ 512 tokens → attention mixing was cheap.
- Long context: $m\gg d$ → attention dominates → [FlashAttention](attn.md#flashattention), [Sparse Attention](attn.md#sparse-attention).

*Is the Transformer "attention only"?*
- ❌. FFN holds 2/3 of encoder-layer params. Residuals & LayerNorm are needed to train the stack.

*Results?*
- WMT14 EN-DE: 28.4 BLEU (big). >2 BLEU over the best prior models, ensembles included.
- WMT14 EN-FR: 41.8 BLEU (big), new single-model SOTA.
    - Caveat: abstract & results table say 41.8. The results text says 41.0.
- Base already beat all prior models on EN-DE at a fraction of their training FLOPs.

*What did the ablations show?* (EN-DE dev, base = 25.8 BLEU)
- $h=1$ → 24.9. $h=32$ → 25.4. → Too few & too many heads both hurt.
- $d_k$⬇️ (16, 32) → worse → computing compatibility is not easy.
- Bigger $d$, $d_{ff}$ → better. $P_\text{drop}=0$ → 24.6.
- Learned PE instead of sinusoidal → 25.7 ≈ same.

*How do BERT & GPT relate?*
- BERT = encoder stack only, bidirectional.
- GPT = decoder stack w/o cross-attention, causal.
- Why decoder-only won for LLMs → [Cross Attention](attn.md#cross-attention).
```

&nbsp;

## Input
### Tokenization
- **What**: Text → subword IDs. Full topic: [Tokenization](token.md).
- **Why**: Embedding table needs a fixed, finite vocab. Subwords → rare words split into known pieces → word-level OOV⬇️.
    - Zero OOV needs full char/byte coverage or an UNK fallback.
- **How**:
    - EN-DE: [BPE](token.md#bpe), shared source-target vocab ≈37000.
    - EN-FR: [WordPiece](token.md#wordpiece), 32000.
    - Shared vocab → one embedding matrix for both languages → enables [Weight Tying](#weight-tying).

&nbsp;

### Token Embedding
- **What**: Row lookup in $E$, scaled by $\sqrt{d}$.
- **Why**: Discrete IDs → dense $d$-dim vectors that attention & FFN can process.
- **How**:
    1. Token ID $t$ → row $E_t$.
    2. Multiply by $\sqrt{d}$.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $t_i\in\{1,\dots,|\mathcal{V}|\}$: Token ID at position $i$.
    - $\mathbf{x}_i\in\mathbb{R}^d$: Embedding at position $i$.

Forward:

$$
\mathbf{x}_i=\sqrt{d}\,E_{t_i}=\sqrt{d}\,E^\top\mathbf{e}_{t_i}
$$

- $\mathbf{e}_{t}$: One-hot vector of $t$. Lookup = [Linear](../dl/foundation/module.md#linear) on a one-hot input, w/o the matmul.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, V, d):
        super().__init__()
        self.E = nn.Parameter(torch.randn(V, d) * d ** -0.5)  ## std d^-1/2, as in tensor2tensor
        self.d = d

    def forward(self, ids):
        return self.E[ids] * self.d ** 0.5                    ## lookup, then x sqrt(d)

    def logits(self, h):
        return h @ self.E.T                                   ## tied pre-softmax linear: same E

## Example
emb = TokenEmbedding(V=100, d=8)
ids = torch.tensor([[3, 14, 15]])
print(emb(ids).shape, emb.logits(emb(ids)).shape)  ## torch.Size([1, 3, 8]) torch.Size([1, 3, 100])
```
````

```{attention} Q&A
:class: dropdown
*Why multiply by $\sqrt{d}$?*
- The paper gives no reason.
- Reference code (tensor2tensor) inits $E$ with std $d^{-1/2}$ → $\times\sqrt{d}$ → unit-scale embeddings.
- → Same scale as PE entries in $[-1,1]$ → PE does not drown the token signal.
- With tying, $E$ is also the output layer → small entries keep initial logits small.
```

&nbsp;

#### Weight Tying
- **What**: One matrix $E$ for source embedding, target embedding & pre-softmax linear. {cite:p}`press2017using`
- **Why**: All 3 map between tokens & the same $d$-dim space → share it.
    - Params⬇️. LM perplexity⬇️ in Press & Wolf.
- **How**: Output logits $=\mathbf{h}E^\top$. Grads from all 3 uses update $E$.

```{attention} Q&A
:class: dropdown
*How much does it save in the base model?*
- 3 matrices → 1: $2\times37000\times512\approx37.9$M params saved.

*Requirement?*
- Source & target share one vocab → shared BPE.

*Do modern LLMs tie?*
- Small models often tie ← embedding = large share of params.
- Large models often untie ← embedding share⬇️ with scale. Input & output roles can then specialize.
```

&nbsp;

### Positional Encoding
- **What**: Fixed sinusoidal vector $\mathbf{p}_i$ added to each embedding. Full topic: [Positional Encoding](pos.md).
- **Why**: Attention is [permutation-equivariant](pos.md#permutation-equivariance) → w/o PE, word order is invisible.
- **How**: $\mathbf{x}_i\leftarrow\mathrm{Dropout}(\mathbf{x}_i+\mathbf{p}_i)$ at the bottom of both stacks. Formula: [Sinusoidal](pos.md#sinusoidal).
    - Learned PE gave nearly identical BLEU. Sinusoidal was chosen because it might extrapolate to longer sequences.

&nbsp;

## Sub-Layers
### Multi-Head Attention
- **What**: $h=8$ parallel scaled dot-product heads, $d_k=d_v=d/h=64$, concatenated → $W_O$. Mechanism: [Multi-Head Attention](attn.md#multi-head-attention).
- **Why**: One head = one weighted average per token → averages away distinct relations.
    - $h$ heads → $h$ relations at ≈ single-head cost, since $d_k$ shrinks by $h$.
- **How**: Used 3 ways.
    1. **Encoder self-attention**: $Q,K,V$ ← previous encoder layer. Every token sees all source tokens.
    2. **Decoder masked self-attention**: $Q,K,V$ ← previous decoder layer. [Causal mask](attn.md#causal) blocks future tokens.
    3. **Encoder-decoder attention**: $Q$ ← previous decoder sub-layer. $K,V$ ← encoder output $Z$. → [Cross Attention](attn.md#cross-attention).

```{note} Math
:class: dropdown
Notations:
- IO:
    - $H_q$: States that emit queries.
    - $H_{kv}$: States that emit keys & values.
- Params:
    - $W_Q^{(r)},W_K^{(r)}\in\mathbb{R}^{d\times d_k}$: Query/Key projection of head $r$.
    - $W_V^{(r)}\in\mathbb{R}^{d\times d_v}$: Value projection of head $r$.
    - $W_O\in\mathbb{R}^{hd_v\times d}$: Output projection.
- Misc:
    - $r\in\{1,\dots,h\}$: Head index.
    - $M$: Additive mask ($0$ = allowed, $-\infty$ = blocked).

Forward:

$$
\mathrm{MHA}(H_q,H_{kv})=\mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_h)\,W_O
$$

$$
\mathrm{head}_r=\mathrm{softmax}\left(\frac{(H_qW_Q^{(r)})(H_{kv}W_K^{(r)})^\top}{\sqrt{d_k}}+M\right)H_{kv}W_V^{(r)}
$$

| Use | $H_q$ | $H_{kv}$ | $M$ |
|:--|:--|:--|:--|
| Encoder self-attention | $H$ | $H$ | $0$ (padding only) |
| Decoder masked self-attention | $H$ | $H$ | Causal |
| Encoder-decoder attention | $H$ | $Z$ | $0$ (padding only) |
```

```{attention} Q&A
:class: dropdown
*Why divide by $\sqrt{d_k}$?*
- Components of $\mathbf{q},\mathbf{k}$ independent, mean 0, variance 1 → $\mathbf{q}\cdot\mathbf{k}$ has variance $d_k$ → softmax saturates → tiny grads. → [Derivation](attn.md#scaled-dot-product-attention)

*Why does every decoder layer attend to the same $Z$?*
- $Z$ = most refined source representation.
- $K,V$ of $Z$ per layer are computed once per source → reused at every decode step.

*What about padding?*
- Batches mix lengths → pad tokens must be blocked as keys. The paper does not discuss it. Implementations add a key-padding mask.
```

&nbsp;

### Position-wise FFN
- **What**: 2-layer MLP with ReLU, applied to each token independently. Same weights across positions, different weights per layer.
- **Why**: Attention = communication across tokens. FFN = computation within a token.
    - Attention output = weighted average of value vectors → no per-feature nonlinearity on the content.
- **How**:
    1. Expand $d\to d_{ff}=4d$.
    2. [ReLU](../dl/foundation/activation.md#relu).
    3. Project back $d_{ff}\to d$.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{h}\in\mathbb{R}^d$: One token's hidden state.
- Params:
    - $W_1\in\mathbb{R}^{d\times d_{ff}}$, $\mathbf{b}_1\in\mathbb{R}^{d_{ff}}$: Expand weights & bias.
    - $W_2\in\mathbb{R}^{d_{ff}\times d}$, $\mathbf{b}_2\in\mathbb{R}^{d}$: Compress weights & bias.

Forward:

$$
\mathrm{FFN}(\mathbf{h})=\max(0,\mathbf{h}W_1+\mathbf{b}_1)\,W_2+\mathbf{b}_2
$$
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class FFN(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.W1 = nn.Linear(d, d_ff)  ## expand
        self.W2 = nn.Linear(d_ff, d)  ## compress

    def forward(self, x):
        ## x: (B, m, d); acts on the last dim only -> tokens never interact here
        return self.W2(torch.relu(self.W1(x)))

## Example
print(FFN(8, 32)(torch.randn(2, 5, 8)).shape)  ## torch.Size([2, 5, 8])
```
````

```{attention} Q&A
:class: dropdown
*Why "position-wise"?*
- Same function at every position, no cross-token mixing. Equivalent to 2 convolutions with kernel size 1.

*Why $d_{ff}=4d$?*
- Empirical. Ablation: $d_{ff}$⬆️ → BLEU⬆️ (1024: 25.4, 2048: 25.8, 4096: 26.2 on EN-DE dev).

*Cons?*
- 2/3 of layer params & most layer FLOPs at short $m$.
```

&nbsp;

### Add & Norm
- **What**: $\mathrm{LayerNorm}(H+\mathrm{Dropout}(\mathrm{Sublayer}(H)))$ around every sub-layer (**Post-LN**).
- **Why**: Make a deep stack trainable.
    - [Residual](../dl/foundation/module.md#residual-connection) → identity path for grads.
    - [LayerNorm](../dl/foundation/module.md#layernorm) → stable scale per token, independent of batch & sequence length.
- **How**:
    1. Sub-layer output → dropout.
    2. Add the sub-layer input.
    3. LayerNorm over the $d$ features of each token.
    - All sub-layers & embeddings output $d$ dims → the residual add needs no projection.

```{note} Math
:class: dropdown
Forward:

$$
H\leftarrow\mathrm{LayerNorm}\big(H+\mathrm{Dropout}(\mathrm{Sublayer}(H))\big)
$$

Pre-LN, for comparison:

$$
H\leftarrow H+\mathrm{Dropout}\big(\mathrm{Sublayer}(\mathrm{LayerNorm}(H))\big)
$$
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class AddNorm(nn.Module):
    def __init__(self, d, p=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(p)

    def forward(self, x, sublayer):
        ## Post-LN: normalize AFTER the residual add
        return self.norm(x + self.drop(sublayer(x)))

## Example
## Reuses FFN from above
x = torch.randn(2, 5, 8)
print(AddNorm(8)(x, FFN(8, 32)).shape)  ## torch.Size([2, 5, 8])
```
````

```{attention} Q&A
:class: dropdown
*Post-LN vs Pre-LN?*
- Post-LN: LayerNorm sits on the residual path → every layer rescales the identity signal.
    - At init, expected grads of params near the output are large → large LR is unstable → needs [warmup](#lr-schedule). {cite:p}`xiong2020layer`
- Pre-LN: residual path is a clean sum → well-behaved grads at init → warmup can be removed.
- Modern LLMs use pre-norm. Pre-norm needs a final norm before the output layer.

*Why LayerNorm, not BatchNorm?*
- Sequences vary in length & are padded → batch statistics per position are noisy & leak padding.
- LayerNorm uses one token's own features → same at train & test, any batch size.
```

&nbsp;

## Encoder
- **What**: $N=6$ identical layers. Each layer: self-attention → FFN, each wrapped in Add & Norm.
- **Why**: Contextual representation of every source token. Bidirectional: each token sees the whole source.
- **How**: Per layer:
    1. Self-attention → gather context from all source tokens.
    2. FFN → process each token's gathered context.
    - Output of the top layer = memory $Z$.

```{note} Math
:class: dropdown
Forward (one layer):

$$\begin{align*}
H&\leftarrow\mathrm{LayerNorm}\big(H+\mathrm{Dropout}(\mathrm{MHA}(H,H))\big)\\
H&\leftarrow\mathrm{LayerNorm}\big(H+\mathrm{Dropout}(\mathrm{FFN}(H))\big)
\end{align*}$$

- Input to layer 1: $\mathrm{Dropout}(\sqrt{d}\,E_{x_i}+\mathbf{p}_i)$ per source position $i$.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

## Reuses FFN & AddNorm from above
class EncoderLayer(nn.Module):
    def __init__(self, d, h, d_ff, p=0.1):
        super().__init__()
        ## nn.MultiheadAttention = MHA from nlp/attn.md (built-in; dropout=0 on attention weights, as in the paper)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.ffn = FFN(d, d_ff)
        self.add_norm1, self.add_norm2 = AddNorm(d, p), AddNorm(d, p)

    def forward(self, x):
        ## no mask: every source token sees every other one
        x = self.add_norm1(x, lambda t: self.attn(t, t, t, need_weights=False)[0])
        return self.add_norm2(x, self.ffn)

## Example
print(EncoderLayer(32, 4, 64)(torch.randn(2, 7, 32)).shape)  ## torch.Size([2, 7, 32])
```
````

```{attention} Q&A
:class: dropdown
*Why attention before FFN?*
- Gather context first, then transform it per token. The paper gives no ablation of the order.

*What breaks if PE is removed from the encoder?*
- The encoder becomes permutation-equivariant → reordering the source only reorders $Z$'s rows.
- Cross-attention is invariant to reordering $Z$'s rows → the decoder sees "dog bites man" = "man bites dog".
- → Order must come from somewhere: PE, or a causal mask ([NoPE](pos.md#nope)). A bidirectional encoder has neither.
```

&nbsp;

## Decoder
- **What**: $N=6$ identical layers. Each layer: masked self-attention → encoder-decoder attention → FFN, each wrapped in Add & Norm.
- **Why**: Generate the target autoregressively, conditioned on the source.
    - Masked self-attention → reads target tokens generated so far.
    - Encoder-decoder attention → reads the source.
- **How**:
    1. Input: target shifted right by 1 → $(\langle\text{bos}\rangle,y_1,\dots,y_{m-1})$.
    2. Per layer: masked self-attention → encoder-decoder attention → FFN.
    3. Position $t$ outputs a prediction of $y_t$.

```{note} Math
:class: dropdown
Forward (one layer):

$$\begin{align*}
H&\leftarrow\mathrm{LayerNorm}\big(H+\mathrm{Dropout}(\mathrm{MHA}_\text{causal}(H,H))\big)\\
H&\leftarrow\mathrm{LayerNorm}\big(H+\mathrm{Dropout}(\mathrm{MHA}(H,Z))\big)\\
H&\leftarrow\mathrm{LayerNorm}\big(H+\mathrm{Dropout}(\mathrm{FFN}(H))\big)
\end{align*}$$
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

## Reuses FFN & AddNorm from above
class DecoderLayer(nn.Module):
    def __init__(self, d, h, d_ff, p=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.ffn = FFN(d, d_ff)
        self.add_norm = nn.ModuleList(AddNorm(d, p) for _ in range(3))

    def forward(self, x, z):
        m = x.shape[1]
        causal = torch.triu(torch.ones(m, m, dtype=torch.bool), diagonal=1)  ## True = blocked future key
        x = self.add_norm[0](x, lambda t: self.self_attn(t, t, t, attn_mask=causal, need_weights=False)[0])
        ## Q from the decoder, K & V from the encoder memory z
        x = self.add_norm[1](x, lambda t: self.cross_attn(t, z, z, need_weights=False)[0])
        return self.add_norm[2](x, self.ffn)

## Example: 5 target tokens attend over 7 source positions
print(DecoderLayer(32, 4, 64)(torch.randn(2, 5, 32), torch.randn(2, 7, 32)).shape)  ## torch.Size([2, 5, 32])
```
````

```{attention} Q&A
:class: dropdown
*Why both shift-right AND a causal mask?*
- Shift: input at position $t$ is $y_{t-1}$ → position $t$ predicts $y_t$.
- Mask: position $t$ sees inputs $\leq t$ only, i.e., $y_{<t}$ → it cannot see its own label or later tokens.
- Either alone leaks: no shift → input at $t$ is $y_t$ itself. No mask → $t$ sees $y_{t},\dots,y_{m-1}$.

*Why self-attention before encoder-decoder attention?*
- The paper gives no rationale.
- With this order, the cross-attention query already carries target context → a better-informed lookup into the source.

*Decoder-only LMs?*
- Remove encoder-decoder attention → 2 sub-layers per layer → GPT-style decoder.
```

&nbsp;

## Output
- **What**: Linear (tied to $E$) + softmax → distribution over $\mathcal{V}$.
- **Why**: Next-token prediction needs one probability per vocab entry.
- **How**:
    1. Top decoder state $\mathbf{h}_t$ → logits $\mathbf{h}_tE^\top\in\mathbb{R}^{|\mathcal{V}|}$.
    2. Softmax → $p(\cdot\mid y_{<t},x)$.

```{note} Math
:class: dropdown
Forward:

$$
p(y_t=v\mid y_{<t},x)=\mathrm{softmax}\big(\mathbf{h}_tE^\top\big)_v
$$

- $\mathbf{h}_t\in\mathbb{R}^d$: Top decoder state at position $t$.
- No $\sqrt{d}$ factor here. The paper scales only in the embedding layers.
```

&nbsp;

## Training
### Objective
- **What**: Token-level CE on the shifted target, all positions in 1 pass (**teacher forcing**), with [label smoothing](../ml/obj.md#label-smoothing) $\epsilon_{ls}=0.1$.
- **Why**: MLE of $p(y\mid x)=\prod_tp(y_t\mid y_{<t},x)$.
    - Teacher forcing + causal mask → $m$ training signals per forward pass.
- **How**:
    1. Batch sentence pairs by similar length. ≈25000 source + 25000 target tokens per batch.
    2. Decoder input $(\langle\text{bos}\rangle,y_1,\dots,y_{m-1})$ → targets $(y_1,\dots,y_m)$.
    3. CE against smoothed targets, summed over positions.

```{note} Math
:class: dropdown
Objective:

$$
\mathcal{L}=-\sum_{t=1}^{m}\sum_{v\in\mathcal{V}}\tilde{q}_{t,v}\log p(y_t=v\mid y_{<t},x),\qquad \tilde{q}_{t,v}=\begin{cases}1-\epsilon_{ls} & v=y_t\\ \frac{\epsilon_{ls}}{|\mathcal{V}|-1} & v\neq y_t\end{cases}
$$

- The paper does not state the form. This is the reference code (tensor2tensor) convention: $\epsilon_{ls}$ spread over the wrong tokens only.
- Szegedy form ($\epsilon_{ls}/|\mathcal{V}|$ over all tokens, [Label Smoothing](../ml/obj.md#label-smoothing)) differs negligibly at $|\mathcal{V}|\approx37000$.
```

```{attention} Q&A
:class: dropdown
*Label smoothing effect?*
- Model is less confident → perplexity⬆️, accuracy & BLEU⬆️.
- EN-DE dev: $\epsilon_{ls}=0$ → PPL 4.67, BLEU 25.3. $\epsilon_{ls}=0.1$ → PPL 4.92, BLEU 25.8.

*Cons of teacher forcing?*
- Train on gold prefixes, test on own prefixes → **exposure bias**. → [SFT](post/sft.md#sft)
```

&nbsp;

### LR Schedule
- **What**: Linear warmup for 4000 steps, then inverse-square-root decay, scaled by $d^{-1/2}$. Optimizer: [Adam](../dl/train/optim.md#adam-adaptive-moment-estimation).
- **Why**: Post-LN + Adam is unstable at large early LR → [warmup](../dl/train/schedule.md#warmup). Decay → stable convergence.
- **How**:
    1. Adam with $\beta_1=0.9$, $\beta_2=0.98$, $\epsilon=10^{-9}$.
    2. Steps $\leq4000$: LR rises linearly.
    3. Steps $>4000$: LR $\propto$ step$^{-1/2}$.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $T_w$: #Warmup steps ($4000$).
- Misc:
    - $\tau$: Step number, starting at 1.

Schedule:

$$
\eta_\tau=d^{-0.5}\cdot\min\left(\tau^{-0.5},\ \tau\cdot T_w^{-1.5}\right)
$$

- Peak at $\tau=T_w$: $\eta=(dT_w)^{-0.5}\approx7.0\times10^{-4}$ for $d=512$.
```

````{important} Code
:class: dropdown
```python
import torch

class WarmupInvSqrt:
    def __init__(self, d=512, warmup=4000):
        self.d, self.warmup = d, warmup

    def __call__(self, step):
        step = max(step, 1)  ## avoid 0^-0.5
        ## linear ramp until warmup, then 1/sqrt(step) decay; both branches meet at step = warmup
        return self.d ** -0.5 * min(step ** -0.5, step * self.warmup ** -1.5)

## Example
sched = WarmupInvSqrt()
print([round(sched(s), 6) for s in (1, 4000, 100000)])  ## [0.0, 0.000699, 0.00014]

## Usage: base lr = 1 so LambdaLR returns the absolute LR
model = torch.nn.Linear(4, 4)
opt = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
lr_sched = torch.optim.lr_scheduler.LambdaLR(opt, sched)
```
````

```{attention} Q&A
:class: dropdown
*Why scale by $d^{-0.5}$?*
- Wider model → smaller LR. Big model ($d=1024$) gets a peak $\sqrt{2}\times$ lower than base.

*Is warmup still needed with Pre-LN?*
- Much less. Pre-LN without warmup matches Post-LN with warmup, with less tuning. {cite:p}`xiong2020layer`
```

&nbsp;

### Residual Dropout
- **What**: [Dropout](../dl/foundation/module.md#dropout) with $P_\text{drop}=0.1$ (base) at 2 places.
- **Why**: Regularize. W/o it, EN-DE dev BLEU drops 25.8 → 24.6.
- **How**:
    1. On each sub-layer output, before the residual add & LayerNorm.
    2. On embedding + PE sums, at the bottom of both stacks.

&nbsp;

## Inference
### Autoregressive Decoding
- **What**: Encode the source once. Decode 1 token at a time, feeding outputs back as inputs.
- **Why**: Target is unknown at test time → no teacher forcing.
- **How**:
    1. $Z\leftarrow$ encoder($x$), once.
    2. Start with $\langle\text{bos}\rangle$.
    3. Each step: decoder($y_{<t}$, $Z$) → $p(y_t)$ → choose token → append.
    4. Stop at $\langle\text{eos}\rangle$ or at max length = source length + 50.
    - Paper: [Beam Search](infer/dec.md#beam-search), beam size 4, length penalty $\alpha=0.6$.
    - Production: cache past keys & values instead of recomputing the prefix → [KV Cache](infer/cache.md#kv-cache).

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\alpha$: Length penalty strength.

Beam score with GNMT length penalty {cite:p}`wu2016google`:

$$
\mathrm{score}(Y)=\frac{\log p(Y\mid x)}{\mathrm{lp}(Y)},\qquad \mathrm{lp}(Y)=\frac{(5+|Y|)^\alpha}{(5+1)^\alpha}
$$

- $|Y|$: Hypothesis length.
- $\alpha=0$ → raw log-prob → favors short outputs. $\alpha$⬆️ → longer outputs.
```

````{important} Code
:class: dropdown
```python
import torch

## Reuses Transformer from the Architecture section
class GreedyDecoder:
    def __init__(self, model, bos, eos, max_extra=50):
        self.model, self.bos, self.eos, self.max_extra = model, bos, eos, max_extra

    @torch.no_grad()
    def __call__(self, src):
        z = self.model.encode(src)                         ## encoder runs ONCE
        y = torch.full((src.shape[0], 1), self.bos)
        done = torch.zeros(src.shape[0], 1, dtype=torch.bool)
        for _ in range(src.shape[1] + self.max_extra):     ## max len = source len + 50
            ## no KV cache: the whole prefix is recomputed every step
            nxt = self.model.decode(y, z)[:, -1].argmax(-1, keepdim=True)
            nxt = nxt.masked_fill(done, self.eos)          ## finished rows keep emitting EOS
            y = torch.cat([y, nxt], dim=1)
            done |= nxt == self.eos
            if done.all():
                break
        return y

## Example
model = Transformer(V=100, d=32, h=4, d_ff=64, N=2).eval()
out = GreedyDecoder(model, bos=1, eos=2)(torch.randint(0, 100, (2, 7)))
print(out.shape[0], out.shape[1] <= 1 + 7 + 50)  ## 2 True
```
````

```{attention} Q&A
:class: dropdown
*Why is training parallel but inference sequential?*
- Training: all $y_{<t}$ are known → 1 pass for all $t$.
- Inference: $y_t$ depends on the model's own $y_{t-1}$ → $m$ sequential decoder passes.

*What is cached per step?*
- Decoder self-attention: $K,V$ of past target tokens, per layer.
- Encoder-decoder attention: $K,V$ of $Z$, per layer, computed once per source.
```

&nbsp;

### Checkpoint Averaging
- **What**: Average the weights of the last checkpoints → 1 model.
    - Base: last 5 checkpoints, saved at 10-min intervals. Big: last 20.
- **Why**: Smooths noise at the end of training → ensemble-like gain at single-model cost.
- **How**: Element-wise mean of each parameter tensor.

````{important} Code
:class: dropdown
```python
import torch

class CheckpointAverager:
    def __call__(self, state_dicts):
        ## element-wise mean of every tensor across checkpoints
        return {k: torch.stack([sd[k].float() for sd in state_dicts]).mean(0) for k in state_dicts[0]}

## Example
sds = [{"w": torch.full((2,), float(i))} for i in range(5)]
print(CheckpointAverager()(sds)["w"])  ## tensor([2., 2.])
```
````

&nbsp;
