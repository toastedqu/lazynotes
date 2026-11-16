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
# Objective
What an LLM minimizes during pretraining, & why the field converged on next-token prediction.

Notations:
- $x$: Token seq
- $x_i$: $i$-th token
- $x_{<i}$: Tokens before position $i$
- $\tilde{x}$: Corrupted input seq
- $L$: Seq length (#tokens)
- $\mathcal{V}$: Vocabulary
- $\mathcal{D}$: Corpus
- $\theta$: Model params
- $p_\theta$: Model distribution

Overrides of global notation: $x$ = 1 whole token seq; $i$ = token position, ❌sample idx; $m$ & $n$ = section-local symbols, ❌#samples & ❌#features.

&nbsp;

## CLM
- **Name**: Causal Language Modeling
- **What**: Predict each token from its left context.
- **Why**: Learn $p(x)$ from unlabeled text.
    - Chain rule factorizes $p(x)$ exactly → ❌variational bound, ❌partition function.
    - Each next token = free label.
- **How**:
    1. Causal mask → position $i$ attends to $x_{\leq i}$ only.
    2. Score all $L$ positions in 1 forward pass (**teacher forcing**).
    3. CE vs true next token at every position.

```{note} Math
:class: dropdown
Model:

$$
p_\theta(x)=\prod_{i=1}^{L}p_\theta(x_i\mid x_{<i})
$$

Objective:

$$
\mathcal{L}_\text{CLM}(\theta)=-\mathbb{E}_{x\sim\mathcal{D}}\left[\sum_{i=1}^{L}\log p_\theta(x_i\mid x_{<i})\right]
$$
```

```{tip} Derivation
:class: dropdown
*Why is CE forced, ❌a design choice?*

1. MLE over the corpus:

    $$
    \max_\theta\ \mathbb{E}_{x\sim p_\mathcal{D}}\left[\log p_\theta(x)\right]
    $$

2. Expand the forward KL from data to model:

    $$
    \text{KL}(p_\mathcal{D}\Vert p_\theta)=-H(p_\mathcal{D})+H(p_\mathcal{D},p_\theta)
    $$

3. $H(p_\mathcal{D})$ is $\theta$-free → min forward KL = min CE = MLE.
4. Substitute the chain rule → per-token sum in the Objective.
5. Exponentiate the per-token CE → **perplexity**:

    $$
    \text{PPL}=\exp\left(-\frac{1}{L}\sum_{i=1}^{L}\log p_\theta(x_i\mid x_{<i})\right)
    $$

6. Uniform $p_\theta$ over $k$ tokens → $\text{PPL}=k$ → PPL = effective branching factor.

→ Chain rule + MLE force CE. ❌Heuristic.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CLMLoss(nn.Module):
    def __init__(self, ignore_index=-100):
        super().__init__()
        self.ignore_index = ignore_index  ## padding / cross-doc targets

    def forward(self, logits, tokens):
        ## logits[:, i] predicts tokens[:, i+1]
        ## -> drop last logit (no target) & first token (never a target)
        logits, targets = logits[:, :-1], tokens[:, 1:]
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),  ## fp32: logsumexp is precision-sensitive
            targets.reshape(-1),
            ignore_index=self.ignore_index,
        )

## Example: uniform logits -> PPL = |V| exactly
B, L, V = 2, 5, 16
tokens = torch.randint(0, V, (B, L))
loss = CLMLoss()(torch.zeros(B, L, V), tokens)  ## zero logits -> softmax = 1/V
print(loss.shape)                               ## torch.Size([])
print(round(torch.exp(loss).item(), 2))         ## 16.0 -> PPL = branching factor
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- **Dense signal**: $L$ targets per seq, 1 forward pass.
- **Exact likelihood** → PPL comparable across models that share a tokenizer.
- Train task = inference task → ❌objective mismatch.
- ❌Extra machinery: ❌sentinels, ❌corruption schedule, ❌mode tokens.

*Cons?*
- Teacher forcing → model never conditions on its own mistakes → [exposure bias](../post/sft.md#sft).
- Left-only context → weaker features when the full input is known up front.
- Fixed left-to-right order → ❌conditioning on right context.
- 1 forward pass per generated token. Speculative decoding amortizes the cost. The left-to-right dependency chain stays.

*Why did CLM beat MLM & denoising at scale?*
- Signal density: MLM supervises ~15% of positions per pass, CLM ~100% → ~6× targets per unit compute.
- Generative: MLM defines no valid $p(x)$ → ❌sampling, ❌in-context learning, ❌PPL.
- Head-to-head at 5B+ params & 170B+ tokens: causal decoder + full LM objective → best zero-shot after pure pretraining. {cite:p}`wang2022what`
- ⚠️ Same study: non-causal + MLM is best after multitask finetuning. CLM's win = zero-shot after unsupervised pretraining only.
- Same study: a causal decoder adapts cheaply to non-causal → choosing CLM first costs little.

*Why does next-token prediction learn more than fluency?*
- Any task written as text = subtask of the loss. Predict the token after "the answer is" → compute the answer.
- Compression: CE⬇️ on a corpus w/ arithmetic, code, & arguments → internal structure that reproduces them.
- ⚠️ Learns only what the corpus makes predictable. ❌Behaviors absent from text. ❌Preference among contradictory continuations → [post-training](../post/sft.md) fills this gap.

*Loss on padding & doc boundaries?*
- Padding: ❌. `ignore_index` masks it out.
- Cross-doc positions → [Document Packing](#document-packing).
```

&nbsp;

## Denoising
- **What**: Corrupt the input → reconstruct the removed content.
- **Why**: A causal mask discards right-side context.
    - Classification, retrieval, tagging: full input known at inference → one-sided conditioning wastes evidence.

&nbsp;

### MLM
- **Name**: Masked Language Modeling {cite:p}`devlin2019bert`
- **What**: Predict randomly masked tokens from both-side context.
- **Why**: Deep bidirectional features.
    - Concat of L2R & R2L models → shallow bidirectionality, ❌joint conditioning at each layer.
    - Full self-attention + next-token target → each position sees its own answer.
    - → Remove the answer from the input, then predict it.
- **How**:
    1. Sample 15% of positions.
    2. Of those: 80% → `[MASK]`, 10% → random token, 10% → unchanged.
    3. CE on sampled positions only.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\mathcal{M}\subset\{1,\dots,L\}$: Sampled positions.

Objective:

$$
\mathcal{L}_\text{MLM}(\theta)=-\mathbb{E}_{x,\mathcal{M}}\left[\sum_{i\in\mathcal{M}}\log p_\theta(x_i\mid\tilde{x})\right]
$$

- $|\mathcal{M}|\approx0.15L$ → loss on ~15% of positions per forward pass.
```

````{important} Code
:class: dropdown
```python
import torch

class MLMMasker:
    def __init__(self, vocab_size, mask_id, rate=0.15, ignore_index=-100):
        self.V, self.mask_id, self.rate, self.ignore = vocab_size, mask_id, rate, ignore_index

    def __call__(self, tokens):
        sel = torch.rand(tokens.shape) < self.rate          ## positions that carry loss
        labels = tokens.masked_fill(~sel, self.ignore)      ## CE skips the other ~85%
        u = torch.rand(tokens.shape)
        x = tokens.clone()
        x[sel & (u < 0.8)] = self.mask_id                   ## 80% -> [MASK]
        rand = sel & (u >= 0.8) & (u < 0.9)                 ## 10% -> random token
        x[rand] = torch.randint(self.V, tokens.shape)[rand]
        ## last 10%: input unchanged, still predicted
        return x, labels

## Example
tokens = torch.randint(5, 100, (2, 8))
x, labels = MLMMasker(vocab_size=100, mask_id=0)(tokens)
print(x.shape, labels.shape)  ## torch.Size([2, 8]) torch.Size([2, 8])
## loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100)
```
````

```{attention} Q&A
:class: dropdown
*Why 80/10/10, ❌always `[MASK]`?*
- `[MASK]` never appears downstream → always masking → pretrain/finetune input mismatch.
- Encoder cannot tell which positions it must predict or which are random → keeps a contextual distribution for every token. BERT's stated advantage.
- 10% unchanged → biases the representation toward the observed word.

*Why 15%? Why do modern encoders use more?*
- 15% = untested assumption: more masking → too little context.
- Wrong: 40% beats 15% for BERT-large on GLUE & SQuAD. 80% masking keeps ~95% of finetuning performance. {cite:p}`wettig2023should`
- 2 opposing effects: rate⬆️ → corruption⬆️ (less context) & #predictions⬆️ (more signal).
- ModernBERT: 30% rate, ❌NSP. {cite:p}`warner2024smarter`

*Where do encoders still win?*
- Retrieval & reranking embeddings, classification, token tagging, cheap safety filters.
- 1 fixed-cost forward pass per doc, ❌autoregressive decoding → far cheaper than an LLM.
- Full input known up front → bidirectional features use all of it.
- Active, ❌legacy: ModernBERT trains on 2T tokens w/ 8192 native context.

*Why is MLM not a generative model?*
- Conditionals $p_\theta(x_i\mid\tilde{x})$ are not guaranteed compatible → in general, no joint $p(x)$ induces all of them.
- → ❌normalized joint, ❌ancestral sampler, ❌exact PPL. Gibbs-style sampling from BERT is possible. It is ❌native decoding rule.
- Fixed 15% rate → model never sees a mostly-masked input → cannot generate from nothing.
- [Diffusion LM](#diffusion-lm) fixes this.

*What is NSP? Is it still used?*
- **NSP** (Next Sentence Prediction): binary task. Does segment B follow segment A? 50% true next, 50% random.
- ❌Used. RoBERTa & ModernBERT drop it. ModernBERT: "noticeable overhead for no performance improvement".
- ← Random negatives come from other docs → topic mismatch alone solves the task → ❌coherence learned.
```

&nbsp;

#### RTD
- **Name**: Replaced Token Detection {cite:p}`clark2020electra`
- **What**: Classify each token as original or generator-replaced.
- **Why**: MLM wastes compute.
    - Forward pass over 100% of positions. Loss on ~15%.
    - Plausible replacements (❌random) → binary task stays hard.
- **How**:
    1. Small MLM generator samples tokens at masked positions.
    2. Discriminator reads the corrupted seq → ❌`[MASK]` in its input.
    3. Binary CE over all $L$ positions.
    4. Train jointly. Keep the discriminator. Discard the generator.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\theta_G$: Generator params.
    - $\theta_D$: Discriminator params.
- Hyperparams:
    - $\lambda$: Discriminator loss weight.
- Misc:
    - $x^\text{c}$: Corrupted seq (generator samples at masked positions).
    - $D(x^\text{c},i)$: Discriminator prob that token $i$ is original.

Objective:

$$
\min_{\theta_G,\theta_D}\sum_{x\in\mathcal{D}}\mathcal{L}_\text{MLM}(x,\theta_G)+\lambda\mathcal{L}_\text{Disc}(x,\theta_D)
$$

$$
\mathcal{L}_\text{Disc}=\mathbb{E}\left[\sum_{i=1}^{L}-\mathbf{1}[x_i^\text{c}=x_i]\log D(x^\text{c},i)-\mathbf{1}[x_i^\text{c}\neq x_i]\log\left(1-D(x^\text{c},i)\right)\right]
$$

- Sampled token = original token → labeled original.
- $\lambda=50$ ← binary CE ≪ $|\mathcal{V}|$-way CE → discriminator term needs upscaling.
- ❌Backprop through sampling → generator trained by MLE, ❌adversarially.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class RTD(nn.Module):
    def __init__(self, generator, discriminator, mask_id, lam=50.0):
        super().__init__()
        self.G, self.D, self.mask_id, self.lam = generator, discriminator, mask_id, lam

    def forward(self, tokens, masked):
        ## generator = plain MLM on the masked positions
        gen_logits = self.G(tokens.masked_fill(masked, self.mask_id))      ## (B,L,V)
        mlm = F.cross_entropy(gen_logits[masked], tokens[masked])
        with torch.no_grad():                                              ## ❌grad through sampling
            fill = torch.distributions.Categorical(logits=gen_logits).sample()
        corrupted = torch.where(masked, fill, tokens)
        is_orig = (corrupted == tokens).float()   ## correct guess counts as original
        ## discriminator: binary label at ALL L positions
        disc = F.binary_cross_entropy_with_logits(self.D(corrupted), is_orig)
        return mlm + self.lam * disc

## Example: tiny stand-in nets
V, d = 20, 8
G = nn.Sequential(nn.Embedding(V, d), nn.Linear(d, V))
D = nn.Sequential(nn.Embedding(V, d), nn.Linear(d, 1), nn.Flatten(-2))  ## (B,L)
tokens = torch.randint(1, V, (2, 6))
masked = torch.rand(2, 6) < 0.25
masked[:, 0] = True                       ## >=1 masked position per row
print(RTD(G, D, mask_id=0)(tokens, masked).shape)  ## torch.Size([])
```
````

```{attention} Q&A
:class: dropdown
*Why more sample-efficient than MLM?*
- Task covers all input tokens, ❌only the masked subset.
- → Beats BERT at equal model size, data, & compute. Gap is largest for small models.

*Why a small, non-adversarial generator?*
- Too strong a generator → replacements indistinguishable → discriminator task too hard.
- Best: generator 1/4–1/2 of discriminator size.
- Adversarial training needs grads through discrete sampling. The RL workaround did worse than MLE.

*Why did ELECTRA-Large mask 25%, ❌15%?*
- At 15%, the generator was too accurate → very few replaced tokens → little to detect.

*Why is RTD absent from generative LLMs?*
- Discriminator outputs 1 binary decision per position, ❌distribution over $\mathcal{V}$ → ❌sampling.
- → Representation learner only. ❌Decoding rule.

*Still relevant?*
- For encoders: DeBERTaV3 = DeBERTa + RTD + gradient-disentangled embedding sharing. Strong NLU encoder. {cite:p}`he2021debertav3`
- ModernBERT returned to plain MLM → RTD's advantage at modern scale is ❌settled.
```

&nbsp;

### Span Corruption
- **What**: Replace contiguous spans w/ unique sentinels → generate only the spans. {cite:p}`raffel2020exploring`
- **Why**: Single-token masks are too easy. Full reconstruction is too long.
    - Lone masked token → often recoverable from local syntax → weak signal.
    - Full reconstruction → decoder steps spent on uncorrupted tokens.
- **How**:
    1. Corrupt $r=15\%$ of tokens as spans w/ mean length $\mu=3$.
    2. Replace each span w/ a unique sentinel `<X>`, `<Y>`, ….
    3. Target = each sentinel + its span content, then 1 closing sentinel.
    4. Encoder-decoder: bidirectional encoder over $\tilde{x}$, causal decoder over the target.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $y$: Target seq (sentinel-tagged spans).
- Hyperparams:
    - $r$: Corruption rate.
    - $\mu$: Mean span length.
- Misc:
    - $q=\text{round}(rL)$: #corrupted tokens.
    - $n=\text{round}(q/\mu)\approx rL/\mu$: #spans.

Objective:

$$
\mathcal{L}_\text{Span}(\theta)=-\mathbb{E}_{x}\left[\sum_{t=1}^{|y|}\log p_\theta(y_t\mid\tilde{x},y_{<t})\right]
$$

- $|y|=q+n+1\approx rL+n+1\ll L$ → decoder cost ∝ corrupted tokens, ❌$L$.
```

```{note} Example
:class: dropdown
| | |
|:--|:--|
| Original | `Thank you for inviting me to your party last week` |
| Input $\tilde{x}$ | `Thank you <X> me to your party <Y> week` |
| Target $y$ | `<X> for inviting <Y> last <Z>` |
```

````{important} Code
:class: dropdown
```python
import numpy as np

class SpanCorruption:
    def __init__(self, rate=0.15, mean_span=3.0):
        self.rate, self.mean_span = rate, mean_span

    def _segments(self, total, n, rng):
        ## random split of `total` tokens into n non-empty segments
        cuts = np.sort(rng.choice(np.arange(1, total), n - 1, replace=False))
        return np.diff(np.concatenate([[0], cuts, [total]])).tolist()

    def __call__(self, tokens, rng):
        L = len(tokens)
        n_noise = max(1, round(L * self.rate))
        n_spans = max(1, round(n_noise / self.mean_span))
        noise = self._segments(n_noise, n_spans, rng)
        keep = self._segments(L - n_noise, n_spans, rng)
        inp, tgt, pos = [], [], 0
        ## alternate: kept segment -> corrupted span
        for k, (a, b) in enumerate(zip(keep, noise)):
            s = f"<X{k}>"                                ## unique sentinel per span
            inp += tokens[pos:pos + a] + [s]             ## whole span -> 1 sentinel
            tgt += [s] + tokens[pos + a:pos + a + b]     ## target = only the span
            pos += a + b
        tgt.append(f"<X{n_spans}>")                      ## closing sentinel
        return inp, tgt

## Example
toks = "Thank you for inviting me to your party last week".split()
inp, tgt = SpanCorruption(rate=0.3, mean_span=1.5)(toks, np.random.default_rng(0))
print(inp)  ## e.g. ['Thank', 'you', '<X0>', 'me', 'to', 'your', 'party', 'last', '<X1>']
print(tgt)  ## e.g. ['<X0>', 'for', 'inviting', '<X1>', 'week', '<X2>']
n = sum(t.startswith("<") for t in inp)
assert len(inp) + len(tgt) == len(toks) + 2 * n + 1  ## each sentinel twice + closing one
```
````

```{attention} Q&A
:class: dropdown
*Why spans, ❌independent tokens?*
- Independent masks → each recoverable from immediate context.
- Multi-token span → model must produce a coherent phrase & model dependencies inside the prediction.
- Shorter target: 1 sentinel per span, ❌per token.

*Why an autoregressive target if the input is bidirectional?*
- Spans must be internally coherent → generate them left-to-right.
- → Encoder-decoder: bidirectional where evidence is complete, causal where output is produced.

*What did T5's ablation conclude?*
- Denoising > plain LM for transfer to downstream tasks (pretrain → finetune).
- Encoder-decoder beat decoder-only & prefix-LM at matched cost, in a text-to-text finetuning setup.
- ⚠️ ❌Contradiction w/ CLM winning for LLMs. T5 measured finetuned transfer at ≤11B params, ❌zero-shot generation at frontier scale.

*Still used?*
- ❌Frontier LLMs. ✅Encoder-decoder models (T5/Flan-T5 derivatives) & inside [MoD](#mod).
- Lasting legacy: the parameterization $(\mu,r,n)$ → the axis that other denoisers move along.
```

&nbsp;

#### BART
- **Name**: Bidirectional and Auto-Regressive Transformers {cite:p}`lewis2020bart`
- **What**: Reconstruct the full original doc from arbitrarily noised input.
- **Why**: Decouple the noise from the architecture.
    - Bidirectional encoder (BERT-like) + autoregressive decoder (GPT-like) → any noising function fits.
    - Decoder generates full text left-to-right → pretraining matches generation finetuning.
- **How**:
    1. **Text infilling**: sample spans w/ length ~ Poisson($\lambda=3$). Replace each span w/ 1 `[MASK]`. 0-length span → inserts a `[MASK]`.
    2. **Sentence permutation**: shuffle all sentences of the doc.
    3. Encoder reads $\tilde{x}$. Decoder generates the full original $x$.
    4. CE on every target token.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $g$: Noising function, $\tilde{x}=g(x)$.

Objective:

$$
\mathcal{L}_\text{BART}(\theta)=-\mathbb{E}_{x}\left[\sum_{i=1}^{L}\log p_\theta(x_i\mid\tilde{x},x_{<i})\right]
$$

- Final config: text infilling + sentence permutation. 30% of tokens masked per doc.
```

```{attention} Q&A
:class: dropdown
*BART vs T5 span corruption?*
- Sentinel: 1 shared `[MASK]` per span vs unique sentinel per span → BART must also infer how many tokens are missing.
- Target: full doc ($L$ tokens) vs spans only (~$rL$ tokens) → BART costs more decoder compute.
- Same architecture family: encoder-decoder.

*Which noise worked best?*
- Text infilling: best or near-best on most tasks in the ablation.
- Document rotation & sentence shuffling alone: poor.
- Sentence permutation added a significant gain only on CNN/DM summarization.
- Token deletion beat token masking on generation tasks ← model must find which positions are missing.

*Where is BART used?*
- Summarization & other seq2seq finetuning. mBART = multilingual BART for translation.
- ❌Frontier LLMs ← decoder-only CLM scales more simply.
```

&nbsp;

### Prefix-LM
- **What**: Bidirectional attention over the prefix + causal attention over the continuation. {cite:p}`dong2019unified`
- **Why**: Prompt is fully known at inference → causal masking on it discards evidence.
    - Prompt tokens are never generated → mutual visibility leaks nothing.
    - 1 stack: ❌separate encoder, ✅generation.
- **How**:
    1. Split the seq at position $P$ → prefix $x_{\leq P}$, target $x_{>P}$.
    2. Block attention mask: prefix sees the full prefix. Target is causal.
    3. CE on target positions only.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $P$: Prefix length.
- Misc:
    - $A_{ij}$: 1 if position $i$ may attend to position $j$.

Attention mask:

$$
A_{ij}=\begin{cases}1,&j\leq P\ \ \text{or}\ \ j\leq i\\ 0,&\text{otherwise}\end{cases}
$$

- $i\leq P$ → sees the whole prefix, incl. later prefix positions → bidirectional.
- $i>P$ → sees the whole prefix + own left context → causal.

Objective:

$$
\mathcal{L}_\text{PrefixLM}(\theta)=-\mathbb{E}_{x}\left[\sum_{i=P+1}^{L}\log p_\theta(x_i\mid x_{<i};A)\right]
$$

- Conditioning set = [CLM](#clm)'s. Only the mask $A$ & the summation range differ.
- $L-P$ supervised positions vs $L$ for CLM.
```

````{important} Code
:class: dropdown
```python
import torch

class PrefixLMMask:
    def __call__(self, L, P):
        i = torch.arange(L)[:, None]  ## query position (0-indexed)
        j = torch.arange(L)[None, :]  ## key position
        ## True = may attend: any prefix key, or any key at/before self
        return (j < P) | (j <= i)

## Example: L=5, prefix = first 2 tokens
print(PrefixLMMask()(5, 2).int())
## tensor([[1, 1, 0, 0, 0],
##         [1, 1, 0, 0, 0],   <- prefix rows: see each other
##         [1, 1, 1, 0, 0],
##         [1, 1, 1, 1, 0],
##         [1, 1, 1, 1, 1]])  <- target rows: causal
## loss: only on targets x[P:]
```
````

```{attention} Q&A
:class: dropdown
*Why is almost every LLM causal, ❌prefix-LM?*
- **Signal loss**: ❌loss on prefix tokens → fewer targets per seq. Same density argument as against MLM.
- **Cache invalidation**: each prefix token's representation depends on the whole prefix → appending a token invalidates the prefix KV cache → ❌prompt caching, ❌cheap multi-turn.
- **Arbitrary split**: $P$ is fixed at training. No single choice matches all prompt lengths.
- → Causal attention makes prefix & continuation the same computation → simple caching & serving.

*So it is useless?*
- ❌. Right mask when the input is fixed & complete: multimodal prefixes, encoder-style prompts, seq2seq finetuning.
- Survives as UL2's S-denoiser.

*Prefix-LM vs encoder-decoder?*
- Same information flow. Different parameterization.
- Encoder-decoder: separate params & cross-attention → ~2× params at ~equal FLOPs.
- Prefix-LM: 1 shared stack, the mask does the work → fewer params, less separation.
```

&nbsp;

### MoD
- **Name**: Mixture-of-Denoisers {cite:p}`tay2022ul2`
- **What**: Per-example corruption config sampled from a fixed mixture, tagged w/ a mode token.
- **Why**:
    - *Why do we need it?* 1 objective → 1 capability profile.
        - Short spans → knowledge & understanding.
        - Long spans / high corruption → fluent long generation.
    - *Why does it work?* Span corruption, CLM, & prefix-LM = 1 objective at different $(\mu,r,n)$ → interleave w/o new machinery.
- **How**:
    1. Define `SpanCorrupt`$(\mu,r,n)$: mean span length, corruption rate, #spans.
    2. Draw each example's config from 7 denoisers in 3 groups: R / S / X.
    3. Prepend a **mode token** `[R]` / `[S]` / `[X]` that names the group.
    4. At finetune/inference: pick the mode token that fits the task → **mode switching**.

````{important} Code
:class: dropdown
```python
import numpy as np

## Reuses SpanCorruption from Span Corruption above
class MoD:
    def __init__(self):
        R = [SpanCorruption(0.15, 3), SpanCorruption(0.15, 8)]
        X = [SpanCorruption(0.5, 3), SpanCorruption(0.5, 8),
             SpanCorruption(0.15, 64), SpanCorruption(0.5, 64)]
        self.denoisers = [("[R]", d) for d in R] + [("[X]", d) for d in X] + [("[S]", None)]

    def __call__(self, tokens, rng):
        ## mixing weights here: uniform, for illustration only
        mode, d = self.denoisers[rng.integers(len(self.denoisers))]
        if d is None:
            ## S: prefix-LM split, target = last L/4 tokens
            P = len(tokens) - len(tokens) // 4
            return [mode] + tokens[:P], tokens[P:]
        inp, tgt = d(tokens, rng)
        return [mode] + inp, tgt  ## mode token tells the model which task it faces

## Example
toks = [f"w{i}" for i in range(512)]
inp, tgt = MoD()(toks, np.random.default_rng(0))
print(inp[0], len(inp), len(tgt))  ## mode token, input length, target length
```
````

````{dropdown} Table: The Mixture & What It Subsumes
| Denoiser | Config $(\mu,r,n)$ | Character |
|:--|:--|:--|
| **R** (regular) | $(3,0.15,n)\cup(8,0.15,n)$ | T5 span corruption → knowledge |
| **S** (sequential) | $(L/4,0.25,1)$, span ends at seq end | Prefix-LM → fluent continuation |
| **X** (extreme) | $(3,0.5,n)\cup(8,0.5,n)\cup(64,0.15,n)\cup(64,0.5,n)$ | Long / aggressive → long-form generation |

Special cases of the same parameterization:

| Objective | Config |
|:--|:--|
| [CLM](#clm) | $(\mu=L,\ r=1.0,\ n=1)$ |
| [Prefix-LM](#prefix-lm) | $(\mu=L-P,\ r=1-P/L,\ n=1)$, span reaches the end |
| [Span Corruption](#span-corruption) | $(\mu=3,\ r=0.15,\ n=rL/\mu)$ |

- $n$: #corrupted spans $=rL/\mu$.
- CLM is ❌in the mixture ← special case of prefix-LM, covered by S.
````

```{attention} Q&A
:class: dropdown
*Why the mode token?*
- The 7 configs demand contradictory behaviors from the same weights: fill a 3-token gap vs generate half the seq.
- ❌Tag → model averages them. ✅Tag → model conditions on the requested task.
- → At inference: `[S]` for open generation, `[R]`/`[X]` for infilling.

*Did it work?*
- On its own terms, yes: UL2 20B beat T5-XXL & GPT-like baselines across finetuned & in-context setups.
- Some denoisers are weak alone. T5's 50% corruption rate underperformed 15% (GLUE 83.28→81.27, SQuAD 80.88→79.80). The mixture makes them useful.

*Why is frontier pretraining ❌MoD?*
- CLM + more data + more params kept scaling.
- MoD adds complexity: corruption pipeline, sentinels, mode tokens, mixture weights.
- Density advantage shrinks when training is token-limited, ❌signal-limited.
- ⚠️ The idea survived. Modern pretraining mixes [FIM](#fim) & [MTP](#mtp) into CLM → a mixture of denoisers w/ different members.

*Biggest transferable lesson?*
- Architecture & objective = independent axes. People often conflate them ("BERT = bidirectional = MLM").
- Each of these objectives can run on each of these masks.
```

&nbsp;

## PLM
- **Name**: Permutation Language Modeling {cite:p}`yang2019xlnet`
- **What**: Autoregressive LM over a random factorization order.
- **Why**: Bidirectional context w/o MLM's flaws.
    - MLM: masked tokens predicted independently → ignores dependencies among them.
    - MLM: `[MASK]` absent at finetuning → pretrain/finetune mismatch.
    - CLM: exact product rule, but left context only.
    - → Average over all orders → each position sees context from both sides, in expectation. Product rule stays exact.
- **How**:
    1. Sample a permutation $\mathbf{z}$ of positions per seq.
    2. Keep input order & positional encodings. Implement $\mathbf{z}$ via the attention mask only.
    3. **Two-stream attention**: content stream sees $x_{z_t}$; query stream knows position $z_t$ but ❌its content.
    4. Predict only the last ~$1/K$ tokens of $\mathbf{z}$ (**partial prediction**).

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $K$: Partial-prediction ratio, $L/(L-c)\approx K$.
- Misc:
    - $\mathcal{Z}_L$: Set of all permutations of $\{1,\dots,L\}$.
    - $\mathbf{z}$: Factorization order.
    - $t$: Step idx in $\mathbf{z}$.
    - $z_t$: Position predicted at step $t$.
    - $\mathbf{z}_{<t}$: Positions before step $t$.
    - $c$: Cutoff step. Steps $>c$ are predicted.
    - $h_{z_t}$: Content-stream representation.
    - $g_{z_t}$: Query-stream representation.
    - $e(\cdot)$: Token embedding.

Objective:

$$
\max_\theta\ \mathbb{E}_{\mathbf{z}\sim\mathcal{Z}_L}\left[\sum_{t=c+1}^{L}\log p_\theta(x_{z_t}\mid x_{\mathbf{z}_{<t}})\right]
$$

- $c=0$ → full PLM objective.

Model (layer $l$):

$$
\begin{align*}
g_{z_t}^{(l)}&\leftarrow\text{Attn}\left(Q=g_{z_t}^{(l-1)},\ KV=h_{\mathbf{z}_{<t}}^{(l-1)}\right)\\
h_{z_t}^{(l)}&\leftarrow\text{Attn}\left(Q=h_{z_t}^{(l-1)},\ KV=h_{\mathbf{z}_{\leq t}}^{(l-1)}\right)
\end{align*}
$$

$$
p_\theta(x_{z_t}=v\mid x_{\mathbf{z}_{<t}})=\frac{\exp\left(e(v)^\top g_{z_t}\right)}{\sum_{v'\in\mathcal{V}}\exp\left(e(v')^\top g_{z_t}\right)}
$$

- $g^{(0)}$: trainable vector. $h^{(0)}$: token embedding.
- Prediction reads the query stream $g$ → knows the target position, ❌its content.
```

````{important} Code
:class: dropdown
```python
import torch

class PermutationMask:
    def __init__(self, L):
        self.L = L

    def __call__(self, gen=None):
        z = torch.randperm(self.L, generator=gen)   ## factorization order
        rank = torch.empty(self.L, dtype=torch.long)
        rank[z] = torch.arange(self.L)              ## rank[i] = step at which x_i is predicted
        ## mask[i, j] = True -> position i may attend to position j
        content = rank[None, :] <= rank[:, None]    ## h: sees x_{z<=t}, incl. itself
        query = rank[None, :] < rank[:, None]       ## g: sees x_{z<t} only -> ❌own content
        return z, content, query

## Example: input order untouched, only the masks change
z, content, query = PermutationMask(4)(torch.Generator().manual_seed(0))
print(z)            ## factorization order, e.g. tensor([2, 0, 3, 1])
print(query.int())  ## row z[0] is all 0: first in order -> sees nothing
```
````

```{attention} Q&A
:class: dropdown
*Why permute the factorization order, ❌the input?*
- Finetuning sees text in natural order → pretraining input must stay in natural order.
- Order lives only in the attention mask → ❌train/finetune input mismatch.

*Why two streams?*
- Standard parameterization $p(x_{z_t}\mid h(x_{\mathbf{z}_{<t}}))$ ignores which position is the target → same distribution for every target position.
- Query stream needs the target position $z_t$, ❌its content → else trivial.
- Later steps need the content of $x_{z_t}$ → content stream keeps it.
- Finetuning: drop the query stream. Content stream = normal Transformer.

*Why partial prediction?*
- Full permutation objective → hard optimization, slow convergence.
- Early steps of $\mathbf{z}$ have little context. Last steps have the most.
- → Predict only the last ~$1/K$ of $\mathbf{z}$. XLNet uses $K=6$ or $7$.

*PLM vs CLM vs MLM?*
- CLM = PLM w/ $\mathbf{z}$ = identity & $c=0$.
- MLM: masked tokens predicted independently. PLM: exact product rule among predicted tokens.
- PLM: ❌`[MASK]` tokens → ❌input mismatch.

*Why did it fade?*
- RoBERTa: BERT was undertrained. Tuned MLM matches or exceeds every model published after it. {cite:p}`liu2019roberta`
- 2 streams & partial prediction → complex for a gain that tuning recovers.
- Decoder-only CLM then took over generation.
```

&nbsp;

## FIM
- **Name**: Fill-in-the-Middle {cite:p}`bavarian2022efficient`
- **What**: Move a middle span to the end → causal model learns infilling.
- **Why**: A causal model can only append. Editing needs both sides.
    - Code completion, refactoring, & text editing condition on text after the cursor.
    - ❌Architecture change, ❌second model → must be a data transformation.
- **How**:
    1. Split a chunk into (prefix, middle, suffix) at 2 uniformly random character offsets.
    2. Reorder w/ sentinels → middle comes last.
    3. Train w/ the ordinary [CLM](#clm) loss. Objective unchanged.
    4. Apply to a fraction (**FIM rate**) of examples. Leave the rest left-to-right.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $p$: Prefix token seq.
    - $m$: Middle token seq.
    - $s$: Suffix token seq. $x=p\Vert m\Vert s$.
    - $z$: Reordered seq.

Formats (both trained, 50/50):

$$
\begin{align*}
\text{PSM}:&\quad \texttt{<PRE>}\ p\ \texttt{<SUF>}\ s\ \texttt{<MID>}\ m\ \texttt{<EOT>}\\
\text{SPM}:&\quad \texttt{<PRE>}\ \texttt{<SUF>}\ s\ \texttt{<MID>}\ p\ m\ \texttt{<EOT>}
\end{align*}
$$

Objective = plain CLM over the whole reordered seq $z$, ❌middle-only loss:

$$
\mathcal{L}_\text{FIM}(\theta)=-\mathbb{E}\left[\sum_{t=1}^{|z|}\log p_\theta(z_t\mid z_{<t})\right]
$$

- Every position stays supervised, as in [CLM](#clm). The middle gives the capability, ❌the loss restriction.
- `<EOT>` ends the middle → model learns when to stop infilling.
- ⚠️ SPM opens w/ `<PRE>` directly followed by `<SUF>` → ❌plain suffix-first reordering.
```

````{important} Code
:class: dropdown
```python
import numpy as np

PRE, SUF, MID, EOT = "<PRE>", "<SUF>", "<MID>", "<EOT>"

class FIM:
    def __init__(self, fim_rate=0.5, spm_rate=0.5):
        self.fim_rate, self.spm_rate = fim_rate, spm_rate

    def __call__(self, text, tok, rng):
        ## leave a fraction of chunks as plain left-to-right
        if rng.random() >= self.fim_rate:
            return tok(text)
        ## cut at CHARACTER offsets, ❌token boundaries
        ## -> a mid-token cursor ("foo_ba|r") stays in-distribution at inference
        i, j = sorted(rng.integers(0, len(text) + 1, size=2))
        ## tokenize each piece SEPARATELY, after the split
        p, m, s = tok(text[:i]), tok(text[i:j]), tok(text[j:])
        if rng.random() < self.spm_rate:
            ## SPM: <PRE> directly followed by <SUF>
            return tok(PRE) + tok(SUF) + s + tok(MID) + p + m + tok(EOT)
        ## PSM
        return tok(PRE) + p + tok(SUF) + s + tok(MID) + m + tok(EOT)

## Example (stand-in tokenizer: 1 token per char, each sentinel 1 token)
tok = lambda t: [t] if t.startswith("<") else list(t)
out = FIM(fim_rate=1.0, spm_rate=1.0)("abcdef", tok, np.random.default_rng(0))
print(out)
## SPM layout: ['<PRE>', '<SUF>', *suffix, '<MID>', *prefix, *middle, '<EOT>']
## invariants for ANY split, in either format:
assert [c for c in out if c.startswith("<")] == [PRE, SUF, MID, EOT]     ## sentinel order
assert sorted(c for c in out if not c.startswith("<")) == list("abcdef")  ## nothing lost
```
````

````{dropdown} Table: The 4 FIM Choices
| Choice | Options | Default | Why |
|:--|:--|:--|:--|
| Format | PSM / SPM | Both, 50/50 | Robust to either serialization at inference |
| Level | Document / Context | Context | Document-level FIM is chunked afterwards → split sentinels & broken examples |
| Split point | Token / Character | Character | Real cursors sit mid-token. Token-aligned training never sees that. |
| FIM rate | $[0,1]$ | 0.5 | L2R ability survives up to 0.9. Only 1.0 degrades it. |
````

```{attention} Q&A
:class: dropdown
*What is "FIM for free"?*
- FIM rate up to 0.9 → left-to-right ability unchanged, by PPL & sampling evals across scales.
- Rate 1.0 degrades L2R ← model never sees an untransformed doc. 0.5 = recommended default, ❌ceiling.
- → Infilling = capability added at ~0 cost. Paper's advice: train every autoregressive model w/ FIM by default.
- ← FIM reorders, ❌corrupts. No token is destroyed.

*Why does SPM exist if PSM is the natural order?*
- **Cache reuse**: SPM puts the suffix first. In an IDE, the suffix (code after the cursor) is stable. The prefix grows as the user types. → Suffix KV cache survives each keystroke.
- PSM: prefix first → each typed char invalidates the cache after it.
- Both trained ← serving stacks differ in serialization.

*Why split at the character level?*
- Token-boundary splits → model only sees prefixes that end cleanly.
- Real cursors sit mid-token (`foo_ba|r`) → out-of-distribution prefix → mispredicts.
- Character-level split, tokenized afterwards → ragged boundaries enter the training distribution.

*Why context-level over document-level?*
- Document-level: FIM first, then pack & chunk → a FIM doc can be cut in half → `<MID>` orphaned from `<PRE>`/`<SUF>`.
- → Model trains on examples whose promised context is absent.
- Context-level: transform the chunk actually fed in → every example well-formed.

*Where is it used?*
- Standard for code models. Cheap transformation, unchanged objective, enables cursor-position completion.
- ⚠️ FIM changes what the model conditions on, ❌what it optimizes. It is CLM on permuted data.
```

&nbsp;

## MTP
- **Name**: Multi-Token Prediction {cite:p}`gloeckle2024better`
- **What**: Predict the next $n$ tokens at each position.
- **Why**: Next-token target is myopic.
    - Position $i$ is only asked about $x_{i+1}$ → representation may encode only the next step.
    - A representation that must serve $x_{i+1},\dots,x_{i+n}$ must plan further ahead.
- **How**:
    1. Shared trunk → hidden state per position.
    2. Extra prediction heads/modules, each for a further future offset.
    3. Add their CE losses to the main loss.
    4. Inference: discard the extra heads, or reuse them as a draft model.

`````{note} Math
:class: dropdown
Notations:
- Params:
    - $M_k\in\mathbb{R}^{d\times2d}$: Projection at depth $k$.
    - $\text{TRM}_k$: Transformer block at depth $k$.
- Hyperparams:
    - $n$: #future tokens per position, incl. the next one (parallel).
    - $D$: #additional tokens per position (sequential).
    - $\lambda$: MTP loss weight.
- Misc:
    - $d$: Hidden dim.
    - $\mathbf{h}_i$: Trunk output at position $i$.
    - $p_\theta^{(k)}$: Head $k$'s distribution.
    - $\mathbf{h}_i^k$: Representation of position $i$ at depth $k$. $k=0$ → main model output.
    - $P_j^k$: Depth-$k$ predicted distribution for token $x_j$.
    - $[\cdot;\cdot]$: Concatenation.
- DeepSeek-V3 writes $t_i$ & $T$. Here: $x_i$ & $L$. Input = $x_1,\dots,x_L$. Targets run to $x_{L+1}$, the token after the input window.

````{tab-set}
```{tab-item} Parallel (Gloeckle)
$n$ independent heads on a shared trunk. All read the same $\mathbf{h}_i$:

$$
\mathcal{L}_\text{MTP}=-\mathbb{E}\left[\sum_{i}\sum_{k=1}^{n}\log p_\theta^{(k)}(x_{i+k}\mid x_{\leq i})\right]
$$

- $k=1$ = ordinary next-token loss.
- Heads are conditionally independent given $\mathbf{h}_i$ → ❌causal chain among the $n$ predictions.
```

```{tab-item} Sequential (DeepSeek-V3)
$D$ chained modules. Each keeps the full causal chain. {cite:p}`deepseekai2024deepseekv3`

$$
\mathbf{h}_i'^k=M_k\left[\text{RMSNorm}(\mathbf{h}_i^{k-1});\ \text{RMSNorm}(\text{Emb}(x_{i+k}))\right]
$$

$$
\mathbf{h}_{1:L-k}^k=\text{TRM}_k(\mathbf{h}_{1:L-k}'^k),\qquad P_{i+k+1}^k=\text{OutHead}(\mathbf{h}_i^k)
$$

$$
\mathcal{L}_\text{MTP}^k=-\frac{1}{L}\sum_{i=2+k}^{L+1}\log P_i^k[x_i],\qquad \mathcal{L}_\text{MTP}=\frac{\lambda}{D}\sum_{k=1}^{D}\mathcal{L}_\text{MTP}^k
$$

- $\text{Emb}$ & $\text{OutHead}$ are shared w/ the main model → new params per depth: $M_k$ + 1 block.
- DeepSeek-V3: $D=1$ (2 tokens total). $\lambda=0.3$ for the first 10T tokens, then $0.1$ for the last 4.8T.
```
````
`````

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MTPHeads(nn.Module):
    ## parallel variant; Gloeckle's heads are transformer layers + shared unembedding
    ## Linear here for brevity
    def __init__(self, d, V, n):
        super().__init__()
        self.heads = nn.ModuleList(nn.Linear(d, V) for _ in range(n))  ## head k -> x_{i+k}

    def forward(self, h, tokens):
        ## h: (B,L,d) trunk output; tokens: (B,L)
        L, loss = tokens.size(1), 0.0
        for k, head in enumerate(self.heads, start=1):
            logits = head(h[:, :L - k])  ## only positions w/ a target k steps ahead
            loss = loss + F.cross_entropy(logits.flatten(0, 1), tokens[:, k:].flatten())
        return loss  ## k=1 term = ordinary CLM loss

## Example
h, tokens = torch.randn(2, 6, 8), torch.randint(0, 16, (2, 6))
print(MTPHeads(d=8, V=16, n=3)(h, tokens).shape)  ## torch.Size([])
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- Denser signal: $n$ targets per position, ❌1.
- Extra heads = built-in draft model → speculative decoding w/o a separate network.

*Parallel vs sequential heads?*
- **Parallel**: all $n$ heads read the same state. Cheap. Head $k$ must guess $x_{i+k}$ w/o knowing $x_{i+1}$.
- **Sequential**: module $k$ reads the depth-$(k-1)$ state & the true token $x_{i+k}$ → causal chain kept. Cost: $D$ extra blocks in sequence.
- Sequential = better conditioned. Parallel = cheaper.

*Why does MTP help large models but ❌small ones?*
- Gain grows w/ model size. Small models: gain absent or negative.
- Likely reason: auxiliary task consumes capacity. Small model has none to spare. Large model has slack.
- ⚠️ Same shape as the FLAN small-model result: auxiliary objectives tax capacity before they help.

*How big are the gains?*
- Gloeckle 13B: solves 12% more problems on HumanEval & 17% more on MBPP vs next-token baseline. Relative, ❌percentage points.
- Gains are strongest on generative benchmarks, weaker on multiple-choice.
- 4-token prediction → up to 3× faster inference via self-speculative decoding.
- DeepSeek-V3: 2nd-token acceptance rate 85–90% → 1.8× TPS.

*Is the MTP head kept at inference?*
- Optional. DeepSeek-V3 discards it by default, or reuses it for speculative decoding.
- GLM-4.5 adds an MoE MTP layer for speculative decoding. {cite:p}`glm2025glm45`
- → Training benefit & decoding benefit are separable.

*Why ❌universal yet?*
- Qwen3 & Llama 3 reports do not mention it. Kimi K2's architecture table lists 0 MTP layers.
- Costs: extra params, 1 more loss weight to schedule, gains depend on scale & domain (largest on code).
- ⚠️ Adoption is growing: DeepSeek-V3, GLM-4.5, & Qwen3-Next ship it.
```

&nbsp;

## Diffusion LM
- **What**: MLM w/ random mask rate $t\sim U(0,1)$ & $1/t$ loss weight. {cite:p}`sahoo2024simple`
- **Why**:
    - *Why do we need it?* Autoregression fixes the decoding order & the decoding cost.
        - Left-to-right order is arbitrary. A token's best evidence is often on its right.
        - Ancestral decoding: 1 forward pass per token → latency ∝ $L$.
    - *Why does it work?* Random rate over $(0,1)$ → model trains on near-empty inputs too → can generate from scratch. Fixed 15% cannot.
- **How**:
    1. **Forward**: at time $t$, mask each token independently w/ prob $t$. $t=1$ → fully masked.
    2. Predict all masked tokens at once from the partially masked seq.
    3. **Reverse** (generation): start fully masked → iteratively unmask → remask low-confidence predictions.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $x_0$: Clean seq.
    - $x_t$: Partially masked seq at time $t$.
    - $x_t^i$: $i$-th token of $x_t$.
- Misc:
    - $t\sim U[0,1]$: Mask ratio (diffusion time).
    - $\text{M}$: Mask token.
- Override: subscript on $x$ = diffusion time, ❌position. Position = superscript $i$.

Objective:

$$
\mathcal{L}(\theta)=-\mathbb{E}_{t,x_0,x_t}\left[\frac{1}{t}\sum_{i=1}^{L}\mathbf{1}[x_t^i=\text{M}]\log p_\theta(x_0^i\mid x_t)\right]
$$

Upper bound on the NLL → principled generative objective:

$$
-\mathbb{E}_{p_\text{data}(x_0)}\left[\log p_\theta(x_0)\right]\leq\mathcal{L}(\theta)
$$

- $\frac{1}{t}$: weight from the variational bound. Small $t$ → few masked tokens → each gets more weight. Drop it → ❌bound.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedDiffusionLoss(nn.Module):
    def __init__(self, model, mask_id, eps=1e-3):
        super().__init__()
        self.model, self.mask_id, self.eps = model, mask_id, eps

    def forward(self, x0):
        B, L = x0.shape
        t = (1 - self.eps) * torch.rand(B, 1) + self.eps  ## t ~ U(eps, 1]: avoids 1/0
        masked = torch.rand(B, L) < t                     ## each token masked w/ prob t
        xt = x0.masked_fill(masked, self.mask_id)
        logits = self.model(xt)                           ## (B,L,V), bidirectional
        nll = F.cross_entropy(logits.transpose(1, 2), x0, reduction="none")  ## (B,L)
        ## loss only on masked tokens, reweighted by 1/t
        ## / L: constant rescale, as in LLaDA's reference code
        return (nll * masked / t).sum() / (B * L)

## Example: stand-in "bidirectional model"; mask_id = V (extra embedding row)
V, d = 10, 8
model = nn.Sequential(nn.Embedding(V + 1, d), nn.Linear(d, V))
print(MaskedDiffusionLoss(model, mask_id=V)(torch.randint(0, V, (2, 6))).shape)  ## torch.Size([])
```
````

```{attention} Q&A
:class: dropdown
*Diffusion LM vs BERT: what differs?*
- Architecture: nothing essential. Both predict masked tokens bidirectionally.
- **Rate**: BERT fixes 15%. Diffusion LM samples $t\sim U(0,1)$ & reweights by $1/t$.
- **Corruption**: LLaDA masks only (pure absorbing state). BERT's 80/10/10 also injects random & unchanged tokens.
- → Loss = variational bound on $\log p_\theta(x_0)$ → real generative model w/ in-context learning & instruction following.
- → A change to the corruption schedule turns a representation learner into a generator.

*Pros?*
- **Parallel decoding**: many tokens per forward pass → #steps = tunable latency/quality knob, ❌fixed at $L$.
- **Order-agnostic**: ❌left-to-right prior. Reversal poem completion: LLaDA-8B 45.6 vs GPT-4o 34.3.
- ⚠️ Forward poem completion: LLaDA 51.8 vs GPT-4o 82.7. The directional gap closes. Forward capability does not win.
- Bidirectional conditioning at every generation step.

*Cons?*
- ❌**KV cache**: each step rewrites the whole seq → AR serving stack does not transfer.
- Fixed-length canvas chosen up front. Semi-autoregressive / block sampling relaxes this. It does not remove it.
- Only an upper bound on likelihood → PPL ❌directly comparable to AR models.
- High compute per token unless #steps is low. Low #steps costs quality.

*Where does it stand?*
- LLaDA 8B ≈ LLaMA3 8B. Parity at 8B, ❌superiority, ❌frontier scale. {cite:p}`nie2025large`
- Compute gap: masked diffusion scales at a rate comparable to AR, but needs ~16× more pretraining compute to match AR on text generation. Then it samples 1.4× faster than a KV-cached AR model. {cite:p}`nie2024scaling`
- → Latency/throughput bought w/ training compute, ❌free win.
- Production systems (Mercury, Gemini Diffusion) market throughput, ❌capability.
- Viable paradigm. Unproven at frontier scale. Treat scaling claims as open.
- ⚠️ Absorbing-state discrete diffusion predates MDLM. MDLM & concurrent work gave this simple weighted-MLM form. LLaDA scaled it to 8B.

*Why does this matter beyond speed?*
- Decouples "language model" from "next-token predictor". In-context learning & instruction following survive w/o autoregression.
- → The objective's exact form is a weaker constraint than the field assumed.
```

&nbsp;

## Auxiliary Losses
- **What**: Extra loss terms that regularize numerics or routing, ❌language modeling.
- **Why**: At scale, the main loss alone does not keep training well-posed.
    - Unbounded logits + low precision → loss spikes & divergence → rollbacks.
    - Sparse routing has a degenerate optimum (few experts take all tokens). The main loss accepts it.

&nbsp;

### Z-Loss
- **What**: Penalty on the squared log of the softmax normalizer. {cite:p}`chowdhery2022palm`
- **Why**: Softmax is shift-invariant → nothing fixes the logit scale.
    - $\text{softmax}(z)=\text{softmax}(z+c)$ → CE ignores logit drift.
    - Large logits + bf16 (8 exponent bits, 7 mantissa bits) → roundoff in $\log\sum e^{z_j}$ → corrupted grads → loss spikes.
- **How**: Add the penalty to CE → $\log Z$ stays near 0 → logits stay in a precision-safe range.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $c_z$: Router z-loss coeff.
- Misc:
    - $z$: Logit vector.
    - $Z=\sum_{j}e^{z_j}$: Softmax normalizer.
    - $N$: #experts.
    - $T$: #tokens.
    - $r_t\in\mathbb{R}^N$: Router logits of token $t$.

Output z-loss:

$$
\mathcal{L}_z=10^{-4}\cdot\log^2 Z
$$

Router z-loss: same penalty on the gating logits. {cite:p}`zoph2022stmoe`

$$
\mathcal{L}_z^\text{router}=\frac{1}{T}\sum_{t=1}^{T}\left(\log\sum_{e=1}^{N}\exp(r_{t,e})\right)^2,\qquad c_z=0.001
$$

Total for a sparse model:

$$
\mathcal{L}_\text{tot}=\mathcal{L}_\text{CE}+\mathcal{L}_B+c_z\mathcal{L}_z^\text{router}
$$

- $\mathcal{L}_B$: balance loss, incl. its coeff $\alpha$ ([Router Balancing](#router-balancing)).
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ZLoss(nn.Module):
    def __init__(self, coef=1e-4):
        super().__init__()
        self.coef = coef

    def forward(self, logits):
        log_z = torch.logsumexp(logits.float(), dim=-1)  ## log Z per position
        return self.coef * log_z.pow(2).mean()

## Example: shift all logits by +30
logits, tgt = torch.randn(2, 5, 16), torch.randint(0, 16, (2, 5))
ce = lambda z: F.cross_entropy(z.flatten(0, 1), tgt.flatten())
print(torch.allclose(ce(logits), ce(logits + 30), atol=1e-4))  ## True: CE blind to the shift
print(ZLoss()(logits) < ZLoss()(logits + 30))                  ## tensor(True): z-loss is not
```
````

```{attention} Q&A
:class: dropdown
*Why $\log^2 Z$, ❌a norm penalty on the logits?*
- $\log Z$ = the quantity that overflows & loses precision in CE → penalize it directly.
- Square → 2-sided: pins $\log Z$ near 0, ❌just pushes it down.
- ~Free: softmax is shift-invariant → at $10^{-4}$ the CE optimum barely moves.

*Does it cost quality?*
- ❌. ST-MoE vs tighter update clipping: clipping stabilized training but hurt quality badly ($-4.21$ vs baseline $-1.76$).
- Router z-loss stabilized 3/3 runs w/ a slight quality gain ($-1.74$).
- → Stabilizer w/o a trade.

*Output z-loss vs router z-loss?*
- Same formula, different logits: vocabulary softmax vs expert gating softmax.
- Router is more fragile ← $N$ is small → a few large logits collapse routing to 1 expert.
- Attention logits also accept a z-loss. QK-Norm is the more common fix there.

*When do you need it?*
- Large scale + low precision. Small fp32 models rarely spike.
- Symptom: loss spike w/o a bad data batch behind it.
```

&nbsp;

### Router Balancing
- **What**: Push MoE token assignment toward uniform across experts. {cite:p}`fedus2021switch`
- **Why**: Balance is needed for quality & throughput. The main loss works against it.
    - **Routing collapse**: early winners get more tokens → train faster → win more.
    - Expert parallelism: experts sit on different devices → most-loaded device sets step time. Overflow tokens get dropped.
- **How**: Add a term minimized by uniform assignment, scaled by $\alpha$.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\alpha$: Balance coeff.
- Misc:
    - $N$: #experts.
    - $e$: Expert idx.
    - $T$: #tokens in batch.
    - $p_{t,e}$: Router softmax prob of token $t$ for expert $e$.
    - $f_e$: Fraction of tokens dispatched to expert $e$.
    - $P_e$: Fraction of router prob mass on expert $e$.

Objective:

$$
\mathcal{L}_B=\alpha N\sum_{e=1}^{N}f_eP_e
$$

$$
f_e=\frac{1}{T}\sum_{t=1}^{T}\mathbf{1}\left\{\arg\max_{e'}p_{t,e'}=e\right\},\qquad P_e=\frac{1}{T}\sum_{t=1}^{T}p_{t,e}
$$

- $f$ = argmax count → ❌differentiable. $P$ carries the grad.
- $\times N$ → loss scale constant in $N$ ← at uniform, $\sum_ef_eP_e=N\cdot\frac{1}{N^2}=\frac{1}{N}$.
- Switch: $\alpha=10^{-2}$.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwitchBalanceLoss(nn.Module):
    def __init__(self, n_experts, alpha=1e-2):
        super().__init__()
        self.N, self.alpha = n_experts, alpha

    def forward(self, router_logits):
        ## router_logits: (T, N)
        p = router_logits.softmax(dim=-1)
        f = F.one_hot(p.argmax(dim=-1), self.N).float().mean(dim=0)  ## dispatch fraction: ❌grad
        P = p.mean(dim=0)                                            ## prob mass: carries grad
        return self.alpha * self.N * (f * P).sum()

## Example: 8 tokens, 4 experts
balanced = 5 * torch.eye(4).repeat(2, 1)              ## each expert wins 2 tokens
collapsed = torch.tensor([[5., 0, 0, 0]]).repeat(8, 1)  ## expert 0 wins all
loss = SwitchBalanceLoss(4)
print(round(loss(balanced).item(), 4))   ## 0.01 = alpha at perfect balance
print(round(loss(collapsed).item(), 4))  ## 0.0392 -> ~4x
```
````

```{attention} Q&A
:class: dropdown
*Why is the auxiliary loss a problem?*
- It is a 2nd objective vs the LM loss. Its grad moves weights toward balance, ❌lower CE.
- $\alpha$ too small → routing collapses. Too large → measurable quality loss.
- → No $\alpha$ is free. The method is a tuned compromise.

*Did Switch invent balance losses?*
- ❌. Load-balancing losses predate Switch (Shazeer et al. 2017, GShard). Cited here for this exact formula & $\alpha$.
```

&nbsp;

#### Aux-Loss-Free Balancing
- **What**: Per-expert bias on the top-$K$ selection, updated by load. {cite:p}`wang2024auxiliary`
- **Why**: The auxiliary loss trades quality for balance.
    - Balance via an update rule, ❌a grad → ❌interference grads on the LM objective.
- **How**:
    1. Add bias $b_e$ to each expert's affinity, for top-$K$ selection only.
    2. Gate weight comes from the unbiased affinity.
    3. After each step: overloaded expert → $b_e$⬇️. Underloaded → $b_e$⬆️.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\gamma$: Bias update speed.
    - $K_r$: #routed experts per token.
- Misc:
    - $s_{e,t}$: Affinity of token $t$ to expert $e$.
    - $b_e$: Routing bias of expert $e$.
    - $g_{e,t}$: Gate weight of expert $e$ for token $t$.
    - $c_e$: #tokens assigned to expert $e$ in the step.
    - $\bar{c}$: Mean of $c_e$ over experts.

Routing:

$$
g'_{e,t}=\begin{cases}s_{e,t},&s_{e,t}+b_e\in\text{Topk}\left(\{s_{j,t}+b_j\}_{j=1}^{N},K_r\right)\\ 0,&\text{otherwise}\end{cases}\qquad g_{e,t}=\frac{g'_{e,t}}{\sum_{j=1}^{N}g'_{j,t}}
$$

Bias update (after each step):

$$
b_e\leftarrow b_e+\gamma\cdot\text{sign}(\bar{c}-c_e)
$$

- Overloaded ($c_e>\bar{c}$) → $b_e$⬇️. Underloaded → $b_e$⬆️. Balanced → unchanged.
- $b_e$ shifts selection only. $g_{e,t}$ uses $s_{e,t}$ → bias never enters the forward value of the gate.
- DeepSeek-V3: $\gamma=0.001$ for the first 14.3T tokens, then $0$ for the last 500B. {cite:p}`deepseekai2024deepseekv3`
- DeepSeek-V3 keeps a sequence-wise balance loss at $\alpha=10^{-4}$ against extreme within-seq imbalance.
```

````{important} Code
:class: dropdown
```python
import torch

class BiasRouter:
    def __init__(self, n_experts, k, gamma=1e-3):
        self.b = torch.zeros(n_experts)  ## ❌param: ❌grad, updated by rule
        self.k, self.gamma = k, gamma

    def __call__(self, s):
        ## s: (T, N) affinities (sigmoid scores in DeepSeek-V3)
        idx = (s + self.b).topk(self.k, dim=-1).indices               ## bias picks experts only
        g = torch.zeros_like(s).scatter(-1, idx, s.gather(-1, idx))
        return g / g.sum(dim=-1, keepdim=True), idx                   ## gate from s, ❌s+b

    def update(self, idx):
        load = torch.bincount(idx.flatten(), minlength=self.b.numel()).float()
        self.b -= self.gamma * torch.sign(load - load.mean())        ## overloaded -> b down

## Example
router = BiasRouter(n_experts=4, k=2)
g, idx = router(torch.rand(6, 4))
router.update(idx)
print(g.sum(dim=-1))  ## all 1: gates normalized over the 2 chosen experts
print(router.b)       ## each entry in {-0.001, 0, 0.001}
```
````

```{attention} Q&A
:class: dropdown
*Why better than the auxiliary loss?*
- Control loop: measure load → nudge bias → repeat. ❌Grad carries the balance objective.
- → Removes the quality/balance trade instead of tuning it.

*Why does the bias touch only top-$K$ selection?*
- Bias on the gate weight → rescales expert outputs → balance signal enters the forward value & its grads.
- Selection is discrete & already non-differentiable → bias re-routes tokens w/o rescaling any expert's output.
- ⚠️ Layer output still changes: different experts run. What is kept: ❌grad carries the balance objective.

*Why keep a sequence-wise balance loss?*
- Bias balances load over the batch. Batch-level balance can hide 1 seq whose tokens all route to 1 expert.
- Sequence-wise loss is stricter → kept at a tiny coeff ($10^{-4}$) as a safety net.

*Why set $\gamma=0$ at the end of training?*
- DeepSeek-V3 states the schedule, ❌the reason.
- Likely reason: routing is stable late in training → more nudging only perturbs a converging model. Same instinct as LR decay to 0.
```

&nbsp;

## Aggregation
- **What**: Reduction of per-token losses to 1 scalar.
- **Why**: Same per-token losses → different grads, depending on the reduction.

&nbsp;

### Loss Normalization
- **What**: Choice of denominator for summed per-token losses.
- **Why**: Token-level & sequence-level differ when lengths differ.
    - Sequence-level: each doc weighted equally → a token in a short doc gets more grad than one in a long doc.
    - Token-level: each token weighted equally.
- **How**: Sum all token losses over the full optimizer step → divide once by the total supervised token count.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $B$: #seqs in the step.
    - $\ell_{b,i}$: Loss at token $i$ of seq $b$.
    - $L_b$: #supervised tokens in seq $b$.

$$
\mathcal{L}_\text{token}=\frac{\sum_{b=1}^{B}\sum_{i=1}^{L_b}\ell_{b,i}}{\sum_{b=1}^{B}L_b}
\qquad\text{vs}\qquad
\mathcal{L}_\text{seq}=\frac{1}{B}\sum_{b=1}^{B}\frac{1}{L_b}\sum_{i=1}^{L_b}\ell_{b,i}
$$

- Equal only if all $L_b$ are equal.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenLevelLoss(nn.Module):
    def __init__(self, ignore_index=-100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, targets, n_tok_step):
        ## SUM over this micro-batch, divide by the token count of the WHOLE step
        ## -> backward() per micro-batch; grads add up to the large-batch grad
        s = F.cross_entropy(logits.flatten(0, 1), targets.flatten(),
                            ignore_index=self.ignore_index, reduction="sum")
        return s / n_tok_step

## Example: 2 micro-batches w/ unequal #supervised tokens (4 vs 1)
V = 8
mbs = [(torch.randn(1, 4, V), torch.tensor([[1, 2, 3, 4]])),
       (torch.randn(1, 4, V), torch.tensor([[5, -100, -100, -100]]))]
n_tok = sum((t != -100).sum() for _, t in mbs)  ## 5
ok = sum(TokenLevelLoss()(l, t, n_tok) for l, t in mbs)
full = F.cross_entropy(torch.cat([l for l, _ in mbs]).flatten(0, 1),
                       torch.cat([t for _, t in mbs]).flatten())  ## 1 big batch
bug = sum(F.cross_entropy(l.flatten(0, 1), t.flatten()) for l, t in mbs) / len(mbs)
print(torch.allclose(ok, full))   ## True
print(torch.allclose(bug, full))  ## False in general: mean of micro-batch means
```
````

```{attention} Q&A
:class: dropdown
*Why does this bite w/ gradient accumulation?*
- Correct: accumulate the sum of token losses & the token count. Divide once at the step boundary.
- Wrong: mean within each micro-batch, then mean the means → each micro-batch weighted equally. Matches neither formula above.
- → Grad accumulation stops being equivalent to a large batch. A well-publicized 2024 bug in several training stacks.

*Which is right for pretraining?*
- Token-level. Packing to a fixed context → constant $L_b$ → both coincide, until packing is imperfect or docs are masked apart.
- The choice matters for variable-length data: SFT, & RL w/ response lengths spanning orders of magnitude.
- Same axis on the RL side: DAPO uses token-level aggregation → long responses ❌down-weighted per token. {cite:p}`yu2025dapo`

*Is per-doc quality weighting done in the loss?*
- ❌Usually. The common lever is data-side: filtering, dedup, mixture reweighting/upsampling.
- Loss reweighting ≈ upsampling, in expectation. Upsampling fits the data pipeline, ❌the optimizer.
- → Treat "quality-weighted loss" claims w/ caution. Standard practice is upstream of the objective.
```

&nbsp;

### Document Packing
- **What**: Concatenate docs to fill a fixed context & mask cross-doc attention.
- **Why**: Padding to context length wastes compute.
    - Web docs are short & variable in length → padding can waste most FLOPs.
    - Naive packing → token attends to an unrelated preceding doc → spurious conditioning.
- **How**:
    1. Greedily concatenate docs up to the context length. Separate w/ `<eos>`.
    2. Apply a **block-diagonal** causal mask → each doc attends only within itself.
    3. Exclude cross-boundary targets from the loss.

````{important} Code
:class: dropdown
```python
import torch

class DocumentPacking:
    def __init__(self, eos_id, ignore_index=-100):
        self.eos_id, self.ignore = eos_id, ignore_index

    def __call__(self, docs):
        tokens = torch.cat([torch.cat([d, torch.tensor([self.eos_id])]) for d in docs])
        doc_ids = torch.cat([torch.full((len(d) + 1,), k) for k, d in enumerate(docs)])
        L = len(tokens)
        causal = torch.arange(L)[:, None] >= torch.arange(L)[None, :]
        mask = causal & (doc_ids[:, None] == doc_ids[None, :])  ## block-diagonal causal
        ## target of position i = token i+1; ignore it if it crosses into the next doc
        labels = tokens[1:].clone()
        labels[doc_ids[1:] != doc_ids[:-1]] = self.ignore
        return tokens, mask, labels

## Example: 2 docs, eos = 0
tokens, mask, labels = DocumentPacking(eos_id=0)([torch.tensor([5, 6, 7]), torch.tensor([8, 9])])
print(tokens.tolist())  ## [5, 6, 7, 0, 8, 9, 0]
print(labels.tolist())  ## [6, 7, 0, -100, 9, 0]: <eos> of doc 0 does not predict doc 1
print(mask[4].int())    ## tensor([0, 0, 0, 0, 1, 0, 0]): token 8 sees only itself
```
````

```{attention} Q&A
:class: dropdown
*How much does the intra-doc mask matter?*
- Llama 3: "limited impact during standard pre-training, but important in continued pre-training on very long sequences." {cite:p}`grattafiori2024llama`
- 8K context → few docs per packed seq → model learns to ignore text before `<eos>`.
- 128K context → many docs per seq → spurious long-range dependencies = what long-context training tries to teach.
- → Cheap insurance in general. Load-bearing for long context.
- ⚠️ Packing itself is old & universal (T5, PaLM, most stacks). Llama 3 citation = the doc-mask finding only.

*Why not avoid packing?*
- Padding to context length = direct FLOP loss. ModernBERT reports >99% packing efficiency w/ a greedy algorithm.
- Unpadding + packing = standard in serious training stacks.

*Does packing interact w/ the loss?*
- ✅. It changes the denominator in [loss normalization](#loss-normalization).
- Boundary target = token from another doc unless masked out.
```

&nbsp;

## Practice

### Recipe
- **What**: Typical modern frontier pretraining loss.
- **How**:
    1. [CLM](#clm) on packed, boundary-masked seqs. ~All of the signal.
    2. [FIM](#fim) on ~50% of code data. Data permutation, same loss.
    3. [MTP](#mtp) as an auxiliary head, $\lambda\approx0.1$–$0.3$, if the model is large enough.
    4. [Z-loss](#z-loss) at $10^{-4}$ for numerical stability.
    5. If MoE: [router balancing](#router-balancing). Trend: [bias-based](#aux-loss-free-balancing), ❌loss-based.
    6. Token-level [normalization](#loss-normalization) over the full step.

```{attention} Q&A
:class: dropdown
*What is deliberately absent?*
- MLM, span corruption, RTD, PLM, mode tokens, NSP: ❌in a frontier decoder-only recipe.
- Exception for small models: [distillation](../post/kd.md#token-level-kd). Gemma-scale models pretrain against a teacher's distribution, ❌hard labels.

*How large are the non-CLM terms?*
- Coeffs: z-loss $10^{-4}$, router balance $10^{-4}$–$10^{-2}$, MTP $\lambda\leq0.3$.
- ⚠️ A coeff ❌= fraction of total loss. The term's contribution also depends on its raw magnitude & aggregation.
- → Objective ~unchanged since GPT. Data, scale, & surrounding machinery changed.
```

&nbsp;

### Design Space
- **What**: Axes that define a pretraining objective.
- **How**:
    - **Corruption** $(\mu,r,n)$: what fraction, in what shapes.
    - **Visibility**: causal / prefix / bidirectional. Independent of corruption.
    - **Target set**: which positions carry loss → signal density.
    - **Horizon**: 1 future token or $n$.
    - **Order**: fixed left-to-right, permuted ([PLM](#plm)), or any order ([Diffusion LM](#diffusion-lm)).

```{attention} Q&A
:class: dropdown
*What has the field converged on?*
- CLM as the objective for every frontier decoder-only model.
- Density matters but is ❌sufficient. MLM & prefix-LM lost partly on density. MTP gains by adding targets. RTD supervises 100% of positions & still did not displace CLM.
- Cheap objective changes = data transformations. FIM won by changing the data, ❌the loss.
- Auxiliary terms stay tiny. A term large enough to compete w/ CE = quality regression.

*What is still contested?*
- **MTP**: real gains at scale. Qwen3, Kimi K2, & Llama 3 skip it. Public results do not cleanly separate training vs decoding benefit.
- **Diffusion LM**: validated at 8B, AR-comparable scaling rate, ~16× training-compute premium. Throughput claims run ahead of capability claims.
- **RTD for encoders**: DeBERTaV3 says yes. ModernBERT uses plain MLM. Unresolved.
- **Masking rate**: 15% ❌universally optimal. Best value depends on model size & masking strategy. ❌Settled rule.

*What is unsolved?*
- **Signal ceiling**: CE vs 1 observed continuation says nothing about other valid ones. The label = 1 sample from a distribution. The loss treats it as truth.
- **No credit assignment**: boilerplate token & crux-of-a-proof token count equally. The objective cannot tell them apart.
- **Proxy evaluation**: PPL tracks downstream ability until it does not. Data & scale confound every comparison here.

*What does this say about intelligence?*
- Objective ≈ trivial: predict the next symbol. Most capability came from scale & data → difficulty is ❌in the loss.
- Diffusion LM sharpens this: in-context learning & instruction following survive w/o autoregression → properties of compressing text, ❌of left-to-right factorization.
- → What matters: model the joint distribution well. The decomposition looks like an engineering choice.
```

&nbsp;
