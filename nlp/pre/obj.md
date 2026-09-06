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
What an LLM is actually trained to minimize during pretraining, and why the field converged on one answer.

Data curation, architecture, and scaling laws are elsewhere.

Notations:
- $x$: Token sequence
- $x_i$: $i$-th token
- $x_{<i}$: Tokens before position $i$
- $L$: Sequence length (#tokens)
- $\mathcal{V}$: Vocabulary
- $\mathcal{D}$: Corpus
- $\theta$: Model params
- $p_\theta$: Model distribution over token seqs

&nbsp;

## CLM
- **Name**: Causal Language Modeling
- **What**: Predict every token from its left context.
- **Why**: Raw text carries no labels, & the joint $p(x)$ must be learned from it alone.
    - Chain rule factorizes $p(x)$ **exactly** → ❌variational bound, ❌intractable partition function.
    - Every position is a free label → $L$ targets per sequence, all scored in 1 forward pass.
    - Training task = inference task → ❌objective mismatch to fix later.
- **How**:
    1. Causal mask → position $i$ attends to $x_{\leq i}$ only.
    2. 1 forward pass scores all $L$ positions in parallel (**teacher forcing**).
    3. CE against the true next token at every position.

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

Perplexity:

$$
\text{PPL}=\exp\left(-\frac{1}{L}\sum_{i=1}^{L}\log p_\theta(x_i\mid x_{<i})\right)
$$
```

```{tip} Derivation
:class: dropdown
*Why is CE the objective rather than a design choice?*

1. MLE over the corpus:

    $$
    \max_\theta\ \mathbb{E}_{x\sim p_\mathcal{D}}\left[\log p_\theta(x)\right]
    $$

2. Expand the **forward KL** from data to model:

    $$
    \text{KL}(p_\mathcal{D}\Vert p_\theta)=-H(p_\mathcal{D})+H(p_\mathcal{D},p_\theta)
    $$

3. $H(p_\mathcal{D})$ is $\theta$-free → minimizing forward KL = minimizing CE = MLE.
4. Substitute the chain-rule factorization → the per-token sum above.
5. → PPL $=\exp(\text{per-token CE})$ = **effective branching factor**: the size of the uniform vocabulary the model is equivalently confused by.

→ Nothing here is a heuristic. Given the chain rule & MLE, CE is forced.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn.functional as F

def clm_loss(logits, tokens, ignore=-100):
    ## logits (B,L,V) are predictions AT each position; token i's target is token i+1
    ## -> drop the last logit (nothing to predict) and the first token (never a target)
    logits, targets = logits[:, :-1], tokens[:, 1:]
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(),
        targets.reshape(-1),
        ignore_index=ignore,          ## padding / cross-document positions
    )

## Example: uniform logits -> PPL must equal |V| exactly
B, L, V = 2, 5, 16
tokens = torch.randint(0, V, (B, L))
loss = clm_loss(torch.zeros(B, L, V), tokens)   ## zero logits -> softmax is 1/V everywhere
print(loss.shape)                               ## torch.Size([])
print(round(torch.exp(loss).item(), 2))         ## 16.0 -> PPL = branching factor
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- **Dense signal**: $L$ supervised targets per sequence, at the cost of 1 forward pass.
- **Exact likelihood** → PPL is a real, comparable number.
- ❌Train/inference gap ← the model is trained on exactly what it does at decode time.
- ❌Extra machinery: no sentinels, no corruption schedule, no mode tokens.

*Cons?*
- **Teacher forcing**: conditioned on ground-truth prefixes, never on its own mistakes → [exposure bias](../post/sft.md#sft).
- **Left-only context** → weaker features for tasks where the full input is available up front.
- **Order is baked in**: the left-to-right factorization is arbitrary; the model cannot condition on the right.
- Sequential decoding → 1 forward pass per token by default. Speculative decoding amortizes the cost, but the left-to-right **dependency chain** remains.

*Why did CLM beat MLM & the denoising objectives at scale?*
- **Signal density**: MLM supervises ~15% of positions per forward pass, CLM 100% → ~6× the targets for the same compute.
- **It is a generative model.** MLM defines no valid $p(x)$ → ❌sampling, ❌in-context learning, ❌PPL.
- **Head-to-head at 5B+ params / 170B+ tokens**: causal decoder + full LM objective gives the strongest zero-shot generalization after pure pretraining. {cite:p}`wang2022what`
- ⚠️ The same study found non-causal + MLM best **after multitask finetuning** — CLM's win is specifically about zero-shot after unsupervised pretraining.
- Objectives are also cheaply convertible: a causal decoder can be adapted into a non-causal one, so picking CLM first costs little optionality.

*Why does predicting the next token produce anything more than fluency?*
- Any task expressible as text becomes a **subtask** of the loss: to predict the token after "the answer is", the model must compute the answer.
- Compression is the mechanism: driving CE down on a corpus that contains arithmetic, code, & argument forces internal structure that reproduces them.
- ⚠️ Only pressures what the corpus makes *predictable*. Behaviors that are never demonstrated in text are not learned, & the loss cannot prefer among contradictory continuations — the gap [post-training](../post/sft.md) fills.

*Is the loss on padding & document boundaries?*
- No — padded positions are masked out with `ignore_index`.
- Cross-document positions are a separate question → see [Document Packing](#document-packing).
```

&nbsp;

## Denoising
- **What**: Corrupt the input, train the model to reconstruct what was removed.
- **Why**: A causal mask discards the right half of the evidence.
    - For classification, retrieval, & tagging the **entire** input is available at inference → conditioning on one side only is a self-inflicted loss.
    - Price depends on the variant: BERT-style MLM buys bidirectional features but ❌a normalized joint; span corruption & prefix-LM keep a generative decoder.

&nbsp;

### MLM
- **Name**: Masked Language Modeling {cite:p}`devlin2019bert`
- **What**: Hide a random subset of tokens; predict them from both sides.
- **Why**: Deep bidirectional features cannot be trained causally.
    - Concatenating a left-to-right & a right-to-left model gives shallow bidirectionality, ❌joint conditioning at every layer.
    - Full self-attention + a next-token target would let each position **see its own answer**.
    - → Delete the answer from the input, then ask for it.
- **How**:
    1. Sample 15% of positions.
    2. Of those: 80% → `[MASK]`, 10% → random token, 10% → unchanged.
    3. CE on the sampled positions only.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\mathcal{M}\subset\{1,\dots,L\}$: Sampled (masked) positions.
    - $\tilde{x}$: Corrupted sequence.

Objective:

$$
\mathcal{L}_\text{MLM}(\theta)=-\mathbb{E}_{x,\mathcal{M}}\left[\sum_{i\in\mathcal{M}}\log p_\theta(x_i\mid\tilde{x})\right]
$$

Loss touches $|\mathcal{M}|\approx0.15L$ positions per forward pass, ❌all $L$.
```

```{attention} Q&A
:class: dropdown
*Why 80/10/10 rather than always `[MASK]`?*
- `[MASK]` never appears downstream → always masking creates a pretrain/finetune input mismatch.
- The encoder **cannot tell** which positions were sampled or which were replaced → it is forced to keep a contextual distribution over *every* token, ❌only the masked ones. That is BERT's stated advantage.
- **10% unchanged** additionally biases the representation toward the actually observed word.

*Why 15%, and why do modern encoders use more?*
- 15% was an assumption — more masking supposedly leaves too little context.
- It does not hold: 40% beats 15% for BERT-large on GLUE & SQuAD, and even **80% masking retains ~95%** of fine-tuning performance. {cite:p}`wettig2023should`
- Two opposing effects: ⬆️rate → ⬆️corruption (harder task, less context) but also ⬆️#predictions (more signal per pass).
- ModernBERT settles on **30%** and drops NSP entirely. {cite:p}`warner2024smarter`

*Where do encoders still win?*
- Retrieval & reranking embeddings, classification, token tagging, cheap guardrail/safety filters.
- 1 fixed-cost forward pass per document, ❌autoregressive decoding → orders of magnitude cheaper than an LLM at the same throughput.
- Bidirectional features are strictly more informed when the whole input is present up front.
- → Not a legacy niche: ModernBERT trains on 2T tokens with 8192 native context.

*Why is MLM not a generative model?*
- Its conditionals $p(x_i\mid\tilde{x})$ are **not guaranteed to be compatible** — in general no single joint $p(x)$ induces all of them.
- → ❌normalized joint, ❌ancestral sampler, ❌exact PPL. (MRF/Gibbs-style sampling from BERT *is* possible; it is just not a native decoding rule.)
- Fixed 15% rate → the model never sees a mostly-masked input → cannot start from nothing.
- Fixing exactly this is what [Diffusion LM](#diffusion-lm) does.

*NSP — dead or alive?*
- Dead. Dropped by RoBERTa & by ModernBERT ("noticeable overhead for no performance improvement").
- Diagnosis: NSP conflates topic prediction with coherence, and the negative is trivially detectable by topic mismatch alone.
```

&nbsp;

#### RTD
- **Name**: Replaced Token Detection {cite:p}`clark2020electra`
- **What**: A small generator fills the masks; the main model classifies **every** token as original or replaced.
- **Why**: MLM buys supervision on 15% of positions with a full forward pass over 100%.
    - The other 85% of the compute produces no gradient signal.
    - Replacing with *plausible* tokens (❌random) makes the binary task non-trivial.
- **How**:
    1. Small MLM generator samples replacements at masked positions.
    2. Discriminator reads the corrupted sequence — ❌`[MASK]` tokens ever appear.
    3. Binary CE over **all** $L$ positions.
    4. Train jointly; discard the generator, keep the discriminator.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\theta_G,\theta_D$: Generator / discriminator params.
- Hyperparams:
    - $\lambda$: Discriminator loss weight.

Objective:

$$
\min_{\theta_G,\theta_D}\sum_{x\in\mathcal{D}}\mathcal{L}_\text{MLM}(x,\theta_G)+\lambda\mathcal{L}_\text{Disc}(x,\theta_D)
$$

- $\lambda=50$ ← binary CE per token is far smaller than $|\mathcal{V}|$-way CE, so the discriminator term needs rescaling to matter.
- ELECTRA-Large raised the mask rate to **25%** ← at 15% the generator was too accurate, leaving too few replaced tokens to learn from.
- ❌Backprop through the sampling step → the generator is trained by MLE, ❌adversarially.
```

```{attention} Q&A
:class: dropdown
*Why is it more sample-efficient than MLM?*
- The task is defined over **all** input tokens, ❌only the masked subset.
- → Beats BERT at equal model size, data, & compute; the gap is largest for small models.

*Why is the generator small & non-adversarial?*
- Too strong a generator makes replacements indistinguishable → the discriminator task becomes unlearnable.
- Adversarial training would need gradients through discrete sampling; the RL workaround performed **worse** than plain MLE.

*Why is RTD absent from generative LLMs?*
- The discriminator emits a **binary** decision per position, ❌a distribution over $\mathcal{V}$ → ❌sampling, ❌generation.
- It is a representation learner. There is no decoding rule to attach.

*Still relevant?*
- Yes for encoders: DeBERTaV3 = DeBERTa + RTD + gradient-disentangled embedding sharing, still a top-tier NLU encoder. {cite:p}`he2021debertav3`
- But ModernBERT deliberately returned to plain MLM → RTD's advantage at modern scale & data quality is **not settled**.
```

&nbsp;

### Span Corruption
- **What**: Mask contiguous spans, collapse each to 1 sentinel, generate only the spans. {cite:p}`raffel2020exploring`
- **Why**: Single-token masking is too easy & the target is too long.
    - A lone masked token is often recoverable from local syntax → weak signal.
    - Reconstructing the *whole* sequence wastes decoder steps on tokens that were never corrupted.
- **How**:
    1. Corrupt $r=15\%$ of tokens as spans of mean length $\mu=3$.
    2. Replace each span in the input w/ a **unique** sentinel `<X>`, `<Y>`, ....
    3. Target = the sentinels followed by their contents.
    4. Encoder-decoder: bidirectional over the corrupted input, causal over the target.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\tilde{x}$: Input w/ spans replaced by sentinels.
    - $y$: Target — concatenated sentinel-tagged spans.

Objective:

$$
\mathcal{L}_\text{Span}(\theta)=-\mathbb{E}_{x}\left[\sum_{t=1}^{|y|}\log p_\theta(y_t\mid\tilde{x},y_{<t})\right]
$$

Example ($r=0.15$, $\mu=3$):

| | |
|:--|:--|
| Original | `Thank you for inviting me to your party last week` |
| Input $\tilde{x}$ | `Thank you <X> me to your party <Y> week` |
| Target $y$ | `<X> for inviting <Y> last <Z>` |

Target length $\approx rL+\#\text{spans}\ll L$ → decoder cost scales w/ what was corrupted, ❌the sequence.
```

```{attention} Q&A
:class: dropdown
*Why spans instead of independent tokens?*
- Independent masks are recoverable one-by-one from immediate context.
- A multi-token span forces the model to produce a **coherent phrase** & to model dependencies *within* the prediction.
- Also shortens the target: 1 sentinel per span, ❌per token.

*Why is the target autoregressive if the input is bidirectional?*
- The spans must be internally coherent → generating them left-to-right is the correct factorization.
- → Encoder-decoder: bidirectional where the evidence is complete, causal where output is produced.

*What did T5's ablation actually conclude?*
- Denoising > plain LM for **transfer to downstream tasks** (the regime studied: pretrain then fine-tune).
- Encoder-decoder beat decoder-only & prefix-LM at matched cost, in a text-to-text fine-tuning setup.
- ⚠️ This does **not** contradict CLM winning for LLMs — T5 measured fine-tuned transfer at ≤11B params, not zero-shot generation at frontier scale.

*Is span corruption still used?*
- Not for frontier LLMs. It survives in encoder-decoder models (T5/Flan-T5 derivatives) and inside [MoD](#mod).
- Its real legacy is the **parameterization** $(\mu,r,n)$ — the axis every other denoiser moves along.
```

&nbsp;

### Prefix-LM
- **What**: Bidirectional attention over the prefix, causal attention over the continuation. {cite:p}`dong2019unified`
- **Why**: At inference the prompt is fully known, so making it causal discards evidence for free.
    - Prompt tokens are never generated → nothing is leaked by letting them see each other.
    - Keeps a single stack: ❌separate encoder, ✅generation.
- **How**:
    1. Split the sequence at position $P$ → prefix $x_{\leq P}$, target $x_{>P}$.
    2. Block attention mask: prefix = fully visible to itself; target = causal.
    3. CE on the **target** positions only.

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

- $i\leq P$ → sees the whole prefix, incl. positions after itself → **bidirectional**.
- $i>P$ → sees the whole prefix + its own left context → **causal**.

Objective:

$$
\mathcal{L}_\text{PrefixLM}(\theta)=-\mathbb{E}_{x}\left[\sum_{i=P+1}^{L}\log p_\theta(x_i\mid x_{<i};A)\right]
$$

- The conditioning **set** matches [CLM](#clm); only the mask $A$ and the summation range differ.
- Only $L-P$ positions are supervised, vs $L$ for CLM.
```

```{attention} Q&A
:class: dropdown
*Why is almost every LLM causal instead of prefix-LM?*
- **Signal loss**: no loss on prefix tokens → fewer targets per sequence, exactly the density argument that sank MLM.
- **Cache invalidation**: bidirectional prefix means every prefix token's representation depends on the whole prefix → appending a token invalidates the entire prefix KV cache → ❌prompt caching, ❌cheap multi-turn.
- **Arbitrary split**: $P$ must be chosen at training time, and no choice matches all downstream prompt lengths.
- → Causal attention makes prefix and continuation the *same* computation; that uniformity is what makes caching and serving simple.

*So it is useless?*
- No. It is the right mask whenever the input is fixed and complete: multimodal prefixes (image tokens have no natural order), encoder-style prompts, seq2seq finetuning.
- It also survives as UL2's **S-denoiser**.

*Prefix-LM vs encoder-decoder?*
- Same information flow; different parameterization.
- Encoder-decoder: separate params & cross-attention → 2× params at ~equal FLOPs.
- Prefix-LM: 1 shared stack, mask does the work → cheaper, less expressive separation.
```

&nbsp;

### MoD
- **Name**: Mixture-of-Denoisers {cite:p}`tay2022ul2`
- **What**: Sample the corruption config per example from a fixed mixture, tagged w/ a mode token.
- **Why**: One objective buys one capability profile.
    - Short spans → knowledge & understanding; long spans / high corruption → fluent long generation.
    - A model trained on either alone is Pareto-dominated on the other.
    - *Why does mixing work at all?* Because span corruption, CLM, and prefix-LM are the **same objective** at different $(\mu,r,n)$ — so they can be interleaved without changing the machinery.
- **How**:
    1. Define `SpanCorrupt`$(\mu,r,n)$: mean span length, corruption rate, #spans.
    2. Draw each example's config from 7 denoisers grouped into R / S / X.
    3. Prepend a **mode token** `[R]` / `[S]` / `[X]` naming the config.
    4. At finetune/inference, pick the mode token matching the task → **mode switching**.

````{dropdown} Table: The Mixture, and What It Subsumes
| Denoiser | Config $(\mu,r,n)$ | Character |
|:--|:--|:--|
| **R** (regular) | $(3,0.15,n)\cup(8,0.15,n)$ | T5 span corruption → knowledge |
| **S** (sequential) | $(L/4,0.25,1)$, span ends at sequence end | Prefix-LM → fluent continuation |
| **X** (extreme) | $(3,0.5,n)\cup(8,0.5,n)\cup(64,0.15,n)\cup(64,0.5,n)$ | Long / aggressive → long-form generation |

Special cases of the same parameterization:

| Objective | Config |
|:--|:--|
| [CLM](#clm) | $(\mu=L,\ r=1.0,\ n=1)$ |
| [Prefix-LM](#prefix-lm) | $(\mu=L-P,\ r=1-P/L,\ n=1)$, span reaches the end |
| [Span Corruption](#span-corruption) | $(\mu=3,\ r=0.15,\ n=rL/\mu)$ |

- $n$: #corrupted spans $=rL/\mu$ — corrupt $rL$ tokens in spans averaging $\mu$ long.
- CLM is omitted from the mixture ← it is a special case of prefix-LM, already covered by S.
````

```{attention} Q&A
:class: dropdown
*Why the mode token?*
- The 7 configs demand contradictory behaviors from the same weights (fill a 3-token gap vs generate half the sequence).
- Without a tag the model must average them; with a tag it **conditions** on which is being asked.
- → At inference the token selects a behavior mode: `[S]` for open generation, `[R]`/`[X]` for infilling.

*Did it work?*
- Yes on its own terms: UL2 20B beat T5-XXL and GPT-like baselines across finetuned + in-context setups on the Pareto frontier.
- Some denoisers are weak **alone** — UL2 notes T5 tried a 50% corruption rate and found it did not work well (T5's own numbers: GLUE 83.28→81.27, SQuAD 80.88→79.80). The mixture is what makes them useful.

*Then why is frontier pretraining not MoD?*
- CLM + more data + more params kept scaling, and MoD adds real complexity: a corruption pipeline, sentinels, mode tokens, and a mixture to tune.
- The objective's *density* advantage shrinks once you are token-limited rather than signal-limited.
- ⚠️ The **idea** won even though the recipe did not: modern pretraining mixes [FIM](#fim) and [MTP](#mtp) into CLM. That is a mixture of denoisers with a different membership.

*Biggest transferable lesson?*
- Architecture and objective are **independent** axes, routinely conflated ("BERT = bidirectional = MLM").
- Any of these objectives can run on any of these masks.
```

&nbsp;

## FIM
- **Name**: Fill-in-the-Middle {cite:p}`bavarian2022efficient`
- **What**: Move a middle span to the end so a causal model learns to infill it.
- **Why**: A causal model can only append, but editing needs both sides.
    - Code completion, refactoring, and text editing all condition on what comes **after** the cursor.
    - ❌Architecture change, ❌second model → it must be a *data* transformation.
- **How**:
    1. Split a chunk into (prefix, middle, suffix) at 2 uniformly random **character** offsets.
    2. Reorder w/ sentinels so the middle lands **last**.
    3. Train w/ the ordinary [CLM](#clm) loss — the objective is untouched.
    4. Apply to a fraction (**FIM rate** 0.5) of examples; leave the rest left-to-right.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $p,m,s$: Prefix / middle / suffix token seqs, $x=p\Vert m\Vert s$.

Formats (both trained, 50/50):

$$
\begin{align*}
\text{PSM}:&\quad \texttt{<PRE>}\ p\ \texttt{<SUF>}\ s\ \texttt{<MID>}\ m\ \texttt{<EOT>}\\
\text{SPM}:&\quad \texttt{<PRE>}\ \texttt{<SUF>}\ s\ \texttt{<MID>}\ p\ m\ \texttt{<EOT>}
\end{align*}
$$

Objective = **plain CLM over the whole reordered sequence** $z$, ❌a middle-only loss:

$$
\mathcal{L}_\text{FIM}(\theta)=-\mathbb{E}\left[\sum_{t=1}^{|z|}\log p_\theta(z_t\mid z_{<t})\right]
$$

- $z$ = the permuted sequence above → every position stays supervised, exactly as in [CLM](#clm). The middle is where the *capability* comes from, ❌where the loss is restricted.
- `<EOT>` terminates the middle → the model learns **when to stop infilling** rather than running to the context end.
- ⚠️ SPM opens w/ `<PRE>` immediately followed by `<SUF>` — it is *not* a plain suffix-first reordering.
```

````{dropdown} Table: The 4 FIM Choices
| Choice | Options | Default | Why |
|:--|:--|:--|:--|
| Format | PSM / SPM | Both, 50/50 | Robustness to either serialization at inference |
| Level | Document / Context | **Context** | Document-level FIM is chunked afterwards → split sentinels & broken examples |
| Split point | Token / **Character** | Character | The cursor lands mid-token in a real IDE; token-aligned training never sees that |
| FIM rate | $[0,1]$ | **0.5** | L2R survives up to 0.9; only rate 1.0 degrades it |
````

````{important} Code
:class: dropdown
```python
import numpy as np

PRE, SUF, MID, EOT = "<PRE>", "<SUF>", "<MID>", "<EOT>"

def fim_permute(text, tok, rng, fim_rate=0.5, spm_rate=0.5):
    ## leave a fraction of chunks as plain left-to-right
    if rng.random() >= fim_rate:
        return tok(text)
    ## cut at CHARACTER offsets, never token boundaries
    ## -> a mid-token cursor ("foo_ba|r") stays in-distribution at inference
    i, j = sorted(rng.integers(0, len(text) + 1, size=2))
    ## tokenize each piece SEPARATELY, i.e. after the split
    p, m, s = tok(text[:i]), tok(text[i:j]), tok(text[j:])
    if rng.random() < spm_rate:
        ## SPM: leading <PRE> immediately followed by <SUF> -- shares its opening with PSM
        return tok(PRE) + tok(SUF) + s + tok(MID) + p + m + tok(EOT)
    ## PSM
    return tok(PRE) + p + tok(SUF) + s + tok(MID) + m + tok(EOT)

## Example (stand-in tokenizer: 1 token per char, each sentinel 1 token)
tok = lambda t: [t] if t.startswith("<") else list(t)
out = fim_permute("abcdef", tok, np.random.default_rng(0), fim_rate=1.0, spm_rate=1.0)
print(out)
## SPM layout, e.g. ['<PRE>','<SUF>','e','f','<MID>','a','b','c','d','<EOT>']
## invariants that hold for ANY split, in either format:
assert [c for c in out if c.startswith("<")] == [PRE, SUF, MID, EOT]   ## sentinel order
assert sorted(c for c in out if not c.startswith("<")) == list("abcdef")  ## nothing lost
```
````

```{attention} Q&A
:class: dropdown
*What is "FIM for free"?*
- FIM-transformed data leaves left-to-right ability **unchanged** — up to a rate of **0.9** — measured by PPL & sampling evals across scales.
- Only rate 1.0 degrades L2R, & only because the model then never sees an untransformed document. 0.5 is the recommended default, ❌the ceiling.
- → Infilling is a strict capability addition, not a trade. The paper's recommendation is to train every autoregressive model with FIM by default.
- Intuition: FIM is a reordering, not a corruption. No token is destroyed; the model still sees a valid document, just permuted.

*Why does SPM exist if PSM is the natural order?*
- **Cache reuse.** In SPM the suffix sits at the front. In an IDE the suffix (code after the cursor) is stable while the prefix grows as the user types → the suffix's KV cache survives every keystroke.
- Under PSM, the prefix comes first, so every typed character invalidates the cache for everything after it.
- Both are trained because inference-time serialization varies across serving stacks.

*Why split at the character level?*
- Splitting on token boundaries means the model only ever sees prefixes that end cleanly.
- Real cursors sit mid-token (`foo_ba|r`) → a token-level-trained model faces an out-of-distribution prefix and mispredicts.
- Character-level splits, tokenized afterwards, put those ragged boundaries in the training distribution.

*Why context-level over document-level?*
- Document-level applies FIM first, then packs & chunks to context length → a FIM document can be cut in half, orphaning `<MID>` from `<PRE>`/`<SUF>`.
- The model then trains on examples where the promised context is simply absent.
- Context-level transforms the chunk that is actually fed in → every example is well-formed.

*Where is it used?*
- The standard for code models: the transformation is cheap, the objective is unchanged, and it is what makes cursor-position completion work at all.
- ⚠️ FIM changes what the model conditions on, not what it optimizes. It is CLM on permuted data.
```

&nbsp;

## MTP
- **Name**: Multi-Token Prediction {cite:p}`gloeckle2024better`
- **What**: Predict the next $n$ tokens at each position, not just the next 1.
- **Why**: A single next-token target is a very local signal.
    - Position $i$ is only ever asked about $x_{i+1}$ → the representation is free to be myopic.
    - Densifies the training signal: $n$ targets per position instead of 1, for ~free.
    - Forces the model to **pre-plan** — a representation that must serve $x_{i+1}\dots x_{i+n}$ cannot encode only the immediate next token.
    - Payoff at inference: the extra heads are a built-in draft model → **speculative decoding** w/ no separate draft network.
- **How**:
    1. Shared trunk produces $h_i$.
    2. $n$ prediction heads/modules read $h_i$, each targeting a different future offset.
    3. Sum their CE losses into the main loss, scaled by $\lambda$.
    4. At inference: discard the extra heads, **or** reuse them to draft.

`````{note} Math
:class: dropdown
Notations:
- IO:
    - $t_i$: $i$-th token.
- Params:
    - $M_k\in\mathbb{R}^{d\times2d}$: Projection at depth $k$.
    - $\text{TRM}_k$: Transformer block at depth $k$.
- Hyperparams:
    - $D$: #additional tokens predicted.
    - $\lambda$: MTP loss weight.
- Misc:
    - $\mathbf{h}_i^k$: Representation of token $i$ at depth $k$; $\mathbf{h}_i^0$ = main model output.

````{tab-set}
```{tab-item} Parallel (Gloeckle)
$n$ **independent** heads on a shared trunk, all reading the same $\mathbf{h}_i$:

$$
\mathcal{L}_\text{MTP}=-\mathbb{E}\left[\sum_{i}\sum_{k=1}^{n}\log p_\theta(x_{i+k}\mid x_{\leq i})\right]
$$

- Heads are conditionally independent given $\mathbf{h}_i$ → ❌causal chain among the $n$ predictions.
- Heads run sequentially in the backward pass to cap the logit memory at 1 head.
```

```{tab-item} Sequential (DeepSeek-V3)
$D$ **chained** modules, each keeping the full causal chain: {cite:p}`deepseekai2024deepseekv3`

$$
\mathbf{h}_i'^k=M_k\left[\text{RMSNorm}(\mathbf{h}_i^{k-1});\ \text{RMSNorm}(\text{Emb}(t_{i+k}))\right]
$$

$$
\mathbf{h}_{1:T-k}^k=\text{TRM}_k(\mathbf{h}_{1:T-k}'^k),\qquad P_{i+k+1}^k=\text{OutHead}(\mathbf{h}_i^k)
$$

$$
\mathcal{L}_\text{MTP}^k=-\frac{1}{T}\sum_{i=2+k}^{T+1}\log P_i^k[t_i],\qquad \mathcal{L}_\text{MTP}=\frac{\lambda}{D}\sum_{k=1}^{D}\mathcal{L}_\text{MTP}^k
$$

- $\text{Emb}$ & $\text{OutHead}$ are **shared** w/ the main model → only $M_k$ + 1 block per depth is new.
- DeepSeek-V3: $D=1$ (predict 2 tokens total), $\lambda=0.3$ for the first 10T tokens then $0.1$ for the remaining 4.8T.
```
````
`````

```{attention} Q&A
:class: dropdown
*Parallel vs sequential heads — what is the trade?*
- **Parallel**: 1 shot, all $n$ heads read the same state. Cheap, but the $n$ predictions are conditionally independent → head $k$ must guess $x_{i+k}$ without knowing $x_{i+1}$.
- **Sequential**: module $k$ consumes both the depth-$(k-1)$ state and the true token $t_{i+k}$ → the causal chain is preserved, at the cost of $D$ extra blocks in sequence.
- Sequential is strictly better conditioned; parallel is cheaper.

*Why does MTP help large models but not small ones?*
- Empirically the gain **grows** w/ model size and is absent or negative below a few B params.
- Reading: the auxiliary task consumes capacity. A small model spends it and has none left; a large one has slack and gets a better-shaped representation.
- ⚠️ Same shape as the FLAN small-model result — auxiliary objectives are a capacity tax before they are a benefit.

*How big are the gains?*
- Gloeckle 13B: solves **12% more** problems on HumanEval & **17% more** on MBPP than a matched next-token model (relative, ❌percentage points); strongest on **generative** benchmarks, weaker on multiple-choice.
- 4-token prediction gives up to **3× faster** inference via self-speculative decoding.
- DeepSeek-V3: 2nd-token acceptance rate **85–90%**, delivering **1.8× TPS**.

*Is the MTP head kept at inference?*
- Optional. DeepSeek-V3 discards it by default — the main model is complete on its own — or repurposes it for speculative decoding.
- GLM-4.5 adds an MoE MTP layer explicitly **for** speculative decoding. {cite:p}`glm2025glm45`
- → The training benefit and the decoding benefit are separable; you can take either.

*Why isn't it universal yet?*
- The Qwen3 & Llama 3 reports never mention it, and Kimi K2's architecture table lists **0 MTP layers** — an explicit no.
- Costs: extra params, an extra loss weight to schedule, and gains that are scale- and domain-dependent (largest on code).
- ⚠️ Trending toward yes: DeepSeek-V3, GLM-4.5 (λ=0.3 for 15T tokens, then 0.1), and Qwen3-Next all ship it.
```

&nbsp;

## Diffusion LM
- **What**: Mask at a **random** rate $t\sim U(0,1)$, predict all masked tokens at once, reweight by $1/t$. {cite:p}`nie2025large`
- **Why**: Autoregression fixes both the decoding order and the decoding cost.
    - Left-to-right factorization is arbitrary; a token's best evidence is often to its right.
    - Ancestral decoding costs 1 forward pass per token → latency scales w/ $L$. Speculative decoding amortizes it, but the **dependency chain** is structural.
    - *Why not just use MLM?* A fixed 15% rate never trains the model on near-empty inputs → it cannot generate from scratch. Making the rate **random over $(0,1)$** is exactly what turns the same architecture into a valid generative model.
- **How**:
    1. **Forward**: at time $t$, mask each token independently w.p. $t$; $t=1$ → fully masked.
    2. Predict all masked tokens **simultaneously** from the partially masked sequence.
    3. **Reverse** (generation): start fully masked, iteratively unmask, remasking the low-confidence predictions.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $x_0$: Clean sequence; $x_t$: partially masked sequence.
    - $x_t^i$: $i$-th token of $x_t$; $\text{M}$ = mask symbol.
- Hyperparams:
    - $t\sim U[0,1]$: Masking ratio / diffusion time.

Objective:

$$
\mathcal{L}(\theta)=-\mathbb{E}_{t,x_0,x_t}\left[\frac{1}{t}\sum_{i=1}^{L}\mathbf{1}[x_t^i=\text{M}]\log p_\theta(x_0^i\mid x_t)\right]
$$

Upper bound on the NLL → a principled generative objective, ❌a heuristic:

$$
-\mathbb{E}_{p_\text{data}(x_0)}\left[\log p_\theta(x_0)\right]\leq\mathcal{L}(\theta)
$$

- $\frac{1}{t}$: Importance weight. Small $t$ → few masked tokens → the inner sum is small; dividing restores an unbiased estimate of the bound. Dropping it breaks the bound.
```

```{attention} Q&A
:class: dropdown
*Diffusion LM vs BERT — what actually differs?*
- Architecture: nothing essential. Both predict masked tokens bidirectionally.
- **Rate**: BERT fixes it at 15%; this samples $t\sim U(0,1)$ **and** reweights by $1/t$.
- **Corruption**: LLaDA masks *only* (a pure absorbing state); BERT's 80/10/10 also injects random & unchanged tokens.
- Consequence: the loss becomes a variational bound on $\log p_\theta(x_0)$ → a real generative model w/ in-context learning & instruction following.
- → A small change to the corruption schedule converts a representation learner into a generator.

*Pros?*
- **Parallel decoding**: many tokens per forward pass → the latency/quality trade becomes a tunable knob (#steps), ❌fixed at $L$.
- **Order-agnostic**: no left-to-right prior. On reversal poem completion LLaDA-8B scores **45.6 vs GPT-4o's 34.3** — the canonical failure of causal factorization.
- ⚠️ Read that honestly: LLaDA's *forward* score is far lower (51.8 vs 82.7). The directional **gap** closes; forward capability does not win.
- Bidirectional conditioning at every generation step.

*Cons?*
- ❌**KV cache**: every step rewrites the whole sequence, so the AR serving stack does not transfer.
- Generation runs on a fixed-length canvas chosen up front — semi-autoregressive / block sampling relaxes this, ❌removes it.
- Only an **upper bound** on likelihood → PPL is not directly comparable to AR models.
- Compute per token is high unless #steps is pushed low, which costs quality.

*Where does it actually stand?*
- LLaDA 8B is competitive with LLaMA3 8B — parity at 8B, not superiority, and not at frontier scale.
- **The compute gap is the real number**: masked diffusion scales at a rate comparable to AR, but needs ~**16× more pretraining compute** to match AR on text generation — after which it samples **1.4×** faster than a KV-cached AR model. {cite:p}`nie2024scaling`
- → A latency/throughput trade bought with training compute, ❌a free win.
- Production systems exist (Mercury, Gemini Diffusion) marketed on **throughput**, ❌capability.
- Honest read: the paradigm is validated as *viable*, and unproven at frontier scale. Treat scaling claims as open.
- ⚠️ The masked-diffusion objective predates LLaDA (discrete/masked diffusion LMs); LLaDA is cited here for carrying it to 8B scale w/ this exact formulation.

*Why does this matter beyond speed?*
- It decouples "language model" from "next-token predictor". The capabilities assumed to require autoregression — ICL, instruction following — survive without it.
- → What the objective must be is a much weaker constraint than the field assumed.
```

&nbsp;

## Auxiliary Losses
- **What**: Extra terms added to the main loss that regularize **numerics** or **routing**, not language modeling.
- **Why**: At scale the main loss alone is not sufficient to keep training well-posed.
    - Unbounded logits + low precision → loss spikes & divergence, recoverable only by rollbacks.
    - Sparse routing has a degenerate optimum (a few experts win everything) that the main loss happily accepts.

&nbsp;

### Z-Loss
- **What**: Penalize the squared log of the softmax normalizer. {cite:p}`chowdhery2022palm`
- **Why**: The softmax is shift-invariant, so nothing pins down the logit scale.
    - $\text{softmax}(z)=\text{softmax}(z+c)$ → CE is indifferent to logits drifting arbitrarily large.
    - Large logits + bf16 (8 exponent bits, 7 mantissa bits) → roundoff in $\log\sum e^{z_j}$ → corrupted gradients → loss spikes.
- **How**: Add $10^{-4}\log^2 Z$, pulling $\log Z\to0$.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $Z=\sum_{j}e^{z_j}$: Softmax normalizer (partition function) over logits $z$.
    - $B$: #tokens; $N$: #experts.
    - $c_z$: Router z-loss coeff.

Output z-loss:

$$
\mathcal{L}_z=10^{-4}\cdot\log^2 Z
$$

Router z-loss (same idea, applied to the gating logits $x\in\mathbb{R}^{B\times N}$): {cite:p}`zoph2022stmoe`

$$
\mathcal{L}_z^\text{router}=\frac{1}{B}\sum_{i=1}^{B}\left(\log\sum_{j=1}^{N}e^{x_j^{(i)}}\right)^2,\qquad c_z=0.001
$$

Total for a sparse model:

$$
\mathcal{L}_\text{tot}=\mathcal{L}_\text{CE}+c_B\mathcal{L}_B+c_z\mathcal{L}_z^\text{router}
$$
```

```{attention} Q&A
:class: dropdown
*Why $\log^2 Z$ and not a norm penalty on the logits?*
- $\log Z$ is exactly the quantity that overflows & loses precision in the CE computation → penalize it directly.
- Squaring makes it two-sided: it pins $\log Z$ near 0, ❌merely pushes it down.
- It is nearly free: at $10^{-4}$ it barely perturbs the CE optimum, since the softmax is shift-invariant anyway.

*Does it cost quality?*
- No — that is the point. ST-MoE compared it against tightening update clipping: clipping stabilized training but **catastrophically** hurt quality ($-4.21$ vs $-1.76$), while the router z-loss stabilized 3/3 runs w/ a slight quality **gain** ($-1.74$).
- → The rare stabilizer that is not a trade.

*Output z-loss vs router z-loss?*
- Same formula, different logits: the vocabulary softmax vs the expert gating softmax.
- The router is more fragile ← $N$ is small and a few large logits collapse routing to one expert.
- Attention logits also accept a z-loss; QK-Norm is the more common fix there.

*When do you need it?*
- Large scale + low precision. Small models in fp32 rarely spike.
- Symptom it treats: a loss spike w/ no bad data batch behind it.
```

&nbsp;

### Router Balancing
- **What**: Push MoE token assignment toward uniform across experts. {cite:p}`fedus2021switch`
- **Why**: Balanced routing is required for both quality and throughput, and the main loss opposes it.
    - **Routing collapse**: early winners get more tokens → train faster → win more. Self-reinforcing.
    - Expert parallelism puts experts on different devices → the slowest (most loaded) device sets step time, and overflow tokens get **dropped**.
- **How**: Add a term minimized by uniform assignment — or skip the loss and bias the routing decision directly.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $f_i$: Fraction of tokens dispatched to expert $i$.
    - $P_i$: Fraction of router probability mass on expert $i$.
    - $s_{i,t}$: Affinity of token $t$ to expert $i$; $b_i$: routing bias.
    - $\alpha$: Balance coeff; $\gamma$: bias update speed.

**Auxiliary loss** (Switch, $\alpha=10^{-2}$):

$$
\mathcal{L}_B=\alpha\cdot N\sum_{i=1}^{N}f_i P_i,\qquad f_i=\frac{1}{T}\sum_{x\in\mathcal{B}}\mathbf{1}\{\arg\max p(x)=i\},\qquad P_i=\frac{1}{T}\sum_{x\in\mathcal{B}}p_i(x)
$$

- $f$ is non-differentiable (an argmax count); $P$ carries the gradient. The product is still minimized at uniform.
- $\times N$ keeps the loss scale constant as $N$ varies ← at uniform, $\sum_i f_iP_i=N\cdot\frac{1}{N^2}=\frac{1}{N}$.

**Auxiliary-loss-free** (DeepSeek-V3): {cite:p}`wang2024auxiliary`

$$
g'_{i,t}=\begin{cases}s_{i,t},&s_{i,t}+b_i\in\text{Topk}(\{s_{j,t}+b_j\},K_r)\\ 0,&\text{otherwise}\end{cases}
$$

- $b_i$ shifts **selection only**; the gate weight multiplying the FFN output derives from $s_{i,t}$ (normalized across the selected experts, $g_{i,t}=g'_{i,t}/\sum_j g'_{j,t}$) → the bias never distorts gradients.
- $b_i\mathrel{-}=\gamma$ if overloaded, $\mathrel{+}=\gamma$ if underloaded, after each step. $\gamma=0.001$ for 14.3T tokens, then $0$ for the last 500B.
- A sequence-wise balance loss is retained at $\alpha=10^{-4}$ purely to prevent extreme within-sequence imbalance.
```

```{attention} Q&A
:class: dropdown
*Why is the auxiliary loss a problem at all?*
- It is a **second objective** competing w/ the LM loss: its gradient moves weights toward balance, ❌toward lower CE.
- Too small → routing collapses. Too large → measurable quality damage.
- → No setting of $\alpha$ is free; the whole method is a tuned compromise.

*Why is the bias trick better?*
- Balance is enforced by an **update rule**, not a gradient → zero interference gradients on the main objective.
- It is a control loop: measure load, nudge the bias, repeat. Balance is corrected where the imbalance is observed.
- → Removes the quality/balance trade instead of tuning it.

*Why does the bias touch only the top-$K$ selection?*
- Adding it to the gate weight would rescale expert outputs, injecting the balance signal into the forward computation & its gradients.
- Selection is discrete & already non-differentiable → biasing it **re-routes** tokens without rescaling any expert's contribution.
- ⚠️ The layer output does of course change — different experts run. What is preserved is that no *gradient* carries the balance objective.

⚠️ Load-balancing losses predate Switch (Shazeer et al. 2017 / GShard); Switch is cited here for this exact formula & $\alpha$.

*Why sequence-wise rather than batch-wise balance?*
- Batch-wise balance can hide a pathological sequence whose tokens all route to one expert.
- Sequence-wise is stricter, so it is kept at a tiny coefficient as a safety net beneath the bias mechanism.

*Why set $\gamma=0$ at the end of training?*
- Late in training the routing has stabilized; further nudging only perturbs a converging model.
- Same instinct as decaying the LR to zero.
```

&nbsp;

## Aggregation
- **What**: How per-token losses become the single scalar that is differentiated.
- **Why**: Silent and consequential — the same per-token losses give different gradients depending on how they are summed.

&nbsp;

### Loss Normalization
- **What**: Divide by total tokens (token-level) or average per-sequence first (sequence-level).
- **Why**: The two differ whenever sequences have unequal lengths.
    - Sequence-level averaging weights every **document** equally → each token in a short document gets more gradient than one in a long document.
    - Token-level weights every **token** equally.
- **How**: Sum all token losses and divide by the total unmasked token count across the whole optimization step.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\ell_{b,i}$: Loss at token $i$ of sequence $b$.
    - $L_b$: #supervised tokens in sequence $b$; $B$: #sequences.

$$
\mathcal{L}_\text{token}=\frac{\sum_{b=1}^{B}\sum_{i=1}^{L_b}\ell_{b,i}}{\sum_{b=1}^{B}L_b}
\qquad\text{vs}\qquad
\mathcal{L}_\text{seq}=\frac{1}{B}\sum_{b=1}^{B}\frac{1}{L_b}\sum_{i=1}^{L_b}\ell_{b,i}
$$

Equal only when all $L_b$ are equal.
```

```{attention} Q&A
:class: dropdown
*Why does this bite specifically with gradient accumulation?*
- Correct: accumulate the **sum** of token losses and the **token count**, divide once at the step boundary.
- Wrong: mean the loss within each micro-batch, then mean those means. That weights each **micro-batch** equally — matching neither formula above — and silently reweights whenever micro-batch token counts differ.
- → Grad accumulation stops being mathematically equivalent to a large batch. A well-publicized 2024 bug in several training stacks.

*Which is right for pretraining?*
- Token-level. Pretraining packs everything to a fixed context length, so $L_b$ is constant and the two coincide — until packing is imperfect or documents are masked apart.
- The choice becomes real for variable-length data: SFT, and RL where response lengths vary by orders of magnitude.
- Same axis DAPO argues on the RL side: token-level aggregation so long responses are not down-weighted per token. {cite:p}`yu2025dapo`

*Is per-document quality weighting a thing?*
- Not as a **loss** weight. The prevalent lever is data-side: filtering, dedup, and mixture reweighting/upsampling.
- Reweighting the loss and upsampling the document are near-equivalent in expectation, and upsampling composes with the data pipeline instead of the optimizer.
- → Be skeptical of "quality-weighted loss" claims; the standard practice is upstream of the objective.
```

&nbsp;

### Document Packing
- **What**: Concatenate documents to fill a fixed context, and mask attention across their boundaries. {cite:p}`grattafiori2024llama`
- **Why**: Padding to the longest document wastes most of the batch.
    - Web documents are short and wildly variable; padding to context length can waste the majority of FLOPs.
    - But naive packing lets a token attend to an **unrelated preceding document** → spurious conditioning.
- **How**:
    1. Greedily concatenate documents up to the context length; separate w/ `<eos>`.
    2. Apply a **block-diagonal** attention mask so each document attends only within itself.
    3. Exclude cross-boundary positions from the loss.

```{attention} Q&A
:class: dropdown
*How much does the intra-document mask actually matter?*
- Llama 3, verbatim: "limited impact during standard pre-training, but important in continued pre-training on very long sequences."
- Why the asymmetry: at 8K context a packed sequence holds few documents and the model learns to ignore text before `<eos>`. At 128K it holds many, and the spurious long-range dependencies are exactly what long-context training is trying to teach.
- → Cheap insurance in general, load-bearing for long context.
- ⚠️ Packing itself is old & universal (T5, PaLM, and essentially every stack). The citation here is for the **document-mask** finding specifically, ❌for packing.

*Why not just avoid packing?*
- Padding to context length is a direct FLOP loss; ModernBERT reports >99% packing efficiency with a greedy algorithm.
- Unpadding + packing is standard in every serious training stack.

*Does packing interact with the loss?*
- Yes: it changes the denominator in [loss normalization](#loss-normalization), and the boundary token's target is a token from a different document unless masked out.
```

&nbsp;

## Practice
### Recipe
- **What**: What a 2025 frontier pretraining loss actually contains.
- **How**:
    1. **[CLM](#clm)** on packed, boundary-masked sequences. This is ~all of the signal.
    2. **[FIM](#fim)** on ~50% of code data — a data permutation, same loss.
    3. **[MTP](#mtp)** as an auxiliary head, $\lambda\approx0.1$–$0.3$, if the model is large enough to pay for it.
    4. **[Z-loss](#z-loss)** at $10^{-4}$ for numerical stability.
    5. **[Router balancing](#router-balancing)** if MoE — increasingly bias-based, ❌loss-based.
    6. Token-level [normalization](#loss-normalization) over the full step.

```{attention} Q&A
:class: dropdown
*What is deliberately absent?*
- MLM, span corruption, RTD, mode tokens, NSP — none of them are in a frontier decoder-only recipe.
- Distillation is the notable exception for **small** models: Gemma-scale models are pretrained against a teacher's distribution instead of hard labels.

*How much of the loss is not next-token prediction?*
- Very little. Z-loss is $10^{-4}$, router balance $10^{-4}$–$10^{-2}$, MTP $\leq0.3$ of one extra head.
- → The objective is ~unchanged since GPT-2. What changed is data, scale, and the machinery around it.
```

&nbsp;

### Design Space
- **What**: The axes any pretraining objective is a point in.
- **How**:
    - **Corruption** $(\mu,r,n)$ — what fraction, in what shapes.
    - **Visibility** — causal / prefix / bidirectional. Independent of corruption.
    - **Target set** — which positions carry loss; drives signal density.
    - **Horizon** — 1 future token or $n$.
    - **Order** — fixed left-to-right, or learned/random.

```{attention} Q&A
:class: dropdown
*What has the field converged on?*
- **CLM** as the objective, at every scale, for every frontier decoder-only model.
- **Density matters, but is not sufficient.** MLM and prefix-LM lost partly on target density, and MTP wins by adding targets — yet RTD supervises 100% of positions and still did not displace CLM. Density is one force, ❌a universal predictor.
- **Objective changes are cheap only as data transformations** — FIM won because it changed the data, not the loss.
- **Auxiliary terms stay tiny.** Anything large enough to compete w/ CE is a quality regression.

*What is still contested?*
- **MTP**: real gains at scale, but Qwen3/Kimi K2/Llama 3 skip it. Training benefit vs decoding benefit are not cleanly separated in public results.
- **Diffusion LM**: validated at 8B and scaling at an AR-comparable *rate*, but at roughly a 16× training-compute premium. Throughput claims run ahead of capability claims.
- **RTD for encoders**: DeBERTaV3 says yes, ModernBERT says plain MLM. Unresolved.
- **Masking rate**: 15% is not universally optimal; the best value depends on model size & masking strategy, w/ no settled rule.

*What is unsolved?*
- **The signal ceiling.** CE against a single observed continuation says nothing about the other valid ones. The label is 1 sample from a distribution, and the loss treats it as the truth.
- **No credit assignment.** Every token contributes equally whether it is a memorized boilerplate token or the crux of a proof. Nothing in the objective distinguishes them.
- **Objectives are evaluated by proxy.** PPL correlates with downstream ability until it doesn't, and every comparison here is confounded by data and scale.

*What does this say about intelligence?*
- The objective is close to trivial — predict the next symbol — and nearly everything interesting came from scale and data instead. That is evidence the *loss* is not where the difficulty lives.
- Diffusion LM sharpens it: in-context learning and instruction following survive removing autoregression entirely. They are properties of **compressing text**, ❌of the left-to-right factorization.
- → The constraint that matters is that the model must model the joint distribution well. How you decompose it looks like an engineering choice.
```

&nbsp;
