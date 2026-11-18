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
# Models
How Transformer stacks became named LMs: the 3 stack types, the modern decoder block, & the model families up to late 2026.

Attention variants, KV-cache compression, pretraining objectives & post-training live on their own pages. This page links to them.

Notations:
- $m$: #Tokens in sequence
- $d$: Model dim (residual stream width)
- $l$: #Layers
- $h$: #Query heads
- $n_g$: #KV heads
- $d_k$: Dim per head
- $d_\text{ff}$: FFN hidden dim
- $\mathcal{V}$: Vocabulary
- $\mathbf{x}\in\mathbb{R}^{d}$: Hidden vector of 1 token

Override: $m$ = #tokens (global $m$ = #samples), matching [Attention](attn.md).

&nbsp;

## Encoder-only
- **What**: Bidirectional self-attention stack → 1 contextual vector per token; ❌native autoregressive decoding.

&nbsp;

### BERT
- **Name**: Bidirectional Encoder Representations from Transformers {cite:p}`devlin2019bert`
- **What**: Transformer encoder pretrained w/ [MLM](pre/obj.md#mlm) + NSP, then finetuned end-to-end per task.
- **Why**: 2018 transfer learning was unidirectional.
    - GPT-1: left-to-right only. ELMo: 2 separate one-directional LSTMs, concatenated → shallow bidirectionality.
    - Token-level tasks (QA span, NER) need both sides of context **at every layer**.
- **How**:
    1. **Input**: `[CLS] A [SEP] B [SEP]` in [WordPiece](token.md#wordpiece) (30K vocab).
    2. **Embedding** = token + segment (A/B) + learned absolute position.
    3. **Pretrain**: MLM on 15% of tokens + **NSP** (Next Sentence Prediction): binary "is B the real next sentence of A?" on `[CLS]`.
    4. **Finetune**: add 1 task head, train all params.
        - Classification → head on `[CLS]`.
        - Tagging / span QA → head on every token.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathcal{M}$: Masked positions.
    - $x_i$: True token at position $i$.
    - $\tilde{x}$: Corrupted input.
    - $y\in\{0,1\}$: NSP label (1 = B follows A).
- Misc:
    - $\hat{q}$: Predicted probability that B follows A.

Objective:

$$
\mathcal{L}=\underbrace{-\frac{1}{|\mathcal{M}|}\sum_{i\in\mathcal{M}}\log p_\theta(x_i\mid\tilde{x})}_{\text{MLM}}\ \underbrace{-\left[y\log\hat{q}+(1-y)\log(1-\hat{q})\right]}_{\text{NSP}}
$$

- Per example. In training, both terms are averaged over the batch, then summed.
- MLM output layer is tied to the input token-embedding matrix.
```

```{dropdown} Table: BERT sizes
| | $l$ | $d$ | $h$ | #Params |
|:--|:--|:--|:--|:--|
| BERT-Base | 12 | 768 | 12 | 110M |
| BERT-Large | 24 | 1024 | 16 | 340M |

- $d_\text{ff}=4d$. Max length 512.
- Data: BooksCorpus (800M words) + English Wikipedia (2,500M words).
- 90% of steps at length 128, last 10% at 512 ← attention cost $\propto m^2$.
```

```{attention} Q&A
:class: dropdown
*Why a `[CLS]` token instead of mean-pooling?*
- Self-attention lets 1 position aggregate the whole sequence → a dedicated slot for the sequence summary.
- NSP trains exactly that slot during pretraining → it is not random at finetune start.
- ⚠️ Raw `[CLS]` vectors are poor **similarity** embeddings w/o finetuning. Sentence-embedding models usually mean-pool & train contrastively.

*Why can't BERT generate text like GPT?*
- Every position sees every other position → ❌causal factorization → ❌native left-to-right decoding.
- ⚠️ Not impossible: iterative masked-token (Gibbs-style) sampling generates text, but slowly & w/ lower quality.
- Details: [MLM](pre/obj.md#mlm).

*What was BERT's real contribution?*
- Not the encoder (Transformer, 2017) & not masking (cloze task).
- **Deep bidirectional** pretraining via MLM → 1 pretrained encoder, finetuned w/ a minimal head, for both sentence-level & token-level tasks.
- ⚠️ Finetuning all params was already GPT-1's recipe; bidirectionality was the new part.
- 11 SOTA results at release, incl. GLUE & SQuAD.

*Why tie the MLM output layer to the input embeddings?*
- Same vocabulary on both sides → 1 matrix serves as lookup & as output projection.
- Saves $|\mathcal{V}|\cdot d$ params (≈23M at Base).
```

&nbsp;

#### RoBERTa
- **Name**: Robustly Optimized BERT Pretraining Approach {cite:p}`liu2019roberta`
- **What**: BERT architecture, retrained w/ a better recipe.
- **Why**: BERT was significantly undertrained, & its design choices were not ablated.
- **How**:
    - **Dynamic masking**: new mask pattern each time a sequence is fed (BERT: 10 static copies).
    - **❌NSP**: inputs = full sentences packed contiguously up to 512 tokens.
    - **Only full-length** 512-token sequences (BERT: mostly 128).
    - Batch 256 → **8K** sequences; Adam $\beta_2$ 0.999 → 0.98 for large-batch stability.
    - Data 16GB → **160GB**.
    - [Byte-level BPE](token.md#byte-level-bpe), 50K vocab (BERT: 30K WordPiece).

```{attention} Q&A
:class: dropdown
*What is the lesson?*
- Same architecture, recipe-only changes → matches or beats every post-BERT model published by mid-2019 (incl. XLNet).
- → Recipe (data, steps, batch) can dominate architecture. Ablate the recipe before you credit the architecture.

*Why drop NSP?*
- Removing NSP matches or slightly improves downstream results.
- NSP negatives come from other documents → solvable by topic mismatch alone → weak signal.

*Why dynamic masking?*
- BERT duplicated the data w/ 10 different masks over 40 epochs → each mask seen 4×. Dynamic masking → a fresh mask every time.
- Dynamic ≈ static or slightly better, & necessary once training runs for more steps & larger data.

*Why byte-level BPE if it was slightly worse on some tasks?*
- Universal encoding → ❌`[UNK]` for any input.
- The authors judged universality worth a minor score loss.
```

&nbsp;

#### DeBERTa
- **Name**: Decoding-enhanced BERT with disentangled attention {cite:p}`he2021deberta`
- **What**: BERT w/ separate content & relative-position vectors per token, & absolute position added only just before the MLM output layer.
- **Why**: BERT adds content & position into 1 vector → "what" & "where" are entangled before attention.
    - Attention between 2 words should depend on their contents **and** their relative distance, as separate terms.
- **How**:
    1. Each token keeps a content vector; each pair $(i,j)$ looks up a relative-position embedding for bucket $\delta(i,j)$.
    2. **Disentangled attention**: score = content→content + content→position + position→content.
        - position→position is dropped ← it carries no token information under relative positions.
    3. **EMD** (Enhanced Mask Decoder): add absolute positions after the encoder layers, before the MLM softmax.
        - Example: "a new **store** opened beside the new **mall**" → same local context, different syntactic roles → absolute position disambiguates.
    4. **DeBERTaV3**: replace MLM w/ [RTD](pre/obj.md#rtd) + gradient-disentangled embedding sharing. {cite:p}`he2021debertav3`

```{note} Math
:class: dropdown
Notations:
- Params:
    - $Q^c,K^c$: Content projections of token vectors.
    - $Q^r,K^r$: Projections of the shared relative-position embeddings.
- Hyperparams:
    - $k$: Max relative distance.

Relative distance bucket:

$$
\delta(i,j)=\begin{cases}0,&i-j\leq-k\\2k-1,&i-j\geq k\\i-j+k,&\text{otherwise}\end{cases}
$$

Attention logits:

$$
\tilde{A}_{ij}=\underbrace{Q^c_iK^{c\top}_j}_{\text{c2c}}+\underbrace{Q^c_iK^{r\top}_{\delta(i,j)}}_{\text{c2p}}+\underbrace{K^c_jQ^{r\top}_{\delta(j,i)}}_{\text{p2c}},\qquad A=\mathrm{softmax}\left(\frac{\tilde{A}}{\sqrt{3d_k}}\right)
$$

- Per head; projections map to $d_k$ dims.
- $\sqrt{3d_k}$ ← 3 terms summed → variance $\approx3\times$ a single dot product.
```

```{attention} Q&A
:class: dropdown
*Why does p2c use $\delta(j,i)$, not $\delta(i,j)$?*
- p2c = how much key **content** $j$ attends to the query's **position** relative to $j$ → the distance is measured from $j$'s side.

*Result?*
- First single model above the human baseline on SuperGLUE.
- DeBERTaV3 remained a top NLU encoder for years after.

*Why inject absolute position so late?*
- Relative position alone cannot tell "subject" from "object" when local contexts are the same.
- Late injection gives the absolute signal to prediction w/o letting it dominate all encoder layers.
```

&nbsp;

#### ModernBERT
- **What**: Encoder-only model rebuilt w/ the decoder-era block & long context. {cite:p}`warner2024smarter`
- **Why**: Encoders still serve retrieval, classification & tagging at scale, but BERT-era encoders had 512 context & a 2018 block.
- **How**:
    - [Pre-norm](#pre-norm), ❌bias terms (except the final MLM decoder layer), GeGLU FFN ([SwiGLU](#swiglu) family), [RoPE](pos.md#rope).
    - **Alternating attention**: global every 3rd layer; other layers use a 128-token [sliding window](attn.md#sliding-window).
    - Native **8192** context.
    - **Unpadding**: drop pad tokens & pack the batch → ❌wasted FLOPs.
    - Plain MLM at **30%** mask rate, ❌NSP. 2T training tokens.
    - Sizes: 149M (Base) & 395M (Large).

```{attention} Q&A
:class: dropdown
*Why do encoders survive in the LLM era?*
- 1 fixed-cost forward pass per document, ❌autoregressive decode → orders of magnitude cheaper per document.
- Bidirectional features are strictly better informed when the full input is present.
- Main users: RAG retrievers & rerankers, classifiers, safety filters.

*Why alternate local & global attention?*
- Global attention is $\mathcal{O}(m^2)$; most token interactions are local.
- 1 global layer in 3 keeps long-range access, at much lower cost. Same idea as Gemma's local:global ratio.
```

&nbsp;

## Encoder-Decoder
- **What**: Bidirectional encoder over the input + causal decoder that [cross-attends](attn.md#cross-attention) to it.
- **Why**: Natural fit when input & output are different sequences (translation, summarization).
    - Encoder: sees the full source. Decoder: generates the target.

&nbsp;

### T5
- **Name**: Text-to-Text Transfer Transformer {cite:p}`raffel2020exploring`
- **What**: Encoder-decoder Transformer that casts every task as text in → text out.
- **Why**: 1 model, 1 loss, 1 decoding rule for all tasks → a controlled testbed to compare transfer-learning choices.
- **How**:
    1. Prefix the task in text: `translate English to German: That is good.` → `Das ist gut.`
    2. Pretrain w/ [span corruption](pre/obj.md#span-corruption) on C4 (~750GB of cleaned Common Crawl).
    3. Finetune (or multitask train) w/ the same CE loss on target text.
    4. Block details:
        - Pre-norm w/ a scale-only LayerNorm (❌mean subtraction, ❌bias) ≈ [RMSNorm](../dl/foundation/module.md#rmsnorm).
        - **Relative position bias**: 1 learned scalar per (bucket, head) added to attention logits; shared across layers.
    5. Sizes: 60M, 220M, 770M, 3B, 11B.

```{attention} Q&A
:class: dropdown
*What did the T5 study conclude?*
- Encoder-decoder + denoising objective was best for **finetuned transfer** at matched cost.
- ⚠️ Not a contradiction of decoder-only winning later: T5 measured finetuned transfer at ≤11B params, ❌zero-shot generation at frontier scale.

*Why did encoder-decoder lose the LLM race?*
- 2 stacks, cross-attention, & a forced input/output split → awkward for open-ended chat & multi-turn context.
- A decoder-only model does the same job w/ 1 stack: concatenate input & output, use a causal mask.
- [CLM](pre/obj.md#clm) supervises 100% of positions → denser signal.

*Where does T5 live on?*
- Flan-T5 (instruction-tuned T5) & T5-style encoders as text encoders, e.g., in diffusion image models.
- T5 v1.1 already switched the FFN to GeGLU → see [SwiGLU](#swiglu).
```

&nbsp;

### BART
- **Name**: Bidirectional and Auto-Regressive Transformers {cite:p}`lewis2020bart`
- **What**: Seq2seq denoising autoencoder: corrupt text arbitrarily → bidirectional encoder → autoregressive decoder rebuilds the **full** original.
- **Why**: BERT has ❌native generation. GPT cannot see right context.
    - BART = BERT-style encoder + GPT-style decoder.
    - Decoder reconstructs the full text → any corruption is allowed, incl. ones that change length.
- **How**:
    1. Noise options: token masking, token deletion, **text infilling**, **sentence permutation**, document rotation.
        - Text infilling: replace spans (length ~ Poisson($\lambda=3$), incl. length 0) w/ 1 `[MASK]` each.
    2. Final recipe: text infilling + sentence permutation.
    3. Loss: CE of the decoder over the whole original document.

```{attention} Q&A
:class: dropdown
*BART vs T5 target?*
- BART: rebuild the **whole** document → decoder cost $\propto m$.
- T5: emit only the corrupted spans after sentinels → decoder cost $\propto$ corrupted length.

*Why was text infilling the best noise?*
- 1 `[MASK]` per span of unknown length → the model must also predict **how many** tokens are missing.
- Harder than single-token masking, & close to generation.

*What was BART best at?*
- Abstractive summarization (CNN/DM, XSum) at release; still a common seq2seq baseline.
```

&nbsp;

## Decoder-only
- **What**: Causal self-attention stack trained w/ [CLM](pre/obj.md#clm); ❌encoder, ❌cross-attention.
- **Why**: 1 stack, 1 objective, generation built in → the architecture of every frontier LLM since GPT-3.

&nbsp;

### GPT
- **Name**: Generative Pre-trained Transformer {cite:p}`radford2018improving`
- **What**: Transformer decoder (causal self-attention only) pretrained w/ CLM, then finetuned per task.
- **Why**: Labeled data is scarce; raw text is plentiful.
    - Learn a general representation generatively → transfer w/ minimal architecture change.
- **How**:
    1. Pretrain CLM on BooksCorpus ← long contiguous text → long-range dependencies.
    2. Finetune: serialize structured inputs into 1 token sequence w/ delimiter tokens (e.g., premise `$` hypothesis).
    3. Linear head on the last token; keep CLM as an auxiliary loss.
    4. Block: Post-LN, GELU, learned positions; 12 layers, $d=768$, 12 heads, context 512, ~117M params.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathcal{C}$: Labeled dataset.
- Hyperparams:
    - $\lambda$: Auxiliary LM weight ($0.5$).

Finetuning objective:

$$
\mathcal{L}_3(\mathcal{C})=\mathcal{L}_2(\mathcal{C})+\lambda\cdot\mathcal{L}_1(\mathcal{C})
$$

- $\mathcal{L}_1$: CLM loss.
- $\mathcal{L}_2$: Supervised task loss.
```

&nbsp;

#### GPT-2
- **What**: GPT scaled to 1.5B params on WebText; tasks done **zero-shot** from the prompt alone. {cite:p}`radford2019language`
- **Why**: Hypothesis: a large enough LM learns tasks implicitly, because natural text contains task demonstrations.
- **How**:
    - Data: **WebText** (40GB; outbound Reddit links w/ ≥3 karma).
    - [Byte-level BPE](token.md#byte-level-bpe), $|\mathcal{V}|=50{,}257$; context 512 → 1024.
    - **Pre-norm**: LayerNorm moved to the input of each sub-block + 1 extra LayerNorm after the last block.
    - Residual-layer weights scaled by $1/\sqrt{N}$ at init ($N$ = #residual layers).
    - Zero-shot via format: append `TL;DR:` → summarization.

```{dropdown} Table: GPT-2 sizes
| #Params (paper) | #Params (corrected) | $l$ | $d$ |
|:--|:--|:--|:--|
| 117M | 124M | 12 | 768 |
| 345M | 355M | 24 | 1024 |
| 762M | 774M | 36 | 1280 |
| 1542M | 1.5B | 48 | 1600 |

- Paper counts were later corrected upward in the official release.
```

```{attention} Q&A
:class: dropdown
*Why move LayerNorm to the sub-block input?*
- Keeps an identity path through the whole residual stream → stable gradients at depth 48. See [Pre-Norm](#pre-norm).

*Why the $1/\sqrt{N}$ init?*
- Each residual block adds its output to the stream → variance grows ~linearly w/ #blocks.
- Scaling each block's output projection keeps the stream variance $\mathcal{O}(1)$ at init.

*Did GPT-2 use nucleus sampling?*
- ❌. Nucleus sampling was published after GPT-2. The GPT-2 paper used top-$k$ sampling for its samples.
- See [Top-p](infer/dec.md#top-p-nucleus).
```

&nbsp;

#### GPT-3
- **What**: GPT-2 architecture scaled to 175B; tasks given **in context** (0/1/few-shot), ❌gradient updates. {cite:p}`brown2020language`
- **Why**: Finetuning needs labeled data per task & can overfit narrow distributions.
    - Scaling laws predicted smooth loss gains from size → test whether capability follows.
- **How**:
    - 175B params, 96 layers, $d=12288$, 96 heads × 128 dims, context 2048.
    - Layers alternate dense & locally banded sparse attention (Sparse Transformer style).
    - 300B training tokens: filtered Common Crawl + WebText2 + Books + Wikipedia, high-quality sets upsampled.
    - 8 sizes, 125M → 175B, to measure scaling.

```{attention} Q&A
:class: dropdown
*Is in-context learning "learning"?*
- ❌Weight update. The forward pass conditions on the examples.
- Few-shot gains grow w/ model size → ICL is an emergent use of the CLM objective at scale.

*What was wrong w/ GPT-3's training budget?*
- 300B tokens for 175B params ≈ 1.7 tokens/param.
- [Chinchilla](#chinchilla) later showed ~20 tokens/param is compute-optimal → GPT-3 was heavily undertrained.

*Why does GPT-3 matter if it is obsolete?*
- It moved the interface from finetuning to **prompting**.
- Every later product (ChatGPT, agents) builds on prompt-specified tasks.
```

&nbsp;

#### InstructGPT
- **What**: GPT-3 + SFT on human demonstrations + [RLHF](post/rl.md#rlhf) → follows instructions. {cite:p}`ouyang2022training`
- **Why**: CLM optimizes "predict internet text", ❌"do what the user intends, helpfully & safely" → misaligned.
- **How**:
    1. **SFT** on labeler-written demonstrations.
    2. **RM**: train a reward model on human rankings of sampled outputs.
    3. **PPO** against the RM, w/ a KL penalty to the SFT model, mixed w/ pretraining gradients (PPO-ptx).
    4. **ChatGPT** (Nov 2022): same method on dialogue data, from a GPT-3.5-series base.

```{attention} Q&A
:class: dropdown
*Headline result?*
- Labelers preferred 1.3B InstructGPT outputs over 175B GPT-3 outputs.
- → Alignment post-training can be worth more than 100× params for user-perceived quality.

*What is the "alignment tax"?*
- RLHF lowered scores on some public NLP benchmarks.
- PPO-ptx (mix in the pretraining loss) reduced most of this loss.

*Why did this become the template?*
- SFT → preference signal → RL is still the post-training skeleton in 2026.
- Main later changes: AI feedback, [DPO](post/sft.md#dpo)-style offline methods, & verifiable rewards ([RLVR](post/rl.md#rlvr)).
```

&nbsp;

## Scaling

&nbsp;

### Scaling Laws
- **What**: Test loss falls as a **power law** in params, data & compute. {cite:p}`kaplan2020scaling`
- **Why**: Training runs are too expensive to tune at full scale.
    - Fit the law on small runs → predict the large run → choose size & data before you spend.
- **How**:
    1. Train a ladder of small models across sizes & token counts.
    2. Fit loss as a function of params & tokens.
    3. For a compute budget $C$, choose the $(N,D)$ that minimizes predicted loss.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $N$: #Non-embedding params.
    - $D$: #Training tokens.
    - $C$: Training compute (FLOPs).
    - $\alpha_N,\alpha_D$: Fitted exponents.
    - $N_c,D_c$: Fitted constants.

Single-variable laws (other factors not bottlenecked):

$$
\mathcal{L}(N)=\left(\frac{N_c}{N}\right)^{\alpha_N},\qquad \mathcal{L}(D)=\left(\frac{D_c}{D}\right)^{\alpha_D}
$$

- $\alpha_N\approx0.076$, $\alpha_D\approx0.095$.

Compute of a dense Transformer:

$$
C\approx6ND
$$

- $6$ = 2 (forward multiply-add) × 3 (forward + 2× backward).
```

```{attention} Q&A
:class: dropdown
*What did Kaplan recommend?*
- Grow params faster than data: $N_\text{opt}\propto C^{0.73}$.
- → Large models trained on relatively few tokens. GPT-3 followed this.

*Why did that advice turn out wrong?*
- Same long LR schedule for all runs → intermediate checkpoints overestimate the loss of a short run w/ a matched schedule.
- → Training on less data looked worse than it is → "grow params faster than data".
- Fix: match the cosine schedule length to each run's token count.
- Corrected by [Chinchilla](#chinchilla).

*Does the law predict capabilities?*
- It predicts **loss**. Specific skills can appear abruptly as loss falls ("emergence").
- ⚠️ Contested: much apparent emergence is a metric artifact (e.g., exact-match scoring); smooth metrics show smooth gains.
```

&nbsp;

#### Chinchilla
- **What**: Compute-optimal scaling: grow params & tokens **equally**, ≈20 tokens/param. {cite:p}`hoffmann2022training`
- **Why**: Kaplan's fit undertrained large models.
- **How**: 3 independent fits (fixed-size curves, IsoFLOP profiles, parametric loss) → all agree $N_\text{opt}\propto C^{\approx0.5}$, $D_\text{opt}\propto C^{\approx0.5}$.
    - Test: Chinchilla (70B, 1.4T tokens) beat Gopher (280B, 300B tokens) at the same compute.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $E$: Irreducible loss (entropy of text).
    - $A,B,\alpha,\beta$: Fitted constants.

Parametric loss:

$$
\hat{\mathcal{L}}(N,D)=E+\frac{A}{N^{\alpha}}+\frac{B}{D^{\beta}}
$$

- Fit: $E=1.69$, $A=406.4$, $B=410.7$, $\alpha=0.34$, $\beta=0.28$.

Optimum under $C=6ND$:

$$
N_\text{opt}\propto C^{\frac{\beta}{\alpha+\beta}},\qquad D_\text{opt}\propto C^{\frac{\alpha}{\alpha+\beta}}
$$
```

```{tip} Derivation
:class: dropdown
*Why does the exponent split as $\beta/(\alpha+\beta)$?*

1. Substitute $D=C/(6N)$:

    $$
    \hat{\mathcal{L}}(N)=E+AN^{-\alpha}+B\left(\frac{C}{6}\right)^{-\beta}N^{\beta}
    $$

2. Set $\partial\hat{\mathcal{L}}/\partial N=0$:

    $$
    \alpha AN^{-\alpha-1}=\beta B\left(\frac{C}{6}\right)^{-\beta}N^{\beta-1}
    $$

3. Solve: $N^{\alpha+\beta}=\frac{\alpha A}{\beta B}\left(\frac{C}{6}\right)^{\beta}$ → $N_\text{opt}\propto C^{\beta/(\alpha+\beta)}$.
4. $D_\text{opt}=C/(6N_\text{opt})\propto C^{1-\beta/(\alpha+\beta)}=C^{\alpha/(\alpha+\beta)}$.
5. With the fitted $\alpha=0.34,\beta=0.28$: $N_\text{opt}\propto C^{0.45}$, $D_\text{opt}\propto C^{0.55}$ → close to equal scaling.
```

```{attention} Q&A
:class: dropdown
*Why do modern models train far past 20 tokens/param?*
- Chinchilla minimizes **training** compute only.
- Deployed models pay inference cost on every token served → a smaller model trained longer is cheaper overall.
- Example: Llama 3 8B on 15T+ tokens ≈ 1,900 tokens/param, & loss was still falling.

*Is "20 tokens/param" a law?*
- ❌. It depends on data quality, tokenizer, architecture & the fitting method.
- Treat it as an order of magnitude for training-compute-optimal dense models only.

*What happens when unique data runs out?*
- Repeated epochs give diminishing returns; a few epochs are almost as good as fresh data, many epochs are not.
- → Synthetic data, multimodal data, & RL compute become the next scaling axes.
```

&nbsp;
## Decoder Block
- **What**: The dominant open-LLM block since 2023: pre-RMSNorm residual → GQA w/ RoPE → pre-RMSNorm residual → SwiGLU or MoE FFN; ❌bias terms.
    - A reference design, ❌universal: e.g., Gemma uses GeGLU & extra post-norms, gpt-oss keeps some biases, OLMo 2 moves the norm.
- **Why**: Each change vs the 2017/GPT-2 block fixes 1 problem at scale.
    - Stability at depth → [pre-norm](#pre-norm) + [RMSNorm](../dl/foundation/module.md#rmsnorm).
    - Quality per FLOP → [SwiGLU](#swiglu).
    - Relative position & length extension → [RoPE](pos.md#rope).
    - Decode memory → [GQA](infer/cache.md#gqa) / [MLA](infer/cache.md#mla).
    - Params ≫ FLOPs → [MoE](#moe).
- **How**: Stack $l$ identical blocks, then final norm + LM head.

```{dropdown} Table: 2017 block → 2026 block
| Component | Transformer / GPT-2 | Llama-style (2023) | Late-2026 frontier open models |
|:--|:--|:--|:--|
| Norm position | Post-LN (2017) / Pre-LN (GPT-2) | Pre-norm | Pre-norm; often + post-norm on sublayer outputs |
| Norm type | LayerNorm | RMSNorm | RMSNorm + [QK-norm](attn.md#qk-norm) |
| Position | Sinusoidal / learned absolute | RoPE | RoPE (partial / mixed bases) + NoPE layers in some models |
| Attention | MHA | GQA | GQA / MLA + sliding-window or sparse or linear hybrids |
| FFN | ReLU / GELU, $d_\text{ff}=4d$ | SwiGLU, $d_\text{ff}\approx\frac{8}{3}d$ | Fine-grained MoE + shared expert |
| Bias terms | Yes | ❌ | ❌ |
| Context | 512–2048 | 2K–4K | 128K–1M |
```

```{attention} Q&A
:class: dropdown
*Why remove bias terms?*
- Fewer params & less memory traffic for ~0 quality cost.
- Empirical choice, ❌mathematical necessity (RMSNorm does not cancel additive shifts).
- PaLM reported better training stability for large models w/o biases.

*Which change matters most?*
- At small scale, each gives a small gain; together they are the "Llama recipe".
- At frontier scale, MoE & attention/KV compression dominate the cost profile. The FFN & norm choices are settled.

*Is the block still changing?*
- Yes, at the edges: attention type (sparse / linear hybrids), residual-stream design (hyper-connections), & norm placement.
- The core (pre-norm, RMSNorm, gated FFN, RoPE, no bias) has not moved since 2023.
```

&nbsp;

### Pre-Norm
- **What**: Put the norm on each sublayer's **input**, inside the residual branch; keep the residual path un-normalized. {cite:p}`baevski2019adaptive`
- **Why**: Post-LN (2017) normalizes **after** the residual add → gradients near the output are large at init.
    - → Post-LN needs a careful LR warmup or it diverges.
    - Pre-LN keeps an identity path from the loss to every layer → well-behaved gradients → warmup can be shortened or dropped. {cite:p}`xiong2020layer`
    - ⚠️ Origin is diffuse: tensor2tensor, Baevski & Auli, GPT-2 & others used it before Xiong's analysis.
- **How**:
    1. $\mathbf{x}\leftarrow\mathbf{x}+\mathrm{Attn}(\mathrm{Norm}(\mathbf{x}))$.
    2. $\mathbf{x}\leftarrow\mathbf{x}+\mathrm{FFN}(\mathrm{Norm}(\mathbf{x}))$.
    3. After the last block: 1 extra final norm before the LM head.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $F_l$: Sublayer $l$ (attention or FFN).
    - $\mathbf{x}_l$: Residual stream entering sublayer $l$.

Post-LN vs Pre-LN:

$$
\text{Post: }\ \mathbf{x}_{l+1}=\mathrm{Norm}\left(\mathbf{x}_l+F_l(\mathbf{x}_l)\right)\qquad\text{Pre: }\ \mathbf{x}_{l+1}=\mathbf{x}_l+F_l\left(\mathrm{Norm}(\mathbf{x}_l)\right)
$$

Unrolled Pre-LN:

$$
\mathbf{x}_L=\mathbf{x}_0+\sum_{l=0}^{L-1}F_l\left(\mathrm{Norm}(\mathbf{x}_l)\right)
$$

- $\partial\mathbf{x}_L/\partial\mathbf{x}_l$ contains an identity term → gradient reaches every layer undamped.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class PreNormBlock(nn.Module):
    def __init__(self, d, n_heads, d_ff):
        super().__init__()
        self.norm1, self.norm2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, bias=False, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, d_ff, bias=False), nn.SiLU(), nn.Linear(d_ff, d, bias=False))

    def forward(self, x, mask):
        ## Norm sits on the BRANCH input; the residual path x -> x is never normalized
        h = self.norm1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        x = x + self.ffn(self.norm2(x))
        ## Post-LN would be: x = norm1(x + attn(x)); x = norm2(x + ffn(x))
        return x

## Example
B, m, d = 2, 5, 16
x = torch.randn(B, m, d)
causal = torch.triu(torch.ones(m, m, dtype=torch.bool), diagonal=1)  ## True = blocked
blocks, final_norm = [PreNormBlock(d, 4, 4 * d) for _ in range(3)], nn.RMSNorm(d)
for blk in blocks:
    x = blk(x, causal)
print(final_norm(x).shape)  ## torch.Size([2, 5, 16]) -> extra norm before the LM head
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- Stable training at depth; less sensitive to LR & warmup.

*Cons?*
- Residual stream norm grows w/ depth → each later layer adds a relatively smaller update → deep layers contribute less.
- Post-LN reaches slightly better final quality when it trains at all (e.g., in shallow MT models).

*What do 2024+ models do about the cons?*
- **Sandwich / peri-norm**: norm on the sublayer input **and** output (Gemma 2–4).
- **Output-only norm inside the residual** (OLMo 2 style): normalize $F_l(\mathbf{x})$ before the add, ❌normalize the input.
- [QK-norm](attn.md#qk-norm): normalize $\mathbf{q},\mathbf{k}$ → bounds attention logits.
- All 3 keep the identity residual path; they only bound what each branch writes.

*Why does the final norm exist?*
- Pre-LN never normalizes the stream itself → the last hidden state has an uncontrolled scale.
- Final norm fixes the scale before the LM head's softmax.
```

&nbsp;

### SwiGLU
- **What**: FFN whose hidden layer is the product of a SiLU-gated projection & a linear projection. {cite:p}`shazeer2020glu`
- **Why**: Better quality at equal params & FLOPs.
    - Shazeer tested GLU, Bilinear, ReGLU, GEGLU & SwiGLU vs ReLU/GELU/Swish FFNs in T5 → GEGLU & SwiGLU had the best perplexity.
- **How**:
    1. 2 parallel projections up: **gate** & **up**.
    2. Apply [SiLU](../dl/foundation/activation.md#silu) to the gate; multiply elementwise w/ up.
    3. Project down.
    4. Shrink $d_\text{ff}$ to $\frac{2}{3}$ ← 3 matrices instead of 2 → same param count. Llama: $d_\text{ff}=\frac{2}{3}\cdot4d$.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{x}\in\mathbb{R}^{1\times d}$: Input as a row vector.
- Params:
    - $W_1,W_3\in\mathbb{R}^{d\times d_\text{ff}}$: Gate / up projections.
    - $W_2\in\mathbb{R}^{d_\text{ff}\times d}$: Down projection.

Forward:

$$
\mathrm{FFN}_\text{SwiGLU}(\mathbf{x})=\left(\mathrm{SiLU}(\mathbf{x}W_1)\odot\mathbf{x}W_3\right)W_2,\qquad\mathrm{SiLU}(z)=z\,\sigma(z)
$$

- GEGLU: replace SiLU w/ GELU.

Param match:

$$
\underbrace{2\cdot d\cdot d_\text{ff}}_{\text{ReLU FFN}}=\underbrace{3\cdot d\cdot d'_\text{ff}}_{\text{GLU FFN}}\ \Rightarrow\ d'_\text{ff}=\frac{2}{3}d_\text{ff}
$$

- Shazeer: $d_\text{ff}=3072$ → $d'_\text{ff}=2048$.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.w_gate = nn.Linear(d, d_ff, bias=False)
        self.w_up = nn.Linear(d, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        ## SiLU(gate) decides, per hidden unit and per token, how much of the linear "up" path passes
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

## Example: the 2/3 rule keeps params equal to a 4d ReLU FFN
d = 12
ffn = SwiGLU(d, d_ff=int(2 / 3 * 4 * d))         ## d_ff = 32
print(sum(p.numel() for p in ffn.parameters()))  ## 1152 = 3 * 12 * 32
print(2 * d * 4 * d)                             ## 1152 = ReLU FFN w/ d_ff = 48
print(ffn(torch.randn(2, 5, d)).shape)           ## torch.Size([2, 5, 12])
```
````

```{attention} Q&A
:class: dropdown
*Why does gating help?*
- The output is **multiplicative** in 2 projections of $\mathbf{x}$ → each hidden unit computes a 2nd-order interaction, ❌only a thresholded linear feature.
- The gate is smooth & input-dependent → each token chooses which hidden units pass.
- ⚠️ No accepted theory. Shazeer, verbatim: "We offer no explanation as to why these architectures seem to work; we attribute their success, as all else, to divine benevolence."

*Who uses it?*
- SwiGLU: PaLM, Llama, Mistral, Qwen, DeepSeek (also inside each MoE expert).
- GEGLU: T5 v1.1, Gemma, ModernBERT.

*Cost?*
- Same params & FLOPs by construction. 1 extra elementwise multiply & 1 extra activation tensor in memory.
```

&nbsp;

### MoE
- **Name**: Mixture of Experts {cite:p}`shazeer2017outrageously`
- **What**: Replace the FFN w/ $N$ expert FFNs + a router; each token runs only its top-$K$ experts.
- **Why**: Decouple params from FLOPs.
    - Quality scales w/ params; cost per token scales w/ **active** params.
    - → $N/K$× more params at ~the same FLOPs per token.
- **How**:
    1. Router: linear map from $\mathbf{x}$ to $N$ scores.
    2. Pick the top-$K$ experts per token.
    3. Gate weights = normalized scores of the chosen experts.
    4. Output = gate-weighted sum of the chosen experts' outputs.
    5. Keep routing balanced → [Router Balancing](pre/obj.md#router-balancing).

```{note} Math
:class: dropdown
Notations:
- Params:
    - $W_r\in\mathbb{R}^{N\times d}$: Router.
    - $E_i$: Expert $i$ (a SwiGLU FFN).
- Hyperparams:
    - $N$: #Routed experts.
    - $K$: #Experts activated per token.
- Misc:
    - $\mathcal{T}(\mathbf{x})$: Top-$K$ expert indices of $\mathbf{x}$.

Forward (softmax over the top-$K$ scores, as in Mixtral):

$$
\mathbf{s}=W_r\mathbf{x},\qquad\mathcal{T}=\mathrm{TopK}(\mathbf{s},K),\qquad g_i=\frac{e^{s_i}}{\sum_{j\in\mathcal{T}}e^{s_j}}\ (i\in\mathcal{T})
$$

$$
\mathbf{y}=\sum_{i\in\mathcal{T}}g_i\,E_i(\mathbf{x})
$$

- DeepSeek-V3 variant: sigmoid scores $s_i=\sigma(\cdot)$, gate $g_i=s_i/\sum_{j\in\mathcal{T}}s_j$.
- FLOPs/token ∝ $K$; params ∝ $N$.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

## Reuses SwiGLU from #swiglu above as each expert
class MoE(nn.Module):
    def __init__(self, d, d_ff, n_experts, k):
        super().__init__()
        self.router = nn.Linear(d, n_experts, bias=False)
        self.experts = nn.ModuleList(SwiGLU(d, d_ff) for _ in range(n_experts))
        self.k = k

    def forward(self, x):
        ## x: (T, d) -- batch & sequence flattened: routing is per TOKEN
        top_val, top_idx = self.router(x).topk(self.k, dim=-1)  ## (T, K)
        gate = top_val.softmax(dim=-1)                           ## normalize over the K chosen only
        y = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):
            tok, slot = (top_idx == e).nonzero(as_tuple=True)    ## tokens that picked expert e
            if tok.numel():                                      ## expert e computes ONLY its tokens
                y.index_add_(0, tok, gate[tok, slot, None] * expert(x[tok]))
        return y

## Example: 8 experts, top-2 -> each token pays for 2 FFNs, the layer stores 8
moe = MoE(d=16, d_ff=32, n_experts=8, k=2)
print(moe(torch.randn(10, 16)).shape)  ## torch.Size([10, 16])
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- Much better quality per training FLOP & per inference FLOP than a dense model of equal active size.
- Most frontier open models since late 2024 are MoE. Exception: Llama 3.1 405B (dense).

*Cons?*
- **Memory**: all $N$ experts must be resident → total params set the hardware bill.
- **Communication**: expert parallelism needs all-to-all token exchange between devices.
- **Imbalance**: hot experts slow the step or drop tokens.
- **Small-batch decode**: different tokens hit different experts → many experts' weights loaded per step → memory-bandwidth bound.
- **RL instability**: routing flips under small weight changes → see [GSPO](post/rl.md#gspo).

*What is capacity factor?*
- Max tokens per expert = capacity factor × (tokens / $N$). Overflow tokens skip the layer (residual only).
- Dropless MoE (block-sparse kernels) avoids dropping; most modern stacks are dropless.

*Do experts specialize by topic?*
- Mostly ❌. Mixtral found no clear topic-based assignment; routing follows syntax & token-level patterns more than domain.

*Top-1 vs top-2?*
- Switch: top-1 → least compute & communication. GShard & Mixtral: top-2.
- ⚠️ Top-1 needs the gate from the softmax over **all** $N$ experts. Renormalizing over the chosen 1 (as in the code above with `k=1`) makes $g=1$ → the router gets ❌gradient from the LM loss.
- 2024+: many **smaller** experts w/ $K=8$ → see [DeepSeekMoE](#deepseekmoe).

*How do you compare a 671B-A37B MoE with a dense 70B?*
- Compute per token: compare **active** params (37B vs 70B).
- Knowledge capacity: closer to **total** params.
- Memory & serving cost: total params.
```

&nbsp;

#### DeepSeekMoE
- **What**: Split each expert into $m$ smaller ones & activate $m$× more; add $K_s$ **shared** experts that every token uses. {cite:p}`dai2024deepseekmoe`
- **Why**: Coarse experts (8 big FFNs, top-2) learn overlapping knowledge.
    - **Fine-grained** → far more expert combinations per token → sharper specialization at the same FLOPs.
    - **Shared** → common knowledge (syntax, frequent patterns) stored once, ❌copied into every routed expert.
- **How**:
    1. Split: expert width $d_\text{ff}/m$; $mN$ experts; $mK$ active → same FLOPs as $N$ experts top-$K$.
    2. Isolate $K_s$ of them as **shared** (always run) → $mN-K_s$ routed, of which $mK-K_s$ selected → still the same FLOPs.
    3. Output = residual + shared outputs + gated routed outputs.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $m$: Split factor.
    - $K_s$: #Shared experts.
- Misc:
    - $\mathbf{u}$: FFN input of 1 token.
    - $g_i$: Gate of routed expert $i$; $0$ if not in the top-$(mK-K_s)$.

Forward:

$$
\mathbf{h}=\mathbf{u}+\sum_{i=1}^{K_s}\mathrm{FFN}_i(\mathbf{u})+\sum_{i=K_s+1}^{mN}g_i\,\mathrm{FFN}_i(\mathbf{u})
$$

Combinatorial flexibility (DeepSeekMoE example):

$$
N=16,K=2:\ \binom{16}{2}=120\qquad\xrightarrow{m=4}\qquad N=64,K=8:\ \binom{64}{8}=4{,}426{,}165{,}368
$$
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

## Reuses SwiGLU & MoE from above
class SharedMoE(nn.Module):
    def __init__(self, d, d_expert, n_routed, k, n_shared):
        super().__init__()
        self.shared = nn.ModuleList(SwiGLU(d, d_expert) for _ in range(n_shared))
        self.routed = MoE(d, d_expert, n_routed, k)

    def forward(self, x):
        ## Shared experts see EVERY token -> common knowledge; routed experts only specialize
        ## Returns the FFN branch only; the block adds the residual u (see Math)
        return sum(e(x) for e in self.shared) + self.routed(x)

## Example: DeepSeek-V3 shape, shrunk -> 1 shared + top-8 of 64 small experts
moe = SharedMoE(d=16, d_expert=8, n_routed=64, k=8, n_shared=1)
u = torch.randn(10, 16)
h = u + moe(u)          ## residual, as in the Math block
print(h.shape)          ## torch.Size([10, 16])
```
````

```{attention} Q&A
:class: dropdown
*Who adopted it?*
- DeepSeek-V2/V3, Qwen3-Next & Qwen3.5+, Kimi K2, GLM-4.5/5, Llama 4 Maverick (1 shared), Mistral Large 4 ("granular MoE").
- Typical 2025–26 shape: 128–512 routed experts, 8–10 active, 0–2 shared.

*Why do some models drop the shared expert?*
- Qwen3 MoE uses 128 experts, top-8, ❌shared. Qwen3-Next brought 1 shared expert back.
- ⚠️ No consensus. The benefit depends on expert count & balancing method.

*What limits granularity?*
- Tiny experts → small matmuls → poor GPU utilization; more routing & all-to-all overhead.
- → Expert width rarely goes below a few hundred dims.
```

&nbsp;

## Reasoning

&nbsp;

### Reasoning Model
- **What**: LLM trained to write a chain of thought (CoT) before the answer, w/ accuracy that grows w/ thinking length. {cite:p}`openai2024o1`
    - Training route: RL (o1, R1), or distillation from an RL-trained model (R1-distilled Qwen/Llama).
- **Why**: 1 forward pass = fixed compute per token.
    - Hard problems need more **serial** computation than 1 pass provides.
    - CoT prompting showed that intermediate tokens help. {cite:p}`wei2022chain`
    - Prompted CoT is brittle → train the model to **use** thinking tokens well: plan, check, backtrack.
- **How**:
    1. Start from a strong base model.
    2. Optional cold start: SFT on a small set of long-CoT examples.
    3. RL w/ verifiable rewards ([RLVR](post/rl.md#rlvr)): math answer checks, unit tests; algorithms such as [GRPO](post/rl.md#grpo).
    4. Inference: generate thinking tokens → final answer. Effort setting controls the thinking budget.

```{attention} Q&A
:class: dropdown
*What are the 2 scaling axes?*
- OpenAI reported that o1 accuracy rises w/ more **RL training compute** & w/ more **test-time thinking compute**.
- → Test-time compute is a new axis, beside params & data.

*What did DeepSeek-R1-Zero show?*
- RL on a base model, ❌SFT first, still produced long CoT, self-verification, & reflection. {cite:p}`deepseekai2025deepseek`
- Cons: poor readability & language mixing → R1 adds cold-start SFT & extra stages.

*Small reasoning models: RL or distillation?*
- R1 paper: distilling R1 traces into Qwen/Llama models beat running large-scale RL on the same small models.
- → Discovery needs a big model; transfer is cheap.

*Sequential vs parallel test-time compute?*
- Sequential: longer CoT (revise, backtrack). Parallel: sample many, vote or verify ([Best-of-N](infer/dec.md#best-of-n), [self-consistency](infer/dec.md#self-consistency)).
- Compute-optimal allocation depends on problem difficulty. {cite:p}`snell2024scaling`

*Is the CoT the model's real reasoning?*
- ⚠️ Not guaranteed. The CoT is a trained output; it can omit factors that drive the answer.
- Use CoT as evidence about the computation, ❌a faithful trace.

*What are the cons?*
- **Overthinking**: long CoT on easy queries → latency & cost ⬆️ w/o accuracy gain.
- Gains concentrate where rewards are verifiable (math, code); open-ended tasks gain less.
```

&nbsp;

#### Hybrid Reasoning
- **What**: 1 model that either answers directly or thinks first; a budget or effort setting decides how long. {cite:p}`anthropic2025claude37`
- **Why**: Separate chat & reasoning models double serving cost & force users to choose.
    - Most queries need no thinking; a few need a lot.
- **How**:
    1. Train both behaviors into 1 checkpoint (mode flag in the prompt or API).
    2. Expose a control: thinking-token budget → effort levels → **adaptive** thinking (model chooses).

```{dropdown} Table: How vendors expose thinking
| Model (first release) | Mechanism |
|:--|:--|
| Claude 3.7 Sonnet (Feb 2025) | First hybrid Claude: same model answers directly or uses extended thinking; API thinking-token budget. |
| Gemini 2.5 Flash (Apr 2025) | Thinking budget, incl. 0 (no thinking). |
| Qwen3 (Apr 2025) | 1 checkpoint, thinking & non-thinking modes. |
| DeepSeek-V3.1 (Aug 2025) | 1 checkpoint, thinking & non-thinking modes. |
| GPT-5 (Aug 2025) | **System**: fast model + thinking model + real-time router; API reasoning effort. |
| Claude Opus 4.6 → 5.5 (2026) | Adaptive thinking + effort levels; Opus 5.5: always-on adaptive thinking. |
```

```{attention} Q&A
:class: dropdown
*1 hybrid model or a router over 2 models?*
- Hybrid: 1 set of weights, cheap to serve; risk that the 2 modes interfere in training.
- Router: each model is specialized; the router can misjudge difficulty.
- Evidence of the trade-off: Qwen3-2507 split back into separate Instruct & Thinking checkpoints.

*Why move from budgets to adaptive thinking?*
- A fixed budget is a user guess about difficulty. The model sees the problem → it can judge better.
- Effort still bounds cost.
```

&nbsp;
## Open-Weight Families

&nbsp;

### Llama
- **What**: Meta's open-weight decoder family; set the 2023 block & the "train small models far past Chinchilla" recipe. {cite:p}`touvron2023llama`
- **Why**: Chinchilla minimizes **training** compute; deployment cost depends on model size.
    - → Train smaller models on many more tokens → cheap inference at a given quality.
    - LLaMA 1 used public data only → reproducible, & weights reached researchers.
- **How**:
    1. **LLaMA 1** (Feb 2023): [pre-norm](#pre-norm) RMSNorm + [SwiGLU](#swiglu) + [RoPE](pos.md#rope); 7B–65B; 1T–1.4T tokens; 2K context.
    2. **Llama 2** (Jul 2023): 2T tokens; 4K context; [GQA](infer/cache.md#gqa) at 34B & 70B only. Chat model: SFT → rejection sampling + PPO w/ 2 RMs (helpfulness, safety). {cite:p}`touvron2023llama2`
    3. **Llama 3** (Apr 2024): 8B & 70B; 128K vocab; GQA (8 KV heads) at all sizes; RoPE base 500K; 15T+ tokens; 8K context. {cite:p}`grattafiori2024llama`
        - **Llama 3.1** (Jul 2024): adds 405B **dense**; all sizes extended to 128K context.
    4. **Llama 3.2** (Sep 2024): 1B/3B pruned from 8B + [distilled](post/kd.md); 11B/90B add vision via **cross-attention adapters** on a frozen LM.
    5. **Llama 4** (Apr 2025): first Llama MoE; early-fusion multimodality; iRoPE; FP8 training; 30T+ tokens. {cite:p}`meta2025llama4`
        - Maverick: 128 routed + 1 shared expert; each token → shared + 1 routed; dense & MoE layers alternate.
        - **iRoPE**: interleave layers w/o positional encoding (NoPE) among RoPE layers + attention-temperature scaling at inference → length generalization (Scout: 10M context).
    6. **Muse** (2026): Meta's successor line. Open weights: Muse Glimmer 30B (Aug 2026, dense, 3:1 local:global, distilled from closed Muse Spark) {cite:p}`abtahizadeh2026muse` & Muse Spark Open (Oct 2026, 816B-A42B MoE, 256 experts top-8, QK-norm, gated attention, 1M context). {cite:p}`meta2026muse`

```{dropdown} Table: Llama configs
| Model | $l$ | $d$ | $h$ / $n_g$ | Total / Active | Tokens | Context |
|:--|:--|:--|:--|:--|:--|:--|
| LLaMA 7B | 32 | 4096 | 32 / 32 | 7B | 1T | 2K |
| LLaMA 65B | 80 | 8192 | 64 / 64 | 65B | 1.4T | 2K |
| Llama 3.1 405B | 126 | 16384 | 128 / 8 | 405B | 15.6T | 128K |
| Llama 4 Scout | | | | 109B / 17B (16 experts) | | 10M |
| Llama 4 Maverick | | | | 400B / 17B (128 + 1 shared) | 30T+ (family) | |
| Muse Spark Open | 62 | 8192 | 128 / 16 | 816B / 42B (256, top-8) | | 1M |

- Empty cell = not stated in the primary source.
```

```{attention} Q&A
:class: dropdown
*Why was Llama 3.1 405B dense, not MoE?*
- Meta's stated reason: maximize training **stability**. MoE added risk at a scale Meta had not yet trained.
- Llama 4 switched to MoE 1 generation later.

*Why did Llama 3 grow the vocab 32K → 128K?*
- Fewer tokens per text (esp. multilingual & code) → more text per context & per FLOP.
- Cost: larger embedding & LM-head matrices → noticeable at 8B, negligible at 405B.

*What is Ghost Attention (Llama 2)?*
- A **training-data** method, ❌an attention operator.
- Attach a system instruction to every turn of synthetic dialogues, then train w/ it removed from later turns → the instruction persists across many turns.

*Cross-attention adapters (3.2) vs early fusion (4)?*
- Adapters: image features enter via new cross-attention layers; LM weights frozen → text quality preserved; vision bolted on.
- Early fusion: image & text tokens share 1 backbone from pretraining → joint representations; needs multimodal pretraining.
- Trend: early fusion (Llama 4, Gemma 4 12B, Qwen3.5, Kimi K2.5).
```

&nbsp;

### Mistral
- **What**: Mistral AI's efficiency-first models; brought sliding-window attention & sparse MoE to open weights. {cite:p}`jiang2023mistral`
- **Why**: Maximize quality per active param & per KV-cache byte.
- **How**:
    1. **Mistral 7B** (Sep 2023): GQA (8 KV heads) + [sliding window](attn.md#sliding-window) $W=4096$ + rolling-buffer cache. Beat Llama 2 13B on all reported benchmarks.
    2. **Mixtral 8×7B** (Dec 2023): [MoE](#moe), top-2 of 8 experts per layer; 47B total / 13B active; 32K context. {cite:p}`jiang2024mixtral`
    3. **Mixtral 8×22B** (Apr 2024): 141B / 39B active; 64K context.
    4. **Magistral** (Jun 2025): RL-trained reasoning on Mistral Medium 3 & Small. {cite:p}`rastogi2025magistral`
        - [GRPO](post/rl.md#grpo) changes: ❌KL penalty, token-level loss normalization, minibatch advantage normalization, higher upper clip, drop zero-advantage groups.
    5. **Mistral 3** (Dec 2025): Large 3 = granular MoE, 675B / 41B active, multimodal, 256K. Ministral 3 (3B/8B/14B dense) via cascade distillation (iterative prune + distill).
    6. **Small 4** (Mar 2026): 119B, 128 experts top-4, ~6B active. **Large 4** (Oct 2026): 1.05T / 52B active, 1M context, open weights. {cite:p}`mistralai2026mistral`

```{attention} Q&A
:class: dropdown
*What is the rolling-buffer cache?*
- Store the key/value of position $i$ at slot $i \bmod W$ → cache size fixed at $W$.
- Valid because SWA never reads keys older than $W$.

*How does a 4K window see a 32K context?*
- Layer $k$ reads layer $k-1$'s window, which already summarized the window before it → reach $\approx l\cdot W$ ($32\times4096\approx131$K).
- ⚠️ Indirect reach ≠ exact recall. Modern models interleave full-attention layers → see [Hybrid Attention](infer/cache.md#hybrid-attention).

*Mixtral 13B active vs a dense 13B?*
- Same FLOPs per token. Mixtral matched or beat Llama 2 70B on most reported benchmarks ← it stores 47B params of knowledge.
- Memory to serve: 47B, ❌13B.
```

&nbsp;

### Gemma
- **What**: Google's open models from Gemini research; small sizes, heavy distillation, local:global attention. {cite:p}`gemmateam2024gemma`
- **Why**: Maximize quality at sizes that run on 1 accelerator or a phone.
- **How**:
    1. **Gemma 1** (Feb 2024): 2B (MQA) / 7B (MHA); GeGLU; RMSNorm; 256K vocab.
    2. **Gemma 2** (Jun 2024): GQA; local (4096) & global (8192) layers alternate 1:1; norm before **and** after each sublayer; logit soft-capping; 2B & 9B [distilled](post/kd.md#token-level-kd) from a larger teacher. {cite:p}`gemmateam2024gemma2`
    3. **Gemma 3** (Mar 2025): local:global **5:1**, window 1024; RoPE base 10K local / 1M global; [QK-norm](attn.md#qk-norm) replaces soft-capping; 128K context (1B: 32K); SigLIP vision (4B+). {cite:p}`gemmateam2025gemma`
    4. **Gemma 3n** (2025): on-device; per-layer embeddings (5B/8B stored, ~2B/4B on the accelerator); MatFormer nesting (E2B inside E4B); KV-cache sharing.
    5. **Gemma 4** (Apr 2026): E2B/E4B, 12B & 31B dense, 26B-A4B MoE (128 experts, top-8). {cite:p}`gemmateam2026gemma`
        - Local:global 5:1 (E2B 4:1); global layers use partial RoPE ($p=0.25$, base $10^6$) & reuse keys as values (except E2B/E4B).
        - 12B (Jun 2026): ❌separate vision/audio encoders; image patches & raw audio projected straight into the LM.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $c$: Soft-cap value (Gemma 2: 50 on attention logits, 30 on final logits).
- Misc:
    - $z$: A logit.

Logit soft-capping:

$$
z\leftarrow c\cdot\tanh\left(\frac{z}{c}\right)
$$

- $|z|\ll c$ → $z$ unchanged. $|z|\to\infty$ → $\pm c$. Smooth, bounded, differentiable.
```

```{attention} Q&A
:class: dropdown
*Why 5:1 local:global w/ a 1024 window?*
- Only global layers cache the full context → KV memory scales w/ #global layers.
- Gemma 3 cut global layers from 1/2 to 1/6 & the window from 4096 to 1024 w/ little perplexity loss.

*Why distill instead of training on hard labels?*
- Teacher's full next-token distribution = richer signal per token than a one-hot target.
- Gemma 2's stated aim: simulate training beyond the number of available tokens.

*Why did QK-norm replace soft-capping?*
- Both bound attention logits. QK-norm bounds them at the source ($\|\mathbf{q}\|,\|\mathbf{k}\|$), ❌an extra elementwise op on the full $m\times m$ score matrix inside the attention kernel.

*What is "effective params" in Gemma 3n/4 E-models?*
- Per-layer embedding tables can stay in CPU/flash memory → only ~2B/4B params need accelerator memory.
- ⚠️ Not the same as MoE active params: all non-embedding params still run for every token.
```

&nbsp;

### Qwen
- **What**: Alibaba's open family; widest size ladder, early hybrid-thinking checkpoints, & the first large Gated DeltaNet hybrids. {cite:p}`yang2025qwen3`
- **Why**: Cover every deployment size w/ 1 recipe; then cut long-context cost via linear-attention hybrids.
- **How**:
    1. **Qwen2.5** (Sep 2024): dense 0.5B–72B; GQA, SwiGLU, RoPE, RMSNorm, QKV bias; 18T tokens. {cite:p}`yang2024qwen25`
    2. **QwQ-32B** (Mar 2025): RL reasoning on Qwen2.5-32B.
    3. **Qwen3** (Apr 2025): dense 0.6B–32B + MoE 30B-A3B & 235B-A22B; 36T tokens.
        - ❌QKV bias → QK-norm.
        - MoE: 128 experts, top-8, ❌shared expert, global-batch balancing loss.
        - 1 checkpoint w/ thinking & non-thinking modes ([Hybrid Reasoning](#hybrid-reasoning)); 32K native → 128K via [YaRN](pos.md#yarn).
        - Small models: strong-to-weak distillation from the flagships.
    4. **Qwen3-2507** (Jul 2025): separate Instruct & Thinking checkpoints, "so we can get the best quality possible"; 256K native.
    5. **Qwen3-Next** (Sep 2025): 80B-A3B; 3:1 [Gated DeltaNet](attn.md#gated-deltanet) : gated attention ([Hybrid Stack](attn.md#hybrid-stack)); 512 experts, 10 routed + 1 shared; [MTP](pre/obj.md#mtp).
    6. **Qwen3.5 → 3.8** (2026): Next architecture scaled & made natively multimodal. Qwen3.5 397B-A17B (Feb); Qwen3.6 35B-A3B + 27B dense (Apr); Qwen3.8 open 2.4T-A95B (Aug), 92 layers. 262K native → ~1M extended. {cite:p}`qwenteam2026qwen38`
        - ⚠️ The open Qwen3.8 checkpoint is text-only & thinking-only; vision & non-thinking are in the hosted Qwen3.8-Max.

```{attention} Q&A
:class: dropdown
*Why drop the QKV bias & add QK-norm in Qwen3?*
- QK-norm bounds attention logits → fewer loss spikes at scale ([QK-Norm](attn.md#qk-norm)).
- Bias terms add little once logits are normalized.

*Why did Qwen move to a linear hybrid?*
- Long-context & agentic workloads → KV cache & attention FLOPs dominate.
- 3 of 4 layers w/ a constant-size state → cache ⬇️ ~4×; the 1-in-4 gated attention layers keep exact recall.

*Global-batch vs micro-batch balancing?*
- Micro-batch balance forces each small batch (often 1 domain) to use all experts evenly → blocks domain specialization.
- Global-batch balance only needs balance across the whole step → experts can specialize.
```

&nbsp;

### DeepSeek
- **What**: Open MoE line that introduced or scaled MLA, fine-grained + shared experts, auxiliary-loss-free balancing, FP8 training, DSA, & open reasoning RL. {cite:p}`deepseekai2024deepseekv3`
- **Why**: Frontier quality under tight compute → efficiency in every component, training & serving.
- **How**:
    1. **[DeepSeekMoE](#deepseekmoe)** (Jan 2024): fine-grained + shared experts.
    2. **V2** (May 2024): 236B-A21B; [MLA](infer/cache.md#mla) (KV latent 512); 160 routed + 2 shared, top-6; 128K. {cite:p}`deepseekai2024deepseek`
    3. **V3** (Dec 2024): 671B-A37B; 61 layers, $d=7168$; 256 routed + 1 shared, top-8; first 3 layers dense.
        - Sigmoid routing + [auxiliary-loss-free balancing](pre/obj.md#router-balancing).
        - 1 [MTP](pre/obj.md#mtp) module; FP8 mixed-precision training; 14.8T tokens.
    4. **R1 / R1-Zero** (Jan 2025): RL ([GRPO](post/rl.md#grpo)) on V3-Base → open reasoning model. See [Reasoning Model](#reasoning-model).
    5. **V3.1** (Aug 2025): 1 checkpoint w/ thinking & non-thinking modes.
    6. **V3.2-Exp** (Sep 2025): [DSA](attn.md#dsa) added to MLA by continued pretraining. **V3.2** (Dec 2025): + scaled RL & thinking during tool use. {cite:p}`deepseekai2025deepseekv32`
    7. **V4** (Apr 2026): Flash 284B-A13B & Pro 1.6T-A49B; 1M context; 32T+ tokens. {cite:p}`deepseekai2026deepseekv4`
        - Attention: interleaved **Compressed Sparse Attention** (compress KV, then select a sparse subset) & **Heavily Compressed Attention**, + a sliding-window branch → replaces MLA.
        - **mHC** (manifold-constrained hyper-connections): learned residual-stream mixing, constrained for stable signal propagation.
        - Muon optimizer.

```{dropdown} Table: DeepSeek configs
| Model | Total / Active | $l$ | $d$ | Attention | Routed / Active / Shared | Tokens |
|:--|:--|:--|:--|:--|:--|:--|
| DeepSeekMoE 16B | 16.4B / 2.8B | 28 | 2048 | MHA | 64 / 6 / 2 | 2T |
| V2 | 236B / 21B | 60 | 5120 | MLA | 160 / 6 / 2 | 8.1T |
| V3 | 671B / 37B | 61 | 7168 | MLA | 256 / 8 / 1 | 14.8T |
| V3.2 | 671B / 37B | 61 | 7168 | MLA + DSA | 256 / 8 / 1 | V3.1 base + continued |
| V4-Flash | 284B / 13B | 43 | 4096 | CSA / HCA hybrid | 256 / 6 / 1 | 32T+ |
| V4-Pro | 1.6T / 49B | 61 | 7168 | CSA / HCA hybrid | 384 / 6 / 1 | 32T+ |
```

```{attention} Q&A
:class: dropdown
*Did V3 cost $5.6M to train?*
- ❌ as commonly stated. The V3 report gives 2.788M H800 GPU-hours × an assumed \$2/GPU-hour ≈ \$5.576M for the **final training run only**.
- The report explicitly excludes prior research & ablation experiments on architecture, algorithms & data.

*Why FP8 training?*
- ~2× matmul throughput & ½ memory vs BF16.
- Risk: outliers overflow FP8 range → fine-grained (tile/block-wise) scaling + higher-precision accumulation.

*Why MTP in V3?*
- Denser training signal per token; at inference the MTP module can serve as a [speculative](infer/dec.md#speculative-decoding) draft.

*Why did V4 drop MLA?*
- At 1M context, even MLA's compressed cache & $\mathcal{O}(m^2)$ attention dominate.
- Compression + sparse selection cuts both memory & compute along $m$.
```

&nbsp;

### Kimi
- **What**: Moonshot AI's trillion-param open MoE line, trained w/ the Muon optimizer at scale. {cite:p}`kimiteam2025kimi`
- **Why**: Data is the bottleneck at frontier scale → maximize **token efficiency** (loss gain per token).
    - Muon: ~2× compute efficiency vs AdamW in Moonshot's experiments. {cite:p}`liu2025muon`
    - Muon at scale → exploding attention logits → QK-Clip.
- **How**:
    1. **K2** (Jul 2025): 1T-A32B; DeepSeek-V3-style (MLA, 61 layers, 1 dense layer); 384 routed + 1 shared, top-8; 64 heads (V3: 128); 15.5T tokens.
        - **MuonClip** = Muon + **QK-Clip**: after each update, if a head's max attention logit exceeds $\tau$, rescale that head's $W_q,W_k$ down.
    2. **K2-Thinking** (Nov 2025): interleaved thinking & tool calls; INT4 quantization-aware post-training; 256K.
    3. **Kimi Linear** (Oct 2025): 3:1 KDA : MLA → see [Hybrid Stack](attn.md#hybrid-stack).
    4. **K2.5** (Jan 2026): K2 + ~15T mixed text/vision continued pretraining; MoonViT encoder; thinking & instant modes.
    5. **K3** (Jul 2026): 2.8T-A104B; 93 layers = 69 KDA + 24 gated MLA; 896 routed, top-16, 2 shared; **Attention Residuals** (layers attend over earlier layers' outputs); 1M context. {cite:p}`kimiteam2026kimi` **K3.1-Pro** (Oct 2026): gated MLA → gated DSA.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\tau$: Logit threshold ($100$ in K2).
- Misc:
    - $S^h_{\max}$: Max pre-softmax attention logit of head $h$ in the batch.
    - $\eta_h=\min\left(\tau/S^h_{\max},1\right)$: Clip factor.

QK-Clip (after the optimizer step, per head):

$$
W_q^h\leftarrow\sqrt{\eta_h}\,W_q^h,\qquad W_k^h\leftarrow\sqrt{\eta_h}\,W_k^h\qquad(\text{only if }S^h_{\max}>\tau)
$$

- Logit $\propto\mathbf{q}^\top\mathbf{k}$ → bilinear → scaling both by $\sqrt{\eta}$ scales the logit by $\eta$ → $S^h_{\max}\to\tau$.
- $S^h_{\max}$ comes from the forward pass before the update → a soft control, ❌a hard post-update bound.
- MLA variant: content query & content key get $\sqrt{\eta_h}$; the head-specific rotary query gets $\eta_h$; the **shared** rotary key is ❌rescaled → the rotary logit term still scales by $\eta_h$.
```

```{attention} Q&A
:class: dropdown
*Why is QK-Clip needed w/ Muon?*
- K2 observed attention-logit explosions more often under Muon than AdamW at scale → loss spikes & divergence.
- ⚠️ The observation is empirical; the full mechanism is not settled.
- K2 reported zero loss spikes over 15.5T tokens w/ QK-Clip.

*Why QK-Clip instead of QK-norm?*
- QK-norm needs the full key at inference; MLA never materializes per-head keys in the cache.
- QK-Clip edits weights offline → ❌inference-time cost, MLA-compatible.

*Why 64 heads instead of V3's 128?*
- Long-context agentic inference cost scales w/ #heads; the K2 team found more heads gave only a small loss gain for a large inference cost.
```

&nbsp;

### GLM
- **Name**: General Language Model {cite:p}`glm2025glm45`
- **What**: Zhipu AI (Z.ai) open MoE line aimed at agentic coding.
- **Why**: Agentic tasks reward multi-step reasoning & long context over raw knowledge.
- **How**:
    1. **GLM-4.5** (Jul 2025): 355B-A32B; **deeper & narrower** than V3/K2 (92 layers, $d=5120$); GQA w/ 96 heads; 160 routed + 1 shared, top-8; 3 dense layers; QK-norm; 1 MTP layer; hybrid thinking. **Air**: 106B-A12B.
    2. **GLM-4.6** (Sep 2025): context 128K → 200K.
    3. **GLM-5** (Feb 2026): 744B-A40B; MLA + DSA; 78 layers; 256 routed + 1 shared, top-8; 28.5T tokens. {cite:p}`glm2026glm5`
    4. **GLM-5.2** (Jun 2026): 1M context; IndexShare (1 DSA indexer reused across 4 layers).
    5. **GLM-5.3-Flash** (Aug 2026): 320B-A18B; native multimodal; sparse + linear attention hybrid; mHC.

```{attention} Q&A
:class: dropdown
*Why deeper & narrower?*
- GLM-4.5 report: deeper models showed better reasoning capacity at similar params.
- Many heads (96 at $d=5120$) did not lower training loss but improved reasoning benchmarks.
- ⚠️ Lab-specific finding; Kimi K2 moved the other way on heads for inference cost.

*Why share 1 DSA indexer across layers?*
- Adjacent layers select similar top-$k$ tokens → recompute is mostly redundant.
- 1 indexer per 4 layers → ~¼ of the indexer cost.
```

&nbsp;

### MiniMax
- **What**: Open models that scaled a linear-attention hybrid, returned to full attention, then moved to sparse attention. {cite:p}`minimax2025minimax01`
- **Why**: 1M-token contexts at acceptable cost.
- **How**:
    1. **MiniMax-01** (Jan 2025): 456B-A45.9B; 80 layers; 7:1 lightning (linear) : softmax attention; 32 experts, top-2; 1M training / 4M inference context.
    2. **M1** (Jun 2025): reasoning RL on 01 w/ [CISPO](post/rl.md#cispo). {cite:p}`minimax2025minimax`
    3. **M2** (Oct 2025): **full attention** (GQA 48/8 heads), 62 layers, 256 experts top-8, QK-norm, 3 MTP modules.
    4. **M3** (Jun 2026): 428B-A23B; native multimodal; MiniMax Sparse Attention; 1M context.

```{attention} Q&A
:class: dropdown
*Why did M2 abandon the linear hybrid?*
- MiniMax's stated reason: efficient attention had not yet proven superior across the **whole industrial system** — quality on complex tasks, RL scaling, low-precision inference, caching, speculative decoding.
- ⚠️ Not a claim that linear attention is inferior in principle.
- Lesson: an architecture must win across the full training & serving stack, ❌on 1 benchmark.

*Why did M3 choose sparse over linear?*
- Sparse attention keeps the full KV cache (any token can be selected) → exact recall preserved; saves compute.
- Linear attention saves memory but loses exact recall → see [Linear Attention](attn.md#linear-attention).
```

&nbsp;

### gpt-oss
- **What**: OpenAI's open-weight reasoning MoE (Aug 2025). {cite:p}`openai2025gptoss`
- **Why**: Reasoning models emit many tokens → cost per token dominates → few active params.
- **How**:
    - **120b**: 116.8B total / 5.1B active; 128 experts, top-4. **20b**: 20.9B / 3.6B; 32 experts, top-4.
    - Layers alternate full attention & a 128-token banded window; GQA (group size 8).
    - Learned per-head bias in the softmax denominator → [attention sink](attn.md#attention-sink).
    - MoE weights in MXFP4 (4-bit) → 120b fits on 1 80GB GPU.
    - 128K context; reasoning effort low / medium / high; SFT + high-compute RL.

```{attention} Q&A
:class: dropdown
*Why is the active ratio so low (≈4%)?*
- Small-batch decode is weight-bandwidth bound → per-token latency ∝ active-weight bytes read (plus KV-cache traffic & communication).
- Reasoning traces are long → active size drives latency & cost.
- Total params still carry knowledge.

*Why quantize only the MoE weights?*
- MoE weights are ~90% of total params → most of the memory.
- Attention & embeddings are small & more precision-sensitive → kept in higher precision.
```

&nbsp;

### OLMo
- **Name**: Open Language Model {cite:p}`teamolmo2024olmo`
- **What**: Ai2's **fully open** LMs: weights + training data + code + logs + intermediate checkpoints.
- **Why**: Open weights alone cannot answer "why does this model behave this way" → science needs the full pipeline.
- **How**:
    1. **OLMo 2** (Nov 2024): RMSNorm on attention & MLP **outputs** before the residual add (❌on inputs) + QK-norm → stability.
    2. **Olmo 3** (Nov 2025): 3 of 4 layers use a 4096-token sliding window; YaRN on full layers; 65K context.
    3. **Olmo Hybrid** (Mar 2026): 3 Gated DeltaNet : 1 attention.

```{note} Math
:class: dropdown
Reordered norm (OLMo 2) vs pre-norm:

$$
\text{Pre: }\ \mathbf{h}=\mathbf{x}+\mathrm{Attn}\left(\mathrm{Norm}(\mathbf{x})\right)\qquad\text{OLMo 2: }\ \mathbf{h}=\mathbf{x}+\mathrm{Norm}\left(\mathrm{Attn}(\mathbf{x})\right)
$$

- Same form for the MLP sublayer, applied to $\mathbf{h}$.
- Still an identity residual path → ❌Post-LN (which normalizes after the add).
```

&nbsp;

### Mamba Hybrids
- **What**: Stacks of mostly [Mamba](ssm.md#mamba)-family SSM layers w/ a few attention layers. {cite:p}`lieber2024jamba`
- **Why**: Same logic as the [Hybrid Stack](attn.md#hybrid-stack): constant-size state for most layers, exact recall from a few attention layers.
- **How**:
    - **Jamba** (2024): first production-scale hybrid (attention : Mamba-1 = 1:7) + MoE.
    - **Nemotron-H** (Apr 2025): 8B = 24 Mamba-2 + 4 attention + 24 FFN layers; ❌positional embeddings. {cite:p}`blakeman2025nemotronh`
    - **Granite 4.0** (Oct 2025): 9:1 Mamba-2 : attention blocks; NoPE; MoE in larger sizes.
    - **Nemotron 3** (Dec 2025 – Jun 2026): hybrid Mamba-2/attention + MoE (LatentMoE in Super/Ultra), MTP, NVFP4 pretraining; Ultra 550B-A55B.

```{attention} Q&A
:class: dropdown
*Why no positional encoding?*
- Mamba layers are recurrent → order is built into the computation.
- The few attention layers read order from the Mamba-processed stream → RoPE is unnecessary.

*Mamba-2 vs Gated DeltaNet hybrids?*
- Both are linear-state layers. Gated DeltaNet adds the delta-rule overwrite → better associative recall ([Gated DeltaNet](attn.md#gated-deltanet)).
- 2026 open models split: NVIDIA/IBM → Mamba-2; Qwen/Kimi/Ai2 → DeltaNet-family.
```

&nbsp;

## Closed Frontier Families
- **What**: Proprietary API models; architecture & params mostly **undisclosed**.

&nbsp;

### GPT
- **What**: OpenAI's closed flagship line after GPT-3.5. {cite:p}`openai2023gpt4`
- **Why**: Frontier capability as a product; details withheld given "the competitive landscape and the safety implications" (GPT-4 report).
- **How**:
    1. **GPT-4** (Mar 2023): "Transformer-style model pre-trained to predict the next token", then RLHF; text + image input; 8K/32K context.
    2. **GPT-4o** (May 2024): 1 network trained end-to-end across text, vision & audio.
    3. **o1** (Sep–Dec 2024): RL-trained chain of thought → [Reasoning Model](#reasoning-model). **o3 / o4-mini** (Apr 2025): tool use during reasoning.
    4. **GPT-4.5** (Feb 2025): scaled pretraining, ❌reasoning step. **GPT-4.1** (Apr 2025): ~1M context, ❌reasoning step.
    5. **GPT-5** (Aug 2025): a **system** = fast model + thinking model + real-time router; API: 400K context, reasoning effort. {cite:p}`openai2025gpt5`
    6. **GPT-5.1 → 5.5** (Nov 2025 – Apr 2026): adaptive reasoning; effort `none` → `xhigh`; Codex variants for agentic coding.
    7. **GPT-5.6** Sol / Terra / Luna (Jul 2026) → **GPT-6** Astra (Sep 2026) & **GPT-6.1** Sol / Terra (Sep–Oct 2026): 1.05M-token API context.

```{attention} Q&A
:class: dropdown
*What is officially known about GPT-4's architecture?*
- Transformer-style, next-token pretrained, RLHF post-trained. Nothing else.
- ⚠️ MoE & parameter-count figures in circulation are **unconfirmed leaks**.

*What is "predictable scaling"?*
- OpenAI predicted some GPT-4 metrics (e.g., final loss) from models trained w/ 1/1,000 – 1/10,000 of its compute.
- → Scaling laws used as an engineering tool, ❌only a research finding.

*Why make GPT-5 a router system?*
- Most queries need a fast answer; few need long thinking → route by difficulty → cost & latency ⬇️.
- Later releases moved control into the model (adaptive reasoning) & the API (effort).
```

&nbsp;

### Claude
- **What**: Anthropic's closed family; architecture undisclosed. {cite:p}`anthropic2024claude3`
- **Why**: Helpful, honest, harmless assistants; training includes RLHF & [Constitutional AI](post/rl.md#rlaif) (principle-guided AI feedback).
- **How**:
    1. **Claude 1 / 2 / 2.1** (Mar–Nov 2023): 100K → 200K context.
    2. **Claude 3** Haiku / Sonnet / Opus (Mar 2024): vision input; 200K. **3.5 Sonnet** (Jun 2024); computer use beta (Oct 2024).
    3. **3.7 Sonnet** (Feb 2025): first [hybrid reasoning](#hybrid-reasoning) Claude; RL-trained extended thinking w/ token budget.
    4. **Claude 4** (May 2025): tool use **during** extended thinking. **4.1 / 4.5** (Aug–Nov 2025): agentic coding & computer use; effort control (Opus 4.5).
    5. **4.6 → 4.8** (Feb–May 2026): adaptive thinking, effort levels, 1M context.
    6. **Claude 5** (Jun–Jul 2026): Fable 5 & Mythos 5 (same underlying model, different configurations & access), Sonnet 5, Opus 5.
    7. **5.5 generation** (Sep–Oct 2026): Opus 5.5 (always-on adaptive thinking), Sonnet 5.5, Haiku 5.5; all 1M context. Fable / Mythos 5.2 (Nov 2026).

```{attention} Q&A
:class: dropdown
*What is known about Claude's architecture?*
- Nothing official: no params, no layer counts, no dense-vs-MoE statement.
- What is documented: context windows, modalities, thinking mechanisms, & training methods (RLHF, Constitutional AI).

*Why tool use during thinking (Claude 4)?*
- Agent tasks interleave reasoning & observation → thinking that can call tools sees real results instead of guessing.
- Same direction as o3 & Kimi K2-Thinking (interleaved thinking & tool calls).
```

&nbsp;

### Gemini
- **What**: Google DeepMind's closed, natively multimodal family. {cite:p}`geminiteam2023gemini`
- **Why**: Train text, image, audio & video together from the start → ❌bolt-on encoders.
- **How**:
    1. **Gemini 1.0** (Dec 2023): Ultra / Pro / Nano; natively multimodal.
    2. **Gemini 1.5** (Feb 2024): sparse MoE Transformer; 1M product context, 10M in research evaluation; Flash distilled from Pro. {cite:p}`geminiteam2024gemini`
    3. **Gemini 2.0 Flash** (Dec 2024): native image & audio output; tools.
    4. **Gemini 2.5** (Mar 2025): thinking models; sparse MoE w/ native text, vision & audio input; thinking budgets (incl. 0 on Flash). {cite:p}`comanici2025gemini`
    5. **Gemini 3 → 3.8** (Nov 2025 – Sep 2026): "thinking level" API control; Flash-tier releases for agentic coding.
    6. **Gemini 4** Argon / Carbon / Hydrogen (Sep–Nov 2026): Carbon: text, image, audio & video input; 1M combined input-output context.

```{attention} Q&A
:class: dropdown
*What does Google disclose that OpenAI & Anthropic do not?*
- Architecture class: Gemini 1.5 & 2.5 reports state **sparse MoE**.
- Still undisclosed: params, expert counts, layer layouts.

*Why is 1M+ context a Gemini hallmark?*
- 1.5 demonstrated near-perfect needle retrieval up to millions of tokens in research settings.
- ⚠️ Needle retrieval ≠ reasoning over the whole context; multi-hop long-context scores are lower.
```

&nbsp;

## Practice

&nbsp;

### Convergence
- **What**: What 2023–2026 model releases agree on, & what is still open.
- **How**:
    - **Dominant** (w/ exceptions): decoder-only; CLM pretraining; pre-norm RMSNorm; gated FFN (SwiGLU / GeGLU); RoPE; few or no biases; GQA or MLA; MoE at the frontier; distillation for small models; RL post-training for reasoning; hybrid or adaptive thinking.
    - **Moving**: attention at 1M context (sparse vs linear hybrids vs compressed); residual-stream design (mHC, Attention Residuals); optimizer (Muon vs AdamW); low precision (FP8 → NVFP4 / MXFP4); native multimodality w/o encoders.

```{dropdown} Table: Open-weight flagships (2024–2026)
| Model | Date | Total / Active | Attention | Experts (routed, top-$K$, shared) | Context |
|:--|:--|:--|:--|:--|:--|
| Llama 3.1 405B | 2024-07 | 405B dense | GQA | — | 128K |
| DeepSeek-V3 | 2024-12 | 671B / 37B | MLA | 256, 8, 1 | 128K |
| MiniMax-01 | 2025-01 | 456B / 45.9B | 7:1 lightning : softmax | 32, 2, 0 | 1M |
| Llama 4 Maverick | 2025-04 | 400B / 17B | iRoPE (RoPE + NoPE layers) | 128, 1, 1 | |
| Qwen3-235B | 2025-04 | 235B / 22B | GQA | 128, 8, 0 | 32K (128K YaRN) |
| Kimi K2 | 2025-07 | 1T / 32B | MLA | 384, 8, 1 | 128K |
| GLM-4.5 | 2025-07 | 355B / 32B | GQA | 160, 8, 1 | 128K |
| gpt-oss-120b | 2025-08 | 116.8B / 5.1B | full + 128-token banded | 128, 4, 0 | 128K |
| Qwen3-Next | 2025-09 | 80B / 3B | 3:1 GDN : gated attn | 512, 10, 1 | 256K |
| GLM-5 | 2026-02 | 744B / 40B | MLA + DSA | 256, 8, 1 | |
| DeepSeek-V4-Pro | 2026-04 | 1.6T / 49B | CSA / HCA hybrid | 384, 6, 1 | 1M |
| Kimi K3 | 2026-07 | 2.8T / 104B | 69 KDA + 24 gated MLA | 896, 16, 2 | 1M |
| Qwen3.8 open | 2026-08 | 2.4T / 95B | 3:1 GDN : gated attn | 512, 10, 1 | 262K (~1M ext.) |
| Muse Spark Open | 2026-10 | 816B / 42B | GQA, QK-norm, gated attn | 256, 8, — | 1M |
| Mistral Large 4 | 2026-10 | 1.05T / 52B | | granular MoE | 1M |

- Empty cell or "—" = not stated in the cited primary source.
- GDN = Gated DeltaNet. KDA = Kimi Delta Attention. CSA / HCA = Compressed Sparse / Heavily Compressed Attention.
```

```{attention} Q&A
:class: dropdown
*Biggest architecture shift 2023 → 2026?*
- Dense → fine-grained MoE: active ratio fell from 100% to 2–6%.
- Full attention → compressed / sparse / linear hybrids, driven by 1M-token agentic contexts.

*Where did capability gains actually come from?*
- Data (scale, quality, synthetic), post-training RL, & test-time compute — more than block changes.
- Block changes mostly bought **efficiency**: same quality at lower training & serving cost.

*Open vs closed gap?*
- Closed labs disclose ~nothing architectural → open reports (DeepSeek, Qwen, Kimi, GLM, Gemma, Llama) are the main public evidence for what works.
- ⚠️ Benchmark gaps shift every release; do not memorize a gap size.

*What should you be able to draw from memory?*
- The [Decoder Block](#decoder-block): pre-RMSNorm → GQA/MLA + RoPE → residual → pre-RMSNorm → SwiGLU / MoE (+ shared expert) → residual.
- The pipeline: CLM pretraining → long-context extension → SFT → RL (RLHF + RLVR) → distillation into small models.
```

&nbsp;
