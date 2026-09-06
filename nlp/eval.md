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
# Evaluation
This page lists common eval metrics for NLP tasks.

Scope: what is still reported for modern LLMs — how the number is produced, what it means, and how it is gamed. Task-generic metrics (accuracy, F1, AUC, calibration) → [ML Evaluation](../ml/eval.md).

Notations:
- $x$: Input (prompt / query / source text).
- $y$: Candidate (the output being scored).
- $y^*$: Reference (gold output).
- $y_t$: $t$-th token of $y$.
- $|y|$: Length of $y$ in tokens.
- $\mathcal{D}$: Eval set.
- $m$: #samples in $\mathcal{D}$.
- $\pi_\theta$: LM under evaluation (next-token distribution).
- $\mathcal{V}$: Vocab.
- $n$: N-gram order.
- $k$: Sample budget per input, or rank cutoff.

Overrides of the global scheme: $n$ = n-gram order (❌#features), $k$ = sample budget / rank cutoff (❌class idx).

&nbsp;

## Protocol
- **What**: Everything between "a model" and "a number" that is not the metric formula.
- **Why**: The same model on the same dataset with the same metric produces different scores under different protocols. Protocol is where most reported gaps actually come from.

&nbsp;

### Scoring Mode
- **What**: How the answer is elicited — rank fixed options by likelihood, or decode text and parse it.
- **Why**: A multiple-choice benchmark does not define how the model is asked.
    - Log-likelihood scoring needs logits → ❌closed APIs.
    - Generative scoring needs a parser → format failures score as ignorance.
    - → Two harnesses report two benchmarks, under one name.
- **How**:
    1. **Cloze**: score each full answer string as a continuation of the question, argmax.
    2. **Letter**: score only `" A"`/`" B"`/`" C"`/`" D"` after a lettered prompt, argmax.
    3. **Generative**: decode (usually w/ CoT), then [extract](#answer-extraction) the answer.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $o$: One candidate answer string, tokenized.
- Misc:
    - $B(o)$: #UTF-8 bytes of $o$.

Unnormalized option score:

$$
s(o)=\log\pi_\theta(o|x)=\sum_{t=1}^{|o|}\log\pi_\theta(o_t|x,o_{<t})
$$

Normalizations, all in use:

$$\begin{align*}
s_\text{token}(o)&=\frac{s(o)}{|o|}\\
s_\text{byte}(o)&=\frac{s(o)}{B(o)}\\
s_\text{PMI}(o)&=s(o)-\log\pi_\theta(o|x_\text{null})
\end{align*}$$
- $x_\text{null}$: Content-free prompt (e.g. `"Answer:"`) → subtracts the option's prior.
```

````{attention} Q&A
:class: dropdown
*Why normalize at all?*
- $\log\pi_\theta(o|x)<0$ per token → longer options score lower regardless of correctness.
- → Raw $s(o)$ systematically picks the shortest option.

*Why isn't length normalization the obvious answer?*
- It over-corrects when options share a long common prefix — the shared tokens are easy & inflate the mean.
- PMI instead removes the option's unconditional frequency, which is the actual confound for common vs rare answer strings.

*How large is the protocol effect?*
- LLaMA-65B on MMLU, same weights & same 14,042 questions: **63.6** (original Berkeley impl.) / **63.7** (HELM) / **48.8** (2023 EleutherAI harness). The paper's published 63.4 came from the original impl. {cite:p}`fourrier2023mmlu`
- → "MMLU = 60" is not a fact about a model until the harness & commit are named.

*Which mode do modern reports use?*
- Generative, because reasoning models must be allowed to think before committing.
- Cloze scoring forbids CoT by construction → it measures a different quantity for the same model.
````

&nbsp;

### Answer Extraction
- **What**: Pulling the committed answer out of free-form generation before grading it.
- **Why**: A verifier compares strings; a CoT trace is not a string answer.
    - Extraction failure is scored as a wrong answer → conflates formatting with capability.
- **How**:
    1. Constrain the format in the prompt: `\boxed{}`, `Answer: X`, or a JSON schema w/ [constrained decoding](infer/dec.md#constrained-decoding).
    2. Regex the **last** match ← models restate & revise mid-trace.
    3. Normalize: case, punctuation, articles, units, LaTeX, `1/2`$\Leftrightarrow$`0.5`.
    4. Check equivalence, ❌string equality: symbolic (SymPy) for math, unit tests for code.

```{attention} Q&A
:class: dropdown
*What silently inflates scores?*
- Unanchored substring search: `"the answer is not 42"` matches `42`.
- Accepting any digit in the trace → an arithmetic scratchpad guarantees a hit.

*What silently deflates them?*
- A model that answers correctly but ignores the requested format.
- → Report the extraction-failure rate separately from the error rate; otherwise a prompt-format change looks like a capability change.

*Why is this a reproducibility problem?*
- Extraction rules are code, not published spec. Two labs grading the same generations disagree.
```

&nbsp;

### Contamination
- **What**: Eval items, or near-duplicates, present in the training corpus.
- **Why**: The score then measures retrieval of a memorized answer, ❌capability.
    - Benchmarks are public text on the web → leakage is the default, not the exception.
- **How**:
    1. **Corpus-side**: n-gram overlap between eval items & training data → needs corpus access → ❌closed models.
    2. **Canaries**: a fixed GUID embedded in the benchmark; ask the model to complete it.
    3. **Behavioral**: score a freshly written, distribution-matched twin of the benchmark & compare.
    4. **Temporal**: date-partition, admit only post-cutoff items, refresh continuously.

```{attention} Q&A
:class: dropdown
*Evidence that it matters?*
- GSM1k = 1,205 new grade-school math problems built to match GSM8K's distribution.
- Accuracy drops up to **13pp** vs GSM8K for some families (Mistral, Phi); frontier models drop ≈0. {cite:p}`zhang2024careful`

*Is a GSM8K→GSM1k gap proof of literal leakage?*
- ❌. Repeatedly selecting checkpoints & hyperparams on a benchmark overfits its **style** with no test item ever being seen.
- → The gap measures overfitting-to-benchmark, of which memorization is one cause.

*Why doesn't decontamination solve it?*
- You can only strip what you can match. Paraphrases, translations & reformattings survive n-gram filters.
- The corpus is re-crawled; a benchmark released today is in next year's pretraining data.

*Practical consequence?*
- Treat any static benchmark's absolute number as a **potentially inflated** estimate, and trust **relative** movement on freshly-collected items instead.
```

&nbsp;

### Statistical Noise
- **What**: A benchmark score is a sample mean → it has a standard error.
- **Why**: Sub-point leaderboard gaps are reported as progress and are usually noise.
    - Two sources: finite #items, and stochastic decoding.
- **How**:
    1. Binomial SE over items, $\sqrt{\hat{p}(1-\hat{p})/m}$.
    2. Repeat w/ different seeds → adds decoding variance.
    3. **Paired** comparison on identical items → cancels item difficulty, far tighter than comparing two independent CIs.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\hat{p}$: Observed accuracy.

Item-sampling standard error:

$$
\text{SE}=\sqrt{\frac{\hat{p}(1-\hat{p})}{m}}
$$

MMLU, $m=14{,}042$, $\hat{p}=0.5$ (worst case):

$$
\text{SE}=\sqrt{\frac{0.25}{14042}}\approx0.42\%\quad\Rightarrow\quad\text{95\% CI}\approx\pm0.83\text{pp}
$$

A 57-subject breakdown gives $m\approx250$ per subject $\Rightarrow$ $\text{SE}\approx3.2\%$.
```

```{attention} Q&A
:class: dropdown
*Why are per-subject / per-category breakdowns nearly meaningless?*
- SE scales as $1/\sqrt{m}$ → a 100-item slice has SE ≈5pp.
- → Ranking models on an MMLU subcategory ranks noise.

*Why do overlapping CIs not imply "no difference"?*
- Both models were scored on the **same** items, so their errors are correlated: $\text{Var}(A-B)=\text{Var}(A)+\text{Var}(B)-2\text{Cov}(A,B)$.
- Shared item difficulty makes $\text{Cov}(A,B)$ strongly positive → the paired difference is usually far tighter than either marginal CI.
- → Use a paired test (McNemar / bootstrap over items) on the per-item outcomes, ❌two independent CIs.

*Do error bars fix the leaderboard?*
- Only partly. They cover sampling error, ❌selection error from testing many checkpoints against the same public test set.
```

&nbsp;

## Likelihood
### Perplexity
- **What**: Exponentiated mean per-token NLL on held-out text.
- **Why**: The pretraining objective itself, and the only score computable from raw text w/o labels, references or judges.
    - Reading: the effective #equally-likely tokens the model is hesitating between at each step.
- **How**:
    1. Teacher-force the held-out text.
    2. Mean NLL over its tokens.
    3. Exponentiate.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $H$: Cross-entropy in nats/token.
    - $L$: Context window.
    - $s$: Stride for long-document evaluation.

$$
\text{PPL}(y)=\exp\left(-\frac{1}{|y|}\sum_{t=1}^{|y|}\log\pi_\theta(y_t|y_{<t})\right)=\exp(H)
$$

Reference points: $\text{PPL}=1$ for a perfect model, $\text{PPL}=|\mathcal{V}|$ for a uniform one. **Unbounded above** — a model worse than uniform exceeds $|\mathcal{V}|$.

Documents longer than $L$: slide a window of size $L$ with stride $s<L$ and score only its last $s$ tokens, so every scored token has $\geq L-s$ tokens of context. Cost $\times L/s$.
```

````{important} Code
:class: dropdown
```python
import math
import torch
import torch.nn.functional as F

def nll_per_token(logits, targets):
    ## logits: (T, V) next-token logits; targets: (T,) the token that actually followed
    return F.cross_entropy(logits, targets, reduction="mean").item()  ## nats / token

def perplexity(logits, targets):
    return math.exp(nll_per_token(logits, targets))

def bits_per_byte(logits, targets, n_bytes):
    ## total nats -> bits, divided by BYTES: the denominator no longer depends on the tokenizer
    total_nats = nll_per_token(logits, targets) * targets.numel()
    return total_nats / (math.log(2) * n_bytes)

## Example: 4 steps, the model puts p=0.5 on the correct token every time
logits = torch.log(torch.tensor([[0.5, 0.2, 0.2, 0.1]] * 4))
targets = torch.tensor([0, 0, 0, 0])
print(round(perplexity(logits, targets), 2))                ## 2.0  <- hesitating between 2 tokens
print(round(bits_per_byte(logits, targets, 8), 2))          ## 0.5  == (4/8) * log2(2.0)
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- ❌Labels, ❌references, ❌judge. Runs on any raw corpus.
- Continuous & low-variance → the only metric fine-grained enough to track a pretraining run.

*Cons?*
- ❌Comparable across tokenizers → [BPB](#bits-per-byte).
- ❌Comparable across corpora — PPL on Wikipedia and on code are unrelated numbers.
- ❌A measure of usefulness: instruction-tuned models have **worse** PPL on raw web text than their base model.
- Dominated by frequent, easy tokens → nearly blind to the rare tokens that carry the meaning.

*Why exponentiate?*
- Puts cross-entropy on the branching-factor scale, where $|\mathcal{V}|$ is the uniform-model reference point → interpretable, and multiplicative gains read as ratios.

*Naive chunking of a long document?*
- The first token of every chunk is predicted from nothing → inflated PPL.
- → Strided windows, or report the context length used.

*Why not use PPL to compare a base model and its RLHF'd version?*
- Alignment shifts the output distribution away from the raw-text distribution on purpose.
- → PPL⬆️ is the intended effect, not a regression.
```

&nbsp;

#### Bits per Byte
- **Name**: Bits per Byte (BPB)
- **What**: Cross-entropy of the same text in bits per UTF-8 byte.
- **Why**: Perplexity's denominator is #tokens, which the tokenizer chooses, not the model.
    - Finer tokenizer → more tokens → less information per token → **lower** PPL at identical modeling quality.
    - Bytes are tokenizer-invariant → the only fair cross-model LM comparison.
- **How**: Total NLL over the text → nats to bits → divide by #bytes.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $B$: #UTF-8 bytes of the text.

$$
\text{BPB}=\frac{1}{\ln2}\cdot\frac{\sum_{t=1}^{|y|}-\log\pi_\theta(y_t|y_{<t})}{B}=\frac{|y|}{B}\log_2\text{PPL}
$$
- $\frac{|y|}{B}$: Tokens per byte, i.e., the inverse of the tokenizer's compression rate.
```

```{attention} Q&A
:class: dropdown
*Why bits, not nats?*
- 8 bits/byte is exactly the cost of a uniform-over-bytes model → BPB reads directly as a compression ratio against raw UTF-8.
- ❌A hard bound: BPB is a **cross**-entropy, so a badly mismatched model exceeds 8 (assigning $2^{-16}$ to an observed byte costs 16 bits).

*Can a tokenizer game BPB?*
- ❌. Splitting text more finely adds tokens to the numerator's sum and leaves $B$ fixed.
- A tokenizer only improves BPB by making the prediction problem genuinely easier.

*BPC?*
- Bits per character. Same construction, denominator = #characters → not comparable across scripts, since a CJK character is 3 UTF-8 bytes and an ASCII one is 1.

*Why is BPB the right metric for tokenizer research?*
- It is the one number that scores the tokenizer and the model **jointly** on the same footing — exactly the quantity a tokenizer change is supposed to improve.
```

&nbsp;

## Verifiable
- **What**: Metrics for tasks that have a checkable ground truth — short-form answers, math, code.

&nbsp;

### Exact Match
- **What**: 1 if the normalized prediction equals a normalized gold answer, else 0 — under the SQuAD normalization protocol. {cite:p}`rajpurkar2016squad`
- **Why**: Overlap metrics give partial credit for being partially wrong; a factual answer is right or it is not.
- **How**:
    1. Normalize both sides: lowercase, strip punctuation, articles, extra whitespace.
    2. Compare against every gold, take the max.

````{important} Code
:class: dropdown
```python
import re
from collections import Counter

def normalize(s):
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)   ## articles carry no answer content
    s = re.sub(r"[^\w\s]", " ", s)          ## punctuation
    return " ".join(s.split())              ## collapse whitespace

def exact_match(pred, golds):
    return float(any(normalize(pred) == normalize(g) for g in golds))

def token_f1(pred, golds):
    def f1(p, g):
        pt, gt = normalize(p).split(), normalize(g).split()
        ## MULTISET intersection: a token repeated twice must appear twice on both sides
        shared = sum((Counter(pt) & Counter(gt)).values())
        if shared == 0:
            return 0.0
        prec, rec = shared / len(pt), shared / len(gt)
        return 2 * prec * rec / (prec + rec)
    return max(f1(pred, g) for g in golds)  ## graded against the most favourable gold

## Example: right answer, wrong string
pred, golds = "Barack Hussein Obama", ["Barack Obama"]
print(exact_match(pred, golds), round(token_f1(pred, golds), 2))   ## 0.0 0.8
```
````

```{attention} Q&A
:class: dropdown
*Cons?*
- Semantically identical answers score 0: `"0.5"` vs `"1/2"`, `"US"` vs `"United States"`.
- → The normalization rules become part of the benchmark, and are almost never published with the score.

*Why is it still the default for QA & math?*
- ❌Judge cost, ❌judge bias, ❌judge drift. The number means the same thing across years and labs.

*Fix for math?*
- Symbolic equivalence (SymPy / `math-verify`) instead of string equality → $\frac{1}{2}$, $0.5$ and $2^{-1}$ all match.

*Fix for open-ended short answers?*
- A [judge](#llm-as-a-judge) with the gold answer in context (reference-guided grading). Costs the judge's biases back.
```

&nbsp;

### Token F1
- **What**: F1 over the multiset of tokens shared by prediction & gold, max over golds. {cite:p}`rajpurkar2016squad`
- **Why**: EM is all-or-nothing, but extractive answers differ by a span boundary or an article — not a wrong answer.
- **How**: Treat both sides as bags of tokens → precision, recall, harmonic mean → max over golds → mean over $\mathcal{D}$.

```{note} Math
:class: dropdown
$$
P=\frac{|y\cap y^*|}{|y|},\quad R=\frac{|y\cap y^*|}{|y^*|},\quad F_1=\frac{2PR}{P+R}
$$
- $\cap$: Multiset intersection, $\min$ of the two counts per token type.

Definition & properties → [F1](../ml/eval.md#f1).
```

```{attention} Q&A
:class: dropdown
*When does it mislead?*
- A rambling answer containing the gold span keeps recall 1 → nonzero F1 where EM gives 0.
- Precision does penalize the padding, but never to 0 → a verbose model always banks partial credit it did not earn.
- → Report EM alongside.

*Why the max over golds and not the mean?*
- Multiple golds are **alternative** correct answers, not a set the model must cover.

*Why token-level and not character-level?*
- Tokens are the unit at which an extractive answer differs. Character overlap would give credit for coincidental substrings.
```

&nbsp;

### pass@k
- **What**: Probability that ≥1 of $k$ sampled programs passes all unit tests. {cite:p}`chen2021evaluating`
- **Why**: Code has an actual **verifier** (execution), so the reference string is unnecessary — and a single greedy sample understates a model whose user can try $k$ times and check.
- **How**:
    1. Sample $S\gg k$ completions per problem at fixed temperature.
    2. Execute against hidden tests → count $c$ passing.
    3. Unbiased estimate of "≥1 of $k$ works", averaged over problems.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $S$: #samples generated per problem, $S\geq k$. (Source paper writes this $n$; here $n$ is n-gram order.)
- Misc:
    - $c$: #samples passing all tests.
    - $p$: True per-sample success probability.

Estimator, unbiased for $1-(1-p)^k$:

$$
\widehat{\text{pass@}k}=\mathbb{E}_{\text{problem}}\left[1-\frac{\binom{S-c}{k}}{\binom{S}{k}}\right]
$$

Numerically stable product form (❌overflowing binomials):

$$
\frac{\binom{S-c}{k}}{\binom{S}{k}}=\prod_{i=S-c+1}^{S}\left(1-\frac{k}{i}\right)
$$
```

````{important} Code
:class: dropdown
```python
import numpy as np

def pass_at_k(S, c, k):
    ## S: samples drawn per problem, c: samples passing every test
    if S - c < k:
        return 1.0   ## fewer than k failures exist -> at least one pass is guaranteed
    ## 1 - C(S-c, k)/C(S, k), expanded as a product so no binomial is ever materialized
    return 1.0 - np.prod(1.0 - k / np.arange(S - c + 1, S + 1))

## Example: 10 samples per problem, 2 of them correct
print(round(pass_at_k(10, 2, 1), 3))   ## 0.2    == c/S, i.e. avg@k
print(round(pass_at_k(10, 2, 5), 3))   ## 0.778  <- a budget of 5 tries nearly quadruples it
```
````

```{attention} Q&A
:class: dropdown
*Why not just $1-(1-\hat{p})^k$ with $\hat{p}=c/S$?*
- Plugging an estimate into a nonlinear function is biased: $\mathbb{E}[f(\hat{p})]\neq f(\mathbb{E}[\hat{p}])$, and the bias is upward.
- The combinatorial form is the exact unbiased estimator of $1-(1-p)^k$ for a $k$-subset drawn from the $S$ samples.

*Assumptions?*
- All $S$ samples i.i.d. from one fixed distribution (fixed prompt & [temperature](infer/dec.md#temperature)).
- The test suite is a **sound** verifier. Weak tests → a wrong program passes → the metric silently inflates.

*Why report both pass@1 and pass@k?*
- They optimize opposite things: low temperature maximizes pass@1, high temperature maximizes pass@$k$ for large $k$.
- Codex's own sweep: optimal $T\approx0.2$ for pass@1, $T\approx0.8$ for pass@100.
- → A single temperature cannot be optimal for both; the temperature must be stated with the score.

*Why do RL-tuned models improve pass@1 while regressing on pass@k?*
- RL maximizes expected reward = pass@1 → the policy sharpens onto one mode → the diversity pass@$k$ measures is destroyed.
- Same phenomenon as [entropy collapse](post/rl.md#entropy-collapse).

*Does pass@k measure capability or search?*
- Search. It is the capability of the *model + $k$ attempts + a perfect verifier*, and the verifier is doing selection the model cannot do alone.
```

&nbsp;

#### avg@k
- **What**: Mean per-sample pass rate over $k$ samples.
- **Why**: Greedy pass@1 is a single Bernoulli draw → variance dominates small differences.
    - Averaging $k$ stochastic samples is an unbiased, much lower-variance estimate of the **same** quantity.
- **How**: $\text{avg@}k=\frac{c}{k}$, averaged over problems.

```{attention} Q&A
:class: dropdown
*Relation to pass@1?*
- $\text{avg@}k=\widehat{\text{pass@}1}$ in expectation. It is not a $k$-attempt budget metric.

*Why does this matter when reading papers?*
- Modern RLVR reports label avg@$k$ (e.g. avg@32) as "pass@1". Two papers writing "pass@1" may be reporting greedy accuracy and a 32-sample mean.
- → Check $k$ and the temperature before comparing.
```

&nbsp;

#### maj@k
- **What**: Accuracy of the plurality answer among $k$ sampled reasoning chains. {cite:p}`wang2022self`
- **Why**: pass@$k$ needs a verifier to pick the winner; agreement across independently sampled chains is a **verifier-free** substitute.
    - Correct reasoning converges on one answer; wrong reasoning scatters.
- **How**: Sample $k$ chains → extract & canonicalize each final answer → take the mode → grade it. See [Self-Consistency](infer/dec.md#self-consistency).

```{attention} Q&A
:class: dropdown
*Ordering?*
- $\text{maj@}k\leq\text{pass@}k$ always: if the modal answer is correct, at least one sample was correct.
- $\text{maj@}k\geq\text{avg@}k$ in practice, but not guaranteed — it fails when a wrong answer is systematically preferred.

*Requirement?*
- A canonicalizable final answer, so votes can be counted. → ❌free-form text, ❌proofs, ❌code (two correct programs rarely match).

*Why is maj@k the honest "no-verifier" number?*
- pass@$k$ assumes an oracle that recognizes correctness. Deployment has no oracle.
- → maj@$k$ is what a user actually gets from $k$ samples.
```

&nbsp;

## NLG (Lexical)
The common cons of all lexical metrics is **a lack of semantic understanding**. This won't be mentioned individually.

### BLEU
- **Name**: Bilingual Evaluation Understudy.
- **What**: N-gram overlap measure between candidate sentence and one/more reference sentences.
- **Why**: Simple, efficient, language-agnostic for MT.
- **How**: Geometric Average of Clipped N-gram Precisions $\times$ Brevity Penalty.
    - **N-gram Precision**: $\frac{\#\text{correct predicted n-grams}}{\#\text{total predicted n-grams}}$.
    - **Clipped**: Limit the count for each correct n-gram to its max count in any reference sentence.
        - *Any reference sentence?*
            - There are many ways to express the same sentence.
            - It's normal to have multiple ref sentences to capture variations of one cand sentence.
            - → Compare cand sentence with each ref sentence. If the n-gram matches any ref sentence, it's correct.
        - *Max count in any reference sentence?*
            - It's very easy to cheat precision by **repetition** of correct words.
            - → Restrain it in the scope of ref sentence instead.
    - **Geometric Average**: Of all n-grams up to the specified n.
    - **Brevity Penality**: Penalize overly short sentences.
        - *Why?*
            - It's very easy to cheat precision with super short sentences ← The occurrence ratio of correct words in short cand sentences is higher

```{note} Math
:class: dropdown
Notations:
- $c$: Candidate length (i.e., #words in cand).
- $r_i$: Reference length (i.e., #words in ref).
- $r=\arg\min_{r_i}|r_i-c|$: Effective reference length (i.e., Reference length with the smallest absolute difference from cand)
    - Choose the shorter one if tied.

Clipped N-gram Precision:

$$
p_n=\frac{\sum_{\text{n-gram}}\min\left(\#_{\text{cand}}\text{n-gram},\max_{\text{ref}}\#_{\text{ref}}\text{n-gram}\right)}{\#_{\text{cand}}\text{total predicted n-grams}}
$$

Geometric Average of Clipped N-gram Precisions:

$$
\bar{p}_N=\prod_{n=1}^Np_n^{\frac{1}{N}}
$$

Brevity Penalty:

$$
BP=\begin{cases}
1 & \text{if }c>r \\
e^{1-\frac{r}{c}} & \text{if }c\leq r
\end{cases}
$$

BLEU:

$$
BLEU_N=BP\times \bar{p}_N
$$
```

```{attention} Q&A
:class: dropdown
*Cons?*
- Exact word matches → No word variations
- Ignore word importance → Useless but frequent words can boost BLEU score
- Ignore n-gram order → You can switch the n-grams and still get the same BLEU score
```

&nbsp;

### ROUGE
- **Name**: Recall-Oriented Understudy for Gisting Evaluation. {cite:p}`lin2004rouge`
- **What**: Recall-based N-gram overlap measure.
- **Why**: BLEU is precision-oriented → a summary that says almost nothing scores well.
    - Summarization is a **coverage** problem: did the summary keep what the reference kept?
- **How**:
    - **ROUGE-N**: Recall over n-grams. ROUGE-1 (content) & ROUGE-2 (fluency/order) are the reported pair.
    - **ROUGE-L**: Longest common subsequence → order-aware, ❌fixed $n$, ❌requirement that matches be contiguous.
    - **ROUGE-Lsum**: **Union-LCS** — for each reference sentence, union the LCS against every candidate sentence, de-duplicating tokens so nothing is counted twice. The variant actually reported on CNN/DM & XSum.
    - Modern implementations report F1, ❌raw recall ← recall alone is maximized by copying the whole document.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\mathcal{G}_n(\cdot)$: Set of **distinct** n-gram types in a sequence.
    - $g$: An n-gram type.
    - $\#_y(g)$: Count of $g$ in $y$.
    - $\text{LCS}(y,y^*)$: Length of the longest common subsequence.
    - $\beta$: Recall weight.

ROUGE-N (recall form):

$$
\text{ROUGE-N}=\frac{\sum_{g\in\mathcal{G}_n(y^*)}\min(\#_y(g),\#_{y^*}(g))}{\sum_{g\in\mathcal{G}_n(y^*)}\#_{y^*}(g)}
$$

ROUGE-L:

$$
R_\text{lcs}=\frac{\text{LCS}(y,y^*)}{|y^*|},\quad P_\text{lcs}=\frac{\text{LCS}(y,y^*)}{|y|},\quad F_\text{lcs}=\frac{(1+\beta^2)P_\text{lcs}R_\text{lcs}}{\beta^2P_\text{lcs}+R_\text{lcs}}
$$

The original paper sets $\beta$ large so $F_\text{lcs}\to R_\text{lcs}$; the widely used `rouge_score` package reports $\beta=1$.
```

````{important} Code
:class: dropdown
```python
def lcs(a, b):
    ## dp[i][j] = LCS length of a[:i] and b[:j]
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if a[i-1] == b[j-1] else max(dp[i-1][j], dp[i][j-1])
    return dp[-1][-1]

def rouge_l(cand, ref):
    c, r = cand.split(), ref.split()
    hit = lcs(c, r)
    if hit == 0:
        return 0.0
    ## subsequence, not substring -> matches may be interrupted, but order is preserved
    prec, rec = hit / len(c), hit / len(r)
    return 2 * prec * rec / (prec + rec)   ## beta=1, what rouge_score reports

## Example: one substituted word inside an otherwise identical sentence
print(round(rouge_l("the cat sat on the mat", "the cat was on the mat"), 3))   ## 0.833
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- Deterministic, free, reproducible from a spec — the number is stable across labs and years.
- ROUGE-L needs no $n$ and matches non-contiguously → tolerant of insertions and of gaps that zero a ROUGE-2 match. (Still order-sensitive: LCS ❌credits reordering.)

*Cons?*
- Same lexical blindness as BLEU: a perfect abstractive summary that shares no wording scores near 0.
- Reference-bound. LLM summaries frequently beat the human reference and are punished for it.
- ROUGE-1 alone is nearly a bag-of-words recall → gamed by dumping keywords.

*ROUGE-L vs ROUGE-Lsum?*
- L: one LCS over the whole flattened text.
- Lsum: split on newlines, union-LCS per reference sentence, aggregate.
- → Different numbers for the same pair, and papers routinely omit which one they used.

*Why does it survive at all?*
- As a regression guard, not a quality measure: a large ROUGE drop still reliably signals that the model stopped tracking the source.
```

&nbsp;

### chrF
- **Name**: Character n-gram F-score {cite:p}`popovic2015chrf`
- **What**: F-score over character n-grams, recall-weighted.
- **Why**: Word n-grams are tokenization-dependent, and one wrong suffix zeroes an entire word match.
    - Morphologically rich & agglutinative languages get systematically underscored by BLEU.
    - → Characters degrade gracefully & need no tokenizer at all.
- **How**:
    1. Extract character n-grams for $n=1..6$.
    2. Average precision (chrP) & recall (chrR) **arithmetically** over the orders present on **both** sides (❌geometric — that's BLEU).
    3. F-score with $\beta=2$.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\beta$: Recall weight, default 2 → recall counts twice as much as precision.
    - $n_\text{max}$: Max character n-gram order, default 6.

$$
\text{chrF}_\beta=(1+\beta^2)\frac{\text{chrP}\cdot\text{chrR}}{\beta^2\cdot\text{chrP}+\text{chrR}}
$$
- chrP: Fraction of candidate character n-grams found in the reference, averaged over the **effective orders** (those with n-grams on both sides).
- chrR: Fraction of reference character n-grams found in the candidate, averaged over the same effective orders.

sacreBLEU's default `chrF2` is $\beta=2,n_\text{max}=6$. **chrF++** additionally averages in word uni- & bi-grams.
```

````{important} Code
:class: dropdown
```python
from collections import Counter

def _ngrams(s, n):
    return Counter(s[i:i+n] for i in range(len(s) - n + 1))

def chrf(cand, ref, n_max=6, beta=2.0):
    tot_p = tot_r = 0.0
    orders = 0
    for n in range(1, n_max + 1):
        c, r = _ngrams(cand, n), _ngrams(ref, n)
        n_c, n_r = sum(c.values()), sum(r.values())
        ## "effective order": an order counts only if BOTH sides have n-grams,
        ## so precision and recall are always averaged over the SAME denominator
        if not n_c or not n_r:
            continue
        shared = sum((c & r).values())
        tot_p += shared / n_c
        tot_r += shared / n_r
        orders += 1
    if orders == 0:
        return float(cand == ref)                ## two empty strings are identical, not wrong
    P, R = tot_p / orders, tot_r / orders        ## ARITHMETIC mean over orders
    if P + R == 0:
        return 0.0
    return (1 + beta**2) * P * R / (beta**2 * P + R)
    ## sacreBLEU strips whitespace before extracting n-grams; kept here for brevity

## Example: a wrong suffix vs a wrong word -- BLEU-1 zeroes both equally
print(round(chrf("the cats sat", "the cat sat"), 3))     ## 0.662  <- inflection error, most credit kept
print(round(chrf("the feline sat", "the cat sat"), 3))   ## 0.318  <- different word, credit lost
print(round(chrf("ab", "a", n_max=2), 3))                ## 0.833  <- matches sacreBLEU on short strings
```
````

```{attention} Q&A
:class: dropdown
*Pros over BLEU?*
- ❌Tokenizer → scores from different systems are directly comparable.
- Better **segment-level** correlation with humans; BLEU is only trustworthy at corpus level.
- Graceful on morphology, compounding & typos.

*Cons?*
- Still lexical: a fluent paraphrase scores low.
- Not on BLEU's scale — a chrF of 55 and a BLEU of 55 mean nothing in common.
- Character n-grams reward shared function words & shared script → an inflated floor between related languages.

*Why does WMT still report it when COMET exists?*
- Neural metrics are checkpoint-dependent and drift between versions; chrF is a deterministic, reproducible baseline that pins the comparison.
```

&nbsp;

### Fuzzy String Matching
- **What**: Similarity derived from edit operations rather than exact equality.
- **Why**: Exact match scores a one-character difference identically to a completely wrong answer.
    - Graders need a tolerance band for typos, spacing, casing, punctuation & unit suffixes.
- **How**:
    1. Levenshtein distance $d$ by DP: min #insertions + deletions + substitutions.
    2. Normalize to $[0,1]$ by the longer string.
    3. Accept above a threshold.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\ell(y)$: Length of $y$ in **characters**. (Overrides the page-level $|y|$, which is a token count.)
    - $d_{i,j}$: Edit distance between the first $i$ chars of $y$ and the first $j$ chars of $y^*$.

$$
d_{i,j}=\begin{cases}
\max(i,j) & \min(i,j)=0\\
\min\begin{cases}
d_{i-1,j}+1\\
d_{i,j-1}+1\\
d_{i-1,j-1}+\mathbb{1}[y_i\neq y^*_j]
\end{cases} & \text{otherwise}
\end{cases}
$$

$$
\text{sim}(y,y^*)=1-\frac{d_{\ell(y),\ell(y^*)}}{\max(\ell(y),\ell(y^*))},\qquad \text{sim}=1\text{ when both are empty}
$$

$\mathcal{O}(\ell(y)\ell(y^*))$ time, $\mathcal{O}(\min(\ell(y),\ell(y^*)))$ memory with a rolling row.
```

````{important} Code
:class: dropdown
```python
def levenshtein(a, b):
    prev = list(range(len(b) + 1))       ## only one row is ever needed
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            ## delete / insert / substitute -- substitution is free when the chars already match
            curr.append(min(prev[j] + 1, curr[j-1] + 1, prev[j-1] + (ca != cb)))
        prev = curr
    return prev[-1]

def fuzzy_ratio(a, b):
    n = max(len(a), len(b))
    return 1.0 if n == 0 else 1 - levenshtein(a, b) / n   ## two empty strings are identical

## Example
print(levenshtein("kitten", "sitting"), round(fuzzy_ratio("kitten", "sitting"), 3))   ## 3 0.571
```
````

```{attention} Q&A
:class: dropdown
*Where it is actually used in LLM eval?*
- Grading short-form answers with a tolerance instead of brittle [exact match](#exact-match).
- Near-duplicate detection for [contamination](#contamination) checks.
- WER / CER for speech & OCR: edit distance at word / character level, divided by the reference length.

*Why can WER exceed 1?*
- Its denominator is the reference length, but insertions are unbounded → a hallucinating transcriber can be >100% wrong.

*When NOT to use it?*
- Long generations: distance is dominated by length, and two unrelated long texts land at similar ratios.
- Anything meaning-sensitive: `"increase"` → `"decrease"` is 4 edits, i.e., "almost right".

*Why is a fixed acceptance threshold dangerous?*
- The right threshold depends on answer length: 2 edits is noise in a 40-char answer and a different answer in a 3-char one.
- → Normalize first, then threshold — or don't use it as a grader at all.
```

&nbsp;

## NLG (Semantic)

### BERTScore
- **What**: Greedy-matched cosine similarity between the contextual embeddings of candidate & reference tokens. {cite:p}`zhang2020bertscore`
- **Why**: N-gram overlap requires the *same words*; a correct paraphrase shares none.
    - Contextual embeddings put paraphrases near each other, and disambiguate homographs that a static embedding cannot.
- **How**:
    1. Embed both sequences with a frozen pretrained encoder.
    2. Cosine similarity for every candidate–reference token pair.
    3. **Greedy**: each token independently takes its best partner → precision from the candidate side, recall from the reference side.
    4. F1, optionally IDF-weighted, optionally baseline-rescaled.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\mathbf{e}_i$: L2-normalized contextual embedding of candidate token $y_i$.
    - $\mathbf{e}^*_j$: L2-normalized contextual embedding of reference token $y^*_j$.
    - $b$: Baseline score from random sentence pairings.

Vectors are L2-normalized → the inner product **is** the cosine.

$$\begin{align*}
P_\text{BERT}&=\frac{1}{|y|}\sum_{i=1}^{|y|}\max_{j}\mathbf{e}_i^\top\mathbf{e}^*_j\\
R_\text{BERT}&=\frac{1}{|y^*|}\sum_{j=1}^{|y^*|}\max_{i}\mathbf{e}_i^\top\mathbf{e}^*_j\\
F_\text{BERT}&=\frac{2P_\text{BERT}R_\text{BERT}}{P_\text{BERT}+R_\text{BERT}}
\end{align*}$$

Baseline rescaling, for readability only:

$$
\hat{F}=\frac{F_\text{BERT}-b}{1-b}
$$

Monotone in $F_\text{BERT}$ → ❌changes any ranking.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn.functional as F

def bert_score(cand_emb, ref_emb):
    ## cand_emb: (Tc, d), ref_emb: (Tr, d) -- contextual embeddings from a FROZEN encoder
    c = F.normalize(cand_emb, dim=-1)
    r = F.normalize(ref_emb, dim=-1)
    sim = c @ r.T                        ## (Tc, Tr) cosine similarity of every pair
    ## greedy, NOT a one-to-one assignment: one reference token may serve several candidates
    P = sim.max(dim=1).values.mean()     ## candidate token -> its best reference token
    R = sim.max(dim=0).values.mean()     ## reference token -> its best candidate token
    return (2 * P * R / (P + R)).item()

## Example
torch.manual_seed(0)
ref = torch.randn(5, 8)
print(round(bert_score(ref, ref), 3))    ## 1.0   <- the greedy max sits on the diagonal
print(bert_score(ref[:3], ref) < 1.0)    ## True  <- P=1, but 2 reference tokens have no true counterpart -> R drops
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- Handles paraphrase, synonymy & word order that n-gram overlap cannot.
- Recall term catches omission, precision term catches padding → asymmetric errors are visible.

*Cons?*
- The score is a property of *(metric, encoder, layer)*. Two BERTScores computed with different encoders are different metrics.
- Raw values sit in a narrow band → rescaling is cosmetic; only differences within one config are meaningful.
- Still reference-bound, and still not factuality: negation and swapped entities barely move it.

*Why greedy matching and not an optimal assignment?*
- Greedy is $\mathcal{O}(|y||y^*|)$ with no solver, and it deliberately allows one reference token to be matched several times.
- Cost of that choice: **repetition is not penalized at all**. `"cat cat cat"` against reference `"cat"` gives $P=R=F_1=1$, since every greedy max is on the same reference token.
- → BERTScore cannot detect degenerate repetition; pair it with a repetition check.

*Which encoder layer?*
- A tuned hyperparameter, ❌automatically the last. The final layer is specialized for the pretraining head; intermediate layers transfer better.
- → Always publish the model + layer hash with the score.

*Why does IDF weighting help?*
- Function words match trivially and dominate the mean.
- → Weighting reference tokens by IDF shifts the score onto content words.
```

&nbsp;

### BLEURT
- **Name**: Bilingual Evaluation Understudy with Representations from Transformers {cite:p}`sellam2020bleurt`
- **What**: A BERT regression head fine-tuned on human ratings, after an intermediate pretraining stage on **synthetic** perturbation pairs.
- **Why**: Learned metrics need human ratings, and human ratings are scarce, domain-bound & drift year to year.
    - → A metric fit directly on them overfits the rating years it saw and collapses out of distribution.
    - Millions of cheap synthetic pairs can teach the general notion of "how bad is this corruption" before the scarce real labels are ever touched.
- **How**:
    1. Perturb Wikipedia sentences (mask-fill, backtranslate, drop words) → millions of (original, perturbed) pairs.
    2. Pretrain on **automatic** signals for those pairs (BLEU, ROUGE, BERTScore, backtranslation likelihood, entailment).
    3. Fine-tune on human ratings (WMT).

```{attention} Q&A
:class: dropdown
*How does it differ from COMET?*
- BLEURT scores (reference, candidate) only; COMET also consumes the **source**, so it catches mistranslations that look fine against the reference.
- BLEURT's contribution is the synthetic pretraining stage; COMET's is the source-aware feature construction.

*Cons?*
- Same as every learned metric: checkpoint-dependent, unauditable, GPU-bound, and gameable when optimized against.
- English-centric by default; the multilingual checkpoints lag COMET's language coverage.

*Why does synthetic pretraining actually help?*
- The synthetic targets are cheap automatic metrics — individually weak, but as a **multi-task** pretraining signal they force the encoder to represent the many axes on which two sentences can differ.
- → Fine-tuning then only has to learn the mapping from those axes onto the human scale, which is a far smaller problem than learning both from a few thousand ratings.
```

&nbsp;

### COMET
- **Name**: Crosslingual Optimized Metric for Evaluation of Translation {cite:p}`rei2020comet`
- **What**: A regression head over a multilingual encoder, trained to predict human translation-quality judgments.
- **Why**: Every unsupervised metric is a *proxy* for human judgment; if human judgments exist, regress on them directly.
    - Lexical & embedding metrics correlate poorly with humans once MT output is fluent — the remaining differences are semantic.
    - Access to the **source** lets the metric catch mistranslations that look fine against the reference.
- **How**:
    1. Encode source, hypothesis & reference with a shared multilingual encoder (XLM-R).
    2. Pool each to a sentence vector.
    3. Build interaction features (element-wise products & absolute differences).
    4. Feed-forward → scalar, trained w/ MSE against human scores.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{s}$: Pooled source embedding.
    - $\mathbf{h}$: Pooled hypothesis embedding.
    - $\mathbf{r}$: Pooled reference embedding.
- Misc:
    - $\odot$: Element-wise product.

Estimator features & head:

$$
\mathbf{x}=[\mathbf{h};\mathbf{r};\mathbf{h}\odot\mathbf{s};\mathbf{h}\odot\mathbf{r};|\mathbf{h}-\mathbf{s}|;|\mathbf{h}-\mathbf{r}|]
$$

$$
\hat{q}=\text{FFN}(\mathbf{x}),\quad\mathcal{L}=(\hat{q}-q_\text{human})^2
$$
- $q_\text{human}$: Human quality score (Direct Assessment, or MQM-derived).

Reference-free (QE) variant: drop $\mathbf{r}$ → score from $(\mathbf{s},\mathbf{h})$ alone.
```

```{attention} Q&A
:class: dropdown
*Pros?*
- Far higher correlation with human MT judgments than BLEU/chrF; the top-ranked family in the WMT metrics task since 2021.
- The QE variant needs no reference → usable online, in production, on live traffic.

*Cons?*
- Scores are **not absolute** and not comparable across COMET checkpoints. Always report the model name.
- Only as good as its training data's languages, domains & annotation protocol.
- ❌Reproducible from a spec, only from a checkpoint → a metric you cannot audit.
- GPU-bound → too expensive for inner-loop use.

*Why is optimizing against it dangerous?*
- Reranking or MBR decoding against COMET turns it into a [reward model](post/rl.md#reward) → the same over-optimization that hits RLHF.
- → Tune with COMET, report with COMET **and** a deterministic metric.

*When does a learned metric fail hardest?*
- Out of distribution: low-resource languages, unusual domains, and outputs far better or far worse than anything in its training range.
- Its errors are systematic, not random → they do not average out across a test set.
```

&nbsp;

## Model-Based
### LLM-as-a-Judge
- **What**: A strong LM scores or ranks outputs in place of a human annotator. {cite:p}`zheng2023judging`
- **Why**: Open-ended instruction following has no reference to overlap with and no verifier to run.
    - Human evaluation is the ground truth, but costs days and $ per model revision → ❌inner loop.
    - GPT-4 as judge agrees with human experts & crowdworkers **>80%** on MT-Bench — the same rate at which humans agree with each other.
- **How**:
    1. **Pairwise**: two responses, judge picks a winner (or tie) → win rate / [Bradley-Terry](#arena) score.
    2. **Pointwise**: one response, absolute score on a rubric.
    3. **Reference-guided**: gold answer given to the judge → mandatory for math & reasoning grading.

```{dropdown} Table: Judge Biases
| Bias | What | Fix |
|:--|:--|:--|
| **Position** | Prefers whichever response sits in a given slot | Score both orders; count a win only if consistent |
| **Verbosity** | Prefers longer answers | [Length-controlled win rate](#length-controlled-win-rate) |
| **Self-enhancement** | Prefers outputs from its own family | Judge ∉ evaluated models; use a jury of judges |
| **Limited reasoning** | Unreliable outside its own competence | Reference-guided grading, or an execution verifier |
| **Scale compression** | Pointwise scores pile up at 8–9/10 | Pairwise, or [G-Eval](#g-eval) probability weighting |
```

```{attention} Q&A
:class: dropdown
*Pros?*
- Applies to any open-ended task with no reference and no verifier — the majority of what LLMs are used for.
- Cheap & fast enough to sit in a development loop.

*Cons?*
- The judge's own competence bounds where it is reliable. Verification is often easier than generation, so a judge can grade above its own generation ceiling — but outside its competence its errors become systematic, not random.
- API judges are **non-stationary**: the endpoint silently changes version → scores drift → historical numbers stop being comparable. Pin the version & publish it.
- Optimizing against a judge is Goodhart's law with extra steps: the judge becomes a reward model and gets hacked.

*Pairwise or pointwise?*
- Pairwise: more reliable ← relative judgments are easier & the judge's scale drift cancels. Exhaustive round-robin is $\mathcal{O}(\#\text{models}^2)$, but Bradley-Terry only needs a **connected** comparison graph, so a sparse sampled subset suffices.
- Pointwise: $\mathcal{O}(\#\text{models})$, and a new model can be scored without rerunning baselines. Its absolute scale means nothing across judges.
- → Pointwise for regression tracking, pairwise for ranking claims.

*Why is ">80% agreement with humans" weaker than it sounds?*
- The human–human ceiling on the same data is also ~80% → the judge is at the ceiling **of a noisy label**, not at truth.
- Agreement is measured on the arena's prompt distribution; it does not transfer to specialist domains.

*What breaks when judge and candidate share a family?*
- Self-enhancement bias becomes a systematic, not random, error → it survives averaging over any number of prompts.
```

&nbsp;

#### G-Eval
- **What**: CoT-generated evaluation steps + form filling, scored by the **probability-weighted average** of the allowed score tokens. {cite:p}`liu2023geval`
- **Why**: A judge asked for an integer 1–5 returns nearly the same integer every time.
    - → Massive ties → almost no ranking signal, whatever the underlying quality differences.
    - The logits already contain the finer signal that sampling one integer throws away.
- **How**:
    1. Give the judge the task definition & evaluation criteria.
    2. Have it generate the detailed evaluation steps (CoT).
    3. Fill the form → read the probability of each score token.
    4. Expected score under that distribution.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\mathcal{S}$: Allowed score set, e.g. $\{1,...,5\}$.

$$
\text{score}=\sum_{s\in\mathcal{S}}p(s)\cdot s,\qquad p(s)=\frac{\pi_\text{judge}(s|\text{prompt})}{\sum_{s'\in\mathcal{S}}\pi_\text{judge}(s'|\text{prompt})}
$$

Renormalization over $\mathcal{S}$ ← the judge's mass on non-score tokens is discarded.
```

```{attention} Q&A
:class: dropdown
*Result?*
- Spearman **0.514** with human judgments on summarization (SummEval), well above lexical & embedding metrics.

*Requirement?*
- Token log-probs for the score tokens → ❌APIs that hide them.
- Fallback: sample the judge $k$ times and average, which estimates the same expectation with far more calls.

*Known failure the authors report themselves?*
- G-Eval prefers LLM-generated summaries over human-written ones that human annotators rated higher.
- → A judge-based metric embeds the judge's stylistic preferences, and those preferences favor machine text.

*Why is the weighted average not just a smoothing trick?*
- Under a discrete 1–5 rubric the true quality is continuous; the integer is a quantization.
- → The probability weights recover the sub-integer position, which is exactly what a rank correlation needs.
```

&nbsp;

#### Length-Controlled Win Rate
- **What**: Pairwise win rate with the judge's length preference regressed out. {cite:p}`dubois2024length`
- **Why**: Verbosity bias makes raw win rate gameable — appending "be more detailed" to the system prompt raises AlpacaEval win rate with no change in quality.
- **How**:
    1. Fit a GLM predicting the judge's preference from model identity, length difference & instruction difficulty.
    2. Report the model term **evaluated at zero length difference** → the counterfactual "if both answers were equally long".

```{attention} Q&A
:class: dropdown
*Result?*
- Spearman correlation with Chatbot Arena rises to **0.98** (raw AlpacaEval ≈0.93–0.94), and the verbose-system-prompt exploit stops working.

*Why regress instead of truncating or length-matching outputs?*
- Truncation changes the answer being judged. Regression leaves the data untouched and removes the confound statistically.

*What it does NOT fix?*
- Every other style confound — markdown, headers, bullet lists, confident tone. [Arena](#arena) adds those as extra regressors.

*The assumption it makes?*
- That length is pure bias. But a complete answer to a hard question **is** longer → the correction also removes some genuine signal.
- → It is a debiased estimate, not a neutral one; report both.
```

&nbsp;

## Human
### Arena
- **What**: Latent per-model strength fit by MLE to crowdsourced pairwise votes on user-submitted prompts. {cite:p}`chiang2024chatbot`
- **Why**: For open-ended chat there is no reference, no verifier & no static test set that stays uncontaminated.
    - Live prompts from live users cannot be memorized in advance.
    - Only *relative* preference is observable — absolute quality of a chat response is undefined.
- **How**:
    1. User submits a prompt → 2 anonymous models answer → user votes.
    2. Fit Bradley-Terry by logistic regression over all votes at once.
    3. Bootstrap over votes → confidence intervals.
    4. **Style control**: add response length & markdown features as regressors → read off the model coefficient alone.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\xi_i$: Latent strength of model $i$ (BT coefficient).
- Misc:
    - $\mathbf{z}\in\{-1,0,1\}^{\#\text{models}}$: Battle indicator, $+1$ for model A, $-1$ for model B, 0 elsewhere.
    - $y\in\{0,\frac{1}{2},1\}$: Vote label — A loses / tie / A wins.
    - $R_i$: Displayed Elo-scale rating.

Bradley-Terry:

$$
P(A\succ B)=\sigma(\xi_A-\xi_B)=\sigma(\mathbf{z}^\top\boldsymbol{\xi})
$$

→ The MLE is exactly logistic regression on $\mathbf{z}$ with $y$ as the label; a tie enters as $y=\frac{1}{2}$. $\boldsymbol{\xi}$ is identified only up to a shift → centre it.

Elo display scale:

$$
R_i=\frac{400}{\ln10}\xi_i+C\quad\Leftrightarrow\quad P(A\succ B)=\frac{1}{1+10^{(R_B-R_A)/400}}
$$
```

````{important} Code
:class: dropdown
```python
import numpy as np

def fit_bt(battles, n_models, lr=1.0, steps=20000):
    ## battles: (i, j, y) -> model i vs model j; y = 1 win, 0 loss, 0.5 tie
    xi = np.zeros(n_models)
    for _ in range(steps):
        g = np.zeros(n_models)
        for i, j, y in battles:
            p = 1 / (1 + np.exp(-(xi[i] - xi[j])))   ## P(i beats j)
            g[i] += y - p                            ## plain logistic-regression gradient
            g[j] -= y - p                            ## ... and its mirror image for the opponent
        xi += lr * g / len(battles)
    return xi - xi.mean()   ## BT is shift-invariant -> centre so the scale is anchored

## Example: A beats B 2 of 3, B beats C 2 of 3
battles = [(0, 1, 1), (0, 1, 1), (0, 1, 0), (1, 2, 1), (1, 2, 1), (1, 2, 0)]
xi = fit_bt(battles, 3)
print(np.round(xi, 4))                                    ## [ 0.6931  0.  -0.6931]  == logit(2/3) = ln 2
print(np.round(400 / np.log(10) * xi + 1000, 1))          ## [1120.4 1000.   879.6]
```
````

```{attention} Q&A
:class: dropdown
*Why BT MLE instead of sequential Elo?*
- Online Elo depends on the order battles arrive and on the K-factor → the same data gives different ratings.
- BT MLE uses all votes at once → order-independent, reproducible, and admits bootstrap CIs.

*What does "+20 Elo" actually mean?*
- $\frac{1}{1+10^{-20/400}}\approx0.529$ → a 52.9% win rate. A 200-point gap is ≈76%.
- → Small Elo deltas are small preference deltas, not qualitative jumps.

*Pros?*
- ❌Reference, ❌verifier, ❌static test set → far less exposed to benchmark contamination than any fixed suite.
- Measures the thing users actually care about, and ships with uncertainty.

*Cons?*
- The prompt distribution is whatever anonymous users type → short, casual, easy; ❌coverage of hard specialist work.
- ❌Contamination-proof: popular public benchmark questions get pasted in, and labs optimize toward the Arena's prompt distribution.
- Style, verbosity & formatting bias → hence style control.
- Gameable: models are identifiable from their formatting, and an org can vote for itself.
- Purely relative & slow: a new model needs thousands of votes before its CI separates it from anything.

*Why is style control not just cosmetic?*
- Removing length & markdown regressors reorders the leaderboard — models that won on presentation drop, concise ones rise.
- → Part of the raw ranking was measuring formatting, not capability.
```

&nbsp;

## IR
- **What**: Rank quality of the context a RAG system retrieves before generating.
- **Why**: RAG failures split cleanly in two — the passage was never retrieved, or it was retrieved and ignored. Only retrieval metrics separate them.

&nbsp;

### Recall@k
- **What**: Fraction of relevant documents appearing in the top $k$.
- **Why**: The **ceiling** on a RAG system: the generator cannot use a passage that was never retrieved.
- **How**: $\frac{|\{\text{relevant}\}\cap\{\text{top-}k\}|}{|\{\text{relevant}\}|}$, averaged over queries.

```{attention} Q&A
:class: dropdown
*Why recall and not precision for RAG?*
- A long-context generator tolerates a few irrelevant passages; it cannot recover a missing one.
- [Precision@k](#precision-k) only starts to matter once $k$ hits the context, latency or cost budget.

*Single-gold case?*
- With exactly one relevant document, Recall@$k$ = hit rate = "is the gold in the top $k$".

*What it hides?*
- Position. Recall@10 is identical whether the gold sits at rank 1 or rank 10 — but the generator's attention is not.
- → Pair with [MRR](#mrr) or [nDCG](#ndcg).
```

&nbsp;

#### Precision@k
- **What**: Fraction of the top $k$ results that are relevant.
- **Why**: Recall ignores what else came along; once $k$ is fixed by a context or latency budget, the junk in those $k$ slots is pure cost.
- **How**: $\frac{|\{\text{relevant}\}\cap\{\text{top-}k\}|}{k}$, averaged over queries.

```{attention} Q&A
:class: dropdown
*Why is it weak on its own?*
- Its ceiling is $\frac{\#\text{relevant}}{k}$ → a query with 1 relevant document caps Precision@10 at 0.1, no matter how good the ranker is.
- → Never compare Precision@$k$ across queries with different numbers of relevant documents.

*Why does it matter more for RAG than classic search?*
- Irrelevant passages are not free: they consume context, add latency & cost, and actively distract the generator.
```

&nbsp;

### MRR
- **Name**: Mean Reciprocal Rank
- **What**: Mean of $\frac{1}{\text{rank}}$ of the **first** relevant result.
- **Why**: When only the top hit is read, the only question is how far down the first correct answer sits.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\mathcal{Q}$: Query set.
    - $\text{rank}_q$: Rank of the first relevant result for query $q$; the term is 0 if none appears within $k$.

$$
\text{MRR@}k=\frac{1}{|\mathcal{Q}|}\sum_{q\in\mathcal{Q}}\frac{1}{\text{rank}_q}
$$
```

```{attention} Q&A
:class: dropdown
*Cons?*
- Every relevant document after the first is ignored → ❌multi-hop, ❌multi-evidence RAG.
- The reciprocal is steep: rank 1 vs 2 costs 0.5, rank 9 vs 10 costs 0.011 → the metric is almost entirely about the top 2–3 slots.

*Where it is the official metric?*
- MS MARCO passage ranking reports MRR@10.
```

&nbsp;

### MAP
- **Name**: Mean Average Precision
- **What**: Mean over queries of the average of Precision@$i$ taken at every rank $i$ holding a relevant document.
- **Why**: Precision@$k$ needs an arbitrary $k$, and Recall@$k$ ignores order.
    - Averaging precision **at each hit** folds the whole ranking into one number with no cutoff to choose.
- **How**: Walk the ranking; each time a relevant document appears, record the running precision; average those; average over queries.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $P@i$: Precision at rank $i$.
    - $rel_i\in\{0,1\}$: Binary relevance at rank $i$.
    - $R_q$: #relevant documents for query $q$.

$$
\text{AP}_q=\frac{1}{R_q}\sum_{i=1}^{k}P@i\cdot rel_i,\qquad\text{MAP}=\frac{1}{|\mathcal{Q}|}\sum_{q\in\mathcal{Q}}\text{AP}_q
$$
```

```{attention} Q&A
:class: dropdown
*MAP vs nDCG?*
- MAP is **binary** relevance; nDCG is graded. MAP is therefore the wrong metric whenever "somewhat relevant" is a real category.
- MAP has no explicit position discount — the discount is implicit, via the precision at each hit.

*Why does it still show up?*
- It is the classic single-number TREC/BEIR summary of a whole ranking, and it is the standard answer to "how do you score a ranker without picking $k$".
```

&nbsp;

### nDCG
- **Name**: normalized Discounted Cumulative Gain {cite:p}`jarvelin2002cumulated`
- **What**: Position-discounted sum of **graded** relevance, divided by its ideal ordering.
- **Why**: Binary, position-blind metrics cannot express "highly relevant at rank 1 beats slightly relevant at rank 1".
    - Raw DCG is incomparable across queries ← queries have different numbers of relevant documents → normalize by the best achievable DCG.
- **How**:
    1. Gain per result from its relevance grade.
    2. Discount by rank.
    3. Sum to $k$ → DCG.
    4. Divide by IDCG, the DCG of the perfectly ordered list of **all judged documents**.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $rel_i$: Graded relevance of the result at rank $i$.
    - $\text{IDCG@}k$: DCG of the relevance-sorted ideal ranking, built from **all judged documents** for the query — ❌only the retrieved ones, or a run that misses everything good still normalizes to 1.

$$
\text{DCG@}k=\sum_{i=1}^{k}\frac{2^{rel_i}-1}{\log_2(i+1)},\qquad\text{nDCG@}k=\frac{\text{DCG@}k}{\text{IDCG@}k}\in[0,1]
$$

The original linear-gain form uses $rel_i$ in place of $2^{rel_i}-1$; both are in active use → always state which.
```

````{important} Code
:class: dropdown
```python
import numpy as np

def recall_at_k(ranked_rel, n_relevant, k):
    ## ranked_rel: relevance grades in retrieved order
    return np.count_nonzero(ranked_rel[:k]) / n_relevant

def mrr_at_k(ranked_rel, k):
    hits = np.flatnonzero(ranked_rel[:k])
    return 1.0 / (hits[0] + 1) if hits.size else 0.0   ## only the FIRST hit counts

def ndcg_at_k(ranked_rel, all_grades, k):
    def dcg(rel):
        ## 2^rel - 1 weights high relevance disproportionately vs several mediocre hits
        return np.sum((2.0 ** rel - 1) / np.log2(np.arange(2, rel.size + 2)))
    ## IDCG must come from ALL judged docs, not just the retrieved ones -- otherwise a run
    ## that misses every good document still normalizes to 1.0
    ideal = np.sort(all_grades)[::-1][:k]
    return dcg(ranked_rel[:k]) / dcg(ideal) if dcg(ideal) else 0.0

## Example: 2 relevant docs, the highly relevant one buried at rank 3
rel = np.array([0, 1, 3, 0, 0])
grades = np.array([3, 1, 0, 0, 0])        ## every judged doc's grade for this query
print(round(recall_at_k(rel, 2, 3), 3))            ## 1.0    <- both retrieved, position ignored
print(round(mrr_at_k(rel, 3), 3))                  ## 0.5    <- first hit at rank 2
print(round(ndcg_at_k(rel, grades, 3), 3))         ## 0.541  <- penalized for the ordering

## Same query, but the grade-3 doc was never retrieved
missed = np.array([0, 1, 0, 0, 0])
print(round(ndcg_at_k(missed, grades, 3), 3))      ## 0.083  <- 0.631 if IDCG came from the run itself
```
````

```{attention} Q&A
:class: dropdown
*Why the exponential gain?*
- It weights relevance grades disproportionately, so a single highly relevant document dominates a handful of marginal ones.
- ❌Lexicographic: enough discounted low-grade documents still out-sum one high-grade document.

*Why $\log_2(i+1)$?*
- Discount 1 at rank 1, and a slow decay after → ranks 1–10 differ meaningfully while rank 50 is not zeroed.
- Any smooth decreasing function would do; the log is the convention, not a derivation.

*Cons?*
- Needs graded human labels → expensive, and grades drift between annotator pools.
- **Pooling bias**: unjudged documents count as irrelevant → a system that surfaces genuinely good documents nobody labelled is punished.
- nDCG@10 on one benchmark says nothing about nDCG@10 on another; the ideal ranking differs.

*Standard for retrieval eval?*
- BEIR reports nDCG@10 → that is the number to match when comparing embedding models for RAG.
```

&nbsp;

### RAG Groundedness
- **What**: Reference-free checks that the answer is supported by the retrieved context & actually addresses the question. {cite:p}`es2024ragas`
- **Why**: Retrieval metrics score the context; task metrics need a gold answer.
    - Production RAG has neither a gold answer nor any guarantee the generator used the context it was given.
- **How**: Each measure is itself computed by an [LLM judge](#llm-as-a-judge).
    1. **Faithfulness**: decompose the answer into atomic claims → fraction entailed by the retrieved context.
    2. **Answer relevance**: generate questions from the answer → mean similarity to the original question.
    3. **Context relevance**: fraction of retrieved sentences actually needed to answer.

```{attention} Q&A
:class: dropdown
*Faithfulness ≠ correctness.*
- An answer faithfully grounded in a wrong retrieved passage is perfectly faithful and wrong.
- → Faithfulness isolates *generator* hallucination; retrieval quality and factual accuracy are separate measurements.

*Why decompose into claims?*
- Entailment over a whole paragraph is dominated by its supported parts → one fabricated clause disappears into the average.
- Per-claim scoring makes a single hallucination visible.

*Cons?*
- An LLM judge underneath → inherits every [judge bias](#llm-as-a-judge), cost and version drift.
- Claim decomposition is brittle on long or highly structured answers.
- Optimizing a RAG system against it produces answers that quote the context and answer nothing.
```

&nbsp;
