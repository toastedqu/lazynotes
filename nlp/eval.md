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
NLP/LLM eval: how a score is produced, what it means, how it gets gamed.

Task-generic metrics (accuracy, F1, AUC, calibration) → [ML Evaluation](../ml/eval.md).

Notations:
- $x$: Input (prompt / query / source text).
- $y$: Candidate (output being scored).
- $y^*$: Reference (gold output).
- $y_t$: $t$-th token of $y$.
- $|y|$: Length of $y$ in tokens.
- $\mathcal{D}$: Eval set.
- $m$: #samples in $\mathcal{D}$.
- $\pi_\theta$: LM under eval (next-token distribution).
- $n$: N-gram order.
- $k$: Sample budget per input, or rank cutoff.

Override: $n$ = n-gram order (❌#features).

Override: $k$ = sample budget / rank cutoff (❌class idx).

&nbsp;

## Protocol
- **What**: Every eval choice between model & score, except the metric formula.
- **Why**: Same model + same data + same metric → different scores under different protocols.
    - → A score w/o its protocol is not reproducible.

&nbsp;

### Run Configuration
- **What**: Prompt, decoding & aggregation settings of one eval run.
- **Why**: Each setting moves the score w/o any change to the model.
- **How**: Fix & report each group:
    - **Prompt**: chat template, system prompt, #few-shot examples & their order, CoT on/off.
    - **Decoding**: temperature, top-p, max new tokens, stop sequences, seed.
    - **Aggregation**: micro (per item) vs macro (per subtask) mean, subtask weights, handling of failed or missing items.

```{attention} Q&A
:class: dropdown
*Why does the chat template matter?*
- Instruction-tuned model sees a format it was never trained on → output quality drops.
- → Base & chat models need different prompts for a fair comparison.

*Why does max new tokens matter?*
- Limit too low → long reasoning is truncated → no final answer → scored as wrong.
- → Reasoning models need a much larger budget than direct-answer models.

*Micro vs macro mean?*
- Subtasks of unequal size → the two means differ.
- Micro: large subtasks dominate. Macro: each subtask counts equally, small subtasks add noise.
- → State which one.

*Why fix the seed if decoding is stochastic?*
- Reproducibility of one run.
- Still report the spread over several seeds → see [Statistical Noise](#statistical-noise).
```

&nbsp;

### Scoring Mode
- **What**: Method to get the answer: rank fixed options by likelihood, or decode & parse.
- **Why**: A multiple-choice benchmark does not fix how the model is asked.
    - Likelihood scoring needs logits → ❌closed APIs.
    - Generative scoring needs a parser → format failure scores as wrong.
    - → One benchmark name, several incompatible numbers.
- **How**:
    1. **Cloze**: score each full answer string as a continuation of the question → argmax.
    2. **Letter**: score only `" A"`/`" B"`/`" C"`/`" D"` after a lettered prompt → argmax.
    3. **Generative**: decode (usually w/ CoT) → [extract](#answer-extraction) the answer.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $o$: One candidate answer string, tokenized.
- Misc:
    - $B(o)$: #UTF-8 bytes of $o$.

Option score:

$$
s(o)=\log\pi_\theta(o|x)=\sum_{t=1}^{|o|}\log\pi_\theta(o_t|x,o_{<t})
$$

Normalizations, all in use:

$$\begin{align*}
s_\text{token}(o)&=\frac{s(o)}{|o|}\\
s_\text{byte}(o)&=\frac{s(o)}{B(o)}\\
s_\text{PMI}(o)&=s(o)-\log\pi_\theta(o|x_\text{null})
\end{align*}$$
- $x_\text{null}$: Content-free prompt (e.g., `"Answer:"`) → subtracts the option's prior.
```

````{important} Code
:class: dropdown
```python
import torch

class OptionScorer:
    def __init__(self, norm="token"):
        self.norm = norm   ## "none" | "token" | "byte" | "pmi"

    def __call__(self, logp, n_bytes=None, logp_null=None):
        ## logp: one (|o|,) tensor of per-token log-probs per option, conditioned on x
        s = torch.stack([lp.sum() for lp in logp])                  ## s(o) = log pi(o|x)
        if self.norm == "token":
            s = s / torch.tensor([float(len(lp)) for lp in logp])
        elif self.norm == "byte":
            s = s / torch.tensor(n_bytes, dtype=s.dtype)
        elif self.norm == "pmi":
            ## same options scored after a content-free prompt -> remove each option's prior
            s = s - torch.stack([lp.sum() for lp in logp_null])
        return s.argmax().item()

## Example: option 0 = short & wrong, option 1 = long & right
logp = [torch.tensor([-1.0]), torch.tensor([-0.4, -0.4, -0.4])]
print(OptionScorer("none")(logp))    ## 0  <- raw sum: -1.0 > -1.2, short option wins
print(OptionScorer("token")(logp))   ## 1  <- per-token mean: -1.0 < -0.4
```
````

````{attention} Q&A
:class: dropdown
*Why normalize?*
- Each token adds a term $\log\pi_\theta<0$ → longer option scores lower, independent of correctness.
- → Raw $s(o)$ is biased toward short options.

*Is length normalization always right?*
- ❌. Options w/ a long shared prefix → easy shared tokens inflate the mean.
- PMI removes the option's prior → corrects the common-vs-rare answer-string bias, which length normalization does not.

*How large is the protocol effect?*
- LLaMA-65B on MMLU, same weights, same 14,042 questions {cite:p}`fourrier2023mmlu`:
    - Original (Berkeley) implementation: 63.6.
    - HELM: 63.7.
    - EleutherAI harness (2023 commit): 48.8.
- LLaMA paper reports 63.4 ← original implementation.
- → "MMLU = 60" is not a model fact until the harness & commit are named.

*Which mode do modern reports use?*
- Mostly generative ← reasoning models must think before they commit.
- Cloze scoring forbids CoT → it measures a different quantity for the same model.
````

&nbsp;

### Answer Extraction
- **What**: Parse the committed final answer out of free-form generation.
- **Why**: A grader compares answers; a CoT trace is not an answer.
    - Extraction failure → scored as wrong → format failure counts as capability failure.
- **How**:
    1. Constrain the format in the prompt: `\boxed{}`, `Answer: X`, or a JSON schema w/ [constrained decoding](infer/dec.md#constrained-decoding).
    2. Take the last match ← models restate & revise mid-trace.
    3. Normalize: case, punctuation, articles, units, LaTeX, `1/2`$\Leftrightarrow$`0.5`.
    4. Check equivalence, ❌string equality: symbolic (SymPy) for math, unit tests for code.

````{important} Code
:class: dropdown
```python
import re

class AnswerExtractor:
    def __init__(self, patterns=(r"\\boxed\{([^{}]*)\}", r"[Aa]nswer:\s*([^\n]+)")):
        self.patterns = [re.compile(p) for p in patterns]

    def normalize(self, s):
        return s.strip().rstrip(".").replace(",", "").lower()   ## "1,000." -> "1000"

    def __call__(self, text):
        ## (end position, answer) for every match of every pattern
        hits = [(m.end(), m.group(1)) for p in self.patterns for m in p.finditer(text)]
        if not hits:
            return None   ## extraction failure: count separately, do not merge into "wrong"
        return self.normalize(max(hits)[1])   ## LAST match in the text <- the model revised its answer

## Example
trace = r"First guess \boxed{12}. Recheck: 3*5=15. Final: \boxed{15}."
print(AnswerExtractor()(trace))          ## 15
print(AnswerExtractor()("I think 42"))   ## None
```
````

```{attention} Q&A
:class: dropdown
*What silently inflates scores?*
- Unanchored substring search: `"the answer is not 42"` matches `42`.
- Accept any number in the trace → an arithmetic scratchpad almost always contains a hit.

*What silently deflates scores?*
- A correct answer in the wrong format.
- → Report extraction-failure rate separately from error rate. Otherwise a prompt-format change looks like a capability change.

*Why is extraction a reproducibility problem?*
- Extraction rules are code, not a published spec.
- → Two labs that grade the same generations can disagree.
```

&nbsp;

### Contamination
- **What**: Eval items, or near-duplicates, inside the training corpus.
- **Why**: Score → measures memorization, ❌capability.
    - Benchmarks are public web text → leakage is the default.
- **How**:
    1. **Corpus-side**: n-gram overlap between eval items & training data → needs corpus access → ❌closed models.
    2. **Canaries**: embed a fixed GUID in the benchmark → ask the model to complete it.
    3. **Behavioral**: score a new, distribution-matched twin of the benchmark → compare.
    4. **Temporal**: date-partition → admit only post-cutoff items → refresh continuously.

```{attention} Q&A
:class: dropdown
*Evidence that it matters?*
- GSM1k: 1,205 new grade-school math problems, matched to GSM8K's distribution. {cite:p}`zhang2024careful`
- Accuracy drops up to 8pp vs GSM8K.
- Several model families overfit across almost all sizes. Frontier models show little drop.

*Is a GSM8K→GSM1k gap proof of literal leakage?*
- ❌. Repeated checkpoint & hyperparam selection on a benchmark overfits its style, w/o any test item in training.
- → The gap measures overfitting-to-benchmark. Memorization is one cause.

*Why doesn't decontamination solve it?*
- Filters strip only what they can match → paraphrases, translations & reformats survive n-gram filters.
- Corpora are re-crawled → a benchmark released today is in next year's pretraining data.

*Practical consequence?*
- Static benchmark absolute score → treat as a possibly inflated estimate.
- Trust relative movement on newly collected items.
```

&nbsp;

### Statistical Noise
- **What**: Sampling error of a benchmark score.
- **Why**: Score = sample mean → it has a standard error.
    - Sub-point leaderboard gaps → usually inside the noise.
    - Two sources: finite #items, stochastic decoding.
- **How**:
    1. Item-sampling SE from the binomial.
    2. Repeat w/ different seeds → adds decoding variance.
    3. **Paired** comparison on identical items → cancels item difficulty → much tighter than two independent CIs.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\hat{p}$: Observed accuracy.

Item-sampling standard error:

$$
\text{SE}=\sqrt{\frac{\hat{p}(1-\hat{p})}{m}}
$$

Variance of a paired difference between models $A$ & $B$:

$$
\text{Var}(\hat{p}_A-\hat{p}_B)=\text{Var}(\hat{p}_A)+\text{Var}(\hat{p}_B)-2\text{Cov}(\hat{p}_A,\hat{p}_B)
$$
- Same items → shared item difficulty → usually $\text{Cov}>0$ → paired variance < sum of marginals.
- $\text{Cov}\leq0$ → pairing gives no gain.
```

```{note} Example
:class: dropdown
MMLU, $m=14{,}042$, worst case $\hat{p}=0.5$:

$$
\text{SE}=\sqrt{\frac{0.25}{14042}}\approx0.42\%\quad\Rightarrow\quad\text{95\% CI}\approx\pm0.83\text{pp}
$$

Per-subject breakdown, $m\approx250$ → $\text{SE}\approx3.2\%$.
```

````{important} Code
:class: dropdown
```python
import numpy as np

class PairedBootstrap:
    def __init__(self, n_boot=2000, seed=0):
        self.n_boot = n_boot
        self.rng = np.random.default_rng(seed)

    def __call__(self, a, b):
        ## a, b: (m,) per-item 0/1 outcomes of two models on the SAME items
        d = a - b                                                   ## pairing cancels item difficulty
        idx = self.rng.integers(0, len(d), (self.n_boot, len(d)))   ## resample items, keep pairs intact
        boot = d[idx].mean(axis=1)
        return d.mean(), np.percentile(boot, [2.5, 97.5])

## Example: B solves all 600 items A solves, plus 30 more, out of 1000
a = np.array([1.0] * 600 + [0.0] * 400)
b = a.copy(); b[600:630] = 1.0
## independent 95% CIs: A = [0.570, 0.630], B = [0.600, 0.660] -> they overlap
diff, ci = PairedBootstrap()(b, a)
print(round(diff, 3), ci[0] > 0)   ## 0.03 True  <- paired CI excludes 0
```
````

```{attention} Q&A
:class: dropdown
*Why are per-subject breakdowns nearly meaningless?*
- SE ∝ $1/\sqrt{m}$ → a 100-item slice has SE ≈5pp.
- → Ranking models on one MMLU subject ranks noise.

*Overlapping CIs → no difference?*
- ❌. Both models answer the same items → errors are usually positively correlated → $\text{Cov}>0$.
- → Paired difference is then much tighter than the marginal CIs suggest.
- → Use a paired test (McNemar, or bootstrap over items) on per-item outcomes.

*Do error bars fix leaderboards?*
- Only partly. They cover sampling error.
- ❌Selection error from testing many checkpoints against the same public test set.
```

&nbsp;

## Likelihood
- **What**: Metrics computed from the LM's own token probabilities on held-out text.

&nbsp;

### Perplexity
- **What**: Exponentiated mean per-token NLL on held-out text.
- **Why**: Score raw text w/o labels, references, or judge.
    - Same quantity as the pretraining loss → tracks a training run directly.
- **How**:
    1. Teacher-force the held-out text.
    2. Mean NLL over its tokens.
    3. Exponentiate.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\mathcal{V}$: Vocab.
    - $H$: Cross-entropy in nats/token.
    - $L$: Context window.
    - $s$: Stride for long-document eval.

Definition:

$$
\text{PPL}(y)=\exp\left(-\frac{1}{|y|}\sum_{t=1}^{|y|}\log\pi_\theta(y_t|y_{<t})\right)=\exp(H)
$$

Properties:
- Reading: effective #equally likely tokens the model hesitates between at each step.
- $\text{PPL}=1$ → perfect model. $\text{PPL}=|\mathcal{V}|$ → uniform model.
- Unbounded above → a model worse than uniform exceeds $|\mathcal{V}|$.
- Document longer than $L$ → slide a window of size $L$ w/ stride $s<L$ → score only its last $s$ tokens.
    - Every scored token then has $\geq L-s$ tokens of context. Cost $\times L/s$.
```

````{important} Code
:class: dropdown
```python
import math
import torch
import torch.nn.functional as F

class Perplexity:
    def nll(self, logits, targets):
        ## logits: (T, V) next-token logits; targets: (T,) tokens that actually followed
        return F.cross_entropy(logits, targets, reduction="mean").item()   ## nats / token

    def __call__(self, logits, targets):
        return math.exp(self.nll(logits, targets))

## Example: model puts p=0.5 on the correct token at all 4 steps
logits = torch.log(torch.tensor([[0.5, 0.2, 0.2, 0.1]] * 4))
targets = torch.tensor([0, 0, 0, 0])
print(round(Perplexity()(logits, targets), 2))   ## 2.0  <- hesitating between 2 tokens
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- ❌Labels, ❌references, ❌judge → runs on any raw corpus.
- Continuous & low-variance → fine-grained enough to track a pretraining run.

*Cons?*
- ❌Comparable across tokenizers → use [BPB](#bits-per-byte).
- ❌Comparable across corpora: PPL on Wikipedia & PPL on code are unrelated numbers.
- ❌Measure of usefulness: instruction-tuned models usually have higher PPL on raw web text than their base model.
- Dominated by frequent, easy tokens → nearly blind to rare tokens that carry the meaning.

*Why exponentiate?*
- Puts cross-entropy on the branching-factor scale → uniform model = $|\mathcal{V}|$ → interpretable.
- Multiplicative gains read as ratios.

*Naive chunking of a long document?*
- First token of each chunk has no context → PPL inflated.
- → Strided windows, or report the context length used.

*Why not compare a base model & its RLHF'd version by PPL?*
- Alignment moves the output distribution away from raw text on purpose.
- → Higher PPL is the intended effect, not a regression.
```

&nbsp;

#### Bits per Byte
- **What**: Cross-entropy of the text in bits per UTF-8 byte (BPB).
- **Why**: PPL's denominator = #tokens → the tokenizer chooses it, not the model.
    - Coarser tokenizer → fewer, harder tokens → higher PPL at identical modeling quality.
    - Bytes are tokenizer-invariant → fair cross-model LM comparison.
- **How**: Total NLL over the text → nats to bits → divide by #bytes.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $B$: #UTF-8 bytes of the text.

$$
\text{BPB}=\frac{1}{\ln2}\cdot\frac{\sum_{t=1}^{|y|}-\log\pi_\theta(y_t|y_{<t})}{B}=\frac{|y|}{B}\log_2\text{PPL}
$$
- $\frac{|y|}{B}$: Tokens per byte, i.e., inverse of the tokenizer's compression rate.
```

````{important} Code
:class: dropdown
```python
import math
## Reuses Perplexity, logits, targets from the Perplexity block above

class BitsPerByte:
    def __call__(self, logits, targets, n_bytes):
        total_nats = Perplexity().nll(logits, targets) * targets.numel()
        ## nats -> bits, then divide by BYTES -> denominator no longer depends on the tokenizer
        return total_nats / (math.log(2) * n_bytes)

## Example: same 4 tokens, spanning 8 bytes
print(round(BitsPerByte()(logits, targets, 8), 2))   ## 0.5  == (4/8) * log2(2.0)
```
````

```{attention} Q&A
:class: dropdown
*Why bits, not nats?*
- Uniform-over-bytes model costs exactly 8 bits/byte → BPB reads as a compression ratio vs raw UTF-8.
- 8 is ❌an upper bound: BPB is a cross-entropy → a badly mismatched model exceeds 8.
    - E.g., probability $2^{-16}$ on an observed byte costs 16 bits.

*Can a tokenizer game BPB?*
- ❌. Finer split → more terms in the numerator sum, $B$ fixed.
- A tokenizer improves BPB only if it makes prediction truly easier.

*BPC?*
- Bits per character: same construction, denominator = #characters.
- ❌Comparable across scripts: a CJK character = 3 UTF-8 bytes, an ASCII character = 1.

*Why is BPB the right metric for tokenizer research?*
- Scores tokenizer + model jointly on one fixed denominator.
- → Exactly the quantity a tokenizer change should improve.
```

&nbsp;

## Verifiable
- **What**: Metrics for tasks w/ checkable ground truth: short-form answers, math, code.

&nbsp;

### Exact Match
- **What**: Binary match between normalized prediction & any normalized gold. {cite:p}`rajpurkar2016squad`
- **Why**: A factual answer is right or wrong.
    - Overlap metrics give partial credit to wrong answers.
- **How**:
    1. Normalize both sides: lowercase, strip punctuation, articles, extra whitespace.
    2. Compare against every gold → max.

````{important} Code
:class: dropdown
```python
import re
import string

class ExactMatch:
    @staticmethod
    def normalize(s):
        ## SQuAD order: lowercase -> drop punctuation -> drop articles -> collapse whitespace
        s = "".join(ch for ch in s.lower() if ch not in string.punctuation)   ## deleted, not spaced
        s = re.sub(r"\b(a|an|the)\b", " ", s)   ## articles carry no answer content
        return " ".join(s.split())              ## collapse whitespace

    def __call__(self, pred, golds):
        return float(any(self.normalize(pred) == self.normalize(g) for g in golds))

## Example
print(ExactMatch()("The Eiffel Tower.", ["Eiffel Tower"]))   ## 1.0
print(ExactMatch()("0.5", ["1/2"]))                          ## 0.0  <- "05" vs "12": same value, different string
```
````

```{attention} Q&A
:class: dropdown
*Cons?*
- Equivalent answers score 0: `"0.5"` vs `"1/2"`, `"US"` vs `"United States"`.
- → Normalization rules become part of the benchmark, and papers rarely publish them.

*Why is it still the default for QA & math?*
- ❌Judge cost, ❌judge bias, ❌judge drift → same meaning across years & labs.

*Fix for math?*
- Symbolic equivalence (SymPy / `math-verify`), ❌string equality → $\frac{1}{2}$, $0.5$, $2^{-1}$ all match.

*Fix for open-ended short answers?*
- [Judge](#llm-as-a-judge) w/ the gold answer in context (reference-guided grading).
- Cost: the judge's biases come back.
```

&nbsp;

### Token F1
- **What**: F1 over the token multiset shared by prediction & gold, max over golds. {cite:p}`rajpurkar2016squad`
- **Why**: EM is all-or-nothing.
    - Extractive answers often differ only by a span boundary or an article → not a wrong answer.
- **How**: Both sides → bags of tokens → precision, recall, harmonic mean → max over golds → mean over $\mathcal{D}$.

```{note} Math
:class: dropdown
$$
P=\frac{|y\cap y^*|}{|y|},\quad R=\frac{|y\cap y^*|}{|y^*|},\quad F_1=\frac{2PR}{P+R}
$$
- $\cap$: Multiset intersection: $\min$ of the two counts per token type.

Definition & properties of F1 → [F1](../ml/eval.md#f1).
```

````{important} Code
:class: dropdown
```python
from collections import Counter
## Reuses ExactMatch from the Exact Match block above

class TokenF1:
    def f1(self, pred, gold):
        p = ExactMatch.normalize(pred).split()
        g = ExactMatch.normalize(gold).split()
        ## MULTISET intersection: a token repeated twice must appear twice on both sides
        shared = sum((Counter(p) & Counter(g)).values())
        if shared == 0:
            return 0.0
        prec, rec = shared / len(p), shared / len(g)
        return 2 * prec * rec / (prec + rec)

    def __call__(self, pred, golds):
        return max(self.f1(pred, g) for g in golds)   ## golds are alternatives -> best one

## Example: right answer, wrong string
pred, golds = "Barack Hussein Obama", ["Barack Obama"]
print(ExactMatch()(pred, golds), round(TokenF1()(pred, golds), 2))   ## 0.0 0.8
```
````

```{attention} Q&A
:class: dropdown
*When does it mislead?*
- Rambling answer that contains the gold span → recall = 1 → nonzero F1 where EM = 0.
- Precision penalizes the padding, but never to 0 → verbose models always get some unearned credit.
- → Report EM alongside.

*Why max over golds, not mean?*
- Multiple golds are alternative correct answers, not a set to cover.

*Why token-level, not character-level?*
- Extractive answers differ at the token level.
- Character overlap gives credit for coincidental substrings.
```

&nbsp;

### pass@k
- **What**: P(≥1 of $k$ sampled programs passes all unit tests). {cite:p}`chen2021evaluating`
- **Why**: Code has a verifier (execution) → no reference string needed.
    - A user can try $k$ times & check → one greedy sample understates usable capability.
- **How**:
    1. Sample $S\gg k$ completions per problem at a fixed temperature.
    2. Execute against hidden tests → count $c$ passing.
    3. Unbiased estimate of "≥1 of $k$ passes" → mean over problems.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $S$: #samples per problem, $S\geq k$. Source paper writes $n$; here $n$ = n-gram order.
- Misc:
    - $c$: #samples that pass all tests.
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

class PassAtK:
    def __init__(self, k):
        self.k = k

    def __call__(self, S, c):
        ## S: samples drawn per problem, c: samples that pass every test
        if S - c < self.k:
            return 1.0   ## fewer than k failures exist -> every k-subset contains a pass
        ## 1 - C(S-c, k)/C(S, k), as a product -> no binomial is ever materialized
        return 1.0 - np.prod(1.0 - self.k / np.arange(S - c + 1, S + 1))

## Example: 10 samples per problem, 2 pass
print(round(PassAtK(1)(10, 2), 3))   ## 0.2    == c/S
print(round(PassAtK(5)(10, 2), 3))   ## 0.778  <- 5 tries nearly quadruple it
```
````

```{attention} Q&A
:class: dropdown
*Why not $1-(1-\hat{p})^k$ with $\hat{p}=c/S$?*
- Plug-in estimate into a nonlinear function → biased: $\mathbb{E}[f(\hat{p})]\neq f(\mathbb{E}[\hat{p}])$.
- $f(p)=1-(1-p)^k$ is concave → plug-in underestimates (Jensen).
- Combinatorial form = exact unbiased estimator for a $k$-subset drawn from the $S$ samples.

*Assumptions?*
- All $S$ samples i.i.d. from one fixed distribution (fixed prompt & [temperature](infer/dec.md#temperature)).
- Test suite = sound verifier. Weak tests → wrong program passes → metric silently inflates.

*Why report both pass@1 & pass@k?*
- They prefer opposite temperatures: low $T$ maximizes pass@1, high $T$ maximizes pass@$k$ at large $k$.
- Codex-679M sweep: best $T\approx0.2$ for pass@1, $T\approx0.8$ for pass@100.
- → State the temperature with the score.

*Why can RL-tuned models gain pass@1 but lose pass@k at large $k$?*
- RL maximizes expected reward = pass@1 → policy sharpens onto one mode → diversity that pass@$k$ needs drops.
- Same phenomenon as [entropy collapse](post/rl.md#entropy-collapse).

*Does pass@k measure capability or search?*
- Search: capability of model + $k$ attempts + a perfect verifier.
- The verifier does selection that the model cannot do alone.
```

&nbsp;

#### avg@k
- **What**: Mean per-sample pass rate over $k$ samples.
- **Why**: Greedy pass@1 = one Bernoulli draw per problem → variance hides small differences.
    - Mean of $k$ stochastic samples → unbiased, lower-variance estimate of pass@1 at that temperature.
- **How**: $\text{avg@}k=\frac{c}{k}$ → mean over problems.

```{attention} Q&A
:class: dropdown
*Relation to pass@1?*
- $\mathbb{E}[\text{avg@}k]=\text{pass@}1$. It is ❌a $k$-attempt budget metric.

*Why does this matter when reading papers?*
- Many RLVR papers report avg@$k$ (e.g., $k=32$) under the label "pass@1".
- Two papers that both write "pass@1" can report greedy accuracy & a 32-sample mean.
- → Check $k$ & temperature before comparing.
```

&nbsp;

#### maj@k
- **What**: Accuracy of the plurality answer among $k$ sampled reasoning chains. {cite:p}`wang2022self`
- **Why**: pass@$k$ needs a verifier to pick the winner → agreement across samples is a verifier-free substitute.
    - Correct reasoning converges on one answer; wrong reasoning scatters.
- **How**: Sample $k$ chains → extract & canonicalize each final answer → take the mode → grade it. See [Self-Consistency](infer/dec.md#self-consistency).

````{important} Code
:class: dropdown
```python
from collections import Counter

class MajAtK:
    def __call__(self, answers, gold):
        ## answers: k canonicalized final answers; None = extraction failure -> no vote
        votes = Counter(a for a in answers if a is not None)
        if not votes:
            return 0.0
        return float(votes.most_common(1)[0][0] == gold)   ## ties -> first seen; fix a tie rule

## Example: 5 chains, 2 agree on the gold answer
answers = ["15", "12", "15", None, "7"]
print(MajAtK()(answers, "15"))   ## 1.0  (avg@5 = 0.4)
```
````

```{attention} Q&A
:class: dropdown
*Ordering?*
- maj@$k$ ≤ pass@$k$ on the same $k$ samples: correct mode → ≥1 correct sample.
- maj@$k$ ≥ avg@$k$ in practice, ❌guaranteed: fails when one wrong answer is systematically preferred.

*Requirement?*
- Canonicalizable final answer, so votes can be counted.
- → ❌Free-form text, ❌proofs, ❌code (two correct programs rarely match as strings).

*Why is maj@k the honest "no-verifier" number?*
- pass@$k$ assumes an oracle that recognizes correctness. Deployment has no oracle.
- → maj@$k$ = what a user actually gets from $k$ samples.
```

&nbsp;

## NLG (Lexical)
- **What**: Surface-form overlap between candidate & reference: n-grams, characters, or edits.
- **Why**: Cheap, deterministic, reproducible from a spec.

&nbsp;

### BLEU
- **Name**: Bilingual Evaluation Understudy {cite:p}`papineni2002bleu`
- **What**: Geometric mean of clipped n-gram precisions vs ≥1 references, × brevity penalty.
- **Why**: Human MT eval is slow & expensive → need a cheap, automatic, language-agnostic proxy.
- **How**:
    1. Count candidate n-grams for $n=1..N$.
    2. **Clip** each n-gram count at its max count in any single reference ← blocks precision gaming by repetition.
    3. Clipped precision $p_n$ per order.
    4. Geometric mean over orders.
    5. × **Brevity penalty** ← blocks precision gaming by very short output.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathcal{R}$: Reference set; $y^*\in\mathcal{R}$.
- Hyperparams:
    - $N$: Max n-gram order, default 4.
- Misc:
    - $\mathcal{G}_n(y)$: Set of distinct n-grams in $y$.
    - $\#_y(g)$: Count of n-gram $g$ in $y$.
    - $r$: Effective reference length: reference length closest to $|y|$; ties → shorter.

Clipped n-gram precision:

$$
p_n=\frac{\sum_{g\in\mathcal{G}_n(y)}\min\left(\#_y(g),\max_{y^*\in\mathcal{R}}\#_{y^*}(g)\right)}{\sum_{g\in\mathcal{G}_n(y)}\#_y(g)}
$$

Brevity penalty:

$$
\text{BP}=\begin{cases}
1 & |y|>r\\
e^{1-r/|y|} & |y|\leq r
\end{cases}
$$

BLEU:

$$
\text{BLEU}_N=\text{BP}\cdot\exp\left(\frac{1}{N}\sum_{n=1}^{N}\log p_n\right)
$$

Properties:
- Corpus-level: sum numerators, denominators, $|y|$ & $r$ over all segments, then divide.
- Any $p_n=0$ → $\text{BLEU}=0$.
```

````{important} Code
:class: dropdown
```python
import math
from collections import Counter

def ngrams(toks, n):
    return Counter(tuple(toks[i:i+n]) for i in range(len(toks) - n + 1))

class BLEU:
    def __init__(self, max_n=4):
        self.max_n = max_n

    def __call__(self, cand, refs):
        c, rs = cand.split(), [r.split() for r in refs]
        log_p = 0.0
        for n in range(1, self.max_n + 1):
            cand_ng = ngrams(c, n)
            max_ref = Counter()
            for r in rs:
                max_ref |= ngrams(r, n)                    ## | = per-n-gram max over references
            clipped = sum((cand_ng & max_ref).values())    ## & = per-n-gram min -> the clip
            if clipped == 0:
                return 0.0                                 ## one zero precision zeroes the geometric mean
            log_p += math.log(clipped / sum(cand_ng.values())) / self.max_n
        r = min((abs(len(x) - len(c)), len(x)) for x in rs)[1]   ## closest ref length, ties -> shorter
        bp = 1.0 if len(c) > r else math.exp(1 - r / len(c))
        return bp * math.exp(log_p)

## Example
ref = ["the cat is on the mat"]
print(round(BLEU(1)("the the the the the the the", ref), 3))   ## 0.286  <- "the" credited 2x, not 7x
print(round(BLEU(2)("the cat", ref), 3))                       ## 0.135  <- p1 = p2 = 1, BP = e^-2
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- Cheap, deterministic, language-agnostic.
- Correlates w/ human MT judgment at corpus level.

*Cons?*
- ❌Semantics: a correct paraphrase w/ no shared words scores near 0. Every lexical metric on this page shares this con.
- Exact surface match → ❌stems, ❌synonyms → see [METEOR](#meteor), [chrF](#chrf).
- Every n-gram has the same weight → frequent function words inflate the score.
- Word order enforced only inside n-gram windows → shuffled blocks keep most n-gram matches.

*Why corpus-level, not sentence-level?*
- Short segment → some $p_n=0$ (often $n=4$) → sentence BLEU = 0.
- Corpus BLEU pools counts before division → zeros vanish.
- Sentence BLEU needs smoothing → each smoothing method gives a different number.

*Why multiple references?*
- One source has many valid translations.
- Clip uses the max over references → wording from any reference earns credit.

*Why are BLEU scores from different papers often not comparable?*
- Tokenization & normalization change the score.
- → sacreBLEU fixes both & prints a version signature. Report it.
```

&nbsp;

### ROUGE
- **Name**: Recall-Oriented Understudy for Gisting Evaluation {cite:p}`lin2004rouge`
- **What**: Recall-oriented n-gram or LCS overlap w/ a reference.
- **Why**: Summarization = coverage problem: did the summary keep the reference's content?
    - Precision rewards saying little but correctly → recall fits coverage.
- **How**:
    - **ROUGE-N**: Recall over n-grams. ROUGE-1 (content) & ROUGE-2 (fluency/order) are the reported pair.
    - **ROUGE-L**: Longest common subsequence → order-aware, ❌fixed $n$, ❌contiguity requirement.
    - **ROUGE-Lsum**: Union-LCS. Per reference sentence, union the LCS matches against every candidate sentence, each token counted once.
        - Variant reported on CNN/DM & XSum.
    - Modern implementations report F1, ❌raw recall ← recall alone is maximized by copying the whole document.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\beta$: Recall weight.
- Misc:
    - $\mathcal{G}_n(\cdot)$: Set of distinct n-grams in a sequence.
    - $g$: An n-gram.
    - $\#_y(g)$: Count of $g$ in $y$.
    - $\text{LCS}(y,y^*)$: Length of the longest common subsequence.

ROUGE-N (recall form):

$$
\text{ROUGE-N}=\frac{\sum_{g\in\mathcal{G}_n(y^*)}\min(\#_y(g),\#_{y^*}(g))}{\sum_{g\in\mathcal{G}_n(y^*)}\#_{y^*}(g)}
$$

ROUGE-L:

$$
R_\text{lcs}=\frac{\text{LCS}(y,y^*)}{|y^*|},\quad P_\text{lcs}=\frac{\text{LCS}(y,y^*)}{|y|},\quad F_\text{lcs}=\frac{(1+\beta^2)P_\text{lcs}R_\text{lcs}}{\beta^2P_\text{lcs}+R_\text{lcs}}
$$
- Original paper: large $\beta$ → $F_\text{lcs}\to R_\text{lcs}$.
- `rouge_score` package: $\beta=1$.
```

````{important} Code
:class: dropdown
```python
class RougeL:
    def lcs(self, a, b):
        ## dp[i][j] = LCS length of a[:i] and b[:j]
        dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
        for i in range(1, len(a) + 1):
            for j in range(1, len(b) + 1):
                dp[i][j] = dp[i-1][j-1] + 1 if a[i-1] == b[j-1] else max(dp[i-1][j], dp[i][j-1])
        return dp[-1][-1]

    def __call__(self, cand, ref):
        c, r = cand.split(), ref.split()
        hit = self.lcs(c, r)
        if hit == 0:
            return 0.0
        ## subsequence, not substring -> matches may have gaps, but order is preserved
        prec, rec = hit / len(c), hit / len(r)
        return 2 * prec * rec / (prec + rec)   ## beta = 1, as in rouge_score

## Example: one substituted word in an otherwise identical sentence
print(round(RougeL()("the cat sat on the mat", "the cat was on the mat"), 3))   ## 0.833
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- Deterministic, free, reproducible from a spec → stable across labs & years.
- ROUGE-L: ❌$n$ to choose, gaps allowed → tolerates insertions that break a ROUGE-2 match.
- ROUGE-L stays order-sensitive: LCS ❌credits reordering.

*Cons?*
- Lexical, like BLEU: a perfect abstractive summary w/ different wording scores low.
- Reference-bound: LLM summaries can beat the human reference & still get punished.
- ROUGE-1 alone ≈ bag-of-words recall → gamed by keyword dumping.

*ROUGE-L vs ROUGE-Lsum?*
- L: one LCS over the whole flattened text.
- Lsum: split on newlines → union-LCS per reference sentence → aggregate.
- → Different numbers for the same pair. Papers often omit which one they used.

*Why does ROUGE survive?*
- Regression guard, ❌quality measure: a large ROUGE drop reliably signals that the model stopped tracking the source.
```

&nbsp;

### METEOR
- **Name**: Metric for Evaluation of Translation with Explicit ORdering {cite:p}`banerjee2005meteor`
- **What**: Recall-weighted unigram F-mean over an exact/stem/synonym alignment, × fragmentation penalty.
- **Why**: BLEU: exact surface match only, ❌explicit recall, word order only via n-grams.
    - Stems & synonyms get no BLEU credit.
    - Recall correlates w/ human judgment better than precision.
- **How**:
    1. Align candidate & reference unigrams in stages: exact → Porter stem → WordNet synonym.
    2. Unigram precision & recall over aligned words → harmonic mean, weighted 9:1 toward recall.
    3. Count **chunks**: maximal runs of aligned words that are adjacent & in the same order on both sides.
    4. More chunks → more fragmented order → larger penalty.
    5. Multiple references → score each → best score.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $P$: Unigram precision of the alignment.
    - $R$: Unigram recall of the alignment.
    - $u$: #aligned unigrams.
    - $ch$: #chunks.

$$\begin{align*}
F_\text{mean}&=\frac{10PR}{R+9P}\\
\text{Pen}&=0.5\left(\frac{ch}{u}\right)^3\\
\text{METEOR}&=F_\text{mean}(1-\text{Pen})
\end{align*}$$

Properties:
- $ch=1$ → minimal penalty. $ch=u$ → max penalty 0.5.
- Later versions tune the constants per language (weight, penalty scale & exponent).
```

```{note} Example
:class: dropdown
Reference `on mat cat sat`, candidate `cat sat on mat`:
- All 4 words align → $P=R=1$ → $F_\text{mean}=1$.
- Chunks: `cat sat`, `on mat` → $ch=2$, $u=4$.
- $\text{Pen}=0.5\cdot(2/4)^3=0.0625$ → $\text{METEOR}=0.9375$.
- Same order as the reference → $ch=1$ → $\text{Pen}\approx0.0078$ → $\text{METEOR}\approx0.992$.
- BLEU-4 of the reordered candidate = 0 ← no shared 4-gram.
```

```{attention} Q&A
:class: dropdown
*Pros?*
- Higher segment-level correlation w/ human judgment than BLEU.
- Credits morphological variants & synonyms.

*Cons?*
- Needs language resources (stemmer, WordNet) → strongest for English.
- Constants tuned per language → scores not comparable across languages.
- Slower than BLEU. Still lexical beyond the synonym list.

*Why a cubed penalty?*
- Small penalty until the alignment is badly fragmented.
- → Local reordering costs little; scrambled word order costs up to 50%.
```

&nbsp;

### chrF
- **Name**: Character n-gram F-score {cite:p}`popovic2015chrf`
- **What**: Recall-weighted F-score over character n-grams.
- **Why**: Word n-grams depend on tokenization, and one wrong suffix zeroes a whole word match.
    - → BLEU underscores morphologically rich & agglutinative languages.
    - Characters degrade gracefully & need no tokenizer.
- **How**:
    1. Extract character n-grams for $n=1..6$.
    2. Average precision (chrP) & recall (chrR) arithmetically over the orders present on both sides (❌geometric, as in BLEU).
    3. F-score w/ $\beta=2$.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\beta$: Recall weight, default 2 → recall counts twice as much as precision.
    - $n_\text{max}$: Max character n-gram order, default 6.

$$
\text{chrF}_\beta=(1+\beta^2)\frac{\text{chrP}\cdot\text{chrR}}{\beta^2\cdot\text{chrP}+\text{chrR}}
$$
- chrP: Fraction of candidate character n-grams found in the reference, averaged over effective orders.
- chrR: Fraction of reference character n-grams found in the candidate, averaged over effective orders.
- Effective order: an order w/ n-grams on both sides.

Properties:
- sacreBLEU default `chrF2`: $\beta=2$, $n_\text{max}=6$.
- **chrF++**: also averages in word uni- & bi-grams.
```

````{important} Code
:class: dropdown
```python
from collections import Counter

class ChrF:
    def __init__(self, n_max=6, beta=2.0):
        self.n_max, self.beta = n_max, beta

    def ngrams(self, s, n):
        return Counter(s[i:i+n] for i in range(len(s) - n + 1))

    def __call__(self, cand, ref):
        tot_p = tot_r = 0.0
        orders = 0
        for n in range(1, self.n_max + 1):
            c, r = self.ngrams(cand, n), self.ngrams(ref, n)
            n_c, n_r = sum(c.values()), sum(r.values())
            ## effective order: count it only if BOTH sides have n-grams
            ## -> precision & recall are averaged over the SAME set of orders
            if not n_c or not n_r:
                continue
            shared = sum((c & r).values())
            tot_p += shared / n_c
            tot_r += shared / n_r
            orders += 1
        if orders == 0:
            return float(cand == ref)              ## two empty strings are identical, not wrong
        P, R = tot_p / orders, tot_r / orders      ## ARITHMETIC mean over orders
        if P + R == 0:
            return 0.0
        b2 = self.beta ** 2
        return (1 + b2) * P * R / (b2 * P + R)
        ## sacreBLEU strips whitespace before extracting n-grams; kept here for brevity

## Example: word-level, "cats" & "feline" both fail to match "cat"
print(round(ChrF()("the cats sat", "the cat sat"), 3))     ## 0.662  <- inflection error, most credit kept
print(round(ChrF()("the feline sat", "the cat sat"), 3))   ## 0.318  <- different word, credit lost
```
````

```{attention} Q&A
:class: dropdown
*Pros over BLEU?*
- ❌Tokenizer → scores from different systems are directly comparable.
- Better segment-level correlation w/ humans. BLEU is trustworthy only at corpus level.
- Graceful on morphology, compounds & typos.

*Cons?*
- Lexical: a fluent paraphrase scores low.
- ❌BLEU's scale: chrF 55 & BLEU 55 mean nothing in common.
- Shared function words & shared script → inflated floor between related languages.

*Why does WMT still report chrF when COMET exists?*
- Neural metrics depend on the checkpoint & drift between versions.
- chrF = deterministic, reproducible baseline → pins the comparison.
```

&nbsp;

### Fuzzy String Matching
- **What**: Similarity from edit operations, ❌exact equality.
- **Why**: Exact match scores a one-character difference the same as a completely wrong answer.
    - Graders need tolerance for typos, spacing, casing, punctuation & unit suffixes.
- **How**:
    1. Levenshtein distance $d$ by DP: min #insertions + deletions + substitutions.
    2. Normalize to $[0,1]$ by the longer string.
    3. Accept above a threshold.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\ell(y)$: Length of $y$ in characters. Overrides page-level $|y|$ (token count).
    - $d_{i,j}$: Edit distance between the first $i$ characters of $y$ & the first $j$ characters of $y^*$.

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
- $y_i$, $y^*_j$: $i$-th / $j$-th character.

$$
\text{sim}(y,y^*)=1-\frac{d_{\ell(y),\ell(y^*)}}{\max(\ell(y),\ell(y^*))},\qquad \text{sim}=1\text{ if both are empty}
$$

Properties:
- $\mathcal{O}(\ell(y)\ell(y^*))$ time.
- $\mathcal{O}(\min(\ell(y),\ell(y^*)))$ memory w/ a rolling row.
```

````{important} Code
:class: dropdown
```python
class FuzzyRatio:
    def levenshtein(self, a, b):
        if len(a) < len(b):
            a, b = b, a                  ## symmetric -> keep the shorter string as the row
        prev = list(range(len(b) + 1))   ## only one previous row is needed
        for i, ca in enumerate(a, 1):
            curr = [i]
            for j, cb in enumerate(b, 1):
                ## delete / insert / substitute -- substitution is free if the characters match
                curr.append(min(prev[j] + 1, curr[j-1] + 1, prev[j-1] + (ca != cb)))
            prev = curr
        return prev[-1]

    def __call__(self, a, b):
        n = max(len(a), len(b))
        return 1.0 if n == 0 else 1 - self.levenshtein(a, b) / n   ## two empty strings are identical

## Example
f = FuzzyRatio()
print(f.levenshtein("kitten", "sitting"), round(f("kitten", "sitting"), 3))   ## 3 0.571
```
````

```{attention} Q&A
:class: dropdown
*Where is it used in LLM eval?*
- Short-form answer grading w/ tolerance, ❌brittle [exact match](#exact-match).
- Near-duplicate detection for [contamination](#contamination) checks.
- WER / CER for speech & OCR: word- / character-level edit distance ÷ reference length.

*Why can WER exceed 1?*
- Denominator = reference length, but insertions are unbounded.
- → A hallucinating transcriber can be >100% wrong.

*When NOT to use it?*
- Long generations: length dominates the distance → unrelated long texts get similar ratios.
- Meaning-sensitive answers: `"increase"` → `"decrease"` = 2 edits → ratio 0.75, "almost right".

*Why is a fixed acceptance threshold dangerous?*
- Right threshold depends on answer length: 2 edits = noise in a 40-character answer, a different answer in a 3-character one.
- → Normalize first, then threshold, or do not use it as a grader.
```

&nbsp;

## NLG (Semantic)
- **What**: Candidate–reference similarity in embedding space, or a model regressed on human ratings.

&nbsp;

### BERTScore
- **What**: Greedy-matched cosine similarity of contextual token embeddings, candidate vs reference. {cite:p}`zhang2020bertscore`
- **Why**: N-gram overlap needs the same words; a correct paraphrase can share none.
    - Contextual embeddings put paraphrases close together & separate homographs.
- **How**:
    1. Embed both sequences w/ a frozen pretrained encoder.
    2. Cosine similarity for every candidate–reference token pair.
    3. Greedy: each token takes its best partner → precision from the candidate side, recall from the reference side.
    4. F1, optionally IDF-weighted, optionally baseline-rescaled.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\mathbf{e}_i$: L2-normalized contextual embedding of candidate token $y_i$.
    - $\mathbf{e}^*_j$: L2-normalized contextual embedding of reference token $y^*_j$.
    - $b$: Baseline score from random sentence pairs.

Vectors are L2-normalized → inner product = cosine.

$$\begin{align*}
P_\text{BERT}&=\frac{1}{|y|}\sum_{i=1}^{|y|}\max_{j}\mathbf{e}_i^\top\mathbf{e}^*_j\\
R_\text{BERT}&=\frac{1}{|y^*|}\sum_{j=1}^{|y^*|}\max_{i}\mathbf{e}_i^\top\mathbf{e}^*_j\\
F_\text{BERT}&=\frac{2P_\text{BERT}R_\text{BERT}}{P_\text{BERT}+R_\text{BERT}}
\end{align*}$$

Baseline rescaling, for readability only:

$$
\hat{F}=\frac{F_\text{BERT}-b}{1-b}
$$
- Monotone in $F_\text{BERT}$ → ❌changes any ranking.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn.functional as F

class BERTScore:
    def __call__(self, cand_emb, ref_emb):
        ## cand_emb: (Tc, d), ref_emb: (Tr, d) -- contextual embeddings from a FROZEN encoder
        c = F.normalize(cand_emb, dim=-1)
        r = F.normalize(ref_emb, dim=-1)
        sim = c @ r.T                         ## (Tc, Tr) cosine of every pair
        ## greedy, NOT one-to-one: one reference token may serve several candidate tokens
        P = sim.max(dim=1).values.mean()      ## candidate token -> best reference token
        R = sim.max(dim=0).values.mean()      ## reference token -> best candidate token
        return (2 * P * R / (P + R)).item()

## Example
torch.manual_seed(0)
ref = torch.randn(5, 8)
print(round(BERTScore()(ref, ref), 3))      ## 1.0   <- every greedy max sits on the diagonal
print(BERTScore()(ref[:3], ref) < 1.0)      ## True  <- P = 1, but 2 reference tokens lack a match -> R drops
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- Handles paraphrase & synonymy that n-gram overlap misses.
- Recall catches omission, precision catches padding → asymmetric errors are visible.

*Cons?*
- Score = property of (metric, encoder, layer). Different encoders → different metrics.
- Raw values sit in a narrow band → rescaling is cosmetic; only differences within one config mean anything.
- Reference-bound & ❌factuality: negation & swapped entities barely move it.

*Why greedy matching, not an optimal one-to-one assignment?*
- Greedy = $\mathcal{O}(|y||y^*|)$, no solver.
- Cost: repeated candidate tokens can all reuse one reference token → repetition is barely penalized.
- → Pair BERTScore w/ a repetition check.

*Which encoder layer?*
- A tuned hyperparam, ❌automatically the last layer.
- Final layer specializes for the pretraining head → intermediate layers often correlate better w/ humans.
- → Publish model + layer (the hash) w/ the score.

*IDF weighting?*
- Function words match trivially & dominate the mean.
- IDF from the reference corpus → weights candidate tokens in $P$, reference tokens in $R$ → score shifts to content words.
```

&nbsp;

### BLEURT
- **Name**: Bilingual Evaluation Understudy with Representations from Transformers {cite:p}`sellam2020bleurt`
- **What**: BERT regression head fine-tuned on human ratings, after pretraining on synthetic perturbation pairs.
- **Why**: Human ratings are scarce, domain-bound & drift year to year.
    - → A metric fit only on them overfits & fails out of distribution.
    - Millions of cheap synthetic pairs teach "how bad is this corruption" before any real label.
- **How**:
    1. Perturb Wikipedia sentences (mask-fill, backtranslation, word dropping) → millions of (original, perturbed) pairs.
    2. Pretrain on automatic signals for those pairs: BLEU, ROUGE, BERTScore, backtranslation likelihood, entailment.
    3. Fine-tune on human ratings (WMT).

```{attention} Q&A
:class: dropdown
*How does it differ from COMET?*
- BLEURT scores (reference, candidate) only.
- [COMET](#comet) also reads the source → catches mistranslations that look fine against the reference.

*Cons?*
- Like every learned metric: checkpoint-dependent, unauditable, GPU-bound, gameable when optimized against.
- Original checkpoints are English-only. Multilingual checkpoints exist.

*Why does synthetic pretraining help?*
- Each synthetic target is a weak metric. Together they form a multi-task signal.
- → Encoder learns the many axes on which two sentences can differ.
- → Fine-tuning only maps those axes to the human scale, a much smaller problem.
```

&nbsp;

### COMET
- **Name**: Crosslingual Optimized Metric for Evaluation of Translation {cite:p}`rei2020comet`
- **What**: Regression head over a multilingual encoder, trained to predict human translation-quality scores.
- **Why**: Every unsupervised metric is a proxy for human judgment → if human scores exist, regress on them directly.
    - Fluent MT output → remaining errors are semantic → lexical & embedding metrics correlate poorly w/ humans.
    - Source access → catches mistranslations that look fine against the reference.
- **How**:
    1. Encode source, hypothesis & reference w/ a shared multilingual encoder (XLM-R).
    2. Pool each to a sentence vector.
    3. Build interaction features: element-wise products & absolute differences.
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

Model:

$$
\mathbf{z}=[\mathbf{h};\mathbf{r};\mathbf{h}\odot\mathbf{s};\mathbf{h}\odot\mathbf{r};|\mathbf{h}-\mathbf{s}|;|\mathbf{h}-\mathbf{r}|],\qquad\hat{q}=\text{FFN}(\mathbf{z})
$$
- $\mathbf{z}$: Interaction features.

Training:

$$
\mathcal{L}=(\hat{q}-q_\text{human})^2
$$
- $q_\text{human}$: Human quality score (Direct Assessment, or MQM-derived).

Reference-free (QE) variant: drop $\mathbf{r}$ → score from $(\mathbf{s},\mathbf{h})$ only.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class CometEstimator(nn.Module):
    def __init__(self, d, hidden=64):
        super().__init__()
        ## encoder (XLM-R) omitted: inputs are already-pooled sentence embeddings
        self.ffn = nn.Sequential(nn.Linear(6 * d, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, s, h, r):
        ## s, h, r: (B, d) pooled source / hypothesis / reference
        z = torch.cat([h, r, h * s, h * r, (h - s).abs(), (h - r).abs()], dim=-1)   ## (B, 6d)
        return self.ffn(z).squeeze(-1)                                              ## (B,) quality

## Example: one regression loss against human scores
B, d = 2, 8
model = CometEstimator(d)
s, h, r = torch.randn(B, d), torch.randn(B, d), torch.randn(B, d)
q_human = torch.tensor([0.7, -0.2])
loss = ((model(s, h, r) - q_human) ** 2).mean()   ## MSE on human judgments
loss.backward()
print(model(s, h, r).shape)   ## torch.Size([2])
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- Much higher correlation w/ human MT judgment than BLEU/chrF.
- QE variant needs no reference → usable online, on live traffic.

*Cons?*
- Scores are ❌absolute & ❌comparable across COMET checkpoints → always report the model name.
- Only as good as its training data's languages, domains & annotation protocol.
- Reproducible only from a checkpoint, ❌from a spec → hard to audit.
- GPU-bound → too expensive for inner-loop use.

*Why is optimizing against COMET dangerous?*
- Reranking or MBR decoding against COMET turns it into a [reward model](post/rl.md#reward) → same over-optimization as RLHF.
- → Tune w/ COMET, report w/ COMET & a deterministic metric.

*When does a learned metric fail hardest?*
- Out of distribution: low-resource languages, unusual domains, outputs far better or worse than its training range.
- Its errors are systematic → they do not average out over a test set.
```

&nbsp;

## Model-Based
- **What**: An LM grades outputs in place of a reference or a human.

&nbsp;

### LLM-as-a-Judge
- **What**: Strong LM scores or ranks outputs in place of a human annotator. {cite:p}`zheng2023judging`
- **Why**: Open-ended instruction following has no reference to overlap w/ & no verifier to run.
    - Human eval = ground truth, but costs days & money per model revision → ❌inner loop.
- **How**:
    1. **Pairwise**: two responses → judge picks a winner (or tie) → win rate / [Bradley-Terry](#arena) score.
    2. **Pointwise**: one response → absolute score on a rubric.
    3. **Reference-guided**: give the gold answer to the judge → required for math & reasoning grading.

```{dropdown} Table: Judge Biases
| Bias | What | Fix |
|:--|:--|:--|
| **Position** | Prefers the response in a given slot | Score both orders; count a win only if consistent |
| **Verbosity** | Prefers longer answers | [Length-controlled win rate](#length-controlled-win-rate) |
| **Self-enhancement** | Prefers outputs from its own model family | Judge ∉ evaluated models; jury of judges |
| **Limited reasoning** | Unreliable outside its own competence | Reference-guided grading, or an execution verifier |
| **Scale compression** | Pointwise scores pile up at 8–9/10 | Pairwise, or [G-Eval](#g-eval) probability weighting |
```

```{attention} Q&A
:class: dropdown
*Is it reliable?*
- GPT-4 judge agrees w/ human experts & crowdworkers >80% on MT-Bench & Chatbot Arena data.
- Human–human agreement on the same data ≈ the same level.
- → Judge reaches the ceiling of a noisy label, ❌truth.
- Agreement measured on general chat prompts → ❌transfer to specialist domains.

*Pros?*
- Works for any open-ended task w/o reference or verifier → most LLM use cases.
- Cheap & fast enough for a development loop.

*Cons?*
- Judge competence bounds reliability.
    - Verification is often easier than generation → a judge can grade above its own generation level.
    - Outside its competence → errors become systematic.
- API judges are non-stationary: endpoint changes version silently → scores drift → pin & publish the version.
- Optimize against a judge → judge becomes a reward model → it gets hacked (Goodhart's law).

*Pairwise or pointwise?*
- Pairwise: more reliable ← relative judgments are easier & the judge's scale drift cancels.
    - Round-robin = $\mathcal{O}(\#\text{models}^2)$. Bradley-Terry needs only a connected comparison graph → a sparse sample suffices.
- Pointwise: $\mathcal{O}(\#\text{models})$; score a new model w/o rerunning baselines. Absolute scale ❌comparable across judges.
- → Pointwise for regression tracking, pairwise for ranking claims.

*What breaks when judge & candidate share a model family?*
- Self-enhancement bias = systematic error → survives averaging over any #prompts.
```

&nbsp;

#### G-Eval
- **What**: Judge w/ auto-generated CoT eval steps; score = probability-weighted mean of score tokens. {cite:p}`liu2023geval`
- **Why**: A judge asked for an integer 1–5 returns nearly the same integer every time.
    - → Many ties → little ranking signal.
    - Logits keep the finer signal that sampling one integer discards.
- **How**:
    1. Give the judge the task definition & eval criteria.
    2. Judge generates detailed eval steps (CoT).
    3. Fill the form → read the probability of each score token.
    4. Expected score under that distribution.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\mathcal{S}$: Allowed score set, e.g., $\{1,\dots,5\}$.
    - $\pi_\text{judge}$: Judge LM's next-token distribution.

$$
\text{score}=\sum_{s\in\mathcal{S}}p(s)\cdot s,\qquad p(s)=\frac{\pi_\text{judge}(s|\text{prompt})}{\sum_{s'\in\mathcal{S}}\pi_\text{judge}(s'|\text{prompt})}
$$
- Renormalize over $\mathcal{S}$ ← discard the judge's mass on non-score tokens.
```

```{attention} Q&A
:class: dropdown
*Result?*
- Spearman 0.514 w/ human judgments on SummEval (GPT-4 judge) → above lexical & embedding metrics.

*Requirement?*
- Log-probs of score tokens → ❌APIs that hide them.
- Fallback: sample the judge $k$ times & average → same expectation, many more calls.

*Known failure that the authors report?*
- G-Eval prefers LLM-written summaries over human-written ones that human annotators rated higher.
- → A judge-based metric carries the judge's style preferences, and they favor machine text.

*Why is the weighted mean more than smoothing?*
- True quality is continuous; a 1–5 integer is a quantization.
- → Probability weights recover the sub-integer position → what a rank correlation needs.
```

&nbsp;

#### Length-Controlled Win Rate
- **What**: Pairwise win rate w/ the judge's length preference regressed out. {cite:p}`dubois2024length`
- **Why**: Verbosity bias makes raw win rate gameable.
    - Prompt the model to be more verbose → AlpacaEval win rate ⬆️, quality unchanged.
- **How**:
    1. Fit a GLM: judge preference ~ model identity + length difference + instruction difficulty.
    2. Report the model term at zero length difference → counterfactual "both answers equally long".

```{attention} Q&A
:class: dropdown
*Result?*
- Spearman correlation w/ Chatbot Arena: 0.94 → 0.98.
- Verbosity-prompt exploit is much weaker: win-rate spread across verbosity prompts shrinks, but not to 0.

*Why regress instead of truncating or length-matching outputs?*
- Truncation changes the answer being judged.
- Regression leaves the data untouched & removes the confound statistically.

*What does it NOT fix?*
- Other style confounds: markdown, headers, bullet lists, confident tone. [Style Control](#style-control) adds some as extra regressors.

*Assumption?*
- Length = pure bias. But a complete answer to a hard question is longer.
- → Correction also removes some real signal → a debiased estimate, ❌neutral. Report both.
```

&nbsp;

## Human
- **What**: Human judgment as the score: ground truth that every automatic metric approximates.

&nbsp;

### Human Rating
- **What**: Annotators score outputs against a fixed rubric.
- **Why**: Ground truth: every automatic metric is validated by its correlation w/ human ratings.
- **How**:
    1. Pick a protocol:
        - **Pointwise**: Likert score per output (e.g., 1–5 fluency, adequacy).
        - **Pairwise**: preference between two outputs.
        - **Error annotation (MQM)**: experts mark error spans w/ category & severity → weighted error score.
    2. Write a rubric w/ anchored examples for each score.
    3. ≥2 annotators per item, blind to system identity.
    4. Measure inter-annotator agreement: [Cohen's κ](../ml/eval.md#cohens-kappa) (2 raters), Fleiss' κ / Krippendorff's α (more).

```{attention} Q&A
:class: dropdown
*Pros?*
- Measures the target quality directly → no proxy.
- Works for any task, incl. ones w/o reference or verifier.

*Cons?*
- Slow & expensive → ❌inner loop.
- Annotator pools & guidelines drift → scores ❌comparable across studies.
- Subjective dimensions (helpfulness, style) → low agreement.
- Non-expert raters miss subtle errors (terminology, factual detail) → use experts for strong systems.

*Why does inter-annotator agreement matter for metric validation?*
- It estimates label reliability: low agreement → noisy label → low ceiling on any metric's correlation w/ one rater.
- Validate a metric against the aggregated (consensus) label, ❌one rater.
- A metric can agree w/ the consensus more than one rater agrees w/ another. This is not overfitting: aggregation removes rater noise.

*Pointwise or pairwise?*
- Pairwise: easier & more consistent ← relative judgment.
- Pointwise: absolute scale, new systems scored w/o re-comparison, but rater calibration drifts.
```

&nbsp;

### Arena
- **What**: Per-model latent strength fit by MLE to crowdsourced pairwise votes on user prompts. {cite:p}`chiang2024chatbot`
- **Why**: Open-ended chat has no reference, no verifier & no static test set that stays uncontaminated.
    - Live prompts from live users cannot be memorized in advance.
    - Only relative preference is observable; absolute chat quality is undefined.
- **How**:
    1. User submits a prompt → 2 anonymous models answer → user votes.
    2. Fit Bradley-Terry by logistic regression over all votes at once.
    3. Bootstrap over votes → confidence intervals.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\xi_i$: Latent strength of model $i$ (BT coefficient).
- Misc:
    - $\mathbf{z}\in\{-1,0,1\}^{\#\text{models}}$: Battle indicator: $+1$ for model A, $-1$ for model B, 0 elsewhere.
    - $v\in\{0,\frac{1}{2},1\}$: Vote label: A loses / tie / A wins.
    - $R_i$: Displayed Elo-scale rating of model $i$.

Model (Bradley-Terry):

$$
P(A\succ B)=\sigma(\xi_A-\xi_B)=\sigma(\mathbf{z}^\top\boldsymbol{\xi})
$$

Training:
- MLE = logistic regression on $\mathbf{z}$ w/ label $v$; a tie enters as $v=\frac{1}{2}$.
- $\boldsymbol{\xi}$ is identified only up to a shift → center it.

Elo display scale:

$$
R_i=\frac{400}{\ln10}\xi_i+C\quad\Leftrightarrow\quad P(A\succ B)=\frac{1}{1+10^{(R_B-R_A)/400}}
$$
- $C$: Arbitrary offset (e.g., 1000).
```

````{important} Code
:class: dropdown
```python
import numpy as np

class BradleyTerry:
    def __init__(self, n_models, lr=1.0, steps=20000):
        self.n, self.lr, self.steps = n_models, lr, steps

    def fit(self, battles):
        ## battles: (i, j, v) -> model i vs model j; v = 1 win, 0 loss, 0.5 tie
        xi = np.zeros(self.n)
        for _ in range(self.steps):
            g = np.zeros(self.n)
            for i, j, v in battles:
                p = 1 / (1 + np.exp(-(xi[i] - xi[j])))   ## P(i beats j)
                g[i] += v - p                            ## logistic-regression gradient
                g[j] -= v - p                            ## mirror image for the opponent
            xi += self.lr * g / len(battles)
        return xi - xi.mean()   ## shift-invariant -> center to anchor the scale

## Example: A beats B 2 of 3, B beats C 2 of 3
battles = [(0, 1, 1), (0, 1, 1), (0, 1, 0), (1, 2, 1), (1, 2, 1), (1, 2, 0)]
xi = BradleyTerry(3).fit(battles)
print(np.round(xi, 4))                              ## [ 0.6931  0.  -0.6931]  == logit(2/3) = ln 2
print(np.round(400 / np.log(10) * xi + 1000, 1))    ## [1120.4 1000.   879.6]
```
````

```{attention} Q&A
:class: dropdown
*Why BT MLE instead of online Elo?*
- Online Elo depends on battle order & K-factor → same data, different ratings.
- BT MLE uses all votes at once → order-independent, reproducible, bootstrap CIs.

*What does "+20 Elo" mean?*
- $\frac{1}{1+10^{-20/400}}\approx0.529$ → 52.9% win rate. A 200-point gap ≈ 76%.
- → Small Elo deltas = small preference deltas, ❌qualitative jumps.

*Pros?*
- ❌Reference, ❌verifier, ❌static test set → much less exposed to contamination than a fixed suite.
- Measures what users care about, w/ uncertainty.

*Cons?*
- Prompt distribution = whatever anonymous users type → mostly short, casual, easy → ❌hard specialist work.
- ❌Contamination-proof: users paste public benchmark questions, and labs tune toward Arena's prompt distribution.
- Style, verbosity & formatting bias → see [Style Control](#style-control).
- Gameable: formatting can reveal the model; an org can vote for its own model.
- Relative & slow: a new model needs thousands of votes before its CI separates it from others.
```

&nbsp;

#### Style Control
- **What**: Bradley-Terry w/ style-difference features as extra regressors. {cite:p}`li2024does`
- **Why**: Voters prefer longer, more formatted answers → raw Arena strength mixes style & substance.
- **How**:
    1. Per battle, compute style features as normalized A–B differences: answer token length, #markdown headers, #bold elements, #list elements.
    2. Fit them jointly w/ the model indicators in the BT logistic regression.
    3. Rank by the model coefficients → strength at equal style.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\boldsymbol{\gamma}$: Style coefficients.
- Misc:
    - $\mathbf{f}$: Style-difference feature vector of one battle.
    - $a_A$, $a_B$: One style feature of answer A / B (e.g., token length).

Model (extends [Arena](#arena)):

$$
P(A\succ B)=\sigma(\mathbf{z}^\top\boldsymbol{\xi}+\mathbf{f}^\top\boldsymbol{\gamma})
$$

Feature, before standardization:

$$
f=\frac{a_A-a_B}{a_A+a_B}
$$
- Relative difference: 500 vs 520 tokens ≈ equal; 20 vs 40 tokens = very different.
```

```{attention} Q&A
:class: dropdown
*Is style control only cosmetic?*
- ❌. It reorders the leaderboard: models that won on presentation drop, concise ones rise.
- → Part of the raw ranking tracked formatting.

*Does the adjusted coefficient measure pure capability?*
- ❌. Observational regression: style correlates w/ unobserved quality.
- → A style-adjusted association, ❌causal effect of capability.
- Better answers to hard prompts can be longer → adjustment also removes some real signal.
```

&nbsp;

## IR
- **What**: Ranking quality of the context a RAG system retrieves before generation.
- **Why**: RAG failure = passage never retrieved, or retrieved & ignored → only retrieval metrics separate the two.

&nbsp;

### Recall@k
- **What**: Fraction of relevant documents in the top $k$.
- **Why**: Ceiling on RAG: the generator cannot use a passage that was never retrieved.
- **How**: $\frac{|\{\text{relevant}\}\cap\{\text{top-}k\}|}{|\{\text{relevant}\}|}$ → mean over queries.

```{attention} Q&A
:class: dropdown
*Why recall, not precision, for RAG?*
- Long-context generator tolerates a few irrelevant passages; it cannot recover a missing one.
- [Precision@k](#precision-k) matters once $k$ hits the context, latency or cost budget.

*Single-gold case?*
- Exactly one relevant document → Recall@$k$ = hit rate = "is the gold in the top $k$?".

*What does it hide?*
- Position: Recall@10 is the same for the gold at rank 1 or rank 10. Generator attention is not.
- → Pair w/ [MRR](#mrr) or [nDCG](#ndcg).
```

&nbsp;

#### Precision@k
- **What**: Fraction of the top $k$ results that are relevant.
- **Why**: $k$ fixed by a context or latency budget → irrelevant results in those $k$ slots are pure cost.
- **How**: $\frac{|\{\text{relevant}\}\cap\{\text{top-}k\}|}{k}$ → mean over queries.

```{attention} Q&A
:class: dropdown
*Why is it weak on its own?*
- Ceiling = $\frac{\#\text{relevant}}{k}$ → 1 relevant document caps Precision@10 at 0.1, for any ranker.
- → ❌Compare Precision@$k$ across queries w/ different #relevant documents.

*Why does it matter more for RAG than for classic search?*
- Irrelevant passages consume context, add latency & cost, and distract the generator.
```

&nbsp;

### MRR
- **Name**: Mean Reciprocal Rank
- **What**: Mean over queries of $\frac{1}{\text{rank}}$ of the first relevant result.
- **Why**: Only the top hit is read → only question: how far down is the first correct answer?
- **How**: Per query, find the first relevant rank within $k$ → reciprocal (0 if none) → mean.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\mathcal{Q}$: Query set.
    - $\text{rank}_q$: Rank of the first relevant result for query $q$.

$$
\text{MRR@}k=\frac{1}{|\mathcal{Q}|}\sum_{q\in\mathcal{Q}}\frac{1}{\text{rank}_q}
$$
- Term = 0 if no relevant result is within $k$.
```

````{important} Code
:class: dropdown
```python
import numpy as np

class MRR:
    def __init__(self, k):
        self.k = k

    def __call__(self, ranked_rel):
        ## ranked_rel: relevance grades in retrieved order, for ONE query
        hits = np.flatnonzero(ranked_rel[:self.k])
        return 1.0 / (hits[0] + 1) if hits.size else 0.0   ## only the FIRST hit counts

## Example
print(MRR(3)(np.array([0, 1, 3, 0, 0])))   ## 0.5  <- first hit at rank 2
```
````

```{attention} Q&A
:class: dropdown
*Cons?*
- Every relevant document after the first is ignored → ❌multi-hop, ❌multi-evidence RAG.
- Steep reciprocal: rank 1 vs 2 costs 0.5, rank 9 vs 10 costs 0.011 → almost all weight on the top 2–3 slots.

*Where is it the official metric?*
- MS MARCO passage ranking: MRR@10.
```

&nbsp;

### MAP
- **Name**: Mean Average Precision
- **What**: Mean over queries of Precision@$i$ averaged over the ranks $i$ that hold a relevant document.
- **Why**: Precision@$k$ needs an arbitrary $k$; Recall@$k$ ignores order.
    - Average precision at each hit → whole ranking in one number, no cutoff to pick.
- **How**: Walk the ranking → at each relevant document, record the running precision → sum ÷ #relevant → mean over queries.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\mathcal{Q}$: Query set.
    - $P@i$: Precision at rank $i$.
    - $rel_i\in\{0,1\}$: Binary relevance at rank $i$.
    - $R_q$: #relevant documents for query $q$.
    - $N_q$: Length of the ranked list for query $q$.

$$
\text{AP}_q=\frac{1}{R_q}\sum_{i=1}^{N_q}P@i\cdot rel_i,\qquad\text{MAP}=\frac{1}{|\mathcal{Q}|}\sum_{q\in\mathcal{Q}}\text{AP}_q
$$
- Relevant documents never retrieved add 0 to the sum, but still count in $R_q$.
- MAP@$k$: sum only to rank $k$. Some variants then divide by $\min(R_q,k)$ → state which.
```

````{important} Code
:class: dropdown
```python
import numpy as np

class AveragePrecision:
    def __call__(self, ranked_rel, n_relevant):
        ## ranked_rel: binary relevance in retrieved order; n_relevant: ALL relevant docs for the query
        rel = np.asarray(ranked_rel, dtype=float)
        prec_at_i = np.cumsum(rel) / np.arange(1, rel.size + 1)   ## running precision P@i
        return (prec_at_i * rel).sum() / n_relevant               ## only ranks holding a hit count

## Example: hits at ranks 2 & 3
ap = AveragePrecision()
print(round(ap([0, 1, 1, 0, 0], 2), 3))   ## 0.583  == (1/2 + 2/3) / 2
print(round(ap([0, 1, 1, 0, 0], 3), 3))   ## 0.389  <- a 3rd relevant doc was never retrieved
## MAP = mean of AP over queries
```
````

```{attention} Q&A
:class: dropdown
*MAP vs nDCG?*
- MAP: binary relevance. nDCG: graded relevance.
- → MAP is wrong when "somewhat relevant" is a real category.
- MAP has no explicit position discount; precision at each hit gives an implicit one.

*Why does it still show up?*
- Classic single-number TREC summary of a whole ranking.
- Standard answer to "how do you score a ranker w/o picking $k$?".
```

&nbsp;

### nDCG
- **Name**: normalized Discounted Cumulative Gain {cite:p}`jarvelin2002cumulated`
- **What**: Position-discounted sum of graded relevance, ÷ the same sum for the ideal ordering.
- **Why**: Binary, position-blind metrics cannot say "highly relevant at rank 1 beats slightly relevant at rank 1".
    - Raw DCG is ❌comparable across queries ← different #relevant documents → normalize by the best achievable DCG.
- **How**:
    1. Gain per result from its relevance grade.
    2. Discount by rank.
    3. Sum to $k$ → DCG.
    4. Divide by IDCG: DCG of the ideal ordering of all judged documents.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $rel_i$: Graded relevance of the result at rank $i$.
    - $\text{IDCG@}k$: DCG of the relevance-sorted ideal ranking of all judged documents for the query.

$$
\text{DCG@}k=\sum_{i=1}^{k}\frac{2^{rel_i}-1}{\log_2(i+1)},\qquad\text{nDCG@}k=\frac{\text{DCG@}k}{\text{IDCG@}k}\in[0,1]
$$

Properties:
- IDCG from only the retrieved documents → a run that misses every good document still normalizes to 1.
- Original form uses linear gain $rel_i$ in place of $2^{rel_i}-1$. Both are in use → state which.
```

````{important} Code
:class: dropdown
```python
import numpy as np

class NDCG:
    def __init__(self, k):
        self.k = k

    def dcg(self, rel):
        ## 2^rel - 1: one highly relevant doc outweighs several marginal ones
        return np.sum((2.0 ** rel - 1) / np.log2(np.arange(2, rel.size + 2)))

    def __call__(self, ranked_rel, all_grades):
        ## IDCG from ALL judged docs, not just the retrieved ones
        ideal = np.sort(all_grades)[::-1][:self.k]
        idcg = self.dcg(ideal)
        return self.dcg(ranked_rel[:self.k]) / idcg if idcg else 0.0

## Example: highly relevant doc buried at rank 3
grades = np.array([3, 1, 0, 0, 0])                           ## every judged doc's grade for the query
print(round(NDCG(3)(np.array([0, 1, 3, 0, 0]), grades), 3))  ## 0.541  <- penalized for the order
print(round(NDCG(3)(np.array([0, 1, 0, 0, 0]), grades), 3))  ## 0.083  <- grade-3 doc missed; 0.631 if IDCG came from the run
```
````

```{attention} Q&A
:class: dropdown
*Why the exponential gain?*
- Weights high grades disproportionately → one highly relevant document dominates a few marginal ones.
- ❌Lexicographic: enough discounted low-grade documents still outweigh one high-grade document.

*Why $\log_2(i+1)$?*
- Discount 1 at rank 1, slow decay after → ranks 1–10 differ meaningfully, rank 50 is not zeroed.
- Any smooth decreasing function works; the log is convention, ❌derivation.

*Cons?*
- Needs graded human labels → expensive; grades drift between annotator pools.
- **Pooling bias**: unjudged documents count as irrelevant → a system that finds good, unlabeled documents is punished.
- nDCG@10 on one benchmark says nothing about nDCG@10 on another; ideal rankings differ.

*Standard for retrieval eval?*
- BEIR reports nDCG@10 → the number to match when comparing embedding models for RAG.
```

&nbsp;

### RAG Groundedness
- **What**: Reference-free checks that the answer is supported by retrieved context & addresses the question. {cite:p}`es2024ragas`
- **Why**: Retrieval metrics score the context; task metrics need a gold answer.
    - Production RAG has no gold answer & no guarantee the generator used its context.
- **How**: Compute each measure w/ an [LLM judge](#llm-as-a-judge).
    1. **Faithfulness**: split the answer into atomic claims → fraction entailed by the retrieved context.
    2. **Answer relevance**: generate questions from the answer → mean embedding similarity to the original question.
    3. **Context relevance**: fraction of retrieved sentences needed to answer.

```{attention} Q&A
:class: dropdown
*Is a faithful answer a correct answer?*
- ❌. An answer grounded in a wrong retrieved passage is faithful & wrong.
- → Faithfulness isolates generator hallucination. Retrieval quality & factual accuracy need separate measurements.

*Why split into claims?*
- Whole-paragraph entailment is dominated by its supported parts → one fabricated clause disappears into the average.
- Per-claim scoring → a single hallucination is visible.

*Cons?*
- LLM judge underneath → inherits every [judge bias](#llm-as-a-judge), cost & version drift.
- Claim splitting is brittle on long or highly structured answers.
- Optimize a RAG system against it → answers that quote the context & answer nothing.
```

&nbsp;
