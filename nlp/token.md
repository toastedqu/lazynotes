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
# Tokenization
Text $\rightarrow$ integer IDs: the LM's only interface to language. Covers the subword algorithms behind every modern LLM tokenizer, their production instantiations, and the artifacts they leak into model behavior.

Notations:
- $x$: Raw text string
- $\mathcal{V}$: Vocab (token set)
- $t$: Token (a byte/char string in $\mathcal{V}$)
- $s=(t_1,\dots,t_k)$: Segmentation of $x$
- $\mathcal{D}$: Tokenizer training corpus
- $\#a$: #occurrences of $a$ in $\mathcal{D}$
- $\#(a,b)$: #occurrences of $a$ immediately followed by $b$

&nbsp;

## Pipeline
- **What**: Raw text $\xrightarrow{\text{normalize}}$ Clean text $\xrightarrow{\text{pre-tokenize}}$ Chunks $\xrightarrow{\text{subword model}}$ Tokens $\xrightarrow{\text{post-process}}$ Token IDs
- **Why**: NNs index a **fixed, finite** embedding table. Text is an unbounded string space → need a lossless, fixed-size encoding of it.
- **How**:
    1. **Normalize**: Unicode canonicalization (+ optional lowercase / accent strip).
    2. **Pre-tokenize**: Split into chunks that merges may never cross.
    3. **Subword model**: Chunk → tokens (BPE / WordPiece / Unigram).
    4. **Post-process**: Insert special tokens → IDs.

```{attention} Q&A
:class: dropdown
*Is the tokenizer trained jointly with the model?*
- ❌ Trained once beforehand on a corpus sample, then **frozen**. The LM never sees text, only IDs.
- → Tokenizer corpus $\neq$ model corpus is a real bug source (glitch tokens).

*Rule of thumb for English on a GPT-style tokenizer?*
- ~4 chars/token, ~0.75 words/token. Code & non-English: worse.

*Which stage owns which artifact?*
- Normalization → information destroyed before the model can see it (case, accents).
- Pre-tokenization → what can never be one token (digit chunking, cross-word tokens).
- Subword model → which strings earn a slot.
- Post-processing → control-token correctness (chat template).
```

&nbsp;

### Normalization
- **What**: Map visually/semantically equivalent strings to one canonical form.
- **Why**: Unicode encodes many glyphs 2+ ways (`é` = U+00E9, or `e` + U+0301) → identical-looking text would otherwise get different IDs.
- **How**: NFC/NFKC → optionally lowercase, strip accents, collapse whitespace.
    - Modern LLMs: **near-identity**. Case, accents & whitespace are information (code indentation, proper nouns) → destroying them costs more than it saves.
    - Byte-level BPE as shipped (GPT-2, tiktoken): **zero** normalization ← operates on raw UTF-8 bytes.
    - BERT-uncased: fully destructive (lowercase + strip accents) ← a discriminative encoder never has to reproduce its input.

```{attention} Q&A
:class: dropdown
*NFC vs NFKC?*
- NFC: canonical composition. Preserves the **abstract characters** — only the encoding of a glyph changes, never which glyph it is.
- NFKC: also **compatibility** folding (`ﬁ`→`fi`, `①`→`1`, full-width→half-width). Lossy — it changes the text itself.
- Neither is invertible: both map several input forms onto one output form.

*Why did aggressive normalization fall out of favor for LLMs?*
- Generative models must **emit** text, so any destroyed distinction is unrecoverable at decode time.
- Code is the killer case: whitespace and case are syntax.
```

&nbsp;

### Pre-Tokenization
- **What**: Split raw text into chunks; merges never cross a chunk boundary.
- **Why**:
    - Bounds the merge search space → training is tractable.
    - Hard-codes priors that corpus statistics would otherwise violate: no token spans two words, no token glues a word to its punctuation.
- **How**: 2 families.
    - **Regex split** (GPT-2 → tiktoken → Llama 3): one pattern isolating contractions, letter runs, digit runs, punctuation runs, whitespace. A leading space stays **attached** (`" the"` is one token, distinct from `"the"`).
    - **Whitespace marker** (SentencePiece): replace `" "` with `▁` so spaces become ordinary data, then split on whitespace anyway (`split_by_whitespace=true` by default). Set it to `false` to allow pieces spanning a space.

````{important} Code
:class: dropdown
```python
import regex  ## `re` lacks \p{L}; tiktoken uses the `regex` module too

## cl100k_base (GPT-3.5/4) pre-tokenizer, verbatim from tiktoken
CL100K = regex.compile(
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"""
)

def pre_tokenize(text):
    ## BPE runs INDEPENDENTLY inside each chunk -> no merge can cross these boundaries
    return CL100K.findall(text)

## Example
print(pre_tokenize("I don't pay $12345."))
## ["I", " don", "'t", " pay", " $", "123", "45", "."]
```
````

```{dropdown} Table: Digit Handling
| Family | Rule | Effect |
|:--|:--|:--|
| GPT-2 (`r50k_base`) | ` ?\p{N}++` | Unbounded digit run per chunk → splits decided purely by learned merges |
| GPT-3.5/4/4o, Llama 3, DeepSeek-V3 | `\p{N}{1,3}` | Hard cap: ≤3 digits/token, chunked left-to-right |
| Qwen2/2.5 | `\p{N}` | Exactly 1 digit/token |
| Llama 1/2, Gemma | `split_digits=true` | Exactly 1 digit/token |
```

```{attention} Q&A
:class: dropdown
*Why is pre-tokenization not just an optimization?*
- It is a **hard constraint on the vocab**. `" the"` can exist; `"the cat"` can never, regardless of frequency.
- Two tokenizers with identical merge algorithms and identical corpora produce different vocabs if their regexes differ.

*Why attach the leading space to the word instead of emitting a space token?*
- Word-initial vs word-internal occurrences of a string are linguistically different → separate embeddings.
- Also halves sequence length vs one token per space.

*Why does SentencePiece refuse to pre-split on whitespace?*
- Chinese/Japanese/Thai have no spaces → the "word" prior does not exist.
- Whitespace-split tokenizers cannot detokenize losslessly (how many spaces were there?).
```

&nbsp;

## Subword
- **What**: Symbol set between chars and words, learned from corpus statistics.
- **Why**: Word-level and char-level fail at opposite extremes.
    - Word: Unbounded vocab + OOV on every typo/name/inflection → `UNK` destroys information irrecoverably.
    - Char: Tiny vocab, but ~4-5× longer sequences → attention $O(L^2)$⬆️ and near-zero meaning per step.
- **How**: Frequent strings get their own slot; rare strings decompose into pieces. The 3 algorithms below differ only in **which strings are worth a slot**.

```{dropdown} Table: BPE vs WordPiece vs Unigram
| | **BPE** | **WordPiece** | **Unigram** |
|:--|:--|:--|:--|
| Direction | Merge up | Merge up | Prune down |
| Criterion | Pair frequency $\#(a,b)$ | Likelihood gain $\frac{\#(a,b)}{\#a\cdot\#b}$ | Corpus NLL increase if token removed |
| Model of text | ❌ (greedy heuristic) | Unigram LM | Unigram LM over all segmentations |
| Encoding | Replay merges in rank order | Greedy longest-match | Viterbi argmax |
| Stochastic encoding | Only via BPE-dropout | ❌ | ✅ Native (sample $P(s\|x)$) |
| OOV | ❌ w/ byte base | `[UNK]` per word | ❌ w/ char + byte fallback |
| Used by | GPT, Llama, Qwen, DeepSeek, Mistral, Gemma | BERT family | T5/mT5, ALBERT, XLNet |
```

&nbsp;

### BPE
- **Name**: Byte-Pair Encoding {cite:p}`sennrich2016neural`
- **What**: Repeatedly merge the most frequent adjacent symbol pair into one new symbol.
- **Why**: Frequency is a cheap, effective proxy for "deserves a vocab slot" → high compression per slot, while every string stays representable from base symbols.
- **How**:
    - **Training**:
        1. Init $\mathcal{V}$ = base alphabet (bytes/chars); split $\mathcal{D}$ into base symbols.
        2. Count all adjacent pairs $\#(a,b)$.
        3. Merge the argmax pair into $ab$; append $(a,b)$ to an **ordered** merge list.
        4. Repeat 2-3 until $|\mathcal{V}|$ hits target.
    - **Encoding**: Per chunk, repeatedly apply the **lowest-rank** applicable merge → deterministic, no search.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\mathcal{M}=[(a_1,b_1),\dots,(a_R,b_R)]$: Ordered merge list; the index is the **rank**.
- Hyperparams:
    - $|\mathcal{V}|$: Target vocab size, $R=|\mathcal{V}|-|\text{base}|$.

Training, step $r$:

$$
(a_r,b_r)=\arg\max_{(a,b)}\#(a,b),\qquad\mathcal{V}\leftarrow\mathcal{V}\cup\{a_rb_r\}
$$

- Counts are recomputed **after** applying merge $r$, so later steps see the merged symbols.

Encoding a chunk $s$: while any merge applies,

$$
r^*=\min\{r:(a_r,b_r)\text{ adjacent in }s\},\qquad s\leftarrow\text{merge}(s,(a_{r^*},b_{r^*}))
$$
```

````{important} Code
:class: dropdown
```python
from collections import Counter

class BPE:
    """Byte-level BPE: base alphabet = the 256 byte values."""
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.merges = {}  ## (a, b) -> rank

    def train(self, chunks):
        ## each pre-token chunk -> tuple of single bytes, kept with its corpus count
        words = {tuple(bytes([b]) for b in c.encode()): n
                 for c, n in Counter(chunks).items()}
        for rank in range(self.vocab_size - 256):  ## the 256 base symbols are free
            pairs = Counter()
            for w, n in words.items():
                for p in zip(w, w[1:]):
                    pairs[p] += n  ## weight each pair by its word's frequency
            if not pairs:
                break
            best = max(pairs, key=pairs.get)  ## the whole algorithm: argmax frequency
            self.merges[best] = rank
            words = {self._apply(w, best): n for w, n in words.items()}
        return self

    @staticmethod
    def _apply(w, pair):
        out, i = [], 0
        while i < len(w):
            if i < len(w) - 1 and (w[i], w[i + 1]) == pair:
                out.append(w[i] + w[i + 1]); i += 2
            else:
                out.append(w[i]); i += 1
        return tuple(out)

    def encode(self, chunk):
        w = tuple(bytes([b]) for b in chunk.encode())
        while len(w) > 1:
            ## replay merges by LEARNED RANK, never by longest match
            cand = [(self.merges[p], p) for p in zip(w, w[1:]) if p in self.merges]
            if not cand:
                break
            w = self._apply(w, min(cand)[1])
        return w

## Example
corpus = "low low low low low lower lower newest newest newest widest widest"
bpe = BPE(vocab_size=256 + 6).train(corpus.split())
print(list(bpe.merges))
## [(b'l', b'o'), (b'lo', b'w'), (b'e', b's'), (b'es', b't'), (b'n', b'e'), (b'ne', b'w')]
print(bpe.encode("lowest"))  ## (b'low', b'est') <- unseen word, built from learned pieces
```
````

```{attention} Q&A
:class: dropdown
*Why replay merges by rank instead of greedy longest-match?*
- Rank order reproduces the exact merge sequence seen during training → encoding is consistent with the statistics the vocab was built from.
- Longest-match can produce a segmentation that BPE training would never have created at that position.

*Is greedy merging optimal?*
- ❌ Each merge is locally optimal for compression only. The resulting size-$|\mathcal{V}|$ vocab minimizes neither corpus length nor corpus likelihood.
- Kept because it is simple, fast, and empirically hard to beat by enough to matter.

*Complexity?*
- Naive training: $O(R\cdot N)$ over corpus length $N$ ← full re-count per merge. Practical: incremental pair counts + priority queue.
- Encoding: quadratic in **chunk** length, but a chunk is one word → irrelevant.

*Does BPE ever emit `UNK`?*
- Only if the base alphabet misses an input symbol. With a byte base (256 symbols): **never**.

*Why weight pairs by word count instead of scanning raw text?*
- Identical result. Collapsing $\mathcal{D}$ to a `{word: count}` table is deduplication → training scales with #distinct words, not corpus size.
```

&nbsp;

#### Byte-Level BPE
- **Name**: Byte-level Byte-Pair Encoding (BBPE) {cite:p}`radford2019language`
- **What**: BPE whose base alphabet is the 256 byte values.
- **Why**: A character base alphabet is huge and version-dependent (~150k assigned Unicode codepoints, growing every release) → either an enormous base vocab or `UNK` for unseen scripts/emoji/binary. 256 bytes cover **everything**, losslessly and permanently.
- **How**:
    1. UTF-8 encode → byte sequence.
    2. Regex pre-tokenize → chunks.
    3. Run BPE over bytes.
    - GPT-2 extra: a bijective byte $\rightarrow$ printable-Unicode map (`" "`→`Ġ`, `"\n"`→`Ċ`) so the vocab file stays readable text and no control bytes leak.
- Dominant modern choice: GPT-2 → GPT-4o, Llama 3, Qwen, DeepSeek-V3, Mistral NeMo.

````{important} Code
:class: dropdown
```python
## Reuses BPE from the block above -- it is already byte-level.
bpe = BPE(vocab_size=256 + 6).train(
    "low low low low low lower lower newest newest newest widest widest".split()
)

## Non-ASCII never seen in training still encodes: it falls back to raw bytes
print(bpe.encode("héllo"))
## (b'h', b'\xc3', b'\xa9', b'l', b'lo')  <- 'é' = 2 UTF-8 bytes, no UNK
```
````

```{attention} Q&A
:class: dropdown
*Byte-level BPE vs SentencePiece byte fallback?*
- BBPE: the base alphabet **is** bytes; merges run over bytes from step 1.
- Byte fallback: the base alphabet is **characters**; `<0xNN>` pieces exist only as an escape hatch for characters absent from the vocab.

*Can one token be an invalid UTF-8 fragment?*
- ✅ A token may be half a multi-byte character → streaming decoders must buffer bytes until they form valid UTF-8, else emoji/CJK output flickers as replacement chars.

*What does the byte base cost?*
- Unmerged non-Latin text degrades to 1 token/byte → 3 tokens per CJK char, 4 per emoji.
- → non-English efficiency is a **vocab budget** decision, not a property of the algorithm.

*Why did GPT-2 remap bytes to printable Unicode instead of using them raw?*
- The vocab/merge files stay plain text; whitespace and control bytes would otherwise be unrepresentable or silently mangled by text tooling.
- It is a bijection → zero effect on what gets merged.
```

&nbsp;

### WordPiece
- **What**: BPE, but merge by **likelihood gain** instead of raw frequency. {cite:p}`schuster2012japanese`
- **Why**: Frequency over-rewards pairs whose parts are individually frequent (`"e"`+`"s"`). Dividing by the parts' own counts asks the real question: does gluing these two explain the corpus **beyond** what they already explain separately?
- **How**:
    - **Training**: Same loop as BPE, argmax over $\frac{\#(a,b)}{\#a\cdot\#b}$.
    - **Encoding**: Greedy **longest-match-first** per word (not merge replay). Non-initial pieces prefixed `##`. No matching prefix → the **whole word** becomes `[UNK]`.
- ⚠️ Google's exact training procedure is unpublished. The paper specifies the **criterion** (pick the merge that most increases corpus likelihood); the ratio above is the standard reconstruction shipped in HF `tokenizers`.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $p(t)=\#t/Z$: Unigram probability of token $t$.
    - $Z=\sum_{t'\in\mathcal{V}}\#t'$: Total token count under the curr segmentation.
- Misc:
    - $c$: New token formed by merging $a,b$.
    - $q(a,b)$: #merges actually performed ($=\#(a,b)$ except for overlapping self-pairs, e.g. `aaa` has $\#(a,a)=2$ but only 1 merge).

Objective — corpus log-likelihood under a unigram LM:

$$
\log L(\mathcal{V})=\sum_{t\in\mathcal{V}}\#t\log p(t)
$$

Merge criterion, as implemented:

$$
\text{score}(a,b)=\frac{\#(a,b)}{\#a\cdot\#b}
$$

Count updates on merging $a,b\rightarrow c$:

$$\begin{align*}
\#c&\leftarrow q(a,b)\\
\#a&\leftarrow\#a-q(a,b)\\
\#b&\leftarrow\#b-q(a,b)
\end{align*}$$
```

```{tip} Derivation
:class: dropdown
*How does the likelihood criterion become that ratio?*
1. Count only the merged occurrences: each contributes $\log p(c)$ instead of $\log p(a)+\log p(b)$. Per-occurrence gain:

    $$
    \Delta(a,b)=\log p(c)-\log p(a)-\log p(b)
    $$

2. Substitute $p(t)=\#t/Z$ and $\#c=q(a,b)\approx\#(a,b)$, with $Z'$ the post-merge total:

    $$
    \Delta(a,b)=\log\frac{\#(a,b)}{\#a\cdot\#b}+2\log Z-\log Z'
    $$

3. $\#(a,b)\ll Z\Rightarrow Z'=Z-\#(a,b)\approx Z$, so the tail collapses to $\log Z$ — identical for every candidate at that step:

    $$
    \Delta(a,b)\approx\log\frac{\#(a,b)}{\#a\cdot\#b}+\log Z
    $$

4. → $\arg\max\Delta=\arg\max\frac{\#(a,b)}{\#a\cdot\#b}$: a PMI-like association score, high only when $a$ and $b$ co-occur beyond chance.

⚠️ This is an approximation, not an identity. Two effects are dropped:
- $Z'$ genuinely depends on the candidate → the $\log Z$ collapse is only valid for $\#(a,b)\ll Z$.
- The **surviving** $a$/$b$ occurrences are re-scored when $\#a,\#b$ shrink. Negligible only when $\#(a,b)$ is a small fraction of $\#a$ and $\#b$ — not guaranteed.
```

````{important} Code
:class: dropdown
```python
from collections import Counter

class WordPiece:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.vocab = set()

    def train(self, words):
        freq = Counter(words)
        ## a word starts fully split; every non-initial piece carries "##"
        splits = {w: [w[0]] + ["##" + c for c in w[1:]] for w in freq}
        self.vocab = {p for s in splits.values() for p in s}
        while len(self.vocab) < self.vocab_size:
            cnt, pair_cnt = Counter(), Counter()
            for w, n in freq.items():
                s = splits[w]
                for p in s:
                    cnt[p] += n
                for pr in zip(s, s[1:]):
                    pair_cnt[pr] += n
            if not pair_cnt:
                break
            ## likelihood gain, NOT raw frequency: normalize by the parts' own counts
            best = max(pair_cnt, key=lambda p: pair_cnt[p] / (cnt[p[0]] * cnt[p[1]]))
            merged = best[0] + best[1].removeprefix("##")
            self.vocab.add(merged)
            splits = {w: self._apply(s, best, merged) for w, s in splits.items()}
        return self

    @staticmethod
    def _apply(s, pair, merged):
        out, i = [], 0
        while i < len(s):
            if i < len(s) - 1 and (s[i], s[i + 1]) == pair:
                out.append(merged); i += 2
            else:
                out.append(s[i]); i += 1
        return out

    def encode(self, word):
        ## greedy LONGEST-MATCH-FIRST from the vocab alone (no merge list needed)
        out, i = [], 0
        while i < len(word):
            j = len(word)
            while j > i:
                sub = word[i:j] if i == 0 else "##" + word[i:j]
                if sub in self.vocab:
                    break
                j -= 1
            if j == i:
                return ["[UNK]"]  ## one unknown char poisons the WHOLE word
            out.append(sub); i = j
        return out

## Example
wp = WordPiece(vocab_size=30).train(
    "low low low low low lower lower newest newest newest widest widest".split())
print(wp.encode("lowest"))  ## ['low', '##e', '##st']
print(wp.encode("héllo"))   ## ['[UNK]'] <- byte-level BPE degrades gracefully instead
```
````

```{attention} Q&A
:class: dropdown
*Why the `##` prefix?*
- Marks "continues the previous piece" → detokenization is unambiguous.
- `"ing"` as a standalone word and `"##ing"` as a suffix become **different IDs with different embeddings**, which is linguistically correct.

*Why does WordPiece still emit `UNK` when BPE does not?*
- Its base vocab is the set of **characters seen in training**, not bytes. One unseen character → the entire word maps to `[UNK]`.
- → BERT silently discards rare scripts and emoji that byte-level BPE preserves exactly.

*Longest-match vs merge replay — practical difference?*
- WordPiece encoding needs only the vocab, no merge list → simpler artifact, $O(L^2)$ per word via repeated prefix search.
- Cost: it can produce segmentations the training loop never built.

*Is WordPiece still relevant?*
- ✅ Via BERT-family encoders — still the workhorse for retrieval, rerankers, classification.
- ❌ For decoder LLMs: none use it. Byte-level BPE's zero-`UNK` guarantee is non-negotiable for generation.
```

&nbsp;

### Unigram
- **What**: Prune a large candidate vocab down, keeping the tokens that best explain the corpus under a unigram LM over **all** segmentations. {cite:p}`kudo2018subword`
- **Why**:
    - BPE/WordPiece build bottom-up with irreversible greedy merges → early mistakes are permanent, and the final vocab is never scored as a whole. Unigram evaluates every token against a **global** objective.
    - Neither models segmentation **ambiguity**: 1 string = 1 segmentation. Unigram puts a distribution over segmentations → can sample.
- **How**:
    1. Seed a large candidate vocab (frequent substrings, ~10× target size).
    2. **EM**-fit $p(t)$: E-step = expected token counts under $P(s|x)$ via forward-backward; M-step = renormalize.
    3. Score each token by the corpus log-likelihood **lost** if it were removed.
    4. Drop the lowest-scoring ~10-20%; always keep base chars for coverage.
    5. Repeat 2-4 until target size.
    - **Encoding**: Viterbi argmax segmentation.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $p(t)$: Unigram probability of $t$, $\sum_{t\in\mathcal{V}}p(t)=1$.
- Misc:
    - $\mathcal{S}(x)$: All segmentations of $x$ realizable with $\mathcal{V}$.
    - $c(t;s)$: #times $t$ appears in segmentation $s$.

Model — segmentations are latent, so the string probability marginalizes over them:

$$
P(s)=\prod_{i=1}^{k}p(t_i),\qquad P(x)=\sum_{s\in\mathcal{S}(x)}P(s),\qquad P(s|x)=\frac{P(s)}{P(x)}
$$

Training — EM over $\mathcal{D}$:

$$\begin{align*}
\text{E-step: }\ &\mathbb{E}[c(t)|x]=\sum_{s\in\mathcal{S}(x)}c(t;s)P(s|x)\\
\text{M-step: }\ &p(t)\leftarrow\frac{C(t)}{\sum_{t'\in\mathcal{V}}C(t')},\quad C(t)=\sum_{x\in\mathcal{D}}\mathbb{E}[c(t)|x]
\end{align*}$$

- Both $P(x)$ and $\mathbb{E}[c(t)|x]$ come from forward-backward DP in $O(|x|\ell_\text{max})$ ($\ell_\text{max}$ = max piece length, SentencePiece default 16; $O(|x|^2)$ if unbounded), never by enumerating $\mathcal{S}(x)$.

Pruning — likelihood lost by deleting $t$, at the refit optimum:

$$
\text{loss}(t)=\ell^*(\mathcal{V})-\ell^*(\mathcal{V}\setminus\{t\})\geq0,\qquad\ell^*(\mathcal{V})=\max_{p}\sum_{x\in\mathcal{D}}\log P_p(x)
$$

- $\geq0$ ← any $p$ on $\mathcal{V}\setminus\{t\}$ extends to $\mathcal{V}$ with $p(t)=0$ → strictly larger feasible set.
- In practice estimated from the curr $p$, without refitting per candidate.

Inference — Viterbi:

$$
s^*=\arg\max_{s\in\mathcal{S}(x)}\sum_{i=1}^{k}\log p(t_i)
$$
```

````{important} Code
:class: dropdown
```python
import math

def viterbi(x, logp):
    """Most probable segmentation of x under a unigram LM. O(len(x)^2)."""
    n = len(x)
    best = [-math.inf] * (n + 1); back = [0] * (n + 1)
    best[0] = 0.0
    for j in range(1, n + 1):
        for i in range(j):  ## every token that could END at position j
            t = x[i:j]
            if t in logp and best[i] + logp[t] > best[j]:
                best[j] = best[i] + logp[t]; back[j] = i
    if best[n] == -math.inf:
        return None  ## unreachable; in practice pinned base chars guarantee coverage
    out, j = [], n
    while j > 0:
        out.append(x[back[j]:j]); j = back[j]
    return out[::-1], best[n]

## Example
p = {"un": .05, "u": .01, "n": .02, "able": .04, "ab": .01, "le": .03,
     "a": .03, "b": .01, "l": .02, "e": .06, "unable": .001}
logp = {t: math.log(v) for t, v in p.items()}
print(viterbi("unable", logp))
## (['un', 'able'], -6.2146) <- beats the single token 'unable' (log .001 = -6.9078)
```
````

```{attention} Q&A
:class: dropdown
*Why forward-backward for training but Viterbi for inference?*
- Training: every segmentation carries evidence about which tokens matter. Hard argmax over-commits and destabilizes EM.
- Inference: the model needs one deterministic tokenization, and it must match what training saw.

*How is this "the reverse of BPE" exactly?*
- BPE/WordPiece: start from base symbols and **merge up** to $|\mathcal{V}|$.
- Unigram: start from a huge substring pool and **prune down** to $|\mathcal{V}|$, re-fitting the LM after each round.

*What is subword regularization?*
- Sample $s\sim P(s|x)$ during training instead of taking the argmax → the model sees several tokenizations of the same string → robustness to tokenization noise. Biggest gains in low-resource MT.
- BPE analog: **BPE-dropout** — randomly skip merges at encode time. {cite:p}`provilkov2020bpedropout`
- ❌ Not standard in modern LLM pretraining ← data is abundant, and it fights the requirement that training and inference tokenize identically. Narrow variants survive (DeepSeek-V3 randomly splits its combined punctuation+newline tokens).

*Unigram vs BPE quality?*
- Unigram segmentations align better with morphology and match or beat BPE downstream at equal vocab size. {cite:p}`bostrom2020byte`
- BPE still dominates ← inertia, simpler/faster encoding, and tiktoken-style BBPE parallelizes trivially.

*Where is Unigram actually used today?*
- T5/mT5, ALBERT, XLNet, and it is SentencePiece's default. Essentially zero modern decoder LLMs.

*What guarantees coverage after aggressive pruning?*
- Base characters seen in training are pinned and never pruned → $\mathcal{S}(x)\neq\emptyset$ for any $x$ over that alphabet.
- Arbitrary Unicode still needs `byte_fallback` on top; pinning alone does not cover unseen characters.
```

&nbsp;

## Implementations

### SentencePiece
- **What**: Tokenizer library that trains BPE or Unigram **directly on raw text**, treating whitespace as an ordinary symbol `▁` (U+2581). {cite:p}`kudo2018sentencepiece`
- **Why**:
    - Whitespace pre-tokenization is **language-specific**: Chinese/Japanese/Thai have no spaces, so the "word" prior it encodes does not exist there.
    - Encoding spaces as data makes detokenization exact: `decode(encode(x))` reproduces the **normalized** text, byte for byte.
- **How**:
    1. Normalize (default `nmt_nfkc`, which also collapses runs of whitespace) → replace `" "` with `▁`, prepend a dummy `▁`.
    2. Train BPE or Unigram, splitting on whitespace by default (`split_by_whitespace=true`).
    3. Optional `byte_fallback=true` (default `false`): any char absent from $\mathcal{V}$ decomposes into `<0xNN>` pieces → no `UNK`. Llama 2, Gemma, and Mistral all enable it.
- Users: Llama 1/2, Gemma 2/3, Mistral ≤v0.3, T5/mT5, ALBERT, XLNet.

```{attention} Q&A
:class: dropdown
*Is "BPE vs SentencePiece" a real comparison?*
- ❌ Category error. SentencePiece is an **implementation** hosting BPE, Unigram, char, or word.
- Llama 2 and Gemma are SentencePiece **BPE**; T5 is SentencePiece **Unigram**.

*What is `▁`?*
- U+2581 LOWER ONE EIGHTH BLOCK — not a space. It marks "a space preceded this piece", so `"▁the"` and `"the"` are distinct tokens with distinct embeddings.

*Byte fallback vs `UNK`?*
- Byte fallback replaces `UNK`: an unseen char becomes its UTF-8 bytes as `<0x..>` pieces → lossless, at a token-count cost (3 tokens for one unseen CJK char).
- Off by default — a SentencePiece model trained without it still emits `UNK`.

*Why does SentencePiece prepend a dummy `▁`?*
- So a word is tokenized identically whether or not it starts the string → removes a position-dependent inconsistency.

*Why did the field drift from SentencePiece to tiktoken-style BBPE?*
- Byte-level base + regex pre-tokenizer is simpler to specify, faster, and needs no normalizer config; SentencePiece's advantages (whitespace-as-data, byte fallback) are reproducible in BBPE by construction.
```

&nbsp;

### Tiktoken
- **What**: OpenAI's byte-level BPE library — a regex pre-tokenizer plus a `bytes → rank` table, Rust core.
- **Why**: An encoding is **fully specified** by (regex, ranks, special-token map): no normalizer, no `UNK`, no config drift → bit-identical across implementations, and fast.
- **How**: Regex split → per chunk, repeatedly merge the adjacent pair of lowest rank → IDs. Special tokens are matched separately and must be explicitly allowed.

```{dropdown} Table: OpenAI Encodings
| Encoding | Models | BPE ranks | $\|\mathcal{V}\|$ | Digit rule |
|:--|:--|:--|:--|:--|
| `r50k_base` / `gpt2` | GPT-2, GPT-3 | 50,256 | 50,257 | ` ?\p{N}++` |
| `cl100k_base` | GPT-3.5, GPT-4, `text-embedding-3` | 100,256 | 100,277 | `\p{N}{1,3}` |
| `o200k_base` | GPT-4o, o-series | 199,998 | 200,019 | `\p{N}{1,3}` |

Notations:
- $\|\mathcal{V}\|$: `n_vocab` = highest assigned ID + 1, **not** #usable tokens.
```

```{attention} Q&A
:class: dropdown
*Why does $|\mathcal{V}|$ exceed the number of merges?*
- Special tokens sit above the merge table, with gaps left for future ones.
- `cl100k_base`: 100,256 merges + 5 specials, the highest at ID 100,276 → `n_vocab` 100,277 (IDs 100,261-100,275 are unassigned).

*What happens if user text contains the literal string `<|endoftext|>`?*
- tiktoken **refuses**: `encode` defaults to `allowed_special=set(), disallowed_special="all"` → `ValueError`.
- To treat it as ordinary characters: `encode_ordinary(text)` or `encode(text, disallowed_special=())`. To honor it as a control token: `allowed_special={...}`.
- Fail-loud by design — silently accepting it would be token-level prompt injection.

*Why is the pre-tokenizer regex part of the encoding's identity?*
- Same ranks + different regex = different tokenization. Reimplementations that "clean up" the regex silently produce a different model input.
```

&nbsp;

## Vocab
- **What**: The frozen token $\leftrightarrow$ ID table, shared by the embedding matrix and the output head.
- **Why**: It is the model's entire ontology of language. Everything below is downstream of choices made **before** pretraining starts, and unchangeable after.

&nbsp;

### Vocab Size
- **What**: $|\mathcal{V}|$ — trades sequence length against embedding/softmax cost.
- **Why**: The most consequential tokenizer hyperparameter.
    - $|\mathcal{V}|$⬆️ → fewer tokens per text → ⬇️seq len → ⬇️attention $O(L^2)$ cost & ⬆️text per context window.
    - $|\mathcal{V}|$⬆️ → ⬆️embedding + output-head params ($2|\mathcal{V}|d$ untied, $|\mathcal{V}|d$ tied) & ⬆️softmax FLOPs at **every** decode step.
    - $|\mathcal{V}|$⬆️ → ⬇️occurrences per token → tail tokens undertrained.
- **How**: Trend 32k (2023) → 128k-262k (2024-25), driven by multilingual + code coverage.

```{attention} Q&A
:class: dropdown
*Is there an optimal $|\mathcal{V}|$?*
- ✅ It depends on the compute budget and grows **with model size**: larger models deserve larger vocabularies. {cite:p}`tao2024scaling`
- Predicted optimum for Llama-2-70B: ≥216K, ~7× its actual 32K → the 2023 generation was systematically undersized here.

*Why is `config.vocab_size` larger than the tokenizer's vocab?*
- Padded to a convenient multiple for tensor-parallel sharding and GPU tile alignment.
- Qwen2.5: 151,665 tokenizer → 152,064 config. Gemma 3: 262,144 → 262,208. DeepSeek-V3: 128,818 → 129,280.
- Padding rows map to no string and are never training targets → the softmax denominator only ever pushes their logits down. Frameworks vary on whether they mask them explicitly.

*Where does $|\mathcal{V}|$ hurt most?*
- Small models: embedding + head can dominate the param count → an oversized vocab wastes capacity that should be in the layers.
- Large models: negligible param share, but the logit tensor $(B,L,|\mathcal{V}|)$ dominates activation memory and the final matmul.

*Does a larger vocab always mean fewer tokens?*
- Only for text resembling the tokenizer's corpus. A 256k vocab spent on languages you never use buys nothing and still costs the softmax.
```

&nbsp;

### Special Tokens
- **What**: Reserved control IDs with no ordinary-text preimage — BOS/EOS/PAD/UNK, chat roles, tool-call markers, FIM sentinels.
- **Why**: Structure must be **unspoofable**. If a user could type the string that produces `<|end_header_id|>`, the chat template becomes a prompt-injection surface.
- **How**:
    - IDs assigned **outside** the merge table; inserted by post-processing, or matched as literal special strings.
    - Encoders differ on that matching: tiktoken rejects unlisted special strings by default; HF `tokenizers` matches registered special strings in the input unless `split_special_tokens=True`. → **sanitize user text** before templating.
    - Llama 3 pre-reserves 256 slots (128,000 base + 256 → 128,256) so post-training can define new roles.

```{attention} Q&A
:class: dropdown
*Why pre-reserve unused slots?*
- Adding a token later resizes the embedding & output matrices → breaks checkpoint shapes, tensor-parallel sharding, and kernels compiled against the old $|\mathcal{V}|$.

*Is the chat template part of the tokenizer or the model?*
- Neither exactly: a **string** template applied before encoding, which emits reserved special-token strings.
- → Wrong template ⇒ wrong control tokens ⇒ the model is off-distribution even though the rendered text looks right. A top source of "the open model is worse than the API" reports.

*Why is `PAD` usually absent from decoder LLM tokenizers?*
- Causal LMs train on packed sequences, so padding never appears. It gets bolted on later for batched inference (often aliased to `EOS`), which is why attention masks must be set correctly.
```

&nbsp;

### Vocab Expansion
- **What**: Add tokens to a pretrained tokenizer, resize the embedding & head, then continue pretraining.
- **Why**: An English-centric tokenizer has terrible fertility on a target language/domain → ⬆️cost, ⬇️effective context, ⬇️quality.
- **How**:
    1. Train a small BPE on the target corpus → take the top-$k$ new pieces.
    2. Append to $\mathcal{V}$; resize embedding & output head.
    3. Init each new row as the **mean of its old subword pieces'** rows, not randomly.
    4. Continued pretraining on target-domain data.
- Risk: new rows start effectively untrained → glitch-token behavior until enough tokens are seen.

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

@torch.no_grad()
def expand_vocab(emb, new_tokens, old_tokenize):
    """emb: nn.Embedding of the pretrained model. old_tokenize: str -> [old ids]."""
    old_n, d = emb.weight.shape
    ## match the original device & dtype -- a fresh nn.Embedding is CPU/fp32 by default
    new = nn.Embedding(old_n + len(new_tokens), d,
                       device=emb.weight.device, dtype=emb.weight.dtype)
    new.weight[:old_n] = emb.weight              ## keep every pretrained row

    for i, tok in enumerate(new_tokens):
        pieces = old_tokenize(tok)               ## how the OLD tokenizer saw this string
        ## mean of its parts -> the new row starts in the right region of the space,
        ## instead of random noise the model has never had to interpret
        new.weight[old_n + i] = emb.weight[pieces].mean(dim=0)
    return new

## Example
emb = nn.Embedding(1000, 8)
vocab = {"to": 5, "ken": 7, "izer": 9}
new_emb = expand_vocab(emb, ["tokenizer"], lambda s: [vocab[p] for p in ("to", "ken", "izer")])
print(new_emb.weight.shape)                                      ## torch.Size([1001, 8])
print(torch.allclose(new_emb.weight[1000], emb.weight[[5, 7, 9]].mean(0)))  ## True
```
````

```{attention} Q&A
:class: dropdown
*Why mean-init instead of random?*
- A random row is an out-of-distribution input the model has never seen → large loss spikes and slow recovery.
- The mean of the token's own subword pieces already sits where the model expects that meaning.

*Must the output head be expanded too?*
- ✅ Both, and identically. If weights are tied, resize once and **re-tie** the head to the new parameter — otherwise the head silently keeps pointing at the old tensor.
- A row added to the embedding but not the head can be read but never generated.

*When is expansion NOT worth it?*
- When the base tokenizer is already multilingual (GPT-4o, Gemma, Tekken): the fertility win shrinks while the retraining cost and regression risk stay.
```

&nbsp;

## Modern Tokenizers
- **What**: Byte-level BPE, 128k-262k vocab, regex pre-tokenization with capped digit runs.
- **Why**: The field converged because each piece solves a distinct failure.
    - Byte base → zero OOV across every script, emoji, and binary blob.
    - Large vocab → multilingual + code efficiency, now cheap relative to model size.
    - Capped digit runs → bounded damage to arithmetic.

```{dropdown} Table: Production Tokenizers
| Family | Algorithm | Implementation | $\|\mathcal{V}\|$ | Notes |
|:--|:--|:--|:--|:--|
| GPT-2 / GPT-3 | Byte-level BPE | tiktoken `r50k_base` | 50,257 | Unbounded digit runs |
| GPT-3.5 / GPT-4 | Byte-level BPE | tiktoken `cl100k_base` | 100,277 | Digits ≤3 |
| GPT-4o / o-series | Byte-level BPE | tiktoken `o200k_base` | 200,019 | Digits ≤3, multilingual push |
| BERT | WordPiece | HF `tokenizers` | 30,522 | `##` continuations, real `[UNK]` |
| T5 / mT5 | Unigram | SentencePiece | 32,100 → 32,128 / 250,100 → 250,112 | +100 sentinels for span corruption |
| Llama 1 / 2 | BPE + byte fallback | SentencePiece | 32,000 | `split_digits`, 1 digit/token |
| Llama 3 | Byte-level BPE | tiktoken-style | 128,256 | 128,000 base + 256 reserved |
| Qwen2 / 2.5 | Byte-level BPE | HF `tokenizers` | 151,665 → 152,064 | `\p{N}` → 1 digit/token |
| Gemma 2 / 3 | BPE + byte fallback | SentencePiece | 256,000 / 262,144 → 262,208 | `split_digits`, whitespace preserved |
| Mistral v0.1-v0.3 | BPE + byte fallback | SentencePiece | 32,000 / 32,768 | |
| Mistral NeMo+ (Tekken) | Byte-level BPE | tiktoken-based | 131,072 | |
| DeepSeek-V3 | Byte-level BPE | HF `tokenizers` | 128,818 → 129,280 | Punctuation+newline merged into single tokens |

Notations:
- $a\rightarrow b$: Tokenizer vocab → padded `config.vocab_size` (embedding rows).
```

```{attention} Q&A
:class: dropdown
*Are token counts comparable across models?*
- ❌ Never. The same text is a different number of tokens per tokenizer → context limits, per-token pricing, and tokens/s benchmarks are not comparable across families.

*Why did everyone abandon 32k?*
- 32k spent almost entirely on English + code leaves other scripts on byte fallback.
- Compute-optimal vocab grows with model size, and the embedding/head cost stopped dominating: a tied 128k × 4096 table is ~0.5B params — 7% of a 7B model, <1% of a 70B one.

*Two models, same $|\mathcal{V}|$ and same algorithm — same tokenizer?*
- ❌ The pre-tokenizer regex, normalization, and the tokenizer's own training corpus all differ → different merges, different IDs. Never mix a tokenizer with another model's weights.
```

&nbsp;

## Failure Modes

### Fertility
- **What**: #tokens per word (or per byte) for a given language/domain.
- **Why**: Everything is priced and bounded in tokens — cost, context window, latency, and the sequence length the model must actually model.
- **How**: Set by what the **tokenizer's** corpus contained. English-heavy corpus → English words get whole slots; other scripts fall back toward bytes.
    - Same text across languages: up to **15×** difference in token count; even char/byte-level models differ by >4×. {cite:p}`petrov2023language`

```{attention} Q&A
:class: dropdown
*Consequences beyond API cost?*
- ⬇️Effective context for the same content.
- ⬆️Latency ← more decode steps for the same output.
- ⬇️Quality ← longer dependency spans and rarer per-token statistics for the same information.

*Fix?*
- Larger multilingual vocab at pretraining (GPT-4o, Gemma, Tekken), or vocab expansion + continued pretraining after the fact.

*Why is this a fairness issue, not just an efficiency one?*
- Per-token pricing and per-token context limits mean speakers of underrepresented languages pay more and get less context for identical content — decided entirely before the model runs.
```

&nbsp;

### Digits
- **What**: How number strings get chunked; determines arithmetic behavior.
- **Why**: `\p{N}{1,3}` chunks **left-to-right**, so a digit's token depends on the number's total length: `"1234"` → `"123"`,`"4"` but `"234"` → `"234"`.
    - → Place value is not aligned with token identity.
    - → The model must relearn addition for every chunking pattern instead of once.
- **How**: 3 regimes.
    - Unbounded runs (GPT-2): worst — splits are whatever the merges happened to learn.
    - ≤3 L2R chunks (cl100k, o200k, Llama 3, DeepSeek-V3): shorter sequences, but boundaries shift with number length.
    - 1 digit/token (Llama 1/2, Gemma via `split_digits`; Qwen2/2.5 via a bare `\p{N}`): uniform place value, ⬆️seq len.

```{attention} Q&A
:class: dropdown
*Why would right-to-left chunking be better?*
- R2L groups digits into thousands blocks, so a digit's chunk position matches its place value regardless of the number's length → one consistent addition algorithm to learn.
- Expressible in one pass with right-edge lookahead — `\p{N}{1,3}(?=(\p{N}{3})*(?!\p{N}))` turns `1234` into `1`,`234` — just not by a plain greedy `{1,3}`.

*Is 1 digit/token strictly better for math?*
- Better aligned, but every number costs $O(\text{digits})$ tokens → long arithmetic chains eat context, and the model spends decode steps on digits instead of reasoning.

*Why does the same arithmetic prompt work in one model and fail in another?*
- Often nothing to do with capability — the two tokenizers chunked the operands differently.
```

&nbsp;

### Glitch Tokens
- **What**: Vocab entries that occur (near-)never in the LM's training data → their rows never get a positive training signal → undefined behavior when triggered. {cite:p}`land2024fishing`
- **Why**: The tokenizer is trained on a **different** corpus than the model — or before dedup/filtering — so a slot can be allocated for a string the model never sees.
    - Canonical: `SolidGoldMagikarp` and friends in the GPT-2/GPT-3 vocab, artifacts of a Reddit counting subreddit later filtered out of the model's data. Prompting them produced evasion, insults, and hallucination.
- **How**:
    - **Detect**: outlier rows in the unembedding matrix, abnormally low max logit, or the model failing to repeat the token verbatim.
    - **Avoid**: train the tokenizer on the **final** data mix; audit low-frequency tokens; mask them from sampling.

```{attention} Q&A
:class: dropdown
*Why does an untrained embedding cause bizarre output rather than nothing?*
- Its input row is never updated toward any meaning, while its output row is only ever pushed **down** by the softmax denominator → it drifts into an outlier region rather than staying at init.
- The model still interprets it, so it acts as a random prompt injected mid-context.

*Is this only a legacy GPT-2 problem?*
- ❌ Under-trained tokens are found across current open models — the tokenizer/model corpus mismatch is structural, not historical.

*Why not just delete them?*
- Deleting shifts every subsequent ID → invalidates the checkpoint. Masking at sampling is the cheap fix; retraining the tokenizer is the real one.
```

&nbsp;

### Token Boundary Bias
- **What**: A prompt ending mid-merge puts the model in a tokenization state it effectively never saw in training.
- **Why**: Merges are greedy and maximal. If `",\n"` is a single token in training, a prompt ending at `","` encodes as a lone `","` — rare in training, so the continuation is off-distribution.
- **How** (mitigations):
    - Training-side: DeepSeek-V3 randomly splits a fraction of its combined punctuation+newline tokens during training, exposing the model to both forms. {cite:p}`deepseekai2024deepseekv3`
    - Inference-side: **token healing** — drop the last token, re-encode with its text as a required prefix.

```{attention} Q&A
:class: dropdown
*Where does this bite hardest?*
- FIM / code completion: the cursor lands mid-identifier by definition.
- Few-shot prompts without a trailing newline.
- Constrained decoding, where a grammar forces a token split the tokenizer would never produce.

*Why does adding a trailing space to a prompt often make output worse?*
- The space would normally be merged into the **next** word (`" the"`). Emitting it alone forces every continuation to start with a space-less token, a distribution the model rarely saw.

*Is this the model's fault?*
- ❌ The model is correctly predicting the continuation of the exact token sequence it was given. The sequence just is not the one the text implies.
```

&nbsp;

## Tokenizer-Free
- **What**: No fixed **subword** vocab — the model learns its own segmentation from raw bytes, end to end. The 256-byte alphabet stays.
- **Why**: Every failure mode above is a tokenizer artifact, and the vocab is a **hand-designed discretization frozen before learning starts** — the last major hand-crafted feature layer left in the stack.

&nbsp;

### BLT
- **Name**: Byte Latent Transformer {cite:p}`pagnoni2025byte`
- **What**: Byte-level LLM that groups bytes into **dynamically sized patches** by next-byte entropy.
- **Why**: A fixed vocab spends the same compute on `"the"` as on a rare identifier. Entropy-based patching allocates compute where the data is actually hard.
- **How**:
    1. A small byte-level LM scores next-byte entropy.
    2. High entropy → patch boundary; predictable runs → long patches.
    3. **Local encoder** pools bytes → patch representations.
    4. **Latent transformer** (the large model) runs over patches.
    5. **Local decoder** unpools patch representations → bytes.
- Status: first FLOP-controlled byte-level scaling study, to **8B params / 4T training bytes**, matching tokenizer-based performance at scale. Not in production models.

```{attention} Q&A
:class: dropdown
*How is this different from a char-level model?*
- Char-level: fixed 1 symbol = 1 step → sequence length explodes and compute is uniform.
- BLT: patch size is **data-dependent**, so predictable spans cost one step while hard spans get more → compute follows entropy, not text length.

*Does this kill tokenizers?*
- Not yet. No frontier production model ships tokenizer-free, and the whole ecosystem — pricing, context limits, KV cache accounting, evals, constrained decoding — is token-denominated.

*What does it buy beyond removing artifacts?*
- Robustness to noisy/adversarial input and better long-tail generalization, since nothing is forced through a frozen vocab.

*Why does this matter for AC?*
- A fixed vocab is a human decision about how language decomposes, imposed before any learning happens.
- Removing it moves one more inductive prior from hand-design into the learned model — the same move as replacing hand-crafted features with conv/attention.
```

&nbsp;
