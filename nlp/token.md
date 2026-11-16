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
Text → integer IDs: the LM's only interface to language. Covers subword algorithms, production tokenizers & the artifacts they leak into model behavior.

Notations:
- $x$: Raw text string
- $\mathcal{V}$: Vocab (token set)
- $t$: Token (byte/char string in $\mathcal{V}$)
- $s=(t_1,\dots,t_k)$: Segmentation of $x$
- $L$: Sequence length (#tokens)
- $\mathcal{D}$: Tokenizer training corpus
- $\#a$: #occurrences of $a$ in $\mathcal{D}$
- $\#(a,b)$: #occurrences of $a$ immediately followed by $b$

Override: $k$ = #tokens in $s$ (global: class idx).

Override: $i$ = token position in $s$ (global: sample idx).

&nbsp;

## Pipeline
- **What**: Raw text $\xrightarrow{\text{normalize}}$ Clean text $\xrightarrow{\text{pre-tokenize}}$ Chunks $\xrightarrow{\text{subword model}}$ Tokens $\xrightarrow{\text{post-process}}$ Token IDs
- **Why**: Fixed, finite embedding table vs. unbounded text space → map any text onto a fixed, finite token set.
- **How**:
    1. **Normalize**: Unicode canonicalization (+ optional lowercase / accent strip).
    2. **Pre-tokenize**: Split into chunks that merges never cross.
    3. **Subword model**: Chunk → tokens (BPE / WordPiece / Unigram).
    4. **Post-process**: Insert special tokens → IDs.

```{attention} Q&A
:class: dropdown
*Is the tokenizer trained jointly with the model?*
- ❌ Trained once beforehand on a corpus sample, then frozen.
- LM never sees text, only IDs.
- Tokenizer corpus ≠ model corpus → real bug source ([glitch tokens](#glitch-tokens)).

*Rule of thumb for English on a GPT-style tokenizer?*
- ~4 chars/token, ~0.75 words/token. Code & non-English: worse.

*Which stage owns which artifact?*
- Normalization → info destroyed before the model sees it (case, accents).
- Pre-tokenization → strings that can never be one token (digit chunks, cross-word tokens).
- Subword model → which strings earn a slot.
- Post-processing → control-token correctness (chat template).
```

&nbsp;

### Normalization
- **What**: Map visually/semantically equivalent strings to one canonical form.
- **Why**: Unicode encodes many glyphs 2+ ways (`é` = U+00E9, or `e` + U+0301) → identical-looking text gets different IDs.
- **How**: NFC/NFKC → optional lowercase, accent strip, whitespace collapse.
    - Modern LLMs: near-identity ← case, accents & whitespace carry info (code indentation, proper nouns).
    - Byte-level BPE as shipped (GPT-2, tiktoken): zero normalization ← runs on raw UTF-8 bytes.
    - BERT-uncased: lowercase + strip accents ← discriminative encoder never reproduces its input.

```{attention} Q&A
:class: dropdown
*NFC vs NFKC?*
- NFC: canonical composition. Keeps the abstract characters; only their encoding changes.
- NFKC: + compatibility folding (`ﬁ`→`fi`, `①`→`1`, full-width→half-width). Lossy → changes the text itself.
- Neither is invertible ← both map several input forms to one output form.

*Why did aggressive normalization fall out of favor for LLMs?*
- Generative model must emit text → destroyed distinction is unrecoverable at decode time.
- Code: whitespace & case are syntax.
```

&nbsp;

### Pre-Tokenization
- **What**: Split text into chunks that merges never cross.
- **Why**:
    - Bounds merge search space → tractable training.
    - Hard-codes priors that corpus stats would violate: no token spans 2 words; the regex controls which punctuation may attach to a word.
- **How**: 2 families.
    - **Regex split** (GPT-2 → tiktoken → Llama 3): one pattern isolates contractions, letter runs, digit runs, punctuation runs, whitespace. Leading space stays attached (`" the"` ≠ `"the"`).
    - **Whitespace marker** (SentencePiece): `" "` → `▁` → spaces become data. Still splits on whitespace by default (`split_by_whitespace=true`); `false` allows pieces that span a space.

````{important} Code
:class: dropdown
```python
import regex  ## `re` lacks \p{L}; tiktoken uses the `regex` module too

## cl100k_base (GPT-3.5/4) pre-tokenizer pattern, verbatim from tiktoken
CL100K = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"""

class PreTokenizer:
    def __init__(self, pattern=CL100K):
        self.pat = regex.compile(pattern)

    def __call__(self, text):
        ## BPE runs INDEPENDENTLY inside each chunk -> no merge can cross these boundaries
        return self.pat.findall(text)

## Example
print(PreTokenizer()("I don't pay $12345."))
## ['I', ' don', "'t", ' pay', ' $', '123', '45', '.']
```
````

```{attention} Q&A
:class: dropdown
*Why is pre-tokenization not just an optimization?*
- Hard constraint on the vocab: `" the"` can exist; `"the cat"` never can, regardless of frequency.
- Same merge algorithm + same corpus + different regex → different vocab.

*Why attach the leading space to the word instead of emitting a space token?*
- Word-initial vs word-internal occurrences differ linguistically → separate embeddings.
- ~Halves $L$ vs. 1 token per space.

*Digit handling?*
- Set here, by the regex or `split_digits`. See [Digits](#digits).
```

&nbsp;

## Subword
- **What**: Units between chars & words, learned from corpus stats.
- **Why**: Word-level & char-level fail at opposite extremes.
    - Word: unbounded vocab + OOV on every typo/name/inflection → `UNK` destroys info.
    - Char: tiny vocab, but ~4-5× longer sequences → attention $O(L^2)$⬆️ & little meaning per step.
- **How**: Frequent strings get their own slot; rare strings decompose into pieces. Algorithms differ in which strings deserve a slot & how text is encoded with them.

```{dropdown} Table: BPE vs WordPiece vs Unigram
| | **BPE** | **WordPiece** | **Unigram** |
|:--|:--|:--|:--|
| Direction | Merge up | Merge up | Prune down |
| Criterion | Pair frequency $\#(a,b)$ | Likelihood gain $\frac{\#(a,b)}{\#a\cdot\#b}$ | Corpus NLL increase if token removed |
| Model of text | ❌ (greedy heuristic) | LM likelihood (exact LM unpublished) | Unigram LM over all segmentations |
| Encoding | Replay merges in rank order | Greedy longest-match | Viterbi argmax |
| Stochastic encoding | Only via BPE-dropout | ❌ | ✅ Native (sample $P(s\|x)$) |
| OOV | ❌ w/ byte base | `[UNK]` per word | ❌ w/ char + byte fallback |
| Used by | GPT, Llama, Qwen, DeepSeek, Mistral, Gemma | BERT family | T5/mT5, ALBERT, XLNet |
```

&nbsp;

### BPE
- **Name**: Byte-Pair Encoding {cite:p}`sennrich2016neural`
- **What**: Iteratively merge the most frequent adjacent symbol pair.
- **Why**: Frequency = cheap proxy for "deserves a slot" → high compression per slot; every string stays representable from base symbols.
- **How**:
    - **Training**:
        1. Init $\mathcal{V}$ = base alphabet (bytes/chars). Split $\mathcal{D}$ into base symbols.
        2. Count all adjacent pairs $\#(a,b)$.
        3. Merge the argmax pair into $ab$. Append $(a,b)$ to an ordered merge list.
        4. Repeat 2-3 until $|\mathcal{V}|$ hits target.
    - **Encoding**: Per chunk, repeatedly apply the lowest-rank applicable merge → deterministic, no search.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\mathcal{M}=[(a_1,b_1),\dots,(a_R,b_R)]$: Ordered merge list; index = rank.
- Hyperparams:
    - $|\mathcal{V}|$: Target vocab size.
- Misc:
    - $R=|\mathcal{V}|-|\text{base}|$: #merges.

Training:

$$
(a_r,b_r)=\arg\max_{(a,b)}\#(a,b),\qquad\mathcal{V}\leftarrow\mathcal{V}\cup\{a_rb_r\}
$$

- $r=1,\dots,R$: Merge step.
- Counts are recomputed after merge $r$ → later steps see merged symbols.

Inference (chunk $s$), while any merge applies:

$$
r^*=\min\{r:(a_r,b_r)\text{ adjacent in }s\},\qquad s\leftarrow\text{merge}(s,(a_{r^*},b_{r^*}))
$$
```

````{important} Code
:class: dropdown
```python
from collections import Counter

class BPE:
    ## Byte-level BPE: base alphabet = the 256 byte values
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
- Rank order reproduces the exact merge sequence of training → encoding matches the stats the vocab came from.
- Longest-match can produce a segmentation that BPE training never built at that position.

*Is greedy merging optimal?*
- ❌ Each merge is only locally optimal for compression. The final vocab minimizes neither corpus length nor corpus likelihood.
- Kept ← simple, fast & hard to beat by a margin that matters.

*Complexity?*
- Naive training: $O(R\cdot N)$, $N$ = corpus length ← full re-count per merge. Practical: incremental pair counts + priority queue.
- Encoding: quadratic in chunk length. Chunk ≈ one word → negligible.

*Does BPE ever emit `UNK`?*
- Only if the base alphabet misses an input symbol. Byte base (256 symbols) → never.

*Why weight pairs by word count instead of scanning raw text?*
- Same result. `{word: count}` table = deduplicated $\mathcal{D}$ → training scales w/ #distinct words, not corpus size.
```

&nbsp;

#### Byte-Level BPE
- **Name**: Byte-level Byte-Pair Encoding (BBPE) {cite:p}`radford2019language`
- **What**: BPE w/ the 256 byte values as base alphabet.
- **Why**: Character base alphabet is huge & version-dependent.
    - 150k+ assigned Unicode characters, growing every release → huge base vocab, or `UNK` for unseen scripts/emoji/binary.
    - 256 bytes cover every string, losslessly & permanently.
- **How**:
    1. UTF-8 encode → byte sequence.
    2. Regex pre-tokenize → chunks.
    3. Run BPE over bytes.
    - GPT-2 extra: bijective byte → printable-Unicode map (`" "`→`Ġ`, `"\n"`→`Ċ`) → vocab file stays readable text.

````{important} Code
:class: dropdown
```python
## Reuses BPE from the BPE block above: it is already byte-level
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
- BBPE: base alphabet = bytes. Merges run over bytes from step 1.
- Byte fallback: base alphabet = characters. `<0xNN>` pieces exist only as an escape for characters absent from the vocab.

*Can one token be an invalid UTF-8 fragment?*
- ✅ A token can be half a multi-byte character.
- → Streaming decoders must buffer bytes until they form valid UTF-8. Else emoji/CJK output flickers as replacement chars.

*What does the byte base cost?*
- Unmerged non-Latin text → 1 token/byte → 3 tokens per CJK char, 4 per emoji.
- → Non-English efficiency is a vocab budget decision, not a property of the algorithm.

*Why did GPT-2 remap bytes to printable Unicode instead of using raw bytes?*
- Vocab/merge files stay plain text. Raw whitespace & control bytes get mangled by text tooling.
- Bijection → zero effect on what gets merged.
```

&nbsp;

#### BPE-Dropout
- **What**: BPE encoding w/ each applicable merge randomly dropped w/ prob $p$. {cite:p}`provilkov2020bpedropout`
- **Why**: Subword regularization w/o leaving BPE.
    - BPE is deterministic → 1 string = 1 segmentation → model never sees how words compose; fragile to segmentation noise.
    - [Unigram](#unigram) sampling needs a separate LM + EM. BPE-dropout reuses the plain merge table.
- **How**:
    1. Each merge step: list every applicable merge occurrence.
    2. Drop each occurrence independently w/ prob $p$.
    3. Apply the highest-priority survivor. No survivor → stop.
    - **Training**: $p>0$ (paper: usually $p=0.1$).
    - **Inference**: $p=0$ → standard BPE.

````{important} Code
:class: dropdown
```python
import random

## Reuses BPE from the BPE block above (nlp/token.md)
class BPEDropout:
    def __init__(self, bpe, p=0.1):
        self.merges = bpe.merges  ## same merge table; only encoding changes
        self.p = p

    def encode(self, chunk):
        w = [bytes([b]) for b in chunk.encode()]
        while len(w) > 1:
            ## each merge OCCURRENCE survives independently w/ prob 1-p
            cand = [(self.merges[(a, b)], i) for i, (a, b) in enumerate(zip(w, w[1:]))
                    if (a, b) in self.merges and random.random() >= self.p]
            if not cand:
                break  ## all dropped -> stop, same as BPE w/ no applicable merge
            _, i = min(cand)  ## lowest rank = highest priority; ties -> leftmost
            w[i:i + 2] = [w[i] + w[i + 1]]
        return tuple(w)

## Example
bpe = BPE(vocab_size=256 + 6).train(
    "low low low low low lower lower newest newest newest widest widest".split())
print(BPEDropout(bpe, p=0.0).encode("lowest"))  ## (b'low', b'est') = plain BPE
print(BPEDropout(bpe, p=1.0).encode("lowest"))  ## (b'l', b'o', b'w', b'e', b's', b't')
print(BPEDropout(bpe, p=0.5).encode("lowest"))  ## sampled: varies per call
```
````

```{attention} Q&A
:class: dropdown
*Why is every dropped-merge output still a valid segmentation?*
- Every merge product is in $\mathcal{V}$ → every intermediate BPE state uses only vocab tokens.
- Skipping or stopping early only changes which state you end in.

*What does $p$ control?*
- Granularity. $p=0$ → standard BPE. $p=1$ → base symbols only (chars/bytes).

*BPE-dropout vs Unigram sampling?*
- Same goal: train on sampled segmentations, decode w/ one deterministic segmentation.
- Unigram samples from an explicit, learned $P(s|x)$. BPE-dropout only induces an implicit distribution via random drops.

*Gain?*
- MT: up to +2.3 BLEU vs BPE; up to +0.9 BLEU vs Unigram subword regularization.

*Used in modern LLM pretraining?*
- ❌ Not standard ← data is abundant, & train-time vs inference-time tokenization then differ.
- Narrow variant survives: DeepSeek-V3 randomly splits combined punctuation+newline tokens ([token boundary bias](#token-boundary-bias)).
```

&nbsp;

### WordPiece
- **What**: BPE w/ merges ranked by likelihood gain, not frequency. {cite:p}`schuster2012japanese`
- **Why**: Frequency over-rewards pairs whose parts are individually frequent (`"e"`+`"s"`).
    - Likelihood gain divides by the parts' own counts → merge only if the pair co-occurs beyond chance.
- **How**:
    - **Training**: BPE loop, argmax $\frac{\#(a,b)}{\#a\cdot\#b}$.
        - ⚠️ Google never published its training code. Paper gives only the criterion (max likelihood gain). The ratio is the HF course reconstruction; HF `tokenizers` itself trains the vocab w/ plain BPE frequency.
    - **Encoding**: Greedy longest-match-first per word (no merge replay). Non-initial pieces get the `##` prefix. No matching prefix → whole word → `[UNK]`.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $p(t)=\#t/Z$: Unigram probability of token $t$.
- Misc:
    - $Z=\sum_{t'\in\mathcal{V}}\#t'$: Total token count under the curr segmentation.
    - $c$: New token from merging $a,b$.
    - $q(a,b)$: #merges actually performed ($=\#(a,b)$ except overlapping self-pairs, e.g., `aaa` has $\#(a,a)=2$ but only 1 merge).

Model (corpus log-likelihood; standard reconstruction assumes a unigram LM):

$$
\ell(\mathcal{V})=\sum_{t\in\mathcal{V}}\#t\log p(t)
$$

Training, merge criterion (standard reconstruction):

$$
\text{score}(a,b)=\frac{\#(a,b)}{\#a\cdot\#b}
$$

Training, count updates on merging $a,b\rightarrow c$:

$$\begin{align*}
\#c&\leftarrow q(a,b)\\
\#a&\leftarrow\#a-q(a,b)\\
\#b&\leftarrow\#b-q(a,b)
\end{align*}$$

- $a=b$: both updates hit one count → $\#a\leftarrow\#a-2q(a,a)$.
```

```{tip} Derivation
:class: dropdown
*How does the likelihood criterion become that ratio?*
1. Count only merged occurrences. Each contributes $\log p(c)$ instead of $\log p(a)+\log p(b)$. Per-occurrence gain:

    $$
    \Delta(a,b)=\log p(c)-\log p(a)-\log p(b)
    $$

2. Substitute $p(t)=\#t/Z$ and $\#c=q(a,b)\approx\#(a,b)$, w/ $Z'$ the post-merge total:

    $$
    \Delta(a,b)=\log\frac{\#(a,b)}{\#a\cdot\#b}+2\log Z-\log Z'
    $$

3. $\#(a,b)\ll Z\Rightarrow Z'=Z-\#(a,b)\approx Z$ → tail collapses to $\log Z$, identical for every candidate at that step:

    $$
    \Delta(a,b)\approx\log\frac{\#(a,b)}{\#a\cdot\#b}+\log Z
    $$

4. → $\arg\max\Delta=\arg\max\frac{\#(a,b)}{\#a\cdot\#b}$: PMI-like association score. High only if $a$ & $b$ co-occur beyond chance.

⚠️ Approximation, not identity. 2 dropped effects:
- $Z'$ depends on the candidate → $\log Z$ collapse valid only for $\#(a,b)\ll Z$.
- Surviving $a$/$b$ occurrences are re-scored when $\#a,\#b$ shrink. Negligible only if $\#(a,b)$ is a small fraction of $\#a$ & $\#b$.
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
- Marks "continues the previous piece" → unambiguous detokenization.
- `"ing"` (word) vs `"##ing"` (suffix) → different IDs & embeddings. Linguistically correct.

*Why does WordPiece still emit `UNK` when BPE does not?*
- Base vocab = characters seen in training, not bytes. One unseen character → whole word → `[UNK]`.
- → BERT discards rare scripts & emoji that byte-level BPE keeps exactly.

*Longest-match vs merge replay: practical difference?*
- WordPiece encoding needs only the vocab, no merge list → simpler artifact. Cost: quadratic in word length (repeated prefix search).
- Can produce segmentations the training loop never built.

*Is WordPiece still relevant?*
- ✅ BERT-family encoders: still the workhorse for retrieval, rerankers, classification.
- ❌ Decoder LLMs: none use it ← generation needs byte-level BPE's zero-`UNK` guarantee.
```

&nbsp;

### Unigram
- **What**: Top-down vocab pruning under a unigram LM over all segmentations. {cite:p}`kudo2018subword`
- **Why**: 2 BPE/WordPiece gaps.
    - Greedy, irreversible merges → final vocab never scored as a whole. Unigram scores every token against one global objective.
    - 1 string = 1 segmentation → no model of ambiguity. Unigram: distribution over segmentations → can sample.
- **How**:
    - **Training**:
        1. Seed a large candidate vocab: required chars + frequent substrings (SentencePiece: 1M).
            - Required chars: cover a `character_coverage` fraction of the corpus (SentencePiece default 0.9995).
        2. EM-fit $p(t)$. E-step: expected token counts under $P(s|x)$ via forward-backward. M-step: renormalize.
        3. Score each token by the corpus log-likelihood lost if removed.
        4. Keep the top-scoring tokens (SentencePiece: 75% per round). Never prune required chars.
        5. Repeat 2-4 until target size.
    - **Encoding**: Viterbi argmax segmentation.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $p(t)$: Unigram probability of $t$, $\sum_{t\in\mathcal{V}}p(t)=1$.
- Misc:
    - $\mathcal{S}(x)$: All segmentations of $x$ realizable w/ $\mathcal{V}$.
    - $c(t;s)$: #times $t$ appears in segmentation $s$.

Model (segmentation is latent → string probability marginalizes over it):

$$
P(s)=\prod_{i=1}^{k}p(t_i),\qquad P(x)=\sum_{s\in\mathcal{S}(x)}P(s),\qquad P(s|x)=\frac{P(s)}{P(x)}
$$

Training, EM over $\mathcal{D}$:

$$\begin{align*}
\text{E-step: }\ &\mathbb{E}[c(t)|x]=\sum_{s\in\mathcal{S}(x)}c(t;s)P(s|x)\\
\text{M-step: }\ &p(t)\leftarrow\frac{C(t)}{\sum_{t'\in\mathcal{V}}C(t')},\quad C(t)=\sum_{x\in\mathcal{D}}\mathbb{E}[c(t)|x]
\end{align*}$$

- $C(t)$: Expected count of $t$ over $\mathcal{D}$.
- $P(x)$ & $\mathbb{E}[c(t)|x]$ come from forward-backward DP in $O(|x|\ell_\text{max})$, never by enumerating $\mathcal{S}(x)$.
- $\ell_\text{max}$: Max piece length (SentencePiece default 16). Unbounded → $O(|x|^2)$.

Training, pruning (likelihood lost by deleting $t$, at the refit optimum):

$$
\text{loss}(t)=\ell^*(\mathcal{V})-\ell^*(\mathcal{V}\setminus\{t\})\geq0,\qquad\ell^*(\mathcal{V})=\max_{p}\sum_{x\in\mathcal{D}}\log P_p(x)
$$

- $\ell^*(\mathcal{V})$: Max corpus log-likelihood w/ vocab $\mathcal{V}$.
- $\geq0$ ← any $p$ on $\mathcal{V}\setminus\{t\}$ extends to $\mathcal{V}$ w/ $p(t)=0$ → larger feasible set.
- In practice: estimated from the curr $p$, w/o refitting per candidate.

Inference (Viterbi):

$$
s^*=\arg\max_{s\in\mathcal{S}(x)}\sum_{i=1}^{k}\log p(t_i)
$$
```

````{important} Code
:class: dropdown
```python
import math

class Unigram:
    ## Inference only: p(t) given (EM training & pruning omitted)
    def __init__(self, p):
        self.logp = {t: math.log(v) for t, v in p.items()}

    def encode(self, x):
        ## Viterbi: most probable segmentation of x. O(len(x)^2)
        n = len(x)
        best = [-math.inf] * (n + 1); back = [0] * (n + 1)
        best[0] = 0.0
        for j in range(1, n + 1):
            for i in range(j):  ## every token that could END at position j
                t = x[i:j]
                if t in self.logp and best[i] + self.logp[t] > best[j]:
                    best[j] = best[i] + self.logp[t]; back[j] = i
        if best[n] == -math.inf:
            return None  ## unreachable; in practice pinned base chars guarantee coverage
        out, j = [], n
        while j > 0:
            out.append(x[back[j]:j]); j = back[j]
        return out[::-1], best[n]

## Example
p = {"un": .05, "u": .01, "n": .02, "able": .04, "ab": .01, "le": .03,
     "a": .03, "b": .01, "l": .02, "e": .06, "unable": .001}
print(Unigram(p).encode("unable"))
## (['un', 'able'], -6.2146...) <- beats the single token 'unable' (log .001 = -6.9078)
```
````

```{attention} Q&A
:class: dropdown
*Why forward-backward for training but Viterbi for inference?*
- Training: every segmentation carries evidence about which tokens matter. Hard argmax over-commits → destabilizes EM.
- Inference: model needs one deterministic tokenization that matches training.

*How is this "the reverse of BPE"?*
- BPE/WordPiece: start from base symbols, merge up to $|\mathcal{V}|$.
- Unigram: start from a huge substring pool, prune down to $|\mathcal{V}|$, refit the LM after each round.

*What is subword regularization?*
- Sample $s\sim P(s|x)$ during training instead of the argmax → model sees several tokenizations of one string → robust to tokenization noise.
- Biggest gains: low-resource MT.
- BPE analog: [BPE-dropout](#bpe-dropout).

*Unigram vs BPE quality?*
- Unigram segmentations align better w/ morphology. Downstream: matches or beats BPE at equal vocab size. {cite:p}`bostrom2020byte`
- BPE still dominates decoder LLMs ← inertia, simpler/faster encoding; tiktoken-style BBPE parallelizes trivially.
- Unigram today: T5/mT5, ALBERT, XLNet; SentencePiece's default `model_type`.

*What guarantees coverage after aggressive pruning?*
- Required chars are never pruned → $\mathcal{S}(x)\neq\emptyset$ for any $x$ over that alphabet.
- Rare chars outside `character_coverage` & unseen chars → `UNK`, unless `byte_fallback=true`.
```

&nbsp;

## Implementations

### SentencePiece
- **What**: Language-independent tokenizer library w/ whitespace as an ordinary symbol `▁` (U+2581). {cite:p}`kudo2018sentencepiece`
- **Why**:
    - Word pre-tokenizers are language-specific ← Chinese/Japanese/Thai have no spaces → no "word" prior.
    - Spaces as data → exact detokenization: `decode(encode(x))` = normalized $x$, byte for byte.
- **How**:
    1. Normalize (default `nmt_nfkc` + `remove_extra_whitespaces=true`) → `" "` → `▁` → prepend dummy `▁`.
    2. Train BPE or Unigram (default). Still splits on whitespace by default (`split_by_whitespace=true`).
    3. Optional `byte_fallback=true` (default `false`): char ∉ $\mathcal{V}$ → `<0xNN>` byte pieces → no `UNK`.

```{attention} Q&A
:class: dropdown
*Is "BPE vs SentencePiece" a real comparison?*
- ❌ Category error. SentencePiece = implementation hosting BPE, Unigram, char, or word models.
- Llama 2 & Gemma: SentencePiece BPE. T5: SentencePiece Unigram.

*What is `▁`?*
- U+2581 LOWER ONE EIGHTH BLOCK, not a space. Marks "a space preceded this piece".
- → `"▁the"` & `"the"` are distinct tokens w/ distinct embeddings.

*Byte fallback vs `UNK`?*
- Byte fallback replaces `UNK`: unseen char → its UTF-8 bytes as `<0x..>` pieces → lossless, more tokens (3 per unseen CJK char).
- Off by default → a model trained w/o it still emits `UNK`.
- Enabled in Llama 2, Gemma, Mistral.

*Why prepend a dummy `▁`?*
- Word tokenizes identically at string start & mid-string → removes a position-dependent inconsistency.

*Why did the field drift from SentencePiece to tiktoken-style BBPE?*
- Byte base + regex pre-tokenizer: simpler to specify, faster, no normalizer config.
- SentencePiece's advantages (whitespace-as-data, byte fallback) hold in BBPE by construction.
```

&nbsp;

### Tiktoken
- **What**: OpenAI's byte-level BPE library.
- **Why**: Reproducible, fast encoding.
    - Encoding fully specified by (regex, ranks, special-token map) → no normalizer, no `UNK`, no config drift → bit-identical across implementations.
- **How**: Regex split → per chunk, merge the lowest-rank adjacent pair until none applies → IDs.
    - Artifact: `bytes → rank` table. Rust core.
    - Special tokens: matched separately; must be explicitly allowed.

```{dropdown} Table: OpenAI Encodings
| Encoding | Models | BPE ranks | $\|\mathcal{V}\|$ | Digit rule |
|:--|:--|:--|:--|:--|
| `r50k_base` / `gpt2` | GPT-2, GPT-3 | 50,256 | 50,257 | ` ?\p{N}++` |
| `cl100k_base` | GPT-3.5, GPT-4, `text-embedding-3` | 100,256 | 100,277 | `\p{N}{1,3}` |
| `o200k_base` | GPT-4o, o-series | 199,998 | 200,019 | `\p{N}{1,3}` |

Notations:
- $\|\mathcal{V}\|$: `n_vocab` = highest assigned ID + 1, not #usable tokens.
```

```{attention} Q&A
:class: dropdown
*Why does $|\mathcal{V}|$ exceed the number of BPE ranks?*
- Special tokens sit above the merge table, w/ gaps left for future ones.
- `cl100k_base`: 100,256 ranks (256 base bytes + 100,000 merges) + 5 specials, highest at ID 100,276 → `n_vocab` 100,277 (IDs 100,261-100,275 unassigned).

*What happens if user text contains the literal string `<|endoftext|>`?*
- tiktoken refuses: `encode` defaults to `allowed_special=set(), disallowed_special="all"` → `ValueError`.
- Treat as ordinary text: `encode_ordinary(text)` or `encode(text, disallowed_special=())`.
- Honor as control token: `allowed_special={...}`.
- Fail-loud by design ← silent acceptance = token-level prompt injection.

*Why is the pre-tokenizer regex part of the encoding's identity?*
- Same ranks + different regex → different tokenization.
- Reimplementations that "clean up" the regex silently change the model input.
```

&nbsp;

## Vocab
- **What**: Frozen token $\Leftrightarrow$ ID table, shared by the input embedding & output head.

&nbsp;

### Vocab Size
- **What**: $|\mathcal{V}|$, #entries in the vocab.
- **Why**: Top tokenizer hyperparam. Sets the seq-len vs. vocab-cost tradeoff.
    - $|\mathcal{V}|$⬆️ → #tokens per text⬇️ → $L$⬇️ → attention cost⬇️ & text per context window⬆️.
    - $|\mathcal{V}|$⬆️ → embedding + head params⬆️ ($2|\mathcal{V}|d$ untied, $|\mathcal{V}|d$ tied; $d$ = model dim) & softmax FLOPs⬆️ at every decode step.
    - $|\mathcal{V}|$⬆️ → occurrences per token⬇️ → tail tokens undertrained.
- **How**: Fix before tokenizer training. Pick by compute budget, model size & target languages/domains.

```{attention} Q&A
:class: dropdown
*Is there an optimal $|\mathcal{V}|$?*
- ✅ Depends on compute budget. Grows w/ model size → larger models deserve larger vocabs. {cite:p}`tao2024scaling`
- Predicted optimum for Llama-2-70B: ≥216K, ~7× its actual 32K → the 2023 generation was undersized.

*Trend?*
- 32k (2023) → 128k-262k (2024-25) ← multilingual + code coverage. See [Modern Tokenizers](#modern-tokenizers).

*Why is `config.vocab_size` larger than the tokenizer's vocab?*
- Padded to a convenient multiple → tensor-parallel sharding & GPU tile alignment.
- Qwen2.5: 151,665 tokenizer → 152,064 config. Gemma 3: 262,144 → 262,208. DeepSeek-V3: 128,818 → 129,280.
- Padding rows map to no string & are never targets → softmax denominator only pushes their logits down. Frameworks differ on explicit masking.

*Where does $|\mathcal{V}|$ hurt most?*
- Small models: embedding + head can dominate param count → oversized vocab wastes capacity the layers need.
- Large models: negligible param share. But logit tensor $(B,L,|\mathcal{V}|)$ dominates activation memory & the final matmul ($B$ = batch size).

*Does a larger vocab always mean fewer tokens?*
- Only for text that resembles the tokenizer's corpus.
- 256k vocab spent on languages you never use → no gain, full softmax cost.
```

&nbsp;

### Special Tokens
- **What**: Reserved control IDs w/o ordinary-text preimage (BOS/EOS/PAD/UNK, chat roles, tool-call markers, FIM sentinels).
- **Why**: Control structure that user text cannot forge.
    - User-typable control string → chat template becomes a prompt-injection surface.
- **How**:
    - IDs assigned outside the merge table. Inserted by post-processing, or matched as literal special strings.
    - Encoders differ on literal matching:
        - tiktoken: rejects unlisted special strings by default.
        - HF `transformers` tokenizers: match registered special strings in input unless `split_special_tokens=True`.
        - → Sanitize user text before templating.
    - Llama 3: 256 reserved slots (128,000 + 256 = 128,256) → post-training can define new roles.

```{attention} Q&A
:class: dropdown
*Why pre-reserve unused slots?*
- New token later → resize embedding & output matrices → breaks checkpoint shapes, tensor-parallel sharding & kernels compiled for the old $|\mathcal{V}|$.

*Is the chat template part of the tokenizer or the model?*
- Neither: a string template applied before encoding. It emits reserved special-token strings.
- Wrong template → wrong control tokens → model off-distribution, although the rendered text looks right.
- Top source of "the open model is worse than the API" reports.

*Why is `PAD` usually absent from decoder LLM tokenizers?*
- Causal LMs train on packed sequences → padding never appears.
- Added later for batched inference (often aliased to `EOS`) → attention masks must be set correctly.
```

&nbsp;

### Vocab Expansion
- **What**: Add tokens to a pretrained tokenizer & model, then continue pretraining.
- **Why**: English-centric tokenizer → high [fertility](#fertility) on target language/domain → cost⬆️, effective context⬇️, quality⬇️.
- **How**:
    1. Train a small BPE on the target corpus → take the top-$k$ new pieces.
    2. Append to $\mathcal{V}$. Resize embedding & output head.
    3. Init each new row as the mean of its old subword pieces' rows, not randomly.
    4. Continue pretraining on target-domain data.

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class VocabExpansion:
    def __init__(self, old_tokenize):
        self.old_tokenize = old_tokenize  ## str -> [old ids]

    @torch.no_grad()
    def _grow(self, W, new_tokens):
        ## W: (V_old, d). nn.Embedding.weight & a bias-free LM head's weight share this layout
        ## mean of its OLD subword pieces -> new row starts in the right region of the space,
        ## not random noise the model never had to interpret
        rows = [W[self.old_tokenize(t)].mean(dim=0) for t in new_tokens]
        ## cat keeps W's device & dtype; every pretrained row is kept as-is
        return nn.Parameter(torch.cat([W, torch.stack(rows)]))

    def __call__(self, emb, head, new_tokens):
        tied = head.weight is emb.weight
        emb.weight = self._grow(emb.weight, new_tokens)
        emb.num_embeddings = emb.weight.shape[0]
        ## tied -> re-tie to the NEW tensor; untied -> grow the head separately
        head.weight = emb.weight if tied else self._grow(head.weight, new_tokens)
        head.out_features = head.weight.shape[0]
        return emb, head

## Example
emb = nn.Embedding(1000, 8)
head = nn.Linear(8, 1000, bias=False)  ## untied LM head, weight: (1000, 8)
vocab = {"to": 5, "ken": 7, "izer": 9}
expand = VocabExpansion(lambda s: [vocab[p] for p in ("to", "ken", "izer")])
old = emb.weight.clone()
emb, head = expand(emb, head, ["tokenizer"])
print(emb.weight.shape, head.weight.shape)              ## torch.Size([1001, 8]) torch.Size([1001, 8])
print(torch.allclose(emb.weight[1000], old[[5, 7, 9]].mean(0)))  ## True
print(head(emb(torch.tensor([1000]))).shape)            ## torch.Size([1, 1001])
```
````

```{attention} Q&A
:class: dropdown
*Why mean-init instead of random?*
- Random row = out-of-distribution input → large loss spikes, slow recovery.
- Mean of the token's own subword pieces already sits where the model expects that meaning.

*Must the output head be expanded too?*
- ✅ Both, identically. Tied weights → resize once & re-tie the head to the new parameter. Else the head keeps pointing at the old tensor.
- Row added to embedding but not head → readable, never generated.

*Risk?*
- New rows start effectively untrained → [glitch-token](#glitch-tokens)-like behavior until enough training tokens.

*When is expansion NOT worth it?*
- Base tokenizer already multilingual (GPT-4o, Gemma, Tekken) → fertility gain shrinks; retraining cost & regression risk stay.
```

&nbsp;

## Modern Tokenizers
- **What**: Dominant recipe: byte-level BPE, 128k-262k vocab, regex pre-tokenizer w/ capped digit runs.
- **Why**: Each part fixes a distinct failure.
    - Byte base → zero OOV (all scripts, emoji, binary).
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
- ❌ Never. Same text → different #tokens per tokenizer.
- → Context limits, per-token pricing & tokens/s benchmarks are not comparable across families.

*Why did everyone abandon 32k?*
- 32k spent almost entirely on English + code → other scripts fall back to bytes.
- Compute-optimal vocab grows w/ model size. Embedding/head cost stopped dominating: tied 128k × 4096 table ≈ 0.5B params = 7% of 7B, <1% of 70B.

*Two models w/ same $|\mathcal{V}|$ & same algorithm: same tokenizer?*
- ❌ Pre-tokenizer regex, normalization & tokenizer corpus all differ → different merges, different IDs.
- Never mix a tokenizer w/ another model's weights.
```

&nbsp;

## Failure Modes

### Fertility
- **What**: #tokens per word (or per byte) for a given language/domain.
- **Why**: Cost, context window, latency & modeled sequence length are all counted in tokens.
- **How**: Set by the tokenizer's corpus. English-heavy corpus → English words get whole slots; other scripts fall back toward bytes.
    - Same text across languages: up to 15× token-count gap. Char/byte-level models: still >4×. {cite:p}`petrov2023language`

```{attention} Q&A
:class: dropdown
*Consequences beyond API cost?*
- Effective context⬇️ for the same content.
- Latency⬆️ ← more decode steps for the same output.
- Quality⬇️ ← longer dependency spans & rarer per-token stats for the same info.

*Fix?*
- Larger multilingual vocab at pretraining (GPT-4o, Gemma, Tekken), or [vocab expansion](#vocab-expansion) + continued pretraining.

*Why is this a fairness issue, not just efficiency?*
- Per-token pricing & context limits → speakers of underrepresented languages pay more & get less context for identical content.
- Decided entirely before the model runs.
```

&nbsp;

### Digits
- **What**: Chunking rule for digit strings.
- **Why**: Token boundaries ≠ place value → arithmetic depends on the tokenizer.
    - `\p{N}{1,3}` chunks left-to-right → a digit's token depends on the number's length: `"1234"` → `"123"`,`"4"`; `"234"` → `"234"`.
    - → Model relearns addition per chunking pattern instead of once.
- **How**: 3 regimes, set at [pre-tokenization](#pre-tokenization).
    - Unbounded runs: splits = whatever merges were learned → worst.
    - ≤3 digits, left-to-right: shorter sequences; boundaries shift w/ number length.
    - 1 digit/token: uniform place value; $L$⬆️.

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
*Why would right-to-left chunking be better?*
- R2L → thousands blocks → a digit's chunk position matches its place value for any number length → one addition algorithm to learn.
- One-pass regex w/ right-edge lookahead: `\p{N}{1,3}(?=(\p{N}{3})*(?!\p{N}))` → `1234` = `1`,`234`. Plain greedy `{1,3}` cannot do it.

*Is 1 digit/token strictly better for math?*
- Better aligned. But every number costs $O(\text{digits})$ tokens → long arithmetic chains eat context & decode steps.

*Why does the same arithmetic prompt work in one model & fail in another?*
- Often not capability: the 2 tokenizers chunked the operands differently.
```

&nbsp;

### Glitch Tokens
- **What**: Vocab tokens (near-)absent from the LM's training data. {cite:p}`land2024fishing`
- **Why**: Tokenizer corpus ≠ LM corpus (or tokenizer trained before dedup/filtering).
    - → Slot allocated for a string the LM never sees → row gets no positive training signal → undefined behavior when triggered.
- **How**:
    - **Detect**: outlier rows in the unembedding matrix; abnormally low max logit; model fails to repeat the token verbatim.
    - **Avoid**: train the tokenizer on the final data mix; audit low-frequency tokens; mask them from sampling.

```{note} Example
:class: dropdown
- `SolidGoldMagikarp` & similar entries in the GPT-2/GPT-3 vocab: Reddit r/counting usernames.
- Frequent in the tokenizer corpus, (near-)absent in the model's training data.
- Prompting them → evasion, insults, hallucination.
```

```{attention} Q&A
:class: dropdown
*Why does an untrained embedding cause bizarre output instead of nothing?*
- Input row: never updated toward any meaning. Output row: only pushed down by the softmax denominator → drifts into an outlier region.
- Model still interprets it → acts as a random prompt injected mid-context.

*Is this only a legacy GPT-2 problem?*
- ❌ Under-trained tokens exist across current open models. Tokenizer/model corpus mismatch is structural.

*Why not just delete them?*
- Deleting shifts every later ID → invalidates the checkpoint.
- Masking at sampling = cheap fix. Retraining the tokenizer = real fix.
```

&nbsp;

### Token Boundary Bias
- **What**: Prompt ending mid-merge → token sequence the model rarely saw in training.
- **Why**: Merges are greedy & maximal.
    - `",\n"` = one token in training → prompt ending at `","` encodes a lone `","` → off-distribution continuation.
- **How**: 2 mitigations.
    - **Training-side**: DeepSeek-V3 randomly splits a fraction of its combined punctuation+newline tokens → model sees both forms. {cite:p}`deepseekai2024deepseekv3`
    - **Inference-side**: **token healing**. Drop the last prompt token. Constrain the next token to start w/ its text.

```{attention} Q&A
:class: dropdown
*Where does this bite hardest?*
- FIM / code completion: cursor lands mid-identifier by definition.
- Few-shot prompts w/o a trailing newline.
- Constrained decoding: grammar forces a token split the tokenizer never produces.

*Why does a trailing space in a prompt often make output worse?*
- Space normally merges into the next word (`" the"`).
- Lone space → every continuation must start w/ a space-less token → distribution the model rarely saw.

*Is this the model's fault?*
- ❌ Model correctly predicts the continuation of the exact token sequence given. That sequence is not the one the text implies.
```

&nbsp;

## Tokenizer-Free
- **What**: LM on raw bytes w/ learned segmentation, no subword vocab.
- **Why**: Every failure mode above is a tokenizer artifact.
    - Vocab = hand-designed discretization, frozen before learning starts → last major hand-crafted feature layer in the stack.

&nbsp;

### BLT
- **Name**: Byte Latent Transformer {cite:p}`pagnoni2025byte`
- **What**: Byte-level LLM over dynamically sized byte patches, split by next-byte entropy.
- **Why**: Token LMs spend equal compute per token, regardless of how predictable it is.
    - Entropy-based patching → compute goes where the data is hard.
- **How**:
    1. Small byte-level LM scores next-byte entropy.
    2. High entropy → patch boundary. Predictable runs → long patches.
    3. **Local encoder**: pools bytes → patch representations.
    4. **Latent transformer** (the large model): runs over patches.
    5. **Local decoder**: unpools patch representations → bytes.

```{attention} Q&A
:class: dropdown
*How is this different from a char-level model?*
- Char-level: 1 symbol = 1 step → $L$ explodes; compute is uniform.
- BLT: data-dependent patch size → predictable spans cost 1 step, hard spans get more → compute follows entropy, not text length.

*Does it scale?*
- First FLOP-controlled byte-level scaling study: up to 8B params / 4T training bytes.
- Matches tokenizer-based LLM performance at scale.

*Does this kill tokenizers?*
- Not yet. No frontier production model ships tokenizer-free.
- Pricing, context limits, KV cache accounting, evals & constrained decoding are all token-denominated.

*What does it buy beyond removing artifacts?*
- Robustness to noisy/adversarial input & better long-tail generalization ← nothing forced through a frozen vocab.

*What does this tell us about AC?*
- Fixed vocab = human decision about how language decomposes, imposed before learning.
- Removing it moves one more inductive prior from hand-design into the learned model. Same move as conv/attention replacing hand-crafted features.
```

&nbsp;
