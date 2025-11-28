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
- **What**: Sequence $\rightarrow$ Tokens
- **Why**: Machines can only read numbers.

```{attention} Q&A
:class: dropdown
*Why subword-level vocab? Why not whole words? Why not characters?*
- Word-level vocab explode with out-of-vocab words.
- Char-level vocab misses morphology.
- Subword offers a fixed-size, open-vocab symbol set which can handle rare words while maintaining morphology.
```

## BPE (Byte-Pair Encoding)
- **What**: Frequency-based subword tokenization.
- **How**:
	1. Start with single characters as tokens.
	2. Count every pair of adjacent tokens.
	3. Merge the **most frequent** pair into one token.
	4. Repeat Step 2-3 till reaching vocab size.

## WordPiece
- **What**: Likelihood-based subword tokenization.
- **Why**: 
	- BPE merges by frequency $\rightarrow$ Greedy $\rightarrow$ Doesn't care about the merge's impact on the overall LM probability
	- WordPiece aims to **maximize corpus likelihood** under a unigram LM over subword tokens.
- **How**:
	1. Start with single characters as tokens.
	2. Compute corpus likelihood gain for each pair of adjacent tokens.
	3. Merge the pair with the **highest likelihood gain** into one token.
	4. Repeat Step 2-3 till reaching vocab size.

```{note} Math
:class: dropdown
Notations:
- IO:
	- $w_i$: $i$th word (i.e., space-separated tokens) in training corpus.
	- $t_{i,j}$: $j$th subtoken in word $w_i$.
- Hyperparams:
	- $M$: #words in training corpus.
	- $m_{w_i}$: #subtokens in word $w_i$.
- Params:
	- $\mathcal{V}$: Vocab of subword tokens.
- Misc:
	- $L(\mathcal{V})$: Corpus likelihood given curr vocab.
	- $a,b$: 2 selected tokens.
	- $c$: New token if merging $ab$.

Objective: Maximize Corpus Likelihood

$$
L(\mathcal{V})=\prod_{i=1}^{M}P(w_i|\mathcal{V})=\prod_{i=1}^{M}\prod_{j=1}^{m_{w_i}}t_{i,j}
$$

Likelihood: #Counts of curr token over #Counts of all tokens in corpus from vocab:

$$
P(a)=\frac{\# a}{\sum_{b\in\mathcal{V}}\# b}
$$

Merging $a$&$b$ into $c$:

$$\begin{align*}
\# c\leftarrow \# c + \# ab \\
\# a\leftarrow \# a - \# ab	\\
\# b\leftarrow \# b - \# ab
\end{align*}$$

Likelihood Gain:

$$
\Delta(a,b)=\sum_{ab}\left[\log P_\mathrm{postmerge}(c)-\log P(a)-\log P(b)\right]
$$
```

## Unigram
- **What**: Reverse of BPE/WordPiece $\leftarrow$ Prune a large initial vocab instead of merging from chars
- **Why**: Pruning + Likelihood can avoid early bad merge decisions & yield more natural segmentations.
- **How**:
    1. Build a large candidate vocab from the full training corpus.
        - **Candidate**: Every single substring in the corpus no longer than a preset max length (e.g., 10-12 for Alphabets; 6 for CJK).
        - **Large**: 10x-30x final vocab size.
    2. Assign each candidate a probability.
        1. Treat each token $t$ as a symbol in a **unigram language model** with parameter $p(t)$, where $\sum_t p(t)=1$.
        2. A string $x$ can be segmented in many ways; for a segmentation $s=(t_1,\dots,t_k)$, define:
            - $P(s)=\prod_{i=1}^{k} p(t_i)$
            - $P(x)=\sum_{s \in \text{Seg}(x)} P(s)$  (sum over all valid segmentations)
        3. Use dynamic programming to compute:
            - **Best segmentation** (Viterbi): $\arg\max_s P(s)$ for tokenizing at inference time.
            - **Marginals / expected counts** (Forward–Backward) needed for training.
    3. Fit token probabilities with EM (Maximum Likelihood).
        1. **E-step**: compute expected counts of each token under $P(s \mid x)$ across the corpus (via Forward–Backward).
        2. **M-step**: update $p(t)$ proportional to its expected count (with smoothing/prior in practice):
            - $p(t) \leftarrow \dfrac{\mathbb{E}[\text{count}(t)]}{\sum_{t'} \mathbb{E}[\text{count}(t')]}$
    4. Prune the vocabulary (the “reverse of merges” part).
        1. Score tokens by how harmful it would be to remove them (utility/importance), often approximated by the **increase in negative log-likelihood** if the token is dropped.
        2. Remove a fraction of the lowest-utility tokens (e.g., 10–20%) while keeping:
            - required tokens (special tokens, sometimes all single characters/bytes)
            - an unknown/fallback mechanism to guarantee coverage
        3. Re-run EM on the reduced vocab.
        4. Repeat prune + re-fit until reaching the target vocab size.
    5. Tokenization at inference:
        - Given final vocab + $p(t)$, find the **most likely** segmentation with Viterbi DP (typically maximizing $\sum_i \log p(t_i)$).