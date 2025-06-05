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

```{admonition} Q&A
:class: tip, dropdown
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

```{admonition} Math
:class: note, dropdown
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
- **Why**:
- **How**:
    1. Build a large candidate vocab from the full training corpus.
        - **Candidate**: Every single substring in the corpus no longer than a preset max length (e.g., 10-12 for Alphabets; 6 for CJK).
        - **Large**: 10x-30x final vocab size.
    2. Assign each candidate a probability.
        1. Treat each token 