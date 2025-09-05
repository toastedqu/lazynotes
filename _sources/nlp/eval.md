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
            - $\rightarrow$ Compare cand sentence with each ref sentence. If the n-gram matches any ref sentence, it's correct.
        - *Max count in any reference sentence?*
            - It's very easy to cheat precision by **repetition** of correct words.
            - $\rightarrow$ Restrain it in the scope of ref sentence instead.
    - **Geometric Average**: Of all n-grams up to the specified n.
    - **Brevity Penality**: Penalize overly short sentences.
        - *Why?*
            - It's very easy to cheat precision with super short sentences $\leftarrow$ The occurrence ratio of correct words in short cand sentences is higher

```{admonition} Math
:class: note, dropdown
Notations:
- $c$: Candidate length (i.e., #words in cand).
- $r_i$: Reference length (i.e., #words in ref).
- $r=\arg\min_{r_i}|r_i-c|$: Effective reference length (i.e., Reference length with the smallest absolute difference from cand)
    - Choose the shorter one if tied.

Clipped N-gram Precision:

$$
p_n=\frac{\max_{\text{ref}}(\#_{\text{ref}}\text{correct predicted n-grams})}{\#_{\text{cand}}\text{total predicted n-grams}}
$$

Geometric Average of Clipped N-gram Precisions:

$$
\bar{p}_N=\prod_{n=1}^Np_n^{\frac{1}{N}}
$$

Brevity Penalty:

$$
BP=\begin{cases}
1 & \text{if }c>r \\
e^{\frac{1-r}{c}} & \text{if }c\leq r
\end{cases}
$$

BLEU:

$$
BLEU_N=BP\times \bar{p}_N
$$
```

```{admonition} Q&A
:class: tip, dropdown
*Cons?*
- Exact word matches $\rightarrow$ No word variations
- Ignore word importance $\rightarrow$ Useless but frequent words can boost BLEU score
- Ignore n-gram order $\rightarrow$ You can switch the n-grams and still get the same BLEU score
```

### ROUGE
- **Name**: Recall-Oriented Understudy for Gisting Evaluation.
- **What**: Recall-based N-gram overlap measure.


## NLG (Semantic)

### Fuzzy String Matching

## IR
### 