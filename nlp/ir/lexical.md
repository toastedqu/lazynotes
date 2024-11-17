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
# Lexical Search
Lexical Search (or Keyword Search) finds information by directly matching exact words or phrases from a search query.

## TF-IDF
- **Name**: **[Term Frequency-Inverse Document Frequency](https://doi.org/10.1007/978-0-387-30164-8_832)**
- **What**: A measure of lexical similarity that combines word frequency within a document with word rarity across the entire corpus.
- **Why**:
	- Words common in many documents are less informative and don't distinguish documents well.
	- Words rare within a specific document also don't aid in distinguishing content effectively.
- **How**:
	1. For each term in the input query,
		1. Calculate TF for each document.
		2. Calculate IDF across the corpus.
		3. Multiple TF and IDF for each document.
	2. Aggregate scores across all query terms for each document.
	3. Rank documents by aggregated scores.
- **Conditions**:
	- High TF-IDF scores for terms common in a document but rare in the corpus represent lexical similarity.
	- Terms are independent.
- **Pros**:
	- Simple.
	- High interpretability -> highlights representative terms for each document.
- **Cons**:
	- Ignores word order and syntax.
	- Lacks semantic understanding (e.g., synonyms).
	- Lacks lexical understanding in diverse or fine-grained corpora.

```{admonition} Math
:class: note, dropdown
Notations:
- IO:
 	- $t$: A term in a query.
	- $d$: A document.
- Hyperparams:
	- $f_{t,d}$: Count of term $t$ in document $d$.
	- $N$: $\#$Docs in corpus.
	- $n_t$: $\#$Docs containing term $t$.
	- $K$: Double normalization weight. $K=0.5$ in common cases.

Formula:

$$\begin{align}
&\text{TF-IDF}_{t,d}=\text{TF}_{t,d}\times\text{IDF}_t \\
&\text{TF}_{t,d}=\frac{f_{t,d}}{\max_{t'\in d}f_{t',d}} \\
&\text{IDF}_{t}=\log\left(\frac{N}{n_t+1}\right)
\end{align}$$

TF variations:

$$\begin{matrix*}[l]
\text{Raw count} & \text{TF}_{t,d}=f_{t,d} \\
\text{Max normalized} & \text{TF}_{t,d}=\frac{f_{t,d}}{\max_{t'\in d}f_{t',d}} \\
\text{Sum normalized} & \text{TF}_{t,d}=\frac{f_{t,d}}{\sum_{t'\in d}f_{t',d}} \\
\text{Log normalized} & \text{TF}_{t,d}=\log(1+f_{t,d}) \\
\text{Double normalized} & \text{TF}_{t,d}=K+(1-K)\times\frac{f_{t,d}}{\max_{t'\in d}f_{t',d}} \\
\end{matrix*}$$

IDF variations:

$$\begin{matrix*}[l]
\text{Raw calc} & \text{IDF}_t=\log\left(\frac{N}{n_t}\right) \\
\text{Smooth} & \text{IDF}_t=\log\left(\frac{N}{n_t+1}\right) \\
\text{Ultra smooth} & \text{IDF}_t=\log\left(\frac{N}{n_t+1}\right)+1 \\
\text{Max} & \text{IDF}_t=\log\left(\frac{\max_{t'\in d}n_{t'}}{n_t+1}\right) \\
\text{Probabilistic} & \text{IDF}_t=\log\left(\frac{N-n_t}{n_t}\right)
\end{matrix*}$$
```