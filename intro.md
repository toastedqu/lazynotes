# LazyNotes
A minimalist AI handbook, focusing on intuition and theory.

Each concept is structured as follows:
- Mandatory:
    - **What**: Intuition.
    - **Why**: Problem/Motivation.
    - **How**: Functionality.
- Optional:
    - Math
    - Derivation
    - Q&A
    - ...

Common notations include (section-specific notations take priority):
- $ a $: scalar / variable
- $ \textbf{a} $: vector
- $ A $: matrix / random variable / upper bound
- $ \mathbf{A} $: tensor (dim > 2)
- $ \mathcal{A} $: set / special notation
- $ \mathbb{A} $: common number set
<!--  -->
- $ [] $: vector brackets
- $ \{\} $: set brackets
- $ || $: norm  (for continuous vectors) / count (for discrete vectors)
- $ \# $: count
- $ * $: arbitrary values
- $ \hat{\ \ } $: estimator
- $ a_\text{sub} $: index (subscript)
- $ a^\text{sup} $: name (superscript)
<!--  -->
- $ m $: #samples per input batch
- $ n $: #features per sample
- $ c $: #classes in dataset
- $ i $: sample index
- $ j $: feature index
- $ k $: class index
<!--  -->
- $ \mathcal{D} $: dataset
- $ X=[\mathbf{x}_1,\cdots,\mathbf{x}_m]^T$: input matrix of shape $(m,n)$ (append $\textbf{1}$ if bias is needed)
- $ \mathbf{y}=[y_1,\cdots,y_{m}]^T$: output vector of shape $(m,1)$
- $ \textbf{w}=[w_1,\cdots,w_n]$: params (append $b$ if bias is needed)