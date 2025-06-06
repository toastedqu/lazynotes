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
This page covers the prevalent Encoder-only, Decoder-only, and Encoder-decoder transformer models.

This page does NOT cover different LLMs.

## Encoder-only
### BERT (Bidirectional Encoder Representations from Transformers)
- **What**: Bidirectional, Encoder-only Transformer model pretrained on MLM & NSP objectives.
    - **MLM (Masked Language Modeling)**: Learn to fill in the blank.
    - **NSP (Next Sentence Prediction)**: Learn to predict whether Sentence B is the next sentence of Sentence A.
- **Why**: To enable truly bidirectional context understanding.
    - A transformer model with decoder relies on causal masking for token prediction $\rightarrow$ Unidirectional understanding.
    - BERT replaces causal masking with random masking to force model to predict missing words.
- **How (Pretraining)**:
    1. Data Processing:
        1. Collect corporra. Split corpora into sentence pairs.
        2. Tokenize each sentence with [WordPiece](tokenizer.md#wordpiece).
        3. Add special tokens.
            - "[CLS]": Classification token. Added at the beginning.
            - "[SEP]": Separation token. Added between & after sentences.
        4. For each sentence pair,
            - 50% chance keep it unchanged $\rightarrow$ Positive
            - 50% chance replace Sentence B with a random sentence $\rightarrow$ Negative.
        5. Randomly select 15% of all input tokens for MLM:
            - 80% of them get replaced with "[MASK]".
            - 10% of them get replaced with a random token.
            - 10% of them are unchanged.
    2. Pretraining:
        1. Compute & Add Token embeddings, Segment embeddings, and Position embeddings.
            - **Token embeddings**: Trainable "token ID $\rightarrow$ embedding" look-up table/matrix.
            - **Segment embeddings**: Trainable "segment ID $\rightarrow$ embedding" look-up table/matrix, to distinguish the two sentences.
            - **Positional embeddings**: Trainable "position index $\rightarrow embedding$" look-up table/matrix.
        2. BERT forward pass.
        3. **MLM**: For each masked position, use Linear + Softmax to convert BERT output to token probability distribution.
            - Linear: Weights = Transpose of Token embeddings matrix.
        4. **NSP**: For the [CLS] position, use Linear + Softmax to convert BERT output to label probability distribution.
            - Linear: Shape transform from hidden dim to two logits.
        5. Compute & Add Cross-Entropy losses from MLM & NSP. Backprop & Update.

```{admonition} Math
:class: note, dropdown
Notations:
- IO:
    - $t_i$: The true token at masked index $i$.
    - $y$: 1 if Sentence B truly follows Sentence A, else 0.
- Hyperparams:
    - $\mathcal{M}$: Set of masked indices.
    - $V$: Vocab size.
- Misc:
	- $\hat{p}_i(t_i)$: Probability of the true token $t_i$ at masked index $i$ being selected by the model.
    - $\hat{q}$: Probability of Sentence B being predicted as the next sentence of Sentence A.


Loss:

$$\begin{align*}
&\text{Total}: &&\mathcal{L}=\mathcal{L}_{\text{MLM}}+\mathcal{L}_{\text{NSP}} \\
&\text{MLM}:   &&\mathcal{L}_{\text{MLM}}=-\frac{1}{|\mathcal{M}|}\sum_{i\in\mathcal{M}}\log \hat{p}_i(t_i) \\
&\text{NSP}:   &&\mathcal{L}_{\text{NSP}}=-\left[y\log\hat{q}+(1-y)\log(1-\hat{q})\right]
\end{align*}$$
```

```{admonition} Q&A
:class: tip, dropdown
*Why select 15% tokens for MLM? Why not more?*
- We want to provide enough context for BERT to learn rich representations $\rightarrow$ Masking too many = Insufficient context
- We also want to create a sufficiently challenging classification task $\rightarrow$ Masking too few = Insufficient learning signal
- 15% = A perfect balance.

*Why the 80-10-10 separation? Why not just mask all of them as [MASK]?*
- 80% [MASK]: Make BERT learn to predict missing words using surrounding context.
- 10% random token: Prevent BERT from overfitting to the [MASK] token as a missing word $\rightarrow$ This is STILL treated as a prediction target.
- 10% unchanged token: Reduce discrepancy between pretraining & finetuning/inference.
    - Pretraining: [MASK] tokens exist.
    - Finetuning/Inference: [MASK] tokens do not exist.
```

### RoBERTa (Robustly Optimized BERT Pretraining Approach)
- **What**: BERT but more robust.
    
    
    - Different training configs.
- **Why**: To improve BERT's robustness.
    - **Robustness**: The ability to maintain reliable performance with noisy inputs.
- **How**:
    - Pretrained on much larger corpora (10x).
    - Use **BPE**.
    - **No NSP**. ONLY MLM.
    - **Dynamic Masking**: Create a new masking pattern every time a sequence is fed to RoBERTa $\rightarrow$ Empirically better than static masking.
    - Training configs:
        - Batch size ⬆️⬆️.
        - #Training steps ⬆️.
        - Max LR ⬆️.
        - Warmup length ⬆️.
        - **Sequence length scheduling**: Alternate between short (128 tokens) & full-length (512 tokens) sequences in a single training run.
            - BERT sticks to 512 tokens.

```{admonition} Q&A
:class: tip, dropdown
*Why no NSP?*
- Empirically, NSP provides negligible benefits.
- So why waste computational resources on it?

*Why dynamic masking?*
- To prevent overfitting to a fixed mask pattern.

*Why the training config changes?*
- Batch size: More stable gradient estimates & Better computational resources.
- #Training steps: Larger corpora requires more data iterations.
- Max LR: Accelerate convergence when paired with larger batches.
- Warmup length: Prevent sudden large updates at the start of training, to accompany larger batch size and larger max LR.

*Why BPE instead of WordPiece?*
- BPE can break ANY unseen character down into a sequence of bytes, all of which are in its vocab.
- WordPiece resorts to "[UNK]" when facing a new Unicode character.
```

### ALBERT (A Lite BERT)
- **What**: BERT but significantly lower memory

## References
- [BERT](https://arxiv.org/pdf/1810.04805)
- [RoBERTa](https://arxiv.org/pdf/1907.11692)