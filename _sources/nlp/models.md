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

This page does NOT cover contemporary LLM for now.

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
        1. Collect corpora. Split corpora into sentence pairs.
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

<!-- ### DeBERTa (Decoding-enhanced BERT with disentangled attention)
- **What**: BERT -->

## Decoder-only
### GPT (Generative Pretrained Transformer)
- **What**: Decoder block stack.
    - ONLY masked attention.
    - Learned PE instead of Sinusoidal PE.
- **Why**: For arbitrary-length text generation with unsupervised pretraining.
- **How (Pretraining)**: 
    1. Data Processing:
        1. Collect corpora.
        2. Tokenize each sentence with [BPE](tokenizer.md#bpe-byte-pair-encoding).
        3. Form input-target pairs $\leftarrow$ Shift each token sequence right by 1 token.
    2. Pretraining:
        1. Compute & Add Token embeddings and Position embeddings.
            - **Token embeddings**: Trainable "token ID $\rightarrow$ embedding" look-up table/matrix.
            - **Positional embeddings**: Trainable "position index $\rightarrow embedding$" look-up table/matrix.
        2. GPT forward pass.
        3. Next Token Prediction: Compute Cross-Entropy loss between predicted and true next tokens.
        4. Backprop & Update.

```{admonition} Math
:class: note, dropdown
Notations:
- IO:
    - $t_i$: The true token at position $i$.
- Hyperparams:
    - $m$: #Tokens.
- Misc:
	- $\hat{p}_i(t_i)$: Probability of the true token $t_i$ being generated at position $i$.

Loss:

$$
\mathcal{L}=-\frac{1}{m}\sum_{i=1}^{m}\log \hat{p}_i(t_i)
$$
```

### GPT-2
- **What**: GPT but larger & better, with **zero-shot** capability.
- **Why**: OpenAI bet on the scaling law.
- **How (improvement)**: 
    - Larger & Better:
        - Model size: 117M $\rightarrow$ 1.5B
        - Training data: 8x
        - #Blocks (i.e., Depth): 12 $\rightarrow$ 48
        - Hidden dim (i.e., Width): 768 $\rightarrow$ 1600
        - Context window: 512 $\rightarrow$ 1024
        - Vocab size: 40478 $\rightarrow$ 50257
        - Batch size: 64 $\rightarrow$ 512
    - [Nucleus sampling](inference.md#top-p-nucleus).
    - Zero-shot prompting.

### GPT-3
- **What**: GPT-2 but larger & better, with **few-shot** capability.
- **Why**: OpenAI bet on the scaling law.
- **How (improvement)**:
    - Larger & Better:
        - Model size: 1.5B $\rightarrow$ 175B
        - Training data: 15x
        - #Blocks (i.e., Depth): 48 $\rightarrow$ 96
        - Hidden dim (i.e., Width): 1600 $\rightarrow$ 12288
        - Context window: 1024 $\rightarrow$ 2048
    - Few-shot prompting.

### GPT-3.5 (ChatGPT / Better InstructGPT)
- **What**: Enhanced GPT-3 as a dialogue system.
- **Why**: OpenAI bet on niche methods - instruction-following, multi-step reasoning, and RLHF.
- **How (improvement)**:
    - Larger & Better:
        - Training data: Basically all texts on the internet till 2022.
        - Context window: 2048 $\rightarrow$ 4096.
    - Instruction tuning: Finetune the model on specific types of system and user prompts to mimic dialogue system behavior.
    - Multi-step reasoning.
    - [RLHF](rlhf.md).

### GPT-4
- **What**: ChatGPT but can read image inputs & better in NLP tasks.
- **Why**: OpenAI bet on multi-modality.
- **How**: They became ClosedAI. We don't know anymore.

## References
- [BERT](https://arxiv.org/pdf/1810.04805)
- [RoBERTa](https://arxiv.org/pdf/1907.11692)
- [GPT](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [GPT-3](https://arxiv.org/pdf/2005.14165)
- [GPT-3.5](https://arxiv.org/pdf/2203.02155)
- [GPT-4](https://arxiv.org/pdf/2303.08774)

<!-- ## Encoder-Decoder
### T5 -->