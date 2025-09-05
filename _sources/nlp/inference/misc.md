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
# Misc
This page collects miscellaneous techniques in LLM inference.

## KV Cache
- **What**: Cache the $K$ & $V$ vectors for previous tokens.
- **Why**: They don't change during inference.
    1. ONLY the last hidden state is used to predict next token.
    2. What's the last hidden state?
        - Recall attention formula: $\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$.
        - $Q$: Query vector for LAST token.
        - $K$: Key vectors for ALL tokens.
        - $V$: Value vectors for ALL tokens.
    3. The causal masking in transformer decoders prevents later tokens from affecting earlier tokens.
    4. $\rightarrow$ At each autoregressive step, $K$ & $V$ of input tokens never change.
    5. $\xrightarrow{\text{cache}}$ Avoid recomputation.
- **How**: Cache.

```{admonition} Q&A
:class: tip, dropdown
*Pro Tip:* The query vectors for previous tokens are NEVER needed during inference.

*If it's so good, any cons?*
- ⬆️Memory cost: $O(m\cdot h\cdot d_k\cdot \#\mathrm{layers})$

*When should you turn it off?*
- Very short prompts: Recomputation is cheaper than caching.
- Parallelism & Hardware Acceleration: Caching adds an ALL-layer ALL-head copy for each step for each GPU.
```