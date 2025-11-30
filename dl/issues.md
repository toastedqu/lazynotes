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
# Issues
This page lists common issues in DL in Alphabetic order.

## Batch-size Pathologies
- **What**: Training behavior depends on batch size.
    - Small: noisy grads, unstable optimization.
    - Large: sharp-minima / generalization drop, fragile training dynamics.
- **Why**:
    1. Gradient noise scale changes with batch size.
        - Mini-batch grad = true grad + noise.
        - ⬆️ Batch size → ⬇️ noise → optimization becomes more "deterministic".
        - BUT what if the noise is beneficial?
        - It helps exploration and can bias toward flatter minima → ⬆️ Generalization.
    2. Batch size interacts with LR & Update Frequency
        - For fixed epochs: ⬆️ Batch size → ⬇️ Param updates per epoch.
        - If fixed LR/schedule, we can under-train (not enough steps) or destabilize (LR too high for the new regime).
    3. Normalization & regularization may depend on batch size.
        - ⬇️ Batch size → BatchNorm stats become noisy → ⬇️ Training stability & perf.
        - Some regularizers (e.g., weight decay) behave differently when we scale LR with batch size.
    4. "Critical batch size" phenomenon.
        - Up to a task-dependent threshold, larger batch improves throughput and keeps convergence similar.
        - Beyond that, returns diminish → More batch doesn’t reduce wall-clock meaningfully and may hurt generalization.
        - Large-batch training can match train loss but lose test accuracy.
- **How to mitigate**:
    - Retune LR + schedule for the new batch size.
    - Stabilize normalization via LayerNorm/GroupNorm/RMSNorm.
    - Preserve "useful noise" with large batch size.
        - Add stochasticity: stronger data augmentation, dropout, stochastic depth, etc.
    - Grad Accumulation.

## Covariate Shift
- **What**: A learned function suddenly sees its input data distribution change, while the conditional distribution of labels given data is presumed unchanged.
    - Dataset Covariate Shift: Training $\leftrightarrow$ Inference.
    - Internal Covariate Shift: Layer $\leftrightarrow$ Layer.
- **Why**:
    - Dataset: Difference between training & inference data.
    - Internal:
        1. The weights of earlier layer keep changing.
        2. → Later layers must re-adapt to each moving distribution
        3. → BUT the underlying mapping is unchanged
        4. → ⬇️Training stability
- **How to mitigate**:
	- BatchNorm: Re-center & Re-scale each layer's output channel to $N(0,1)$ → Steady distribution → Steady training
        - BUT less effective than LayerNorm ← The normalization is independent between channels.
    - LayerNorm: Re-center & Re-scale each layer's outputs to $N(0,1)$ → Steady distribution → Steady training
	- Proper weight initialization (e.g., Xavier, He, etc.) → Stable gradients
	- Smaller learning rates → Reduce the magnitude of param updates
		- BUT slower convergence.
	- Adaptive learning rates (e.g., Adam) → dynamically control the magnitude of param updates.

## Overfitting
- **What**: The model performs well on the training data but badly on the test/unseen data.
- **Why**: Too high model complexity relative to training data size → Poor generalization.
- **How to mitigate**:
	- Parameter regularization (L1/L2) → reduce the magnitude of params.
    - Dropout → reduce dependency on certain neurons → promote learning of robust features.
    - Data augmentation → broaden training data distribution.
    - Early stopping → halt training before validation performance drops.
    - Cross validation → ensure consistent model performance across different data splits.
	- Reduce model complexity → a simpler model is less likely to capture too many details in the training data.

## Vanishing/Exploding Gradient
- **What**:
    - **Vanishing gradients**: The gradients become smaller and smaller during backprop. They have almost no effect to the front layers.
        - → Slows & potentially stops the learning process for the front layers.
    - **Exploding gradients**: The gradients become larger and larger during backprop. They change the weights of the front layers too much.
        - → Harder convergence & Training instability.
- **Why**:
    - Too many layers.
    - Too small/large gradient on certain layers (e.g., Sigmoid, Tanh, extremely large weight initialization, etc.) → propagates via chain rule.
- **How to mitigate**:
	- Proper activation functions (e.g., ReLU and variants) → Low likelihood of tiny gradients
    - Proper weight initialization (e.g., Xavier, He, etc.) → Maintain stable gradients
    - Batch Normalization → Maintain scaled gradients
    - Residual connection → Shortcut paths for gradient flow
    - Gradient clipping → Caps gradient magnitude