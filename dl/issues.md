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

## Covariate Shift
- **What**: A learned function suddenly sees its input data distribution change, while the conditional distribution of labels given data is presumed unchanged.
    - Dataset Covariate Shift: Training $\leftrightarrow$ Inference.
    - Internal Covariate Shift: Layer $\leftrightarrow$ Layer.
- **Why**:
    - Dataset: Difference between training & inference data.
    - Internal:
        1. The weights of earlier layer keep changing.
        2. $\rightarrow$ Later layers must re-adapt to each moving distribution
        3. $\rightarrow$ BUT the underlying mapping is unchanged
        4. $\rightarrow$ ⬇️Training stability
- **How to mitigate**:
	- BatchNorm: Re-center & Re-scale each layer's output channel to $N(0,1)$ $\rightarrow$ Steady distribution $\rightarrow$ Steady training
        - BUT less effective than LayerNorm $\leftarrow$ The normalization is independent between channels.
    - LayerNorm: Re-center & Re-scale each layer's outputs to $N(0,1)$ $\rightarrow$ Steady distribution $\rightarrow$ Steady training
	- Proper weight initialization (e.g., Xavier, He, etc.) $\rightarrow$ Stable gradients
	- Smaller learning rates $\rightarrow$ Reduce the magnitude of param updates
		- BUT slower convergence.
	- Adaptive learning rates (e.g., Adam) $\rightarrow$ dynamically control the magnitude of param updates.

## Overfitting
- **What**: The model performs well on the training data but badly on the test/unseen data.
- **Why**: Too high model complexity relative to training data size $\rightarrow$ Poor generalization.
- **How to mitigate**:
	- Parameter regularization (L1/L2) $\rightarrow$ reduce the magnitude of params.
    - Dropout $\rightarrow$ reduce dependency on certain neurons $\rightarrow$ promote learning of robust features.
    - Data augmentation $\rightarrow$ broaden training data distribution.
    - Early stopping $\rightarrow$ halt training before validation performance drops.
    - Cross validation $\rightarrow$ ensure consistent model performance across different data splits.
	- Reduce model complexity $\rightarrow$ a simpler model is less likely to capture too many details in the training data.

## Vanishing/Exploding Gradient
- **What**:
    - **Vanishing gradients**: The gradients become smaller and smaller during backprop. They have almost no effect to the front layers.
        - $\rightarrow$ Slows & potentially stops the learning process for the front layers.
    - **Exploding gradients**: The gradients become larger and larger during backprop. They change the weights of the front layers too much.
        - $\rightarrow$ Harder convergence & Training instability.
- **Why**:
    - Too many layers.
    - Too small/large gradient on certain layers (e.g., Sigmoid, Tanh, extremely large weight initialization, etc.) $\rightarrow$ propagates via chain rule.
- **How to mitigate**:
	- Proper activation functions (e.g., ReLU and variants) $\rightarrow$ Low likelihood of tiny gradients
    - Proper weight initialization (e.g., Xavier, He, etc.) $\rightarrow$ Maintain stable gradients
    - Batch Normalization $\rightarrow$ Maintain scaled gradients
    - Residual connection $\rightarrow$ Shortcut paths for gradient flow
    - Gradient clipping $\rightarrow$ Caps gradient magnitude