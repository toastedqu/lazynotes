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
This page lists the common issues in deep learning in Alphabetic order.

# Internal Covariate Shift
- **What**: During training, the distribution of inputs to each layer changes. Each layer has to continuously adapt to a new distribution.
- **Cause**: During training, the params of all preceding layers change, causing changes in their outputs.
- **Consequence**: Harder convergence & Training instability.
- **Mitigation**:
	- Batch normalization $\rightarrow$ stabilize input distribution.
	- Proper weight initialization (e.g., Xavier, He, etc.) $\rightarrow$ maintain stable gradients.
	- Smaller learning rates $\rightarrow$ reduce the magnitude of param updates.
		- Cons: slower convergence.
	- Adaptive learning rates (e.g., Adam) $\rightarrow$ dynamically control the magnitude of param updates.

# Overfitting
- **What**: The model performs well on the training data but badly on the test/unseen data.
- **Cause**: Too high model complexity relative to training data size.
- **Consequence**: Poor generalization.
- **How to mitigate**:
	- Parameter regularization (L1/L2) $\rightarrow$ reduce the magnitude of params.
    - Dropout $\rightarrow$ reduce dependency on certain neurons $\rightarrow$ promote learning of robust features.
    - Data augmentation $\rightarrow$ broaden training data distribution.
    - Early stopping $\rightarrow$ halt training before validation performance drops.
    - Cross validation $\rightarrow$ ensure consistent model performance across different data splits.
	- Reduce model complexity $\rightarrow$ a simpler model is less likely to capture too many details in the training data.

# Vanishing/Exploding Gradient
- **What**:
    - **Vanishing gradients**: The gradients become smaller and smaller during backprop. They have almost no effect to the front layers.
    - **Exploding gradients**: The gradients become larger and larger during backprop. They change the weights of the front layers too much.
- **Cause**:
    - Too many layers.
    - Too small/large gradient on certain layers (e.g., Sigmoid, Tanh, extremely large weight initialization, etc.) $\rightarrow$ propagates via chain rule.
- **Consequence**:
	- **Vanishing gradients**: Slows & potentially stops the learning process for the front layers.
	- **Exploding gradients**: Harder convergence & Training instability.
- **How to mitigate**:
	- Proper activation functions (e.g., ReLU and variants) $\rightarrow$ low likelihood of tiny gradients.
    - Proper weight initialization (e.g., Xavier, He, etc.) $\rightarrow$ maintain stable gradients.
    - Batch Normalization $\rightarrow$ 
    - Residual connection.
    - Gradient clipping.