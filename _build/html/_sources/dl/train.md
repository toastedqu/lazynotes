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
# Training
## Weight Initialization
In any neural network, all parameters need to be initialized to be updated during training.

The choice of initialization methods matters.
- If the initial weights are too large / too small, exploding / vanishing gradients will take place.
- If the weights are initialized improperly, convergence can be slow, nonexistent, or stuck in suboptimal solutions.

### Zero Initialization
- **What**: All parameters are 0.
- **Why**: Back in the days, people thought it was simple and unbiased due to a lack of understanding of deep learning.
- **Pros**: Simple, unbiased, good performance with small models.
- **Cons**:
    - **Failure to break symmetry**: All weights produce the same output and receive the same gradients $\rightarrow$ No learning.
	- Vanishing gradients.
	- No non-linearity.
	- Dead neurons if activated by the ReLU family.

### Random Initialization
### He Initialization
### Xavier/Glorot Initialization

## Gradient Clipping

## Early Stopping

## Checkpointing

## Weight Regularization

## Hyperparameter Tuning
### Grid Search
### Random Search
### Bayesian Optimization
### Hyperband

## Transfer Learning
### Pre-training
### Fine-tuning

## Distributed Training
### Data Parallelism
### Model Parallelism
### Distributed Training Frameworks

## Mixed Precision Training

## Model Compression
### Pruning
### Quantization
### Knowledge Distillation