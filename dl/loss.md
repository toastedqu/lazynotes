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
# Loss
- **What**: The discrepancy between predicted and actual values as the optimization (i.e., minimization) objective.
- **Why**: **ML = Function Approximation**.
	- There is an underlying function which maps features to targets, but we don't know what it is.
	- $\rightarrow$ Function Approximation
	- $\rightarrow$ Many ways of doing it...
		1. **Discrepancy Minimization**
			- BUT how do we measure discrepancy w/o knowing the underlying function?
			- $\rightarrow$ Get samples
			- $\rightarrow$ Predict on samples
			- $\rightarrow$ Measure discrepancy over the samples
			- $\rightarrow$ LOSS!
		2. **Distribution Estimation**
			- BUT how do we directly estimate the underlying data distribution with no knowledge of it?
			- $\rightarrow$ Get samples
			- Distribution Estimation $\xrightarrow{become}$ Likelihood Estimation
			- $\rightarrow$ Tune our function to maximize the likelihood of observing these samples (i.e., **MLE**)
	- MLE & Loss Minimization are essentially doing the same thing but differently.
- **How**:
	1. For each sample, measure discrepancy between predicted and actual target value.
	2. Aggregate the discrepancies over all samples (i.e., **reduction**).
	- This page only uses **sum reduction**, for interpretability.

```{admonition} Q&A
:class: tip, dropdown
*Reduction: Sum vs Mean*
- Sum:
	- Direct objective
	- Preserves sample importance $\leftarrow$ Each sample contributes its full error.
	- Sensitive to batch size ($\propto$ batch size)
- Mean:
	- Default objective
	- Batch size invariance
		- $\rightarrow$ Optimization stability
		- $\rightarrow$ Awful performance when samples have uneven distributions across small batches
	- Insensitive to outliers $\rightarrow$ Reduces the impact of important but rare samples
```

<br/>

## Regression
### MSE
- **What**: Mean Squared Error.
- **Why**: Assumption: **Gaussian Distribution**.
- **How**: Get errors $\rightarrow$ Square errors $\rightarrow$ Aggregate

```{admonition} Math
:class: note, dropdown
Forward:

$$
\mathcal{L}=\sum_{i=1}^{m}(y_i-\hat{y}_i)^2
$$

Backward:

$$
\frac{\partial\mathcal{L}}{\partial\mathcal{\hat{y}_i}}=\frac{2}{m}(\hat{y}_i-y_i)
$$
```

```{admonition} Derivation
:class: important, dropdown
1. Relationship between true & predicted values:

	$$y_i=\hat{y}_i+\varepsilon_i$$

2. Gaussian Distribution:

	$$\varepsilon_i\sim N(0,\sigma^2)\Rightarrow y_i\sim N(\hat{y}_i,\sigma^2)$$

	- $y_i$ is a random variable ONLY because $\varepsilon_i$ is a random variable. $\hat{y}_i$ is deterministic.

3. MLE:

$$\begin{align*}
&p(y_i|\hat{y}_i)=\frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{(y_i-\hat{y}_i)}{2\sigma^2}\right) \\
&L(f)=\prod_{i=1}^mp(y_i|\hat{y}_i)\\
&\log L(f)=-\frac{m}{2}\log(2\pi\sigma^2)-\frac{1}{2\sigma^2}\sum_{i=1}^m(y_i-\hat{y}_i)^2 \\
&\arg\max_{\hat{y}_i}\log L(f)=\arg\max_{\hat{y}_i}\left(-\sum_{i=1}^m(y_i-\hat{y}_i)^2\right)=\arg\min_{\hat{y}_i}\sum_{i=1}^m(y_i-\hat{y}_i)^2
\end{align*}$$

4. MSE = MLE:

$$
\mathcal{L}=\sum_{i=1}^{m}(y_i-\hat{y}_i)^2
$$
```

```{admonition} Q&A
:class: tip, dropdown
*Pros?*
- Penalizes higher/lower errors more/less.
- Smooth $\rightarrow$ Differentiable
- Convex $\rightarrow$ Guaranteed global minimum.

*Cons?*
- Sensitive to outliers $\leftarrow$ Outliers take too much gradient
- Scale variant.
```

### MAE
- **What**: Mean Absolute Error.
- **Why**: Assumption: **Laplace Distribution**.
- **How**: Get errors $\rightarrow$ Absolute values $\rightarrow$ Aggregate

```{admonition} Math
:class: note, dropdown
Forward:

$$
\mathcal{L}=\sum_{i=1}^{m}|y_i-\hat{y}_i|
$$

Backward:

$$
\frac{\partial\mathcal{L}}{\partial\mathcal{\hat{y}_i}}=\frac{1}{m}\text{sign}(\hat{y}_i-y_i)
$$
```

```{admonition} Derivation
:class: important, dropdown
1. Relationship between true & predicted values:

	$$y_i=\hat{y}_i+\varepsilon_i$$

2. Laplace Distribution:

	$$\varepsilon_i\sim \text{Laplace}(0,b)\Rightarrow y_i\sim \text{Laplace}(\hat{y}_i,b)$$

	- $y_i$ is a random variable ONLY because $\varepsilon_i$ is a random variable. $\hat{y}_i$ is deterministic.

3. MLE:

$$\begin{align*}
&p(y_i|\hat{y}_i)=\frac{1}{2b}\exp\left(-\frac{|y_i-\hat{y}_i|}{b}\right) \\
&L(f)=\prod_{i=1}^mp(y_i|\hat{y}_i)\\
&\log L(f)=-m\log(2b)-\frac{1}{b}\sum_{i=1}^m|y_i-\hat{y}_i| \\
&\arg\max_{\hat{y}_i}\log L(f)=\arg\max_{\hat{y}_i}\left(-\sum_{i=1}^m|y_i-\hat{y}_i|\right)=\arg\min_{\hat{y}_i}\sum_{i=1}^m|y_i-\hat{y}_i|
\end{align*}$$

4. MAE = MLE:

$$
\mathcal{L}=\sum_{i=1}^{m}|y_i-\hat{y}_i|
$$
```

```{admonition} Q&A
:class: tip, dropdown
*Pros?*
- Robust to outliers $\leftarrow$ Equal gradient for all errors
- More interpretable (average error magnitude).
- Scale follows the original units.

*Cons?*
- Non-differentiable at zero $\rightarrow$ Requires subgradients
- Constant gradient $\rightarrow$ Slower learning
- No closed-form solution.
```

### Huber Loss


### Quantile Loss

<br/>

## Classification
### Cross Entropy
- **What**: Entropy of 2 probability distributions (predicted & actual) crossed over each other.
- **Why**: Assumption: **Categorical Dsitribution** (i.e., the probability of an observation belonging to each class $k$).
- **How**: Get cross entropy per sample $\rightarrow$ Aggregate

```{admonition} Math
:class: note, dropdown
Forward:

$$
\mathcal{L}=-\sum_{i=1}^{m}\sum_{k=1}^{K}y_{ik}\log\hat{p}_{ik}
$$

Backward:

$$\begin{align*}
\frac{\partial\mathcal{L}}{\partial\mathcal{\hat{p}_{ik}}}&=-\frac{y_{ik}}{\hat{p}_{ik}}\\
\frac{\partial\mathcal{L}}{\partial z_{ic}}&=\hat{p}_{ic}-y_{ic}
\end{align*}$$
```

```{admonition} Derivation
:class: important, dropdown
**Forward Formula**:
1. Categorical Distribution:

	$$p(y=k|p_1,\cdots,p_K)=p_k$$

2. One-hot representation:

	$$\mathbf{y}_i=[0;\cdots;1;\cdots;0]$$

	- $\text{index}(1)=k\Leftrightarrow y_{ik}=1$
	- $\text{index}(0)\neq k \Leftrightarrow y_{i,\neg k}=1$

3. Sample Likelihood:

	$$l(\hat{\mathbf{p}}_{i})=\prod_{k=1}^{K}\hat{p}_{ik}^{y_{ik}}$$

	- Assume target label is $k'$
	- $k=k'\rightarrow y_{ik}=1\rightarrow \hat{p}_{ik}^{y_{ik}}=\hat{p}_{ik}$
	- $k\neq k'\rightarrow y_{ik}=0\rightarrow \hat{p}_{ik}^{y_{ik}}=1$

4. Likelihood:

$$\begin{align*}
L(f)&=\prod_{i=1}^{m}\prod_{k=1}^{K}\hat{p}_{ik}^{y_{ik}} \\
\log L(f)&=\sum_{i=1}^{m}\sum_{k=1}^{K}y_{ik}\log\hat{p}_{ik}
\end{align*}$$

5. NLL:

$$
\mathcal{L}=-\sum_{i=1}^{m}\sum_{k=1}^{K}y_{ik}\log\hat{p}_{ik}
$$

**Gradient**:
1. Gradient over Probability Estimates:

$$
\frac{\partial\mathcal{L}}{\partial\mathcal{\hat{p}_{k}}}=-\frac{y_{k}}{\hat{p}_{k}}
$$

2. Logits $\xrightarrow{\text{softmax}}$ Probability Estimates:

$$
\hat{p}_{k}=\frac{e^{z_k}}{\sum_{c=1}^{K}e^{z_c}}
$$

3. Derivative of softmax:

$$
\frac{\partial\mathcal{\hat{p}_{k}}}{\partial\mathcal{z_c}}=\begin{cases}
\hat{p}_k(1-\hat{p}_k) & k=c \\
-\hat{p}_c\hat{p}_k & k\neq c
\end{cases}
$$

4. Chain:

$$\begin{align*}
\frac{\partial\mathcal{L}}{\partial\mathcal{z_{k}}}&=-\sum_{c=1}^{K}\frac{y_c}{\hat{p}_c}\frac{\partial\mathcal{\hat{p}_c}}{\partial\mathcal{z_k}} \\
&=-y_k(1-\hat{p}_k)+\sum_{c\neq k}y_c\hat{p}_k \\
&=-y_k(1-\hat{p}_k)+\hat{p}_k(1-y_k)\ \ \ \ \ \ (\Leftarrow\sum_{c=1}^{K}y_c=1)\\
&=\hat{p}_k-y_k
\end{align*}$$
```

```{admonition} Q&A
:class: tip, dropdown
*What is Entropy?*
- Degree of uncertainty in a probability distribution.
- ⬆️Entropy $\rightarrow$ More uncertain/balanced outcomes.
- ⬇️Entropy $\rightarrow$ More deterministic outcomes.

*Pros?*
- Penalizes higher/lower errors more/less.
- Smooth $\rightarrow$ Differentiable
- Convex $\rightarrow$ Guaranteed global minimum.

*Cons?*
- Sensitive to outliers $\leftarrow$ Outliers take too much gradient
- Scale variant.
```

### Hinge

<br/>

## Ranking
### Contrastive
- **What**: Create an embedding space where similar embeddings are closer & dissimilar embeddings are farther.
- **Why**: To learn **similarity**.
- **How**:
	1. Prepare positive & negative sample pairs.
	2. For positive/negative pairs, **penalize** wide/narrow distances.

```{admonition} Math
:class: note, dropdown
Notations:
- IO:
	- $X_1, X_2$: Embedding 1 & 2.
	- $Y$: Similarity indicator.
		- $Y=0$ $\leftarrow$ $X_1$ & $X_2$ are similar.
		- $Y=1$ $\leftarrow$ $X_1$ & $X_2$ are dissimilar.
- Hyperparams:
	- $M$: Margin, as minimal distance threshold for negative pairs.
		- If the distance is below the margin, penalize.
		- If the distance is above the margin, no penalty.
	- $D_W(X_1,X_2)$: Distance measure, parametrized by $W$.

Forward:

$$\begin{align*}
&\mathcal{L}_\text{pos}=\frac{1}{2} (D_W)^2 \\
&\mathcal{L}_\text{neg}=\frac{1}{2}[\max(0, M-D_W)]^2 \\
&\mathcal{L}=(1-Y)\mathcal{L}_\text{pos}+Y\mathcal{L}_\text{neg}
\end{align*}$$

Backward:

$$\begin{align*}
&\frac{\partial\mathcal{L}_\text{pos}}{\partial D_W}=D_W \\
&\frac{\partial\mathcal{L}_\text{neg}}{\partial D_W}=\begin{cases}
0 & D_W > M \\
-D_W & D_W < M
\end{cases}
\end{align*}$$
```

```{admonition} Q&A
:class: tip, dropdown
*Cons*:
- Difficult to find high-quality data.
- Pairwise relations only.
- ✅Hard measure (distance/similarity) & ❌Soft measure (probability).
- ✅Hyperparameter Tuning (for dissimilar samples)
	- $M$ too small $\rightarrow$ Cannot learn separation between dissimilar samples.
	- $M$ too large
		$\rightarrow$ Too difficult to minimize this loss
		$\rightarrow$ Ignore negative pair constraint
		$\rightarrow$ All embeddings become similar to best satisfy positive pair constraint
		$\rightarrow$ **Collapsing**
```


### Triplet

<br/>

## Generative Model
### Adversarial

### KL Divergence
- **What**: Relative entropy (i.e., Cross Entropy - Entropy).
- **Why**: To force the learned distribution to resemble the target distribution.
- **How**: 

```{admonition} Q&A
:class: tip, dropdown
*Why KL Divergence if we already have Cross Entropy?*:
1. Information Theory:
	- Entropy tells us the minimum possible average #bits needed to encode events drawn from $P$.
	- Cross Entropy tells us the average #bits needed to encode events drawn from $P$, if we are using an encoding scheme optimized for $Q$.
	- KL Divergence tells us the **extra bits** incurred by using $Q$-optimized encoding for events that actually follow $P$ $\rightarrow$ Quantifies **information lost / inefficiency** when approximating $P$ with $Q$.
2. Entropy may not always be constant:
	- If we use a fixed model $Q$ to approximate 2 different true distributions $P_1$ and $P_2$, then KL Divergence may not necessarily follow the pattern of cross entropy.
```

### Wasserstein

<!-- <br/> -->

<!-- ## Sequence Model
### CTC
- **What**: Connectionist Temporal Classification.

<br/>

# Segmentation
## Dice
## Jaccard (IoU)
## GDL
- **What**: Generalized Dice Loss.

<br/>

# Object Detection
## YOLO (Composite)
## Faster R-CNN (Composite)
## Focal (Dense Object Detection)
## SSD -->