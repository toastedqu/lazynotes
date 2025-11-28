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

A loss function is the discrepancy between predicted and actual values, as the optimization (minimization) objective.

Why? **ML = Function Approximation**.
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

How?
1. For each sample, measure discrepancy between predicted and actual target value.
2. Aggregate the discrepancies over all samples (i.e., **reduction**).

```{attention} Q&A
:class: dropdown
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

&nbsp;

## Regression
### MSE
- **What**: Mean Squared Error.
- **Why**: Assumption: **Gaussian Distribution**.
- **How**: Get errors $\rightarrow$ Square errors $\rightarrow$ Aggregate

```{note} Math
:class: dropdown
Forward:

$$
\mathcal{L}=\frac{1}{m}\sum_{i=1}^{m}(y_i-\hat{y}_i)^2
$$

Backward:

$$
\frac{\partial\mathcal{L}}{\partial\mathcal{\hat{y}_i}}=\frac{2}{m}(\hat{y}_i-y_i)
$$
```

```{tip} Derivation
:class: dropdown
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

```{attention} Q&A
:class: dropdown
*Pros?*
- Penalizes higher/lower errors more/less.
- Smooth $\rightarrow$ Differentiable
- Convex $\rightarrow$ Guaranteed global minimum.

*Cons?*
- Sensitive to outliers $\leftarrow$ Outliers take too much gradient
- Scale variant.
```

&nbsp;

### MAE
- **What**: Mean Absolute Error.
- **Why**: Assumption: **Laplace Distribution**.
- **How**: Get errors $\rightarrow$ Absolute values $\rightarrow$ Aggregate

```{note} Math
:class: dropdown
Forward:

$$
\mathcal{L}=\frac{1}{m}\sum_{i=1}^{m}|y_i-\hat{y}_i|
$$

Backward:

$$
\frac{\partial\mathcal{L}}{\partial\mathcal{\hat{y}_i}}=\frac{1}{m}\text{sign}(\hat{y}_i-y_i)
$$
```

```{tip} Derivation
:class: dropdown
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

```{attention} Q&A
:class: dropdown
*Pros?*
- Robust to outliers $\leftarrow$ Equal gradient for all errors
- More interpretable (average error magnitude).
- Scale follows the original units.

*Cons?*
- Non-differentiable at zero $\rightarrow$ Requires subgradients
- Constant gradient $\rightarrow$ Slower learning
- No closed-form solution.
```

&nbsp;

### RMSE
- **What**: Root Mean Squared Error.
- **Why**: MSE in original units.
- **How**: Get errors $\rightarrow$ Square errors $\rightarrow$ Mean $\rightarrow$ Square root

```{note} Math
:class: dropdown
Forward:

$$
\mathcal{L}=\sqrt{\frac{1}{m}\sum_{i=1}^{m}(y_i-\hat{y}_i)^2}
$$

Backward:

$$
\frac{\partial \mathcal{L}}{\partial \hat{y}_i}= \frac{1}{m\,\mathcal{L}}(\hat{y}_i-y_i)
$$
```

```{attention} Q&A
:class: dropdown
*Pros?*
- Same ordering as MSE (monotonic transform) but in original units.
- Penalizes higher/lower errors more/less (like MSE).
- Interpretable as “typical” error magnitude in target units.

*Cons?*
- Still sensitive to outliers (inherits MSE behavior).
- Gradient depends on loss value $\rightarrow$ Unstable as $\mathcal{L} \rightarrow 0$
- Inconvenient than MSE.
```

&nbsp;

### Huber Loss
- **What**: Piecewise loss: **MSE** for small errors, **MAE** for large errors.
- **Why**: Balance sensitivity + robustness (quadratic near 0, linear for outliers).
- **How**: Get errors $\rightarrow$ Apply piecewise function with threshold $\delta$ $\rightarrow$ Aggregate

```{note} Math
:class: dropdown
Forward: 
- Let $e_i = y_i-\hat{y}_i$. Then

$$
\mathcal{L}_\delta(e_i)=
\begin{cases}
\frac{1}{2}e_i^2, & |e_i|\le \delta \\
\delta\left(|e_i|-\frac{1}{2}\delta\right), & |e_i|>\delta
\end{cases}
$$

- Aggregate:

$$
\mathcal{L}=\frac{1}{m}\sum_{i=1}^{m}\mathcal{L}_\delta(y_i-\hat{y}_i)
$$

Backward:

$$
\frac{\partial \mathcal{L}}{\partial \hat{y}_i}
=
\begin{cases}
\frac{1}{m}(\hat{y}_i-y_i), & |y_i-\hat{y}_i|\le \delta \\
\frac{1}{m}\delta\ \text{sign}(\hat{y}_i-y_i), & |y_i-\hat{y}_i|>\delta
\end{cases}
$$
````

```{tip} Derivation
:class: dropdown
1. Define error:

$$
e_i=y_i-\hat{y}_i
$$

2. Want a loss that behaves like:
	- MSE near 0: $\frac{1}{2}e_i^2$ (smooth, strong pull to correct small errors)
	- MAE for large errors: linear growth to reduce outlier influence

3. Build a piecewise function that is continuous and differentiable at $|e_i|=\delta$:
	- For $|e_i|\le \delta$ use $\frac{1}{2}e_i^2$.
	- For $|e_i|>\delta$ use a line with slope $\delta$ matching value at $\delta$:

$$
\delta\left(|e_i|-\frac{1}{2}\delta\right)
$$

4. Differentiate (chain rule with $e_i=y_i-\hat{y}_i$):

$$
\frac{\partial \mathcal{L}_\delta}{\partial \hat{y}_i}
=
\begin{cases}
-e_i, & |e_i|\le\delta \\
-\delta\,\text{sign}(e_i), & |e_i|>\delta
\end{cases}
\Rightarrow
\frac{\partial \mathcal{L}}{\partial \hat{y}_i}=
\begin{cases}
\frac{1}{m}(\hat{y}_i-y_i), & |e_i|\le \delta \\
\frac{1}{m}\delta\ \text{sign}(\hat{y}_i-y_i), & |e_i|>\delta
\end{cases}
$$
```

```{attention} Q&A
:class: dropdown
*Pros?*
- Robust to outliers (linear tail like MAE).
- Smooth around zero (unlike MAE) $\rightarrow$ stable gradients.
- Interpolates between MSE and MAE via $\delta$.

*Cons?*
- Need to tune $\delta$ (problem-dependent, scale-dependent).
	- If $\delta$ too large $\rightarrow$ behaves like MSE (outlier sensitive).
	- If $\delta$ too small $\rightarrow$ behaves like MAE (slower learning).
```

&nbsp;

## Classification
### Cross Entropy
- **What**: Entropy of 2 probability distributions (predicted & actual) crossed over each other.
- **Why**: Assumption: **Categorical Dsitribution** (i.e., the probability of an observation belonging to each class $k$).
- **How**: Get cross entropy per sample $\rightarrow$ Aggregate

```{note} Math
:class: dropdown
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

```{tip} Derivation
:class: dropdown
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

```{attention} Q&A
:class: dropdown
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

&nbsp;

## Ranking
### Contrastive
- **What**: Create an embedding space where similar embeddings are closer & dissimilar embeddings are farther.
- **Why**: To learn **similarity**.
- **How**:
	1. Prepare positive & negative sample pairs.
	2. For positive/negative pairs, **penalize** wide/narrow distances.

```{note} Math
:class: dropdown
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

```{attention} Q&A
:class: dropdown
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

&nbsp;

## Generative Model
### KL Divergence
- **What**: Relative entropy (i.e., Cross Entropy - Entropy).
- **Why**: Learned distribution $\xrightarrow{\text{resemble}}$ Target distribution
- **How**: Compute & Minimize divergence.


`````{note} Math
:class: dropdown
````{tab-set}
```{tab} Distribution
Notations:
- Distributions:
	- $P$: Target / true distribution.
	- $Q$: Learned / model distribution.
- Events:
	- $x$: An outcome/event in the sample space.
- Information measures:
	- $H(P)=-\sum_x P(x)\log P(x)$: Entropy of $P$.
	- $H(P,Q)=-\sum_x P(x)\log Q(x)$: Cross entropy of $P$ relative to $Q$.

Forward (discrete):

$$
D_{\text{KL}}(P\|Q)=H(P,Q)-H(P)=\sum_x P(x)\log\frac{P(x)}{Q(x)}
$$
```

```{tab} Dataset
Notations:
- IO:
	- $y_{ik}$: Target probability for sample $i$ and class $k$ (often one-hot).
	- $\hat{p}_{ik}=Q(k|x_i)$: Predicted probability for sample $i$ and class $k$.
	- $z_{ik}$: Logit for sample $i$, class $k$ (before softmax).

Forward:
- Normal:

$$
\mathcal{L}=\frac{1}{m}\sum_{i=1}^{m}\sum_{k=1}^{K} y_{ik}\log\frac{y_{ik}}{\hat{p}_{ik}}
$$

- One-hot ($y_{ik}\in\{0,1\}$):

$$
\mathcal{L}=\frac{1}{m}\sum_{i=1}^{m}\left(-\sum_{k=1}^{K}y_{ik}\log \hat{p}_{ik}\right)
$$

Backward:
- w.r.t. predicted probabilities:

$$
\frac{\partial \mathcal{L}}{\partial \hat{p}_{ik}}=-\frac{1}{m}\frac{y_{ik}}{\hat{p}_{ik}}
$$

- w.r.t softmax logits:

$$
\frac{\partial \mathcal{L}}{\partial z_{ik}}=\frac{1}{m}\left(\hat{p}_{ik}-y_{ik}\right)
$$
```
````
`````

```{tip} Derivation
:class: dropdown
1. Start from definition (discrete case):

$$
D_{\text{KL}}(P\|Q)=\sum_x P(x)\log\frac{P(x)}{Q(x)}
$$

2. Split the log ratio:

$$
\sum_x P(x)\left(\log P(x)-\log Q(x)\right)
= \sum_x P(x)\log P(x) - \sum_x P(x)\log Q(x)
$$

3. Recognize entropy and cross entropy:

$$
-\sum_x P(x)\log P(x)=H(P),\qquad
-\sum_x P(x)\log Q(x)=H(P,Q)
$$

So

$$
D_{\text{KL}}(P\|Q)=H(P,Q)-H(P)
$$

4. Why minimizing KL often equals minimizing cross entropy:
- If $P$ is fixed (the data-generating / target distribution), then $H(P)$ does not depend on $Q$.
- Therefore,

$$
\arg\min_Q D_{\text{KL}}(P\|Q)=\arg\min_Q H(P,Q)
$$

5. 
```

```{attention} Q&A
:class: dropdown
*Why minimizing KL often equals minimizing cross entropy?*
- If $P$ is fixed (the data-generating / target distribution), then $H(P)$ does not depend on $Q$.
- Therefore,

$$
\arg\min_Q D_{\text{KL}}(P\|Q)=\arg\min_Q H(P,Q)
$$

- From a dataset's viewpoint, let $P$ be the empirical distribution over labels given inputs. Then minimizing KL pushes $Q(\cdot|x)$ to match $P(\cdot|x)$.
- For one-hot labels, $H(P)$ becomes constant (often 0 per sample), so KL reduces to the standard cross-entropy objective.

*Why KL Divergence if we already have Cross Entropy?*
1. Information Theory:
	- Entropy tells us the minimum possible average #bits needed to encode events drawn from $P$.
	- Cross Entropy tells us the average #bits needed to encode events drawn from $P$, if we are using an encoding scheme optimized for $Q$.
	- KL Divergence tells us the **extra bits** incurred by using $Q$-optimized encoding for events that actually follow $P$ $\rightarrow$ Quantifies **information lost / inefficiency** when approximating $P$ with $Q$.
2. Entropy may not always be constant:
	- If we use a fixed model $Q$ to approximate 2 different true distributions $P_1$ and $P_2$, then KL Divergence may not necessarily follow the pattern of cross entropy.
```

<!-- ### Adversarial -->
<!-- ### Wasserstein -->

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