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
# Data Geometry
Real data lives near low-dimensional manifolds: how to measure their dimension & recover their geometry (nonlinear dimensionality reduction).
[PCA](../../ml/unsupervised.md#pca), [t-SNE](../../ml/unsupervised.md#t-sne), [UMAP](../../ml/unsupervised.md#umap) & [Spectral Clustering](../../ml/unsupervised.md#spectral-clustering) → `ml/unsupervised.md`. [Graph Laplacian](../linalg.md#graph-laplacian) → `math/linalg.md`.

Notations:
- $X=[\mathbf{x}_1,\cdots,\mathbf{x}_m]^T\in\mathbb{R}^{m\times n}$: Data matrix (rows = points)
- $\mathbf{x}_i\in\mathbb{R}^n$: $i$-th data point
- $\mathcal{M}\subset\mathbb{R}^n$: Data manifold
- $d$: Intrinsic dim (ID sections) / chosen output dim (manifold-learning sections). Need not match: circle → ID 1, needs 2 coords.
- $\mathbf{y}_i\in\mathbb{R}^d$: Embedding of $\mathbf{x}_i$
- $Y=[\mathbf{y}_1,\cdots,\mathbf{y}_m]^T\in\mathbb{R}^{m\times d}$: Embedding matrix
- $k$: #nearest neighbors
- $\mathcal{N}(i)$: Index set of the $k$ nearest neighbors of $\mathbf{x}_i$
- $r_{i,j}$: Euclidean distance from $\mathbf{x}_i$ to its $j$-th nearest neighbor
- $W\in\mathbb{R}^{m\times m}$: Weight matrix of the neighbor graph
- $D=\mathrm{diag}(W\mathbf{1})$: Degree matrix
- $L=D-W$: Graph Laplacian
- $\epsilon$: Gaussian kernel bandwidth

Overrides of global notation: $n$ = ambient dim. $k$ = #neighbors, ❌class idx. $j$ = neighbor rank or point idx, ❌feature idx. $\mathbf{y}_i$ = embedding, ❌label. $D$ = degree matrix, ❌distance.

&nbsp;

## Low-Dimensional Structure

### Manifold Hypothesis
- **What**: High-dim real data concentrates near a low-dim manifold $\mathcal{M}$, $d\ll n$. {cite:p}`bengio2013representation`
- **Why**: Explain one route to learning high-dim data efficiently.
    - Generic function on $\mathbb{R}^n$ → #samples exponential in $n$ ← [curse of dimensionality](../../ml/misc.md#curse-of-dimensionality).
    - Low effective dim can reduce sample complexity.
    - Simple target structure & inductive bias are other routes. E.g., full-dim Gaussian inputs + linear target → learnable.
- **How**: Few latent factors → smooth generator → manifold.
    1. Few factors of variation $\mathbf{z}\in\mathbb{R}^d$: pose, lighting, identity.
    2. Smooth embedding $g:\mathbb{R}^d\to\mathbb{R}^n$ renders them → $\{g(\mathbf{z})\}$ = $d$-dim manifold.
        - Smooth alone ❌: constant map → a point; self-intersecting immersion → not a manifold.
    3. Noise → data lies *near* $\mathcal{M}$, not on it. Discrete factors (class) → several disconnected pieces.
    4. Rest of $\mathbb{R}^n$ → ≈ zero probability. Uniform random pixels ≈ never a natural image.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $P$: Data distribution on $\mathbb{R}^n$.
    - $\mathrm{dist}(\mathbf{x},\mathcal{M})=\inf_{\mathbf{p}\in\mathcal{M}}\|\mathbf{x}-\mathbf{p}\|$: Distance to the manifold.
    - $\eta$: Small tolerance.

Definition:

$$
\mathbb{E}_{\mathbf{x}\sim P}\left[\mathrm{dist}(\mathbf{x},\mathcal{M})^2\right]\le\eta,\qquad \dim\mathcal{M}=d\ll n
$$

- **Reach** $\tau$: largest $r$ such that every point within distance $r$ of $\mathcal{M}$ has a unique nearest point on $\mathcal{M}$.
    - $\tau$⬇️ ← high curvature, or two parts of $\mathcal{M}$ nearly touch.
- **Union of manifolds** variant: support = disjoint union of manifolds w/ different $d$ (Brown et al., ICLR 2023).
- Consequence: sample complexity scales w/ $d$, volume & reach of $\mathcal{M}$, not w/ $n$.
```

```{attention} Q&A
:class: dropdown
*What is the evidence?*

- ID estimates of image datasets: tens vs. $10^3$–$10^5$ pixels → [Intrinsic Dimension of Images](#intrinsic-dimension-of-images).
- GANs generate realistic images from a latent vector w/ #dims ≪ #pixels → generated images lie on a set of dim ≤ latent dim.
- A high-density subset of log-intensity, mean-centered, contrast-normalized $3\times3$ natural image patches has Klein-bottle topology (Carlsson et al., 2008) → [Persistent Homology](topology.md#persistent-homology).

*Is it exactly true?*

- ❌ Approximation.
- Noise → full-dim support at small scales.
- Class structure → disconnected pieces w/ different ID; higher-ID classes are harder to classify (Brown et al., ICLR 2023).
- "Near a manifold" holds only in a window of scales → [Intrinsic Dimension](#intrinsic-dimension).

*How does it relate to DL?*

- Autoencoder / VAE: learned encoder & decoder (❌ guaranteed inverses).
- Standard GAN: generator only, ❌ encoder.
- Classifier: flatten & separate class manifolds layer by layer → [Intrinsic Dimension of Representations](#intrinsic-dimension-of-representations).
- Adversarial examples: one hypothesis = model is unconstrained off $\mathcal{M}$ → small off-manifold steps flip labels.
    - Contested: adversarial examples also exist on a known, exact data manifold (Gilmer et al., 2018, "Adversarial Spheres").
- Representation learning: a good representation = coordinates on $\mathcal{M}$ that disentangle the factors $\mathbf{z}$.

*Why does a low $d$ beat the curse of dimensionality?*

- Covering $\mathcal{M}$ at resolution $r$ needs $\propto r^{-d}$ balls, not $r^{-n}$.
- Local methods (kNN, kernels, NNs) only need data density on $\mathcal{M}$.

*What does this tell us about intelligence/AC?*

- Learning = discover $\mathcal{M}$ & coordinates on it → compress each observation to $d$ numbers.
- Brains also compress: neural population activity lies on low-dim neural manifolds → [Neural Manifold](repr.md#neural-manifold).
- Hypothesis: a world model = a learned chart of the data manifold. Low-dim structure alone says nothing about experience.
```

&nbsp;

#### Testing the Manifold Hypothesis
- **What**: Decide from i.i.d. samples whether $P$ lies near some $d$-dim manifold w/ bounded volume & reach. {cite:p}`fefferman2016testing`
- **Why**: The manifold hypothesis is usually assumed, not tested. Need a test w/ sample-complexity guarantees.
- **How**: Empirical risk minimization over a class of manifolds.
    1. Loss of a candidate $\mathcal{M}$ = mean squared distance of samples to it.
    2. Fit the best manifold in the class $\mathcal{G}(d,V,\tau)$.
    3. Compare its loss to the tolerance → answer (a) or (b) below.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $V$: Max volume of $\mathcal{M}$.
    - $\tau$: Min reach of $\mathcal{M}$.
    - $\eta$: Loss tolerance ($\varepsilon$ in the paper).
    - $\delta$: Failure probability.
- Misc:
    - $P$: Distribution supported on the unit ball of a separable Hilbert space.
    - $\mathcal{G}(d,V,\tau)$: $d$-dim submanifolds of the unit ball w/ volume $\le V$ & reach $\ge\tau$.
    - $C$: Constant that depends only on $d$.

Objective:

$$
\mathcal{L}(\mathcal{M},P)=\int\mathrm{dist}(\mathbf{x},\mathcal{M})^2\,dP(\mathbf{x})
$$

Test output (correct w/ probability $\ge1-\delta$; at least one holds):
- (a) $\exists\,\mathcal{M}\in\mathcal{G}(d,CV,\tau/C)$ w/ $\mathcal{L}(\mathcal{M},P)\le C\eta$.
- (b) $\nexists\,\mathcal{M}\in\mathcal{G}(d,V/C,C\tau)$ w/ $\mathcal{L}(\mathcal{M},P)\le\eta/C$.

Properties:
- #samples depends on $d,V,\tau,\eta,\delta$, ❌ on ambient dim → result holds in a separable Hilbert space.
- Cost bounds involve $N_p=V\left(\tau^{-d}+(\eta\tau)^{-d/2}\right)$ → exponential in $d$.
```

```{attention} Q&A
:class: dropdown
*Why must the class bound the reach & volume?*

- W/o bounds, a 1-dim curve can wiggle through every sample (space-filling) → loss 0 for any data.
- → "Data lies near a 1-dim manifold" becomes always true & empty.
- Reach ⬇️ / volume ⬆️ → more complex manifolds → more samples needed.

*Is it a practical tool?*

- ❌ Mainly a theory result: guarantees exist, but cost is exponential in $d$.
- Practice: estimate $d$ w/ [MLE](#mle-estimator) or [TwoNN](#twonn), then check the estimate is stable across scales.
```

&nbsp;

### Intrinsic Dimension
- **What**: Min #coordinates that describe the data locally, i.e., $\dim\mathcal{M}$.
- **Why**: $n$ (#pixels, #units) overstates complexity.
    - $d$ sets sample complexity.
    - $d$ guides dimensionality reduction; topology can require a larger output dim.
    - $d$ compares the complexity of datasets or layers w/ different $n$.
- **How**: Count how fast #neighbors grows w/ radius.
    1. On a $d$-dim manifold: #points within radius $r$ $\propto r^d$ for small $r$.
    2. → $d$ = slope of $\log(\#\text{neighbors})$ vs. $\log r$.
    3. kNN estimators ([MLE](#mle-estimator), [TwoNN](#twonn)) = likelihood versions of this slope.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $B(\mathbf{x},r)$: Ball of radius $r$ around $\mathbf{x}$.
    - $V_d$: Volume of the unit $d$-ball.
    - $f$: Density on $\mathcal{M}$.

Local ID:

$$
d(\mathbf{x})=\lim_{r\to0}\frac{\log P\big(B(\mathbf{x},r)\big)}{\log r}
$$

- Smooth $f$ on $\mathcal{M}$: $P\big(B(\mathbf{x},r)\big)\approx f(\mathbf{x})V_dr^d$ → $d(\mathbf{x})=d$.

Correlation (global) dimension:

$$
C(r)=P\big(\|\mathbf{x}-\mathbf{x}'\|<r\big)\propto r^d\quad(r\to0)
$$

- $\mathbf{x},\mathbf{x}'$: Two independent samples from $P$.
- $d$ = slope of $\log C(r)$ vs. $\log r$.

Properties:
- **Local vs. global**: ID can vary over the data (line ∪ plane → 1 & 2).
    - A global estimate hides this. Aggregation rule depends on the estimator.
    - E.g., correlation dimension, separated components → smallest component dim ← $r^{d_\text{min}}$ dominates as $r\to0$.
- **Scale**: finite data → ❌ $r\to0$. Measure in a window: noise scale ≪ $r$ ≪ curvature radius.
    - $r$ < noise scale → ID → $n$ ← noise fills all directions.
    - $r$ ~ curvature radius → curved $\mathcal{M}$ looks higher-dim.
    - $r$ ≫ data diameter → data ≈ one point → ID → 0.
- **Linear vs. nonlinear**: affine-span dim ≥ manifold dim. A curved $d$-dim manifold can span up to $n$ dims.
    - Variance-threshold PCA-ID can under- or overestimate $d$.
```

```{note} Example
:class: dropdown
Noisy unit circle in $\mathbb{R}^2$, noise std $0.01$:
- $r\ll0.01$ → ID ≈ 2 (noise).
- $0.01\ll r\ll1$ → ID ≈ 1 (circle).
- Noise-free circle: $r>2$ → $C(r)=1$ → ID = 0. W/ noise the threshold is not 2 in general.
- PCA: 2 components, each w/ 50% variance → PCA-ID = 2.

Swiss roll in $\mathbb{R}^3$: ID = 2, PCA-ID = 3.
```

```{dropdown} Table: ID Estimators
| Estimator | Input | Pros | Cons |
|:--|:--|:--|:--|
| PCA-ID | #PCs for e.g. 90% variance | Fast, stable | Overestimates curved $\mathcal{M}$; threshold is arbitrary |
| Correlation dimension (Grassberger–Procaccia) | Slope of $\log C(r)$ vs. $\log r$ | Shows the full scale curve | High variance; slope window chosen by hand |
| [MLE](#mle-estimator) | Distances to $k$ NNs | Closed form, low variance | Needs ≈ constant density on the $k$-NN scale |
| [TwoNN](#twonn) | Distances to 1st & 2nd NN | Smallest neighborhood → robust to curvature & density change | Noise at small scales → use block analysis |

- Levina & Bickel's simulations: correlation dimension has slightly lower bias than MLE, but ≥10× the SD.
```

```{attention} Q&A
:class: dropdown
*Why does PCA overestimate ID?*

- PCA counts flat directions. A curved manifold spans more of them.
- Circle: ID 1, PCA-ID 2.
- VGG-16 last hidden layer: PCA-ID ≈ 200, TwoNN ID ≈ 18 → [Intrinsic Dimension of Representations](#intrinsic-dimension-of-representations).

*Why do kNN estimators underestimate a large $d$?*

- kNN radius $\propto(k/m)^{1/d}$ → shrinks very slowly for large $d$ → neighborhoods are not local.
- Seeing $r^d$ scaling at large $d$ needs #samples exponential in $d$.
- → Large estimates are often biased downward. ❌ Guaranteed lower bounds.

*How does ID relate to compression?*

- #balls of radius $r$ to cover $\mathcal{M}$ $\propto r^{-d}$.
- → Describing a point to precision $r$ costs ≈ $d\log_2(1/r)$ bits.
- → ID = growth rate of description length as precision ⬆️ → [Lossy Compression](../info.md#lossy-compression).

*Same name, different object?*

- **ID of objective landscapes** (Li et al., ICLR 2018): min #params of a random subspace of weight space that reaches 90% of full-training performance.
- Low landscape ID of LLM fine-tuning (Aghajanyan et al., 2021) motivates [LoRA](../../nlp/post/sft.md#lora).
- Data/representation ID (this page) ≠ landscape ID.
```

&nbsp;

#### MLE Estimator
- **What**: Max-likelihood ID from $k$-NN distances, w/ neighbor counts modeled as a Poisson process. {cite:p}`levina2004maximum`
- **Why**: Correlation dimension → high variance & a hand-picked slope window. MLE → closed form, low variance.
- **How**:
    1. Per point: distances to its $k$ nearest neighbors.
    2. Given $r_{i,k}$: log-ratios of the interior points (unordered) ~ i.i.d. Exponential(rate $d$) → mean = $1/d$.
    3. Invert the mean → local $d$.
    4. Pool: average the log-ratios over all points, then invert.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $k_1$: Smallest $k$ to average over.
    - $k_2$: Largest $k$ to average over.

Process:
1. Local estimate:

    $$
    \hat d_k(\mathbf{x}_i)=\left[\frac{1}{k-1}\sum_{j=1}^{k-1}\log\frac{r_{i,k}}{r_{i,j}}\right]^{-1}
    $$

2. Global estimate (Levina & Bickel): average $\hat d_k(\mathbf{x}_i)$ over $i$, then over $k\in[k_1,k_2]$. Paper: $k_1=10$, $k_2=20$.
3. Global estimate (MacKay & Ghahramani correction): average the inverses, then invert:

    $$
    \bar d_k=\left[\frac1m\sum_{i=1}^m\hat d_k(\mathbf{x}_i)^{-1}\right]^{-1}=\left[\frac{1}{m(k-1)}\sum_{i=1}^m\sum_{j=1}^{k-1}\log\frac{r_{i,k}}{r_{i,j}}\right]^{-1}
    $$

Properties:
- Assumption: density ≈ constant inside $B(\mathbf{x}_i,r_{i,k})$.
- Local estimate is biased: $\mathbb{E}[\hat d_k(\mathbf{x}_i)]=d\,\frac{k-1}{k-2}$ → divide by $k-2$ for an unbiased local estimate.
```

```{tip} Derivation
:class: dropdown
*Where does the formula come from?*

1. Fix $\mathbf{x}$ & radius $R$. $N(t)$ = #points within distance $t\le R$.
2. Locally uniform density $f$ on a $d$-dim $\mathcal{M}$ → $N(t)$ ≈ Poisson process w/ rate

    $$
    \lambda(t)=\frac{d}{dt}\big(fV_dt^d\big)=fV_d\,d\,t^{d-1}
    $$

3. Log-likelihood of the observed distances $T_1<\cdots<T_N$, w/ $\theta=\log f$:

    $$
    \ell(d,\theta)=\sum_{j=1}^N\log\lambda(T_j)-\int_0^R\lambda(t)\,dt=\sum_{j=1}^N\big[\theta+\log V_d+\log d+(d-1)\log T_j\big]-e^\theta V_dR^d
    $$

4. $\partial\ell/\partial\theta=0$ → $e^\theta V_dR^d=N$.
5. $\partial\ell/\partial d=0$, then substitute step 4 → the $V_d'$ terms cancel:

    $$
    \frac Nd+\sum_j\log T_j=N\log R\ \Rightarrow\ \hat d=\left[\frac1N\sum_{j=1}^N\log\frac{R}{T_j}\right]^{-1}
    $$

6. Set $R=r_{i,k}$ → $N=k-1$ points inside → local formula.

*Why average the inverses?*

1. Given $r_{i,k}$: the $k-1$ interior points, taken unordered, have i.i.d. ratios $(r/r_{i,k})^d$ ~ Uniform$(0,1)$.
2. → Unordered log-ratios ~ i.i.d. Exponential(rate $d$) → mean $1/d$.
    - Ranked log-ratios $\log(r_{i,k}/r_{i,j})$ = dependent order statistics. Same sum → the Gamma result below still holds.
3. → $\hat d_k(\mathbf{x}_i)^{-1}$ is unbiased for $1/d$. Averaging these → unbiased for $1/d$.
4. $\hat d_k(\mathbf{x}_i)=(k-1)/S$ w/ $S\sim\text{Gamma}(k-1,\text{rate }d)$ → $\mathbb{E}[1/S]=d/(k-2)$ → biased up. Averaging $\hat d_k$ keeps this bias.
5. Pooling all log-ratios = joint MLE only if points are independent.
    - Neighborhoods overlap → dependence → pooling is an independence approximation.
```

````{important} Code
:class: dropdown
```python
import numpy as np

class MLEDimension:
    def __init__(self, k=10):
        self.k = k

    def __call__(self, X):
        sq = (X ** 2).sum(1)
        dist = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0))
        T = np.sort(dist, axis=1)[:, 1:self.k + 1]   ## col 0 of the sort = the point itself
        logs = np.log(T[:, -1:] / T[:, :-1])          ## log(r_ik / r_ij), j = 1..k-1 -> (m, k-1)
        local = 1.0 / logs.mean(1)                    ## Levina-Bickel local estimate
        pooled = 1.0 / logs.mean()                    ## MacKay-Ghahramani: average inverses, then invert
        return local, pooled

## Example: uniform 2-D square, linearly embedded in R^10
rng = np.random.default_rng(0)
X = rng.random((1000, 2)) @ rng.standard_normal((2, 10))
local, d = MLEDimension(k=10)(X)
print(local.shape, round(d, 1))  ## (1000,) and a value close to 2
```
````

```{attention} Q&A
:class: dropdown
*How to choose $k$?*

- $k$⬇️ → variance⬆️ & bias $\frac{k-1}{k-2}$⬆️.
- $k$⬆️ → ball leaves the locally uniform, locally flat region → bias from curvature & density change.
- Report several $k$ (e.g., 3, 5, 10, 20) & check the ranking is stable.

*What breaks in practice?*

- Duplicate points → $r_{i,1}=0$ → $\log$ blows up → deduplicate first.
- Near boundaries: radius-dependent truncation can bias estimates downward.
    - Exact flat half-ball keeps $r^d$ scaling → no bias there.
```

&nbsp;

#### TwoNN
- **What**: ID from the ratio of 2nd- to 1st-NN distances, which is Pareto-distributed w/ shape $d$. {cite:p}`facco2017estimating`
- **Why**: Large neighborhoods → curvature & density change bias the estimate. Smallest possible neighborhood → assumption needed only on the scale $r_{i,2}$.
- **How**:
    1. Per point: $\mu_i=r_{i,2}/r_{i,1}$.
    2. Theory: $F(\mu)=1-\mu^{-d}$ → $-\log(1-F(\mu))=d\log\mu$.
    3. Fit a line through the origin to the empirical points → slope = $d$. Discard the 10% largest $\mu$.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\mu_i=r_{i,2}/r_{i,1}\ge1$: Neighbor-distance ratio of point $i$.
    - $F$: CDF of $\mu$.

Process:
1. Compute $\mu_i$ for all $i$; sort ascending.
2. Empirical CDF: $F^{\text{emp}}(\mu_{(i)})=i/m$.
3. Fit $y=d\,x$ to $\{(\log\mu_{(i)},\,-\log(1-F^{\text{emp}}(\mu_{(i)})))\}$ on the smallest 90% of $\mu$.

Distribution:

$$
f(\mu)=d\,\mu^{-d-1},\qquad F(\mu)=1-\mu^{-d},\qquad \mu\ge1
$$

- Density $f$ of the data cancels → only local uniformity on the $r_{i,2}$ scale is needed.
- Alternative: Pareto MLE $\hat d=m/\sum_i\log\mu_i$.
- **Block analysis**: estimate on random subsets of size $m/2,m/4,\dots$ → NN distances ⬆️ → larger scale. Plateau of ID vs. scale = ID; ID⬆️ at small scale = noise.
```

```{tip} Derivation
:class: dropdown
*Why is $\mu$ Pareto($d$)?*

1. Shell volumes: $\Delta v_1=V_dr_1^d$, $\Delta v_2=V_d(r_2^d-r_1^d)$.
2. Constant density $\rho$ → $\Delta v_1,\Delta v_2$ i.i.d. Exponential(rate $\rho$).
3. Ratio of i.i.d. exponentials: $P(\Delta v_2/\Delta v_1>s)=\mathbb{E}\big[e^{-\rho s\Delta v_1}\big]=\frac{1}{1+s}$. $\rho$ cancels.
4. $\mu^d=r_2^d/r_1^d=1+\Delta v_2/\Delta v_1$ → $P(\mu>u)=P(\Delta v_2/\Delta v_1>u^d-1)=u^{-d}$.
5. → $F(\mu)=1-\mu^{-d}$ → $-\log(1-F)=d\log\mu$: a line through the origin w/ slope $d$.
```

````{important} Code
:class: dropdown
```python
import numpy as np

class TwoNN:
    def __init__(self, discard=0.1):
        self.discard = discard  ## drop the largest mu: heavy Pareto tail destabilizes the fit

    def __call__(self, X):
        sq = (X ** 2).sum(1)
        dist = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0))
        r = np.sort(dist, axis=1)[:, 1:3]             ## r1, r2 per point (col 0 = self)
        mu = np.sort(r[:, 1] / r[:, 0])
        m = len(mu)
        F = np.arange(1, m + 1) / m                   ## empirical CDF
        keep = int(m * (1 - self.discard))            ## also removes F = 1 -> log(0)
        x, y = np.log(mu[:keep]), -np.log(1 - F[:keep])
        d_fit = (x @ y) / (x @ x)                     ## least squares line through the origin
        d_mle = m / np.log(mu).sum()                  ## Pareto MLE on all points
        return d_fit, d_mle

## Example: Swiss roll = 2-D manifold in R^3
rng = np.random.default_rng(0)
t = 1.5 * np.pi * (1 + 2 * rng.random(2000))
X = np.c_[t * np.cos(t), 21 * rng.random(2000), t * np.sin(t)]
print(TwoNN()(X))  ## both values close to 2
```
````

```{attention} Q&A
:class: dropdown
*Why discard the largest 10% of $\mu$?*

- Heavy tail: sometimes $r_2\gg r_1$ → huge $\mu$ → the fit is unstable.
- Facco et al.: the estimate is robust to the exact fraction.

*How does noise affect it?*

- Noise acts at the smallest scale → TwoNN sees noise directions → ID⬆️.
- Fix: block analysis → pick the ID on the plateau at larger scales.

*Why is it popular for DNN representations?*

- Uses only 2 NNs → fast: $\sim10^4$ points w/ $\sim10^5$ coords in seconds.
- Works on curved, non-uniformly sampled manifolds.
```

&nbsp;

### Intrinsic Dimension of Images
- **What**: Natural image datasets: MLE ID ≈ 7–43 vs. 784–150528 pixel dims. {cite:p}`pope2021intrinsic`
- **Why**: Test the manifold hypothesis on real data; link ID to how hard a dataset is to learn.
- **How**:
    1. Validate MLE on GAN images: latent dim → known upper bound on ID.
    2. Estimate the ID of real datasets w/ the pooled [MLE](#mle-estimator), $k=3,5,10,20$.
    3. Train classifiers on data w/ controlled intrinsic vs. extrinsic dim → measure sample complexity.

```{dropdown} Table: MLE ID of Image Datasets
| Dataset | Pixel dims | $k=3$ | $k=5$ | $k=10$ | $k=20$ |
|:--|:--|:--|:--|:--|:--|
| MNIST | 784 | 7 | 11 | 12 | 13 |
| SVHN | 3072 | 9 | 14 | 18 | 19 |
| CIFAR-100 | 3072 | 11 | 18 | 22 | 23 |
| CelebA | — | 9 | 17 | 24 | 26 |
| CIFAR-10 | 3072 | 13 | 21 | 25 | 26 |
| MS-COCO | — | 22 | 33 | 37 | 36 |
| ImageNet | 150528 | 26 | 38 | 43 | 43 |

- Source: Pope et al. 2021, Table 1. Estimator: pooled MLE (MacKay–Ghahramani form).
- Rank order of datasets is stable across $k$.
```

```{attention} Q&A
:class: dropdown
*Does intrinsic or extrinsic dim drive sample complexity?*

- Intrinsic. GAN data w/ ID 16, 32, 64, 128 at fixed $3\times32\times32$ → higher ID needs more samples for the same test error.
- Fixed ID 128, images resized from 16 to 256 px → little to no effect.

*Does it hold on real data?*

- ✅ Lower-ID datasets → easier to learn & better generalization.
- Add noise from a random $\underline d$-dim hypercube to CIFAR-10 → measured ID⬆️ → sample complexity⬆️ in about the same order.
- Measured ID of the noised data stays below the true value ← too few samples.

*Caveats?*

- MLE assumptions (locally uniform, smooth) are unverified for images.
- Estimates rise w/ $k$ → read them as ranges, not exact values.
- Large $d$ → kNN estimators underestimate.
- One global number hides per-class differences ([union of manifolds](#manifold-hypothesis)).

*Why does DL avoid the curse of dimensionality on images?*

- Theory (cited by Pope et al.): samples to learn a well-conditioned boundary between manifold classes grow exponentially in ID, not in $n$.
- ImageNet ID ≈ 26–43 ≪ 150528 → learnable w/ realistic data. Architectural priors ([CNN](../../cv/cnn.md#cnn)) add more.
```

&nbsp;

### Intrinsic Dimension of Representations
- **What**: TwoNN ID of DNN layer activations: ≪ #units; first rises, then falls across layers. {cite:p}`ansuini2019intrinsic`
- **Why**: Ask what geometry a trained network builds, & whether it predicts generalization.
- **How**:
    1. Feed images (one class or one dataset) through a trained network.
    2. Per layer: [TwoNN](#twonn) ID of the activations.
    3. Compare profiles across architectures, training states & label noise.

```{dropdown} Table: Findings
| Finding | Detail |
|:--|:--|
| ID ≪ #units | Orders of magnitude smaller than layer width |
| Hunchback profile | ID⬆️ in the first layers, then ⬇️ gradually to the output (AlexNet, VGG, ResNet on ImageNet) |
| Last-hidden-layer ID predicts accuracy | Lower ID (train set) ↔ lower test top-5 error. $r=0.94$ (ID vs. error) across networks; $r=0.99$ within ResNets |
| Curved, not flat | VGG-16 last hidden layer: PCA-ID (90% var) ≈ 200 vs. TwoNN ≈ 18; no gap in the PCA spectrum |
| Untrained network | Flat ID profile |
| Random labels (MNIST) | ID does not fall; it grows in late layers, near the layer-width bound |
| Early ID rise | Can come from task-irrelevant, correlated input features (tested w/ random luminance shifts on MNIST) |
```

```{attention} Q&A
:class: dropdown
*What does the hunchback mean?*

- Early layers: finite-scale TwoNN estimate⬆️.
    - Exact pointwise ID cannot rise through deterministic Lipschitz layers → only finite-scale estimates can.
- Late layers: ID⬇️. Hypothesis: task-relevant compression → low-dim class manifolds → linear head separates them.
- Interpretation by the authors; the mechanism is not proven.

*Is last-layer ID a validation-free generalization proxy?*

- Partly. Strong correlation across ImageNet-pretrained networks.
- During training, last-layer ID is non-monotonic even w/o overfitting → ❌ reliable signal for overfitting onset.

*How does it relate to the Information Bottleneck?*

- Both say: late layers compress.
- IB compression = $I(X;T)$⬇️; ID compression = geometric dim⬇️. Different quantities.
- The IB "compression phase" is contested (depends on activation & MI estimator) → [Information Bottleneck](../info.md#information-bottleneck).

*What does this tell us about intelligence/AC?*

- ID profile = finite-scale representation statistic, ❌ direct measure of abstraction.
- Brains: visual cortex also builds low-dim, separable object manifolds → [Neural Manifold](repr.md#neural-manifold).
- Hypothesis: late-layer ID drop = task-relevant compression.
- Low TwoNN ID alone shows neither information loss nor experience.
```

&nbsp;

## Manifold Learning
- **What**: Nonlinear dimensionality reduction: recover low-dim coordinates $\mathbf{y}_i$ of points sampled from a curved $\mathcal{M}$.
- **Why**: [PCA](../../ml/unsupervised.md#pca) is linear → cannot unroll curved manifolds (Swiss roll → layers overlap).
- **How**: Shared pipeline:
    1. Build a kNN graph → trust only local Euclidean distances ← $\mathcal{M}$ is locally flat ([Tangent Space](manifold.md#tangent-space)).
    2. Encode one structure on the graph: geodesics / reconstruction weights / adjacency / random walk.
    3. Solve one eigenproblem → coordinates that preserve that structure.

```{dropdown} Table: Manifold Learning Methods
| Method | Preserves | Core solve | Out-of-sample | Failure modes |
|:--|:--|:--|:--|:--|
| [Isomap](#isomap) | Global geodesic distances | All-pairs shortest paths + dense top eigvecs, $O(m^3)$ | ❌ native; Nyström | Short-circuit edges; holes / non-convex domain; non-flat $\mathcal{M}$ (sphere) |
| [LLE](#lle) | Local linear reconstruction weights | Sparse bottom eigvecs | Reuse weights of new point | Sensitive to $k$ & regularization; non-uniform sampling; global shape distorted |
| [Laplacian Eigenmaps](#laplacian-eigenmaps) | Local adjacency (neighbors stay close) | Sparse generalized bottom eigvecs | ❌ native; Nyström | Global distances lost; density bias; repeated harmonics |
| [Diffusion Maps](#diffusion-maps) | Diffusion distance at scale $t$ | Eigvecs of a Markov matrix | Nyström | $\epsilon$ choice; dense kernel $O(m^2)$ memory |
| [t-SNE](../../ml/unsupervised.md#t-sne) | Local neighbor probabilities | Gradient descent on KL | ❌ | Global distances & cluster sizes not meaningful; non-convex |
| [UMAP](../../ml/unsupervised.md#umap) | Local fuzzy kNN graph | SGD w/ negative sampling | ✅ `transform` | Global distances only partly kept; density distorted |

- Spectral methods (top 4) → closed-form eigenproblem, global optimum.
- t-SNE & UMAP → non-convex, built for 2–3D visualization.
```

```{attention} Q&A
:class: dropdown
*Which method when?*

- Recover true coordinates of a flat-able manifold → Isomap.
- Cheap, sparse, local structure → LLE / Laplacian Eigenmaps.
- Noise robustness & a scale knob → Diffusion Maps.
- 2D visualization of clusters → t-SNE / UMAP.

*Why do all spectral methods end in an eigenproblem?*

- Each = kernel PCA w/ a data-dependent Gram matrix (Ham et al., ICML 2004).
    - Isomap: $-\frac12HD_G^{(2)}H$. LLE & Laplacian Eigenmaps: functions of $M$ or $L$.
- Quadratic objective + orthogonality constraint → Rayleigh quotient → eigenvectors.

*How to pick $k$?*

- $k$⬇️ → graph disconnects → infinite geodesics / several zero eigenvalues.
- $k$⬆️ → edges jump across folds → short-circuit.

*How does this relate to deep learning?*

- Autoencoder = parametric manifold learning: encoder = learned chart → out-of-sample ✅, scales to large $m$; ❌ global-optimum guarantee.
- Laplacian eigenvectors → positional encodings in graph transformers.

*What does this tell us about intelligence/AC?*

- Tenenbaum et al. frame Isomap as the problem perception solves: few relevant features from high-dim sensory input.
- Hypothesis: brains build maps from local similarity, like manifold learning. Example: grid cells → [Toroidal Grid-Cell Manifold](topology.md#toroidal-grid-cell-manifold).
```

&nbsp;

### Isomap
- **What**: Classical MDS on kNN-graph shortest-path distances. {cite:p}`tenenbaum2000global`
- **Why**: Euclidean distance cuts through ambient space → wrong for a curved $\mathcal{M}$.
    - Swiss roll: two layers are close in $\mathbb{R}^3$ but far along the roll.
    - → Preserve [geodesic](manifold.md#geodesic) distance instead.
- **How**:
    1. kNN graph; edge weight = Euclidean distance (valid locally).
    2. All-pairs shortest paths ≈ geodesic distances.
    3. Classical MDS → coordinates whose Euclidean distances match them.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $D_G\in\mathbb{R}^{m\times m}$: Shortest-path distances in the kNN graph.
    - $D_G^{(2)}$: Elementwise square of $D_G$.
    - $H=I-\frac1m\mathbf{1}\mathbf{1}^T$: Centering matrix.

Process:
1. Graph: edge $(i,j)$ w/ weight $\|\mathbf{x}_i-\mathbf{x}_j\|$ if $j\in\mathcal{N}(i)$ or $i\in\mathcal{N}(j)$.
2. $D_G$ ← Dijkstra / Floyd–Warshall.
3. Double centering:

    $$
    B=-\tfrac12HD_G^{(2)}H
    $$

4. $B=V\Lambda V^T$, eigenvalues descending → $Y=V_d(\Lambda_d^+)^{1/2}$.
    - $V_d$: Top-$d$ eigenvectors.
    - $\Lambda_d$: Top-$d$ eigenvalues.
    - $\Lambda_d^+$: $\Lambda_d$ w/ negative eigenvalues clipped to 0 ← $B$ can be indefinite.

Objective:

$$
\min_Y\big\|\tau(D_G)-\tau(D_Y)\big\|_F,\qquad\tau(D)=-\tfrac12HD^{(2)}H
$$

- $D_Y$: Euclidean distances between the $\mathbf{y}_i$.
```

```{tip} Derivation
:class: dropdown
*Why does double centering give a Gram matrix?*

1. Assume $D^{(2)}_{ij}=\|\mathbf{y}_i-\mathbf{y}_j\|^2$ w/ centered points $\sum_i\mathbf{y}_i=\mathbf{0}$.
2. Expand: $D^{(2)}=\mathbf{g}\mathbf{1}^T+\mathbf{1}\mathbf{g}^T-2YY^T$, w/ $\mathbf{g}=\mathrm{diag}(YY^T)$.
3. $H\mathbf{1}=\mathbf{0}$ → both rank-1 terms vanish under $H(\cdot)H$.
4. $HY=Y$ (centered) → $-\frac12HD^{(2)}H=YY^T$.
5. Best rank-$d$ PSD fit of $B$ in Frobenius norm (Eckart–Young) → top-$d$ eigenpairs → $Y=V_d(\Lambda_d^+)^{1/2}$.
```

````{important} Code
:class: dropdown
```python
import numpy as np
from scipy.sparse.csgraph import shortest_path

class Isomap:
    def __init__(self, k=10, d=2):
        self.k, self.d = k, d

    def fit_transform(self, X):
        m = len(X)
        sq = (X ** 2).sum(1)
        dist = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0))
        ## 1. kNN graph: Euclidean edges to the k nearest points only (0 = no edge)
        nn = np.argsort(dist, axis=1)[:, 1:self.k + 1]
        rows = np.repeat(np.arange(m), self.k)
        G = np.zeros_like(dist)
        G[rows, nn.ravel()] = dist[rows, nn.ravel()]
        ## 2. shortest paths ~ geodesics; graph must be connected (else inf)
        DG = shortest_path(G, directed=False)
        ## 3. classical MDS: double-center squared distances -> Gram matrix
        H = np.eye(m) - np.ones((m, m)) / m
        B = -0.5 * H @ (DG ** 2) @ H
        lam, V = np.linalg.eigh(B)                        ## ascending
        lam, V = lam[::-1][:self.d], V[:, ::-1][:, :self.d]
        return V * np.sqrt(np.maximum(lam, 0))            ## clip: B can be indefinite

## Example: Swiss roll
rng = np.random.default_rng(0)
t = 1.5 * np.pi * (1 + 2 * rng.random(800))
X = np.c_[t * np.cos(t), 21 * rng.random(800), t * np.sin(t)]
print(Isomap(k=10, d=2).fit_transform(X).shape)  ## (800, 2)
```
````

```{attention} Q&A
:class: dropdown
*When does Isomap recover the true coordinates?*

- $\mathcal{M}$ isometric to a convex subset of $\mathbb{R}^d$ + dense sampling + small neighborhoods.
- Tenenbaum et al.: under these conditions, graph distances → geodesic distances as density⬆️.

*When does it fail?*

- **Short-circuit**: one edge across two folds → many geodesics shortcut → global embedding folds. Risk⬆️ w/ $k$⬆️ or noise⬆️.
- **Holes / non-convex domain**: geodesics detour around the hole → distances inflated → distortion.
- **Non-flat $\mathcal{M}$** (sphere): curvature ≠ 0 → no isometric map to $\mathbb{R}^d$ exists → $B$ has negative eigenvalues.

*How to pick $d$?*

- Residual variance $1-R^2(D_G,D_Y)$ vs. $d$ → pick the elbow.

*Relation to PCA?*

- Complete graph → shortest paths = Euclidean distances → classical MDS = PCA scores of the centered data.

*Cost?*

- Shortest paths: $O(m^2\log m+m^2k)$ w/ Dijkstra. Dense eigendecomposition: $O(m^3)$. Memory: $O(m^2)$.
- Landmark Isomap: shortest paths & MDS on a landmark subset → scales.
```

&nbsp;

### LLE
- **Name**: Locally Linear Embedding {cite:p}`roweis2000nonlinear`
- **What**: Embedding that keeps each point's linear reconstruction weights from its neighbors.
- **Why**: $\mathcal{M}$ is locally ≈ a linear patch.
    - Reconstruction weights are invariant to rotation, scaling & translation of the patch.
    - → Same weights hold in the intrinsic coordinates. ❌ Distances or shortest paths needed.
- **How**:
    1. Find $k$ nearest neighbors of each point.
    2. Solve weights: reconstruct $\mathbf{x}_i$ from its neighbors; weights sum to 1.
    3. Fix the weights; find $\mathbf{y}_i$ w/ the same reconstructions → bottom eigenvectors of $M$.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $W\in\mathbb{R}^{m\times m}$: Reconstruction weights. Override: not the symmetric graph weights; row $i$ holds the weights of $\mathbf{x}_i$'s neighbors.
- Hyperparams:
    - $\gamma$: Regularization strength.
- Misc:
    - $C^{(i)}\in\mathbb{R}^{k\times k}$: Local Gram matrix of point $i$.
    - $M=(I-W)^T(I-W)$: Embedding cost matrix.

Objective (weights):

$$
\min_W\sum_{i=1}^m\Big\|\mathbf{x}_i-\sum_{j\in\mathcal{N}(i)}W_{ij}\mathbf{x}_j\Big\|^2\quad\text{s.t.}\ \sum_jW_{ij}=1,\ W_{ij}=0\ \text{for}\ j\notin\mathcal{N}(i)
$$

Solution per point:

$$
C^{(i)}_{jl}=(\mathbf{x}_i-\mathbf{x}_j)^T(\mathbf{x}_i-\mathbf{x}_l),\qquad \mathbf{w}_i=\frac{(C^{(i)})^{-1}\mathbf{1}}{\mathbf{1}^T(C^{(i)})^{-1}\mathbf{1}}
$$

- $j,l\in\mathcal{N}(i)$.
- $k>n$ → $C^{(i)}$ singular → use $C^{(i)}+\gamma\,\mathrm{tr}(C^{(i)})I$.

Objective (embedding):

$$
\min_Y\sum_{i=1}^m\Big\|\mathbf{y}_i-\sum_jW_{ij}\mathbf{y}_j\Big\|^2=\mathrm{tr}(Y^TMY)\quad\text{s.t.}\ \sum_i\mathbf{y}_i=\mathbf{0},\ \tfrac1mY^TY=I
$$

- Solution: eigenvectors of $M$ for the 2nd … $(d+1)$-th smallest eigenvalues.
- Smallest eigenvalue 0 w/ eigenvector $\mathbf{1}$ ← $W\mathbf{1}=\mathbf{1}$ → discard it.
```

```{tip} Derivation
:class: dropdown
*Why the closed form for $\mathbf{w}_i$?*

1. $\sum_jw_j=1$ → $\mathbf{x}_i-\sum_jw_j\mathbf{x}_j=\sum_jw_j(\mathbf{x}_i-\mathbf{x}_j)$.
2. → Error $=\mathbf{w}^TC^{(i)}\mathbf{w}$.
3. Lagrangian $\mathbf{w}^TC^{(i)}\mathbf{w}-\lambda(\mathbf{1}^T\mathbf{w}-1)$ → $2C^{(i)}\mathbf{w}=\lambda\mathbf{1}$ → $\mathbf{w}\propto(C^{(i)})^{-1}\mathbf{1}$.
4. Normalize to sum 1.

*Why the trace form?*

1. Row $i$ of $(I-W)Y$ = $\mathbf{y}_i^T-\sum_jW_{ij}\mathbf{y}_j^T$.
2. → Cost $=\|(I-W)Y\|_F^2=\mathrm{tr}\big(Y^T(I-W)^T(I-W)Y\big)$.
3. Orthogonality constraint → Rayleigh quotient → bottom eigenvectors.
```

````{important} Code
:class: dropdown
```python
import numpy as np

class LLE:
    def __init__(self, k=10, d=2, reg=1e-3):
        self.k, self.d, self.reg = k, d, reg

    def fit_transform(self, X):
        m = len(X)
        sq = (X ** 2).sum(1)
        dist = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0))
        nn = np.argsort(dist, axis=1)[:, 1:self.k + 1]
        W = np.zeros((m, m))
        for i in range(m):
            Z = X[nn[i]] - X[i]                              ## neighbors relative to x_i: (k, n)
            C = Z @ Z.T                                      ## local Gram C^(i): (k, k)
            C += self.reg * np.trace(C) * np.eye(self.k)     ## k > n -> C singular -> regularize
            w = np.linalg.solve(C, np.ones(self.k))
            W[i, nn[i]] = w / w.sum()                        ## sum-to-one -> translation invariance
        M = (np.eye(m) - W).T @ (np.eye(m) - W)
        lam, V = np.linalg.eigh(M)                           ## ascending
        return V[:, 1:self.d + 1] * np.sqrt(m)               ## skip constant eigvec; (1/m) Y^T Y = I

## Example: Swiss roll
rng = np.random.default_rng(0)
t = 1.5 * np.pi * (1 + 2 * rng.random(800))
X = np.c_[t * np.cos(t), 21 * rng.random(800), t * np.sin(t)]
print(LLE(k=12, d=2).fit_transform(X).shape)  ## (800, 2)
```
````

```{attention} Q&A
:class: dropdown
*Why must the weights sum to 1?*

- Translate the patch by $\mathbf{c}$: $\mathbf{x}_i+\mathbf{c}-\sum_jw_j(\mathbf{x}_j+\mathbf{c})=\mathbf{x}_i-\sum_jw_j\mathbf{x}_j$ only if $\sum_jw_j=1$.
- → Weights encode shape, not position.

*Pros?*

- One sparse eigenproblem; no distances beyond the neighborhood.
- No local minima.

*Cons?*

- Sensitive to $k$ & $\gamma$.
- ❌ Preserves distances → global shape & aspect ratio distorted.
- Non-uniform sampling → distortion. Variants (Hessian LLE, Modified LLE) fix some of this.

*Out-of-sample?*

- New $\mathbf{x}$: compute its weights on training neighbors → $\mathbf{y}=\sum_jw_j\mathbf{y}_j$.
```

&nbsp;

### Laplacian Eigenmaps
- **What**: Embedding by the bottom generalized eigenvectors of a heat-kernel graph Laplacian. {cite:p}`belkin2003laplacian`
- **Why**: Locality preservation: close points → close embeddings.
    - Graph Laplacian ≈ discrete Laplace–Beltrami operator → embedding has a geometric meaning.
    - Sparse graph → cheap.
- **How**:
    1. kNN graph; heat-kernel edge weights.
    2. Solve $L\mathbf{f}=\lambda D\mathbf{f}$.
    3. Drop the constant $\mathbf{f}_0$; $\mathbf{y}_i=(f_1(i),\dots,f_d(i))$.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\mathbf{f}_\ell\in\mathbb{R}^m$: $\ell$-th generalized eigenvector; $f_\ell(i)$ = its $i$-th entry.
    - $\lambda_\ell$: Its eigenvalue, $0=\lambda_0\le\lambda_1\le\cdots$.

Process:
1. Edge $(i,j)$ if $j\in\mathcal{N}(i)$ or $i\in\mathcal{N}(j)$.
2. Weights:

    $$
    W_{ij}=\begin{cases}\exp\left(-\|\mathbf{x}_i-\mathbf{x}_j\|^2/\epsilon\right)&\text{edge }(i,j)\\0&\text{otherwise}\end{cases}
    $$

    - Simple variant: $W_{ij}=1$ on edges ($\epsilon=\infty$).
3. Solve $L\mathbf{f}=\lambda D\mathbf{f}$.
4. $\mathbf{y}_i=(f_1(i),\dots,f_d(i))$.

Objective:

$$
\min_Y\ \tfrac12\sum_{i,j}W_{ij}\|\mathbf{y}_i-\mathbf{y}_j\|^2=\mathrm{tr}(Y^TLY)\quad\text{s.t.}\ Y^TDY=I,\ Y^TD\mathbf{1}=\mathbf{0}
$$

- $Y^TDY=I$ → fixes scale, prevents collapse to a point.
- $Y^TD\mathbf{1}=\mathbf{0}$ → excludes the constant eigenvector.
- Continuum: minimize $\int_\mathcal{M}\|\nabla f\|^2$ s.t. $\|f\|_{L^2(\mathcal{M})}=1$ → eigenfunctions of the Laplace–Beltrami operator $\Delta_\mathcal{M}$.
```

```{tip} Derivation
:class: dropdown
*Why a generalized eigenproblem?*

1. One coordinate $\mathbf{f}$: minimize $\mathbf{f}^TL\mathbf{f}$ s.t. $\mathbf{f}^TD\mathbf{f}=1$.
2. Lagrangian → $L\mathbf{f}=\lambda D\mathbf{f}$; cost at the optimum = $\lambda$.
3. Smallest: $\lambda_0=0$, $\mathbf{f}_0=\mathbf{1}$ ← $L\mathbf{1}=\mathbf{0}$ → all points get the same value → useless → drop.
4. $d$ coordinates + $D$-orthogonality → next $d$ eigenvectors.

*How to solve it w/ a symmetric eigensolver?*

1. $\mathbf{f}=D^{-1/2}\mathbf{v}$ → $D^{-1/2}LD^{-1/2}\mathbf{v}=\lambda\mathbf{v}$ (i.e., $L_\text{sym}$).
2. $\mathbf{f}^TD\mathbf{f}=\mathbf{v}^T\mathbf{v}$ → orthonormal $\mathbf{v}$ ↔ $D$-orthonormal $\mathbf{f}$.
3. Equivalent: $\mathbf{f}$ = eigenvectors of $L_\text{rw}=D^{-1}L$.
```

````{important} Code
:class: dropdown
```python
import numpy as np

class LaplacianEigenmaps:
    def __init__(self, k=10, d=2, eps=None):
        self.k, self.d, self.eps = k, d, eps

    def fit_transform(self, X):
        m = len(X)
        sq = (X ** 2).sum(1)
        D2 = np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0)  ## squared distances
        nn = np.argsort(D2, axis=1)[:, 1:self.k + 1]
        A = np.zeros((m, m), dtype=bool)
        A[np.repeat(np.arange(m), self.k), nn.ravel()] = True
        A = A | A.T                                       ## symmetric kNN graph
        eps = self.eps or np.median(D2[A])                ## heuristic bandwidth
        W = np.where(A, np.exp(-D2 / eps), 0.0)           ## heat kernel on edges only
        deg = W.sum(1)
        L = np.diag(deg) - W
        ## L f = lam D f  <=>  L_sym v = lam v, f = D^{-1/2} v
        s = 1 / np.sqrt(deg)
        lam, V = np.linalg.eigh(s[:, None] * L * s[None, :])  ## ascending
        F = s[:, None] * V
        return F[:, 1:self.d + 1]                         ## drop constant f_0 (lam = 0)

## Example: noisy circle in R^2
rng = np.random.default_rng(0)
th = 2 * np.pi * rng.random(300)
X = np.c_[np.cos(th), np.sin(th)] + 0.03 * rng.standard_normal((300, 2))
print(LaplacianEigenmaps(k=10, d=2).fit_transform(X).shape)  ## (300, 2)
```
````

```{attention} Q&A
:class: dropdown
*Why $Y^TDY=I$, not $Y^TY=I$?*

- High-degree (dense-region) points get more weight → solution = eigenvectors of $L_\text{rw}$ (random walk).
- Matches the normalized-cut relaxation → same eigenvectors as normalized [Spectral Clustering](../../ml/unsupervised.md#spectral-clustering).

*What is the continuum limit?*

- Uniform sampling on $\mathcal{M}$, $m\to\infty$, $\epsilon\to0$ at a suitable rate → scaled graph Laplacian → $\Delta_\mathcal{M}$ (Belkin & Niyogi, later work).
- Non-uniform density → limit includes a density-drift term → fix: [Diffusion Maps](#diffusion-maps) w/ $\alpha=1$.
- Heat kernel on $\mathcal{M}$ ≈ Gaussian for small time → motivates the weights.

*Relation to spectral clustering?*

- Spectral clustering = Laplacian Eigenmaps + K-Means on the embedding.
- Clusters → near-zero eigenvalues → embedding separates them into blobs.

*Relation to LLE?*

- Belkin & Niyogi: under some conditions, LLE's $M$ ≈ the squared Laplacian → same eigenvectors, approximately.

*Cons?*

- Only local constraints → global distances & aspect ratio lost.
- **Repeated harmonics**: long strip $[0,a]\times[0,b]$, $a>2b$ → eigenfunctions $\cos(\pi x/a)$, $\cos(2\pi x/a)$ come before $\cos(\pi y/b)$ → 2nd coordinate is a function of the 1st → 2nd intrinsic direction missed.
- Sensitive to $k$ & $\epsilon$.
```

&nbsp;

### Diffusion Maps
- **What**: Embedding by the eigenvectors of a random-walk (Markov) matrix on the data, scaled by $\lambda_\ell^t$. {cite:p}`coifman2006diffusion`
- **Why**: Robust, multi-scale geometry.
    - One short-circuit edge corrupts shortest paths. Diffusion distance averages over all paths → robust to noise.
    - $t$ = scale knob.
    - $\alpha$ removes the sampling-density effect → geometry only.
- **How**:
    1. Gaussian kernel → $\alpha$-normalize (divide out density).
    2. Row-normalize → Markov matrix $P$.
    3. Eigenvectors $\boldsymbol\psi_\ell$ & eigenvalues $\lambda_\ell$ of $P$.
    4. Coordinates $\lambda_\ell^t\psi_\ell(i)$ → Euclidean distance = diffusion distance at time $t$.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\alpha\in[0,1]$: Density normalization.
    - $t$: Diffusion time (scale).
- Misc:
    - $K\in\mathbb{R}^{m\times m}$: Gaussian kernel matrix.
    - $K^{(\alpha)}$: Density-normalized kernel.
    - $D=\mathrm{diag}(K^{(\alpha)}\mathbf{1})$: Degree matrix of $K^{(\alpha)}$ (override of Tier 2).
    - $P=D^{-1}K^{(\alpha)}$: Markov transition matrix.
    - $\boldsymbol\psi_\ell$: $\ell$-th right eigenvector of $P$.
    - $\boldsymbol\phi_\ell$: $\ell$-th left eigenvector of $P$.
    - $\lambda_\ell$: Eigenvalues, $1=\lambda_0>|\lambda_1|\ge|\lambda_2|\ge\cdots$ (connected graph).
    - $\phi_0(u)=D_{uu}/\sum_vD_{vv}$: Stationary distribution.

Process:
1. $K_{ij}=\exp\left(-\|\mathbf{x}_i-\mathbf{x}_j\|^2/\epsilon\right)$.
2. $q_i=\sum_jK_{ij}$ (density estimate) → $K^{(\alpha)}_{ij}=K_{ij}/(q_i^\alpha q_j^\alpha)$.
3. $P=D^{-1}K^{(\alpha)}$ → rows sum to 1.
4. $P\boldsymbol\psi_\ell=\lambda_\ell\boldsymbol\psi_\ell$; $\boldsymbol\psi_0=\mathbf{1}$; normalize $\sum_u\phi_0(u)\psi_\ell(u)\psi_{\ell'}(u)=\delta_{\ell\ell'}$.
5. Diffusion map:

    $$
    \Psi_t(\mathbf{x}_i)=\big(\lambda_1^t\psi_1(i),\dots,\lambda_d^t\psi_d(i)\big)
    $$

Diffusion distance:

$$
\mathrm{Dist}_t(i,j)^2=\sum_u\frac{\big(P^t_{iu}-P^t_{ju}\big)^2}{\phi_0(u)}=\sum_{\ell\ge1}\lambda_\ell^{2t}\big(\psi_\ell(i)-\psi_\ell(j)\big)^2
$$

- Truncate at $d$ where $\lambda_{d+1}^t$ is negligible → $\|\Psi_t(\mathbf{x}_i)-\Psi_t(\mathbf{x}_j)\|\approx\mathrm{Dist}_t(i,j)$.

$\alpha$ limits ($\epsilon\to0$, $m\to\infty$):
- $\alpha=0$ → classical normalized graph Laplacian; limit = Laplace–Beltrami + density drift.
- $\alpha=\frac12$ → Fokker–Planck operator (Langevin dynamics).
- $\alpha=1$ → Laplace–Beltrami only → geometry independent of sampling density.
```

```{tip} Derivation
:class: dropdown
*Why does the embedding distance equal the diffusion distance?*

1. Biorthogonal expansion: $P^t_{iu}=\sum_\ell\lambda_\ell^t\psi_\ell(i)\phi_\ell(u)$, w/ $\phi_\ell(u)=\phi_0(u)\psi_\ell(u)$.
2. → $P^t_{iu}-P^t_{ju}=\sum_\ell\lambda_\ell^t\big(\psi_\ell(i)-\psi_\ell(j)\big)\phi_0(u)\psi_\ell(u)$.
3. Square, divide by $\phi_0(u)$, sum over $u$ → cross terms vanish by $\sum_u\phi_0\psi_\ell\psi_{\ell'}=\delta_{\ell\ell'}$.
4. $\ell=0$: $\psi_0=\mathbf{1}$ → difference 0 → drop.

*How to get the eigenvectors w/ a symmetric solver?*

1. $S=D^{-1/2}K^{(\alpha)}D^{-1/2}$ is symmetric & similar to $P$.
2. $S\mathbf{v}=\lambda\mathbf{v}$ → $\boldsymbol\psi=\sqrt{\textstyle\sum_vD_{vv}}\,D^{-1/2}\mathbf{v}$ (normalized in $\phi_0$).
3. Gaussian $K$ is PSD → $S$ PSD → $\lambda_\ell\in[0,1]$.
```

````{important} Code
:class: dropdown
```python
import numpy as np

class DiffusionMap:
    def __init__(self, eps, d=2, t=1, alpha=1.0):
        self.eps, self.d, self.t, self.alpha = eps, d, t, alpha

    def fit_transform(self, X):
        sq = (X ** 2).sum(1)
        D2 = np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0)
        K = np.exp(-D2 / self.eps)
        q = K.sum(1)                                   ## kernel density estimate
        K = K / np.outer(q, q) ** self.alpha           ## alpha-normalization: divide out density
        deg = K.sum(1)
        ## P = D^-1 K is not symmetric -> eigen-solve the similar S = D^-1/2 K D^-1/2
        s = 1 / np.sqrt(deg)
        lam, V = np.linalg.eigh(s[:, None] * K * s[None, :])
        lam, V = lam[::-1], V[:, ::-1]                 ## descending: lam_0 = 1 first
        psi = V * s[:, None] * np.sqrt(deg.sum())      ## right eigvecs of P; psi_0 = 1
        return psi[:, 1:self.d + 1] * lam[1:self.d + 1] ** self.t

## Example: non-uniformly sampled noisy circle
rng = np.random.default_rng(0)
th = 2 * np.pi * rng.random(300) ** 2                  ## denser near angle 0
X = np.c_[np.cos(th), np.sin(th)] + 0.02 * rng.standard_normal((300, 2))
print(DiffusionMap(eps=0.05, d=2, t=2).fit_transform(X).shape)  ## (300, 2)
```
````

```{attention} Q&A
:class: dropdown
*Why $\alpha=1$?*

- $\alpha=0$: embedding mixes geometry w/ sampling density → dense regions dominate.
- $\alpha=1$: divides out density → recovers $\mathcal{M}$'s Riemannian geometry ([Riemannian Metric](manifold.md#riemannian-metric)) regardless of how it was sampled.
- $\alpha=\frac12$: use to model a stochastic dynamical system (Langevin).

*Relation to Laplacian Eigenmaps?*

- $P\boldsymbol\psi=\lambda\boldsymbol\psi$ ⇔ $(D-K^{(\alpha)})\boldsymbol\psi=(1-\lambda)D\boldsymbol\psi$.
- → $\alpha=0$ + same kernel → same eigenvectors as Laplacian Eigenmaps.
- DM adds: $\lambda^t$ scaling, a distance w/ meaning (diffusion distance), & density control.

*What does $t$ do?*

- $t$⬆️ → $\lambda_\ell^t$ of small eigenvalues → 0 → fewer coordinates matter → coarser scale.
- Cluster at scale $t$ = region a random walk rarely leaves within $t$ steps.

*Are diffusion maps related to diffusion models?*

- ❌ Name only. Diffusion maps: random walk on the data graph. [Diffusion models](../../dl/gen/diffusion.md): noising process in ambient space + learned reverse.

*Pros?*

- Robust to noise ← averages over all paths.
- Multi-scale ← $t$.
- Density-independent geometry ← $\alpha=1$.

*Cons?*

- $\epsilon$ choice: ⬇️ → graph disconnects; ⬆️ → curvature smoothed away.
- Dense kernel → $O(m^2)$ memory. Fix: kNN-sparsified kernel.
- Same repeated-harmonics issue as Laplacian Eigenmaps.
```

&nbsp;
