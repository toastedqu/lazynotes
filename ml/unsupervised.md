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
# Unsupervised Learning
Find structure in data with NO labels: group it, compress it, or flag what doesn't belong.

This page covers prevalent traditional unsupervised methods ONLY.

Default notations:
- $X=[\mathbf{x}_1,\cdots,\mathbf{x}_m]^T\in\mathbb{R}^{m\times n}$: Input matrix ($m$ samples, $n$ features). No $\mathbf{y}$.
- $k$: #clusters / #components / #latent dims.

&nbsp;

## Clustering
- **What**: Samples partitioned into groups with ⬆️within-group & ⬇️between-group similarity.

### K-Means
- **What**: Samples partitioned into $k$ clusters by nearest centroid.
- **Why**: Minimizing within-cluster variance over all partitions is NP-hard.
    - → Alternating minimization: monotonically decreasing objective at $O(mnkT)$.
    - Centroids double as interpretable cluster prototypes.
- **How**:
    1. Initialize $k$ centroids.
    2. Repeat until centroids stabilize (or max iterations):
        1. **Assign**: each sample → nearest centroid.
        2. **Update**: each centroid → mean of its assigned samples.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\boldsymbol{\mu}_c\in\mathbb{R}^n$: Centroid of cluster $c$.
- Hyperparams:
    - $k$: #clusters.
    - $T$: Max #iterations.
- Misc:
    - $c_i\in\{1,\cdots,k\}$: Cluster assignment of sample $i$.
    - $C_c=\{i:c_i=c\}$: Index set of cluster $c$.

Objective (WCSS / inertia):

$$
L(\{c_i\},\{\boldsymbol{\mu}_c\})=\sum_{i=1}^{m}||\mathbf{x}_i-\boldsymbol{\mu}_{c_i}||_2^2
$$

Process (Lloyd's algorithm) — alternating minimization of $L$:
1. Assign (minimize over $c_i$ with $\boldsymbol{\mu}$ fixed):

    $$
    c_i=\arg\min_c||\mathbf{x}_i-\boldsymbol{\mu}_c||_2^2
    $$

2. Update (minimize over $\boldsymbol{\mu}_c$ with $c$ fixed):

    $$
    \boldsymbol{\mu}_c=\frac{1}{|C_c|}\sum_{i\in C_c}\mathbf{x}_i
    $$

Complexity: $O(mnkT)$ time, $O((m+k)n)$ memory.
```

```{tip} Derivation
:class: dropdown
*Why is the centroid the mean, and why must it converge?*
1. With assignments fixed, $L$ decomposes per cluster: $L_c=\sum_{i\in C_c}||\mathbf{x}_i-\boldsymbol{\mu}_c||_2^2$.
2. $\frac{\partial L_c}{\partial\boldsymbol{\mu}_c}=-2\sum_{i\in C_c}(\mathbf{x}_i-\boldsymbol{\mu}_c)=0\ \Rightarrow\ \boldsymbol{\mu}_c=\frac{1}{|C_c|}\sum_{i\in C_c}\mathbf{x}_i$.
    - → This is exactly why K-Means is welded to **squared Euclidean** distance: the mean is the minimizer of squared error, not of L1 or cosine.
3. Both steps minimize the SAME $L$ → $L$ is non-increasing every iteration.
4. #Possible partitions is finite → $L$ can't decrease forever → converges in finite steps.
5. ⚠️ Converges to a **local** minimum only — nothing guarantees the global one.
```

````{important} Code
:class: dropdown
```python
import numpy as np

class KMeans:
    def __init__(self, k=3, n_iter=100, tol=1e-6, seed=0):
        self.k, self.n_iter, self.tol, self.seed = k, n_iter, tol, seed
        self.centroids = None

    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        ## naive init: k random distinct samples (see K-Means++ for the smart version)
        self.centroids = X[rng.choice(len(X), self.k, replace=False)]
        for _ in range(self.n_iter):
            ## (m, 1, n) - (1, k, n) -> (m, k) squared distances
            d = ((X[:, None, :] - self.centroids[None, :, :]) ** 2).sum(axis=2)
            labels = d.argmin(axis=1)
            new = np.array([X[labels == c].mean(axis=0) if (labels == c).any()
                            else self.centroids[c] for c in range(self.k)])
            if np.abs(new - self.centroids).max() < self.tol:   ## centroids stabilized
                break
            self.centroids = new
        return self

    def predict(self, X):
        return ((X[:, None, :] - self.centroids[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)

## Example: two well-separated blobs
X = np.array([[0.0, 0.0], [0.2, 0.1], [5.0, 5.0], [5.1, 4.9]])
print(KMeans(k=2).fit(X).predict(X))  ## e.g. [0 0 1 1]
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- ✅Simple, ✅Fast ($O(mnkT)$, linear in $m$), ✅Scales (mini-batch variant).
- ✅Interpretable centroids, ✅Guaranteed to converge.

*Cons?*
- Requires $k$ up front.
- ⬆️Sensitivity to initialization → local minima → run multiple restarts (`n_init`).
- ⬆️Sensitivity to outliers ← the mean is not robust.
- Assumes **spherical, equal-size, equal-density** clusters → fails on elongated / nested / varying-density shapes.
- Requires feature scaling ← Euclidean distance.
- Hard assignment only → no notion of uncertainty (use GMM for that).
- Degrades in high dims ← curse of dimensionality.

*How to choose $k$?*
- **Elbow**: plot WCSS vs $k$, take the kink. Subjective — WCSS decreases monotonically by construction.
- **Silhouette**: maximize mean silhouette over $k$.
- **Gap statistic**: compare WCSS to that of uniform random reference data.
- **BIC/AIC** via a GMM fit.
- Domain knowledge beats all of these.

*Why does it fail on non-spherical clusters?*
- The assign step draws a **Voronoi** partition → every decision boundary is a straight hyperplane equidistant between two centroids.
- → Concentric rings, moons, and elongated clusters are impossible. Use DBSCAN or spectral clustering.

*Does it always converge to the same answer?*
- No. Different inits → different local minima. Fix: K-Means++ seeding, and `n_init` restarts keeping the lowest WCSS.

*Empty cluster?*
- Possible when a centroid wins no points. Fix: reinitialize it at the point furthest from its centroid, or drop it.

*K-Means vs GMM?*
- K-Means = GMM restricted to shared spherical covariance, equal weights, and hard assignments (the $\sigma\to0$ limit of EM).
- GMM adds soft assignments + elliptical, differently-sized clusters, at the price of more params and more compute.

*Why not use the median (K-Medians) or a medoid?*
- You can — but then the objective is L1/arbitrary distance, not squared Euclidean. Robustness⬆️, compute⬆️.
```

&nbsp;

#### K-Means++
- **What**: Seeding that picks initial centroids far apart, with probability ∝ squared distance to the nearest chosen centroid.
- **Why**: Bad local minima.
    - Random init can put 2 centroids inside the same true cluster → restarts may never escape.
    - → Makes expected WCSS $O(\log k)$-competitive with the optimum.
- **How**:
    1. Pick centroid 1 uniformly at random from the data.
    2. For each remaining sample, compute $D(\mathbf{x})$ = distance to the nearest chosen centroid.
    3. Pick the next centroid from the data with probability $\propto D(\mathbf{x})^2$.
    4. Repeat until $k$ centroids; then run standard K-Means.

```{note} Math
:class: dropdown
Sampling probability for the next centroid:

$$
P(\mathbf{x}_i)=\frac{D(\mathbf{x}_i)^2}{\sum_{l=1}^{m}D(\mathbf{x}_l)^2},\qquad D(\mathbf{x})=\min_{c\ \text{chosen}}||\mathbf{x}-\boldsymbol{\mu}_c||_2
$$

Guarantee: $\mathbb{E}[L_\text{K-Means++}]\leq8(\ln k+2)\cdot L^*$, where $L^*$ is the optimal WCSS.
```

```{attention} Q&A
:class: dropdown
*Why $D^2$ and not $D$?*
- $D^2$ matches the WCSS objective (also squared) → the sampling distribution is proportional to each point's current contribution to the loss.
- Linear $D$ under-weights far points; uniform gives no guarantee at all.

*Why sample instead of just taking the furthest point?*
- Deterministic furthest-point seeding always grabs the biggest OUTLIER. Sampling makes outliers likely-but-not-certain picks.

*Cost?*
- $k$ extra passes over the data → $O(mnk)$ seeding, negligible vs the main loop, and it usually reduces the #iterations needed.
- It is sklearn's default `init`.
```

&nbsp;

#### K-Medoids
- **What**: K-Means with **medoids** (actual data points) as centers.
- **Why**: Robustness + arbitrary distances.
    - The mean is not robust — one outlier drags a centroid off the cluster.
    - The mean needs Euclidean geometry — undefined for categorical data, edit distance, or a precomputed distance matrix.
    - A medoid is a real, inspectable sample → interpretable representative.
- **How**: Same alternating loop, but the update step picks the member minimizing the sum of distances to all other members of its cluster.

```{note} Math
:class: dropdown
Objective (any distance $d$, not necessarily squared Euclidean):

$$
L=\sum_{i=1}^{m}d(\mathbf{x}_i,\mathbf{x}_{\text{med}(c_i)}),\qquad\text{med}(c)=\arg\min_{j\in C_c}\sum_{i\in C_c}d(\mathbf{x}_i,\mathbf{x}_j)
$$

Complexity: $O(k(m-k)^2)$ per iteration for the classic PAM swap search → far heavier than K-Means.
```

```{attention} Q&A
:class: dropdown
*When K-Medoids over K-Means?*
- Outlier-heavy data, non-Euclidean/precomputed distances, or when the center must be a real sample.
- ❌ when $m$ is large — the quadratic cost is prohibitive.

*Which algorithm?*
- **PAM** (Partitioning Around Medoids): the classic solver — exhaustive single-swap **local** search, $O(k(m-k)^2)$ per iteration. Not a global optimum.
- CLARA / FasterPAM: sampling & swap-ordering variants for large $m$.
```

&nbsp;


### GMM
- **Name**: Gaussian Mixture Model
- **What**: Weighted sum of $k$ Gaussians; clusters = posterior over components.
- **Why**: Soft + elliptical.
    - K-Means gives **hard** assignments → boundary points get no confidence signal.
    - K-Means' spherical, equal-size assumption fails on elongated or unequal clusters.
    - Full covariance per component → arbitrary orientation, size, density + a real probability model.
- **How**: Fit by **EM** (Expectation-Maximization).
    1. **E-step**: given current params, compute each point's **responsibility** (posterior probability) for each component.
    2. **M-step**: re-estimate each component's weight, mean, and covariance as responsibility-weighted statistics.
    3. Repeat until the log-likelihood stabilizes.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\pi_c$: Mixing weight of component $c$, $\sum_c\pi_c=1$.
    - $\boldsymbol{\mu}_c\in\mathbb{R}^n$: Mean of component $c$.
    - $\Sigma_c\in\mathbb{R}^{n\times n}$: Covariance of component $c$.
- Misc:
    - $\gamma_{ic}$: Responsibility of component $c$ for sample $i$.
    - $N_c=\sum_i\gamma_{ic}$: Effective #samples in component $c$.

Model (density):

$$
P(\mathbf{x})=\sum_{c=1}^{k}\pi_cN(\mathbf{x}|\boldsymbol{\mu}_c,\Sigma_c)
$$

Training (EM), maximizing $\sum_i\log P(\mathbf{x}_i)$:
1. E-step:

    $$
    \gamma_{ic}=\frac{\pi_cN(\mathbf{x}_i|\boldsymbol{\mu}_c,\Sigma_c)}{\sum_{l=1}^{k}\pi_lN(\mathbf{x}_i|\boldsymbol{\mu}_l,\Sigma_l)}
    $$

2. M-step:

    $$
    \pi_c=\frac{N_c}{m},\quad\boldsymbol{\mu}_c=\frac{1}{N_c}\sum_{i=1}^m\gamma_{ic}\mathbf{x}_i,\quad\Sigma_c=\frac{1}{N_c}\sum_{i=1}^m\gamma_{ic}(\mathbf{x}_i-\boldsymbol{\mu}_c)(\mathbf{x}_i-\boldsymbol{\mu}_c)^T
    $$
```

````{important} Code
:class: dropdown
```python
import numpy as np

class GMM:
    def __init__(self, k=2, n_iter=100, reg=1e-6, seed=0):
        self.k, self.n_iter, self.reg, self.seed = k, n_iter, reg, seed
        self.pi, self.mu, self.sigma = None, None, None

    def _pdf(self, X):
        ## multivariate normal density for every (sample, component) pair -> (m, k)
        m, n = X.shape
        out = np.empty((m, self.k))
        for c in range(self.k):
            d = X - self.mu[c]
            inv = np.linalg.inv(self.sigma[c])
            out[:, c] = np.exp(-0.5 * np.einsum('ij,jk,ik->i', d, inv, d)) / \
                        np.sqrt((2 * np.pi) ** n * np.linalg.det(self.sigma[c]))
        return out

    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        m, n = X.shape
        self.pi = np.full(self.k, 1 / self.k)
        self.mu = X[rng.choice(m, self.k, replace=False)]
        ## reg on the diagonal stops a component from collapsing onto a single point
        self.sigma = np.array([np.cov(X.T) + self.reg * np.eye(n)] * self.k)
        for _ in range(self.n_iter):
            w = self.pi * self._pdf(X)                    ## E-step: joint
            gamma = w / w.sum(axis=1, keepdims=True)      ## E-step: normalize -> posterior
            N = gamma.sum(axis=0)
            self.pi = N / m                               ## M-step
            self.mu = (gamma.T @ X) / N[:, None]
            for c in range(self.k):
                d = X - self.mu[c]
                self.sigma[c] = (gamma[:, c] * d.T) @ d / N[c] + self.reg * np.eye(n)
        return self

    def predict_proba(self, X):
        w = self.pi * self._pdf(X)
        return w / w.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

## Example
rng = np.random.default_rng(0)
X = np.vstack([rng.normal(0, 1, (100, 2)), rng.normal(6, 1, (100, 2))])
print(np.unique(GMM(k=2).fit(X).predict(X)))  ## [0 1]
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- ✅Soft assignments → per-point uncertainty.
- ✅Elliptical clusters of differing size, shape, orientation, density.
- ✅A real generative density model → likelihood, BIC/AIC for choosing $k$, sampling, anomaly scoring.

*Cons?*
- ⬆️#params ($O(kn^2)$ with full covariance) → needs $m\gg n$; else restrict covariance (`spherical`/`diag`/`tied`).
- Local optima + init sensitivity → run K-Means first to initialize.
- ❌Guaranteed global optimum; likelihood is unbounded (see below).
- Still assumes Gaussian components → fails on genuinely non-Gaussian shapes (rings, moons).
- Slower than K-Means (matrix inversions per component per iteration).

*Why can the likelihood go to infinity?*
- A component centered exactly on one point with $\Sigma_c\to0$ gives that point infinite density → $\log L\to\infty$ → a **degenerate** solution.
- Fix: add $\epsilon I$ to each covariance (`reg_covar`), restrict the covariance type, or re-initialize collapsed components.

*Relationship to K-Means?*
- Fix $\Sigma_c=\sigma^2I$ (shared, spherical), fix $\pi_c=\frac{1}{k}$, and let $\sigma\to0$ → responsibilities become 0/1 → EM literally becomes Lloyd's algorithm.
- → K-Means is hard-EM on a constrained GMM.

*Does EM always improve the likelihood?*
- Yes, monotonically — but only to a local maximum or saddle point.

*How to choose $k$?*
- BIC $=-2\log\hat{L}+p\log m$ or AIC $=-2\log\hat{L}+2p$; lower is better.
    - $\hat{L}$: Maximized likelihood.
    - $p$: #free params; full covariance → $p=k\left(1+n+\frac{n(n+1)}{2}\right)-1$.
- BIC penalizes complexity harder → picks fewer components.
- Unlike K-Means' WCSS, these do NOT improve monotonically with $k$.
```

&nbsp;

### Hierarchical Clustering
- **What**: A tree (dendrogram) of nested clusters, built by repeated merging or splitting.
- **Why**: ❌Pick $k$ up front.
    - Cut the dendrogram at any height afterwards.
    - Reveals **nested** structure (sub-clusters within clusters), which flat methods destroy.
    - Works from an arbitrary distance/similarity matrix → no vector space required.
    - Deterministic — no random init.
- **How**:
    - **Agglomerative** (bottom-up, the practical default):
        1. Start with each sample as its own cluster.
        2. Merge the two closest clusters (per the chosen **linkage**).
        3. Repeat until one cluster remains → dendrogram.
        4. Cut at a chosen height / #clusters.
    - **Divisive** (top-down): start with one cluster, recursively split the "worst" one. Rarely used ← $O(2^m)$ split choices per level.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $A,B$: Two clusters.
    - $d(\cdot,\cdot)$: Point-level distance.

Linkage criteria $D(A,B)$:

$$\begin{align*}
\text{Single}&: \min_{a\in A,b\in B}d(a,b) \\
\text{Complete}&: \max_{a\in A,b\in B}d(a,b) \\
\text{Average (UPGMA)}&: \frac{1}{|A||B|}\sum_{a\in A}\sum_{b\in B}d(a,b) \\
\text{Centroid}&: ||\boldsymbol{\mu}_A-\boldsymbol{\mu}_B||_2 \\
\text{Ward}&: \frac{|A||B|}{|A|+|B|}||\boldsymbol{\mu}_A-\boldsymbol{\mu}_B||_2^2
\end{align*}$$

- Ward's $D(A,B)$ = the increase in total within-cluster variance caused by merging $A$ and $B$.
- Complexity: $O(m^3)$ naive, $O(m^2\log m)$ with a priority queue, $O(m^2)$ for single linkage (SLINK). Memory $O(m^2)$ ← full distance matrix.
```

```{dropdown} Table: Linkage Criteria
| Linkage | Merges on | Cluster shape | Weakness |
|:--|:--|:--|:--|
| Single | Closest pair | Arbitrary, elongated | **Chaining**: one bridge of noise merges two clusters |
| Complete | Furthest pair | Compact, roughly equal diameter | Breaks large clusters; outlier-sensitive |
| Average | Mean pairwise | Compromise | Not scale/monotonicity invariant |
| Centroid | Centroid distance | Compact | **Inversions** (a merge can lower the dendrogram height) |
| Ward | Min variance increase | Spherical, equal-size | Euclidean only; K-Means-like bias |
```

```{attention} Q&A
:class: dropdown
*Pros?*
- ❌Need to pick $k$ in advance; ✅Full nested hierarchy; ✅Deterministic.
- ✅Any distance metric (incl. precomputed / non-metric similarities).
- ✅Dendrogram is a genuinely useful visualization.

*Cons?*
- ❌Scales: $O(m^2)$ memory, $\ge O(m^2\log m)$ time → impractical past ~$10^4$ samples.
- **Greedy & irreversible** — a bad early merge can never be undone.
- Outlier-sensitive (especially single & complete linkage).
- The cut height is still a judgment call.

*Which linkage by default?*
- **Ward** for compact, roughly spherical clusters in Euclidean space (sklearn's default) — it directly minimizes the same variance objective as K-Means.
- **Single** only when clusters are genuinely elongated/chained AND the data is clean.
- **Average/Complete** with non-Euclidean metrics (e.g., cosine on text).

*What is chaining?*
- Single linkage merges on the single closest pair → a thin "bridge" of noise points between two well-separated clusters causes them to merge into one snake.

*How to read a dendrogram?*
- Y-axis = merge distance. A merge at a large height = the two clusters were far apart.
- Cut where there's a big **vertical gap** → that's the most "natural" $k$.

*Agglomerative vs Divisive?*
- Agglomerative: $m-1$ merges, each an $O(m^2)$-ish search → tractable, and best at finding small clusters.
- Divisive: exponentially many split choices → needs a heuristic (e.g., run 2-means at each node), but it makes better GLOBAL decisions early.
```

&nbsp;

### DBSCAN
- **Name**: Density-Based Spatial Clustering of Applications with Noise
- **What**: Densely packed points grouped together; the rest labeled noise.
- **Why**: Shape + outliers.
    - K-Means/GMM force EVERY point into a cluster → outliers absorbed, centroids distorted.
    - Both assume convex/elliptical shapes → useless on rings, moons, spirals.
    - #Clusters should be **discovered**, not specified.
- **How**:
    1. Label each point **core** if it has $\geq$ `minPts` neighbors within radius $\epsilon$ (itself included).
    2. Connect core points within $\epsilon$ of each other → each connected component is a cluster.
    3. Non-core points within $\epsilon$ of a core point → **border**, joined to that cluster.
    4. Everything else → **noise**.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\epsilon$: Neighborhood radius.
    - $\text{minPts}$: Min #points to be a core point.
- Misc:
    - $N_\epsilon(\mathbf{x})=\{\mathbf{x}_i:d(\mathbf{x},\mathbf{x}_i)\leq\epsilon\}$: $\epsilon$-neighborhood.

Point types:

$$
\begin{cases}
\text{Core} & |N_\epsilon(\mathbf{x})|\geq\text{minPts} \\
\text{Border} & |N_\epsilon(\mathbf{x})|<\text{minPts}\ \text{ and }\ \exists\text{ core }\mathbf{p}\text{ with }\mathbf{x}\in N_\epsilon(\mathbf{p}) \\
\text{Noise} & \text{otherwise}
\end{cases}
$$

- **Directly density-reachable**: $\mathbf{q}$ from core $\mathbf{p}$ if $\mathbf{q}\in N_\epsilon(\mathbf{p})$.
- **Density-reachable**: a chain of directly density-reachable core points.
- **Density-connected**: $\exists\mathbf{o}$ from which both $\mathbf{p}$ and $\mathbf{q}$ are density-reachable → a cluster is a maximal density-connected set.

Complexity: $O(m\log m)$ with a spatial index in low dims, $O(m^2)$ worst case. Memory $O(m)$ with on-demand neighbor queries ($O(m^2)$ if the full distance matrix is materialized, as in the snippet below).
```

````{important} Code
:class: dropdown
```python
import numpy as np

class DBSCAN:
    NOISE, UNVISITED = -1, 0

    def __init__(self, eps=0.5, min_pts=5):
        self.eps, self.min_pts = eps, min_pts
        self.labels = None

    def fit(self, X):
        m = len(X)
        ## O(m^2) distance matrix for clarity; real implementations use a spatial index
        D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
        self.labels = np.full(m, self.UNVISITED)
        cid = 0
        for i in range(m):
            if self.labels[i] != self.UNVISITED:
                continue
            nbrs = np.flatnonzero(D[i] <= self.eps)
            if len(nbrs) < self.min_pts:
                self.labels[i] = self.NOISE      ## may be reclaimed later as a border point
                continue
            cid += 1
            self.labels[i] = cid
            queue = list(nbrs)
            while queue:                          ## BFS over density-connected core points
                j = queue.pop()
                if self.labels[j] == self.NOISE:
                    self.labels[j] = cid          ## noise -> border of this cluster
                if self.labels[j] != self.UNVISITED:
                    continue
                self.labels[j] = cid
                j_nbrs = np.flatnonzero(D[j] <= self.eps)
                if len(j_nbrs) >= self.min_pts:   ## only CORE points expand the cluster
                    queue.extend(j_nbrs)
        return self

## Example: two concentric rings -> K-Means fails, DBSCAN doesn't
t = np.linspace(0, 2 * np.pi, 60)
X = np.vstack([np.c_[np.cos(t), np.sin(t)], np.c_[4 * np.cos(t), 4 * np.sin(t)]])
print(len(set(DBSCAN(eps=0.5, min_pts=3).fit(X).labels) - {-1}))  ## 2
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- ❌Need to pick $k$ — #clusters is discovered.
- ✅Arbitrary cluster shapes (non-convex, nested, elongated).
- ✅Explicit noise/outlier label → doubles as an anomaly detector.
- ✅Robust to outliers ← they never join a cluster.

*Cons?*
- ⬆️Sensitivity to $\epsilon$ & `minPts`.
- ❌**Varying density**: a single global $\epsilon$ can't serve a dense and a sparse cluster at once.
- ❌High dims ← distances concentrate → $\epsilon$ stops discriminating.
- Border points are assigned by processing order → mildly non-deterministic (core points and noise are deterministic).
- ❌Native `predict` for new points (it's transductive).

*How to set $\epsilon$ and `minPts`?*
- `minPts`: rule of thumb $\geq n+1$, commonly $2n$; larger → more noise, smoother clusters. Raise it on noisy data.
- $\epsilon$: plot each point's distance to its `minPts`-th nearest neighbor, sort descending → pick the **elbow**.

*DBSCAN vs K-Means?*
- Shape: arbitrary vs spherical. $k$: discovered vs given. Outliers: labeled vs absorbed. Density: assumed uniform-ish vs assumed spherical. Cost: $O(m\log m)$–$O(m^2)$ vs $O(mnkT)$.

*Fixing varying density?*
- **HDBSCAN**: runs DBSCAN over all $\epsilon$, builds a cluster hierarchy, and extracts the most **stable** clusters → main knobs are `min_cluster_size` & `min_samples` (the latter defaults to the former but is independently tunable).
- **OPTICS**: produces a reachability plot instead of a flat partition.
```

&nbsp;

### Spectral Clustering
- **What**: Clustering the eigenvectors of a graph Laplacian instead of the raw features.
- **Why**: Connectivity, not compactness.
    - K-Means separates by straight hyperplanes → hopeless on rings, moons, spirals.
    - → Reframe as **graph partitioning** (cut the fewest/weakest edges).
    - The Laplacian's bottom eigenvectors turn those groups into linearly separable blobs.
- **How**:
    1. Build a similarity graph → affinity matrix $W$ (usually a Gaussian/RBF kernel or a k-NN graph).
    2. Form the graph Laplacian $L$.
    3. Take the $k$ eigenvectors of the $k$ smallest eigenvalues → each row = a point's embedding.
    4. (Normalized version) row-normalize, then run K-Means on the embedded rows.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $W\in\mathbb{R}^{m\times m}$: Affinity/similarity matrix, $W_{ij}\geq0$, symmetric.
    - $D=\text{diag}(d_i)$, $d_i=\sum_jW_{ij}$: Degree matrix.

Affinity (Gaussian):

$$
W_{ij}=\exp\left(-\frac{||\mathbf{x}_i-\mathbf{x}_j||_2^2}{2\sigma^2}\right)\quad(i\neq j),\qquad W_{ii}=0
$$

Laplacians:

$$
L=D-W,\qquad L_\text{sym}=D^{-1/2}LD^{-1/2}=I-D^{-1/2}WD^{-1/2},\qquad L_\text{rw}=D^{-1}L
$$

Key property:

$$
\mathbf{v}^TL\mathbf{v}=\frac{1}{2}\sum_{i,j}W_{ij}(v_i-v_j)^2\geq0
$$
- → $L$ is PSD; smallest eigenvalue 0 with eigenvector $\mathbf{1}$.
- **#zero eigenvalues = #connected components** of the graph, and their eigenvectors are the component indicators.

Process:
1. Compute $L_\text{sym}$ (or $L$).
2. Stack the $k$ eigenvectors of the $k$ smallest eigenvalues as columns → $U\in\mathbb{R}^{m\times k}$.
3. Normalize each row of $U$ to unit length.
4. K-Means on the $m$ rows.

Complexity: $O(m^2)$ memory for $W$, $O(m^3)$ for a dense eigendecomposition.
```

```{attention} Q&A
:class: dropdown
*Why do the bottom eigenvectors reveal clusters?*
- $\mathbf{v}^TL\mathbf{v}=\frac{1}{2}\sum_{ij}W_{ij}(v_i-v_j)^2$ → minimizing it forces strongly-connected points to receive nearly the same value.
- With $c$ perfectly disconnected components, the 0-eigenspace is spanned by their indicator vectors. Weakly-connected clusters are a smooth perturbation of that → near-indicator eigenvectors.
- → It is a continuous relaxation of the NP-hard normalized-cut objective.

*Why still run K-Means at the end?*
- The eigenvectors are only a *relaxation* — real-valued, not 0/1 indicators, and rotation-invariant within the eigenspace. K-Means rounds the embedding back to a discrete partition, and in the embedding the clusters ARE compact blobs.

*Pros?*
- ✅Arbitrary, non-convex cluster shapes.
- ✅Only needs a similarity matrix → works on graphs & non-vector data.
- ✅Strong on small-to-medium data where K-Means visibly fails.

*Cons?*
- ❌Scales: $O(m^2)$ memory + $O(m^3)$ eigendecomposition (Nyström/sparse k-NN graphs help).
- ⬆️Sensitivity to $\sigma$ (or the k-NN graph's $k$) — the single biggest failure mode.
- Still needs $k$ (though the **eigengap** heuristic suggests it: pick $k$ maximizing $\lambda_{k+1}-\lambda_k$).
- ❌Native out-of-sample prediction.

*Which Laplacian?*
- Unnormalized $L$ → RatioCut relaxation; biased toward equal-SIZE clusters.
- $L_\text{sym}$/$L_\text{rw}$ → NormalizedCut relaxation; balances by total edge WEIGHT (degree). Usually preferred, especially with irregular degrees.
```

&nbsp;

## Dimensionality Reduction
- **What**: $n$ features compressed into $k\ll n$, preserving a chosen structure (variance, distances, independence, topology).

### PCA
- **Name**: Principal Component Analysis
- **What**: Projection onto the orthogonal directions of maximum variance.
- **Why**: Curse of dimensionality + redundancy.
    - Distance-based models degrade and #samples needed grows exponentially in $n$.
    - Correlated features are redundant → most signal lives in far fewer dims.
    - Low-variance directions are usually noise → denoising, visualization, decorrelation.
- **How**:
    1. **Center** the data (and standardize if features have different units).
    2. Compute the covariance matrix.
    3. Eigendecompose → eigenvectors = principal components, eigenvalues = variance explained.
    4. Keep the top-$k$ eigenvectors → project.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\tilde{X}$: Column-centered $X$.
    - $C\in\mathbb{R}^{n\times n}$: Covariance matrix.
    - $\mathbf{v}_j,\lambda_j$: $j$-th eigenvector/eigenvalue of $C$, sorted $\lambda_1\geq\cdots\geq\lambda_n$.
    - $V_k=[\mathbf{v}_1,\cdots,\mathbf{v}_k]$: Top-$k$ loading matrix.
Covariance:

$$
C=\frac{1}{m-1}\tilde{X}^T\tilde{X}
$$

Objective (two equivalent views):

$$
\max_{||\mathbf{v}||_2=1}\mathbf{v}^TC\mathbf{v}\quad\Leftrightarrow\quad\min_{V_k}||\tilde{X}-\tilde{X}V_kV_k^T||_F^2
$$
- Maximize projected variance $\Leftrightarrow$ Minimize squared reconstruction error.

Solution:

$$
C\mathbf{v}_j=\lambda_j\mathbf{v}_j,\qquad Z=\tilde{X}V_k,\qquad\hat{X}=ZV_k^T+\boldsymbol{\mu}
$$
- $Z\in\mathbb{R}^{m\times k}$: Scores (projected data).
- $\boldsymbol{\mu}$: Column means.

Explained variance ratio:

$$
\frac{\lambda_j}{\sum_{l=1}^{n}\lambda_l}
$$

Whitening: $Z_\text{white}=Z\Lambda_k^{-1/2}$ → unit variance in every component.

Complexity: $O(mn^2+n^3)$ via covariance eigendecomposition; $O(mn\min(m,n))$ via SVD of $\tilde{X}$ (preferred).
```

```{tip} Derivation
:class: dropdown
*Why are the principal components the eigenvectors of $C$?*
1. Variance of the projection onto unit $\mathbf{v}$: $\text{Var}[\tilde{X}\mathbf{v}]=\mathbf{v}^TC\mathbf{v}$.
2. Constrained maximization → Lagrangian $\mathcal{L}=\mathbf{v}^TC\mathbf{v}-\lambda(\mathbf{v}^T\mathbf{v}-1)$.
3. $\frac{\partial\mathcal{L}}{\partial\mathbf{v}}=2C\mathbf{v}-2\lambda\mathbf{v}=0\ \Rightarrow\ C\mathbf{v}=\lambda\mathbf{v}$ → eigenvector equation.
4. Substituting back: $\mathbf{v}^TC\mathbf{v}=\lambda$ → the eigenvalue IS the variance captured → take the largest.
5. $C$ symmetric PSD → eigenvectors orthogonal, eigenvalues $\geq0$ → components are uncorrelated.

*Why is centering mandatory?*
- Without it, $\frac{1}{m-1}X^TX$ is the second-moment matrix, not the covariance → PC1 collapses toward the direction of the MEAN vector instead of the direction of maximum spread.
```

````{important} Code
:class: dropdown
```python
import numpy as np

class PCA:
    def __init__(self, k=2):
        self.k = k
        self.mean, self.components, self.explained_variance_ratio = None, None, None

    def fit(self, X):
        self.mean = X.mean(axis=0)
        Xc = X - self.mean                       ## centering is NOT optional
        ## SVD of the centered data == eigendecomposition of its covariance, but stabler
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        var = S ** 2 / (len(X) - 1)              ## eigenvalues of the covariance matrix
        self.explained_variance_ratio = var[:self.k] / var.sum()
        self.components = Vt[:self.k]            ## (k, n) rows = principal directions
        return self

    def transform(self, X):
        return (X - self.mean) @ self.components.T

    def inverse_transform(self, Z):
        return Z @ self.components + self.mean

## Example: 2D data on a line -> 1 component explains ~everything
rng = np.random.default_rng(0)
t = rng.normal(size=200)
X = np.c_[t, 2 * t + 0.01 * rng.normal(size=200)]
p = PCA(k=1).fit(X)
print(np.round(p.explained_variance_ratio, 4))  ## [~1.0]
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- ✅Closed-form, deterministic, fast.
- ✅Decorrelates features → fixes multicollinearity.
- ✅Denoises + compresses; reconstruction is available (unlike t-SNE/UMAP).
- ✅Optimal linear reconstruction in the least-squares sense (Eckart-Young).

*Cons?*
- **Linear only** → can't unfold a curved manifold (swiss roll).
- ❌Interpretable components ← each PC is a dense mix of all original features.
- ⬆️Sensitivity to scaling & outliers (variance is not robust).
- **Unsupervised** → the max-variance direction may carry zero label information.
- Assumes large variance = important, which is false when noise is high-variance.

*Standardize or just center?*
- Center: always. Standardize: whenever features have different units/scales — otherwise the feature with the largest numeric range hijacks PC1.
- Standardizing = doing PCA on the CORRELATION matrix instead of the covariance matrix.

*How to choose $k$?*
- Cumulative explained variance threshold (e.g., 95%).
- Scree plot elbow.
- Kaiser criterion ($\lambda_j>1$ on standardized data) — crude.
- Or: whatever maximizes downstream CV performance.

*PCA vs LDA?*
- PCA: unsupervised, maximizes variance, up to $n$ components.
- LDA: supervised, maximizes class separability, at most $K-1$ components.

*Is PCA the same as SVD?*
- PCA = SVD of the CENTERED matrix. $\tilde{X}=U\Sigma V^T$ → $V$'s columns are the PCs and $\lambda_j=\frac{\sigma_j^2}{m-1}$.
- Doing SVD without centering (e.g., sklearn's `TruncatedSVD` on sparse text) is NOT PCA.

*Can I run PCA on categorical / one-hot data?*
- Poorly — variance is not meaningful for indicators. Use MCA, or feed the categoricals to a model that handles them natively.

*Does PCA help every model?*
- No. Tree ensembles usually get WORSE: PCA replaces interpretable axis-aligned features with dense rotations, exactly the thing trees split on.
```

&nbsp;

### SVD
- **Name**: Singular Value Decomposition
- **What**: Any real matrix factorized into (orthonormal basis) × (nonneg. diagonal scaling) × (orthonormal basis).
- **Why**: Generality + optimal truncation.
    - Eigendecomposition needs a square (and, for orthogonality, symmetric) matrix; SVD works for ANY $m\times n$.
    - Truncating gives the provably best low-rank approximation → the engine under PCA, LSA, matrix completion, pseudo-inverses.
- **How**: Compute directly (Golub-Kahan / randomized SVD); keep the top-$k$ singular triplets for a rank-$k$ approximation.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $U\in\mathbb{R}^{m\times m}$: Left singular vectors (orthonormal).
    - $\Sigma\in\mathbb{R}^{m\times n}$: Diagonal, $\sigma_1\geq\sigma_2\geq\cdots\geq0$.
    - $V\in\mathbb{R}^{n\times n}$: Right singular vectors (orthonormal).

Decomposition:

$$
X=U\Sigma V^T=\sum_{j=1}^{r}\sigma_j\mathbf{u}_j\mathbf{v}_j^T,\qquad r=\text{rank}(X)
$$

Relation to eigendecomposition:

$$
X^TX=V\Sigma^T\Sigma V^T,\qquad XX^T=U\Sigma\Sigma^TU^T
$$
- → $\sigma_j^2$ = eigenvalues of $X^TX$; $V$ = its eigenvectors.

**Eckart-Young-Mirsky**: the truncation $X_k=\sum_{j=1}^{k}\sigma_j\mathbf{u}_j\mathbf{v}_j^T$ is the best rank-$k$ approximation:

$$
X_k=\arg\min_{\text{rank}(B)\leq k}||X-B||_F,\qquad||X-X_k||_F=\sqrt{\sum_{j>k}\sigma_j^2}
$$
- Also optimal in the spectral norm, where $||X-X_k||_2=\sigma_{k+1}$.

Moore-Penrose pseudo-inverse: $X^+=V\Sigma^+U^T$ ($\Sigma^+$ inverts nonzero singular values).
```

```{attention} Q&A
:class: dropdown
*SVD vs PCA?*
- PCA = SVD applied to the **centered** matrix, reinterpreted statistically (variance, covariance).
- SVD is pure linear algebra: no centering, no statistical assumption. Truncated SVD without centering is what LSA uses on sparse term-document matrices (centering would destroy sparsity).

*What is LSA/LSI?*
- Truncated SVD on a term-document (TF-IDF) matrix → dense "topic" dims that merge synonyms and split polysemy. Components can be negative → less interpretable than NMF/LDA.

*Why is SVD numerically preferred over forming $X^TX$?*
- Forming $X^TX$ SQUARES the condition number → catastrophic precision loss on ill-conditioned data. SVD works on $X$ directly.

*Cost?*
- Full: $O(mn\min(m,n))$. Truncated/randomized: roughly $O(mnk)$ — the only option for large sparse matrices.
```

&nbsp;

### Factor Analysis
- **What**: Observed features = linear combinations of a few shared latent factors + **per-feature** independent noise.
- **Why**: PCA has no noise model.
    - It treats ALL variance as signal → a single noisy feature can dominate a component.
    - → Separate **common** variance (the factors) from **unique** variance (feature-specific noise).
- **How**: Fit $W$ and the diagonal noise covariance $\Psi$ by maximum likelihood (EM); latent factors inferred as posteriors.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $W\in\mathbb{R}^{n\times k}$: Factor loading matrix.
    - $\Psi=\text{diag}(\psi_1,\cdots,\psi_n)$: Unique-noise variances.
- Misc:
    - $\mathbf{z}\in\mathbb{R}^k$: Latent factors.

Generative model:

$$
\mathbf{x}=W\mathbf{z}+\boldsymbol{\mu}+\boldsymbol{\epsilon},\qquad\mathbf{z}\sim N(0,I),\quad\boldsymbol{\epsilon}\sim N(0,\Psi)
$$

Implied covariance:

$$
\text{Cov}[\mathbf{x}]=WW^T+\Psi
$$
- PCA is the special case $\Psi=\sigma^2I$ (isotropic noise) — that's **Probabilistic PCA**.
```

```{attention} Q&A
:class: dropdown
*FA vs PCA in one line?*
- PCA explains **total** variance with orthogonal directions; FA explains **shared** variance with a latent model and throws the per-feature noise away.

*Why is FA scale-invariant and PCA isn't?*
- $\Psi$ is free per feature → rescaling a feature is absorbed by $\psi_j$ and the corresponding row of $W$. PCA has no such slack → it needs manual standardization.

*Rotation problem?*
- $W$ is only identified up to an orthogonal rotation ($WR$ with $RR^T=I$ gives the same $WW^T$) → the "factors" are not unique.
- → Varimax/oblimin rotations are applied to make the loadings interpretable. Unlike PCA, there is no canonical ordering of factors.
```

&nbsp;

### ICA
- **Name**: Independent Component Analysis
- **What**: Observed signals decomposed into a linear mixture of statistically **independent**, non-Gaussian sources.
- **Why**: Uncorrelated ≠ independent.
    - PCA removes only **correlation** (2nd-order) → components can still be strongly dependent.
    - PCA ranks by variance, which says nothing about which physical source generated the data.
    - → Blind source separation (cocktail party, EEG/fMRI artifact removal) needs full independence.
- **How**:
    1. Center & **whiten** the data (PCA + scaling → uncorrelated, unit variance).
    2. Search for a rotation of the whitened data maximizing **non-Gaussianity** of each component (kurtosis or negentropy).
    3. Deflate/orthogonalize and repeat for $k$ components (FastICA).

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\mathbf{s}\in\mathbb{R}^k$: Independent sources.
    - $A\in\mathbb{R}^{n\times k}$: Mixing matrix.
    - $W$: Unmixing matrix.

Model:

$$
\mathbf{x}=A\mathbf{s}\quad\Rightarrow\quad\hat{\mathbf{s}}=W\mathbf{x}
$$

Independence assumption:

$$
P(\mathbf{s})=\prod_{j=1}^{k}P(s_j)
$$

Contrast function (negentropy approximation, FastICA):

$$
J(\mathbf{v})\propto\left[\mathbb{E}[G(\mathbf{v}^T\mathbf{x})]-\mathbb{E}[G(\nu)]\right]^2,\qquad\nu\sim N(0,1)
$$
- $G$: Non-quadratic contrast, e.g., $G(u)=\frac{1}{a}\log\cosh(au)$ or $G(u)=-e^{-u^2/2}$.
- Negentropy $\geq0$, and $=0$ **iff** the variable is Gaussian → maximizing it = maximizing non-Gaussianity.
```

```{attention} Q&A
:class: dropdown
*Why must the sources be non-Gaussian?*
- A rotation of i.i.d. Gaussians is again i.i.d. Gaussian with the same distribution → the mixing matrix is unidentifiable.
- → At most ONE source may be Gaussian. Non-Gaussianity is literally the signal ICA extracts.

*Why does maximizing non-Gaussianity find the sources?*
- CLT: a mixture of independent sources is MORE Gaussian than any single source.
- → The most non-Gaussian projection is (approximately) an unmixed single source.

*What ambiguities remain?*
- **Scale** (and sign): $A\mathbf{s}=(A c^{-1})(c\mathbf{s})$ → magnitude of each source is arbitrary; usually fixed to unit variance.
- **Permutation**: source order is arbitrary → there is no "first" independent component (unlike PCA's variance ordering).

*ICA vs PCA?*
- PCA: 2nd-order only, orthogonal components, ordered by variance, no distributional assumption, closed form.
- ICA: higher-order statistics, non-orthogonal components, unordered, requires non-Gaussianity, iterative.
- PCA (whitening) is a standard PREPROCESSING step for ICA, not a competitor.

*Why whiten first?*
- After whitening, the remaining unmixing matrix is constrained to be **orthogonal** → the search drops from $k^2$ free params to $\frac{k(k-1)}{2}$ → far faster and better conditioned.
```

&nbsp;

### NMF
- **Name**: Non-negative Matrix Factorization
- **What**: A non-negative matrix factorized into two non-negative low-rank factors.
- **Why**: Interpretability.
    - PCA/SVD components contain negatives → "−0.3 of topic 2" is meaningless for counts, pixels, spectra.
    - → Non-negativity forbids cancellation → purely additive, parts-based representation.
- **How**: Minimize reconstruction error subject to $W,H\geq0$, using multiplicative updates (or projected gradient / ALS) that preserve non-negativity by construction.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $X\in\mathbb{R}_{\geq0}^{m\times n}$: Non-negative data.
    - $W\in\mathbb{R}_{\geq0}^{m\times k}$: Coefficients.
    - $H\in\mathbb{R}_{\geq0}^{k\times n}$: Basis ("parts"/topics).

Objective:

$$
\min_{W,H\geq0}||X-WH||_F^2\qquad\text{or}\qquad\min_{W,H\geq0}\sum_{ij}\left(X_{ij}\log\frac{X_{ij}}{(WH)_{ij}}-X_{ij}+(WH)_{ij}\right)
$$
- Left: Frobenius (Gaussian noise). Right: generalized KL divergence (count/Poisson data, better for text).

Multiplicative updates (Frobenius), applied elementwise ($\odot$ = elementwise product, fractions elementwise):

$$
H\leftarrow H\odot\frac{W^TX}{W^TWH},\qquad W\leftarrow W\odot\frac{XH^T}{WHH^T}
$$
- Multiplying by non-negative ratios → $W,H$ stay $\geq0$ automatically, no projection step needed.
```

```{attention} Q&A
:class: dropdown
*Pros?*
- ✅Interpretable, parts-based, additive components.
- ✅Tends toward sparse factors ← non-negativity forbids cancellation (but sparsity is NOT guaranteed; add an explicit L1 penalty if you need it).
- ✅Matches the domain when the data genuinely can't be negative (counts, intensities, spectra).

*Cons?*
- **Non-convex** in $(W,H)$ jointly → local minima, init-dependent, non-unique solutions.
- Requires $X\geq0$.
- $k$ must be chosen in advance; no nested structure (the rank-$k$ and rank-$(k{+}1)$ solutions are unrelated, unlike SVD).
- ❌Orthogonality → components can overlap.
- No closed form; slower than SVD.

*NMF vs PCA?*
- PCA: signed, orthogonal, unique, ordered, closed form, best linear reconstruction.
- NMF: non-negative, non-orthogonal, non-unique, unordered, iterative, interpretable.

*NMF vs LDA for topics?*
- NMF: linear-algebraic, deterministic given init, fast, works well on TF-IDF.
- LDA: probabilistic generative model with Dirichlet priors → principled uncertainty & better on short documents, but slower.

*Why are multiplicative updates convex-safe?*
- They're not globally convex-safe — but each update is a gradient step with an adaptive per-element step size chosen so the factor stays non-negative, and it provably does not increase the objective.
```

&nbsp;

### t-SNE
- **Name**: t-distributed Stochastic Neighbor Embedding
- **What**: 2–3D embedding matching a neighbor-pair probability distribution in high-dim space with one in the low-dim map.
- **Why**: Crowding.
    - PCA is linear → cannot unfold curved manifolds, and often collapses distinct clusters in 2D.
    - Volume at radius $r$ grows as $r^n$ → a high-dim neighborhood doesn't fit in 2D → moderately-distant points get squashed onto near ones.
    - → A heavy-tailed Student-t in the map allocates room to moderate distances → clusters separate.
- **How**:
    1. High-dim: Gaussian conditional similarities $p_{j|i}$, each point's Gaussian width tuned so its distribution has the requested **perplexity**; symmetrize into $p_{ij}$.
    2. Low-dim: similarities $q_{ij}$ from a Student-t with 1 degree of freedom.
    3. Minimize $\text{KL}(P||Q)$ over the map coordinates by gradient descent.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\mathbf{y}_i\in\mathbb{R}^2$: Low-dim map point.
    - $\sigma_i$: Per-point Gaussian bandwidth.
- Hyperparams:
    - Perplexity: effective #neighbors; the authors report typical values **5–50**.

High-dim similarities:

$$
p_{j|i}=\frac{\exp(-||\mathbf{x}_i-\mathbf{x}_j||_2^2/2\sigma_i^2)}{\sum_{l\neq i}\exp(-||\mathbf{x}_i-\mathbf{x}_l||_2^2/2\sigma_i^2)},\qquad p_{ij}=\frac{p_{j|i}+p_{i|j}}{2m}
$$
- $\sigma_i$ is solved per point by binary search so that $2^{H(P_i)}$ = the target perplexity, where $H(P_i)=-\sum_jp_{j|i}\log_2p_{j|i}$.

Low-dim similarities (Student-t, 1 dof):

$$
q_{ij}=\frac{(1+||\mathbf{y}_i-\mathbf{y}_j||_2^2)^{-1}}{\sum_{l\neq p}(1+||\mathbf{y}_l-\mathbf{y}_p||_2^2)^{-1}}
$$

Objective & gradient:

$$
\text{KL}(P||Q)=\sum_{i\neq j}p_{ij}\log\frac{p_{ij}}{q_{ij}},\qquad\frac{\partial\text{KL}}{\partial\mathbf{y}_i}=4\sum_j(p_{ij}-q_{ij})q_{ij}Z(\mathbf{y}_i-\mathbf{y}_j)
$$
- $Z=\sum_{l\neq p}(1+||\mathbf{y}_l-\mathbf{y}_p||_2^2)^{-1}$: Normalizer.

Complexity: $O(m^2)$ naive; $O(m\log m)$ with Barnes-Hut.
```

```{attention} Q&A
:class: dropdown
*Why the Student-t in low dims but a Gaussian in high dims?*
- Its heavy tail means a moderate $q_{ij}$ requires a MUCH larger low-dim distance than the Gaussian would → dissimilar points are pushed far apart → directly fixes the crowding problem.
- Bonus: $(1+d^2)^{-1}$ needs no exponential → cheaper gradients.

*Why KL(P||Q) and not KL(Q||P)?*
- $\text{KL}(P||Q)$ heavily penalizes small $q_{ij}$ where $p_{ij}$ is large → **nearby points must stay nearby**.
- It barely penalizes large $q_{ij}$ where $p_{ij}$ is small → distant points may be placed anywhere.
- → t-SNE preserves LOCAL structure and gives no guarantee about global structure. This is a design choice, not a bug.

*What must you NOT read off a t-SNE plot?*
- **Distances between clusters** — meaningless.
- **Cluster sizes/densities** — t-SNE equalizes densities by construction (per-point $\sigma_i$).
- **Apparent clusters at low perplexity** — random noise splits into convincing-looking blobs. Always inspect several perplexities.

*Cons?*
- Non-convex → different seeds give different maps.
- ❌`transform` for new points (non-parametric) → must re-run.
- $O(m^2)$ memory naive; slow on large $m$.
- Perplexity-sensitive.
- Practically limited to 2–3 output dims (the heavy tail is tuned for that) → a visualization tool, NOT a general preprocessing step.

*Standard practice?*
- Run PCA to ~50 dims first → denoises and massively speeds up the neighbor computation.
```

&nbsp;

### UMAP
- **Name**: Uniform Manifold Approximation and Projection
- **What**: Weighted k-NN graph read as a fuzzy topological structure → low-dim layout matching it.
- **Why**: t-SNE's limits.
    - Slow ($O(m\log m)$ at best, large constant), scales poorly past ~$10^5$ points.
    - Discards global structure entirely, and cannot embed new points without refitting.
    - → Topological formulation + negative sampling: better global structure, faster, any output dim.
- **How**:
    1. Build a k-NN graph; convert distances to fuzzy memberships with a per-point local scale (each point's nearest neighbor is at fuzzy distance 0 → "locally uniform" density).
    2. Symmetrize into a single fuzzy graph.
    3. Initialize the layout spectrally, then minimize the fuzzy-set cross-entropy with attractive forces on edges and repulsive forces via **negative sampling** (SGD).

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - `n_neighbors`: Local neighborhood size (default 15) → local vs global structure tradeoff.
    - `min_dist`: Min packing distance in the embedding (default 0.1) → mostly aesthetic; small → tight clumps.
- Misc:
    - $\rho_i$: Distance from $\mathbf{x}_i$ to its NEAREST neighbor.
    - $\sigma_i$: Local bandwidth.

High-dim fuzzy membership:

$$
w(\mathbf{x}_i,\mathbf{x}_j)=\exp\left(-\frac{\max(0,d(\mathbf{x}_i,\mathbf{x}_j)-\rho_i)}{\sigma_i}\right)
$$
- Subtracting $\rho_i$ guarantees each point has $\geq1$ neighbor at membership 1 → no point is ever isolated.
- $\sigma_i$ solved so $\sum_jw(\mathbf{x}_i,\mathbf{x}_j)=\log_2(\texttt{n\_neighbors})$.

Symmetrization (fuzzy union):

$$
W_{ij}=w_{ij}+w_{ji}-w_{ij}w_{ji}
$$

Objective (fuzzy-set cross-entropy):

$$
L=\sum_{ij}\left[W_{ij}\log\frac{W_{ij}}{V_{ij}}+(1-W_{ij})\log\frac{1-W_{ij}}{1-V_{ij}}\right]
$$
- $V_{ij}=(1+a||\mathbf{y}_i-\mathbf{y}_j||_2^{2b})^{-1}$: Low-dim membership; $a,b$ fit from `min_dist`.
- The SECOND term is an **explicit** repulsion over non-edges, applied via negative sampling.
    - ⚠️ t-SNE is not repulsion-free — its normalized $q_{ij}$ produces repulsion too. The difference is explicit-and-sampled vs normalization-induced.
```

```{dropdown} Table: t-SNE vs UMAP vs PCA
| | PCA | t-SNE | UMAP |
|:--|:--|:--|:--|
| Type | Linear | Nonlinear, probabilistic | Nonlinear, topological |
| Preserves | Global variance | Local neighborhoods | Local + some global |
| Speed | Fastest | Slowest | Fast |
| Deterministic | ✅ | ❌ | ❌ (seedable) |
| `transform` new data | ✅ | ❌ | ✅ |
| Output dims | Any | 2–3 | Any |
| Inverse transform | ✅ | ❌ | Approximate |
| Main knob | $k$ | perplexity | `n_neighbors`, `min_dist` |
```

```{attention} Q&A
:class: dropdown
*UMAP vs t-SNE — what actually differs?*
- Cross-entropy with an explicit non-edge (repulsive) term vs KL, whose repulsion comes only from the global normalizer → UMAP keeps more global structure.
- k-NN graph + negative sampling SGD vs dense pairwise gradients → UMAP is much faster.
- UMAP supports `transform` on unseen points and any output dimension.

*What do the two knobs do?*
- `n_neighbors`⬆️ → each point sees a wider neighborhood → more global structure, fewer fine-grained clusters. ⬇️ → very local, more fragmented.
- `min_dist`⬆️ → points spread out, broader topology visible. ⬇️ → tight, visually crisp clumps. It affects the PICTURE, not the underlying graph.

*Is UMAP's global structure trustworthy?*
- More than t-SNE's, but still not metric. Inter-cluster distances remain qualitative; the embedding also depends on initialization.

*Same warnings as t-SNE?*
- Yes: don't read cluster sizes or exact inter-cluster distances; always vary hyperparameters before believing a structure.
```

&nbsp;

## Topic Modeling

### LDA
- **Name**: Latent Dirichlet Allocation
- **What**: Documents as mixtures over topics; topics as distributions over words.
- **Why**: Mixed membership + a generative story.
    - Hard clustering forces one topic per document — real documents mix topics.
    - LSA yields signed, uninterpretable components & no generative likelihood → no principled uncertainty (it can still fold new docs into its SVD space).
    - Dirichlet priors give sparse, normalized mixtures → "70% sports, 30% politics".
- **How**:
    1. Assume the generative story: draw a topic mixture per document, then per word draw a topic and then a word.
    2. Invert it: infer the posterior over topic assignments (variational EM or collapsed Gibbs sampling).
    3. Read off topic-word and document-topic distributions.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\theta_d\in\Delta^{k-1}$: Topic distribution of document $d$.
    - $\beta_z\in\Delta^{V-1}$: Word distribution of topic $z$.
    - $V$: Vocab size.
- Hyperparams:
    - $\alpha$: Dirichlet prior on $\theta$ (⬇️ → documents concentrate on fewer topics).
    - $\eta$: Dirichlet prior on $\beta$ (⬇️ → topics concentrate on fewer words).

Generative process:
1. For each topic $z=1,\cdots,k$: $\beta_z\sim\text{Dir}(\eta)$.
2. For each document $d$:
    1. $\theta_d\sim\text{Dir}(\alpha)$.
    2. For each word position: $z_{dj}\sim\text{Mult}(\theta_d)$, then $w_{dj}\sim\text{Mult}(\beta_{z_{dj}})$.

Joint:

$$
P(\mathbf{w},\mathbf{z},\theta,\beta|\alpha,\eta)=\prod_z P(\beta_z|\eta)\prod_d P(\theta_d|\alpha)\prod_j P(z_{dj}|\theta_d)P(w_{dj}|\beta_{z_{dj}})
$$

Inference: exact posterior intractable → variational EM or collapsed Gibbs sampling.
```

```{attention} Q&A
:class: dropdown
*Why a Dirichlet prior?*
- Conjugate to the multinomial → the **conditionals** ($\theta_d$ and $\beta_z$ given topic assignments) stay Dirichlet → clean Gibbs / variational updates.
    - ⚠️ The full posterior given only the words stays intractable ← topic assignments couple everything.
- $\alpha<1$ makes the prior favor **sparse** mixtures (mass near the simplex corners) → each document uses few topics, which is both realistic and interpretable.

*Assumptions?*
- **Bag of words** — word order ignored, documents exchangeable.
- #topics $k$ fixed in advance.
- Topics are static (no drift over time).

*How to choose $k$?*
- Held-out perplexity (but it correlates poorly with human judgment) or **topic coherence** (e.g., $C_v$, NPMI), which correlates better.

*Cons?*
- Slow inference on large corpora; sensitive to $\alpha,\eta$ and to preprocessing (stopwords, rare-word pruning).
- Unstable across runs (different local optima / sampling chains).
- Weak on very short texts (tweets) ← too little co-occurrence evidence per document.
- Topics are unlabeled and require human interpretation.

*LDA vs NMF?*
- LDA: probabilistic, priors, uncertainty, better on raw counts & short docs, slower.
- NMF: deterministic given init, fast, often crisper topics on TF-IDF for medium corpora, no generative model.

*Name clash*: LDA also = **Linear Discriminant Analysis**, a supervised classifier. Unrelated.
```

&nbsp;

## Anomaly Detection
- **What**: The rare & different flagged w/o labels. Three families: **isolation-based** (Isolation Forest), **density-based** (LOF), **model-based** (One-Class SVM).

### Isolation Forest
- **What**: Anomalies scored by how few random splits it takes to isolate a point.
- **Why**: Profiling normality is expensive.
    - Density/distance methods build a profile of normal data first → $O(m^2)$ + degradation in high dims.
    - Anomalies are *few and different* → they land in sparse regions → a random split separates them immediately.
    - → Isolate directly: linear time, constant memory, ❌distance computation.
- **How**:
    1. Build many **iTrees**: on a small subsample, repeatedly pick a random feature and a random split value between its min and max, until each point is isolated (or a height limit is hit).
    2. Average each point's path length across trees.
    3. Short average path → anomaly.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $h(\mathbf{x})$: Path length of $\mathbf{x}$ in one iTree.
    - $\psi$: Subsample size (default 256).
    - $H(i)\approx\ln i+0.5772156649$ (Euler-Mascheroni): Harmonic number.

Expected path length of an unsuccessful BST search (the normalization constant):

$$
c(\psi)=2H(\psi-1)-\frac{2(\psi-1)}{\psi}
$$

Anomaly score:

$$
s(\mathbf{x},\psi)=2^{-\frac{\mathbb{E}[h(\mathbf{x})]}{c(\psi)}}\in(0,1)
$$
- $s\to1$ ($\mathbb{E}[h]\to0$): definitely an anomaly.
- $s\ll0.5$ ($\mathbb{E}[h]\to\psi-1$): safely normal.
- $s\approx0.5$ for all points: no distinct anomalies in the data.

Complexity: $O(t\psi\log\psi)$ training, $O(t\log\psi)$ per query, memory $O(t\psi)$ — independent of $m$.
```

```{attention} Q&A
:class: dropdown
*Why subsample (default 256) instead of using all data?*
- **Swamping** (normals near a dense anomaly cluster get flagged) and **masking** (a dense anomaly cluster hides itself) both worsen with more data.
- A small subsample makes anomalies stand out more, AND makes training essentially $O(1)$ in $m$.

*Pros?*
- ✅Linear time, low constant memory, trivially parallel.
- ✅No distance/density computation → survives higher dims better than LOF/kNN.
- ❌Feature scaling needed ← axis-aligned random splits.

*Cons?*
- Axis-aligned splits only → blind to anomalies that are only anomalous in a rotated/oblique direction (fix: Extended Isolation Forest).
- Struggles with **local** anomalies (a point normal globally but abnormal within its own dense cluster) → LOF is better there.
- `contamination` (assumed anomaly fraction) must be set to convert scores into labels.

*Why $2^{-\mathbb{E}[h]/c(\psi)}$ and not just $\mathbb{E}[h]$?*
- Path length depends on $\psi$ → the normalization makes scores comparable across sample sizes, and maps them into $(0,1)$ with a fixed interpretation.
```

&nbsp;

### LOF
- **Name**: Local Outlier Factor
- **What**: A point scored by the ratio of its neighbors' local densities to its own.
- **Why**: Local vs global density.
    - A single global density threshold either flags a whole sparse cluster, or misses local outliers next to a dense one.
    - → A **relative** density measure is comparable across regions of different density.
- **How**:
    1. For each point, find its $k$ nearest neighbors.
    2. Compute its local reachability density (inverse mean reachability distance to those neighbors).
    3. LOF = mean neighbor density ÷ own density.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $N_k(\mathbf{x})$: $k$-nearest-neighbor set.
    - $d_k(\mathbf{x})$: Distance to the $k$-th nearest neighbor.

Reachability distance (smoothing → stabilizes the density estimate):

$$
\text{reach-dist}_k(\mathbf{x},\mathbf{o})=\max\left(d_k(\mathbf{o}),\ d(\mathbf{x},\mathbf{o})\right)
$$

Local reachability density:

$$
\text{lrd}_k(\mathbf{x})=\left(\frac{1}{|N_k(\mathbf{x})|}\sum_{\mathbf{o}\in N_k(\mathbf{x})}\text{reach-dist}_k(\mathbf{x},\mathbf{o})\right)^{-1}
$$

LOF:

$$
\text{LOF}_k(\mathbf{x})=\frac{1}{|N_k(\mathbf{x})|}\sum_{\mathbf{o}\in N_k(\mathbf{x})}\frac{\text{lrd}_k(\mathbf{o})}{\text{lrd}_k(\mathbf{x})}
$$
- $\approx1$: same density as neighbors → inlier.
- $\gg1$: much sparser than neighbors → outlier.
- $<1$: denser than its neighbors → deep inside a cluster.

Complexity: $O(m^2)$ naive, $O(m\log m)$ with a spatial index in low dims.
```

```{attention} Q&A
:class: dropdown
*Why "local"?*
- The score is a RATIO against the point's own neighborhood → a sparse-but-legitimate cluster gets LOF ≈ 1, while a point slightly off a dense cluster gets a high LOF. A global density threshold can do neither.

*Why the `max` in reachability distance?*
- It floors the distance at $d_k(\mathbf{o})$ → removes statistical fluctuation for points very close to $\mathbf{o}$ → much more stable density estimates. Note it is deliberately **asymmetric**.

*Cons?*
- $O(m^2)$ without an index; ❌ in high dims (distances concentrate).
- ⬆️Sensitivity to $k$: too small → noise-driven; too large → loses locality.
- Scores have no absolute meaning → the 1.0 boundary is fuzzy and dataset-dependent.
- sklearn's `LocalOutlierFactor` is transductive by default (`novelty=False`) → no `predict` on new data unless `novelty=True`.
```

&nbsp;

### One-Class SVM
- **What**: Boundary separating the bulk of the data from the origin in kernel feature space.
- **Why**: No anomaly examples.
    - Clean "normal" data + zero labeled anomalies → a two-class classifier is impossible.
    - Estimating the full density is far harder than estimating just its **support**.
- **How**: Solve an SVM-like QP maximizing the margin between the data and the origin in feature space; $\nu$ sets the fraction allowed outside.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\nu\in(0,1]$: Upper bound on the fraction of outliers, lower bound on the fraction of SVs.
- Misc:
    - $\phi$: Kernel feature map.
    - $\rho$: Offset.
    - $\xi_i$: Slack.

Objective:

$$
\min_{\mathbf{w},\boldsymbol{\xi},\rho}\ \frac{1}{2}||\mathbf{w}||_2^2+\frac{1}{\nu m}\sum_{i=1}^{m}\xi_i-\rho\quad\text{s.t.}\quad\mathbf{w}^T\phi(\mathbf{x}_i)\geq\rho-\xi_i,\ \xi_i\geq0
$$

Decision:

$$
f(\mathbf{x})=\text{sign}\left(\mathbf{w}^T\phi(\mathbf{x})-\rho\right)
$$
- $f=+1$ → normal (inside the support). $f=-1$ → anomaly.
```

```{attention} Q&A
:class: dropdown
*What does $\nu$ mean exactly?*
- It simultaneously upper-bounds the fraction of training points classified as outliers and lower-bounds the fraction of support vectors → a single, interpretable knob (unlike SVM's $C$).

*Cons?*
- $O(m^2)$–$O(m^3)$ → doesn't scale (use `SGDOneClassSVM` or Isolation Forest instead).
- ⬆️Sensitivity to $\nu$ and the RBF $\gamma$; requires feature scaling.
- Assumes the training data is (mostly) clean — contamination directly distorts the boundary.

*Which anomaly detector should I reach for?*
- Large $m$, high $n$, global anomalies → **Isolation Forest** (default choice).
- Varying local density, moderate $m$ → **LOF**.
- Small, clean, well-scaled data with a known clean training set → **One-Class SVM**.
- Elliptical Gaussian-ish data → Elliptic Envelope (robust covariance).
```

&nbsp;

## Association Rule Learning
- **What**: Items that co-occur ("market basket analysis"): if a transaction contains $A$, it likely contains $B$.

### Metrics
- **What**: Support, confidence & lift — the three numbers deciding whether a rule $A\Rightarrow B$ is worth anything.
- **Why**: Co-occurrence alone is misleading.
    - A rule about two ubiquitous items looks strong but carries zero information.
    - → Each metric kills a different failure mode.
- **How**: Support prunes rare rules, confidence measures conditional reliability, lift corrects confidence for the consequent's base rate.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $A,B$: Disjoint itemsets.
    - $\text{supp}(A)$: Fraction of transactions containing $A$.

$$\begin{align*}
\text{support}(A\Rightarrow B)&=P(A\cap B) \\
\text{confidence}(A\Rightarrow B)&=P(B|A)=\frac{\text{supp}(A\cap B)}{\text{supp}(A)} \\
\text{lift}(A\Rightarrow B)&=\frac{P(A\cap B)}{P(A)P(B)}=\frac{\text{confidence}(A\Rightarrow B)}{\text{supp}(B)} \\
\text{leverage}(A\Rightarrow B)&=P(A\cap B)-P(A)P(B) \\
\text{conviction}(A\Rightarrow B)&=\frac{1-\text{supp}(B)}{1-\text{confidence}(A\Rightarrow B)}
\end{align*}$$

- Lift $>1$: positively associated. $=1$: independent. $<1$: negatively associated.
- Support & lift are symmetric in $A,B$; confidence and conviction are NOT.
```

```{attention} Q&A
:class: dropdown
*Why isn't high confidence enough?*
- If 80% of ALL transactions contain milk, then `{anything} ⇒ {milk}` has 80% confidence while telling you nothing.
- Lift divides out $P(B)$ → lift ≈ 1 exposes the rule as worthless.

*Why is support needed at all?*
- A rule holding in 3 of 1,000,000 transactions can have 100% confidence and huge lift, and still be pure noise. Minimum support enforces statistical and business significance.
- It's also what makes mining tractable (see the Apriori property).

*Association ≠ causation?*
- Right. These are co-occurrence statistics on a static log. Beer⇒diapers does not mean beer causes diaper purchases.
```

&nbsp;

### Apriori
- **What**: Frequent itemsets grown level by level, pruning any candidate with an infrequent subset.
- **Why**: Combinatorial explosion.
    - $d$ items → $2^d-1$ possible itemsets → brute force is impossible.
    - → The **Apriori property** (downward closure) prunes enormous swaths of that lattice with one always-valid rule.
- **How**:
    1. Scan the DB → frequent 1-itemsets $L_1$.
    2. Join $L_{k-1}$ with itself → candidates $C_k$.
    3. **Prune**: drop any candidate having a $(k-1)$-subset not in $L_{k-1}$.
    4. Scan the DB → count survivors → $L_k$.
    5. Repeat until $L_k=\emptyset$; then generate rules from the frequent itemsets by confidence.

```{note} Math
:class: dropdown
**Apriori property (downward closure)**:

$$
\text{supp}(A)\geq\text{minsup}\ \Rightarrow\ \text{supp}(A')\geq\text{minsup}\quad\forall A'\subseteq A
$$

Contrapositive (the one actually used for pruning):

$$
\text{supp}(A')<\text{minsup}\ \Rightarrow\ \text{supp}(A)<\text{minsup}\quad\forall A\supseteq A'
$$
- → Any superset of an infrequent itemset is infrequent → never generate it.
- Valid because support is **anti-monotone**: adding an item can only shrink the matching transaction set.
```

````{important} Code
:class: dropdown
```python
from itertools import combinations

def apriori(transactions, minsup):
    """Return {frozenset(itemset): support} for all frequent itemsets."""
    n = len(transactions)
    transactions = [frozenset(t) for t in transactions]
    items = {frozenset([i]) for t in transactions for i in t}
    freq, k_sets = {}, items
    while k_sets:
        ## one DB scan per level -- the main cost of Apriori
        counts = {c: sum(c <= t for t in transactions) for c in k_sets}
        level = {c: v / n for c, v in counts.items() if v / n >= minsup}
        if not level:
            break
        freq.update(level)
        prev, k = set(level), len(next(iter(level))) + 1
        ## join step: union pairs, then PRUNE any candidate with an infrequent subset
        k_sets = {a | b for a in prev for b in prev if len(a | b) == k
                  and all(frozenset(s) in prev for s in combinations(a | b, k - 1))}
    return freq

## Example
T = [["milk", "bread"], ["milk", "bread", "eggs"], ["bread"], ["milk", "bread", "eggs"]]
f = apriori(T, minsup=0.5)
print(sorted((sorted(k), round(v, 2)) for k, v in f.items()))
## [(['bread'], 1.0), (['bread', 'milk'], 0.75), (['eggs'], 0.5), ...]
```
````

```{attention} Q&A
:class: dropdown
*Cons?*
- **Many DB scans** — one per level $k$ → I/O bound on large databases.
- **Candidate explosion** — the join step can generate a huge $C_k$ before counting prunes it.
- Very sensitive to `minsup`: too high → nothing found; too low → combinatorial blow-up.
- Only supports a single global `minsup` → rare-but-valuable items are invisible ("rare item problem").

*Closed vs maximal itemsets?*
- **Closed**: no superset has the SAME support → lossless compression of the frequent set (supports recoverable).
- **Maximal**: no superset is frequent → smallest output, but supports of subsets are lost.

*ECLAT?*
- Same goal, **vertical** data layout: store a TID-set (transaction id set) per item, then compute support by TID-set **intersection** and search depth-first.
- → No repeated DB scans; fast on dense data, but TID-sets blow up memory on large sparse databases.
```

&nbsp;

### FP-Growth
- **Name**: Frequent Pattern Growth
- **What**: Frequent itemsets mined from a prefix tree by recursively mining conditional sub-trees, ❌candidate generation.
- **Why**: Apriori's two killers.
    - Repeated DB scans (one per level) & candidate generation.
    - → A prefix tree shares common transaction prefixes → the DB often fits in memory compressed → exactly **2** scans, **zero** candidates.
- **How**:
    1. Scan 1: count item frequencies, drop infrequent items, sort the rest by descending frequency.
    2. Scan 2: insert each transaction (items in that order) into the **FP-tree**, sharing prefixes; maintain a header table linking all nodes of each item.
    3. For each item (least frequent first), extract its **conditional pattern base** → build a conditional FP-tree → recurse.
    4. Concatenate the recursion's suffixes → frequent itemsets.

```{attention} Q&A
:class: dropdown
*Why sort items by descending frequency before insertion?*
- Frequent items sit near the root → maximal prefix sharing → the smallest possible tree. A bad ordering makes the tree degenerate toward one path per transaction.

*Why is it faster than Apriori?*
- 2 DB scans total (vs one per level). ❌Candidate generation & testing. Divide-and-conquer on progressively smaller conditional trees.
- Typically an order of magnitude faster, and the gap widens as `minsup` drops.

*Cons?*
- **Memory**: the FP-tree must fit in RAM; on large, sparse, low-`minsup` data it can be enormous (mitigation: projected/partitioned databases).
- Recursive conditional-tree construction is complex to implement and hard to parallelize compared to Apriori's simple level-wise scans.
- Poor prefix sharing (very sparse data) → the tree is barely smaller than the DB → the advantage evaporates.
```

&nbsp;