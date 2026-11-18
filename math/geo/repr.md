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
# Representation Geometry
How neural networks & brains lay out information in activation space, how to compare 2 such layouts, & geometric theories of concepts & conscious qualities.
Framing: {cite:t}`kriegeskorte2013representational`. Representation-learning background: {cite:t}`bengio2013representation`.

Notations:
- $m$: #stimuli (inputs) shown to a system.
- $d$: Representation width (#neurons, #hidden units, or #voxels).
- $n$: #latent features (concepts) that a representation encodes.
- $\mathbf{h}\in\mathbb{R}^d$: Activation vector (representation) of one stimulus.
- $X\in\mathbb{R}^{m\times d_1}$: Representation matrix of system 1. Row $i$ = response to stimulus $i$.
- $Y\in\mathbb{R}^{m\times d_2}$: Representation matrix of system 2 for the same $m$ stimuli.
- $\delta(a,b)$: Dissimilarity between stimuli $a$ & $b$ in a representation or psychological space.

- **Overrides**: $Y$ = 2nd representation, ❌output. $n$ counts latent features, ❌input columns. $d_1,d_2$ = widths of $X,Y$.

&nbsp;

## Linear Structure

### Linear Representation Hypothesis
- **What**: High-level concepts = directions in representation space {cite:p}`park2024linear`.
    - 3 readings: **subspace** (counterfactual differences share 1 direction), **measurement** (linear probe reads the concept), **intervention** (adding 1 vector changes only the concept).
- **Why**: Concept = direction → read & edit model states with linear algebra (projection, cosine, vector addition).
    - Problem: "Linear" & "cosine" are undefined until we fix which space & which inner product.
- **How**:
    1. Collect counterfactual pairs that differ only in 1 concept (man→woman, king→queen).
    2. Unembedding differences of the pairs share 1 direction $\bar\gamma$ → probe direction.
    3. Map $\bar\gamma$ through the **causal inner product** → embedding direction $\bar\lambda$ → steering vector.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $x$: Context (prompt).
    - $y$: Next token.
- Params:
    - $\lambda(x)\in\mathbb{R}^d$: Embedding (context) vector = final hidden state.
    - $\gamma(y)\in\mathbb{R}^d$: Unembedding vector of token $y$.
- Misc:
    - $Z$: Binary concept w/ an order (e.g., male⇒female). Paper: $W$.
    - $y(0),y(1)$: Counterfactual outputs that differ only in $Z$.
    - $\operatorname{Cone}(\mathbf{v})=\{\alpha\mathbf{v}:\alpha>0\}$.

Model:

$$
P(y\mid x)\propto\exp\left(\lambda(x)^T\gamma(y)\right)
$$

Definitions:
- **Causally separable** $Z,Z'$: $y(Z=z,Z'=z')$ is well defined for all $z,z'$.
    - ✅ English⇒French & male⇒female ("king, queen, roi, reine").
    - ❌ English⇒French & English⇒Russian.
- **Unembedding representation** $\bar\gamma_Z$: $\gamma(y(1))-\gamma(y(0))\in\operatorname{Cone}(\bar\gamma_Z)$ almost surely.
- **Embedding representation** $\bar\lambda_Z$: $\lambda_1-\lambda_0\in\operatorname{Cone}(\bar\lambda_Z)$ for every context pair that satisfies both conditions:
    - Raises $P(Z=1)$.
    - Preserves $P(Z'\mid Z)$ for every causally separable $Z'$ (conditional, not marginal).
- **Causal inner product**: $\langle\bar\gamma_Z,\bar\gamma_{Z'}\rangle_C=0$ for every causally separable pair $Z,Z'$.

Properties:
- **Measurement**: $\bar\gamma_Z$ is a linear probe:

    $$
    \operatorname{logit}P\left(y=y(1)\mid y\in\{y(0),y(1)\},\lambda\right)=\alpha\,\lambda^T\bar\gamma_Z
    $$

    - $\alpha>0$: Scalar that depends on the pair $\{y(0),y(1)\}$.
- **Intervention**: $\lambda\to\lambda+c\bar\lambda_Z$ for a causally separable $Z'$:
    - Within a matched $Z$-pair $\{y(0,z'),y(1,z')\}$: $P(y(1,z'))$ increases in $c$.
    - Within a matched $Z'$-pair $\{y(z,0),y(z,1)\}$: $P(y(z,1))$ stays constant.
    - Off-target marginals $P(Z')$ need not stay constant.
- **Explicit form** holds under 3 assumptions:
    1. A causal inner product exists.
    2. $d$ mutually causally separable concepts have representations that form a basis.
    3. Their projections $\bar\lambda_Z^T\gamma$ are uncorrelated for $y$ uniform over the vocabulary.
    - Then the inner product is fixed up to a free diagonal factor $D$. Paper chooses $D=I$:

    $$
    \langle\bar\gamma,\bar\gamma'\rangle_C=\bar\gamma^T\operatorname{Cov}(\gamma)^{-1}\bar\gamma'
    $$

    - $\operatorname{Cov}(\gamma)$: Covariance of $\gamma(y)$ for $y$ uniform over the vocabulary.
- **Unification**: the causal inner product maps the probe to the steering vector:

    $$
    \bar\lambda_Z=\operatorname{Cov}(\gamma)^{-1}\bar\gamma_Z
    $$
```

```{tip} Derivation
:class: dropdown
*Why is the unembedding representation a linear probe?*

1. Restrict the softmax to 2 tokens:

    $$
    \operatorname{logit}P(y(1)\mid\cdot)=\log\frac{\exp(\lambda^T\gamma(y(1)))}{\exp(\lambda^T\gamma(y(0)))}=\lambda^T\left(\gamma(y(1))-\gamma(y(0))\right)
    $$

2. Cone definition → $\gamma(y(1))-\gamma(y(0))=\alpha\bar\gamma_Z$, $\alpha>0$.
3. → Same direction for every pair. Only the scale $\alpha$ changes.

*Why is Euclidean geometry in representation space arbitrary?*

1. Transform $\gamma\to A\gamma+\boldsymbol\beta$ & $\lambda\to A^{-T}\lambda$, $A$ invertible:

    $$
    (A^{-T}\lambda)^T(A\gamma+\boldsymbol\beta)=\lambda^T\gamma+\lambda^TA^{-1}\boldsymbol\beta
    $$

2. 2nd term is constant over $y$ → softmax unchanged → training cannot identify $A$.
3. Euclidean $\cos(\bar\gamma_Z,\bar\gamma_{Z'})$ changes with $A$ → no semantic meaning.
4. Causal inner product fixes geometry by semantics (separable ⇒ orthogonal), not by an arbitrary basis.
```

````{important} Code
:class: dropdown
```python
import torch

class LinearConcept:
    def __init__(self, U):
        ## U: (V, d) unembedding matrix, row = gamma(y)
        Uc = U - U.mean(0)
        cov = Uc.T @ Uc / U.shape[0]          ## Cov(gamma), y uniform over vocab
        self.M = torch.linalg.inv(cov)        ## causal inner product <a,b>_C = a^T M b

    def fit(self, U, idx0, idx1):
        ## gamma_bar: shared direction of counterfactual differences (man->woman, king->queen, ...)
        g = (U[idx1] - U[idx0]).mean(0)
        self.gamma = g / torch.sqrt(g @ self.M @ g)   ## canonical: <gamma, gamma>_C = 1
        self.lam = self.M @ self.gamma                 ## embedding repr = steering vector
        return self

    def measure(self, lam_x):
        ## probe: logit of y(1) vs y(0) is proportional to lambda(x)^T gamma_bar
        return lam_x @ self.gamma

    def steer(self, lam_x, c):
        ## intervention: move along lambda_bar
        return lam_x + c * self.lam

## Example: V=100 tokens, d=8; pairs (0->1), (2->3), (4->5) stand in for real word pairs
torch.manual_seed(0)
U = torch.randn(100, 8)
lc = LinearConcept(U).fit(U, idx0=[0, 2, 4], idx1=[1, 3, 5])
lam_x = torch.randn(8)
print(lc.measure(lc.steer(lam_x, 2.0)) - lc.measure(lam_x))  ## ≈ 2.0: lambda_bar^T gamma_bar = <gamma,gamma>_C = 1
```
````

```{attention} Q&A
:class: dropdown
*Classic evidence?*

- Word vectors: $\mathbf{v}_\text{king}-\mathbf{v}_\text{man}+\mathbf{v}_\text{woman}$ ≈ nearest to $\mathbf{v}_\text{queen}$ (vector offset method) {cite:p}`mikolov2013linguistic`.
- → Relation = constant offset = direction.

*Probe vs steering vector?*

- Probe reads a concept → lives in the unembedding (output) space → $\bar\gamma$.
- Steering vector writes a concept → lives in the embedding (context) space → $\bar\lambda$.
- Under the Euclidean inner product they differ. Under the causal inner product they coincide (Riesz map).

*Does high probe accuracy prove the model uses the concept?*

- ❌. Probe = decodability, not causal use.
- A trained probe absorbs correlated off-target info. Example: French text is mostly about men → a French probe also learns gender.
- $\bar\gamma$ from counterfactual pairs excludes such correlations → "ideal probe".
- Causal evidence needs an intervention that changes the output as predicted.

*Why not plain cosine similarity?*

- Inner product is unidentifiable ← softmax invariant to $A$ (Derivation).
- Causal inner product → causally separable concepts orthogonal. Experiments on LLaMA-2 support it.

*When does the hypothesis fail?*

- Some features are multi-dimensional (circles) → [Multi-Dimensional Features](#multi-dimensional-features).
- Linearity depends on the variable: Othello board is linear in "mine/theirs", not in "black/white" → [Emergent World Representation](#emergent-world-representation).
- Causal inner product is unique only up to $d$ free params ($D$). The choice $D=I$ is a convention.

*How does it relate to superposition?*

- More concept directions than dims → directions only nearly orthogonal → [Superposition](#superposition).

*What does it tell us about AC?*

- Concept = direction → internal states can be read (probe) & set (steer) → base of interpretability checks of AI self-reports.
- Hypothesis: a self-report is trustworthy only if a matching internal direction causally drives it. Readout ≠ experience.
```

&nbsp;

### Superposition
- **What**: Representation of more features than dims as nearly orthogonal directions {cite:p}`elhage2022toy`.
- **Why**: Width $d$ fits only $d$ orthogonal features, but the world has $n\gg d$ useful features.
    - Sparse features rarely co-occur → interference is rare → tolerated.
- **How**:
    1. Assign each feature a direction $\mathbf{w}_i$. #features $n>d$ → directions overlap.
    2. Read feature $i$ by projection → small interference from other active features.
    3. ReLU + negative bias filters small interference → clean readout when few features are active.
    4. Sparsity ⬆️ → more features packed. Packing changes via **phase changes** into polytopes (digons, triangles, pentagons, tetrahedra).

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{x}\in[0,1]^n$: Sparse feature vector.
    - $\hat{\mathbf{x}}\in\mathbb{R}^n$: Reconstruction.
- Params:
    - $W\in\mathbb{R}^{d\times n}$: Embedding. Paper: $m$ hidden dims → $d$ here.
    - $\mathbf{w}_i\in\mathbb{R}^d$: Column $i$ of $W$ = direction of feature $i$.
    - $\hat{\mathbf{w}}_i$: Unit vector of $\mathbf{w}_i$.
    - $\mathbf{b}\in\mathbb{R}^n$: Output bias.
- Hyperparams:
    - $S$: Sparsity = $P(x_i=0)$, same for all $i$.
    - $I_i$: Importance of feature $i$.

Model (ReLU output model):

$$
\mathbf{h}=W\mathbf{x},\qquad \hat{\mathbf{x}}=\operatorname{ReLU}(W^TW\mathbf{x}+\mathbf{b})
$$

- Data: $x_i=0$ w.p. $S$, else $x_i\sim U[0,1]$.
- Linear baseline: $\hat{\mathbf{x}}=W^TW\mathbf{x}+\mathbf{b}$.

Training:

$$
\mathcal{L}=\mathbb{E}_\mathbf{x}\sum_{i=1}^nI_i(x_i-\hat x_i)^2
$$

Properties:
- #features represented ≈ $\|W\|_F^2$ ← $\|\mathbf{w}_i\|^2\approx1$ (represented) or $\approx0$ (dropped).
- **Feature dimensionality** (fraction of a dim that feature $i$ owns):

    $$
    D_i=\frac{\|\mathbf{w}_i\|^2}{\sum_{j=1}^n(\hat{\mathbf{w}}_i\cdot\mathbf{w}_j)^2}
    $$

- Observed sticky values: $1$ (own dim), $3/4$ (tetrahedron), $2/3$ (triangle), $1/2$ (antipodal pair = digon), $2/5$ (pentagon), $3/8$ (square antiprism), $0$ (dropped).
- Compressed-sensing bound (paper): $d=\Omega\left(-n(1-S)\log(1-S)\right)$ → #features linear in $d$, scaled up by sparsity.
```

```{tip} Derivation
:class: dropdown
*Why does a linear model not use superposition?*

1. Linear model, $\mathbf{b}=0$, $\mathbf{x}\sim\mathcal{N}(0,I)$ (Saxe et al. setting used in the paper).
2. Error on feature $j$: $\hat x_j-x_j=\sum_i(\mathbf{w}_j\cdot\mathbf{w}_i-\delta_{ij})x_i$.
3. Independent unit-variance $x_i$ →

    $$
    \mathcal{L}=\underbrace{\sum_jI_j(1-\|\mathbf{w}_j\|^2)^2}_{\text{feature benefit}}+\underbrace{\sum_jI_j\sum_{i\ne j}(\mathbf{w}_j\cdot\mathbf{w}_i)^2}_{\text{interference}}
    $$

4. Interference is paid on every input → overlap never pays → top-$d$ features, PCA-like.

*Why do ReLU & sparsity allow it?*

1. Group inputs by #active features: $\mathcal{L}=(1-S)^n\mathcal{L}_n+\dots+(1-S)S^{n-1}\mathcal{L}_1+S^n\mathcal{L}_0$.
2. $S\to1$ → $\mathcal{L}_1$ & $\mathcal{L}_0$ dominate.
3. 1-sparse input ($x_i=x$, rest 0): $\hat x_j=\operatorname{ReLU}(x\,\mathbf{w}_j\cdot\mathbf{w}_i+b_j)$ for $j\neq i$.
4. $\mathbf{w}_j\cdot\mathbf{w}_i<0$ & $b_j\le0$ → $\hat x_j=0$ → negative interference is free.
5. → Pack features w/ negative overlaps (antipodal pairs, regular polygons) → $n>d$.

*Why are the sticky values $2/5$, $1/2$, ...?*

1. $k$ unit vectors that form a tight frame in an $r$-dim subspace: $\sum_j(\hat{\mathbf{w}}_i\cdot\mathbf{w}_j)^2=k/r$.
2. → $D_i=r/k$.
3. Pentagon: $r=2,k=5$ → $2/5$. Digon: $2$ features in $1$ dim → $1/2$. Tetrahedron: $4$ in $3$ dims → $3/4$.
```

```{note} Example
:class: dropdown
- 5 features of decreasing importance → 2 dims.
- Dense ($S=0$): 2 most important features get orthogonal dims. 3 features dropped (PCA-like).
- Sparse: all 5 features as a pentagon. Each $D_i=2/5$.
- Each readout $\hat x_i$ picks up $\cos72^\circ\approx0.31$ from each neighbor & $\cos144^\circ\approx-0.81$ from the 2 far features. ReLU clips the negative parts.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class ToyModel(nn.Module):
    def __init__(self, n, d):
        super().__init__()
        self.W = nn.Parameter(torch.randn(d, n) * 0.1)  ## column i = direction of feature i
        self.b = nn.Parameter(torch.zeros(n))

    def forward(self, x):                          ## x: (B, n)
        h = x @ self.W.T                           ## (B, d): compress, h = W x
        return torch.relu(h @ self.W + self.b)     ## (B, n): ReLU(W^T W x + b) filters interference

    def dimensionality(self):
        Wn = self.W / (self.W.norm(dim=0, keepdim=True) + 1e-8)
        overlap = ((Wn.T @ self.W) ** 2).sum(1)    ## sum_j (w_i_hat . w_j)^2
        return self.W.norm(dim=0) ** 2 / (overlap + 1e-8)

def sample(B, n, S):
    ## each feature is 0 w.p. S, else U[0, 1]
    return torch.rand(B, n) * (torch.rand(B, n) > S)

## Example: 5 features -> 2 dims
n, d, S = 5, 2, 0.95
I = 0.9 ** torch.arange(n)                         ## importance weights
model = ToyModel(n, d)
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
for _ in range(5000):
    x = sample(1024, n, S)
    loss = (I * (x - model(x)) ** 2).sum(-1).mean()  ## importance-weighted MSE
    opt.zero_grad(); loss.backward(); opt.step()
print(model.W.norm(dim=0))       ## S=0 -> 2 norms near 1; high S -> typically more than 2 (superposition)
print(model.dimensionality())    ## e.g., pentagon -> all ≈ 0.4
```
````

```{attention} Q&A
:class: dropdown
*What is a polysemantic neuron?*

- Neuron that fires for several unrelated features.
- ← Features in superposition do not align w/ neurons → each neuron sees a mix.

*What is a privileged basis?*

- Basis that the architecture makes special, e.g., an elementwise activation → neurons.
- Residual stream & word embeddings: no privileged basis ← any rotation $A$ + $A^{-1}$ in the next layer gives the same model.
- Privileged basis pulls features toward neurons. Superposition pushes them away.

*How can $n\gg d$ directions be nearly orthogonal?*

- High dims: $\exp(d)$ many vectors w/ small pairwise cosine → [Near-Orthogonality](highdim.md#near-orthogonality), [Johnson-Lindenstrauss Lemma](highdim.md#johnson-lindenstrauss-lemma).

*When does superposition appear?*

- Phase change in (sparsity, importance): dropped ↔ own dim ↔ superposition.
- Sparsity ⬆️ → interference rarer → superposition wins.
- Importance ⬆️ → feature gets its own dim.

*Is it real in large models?*

- Toy-model result. The authors state it is unclear how it generalizes.
- Indirect support: SAEs recover many monosemantic directions from polysemantic layers → [SAE](#sae).

*Side effects?*

- Toy model: adversarial vulnerability rises sharply (>3×) as superposition forms. Preliminary; adversarial examples have other explanations too.

*How does it relate to hyperdimensional computing?*

- HDC superposes symbols on purpose (bundling) → [Hyperdimensional Computing](highdim.md#hyperdimensional-computing).
- Same trade-off: capacity ⬆️ ↔ crosstalk ⬆️.

*What does it tell us about AC?*

- Neuron-by-neuron reading of an AI's state fails ← superposition.
- → Any claim about an AI's internal states needs a feature-level decomposition first.
```

&nbsp;

### SAE
- **Name**: Sparse Autoencoder {cite:p}`bricken2023towards`
- **What**: Overcomplete dictionary of sparse, nonneg features learned to reconstruct activations.
- **Why**: Undo superposition.
    - Neurons are polysemantic → poor units of analysis.
    - Sparse overcomplete basis → each learned feature ≈ 1 concept (monosemantic).
- **How**:
    1. Collect activations $\mathbf{h}$ from 1 layer over many tokens.
    2. Encode: linear map to $F\gg d$ units + ReLU → sparse codes $\mathbf{f}$.
    3. Decode: $\hat{\mathbf{h}}$ = sum of dictionary columns weighted by $\mathbf{f}$.
    4. Train on reconstruction + L1 sparsity. Keep decoder columns unit norm.
    5. Interpret each feature from its top-activating inputs & its effect on outputs.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\hat{\mathbf{h}}\in\mathbb{R}^d$: Reconstructed activation.
    - $\mathbf{f}\in\mathbb{R}^F$: Feature activations (sparse codes).
- Params:
    - $W_e\in\mathbb{R}^{F\times d}$: Encoder.
    - $\mathbf{b}_e\in\mathbb{R}^F$: Encoder bias.
    - $W_d\in\mathbb{R}^{d\times F}$: Decoder = dictionary. Columns unit norm.
    - $\mathbf{b}_d\in\mathbb{R}^d$: Decoder bias, also subtracted before encoding.
- Hyperparams:
    - $F$: Dictionary size. $F>d$ → overcomplete. $F/d$ = expansion factor.
    - $\lambda$: L1 coeff.
- Paper notation: input dim $n$ → $d$ here; hidden dim $m$ → $F$ here.

Forward:

$$\begin{align*}
\bar{\mathbf{h}}&=\mathbf{h}-\mathbf{b}_d\\
\mathbf{f}&=\operatorname{ReLU}(W_e\bar{\mathbf{h}}+\mathbf{b}_e)\\
\hat{\mathbf{h}}&=W_d\mathbf{f}+\mathbf{b}_d
\end{align*}$$

Objective:

$$
\mathcal{L}=\frac{1}{|\mathcal{D}|}\sum_{\mathbf{h}\in\mathcal{D}}\left(\|\mathbf{h}-\hat{\mathbf{h}}\|_2^2+\lambda\|\mathbf{f}(\mathbf{h})\|_1\right)
$$

- $\mathcal{D}$: Dataset of activations.
- Paper averages the MSE over elements & sums the L1 → only rescales $\lambda$.
- Later variant (Anthropic, "Scaling Monosemanticity", 2024): drop the unit-norm constraint & penalize $\lambda\sum_if_i\|W_{d,:i}\|_2$.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class SAE(nn.Module):
    def __init__(self, d, F):
        super().__init__()
        self.W_e = nn.Parameter(torch.randn(F, d) * d ** -0.5)
        self.b_e = nn.Parameter(torch.zeros(F))
        self.W_d = nn.Parameter(torch.randn(d, F) * F ** -0.5)  ## untied decoder = dictionary
        self.b_d = nn.Parameter(torch.zeros(d))
        self.normalize_decoder()

    def forward(self, h):                                          ## h: (B, d)
        f = torch.relu((h - self.b_d) @ self.W_e.T + self.b_e)     ## (B, F) sparse, nonneg codes
        h_hat = f @ self.W_d.T + self.b_d                          ## (B, d)
        return f, h_hat

    def loss(self, h, lam):
        f, h_hat = self(h)
        return ((h - h_hat) ** 2).sum(-1).mean() + lam * f.abs().sum(-1).mean()

    @torch.no_grad()
    def normalize_decoder(self):
        ## unit-norm atoms -> L1 cannot be cut by shrinking f & growing W_d; call after each step
        self.W_d /= self.W_d.norm(dim=0, keepdim=True)

## Example: d=8 activations, 4x expansion
sae = SAE(d=8, F=32)
h = torch.randn(4, 8)
f, h_hat = sae(h)
print(f.shape, h_hat.shape, sae.loss(h, lam=1e-3).item() > 0)  ## torch.Size([4, 32]) torch.Size([4, 8]) True
```
````

```{attention} Q&A
:class: dropdown
*Why ReLU + L1?*

- L1 = convex surrogate for L0 → few active features per input.
- ReLU → exact zeros & nonneg codes.

*Why unit-norm decoder columns?*

- W/o it: scale $\mathbf{f}$ ⬇️ & $W_d$ ⬆️ → same $\hat{\mathbf{h}}$, smaller L1 → the penalty is gamed.

*Why untied weights?*

- Decoder = dictionary. Encoder = cheap, amortized approximation to sparse inference.
- Roles differ → paper unties them.

*What are dead features?*

- Features that never fire → wasted capacity.
- Fix (paper): periodically resample dead features. Reset their encoder weights toward inputs that the SAE reconstructs badly.

*Setup of the original study?*

- 1-layer transformer, 512-neuron MLP, 8B activation samples.
- Expansion 1× (512) → 256× (131,072 features). Detailed analysis on a 4,096-feature run.
- Larger dictionaries → features split into finer features (feature splitting).

*Independent evidence?*

- Concurrent work: SAE features score more interpretable than alternative decompositions (automated scoring) & localize indirect-object-identification behavior more finely {cite:p}`cunningham2024sparse`.

*Cons?*

- L1 shrinks active codes → magnitudes underestimated. Variants: TopK, JumpReLU, gated SAEs.
- Reconstruction error ≠ 0 → part of the activation is not explained by any feature.
- Features depend on $F$, $\lambda$, & seed → not a unique ground truth.
- Assumes features are 1-D directions → [Multi-Dimensional Features](#multi-dimensional-features).

*How does it relate to classic methods?*

- = Sparse coding / dictionary learning on activations.
- vs [PCA](../../ml/unsupervised.md#pca): PCA = orthogonal components, often truncated (undercomplete), no sparsity. SAE = overcomplete dictionary & sparse codes.
- vs [ICA](../../ml/unsupervised.md#ica): ICA finds $\le d$ independent components. SAE allows $F\gg d$.

*What does it tell us about AC?*

- SAEs list an AI's internal features at scale → today's main tool to inspect AI internal states.
- Hypothesis: features tied to self-modeling could be tested against the model's self-reports. Finding such features ≠ evidence of experience.
```

&nbsp;

### Multi-Dimensional Features
- **What**: Features that are irreducibly multi-dim, e.g., circles for days of the week & months of the year {cite:p}`engels2025not`.
- **Why**: "1 concept = 1 direction" is too narrow.
    - Cyclic variables need a circle. A single direction cannot express "Sunday is next to Monday".
- **How**:
    1. Cluster SAE decoder columns by cosine similarity.
    2. Reconstruct activations from 1 cluster → project to its top PCs.
    3. Test irreducibility: not separable into independent parts & not a mixture of non-co-occurring parts.
    4. Confirm use: intervene on the circle → the model's modular-arithmetic answer changes as predicted.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\mathbf{f}(t)\in\mathbb{R}^{d_f}$: Value of a $d_f$-dim feature on token $t$ (defined where the feature is active).
    - $R$: Orthonormal $d_f\times d_f$ matrix.
    - $\mathbf{c}$: Constant vector.
    - $\mathbf{a},\mathbf{b}$: Lower-dim parts after the transform.

Definition (reducible feature):
- $\mathbf{f}$ is reducible into $\mathbf{a},\mathbf{b}$ if some $R\mathbf{f}+\mathbf{c}=(\mathbf{a},\mathbf{b})$ satisfies at least one condition:
    - **Separable**: $p(\mathbf{a},\mathbf{b})=p(\mathbf{a})\,p(\mathbf{b})$.
    - **Mixture**: $p(\mathbf{a},\mathbf{b})=w\,p_1(\mathbf{a})\,\delta(\mathbf{b})+(1-w)\,p_2(\mathbf{a},\mathbf{b})$, disjoint supports, $0<w<1$.
- **Irreducible** = not reducible. Soft versions: separability index (min mutual information over rotations) & $\epsilon$-mixture index.

Circle feature (weekday $k\in\{0,\dots,6\}$):

$$
\mathbf{f}(k)\approx r\left(\cos\frac{2\pi k}{7},\ \sin\frac{2\pi k}{7}\right)
$$

- "$+j$ days" = rotation by $2\pi j/7$ → addition mod 7.
- $r$: Radius.
```

```{attention} Q&A
:class: dropdown
*Where were they found?*

- GPT-2 small (layer 7): circles for weekdays, months & years of the 20th century.
- Mistral 7B: circles for weekdays & months.

*Are the circles used for computation?*

- Tasks: "Two days from Monday is", "Four months from January is".
- Accuracy: Llama 3 8B 29/49 & 143/144; Mistral 7B 31/49 & 125/144; GPT-2 8/49 & 10/144.
- Interventions on the circle in Mistral 7B & Llama 3 8B change the answer as the rotation predicts.

*Why a circle?*

- Cyclic group $\mathbb{Z}_7$ → rotation representation → [Group Representation](symmetry.md#group-representation).
- Same mechanism as networks that learn modular addition → [Grokking Modular Addition](symmetry.md#grokking-modular-addition).

*Why is irreducibility needed?*

- 2 independent 1-D features also span a plane → a 2-D cluster alone proves nothing.
- Separable / mixture tests rule out those trivial cases.

*What does it change for interpretability?*

- Unit of analysis = low-dim subspace or manifold, not only a direction.
- 1 circle is spread over several 1-D SAE latents → cluster them back.
```

&nbsp;

## World Models
- **What**: Internal variables that track the state of the process that generates the data, beyond surface statistics.

### Emergent World Representation
- **What**: Board state that a sequence model trained only on Othello move sequences represents internally {cite:p}`li2023emergent`.
- **Why**: Test "next-token prediction = surface statistics only" in a world w/ known ground truth.
- **How**:
    1. Train an 8-layer GPT (Othello-GPT) on legal move sequences. No rules, no board.
    2. Probe each layer's activations for the state of each tile (empty / black / white).
    3. Linear probes → poor. 2-layer MLP probes → high accuracy. Random-network baseline → low.
    4. Intervene: edit activations until the probe reads a counterfactual board → predicted legal moves follow the edited board.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\mathbf{h}_t^\ell$: Activation of token $t$ after layer $\ell$.
    - $W_1,W_2$: Probe weights.

Nonlinear probe (per tile, 3 classes):

$$
p_\theta(\mathbf{h}_t^\ell)=\operatorname{softmax}\left(W_1\operatorname{ReLU}(W_2\mathbf{h}_t^\ell)\right)
$$

Intervention:
- Gradient descent on $\mathbf{h}_t^\ell$ (not on weights) → probe outputs the counterfactual tile state.
- Continue the forward pass from the edited activation.
```

```{attention} Q&A
:class: dropdown
*Is the model only memorizing?*

- Legal-move error: 0.01% (synthetic data, 20M games), 5.17% (championship data), 93.29% (untrained).
- Skewed-dataset test: remove all games that open w/ C5 (¼ of the game tree) → error on those unseen games still 0.02%.

*Why do linear probes fail?*

- Wrong variable. Board is linear in "mine vs theirs" (relative to the player to move), not "black vs white" {cite:p}`nanda2023emergent`.
- → Linear probes & vector-addition interventions work.
- Lesson: "no linear probe" ≠ "no linear representation".

*What are latent saliency maps?*

- Flip each tile in the world representation → measure the change in the predicted move → attribution in board terms.

*Limits?*

- Tiny, fully observed, synthetic world.
- Does not show that LLMs trained on natural text build comparable world models → contested.

*What does it tell us about AC?*

- World model = core ingredient of several theories, e.g., [Predictive Processing](../../ac/theories.md#predictive-processing-pp).
- Self-supervised prediction alone can produce one. Hypothesis: necessary, not sufficient for consciousness.
```

&nbsp;

### Space & Time Representations
- **What**: Linear representations of real-world location & time of entities in LLM activations {cite:p}`gurnee2024language`.
- **Why**: Test whether LLMs build coherent world representations or only collect surface statistics.
- **How**:
    1. 6 datasets: world, US, NYC places; historical figures, artworks, news headlines.
    2. Feed entity names to Llama-2 (7B–70B) → take the last-token activation per layer.
    3. Fit linear ridge-regression probes → latitude & longitude, or year.
    4. Find "space neurons" & "time neurons" → test them w/ causal interventions.

```{attention} Q&A
:class: dropdown
*Linear or nonlinear?*

- MLP probes (256 hidden units) give minimal $R^2$ gain over linear probes → linear, or at least linearly decodable.

*Where in the network?*

- Builds through early layers → plateaus near the middle layer.
- Larger models → better probes.

*Robust?*

- Fairly robust to prompt variations.
- Unified across entity types (cities & natural landmarks share one map).

*Does this prove a world model?*

- Contested. Linear decodability ≠ use in reasoning.
- A memorized coordinate per entity is also linearly decodable.
- Paper's own framing: "basic ingredients of a world model".

*How does it relate to the brain?*

- Hippocampal place & grid cells encode space as low-dim manifolds → [Toroidal Grid-Cell Manifold](topology.md#toroidal-grid-cell-manifold).
- LLM version comes from text only, with no sensorimotor data.
```

&nbsp;

## Comparing Representations

### RSA
- **Name**: Representational Similarity Analysis {cite:p}`kriegeskorte2008representational`
- **What**: Comparison of 2 systems via their stimulus-by-stimulus dissimilarity matrices.
- **Why**: Units do not correspond across brains, models & recording methods (correspondence problem).
    - RDM abstracts away units → any 2 systems shown the same stimuli become comparable.
- **How**:
    1. Show the same $m$ stimuli to each system (brain region, model layer, behavior).
    2. Per system: dissimilarity of each stimulus pair → $m\times m$ RDM.
    3. Compare the upper triangles of 2 RDMs by rank correlation.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\mathbf{x}_a$: Row $a$ of $X$ = response pattern to stimulus $a$.
    - $\Delta^X\in\mathbb{R}^{m\times m}$: RDM of system $X$.
    - $\rho_S$: Spearman rank correlation.

Process:
1. RDM (correlation distance, as in the paper):

    $$
    \Delta^X_{ab}=\delta(a,b)=1-\operatorname{corr}(\mathbf{x}_a,\mathbf{x}_b)
    $$

    - Range $[0,2]$. Symmetric. Zero diagonal.
2. Compare:

    $$
    \operatorname{RSA}(X,Y)=\rho_S\left(\operatorname{vec}_{a<b}(\Delta^X),\ \operatorname{vec}_{a<b}(\Delta^Y)\right)
    $$

    - $\operatorname{vec}_{a<b}$: Vector of the $m(m-1)/2$ upper-triangle entries.

Properties:
- No unit-to-unit mapping. $d_1\neq d_2$ allowed.
- Rank correlation → invariant to monotone transforms of dissimilarities.
- Correlation distance → invariant to a shift & positive scaling of each response pattern.
```

````{important} Code
:class: dropdown
```python
import numpy as np

class RSA:
    @staticmethod
    def rdm(X):
        ## X: (m, d); np.corrcoef treats rows as variables -> (m, m) stimulus correlations
        return 1 - np.corrcoef(X)

    @staticmethod
    def rank(v):
        r = np.empty(len(v))
        r[np.argsort(v)] = np.arange(len(v))   ## ties ignored (continuous data)
        return r

    def __call__(self, X, Y):
        iu = np.triu_indices(X.shape[0], k=1)  ## RDM symmetric, diag = 0 -> upper triangle only
        a, b = self.rank(self.rdm(X)[iu]), self.rank(self.rdm(Y)[iu])
        return np.corrcoef(a, b)[0, 1]         ## Spearman = Pearson on ranks

## Example: 6 stimuli, systems of different widths
rng = np.random.default_rng(0)
X = rng.normal(size=(6, 10))
print(RSA()(X, 3 * X + 1))                     ## ≈ 1.0: per-pattern affine change keeps the RDM
print(RSA()(X, rng.normal(size=(6, 40))))      ## random, near 0 on average (only 15 pairs -> noisy)
```
````

```{attention} Q&A
:class: dropdown
*Why rank correlation, not Pearson?*

- Model & brain dissimilarities need not relate linearly.
- Noise in high-dim patterns acts largely as a monotone transform of dissimilarities (paper's argument).

*Pros?*

- Model-free bridge between brain activity, behavior (similarity judgments) & computational models.
- Works across fMRI, electrophysiology, species, & DNN layers.

*Cons?*

- Needs the same stimulus set in both systems.
- Result depends on the dissimilarity measure & the stimulus sample.
- $m^2$ entries from $m$ stimuli → entries not independent → use stimulus bootstrap for errors.

*How does it relate to CKA?*

- Euclidean RDM: $-\tfrac12 H\Delta^{(2)}H=HKH$ (classical MDS). $\Delta^{(2)}$ = squared distances, $K$ = Gram matrix, $H$ = centering.
- → Linear [CKA](#cka) = cosine between double-centered squared-distance matrices.
- Both compare stimulus-by-stimulus geometry, not units.

*What does it tell us about AC?*

- Same tool compares human & AI similarity structure on the same stimuli.
- If qualities are individuated by similarity structure ([Quality Space](#quality-space)), RSA is the measurement. Hypothesis.
```

&nbsp;

### SVCCA
- **Name**: Singular Vector Canonical Correlation Analysis {cite:p}`raghu2017svcca`
- **What**: SVD to keep the top directions of each representation → CCA → mean canonical correlation.
- **Why**: Compare layers & networks fast while ignoring noise directions & linear re-parameterizations.
- **How**:
    1. SVD each centered representation. Keep directions that explain 99% of the variance.
    2. CCA: find paired linear combinations w/ max correlation.
    3. Report the mean of the canonical correlations.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $X'\in\mathbb{R}^{m\times r_X}$: Top-$r_X$ SVD projection of centered $X$.
    - $Y'\in\mathbb{R}^{m\times r_Y}$: Top-$r_Y$ SVD projection of centered $Y$.
    - $\mathbf{a}_k,\mathbf{b}_k$: $k$-th canonical weight vectors.
    - $r=\min(r_X,r_Y)$.

Process:
1. $X=U\Sigma V^T$ → $X'=U_{:,1:r_X}\Sigma_{1:r_X}$, $r_X$ = smallest rank w/ ≥99% of $\sum\sigma^2$. Same for $Y$.
2. CCA:

    $$
    \rho_k=\max_{\mathbf{a}_k,\mathbf{b}_k}\operatorname{corr}(X'\mathbf{a}_k,\ Y'\mathbf{b}_k)
    $$

    - Each new pair uncorrelated w/ the previous pairs.
3. Output: $\bar\rho=\frac{1}{r}\sum_{k=1}^r\rho_k$.

Properties:
- $\rho_k$ = singular values of $Q_X^TQ_Y$. $Q_X,Q_Y$: orthonormal bases of the column spaces of $X',Y'$.
- Invariant to invertible linear maps & shifts within the kept subspaces.
```

````{important} Code
:class: dropdown
```python
import numpy as np

class SVCCA:
    def __init__(self, keep=0.99):
        self.keep = keep

    def top(self, X):
        X = X - X.mean(0)
        U, s, _ = np.linalg.svd(X, full_matrices=False)
        r = np.searchsorted(np.cumsum(s**2) / np.sum(s**2), self.keep) + 1  ## dirs for 99% variance
        return U[:, :r] * s[:r]                                             ## (m, r)

    def __call__(self, X, Y):
        Qx, _ = np.linalg.qr(self.top(X))
        Qy, _ = np.linalg.qr(self.top(Y))
        rho = np.linalg.svd(Qx.T @ Qy, compute_uv=False)  ## canonical correlations
        return rho.mean()

## Example: m=200 stimuli >> width 30
rng = np.random.default_rng(0)
X = rng.normal(size=(200, 30))
print(SVCCA(keep=1.0)(X, X @ rng.normal(size=(30, 30))))  ## ≈ 1.0: CCA ignores invertible linear maps
print(SVCCA()(X, rng.normal(size=(200, 30))))             ## much lower: unrelated representation
```
````

```{attention} Q&A
:class: dropdown
*Why SVD before CCA?*

- Many low-variance directions ≈ noise. CCA weights all directions equally → it fits noise.

*Findings?*

- Networks converge bottom-up: lower layers reach their final representation first → "freeze training" (freeze lower layers early).
- #directions needed ≪ #neurons → some layers over-parameterized.
- Conv layers: DFT makes exact SVCCA cheap.

*Cons?*

- Untruncated CCA degenerates when both centered representations span the full $(m-1)$-dim centered sample space → any 2 such representations look identical → [CKA](#cka).
- SVCCA's truncation can avoid this exact collapse.
- Identifying the matching layer across 10 CNNs trained from different seeds: SVCCA 9.9–15.1%, linear CKA 99.3% (Kornblith et al.).
- Result depends on the truncation threshold.
- Equal weight on all kept directions. Weighted fix: PWCCA.
```

&nbsp;

### CKA
- **Name**: Centered Kernel Alignment {cite:p}`kornblith2019similarity`
- **What**: Normalized HSIC between the centered Gram matrices of 2 representations.
- **Why**: CCA-type indices are invariant to every invertible linear map → degenerate when both representations span the full centered sample space (e.g., width ≥ #stimuli).
    - Right invariance: orthogonal transforms & isotropic scaling only.
- **How**:
    1. Center each unit over the $m$ stimuli.
    2. Build stimulus-by-stimulus similarity (Gram) matrices.
    3. CKA = cosine similarity between the 2 centered Gram matrices.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $H=I_m-\frac1m\mathbf{1}\mathbf{1}^T$: Centering matrix.
    - $K,L\in\mathbb{R}^{m\times m}$: Kernel (Gram) matrices of $X,Y$. Linear: $K=XX^T$, $L=YY^T$.

Definition:

$$
\operatorname{HSIC}(K,L)=\frac{\operatorname{tr}(KHLH)}{(m-1)^2},\qquad
\operatorname{CKA}(K,L)=\frac{\operatorname{HSIC}(K,L)}{\sqrt{\operatorname{HSIC}(K,K)\operatorname{HSIC}(L,L)}}
$$

Linear CKA (columns of $X,Y$ centered):

$$
\operatorname{CKA}(X,Y)=\frac{\|Y^TX\|_F^2}{\|X^TX\|_F\,\|Y^TY\|_F}
$$

Properties:
- $\in[0,1]$ for PSD kernels.
- Invariant: $X\to\alpha XQ+\mathbf{1}\mathbf{c}^T$, $Q$ orthogonal, $\alpha>0$.
- ❌ Invariant to invertible linear $X\to XA$.
- RBF kernel → local, nonlinear similarity. Bandwidth = fraction of the median pairwise distance.
- Cost: $O(m(d_1^2+d_1d_2+d_2^2))$ in feature form. $O(m^2(d_1+d_2))$ in Gram form.
```

```{tip} Derivation
:class: dropdown
*Gram form = feature form?*

1. Centered $X$ → $HX=X$ → $\operatorname{tr}(KHLH)=\operatorname{tr}(XX^TYY^T)=\operatorname{tr}(Y^TXX^TY)=\|X^TY\|_F^2$.
2. Same for $K=L$: $\operatorname{tr}(KHKH)=\|X^TX\|_F^2$.
3. $(m-1)^2$ cancels → linear CKA formula.

*Why "cosine of centered Gram matrices"?*

1. $H$ symmetric & $H^2=H$ → $\langle HKH,HLH\rangle_F=\operatorname{tr}(HKHHLH)=\operatorname{tr}(KHLH)$.
2. → CKA = $\langle HKH,HLH\rangle_F/(\|HKH\|_F\|HLH\|_F)$.

*Which directions dominate linear CKA?*

1. Eigendecompose $XX^T=\sum_i\lambda_i^X\mathbf{u}_i^X\mathbf{u}_i^{XT}$, same for $Y$.
2. $\|Y^TX\|_F^2=\sum_{i,j}\lambda_i^X\lambda_j^Y\langle\mathbf{u}_i^X,\mathbf{u}_j^Y\rangle^2$.
3. → Weighted by eigenvalue products → top principal components dominate. CCA weights all directions equally.

*Why is invariance to invertible linear maps bad?*

1. Kornblith et al., Theorem 1: $s(X,Z)=s(XA,Z)$ for every full-rank $A$, & $\operatorname{rank}(X)=\operatorname{rank}(Y)=m$ → $s(X,Z)=s(Y,Z)$.
2. Intuition: rank-$m$ $X$ → $XA$ reaches any other rank-$m$ representation → the index sees no difference.
3. Wide layers often span the full $(m-1)$-dim centered sample space → untruncated CCA-type indices become uninformative.
```

````{important} Code
:class: dropdown
```python
import numpy as np

class LinearCKA:
    def __call__(self, X, Y):
        X = X - X.mean(0)   ## center each unit over stimuli
        Y = Y - Y.mean(0)
        num = np.linalg.norm(Y.T @ X, "fro") ** 2
        return num / (np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro"))

## Example
rng = np.random.default_rng(0)
X = rng.normal(size=(100, 20))
Q, _ = np.linalg.qr(rng.normal(size=(20, 20)))         ## random orthogonal matrix
cka = LinearCKA()
print(cka(X, 5 * X @ Q + 3))                           ## ≈ 1.0: rotation + isotropic scale + shift
print(cka(X, X @ rng.normal(size=(20, 20))))           ## < 1: general invertible map changes CKA
print(cka(X, rng.normal(size=(100, 20))))              ## small: unrelated
```
````

```{attention} Q&A
:class: dropdown
*Why invariance to orthogonal transforms?*

- Preserves dot products & distances between stimuli → same geometry.
- Implies permutation invariance → neuron order does not matter.
- Linear model: orthogonal transform of the inputs + matching rotation of the weights → same gradient-descent dynamics.
- Rotation-symmetric init (e.g., i.i.d. Gaussian) → fixed orthogonal transforms of activations keep the distribution of training trajectories. Arbitrary linear maps do not (paper's argument).

*Why not invariance to non-isotropic scaling?*

- Orthogonal + non-isotropic scaling = every invertible linear map (via SVD) → same failure as CCA.

*Evidence that CKA works?*

- Finds the matching layer across 10 CNNs trained from different seeds: linear CKA 99.3% vs CCA 1.4–10.6% & SVCCA 9.9–15.1%.

*Cons?*

- Dominated by high-variance directions (Derivation) → can miss low-variance but functional differences.
- Needs the same stimuli.
- Global index → small for models that agree only locally. → Platonic study switched to mutual k-NN.
- Independent isotropic noise can give linear CKA → 1 when widths ≫ #stimuli ← both normalized centered Gram matrices → $H$.
- → Compare against null baselines of matched width.
```

```{dropdown} Table: Representation Similarity Measures
| | RSA | SVCCA | Linear CKA | Mutual k-NN |
|:--|:--|:--|:--|:--|
| Input | 1 RDM per system ($m\times m$) | $X,Y$ | $X,Y$ (or Gram matrices) | $X,Y$ |
| Compares | Global dissimilarity ranks | Shared linear subspace | Global centered similarity | Local neighborhoods |
| Invariant to | Monotone maps of dissimilarities; whatever the dissimilarity ignores | Invertible linear maps & shifts (in kept subspace) | Orthogonal maps, isotropic scaling, shifts | Any map that keeps each stimulus's $k$ nearest neighbors |
| Not invariant to | Change of dissimilarity measure | Truncation threshold | Invertible linear maps | Change of $k$ or batch |
| Cost | $O(m^2d)$ + ranking | SVD $O(md^2)$ + CCA | $O(m(d_1^2+d_1d_2+d_2^2))$ | $O(m^2d)$ + sort |
| Failure modes | Few stimuli → low power; correlated RDM entries | Low truncation → fits noise; untruncated & full $(m-1)$-dim span → all look identical | Top PCs dominate; misses local agreement | Ignores global geometry; low absolute scores |
| Typical use | Brain ↔ model ↔ behavior | Training dynamics, layer dimension | Layer-to-layer, model-to-model | Cross-model & cross-modal convergence |
```

&nbsp;

### Platonic Representation Hypothesis
- **What**: Representations of different models, objectives & modalities converge toward a shared statistical model of reality {cite:p}`huh2024platonic`.
    - ICML 2024 position paper. Hypothesis, not theorem.
- **Why**: Explain why foundation models transfer & why vision & language embeddings can be aligned.
    - Images & text = 2 projections of 1 world $\mathbf{z}$ → good models should recover the same structure of $\mathbf{z}$.
- **How**:
    1. Measure alignment as kernel agreement: do 2 models see the same stimuli as neighbors?
    2. Metric: mutual k-NN overlap on paired data (image–caption pairs for cross-modal).
    3. Observe: alignment ⬆️ w/ competence & scale, within & across modalities.
    4. Explain: multitask scaling, capacity, & simplicity bias shrink the set of valid solutions.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\phi_i=f(x_i)$, $\psi_i=g(y_i)$: Features of model $f$ & model $g$ for paired sample $i$.
    - $\mathcal{S}(\phi_i)$: Indices of the $k$ nearest neighbors of $\phi_i$ in the batch, excluding $i$.
    - $b$: Batch size.
    - $\mathbf{z}$: World event. $x=\operatorname{obs}_X(\mathbf{z})$, $y=\operatorname{obs}_Y(\mathbf{z})$.

Mutual k-NN alignment:

$$
\operatorname{mNN}=\frac1b\sum_{i=1}^b\frac{1}{k}\left|\mathcal{S}(\phi_i)\cap\mathcal{S}(\psi_i)\right|
$$

- $\in[0,1]$. Independent random features → $\approx k/(b-1)$.

Idealized endpoint (discrete world, bijective observations):
- Contrastive learner on co-occurrence, at a realizable population optimum → $\langle f(x_a),f(x_b)\rangle=K_\text{PMI}(x_a,x_b)+c_X$.
- Exact realization needs $K_\text{PMI}+c_X$ PSD & enough embedding dims.
- $K_\text{PMI}(x_a,x_b)=\log\frac{P_\text{coor}(x_a,x_b)}{P_\text{coor}(x_a)P_\text{coor}(x_b)}$: Pointwise mutual information of co-occurrence.
- Bijective $\operatorname{obs}$ → $K_\text{PMI}$ over $x$ = $K_\text{PMI}$ over $\mathbf{z}$ → same kernel for every modality.
```

````{important} Code
:class: dropdown
```python
import numpy as np

class MutualKNN:
    def __init__(self, k=10):
        self.k = k

    def knn(self, X):
        X = X / np.linalg.norm(X, axis=1, keepdims=True)  ## cosine neighbors (a choice here)
        S = X @ X.T
        np.fill_diagonal(S, -np.inf)                      ## exclude self
        return np.argsort(-S, axis=1)[:, :self.k]         ## (b, k) neighbor indices

    def __call__(self, X, Y):
        A, B = self.knn(X), self.knn(Y)
        return np.mean([len(set(a) & set(b)) / self.k for a, b in zip(A, B)])

## Example: b=50 paired samples
rng = np.random.default_rng(0)
X = rng.normal(size=(50, 8))
Q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
print(MutualKNN(5)(X, 2 * X @ Q))                     ## 1.0: same neighborhoods
print(MutualKNN(5)(X, rng.normal(size=(50, 16))))     ## ≈ 5/49 ≈ 0.1: chance level
```
````

```{attention} Q&A
:class: dropdown
*Evidence?*

- 78 vision models (Places-365): models that solve more VTAB tasks align more w/ each other.
- LLM ↔ vision (Wikipedia image–caption pairs): alignment rises ≈ linearly w/ language-modeling score. CLIP-type models align more.
- LLMs more aligned w/ DINOv2 → better on HellaSwag & GSM8K.
- Color: color distances from co-occurrence in images & in text resemble perceptual (CIELAB) distances.

*Why would representations converge?*

- **Multitask scaling**: fewer representations solve $N$ tasks than $M<N$ tasks.
- **Capacity**: bigger models reach the shared optimum more easily.
- **Simplicity bias**: deep nets prefer simple fits among the many that fit the data.

*Why mutual k-NN, not CKA?*

- CKA showed only a weak trend (paper). CKA scores all pairs, incl. unrelated ones ("an orange & Bill Gates").
- k-NN scores only local structure. Paper also reports a CKA-k-NN hybrid (CKNNA).

*Counterarguments (listed in the paper)?*

- Modalities carry different information → convergence only for shared information. The math needs bijective observations.
- Not all domains converge yet (e.g., robotics).
- Sociological bias: researchers copy human-like designs → convergence by design, not by reality.
- Special-purpose systems can use shortcuts → no convergence.
- Alignment metrics are debated. Max cross-modal score = 0.16 of 1.0 → is the gap noise or real difference? Open.

*What does it tell us about AC/ASI?*

- If true: any capable mind trained on our world shares much of our similarity geometry → human–AI concept spaces become comparable.
- Supports structural (relational) comparisons of qualities across humans & AIs → [Quality Space](#quality-space). Contested.
- ASI: scale → convergence toward the same world model, not toward alien representations. Hypothesis.
```

&nbsp;

## Neural Population Geometry

### Neural Manifold
- **What**: Low-dim set in neuron-activity space traced by population responses as task variables vary {cite:p}`chung2021neural`.
    - Population activity = point in $\mathbb{R}^d$ (1 axis per neuron).
- **Why**: Single neurons are noisy & mixed-selective. Computation is visible at the population level.
    - Same description fits brains & ANNs → common language for both.
- **How**:
    1. Record population responses to stimuli w/ nuisance variation (pose, scale, position).
    2. Each object or condition → its own manifold. Manifold dim ≈ #varying latent variables ≪ $d$ → [Intrinsic Dimension](data.md#intrinsic-dimension).
    3. Track the geometry across stages: radius, dimension, curvature, separability.
    4. Processing hierarchy **untangles** class manifolds → linearly separable at the top.

```{attention} Q&A
:class: dropdown
*Why manifolds, not points?*

- Same object under nuisance variation → many responses → a set.
- Invariant recognition = separating whole manifolds, not single points.

*Which geometric phenomena does the review cover?*

- **Untangling**: object manifolds become separable along the visual hierarchy.
- **Capacity**: separability as a function of radius & dimension → [Manifold Capacity](#manifold-capacity).
- **Temporal straightening**: natural-video response trajectories become straighter → easier prediction.
- **Abstraction**: coding directions parallel across contexts (parallelism score) → cross-condition generalization.
- **Topology**: head direction on a ring → [Ring Attractor](topology.md#ring-attractor). Grid cells on a torus → [Toroidal Grid-Cell Manifold](topology.md#toroidal-grid-cell-manifold).
- **Dynamics**: motor-cortex trajectories untangled to avoid ambiguous flow.

*How does it link biological & artificial nets?*

- Geometry is architecture-agnostic → same measures on cortex & DNN layers.
- Capacity analysis shows categorical info emerges through layers via geometric change in ANNs for vision, speech & language prediction.

*How does it relate to the manifold hypothesis?*

- Data manifold ([Manifold Hypothesis](data.md#manifold-hypothesis)) → mapped by the network into neural manifolds.
- Same object, 2 spaces: input space vs activation space.

*What does it tell us about AC?*

- Candidate level for the neural correlate of content: geometry of population codes, not single cells.
- Same geometry in brain & AI ≠ same experience. Function vs causal-structure debate → [Qualia Space](#qualia-space).
```

&nbsp;

### Manifold Capacity
- **What**: Max #object manifolds per neuron that a linear readout can separate under random labels {cite:p}`chung2018classification`.
- **Why**: Measure untangling w/ 1 number that ties geometry to readout performance.
    - Points: Cover's capacity $P/d=2$. Manifolds are extended → capacity lower.
- **How**:
    1. Assign random binary labels to $P$ manifolds in $\mathbb{R}^d$.
    2. Ask: does a hyperplane separate every point of every manifold?
    3. Max load $\alpha=P/d$ w/ separation (as $d\to\infty$) = capacity $\alpha_M$.
    4. Mean-field theory: $\alpha_M$ ≈ function of manifold radius $R_M$ & dimension $D_M$.
    5. Apply per layer → capacity ⬆️ along layers = untangling.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $P$: #manifolds. Paper: $N$ neurons → $d$ here.
    - $\alpha=P/d$: Load.
    - $\kappa$: Required margin.
    - $Dt=\frac{1}{\sqrt{2\pi}}e^{-t^2/2}dt$: Gaussian measure.
    - $D$: Affine dimension of each manifold.
    - $R_M$: Manifold (anchor) radius, relative to the center norm.
    - $D_M$: Manifold (anchor) dimension.

Definition:
- Points (Gardner):

    $$
    \alpha_0^{-1}(\kappa)=\int_{-\infty}^{\kappa}Dt\,(t-\kappa)^2
    $$

    - $\kappa=0$ → $\alpha_0=2$ (Cover).

Properties:
- Bounds: $\alpha_0^{-1}(\kappa)\le\alpha_M^{-1}(\kappa)\le\alpha_0^{-1}(\kappa)+D$.
- Approximation (high-dim manifolds):

    $$
    \alpha_M(\kappa)\approx\alpha_0\left(\frac{\kappa+R_M\sqrt{D_M}}{\sqrt{1+R_M^2}}\right)
    $$

- Large effective margin $R_M\sqrt{D_M}$: $\alpha_M\approx\frac{1+R_M^{-2}}{D_M}$.
- Size $\to0$ → $\alpha_M(\kappa)\to\alpha_0(\kappa)$. Size $\to\infty$ → $\alpha_M(\kappa)\to\left[D+\alpha_0^{-1}(\kappa)\right]^{-1}$.
    - $\kappa=0$: limits $2$ & $1/(D+\tfrac12)$.
- High-dim manifold capacity ≈ capacity of $\ell_2$ balls w/ radius $R_M$ & dimension $D_M$. The manifolds need not be balls.
```

```{tip} Derivation
:class: dropdown
*Why is the capacity for points 2?*

1. Cover's counting: $P$ points in general position in $\mathbb{R}^d$ → #dichotomies separable by a hyperplane through the origin:

    $$
    C(P,d)=2\sum_{k=0}^{d-1}\binom{P-1}{k}
    $$

2. $P=2d$: $\sum_{k=0}^{d-1}\binom{2d-1}{k}=2^{2d-2}$ (half of a symmetric binomial sum).
3. → $C(2d,d)=2^{2d-1}$ = exactly half of all $2^{2d}$ dichotomies.
4. $d\to\infty$: fraction separable → sharp step from 1 to 0 at $\alpha=2$.
5. Gardner check: $\alpha_0^{-1}(0)=\int_{-\infty}^0t^2Dt=\frac12$ → $\alpha_0=2$.
```

```{attention} Q&A
:class: dropdown
*Why per neuron?*

- $\alpha=P/d$ is intensive → compares layers & networks of different widths.

*Why do radius & dimension trade off?*

- Effective margin $R_M\sqrt{D_M}$ → large-radius low-dim & small-radius high-dim manifolds can give the same capacity.

*Assumptions?*

- Random labels, $d\to\infty$ mean-field limit, random manifold orientations.
- Correlations between manifolds change capacity → later extensions.

*How does it relate to SVMs?*

- Max-margin hyperplane for whole manifolds = SVM w/ infinitely many points per class.
- Capacity = largest $P/d$ w/ zero training error.

*Use?*

- Score each layer → capacity ⬆️, $R_M$ ⬇️, $D_M$ ⬇️ through trained deep vision nets (Cohen et al., 2020).
- Compare models & brain areas w/ the same metric.
```

&nbsp;

## Geometry of Mind

### Universal Law of Generalization
- **What**: Probability that a response learned to 1 stimulus generalizes to another decays ≈ exponentially w/ their distance in psychological space {cite:p}`shepard1987toward`.
- **Why**: Find a law of mind that holds across stimuli, modalities & species, like a law of physics.
    - Raw physical stimulus distance gives no invariant curve. Psychological distance does.
- **How**:
    1. Measure generalization (confusion) probabilities between stimulus pairs.
    2. Nonmetric MDS → place stimuli so that generalization is a monotone function of distance.
    3. Observe: that function ≈ exponential decay.
    4. Explain: learner infers an unknown **consequential region** (natural kind) from 1 consequential stimulus.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $g(a,b)$: Probability that a response learned to $a$ generalizes to $b$.
    - $\ell$: Length scale of generalization.
    - $s$: Size of the consequential region.
    - $q(s)$: Prior on $s$.
    - $a_j,b_j$: Coordinate $j$ of stimuli $a,b$ in psychological space.

Law:

$$
g(a,b)\approx e^{-\delta(a,b)/\ell}
$$

Metric:
- Separable dimensions (e.g., size & orientation of a shape) → $L_1$: $\delta=\sum_j|a_j-b_j|$.
- Integral dimensions (e.g., hue, saturation & lightness of a color) → $L_2$: $\delta=\sqrt{\sum_j(a_j-b_j)^2}$.
- Uncorrelated region extents along dims → $L_1$. Correlated (or all orientations) → $L_2$.
```

```{tip} Derivation
:class: dropdown
*Why exponential? (1-D)*

1. Consequential region = interval of unknown size $s$ & uniform position.
2. Stimulus $a$ is known to be inside. Bigger regions contain $a$ more often → posterior $p(s)\propto s\,q(s)$.
3. Interval of size $s$ that contains $a$ also contains $b$ at distance $\delta$ w.p. $\max(0,1-\delta/s)$.
4. → $g(\delta)=\int_\delta^\infty p(s)\left(1-\frac{\delta}{s}\right)ds$.
5. Max-entropy prior w/ mean $\ell$: $q(s)=\frac1\ell e^{-s/\ell}$ → $p(s)=\frac{s}{\ell^2}e^{-s/\ell}$ (Erlang).
6. Substitute $u=s-\delta$:

    $$
    g(\delta)=\frac{1}{\ell^2}\int_\delta^\infty(s-\delta)e^{-s/\ell}ds=\frac{e^{-\delta/\ell}}{\ell^2}\int_0^\infty u\,e^{-u/\ell}du=e^{-\delta/\ell}
    $$

7. Other priors → nonincreasing convex curves, not necessarily exponential.
    - Fixed size $s_0$ → linear decay $1-\delta/s_0$.
```

```{attention} Q&A
:class: dropdown
*Is the law circular? MDS builds the space from generalization data.*

- ❌. MDS fixes distances only up to a monotone function. The **shape** of that function is free.
- Empirical claim: the recovered shape is exponential across many data sets.

*Why exponential & not Gaussian?*

- Max-entropy size prior → exactly exponential (Derivation). Other priors → convex decay, not necessarily exponential.
- Concave-up decay: generalization drops fast near the trained stimulus, then flattens.

*How does it relate to ML?*

- Laplace kernel = exponential decay w/ distance → same shape as the law.
- Gaussian RBF kernel = decay w/ squared distance → different shape.
- kNN & nearest-prototype classifiers use only distance order, not a decay law.
- Bayesian concept learning (Tenenbaum & Griffiths) extends the consequential-region argument w/ the size principle.

*What does it tell us about AC?*

- Shepard proposed the law may govern all sentient organisms ← evolutionary internalization of natural-kind structure.
- Testable on AI: does generalization in a model's own representation space decay exponentially? Open hypothesis.
```

&nbsp;

### Conceptual Space
- **What**: Geometric level of representation: quality dimensions grouped into domains, w/ natural concepts as convex regions {cite:p}`gardenfors2000conceptual`.
- **Why**: Symbolic (logic) & connectionist (neurons) levels both fail to explain similarity, prototypes, & fast concept learning.
    - Geometric middle level bridges them.
- **How**:
    1. **Quality dimensions**: hue, pitch, temperature, weight, ... each w/ a metric.
    2. **Domain**: set of integral dimensions (e.g., color = hue + saturation + brightness), separable from other domains.
    3. **Property** = convex region of 1 domain ("red").
    4. **Concept** = regions in several domains + salience weights + correlations ("apple": color, taste, shape, ...).
    5. Learn categories from prototypes → Voronoi tessellation → convex cells.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\mathcal{C}$: Region of a domain.
    - $\mathbf{p}_k$: Prototype of category $k$.
    - $V_k$: Voronoi cell of $\mathbf{p}_k$.

Definitions:
- Convex region: $\mathbf{a},\mathbf{b}\in\mathcal{C}$, $t\in[0,1]$ → $t\mathbf{a}+(1-t)\mathbf{b}\in\mathcal{C}$.
- **Criterion P**: natural property = convex region of a domain.
- Distance: Euclidean within a domain (integral dims). City-block across domains (separable) → matches [Universal Law of Generalization](#universal-law-of-generalization).
- Voronoi categorization:

    $$
    V_k=\left\{\mathbf{a}:\|\mathbf{a}-\mathbf{p}_k\|\le\|\mathbf{a}-\mathbf{p}_j\|\ \forall j\right\}
    $$
```

```{tip} Derivation
:class: dropdown
*Why are Voronoi cells convex?*

1. $\|\mathbf{a}-\mathbf{p}_k\|^2\le\|\mathbf{a}-\mathbf{p}_j\|^2$ ⇔ $2(\mathbf{p}_j-\mathbf{p}_k)^T\mathbf{a}\le\|\mathbf{p}_j\|^2-\|\mathbf{p}_k\|^2$.
2. Linear inequality in $\mathbf{a}$ → half-space.
3. $V_k$ = intersection of half-spaces → convex polyhedron, possibly unbounded.
4. → Prototype learning + Euclidean metric gives convex concepts automatically.
```

```{attention} Q&A
:class: dropdown
*Why convexity?*

- Betweenness: $a$ & $b$ are red → every color between them is red.
- Few examples suffice: prototypes + nearest-prototype rule → convex categories.
- Shared regions across speakers → communication works.

*Pros?*

- Explains typicality (distance to prototype), vagueness (region borders), & similarity-based induction.
- Same structure for perception & concepts.

*Cons?*

- Abstract & relational concepts ("justice", "uncle") lack obvious quality dimensions.
- Context changes regions: "red" in "red wine" vs "red hair".
- Choice of dimensions is a modeling decision.

*How does it relate to embeddings?*

- Learned embedding spaces ≈ data-driven conceptual spaces.
- Difference: embedding axes are not interpretable quality dimensions; convexity is not guaranteed.
- Linear concept directions ([Linear Representation Hypothesis](#linear-representation-hypothesis)) → half-space regions → convex.

*What does it tell us about AC?*

- Gives a format for "how a concept feels similar to another" w/o symbols or neurons.
- If AI embeddings form convex, human-aligned regions → shared conceptual geometry. Hypothesis.
```

&nbsp;

### Quality Space
- **What**: Similarity space of sensory qualities, built from discrimination data, in which each quality is individuated by its relations to all others {cite:p}`clark1993sensory`.
- **Why**: Qualia look private & ineffable. A relational structure makes them explainable & measurable.
- **How**:
    1. Measure which stimulus pairs a subject can & cannot discriminate (matching, just-noticeable differences).
    2. Derive relative similarities → embed in a low-dim space (e.g., color solid).
    3. Quality = position in that space. Same relations to all other qualities → same quality.
    4. Explain the space w/ psychophysics & neurophysiology (e.g., opponent color channels).

```{attention} Q&A
:class: dropdown
*Why not define qualities by matching alone?*

- Matching is non-transitive: $a$ matches $b$, $b$ matches $c$, $a$ does not match $c$.
- → Need the whole pattern of discriminations → a space.

*What is the AC relevance?*

- Relational account: quality = role in a discrimination structure → in principle measurable in any system.
- Test on AI: build the AI's quality space from its discriminations → compare w/ the human space via [RSA](#rsa).
- [Platonic convergence](#platonic-representation-hypothesis) predicts partial agreement (e.g., color). Hypothesis.

*Main objection?*

- Structure may not fix "what it is like": 2 systems w/ isomorphic spaces could differ in experience (hard problem).
- Intrinsic theories ([Qualia Space](#qualia-space), IIT) reject a purely relational, functional account.
```

&nbsp;

#### Quality-Space Theory
- **What**: Mental qualities = properties of perceptual states, individuated by their locations in a quality space built from perceptual discriminations {cite:p}`rosenthal2010think`.
- **Why**: Account for mental qualities independent of consciousness.
    - Perceptual discriminations occur unconsciously too → a theory of qualities must not rely on introspection.
- **How**:
    1. Map which perceptible properties a creature can discriminate → quality space of perceptible properties.
    2. Mental qualities = state properties that enable those discriminations → their space mirrors that space.
    3. Consciousness of a quality is added separately → [Higher-Order Thought Theory](../../ac/theories.md#higher-order-thought-theory-hott).

```{attention} Q&A
:class: dropdown
*Inverted spectrum?*

- Undetectable inversion needs a symmetry axis in the quality space.
- Qualities on both sides of such an axis would be indiscriminable → redundant → no undetectable inversion.

*How does it differ from Clark's view?*

- Same core: individuation by discrimination-based position.
- Rosenthal adds: qualities can occur w/o consciousness. Consciousness comes from a higher-order thought.

*What does it tell us about AC?*

- An AI w/ a human-like discrimination space would have human-like mental qualities by this theory.
- Whether they are conscious qualities is a separate question (HOT) → 2 separate tests. Contested.
```

&nbsp;

### Qualia Space
- **What**: IIT's space w/ 1 axis per system state, in which an experience (quale) = shape of the informational relationships its mechanisms specify {cite:p}`balduzzi2009qualia`.
    - Shape height = $\Phi$ = quantity of consciousness.
- **Why**: IIT measured how much consciousness ($\Phi$). Qualia space adds which experience (quality).
- **How**:
    1. List all submechanisms (subsets of connections) of a complex → lattice.
    2. Each submechanism + curr state → repertoire = distribution over past states → 1 point in Q.
    3. Engage connections 1 by 1 → q-arrows from the uniform repertoire up to the full repertoire.
    4. All q-arrows → shape = quale. Similar experiences ⇔ similar shapes.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $n_e$: #binary elements in the complex.
    - $\mathbf{s}_1$: Curr state.
    - $\mathcal{M}$: Submechanism (subset of connections). $\bot$ = none, $\top$ = all.
    - $p_\mathcal{M}$: Repertoire = distribution over previous states given $\mathcal{M}$ & $\mathbf{s}_1$.

Definition:
- Q-space: $\mathbb{R}^{2^{n_e}}$, 1 axis per state, coordinates = state probabilities.
- Points: $p_\mathcal{M}$ for every $\mathcal{M}$. $p_\bot$ = uniform (max entropy).
- q-arrow $\mathcal{M}\to\mathcal{M}\cup\mathcal{M}'$: length = relative entropy between the 2 repertoires →

    $$
    D_{KL}\left(p_{\mathcal{M}\cup\mathcal{M}'}\,\|\,p_\mathcal{M}\right)
    $$

- Quale = set of all q-arrows (a polytope-like shape).
- $\Phi$ = length of the q-arrow from the minimum-information-partition repertoire to $p_\top$.
```

```{attention} Q&A
:class: dropdown
*Corollaries the paper states?*

- Quale depends on mechanism & state → same activity, different wiring → different qualia.
- Inactive (but not inactivated) elements also shape the quale.
- Modalities & submodalities = subshapes.
- "Redness of red" needs the whole quale, not only the local mechanism.

*Is this the current IIT?*

- ❌. Formulation of IIT 2.0 (2009).
- IIT 4.0: quality = cause–effect structure (Φ-structure) of distinctions & relations → [IIT](../../ac/theories.md#integrated-information-theory-iit).
- Same idea: experience = intrinsic causal geometry.

*How does it relate to information geometry?*

- Repertoires = points on the probability simplex. KL measures q-arrows → [Integrated Information as Projection](info.md#integrated-information-as-projection), [KL Divergence](../info.md#kl-divergence).

*Quality Space vs Qualia Space?*

- Quality space: relational, from behavior (discriminations) → functionalist → AI with the same discriminations shares qualities.
- Qualia space: intrinsic, from causal mechanism → same function on different hardware can differ → IIT claims standard digital computers have ≈ no consciousness.
- → Opposite predictions for AC.

*What does it tell us about AC?*

- Makes "quality of experience" a math object → in principle computable.
- Cost: #states $=2^{n_e}$ → intractable beyond tiny systems.
- IIT is contested; its axioms & predictions are disputed → treat as hypothesis.
```

&nbsp;
