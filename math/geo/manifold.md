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
# Differential & Riemannian Geometry
Curved spaces that look flat locally: how to measure lengths, straight lines & curvature on them, how to optimize on them, & hyperbolic space for hierarchies.
Study notes from {cite:t}`smooth_manifolds` & {cite:t}`riemannian_manifolds`.
Shape w/o distances → [Topology](topology.md). Metrics on probability models → [Information Geometry](info.md).

Notations:
- $\mathcal{M}$: Smooth manifold.
- $n$: Dimension of $\mathcal{M}$.
- $D$: Dimension of an ambient Euclidean space $\mathbb{R}^D$.
- $p$: Point on $\mathcal{M}$.
- $q$: Second point on $\mathcal{M}$.
- $\mathbf{z}\in\mathbb{R}^n$: Local coordinates of a point (e.g., latent code).
- $T_p\mathcal{M}$: Tangent space at $p$.
- $\mathbf{v}$: Tangent vector.
- $\mathbf{w}$: Second tangent vector.
- $\gamma:[0,1]\to\mathcal{M}$: Smooth curve.
- $\dot\gamma$: Velocity of $\gamma$.
- $g_p$: Riemannian metric (inner product on $T_p\mathcal{M}$).
- $G(\mathbf{z})\in\mathbb{R}^{n\times n}$: Metric tensor in coordinates (SPD matrix).
- $g_{ij}$: Entry of $G$.
- $g^{ij}$: Entry of $G^{-1}$.
- $\|\mathbf{v}\|_g$: Riemannian norm $\sqrt{g_p(\mathbf{v},\mathbf{v})}$.
- $d(p,q)$: Riemannian (geodesic) distance.
- $\exp_p$: Exponential map at $p$.
- $\log_p$: Logarithmic map at $p$ (local inverse of $\exp_p$).
- $\Gamma^k_{ij}$: Christoffel symbols.
- $K$: Gaussian or sectional curvature.
- $f$: Smooth map between manifolds (e.g., decoder).
- $J_f$: Jacobian of $f$ in coordinates.
- $\mathcal{L}:\mathcal{M}\to\mathbb{R}$: Objective.
- $\eta$: Learning rate.
- $\mathbb{B}^n$: Poincaré ball (open unit ball w/ hyperbolic metric).
- $\lambda_{\mathbf{x}}$: Conformal factor of $\mathbb{B}^n$ at $\mathbf{x}$.
- $\mathbf{x}$: Point of $\mathbb{B}^n$.
- $\mathbf{y}$: Second point of $\mathbb{B}^n$.

- **Overrides**: $n$ = manifold dimension, ❌#features. $i,j,k,l$ = coordinate indices, ❌sample/feature/class idx. $\mathbf{x},\mathbf{y}$ = points of $\mathbb{B}^n$, ❌model input/output. $K$ = scalar curvature value, ❌matrix.
- **Conventions**: Einstein summation over a repeated upper & lower index ($\Gamma^k_{ij}\dot z^i\dot z^j=\sum_{i,j}\Gamma^k_{ij}\dot z^i\dot z^j$). $\|\cdot\|$ = Euclidean norm. Hyperbolic curvature $=-1$ unless stated.

&nbsp;

## Manifolds

### Smooth Manifold
- **What**: Space that looks like $\mathbb{R}^n$ near every point, w/ smooth changes of coordinates.
- **Why**: Calculus needs coordinates. Spheres, rotations, orthonormal frames & data manifolds have no single global coordinate system.
    - → Glue local coordinate patches; require smooth overlaps → derivatives are defined everywhere & do not depend on the patch.
- **How**:
    1. Cover $\mathcal{M}$ w/ open sets $U_\alpha$.
    2. **Chart**: homeomorphism $\varphi_\alpha:U_\alpha\to\varphi_\alpha(U_\alpha)\subseteq\mathbb{R}^n$ → local coordinates $\mathbf{z}$.
    3. **Atlas**: set of charts that covers $\mathcal{M}$.
    4. Overlap of two charts → **transition map** $\varphi_\beta\circ\varphi_\alpha^{-1}$ must be $C^\infty$.
    5. → "$f$ is smooth" = "$f$ in coordinates is $C^\infty$" gives the same answer in every chart.

```{note} Math
:class: dropdown
Notations:
- $U_\alpha$: Open subset of $\mathcal{M}$.
- $\varphi_\alpha$: Chart on $U_\alpha$.
- $\mathcal{N}$: Second smooth manifold.

Definition:
- **Topological manifold**: Hausdorff, second-countable space; every point has a neighborhood homeomorphic to an open subset of $\mathbb{R}^n$.
- **Smooth atlas**: charts $\{(U_\alpha,\varphi_\alpha)\}$ w/ $\bigcup_\alpha U_\alpha=\mathcal{M}$ & every transition map smooth:

$$
\varphi_\beta\circ\varphi_\alpha^{-1}:\varphi_\alpha(U_\alpha\cap U_\beta)\to\varphi_\beta(U_\alpha\cap U_\beta)\ \text{ is }C^\infty
$$

- **Smooth manifold**: topological manifold + maximal smooth atlas (smooth structure).

Properties:
- $f:\mathcal{M}\to\mathcal{N}$ smooth ⇔ $\psi\circ f\circ\varphi^{-1}$ is $C^\infty$ for all charts $\varphi$ of $\mathcal{M}$ & $\psi$ of $\mathcal{N}$.
- **Diffeomorphism**: smooth bijection w/ smooth inverse. Same manifold up to diffeomorphism.
- **Regular level set theorem**: $F:\mathbb{R}^D\to\mathbb{R}^{D-n}$ smooth, $dF$ surjective at every point of $F^{-1}(c)$ → $F^{-1}(c)$ is an embedded $n$-dim submanifold of $\mathbb{R}^D$.
    - $S^{D-1}=F^{-1}(1)$ w/ $F(\mathbf{a})=\|\mathbf{a}\|^2$.
- **Whitney embedding**: every smooth $n$-manifold embeds in $\mathbb{R}^{2n}$. Easier version (proved by Lee): $\mathbb{R}^{2n+1}$.
    - → Abstract manifolds = embedded submanifolds, w/o loss of generality.
```

```{note} Example
:class: dropdown
- $S^n$ w/ 2 stereographic charts (from north & south poles). Transition map on $\mathbb{R}^n\setminus\{\mathbf{0}\}$: $\mathbf{u}\mapsto\mathbf{u}/\|\mathbf{u}\|^2$ → smooth.
- ✅ Torus, $\mathrm{SO}(3)$, open subsets of $\mathbb{R}^n$, graphs of smooth functions.
- ❌ Two crossing lines "X": the crossing point has no neighborhood homeomorphic to $\mathbb{R}$.
```

```{dropdown} Table: Manifolds in ML
| Manifold | Points | Dim | ML use |
|:--|:--|:--|:--|
| Sphere $S^{D-1}$ | Unit vectors in $\mathbb{R}^D$ | $D-1$ | Normalized embeddings, directions |
| $\mathrm{SO}(3)$ | 3-D rotation matrices | $3$ | Poses, robotics, 3-D vision |
| Stiefel $\mathrm{St}(D,r)$ | $D\times r$ matrices w/ orthonormal columns | $Dr-\frac{r(r+1)}{2}$ | Orthogonal weights, PCA frames |
| Grassmann $\mathrm{Gr}(D,r)$ | $r$-dim subspaces of $\mathbb{R}^D$ | $r(D-r)$ | Subspace tracking, PCA |
| $\mathrm{SPD}(n)$ | $n\times n$ SPD matrices | $\frac{n(n+1)}{2}$ | Covariances, kernels, diffusion tensors |
| Hyperbolic $\mathbb{H}^n$ | See [Hyperbolic Space](#hyperbolic-space) | $n$ | Hierarchies |
```

```{attention} Q&A
:class: dropdown
*Why Hausdorff & second-countable?*

- Exclude pathologies. Line w/ two origins: locally Euclidean, but the two origins have no disjoint neighborhoods → limits not unique.
- Second-countable → partitions of unity exist → glue local objects (metrics, integrals) into global ones.

*Why can one chart not cover the sphere?*

- Chart image = open subset of $\mathbb{R}^n$. $S^n$ compact → image compact.
- Nonempty subset of $\mathbb{R}^n$ that is open & compact (closed) → all of $\mathbb{R}^n$ (connected) → not compact. Contradiction.

*Why are Euler angles bad targets for a rotation-regression net?*

- $\mathrm{SO}(3)$ = compact 3-manifold → no continuous injective map into $\mathbb{R}^3$ (invariance of domain).
- → Every 3-number parametrization has a singularity or a discontinuity. Gimbal lock = chart singularity.
- Quaternions: continuous, but 2-to-1 ($\mathbf{q}$ & $-\mathbf{q}$ give the same rotation).

*Topological vs smooth manifold?*

- Topological: continuity only ([Homeomorphism](topology.md#homeomorphism)). Smooth: derivatives too.
- Some topological manifolds carry several non-diffeomorphic smooth structures (exotic $S^7$, exotic $\mathbb{R}^4$).

*How does this relate to the manifold hypothesis?*

- [Manifold Hypothesis](data.md#manifold-hypothesis): real data concentrate near a low-dim manifold in $\mathbb{R}^D$.
- Empirical claim about data, not a theorem. Data manifolds may be noisy, non-smooth, or of varying dimension.
```

&nbsp;

### Tangent Space
- **What**: $n$-dim vector space of all velocities of curves through $p$.
- **Why**: $\mathcal{M}$ is curved → no addition or scaling of points. Derivatives need a linear space → linearize $\mathcal{M}$ at $p$.
- **How**:
    1. Take curves $\gamma$ w/ $\gamma(0)=p$.
    2. In a chart: velocity = $\frac{d}{dt}\mathbf{z}(t)\big|_{t=0}\in\mathbb{R}^n$.
    3. Curves w/ equal coordinate velocity → same tangent vector.
    4. $\mathcal{M}\subset\mathbb{R}^D$ embedded → $T_p\mathcal{M}$ = $n$-dim linear subspace of $\mathbb{R}^D$ that touches $\mathcal{M}$ at $p$.
    5. Smooth map $f$ → **differential** $df_p$ maps velocities to velocities. In coordinates: $df_p=J_f$.

```{note} Math
:class: dropdown
Notations:
- $\partial_i$: Coordinate basis vector $\partial/\partial z^i\big|_p$.
- $\mathcal{N}$: Second smooth manifold.
- $h$: Smooth map out of $\mathcal{N}$.

Definition:

$$
T_p\mathcal{M}=\{\dot\gamma(0):\gamma \text{ smooth},\ \gamma(0)=p\},\qquad \mathbf{v}=v^i\partial_i
$$

- $\dim T_p\mathcal{M}=n$.
- Level set $\mathcal{M}=F^{-1}(c)$: $T_p\mathcal{M}=\ker dF_p$.
    - Sphere: $T_{\mathbf{p}}S^{D-1}=\{\mathbf{v}\in\mathbb{R}^D:\mathbf{p}^T\mathbf{v}=0\}$.

Differential (pushforward):

$$
df_p:T_p\mathcal{M}\to T_{f(p)}\mathcal{N},\qquad df_p(\dot\gamma(0))=\frac{d}{dt}f(\gamma(t))\Big|_{t=0}
$$

- Coordinates: $\mathbf{v}\mapsto J_f(\mathbf{z})\,\mathbf{v}$. See [Jacobian](../linalg.md#jacobian).
- Chain rule: $d(h\circ f)_p=dh_{f(p)}\circ df_p$.
- Change of chart $\mathbf{z}'=\phi(\mathbf{z})$: components transform as $\mathbf{v}'=J_\phi\mathbf{v}$.

Bundles:
- **Tangent bundle** $T\mathcal{M}=\{(p,\mathbf{v}):p\in\mathcal{M},\ \mathbf{v}\in T_p\mathcal{M}\}$: smooth manifold of dim $2n$.
- **Vector field**: smooth choice $p\mapsto\mathbf{v}(p)\in T_p\mathcal{M}$ → ODE on $\mathcal{M}$.
- **Cotangent space** $T_p^*\mathcal{M}$: linear functionals on $T_p\mathcal{M}$.
    - Scalar $\mathcal{L}$ → $d\mathcal{L}_p\in T_p^*\mathcal{M}$: covector, not vector.
```

```{attention} Q&A
:class: dropdown
*Why is the derivative of a loss a covector, not a vector?*

- $d\mathcal{L}_p$ takes a direction & returns a rate → linear functional.
- Covector → direction (gradient) needs an inner product → metric choice → [Riemannian Gradient Descent](#riemannian-gradient-descent).
- Euclidean GD silently picks $G=I$ in the current coordinates → reparametrize → different "steepest" direction.

*How does this relate to autodiff?*

- Forward mode = JVP $\mathbf{v}\mapsto J_f\mathbf{v}$ = pushforward of tangent vectors.
- Reverse mode (backprop) = VJP $\mathbf{u}^T\mapsto\mathbf{u}^TJ_f$ = pullback of covectors.

*Is $T_p\mathcal{M}$ part of $\mathcal{M}$?*

- ❌. Separate vector space attached at $p$.
- Embedded case: $p+T_p\mathcal{M}$ touches $\mathcal{M}$ at $p$ only to first order → a step $p+\mathbf{v}$ leaves $\mathcal{M}$ → need [Exponential Map](#exponential-map) or [Retraction](#retraction).

*Why does the tangent bundle matter?*

- State of a mechanical system = (position, velocity) $\in T\mathcal{M}$.
- Vector field on $\mathcal{M}$ = dynamics constrained to $\mathcal{M}$.
```

&nbsp;

## Riemannian Geometry

### Riemannian Metric
- **What**: Smoothly varying [inner product](../linalg.md#inner-product) $g_p$ on each tangent space.
- **Why**: Smooth structure gives directions, not sizes. Lengths, angles, distances, volumes & gradients all need an inner product.
- **How**:
    1. At each $p$, pick an inner product on $T_p\mathcal{M}$.
    2. Coordinates: [SPD matrix](../linalg.md#positive-definite-matrix) $G(\mathbf{z})$ → $g_p(\mathbf{v},\mathbf{w})=\mathbf{v}^TG(\mathbf{z})\mathbf{w}$.
    3. Curve length = integral of speed $\sqrt{\dot{\mathbf{z}}^TG\dot{\mathbf{z}}}$.
    4. Distance = length of the shortest curve.

```{note} Math
:class: dropdown
Notations:
- $\theta$: Angle between $\mathbf{v}$ & $\mathbf{w}$.
- $dV$: Riemannian volume element.

Definition: symmetric, positive-definite bilinear form $g_p:T_p\mathcal{M}\times T_p\mathcal{M}\to\mathbb{R}$, smooth in $p$.

$$
g_p(\mathbf{v},\mathbf{w})=g_{ij}(\mathbf{z})\,v^iw^j=\mathbf{v}^TG(\mathbf{z})\,\mathbf{w}
$$

Induced quantities:

$$
\begin{align*}
L(\gamma)&=\int_0^1\sqrt{\dot{\mathbf{z}}(t)^TG(\mathbf{z}(t))\,\dot{\mathbf{z}}(t)}\,dt\\
d(p,q)&=\inf_{\gamma:\ \gamma(0)=p,\ \gamma(1)=q}L(\gamma)\\
\cos\theta&=\frac{g_p(\mathbf{v},\mathbf{w})}{\|\mathbf{v}\|_g\|\mathbf{w}\|_g}\\
dV&=\sqrt{\det G(\mathbf{z})}\,d\mathbf{z}
\end{align*}
$$

Properties:
- Change of chart $\mathbf{z}=\phi(\mathbf{z}')$, $J=\partial\mathbf{z}/\partial\mathbf{z}'$: $G'(\mathbf{z}')=J^TG(\mathbf{z})J$ → lengths, angles & volumes are chart-independent.
- Embedded $\mathcal{M}\subset\mathbb{R}^D$: **induced metric** = Euclidean dot product restricted to $T_p\mathcal{M}$.
- Every smooth manifold admits a Riemannian metric (partition of unity).
- **Isometry**: diffeomorphism that preserves $g$ → preserves lengths, distances, geodesics, curvature.
```

```{tip} Derivation
:class: dropdown
*Why is the volume element $\sqrt{\det G}$?*

1. A chart maps a small coordinate cube $\prod_i dz^i$ to a parallelepiped spanned by $\partial_1dz^1,\dots,\partial_ndz^n$ in $T_p\mathcal{M}$.
2. [Gram matrix](../linalg.md#gram-matrix) of $\partial_1,\dots,\partial_n$ under $g$ = $G$.
3. Volume of a parallelepiped = $\sqrt{\det(\text{Gram matrix})}$ → $dV=\sqrt{\det G}\,d\mathbf{z}$.

*Why is length chart-independent?*

1. $\mathbf{z}=\phi(\mathbf{z}')$ → $\dot{\mathbf{z}}=J\dot{\mathbf{z}}'$.
2. $\dot{\mathbf{z}}^TG\dot{\mathbf{z}}=\dot{\mathbf{z}}'^T(J^TGJ)\dot{\mathbf{z}}'=\dot{\mathbf{z}}'^TG'\dot{\mathbf{z}}'$ → same speed → same length.
```

```{note} Example
:class: dropdown
Polar coordinates on $\mathbb{R}^2$: $\mathbf{z}=(r,\vartheta)$, point $=(r\cos\vartheta,\ r\sin\vartheta)$.

$$
J=\begin{bmatrix}\cos\vartheta&-r\sin\vartheta\\ \sin\vartheta&r\cos\vartheta\end{bmatrix},\qquad G=J^TJ=\begin{bmatrix}1&0\\0&r^2\end{bmatrix}
$$

- $\sqrt{\det G}=r$ → $dA=r\,dr\,d\vartheta$.
- Circle $r=R$: $L=\int_0^{2\pi}\sqrt{R^2}\,d\vartheta=2\pi R$.
- Same flat plane, non-constant $G$ → $G$ alone does not show curvature.
```

```{attention} Q&A
:class: dropdown
*Is the metric determined by the manifold?*

- ❌. Extra choice. Same $\mathcal{M}$, different $g$ → different distances, geodesics, curvature.
- ML: the choice = which differences count.
- Statistical models: [Fisher-Rao Metric](info.md#fisher-rao-metric) is the canonical choice.

*How does Mahalanobis distance fit?*

- Constant metric $G=\Sigma^{-1}$ on $\mathbb{R}^n$ → flat. Geodesics = straight lines.
- $d(\mathbf{a},\mathbf{b})=\sqrt{(\mathbf{a}-\mathbf{b})^T\Sigma^{-1}(\mathbf{a}-\mathbf{b})}$.

*Why must $G$ be positive definite?*

- PSD only → nonzero vectors w/ zero length → degenerate (e.g., pullback through a rank-deficient Jacobian).
- Indefinite → pseudo-Riemannian (Lorentzian spacetime in GR). Not a distance.

*Intrinsic vs extrinsic?*

- Intrinsic: only $g$. Extrinsic: how $\mathcal{M}$ sits in $\mathbb{R}^D$.
- Nash embedding: every Riemannian manifold embeds isometrically in some $\mathbb{R}^D$ → both views are equivalent in principle. $D$ can be large.

*What does it tell us about AC?*

- Hypothesis: similarity of experiences ↔ distance in a perceptual [Quality Space](repr.md#quality-space). Its metric need not be Euclidean.
- Contested: no agreed method to measure the metric of experience; similarity judgments give only indirect evidence.
```

&nbsp;

#### Pullback Metric
- **What**: Metric induced on the latent space by a smooth map $f$: $G(\mathbf{z})=J_f(\mathbf{z})^TJ_f(\mathbf{z})$. {cite:p}`arvanitidis2018latent`
- **Why**: Euclidean latent distances ignore the decoder → distorted view of data space.
    - Decoder stretches & compresses regions → equal latent steps ≠ equal output changes.
    - Measure a latent curve by the length of its decoded curve → latent geometry = data-space geometry.
- **How**:
    1. Decode the curve: $\mathbf{z}(t)\mapsto f(\mathbf{z}(t))\in\mathbb{R}^D$.
    2. Output velocity: $J_f\dot{\mathbf{z}}$ (chain rule).
    3. Squared speed: $\dot{\mathbf{z}}^TJ_f^TJ_f\dot{\mathbf{z}}$ → $G=J_f^TJ_f$.
    4. Stochastic decoder (VAE): expected metric adds a std-Jacobian term → path cost ⬆️ where the std changes fast (e.g., at the data boundary).

```{note} Math
:class: dropdown
Notations:
- $\mathcal{Z}\subseteq\mathbb{R}^n$: Latent space.
- $\boldsymbol{\mu}(\mathbf{z})\in\mathbb{R}^D$: Decoder mean.
- $\boldsymbol{\sigma}(\mathbf{z})\in\mathbb{R}^D$: Decoder std.
- $\boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},I_D)$: Noise.

Definition ($f:\mathcal{Z}\to\mathbb{R}^D$ smooth, $J_f\in\mathbb{R}^{D\times n}$):

$$
G(\mathbf{z})=J_f(\mathbf{z})^TJ_f(\mathbf{z})
$$

- Always PSD. PD ⇔ $\operatorname{rank}J_f=n$ ($f$ is an immersion).
- $L_G(\mathbf{z}(\cdot))=$ Euclidean length of $f(\mathbf{z}(\cdot))$.
- $\sqrt{\det(J_f^TJ_f)}$: local volume magnification of the decoder.

Stochastic generator $f(\mathbf{z})=\boldsymbol{\mu}(\mathbf{z})+\boldsymbol{\sigma}(\mathbf{z})\odot\boldsymbol{\epsilon}$, expected metric:

$$
\bar{G}(\mathbf{z})=\mathbb{E}_{\boldsymbol{\epsilon}}\left[J_f^TJ_f\right]=J_{\boldsymbol{\mu}}^TJ_{\boldsymbol{\mu}}+J_{\boldsymbol{\sigma}}^TJ_{\boldsymbol{\sigma}}
$$

- Paper models the precision $1/\boldsymbol{\sigma}^2$ w/ an RBF network → RBF precision controls decoder uncertainty.
- $\bar{G}$ depends on the gradient $J_{\boldsymbol{\sigma}}$, not on the size of $\boldsymbol{\sigma}$ → std gradients (at the data boundary) raise path costs.
- Large uncertainty alone does not guarantee a large metric: far from the RBF centers the precision reaches its floor → $J_{\boldsymbol{\sigma}}\to0$.
```

```{tip} Derivation
:class: dropdown
*Why does the expected metric split into a mean & a std term?*

1. $J_f=J_{\boldsymbol{\mu}}+\operatorname{diag}(\boldsymbol{\epsilon})J_{\boldsymbol{\sigma}}$.
2. $J_f^TJ_f=J_{\boldsymbol{\mu}}^TJ_{\boldsymbol{\mu}}+J_{\boldsymbol{\mu}}^T\operatorname{diag}(\boldsymbol{\epsilon})J_{\boldsymbol{\sigma}}+J_{\boldsymbol{\sigma}}^T\operatorname{diag}(\boldsymbol{\epsilon})J_{\boldsymbol{\mu}}+J_{\boldsymbol{\sigma}}^T\operatorname{diag}(\boldsymbol{\epsilon})^2J_{\boldsymbol{\sigma}}$.
3. $\mathbb{E}[\boldsymbol{\epsilon}]=\mathbf{0}$ → cross terms vanish.
4. $\mathbb{E}[\operatorname{diag}(\boldsymbol{\epsilon})^2]=I_D$ → last term $=J_{\boldsymbol{\sigma}}^TJ_{\boldsymbol{\sigma}}$.
```

```{note} Example
:class: dropdown
Paraboloid decoder $f(\mathbf{z})=(z_1,\ z_2,\ \|\mathbf{z}\|^2)$:

$$
J_f=\begin{bmatrix}1&0\\0&1\\2z_1&2z_2\end{bmatrix},\qquad G(\mathbf{z})=I+4\mathbf{z}\mathbf{z}^T
$$

- $\mathbf{z}=\mathbf{0}$: $G=I$ → latent ≈ data space.
- $\mathbf{z}=(1,0)$: $G=\operatorname{diag}(5,1)$ → a radial latent step costs $\sqrt5\times$ a tangential one.
```

````{important} Code
:class: dropdown
```python
import torch
from torch.autograd.functional import jacobian

class PullbackMetric:
    def __init__(self, f):
        self.f = f  ## decoder R^n -> R^D

    def G(self, z):
        J = jacobian(self.f, z)  ## (D, n)
        return J.T @ J           ## (n, n): G(z) = J^T J

    def curve_length(self, zs):
        ## zs: (T, n) latent polyline -> length of the decoded polyline
        xs = self.f(zs)
        return (xs[1:] - xs[:-1]).norm(dim=-1).sum()

## Example: paraboloid decoder f(z) = (z1, z2, ||z||^2)
f = lambda z: torch.cat([z, (z ** 2).sum(-1, keepdim=True)], dim=-1)
M = PullbackMetric(f)
print(M.G(torch.tensor([1., 0.])))  ## tensor([[5., 0.], [0., 1.]]) = I + 4 z z^T
line = torch.linspace(0, 1, 101)[:, None] * torch.tensor([1., 0.])
print(M.curve_length(line))         ## ≈ 1.479 > 1 = Euclidean latent length
```
````

```{attention} Q&A
:class: dropdown
*Why is linear interpolation in latent space misleading?*

- Straight latent line ≠ geodesic under $G$ → decoded path can cross high-magnification regions or leave the data.
- Fix: interpolate along a [Geodesic](#geodesic) of $G$ (or $\bar{G}$).

*Is the latent space curved?*

- Under $G$: in general ✅ (nonzero [Curvature](#curvature)). The Euclidean view hides it.

*Why add decoder uncertainty?*

- Deterministic decoder → $G$ ignores where data exist → geodesics can cut through regions w/o data.
- Std term $J_{\boldsymbol{\sigma}}^TJ_{\boldsymbol{\sigma}}$ → path cost ⬆️ where uncertainty changes fast (data boundary) → geodesics tend to stay near the data.
- No guarantee: large but flat uncertainty → $J_{\boldsymbol{\sigma}}\approx0$ → no extra cost.

*How does this relate to Isomap?*

- Same goal: distances along the data manifold.
- [Isomap](data.md#isomap): approximate geodesic distances w/ shortest paths in a kNN graph; no decoder.
- Pullback: exact metric from a trained decoder's Jacobian.

*How does this relate to Gauss-Newton & Fisher?*

- Loss $\frac12\|f(\boldsymbol{\theta})-\mathbf{t}\|^2$ ($\mathbf{t}$: target) → Gauss-Newton matrix $J_f^TJ_f$ = pullback of the Euclidean output metric onto parameter space.
- Fisher metric = pullback of the Fisher-Rao metric on distributions through $\boldsymbol{\theta}\mapsto p_{\boldsymbol{\theta}}$ → [Natural Gradient](info.md#natural-gradient).

*Cons?*

- One $D\times n$ Jacobian per point → expensive for large $n$.
- Geodesics need an ODE or optimization (no closed form).
- Metric quality = decoder quality.

*What does it tell us about AC?*

- World models plan & compare states in latent space.
- Euclidean latent distances can misrank state similarity & plan paths through states the decoder never saw → the decoder's geometry matters for any latent-space reasoning.
```

&nbsp;

### Geodesic
- **What**: Curve w/ zero intrinsic acceleration; locally length-minimizing.
- **Why**: "Straight line" of a curved space → shortest paths, distances, interpolation, averages.
- **How**:
    1. Euclidean: straight line = zero acceleration.
    2. Curved space or curved coordinates: coordinate acceleration $\ddot{\mathbf{z}}\ne0$ even for "straight" motion → correct it w/ Christoffel symbols.
    3. Initial value: start at $p$ w/ velocity $\mathbf{v}$ → solve the 2nd-order geodesic ODE.
    4. Boundary value: fix $p$ & $q$ → minimize curve energy.

```{note} Math
:class: dropdown
Notations:
- $E(\gamma)$: Curve energy.
- $\partial_l$: $\partial/\partial z^l$.

Geodesic equation:

$$
\ddot z^k+\Gamma^k_{ij}\,\dot z^i\dot z^j=0,\qquad \Gamma^k_{ij}=\frac12g^{kl}\left(\partial_ig_{jl}+\partial_jg_{il}-\partial_lg_{ij}\right)
$$

Energy:

$$
E(\gamma)=\frac12\int_0^1g_{\gamma(t)}(\dot\gamma,\dot\gamma)\,dt
$$

Properties:
- Critical points of $E$ w/ fixed endpoints = geodesics.
- $L(\gamma)^2\le2E(\gamma)$; equality ⇔ constant speed → minimizers of $E$ = constant-speed minimizers of $L$.
- Geodesics have constant speed $\|\dot\gamma\|_g$.
- Minimizing only locally: past a conjugate/cut point a geodesic stops being shortest.
- **Hopf–Rinow**: $(\mathcal{M},d)$ complete ⇔ every geodesic extends to all $t\in\mathbb{R}$. Complete → every $p,q$ in the same connected component joined by a minimizing geodesic.
```

```{tip} Derivation
:class: dropdown
*Why does minimizing energy give the geodesic equation?*

1. Lagrangian $\frac12g_{ij}(\mathbf{z})\dot z^i\dot z^j$ → Euler–Lagrange for coordinate $k$:

$$
\frac{d}{dt}\left(g_{kj}\dot z^j\right)-\frac12\partial_kg_{ij}\dot z^i\dot z^j=0
$$

2. Expand: $g_{kj}\ddot z^j+\partial_ig_{kj}\dot z^i\dot z^j-\frac12\partial_kg_{ij}\dot z^i\dot z^j=0$.
3. Symmetrize in $i,j$: $\partial_ig_{kj}\dot z^i\dot z^j=\frac12(\partial_ig_{kj}+\partial_jg_{ki})\dot z^i\dot z^j$.
4. → $g_{kj}\ddot z^j+\frac12(\partial_ig_{kj}+\partial_jg_{ki}-\partial_kg_{ij})\dot z^i\dot z^j=0$.
5. Multiply by $g^{lk}$ → $\ddot z^l+\Gamma^l_{ij}\dot z^i\dot z^j=0$.

*Why $L^2\le2E$?*

1. Cauchy–Schwarz on $[0,1]$: $\left(\int_0^1\|\dot\gamma\|_g\cdot1\,dt\right)^2\le\int_0^1\|\dot\gamma\|_g^2dt\cdot\int_0^11\,dt$.
2. → $L^2\le2E$. Equality ⇔ $\|\dot\gamma\|_g$ constant.
```

```{note} Example
:class: dropdown
Flat plane in polar coordinates, $G=\operatorname{diag}(1,r^2)$:
- $\Gamma^r_{\vartheta\vartheta}=-r$, $\Gamma^\vartheta_{r\vartheta}=\Gamma^\vartheta_{\vartheta r}=1/r$, all others $0$.
- Geodesic equations:

    $$
    \ddot r-r\dot\vartheta^2=0,\qquad \ddot\vartheta+\frac{2}{r}\dot r\dot\vartheta=0
    $$

- Solutions = straight lines (e.g., $r=1/\cos\vartheta$ for $x=1$). Christoffel symbols $\ne0$, yet the plane is flat.

Unit sphere: geodesics = great circles. Two non-antipodal points → 2 geodesic arcs; only the shorter one minimizes.
```

````{important} Code
:class: dropdown
```python
import torch

## Reuses decoder f from Pullback Metric (this page)
class DiscreteGeodesic:
    def __init__(self, f, z0, z1, T=16):
        self.f, self.z0, self.z1 = f, z0, z1
        t = torch.linspace(0, 1, T + 1)[1:-1, None]           ## interior times
        self.inner = torch.nn.Parameter((1 - t) * z0 + t * z1)  ## init: latent straight line

    def energy(self):
        zs = torch.cat([self.z0[None], self.inner, self.z1[None]])
        xs = self.f(zs)
        T = len(zs) - 1
        ## E = 1/2 ∫||dx/dt||^2 dt ≈ 1/2 Σ ||Δx||^2 / Δt, Δt = 1/T
        return 0.5 * T * ((xs[1:] - xs[:-1]) ** 2).sum()

    def fit(self, steps=500, lr=1e-2):
        opt = torch.optim.Adam([self.inner], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            loss = self.energy()
            loss.backward()
            opt.step()
        return self.inner.detach()

## Example
geo = DiscreteGeodesic(f, torch.tensor([-1., .5]), torch.tensor([1., .5]))
E0 = geo.energy().item()
path = geo.fit()            ## (15, 2) interior latent points
print(geo.energy().item() < E0)  ## True: the latent straight line is not a geodesic
```
````

```{attention} Q&A
:class: dropdown
*Why minimize energy instead of length?*

- Length is invariant to reparametrization → infinitely many minimizers; $\sqrt{\cdot}$ non-smooth at zero speed.
- Energy: same minimizing paths, fixed to constant speed, smooth.

*Geodesic = shortest path?*

- Locally ✅. Globally ❌: long great-circle arc is a geodesic, not shortest.
- Shortest path may be non-unique: antipodes on a sphere → infinitely many.

*Do nonzero Christoffel symbols mean curvature?*

- ❌. Coordinate artifact (polar coordinates on a flat plane).
- Normal coordinates → $\Gamma^k_{ij}(p)=0$ at any one point, even on a curved space.
- Curvature = obstruction to making $\Gamma=0$ on a whole neighborhood.

*How do ML methods compute geodesics?*

- Closed form: sphere, hyperbolic space, SPD w/ affine-invariant metric.
- Shooting: solve the ODE, adjust $\mathbf{v}$ until the curve hits $q$.
- Discretize & minimize energy (Code).
- Graph shortest paths on a kNN graph ([Isomap](data.md#isomap)).

*Why slerp for Gaussian latents?*

- High-dim Gaussian samples concentrate near a sphere shell ([Gaussian Annulus](highdim.md#gaussian-annulus)).
- Linear interpolation → midpoint norm shrinks → atypical latent. Slerp = great-circle geodesic → stays on the shell.
- Slerp is a geodesic of the sphere, not of the decoder's pullback metric.
```

&nbsp;

#### Exponential Map
- **What**: $\exp_p:T_p\mathcal{M}\to\mathcal{M}$: follow the geodesic from $p$ w/ initial velocity $\mathbf{v}$ for unit time.
    - Defined on all of $T_p\mathcal{M}$ only if $\mathcal{M}$ is complete. Otherwise: only for $\mathbf{v}$ whose geodesic exists up to $t=1$.
- **Why**: Turn a tangent-space step (gradient, velocity) into a point on $\mathcal{M}$. $\log_p$ inverts it → local linearization of $\mathcal{M}$ at $p$.
- **How**:
    1. Start at $p$; shoot the geodesic in direction $\mathbf{v}/\|\mathbf{v}\|_g$.
    2. Walk distance $\|\mathbf{v}\|_g$ → $\exp_p(\mathbf{v})$.
    3. $\log_p(q)$ = initial velocity of the shortest geodesic $p\to q$. $\|\log_pq\|_g=d(p,q)$.
    4. Bijective only inside the **injectivity radius** $\mathrm{inj}(p)$.

```{note} Math
:class: dropdown
Notations:
- $\gamma_{\mathbf{v}}$: Geodesic w/ $\gamma_{\mathbf{v}}(0)=p$, $\dot\gamma_{\mathbf{v}}(0)=\mathbf{v}$.
- $\mathrm{inj}(p)$: Injectivity radius at $p$.

Definition:

$$
\exp_p(\mathbf{v})=\gamma_{\mathbf{v}}(1),\qquad \exp_p(t\mathbf{v})=\gamma_{\mathbf{v}}(t)
$$

Properties:
- $d(\exp_p)_{\mathbf{0}}=\mathrm{id}$ → local diffeomorphism → **normal coordinates**: $G(p)=I$, $\Gamma^k_{ij}(p)=0$.
- $\mathrm{inj}(p)=\sup\{r:\exp_p\text{ is a diffeomorphism on }\{\|\mathbf{v}\|_g<r\}\}$.
- $\log_p=\exp_p^{-1}$ on $\{q:d(p,q)<\mathrm{inj}(p)\}$; $\|\log_pq\|_g=d(p,q)$.
- Riemannian gradient of half squared distance (inside $\mathrm{inj}$): $\operatorname{grad}_p\tfrac12d(p,q)^2=-\log_pq$.
- **Cartan–Hadamard**: complete, simply connected, $K\le0$ → $\exp_p$ is a global diffeomorphism → $\mathrm{inj}=\infty$.
- Lie group w/ bi-invariant metric (e.g., $\mathrm{SO}(n)$): $\exp$ at the identity = [matrix exponential](../linalg.md#matrix-exponential).
```

```{dropdown} Table: Closed-Form Exp & Log
| Space | $\exp_p(\mathbf{v})$ | $\log_p(q)$ | $\mathrm{inj}$ |
|:--|:--|:--|:--|
| $\mathbb{R}^D$ | $\mathbf{p}+\mathbf{v}$ | $\mathbf{q}-\mathbf{p}$ | $\infty$ |
| Unit sphere $S^{D-1}$ | $\cos\|\mathbf{v}\|\,\mathbf{p}+\sin\|\mathbf{v}\|\,\frac{\mathbf{v}}{\|\mathbf{v}\|}$ | $\vartheta\,\frac{\mathbf{q}-\cos\vartheta\,\mathbf{p}}{\|\mathbf{q}-\cos\vartheta\,\mathbf{p}\|}$ | $\pi$ |
| $\mathbb{B}^n$ at $\mathbf{0}$ | $\tanh(\|\mathbf{v}\|)\frac{\mathbf{v}}{\|\mathbf{v}\|}$ | $\operatorname{artanh}(\|\mathbf{y}\|)\frac{\mathbf{y}}{\|\mathbf{y}\|}$ | $\infty$ |

- $\vartheta=\arccos(\mathbf{p}^T\mathbf{q})$: angle between $\mathbf{p}$ & $\mathbf{q}$.
- $\mathbb{B}^n$: $\|\mathbf{v}\|$ is Euclidean; $\|\mathbf{v}\|_g=\lambda_{\mathbf{0}}\|\mathbf{v}\|=2\|\mathbf{v}\|$ = hyperbolic distance travelled. General base point → [Poincare Ball](#poincare-ball).
```

````{important} Code
:class: dropdown
```python
import torch

class Sphere:
    ## unit sphere S^{D-1} in R^D w/ the induced metric
    def exp(self, p, v):
        nv = v.norm()
        if nv < 1e-12:
            return p
        return torch.cos(nv) * p + torch.sin(nv) * v / nv  ## great circle, arc length ||v||

    def log(self, p, q):
        c = (p @ q).clamp(-1, 1)
        if c <= -1 + 1e-6:
            ## antipode = cut locus: every direction reaches q -> log undefined
            raise ValueError("log_p(q) undefined for q = -p")
        u = q - c * p                            ## component of q tangent at p
        return torch.arccos(c) * u / u.norm().clamp_min(1e-12)

## Example
S = Sphere()
p, q = torch.tensor([0., 0., 1.]), torch.tensor([1., 0., 0.])
v = S.log(p, q)
print(v)           ## tensor([1.5708, 0., 0.]): length π/2 = d(p, q)
print(S.exp(p, v)) ## ≈ tensor([1., 0., 0.]) = q
```
````

```{attention} Q&A
:class: dropdown
*Where does ML use exp & log?*

- Riemannian GD steps ([Riemannian Gradient Descent](#riemannian-gradient-descent)).
- Hyperbolic layers: $\exp_{\mathbf{0}}$ / $\log_{\mathbf{0}}$ ([Hyperbolic Neural Network](#hyperbolic-neural-network)).
- Tangent-space statistics: map data w/ $\log_\mu$ → run a Euclidean method (e.g., PCA) in $T_\mu\mathcal{M}$.

*How do you average points on a manifold?*

- **Fréchet mean**: $\arg\min_p\frac{1}{2m}\sum_id(p,q_i)^2$.
- Gradient $=-\frac1m\sum_i\log_pq_i$ → iterate $p\leftarrow\exp_p\left(\frac1m\sum_i\log_pq_i\right)$.
- $\mathbb{R}^D$: one step → arithmetic mean.
- Unique on Hadamard manifolds ($K\le0$, complete, simply connected); can be non-unique on a sphere for spread-out data.

*What happens past the injectivity radius?*

- $\exp_p$ stops being injective. Sphere: every $\|\mathbf{v}\|=\pi$ maps to the antipode → $\log_p$ undefined there.

*Exp map vs retraction?*

- Exp = exact geodesic step, often expensive. [Retraction](#retraction) = first-order approximation; enough for first-order optimization.
```

&nbsp;

### Curvature
- **What**: Intrinsic deviation of a space from flatness.
- **Why**: Decides how geodesics behave → triangle angle sums, circle sizes, convergence or divergence of nearby geodesics, which data embed w/o distortion.
- **How**:
    - **Surface**: Gaussian curvature $K=\kappa_1\kappa_2$ (product of principal curvatures). Intrinsic (Theorema Egregium): measurable w/o leaving the surface.
    - **Higher dim**: sectional curvature $K(\sigma)$ = Gaussian curvature of the surface swept by geodesics tangent to the 2-plane $\sigma\subset T_p\mathcal{M}$.
    - Sign → geometry:
        - $K>0$: nearby geodesics converge; triangle angle sum $>\pi$; circles shorter than $2\pi r$.
        - $K=0$: Euclidean behavior.
        - $K<0$: nearby geodesics diverge; angle sum $<\pi$; circles longer than $2\pi r$.

```{note} Math
:class: dropdown
Notations:
- $\kappa_s$: $s$-th principal curvature of a surface in $\mathbb{R}^3$, $s\in\{1,2\}$.
- $\mathcal{T}$: Geodesic triangle.
- $\vartheta_s$: $s$-th interior angle of $\mathcal{T}$, $s\in\{1,2,3\}$.
- $\mathbf{u}$: Vector field.
- $[\cdot,\cdot]$: Lie bracket of vector fields.
- $\nabla$: Levi-Civita connection ([Parallel Transport](#parallel-transport)).
- $R$: Riemann curvature tensor.
- $\{\mathbf{e}_i\}$: Orthonormal basis.
- $\kappa$: Constant curvature value.

Surfaces:

$$
C(r)=2\pi r\left(1-\frac{K}{6}r^2+O(r^4)\right),\qquad \vartheta_1+\vartheta_2+\vartheta_3=\pi+\iint_{\mathcal{T}}K\,dA
$$

- $C(r)$: circumference of a small geodesic circle of radius $r$.
- Right identity: Gauss–Bonnet for a geodesic triangle.

General dimension:

$$
\begin{align*}
R(\mathbf{u},\mathbf{v})\mathbf{w}&=\nabla_{\mathbf{u}}\nabla_{\mathbf{v}}\mathbf{w}-\nabla_{\mathbf{v}}\nabla_{\mathbf{u}}\mathbf{w}-\nabla_{[\mathbf{u},\mathbf{v}]}\mathbf{w}\\
K(\mathbf{u},\mathbf{v})&=\frac{g(R(\mathbf{u},\mathbf{v})\mathbf{v},\mathbf{u})}{g(\mathbf{u},\mathbf{u})\,g(\mathbf{v},\mathbf{v})-g(\mathbf{u},\mathbf{v})^2}
\end{align*}
$$

- $\mathbf{v},\mathbf{w}$: extended to vector fields.
- $K(\mathbf{u},\mathbf{v})$ depends only on the plane $\operatorname{span}(\mathbf{u},\mathbf{v})$.
- Sign conventions for $R$ differ across textbooks. This is Lee's convention; $K$ has the same value in all conventions.
- **Ricci** (unit $\mathbf{u}$, $\{\mathbf{e}_i\}_{i=1}^{n-1}$ orthonormal basis of $\mathbf{u}^\perp$): $\mathrm{Ric}(\mathbf{u},\mathbf{u})=\sum_{i=1}^{n-1}K(\mathbf{u},\mathbf{e}_i)$.
- **Scalar**: $S=\sum_{i=1}^n\mathrm{Ric}(\mathbf{e}_i,\mathbf{e}_i)$ for an orthonormal basis of $T_p\mathcal{M}$.
- Constant curvature $\kappa$: $\mathrm{Ric}=(n-1)\kappa\,g$, $S=n(n-1)\kappa$.
- Geodesic deviation, constant $\kappa$: separation of two geodesics from one point grows as $\frac{\sin(\sqrt{\kappa}t)}{\sqrt{\kappa}}$ / $t$ / $\frac{\sinh(\sqrt{-\kappa}t)}{\sqrt{-\kappa}}$ for $\kappa>0$ / $=0$ / $<0$.
- **Model spaces**: complete, simply connected, constant $\kappa$ → unique up to isometry: sphere of radius $1/\sqrt{\kappa}$, $\mathbb{R}^n$, hyperbolic space (scaled).
```

```{note} Example
:class: dropdown
- Unit sphere, octant triangle (vertices on the 3 axes): 3 right angles → sum $=3\pi/2=\pi+K\cdot\text{Area}=\pi+\frac{4\pi}{8}$.
- Sphere of radius $R$: $K=1/R^2$ → large sphere ≈ flat locally.
- Cylinder: $\kappa_1=1/R$, $\kappa_2=0$ → $K=0$ → paper rolls into a cylinder w/o stretching.
- Sphere: $K>0$ → no distance-preserving flat map of the Earth.
```

```{dropdown} Table: Constant-Curvature Model Spaces (2-D)
| | Sphere $S^2$ | Plane $\mathbb{R}^2$ | Hyperbolic plane $\mathbb{H}^2$ |
|:--|:--|:--|:--|
| $K$ | $+1$ | $0$ | $-1$ |
| Geodesics | Great circles | Lines | Diameters & arcs ⊥ boundary (Poincaré disk) |
| Triangle angle sum | $>\pi$ | $=\pi$ | $<\pi$ |
| Circle circumference $C(r)$ | $2\pi\sin r$ | $2\pi r$ | $2\pi\sinh r$ |
| Parallels through an outside point | 0 | 1 | $\infty$ |
| Nearby geodesics | Converge | Constant gap | Diverge exponentially |
| ML fit | Directions, normalized embeddings | Default | Trees, hierarchies |
```

```{attention} Q&A
:class: dropdown
*Loss-landscape curvature vs Riemannian curvature?*

- Loss curvature: Hessian of a function (sharpness). Property of $\mathcal{L}$.
- Riemannian curvature: property of the space & its metric. No function involved.
- Link: on a curved $\mathcal{M}$, the Riemannian Hessian of $\mathcal{L}$ includes Christoffel terms.

*Intrinsic vs extrinsic curvature?*

- Gaussian/sectional: intrinsic → an ant w/ a ruler measures it (circle circumference, angle sums).
- Mean curvature: extrinsic. Cylinder: mean $\ne0$, Gaussian $=0$.

*Why does the sign matter for embeddings?*

- Embedding metric must match data growth. Tree-like data → exponential growth → $K<0$.
- Wrong sign → distortion, or many more dimensions to compensate.

*What does Ricci curvature control?*

- Average convergence of geodesics → volume growth of geodesic balls.
- Bishop–Gromov: complete $\mathcal{M}$ w/ $\mathrm{Ric}\ge(n-1)\kappa g$ → geodesic-ball volume ≤ same-radius ball volume in the model space of curvature $\kappa$.
- General relativity: Einstein equations constrain Ricci curvature by matter.

*How does curvature relate to parallel transport?*

- $R(\mathbf{u},\mathbf{v})\mathbf{w}$ = infinitesimal rotation of $\mathbf{w}$ after transport around a small loop spanned by $\mathbf{u},\mathbf{v}$ → [Parallel Transport](#parallel-transport).
```

&nbsp;

### Parallel Transport
- **What**: Move a tangent vector along a curve w/o turning it, as judged by a connection.
- **Why**: Tangent spaces at different points are different vector spaces → no way to compare or add vectors at $p$ & $q$ (e.g., momentum + new gradient) w/o a transport rule.
- **How**:
    1. **Connection** $\nabla$: rule to differentiate vector fields along curves (covariant derivative).
    2. **Levi-Civita connection**: unique connection that is torsion-free & preserves $g$ → canonical choice on a Riemannian manifold.
    3. Transport $\mathbf{v}$ along $\gamma$: solve the linear ODE $D_t\mathbf{v}=0$.
    4. Embedded intuition: move a small step → project $\mathbf{v}$ onto the new tangent space → repeat.
    5. Closed loop → $\mathbf{v}$ returns rotated (**holonomy**) ← curvature.

```{note} Math
:class: dropdown
Notations:
- $D_t$: Covariant derivative along $\gamma$.
- $P_\gamma$: Parallel transport map along $\gamma$.
- $\mathbf{u}$: Vector field.
- $\Omega$: Simply connected surface region bounded by a loop.

Covariant derivative along $\gamma(t)=\mathbf{z}(t)$:

$$
D_t\mathbf{v}=\left(\dot v^k+\Gamma^k_{ij}\,\dot z^i v^j\right)\partial_k
$$

- Parallel: $D_t\mathbf{v}=0$. Geodesic: $D_t\dot\gamma=0$ → velocity transports itself.

Levi-Civita conditions:

$$
\begin{align*}
\nabla_{\mathbf{u}}\mathbf{w}-\nabla_{\mathbf{w}}\mathbf{u}&=[\mathbf{u},\mathbf{w}]&&\text{(torsion-free)}\\
\mathbf{u}\,g(\mathbf{v},\mathbf{w})&=g(\nabla_{\mathbf{u}}\mathbf{v},\mathbf{w})+g(\mathbf{v},\nabla_{\mathbf{u}}\mathbf{w})&&\text{(metric-compatible)}
\end{align*}
$$

- Unique solution → Christoffel formula of [Geodesic](#geodesic).

Properties:
- $P_\gamma:T_{\gamma(0)}\mathcal{M}\to T_{\gamma(1)}\mathcal{M}$ is a linear isometry ← metric compatibility.
- Path-independent on a simply connected region ⇔ $R=0$ there.
- Surface holonomy: loop $\partial\Omega$ → rotation angle $=\iint_\Omega K\,dA$ (mod $2\pi$).
- $\mathbb{B}^n$, from the origin: $P_{\mathbf{0}\to\mathbf{x}}(\mathbf{v})=\frac{\lambda_{\mathbf{0}}}{\lambda_{\mathbf{x}}}\mathbf{v}$.
```

```{note} Example
:class: dropdown
Unit sphere, octant triangle (north pole → equator → 90° along equator → north pole):
- Transport a vector around the loop → returns rotated by $\pi/2$.
- $\pi/2=\iint K\,dA=1\cdot\frac{4\pi}{8}$ = enclosed area.
```

```{attention} Q&A
:class: dropdown
*Why do Riemannian momentum methods need it?*

- Momentum $\mathbf{m}_t\in T_{p_t}\mathcal{M}$. After the step, the new gradient lives in $T_{p_{t+1}}\mathcal{M}$.
- Transport $\mathbf{m}_t$ (or project it: cheap **vector transport**) before adding.
- Ganea et al. (2018) trained hyperbolic params w/ plain RSGD because no Riemannian Adam existed then.

*Why is transport path-dependent?*

- ← curvature. Flat & simply connected → path-independent → ordinary translation of vectors.

*Where does ML use it directly?*

- Hyperbolic bias: carry a Euclidean bias from $T_{\mathbf{0}}$ to $T_{\mathbf{x}}$, then $\exp_{\mathbf{x}}$ → equals Möbius addition ([Hyperbolic Neural Network](#hyperbolic-neural-network)).
- Hyperbolic MLR: normal vectors $\mathbf{a}_k\in T_{\mathbf{p}_k}$ parametrized by transport from $T_{\mathbf{0}}$.

*Is the Levi-Civita connection the only connection?*

- ❌. Many connections exist; Levi-Civita is the unique torsion-free, metric-compatible one.
- Information geometry uses a dual pair of non-metric connections → [Dually Flat Manifold](info.md#dually-flat-manifold).
```

&nbsp;

## Hyperbolic Geometry

### Hyperbolic Space
- **What**: Complete, simply connected manifold w/ constant negative sectional curvature (usually $-1$), $\mathbb{H}^n$.
- **Why**: Continuous analog of a tree.
    - Ball volume & sphere area grow exponentially in radius. Trees: #nodes grows exponentially in depth.
    - Euclidean volume grows polynomially → trees get crowded → distortion or many dimensions.
- **How**: Pick one of several isometric coordinate models:
    - **Poincaré ball**: open unit ball, conformal → [Poincare Ball](#poincare-ball).
    - **Hyperboloid (Lorentz)**: upper sheet of a hyperboloid in $\mathbb{R}^{n+1}$.
    - **Upper half-space**: $\{\mathbf{a}\in\mathbb{R}^n:a_n>0\}$ w/ metric $\|d\mathbf{a}\|^2/a_n^2$.

```{note} Math
:class: dropdown
Notations:
- $r$: Geodesic radius.
- $\beta$: Tree branching factor.
- $\langle\cdot,\cdot\rangle_{\mathcal{L}}$: Lorentzian inner product on $\mathbb{R}^{n+1}$.
- $\mathbf{a}$: Point in the hyperboloid or half-space model.
- $\mathbf{b}$: Second point in the same model.

Growth ($K=-1$):
- Circle circumference $2\pi\sinh r$; disk area $2\pi(\cosh r-1)$.
- $\mathbb{H}^n$: area of a geodesic sphere $\propto\sinh^{n-1}r\sim e^{(n-1)r}$.
- $\beta$-ary tree: $\beta^\ell$ nodes at depth $\ell$ → exponential, like $\mathbb{H}^n$.

Hyperboloid model:

$$
\mathbb{H}^n=\{\mathbf{a}\in\mathbb{R}^{n+1}:\langle\mathbf{a},\mathbf{a}\rangle_{\mathcal{L}}=-1,\ a_0>0\},\qquad \langle\mathbf{a},\mathbf{b}\rangle_{\mathcal{L}}=-a_0b_0+\sum_{j=1}^na_jb_j
$$

- $d(\mathbf{a},\mathbf{b})=\operatorname{arcosh}(-\langle\mathbf{a},\mathbf{b}\rangle_{\mathcal{L}})$.
- Map to $\mathbb{B}^n$: $\mathbf{a}\mapsto\frac{(a_1,\dots,a_n)}{1+a_0}$.

Upper half-space: $d(\mathbf{a},\mathbf{b})=\operatorname{arcosh}\left(1+\frac{\|\mathbf{a}-\mathbf{b}\|^2}{2a_nb_n}\right)$.

Gromov $\delta$-hyperbolicity: every geodesic triangle is $\delta$-thin (each side within $\delta$ of the union of the other two).
- Metric trees: $\delta=0$. $\mathbb{H}^n$: finite $\delta$. $\mathbb{R}^n$ ($n\ge2$): no finite $\delta$.
```

```{dropdown} Table: Models of Hyperbolic Space
| Model | Domain | Distance | Pros | Cons |
|:--|:--|:--|:--|:--|
| Poincaré ball | $\|\mathbf{x}\|<1$ | $\operatorname{arcosh}\left(1+\frac{2\|\mathbf{x}-\mathbf{y}\|^2}{(1-\|\mathbf{x}\|^2)(1-\|\mathbf{y}\|^2)}\right)$ | Bounded, conformal, easy to visualize | $1-\|\mathbf{x}\|^2$ underflows near the boundary |
| Hyperboloid | Sheet in $\mathbb{R}^{n+1}$ | $\operatorname{arcosh}(-\langle\mathbf{a},\mathbf{b}\rangle_{\mathcal{L}})$ | Simple exp/log & geodesics, no boundary division | Unbounded coords, 1 extra dim |
| Upper half-space | $a_n>0$ | $\operatorname{arcosh}\left(1+\frac{\|\mathbf{a}-\mathbf{b}\|^2}{2a_nb_n}\right)$ | Simple isometries (scaling, translation in $a_1..a_{n-1}$) | Asymmetric, boundary at $a_n=0$ & $\infty$ |
```

```{attention} Q&A
:class: dropdown
*Why can Euclidean space not embed trees well?*

- Star w/ center & 3 leaves (edge length 1): leaves at unit distance from the center & distance 2 from each other → each leaf pair must be antipodal through the center → impossible for 3 leaves in any $\mathbb{R}^D$.
- Volume: a ball of radius $r$ in $\mathbb{R}^n$ holds $O(r^n)$ well-separated points ≪ $\beta^r$ leaves of a $\beta$-ary tree of depth $r$.

*Why curvature exactly $-1$?*

- Any constant $K<0$ → rescale distances by $\sqrt{-K}$ → $K=-1$.
- Some models learn the curvature magnitude $c$ as a param ($K=-c$).

*Does the brain use hyperbolic geometry?*

- One study: rat CA1 place-cell representations fit a hyperbolic geometry whose size grows w/ the log of exploration time {cite:p}`zhang2023hippocampal`.
- Single-study evidence. Not an established general principle.

*What does it tell us about AC?*

- Hypothesis: concept hierarchies (taxonomies, part–whole) are more efficiently represented in negatively curved spaces than in flat ones.
- Evidence: low-dim hyperbolic embeddings of WordNet beat high-dim Euclidean ones ([Hyperbolic Embedding](#hyperbolic-embedding)).
- Whether learned LLM or brain concept spaces are hyperbolic is open.
```

&nbsp;

#### Poincare Ball
- **What**: Model of $\mathbb{H}^n$ on the open unit ball w/ the Euclidean metric scaled by $\lambda_{\mathbf{x}}^2$.
- **Why**: Bounded, conformal (angles = Euclidean angles), closed-form distance → gradient-based learning in ordinary $\mathbb{R}^n$ coordinates.
- **How**:
    1. Points: $\|\mathbf{x}\|<1$.
    2. Metric: $g_{\mathbf{x}}=\lambda_{\mathbf{x}}^2g^E$, $\lambda_{\mathbf{x}}=\frac{2}{1-\|\mathbf{x}\|^2}$ → blows up at the boundary → boundary infinitely far.
    3. Root placed at the origin → small distance to all points. Near the boundary: Euclidean-close points are hyperbolically far → room for exponentially many leaves.
    4. Geodesics: diameters & circular arcs orthogonal to the boundary sphere.
    5. **Möbius addition** $\oplus$: hyperbolic analog of vector addition → translations, exp/log maps at any point.

```{note} Math
:class: dropdown
Notations:
- $g^E$: Euclidean metric.
- $c>0$: Curvature magnitude ($K=-c$).
- $\lambda^c_{\mathbf{x}}$: Conformal factor at curvature $-c$.

Definition:

$$
\mathbb{B}^n=\{\mathbf{x}\in\mathbb{R}^n:\|\mathbf{x}\|<1\},\qquad g_{\mathbf{x}}=\lambda_{\mathbf{x}}^2\,g^E,\qquad \lambda_{\mathbf{x}}=\frac{2}{1-\|\mathbf{x}\|^2}
$$

Distance:

$$
d(\mathbf{x},\mathbf{y})=\operatorname{arcosh}\left(1+\frac{2\|\mathbf{x}-\mathbf{y}\|^2}{(1-\|\mathbf{x}\|^2)(1-\|\mathbf{y}\|^2)}\right)
$$

- $d(\mathbf{0},\mathbf{x})=2\operatorname{artanh}\|\mathbf{x}\|$.
- Conformal → Euclidean angles = hyperbolic angles.

Möbius addition (curvature $-c$; $\oplus=\oplus_1$):

$$
\mathbf{x}\oplus_c\mathbf{y}=\frac{(1+2c\,\mathbf{x}^T\mathbf{y}+c\|\mathbf{y}\|^2)\,\mathbf{x}+(1-c\|\mathbf{x}\|^2)\,\mathbf{y}}{1+2c\,\mathbf{x}^T\mathbf{y}+c^2\|\mathbf{x}\|^2\|\mathbf{y}\|^2}
$$

- ❌Commutative, ❌associative. $(-\mathbf{x})\oplus_c(\mathbf{x}\oplus_c\mathbf{y})=\mathbf{y}$ (left cancellation).
- $c\to0$ → ordinary $\mathbf{x}+\mathbf{y}$.
- Distance: $d_c(\mathbf{x},\mathbf{y})=\frac{2}{\sqrt c}\operatorname{artanh}\left(\sqrt c\,\|(-\mathbf{x})\oplus_c\mathbf{y}\|\right)$.
- Curvature $-c$: ball radius $1/\sqrt c$, $\lambda^c_{\mathbf{x}}=\frac{2}{1-c\|\mathbf{x}\|^2}$.

Exp & log maps (curvature $-c$, $\mathbf{v}\ne\mathbf{0}$, $\mathbf{y}\ne\mathbf{x}$):

$$
\begin{align*}
\exp^c_{\mathbf{x}}(\mathbf{v})&=\mathbf{x}\oplus_c\left(\tanh\left(\frac{\sqrt c\,\lambda^c_{\mathbf{x}}\|\mathbf{v}\|}{2}\right)\frac{\mathbf{v}}{\sqrt c\,\|\mathbf{v}\|}\right)\\
\log^c_{\mathbf{x}}(\mathbf{y})&=\frac{2}{\sqrt c\,\lambda^c_{\mathbf{x}}}\operatorname{artanh}\left(\sqrt c\,\|(-\mathbf{x})\oplus_c\mathbf{y}\|\right)\frac{(-\mathbf{x})\oplus_c\mathbf{y}}{\|(-\mathbf{x})\oplus_c\mathbf{y}\|}
\end{align*}
$$

- At $\mathbf{x}=\mathbf{0}$: $\exp^c_{\mathbf{0}}(\mathbf{v})=\tanh(\sqrt c\|\mathbf{v}\|)\frac{\mathbf{v}}{\sqrt c\|\mathbf{v}\|}$, $\log^c_{\mathbf{0}}(\mathbf{y})=\operatorname{artanh}(\sqrt c\|\mathbf{y}\|)\frac{\mathbf{y}}{\sqrt c\|\mathbf{y}\|}$.
```

```{tip} Derivation
:class: dropdown
*Why is $d(\mathbf{0},\mathbf{x})=2\operatorname{artanh}\|\mathbf{x}\|$?*

1. Radial line = geodesic (symmetry). Speed at radius $s$: $\lambda=\frac{2}{1-s^2}$.
2. $d=\int_0^{\|\mathbf{x}\|}\frac{2}{1-s^2}\,ds=2\operatorname{artanh}\|\mathbf{x}\|$.
3. Check w/ the formula: $\operatorname{arcosh}\frac{1+\rho^2}{1-\rho^2}=2\operatorname{artanh}\rho$ for $\rho=\|\mathbf{x}\|$.

*Why is the curvature $-1$? ($n=2$)*

1. Conformal metric $e^{2\phi}g^E$ in 2-D → $K=-e^{-2\phi}\Delta\phi$.
2. $\phi=\log2-\log(1-\rho^2)$, $\rho=\|\mathbf{x}\|$.
3. Radial Laplacian $\Delta h=h''+h'/\rho$ on $h=\log(1-\rho^2)$: $h'=\frac{-2\rho}{1-\rho^2}$, $h''=\frac{-2-2\rho^2}{(1-\rho^2)^2}$ → $\Delta h=\frac{-4}{(1-\rho^2)^2}$.
4. $\Delta\phi=\frac{4}{(1-\rho^2)^2}$, $e^{-2\phi}=\frac{(1-\rho^2)^2}{4}$ → $K=-1$.
```

```{note} Example
:class: dropdown
$\mathbf{x}=(0.9,0)$, $\mathbf{y}=(0,0.9)$:
- $d(\mathbf{0},\mathbf{x})=2\operatorname{artanh}0.9=\ln19\approx2.944$.
- $d(\mathbf{x},\mathbf{y})=\operatorname{arcosh}(1+2\cdot1.62/0.19^2)=\operatorname{arcosh}(90.75)\approx5.201$.
- Path through the origin: $2\times2.944=5.889$ → only 13% longer than the geodesic → shortest path bends toward the root, like tree distance (leaf → parent → leaf).
- Euclidean: $\|\mathbf{x}-\mathbf{y}\|\approx1.27$ vs $\|\mathbf{x}\|=0.9$ → no such effect.
```

````{important} Code
:class: dropdown
```python
import torch

class PoincareBall:
    def __init__(self, eps=1e-5):
        self.eps = eps  ## keep points at Euclidean norm <= 1 - eps

    def lam(self, x):
        return 2 / (1 - (x * x).sum(-1, keepdim=True))  ## conformal factor

    def mobius_add(self, x, y):
        xy = (x * y).sum(-1, keepdim=True)
        x2 = (x * x).sum(-1, keepdim=True)
        y2 = (y * y).sum(-1, keepdim=True)
        return ((1 + 2 * xy + y2) * x + (1 - x2) * y) / (1 + 2 * xy + x2 * y2)

    def dist(self, x, y):
        sq = ((x - y) ** 2).sum(-1)
        den = (1 - (x * x).sum(-1)) * (1 - (y * y).sum(-1))
        return torch.acosh((1 + 2 * sq / den).clamp_min(1.0))  ## clamp: rounding can give < 1

    def expmap0(self, v):
        n = v.norm(dim=-1, keepdim=True).clamp_min(1e-15)
        return self.project(torch.tanh(n) * v / n)  ## tanh can round to 1 in float -> keep inside

    def logmap0(self, x):
        n = x.norm(dim=-1, keepdim=True).clamp_min(1e-15)
        return torch.atanh(n.clamp_max(1 - self.eps)) * x / n

    def project(self, x):
        ## pull points that drifted to/over the boundary back inside
        n = x.norm(dim=-1, keepdim=True).clamp_min(1e-15)
        r = 1 - self.eps
        return torch.where(n > r, x / n * r, x)

## Example
B = PoincareBall()
o, x, y = torch.zeros(2), torch.tensor([0.9, 0.]), torch.tensor([0., 0.9])
print(B.dist(o, x))                                   ## ≈ 2.944 = ln 19
print(B.dist(x, y))                                   ## ≈ 5.201 < 5.889 = path via origin
print(2 * torch.atanh(B.mobius_add(-x, y).norm()))    ## ≈ 5.201: same distance via ⊕
```
````

```{attention} Q&A
:class: dropdown
*Why do distances explode near the boundary?*

- $\|\mathbf{x}\|=1-\delta$ → $1-\|\mathbf{x}\|^2\approx2\delta$ → $\lambda_{\mathbf{x}}\approx1/\delta$.
- → Euclidean step $\epsilon$ costs ≈ $\epsilon/\delta$ hyperbolic distance.

*What are the numerical limits?*

- Max norm in float32 ≈ $1-2^{-24}$ → max $d(\mathbf{0},\mathbf{x})\approx17$. Float64 → ≈ 37.
- Clip every norm to $\le1-10^{-5}$ → max $d(\mathbf{0},\mathbf{x})=2\operatorname{artanh}(1-10^{-5})\approx12.2$.
- Boundary-only projection (triggers at $\|\mathbf{x}\|\ge1$, as in Nickel & Kiela) does not impose this cap.
- → Deep trees need large radii → precision limits. Hyperboloid avoids the $1-\|\mathbf{x}\|^2$ term but not the precision cost of large radii.

*Why Möbius addition, not $+$?*

- $\mathbf{x}+\mathbf{y}$ can leave the ball & is not an isometry.
- $\mathbf{y}\mapsto\mathbf{x}\oplus\mathbf{y}$ is an isometry of $\mathbb{B}^n$ that maps $\mathbf{0}\mapsto\mathbf{x}$ → hyperbolic "translation".

*Poincaré ball vs hyperboloid?*

- Same space, different coordinates → see the table in [Hyperbolic Space](#hyperbolic-space).
```

&nbsp;

### Hyperbolic Embedding
- **What**: Embed symbols (nodes, words, concepts) as points in $\mathbb{B}^n$ so hyperbolic distance reflects relatedness. {cite:p}`nickel2017poincare`
- **Why**: Hierarchy.
    - Symbolic data (taxonomies, knowledge graphs, social networks) often have a latent tree structure.
    - Any tree embeds in the hyperbolic plane w/ distortion $<1+\varepsilon$ for any $\varepsilon>0$, after rescaling edge lengths {cite:p}`sarkar2011low`.
    - Euclidean space of any dim cannot embed even a 3-leaf star w/o distortion ([Hyperbolic Space](#hyperbolic-space)).
- **How**:
    1. Init all embeddings near the origin.
    2. Loss: related pairs closer than sampled unrelated pairs (softmax over negative distances).
    3. Riemannian SGD: rescale the Euclidean grad by the inverse metric → step → project back inside the ball.
    4. Result: distance ↔ similarity. Root at the origin → norm can encode depth (general near the origin, specific near the boundary).
        - Loss uses only pairwise distances → invariant under hyperbolic isometries (translations change norms) → loss alone does not fix the origin.
        - Empirically, init near the origin tends to give this layout.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathcal{D}=\{(a,b)\}$: Observed related pairs (e.g., hypernymy).
    - $\mathcal{N}(a)$: Sampled negatives: symbols $b'$ w/ $(a,b')\notin\mathcal{D}$.
- Params:
    - $\boldsymbol{\theta}_a\in\mathbb{B}^n$: Embedding of symbol $a$.
- Hyperparams:
    - $\varepsilon$: Boundary margin of the projection.
- Misc:
    - $\nabla_E$: Euclidean gradient of the loss w.r.t. $\boldsymbol{\theta}$.

Model: relatedness score $=-d(\boldsymbol{\theta}_a,\boldsymbol{\theta}_b)$ ([Poincare Ball](#poincare-ball) distance).

Training:

$$
\mathcal{L}(\Theta)=-\sum_{(a,b)\in\mathcal{D}}\log\frac{e^{-d(\boldsymbol{\theta}_a,\boldsymbol{\theta}_b)}}{\sum_{b'\in\mathcal{N}(a)\cup\{b\}}e^{-d(\boldsymbol{\theta}_a,\boldsymbol{\theta}_{b'})}}
$$

- $\Theta$: all embeddings.
- Paper: 10 negatives per positive.
- Paper's Eq. 6 omits the minus sign. Its $\mathcal{N}(u)$ explicitly includes $u$.
- Formula above: positive + sampled negatives, the standard softmax-ranking form.

Update (RSGD w/ first-order retraction $R_{\boldsymbol{\theta}}(\mathbf{v})=\boldsymbol{\theta}+\mathbf{v}$):

$$
\boldsymbol{\theta}_{t+1}\leftarrow\operatorname{proj}\left(\boldsymbol{\theta}_t-\eta_t\frac{(1-\|\boldsymbol{\theta}_t\|^2)^2}{4}\nabla_E\right)
$$

- $\operatorname{proj}$: if $\|\boldsymbol{\theta}\|\ge1$ → rescale to norm $1-\varepsilon$. Paper: $\varepsilon=10^{-5}$.
- Init: $\mathcal{U}(-0.001,0.001)$. Burn-in: 10 epochs at LR $\eta/10$ → angular layout first.
```

```{tip} Derivation
:class: dropdown
*Why the factor $(1-\|\boldsymbol{\theta}\|^2)^2/4$?*

1. $G(\boldsymbol{\theta})=\lambda_{\boldsymbol{\theta}}^2I$ (conformal).
2. Riemannian grad $=G^{-1}\nabla_E=\lambda_{\boldsymbol{\theta}}^{-2}\nabla_E=\frac{(1-\|\boldsymbol{\theta}\|^2)^2}{4}\nabla_E$.
3. Near the boundary the factor → 0 → small Euclidean steps = normal-sized hyperbolic steps.
4. Retraction $\boldsymbol{\theta}+\mathbf{v}$ instead of $\exp_{\boldsymbol{\theta}}$ → the update equals a natural-gradient step w/ the Poincaré metric.
```

```{note} Example
:class: dropdown
WordNet nouns, transitive closure (82,115 nouns, 743,241 hypernymy pairs), reconstruction:

| Embedding | Dim | Mean rank | MAP |
|:--|:--|:--|:--|
| Euclidean | 200 | 1157.3 | 0.168 |
| Translational | 200 | 91.0 | 0.565 |
| Poincaré | 5 | 4.9 | 0.823 |
| Poincaré | 200 | 3.83 | 0.870 |
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn.functional as F

## Reuses PoincareBall from Poincare Ball (this page)
class PoincareRSGD:
    def __init__(self, params, lr, ball):
        self.params, self.lr, self.ball = params, lr, ball

    @torch.no_grad()
    def step(self):
        for p in self.params:
            ## Riemannian grad = G^{-1} grad_E, G = lambda^2 I
            scale = (1 - (p * p).sum(-1, keepdim=True)) ** 2 / 4
            p -= self.lr * scale * p.grad
            p.copy_(self.ball.project(p))
            p.grad.zero_()

## Example: root 0, leaves 1..3
B = PoincareBall()
emb = torch.nn.Parameter(torch.empty(4, 2, dtype=torch.float64).uniform_(-1e-3, 1e-3))
pos = torch.tensor([[1, 0], [2, 0], [3, 0]])  ## (child, parent)
neg = torch.tensor([[2, 3], [1, 3], [1, 2]])  ## unrelated symbols per child
cand = torch.cat([pos[:, 1:], neg], dim=1)    ## (3, 3): positive in column 0
opt = PoincareRSGD([emb], lr=0.1, ball=B)
for t in range(300):
    d = B.dist(emb[pos[:, 0]].unsqueeze(1), emb[cand])     ## (3, 3)
    loss = F.cross_entropy(-d, torch.zeros(3, dtype=torch.long))
    if t == 0:
        loss0 = loss.item()                                ## ≈ ln 3 at init
    loss.backward()
    opt.step()
print(loss0, loss.item())  ## loss decreases
```
````

```{attention} Q&A
:class: dropdown
*Pros?*

- Low dim: 5-D Poincaré > 200-D Euclidean on WordNet.
- Hierarchy for free: once the root sits at the origin, norm ≈ depth; no depth labels needed.
- Better link prediction on tree-like graphs.

*Cons?*

- Precision limits near the boundary ([Poincare Ball](#poincare-ball)).
- Gain only for tree-like data; cycles & grids gain little.
- Distance is symmetric → direction of "is-a" only through norms.
- Low-distortion constructions rescale edges → points pushed toward the boundary → more bits of precision.

*Why a ranking loss, not "distance = graph distance"?*

- Unrelated subtrees need not be arbitrarily far; only farther than related pairs.

*Why init near the origin + burn-in?*

- Fix the angular layout first while points are near the origin.
- Large early steps push points to the boundary too early → hard to reorganize.

*How does this relate to Riemannian optimization?*

- Update = [Riemannian Gradient Descent](#riemannian-gradient-descent) w/ the simplest [Retraction](#retraction) + boundary projection.

*What does it tell us about AC?*

- Geometry = inductive bias: pick a space whose volume growth matches the data's branching → fewer dims, cleaner structure.
- Hypothesis: concept hierarchies in minds & models may favor negatively curved codes. Evidence so far: engineered embeddings & one neuroscience study ([Hyperbolic Space](#hyperbolic-space)).
```

&nbsp;

### Hyperbolic Neural Network
- **What**: NN whose layers map $\mathbb{B}^n$ to $\mathbb{B}^m$ via gyrovector operations (Möbius addition & matrix-vector product) & exp/log maps. {cite:p}`ganea2018hyperbolic`
- **Why**: Hyperbolic embeddings need downstream layers (FFNN, RNN, softmax) that respect hyperbolic geometry. Euclidean ops on ball coordinates ignore the metric & can leave the ball.
- **How**:
    1. **Möbius version** of a Euclidean map $h$: $\log_{\mathbf{0}}$ → apply $h$ in $T_{\mathbf{0}}\mathbb{B}^n=\mathbb{R}^n$ → $\exp_{\mathbf{0}}$.
    2. Linear: Möbius matvec $M\otimes\mathbf{x}$. Nonlinearity: Möbius version $\varphi^{\otimes}$.
    3. Bias: Möbius addition $\oplus\mathbf{b}$ = parallel transport of the bias from the origin + exp map.
    4. Output: **hyperbolic MLR**: logit = signed, scaled hyperbolic distance to a learned hyperbolic hyperplane.
    5. Train: Euclidean params (matrices) w/ Adam; ball-valued params (embeddings, biases) w/ RSGD.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $M\in\mathbb{R}^{m\times n}$: Weight matrix.
    - $\mathbf{b}\in\mathbb{B}^m$: Hyperbolic bias.
    - $\mathbf{p}_k\in\mathbb{B}^n$: Hyperplane offset for class $k$.
    - $\mathbf{a}'_k\in\mathbb{R}^n$: Euclidean param for the hyperplane normal of class $k$.
- Misc:
    - $\varphi$: Pointwise nonlinearity.
    - $\mathbf{a}_k=\frac{\lambda_{\mathbf{0}}}{\lambda_{\mathbf{p}_k}}\mathbf{a}'_k\in T_{\mathbf{p}_k}\mathbb{B}^n$: Normal vector (transported from $T_{\mathbf{0}}$).

Model ($c=1$):

$$
\begin{align*}
M\otimes\mathbf{x}&=\exp_{\mathbf{0}}\left(M\log_{\mathbf{0}}(\mathbf{x})\right)\\
\varphi^{\otimes}(\mathbf{x})&=\exp_{\mathbf{0}}\left(\varphi(\log_{\mathbf{0}}(\mathbf{x}))\right)\\
\text{layer}(\mathbf{x})&=\varphi^{\otimes}\left((M\otimes\mathbf{x})\oplus\mathbf{b}\right)
\end{align*}
$$

- $M\otimes\mathbf{x}=M\mathbf{x}$ for orthogonal $M$ (rotations preserved).
- Composition w/o biases collapses: $h_2^{\otimes}\circ h_1^{\otimes}=\exp_{\mathbf{0}}\circ h_2\circ h_1\circ\log_{\mathbf{0}}$ → a Euclidean net in disguise. Bias translations make it hyperbolic.
- $c\to0$ → ordinary Euclidean layers.

Hyperbolic MLR ($c=1$):

$$
p(y=k\mid\mathbf{x})\propto\exp\left(\lambda_{\mathbf{p}_k}\|\mathbf{a}_k\|\,\operatorname{arsinh}\left(\frac{2\,\left((-\mathbf{p}_k)\oplus\mathbf{x}\right)^T\mathbf{a}_k}{(1-\|(-\mathbf{p}_k)\oplus\mathbf{x}\|^2)\,\|\mathbf{a}_k\|}\right)\right)
$$

- $k$: class idx (Tier-1 meaning, local to this formula).
- Euclidean MLR rewritten as signed distance to hyperplanes $\{\mathbf{x}:(\mathbf{x}-\mathbf{p}_k)^T\mathbf{a}_k=0\}$ → replace by hyperbolic distance to a hyperbolic hyperplane.
- $c\to0$ → $p(y=k\mid\mathbf{x})\propto\exp(4(\mathbf{x}-\mathbf{p}_k)^T\mathbf{a}_k)$ → Euclidean softmax.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

## Reuses PoincareBall from Poincare Ball (this page)
class MobiusLinear(nn.Module):
    def __init__(self, d_in, d_out, ball):
        super().__init__()
        self.ball = ball
        self.W = nn.Parameter(0.1 * torch.randn(d_out, d_in))
        ## bias kept as a tangent vector at 0 -> mapped into the ball -> any Euclidean optimizer works
        self.b = nn.Parameter(torch.zeros(d_out))

    def forward(self, x):
        ## M ⊗ x: matvec in the tangent space at the origin
        h = self.ball.expmap0(self.ball.logmap0(x) @ self.W.T)
        ## Möbius translation by a ball point: the step that is not Euclidean in disguise
        out = self.ball.mobius_add(h, self.ball.expmap0(self.b))
        return self.ball.project(out)

## Example
B = PoincareBall()
layer = MobiusLinear(3, 4, B)
x = B.expmap0(torch.randn(2, 3))  ## 2 points in the 3-D ball
y = layer(x)
print(y.shape, bool((y.norm(dim=-1) < 1).all()))  ## torch.Size([2, 4]) True
```
````

```{attention} Q&A
:class: dropdown
*Does it beat Euclidean nets?*

- Tree-like data ✅: PREFIX-10% (synthetic prefix detection), error ÷3.35 (RNN) & ÷1.5 (GRU) vs Euclidean.
- Less tree-like (PREFIX-50%) → gap shrinks. SNLI → ≈ Euclidean.
- Hyperbolic MLR ≈ Euclidean MLR on top of hyperbolic sentence embeddings.

*Why are biases the key?*

- Möbius matvecs & nonlinearities all go through $T_{\mathbf{0}}$ → stack collapses to one Euclidean net between $\log_{\mathbf{0}}$ & $\exp_{\mathbf{0}}$.
- Bias = translation away from the origin → the only step that uses curvature.

*Cons?*

- Numerics: clip to radius $1-10^{-5}$, clip $\tanh$ args to $\pm15$, clip $\operatorname{artanh}$ args to $1-10^{-5}$.
- Ball-valued params need RSGD (no momentum/Adam in the original work).
- Extra cost per layer (norms, $\tanh$, $\operatorname{artanh}$); gains depend on data.

*What does it tell us about AC/ASI?*

- Architecture = choice of space = prior on structure.
- A system that must represent nested categories may gain from a space w/ exponential capacity. Engineering evidence only.
```

&nbsp;

## Optimization on Manifolds

### Riemannian Gradient Descent
- **What**: (Stochastic) GD that steps along $\mathcal{M}$ in the direction of the metric's gradient. {cite:p}`bonnabel2013stochastic`
- **Why**: Constraints & geometries are manifolds (unit norm, orthogonality, SPD, rotations, hyperbolic).
    - Euclidean step leaves $\mathcal{M}$. Euclidean gradient ignores $g$.
    - → Constraint holds at every iterate; step sizes measured in the right metric; GD convergence theory carries over.
- **How**:
    1. Euclidean grad $\nabla\mathcal{L}$ (autodiff).
    2. Riemannian grad: $G^{-1}\nabla\mathcal{L}$ (coordinates), or project $\nabla\mathcal{L}$ onto $T_p\mathcal{M}$ (embedded, induced metric).
    3. Step: $p\leftarrow\exp_p(-\eta\operatorname{grad}\mathcal{L})$, or a cheaper [Retraction](#retraction).
    4. RSGD: stochastic grad estimate + decaying LR.

```{note} Math
:class: dropdown
Notations:
- $\operatorname{grad}\mathcal{L}(p)\in T_p\mathcal{M}$: Riemannian gradient.
- $\nabla\mathcal{L}$: Euclidean gradient (in coordinates, or of a smooth extension to $\mathbb{R}^D$).
- $P_{T_p}$: Orthogonal projection onto $T_p\mathcal{M}\subset\mathbb{R}^D$.
- $R_p$: Retraction at $p$.
- $\mathbf{h}_t$: Stochastic gradient estimate at step $t$.

Objective: $\min_{p\in\mathcal{M}}\mathcal{L}(p)$.

Riemannian gradient:

$$
g_p(\operatorname{grad}\mathcal{L}(p),\mathbf{v})=d\mathcal{L}_p(\mathbf{v})\quad\forall\mathbf{v}\in T_p\mathcal{M}
$$

- Coordinates: $\operatorname{grad}\mathcal{L}=G(\mathbf{z})^{-1}\nabla_{\mathbf{z}}\mathcal{L}$.
- Embedded, induced metric: $\operatorname{grad}\mathcal{L}=P_{T_p}\nabla\mathcal{L}$.
- $\mathbb{B}^n$: $\operatorname{grad}\mathcal{L}=\frac{(1-\|\mathbf{x}\|^2)^2}{4}\nabla\mathcal{L}$.

Process:
1. $\mathbf{h}_t$: unbiased estimate of $\operatorname{grad}\mathcal{L}(p_t)$.
2. $p_{t+1}=\exp_{p_t}(-\eta_t\mathbf{h}_t)$ or $R_{p_t}(-\eta_t\mathbf{h}_t)$.

RSGD convergence (Bonnabel):
- Assume: connected $\mathcal{M}$; injectivity radius bounded below by a positive constant; $\sum_t\eta_t=\infty$, $\sum_t\eta_t^2<\infty$; iterates stay in a compact set; $\|\mathbf{h}_t\|_g$ bounded there.
- → $\mathcal{L}(p_t)$ converges a.s. & $\operatorname{grad}\mathcal{L}(p_t)\to0$ a.s.
- Same result w/ a $C^2$ retraction instead of $\exp$.
- Hadamard manifolds: no compactness assumption; step $\eta_t$ divided by a state-dependent factor (under extra growth assumptions).
```

```{tip} Derivation
:class: dropdown
*Why is $-G^{-1}\nabla\mathcal{L}$ the steepest-descent direction?*

1. Minimize the linearized change $\nabla\mathcal{L}^T\mathbf{v}$ s.t. step size $\mathbf{v}^TG\mathbf{v}=\epsilon^2$.
2. [Lagrange](../calc.md#lagrange-multipliers): $\nabla\mathcal{L}=2\mu G\mathbf{v}$ → $\mathbf{v}\propto-G^{-1}\nabla\mathcal{L}$ (sign chosen for descent).
3. $G=I$ → ordinary GD. Different $G$ → different "steepest".

*Why is the embedded gradient a projection?*

1. Induced metric → $g_p(\mathbf{u},\mathbf{v})=\mathbf{u}^T\mathbf{v}$ on $T_p\mathcal{M}$.
2. Need $\operatorname{grad}^T\mathbf{v}=\nabla\mathcal{L}^T\mathbf{v}$ for all $\mathbf{v}\in T_p\mathcal{M}$ & $\operatorname{grad}\in T_p\mathcal{M}$.
3. → $\operatorname{grad}=P_{T_p}\nabla\mathcal{L}$; the normal component is invisible to all tangent directions.
```

```{note} Example
:class: dropdown
Leading eigenvector as RGD on the sphere:
- $\min_{\|\mathbf{p}\|=1}-\mathbf{p}^TA\mathbf{p}$, $A$ symmetric.
- $\nabla\mathcal{L}=-2A\mathbf{p}$ → $\operatorname{grad}=-2(A\mathbf{p}-(\mathbf{p}^TA\mathbf{p})\mathbf{p})$.
- $\operatorname{grad}=0$ ⇔ $A\mathbf{p}=(\mathbf{p}^TA\mathbf{p})\mathbf{p}$ ⇔ $\mathbf{p}$ is an eigenvector. Minimum → top eigenvector.
```

````{important} Code
:class: dropdown
```python
import torch

## Reuses Sphere (exp map) from Exponential Map (this page)
class SphereRGD:
    def __init__(self, lr):
        self.lr, self.S = lr, Sphere()

    def step(self, p, egrad):
        rgrad = egrad - (p @ egrad) * p        ## project onto T_p S: drop the radial part
        return self.S.exp(p, -self.lr * rgrad)  ## exact geodesic step

## Example: min -p^T A p on S^2 -> top eigenvector of A
A = torch.diag(torch.tensor([3., 1., 0.5]))
p = torch.ones(3) / 3 ** 0.5
opt = SphereRGD(lr=0.1)
for _ in range(100):
    p = opt.step(p, -2 * A @ p)
print(p)  ## ≈ tensor([1., 0., 0.])
```
````

```{attention} Q&A
:class: dropdown
*RGD vs projected GD?*

- Projected GD: Euclidean step w/ the full gradient, then project onto the constraint set.
- RGD: tangent gradient under $g$, then move along $\mathcal{M}$.
- Embedded + induced metric + projection retraction → nearly identical for small steps.
- RGD also covers non-Euclidean metrics ($\mathbb{B}^n$) & quotient manifolds (Grassmann) w/o a natural ambient projection.

*How does it relate to natural gradient?*

- Natural gradient = RGD on parameter space w/ the Fisher metric & retraction $\boldsymbol{\theta}+\mathbf{v}$ → [Natural Gradient](info.md#natural-gradient).

*How does it relate to Wasserstein gradient flows?*

- Continuous-time RGD = [Gradient Flow](../diffeq.md#gradient-flow) $\dot p=-\operatorname{grad}\mathcal{L}$.
- On probability distributions w/ the $W_2$ metric (formal Riemannian structure) → [Wasserstein Gradient Flow](ot.md#wasserstein-gradient-flow).

*What is geodesic convexity?*

- $\mathcal{L}$ convex along every geodesic → local min = global min.
- Some Euclidean-nonconvex problems are geodesically convex: $\frac12d(\cdot,q)^2$ on Hadamard manifolds → unique Fréchet mean.
- Compact manifolds w/o boundary (sphere): geodesically convex functions are constant on each connected component.

*Pros?*

- Constraints hold exactly at every iterate.
- Metric-aware steps. Continuous-time flow is invariant to reparametrization (for fixed $g$).

*Cons?*

- exp/log/transport can be costly; few manifolds have cheap closed forms.
- Momentum & Adam need vector transport ([Parallel Transport](#parallel-transport)).
- Non-compact manifolds w/o lower-bounded injectivity radius → weaker guarantees.

*Where is it used?*

- Orthogonal weights & PCA (Stiefel, Grassmann), covariance learning (SPD), rotations ($\mathrm{SO}(3)$), hyperbolic embeddings, low-rank matrix completion (fixed-rank manifold).
- Classic DL optimizers are the $G=I$ special case → [Optimization (GD)](../../dl/train/optim.md).
```

&nbsp;

#### Retraction
- **What**: Smooth map $R_p:T_p\mathcal{M}\to\mathcal{M}$ that matches the exponential map to first order. {cite:p}`absil2008optimization`
- **Why**: Exp map often costly or not in closed form. First-order agreement suffices for first-order methods.
- **How**:
    1. Step in ambient space: $p+\mathbf{v}$ (leaves $\mathcal{M}$).
    2. Map back onto $\mathcal{M}$: normalize, QR, polar decomposition, projection.
    3. Require $R_p(\mathbf{0})=p$ & $dR_p(\mathbf{0})=\mathrm{id}$ → small steps ≈ geodesic steps.

```{note} Math
:class: dropdown
Notations:
- $\mathrm{St}(D,r)$: Stiefel manifold.
- $X\in\mathrm{St}(D,r)$: Point (matrix w/ orthonormal columns).
- $V\in T_X\mathrm{St}$: Tangent vector (matrix).
- $Z\in\mathbb{R}^{D\times r}$: Ambient matrix (e.g., Euclidean gradient).
- $\operatorname{sym}(A)=\frac12(A+A^T)$.
- $\operatorname{qf}(A)$: $Q$ factor of the [QR](../linalg.md#qr) decomposition $A=QR$ w/ $\operatorname{diag}(R)>0$.

Definition:

$$
R_p(\mathbf{0})=p,\qquad dR_p(\mathbf{0})=\mathrm{id}_{T_p\mathcal{M}}
$$

- Second-order retraction: also matches $\exp_p$ in the second derivative along rays.

Stiefel manifold:

$$
\mathrm{St}(D,r)=\{X\in\mathbb{R}^{D\times r}:X^TX=I_r\},\qquad \dim=Dr-\tfrac{r(r+1)}{2}
$$

- Tangent space: $T_X\mathrm{St}=\{V:X^TV+V^TX=0\}$ ($X^TV$ skew-symmetric).
- Projection (induced metric): $P_X(Z)=Z-X\operatorname{sym}(X^TZ)$.
- QR retraction: $R_X(V)=\operatorname{qf}(X+V)$.
- Polar retraction: $R_X(V)=(X+V)(I_r+V^TV)^{-1/2}$.
- $r=1$ (sphere): both reduce to $R_{\mathbf{x}}(\mathbf{v})=\frac{\mathbf{x}+\mathbf{v}}{\|\mathbf{x}+\mathbf{v}\|}$.
```

```{tip} Derivation
:class: dropdown
*Why is polar a valid point on $\mathrm{St}$?*

1. $V$ tangent → $X^TV+V^TX=0$.
2. $(X+V)^T(X+V)=I_r+X^TV+V^TX+V^TV=I_r+V^TV$.
3. $Y=(X+V)(I_r+V^TV)^{-1/2}$ → $Y^TY=(I_r+V^TV)^{-1/2}(I_r+V^TV)(I_r+V^TV)^{-1/2}=I_r$.

*Why is QR a retraction?*

1. From step 2: $X+V$ is orthonormal up to $O(\|V\|^2)$.
2. → Gram–Schmidt changes it only by $O(\|V\|^2)$ → $\operatorname{qf}(X+V)=X+V+O(\|V\|^2)$.
3. → $R_X(\mathbf{0})=X$ & $dR_X(\mathbf{0})=\mathrm{id}$.
```

````{important} Code
:class: dropdown
```python
import torch

class StiefelQR:
    @staticmethod
    def proj(X, Z):
        ## tangent space needs X^T V skew -> remove the symmetric part of X^T Z
        XtZ = X.T @ Z
        return Z - X @ (XtZ + XtZ.T) / 2

    @staticmethod
    def retract(X, V):
        Q, R = torch.linalg.qr(X + V)  ## reduced QR: Q (D, r), R (r, r)
        ## fix column signs so diag(R) > 0 -> unique qf, & R_X(0) = X
        return Q * torch.sign(torch.diagonal(R)).unsqueeze(0)

## Example: one RGD step for min -tr(X^T A X) (top-r eigenspace)
torch.manual_seed(0)
D, r = 5, 2
A = torch.diag(torch.arange(D, 0, -1).float())  ## eigenvalues 5..1
X, _ = torch.linalg.qr(torch.randn(D, r))
egrad = -2 * A @ X
X = StiefelQR.retract(X, -0.1 * StiefelQR.proj(X, egrad))
print(torch.allclose(X.T @ X, torch.eye(r), atol=1e-5))  ## True: still orthonormal
```
````

```{attention} Q&A
:class: dropdown
*Why fix the signs of the QR factor?*

- QR is unique only up to column signs.
- Without $\operatorname{diag}(R)>0$: $R_X(\mathbf{0})$ may flip columns of $X$ → violates $R_X(\mathbf{0})=X$ → iterates jump.

*QR vs polar vs exp?*

- QR: cheapest, $O(Dr^2)$.
- Polar: closest matrix w/ orthonormal columns to $X+V$ (Frobenius norm); needs an $r\times r$ inverse square root.
- Exp: exact geodesic; most expensive.
- Cayley transform: another orthogonality-preserving retraction; needs a linear solve.

*Why constrain weights to be orthogonal?*

- All singular values $=1$ → linear map preserves norms → no exploding/vanishing signals through long chains of linear maps.
- Only the linear part: nonlinearities & gates can still cause vanishing gradients in RNNs.
- 1-Lipschitz layers → robustness bounds.

*Retraction vs parametrization?*

- Parametrization: $W=\Phi(A)$ w/ unconstrained $A$ (matrix exp, Cayley, Householder) → any Euclidean optimizer (Adam). PyTorch: `torch.nn.utils.parametrizations.orthogonal`.
- Retraction: optimize on $\mathcal{M}$ directly. Parametrization changes the optimization geometry.

*Does convergence survive replacing exp by a retraction?*

- ✅ for RSGD w/ a $C^2$ retraction (Bonnabel) → [Riemannian Gradient Descent](#riemannian-gradient-descent).
```

&nbsp;
