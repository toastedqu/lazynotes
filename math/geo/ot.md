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
# Optimal Transport
Cheapest way to move one distribution onto another, the Wasserstein geometry it induces on the space of distributions, & its computation & ML uses.
Theory from {cite:t}`villani2009optimal`. Computation from {cite:t}`peyre2019computational`.
Fisher–Rao (statistical) geometry of distributions → [Information Geometry](info.md).

Notations:
- $\mathcal{X}\subseteq\mathbb{R}^d$: Ground space.
- $d$: Ground-space dimension.
- $x$: Source point.
- $y$: Target point.
- $\mu$: Source probability measure.
- $\nu$: Target probability measure.
- $\mathcal{P}_p(\mathcal{X})$: Probability measures on $\mathcal{X}$ w/ finite $p$-th moment.
- $\delta_x$: Dirac mass at $x$.
- $c(x,y)$: Ground cost of moving a unit mass from $x$ to $y$.
- $T$: Transport map.
- $T_\#\mu$: Pushforward of $\mu$ by $T$.
- $\pi$: Coupling (transport plan).
- $\Pi(\mu,\nu)$: Couplings w/ marginals $\mu$ & $\nu$.
- $W_p$: $p$-Wasserstein distance.
- $\mathbf{a}\in\mathbb{R}_+^n$: Source weights of a discrete $\mu=\sum_ia_i\delta_{x_i}$, $\sum_ia_i=1$.
- $\mathbf{b}\in\mathbb{R}_+^m$: Target weights of a discrete $\nu=\sum_jb_j\delta_{y_j}$, $\sum_jb_j=1$.
- $C\in\mathbb{R}^{n\times m}$: Cost matrix, $C_{ij}=c(x_i,y_j)$.
- $P\in\mathbb{R}_+^{n\times m}$: Discrete plan. $P_{ij}$ = mass moved from $x_i$ to $y_j$.
- $\langle P,C\rangle=\sum_{ij}P_{ij}C_{ij}$: Total cost of plan $P$.
- $U(\mathbf{a},\mathbf{b})=\{P\in\mathbb{R}_+^{n\times m}:P\mathbf{1}=\mathbf{a},\ P^T\mathbf{1}=\mathbf{b}\}$: Transportation polytope (discrete couplings).
- $\mathrm{OT}_c(\mu,\nu)$: Min Kantorovich cost w/ ground cost $c$.
- $\mathbf{1}$: All-ones vector.
- $\rho_t$: Density at time $t$ (measures w/ a density are identified w/ the density).

Overrides of global notation: $x,y$ = source/target points, ❌model input/output. $n$/$m$ = #source/#target atoms, $i$/$j$ = source/target atom idx, ❌sample/feature. $T$ = map, $\pi$ = coupling, ❌matrix, ❌policy.

&nbsp;

## Transport Problems

### Monge Problem
- **What**: Cheapest map $T$ w/ $T_\#\mu=\nu$.
- **Why**: Move one mass distribution onto another w/ min total cost.
    - Monge (1781): move a pile of soil into a hole w/ min work.
- **How**:
    1. Send each source point $x$ to exactly one destination $T(x)$.
    2. Require: mass that arrives in each target region = $\nu$ of that region.
    3. Pay $c(x,T(x))$ per unit mass → minimize the $\mu$-average cost.

```{note} Math
:class: dropdown
Definition:

$$
\inf_{T:\,T_\#\mu=\nu}\int_\mathcal{X}c\big(x,T(x)\big)\,d\mu(x)
$$

- $T_\#\mu(B)=\mu\big(T^{-1}(B)\big)$ for every measurable $B$ ⇔ $X\sim\mu$ → $T(X)\sim\nu$.

Properties:
- Feasible set can be empty: $T_\#\delta_0=\delta_{T(0)}$ → ❌ reaches $\frac12\delta_{-1}+\frac12\delta_1$.
- Feasible set is nonconvex: the average of 2 feasible maps can be infeasible (Example).
- Inf can be unattained, even if the feasible set is non-empty.
- Densities $\rho_0,\rho_1$ & smooth injective $T$: $\rho_1\big(T(x)\big)\,\lvert\det\nabla T(x)\rvert=\rho_0(x)$ → nonlinear (Jacobian) constraint.
- Quadratic cost & $T=\nabla\varphi$ → **Monge–Ampère** equation $\det\nabla^2\varphi(x)=\rho_0(x)/\rho_1\big(\nabla\varphi(x)\big)$.
```

```{note} Example
:class: dropdown
- Assignment: $\mu=\frac1n\sum_i\delta_{x_i}$, $\nu=\frac1n\sum_j\delta_{y_j}$ → feasible maps = permutations $\sigma$ → $\min_\sigma\frac1n\sum_ic\big(x_i,y_{\sigma(i)}\big)$.
- Nonconvexity: $\mu=\nu=\frac12\delta_0+\frac12\delta_1$.
    - $T_0=\mathrm{Id}$ & $T_1$ = swap are feasible.
    - $\frac12(T_0+T_1)$ sends both atoms to $\frac12$ → $\delta_{1/2}\neq\nu$.
```

```{attention} Q&A
:class: dropdown
*Why can no feasible map exist?*

- A map sends all mass at $x$ to one point → ❌split an atom.
- Fix: allow splitting → [Kantorovich Problem](#kantorovich-problem).

*Why is it hard even if a solution exists?*

- $T_\#\mu=\nu$ is nonlinear in $T$ → nonconvex optimization.
- W/ densities → Jacobian PDE (Monge–Ampère).

*When does an optimal map exist?*

- $\mathbb{R}^d$, quadratic cost, $\mu$ absolutely continuous → ✅ unique (Brenier, see [Kantorovich Problem](#kantorovich-problem)).
- 1-D, $\mu$ atomless, convex cost $c(x-y)$ → monotone rearrangement $T=F_\nu^{-1}\circ F_\mu$ is optimal.
    - $F_\mu$: CDF of $\mu$.
    - $F_\nu^{-1}$: Quantile function of $\nu$.

*How does it relate to generative models?*

- Flow/ODE sampler = map $T$ w/ $T_\#\mathcal{N}(0,I)\approx p_\text{data}$ → [Continuous Normalizing Flow](../diffeq.md#continuous-normalizing-flow).
- Many maps push $\mathcal{N}(0,I)$ onto $p_\text{data}$. OT selects the cheapest. Likelihood training does not.
- Cheaper (straighter) map → fewer ODE steps → [Rectified Flow](../../dl/gen/diffusion.md#rectified-flow).
```

&nbsp;

### Kantorovich Problem
- **What**: Cheapest joint distribution $\pi$ w/ marginals $\mu$ & $\nu$. {cite:p}`kantorovich1942translocation`
- **Why**: Monge → ❌mass splitting → can be infeasible & is nonconvex.
    - Plan instead of map → always feasible ($\mu\otimes\nu$) + linear program → convex, min attained, has a dual.
- **How**:
    1. $\pi(x,y)$ = mass moved from $x$ to $y$. Mass at $x$ can split over many $y$.
    2. Constrain: mass that leaves $x$ = $\mu$; mass that arrives at $y$ = $\nu$.
    3. Minimize expected cost $\mathbb{E}_\pi[c]$ → linear objective + linear constraints → LP.
    4. Dual: price $f(x)$ at the source + price $g(y)$ at the target, never above $c(x,y)$ → maximize total price.

```{note} Math
:class: dropdown
Primal:

$$
\mathrm{OT}_c(\mu,\nu)=\min_{\pi\in\Pi(\mu,\nu)}\int_{\mathcal{X}\times\mathcal{X}}c(x,y)\,d\pi(x,y)
$$

- $\Pi(\mu,\nu)=\{\pi:\pi(B\times\mathcal{X})=\mu(B),\ \pi(\mathcal{X}\times B)=\nu(B)\ \forall B\}$.
- Min attained ← $c$ lower semicontinuous & bounded below.
- Map $T$ → plan $(\mathrm{Id},T)_\#\mu$ → Kantorovich ≤ Monge. Equal if an optimal plan comes from a map.

Discrete primal:

$$
\mathrm{OT}_c(\mathbf{a},\mathbf{b})=\min_{P\in U(\mathbf{a},\mathbf{b})}\langle P,C\rangle
$$

- $U(\mathbf{a},\mathbf{b})$: $nm$ variables, $n+m$ equality constraints (rank $n+m-1$).
- Optimum at a vertex → ≤ $n+m-1$ nonzeros → sparse plan.

Dual:

$$
\mathrm{OT}_c(\mu,\nu)=\sup_{f(x)+g(y)\le c(x,y)}\int f\,d\mu+\int g\,d\nu
$$

- $f,g$: Kantorovich potentials. Strong duality ← $c$ lower semicontinuous & bounded below.
- Discrete: $\max_{\mathbf{f},\mathbf{g}}\langle\mathbf{f},\mathbf{a}\rangle+\langle\mathbf{g},\mathbf{b}\rangle$ s.t. $f_i+g_j\le C_{ij}$.
- Complementary slackness: $P_{ij}>0$ → $f_i+g_j=C_{ij}$.
- Best $g$ for fixed $f$ = **c-transform** $f^c(y)=\inf_x\,c(x,y)-f(x)$.

**Brenier theorem** (Brenier, 1991):
- Assume $\mu,\nu\in\mathcal{P}_2(\mathbb{R}^d)$, $c(x,y)=\|x-y\|^2$, $\mu$ absolutely continuous.
- → Optimal plan is unique & equals $(\mathrm{Id},\nabla\varphi)_\#\mu$.
    - $\varphi$: Convex function. $\nabla\varphi$ is unique $\mu$-a.e.
- → Monge solution exists, $T=\nabla\varphi$, & Monge = Kantorovich.
```

```{tip} Derivation
:class: dropdown
*Where does the discrete dual come from?*

1. Lagrangian w/ multipliers $\mathbf{f}$ (rows) & $\mathbf{g}$ (columns):

    $$
    \mathcal{L}(P,\mathbf{f},\mathbf{g})=\langle P,C\rangle+\langle\mathbf{f},\mathbf{a}-P\mathbf{1}\rangle+\langle\mathbf{g},\mathbf{b}-P^T\mathbf{1}\rangle
    $$

2. Regroup: $\mathcal{L}=\langle\mathbf{f},\mathbf{a}\rangle+\langle\mathbf{g},\mathbf{b}\rangle+\sum_{ij}P_{ij}\,(C_{ij}-f_i-g_j)$.
3. $\min_{P\ge0}$ → $-\infty$ if some $C_{ij}-f_i-g_j<0$, else $0$.
4. → Dual: maximize $\langle\mathbf{f},\mathbf{a}\rangle+\langle\mathbf{g},\mathbf{b}\rangle$ s.t. $f_i+g_j\le C_{ij}$.
5. LP feasible & bounded → strong duality.

*Why is every dual value a lower bound (weak duality)?*

1. Feasible $(f,g)$ & any $\pi\in\Pi(\mu,\nu)$: $\int c\,d\pi\ge\int\big(f(x)+g(y)\big)\,d\pi$.
2. Marginals of $\pi$ → $\int f(x)\,d\pi=\int f\,d\mu$ & $\int g(y)\,d\pi=\int g\,d\nu$.
```

```{note} Example
:class: dropdown
- Split: $\mu=\delta_0$, $\nu=\frac12\delta_{-1}+\frac12\delta_1$ → plan $\pi=\frac12\delta_{(0,-1)}+\frac12\delta_{(0,1)}$. Monge: infeasible.
- 2 → 3 atoms, $c=(x-y)^2$:
    - $x=(0,2)$, $\mathbf{a}=(0.5,0.5)$. $y=(-1,1,3)$, $\mathbf{b}=(0.25,0.5,0.25)$.
    - $C=\begin{pmatrix}1&1&9\\9&1&1\end{pmatrix}$.
    - Mass 1 & every cost ≥ 1 → total ≥ 1. Plan w/ no mass on the 9s reaches it:

      $$
      P^*=\begin{pmatrix}0.25&0.25&0\\0&0.25&0.25\end{pmatrix},\qquad\langle P^*,C\rangle=1
      $$

    - Each source atom splits in 2 → no Monge map.
```

````{important} Code
:class: dropdown
```python
import numpy as np
from scipy.optimize import linprog

class DiscreteOT:
    def __call__(self, a, b, C):
        n, m = C.shape
        ## P flattened row-major: P[i, j] -> index i*m + j
        rows = np.kron(np.eye(n), np.ones((1, m)))   ## row i sums P[i, :] -> = a_i
        cols = np.kron(np.ones((1, n)), np.eye(m))   ## row j sums P[:, j] -> = b_j
        res = linprog(C.ravel(), A_eq=np.vstack([rows, cols]),
                      b_eq=np.concatenate([a, b]), bounds=(0, None))
        return res.x.reshape(n, m), res.fun          ## plan, OT cost

## Example: 2 -> 3 atoms on the line, squared cost
x, y = np.array([0., 2.]), np.array([-1., 1., 3.])
a, b = np.array([.5, .5]), np.array([.25, .5, .25])
P, cost = DiscreteOT()(a, b, (x[:, None] - y[None, :]) ** 2)
print(P.round(2), cost)   ## [[0.25 0.25 0.] [0. 0.25 0.25]] 1.0 -> each atom splits
```
````

```{attention} Q&A
:class: dropdown
*Uniform weights & $n=m$: why is the optimal plan a permutation?*

- Birkhoff–von Neumann: vertices of $U(\frac1n\mathbf{1},\frac1n\mathbf{1})$ = permutation matrices ÷ $n$.
- LP optimum at a vertex → Monge = Kantorovich → assignment problem.
- Solve w/ the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`). DETR matches predicted & true boxes this way.

*Complexity?*

- Exact LP solvers (network simplex, Hungarian): ~$O(n^3)$ for $n=m$, up to log factors.
- Large $n$ → [Sinkhorn](#sinkhorn-algorithm).

*What do the potentials mean?*

- Economics: $f(x)$ = price to pick up a unit at $x$; $g(y)$ = price to deliver it at $y$.
- $f(x)+g(y)\le c(x,y)$ → no route is cheaper to do yourself.
- Strong duality → max total price = min transport cost.
- Kantorovich: 1975 Nobel Memorial Prize in Economics (w/ Koopmans), optimal allocation of resources.

*Why does Brenier's theorem matter?*

- Quadratic-cost OT map = gradient of a convex function → multi-D version of "sort & match".
- Gives the map used by [displacement interpolation](#benamou-brenier-formula) & the [Gaussian closed form](#wasserstein-distance).

*What breaks if $\mu$ has atoms?*

- Atoms can force splitting → a map may fail to exist. Plan still exists & is optimal.
- Atoms alone do not rule out a map: $\delta_0\to\delta_0$ has one.
```

&nbsp;

## Wasserstein Geometry
- **What**: Metric structure that OT cost induces on $\mathcal{P}_p(\mathcal{X})$; formal Riemannian (Otto) structure only for $p=2$.

### Wasserstein Distance
- **What**: Kantorovich cost w/ ground cost $\|x-y\|^p$, to the power $1/p$.
- **Why**: Need a distance between distributions that sees ground-space geometry.
    - KL / JS / TV compare densities point by point → ❌see how far mass sits → $\infty$ or constant on disjoint supports.
- **How**:
    1. Cost of a unit mass = distance$^p$.
    2. Solve the [Kantorovich Problem](#kantorovich-problem).
    3. Take the $p$-th root → same units as $x$.
    - Shift a distribution by $\mathbf{v}$ → $W_p=\|\mathbf{v}\|$.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $F_\mu$: CDF of $\mu$ on $\mathbb{R}$.
    - $F_\mu^{-1}$: Quantile function of $\mu$.
    - $\mathbf{m}_1,\mathbf{m}_2$: Means.
    - $\Sigma_1,\Sigma_2$: Covariance matrices.

Definition:

$$
W_p(\mu,\nu)=\left(\min_{\pi\in\Pi(\mu,\nu)}\int\|x-y\|^p\,d\pi(x,y)\right)^{1/p},\qquad p\ge1
$$

- Metric on $\mathcal{P}_p(\mathcal{X})$: symmetric, $=0$ iff $\mu=\nu$, triangle inequality.
- $W_p(\delta_x,\delta_y)=\|x-y\|$ → $\mathcal{X}$ embeds isometrically in $\mathcal{P}_p(\mathcal{X})$.
- $p\le q$ → $W_p\le W_q$ ← Jensen.
- $W_p(\mu_k,\mu)\to0$ ⇔ weak convergence + convergence of $p$-th moments.

1-D:

$$
W_p^p(\mu,\nu)=\int_0^1\big\lvert F_\mu^{-1}(u)-F_\nu^{-1}(u)\big\rvert^p\,du,\qquad W_1(\mu,\nu)=\int_\mathbb{R}\lvert F_\mu(t)-F_\nu(t)\rvert\,dt
$$

- Equal-size uniform samples → sort both → match $i$-th to $i$-th.

Gaussians, $\mu=\mathcal{N}(\mathbf{m}_1,\Sigma_1)$, $\nu=\mathcal{N}(\mathbf{m}_2,\Sigma_2)$:

$$
W_2^2(\mu,\nu)=\|\mathbf{m}_1-\mathbf{m}_2\|^2+\operatorname{tr}\left(\Sigma_1+\Sigma_2-2\left(\Sigma_2^{1/2}\Sigma_1\Sigma_2^{1/2}\right)^{1/2}\right)
$$

- Symmetric: $\Sigma_2^{1/2}\Sigma_1\Sigma_2^{1/2}$ & $\Sigma_1^{1/2}\Sigma_2\Sigma_1^{1/2}$ share eigenvalues (those of $\Sigma_1\Sigma_2$) → same trace of the square root.
- Commuting $\Sigma_1,\Sigma_2$ → trace term $=\|\Sigma_1^{1/2}-\Sigma_2^{1/2}\|_F^2$.
- 1-D: $W_2^2=(m_1-m_2)^2+(\sigma_1-\sigma_2)^2$.
- Optimal map ($\Sigma_1\succ0$): $T(x)=\mathbf{m}_2+A(x-\mathbf{m}_1)$, $A=\Sigma_1^{-1/2}\left(\Sigma_1^{1/2}\Sigma_2\Sigma_1^{1/2}\right)^{1/2}\Sigma_1^{-1/2}$.
    - $A$ symmetric PSD → $T$ = gradient of a convex quadratic (Brenier).
    - Check: $A\Sigma_1A=\Sigma_2$.
```

```{tip} Derivation
:class: dropdown
*Why does the mean split off in $W_2^2$?*

1. Any coupling $(X,Y)$. Write $X=\mathbf{m}_1+\tilde X$, $Y=\mathbf{m}_2+\tilde Y$; $\tilde X,\tilde Y$ centered.
2. $\mathbb{E}\|X-Y\|^2=\|\mathbf{m}_1-\mathbf{m}_2\|^2+2\langle\mathbf{m}_1-\mathbf{m}_2,\mathbb{E}[\tilde X-\tilde Y]\rangle+\mathbb{E}\|\tilde X-\tilde Y\|^2$.
3. Middle term $=0$ ← centered.
4. Shifts map couplings of $(\mu,\nu)$ 1-to-1 onto couplings of the centered laws → $W_2^2=\|\mathbf{m}_1-\mathbf{m}_2\|^2+W_2^2(\tilde\mu,\tilde\nu)$.

*Why $W_2^2=(m_1-m_2)^2+(\sigma_1-\sigma_2)^2$ in 1-D?*

1. Quantile: $F^{-1}(u)=m+\sigma\Phi^{-1}(u)$.
    - $\Phi$: Standard normal CDF.
2. Plug into the 1-D formula, $p=2$:

    $$
    W_2^2=\int_0^1\big((m_1-m_2)+(\sigma_1-\sigma_2)\Phi^{-1}(u)\big)^2du
    $$

3. $u\sim\mathcal{U}[0,1]$ → $\Phi^{-1}(u)\sim\mathcal{N}(0,1)$ → $\int\Phi^{-1}=0$, $\int(\Phi^{-1})^2=1$.
4. → $(m_1-m_2)^2+(\sigma_1-\sigma_2)^2$.
```

```{note} Example
:class: dropdown
Parallel lines in $\mathbb{R}^2$: $Z\sim\mathcal{U}[0,1]$, $\mu_0$ = law of $(0,Z)$, $\mu_\theta$ = law of $(\theta,Z)$.

| | $\theta\ne0$ | $\theta=0$ |
|:--|:--|:--|
| $W_p(\mu_0,\mu_\theta)$ | $\lvert\theta\rvert$ | 0 |
| JS (nats) | $\log2$ | 0 |
| KL | $+\infty$ | 0 |
| TV | 1 | 0 |

- $W_p$: coupling $(0,z)\mapsto(\theta,z)$ costs $\lvert\theta\rvert$. Every coupling moves the 1st coordinate by $\lvert\theta\rvert$ → optimal.
- Only $W_p$ is continuous in $\theta$ → only $W_p$ gives a gradient toward $\theta=0$.
```

````{important} Code
:class: dropdown
```python
import numpy as np

class Wasserstein1D:
    def __init__(self, p=2):
        self.p = p

    def __call__(self, x, y):
        ## equal-size uniform samples: optimal plan = match sorted orders (quantile coupling)
        xs, ys = np.sort(x), np.sort(y)
        return np.mean(np.abs(xs - ys) ** self.p) ** (1 / self.p)

## Example
rng = np.random.default_rng(0)
x = rng.normal(size=1000)
W2 = Wasserstein1D(p=2)
print(W2(x, x + 3))       ## 3.0: pure shift -> W_p = shift size
print(W2(x, 2 * x + 3))   ## ~3.16 = sqrt(3^2 + (2-1)^2): N(0,1) vs N(3,4) Gaussian formula
```
````

```{dropdown} Table: Distances Between Distributions
| Divergence | Metric | Disjoint supports | Sees ground distance |
|:--|:--|:--|:--|
| [KL](../info.md#kl-divergence) | ❌ | $+\infty$ | ❌ |
| [JS](../info.md#jensen-shannon-divergence) | $\sqrt{\text{JS}}$ ✅ | $\log2$ (constant) | ❌ |
| TV | ✅ | 1 (constant) | ❌ |
| MMD | ✅ (characteristic kernel) | Finite, kernel-dependent | Via kernel |
| $W_p$ | ✅ | Grows w/ distance | ✅ |
```

```{attention} Q&A
:class: dropdown
*Why is it meaningful for disjoint supports when KL & JS are not?*

- KL & JS depend only on density ratios at each point.
- Disjoint supports → every ratio is 0 or $\infty$ → value fixed, independent of how far apart the supports are.
- $W_p$ pays per unit distance moved → shrinks continuously as the supports approach.

*$W_1$ vs $W_2$?*

- $W_1$: [K-R dual](#kantorovich-rubinstein-duality) → 1 critic function → WGAN. Linear cost → less sensitive to far outliers.
- $W_2$: Brenier map, Gaussian closed form, geodesics & [gradient flows](#wasserstein-gradient-flow).

*What is FID?*

- Fréchet Inception Distance = $W_2^2$ between Gaussians fit to Inception-v3 pool features (2048-d) of real & generated images.
- Uses the Gaussian formula w/ $\operatorname{tr}\big((\Sigma_1\Sigma_2)^{1/2}\big)$ (same eigenvalues → same value).
- Caveats: features are not Gaussian; biased at small sample sizes.

*How many samples to estimate $W_p$?*

- Empirical $\hat\mu_m$ from $m$ samples, bounded support, $d\ge3$: $\mathbb{E}\,W_1(\mu,\hat\mu_m)=O(m^{-1/d})$ → [curse of dimensionality](../../ml/misc.md#curse-of-dimensionality).
    - Rate attained, e.g., by uniform $\mu$ on $[0,1]^d$.
- Faster alternatives: [Sinkhorn divergence](#sinkhorn-algorithm), sliced $W_p$ → they change the target discrepancy, ❌estimate $W_p$ itself.

*What is sliced Wasserstein?*

- $SW_p(\mu,\nu)=\left(\mathbb{E}_\theta\,W_p^p(\theta_\#\mu,\theta_\#\nu)\right)^{1/p}$, $\theta$ uniform on the unit sphere.
    - $\theta_\#\mu$: Law of the 1-D projection $\langle\theta,x\rangle$, $x\sim\mu$.
- Each projection: sort → $O(m\log m)$. ❌Same value as $W_p$.
- Exact sphere integral → metric. Finite random projections → approximation; can fail to separate distinct distributions.

*Where else is it used in ML?*

- GAN loss → [WGAN](#wgan). Evaluation → FID.
- Distributional RL: distributional Bellman operator is a $\gamma$-contraction in max-$W_p$ → [C51](../../rl/value.md#c51).
- Domain adaptation: transport source features onto target features (Courty et al., 2017).
- Text: Word Mover's Distance = $W_1$ between word-embedding clouds of 2 documents (Kusner et al., 2015).
- Flow models: minibatch OT pairs of (noise, data) → straighter paths (Pooladian et al., 2023; Tong et al., 2024) → [Flow Matching](../../dl/gen/diffusion.md#flow-matching).

*What does it tell us about AC/ASI?*

- Beliefs as distributions over world states: "object 1 cm off" vs "object 1 km off".
    - Disjoint supports → KL = $\infty$ for both. $W_p$ separates them.
- → $W_p$ measures how wrong a belief is in world units. KL measures surprise only.
- Hypothesis, not established: agents that need graded "almost right" signals over physical state may benefit from transport-based comparisons.
```

&nbsp;

#### Kantorovich-Rubinstein Duality
- **What**: $W_1(\mu,\nu)=\sup_{\|f\|_L\le1}\mathbb{E}_\mu f-\mathbb{E}_\nu f$.
- **Why**: Primal needs a coupling over $\mathcal{X}\times\mathcal{X}$ → intractable in high $d$.
    - Dual needs 1 function & 2 sample means → parameterize $f$ by a NN → [WGAN](#wgan) critic.
- **How**:
    - Cost = metric → best partner potential of a 1-Lipschitz $f$ is $g=-f$ → 2 potentials collapse to 1.
    - $f$ = height function that rises ≤ 1 per unit distance. $W_1$ = largest gap in mean height.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\|f\|_L=\sup_{x\ne y}\frac{\lvert f(x)-f(y)\rvert}{\|x-y\|}$: Lipschitz constant.

Definition:

$$
W_1(\mu,\nu)=\sup_{\|f\|_L\le1}\ \mathbb{E}_{x\sim\mu}[f(x)]-\mathbb{E}_{y\sim\nu}[f(y)]
$$

- Holds for $\mu,\nu\in\mathcal{P}_1(\mathcal{X})$ & any metric ground cost.
- Sup over $\|f\|_L\le K$ → $K\,W_1$.
- $f$ & $f+\text{const}$ → same value.
```

```{tip} Derivation
:class: dropdown
1. Kantorovich dual w/ $c(x,y)=\|x-y\|$: $\sup\int f\,d\mu+\int g\,d\nu$ s.t. $f(x)+g(y)\le\|x-y\|$.
2. Replace $f$ by $g^c(x)=\inf_y\|x-y\|-g(y)$ → still feasible, objective ⬆️ or equal.
3. $g^c$ = inf of 1-Lipschitz functions of $x$ → 1-Lipschitz → restrict to 1-Lipschitz $f$.
4. For 1-Lipschitz $f$: $f^c(y)=\inf_x\|x-y\|-f(x)$.
    - Lipschitz → $\|x-y\|-f(x)\ge-f(y)$.
    - $x=y$ → equality → $f^c=-f$.
5. Best $g=f^c=-f$ → $W_1=\sup_{\|f\|_L\le1}\int f\,d\mu-\int f\,d\nu$.
```

```{attention} Q&A
:class: dropdown
*Why does it only work for $p=1$?*

- Step 4 needs the cost to be a metric. $\|x-y\|^p$ for $p>1$ violates the triangle inequality.
- → $f^c\ne-f$ → keep 2 potentials.

*How does it relate to other distances?*

- Integral probability metric (IPM): $\sup_{f\in\mathcal{F}}\mathbb{E}_\mu f-\mathbb{E}_\nu f$.
    - $\mathcal{F}$ = 1-Lipschitz → $W_1$.
    - $\mathcal{F}$ = unit ball of an RKHS → MMD.
    - $\mathcal{F}=\{\|f\|_\infty\le1\}$ → $2\,\mathrm{TV}$.
- Function class sets which differences count. Lipschitz class → geometry-aware.

*Why is the critic not a classifier?*

- $f$ outputs an unbounded score, not a probability.
- Only the difference of means matters → no sigmoid, no log.
```

&nbsp;

#### Benamou-Brenier Formula
- **What**: $W_2^2$ = min kinetic energy of a mass flow that morphs $\mu$ into $\nu$ in unit time. {cite:p}`benamou2000computational`
- **Why**: Static coupling → dynamic picture: OT as fluid flow.
    - → geodesics in $(\mathcal{P}_2,W_2)$, Riemannian structure, links to flow models & gradient flows.
- **How**: Each particle moves at constant speed on a straight line from $x$ to $T(x)$ → intermediate densities = geodesic (**displacement interpolation**).

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\mathbf{v}_t(x)$: Velocity field at time $t$.

Definition:

$$
W_2^2(\mu,\nu)=\min_{(\mu_t,\mathbf{v}_t)}\int_0^1\!\!\int\|\mathbf{v}_t(x)\|^2\,d\mu_t(x)\,dt\quad\text{s.t.}\quad\partial_t\mu_t+\nabla\cdot(\mu_t\mathbf{v}_t)=0,\ \mu_0=\mu,\ \mu_1=\nu
$$

- $\mu_t$: Curve of probability measures, $t\in[0,1]$.
- Constraint = [continuity equation](../diffeq.md#continuity-equation) in the distributional (weak) sense → mass conserved.
    - Atomic endpoints allowed, e.g., $\delta_0\to\delta_1$.
- Density case $d\mu_t=\rho_t\,dx$ → cost $\int_0^1\!\int\|\mathbf{v}_t\|^2\rho_t\,dx\,dt$.
- Optimal $\mathbf{v}_t=\nabla\phi_t$, $\partial_t\phi_t+\frac12\|\nabla\phi_t\|^2=0$ (Hamilton–Jacobi).
    - $\phi_t$: Velocity potential.

Displacement interpolation (McCann), $T$ = Brenier map:

$$
\mu_t=\big((1-t)\,\mathrm{Id}+t\,T\big)_\#\mu,\qquad W_2(\mu_s,\mu_t)=\lvert t-s\rvert\,W_2(\mu,\nu)
$$

- → constant-speed [geodesic](manifold.md#geodesic) in $(\mathcal{P}_2,W_2)$.
- ≠ mixture $(1-t)\mu+t\nu$: mixture reweights the 2 endpoints in place; displacement moves mass.
```

```{note} Example
:class: dropdown
$\mu=\mathcal{N}(m_1,\sigma_1^2)$, $\nu=\mathcal{N}(m_2,\sigma_2^2)$ on $\mathbb{R}$:
- Brenier map $T(x)=m_2+\frac{\sigma_2}{\sigma_1}(x-m_1)$.
- Displacement: $\mu_t=\mathcal{N}\big((1-t)m_1+tm_2,\ ((1-t)\sigma_1+t\sigma_2)^2\big)$ → 1 bump slides & reshapes.
- Mixture: $(1-t)\mu+t\nu$ → reweights the 2 fixed components; 1 or 2 modes.
```

```{attention} Q&A
:class: dropdown
*How does it relate to flow matching?*

- [Flow Matching](../../dl/gen/diffusion.md#flow-matching) learns $\mathbf{v}_t$ that satisfies the continuity equation between noise & data.
- Straight conditional paths = displacement interpolation only if pairs come from the OT coupling.
- Independent pairs → valid flow, ❌min kinetic energy → curved marginal paths.
- Minibatch OT pairing or [reflow](../../dl/gen/diffusion.md#rectified-flow) → closer to OT → straighter.

*Why is $W_2$ a geodesic distance?*

- Formula = min "length²" over curves $\mu_t$ w/ speed $\|\mathbf{v}_t\|_{L^2(\mu_t)}$ → same form as a Riemannian distance → [Riemannian Metric](manifold.md#riemannian-metric).
- Formal tangent vectors = gradient fields $\nabla\phi$ → [Otto calculus](#wasserstein-gradient-flow).

*What does it tell us about AC?*

- 2 ways to interpolate beliefs: mixture = "either A or B"; displacement = "the object moved halfway".
- Transport interpolation fits beliefs about things that move continuously.
```

&nbsp;

### Wasserstein Gradient Flow
- **What**: Steepest descent of a functional $F(\rho)$ in the $W_2$ geometry. {cite:p}`jordan1998variational`
- **Why**: Many density PDEs (Fokker–Planck, heat, porous medium) = $W_2$ gradient flows of an energy.
    - → convergence proofs via convexity, stable time steps, "sampling = optimization over distributions".
- **How**: **JKO scheme** = implicit Euler step in $W_2$:
    1. Start at $\rho_k$.
    2. Find $\rho$ that lowers $F$ but pays $\frac{1}{2\tau}W_2^2(\rho,\rho_k)$ to move.
    3. Set $\rho_{k+1}$ = that minimizer.
    4. $\tau\to0$ → continuous flow.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\tau$: Time step.
- Misc:
    - $\rho_k$: $k$-th JKO iterate.
    - $F:\mathcal{P}_2\to\mathbb{R}\cup\{+\infty\}$: Energy functional.
    - $\frac{\delta F}{\delta\rho}$: First variation of $F$.
    - $V$: Potential.
    - $\beta$: Inverse temperature.
    - $B_t$: Brownian motion.

Process (JKO):

$$
\rho_{k+1}=\arg\min_{\rho\in\mathcal{P}_2}\ \frac{1}{2\tau}W_2^2(\rho,\rho_k)+F(\rho)
$$

- Euclidean analogue: $x_{k+1}=\arg\min_x\frac{1}{2\tau}\|x-x_k\|^2+F(x)$ = [implicit Euler](../diffeq.md#implicit-euler) for [gradient flow](../diffeq.md#gradient-flow) $\dot x=-\nabla F$.

Continuous limit:

$$
\partial_t\rho_t=\nabla\cdot\left(\rho_t\,\nabla\frac{\delta F}{\delta\rho}(\rho_t)\right)
$$

- = continuity equation w/ velocity $\mathbf{v}_t=-\nabla\frac{\delta F}{\delta\rho}$.
- Energy dissipation: $\frac{d}{dt}F(\rho_t)=-\int\rho_t\left\|\nabla\frac{\delta F}{\delta\rho}\right\|^2dx\le0$.

Fokker–Planck = $W_2$ gradient flow of the free energy:

$$
F(\rho)=\int V\rho\,dx+\beta^{-1}\int\rho\log\rho\,dx\quad\Rightarrow\quad\partial_t\rho=\nabla\cdot(\rho\nabla V)+\beta^{-1}\Delta\rho
$$

- $\Delta$: Laplacian.
- Particle view: $dX_t=-\nabla V(X_t)\,dt+\sqrt{2\beta^{-1}}\,dB_t$ = [Langevin dynamics](../prob/process.md#langevin-dynamics) w/ $U=V$, $T_\text{temp}=\beta^{-1}$.
- $Z=\int e^{-\beta V}dx<\infty$ → $\rho_\infty=Z^{-1}e^{-\beta V}$ (Gibbs) & $F(\rho)=\beta^{-1}D_{KL}(\rho\|\rho_\infty)-\beta^{-1}\log Z$.
    - $Z$: Partition function.
- $\rho_\infty\in\mathcal{P}_2$ → $\rho_\infty$ minimizes $F$ on $\mathcal{P}_2$ & is stationary.
- $Z=\infty$ (e.g., $V=0$ on $\mathbb{R}^d$) → ❌equilibrium.
- JKO convergence assumptions: smooth $V\ge0$, $\|\nabla V\|\le C(V+1)$; $\rho_0\in\mathcal{P}_2$ w/ $F(\rho_0)<\infty$.
- $\nabla^2V\succeq\lambda I$, $\lambda>0$ → $F$ is $\lambda$-convex along $W_2$ geodesics → $W_2(\rho_t,\rho_\infty)\le e^{-\lambda t}W_2(\rho_0,\rho_\infty)$.
```

```{tip} Derivation
:class: dropdown
*Why is Fokker–Planck the $W_2$ gradient flow of $F$?*

1. First variation: $\frac{\delta F}{\delta\rho}=V+\beta^{-1}(\log\rho+1)$.
2. Gradient: $\nabla\frac{\delta F}{\delta\rho}=\nabla V+\beta^{-1}\frac{\nabla\rho}{\rho}$.
3. Multiply by $\rho$: $\rho\nabla V+\beta^{-1}\nabla\rho$.
4. Divergence: $\partial_t\rho=\nabla\cdot(\rho\nabla V)+\beta^{-1}\Delta\rho$.
5. Check vs the SDE: [Fokker–Planck](../prob/process.md#fokker-planck-equation) w/ $f=-\nabla V$, $g=\sqrt{2\beta^{-1}}$ → $-\nabla\cdot(f\rho)+\frac12g^2\Delta\rho$ = same PDE.

*Why does $F$ decrease?*

1. $\frac{d}{dt}F(\rho_t)=\int\frac{\delta F}{\delta\rho}\,\partial_t\rho_t\,dx=\int\frac{\delta F}{\delta\rho}\,\nabla\cdot\left(\rho_t\nabla\frac{\delta F}{\delta\rho}\right)dx$.
2. Integrate by parts (boundary terms vanish) → $-\int\rho_t\left\|\nabla\frac{\delta F}{\delta\rho}\right\|^2dx\le0$.
```

```{note} Example
:class: dropdown
- Heat equation: $V=0$, $\beta=1$ → $\partial_t\rho=\Delta\rho$ = $W_2$ gradient flow of $\int\rho\log\rho$ (negative entropy).
    - $\rho_0=\mathcal{N}(0,\sigma_0^2)$ → $\rho_t=\mathcal{N}(0,\sigma_0^2+2t)$ → entropy ⬆️.
- OU: $V=\frac\lambda2\|x\|^2$ → $\rho_\infty=\mathcal{N}(0,(\lambda\beta)^{-1}I)$; means contract as $e^{-\lambda t}$.
- Code for the particle view → `LangevinSampler` in [Langevin Dynamics](../prob/process.md#langevin-dynamics).
```

```{attention} Q&A
:class: dropdown
*What is Otto calculus?*

- Treat $(\mathcal{P}_2,W_2)$ as a formal Riemannian manifold (Otto, 2001).
    - Tangent vector at $\rho$ = velocity field $\nabla\phi$.
    - Metric: $\langle\nabla\phi_1,\nabla\phi_2\rangle_\rho=\int\rho\,\nabla\phi_1\cdot\nabla\phi_2\,dx$.
    - Geodesic distance = $W_2$ ([Benamou-Brenier](#benamou-brenier-formula)).
- → $W_2$ gradient of $F$ = the PDE above. Finite-dim analogue → [Riemannian Metric](manifold.md#riemannian-metric).

*Why implicit (JKO), not explicit Euler?*

- Explicit step needs $\nabla\log\rho$ → undefined for empirical (particle) measures.
- Implicit step = well-posed optimization; stable for any $\tau$, like implicit Euler.

*How does it relate to Langevin sampling?*

- Particles follow the Langevin SDE. Their density follows Fokker–Planck = $W_2$ descent on $F$.
- $Z<\infty$ → $F=\beta^{-1}D_{KL}(\rho\|\rho_\infty)$ + const → descent on KL to the Gibbs target.
- → Langevin sampling = optimization of KL over distributions. ULA = noisy explicit discretization.

*How does it relate to diffusion models?*

- Forward VP-SDE $dx=-\frac12\beta(t)x\,dt+\sqrt{\beta(t)}\,dB_t$ = Langevin toward $\mathcal{N}(0,I)$ after time change $s=\frac12\int_0^t\beta$.
    - $\beta(t)$: noise schedule here, ❌inverse temperature.
- → Noising = $W_2$ gradient flow of $D_{KL}(\rho\|\mathcal{N}(0,I))$.
- Generation = time reversal → needs the score $\nabla\log\rho_t$ → learned → [Score SDE](../../dl/gen/diffusion.md#score-sde).

*Fisher–Rao vs Wasserstein?*

- 2 geometries on distributions.
    - [Fisher–Rao](info.md#fisher-rao-metric): reweights mass in place; ignores ground distance; invariant to reparameterizing $x$.
    - $W_2$: moves mass; uses ground distance; changes if $x$ is reparameterized.
- Same KL, 2 flows:
    - $W_2$ → Fokker–Planck (transport).
    - Fisher–Rao → birth–death $\partial_t\rho=-\rho\left(\log\frac{\rho}{\rho_\infty}-\mathbb{E}_\rho\log\frac{\rho}{\rho_\infty}\right)$ → ❌creates mass where $\rho=0$.

*How does it relate to NN training?*

- Mean-field limit: 2-layer NN, #neurons $\to\infty$ → neuron-param distribution follows a $W_2$ gradient flow of the risk (Mei, Montanari & Nguyen, 2018; Chizat & Bach, 2018).
- Global-convergence results in that limit need extra structural & initialization assumptions. Infinite width alone ❌suffices.
    - Chizat & Bach also assume the limiting flow converges.

*What does it tell us about intelligence/AC?*

- Learning (mean-field), sampling (Langevin) & noising (diffusion) = descent of a free energy over distributions.
- Free energy here = energy + temperature × negative entropy, w/ $W_2$ geometry.
- Caveat: Friston's free energy in [Predictive Processing](../../ac/theories.md#predictive-processing-pp) is variational (KL-based), not this $W_2$ flow. Shared name ≠ shared geometry.
```

&nbsp;

## Computational OT

### Sinkhorn Algorithm
- **What**: Entropy-regularized OT solved by alternate row & column rescaling of $K=e^{-C/\varepsilon}$. {cite:p}`cuturi2013sinkhorn`
- **Why**: Exact OT LP: ~$O(n^3)$, sequential, nonsmooth in its inputs → ❌GPU, ❌easy backprop.
    - Entropy term → smooth, strictly convex problem w/ a matrix-scaling solution → only mat-vec products.
- **How**:
    1. Kernel $K_{ij}=e^{-C_{ij}/\varepsilon}$ → cheap routes get large weight.
    2. Rescale rows → row sums $=\mathbf{a}$.
    3. Rescale columns → column sums $=\mathbf{b}$.
    4. Repeat 2–3 until both marginals match.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\varepsilon>0$: Entropic regularization strength (cost units).
- Misc:
    - $H(P)=-\sum_{ij}P_{ij}(\log P_{ij}-1)$: Entropy of a plan.
    - $K=e^{-C/\varepsilon}$ (elementwise): Gibbs kernel.
    - $\mathbf{u}\in\mathbb{R}_+^n$: Row scaling.
    - $\mathbf{v}\in\mathbb{R}_+^m$: Column scaling.
    - $\mathbf{f}=\varepsilon\log\mathbf{u}$: Row dual potential.
    - $\mathbf{g}=\varepsilon\log\mathbf{v}$: Column dual potential.
    - $\oslash$: Elementwise division.

Objective:

$$
P^\varepsilon=\arg\min_{P\in U(\mathbf{a},\mathbf{b})}\langle P,C\rangle-\varepsilon H(P)
$$

- $=\arg\min_{P\in U(\mathbf{a},\mathbf{b})}\mathrm{KL}(P\|K)$ (generalized KL) → [I-projection](info.md#information-projection) of $K$ onto $U(\mathbf{a},\mathbf{b})$.
- Solution form: $P^\varepsilon=\operatorname{diag}(\mathbf{u})\,K\,\operatorname{diag}(\mathbf{v})$.

Process:
1. $\mathbf{v}\leftarrow\mathbf{1}$.
2. $\mathbf{u}\leftarrow\mathbf{a}\oslash K\mathbf{v}$.
3. $\mathbf{v}\leftarrow\mathbf{b}\oslash K^T\mathbf{u}$.
4. Repeat 2–3.
- $O(nm)$ per iteration (2 mat-vecs). Batched → GPU.
- Linear convergence. #iterations ⬆️ as $\varepsilon$ ⬇️.

Log domain (stable):

$$
f_i\leftarrow\varepsilon\log a_i-\varepsilon\log\sum_je^{(g_j-C_{ij})/\varepsilon},\qquad g_j\leftarrow\varepsilon\log b_j-\varepsilon\log\sum_ie^{(f_i-C_{ij})/\varepsilon}
$$

- $P^\varepsilon_{ij}=e^{(f_i+g_j-C_{ij})/\varepsilon}$.
- Logsumexp → ❌underflow of $e^{-C/\varepsilon}$ for small $\varepsilon$.

Dual:

$$
\max_{\mathbf{f},\mathbf{g}}\ \langle\mathbf{f},\mathbf{a}\rangle+\langle\mathbf{g},\mathbf{b}\rangle-\varepsilon\sum_{ij}e^{(f_i+g_j-C_{ij})/\varepsilon}
$$

- Smooth penalty replaces the hard constraint $f_i+g_j\le C_{ij}$.
- Sinkhorn = exact block-coordinate ascent: $\mathbf{f}$-step, then $\mathbf{g}$-step.

Limits:
- $\varepsilon\to0$ → $P^\varepsilon$ → max-entropy optimal plan of the LP. Cost → $\mathrm{OT}_c$.
- $\varepsilon\to\infty$ → $P^\varepsilon\to\mathbf{a}\mathbf{b}^T$ (independent coupling).
- $\langle P^\varepsilon,C\rangle\ge\mathrm{OT}_c$ ← $P^\varepsilon$ is feasible.
```

```{tip} Derivation
:class: dropdown
*Why the scaling form?*

1. Lagrangian: $\langle P,C\rangle+\varepsilon\sum_{ij}P_{ij}(\log P_{ij}-1)-\langle\mathbf{f},P\mathbf{1}-\mathbf{a}\rangle-\langle\mathbf{g},P^T\mathbf{1}-\mathbf{b}\rangle$.
2. $\partial/\partial P_{ij}=0$: $C_{ij}+\varepsilon\log P_{ij}-f_i-g_j=0$.
3. → $P_{ij}=e^{f_i/\varepsilon}\,e^{-C_{ij}/\varepsilon}\,e^{g_j/\varepsilon}=u_iK_{ij}v_j$.
    - Assume finite $C_{ij}$ & $a_i,b_j>0$ (drop zero-weight atoms first) → every $P_{ij}>0$ → constraint $P\ge0$ inactive.
4. Row constraint $\operatorname{diag}(\mathbf{u})K\mathbf{v}=\mathbf{a}$ → $\mathbf{u}=\mathbf{a}\oslash K\mathbf{v}$. Columns: same.

*Why the generalized-KL form?*

1. $\langle P,C\rangle=-\varepsilon\sum_{ij}P_{ij}\log K_{ij}$.
2. → $\langle P,C\rangle-\varepsilon H(P)=\varepsilon\sum_{ij}\left(P_{ij}\log\frac{P_{ij}}{K_{ij}}-P_{ij}+K_{ij}\right)-\varepsilon\sum_{ij}K_{ij}$.
3. Last sum is constant → same minimizer as $\mathrm{KL}(P\|K)$.

*Why $\varepsilon\to\infty$ → $\mathbf{a}\mathbf{b}^T$?*

1. Objective dominated by $-\varepsilon H(P)$ → maximize entropy on $U(\mathbf{a},\mathbf{b})$.
2. Max-entropy joint w/ fixed marginals = product of marginals.
```

```{note} Example
:class: dropdown
Same 2 → 3 problem as [Kantorovich Problem](#kantorovich-problem), $\varepsilon=0.1$:
1. $\mathbf{g}=\mathbf{0}$. Row 0: $-C_{0j}/\varepsilon=(-10,-10,-90)$ → $f_0=0.1(\log0.5+10-\log2)=1-0.2\log2$. Same for $f_1$.
2. Column 0: exponents $(f_0-1)/0.1=-2\log2$ & $\approx-80$ → $g_0=0.1(\log0.25+2\log2)=0$. Same for $g_1,g_2$.
3. $\mathbf{g}$ unchanged → converged after 1 iteration.
4. $P_{00}=e^{(f_0-1)/0.1}=e^{-2\log2}=0.25$; $P_{02}=0.25\,e^{-80}\approx0$ → $P^\varepsilon\approx P^*$.
```

````{important} Code
:class: dropdown
```python
import torch

class Sinkhorn:
    def __init__(self, eps=0.1, n_iters=200):
        self.eps, self.n_iters = eps, n_iters

    def __call__(self, a, b, C):
        f, g = torch.zeros_like(a), torch.zeros_like(b)   ## f = eps*log u, g = eps*log v
        for _ in range(self.n_iters):
            ## u = a / (K v) in log domain: logsumexp never forms exp(-C/eps) -> no underflow
            f = self.eps * (a.log() - torch.logsumexp((g[None, :] - C) / self.eps, dim=1))
            ## v = b / (K^T u): after this step column sums are exactly b
            g = self.eps * (b.log() - torch.logsumexp((f[:, None] - C) / self.eps, dim=0))
        P = torch.exp((f[:, None] + g[None, :] - C) / self.eps)
        return P, (P * C).sum()        ## plan, transport cost <P, C>; differentiable in a, b, C

## Example: same 2 -> 3 problem as the Kantorovich code
x, y = torch.tensor([0., 2.]), torch.tensor([-1., 1., 3.])
a, b = torch.tensor([.5, .5]), torch.tensor([.25, .5, .25])
C = (x[:, None] - y[None, :]) ** 2
P, cost = Sinkhorn(eps=0.1)(a, b, C)
print(P.round(decimals=2), cost)   ## [[0.25,0.25,0.],[0.,0.25,0.25]], ~1.0 -> matches the LP
P_big, _ = Sinkhorn(eps=1e3)(a, b, C)
print((P_big - torch.outer(a, b)).abs().max())   ## small (< 1e-2): eps -> inf gives independent coupling
```
````

```{dropdown} Table: Exact LP vs Sinkhorn
| | Exact LP | Sinkhorn |
|:--|:--|:--|
| Cost | ~$O(n^3)$ for $n=m$, up to logs | $O(nm)$ per iteration |
| Plan | Sparse (≤ $n+m-1$ nonzeros) | Dense (every $P_{ij}>0$ on retained supports) |
| Gradient w.r.t. $C$, $\mathbf{a}$, $\mathbf{b}$ | Nonsmooth | Smooth |
| Hardware | CPU, sequential | GPU, batched |
| Value | Exact $\mathrm{OT}_c$ | Blurred; $\langle P^\varepsilon,C\rangle\ge\mathrm{OT}_c$ |
```

```{attention} Q&A
:class: dropdown
*How to choose $\varepsilon$?*

- $\varepsilon$ is in cost units → set it relative to the scale of $C$.
- Small → close to OT, slow, underflow risk (use log domain).
- Large → fast, blurry plan, biased cost.

*Why GPU-friendly & differentiable?*

- Only mat-vecs, logsumexp & elementwise ops → batch many problems.
- Gradients: unroll the iterations w/ autodiff, or use converged potentials (envelope theorem: $\nabla_\mathbf{a}$ of the regularized optimum $=\mathbf{f}$, up to a constant).

*What is the Sinkhorn divergence?*

- Regularized cost of $\mu$ vs itself ≠ 0 → entropic bias.
- Debias: $S_\varepsilon(\mu,\nu)=\mathrm{OT}_\varepsilon(\mu,\nu)-\frac12\mathrm{OT}_\varepsilon(\mu,\mu)-\frac12\mathrm{OT}_\varepsilon(\nu,\nu)$ → $S_\varepsilon(\mu,\mu)=0$.
    - $\mathrm{OT}_\varepsilon$: regularized OT value w/ $\mathrm{KL}(\pi\|\mu\otimes\nu)$ penalty.
- $\varepsilon\to0$ → OT (Genevay et al., 2018; Feydy et al., 2019).
- $\varepsilon\to\infty$, symmetric $c$ → $\mathbb{E}\,c(X,Y)-\frac12\mathbb{E}\,c(X,X')-\frac12\mathbb{E}\,c(Y,Y')$.
    - $X,X'\sim\mu$ & $Y,Y'\sim\nu$, all independent.
    - $-c$ conditionally positive definite → $=\frac12\mathrm{MMD}^2$ w/ kernel $-c$.
- Fixed $\varepsilon>0$, bounded supports, smooth cost → estimation error $O(m^{-1/2})$. Bounds worsen as $\varepsilon\to0$ (Genevay et al., 2019).

*How does it relate to information geometry?*

- Each half-step = exact KL projection onto 1 marginal constraint set → alternating [I-projections](info.md#information-projection).
- Same algorithm as iterative proportional fitting (IPF) & matrix scaling (Sinkhorn, 1964).

*How does it relate to attention?*

- $K=e^{-C/\varepsilon}$ = softmax numerators w/ logits $-C/\varepsilon$.
- Softmax attention normalizes rows only. Sinkhorn alternates rows & columns → doubly stochastic attention (Sinkformers; Sander et al., 2022).
```

&nbsp;

## OT in ML

### WGAN
- **Name**: Wasserstein Generative Adversarial Network {cite:p}`arjovsky2017wasserstein`
- **What**: GAN whose generator minimizes a critic's estimate of $W_1(p_\text{data},p_\theta)$.
- **Why**: Standard GAN ≈ minimizes JS → vanishing or unstable gradients.
    - Data & model lie on low-dim manifolds → supports ~disjoint → JS ≈ $\log2$ (constant) → ❌signal.
    - Under regularity assumptions: $W_1$ continuous & differentiable a.e. in generator params.
    - Nonzero useful gradient: ❌guaranteed.
- **How**:
    1. Critic $f_w$ outputs a scalar score; no sigmoid.
    2. Critic steps ($n_\text{critic}=5$): maximize $\mathbb{E}_\text{data}f_w-\mathbb{E}_{p_\theta}f_w$ → [K-R dual](#kantorovich-rubinstein-duality).
    3. After each critic step: clip every weight to $[-c,c]$ → $f_w$ is $K$-Lipschitz.
    4. Generator step: minimize $-\mathbb{E}_zf_w(G_\theta(z))$.
    - Paper defaults: RMSProp, LR $5\times10^{-5}$, $c=0.01$, batch 64.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $G_\theta$: Generator.
    - $f_w$: Critic.
- Hyperparams:
    - $c$: Weight-clipping bound (here $c$ ≠ ground cost).
    - $n_\text{critic}$: Critic steps per generator step.
- Misc:
    - $z\sim p_z$: Latent noise.
    - $p_\text{data}$: Data distribution.
    - $f^*$: Optimal critic.

Model:

$$
p_\theta=(G_\theta)_\#p_z
$$

Training:

$$
\min_\theta\ \max_{w\in[-c,c]^{|w|}}\ \mathbb{E}_{x\sim p_\text{data}}[f_w(x)]-\mathbb{E}_{z\sim p_z}[f_w(G_\theta(z))]
$$

- Clipped family ⊂ $K$-Lipschitz functions, $K$ depends on $c$ & architecture.
- Inner max ≤ $K\,W_1(p_\text{data},p_\theta)$. Equality if the family contains a maximizer.
- Generator gradient: $\nabla_\theta W_1(p_\text{data},p_\theta)=-\mathbb{E}_z\big[\nabla_\theta f^*(G_\theta(z))\big]$.

Properties:
- Compact $\mathcal{X}$ & $G_\theta$ continuous in $\theta$ → $W_1(p_\text{data},p_\theta)$ continuous in $\theta$.
- Unbounded $\mathcal{X}$ → needs extra moment / uniform-integrability control.
- $G_\theta$ locally Lipschitz w/ integrable local constants (e.g., feedforward NN, $\mathbb{E}\|z\|<\infty$) → $W_1$ continuous everywhere & differentiable a.e.
- JS & KL: ❌ these guarantees.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class WGANCritic:
    def __init__(self, critic, clip=0.01):
        self.f, self.clip = critic, clip

    def critic_loss(self, x_real, x_fake):
        ## K-R dual: maximize E_data f - E_G f -> minimize its negative
        return -(self.f(x_real).mean() - self.f(x_fake).mean())

    def generator_loss(self, x_fake):
        ## only E_G f depends on the generator
        return -self.f(x_fake).mean()

    @torch.no_grad()
    def clip_weights(self):
        ## box-constrained weights -> f is K-Lipschitz for some K (crude Lipschitz control)
        for p in self.f.parameters():
            p.clamp_(-self.clip, self.clip)

## Example
f = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))   ## no sigmoid
wgan = WGANCritic(f)
x_real, x_fake = torch.randn(8, 2) + 3, torch.randn(8, 2)
loss = wgan.critic_loss(x_real, x_fake)
loss.backward()            ## then an RMSProp step on f
wgan.clip_weights()
print(loss.shape)          ## torch.Size([]); -loss = critic's estimate of K * W1
```
````

```{attention} Q&A
:class: dropdown
*Why does JS give vanishing gradients?*

- Supports on low-dim manifolds → intersection has measure 0 → a perfect discriminator exists.
- → JS stuck at $\log2$ → minimax generator gradient → 0 as the discriminator → optimal.
- Non-saturating generator loss avoids the zero but gives unstable updates.
- Worked case → Example in [Wasserstein Distance](#wasserstein-distance).

*What is wrong w/ weight clipping?*

- $c$ large → slow critic training. $c$ small → vanishing gradients.
- Biases the critic toward simple functions → [WGAN-GP](#wgan-gp).

*Why train the critic more than the generator?*

- Gradient formula needs a near-optimal critic.
- Strong critic does not saturate here (unlike a JS discriminator) → train it longer.

*Why RMSProp, not Adam?*

- Paper: momentum (Adam) on the critic → unstable training.

*Other claimed benefits?*

- Critic loss tracks sample quality → usable as a training curve.
- Paper reports no mode collapse in its experiments.

*Does the critic really estimate $W_1$?*

- Contested: Stanczuk et al. (2021) find the WGAN loss approximates $W_1$ poorly. Better $W_1$ estimates do not improve samples.
- → Success may come from the critic's inductive bias, not from $W_1$ itself.
```

&nbsp;

#### WGAN-GP
- **Name**: Wasserstein GAN with Gradient Penalty {cite:p}`gulrajani2017improved`
- **What**: WGAN w/ weight clipping replaced by a penalty $(\|\nabla f\|_2-1)^2$ at points between real & fake samples.
- **Why**: Clipping → underused critic capacity + exploding or vanishing gradients.
    - Differentiable $f$: 1-Lipschitz ⇔ $\|\nabla f\|_2\le1$ everywhere → constrain the gradient directly.
- **How**:
    1. Sample $x\sim p_\text{data}$, $\tilde x\sim p_\theta$, $t\sim\mathcal{U}[0,1]$.
    2. Interpolate $\hat x=t\,x+(1-t)\,\tilde x$.
    3. Add $\lambda(\|\nabla_{\hat x}f(\hat x)\|_2-1)^2$ to the critic loss.
    - Defaults: $\lambda=10$, $n_\text{critic}=5$, Adam (LR $10^{-4}$, $\beta_1=0$, $\beta_2=0.9$).
    - ❌BatchNorm in the critic; layer norm instead.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\lambda$: Penalty weight.

Training (critic):

$$
L=\mathbb{E}_{\tilde x\sim p_\theta}[f(\tilde x)]-\mathbb{E}_{x\sim p_\text{data}}[f(x)]+\lambda\,\mathbb{E}_{\hat x}\Big[\big(\|\nabla_{\hat x}f(\hat x)\|_2-1\big)^2\Big]
$$

- $\hat x=t\,x+(1-t)\,\tilde x$, $t\sim\mathcal{U}[0,1]$.
- Differentiable optimal critic: $\|\nabla f^*\|=1$ on transport segments between optimally coupled $x\ne y$ → target norm 1, not ≤ 1.
- ❌Requirement where $x=y$: e.g., identical distributions → constant critic.
- Random (not optimal) pairs → approximation.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

## Reuses WGANCritic from WGAN above
class WGANGP(WGANCritic):
    def __init__(self, critic, lam=10.0):
        super().__init__(critic)
        self.lam = lam

    def critic_loss(self, x_real, x_fake):
        t = torch.rand(x_real.shape[0], 1)                     ## 1 mix weight per sample
        x_hat = (t * x_real + (1 - t) * x_fake).requires_grad_(True)
        ## create_graph -> the penalty itself is differentiated w.r.t. critic params
        grad, = torch.autograd.grad(self.f(x_hat).sum(), x_hat, create_graph=True)
        gp = ((grad.norm(dim=1) - 1) ** 2).mean()              ## push ||grad f|| -> 1
        return super().critic_loss(x_real, x_fake) + self.lam * gp

## Example: no clip_weights() call; Adam on f
f = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
loss = WGANGP(f).critic_loss(torch.randn(8, 2) + 3, torch.randn(8, 2))
loss.backward()
print(loss.shape)   ## torch.Size([])
```
````

```{attention} Q&A
:class: dropdown
*Why no BatchNorm in the critic?*

- Penalty is on each sample's input gradient.
- BatchNorm couples samples in a batch → per-sample gradient no longer matches the penalized objective.

*Why penalize only at interpolates?*

- Enforcing the constraint everywhere is intractable.
- Optimal critic has unit gradient along transport lines → sample along lines between real & fake points.

*Other ways to enforce Lipschitz?*

- Spectral normalization: divide each weight matrix by its largest singular value → each linear layer 1-Lipschitz.
```

&nbsp;
