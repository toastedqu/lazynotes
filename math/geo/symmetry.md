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
# Symmetry & Groups
Symmetry = transformation that leaves something unchanged. Groups formalize symmetries. Actions & representations make them move data, layers & latents.
Covers symmetry as an architecture prior (Geometric Deep Learning) & symmetry found inside trained nets. Main reference: {cite:t}`bronstein2021geometric`.
Group axioms → [Group](../linalg.md#group).

Notations:
- $G$: Group.
- $g,h\in G$: Group elements.
- $e$: Identity element of $G$.
- $|G|$: Order (#elements) of a finite group.
- $\Omega$: Domain that $G$ acts on (pixel grid, node set, $\mathbb{R}^3$, ...).
- $u\in\Omega$: Point of the domain.
- $g\cdot u$: Action of $g$ on $u$.
- $\mathcal{X}$: Input (signal) space.
- $\mathcal{Y}$: Output (feature) space.
- $\rho_\mathcal{X}(g)$: Action of $g$ on $\mathcal{X}$ (a matrix if linear).
- $\rho_\mathcal{Y}(g)$: Action of $g$ on $\mathcal{Y}$.
- $f:\mathcal{X}\to\mathcal{Y}$: Model or layer.
- $\sigma$: Pointwise nonlinearity.
- $N$: #elements of a set / #nodes of a graph.
- $S_N$: Symmetric group: all permutations of $N$ items.
- $\mathbb{Z}_n$: Cyclic group $\{0,\dots,n-1\}$ w/ addition mod $n$.
- $SO(n)$: Rotations of $\mathbb{R}^n$.
- $O(n)$: Rotations & reflections of $\mathbb{R}^n$.
- $E(n)$: Translations, rotations & reflections of $\mathbb{R}^n$.
- $GL(V)$: Invertible linear maps on vector space $V$.

Overrides: $n$ = space dimension in $SO(n),O(n),E(n)$ & group order in $\mathbb{Z}_n$, ❌#features. $N$ = set size. $\rho$ = action / representation everywhere on this page.

&nbsp;

## Groups & Actions

### Group Action
- **What**: Map $G\times\Omega\to\Omega$, $(g,u)\mapsto g\cdot u$, compatible w/ the group law.
- **Why**: A group is abstract → need it to move concrete objects (pixels, nodes, atoms, latent vectors).
    - One group, many actions: $\mathbb{Z}_4$ rotates an image by 90°, cyclically shifts a 4-list, or rotates a 2-D vector.
- **How**:
    1. Each $g$ → invertible map $u\mapsto g\cdot u$. Inverse map = action of $g^{-1}$.
    2. Composition of maps = group multiplication. $e$ → do nothing.
    3. **Orbit** = all points reachable from $u$. **Stabilizer** = elements that fix $u$.

```{note} Math
:class: dropdown
Definition (left action):

$$
e\cdot u=u,\qquad g\cdot(h\cdot u)=(gh)\cdot u\qquad\forall g,h\in G,\ u\in\Omega
$$

- $gh$ acts as "first $h$, then $g$".
- **Orbit**: $G\cdot u=\{g\cdot u:g\in G\}$. Orbits partition $\Omega$.
- **Stabilizer**: $G_u=\{g\in G:g\cdot u=u\}$. Always a subgroup of $G$.
- **Transitive**: one orbit = all of $\Omega$ → $\Omega$ is a **homogeneous space**, $\Omega\cong G/G_u$.
- **Free**: every stabilizer is $\{e\}$.
- **Orbit–stabilizer** (finite $G$):

    $$
    |G\cdot u|\cdot|G_u|=|G|
    $$

- Action on signals $x:\Omega\to\mathbb{R}^C$:

    $$
    (g\cdot x)(u)=x(g^{-1}\cdot u)
    $$

    - $C$: #channels.
    - $g^{-1}$ ← keeps it a left action: $(g\cdot(h\cdot x))(u)=x(h^{-1}g^{-1}\cdot u)=x((gh)^{-1}\cdot u)$.
    - Any action on $\Omega$ → linear action on signals → see [Group Representation](#group-representation).
```

```{tip} Derivation
:class: dropdown
*Why does orbit–stabilizer hold?*

1. Map each left coset $gG_u$ to the point $g\cdot u$.
2. Well defined & injective: $g\cdot u=g'\cdot u$ ⇔ $g^{-1}g'\cdot u=u$ ⇔ $g^{-1}g'\in G_u$ ⇔ $gG_u=g'G_u$.
3. Surjective onto $G\cdot u$ by definition of the orbit.
4. → #cosets $=|G\cdot u|$. Lagrange: #cosets $=|G|/|G_u|$.
```

```{note} Example
:class: dropdown
- **Cube rotations** ($|G|=24$) acting on faces: orbit of a face = 6 faces; stabilizer = 4 rotations about the axis through that face → $6\cdot4=24$.
- **Permutations** $S_N$ acting on $N$ items: orbit = all $N$ items; stabilizer of item 1 ≅ $S_{N-1}$ → $N\cdot(N-1)!=N!$.
- **Translations** $\mathbb{Z}^2$ acting on the pixel grid $u\mapsto u+t$: transitive & free → stabilizer $\{e\}$.
- **$E(n)$** acting on $\mathbb{R}^n$ by $u\mapsto Qu+t$, $Q\in O(n)$: transitive; stabilizer of the origin = $O(n)$ → $\mathbb{R}^n\cong E(n)/O(n)$.
```

```{dropdown} Table: Group Actions in ML
| Group | Domain $\Omega$ | Action | Architecture |
|:--|:--|:--|:--|
| Translations $\mathbb{Z}^2$ | Pixel grid | Shift | [CNN](../../cv/cnn.md#cnn) |
| $\mathbb{Z}_n$ | $n$ points on a circle | Cyclic shift | Circular conv, modular arithmetic |
| $S_N$ | Set / graph nodes | Permute indices | [Deep Sets](#deep-sets), GNN, Transformer |
| $p4$ | Pixel grid | Shift + 90° rotation | [G-CNN](#g-cnn) |
| $SO(3)$ | Sphere $S^2$ | Rotate | Spherical CNN |
| $E(n)$ | $\mathbb{R}^n$ | $u\mapsto Qu+t$ | [EGNN](#egnn), point-cloud nets |
```

```{attention} Q&A
:class: dropdown
*What is a Lie group?*

- Group that is also a smooth manifold; multiplication & inversion are smooth. E.g., $SO(3)$, $E(n)$.
- → Continuous symmetries. Infinitesimal generators form the Lie algebra (tangent space at $e$).

*Discrete vs continuous groups in practice?*

- Finite orientation groups ($C_4$, $D_4$) → enumerate all orientations → [G-CNN](#g-cnn).
    - $C_4$: 4 rotations by multiples of 90°.
    - $D_4$: $C_4$ + 4 reflections.
- $p4$, $p4m$ also contain infinitely many integer translations → handled by ordinary sliding conv.
- Continuous ($SO(3)$, $E(n)$) → cannot enumerate → use invariant scalars ([EGNN](#egnn)) or irreps (steerable / spherical-harmonic nets).

*Why do orbits matter for learning?*

- Orbit of an image under rotations = all poses of one object.
- Invariant classifier = function on the quotient $\mathcal{X}/G$ of input signals (one value per signal orbit) → the task has fewer effective inputs.

*What does this tell us about intelligence/AC?*

- Object identity = what stays the same along an orbit (pose, position, lighting) → recognition = collapsing orbits → abstraction.
- Hypothesis, not established: perception builds world-symmetry structure into its representations → [Symmetry-Based Disentanglement](#symmetry-based-disentanglement).
```

&nbsp;

### Group Representation
- **What**: Group homomorphism $\rho:G\to GL(V)$: each $g$ → invertible matrix; products preserved.
- **Why**: Linear layers act on vectors → need group elements as matrices to state & build linear equivariance.
    - Irreducible pieces (irreps) = smallest blocks → simplest basis to analyze any equivariant linear map.
- **How**:
    1. Pick a vector space $V$ (signals on $\Omega$, $\mathbb{R}^3$, a latent space).
    2. Assign each $g$ a matrix w/ $\rho(gh)=\rho(g)\rho(h)$.
    3. Change basis to split $V$ into invariant subspaces → block-diagonal $\rho$ → blocks = irreps.
    4. Cyclic / abelian groups: irrep basis = Fourier basis.

```{note} Math
:class: dropdown
Definition:

$$
\rho(gh)=\rho(g)\rho(h)\quad\Rightarrow\quad\rho(e)=I,\ \ \rho(g^{-1})=\rho(g)^{-1}
$$

- **Invariant subspace** $W\subseteq V$: $\rho(g)W\subseteq W$ for all $g$.
- **Irreducible**: only invariant subspaces are $\{0\}$ & $V$.
- **Complete reducibility** (finite or compact $G$, over $\mathbb{R}$ or $\mathbb{C}$):

    $$
    \rho(g)=Q\Big(\bigoplus_{\ell}\rho_\ell(g)\Big)Q^{-1}
    $$

    - $\rho_\ell$: Irreps (may repeat).
    - $Q$: Fixed change of basis.
- **Regular rep**: $G$ acts on $x:G\to\mathbb{R}$ by $(\rho_\text{reg}(g)x)(h)=x(g^{-1}h)$. Finite $G$ → over $\mathbb{C}$ contains every irrep, multiplicity = its dimension.
- **Schur's lemma** (over $\mathbb{C}$): $A\rho(g)=\rho(g)A\ \forall g$, $\rho$ irreducible → $A=cI$. Equivariant map between non-isomorphic irreps = 0.
- Abelian $G$ over $\mathbb{C}$ → every irrep is 1-D. For $\mathbb{Z}_n$:

    $$
    \chi_k(a)=e^{2\pi i ka/n},\qquad k=0,\dots,n-1
    $$

    - $a\in\mathbb{Z}_n$: Group element.
    - $k$: Frequency (overrides class idx).
- $\mathbb{Z}_n$ over $\mathbb{R}$: merge $k$ & $n-k$ → 2-D rotation irrep $\rho_k(a)=R(2\pi ka/n)$; plus trivial ($k=0$) & sign ($k=n/2$, $n$ even).
    - $R(\theta)$: 2-D rotation matrix by angle $\theta$.
- **Fourier transform** on $\mathbb{Z}_n$ = coordinates of a signal in the irrep basis:

    $$
    \hat{x}(k)=\sum_{a=0}^{n-1}x(a)e^{-2\pi i ka/n}
    $$
```

```{tip} Derivation
:class: dropdown
*Why does convolution diagonalize in the Fourier basis?*

1. Circular conv on $\mathbb{Z}_n$: $(x*\kappa)(a)=\sum_b x(b)\kappa(a-b)$. $\kappa$: filter. It commutes w/ every cyclic shift.
2. Let $\mathbf{v}_k(a)=e^{2\pi i ka/n}$. Substitute $c=a-b$:

    $$
    (\mathbf{v}_k*\kappa)(a)=\sum_c e^{2\pi i k(a-c)/n}\kappa(c)=\hat{\kappa}(k)\,\mathbf{v}_k(a)
    $$

3. → Every $\mathbf{v}_k$ is an eigenvector of every convolution. Eigenvalue = $\hat{\kappa}(k)$.
4. Expand $x=\frac1n\sum_k\hat{x}(k)\mathbf{v}_k$ → $\widehat{x*\kappa}(k)=\hat{x}(k)\hat{\kappa}(k)$ (convolution theorem).
5. Converse: shift $(Sx)(a)=x(a-1)$ has $n$ distinct eigenvalues $e^{-2\pi ik/n}$ (eigenvectors $\mathbf{v}_k$) → any matrix that commutes w/ $S$ is diagonal in the same basis → circulant → a convolution.
6. Non-abelian compact $G$: irreps are matrices → convolution becomes block-diagonal: $\widehat{x*\kappa}(\rho_\ell)$ = product of the two $d_\ell\times d_\ell$ Fourier matrices (order depends on convention).
    - $d_\ell$: Dimension of irrep $\rho_\ell$.
```

```{note} Example
:class: dropdown
$\mathbb{Z}_4$ acting on $\mathbb{R}^4$ by cyclic shift (regular rep):

- Shift eigenvalues: $1, i, -1, -i$.
- $(1,1,1,1)$ → trivial irrep (eigenvalue $1$).
- $(1,-1,1,-1)$ → sign irrep (eigenvalue $-1$).
- Remaining 2-D subspace → shift acts as rotation by 90° → 2-D real irrep (pair $\pm i$).
- Dimensions: $1+1+2=4$ ✓.
```

```{attention} Q&A
:class: dropdown
*Representation vs action?*

- Representation = linear action on a vector space.
- Any action on a finite $\Omega$ → permutation representation on $\mathbb{R}^{|\Omega|}$ (signals on $\Omega$).

*Why do irreps matter for equivariant nets?*

- Schur → in the irrep basis, an equivariant linear map is block-structured: 0 between different irreps, scalar within one (over $\mathbb{C}$).
- → #free params collapses. Steerable CNNs, Tensor Field Networks & e3nn build layers this way.
- $SO(3)$ irreps: dimension $2\ell+1$, realized by spherical harmonics of degree $\ell$.

*Why is a 2-D rotation irreducible over $\mathbb{R}$ but not over $\mathbb{C}$?*

- Rotation by $\theta\notin\{0,\pi\}$ has no real eigenvector → no invariant line in $\mathbb{R}^2$.
- Over $\mathbb{C}$: eigenvectors $(1,\mp i)$, eigenvalues $e^{\pm i\theta}$ → splits into two 1-D irreps.
- Over $\mathbb{R}$, Schur weakens: maps commuting w/ all 2-D rotations = $\{aI+bR(\pi/2)\}$ ≅ $\mathbb{C}$ → 2 free params.

*How does this relate to RoPE?*

- [RoPE](../../nlp/pos.md#rope) = representation of position shifts on $\mathbb{R}^d$: block-diagonal 2-D rotations at several frequencies.
- $\langle R_m\mathbf{q},R_{m'}\mathbf{k}\rangle=\langle\mathbf{q},R_{m'-m}\mathbf{k}\rangle$ ← $R_m^\top R_{m'}=R_{m'-m}$ → score depends only on the offset.
    - $R_m$: Rotation for position $m$.

*How does this relate to features in LMs?*

- Cyclic concepts (days, months) appear as circles in LM activations → 2-D irreps of $\mathbb{Z}_n$ in disguise → [Multi-Dimensional Features](repr.md#multi-dimensional-features).
- HRR binding = circular convolution → diagonal in Fourier → [HRR](highdim.md#hrr).
```

&nbsp;

### Invariance & Equivariance
- **What**: **Equivariant**: transform the input → output transforms correspondingly. **Invariant**: output does not change.
- **Why**: Known task symmetry → build it in → ❌spend data & params to learn it.
    - Classifier: label invariant to pose. Segmenter, detector, force predictor: output moves w/ the input.
- **How**:
    1. Stack equivariant layers → whole stack is equivariant.
    2. End w/ an invariant readout: sum / mean / max pooling over the group or domain.
    3. Alternatives: data augmentation (approximate), canonicalization (map input to a standard pose), invariant input features (distances, angles).

```{note} Math
:class: dropdown
Definitions:

$$
\text{Invariant: }f(\rho_\mathcal{X}(g)x)=f(x),\qquad\text{Equivariant: }f(\rho_\mathcal{X}(g)x)=\rho_\mathcal{Y}(g)f(x)\qquad\forall g,x
$$

- Invariance = equivariance w/ trivial $\rho_\mathcal{Y}(g)=I$.
- **Composition**: $f_1:\rho_\mathcal{X}\to\rho_\mathcal{Z}$ & $f_2:\rho_\mathcal{Z}\to\rho_\mathcal{Y}$ equivariant → $f_2\circ f_1$ equivariant.
    - $\mathcal{Z}$, $\rho_\mathcal{Z}$: Intermediate space & its action.
- Pointwise $\sigma$ commutes w/ permutation representations (translations, regular reps, $S_N$) → stays equivariant.
    - ❌ for general reps: rotate a 3-D vector, then apply ReLU per coordinate ≠ ReLU then rotate.
- **Linear** equivariant maps $W\rho_\mathcal{X}(g)=\rho_\mathcal{Y}(g)W$ form a linear subspace.
- **Reynolds operator / group averaging** (finite $G$):

    $$
    \bar{f}(x)=\frac{1}{|G|}\sum_{g\in G}f(\rho_\mathcal{X}(g)x),\qquad
    \tilde{f}(x)=\frac{1}{|G|}\sum_{g\in G}\rho_\mathcal{Y}(g)^{-1}f(\rho_\mathcal{X}(g)x)
    $$

    - $\bar{f}$: Invariant version of any $f$.
    - $\tilde{f}$: Equivariant version of any $f$.
    - $f$ already invariant / equivariant → unchanged (projection).
    - Compact continuous $G$ → normalized Haar integral replaces the sum.
```

```{tip} Derivation
:class: dropdown
*Why is $\tilde{f}$ equivariant?*

1. $\tilde{f}(\rho_\mathcal{X}(h)x)=\frac1{|G|}\sum_g\rho_\mathcal{Y}(g)^{-1}f(\rho_\mathcal{X}(gh)x)$.
2. Substitute $g'=gh$ → $g=g'h^{-1}$ → $\rho_\mathcal{Y}(g)^{-1}=\rho_\mathcal{Y}(h)\rho_\mathcal{Y}(g')^{-1}$.
3. $g'$ runs over all of $G$ exactly once → sum $=\rho_\mathcal{Y}(h)\tilde{f}(x)$.
4. Set $\rho_\mathcal{Y}=I$ → $\bar{f}$ is invariant.

*Why does composition preserve equivariance?*

1. $f_2(f_1(\rho_\mathcal{X}(g)x))=f_2(\rho_\mathcal{Z}(g)f_1(x))=\rho_\mathcal{Y}(g)f_2(f_1(x))$.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn.functional as F

class EquivarianceTest:
    def __init__(self, f, act_in, act_out=None):
        self.f = f
        self.act_in = act_in    ## (g, x) -> rho_X(g) x
        self.act_out = act_out  ## (g, y) -> rho_Y(g) y; None -> test invariance

    def __call__(self, x, gs, atol=1e-5):
        fx = self.f(x)
        for g in gs:
            lhs = self.f(self.act_in(g, x))                              ## f(rho_X(g) x)
            rhs = fx if self.act_out is None else self.act_out(g, fx)    ## f(x) or rho_Y(g) f(x)
            if not torch.allclose(lhs, rhs, atol=atol):
                return False
        return True

## Example: Z_n acting on a 1-D signal by cyclic shift
n = 8
w = torch.randn(1, 1, 3)
conv = lambda x: F.conv1d(F.pad(x, (1, 1), mode="circular"), w)  ## circular conv: (1, 1, n) -> (1, 1, n)
shift = lambda g, x: torch.roll(x, shifts=g, dims=-1)
x = torch.randn(1, 1, n)
print(EquivarianceTest(conv, shift, shift)(x, range(n)))                 ## True: equivariant
print(EquivarianceTest(lambda x: conv(x).mean(-1), shift)(x, range(n)))  ## True: pooling -> invariant
print(EquivarianceTest(lambda x: conv(x)[..., 0], shift)(x, range(n)))   ## False: reading one position
```
````

```{dropdown} Table: Ways to Get a Symmetry
| Method | Guarantee | Cost | Fails when |
|:--|:--|:--|:--|
| Data augmentation | Approximate, near training data only | More compute & capacity | Group large/continuous; test pose rare |
| Built-in equivariance | Exact for every input | Custom layers; ×$\lvert H\rvert$ activations for [G-CNN](#g-cnn) | Symmetry wrong or only approximate |
| Canonicalization | Exact if canonical pose is stable | Need a pose estimator | Pose ambiguous (symmetric objects) |
| Invariant features | Exact | Cheap | Features discard needed info |

- $H$: Group of rotations / flips per position.
```

```{attention} Q&A
:class: dropdown
*Augmentation vs built-in equivariance?*

- Augmentation: any architecture, easy. ❌Guarantee off the training distribution. Capacity spent on learning the symmetry.
- Built-in: exact for all inputs, better sample efficiency. Needs a known, exact symmetry & custom layers.
- Wrong symmetry hurts: full rotation invariance confuses 6 & 9; reflection invariance confuses b & d.

*Why not make every layer invariant?*

- Premature invariance discards relative pose: eyes above mouth vs below.
- G-CNN on rotated MNIST: pool over rotations after every layer → 3.21% error; keep equivariance, pool at the end → 2.28% {cite:p}`cohen2016group`.

*Does a real CNN keep exact equivariance?*

- Stride → equivariant only to shifts by multiples of the stride.
- Zero padding → breaks equivariance at borders → [CNN](../../cv/cnn.md#cnn) Q&A.

*Is invariance always desired?*

- ❌. Many symmetries are approximate: gravity fixes "up" in natural images.
- → Soft / partial equivariance, or learn which symmetry to use.

*What does this tell us about intelligence/AC?*

- Concepts = invariants: "cup" is constant across pose, lighting, position.
- Primate ventral stream: later areas (IT) respond more tolerantly to position & size changes than early areas (V1).
- Physics: Noether's theorem → each continuous symmetry of the action ↔ a conserved quantity (time shift ↔ energy, space shift ↔ momentum, rotation ↔ angular momentum).
- Hypothesis: general intelligence = discover which transformations leave outcomes unchanged. Suggestive analogy, not evidence that brains compute group averages.
```

&nbsp;

## Equivariant Architectures

### Geometric Deep Learning
- **What**: Derive architectures from the symmetry group of the input domain. {cite:p}`bronstein2021geometric`
- **Why**: Learning in high dimensions is hopeless w/o priors → symmetry & scale separation are priors most physical data has.
    - One recipe explains CNN, GNN, Deep Sets, Transformer, spherical & mesh CNNs.
- **How**: Blueprint:
    1. Fix domain $\Omega$, signals $x:\Omega\to\mathbb{R}^C$, symmetry group $G$.
    2. Repeat: local $G$-equivariant linear layer → pointwise $\sigma$ → local pooling (coarsen $\Omega$).
    3. End w/ a $G$-invariant global pooling.

```{note} Math
:class: dropdown
Notations:
- $B_j$: $G$-equivariant linear layer: $B(g\cdot x)=g\cdot B(x)$.
- $P_j$: Local pooling / coarsening: signals on $\Omega$ → signals on $\Omega'\subseteq\Omega$.
- $A$: $G$-invariant global pooling: $A(g\cdot x)=A(x)$.
- $J$: #blocks.

Model:

$$
f=A\circ\sigma_J\circ B_J\circ P_{J-1}\circ\dots\circ P_1\circ\sigma_1\circ B_1
$$

- Three principles:
    - **Symmetry**: $B$ equivariant, $A$ invariant.
    - **Geometric stability**: a small deformation (not an exact symmetry) → a small output change.
    - **Scale separation**: local layers + coarsening → capture structure at growing scales.
- Image classification CNN = instance: $\Omega$ = grid, $G$ = translations, $B$ = conv, $P$ = strided pooling, $A$ = global average pooling.
```

```{dropdown} Table: The 5G
| "G" | Domain $\Omega$ | Symmetry $G$ | Architecture |
|:--|:--|:--|:--|
| Grids | 1-D / 2-D grid | Translations | [CNN](../../cv/cnn.md#cnn) |
| Groups | Homogeneous space (sphere, group itself) | $SO(3)$, $p4$, ... | Spherical CNN, [G-CNN](#g-cnn) |
| Graphs | Graph | $S_N$ | GNN |
| Graphs | Set (no edges) | $S_N$ | [Deep Sets](#deep-sets) |
| Graphs | Complete graph | $S_N$ | Transformer |
| Graphs + geometry | Points in $\mathbb{R}^n$ | $E(n)\times S_N$ | [EGNN](#egnn) |
| Geodesics | Manifold / mesh | Isometries $\mathrm{Iso}(\Omega)$ | Intrinsic (geodesic) mesh CNN |
| Gauges | Manifold w/ local tangent frames | Gauge group $SO(2)$ | Gauge-equivariant mesh CNN |

- Bronstein et al. also list LSTM: 1-D grid, symmetry = time warping (via gating).
- Transformer: position encodings deliberately break $S_N$ → [Permutation Equivariance](../../nlp/pos.md#permutation-equivariance).
```

```{attention} Q&A
:class: dropdown
*Why is the curse of dimensionality the motivation?*

- Learning a generic Lipschitz function on $\mathbb{R}^d$ to accuracy $\epsilon$ needs #samples growing like $\epsilon^{-d}$ → [Curse of Dimensionality](../../ml/misc.md#curse-of-dimensionality).
- Symmetry & scale separation restrict the function class → can reduce sample complexity.
- Removing the dependence on dimension needs extra assumptions.

*What is a gauge?*

- Gauge = choice of local tangent frame at each point.
- Topology can prevent a global smooth frame (e.g., $S^2$) → must work w/ local frames.
- Frame choice is arbitrary → output must not depend on it → gauge equivariance. Frame change on a 2-D surface = $SO(2)$ rotation.
- Comparing features at different points → [Parallel Transport](manifold.md#parallel-transport).

*Is a Transformer a GDL model?*

- ✅ Attention w/o positions = $S_N$-equivariant message passing on a complete graph w/ learned edge weights.

*Limits of the program?*

- Real symmetries are often approximate.
- Scale can replace priors: ViT trails ResNets on mid-size data (ImageNet) but wins after pretraining on 14M–300M images (Dosovitskiy et al., ICLR 2021).
- → Priors matter most when data is scarce or the symmetry is exact (physics, molecules).

*How does this relate to the Erlangen Program?*

- Klein (1872): a geometry = the invariants of a transformation group.
- GDL: an architecture = the maps equivariant to the domain's group. Same move, applied to learning.
```

&nbsp;

### G-CNN
- **Name**: Group Equivariant Convolutional Network {cite:p}`cohen2016group`
- **What**: CNN whose feature maps are functions on a group $G\supset$ translations, w/ convolution over $G$.
- **Why**: Plain conv is equivariant to translations only → rotated input ≠ rotated features.
    - Plain CNN must learn rotated copies of each filter from data.
    - G-CNN shares each filter across rotations / flips → one learned filter serves all orientations.
- **How**: For $p4$ (translations + 90° rotations):
    1. **Lifting layer**: correlate the image w/ the 4 rotated copies of each filter → feature map over (rotation, position).
    2. **Group conv**: filters live on $p4$. Rotate a filter = rotate each spatial slice + cyclically shift its rotation axis.
    3. Pointwise $\sigma$ & spatial pooling keep equivariance.
    4. Pool over rotations at the end → rotation invariance.
    - $p4m$ adds reflections → 8 rotation-flips per position.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $x_c$: Input channel $c$: function on $\mathbb{Z}^2$ (layer 1) or on $G$ (later layers).
- Params:
    - $\kappa_c$: Filter for input channel $c$.
- Misc:
    - $p4$ element $g=(t,r)$: $g\cdot u=R_r u+t$; $R_r$ = rotation by $r\cdot 90°$, $r\in\mathbb{Z}_4$; $t\in\mathbb{Z}^2$.
    - $H$: Stabilizer of the origin ($|H|=4$ for $p4$, $8$ for $p4m$).

Forward (correlation, as in CNN libraries):

$$\begin{align*}
\text{Lifting: }[x\star\kappa](g)&=\sum_{u\in\mathbb{Z}^2}\sum_c x_c(u)\,\kappa_c(g^{-1}\cdot u)\\
\text{Group conv: }[x\star\kappa](g)&=\sum_{h\in G}\sum_c x_c(h)\,\kappa_c(g^{-1}h)
\end{align*}$$

- Output = function on $G$ → one channel = $|H|$ planar maps.
- Equivariance: $(a\cdot x)\star\kappa=a\cdot(x\star\kappa)$ for all $a\in G$.
- Implementation: transform each filter by every $r\in H$ → stack → one planar conv. Input orientations become extra input channels.
```

```{tip} Derivation
:class: dropdown
*Why is group correlation equivariant?*

1. $[(a\cdot x)\star\kappa](g)=\sum_h x(a^{-1}h)\,\kappa(g^{-1}h)$.
2. Substitute $h'=a^{-1}h$: $=\sum_{h'}x(h')\,\kappa(g^{-1}ah')$.
3. $g^{-1}a=(a^{-1}g)^{-1}$ → $=[x\star\kappa](a^{-1}g)=(a\cdot[x\star\kappa])(g)$.
4. Lifting layer: same steps w/ $h\in\mathbb{Z}^2$.

*What does a rotation $r$ do to a $p4$ filter?*

1. $g=(0,r)$, $h=(v,s)$ → $g^{-1}h=(R_{-r}v,\,s-r)$.
2. → Filter for output rotation $r$: $\kappa(R_{-r}v,\,s-r)$ = rotate every spatial slice by $r$ + cyclically shift the rotation index by $r$.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def rot(x, r):  ## rotate the last two (spatial) dims by r * 90 degrees
    return torch.rot90(x, r, dims=(-2, -1))

class LiftC4(nn.Module):
    ## Z^2 -> p4: correlate with the 4 rotated copies of each filter
    def __init__(self, c_in, c_out, k):
        super().__init__()
        self.w = nn.Parameter(torch.randn(c_out, c_in, k, k) / k)
        self.pad = k // 2

    def forward(self, x):  ## (B, C_in, H, W) -> (B, C_out, 4, H, W)
        return torch.stack([F.conv2d(x, rot(self.w, r), padding=self.pad) for r in range(4)], dim=2)

class GroupConvC4(nn.Module):
    ## p4 -> p4: filters are functions on (rotation, position)
    def __init__(self, c_in, c_out, k):
        super().__init__()
        self.w = nn.Parameter(torch.randn(c_out, c_in, 4, k, k) / k)
        self.pad = k // 2

    def forward(self, x):  ## (B, C_in, 4, H, W) -> (B, C_out, 4, H, W)
        B, C, _, H, W = x.shape
        outs = []
        for r in range(4):
            ## kappa(R_{-r} v, s - r): shift the rotation axis by r, rotate each slice by r
            w_r = rot(torch.roll(self.w, r, dims=2), r)
            outs.append(F.conv2d(x.reshape(B, C * 4, H, W), w_r.reshape(-1, C * 4, *w_r.shape[-2:]), padding=self.pad))
        return torch.stack(outs, dim=2)

## Example: rotate input by 90 deg -> output rotates spatially AND shifts cyclically along the rotation axis
lift, gconv = LiftC4(1, 3, 3), GroupConvC4(3, 2, 3)
net = lambda x: gconv(torch.relu(lift(x)))
x = torch.randn(1, 1, 9, 9)
y, y_rot = net(x), net(rot(x, 1))
print(torch.allclose(y_rot, rot(torch.roll(y, 1, dims=2), 1), atol=1e-4))          ## True: equivariant
print(torch.allclose(y.amax(dim=(2, 3, 4)), y_rot.amax(dim=(2, 3, 4)), atol=1e-4))  ## True: pool over group -> invariant
```
````

```{attention} Q&A
:class: dropdown
*Why only 90° rotations?*

- 90° rotations map the square pixel grid onto itself → exact, interpolation-free.
- Other angles need interpolation → only approximate equivariance → use steerable filters or continuous-group methods.

*Params vs compute?*

- Lifting filter: same #params as a planar filter. Group-conv filter: $|H|\times$ more (one $k\times k$ slice per input orientation).
    - $k$: Kernel size (overrides class idx).
- At equal #feature channels:
    - Activations ×$|H|$ ← each filter outputs $|H|$ maps.
    - Lifting compute ×$|H|$.
    - Dense group-conv compute ×$|H|^2$ ← $|H|$ output orientations × $|H|$ input orientations.
- Cohen & Welling divide #channels by $\sqrt{|H|}$ → #params ≈ planar baseline.

*Where does the gain come from?*

- Weight sharing across rotations → filters do not need to be relearned per orientation → better sample efficiency.
- Rotated MNIST: P4CNN 2.28% error vs 3.98% prior SOTA. CIFAR-10: $p4m$ improves over planar baselines.

*Why are feature maps functions on the group, not on the plane?*

- Rotating a plane feature map also changes which filter orientation fired → need the extra rotation axis to record it.
- Pool over the rotation axis too early → [Invariance & Equivariance](#invariance-equivariance) Q&A.
```

&nbsp;

#### Equivariance Implies Convolution
- **What**: Compact $G$ acting on homogeneous spaces → a linear map between feature spaces on them is $G$-equivariant iff it is a generalized group convolution. {cite:p}`kondor2018generalization`
- **Why**: Convolution is not one design choice among many → under these conditions it is the only linear equivariant layer.
- **How**:
    1. Each layer's domain = homogeneous space $G/H_\ell$. Activations = functions on it. $G$ acts by $(g\cdot x)(u)=x(g^{-1}\cdot u)$.
    2. "If": convolution commutes w/ this action (as in [G-CNN](#g-cnn)).
    3. Convolutions + pointwise $\sigma$ → equivariant network.
    4. "Only if": Fourier transform over $G$ + Schur's lemma → equivariant linear map = fixed matrix per irrep → inverse Fourier → convolution w/ some filter.

```{note} Math
:class: dropdown
Notations:
- $H_\ell\le G$: Subgroup; layer-$\ell$ domain $\Omega_\ell=G/H_\ell$.
- $x_\ell:\Omega_\ell\to V_\ell$: Layer-$\ell$ activations.
- $V_\ell$: Channel space of layer $\ell$.
- $\phi_\ell$: Linear map of layer $\ell$.
- $\chi_\ell$: Filter of layer $\ell$.

Setup:

$$
x_\ell=\sigma_\ell\big(\phi_\ell(x_{\ell-1})\big)
$$

- $G$-equivariant layer: $\phi_\ell(g\cdot x)=g\cdot\phi_\ell(x)$ for **every** input $x$ on $\Omega_{\ell-1}$, not only for activations the network reaches.

Theorem:
- $G$ compact, $\Omega_\ell=G/H_\ell$ → linear $\phi_\ell$ is $G$-equivariant ⇔ $\phi_\ell(x)=x*\chi_\ell$, w/ filter $\chi_\ell$ on the double-coset space $H_{\ell-1}\backslash G/H_\ell$.
- → Every layer equivariant ⇔ every $\phi_\ell$ is a convolution. Such layers + pointwise $\sigma$ → equivariant network.
- ❌ "Network equivariant on its reachable activations ⇒ every $\phi_\ell$ is a convolution".
    - Counterexample: $\phi_1=0$ → $x_1$ = one fixed constant for every input → later $\phi_\ell$ only need to map that constant to an invariant output → need not be convolutions.
- Case $\Omega=G$ (finite $G$):

    $$
    (x*\chi)(u)=\sum_{v\in G}x(uv^{-1})\,\chi(v)
    $$

- Continuous compact $G$ → Haar integral replaces the sum.
- Conditions: compact $G$; transitive actions (homogeneous spaces); $G$ acts by moving points (not arbitrary reps on channels); pointwise $\sigma$; equivariance required of each linear map on all inputs.
```

```{note} Example
:class: dropdown
- $\mathbb{Z}_n$ acting on itself: equivariant linear maps = circulant matrices = circular convolutions → [Group Representation](#group-representation) Derivation.
- $S_N$ acting on $N$ points: $H=S_{N-1}$ (fix one point). Double cosets $S_{N-1}\backslash S_N/S_{N-1}$: 2 classes ("same point" / "different point").
    - → Filter has 2 values → $\Theta=\lambda I+\gamma\mathbf{1}\mathbf{1}^\top$ = the [Deep Sets](#deep-sets) equivariant layer.
```

```{attention} Q&A
:class: dropdown
*Does it say every equivariant network is a CNN?*

- ❌. Only within its assumptions: linear map + pointwise $\sigma$, homogeneous-space domains, point-moving actions.
- Outside: norm / gated nonlinearities, non-transitive domains (general graphs), attention, [EGNN](#egnn) (nonlinear in coordinates).
- Cohen, Geiger & Weiler (NeurIPS 2019) extend the result to general field types on homogeneous spaces.

*Why compact?*

- Finite, normalizable Haar measure → averaging & Fourier analysis (Peter–Weyl) work → complete reducibility & Schur apply.
- Every finite group is compact.

*How does it relate to Fourier?*

- Proof core = the convolution theorem generalized: equivariant maps act on each irrep block independently.
```

&nbsp;

### Deep Sets
- **What**: Permutation-invariant set model $f(X)=\psi\big(\sum_{x\in X}\phi(x)\big)$. {cite:p}`zaheer2017deep`
- **Why**: Sets (point clouds, bags of instances, particles) have no order → model must give the same answer for every ordering.
- **How**:
    1. Encode each element independently w/ a shared $\phi$.
    2. Sum-pool → order-free summary.
    3. Decode the summary w/ $\psi$.
    - Equivariant variant: per-element output; each element sees itself + the set sum.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $X=\{x_1,\dots,x_N\}$: Input set. Stacked as $X\in\mathbb{R}^{N\times d}$ in the equivariant layer.
    - $d$: Element dimension.
- Params:
    - $\phi$: Element encoder $\mathbb{R}^d\to\mathbb{R}^{d_z}$.
    - $\psi$: Set decoder. Paper's $\rho$; renamed ← $\rho$ = representation on this page.
    - $\lambda,\gamma$: Scalars (1-D elements).
    - $\Lambda,\Gamma\in\mathbb{R}^{d\times d'}$: Matrix version.
- Hyperparams:
    - $d_z$: Latent dimension.

Model (invariant):

$$
f(X)=\psi\Big(\sum_{i=1}^N\phi(x_i)\Big)
$$

Model (equivariant layer):

$$
f(X)=\sigma(\Theta X),\quad\Theta=\lambda I+\gamma\mathbf{1}\mathbf{1}^\top
\qquad\Big/\qquad
f(X)=\sigma\big(X\Lambda+\mathbf{1}\mathbf{1}^\top X\Gamma\big)
$$

- $\mathbf{1}\in\mathbb{R}^N$: All-ones vector.
- Injective pointwise $\sigma$ → $\sigma(\Theta X)$ is $S_N$-equivariant ⇔ $\Theta=\lambda I+\gamma\mathbf{1}\mathbf{1}^\top$ (paper's lemma).
- The form is sufficient for any pointwise $\sigma$.
- Sum → max pooling also works (paper's variant).

Representation theorems:
- **Countable** element universe: $f$ invariant ⇔ $f=\psi(\sum\phi)$ for some $\phi,\psi$.
    - Proof uses $\phi(x)=4^{-c(x)}$, $c$ = enumeration of the universe → 1-D latent.
    - No continuity guarantee for $\phi,\psi$ → does not establish universality w/ continuous NNs.
- **Uncountable**, fixed set size $N$, elements in $[0,1]$: continuous invariant $f$ ⇔ continuous $\phi:\mathbb{R}\to\mathbb{R}^{N+1}$ (power sums $x^0..x^N$) & continuous $\psi$.
- Wagstaff et al. (ICML 2019): universal continuous sum-decomposition for sets of $\le N$ real scalars needs latent dim $\ge N$; $N$ suffices.
```

```{tip} Derivation
:class: dropdown
*Why must $\Theta=\lambda I+\gamma\mathbf{1}\mathbf{1}^\top$?*

1. Equivariance: $\sigma(\Theta PX)=P\sigma(\Theta X)=\sigma(P\Theta X)$ for every permutation matrix $P$. ($\sigma$ pointwise → commutes w/ $P$.)
2. $\sigma$ injective (e.g., sigmoid) or identity → $\Theta P=P\Theta$ → $P\Theta P^\top=\Theta$ → $\Theta_{\pi(i)\pi(j)}=\Theta_{ij}$ for every permutation $\pi$.
3. Permutations map any diagonal entry to any diagonal entry & any off-diagonal entry to any off-diagonal entry.
4. → 2 free values: diagonal $a$, off-diagonal $b$ → $\Theta=(a-b)I+b\mathbf{1}\mathbf{1}^\top$.
5. Same answer as the double-coset count in [Equivariance Implies Convolution](#equivariance-implies-convolution).
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class DeepSets(nn.Module):
    def __init__(self, d_in, d_z, d_out):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(d_in, d_z), nn.ReLU(), nn.Linear(d_z, d_z))   ## per-element encoder
        self.lam = nn.Linear(d_z, d_z)                                                   ## Lambda: own features
        self.gam = nn.Linear(d_z, d_z, bias=False)                                       ## Gamma: set-sum features
        self.psi = nn.Sequential(nn.Linear(d_z, d_z), nn.ReLU(), nn.Linear(d_z, d_out))  ## set decoder

    def equivariant(self, Z):  ## (B, N, d_z) -> (B, N, d_z)
        ## lambda*I + gamma*11^T: each element sees itself + the order-free set sum
        return torch.relu(self.lam(Z) + self.gam(Z.sum(dim=1, keepdim=True)))

    def forward(self, X):  ## (B, N, d_in) -> (B, d_out)
        Z = self.equivariant(self.phi(X))
        return self.psi(Z.sum(dim=1))  ## sum-pool -> permutation invariant

## Example
model = DeepSets(3, 16, 2)
X, perm = torch.randn(1, 5, 3), torch.randperm(5)
print(torch.allclose(model(X), model(X[:, perm]), atol=1e-5))  ## True: invariant
Z, Z_perm = model.equivariant(model.phi(X)), model.equivariant(model.phi(X[:, perm]))
print(torch.allclose(Z[:, perm], Z_perm, atol=1e-5))  ## True: equivariant
```
````

```{attention} Q&A
:class: dropdown
*Sum vs mean vs max pooling?*

- Sum: counts multiplicities & set size. Scale grows w/ $N$.
    - Raw sum of elements collides: $\{1,3\}$ & $\{2,2\}$ → same sum.
- Mean: loses set size (keeps proportions). Stable across $N$.
- Max: robust, but loses counts.
- Sum of **encoded** elements can be injective: suitable encoder $\phi$, bounded-size multisets over a countable universe (GIN result for GNN aggregation).

*Does the universality theorem justify small latent sizes?*

- ❌. Countable-case construction gives no continuity guarantee → does not establish universality w/ continuous NNs.
- Width $\ge N$ needed for universality over set functions of scalar elements (Wagstaff et al.).
- Single tasks can need less.

*What can Deep Sets not do well?*

- Elements interact only through one pooled vector → pairwise interactions are hard.
- Fix: self-attention w/o positions = $S_N$-equivariant & models interactions (Set Transformer) → [Permutation Equivariance](../../nlp/pos.md#permutation-equivariance).

*How does it relate to GNNs?*

- Deep Sets = GNN on a graph w/o edges + a global sum node.
- GNNs restrict mixing to graph neighbors.
```

&nbsp;

### EGNN
- **Name**: E(n) Equivariant Graph Neural Network {cite:p}`satorras2021en`
- **What**: Message-passing GNN that updates node features (invariant) & coordinates (equivariant) from pairwise squared distances.
- **Why**: 3-D point data (molecules, particles): physics does not change under rotation, translation, reflection & relabeling → need $E(n)\times S_N$ equivariance.
    - Earlier equivariant nets (Tensor Field Networks, SE(3)-Transformer) use spherical harmonics → expensive & complex.
- **How**:
    1. Message $j\to i$: MLP of both node features, their squared distance & edge attribute → invariant.
    2. Coordinate update: move $\mathbf{x}_i$ along relative vectors $\mathbf{x}_i-\mathbf{x}_j$, each scaled by a learned scalar → equivariant.
    3. Feature update: sum messages → MLP → invariant.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{h}_i^l$: Node-$i$ features at layer $l$ ($E(n)$-invariant).
    - $\mathbf{x}_i^l\in\mathbb{R}^n$: Node-$i$ coordinates at layer $l$ ($E(n)$-equivariant). Overrides input $x$.
    - $a_{ij}$: Edge attribute.
- Params:
    - $\phi_e,\phi_h$: MLPs.
    - $\phi_x$: MLP w/ scalar output.
- Misc:
    - $\mathbf{m}_{ij}$: Message from $j$ to $i$.
    - $\mathbf{m}_i$: Aggregated message.
    - $C=1/(N-1)$: Normalizer.

Forward:

$$\begin{align*}
\mathbf{m}_{ij}&=\phi_e\big(\mathbf{h}_i^l,\mathbf{h}_j^l,\|\mathbf{x}_i^l-\mathbf{x}_j^l\|^2,a_{ij}\big)\\
\mathbf{x}_i^{l+1}&=\mathbf{x}_i^l+C\sum_{j\ne i}(\mathbf{x}_i^l-\mathbf{x}_j^l)\,\phi_x(\mathbf{m}_{ij})\\
\mathbf{m}_i&=\sum_{j\ne i}\mathbf{m}_{ij}\\
\mathbf{h}_i^{l+1}&=\phi_h(\mathbf{h}_i^l,\mathbf{m}_i)
\end{align*}$$

- Sums over all $j\ne i$ (fully connected). Paper also restricts $\mathbf{m}_i$ to neighbors $\mathcal{N}(i)$ & offers a velocity variant.
- Guarantee for every $Q\in O(n)$, $\mathbf{t}\in\mathbb{R}^n$:

    $$
    \big(\mathbf{h}^l,\ Q\mathbf{x}^l+\mathbf{t}\big)\mapsto\big(\mathbf{h}^{l+1},\ Q\mathbf{x}^{l+1}+\mathbf{t}\big)
    $$
```

```{tip} Derivation
:class: dropdown
*Why is the layer $E(n)$-equivariant?*

1. $\|(Q\mathbf{x}_i+\mathbf{t})-(Q\mathbf{x}_j+\mathbf{t})\|^2=\|Q(\mathbf{x}_i-\mathbf{x}_j)\|^2=\|\mathbf{x}_i-\mathbf{x}_j\|^2$ ← $Q^\top Q=I$.
2. → $\mathbf{m}_{ij}$, $\mathbf{m}_i$, $\mathbf{h}_i^{l+1}$ are invariant.
3. Coordinates: $Q\mathbf{x}_i+\mathbf{t}+C\sum_j Q(\mathbf{x}_i-\mathbf{x}_j)\phi_x(\mathbf{m}_{ij})=Q\mathbf{x}_i^{l+1}+\mathbf{t}$ ← $\mathbf{t}$ cancels in differences; $Q$ is linear.
4. Shared MLPs + sums over $j$ → relabeling nodes permutes outputs the same way.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class EGNNLayer(nn.Module):
    def __init__(self, d_h, d_m=32):
        super().__init__()
        self.phi_e = nn.Sequential(nn.Linear(2 * d_h + 1, d_m), nn.SiLU(), nn.Linear(d_m, d_m), nn.SiLU())
        self.phi_x = nn.Sequential(nn.Linear(d_m, d_m), nn.SiLU(), nn.Linear(d_m, 1))  ## one scalar per edge
        self.phi_h = nn.Sequential(nn.Linear(d_h + d_m, d_m), nn.SiLU(), nn.Linear(d_m, d_h))

    def forward(self, h, x):  ## h: (N, d_h), x: (N, n); edge attributes a_ij omitted
        N = h.shape[0]
        diff = x[:, None, :] - x[None, :, :]       ## (N, N, n): x_i - x_j, rotates with x, translation cancels
        dist2 = (diff ** 2).sum(-1, keepdim=True)  ## (N, N, 1): E(n)-invariant
        m = self.phi_e(torch.cat([h[:, None].expand(N, N, -1), h[None, :].expand(N, N, -1), dist2], -1))
        mask = 1 - torch.eye(N)[..., None]         ## exclude j == i
        m = m * mask
        x = x + (diff * self.phi_x(m) * mask).sum(1) / (N - 1)  ## move only along relative vectors
        h = self.phi_h(torch.cat([h, m.sum(1)], -1))
        return h, x

## Example: random orthogonal Q + translation t
layer = EGNNLayer(d_h=4)
h, x = torch.randn(5, 4), torch.randn(5, 3)
Q, _ = torch.linalg.qr(torch.randn(3, 3))
t = torch.randn(3)
h1, x1 = layer(h, x)
h2, x2 = layer(h, x @ Q.T + t)
print(torch.allclose(h1, h2, atol=1e-4), torch.allclose(x1 @ Q.T + t, x2, atol=1e-4))  ## True True
```
````

```{attention} Q&A
:class: dropdown
*Why squared distances & relative vectors only?*

- Distance = the basic $E(n)$-invariant of a point pair → safe input to any MLP.
- Weighted sum of relative vectors = simplest equivariant vector → safe coordinate update.

*What does dropping spherical harmonics cost?*

- EGNN carries only scalars (type 0) & coordinate vectors (type 1) → cheap, works in any $n$.
- ❌Higher-order geometric features inside a layer; angles & orientations only emerge through stacking.

*Can EGNN tell mirror-image molecules apart?*

- ❌ for invariant outputs ← $E(n)$ includes reflections → enantiomers get the same prediction.
- Chirality-sensitive tasks need $SE(3)$-only equivariance (e.g., add cross products / signed volumes).

*Where is it used?*

- Paper: N-body dynamics, graph autoencoders, QM9 molecular properties.
- Denoiser in $E(3)$-equivariant diffusion for 3-D molecule generation.
```

&nbsp;

## Symmetry in Learned Representations

### Symmetry-Based Disentanglement
- **What**: Representation $\mathcal{Z}=\mathcal{Z}_1\oplus\dots\oplus\mathcal{Z}_K$ where each subspace changes only under its own subgroup of the world's symmetry group $G=G_1\times\dots\times G_K$. {cite:p}`higgins2018towards`
- **Why**: "Disentangled" had no agreed definition → models & metrics disagreed.
    - Ground it in world symmetries, not in hand-picked "factors of variation".
- **How**:
    1. Find transformations of the world that change one aspect & keep the rest (move in $x$, move in $y$, change color).
    2. Require the map world → latent to be equivariant: $f(g\cdot w)=g\cdot f(w)$.
    3. Require each latent subspace to be moved only by its own subgroup.
    - Linear version: the action on $\mathcal{Z}$ is a representation → block-diagonal.

```{note} Math
:class: dropdown
Notations:
- $\mathcal{W}$: Set of world states.
- $w\in\mathcal{W}$: World state.
- $\mathcal{Z}$: Latent space.
- $f:\mathcal{W}\to\mathcal{Z}$: Observation process followed by the encoder. Overrides model $f$.
- $K$: #subgroups.

Definition — $f$ is disentangled w.r.t. $G=G_1\times\dots\times G_K$ if:
1. $G$ acts on $\mathcal{Z}$.
2. $f$ is equivariant:

    $$
    f(g\cdot w)=g\cdot f(w)\qquad\forall g\in G,\ w\in\mathcal{W}
    $$

3. $\mathcal{Z}=\mathcal{Z}_1\oplus\dots\oplus\mathcal{Z}_K$ (or $\times$), each $\mathcal{Z}_k$ fixed by every $G_j$, $j\ne k$, & affected only by $G_k$.

- **Linear disentangled**: the action is a representation $\rho(g_1,\dots,g_K)=\rho_1(g_1)\oplus\dots\oplus\rho_K(g_K)$.
- $\mathcal{Z}_k$ may be multi-dimensional & need not align w/ the latent coordinate axes.
- Defined only w.r.t. a chosen direct-product decomposition of $G$.
```

```{note} Example
:class: dropdown
Grid world: object on an $n\times n$ cyclic grid w/ $n$ cyclic colors.

- $G=\mathbb{Z}_n\times\mathbb{Z}_n\times\mathbb{Z}_n$ (move $x$, move $y$, shift color).
- Encoder:

    $$
    f(x,y,c)=\big(e^{2\pi ix/n},\ e^{2\pi iy/n},\ e^{2\pi ic/n}\big)\in\mathbb{C}^3\cong\mathbb{R}^6
    $$

    - $x,y$: Grid position.
    - $c$: Color index.
- Generator of each subgroup multiplies one coordinate by $e^{2\pi i/n}$ → linear & disentangled.
- Over $\mathbb{R}$: block-diagonal 2×2 rotations → each factor takes 2 latent dims.
```

```{attention} Q&A
:class: dropdown
*Why does "one latent per factor" fail for cyclic factors?*

- Real 1-D rep of $\mathbb{Z}_n$: $\rho(g)^n=1$ → $\rho(g)=\pm1$ → faithful only for $n\le2$.
- → Faithful real **linear** rep of a cyclic factor ($n>2$: angle, hue, position on a torus) needs ≥2 dims (cos, sin).
- General (nonlinear) disentanglement does not impose this bound.

*Can every symmetry be disentangled?*

- ❌. Needs a direct-product decomposition. $SO(3)$ has none → rotations about $x$, $y$, $z$ cannot be disentangled under this definition.

*Can it be learned from static images?*

- ❌ in general: still images do not reveal how symmetries act. Caselles-Dupré et al. (NeurIPS 2019) argue it needs transitions $(o_t,a_t,o_{t+1})$, i.e., interaction.
    - $o_t$: Observation at step $t$.
    - $a_t$: Action at step $t$.
- Related: Locatello et al. (ICML 2019): unsupervised disentanglement is impossible w/o inductive biases on models & data.

*What does this tell us about world models & AC?*

- Equivariant world model: internal transformations mirror world transformations → simulate an action by applying $\rho(g)$ in latent space → planning, imagination.
- Hypothesis, not established: head-direction & grid cells encode heading & position w/ periodic codes, like cyclic-group reps → [Ring Attractor](topology.md#ring-attractor), [Toroidal Grid-Cell Manifold](topology.md#toroidal-grid-cell-manifold).
```

&nbsp;

### Grokking Modular Addition
- **What**: 1-layer transformer that learns $(a+b)\bmod p$ by composing rotations at a few Fourier frequencies, long after it memorizes. {cite:p}`nanda2023progress`
- **Why**: Fully reverse-engineered case → tests whether nets learn structure or a lookup table.
    - Gives continuous progress measures behind a "sudden" generalization jump.
- **How**:
    1. Embed $a\mapsto(\cos w_ka,\sin w_ka)$ at a few key frequencies → 2-D rotation irreps of $\mathbb{Z}_p$.
    2. Attention + MLP: trig identities → $\cos w_k(a+b)$, $\sin w_k(a+b)$ → compose rotations.
    3. Unembed: logit$(c)\approx\sum_k\alpha_k\cos(w_k(a+b-c))$ → all terms peak at $c\equiv a+b$ → constructive interference.
    4. Training: **memorization** → **circuit formation** → **cleanup** (weight decay removes memorization). Test accuracy jumps during cleanup.

```{note} Math
:class: dropdown
Notations:
- $a,b\in\mathbb{Z}_p$: Inputs.
- $c\in\mathbb{Z}_p$: Candidate output.
- $p$: Prime modulus.
- $\mathcal{K}$: Set of key frequencies.
- $w_k=2\pi k/p$: Angular frequency of $k\in\mathcal{K}$.
- $\alpha_k>0$: Per-frequency logit weight (positive → constructive interference).

Model (idealized algorithm):

$$\begin{align*}
\cos(w_k(a+b))&=\cos(w_ka)\cos(w_kb)-\sin(w_ka)\sin(w_kb)\\
\sin(w_k(a+b))&=\sin(w_ka)\cos(w_kb)+\cos(w_ka)\sin(w_kb)\\
\text{logit}(c)&\approx\sum_{k\in\mathcal{K}}\alpha_k\cos(w_k(a+b-c))\\
&=\sum_{k\in\mathcal{K}}\alpha_k\big[\cos(w_k(a+b))\cos(w_kc)+\sin(w_k(a+b))\sin(w_kc)\big]
\end{align*}$$

- $c\equiv a+b$ → every cosine $=1$. $p$ prime & $k\not\equiv0$ → every other $c$ gives $<1$.
- Group view: $R(w_ka)R(w_kb)=R(w_k(a+b))$; $\cos\theta=\tfrac12\mathrm{tr}\,R(\theta)$ → logit = sum of characters $\chi(a+b-c)$.

Training setup (mainline run):
- $p=113$; 1 layer, $d=128$, 4 heads, 512 ReLU MLP neurons, no LayerNorm.
- Train on 30% of the $113^2$ pairs. Full-batch AdamW, LR $10^{-3}$, weight decay $\lambda=1$, 40k epochs.
- Key frequencies $\mathcal{K}=\{14,35,41,42,52\}$. Their number & identity vary across runs.
- $\sum_k\alpha_k\cos(w_k(a+b-c))$ w/ least-squares $\alpha_k$ explains 95% of logit variance.
- Progress measures: **restricted loss** (ablate all non-key frequencies) & **excluded loss** (ablate key frequencies).
```

````{important} Code
:class: dropdown
```python
import torch

class FourierModAdd:
    def __init__(self, p, freqs):
        self.p = p
        self.w = 2 * torch.pi * torch.tensor(freqs, dtype=torch.float64) / p  ## key frequencies w_k

    def embed(self, a):  ## (B,) -> two (B, K): 2-D rotation irreps of Z_p
        return torch.cos(self.w * a[:, None]), torch.sin(self.w * a[:, None])

    def __call__(self, a, b):  ## (B,), (B,) -> (B, p) logits
        ca, sa = self.embed(a)
        cb, sb = self.embed(b)
        ## compose rotations via angle-addition identities -> no explicit "mod p" anywhere
        cab, sab = ca * cb - sa * sb, sa * cb + ca * sb
        cc, sc = self.embed(torch.arange(self.p))  ## (p, K)
        ## sum_k cos(w_k(a+b-c)) = sum_k [cos(w_k(a+b)) cos(w_k c) + sin(w_k(a+b)) sin(w_k c)]
        return cab @ cc.T + sab @ sc.T

## Example: all 113^2 pairs w/ the paper's 5 key frequencies (alpha_k = 1)
p = 113
model = FourierModAdd(p, [14, 35, 41, 42, 52])
a, b = torch.cartesian_prod(torch.arange(p), torch.arange(p)).T
print((model(a, b).argmax(-1) == (a + b) % p).all())  ## tensor(True)
```
````

```{attention} Q&A
:class: dropdown
*Why Fourier features?*

- Irreps of $\mathbb{Z}_p$ = Fourier modes → addition mod $p$ becomes rotation composition = multiplication of 2×2 matrices.
- Products of embeddings are cheap for attention + ReLU MLP to approximate.

*Why several frequencies, if one is already correct?*

- One cosine also peaks only at $c\equiv a+b$, but the margin to near neighbors in angle is small → high cross-entropy.
- Sum of several frequencies → off-target terms partly cancel → sharp peak.

*Why does weight decay matter?*

- Memorization & the Fourier circuit both fit the training set. The Fourier circuit needs a smaller weight norm.
- Weight decay favors the low-norm circuit & prunes memorization. Their runs w/o weight decay (or other regularization) did not grok.

*Is grokking sudden?*

- ❌ internally. Restricted & excluded loss move smoothly well before test accuracy jumps.
- Jump occurs during cleanup, after circuit formation. (Grokking first reported by Power et al., 2022.)

*Does this extend to other groups?*

- Group composition via representations (GCR): embed $a\mapsto\rho(a)$, compute $\rho(a)\rho(b)$, logit$(c)=\mathrm{tr}\,\rho(a)\rho(b)\rho(c^{-1})$ → peaks at $c=ab$ {cite:p}`chughtai2023toy`.
    - Studied on finite groups incl. non-abelian $S_5$ ($|G|=120$), w/ MLPs & transformers.
    - Finding: every network uses irreps (weak universality ✅), but which irreps & their learning order vary by seed (strong universality ❌).
- Contested: Stander et al. (ICML 2024) reverse-engineer $S_5$/$S_6$ nets as coset circuits & argue much GCR evidence fits both accounts.

*What does this tell us about intelligence/AC?*

- Nets can discover a task's symmetry group & its irreps w/o being told → structure, not a lookup table, when regularization favors compact solutions.
- Generalization = finding the algorithm that respects the symmetry.
- Caveat: tiny algorithmic task. Evidence at LLM scale is partial: circular features for days & months → [Multi-Dimensional Features](repr.md#multi-dimensional-features); feature directions → [Linear Representation Hypothesis](repr.md#linear-representation-hypothesis).
```

&nbsp;
