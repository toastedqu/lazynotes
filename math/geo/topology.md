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
# Topology & TDA
Shape invariants that survive continuous deformation, how to compute them from point clouds, & why neural codes have ring & torus shapes.
Standard references: {cite:t}`hatcher2002algebraic` (algebraic topology) & {cite:t}`carlsson2009topology` (TDA).

Notations:
- $X,Y$: Topological spaces.
- $\mathcal{P}=\{\mathbf{x}_i\}_{i=1}^m\subset\mathbb{R}^n$: Point cloud.
- $d(\cdot,\cdot)$: Metric. Euclidean on $\mathbb{R}^n$ unless stated.
- $K$: Simplicial complex.
- $\sigma$: Simplex.
- $\mathbb{F}$: Coefficient field ($\mathbb{Z}/2$ or $\mathbb{Q}$).
- $C_k$: $k$-chain space of $K$ over $\mathbb{F}$.
- $c_k$: #$k$-simplices of $K$ ($=\dim C_k$).
- $\partial_k:C_k\to C_{k-1}$: Boundary operator.
- $H_k$: $k$-th homology over $\mathbb{F}$.
- $\beta_k$: $k$-th Betti number.
- $\chi$: Euler characteristic.
- $\varepsilon$: Scale param.
- $S^1$: Circle.
- $S^2$: Sphere.
- $T^2$: Torus.
- $\cong$: Homeomorphic (spaces) / isomorphic (vector spaces).
- $\simeq$: Homotopy equivalent.

- **Overrides**: $k$ = simplex or homology dim, ❌class idx. $K$ = simplicial complex, ❌matrix. $X,Y$ = spaces, ❌RVs or input/output.

&nbsp;

## Topological Spaces

### Topological Space
- **What**: Set + family of "open" subsets, closed under arbitrary unions & finite intersections.
- **Why**: Shape without distance.
    - Continuity, connectedness & holes need only "nearness", not distance values.
    - → Invariants survive homeomorphic coordinate changes & metric changes that keep the topology.
    - ❌ Any metric: discrete metric on $\mathbb{R}$ → every point open → $\mathbb{R}$ disconnected.
- **How**:
    - Declare which subsets are open = which points are "near" each other.
    - Continuous map: preimage of every open set is open → "nearby stays nearby", w/o numbers.
    - Metric space → open balls generate the **metric topology**. Different metrics can give the same topology.

```{note} Math
:class: dropdown
Notations:
- $\tau$: Topology on $X$ (family of open sets).
- $U,V$: Open sets.

Definition: $(X,\tau)$ is a topological space if

1. $\emptyset\in\tau$ & $X\in\tau$.
2. $U_\alpha\in\tau\ \forall\alpha$ → $\bigcup_\alpha U_\alpha\in\tau$ (arbitrary unions).
3. $U,V\in\tau$ → $U\cap V\in\tau$ (finite intersections).

Continuity:

$$
f:X\to Y\text{ continuous}\iff f^{-1}(V)\in\tau_X\quad\forall V\in\tau_Y
$$

Properties:
- Metric topology: $U$ open ⇔ every $x\in U$ has a ball $B(x,r)\subseteq U$, $r>0$.
- Metric continuity ($\epsilon$-$\delta$) = topological continuity for the metric topology.
- All norms on $\mathbb{R}^n$ → same topology.
- **Connected**: $X$ is not a union of 2 disjoint nonempty open sets.
- **Compact**: every open cover has a finite subcover. In $\mathbb{R}^n$: compact ⇔ closed & bounded.
- **Hausdorff**: distinct points have disjoint open neighborhoods. Metric spaces are Hausdorff.
```

```{attention} Q&A
:class: dropdown
*Why only finite intersections?*

- Infinite intersections of open sets need not be open: $\bigcap_{t\ge1}(-1/t,1/t)=\{0\}$, which is not open in $\mathbb{R}$.

*Why care about topology in ML?*

- Coordinates of a learned representation are arbitrary: any homeomorphic re-coding is equally valid.
- Continuous bijection is not enough: the inverse must also be continuous.
- Metric quantities change under re-coding. Topological invariants do not.
- → Topology = label-free, coordinate-free description of a representation's "shape".

*How does this relate to a manifold?*

- [Smooth manifold](manifold.md#smooth-manifold) = Hausdorff topological space, locally homeomorphic to $\mathbb{R}^d$, + smooth structure.
- Topology gives the global shape. Differential geometry adds tangent spaces, metrics & curvature.
```

&nbsp;

### Homeomorphism
- **What**: Continuous bijection w/ continuous inverse.
- **Why**: Precise meaning of "same shape" → classify spaces up to homeomorphism.
    - Invariant = property kept by every homeomorphism → different invariant ⇒ ❌homeomorphic.
- **How**: Stretch & bend, ❌tear, ❌glue.
    - Coffee mug ≅ solid torus (donut): the handle becomes the hole.

```{note} Math
:class: dropdown
Definition:

$$
X\cong Y\iff\exists f:X\to Y\text{ bijective},\ f\ \&\ f^{-1}\text{ continuous}
$$

Topological invariants (kept by every homeomorphism):
- #connected components, compactness.
- Homology $H_k$, Betti numbers $\beta_k$, Euler characteristic $\chi$.
- Fundamental group $\pi_1$.
- Dimension: $\mathbb{R}^a\cong\mathbb{R}^b\Rightarrow a=b$ (Brouwer, invariance of domain).

Properties:
- Same invariants ⇏ homeomorphic. Every finite list of invariants is incomplete in general.
- Continuous bijection ⇏ homeomorphism: the inverse can fail to be continuous.
```

```{note} Example
:class: dropdown
- $(0,1)\cong\mathbb{R}$ via $x\mapsto\tan(\pi(x-\tfrac12))$ → length & boundedness are not topological.
- $[0,1)\to S^1$, $t\mapsto(\cos2\pi t,\sin2\pi t)$: continuous bijection, ❌homeomorphism.
    - Inverse jumps from near $1$ to $0$ at the point $(1,0)$.
- $S^1\not\cong\mathbb{R}$: remove 1 point → $S^1$ stays connected, $\mathbb{R}$ splits into 2 parts.
```

```{attention} Q&A
:class: dropdown
*Homeomorphism vs embedding?*

- Embedding: map $X\to Y$ that is a homeomorphism onto its image.
- Knotted & unknotted circles in $\mathbb{R}^3$: same space ($S^1$), different embeddings.

*Can an invertible NN change the topology of data?*

- ❌. Continuous invertible map w/ continuous inverse = homeomorphism → topology kept.
- Normalizing flows: full-support Gaussian → target supported on an annulus only approximately.
    - Homeomorphism of $\mathbb{R}^n$ keeps full support → output keeps some mass in the hole.
- Neural ODE flows are homeomorphisms → they cannot make nested classes (e.g., concentric annuli) linearly separable in the original dim.
    - Augmentation (extra dims) removes this obstruction (Dupont et al., Augmented Neural ODEs, NeurIPS 2019).

*Can a ReLU network change topology?*

- ✅. ReLU layers are non-injective → they glue points → they can kill holes & merge components.
- Empirically, Betti numbers of class manifolds decrease layer by layer in trained classifiers (Naitzat et al., JMLR 2020).
```

&nbsp;

### Homotopy Equivalence
- **What**: Continuous maps $f:X\to Y$ & $g:Y\to X$ w/ $g\circ f$ & $f\circ g$ continuously deformable to the identities.
- **Why**: Homeomorphism is too strict: thickness & dimension count.
    - Need a coarser equivalence that keeps holes & forgets thickness.
    - Homology sees only this coarser equivalence.
- **How**: Allow collapse: shrink thick parts onto a skeleton.
    - **Deformation retract**: shrink $X$ continuously onto a subspace $A$, w/ $A$ fixed at all times.
    - Annulus → $S^1$. Solid torus → $S^1$. $\mathbb{R}^n$ → point (**contractible**).

```{note} Math
:class: dropdown
Notations:
- $I=[0,1]$: Unit interval (deformation time).
- $A\subseteq X$: Subspace.

Homotopy between maps $f_0,f_1:X\to Y$:

$$
f_0\simeq f_1\iff\exists\ H:X\times I\to Y\text{ continuous},\ H(\cdot,0)=f_0,\ H(\cdot,1)=f_1
$$

Homotopy equivalence:

$$
X\simeq Y\iff\exists f:X\to Y,\ g:Y\to X:\quad g\circ f\simeq\mathrm{id}_X,\ f\circ g\simeq\mathrm{id}_Y
$$

- $f,g$: Continuous.

Deformation retraction of $X$ onto $A$:

$$
H:X\times I\to X,\quad H(x,0)=x,\quad H(x,1)\in A,\quad H(a,t)=a\ \ \forall a\in A,t\in I
$$

Properties:
- $X\cong Y\Rightarrow X\simeq Y$. Converse ❌: disk $\simeq$ point, disk $\not\cong$ point.
- Deformation retract onto $A$ → $X\simeq A$.
- **Homotopy invariance**: $X\simeq Y\Rightarrow H_k(X)\cong H_k(Y)\ \forall k$.
```

```{tip} Derivation
:class: dropdown
*Why does homology see only homotopy type?*

1. Every continuous $f:X\to Y$ induces a linear map $f_*:H_k(X)\to H_k(Y)$.
2. Functoriality: $(g\circ f)_*=g_*\circ f_*$ & $(\mathrm{id})_*=\mathrm{id}$.
3. Homotopic maps induce equal maps: $f_0\simeq f_1\Rightarrow f_{0*}=f_{1*}$.
4. $g\circ f\simeq\mathrm{id}_X$ → $g_*f_*=\mathrm{id}$. Same for $f_*g_*$.
5. → $f_*$ is an isomorphism.
6. → Two homotopy-equivalent spaces have the same homology, even if they are not homeomorphic.
```

```{attention} Q&A
:class: dropdown
*Why does TDA only need homotopy type?*

- Union of balls around samples is "thick", not the true shape.
- Sample density & ball radius suitable relative to the reach → thick version $\simeq$ true shape → same homology → recoverable.
- Density alone is not enough: radius too large vs reach → fake fills & merges.

*Why compute homology, not homotopy groups?*

- Fundamental group $\pi_1$ & higher $\pi_k$ are finer invariants.
- $\pi_1$ is non-abelian in general. Equality of 2 finitely presented groups is undecidable.
- Homology over a field = linear algebra → polynomial-time computable.

*What is lost?*

- Thickness, dimension of the solid parts, & all geometry.
- A contractible space (ball, tree, $\mathbb{R}^n$) has the same homology as a point.
```

&nbsp;

## Homology

### Simplicial Complex
- **What**: Collection of simplices (vertices, edges, triangles, tetrahedra, ...), closed under taking faces.
- **Why**: Finite, discrete model of a space → topology becomes combinatorics & linear algebra.
- **How**:
    1. $k$-simplex = $k+1$ vertices: vertex ($k=0$), edge (1), triangle (2), tetrahedron (3).
    2. Include every face of every simplex.
    3. Two simplices meet only along a shared face.
    4. Boundary operator: $k$-simplex → signed sum of its $(k-1)$-faces.

```{note} Math
:class: dropdown
Notations:
- $V$: Vertex set.
- $v_0,\dots,v_k$: Ordered vertices of a $k$-simplex.

Definition (abstract simplicial complex):

$$
K\subseteq2^V\setminus\{\emptyset\},\qquad\sigma\in K,\ \emptyset\neq\tau\subseteq\sigma\Rightarrow\tau\in K
$$

- $\dim\sigma=|\sigma|-1$.
- $\tau$: Face of $\sigma$.

Chains: $C_k$ = $\mathbb{F}$-vector space w/ basis = $k$-simplices of $K$.

Boundary operator:

$$
\partial_k[v_0,\dots,v_k]=\sum_{i=0}^k(-1)^i[v_0,\dots,\hat v_i,\dots,v_k]
$$

- $\hat v_i$: Omitted vertex.
- Matrix form: $\partial_k\in\mathbb{F}^{c_{k-1}\times c_k}$.
- $\partial_0=0$.
- Over $\mathbb{Z}/2$: signs vanish → boundary = sum of faces.

Key property:

$$
\partial_{k-1}\circ\partial_k=0
$$
```

```{tip} Derivation
:class: dropdown
*Why is $\partial\partial=0$?*

1. Apply $\partial_{k-1}$ to $\partial_k[v_0,\dots,v_k]$ → terms w/ 2 vertices $v_i,v_j$ removed, $i<j$.
2. Remove $v_j$ first (sign $(-1)^j$), then $v_i$ (still at position $i$, sign $(-1)^i$) → $(-1)^{i+j}$.
3. Remove $v_i$ first (sign $(-1)^i$), then $v_j$ (now at position $j-1$, sign $(-1)^{j-1}$) → $(-1)^{i+j-1}$.
4. Same face, opposite signs → each pair cancels → sum $=0$.
5. Geometric meaning: a boundary has no boundary.
```

```{note} Example
:class: dropdown
Hollow triangle, vertices $a<b<c$, edges $ab,ac,bc$:

$$
\partial_1=\begin{bmatrix}-1&-1&0\\1&0&-1\\0&1&1\end{bmatrix}\quad\begin{matrix}\leftarrow a\\\leftarrow b\\\leftarrow c\end{matrix}
$$

- Columns: $ab,ac,bc$.
- Chain $ab-ac+bc$: boundary $(b-a)-(c-a)+(c-b)=0$ → closed loop.
- Fill in triangle $abc$: $\partial_2[a,b,c]=bc-ac+ab$ → the loop is now a boundary.
```

````{important} Code
:class: dropdown
```python
import itertools
import numpy as np

class SimplicialComplex:
    def __init__(self, top_simplices):
        ## closure: add every nonempty face -> valid complex
        faces = set()
        for s in top_simplices:
            s = tuple(sorted(s))
            for r in range(1, len(s) + 1):
                faces.update(itertools.combinations(s, r))
        self.simplices = {}  ## dim k -> sorted list of k-simplices
        for f in sorted(faces):
            self.simplices.setdefault(len(f) - 1, []).append(f)

    def boundary(self, k):
        ## matrix of d_k: rows = (k-1)-simplices, cols = k-simplices
        rows = self.simplices.get(k - 1, [])
        cols = self.simplices.get(k, [])
        D = np.zeros((len(rows), len(cols)))
        if k == 0:
            return D  ## d_0 = 0
        idx = {s: r for r, s in enumerate(rows)}
        for c, s in enumerate(cols):
            for i in range(len(s)):
                D[idx[s[:i] + s[i + 1:]], c] = (-1) ** i  ## drop i-th vertex, sign (-1)^i
        return D

## Example
K = SimplicialComplex([(0, 1), (1, 2), (0, 2)])  ## hollow triangle
print(K.boundary(1))  ## [[-1,-1,0],[1,0,-1],[0,1,1]]
F = SimplicialComplex([(0, 1, 2)])  ## filled triangle
print(F.boundary(1) @ F.boundary(2))  ## [[0],[0],[0]] -> boundary of a boundary = 0
```
````

```{attention} Q&A
:class: dropdown
*Simplicial complex vs graph vs hypergraph?*

- Graph = 1-dim simplicial complex.
- Hypergraph: need not be closed under faces → not a simplicial chain complex.
    - Homology needs an extra construction (e.g., embedded homology).

*Can every space be triangulated?*

- Every smooth manifold ✅ (Whitehead).
- Some topological manifolds ❌ (dim $\ge4$).
- Point clouds: build the complex from distances → [Vietoris-Rips](#vietoris-rips-complex).

*How few simplices does a torus need?*

- Simplicial: 7 vertices, 21 edges, 14 triangles (minimal).
- CW complex (cells, not simplices): 1 vertex, 2 edges, 1 square face.

*How does this relate to the graph Laplacian?*

- $\partial_1$ = signed vertex–edge incidence matrix → $\partial_1\partial_1^T$ = [graph Laplacian](../linalg.md#graph-laplacian) $D-A$.
- **Hodge Laplacian**: $L_k=\partial_k^T\partial_k+\partial_{k+1}\partial_{k+1}^T$ (over $\mathbb{R}$).
- $\dim\ker L_k=\beta_k$ → harmonic $k$-chains = holes. $k=0$: #zero eigenvalues of the graph Laplacian = #components.
- Basis of simplicial / higher-order message passing NNs.
```

&nbsp;

### Homology
- **What**: Vector space of $k$-dim holes: $k$-cycles modulo $k$-boundaries.
- **Why**: "Hole" is intuitive in 2-D/3-D, vague in high dim.
    - Homology turns "hole" into linear algebra → exact, computable, coordinate-free.
- **How**:
    1. **Chains**: formal sums of $k$-simplices.
    2. **Cycles**: chains w/o boundary ($\partial c=0$) → closed loops, closed shells.
    3. **Boundaries**: chains that bound a $(k+1)$-chain → filled-in cycles.
    4. $\partial\partial=0$ → every boundary is a cycle.
    5. Hole = cycle that is not a boundary → $H_k$ = cycles / boundaries.
    6. Two cycles are equivalent if they differ by a boundary.
    7. $H_k$ also holds $0$ & sums of classes → a basis of $H_k$ = $\beta_k$ independent holes.

```{note} Math
:class: dropdown
Notations:
- $Z_k$: $k$-cycles.
- $B_k$: $k$-boundaries.

Definition:

$$
Z_k=\ker\partial_k,\qquad B_k=\operatorname{im}\partial_{k+1}\subseteq Z_k,\qquad H_k=Z_k/B_k
$$

Betti number over a field:

$$
\beta_k=\dim H_k=c_k-\operatorname{rank}\partial_k-\operatorname{rank}\partial_{k+1}
$$

Properties:
- Homotopy invariant: $X\simeq Y\Rightarrow H_k(X)\cong H_k(Y)$.
- **Functorial**: $f:X\to Y$ → linear $f_*:H_k(X)\to H_k(Y)$. Basis of [persistence](#persistent-homology).
- Independent of the triangulation. Singular homology extends it to every space.
- Over $\mathbb{Z}$: $H_k$ can have torsion (e.g., $\mathbb{Z}/2$) → field choice can change $\beta_k$.
- Cost: Gaussian elimination on $\partial_k$ → cubic in #simplices in the worst case.
```

```{tip} Derivation
:class: dropdown
*Why $\beta_k=c_k-\operatorname{rank}\partial_k-\operatorname{rank}\partial_{k+1}$?*

1. Rank–nullity on $\partial_k:C_k\to C_{k-1}$: $\dim Z_k=c_k-\operatorname{rank}\partial_k$.
2. $B_k$ = image of $\partial_{k+1}$ → $\dim B_k=\operatorname{rank}\partial_{k+1}$.
3. Quotient of vector spaces: $\dim(Z_k/B_k)=\dim Z_k-\dim B_k$.
```

```{note} Example
:class: dropdown
Hollow triangle: $c_0=3$, $c_1=3$, $\operatorname{rank}\partial_1=2$, $\partial_2=0$.

- $\beta_0=3-0-2=1$: one component.
- $\beta_1=3-2-0=1$: one loop.

Filled triangle: add $c_2=1$, $\operatorname{rank}\partial_2=1$.

- $\beta_1=3-2-1=0$: the loop is now a boundary.
- $\beta_2=1-1-0=0$.
```

````{important} Code
:class: dropdown
```python
import itertools
import numpy as np

class Homology:
    ## Reuses SimplicialComplex from Simplicial Complex (this page)
    def __init__(self, K):
        self.K = K

    def rank(self, k):
        D = self.K.boundary(k)
        return 0 if D.size == 0 else int(np.linalg.matrix_rank(D))  ## rank over Q (tiny integer matrices)

    def __call__(self):
        ## beta_k = dim C_k - rank d_k - rank d_{k+1}
        top = max(self.K.simplices)
        return [len(self.K.simplices[k]) - self.rank(k) - self.rank(k + 1) for k in range(top + 1)]

## Example
sphere = SimplicialComplex(itertools.combinations(range(4), 3))  ## hollow tetrahedron
print(Homology(sphere)())  ## [1, 0, 1]
## 7-vertex torus: triangles {i, i+1, i+3} & {i, i+2, i+3} mod 7
torus = SimplicialComplex([t for i in range(7) for t in
                           [(i, (i + 1) % 7, (i + 3) % 7), (i, (i + 2) % 7, (i + 3) % 7)]])
print(Homology(torus)())  ## [1, 2, 1]
```
````

```{attention} Q&A
:class: dropdown
*Why mod out boundaries?*

- Every boundary is a cycle, but it encloses nothing.
- Quotient → a cycle counts only if it is not "filled in".
- Two loops around the same hole differ by a boundary → same class → counted once.

*Why use field coefficients?*

- Field → $H_k$ is a vector space → one number ($\beta_k$) describes it.
- $\mathbb{Z}/2$ → no orientation bookkeeping, fast bit operations → TDA default.

*How does homology relate to the Euler characteristic?*

- $\chi$ = alternating sum of $\beta_k$ → [Euler Characteristic](#euler-characteristic).
```

&nbsp;

#### Betti Numbers
- **What**: $\beta_k=\dim H_k$: #independent $k$-dim holes.
- **Why**: One integer per dim → compact, comparable shape signature.
- **How**:
    - $\beta_0$: #connected components.
    - $\beta_1$: #independent loops (tunnels).
    - $\beta_2$: #enclosed voids (cavities).
    - $\beta_k=0$ for $k>\dim X$.

```{dropdown} Table: Betti Numbers of Common Spaces
| Space | $\beta_0$ | $\beta_1$ | $\beta_2$ | $\chi$ |
|:--|:--|:--|:--|:--|
| Point, disk, ball, $\mathbb{R}^n$ | 1 | 0 | 0 | 1 |
| Circle $S^1$ | 1 | 1 | 0 | 0 |
| Figure eight | 1 | 2 | 0 | −1 |
| Sphere $S^2$ | 1 | 0 | 1 | 2 |
| Torus $T^2$ | 1 | 2 | 1 | 0 |
| Genus-$g$ surface | 1 | $2g$ | 1 | $2-2g$ |
| Klein bottle over $\mathbb{Z}/2$ | 1 | 2 | 1 | 0 |
| Klein bottle over $\mathbb{Q}$ | 1 | 1 | 0 | 0 |
| $m$ isolated points | $m$ | 0 | 0 | $m$ |

- Genus $g$: #handles of a closed orientable surface.
```

```{attention} Q&A
:class: dropdown
*Why does the torus have $\beta_1=2$?*

- 2 independent loops: around the tube (meridian) & around the central hole (longitude).
- $T^2=S^1\times S^1$ → Künneth formula: Betti polynomials multiply: $(1+t)^2=1+2t+t^2$ → $(1,2,1)$.

*Why do Betti numbers depend on the field?*

- Torsion: Klein bottle has $H_1(\cdot;\mathbb{Z})=\mathbb{Z}\oplus\mathbb{Z}/2$.
- $\mathbb{Z}/2$ coefficients see the $\mathbb{Z}/2$ part. $\mathbb{Q}$ coefficients do not.
- Orientable surfaces: no torsion → same $\beta_k$ over every field.

*Do equal Betti numbers imply the same space?*

- ❌. Poincaré homology sphere: same homology as the 3-sphere, but nontrivial $\pi_1$ → not homeomorphic.
- Betti numbers ignore all geometry: a small & a large circle are equal.

*How does this relate to classification in NNs?*

- Class regions w/ complex topology (many components, loops) → harder to separate.
- Trained ReLU classifiers simplify Betti numbers of class data layer by layer (Naitzat et al., JMLR 2020).
```

&nbsp;

### Euler Characteristic
- **What**: $\chi=\sum_k(-1)^kc_k=\sum_k(-1)^k\beta_k$.
- **Why**: Cheapest topological invariant.
    - Count simplices, ❌linear algebra.
    - Same value for every triangulation of the same space.
- **How**: Count vertices − edges + triangles − ... → equals the alternating Betti sum, because the ranks cancel.

```{note} Math
:class: dropdown
Definition:

$$
\chi(K)=\sum_{k\ge0}(-1)^kc_k=\sum_{k\ge0}(-1)^k\beta_k
$$

Properties:
- Homotopy invariant.
- Independent of the field (although $\beta_k$ can depend on it).
- Closed orientable surface of genus $g$: $\chi=2-2g$.
- Product: $\chi(X\times Y)=\chi(X)\chi(Y)$ → $\chi(T^2)=0\cdot0=0$.
- Inclusion–exclusion (subcomplexes): $\chi(A\cup B)=\chi(A)+\chi(B)-\chi(A\cap B)$.
- Gauss–Bonnet (closed surface $M$): $\int_M\kappa\,dA=2\pi\chi(M)$ → total [curvature](manifold.md#curvature) is topological.
    - $\kappa$: Gaussian curvature.
    - $dA$: Area element.
```

```{tip} Derivation
:class: dropdown
*Why is the alternating simplex count equal to the alternating Betti sum?*

1. Rank–nullity: $c_k=\dim Z_k+\operatorname{rank}\partial_k$.
2. $\dim Z_k=\beta_k+\operatorname{rank}\partial_{k+1}$.
3. → $c_k=\beta_k+\operatorname{rank}\partial_k+\operatorname{rank}\partial_{k+1}$.
4. Alternating sum: each $\operatorname{rank}\partial_k$ appears twice w/ opposite signs → cancels.
5. Ends: $\partial_0=0$ & $\partial_{N+1}=0$ ($N$ = top dim) → $\sum(-1)^kc_k=\sum(-1)^k\beta_k$.
```

```{note} Example
:class: dropdown
- Tetrahedron surface: $4-6+4=2$ → $S^2$.
- Cube surface: $8-12+6=2$ → $S^2$ (Euler's polyhedron formula $V-E+F=2$).
- 7-vertex torus: $7-21+14=0$.
- Hollow triangle: $3-3=0=\beta_0-\beta_1=1-1$.
```

```{attention} Q&A
:class: dropdown
*Why is $\chi$ weaker than Betti numbers?*

- One number for all dims → cancellations.
- $S^1$, $T^2$ & the Klein bottle all have $\chi=0$.

*Use in TDA?*

- **Euler characteristic curve**: $\chi$ of the complex at each scale $\varepsilon$.
- Cheap (counts only) → fast summary for large data, e.g., images & random fields.
```

&nbsp;

## Topological Data Analysis
- **What**: Infer the topology of the space behind a finite sample.

### Vietoris-Rips Complex
- **What**: Simplex for every vertex subset w/ all pairwise distances $\le\varepsilon$.
- **Why**: Finite point cloud has trivial topology: $\beta_0=m$, all other $\beta_k=0$.
    - Connect nearby points → recover the shape of the underlying space at scale $\varepsilon$.
- **How**:
    1. Select scale $\varepsilon$.
    2. Add an edge for every pair at distance $\le\varepsilon$.
    3. Add every clique of that graph as a simplex (**flag** / clique complex).
    4. Increase $\varepsilon$ → nested complexes → **filtration** → [Persistent Homology](#persistent-homology).

```{note} Math
:class: dropdown
Definition:

$$
\mathrm{VR}_\varepsilon(\mathcal{P})=\{\sigma\subseteq\mathcal{P}:\sigma\neq\emptyset,\ d(\mathbf{x},\mathbf{x}')\le\varepsilon\ \ \forall\mathbf{x},\mathbf{x}'\in\sigma\}
$$

Properties:
- Flag complex: the edges alone define it.
- Nested: $\varepsilon\le\varepsilon'\Rightarrow\mathrm{VR}_\varepsilon\subseteq\mathrm{VR}_{\varepsilon'}$.
- Needs only pairwise distances → works in any metric space, any dim.
- Size: up to $2^m-1$ simplices → compute $H_k$ w/ simplices up to dim $k+1$ only.
- Convention: some sources use diameter $\le2\varepsilon$ (radius scale) → all scales ×½.
```

````{important} Code
:class: dropdown
```python
import itertools
import numpy as np

class VietorisRips:
    ## Reuses SimplicialComplex & Homology from this page
    def __init__(self, max_dim=2):
        self.max_dim = max_dim  ## H_k needs simplices up to dim k+1

    def __call__(self, P, eps):
        m = len(P)
        D = np.linalg.norm(P[:, None] - P[None], axis=-1)  ## (m, m) pairwise distances
        simplices = [(i,) for i in range(m)]
        for r in range(2, self.max_dim + 2):  ## r = #vertices = dim + 1
            for s in itertools.combinations(range(m), r):
                ## flag condition: every pair within eps
                if all(D[a, b] <= eps for a, b in itertools.combinations(s, 2)):
                    simplices.append(s)
        return SimplicialComplex(simplices)

## Example: 4 corners of a unit square (diagonal = 1.414)
P = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
for eps in [0.5, 1.0, 1.5]:
    print(eps, Homology(VietorisRips(max_dim=2)(P, eps))())
## 0.5 [4]         -> 4 isolated points
## 1.0 [1, 1]      -> square loop
## 1.5 [1, 0, 1]   -> beta_2 = 1 is an artifact: tetrahedron missing (max_dim=2)
print(Homology(VietorisRips(max_dim=3)(P, 1.5))())  ## [1, 0, 0, 0] -> filled, contractible
```
````

```{attention} Q&A
:class: dropdown
*Why Rips & not Čech?*

- Rips: only pairwise distances → cheap test, any metric.
- Čech: needs common intersections of balls → expensive in high dim.
- The two are interleaved (Čech Math) → features w/ a large death/birth ratio appear in both.

*Why can a Rips complex show fake holes?*

- Rips at one fixed $\varepsilon$ need not be $\simeq$ the union of balls.
- Truncation: dropping simplices above dim $k+1$ → fake $H_{k+1}$ classes (see Code).

*How to scale to large $m$?*

- Cap dim & $\varepsilon$.
- Subsample landmarks (maxmin / farthest-point sampling).
- Optimized software (e.g., Ripser) → apparent pairs, cohomology, implicit matrices.
```

&nbsp;

#### Čech Complex
- **What**: Simplex for every vertex subset whose radius-$r$ balls have a common point.
- **Why**: Exact link to the union of balls → provable homotopy type.
- **How**:
    1. Grow a ball of radius $r$ around each point.
    2. Add a simplex when its balls share a common point.
    3. **Nerve theorem** → complex $\simeq$ union of balls.

```{note} Math
:class: dropdown
Notations:
- $r$: Ball radius.
- $B(\mathbf{x},r)$: Closed ball of radius $r$ around $\mathbf{x}$.
- $\mathcal{U}=\{U_\alpha\}$: Cover of a space.

Čech complex:

$$
\check{C}_r(\mathcal{P})=\Big\{\sigma\subseteq\mathcal{P}:\sigma\neq\emptyset,\ \bigcap_{\mathbf{x}\in\sigma}B(\mathbf{x},r)\neq\emptyset\Big\}
$$

Nerve of a cover:

$$
N(\mathcal{U})=\Big\{\{\alpha_0,\dots,\alpha_k\}:U_{\alpha_0}\cap\dots\cap U_{\alpha_k}\neq\emptyset\Big\}
$$

**Good cover**: every nonempty finite intersection is contractible.

**Nerve theorem**: $N(\mathcal{U})\simeq\bigcup_\alpha U_\alpha$ if either:
- $\mathcal{U}$ = open good cover of a paracompact space.
- $\mathcal{U}$ = finite cover by closed convex sets in $\mathbb{R}^n$ (always good).
- Hypotheses matter: $\{\{0\},(0,1]\}$ covers $[0,1]$, sets are contractible & disjoint → nerve = 2 points $\not\simeq[0,1]$.
- Balls are convex → $\check{C}_r(\mathcal{P})\simeq\bigcup_i B(\mathbf{x}_i,r)$.

Rips–Čech interleaving (diameter-convention Rips):

$$
\check{C}_r(\mathcal{P})\subseteq\mathrm{VR}_{2r}(\mathcal{P})\subseteq\check{C}_{2r}(\mathcal{P})
$$
```

```{tip} Derivation
:class: dropdown
*Why $\check{C}_r\subseteq\mathrm{VR}_{2r}\subseteq\check{C}_{2r}$?*

1. $\sigma\in\check{C}_r$ → common point $\mathbf{z}$ w/ $d(\mathbf{z},\mathbf{x})\le r$ for all $\mathbf{x}\in\sigma$.
2. Triangle inequality → $d(\mathbf{x},\mathbf{x}')\le2r$ → $\sigma\in\mathrm{VR}_{2r}$.
3. $\sigma\in\mathrm{VR}_{2r}$ → pick any vertex $\mathbf{x}_0\in\sigma$ → $d(\mathbf{x}_0,\mathbf{x})\le2r$ for all $\mathbf{x}\in\sigma$.
4. → $\mathbf{x}_0$ lies in every radius-$2r$ ball → $\sigma\in\check{C}_{2r}$.
```

```{attention} Q&A
:class: dropdown
*Why does the nerve theorem matter?*

- It turns a continuous space (union of balls) into a finite complex w/ the same homotopy type → same homology.
- Same idea in [Mapper](#mapper): nerve of a cover = summary graph.

*When does the union of balls recover the true shape?*

- Niyogi, Smale & Weinberger (2008), conditions:
    - Compact smooth manifold w/o boundary, reach $>0$ (bounds curvature & near self-intersections).
    - Noiseless sample.
    - Ball radius small relative to the reach.
    - Sample dense relative to the ball radius.
- → Union of balls deformation retracts onto the manifold.
- Density alone is not enough: the radius-vs-reach condition is also needed.
- → Under the [manifold hypothesis](data.md#manifold-hypothesis), TDA can recover the data manifold's homology.
```

&nbsp;

### Persistent Homology
- **What**: Homology tracked across a growing filtration: birth & death scale of every hole. {cite:p}`edelsbrunner2002topological,zomorodian2005computing`
- **Why**: One fixed $\varepsilon$ is arbitrary.
    - $\varepsilon$ too small → dust ($\beta_0=m$). $\varepsilon$ too large → one contractible blob.
    - Keep all scales → long-lived features = signal, short-lived = noise.
- **How**:
    1. Build a filtration $K_0\subseteq K_1\subseteq\dots\subseteq K_N$, e.g., Rips over increasing $\varepsilon$.
    2. Inclusions induce linear maps $H_k(K_i)\to H_k(K_j)$, $i\le j$.
    3. Class is born at the first scale where it appears.
    4. Class dies when it gets filled in, or merges into an older class (**elder rule**).
    5. Output per dim $k$: multiset of intervals $[b,d)$ → **barcode** / [persistence diagram](#persistence-diagram).

```{note} Math
:class: dropdown
Notations:
- $K_0\subseteq\dots\subseteq K_N$: Filtration.
- $\varepsilon_i$: Scale at which $K_i$ appears.
- $\iota_*^{i,j}:H_k(K_i)\to H_k(K_j)$: Map induced by inclusion, $i\le j$.
- $b,d$: Birth & death scales of a class.

Persistent homology & persistent Betti number:

$$
H_k^{i,j}=\operatorname{im}\iota_*^{i,j},\qquad\beta_k^{i,j}=\dim H_k^{i,j}
$$

- $\beta_k^{i,j}$: #$k$-dim classes alive at $\varepsilon_i$ that are still alive at $\varepsilon_j$.

**Structure theorem** (field coefficients, finite filtration):

$$
\big(H_k(K_i)\big)_{i=0}^N\cong\bigoplus_l\mathbb{I}[b_l,d_l)
$$

- $\mathbb{I}[b,d)$: Interval module: $\mathbb{F}$ on scales in $[b,d)$, $0$ elsewhere, identity maps inside.
- Decomposition is unique → barcode $\{[b_l,d_l)\}$ is a complete invariant of the filtration's homology.
- $d_l=\infty$: class never dies (essential class = a hole of the final complex).

Process (standard reduction):
1. Order all simplices by entry scale; break ties by dim (faces first).
2. Build the full boundary matrix $\partial$ in that order.
3. $\mathrm{low}(j)$: row idx of the lowest nonzero entry of column $j$.
4. For $j=1..$: while some $j'<j$ has $\mathrm{low}(j')=\mathrm{low}(j)$, add column $j'$ to column $j$ (over $\mathbb{Z}/2$).
5. Nonzero column $j$ w/ $\mathrm{low}(j)=i$ → pair: $\sigma_i$ creates a class, $\sigma_j$ kills it → bar $[\varepsilon(\sigma_i),\varepsilon(\sigma_j))$.
6. Zero column $j$ never paired as a low → bar $[\varepsilon(\sigma_j),\infty)$.

- Cost: cubic in #simplices in the worst case. Much faster in practice ← sparse columns.
```

```{note} Example
:class: dropdown
40 noisy samples of a circle, Rips filtration:

- $H_0$: 40 bars born at 0. 39 die at small $\varepsilon$ (neighbor spacing). 1 bar $[0,\infty)$.
- $H_1$: many short bars (noise loops) + 1 long bar.
    - Born ≈ when neighbors connect around the ring.
    - Dies ≈ when edges span the ring & fill it.
- Read-off: long bars → $\beta_0=1$, $\beta_1=1$ → circle Betti signature, consistent w/ a circle.
- Betti numbers alone do not identify the space.
```

````{important} Code
:class: dropdown
```python
import itertools
import numpy as np

class ZeroDimPersistence:
    ## H_0 bars of the Rips filtration = Kruskal's MST w/ union-find
    def __init__(self, max_eps=np.inf):
        self.max_eps = max_eps

    def __call__(self, P):
        m = len(P)
        D = np.linalg.norm(P[:, None] - P[None], axis=-1)  ## (m, m)
        edges = sorted((D[i, j], i, j) for i, j in itertools.combinations(range(m), 2))
        parent = list(range(m))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]  ## path halving
                a = parent[a]
            return a

        bars = []
        for eps, i, j in edges:  ## edges enter the filtration in length order
            if eps > self.max_eps:
                break
            ri, rj = find(i), find(j)
            if ri != rj:  ## edge merges 2 components -> one H_0 class dies
                parent[ri] = rj
                bars.append((0.0, float(eps)))  ## all points born at 0 -> elder-rule tie
        n_alive = len({find(a) for a in range(m)})
        return bars + [(0.0, np.inf)] * n_alive

## Example: 2 tight pairs, far apart
P = np.array([[0., 0.], [1., 0.], [10., 0.], [11., 0.]])
print(ZeroDimPersistence()(P))  ## [(0.0, 1.0), (0.0, 1.0), (0.0, 9.0), (0.0, inf)]
## long finite bar (0, 9) -> 2 clusters at every scale in [1, 9)
```
````

```{attention} Q&A
:class: dropdown
*Is a long bar always signal & a short bar always noise?*

- Heuristic, justified by the [Stability Theorem](#stability-theorem).
- Short bars can carry real info (e.g., curvature, small features).
- Statistical approach: bootstrap confidence bands on the diagram → separate signal from noise.

*How does $H_0$ persistence relate to clustering?*

- $H_0$ deaths = MST edge lengths = merge heights of single-linkage [hierarchical clustering](../../ml/unsupervised.md#hierarchical-clustering).
- → Persistent homology = generalization of single linkage to loops & voids.

*What other filtrations exist?*

- Sublevel sets of a function $f$: $K_a=\{\sigma:f(\sigma)\le a\}$ → images (pixel intensity), loss landscapes, graphs (node degree).
    - Requires monotone $f$: $f(\tau)\le f(\sigma)$ whenever $\tau\subseteq\sigma$ → each $K_a$ is a complex.
- Rips / Čech / alpha (Delaunay-based, low dim) on point clouds.

*Is persistent homology invariant to everything a homeomorphism keeps?*

- ❌. Rips uses distances → invariant to isometries (rotations, translations), not to rescaling or nonlinear warps.
- Bar lengths scale w/ the data.
- Topology (#long bars) is robust to moderate warps. Exact bar lengths are geometric.

*How is it used w/ NNs?*

- Input features: vectorize diagrams (persistence images, landscapes) → feed a model.
- Topological losses: birth & death values are differentiable a.e. w.r.t. the filtration values → regularize shape of outputs or latents.
- Analysis: persistent homology of activations / representations → shape of internal codes, e.g., [ring](#ring-attractor) & [torus](#toroidal-grid-cell-manifold) codes in brains, [cyclic features](repr.md#multi-dimensional-features) in LMs.
```

&nbsp;

#### Persistence Diagram
- **What**: Multiset of points $(b,d)$, one per bar, + the diagonal $b=d$ w/ infinite multiplicity.
- **Why**: Barcode needs a geometry to compare & average shapes.
    - Diagram → points far from the diagonal = persistent; near the diagonal = noise.
    - Metrics between diagrams = optimal matchings.
- **How**:
    - **Barcode**: one horizontal bar $[b,d)$ per class, stacked. Same info as the diagram.
    - **Diagram**: plot each bar as point $(b,d)$ above the diagonal.
    - Persistence $=d-b$ ∝ distance to the diagonal.
    - Compare 2 diagrams: match points; unmatched points go to the diagonal.

```{note} Math
:class: dropdown
Notations:
- $D_1,D_2$: Persistence diagrams.
- $\Delta=\{(t,t)\}$: Diagonal, infinite multiplicity.
- $\gamma$: Bijection $D_1\cup\Delta\to D_2\cup\Delta$.
- $q$: Wasserstein order.

Bottleneck distance:

$$
d_B(D_1,D_2)=\inf_\gamma\sup_{\mathbf{p}\in D_1\cup\Delta}\|\mathbf{p}-\gamma(\mathbf{p})\|_\infty
$$

$q$-Wasserstein distance:

$$
W_q(D_1,D_2)=\Big(\inf_\gamma\sum_{\mathbf{p}\in D_1\cup\Delta}\|\mathbf{p}-\gamma(\mathbf{p})\|_\infty^q\Big)^{1/q}
$$

Properties:
- Cost to send $(b,d)$ to $\Delta$: $\min_t\max(|b-t|,|d-t|)=(d-b)/2$.
- $d_B=\lim_{q\to\infty}W_q$.
- $W_q$ = [Wasserstein distance](ot.md#wasserstein-distance) w/ the diagonal as a free sink.
```

```{note} Example
:class: dropdown
$D_1=\{(0,4)\}$, $D_2=\{(0,5),(1,1.4)\}$:

- Match $(0,4)\leftrightarrow(0,5)$: cost $1$.
- $(1,1.4)\to\Delta$: cost $0.2$.
- Alternative: $(0,4)\to\Delta$ & $(0,5)\to\Delta$: costs $2$ & $2.5$ → worse.
- → $d_B=1$, $W_1=1.2$.
```

```{attention} Q&A
:class: dropdown
*Bottleneck vs Wasserstein?*

- Bottleneck: only the worst matched pair → robust to many small noise points, blind to them.
- $W_q$: sums all costs → sensitive to the total noise mass.

*Why add the diagonal?*

- Diagrams have different #points.
- Diagonal = free sink → noise points vanish at cost ∝ their persistence.

*How to feed a diagram to ML?*

- A diagram is a multiset, not a vector.
- Vectorize: persistence images (smoothed density on a grid), persistence landscapes (functions), Betti curves.
- Or learn: permutation-invariant nets on the point set ([Deep Sets](symmetry.md#deep-sets)).
```

&nbsp;

#### Stability Theorem
- **What**: Small perturbation of the input → small change of the persistence diagram. {cite:p}`cohensteiner2007stability`
- **Why**: Real data are noisy.
    - W/o stability, diagrams would be useless features.
    - Justifies "long bar = real feature".
- **How**:
    - Functions: sublevel-set diagrams move by at most the sup-norm change of the function.
    - Point clouds (Čech): distance function to the sample → Hausdorff perturbation bound.
    - Point clouds (Rips): correspondence → Gromov–Hausdorff perturbation bound.

```{note} Math
:class: dropdown
Notations:
- $f,g:X\to\mathbb{R}$: Functions on a triangulable space $X$.
- $\mathrm{Dgm}_k(f)$: $k$-dim diagram of the sublevel-set filtration $f^{-1}(-\infty,a]$.
- $\mathcal{Q}$: Second point cloud.
- $d_H$: Hausdorff distance.
- $d_{GH}$: Gromov–Hausdorff distance.

Theorem (tame continuous $f,g$):

$$
d_B\big(\mathrm{Dgm}_k(f),\mathrm{Dgm}_k(g)\big)\le\|f-g\|_\infty\quad\forall k
$$

- Tame: finitely many homological critical values & finite-dim homology of every sublevel set.

Point-cloud corollaries (via the later interleaving form of the theorem):
- Čech, radius scale: $d_B\le d_H(\mathcal{P},\mathcal{Q})$ for $\mathcal{P},\mathcal{Q}\subset\mathbb{R}^n$.
- Rips, diameter scale ($\mathrm{VR}_\varepsilon$ on this page): $d_B\le2\,d_{GH}(\mathcal{P},\mathcal{Q})$.
- Rips, radius scale (diameter $\le2\varepsilon$): $d_B\le d_{GH}(\mathcal{P},\mathcal{Q})$ (Chazal et al., 2009).
- Each bound is tight: its constant cannot be improved in general.
```

```{tip} Derivation
:class: dropdown
*Why $d_B\le d_H$ for Čech?*

1. Distance function: $d_\mathcal{P}(\mathbf{x})=\min_{\mathbf{p}\in\mathcal{P}}\|\mathbf{x}-\mathbf{p}\|$.
2. Sublevel set $d_\mathcal{P}^{-1}(-\infty,r]=\bigcup_iB(\mathbf{x}_i,r)\simeq\check{C}_r(\mathcal{P})$ (nerve theorem).
3. For any $\mathbf{x}$: nearest $\mathbf{p}\in\mathcal{P}$ has a point of $\mathcal{Q}$ within $d_H$ → $d_\mathcal{Q}(\mathbf{x})\le d_\mathcal{P}(\mathbf{x})+d_H$. Symmetric.
4. → $\|d_\mathcal{P}-d_\mathcal{Q}\|_\infty\le d_H$ → stability theorem → $d_B\le d_H$.

*Why $d_B\le2\,d_{GH}$ for diameter-scale Rips?*

1. $d_{GH}(\mathcal{P},\mathcal{Q})=\eta$ → correspondence $\mathcal{C}\subseteq\mathcal{P}\times\mathcal{Q}$ w/ distortion $\sup|d(\mathbf{p},\mathbf{p}')-d(\mathbf{q},\mathbf{q}')|\le2\eta$.
2. $\sigma\in\mathrm{VR}_\varepsilon(\mathcal{P})$ → replace each vertex by a partner in $\mathcal{Q}$ → pairwise distances $\le\varepsilon+2\eta$ → simplex of $\mathrm{VR}_{\varepsilon+2\eta}(\mathcal{Q})$.
3. Symmetric → the 2 filtrations are $2\eta$-interleaved → $d_B\le2\eta$.
```

```{attention} Q&A
:class: dropdown
*Is persistent homology robust to outliers?*

- ❌. One far outlier → large $d_H$ → bound is useless. One outlier between clusters can merge them early.
- Fix: density-aware filtrations, e.g., distance to a measure, or subsample + denoise first.

*Why the sup-norm & not $L_2$?*

- A tall, narrow spike in $f$ has small $L_2$ norm but creates a long bar.
- Sup-norm is the norm that controls sublevel sets.

*Does a small $d_B$ imply similar inputs?*

- ❌. Diagram map is not injective: very different spaces can share a diagram.
- Stability = Lipschitz in one direction only.
```

&nbsp;

### Mapper
- **What**: Graph summary of data: nerve of clusters inside overlapping slices of a filter function. {cite:p}`singh2007topological`
- **Why**: Barcodes count holes, but give no picture.
    - Need an interpretable skeleton of high-dim data: branches, flares, loops.
- **How**:
    1. Select a filter (lens) $f:\mathcal{P}\to\mathbb{R}$, e.g., density, PCA coordinate, eccentricity, model output.
    2. Cover the range of $f$ w/ overlapping intervals (#intervals, overlap %).
    3. Cluster each preimage $f^{-1}(I)$ separately (e.g., single linkage).
    4. Node = cluster. Edge = 2 clusters that share a point.

```{note} Math
:class: dropdown
Notations:
- $f:\mathcal{P}\to\mathbb{R}$: Filter function.
- $\{I_t\}$: Overlapping intervals that cover $f(\mathcal{P})$.
- $V_{t,s}$: $s$-th cluster of $f^{-1}(I_t)$.

Process:
1. Pullback cover: $\{f^{-1}(I_t)\}$.
2. Refine: split each $f^{-1}(I_t)$ into clusters $V_{t,s}$.
3. Mapper graph = nerve $N(\{V_{t,s}\})$ (1-skeleton; higher simplices for multi-overlaps).

Properties:
- Continuous analog: **Reeb graph** $X/\!\sim$, $x\sim y$ iff same component of the same level set of $f$.
- Mapper ≈ discretized Reeb graph; converges as the cover gets finer (under conditions).
- Filter $f:\mathcal{P}\to\mathbb{R}^2$ → cover by overlapping rectangles → simplicial complex.
```

````{important} Code
:class: dropdown
```python
import itertools
import numpy as np

class Mapper:
    def __init__(self, n_intervals=4, overlap=0.3, eps=0.5):
        self.n, self.o, self.eps = n_intervals, overlap, eps

    def clusters(self, P, idx):
        ## single linkage cut at eps = connected components of the eps-graph
        D = np.linalg.norm(P[idx][:, None] - P[idx][None], axis=-1)
        left, out = set(range(len(idx))), []
        while left:
            stack, comp = [left.pop()], set()
            while stack:
                a = stack.pop()
                comp.add(int(idx[a]))
                nbrs = {b for b in left if D[a, b] <= self.eps}
                left -= nbrs
                stack += nbrs
            out.append(frozenset(comp))
        return out

    def __call__(self, P, f):
        lo, hi = f.min(), f.max()
        L = (hi - lo) / (1 + (self.n - 1) * (1 - self.o))  ## length so that n overlapping intervals cover [lo, hi]
        nodes = []
        for t in range(self.n):
            a = lo + t * L * (1 - self.o)
            idx = np.where((f >= a) & (f <= a + L + 1e-9))[0]  ## pullback of interval t; 1e-9 = float slack
            nodes += self.clusters(P, idx)
        ## nerve: edge iff 2 clusters share a point
        edges = [(u, v) for u, v in itertools.combinations(range(len(nodes)), 2) if nodes[u] & nodes[v]]
        return nodes, edges

## Example: circle, filter = height
theta = np.linspace(0, 2 * np.pi, 40, endpoint=False)
P = np.stack([np.cos(theta), np.sin(theta)], axis=1)  ## (40, 2)
nodes, edges = Mapper()(P, f=P[:, 1])
print(len(nodes), len(edges))  ## 6 6 -> one cycle: beta_1 = E - V + 1 = 1
```
````

```{attention} Q&A
:class: dropdown
*Pros?*

- Interpretable picture of high-dim data; color nodes by any variable.
- Coordinate-free; works w/ any metric & clusterer.

*Cons?*

- Output depends on filter, #intervals, overlap & clusterer → unstable; no unique answer.
- Exploratory tool, not a statistical test. Check features across param settings.

*Real use?*

- Breast-cancer gene expression: Mapper found a subgroup w/ excellent survival (Nicolau, Levine & Carlsson, PNAS 2011).
- NN analysis: filter = model output / loss / layer PCA → find sub-populations of inputs & failure modes.

*How does it relate to UMAP?*

- Both build a graph from local neighborhoods + overlaps.
- [UMAP](../../ml/unsupervised.md#umap): fuzzy neighbor graph → embedding. Mapper: cover + clustering → summary graph, no embedding.
```

&nbsp;

## Neural Topology
- **What**: Topology of the set of population-activity states of a neural circuit ([neural manifold](repr.md#neural-manifold)).

### Ring Attractor
- **What**: Recurrent circuit whose stable states form a continuous 1-D ring → stores one circular variable. {cite:p}`chaudhuri2019intrinsic`
- **Why**: Brain must hold a circular variable (heading) w/o sensory input, e.g., in darkness.
    - Need a continuum of stable states w/ the topology of the variable: $S^1$.
- **How**:
    1. Neurons ordered by preferred direction. Local excitation + broad inhibition → one activity bump.
    2. Rotation symmetry → bump is stable at every angle → ring of fixed points.
    3. Angular-velocity input → bump moves along the ring → integrates heading.
    4. Evidence (Chaudhuri et al., 2019), mouse head-direction cells:
        - Population activity vectors → persistent homology + nonlinear dim reduction → 1-D ring ($\beta_0=1$, $\beta_1=1$).
        - Same ring in waking & REM sleep; ring is isometric to heading.
        - Heading decoded from the ring alone, w/o behavioral labels.

```{note} Math
:class: dropdown
Notations:
- $N$: #neurons.
- $r_a$: Firing rate of neuron $a$.
- $\theta_a$: Preferred angle of neuron $a$.
- $W_{ab}$: Weight from neuron $b$ to $a$.
- $J_0<0$: Uniform (inhibitory) weight.
- $J_1>0$: Tuned (excitatory) weight.
- $\phi$: Rectifying nonlinearity.
- $I$: Uniform external drive.
- $\tau$: Time constant.

Model (rate network):

$$
\tau\frac{dr_a}{dt}=-r_a+\phi\Big(\sum_{b=1}^N W_{ab}r_b+I\Big),\qquad W_{ab}=\frac{1}{N}\big(J_0+J_1\cos(\theta_a-\theta_b)\big)
$$

- Uniformly spaced $\theta_a$ & $W_{ab}$ depends only on $\theta_a-\theta_b$ → discrete rotation equivariance at finite $N$; continuous in the continuum limit.
- Parameter regime w/ a stable bump (suitable inhibition, drive & nonlinearity) → rotate the bump → ring of fixed points $\cong S^1$ in the continuum limit.
- Perturbation along the ring: no restoring force (neutral). Across the ring: decays (stable).
- Velocity input: add an odd component $\propto\omega\sin(\theta_a-\theta_b)$ ($\omega$: angular velocity) → bump drifts at a rate set by $\omega$.
```

```{attention} Q&A
:class: dropdown
*Why does topology identify the encoded variable?*

- Manifold dim & topology must match the variable: angle → $S^1$, 2-D position mod period → $T^2$.
- Geometry (shape, size) depends on tuning curves & recording. Topology does not.

*Why is label-free discovery important?*

- Decoding usually needs behavior as a label.
- Intrinsic manifold + topology → find & decode the variable from activity alone → also in sleep, when no behavior exists.

*What happens outside waking & REM?*

- Non-REM sleep: activity often leaves the wake ring, incl. lower-activity states inside it.
- → The manifold view reveals sleep dynamics invisible to tuning curves.

*Is there a literal ring?*

- Fly (Drosophila): heading bump moves around the anatomical ring of the ellipsoid body (Seelig & Jayaraman, Nature 2015; Kim et al., Science 2017).
- Mammals: ring exists in activity space, not in anatomy.

*Weaknesses of ring attractors?*

- Exact continuum needs exact symmetry. Heterogeneity → few discrete fixed points → drift.
- Noise → bump diffuses → memory error grows w/ time.

*How does this relate to ML?*

- RNNs trained to integrate angular velocity develop ring-attractor-like dynamics (Cueva et al., ICLR 2020).
- Networks trained on modular arithmetic arrange embeddings on circles → [Grokking Modular Addition](symmetry.md#grokking-modular-addition).

*What does it tell us about AC?*

- Variable encoded by the intrinsic shape of activity, kept w/o input & reactivated in REM → internal model that runs offline.
- Hypothesis, contested: such self-sustained latent manifolds are a substrate of internal experience. Encoding a variable ≠ experiencing it.
```

&nbsp;

### Toroidal Grid-Cell Manifold
- **What**: Joint activity of one grid-cell module lies on a 2-torus. {cite:p}`gardner2022toroidal`
- **Why**: Grid cell fires on a hexagonal lattice → periodic in 2 independent directions.
    - Module state = 2-D position modulo the lattice → $T^2$.
    - Direct test of the continuous attractor network (CAN) model.
- **How**:
    1. Record many hundreds of grid cells simultaneously (rat medial entorhinal cortex).
    2. Group cells into modules (shared spacing & orientation).
    3. Population activity vectors → point cloud → persistent homology.
    4. Barcode: 1 long $H_0$, 2 long $H_1$, 1 long $H_2$ bar → $(1,2,1)$ = torus Betti signature, consistent w/ a torus.
        - ❌ Proof of a torus: $S^1\vee S^1\vee S^2$ also has $(1,2,1)$.
    5. Torus coordinates track the animal's position (mod grid period).
    6. Each cell's position on the torus is kept across environments & from waking to sleep → supports CAN, not feedforward models.

```{note} Math
:class: dropdown
Notations:
- $\mathbf{s}\in\mathbb{R}^2$: Animal position.
- $\mathbf{u}_1,\mathbf{u}_2$: Lattice basis vectors, equal length, $60°$ apart.
- $\Lambda=\{a\mathbf{u}_1+b\mathbf{u}_2:a,b\in\mathbb{Z}\}$: Grid lattice.
- $\mathbf{r}(\mathbf{s})\in\mathbb{R}^N$: Population rate vector of one module.

Definition (periodic code):

$$
\mathbf{r}(\mathbf{s}+\boldsymbol{\lambda})=\mathbf{r}(\mathbf{s})\quad\forall\boldsymbol{\lambda}\in\Lambda
$$

- → $\mathbf{r}$ factors through the quotient $\mathbb{R}^2/\Lambda\cong T^2$.
- Injective on $\mathbb{R}^2/\Lambda$ (cells w/ different phases) → activity manifold $\cong T^2$.
- $T^2=S^1\times S^1$ → $\beta=(1,2,1)$.
- Hexagonal $\Lambda$ changes the flat metric (twisted torus), not the topology.
```

```{attention} Q&A
:class: dropdown
*Why a torus & not a plane?*

- Code repeats every lattice step in 2 directions → opposite edges of a unit cell are glued → torus.
- One module alone cannot tell positions 1 period apart.

*How does the brain code large spaces then?*

- Several modules w/ different periods → joint state on a product of tori → unique over a range ≫ any single period.

*Why persistent homology & not PCA?*

- Torus embedding is nonlinear & needs ≥3 dims → linear projection distorts it.
- Persistent homology: label-free, coordinate-free test of $(1,2,1)$.

*Why does conservation across states matter?*

- CAN: fixed recurrent connectivity → same torus & same cell positions in every state, incl. sleep.
- Feedforward models: pattern inherited from spatial inputs → no reason to keep it w/o input.

*How does this relate to ML?*

- RNNs trained for path integration can develop grid-like units (Cueva & Wei, ICLR 2018; Banino et al., Nature 2018).
- Contested: emergence depends on specific readout & training choices (Schaeffer, Khona & Fiete, NeurIPS 2022).

*What does it tell us about intelligence/AC?*

- Cognitive map = internal coordinate system w/ the correct topology, kept offline → candidate building block of a world model.
- TDA = general tool to find the shape of internal codes in brains & NNs w/o labels.
- Speculative: link from map topology to spatial experience is unproven.
```

&nbsp;
