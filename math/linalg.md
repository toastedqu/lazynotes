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
# Linear Algebra
Study notes from {cite:t}`math`, {cite:t}`mml`, and {cite:t}`linalg`.

Notations:
- $m$: #rows of a rectangular matrix.
- $n$: #cols of a rectangular matrix / dimension of a square matrix.
- $r$: Matrix rank.
- $I_n$: $n\times n$ identity matrix.
- $\mathbf{e}_j$: $j$-th standard basis vector.
- $\langle\mathbf{x},\mathbf{y}\rangle$: Inner product.
- $\lVert\mathbf{x}\rVert_2$: Euclidean norm.
- $\lambda_j$: $j$-th eigenvalue.
- $\sigma_j$: $j$-th singular value, ordered largest → smallest.

Here $m,n$ describe matrix dimensions, overriding the global sample/feature meanings; matrices & vectors are real unless stated otherwise.

&nbsp;

## Matrix

| Property               | Statement                  |
|:---------------------- |:-------------------------- |
| Associativity          | $\forall A\in\mathbb{R}^{m\times n},B\in\mathbb{R}^{n\times p},C\in\mathbb{R}^{p\times q}: (AB)C=A(BC)$ |
| Distributivity         | $\forall A,B\in\mathbb{R}^{m\times n},\ C,D\in\mathbb{R}^{n\times p}:$<br>$(A+B)C=AC+BC$<br>$A(C+D)=AC+AD$ |
| Associativity w/ Scalar | $(\lambda\gamma)A=\lambda(\gamma A)$<br>$\lambda(AB)=(\lambda A)B=A(\lambda B)=(AB)\lambda$<br>$(\lambda A)^T=\lambda A^T$ |
| Distributivity w/ Scalar | $(\lambda+\gamma)A=\lambda A+\gamma A$<br>$\lambda(A+B)=\lambda A+\lambda B$ |
| Multiplication with Identity | $\forall A\in\mathbb{R}^{m\times n}: I_mA=AI_n=A$ |
| Inverse of Product | $(AB)^{-1}=B^{-1}A^{-1}$ |
| Transpose of Transpose | $(A^T)^T=A$ |
| Transpose of Product | $(AB)^T=B^TA^T$ |
| Transpose of Sum | $(A+B)^T=A^T+B^T$ |
| Trace | $\text{Tr}(A)=\sum_{i=1}^na_{ii}$ |
| Symmetric Matrix | $A\in\mathbb{R}^{n\times n}$ is symmetric if $A=A^T$. |
| Sum of Symmetric | $A,B\in\mathbb{R}^{n\times n}$ are symmetric → $A+B$ is symmetric. |
| Diagonal Matrix | $D\in\mathbb{R}^{n\times n}$ is a diagonal matrix if $\forall i\ne j: d_{ij}=0$. |
| Scalar Matrix | A diagonal matrix $S\in\mathbb{R}^{n\times n}$ is a scalar matrix if $\forall i\in\{1,\dots,n\}: s_{ii}=c\in\mathbb{R}$. |
| Upper Triangular Matrix | $U=[u_{ij}]$, where $\forall i>j: u_{ij}=0$. |
| Lower Triangular Matrix | $L=[l_{ij}]$, where $\forall i<j: l_{ij}=0$. |

&nbsp;

### Inverse Matrix
- **What**: For a square matrix $A\in\mathbb{R}^{n\times n}$, if $\exists A^{-1}\in\mathbb{R}^{n\times n}: AA^{-1}=A^{-1}A=I_n$, then $A^{-1}$ is the inverse matrix.
    - $\exists A^{-1}$: $A$ is invertible/nonsingular.
    - $\nexists A^{-1}$: $A$ is noninvertible/singular.
- **Why**: The cleanest way to **UNDO** a linear transform.
- **How**:
    - $n=2$:
    $$
    A^{-1}=\frac{1}{a_{11}a_{22}-a_{12}a_{21}}\begin{bmatrix*}
    a_{22} & -a_{12} \\
    -a_{21} & a_{11}
    \end{bmatrix*}
    $$
    - $n>2$:
    $$
    A^{-1}=\frac{1}{\text{det}(A)}\text{adj}(A)
    $$
        - **Minor Matrix** $M_{ij}$: The $(n-1)\times(n-1)$ matrix obtained by deleting row $i$ and col $j$ from $A$.
        - **Cofactor Matrix** $C$: $C_{ij}=(-1)^{i+j}\text{det}(M_{ij})$.
        - **Adjugate Matrix**: $\text{adj}(A)=C^T$.
        - **Determinant**: Pick any row $i$, $\text{det}(A)=\sum_{j=1}^na_{ij}C_{ij}$.

&nbsp;

### Orthogonal Matrix
- **What**: $A$ if $A^T=A^{-1}$.
- **Why**: Orthogonal transformation changes coordinates **w/o changing geometry**.
	- For any vectors $\mathbf{x},\mathbf{y}$ in a vector space $V$:
		- **Length** preserved: $\lVert A\mathbf{x} \rVert=\lVert\mathbf{x}\rVert$
		- **Angle** preserved: $(A\mathbf{x})\cdot(A\mathbf{y})=\mathbf{x}\cdot\mathbf{y}$
- **How**: Properties:
	- $A^T$ and $A^{-1}$ are orthogonal.
	- $\text{det}(A)=\pm 1$.
	- Rotation Matrix (i.e., rotation of a coordinate system) is orthogonal.

&nbsp;

### Systems of Linear Equations
- **What**:
	- General:
$$\begin{align*}
a_{11}x_1+\cdots+a_{1n}x_n&=b_1 \\
&\vdots \\
a_{m1}x_1+\cdots+a_{mn}x_n&=b_m
\end{align*}$$
	- Matrix:
$$
A\mathbf{x}=\mathbf{b}
$$
- **Why**: Many world probs can be reduced to "**find some numbers that satisfy a set of linear constraints**" (esp. optimization).
- **How**:
	- Defs:
		- **Particular solution**: A specific solution to $A\mathbf{x}=\mathbf{b}$. (not necessarily unique)
		- **General solution**: Particular solution + All solutions to $A\mathbf{x}=\mathbf{0}$. (not necessarily unique)
		- **Augmented matrix**: $[A|\mathbf{b}]$.
		- **Row Echelon Form** (REF): A matrix where
			- All 0-only rows are at the bottom.
			- Each row's **pivot** (i.e., first non-0 number) is to the right of the row above it.
		- **Reduced REF**: A REF where
			- Every pivot is 1.
			- Every pivot is the ONLY non-0 entry in its column.
		- **Basic variable**: The vars $x_i$ corresponding to the cols of pivots.
		- **Free variable**: Other vars.
	- **Gaussian Elimination** (= row operations + back substitution):
		1. Get augmented matrix.
		2. Do row operations to get REF.
			- Row operations:
				- Swap: $R_i\leftrightarrow R_j$.
				- Scale: $R_i\leftarrow kR_i, k\ne 0$.
				- Replace: $R_i\leftarrow R_i+kR_j, k\ne 0$.
			- Get REF: For each col (left → right):
				1. Choose a pivot in curr col.
				2. If top = 0, swap pivot up.
				3. Use pivot row to eliminate (i.e., make 0) ALL entries below the pivot.
					- $R_\text{below}\leftarrow R_\text{below}-kR_\text{pivot}$
		3. For each row (bottom → top):
			1. Solve curr row.
			2. Plug into next row.
		4. Outcomes:
			- No solution: An impossible-to-solve row (e.g., $[0\ 0\ 0|1]$).
			- Unique solution: #pivots = #vars (i.e., NO free var).
			- Infinite solutions: #pivots < #vars (i.e., YES free var).

&nbsp;

## Vector Space
- **What**: $V:=(\mathcal{V},+,\cdot)$: A group where we can form **linear combinations**, with 2 operations:
$$\begin{align*}
+: \mathcal{V}\times\mathcal{V}\rightarrow\mathcal{V} \\
\cdot: \mathbb{R}\times\mathcal{V}\rightarrow\mathcal{V}
\end{align*}$$
The following properties hold:

| Property               | Statement                  |
|:---------------------- |:-------------------------- |
| Commutativity          | $\forall\mathbf{u},\mathbf{v}\in\mathcal{V}: \mathbf{u}+\mathbf{v}=\mathbf{v}+\mathbf{u}$ |
| Associativity          | $\forall\mathbf{u},\mathbf{v},\mathbf{w}\in\mathcal{V},\forall a,b\in\mathbb{R}:$<br>$(\mathbf{u}+\mathbf{v})+\mathbf{w}=\mathbf{u}+(\mathbf{v}+\mathbf{w})$<br>$(ab)\mathbf{v}=a(b\mathbf{v})$ |
| Distributivity         | $\forall\mathbf{u},\mathbf{v}\in\mathcal{V}, \forall a,b\in\mathbb{R}:$<br>$a(\mathbf{u}+\mathbf{v})=a\mathbf{u}+a\mathbf{v}$<br>$(a+b)\mathbf{v}=a\mathbf{v}+b\mathbf{v}$ |
| Additive Identity      | $\forall\mathbf{v}\in\mathcal{V}\ \exists\mathbf{0}\in\mathcal{V}: \mathbf{v}+\mathbf{0}=\mathbf{v}$ |
| Additive Inverse       | $\forall\mathbf{v}\in\mathcal{V}\ \exists\mathbf{u}\in\mathcal{V}: \mathbf{v}+\mathbf{u}=\mathbf{0}$ |
| Multiplicative identity | $\forall\mathbf{v}\in\mathcal{V}:1\mathbf{v}=\mathbf{v}$ |

&nbsp;

### Group
- **What**: $G:=(\mathcal{G},\otimes)$: A set $\mathcal{G}$ + An operation $\otimes:\mathcal{G}\times\mathcal{G}\rightarrow\mathcal{G}$ with the following properties:

| Property               | Statement                  |
|:---------------------- |:-------------------------- |
| Closure of $\mathcal{G}$ under $\otimes$ | $\forall u,v\in\mathcal{G}: u\otimes v\in\mathcal{G}$ |
| Associativity          | $\forall u,v,w\in\mathcal{G}: (u\otimes v)\otimes w=u\otimes(v\otimes w)$ |
| Neutral Element      | $\forall v\in\mathcal{G}\ \exists e\in\mathcal{G}: v\otimes e=e\otimes v=v$ |
| Inverse Element      | $\forall v\in\mathcal{G}\ \exists u\in\mathcal{G}: v\otimes u=u\otimes v=e$ |
| (Abelian group ONLY) Commutativity | $\forall u,v\in\mathcal{G}:u\otimes v=v\times u$ |

&nbsp;

### Vector Subspace
- **What**: $U:=(\mathcal{U},+,\cdot)\subseteq V$: A subspace of $V$ with the following properties (i.e., A vector subspace must be closed under **all linear combinations**):

| Property               | Statement                  |
|:---------------------- |:-------------------------- |
| Additive Identity | $\mathbf{0}\in\mathcal{U}$ |
| Closure under Addition | $\forall\mathbf{u},\mathbf{v}\in\mathcal{U}: \mathbf{u}+\mathbf{v}\in\mathcal{U}$ |
| Closure under Scalar Multiplication | $\forall\mathbf{u}\in\mathcal{U},\forall a\in\mathbb{R}:a\mathbf{u}\in\mathcal{U}$ |

- **Why**: *Why do we need a vector subspace if it's basically the same as a vector space?*
	- Vector space = the entire ocean.
	- Vector subspace = a region we can sail in w/o ever falling out.
	- Quite many things we can do when zooming in on this region:
		- Solutions to linear systems
		- Linear mapping/transformation
		- Bases, dimension, spans, rank, ...

&nbsp;

### Linear Combination
- **What**: A vector $\mathbf{v}\in V$ of the form:
$$
\mathbf{v}=\sum_{i=1}^n w_i\mathbf{x}_i,\quad n\in\mathbb{N},w_i\in\mathbb{R},\mathbf{x}_i\in V
$$

### Linear Independence
- **What**: Suppose $\sum_{i=1}^n w_i\mathbf{x}_i=\mathbf{0}$,
	- **Linear independence**: $w_1=\dots=w_n=0$. (i.e., ONLY trivial solution exists)
	- **Linear dependence**: $\exists w_i\ne0$. (i.e., a non-trivial solution exists)
- **Why**: *Who cares?*
	- Linear independence is THE MOST IMPORTANT concept in LinAlg.
	- A set of linearly independent vectors contains vectors with no redundancy.
	- If we remove ANY vector from the set, we lose info.

```{dropdown} Table: Linear Independence - Equivalent Statements
If $A\in\mathbb{R}^{n\times n}$, the statements below are equivalent:

|    | Statement |
|----|-----------|
| 1  | $A\mathbf{x}=\mathbf{0}\Rightarrow \mathbf{x}=\mathbf{0}$ (i.e., ONLY trivial solution) |
| 2  | $det(A)\ne 0$ (i.e., $A$ is invertible) |
| 3  | Reduced row echelon form of $A$ is $I_n$. |
| 4  | $\forall\mathbf{b}\in\mathbb{R}^{n\times 1}: A\mathbf{x}=\mathbf{b}$. |
| 4  | Col vectors of $A$ are linearly independent. |
| 5  | Row vectors of $A$ are linearly independent. |
| 6  | Col vectors of $A$ span $\mathbb{R}^n$. |
| 7  | Row vectors of $A$ span $\mathbb{R}^n$. |
| 8  | Col vectors of $A$ form a basis for $\mathbb{R}^n$. |
| 9  | Row vectors of $A$ form a basis for $\mathbb{R}^n$. |
| 10 | $\text{rank}(A)=n$. |
| 11 | $\text{nullity}(A)=0$. |
| 12 | Eigenvalues $\lambda\ne 0$. |

(TBC)
```

&nbsp;

### Basis
- **What**: A minimal (i.e., linearly independent) generating set of $V$.
	- **Generating Set**: A set of vectors $\mathcal{A}=\{\mathbf{x}_1,\dots,\mathbf{x}_n\}\subseteq\mathcal{V}$, where each vector $\mathbf{v}\in\mathcal{V}$ can be expressed as a linear combination of $\mathbf{x}_1,\dots,\mathbf{x}_n$.
	- **Span**: The set of all linear combinations of vectors in $\mathcal{A}$ ($V=\text{span}(\mathcal{A})$).
	- **Minimal**: There exists NO smaller set $\tilde{\mathcal{A}}\subset\mathcal{A}\subseteq\mathcal{V}$ that spans $V$.
		- Every linearly independent generating set is minimal.
	- Equivalent statements:
		- $\mathcal{A}$ is a basis of $V$.
		- $\mathcal{A}$ is a minimal generating set of $V$.
		- $\mathcal{A}$ is a maximal linearly independent set of vectors in $V$.
	- Remarks:
		- Every vector space has a basis.
		- There is no unique basis of a vector space.
		- All bases have the same # of basis vectors.
- **Why**: Basis = the **coordinate system** of a vector space.
	- Basis = minimal info needed to describe & rebuild the entire space.
	- #Basis vectors = **dimension** of the space (NOTE: NOT #elems in the vector).
- **How**: How to find a basis:
	- Given a set of vectors $\mathcal{A}$,
		1. Concatenate them into matrix $A$.
		2. Gaussian Elimination on $A\mathbf{x}=\mathbf{0}$.
		3. Vars of pivot cols = Basis.

&nbsp;

### Rank
- **What**: #linearly independent cols/rows of a matrix.
- **Why**: Rank = How much **REAL INFO** the matrix has.
	- Rank = Dimension = # of independent degrees of action in a matrix.
	- Everything else (solvability, uniqueness, invertibility, compression, bottlenecks, etc.) is a consequence of the basic actions.

| Property | Statement |
|----------|-----------|
| Col rank = Row rank | $\text{rank}(A)=\text{rank}(A^T)$ |
| Full rank | $\text{rank}(A)=\min(m,n)$ (if not, then *rank deficient*) |
| Invertible | $\forall A\in\mathbb{R}^{n\times n}: A$ is invertible $\Leftrightarrow\text{rank}(A)=n$ |
| Solvable | $\forall A\in\mathbb{R}^{m\times n},\mathbf{b}\in\mathbb{R}^{m}: A\mathbf{x}=\mathbf{b}$ is solvable $\Leftrightarrow\text{rank}(A)=\text{rank}([A\|\mathbf{b}])$ |

&nbsp;

## Linear Transformation
- **What**: For vector spaces $V,W$, $f:V\rightarrow W$ is a linear transformation/mapping if
$$\begin{align*}
\forall \mathbf{x},\mathbf{y}\in V, \forall a\in\mathbb{R}:\quad &f(\mathbf{x}+\mathbf{y})=f(\mathbf{x})+f(\mathbf{y}), \\
&f(a\mathbf{x})=af(\mathbf{x})
\end{align*}$$

```{dropdown} Table: Mapping Types
Suppose $f:V\rightarrow W$ is a mapping,

| Type | Def | Meaning |
|------|-----|---------|
| Injective  | $\forall\mathbf{x},\mathbf{y}\in V:f(\mathbf{x})=f(\mathbf{y})\Rightarrow \mathbf{x}=\mathbf{y}$ | Every input has its own unique output.<br>(i.e., one-to-one / uniqueness) | 
| Surjective | $f(\mathcal{V})=\mathcal{W}$ | Every output is covered.<br>(i.e., full coverage) |
| Bijective  | Injective + Surjective<br>NOTE: A bijective $f$ can be undone by $f^{-1}:W\rightarrow V$ with $f^{-1}\circ f(\mathbf{x})=\mathbf{x}$. | Every output is paired with exactly every input.<br>(i.e., perfect match) |
```

```{dropdown} Table: Linear Mapping Names
| Name | Def | Meaning |
|------|-----|---------|
| Homomorphism | $f:V\rightarrow W$, linear | Structure-preserving map between objects of the same kind. |
| Isomorphism | $f:V\rightarrow W$, linear + bijective | Homomorphism that shows 2 objects are structurally the same.<br>(i.e., relabelling / change of coordinates) |
| Endomorphism | $f:V\rightarrow V$, linear | Homomorphism from an object to itself.<br>(i.e., internal transformation) |
| Automorphism | $f:V\rightarrow V$, linear + bijective | Isomorphism from an object to itself.<br>(i.e., internal shuffling, nothing gets changed) |
| Identity Automorphism | $\text{id}_V:V\rightarrow V,\mathbf{x}\mapsto\mathbf{x}$ | Do nothing. |

English lessons to better understand the cryptic words in the table:
| Part | Meaning |
|------|---------|
| morph | form/shape |
| morphism | shape-preserving mapping |
| homo | same kind |
| iso | equal/same |
| endo | within |
| auto | self |
```

&nbsp;

### Matrix Representation
- **What**: A linear map written in chosen input/output bases.
- **Why**: An abstract map → numbers we can compute with.
- **How**:
    1. Apply the map to each input basis vector.
    2. Express each output in the output basis.
    3. Stack those coordinates as cols.

```{note} Math
:class: dropdown
For $f:\mathbb{R}^n\rightarrow\mathbb{R}^m$ in standard bases:

$$
A=\begin{bmatrix}f(\mathbf{e}_1)&\cdots&f(\mathbf{e}_n)\end{bmatrix},
\qquad f(\mathbf{x})=A\mathbf{x}=\sum_{j=1}^n x_jf(\mathbf{e}_j)
$$

- Col $j$ = where input axis $j$ lands.
- Row $i$ = weights used to form output coordinate $i$.
- Composition: $g(\mathbf{y})=B\mathbf{y}$ → $(g\circ f)(\mathbf{x})=BA\mathbf{x}$.
```

```{attention} Q&A
:class: dropdown
*Is the matrix the map?*

- Map = action on vectors. Matrix = its representation after choosing bases.
- Change the bases → change the matrix, NOT the underlying action.

*Why does multiplication order matter?*

- Rightmost map acts first.
- Rotate → stretch along fixed axes ≠ stretch → rotate.
```

&nbsp;

### Change of Basis
- **What**: New coordinates for the same vector or linear map.
- **Why**: The right coordinates expose structure hidden by the original axes.
- **How**:
    1. New basis vectors → cols of an invertible matrix.
    2. Multiply by the basis matrix → coordinates become the actual vector.
    3. Solve against the basis matrix → actual vector becomes coordinates.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $B\in\mathbb{R}^{n\times n}$: Input basis matrix.
    - $C\in\mathbb{R}^{m\times m}$: Output basis matrix.
    - $[\mathbf{x}]_B$: Coordinates of $\mathbf{x}$ in basis $B$.

$$
\mathbf{x}=B[\mathbf{x}]_B,\qquad
[\mathbf{x}]_B=B^{-1}\mathbf{x},\qquad
[A\mathbf{x}]_C=C^{-1}AB[\mathbf{x}]_B
$$

For an endomorphism using the same new basis on both sides:

$$
\tilde{A}=B^{-1}AB
$$

- $\tilde{A}$: Matrix similar to $A$.
- **Similarity** preserves rank, trace, determinant & eigenvalues.
- Orthogonal $B$ → $B^{-1}=B^T$.
```

```{attention} Q&A
:class: dropdown
*Do arbitrary basis changes preserve lengths & angles of coordinate vectors?*

- ❌Only orthonormal bases preserve the ordinary dot-product formula.
- General basis: $\langle\mathbf{x},\mathbf{y}\rangle=[\mathbf{x}]_B^TB^TB[\mathbf{y}]_B$.
- Geometry unchanged; its coordinate formula changes.

*Why relevant to learned representations?*

- Two hidden representations can differ by an invertible basis change yet encode the same info.
- An adjacent linear decoder can absorb its inverse; coordinate axes alone need not identify unique concepts.
- ❌Arbitrary basis changes do NOT commute with elementwise nonlinearities.
```

&nbsp;

### Affine Transformation
- **What**: Linear transformation + translation.
- **Why**: Linear maps must fix the origin; many useful maps should not.
- **How**: Transform relative positions → shift every result by the same vector.

```{note} Math
:class: dropdown
$$
f(\mathbf{x})=A\mathbf{x}+\mathbf{b},\qquad
f(\mathbf{x})-f(\mathbf{y})=A(\mathbf{x}-\mathbf{y})
$$

- $\mathbf{b}\in\mathbb{R}^m$: Translation.
- $f$ linear $\Leftrightarrow\mathbf{b}=\mathbf{0}$.

**Homogeneous coordinates**:

$$
\begin{bmatrix}f(\mathbf{x})\\1\end{bmatrix}
=
\begin{bmatrix}A&\mathbf{b}\\\mathbf{0}^T&1\end{bmatrix}
\begin{bmatrix}\mathbf{x}\\1\end{bmatrix}
$$
```

```{attention} Q&A
:class: dropdown
*Is a NN's “linear” layer actually linear?*

- With bias → affine; see [Linear](../dl/module.md#linear).
- Stacking affine layers w/o nonlinearities → still one affine map.
- More such layers can change parameterization, NOT the class of representable functions.
```

&nbsp;

### Image & Kernel
- **What**: Reachable outputs & inputs erased by a linear map.
- **Why**: Separate what a representation can express from what it loses.
- **How**:
    - **Image / col space**: All combinations of the matrix's cols.
    - **Kernel / null space**: All inputs mapped to $\mathbf{0}$.

```{note} Math
:class: dropdown
$$
\begin{align*}
\operatorname{im}(A)&=\{A\mathbf{x}:\mathbf{x}\in\mathbb{R}^n\}\subseteq\mathbb{R}^m \\
\ker(A)&=\{\mathbf{x}\in\mathbb{R}^n:A\mathbf{x}=\mathbf{0}\} \\
\dim\operatorname{im}(A)&=r,\qquad \dim\ker(A)=n-r
\end{align*}
$$

- **Rank-nullity**: Surviving degrees of freedom + erased degrees of freedom = input dimension.

For a solvable system:

$$
\{\mathbf{x}:A\mathbf{x}=\mathbf{b}\}=\mathbf{x}_p+\ker(A)
$$

- $\mathbf{x}_p$: Any particular solution.
- Injective $\Leftrightarrow\ker(A)=\{\mathbf{0}\}\Leftrightarrow r=n$.
- Surjective onto $\mathbb{R}^m\Leftrightarrow r=m$.
```

```{note} Example
:class: dropdown
$$
A=\begin{bmatrix}1&1\end{bmatrix},\qquad
A\begin{bmatrix}x_1\\x_2\end{bmatrix}=x_1+x_2
$$

- Image = $\mathbb{R}$; kernel = $\operatorname{span}\{[1,-1]^T\}$.
- $[2,0]^T$ & $[0,2]^T$ become identical → their difference lies in the kernel.
```

```{attention} Q&A
:class: dropdown
*Does a bottleneck always destroy useful info?*

- A linear map with $m<n$ cannot be injective on all of $\mathbb{R}^n$.
- It can still be injective on a data subspace $S$ if $S\cap\ker(A)=\{\mathbf{0}\}$.
- Lost dimensions ≠ lost task-relevant info.
```

&nbsp;

## Geometry

### Inner Product
- **What**: A symmetric positive-definite bilinear measure of alignment.
- **Why**: Vector-space operations alone do NOT define length, angle or orthogonality.
- **How**: Compare corresponding coordinates → aggregate their agreement.

```{note} Math
:class: dropdown
For a real vector space:

$$
\begin{align*}
\langle a\mathbf{x}+b\mathbf{y},\mathbf{z}\rangle
&=a\langle\mathbf{x},\mathbf{z}\rangle+b\langle\mathbf{y},\mathbf{z}\rangle \\
\langle\mathbf{x},\mathbf{y}\rangle&=\langle\mathbf{y},\mathbf{x}\rangle \\
\langle\mathbf{x},\mathbf{x}\rangle&>0\quad\text{if }\mathbf{x}\ne\mathbf{0}
\end{align*}
$$

Euclidean inner product:

$$
\langle\mathbf{x},\mathbf{y}\rangle=\mathbf{x}^T\mathbf{y}
=\sum_{j=1}^n x_jy_j
$$

**Cauchy-Schwarz**:

$$
|\langle\mathbf{x},\mathbf{y}\rangle|
\leq\sqrt{\langle\mathbf{x},\mathbf{x}\rangle}
\sqrt{\langle\mathbf{y},\mathbf{y}\rangle}
$$

- Equality $\Leftrightarrow$ the two vectors are linearly dependent.
- For nonzero vectors, cosine = inner product divided by the two induced lengths.
```

```{attention} Q&A
:class: dropdown
*Dot product vs cosine similarity?*

- Dot product = length × length × cosine → mixes magnitude & direction.
- Cosine removes magnitude; undefined for a zero vector.
- Neither is automatically semantic similarity; meaning depends on the representation.

*What changes over complex vectors?*

- Use conjugate transpose: $\langle\mathbf{x},\mathbf{y}\rangle=\mathbf{x}^*\mathbf{y}$.
- Conjugate-linear in the first argument; conjugate symmetry replaces symmetry.
```

&nbsp;

### Vector Norm
- **What**: A nonnegative measure of vector size.
- **Why**: Quantify distance, error & perturbation size.
- **How**: Aggregate coordinate magnitudes according to the geometry we want.

```{note} Math
:class: dropdown
$$
\begin{align*}
\lVert\mathbf{x}\rVert&\geq0,\qquad
\lVert\mathbf{x}\rVert=0\Leftrightarrow\mathbf{x}=\mathbf{0} \\
\lVert a\mathbf{x}\rVert&=|a|\lVert\mathbf{x}\rVert \\
\lVert\mathbf{x}+\mathbf{y}\rVert&\leq\lVert\mathbf{x}\rVert+\lVert\mathbf{y}\rVert
\end{align*}
$$

$$
\lVert\mathbf{x}\rVert_p=\left(\sum_{j=1}^n|x_j|^p\right)^{1/p},
\qquad 1\leq p<\infty
$$

- $p$: Norm order.
- Distance induced by a norm: $d(\mathbf{x},\mathbf{y})=\lVert\mathbf{x}-\mathbf{y}\rVert$.
- Inner product → norm via $\lVert\mathbf{x}\rVert=\sqrt{\langle\mathbf{x},\mathbf{x}\rangle}$.
```

```{dropdown} Table: Vector Norms
| Norm | Formula | Measures |
|:--|:--|:--|
| $L_1$ | $\sum_j|x_j|$ | Total absolute magnitude |
| $L_2$ | $\sqrt{\sum_jx_j^2}$ | Euclidean length |
| $L_\infty$ | $\max_j|x_j|$ | Largest coordinate magnitude |
| “$L_0$” | $\#\{j:x_j\ne0\}$ | Nonzero count; NOT a norm |
```

```{attention} Q&A
:class: dropdown
*Does every norm come from an inner product?*

- ❌Only norms satisfying the **parallelogram law**:

    $$
    \lVert\mathbf{x}+\mathbf{y}\rVert^2+\lVert\mathbf{x}-\mathbf{y}\rVert^2
    =2\lVert\mathbf{x}\rVert^2+2\lVert\mathbf{y}\rVert^2
    $$

- $L_2$ does; $L_1$ & $L_\infty$ do not in dimensions $\geq2$.

*Why isn't “$L_0$” a norm?*

- Scaling by any nonzero scalar preserves the count → violates absolute homogeneity.
```

&nbsp;

### Orthogonality
- **What**: Zero inner product.
- **Why**: Separate directions w/o overlap; coordinates can be extracted independently.
- **How**:
    - **Orthogonal**: Pairwise perpendicular.
    - **Orthonormal**: Orthogonal + unit length.
    - **Orthogonal complement**: All vectors perpendicular to a subspace.

```{note} Math
:class: dropdown
$$
S^\perp=\{\mathbf{z}:\mathbf{z}^T\mathbf{s}=0\ \forall\mathbf{s}\in S\},
\qquad \mathbb{R}^n=S\oplus S^\perp
$$

- $S$: Subspace of $\mathbb{R}^n$.
- $\oplus$: Direct sum; every vector has a unique decomposition into the two subspaces.

For an orthonormal basis $\{\mathbf{q}_1,\ldots,\mathbf{q}_d\}$ of $S$:

$$
\mathbf{x}\in S\Rightarrow
\mathbf{x}=\sum_{j=1}^d(\mathbf{q}_j^T\mathbf{x})\mathbf{q}_j,
\qquad
\lVert\mathbf{x}\rVert_2^2=\sum_{j=1}^d(\mathbf{q}_j^T\mathbf{x})^2
$$

- $d$: Dimension of $S$.
- **Pythagoras**: $\mathbf{x}\perp\mathbf{y}$ → $\lVert\mathbf{x}+\mathbf{y}\rVert_2^2=\lVert\mathbf{x}\rVert_2^2+\lVert\mathbf{y}\rVert_2^2$.
```

```{dropdown} Table: Four Fundamental Subspaces
For $A\in\mathbb{R}^{m\times n}$:

| Subspace | Ambient space | Dimension | Orthogonal complement |
|:--|:--|:--|:--|
| Row space $\operatorname{im}(A^T)$ | $\mathbb{R}^n$ | $r$ | $\ker(A)$ |
| Null space $\ker(A)$ | $\mathbb{R}^n$ | $n-r$ | $\operatorname{im}(A^T)$ |
| Col space $\operatorname{im}(A)$ | $\mathbb{R}^m$ | $r$ | $\ker(A^T)$ |
| Left null space $\ker(A^T)$ | $\mathbb{R}^m$ | $m-r$ | $\operatorname{im}(A)$ |
```

```{attention} Q&A
:class: dropdown
*Orthogonal vs linearly independent?*

- Nonzero orthogonal vectors → linearly independent.
- Independent vectors need NOT be orthogonal.

*Orthogonal vs statistically independent?*

- Orthogonality is geometric; independence is probabilistic.
- Centered random variables can be orthogonal under $\langle X,Y\rangle=\mathbb{E}[XY]$ yet dependent.
- For jointly Gaussian variables, zero covariance does imply independence.
```

&nbsp;

### Orthogonal Projection
- **What**: Closest vector in a subspace under Euclidean distance.
- **Why**: Keep the component a representation can express; discard the perpendicular remainder.
- **How**:
    1. Extract coordinates along an orthonormal basis of the subspace.
    2. Rebuild using only those basis directions.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $Q\in\mathbb{R}^{n\times d}$: Orthonormal basis cols for a subspace $S$.
    - $d$: Dimension of $S$.

$$
P=QQ^T,\qquad
\hat{\mathbf{x}}=P\mathbf{x}
=\arg\min_{\mathbf{z}\in S}\lVert\mathbf{x}-\mathbf{z}\rVert_2^2
$$

- $P\in\mathbb{R}^{n\times n}$: Orthogonal projector.
- $\hat{\mathbf{x}}$: Projected vector.
- Residual $\mathbf{x}-\hat{\mathbf{x}}\in S^\perp$.
- $P^2=P$; $P^T=P$; $I_n-P$ projects onto $S^\perp$.

For any full-col-rank basis matrix $B$ of $S$:

$$
P=B(B^TB)^{-1}B^T
$$
```

```{tip} Derivation
:class: dropdown
*Why is the perpendicular residual the shortest?*

1. For any $\mathbf{z}\in S$, split $\mathbf{x}-\mathbf{z}=(\mathbf{x}-\hat{\mathbf{x}})+(\hat{\mathbf{x}}-\mathbf{z})$.
2. The first term lies in $S^\perp$; the second in $S$.
3. Pythagoras:

    $$
    \lVert\mathbf{x}-\mathbf{z}\rVert_2^2
    =\lVert\mathbf{x}-\hat{\mathbf{x}}\rVert_2^2
    +\lVert\hat{\mathbf{x}}-\mathbf{z}\rVert_2^2
    $$

4. Minimum attained only at $\mathbf{z}=\hat{\mathbf{x}}$.
```

```{attention} Q&A
:class: dropdown
*Is every idempotent matrix an orthogonal projector?*

- $P^2=P$ → projector, possibly oblique.
- Also $P^T=P$ → orthogonal projector under the Euclidean inner product.

*Projection vs change of basis?*

- Change of basis → same vector, different coordinates, no info loss.
- Projection onto a proper subspace → discards a component.
```

&nbsp;

### Least Squares
- **What**: Smallest squared residual for a possibly inconsistent linear system.
- **Why**: Noisy or overdetermined constraints may have no exact common solution.
- **How**: Project the target onto the col space → find inputs producing that projection.

```{note} Math
:class: dropdown
Objective:

$$
\min_{\mathbf{x}\in\mathbb{R}^n}\frac12\lVert A\mathbf{x}-\mathbf{b}\rVert_2^2
$$

- $\mathbf{b}\in\mathbb{R}^m$: Target.

$$
A^T(A\hat{\mathbf{x}}-\mathbf{b})=\mathbf{0}
\quad\Leftrightarrow\quad
A^TA\hat{\mathbf{x}}=A^T\mathbf{b}
$$

- $\hat{\mathbf{x}}$: Any minimizer.
- **Normal equations**: Residual perpendicular to every col of $A$.
- Full col rank → unique $\hat{\mathbf{x}}=(A^TA)^{-1}A^T\mathbf{b}$.
- Rank deficient → multiple minimizers, but the fitted vector $A\hat{\mathbf{x}}$ is unique.
```

```{attention} Q&A
:class: dropdown
*Does least squares require Gaussian noise?*

- ❌The geometric optimization problem makes no distributional assumption.
- Gaussian noise is needed for its usual Gaussian-likelihood interpretation; see [Linear Regression](../ml/supervised.md#linear-regression).

*Should we compute the displayed inverse?*

- ❌Solve via QR or SVD; forming $A^TA$ worsens conditioning.
- Under rank deficiency, the [pseudoinverse](#pseudoinverse) selects the minimum-length minimizer.
```

&nbsp;

## Spectral Structure

### Eigenvalues & Eigenvectors
- **What**: Nonzero directions a square linear map only scales.
- **Why**: Identify intrinsic modes instead of tracking interacting coordinates.
- **How**: Find a scale that makes the shifted map singular → find its null space.

```{note} Math
:class: dropdown
For $A\in\mathbb{R}^{n\times n}$:

$$
A\mathbf{v}=\lambda\mathbf{v},\qquad \mathbf{v}\ne\mathbf{0}
$$

- $\mathbf{v}$: Right eigenvector.
- $\lambda$: Corresponding eigenvalue.
- **Eigenspace**: $\ker(A-\lambda I_n)$, including $\mathbf{0}$.

$$
\det(A-\lambda I_n)=0,\qquad
\operatorname{Tr}(A)=\sum_{j=1}^n\lambda_j,\qquad
\det(A)=\prod_{j=1}^n\lambda_j
$$

- Eigenvalues counted with algebraic multiplicity; may be complex even when $A$ is real.
- Real eigenvalue: positive → preserve direction; negative → reverse; zero → erase.
- Left eigenvector: $\mathbf{w}^TA=\lambda\mathbf{w}^T$.
```

```{note} Example
:class: dropdown
$$
A=\begin{bmatrix}2&0\\0&-1\end{bmatrix}
$$

- $\mathbf{e}_1$ → doubled; $\mathbf{e}_2$ → reversed.
- $A^t\mathbf{e}_1=2^t\mathbf{e}_1$; $A^t\mathbf{e}_2=(-1)^t\mathbf{e}_2$ for integer $t\geq0$.
```

```{attention} Q&A
:class: dropdown
*Does every real matrix have real eigenvectors?*

- ❌A $90^\circ$ planar rotation has eigenvalues $\pm i$ & no nonzero real eigenvectors.
- Over $\mathbb{C}$, every nonempty square matrix has an eigenvalue.

*Are distinct eigenvectors independent?*

- Eigenvectors belonging to distinct eigenvalues → independent.
- Different vectors from one eigenspace may be dependent.
- Orthogonality needs extra structure, e.g., a real symmetric matrix.

*Should we solve the characteristic polynomial numerically?*

- Useful for tiny symbolic examples; not the standard large-matrix algorithm.
- Numerical eigensolvers use matrix factorizations rather than polynomial root-finding.
```

&nbsp;

### Eigendecomposition
- **What**: A linear map expressed in a basis of eigenvectors.
- **Why**: Coupled coordinates → independently scaled modes.
- **How**: Enter eigen-coordinates → scale each coordinate → return.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $V\in\mathbb{C}^{n\times n}$: Invertible matrix of eigenvector cols.
    - $\Lambda\in\mathbb{C}^{n\times n}$: Diagonal eigenvalue matrix.

$$
AV=V\Lambda
\quad\Leftrightarrow\quad
A=V\Lambda V^{-1},\qquad
A^t=V\Lambda^tV^{-1}
$$

- $t$: Nonnegative integer.
- **Diagonalizable** $\Leftrightarrow$ $n$ linearly independent eigenvectors exist over the chosen field.
- Distinct eigenvalues → sufficient, NOT necessary.
- Repeated eigenvalue → its eigenspace dimension must equal its algebraic multiplicity.
```

```{attention} Q&A
:class: dropdown
*What breaks when there aren't enough eigenvectors?*

- **Defective** matrix → no eigenvector basis.
- Example:

    $$
    A=\begin{bmatrix}1&1\\0&1\end{bmatrix},\qquad
    A^t=\begin{bmatrix}1&t\\0&1\end{bmatrix}
    $$

- Both eigenvalues = 1, yet repeated application can grow linearly.
- Eigenvalues alone miss coupling within a defective mode.

*Does diagonalizable mean numerically easy?*

- Nearly parallel eigenvectors → ill-conditioned $V$.
- A formally correct decomposition can still amplify rounding error.
```

&nbsp;

#### Spectral Theorem
- **What**: Orthogonal eigendecomposition of every real symmetric matrix.
- **Why**: Symmetry removes defective modes & nonorthogonal eigen-coordinate geometry.
- **How**: Choose an orthonormal eigenbasis; repeated eigenspaces can be orthonormalized.

```{note} Math
:class: dropdown
$$
A=A^T\Rightarrow
A=Q\Lambda Q^T=\sum_{j=1}^n\lambda_j\mathbf{q}_j\mathbf{q}_j^T,
\qquad Q^TQ=I_n
$$

- $Q$: Orthogonal matrix of eigenvectors.
- $\mathbf{q}_j$: Col $j$ of $Q$.
- $\Lambda$: Real diagonal matrix of eigenvalues.
- Each $\mathbf{q}_j\mathbf{q}_j^T$ projects onto one eigen-direction.
```

```{attention} Q&A
:class: dropdown
*What survives for complex matrices?*

- **Hermitian** $A=A^*$ → real eigenvalues & unitary eigenbasis.
- More generally, **normal** $A^*A=AA^*$ → unitary eigenbasis, possibly complex eigenvalues.
- $^*$ = conjugate transpose; unitary means $Q^*Q=I_n$.
```

&nbsp;

### Quadratic Form
- **What**: A scalar built from pairwise products of vector coordinates.
- **Why**: Express energy, curvature & direction-dependent penalties.
- **How**: Weight each coordinate interaction → sum.

```{note} Math
:class: dropdown
$$
q(\mathbf{x})=\mathbf{x}^TA\mathbf{x}
=\sum_{i=1}^n\sum_{j=1}^n a_{ij}x_ix_j
=\mathbf{x}^T\frac{A+A^T}{2}\mathbf{x}
$$

- $q$: Quadratic form.
- Only the symmetric part of $A$ contributes.

For symmetric $A=Q\Lambda Q^T$:

$$
q(\mathbf{x})=\sum_{j=1}^n\lambda_jz_j^2,\qquad
\mathbf{z}=Q^T\mathbf{x}
$$

- $\mathbf{z}$: Eigen-coordinates.

**Rayleigh quotient**:

$$
R_A(\mathbf{x})=\frac{\mathbf{x}^TA\mathbf{x}}{\mathbf{x}^T\mathbf{x}},
\qquad
\lambda_{\min}\leq R_A(\mathbf{x})\leq\lambda_{\max},
\qquad \mathbf{x}\ne\mathbf{0}
$$

- $R_A$: Energy per unit squared length.
- $\lambda_{\min}$: Smallest eigenvalue.
- $\lambda_{\max}$: Largest eigenvalue.
- Bounds attained in the corresponding eigenspaces.
```

```{attention} Q&A
:class: dropdown
*Why do eigenvectors solve so many constrained optimization problems?*

- On the unit sphere, a symmetric quadratic form is a weighted average of eigenvalues.
- Maximizing/minimizing it selects the largest/smallest eigenvalue's eigenspace.
- [PCA](../ml/unsupervised.md#pca) applies this to variance.
```

&nbsp;

### Positive Definite Matrix
- **What**: Symmetric matrix with positive energy in every nonzero direction.
- **Why**: Distances & convex curvature must not assign negative squared size.
- **How**: Inspect the quadratic form, or equivalently its eigenvalues.

```{dropdown} Table: Definiteness
For real symmetric $A$ & every $\mathbf{x}\ne\mathbf{0}$:

| Type | Quadratic form | Eigenvalues |
|:--|:--|:--|
| Positive definite (PD), $A\succ0$ | $\mathbf{x}^TA\mathbf{x}>0$ | All $>0$ |
| Positive semidefinite (PSD), $A\succeq0$ | $\mathbf{x}^TA\mathbf{x}\geq0$ | All $\geq0$ |
| Negative definite | $\mathbf{x}^TA\mathbf{x}<0$ | All $<0$ |
| Negative semidefinite | $\mathbf{x}^TA\mathbf{x}\leq0$ | All $\leq0$ |
| Indefinite | Positive & negative values for different directions | Both positive & negative |
```

```{attention} Q&A
:class: dropdown
*Positive entries vs positive definite?*

- Unrelated: $\begin{bmatrix}1&2\\2&1\end{bmatrix}$ has positive entries but eigenvalues $3,-1$.
- PD matrices may have negative off-diagonal entries.

*PD vs PSD?*

- PD → invertible, strictly convex quadratic, genuine inner product $\mathbf{x}^TA\mathbf{y}$.
- Singular PSD → flat directions; $\sqrt{\mathbf{x}^TA\mathbf{x}}$ is only a seminorm.
- $A\succeq0$ & $\epsilon>0$ → $A+\epsilon I_n\succ0$.

*Why does this matter for optimization?*

- PD Hessian at a stationary point → strict local minimum.
- PSD Hessian at one point alone is NOT sufficient; zero-curvature directions need higher-order analysis.
```

&nbsp;

### Gram Matrix
- **What**: Matrix of pairwise inner products.
- **Why**: Encode geometry through similarities instead of explicit coordinates.
- **How**: Arrange vectors as cols → multiply transpose × original.

```{note} Math
:class: dropdown
For $A=[\mathbf{a}_1,\ldots,\mathbf{a}_n]\in\mathbb{R}^{m\times n}$:

$$
G=A^TA,\qquad G_{ij}=\mathbf{a}_i^T\mathbf{a}_j,\qquad
\mathbf{x}^TG\mathbf{x}=\lVert A\mathbf{x}\rVert_2^2\geq0
$$

- $G\in\mathbb{R}^{n\times n}$: Gram matrix.
- $G$ PSD; PD $\Leftrightarrow$ cols of $A$ are independent.
- $\ker(G)=\ker(A)$; $\operatorname{rank}(G)=r$.
- Every real symmetric PSD matrix admits a Gram factorization.

$$
\lVert\mathbf{a}_i-\mathbf{a}_j\rVert_2^2=G_{ii}+G_{jj}-2G_{ij}
$$
```

```{attention} Q&A
:class: dropdown
*How does this connect to covariance?*

- Centered data matrix $X_c$ with samples as rows → sample covariance $X_c^TX_c/(m-1)$ for $m>1$.
- Covariance = scaled feature Gram matrix → automatically PSD.
- Uncentered Gram matrices describe raw inner products, NOT covariance.

*Can a Gram matrix recover the original coordinates?*

- Not uniquely: $A$ & $QA$ have the same Gram matrix for any orthogonal $Q$.
- In a fixed ambient dimension, equal Gram matrices determine configurations up to an orthogonal transformation.
- A [kernel](../ml/supervised.md#kernel-trick) evaluates inner products in a feature space w/o explicitly constructing those coordinates.
```

&nbsp;

### Singular Values & Vectors
- **What**: Orthogonal input/output direction pairs linked by nonnegative gains.
- **Why**: Rectangular & defective maps still need a geometric description.
- **How**: [SVD](../ml/unsupervised.md#svd) chooses input axes → scales them → places them along output axes.

```{note} Math
:class: dropdown
For each positive singular value:

$$
A\mathbf{v}_j=\sigma_j\mathbf{u}_j,\qquad
A^T\mathbf{u}_j=\sigma_j\mathbf{v}_j
$$

- $\mathbf{v}_j\in\mathbb{R}^n$: Unit right singular vector.
- $\mathbf{u}_j\in\mathbb{R}^m$: Unit left singular vector.

$$
A^TA\mathbf{v}_j=\sigma_j^2\mathbf{v}_j,\qquad
AA^T\mathbf{u}_j=\sigma_j^2\mathbf{u}_j
$$

- Positive right singular vectors span row space.
- Positive left singular vectors span col space.
- Complete the right/left bases with bases of $\ker(A)$ / $\ker(A^T)$.
- Unit sphere → ellipsoid, possibly collapsed; positive semiaxis lengths = positive singular values.
```

```{attention} Q&A
:class: dropdown
*Eigenvalues vs singular values?*

- Eigenvectors compare input & output in the same space; singular vectors may use different spaces.
- Eigenvalues can be negative/complex; singular values are real & nonnegative.
- Symmetric $A$ → singular values are absolute eigenvalues, reordered.
- General $A$ → eigenvalue magnitudes need NOT equal singular values.

*Is low numerical rank the same as low exact rank?*

- Exact rank counts nonzero singular values.
- Numerical rank counts singular values above a chosen tolerance.
- A tiny singular value represents an almost-erased direction, even if exact rank is full.
```

&nbsp;

### Low-Rank Factorization
- **What**: A linear map routed through a smaller latent space.
- **Why**: Compress redundant transformations & expose their effective degrees of freedom.
- **How**: Encode input into latent coordinates → decode into output coordinates.

```{note} Math
:class: dropdown
$$
A=BC,\qquad A\mathbf{x}=B(C\mathbf{x})
$$

- $B\in\mathbb{R}^{m\times k}$: Decoder.
- $C\in\mathbb{R}^{k\times n}$: Encoder.
- $k$: Latent width.
- $\operatorname{rank}(BC)\leq k$; exact factorization exists whenever $k\geq r$.
- Factor storage = $k(m+n)$ entries vs $mn$; savings only when $k(m+n)<mn$.
- For $k<r$, [truncated SVD](../ml/unsupervised.md#svd) gives optimal approximation under Frobenius/spectral error.
```

```{attention} Q&A
:class: dropdown
*Are the latent coordinates uniquely determined?*

- ❌For any invertible $R\in\mathbb{R}^{k\times k}$: $BC=(BR)(R^{-1}C)$.
- Different factors can implement exactly the same map.
- Rank-$k$ matrices have $k(m+n-k)$ local degrees of freedom, not $k(m+n)$.

*Does the best low-rank approximation preserve the most useful concepts?*

- It preserves matrix energy under the chosen norm.
- ❌Small reconstruction error does NOT guarantee preservation of task-relevant or interpretable features.
```

&nbsp;

### Pseudoinverse
- **What**: Canonical inverse on surviving directions, zero on unreachable directions.
- **Why**: Rectangular or singular systems lack an ordinary inverse.
- **How**:
    1. Resolve the target along left singular vectors.
    2. Divide by each positive singular value.
    3. Rebuild along right singular vectors; set null-space coordinates to zero.

```{note} Math
:class: dropdown
$$
A^+=\sum_{j=1}^r\sigma_j^{-1}\mathbf{v}_j\mathbf{u}_j^T
\in\mathbb{R}^{n\times m}
$$

- $A^+$: Moore-Penrose pseudoinverse.
- $\mathbf{u}_j$: Positive left singular vectors.
- $\mathbf{v}_j$: Corresponding right singular vectors.
- Zero matrix → zero pseudoinverse.

$$
AA^+A=A,\quad A^+AA^+=A^+,\quad
(AA^+)^T=AA^+,\quad(A^+A)^T=A^+A
$$

- These four conditions uniquely characterize $A^+$.
- $AA^+$ projects onto col space; $A^+A$ projects onto row space.
- $\hat{\mathbf{x}}=A^+\mathbf{b}$ = minimum-$L_2$-norm least-squares solution.
- All least-squares solutions: $A^+\mathbf{b}+\mathbf{z}$ with $\mathbf{z}\in\ker(A)$.
```

```{attention} Q&A
:class: dropdown
*When does it reduce to familiar formulas?*

- Invertible square $A$: $A^+=A^{-1}$.
- Full col rank: $A^+=(A^TA)^{-1}A^T$.
- Full row rank: $A^+=A^T(AA^T)^{-1}$.

*Why can a pseudoinverse amplify noise?*

- Small positive $\sigma_j$ → large reciprocal gain.
- Truncation treats tiny singular values as zero → changes the exact problem to stabilize the solution.
- [Ridge](../ml/supervised.md#ridge-regression) instead replaces reciprocal gain $1/\sigma_j$ by $\sigma_j/(\sigma_j^2+\lambda)$ for objective $\lVert A\mathbf{x}-\mathbf{b}\rVert_2^2+\lambda\lVert\mathbf{x}\rVert_2^2$.
- $\lambda>0$: Regularization strength.
```

&nbsp;

## Numerical Linear Algebra

### Matrix Norm
- **What**: A scalar measure of matrix size or amplification.
- **Why**: Quantify how strongly a map can enlarge errors or signals.
- **How**:
    - **Operator norm**: Largest output length per unit input length.
    - **Frobenius norm**: Euclidean length of all entries treated as one vector.

```{note} Math
:class: dropdown
$$
\begin{align*}
\lVert A\rVert_2
&=\max_{\lVert\mathbf{x}\rVert_2=1}\lVert A\mathbf{x}\rVert_2=\sigma_1 \\
\lVert A\rVert_F
&=\sqrt{\sum_{i,j}a_{ij}^2}
=\sqrt{\operatorname{Tr}(A^TA)}
=\sqrt{\sum_{j=1}^r\sigma_j^2}
\end{align*}
$$

- $\lVert A\rVert_2$: Spectral norm.
- $\lVert A\rVert_F$: Frobenius norm.
- $\lVert A\mathbf{x}\rVert_2\leq\lVert A\rVert_2\lVert\mathbf{x}\rVert_2$.
- $\lVert AB\rVert_2\leq\lVert A\rVert_2\lVert B\rVert_2$.
- For $r>0$: $\lVert A\rVert_2\leq\lVert A\rVert_F\leq\sqrt{r}\lVert A\rVert_2$.
- Induced $L_1$ norm = largest absolute col sum; induced $L_\infty$ norm = largest absolute row sum.
```

```{attention} Q&A
:class: dropdown
*Does a small spectral radius imply small one-step amplification?*

- ❌Spectral radius measures eigenvalue magnitudes; spectral norm measures worst-case gain.
- $A=\begin{bmatrix}0&10\\0&0\end{bmatrix}$: eigenvalues all zero, but $\lVert A\rVert_2=10$.
- $A^2=0$: large transient response can coexist with eventual disappearance.

*Why care about products of matrix norms?*

- Composed linear maps have gain bounded by the product of their norms.
- Jacobians obey the same rule → a route to understanding gradient amplification & attenuation.
- The bound can be loose: the strongest directions of consecutive maps need not align.
```

&nbsp;

### Condition Number
- **What**: Worst-case relative sensitivity of a problem to small perturbations.
- **Why**: Tiny input or rounding errors can become large solution errors.
- **How**: Compare strongest & weakest gains of the map being inverted.

```{note} Math
:class: dropdown
For invertible $A\in\mathbb{R}^{n\times n}$:

$$
\kappa_2(A)=\lVert A\rVert_2\lVert A^{-1}\rVert_2
=\frac{\sigma_1}{\sigma_n}\geq1
$$

- $\kappa_2$: Spectral condition number for inversion.

For $A\mathbf{x}=\mathbf{b}\ne\mathbf{0}$ with $A$ fixed:

$$
A\,\delta\mathbf{x}=\delta\mathbf{b}
\quad\Rightarrow\quad
\frac{\lVert\delta\mathbf{x}\rVert_2}{\lVert\mathbf{x}\rVert_2}
\leq\kappa_2(A)
\frac{\lVert\delta\mathbf{b}\rVert_2}{\lVert\mathbf{b}\rVert_2}
$$

- $\delta\mathbf{b}$: Target perturbation.
- $\delta\mathbf{x}$: Resulting solution perturbation.
- Full-col-rank rectangular $A$: $\kappa_2(A^TA)=(\sigma_1/\sigma_n)^2$.
- Singular square $A$: inversion condition number is infinite.
```

```{attention} Q&A
:class: dropdown
*Conditioning vs numerical stability?*

- **Conditioning**: Property of the problem.
- **Stability**: Property of the algorithm.
- A backward-stable algorithm returns the exact answer to a nearby problem; an ill-conditioned problem can still yield large forward error.

*Does a tiny determinant imply bad conditioning?*

- ❌$A=\epsilon I_n$ has determinant $\epsilon^n$ but $\kappa_2(A)=1$ for $\epsilon\ne0$.
- Absolute scale ≠ ratio of gains.

*What helps?*

- Avoid explicit inverses & unnecessary Gram matrices.
- Rescale/precondition to improve the system presented to the solver.
- Regularization limits amplification by changing the problem, not by recovering missing info.
```

&nbsp;

### QR
- **Name**: Orthogonal–Upper-Triangular Factorization ($Q$ & $R$ name the factors).
- **What**: Orthonormal col basis × upper-triangular coordinates.
- **Why**: Solve least squares w/o forming $A^TA$.
- **How**:
    1. Extract an orthonormal basis for the col space.
    2. Record each original col's coordinates in that basis.
    3. Rotate the target into those coordinates → back-substitute.

```{note} Math
:class: dropdown
For full-col-rank $A\in\mathbb{R}^{m\times n}$, $m\geq n$:

$$
A=QR,\qquad Q^TQ=I_n
$$

- $Q\in\mathbb{R}^{m\times n}$: Orthonormal cols.
- $R\in\mathbb{R}^{n\times n}$: Invertible upper-triangular matrix.

$$
\lVert A\mathbf{x}-\mathbf{b}\rVert_2^2
=\lVert R\mathbf{x}-Q^T\mathbf{b}\rVert_2^2
+\lVert(I_m-QQ^T)\mathbf{b}\rVert_2^2
$$

- $\mathbf{b}\in\mathbb{R}^m$: Target.
- Minimizer solves $R\hat{\mathbf{x}}=Q^T\mathbf{b}$.
```

```{attention} Q&A
:class: dropdown
*Is reduced $Q$ an orthogonal matrix?*

- Its cols are orthonormal: $Q^TQ=I_n$.
- If $m>n$, $QQ^T\ne I_m$; it is a projector, not an identity.

*How is QR computed reliably?*

- **Householder reflections** eliminate subdiagonal entries using orthogonal transforms.
- Reflection: $H=I-2\mathbf{u}\mathbf{u}^T$ for unit $\mathbf{u}$.
- Rank-deficient inputs need pivoting/rank handling or SVD; an invertible $R$ cannot be assumed.
```

&nbsp;

#### Gram-Schmidt
- **What**: Successive removal of already-explained vector components.
- **Why**: Turn independent directions into orthonormal coordinates.
- **How**: Remove previous projections → normalize what remains.

```{note} Math
:class: dropdown
Process:

1. For each col $\mathbf{a}_j$ of a full-col-rank $A$, initialize $\mathbf{v}\leftarrow\mathbf{a}_j$.
2. For each previously computed $\mathbf{q}_i$, $i<j$:

    $$
    r_{ij}\leftarrow\mathbf{q}_i^T\mathbf{v},\qquad
    \mathbf{v}\leftarrow\mathbf{v}-r_{ij}\mathbf{q}_i
    $$

3. Normalize:

    $$
    r_{jj}\leftarrow\lVert\mathbf{v}\rVert_2,\qquad
    \mathbf{q}_j\leftarrow\mathbf{v}/r_{jj}
    $$

- $\mathbf{v}$: Curr residual direction.
- $\mathbf{q}_j$: New orthonormal basis vector.
- $r_{ij}$: Entry of the upper-triangular factor $R$; entries below the diagonal are zero.
```

```{attention} Q&A
:class: dropdown
*Classical vs modified Gram-Schmidt?*

- Above = **modified**: each projection uses the curr residual.
- Classical: compute all projections from the original col before subtraction.
- Same in exact arithmetic; modified usually loses less orthogonality in floating point.
- Near-dependent cols can still require reorthogonalization; zero residual means no new basis direction.
```

&nbsp;

### Cholesky Factorization
- **What**: Positive-definite matrix = lower-triangular factor × its transpose.
- **Why**: Reuse symmetry & positivity when solving systems or representing covariance.
- **How**: Resolve one diagonal entry → remove its contribution from the remaining block → repeat.

```{note} Math
:class: dropdown
For $A=A^T\succ0$:

$$
A=LL^T
$$

- $L\in\mathbb{R}^{n\times n}$: Unique lower-triangular factor with positive diagonal.

Process:

$$
L_{jj}=\sqrt{a_{jj}-\sum_{k<j}L_{jk}^2},\qquad
L_{ij}=\frac{a_{ij}-\sum_{k<j}L_{ik}L_{jk}}{L_{jj}}\quad(i>j)
$$

- Solve $A\mathbf{x}=\mathbf{b}$ via $L\mathbf{z}=\mathbf{b}$ then $L^T\mathbf{x}=\mathbf{z}$.
- $\mathbf{z}$: Intermediate solve result.
- $\log\det(A)=2\sum_j\log L_{jj}$.
- If $\operatorname{Cov}(\boldsymbol{\epsilon})=I_n$, then $\operatorname{Cov}(L\boldsymbol{\epsilon})=A$.
- $\boldsymbol{\epsilon}$: Random vector with identity covariance.
```

```{attention} Q&A
:class: dropdown
*Will it work for every PSD matrix?*

- A singular PSD matrix has no Cholesky factor with strictly positive diagonal.
- The unpivoted recurrence can encounter division by zero.
- Adding $\epsilon I_n$ with $\epsilon>0$ makes PSD input PD, but changes the matrix.

*Cholesky factor vs symmetric square root?*

- Cholesky: $A=LL^T$, usually $L^2\ne A$.
- Symmetric PSD square root: $A^{1/2}=Q\Lambda^{1/2}Q^T$, with $(A^{1/2})^2=A$.
```

&nbsp;

### Power Iteration
- **What**: Repeated matrix application to isolate a dominant eigen-direction.
- **Why**: Need a leading mode, not an entire eigendecomposition.
- **How**: Multiply → normalize → repeat; the largest-magnitude mode eventually dominates.

```{note} Math
:class: dropdown
For symmetric $A$:

$$
\mathbf{v}_{t+1}=\frac{A\mathbf{v}_t}{\lVert A\mathbf{v}_t\rVert_2},
\qquad
\hat{\lambda}_t=\mathbf{v}_t^TA\mathbf{v}_t
$$

- $\mathbf{v}_t$: Unit iterate.
- $\hat{\lambda}_t$: Rayleigh-quotient eigenvalue estimate.
- $t$: Iteration index.
- Assume a unique largest-magnitude eigenvalue & nonzero initial component in its eigenspace.
- Eigenvector direction converges; negative dominant eigenvalue → sign alternates.
- Asymptotic directional error shrinks by the ratio of second-largest to largest eigenvalue magnitude per step.
```

```{attention} Q&A
:class: dropdown
*What breaks?*

- Equal largest magnitudes → no guaranteed single limiting direction.
- Tiny spectral gap → slow convergence.
- Zero initial overlap → dominant mode never appears in exact arithmetic.
- $A\mathbf{v}_t=\mathbf{0}$ → normalization undefined.

*How do we find a leading singular vector of rectangular $A$?*

- Apply iteration to $A^TA$, but compute each product as $A^T(A\mathbf{v})$.
- No need to materialize the Gram matrix.
- This is an iterative approximation, not a replacement for a robust full SVD.
```

&nbsp;

### Schur Complement
- **What**: Remaining block after eliminating another block of a linear system.
- **Why**: Reason about part of a coupled system w/o solving every variable simultaneously.
- **How**: Solve one block symbolically → substitute into the other.

```{note} Math
:class: dropdown
For a block system with invertible $A$:

$$
\begin{bmatrix}A&B\\C&D\end{bmatrix}
\begin{bmatrix}\mathbf{x}\\\mathbf{y}\end{bmatrix}
=\begin{bmatrix}\mathbf{a}\\\mathbf{b}\end{bmatrix}
$$

- $A\in\mathbb{R}^{p\times p}$: Block to eliminate.
- $B\in\mathbb{R}^{p\times q}$: Upper-right coupling block.
- $C\in\mathbb{R}^{q\times p}$: Lower-left coupling block.
- $D\in\mathbb{R}^{q\times q}$: Remaining diagonal block.
- $\mathbf{x}\in\mathbb{R}^p$: Eliminated unknowns.
- $\mathbf{y}\in\mathbb{R}^q$: Retained unknowns.
- $\mathbf{a}\in\mathbb{R}^p$: First target block.
- $\mathbf{b}\in\mathbb{R}^q$: Second target block.
- $p$: Eliminated block dimension.
- $q$: Retained block dimension.

$$
S=D-CA^{-1}B,\qquad
S\mathbf{y}=\mathbf{b}-CA^{-1}\mathbf{a}
$$

- $S$: Schur complement of $A$.
- $\det\begin{bmatrix}A&B\\C&D\end{bmatrix}=\det(A)\det(S)$.
- Symmetric block matrix with $A\succ0$: whole matrix PD $\Leftrightarrow S\succ0$.
```

```{attention} Q&A
:class: dropdown
*Where does this appear in probabilistic reasoning?*

- For jointly Gaussian vectors with covariance blocks $A,B,B^T,D$ & $A\succ0$, conditioning the second vector on the first leaves covariance $D-B^TA^{-1}B$.
- The subtracted term measures uncertainty removed through cross-covariance.
- Marginalizing the first vector instead leaves the covariance block $D$.
```

&nbsp;

### Low-Rank Inverse Update
- **What**: Updated inverse obtained through a small auxiliary system.
- **Why**: A low-rank matrix change should not require solving the entire problem from scratch.
- **How**: Pass the update through the old solver → solve only in the update's latent dimension.

```{note} Math
:class: dropdown
**Woodbury identity**:

$$
(A+UV^T)^{-1}
=A^{-1}-A^{-1}U(I_k+V^TA^{-1}U)^{-1}V^TA^{-1}
$$

- $A\in\mathbb{R}^{n\times n}$: Invertible original matrix.
- $U\in\mathbb{R}^{n\times k}$: Left update factor.
- $V\in\mathbb{R}^{n\times k}$: Right update factor.
- $k$: Update width.
- Requires $I_k+V^TA^{-1}U$ invertible, equivalently $A+UV^T$ invertible.

**Sherman-Morrison**, $k=1$:

$$
(A+\mathbf{u}\mathbf{v}^T)^{-1}
=A^{-1}-\frac{A^{-1}\mathbf{u}\mathbf{v}^TA^{-1}}
{1+\mathbf{v}^TA^{-1}\mathbf{u}}
$$

- $\mathbf{u}\in\mathbb{R}^n$: Left rank-one factor.
- $\mathbf{v}\in\mathbb{R}^n$: Right rank-one factor.
- Denominator must be nonzero.
```

```{attention} Q&A
:class: dropdown
*Must we store $A^{-1}$?*

- ❌Reuse a factorization of $A$; expressions like $A^{-1}U$ mean solve $AZ=U$.
- $Z$: Solution matrix.
- A nearly singular auxiliary system still creates numerical sensitivity.
```

&nbsp;

## Matrix Calculus

### Differential & Gradient
- **What**: First-order scalar change & its coordinate representation.
- **Why**: Differentiate matrix computations w/o expanding every coordinate.
- **How**: Express the output perturbation as an inner product with the input perturbation.

```{note} Math
:class: dropdown
For a differentiable scalar function:

$$
df=(\nabla_{\mathbf{x}}f)^T\,d\mathbf{x},\qquad
df=\operatorname{Tr}\big((\nabla_Af)^T\,dA\big)
$$

- $df$: First-order change in $f$.
- $d\mathbf{x}$: Vector perturbation.
- $dA$: Matrix perturbation.
- $\nabla_{\mathbf{x}}f$: Col gradient.
- $\nabla_Af$: Gradient with the same shape as $A$.
- Vector/matrix formulas apply to the respective input type.

$$
d(AB)=(dA)B+A(dB),\qquad
d(A^{-1})=-A^{-1}(dA)A^{-1}
$$

- Inverse differential requires invertible $A$.
- Trace cyclicity: $\operatorname{Tr}(ABC)=\operatorname{Tr}(BCA)=\operatorname{Tr}(CAB)$ for compatible shapes.
- Cyclic permutation ≠ arbitrary reordering.
```

```{dropdown} Table: Gradient Identities
| Function | Gradient | Condition |
|:--|:--|:--|
| $\mathbf{a}^T\mathbf{x}$ | $\nabla_{\mathbf{x}}f=\mathbf{a}$ | Fixed $\mathbf{a}$ |
| $\mathbf{x}^TA\mathbf{x}$ | $\nabla_{\mathbf{x}}f=(A+A^T)\mathbf{x}$ | Fixed square $A$ |
| $\frac12\lVert A\mathbf{x}-\mathbf{b}\rVert_2^2$ | $\nabla_{\mathbf{x}}f=A^T(A\mathbf{x}-\mathbf{b})$ | Fixed $A,\mathbf{b}$ |
| $\frac12\lVert A\rVert_F^2$ | $\nabla_Af=A$ | Any real $A$ |
| $\log\det(A)$ | $\nabla_Af=A^{-T}$ | $A$ PD; ambient gradient |
```

```{attention} Q&A
:class: dropdown
*Why do transposes appear in backprop?*

- Forward perturbations move through a map; gradients must reproduce the same scalar change by an inner product.
- Transpose = the map that transfers that inner product back to the input.
- It is NOT an inverse; backprop does not undo the forward computation.
```

&nbsp;

### Jacobian
- **What**: Matrix of first derivatives for a vector-valued map.
- **Why**: A nonlinear map is approximately linear near a point.
- **How**: Perturb one input coordinate → record the change in every output coordinate.

```{note} Math
:class: dropdown
For differentiable $f:\mathbb{R}^n\rightarrow\mathbb{R}^m$:

$$
J_f(\mathbf{x})=
\left[\frac{\partial f_i}{\partial x_j}\right]\in\mathbb{R}^{m\times n},
\qquad
f(\mathbf{x}+\delta\mathbf{x})
=f(\mathbf{x})+J_f(\mathbf{x})\delta\mathbf{x}
+o(\lVert\delta\mathbf{x}\rVert_2)
$$

- $J_f$: Jacobian; output coordinates index rows.
- $\delta\mathbf{x}$: Input perturbation.
- $o(\lVert\delta\mathbf{x}\rVert_2)$: Remainder negligible relative to perturbation size as it tends to zero.

$$
J_{g\circ f}(\mathbf{x})=J_g(f(\mathbf{x}))J_f(\mathbf{x})
$$

- $g$: Differentiable map applied after $f$.

$$
\nabla_{\mathbf{x}}\mathcal{L}
=J_f(\mathbf{x})^T\nabla_{\mathbf{y}}\mathcal{L},
\qquad \mathbf{y}=f(\mathbf{x})
$$

- $\mathcal{L}$: Scalar loss depending on $\mathbf{x}$ through $\mathbf{y}$.
- **JVP**: Jacobian-vector product $J_f\mathbf{v}$ propagates an input direction.
- **VJP**: Vector-Jacobian product, written $J_f^T\mathbf{u}$ under col-gradient convention, pulls back an output gradient.
- $\mathbf{v}\in\mathbb{R}^n$: Input direction.
- $\mathbf{u}\in\mathbb{R}^m$: Output-space vector.
```

```{attention} Q&A
:class: dropdown
*Why not form the whole Jacobian?*

- Many tasks need only its action on a vector.
- Forward/reverse automatic differentiation compute JVPs/VJPs w/o storing the full matrix.

*Does a full-rank Jacobian prove global invertibility?*

- ❌For a continuously differentiable square map, invertibility at one point gives only a local inverse near that point.
- Global folding or repeated outputs may still occur elsewhere.

*What does its rank mean?*

- #independent output directions accessible to first order at that point.
- A null direction causes zero first-order change, NOT necessarily zero finite change.
```

&nbsp;

### Hessian
- **What**: Matrix of second derivatives of a scalar function.
- **Why**: Grad gives slope; Hessian gives how slope changes with direction.
- **How**: Differentiate the gradient → inspect curvature along candidate directions.

```{note} Math
:class: dropdown
For twice continuously differentiable $f:\mathbb{R}^n\rightarrow\mathbb{R}$:

$$
H_f(\mathbf{x})=
\left[\frac{\partial^2f}{\partial x_i\partial x_j}\right],
\qquad H_f=H_f^T
$$

- $H_f\in\mathbb{R}^{n\times n}$: Hessian.

$$
f(\mathbf{x}+\delta\mathbf{x})
=f(\mathbf{x})+\nabla f(\mathbf{x})^T\delta\mathbf{x}
+\frac12\delta\mathbf{x}^TH_f(\mathbf{x})\delta\mathbf{x}
+o(\lVert\delta\mathbf{x}\rVert_2^2)
$$

- $\delta\mathbf{x}$: Input perturbation.
- $\mathbf{v}^TH_f\mathbf{v}$: Second directional derivative along $\mathbf{v}$.
- $\mathbf{v}\in\mathbb{R}^n$: Direction.
- Hessian eigenvectors = principal curvature directions; eigenvalues = curvatures.
- On an open convex domain, $H_f(\mathbf{x})\succeq0$ everywhere $\Leftrightarrow f$ convex.
```

```{attention} Q&A
:class: dropdown
*What does the Hessian say at a stationary point?*

- PD → strict local min; negative definite → strict local max.
- Indefinite → saddle.
- Semidefinite with zero eigenvalues → inconclusive; e.g., $x^4$ vs $-x^4$ at zero.
- Full optimality conditions: [Optimality Conditions](../ml/optim.md#optimality-conditions).

*Hessian-vector product w/o storing the Hessian?*

- For constant $\mathbf{v}$: $H_f\mathbf{v}=\nabla_{\mathbf{x}}(\nabla f(\mathbf{x})^T\mathbf{v})$.
- Differentiate a directional gradient; [Newton's Method](../ml/optim.md#newtons-method) & related solvers use curvature actions rather than necessarily materializing an inverse.
```

&nbsp;

## Dynamics

### Matrix Powers & Stability
- **What**: Repeated linear state updates & their long-term response to perturbations.
- **Why**: Recurrent computation must retain useful state w/o uncontrolled amplification.
- **How**: Follow each mode through repeated multiplication; distinguish eventual behavior from transient gain.

```{note} Math
:class: dropdown
Model:

$$
\mathbf{x}_{t+1}=A\mathbf{x}_t,\qquad
\mathbf{x}_t=A^t\mathbf{x}_0,\qquad
A\in\mathbb{R}^{n\times n}
$$

- $\mathbf{x}_t$: State at discrete time $t$.
- $\mathbf{x}_0$: Initial state.

$$
\rho(A)=\max_j|\lambda_j|,\qquad
A^t\rightarrow0\Leftrightarrow\rho(A)<1
$$

- $\rho(A)$: Spectral radius.
- Real mode with $|\lambda|<1$ → decays; $|\lambda|>1$ → grows.
- $|\lambda|=1$ → boundary case; defective blocks can grow polynomially.
- All trajectories bounded $\Leftrightarrow$ eigenvalues lie in the closed unit disk & every boundary eigenvalue's eigenspace dimension equals its algebraic multiplicity.
```

```{attention} Q&A
:class: dropdown
*Does asymptotic stability rule out temporary amplification?*

- ❌A nonnormal map can amplify a state before it decays.
- $\rho(A)<1$ controls the limit, not every $\lVert A^t\rVert_2$.
- $\lVert A\rVert_2<1$ is stronger: guarantees contraction at every step.

*Can we apply this directly to a nonlinear recurrent model?*

- At a fixed point, linearize using its Jacobian.
- Jacobian spectral radius $<1$ gives local asymptotic stability for a continuously differentiable map.
- Away from a fixed point, different Jacobians multiply; individual eigenvalues do NOT determine the product's behavior.

*Stable dynamics = memory?*

- Strict contraction eventually forgets initial differences.
- Modes near the unit circle decay slowly; persistent modes can retain state but also retain perturbations.
- Stability alone establishes neither intelligent behavior nor consciousness.
```

&nbsp;

### Matrix Exponential
- **What**: Continuous-time linear evolution operator.
- **Why**: Replace repeated discrete updates with a continuously evolving state.
- **How**: Accumulate all orders of the generator's action.

```{note} Math
:class: dropdown
Model:

$$
\frac{d\mathbf{x}(t)}{dt}=A\mathbf{x}(t),\qquad
\mathbf{x}(t)=e^{tA}\mathbf{x}(0)
$$

- $A\in\mathbb{R}^{n\times n}$: Constant generator.
- $t\geq0$: Continuous time.
- $\mathbf{x}(t)$: State.

$$
e^{tA}=\sum_{k=0}^{\infty}\frac{t^kA^k}{k!},
\qquad
\frac{d}{dt}e^{tA}=Ae^{tA},
\qquad
e^{(s+t)A}=e^{sA}e^{tA}
$$

- $s$: Additional time interval.
- Diagonalizable $A=V\Lambda V^{-1}$ → $e^{tA}=Ve^{t\Lambda}V^{-1}$.
- $V$: Invertible eigenvector matrix.
- $\Lambda$: Diagonal eigenvalue matrix; exponentiate its diagonal entries.
- $\mathbf{x}(t)\rightarrow0$ for every initial state $\Leftrightarrow\operatorname{Re}(\lambda_j)<0$ for every $j$.
- Exact discrete transition over interval $\Delta t$: $A_d=e^{\Delta t A}$.
- $A_d$: Discrete-time transition matrix.
- $\Delta t>0$: Sampling interval.
```

```{attention} Q&A
:class: dropdown
*Is this elementwise exponentiation?*

- ❌Powers & products are matrix operations; $e^0=I_n$, not an all-ones matrix.
- $e^{A+B}=e^Ae^B$ is guaranteed when $AB=BA$, not in general.

*Can discretization destroy stability?*

- Forward Euler uses $I_n+\Delta t A$, not the exact exponential.
- Example: $\dot{x}=-ax$, $a>0$ → Euler stable only when $0<a\Delta t<2$.
- Exact transition $e^{-a\Delta t}$ is stable for every $\Delta t>0$.
```

&nbsp;

### Controllability
- **What**: Ability to reach any state through a sequence of inputs.
- **Why**: Modeling state evolution is not enough; an agent must be able to influence it.
- **How**: Stack the state directions reachable through inputs at successive times → inspect their span.

```{note} Math
:class: dropdown
Model:

$$
\mathbf{x}_{t+1}=A\mathbf{x}_t+B\mathbf{u}_t,\qquad
\mathbf{x}_n=A^n\mathbf{x}_0+
\begin{bmatrix}B&AB&\cdots&A^{n-1}B\end{bmatrix}
\begin{bmatrix}\mathbf{u}_{n-1}\\\mathbf{u}_{n-2}\\\vdots\\\mathbf{u}_0\end{bmatrix}
$$

- $A\in\mathbb{R}^{n\times n}$: State transition.
- $B\in\mathbb{R}^{n\times p}$: Input map.
- $\mathbf{x}_t\in\mathbb{R}^n$: State.
- $\mathbf{u}_t\in\mathbb{R}^p$: Unconstrained input.
- $p$: Input dimension.
- $t$: Discrete time.

$$
\mathcal{C}=\begin{bmatrix}B&AB&\cdots&A^{n-1}B\end{bmatrix},
\qquad
(A,B)\text{ controllable}\Leftrightarrow\operatorname{rank}(\mathcal{C})=n
$$

- $\mathcal{C}\in\mathbb{R}^{n\times np}$: Controllability matrix.
- Reachable states from zero = $\operatorname{im}(\mathcal{C})$.
```

```{attention} Q&A
:class: dropdown
*Why are $n$ blocks enough?*

- Cayley-Hamilton: $A$ satisfies its degree-$n$ characteristic polynomial.
- Higher powers of $A$ are combinations of lower ones → no new reachable directions after the span saturates.

*Does full rank guarantee easy control?*

- ❌Tiny positive singular values of $\mathcal{C}$ mean some state changes require large input norm.
- Rank criterion assumes unconstrained inputs & an exact linear model.
- State reachability is not the same as planning a good action sequence.
```

&nbsp;

### Observability
- **What**: Ability to reconstruct initial state from a finite output history.
- **Why**: Internal state may not be directly visible in any single observation.
- **How**: Let dynamics expose hidden directions over time → stack the resulting measurements.

```{note} Math
:class: dropdown
Model:

$$
\mathbf{x}_{t+1}=A\mathbf{x}_t,\qquad
\mathbf{y}_t=C\mathbf{x}_t,\qquad
\begin{bmatrix}\mathbf{y}_0\\\mathbf{y}_1\\\vdots\\\mathbf{y}_{n-1}\end{bmatrix}
=
\begin{bmatrix}C\\CA\\\vdots\\CA^{n-1}\end{bmatrix}\mathbf{x}_0
$$

- $A\in\mathbb{R}^{n\times n}$: State transition.
- $C\in\mathbb{R}^{q\times n}$: Observation map.
- $\mathbf{x}_t\in\mathbb{R}^n$: State.
- $\mathbf{y}_t\in\mathbb{R}^q$: Observation.
- $q$: Observation dimension.
- $t$: Discrete time.

$$
\mathcal{O}=\begin{bmatrix}C\\CA\\\vdots\\CA^{n-1}\end{bmatrix},
\qquad
(A,C)\text{ observable}\Leftrightarrow\operatorname{rank}(\mathcal{O})=n
$$

- $\mathcal{O}\in\mathbb{R}^{nq\times n}$: Observability matrix.
- $\ker(\mathcal{O})$: Initial-state differences invisible to the output history.
- Known inputs can be included after subtracting their known output contribution.
```

```{attention} Q&A
:class: dropdown
*How is this related to controllability?*

- $(A,C)$ observable $\Leftrightarrow(A^T,C^T)$ controllable.
- One spans influence directions; the other spans measurement directions.

*Full rank means reliable inference?*

- ❌Nearly invisible directions amplify observation noise during reconstruction.
- With an exact noiseless model, the [pseudoinverse](#pseudoinverse) recovers the state; with noise, estimation requires additional assumptions.

*Does observability establish consciousness?*

- ❌It establishes state identifiability relative to a model & measurement map.
- It makes no claim about subjective experience.
```

&nbsp;

## Structured Representations

### Tensor Product
- **What**: A space representing bilinear combinations of two vector spaces.
- **Why**: Represent interactions between components, not merely concatenate them.
- **How**: Pair every basis direction of one space with every basis direction of the other.

```{note} Math
:class: dropdown
$$
\mathbf{x}\otimes\mathbf{y}
=\sum_{i=1}^m\sum_{j=1}^n x_iy_j(\mathbf{e}_i\otimes\mathbf{f}_j),
\qquad
\dim(V\otimes W)=mn
$$

- $V$: Real vector space of dimension $m$.
- $W$: Real vector space of dimension $n$.
- $\mathbf{x}\in V$: First vector.
- $\mathbf{y}\in W$: Second vector.
- $\mathbf{e}_i$: Basis vectors of $V$.
- $\mathbf{f}_j$: Basis vectors of $W$.
- Here $\mathbf{e}_i$ denotes the chosen basis of $V$, not necessarily the page's standard basis.
- $\otimes$: Tensor product; bilinear in its arguments.
- In chosen bases, $\mathbf{x}\otimes\mathbf{y}$ can be represented by the outer-product matrix $\mathbf{x}\mathbf{y}^T$.
- General tensor = sum of such products; need NOT be one product.
```

```{attention} Q&A
:class: dropdown
*Tensor product vs direct sum?*

- Direct sum $V\oplus W$: store both components separately; dimension $m+n$.
- Tensor product $V\otimes W$: represent pairwise interactions; dimension $mn$.

*Is every multidimensional array a tensor?*

- In ML libraries, “tensor” usually means a multidimensional array.
- In multilinear algebra, an array represents a tensor relative to bases; coordinates obey the corresponding basis-change rules.
- #axes = tensor order; NOT matrix rank or tensor rank.
```

&nbsp;

#### Kronecker Product
- **What**: Block-matrix representation of a tensor product of linear maps.
- **Why**: Express separable operations on structured arrays w/o materializing one huge operator.
- **How**: Replace every entry of the first matrix by that scalar times the second matrix.

```{note} Math
:class: dropdown
$$
A\otimes B=
\begin{bmatrix}
a_{11}B&\cdots&a_{1n}B\\
\vdots&\ddots&\vdots\\
a_{m1}B&\cdots&a_{mn}B
\end{bmatrix}
\in\mathbb{R}^{mp\times nq}
$$

- $A\in\mathbb{R}^{m\times n}$: First map.
- $B\in\mathbb{R}^{p\times q}$: Second map.
- $p$: #rows of $B$.
- $q$: #cols of $B$.

$$
\operatorname{vec}(AXB^T)=(B\otimes A)\operatorname{vec}(X),
\qquad X\in\mathbb{R}^{n\times q}
$$

- $X$: Matrix acted on from the left & right.
- $\operatorname{vec}$: Col-stacking vectorization.
- $\operatorname{rank}(A\otimes B)=\operatorname{rank}(A)\operatorname{rank}(B)$.
- Compatible shapes: $(A\otimes B)(C\otimes D)=(AC)\otimes(BD)$.
```

```{attention} Q&A
:class: dropdown
*Kronecker vs outer vs Hadamard product?*

- **Outer**: $\mathbf{x}\mathbf{y}^T$; every coordinate pair.
- **Kronecker**: Every pair of matrix entries arranged as blocks.
- **Hadamard**: $A\odot B$; same-shaped arrays multiplied entrywise.
- ❌None is ordinary matrix multiplication.

*Why avoid explicitly building the Kronecker matrix?*

- Apply the equivalent left/right products to $X$ instead.
- Structure can reduce storage & work; benefit depends on factor shapes.
```

&nbsp;

### Tensor Contraction
- **What**: Summation over paired tensor indices.
- **Why**: Combine structured representations while preserving uncontracted axes.
- **How**: Multiply matching index entries → sum over the shared index → retain the others.

```{note} Math
:class: dropdown
$$
c_{ijk}=\sum_{\ell=1}^d a_{ij\ell}b_{\ell k}
$$

- $\mathbf{A}\in\mathbb{R}^{m\times n\times d}$: Order-3 input tensor.
- $B\in\mathbb{R}^{d\times p}$: Map on the last axis.
- $\mathbf{C}\in\mathbb{R}^{m\times n\times p}$: Output tensor.
- $p$: Output-axis size.
- $d$: Contracted-axis size.
- Free indices $i,j,k$ determine output axes; $\ell$ is summed out.
- Dot product, matrix multiplication & trace are contractions.
```

```{attention} Q&A
:class: dropdown
*Why is index bookkeeping more than notation?*

- It states exactly which axes interact & which survive.
- Broadcasting introduces/reuses axes; contraction removes a paired index through summation.
- Attention's query-key score is a feature-axis contraction; see [Scaled Dot-Product Attention](../nlp/attn.md#scaled-dot-product-attention).

*Can the contraction order matter?*

- Algebraically equivalent orders can require very different intermediate storage & arithmetic.
- Floating-point results may also differ because addition is not exactly associative.
```

&nbsp;

### Graph Laplacian
- **What**: Linear operator measuring deviation from weighted neighbors.
- **Why**: Represent relations, smooth signals & diffusion on a graph.
- **How**: Degree-weighted self-value − weighted neighbor values.

```{note} Math
:class: dropdown
For an undirected graph with nonnegative weights & no self-loops:

$$
D_{ii}=\sum_{j=1}^n w_{ij},\qquad
L=D-W,\qquad
(L\mathbf{x})_i=\sum_{j=1}^n w_{ij}(x_i-x_j)
$$

- $W\in\mathbb{R}^{n\times n}$: Symmetric weighted adjacency matrix.
- $w_{ij}$: Edge weight; zero means no edge.
- $D\in\mathbb{R}^{n\times n}$: Diagonal degree matrix.
- $L\in\mathbb{R}^{n\times n}$: Unnormalized graph Laplacian.
- $\mathbf{x}\in\mathbb{R}^n$: Scalar signal over nodes.
- Here $n$ is #nodes.

$$
\mathbf{x}^TL\mathbf{x}
=\frac12\sum_{i,j=1}^n w_{ij}(x_i-x_j)^2\geq0
$$

- $L$ symmetric PSD; $L\mathbf{1}=\mathbf{0}$.
- $\mathbf{1}$: All-ones vector.
- Nullity = #connected components using positive-weight edges.
- Zero energy $\Leftrightarrow$ signal constant on each connected component.
```

```{attention} Q&A
:class: dropdown
*What do Laplacian eigenvectors represent?*

- Small eigenvalues → slowly varying graph signals; large eigenvalues → stronger neighbor disagreement.
- Connected graph → second-smallest eigenvalue $>0$.
- [Spectral Clustering](../ml/unsupervised.md#spectral-clustering) uses low-frequency graph structure to separate groups.

*Why does diffusion smooth the signal?*

- $\dot{\mathbf{x}}=-L\mathbf{x}$ → $\mathbf{x}(t)=e^{-tL}\mathbf{x}(0)$.
- Positive-eigenvalue modes decay; zero modes survive.
- Each connected component converges to its initial arithmetic mean.
- Connectedness describes a relation graph, NOT a measure or proof of consciousness.
```

&nbsp;