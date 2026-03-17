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