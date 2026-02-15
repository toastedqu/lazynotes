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
| 11 | $\text{mullity}(A)=0$. |
| 12 | Eigenvalues $\lambda\ne 0$. |

(TBC)
```