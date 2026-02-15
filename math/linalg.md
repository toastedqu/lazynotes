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

<!-- ```{dropdown} Table: Linear Independence - Equivalent Statements
If $A\in\mathbb{R}^{n\times n}$, the statements below are equivalent:

|    | Statement |
|----|-----------|
| 1  | $\exists A^{-1}$ (i.e., $A$ is invertible.) |
| 2  | $A\mathbf{x}=\mathbf{0}\Rightarrow \mathbf{x}=\mathbf{0}$ (i.e., ONLY trivial solution.) |
| 3  | The reduced row echelon form of $A$ is $I_n$. |

(TBC)
``` -->