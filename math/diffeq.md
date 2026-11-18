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
# Differential Equations
ODEs, dynamical systems, numerical solvers, PDEs & SDEs. These are the continuous-time math behind optimizers, SSMs, diffusion models & Neural ODEs.

Notations:
- $t$: Time (independent variable).
- $\mathbf{x}(t)\in\mathbb{R}^d$: State.
- $x(t)\in\mathbb{R}$: Scalar state ($d=1$).
- $\dot{\mathbf{x}}$: Time derivative $d\mathbf{x}/dt$.
- $\mathbf{f}(\mathbf{x},t)$: Vector field (drift in SDEs); $\dot{\mathbf{x}}=\mathbf{f}(\mathbf{x},t)$.
- $\mathbf{x}_0$: Initial state $\mathbf{x}(t_0)$.
- $A\in\mathbb{R}^{d\times d}$: Constant system matrix.
- $\lambda$: Eigenvalue of $A$ or of a Jacobian.
- $h$: Solver step size.
- $t_k=t_0+kh$: Grid time.
- $\mathbf{x}_k$: Numerical estimate of $\mathbf{x}(t_k)$.
- $\mathbf{W}_t$: Standard Brownian motion (Wiener process).

$\mathbf{x}$ = state & $k$ = step idx, overriding the global input & class-idx meanings.

&nbsp;

## Basics

### Ordinary Differential Equation
- **What**: Equation linking an unknown function of 1 variable to its derivatives.
- **Why**: Laws of change are local: rate now = function of state now.
    - Integrate the local rule → global trajectory.
- **How**: Specify $\mathbf{f}$ & $\mathbf{x}_0$ → solve analytically or numerically for $\mathbf{x}(t)$.

```{note} Math
:class: dropdown
Reduction to a first-order system:

$$
x^{(n)}=F\left(t,x,\dot x,\dots,x^{(n-1)}\right)
\quad\Rightarrow\quad
\mathbf{x}=\begin{bmatrix}x\\\dot x\\\vdots\\x^{(n-1)}\end{bmatrix},\quad
\dot{\mathbf{x}}=\begin{bmatrix}x_2\\x_3\\\vdots\\F(t,x_1,\dots,x_n)\end{bmatrix}
$$

- $F$: Right side of the $n$-th order ODE.
- $x_j$: $j$-th component of $\mathbf{x}$.
- Non-autonomous → autonomous: add $t$ as a state w/ $\dot t=1$.
```

```{dropdown} Table: ODE Classification
| Property | Definition | Example |
|:--|:--|:--|
| Order | Highest derivative | $\ddot x+x=0$: 2nd order |
| Linear | Unknown & its derivatives appear linearly | ✅ $\ddot x+t^2x=\sin t$; ❌ $\dot x=x^2$ |
| Homogeneous (linear) | No forcing term | $\dot{\mathbf{x}}=A\mathbf{x}$ |
| Autonomous | $\mathbf{f}$ does not depend on $t$ | $\dot x=x(1-x)$ |
| IVP | Conditions at 1 time point | $\mathbf{x}(t_0)=\mathbf{x}_0$ |
| BVP | Conditions at $\ge2$ points | $x(0)=x(1)=0$ |
| PDE | Unknown depends on $\ge2$ variables | [Heat Equation](#heat-equation) |
| SDE | Random forcing term | [Stochastic Differential Equations](#stochastic-differential-equations) |
```

```{attention} Q&A
:class: dropdown
*Why study only first-order systems?*
- Every $n$-th order ODE reduces to one → 1 theory, 1 solver API (e.g., `scipy.integrate.solve_ivp`, `torchdiffeq.odeint`).

*ODE vs difference equation?*
- Difference equation: discrete time, $\mathbf{x}_{k+1}=\mathbf{g}(\mathbf{x}_k)$.
- Euler step w/ $h\to0$ → ODE. RNN = difference equation. Neural ODE & SSM = ODE.
```

&nbsp;

### Existence & Uniqueness
- **What**: Conditions under which an IVP has exactly 1 solution.
- **Why**: W/o them, "the" trajectory is undefined → simulation, Neural ODEs & adjoint gradients lose meaning.
- **How**: Lipschitz $\mathbf{f}$ → Picard iteration is a contraction → unique fixed point = solution.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $L$: Lipschitz constant of $\mathbf{f}$ in $\mathbf{x}$.

Lipschitz condition:

$$
\lVert\mathbf{f}(\mathbf{x},t)-\mathbf{f}(\mathbf{y},t)\rVert\le L\lVert\mathbf{x}-\mathbf{y}\rVert
$$

**Picard–Lindelöf**: $\mathbf{f}$ continuous in $t$ & Lipschitz in $\mathbf{x}$ near $(t_0,\mathbf{x}_0)$ → unique solution on some $[t_0-\epsilon,t_0+\epsilon]$.
- $\epsilon>0$: Length of the guaranteed interval.

Properties:
- **Peano**: $\mathbf{f}$ only continuous → existence ✅, uniqueness ❌.
- Globally Lipschitz → unique solution for all $t$.
- **Grönwall** → continuous dependence on the initial state:

    $$
    \lVert\mathbf{x}(t)-\mathbf{y}(t)\rVert\le\lVert\mathbf{x}_0-\mathbf{y}_0\rVert\,e^{L\lvert t-t_0\rvert}
    $$

    - $\mathbf{y}(t)$: Solution from initial state $\mathbf{y}_0$.
```

```{tip} Derivation
:class: dropdown
*Why does Lipschitz give uniqueness?*
1. IVP ⇔ integral equation $\mathbf{x}(t)=\mathbf{x}_0+\int_{t_0}^t\mathbf{f}(\mathbf{x}(s),s)\,ds$.
2. Define the Picard map $T$: $(T\mathbf{x})(t)=$ right side.
3. $\lVert T\mathbf{x}-T\mathbf{y}\rVert_\infty\le L\epsilon\,\lVert\mathbf{x}-\mathbf{y}\rVert_\infty$ on $[t_0-\epsilon,t_0+\epsilon]$.
4. Choose $\epsilon$ small: $L\epsilon<1$ & $T$ maps a ball around $\mathbf{x}_0$ into itself → $T$ is a contraction.
5. Banach fixed-point theorem → unique fixed point = unique solution. Iterating $T$ converges to it.
```

```{note} Example
:class: dropdown
Picard iteration for $\dot x=x$, $x(0)=1$, $x^{(n+1)}(t)=1+\int_0^tx^{(n)}(s)\,ds$:
- $x^{(0)}=1$
- $x^{(1)}=1+t$
- $x^{(2)}=1+t+\frac{t^2}{2}$
- → $x^{(n)}$ = Taylor partial sums of $e^t$ → $e^t$.
```

```{attention} Q&A
:class: dropdown
*Example of non-uniqueness?*
- $\dot x=\sqrt{\lvert x\rvert}$, $x(0)=0$: both $x\equiv0$ & $x=\frac{t^2}{4}$ ($t\ge0$) solve it.
- $\sqrt{\lvert x\rvert}$ is not Lipschitz at $0$ ← slope → $\infty$.

*Example of finite-time blow-up?*
- $\dot x=x^2$, $x(0)=1$ → $x=\frac{1}{1-t}$ → $\infty$ at $t=1$.
- $x^2$ is only locally Lipschitz → local existence only. Superlinear growth can escape in finite time.

*Why does this matter for Neural ODEs?*
- MLP w/ Lipschitz activations (tanh, ReLU, softplus) & finite weights → globally Lipschitz in $\mathbf{x}$ → unique global solution.
- Uniqueness → trajectories never cross → [flow map](#phase-portrait) is invertible.

*What does Grönwall say about sensitivity?*
- Perturbations grow at most like $e^{Lt}$. The bound can be loose. [Chaos](#chaos) reaches exponential growth.
```

&nbsp;

## Analytic Solutions

### Separable Equation
- **What**: $\dot x=g(t)\,u(x)$.
- **Why**: Simplest nonlinear class w/ a closed form → exponential decay, logistic growth.
- **How**:
    1. Divide by $u(x)$.
    2. Integrate both sides: $\int\frac{dx}{u(x)}=\int g(t)\,dt+C$.
    3. Solve for $x$.
    4. Add the constant solutions $x\equiv x^*$ w/ $u(x^*)=0$.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $g(t)$: Time factor.
    - $u(x)$: State factor.
    - $C$: Integration constant.
```

```{tip} Derivation
:class: dropdown
*Why is the "multiply by $dt$" trick valid?*
1. Let $U'(x)=\frac{1}{u(x)}$.
2. Chain rule: $\frac{d}{dt}U(x(t))=\frac{\dot x}{u(x)}=g(t)$.
3. Integrate in $t$: $U(x(t))=\int g(t)\,dt+C$.
```

```{note} Example
:class: dropdown
Logistic growth $\dot x=rx\left(1-\frac{x}{K}\right)$:

$$
x(t)=\frac{K}{1+\frac{K-x_0}{x_0}e^{-rt}}
$$

- $r$: Growth rate.
- $K$: Carrying capacity.
- $K=r=1$ → $x(t)=\sigma(t+\operatorname{logit}x_0)$ → the sigmoid solves $\dot x=x(1-x)$ ← $\sigma'=\sigma(1-\sigma)$.
```

```{attention} Q&A
:class: dropdown
*Which solutions does the division lose?*
- Constant solutions w/ $u(x^*)=0$. Logistic: $x\equiv0$ & $x\equiv K$.
```

&nbsp;

### First-Order Linear Equation
- **What**: $\dot x+p(t)\,x=q(t)$.
- **Why**: Closed form for any $p,q$. Models relaxation toward a moving target (RC circuit, EMA).
- **How**:
    1. Multiply by the integrating factor $\mu(t)=e^{\int p\,dt}$.
    2. Left side becomes $(\mu x)'$.
    3. Integrate.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $p(t)$: Decay-rate coeff.
    - $q(t)$: Forcing term.
    - $P(t)=\int_{t_0}^tp(s)\,ds$: Accumulated rate.

Solution:

$$
x(t)=e^{-P(t)}x_0+\int_{t_0}^te^{-(P(t)-P(s))}q(s)\,ds
$$

- 1st term: free decay of the initial state.
- 2nd term: each past input $q(s)$, decayed by the rate accumulated since $s$.
```

```{tip} Derivation
:class: dropdown
1. $\mu=e^{P}$ → $\dot\mu=p\mu$.
2. $(\mu x)'=\dot\mu x+\mu\dot x=\mu(\dot x+px)=\mu q$.
3. Integrate from $t_0$ to $t$: $\mu(t)x(t)-x_0=\int_{t_0}^t\mu(s)q(s)\,ds$.
4. Divide by $\mu(t)$.
```

```{note} Example
:class: dropdown
EMA = exact discretization of $\dot x=\frac{u(t)-x}{\tau}$:
- $\tau$: Time constant.
- $u$: Input, held constant at $u_k$ over each step of length $h$.
- Exact step: $x_k=e^{-h/\tau}x_{k-1}+\left(1-e^{-h/\tau}\right)u_k$ → EMA w/ $\beta=e^{-h/\tau}$.
- Memory: $\tau=-\frac{h}{\ln\beta}\approx\frac{h}{1-\beta}$ → $\beta=0.9$ ≈ 10 steps.
```

```{attention} Q&A
:class: dropdown
*Steady state for constant $p>0$ & $q$?*
- $x\to\frac{q}{p}$. Error decays as $e^{-pt}$. Time constant $\tau=\frac1p$.

*How does it generalize to vectors?*
- [Duhamel's Formula](#duhamels-formula) w/ $e^{tA}$ in place of $e^{-P(t)}$.
```

&nbsp;

### Other First-Order Forms
- **What**: First-order ODEs that a substitution reduces to separable, linear or exact form.
- **How**: Substitute → solve the reduced ODE → substitute back.

```{dropdown} Table: Reducible Forms
| Form | Equation | Reduction |
|:--|:--|:--|
| Exact | $M(t,x)\,dt+N(t,x)\,dx=0$, $\partial_xM=\partial_tN$ | Find $\Phi$: $\partial_t\Phi=M$, $\partial_x\Phi=N$ → $\Phi(t,x)=C$ |
| Integrating factor | Not exact, but $\frac{\partial_xM-\partial_tN}{N}$ depends only on $t$ | Multiply by $\mu(t)=e^{\int\frac{\partial_xM-\partial_tN}{N}dt}$ → exact |
| Bernoulli | $\dot x+p(t)x=q(t)x^n$, $n\ne0,1$ | $v=x^{1-n}$ → $\dot v+(1-n)pv=(1-n)q$ (linear) |
| Homogeneous | $\dot x=F\left(\frac xt\right)$ | $v=\frac xt$ → $t\dot v=F(v)-v$ (separable) |

- $M,N$: Coefficient functions.
- Exact ⇔ $(M,N)$ is a gradient field in the $(t,x)$ plane → [Conservative Fields](calc.md#conservative-fields).
```

&nbsp;

### Linear Constant-Coefficient Equation
- **What**: $a_nx^{(n)}+\dots+a_1\dot x+a_0x=u(t)$, constant $a_i$.
- **Why**: Oscillators, circuits, control & linearized dynamics near equilibria. Closed form via polynomial roots.
- **How**:
    1. Homogeneous part: try $x=e^{rt}$ → characteristic polynomial $\sum_ia_ir^i=0$.
    2. Each root → basis solutions (table).
    3. Forced: general solution = homogeneous solution + 1 particular solution.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $a_i$: Constant coefficients, $a_n\ne0$.
    - $u(t)$: Forcing term.
    - $r$: Root of the characteristic polynomial.

Characteristic polynomial:

$$
a_nr^n+\dots+a_1r+a_0=0
$$

| Root | Basis solutions |
|:--|:--|
| Real $r$, multiplicity 1 | $e^{rt}$ |
| Real $r$, multiplicity $m$ | $e^{rt},\,te^{rt},\,\dots,\,t^{m-1}e^{rt}$ |
| Complex pair $\alpha\pm i\omega$ | $e^{\alpha t}\cos\omega t,\ e^{\alpha t}\sin\omega t$ |

Particular solution (undetermined coefficients):

| Forcing $u(t)$ | Guess |
|:--|:--|
| Polynomial of degree $q$ | Polynomial of degree $q$ |
| $e^{ct}$ | $Be^{ct}$ |
| $\cos\omega t$ or $\sin\omega t$ | $B_1\cos\omega t+B_2\sin\omega t$ |
| Guess solves the homogeneous equation | Multiply the guess by $t^m$, $m$ = root multiplicity |

- $B,B_1,B_2$: Unknown constants to fit.
```

```{note} Example
:class: dropdown
Damped oscillator $\ddot x+2\zeta\omega_0\dot x+\omega_0^2x=0$ → roots $r=-\zeta\omega_0\pm\omega_0\sqrt{\zeta^2-1}$:
- $\omega_0$: Natural frequency.
- $\zeta$: Damping ratio.

| Regime | Condition | Behavior |
|:--|:--|:--|
| Underdamped | $\zeta<1$ | $e^{-\zeta\omega_0t}\cos(\omega_dt+\phi)$, $\omega_d=\omega_0\sqrt{1-\zeta^2}$ → decaying oscillation |
| Critically damped | $\zeta=1$ | $(c_1+c_2t)e^{-\omega_0t}$ → fastest decay w/o overshoot |
| Overdamped | $\zeta>1$ | 2 real negative roots → slow, no oscillation |

- $\phi,c_1,c_2$: Constants fixed by the initial conditions.
```

```{attention} Q&A
:class: dropdown
*Why does $e^{rt}$ work?*
- $e^{rt}$ is an eigenfunction of $\frac{d}{dt}$: $\frac{d}{dt}e^{rt}=re^{rt}$ → the ODE becomes a polynomial in $r$.

*When does every solution decay?*
- All roots have $\operatorname{Re}r<0$.

*What is resonance?*
- Forcing at a natural frequency of an undamped system → amplitude grows ∝ $t$.
- $\ddot x+\omega^2x=\cos\omega t$ → $x_p=\frac{t\sin\omega t}{2\omega}$.

*How does this relate to momentum?*
- Heavy-ball momentum on $\mathcal{L}(x)=\frac12\mu x^2$ → $\ddot x+\gamma\dot x+\mu x=0$ ([Gradient Flow](#gradient-flow)).
- $\omega_0=\sqrt\mu$, $\zeta=\frac{\gamma}{2\sqrt\mu}$ → critical damping at $\gamma=2\sqrt\mu$.
- Ill-conditioned loss: curvatures differ → 1 friction value cannot critically damp all directions.
```

&nbsp;

### Laplace Transform
- **What**: $X(s)=\int_0^\infty e^{-st}x(t)\,dt$.
- **Why**: Turns linear constant-coefficient ODEs into algebra.
    - Derivative → multiply by $s$ (+ initial conditions).
    - Convolution → product.
- **How**:
    1. Transform the ODE incl. initial conditions.
    2. Solve for $X(s)$ algebraically.
    3. Invert w/ partial fractions & the table.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $s\in\mathbb{C}$: Complex frequency.
    - $X(s),U(s),Y(s)$: Transforms of $x,u,y$.

Transfer function of $\dot{\mathbf{x}}=A\mathbf{x}+B\mathbf{u}$, $\mathbf{y}=C\mathbf{x}$, $\mathbf{x}(0)=\mathbf{0}$:

$$
Y(s)=C(sI-A)^{-1}B\,U(s)
$$

- $B,C$: Input & output matrices (as in [Duhamel's Formula](#duhamels-formula)).
- $\mathcal{L}^{-1}\{(sI-A)^{-1}\}=e^{tA}$ → impulse response $C\,e^{tA}B$.
- Poles ⊆ eigenvalues of $A$. Uncontrollable/unobservable modes cancel → equality only for a minimal realization.
```

```{dropdown} Table: Laplace Pairs
| $x(t)$ | $X(s)$ |
|:--|:--|
| $\delta(t)$ | $1$ |
| $1$ | $\frac1s$ |
| $t^n$ | $\frac{n!}{s^{n+1}}$ |
| $e^{at}$ | $\frac1{s-a}$ |
| $\sin\omega t$ | $\frac{\omega}{s^2+\omega^2}$ |
| $\cos\omega t$ | $\frac{s}{s^2+\omega^2}$ |
| $e^{at}x(t)$ | $X(s-a)$ |
| $\dot x$ | $sX(s)-x(0)$ |
| $\ddot x$ | $s^2X(s)-sx(0)-\dot x(0)$ |
| $\int_0^tx(\tau)u(t-\tau)\,d\tau$ | $X(s)U(s)$ |

- $\delta$: Dirac delta.
- Each pair holds for $\operatorname{Re}s$ larger than the growth rate of $x$.
```

```{attention} Q&A
:class: dropdown
*Laplace vs Fourier?*
- Fourier = two-sided transform on $s=i\omega$.
- Laplace: one-sided → handles initial conditions & growing signals.

*How does it connect to SSMs?*
- SSM convolution kernel = sampled impulse response of the discretized system → [Convolutional View](../nlp/ssm.md#convolutional-view).
```

&nbsp;

## Linear Systems

### Linear System
- **What**: $\dot{\mathbf{x}}=A\mathbf{x}$, $\mathbf{x}(t_0)=\mathbf{x}_0$.
- **Why**: Exactly solvable in any dimension.
    - Every smooth system ≈ linear near an equilibrium → local behavior = eigenvalues ([Equilibrium & Stability](#equilibrium-stability)).
- **How**:
    1. Solution: $\mathbf{x}(t)=e^{(t-t_0)A}\mathbf{x}_0$.
    2. Diagonalizable $A$ → write $\mathbf{x}_0$ in the eigenbasis → each mode evolves independently as $e^{\lambda_jt}$.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\mathbf{v}_j$: Eigenvector for $\lambda_j$.
    - $V=[\mathbf{v}_1,\dots,\mathbf{v}_d]$: Eigenvector matrix.
    - $c_j$: Coordinates of $\mathbf{x}_0$ in the eigenbasis, $\mathbf{c}=V^{-1}\mathbf{x}_0$.

Solution:

$$
\mathbf{x}(t)=e^{(t-t_0)A}\mathbf{x}_0=\sum_{j=1}^dc_j\,e^{\lambda_j(t-t_0)}\mathbf{v}_j
$$

- $\operatorname{Re}\lambda_j$: Growth/decay rate of mode $j$.
- $\operatorname{Im}\lambda_j$: Rotation frequency. Complex pairs → real solutions oscillate.
- Defective $A$ → Jordan blocks → extra polynomial factors $t^me^{\lambda t}$.
- Eigendecomposition: [Eigendecomposition](linalg.md#eigendecomposition).
```

```{note} Example
:class: dropdown
$A=\begin{bmatrix}0&1\\-1&0\end{bmatrix}$ → $\lambda=\pm i$:

$$
e^{tA}=\begin{bmatrix}\cos t&\sin t\\-\sin t&\cos t\end{bmatrix}
$$

- Clockwise rotation by angle $t$ → harmonic oscillator ($\dot x_1=x_2$, $\dot x_2=-x_1$).
```

```{attention} Q&A
:class: dropdown
*When is $\mathbf{x}=\mathbf{0}$ stable?*
- Asymptotically stable ⇔ all $\operatorname{Re}\lambda<0$.
- Bounded (Lyapunov stable) ⇔ all $\operatorname{Re}\lambda\le0$ & eigenvalues w/ $\operatorname{Re}\lambda=0$ have no Jordan blocks.

*Can $\lVert\mathbf{x}(t)\rVert$ grow when all $\operatorname{Re}\lambda<0$?*
- ✅ Yes, temporarily, if $A$ is non-normal ($AA^T\ne A^TA$).
- E.g., $A=\begin{bmatrix}-1&100\\0&-2\end{bmatrix}$: $x_2$ feeds $x_1$ w/ gain 100 → large transient before decay.
- Eigenvalues give only the asymptotic rate.

*Discrete-time analog?*
- $\mathbf{x}_{k+1}=M\mathbf{x}_k$ → $\mathbf{x}_k=M^k\mathbf{x}_0$ → asymptotically stable ⇔ spectral radius $<1$.
- $M=e^{hA}$ maps $\operatorname{Re}\lambda<0$ to $\lvert e^{h\lambda}\rvert<1$ → [SSM Discretization](../nlp/ssm.md#discretization).
```

&nbsp;

#### Matrix Exponential
- **What**: $e^{A}=\sum_{n=0}^\infty\frac{A^n}{n!}$.
- **Why**: Solves $\dot{\mathbf{x}}=A\mathbf{x}$ ← $\frac{d}{dt}e^{tA}=Ae^{tA}$ → matrix analog of $e^{at}$.
- **How**:
    - Diagonalizable: $e^{tA}=Ve^{t\Lambda}V^{-1}$ → exponentiate each eigenvalue.
    - General (numerical): scaling & squaring: $e^{A}=\left(e^{A/2^s}\right)^{2^s}$ w/ Padé or Taylor for the small matrix.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\Lambda=\operatorname{diag}(\lambda_1,\dots,\lambda_d)$: Eigenvalue matrix.
    - $s$: #Squarings.
```

```{dropdown} Table: Matrix Exponential Properties
| Property | Statement |
|:--|:--|
| Derivative | $\frac{d}{dt}e^{tA}=Ae^{tA}=e^{tA}A$ |
| Group | $e^{(t+\tau)A}=e^{tA}e^{\tau A}$ |
| Inverse | $\left(e^A\right)^{-1}=e^{-A}$ → always invertible |
| Commuting sum | $AB=BA$ → $e^{A+B}=e^Ae^B$; ❌ in general |
| Similarity | $e^{PAP^{-1}}=Pe^AP^{-1}$ |
| Eigenvalues | $\lambda\left(e^A\right)=e^{\lambda(A)}$ |
| Determinant (Liouville) | $\det e^A=e^{\operatorname{tr}A}$ |
| Skew-symmetric | $A^T=-A$ → $e^A$ orthogonal |

- $B$: Any $d\times d$ matrix. $P$: Any invertible matrix. $\tau$: Second time.
```

````{important} Code
:class: dropdown
```python
import math
import torch

class MatrixExp:
    def __init__(self, n_terms=12):
        self.n_terms = n_terms

    def __call__(self, A):
        ## Scale so ||A / 2^s|| < 1/2 -> short Taylor series is accurate
        norm = torch.linalg.matrix_norm(A, ord=1).item()
        s = max(0, math.frexp(norm)[1] + 1)
        As = A / 2**s
        ## Truncated Taylor series: sum_n As^n / n!
        E = torch.eye(A.shape[-1], dtype=A.dtype)
        term = E.clone()
        for n in range(1, self.n_terms):
            term = term @ As / n
            E = E + term
        ## Undo the scaling: e^A = (e^{A / 2^s})^{2^s}
        for _ in range(s):
            E = E @ E
        return E

## Example: rotation generator -> e^{tA} rotates by angle t
A = torch.tensor([[0., 1.], [-1., 0.]], dtype=torch.float64)
E = MatrixExp()(torch.pi / 2 * A)
print(E)  ## ~[[0, 1], [-1, 0]]
print(torch.allclose(E, torch.linalg.matrix_exp(torch.pi / 2 * A)))  ## True
```
````

```{attention} Q&A
:class: dropdown
*Why is $e^{A+B}\ne e^Ae^B$ in general?*
- Expansions differ at 2nd order by the commutator: $e^Ae^B=e^{A+B+\frac12[A,B]+\dots}$, $[A,B]=AB-BA$ (BCH formula).

*Why scale & square instead of a plain Taylor series?*
- Large $\lVert A\rVert$ → huge terms of alternating sign cancel → roundoff error.
- Scaling first → small norm → few terms, no cancellation.

*Why is $\det e^A=e^{\operatorname{tr}A}$?*
- $\det$ = product of eigenvalues = $\prod_je^{\lambda_j}=e^{\sum_j\lambda_j}$.
- → Volume under $\dot{\mathbf{x}}=A\mathbf{x}$ scales by $e^{t\operatorname{tr}A}$ → [Continuity Equation](#continuity-equation).
```

&nbsp;

#### Duhamel's Formula
- **What**: Solution of the forced system $\dot{\mathbf{x}}=A\mathbf{x}+B\mathbf{u}(t)$: free response + convolution of the input w/ $e^{tA}B$.
- **Why**: Input-driven linear systems (control, SSMs) need the response to an arbitrary input $\mathbf{u}$.
- **How**:
    1. Each past input $B\mathbf{u}(s)\,ds$ = kick to the state at time $s$.
    2. Kick evolves freely for $t-s$ → $e^{(t-s)A}B\mathbf{u}(s)\,ds$.
    3. Linearity → sum all kicks + free response of $\mathbf{x}_0$.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{u}(t)\in\mathbb{R}^p$: Input.
    - $\mathbf{y}(t)\in\mathbb{R}^q$: Output.
- Params:
    - $B\in\mathbb{R}^{d\times p}$: Input matrix.
    - $C\in\mathbb{R}^{q\times d}$: Output matrix.

Solution:

$$
\mathbf{x}(t)=e^{(t-t_0)A}\mathbf{x}_0+\int_{t_0}^te^{(t-s)A}B\mathbf{u}(s)\,ds
$$

Output ($\mathbf{y}=C\mathbf{x}$, $t_0=0$, $\mathbf{x}_0=\mathbf{0}$):

$$
\mathbf{y}(t)=\int_0^tK(t-s)\,\mathbf{u}(s)\,ds,\qquad K(t)=Ce^{tA}B
$$

- $K(t)$: Impulse response (convolution kernel).
```

```{tip} Derivation
:class: dropdown
*Variation of constants*:
1. Guess $\mathbf{x}=e^{tA}\mathbf{c}(t)$.
2. $\dot{\mathbf{x}}=Ae^{tA}\mathbf{c}+e^{tA}\dot{\mathbf{c}}=A\mathbf{x}+B\mathbf{u}$ → $\dot{\mathbf{c}}=e^{-tA}B\mathbf{u}$.
3. Integrate: $\mathbf{c}(t)=e^{-t_0A}\mathbf{x}_0+\int_{t_0}^te^{-sA}B\mathbf{u}(s)\,ds$.
4. Multiply by $e^{tA}$ → result.
```

```{attention} Q&A
:class: dropdown
*How do SSMs use it?*
- ZOH discretization = Duhamel over 1 step w/ constant input → [Discretization](../nlp/ssm.md#discretization).
- Kernel $K$, sampled → [Convolutional View](../nlp/ssm.md#convolutional-view).

*Time-varying $A(t)$?*
- Replace $e^{(t-s)A}$ by the state-transition matrix $\Phi(t,s)$.
- $\Phi(t,s)=e^{\int_s^tA(\tau)d\tau}$ only if all $A(\tau)$ commute.
```

&nbsp;

## Dynamical Systems
- **What**: Geometry & long-run behavior of $\dot{\mathbf{x}}=\mathbf{f}(\mathbf{x})$, w/o a closed-form solution.

### Phase Portrait
- **What**: Picture of all trajectories of an autonomous ODE in state space.
- **Why**: Most nonlinear ODEs have no closed form. Geometry (equilibria, cycles, basins) still answers "where does the state go?".
- **How**:
    1. Draw $\mathbf{f}$ as arrows (vector field).
    2. Trajectories = curves tangent to the arrows.
    3. Mark equilibria & their stability → infer the global flow.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\varphi_t$: Flow map, $\varphi_t(\mathbf{x}_0)=\mathbf{x}(t)$ for $\mathbf{x}(0)=\mathbf{x}_0$.
    - $\Omega\subset\mathbb{R}^d$: Region of initial states.

Flow map:
- Group: $\varphi_0=\mathrm{id}$, $\varphi_{t+s}=\varphi_t\circ\varphi_s$ (where defined).
- Uniqueness → trajectories never cross → $\varphi_t$ invertible onto its image, $\varphi_t^{-1}=\varphi_{-t}$.
- $\mathbf{f}\in C^1$ → $\varphi_t$ is a diffeomorphism onto its image.
- Global on $\mathbb{R}^d$ for all $t$ only if solutions exist for all $t$ (complete flow, e.g., globally Lipschitz $\mathbf{f}$).
    - ❌ $\dot x=-x^3$: $\varphi_t(x)=\frac{x}{\sqrt{1+2tx^2}}$ → image of $\mathbb{R}$ is a bounded interval for $t>0$; backward solutions blow up.
- Volume (Liouville):

    $$
    \frac{d}{dt}\operatorname{vol}\big(\varphi_t(\Omega)\big)=\int_{\varphi_t(\Omega)}\nabla\cdot\mathbf{f}\,d\mathbf{x}
    $$

    - $\nabla\cdot\mathbf{f}=0$ → volume-preserving (e.g., Hamiltonian systems).
```

```{attention} Q&A
:class: dropdown
*Why can't a 1D autonomous ODE oscillate?*
- A trajectory on a line cannot turn back w/o reaching a point w/ $\dot x=0$ → monotone in $t$.

*Why is there no chaos in 2D?*
- **Poincaré–Bendixson**: bounded planar trajectory w/ finitely many equilibria nearby → approaches an equilibrium, a periodic orbit, or a cycle of equilibria joined by orbits.
- → [Chaos](#chaos) needs $d\ge3$ for continuous autonomous flows. Forced 2D systems count as 3D ($t$ becomes a state).

*What does the flow map imply for Neural ODEs?*
- $\varphi_t$ is a homeomorphism → it cannot change topology (e.g., cannot separate 2 nested rings in the same dimension) → [Neural ODE](#neural-ode).
```

&nbsp;

### Equilibrium & Stability
- **What**: State $\mathbf{x}^*$ w/ $\mathbf{f}(\mathbf{x}^*)=\mathbf{0}$ & its response to small perturbations.
- **Why**: Long-run behavior.
    - Loss minima = stable equilibria of [gradient flow](#gradient-flow).
    - RNN memory & training convergence = stability questions.
- **How**: Linearize: $\dot{\boldsymbol{\delta}}\approx J\boldsymbol{\delta}$ → eigenvalues of $J$ decide.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\boldsymbol{\delta}=\mathbf{x}-\mathbf{x}^*$: Perturbation.
    - $J=\frac{\partial\mathbf{f}}{\partial\mathbf{x}}\big|_{\mathbf{x}^*}$: Jacobian at the equilibrium.

Definitions:
- **Lyapunov stable**: $\forall\varepsilon>0\ \exists\eta>0$: $\lVert\mathbf{x}_0-\mathbf{x}^*\rVert<\eta$ → $\lVert\mathbf{x}(t)-\mathbf{x}^*\rVert<\varepsilon$ $\forall t\ge0$.
- **Asymptotically stable**: Lyapunov stable & $\mathbf{x}(t)\to\mathbf{x}^*$ for all $\mathbf{x}_0$ near $\mathbf{x}^*$.
- **Exponentially stable**: $\lVert\mathbf{x}(t)-\mathbf{x}^*\rVert\le ce^{-\alpha t}\lVert\mathbf{x}_0-\mathbf{x}^*\rVert$ for some $c,\alpha>0$.

Linearization (Lyapunov's indirect method):
- All $\operatorname{Re}\lambda(J)<0$ → exponentially stable.
- Any $\operatorname{Re}\lambda(J)>0$ → unstable.
- Max $\operatorname{Re}\lambda(J)=0$ → inconclusive.
- **Hartman–Grobman**: no $\operatorname{Re}\lambda=0$ (hyperbolic) → near $\mathbf{x}^*$, the flow is a continuous deformation of the linear flow.
```

```{dropdown} Table: 2D Equilibria
| Condition | Type |
|:--|:--|
| $\Delta<0$ | Saddle |
| $\Delta>0$, $\tau<0$, $\tau^2\ge4\Delta$ | Stable node |
| $\Delta>0$, $\tau<0$, $\tau^2<4\Delta$ | Stable spiral |
| $\Delta>0$, $\tau=0$ | Center (linear); inconclusive for nonlinear $\mathbf{f}$ |
| $\Delta>0$, $\tau>0$ | Unstable node / spiral |
| $\Delta=0$ | Nonhyperbolic. Linear & $J\ne0$: line of equilibria. Nonlinear: higher-order terms decide |

- $\tau=\operatorname{tr}J$, $\Delta=\det J$ → $\lambda=\frac{\tau\pm\sqrt{\tau^2-4\Delta}}{2}$.
```

```{note} Example
:class: dropdown
Damped pendulum $\ddot\theta=-\sin\theta-\gamma\dot\theta$, state $(\theta,\dot\theta)$, $0<\gamma<2$:
- $(0,0)$: $J=\begin{bmatrix}0&1\\-1&-\gamma\end{bmatrix}$, $\tau=-\gamma$, $\Delta=1$ → stable spiral (hangs down).
- $(\pi,0)$: $J=\begin{bmatrix}0&1\\1&-\gamma\end{bmatrix}$, $\Delta=-1$ → saddle (balanced upright).
```

```{attention} Q&A
:class: dropdown
*Why is linearization inconclusive at $\operatorname{Re}\lambda=0$?*
- Higher-order terms decide. $\dot x=-x^3$ (stable) & $\dot x=x^3$ (unstable) both have $J=0$.

*Minima vs saddles under gradient flow?*
- $\dot{\mathbf{x}}=-\nabla\mathcal{L}$ → $J=-H$, $H$ = Hessian of $\mathcal{L}$.
- Strict minimum ($H\succ0$) → stable. Strict saddle ($H$ has a negative eigenvalue) → unstable.
- Stable manifold of a strict saddle has measure 0 → almost every initial state escapes it.

*Discrete-time version?*
- $\mathbf{x}_{k+1}=\mathbf{g}(\mathbf{x}_k)$: fixed point stable if spectral radius of $\frac{\partial\mathbf{g}}{\partial\mathbf{x}}<1$.
- GD at a minimum: $\frac{\partial\mathbf{g}}{\partial\mathbf{x}}=I-\eta H$ → stable ⇔ $0<\eta<\frac{2}{\lambda_{\max}(H)}$ ($\eta$: LR).
```

&nbsp;

### Lyapunov Function
- **What**: Energy-like $V(\mathbf{x})$ that never increases along trajectories.
- **Why**: Proves stability w/o solving the ODE. Works where linearization is inconclusive.
- **How**:
    1. Guess $V\ge0$, zero only at $\mathbf{x}^*$.
    2. Compute $\dot V=\nabla V^T\mathbf{f}$.
    3. $\dot V\le0$ → stable. $\dot V<0$ away from $\mathbf{x}^*$ → asymptotically stable.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $V:\mathcal{U}\to\mathbb{R}$: $C^1$, $V(\mathbf{x}^*)=0$, $V(\mathbf{x})>0$ for $\mathbf{x}\ne\mathbf{x}^*$.
    - $\mathcal{U}$: Neighborhood of $\mathbf{x}^*$.

Theorem:
- $\dot V(\mathbf{x})=\nabla V(\mathbf{x})^T\mathbf{f}(\mathbf{x})\le0$ on $\mathcal{U}$ → Lyapunov stable.
- $\dot V(\mathbf{x})<0$ for $\mathbf{x}\ne\mathbf{x}^*$ → asymptotically stable.
- + $\mathcal{U}=\mathbb{R}^d$ & $V\to\infty$ as $\lVert\mathbf{x}\rVert\to\infty$ → globally asymptotically stable.
- **LaSalle**: $\dot V\le0$ → bounded trajectories approach the largest invariant set inside $\{\dot V=0\}$.
```

```{note} Example
:class: dropdown
- $\dot x=-x^3$, $V=x^2$ → $\dot V=-2x^4<0$ → asymptotically stable (linearization fails: $J=0$).
- Damped pendulum, $V=\frac12\dot\theta^2+1-\cos\theta$ (energy) → $\dot V=-\gamma\dot\theta^2\le0$ → stable.
    - $\{\dot V=0\}=\{\dot\theta=0\}$. Only invariant subset near $0$ is the equilibrium → LaSalle → asymptotically stable.
```

```{attention} Q&A
:class: dropdown
*How to find $V$?*
- Physical energy.
- Loss $\mathcal{L}-\mathcal{L}^*$ for gradient flows.
- Linear systems: $V=\mathbf{x}^TP\mathbf{x}$ w/ $A^TP+PA=-Q$ (Lyapunov equation). Solvable w/ $P\succ0$ for any $Q\succ0$ ⇔ all $\operatorname{Re}\lambda(A)<0$.

*Converse?*
- Asymptotic stability → a Lyapunov function exists (converse theorems). No general recipe to build it.

*Link to ML & AC?*
- Training loss = Lyapunov function of gradient flow.
- Hopfield networks: energy = Lyapunov function → state settles into an attractor = stored memory ([Attractor](#attractor)).
```

&nbsp;

### Gradient Flow
- **What**: $\dot{\mathbf{x}}=-\nabla\mathcal{L}(\mathbf{x})$.
- **Why**: Continuous-time limit of GD ($\eta\to0$) → analyze optimizers w/ calculus instead of step-by-step algebra.
- **How**:
    - GD = forward [Euler](#euler-method) on gradient flow w/ $h=\eta$.
    - Loss = Lyapunov function: $\dot{\mathcal{L}}=-\lVert\nabla\mathcal{L}\rVert^2\le0$.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\mathbf{x}$: Params (state of the flow).
- Hyperparams:
    - $\eta$: LR.
    - $\gamma$: Friction coeff.
- Misc:
    - $\mathcal{L}:\mathbb{R}^d\to\mathbb{R}$: Objective.
    - $\mathbf{x}^*$: Minimizer. $\mathcal{L}^*=\mathcal{L}(\mathbf{x}^*)$.
    - $\mu$: PL / strong-convexity constant.

Rates:
- Convex:

    $$
    \mathcal{L}(\mathbf{x}(t))-\mathcal{L}^*\le\frac{\lVert\mathbf{x}_0-\mathbf{x}^*\rVert^2}{2t}
    $$

- $\mu$-PL ($\lVert\nabla\mathcal{L}\rVert^2\ge2\mu(\mathcal{L}-\mathcal{L}^*)$; implied by $\mu$-strong convexity):

    $$
    \mathcal{L}(\mathbf{x}(t))-\mathcal{L}^*\le e^{-2\mu t}\big(\mathcal{L}(\mathbf{x}_0)-\mathcal{L}^*\big)
    $$

Second-order flows:
- Heavy ball (continuous [momentum](../dl/train/optim.md#momentum)): $\ddot{\mathbf{x}}+\gamma\dot{\mathbf{x}}+\nabla\mathcal{L}(\mathbf{x})=\mathbf{0}$.
- Nesterov ODE {cite:p}`su2014differential`:

    $$
    \ddot{\mathbf{x}}+\frac3t\dot{\mathbf{x}}+\nabla\mathcal{L}(\mathbf{x})=\mathbf{0},\quad\dot{\mathbf{x}}(0)=\mathbf{0}
    \quad\Rightarrow\quad
    \mathcal{L}(\mathbf{x}(t))-\mathcal{L}^*\le\frac{2\lVert\mathbf{x}_0-\mathbf{x}^*\rVert^2}{t^2}\ \text{(convex)}
    $$

SGD as an SDE (weak approximation, $t=k\eta$) {cite:p}`li2017stochastic`:

$$
d\mathbf{x}=-\nabla\mathcal{L}(\mathbf{x})\,dt+\sqrt{\eta}\,\Sigma(\mathbf{x})^{1/2}\,d\mathbf{W}_t
$$

- $\Sigma(\mathbf{x})$: Covariance of the minibatch gradient noise.
```

```{tip} Derivation
:class: dropdown
*Convex rate*: let $E(t)=t(\mathcal{L}-\mathcal{L}^*)+\frac12\lVert\mathbf{x}-\mathbf{x}^*\rVert^2$.
1. $\dot E=(\mathcal{L}-\mathcal{L}^*)-t\lVert\nabla\mathcal{L}\rVert^2-\nabla\mathcal{L}^T(\mathbf{x}-\mathbf{x}^*)$.
2. Convexity: $\mathcal{L}-\mathcal{L}^*\le\nabla\mathcal{L}^T(\mathbf{x}-\mathbf{x}^*)$ → $\dot E\le-t\lVert\nabla\mathcal{L}\rVert^2\le0$.
3. $t(\mathcal{L}-\mathcal{L}^*)\le E(t)\le E(0)=\frac12\lVert\mathbf{x}_0-\mathbf{x}^*\rVert^2$.

*PL rate*:
1. $\frac{d}{dt}(\mathcal{L}-\mathcal{L}^*)=-\lVert\nabla\mathcal{L}\rVert^2\le-2\mu(\mathcal{L}-\mathcal{L}^*)$.
2. Grönwall → $e^{-2\mu t}$ decay.

*Nesterov rate*: let $E(t)=t^2(\mathcal{L}-\mathcal{L}^*)+2\left\lVert\mathbf{x}+\frac t2\dot{\mathbf{x}}-\mathbf{x}^*\right\rVert^2$.
1. ODE → $\frac{d}{dt}\left(\mathbf{x}+\frac t2\dot{\mathbf{x}}\right)=\frac32\dot{\mathbf{x}}+\frac t2\ddot{\mathbf{x}}=-\frac t2\nabla\mathcal{L}$.
2. $\dot E=2t(\mathcal{L}-\mathcal{L}^*)+t^2\nabla\mathcal{L}^T\dot{\mathbf{x}}-2t\nabla\mathcal{L}^T\left(\mathbf{x}+\frac t2\dot{\mathbf{x}}-\mathbf{x}^*\right)$.
3. $=2t\left[(\mathcal{L}-\mathcal{L}^*)-\nabla\mathcal{L}^T(\mathbf{x}-\mathbf{x}^*)\right]\le0$ (convexity).
4. $t^2(\mathcal{L}-\mathcal{L}^*)\le E(0)=2\lVert\mathbf{x}_0-\mathbf{x}^*\rVert^2$.

*Momentum → heavy ball*:
1. Momentum ($\beta$: momentum coeff): $\mathbf{x}_{k+1}-\mathbf{x}_k=\beta(\mathbf{x}_k-\mathbf{x}_{k-1})-\eta\nabla\mathcal{L}(\mathbf{x}_k)$.
2. Rearrange: $(\mathbf{x}_{k+1}-2\mathbf{x}_k+\mathbf{x}_{k-1})+(1-\beta)(\mathbf{x}_k-\mathbf{x}_{k-1})=-\eta\nabla\mathcal{L}$.
3. Divide by $\eta=h^2$: 1st term → $\ddot{\mathbf{x}}$, 2nd → $\frac{1-\beta}{h}\dot{\mathbf{x}}$.
4. → $\gamma=\frac{1-\beta}{\sqrt\eta}$, time $t=k\sqrt\eta$.
5. Finite-friction limit needs coupled scaling: hold $\gamma$ fixed → $\beta=1-\gamma\sqrt\eta\to1$ as $\eta\to0$.
    - Fixed $\beta<1$ → $\gamma\to\infty$ → overdamped → gradient flow in time $\frac{k\eta}{1-\beta}$.
```

```{attention} Q&A
:class: dropdown
*Why is time $k\eta$ for GD but $k\sqrt\eta$ for momentum?*
- GD: displacement per step ∝ $\eta$ → 1st-order ODE, $h=\eta$.
- Momentum: velocity persists → 2nd-order ODE → acceleration per step ∝ $\eta=h^2$.

*When is gradient flow a bad model of GD?*
- Large $\eta$. Flow converges for any curvature. GD needs $\eta<\frac{2}{\lambda_{\max}(H)}$ ($H$: Hessian at the minimum) ← Euler stability ([Euler Method](#euler-method)).

*Why is Nesterov faster than GD on convex problems?*
- $O(1/t^2)$ vs $O(1/t)$ in ODE time.
- Damping $\frac3t$ vanishes → little friction late → the state keeps speed along flat directions.
- Damping $\frac rt$: $r\ge3$ guarantees $O(1/t^2)$; smaller $r$ can fail.

*What does the SGD SDE tell us?*
- Minibatch noise covariance $\Sigma\propto\frac1B$ ($B$: batch size) → noise scale ∝ $\frac\eta B$.
- → LR & batch size trade off. Larger $\eta$ → wider spread around minima.
```

&nbsp;

### Attractor
- **What**: Invariant set that attracts all trajectories from an open neighborhood (its basin).
- **Why**: Long-run behavior of a dissipative system = its attractors → memory states, rhythms, chaos.
- **How**: Classify by the dimension of the long-run motion (table).

```{dropdown} Table: Attractor Types
| Type | Long-run motion | Example |
|:--|:--|:--|
| Fixed point | Rest | Loss minimum under gradient flow; Hopfield memory |
| Stable limit cycle | Isolated periodic orbit that attracts nearby trajectories | Van der Pol $\ddot x-\nu(1-x^2)\dot x+x=0$ ($\nu>0$); heartbeat |
| Torus | Quasi-periodic (incommensurate frequencies) | 2 coupled oscillators |
| Strange attractor | Chaotic, fractal set | Lorenz system ([Chaos](#chaos)) |

- $\nu$: Van der Pol damping parameter.
```

```{attention} Q&A
:class: dropdown
*Limit cycle vs center?*
- Center: continuous family of closed orbits; amplitude set by the initial state; destroyed by small perturbations.
- Limit cycle: isolated periodic orbit. Can be stable, unstable or semistable.
    - Stable: nearby trajectories converge to it → robust oscillation.
    - Unstable: $\dot r=r(r^2-1)$, $\dot\theta=1$ → cycle $r=1$ repels.

*What does it mean for intelligence/AC?*
- Attractor networks model memory: fixed points = stored patterns; recall = flow into the basin.
- Continuous (line/ring) attractors are proposed models for working memory & head direction.
```

&nbsp;

#### Chaos
- **What**: Bounded, aperiodic, deterministic dynamics w/ sensitive dependence on initial conditions.
- **Why**: Determinism ≠ predictability: tiny errors grow exponentially → finite prediction horizon.
- **How**: Stretch (nearby states separate) + fold (state stays bounded) → repeat.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\boldsymbol{\delta}(t)=D\varphi_t(\mathbf{x}_0)\,\boldsymbol{\delta}(0)$: Infinitesimal perturbation, evolved by the linearized flow.
    - $D\varphi_t(\mathbf{x}_0)$: Jacobian of the flow map.
    - $\lambda_{\max}$: Max Lyapunov exponent.

Definition:

$$
\lambda_{\max}(\mathbf{x}_0)=\limsup_{t\to\infty}\frac1t\ln\lVert D\varphi_t(\mathbf{x}_0)\rVert
$$

- Generic $\boldsymbol{\delta}(0)$ → $\frac1t\ln\frac{\lVert\boldsymbol{\delta}(t)\rVert}{\lVert\boldsymbol{\delta}(0)\rVert}\to\lambda_{\max}$. Special directions can give a smaller exponent.
- Chaos = bounded & aperiodic & $\lambda_{\max}>0$.
    - $\lambda_{\max}>0$ alone ❌: $\dot x=x-x^3$ at the unstable equilibrium $x_0=0$ has exponent 1, no chaos.
- Prediction horizon $\approx\frac{1}{\lambda_{\max}}\ln\frac{\text{tolerance}}{\text{initial error}}$ → 10× better measurement → only $+\frac{\ln10}{\lambda_{\max}}$ time.
```

```{note} Example
:class: dropdown
Lorenz system:

$$
\dot x_1=\sigma(x_2-x_1),\quad\dot x_2=x_1(\rho-x_3)-x_2,\quad\dot x_3=x_1x_2-\beta x_3
$$

- $\sigma=10$, $\rho=28$, $\beta=\frac83$ → strange attractor, $\lambda_{\max}\approx0.906$.

Logistic map $x_{k+1}=rx_k(1-x_k)$, $r=4$ → chaotic, $\lambda=\ln2$.
- Discrete maps can be chaotic in 1D. Continuous flows need $d\ge3$ ([Phase Portrait](#phase-portrait)).
```

```{attention} Q&A
:class: dropdown
*How does chaos relate to RNN training?*
- RNN hidden state = discrete dynamical system. Backprop multiplies the same Jacobians as $\boldsymbol\delta$.
- $\lambda_{\max}>0$ → perturbations & gradients grow exponentially → [exploding gradients](../dl/train/issues.md#vanishing-exploding-gradient).
- $\lambda_{\max}<0$ → information decays exponentially → vanishing gradients, short memory.
- → Initialize near $\lambda_{\max}\approx0$ ("edge of chaos").
```

&nbsp;

### Bifurcation
- **What**: Qualitative change in the phase portrait as a parameter crosses a critical value.
    - Equilibria or cycles appear, disappear or change stability.
- **Why**: Smooth parameter change → abrupt behavior change (onset of oscillation, loss of a memory state, tipping points).
- **How**: At the critical value, an equilibrium becomes nonhyperbolic.
    - Real eigenvalue crosses 0 → saddle-node, transcritical or pitchfork.
    - Complex pair crosses the imaginary axis → Hopf.

```{dropdown} Table: Normal Forms
| Type | Normal form | Change as $r$ crosses 0 |
|:--|:--|:--|
| Saddle-node | $\dot x=r+x^2$ | $r<0$: stable $-\sqrt{-r}$ & unstable $+\sqrt{-r}$ → collide → $r>0$: no equilibrium |
| Transcritical | $\dot x=rx-x^2$ | Equilibria $0$ & $r$ exchange stability |
| Supercritical pitchfork | $\dot x=rx-x^3$ | $0$ stable → $0$ unstable + 2 stable $\pm\sqrt r$ (symmetry breaking) |
| Supercritical Hopf | $\dot\rho=r\rho-\rho^3$, $\dot\theta=\omega$ | Stable spiral → unstable spiral + stable limit cycle of radius $\sqrt r$ |

- $r$: Bifurcation parameter.
- $\rho,\theta$: Polar coordinates (local override).
- $\omega$: Rotation frequency.
- Subcritical versions (sign of the cubic flipped) → jump to a distant state & hysteresis.
```

```{attention} Q&A
:class: dropdown
*Why does a bifurcation happen only at a nonhyperbolic point?*
- Hyperbolic equilibrium → Hartman–Grobman + implicit function theorem → it persists w/ the same type under small parameter changes.

*How does it relate to RNN training?*
- Weights = parameters of the hidden-state dynamics. Crossing a bifurcation during training → attractors change abruptly → state Jacobians change sharply → gradient spikes.

*What does it mean for intelligence/AC?*
- Pitchfork = symmetry breaking → a decision between 2 equivalent options.
- Hopf = onset of rhythm (neural oscillations).
```

&nbsp;

## Numerical Methods

### Euler Method
- **What**: $\mathbf{x}_{k+1}=\mathbf{x}_k+h\,\mathbf{f}(\mathbf{x}_k,t_k)$.
- **Why**: Simplest solver: follow the tangent for 1 step.
    - Baseline for every other method. Same update as a ResNet block & GD.
- **How**:
    1. Evaluate the slope at the current point.
    2. Step $h$ along it.
    3. Repeat.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $T=t_N-t_0$: Integration horizon, $N$ steps.
    - $M=\max\lVert\ddot{\mathbf{x}}\rVert$ on $[t_0,t_N]$.

Error:
- Local truncation error: $\mathbf{x}(t_{k+1})-\mathbf{x}(t_k)-h\mathbf{f}(\mathbf{x}(t_k),t_k)=\frac{h^2}{2}\ddot{\mathbf{x}}(\xi)=O(h^2)$, $\xi\in(t_k,t_{k+1})$ (componentwise).
- Global error: $\lVert\mathbf{x}(t_N)-\mathbf{x}_N\rVert\le\frac{Mh}{2L}\left(e^{LT}-1\right)=O(h)$ → order 1.

Stability (test equation $\dot x=\lambda x$):

$$
x_{k+1}=(1+h\lambda)\,x_k\quad\Rightarrow\quad\text{stable}\Leftrightarrow\lvert1+h\lambda\rvert\le1
$$

- Region: disk of radius 1 centered at $-1$. Real $\lambda<0$ → $h\le\frac{2}{\lvert\lambda\rvert}$.
```

```{tip} Derivation
:class: dropdown
*Why is the global error $O(h)$ if each step is $O(h^2)$?*
1. Error $\mathbf{e}_k=\mathbf{x}(t_k)-\mathbf{x}_k$ → $\mathbf{e}_{k+1}=\mathbf{e}_k+h\big(\mathbf{f}(\mathbf{x}(t_k))-\mathbf{f}(\mathbf{x}_k)\big)+\boldsymbol{\tau}_{k+1}$, $\boldsymbol{\tau}$ = local error.
2. Lipschitz: $\lVert\mathbf{e}_{k+1}\rVert\le(1+hL)\lVert\mathbf{e}_k\rVert+\frac{Mh^2}{2}$.
3. Unroll: $\lVert\mathbf{e}_N\rVert\le\frac{Mh^2}{2}\cdot\frac{(1+hL)^N-1}{hL}\le\frac{Mh}{2L}(e^{LT}-1)$.
4. $N=T/h$ steps × $O(h^2)$ each → $O(h)$, amplified by at most $e^{LT}$.
```

````{important} Code
:class: dropdown
```python
import torch

class Euler:
    def __init__(self, f):
        self.f = f  ## vector field f(x, t)

    def step(self, x, t, h):
        ## Follow the tangent at the current point
        return x + h * self.f(x, t)

    def solve(self, x0, t0, t1, n_steps):
        ## h < 0 integrates backward in time
        h = (t1 - t0) / n_steps
        x, t = x0, t0
        for _ in range(n_steps):
            x, t = self.step(x, t, h), t + h
        return x

## Example: dx/dt = -x, x(0) = 1 -> x(1) = e^{-1}
f = lambda x, t: -x
x0 = torch.tensor([1.0], dtype=torch.float64)
for n in [10, 100, 1000]:
    err = (Euler(f).solve(x0, 0.0, 1.0, n) - torch.exp(-x0)).abs().item()
    print(n, err)  ## ~1.9e-2, ~1.8e-3, ~1.8e-4 -> 10x steps, 10x smaller: order 1
```
````

```{attention} Q&A
:class: dropdown
*How does Euler relate to ResNets & GD?*
- ResNet block $\mathbf{x}_{l+1}=\mathbf{x}_l+F(\mathbf{x}_l)$ = Euler step w/ $h=1$ → [Neural ODE](#neural-ode).
- GD = Euler on [gradient flow](#gradient-flow) w/ $h=\eta$ → GD stability limit $\eta<\frac{2}{\lambda_{\max}(H)}$ = Euler stability, $H$ = Hessian.

*Why not always use Euler?*
- Order 1 → error $10^{-4}$ needs ~$10^4$ steps. [Runge–Kutta](#runge-kutta-methods) needs far fewer.
- Explicit → small stability region → fails on [stiff](#stiffness) problems.
```

&nbsp;

#### Implicit Euler
- **What**: $\mathbf{x}_{k+1}=\mathbf{x}_k+h\,\mathbf{f}(\mathbf{x}_{k+1},t_{k+1})$.
- **Why**: Explicit Euler blows up on stiff ODEs unless $h$ is tiny. Implicit Euler is stable for every $h$ on decaying modes.
- **How**: Solve for $\mathbf{x}_{k+1}$ each step: Newton for nonlinear $\mathbf{f}$; 1 linear solve for linear $\mathbf{f}$.

```{note} Math
:class: dropdown
Linear system: $\mathbf{x}_{k+1}=(I-hA)^{-1}\mathbf{x}_k$.

Test equation:

$$
x_{k+1}=\frac{1}{1-h\lambda}\,x_k
$$

- $\left\lvert\frac{1}{1-h\lambda}\right\rvert\le1$ for all $\operatorname{Re}\lambda\le0$ & all $h>0$ → **A-stable**.
- $h\lambda\to-\infty$ → factor $\to0$ → fast modes die in 1 step (**L-stable**).
- Order 1.
```

```{attention} Q&A
:class: dropdown
*Cost?*
- 1 linear solve or Newton iteration per step → expensive per step, cheap overall on stiff problems.

*Downside on oscillations?*
- $\lambda=i\omega$ → $\lvert\frac{1}{1-ih\omega}\rvert<1$ → artificial damping (energy loss).
- Trapezoid / bilinear: factor $\frac{1+h\lambda/2}{1-h\lambda/2}$ → A-stable & modulus exactly 1 on the imaginary axis → used by S4 ([Discretization](../nlp/ssm.md#discretization)).
```

&nbsp;

### Stiffness
- **What**: Property of an ODE whose fast decaying modes force explicit solvers to take steps far smaller than accuracy needs.
- **Why**: Multi-scale systems (chemistry, circuits, some learned dynamics) → explicit step count explodes.
- **How**:
    - Detect: Jacobian eigenvalues spread widely; max $\lvert\operatorname{Re}\lambda\rvert$ ≫ $\frac1{\text{timescale of interest}}$.
    - Fix: implicit methods (implicit Euler, BDF, implicit RK) or exponential integrators (solve the linear part exactly).

```{note} Example
:class: dropdown
$\dot x=-1000\,(x-\cos t)$:
- After a transient of ~$10^{-3}$, $x\approx\cos t$ → smooth, slowly varying.
- Explicit Euler still needs $h<\frac{2}{1000}$ on the whole interval ← stability, not accuracy.
- Implicit Euler: $h=0.1$ is stable & tracks $\cos t$.
```

```{attention} Q&A
:class: dropdown
*How does an adaptive explicit solver behave on a stiff problem?*
- Step controller shrinks $h$ to the stability limit → correct but very slow → typical symptom of stiffness.

*Where does stiffness appear in ML?*
- Diffusion probability-flow ODE: linear term + learned term → exponential integrators solve the linear part exactly → [DPM-Solver](../dl/gen/diffusion.md#dpm-solver).
- Neural ODEs: NFE can grow during training as the learned dynamics get harder ([Neural ODE](#neural-ode)).
```

&nbsp;

### Runge–Kutta Methods
- **What**: One-step methods that combine several slope evaluations inside each step for higher order.
- **Why**: Euler error $O(h)$ → many steps. Order $p$ → error $O(h^p)$ → far fewer steps for the same accuracy.
- **How**:
    1. Sample slopes at trial points inside $[t_k,t_k+h]$.
    2. Step w/ a weighted average of the slopes.
    3. Weights make the step match the Taylor series of the true solution up to $h^p$.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\mathbf{s}_i$: Slope at stage $i$.
    - $a_{ij},b_i,c_i$: Method coefficients (Butcher tableau).

Explicit $S$-stage RK:

$$
\mathbf{s}_i=\mathbf{f}\Big(\mathbf{x}_k+h\sum_{j<i}a_{ij}\mathbf{s}_j,\ t_k+c_ih\Big),\qquad
\mathbf{x}_{k+1}=\mathbf{x}_k+h\sum_{i=1}^Sb_i\mathbf{s}_i
$$

Heun (RK2, trapezoid predictor–corrector):

$$
\mathbf{s}_1=\mathbf{f}(\mathbf{x}_k,t_k),\quad\mathbf{s}_2=\mathbf{f}(\mathbf{x}_k+h\mathbf{s}_1,t_k+h),\quad\mathbf{x}_{k+1}=\mathbf{x}_k+\frac h2(\mathbf{s}_1+\mathbf{s}_2)
$$

RK4:

$$
\begin{align*}
\mathbf{s}_1&=\mathbf{f}(\mathbf{x}_k,t_k)\\
\mathbf{s}_2&=\mathbf{f}\left(\mathbf{x}_k+\tfrac h2\mathbf{s}_1,t_k+\tfrac h2\right)\\
\mathbf{s}_3&=\mathbf{f}\left(\mathbf{x}_k+\tfrac h2\mathbf{s}_2,t_k+\tfrac h2\right)\\
\mathbf{s}_4&=\mathbf{f}(\mathbf{x}_k+h\mathbf{s}_3,t_k+h)\\
\mathbf{x}_{k+1}&=\mathbf{x}_k+\tfrac h6(\mathbf{s}_1+2\mathbf{s}_2+2\mathbf{s}_3+\mathbf{s}_4)
\end{align*}
$$

Adaptive step (embedded pair, e.g., Dormand–Prince 5(4)):
- Error estimate = difference of the order-5 & order-4 results (shared slopes → no extra NFE).
- Accept if error ≤ tol. Next step: $h\leftarrow0.9\,h\left(\frac{\text{tol}}{\text{err}}\right)^{1/5}$.
```

```{dropdown} Table: Explicit Solvers
| Method | Order | NFE/step | Use |
|:--|:--|:--|:--|
| Euler | 1 | 1 | Baseline; DDIM ([DDIM](../dl/gen/diffusion.md#ddim)) |
| Heun | 2 | 2 | EDM sampler ([EDM](../dl/gen/diffusion.md#edm)) |
| Midpoint | 2 | 2 | $\mathbf{x}_{k+1}=\mathbf{x}_k+h\,\mathbf{f}\left(\mathbf{x}_k+\frac h2\mathbf{s}_1,t_k+\frac h2\right)$ |
| RK4 | 4 | 4 | Fixed-step default |
| Dormand–Prince 5(4) | 5 | 6 (7 stages, last reused) | Adaptive default: SciPy `RK45`, torchdiffeq `dopri5` |
```

````{important} Code
:class: dropdown
```python
import torch

## Reuses Euler (solve loop) from Euler Method above
class RK4(Euler):
    def step(self, x, t, h):
        ## Slopes at start, midpoint (twice), end -> weights 1-2-2-1 (Simpson's rule)
        s1 = self.f(x, t)
        s2 = self.f(x + h / 2 * s1, t + h / 2)
        s3 = self.f(x + h / 2 * s2, t + h / 2)
        s4 = self.f(x + h * s3, t + h)
        return x + h / 6 * (s1 + 2 * s2 + 2 * s3 + s4)

## Example: same ODE as Euler, dx/dt = -x
f = lambda x, t: -x
x0 = torch.tensor([1.0], dtype=torch.float64)
for n in [10, 100]:
    err = (RK4(f).solve(x0, 0.0, 1.0, n) - torch.exp(-x0)).abs().item()
    print(n, err)  ## ~3e-7, ~3e-11 -> 10x steps, 1e4x smaller: order 4
```
````

```{attention} Q&A
:class: dropdown
*Why is RK4 so popular?*
- 4 NFE → order 4. Beyond order 4, explicit RK needs more stages than its order (order 5 → 6 stages).

*RK vs multistep (Adams, BDF)?*
- Explicit Adams–Bashforth reuses past slopes → 1 new NFE/step after startup.
- Implicit (Adams–Moulton, BDF) → nonlinear solve per step → several NFE. E.g., BDF2: $3\mathbf{x}_{k+1}-4\mathbf{x}_k+\mathbf{x}_{k-1}=2h\,\mathbf{f}(\mathbf{x}_{k+1})$.
- Needs a startup phase & handles step-size changes poorly. RK is self-starting.

*Is higher order always better?*
- ❌ Non-smooth $\mathbf{f}$ → actual order drops.
- Stiff problems → stability limits $h$, not accuracy → use implicit methods.
```

&nbsp;

### Symplectic Integrator
- **What**: Solver for Hamiltonian systems whose step map preserves the symplectic structure (and hence phase-space volume).
- **Why**: Non-symplectic solvers drift in energy over long runs. Symplectic → energy error stays bounded for very long times.
- **How**: Leapfrog: half-kick the momentum → full drift of the position → half-kick again.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\mathbf{q}$: Position.
    - $\mathbf{p}$: Momentum.
    - $U(\mathbf{q})$: Potential energy.
    - $\mathcal{H}(\mathbf{q},\mathbf{p})=U(\mathbf{q})+\frac12\lVert\mathbf{p}\rVert^2$: Hamiltonian (unit mass).

Hamilton's equations: $\dot{\mathbf{q}}=\mathbf{p}$, $\dot{\mathbf{p}}=-\nabla U(\mathbf{q})$.

Leapfrog (Störmer–Verlet):

$$
\begin{align*}
\mathbf{p}_{k+1/2}&=\mathbf{p}_k-\tfrac h2\nabla U(\mathbf{q}_k)\\
\mathbf{q}_{k+1}&=\mathbf{q}_k+h\,\mathbf{p}_{k+1/2}\\
\mathbf{p}_{k+1}&=\mathbf{p}_{k+1/2}-\tfrac h2\nabla U(\mathbf{q}_{k+1})
\end{align*}
$$

- Order 2, time-reversible, volume-preserving (Jacobian determinant 1).
- Nearly conserves a truncated modified Hamiltonian $\tilde{\mathcal{H}}=\mathcal{H}+O(h^2)$ → energy error stays $O(h^2)$ w/o drift for exponentially long times.
    - Conditions: analytic $U$, small fixed $h$, bounded numerical trajectory. The full modified series usually diverges → ❌ exact conservation.
```

```{note} Example
:class: dropdown
Harmonic oscillator $U=\frac12q^2$, $h=0.1$:
- Explicit Euler: $(q,p)\to(q+hp,\ p-hq)$ → $q^2+p^2$ multiplies by $1+h^2$ each step → ×$1.01^{1000}\approx2\times10^4$ after 1000 steps.
- Leapfrog: energy oscillates within $O(h^2)$ of the true value.
```

```{attention} Q&A
:class: dropdown
*Why does HMC use leapfrog?*
- Volume-preserving & reversible → Metropolis acceptance needs no Jacobian term.
- Small energy error → high acceptance rate → [HMC](prob/sampling.md#hmc).

*Does a symplectic integrator conserve energy exactly?*
- ❌ It nearly conserves a truncated modified Hamiltonian. $\mathcal{H}$ oscillates within $O(h^2)$ but does not drift, for exponentially long times.
```

&nbsp;

## Partial Differential Equations
- **What**: Equations linking an unknown function of $\ge2$ variables to its partial derivatives.

### Classification
- **What**: Elliptic / parabolic / hyperbolic, by the sign of the discriminant of the 2nd-order part.
- **Why**: Type decides the data that make the problem well-posed (boundary vs initial) & how solutions behave (smooth, smoothing, propagating).
- **How**: Read the coefficients of the 2nd-order terms → sign of $b^2-4ac$ (table).

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $u(x,y)$: Unknown function of 2 independent variables $x,y$ (local override).
    - $a,b,c$: Coefficients of the 2nd-order terms.
    - $\nu$: Wave speed. $\mathbf{v}$: Transport velocity.
    - $\Delta=\sum_j\partial_j^2$: Laplacian.

2nd-order linear PDE:

$$
a\,u_{xx}+b\,u_{xy}+c\,u_{yy}+(\text{lower-order terms})=0
$$
```

```{dropdown} Table: PDE Types
| Type | Condition | Prototype | Behavior | ML link |
|:--|:--|:--|:--|:--|
| Elliptic | $b^2-4ac<0$ | Laplace $\Delta u=0$ | Equilibrium; smooth; boundary data | Harmonic functions |
| Parabolic | $b^2-4ac=0$ | Heat $\partial_tu=\Delta u$ | Smoothing; irreversible; initial + boundary data | [Fokker–Planck](prob/process.md#fokker-planck-equation), diffusion models |
| Hyperbolic | $b^2-4ac>0$ | Wave $\partial_t^2u=\nu^2\Delta u$ | Finite-speed propagation; no smoothing | — |
| 1st-order linear | — | Transport $\partial_tu+\mathbf{v}\cdot\nabla u=0$ | Values carried along characteristics | [Continuity Equation](#continuity-equation), flow matching |
| 1st-order nonlinear | — | Hamilton–Jacobi–Bellman | Value function of optimal control | Continuous-time RL |
```

```{attention} Q&A
:class: dropdown
*Why do PDEs appear on an ODE/SDE page?*
- 1 trajectory → ODE/SDE. Density of many trajectories → PDE.
- ODE → [Continuity Equation](#continuity-equation). SDE → [Fokker–Planck](prob/process.md#fokker-planck-equation).

*What is the method of characteristics?*
- Transport $\partial_tu+\mathbf{v}\cdot\nabla u=0$ → $u$ is constant along curves $\dot{\mathbf{x}}=\mathbf{v}$ → PDE reduces to ODEs.
```

&nbsp;

### Continuity Equation
- **What**: $\partial_tp+\nabla\cdot(p\,\mathbf{f})=0$.
- **Why**: Probability mass is conserved while particles move along $\dot{\mathbf{x}}=\mathbf{f}(\mathbf{x},t)$ → density evolution of any ODE.
- **How**: Mass change in a fixed region = −net outflow through its boundary → [divergence theorem](calc.md#divergence-theorem-gauss) → local form.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $p(\mathbf{x},t)$: Density of the state at time $t$.

Along a trajectory (instantaneous change of variables):

$$
\frac{d}{dt}\log p\big(\mathbf{x}(t),t\big)=-\nabla\cdot\mathbf{f}\big(\mathbf{x}(t),t\big)
$$
```

```{tip} Derivation
:class: dropdown
1. Fixed region $\Omega$: $\frac{d}{dt}\int_\Omega p\,d\mathbf{x}=-\oint_{\partial\Omega}p\,\mathbf{f}\cdot\mathbf{n}\,dS=-\int_\Omega\nabla\cdot(p\mathbf{f})\,d\mathbf{x}$.
    - $\mathbf{n}$: Outward unit normal.
2. $\Omega$ arbitrary → integrands equal → $\partial_tp=-\nabla\cdot(p\mathbf{f})$.
3. Along $\mathbf{x}(t)$: $\frac{d}{dt}p=\partial_tp+\nabla p\cdot\mathbf{f}=-\nabla\cdot(p\mathbf{f})+\nabla p\cdot\mathbf{f}=-p\,\nabla\cdot\mathbf{f}$.
4. Divide by $p$.
```

```{attention} Q&A
:class: dropdown
*Where does ML use it?*
- [Continuous Normalizing Flow](#continuous-normalizing-flow): exact log-likelihood via the trace integral.
- Probability-flow ODE has the same marginals as the diffusion SDE → [Score SDE](../dl/gen/diffusion.md#score-sde).
- Flow matching: velocity $\mathbf{v}$ generates path $p_t$ ⇔ $(p_t,\mathbf{v})$ satisfy this equation → [Flow Matching](../dl/gen/diffusion.md#flow-matching).

*What if $\nabla\cdot\mathbf{f}=0$?*
- Incompressible → density constant along trajectories → volume-preserving flow (Liouville; Hamiltonian systems).
```

&nbsp;

### Heat Equation
- **What**: $\partial_tu=D\,\Delta u$.
- **Why**: Prototype of diffusion: heat, concentration, density of Brownian particles.
    - VE forward noising in diffusion models = heat flow on the data density.
- **How**: $\Delta u$ ∝ (neighbor average − value) → each point relaxes toward its neighbors → smoothing.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $D>0$: Diffusion coeff.
- Misc:
    - $u(\mathbf{x},t)$: Field on $\mathbb{R}^d$ (temperature, density).
    - $\boldsymbol{\omega}$: Spatial frequency.

Solution on $\mathbb{R}^d$:

$$
u(\cdot,t)=G_t*u(\cdot,0),\qquad G_t(\mathbf{x})=(4\pi Dt)^{-d/2}\exp\left(-\frac{\lVert\mathbf{x}\rVert^2}{4Dt}\right)
$$

- $G_t$: Heat kernel = density of $\mathcal{N}(\mathbf{0},2Dt\,I)$ → Gaussian blur w/ variance $2Dt$.
- Fourier mode $e^{i\boldsymbol{\omega}\cdot\mathbf{x}}$ decays as $e^{-D\lVert\boldsymbol{\omega}\rVert^2t}$ → high frequencies die first.
- Grid w/ spacing $\delta$: $\Delta u(\mathbf{x})\approx\frac{2d}{\delta^2}\big(\text{avg of }2d\text{ neighbors}-u(\mathbf{x})\big)$ → [Finite Differences](calc.md#finite-differences).
```

```{tip} Derivation
:class: dropdown
1. Fourier transform in $\mathbf{x}$: $\partial_t\hat u=-D\lVert\boldsymbol{\omega}\rVert^2\hat u$ → $\hat u(\boldsymbol{\omega},t)=e^{-D\lVert\boldsymbol{\omega}\rVert^2t}\hat u(\boldsymbol{\omega},0)$.
2. Product in Fourier space = convolution in $\mathbf{x}$.
3. $e^{-\sigma^2\lVert\boldsymbol{\omega}\rVert^2/2}$ = Fourier transform of the $\mathcal{N}(\mathbf{0},\sigma^2I)$ density → $\sigma^2=2Dt$.
```

```{attention} Q&A
:class: dropdown
*Why is the backward heat equation ill-posed?*
- Reverse time → modes multiply by $e^{+D\lVert\boldsymbol{\omega}\rVert^2t}$ → tiny high-frequency noise explodes.
- Diffusion models do not invert the density PDE. They reverse the stochastic process w/ a learned score (extra information) → [Score SDE](../dl/gen/diffusion.md#score-sde).

*Link to Brownian motion?*
- Density of $\mathbf{x}_0+\mathbf{W}_t$ solves the heat equation w/ $D=\frac12$.

*Maximum principle?*
- Max of $u$ never increases, min never decreases → no new extremes → smoothing.
```

&nbsp;

## Stochastic Differential Equations
- **What**: $d\mathbf{x}=\mathbf{f}(\mathbf{x},t)\,dt+g(\mathbf{x},t)\,d\mathbf{W}_t$: ODE + Brownian forcing.
    - $g$: Scalar diffusion coeff. General: matrix $G(\mathbf{x},t)$ → correlated noise.
    - Brownian motion, Itô calculus, Euler–Maruyama, Fokker–Planck, OU & Langevin → [Stochastic Processes](prob/process.md#continuous-paths).

&nbsp;

## Neural Differential Equations

### Neural ODE
- **Name**: Neural Ordinary Differential Equation {cite:p}`chen2018neural`
- **What**: NN that parameterizes the vector field: $\dot{\mathbf{x}}=\mathbf{f}_\theta(\mathbf{x},t)$; output = ODE solution at $t_1$.
- **Why**: Continuous depth.
    - ResNet block $\mathbf{x}_{l+1}=\mathbf{x}_l+F(\mathbf{x}_l)$ = [Euler](#euler-method) step w/ $h=1$ → limit of many small steps = ODE.
    - Solver picks #steps per input → trade precision for speed.
    - [Adjoint](#adjoint-method) gradients → memory constant in #steps.
    - Continuous time → irregularly sampled time series.
    - Invertible flow → [Continuous Normalizing Flow](#continuous-normalizing-flow).
- **How**:
    1. Define $\mathbf{f}_\theta$ (MLP, CNN; may take $t$ as input).
    2. Forward: ODE solver integrates from $\mathbf{x}(t_0)$ = input to $t_1$.
    3. Backward: backprop through the solver steps, or solve the adjoint ODE.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\theta$: Params of $\mathbf{f}_\theta$, shared across all $t$.
- Misc:
    - $\mathcal{L}$: Loss on the output $\mathbf{x}(t_1)$.

Model:

$$
\frac{d\mathbf{x}}{dt}=\mathbf{f}_\theta(\mathbf{x},t),\qquad\mathbf{x}(t_0)=\text{input}
$$

Inference:

$$
\mathbf{x}(t_1)=\mathbf{x}(t_0)+\int_{t_0}^{t_1}\mathbf{f}_\theta(\mathbf{x}(t),t)\,dt=\operatorname{ODESolve}(\mathbf{f}_\theta,\mathbf{x}(t_0),t_0,t_1)
$$

Training: minimize $\mathcal{L}(\mathbf{x}(t_1))$; get $\frac{d\mathcal{L}}{d\theta}$ by backprop through the solver or by the [adjoint method](#adjoint-method).
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

## Reuses RK4 from Runge–Kutta Methods above
class ODEFunc(nn.Module):
    def __init__(self, dim, hidden=32):
        super().__init__()
        ## t is an extra input -> time-dependent vector field
        self.net = nn.Sequential(nn.Linear(dim + 1, hidden), nn.Tanh(), nn.Linear(hidden, dim))

    def forward(self, x, t):
        t_col = torch.full_like(x[:, :1], t)
        return self.net(torch.cat([x, t_col], dim=1))

class NeuralODE(nn.Module):
    def __init__(self, dim, n_steps=10):
        super().__init__()
        self.f = ODEFunc(dim)
        self.n_steps = n_steps

    def forward(self, x):
        ## Depth = time [0, 1]; "layers" = solver steps that share params
        return RK4(self.f).solve(x, 0.0, 1.0, self.n_steps)

## Example
model = NeuralODE(dim=2)
x = torch.randn(4, 2)
y = model(x)
y.sum().backward()  ## backprop through all RK4 stages -> memory O(#steps)
print(y.shape, model.f.net[0].weight.grad.shape)  ## torch.Size([4, 2]) torch.Size([32, 3])
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- Params shared across depth → fewer params. MNIST: 0.42% error w/ 0.22M params vs ResNet 0.41% w/ 0.60M.
- Memory constant in #steps (adjoint). Adaptive accuracy–speed trade-off.
- Natural for continuous-time data & invertible generative models.

*Cons?*
- Slow: NFE set by the adaptive solver; it grows during training as the dynamics get harder.
- Adjoint gradients can be inaccurate ([Adjoint Method](#adjoint-method)).
- Limited expressivity in a fixed dimension (next question).

*What can a Neural ODE not represent?*
- Flow map = homeomorphism → trajectories cannot cross ([Phase Portrait](#phase-portrait)).
- 1D: $x\mapsto-x$ is impossible ← trajectories from $1$ & $-1$ would have to cross.
- 2D: cannot pull an inner disk out of a surrounding ring.
- Fix: add extra zero-initialized dims → room to go around (Augmented Neural ODEs) {cite:p}`dupont2019augmented`.
- ResNets avoid the limit ← discrete steps can jump.

*Is a ResNet exactly an Euler-discretized ODE?*
- Only approximately: standard ResNets have different weights per block → time-dependent $\mathbf{f}$ w/ piecewise-constant $\theta(t)$, and $h=1$ is not small.

*How does it relate to SSMs & diffusion?*
- SSM = linear ODE w/ input → [State Space Model](../nlp/ssm.md#state-space-model).
- Probability-flow ODE = Neural ODE w/ a score-based vector field → [Score SDE](../dl/gen/diffusion.md#score-sde).
- Flow matching trains the same kind of ODE w/o simulation → [Flow Matching](../dl/gen/diffusion.md#flow-matching).
```

&nbsp;

#### Adjoint Method
- **What**: Get $\frac{d\mathcal{L}}{d\theta}$ by solving an ODE for $\mathbf{a}(t)=\frac{\partial\mathcal{L}}{\partial\mathbf{x}(t)}$ backward in time.
- **Why**: Backprop through the solver stores every intermediate state → memory $O(\#\text{steps})$. Adjoint → memory constant in #steps.
- **How**:
    1. Forward solve. Keep only $\mathbf{x}(t_1)$.
    2. Solve 3 ODEs jointly from $t_1$ back to $t_0$: reconstruct $\mathbf{x}(t)$, evolve $\mathbf{a}(t)$, accumulate $\frac{d\mathcal{L}}{d\theta}$.
    3. Each step needs only vector–Jacobian products (1 autograd call), never a full Jacobian.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\mathbf{a}(t)\in\mathbb{R}^d$: Adjoint state (column vector).

Process (backward from $t_1$ to $t_0$):

$$
\begin{align*}
\mathbf{a}(t_1)&=\frac{\partial\mathcal{L}}{\partial\mathbf{x}(t_1)}\\
\frac{d\mathbf{a}}{dt}&=-\left(\frac{\partial\mathbf{f}_\theta}{\partial\mathbf{x}}\right)^T\mathbf{a}(t)\\
\frac{d\mathcal{L}}{d\theta}&=\int_{t_0}^{t_1}\left(\frac{\partial\mathbf{f}_\theta}{\partial\theta}\right)^T\mathbf{a}(t)\,dt
\end{align*}
$$

- Jacobians evaluated at $(\mathbf{x}(t),t)$.
- $\mathbf{a}(t_0)=\frac{\partial\mathcal{L}}{\partial\mathbf{x}(t_0)}$ → gradient w.r.t. the input.
```

```{tip} Derivation
:class: dropdown
*Backprop through Euler → adjoint ODE*:
1. Forward: $\mathbf{x}_{k+1}=\mathbf{x}_k+h\,\mathbf{f}_\theta(\mathbf{x}_k)$.
2. Backprop: $\mathbf{a}_k=\left(\frac{\partial\mathbf{x}_{k+1}}{\partial\mathbf{x}_k}\right)^T\mathbf{a}_{k+1}=\mathbf{a}_{k+1}+h\left(\frac{\partial\mathbf{f}_\theta}{\partial\mathbf{x}}\right)^T\mathbf{a}_{k+1}$.
3. Rearrange: $\frac{\mathbf{a}_{k+1}-\mathbf{a}_k}{h}=-\left(\frac{\partial\mathbf{f}_\theta}{\partial\mathbf{x}}\right)^T\mathbf{a}_{k+1}$ → $h\to0$ → adjoint ODE.
4. Param grad: each step adds $h\left(\frac{\partial\mathbf{f}_\theta}{\partial\theta}\right)^T\mathbf{a}_{k+1}$ → $h\to0$ → integral.
```

```{attention} Q&A
:class: dropdown
*Is the adjoint gradient exact?*
- Exact for the continuous problem. Numerically ≠ gradient of the discretized forward pass ("optimize-then-discretize" vs "discretize-then-optimize").
- Backward reconstruction of $\mathbf{x}(t)$ can be inaccurate: forward contracts strongly → reverse expands errors. Fix: checkpoint states.

*Where does it come from?*
- Costate of Pontryagin's maximum principle (optimal control). Reverse-mode AD = its discrete counterpart.

*Cost?*
- 1 extra solve of an augmented system of size $2d+\lvert\theta\rvert$; 1 VJP per evaluation.
```

&nbsp;

### Continuous Normalizing Flow
- **What**: Normalizing flow whose invertible map is a Neural ODE flow; density tracked by the instantaneous change of variables. {cite:p}`chen2018neural`
- **Why**: Discrete flows need restricted layers (e.g., triangular Jacobians) for a cheap log-determinant.
    - Continuous time: log-det → integral of a trace → free-form $\mathbf{f}_\theta$.
- **How**:
    1. Base density $p_0=\mathcal{N}(\mathbf{0},I)$ at $t=0$, data at $t=1$.
    2. Data point → integrate state & log-density change backward to $t=0$.
    3. Train by MLE on $\log p_1(\text{data})$.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $p_t$: Density of $\mathbf{x}(t)$.
    - $\boldsymbol{\varepsilon}$: Probe vector w/ $\mathbb{E}[\boldsymbol{\varepsilon}\boldsymbol{\varepsilon}^T]=I$ (Gaussian or Rademacher).

Model (instantaneous change of variables, [Continuity Equation](#continuity-equation)):

$$
\frac{d\log p_t(\mathbf{x}(t))}{dt}=-\operatorname{tr}\left(\frac{\partial\mathbf{f}_\theta}{\partial\mathbf{x}(t)}\right)
$$

Training (MLE):

$$
\log p_1(\mathbf{x}(1))=\log p_0(\mathbf{x}(0))-\int_0^1\operatorname{tr}\left(\frac{\partial\mathbf{f}_\theta}{\partial\mathbf{x}(t)}\right)dt
$$

Hutchinson trace estimator:

$$
\operatorname{tr}(J)=\mathbb{E}_{\boldsymbol{\varepsilon}}\left[\boldsymbol{\varepsilon}^TJ\boldsymbol{\varepsilon}\right]
$$

- $J$: Jacobian $\frac{\partial\mathbf{f}_\theta}{\partial\mathbf{x}}$.
- 1 VJP per sample vs $d$ for the exact trace → scalable CNFs {cite:p}`grathwohl2019ffjord`.
```

```{tip} Derivation
:class: dropdown
*Discrete view*:
1. Euler step $\mathbf{x}\mapsto\mathbf{x}+h\mathbf{f}_\theta(\mathbf{x})$ has Jacobian $I+hJ$.
2. Change of variables: $\log p$ drops by $\log\det(I+hJ)=h\operatorname{tr}J+O(h^2)$.
3. Sum over steps, $h\to0$ → $-\int\operatorname{tr}J\,dt$.
- Continuous derivation: [Continuity Equation](#continuity-equation).
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

## Reuses ODEFunc from Neural ODE & RK4 from Runge–Kutta Methods above
class CNF(nn.Module):
    def __init__(self, dim, n_steps=10):
        super().__init__()
        self.f = ODEFunc(dim)
        self.n_steps = n_steps
        self.base = torch.distributions.Normal(0.0, 1.0)

    def aug_f(self, s, t):
        ## Augmented state s = [x, delta]; d(delta)/dt = -tr(df/dx)
        x = s[:, :-1]
        dx = self.f(x, t)
        ## Exact trace: d autograd calls (Hutchinson: 1 call)
        tr = sum(torch.autograd.grad(dx[:, i].sum(), x, create_graph=True)[0][:, i]
                 for i in range(x.shape[1]))
        return torch.cat([dx, -tr[:, None]], dim=1)

    def log_prob(self, x):
        ## x needs grad so the trace can differentiate f w.r.t. it
        s1 = torch.cat([x.detach().requires_grad_(True), torch.zeros(x.shape[0], 1)], dim=1)
        ## Integrate data (t=1) back to base (t=0); delta(0) = int_0^1 tr dt
        s0 = RK4(self.aug_f).solve(s1, 1.0, 0.0, self.n_steps)
        z, delta = s0[:, :-1], s0[:, -1]
        return self.base.log_prob(z).sum(dim=1) - delta

## Example
cnf = CNF(dim=2)
x = torch.randn(5, 2)
lp = cnf.log_prob(x)
(-lp.mean()).backward()  ## MLE step
print(lp.shape)  ## torch.Size([5])
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- Free-form architecture. Exact likelihood. Invertible by integrating backward.

*Cons?*
- MLE needs a full ODE solve (+ trace) per training step → slow.
- → Flow matching trains the same ODE model w/o simulation → [Flow Matching](../dl/gen/diffusion.md#flow-matching).

*Link to diffusion?*
- Exact likelihood of the probability-flow ODE = this formula → [Score SDE](../dl/gen/diffusion.md#score-sde).
```

&nbsp;
