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
# Optimization
Search for the params &/or hyperparams minimizing an objective.

An [objective](obj.md) states *what* is good; optimization is the only thing that turns that statement into a fitted model.

This page covers solvers applicable across models & objectives. Model-specific solvers (e.g. SMO for [SVM](supervised.md#svm)) live on the model's page, and [Gradient Descent & family](../dl/optim.md) has its own.

Study notes from {cite:t}`nocedal2006numerical`, {cite:t}`garnett2023bayesian` & {cite:t}`eiben2015introduction`.

Default notations:
- $f:\mathbb{R}^n\to\mathbb{R}$: Objective, minimized by convention.
- $\mathbf{x}\in\mathbb{R}^n$: Decision vector — params or hyperparams, NOT the model input (data is already absorbed into $f$).
- $\mathbf{x}^*$: Minimizer.
- $t$: Iteration index.
- $\mathbf{g}_t=\nabla f(\mathbf{x}_t)$: Gradient.
- $H_t=\nabla^2f(\mathbf{x}_t)$: Hessian.

&nbsp;

## Foundations

### Convexity
- **What**: Every chord lies above the graph.
- **Why**: The only structural property that makes "local min ⇒ global min" free.
    - w/o it: the solution depends on init, restarts are mandatory, and no solver can certify it is done.
    - w/ it: any stationary point is the answer.
- **How**: Three equivalent tests, in increasing smoothness requirements.
    1. **0th order**: the segment between any two points on the graph never dips below it.
    2. **1st order**: the graph lies above every tangent hyperplane.
    3. **2nd order**: Hessian PSD everywhere.
    4. **Calculus of convexity**: preserved by nonneg weighted sums, pointwise max/sup, and affine precomposition → assemble convex objectives from convex atoms instead of re-proving each one.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $f:\mathcal{C}\to\mathbb{R}$: Objective on a convex domain $\mathcal{C}$.
- Misc:
    - $\lambda\in[0,1]$: Interpolation weight.
    - $\mu>0$: Strong-convexity modulus.
    - $L>0$: Lipschitz constant of $\nabla f$ (smoothness).

0th order (definition):

$$
f(\lambda\mathbf{x}+(1-\lambda)\mathbf{y})\leq\lambda f(\mathbf{x})+(1-\lambda)f(\mathbf{y})\quad\forall\mathbf{x},\mathbf{y}\in\mathcal{C}
$$

1st order:

$$
f(\mathbf{y})\geq f(\mathbf{x})+\nabla f(\mathbf{x})^T(\mathbf{y}-\mathbf{x})
$$

2nd order:

$$
\nabla^2f(\mathbf{x})\succeq0
$$

Strong convexity ($\Leftrightarrow\nabla^2f\succeq\mu I$):

$$
f(\mathbf{y})\geq f(\mathbf{x})+\nabla f(\mathbf{x})^T(\mathbf{y}-\mathbf{x})+\frac{\mu}{2}||\mathbf{y}-\mathbf{x}||_2^2
$$

$L$-smoothness ($\Leftrightarrow\nabla^2f\preceq LI$):

$$
||\nabla f(\mathbf{x})-\nabla f(\mathbf{y})||_2\leq L||\mathbf{x}-\mathbf{y}||_2
$$

Condition number:

$$
\kappa=\frac{L}{\mu}
$$
```

```{attention} Q&A
:class: dropdown
*Pros?*
- Local min = global min → ❌restarts, ❌init sensitivity.
- Stationarity is necessary AND sufficient → a clean stopping test.
- Argmin set is convex.
- Duality gap is computable → certified optimality.

*Does convex ⇒ unique minimizer?*
- ❌. Convex → the argmin **set** is convex: possibly a flat valley (e.g. $m<n$ least squares), possibly empty ($e^{-x}$).
- Strictly convex → at most one minimizer.
- Strongly convex + closed domain → exactly one, and $\kappa$ bounds the convergence rate.

*Which ML objectives are convex?*
- ✅ OLS, Ridge, Lasso, Elastic Net, logistic/softmax regression, SVM (hinge + L2), any GLM w/ canonical link, quantile regression.
- ❌ NNs, GMM likelihood, K-Means, NMF jointly in $(W,H)$, matrix factorization, tree structure search, anything with a discrete search space.

*Why care when deep learning is non-convex anyway?*
- Every practical non-convex solver is built from convex **local** models (Newton fits a convex quadratic; trust region forces one).
- $\kappa$, $L$, $\mu$ are the vocabulary for conditioning, preconditioning & LR selection.
- Convex subproblems sit inside non-convex loops (EM's M-step, the prox step, each coordinate step).

*Convex but hard?*
- Yes — convexity bounds the *landscape*, not the *cost*. A convex problem can still be huge, ill-conditioned ($\kappa\sim10^{10}$), or nonsmooth.

*What replaces $\nabla f=0$ when $f$ is convex but nonsmooth (L1, hinge)?*
- $\mathbf{0}\in\partial f(\mathbf{x}^*)$, where $\partial f$ is the subdifferential — the set of all valid tangent slopes at that point.
- $|x|$ at 0 has $\partial f=[-1,1]\ni0$ → 0 is the minimizer despite no derivative.
```

&nbsp;

### Optimality Conditions
- **What**: Algebraic tests certifying that a point is a local/global minimum.
- **Why**: "The loss stopped moving" is not a certificate.
    - A solver needs a **target** to aim at and a **stopping rule** that is not just patience.
    - Also classifies *what* was reached: min, max, or saddle.
- **How**:
    1. **Necessary (1st order)**: $\mathbf{g}=\mathbf{0}$ → stationary point.
    2. **Sufficient (2nd order)**: $H\succ0$ → strict local min.
    3. $H$ singular → inconclusive, needs higher-order or structural information.
    4. $f$ convex → step 1 alone is necessary **and** sufficient for a global min.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\mathbf{g}=\nabla f(\mathbf{x}^*)$: Gradient at the candidate.
    - $H=\nabla^2f(\mathbf{x}^*)$: Hessian at the candidate.
    - $\epsilon_g$: Gradient tolerance.

Unconstrained, 1st-order necessary:

$$
\mathbf{g}=\mathbf{0}
$$

2nd-order, given $\mathbf{g}=\mathbf{0}$:

$$
H\succ0\Rightarrow\text{strict local min},\quad H\prec0\Rightarrow\text{strict local max},\quad H\text{ indefinite}\Rightarrow\text{saddle}
$$

Nonsmooth convex:

$$
\mathbf{0}\in\partial f(\mathbf{x}^*)
$$

Scale-free stopping test:

$$
||\mathbf{g}_t||_\infty\leq\epsilon_g(1+|f(\mathbf{x}_t)|)
$$
```

```{attention} Q&A
:class: dropdown
*Why is $\nabla f=0$ not "done"?*
- Stationary $\supseteq$ {min, max, saddle}.
- In high dim a random critical point of a non-convex $f$ is overwhelmingly a **saddle**: it needs all $n$ eigenvalues to share a sign to be an extremum.

*Why not stop on raw $||\mathbf{g}||\leq\epsilon$?*
- Scale-dependent: rescaling $f$ by $10^6$ rescales the gradient by $10^6$ without changing the problem. → use a relative test.

*Second-order sufficient but $H$ only PSD?*
- $H\succeq0$ is necessary, not sufficient — $f=x^3$ at $0$ has $g=0,H=0$ and is neither.
```

&nbsp;

### Duality
- **What**: Lower-bounding relaxation formed by folding constraints into the objective with multipliers.
- **Why**: Constraints block direct descent.
    - *Why do we need it?* A feasible-set boundary has no gradient information — move the constraints **into** the objective so ordinary calculus applies.
    - *Why does it work?* The relaxation is concave regardless of the primal, so its maximum is always a valid, computable lower bound.
- **How**:
    1. Build the **Lagrangian**: objective + multiplier $\times$ constraint, one multiplier per constraint.
    2. Minimize over the primal vars → **dual function**, concave & $\leq$ the primal optimum for any feasible multipliers.
    3. Maximize the dual → tightest lower bound.
    4. Strong duality → gap $=0$ → solve whichever side is cheaper.
    5. At the optimum **KKT** holds; complementary slackness reveals which constraints are active.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $f_0(\mathbf{x})$: Objective.
    - $f_i(\mathbf{x})\leq0$: Inequality constraint $i$.
    - $h_j(\mathbf{x})=0$: Equality constraint $j$.
- Params:
    - $\lambda_i\geq0$: Multiplier for inequality $i$.
    - $\nu_j\in\mathbb{R}$: Multiplier for equality $j$.
- Misc:
    - $p^*$: Primal optimum.
    - $d^*$: Dual optimum.

Lagrangian:

$$
\mathcal{L}(\mathbf{x},\boldsymbol{\lambda},\boldsymbol{\nu})=f_0(\mathbf{x})+\sum_i\lambda_if_i(\mathbf{x})+\sum_j\nu_jh_j(\mathbf{x})
$$

Dual function & dual problem:

$$
g(\boldsymbol{\lambda},\boldsymbol{\nu})=\inf_\mathbf{x}\mathcal{L}(\mathbf{x},\boldsymbol{\lambda},\boldsymbol{\nu}),\qquad d^*=\max_{\boldsymbol{\lambda}\succeq0,\ \boldsymbol{\nu}}g(\boldsymbol{\lambda},\boldsymbol{\nu})
$$

Weak duality (always) / strong duality (convex + Slater):

$$
d^*\leq p^*\qquad\qquad d^*=p^*
$$

KKT conditions:

$$
\begin{align*}
\nabla_\mathbf{x}\mathcal{L}(\mathbf{x}^*,\boldsymbol{\lambda}^*,\boldsymbol{\nu}^*)&=\mathbf{0} &&\text{stationarity}\\
f_i(\mathbf{x}^*)\leq0,\ h_j(\mathbf{x}^*)&=0 &&\text{primal feasibility}\\
\lambda_i^*&\geq0 &&\text{dual feasibility}\\
\lambda_i^*f_i(\mathbf{x}^*)&=0 &&\text{complementary slackness}
\end{align*}
$$
```

```{tip} Derivation
:class: dropdown
*Why is the dual function always concave?*
1. For fixed $\mathbf{x}$, $\mathcal{L}(\mathbf{x},\boldsymbol{\lambda},\boldsymbol{\nu})$ is **affine** in $(\boldsymbol{\lambda},\boldsymbol{\nu})$.
2. $g$ is the pointwise **infimum** over $\mathbf{x}$ of a family of affine functions.
3. A pointwise inf of affine functions is concave.
4. → concave for *any* $f_0,f_i,h_j$, convex or not. This is what makes the dual bound universally available.

*Why is $d^*\leq p^*$?*
1. Let $\mathbf{x}$ be primal feasible: $f_i(\mathbf{x})\leq0$, $h_j(\mathbf{x})=0$.
2. With $\boldsymbol{\lambda}\succeq0$: $\sum_i\lambda_if_i(\mathbf{x})\leq0$ and $\sum_j\nu_jh_j(\mathbf{x})=0$.
3. → $\mathcal{L}(\mathbf{x},\boldsymbol{\lambda},\boldsymbol{\nu})\leq f_0(\mathbf{x})$.
4. → $g(\boldsymbol{\lambda},\boldsymbol{\nu})=\inf_{\mathbf{x}'}\mathcal{L}(\mathbf{x}',\cdot)\leq f_0(\mathbf{x})$ for every feasible $\mathbf{x}$.
5. Take the inf over feasible $\mathbf{x}$ → $g\leq p^*$ → $d^*\leq p^*$.
```

```{attention} Q&A
:class: dropdown
*When is the gap zero?*
- Convex primal + **Slater's condition** ($\exists$ a strictly feasible point for the nonaffine inequalities) → strong duality. All-affine constraints only need feasibility.
- Non-convex → gap can be $>0$, but is provably $0$ for special cases (trust-region subproblem, QCQP w/ one constraint).

*What does complementary slackness buy you?*
- $\lambda_i^*>0\Rightarrow f_i$ active; $f_i$ inactive $\Rightarrow\lambda_i^*=0$.
- → [SVM](supervised.md#svm) support vectors are exactly the $\alpha_i>0$ points; deleting every other sample leaves the solution unchanged.

*Why solve the dual?*
- #constraints $\ll$ #vars → smaller problem.
- Data enters only through inner products → the [kernel trick](supervised.md#kernel-trick).
- Any dual-feasible point gives a running lower bound → certified stopping.

*What do the multipliers mean?*
- Shadow prices: perturbing $f_i(\mathbf{x})\leq u_i$ gives $\frac{\partial p^*}{\partial u_i}=-\lambda_i^*$ → sensitivity of the optimum to tightening constraint $i$.
- → the penalty weight $\lambda$ in [SRM](obj.md#srm) is the multiplier of the equivalent constrained problem, which is why the penalized & constrained forms trace the same solution path.

*Does the dual always help?*
- ❌. Non-convex → the gap is real, so the dual only bounds. And the dual can be *larger* than the primal when #constraints $\gg$ #vars.
```

&nbsp;

### Line Search
- **What**: 1-D subproblem choosing the step length along a fixed direction.
- **Why**: A descent direction alone does not say **how far**.
    - Too long → overshoot, possibly increase $f$.
    - Too short → arbitrarily slow, and convergence proofs break.
    - Exactly minimizing along the ray is wasteful → accept a "good enough" step via inequality tests.
- **How**:
    1. Get a descent direction $\mathbf{p}_t$ ($\mathbf{g}_t^T\mathbf{p}_t<0$).
    2. Trial $\alpha$: $1$ for Newton/quasi-Newton (so the natural step is tried first), else extrapolated from the previous iteration.
    3. **Armijo** test: did $f$ drop by at least a fraction of what the linear model predicted? Fail → shrink $\alpha\leftarrow\rho\alpha$, retry.
    4. **Curvature** test: is the slope now flat enough? Rejects uselessly short steps.
    5. Armijo + curvature = **Wolfe** → accept.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\mathbf{p}_t$: Search direction, $\mathbf{g}_t^T\mathbf{p}_t<0$.
    - $\alpha>0$: Step length.
    - $\phi(\alpha)=f(\mathbf{x}_t+\alpha\mathbf{p}_t)$: Restriction of $f$ to the ray.
- Hyperparams:
    - $c_1$: Sufficient-decrease constant, typically $10^{-4}$.
    - $c_2$: Curvature constant, $0<c_1<c_2<1$; typically $0.9$ for Newton/quasi-Newton, $0.1$ for CG.
    - $\rho\in(0,1)$: Backtracking factor, typically $0.5$.

Armijo (sufficient decrease):

$$
f(\mathbf{x}_t+\alpha\mathbf{p}_t)\leq f(\mathbf{x}_t)+c_1\alpha\mathbf{g}_t^T\mathbf{p}_t
$$

Curvature:

$$
\nabla f(\mathbf{x}_t+\alpha\mathbf{p}_t)^T\mathbf{p}_t\geq c_2\mathbf{g}_t^T\mathbf{p}_t
$$

Strong Wolfe replaces the curvature test with:

$$
|\nabla f(\mathbf{x}_t+\alpha\mathbf{p}_t)^T\mathbf{p}_t|\leq c_2|\mathbf{g}_t^T\mathbf{p}_t|
$$
```

````{important} Code
:class: dropdown
```python
import torch

def backtracking_line_search(f, x, p, g, c1=1e-4, rho=0.5, alpha=1.0, max_iter=50):
    ## g^T p < 0 is required, otherwise no alpha can satisfy Armijo
    slope = g @ p
    assert slope < 0, "not a descent direction"
    f0 = f(x)
    for _ in range(max_iter):
        ## Armijo: accept only if the drop is >= c1 * (linear model's predicted drop)
        if f(x + alpha * p) <= f0 + c1 * alpha * slope:
            return alpha
        alpha *= rho
    return alpha

## Example
f = lambda x: (x ** 2).sum()          ## bowl, minimum at 0
x = torch.tensor([3.0, 4.0])
g = 2 * x                             ## exact gradient
print(backtracking_line_search(f, x, -g, g))  ## 0.25 -> full step alpha=1 would overshoot
```
````

```{attention} Q&A
:class: dropdown
*Why is Armijo alone not enough?*
- $\alpha\to0$ satisfies it trivially → the iterate stalls with a "valid" step every time. The curvature condition is what forbids that.
- Backtracking sidesteps this in practice: it starts at $\alpha=1$ and only shrinks, so it never *returns* a needlessly tiny step.

*Why do quasi-Newton methods need the curvature condition specifically?*
- It is equivalent to $\mathbf{s}_t^T\mathbf{y}_t>0$ (the curvature condition on the secant pair) → guarantees the BFGS update keeps $B_{t+1}\succ0$.
- w/o it the approximate Hessian can lose positive definiteness → the next "direction" is no longer a descent direction.

*Why try $\alpha=1$ first?*
- Newton & quasi-Newton derive a step whose *natural* length is 1. Accepting it unmodified near the solution is exactly what preserves their superlinear/quadratic rate.

*Why is line search rare in deep learning?*
- Every trial is another forward pass, and on a mini-batch $f$ is a noisy estimate → the deterministic decrease test is not even well-posed.
- A tuned [LR schedule](../dl/optim.md#lr-scheduler) costs zero extra passes.
```

&nbsp;

### Trust Region
- **What**: Step obtained by minimizing a local model inside a bounded radius.
- **Why**: Line search commits to a direction **before** choosing a length.
    - If the local model is bad (indefinite $H$, huge curvature), *no* length along that ray is good.
    - Indefinite $H$ → the Newton direction may not even be a descent direction → line search has nothing to search.
    - → bound how far the model is trusted, and let the direction fall out of that bound.
- **How**:
    1. Build a quadratic model $m_t$ around $\mathbf{x}_t$.
    2. Minimize $m_t$ subject to $||\mathbf{p}||\leq\Delta_t$ → direction & length chosen jointly.
    3. Measure agreement $\rho_t$ = actual reduction / predicted reduction.
    4. $\rho_t$ high → the model is trustworthy → accept & ⬆️$\Delta$. $\rho_t$ low → reject the step & ⬇️$\Delta$.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $B_t$: Symmetric approximation of $H_t$ (may be indefinite).
    - $\Delta_t>0$: Trust-region radius.
    - $\rho_t$: Agreement ratio.
    - $\lambda\geq0$: Multiplier of the radius constraint.

Model & subproblem:

$$
m_t(\mathbf{p})=f(\mathbf{x}_t)+\mathbf{g}_t^T\mathbf{p}+\frac{1}{2}\mathbf{p}^TB_t\mathbf{p},\qquad\min_\mathbf{p}m_t(\mathbf{p})\ \ \text{s.t.}\ \ ||\mathbf{p}||_2\leq\Delta_t
$$

Agreement ratio:

$$
\rho_t=\frac{f(\mathbf{x}_t)-f(\mathbf{x}_t+\mathbf{p}_t)}{m_t(\mathbf{0})-m_t(\mathbf{p}_t)}
$$

Exact solution characterization — $\mathbf{p}_t$ is optimal $\Leftrightarrow\exists\lambda\geq0$:

$$
(B_t+\lambda I)\mathbf{p}_t=-\mathbf{g}_t,\qquad B_t+\lambda I\succeq0,\qquad\lambda(\Delta_t-||\mathbf{p}_t||_2)=0
$$
```

```{attention} Q&A
:class: dropdown
*What is the $\lambda I$ term doing?*
- Shifting every eigenvalue of $B_t$ up by $\lambda$ → an indefinite or singular Hessian is regularized into a PD one → a well-defined step even at a saddle.
- This is exactly [LM](#lm) damping, and exactly the SVD-level connection to [Ridge](supervised.md#ridge-regression).

*What does $\Delta$ interpolate between?*
- $\Delta$ small → $\lambda$ large → $(B+\lambda I)^{-1}\approx\frac{1}{\lambda}I$ → step $\approx$ scaled steepest descent.
- $\Delta$ large → $\lambda=0$ → the full Newton step.

*Trust region vs line search?*
- LS: direction → length. TR: length budget → direction.
- TR handles indefinite $H$ natively; LS needs $H$ modified to PD first (e.g. add $\tau I$, or modified Cholesky).
- TR can **reject** a step outright; LS always moves.

*Is the constrained subproblem expensive?*
- Solvable exactly, but it needs an eigendecomposition or a 1-D root find on $\lambda$.
- Practice → approximate: Cauchy point (steepest-descent minimizer inside the ball), dogleg, or Steihaug-CG (truncated CG that halts on the boundary or on negative curvature).

*Why is the gap $d^*=p^*$ for this non-convex subproblem?*
- The TR subproblem is the classic exception: a quadratic objective with a single quadratic constraint has **zero** duality gap, which is why the $\lambda$ characterization above is exact rather than a bound.
```

&nbsp;

## Second-Order
- **What**: Solvers whose step comes from a local **quadratic** model instead of a local linear one.

### Newton's Method
- **What**: Step to the minimizer of the local quadratic model.
- **Why**: First-order methods rescale one direction by one global LR.
    - Ill-conditioned $f$ ($\kappa\gg1$) → the same LR is simultaneously too large along steep directions & too small along flat ones → zig-zag.
    - Curvature is exactly the missing per-direction scale → $H^{-1}$ **is** an automatically tuned, per-direction LR.
    - → affine invariance: iterates are unchanged by any linear reparameterization → ❌feature scaling, ❌LR tuning.
- **How**:
    1. Taylor-expand $f$ to 2nd order at $\mathbf{x}_t$.
    2. Set the model's gradient to $\mathbf{0}$ → solve $H_t\mathbf{p}_t=-\mathbf{g}_t$.
    3. Step, safeguarded by [line search](#line-search) or a [trust region](#trust-region).
    4. Repeat — near $\mathbf{x}^*$ the model is nearly exact → the error **squares** every iteration.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\mathbf{p}_t$: Newton direction.
    - $\tau\geq0$: Hessian modification.
    - $\lambda_\min(H_t)$: Smallest eigenvalue of $H_t$.
    - $C$: Constant depending on $||H(\mathbf{x}^*)^{-1}||$ & the Lipschitz constant of $\nabla^2f$.

Local model:

$$
m_t(\mathbf{p})=f(\mathbf{x}_t)+\mathbf{g}_t^T\mathbf{p}+\frac{1}{2}\mathbf{p}^TH_t\mathbf{p}
$$

Direction (solve the system, never form $H^{-1}$):

$$
H_t\mathbf{p}_t=-\mathbf{g}_t,\qquad\mathbf{x}_{t+1}=\mathbf{x}_t+\alpha_t\mathbf{p}_t
$$

Safeguard when $H_t\not\succ0$:

$$
(H_t+\tau I)\mathbf{p}_t=-\mathbf{g}_t,\qquad\tau>-\lambda_\min(H_t)
$$

Newton decrement (affine-invariant stopping test, $\frac{1}{2}\lambda^2$ estimates $f-f^*$):

$$
\lambda(\mathbf{x}_t)^2=\mathbf{g}_t^TH_t^{-1}\mathbf{g}_t
$$

Local rate — $\alpha_t=1$, $H(\mathbf{x}^*)\succ0$, $\nabla^2f$ Lipschitz:

$$
||\mathbf{x}_{t+1}-\mathbf{x}^*||\leq C||\mathbf{x}_t-\mathbf{x}^*||^2
$$
```

```{tip} Derivation
:class: dropdown
*Where does $-H^{-1}\mathbf{g}$ come from?*
1. 2nd-order Taylor: $f(\mathbf{x}_t+\mathbf{p})\approx f(\mathbf{x}_t)+\mathbf{g}_t^T\mathbf{p}+\frac{1}{2}\mathbf{p}^TH_t\mathbf{p}=m_t(\mathbf{p})$.
2. $\nabla_\mathbf{p}m_t=\mathbf{g}_t+H_t\mathbf{p}$.
3. Set to $\mathbf{0}$ → $\mathbf{p}=-H_t^{-1}\mathbf{g}_t$, a **minimizer** of $m_t$ only if $H_t\succ0$ (else it is a max or a saddle of the model).
4. $f$ exactly quadratic → $m_t=f$ → one step lands on $\mathbf{x}^*$.

*Why quadratic convergence?*
1. Newton is root-finding applied to $\nabla f$: $\mathbf{x}_{t+1}=\mathbf{x}_t-H_t^{-1}\nabla f(\mathbf{x}_t)$.
2. Taylor with remainder: $\mathbf{0}=\nabla f(\mathbf{x}^*)=\mathbf{g}_t+H_t(\mathbf{x}^*-\mathbf{x}_t)+O(||\mathbf{x}^*-\mathbf{x}_t||^2)$.
3. Substituting → $\mathbf{x}_{t+1}-\mathbf{x}^*=H_t^{-1}O(||\mathbf{x}_t-\mathbf{x}^*||^2)$.
4. $H_t^{-1}$ bounded near $\mathbf{x}^*$ → the error squares.
```

````{important} Code
:class: dropdown
```python
import torch

class Newton:
    def __init__(self, f, tau=1e-8):
        self.f, self.tau = f, tau

    def step(self, x):
        g = torch.autograd.functional.jacobian(self.f, x)
        H = torch.autograd.functional.hessian(self.f, x)
        ## damping keeps H PD -> p is guaranteed to point downhill even at a saddle
        H = H + self.tau * torch.eye(x.numel())
        ## solve, never invert: same O(n^3) but far better conditioned
        return x + torch.linalg.solve(H, -g)

## Example
A = torch.tensor([[4.0, 1.0], [1.0, 3.0]])
b = torch.tensor([1.0, 2.0])
f = lambda x: 0.5 * x @ A @ x - b @ x   ## quadratic -> minimizer solves A x = b
print(Newton(f).step(torch.zeros(2)))   ## tensor([0.0909, 0.6364])
print(torch.linalg.solve(A, b))         ## tensor([0.0909, 0.6364]) -> exact in 1 step
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- Quadratic local convergence → a handful of iterations.
- Affine invariant → immune to feature scaling & to the choice of parameterization.
- Newton decrement gives a scale-free stopping criterion.

*Cons?*
- $O(n^2)$ memory, $O(n^3)$ per step → dead beyond $n\sim10^4$.
- Needs $H\succ0$; at a saddle the raw step moves **toward** the stationary point.
- Only **locally** convergent — a full step far from $\mathbf{x}^*$ can diverge.
- Needs analytic/AD second derivatives.

*Why does it get attracted to saddles?*
- It solves $\nabla f=\mathbf{0}$, and saddles satisfy that. With $H$ indefinite, $-H^{-1}\mathbf{g}$ is a stationary point of the model, not a minimizer — it flips the sign of the step along negative-curvature directions.
- Fixes: damping $H+\tau I$, absolute-value the eigenvalues (saddle-free Newton), or a trust region.

*Root-finding Newton vs optimization Newton?*
- Root: $x\leftarrow x-\frac{h(x)}{h'(x)}$. Optimization = root-finding on $h=\nabla f$ → $\mathbf{x}\leftarrow\mathbf{x}-H^{-1}\mathbf{g}$. Same algorithm, one derivative up.

*Where does it actually appear in ML?*
- [Logistic regression](supervised.md#logistic-regression) & [GLM](supervised.md#glm) fitting via [IRLS](#irls).
- [XGBoost](supervised.md#xgboost) leaf weights $-\frac{G}{H+\lambda}$ are a 1-D Newton step on the loss.
- Natural gradient / K-FAC swap $H$ for the Fisher information.
```

&nbsp;

#### Gauss-Newton
- **What**: Newton with $H$ replaced by $J^TJ$ for sum-of-squares objectives.
- **Why**: The exact Hessian of a least-squares objective has a second term weighted by the residuals themselves.
    - Small residuals at the solution → that term is negligible.
    - Dropping it removes **all** second derivatives → only the Jacobian is needed, which is one backward pass per residual.
    - $J^TJ\succeq0$ always → the direction is always a descent direction, unlike raw Newton.
- **How**:
    1. Stack residuals $\mathbf{r}(\mathbf{x})$ & their Jacobian $J$.
    2. Solve the linear least-squares problem $\min_\mathbf{p}||J\mathbf{p}+\mathbf{r}||_2^2$ (via QR/SVD of $J$).
    3. Step & repeat.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{r}(\mathbf{x})\in\mathbb{R}^m$: Residual vector.
- Misc:
    - $J\in\mathbb{R}^{m\times n}$: Jacobian, $J_{ij}=\frac{\partial r_i}{\partial x_j}$.

Objective:

$$
f(\mathbf{x})=\frac{1}{2}||\mathbf{r}(\mathbf{x})||_2^2
$$

Exact derivatives:

$$
\mathbf{g}=J^T\mathbf{r},\qquad H=J^TJ+\sum_{i=1}^m r_i\nabla^2r_i
$$

Gauss-Newton drops the second term:

$$
J^TJ\mathbf{p}=-J^T\mathbf{r}
$$
```

```{attention} Q&A
:class: dropdown
*When does it break?*
- **Large-residual** problems (wrong model class, heavy noise) → the dropped $\sum_ir_i\nabla^2r_i$ term dominates → the rate drops from near-quadratic to linear, or it fails outright.
- Rank-deficient or ill-conditioned $J$ → $J^TJ$ singular → the step is undefined. [LM](#lm) fixes exactly this.

*Why not form $J^TJ$ explicitly?*
- $\kappa(J^TJ)=\kappa(J)^2$ → forming the normal equations squares the condition number and throws away half the significant digits. Factorize $J$ instead.

*Relation to Fisher scoring / natural gradient?*
- For a Gaussian likelihood, $J^TJ$ **is** the Fisher information → Gauss-Newton = Fisher scoring. The generalized Gauss-Newton matrix is the standard PSD Hessian surrogate for NNs.
```

&nbsp;

#### LM
- **Name**: Levenberg-Marquardt
- **What**: Gauss-Newton + a damping term interpolating toward gradient descent.
- **Why**: Gauss-Newton's step is undefined when $J^TJ$ is singular & unreliable far from the solution.
    - Damping makes the system always solvable and implicitly bounds the step length.
    - → the trust-region form of Gauss-Newton, with $\lambda$ adapted instead of $\Delta$.
- **How**:
    1. Solve $(J^TJ+\lambda D)\mathbf{p}=-J^T\mathbf{r}$.
    2. $f$ improved → accept & ⬇️$\lambda$ (→ Gauss-Newton, fast).
    3. $f$ worsened → reject & ⬆️$\lambda$ (→ short gradient steps, safe).

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\lambda\geq0$: Damping factor.
- Misc:
    - $D\succ0$: Scaling matrix; $D=I$ (Levenberg) or $D=\text{diag}(J^TJ)$ (Marquardt).

Damped normal equations:

$$
(J^TJ+\lambda D)\mathbf{p}=-J^T\mathbf{r}
$$

Limits:

$$
\lambda\to0\Rightarrow\mathbf{p}\to\mathbf{p}^\text{GN},\qquad\lambda\to\infty\Rightarrow\mathbf{p}\to-\frac{1}{\lambda}D^{-1}J^T\mathbf{r}
$$
```

```{attention} Q&A
:class: dropdown
*Why $D=\text{diag}(J^TJ)$ rather than $I$?*
- $D=I$ damps every direction by the same absolute amount → it over-penalizes directions that already have small curvature in the *units* they were measured in.
- The diagonal damps each direction **relative to its own curvature** → invariant to rescaling the variables.

*Is it a trust-region method?*
- Yes. $\lambda$ is exactly the multiplier of the constraint $||D^{1/2}\mathbf{p}||\leq\Delta$ (see [Trust Region](#trust-region)) — adapting $\lambda$ directly is just a cheaper way to adapt $\Delta$.

*Where is it used?*
- Nonlinear least squares: curve fitting, camera calibration, bundle adjustment, `scipy.optimize.least_squares`.
- ❌NNs — $J$ is $m\times n$ with $m$ = #samples, $n$ = #params.
```

&nbsp;

#### IRLS
- **Name**: Iteratively Reweighted Least Squares
- **What**: Newton on a [GLM](supervised.md#glm) log-likelihood, algebraically rearranged into a weighted least-squares solve.
- **Why**: GLM MLEs have no closed form.
    - The link function makes the score equations nonlinear in $\mathbf{w}$.
    - But the Newton step for a GLM has exactly the **shape** of weighted normal equations.
    - → reuse the linear least-squares solver already available, once per iteration.
- **How**:
    1. Current $\mathbf{w}$ → fitted means → per-sample weight $s_i$ (curvature) & working response $z_i$ (locally linearized target).
    2. Solve the weighted least-squares problem for a new $\mathbf{w}$.
    3. Repeat until the deviance stops changing.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $X\in\mathbb{R}^{m\times n}$: Design matrix.
    - $\mathbf{y}\in\{0,1\}^m$: Labels (logistic case).
- Params:
    - $\mathbf{w}\in\mathbb{R}^n$: Coefficients.
- Misc:
    - $\mathbf{p}=\sigma(X\mathbf{w})$: Fitted probabilities.
    - $S=\text{diag}(p_i(1-p_i))$: Weight matrix.
    - $\mathbf{z}$: Working response.

Newton step on the logistic log-likelihood:

$$
\mathbf{w}\leftarrow\mathbf{w}+(X^TSX)^{-1}X^T(\mathbf{y}-\mathbf{p})
$$

Identical weighted least-squares form:

$$
\mathbf{w}\leftarrow(X^TSX)^{-1}X^TS\mathbf{z},\qquad\mathbf{z}=X\mathbf{w}+S^{-1}(\mathbf{y}-\mathbf{p})
$$

General GLM with canonical link — only $S$ & $\mathbf{z}$ change:

$$
s_i=\frac{1}{\text{Var}(y_i)},\qquad z_i=\eta_i+(y_i-\mu_i)\frac{d\eta_i}{d\mu_i}
$$
- $\eta_i=\mathbf{x}_i^T\mathbf{w}$: Linear predictor.
- $\mu_i$: Fitted mean.
```

```{tip} Derivation
:class: dropdown
*Why is the Newton step a weighted least-squares solve?*
1. Start from the Newton step: $\mathbf{w}^+=\mathbf{w}+(X^TSX)^{-1}X^T(\mathbf{y}-\mathbf{p})$.
2. Insert $I=(X^TSX)^{-1}(X^TSX)$ into the first term: $\mathbf{w}=(X^TSX)^{-1}X^TSX\mathbf{w}$.
3. Combine: $\mathbf{w}^+=(X^TSX)^{-1}X^TS\left[X\mathbf{w}+S^{-1}(\mathbf{y}-\mathbf{p})\right]$.
4. The bracket is $\mathbf{z}$ → $\mathbf{w}^+=(X^TSX)^{-1}X^TS\mathbf{z}$, the normal equations of $\min_\mathbf{w}\sum_is_i(z_i-\mathbf{x}_i^T\mathbf{w})^2$.
```

```{attention} Q&A
:class: dropdown
*Why does one algorithm fit every GLM?*
- Canonical link → observed Hessian = expected Hessian (Fisher information) → Newton = **Fisher scoring**, and the weight is just the inverse variance function of the family.
- → swapping Gaussian ↔ Bernoulli ↔ Poisson changes only $s_i$ & $z_i$, never the loop.

*Does it always converge?*
- Log-likelihood concave → yes, and fast (typically <10 iterations).
- **Except** under perfect separation: the MLE sits at infinity → $||\mathbf{w}||\to\infty$, $S\to0$, $X^TSX$ becomes singular. Fix: any penalty ([Ridge](supervised.md#ridge-regression) / Firth), or detect and report.

*Why not for NNs?*
- $X^TSX$ is $n\times n$ and is re-formed & re-factorized every iteration → $O(mn^2+n^3)$.
```

&nbsp;

### Quasi-Newton
- **What**: Curvature matrix accumulated from successive gradients instead of computed from second derivatives.
- **Why**: Newton demands $H$: $O(n^2)$ entries, second derivatives, $O(n^3)$ solve.
    - Two consecutive gradients already encode the curvature along the direction actually travelled — the **secant** relation.
    - → build the approximation out of information the solver computes anyway, for free.
- **How**:
    1. Maintain $B_t\approx H_t$, or its inverse $M_t$ directly.
    2. Direction $\mathbf{p}_t=-M_t\mathbf{g}_t$; step with a **Wolfe** line search.
    3. Record $\mathbf{s}_t=\mathbf{x}_{t+1}-\mathbf{x}_t$ and $\mathbf{y}_t=\mathbf{g}_{t+1}-\mathbf{g}_t$.
    4. Update by the **smallest** change (in a weighted Frobenius norm) that satisfies the secant equation and preserves symmetry + PD.

```{note} Math
:class: dropdown
Notations (overriding the page default $H$ for this block):
- Misc:
    - $B_t$: Approximation of the Hessian $\nabla^2f(\mathbf{x}_t)$.
    - $M_t=B_t^{-1}$: Approximation of the inverse Hessian.
    - $\mathbf{s}_t=\mathbf{x}_{t+1}-\mathbf{x}_t$: Step.
    - $\mathbf{y}_t=\mathbf{g}_{t+1}-\mathbf{g}_t$: Gradient difference.
    - $\rho_t=\frac{1}{\mathbf{y}_t^T\mathbf{s}_t}$

Secant equation (the mean-value theorem, one dimension at a time):

$$
B_{t+1}\mathbf{s}_t=\mathbf{y}_t\qquad\Leftrightarrow\qquad M_{t+1}\mathbf{y}_t=\mathbf{s}_t
$$

Curvature condition — necessary for a PD solution, guaranteed by the Wolfe curvature test:

$$
\mathbf{s}_t^T\mathbf{y}_t>0
$$
```

&nbsp;

#### DFP
- **Name**: Davidon-Fletcher-Powell
- **What**: First quasi-Newton method: rank-2 update of the Hessian approximation.
- **Why**: Historically the proof that the secant equation + minimal change + symmetry pins down a usable update at all.
- **How**: Minimize $||B_{t+1}-B_t||_W$ subject to symmetry & the secant equation → closed form.

```{note} Math
:class: dropdown
Hessian form:

$$
B_{t+1}=(I-\rho_t\mathbf{y}_t\mathbf{s}_t^T)B_t(I-\rho_t\mathbf{s}_t\mathbf{y}_t^T)+\rho_t\mathbf{y}_t\mathbf{y}_t^T
$$

Inverse form:

$$
M_{t+1}=M_t+\rho_t\mathbf{s}_t\mathbf{s}_t^T-\frac{M_t\mathbf{y}_t\mathbf{y}_t^TM_t}{\mathbf{y}_t^TM_t\mathbf{y}_t}
$$
```

&nbsp;

#### BFGS
- **Name**: Broyden-Fletcher-Goldfarb-Shanno
- **What**: DFP's dual — the same minimal-change argument applied to the **inverse** Hessian.
- **Why**: DFP recovers slowly from a bad curvature estimate under an inexact line search.
    - Swapping which matrix is updated makes the correction mechanism act on the quantity actually used to form the direction.
    - → self-correcting: a corrupted approximation is washed out within a few iterations.
- **How**: Same loop as [Quasi-Newton](#quasi-newton), updating $M$ so the direction is one matrix-vector product ($O(n^2)$), never a solve ($O(n^3)$).

```{note} Math
:class: dropdown
Inverse form (the one implemented):

$$
M_{t+1}=(I-\rho_t\mathbf{s}_t\mathbf{y}_t^T)M_t(I-\rho_t\mathbf{y}_t\mathbf{s}_t^T)+\rho_t\mathbf{s}_t\mathbf{s}_t^T
$$

Hessian form:

$$
B_{t+1}=B_t-\frac{B_t\mathbf{s}_t\mathbf{s}_t^TB_t}{\mathbf{s}_t^TB_t\mathbf{s}_t}+\rho_t\mathbf{y}_t\mathbf{y}_t^T
$$

→ BFGS-on-$B$ and DFP-on-$M$ are the same formula with $(\mathbf{s},\mathbf{y})$ swapped. That is the duality.
```

```{attention} Q&A
:class: dropdown
*Pros?*
- Superlinear local convergence, ❌second derivatives.
- $O(n^2)$ per iteration vs Newton's $O(n^3)$.
- PD preserved automatically under Wolfe → every direction is a descent direction.
- Self-correcting → tolerant of a sloppy line search.

*Cons?*
- $O(n^2)$ memory → dead above $n\sim10^4$.
- Requires **deterministic** gradients.
- Non-convex $f$ → $\mathbf{s}^T\mathbf{y}\leq0$ can occur → the update must be skipped or damped.

*BFGS vs DFP?*
- Both are in the Broyden class and both are superlinear in theory. BFGS dominates in practice purely because of the self-correction property.

*What if $\mathbf{s}_t^T\mathbf{y}_t\leq0$?*
- Means the observed curvature along the step was negative → no PD matrix satisfies the secant equation.
- Fix: skip the update, or **Powell damping** — replace $\mathbf{y}_t$ by a convex blend of $\mathbf{y}_t$ and $B_t\mathbf{s}_t$ so positivity is restored.
```

&nbsp;

#### L-BFGS
- **Name**: Limited-memory BFGS
- **What**: BFGS with the matrix replaced by the last $\ell$ $(\mathbf{s},\mathbf{y})$ pairs.
- **Why**: BFGS stores a dense $n\times n$ matrix.
    - $n=10^6$ → $10^{12}$ entries → impossible.
    - But the BFGS direction is only ever *applied* to a vector, and the update is a chain of rank-1 factors.
    - → replay the chain from the stored pairs instead of materializing the matrix.
- **How**: **two-loop recursion**.
    1. Keep the last $\ell$ pairs in a ring buffer; discard the oldest.
    2. Backward loop, newest → oldest: peel off the right-hand factors, caching $\alpha_i$.
    3. Scale by the initial guess $M_t^0=\gamma_tI$.
    4. Forward loop, oldest → newest: apply the left-hand factors, correcting with $\beta$.
    5. Out comes $M_t\mathbf{g}_t$ in $O(\ell n)$ time & memory.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\ell$: Memory, typically 3-20.
- Misc:
    - $\gamma_t=\frac{\mathbf{s}_{t-1}^T\mathbf{y}_{t-1}}{\mathbf{y}_{t-1}^T\mathbf{y}_{t-1}}$: Initial inverse-Hessian scale.
    - $\mathbf{q}$: Working vector.

Two-loop recursion, returning $\mathbf{q}=M_t\mathbf{g}_t$:
1. $\mathbf{q}\leftarrow\mathbf{g}_t$
2. For $i=t-1,\cdots,t-\ell$: $\ \alpha_i\leftarrow\rho_i\mathbf{s}_i^T\mathbf{q}$, $\ \mathbf{q}\leftarrow\mathbf{q}-\alpha_i\mathbf{y}_i$
3. $\mathbf{q}\leftarrow\gamma_t\mathbf{q}$
4. For $i=t-\ell,\cdots,t-1$: $\ \beta\leftarrow\rho_i\mathbf{y}_i^T\mathbf{q}$, $\ \mathbf{q}\leftarrow\mathbf{q}+(\alpha_i-\beta)\mathbf{s}_i$
5. Direction $\mathbf{p}_t=-\mathbf{q}$
```

````{important} Code
:class: dropdown
```python
import torch
from collections import deque

class LBFGS:
    def __init__(self, memory=10):
        self.pairs = deque(maxlen=memory)   ## ring buffer of (s, y); oldest is auto-evicted

    def record(self, s, y):
        ## only curvature-positive pairs may enter, else the implicit matrix loses PD
        if s @ y > 1e-10:
            self.pairs.append((s, y))

    def direction(self, g):
        q, alphas = g.clone(), []
        ## loop 1: newest -> oldest
        for s, y in reversed(self.pairs):
            a = (s @ q) / (y @ s)
            q -= a * y
            alphas.append(a)
        ## initial inverse-Hessian scale from the most recent pair
        if self.pairs:
            s, y = self.pairs[-1]
            q *= (s @ y) / (y @ y)
        ## loop 2: oldest -> newest, undoing loop 1's left factors
        for (s, y), a in zip(self.pairs, reversed(alphas)):
            b = (y @ q) / (y @ s)
            q += (a - b) * s
        return -q

## Example
A = torch.tensor([[10.0, 0.0], [0.0, 1.0]])     ## kappa = 10
grad = lambda x: A @ x
opt, x = LBFGS(memory=5), torch.tensor([1.0, 1.0])
for _ in range(20):
    g = grad(x)
    p = opt.direction(g)
    x_new = x + p                                ## unit step; a real solver runs a Wolfe search
    opt.record(x_new - x, grad(x_new) - g)
    x = x_new
print(x.norm() < 1e-6)                           ## True
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- $O(\ell n)$ memory & time → scales to $n\sim10^7$.
- ❌LR hyperparam — the line search sets the step.
- The default solver for convex ML at scale: `sklearn` logistic regression, CRFs, MaxEnt models, GP hyperparameter fitting.

*Cons?*
- Needs **full-batch deterministic** gradients.
- Only **R-linear** convergence in theory (finite memory kills the superlinear rate), though it behaves near-superlinearly in practice.
- $\ell$ is a hyperparam: too small → forgets curvature; too large → dominated by stale pairs from a different region.

*Why does mini-batch noise destroy it?*
- $\mathbf{y}_t=\mathbf{g}_{t+1}-\mathbf{g}_t$ is a **difference** of two noisy vectors → the noise dominates the curvature signal.
- Worse, the two gradients come from different batches → $\mathbf{y}_t\neq\mathbf{0}$ even when $\mathbf{x}$ does not move.
- → this, not memory, is why deep learning uses [Adam](../dl/optim.md) instead.

*L-BFGS-B?*
- Same recursion + box constraints $\mathbf{l}\leq\mathbf{x}\leq\mathbf{u}$: a gradient-projection step identifies the active bounds, then L-BFGS minimizes over the free subspace. The default behind `scipy.optimize.minimize`.
```

&nbsp;

### CG
- **Name**: Conjugate Gradient
- **What**: Successive directions made mutually conjugate w.r.t. the Hessian.
- **Why**: Steepest descent zig-zags in an ill-conditioned bowl.
    - Consecutive exact-line-search steepest-descent directions are **orthogonal** → each step partially undoes the last.
    - Conjugacy ($\mathbf{p}_i^TA\mathbf{p}_j=0$) is the correct notion of "independent" for a quadratic → progress made along one direction is never destroyed.
    - And it stores **no matrix** — only matrix-vector products, which for a loss are Hessian-vector products from double backprop.
- **How**:
    1. $\mathbf{p}_0=-\mathbf{g}_0$.
    2. Exact line search along $\mathbf{p}_t$ (closed form for a quadratic).
    3. New direction = new steepest descent + $\beta_{t+1}\times$ previous direction.
    4. Repeat — on an $n$-dim quadratic it is exact within $n$ steps.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $A\succ0$: System matrix (the Hessian of the quadratic).
    - $\mathbf{b}$: Right-hand side.
- Misc:
    - $\mathbf{r}_t=A\mathbf{x}_t-\mathbf{b}=\nabla f(\mathbf{x}_t)$: Residual.
    - $||\mathbf{v}||_A=\sqrt{\mathbf{v}^TA\mathbf{v}}$: Energy norm.

Problem — minimizing this quadratic $\Leftrightarrow$ solving $A\mathbf{x}=\mathbf{b}$:

$$
f(\mathbf{x})=\frac{1}{2}\mathbf{x}^TA\mathbf{x}-\mathbf{b}^T\mathbf{x}
$$

Iteration:

$$
\alpha_t=\frac{\mathbf{r}_t^T\mathbf{r}_t}{\mathbf{p}_t^TA\mathbf{p}_t},\quad\mathbf{x}_{t+1}=\mathbf{x}_t+\alpha_t\mathbf{p}_t,\quad\mathbf{r}_{t+1}=\mathbf{r}_t+\alpha_tA\mathbf{p}_t
$$

$$
\beta_{t+1}=\frac{\mathbf{r}_{t+1}^T\mathbf{r}_{t+1}}{\mathbf{r}_t^T\mathbf{r}_t},\quad\mathbf{p}_{t+1}=-\mathbf{r}_{t+1}+\beta_{t+1}\mathbf{p}_t
$$

Rate:

$$
||\mathbf{x}_t-\mathbf{x}^*||_A\leq2\left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^t||\mathbf{x}_0-\mathbf{x}^*||_A
$$

Nonlinear CG replaces $\mathbf{r}$ by $\mathbf{g}$ and $\beta$ by a gradient-only formula:

$$
\beta^\text{FR}_{t+1}=\frac{\mathbf{g}_{t+1}^T\mathbf{g}_{t+1}}{\mathbf{g}_t^T\mathbf{g}_t},\qquad\beta^\text{PR}_{t+1}=\frac{\mathbf{g}_{t+1}^T(\mathbf{g}_{t+1}-\mathbf{g}_t)}{\mathbf{g}_t^T\mathbf{g}_t}
$$
- FR: Fletcher-Reeves.
- PR: Polak-Ribière.
```

````{important} Code
:class: dropdown
```python
import torch

def cg(Av, b, x0, tol=1e-10, max_iter=None):
    ## Av is a callable v -> A @ v: the matrix itself is never stored
    x = x0.clone()
    r = Av(x) - b            ## residual == gradient of 0.5 x'Ax - b'x
    p = -r
    rs = r @ r
    for _ in range(max_iter or b.numel()):
        Ap = Av(p)
        alpha = rs / (p @ Ap)         ## exact line search along p
        x += alpha * p
        r += alpha * Ap
        rs_new = r @ r
        if rs_new < tol:
            break
        p = -r + (rs_new / rs) * p    ## conjugate correction, NOT plain steepest descent
        rs = rs_new
    return x

## Example
A = torch.tensor([[4.0, 1.0], [1.0, 3.0]])
b = torch.tensor([1.0, 2.0])
print(cg(lambda v: A @ v, b, torch.zeros(2)))  ## tensor([0.0909, 0.6364]) -> exact in n=2 steps
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- $O(n)$ memory, ❌matrix storage — needs only $A\mathbf{v}$.
- $\sqrt{\kappa}$ rate vs steepest descent's $\kappa$ → quadratically fewer iterations.
- Exact within $n$ steps on a quadratic, and far fewer when the eigenvalues cluster.

*Cons?*
- Assumes $A\succ0$ — indefinite $A$ makes $\mathbf{p}^TA\mathbf{p}\leq0$ and the recursion breaks down.
- Nonlinear CG needs an accurate line search to keep conjugacy meaningful.
- Rounding error erodes conjugacy → periodic restarts ($\beta\leftarrow0$) are required.

*PR vs FR?*
- Near a stall $\mathbf{g}_{t+1}\approx\mathbf{g}_t$ → $\beta^\text{PR}\approx0$ → the method **automatically restarts** to steepest descent. FR has no such mechanism and can stall for many iterations.
- PR is not guaranteed convergent → use PR+ $=\max(\beta^\text{PR},0)$.

*What happens on negative curvature?*
- $\mathbf{p}^TA\mathbf{p}\leq0$ means the model is unbounded along $\mathbf{p}$ → **Steihaug-CG** stops there and returns the point where the direction hits the trust-region boundary. That is what makes truncated-Newton usable on non-convex $f$.

*Where does it appear in ML?*
- Inner solver of Newton-CG / Hessian-free optimization.
- Trust-region subproblem (Steihaug).
- GP inference: solving $(K+\sigma^2I)^{-1}\mathbf{y}$ without an $O(m^3)$ Cholesky.
```

&nbsp;

```{dropdown} Table: Second-Order Methods
| Method | Memory | Per-iteration | Local rate | Needs |
|:--|:--|:--|:--|:--|
| [Gradient Descent](../dl/optim.md) | $O(n)$ | $O(n)$ | Linear, $\frac{\kappa-1}{\kappa+1}$ | $\mathbf{g}$ |
| [CG](#cg) | $O(n)$ | $O(n)$ + 1 matvec | Linear, $\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}$ | $\mathbf{g}$, $A\mathbf{v}$ |
| [L-BFGS](#l-bfgs) | $O(\ell n)$ | $O(\ell n)$ | R-linear | $\mathbf{g}$ |
| [BFGS](#bfgs) / [DFP](#dfp) | $O(n^2)$ | $O(n^2)$ | Superlinear | $\mathbf{g}$ |
| [Gauss-Newton](#gauss-newton) / [LM](#lm) | $O(mn)$ | $O(mn^2)$ | Superlinear if small residual | $J$ |
| [Newton](#newtons-method) | $O(n^2)$ | $O(n^3)$ | Quadratic | $\mathbf{g}$, $H$ |

- $\kappa$: Condition number.
- $\ell$: L-BFGS memory.
- $m$: #residuals.
- Rates are **local** and assume $H(\mathbf{x}^*)\succ0$.
```

&nbsp;

## Splitting
- **What**: One hard problem broken into subproblems each solvable in closed form.

### Coordinate Descent
- **What**: One coordinate exactly minimized at a time, the rest frozen.
- **Why**: Many ML objectives are jointly hard but **trivially solvable one coordinate at a time**.
    - An L1 penalty is nonsmooth in $\mathbf{x}$, yet the 1-D subproblem has a closed-form solution.
    - ❌step size, ❌line search, ❌full gradient.
    - Sparse features → updating coordinate $j$ touches only the samples where $x_{ij}\neq0$.
- **How**:
    1. Pick a coordinate $j$ — cyclic, random, or greedy (largest $|\partial_jf|$).
    2. Exactly minimize $f$ over $x_j$ with everything else fixed.
    3. Repeat until no coordinate moves.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $g$: Smooth convex part.
    - $h_j$: Convex, possibly nonsmooth part acting on coordinate $j$ alone.

Separable structure required for correctness:

$$
f(\mathbf{x})=g(\mathbf{x})+\sum_{j=1}^n h_j(x_j)
$$

Update:

$$
x_j\leftarrow\arg\min_{u}f(x_1,\cdots,x_{j-1},u,x_{j+1},\cdots,x_n)
$$

Quadratic case $f=\frac{1}{2}\mathbf{x}^TA\mathbf{x}-\mathbf{b}^T\mathbf{x}$ — the exact coordinate minimizer (Gauss-Seidel):

$$
x_j\leftarrow\frac{b_j-\sum_{l\neq j}A_{jl}x_l}{A_{jj}}
$$

Guarantee: $g$ convex & differentiable + $h$ separable as above → every limit point is a global minimizer.
```

````{important} Code
:class: dropdown
```python
import torch

class CoordinateDescent:
    ## exact cyclic minimization of 0.5 x'Ax - b'x, one coordinate at a time
    def __init__(self, A, b, n_sweeps=100):
        self.A, self.b, self.n_sweeps = A, b, n_sweeps

    def solve(self, x0):
        x = x0.clone()
        n = x.numel()
        for _ in range(self.n_sweeps):
            for j in range(n):
                ## everything except coordinate j is frozen -> a 1-D quadratic -> exact argmin
                rest = self.A[j] @ x - self.A[j, j] * x[j]
                x[j] = (self.b[j] - rest) / self.A[j, j]
        return x

## Example
A = torch.tensor([[4.0, 1.0], [1.0, 3.0]])
b = torch.tensor([1.0, 2.0])
print(CoordinateDescent(A, b).solve(torch.zeros(2)))  ## tensor([0.0909, 0.6364])
print(torch.linalg.solve(A, b))                       ## same, with no linear solve at all
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- ❌step size, ❌line search, ❌matrix factorization.
- Each update is closed-form & $O(m)$, or $O(\text{nnz})$ with sparse features.
- Warm starts along a $\lambda$ path are nearly free → this is why `glmnet`/`sklearn` fit the whole [Lasso](supervised.md#lasso-regression) regularization path in one pass.
- Produces exact zeros directly, no thresholding heuristic.

*Cons?*
- Correlated features → the coordinate axes are a bad basis → zig-zag, slow.
- Inherently **sequential** → hard to parallelize without losing the exactness.
- Fails on non-separable nonsmoothness.

*When does it stall at a non-optimal point?*
- $h$ nonsmooth and **not** separable across coordinates (fused lasso $\sum_j|x_j-x_{j+1}|$, group lasso, total variation) → every single-coordinate move increases $f$ while a joint move would decrease it → it halts at a non-stationary point.
- Fix: block CD over the coupled groups, or [Proximal Gradient](#proximal-gradient).

*Cyclic vs random vs greedy?*
- **Cyclic**: default, cache-friendly, but vulnerable to adversarial orderings.
- **Random**: cleanest convergence rates, robust to ordering.
- **Greedy** (Gauss-Southwell): fewest iterations, but $O(n)$ to select each coordinate.

*Why is it not used for NNs?*
- $n\sim10^9$ params and no closed-form per-coordinate minimizer → a sweep costs a full forward pass per coordinate.
```

&nbsp;

#### Block Coordinate Descent
- **What**: Coordinate descent over **groups** of variables.
- **Why**: Single-coordinate moves are not always enough.
    - Strongly coupled coordinates must move together or nothing improves.
    - Many problems are non-convex **jointly** yet convex **in each block** → each block update becomes a solved problem.
- **How**: Partition the variables into blocks → cycle, exactly (or approximately) minimizing over one block with the rest fixed.

```{note} Example
:class: dropdown
| Problem | Blocks | Per-block solve |
|:--|:--|:--|
| [K-Means](unsupervised.md#k-means) | (assignments, centroids) | nearest centroid / mean → Lloyd's algorithm |
| [NMF](unsupervised.md#nmf) | $(W,H)$ | nonneg least squares (ALS) |
| Matrix factorization | (user factors, item factors) | ridge solve per row |
| Group Lasso | feature groups | block soft-threshold |
| SVM (SMO) | one pair $(\alpha_i,\alpha_j)$ | analytic 2-var QP |
```

```{attention} Q&A
:class: dropdown
*Why blocks of exactly 2 in SMO?*
- The SVM dual has the equality constraint $\sum_i\alpha_iy_i=0$ → moving a single $\alpha$ violates it. Two is the smallest block that can move while staying feasible.

*Does it converge?*
- Monotone decrease always.
- Convex + separable nonsmoothness → global min.
- Non-convex → only a **block-wise** stationary point, which need NOT be stationary for $f$. → K-Means & NMF need restarts; their answer depends on init.
```

&nbsp;

### Proximal Gradient
- **What**: Gradient step on the smooth part + proximal step on the nonsmooth part.
- **Why**: [Composite objectives](obj.md#composite-objective) $f=g+h$ have no gradient exactly where the interesting solutions live.
    - Subgradient descent works but is $O(1/\sqrt{t})$ and never returns exact zeros (it lands *near* the kink, not on it).
    - $h$ is usually simple enough that its **proximal operator** is closed-form.
    - → keep smooth gradient descent's $O(1/t)$ rate while handling $h$ exactly.
- **How**:
    1. Gradient step on $g$ alone.
    2. Apply $\text{prox}_{\eta h}$ → trade off staying near that point against reducing $h$.
    3. Repeat. $h=\lambda||\cdot||_1$ → the prox **is** soft-thresholding → the algorithm is ISTA.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $g$: Convex, $L$-smooth part.
    - $h$: Convex, possibly nonsmooth part.
- Hyperparams:
    - $\eta\leq\frac{1}{L}$: Step size.
- Misc:
    - $\iota_\mathcal{C}$: Indicator of a convex set ($0$ inside, $\infty$ outside).
    - $\Pi_\mathcal{C}$: Euclidean projection onto $\mathcal{C}$.
    - $S_\lambda$: Soft-thresholding, $S_\lambda(z)=\text{sign}(z)\max(|z|-\lambda,0)$.

Objective:

$$
f(\mathbf{x})=g(\mathbf{x})+h(\mathbf{x})
$$

Proximal operator:

$$
\text{prox}_{\eta h}(\mathbf{v})=\arg\min_\mathbf{u}\left(h(\mathbf{u})+\frac{1}{2\eta}||\mathbf{u}-\mathbf{v}||_2^2\right)
$$

Update (ISTA), converging at $f(\mathbf{x}_t)-f^*=O(1/t)$:

$$
\mathbf{x}_{t+1}=\text{prox}_{\eta h}\left(\mathbf{x}_t-\eta\nabla g(\mathbf{x}_t)\right)
$$

Closed-form proxes:

$$
h=\lambda||\mathbf{x}||_1\Rightarrow S_{\eta\lambda}(\mathbf{x}),\qquad h=\frac{\lambda}{2}||\mathbf{x}||_2^2\Rightarrow\frac{\mathbf{x}}{1+\eta\lambda},\qquad h=\iota_\mathcal{C}\Rightarrow\Pi_\mathcal{C}(\mathbf{x})
$$
```

````{important} Code
:class: dropdown
```python
import torch

def soft_threshold(v, t):
    ## prox of t*||.||_1 -- the ONLY nonsmooth-aware part of the whole method
    return v.sign() * (v.abs() - t).clamp(min=0)

def proximal_gradient(grad_g, prox, x0, eta, n_iter=500):
    x = x0.clone()
    for _ in range(n_iter):
        x = prox(x - eta * grad_g(x), eta)   ## gradient on g, then prox on h
    return x

## Example: lasso, g = 0.5||y - Xw||^2, h = lam*||w||_1
torch.manual_seed(0)
X = torch.randn(50, 3)
y = X @ torch.tensor([3.0, 0.0, -2.0])
lam = 5.0
L = torch.linalg.matrix_norm(X, 2) ** 2      ## Lipschitz constant of grad g -> eta = 1/L
w = proximal_gradient(lambda w: X.T @ (X @ w - y),
                      lambda v, e: soft_threshold(v, e * lam),
                      torch.zeros(3), eta=1.0 / L)
print(w.round(decimals=2))                   ## tensor([ 2.9000,  0.0000, -1.9000]) -> middle weight EXACTLY 0
```
````

```{attention} Q&A
:class: dropdown
*Why is the prox step exact rather than approximate?*
- $\text{prox}_{\eta h}$ is the argmin of the **exact** $h$ plus a quadratic. Only $g$ is linearized. → the nonsmooth geometry (the kink that creates sparsity) is never approximated away.

*Why does subgradient descent not produce zeros?*
- Its update is $\mathbf{x}-\eta(\nabla g+\lambda\,\text{sign}(\mathbf{x}))$: a coordinate at $0$ gets pushed off $0$ by any nonzero $\nabla g$, and nothing pulls it exactly back. The prox has a **dead zone** of width $2\eta\lambda$ that maps a whole interval to exactly $0$.

*What is projected gradient descent?*
- Proximal gradient with $h=\iota_\mathcal{C}$ → $\text{prox}=\Pi_\mathcal{C}$ → step then project. Constrained optimization for free, provided the projection is cheap (box, simplex, ball, PSD cone).

*Why $\eta\leq\frac{1}{L}$?*
- $L$-smoothness gives $g(\mathbf{y})\leq g(\mathbf{x})+\nabla g^T(\mathbf{y}-\mathbf{x})+\frac{L}{2}||\mathbf{y}-\mathbf{x}||^2$. The prox step minimizes exactly that upper bound plus $h$ when $\eta=\frac1L$ → guaranteed descent. → proximal gradient is [MM](#mm).
- $L$ unknown → backtrack on $\eta$ until the bound holds.
```

&nbsp;

#### FISTA
- **Name**: Fast Iterative Shrinkage-Thresholding Algorithm {cite:p}`beck2009fast`
- **What**: Proximal gradient evaluated at a Nesterov-extrapolated point.
- **Why**: $O(1/t)$ is not the best possible for this problem class.
    - The optimal first-order rate for convex $L$-smooth + simple nonsmooth is $O(1/t^2)$.
    - One extra stored vector buys the whole gap.
- **How**:
    1. Take the prox-gradient step from the extrapolated point $\mathbf{v}_t$, not from $\mathbf{x}_t$.
    2. Update the momentum weight $\theta$.
    3. Extrapolate past the new iterate along the last movement direction.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\mathbf{v}_t$: Extrapolated point, $\mathbf{v}_1=\mathbf{x}_0$.
    - $\theta_t$: Momentum weight, $\theta_1=1$.

Iteration:

$$
\mathbf{x}_{t+1}=\text{prox}_{\eta h}\left(\mathbf{v}_t-\eta\nabla g(\mathbf{v}_t)\right)
$$

$$
\theta_{t+1}=\frac{1+\sqrt{1+4\theta_t^2}}{2},\qquad\mathbf{v}_{t+1}=\mathbf{x}_{t+1}+\frac{\theta_t-1}{\theta_{t+1}}(\mathbf{x}_{t+1}-\mathbf{x}_t)
$$

Rate:

$$
f(\mathbf{x}_t)-f^*\leq\frac{2L||\mathbf{x}_0-\mathbf{x}^*||_2^2}{(t+1)^2}
$$
```

````{important} Code
:class: dropdown
```python
import torch
## Reuses soft_threshold from the Proximal Gradient block above

def fista(grad_g, prox, x0, eta, n_iter=500):
    x, v, theta = x0.clone(), x0.clone(), 1.0
    for _ in range(n_iter):
        x_new = prox(v - eta * grad_g(v), eta)          ## prox step FROM the extrapolated point
        theta_new = (1 + (1 + 4 * theta ** 2) ** 0.5) / 2
        ## overshoot past x_new along the last movement -> this is the whole O(1/t) -> O(1/t^2)
        v = x_new + ((theta - 1) / theta_new) * (x_new - x)
        x, theta = x_new, theta_new
    return x

## Example: same lasso as above, far fewer iterations for the same accuracy
torch.manual_seed(0)
X = torch.randn(50, 3)
y = X @ torch.tensor([3.0, 0.0, -2.0])
L = torch.linalg.matrix_norm(X, 2) ** 2
w = fista(lambda w: X.T @ (X @ w - y),
          lambda v, e: soft_threshold(v, e * 5.0),
          torch.zeros(3), eta=1.0 / L, n_iter=50)
print(w.round(decimals=2))                              ## tensor([ 2.9000,  0.0000, -1.9000])
```
````

```{attention} Q&A
:class: dropdown
*Is it monotone?*
- ❌. $f(\mathbf{x}_t)$ oscillates — the extrapolation deliberately overshoots. MFISTA restores monotonicity by keeping the better of the extrapolated and plain steps.

*When does the acceleration NOT help?*
- Strongly convex $g$ → plain proximal gradient is already linearly convergent, and the generic $\theta$ schedule is then suboptimal; use the constant momentum $\frac{\sqrt\kappa-1}{\sqrt\kappa+1}$ instead.
- Very ill-conditioned + few iterations → the oscillation can dominate. **Adaptive restart** (reset $\theta\leftarrow1$ when $f$ rises) fixes it.

*Relation to Nesterov momentum in deep learning?*
- Same extrapolation, same $\theta$ recursion; FISTA adds the prox and requires convexity + exact gradients. [NAG](../dl/optim.md#nag-nesterov-accelerated-gradient) is the $h=0$, stochastic-gradient cousin.
```

&nbsp;

### ADMM
- **Name**: Alternating Direction Method of Multipliers {cite:p}`boyd2011distributed`
- **What**: Constraint-split problem solved by alternating minimizations plus a dual ascent on the coupling constraint.
- **Why**: Proximal gradient needs $\text{prox}_h$ to be cheap, and it is not when $h$ is composed with a matrix.
    - $||D\mathbf{x}||_1$ (fused lasso, total variation) has no closed-form prox, but $||\mathbf{z}||_1$ does.
    - Introduce a copy $\mathbf{z}=D\mathbf{x}$ → each block now sees a function it can handle exactly.
    - Plain dual decomposition converges only under strict assumptions; the augmented quadratic term restores robustness for any $\rho>0$.
- **How**:
    1. Split into $f(\mathbf{x})+h(\mathbf{z})$ subject to $A\mathbf{x}+B\mathbf{z}=\mathbf{c}$.
    2. Minimize the augmented Lagrangian over $\mathbf{x}$.
    3. Minimize it over $\mathbf{z}$.
    4. Dual ascent: push the scaled multiplier by the constraint residual.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $A,B,\mathbf{c}$: Coupling constraint data.
- Params:
    - $\mathbf{u}=\frac{\boldsymbol{\lambda}}{\rho}$: Scaled dual variable.
- Hyperparams:
    - $\rho>0$: Augmented-Lagrangian penalty.

Problem:

$$
\min_{\mathbf{x},\mathbf{z}}\ f(\mathbf{x})+h(\mathbf{z})\quad\text{s.t.}\quad A\mathbf{x}+B\mathbf{z}=\mathbf{c}
$$

Iteration (scaled form):

$$
\begin{align*}
\mathbf{x}_{t+1}&=\arg\min_\mathbf{x}\left(f(\mathbf{x})+\frac{\rho}{2}||A\mathbf{x}+B\mathbf{z}_t-\mathbf{c}+\mathbf{u}_t||_2^2\right)\\
\mathbf{z}_{t+1}&=\arg\min_\mathbf{z}\left(h(\mathbf{z})+\frac{\rho}{2}||A\mathbf{x}_{t+1}+B\mathbf{z}-\mathbf{c}+\mathbf{u}_t||_2^2\right)\\
\mathbf{u}_{t+1}&=\mathbf{u}_t+A\mathbf{x}_{t+1}+B\mathbf{z}_{t+1}-\mathbf{c}
\end{align*}
$$

Lasso instance ($A=I$, $B=-I$, $\mathbf{c}=\mathbf{0}$) — one cached factorization + one soft-threshold per iteration:

$$
\mathbf{x}_{t+1}=(X^TX+\rho I)^{-1}\left(X^T\mathbf{y}+\rho(\mathbf{z}_t-\mathbf{u}_t)\right),\qquad\mathbf{z}_{t+1}=S_{\lambda/\rho}(\mathbf{x}_{t+1}+\mathbf{u}_t)
$$
```

```{attention} Q&A
:class: dropdown
*Pros?*
- Splits nearly any composite/constrained convex problem into pieces with known solutions.
- Converges for **any** $\rho>0$ → correctness needs no step-size tuning.
- Decomposes across samples or features (consensus ADMM) → the classic distributed convex-learning recipe.
- Handles $||D\mathbf{x}||_1$, set constraints, and matrix constraints that proximal gradient cannot.

*Cons?*
- Reaches modest accuracy fast, then crawls → bad when you need many digits.
- $\rho$ does not affect correctness but strongly affects **speed** → needs adaptive tuning off the primal/dual residual ratio.
- The $\mathbf{x}$-update can itself be an expensive solve (mitigated by caching a factorization when $A,\rho$ are fixed).
- Non-convex → no general guarantee; used heuristically.

*Why the augmented (quadratic) term?*
- Plain dual ascent on $\mathcal{L}$ requires $f$ strictly convex & finite for the inner min to be well-defined. Adding $\frac{\rho}{2}||\cdot||^2$ makes every subproblem strongly convex → well-posed even for piecewise-linear $f$.

*Why not just minimize the augmented Lagrangian jointly?*
- That is the method of multipliers — correct, but the quadratic term **couples** $\mathbf{x}$ and $\mathbf{z}$, destroying separability. ADMM's single Gauss-Seidel sweep is what buys decomposability back, at the cost of a slower rate.
```

&nbsp;

## Surrogate Minimization
- **What**: Objective replaced each iteration by an easier surrogate that touches it at the current point.

### MM
- **Name**: Majorization-Minimization {cite:p}`hunter2004tutorial`
- **What**: Iterative minimization of a tangent upper bound.
- **Why**: Minimizing $f$ directly can be intractable, while *any* tangent upper bound gives guaranteed progress.
    - Touching + dominating ⇒ minimizing the surrogate can never increase $f$ → **monotone descent with no step size and no line search**.
    - The surrogate is free to be chosen so that it decouples variables, removes nonsmoothness, or is quadratic.
- **How**:
    1. Construct $Q(\mathbf{x}|\mathbf{x}_t)\geq f(\mathbf{x})$ with equality at $\mathbf{x}_t$ — via Jensen, convexity/concavity, a quadratic upper bound, or Cauchy-Schwarz.
    2. Minimize $Q$ → $\mathbf{x}_{t+1}$.
    3. Rebuild the surrogate at the new point & repeat.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $Q(\mathbf{x}|\mathbf{x}_t)$: Surrogate anchored at $\mathbf{x}_t$.

Majorization conditions:

$$
Q(\mathbf{x}|\mathbf{x}_t)\geq f(\mathbf{x})\ \ \forall\mathbf{x},\qquad Q(\mathbf{x}_t|\mathbf{x}_t)=f(\mathbf{x}_t)
$$

Update:

$$
\mathbf{x}_{t+1}=\arg\min_\mathbf{x}Q(\mathbf{x}|\mathbf{x}_t)
$$

Descent, for free:

$$
f(\mathbf{x}_{t+1})\leq Q(\mathbf{x}_{t+1}|\mathbf{x}_t)\leq Q(\mathbf{x}_t|\mathbf{x}_t)=f(\mathbf{x}_t)
$$
- 1st $\leq$: majorization.
- 2nd $\leq$: $\mathbf{x}_{t+1}$ minimizes $Q$.
- $=$: tangency.
```

```{attention} Q&A
:class: dropdown
*Which familiar methods are MM in disguise?*
- [EM](#em) — surrogate = the negative ELBO.
- [Proximal Gradient](#proximal-gradient) — surrogate = the $L$-smoothness quadratic upper bound + $h$.
- [NMF](unsupervised.md#nmf) multiplicative updates.
- IRLS for robust/L1 regression — quadratic majorization of $|r|$.
- CCCP for a difference of convex functions — linearize the concave part.

*What is the cost of the free descent?*
- The bound is loose away from $\mathbf{x}_t$ → each step is conservative → **linear** convergence at best, and it slows down further as the majorizer's curvature exceeds $f$'s.
- Tightness of the surrogate is exactly the speed knob.

*Minorization-Maximization?*
- The mirror image: for maximization, use a tangent **lower** bound. Same acronym. EM is normally stated this way.
```

&nbsp;

### EM
- **Name**: Expectation-Maximization {cite:p}`dempster1977maximum`
- **What**: MLE/MAP with latent variables, by alternating posterior inference & posterior-weighted refitting.
- **Why**: The observed-data likelihood requires marginalizing the latents.
    - $\log\sum_\mathbf{z}p(\mathbf{x},\mathbf{z}|\theta)$ puts the log **outside** a sum → terms do not separate → no closed form, and the gradient couples every param.
    - If $\mathbf{z}$ were observed, the complete-data MLE would be a trivial weighted count/average.
    - → impute $\mathbf{z}$ by its posterior, then use the easy complete-data update, and repeat.
- **How**:
    1. **E-step**: with $\theta_t$ fixed, compute the posterior $p(\mathbf{z}|\mathbf{x},\theta_t)$ — a soft assignment of each sample to each latent configuration.
    2. **M-step**: maximize the posterior-weighted complete-data log-likelihood over $\theta$.
    3. Repeat until the observed log-likelihood stops rising.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $X=\{\mathbf{x}_1,\cdots,\mathbf{x}_m\}$: Observed data.
    - $\mathbf{z}_i\in\mathcal{Z}$: Latent variable of sample $i$.
- Params:
    - $\theta$: Model params.
- Misc:
    - $q_i(\mathbf{z}_i)$: Variational distribution over $\mathbf{z}_i$.
    - $\mathcal{F}(q,\theta)$: Evidence lower bound (ELBO).

Goal — intractable because the log sits outside the sum:

$$
\ell(\theta)=\log p(X|\theta)=\sum_{i=1}^m\log\sum_{\mathbf{z}_i\in\mathcal{Z}}p(\mathbf{x}_i,\mathbf{z}_i|\theta)
$$

Exact decomposition, valid for **any** $q$:

$$
\ell(\theta)=\mathcal{F}(q,\theta)+\sum_{i=1}^mD_{KL}\left(q_i\ ||\ p(\mathbf{z}_i|\mathbf{x}_i,\theta)\right)
$$

$$
\mathcal{F}(q,\theta)=\sum_{i=1}^m\mathbb{E}_{q_i}\left[\log\frac{p(\mathbf{x}_i,\mathbf{z}_i|\theta)}{q_i(\mathbf{z}_i)}\right]
$$

EM = coordinate ascent on $\mathcal{F}$:

$$
\text{E: }q_i^{t+1}=p(\mathbf{z}_i|\mathbf{x}_i,\theta_t),\qquad\text{M: }\theta_{t+1}=\arg\max_\theta\sum_{i=1}^m\mathbb{E}_{q_i^{t+1}}\left[\log p(\mathbf{x}_i,\mathbf{z}_i|\theta)\right]
$$
```

```{tip} Derivation
:class: dropdown
*Why does the likelihood increase monotonically?*
1. $D_{KL}\geq0$ → $\ell(\theta)\geq\mathcal{F}(q,\theta)$ for every $q$. (Equivalently: Jensen on $\log\sum_\mathbf{z}q\frac{p}{q}$.)
2. **E-step** sets $q^{t+1}=p(\mathbf{z}|\mathbf{x},\theta_t)$ → $D_{KL}=0$ → $\mathcal{F}(q^{t+1},\theta_t)=\ell(\theta_t)$. The bound becomes **tight**, touching $\ell$ at $\theta_t$.
3. **M-step** maximizes $\mathcal{F}$ over $\theta$ → $\mathcal{F}(q^{t+1},\theta_{t+1})\geq\mathcal{F}(q^{t+1},\theta_t)$.
4. Step 1 at the new params → $\ell(\theta_{t+1})\geq\mathcal{F}(q^{t+1},\theta_{t+1})$.
5. Chain 2-4: $\ell(\theta_{t+1})\geq\mathcal{F}(q^{t+1},\theta_{t+1})\geq\mathcal{F}(q^{t+1},\theta_t)=\ell(\theta_t)$.
6. → tangent lower bound, maximized each step ⇒ EM is [MM](#mm) with surrogate $-\mathcal{F}$.

*Why does the $-\mathbb{E}_q[\log q]$ term vanish from the M-step?*
- It does not depend on $\theta$ → it is a constant of the maximization → only $\mathbb{E}_q[\log p(\mathbf{x},\mathbf{z}|\theta)]$ (the "$Q$ function") matters.
```

```{attention} Q&A
:class: dropdown
*Pros?*
- Monotone ⬆️ likelihood → ❌LR, ❌line search, ❌divergence.
- The M-step is usually the same closed form as the fully-observed MLE → near-trivial to implement.
- Handles latent structure & missing data with the same machinery.
- E-step outputs calibrated posteriors, not just hard assignments.

*Cons?*
- Local optima & saddle points → init-sensitive → multiple restarts required.
- **Linear** convergence, and the rate degrades as the fraction of missing information grows.
- Intractable E-step whenever the posterior has no closed form.
- Can chase degenerate solutions when the likelihood is unbounded (a [GMM](unsupervised.md#gmm) component collapsing onto one point).

*Why not just run gradient ascent on $\ell$?*
- You can — Fisher's identity gives $\nabla_\theta\ell(\theta)=\mathbb{E}_{p(\mathbf{z}|\mathbf{x},\theta)}[\nabla_\theta\log p(\mathbf{x},\mathbf{z}|\theta)]$, i.e. the E-step already hands you the gradient.
- EM wins when the M-step is closed-form: it takes an **exact** maximizing step, with no LR and no overshoot risk. GD wins when the M-step is not closed-form or the data does not fit in memory.

*What breaks if the E-step is only approximate?*
- The bound stays loose ($D_{KL}>0$) → you are monotone in $\mathcal{F}$, NOT in $\ell$. Variational EM optimizes a lower bound whose gap you cannot measure.

*Where is it used?*
- [GMM](unsupervised.md#gmm), [K-Means](unsupervised.md#k-means) (the hard/zero-variance limit), HMM (Baum-Welch), [LDA](unsupervised.md#lda) (variational EM), factor analysis, missing-data imputation, mixture-of-experts.

*Does it find the MLE?*
- ❌. It finds a **stationary point** of $\ell$ — local max, or (from adversarial inits) a saddle.
```

&nbsp;

#### GEM
- **Name**: Generalized EM
- **What**: EM whose M-step merely **increases** the surrogate instead of maximizing it.
- **Why**: The M-step argmax can itself be intractable, while one ascent step on it is cheap.
    - The monotonicity proof only uses $\mathcal{F}(q^{t+1},\theta_{t+1})\geq\mathcal{F}(q^{t+1},\theta_t)$ — the argmax is never needed.
- **How**: Replace the M-step argmax by any move that does not decrease $\mathbb{E}_q[\log p(\mathbf{x},\mathbf{z}|\theta)]$ — a gradient step, a Newton step, or a conditional maximization over one param block.

```{dropdown} Table: EM Variants
| Variant | Change | Buys |
|:--|:--|:--|
| GEM | M-step ascends instead of maximizing | Intractable M-step |
| ECM | M-step = conditional maximization over param blocks | Coupled params |
| MAP-EM | $+\log p(\theta)$ in the M-step | Regularization; fixes covariance collapse |
| Variational EM | E-step restricted to a tractable family $\mathcal{Q}$ | Intractable posterior; monotone in $\mathcal{F}$ only |
| MCEM | E-step expectation estimated by Monte Carlo | Intractable expectation; loses strict monotonicity |
| Online/Incremental EM | E-step on a mini-batch, sufficient statistics kept as a running average | Data that does not fit in memory |
```

&nbsp;

## Search
- **What**: Black-box methods that only ever **query** $f$ — ❌gradient, ❌structure, ❌closed form.

### Grid Search
- **What**: Exhaustive evaluation of a Cartesian product of per-dim value lists.
- **Why**: Hyperparams have no gradient.
    - config → validation score is a black box: non-differentiable, expensive, noisy, and defined over mixed discrete/continuous/conditional spaces.
    - → the only universally valid move is to *evaluate points*, and the simplest schedule is "all of them".
- **How**:
    1. Discretize each dim into a list — **log-scale** for scale params (LR, $\lambda$, $C$).
    2. Take the Cartesian product.
    3. Evaluate every combination by cross-validation.
    4. Keep the best.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $d$: #hyperparams.
    - $k_j$: #values tried for hyperparam $j$.

#evaluations — exponential in $d$:

$$
|\mathcal{H}|=\prod_{j=1}^dk_j
$$

#distinct values explored along any single axis, regardless of $|\mathcal{H}|$:

$$
k_j
$$
```

```{attention} Q&A
:class: dropdown
*Pros?*
- Embarrassingly parallel, deterministic, exactly reproducible.
- Trivial to implement, explain & audit.
- Genuinely fine for $d\leq2$-$3$ on a cheap model.

*Cons?*
- $O(k^d)$ → dead past a handful of dims.
- **Uniform resolution on every axis** regardless of importance → most evaluations differ only in a dim that has no effect.
- Resolution is fixed a priori → an optimum between grid points is unreachable.
- ❌Conditional spaces (`kernel=rbf` implies `gamma`; `kernel=linear` does not).

*Why log-scale grids?*
- LR, $\lambda$, $C$ act **multiplicatively**: $10^{-5}\to10^{-4}$ matters as much as $10^{-1}\to1$. A linear grid spends nearly all its points on the large end.
```

&nbsp;

### Random Search
- **What**: Configs drawn i.i.d. from a distribution over the space {cite:p}`bergstra2012random`
- **Why**: Grid spends its budget on axes that do not matter.
    - Hyperparam importance is extremely uneven — usually only a few dims move the score at all (**low effective dimensionality**), and you do not know which in advance.
    - A grid of $k^d$ points has only $k$ **distinct values** per axis; $n$ random draws have $n$ distinct values on **every** axis.
    - → at equal budget, random resolves the important dims $\frac{n}{k}$ times more finely.
- **How**:
    1. Define a distribution per dim — log-uniform for scales, uniform for bounded, categorical for discrete.
    2. Sample $n$ configs i.i.d.
    3. Evaluate & keep the best.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $p$: Measure of the "good enough" region (e.g. top 5% of configs).
    - $\delta$: Failure probability.
    - $n$: #trials.

Probability that at least one of $n$ i.i.d. draws lands in the good region:

$$
P(\text{hit})=1-(1-p)^n
$$

Required #trials:

$$
n\geq\frac{\log\delta}{\log(1-p)}
$$
```

```{tip} Derivation
:class: dropdown
*Why is 60 trials the folk number?*
1. Want $P(\text{hit})\geq1-\delta$ → $1-(1-p)^n\geq1-\delta$ → $(1-p)^n\leq\delta$.
2. Take logs (both sides negative) → $n\geq\frac{\log\delta}{\log(1-p)}$.
3. Top-5% region, 95% confidence: $p=\delta=0.05$ → $n\geq\frac{\log0.05}{\log0.95}=58.4$ → **60 trials**.
4. $d$ never appears. → the guarantee is **dimension-free**, which is the entire argument for random over grid.
5. Caveat: it bounds the chance of hitting the top-$p$ **quantile of the sampling distribution**, not of finding the optimum. A badly chosen prior moves the target, not the math.
```

```{attention} Q&A
:class: dropdown
*Pros?*
- ⬆️Resolution along the dims that matter, at identical budget.
- Budget decoupled from the design → stop any time, extend any time, no re-planning.
- Embarrassingly parallel.
- Handles conditional & mixed spaces natively.
- Dimension-free coverage guarantee.

*Cons?*
- ❌Learning: trial $n+1$ ignores trials $1..n$ entirely. Exactly what [SMBO](#smbo) fixes.
- Keeps sampling regions already proven bad.
- "Good with high probability", never "optimal".

*Why is it such a strong baseline?*
- Real response surfaces have low effective dimensionality → random is effectively searching a low-dim problem, so ⬆️$d$ (with the effective dim fixed) costs it nothing, while it destroys grid.

*Latin hypercube / Sobol?*
- Low-discrepancy quasi-random sequences: better space-filling at small $n$, identical asymptotics. Standard choice for the **initial design** of [BO](#smbo).
```

&nbsp;

### SA
- **Name**: Simulated Annealing {cite:p}`kirkpatrick1983optimization`
- **What**: Local search accepting worsening moves with a temperature-decaying probability.
- **Why**: Hill climbing freezes in the first local optimum it reaches.
    - Always-accept-improvement = greedy; never-accept-worse = never escaping.
    - Physical analogy: slowly cooled metal reaches a lower-energy configuration than quenched metal.
    - → permit uphill moves early (explore), suppress them later (exploit), on a schedule.
- **How**:
    1. Propose a random neighbor of the current point.
    2. Better → always accept.
    3. Worse → accept w/ probability $e^{-\Delta f/T}$.
    4. ⬇️$T$.
    5. $T\to0$ → the chain freezes at a minimum.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\mathbf{x}'$: Proposed neighbor.
    - $\Delta f=f(\mathbf{x}')-f(\mathbf{x}_t)$: Change in objective.
- Hyperparams:
    - $T_0$: Initial temperature.
    - $T_t$: Temperature at step $t$.
    - $\alpha\in(0,1)$: Geometric cooling rate, typically 0.85-0.99.
    - $c$: Constant in the logarithmic schedule.

Metropolis acceptance:

$$
P(\text{accept }\mathbf{x}')=\min\left(1,\ e^{-\Delta f/T_t}\right)
$$

Cooling schedules:

$$
T_t=\alpha^tT_0\ \ \text{(geometric — used in practice)},\qquad T_t=\frac{c}{\log(1+t)}\ \ \text{(logarithmic — required by the guarantee)}
$$

Stationary distribution at fixed $T$:

$$
p_T(\mathbf{x})\propto e^{-f(\mathbf{x})/T}
$$
```

````{important} Code
:class: dropdown
```python
import torch

class SimulatedAnnealing:
    def __init__(self, f, T0=1.0, alpha=0.99, step=0.5):
        self.f, self.T0, self.alpha, self.step = f, T0, alpha, step

    def run(self, x0, n_iter=5000):
        x, fx = x0.clone(), self.f(x0)
        best, fbest, T = x.clone(), fx, self.T0
        for _ in range(n_iter):
            xp = x + self.step * torch.randn_like(x)      ## random neighbor
            fp = self.f(xp)
            d = fp - fx
            ## downhill always accepted; uphill accepted w.p. exp(-d/T) -> shrinks as T falls
            if d < 0 or torch.rand(1) < torch.exp(-d / T):
                x, fx = xp, fp
                if fx < fbest:
                    best, fbest = x.clone(), fx           ## SA is not monotone -> track the best seen
            T *= self.alpha
        return best, fbest

## Example: many local minima, one global minimum at x = 0
f = lambda x: (x ** 2).sum() + 10 * torch.sin(3 * x).sum()
best, fbest = SimulatedAnnealing(f).run(torch.tensor([5.0]))
print(best.round(decimals=1))                             ## tensor([-0.5]) -> global basin, not the nearest local one
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- ❌gradient, ❌continuity, ❌convexity → works on discrete, combinatorial & mixed spaces.
- $O(1)$ memory — one incumbent.
- Provably converges to the global optimum under a logarithmic schedule.

*Cons?*
- The logarithmic schedule is **impractically slow** → real runs use geometric cooling and forfeit the guarantee.
- Very sensitive to $T_0$, $\alpha$, and the neighborhood definition.
- Strictly sequential → poor parallelism, unlike [Evolution](#evolution).
- Sample-inefficient → useless when each evaluation costs GPU-hours.

*Why an exponential acceptance rule specifically?*
- It makes the chain's stationary distribution Boltzmann, $p_T\propto e^{-f/T}$.
- $T\to\infty$ → uniform (pure exploration). $T\to0$ → a point mass on the global minima.
- → the cooling schedule is a slow homotopy from uniform sampling to the answer, and "slow enough" is exactly what the logarithmic schedule encodes.

*How to set $T_0$?*
- Calibrate rather than guess: sample random moves, measure mean $|\Delta f|$, choose $T_0$ so that ~80% of *worsening* moves are accepted initially.

*Relation to SGD noise?*
- Same idea, different mechanism: the mini-batch gradient noise plays the role of temperature and the LR schedule plays the role of cooling. Neither has SA's guarantee.
```

&nbsp;

### SH
- **Name**: Successive Halving {cite:p}`jamieson2016non`
- **What**: Configs evaluated on a small budget, the worst repeatedly discarded & the survivors' budget multiplied.
- **Why**: Training every config to completion is wasteful.
    - Most configs are visibly hopeless after a small fraction of their budget.
    - → treat it as **best-arm identification**: spend the next unit of budget where the current evidence is strongest.
- **How**:
    1. Sample $n$ configs; give each budget $r$.
    2. Evaluate all → keep the best $1/\eta$.
    3. Multiply the survivors' budget by $\eta$.
    4. Repeat until one config remains.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $n$: Initial #configs.
    - $r$: Initial budget per config.
    - $\eta>1$: Reduction factor.
- Misc:
    - $k$: Round index.
    - $B$: Total budget.

Round $k$:

$$
n_k=\lfloor n\eta^{-k}\rfloor,\qquad r_k=r\eta^k
$$

Cost per round is constant, $n_kr_k\approx nr$ → total:

$$
B\approx nr\left(\lfloor\log_\eta n\rfloor+1\right)
$$
```

````{important} Code
:class: dropdown
```python
import math

def successive_halving(configs, evaluate, r=1, eta=3):
    ## evaluate(config, budget) -> score (higher is better)
    survivors, budget = list(configs), r
    while len(survivors) > 1:
        scores = [evaluate(c, budget) for c in survivors]
        keep = max(1, len(survivors) // eta)          ## cull all but the top 1/eta
        order = sorted(range(len(survivors)), key=lambda i: -scores[i])
        survivors = [survivors[i] for i in order[:keep]]
        budget *= eta                                 ## survivors inherit eta x the budget
    return survivors[0]

## Example: config = a "true quality"; low budget gives a noisy read of it
import random
random.seed(0)
configs = [round(random.random(), 3) for _ in range(27)]
evaluate = lambda c, b: c + random.gauss(0, 1.0 / b)  ## noise shrinks as budget grows
print(successive_halving(configs, evaluate), max(configs))  ## 0.874 0.874
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- Exponentially more configs explored per unit budget than uniform allocation.
- Fully parallel within a round.
- Exactly one new hyperparam ($\eta$), with a sane default of 3.

*Cons?*
- The **$n$ vs $B/n$ trade-off** must be fixed up front: many configs briefly, or few thoroughly? There is no universally right answer. → [Hyperband](#hyperband).
- Assumes low-budget ranking ≈ high-budget ranking. **Slow starters** (large models, low LR + long warmup) are killed before they show their value.

*What counts as "budget"?*
- Anything monotone: epochs, #training samples, #trees, image resolution, CV folds, wall-clock.

*Relation to bandits?*
- It IS a fixed-budget best-arm identification algorithm: each config is an arm, each unit of budget is a pull, and the score is a noisy reward whose noise shrinks with the budget.
```

&nbsp;

#### Hyperband
- **What**: SH run at several $(n,r)$ trade-offs, hedging the choice {cite:p}`li2018hyperband`
- **Why**: SH's $n$ vs $B/n$ trade-off has no universally correct setting.
    - Aggressive (large $n$, small $r$) wins when bad configs are obvious early.
    - Conservative (small $n$, large $r$) wins when rankings only stabilize late.
    - → grid-search **over the trade-off itself**: one SH bracket per setting.
- **How**:
    1. From $R$ (max budget per config) & $\eta$, get $s_\max=\lfloor\log_\eta R\rfloor$.
    2. For $s=s_\max$ down to $0$, run one SH bracket: $n_s$ configs starting at budget $r_s$.
    3. $s=s_\max$ → most aggressive. $s=0$ → plain [Random Search](#random-search) at full budget, i.e. the safety net.
    4. Return the best config across all brackets.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $R$: Max budget a single config may consume.
    - $\eta$: Reduction factor, suggested 3 or 4.
- Misc:
    - $s$: Bracket index.
    - $s_\max$: Most aggressive bracket.
    - $B$: Budget per bracket.

Bracket schedule:

$$
s_\max=\lfloor\log_\eta R\rfloor,\qquad B=(s_\max+1)R
$$

$$
n_s=\left\lceil\frac{B}{R}\cdot\frac{\eta^s}{s+1}\right\rceil=\left\lceil\frac{s_\max+1}{s+1}\eta^s\right\rceil,\qquad r_s=R\eta^{-s}
$$

Inner SH loop within bracket $s$, for $k=0,\cdots,s$:

$$
n_k=\lfloor n_s\eta^{-k}\rfloor,\qquad r_k=r_s\eta^k
$$
```

```{attention} Q&A
:class: dropdown
*Pros?*
- Only 2 inputs ($R$, $\eta$), both physically meaningful.
- Provably within a log factor of the best fixed SH setting.
- Bracket $s=0$ is plain random search → a hard floor on how badly it can do.
- Reported order-of-magnitude speedups over black-box BO on NN & kernel benchmarks.

*Cons?*
- Configs are still sampled **at random** → zero learning across brackets. Exactly what [BOHB](#bohb) fixes.
- Repeats cheap low-budget work across brackets.
- Requires a low-budget proxy that actually correlates with the final score.

*How to pick $\eta$?*
- 3 or 4. ⬆️$\eta$ → harsher culling, fewer rounds, ⬆️risk of discarding a slow starter.

*Why not just run the most aggressive bracket?*
- That IS the failure mode Hyperband insures against: if the low-budget signal is misleading, the aggressive bracket is *worse* than random. The conservative brackets pay a constant factor for that insurance.
```

&nbsp;

## SMBO
- **Name**: Sequential Model-Based Optimization
- **What**: Loop of {fit a cheap surrogate to all past evaluations → maximize an acquisition function over it → evaluate the winner}.
- **Why**: [Random Search](#random-search) discards every observation it pays for.
    - *Why do we need it?* Real objectives are **expensive** (one evaluation = one full training run), **black-box** (❌gradient), and **noisy**. Under a budget of tens-to-hundreds of evaluations, the only remaining lever is to reuse what you already measured.
    - *Why does it work?* The expensive $f$ is replaced by a cheap surrogate, so the *search* becomes free — you can afford to optimize the surrogate exhaustively and spend the real budget only on the single most informative point.
- **How**:
    1. Initial design: a handful of quasi-random configs (Sobol / Latin hypercube).
    2. Fit a surrogate $p(f|\mathcal{D}_t)$ — cheap, and it must report **uncertainty**, not just a point estimate.
    3. Maximize an acquisition $a(\mathbf{x}|\mathcal{D}_t)$ — this inner problem is cheap, so multi-start gradient ascent is affordable.
    4. Evaluate the true $f$ there; append to $\mathcal{D}$.
    5. Repeat until the budget runs out.
    - Probabilistic surrogate → **BO** (Bayesian Optimization). Surrogate = [GP](#gp) → the classical form; = KDE ratio → [TPE](#tpe); = random forest → [SMAC](#smac).

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathcal{X}$: Search space.
    - $\mathcal{D}_t=\{(\mathbf{x}_i,y_i)\}_{i=1}^t$: Observations so far.
    - $y_i=f(\mathbf{x}_i)+\epsilon_i$: Noisy observation, $\epsilon_i\sim\mathcal{N}(0,\sigma_n^2)$.
- Misc:
    - $a(\mathbf{x}|\mathcal{D}_t)$: Acquisition function.
    - $\mu(\mathbf{x}),\sigma(\mathbf{x})$: Surrogate posterior mean & std.
    - $f^+=\max_{i\leq t}y_i$: Incumbent (maximization convention).

Selection rule:

$$
\mathbf{x}_{t+1}=\arg\max_{\mathbf{x}\in\mathcal{X}}a(\mathbf{x}|\mathcal{D}_t)
$$

Simple regret — what BO tries to drive to 0:

$$
r_T=f(\mathbf{x}^*)-\max_{t\leq T}f(\mathbf{x}_t)
$$

Cumulative regret — what no-regret acquisitions bound:

$$
R_T=\sum_{t=1}^T\left(f(\mathbf{x}^*)-f(\mathbf{x}_t)\right)
$$
```

```{attention} Q&A
:class: dropdown
*Pros?*
- Sample-efficient: an order of magnitude fewer evaluations than random on expensive objectives.
- Handles observation noise & reports uncertainty.
- Surrogate-agnostic → swap the model to match the space.

*Cons?*
- **Sequential by construction** → poor wall-clock parallelism unless explicitly batched.
- Exact GP fitting is $O(t^3)$ → past $t\sim10^3$ the *optimizer* becomes the bottleneck.
- Degrades badly in high dim ($d\gtrsim20$).
- Model misspecification → confidently wrong → worse than random.

*Why is uncertainty mandatory?*
- With a point-estimate surrogate the acquisition collapses to "go to $\arg\max\hat f$" → pure exploitation → it locks into the first basin and never leaves.
- Uncertainty is the only thing that lets "somewhere I have never looked" outbid "somewhere that looked good".

*When should you NOT use it?*
- Cheap objective → run [Random Search](#random-search) in parallel; the surrogate overhead dominates.
- $>10^3$ affordable evaluations → [Evolution](#evolution) or [SH](#sh)-style bandits scale better.
- Deeply conditional/structured spaces → [TPE](#tpe)/[SMAC](#smac), not a GP.

*Is BO itself hyperparameter-free?*
- ❌ — kernel family, acquisition, $\xi$/$\beta$, initial design size, and the input warping all matter. The saving grace is that its hyperparams are far cheaper to get wrong than the ones it is tuning.
```

&nbsp;

### GP
- **Name**: Gaussian Process
- **What**: Distribution over functions in which any finite set of function values is jointly Gaussian.
- **Why**: The surrogate must give a **calibrated posterior**, not a fit.
    - Regression returns $\hat f(\mathbf{x})$; BO needs $p(f(\mathbf{x})|\mathcal{D})$ to know *where it is ignorant*.
    - Gaussian prior + Gaussian likelihood → the posterior mean & variance at any $\mathbf{x}$ are closed-form, exact, no sampling.
    - Nonparametric → capacity grows with the data, which is exactly right at $t\sim10$-$100$.
- **How**:
    1. Pick a mean function ($0$ after standardizing $y$) & a kernel encoding smoothness + per-dim length scales.
    2. Condition on $\mathcal{D}_t$ → closed-form Gaussian posterior at any test point.
    3. Fit kernel hyperparams (length scales, output scale, noise) by maximizing the **marginal likelihood**.
    4. Hand $\mu(\mathbf{x}),\sigma(\mathbf{x})$ to the acquisition.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $X_t\in\mathbb{R}^{t\times d}$: Observed inputs.
    - $\mathbf{y}\in\mathbb{R}^t$: Observed values.
    - $\mathbf{x}_*$: Test point.
- Params:
    - $k(\cdot,\cdot)$: Kernel.
    - $\theta_0$: Output scale.
    - $\ell_j$: Length scale of dim $j$ (ARD).
    - $\sigma_n^2$: Observation noise variance.
- Misc:
    - $K\in\mathbb{R}^{t\times t}$: Gram matrix, $K_{ij}=k(\mathbf{x}_i,\mathbf{x}_j)$.
    - $\mathbf{k}_*\in\mathbb{R}^t$: $[k(\mathbf{x}_*,\mathbf{x}_i)]_{i=1}^t$.

Prior:

$$
f\sim\mathcal{GP}(0,k)
$$

Posterior — closed form, no sampling:

$$
\mu(\mathbf{x}_*)=\mathbf{k}_*^T(K+\sigma_n^2I)^{-1}\mathbf{y}
$$

$$
\sigma^2(\mathbf{x}_*)=k(\mathbf{x}_*,\mathbf{x}_*)-\mathbf{k}_*^T(K+\sigma_n^2I)^{-1}\mathbf{k}_*
$$

Matérn-5/2 kernel — the BO default:

$$
k(\mathbf{x},\mathbf{x}')=\theta_0\left(1+\sqrt{5}r+\frac{5}{3}r^2\right)e^{-\sqrt{5}r},\qquad r=\sqrt{\sum_{j=1}^d\frac{(x_j-x'_j)^2}{\ell_j^2}}
$$

Kernel hyperparams by maximizing the log marginal likelihood:

$$
\log p(\mathbf{y}|X_t)=-\frac{1}{2}\mathbf{y}^T(K+\sigma_n^2I)^{-1}\mathbf{y}-\frac{1}{2}\log|K+\sigma_n^2I|-\frac{t}{2}\log2\pi
$$
- Term 1: Data fit.
- Term 2: Complexity penalty.
```

````{important} Code
:class: dropdown
```python
import torch

class GP:
    ## zero-mean GP with an ARD Matern-5/2 kernel
    def __init__(self, lengthscale=1.0, outputscale=1.0, noise=1e-4):
        self.ls, self.os, self.noise = lengthscale, outputscale, noise

    def _kernel(self, A, B):
        r = torch.cdist(A / self.ls, B / self.ls)
        s5r = 5 ** 0.5 * r
        return self.os * (1 + s5r + s5r ** 2 / 3) * torch.exp(-s5r)

    def fit(self, X, y):
        self.X, self.y_mean, self.y_std = X, y.mean(), y.std().clamp(min=1e-8)
        self.y = (y - self.y_mean) / self.y_std        ## zero-mean prior needs standardized y
        K = self._kernel(X, X) + self.noise * torch.eye(len(X))
        self.L = torch.linalg.cholesky(K)              ## O(t^3) once, O(t^2) per prediction after
        self.alpha = torch.cholesky_solve(self.y[:, None], self.L)
        return self

    def predict(self, Xs):
        Ks = self._kernel(Xs, self.X)
        mu = (Ks @ self.alpha).squeeze(-1)
        v = torch.cholesky_solve(Ks.T, self.L)
        ## variance depends ONLY on where you looked, never on what you saw
        var = (self._kernel(Xs, Xs).diag() - (Ks * v.T).sum(-1)).clamp(min=1e-12)
        return mu * self.y_std + self.y_mean, var.sqrt() * self.y_std

## Example
X = torch.tensor([[0.0], [1.0], [3.0]])
y = torch.tensor([0.0, 1.0, 0.5])
mu, sd = GP().fit(X, y).predict(torch.tensor([[1.0], [2.0]]))
print(mu.round(decimals=2), sd.round(decimals=2))
## tensor([1.0000, 0.6700]) tensor([0.0000, 0.2000]) -> sd ~0 at an observed point, larger in the gap
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- Exact closed-form posterior mean & variance.
- Uncertainty grows automatically away from the data → exploration for free.
- ARD length scales double as a per-dim **importance** measure.
- Marginal likelihood tunes the kernel with no held-out split — which matters at $t=15$.

*Cons?*
- $O(t^3)$ to fit, $O(t^2)$ per prediction.
- Everything hinges on the kernel; wrong smoothness → confidently wrong posterior.
- ❌Native categorical/conditional dims.
- Assumes homoscedastic Gaussian noise unless explicitly modelled.

*Why Matérn-5/2 rather than RBF?*
- RBF implies **infinitely** differentiable sample paths — absurdly smooth for a validation-score surface with kinks and noise, and it makes the posterior over-confident between points.
- Matérn-5/2 gives twice-differentiable paths: rough enough to be realistic, smooth enough that gradient-based acquisition optimization still works.

*Why maximize the marginal likelihood instead of cross-validating?*
- It **integrates** over $f$ rather than fitting it → it automatically penalizes over-flexible kernels (Occam's razor), and it needs no validation split.
- Caveat: with very few points it can still overfit the hyperparams → put priors on them, or marginalize by MCMC.

*Why standardize $y$ & normalize $\mathbf{x}$?*
- Zero-mean prior + unit output scale are baked-in assumptions. Un-standardized $y$ makes the prior drag predictions toward an arbitrary $0$; un-normalized $\mathbf{x}$ makes a shared length scale meaningless.

*Why does the posterior variance not depend on $\mathbf{y}$?*
- For a Gaussian likelihood, the posterior covariance is a function of the **inputs** only. → you can update uncertainty for a pending evaluation before its result arrives, which is precisely what [Batch BO](#batch-bo) exploits.
```

&nbsp;

### Acquisition Function
- **What**: Cheap scalar utility over the surrogate posterior whose argmax is the next point to evaluate.
- **Why**: The surrogate states what is *known*; something has to turn that into a **decision**.
    - Pure exploitation ($\arg\max\mu$) → traps in the first basin found.
    - Pure exploration ($\arg\max\sigma$) → never converges, just tiles the space.
    - → collapse the trade-off into one scalar that is cheap & differentiable, so the inner optimization is free.

&nbsp;

#### PI
- **Name**: Probability of Improvement
- **What**: Posterior probability of beating the incumbent.
- **Why**: The most literal utility — a point counts iff it beats the best so far.
- **How**: Standardize the gap to the incumbent → read the Gaussian CDF.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\Phi$: Standard normal CDF.
- Hyperparams:
    - $\xi\geq0$: Required margin of improvement.

$$
a_\text{PI}(\mathbf{x})=P\left(f(\mathbf{x})\geq f^++\xi\right)=\Phi(z),\qquad z=\frac{\mu(\mathbf{x})-f^+-\xi}{\sigma(\mathbf{x})}
$$
```

```{attention} Q&A
:class: dropdown
*Cons?*
- Rewards the **probability** of improvement, never its **size** → a point beating the incumbent by $10^{-9}$ with probability 0.99 outranks one beating it by 10 with probability 0.4.
- → pathologically exploitative at $\xi=0$: it samples right next to the incumbent forever.
- $\xi$ must be tuned and decayed by hand, and the right value is scale-dependent.

*When is it still useful?*
- When you genuinely only care about "is it better, yes or no" — e.g. a satisficing threshold rather than a maximization.
```

&nbsp;

#### EI
- **Name**: Expected Improvement {cite:p}`jones1998efficient`
- **What**: Expected **magnitude** of improvement over the incumbent.
- **Why**: PI throws away the size of the win.
    - Weight each improvement by how large it is → a small chance of a big gain competes fairly with a big chance of a tiny gain.
    - Under a Gaussian posterior the integral is closed-form → no sampling, and it is differentiable in $\mathbf{x}$.
- **How**: Integrate $\max(f(\mathbf{x})-f^+,0)$ under the posterior → it splits into an **exploit** term (the mean gap, weighted by how likely it is real) and an **explore** term (the std, weighted by how close the point is to the threshold).

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\phi$: Standard normal PDF.
    - $\Phi$: Standard normal CDF.
- Hyperparams:
    - $\xi$: Exploration offset, commonly $0.01$ on standardized $y$.

$$
a_\text{EI}(\mathbf{x})=\mathbb{E}\left[\max\left(f(\mathbf{x})-f^+-\xi,\ 0\right)\right]=\left(\mu(\mathbf{x})-f^+-\xi\right)\Phi(z)+\sigma(\mathbf{x})\phi(z)
$$

$$
z=\frac{\mu(\mathbf{x})-f^+-\xi}{\sigma(\mathbf{x})},\qquad a_\text{EI}(\mathbf{x})=0\ \text{ if }\ \sigma(\mathbf{x})=0
$$
- Term 1: Exploit.
- Term 2: Explore.
```

```{tip} Derivation
:class: dropdown
*Where does the closed form come from?* (take $\xi=0$)
1. Improvement: $I(\mathbf{x})=\max(f(\mathbf{x})-f^+,0)$, with $f(\mathbf{x})\sim\mathcal{N}(\mu,\sigma^2)$.
2. Substitute $f=\mu+\sigma u$, $u\sim\mathcal{N}(0,1)$ → $I>0\Leftrightarrow u>-z$ where $z=\frac{\mu-f^+}{\sigma}$.
3. $\mathbb{E}[I]=\int_{-z}^{\infty}\left(\mu-f^++\sigma u\right)\phi(u)\,du$.
4. First piece: $(\mu-f^+)\int_{-z}^{\infty}\phi(u)\,du=(\mu-f^+)\Phi(z)$ by symmetry of $\phi$.
5. Second piece: $\phi'(u)=-u\phi(u)$ → $\int_{-z}^{\infty}u\phi(u)\,du=\left[-\phi(u)\right]_{-z}^{\infty}=\phi(z)$ → contributes $\sigma\phi(z)$.
6. Sum → $(\mu-f^+)\Phi(z)+\sigma\phi(z)$.
7. $\sigma\to0$: $\phi(z)\to0$ and $\Phi(z)\to\mathbb{1}[\mu>f^+]$, both terms $\to0$ for any already-observed point → EI never re-samples a noiselessly-measured location.
```

````{important} Code
:class: dropdown
```python
import torch
## Reuses the GP class from the Gaussian Process block above

def expected_improvement(gp, Xs, best, xi=0.01):
    mu, sd = gp.predict(Xs)
    z = (mu - best - xi) / sd
    normal = torch.distributions.Normal(0.0, 1.0)
    ## term 1 = exploit (mean gap), term 2 = explore (uncertainty)
    ei = (mu - best - xi) * normal.cdf(z) + sd * torch.exp(normal.log_prob(z))
    return ei.clamp(min=0)                    ## sigma -> 0 => EI -> 0

## Example: 1-D BO loop on a black box
f = lambda x: (-(x - 2.0) ** 2 + 3).squeeze(-1)
X, y = torch.tensor([[0.0], [1.0], [5.0]]), None
y = f(X)
grid = torch.linspace(0, 5, 201)[:, None]
for _ in range(6):
    gp = GP(lengthscale=1.0).fit(X, y)
    x_next = grid[expected_improvement(gp, grid, y.max()).argmax()][None]
    X, y = torch.cat([X, x_next]), torch.cat([y, f(x_next)])
print(X[y.argmax()].round(decimals=2))        ## tensor([2.0000]) -> found in 6 evaluations
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- Closed form, cheap, differentiable → multi-start L-BFGS solves the inner problem.
- Usable with $\xi=0$, unlike PI.
- Self-balancing: $\Phi$ dominates near the incumbent, $\phi\sigma$ dominates in unexplored regions.
- The default acquisition in essentially every BO library.

*Cons?*
- **Myopic** — 1-step greedy, never plans for the remaining budget.
- Under noise, $f^+$ = best *observed* value is upward-biased by the noise → use $\max_i\mu(\mathbf{x}_i)$ ("noisy EI") instead.
- Numerically underflows to exactly 0 far from the data (short length scales, high $d$) → the inner optimizer sees a flat surface → use **log-EI**.

*What does $\xi$ actually do?*
- Demands a minimum margin before a point counts as an improvement → ⬆️$\xi$ → ⬆️exploration. It is scale-dependent, hence "0.01 on standardized $y$".

*EI vs UCB in one line?*
- EI asks "how much better, in expectation". UCB asks "how good could it plausibly be". EI is better calibrated out of the box; UCB is the one with a regret bound.
```

&nbsp;

#### UCB
- **Name**: Upper Confidence Bound {cite:p}`srinivas2010gaussian`
- **What**: Optimistic score $\mu+\sqrt{\beta}\sigma$.
- **Why**: EI's trade-off is implicit and cannot be dialled.
    - "Optimism in the face of uncertainty" makes exploration a single explicit knob.
    - And, uniquely among the common acquisitions, the right $\beta_t$ schedule yields **sublinear cumulative regret** → provably no-regret.
- **How**: Score every point by its optimistic upper confidence bound; take the argmax.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\beta_t$: Exploration weight at round $t$.
    - $\delta$: Failure probability.
- Misc:
    - $\gamma_T$: Maximum information gain about $f$ obtainable from $T$ observations.

Acquisition:

$$
a_\text{UCB}(\mathbf{x})=\mu(\mathbf{x})+\sqrt{\beta_t}\,\sigma(\mathbf{x})
$$

No-regret schedule on a **finite** domain:

$$
\beta_t=2\log\left(\frac{|\mathcal{X}|t^2\pi^2}{6\delta}\right)
$$

Regret bound, w.p. $\geq1-\delta$:

$$
R_T=O^*\left(\sqrt{T\gamma_T\beta_T}\right)
$$
```

```{attention} Q&A
:class: dropdown
*Pros?*
- One interpretable knob, monotone in exploration.
- The only common acquisition with a no-regret guarantee.
- Extends cleanly to constraints & batches.

*Cons?*
- The theoretical $\beta_t$ is far too conservative → over-explores in practice → people hard-code $\sqrt{\beta}\in[1,3]$ and forfeit the guarantee.
- Directly inherits any miscalibration of $\sigma$ — a mis-scaled posterior mis-scales exploration linearly.

*LCB?*
- Identical for **minimization**: $\mu-\sqrt{\beta}\sigma$. Sign convention only.

*Why does $\beta_t$ grow with $t$?*
- The bound must hold simultaneously at every point and every round → a union bound over $|\mathcal{X}|$ and over $t$ → the confidence width has to widen logarithmically to keep the total failure probability at $\delta$.
```

&nbsp;

#### TS
- **Name**: Thompson Sampling
- **What**: One function drawn from the posterior; its argmax is the next point.
- **Why**: PI/EI/UCB all need an explicit trade-off formula or constant.
    - Drawing from the posterior encodes the trade automatically: a point is selected in proportion to its **posterior probability of being the optimum**.
    - Randomized by construction → independent draws give diverse points → parallelism for free.
- **How**:
    1. Draw a sample path $\tilde f\sim p(f|\mathcal{D}_t)$ — exactly on a discretization, or approximately via random Fourier features.
    2. Evaluate at $\arg\max\tilde f$.

```{attention} Q&A
:class: dropdown
*Pros?*
- ❌tuning parameter of any kind.
- **Batching for free**: $q$ independent draws → $q$ diverse points, no penalization machinery needed.
- Has regret guarantees, and is the workhorse acquisition for [High-Dim BO](#high-dim-bo).

*Cons?*
- Exact path sampling is $O(t^3)$; random-feature approximations distort the tails, which is exactly where the argmax lives.
- Higher variance than EI in low dim → can look erratic run-to-run.

*Why does randomization suffice as exploration?*
- The probability that $\mathbf{x}$ is chosen equals the posterior probability that $\mathbf{x}$ is the maximizer. → uncertain regions get chosen exactly as often as the evidence says they deserve, with no explicit bonus term.
```

&nbsp;

#### KG
- **Name**: Knowledge Gradient {cite:p}`frazier2008knowledge`
- **What**: Expected improvement in the **posterior optimum** after taking one more observation.
- **Why**: EI credits a point only if that point itself turns out to be good.
    - Wrong accounting under noise, and wrong whenever the final answer will be **reported from the model** rather than from an evaluated point.
    - → value a measurement by how much it improves the best-we-would-report. It will happily pay to measure somewhere it already knows is bad, if doing so pins down where the optimum is.
- **How**: For each candidate, simulate the observation, refit the posterior, and take the expected rise in $\max_\mathbf{x}\mu$.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\mu_t$: Posterior mean after $t$ observations.
    - $\mu_{t+1}$: Posterior mean after additionally observing $y$ at $\mathbf{x}$.

$$
a_\text{KG}(\mathbf{x})=\mathbb{E}_{y|\mathbf{x},\mathcal{D}_t}\left[\max_{\mathbf{x}'}\mu_{t+1}(\mathbf{x}')\right]-\max_{\mathbf{x}'}\mu_t(\mathbf{x}')
$$
```

```{attention} Q&A
:class: dropdown
*Pros?*
- Correct under noise, and correct when the reported solution need not be a point you evaluated.
- Strictly generalizes EI: noiseless observations + "report an evaluated point" → KG reduces to EI.

*Cons?*
- A nested optimization inside an expectation → far more expensive per iteration; needs Monte Carlo + the reparameterization trick to be differentiable.
- Overkill for cheap or low-noise problems, where EI is indistinguishable.

*Concrete case where EI fails and KG does not?*
- Heavily noisy $f$: EI chases $f^+$, which is the largest **noise realization** seen so far, not the largest true value → it repeatedly exploits a lucky observation. KG never looks at $f^+$ at all.
```

&nbsp;

#### Entropy Search
- **What**: Acquisition scoring a point by the information it reveals about the **location** of the optimum {cite:p}`hennig2012entropy`
- **Why**: EI/PI/UCB/KG all value a point through function *values*.
    - The actual goal is not a good $y$ — it is **knowing where $\mathbf{x}^*$ is**.
    - → maximize the mutual information between the next observation and $\mathbf{x}^*$ → an explicitly information-theoretic objective rather than a heuristic utility.
- **How**:
    1. Maintain a posterior over $\mathbf{x}^*$ induced by the GP posterior.
    2. Score each candidate by the expected reduction in that distribution's entropy.
    3. Evaluate the maximizer.
    - **PES** (Predictive Entropy Search) computes the same mutual information in the symmetric, cheaper direction: entropy of $y$ minus its entropy conditioned on $\mathbf{x}^*$.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $H[\cdot]$: Differential entropy.
    - $p(\mathbf{x}^*|\mathcal{D}_t)$: Posterior over the argmax.

Mutual information between the observation at $\mathbf{x}$ and the optimum's location:

$$
a_\text{PES}(\mathbf{x})=H\left[p(y|\mathcal{D}_t,\mathbf{x})\right]-\mathbb{E}_{\mathbf{x}^*|\mathcal{D}_t}\left[H\left[p(y|\mathcal{D}_t,\mathbf{x},\mathbf{x}^*)\right]\right]
$$
```

```{attention} Q&A
:class: dropdown
*Pros?*
- Directly optimizes what you actually want (identify $\mathbf{x}^*$), not a proxy.
- Non-myopic in spirit: it will probe a region purely to disambiguate two candidate optima.

*Cons?*
- $p(\mathbf{x}^*|\mathcal{D}_t)$ has no closed form → every variant needs heavy approximation (expectation propagation, sampled argmaxes).
- ⬆️Cost per iteration → only pays off when $f$ is genuinely expensive.
```

&nbsp;

#### MES
- **Name**: Max-value Entropy Search {cite:p}`wang2017max`
- **What**: Entropy Search w.r.t. the optimum's **value** $f^*$ instead of its location $\mathbf{x}^*$.
- **Why**: $\mathbf{x}^*$ is $d$-dimensional and its posterior is intractable.
    - $f^*$ is a **scalar** → its posterior can be sampled cheaply (e.g. via a Gumbel approximation of the max).
    - Same information-theoretic principle, drastically cheaper and lower-variance.
- **How**: Sample a set of plausible $f^*$ values → each induces a truncated-Gaussian conditional at $\mathbf{x}$ → average the entropy reduction in closed form.

```{note} Math
:class: dropdown
$$
a_\text{MES}(\mathbf{x})=H\left[p(y|\mathcal{D}_t,\mathbf{x})\right]-\mathbb{E}_{f^*|\mathcal{D}_t}\left[H\left[p(y|\mathcal{D}_t,\mathbf{x},f^*)\right]\right]
$$
- Conditioning on $f^*$ truncates the posterior at $\mathbf{x}$ from above → the inner entropy is that of a truncated Gaussian → closed form.
```

```{attention} Q&A
:class: dropdown
*Why is it strictly more practical than PES?*
- Sampling a scalar $f^*$ vs sampling a $d$-dim argmax: cheaper, lower variance, and the inner entropy is analytic instead of approximated.

*Cons?*
- Knowing $f^*$ precisely does NOT pin down $\mathbf{x}^*$ when the function has near-ties → on multi-modal plateaus it can under-explore relative to PES.
```

&nbsp;

```{dropdown} Table: Acquisition Functions
| Acquisition | Values | Closed form | Tuning | Notes |
|:--|:--|:--|:--|:--|
| [PI](#pi) | $P(\text{improve})$ | ✅ | $\xi$ (critical) | Over-exploits; ignores magnitude |
| [EI](#ei) | $\mathbb{E}[\text{improvement}]$ | ✅ | $\xi$ (optional) | Default everywhere; myopic; noise-sensitive |
| [UCB](#ucb) | Optimistic bound | ✅ | $\beta$ | Only one with a no-regret bound |
| [TS](#ts) | Posterior draw's argmax | Sampling | ❌ | Free batching; high variance |
| [KG](#kg) | $\Delta$ posterior optimum | ❌ (MC) | ❌ | Correct under noise; expensive |
| [Entropy Search](#entropy-search) | Info about $\mathbf{x}^*$ | ❌ (approx) | ❌ | Most principled, least practical |
| [MES](#mes) | Info about $f^*$ | Partly | #$f^*$ samples | ES's practical form |

Rows are ordered by increasing sophistication, which is also increasing cost. In practice EI is the default and UCB is the fallback when EI's incumbent is noise-corrupted.
```

&nbsp;

### TPE
- **Name**: Tree-structured Parzen Estimator {cite:p}`bergstra2011algorithms`
- **What**: Density-ratio surrogate — model $p(\mathbf{x}|y)$ instead of $p(y|\mathbf{x})$.
- **Why**: A [GP](#gp) is the wrong model for a real hyperparam space.
    - $O(t^3)$, and no native handling of categorical, integer or **conditional** dims (`n_layers` decides how many `layer_size` params even exist).
    - → invert the modelling direction: split observations by quality and model the **input** density of each group. Densities over structured, conditional spaces are easy; regression onto them is not.
    - The density ratio turns out to be monotone in EI → maximizing the ratio maximizes EI, without ever fitting $p(y|\mathbf{x})$.
- **How**:
    1. Split the observations at the $\gamma$-quantile of $y$ → "good" & "bad".
    2. Fit a kernel density estimate to each group's inputs: $l(\mathbf{x})$ on good, $g(\mathbf{x})$ on bad. Conditional structure → a **tree** of KDEs, one per node of the search space.
    3. Draw many candidates from $l$.
    4. Return $\arg\max\frac{l(\mathbf{x})}{g(\mathbf{x})}$.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $y^*$: The $\gamma$-quantile of the observed $y$ (minimization convention).
    - $l(\mathbf{x})$: KDE over inputs with $y<y^*$.
    - $g(\mathbf{x})$: KDE over inputs with $y\geq y^*$.
- Hyperparams:
    - $\gamma$: Split quantile; $0.15$ in the original paper.

Surrogate:

$$
p(\mathbf{x}|y)=\begin{cases}l(\mathbf{x})&y<y^*\\g(\mathbf{x})&y\geq y^*\end{cases},\qquad\gamma=P(y<y^*)
$$

Resulting acquisition:

$$
a_\text{EI}(\mathbf{x})\propto\left(\gamma+\frac{g(\mathbf{x})}{l(\mathbf{x})}(1-\gamma)\right)^{-1}\quad\Rightarrow\quad\arg\max a_\text{EI}=\arg\max\frac{l(\mathbf{x})}{g(\mathbf{x})}
$$
```

```{tip} Derivation
:class: dropdown
*Why is the density ratio equivalent to EI?* (minimization, improvement below the threshold $y^*$)
1. $a_\text{EI}(\mathbf{x})=\int_{-\infty}^{y^*}(y^*-y)\,p(y|\mathbf{x})\,dy$, and Bayes gives $p(y|\mathbf{x})=\frac{p(\mathbf{x}|y)p(y)}{p(\mathbf{x})}$.
2. Denominator: $p(\mathbf{x})=\int p(\mathbf{x}|y)p(y)\,dy=\gamma\,l(\mathbf{x})+(1-\gamma)\,g(\mathbf{x})$.
3. Numerator: on $y<y^*$ the model says $p(\mathbf{x}|y)=l(\mathbf{x})$, a constant in $y$ → it factors out:

    $$
    \int_{-\infty}^{y^*}(y^*-y)p(\mathbf{x}|y)p(y)\,dy=l(\mathbf{x})\int_{-\infty}^{y^*}(y^*-y)p(y)\,dy=l(\mathbf{x})\cdot c
    $$

    with $c=\gamma y^*-\int_{-\infty}^{y^*}y\,p(y)\,dy>0$ independent of $\mathbf{x}$.
4. Divide → $a_\text{EI}(\mathbf{x})=\frac{c\,l(\mathbf{x})}{\gamma l(\mathbf{x})+(1-\gamma)g(\mathbf{x})}=c\left(\gamma+\frac{g(\mathbf{x})}{l(\mathbf{x})}(1-\gamma)\right)^{-1}$.
5. Monotone decreasing in $\frac{g}{l}$ → maximizing EI $\Leftrightarrow$ maximizing $\frac{l}{g}$.
6. → sample candidates from $l$ (where good points live) and rank them by $\frac{l}{g}$ (down-weighting where bad points also live).
```

````{important} Code
:class: dropdown
```python
import torch

class TPE:
    def __init__(self, gamma=0.15, bandwidth=0.1, n_candidates=64):
        self.gamma, self.bw, self.n_cand = gamma, bandwidth, n_candidates

    def _kde(self, points, xs):
        ## Parzen estimator: one Gaussian bump per observation
        d = (xs[:, None, :] - points[None, :, :]) / self.bw
        return torch.exp(-0.5 * (d ** 2).sum(-1)).mean(1) + 1e-12

    def suggest(self, X, y, lo, hi):
        k = max(1, int(self.gamma * len(y)))
        idx = y.argsort()                                  ## minimization: small y is "good"
        good, bad = X[idx[:k]], X[idx[k:]]
        ## candidates are drawn from l(x) itself -> jitter around known-good points
        cand = good[torch.randint(len(good), (self.n_cand,))] + self.bw * torch.randn(self.n_cand, X.shape[1])
        cand = cand.clamp(lo, hi)
        ratio = self._kde(good, cand) / self._kde(bad, cand)
        return cand[ratio.argmax()]                        ## argmax l/g == argmax EI

## Example
f = lambda x: ((x - 0.3) ** 2).sum(-1)
torch.manual_seed(0)
X = torch.rand(20, 1)
y = f(X)
tpe = TPE()
for _ in range(40):
    x_next = tpe.suggest(X, y, 0.0, 1.0)[None]
    X, y = torch.cat([X, x_next]), torch.cat([y, f(x_next)])
print(X[y.argmin()].round(decimals=2))                     ## tensor([0.3000])
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- $O(t)$ per iteration → ❌cubic scaling, runs to $10^4+$ trials.
- Native conditional, categorical & integer dims.
- Robust default with almost no tuning; the engine behind Hyperopt & Optuna.

*Cons?*
- Models dims **independently** within each tree node → ❌hyperparam interactions. (Multivariate TPE partially repairs this with a joint KDE.)
- ❌Calibrated uncertainty → no UCB, no regret bound, no principled stopping rule.
- Sensitive to $\gamma$ and to the KDE bandwidth, both of which are usually left at defaults.

*Why model $p(\mathbf{x}|y)$ instead of $p(y|\mathbf{x})$?*
- All the awkward structure lives in $\mathbf{x}$ (categorical, conditional, mixed); $y$ is a scalar.
- Fitting densities over $\mathbf{x}$ conditioned on a scalar split is far easier than regressing a scalar onto a structured $\mathbf{x}$.

*Where does "tree-structured" come from?*
- The **search space** is a tree — a hyperparam exists only when its parent takes certain values — so the density is a tree of conditional KDEs. Nothing to do with decision trees.

*Why does it sample candidates from $l$?*
- $\frac{l}{g}$ must be maximized over a structured space with no gradient. Sampling from $l$ concentrates candidates exactly where the ratio can be large → a cheap, structure-respecting inner optimizer.
```

&nbsp;

### SMAC
- **Name**: Sequential Model-based Algorithm Configuration {cite:p}`hutter2011sequential`
- **What**: SMBO with a **random forest** surrogate + an intensification schedule.
- **Why**: GPs fail on algorithm-configuration spaces.
    - Dozens of categorical & conditional params with no natural metric → no sensible kernel.
    - The objective is an average over many **problem instances** → heavily non-stationary & heteroscedastic.
    - RF regression splits on categoricals natively, costs $O(t\log t)$, and its cross-tree spread is a usable (if uncalibrated) uncertainty estimate.
- **How**:
    1. Fit an RF on all $(\mathbf{x}_i,y_i)$.
    2. $\mu(\mathbf{x})$ = mean across trees; $\sigma^2(\mathbf{x})$ = variance across trees.
    3. Maximize EI, searched by local mutation around incumbents + random candidates (the surrogate is piecewise-constant, so ❌gradients).
    4. **Intensification**: a challenger must beat the incumbent on progressively more instances/seeds before it is allowed to replace it.

```{attention} Q&A
:class: dropdown
*Pros?*
- Native categorical / conditional / integer dims.
- Scales to $10^4+$ evaluations.
- Intensification makes it robust to a noisy, multi-instance objective — budget is spent confirming a challenger only as long as it keeps winning.
- The standard engine for AutoML (auto-sklearn) and SAT/MIP solver configuration.

*Cons?*
- RF uncertainty is **not** calibrated: it does not grow properly far from the data, and extrapolation is flat by construction.
- Piecewise-constant surrogate → the acquisition surface has no gradient → the inner search must be local search, not L-BFGS.
- Loses to a GP on smooth, low-dim, continuous problems.

*Why is intensification necessary at all?*
- With a per-instance objective, one lucky instance can make a bad config look great. Racing the challenger against the incumbent on shared instances/seeds removes that variance without paying for a full evaluation of every config.
```

&nbsp;

### BOHB
- **Name**: Bayesian Optimization & HyperBand {cite:p}`falkner2018bohb`
- **What**: [Hyperband](#hyperband)'s budget schedule with random sampling replaced by a [TPE](#tpe)-style model.
- **Why**: The two parents fail in exactly complementary ways.
    - Hyperband: excellent **any-time** performance, but samples at random forever → its final quality plateaus.
    - BO: excellent **final** performance, but every evaluation is a full training run → slow start.
    - → keep Hyperband's brackets, and let a model choose *which* configs enter them.
- **How**:
    1. Run Hyperband's bracket schedule unchanged.
    2. Maintain a TPE-style KDE pair per budget level.
    3. Sample new configs from the KDE of the **largest budget** that has enough observations; otherwise sample at random.
    4. Always keep a fixed fraction of purely random samples.

```{attention} Q&A
:class: dropdown
*Pros?*
- Any-time performance of Hyperband AND final performance of BO.
- Parallel by construction (Hyperband's brackets are).
- The random fraction preserves Hyperband's worst-case robustness → it cannot be trapped by a misfit model.

*Cons?*
- Inherits TPE's per-dim independence assumption.
- Still needs a low-budget proxy that genuinely correlates with the final score.
- More moving parts (budget levels × KDEs) → more ways to misconfigure.

*Why fit the KDE at the largest sufficiently-populated budget?*
- Low-budget observations are plentiful but **biased**; high-budget observations are faithful but **few**.
- Taking the highest budget that clears a minimum count is the compromise between sample size and fidelity.

*Why keep a random fraction at all?*
- Insurance. If the KDE is misfit (wrong bandwidth, misleading early observations), pure model-driven sampling can collapse onto a bad region permanently. The random stream guarantees continued coverage.
```

&nbsp;

### Constrained BO
- **What**: BO where feasibility is itself an unknown, expensive black box.
- **Why**: Real objectives come with constraints you cannot check without running the experiment.
    - Latency < 10ms, memory < 8GB, fairness gap < $\epsilon$ — all unknown until the model is trained & benchmarked.
    - Filtering infeasible points *after* evaluating them burns the entire budget on unusable configs.
    - → model each constraint with its own surrogate & fold feasibility into the acquisition **before** spending an evaluation.
- **How**:
    1. One GP for $f$, one GP per constraint $c_k$.
    2. Multiply EI by the posterior probability that every constraint holds.
    3. Define the incumbent as the best **feasible** observation.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $c_k(\mathbf{x})\leq0$: Unknown constraint $k$.

Constrained EI, assuming the constraints are conditionally independent given $\mathbf{x}$ {cite:p}`gardner2014bayesian`:

$$
a_\text{EIC}(\mathbf{x})=a_\text{EI}(\mathbf{x})\prod_kP\left(c_k(\mathbf{x})\leq0\ \middle|\ \mathcal{D}_t\right)
$$
- Each factor is a Gaussian CDF under a GP posterior on $c_k$.
```

```{attention} Q&A
:class: dropdown
*What if nothing feasible has been observed yet?*
- $f^+$ is undefined → EI is undefined. → fall back to maximizing $\prod_kP(c_k\leq0)$ alone until the first feasible point appears.

*Cons?*
- The product assumes independent constraints; correlated ones (latency & memory) are mis-scored.
- ❌Distinguishes "barely feasible" from "safely feasible" → for safety-critical settings use a safe-BO variant that never evaluates a point whose feasibility is uncertain.
- Doubles-plus the surrogate fitting cost (one GP per constraint).

*Why not just add a penalty to $f$?*
- A penalty requires knowing the constraint value, which is exactly what you cannot observe without paying. And it conflates "infeasible" with "bad objective", which the surrogate then has to unlearn.
```

&nbsp;

### Multi-Objective BO
- **What**: BO returning a **Pareto front** instead of one optimum.
- **Why**: Real model selection trades accuracy against latency, size, cost, fairness.
    - Fixed-weight scalarization picks one point on the front and hides the trade-off entirely.
    - Worse, sweeping a linear weight can only reach the **convex hull** of the front — non-convex regions are unreachable at any weight ([composite objectives](obj.md#composite-objective)).
    - → target the front itself.
- **How**: Two families.
    - **ParEGO** {cite:p}`knowles2006parego`: draw fresh **augmented Chebyshev** scalarization weights each iteration → one GP, standard EI → the front is assembled across iterations.
    - **EHVI**: expected increase in the **hypervolume** dominated by the observed front w.r.t. a reference point → one GP per objective → directly front-aware.

```{attention} Q&A
:class: dropdown
*Why Chebyshev rather than linear scalarization?*
- Linear weights recover only the convex hull of the front. The augmented Chebyshev scalarization $\max_k w_k\tilde f_k+\rho\sum_kw_k\tilde f_k$ can reach non-convex regions, so randomizing $\mathbf{w}$ actually sweeps the whole front.

*Cons of EHVI?*
- Hypervolume computation is exponential in #objectives → practical up to ~4.
- It needs a reference point, and the answer depends on where you put it.

*How do you even report the answer?*
- The front, not a point. The choice among front points is a **preference** decision, not an optimization one — which is exactly why it should not be baked into the objective.
```

&nbsp;

### Batch BO
- **What**: BO proposing $q$ points per round instead of 1.
- **Why**: SMBO is sequential; hardware is parallel.
    - $q$ workers idle while one config trains → wall-clock is $q\times$ worse than necessary.
    - Naively taking the top-$q$ acquisition values returns $q$ nearly identical points — the acquisition does not know the other $q-1$ are already pending.
    - → the batch must be selected **jointly**, with redundancy penalized.
- **How**: Four standard routes.
    - **qEI** {cite:p}`wilson2018maximizing`: the true joint EI of a batch, estimated by Monte Carlo + the reparameterization trick → differentiable → optimize all $q$ points together by gradient ascent.
    - **Fantasizing / Kriging believer**: pick greedily, then condition the posterior on a *hallucinated* outcome (usually $\mu$) before picking the next.
    - **Local penalization**: multiply the acquisition by a decaying factor around each already-chosen point.
    - **[Thompson Sampling](#ts)**: $q$ independent posterior draws → diversity with no extra machinery.

```{attention} Q&A
:class: dropdown
*Why is fantasizing valid at all?*
- Under a Gaussian likelihood, the GP posterior **variance depends only on the input locations**, never on the observed values.
- → conditioning on a pending point with a made-up $y$ gives the *exactly correct* updated uncertainty, and only the mean is hallucinated.

*What does batching cost?*
- Sample efficiency **per evaluation** drops — you commit to $q$ points before seeing any of their results — while wall-clock improves. The gap widens with $q$.

*Rule of thumb for $q$?*
- Keep $q$ well below the number of evaluations you can afford in total; $q$ approaching the budget degenerates to random search.
```

&nbsp;

### High-Dim BO
- **What**: BO variants for $d\gg20$.
- **Why**: A GP cannot learn a $d$-dim function from $O(d)$ points.
    - Volume concentrates near the boundary → nearly every candidate is far from all data → $\sigma$ sits at the prior almost everywhere.
    - → the acquisition becomes undirected exploration, and EI in particular flattens to numerically zero, so the inner optimizer returns noise.
    - → the only fix is to impose structure: low effective dimension, sparsity, or locality.
- **How**: Three families.
    - **Random embeddings** (REMBO / ALEBO): assume a low **effective** dimension → optimize inside a random low-dim subspace and map back.
    - **Sparse priors** — SAASBO {cite:p}`eriksson2021high`: a hierarchical half-Cauchy prior on inverse length scales shrinks most dims to irrelevance, letting the few that matter survive.
    - **Local models** — TuRBO {cite:p}`eriksson2019scalable`: maintain several trust regions, fit an independent local GP in each, and allocate evaluations across them by Thompson sampling.

```{attention} Q&A
:class: dropdown
*Why is locality the fix rather than a better global model?*
- A global GP must explain the whole domain with one set of length scales, so it is forced to be either too smooth or too jumpy.
- Restricting to a trust region makes the local function nearly stationary → the model is accurate exactly where it is used → and the region shrinks/grows on failure/success precisely like a classical [Trust Region](#trust-region).

*When does the low-effective-dimension assumption break?*
- Genuinely coupled dims — e.g. architecture search where every layer's width matters — → a random embedding destroys the signal it needs.

*Practical alternative?*
- Above $d\sim100$ with a cheap-ish objective, [CMA-ES](#cma-es) is frequently the stronger choice: it also learns a low-rank structure (the covariance), but scales far better in #evaluations.
```

&nbsp;

## Evolution
- **What**: Population-based black-box search driven by **variation** + **selection**.

### GA
- **Name**: Genetic Algorithm {cite:p}`holland1975adaptation`
- **What**: Population of encoded candidates evolved by selection, crossover & mutation.
- **Why**: Some search spaces admit nothing but sampling.
    - ❌gradient — discrete structures, simulators, non-differentiable metrics (AUC, a compiler's output, a physical experiment).
    - Rugged & multi-modal → any single-incumbent method ([SA](#sa), hill climbing) commits to one basin at a time.
    - A **population** occupies many basins simultaneously, and recombination lets them exchange information — something no single-state method can do.
- **How**:
    1. **Initialize**: sample a population of genotypes at random.
    2. **Evaluate**: fitness of every individual.
    3. **Select**: parents drawn with probability increasing in fitness (roulette / tournament / rank).
    4. **Crossover**: recombine two parents into offspring, w.p. $p_c$.
    5. **Mutate**: perturb each gene w.p. $p_m$.
    6. **Replace**: build the next generation — generational, steady-state, or **elitist**.
    7. Repeat 2-6 until the budget or a convergence criterion is hit.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathcal{P}_t$: Population at generation $t$.
    - $F(\mathbf{x})$: Fitness, maximized.
- Hyperparams:
    - $\mu$: Population size.
    - $p_c$: Crossover probability.
    - $p_m$: Per-gene mutation probability, commonly $\frac{1}{n}$.
    - $s$: Tournament size.
    - $\sigma$: Mutation std.
- Misc:
    - $p_i$: Selection probability of individual $i$.
    - $\alpha\sim\text{Uniform}(0,1)$: Blend coefficient.
    - $\mathbf{o}$: Offspring.

Fitness-proportional (roulette) selection — requires $F>0$:

$$
p_i=\frac{F(\mathbf{x}^{(i)})}{\sum_{j=1}^{\mu}F(\mathbf{x}^{(j)})}
$$

Tournament selection: draw $s$ individuals uniformly, keep the best. Selection pressure ⬆️ with $s$, and it is **invariant to fitness scaling**.

Real-coded blend crossover:

$$
\mathbf{o}=\alpha\mathbf{x}^{(i)}+(1-\alpha)\mathbf{x}^{(j)}
$$

Gaussian mutation, applied per gene w.p. $p_m$:

$$
o_k\leftarrow o_k+\epsilon,\qquad\epsilon\sim\mathcal{N}(0,\sigma^2)
$$
```

````{important} Code
:class: dropdown
```python
import torch

class GA:
    def __init__(self, f, n_dim, mu=50, p_m=None, sigma=0.3, tournament=3, elite=1):
        self.f, self.n, self.mu = f, n_dim, mu
        self.p_m = p_m if p_m is not None else 1.0 / n_dim   ## one gene per genome, on average
        self.sigma, self.k, self.elite = sigma, tournament, elite

    def _select(self, pop, fit):
        ## tournament: rank-based -> immune to fitness rescaling, unlike roulette
        idx = torch.randint(len(pop), (self.mu, self.k))
        return pop[idx[torch.arange(self.mu), fit[idx].argmin(1)]]

    def run(self, n_gen=200):
        pop = torch.randn(self.mu, self.n)
        for _ in range(n_gen):
            fit = self.f(pop)
            keep = pop[fit.argsort()[:self.elite]]           ## elitism: never lose the best
            parents = self._select(pop, fit)
            a = torch.rand(self.mu, 1)
            pop = a * parents + (1 - a) * parents[torch.randperm(self.mu)]   ## blend crossover
            mask = torch.rand(self.mu, self.n) < self.p_m
            pop = pop + mask * self.sigma * torch.randn(self.mu, self.n)     ## mutation
            pop[:self.elite] = keep
        fit = self.f(pop)
        return pop[fit.argmin()], fit.min()

## Example: Rastrigin-like, many local minima, global minimum at 0
f = lambda X: (X ** 2 - 10 * torch.cos(2 * torch.pi * X) + 10).sum(-1)
best, fbest = GA(f, n_dim=2).run()
print(best.round(decimals=2), fbest.round(decimals=2))   ## tensor([0., 0.]) tensor(0.)
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- ❌gradient, ❌continuity, ❌convexity; handles discrete, mixed & variable-length representations.
- Embarrassingly parallel — the whole population evaluates independently.
- Population diversity → many basins searched at once.
- Extends directly to multi-objective ([NSGA-II](#nsga-ii)) and to constraints (repair or penalty).

*Cons?*
- Sample-inefficient — thousands of evaluations for what [BO](#smbo) does in tens.
- Many interacting hyperparams ($\mu$, $p_c$, $p_m$, selection scheme, encoding), and performance depends on all of them.
- **Premature convergence**: one strong individual takes over the population.
- ❌Convergence guarantee, ❌stopping certificate.

*Why does crossover help at all?*
- Building-block intuition: if fitness partly decomposes into independently good sub-solutions, recombination assembles them in a single step, while mutation would have to rediscover each.
- When fitness does NOT decompose (**deceptive** or highly **epistatic** problems), crossover is neutral or actively harmful — which is exactly when mutation-only [ES](#es) wins.

*How to fix premature convergence?*
- Tournament/rank selection instead of fitness-proportional → removes sensitivity to the fitness scale, which is what lets one outlier dominate.
- Fitness sharing / crowding → penalize crowded regions.
- ⬆️$\mu$, ⬆️$p_m$, or restarts.

*Elitism — why and why not?*
- ✅ The best individual can never be lost → best-so-far is monotone.
- ❌ It accelerates takeover → ⬇️diversity. Standard compromise: carry over the top 1-2 only.

*GA vs SA?*
- Both accept non-improving states. GA is population-based, parallel, with recombination; [SA](#sa) is single-state, sequential, with a temperature schedule.
- GA explores many basins at once; SA tunnels between them one at a time.
```

&nbsp;

### Genetic Programming
- **What**: Evolution over **programs** (expression trees) rather than fixed-length vectors {cite:p}`koza1992genetic`
- **Why**: Some search spaces are spaces of *structures*, not of numbers.
    - Symbolic regression, controller synthesis, feature construction: the answer is a formula whose size & shape are unknown a priori.
    - A fixed-length genotype cannot represent a variable-size tree → the variation operators themselves must be structure-aware.
- **How**:
    1. Genotype = expression tree over a function set ($+,\times,\sin,\text{if}$) & a terminal set (variables, constants).
    2. Initialize by random tree growth (grow / full / ramped half-and-half).
    3. **Crossover** = swap random subtrees between two parents.
    4. **Mutation** = replace a random subtree with a freshly generated one.
    5. Fitness = error on data $+$ a **parsimony** penalty on tree size.

```{attention} Q&A
:class: dropdown
*Pros?*
- Output is a **human-readable formula** → interpretable by construction, not by post-hoc explanation.
- Structure & params are discovered together — no fixed hypothesis class.
- Its modern form (symbolic regression) genuinely recovers closed-form physical laws from data.

*Cons?*
- **Bloat**: trees grow without any fitness improvement.
- Enormous, massively redundant search space → sample-hungry even by evolutionary standards.
- Subtree crossover is highly disruptive — a tiny structural edit can arbitrarily change semantics, so the fitness landscape is extremely rugged.

*Why does bloat happen?*
- **Introns** (semantically inert subtrees) absorb destructive crossover events → a larger individual is more likely to produce a surviving offspring → size gets selected for even though it contributes nothing to fitness.
- Fixes: explicit parsimony penalty, depth/size caps, or double-tournament selection on size.

*Why not just use a NN?*
- Different product. A NN gives a black-box function; GP gives an equation you can read, differentiate by hand, and check against theory. GP wins exactly when the *form* of the answer is the deliverable.
```

&nbsp;

### ES
- **Name**: Evolution Strategies
- **What**: Real-valued evolution by Gaussian mutation, with the mutation distribution itself adapted.
- **Why**: GA's discrete genetic metaphor is the wrong prior for $\mathbb{R}^n$.
    - In continuous space what matters is the **step size & shape** of the search distribution, not the recombination of symbols.
    - Fixed $\sigma$ is wrong everywhere: too large near the optimum (never converges), too small far away (never arrives).
    - → let the distribution's params evolve alongside the solutions.
- **How**:
    1. Sample $\lambda$ offspring from $\mathcal{N}(\mathbf{m},\sigma^2C)$.
    2. Evaluate & **rank** (values are used only through their order).
    3. Recombine the best $\mu$ into a new mean.
    4. Adapt $\sigma$ (and possibly $C$) from the observed pattern of successes.
    5. Repeat.
    - **$(\mu,\lambda)$**: the next generation is drawn from offspring only → parents die → can leave a local optimum, and cannot stagnate.
    - **$(\mu+\lambda)$**: parents compete with offspring → elitist → monotone but prone to stagnation.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\mathbf{m}\in\mathbb{R}^n$: Distribution mean — the current best guess.
    - $\sigma>0$: Global step size.
    - $C\in\mathbb{R}^{n\times n}$: Covariance matrix, shape of the search distribution.
- Hyperparams:
    - $\lambda$: #offspring per generation.
    - $\mu$: #selected parents.
    - $w_i$: Recombination weights, $\sum_{i=1}^\mu w_i=1$.
- Misc:
    - $\mathbf{x}^{(i:\lambda)}$: $i$-th best of the $\lambda$ offspring.

Sampling:

$$
\mathbf{x}^{(i)}=\mathbf{m}+\sigma\mathbf{z}^{(i)},\qquad\mathbf{z}^{(i)}\sim\mathcal{N}(\mathbf{0},C)
$$

Weighted recombination:

$$
\mathbf{m}\leftarrow\sum_{i=1}^{\mu}w_i\mathbf{x}^{(i:\lambda)}
$$

1/5th success rule, the classical $(1+1)$-ES step control:

$$
\sigma\leftarrow\begin{cases}\sigma\cdot a&\text{success rate}>\frac{1}{5}\\\sigma/a&\text{success rate}<\frac{1}{5}\end{cases},\qquad a>1
$$
```

```{attention} Q&A
:class: dropdown
*Why is the target success rate $\frac{1}{5}$?*
- Too high a success rate means the steps are so small that half of them improve by luck → progress per evaluation is wasted → ⬆️$\sigma$.
- Too low means most samples are thrown away → ⬇️$\sigma$. $\frac{1}{5}$ is the progress-rate optimum derived on the sphere & corridor models.

*$(\mu,\lambda)$ vs $(\mu+\lambda)$?*
- Comma discards parents → tolerates noisy fitness (a lucky evaluation cannot survive forever) and can escape local optima. Requires $\lambda>\mu$, typically $\lambda\approx7\mu$ for the classical rule.
- Plus is elitist → faster on unimodal, noiseless problems; sticky on everything else.

*Why rank-based rather than value-based selection?*
- It makes the whole method **invariant to any strictly increasing transformation of $f$** → immune to outliers, to objective rescaling, and to whether you optimize $f$ or $\log f$.
```

&nbsp;

#### CMA-ES
- **Name**: Covariance Matrix Adaptation Evolution Strategy {cite:p}`hansen2001completely`
- **What**: ES that learns the full covariance of its search distribution from the path of successful steps.
- **Why**: Isotropic mutation is hopeless on ill-conditioned or rotated problems.
    - A spherical proposal in a narrow valley makes tiny steps along the valley and wasted steps across it — the same $\kappa$ problem gradient descent has.
    - Adapting $C$ aligns the proposal with the local level sets → it **learns an approximation of $H^{-1}$ from rankings alone**, with no derivatives.
    - → invariant to rotation, translation, and to any strictly increasing transform of $f$.
- **How**:
    1. Sample $\lambda$ offspring from $\mathcal{N}(\mathbf{m},\sigma^2C)$; rank them.
    2. Recombine the best $\mu$ into a new mean.
    3. **Rank-$\mu$ update**: pull $C$ toward the covariance of *this generation's* successful steps.
    4. **Rank-one update**: accumulate an **evolution path** $\mathbf{p}_c$ (exponentially smoothed mean shifts) and add $\mathbf{p}_c\mathbf{p}_c^T$ → captures correlation *across* generations, which a single generation cannot see.
    5. **Step-size control**: a second path $\mathbf{p}_\sigma$ is compared against its expected length under pure randomness → longer ⇒ ⬆️$\sigma$, shorter ⇒ ⬇️$\sigma$.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\mathbf{p}_c$: Evolution path for the rank-one covariance update.
    - $\mathbf{p}_\sigma$: Conjugate evolution path for step-size control.
- Hyperparams:
    - $c_1$: Rank-one learning rate.
    - $c_\mu$: Rank-$\mu$ learning rate.
    - $c_\sigma,d_\sigma$: Step-size adaptation rate & damping.
- Misc:
    - $\mathbf{y}^{(i:\lambda)}=\frac{\mathbf{x}^{(i:\lambda)}-\mathbf{m}_t}{\sigma_t}$: Normalized successful step.

Defaults, all derived from $n$:

$$
\lambda=4+\lfloor3\ln n\rfloor,\qquad\mu=\left\lfloor\frac{\lambda}{2}\right\rfloor,\qquad w_i\propto\ln\left(\mu+\frac{1}{2}\right)-\ln i
$$

Mean:

$$
\mathbf{m}_{t+1}=\sum_{i=1}^{\mu}w_i\mathbf{x}^{(i:\lambda)}
$$

Covariance — rank-one (across generations) + rank-$\mu$ (within a generation):

$$
C_{t+1}=(1-c_1-c_\mu)C_t+c_1\mathbf{p}_c\mathbf{p}_c^T+c_\mu\sum_{i=1}^{\mu}w_i\mathbf{y}^{(i:\lambda)}\mathbf{y}^{(i:\lambda)T}
$$

Step size — cumulative step-length adaptation:

$$
\sigma_{t+1}=\sigma_t\exp\left(\frac{c_\sigma}{d_\sigma}\left(\frac{||\mathbf{p}_\sigma||}{\mathbb{E}\left[||\mathcal{N}(\mathbf{0},I)||\right]}-1\right)\right)
$$
```

````{important} Code
:class: dropdown
```python
import torch

class CMAES:
    ## rank-mu update only: evolution paths & step-size control omitted for readability,
    ## so this shows covariance LEARNING but not CMA-ES's full step-size adaptation
    def __init__(self, f, m, sigma=0.5, c_mu=None):
        self.f, self.m, self.sigma = f, m.clone(), sigma
        n = m.numel()
        self.lam = 4 + int(3 * torch.log(torch.tensor(float(n))))   ## 4 + floor(3 ln n)
        self.mu = self.lam // 2
        w = torch.log(torch.tensor(self.mu + 0.5)) - torch.arange(1, self.mu + 1).log()
        self.w = w / w.sum()                                        ## log-decreasing weights
        self.C = torch.eye(n)
        self.c_mu = c_mu if c_mu is not None else 1.0 / n

    def step(self):
        A = torch.linalg.cholesky(self.C)
        z = torch.randn(self.lam, self.m.numel())
        X = self.m + self.sigma * z @ A.T                           ## sample N(m, sigma^2 C)
        order = self.f(X).argsort()[:self.mu]                       ## RANK only -- values unused
        Y = (X[order] - self.m) / self.sigma                        ## normalized successful steps
        self.m = self.m + self.sigma * (self.w[:, None] * Y).sum(0)
        ## pull C toward the covariance of the steps that just worked
        rank_mu = torch.einsum('i,ij,ik->jk', self.w, Y, Y)
        self.C = (1 - self.c_mu) * self.C + self.c_mu * rank_mu
        return self.m

## Example: an ill-conditioned rotated ellipse -- isotropic mutation would crawl
torch.manual_seed(0)
D = torch.tensor([1.0, 100.0])
f = lambda X: (D * X ** 2).sum(-1)
opt = CMAES(f, torch.tensor([3.0, 3.0]))
for _ in range(300):
    opt.step()
print(opt.m.abs().max() < 0.1)                                      ## True
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- Effectively **hyperparameter-free** — $\lambda,\mu,w,c_1,c_\mu,c_\sigma,d_\sigma$ all have well-tested defaults derived from $n$.
- Invariant to rotation, translation & any strictly increasing transform of $f$ (rankings only) → immune to outliers and to objective rescaling.
- Learns curvature → competitive with quasi-Newton on ill-conditioned problems, with ❌derivatives.
- The de facto standard for continuous black-box optimization at $10\lesssim n\lesssim100$.

*Cons?*
- $O(n^2)$ memory and an $O(n^3)$ eigendecomposition of $C$ (amortized over generations) → needs sep-CMA-ES or limited-memory variants above $n\sim1000$.
- Sample-inefficient vs [BO](#smbo) when one evaluation costs GPU-hours.
- Assumes continuous $\mathbb{R}^n$ → discrete/conditional spaces need encoding hacks.

*Why an evolution path instead of just this generation's covariance?*
- One generation cannot distinguish a consistent direction from a coincidence.
- Worse, $\mathbf{y}\mathbf{y}^T$ is **sign-blind**: $\mathbf{y}$ and $-\mathbf{y}$ give the same rank-one term.
- The path accumulates mean shifts across generations → consistent directions reinforce, random ones cancel → the sign information is recovered and $C$ becomes reliable with far fewer samples.

*Why compare $||\mathbf{p}_\sigma||$ against its expectation under randomness?*
- With a well-tuned $\sigma$, consecutive steps should look roughly uncorrelated, like a random walk.
- Longer than random ⇒ the search keeps heading the same way ⇒ steps are too short ⇒ ⬆️$\sigma$.
- Shorter than random ⇒ steps keep cancelling ⇒ overshooting ⇒ ⬇️$\sigma$.

*Relation to second-order methods?*
- $C$ converges toward $H^{-1}$ (up to scale) on a convex quadratic → CMA-ES is a **derivative-free, rank-based quasi-Newton**. Same goal as [BFGS](#bfgs); it just estimates curvature from ranked samples instead of gradient differences.
```

&nbsp;

#### NES
- **Name**: Natural Evolution Strategies {cite:p}`wierstra2014natural`
- **What**: Gradient ascent on the expected fitness of a parameterized search distribution, preconditioned by the Fisher information.
- **Why**: ES adapts its distribution by hand-designed heuristics.
    - Writing the goal as $J(\theta)=\mathbb{E}_{p_\theta}[f]$ makes it **differentiable in $\theta$ even when $f$ is not differentiable in $\mathbf{x}$** — the log-derivative trick moves the derivative onto the density.
    - Plain gradient ascent in $\theta$ depends on how the distribution happens to be parameterized → use the **natural** gradient, which follows the steepest direction in distribution space instead.
- **How**:
    1. Sample a population from $p_\theta$.
    2. Estimate $\nabla_\theta J$ with the score-function (REINFORCE) estimator.
    3. Precondition by $F^{-1}$ → natural gradient.
    4. Update $\theta$ (mean & covariance), using **fitness shaping** (rank-based utilities) instead of raw $f$.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\theta$: Params of the search distribution $p_\theta$ (e.g. $\mathbf{m},\sigma,C$).
- Misc:
    - $F$: Fisher information matrix of $p_\theta$.
    - $\boldsymbol{\epsilon}$: Standard normal noise.

Objective — smoothed, hence differentiable in $\theta$ regardless of $f$:

$$
J(\theta)=\mathbb{E}_{\mathbf{x}\sim p_\theta}\left[f(\mathbf{x})\right]
$$

Score-function gradient:

$$
\nabla_\theta J=\mathbb{E}_{\mathbf{x}\sim p_\theta}\left[f(\mathbf{x})\nabla_\theta\log p_\theta(\mathbf{x})\right]
$$

Natural gradient:

$$
\tilde{\nabla}_\theta J=F^{-1}\nabla_\theta J,\qquad F=\mathbb{E}_{p_\theta}\left[\nabla_\theta\log p_\theta\ \nabla_\theta\log p_\theta^T\right]
$$

Isotropic Gaussian with fixed $\sigma$ → the mean update is a smoothed finite-difference gradient:

$$
\nabla_\mathbf{m}J=\frac{1}{\sigma}\mathbb{E}_{\boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},I)}\left[f(\mathbf{m}+\sigma\boldsymbol{\epsilon})\,\boldsymbol{\epsilon}\right]
$$
```

```{attention} Q&A
:class: dropdown
*In what sense is this evolution?*
- Population = the sample from $p_\theta$; selection = the fitness weighting; variation = the Gaussian noise. It is ES with the heuristics replaced by an explicit stochastic-gradient derivation.

*Why fitness shaping?*
- Weighting by raw $f$ makes the estimator scale-dependent and outlier-dominated (one huge $f$ swamps the batch). Replacing $f$ by a rank-based utility restores [CMA-ES](#cma-es)'s invariance to monotone transformations.

*Pros?*
- Scales to millions of params with a diagonal/isotropic covariance.
- Trivially parallel: workers need only exchange a random **seed** and a scalar fitness, not gradients → communication is $O(1)$ per worker.
- ❌backprop → tolerates non-differentiable, long-horizon, sparse-signal objectives.

*Cons?*
- Score-function estimator variance grows with dimension → many samples per update.
- Isotropic covariance discards the curvature that CMA-ES's full $C$ captures.
- Needs a step size, unlike CMA-ES.

*Why is it not just finite differences?*
- It is a **smoothed** gradient: it estimates $\nabla\mathbb{E}[f]$, not $\nabla f$. That smoothing is what makes it work on discontinuous and deceptive objectives where the true gradient is useless or nonexistent.
```

&nbsp;

### DE
- **Name**: Differential Evolution {cite:p}`storn1997differential`
- **What**: Mutation vectors built from **scaled differences between population members**.
- **Why**: ES has to adapt a mutation distribution from scratch.
    - But the population **already encodes** the local scale & orientation of the landscape — the distribution of pairwise differences *is* an implicit covariance estimate.
    - → build the mutation out of the population itself → self-scaling & orientation-aware with zero adaptation machinery.
- **How** (DE/rand/1/bin):
    1. For each target $\mathbf{x}^{(i)}$, draw 3 distinct others.
    2. **Mutation**: $\mathbf{v}=\mathbf{x}^{(r_1)}+F(\mathbf{x}^{(r_2)}-\mathbf{x}^{(r_3)})$.
    3. **Crossover**: per coordinate, take $\mathbf{v}$'s value w.p. $CR$, else the target's; force at least one coordinate from $\mathbf{v}$.
    4. **Selection**: greedy 1-to-1 — the trial replaces its own target only if it is at least as good.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $F$: Differential weight, typically $0.5$-$0.9$.
    - $CR\in[0,1]$: Crossover rate, typically $0.9$.
    - $\mu$: Population size.
- Misc:
    - $r_1,r_2,r_3$: Distinct random indices, all $\neq i$.
    - $j_\text{rand}$: A coordinate forced to come from the mutant.

Mutation:

$$
\mathbf{v}^{(i)}=\mathbf{x}^{(r_1)}+F\left(\mathbf{x}^{(r_2)}-\mathbf{x}^{(r_3)}\right)
$$

Binomial crossover:

$$
u^{(i)}_j=\begin{cases}v^{(i)}_j&\text{rand}_j\leq CR\ \text{ or }\ j=j_\text{rand}\\x^{(i)}_j&\text{otherwise}\end{cases}
$$

Greedy selection (minimization):

$$
\mathbf{x}^{(i)}\leftarrow\begin{cases}\mathbf{u}^{(i)}&f(\mathbf{u}^{(i)})\leq f(\mathbf{x}^{(i)})\\\mathbf{x}^{(i)}&\text{otherwise}\end{cases}
$$
```

````{important} Code
:class: dropdown
```python
import torch

class DE:
    def __init__(self, f, n_dim, mu=40, F=0.8, CR=0.9):
        self.f, self.n, self.mu, self.F, self.CR = f, n_dim, mu, F, CR

    def run(self, n_gen=300, scale=5.0):
        X = scale * (2 * torch.rand(self.mu, self.n) - 1)
        fx = self.f(X)
        for _ in range(n_gen):
            idx = torch.stack([torch.randperm(self.mu)[:3] for _ in range(self.mu)])
            a, b, c = X[idx[:, 0]], X[idx[:, 1]], X[idx[:, 2]]
            V = a + self.F * (b - c)                       ## step size comes FROM the population
            mask = torch.rand(self.mu, self.n) < self.CR
            mask[torch.arange(self.mu), torch.randint(self.n, (self.mu,))] = True  ## >=1 gene
            U = torch.where(mask, V, X)
            fu = self.f(U)
            better = fu <= fx                              ## greedy 1-to-1 -> per-slot monotone
            X, fx = torch.where(better[:, None], U, X), torch.where(better, fu, fx)
        return X[fx.argmin()], fx.min()

## Example: Rastrigin, global minimum at 0
f = lambda X: (X ** 2 - 10 * torch.cos(2 * torch.pi * X) + 10).sum(-1)
torch.manual_seed(0)
best, fbest = DE(f, n_dim=2).run()
print(best.round(decimals=2), fbest.round(decimals=3))     ## tensor([0., 0.]) tensor(0.)
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- 3 hyperparams only ($\mu,F,CR$), and it works across wide ranges of them.
- **Self-scaling**: the mutation magnitude shrinks automatically as the population converges — no step-size schedule at all.
- Greedy 1-to-1 selection → each slot is monotone → ❌separate best-so-far bookkeeping.
- Consistently strong on multi-modal continuous benchmarks.

*Cons?*
- Continuous $\mathbb{R}^n$ only; integer/categorical dims need encoding.
- **Stagnation**: if the population collapses, the differences vanish and the search halts, with no mechanism to reopen it (a fixed mutation $\sigma$ would at least keep moving).
- Binomial crossover is coordinate-wise → ⬇️rotation invariance. DE/current-to-best & exponential crossover partly restore it.

*Why does the difference vector work?*
- $\mathbf{x}^{(r_2)}-\mathbf{x}^{(r_3)}$ is a draw from the population's own difference distribution → its covariance is twice the population covariance.
- → the proposal automatically matches the current spread AND orientation of the search — precisely what [CMA-ES](#cma-es) spends an entire adaptation mechanism to learn explicitly.

*How to choose $F$ & $CR$?*
- ⬆️$F$ → ⬆️exploration, ⬇️convergence speed.
- ⬆️$CR$ → more coordinates change at once → better on **non-separable** problems, worse on separable ones.
- Self-adaptive variants (jDE, SHADE, L-SHADE) tune both online and are the modern default.
```

&nbsp;

### PSO
- **Name**: Particle Swarm Optimization {cite:p}`kennedy1995particle`
- **What**: Particles moving with velocity pulled toward their own best & the swarm's best.
- **Why**: Selection throws away the trajectory that produced a good point.
    - GA/DE delete the loser entirely; the *path* it was on is lost with it.
    - → keep every individual alive and let it **remember** and be **attracted**, rather than be replaced.
    - Velocity adds momentum → the particle coasts through shallow local optima instead of settling in them.
- **How**:
    1. Init positions & velocities at random.
    2. Each particle stores its personal best $\mathbf{p}^{(i)}$; the swarm stores the global best $\mathbf{g}$.
    3. Velocity = inertia + a random pull toward $\mathbf{p}^{(i)}$ + a random pull toward $\mathbf{g}$.
    4. Move, update bests, repeat.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\mathbf{p}^{(i)}$: Personal best of particle $i$.
    - $\mathbf{g}$: Global (or neighbourhood) best.
- Hyperparams:
    - $w$: Inertia weight.
    - $c_1$: Cognitive coefficient (pull to own best).
    - $c_2$: Social coefficient (pull to swarm best).
- Misc:
    - $\mathbf{r}_1,\mathbf{r}_2\sim\text{Uniform}(0,1)^n$: Fresh elementwise random weights each step.

Velocity & position:

$$
\mathbf{v}^{(i)}\leftarrow w\mathbf{v}^{(i)}+c_1\mathbf{r}_1\odot\left(\mathbf{p}^{(i)}-\mathbf{x}^{(i)}\right)+c_2\mathbf{r}_2\odot\left(\mathbf{g}-\mathbf{x}^{(i)}\right)
$$

$$
\mathbf{x}^{(i)}\leftarrow\mathbf{x}^{(i)}+\mathbf{v}^{(i)}
$$

Clerc's constriction values, which give convergence without velocity clamping:

$$
w=0.7298,\qquad c_1=c_2=1.49618
$$
```

```{attention} Q&A
:class: dropdown
*Is it actually evolution?*
- ❌Strictly. There is no selection, no inheritance, no death — nothing is ever replaced.
- It is **swarm intelligence**: same population-based black-box family, but individuals persist and are only *attracted*.

*Pros?*
- Extremely simple, few params, fast early convergence.
- Fully parallel.
- Personal bests act as a distributed memory of the landscape.

*Cons?*
- **Premature convergence**: with a fully-connected (global-best) topology every particle is pulled toward one point → the swarm collapses. Ring/local topologies slow information spread and preserve diversity.
- ⬇️Rotation invariance ← the elementwise $\mathbf{r}_1,\mathbf{r}_2$ make the dynamics axis-aligned.
- Weaker theory than [CMA-ES](#cma-es)/[DE](#de), and typically weaker on hard multimodal benchmarks.

*Why is inertia essential?*
- w/o it ($w=0$) a particle jumps straight to a random convex combination of its two attractors → the swarm collapses in a few steps.
- Inertia makes the trajectory **overshoot and oscillate** around the attractors, and that oscillation is the entire exploration mechanism.
```

&nbsp;

### NSGA-II
- **Name**: Non-dominated Sorting Genetic Algorithm II {cite:p}`deb2002fast`
- **What**: Elitist multi-objective GA ranking by Pareto dominance, tie-broken by crowding distance.
- **Why**: Scalarizing objectives into one destroys the trade-off surface.
    - A population is the natural representation of a **front** — it can hold many mutually non-dominated solutions at once.
    - But "better" is now a **partial** order → you need a way to turn dominance into a total ranking that also **spreads** solutions along the front rather than clumping them.
- **How**:
    1. Merge parents & offspring (elitism).
    2. **Fast non-dominated sort**: rank 1 = non-dominated; rank 2 = non-dominated after deleting rank 1; and so on.
    3. Fill the next generation front by front.
    4. The front that overflows is tie-broken by **crowding distance** — larger = more isolated = preferred.
    5. Vary (tournament on (rank, crowding) + crossover + mutation), repeat.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $f_k$: Objective $k$, minimized.
- Misc:
    - $K$: #objectives.
    - $d_i$: Crowding distance of solution $i$.
    - $\mathbf{x}^{(i)}$: $i$-th solution in the front, sorted by $f_k$.

Pareto dominance:

$$
\mathbf{x}\prec\mathbf{x}'\ \Leftrightarrow\ \left(\forall k:f_k(\mathbf{x})\leq f_k(\mathbf{x}')\right)\wedge\left(\exists k:f_k(\mathbf{x})<f_k(\mathbf{x}')\right)
$$

Crowding distance, accumulated over objectives on each sorted front:

$$
d_i\ \mathrel{+}=\ \frac{f_k(\mathbf{x}^{(i+1)})-f_k(\mathbf{x}^{(i-1)})}{f_k^\max-f_k^\min}
$$
- Boundary solutions get $d=\infty$ → the extremes of the front are never discarded.

Crowded-comparison operator — $i$ beats $j$ iff:

$$
\text{rank}_i<\text{rank}_j\quad\text{or}\quad\left(\text{rank}_i=\text{rank}_j\ \wedge\ d_i>d_j\right)
$$
```

```{attention} Q&A
:class: dropdown
*Why crowding distance rather than fitness sharing?*
- Fitness sharing needs a niche-radius hyperparam that must be guessed per problem.
- Crowding distance is **parameter-free** — it is read directly off the front's own spacing.

*Cons?*
- Dominance becomes nearly useless past ~4 objectives: almost everything is non-dominated → rank 1 fills the whole population → selection pressure collapses to crowding distance alone. → many-objective methods (NSGA-III, MOEA/D) switch to reference directions or decomposition.
- The non-dominated sort is the bottleneck at large populations.

*NSGA-II vs [Multi-Objective BO](#multi-objective-bo)?*
- Same goal, opposite budget regime: NSGA-II wants $10^4$-$10^5$ cheap evaluations; multi-objective BO targets $10^2$ expensive ones.

*Why merge parents & offspring before sorting?*
- Elitism at the front level: without it, a rank-1 solution can be lost to a whole generation of worse offspring, and the front can move **backwards**.
```

&nbsp;

### QD
- **Name**: Quality-Diversity
- **What**: Search returning an **archive** of diverse high-performing solutions rather than one optimum.
- **Why**: A single optimum is often not the goal, and pursuing it directly is often not how you reach it.
    - **Deception**: on hard problems the objective's gradient points *away* from the solution — the necessary stepping stones score worse than the dead end → objective-driven search reliably converges to the deception.
    - A collection of diverse working solutions is directly more useful: robustness, fast adaptation, downstream selection.
    - → promote **behavioral novelty** to a first-class search pressure instead of a diversity-preservation patch.
- **How**:
    1. Define a **behavior descriptor** — a low-dim characterization of *what a solution does*, deliberately distinct from *how well* it does it.
    2. Search for behavioral novelty, keeping the best performer per behavior niche.
    3. Return the whole archive.

&nbsp;

#### Novelty Search
- **What**: Selection on behavioral novelty **alone**, with the objective discarded entirely {cite:p}`lehman2011abandoning`
- **Why**: On a deceptive problem the objective is not a weak signal — it is an actively misleading one.
    - Deleting it deletes the deception; what remains is a pressure to keep doing something not done before.
    - That pressure sweeps the reachable behavior space, and the solution is typically found *along the way*.
- **How**:
    1. Novelty of an individual = mean distance to its $k$ nearest behaviors in (archive ∪ current population).
    2. Select on novelty.
    3. Add sufficiently novel individuals to a permanent archive → prevents cycling back through old behaviors.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $b(\mathbf{x})$: Behavior descriptor of $\mathbf{x}$.
    - $\mathbf{x}^{(j)}$: $j$-th nearest neighbour of $\mathbf{x}$ in behavior space.
    - $k$: #neighbours.

Novelty score:

$$
\rho(\mathbf{x})=\frac{1}{k}\sum_{j=1}^{k}\text{dist}\left(b(\mathbf{x}),\ b(\mathbf{x}^{(j)})\right)
$$
```

```{attention} Q&A
:class: dropdown
*Why does abandoning the objective ever beat optimizing it?*
- Objective-driven search rejects any step that scores worse — but on a deceptive problem the required intermediate behaviors *do* score worse.
- Novelty accepts them because they are new, so the stepping stones survive long enough to be built on.

*When does it fail?*
- Large or unbounded behavior spaces where most novelty is worthless (random noise is maximally novel) → it wanders forever.
- → in practice blend it with quality: novelty + fitness, or [MAP-Elites](#map-elites).

*What makes a good behavior descriptor?*
- Low-dimensional, bounded, and capturing *what the solution does* rather than *how well*. It is hand-designed, and it is the single biggest lever on whether QD works.
```

&nbsp;

#### MAP-Elites
- **Name**: Multi-dimensional Archive of Phenotypic Elites {cite:p}`mouret2015illuminating`
- **What**: Behavior space discretized into a grid, with the best solution retained per cell.
- **Why**: Pure novelty search abandons quality entirely.
    - Keeping only the **elite** of each behavior niche restores a quality pressure while retaining the coverage.
    - The filled archive **illuminates** the search space: it shows how performance varies *across* behaviors, not merely where the peak is.
- **How**:
    1. Discretize behavior space into cells.
    2. Init: random solutions → evaluate → each occupies its cell if it beats the incumbent elite there.
    3. Loop: pick a **random** elite, mutate it, evaluate, and place it in its (possibly different) cell if it beats that cell's elite.
    4. Return the archive.

````{important} Code
:class: dropdown
```python
import torch

class MAPElites:
    def __init__(self, f, behavior, n_dim, n_cells=20, sigma=0.1):
        self.f, self.b, self.n = f, behavior, n_dim
        self.n_cells, self.sigma = n_cells, sigma
        self.elites = {}                       ## cell index -> (genome, fitness)

    def _cell(self, x):
        ## discretize the behavior descriptor (assumed in [0,1]) into a grid index
        return int((self.b(x).clamp(0, 1 - 1e-9) * self.n_cells).item())

    def _add(self, x):
        c, fx = self._cell(x), self.f(x)
        if c not in self.elites or fx > self.elites[c][1]:
            self.elites[c] = (x, fx)           ## per-cell elitism: quality WITHIN a niche

    def run(self, n_init=100, n_iter=5000):
        for _ in range(n_init):
            self._add(torch.randn(self.n))
        for _ in range(n_iter):
            ## uniform over occupied cells -> implicit diversity pressure, no novelty term needed
            parent = self.elites[list(self.elites)[torch.randint(len(self.elites), (1,))]][0]
            self._add(parent + self.sigma * torch.randn(self.n))
        return self.elites

## Example: quality = closeness to the origin; behavior = first coordinate
f = lambda x: -(x ** 2).sum()
behavior = lambda x: (x[0] + 3) / 6
arch = MAPElites(f, behavior, n_dim=2).run()
print(len(arch))            ## ~20 -> the whole behavior range is filled, each cell with its best
```
````

```{attention} Q&A
:class: dropdown
*Why does picking a uniformly random elite work as a search strategy?*
- The elites are already spread across behavior space → uniform sampling among them **is** the diversity pressure, with no explicit novelty term.
- Mutating a distant niche's elite is exactly the stepping-stone move that objective-driven search cannot produce.

*Cons?*
- The behavior descriptor must be hand-designed & low-dim; a bad one makes the archive meaningless. Fixes: learned descriptors (AURORA), or CVT-MAP-Elites for high-dim behavior spaces.
- Archive size grows exponentially in #behavior dimensions.
- Huge evaluation budgets ($10^5$-$10^7$).

*Where does it pay off?*
- Damage recovery in robotics: pre-compute an archive of behaviorally diverse gaits offline, then treat it as a prior so a damaged robot can find a still-working gait in a handful of trials.
- Any setting where you need a *portfolio* of solutions rather than one: procedural content, design exploration, adversarial test-case generation.

*Why does this matter beyond optimization?*
- QD is the operational form of "search with no fixed objective". Processes that keep generating new stepping stones instead of converging are the closest existing handle on **open-endedness**, which is the property natural evolution has and ordinary optimization does not.
```

&nbsp;

```{dropdown} Table: Population-Based Methods
| Method | Variation | Selection | Adapts | Best at |
|:--|:--|:--|:--|:--|
| [GA](#ga) | Crossover + mutation | Tournament / roulette | ❌ | Discrete, mixed, structured |
| [Genetic Programming](#genetic-programming) | Subtree swap / regrow | Tournament + parsimony | ❌ | Programs, formulas |
| [ES](#es) | Gaussian mutation | Rank, top $\mu$ of $\lambda$ | $\sigma$ | Continuous, unimodal |
| [CMA-ES](#cma-es) | $\mathcal{N}(\mathbf{m},\sigma^2C)$ | Rank, weighted | $\sigma$ & full $C$ | Continuous, ill-conditioned, $n\lesssim100$ |
| [NES](#nes) | $\mathcal{N}(\mathbf{m},\sigma^2I)$ | Rank-shaped weights | $\theta$ by natural gradient | Very high $n$, massively parallel |
| [DE](#de) | Scaled member differences | Greedy 1-to-1 | Implicit, via the population | Continuous, multi-modal |
| [PSO](#pso) | Velocity + attraction | ❌ (nothing dies) | ❌ | Cheap, quick-and-dirty |
| [NSGA-II](#nsga-ii) | Crossover + mutation | Dominance rank + crowding | ❌ | Multi-objective fronts |
| [MAP-Elites](#map-elites) | Mutation of a random elite | Per-cell elitism | ❌ | Archives, deception, open-endedness |
```

&nbsp;
