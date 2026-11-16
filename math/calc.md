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
# Calculus
Handbook of single- & multivariable differentiation & integration.

Notations:
- $f,g$: Real functions; smooth/integrable as each row requires.
- $u,v$: Functions of $x$.
- $a,b,c$: Real constants; $[a,b]$: interval.
- $\alpha$: Real exponent.
- $n,k$: Nonnegative integers (order, exponent, index).
- $d$: Dimension.
- $\ln$: Natural log.
- $\delta_{ij}$: Kronecker delta.
- Integration constant $C$ omitted in antiderivative tables.

$x$ = real variable & $n$ = integer order, overriding the global input & #features meanings.

&nbsp;

## Limits

### Standard Limits

| Limit | Value | Condition |
|:--|:--|:--|
| $\lim_{x\to0}\frac{\sin x}{x}$ | $1$ | |
| $\lim_{x\to0}\frac{\tan x}{x}$ | $1$ | |
| $\lim_{x\to0}\frac{1-\cos x}{x^2}$ | $\frac12$ | |
| $\lim_{x\to0}\frac{e^x-1}{x}$ | $1$ | |
| $\lim_{x\to0}\frac{a^x-1}{x}$ | $\ln a$ | $a>0$ |
| $\lim_{x\to0}\frac{\ln(1+x)}{x}$ | $1$ | |
| $\lim_{x\to0}\frac{(1+x)^\alpha-1}{x}$ | $\alpha$ | |
| $\lim_{x\to0}(1+x)^{1/x}$ | $e$ | |
| $\lim_{x\to\infty}\left(1+\frac{a}{x}\right)^x$ | $e^a$ | |
| $\lim_{x\to\infty}\frac{x^\alpha}{e^{cx}}$ | $0$ | $c>0$ |
| $\lim_{x\to\infty}\frac{(\ln x)^\alpha}{x^c}$ | $0$ | $c>0$ |
| $\lim_{x\to0^+}x^c\ln x$ | $0$ | $c>0$ |
| $\lim_{x\to0^+}x^x$ | $1$ | |
| $\lim_{n\to\infty}n^{1/n}$ | $1$ | |
| $\lim_{n\to\infty}\left(\sum_{k=1}^n\frac1k-\ln n\right)$ | $\gamma\approx0.5772$ | Euler–Mascheroni constant |

Growth order as $n\to\infty$ ($c>0$, $a>1$): $\ln n\ll n^c\ll a^n\ll n!\ll n^n$.

&nbsp;

### Asymptotic Notation
As $x\to x_0$:

| Notation | Definition |
|:--|:--|
| $f=O(g)$ | $\lvert f\rvert\le M\lvert g\rvert$ near $x_0$ for some constant $M$ |
| $f=o(g)$ | $f/g\to0$ |
| $f\sim g$ | $f/g\to1$ |
| $f=\Theta(g)$ | $f=O(g)$ & $g=O(f)$ |

&nbsp;

### L'Hôpital's Rule

| Form | Rewrite |
|:--|:--|
| $\frac00$, $\frac\infty\infty$ | $\lim\frac fg=\lim\frac{f'}{g'}$ |
| $0\cdot\infty$ | $fg=\frac{f}{1/g}$ |
| $\infty-\infty$ | Common denominator / factor out the dominant term |
| $1^\infty$, $0^0$, $\infty^0$ | $f^g=e^{g\ln f}$ → evaluate $\lim g\ln f$ |

- Conditions: $f,g$ differentiable near $x_0$ (except possibly at $x_0$); $g'\ne0$ near $x_0$; $\lim\frac{f'}{g'}$ exists in $\mathbb{R}\cup\{\pm\infty\}$.
- $\frac\infty\infty$ case needs only $\lvert g\rvert\to\infty$.
- ❌ Converse: $\lim\frac fg$ can exist while $\lim\frac{f'}{g'}$ does not, e.g., $\frac{x+\sin x}{x}$ as $x\to\infty$.

&nbsp;

## Differentiation

### Definition

$$
f'(x)=\lim_{h\to0}\frac{f(x+h)-f(x)}{h}
\quad\Leftrightarrow\quad
f(x+h)=f(x)+f'(x)h+o(h)
$$

- Differentiable → continuous. ❌ Converse: $\lvert x\rvert$ at $0$.

&nbsp;

### Rules

| Rule | Formula |
|:--|:--|
| Linearity | $(af+bg)'=af'+bg'$ |
| Product | $(uv)'=u'v+uv'$ |
| Quotient | $\left(\frac uv\right)'=\frac{u'v-uv'}{v^2}$ |
| Chain | $(f\circ g)'(x)=f'(g(x))\,g'(x)$ |
| Chain, 2nd order | $(f\circ g)''(x)=f''(g(x))\,g'(x)^2+f'(g(x))\,g''(x)$ |
| Inverse | $(f^{-1})'(y)=\frac1{f'(x)}$, $y=f(x)$, $f'(x)\ne0$ |
| Log derivative | $(\ln\lvert f\rvert)'=\frac{f'}{f}$ |
| General power | $(u^v)'=u^v\left(v'\ln u+\frac{vu'}{u}\right)$, $u>0$ |
| Leibniz | $(uv)^{(n)}=\sum_{k=0}^n\binom nk u^{(k)}v^{(n-k)}$ |
| Implicit | $F(x,y)=0$ → $y'=-\frac{F_x}{F_y}$, $F_y\ne0$ |
| Parametric | $\frac{dy}{dx}=\frac{\dot y}{\dot x}$; $\frac{d^2y}{dx^2}=\frac{\dot x\ddot y-\dot y\ddot x}{\dot x^3}$ |

- $F_x$: $\partial F/\partial x$.
- $\dot{\ }$: $d/dt$.

&nbsp;

### Elementary Derivatives

| $f(x)$ | $f'(x)$ | Domain |
|:--|:--|:--|
| $x^\alpha$ | $\alpha x^{\alpha-1}$ | |
| $e^x$ | $e^x$ | |
| $a^x$ | $a^x\ln a$ | $a>0$ |
| $\ln\lvert x\rvert$ | $\frac1x$ | $x\ne0$ |
| $\log_a\lvert x\rvert$ | $\frac1{x\ln a}$ | $x\ne0$; $a>0$, $a\ne1$ |
| $\sin x$ | $\cos x$ | |
| $\cos x$ | $-\sin x$ | |
| $\tan x$ | $\sec^2x=1+\tan^2x$ | |
| $\cot x$ | $-\csc^2x$ | |
| $\sec x$ | $\sec x\tan x$ | |
| $\csc x$ | $-\csc x\cot x$ | |
| $\arcsin x$ | $\frac1{\sqrt{1-x^2}}$ | $\lvert x\rvert<1$ |
| $\arccos x$ | $-\frac1{\sqrt{1-x^2}}$ | $\lvert x\rvert<1$ |
| $\arctan x$ | $\frac1{1+x^2}$ | |
| $\operatorname{arccot}x$ | $-\frac1{1+x^2}$ | |
| $\sinh x$ | $\cosh x$ | |
| $\cosh x$ | $\sinh x$ | |
| $\tanh x$ | $1-\tanh^2x=\operatorname{sech}^2x$ | |
| $\operatorname{arsinh}x$ | $\frac1{\sqrt{x^2+1}}$ | |
| $\operatorname{arcosh}x$ | $\frac1{\sqrt{x^2-1}}$ | $x>1$ |
| $\operatorname{artanh}x$ | $\frac1{1-x^2}$ | $\lvert x\rvert<1$ |
| $\lvert x\rvert$ | $\operatorname{sgn}x$ | $x\ne0$ |
| $x^x$ | $x^x(1+\ln x)$ | $x>0$ |

&nbsp;

### Higher-Order Derivatives

| $f(x)$ | $f^{(n)}(x)$ |
|:--|:--|
| $x^\alpha$ | $\alpha(\alpha-1)\cdots(\alpha-n+1)\,x^{\alpha-n}$ |
| $e^{cx}$ | $c^ne^{cx}$ |
| $a^x$ | $a^x(\ln a)^n$ |
| $\ln x$ | $(-1)^{n-1}\frac{(n-1)!}{x^n}$, $n\ge1$ |
| $\frac1{x+c}$ | $(-1)^n\frac{n!}{(x+c)^{n+1}}$ |
| $\sin x$ | $\sin\left(x+\frac{n\pi}2\right)$ |
| $\cos x$ | $\cos\left(x+\frac{n\pi}2\right)$ |

&nbsp;

### Mean Value Theorems
$f,g$ continuous on $[a,b]$ & differentiable on $(a,b)$ → $\exists c\in(a,b)$:

| Theorem | Statement |
|:--|:--|
| Rolle | $f(a)=f(b)$ → $f'(c)=0$ |
| Lagrange | $f'(c)=\frac{f(b)-f(a)}{b-a}$ |
| Cauchy | $\big(f(b)-f(a)\big)g'(c)=\big(g(b)-g(a)\big)f'(c)$ |
| Lipschitz bound (corollary) | $\lvert f(b)-f(a)\rvert\le\sup_{(a,b)}\lvert f'\rvert\,(b-a)$ |

- Lagrange: some tangent is parallel to the chord.
- Cauchy: for the parametric curve $t\mapsto(g(t),f(t))$, velocity at some $c$ is linearly dependent on the endpoint chord → parallel tangent if both are nonzero; Lagrange = case $g(t)=t$. Proves L'Hôpital.

&nbsp;

### Extrema & Convexity

| Test | Statement |
|:--|:--|
| Fermat | Interior local extremum at $c$ & $f$ differentiable at $c$ → $f'(c)=0$ |
| 1st derivative | $f'$ sign $+\to-$ at $c$ → local max; $-\to+$ → local min |
| 2nd derivative | $f'(c)=0$: $f''(c)>0$ → min; $f''(c)<0$ → max; $f''(c)=0$ → inconclusive |
| Higher order | $f'(c)=\dots=f^{(k-1)}(c)=0\ne f^{(k)}(c)$: $k$ even → extremum (min if $f^{(k)}(c)>0$); $k$ odd → inflection, no extremum |
| Convex, $C^1$ | $f$ convex $\Leftrightarrow f(y)\ge f(x)+f'(x)(y-x)$ |
| Convex, $C^2$ | $f$ convex on an interval $\Leftrightarrow f''\ge0$ there |
| Inflection | Concavity changes; necessary: $f''(c)=0$ or undefined |
| Jensen | $f$ convex → $f(\mathbb E[X])\le\mathbb E[f(X)]$ |

- Multivariate convexity: [Convexity](../ml/optim.md#convexity).

&nbsp;

### Taylor Series

$$
f(x)=\sum_{k=0}^{n}\frac{f^{(k)}(a)}{k!}(x-a)^k+R_n(x)
$$

| Remainder | $R_n(x)$ | Condition |
|:--|:--|:--|
| Peano | $o\big((x-a)^n\big)$ | $f^{(n)}(a)$ exists, $n\ge1$ |
| Lagrange | $\frac{f^{(n+1)}(\xi)}{(n+1)!}(x-a)^{n+1}$, $\xi$ between $a$ & $x$ | $f\in C^{n+1}$ |
| Integral | $\frac1{n!}\int_a^xf^{(n+1)}(t)(x-t)^n\,dt$ | $f\in C^{n+1}$ |

- Smooth ⇏ analytic: $f(x)=e^{-1/x^2}$ ($f(0)=0$) has all derivatives $0$ at $0$ → Maclaurin series $\equiv0\ne f$.
- Multivariate 2nd-order expansion: [Hessian](linalg.md#hessian).

Power series $\sum_{n\ge0}c_n(x-a)^n$:

| Item | Statement |
|:--|:--|
| Radius (Cauchy–Hadamard) | $R=\frac1{\limsup_{n\to\infty}\lvert c_n\rvert^{1/n}}$; converges absolutely for $\lvert x-a\rvert<R$, diverges for $\lvert x-a\rvert>R$ |
| Radius (ratio) | $R=\lim_{n\to\infty}\left\lvert\frac{c_n}{c_{n+1}}\right\rvert$ if the limit exists |
| Termwise derivative | $\frac{d}{dx}\sum c_n(x-a)^n=\sum_{n\ge1}nc_n(x-a)^{n-1}$, same $R$ |
| Termwise integral | $\int_a^x\sum c_n(t-a)^n\,dt=\sum_{n\ge0}\frac{c_n}{n+1}(x-a)^{n+1}$, same $R$ |

- Cauchy–Hadamard ← root test: $\lvert c_n(x-a)^n\rvert^{1/n}\to\frac{\lvert x-a\rvert}R$ (as $\limsup$) → $<1$ converges, $>1$ diverges.
- Endpoints $\lvert x-a\rvert=R$: check separately; termwise operations can gain or lose endpoint convergence.

Maclaurin series ($a=0$):

| $f(x)$ | Series | Converges to $f$ for |
|:--|:--|:--|
| $e^x$ | $\sum_{n\ge0}\frac{x^n}{n!}=1+x+\frac{x^2}2+\frac{x^3}6+\cdots$ | $\mathbb R$ |
| $\sin x$ | $\sum_{n\ge0}\frac{(-1)^nx^{2n+1}}{(2n+1)!}=x-\frac{x^3}6+\frac{x^5}{120}-\cdots$ | $\mathbb R$ |
| $\cos x$ | $\sum_{n\ge0}\frac{(-1)^nx^{2n}}{(2n)!}=1-\frac{x^2}2+\frac{x^4}{24}-\cdots$ | $\mathbb R$ |
| $\tan x$ | $x+\frac{x^3}3+\frac{2x^5}{15}+\cdots$ | $\lvert x\rvert<\frac\pi2$ |
| $\sinh x$ | $\sum_{n\ge0}\frac{x^{2n+1}}{(2n+1)!}$ | $\mathbb R$ |
| $\cosh x$ | $\sum_{n\ge0}\frac{x^{2n}}{(2n)!}$ | $\mathbb R$ |
| $\tanh x$ | $x-\frac{x^3}3+\frac{2x^5}{15}-\cdots$ | $\lvert x\rvert<\frac\pi2$ |
| $\frac1{1-x}$ | $\sum_{n\ge0}x^n$ | $\lvert x\rvert<1$ |
| $\ln(1+x)$ | $\sum_{n\ge1}\frac{(-1)^{n+1}x^n}{n}=x-\frac{x^2}2+\frac{x^3}3-\cdots$ | $-1<x\le1$ |
| $\ln\frac{1+x}{1-x}$ | $2\sum_{n\ge0}\frac{x^{2n+1}}{2n+1}$ | $\lvert x\rvert<1$ |
| $(1+x)^\alpha$ | $\sum_{n\ge0}\binom\alpha nx^n$, $\binom\alpha n=\frac{\alpha(\alpha-1)\cdots(\alpha-n+1)}{n!}$ | $\lvert x\rvert<1$; endpoints depend on $\alpha$ |
| $\sqrt{1+x}$ | $1+\frac x2-\frac{x^2}8+\frac{x^3}{16}-\cdots$ | $\lvert x\rvert\le1$ |
| $\arctan x$ | $\sum_{n\ge0}\frac{(-1)^nx^{2n+1}}{2n+1}$ | $\lvert x\rvert\le1$ |
| $\arcsin x$ | $\sum_{n\ge0}\frac{(2n)!}{4^n(n!)^2(2n+1)}x^{2n+1}=x+\frac{x^3}6+\frac{3x^5}{40}+\cdots$ | $\lvert x\rvert\le1$ |

&nbsp;

### Finite Differences

| Scheme | Formula | Error (formula $-$ exact) |
|:--|:--|:--|
| Forward | $\frac{f(x+h)-f(x)}h$ | $\frac h2f''(\xi)=O(h)$ |
| Central | $\frac{f(x+h)-f(x-h)}{2h}$ | $\frac{h^2}6f'''(\xi)=O(h^2)$ |
| 2nd-order central | $\frac{f(x+h)-2f(x)+f(x-h)}{h^2}$ | $\frac{h^2}{12}f^{(4)}(\xi)=O(h^2)$ |
| Complex step | $\frac{\operatorname{Im}f(x+ih)}h$ | $O(h^2)$ |

- Roundoff $\varepsilon$ → error $\approx\frac{A\varepsilon}h+Bh^p$ → optimal step $h=\Theta(\varepsilon^{1/(p+1)})$: $\Theta(\sqrt\varepsilon)$ (forward, $p=1$), $\Theta(\varepsilon^{1/3})$ (central, $p=2$).
- Complex step: ❌ subtraction → ❌ cancellation → $h$ can be tiny (e.g., $10^{-20}$). Needs $f$ real-analytic & complex-safe code.

&nbsp;

## Multivariable Differentiation

### Partial & Directional Derivatives
For $f:\mathbb R^d\to\mathbb R$:

| Object | Formula |
|:--|:--|
| Partial | $\partial_jf(\mathbf x)=\lim_{h\to0}\frac{f(\mathbf x+h\mathbf e_j)-f(\mathbf x)}h$ |
| Gradient | $\nabla f=(\partial_1f,\dots,\partial_df)^T$ |
| Total differential | $df=\sum_j\partial_jf\,dx_j=\nabla f^Td\mathbf x$ |
| Directional | $D_{\mathbf v}f=\lim_{h\to0}\frac{f(\mathbf x+h\mathbf v)-f(\mathbf x)}h=\nabla f^T\mathbf v$ ($f$ differentiable) |
| Steepest ascent | $\arg\max_{\lVert\mathbf v\rVert_2=1}D_{\mathbf v}f=\frac{\nabla f}{\lVert\nabla f\rVert_2}$; max rate $=\lVert\nabla f\rVert_2$ ($\nabla f\ne\mathbf 0$) |
| Level sets | $\nabla f\perp\{\mathbf x:f(\mathbf x)=c\}$ |
| Schwarz | $f\in C^2$ → $\partial_i\partial_jf=\partial_j\partial_if$ |

- $\mathbf e_j$: $j$-th standard basis vector.
- Continuous partials → differentiable → continuous & all directional derivatives exist.
- ❌ Converses ($f(\mathbf 0)=0$ in each case):
    - Partials exist ⇏ continuous: $\frac{xy}{x^2+y^2}$.
    - All directional derivatives exist ⇏ continuous: $\frac{x^2y}{x^4+y^2}$ ($f=\frac12$ on $y=x^2$).
    - Differentiable ⇏ continuous partials: $r^2\sin\frac1r$, $r=\lVert\mathbf x\rVert_2$.
- Vector/matrix forms: [Differential & Gradient](linalg.md#differential-gradient), [Jacobian](linalg.md#jacobian), [Hessian](linalg.md#hessian).

&nbsp;

### Multivariable Chain Rule

| Case | Formula |
|:--|:--|
| $z=f(x(t),y(t))$ | $\frac{dz}{dt}=f_x\dot x+f_y\dot y$ |
| $z=f(\mathbf g(\mathbf t))$, $\mathbf g:\mathbb R^p\to\mathbb R^d$ | $\frac{\partial z}{\partial t_l}=\sum_{j=1}^d\partial_jf\,\frac{\partial g_j}{\partial t_l}$ |
| Total derivative | $\frac{d}{dt}f(t,\mathbf x(t))=\partial_tf+\nabla_{\mathbf x}f^T\dot{\mathbf x}$ |
| General | $J_{\mathbf f\circ\mathbf g}(\mathbf t)=J_{\mathbf f}(\mathbf g(\mathbf t))\,J_{\mathbf g}(\mathbf t)$ |

- $J$: Jacobian.
- $l$: Index over the $p$ coordinates of $\mathbf t$.

&nbsp;

### Implicit & Inverse Function Theorems

| Theorem | Statement |
|:--|:--|
| Implicit | $F:\mathbb R^d\times\mathbb R^p\to\mathbb R^p$, $C^1$, $F(\mathbf x_0,\mathbf y_0)=\mathbf 0$, $J_{\mathbf y}F$ invertible → locally $\mathbf y=\mathbf y(\mathbf x)$ with $\frac{\partial\mathbf y}{\partial\mathbf x}=-(J_{\mathbf y}F)^{-1}J_{\mathbf x}F$ |
| Inverse | $\mathbf f:\mathbb R^d\to\mathbb R^d$, $C^1$, $J_{\mathbf f}(\mathbf x_0)$ invertible → local $C^1$ inverse near $\mathbf f(\mathbf x_0)$ with $J_{\mathbf f^{-1}}(\mathbf f(\mathbf x))=J_{\mathbf f}(\mathbf x)^{-1}$ |

- **Implicit, why**: Linearize $F=\mathbf 0$ → $J_{\mathbf x}F\,d\mathbf x+J_{\mathbf y}F\,d\mathbf y=\mathbf 0$.
    - $J_{\mathbf y}F$ invertible → the linear system has a unique $d\mathbf y$ for each $d\mathbf x$.
    - Theorem: the nonlinear system inherits this near $(\mathbf x_0,\mathbf y_0)$ → $\mathbf y$ is a function of $\mathbf x$.
    - E.g., $x^2+y^2=1$: $F_y=2y\ne0$ → $y=\pm\sqrt{1-x^2}$ locally; fails at $(\pm1,0)$ (vertical tangent).
- **Inverse, why**: Near $\mathbf x_0$, $\mathbf f\approx$ affine map with linear part $J_{\mathbf f}(\mathbf x_0)$ → invertible linear part → $\mathbf f$ locally one-to-one & onto.
    - Local only: $(r,\theta)\mapsto(r\cos\theta,r\sin\theta)$ has $\det J=r>0$ but repeats every $2\pi$ in $\theta$.

&nbsp;

### Multivariable Extrema

| Test | Statement |
|:--|:--|
| Stationary point | Interior extremum & $f$ differentiable → $\nabla f(\mathbf x^*)=\mathbf 0$ |
| 2D | $D=f_{xx}f_{yy}-f_{xy}^2$ at a stationary point: $D>0,f_{xx}>0$ → min; $D>0,f_{xx}<0$ → max; $D<0$ → saddle; $D=0$ → inconclusive |
| $d$-D | At a stationary point, $f\in C^2$: Hessian PD → strict local min; ND → strict local max; indefinite → saddle; semidefinite → inconclusive |

- Full conditions: [Optimality Conditions](../ml/optim.md#optimality-conditions).

&nbsp;

### Lagrange Multipliers
Extremize $f(\mathbf x)$ s.t. $g_i(\mathbf x)=c_i$, $i=1,\dots,p$:

$$
\mathcal L(\mathbf x,\boldsymbol\lambda)=f(\mathbf x)-\sum_{i=1}^p\lambda_i\big(g_i(\mathbf x)-c_i\big),
\qquad
\nabla_{\mathbf x}\mathcal L=\mathbf 0,\quad g_i(\mathbf x)=c_i
$$

- $\lambda_i$: Multiplier of constraint $i$.
- Necessary at a local extremum $\mathbf x^*$ if $\{\nabla g_i(\mathbf x^*)\}$ is linearly independent (LICQ).
- **Why**: $\nabla_{\mathbf x}\mathcal L=\mathbf 0$ ⇔ $\nabla f=\sum_i\lambda_i\nabla g_i$.
    1. Feasible directions at $\mathbf x^*$ = tangent space $\{\mathbf v:\nabla g_i^T\mathbf v=0\ \forall i\}$.
    2. Extremum → $\nabla f^T\mathbf v=0$ for every feasible $\mathbf v$ → $\nabla f\perp$ tangent space.
    3. Orthogonal complement of the tangent space = $\operatorname{span}\{\nabla g_i\}$ → $\nabla f$ is a combination of the $\nabla g_i$.
    - 1 constraint & $\nabla f(\mathbf x^*)\ne\mathbf 0$: level set of $f$ is tangent to the constraint surface at $\mathbf x^*$.
- Sensitivity: $\frac{\partial f^*}{\partial c_i}=\lambda_i$.
    - $f^*(\mathbf c)$: Optimal value as a function of the constraint levels; needs a regular, smoothly varying optimum.
- Inequality constraints → KKT: [Duality](../ml/optim.md#duality).

&nbsp;

## Integration

### Fundamental Theorems

| Item | Statement |
|:--|:--|
| Riemann integral | $\int_a^bf\,dx=\lim_{\lVert P\rVert\to0}\sum_if(\xi_i)\,\Delta x_i$ |
| Riemann integrability | Bounded & discontinuities of measure zero (Lebesgue criterion) |
| FTC 1 | $f$ continuous on $[a,b]$ → $\frac{d}{dx}\int_a^xf(t)\,dt=f(x)$ |
| FTC 2 | $F'=f$ on $[a,b]$, $f$ integrable → $\int_a^bf\,dx=F(b)-F(a)$ |
| Variable limits | $\frac{d}{dx}\int_{a(x)}^{b(x)}f(t)\,dt=f(b(x))\,b'(x)-f(a(x))\,a'(x)$ |
| Integral MVT | $f$ continuous → $\exists c\in[a,b]:\int_a^bf=f(c)(b-a)$ |
| Weighted MVT | $f$ continuous, $g\ge0$ integrable → $\exists c\in[a,b]:\int_a^bfg=f(c)\int_a^bg$ |

- $P$: Partition of $[a,b]$; $\lVert P\rVert$: max subinterval width.
- $\xi_i$: Any point in subinterval $i$; $\Delta x_i$: its width.

&nbsp;

### Properties

| Property | Formula |
|:--|:--|
| Linearity | $\int(af+bg)=a\int f+b\int g$ |
| Additivity | $\int_a^bf+\int_b^cf=\int_a^cf$ |
| Orientation | $\int_a^bf=-\int_b^af$ |
| Monotonicity | $f\le g$, $a\le b$ → $\int_a^bf\le\int_a^bg$ |
| Triangle | $\left\lvert\int_a^bf\right\rvert\le\int_a^b\lvert f\rvert$, $a\le b$ |
| Odd $f$ | $\int_{-a}^af=0$ |
| Even $f$ | $\int_{-a}^af=2\int_0^af$ |
| Periodic $f$, period $T$ | $\int_c^{c+T}f=\int_0^Tf$ |
| Reflection | $\int_a^bf(x)\,dx=\int_a^bf(a+b-x)\,dx$ |
| Scaling | $\int_a^bf(cx)\,dx=\frac1c\int_{ca}^{cb}f(u)\,du$, $c\ne0$ |
| Cauchy–Schwarz | $\left(\int_a^bfg\right)^2\le\int_a^bf^2\int_a^bg^2$ |
| Average value | $\bar f=\frac1{b-a}\int_a^bf$ |

&nbsp;

### Techniques

| Technique | Formula / Substitution |
|:--|:--|
| Substitution | $\int f(g(x))\,g'(x)\,dx=\int f(u)\,du$; definite: limits $a,b\to g(a),g(b)$ |
| By parts | $\int u\,dv=uv-\int v\,du$; definite: $[uv]_a^b-\int_a^bv\,du$ |
| Tabular parts | $\int p(x)e^{cx}\,dx=e^{cx}\sum_{k\ge0}(-1)^k\frac{p^{(k)}(x)}{c^{k+1}}$, $p$ polynomial, $c\ne0$ |
| Cyclic parts | $\int e^{ax}\sin bx\,dx$: parts twice → original integral reappears → solve for it |
| $\sqrt{a^2-x^2}$, $a>0$ | $x=a\sin\theta$, $\theta\in[-\frac\pi2,\frac\pi2]$ → $a\cos\theta$ |
| $\sqrt{a^2+x^2}$, $a>0$ | $x=a\tan\theta$, $\theta\in(-\frac\pi2,\frac\pi2)$ → $a\sec\theta$ |
| $\sqrt{x^2-a^2}$, $a>0$ | $x=a\sec\theta$ → $a\lvert\tan\theta\rvert$ |
| Rational in $\sin,\cos$ | $t=\tan\frac x2$: $\sin x=\frac{2t}{1+t^2}$, $\cos x=\frac{1-t^2}{1+t^2}$, $dx=\frac{2\,dt}{1+t^2}$ |
| $\sin^mx\cos^nx$ | $m$ odd → $u=\cos x$; $n$ odd → $u=\sin x$; both even → $\sin^2x=\frac{1-\cos2x}2$, $\cos^2x=\frac{1+\cos2x}2$ |
| Partial fractions | $\frac{P}{Q}$, $\deg P<\deg Q$: factor $(x-r)^k$ → $\sum_{i=1}^k\frac{A_i}{(x-r)^i}$; factor $(x^2+px+q)^k$, $p^2<4q$ → $\sum_{i=1}^k\frac{B_ix+C_i}{(x^2+px+q)^i}$ |
| Symmetrization | $I=\int_0^af(x)\,dx=\frac12\int_0^a\big[f(x)+f(a-x)\big]\,dx$ |
| Parameter differentiation | [Differentiation Under the Integral Sign](#calc-leibniz-rule) |

&nbsp;

### Reduction Formulas

| Integral | Reduction |
|:--|:--|
| $\int\sin^nx\,dx$ | $-\frac{\sin^{n-1}x\cos x}n+\frac{n-1}n\int\sin^{n-2}x\,dx$, $n\ge2$ |
| $\int\cos^nx\,dx$ | $\frac{\cos^{n-1}x\sin x}n+\frac{n-1}n\int\cos^{n-2}x\,dx$, $n\ge2$ |
| $\int\tan^nx\,dx$ | $\frac{\tan^{n-1}x}{n-1}-\int\tan^{n-2}x\,dx$, $n\ge2$ |
| $\int x^ne^{cx}\,dx$ | $\frac{x^ne^{cx}}c-\frac nc\int x^{n-1}e^{cx}\,dx$, $c\ne0$ |
| $\int(\ln x)^n\,dx$ | $x(\ln x)^n-n\int(\ln x)^{n-1}\,dx$ |
| $\int\frac{dx}{(x^2+a^2)^n}$ | $\frac{x}{2a^2(n-1)(x^2+a^2)^{n-1}}+\frac{2n-3}{2a^2(n-1)}\int\frac{dx}{(x^2+a^2)^{n-1}}$, $n\ge2$, $a\ne0$ |
| Wallis | $\int_0^{\pi/2}\sin^nx\,dx=\int_0^{\pi/2}\cos^nx\,dx=\frac{(n-1)!!}{n!!}\cdot\frac\pi2$ ($n$ even); $\frac{(n-1)!!}{n!!}$ ($n$ odd) |

&nbsp;

### Antiderivatives

| $f(x)$ | $\int f\,dx$ |
|:--|:--|
| $x^\alpha$ | $\frac{x^{\alpha+1}}{\alpha+1}$, $\alpha\ne-1$ |
| $\frac1x$ | $\ln\lvert x\rvert$ |
| $\frac1{ax+b}$ | $\frac1a\ln\lvert ax+b\rvert$ |
| $\frac{f'(x)}{f(x)}$ | $\ln\lvert f(x)\rvert$ |
| $e^{ax}$ | $\frac{e^{ax}}a$ |
| $a^x$ | $\frac{a^x}{\ln a}$ |
| $xe^{ax}$ | $\frac{e^{ax}(ax-1)}{a^2}$ |
| $\ln x$ | $x\ln x-x$ |
| $x\ln x$ | $\frac{x^2}2\ln x-\frac{x^2}4$ |
| $\sin x$ | $-\cos x$ |
| $\cos x$ | $\sin x$ |
| $\tan x$ | $-\ln\lvert\cos x\rvert$ |
| $\cot x$ | $\ln\lvert\sin x\rvert$ |
| $\sec x$ | $\ln\lvert\sec x+\tan x\rvert$ |
| $\csc x$ | $-\ln\lvert\csc x+\cot x\rvert=\ln\left\lvert\tan\frac x2\right\rvert$ |
| $\sec^2x$ | $\tan x$ |
| $\csc^2x$ | $-\cot x$ |
| $\sec x\tan x$ | $\sec x$ |
| $\csc x\cot x$ | $-\csc x$ |
| $\sin^2x$ | $\frac x2-\frac{\sin2x}4$ |
| $\cos^2x$ | $\frac x2+\frac{\sin2x}4$ |
| $\sinh x$ | $\cosh x$ |
| $\cosh x$ | $\sinh x$ |
| $\tanh x$ | $\ln\cosh x$ |
| $\frac1{a^2+x^2}$ | $\frac1a\arctan\frac xa$ |
| $\frac x{a^2+x^2}$ | $\frac12\ln(a^2+x^2)$ |
| $\frac1{a^2-x^2}$ | $\frac1{2a}\ln\left\lvert\frac{a+x}{a-x}\right\rvert$ |
| $\frac1{\sqrt{a^2-x^2}}$ | $\arcsin\frac xa$ |
| $\frac1{\sqrt{x^2\pm a^2}}$ | $\ln\left\lvert x+\sqrt{x^2\pm a^2}\right\rvert$ |
| $\frac1{x\sqrt{x^2-a^2}}$ | $\frac1a\operatorname{arcsec}\frac{\lvert x\rvert}a$ |
| $\sqrt{a^2-x^2}$ | $\frac x2\sqrt{a^2-x^2}+\frac{a^2}2\arcsin\frac xa$ |
| $\sqrt{x^2\pm a^2}$ | $\frac x2\sqrt{x^2\pm a^2}\pm\frac{a^2}2\ln\left\lvert x+\sqrt{x^2\pm a^2}\right\rvert$ |
| $\arcsin x$ | $x\arcsin x+\sqrt{1-x^2}$ |
| $\arctan x$ | $x\arctan x-\frac12\ln(1+x^2)$ |
| $e^{ax}\sin bx$ | $\frac{e^{ax}(a\sin bx-b\cos bx)}{a^2+b^2}$ |
| $e^{ax}\cos bx$ | $\frac{e^{ax}(a\cos bx+b\sin bx)}{a^2+b^2}$ |

- $a\ne0$; $a>0$ inside square roots, $\arctan$, $\arcsin$, $\operatorname{arcsec}$; $a>0$, $a\ne1$ for $a^x$.

&nbsp;

### Definite Integrals

| Integral | Value | Condition |
|:--|:--|:--|
| $\int_{-\infty}^\infty e^{-ax^2}dx$ | $\sqrt{\frac\pi a}$ | $a>0$ |
| $\int_{-\infty}^\infty e^{-ax^2+bx}dx$ | $\sqrt{\frac\pi a}\,e^{b^2/(4a)}$ | $a>0$ |
| $\int_{-\infty}^\infty x^2e^{-ax^2}dx$ | $\frac{\sqrt\pi}{2a^{3/2}}$ | $a>0$ |
| $\int_0^\infty x^{2n}e^{-ax^2}dx$ | $\frac{(2n-1)!!}{2^{n+1}a^n}\sqrt{\frac\pi a}$ | $a>0$ |
| $\int_0^\infty x^{2n+1}e^{-ax^2}dx$ | $\frac{n!}{2a^{n+1}}$ | $a>0$ |
| $\int_0^\infty x^{s-1}e^{-ax}dx$ | $\frac{\Gamma(s)}{a^s}$; $\frac{n!}{a^{n+1}}$ for $s=n+1$ | $a,s>0$ |
| $\int_0^\infty e^{-ax}\cos bx\,dx$ | $\frac a{a^2+b^2}$ | $a>0$ |
| $\int_0^\infty e^{-ax}\sin bx\,dx$ | $\frac b{a^2+b^2}$ | $a>0$ |
| $\int_0^\infty\frac{\sin x}x\,dx$ | $\frac\pi2$ | Conditionally convergent |
| $\int_0^\infty\frac{dx}{1+x^2}$ | $\frac\pi2$ | |
| $\int_0^\infty\frac{x^{s-1}}{1+x}\,dx$ | $\frac\pi{\sin\pi s}$ | $0<s<1$ |
| $\int_0^1x^{a-1}(1-x)^{b-1}dx$ | $B(a,b)$ | $a,b>0$ |
| $\int_0^1x^c\ln x\,dx$ | $-\frac1{(c+1)^2}$ | $c>-1$ |
| $\int_{-\pi}^\pi\sin mx\sin nx\,dx$ | $\pi\delta_{mn}$ | $m,n\ge1$ integers |
| $\int_{-\pi}^\pi\cos mx\cos nx\,dx$ | $\pi\delta_{mn}$ | $m,n\ge1$ integers |
| $\int_{-\pi}^\pi\sin mx\cos nx\,dx$ | $0$ | $m,n$ integers |

&nbsp;

### Gamma & Beta Functions

| Item | Formula |
|:--|:--|
| Gamma | $\Gamma(s)=\int_0^\infty x^{s-1}e^{-x}\,dx$, $s>0$ |
| Recurrence | $\Gamma(s+1)=s\,\Gamma(s)$; $\Gamma(n+1)=n!$ |
| Half-integer | $\Gamma\left(\frac12\right)=\sqrt\pi$; $\Gamma\left(n+\frac12\right)=\frac{(2n)!}{4^nn!}\sqrt\pi$ |
| Reflection | $\Gamma(s)\Gamma(1-s)=\frac\pi{\sin\pi s}$, $s\notin\mathbb Z$ |
| Beta | $B(a,b)=\int_0^1x^{a-1}(1-x)^{b-1}dx=\frac{\Gamma(a)\Gamma(b)}{\Gamma(a+b)}$ |
| Beta, trig form | $B(a,b)=2\int_0^{\pi/2}\sin^{2a-1}\theta\cos^{2b-1}\theta\,d\theta$ |
| Digamma | $\psi(s)=\frac{d}{ds}\ln\Gamma(s)$; $\psi(s+1)=\psi(s)+\frac1s$; $\psi(1)=-\gamma$ |
| Stirling | $n!\sim\sqrt{2\pi n}\left(\frac ne\right)^n$; $\ln n!=n\ln n-n+\frac12\ln(2\pi n)+O\left(\frac1n\right)$ |
| Error function | $\operatorname{erf}x=\frac2{\sqrt\pi}\int_0^xe^{-t^2}dt$; $\operatorname{erf}(\pm\infty)=\pm1$ |

&nbsp;

### Improper Integrals

| Item | Statement |
|:--|:--|
| Infinite range | $\int_a^\infty f=\lim_{b\to\infty}\int_a^bf$ |
| Singular endpoint | $f$ unbounded near $a$: $\int_a^bf=\lim_{\epsilon\to0^+}\int_{a+\epsilon}^bf$ |
| Several singular points | Split at each one; converges ⇔ every piece converges independently |
| $p$-test at $\infty$ | $\int_1^\infty x^{-p}dx<\infty\Leftrightarrow p>1$ |
| $p$-test at $0$ | $\int_0^1x^{-p}dx<\infty\Leftrightarrow p<1$ |
| Comparison | $0\le f\le g$: $\int g<\infty\Rightarrow\int f<\infty$ |
| Limit comparison | $f,g>0$, $\frac fg\to L\in(0,\infty)$ → both converge or both diverge |
| Absolute convergence | $\int\lvert f\rvert<\infty\Rightarrow\int f$ converges; ❌ converse: $\frac{\sin x}x$ on $[1,\infty)$ |
| Dirichlet test | $f$ monotone, $f\to0$; $\left\lvert\int_a^Bg\right\rvert\le M$ $\forall B$ → $\int_a^\infty fg$ converges |
| Dirichlet, why ($f\in C^1$) | $G(x)=\int_a^xg$ bounded → by parts: $\int_a^Bfg=f(B)G(B)-\int_a^Bf'G$; $f(B)G(B)\to0$; $\int\lvert f'G\rvert\le M\int\lvert f'\rvert=M\lvert f(a)\rvert<\infty$ |
| Cauchy principal value | $\mathrm{PV}\int_{-\infty}^\infty f=\lim_{R\to\infty}\int_{-R}^Rf$; $\mathrm{PV}\int x\,dx=0$, but $\int_{-\infty}^\infty x\,dx$ diverges |
| Integral test | $f\ge0$ decreasing on $[1,\infty)$: $\sum_{n\ge1}f(n)<\infty\Leftrightarrow\int_1^\infty f<\infty$ |
| Sum–integral bounds | Same $f$: $\int_1^{N+1}f\le\sum_{n=1}^Nf(n)\le f(1)+\int_1^Nf$ |

&nbsp;

(calc-leibniz-rule)=
### Differentiation Under the Integral Sign
- **What**: Swap $\frac{d}{dx}$ & $\int$ (Leibniz integral rule).

$$
\frac{d}{dx}\int_{a(x)}^{b(x)}f(x,t)\,dt=f(x,b(x))\,b'(x)-f(x,a(x))\,a'(x)+\int_{a(x)}^{b(x)}\partial_xf(x,t)\,dt
$$

- **Why** (3 sources of change):
    - Upper limit moves at speed $b'$ → adds a sliver of height $f(x,b)$.
    - Lower limit moves at speed $a'$ → removes a sliver of height $f(x,a)$.
    - Integrand changes → $\int\partial_xf$.
- Conditions: $f$ & $\partial_xf$ continuous; $a,b$ differentiable.
- Infinite range: swap valid if $\lvert\partial_xf(x,t)\rvert\le h(t)$ with $\int h<\infty$ (dominated convergence).
- Feynman trick: insert a parameter → differentiate → integrate back.
    - E.g., $I(c)=\int_0^\infty e^{-cx}\frac{\sin x}x\,dx$ → $I'(c)=-\frac1{1+c^2}$, $I(\infty)=0$ → $I(c)=\frac\pi2-\arctan c$ → $I(0)=\frac\pi2$.

&nbsp;

### Applications

| Quantity | Formula |
|:--|:--|
| Area between curves | $\int_a^b\lvert f-g\rvert\,dx$ |
| Polar area | $\frac12\int_{\theta_1}^{\theta_2}r(\theta)^2\,d\theta$ |
| Arc length, graph | $\int_a^b\sqrt{1+f'(x)^2}\,dx$ |
| Arc length, parametric | $\int_{t_1}^{t_2}\sqrt{\dot x^2+\dot y^2}\,dt$ |
| Arc length, polar | $\int_{\theta_1}^{\theta_2}\sqrt{r^2+r'(\theta)^2}\,d\theta$ |
| Volume, disk (about $x$-axis) | $\pi\int_a^bf(x)^2\,dx$ |
| Volume, shell (about $y$-axis) | $2\pi\int_a^bx\,f(x)\,dx$, $0\le a$, $f\ge0$ |
| Surface of revolution (about $x$-axis) | $2\pi\int_a^b\lvert f(x)\rvert\sqrt{1+f'(x)^2}\,dx$ |

&nbsp;

### Numerical Integration
$N$ panels, $h=\frac{b-a}N$, $x_i=a+ih$, $m_i=\frac{x_{i-1}+x_i}2$:

| Rule | Formula | Error (exact $-$ rule) |
|:--|:--|:--|
| Midpoint | $h\sum_{i=1}^Nf(m_i)$ | $\frac{(b-a)h^2}{24}f''(\xi)$ |
| Trapezoid | $\frac h2\left[f(x_0)+2\sum_{i=1}^{N-1}f(x_i)+f(x_N)\right]$ | $-\frac{(b-a)h^2}{12}f''(\xi)$ |
| Simpson ($N$ even) | $\frac h3\left[f(x_0)+4\sum_{i\text{ odd}}f(x_i)+2\sum_{i\text{ even},\,0<i<N}f(x_i)+f(x_N)\right]$ | $-\frac{(b-a)h^4}{180}f^{(4)}(\xi)$ |
| Gauss–Legendre ($N$ nodes) | $\frac{b-a}2\sum_{i=1}^Nw_i\,f\left(\frac{a+b}2+\frac{b-a}2t_i\right)$ | Exact for polynomials of degree $\le2N-1$ |
| Monte Carlo | $\frac{b-a}N\sum_{i=1}^Nf(U_i)$, $U_i\sim\mathcal U[a,b]$ i.i.d. | Std. error $(b-a)\sqrt{\operatorname{Var}f(U)/N}$; needs finite variance; rate $N^{-1/2}$ indep. of dimension |
| Importance sampling | $\int fp=\mathbb E_q\left[f\frac pq\right]$ | $q>0$ wherever $fp\ne0$ |

- $t_i,w_i$: Gauss–Legendre nodes (roots of the degree-$N$ Legendre polynomial) & weights on $[-1,1]$.

&nbsp;

## Multivariable Integration

### Multiple Integrals

| Item | Formula |
|:--|:--|
| Fubini | $\iint_{[a,b]\times[c,d]}f\,dA=\int_a^b\int_c^df\,dy\,dx=\int_c^d\int_a^bf\,dx\,dy$ if $\iint\lvert f\rvert<\infty$ |
| Tonelli | Same swap for $f\ge0$, no integrability condition |
| Change of variables | $\int_{T(U)}f(\mathbf x)\,d\mathbf x=\int_Uf(T(\mathbf u))\,\lvert\det J_T(\mathbf u)\rvert\,d\mathbf u$, $T$ a $C^1$ diffeomorphism |
| Polar | $dx\,dy=r\,dr\,d\theta$ |
| Cylindrical | $dx\,dy\,dz=r\,dr\,d\theta\,dz$ |
| Spherical | $dx\,dy\,dz=\rho^2\sin\phi\,d\rho\,d\phi\,d\theta$ |
| Radial, $\mathbb R^d$ | $\int_{\mathbb R^d}f(\lVert\mathbf x\rVert_2)\,d\mathbf x=S_{d-1}\int_0^\infty f(r)\,r^{d-1}dr$ |
| Unit sphere area | $S_{d-1}=\frac{2\pi^{d/2}}{\Gamma(d/2)}$ |
| Unit ball volume | $V_d=\frac{\pi^{d/2}}{\Gamma(d/2+1)}$ |
| Gaussian | $\int_{\mathbb R^d}e^{-\frac12\mathbf x^TA\mathbf x+\mathbf b^T\mathbf x}\,d\mathbf x=\sqrt{\frac{(2\pi)^d}{\det A}}\,e^{\frac12\mathbf b^TA^{-1}\mathbf b}$, $A$ symmetric PD |

- **Fubini, why $\iint\lvert f\rvert<\infty$**: w/o it, the order can change the answer.
    - $\int_0^1\!\int_0^1\frac{x^2-y^2}{(x^2+y^2)^2}\,dy\,dx=\frac\pi4$, but $dx\,dy$ order gives $-\frac\pi4$ ← $\iint\lvert f\rvert=\infty$ near $\mathbf 0$.
- **Change of variables, why $\lvert\det J_T\rvert$**: $T$ maps a tiny box $d\mathbf u$ → parallelepiped spanned by the columns of $J_T$ (scaled by $du_j$).
    - Its volume $=\lvert\det J_T\rvert\,d\mathbf u$ → $\lvert\det J_T\rvert$ = local volume scaling.
    - E.g., polar: $J=\begin{bmatrix}\cos\theta&-r\sin\theta\\\sin\theta&r\cos\theta\end{bmatrix}$ → $\det J=r$.
- Spherical: $\rho$ = radius; $\phi\in[0,\pi]$ = angle from $+z$; $\theta$ = azimuth.

&nbsp;

### Vector Calculus Operators
$f,g$: scalar fields; $\mathbf F,\mathbf G$: vector fields on $\mathbb R^3$; all $C^2$.

| Operator | Formula | Meaning at point $\mathbf p$ |
|:--|:--|:--|
| Gradient | $\nabla f=(\partial_1f,\partial_2f,\partial_3f)^T$ | Direction & rate of steepest increase of $f$ |
| Divergence | $\nabla\cdot\mathbf F=\sum_j\partial_jF_j$ | Net outflow per unit volume: $\lim_{V\to\mathbf p}\frac1{\lvert V\rvert}\iint_{\partial V}\mathbf F\cdot d\mathbf S$ |
| Curl | $\nabla\times\mathbf F=(\partial_2F_3-\partial_3F_2,\ \partial_3F_1-\partial_1F_3,\ \partial_1F_2-\partial_2F_1)^T$ | Local rotation: $(\nabla\times\mathbf F)\cdot\mathbf n=\lim_{A\to\mathbf p}\frac1{\lvert A\rvert}\oint_{\partial A}\mathbf F\cdot d\mathbf r$ |
| Laplacian | $\Delta f=\nabla\cdot\nabla f=\sum_j\partial_j^2f$ | $f$ vs its surroundings: (mean of $f$ on sphere of radius $r$) $-f(\mathbf p)\approx\frac{r^2}{2d}\Delta f(\mathbf p)$ |

- $A$: Small flat disk through $\mathbf p$ w/ unit normal $\mathbf n$; $\partial A$ oriented by the right-hand rule.
- Curl direction = rotation axis; $\frac12\nabla\times\mathbf F$ = angular velocity of the local rigid-rotation part of the flow (e.g., $\mathbf F=\boldsymbol\omega\times\mathbf x$ → $\nabla\times\mathbf F=2\boldsymbol\omega$).
- Gradient, divergence & Laplacian extend to $\mathbb R^d$; curl is 3D only.

| Identity | Formula |
|:--|:--|
| Curl of grad | $\nabla\times(\nabla f)=\mathbf 0$ |
| Div of curl | $\nabla\cdot(\nabla\times\mathbf F)=0$ |
| Product, grad | $\nabla(fg)=f\nabla g+g\nabla f$ |
| Product, div | $\nabla\cdot(f\mathbf F)=f\,\nabla\cdot\mathbf F+\mathbf F\cdot\nabla f$ |
| Product, curl | $\nabla\times(f\mathbf F)=f\,\nabla\times\mathbf F+\nabla f\times\mathbf F$ |
| Div of cross | $\nabla\cdot(\mathbf F\times\mathbf G)=\mathbf G\cdot(\nabla\times\mathbf F)-\mathbf F\cdot(\nabla\times\mathbf G)$ |
| Curl of curl | $\nabla\times(\nabla\times\mathbf F)=\nabla(\nabla\cdot\mathbf F)-\Delta\mathbf F$ |

- Curl of grad $=\mathbf 0$: gradient fields have no rotation.
- Div of curl $=0$: curl fields have no sources/sinks.
- Both ← mixed partials commute (Schwarz).

&nbsp;

### Line & Surface Integrals
Curve $\mathbf r(t)$, $t\in[t_1,t_2]$; surface $\mathbf r(u,v)$, $(u,v)\in D$:

| Integral | Formula | Meaning |
|:--|:--|:--|
| Scalar line | $\int_Cf\,ds=\int_{t_1}^{t_2}f(\mathbf r(t))\,\lVert\mathbf r'(t)\rVert_2\,dt$ | Total of $f$ along $C$ (e.g., mass of a wire) |
| Vector line | $\int_C\mathbf F\cdot d\mathbf r=\int_{t_1}^{t_2}\mathbf F(\mathbf r(t))\cdot\mathbf r'(t)\,dt$ | Sum of the component of $\mathbf F$ along $C$ (work); $\oint$ = circulation |
| Scalar surface | $\iint_Sf\,dS=\iint_Df(\mathbf r(u,v))\,\lVert\mathbf r_u\times\mathbf r_v\rVert_2\,du\,dv$ | Total of $f$ over $S$ |
| Flux | $\iint_S\mathbf F\cdot d\mathbf S=\iint_D\mathbf F(\mathbf r(u,v))\cdot(\mathbf r_u\times\mathbf r_v)\,du\,dv$ | Sum of the component of $\mathbf F$ through $S$ (flow rate) |

- $\mathbf r_u,\mathbf r_v$: Partial derivatives of the surface parametrization.
- $\mathbf r_u\times\mathbf r_v$: Normal to $S$; its length = area scaling of the parametrization.
- $\mathbf r(u,v)$ must cover $S$ once (except a zero-area set), piecewise smooth w/ $\mathbf r_u\times\mathbf r_v\ne\mathbf 0$; for flux, $\mathbf r_u\times\mathbf r_v$ must point along the chosen orientation.
- Vector line integral & flux reverse sign when orientation reverses; scalar versions do not.

&nbsp;

### Integral Theorems
- **What**: Integral of a derivative over a region = integral of the function over the region's boundary.
    - All are multivariable versions of $\int_a^bF'\,dx=F(b)-F(a)$; boundary of $[a,b]$ = $\{a,b\}$.
- **Why** (one argument for all):
    1. Chop the region into tiny cells.
    2. Per cell: derivative × cell size ≈ integral over the cell's boundary (definitions of div & curl above).
    3. Sum all cells: each interior edge/face is shared by 2 cells w/ opposite orientations → cancels.
    4. Only the region's boundary survives, incl. boundaries of holes. Cell size → 0 makes it exact.
- Assumptions: bounded region / compact oriented surface w/ piecewise smooth boundary; fields $C^1$ up to the boundary.

&nbsp;

#### Gradient Theorem

$$
\int_C\nabla f\cdot d\mathbf r=f(\mathbf r(t_2))-f(\mathbf r(t_1))
$$

- **What**: Work of a gradient field depends only on the endpoints.
- **Why**: FTC along the curve: $\frac{d}{dt}f(\mathbf r(t))=\nabla f(\mathbf r(t))\cdot\mathbf r'(t)$.
- Closed $C$ → $\oint_C\nabla f\cdot d\mathbf r=0$.

&nbsp;

#### Green's Theorem

$$
\oint_{\partial D}(P\,dx+Q\,dy)=\iint_D(\partial_xQ-\partial_yP)\,dA
$$

- $P,Q$: Components of a planar field $(P,Q)$ on region $D\subset\mathbb R^2$.
- **What**: Circulation around the boundary of $D$ = total 2D curl inside $D$.
- **Why**: Tiny square $[x,x+h]\times[y,y+h]$, counterclockwise:
    1. Right − left edges: $\big(Q(x+h,y)-Q(x,y)\big)h\approx\partial_xQ\,h^2$.
    2. Bottom − top edges: $\big(P(x,y)-P(x,y+h)\big)h\approx-\partial_yP\,h^2$.
    3. Circulation $\approx(\partial_xQ-\partial_yP)\,h^2$; shared edges cancel when squares are summed.
- Orientation: positive = region on the left (outer boundary counterclockwise, holes clockwise).
- Flux form: $\oint_{\partial D}(P\,dy-Q\,dx)=\iint_D(\partial_xP+\partial_yQ)\,dA$ → outward flux = total divergence (2D Gauss).
- Area: $\lvert D\rvert=\frac12\oint_{\partial D}(x\,dy-y\,dx)$.

&nbsp;

#### Stokes' Theorem

$$
\oint_{\partial S}\mathbf F\cdot d\mathbf r=\iint_S(\nabla\times\mathbf F)\cdot d\mathbf S
$$

- **What**: Circulation of $\mathbf F$ around the edge of a surface = flux of $\nabla\times\mathbf F$ through the surface.
- **Why**: Green's theorem on every tiny patch of a curved surface.
    1. Tiny patch w/ normal $\mathbf n$ & area $dS$: circulation around it $=(\nabla\times\mathbf F)\cdot\mathbf n\,dS$ (definition of curl).
    2. Adjacent patches traverse their shared edge in opposite directions → cancel.
    3. Sum over $S$ = circulation around the edge $\partial S$ only.
- Orientation (right-hand rule): thumb along $\mathbf n$ → fingers curl along $\partial S$.
    - Equivalently: walk along $\partial S$ w/ head along $\mathbf n$ → $S$ is on your left.
- Consequences:
    - Flux of $\nabla\times\mathbf F$ depends only on the edge $\partial S$, not on the shape of $S$.
    - Closed surface (no edge) → $\iint_S(\nabla\times\mathbf F)\cdot d\mathbf S=0$.
    - Flat $S$ in the $xy$-plane, $\mathbf F=(P,Q,0)$ → Green's theorem.

&nbsp;

#### Divergence Theorem (Gauss)

$$
\iint_{\partial V}\mathbf F\cdot d\mathbf S=\iiint_V\nabla\cdot\mathbf F\,dV
$$

- **What**: Net outward flux through a closed surface = total divergence (source strength) inside.
- **Why**:
    1. Tiny box: net outflow $=\nabla\cdot\mathbf F\,dV$ (definition of divergence).
    2. Outflow through a shared face of one box = inflow of its neighbor → cancel.
    3. Sum over $V$ = flux through the boundary $\partial V$ only (all components, e.g., both spheres of a shell).
- Orientation: $\mathbf n$ points outward.
- $\nabla\cdot\mathbf F=0$ → zero net flux through every closed surface (what enters, exits).
- Any dimension: $\int_{\partial\Omega}\mathbf F\cdot\mathbf n\,dS=\int_\Omega\nabla\cdot\mathbf F\,d\mathbf x$, $\Omega\subset\mathbb R^d$.
    - $\mathbf n=(n_1,\dots,n_d)$: Outward unit normal.

&nbsp;

#### Integration by Parts in $\mathbb R^d$

$$
\int_\Omega u\,\partial_jv\,d\mathbf x=\int_{\partial\Omega}uv\,n_j\,dS-\int_\Omega v\,\partial_ju\,d\mathbf x
$$

- $u,v$: Scalar fields on $\Omega\subset\mathbb R^d$ (here, not functions of $x$).
- **What**: Move a derivative from $v$ to $u$; pay with a boundary term.
- **Why**: Product rule $\partial_j(uv)=u\,\partial_jv+v\,\partial_ju$ → integrate → divergence theorem on the field $uv\,\mathbf e_j$.
- $d=1$: $\int_a^buv'\,dx=[uv]_a^b-\int_a^bvu'\,dx$.

| Identity | Formula | Field used |
|:--|:--|:--|
| Green's 1st | $\int_\Omega(u\,\Delta v+\nabla u\cdot\nabla v)\,d\mathbf x=\int_{\partial\Omega}u\,\partial_{\mathbf n}v\,dS$ | $u\nabla v$ |
| Green's 2nd | $\int_\Omega(u\,\Delta v-v\,\Delta u)\,d\mathbf x=\int_{\partial\Omega}(u\,\partial_{\mathbf n}v-v\,\partial_{\mathbf n}u)\,dS$ | 1st minus 1st w/ $u\leftrightarrow v$ |

- $\partial_{\mathbf n}v=\nabla v\cdot\mathbf n$: Outward normal derivative.
- Green's identities need $u,v\in C^2$.

&nbsp;

#### Conservative Fields
- **What**: $\mathbf F=\nabla f$ for some potential $f$.

| Statement | Holds on |
|:--|:--|
| $\mathbf F=\nabla f$ ⇔ $\int_C\mathbf F\cdot d\mathbf r$ path-independent ⇔ $\oint_C\mathbf F\cdot d\mathbf r=0$ $\forall$ closed $C$ | Open connected domain |
| $\mathbf F=\nabla f$ ⇒ $\nabla\times\mathbf F=\mathbf 0$ | Any domain ($\mathbf F\in C^1$) |
| $\nabla\times\mathbf F=\mathbf 0$ ⇒ $\mathbf F=\nabla f$ | Simply connected domain |

- **Why simply connected**: $\nabla\times\mathbf F=\mathbf 0$ + Stokes → $\oint_C=0$ only if $C$ bounds a surface inside the domain.
    - A loop around a hole bounds no such surface.
    - E.g., $\mathbf F=\frac{(-y,x)}{x^2+y^2}$ on $\mathbb R^2\setminus\{\mathbf 0\}$: 2D curl $=0$, but $\oint$ around the unit circle $=2\pi$.
- Potential: $f(\mathbf x)=\int_{\mathbf x_0}^{\mathbf x}\mathbf F\cdot d\mathbf r$ along any path.

&nbsp;

#### Generalized Stokes Theorem

$$
\int_{\partial\Omega}\omega=\int_\Omega d\omega
$$

- **What**: One theorem behind all the above.
    - $\omega$: Differential form = integrand for a $(k-1)$-dim set, $k=\dim\Omega$.
    - $d$: Exterior derivative; raises form degree by 1; = grad / curl / div in $\mathbb R^3$, depending on $k$.
    - $\wedge$: Oriented product; $dx\wedge dy$ = oriented area element; $dx\wedge dy=-dy\wedge dx$.
    - Assumptions: $\Omega$ compact oriented manifold w/ boundary; $\omega\in C^1$; $\partial\Omega$ gets the induced orientation.

| $\Omega$ ($\dim$) | $\omega$ | $d\omega$ | Theorem |
|:--|:--|:--|:--|
| Interval (1) | $F$ | $F'\,dx$ | FTC |
| Curve (1) | $f$ | $\nabla f\cdot d\mathbf r$ | Gradient |
| Planar region (2) | $P\,dx+Q\,dy$ | $(\partial_xQ-\partial_yP)\,dx\wedge dy$ | Green |
| Surface (2) | $\mathbf F\cdot d\mathbf r$ | $(\nabla\times\mathbf F)\cdot d\mathbf S$ | Stokes |
| Solid (3) | $\mathbf F\cdot d\mathbf S$ | $(\nabla\cdot\mathbf F)\,dV$ | Divergence |

- $d(d\omega)=0$ ⇔ curl grad $=\mathbf 0$ & div curl $=0$.
- Dual fact: a boundary has no boundary ($\partial\partial\Omega=\varnothing$) → e.g., $\partial S$ of a closed surface is empty → flux of a curl through it $=0$.

&nbsp;
