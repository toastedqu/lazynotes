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
# Neural Network
Foundations of feed-forward NNs: single neuron → MLP → universal approximation, plus how gradients are computed (backprop, autodiff). Layers → [Module](module.md). Nonlinearities → [Activation](activation.md). Optimizers → [Optimization](../train/optim.md).

Notations:
- $\mathbf{x}\in\mathbb{R}^{n}$: Input vector
- $\mathbf{y}$: Target
- $\hat{\mathbf{y}}$: NN output
- $\mathcal{L}$: Loss (scalar)
- $\theta$: All learnable params
- $L$: #layers (= #linear layers)
- $\ell\in\{1,\dots,L\}$: Layer idx
- $d_\ell$: Width (#units) of layer $\ell$. $d_0=n$
- $W^{(\ell)}\in\mathbb{R}^{d_\ell\times d_{\ell-1}}$: Weight matrix of layer $\ell$
- $\mathbf{b}^{(\ell)}\in\mathbb{R}^{d_\ell}$: Bias vector of layer $\ell$
- $\mathbf{z}^{(\ell)}\in\mathbb{R}^{d_\ell}$: Pre-activation of layer $\ell$
- $\mathbf{a}^{(\ell)}\in\mathbb{R}^{d_\ell}$: Activation (output) of layer $\ell$. $\mathbf{a}^{(0)}=\mathbf{x}$
- $\varphi$: Element-wise activation function
- $\odot$: Element-wise product
- $f=f_T\circ\cdots\circ f_1$: Program as a chain of $T$ primitive ops
- $\mathbf{h}_t=f_t(\mathbf{h}_{t-1})$: Intermediate value after op $t$. $\mathbf{h}_0=\mathbf{x}$, $\mathbf{h}_T=f(\mathbf{x})$
- $J_t=\frac{\partial\mathbf{h}_t}{\partial\mathbf{h}_{t-1}}$: Local Jacobian of op $t$
- $n_\text{in}$: Input dim of $f$
- $n_\text{out}$: Output dim of $f$

&nbsp;

## Perceptron
- **What**: Linear binary classifier w/ a hard-threshold output, trained by mistake-driven updates.
- **Why**: Learn neuron weights from labeled data, ❌hand-set.
    - McCulloch–Pitts neuron (1943) = threshold unit w/ fixed weights → computes logic, ❌learns.
- **How**:
    - **Inference**: weighted sum of inputs + bias → threshold at 0 → class $\pm1$.
    - **Training**:
        1. Visit samples one by one.
        2. Correct → no change.
        3. Mistake → add $y_i\mathbf{x}_i$ to $\mathbf{w}$ → score $\mathbf{w}^T\mathbf{x}_i$ moves toward the correct sign.
        4. Stop after 1 epoch w/ 0 mistakes.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $y_i\in\{-1,+1\}$: Label of sample $i$.
- Params:
    - $\mathbf{w}\in\mathbb{R}^{n}$: Weight vector.
    - $b\in\mathbb{R}$: Bias.
- Hyperparams:
    - $\eta>0$: LR.

Model:

$$
\hat{y}=\begin{cases}+1 & \mathbf{w}^T\mathbf{x}+b>0\\-1 & \mathbf{w}^T\mathbf{x}+b\leq0\end{cases}
$$

- Score $=0$ → $-1$. Convention only. Training treats it as a mistake either way.

Training:

Perceptron rule. Apply only on a mistake, $y_i(\mathbf{w}^T\mathbf{x}_i+b)\leq0$:

$$
\mathbf{w}\leftarrow\mathbf{w}+\eta\,y_i\mathbf{x}_i,\qquad b\leftarrow b+\eta\,y_i
$$

- = SGD on the **perceptron criterion** $\mathcal{L}_i=\max\left(0,-y_i(\mathbf{w}^T\mathbf{x}_i+b)\right)$.
- **Convergence theorem** (Novikoff, 1962): absorb the bias → $\tilde{\mathbf{x}}_i=(\mathbf{x}_i,1)$, $\tilde{\mathbf{w}}=(\mathbf{w},b)$, $\tilde{\mathbf{w}}_0=\mathbf{0}$. If $\|\tilde{\mathbf{x}}_i\|\leq R$ & $\exists\,\tilde{\mathbf{u}}$ w/ $\|\tilde{\mathbf{u}}\|=1$, $y_i\tilde{\mathbf{u}}^T\tilde{\mathbf{x}}_i\geq\gamma>0\ \forall i$ →

    $$
    \#\text{mistakes}\leq\left(\frac{R}{\gamma}\right)^2
    $$

    - $\gamma$: Margin of the separator $\tilde{\mathbf{u}}$, measured on augmented vectors.
    - $R$: Radius of the augmented data. $R\geq1$ ← constant feature.
```

```{tip} Derivation
:class: dropdown
*Why is the perceptron rule SGD?*
1. Perceptron criterion: $\mathcal{L}_i=\max(0,-y_is_i)$, $s_i=\mathbf{w}^T\mathbf{x}_i+b$.
2. Correct ($y_is_i>0$) → $\mathcal{L}_i=0$ → grad $=0$ → no update.
3. Mistake ($y_is_i<0$) → $\nabla_\mathbf{w}\mathcal{L}_i=-y_i\mathbf{x}_i$, $\partial\mathcal{L}_i/\partial b=-y_i$.
4. SGD step $-\eta\nabla$ → $\mathbf{w}+\eta y_i\mathbf{x}_i$, $b+\eta y_i$.
5. $y_is_i=0$ → kink. The rule takes the update (treats it as a mistake), so it can leave $\mathbf{w}_0=\mathbf{0}$.

*Why does it converge on separable data?*
1. Use augmented vectors $\tilde{\mathbf{x}}_i,\tilde{\mathbf{w}},\tilde{\mathbf{u}}$ from Math. Drop the tildes below.
    - Let $\mathbf{w}_k$ = weights after $k$ mistakes. Set $\eta=1$ (see Q&A).
2. Alignment grows linearly: $\mathbf{w}_{k}^T\mathbf{u}=\mathbf{w}_{k-1}^T\mathbf{u}+y_i\mathbf{x}_i^T\mathbf{u}\geq\mathbf{w}_{k-1}^T\mathbf{u}+\gamma$ → $\mathbf{w}_k^T\mathbf{u}\geq k\gamma$.
3. Norm grows only like $\sqrt{k}$: $\|\mathbf{w}_k\|^2=\|\mathbf{w}_{k-1}\|^2+2y_i\mathbf{w}_{k-1}^T\mathbf{x}_i+\|\mathbf{x}_i\|^2\leq\|\mathbf{w}_{k-1}\|^2+R^2$.
    - Middle term $\leq0$ ← update happens only on a mistake.
    - → $\|\mathbf{w}_k\|^2\leq kR^2$.
4. Cauchy–Schwarz: $k\gamma\leq\mathbf{w}_k^T\mathbf{u}\leq\|\mathbf{w}_k\|\leq\sqrt{k}R$ → $k\leq(R/\gamma)^2$.

*Why can't 1 perceptron learn XOR?*
1. Need $(\mathbf{w},b)$ w/ $y_is_i>0$ on all 4 points:
    - $(0,0)\mapsto-1$: $b<0$.
    - $(1,1)\mapsto-1$: $w_1+w_2+b<0$.
    - $(1,0)\mapsto+1$: $w_1+b>0$.
    - $(0,1)\mapsto+1$: $w_2+b>0$.
2. Add the last 2: $w_1+w_2+2b>0$ → $w_1+w_2+b>-b>0$.
3. Contradicts line 2 → no separating line exists.
```

````{important} Code
:class: dropdown
```python
import torch

class Perceptron:
    def __init__(self, n, lr=1.0):
        self.w = torch.zeros(n)
        self.b = torch.zeros(())
        self.lr = lr

    def __call__(self, X):
        return torch.where(X @ self.w + self.b > 0, 1.0, -1.0)   ## hard threshold -> {-1, +1}

    def fit(self, X, y, epochs=100):
        for _ in range(epochs):
            mistakes = 0
            for xi, yi in zip(X, y):
                if yi * (xi @ self.w + self.b) <= 0:   ## wrong side OR on the boundary
                    self.w += self.lr * yi * xi        ## rotate w toward y_i x_i
                    self.b += self.lr * yi
                    mistakes += 1
            if mistakes == 0:                          ## converged within the epoch budget
                return True
        return False                                   ## no convergence within budget (always the case if non-separable)

## Example
X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y_and = torch.tensor([-1., -1., -1., 1.])
y_xor = torch.tensor([-1., 1., 1., -1.])
p = Perceptron(2)
print(p.fit(X, y_and), p(X))        ## True tensor([-1., -1., -1.,  1.])
print(Perceptron(2).fit(X, y_xor))  ## False -> XOR is not linearly separable
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- Simple, online, 1 sample at a time.
- Finite #mistakes on separable data, independent of $n$ & $m$.
- ❌LR tuning (see below).

*Cons?*
- Linear boundary only → ❌XOR.
- Non-separable data → never converges. Weights cycle forever.
    - Fixes: **pocket** algorithm (keep the best $\mathbf{w}$ so far), **averaged/voted** perceptron.
- Returns any separator, ❌max margin → fragile → [SVM](../../ml/supervised.md#svm).
- ❌Probabilities → [Logistic Regression](../../ml/supervised.md#logistic-regression).

*Why does $\eta$ not matter?*
- Start at $\mathbf{w}_0=\mathbf{0}$, $b_0=0$ → every update is a multiple of $\eta$ → $(\mathbf{w},b)=\eta\cdot(\mathbf{w},b)_{\eta=1}$.
- $\text{sign}$ ignores positive scale → same predictions, same mistake sequence.
- ❌ if $\mathbf{w}_0\neq\mathbf{0}$.

*Perceptron vs logistic regression vs SVM?*
- Same linear score $s=\mathbf{w}^T\mathbf{x}+b$. Different loss on the margin $ys$:
    - Perceptron: $\max(0,-ys)$ → 0 once correct → stops at any separator.
    - Logistic: $\log(1+e^{-ys})$ → never exactly 0 → keeps pushing → probabilities.
    - SVM hinge: $\max(0,1-ys)$ + $\|\mathbf{w}\|^2$ → requires margin $\geq1$ → max margin.

*Why not stack perceptrons & train by gradient descent?*
- Step function: derivative $=0$ almost everywhere → ❌grad signal to earlier layers.
- Fix: smooth activation ([sigmoid](activation.md#sigmoid), later [ReLU](activation.md#relu)) → differentiable → [backprop](#backpropagation).
- → An "MLP" is ❌a stack of perceptrons. The name is historical.

*How do you get XOR anyway?*
- Hidden layer → [MLP](#mlp).
- Or a nonlinear feature map, e.g., add $x_1x_2$ → separable in 3D → kernel idea.
```

&nbsp;

## MLP
- **Name**: Multilayer Perceptron
- **What**: Stack of [Linear](module.md#linear) layers w/ an element-wise [activation](activation.md#activation) between them.
    - a.k.a. feed-forward NN, fully connected NN.
- **Why**: Perceptron = 1 linear boundary → ❌XOR, ❌nonlinear targets.
    - Hidden layers learn features → output layer stays linear, but in learned features.
- **How**:
    1. Each hidden layer: linear map → nonlinearity → new representation.
    2. Output layer: usually linear → logits / unconstrained regression values. Softmax/sigmoid live in the loss.
        - Else a task-specific output activation, e.g., softplus for a positive target.
    3. Train all layers jointly: loss → [backprop](#backpropagation) → optimizer step.

```{note} Math
:class: dropdown
Model:

$$\begin{align*}
&\mathbf{z}^{(\ell)}=W^{(\ell)}\mathbf{a}^{(\ell-1)}+\mathbf{b}^{(\ell)}, && \ell=1,\dots,L \\
&\mathbf{a}^{(\ell)}=\varphi\left(\mathbf{z}^{(\ell)}\right), && \ell=1,\dots,L-1 \\
&\hat{\mathbf{y}}=\mathbf{z}^{(L)}
\end{align*}$$

- $L$ linear layers = $L-1$ hidden layers.
- #params $=\sum_{\ell=1}^{L}\left(d_\ell d_{\ell-1}+d_\ell\right)$.
- Forward FLOPs per sample $\approx2\sum_{\ell=1}^{L}d_\ell d_{\ell-1}$.

Training:

$$
\min_\theta\ \frac{1}{m}\sum_{i=1}^{m}\mathcal{L}\left(\hat{\mathbf{y}}_i,\mathbf{y}_i\right),\qquad\theta=\left\{W^{(\ell)},\mathbf{b}^{(\ell)}\right\}_{\ell=1}^{L}
$$

- Non-convex in $\theta$ → SGD/Adam w/ grads from [backprop](#backpropagation).
```

```{note} Example
:class: dropdown
XOR w/ 2 hidden ReLU units ($n=2$, $d_1=2$, $d_2=1$):
- $h_1=\text{ReLU}(x_1+x_2)$
- $h_2=\text{ReLU}(x_1+x_2-1)$
- $\hat{y}=h_1-2h_2$

| $(x_1,x_2)$ | $h_1$ | $h_2$ | $\hat{y}$ |
|:--|:--|:--|:--|
| $(0,0)$ | 0 | 0 | 0 |
| $(0,1)$ | 1 | 0 | 1 |
| $(1,0)$ | 1 | 0 | 1 |
| $(1,1)$ | 2 | 1 | 0 |

- Hidden layer maps $(0,1)$ & $(1,0)$ to the same point $(1,0)$ → 3 points in $h$-space → XOR is linear there.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, dims, act=nn.ReLU):
        super().__init__()
        ## dims = [d_0, d_1, ..., d_L] -> L Linear layers
        self.layers = nn.ModuleList([nn.Linear(i, o) for i, o in zip(dims[:-1], dims[1:])])
        self.act = act()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.act(layer(x))    ## hidden: linear -> nonlinearity
        return self.layers[-1](x)     ## output: linear only (softmax/sigmoid live in the loss)

## Example: hand-set weights from the Example block -> XOR
mlp = MLP([2, 2, 1])
with torch.no_grad():
    mlp.layers[0].weight.copy_(torch.tensor([[1., 1.], [1., 1.]]))
    mlp.layers[0].bias.copy_(torch.tensor([0., -1.]))   ## h1 = ReLU(x1+x2), h2 = ReLU(x1+x2-1)
    mlp.layers[1].weight.copy_(torch.tensor([[1., -2.]]))
    mlp.layers[1].bias.zero_()                          ## y = h1 - 2*h2
X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
print(mlp(X).squeeze(-1))   ## tensor([0., 1., 1., 0.], grad_fn=...)
```
````

```{attention} Q&A
:class: dropdown
*Why is the activation required?*
- W/o $\varphi$: $W^{(L)}\cdots W^{(1)}\mathbf{x}+\mathbf{b}'$ = 1 linear map → depth adds nothing.

*Why not initialize all weights equal (e.g., 0)?*
- Units in 1 layer get identical grads → stay identical forever → width wasted → [Zero Initialization](../train/train.md#zero-initialization).

*Why is the output layer linear?*
- Loss applies the link (softmax/sigmoid) → fused log-softmax + CE is numerically stable.
- Fused grad is simple: $\partial\mathcal{L}/\partial\mathbf{z}^{(L)}=\text{softmax}(\mathbf{z}^{(L)})-\mathbf{y}$ → [Cross Entropy](../../ml/obj.md#cross-entropy).

*Wide vs deep?*
- 1 wide hidden layer is already universal → [UAT](#universal-approximation-theorem).
- Depth → features compose → exponentially more expressive per param on some targets.
- Depth costs: harder optimization → [vanishing/exploding grads](../train/issues.md#vanishing-exploding-gradient) → [residual connections](module.md#residual-connection), normalization.

*Inductive bias?*
- Weak: every input connects to every unit. ❌locality, ❌weight sharing, ❌order.
- → Images/sequences: needs far more data than CNN/Transformer → [Inductive Bias](../../ml/misc.md#inductive-bias).
- Tabular data w/ no structure → MLP bias is not wrong, but tree ensembles often still win.

*Are the weights identifiable?*
- ❌. Permute the hidden units of layer $\ell$ (rows of $W^{(\ell)}$, $\mathbf{b}^{(\ell)}$ & columns of $W^{(\ell+1)}$) → same function.
- → Up to $\prod_{\ell=1}^{L-1}d_\ell!$ distinct param vectors per minimum. Generic weights (all units distinct) → exactly this many.
- ReLU: scale a unit's input weights & bias by $c>0$ & its output weights by $1/c$ → same function → continuous families of minima.
- → Non-convex landscape → [Mode Connectivity](../theory.md#mode-connectivity).

*Where do MLPs appear in modern models?*
- Transformer FFN = 2-layer MLP per token → [Position-wise FFN](../../nlp/transformer.md#position-wise-ffn).
- Projection/classifier heads, timestep & condition embeddings.
```

&nbsp;

## Universal Approximation Theorem
- **What**: 1-hidden-layer NNs w/ a non-polynomial activation are dense in the continuous functions on compact sets. {cite:p}`cybenko1989approximation`
- **Why**: Need a guarantee that the NN hypothesis class can represent any reasonable target.
    - ✅ → failures come from size, optimization, or data, ❌from the function form.
- **How**: 1D intuition.
    1. Steep sigmoid ≈ step. 2 shifted steps → bump (indicator of an interval).
    2. Sum of bumps, each w/ height = $f^*$ at its grid point → piecewise-constant approx of $f^*$.
    3. Finer grid → error → 0 ← $f^*$ is uniformly continuous on a compact set.
    - ReLU (1D): sum of shifted ReLUs = any continuous piecewise-linear function → interpolates $f^*$ (Derivation).
    - **History**:
        - Cybenko 1989: continuous sigmoidal $\varphi$, unit hypercube.
        - Hornik, Stinchcombe & White 1989; Funahashi 1989: similar results, same year.
        - Hornik 1991: any continuous, bounded, nonconstant $\varphi$.
        - Leshno, Lin, Pinkus & Schocken 1993: universal ⇔ $\varphi$ non-polynomial → covers ReLU. {cite:p}`leshno1993multilayer`

```{note} Math
:class: dropdown
Notations:
- IO:
    - $f^*:K\to\mathbb{R}$: Continuous target function.
    - $K\subset\mathbb{R}^{n}$: Compact input domain.
- Params:
    - $\mathbf{w}_r\in\mathbb{R}^{n}$: Input weights of hidden unit $r$.
    - $b_r\in\mathbb{R}$: Bias of hidden unit $r$.
    - $\alpha_r\in\mathbb{R}$: Output weight of hidden unit $r$.
- Hyperparams:
    - $N$: #hidden units (width).
- Misc:
    - $\varepsilon>0$: Target accuracy.

Definition: Let $\varphi:\mathbb{R}\to\mathbb{R}$ be continuous & not a polynomial. Then $\forall K,f^*,\varepsilon$, $\exists N,\{\alpha_r,\mathbf{w}_r,b_r\}_{r=1}^{N}$:

$$
\sup_{\mathbf{x}\in K}\left|f^*(\mathbf{x})-\sum_{r=1}^{N}\alpha_r\,\varphi\left(\mathbf{w}_r^T\mathbf{x}+b_r\right)\right|<\varepsilon
$$

- Leshno et al.: continuity can relax to locally bounded & piecewise continuous. Non-polynomial is necessary & sufficient.
- Bias $b_r$ is required. Leshno et al. show the result fails w/o it.
- Vector-valued $f^*$ → approximate each output coordinate → stack the hidden units.
- $L^p(\mu)$ version (Hornik 1991): bounded, nonconstant $\varphi$ suffices. ❌continuity needed.
- Deep & narrow dual (Hanin & Sellke 2017): ReLU nets w/ width $n+1$ & unbounded depth are universal on $[0,1]^n$. Width $n$ is not.
```

```{tip} Derivation
:class: dropdown
*Why is a 1-hidden-layer ReLU net universal on $[a,b]\subset\mathbb{R}$?*
1. $f^*$ continuous on compact $[a,b]$ → uniformly continuous → $\exists\delta$: $|x-x'|<\delta$ ⇒ $|f^*(x)-f^*(x')|<\varepsilon/2$.
2. Knots $a=t_0<t_1<\dots<t_N=b$ w/ spacing $<\delta$. Let $g$ = linear interpolant of $f^*$ at the knots.
3. On $[t_{k-1},t_k]$, $g$ lies between $f^*(t_{k-1})$ & $f^*(t_k)$:

    $$
    |f^*(x)-g(x)|\leq|f^*(x)-f^*(t_{k-1})|+|g(x)-f^*(t_{k-1})|<\tfrac{\varepsilon}{2}+\tfrac{\varepsilon}{2}=\varepsilon
    $$

4. Write $g$ as a ReLU net. Let $s_k$ = slope of $g$ on $[t_{k-1},t_k]$, $s_0=0$:

    $$
    g(x)=f^*(t_0)+\sum_{k=0}^{N-1}(s_{k+1}-s_k)\,\text{ReLU}(x-t_k),\qquad x\in[a,b]
    $$

    - Each ReLU switches on at knot $t_k$ & adds the slope change there.
    - On $[t_j,t_{j+1}]$: slope $=\sum_{k\leq j}(s_{k+1}-s_k)=s_{j+1}$ ✅. $g(t_0)=f^*(t_0)$ ✅.
5. → $N$ hidden ReLU units + output bias $f^*(t_0)$ → $\sup|f^*-g|<\varepsilon$.

*Why does it extend to $n>1$?* (Cybenko's argument)
1. Suppose the span of $\{\varphi(\mathbf{w}^T\mathbf{x}+b)\}$ is not dense in $C(K)$.
2. Hahn–Banach + Riesz → $\exists$ nonzero signed measure $\mu$ on $K$ w/ $\int\varphi(\mathbf{w}^T\mathbf{x}+b)\,d\mu=0$ for all $\mathbf{w},b$.
3. Sigmoidal $\varphi$ is **discriminatory**: this forces $\mu=0$. Contradiction.
```

```{note} Example
:class: dropdown
Approximate $f^*(x)=x^2$ on $[0,1]$ w/ 4 ReLUs:
- Interpolation knots: $0,0.25,0.5,0.75,1$. ReLU shifts: the first 4 knots.
- Slopes of the interpolant: $0.25, 0.75, 1.25, 1.75$ → slope changes: $0.25, 0.5, 0.5, 0.5$.
- $g(x)=0.25\,\text{ReLU}(x)+0.5\,\text{ReLU}(x-0.25)+0.5\,\text{ReLU}(x-0.5)+0.5\,\text{ReLU}(x-0.75)$.
- Max error at piece midpoints: $g(0.125)=0.03125$ vs $f^*(0.125)=0.015625$ → error $=h^2/4=0.0156$.
    - $h$: Knot spacing.
- Error $\propto h^2=1/N^2$ → $N\approx\frac{1}{2\sqrt{\varepsilon}}$ units in 1D.
- Same grid in $n$ dims → $\sim N^n$ cells → #units exponential in $n$ → [Curse of Dimensionality](../../ml/misc.md#curse-of-dimensionality).
```

```{attention} Q&A
:class: dropdown
*What does UAT NOT guarantee?*
- **Size**: $N$ is unbounded. Standard $\varphi$ (ReLU, sigmoid) & generic smooth targets → #units grows exponentially w/ $n$.
- **Learnability**: existence only. ❌GD finds these weights. ❌sample-complexity bound.
- **Generalization**: fit on $K$ says nothing about unseen samples or inputs outside $K$ (extrapolation).
- **Target class**: continuous $f^*$ in sup norm. Discontinuous targets only in the $L^p$ sense.

*Why must $\varphi$ be non-polynomial?*
- $\varphi$ = polynomial of degree $q$ → every $\sum_r\alpha_r\varphi(\mathbf{w}_r^T\mathbf{x}+b_r)$ = polynomial of degree $\leq q$.
- → Finite-dim function space, closed → ❌dense, for any $N$.

*If 1 hidden layer suffices, why go deep?*
- **Depth efficiency**: some functions need exponentially more units when shallow.
- Sawtooth: tent map $\tau(x)=2\,\text{ReLU}(x)-4\,\text{ReLU}(x-\tfrac{1}{2})$ on $[0,1]$.
    - Compose $\tau$ $k$ times → $2^k$ linear pieces from $2k$ ReLUs.
    - 1 hidden layer w/ $N$ ReLUs on $\mathbb{R}$ → $\leq N+1$ pieces → exact match needs $\geq2^k-1$ units.
    - Telgarsky 2016: also holds for approximation. Some depth-$\Theta(k^3)$, width-$\Theta(1)$ nets need $\Omega(2^k)$ units to approximate at depth $O(k)$.
- Eldan & Shamir 2016: an approximately radial function on $\mathbb{R}^n$ → 2 hidden layers, poly width ✅; 1 hidden layer needs width exponential in $n$.
- Real targets are often compositional (pixels → edges → parts → objects) → depth matches the structure.

*Any dimension-free rate?*
- Barron 1993: targets w/ finite first Fourier moment $C_f$ → $L^2$ error $O(C_f/\sqrt{N})$. Rate has no explicit $n$.
- Catch: $C_f$ itself can grow exponentially w/ $n$.

*Is universality special to NNs?*
- ❌. Polynomials (Stone–Weierstrass), Fourier series, RBF kernel machines, deep-enough trees are all universal.
- → UAT does NOT explain why NNs win. Inductive bias + SGD dynamics + scale do → [Inductive Bias](../../ml/misc.md#inductive-bias).
```

&nbsp;

## Gradient Computation

### Computational Graph
- **What**: DAG of a computation: nodes = variables & primitive ops, edges = data dependencies.
- **Why**: Hand-deriving the gradient of a huge composite function is infeasible.
    - Primitives have known local derivatives. The graph tells how to combine them.
- **How**:
    1. **Forward**: evaluate nodes in topological order. Save the inputs each op needs for its derivative.
    2. **Backward**: visit nodes in reverse topological order → chain rule → [backprop](#backpropagation).
    - **Static** (define-and-run): build the graph once, then execute.
    - **Dynamic** (define-by-run): build the graph while the code runs, every run.

```{note} Example
:class: dropdown
$f(x_1,x_2)=x_1x_2+\sin x_1$ at $(x_1,x_2)=(2,3)$:
- Nodes: $v_1=x_1x_2=6$, $v_2=\sin x_1\approx0.909$, $f=v_1+v_2\approx6.909$.
- $x_1$ fans out to $v_1$ & $v_2$ → 2 paths to $f$.
- $\frac{\partial f}{\partial x_1}=\underbrace{x_2}_{\text{via }v_1}+\underbrace{\cos x_1}_{\text{via }v_2}=3+\cos2\approx2.584$ → path contributions **add**.
```

```{dropdown} Table: Static vs Dynamic Graph
| | Static (define-and-run) | Dynamic (define-by-run) |
|:--|:--|:--|
| Built | Once, before execution | During execution, every run |
| Examples | TF 1.x, `jax.jit`, `torch.compile` (captured subgraphs) | PyTorch eager, Chainer, TF 2 eager |
| Control flow | Graph ops (`tf.cond`, `lax.scan`) | Native Python `if`/`for` |
| Optimization | Whole-graph: op fusion, memory planning | Per-op; little global optimization |
| Debugging | Harder ← code ≠ executed graph | Easy ← `print`, debugger |
```

```{attention} Q&A
:class: dropdown
*Why must it be acyclic?*
- Cycle → value depends on itself → ❌topological order → ❌evaluation order.
- RNN: unroll over time → DAG → backprop-through-time → [RNN](module.md#rnn).

*Why do gradients add at a fan-out node?*
- Multivariate chain rule: total derivative = sum over all paths.
- Shared weights (RNN steps, conv positions, [tied embeddings](../../nlp/transformer.md#weight-tying)) → grad = sum over all uses.

*What does the graph store?*
- Tensors that backward needs: inputs of matmuls, masks of ReLUs, softmax outputs.
- → Activation memory ∝ depth × batch size → dominant training memory cost.

*Non-differentiable nodes?*
- Kinks (ReLU at 0, $|x|$, max): pick a subgradient. PyTorch: $\text{ReLU}'(0)=0$.
- Discrete ops (argmax, sampling): grad $=0$ or undefined → straight-through estimator, [reparameterization trick](../gen/vae.md#reparameterization-trick), or score-function estimators.
```

&nbsp;

### Backpropagation
- **What**: [Reverse-mode autodiff](#reverse-mode) of a scalar loss w.r.t. all NN params.
- **Why**:
    - *Why do we need it?* Gradient-based training needs $\nabla_\theta\mathcal{L}$ for $10^6$–$10^{12}$ params.
        - Finite differences → $\geq1$ forward pass per param & inexact → infeasible.
    - *Why does it work?* Dynamic programming on the chain rule.
        - Every path from layer $\ell$ to $\mathcal{L}$ passes through $\mathbf{z}^{(\ell+1)}$.
        - Compute error $\boldsymbol{\delta}^{(\ell+1)}$ once → reuse it for all earlier layers → all grads in 1 backward pass.
- **How**:
    1. **Forward**: compute & cache $\mathbf{z}^{(\ell)},\mathbf{a}^{(\ell)}$ for every layer.
    2. **Output error**: grad of the loss w.r.t. the output logits.
    3. **Backward**: send the error through the transposed weights → gate it by $\varphi'$ → error of the previous layer.
    4. **Param grads**: layer error × cached layer input (outer product).
    5. **Update**: hand grads to the optimizer → [Optimization](../train/optim.md).

`````{note} Math
:class: dropdown
````{tab-set}
```{tab} Vector
Notations:
- Intermediate values:
    - $\boldsymbol{\delta}^{(\ell)}=\frac{\partial\mathcal{L}}{\partial\mathbf{z}^{(\ell)}}\in\mathbb{R}^{d_\ell}$: Error signal of layer $\ell$.

Process (1 sample):
1. Forward: $\mathbf{z}^{(\ell)}=W^{(\ell)}\mathbf{a}^{(\ell-1)}+\mathbf{b}^{(\ell)}$, $\mathbf{a}^{(\ell)}=\varphi(\mathbf{z}^{(\ell)})$ for $\ell<L$, $\hat{\mathbf{y}}=\mathbf{z}^{(L)}$.
2. Output error:

    $$
    \boldsymbol{\delta}^{(L)}=\nabla_{\hat{\mathbf{y}}}\mathcal{L}=\begin{cases}\hat{\mathbf{y}}-\mathbf{y} & \text{MSE }\mathcal{L}=\tfrac{1}{2}\|\hat{\mathbf{y}}-\mathbf{y}\|^2\\ \text{softmax}(\mathbf{z}^{(L)})-\mathbf{y} & \text{softmax + CE, one-hot }\mathbf{y}\end{cases}
    $$

3. Backward ($\ell=L-1,\dots,1$):

    $$
    \boldsymbol{\delta}^{(\ell)}=\left(W^{(\ell+1)T}\boldsymbol{\delta}^{(\ell+1)}\right)\odot\varphi'\left(\mathbf{z}^{(\ell)}\right)
    $$

    - $(d_\ell\times d_{\ell+1})(d_{\ell+1})\to d_\ell$ ✅.
4. Param grads ($\ell=1,\dots,L$):

    $$
    \frac{\partial\mathcal{L}}{\partial W^{(\ell)}}=\boldsymbol{\delta}^{(\ell)}\mathbf{a}^{(\ell-1)T}\in\mathbb{R}^{d_\ell\times d_{\ell-1}},\qquad\frac{\partial\mathcal{L}}{\partial\mathbf{b}^{(\ell)}}=\boldsymbol{\delta}^{(\ell)}
    $$
```
```{tab} Tensor
Notations:
- IO:
    - $X\in\mathbb{R}^{m\times n}$: Batch of inputs, 1 row per sample.
    - $Y$: Batch of targets, 1 row per sample.
- Intermediate values:
    - $A^{(\ell)}\in\mathbb{R}^{m\times d_\ell}$: Batch activations. $A^{(0)}=X$.
    - $Z^{(\ell)}\in\mathbb{R}^{m\times d_\ell}$: Batch pre-activations.
    - $\Delta^{(\ell)}=\frac{\partial\mathcal{L}}{\partial Z^{(\ell)}}\in\mathbb{R}^{m\times d_\ell}$: Batch error signal.
- Misc:
    - $\mathbf{1}_m$: All-ones vector of length $m$.

Process ($\mathcal{L}$ = mean over the batch):
1. Forward: $Z^{(\ell)}=A^{(\ell-1)}W^{(\ell)T}+\mathbf{1}_m\mathbf{b}^{(\ell)T}$, $A^{(\ell)}=\varphi(Z^{(\ell)})$ for $\ell<L$, $\hat{Y}=Z^{(L)}$.
2. Output error: $\Delta^{(L)}=\frac{1}{m}\left(P-Y\right)$, $P=\hat{Y}$ (MSE) or $P=\text{softmax}(Z^{(L)})$ row-wise (softmax + CE).
3. Backward:

    $$
    \Delta^{(\ell)}=\left(\Delta^{(\ell+1)}W^{(\ell+1)}\right)\odot\varphi'\left(Z^{(\ell)}\right)
    $$

4. Param grads:

    $$
    \frac{\partial\mathcal{L}}{\partial W^{(\ell)}}=\Delta^{(\ell)T}A^{(\ell-1)}\in\mathbb{R}^{d_\ell\times d_{\ell-1}},\qquad\frac{\partial\mathcal{L}}{\partial\mathbf{b}^{(\ell)}}=\Delta^{(\ell)T}\mathbf{1}_m
    $$

    - Weight grad = sum of per-sample outer products.
```
````
`````

```{tip} Derivation
:class: dropdown
*Where does the $\boldsymbol{\delta}$ recursion come from?*
1. Unit $q$ of layer $\ell$: $a^{(\ell)}_q=\varphi(z^{(\ell)}_q)$ feeds every $z^{(\ell+1)}_p=\sum_qW^{(\ell+1)}_{pq}a^{(\ell)}_q+b^{(\ell+1)}_p$.
    - $p$: Unit idx in layer $\ell+1$.
    - $q$: Unit idx in layer $\ell$.
2. $z^{(\ell)}_q$ reaches $\mathcal{L}$ only through $\{z^{(\ell+1)}_p\}_p$ → chain rule, sum over paths:

    $$
    \delta^{(\ell)}_q=\sum_p\delta^{(\ell+1)}_p\,W^{(\ell+1)}_{pq}\,\varphi'(z^{(\ell)}_q)
    $$

3. $\sum_pW_{pq}\delta_p=(W^T\boldsymbol{\delta})_q$ → $\boldsymbol{\delta}^{(\ell)}=(W^{(\ell+1)T}\boldsymbol{\delta}^{(\ell+1)})\odot\varphi'(\mathbf{z}^{(\ell)})$.
4. $W^{(\ell)}_{pq}$ reaches $\mathcal{L}$ only through $z^{(\ell)}_p$, w/ $\partial z^{(\ell)}_p/\partial W^{(\ell)}_{pq}=a^{(\ell-1)}_q$ → $\frac{\partial\mathcal{L}}{\partial W^{(\ell)}_{pq}}=\delta^{(\ell)}_pa^{(\ell-1)}_q$ → outer product.

*Why is softmax + CE's output error $\text{softmax}(\mathbf{z})-\mathbf{y}$?*
1. $\mathbf{p}=\text{softmax}(\mathbf{z})$, $\mathcal{L}=-\sum_{k'}y_{k'}\log p_{k'}$, $\frac{\partial p_{k'}}{\partial z_k}=p_{k'}(\mathbb{1}[k'=k]-p_k)$.
2. $\frac{\partial\mathcal{L}}{\partial z_k}=-\sum_{k'}y_{k'}(\mathbb{1}[k'=k]-p_k)=-y_k+p_k\sum_{k'}y_{k'}=p_k-y_k$ ← $\sum_{k'}y_{k'}=1$.

*Why is backward ≈ 2× forward FLOPs?*
1. Linear layer forward: $Z=AW^T$ → 1 matmul, $2md_\ell d_{\ell-1}$ FLOPs.
2. Backward: 2 matmuls of the same size: input grad $\Delta W$ & weight grad $\Delta^TA$.
3. → Training step ≈ 3× forward → $\approx6$ FLOPs per param per token (2 forward + 4 backward) → [FLOP](../sys.md#flop).
4. Layer 1 needs no input grad → slightly less.
```

````{important} Code
:class: dropdown
```python
import torch

class MLPBackprop:
    ## MLP w/ ReLU hidden layers, linear output, MSE loss; backward written by hand
    def __init__(self, dims):
        self.W = [torch.randn(o, i) / i**0.5 for i, o in zip(dims[:-1], dims[1:])]
        self.b = [torch.zeros(o) for o in dims[1:]]

    def forward(self, X):
        self.A, self.Z = [X], []               ## cache layer inputs & pre-activations
        for l, (W, b) in enumerate(zip(self.W, self.b)):
            Z = self.A[-1] @ W.T + b           ## (m, d_l)
            self.Z.append(Z)
            self.A.append(Z.relu() if l < len(self.W) - 1 else Z)   ## output layer: linear
        return self.A[-1]

    def backward(self, Y):
        D = (self.A[-1] - Y) / Y.shape[0]      ## Delta^(L) for L = mean_i 0.5*||y_hat_i - y_i||^2
        dW, db = [None] * len(self.W), [None] * len(self.W)
        for l in reversed(range(len(self.W))):
            dW[l] = D.T @ self.A[l]            ## (d_l, d_{l-1}): error x cached input
            db[l] = D.sum(0)                   ## (d_l,)
            if l > 0:
                ## back through W, then gate by ReLU'(Z) = 1[Z > 0]
                D = (D @ self.W[l]) * (self.Z[l - 1] > 0)
        return dW, db

## Example: manual grads == autograd grads
torch.manual_seed(0)
X, Y = torch.randn(4, 3), torch.randn(4, 2)
net = MLPBackprop([3, 5, 2])
for p in net.W + net.b:
    p.requires_grad_()
loss = 0.5 * ((net.forward(X) - Y) ** 2).sum(1).mean()
loss.backward()                                ## autograd reference
dW, db = net.backward(Y)                       ## manual, from the same cached forward
print(all(torch.allclose(g, p.grad, atol=1e-6) for g, p in zip(dW + db, net.W + net.b)))  ## True
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- Exact grads (up to float precision).
- Cost ≈ constant × forward cost. ❌Extra factor of #params.
- Generic: works on any differentiable DAG, not only MLPs.

*Cons?*
- Memory: must cache every activation → [Gradient Checkpointing](../sys.md#gradient-checkpointing).
- Sequential: layer $\ell$ waits for layer $\ell+1$ (update locking) → pipeline bubbles in model parallelism.
- Grad = product of $L$ Jacobians → [vanishing/exploding](../train/issues.md#vanishing-exploding-gradient).

*Backprop vs gradient descent?*
- Backprop computes $\nabla_\theta\mathcal{L}$. GD/Adam uses it to update $\theta$.
- Different jobs. Swapping the optimizer does not change backprop.

*Why do grads vanish or explode?*
- $\boldsymbol{\delta}^{(1)}$ contains the product $\prod_\ell W^{(\ell+1)T}\text{diag}(\varphi'(\mathbf{z}^{(\ell)}))$.
- Each factor's gain $<1$ / $>1$ → shrinks / grows exponentially in $L$.
- Sigmoid: $\varphi'\leq\frac{1}{4}$ → vanishing by default.

*Why cache activations, ❌recompute?*
- $\partial\mathcal{L}/\partial W^{(\ell)}$ needs $\mathbf{a}^{(\ell-1)}$. $\varphi'(\mathbf{z}^{(\ell)})$ needs $\mathbf{z}^{(\ell)}$.
- Recompute = trade compute for memory = gradient checkpointing.

*Why not finite differences?*
- Central difference: $2|\theta|$ forward passes per grad.
- Step $h$: large → truncation error, small → round-off error.
- → Use only for **gradient checking** of a custom backward on a tiny input.

*Is backprop biologically plausible?*
- Mostly ❌. **Weight transport problem**: backward path reuses $W^T$, but synapses are one-way.
- Also needs separate forward/backward phases & stored activations.
- Feedback alignment (Lillicrap et al. 2016): fixed random $B$ instead of $W^T$ → still learns.

*Who invented it?*
- Credit is contested. Linnainmaa 1970: reverse-mode AD. Werbos 1974: applied to NNs. Rumelhart, Hinton & Williams 1986: popularized it.
```

&nbsp;

### Autodiff
- **Name**: Automatic Differentiation (AD)
- **What**: Exact derivatives of a program by the chain rule over its primitive ops.
- **Why**: Need derivatives of arbitrary code (loops, branches), exact & cheap.
    - Numerical differentiation → inexact & slow. Symbolic → expression swell, ❌control flow.
- **How**:
    1. Run the program → trace of primitive ops = [computational graph](#computational-graph).
    2. Each primitive ships a local derivative rule (JVP and/or VJP).
    3. Compose the local rules along the graph:
        - input → output: [Forward Mode](#forward-mode).
        - output → input: [Reverse Mode](#reverse-mode).

```{note} Math
:class: dropdown
Definition (chain rule on Jacobians):

$$
J_f(\mathbf{x})=\frac{\partial f}{\partial\mathbf{x}}=J_TJ_{T-1}\cdots J_1\in\mathbb{R}^{n_\text{out}\times n_\text{in}}
$$

- 1 AD pass does not form $J_f$. It evaluates 1 product w/ a vector:
    - Forward mode: $J_f\mathbf{v}$ (JVP), $\mathbf{v}\in\mathbb{R}^{n_\text{in}}$ → full $J_f$ in $n_\text{in}$ passes.
    - Reverse mode: $\mathbf{u}^TJ_f$ (VJP), $\mathbf{u}\in\mathbb{R}^{n_\text{out}}$ → full $J_f$ in $n_\text{out}$ passes.
- General DAG: same, plus a sum over paths at every fan-out.
```

```{dropdown} Table: Gradient Computation Methods
Cost of the full gradient of a scalar $f:\mathbb{R}^{n_\text{in}}\to\mathbb{R}$.

| Method | Exact? | Cost of full $\nabla f$ | Extra memory | Control flow |
|:--|:--|:--|:--|:--|
| Numerical (finite differences) | ❌ truncation + round-off | $n_\text{in}+1$ evals of $f$ | 1 perturbed input copy | ✅ black box |
| Symbolic | ✅ | Can blow up (expression swell) | Expression size | ❌ closed form only |
| Forward-mode AD | ✅ (float precision) | $n_\text{in}$ JVP passes | 1 tangent per live value (≈2× live primal memory), ❌tape | ✅ |
| Reverse-mode AD | ✅ (float precision) | 1 VJP pass | Tape = total size of all saved intermediates | ✅ |
```

```{attention} Q&A
:class: dropdown
*AD vs symbolic differentiation?*
- AD = symbolic rule per primitive + numeric evaluation of the composition.
- → ❌expression swell. Shared sub-expressions are values, computed once.
- Handles loops/branches ← differentiates the path that actually ran.

*Pitfall: AD differentiates the program, not the math.*
- `x if x != 0 else 0.0` → grad at $x=0$ is 0, though $f(x)=x$.
- Unstable formula (e.g., naive $\log\sum e^{z}$) → inf/NaN in forward → NaN grads → use stable primitives (`logsumexp`).
- Derivative of an approximation ≠ approximation of the derivative (e.g., unrolled iterative solvers).

*Which mode when?*
- $n_\text{in}\ll n_\text{out}$ → forward.
- $n_\text{in}\gg n_\text{out}$ → reverse. NN training: $n_\text{in}=|\theta|$, $n_\text{out}=1$ → reverse.

*Higher-order derivatives?*
- Nest the modes. Hessian-vector product $H\mathbf{v}$ = JVP of $\nabla f$ (forward-over-reverse) → constant × grad cost, ❌forming $H$.
- Used by CG & Newton-CG → [CG](../../ml/optim.md#cg).
- Full Hessian = $|\theta|$ HVPs → infeasible for large NNs.
```

&nbsp;

#### Forward Mode
- **What**: Propagate a tangent alongside each value, input → output → JVP $J_f\mathbf{v}$.
- **Why**: 1 directional derivative in 1 pass, ❌stored graph.
    - Cheapest mode for few inputs & many outputs.
- **How**:
    1. Seed: each input carries (value, tangent). Tangent = direction $\mathbf{v}$.
    2. Each primitive maps (value, tangent) → (new value, local derivative × tangent).
    3. Output tangent = $J_f\mathbf{v}$.
    - **Dual numbers**: $a+b\varepsilon$ w/ $\varepsilon^2=0$ → plain arithmetic on duals = forward mode.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{v}\in\mathbb{R}^{n_\text{in}}$: Seed direction (JVP input).
- Intermediate values:
    - $\dot{\mathbf{h}}_t=\frac{\partial\mathbf{h}_t}{\partial\mathbf{x}}\mathbf{v}$: Tangent of $\mathbf{h}_t$.
- Misc:
    - $\varepsilon$: Dual unit, $\varepsilon\neq0$, $\varepsilon^2=0$.

Process:

$$
\dot{\mathbf{h}}_0=\mathbf{v},\qquad\dot{\mathbf{h}}_t=J_t\,\dot{\mathbf{h}}_{t-1}\quad\Rightarrow\quad\dot{\mathbf{h}}_T=J_T\cdots J_1\mathbf{v}=J_f\mathbf{v}
$$

- Runs in lockstep w/ the forward pass → $\dot{\mathbf{h}}_{t-1}$ can be dropped once used.

Dual numbers (scalar):

$$
g(a+b\varepsilon)=g(a)+g'(a)\,b\,\varepsilon
$$

- $g$: Any differentiable scalar function.
- Product: $(a+b\varepsilon)(c+d\varepsilon)=ac+(ad+bc)\varepsilon$ → product rule for free.
```

```{tip} Derivation
:class: dropdown
*Why does $g(a+b\varepsilon)=g(a)+g'(a)b\varepsilon$ hold?*
1. Polynomials: expand $(a+b\varepsilon)^k=a^k+ka^{k-1}b\varepsilon$ ← all $\varepsilon^{\geq2}$ terms $=0$ → rule holds by pure algebra.
2. Analytic $g$ (exp, sin, log): Taylor series around $a$ → $\varepsilon^2=0$ kills every term after the linear one.
3. Merely differentiable $g$: no series needed. **Define** the dual extension by the rule.
4. Rule is consistent w/ composition: $g_2(g_1(a+b\varepsilon))=g_2(g_1(a))+g_2'(g_1(a))g_1'(a)b\varepsilon$ = chain rule.
5. → The $\varepsilon$-coefficient carries the exact derivative through every op. ❌truncation error, unlike finite differences.
```

````{important} Code
:class: dropdown
```python
import torch

class Dual:
    ## a + b*eps, eps^2 = 0: val = a (primal), tan = b (tangent)
    def __init__(self, val, tan):
        self.val = torch.as_tensor(val, dtype=torch.float32)
        self.tan = torch.as_tensor(tan, dtype=torch.float32)

    def __add__(self, o):
        return Dual(self.val + o.val, self.tan + o.tan)

    def __mul__(self, o):
        ## (a + b eps)(c + d eps) = ac + (ad + bc) eps -> product rule
        return Dual(self.val * o.val, self.val * o.tan + self.tan * o.val)

    def sin(self):
        ## g(a + b eps) = g(a) + g'(a) b eps
        return Dual(torch.sin(self.val), torch.cos(self.val) * self.tan)

## Example: f(x1, x2) = x1*x2 + sin(x1) at (2, 3); seed v = (1, 0) -> df/dx1
x1, x2 = Dual(2.0, 1.0), Dual(3.0, 0.0)
f = x1 * x2 + x1.sin()
print(f.val.item(), f.tan.item())   ## 6.909... 2.583... (= x2 + cos x1); df/dx2 needs a 2nd pass
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- Memory: ❌tape. Tangents live & die w/ their values.
- Simple: operator overloading, no graph, no reverse pass.
- Best for few inputs, many outputs: sensitivity of a whole trajectory/output vector to 1 param.

*Cons?*
- Grad of a scalar loss w.r.t. $|\theta|$ params → $|\theta|$ passes → infeasible for NN training.

*Forward mode vs finite differences?*
- Same quantity: $J_f\mathbf{v}\approx\frac{f(\mathbf{x}+h\mathbf{v})-f(\mathbf{x})}{h}$.
- Forward mode is exact ($\varepsilon^2=0$, ❌step size $h$). 1 JVP costs a small constant × cost of $f$.

*Where is it used in DL?*
- Hessian-vector products (forward-over-reverse).
- JVPs in some generative-model objectives & Jacobian regularizers.
- Forward gradient (Baydin et al. 2022): $(\nabla f\cdot\mathbf{v})\mathbf{v}$, $\mathbf{v}\sim\mathcal{N}(0,I)$ → unbiased grad estimate w/o backward pass, but high variance.
```

&nbsp;

#### Reverse Mode
- **What**: After a forward pass, propagate adjoints output → input → VJP $\mathbf{u}^TJ_f$.
- **Why**: Scalar loss, many params → full gradient in 1 backward pass.
- **How**:
    1. Forward: run the program. Record ops & the values their derivatives need (the **tape**).
    2. Seed: adjoint of the output $=1$.
    3. Reverse topological order: each node sends adjoint × local partial to its parents.
    4. Parents **sum** incoming adjoints. Leaf adjoints = gradient.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{u}\in\mathbb{R}^{n_\text{out}}$: Seed cotangent (VJP input). Scalar loss → $\mathbf{u}=1$.
- Intermediate values:
    - $\bar{\mathbf{h}}_t=\left(\mathbf{u}^T\frac{\partial\mathbf{h}_T}{\partial\mathbf{h}_t}\right)^T$: Adjoint of $\mathbf{h}_t$. Scalar loss → $\bar{\mathbf{h}}_t=\nabla_{\mathbf{h}_t}\mathcal{L}$.

Process:

$$
\bar{\mathbf{h}}_T=\mathbf{u},\qquad\bar{\mathbf{h}}_{t-1}=J_t^T\,\bar{\mathbf{h}}_t\quad\Rightarrow\quad\bar{\mathbf{h}}_0=J_1^T\cdots J_T^T\mathbf{u}=\left(\mathbf{u}^TJ_f\right)^T
$$

- $J_t^T\bar{\mathbf{h}}_t$ needs $\mathbf{h}_{t-1}$ ← saved on the tape during forward.
- DAG w/ scalar nodes: $\bar{v}=\sum_{c\in\text{ch}(v)}\bar{c}\,\frac{\partial c}{\partial v}$.
    - $v$: Any node.
    - $\text{ch}(v)$: Children of $v$ (nodes that consume $v$).
- Backprop = this w/ $\mathcal{L}$ scalar, $\mathbf{u}=1$: $\boldsymbol{\delta}^{(\ell)}=\bar{\mathbf{z}}^{(\ell)}$.
- Cost: time $\leq5\times$ cost of $f$ in op count (Griewank's cheap-gradient bound). Memory = total size of the saved intermediates.
```

```{tip} Derivation
:class: dropdown
*Why reverse for a scalar loss?* Matrix-chain association.
1. $J_f=J_T\cdots J_1$. Same product, 2 orders:
    - Forward: $J_T(\cdots(J_2(J_1\mathbf{v})))$ → matrix × vector per step, but 1 input direction per pass.
    - Reverse: $((\mathbf{u}^TJ_T)J_{T-1})\cdots J_1$ → vector × matrix per step, 1 output direction per pass.
2. Full $\nabla\mathcal{L}$: $n_\text{out}=1$ → 1 reverse pass vs $n_\text{in}$ forward passes.
3. E.g., $\mathbf{h}_0,\dots,\mathbf{h}_{T-1}\in\mathbb{R}^{d}$, $\mathbf{h}_T\in\mathbb{R}$ ($n_\text{in}=d$, $n_\text{out}=1$) → reverse $O(Td^2)$ vs forward $O(Td^3)$ → factor $n_\text{in}$.
```

```{note} Example
:class: dropdown
Same $f(x_1,x_2)=x_1x_2+\sin x_1$ at $(2,3)$ as in [Computational Graph](#computational-graph):
1. Forward: $v_1=6$, $v_2=\sin2$, $f\approx6.909$. Tape: $x_1,x_2$.
2. Seed: $\bar{f}=1$.
3. $f=v_1+v_2$ → $\bar{v}_1=1$, $\bar{v}_2=1$.
4. $v_2=\sin x_1$ → $\bar{x}_1\mathrel{+}=\cos2\cdot\bar{v}_2$.
5. $v_1=x_1x_2$ → $\bar{x}_1\mathrel{+}=x_2\bar{v}_1=3$, $\bar{x}_2\mathrel{+}=x_1\bar{v}_1=2$.
6. → $\nabla f=(3+\cos2,\ 2)\approx(2.584,\ 2)$ in 1 backward pass. Forward mode needs 2 passes.
```

````{important} Code
:class: dropdown
```python
import torch

class Value:
    ## graph node: data + adjoint (grad) + local VJP that pushes grad to parents
    def __init__(self, data, parents=()):
        self.data = torch.as_tensor(data, dtype=torch.float32)
        self.grad = torch.zeros_like(self.data)
        self.parents = parents
        self._backward = lambda: None          ## set by the op that created this node

    def __add__(self, o):
        out = Value(self.data + o.data, (self, o))
        def _backward():
            self.grad += out.grad              ## d(a+b)/da = 1
            o.grad += out.grad                 ## += -> fan-out adjoints SUM
        out._backward = _backward
        return out

    def __mul__(self, o):
        out = Value(self.data * o.data, (self, o))
        def _backward():
            self.grad += o.data * out.grad     ## needs forward value of o -> tape memory
            o.grad += self.data * out.grad
        out._backward = _backward
        return out

    def sin(self):
        out = Value(torch.sin(self.data), (self,))
        def _backward():
            self.grad += torch.cos(self.data) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        ## topological sort -> reverse order: a node's adjoint is complete before it is pushed on
        order, seen = [], set()
        def visit(v):
            if v not in seen:
                seen.add(v)
                for p in v.parents:
                    visit(p)
                order.append(v)
        visit(self)
        self.grad = torch.ones_like(self.data)  ## seed: dL/dL = 1
        for v in reversed(order):
            v._backward()

## Example: same f as in Forward Mode -> ALL partials in 1 backward pass
x1, x2 = Value(2.0), Value(3.0)
f = x1 * x2 + x1.sin()
f.backward()
print(x1.grad.item(), x2.grad.item())   ## 2.583... (= x2 + cos x1) 2.0 (= x1)
```
````

```{attention} Q&A
:class: dropdown
*Why reverse mode for NN training?*
- $n_\text{out}=1$ (loss), $n_\text{in}=|\theta|$ → 1 backward pass vs $|\theta|$ forward-mode passes.

*Costs?*
- Time: matmul-heavy NNs → backward ≈ 2× forward → [Backpropagation](#backpropagation) Derivation.
- Memory: tape holds every needed intermediate → activation memory ∝ depth × batch.
    - Fix: [Gradient Checkpointing](../sys.md#gradient-checkpointing). E.g., $O(\sqrt{L})$ memory for ≈1 extra forward pass (Chen et al. 2016).
    - Fix: reversible layers (RevNet) → recompute inputs from outputs.

*Why must intermediates be stored?*
- Local VJPs depend on forward values: $\times$ needs the other operand, ReLU needs $\text{sign}(z)$, $\partial\mathcal{L}/\partial W$ needs the layer input.

*When does reverse mode lose?*
- $n_\text{out}\gg n_\text{in}$ (e.g., Jacobian of a long output vector w.r.t. a few params) → forward mode.
- Tight memory & very long graphs (long unrolls, ODE solvers) → forward mode, checkpointing, or adjoint methods.

*Gradient vs VJP?*
- Scalar $\mathcal{L}$, $\mathbf{u}=1$ → VJP = gradient.
- Non-scalar output → PyTorch `y.backward()` errors. Pass $\mathbf{u}$: `y.backward(u)` → $\mathbf{u}^TJ$.

*Why `optimizer.zero_grad()` every step?*
- Adjoints accumulate (`+=`) ← needed for fan-out.
- PyTorch also accumulates into `.grad` across `backward()` calls → stale grads add up unless reset.
- Side benefit: deliberate accumulation over micro-batches → [Gradient Accumulation](../sys.md#gradient-accumulation).
```

&nbsp;
