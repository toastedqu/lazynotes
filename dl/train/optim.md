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
# Optimization (GD)
This page ONLY covers Gradient Descent & family because it is most widely used for optimizing NNs.

GD updates params based on grad's **size & direction**.
- **Gradient**: First-order derivative of the loss w.r.t. the corresponding param.
- **Descent**: Subtract the grad.
	- Objective: **Loss minimization**.
	- Assumption: Loss function is **convex**.
	- Descent in both cases:
		- $\frac{\partial\mathcal{L}}{\partial\mathcal{w}}>0\rightarrow \mathcal{L}\propto w\rightarrow$ Reduce $w$ to reduce $\mathcal{L}$ → Subtract the positive grad.
		- $\frac{\partial\mathcal{L}}{\partial\mathcal{w}}<0\rightarrow \mathcal{L}\propto -w\rightarrow$ Raise $w$ to reduce $\mathcal{L}$ → Subtract the negative grad.

But *Why GD only?* Why not others? Because GD is **practical**.
- *Analytical solutions?* Often impossible.
- *Search?* Impractical.
- *Other optimization algorithms?* High computational cost.
- *Second-order derivative?* High computational cost.
- GD: practical, simple, intuitive, iterative, scalable, empirically effective, ...

*How does GD work?* For each param:
1. Calculate grad.
2. Subtract grad.
3. Use new params to calculate loss.
4. Repeat Steps 1-3 till training ends.

```{note} Math
:class: dropdown
Notations:
- IO:
	- $(x_i,y_i)$: Sample.
	- $\mathcal{D}_{\mathrm{mini}}$: Mini-batch.
	- $\mathcal{D}$: Dataset.
- Params:
    - $w_t$: Param at step $t$.
- Hyperparams:
    - $\eta$: Learning rate.

Types:
- **Stochastic (SGD)**:

$$
w_{t+1} \leftarrow w_t - \eta \frac{\partial \mathcal{L}(w_t; x_i,y_i)}{\partial w_t}
$$

- **Mini-Batch**:

$$
w_{t+1} \leftarrow w_t - \eta \frac{\partial \mathcal{L}(w_t; \mathcal{D}_{\mathrm{mini}})}{\partial w_t}
$$

- **Batch**:

$$
w_{t+1} \leftarrow w_t - \eta \frac{\partial L(w_t; \mathcal{D})}{\partial w_t}
$$
```

## Momentum
- **What**: GD + Cache of past movements.
- **Why**: GD is:
	- too slow in flat regions.
	- oscillates too much in steep valley regions.
	- stuck in local minima.
- **How**: **Velocity**: Exponentially decaying moving average of past grads.
	- At each param update step, add a fraction of the previous update to the curr grad.
		- Flat region: Prev grad & Curr grad same direction → Velocity builds up → Faster convergence
		- Valley region: Prev grad & Curr grad diff direction → Velocity cancels out → Oscillations are dampened
	
```{note} Math
:class: dropdown
Notations:
- Params:
	- $w_t$: Param at step $t$.
- Hyperparams:
	- $\beta$: Momentum coeff.
	- $\eta$: Learning rate.
- Misc:
	- $g_t$: Gradient $\frac{\partial\mathcal{L}}{\partial w_{t-1}}$.
	- $v_t$: Velocity at step $t$.

Process:
1. Velocity update:

$$\begin{align*}
&\text{Init}: &&v_0 = 0\\
&\text{EWMA}: &&v_t = \beta v_{t-1} + (1 - \beta) g_t \\
&\text{Accumulation}: &&v_t= \beta v_{t-1} + g_t \\
&\text{Direct}: &&v_t= \beta v_{t-1} + \eta g_t
\end{align*}$$

2. Param update:

$$\begin{align*}
&\text{EWMA \& Accumulation} &&w_t \leftarrow w_{t-1} - \eta v_t \\
&\text{Direct} &&w_t \leftarrow w_{t-1} - v_t 
\end{align*}$$
```

&nbsp;

### NAG (Nesterov Accelerated Gradient)
- **What**: Momentum + "Look-ahead"
- **Why**: Momentum can sometimes jump over the global minimum.
	- It adds the accumulated momentum, THEN considers curr grad.
- **How**:
	1. Look ahead.
	2. Compute grad from the preliminary position (i.e., preliminary jump).
	3. Update param with this preliminary grad + accumulated momentum.

```{note} Math
:class: dropdown
Notations:
- Params:
	- $w_t$: Param at step $t$.
	- $\tilde{w}_t$: Preliminary param at step $t$.
- Hyperparams:
	- $\beta$: Momentum coeff.
	- $\eta$: Learning rate.
- Misc:
	- $g_t$: Gradient $\frac{\partial\mathcal{L}}{\partial w_{t-1}}$.
	- $\tilde{g}_t$: Preliminary grad $\frac{\partial\mathcal{L}}{\partial \tilde{w}_t}$ at step $t$.
	- $v_t$: Velocity at step $t$.

Process:
1. Look-ahead position:

$$
\tilde{w}_t = w_t - \beta v_{t-1}
$$

2. Velocity update (accumulation):

$$
v_t = \beta v_{t-1} + \tilde{g}_t
$$

3. Param update:

$$
w_t \leftarrow w_{t-1} - \eta v_t 
$$
```

&nbsp;

## Adaptive LR
### Adagrad (Adaptive Gradient)
- **What**: GD + Adaptive learning rate for each param.
- **Why**: A single, fixed learning rate is problematic:
	- Diff feature frequencies: Rare/Common features → Rarely/Commonly update their corresponding params → Need to take larger/smaller steps.
	- Diff param sizes: Large/Small params → Large/Small grads in nature
- **How**:
	1. Track past grads for each param.
	2. Scale learning rate by $\sqrt{\sum\text{grads}^2}$.
		- Large/Small past grads → Small/Large learning rate

```{note} Math
:class: dropdown
Notations:
- Params:
	- $w_t$: Param at step $t$.
- Hyperparams:
	- $\eta$: Global learning rate.
	- $\epsilon$: Small constant (prevent division by 0).
- Misc:
	- $g_t$: Gradient $\frac{\partial\mathcal{L}}{\partial w_{t-1}}$.
	- $G_t$: Accumulated squared gradients.

Process:
1. Grad accumulation:

$$\begin{align*}
&\text{Matrix form}: &&G_{t,ii}=G_{t-1,ii}+(g_{t,i})^2 \\
&\text{Vector form}: &&G_t=G_{t-1}+g_t\odot g_t
\end{align*}$$

2. Param update:

$$
w_t \leftarrow w_{t-1} - \frac{\eta}{\sqrt{G_t+\epsilon}}\odot g_t
$$
```

```{attention} Q&A
:class: dropdown
*Why square?*
- Positive + Penalizes larger grads more.

*Why sum?*
- Memory of how active the param has been throughout the entire training process.
```

&nbsp;

### RMSprop (Root Mean Square Propagation)
- **What**: Adagrad w EWMA (Exponentially Weighted Moving Average) of squared grads instead of sum.
- **Why**:
	- Learning rate decays too much ← Grad sum grows till training ends.
- **How**: ❌Grad sum, ✅EWMA of past squared grads → No infinite growth

```{note} Math
:class: dropdown
Notations:
- Params:
    - $w_t$: Param at step $t$.
- Hyperparams:
    - $\beta$: Decay rate for the EWMAs.
	- $\eta$: Learning rate.
    - $\epsilon$: Small constant (prevent division by 0).
- Misc:
    - $g_t$: Grad $\frac{\partial\mathcal{L}}{\partial w_{t-1}}$.
    - $G_t$: EWMA of squared grads.

Process:
1. Accumulate squared grads (EWMA):

$$
G_t=\beta G_{t-1}+(1-\beta)g_t\odot g_t
$$

2. Update param:

$$
w_t \leftarrow w_{t-1} - \frac{\eta}{\sqrt{G_t+\epsilon}}\odot g_t
$$
```

```{attention} Q&A
:class: dropdown
*Why name it RMS?*
- Square: $g^2_t$.
- Mean: EWMA.
- Root: $\frac{1}{\sqrt{G_t+\epsilon}}$.

*Why RMS?*
- Same unit.

*Why EWMA?*
- The older/newer the grad, the less/more influence it has on curr param update.
```

&nbsp;

### Adadelta
- **What**: RMSprop w/o manually set global learning rate + EWMA on param updates.
- **Why**:
	- Performance is highly sensitive to learning rate.
- **How**:
    1.  ❌Grad sum, ✅EWMA of past squared grads ($G_t$) → No infinite growth
	2.  ❌Global LR, ✅EWMA of past squared param updates ($\Delta W_{t}$) → No LR decay
    3.  ❌Fixed LR, ✅Adaptive ratio between RMS of prev param updates & RMS of curr accumulated squared grads ($\frac{\text{RMS}[\Delta w]_{\text{prev}}}{\text{RMS}[g]_{\text{curr}}}$)

```{note} Math
:class: dropdown
Notations:
- Params:
    - $w_t$: Param at step $t$.
- Hyperparams:
    - $\beta$: Decay rate for the EWMAs.
    - $\epsilon$: Small constant (prevent division by 0).
- Misc:
    - $g_t$: Grad $\frac{\partial\mathcal{L}}{\partial w_t}$.
	- $\Delta w_t$: param update at step $t$.
    - $G_t$: EWMA of squared grads.
    - $\Delta W_{t}$: EWMA of squared param updates.

Process:
1.  Accumulate squared grads (EWMA):

$$
G_t = \beta G_{t-1} + (1-\beta) (g_t \odot g_t)
$$

2.  Calculate param update:

$$\begin{align*}
\Delta w_t &=-\frac{\sqrt{\Delta W_{t-1} + \epsilon}}{\sqrt{G_t + \epsilon}} \odot g_t \\
&=-\frac{\text{RMS}[\Delta w]_{t-1}}{\text{RMS}[g]_t}\odot g_t
\end{align*}$$

3.  Accumulate squared param updates (EWMA):**

$$
\Delta W_{t} = \beta \Delta W_{t-1} + (1-\beta) (\Delta w_t \odot \Delta w_t)
$$

4.  Update param:

$$
w_{t+1} = w_t + \Delta w_t
$$
```

```{attention} Q&A
:class: dropdown
*Why RMS ratio?*
1. It's in the exact **same unit** as the value it applies to.
2. $\frac{\text{RMS}[\Delta w]_{t-1}}{\text{RMS}[g]_t}$ has units "$\Delta$param/grad".
3. Multiply this by grad gives $\Delta$param.

*Why RMSprop instead of Adadelta?*
- ⬆️Empirical performance & convergence speed.
- ⬆️Simplicity & Interpretability.
- Adam is built upon RMSprop & outperforms everything above.
```

&nbsp;

## Adam (Adaptive Moment Estimation)
- **What**: Momentum + RMSprop.
- **Why**: Combine benefits from Momentum & RMSprop:
	- **Momentum = 1st moment (mean) of grads**
		- Momentum keeps track of the **direction** of recent grads.
		- Momentum speeds up optimization in that direction.
	- **RMSprop = 2nd moment (uncentered variance) of grads**
		- RMSprop keep track of the **magnitude** of recent grads.
		- RMSprop adapts LR for each param based on its grad size.
- **How**:
	1. Update 1st moment using new grads.
	2. Update 2nd moment using new squared grads.
	3. Bias Correction: EWMAs are init to 0 → They are biased toward 0 early in training → Need to correct it
	4. Update params.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $w_t$: Param at step $t$.
- Hyperparams:
    - $\beta_1$: Decay rate for 1st moment. Default 0.9.
	- $\beta_2$: Decay rate for 2nd moment. Default 0.999.
	- $\eta$: Learning rate.
    - $\epsilon$: Small constant (prevent division by 0).
- Misc:
	- $g_t$: Grad $\frac{\partial\mathcal{L}}{\partial w_{t-1}}$.
    - $v_t$: 1st moment estimate at step $t$.
	- $G_t$: 2nd moment estimate at step $t$.

Process:
1.  Update 1st moment:

$$
v_t = \beta_1 v_{t-1} + (1 - \beta_1) g_t
$$

2. Update 2nd moment:

$$
G_t = \beta_2 G_{t-1} + (1-\beta_2) (g_t \odot g_t)
$$

3. Correct bias:

$$\begin{align*}
&\hat{v}_t=\frac{v_t}{1-\beta_1^t} \\
&\hat{G}_t=\frac{G_t}{1-\beta_2^t}
\end{align*}$$

4. Update params:

$$
w_t \leftarrow w_{t-1} - \frac{\eta}{\sqrt{\hat{G}_t}+\epsilon} \hat{v}_t
$$
```

```{attention} Q&A
:class: dropdown
*Pros?*
- ✅Best empirical performance & convergence speed.

*Cons?*
- ⬆️Memory cost.
- May still get stuck in local optima.
```

&nbsp;

### AdamW
- **What**: Adam + Weight Decay.
- **Why**: L2 regularization.
	- L2 adds an additional change to the original grad: $\lambda w$.
	- → Designed to decay all weights toward 0, at a rate proportional to their magnitudes, regardless of grad history.
	- BUT in Adam, weights with large/small historical grads get a small/large learning rate, thus a small/large weight decay.
	- Adam & L2 conflict by design.
- **How**: Add an extra weight decay step after Adam.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $w_t$: Param at step $t$.
- Hyperparams:
    - $\lambda$: Decay rate.
	- $\eta$: Learning rate.
- Misc:
	- $\Delta w_t^{\text{Adam}}$: Weight decay via Adam.

Final param update:

$$
w_t \leftarrow w_{t-1} - \Delta w_t^{\text{Adam}} - \eta\lambda w_{t-1}
$$
```

```{attention} Q&A
:class: dropdown
*Why not replace Adam?*
- ONLY applicable when L2 regularization is used (or set $\lambda=0$ for other cases for AdamW).
```

&nbsp;

### Nadam
- **What**: Adam w/ Nesterov look-ahead on the 1st moment. {cite:p}`dozat2016incorporating`
- **Why**: [NAG](#nag-nesterov-accelerated-gradient) corrects course before overshooting → often ⬆️ classical Momentum. Adam's 1st moment = classical momentum → swap in NAG.
- **How**:
    1. Update 1st & 2nd moments exactly as Adam.
    2. Look-ahead: in the param update, use the curr momentum $v_t$ where Adam implicitly uses the prev one $v_{t-1}$.
    3. Keep the curr grad term → update params.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\beta_1$: Decay rate for 1st moment.
    - $\beta_2$: Decay rate for 2nd moment.
    - $\eta$: Learning rate.
    - $\epsilon$: Small constant (prevent division by 0).
- Misc:
    - $g_t$: Grad $\frac{\partial\mathcal{L}}{\partial w_{t-1}}$.
    - $v_t$: 1st moment estimate (as in Adam).
    - $G_t$: 2nd moment estimate (as in Adam).
    - $\bar{v}_t$: Nesterov 1st moment.

Process:
1. Update moments (same as Adam):

    $$
    v_t=\beta_1v_{t-1}+(1-\beta_1)g_t,\qquad G_t=\beta_2G_{t-1}+(1-\beta_2)g_t\odot g_t,\qquad \hat{G}_t=\frac{G_t}{1-\beta_2^t}
    $$

2. Nesterov 1st moment:

    $$
    \bar{v}_t=\frac{\beta_1v_t}{1-\beta_1^{t+1}}+\frac{(1-\beta_1)g_t}{1-\beta_1^t}
    $$

3. Update params:

    $$
    w_t\leftarrow w_{t-1}-\frac{\eta}{\sqrt{\hat{G}_t}+\epsilon}\bar{v}_t
    $$

- Constant-$\beta_1$ form. Paper: general schedule $\beta_{1,t}$ w/ products $\prod_i\beta_{1,i}$ in the bias corrections; its experiments use constant $\beta_1=0.975$.
- PyTorch `NAdam` default: warming schedule $\beta_{1,t}=\beta_1\left(1-0.5\cdot0.96^{t/250}\right)$ (also in Dozat's earlier CS229 report).
```

```{tip} Derivation
:class: dropdown
*Why is $\bar{v}_t$ "Nesterov"?*
1. Expand Adam's bias-corrected 1st moment:

    $$
    \hat{v}_t=\frac{v_t}{1-\beta_1^t}=\underbrace{\frac{\beta_1v_{t-1}}{1-\beta_1^t}}_{\text{past momentum}}+\underbrace{\frac{(1-\beta_1)g_t}{1-\beta_1^t}}_{\text{curr grad}}
    $$

2. NAG rewritten: apply the **next** step's momentum now, ❌grad at a shifted point. Dozat's NAG form: $\bar{m}_t=g_t+\mu_{t+1}m_t$.
3. → Replace the past-momentum term $v_{t-1}$ w/ the curr momentum $v_t$.
4. $v_t$ stands in for the next step's momentum → bias-correct it w/ the next step's factor $1-\beta_1^{t+1}$.
5. → $\bar{v}_t$.
```

````{important} Code
:class: dropdown
```python
import torch

class Nadam(torch.optim.Optimizer):
    def __init__(self, params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for w in group["params"]:
                if w.grad is None:
                    continue
                g, s = w.grad, self.state[w]
                if not s:
                    s["t"], s["v"], s["G"] = 0, torch.zeros_like(w), torch.zeros_like(w)
                s["t"] += 1
                t, v, G = s["t"], s["v"], s["G"]
                v.mul_(b1).add_(g, alpha=1 - b1)          ## 1st moment, same as Adam
                G.mul_(b2).addcmul_(g, g, value=1 - b2)   ## 2nd moment, same as Adam
                G_hat = G / (1 - b2 ** t)
                ## look-ahead: curr momentum v_t (corrected as step t+1) + curr grad
                v_bar = b1 * v / (1 - b1 ** (t + 1)) + (1 - b1) * g / (1 - b1 ** t)
                w.sub_(group["lr"] * v_bar / (G_hat.sqrt() + group["eps"]))

## Example
w = torch.nn.Parameter(torch.randn(2, 4))
opt = Nadam([w])
(w ** 2).sum().backward()
opt.step()
print(w.shape)  ## torch.Size([2, 4])
```
````

```{attention} Q&A
:class: dropdown
*Nadam vs Adam?*
- Same memory (2 states) & same cost. Only the numerator of the step changes.
- Gain over Adam: problem-dependent, often small. LLM default stays AdamW.

*Why not evaluate the grad at the look-ahead point, like NAG?*
- Original NAG evaluates the grad at shifted params $w-\beta v$ → must shift the params & shift back.
- Rewritten form: grad at the curr params; look-ahead applied in the update. Same effect, ❌param shifting.

*Why a warming momentum schedule?*
- Rationale (Dozat's CS229 report, after Sutskever et al., 2013): momentum works best w/ a warming schedule.
- PyTorch schedule: $\beta_{1,t}$ starts at $\beta_1/2$ & rises toward $\beta_1$.
- Not required: the ICLR workshop paper's experiments use a constant $\beta_1$.
```

&nbsp;

### AMSGrad
- **What**: Adam w/ a running max of the 2nd moment in the denominator. {cite:p}`reddi2018on`
- **Why**: Adam can fail to converge, even on a 1D convex problem.
    - EWMA forgets rare large grads → $G_t$ shrinks → effective LR $\frac{\eta}{\sqrt{G_t}}$ grows → frequent small grads undo the progress from the rare informative ones.
- **How**: Divide by $G^{\max}_t=\max(G^{\max}_{t-1},G_t)$ + non-increasing $\eta_t$ → per-coordinate effective LR never increases.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\beta_1$: Decay rate for 1st moment.
    - $\beta_2$: Decay rate for 2nd moment.
    - $\eta_t$: Learning rate, non-increasing (theory: $\eta/\sqrt{t}$).
    - $\epsilon$: Small constant (prevent division by 0).
- Misc:
    - $g_t$: Grad $\frac{\partial\mathcal{L}}{\partial w_{t-1}}$.
    - $v_t$: 1st moment estimate (as in Adam).
    - $G_t$: 2nd moment estimate (as in Adam).
    - $G^{\max}_t$: Elementwise running max of $G_t$.

Process:
1. Update moments (same as Adam):

    $$
    v_t=\beta_1v_{t-1}+(1-\beta_1)g_t,\qquad G_t=\beta_2G_{t-1}+(1-\beta_2)g_t\odot g_t
    $$

2. Running max (elementwise):

    $$
    G^{\max}_t=\max\left(G^{\max}_{t-1},G_t\right),\qquad G^{\max}_0=0
    $$

3. Update params:

    $$
    w_t\leftarrow w_{t-1}-\frac{\eta_t}{\sqrt{G^{\max}_t}+\epsilon}v_t
    $$

- Original form: ❌bias correction. Paper: $\epsilon=0$, a projection onto the bounded feasible set, $\beta_1$ may decay over $t$.
- PyTorch `Adam(amsgrad=True)`: max over raw $G_t$, then divide by $1-\beta_2^t$ & bias-correct $v_t$.
    - → Corrected denominator CAN shrink early. E.g., $g_1\ne0$, $g_2=0$ → shrinks by $\sqrt{1+\beta_2}$ → monotonicity only approximate.
```

```{tip} Derivation
:class: dropdown
*Why can Adam fail on a convex problem?* (Reddi et al. counterexample)
1. Online problem on $w\in[-1,1]$: $f_t(w)=Cw$ if $t\bmod3=1$, else $f_t(w)=-w$, w/ $C>2$.
2. Grad sum per 3 steps $=C-2>0$ → optimum $w=-1$.
3. Adam w/ $\beta_1=0$, $\beta_2=\frac{1}{1+C^2}$: $G_t$ has short memory → each grad is divided by roughly its own magnitude.
4. → Big grad $C$ → step ≈ $\eta$ down. Each small grad $-1$ → step ≈ $\eta$ up (≈ $0.7\eta$ right after the big grad).
5. 2 up-steps beat 1 down-step → Adam drifts to $w=+1$, the worst point.

*What is the root cause?*
- Key quantity: change in inverse effective LR, $\Gamma_t=\frac{\sqrt{G_t}}{\eta_t}-\frac{\sqrt{G_{t-1}}}{\eta_{t-1}}$.
- SGD & Adagrad: $\Gamma_t\succeq0$ always (non-increasing LR). Adam: $\Gamma_t$ can be indefinite.
- Running max + non-increasing $\eta_t$ → $\Gamma_t\succeq0$ again.
- Paper's regret bound also assumes: bounded grads, bounded domain (projection), $\eta_t=\eta/\sqrt{t}$, $\beta_1$ conditions.
- ⚠️ The original proof mishandles the hyperparams (treats unequal ones as equal) → repaired later w/ adjusted conditions. {cite:p}`tran2019on`
```

````{important} Code
:class: dropdown
```python
import torch

class AMSGrad(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for w in group["params"]:
                if w.grad is None:
                    continue
                g, s = w.grad, self.state[w]
                if not s:
                    s["v"], s["G"], s["G_max"] = (torch.zeros_like(w) for _ in range(3))
                v, G, G_max = s["v"], s["G"], s["G_max"]
                v.mul_(b1).add_(g, alpha=1 - b1)
                G.mul_(b2).addcmul_(g, g, value=1 - b2)
                torch.maximum(G_max, G, out=G_max)    ## denominator never shrinks
                ## original form: no bias correction; pair w/ a non-increasing lr schedule
                w.sub_(group["lr"] * v / (G_max.sqrt() + group["eps"]))

## Example
w = torch.nn.Parameter(torch.randn(2, 4))
opt = AMSGrad([w])
(w ** 2).sum().backward()
opt.step()
print(opt.state[w]["G_max"].shape)  ## torch.Size([2, 4])
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- Convergence guarantee in the online convex setting (under the assumptions above) → fixes Adam's counterexample.
- ⚠️ The guarantee needed a later proof fix; it does not cover the bias-corrected PyTorch variant as written.

*Cons?*
- +1 state buffer → 3 states per param.
- Effective LR can only shrink → 1 early huge grad caps that coordinate's LR for the rest of training.
- Empirical gain over Adam in DL: small or none in many reports. ⚠️ Results are mixed.

*Does Adam's non-convergence matter in practice?*
- Paper: for any $\beta_1<\sqrt{\beta_2}$, some stochastic convex problem breaks Adam. But that problem depends on the $\beta$s.
- DL practice w/ $\beta_2=0.999$: this failure is rarely the bottleneck → AMSGrad is rarely the default.

*AMSGrad vs Adagrad?*
- Both: non-increasing effective LR.
- Adagrad: denominator = sum → grows forever → LR → 0.
- AMSGrad: denominator = max of an EWMA → bounded by the largest recent-average $g^2$ → LR does not vanish.
```

&nbsp;

### AdaMax
- **What**: Adam w/ the $L^2$-based 2nd moment replaced by an exponentially weighted $L^\infty$ norm. {cite:p}`kingma2015adam`
- **Why**: Generalize Adam's $L^2$ scaling to $L^p$.
    - Large $p$ → numerically unstable, EXCEPT the limit $p\to\infty$ → simple & stable max rule.
- **How**:
    1. 1st moment: same as Adam.
    2. Scale $u_t$: max of decayed past scale & curr $|g_t|$.
    3. Update w/ $v_t/u_t$. Only $v_t$ needs bias correction.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\beta_1$: Decay rate for 1st moment. Default 0.9.
    - $\beta_2$: Decay rate for the $L^\infty$ norm. Default 0.999.
    - $\eta$: Learning rate. Default 0.002.
- Misc:
    - $g_t$: Grad $\frac{\partial\mathcal{L}}{\partial w_{t-1}}$.
    - $v_t$: 1st moment estimate (as in Adam).
    - $u_t$: Exponentially weighted infinity norm.

Process:
1. Update 1st moment:

    $$
    v_t=\beta_1v_{t-1}+(1-\beta_1)g_t
    $$

2. Update infinity norm (elementwise):

    $$
    u_t=\max\left(\beta_2u_{t-1},|g_t|\right),\qquad u_0=0
    $$

3. Update params:

    $$
    w_t\leftarrow w_{t-1}-\frac{\eta}{1-\beta_1^t}\cdot\frac{v_t}{u_t}
    $$

- PyTorch `Adamax` adds $\epsilon$ to $|g_t|$ → avoids $0/0$ for coordinates w/ all-zero grads so far.
```

```{tip} Derivation
:class: dropdown
*Why does $p\to\infty$ give a max?*
1. $L^p$ version of Adam's 2nd moment (decay reparameterized as $\beta_2^p$):

    $$
    G^{(p)}_t=\beta_2^pG^{(p)}_{t-1}+(1-\beta_2^p)|g_t|^p,\qquad \text{scale}=\left(G^{(p)}_t\right)^{1/p}
    $$

2. Unroll from $G^{(p)}_0=0$:

    $$
    \left(G^{(p)}_t\right)^{1/p}=(1-\beta_2^p)^{1/p}\left(\sum_{i=1}^t\left(\beta_2^{t-i}|g_i|\right)^p\right)^{1/p}
    $$

3. $p\to\infty$: $(1-\beta_2^p)^{1/p}\to1$ ← $\beta_2^p\to0$.
4. $p\to\infty$: $L^p$ norm → $L^\infty$ norm → $\max_i\beta_2^{t-i}|g_i|$.
5. $\max_i\beta_2^{t-i}|g_i|=\max\left(\beta_2\max_{i<t}\beta_2^{t-1-i}|g_i|,\ |g_t|\right)=\max(\beta_2u_{t-1},|g_t|)=u_t$.
```

````{important} Code
:class: dropdown
```python
import torch

class AdaMax(torch.optim.Optimizer):
    def __init__(self, params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for w in group["params"]:
                if w.grad is None:
                    continue
                g, s = w.grad, self.state[w]
                if not s:
                    s["t"], s["v"], s["u"] = 0, torch.zeros_like(w), torch.zeros_like(w)
                s["t"] += 1
                t, v, u = s["t"], s["v"], s["u"]
                v.mul_(b1).add_(g, alpha=1 - b1)
                ## L-inf norm: max of decayed past scale & curr |g|; eps guards 0/0
                torch.maximum(b2 * u, g.abs() + group["eps"], out=u)
                ## max is not an average from 0 -> only v needs bias correction
                w.sub_(group["lr"] / (1 - b1 ** t) * v / u)

## Example
w = torch.nn.Parameter(torch.randn(2, 4))
opt = AdaMax([w])
(w ** 2).sum().backward()
opt.step()
print(opt.state[w]["u"].shape)  ## torch.Size([2, 4])
```
````

```{attention} Q&A
:class: dropdown
*Why no bias correction for $u_t$?*
- $u_0=0$ → $u_1=\max(0,|g_1|)=|g_1|$ exactly.
- Max ≠ average → ❌pull toward the zero init.

*Pros?*
- Simple: ❌square, ❌sqrt.
- $u_t\ge|g_t|$ → per-coordinate step magnitude stays bounded relative to $\eta$.

*Cons?*
- Outlier-sensitive: max lets 1 huge grad dominate $u_t$.
    - Outlier $10\times$ typical $|g|$ → dominates for ≈ $\frac{\ln10}{1-\beta_2}\approx2300$ steps at $\beta_2=0.999$ → that coordinate is frozen.
    - Adam: the outlier enters $G_t$ only w/ weight $1-\beta_2$ → diluted.
- Rarely beats Adam in practice.

*Degenerate case?*
- $\beta_1=\beta_2=0$ → $u_t=|g_t|$ → update $=\eta\,\text{sign}(g_t)$ → sign descent. Same limit as Adam w/ $\beta_1=\beta_2=\epsilon=0$. See [Lion](#lion).
```

&nbsp;

## Large-Batch
- **What**: Optimizers that scale each layer's step by a **trust ratio** $\frac{\|w^l\|}{\|\text{update}^l\|}$.
- **Why**: Large batch → fewer steps per epoch → need a larger LR ([Linear Scaling Rule](train.md#linear-scaling-rule)).
    - 1 global large LR → layers w/ small $\|w\|$ & large $\|g\|$ take huge relative steps → divergence. Warmup alone does not fix it at very large batch.

### LARS
- **Name**: Layer-wise Adaptive Rate Scaling {cite:p}`you2017large`
- **What**: SGD-momentum w/ a per-layer LR $\propto\frac{\|w^l\|}{\|g^l\|}$.
- **Why**: $\frac{\|w^l\|}{\|g^l\|}$ differs by >100× across layers (AlexNet-BN at step 1: 5.76 for conv1 weights, 1345 for fc6 weights).
    - Global LR safe for one layer → too large for another → divergence, or too small → slow.
- **How**:
    1. Per layer: trust ratio $=\frac{\|w^l\|}{\|g^l\|+\lambda\|w^l\|}$.
    2. Local LR = global LR × trust coeff × trust ratio.
    3. → Step size $\propto\|w^l\|$, ❌depends on $\|g^l\|$.
    4. Plug the local LR into SGD-momentum.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $w^l_t$: Params of layer $l$ at step $t$.
- Hyperparams:
    - $\eta_t$: Global LR (paper: polynomial decay, power 2).
    - $\kappa$: Trust coeff, $\kappa<1$ (paper's $\eta$).
    - $\lambda$: Weight decay (paper's $\beta$).
    - $\beta$: Momentum coeff (paper's $m$).
- Misc:
    - $g^l_t$: Grad of layer $l$, $\frac{\partial\mathcal{L}}{\partial w^l_{t-1}}$.
    - $r^l_t$: Local LR of layer $l$.
    - $v^l_t$: Velocity of layer $l$.
    - $\|\cdot\|$: $L^2$ norm over all params of the layer.

Process (per layer $l$):
1. Local LR:

    $$
    r^l_t=\kappa\frac{\|w^l_{t-1}\|}{\|g^l_t\|+\lambda\|w^l_{t-1}\|}
    $$

2. Velocity (momentum + L2 term):

    $$
    v^l_t=\beta v^l_{t-1}+\eta_t\,r^l_t\left(g^l_t+\lambda w^l_{t-1}\right)
    $$

3. Update params:

    $$
    w^l_t\leftarrow w^l_{t-1}-v^l_t
    $$

- Paper's Algorithm 1 omits $\kappa$ in the local LR; its Eq. 6 includes it.
```

````{important} Code
:class: dropdown
```python
import torch

class LARS(torch.optim.Optimizer):
    def __init__(self, params, lr=0.1, momentum=0.9, weight_decay=1e-4, trust_coef=1e-3):
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                      weight_decay=weight_decay, trust_coef=trust_coef))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, beta, wd, k = group["lr"], group["momentum"], group["weight_decay"], group["trust_coef"]
            for w in group["params"]:  ## 1 tensor = 1 "layer"
                if w.grad is None:
                    continue
                g, s = w.grad, self.state[w]
                if not s:
                    s["v"] = torch.zeros_like(w)
                w_norm, g_norm = w.norm(), g.norm()
                denom = g_norm + wd * w_norm
                ## trust ratio: layer-wise LR from weight & grad norms
                ## ||w|| = 0 -> ratio 0 -> zero-init layer never moves; common impls use ratio 1
                r = k * w_norm / denom if w_norm > 0 and denom > 0 else 1.0
                s["v"].mul_(beta).add_(lr * r * (g + wd * w))
                w.sub_(s["v"])

## Example
w = torch.nn.Parameter(torch.randn(2, 4))
opt = LARS([w])
(w ** 2).sum().backward()
opt.step()
print(w.shape)  ## torch.Size([2, 4])
```
````

```{attention} Q&A
:class: dropdown
*Why does it work?*
- Ignore momentum & WD: $\|\Delta w^l\|=\eta_t\kappa\|w^l\|$ → every layer changes by the same **relative** amount $\eta_t\kappa$ per step.
- $\|g^l\|$ cancels → vanishing/exploding grads cannot set the step size.

*Results?*
- AlexNet-BN up to batch 8K & ResNet-50 up to batch 32K on ImageNet w/o accuracy loss (paper).

*Cons?*
- Weak on attention models (e.g., BERT) → [LAMB](#lamb).
- Still needs warmup.
- Common practice: exclude biases & norm params from the trust ratio & WD.

*Where is it used?*
- Large-batch ImageNet training.
- Contrastive SSL w/ huge batches, e.g., [SimCLR](../repr/ssl.md#simclr).
```

&nbsp;

### LAMB
- **Name**: Layer-wise Adaptive Moments optimizer for Batch training {cite:p}`you2020large`
- **What**: Adam ratio + WD term, normalized & rescaled per layer by a LARS-style trust ratio.
- **Why**: LARS (SGD-based) performs poorly on attention models (BERT).
    - → Combine Adam's per-coordinate adaptivity w/ per-layer trust ratio.
- **How**:
    1. Compute the direction: bias-corrected Adam ratio $\frac{\hat{v}_t}{\sqrt{\hat{G}_t}+\epsilon}$ + WD term $\lambda w$.
        - WD sits INSIDE the normalized direction → its effective strength is rescaled by the trust ratio, unlike AdamW.
    2. Per layer: normalize this direction to unit norm.
    3. Rescale by $\phi(\|w^l\|)$ → step norm $=\eta_t\,\phi(\|w^l\|)$.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $w^l_t$: Params of layer $l$ at step $t$.
- Hyperparams:
    - $\beta_1$: Decay rate for 1st moment.
    - $\beta_2$: Decay rate for 2nd moment.
    - $\eta_t$: LR.
    - $\lambda$: Weight decay (inside the normalized direction).
    - $\epsilon$: Small constant.
    - $\gamma_\text{lo}$: Lower clip bound of $\phi$.
    - $\gamma_\text{hi}$: Upper clip bound of $\phi$.
- Misc:
    - $\hat{v}_t$: Bias-corrected 1st moment (as in Adam).
    - $\hat{G}_t$: Bias-corrected 2nd moment (as in Adam).
    - $d_t$: Update direction (Adam ratio + WD term).
    - $\phi$: Scaling fn.

Process (per layer $l$):
1. Adam moments & bias correction → $\hat{v}_t$, $\hat{G}_t$.
2. Direction:

    $$
    d_t=\frac{\hat{v}_t}{\sqrt{\hat{G}_t}+\epsilon}+\lambda w_{t-1}
    $$

3. Layer-wise trust ratio & update:

    $$
    w^l_t\leftarrow w^l_{t-1}-\eta_t\frac{\phi\left(\|w^l_{t-1}\|\right)}{\|d^l_t\|}d^l_t
    $$

- $\phi(z)=\min(\max(z,\gamma_\text{lo}),\gamma_\text{hi})$ in practice. $\phi(z)=z$: plain trust ratio.
```

````{important} Code
:class: dropdown
```python
import torch

class LAMB(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for w in group["params"]:  ## 1 tensor = 1 "layer"
                if w.grad is None:
                    continue
                g, s = w.grad, self.state[w]
                if not s:
                    s["t"], s["v"], s["G"] = 0, torch.zeros_like(w), torch.zeros_like(w)
                s["t"] += 1
                t, v, G = s["t"], s["v"], s["G"]
                v.mul_(b1).add_(g, alpha=1 - b1)
                G.mul_(b2).addcmul_(g, g, value=1 - b2)
                ## Adam ratio (bias-corrected) + WD term, both normalized below
                d = (v / (1 - b1 ** t)) / ((G / (1 - b2 ** t)).sqrt() + group["eps"]) \
                    + group["weight_decay"] * w
                w_norm, d_norm = w.norm(), d.norm()
                ## trust ratio phi(||w||)/||d|| w/ phi = identity -> step norm = lr * ||w||
                ## ||w|| = 0 -> phi = 0 -> never moves; common impls use ratio 1 instead
                r = w_norm / d_norm if w_norm > 0 and d_norm > 0 else 1.0
                w.sub_(group["lr"] * r * d)

## Example
w = torch.nn.Parameter(torch.randn(2, 4))
opt = LAMB([w])
(w ** 2).sum().backward()
opt.step()
print(w.shape)  ## torch.Size([2, 4])
```
````

```{attention} Q&A
:class: dropdown
*Results?*
- BERT pretraining: 3 days → 76 min on 1024 TPUv3 chips, batch 64K (phase 1) / 32K (phase 2).

*LAMB vs LARS?*
- Base optimizer: Adam vs SGD-momentum.
- LAMB has 2 adaptivities: per-coordinate (Adam) & per-layer (trust ratio).
- $\beta_1=\beta_2=0$ (& $\epsilon=\lambda=0$) → layer step $=\eta\frac{\phi(\|w^l\|)}{\sqrt{N_l}}\text{sign}(g^l)$ ← $\|\text{sign}(g^l)\|=\sqrt{N_l}$ → sign SGD w/ a per-layer LR.
    - $N_l$: #params in layer $l$.

*Why normalize the update, ❌the grad?*
- Trust ratio should bound the **actual** step relative to $\|w\|$.
- Adam direction ≠ grad direction → normalizing the grad would not bound the step.

*How to scale the LR w/ batch size?*
- Paper: square-root scaling ($\eta\propto\sqrt{B}$) + linear-epoch warmup. Linear scaling is the usual rule for SGD.
```

&nbsp;

## Adafactor
- **What**: Adam w/ a rank-1 factored 2nd moment & no 1st moment by default. {cite:p}`shazeer2018adafactor`
- **Why**: Memory. Adam stores 2 extra values per param → $2pq$ for a $p\times q$ weight.
    - Adafactor ($\beta_1=0$): $p+q$ per matrix → sublinear.
- **How**:
    1. Keep EWMAs of the **row sums** & **col sums** of $g^2$, ❌full $g^2$.
    2. Rebuild the 2nd moment as (row stats) × (col stats) / total.
    3. Step $=g/\sqrt{\text{rebuilt }G}$, then **update clipping**: cap the RMS of the step at $d$.
    4. **Relative step size**: LR $=\rho_t\times$ RMS of the param.
    5. Decay $\hat{\beta}_{2t}$ rises from 0 → ❌bias correction.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $W_t\in\mathbb{R}^{p\times q}$: Weight matrix at step $t$.
- Hyperparams:
    - $\hat{\beta}_{2t}=1-t^{-0.8}$: 2nd-moment decay (rises w/ $t$).
    - $\rho_t=\min\left(10^{-2},\frac{1}{\sqrt{t}}\right)$: Relative step size.
    - $d=1$: Clip threshold.
    - $\epsilon_1=10^{-30}$: Added to $g^2$.
    - $\epsilon_2=10^{-3}$: Floor on param RMS.
- Misc:
    - $g_t\in\mathbb{R}^{p\times q}$: Grad $\frac{\partial\mathcal{L}}{\partial W_{t-1}}$. Lowercase kept to match Adam.
    - $\mathbf{r}_t\in\mathbb{R}^{p}$: EWMA of row sums of $g^2$.
    - $\mathbf{c}_t\in\mathbb{R}^{q}$: EWMA of col sums of $g^2$.
    - $\tilde{G}_t\in\mathbb{R}^{p\times q}$: Factored 2nd moment.
    - $\alpha_t$: Absolute step size.
    - $\mathbf{1}_k$: All-ones vector of length $k$.
    - $\text{RMS}(A)=\sqrt{\text{mean}(A\odot A)}$.

Process:
1. Absolute step size:

    $$
    \alpha_t=\max\left(\epsilon_2,\text{RMS}(W_{t-1})\right)\rho_t
    $$

2. Row & col stats:

    $$\begin{align*}
    \mathbf{r}_t&=\hat{\beta}_{2t}\mathbf{r}_{t-1}+(1-\hat{\beta}_{2t})\left(g_t\odot g_t+\epsilon_1\right)\mathbf{1}_q\\
    \mathbf{c}_t^T&=\hat{\beta}_{2t}\mathbf{c}_{t-1}^T+(1-\hat{\beta}_{2t})\mathbf{1}_p^T\left(g_t\odot g_t+\epsilon_1\right)
    \end{align*}$$

3. Rank-1 reconstruction:

    $$
    \tilde{G}_t=\frac{\mathbf{r}_t\mathbf{c}_t^T}{\mathbf{1}_p^T\mathbf{r}_t}
    $$

4. Unscaled update & clipping (elementwise division & sqrt):

    $$
    U_t=\frac{g_t}{\sqrt{\tilde{G}_t}},\qquad \hat{U}_t=\frac{U_t}{\max\left(1,\text{RMS}(U_t)/d\right)}
    $$

5. Update params:

    $$
    W_t\leftarrow W_{t-1}-\alpha_t\hat{U}_t
    $$

- Vectors (biases, gains): ❌factoring → $\tilde{G}_t$ = EWMA of $g^2+\epsilon_1$.
- $\hat{\beta}_{21}=0$ → first estimate = $g_1^2$ exactly → ❌bias correction.
```

```{tip} Derivation
:class: dropdown
*Why $\tilde{G}=\frac{\mathbf{r}\mathbf{c}^T}{\mathbf{1}^T\mathbf{r}}$?*
1. Goal: rank-1 $\mathbf{a}\mathbf{b}^T$ ($\mathbf{a},\mathbf{b}\ge0$) close to the nonneg matrix $G$.
2. Loss = I-divergence (generalized KL), summed over entries:

    $$
    \sum_{i,j}\left(G_{ij}\log\frac{G_{ij}}{a_ib_j}-G_{ij}+a_ib_j\right)
    $$

3. $\partial/\partial a_i=0$ → $a_i=\frac{\sum_jG_{ij}}{\sum_jb_j}$. $\partial/\partial b_j=0$ → $b_j=\frac{\sum_iG_{ij}}{\sum_ia_i}$.
4. Scale ambiguity ($c\mathbf{a}$, $\mathbf{b}/c$) → choose $\mathbf{a}=G\mathbf{1}$ (row sums) → $\mathbf{b}=\frac{\mathbf{1}^TG}{\mathbf{1}^TG\mathbf{1}}$ (col sums / total).
    - Check: $a_i=\frac{\sum_jG_{ij}}{\sum_jb_j}=\frac{\sum_jG_{ij}}{1}$ ✅.
5. → $\tilde{G}=\frac{(G\mathbf{1})(\mathbf{1}^TG)}{\mathbf{1}^TG\mathbf{1}}$.
6. Row & col sums are linear → EWMA of the sums = sums of the EWMA → track only $\mathbf{r}_t$ & $\mathbf{c}_t$. $\mathbf{1}^T\mathbf{r}_t$ = total.
```

````{important} Code
:class: dropdown
```python
import torch

class Adafactor(torch.optim.Optimizer):
    def __init__(self, params, eps1=1e-30, eps2=1e-3, d=1.0, decay=-0.8):
        super().__init__(params, dict(eps1=eps1, eps2=eps2, d=d, decay=decay))

    @staticmethod
    def rms(x):
        return x.pow(2).mean().sqrt()

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for W in group["params"]:
                if W.grad is None:
                    continue
                g, s = W.grad, self.state[W]
                if not s:
                    s["t"] = 0
                    if W.ndim == 2:
                        s["r"] = torch.zeros(W.shape[0], device=W.device)  ## p row stats
                        s["c"] = torch.zeros(W.shape[1], device=W.device)  ## q col stats
                    else:
                        s["G"] = torch.zeros_like(W)
                s["t"] += 1
                t = s["t"]
                beta2 = 1 - t ** group["decay"]   ## 0 at t=1 -> no bias correction
                rho = min(1e-2, t ** -0.5)        ## relative step size
                alpha = max(group["eps2"], self.rms(W).item()) * rho  ## step scales w/ param RMS
                g2 = g * g + group["eps1"]
                if W.ndim == 2:
                    s["r"].mul_(beta2).add_(g2.sum(dim=1), alpha=1 - beta2)
                    s["c"].mul_(beta2).add_(g2.sum(dim=0), alpha=1 - beta2)
                    ## rank-1 rebuild: outer(row, col) / total; only r & c are stored
                    G_tilde = torch.outer(s["r"], s["c"]) / s["r"].sum()
                else:
                    s["G"].mul_(beta2).add_(g2, alpha=1 - beta2)
                    G_tilde = s["G"]
                U = g / G_tilde.sqrt()
                U = U / max(1.0, (self.rms(U) / group["d"]).item())  ## update clipping
                W.sub_(alpha * U)

## Example
W = torch.nn.Parameter(torch.randn(3, 5))
opt = Adafactor([W])
(W ** 2).sum().backward()
opt.step()
print(opt.state[W]["r"].shape, opt.state[W]["c"].shape)  ## torch.Size([3]) torch.Size([5])
```
````

```{attention} Q&A
:class: dropdown
*Memory?*
- Adam: $2pq$ per matrix.
- Adafactor, $\beta_1=0$: $p+q$.
- Adafactor, $\beta_1>0$: $pq+p+q$.

*Why update clipping?*
- ❌Momentum + slow-decaying 2nd moment → estimate goes stale → it can underestimate the curr $g^2$ → step too large → instability.
- Cap $\text{RMS}(U_t)\le d$ → bounds the step w/o a separate grad-clipping rule.

*Why a rising $\hat{\beta}_{2t}=1-t^{-0.8}$?*
- $t=1$: $\hat{\beta}_{2t}=0$ → no zero-init bias.
- Large $t$: $\hat{\beta}_{2t}\to1$ → long memory, stable estimate.

*Why a relative step size?*
- Step ∝ param RMS → same relative change for params of different scales.
- $\epsilon_2$ floor → params initialized near 0 can still move.

*Cons?*
- Factoring assumes $g^2$ ≈ row effect × col effect. Poor fit → poor per-coordinate scaling.
- ❌Momentum → less smoothing of noisy grads. Many large runs re-add $\beta_1>0$ (e.g., PaLM).

*Where is it used?*
- T5 pretraining.
- PaLM: Adafactor w/o factorization ≈ Adam w/ param-scaled LR.
```

&nbsp;

## Lion
- **Name**: EvoLved Sign Momentum {cite:p}`chen2023symbolic`
- **What**: Sign of an interpolated momentum + decoupled WD.
- **Why**: Found by symbolic program search over optimizers, then simplified.
    - Memory: 1 state (momentum) vs AdamW's 2.
- **How**:
    1. Interpolate momentum & curr grad w/ $\beta_1$.
    2. Take the **sign** → every coordinate moves exactly $\pm\eta$.
    3. Add decoupled WD.
    4. Update momentum w/ a **different**, slower $\beta_2$.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\beta_1$: Interpolation coeff for the update. Default 0.9.
    - $\beta_2$: Momentum decay. Default 0.99.
    - $\eta$: LR.
    - $\lambda$: Decoupled weight decay.
- Misc:
    - $g_t$: Grad $\frac{\partial\mathcal{L}}{\partial w_{t-1}}$.
    - $v_t$: Momentum (EWMA of grads).
    - $c_t$: Update direction before sign.

Process:
1. Interpolate:

    $$
    c_t=\beta_1v_{t-1}+(1-\beta_1)g_t
    $$

2. Update params:

    $$
    w_t\leftarrow w_{t-1}-\eta\left(\text{sign}(c_t)+\lambda w_{t-1}\right)
    $$

3. Update momentum:

    $$
    v_t=\beta_2v_{t-1}+(1-\beta_2)g_t
    $$

- vs AdamW: LR 3–10× smaller, $\lambda$ 3–10× larger → keep $\eta\lambda$ similar.
```

````{important} Code
:class: dropdown
```python
import torch

class Lion(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            b1, b2 = group["betas"]
            lr, wd = group["lr"], group["weight_decay"]
            for w in group["params"]:
                if w.grad is None:
                    continue
                g, s = w.grad, self.state[w]
                if not s:
                    s["v"] = torch.zeros_like(w)
                v = s["v"]
                c = b1 * v + (1 - b1) * g        ## interpolate momentum & curr grad
                w.mul_(1 - lr * wd)              ## decoupled WD: w -= lr * wd * w
                w.sub_(lr * torch.sign(c))       ## uniform-magnitude step
                v.mul_(b2).add_(g, alpha=1 - b2) ## momentum update w/ slower beta2

## Example
w = torch.nn.Parameter(torch.randn(2, 4))
opt = Lion([w])
(w ** 2).sum().backward()
opt.step()
print(w.shape)  ## torch.Size([2, 4])
```
````

```{attention} Q&A
:class: dropdown
*Why a smaller LR?*
- Sign → each coordinate moves exactly $\eta$ → update norm $=\eta\sqrt{\#\text{params}}$ → larger than Adam's at the same $\eta$.

*Why a larger $\lambda$?*
- Effective decay per step $=\eta\lambda$ → $\eta$⬇️ → $\lambda$⬆️ to keep the same strength.

*Why 2 different betas?*
- Update: curr grad weight $1-\beta_1=0.1$ → reacts fast.
- Momentum: memory ≈ $\frac{1}{1-\beta_2}=100$ steps → smooth.
- → Like Nesterov momentum, the step weights the curr grad more than the stored momentum does.

*Pros?*
- ⬇️Memory: 1 state.
- ⬇️Compute: ❌sqrt, ❌division.
- Paper: gain over AdamW grows w/ batch size.

*Cons?*
- ❌Drop-in: must retune LR & WD.
- Paper: gap to AdamW not significant on some large LM & image-text tasks. Smaller gain w/ strong augmentation or batch < 64.

*Relation to other optimizers?*
- $\beta_1=\beta_2=0$ → signSGD.
- Adam w/ $\beta_1=\beta_2=\epsilon=0$ → also signSGD → Lion ≈ Adam's normalization w/o the 2nd moment.
- Paper intuition: sign adds noise → regularization. ⚠️ Heuristic, not proven.
```

&nbsp;

## Preconditioned
- **What**: Step $=-\eta P^{-1}g$ w/ a learned, **non-diagonal** preconditioner $P$.
- **Why**: Adam's $P$ is diagonal → ignores correlations between params → slow on ill-conditioned, rotated loss surfaces.
    - Full-matrix $P$ for $N$ params: $N^2$ memory, $N^3$ inverse → impossible for NNs.
    - → Exploit the matrix shape of each weight: Kronecker-factored $P$ per layer.

### Shampoo
- **What**: Per weight matrix, precondition the grad from both sides by inverse 4th roots of accumulated $GG^T$ & $G^TG$. {cite:p}`gupta2018shampoo`
- **Why**: Full-matrix Adagrad on a $p\times q$ layer: $(pq)^2$ memory. Shampoo: $p^2+q^2$.
- **How**:
    1. Left stats $L$: accumulate $GG^T$ ($p\times p$) → correlations between rows.
    2. Right stats $R$: accumulate $G^TG$ ($q\times q$) → correlations between cols.
    3. Step $=L^{-1/4}GR^{-1/4}$ → Kronecker-factored approximation of full-matrix Adagrad.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $W_t\in\mathbb{R}^{p\times q}$: Weight matrix at step $t$.
- Hyperparams:
    - $\eta$: LR.
    - $\epsilon$: Init (damping) of the stats.
- Misc:
    - $G_t\in\mathbb{R}^{p\times q}$: Grad matrix $\frac{\partial\mathcal{L}}{\partial W_{t-1}}$. Overrides Adam's $G_t$ (2nd moment) in this section.
    - $L_t\in\mathbb{R}^{p\times p}$: Left stats.
    - $R_t\in\mathbb{R}^{q\times q}$: Right stats.
    - $I_k$: $k\times k$ identity.

Process:
1. Init: $L_0=\epsilon I_p$, $R_0=\epsilon I_q$.
2. Accumulate stats:

    $$
    L_t=L_{t-1}+G_tG_t^T,\qquad R_t=R_{t-1}+G_t^TG_t
    $$

3. Update params:

    $$
    W_t\leftarrow W_{t-1}-\eta L_t^{-1/4}G_tR_t^{-1/4}
    $$

- Flattened view (row-major $\text{vec}$): $\text{vec}\left(L^{-1/4}GR^{-1/4}\right)=\left(L^{-1/4}\otimes R^{-1/4}\right)\text{vec}(G)$.
- Order-$k$ tensor: 1 stats matrix per mode, exponent $-\frac{1}{2k}$. Matrix: $k=2$ → $-\frac{1}{4}$.
```

```{tip} Derivation
:class: dropdown
*Why the exponent $-\frac{1}{4}$?*
1. Full-matrix Adagrad: $H_t=\epsilon I+\sum_{s\le t}\text{vec}(G_s)\text{vec}(G_s)^T$, step $=H_t^{-1/2}\text{vec}(G_t)$.
2. Shampoo's Lemma 8 (rank of each $G_s\le r$):

    $$
    \epsilon I+\frac{1}{r}\sum_{s\le t}\text{vec}(G_s)\text{vec}(G_s)^T\preceq L_t^{1/2}\otimes R_t^{1/2}
    $$

3. → $L_t^{1/2}\otimes R_t^{1/2}$ is a structured stand-in for $H_t$ (up to $r$).
4. Adagrad's $-\frac{1}{2}$ power: $\left(L_t^{1/2}\otimes R_t^{1/2}\right)^{-1/2}=L_t^{-1/4}\otimes R_t^{-1/4}$.
5. Kronecker identity → $\left(L^{-1/4}\otimes R^{-1/4}\right)\text{vec}(G)=\text{vec}\left(L^{-1/4}GR^{-1/4}\right)$ ← $L,R$ symmetric.
```

````{important} Code
:class: dropdown
```python
import torch

class Shampoo(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-1, eps=1e-4):
        super().__init__(params, dict(lr=lr, eps=eps))

    @staticmethod
    def inv_root(M, k):
        ## symmetric PSD -> eigendecomposition gives M^{-1/k}
        lam, Q = torch.linalg.eigh(M)
        return Q @ torch.diag(lam.clamp_min(1e-12) ** (-1.0 / k)) @ Q.T

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for W in group["params"]:
                if W.grad is None:
                    continue
                G, s = W.grad, self.state[W]
                assert W.ndim == 2, "matrix case only"
                if not s:
                    p, q = W.shape
                    s["L"] = group["eps"] * torch.eye(p, device=W.device)
                    s["R"] = group["eps"] * torch.eye(q, device=W.device)
                s["L"] += G @ G.T  ## row correlations, p x p
                s["R"] += G.T @ G  ## col correlations, q x q
                ## precondition from both sides: Kronecker-factored full-matrix Adagrad
                W.sub_(group["lr"] * self.inv_root(s["L"], 4) @ G @ self.inv_root(s["R"], 4))

## Example
W = torch.nn.Parameter(torch.randn(3, 5))
opt = Shampoo([W])
(W ** 2).sum().backward()
opt.step()
print(opt.state[W]["L"].shape, opt.state[W]["R"].shape)  ## torch.Size([3, 3]) torch.Size([5, 5])
```
````

```{attention} Q&A
:class: dropdown
*Cost?*
- Memory: $p^2+q^2$ per matrix.
- Inverse roots: eigendecomposition $O(p^3+q^3)$ → in practice recomputed every $k$ steps, often asynchronously.
- Huge dims (e.g., vocab × hidden embedding): $L$ too large → block-partition the matrix, or use a diagonal fallback for that dim.

*Shampoo vs Adam/Adagrad?*
- Adagrad/Adam: diagonal $P$ → per-coordinate scale only.
- Shampoo: Kronecker $P$ → rotates & scales → captures row & col correlations.

*Practical variants?*
- EWMA stats ($\beta_2$) instead of sums.
- **Grafting**: take the step direction from Shampoo, the step norm from Adam/Adagrad → reuses a known-good LR.

*Degenerate case?*
- ❌Accumulation (stats = curr $GG^T$, $G^TG$ only), $\epsilon\to0$ → step $=UV^T$ → [Muon](#muon) w/o momentum.
```

&nbsp;

#### SOAP
- **Name**: ShampoO with Adam in the Preconditioner's eigenbasis {cite:p}`vyas2024soap`
- **What**: Adam run in the eigenbasis of Shampoo's $L$ & $R$.
- **Why**: Shampoo needs costly inverse roots, frequent recompute & extra hyperparams (damping, grafting).
    - Insight: Shampoo w/ power $\frac{1}{2}$ = Adafactor in Shampoo's eigenbasis.
    - → Run Adam there instead → the 2nd moment updates every step, even between eigenbasis refreshes.
- **How**:
    1. Track $L$, $R$ (EWMA) like Shampoo.
    2. Every $f$ steps: refresh eigenbases $Q_L$, $Q_R$.
    3. Each step: rotate the grad into the eigenbasis → Adam step there → rotate back.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $W_t\in\mathbb{R}^{p\times q}$: Weight matrix at step $t$.
- Hyperparams:
    - $\beta_1$: 1st-moment decay.
    - $\beta_2$: 2nd-moment & stats decay.
    - $\eta$: LR.
    - $\epsilon$: Small constant.
    - $f$: Preconditioning frequency (paper: 10).
- Misc:
    - $G_t\in\mathbb{R}^{p\times q}$: Grad matrix (as in Shampoo).
    - $Q_L\in\mathbb{R}^{p\times p}$: Eigenvectors of $L$.
    - $Q_R\in\mathbb{R}^{q\times q}$: Eigenvectors of $R$.
    - $M_t$: 1st moment, original space ($\Leftrightarrow$ Adam's $v_t$).
    - $V_t$: 2nd moment, rotated space ($\Leftrightarrow$ Adam's $G_t$).
    - $X'$: $X$ in the eigenbasis, $Q_L^TXQ_R$.

Process:
1. Rotate grad:

    $$
    G'_t=Q_L^TG_tQ_R
    $$

2. 1st moment (original space), then rotate:

    $$
    M_t=\beta_1M_{t-1}+(1-\beta_1)G_t,\qquad M'_t=Q_L^TM_tQ_R
    $$

3. 2nd moment (rotated space):

    $$
    V_t=\beta_2V_{t-1}+(1-\beta_2)G'_t\odot G'_t
    $$

4. Adam step in the eigenbasis, rotated back:

    $$
    W_t\leftarrow W_{t-1}-\eta\,Q_L\left(\frac{M'_t}{\sqrt{V_t}+\epsilon}\right)Q_R^T
    $$

5. Update stats:

    $$
    L_t=\beta_2L_{t-1}+(1-\beta_2)G_tG_t^T,\qquad R_t=\beta_2R_{t-1}+(1-\beta_2)G_t^TG_t
    $$

6. If $t\bmod f=0$: $Q_L\leftarrow\text{eigvecs}(L_t)$, $Q_R\leftarrow\text{eigvecs}(R_t)$.

- Bias correction omitted (as in the paper's pseudocode).
- Paper: eigvecs via 1 power-iteration step + QR, warm-started from the prev $Q$.
```

```{tip} Derivation
:class: dropdown
*Why is Shampoo$^{1/2}$ = Adafactor in the eigenbasis?* (idealized: $L=\mathbb{E}[GG^T]$, $R=\mathbb{E}[G^TG]$)
1. $L=Q_L\,\text{diag}(\lambda)\,Q_L^T$, $R=Q_R\,\text{diag}(\mu)\,Q_R^T$.
2. Shampoo w/ power $\frac{1}{2}$ & trace normalization: $\hat{H}=\frac{L\otimes R}{\text{tr}(L)}$ → scales rotated coordinate $(i,j)$ by $\left(\frac{\lambda_i\mu_j}{\sum_k\lambda_k}\right)^{-1/2}$.
3. Adafactor on $G'=Q_L^TGQ_R$: row sum $i$ of $\mathbb{E}[G'\odot G']$ $=\mathbb{E}\|\mathbf{u}_i^TG\|^2=\mathbf{u}_i^TL\mathbf{u}_i=\lambda_i$. Col sum $j$ $=\mu_j$ likewise.
    - $\mathbf{u}_i$: $i$-th column of $Q_L$.
4. Adafactor scale: $\left(\frac{\lambda_i\mu_j}{\sum_k\lambda_k}\right)^{-1/2}$ → identical to Step 2.
5. Adafactor's rank-1 2nd moment → swap for Adam's full $V$ → SOAP.
```

````{important} Code
:class: dropdown
```python
import torch

class SOAP(torch.optim.Optimizer):
    def __init__(self, params, lr=3e-3, betas=(0.95, 0.95), eps=1e-8, freq=10):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, freq=freq))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for W in group["params"]:
                if W.grad is None:
                    continue
                G, s = W.grad, self.state[W]
                assert W.ndim == 2, "matrix case only"
                if not s:
                    p, q = W.shape
                    s["t"], s["M"], s["V"] = 0, torch.zeros_like(W), torch.zeros_like(W)
                    s["L"] = torch.zeros(p, p, device=W.device)
                    s["R"] = torch.zeros(q, q, device=W.device)
                    ## identity basis until the 1st refresh -> plain Adam (simplification)
                    s["QL"] = torch.eye(p, device=W.device)
                    s["QR"] = torch.eye(q, device=W.device)
                s["t"] += 1
                QL, QR = s["QL"], s["QR"]
                G_rot = QL.T @ G @ QR                                   ## grad in eigenbasis
                s["M"].mul_(b1).add_(G, alpha=1 - b1)                   ## momentum in original space
                M_rot = QL.T @ s["M"] @ QR
                s["V"].mul_(b2).addcmul_(G_rot, G_rot, value=1 - b2)    ## Adam 2nd moment, rotated
                N = QL @ (M_rot / (s["V"].sqrt() + group["eps"])) @ QR.T  ## Adam step, rotated back
                W.sub_(group["lr"] * N)
                s["L"].mul_(b2).add_(G @ G.T, alpha=1 - b2)
                s["R"].mul_(b2).add_(G.T @ G, alpha=1 - b2)
                if s["t"] % group["freq"] == 0:
                    ## refresh basis (paper: 1 power-iteration step + QR; eigh for clarity)
                    s["QL"] = torch.linalg.eigh(s["L"]).eigenvectors
                    s["QR"] = torch.linalg.eigh(s["R"]).eigenvectors

## Example
W = torch.nn.Parameter(torch.randn(3, 5))
opt = SOAP([W], freq=1)
(W ** 2).sum().backward()
opt.step()
print(opt.state[W]["QL"].shape)  ## torch.Size([3, 3])
```
````

```{attention} Q&A
:class: dropdown
*Results?*
- 360M & 660M LMs, large batch: >40% fewer iterations & >35% less wall-clock vs AdamW; ≈20% better than Shampoo on both.

*Extra hyperparams vs Adam?*
- Only $f$.

*Why Adam, ❌Adafactor, in the eigenbasis?*
- Full per-coordinate $V$ → ⬆️fidelity vs a rank-1 estimate.
- $V$ updates every step → still adapts while $Q_L,Q_R$ are stale → tolerates a larger $f$.

*Why keep $M$ in the original space?*
- Basis changes every $f$ steps. $M$ in the original space stays valid → rotate it w/ the curr basis at each step.

*Cons?*
- Memory: $L,R,Q_L,Q_R$ ($2p^2+2q^2$) + $M,V$ ($2pq$) → more than Adam.
- Periodic eigendecomposition → extra compute & engineering.
```

&nbsp;

### Muon
- **Name**: MomentUm Orthogonalized by Newton-Schulz {cite:p}`jordan2024muon`
- **What**: SGD-momentum whose update matrix is replaced by its nearest semi-orthogonal matrix $\approx UV^T$.
- **Why**: Momentum & Adam updates of 2D weights have a very high condition number → nearly low-rank → a few directions dominate.
    - Orthogonalize → every singular direction gets the same step size → rare directions amplified. ⚠️ Authors' empirical speculation.
    - Theory link: = steepest descent under the spectral norm; = Shampoo w/o accumulation.
- **How**:
    1. Nesterov momentum on the grad matrix.
    2. Orthogonalize: 5 Newton–Schulz iterations ≈ $UV^T$. Only matmuls → runs stably in bf16.
    3. Shape-dependent scale → update params.
    4. Apply **only to hidden 2D weights** (& conv filters flattened to 2D).
        - AdamW for: embeddings, output head, biases, norm gains.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $W_t\in\mathbb{R}^{p\times q}$: Hidden weight matrix at step $t$.
- Hyperparams:
    - $\mu$: Momentum. Default 0.95.
    - $\eta$: LR. Default 0.02 (reference implementation).
    - $\lambda$: Decoupled weight decay.
    - $(a,b,c)=(3.4445,-4.7750,2.0315)$: Newton–Schulz coefficients.
    - $K$: #Newton–Schulz steps. Default 5.
- Misc:
    - $G_t$: Grad matrix (as in Shampoo).
    - $B_t$: Momentum buffer.
    - $X_k$: Newton–Schulz iterate.
    - $O_t$: Orthogonalized update.
    - $s$: Shape scale.
    - $\text{Ortho}(X)=UV^T$, where $X=USV^T$ is the SVD.

Process:
1. Momentum & Nesterov look-ahead:

    $$
    B_t=\mu B_{t-1}+G_t,\qquad \tilde{B}_t=G_t+\mu B_t
    $$

2. Normalize → all singular values in $[0,1]$:

    $$
    X_0=\frac{\tilde{B}_t}{\|\tilde{B}_t\|_F}
    $$

3. Newton–Schulz, $k=0,\dots,K-1$:

    $$
    X_{k+1}=aX_k+b\left(X_kX_k^T\right)X_k+c\left(X_kX_k^T\right)^2X_k
    $$

4. $O_t=X_K\approx\text{Ortho}(\tilde{B}_t)$.
5. Update params:

    $$
    W_t\leftarrow W_{t-1}-\eta\left(s\,O_t+\lambda W_{t-1}\right)
    $$

- $s=\sqrt{\max(1,p/q)}$ (reference implementation).
- $s=0.2\sqrt{\max(p,q)}$ (Moonlight) → update RMS ≈ AdamW's → share $\eta,\lambda$ w/ AdamW params.
- Tuned $(a,b,c)$ → singular values end near 1 (≈ $[0.7,1.3]$), ❌exactly 1. Harmless empirically.
```

```{tip} Derivation
:class: dropdown
*Why does Newton–Schulz orthogonalize?*
1. $X=USV^T$ → $XX^T=US^2U^T$.
2. One step:

    $$
    aX+b(XX^T)X+c(XX^T)^2X=U\left(aS+bS^3+cS^5\right)V^T
    $$

3. → Acts on singular values only: $\sigma\mapsto\varphi(\sigma)=a\sigma+b\sigma^3+c\sigma^5$. $U,V$ unchanged.
4. Frobenius normalization → $\sigma\in[0,1]$. Coefficients chosen so $\varphi^K(\sigma)\approx1$ for all $\sigma\in(0,1]$.
    - $\varphi'(0)=a$ large → small $\sigma$ grow fast → few steps.
5. → $X_K\approx UV^T$.

*Why = Shampoo w/o accumulation?*
1. Shampoo w/ stats = curr $GG^T$ & $G^TG$ only, $\epsilon\to0$:

    $$
    (GG^T)^{-1/4}G(G^TG)^{-1/4}=\left(US^{-1/2}U^T\right)\left(USV^T\right)\left(VS^{-1/2}V^T\right)=UV^T
    $$

2. Rank-deficient $G$: read the inverse roots as pseudo-inverses (on the range of $G$).

*Why = steepest descent under the spectral norm?*
1. Solve $\min_{\|\Delta\|_2\le\eta}\langle G,\Delta\rangle$.
2. Dual norm of spectral = nuclear → $\langle G,\Delta\rangle\ge-\|G\|_*\|\Delta\|_2\ge-\eta\|G\|_*$.
3. $\Delta=-\eta UV^T$: $\langle USV^T,-\eta UV^T\rangle=-\eta\,\text{tr}(S)=-\eta\|G\|_*$ → bound attained.
```

````{important} Code
:class: dropdown
```python
import torch

class Muon(torch.optim.Optimizer):
    ## for hidden 2D weights ONLY; use AdamW for embeddings, head, biases, gains
    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.0, ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                      weight_decay=weight_decay, ns_steps=ns_steps))

    @staticmethod
    def newton_schulz(G, steps):
        a, b, c = 3.4445, -4.7750, 2.0315
        X = G / (G.norm() + 1e-7)       ## Frobenius-normalize -> singular values <= 1
        tall = X.size(0) > X.size(1)
        if tall:
            X = X.T                     ## make X X^T the smaller Gram matrix
        for _ in range(steps):
            A = X @ X.T
            X = a * X + (b * A + c * A @ A) @ X   ## odd polynomial of singular values
        return X.T if tall else X

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, mu, wd = group["lr"], group["momentum"], group["weight_decay"]
            for W in group["params"]:
                if W.grad is None:
                    continue
                G, s = W.grad, self.state[W]
                assert W.ndim == 2, "Muon is for 2D hidden weights"
                if not s:
                    s["B"] = torch.zeros_like(W)
                s["B"].mul_(mu).add_(G)
                B_nes = G + mu * s["B"]                          ## Nesterov look-ahead
                O = self.newton_schulz(B_nes, group["ns_steps"])  ## ≈ Ortho(B_nes)
                scale = max(1.0, W.size(0) / W.size(1)) ** 0.5
                W.mul_(1 - lr * wd)                          ## decoupled WD
                W.sub_(lr * scale * O)

## Example
W = torch.nn.Parameter(torch.randn(4, 3))
opt = Muon([W])
(W ** 2).sum().backward()
opt.step()
print(torch.linalg.svdvals(Muon.newton_schulz(torch.randn(4, 3), 5)))  ## all roughly 1
```
````

```{attention} Q&A
:class: dropdown
*Which params get Muon?*
- Hidden 2D weights: attention projections, MLP matrices. Conv filters: flatten the last 3 dims.
- AdamW: embeddings & output head (2D, but empirically better w/ AdamW), biases, norm gains, scalars.
- Apply to $W_Q,W_K,W_V$ separately → better than to a fused QKV matrix.

*Cost?*
- Memory = SGD-momentum (1 buffer).
- FLOP overhead ≤ $\frac{Kd}{B}$ ($d$: model dim, $B$: tokens per batch) → <1% for typical LM training.

*Why Newton–Schulz, ❌SVD?*
- SVD: too slow on GPU.
- Coupled Newton iteration (used in Shampoo): needs fp32.
- Newton–Schulz: matmuls only, stable in bf16.

*Why doesn't the momentum convention matter?*
- $\text{Ortho}(cX)=\text{Ortho}(X)$ & Frobenius normalization → overall scale of $B_t$ is irrelevant.
- → Sum ($B_t=\mu B_{t-1}+G_t$) vs EWMA ($(1-\mu)$ factor) give the same update.

*Results?*
- NanoGPT speedrun: 1.35× faster training record vs AdamW.
- Moonlight: ~2× compute efficiency vs AdamW in compute-optimal scaling-law runs. {cite:p}`liu2025muon`
    - Needed at scale: decoupled WD (else weights & layer-output RMS keep growing) & update-RMS matching ($s=0.2\sqrt{\max(p,q)}$).

*Cons?*
- 2 optimizers (Muon + AdamW) → 2 sets of hyperparams, unless RMS-matched.
- Needs the full matrix per param → sharded training (ZeRO/FSDP) must gather it before Newton–Schulz.
- At trillion scale: attention-logit explosions → QK-Clip ([Kimi](../../nlp/models.md#kimi)).
```

&nbsp;

## SAM
- **Name**: Sharpness-Aware Minimization {cite:p}`foret2021sharpness`
- **What**: Minimize the worst-case loss in an $L^2$ ball of radius $\rho$ around the params.
- **Why**: Flat minima tend to generalize better than sharp ones ([Flat Minima](../theory.md#flat-minima)).
    - Plain loss minimization ignores sharpness → can land in sharp minima.
    - ⚠️ Flatness ↔ generalization is an empirical link. Sharpness is not reparameterization-invariant.
- **How**:
    1. Ascent: grad at $w$ → step $\rho$ along the normalized grad → $w+\hat{\delta}$ ≈ worst point in the ball.
    2. Descent: grad at $w+\hat{\delta}$ → apply it to the ORIGINAL $w$ via any base optimizer.
    3. Cost: 2 forward-backward passes per step.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\rho$: Neighborhood radius. Paper: 0.05 is a solid default.
    - $\eta$: LR of the base optimizer.
- Misc:
    - $g_t$: Grad $\frac{\partial\mathcal{L}}{\partial w_{t-1}}$.
    - $\delta$: Weight perturbation (paper's $\epsilon$).
    - $\hat{\delta}_t$: Approx worst-case perturbation.
    - $g^\text{SAM}_t$: SAM grad.

Objective:

$$
\min_w\ \max_{\|\delta\|_2\le\rho}\mathcal{L}(w+\delta)
$$

- Paper adds a standard $L^2$ penalty; omitted here.

Process:
1. Ascent direction:

    $$
    \hat{\delta}_t=\rho\frac{g_t}{\|g_t\|_2}
    $$

2. SAM grad (Hessian term dropped):

    $$
    g^\text{SAM}_t=\nabla_w\mathcal{L}(w)\Big|_{w=w_{t-1}+\hat{\delta}_t}
    $$

3. Base optimizer step from the original params, e.g., SGD:

    $$
    w_t\leftarrow w_{t-1}-\eta\,g^\text{SAM}_t
    $$
```

```{tip} Derivation
:class: dropdown
*Where do $\hat{\delta}$ & $g^\text{SAM}$ come from?*
1. 1st-order Taylor: $\mathcal{L}(w+\delta)\approx\mathcal{L}(w)+\delta^Tg$.
2. $\max_{\|\delta\|_2\le\rho}\delta^Tg$ → $\hat{\delta}=\rho\frac{g}{\|g\|_2}$ ← Cauchy–Schwarz. General $L^p$ ball: dual-norm solution.
3. Exact grad of $\mathcal{L}(w+\hat{\delta}(w))$ (chain rule):

    $$
    \left(I+\frac{\partial\hat{\delta}}{\partial w}\right)^T\nabla\mathcal{L}\Big|_{w+\hat{\delta}}
    $$

4. $\frac{\partial\hat{\delta}}{\partial w}$ contains the Hessian → drop it → $g^\text{SAM}=\nabla\mathcal{L}|_{w+\hat{\delta}}$.
5. Paper: keeping the Hessian term degraded performance in an initial test.
```

````{important} Code
:class: dropdown
```python
import torch

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer_cls, rho=0.05, **kwargs):
        super().__init__(params, dict(rho=rho, **kwargs))
        self.rho = rho
        self.base = base_optimizer_cls(self.param_groups, **kwargs)  ## shares param groups

    @torch.no_grad()
    def step(self, closure):
        ## closure: zero grads -> forward -> backward -> return loss
        with torch.enable_grad():
            closure()                                              ## pass 1: g at w
        ps = [p for grp in self.param_groups for p in grp["params"] if p.grad is not None]
        g_norm = torch.stack([p.grad.norm() for p in ps]).norm()  ## global L2 norm
        deltas = []
        for p in ps:
            d = self.rho * p.grad / (g_norm + 1e-12)               ## ascent: rho * g / ||g||
            p.add_(d)
            deltas.append(d)
        with torch.enable_grad():
            loss = closure()                                       ## pass 2: g_SAM at w + delta
        for p, d in zip(ps, deltas):
            p.sub_(d)                                              ## back to the original w
        self.base.step()                                           ## descend from w w/ g_SAM
        return loss

## Example
w = torch.nn.Parameter(torch.randn(2, 4))
opt = SAM([w], torch.optim.SGD, rho=0.05, lr=0.1)

def closure():
    opt.zero_grad()
    loss = (w ** 2).sum()
    loss.backward()
    return loss

print(opt.step(closure))  ## loss at the perturbed point w + delta
```
````

```{attention} Q&A
:class: dropdown
*Cost?*
- 2× forward-backward per step.

*Why normalize the grad?*
- Ball constraint → only the direction matters. Step length is fixed at $\rho$.

*Which base optimizer?*
- Any (SGD, AdamW). SAM changes only WHERE the grad is evaluated.

*What is $m$-sharpness?*
- Data-parallel SAM: each device computes its own $\hat{\delta}$ on its shard of $m$ samples.
- Paper: smaller $m$ → better generalization → per-device perturbation is a feature, ❌a bug.

*Relation to adversarial training?*
- Same min-max structure. Adversary perturbs the **weights**, ❌the inputs.

*Cons?*
- 2× compute per step.
- $\rho$ is a new hyperparam.
- Sharpness changes under reparameterization (e.g., rescaling layers in a ReLU net) → the flatness rationale is not airtight.
```

&nbsp;

## Schedule-Free
- **What**: Replace momentum & the LR schedule w/ interpolation between a base iterate & its running average. {cite:p}`defazio2024the`
- **Why**: Best schedules (cosine, linear decay) need the stopping step $T$ in advance → ❌stop early, ❌extend for free.
    - Classical iterate averaging (Polyak) needs no $T$ → but performs much worse than schedules in practice.
    - → Find an averaging scheme that matches schedules w/o knowing $T$.
- **How**: Keep 3 sequences.
    1. $z$: base sequence. Plain SGD/Adam steps at a constant LR.
    2. $w$: uniform running average of $z$ → evaluate & ship this.
    3. $\tilde{w}$: interpolation of $z$ & $w$ → compute the grad HERE.
    - $\beta=0$ → Polyak averaging. $\beta=1$ → primal averaging. $\beta=0.9$ works in between.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $w_t$: Averaged params, used for eval (paper's $x$).
    - $z_t$: Base iterate (paper's $z$).
    - $\tilde{w}_t$: Grad-evaluation point (paper's $y$).
- Hyperparams:
    - $\eta$: Constant LR.
    - $\beta$: Interpolation coeff. Default 0.9.
- Misc:
    - $c_{t+1}$: Averaging weight.

Process (SGD):
1. Grad point:

    $$
    \tilde{w}_t=(1-\beta)z_t+\beta w_t
    $$

2. Base step:

    $$
    z_{t+1}=z_t-\eta\nabla\mathcal{L}(\tilde{w}_t)
    $$

3. Running average:

    $$
    w_{t+1}=(1-c_{t+1})w_t+c_{t+1}z_{t+1},\qquad c_{t+1}=\frac{1}{t+1}
    $$

- Init $z_1=w_1$ → $w_t$ = equal-weight mean of $z_1,\dots,z_t$.

AdamW variant (paper's Algorithm 1):

$$\begin{align*}
g_t&=\nabla\mathcal{L}(\tilde{w}_t),\qquad G_t=\beta_2G_{t-1}+(1-\beta_2)g_t\odot g_t\\
\eta_t&=\eta\sqrt{1-\beta_2^t}\,\min\left(1,\frac{t}{T_\text{warmup}}\right)\\
z_{t+1}&=z_t-\eta_t\frac{g_t}{\sqrt{G_t}+\epsilon}-\eta_t\lambda\tilde{w}_t\\
c_{t+1}&=\frac{\eta_t^2}{\sum_{i=1}^t\eta_i^2}
\end{align*}$$

- $G_t$: 2nd moment (as in Adam).
- $\lambda$: Weight decay, applied at $\tilde{w}_t$.
- $T_\text{warmup}$: #Warmup steps.
- $\sqrt{1-\beta_2^t}$: Adam's 2nd-moment bias correction, fused into the LR.
- $c_{t+1}\propto\eta_t^2$ → warmup-phase iterates get less weight. After warmup, $c_{t+1}$ decays like $\frac{1}{t}$.
```

```{tip} Derivation
:class: dropdown
*Why can an implementation store only $z$?*
1. Param buffer holds $\tilde{w}_t$; state holds $z_t$. Recover $w_t=\frac{\tilde{w}_t-(1-\beta)z_t}{\beta}$ ← Step 1.
2. Substitute Steps 2–3 into $\tilde{w}_{t+1}=(1-\beta)z_{t+1}+\beta w_{t+1}$:

    $$
    \tilde{w}_{t+1}=(1-c_{t+1})\tilde{w}_t+c_{t+1}z_t+\eta\left(\beta(1-c_{t+1})-1\right)g_t
    $$

    - $g_t=\nabla\mathcal{L}(\tilde{w}_t)$.
    - $z_t$ coefficient: $(1-\beta+\beta c_{t+1})-(1-c_{t+1})(1-\beta)=c_{t+1}$.
3. → Update $\tilde{w}$ & $z$ in place → 2 buffers ($\tilde{w}$, $z$) = SGD-momentum memory.
4. Eval: overwrite the buffer w/ $w_t$ (Step 1). Back to training: $\tilde{w}_t=(1-\beta)z_t+\beta w_t$. Both maps are exact inverses.
```

````{important} Code
:class: dropdown
```python
import torch

class ScheduleFreeSGD(torch.optim.Optimizer):
    ## param buffer holds w_tilde (train mode) or w (eval mode); state holds only z
    def __init__(self, params, lr=0.1, beta=0.9):
        super().__init__(params, dict(lr=lr, beta=beta))
        self.training = True

    @torch.no_grad()
    def step(self):
        assert self.training, "call opt.train() before stepping"
        for group in self.param_groups:
            lr, beta = group["lr"], group["beta"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                s = self.state[p]
                if not s:
                    s["t"], s["z"] = 0, p.clone()  ## z_1 = w_1 = w_tilde_1
                s["t"] += 1
                c, z, g = 1.0 / (s["t"] + 1), s["z"], p.grad
                ## w_tilde_{t+1} = lerp(w_tilde_t, z_t, c) + lr*(beta*(1-c) - 1)*g  (uses old z)
                p.lerp_(z, c).add_(g, alpha=lr * (beta * (1 - c) - 1))
                z.sub_(g, alpha=lr)  ## base SGD step; grad was taken at w_tilde

    @torch.no_grad()
    def _swap(self, to_eval):
        for group in self.param_groups:
            beta = group["beta"]
            for p in group["params"]:
                if not self.state[p]:
                    continue
                z = self.state[p]["z"]
                if to_eval:
                    p.sub_((1 - beta) * z).div_(beta)  ## w = (w_tilde - (1-beta) z) / beta
                else:
                    p.mul_(beta).add_((1 - beta) * z)  ## w_tilde = (1-beta) z + beta w

    def eval(self):
        if self.training:
            self._swap(to_eval=True)
            self.training = False

    def train(self):
        if not self.training:
            self._swap(to_eval=False)
            self.training = True

## Example
p = torch.nn.Parameter(torch.randn(2, 4))
opt = ScheduleFreeSGD([p])
for _ in range(3):
    opt.zero_grad()
    (p ** 2).sum().backward()
    opt.step()
opt.eval()   ## buffer now holds the average w -> evaluate / ship
opt.train()  ## exact inverse -> resume training at w_tilde
print(p.shape)  ## torch.Size([2, 4])
```
````

```{attention} Q&A
:class: dropdown
*Why does it work?*
- Theory: convex Lipschitz losses, $\eta=\frac{D}{G\sqrt{T}}$ → any $\beta\in[0,1]$ gives the worst-case optimal rate $\frac{DG}{\sqrt{T}}$ for the average $w_T$.
    - $D$: initial distance to the optimum.
    - $G$: Lipschitz constant.
    - The theory's constant $\eta$ still uses $T$. Practice: tune $\eta$; ❌$T$ in the update rule.
- Averaging & LR schedules play the same role: both damp the noise of late iterates. Averaging does it w/o knowing $T$.

*How is it "momentum"?*
- Curr grad enters $\tilde{w}$ immediately w/ weight $1-\beta$ (like EWMA momentum).
- The rest enters slowly via the average $w$ → longer delay than EWMA momentum.

*Results?*
- Matches or beats tuned cosine schedules on 28 problems (convex to large-scale DL).
- Schedule-Free AdamW won the MLCommons AlgoPerf 2024 self-tuning track.

*Still need warmup?*
- Yes. Paper uses linear warmup + $\eta_t^2$-weighted averaging.

*Memory?*
- SGD version: same as SGD-momentum.

*Gotchas?*
- Eval at $w$, ❌at the grad point $\tilde{w}$ → swap params before eval.
- BatchNorm: running stats are collected at $\tilde{w}$ → recompute them at $w$ before eval.

*vs WSD & weight averaging?*
- [WSD](schedule.md#wsd): keeps a schedule but defers the decay to the end. Schedule-Free: no decay at all.
- [EMA](train.md#ema) / [SWA](train.md#swa): average post hoc; the grad is still taken at the raw iterate. Schedule-Free: the grad point depends on the average → coupled.
```

&nbsp;
