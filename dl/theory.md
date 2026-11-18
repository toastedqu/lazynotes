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
# Theory
Why over-parameterized deep nets train & generalize: empirical phenomena & the partial theory behind them. Loss vs scale → [Scaling Laws](../nlp/models.md#scaling-laws).

Notations:
- $\theta\in\mathbb{R}^p$: All params, flattened
- $p$: #params
- $\theta_0$: Init params
- $\theta_t$: Params after $t$ steps
- $f_\theta(x)$: Network output
- $\mathcal{L}(\theta)$: Training loss
- $\eta$: LR
- $B$: Batch size
- $H=\nabla_\theta^2\mathcal{L}(\theta)$: Loss Hessian
- $\lambda_\text{max}$: Largest eigenvalue of $H$ (**sharpness**)

Overrides of global notation: $m$ = #training samples, ❌#samples per batch (batch size = $B$). $B$ = scalar batch size, ❌matrix. $\theta$ = vector, non-bold.

&nbsp;

## Generalization

### Double Descent
- **What**: Test error vs capacity: falls → peaks at the interpolation threshold → falls again. {cite:p}`belkin2019reconciling`
    - **Interpolation threshold**: smallest capacity that fits the training set exactly. Linear-in-params models: $p\approx m$.
- **Why**: Reconcile 2 facts.
    - Classical [bias-variance](../ml/misc.md#bias-variance-tradeoff) U-curve → a model that interpolates the data = overfit.
    - Practice: nets w/ $p\gg m$ interpolate & still generalize.
- **How**:
    1. **Under-param** ($p<m$): classical U-curve.
    2. **Threshold** ($p\approx m$): ~1 model fits all labels, noise included → huge-norm, wiggly fit → test error peaks.
    3. **Over-param** ($p>m$): many interpolants → the optimizer's [implicit bias](#implicit-bias-of-sgd) picks a low-norm, smooth one → test error falls again.
    - **Deep double descent**: the same peak shows on 3 axes. {cite:p}`nakkiran2020deep`
        - **Model-wise**: width ⬆️ at a fixed training budget.
        - **Epoch-wise**: training time ⬆️ at a fixed large model → test error falls, rises, falls.
        - **Sample-wise**: $m$ ⬆️ → threshold moves right → near the threshold, more data hurts.
    - Label noise ⬆️ → peak ⬆️.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\mathcal{T}$: Training procedure (arch + optimizer + budget). Maps a training set to a model.
    - $\mathcal{D}$: Data distribution.
    - $\tau$: Train-error tolerance.

Effective model complexity (EMC):

$$
\text{EMC}_{\mathcal{D},\tau}(\mathcal{T})=\max\left\{m\ \middle|\ \mathbb{E}_{S\sim\mathcal{D}^m}\left[\text{Err}_S(\mathcal{T}(S))\right]\le\tau\right\}
$$

- $\text{Err}_S$: Train error on $S$.
- = largest training set that $\mathcal{T}$ fits to ~0 train error.

Generalized double descent (conjecture):
- $\text{EMC}\ll m$ (under-param): EMC ⬆️ → test error ⬇️.
- $\text{EMC}\gg m$ (over-param): EMC ⬆️ → test error ⬇️.
- $\text{EMC}\approx m$ (critical): EMC ⬆️ → test error ⬆️ or ⬇️.

Linear toy model: rows of $X$ i.i.d. $\mathcal{N}(0,I_p)$, $\mathbf{y}=X\boldsymbol\beta+\boldsymbol\varepsilon$, $\mathbb{E}[\boldsymbol\varepsilon]=0$, $\text{Cov}[\boldsymbol\varepsilon]=\sigma^2I_m$, $\boldsymbol\varepsilon$ independent of $X$. Min-norm least squares $\hat{\boldsymbol\beta}=X^+\mathbf{y}$:

$$
\mathbb{E}\|\hat{\boldsymbol\beta}-\boldsymbol\beta\|^2=\begin{cases}
\sigma^2\dfrac{p}{m-p-1}, & p\le m-2\\[3mm]
\|\boldsymbol\beta\|^2\left(1-\dfrac{m}{p}\right)+\sigma^2\dfrac{m}{p-m-1}, & p\ge m+2
\end{cases}
$$

- $X\in\mathbb{R}^{m\times p}$: Design matrix. $X^+$: pseudo-inverse.
- $\boldsymbol\beta\in\mathbb{R}^p$: True weights.
- $\boldsymbol\varepsilon\in\mathbb{R}^m$: Label noise.
- LHS = excess test MSE ← isotropic test input.
- Noise terms → $\infty$ as $p\to m$ → the peak.
- $p\gg m$: noise term → 0. Bias term → $\|\boldsymbol\beta\|^2$.
- Caveat: this isotropic toy shows the peak & the second descent of the noise term. A second descent below the classical optimum needs more structure (e.g., many weak features, decaying feature spectrum).
```

```{tip} Derivation
:class: dropdown
*Why does the test error blow up at $p\approx m$?*
1. $\mathbf{y}=X\boldsymbol\beta+\boldsymbol\varepsilon$. Test input isotropic → excess risk $=\mathbb{E}\|\hat{\boldsymbol\beta}-\boldsymbol\beta\|^2$.
2. **$p<m$**: $X$ full column rank → $\hat{\boldsymbol\beta}=(X^TX)^{-1}X^T\mathbf{y}$ → $\hat{\boldsymbol\beta}-\boldsymbol\beta=(X^TX)^{-1}X^T\boldsymbol\varepsilon$.
3. $\mathbb{E}\|\cdot\|^2=\sigma^2\,\mathbb{E}\,\text{tr}\left[(X^TX)^{-1}\right]$.
4. $X^TX\sim\text{Wishart}_p(m,I)$ → $\mathbb{E}\left[(X^TX)^{-1}\right]=\frac{I_p}{m-p-1}$ → $\sigma^2\frac{p}{m-p-1}$.
5. **$p>m$**: $\hat{\boldsymbol\beta}=X^T(XX^T)^{-1}\mathbf{y}=\Pi\boldsymbol\beta+X^T(XX^T)^{-1}\boldsymbol\varepsilon$.
    - $\Pi$: projector onto the row space of $X$.
6. Noise part: $\mathbb{E}\|X^T(XX^T)^{-1}\boldsymbol\varepsilon\|^2=\sigma^2\,\mathbb{E}\,\text{tr}\left[(XX^T)^{-1}\right]$. $XX^T\sim\text{Wishart}_m(p,I)$ → $\sigma^2\frac{m}{p-m-1}$.
7. Bias part: $\|(I-\Pi)\boldsymbol\beta\|^2$. Gaussian $X$ → row space = uniformly random $m$-dim subspace → $\mathbb{E}=\|\boldsymbol\beta\|^2(1-\frac{m}{p})$.
8. Cross terms vanish ← $\boldsymbol\varepsilon$ zero-mean, independent of $X$.
9. Intuition: $p\to m$ → $X$ nearly square → smallest singular value → 0 → $X^+$ amplifies label noise.
```

```{attention} Q&A
:class: dropdown
*Does double descent refute the bias-variance tradeoff?*
- ❌. The decomposition still holds.
- What fails: "#params = capacity". The variance term is non-monotone in #params & peaks at the threshold.

*Why the peak exactly at the interpolation threshold?*
- Only ~1 model fits the data → it must fit the noise → no freedom to pick a smooth solution.
- Linear case: $X$ ill-conditioned → $X^+$ amplifies noise.

*What shrinks or removes the peak?*
- Tuned $\ell_2$ regularization. Certain isotropic linear regression models: optimal ridge → test risk monotone in $p$ & $m$ (proven).
- [Early stopping](train/reg.md#early-stopping) → removes the epoch-wise peak.
- Less label noise → smaller peak.

*How can more data hurt?*
- $m$ ⬆️ → threshold moves to larger models.
- A model that was over-param for the small set becomes critical for the larger set → test error ⬆️.

*Is it specific to NNs?*
- ❌. Also in linear regression, random features, random forests, boosting.

*Practical takeaway?*
- Do not choose model size from the classical U-curve alone.
- Near the threshold: go bigger, or regularize / stop early.
- LLM pretraining: ~1 epoch, ❌interpolation → far from the threshold → smooth [Scaling Laws](../nlp/models.md#scaling-laws) instead.
```

&nbsp;

### Implicit Bias of SGD
- **What**: The optimizer's preference among the many zero-loss solutions of an over-param model. {cite:p}`soudry2018implicit`
- **Why**: Over-param → infinitely many global minima. Most generalize badly.
    - Nets generalize w/o explicit regularization → the optimizer must pick a good minimum.
- **How**:
    - **Least squares, zero init**: GD/SGD never leaves the span of the inputs → converges to the **min-$\ell_2$-norm** interpolant.
    - **Separable logistic / CE**: $\|\mathbf{w}\|\to\infty$. Direction → **hard-margin SVM** direction. Only logarithmically fast.
    - **SGD noise**: covariance ∝ $\eta/B$ → $\eta/B$ ⬆️ → wider minima. {cite:p}`jastrzebski2018three`
    - **Finite $\eta$**: GD ≈ gradient flow on a modified loss that penalizes $\|\nabla\mathcal{L}\|^2$ → prefers flat regions.
    - Caveat: exact results only for linear & simplified models. Deep nets: partial, still open.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $X\in\mathbb{R}^{m\times n}$: Design matrix, rows $\mathbf{x}_i^T$.
    - $\mathbf{y}\in\mathbb{R}^m$: Regression targets.
    - $y_i\in\{-1,+1\}$: Binary labels (classification).
- Params:
    - $\mathbf{w}\in\mathbb{R}^n$: Linear weights.
- Misc:
    - $\Sigma(\theta)$: Covariance of per-sample gradients.

Least squares, $n>m$, $\text{rank}(X)=m$, $\mathbf{w}_0=0$, $\eta<2/\lambda_\text{max}(X^TX)$:

$$
\mathcal{L}(\mathbf{w})=\frac{1}{2}\|X\mathbf{w}-\mathbf{y}\|^2\ \Rightarrow\ \mathbf{w}_\infty=X^T(XX^T)^{-1}\mathbf{y}=\arg\min_{X\mathbf{w}=\mathbf{y}}\|\mathbf{w}\|_2
$$

- Init $\mathbf{w}_0\neq0$ → $\mathbf{w}_\infty=\arg\min_{X\mathbf{w}=\mathbf{y}}\|\mathbf{w}-\mathbf{w}_0\|_2$.

Separable logistic regression:

$$
\mathcal{L}(\mathbf{w})=\sum_{i=1}^m\log\left(1+e^{-y_i\mathbf{w}^T\mathbf{x}_i}\right)\ \Rightarrow\ \frac{\mathbf{w}_t}{\|\mathbf{w}_t\|}\to\frac{\hat{\mathbf{w}}}{\|\hat{\mathbf{w}}\|},\quad\hat{\mathbf{w}}=\arg\min_\mathbf{w}\|\mathbf{w}\|_2\ \text{s.t.}\ y_i\mathbf{w}^T\mathbf{x}_i\ge1\ \forall i
$$

- $\|\mathbf{w}_t\|$ grows like $\log t$.
- Holds for any loss w/ an exponential tail. Multiclass CE: analogous multiclass SVM.

SGD as noisy GD:

$$
\theta_{t+1}=\theta_t-\eta\nabla\mathcal{L}(\theta_t)-\eta\,\xi_t,\qquad\mathbb{E}[\xi_t]=0,\quad\text{Cov}[\xi_t]\approx\frac{\Sigma(\theta_t)}{B}
$$

SDE approximation, time $s=t\eta$:

$$
d\theta=-\nabla\mathcal{L}(\theta)\,ds+\sqrt{\frac{\eta}{B}\Sigma(\theta)}\,dW_s
$$

- $W_s$: Standard Wiener process.
- Noise "temperature" ∝ $\eta/B$.

Finite step size (backward error analysis):

$$
\tilde{\mathcal{L}}(\theta)=\mathcal{L}(\theta)+\frac{\eta}{4}\|\nabla\mathcal{L}(\theta)\|^2
$$

- Gradient flow on $\tilde{\mathcal{L}}$ for time $\eta$ matches 1 GD step up to $O(\eta^3)$.
```

````{tip} Derivation
:class: dropdown
*Why does GD from 0 find the min-norm interpolant?*
1. $\nabla\mathcal{L}=X^T(X\mathbf{w}-\mathbf{y})\in\text{row}(X)=\text{span}\{\mathbf{x}_i\}$.
2. $\mathbf{w}_0=0\in\text{row}(X)$ → every $\mathbf{w}_t\in\text{row}(X)$.
3. Convex loss, small $\eta$, $\text{rank}(X)=m$ → $X\mathbf{w}_\infty=\mathbf{y}$.
4. Only 1 interpolant lies in $\text{row}(X)$: $\mathbf{w}=X^T\boldsymbol\alpha$, $XX^T\boldsymbol\alpha=\mathbf{y}$ → $\mathbf{w}=X^T(XX^T)^{-1}\mathbf{y}$.
5. Any other interpolant: $\mathbf{w}'=\mathbf{w}_\infty+\mathbf{v}$, $\mathbf{v}\in\text{null}(X)\perp\text{row}(X)$ → $\|\mathbf{w}'\|^2=\|\mathbf{w}_\infty\|^2+\|\mathbf{v}\|^2$.
6. SGD: each per-sample grad ∝ $\mathbf{x}_i$ → same argument.

*Why does logistic GD drift toward max-margin?*
1. Separable → loss → 0 only as $\|\mathbf{w}\|\to\infty$.
2. Large margins $u_i=y_i\mathbf{w}^T\mathbf{x}_i$: $\nabla\mathcal{L}\approx-\sum_ie^{-u_i}y_i\mathbf{x}_i$.
3. $e^{-u_i}$ → samples w/ the smallest margin dominate exponentially → gradient ∈ cone of the support vectors.
4. → $\mathbf{w}_t$ direction → $\sum_i\alpha_iy_i\mathbf{x}_i$ over support vectors, $\alpha_i\ge0$ = KKT conditions of the hard-margin [SVM](../ml/supervised.md#svm).

*Why does only $\eta/B$ matter?*
1. Mini-batch grad $\mathbf{g}_B=\frac{1}{B}\sum_{i\in\text{batch}}\nabla\ell_i$. i.i.d. samples → $\text{Cov}[\mathbf{g}_B]=\Sigma/B$.
2. Per-step noise: $\eta\xi_t$ → covariance $\eta^2\Sigma/B$.
3. SDE w/ time step $ds=\eta$: diffusion variance per step $=\frac{\eta}{B}\Sigma\cdot\eta$ ✅ matches.
4. Drift & diffusion depend on $(\eta,B)$ only via time scale $\eta$ & temperature $\eta/B$ → [Linear Scaling Rule](train/train.md#linear-scaling-rule).
5. Breaks for large $\eta$ (SDE invalid) or $B$ above the critical batch size.

*Where does $\frac{\eta}{4}\|\nabla\mathcal{L}\|^2$ come from?*
1. Seek a flow $\dot\theta=-\nabla\mathcal{L}+\eta\,\mathbf{c}(\theta)$ whose solution at time $\eta$ equals 1 GD step up to $O(\eta^3)$.
2. Taylor: $\theta(\eta)=\theta+\eta\dot\theta+\frac{\eta^2}{2}\ddot\theta+O(\eta^3)$, w/ $\ddot\theta=-H\dot\theta=H\nabla\mathcal{L}+O(\eta)$.
3. → $\theta(\eta)=\theta-\eta\nabla\mathcal{L}+\eta^2\left(\mathbf{c}+\frac{1}{2}H\nabla\mathcal{L}\right)+O(\eta^3)$.
4. Match GD $\theta-\eta\nabla\mathcal{L}$ → $\mathbf{c}=-\frac{1}{2}H\nabla\mathcal{L}=-\frac{1}{4}\nabla\|\nabla\mathcal{L}\|^2$.
5. → $\dot\theta=-\nabla\left(\mathcal{L}+\frac{\eta}{4}\|\nabla\mathcal{L}\|^2\right)$.
````

```{attention} Q&A
:class: dropdown
*Is implicit bias the full reason nets generalize?*
- ❌ proven. Exact characterizations exist only for linear models & simplified settings.
- Matrix factorization: GD from small init → low-rank solutions. Exact "min nuclear norm" characterization: disputed.
- Homogeneous nets (e.g., ReLU w/o bias) on separable data: direction → KKT point of a max-margin problem. ❌guaranteed global max-margin.

*Does the optimizer change the bias?*
- ✅. Steepest descent w.r.t. a norm $\|\cdot\|$ on separable data → max-margin w.r.t. that norm.
- e.g., sign descent → $\ell_\infty$ margin. Adam & SGD can pick different solutions.

*Why keep training after 0 train error?*
- Logistic/CE: margin keeps growing (slowly) → test error can still ⬇️, even when val loss ⬆️.
- Final geometry → [Neural Collapse](#neural-collapse).

*Implicit vs explicit regularization?*
- Gradient flow on least squares from 0, time $s$: component along eigvec $j$ of $X^TX$ (eigval $\nu_j$) = $(1-e^{-\nu_js})$ × OLS component.
- Ridge: $\frac{\nu_j}{\nu_j+\lambda}$ × OLS component.
- → [Early stopping](train/reg.md#early-stopping) at time $s$ ≈ ridge w/ $\lambda\approx1/s$.

*Role of init scale?*
- Large init → lazy, kernel-like regime → [NTK](#ntk) → bias = min RKHS norm.
- Small init → rich, feature-learning regime → bias toward low-rank / sparse solutions.

*When does the SDE view fail?*
- Large $\eta$ → discrete-step effects dominate → [Edge of Stability](#edge-of-stability).
- $B$ beyond the critical batch size → less noise ❌compensated by more steps.
```

&nbsp;

### Grokking
- **What**: Test accuracy jumps from chance to ~100% long after train accuracy reached 100%. {cite:p}`power2022grokking`
- **Why**: Generalization can start long after [overfitting](train/issues.md#overfitting) → stopping at the first val plateau can stop too early.
    - Small algorithmic tasks → clean test bed for how memorization turns into generalization.
- **How**:
    1. Train a small transformer on a fraction of a binary-op table, e.g., $c=(a+b)\bmod P$.
    2. **Memorization**: train acc → 100% fast. Test acc ≈ chance.
    3. **Circuit formation**: a generalizing algorithm grows inside the weights. Test acc still ≈ flat.
    4. **Cleanup**: [weight decay](train/reg.md#weight-decay) removes the memorizing components → test acc jumps.
    - Less data → more steps until generalization.
    - Mechanism for modular addition: Fourier features + trig identities (Math). {cite:p}`nanda2023progress`

```{note} Math
:class: dropdown
Notations:
- IO:
    - $a,b\in\{0,\dots,P-1\}$: Input tokens.
    - $c\in\{0,\dots,P-1\}$: Candidate output (logit idx).
- Hyperparams:
    - $P$: Prime modulus.
- Misc:
    - $\omega_k=2\pi k/P$: Fourier frequency $k$.
    - $\mathcal{K}$: Key frequencies (small subset of $\{1,\dots,P-1\}$).

Task: $c^*=(a+b)\bmod P$.

Learned algorithm (1-layer transformer):
1. **Embed**: $a\mapsto(\cos\omega_ka,\ \sin\omega_ka)$ for $k\in\mathcal{K}$ = a point on a circle per frequency.
2. **Attention + MLP**: rotate by $b$ via trig identities:

    $$
    \begin{align*}
    \cos(\omega_k(a+b))&=\cos\omega_ka\cos\omega_kb-\sin\omega_ka\sin\omega_kb\\
    \sin(\omega_k(a+b))&=\sin\omega_ka\cos\omega_kb+\cos\omega_ka\sin\omega_kb
    \end{align*}
    $$

3. **Unembed**: rotate by $-c$ & sum over key frequencies:

    $$
    \text{logit}(c)\approx\sum_{k\in\mathcal{K}}\alpha_k\cos(\omega_k(a+b-c))=\sum_{k\in\mathcal{K}}\alpha_k\left[\cos(\omega_k(a+b))\cos\omega_kc+\sin(\omega_k(a+b))\sin\omega_kc\right]
    $$

    - $\alpha_k$: Per-frequency weight.
    - Fits 95% of the logit variance in the reverse-engineered model.

Progress measures:
- **Restricted loss**: ablate all non-key frequencies → measures the generalizing circuit.
- **Excluded loss**: ablate only the key frequencies → measures memorization.
- Both improve continuously before the test-acc jump → grokking = gradual inside, sudden outside.
```

```{tip} Derivation
:class: dropdown
*Why does the logit peak only at $c^*$?*
1. Let $d=(a+b-c)\bmod P$. Then $\cos(\omega_k(a+b-c))=\cos(2\pi kd/P)$.
2. $d=0$ → every term $=1$ → $\text{logit}=\sum_k\alpha_k$ (max, for $\alpha_k>0$).
3. $d\neq0$: $P$ prime, $k,d\not\equiv0$ → $kd\not\equiv0\pmod P$ → $\cos(2\pi kd/P)<1$ for every $k$.
4. → every wrong $c$ has a strictly lower logit.
5. Several frequencies → their cosines rarely align at a wrong $d$ → larger margin than 1 frequency.
```

```{note} Example
:class: dropdown
Setup in the mechanism study:
- $P=113$. Train on 30% of all $113\cdot113$ pairs. Test on the rest.
- 1-layer ReLU transformer: $d_\text{model}=128$, 4 heads of dim 32, 512 MLP units, ❌LayerNorm.
- Full-batch AdamW, LR $10^{-3}$, weight decay $\lambda=1$, 40k epochs.
- Key frequencies of the main run: $k\in\{14,35,41,42,52\}$. Seed-specific.
```

````{important} Code
:class: dropdown
```python
import torch

class FourierModAdd:
    ## Hand-written version of the circuit a grokked 1-layer transformer learns
    def __init__(self, P=113, keys=(14, 35, 41, 42, 52)):
        self.P = P
        self.w = 2 * torch.pi * torch.tensor(keys, dtype=torch.float64) / P   ## (K,) key freqs

    def __call__(self, a, b):
        wa, wb = self.w * a[:, None], self.w * b[:, None]                    ## (N, K)
        ## embed on circles + trig identities -> cos/sin of w_k(a+b)
        cos_ab = wa.cos() * wb.cos() - wa.sin() * wb.sin()
        sin_ab = wa.sin() * wb.cos() + wa.cos() * wb.sin()
        wc = self.w * torch.arange(self.P, dtype=torch.float64)[:, None]    ## (P, K)
        ## logit(c) = sum_k cos(w_k(a+b-c)), alpha_k = 1
        return cos_ab @ wc.cos().T + sin_ab @ wc.sin().T                    ## (N, P)

## Example
a, b = torch.tensor([5, 100]), torch.tensor([7, 50])
print(FourierModAdd()(a, b).argmax(-1))   ## tensor([12, 37]) = (a + b) % 113
```
````

```{attention} Q&A
:class: dropdown
*Is grokking a sudden phase change?*
- Outside (test acc): yes. Inside (weights): ❌.
- Restricted/excluded loss improve smoothly long before the jump.
- The jump happens in cleanup, after the generalizing circuit already exists.

*Why does weight decay matter?*
- Weight decay: particularly effective among the tested interventions. On $S_5$ composition: more than halved the data needed vs most other interventions.
- Generalizing circuit = fewer, structured weights → lower norm for the same logits. Memorization = per-example lookup → high norm.
- Weight decay → norm pressure → slowly favors the generalizing circuit.
- Both limited data & regularization are needed for grokking in the mechanism study.

*Why does memorization come first?*
- Memorizing is fast to learn. The general algorithm is slow to form but cheaper in norm.
- Caveat: this "efficiency" explanation is widely held but not fully settled.

*Relation to double descent?*
- Both: test error non-monotone in training time ([epoch-wise double descent](#double-descent)).
- Grokking: extreme delay, ❌label noise, driven by regularization.

*Practical lesson?*
- Small/algorithmic data: a val plateau ≠ convergence.
- Aggressive [early stopping](train/reg.md#early-stopping) can stop before generalization.

*What does it say about intelligence?*
- Fitting ≠ understanding. A system can match all seen data by lookup & find the compact rule much later.
- Compression pressure (lower norm) drives the switch from lookup to algorithm.
```

&nbsp;

### Neural Collapse
- **What**: Last-layer features & classifier converge to a symmetric simplex geometry in the terminal phase of training (TPT). {cite:p}`papyan2020prevalence`
    - **TPT**: training continues after train error = 0 → train loss → 0.
- **Why**: Shows what training past 0 train error does to the representation → a clean, analyzable endpoint for classifier geometry.
- **How**: 4 linked properties emerge in TPT:
    1. **NC1**: within-class variability → 0. Each feature → its class mean.
    2. **NC2**: centered class means → **simplex ETF** (equiangular tight frame): equal norms, equal & maximal pairwise angles.
    3. **NC3**: classifier rows → centered class means, up to scale (self-duality).
    4. **NC4**: classifier decision → nearest class center (NCC).

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{h}_i\in\mathbb{R}^d$: Last-layer feature of sample $i$.
    - $y_i\in\{1,\dots,K\}$: Label.
- Params:
    - $W\in\mathbb{R}^{K\times d}$: Classifier, rows $\mathbf{w}_k^T$.
    - $\mathbf{b}\in\mathbb{R}^K$: Classifier bias.
- Misc:
    - $K$: #classes. $d$: Feature dim.
    - $\boldsymbol\mu_k$: Train mean of class $k$ features.
    - $\boldsymbol\mu_G$: Global train feature mean.
    - $M=[\boldsymbol\mu_1-\boldsymbol\mu_G,\dots,\boldsymbol\mu_K-\boldsymbol\mu_G]\in\mathbb{R}^{d\times K}$: Centered class means.

Scatter matrices:

$$
\Sigma_W=\frac{1}{m}\sum_{i=1}^m(\mathbf{h}_i-\boldsymbol\mu_{y_i})(\mathbf{h}_i-\boldsymbol\mu_{y_i})^T,\qquad\Sigma_B=\frac{1}{K}\sum_{k=1}^K(\boldsymbol\mu_k-\boldsymbol\mu_G)(\boldsymbol\mu_k-\boldsymbol\mu_G)^T
$$

Definition (limits as training time → ∞):
- **NC1**: $\Sigma_W\to0$. Metric: $\frac{1}{K}\text{tr}(\Sigma_W\Sigma_B^+)\to0$.
- **NC2**: $\|\boldsymbol\mu_k-\boldsymbol\mu_G\|$ all equal. $\cos(\boldsymbol\mu_k-\boldsymbol\mu_G,\boldsymbol\mu_l-\boldsymbol\mu_G)\to-\frac{1}{K-1}$ for $k\neq l$.
- **NC3**: $\frac{W^T}{\|W\|_F}-\frac{M}{\|M\|_F}\to0$.
- **NC4**: $\arg\max_k(\mathbf{w}_k^T\mathbf{h}+b_k)\to\arg\min_k\|\mathbf{h}-\boldsymbol\mu_k\|$.

Simplex ETF:

$$
M^\star=\sqrt{\frac{K}{K-1}}\,U\left(I_K-\frac{1}{K}\mathbf{1}\mathbf{1}^T\right)
$$

- $U\in\mathbb{R}^{d\times K}$: $U^TU=I_K$ (rotation, needs $d\ge K$).
- Columns: unit norm, pairwise cosine $-\frac{1}{K-1}$, sum to 0.
- NC2 = $M\to\alpha M^\star$ for some scale $\alpha>0$.
```

```{tip} Derivation
:class: dropdown
*Why is $-\frac{1}{K-1}$ the most separated configuration?*
1. Take $K$ unit vectors $\mathbf{v}_k$.
2. $0\le\|\sum_k\mathbf{v}_k\|^2=K+\sum_{k\neq l}\cos(\mathbf{v}_k,\mathbf{v}_l)$.
3. → mean pairwise cosine $\ge-\frac{K}{K(K-1)}=-\frac{1}{K-1}$.
4. Equality iff $\sum_k\mathbf{v}_k=0$. All pairs equal → the ETF.
5. → the ETF maximizes the smallest pairwise angle → max-margin geometry for $K$ balanced classes.
```

````{important} Code
:class: dropdown
```python
import torch

class NeuralCollapse:
    ## NC1-NC4 metrics from last-layer train features h, labels y, classifier (W, b)
    def __call__(self, h, y, W, b):
        K = W.shape[0]
        mu = torch.stack([h[y == k].mean(0) for k in range(K)])     ## (K, d) class means
        M = mu - h.mean(0)                                          ## (K, d) centered means
        ## NC1: within-class scatter relative to between-class scatter
        Sw = sum((h[y == k] - mu[k]).T @ (h[y == k] - mu[k]) for k in range(K)) / len(h)
        Sb = M.T @ M / K
        nc1 = torch.trace(Sw @ torch.linalg.pinv(Sb)) / K
        ## NC2: pairwise cosines of centered means -> -1/(K-1)
        Mn = M / M.norm(dim=1, keepdim=True)
        off = (Mn @ Mn.T)[~torch.eye(K, dtype=torch.bool)]
        nc2 = (off + 1 / (K - 1)).abs().mean()
        ## NC3: classifier rows ∝ centered means (self-duality)
        nc3 = (W / W.norm() - M / M.norm()).norm()
        ## NC4: agreement of classifier with nearest-class-center rule
        nc4 = ((h @ W.T + b).argmax(1) == torch.cdist(h, mu).argmin(1)).float().mean()
        return dict(nc1=nc1.item(), nc2=nc2.item(), nc3=nc3.item(), nc4_agree=nc4.item())

## Example: features already collapsed onto a simplex ETF
K, d = 3, 4
etf = torch.eye(K) - 1 / K                              ## rows: equal norms, cos = -1/(K-1)
M = torch.cat([etf, torch.zeros(K, d - K)], dim=1)      ## (K, d), embed in R^d
y = torch.arange(K).repeat(10)                          ## 30 balanced samples
h = M[y] + 0.01 * torch.randn(30, d)
print(NeuralCollapse()(h, y, W=M, b=torch.zeros(K)))    ## nc1, nc2, nc3 ≈ 0; nc4_agree = 1.0
```
````

```{attention} Q&A
:class: dropdown
*Why does it happen?*
- Unconstrained features model: treat last-layer features as free variables (expressive backbone).
- Theory in this model: CE + weight decay on features & classifier → only global minimizers = NC configurations. Holds for the model, ❌proven for real nets.
- Plus symmetry: balanced classes + class-symmetric loss → symmetric optimum.

*Relation to implicit bias?*
- CE past 0 error → margins keep growing → max-margin solution ([Implicit Bias of SGD](#implicit-bias-of-sgd)).
- Simplex ETF = max-margin geometry for $K$ balanced classes.

*Does NC explain generalization?*
- Contested. NC often holds on the train set but ❌on the test set → mainly an optimization phenomenon. {cite:p}`hui2022limitations`
- Longer training → more collapse → can give worse last-layer features for transfer.

*Class imbalance?*
- Imbalance → generally ❌ETF.
- Imbalance ratio ⬆️ → minority-class classifiers converge toward each other (**minority collapse**) → worse minority accuracy.

*Can a LM head collapse to an ETF?*
- ❌ if $K>d+1$. $K$ vectors w/ pairwise cosine $-\frac{1}{K-1}$ span $K-1$ dims.
- LLM vocab $K\gg d$ → exact NC2 is impossible.

*NC4 vs softmax classifier?*
- NC1 + NC3 → logits $\mathbf{w}_k^T\mathbf{h}$ ∝ $\boldsymbol\mu_k^T\mathbf{h}$, equal norms → argmax = nearest mean.
- → the trained linear head adds nothing beyond nearest class center.
```

&nbsp;

## Loss Landscape

### Flat Minima
- **What**: Minima where the loss stays low in a large param neighborhood, i.e., small Hessian eigenvalues. {cite:p}`hochreiter1997flat`
- **Why**: Hypothesis: flat → generalizes better.
    - Test loss surface ≈ shifted train surface → sharp minimum: small shift → large loss ⬆️. Flat minimum: robust.
    - MDL view: flat → low-precision weights suffice → fewer bits → simpler model.
- **How**:
    - **Measure**: $\lambda_\text{max}$, $\text{tr}(H)$, or worst-case loss ⬆️ in a $\rho$-ball.
    - **Small $B$ → flat, large $B$ → sharp**: SGD noise kicks the iterate out of narrow basins. {cite:p}`keskar2017large`
    - **Seek flatness explicitly**: [SAM](train/optim.md#sam) (minimize the worst loss in a $\rho$-ball), [SWA](train/train.md#swa) (average weights along the trajectory).
    - Caveat: flatness is ❌reparametrization-invariant. Rescale ReLU layers → same function, arbitrarily sharp minimum. (Dinh et al., 2017)

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\rho$: Perturbation radius.
    - $\sigma$: Perturbation std.
- Misc:
    - $\theta^*$: A minimum, $\nabla\mathcal{L}(\theta^*)=0$.

Sharpness measures:

$$
\lambda_\text{max}(H(\theta^*)),\qquad S_\rho(\theta^*)=\max_{\|\boldsymbol\delta\|\le\rho}\mathcal{L}(\theta^*+\boldsymbol\delta)-\mathcal{L}(\theta^*)\approx\frac{\rho^2}{2}\lambda_\text{max}
$$

$$
\mathbb{E}_{\boldsymbol\delta\sim\mathcal{N}(0,\sigma^2I)}\left[\mathcal{L}(\theta^*+\boldsymbol\delta)\right]-\mathcal{L}(\theta^*)\approx\frac{\sigma^2}{2}\text{tr}(H)
$$

- $\boldsymbol\delta$: Param perturbation.
- Worst case → $\lambda_\text{max}$. Average case → $\text{tr}(H)$.
```

````{tip} Derivation
:class: dropdown
*Where do the approximations come from?*
1. Taylor at a minimum: $\mathcal{L}(\theta^*+\boldsymbol\delta)\approx\mathcal{L}(\theta^*)+\frac{1}{2}\boldsymbol\delta^TH\boldsymbol\delta$ ← $\nabla\mathcal{L}=0$.
2. $\max_{\|\boldsymbol\delta\|\le\rho}\frac{1}{2}\boldsymbol\delta^TH\boldsymbol\delta=\frac{\rho^2}{2}\lambda_\text{max}$ ← Rayleigh quotient, $\lambda_\text{max}\ge0$.
3. $\mathbb{E}[\boldsymbol\delta^TH\boldsymbol\delta]=\text{tr}(H\,\mathbb{E}[\boldsymbol\delta\boldsymbol\delta^T])=\sigma^2\text{tr}(H)$.

*Why can sharpness be made arbitrarily large w/o changing the function?*
1. 2-layer ReLU net $f(x)=W_2\,\text{ReLU}(W_1x)$. ReLU positively homogeneous: $\text{ReLU}(\alpha z)=\alpha\,\text{ReLU}(z)$, $\alpha>0$.
2. $T_\alpha:(W_1,W_2)\mapsto(\alpha W_1,W_2/\alpha)$ → same $f$ → $\mathcal{L}\circ T_\alpha=\mathcal{L}$.
3. $T_\alpha$ linear w/ Jacobian $D=\text{diag}(\alpha I,\alpha^{-1}I)$. Differentiate twice: $H(\theta)=D\,H(T_\alpha\theta)\,D$.
4. → $H(T_\alpha\theta)=D^{-1}H(\theta)D^{-1}$ → $W_2$ block scaled by $\alpha^2$.
5. $\alpha\to\infty$ → $\lambda_\text{max}\to\infty$ (if the $W_2$ block $\neq0$). Same function, same test error.
````

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class Sharpness:
    ## lambda_max(H) by power iteration on Hessian-vector products (never forms H)
    ## valid at/near a minimum (H PSD); in general it returns the eigenvalue of largest |.|
    def __init__(self, model, loss_fn, iters=50):
        self.model, self.loss_fn, self.iters = model, loss_fn, iters

    def __call__(self, x, y):
        params = [p for p in self.model.parameters() if p.requires_grad]
        loss = self.loss_fn(self.model(x), y)
        grads = torch.autograd.grad(loss, params, create_graph=True)   ## keep graph for 2nd derivative
        v = [torch.randn_like(p) for p in params]
        for _ in range(self.iters):
            norm = torch.sqrt(sum((u * u).sum() for u in v))
            v = [u / norm for u in v]
            ## Hv = d/dtheta <grad L, v>
            hv = torch.autograd.grad(grads, params, grad_outputs=v, retain_graph=True)
            lam = sum((h * u).sum() for h, u in zip(hv, v)).item()       ## Rayleigh quotient v^T H v
            v = [h.detach() for h in hv]
        return lam   ## non-PSD H: may be a negative eigenvalue -> use Lanczos for the largest algebraic one

## Example: linear model + MSE -> H = 2 X^T X / m exactly
X, y = torch.randn(32, 5), torch.randn(32, 1)
net = nn.Linear(5, 1, bias=False)
print(Sharpness(net, nn.MSELoss())(X, y))
print(torch.linalg.eigvalsh(2 * X.T @ X / 32).max().item())   ## ≈ same value
```
````

```{attention} Q&A
:class: dropdown
*Pros of flatness as a generalization measure?*
- Perturbation-based sharpness ranks among the most predictive measures in large empirical studies.
- Actionable: [SAM](train/optim.md#sam), [SWA](train/train.md#swa), small batch, large LR all ⬇️ sharpness.

*Cons?*
- ❌Reparametrization-invariant → raw $\lambda_\text{max}$ cannot be a sound measure alone.
- Fix: scale-invariant variants (normalize the perturbation by weight magnitude).
- Correlation ≠ cause: flatness & generalization can both come from a 3rd factor (e.g., noise, LR).

*Is the large-batch gap really caused by sharp minima?*
- Partly contested. Much of the gap closes w/ LR scaling, warmup & more steps → partly an optimization-budget artifact.
- → [Batch-size Pathologies](train/issues.md#batch-size-pathologies).

*Why does SGD find flat minima?*
- Noise ∝ $\eta/B$ → escapes narrow basins ([Implicit Bias of SGD](#implicit-bias-of-sgd)).
- Stability: GD cannot stay at a minimum w/ $\lambda_\text{max}>2/\eta$ → [Edge of Stability](#edge-of-stability).

*What does a NN Hessian look like at a minimum?*
- Bulk of eigenvalues ≈ 0 + few large outliers.
- → "flat" in most directions. Sharpness = a few directions.

*Which matters: $\lambda_\text{max}$ or $\text{tr}(H)$?*
- $\lambda_\text{max}$: worst case & stability limit. $\text{tr}(H)$: average case (PAC-Bayes w/ Gaussian posterior).
- They can disagree. No single measure is settled.
```

&nbsp;

### Edge of Stability
- **What**: Full-batch GD regime where $\lambda_\text{max}$ hovers just above $2/\eta$ & the loss still decreases, non-monotonically. {cite:p}`cohen2021gradient`
- **Why**: Classical theory: step size must satisfy $\eta<2/\lambda_\text{max}$, else divergence. NN training violates this & still converges.
    - → $\eta$ decides the curvature of the region training ends in.
- **How**:
    1. **Progressive sharpening**: while $\lambda_\text{max}<2/\eta$, $\lambda_\text{max}$ keeps rising.
    2. $\lambda_\text{max}$ crosses $2/\eta$ → iterates oscillate along the top Hessian eigvec w/ growing amplitude.
    3. **Self-stabilization**: the oscillation moves $\theta$ toward lower-sharpness regions → $\lambda_\text{max}$ drops back to ≈ $2/\eta$. One proposed mechanism, not settled.
    4. Loss: non-monotone short-term, ⬇️ long-term.
    - Momentum & Adam: same pattern at other thresholds (Math).

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\beta$: Momentum coeff.
    - $\beta_1$: Adam 1st-moment coeff.
- Misc:
    - $\mathbf{q}_i$: Eigvec of $H$ w/ eigval $\lambda_i$.
    - $P_t$: Adam preconditioner (diag).

Quadratic model $\mathcal{L}(\theta)=\frac{1}{2}\theta^TH\theta$:

$$
\theta_{t+1}=(I-\eta H)\theta_t\ \Rightarrow\ \mathbf{q}_i^T\theta_t=(1-\eta\lambda_i)^t\,\mathbf{q}_i^T\theta_0
$$

- Assume $H\succ0$ (strict local min).
- Stable iff $|1-\eta\lambda_i|<1$ ⇔ $0<\eta\lambda_i<2$ for all $i$ ⇔ $\lambda_\text{max}<2/\eta$.
- $\lambda_i>2/\eta$ → factor $<-1$ → sign-flipping oscillation w/ growing amplitude.
```

```{tip} Derivation
:class: dropdown
*Why is the momentum threshold $(2+2\beta)/\eta$?*
1. Polyak on 1 eigen-direction: $\theta_{t+1}=\theta_t-\eta\lambda\theta_t+\beta(\theta_t-\theta_{t-1})$.
    - PyTorch form $v\leftarrow\beta v+g$, $\theta\leftarrow\theta-\eta v$ gives the same recursion.
2. Characteristic polynomial: $z^2-(1+\beta-\eta\lambda)z+\beta=0$.
3. Both roots inside the unit circle (Jury) ⇔ $|\beta|<1$ & $|1+\beta-\eta\lambda|<1+\beta$.
4. → $0<\eta\lambda<2+2\beta$ → $\lambda<(2+2\beta)/\eta$.
5. $\beta=0$ → $2/\eta$ ✅.
```

```{note} Example
:class: dropdown
$\eta=0.01$ → threshold $2/\eta=200$.
- Direction w/ $\lambda=150$: factor $1-1.5=-0.5$ → oscillates, shrinks → stable.
- Direction w/ $\lambda=250$: factor $1-2.5=-1.5$ → oscillates, grows ×1.5 per step → unstable.
- EoS: training pushes $\lambda_\text{max}$ up to ≈200, then the dynamics hold it there.
- Measure $\lambda_\text{max}$ w/ the power-iteration [Sharpness](#flat-minima) code.
```

```{dropdown} Table: Maximum Stable Sharpness
| Optimizer | Threshold |
|:--|:--|
| GD | $2/\eta$ |
| Polyak momentum | $(2+2\beta)/\eta$ |
| Nesterov momentum | $(2+2\beta)/\big((1+2\beta)\eta\big)$ |
| Adam (on $\lambda_\text{max}(P_t^{-1}H)$) | $(2+2\beta_1)/\big((1-\beta_1)\eta\big)=38/\eta$ at $\beta_1=0.9$ {cite:p}`cohen2022adaptive` |

- Thresholds from the quadratic model. The NN's top curvature equilibrates near them.
```

```{attention} Q&A
:class: dropdown
*Why does it matter?*
- $\eta$ implicitly caps the sharpness of the final solution → large $\eta$ → flatter solutions.
- Classical "$\eta<2/L$" step-size theory does not describe NN training.

*Does it hold for SGD?*
- Similar effect. Stronger as $B$ ⬆️ (closer to full batch).

*Cross-entropy?*
- Sharpness often drops at the end of training.
- ← CE 2nd derivative → 0 as margins grow → Gauss-Newton part of $H$ shrinks.

*Adam?*
- Preconditioned sharpness $\lambda_\text{max}(P_t^{-1}H)$ equilibrates at its threshold.
- Unlike GD, Adam can keep moving into higher raw-curvature regions & adapt $P_t$ to compensate.

*Is $2/\eta$ exact for real nets?*
- Derived from a quadratic approximation.
- Non-differentiable parts (e.g., ReLU) → instability can start slightly below $2/\eta$.

*Link to warmup?*
- One explanation of [Warmup](train/schedule.md#warmup): small early $\eta$ → sharpness can drop before $\eta$ is large → avoids early instability.

*Why progressive sharpening?*
- Open.
- Weak for shallow or wide nets & simple datasets.
```

&nbsp;

### Mode Connectivity
- **What**: Independently trained minima are joined by simple curves of near-constant low loss. {cite:p}`garipov2018loss,draxler2018essentially`
- **Why**: Old picture: isolated minima separated by high barriers. Observed: pairs of trained minima are joined by low-loss paths.
    - → cheap ensembles & weight averaging.
- **How**:
    - Straight line between 2 minima from different inits: usually a high-loss barrier.
    - Curved path: fix endpoints $\theta_A,\theta_B$ → learn a bend point $\theta_C$ → minimize the average loss along the curve.
    - Curve families: polygonal chain w/ 1 bend, quadratic Bezier. Alternative: nudged elastic band (AutoNEB).
    - **FGE** (Fast Geometric Ensembling): cyclical LR → collect models along the low-loss region → ensemble at ~1 training run's cost.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\theta_C$: Bend point (trainable).
- Misc:
    - $\theta_A,\theta_B$: 2 trained minima (fixed endpoints).
    - $s\in[0,1]$: Position along the curve.

Quadratic Bezier curve:

$$
\phi_{\theta_C}(s)=(1-s)^2\theta_A+2s(1-s)\theta_C+s^2\theta_B
$$

Objective:

$$
\min_{\theta_C}\ \mathbb{E}_{s\sim U[0,1]}\left[\mathcal{L}(\phi_{\theta_C}(s))\right]
$$

- Each step: sample $s$ → 1 forward/backward at $\phi(s)$ → update $\theta_C$.
- Uniform $s$ in the curve parameter ❌= uniform in arc length. Simple & works.
```

```{attention} Q&A
:class: dropdown
*Why do such paths exist?*
- Over-param → many directions keep the loss flat.
- AutoNEB study: each minimum found has ≥1 vanishing Hessian eigenvalue beyond the trivial invariances.
- Permutation symmetry: permuting hidden units → $\prod_l(\text{width}_l)!$ equivalent copies of every solution.

*Practical uses?*
- FGE: ensemble of models along the low-loss region.
- [SWA](train/train.md#swa): average weights along the SGD trajectory → wider, flatter solution.
- Model merging & weight averaging → [Linear Mode Connectivity](#linear-mode-connectivity).

*Straight-line barrier between different inits: removable?*
- Conjecture: after aligning neuron permutations, SGD solutions are linearly connected. {cite:p}`entezari2022role`
- Evidence mixed. Stronger for wide nets.

*What does it say about the landscape?*
- Empirically: tested pairs of minima are joined by low-loss paths → ❌isolated basins in those tests.
- Global connectivity of the whole low-loss set: ❌proven.
```

&nbsp;

#### Linear Mode Connectivity
- **What**: 2 solutions w/ ~0 loss barrier along the straight line between them. {cite:p}`frankle2020linear`
- **Why**: Linear path → averaging the weights works. Tests whether SGD noise changes the outcome of training.
- **How**: **Instability analysis**:
    1. Train $\theta_0\to\theta_k$ (step $k$).
    2. Copy $\theta_k$ twice → train each copy w/ different SGD noise (data order, augmentation) → $\theta_T^1,\theta_T^2$.
    3. Barrier between $\theta_T^1,\theta_T^2$ ≈ 0 → $\theta_k$ is **stable to SGD noise**.
    - Small settings (MNIST): stable at init. Large settings (ResNet-50 on ImageNet): stable only early in training, ❌at init.
    - After that point, all runs end in 1 linearly connected region.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\mathcal{E}(\theta)$: Train or test error (or loss).
    - $\alpha\in[0,1]$: Interpolation coeff.

Error barrier height:

$$
\text{Barrier}(\theta^1,\theta^2)=\max_{\alpha\in[0,1]}\mathcal{E}\left(\alpha\theta^1+(1-\alpha)\theta^2\right)-\frac{1}{2}\left(\mathcal{E}(\theta^1)+\mathcal{E}(\theta^2)\right)
$$

- Linearly mode-connected ⇔ barrier ≈ 0 (within noise).
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn
from torch.func import functional_call

class LinearBarrier:
    ## loss barrier on the straight line between 2 weight sets of the same arch
    def __init__(self, model, loss_fn, n=11):
        self.model, self.loss_fn = model, loss_fn
        self.alphas = torch.linspace(0, 1, n)

    @torch.no_grad()
    def __call__(self, theta1, theta2, x, y):
        losses = torch.stack([
            self.loss_fn(functional_call(self.model,
                {k: a * theta1[k] + (1 - a) * theta2[k] for k in theta1}, (x,)), y)
            for a in self.alphas])
        ## worst point on the path minus mean of the 2 endpoints
        return (losses.max() - 0.5 * (losses[0] + losses[-1])).item(), losses

## Example: 2 nets of the same arch (train them first in practice)
net1 = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1))
net2 = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1))
x, y = torch.randn(64, 4), torch.randn(64, 1)
barrier, path = LinearBarrier(net1, nn.MSELoss())(net1.state_dict(), net2.state_dict(), x, y)
print(barrier, path.shape)   ## scalar, torch.Size([11])
```
````

```{attention} Q&A
:class: dropdown
*Why does it matter for the lottery ticket hypothesis?*
- IMP subnetworks reach full accuracy only when they are stable to SGD noise.
- Large nets: stable only after $k>0$ steps → rewind to $\theta_k$, ❌$\theta_0$ → [Lottery Ticket Hypothesis](#lottery-ticket-hypothesis).

*Why does weight averaging of fine-tuned models work?*
- Fine-tunes from 1 pretrained checkpoint start in 1 stable region → often linearly connected.
- → average weights → 1 model, ❌extra inference cost.

*Interpolating nets w/ BatchNorm?*
- Interpolated weights ≠ interpolated activation stats → recompute BN running stats before evaluation.

*Linear vs nonlinear mode connectivity?*
- Nonlinear: curved path, holds even for different inits.
- Linear: straight path. Usually needs a shared early trajectory.
```

&nbsp;

## Sparsity & Width

### Lottery Ticket Hypothesis
- **What**: Dense random-init nets contain sparse subnets ("winning tickets") that, trained alone from their original init, reach full accuracy in a similar #steps. {cite:p}`frankle2019lottery`
- **Why**: [Pruning](compress.md#pruning) removes most weights after training w/o accuracy loss. But the same sparse arch trained from a new random init does worse.
    - → the init values matter, not only the sparse structure.
- **How**: **IMP** (Iterative Magnitude Pruning):
    1. Init $\theta_0$ → train to $\theta_T$.
    2. Prune the fraction $r$ of surviving weights w/ the smallest $|\theta_T|$ → update the mask.
    3. Reset survivors to $\theta_0$ (❌random re-init).
    4. Repeat 1–3 until the target sparsity.
    - **Rewinding**: reset to $\theta_k$ (early step $k$), ❌$\theta_0$ → needed for deep nets on large data (e.g., ResNet-50 on ImageNet). {cite:p}`frankle2020linear`
    - Found tickets: <10–20% of the original size for fully-connected & conv nets on MNIST & CIFAR-10.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\mathbf{m}\in\{0,1\}^p$: Pruning mask.
- Hyperparams:
    - $r$: Pruning fraction per round.
    - $N$: #rounds.
    - $k$: Rewind step ($k=0$ → original LTH).
- Misc:
    - $\mathcal{A}^{a\to b}(\theta)$: Train from step $a$ to step $b$.

Process:
1. $\theta_k=\mathcal{A}^{0\to k}(\theta_0)$, $\mathbf{m}=\mathbf{1}$.
2. For $n=1..N$:
    1. $\theta_T=\mathcal{A}^{k\to T}(\mathbf{m}\odot\theta_k)$.
    2. Set $m_j=0$ for the fraction $r$ of surviving $j$ w/ smallest $|\theta_{T,j}|$.
3. Ticket: $\mathbf{m}\odot\theta_k$.

- Final density $=(1-r)^N$.
```

````{important} Code
:class: dropdown
```python
import copy
import torch
import torch.nn as nn

class IMP:
    def __init__(self, model, train, rate=0.2, rewind_steps=0):
        self.model, self.train, self.rate = model, train, rate
        ## mask weight matrices only; biases stay dense
        self.masks = {n: torch.ones_like(p) for n, p in model.named_parameters() if p.dim() > 1}
        train(model, self.masks, rewind_steps)              ## 0 -> original LTH; k>0 -> rewinding
        self.theta_k = copy.deepcopy(model.state_dict())    ## rewind target

    def step(self, steps):
        self.train(self.model, self.masks, steps)           ## 1. train the masked net
        for n, p in self.model.named_parameters():          ## 2. prune smallest surviving weights, per layer
            if n in self.masks:
                w = p.detach().abs()
                thr = w[self.masks[n].bool()].quantile(self.rate)
                self.masks[n] *= (w > thr).float()
        self.model.load_state_dict(self.theta_k)            ## 3. rewind survivors, NOT re-init
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if n in self.masks:
                    p.mul_(self.masks[n])
        return {n: m.mean().item() for n, m in self.masks.items()}   ## density per layer

def train(model, masks, steps):
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(steps):
        x, y = torch.randn(16, 4), torch.randn(16, 1)
        opt.zero_grad()
        nn.functional.mse_loss(model(x), y).backward()
        opt.step()
        with torch.no_grad():                               ## pruned weights stay exactly 0
            for n, p in model.named_parameters():
                if n in masks:
                    p.mul_(masks[n])

## Example
net = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 1))
imp = IMP(net, train, rate=0.2, rewind_steps=5)
for _ in range(3):
    print(imp.step(steps=50))   ## density ≈ 0.8 -> 0.64 -> 0.51 per layer
```
````

```{attention} Q&A
:class: dropdown
*Why does random re-init kill the ticket?*
- Mask & init values are co-adapted. Same mask + new init → slower training, lower accuracy.
- → the "lottery" = the init, ❌only the architecture.

*Why is rewinding needed for large nets?*
- Ticket works ⇔ its start point is stable to SGD noise ([Linear Mode Connectivity](#linear-mode-connectivity)).
- Large nets reach stability only after a few % of training → rewind to $\theta_k$.

*Cons?*
- IMP costs $N$ full trainings → ❌training savings in practice.
- Unstructured sparsity → little real speedup on GPUs.
- Pruning-at-init methods (no training before pruning) so far fall short of IMP w/ rewinding.

*Iterative vs one-shot pruning?*
- Iterative: smaller tickets at the same accuracy. One-shot: cheaper.

*What does it say about over-parameterization?*
- Hypothesis: bigger net → more candidate subnets → higher chance of a good one → easier optimization.
- **Strong LTH**: for a bounded input distribution & a target net w/ bounded weights, a sufficiently over-param random net contains a subnet w/ ~the same accuracy as the target, w/o any weight training (proven).

*LTH vs pruning for compression?*
- Pruning for inference: keep trained weights, fine-tune → [Pruning](compress.md#pruning).
- LTH: keep init weights, retrain → a statement about trainability.
```

&nbsp;

### NTK
- **Name**: Neural Tangent Kernel {cite:p}`jacot2018neural`
- **What**: Kernel $\Theta(x,x')=\nabla_\theta f_\theta(x)^T\nabla_\theta f_\theta(x')$ that governs how GD moves the network function.
- **Why**: Param space is non-convex & hard to analyze. In function space, GD = a kernel flow.
    - Infinite width → $\Theta$ fixed → training = convex kernel regression → convergence & generalization provable.
- **How**:
    1. 1 GD step on $\theta$ → change of $f(x)$ = $-\eta\,\Theta(x,\cdot)$ × residuals on the train set.
    2. **Infinite width, NTK param**: each weight moves $O(1/\sqrt{\text{width}})$ → $\Theta$ deterministic at init & constant in training.
    3. → net ≈ its 1st-order Taylor expansion around $\theta_0$ (**lazy regime**).
    4. MSE + constant $\Theta$ → linear ODE → closed form = [kernel](../ml/supervised.md#kernel-trick) regression w/ $\Theta$.
    - Finite width, small init, or [μP](train/train.md#mup) → $\Theta$ moves → features learned (**rich regime**) → NTK theory no longer applies.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $X=\{x_1,\dots,x_m\}$: Train inputs.
    - $Y\in\mathbb{R}^m$: Train targets.
    - $\mathbf{u}_s=f_{\theta_s}(X)\in\mathbb{R}^m$: Train-set outputs at time $s$.
- Misc:
    - $s$: Gradient-flow time.
    - $\Theta(X,X)\in\mathbb{R}^{m\times m}$: NTK Gram matrix.
    - $\Theta(x,X)\in\mathbb{R}^{1\times m}$: Kernel row for input $x$.
    - $\kappa_i,\mathbf{e}_i$: Eigvals / eigvecs of $\Theta(X,X)$.

Definition (scalar output):

$$
\Theta(x,x')=\nabla_\theta f_\theta(x)^T\,\nabla_\theta f_\theta(x')
$$

Gradient flow on $\mathcal{L}=\frac{1}{2}\|\mathbf{u}-Y\|^2$:

$$
\frac{d\mathbf{u}_s}{ds}=-\Theta(X,X)\,(\mathbf{u}_s-Y),\qquad\frac{df_{\theta_s}(x)}{ds}=-\Theta(x,X)\,(\mathbf{u}_s-Y)
$$

Constant $\Theta$ (infinite width) → closed form:

$$
\begin{align*}
\mathbf{u}_s&=Y+e^{-\Theta(X,X)s}(\mathbf{u}_0-Y)\\
f_{\theta_s}(x)&=f_{\theta_0}(x)+\Theta(x,X)\,\Theta(X,X)^{-1}\left(I-e^{-\Theta(X,X)s}\right)(Y-\mathbf{u}_0)
\end{align*}
$$

- $s\to\infty$: $f_\infty(x)=f_{\theta_0}(x)+\Theta(x,X)\Theta(X,X)^{-1}(Y-\mathbf{u}_0)$ = ridgeless kernel regression on the residual of the init.
- Residual along $\mathbf{e}_i$ decays as $e^{-\kappa_is}$ → large-$\kappa$ directions learned first (**spectral bias**). Often low-frequency for common kernels.
- $\Theta(X,X)\succ0$ → train loss → 0 (global convergence).

Linearized model (lazy regime):

$$
f^\text{lin}_\theta(x)=f_{\theta_0}(x)+\nabla_\theta f_{\theta_0}(x)^T(\theta-\theta_0)
$$
```

```{tip} Derivation
:class: dropdown
*Why does the function follow a kernel flow?*
1. Gradient flow: $\frac{d\theta}{ds}=-\nabla_\theta\mathcal{L}=-\sum_i\nabla_\theta f(x_i)\,(f(x_i)-y_i)$.
2. Chain rule: $\frac{df(x)}{ds}=\nabla_\theta f(x)^T\frac{d\theta}{ds}=-\sum_i\Theta(x,x_i)(f(x_i)-y_i)$.
3. Stack over train inputs → $\frac{d\mathbf{u}}{ds}=-\Theta(X,X)(\mathbf{u}-Y)$.

*Closed form for constant $\Theta$?*
1. Linear ODE in $\mathbf{u}-Y$ → $\mathbf{u}_s-Y=e^{-\Theta s}(\mathbf{u}_0-Y)$.
2. Test input: $\frac{df(x)}{ds}=-\Theta(x,X)e^{-\Theta s}(\mathbf{u}_0-Y)$.
3. Integrate: $\int_0^se^{-\Theta s'}ds'=\Theta^{-1}(I-e^{-\Theta s})$ → result above.

*Why is $\Theta$ constant at infinite width?*
- NTK param: layer output scaled by $1/\sqrt{\text{width}}$.
- Function change $O(1)$ needs each weight to move only $O(1/\sqrt{\text{width}})$.
- → Jacobian $\nabla_\theta f$ changes $\to0$ → $\Theta_s=\Theta_0$.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn
from torch.func import functional_call, jacrev, vmap

class EmpiricalNTK:
    ## Theta(x, x') = <df(x)/dtheta, df(x')/dtheta> at the current params, scalar output
    def __init__(self, model):
        self.model = model
        self.params = {k: v.detach() for k, v in model.named_parameters()}

    def _f(self, params, x):
        return functional_call(self.model, params, (x.unsqueeze(0),)).squeeze()   ## 1 sample -> scalar

    def __call__(self, x1, x2):
        ## per-sample Jacobians: {name: (N, *param_shape)}
        jac = vmap(jacrev(self._f), in_dims=(None, 0))
        j1, j2 = jac(self.params, x1), jac(self.params, x2)
        ## sum of inner products over every param tensor
        return sum(j1[k].flatten(1) @ j2[k].flatten(1).T for k in j1)

## Example
net = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 1))
x = torch.randn(4, 3)
K = EmpiricalNTK(net)(x, x)
print(K.shape, torch.allclose(K, K.T))   ## torch.Size([4, 4]) True -> symmetric PSD Gram matrix
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- Provable global convergence & generalization bounds for very wide nets.
- Explains why wide nets train easily despite non-convexity.
- Spectral bias: large-kernel-eigenvalue modes learned first. For common kernels, these are often low-frequency.

*Cons?*
- Infinite-width NTK limit → ❌feature learning → cannot explain pretraining, transfer, representation learning.
- NTK kernel regression usually underperforms the trained finite net on standard benchmarks.
- Kernel cost: $O(m^2)$ memory, $O(m^3)$ solve.

*NTK vs NNGP?*
- NNGP: infinite-width net at random init = Gaussian process prior. Exact Bayesian inference → GP posterior w/ the NNGP kernel.
- Separately: GD on the last layer only (random fixed features) → kernel regression w/ the NNGP kernel.
- NTK: covers training all layers w/ GD.

*Lazy vs rich: what decides it?*
- Not only width: large output scale or large init → lazy at any width.
- NTK param: features move $O(1/\sqrt{\text{width}})$ → ❌feature learning in the limit.
- [μP](train/train.md#mup): rescales init & per-layer LR → features move $\Theta(1)$ → feature learning survives infinite width & hyperparams transfer across width.

*How does NTK relate to sharpness?*
- MSE $\mathcal{L}=\frac{1}{2m}\|\mathbf{u}-Y\|^2$: $H=\frac{1}{m}J^TJ+(\text{residual term})$. $J$: Jacobian $\partial\mathbf{u}/\partial\theta$.
- Nonzero eigvals of $J^TJ$ = eigvals of $JJ^T=\Theta(X,X)$.
- → near interpolation: $\lambda_\text{max}\approx\kappa_\text{max}/m$ → [Edge of Stability](#edge-of-stability) = top NTK eigenvalue held at $2m/\eta$.

*Is the empirical NTK useful at finite width?*
- ✅ as a diagnostic: how much $\Theta$ moves during training measures feature learning.
```

&nbsp;
