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
# Interpretability
Methods to explain what a trained NN computes & how. Attribution: input → output. Probing: what a layer encodes. Features: how concepts are stored. Circuits: which components implement a behavior.

Notations:
- $f$: Trained network (frozen)
- $\mathbf{x}\in\mathbb{R}^n$: Input
- $S_k(\mathbf{x})$: Pre-softmax score (logit) of class $k$
- $L$: #layers
- $l$: Layer idx
- $t$: Token position idx
- $d$: Hidden dim (residual stream width)
- $\mathbf{h}_l\in\mathbb{R}^d$: Hidden activation after layer $l$ (residual stream in a transformer)
- $\mathcal{V}$: Vocab
- $W_U\in\mathbb{R}^{|\mathcal{V}|\times d}$: Unembedding matrix

&nbsp;

## Attribution
- **What**: Score each input feature by its contribution to 1 output.

&nbsp;

### Saliency Map
- **What**: Magnitude of the class-score gradient w.r.t. the input, $|\partial S_k/\partial\mathbf{x}|$. {cite:p}`simonyan2014deep`
- **Why**: Which input features does the prediction depend on? → 1 backward pass, ❌model change, ❌extra training.
- **How**:
    1. Forward $\mathbf{x}$ → class score $S_k$ (pre-softmax).
    2. Backprop $S_k$ to the input → $\nabla_\mathbf{x}S_k$.
    3. Take $|\cdot|$. For RGB: max over channels → 2D heatmap.
    - Intuition: 1st-order Taylor $S_k(\mathbf{x}+\boldsymbol\delta)\approx S_k(\mathbf{x})+\nabla_\mathbf{x}S_k^T\boldsymbol\delta$ → large $|\partial S_k/\partial x_j|$ = a small change of $x_j$ moves the score most.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\mathbf{x}_0$: Image to explain.
    - $(u,v)$: Pixel coordinates.
    - $c$: Color channel.

Definition:

$$
\mathbf{w}=\left.\frac{\partial S_k}{\partial\mathbf{x}}\right|_{\mathbf{x}_0},\qquad M_{uv}=\max_c\left|w_{u,v,c}\right|
$$

- $M$: Saliency map, 1 value per pixel.

Properties:
- Linear model $S_k=\mathbf{w}_k^T\mathbf{x}+b_k$ → saliency = $|\mathbf{w}_k|$, the same for every input.
- ReLU net = piecewise linear → saliency = weights of the local linear region around $\mathbf{x}_0$.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class SaliencyMap:
    def __init__(self, model):
        self.model = model.eval()

    def __call__(self, x, k):
        x = x.clone().requires_grad_(True)          ## (B, C, H, W)
        ## pre-softmax score S_k, not p_k: p_k also rises when other classes' scores drop
        score = self.model(x)[:, k].sum()           ## sum -> 1 backward gives per-sample grads
        grad, = torch.autograd.grad(score, x)       ## (B, C, H, W) = dS_k/dx
        return grad.abs().amax(dim=1)               ## max |grad| over channels -> (B, H, W)

## Example
model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(), nn.Flatten(), nn.Linear(4 * 8 * 8, 10))
x = torch.randn(2, 3, 8, 8)
print(SaliencyMap(model)(x, k=3).shape)   ## torch.Size([2, 8, 8])
```
````

```{attention} Q&A
:class: dropdown
*Why the score $S_k$, ❌the softmax prob $p_k$?*
- $p_k$ ⬆️ also when other classes' scores ⬇️ → $\nabla p_k$ mixes in evidence *against* other classes.
- $\nabla S_k$ isolates evidence *for* class $k$.

*Pros?*
- Cheap: 1 forward + 1 backward.
- Architecture-agnostic: any differentiable model w/ gradient access (white-box).

*Cons?*
- Noisy, speckled maps ← gradient is very local. ReLU nets have discontinuous gradients.
- **Saturation**: a feature pushed the score into a flat region → gradient ≈ 0, but the feature matters → [Integrated Gradients](#integrated-gradients).
- Sensitivity ≠ contribution. Gradient answers "which change moves the score?", ❌"what produced the score?".

*Is a plausible-looking map a faithful one?*
- ❌. Some gradient-based methods (Guided Backprop, Guided Grad-CAM) barely change when the model's higher-layer weights are randomized {cite:p}`adebayo2018sanity`.
    - → they show input structure (edges), ❌model reasoning.
    - Plain gradients pass this model-randomization test.
- **Faithfulness** = the map reflects what the model actually uses. Visual plausibility does not test it.

*Why are raw gradient maps so noisy?*
- High-dim input → the score is sensitive to many tiny, non-semantic directions.
- Same reason as adversarial examples: a tiny per-pixel step along the gradient → big score change.
- Fixes: average gradients over noisy copies of $\mathbf{x}$ (SmoothGrad), gradient × input, path integration (IG).
```

&nbsp;

### Integrated Gradients
- **What**: Average gradient along the straight path from a baseline $\mathbf{x}'$ to $\mathbf{x}$, times $(\mathbf{x}-\mathbf{x}')$. {cite:p}`sundararajan2017axiomatic`
- **Why**: Plain gradients break 2 natural axioms.
    - **Sensitivity**: input & baseline differ in 1 feature & give different outputs → that feature must get non-zero attribution. Saturation breaks it.
    - **Implementation invariance**: 2 networks w/ the same input-output function → same attributions. Methods that use internal structure (e.g., DeepLIFT, LRP) break it.
- **How**:
    1. Choose a baseline $\mathbf{x}'$ = "absence of signal" (black image, all-zero embedding). Ideally $F(\mathbf{x}')\approx$ neutral.
    2. Walk the straight line $\mathbf{x}'\to\mathbf{x}$ in $s$ steps. Take the gradient at each point.
    3. Average the gradients → multiply elementwise by $(\mathbf{x}-\mathbf{x}')$.
    - The gradient at 1 point = local slope. The average over the path = the slope the feature really "travelled" → a saturated feature still gets credit.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $F:\mathbb{R}^n\to\mathbb{R}$: Explained scalar output (e.g., $S_k$).
    - $\mathbf{x}'\in\mathbb{R}^n$: Baseline.
- Hyperparams:
    - $s$: #Riemann steps. 20–300 steps approximate the integral within 5% in the original experiments.

Definition:

$$
\text{IG}_j(\mathbf{x})=(x_j-x'_j)\int_0^1\frac{\partial F}{\partial x_j}\Big(\mathbf{x}'+\alpha(\mathbf{x}-\mathbf{x}')\Big)\,d\alpha
$$

Riemann approximation:

$$
\text{IG}_j(\mathbf{x})\approx(x_j-x'_j)\cdot\frac{1}{s}\sum_{r=1}^{s}\frac{\partial F}{\partial x_j}\Big(\mathbf{x}'+\tfrac{r}{s}(\mathbf{x}-\mathbf{x}')\Big)
$$

Properties:
- **Completeness**: $\sum_{j=1}^{n}\text{IG}_j(\mathbf{x})=F(\mathbf{x})-F(\mathbf{x}')$.
    - Holds if $g(\alpha)=F(\mathbf{x}'+\alpha(\mathbf{x}-\mathbf{x}'))$ is absolutely continuous on $[0,1]$. E.g., $F$ Lipschitz: ReLU, sigmoid, max-pool nets.
    - Continuous + differentiable a.e. alone is ❌enough. Counterexample: the Cantor function rises 0 → 1 w/ derivative 0 a.e.
- Completeness → Sensitivity(a).
- Implementation invariance ← uses only gradients of $F$.
- Among path methods (any monotone path $\mathbf{x}'\to\mathbf{x}$), IG = the unique one that is **symmetry-preserving**: 2 features w/ symmetric roles & equal values get equal attribution.
```

```{tip} Derivation
:class: dropdown
*Why does completeness hold?*
1. Parametrize the path: $\boldsymbol\gamma(\alpha)=\mathbf{x}'+\alpha(\mathbf{x}-\mathbf{x}')$, $\alpha\in[0,1]$.
2. Chain rule:

    $$
    \frac{d}{d\alpha}F(\boldsymbol\gamma(\alpha))=\sum_{j=1}^{n}\frac{\partial F}{\partial x_j}(\boldsymbol\gamma(\alpha))\,(x_j-x'_j)
    $$

3. Integrate over $[0,1]$. Fundamental theorem of calculus on the left:

    $$
    F(\mathbf{x})-F(\mathbf{x}')=\sum_{j=1}^{n}(x_j-x'_j)\int_0^1\frac{\partial F}{\partial x_j}(\boldsymbol\gamma(\alpha))\,d\alpha=\sum_{j=1}^{n}\text{IG}_j(\mathbf{x})
    $$

4. The right side swaps sum & integral → each term = 1 feature's share.
```

```{note} Example
:class: dropdown
1 ReLU unit: $F(x)=1-\text{ReLU}(1-x)$. Baseline $x'=0$, input $x=2$.
- $F(0)=0$, $F(2)=1$ → the output changed by 1.
- Gradient at $x=2$: $F$ is flat for $x>1$ → $F'(2)=0$ → saliency = 0. ❌Sensitivity.
- IG: $F'(z)=1$ for $z<1$, $0$ for $z>1$. On the path $z=2\alpha$: $F'=1$ iff $\alpha<\tfrac12$.

$$
\text{IG}(2)=(2-0)\int_0^1F'(2\alpha)\,d\alpha=2\cdot\tfrac12=1=F(2)-F(0)
$$

- ✅ Full credit to the saturated feature. Completeness holds.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class IntegratedGradients:
    def __init__(self, model, steps=64):
        self.model, self.steps = model.eval(), steps

    def __call__(self, x, baseline, k):
        ## points on the straight path x' + (r/s)(x - x'), r = 1..s -> (s, n)
        alphas = torch.arange(1, self.steps + 1).float().div(self.steps).unsqueeze(1)
        path = (baseline + alphas * (x - baseline)).requires_grad_(True)
        ## 1 batched backward: grad of F = S_k at every path point -> (s, n)
        grads, = torch.autograd.grad(self.model(path)[:, k].sum(), path)
        ## Riemann average of path grads, scaled by the distance from the baseline
        return (x - baseline) * grads.mean(dim=0)                 ## (n,)

## Example
torch.manual_seed(0)
model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 3))
x, x0 = torch.randn(4), torch.zeros(4)
ig = IntegratedGradients(model, steps=300)(x, x0, k=1)
with torch.no_grad():
    gap = model(x)[1] - model(x0)[1]                ## F(x) - F(x')
print(ig.sum().item(), gap.item())                  ## ≈ equal -> completeness
```
````

```{attention} Q&A
:class: dropdown
*Why does the baseline matter so much?*
- IG = contribution *relative to the baseline*. Different baseline → different question → different map.
- Black baseline → black pixels have $x_j-x'_j=0$ → 0 attribution, even if "dark" is the evidence.
- Options: zero/black, blurred input, uniform noise, average over many baselines. No universally correct choice.

*Why the straight-line path?*
- Simplest path. Unique symmetry-preserving path method.
- Other paths also satisfy completeness, but give different per-feature splits.

*Pros?*
- Axiomatic: completeness, sensitivity, implementation invariance.
- ❌Model change. Works on any differentiable model.
- Built-in sanity check: $|\sum_j\text{IG}_j-(F(\mathbf{x})-F(\mathbf{x}'))|$ large → too few steps.

*Cons?*
- Cost: $s$ forward + backward passes per input.
- Path inputs (half-faded images) are off-distribution → gradients there may not reflect behavior on real data.
- Completeness ≠ faithfulness. The total is correct, but the split among interacting features can mislead.

*IG vs saliency?*
- [Saliency](#saliency-map) = gradient at 1 point = local sensitivity.
- IG = integrated gradient × displacement = global contribution vs baseline.
- $F$ linear → IG = gradient × input (w/ zero baseline).
```

&nbsp;

## Probing
- **What**: Read out what a layer encodes. Fit a simple model on its frozen activations.

&nbsp;

### Linear Probe
- **What**: Linear classifier trained on frozen activations $\mathbf{h}_l$ to predict a property $z$. {cite:p}`alain2016understanding`
- **Why**: Is $z$ linearly decodable at layer $l$? → track where information appears & disappears across depth. Cheap.
- **How**:
    1. Freeze $f$. Run a labeled dataset → collect pairs $(\mathbf{h}_l^{(i)},z^{(i)})$.
    2. Fit [softmax regression](../../ml/supervised.md#softmax-regression) on a train split. Probe gradients never reach $f$.
    3. Report held-out accuracy per layer. Compare to baselines: majority class, control task, randomly initialized $f$.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $z\in\{1,\dots,K\}$: Property label (e.g., part of speech).
- Params:
    - $V\in\mathbb{R}^{K\times d}$: Probe weights.
    - $\mathbf{b}\in\mathbb{R}^K$: Probe bias.

Model:

$$
\hat p(z|\mathbf{h}_l)=\text{softmax}(V\mathbf{h}_l+\mathbf{b})
$$

Training:

$$
\min_{V,\mathbf{b}}\ -\frac{1}{m}\sum_{i=1}^{m}\log\hat p\big(z^{(i)}\big|\mathbf{h}_l^{(i)}\big)
$$

Control task & selectivity:

$$
\text{Selectivity}=\text{Acc}(\text{task})-\text{Acc}(\text{control})
$$

- Control task: each word type gets 1 fixed random label, w/ the same label distribution as the real task.
    - Solvable only by memorizing word identity → the probe's own capacity.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearProbe(nn.Module):
    def __init__(self, d, num_classes):
        super().__init__()
        self.head = nn.Linear(d, num_classes)      ## the probe is plain softmax regression

    def forward(self, h):
        return self.head(h.detach())               ## detach: the probe never changes the model

    def fit(self, H, z, steps=300, lr=0.1):
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        for _ in range(steps):
            loss = F.cross_entropy(self(H), z)
            opt.zero_grad(); loss.backward(); opt.step()
        return self

    @torch.no_grad()
    def accuracy(self, H, z):
        return (self(H).argmax(-1) == z).float().mean().item()

## Example: real property z is linearly encoded along a direction u
torch.manual_seed(0)
n_types, d, m = 50, 16, 600
E = torch.randn(n_types, d)                        ## word-type identity directions
tok = torch.randint(0, n_types, (m,))              ## token ids: word types repeat across samples
H = E[tok] + 0.5 * torch.randn(m, d)               ## frozen activations h_l = identity + context
u = torch.randn(d)
z = (H @ u > 0).long()                             ## real property
## control task: 1 fixed random label per word TYPE, same map in train & test
ctrl_map = torch.randint(0, 2, (n_types,))
z_ctrl = ctrl_map[tok]
tr, te = slice(0, 450), slice(450, None)
acc_task = LinearProbe(d, 2).fit(H[tr], z[tr]).accuracy(H[te], z[te])
acc_ctrl = LinearProbe(d, 2).fit(H[tr], z_ctrl[tr]).accuracy(H[te], z_ctrl[te])
## control acc > 0.5 <- probe memorizes word identity; selectivity discounts that
print(acc_task, acc_ctrl, acc_task - acc_ctrl)     ## task acc, control acc, selectivity
```
````

```{attention} Q&A
:class: dropdown
*High probe accuracy → the model uses $z$?*
- ❌. Correlation, ❌causation. Decodable ≠ used downstream.
- The probe can read a by-product the model ignores.
- Test causally: remove or steer along the probe direction → does behavior change? → [Activation Patching](#activation-patching), [Activation Steering](#activation-steering).

*Why linear, ❌MLP?*
- Expressive probe → learns the task itself → measures the probe, ❌the representation.
- The next layer reads $\mathbf{h}_l$ through a linear map → "linearly decodable" = "readily available to the model".
- Mutual information is the wrong measure: $I(\mathbf{h}_l;z)\le I(\mathbf{x};z)$ (data processing inequality). Depth never adds information, only makes it more accessible.

*What do control tasks show?*
- Expressive (MLP) probes reach high accuracy on control tasks too → low selectivity {cite:p}`hewitt2019designing`.
- → Report selectivity, ❌raw accuracy. Prefer the simplest probe w/ high selectivity.

*What did the original probes find?*
- In the original CNN experiments: linear separability of the class ⬆️ monotonically w/ depth.

*Pitfalls?*
- Randomly initialized $f$ also gives non-trivial probe accuracy → always compare.
- Lexical leakage: the label is predictable from word identity → the probe memorizes tokens.
- Class imbalance → compare to the majority baseline.

*How does it relate to the tuned lens?*
- [Tuned lens](#tuned-lens) = linear probe whose target = the model's own final next-token distribution.
```

&nbsp;

### Logit Lens
- **What**: Decode an intermediate $\mathbf{h}_l$ with the model's own final norm & unembedding. {cite:p}`nostalgebraist2020interpreting`
- **Why**: What does the model "currently predict" at layer $l$? → ❌training, ❌extra params.
    - Works ← [residual stream](../foundation/module.md#residual-connection): every layer adds into the same vector space that $W_U$ reads at the end.
- **How**:
    1. Run 1 forward pass. Cache $\mathbf{h}_l$ for every layer.
    2. Apply the final [LayerNorm](../foundation/module.md#layernorm) + $W_U$ → logits → softmax.
    3. Read the top tokens per layer → prediction trajectory over depth.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\text{LN}_f$: Final LayerNorm of the model.

Definition:

$$
\text{LogitLens}(\mathbf{h}_l)=W_U\,\text{LN}_f(\mathbf{h}_l)\in\mathbb{R}^{|\mathcal{V}|},\qquad p_l=\text{softmax}\big(\text{LogitLens}(\mathbf{h}_l)\big)
$$

- $l=L$ (pre-norm transformer) → exactly the model's output logits.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class LogitLens(nn.Module):
    def __init__(self, ln_f, W_U):
        super().__init__()
        self.ln_f, self.W_U = ln_f, W_U            ## the model's OWN final norm & unembedding, frozen

    def forward(self, h):                          ## h: (B, T, d) residual stream after any layer
        return self.W_U(self.ln_f(h))              ## logits (B, T, |V|)

## Example
d, V = 8, 10
lens = LogitLens(nn.LayerNorm(d), nn.Linear(d, V, bias=False))
hs = [torch.randn(1, 5, d) for _ in range(3)]      ## cached h_l for l = 1..3
for l, h in enumerate(hs, 1):
    print(l, lens(h)[0, -1].argmax().item())       ## layer-l guess for the next token at the last position
```
````

```{attention} Q&A
:class: dropdown
*What does it reveal about internal representations?*
- In GPT-2: a rough guess for the next token forms early, then later layers refine it.
- → "Iterative inference" view: layers refine 1 running prediction in a shared space.
- ❌Proof that the model "thinks in tokens". Non-token information is invisible to the lens.

*Cons?*
- Assumes every layer uses the final layer's basis. Early layers often do not → garbage output.
- Brittle: works poorly on some model families.
- Biased: puts systematically more mass on some tokens than the final output does.

*Logit lens vs linear probe?*
- Logit lens: ❌training, fixed decoder = the model's own.
- Linear probe: trained, any target property.
```

&nbsp;

#### Tuned Lens
- **What**: Logit lens w/ a learned affine translator per layer, trained to match the final output distribution. {cite:p}`belrose2023eliciting`
- **Why**: Logit lens is brittle & biased ← intermediate layers use a different basis than the final layer (**representational drift**).
- **How**:
    1. Per layer $l$: translator $(A_l,\mathbf{b}_l)$ maps $\mathbf{h}_l$ into the final layer's basis.
    2. Decode w/ the frozen logit lens.
    3. Train only the translators: minimize KL(final distribution ‖ lens distribution). Model frozen.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $A_l\in\mathbb{R}^{d\times d}$: Translator matrix of layer $l$.
    - $\mathbf{b}_l\in\mathbb{R}^d$: Translator bias of layer $l$.
- Misc:
    - $p_L(\cdot|\mathbf{x})$: Model's final next-token distribution.
    - $q_l(\cdot|\mathbf{x})$: Tuned-lens distribution at layer $l$.

Model:

$$
\text{TunedLens}_l(\mathbf{h}_l)=\text{LogitLens}(A_l\mathbf{h}_l+\mathbf{b}_l),\qquad q_l=\text{softmax}\big(\text{TunedLens}_l(\mathbf{h}_l)\big)
$$

- Init $A_l=I$, $\mathbf{b}_l=\mathbf{0}$ → training starts at the logit lens.

Training:

$$
\min_{A_l,\mathbf{b}_l}\ \mathbb{E}_\mathbf{x}\Big[D_\text{KL}\big(p_L(\cdot|\mathbf{x})\,\|\,q_l(\cdot|\mathbf{x})\big)\Big]
$$
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

## Reuses LogitLens from the Logit Lens block above
class TunedLens(nn.Module):
    def __init__(self, logit_lens, d, L):
        super().__init__()
        self.lens = logit_lens.requires_grad_(False)                  ## frozen
        self.translators = nn.ModuleList(nn.Linear(d, d) for _ in range(L))
        for A in self.translators:                                    ## identity init -> starts as logit lens
            nn.init.eye_(A.weight)
            nn.init.zeros_(A.bias)

    def forward(self, h, l):
        return self.lens(self.translators[l](h))                      ## logits (B, T, |V|)

    def loss(self, h, l, p_final):
        ## KL(p_final || q_l): distill the model's own final belief, not the ground-truth token
        log_q = self(h, l).log_softmax(-1)
        ## flatten (B, T, V) -> (B*T, V): batchmean then averages over tokens, not only over B
        return F.kl_div(log_q.flatten(0, -2), p_final.flatten(0, -2), reduction='batchmean')

## Example
d, V, L = 8, 10, 3
lens = LogitLens(nn.LayerNorm(d), nn.Linear(d, V, bias=False))
tl = TunedLens(lens, d, L)
h1, hL = torch.randn(1, 5, d), torch.randn(1, 5, d)
p_final = lens(hL).softmax(-1)                    ## stand-in for the model's final output distribution
print(tl.loss(h1, 0, p_final).item())             ## >= 0; minimize w.r.t. translators only
```
````

```{attention} Q&A
:class: dropdown
*Why KL to the final output, ❌cross-entropy to the next token?*
- Goal = what the *model* believes at layer $l$, ❌ground truth → distillation target.
- CE to labels → the translator could learn information the model never computed.

*Is the tuned lens a probe?*
- Yes: a linear probe per layer w/ a fixed target. → Same caveat: a decodable prediction ≠ the model uses it at that layer.
- Affine only → limited capacity → less risk than an MLP probe.

*Pros vs logit lens?*
- Lower perplexity than the logit lens. Closer to the final distribution. Less biased.
- Works on model families where the logit lens fails.

*Uses?*
- Track how predictions form over depth.
- Prediction trajectories over layers can flag anomalous or malicious inputs.
```

&nbsp;

## Features
- **What**: Properties of the input that a model represents internally (e.g., "text is French", "curve at 45°").

&nbsp;

### Linear Representation Hypothesis
- **What**: High-level concepts = directions in activation space. {cite:p}`park2024linear`
    - Concept strength = projection length along the direction.
    - Concepts combine by vector addition.
- **Why**: Explains why linear probes, word-vector arithmetic ($\text{king}-\text{man}+\text{woman}\approx\text{queen}$) & steering vectors work.
    - Plausible ← each layer reads its input through a linear map. A direction = what 1 downstream dot product can read.
- **How**: Formalize "linear" w/ counterfactual pairs. 2 senses:
    - **Unembedding representation** $\bar{\boldsymbol\gamma}_Z$: for every counterfactual word pair (man → woman, king → queen), the unembedding difference points along 1 direction. ⇔ measurement (linear probe).
    - **Embedding representation** $\bar{\boldsymbol\lambda}_Z$: adding it to the context representation changes P(concept) & leaves causally separable concepts unchanged. ⇔ intervention (steering).
    - **Causal inner product**: an inner product under which causally separable concepts are orthogonal. It maps $\bar{\boldsymbol\gamma}_Z$ to $\bar{\boldsymbol\lambda}_Z$ → measurement & intervention unify.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $Z$: Binary concept (e.g., male ⇒ female).
    - $Y(0),Y(1)$: Counterfactual word pair for $Z$ (e.g., "king", "queen").
- Misc:
    - $\boldsymbol\gamma(y)\in\mathbb{R}^d$: Unembedding vector of token $y$ (row of $W_U$).
    - $\boldsymbol\lambda(\mathbf{x})\in\mathbb{R}^d$: Final context representation. Logits: $\boldsymbol\lambda(\mathbf{x})^T\boldsymbol\gamma(y)$.

Unembedding representation:

$$
\boldsymbol\gamma(Y(1))-\boldsymbol\gamma(Y(0))\in\{\alpha\bar{\boldsymbol\gamma}_Z:\alpha>0\}\quad\text{almost surely}
$$

- A cone, ❌subspace: the sign matters (king → queen ≠ queen → king).

Causal inner product:

$$
\langle\mathbf{u},\mathbf{v}\rangle_C=\mathbf{u}^T\text{Cov}(\boldsymbol\gamma)^{-1}\mathbf{v}
$$

- $\text{Cov}(\boldsymbol\gamma)$: Covariance of $\boldsymbol\gamma(y)$ over $y$ uniform on $\mathcal{V}$.
- 1 choice in a family. The orthogonality constraints leave $d$ degrees of freedom.

Unification:

$$
\bar{\boldsymbol\lambda}_Z^T=\langle\bar{\boldsymbol\gamma}_Z,\cdot\rangle_C\quad\Rightarrow\quad\bar{\boldsymbol\lambda}_Z=\text{Cov}(\boldsymbol\gamma)^{-1}\bar{\boldsymbol\gamma}_Z
$$

- → Best steering direction ≠ probe direction in Euclidean terms. They coincide only after whitening by $\text{Cov}(\boldsymbol\gamma)$.
```

```{attention} Q&A
:class: dropdown
*Evidence?*
- Word-vector analogies.
- Linear probes decode many concepts from LLM activations.
- Steering vectors & SAE latents change behavior predictably when added.
- Othello-GPT: the board state is linearly decodable when encoded as "mine/theirs", ❌as "black/white".

*Is every feature a 1-D direction?*
- ❌. Some features are irreducibly multi-dimensional, e.g., circular representations of days of the week & months {cite:p}`engels2025not`.
- → LRH = good first approximation, ❌universal law.

*"Linear" means the network is linear?*
- ❌. Only the *representation* of a concept is linear (a direction). The computation between layers stays nonlinear.

*Why is Euclidean cosine similarity in activation space not canonical?*
- Logits $\boldsymbol\lambda^T\boldsymbol\gamma$ do not change under $\boldsymbol\gamma\to A\boldsymbol\gamma$, $\boldsymbol\lambda\to A^{-T}\boldsymbol\lambda$ for any invertible $A$.
- → The model's behavior does not fix the Euclidean geometry. Angles depend on an arbitrary basis.
- The causal inner product picks a geometry from the semantics (orthogonal = causally separable).

*How does it relate to superposition?*
- LRH: features = directions. [Superposition](#superposition): more feature directions than dims → directions cannot all be orthogonal → interference.

*What does it tell us about internal representations?*
- If true → activations decompose into readable, editable units → probes, SAEs & steering are principled.
- Where false (multi-dim or nonlinear features) → these tools miss part of the computation.
```

&nbsp;

### Superposition
- **What**: NN represents more features than it has dimensions, as non-orthogonal directions. {cite:p}`elhage2022toy`
- **Why**: Far more useful features than neurons. Most features are sparse (rarely active).
    - Sparse → 2 features rarely co-occur → interference rarely costs loss → packing extra features pays off.
- **How**:
    1. Assign each feature $i$ a direction $\mathbf{w}_i\in\mathbb{R}^d$, w/ #features $n>d$.
    2. Make directions nearly orthogonal → small interference. High dim → exponentially many nearly-orthogonal directions fit.
    3. Nonlinearity (ReLU + negative bias) filters small interference when a feature is absent.
    - Consequence: **polysemantic neurons**. 1 neuron responds to several unrelated features ← features do not align w/ the neuron basis.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{x}\in\mathbb{R}^n$: Feature vector (overrides page-level input). $x_i=0$ w/ prob $S$, else $x_i\sim U[0,1]$.
- Params:
    - $W\in\mathbb{R}^{d\times n}$, $d<n$: Embedding. Column $\mathbf{w}_i$ = direction of feature $i$.
    - $\mathbf{b}\in\mathbb{R}^n$: Output bias.
- Hyperparams:
    - $S$: Sparsity.
    - $I_i$: Importance of feature $i$.

Model (toy):

$$
\mathbf{h}=W\mathbf{x},\qquad\hat{\mathbf{x}}=\text{ReLU}(W^T\mathbf{h}+\mathbf{b})=\text{ReLU}(W^TW\mathbf{x}+\mathbf{b})
$$

Training:

$$
\mathcal{L}=\mathbb{E}_\mathbf{x}\left[\sum_{i=1}^{n}I_i(x_i-\hat x_i)^2\right]
$$

Interference (per output):

$$
\hat x_i=\text{ReLU}\Big(\|\mathbf{w}_i\|^2x_i+\sum_{j\neq i}(\mathbf{w}_i^T\mathbf{w}_j)\,x_j+b_i\Big)
$$

- $\mathbf{w}_i^T\mathbf{w}_j\neq0$ → feature $j$ leaks into output $i$.

Properties:
- $S=0$ (dense) → store the $d$ most important features in orthogonal directions, drop the rest. ≈ [PCA](../../ml/unsupervised.md#pca).
- $S\to1$ (sparse) → superposition. Geometry = antipodal pairs, triangles, pentagons, ….
- Linear model ($\hat{\mathbf{x}}=W^TW\mathbf{x}+\mathbf{b}$) → ❌superposition at any $S$.
- Random unit vectors in $\mathbb{R}^d$: $\mathbf{u}^T\mathbf{v}$ has mean 0, variance $1/d$ → nearly orthogonal for large $d$.
- #unit vectors w/ pairwise $|\cos|\le\epsilon$ can reach $e^{c\epsilon^2d}$ ($c$ = constant) → exponential in $d$.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class ToyModel(nn.Module):
    def __init__(self, n, d):
        super().__init__()
        self.W = nn.Parameter(torch.randn(d, n) * 0.1)   ## column i = direction of feature i in R^d
        self.b = nn.Parameter(torch.zeros(n))

    def forward(self, x):                                ## x: (B, n)
        h = x @ self.W.T                                 ## (B, d): squeeze n features into d < n dims
        ## W^T W x mixes features (interference); ReLU + negative bias filters small interference
        return torch.relu(h @ self.W + self.b)           ## (B, n)

## Example: 5 features, 2 dims, sparse inputs
torch.manual_seed(0)
n, d, S = 5, 2, 0.95                                     ## S = P(feature is 0)
model = ToyModel(n, d)
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
for _ in range(3000):
    x = torch.rand(1024, n) * (torch.rand(1024, n) > S)  ## each feature active w.p. 1 - S
    loss = ((model(x) - x) ** 2).sum(-1).mean()          ## uniform importance I_i = 1
    opt.zero_grad(); loss.backward(); opt.step()
print(model.W.norm(dim=0))   ## sparse: typically > 2 norms ≈ 1 -> more features than dims; S = 0: ~2
```
````

```{attention} Q&A
:class: dropdown
*Why does sparsity enable superposition?*
- Interference costs loss only when ≥ 2 superposed features are active at once. P(co-active) ∝ $(1-S)^2$.
- Benefit of storing 1 more feature ∝ $(1-S)$.
- $S$⬆️ → benefit/cost ⬆️ → superposition wins.

*Why does the nonlinearity matter?*
- Feature $i$ absent, feature $j$ active → pre-activation $=(\mathbf{w}_i^T\mathbf{w}_j)x_j+b_i$.
- Negative interference → ReLU clips it to 0 → no error.
- Small positive interference + negative bias $b_i$ → also clipped.
- Linear model: every leak is an error → the MSE optimum keeps $d$ orthogonal directions (Eckart–Young).

*Superposition vs polysemanticity?*
- Superposition → polysemanticity: $n>d$ features cannot each own a neuron.
- Converse ❌necessarily: w/o a **privileged basis**, features can also rotate off the neuron axes w/o superposition.
- Privileged basis ← elementwise nonlinearity (ReLU on neurons). A transformer's residual stream has no privileged basis.

*Why is it a problem for interpretability?*
- Neurons are polysemantic → reading features off single neurons fails.
- → Need an overcomplete feature basis → [SAE](#sae).

*Is the transition gradual?*
- ❌. Toy models show **phase changes**: as sparsity or relative importance changes, a feature jumps between "not stored" → "own dimension" → "superposed".

*How does it relate to compressed sensing?*
- Compressed sensing: a $k$-sparse vector in $\mathbb{R}^n$ is recoverable from $d\ll n$ random linear measurements if $d\gtrsim k\log(n/k)$.
- Superposition = the same idea, learned: $W$ = measurement matrix, ReLU = cheap nonlinear decoder.

*What does it tell us about internal representations?*
- Activations = sparse sums of many more feature directions than dims.
- Single neurons are the wrong unit of analysis. Feature directions are the right one.
```

&nbsp;

### SAE
- **Name**: Sparse Autoencoder {cite:p}`cunningham2024sparse,bricken2023towards`
- **What**: Overcomplete autoencoder w/ sparse latents, trained on a layer's activations → dictionary of feature directions.
- **Why**: Undo superposition. Neurons are polysemantic → need $n_f\gg d$ basis directions, each ideally 1 interpretable (**monosemantic**) feature.
- **How**:
    1. Collect activations $\mathbf{h}$ (residual stream or MLP output) over many tokens.
    2. Encoder: $\mathbf{h}$ → $n_f\gg d$ nonnegative latents $\mathbf{a}$.
    3. Decoder: $\hat{\mathbf{h}}$ = sum of active latents × their decoder directions.
    4. Loss = reconstruction MSE + L1 penalty → each input uses few latents.
    5. Interpret each latent: top-activating examples + logit effect of its decoder direction. Verify causally by clamping it.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{h}\in\mathbb{R}^d$: Activation at a fixed layer & position.
    - $\mathbf{a}\in\mathbb{R}^{n_f}$: Latent activations.
- Params:
    - $W_e\in\mathbb{R}^{n_f\times d}$: Encoder weights.
    - $\mathbf{b}_e\in\mathbb{R}^{n_f}$: Encoder bias.
    - $W_d\in\mathbb{R}^{d\times n_f}$: Decoder. Column $\mathbf{d}_i$ = direction of feature $i$.
    - $\mathbf{b}_d\in\mathbb{R}^d$: Decoder bias.
- Hyperparams:
    - $n_f$: Dictionary size, $n_f\gg d$.
    - $\lambda$: L1 coeff.

Forward:

$$
\mathbf{a}=\text{ReLU}\big(W_e(\mathbf{h}-\mathbf{b}_d)+\mathbf{b}_e\big),\qquad\hat{\mathbf{h}}=W_d\mathbf{a}+\mathbf{b}_d=\mathbf{b}_d+\sum_{i=1}^{n_f}a_i\mathbf{d}_i
$$

Training:

$$
\mathcal{L}=\|\mathbf{h}-\hat{\mathbf{h}}\|_2^2+\lambda\|\mathbf{a}\|_1\quad\text{s.t.}\quad\|\mathbf{d}_i\|_2=1\ \ \forall i
$$

- Unit-norm decoder columns ← else shrink $\mathbf{a}$ & grow $W_d$ → L1 term → 0 for free.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class SAE(nn.Module):
    def __init__(self, d, n_f):
        super().__init__()
        self.enc = nn.Linear(d, n_f)                                ## W_e, b_e
        self.W_d = nn.Parameter(torch.randn(d, n_f) / d ** 0.5)     ## columns d_i = feature directions
        self.b_d = nn.Parameter(torch.zeros(d))

    def encode(self, h):
        return torch.relu(self.enc(h - self.b_d))                   ## (B, n_f), nonnegative

    def forward(self, h):
        ## unit-norm columns: else a -> 0 & W_d -> large would cheat the L1 term
        W_d = self.W_d / self.W_d.norm(dim=0, keepdim=True)
        a = self.encode(h)
        return a @ W_d.T + self.b_d, a                              ## h_hat = b_d + sum_i a_i d_i

    def loss(self, h, lam=1e-3):
        h_hat, a = self(h)
        return ((h - h_hat) ** 2).sum(-1).mean() + lam * a.abs().sum(-1).mean()

## Example
sae = SAE(d=8, n_f=32)                     ## overcomplete: 32 latents for 8 dims
h = torch.randn(4, 8)
h_hat, a = sae(h)
print(h_hat.shape, a.shape)                ## torch.Size([4, 8]) torch.Size([4, 32])
```
````

```{attention} Q&A
:class: dropdown
*Why overcomplete?*
- Superposition → #features > $d$. A complete basis ($n_f=d$) cannot separate them.

*SAE vs PCA / ICA?*
- [PCA](../../ml/unsupervised.md#pca): ≤ $d$ orthogonal, dense components → cannot recover superposed features.
- [ICA](../../ml/unsupervised.md#ica): standard ICA = square unmixing, assumes independence → also ≤ $d$ components.
    - Overcomplete ICA exists ≈ sparse coding w/ a sparse prior → close to the SAE objective.
- SAE = sparse dictionary learning (sparse coding) w/ an amortized 1-layer encoder → fast, but the encoder only approximates the optimal sparse code.

*Why L1, ❌L0?*
- L0 (#active latents) is not differentiable. L1 = convex relaxation.
- Side effect: **shrinkage**. L1 pulls every active latent toward 0 → systematic under-reconstruction → [TopK SAE](#topk-sae).

*How do you judge an SAE?*
- Sparsity: L0 = avg #active latents per token.
- Fidelity: variance explained. Also: model loss increase when $\hat{\mathbf{h}}$ replaces $\mathbf{h}$ in the forward pass.
- Interpretability: human or LLM rating of latents; causal effect of clamping.

*Cons & open problems?*
- Reconstruction error: splicing $\hat{\mathbf{h}}$ in ⬆️ model loss → the SAE misses part of the computation.
- **Dead latents**: never fire → wasted capacity.
- **Feature splitting**: larger $n_f$ → 1 feature splits into finer ones → no unique "true" feature set.
- Assumes features = 1-D directions → misses multi-dimensional features.
- Top activations look clean, but a latent can still be polysemantic in its weak tail.

*What does it tell us about internal representations?*
- Many SAE latents are monosemantic & causally usable → supports the LRH + superposition picture.
- Clamping a "Golden Gate Bridge" latent to a high value → Claude 3 Sonnet starts to self-identify as the bridge {cite:p}`templeton2024scaling`.
```

&nbsp;

#### TopK SAE
- **What**: SAE w/ a TopK activation: keep the $k$ largest pre-activations, zero the rest. ❌L1 penalty. {cite:p}`gao2025scaling`
- **Why**: L1 = imperfect proxy for L0 → shrinkage, & $\lambda$ needs tuning. TopK fixes L0 ≤ $k$ directly.
- **How**:
    1. Encoder pre-activations → keep the top-$k$ → ReLU on the kept values → decode.
    2. Train on reconstruction MSE.
    3. Add an auxiliary loss (AuxK) → dead latents get gradient.
    - Predecessor: the **k-sparse autoencoder** keeps exactly the $k$ largest *linear* activations (❌ReLU, ❌AuxK) {cite:p}`makhzani2014ksparse`. Gao et al. scale it to LM activations & add AuxK.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $k$: #active latents per input.
    - $k_\text{aux}$: #dead latents used by AuxK (e.g., 512).
    - $\alpha$: AuxK coeff (typically $1/32$).

Forward:

$$
\mathbf{a}=\text{TopK}\big(W_e(\mathbf{h}-\mathbf{b}_d)\big),\qquad\hat{\mathbf{h}}=W_d\mathbf{a}+\mathbf{b}_d
$$

- $\text{TopK}$: keep the $k$ largest entries, set the rest to 0.
- Reference code applies ReLU to the kept values → $\mathbf{a}\ge0$, ≤ $k$ non-zero. In practice, the top-$k$ values are almost always positive → same training curves.

Training:

$$
\mathcal{L}=\|\mathbf{h}-\hat{\mathbf{h}}\|_2^2+\alpha\,\mathcal{L}_\text{aux}
$$

- 1st term: main loss, MSE only.
- $\mathcal{L}_\text{aux}$: MSE of reconstructing the residual error $\mathbf{h}-\hat{\mathbf{h}}$ w/ the top-$k_\text{aux}$ dead latents → dead latents get gradient.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn.functional as F

## Reuses SAE from the SAE block above
class TopKSAE(SAE):
    def __init__(self, d, n_f, k):
        super().__init__(d, n_f)
        self.k = k

    def encode(self, h):
        pre = F.linear(h - self.b_d, self.enc.weight)               ## (B, n_f), no encoder bias
        top = pre.topk(self.k, dim=-1)
        ## keep k largest, ReLU them, zero the rest -> L0 <= k, no L1 shrinkage
        return torch.zeros_like(pre).scatter(-1, top.indices, torch.relu(top.values))

    def loss(self, h):
        h_hat, _ = self(h)
        return ((h - h_hat) ** 2).sum(-1).mean()                    ## main MSE term; AuxK omitted here

## Example
sae = TopKSAE(d=8, n_f=32, k=4)
_, a = sae(torch.randn(4, 8))
print((a > 0).sum(-1))   ## each entry <= k = 4
```
````

```{attention} Q&A
:class: dropdown
*Pros vs L1 SAE?*
- ❌Shrinkage ← kept values are not penalized.
- Sparsity set directly by $k$ → ❌$\lambda$ sweep.
- Better reconstruction-sparsity frontier. Clean scaling laws in $n_f$ & $k$.

*Cons?*
- Fixed $k$ for every token → simple tokens & complex tokens get the same budget.
- TopK is not smooth → only the selected latents get gradient → dead latents → needs AuxK.
```

&nbsp;

### CKA
- **Name**: Centered Kernel Alignment {cite:p}`kornblith2019similarity`
- **What**: Similarity index between 2 representations of the same $m$ inputs, invariant to orthogonal transforms & isotropic scaling.
- **Why**: Compare layers across models, seeds & widths.
    - Neuron-by-neuron comparison fails ← neurons permute & rotate between runs.
    - CCA-style invariance to *any* invertible linear map is too strong: width ≥ $m$ → every pair of representations looks identical.
- **How**:
    1. Feed the same $m$ inputs to both nets → $X\in\mathbb{R}^{m\times p_1}$, $Y\in\mathbb{R}^{m\times p_2}$.
    2. Center each column.
    3. Compare the $m\times m$ Gram matrices $XX^T$ & $YY^T$ → normalized alignment.
    - Compares "which inputs look alike to which", ❌coordinates → widths $p_1\neq p_2$ are fine.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $X\in\mathbb{R}^{m\times p_1}$: Activations of net 1. Row = input.
    - $Y\in\mathbb{R}^{m\times p_2}$: Activations of net 2, same inputs, same row order.
- Misc:
    - $K=XX^T$, $L=YY^T$: Gram matrices (any kernel in general).
    - $H=I_m-\frac{1}{m}\mathbf{1}\mathbf{1}^T$: Centering matrix.

Definition:

$$
\text{HSIC}(K,L)=\frac{\text{tr}(KHLH)}{(m-1)^2},\qquad\text{CKA}(K,L)=\frac{\text{HSIC}(K,L)}{\sqrt{\text{HSIC}(K,K)\,\text{HSIC}(L,L)}}
$$

- $\text{HSIC}$: Hilbert-Schmidt Independence Criterion.

Linear CKA ($X$, $Y$ column-centered):

$$
\text{CKA}(X,Y)=\frac{\|Y^TX\|_F^2}{\|X^TX\|_F\,\|Y^TY\|_F}
$$

Properties:
- $\text{CKA}\in[0,1]$.
- Invariant to $X\to cXQ$ ($Q$ orthogonal, $c\neq0$). ❌Invariant to general invertible linear maps.
- Kernel version: replace $XX^T$ w/ e.g. an RBF Gram matrix.
```

```{tip} Derivation
:class: dropdown
*Why does linear CKA reduce to the Frobenius formula?*
1. Centered columns → $HX=X$, $HY=Y$. $H$ symmetric.
2. Cyclic trace: $\text{tr}(XX^THYY^TH)=\text{tr}\big((HX)^TYY^T(HX)\big)=\text{tr}(X^TYY^TX)=\|Y^TX\|_F^2$.
3. Same step: $\text{HSIC}(K,K)\propto\|X^TX\|_F^2$, $\text{HSIC}(L,L)\propto\|Y^TY\|_F^2$.
4. The $(m-1)^2$ factors cancel in the ratio.

*Why is it in $[0,1]$?*
1. $\|Y^TX\|_F^2=\text{tr}(KL)=\langle K,L\rangle_F$ and $\|X^TX\|_F=\|XX^T\|_F=\|K\|_F$.
2. → $\text{CKA}=\langle K,L\rangle_F/(\|K\|_F\|L\|_F)$ = cosine between Gram matrices ≤ 1.
3. $K,L$ PSD → $\text{tr}(KL)\ge0$.

*Why invariant to $X\to cXQ$?*
- $\|Y^TXQ\|_F=\|Y^TX\|_F$ & $\|Q^TX^TXQ\|_F=\|X^TX\|_F$ ← orthogonal $Q$ preserves Frobenius norm.
- $c$: factor $c^2$ in numerator & denominator → cancels.
```

````{important} Code
:class: dropdown
```python
import torch

class LinearCKA:
    def __call__(self, X, Y):                 ## X: (m, p1), Y: (m, p2); same m inputs, rows aligned
        X = X - X.mean(0, keepdim=True)       ## center each column
        Y = Y - Y.mean(0, keepdim=True)
        ## ||Y^T X||_F^2 = <XX^T, YY^T>_F: compares input-by-input similarity, not coordinates
        num = (Y.T @ X).norm() ** 2
        return (num / ((X.T @ X).norm() * (Y.T @ Y).norm())).item()

## Example
torch.manual_seed(0)
X = torch.randn(50, 8)
Q, _ = torch.linalg.qr(torch.randn(8, 8))     ## random orthogonal matrix
cka = LinearCKA()
print(cka(X, 3 * X @ Q))                      ## 1.0 -> invariant to rotation & isotropic scale
print(cka(X, X @ torch.randn(8, 8)))          ## < 1 -> NOT invariant to general invertible maps
print(cka(X, torch.randn(50, 8)))             ## low -> unrelated representations
```
````

```{attention} Q&A
:class: dropdown
*Why not CCA / SVCCA?*
- CCA is invariant to any invertible linear map.
- Width $p\ge m$ → any 2 full-rank representations are related by such a map → CCA similarity is trivially maximal.
- CKA keeps only orthogonal + isotropic-scale invariance → still discriminative.

*How was it validated?*
- Correspondence test: 2 nets from different random seeds → the most similar layer should be the same layer.
- CKA ✅ identifies corresponding layers. CCA-based indices often ❌.

*Cons?*
- Linear CKA is dominated by high-variance directions → misses low-variance but functionally important directions.
- Sensitive to a few outlier inputs.
- Similar representations ≠ same function or same mechanism.
- Biased for small $m$ → needs many inputs.

*How does it relate to RSA?*
- Same family as representational similarity analysis (neuroscience): compare input × input similarity matrices.
- → Compare nets to nets, or nets to brain recordings, w/o matching units.
```

&nbsp;

## Circuits

&nbsp;

### Circuit
- **What**: Subgraph of a NN: features (nodes) connected by weights (edges) that implements 1 behavior. {cite:p}`olah2020zoom`
- **Why**: Reverse-engineer the algorithm, ❌only correlate inputs w/ outputs → mechanistic, checkable explanation.
- **How**: 3 speculative claims:
    1. **Features** = the fundamental unit. They correspond to directions in activation space (often neurons in vision models).
    2. **Circuits**: features connect via weights. Read the weights → read the algorithm.
    3. **Universality**: analogous features & circuits form across models & tasks.
    - Example (InceptionV1): curve detectors built from earlier edge detectors; a car detector = windows on top + car body + wheels at the bottom.

```{attention} Q&A
:class: dropdown
*Why is this harder in transformers?*
- Residual stream has no privileged basis → features need not align w/ any coordinate.
- [Attention heads](../../nlp/attn.md#multi-head-attention) move information between positions → circuits span positions.
- Superposition → 1 neuron takes part in many circuits.
- Framework: residual stream = shared communication channel. Each head = **QK circuit** (where to attend) + **OV circuit** (what to move).

*Typical workflow?*
1. Pick a behavior & a dataset that isolates it.
2. Find important components w/ [activation patching](#activation-patching) or ablation.
3. Read their weights & attention patterns → hypothesize the algorithm.
4. Test by intervention: does the circuit alone reproduce the behavior?

*Is universality established?*
- ❌. Partial evidence: Gabor filters & curve detectors across CNNs; [induction heads](#induction-head) across LMs.

*Fully reverse-engineered example?*
- Modular addition: a small transformer uses discrete Fourier features & trig identities {cite:p}`nanda2023progress`. → Mechanism behind [grokking](../theory.md#grokking).

*Pros?*
- Mechanistic & falsifiable → strongest form of explanation.

*Cons?*
- Labor-intensive. Mostly small models or narrow behaviors.
- Explanations are partial → unexplained remainder.
- Risk of interpretability illusions: a clean story on a narrow dataset that does not generalize.
```

&nbsp;

### Induction Head
- **What**: Attention head that completes a repeated pattern: $[A][B]\dots[A]\to[B]$. {cite:p}`elhage2021mathematical`
- **Why**: Simplest general in-context learning mechanism: "predict what followed the current token last time". Needs ≥ 2 attention layers.
- **How**: 2 heads in 2 layers compose:
    1. **Previous-token head** (earlier layer): writes "my previous token = $x_{t-1}$" into the residual stream at $t$.
    2. **Induction head** (later layer): query = current token $A$. Key = previous-token info → highest score at the position right after the earlier $A$, i.e., at $B$. Keys built from an earlier head's output = **K-composition**.
    3. Its OV circuit copies $B$'s identity to the output → ⬆️ logit of $B$.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $x_t$: Token at position $t$ (non-bold, ❌input vector).
- Params:
    - $\mathbf{e}_v\in\mathbb{R}^d$: Embedding of token $v$.
    - $W_Q,W_K$: Query & key maps of the induction head.
    - $W_{OV}^{(1)}=W_O^{(1)}W_V^{(1)}$: OV map of the previous-token head.

Attention target:

$$
s^*\in\{s\le t:\ x_{s-1}=x_t\}\quad\Rightarrow\quad\text{raise the logit of}\ x_{s^*}
$$

K-composition (QK score of query position $t$ at key position $s$):

$$
\text{score}(t,s)\approx\big(W_Q\mathbf{e}_{x_t}\big)^T\big(W_KW_{OV}^{(1)}\mathbf{e}_{x_{s-1}}\big)
$$

- The previous-token head put $W_{OV}^{(1)}\mathbf{e}_{x_{s-1}}$ into the residual stream at $s$.
- Induction ⇔ $W_Q^TW_KW_{OV}^{(1)}$ scores matching token pairs high → max when $x_{s-1}=x_t$.
- Copying ⇔ OV circuit $W_UW_{OV}W_E$ has mostly positive eigenvalues → attending to $v$ ⬆️ logit of $v$.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn.functional as F

class InductionHead:
    ## hand-wired 2-layer circuit on one-hot tokens (no training) to expose the mechanism
    def __init__(self, vocab_size, sharpness=10.0):
        self.V, self.beta = vocab_size, sharpness

    def __call__(self, tokens):                          ## tokens: (T,)
        T = tokens.numel()
        E = F.one_hot(tokens, self.V).float()            ## (T, V) residual stream = token identity
        ## layer 1, previous-token head: position s receives the token at s-1
        prev = torch.zeros_like(E)
        prev[1:] = E[:-1]
        ## layer 2 QK (K-composition): query = current token x_t, key = previous token x_{s-1}
        scores = self.beta * (E @ prev.T)                ## (T, T): high iff x_{s-1} == x_t
        causal = torch.ones(T, T).tril().bool()
        attn = scores.masked_fill(~causal, float('-inf')).softmax(-1)
        ## layer 2 OV: copy the attended token's identity -> next-token prediction
        return attn @ E                                  ## (T, V)

## Example: [A][B] ... [A] -> [B]
tokens = torch.tensor([5, 2, 7, 1, 5])                  ## A = 5, B = 2
print(InductionHead(10)(tokens)[-1].argmax().item())    ## 2
```
````

```{attention} Q&A
:class: dropdown
*Why ≥ 2 layers?*
- 1 layer: the key at $s$ sees only $x_s$ (+ position), ❌$x_{s-1}$ → cannot express "attend to the token *after* $A$".
- An earlier head must first copy $x_{s-1}$ into position $s$ → composition.

*Evidence that induction heads drive in-context learning?*
- Induction heads form abruptly in a **phase change** early in training {cite:p}`olsson2022context`.
    - In the same window: in-context learning score (loss at token 500 − loss at token 50) improves sharply → a visible bump in the training loss curve.
    - Architecture changes that ease induction heads shift the phase change accordingly.
    - Causal (ablation) evidence is strong for small attention-only models. For large models, the evidence is mostly correlational.

*Only literal copying?*
- ❌. In large models, the same heads also do fuzzy matching: similar tokens, translated tokens, abstract patterns.

*Induction head = all of in-context learning?*
- ❌. Copy & complete is 1 mechanism. Learning a function from examples needs more.

*How do you detect one?*
- Repeated random token sequence → **prefix-matching score**: attention from $t$ to the token after the previous occurrence of $x_t$.
- **Copying score**: positive eigenvalues of the OV circuit.
```

&nbsp;

### Activation Patching
- **What**: Replace 1 internal activation in a run w/ its value from another run → measure the output change. {cite:p}`meng2022locating`
    - Meng et al. call it **causal tracing**. Origin contested: causal mediation analysis on LMs came earlier {cite:p}`vig2020investigating`.
- **Why**: Find which components (layer × position × head) *causally* carry the information for a behavior. Probes & attribution are only correlational.
- **How**:
    1. **Clean run**: prompt w/ the behavior ("The Eiffel Tower is in" → "Paris"). Cache all activations.
    2. **Corrupted run**: destroy the key information (replace the subject, or add noise to its embeddings) → "Paris" is lost.
    3. **Patch**: rerun the corrupted input. Overwrite 1 activation $(l,t)$ w/ its clean value.
    4. **Metric**: fraction of the clean output restored.
    5. Sweep over $(l,t)$ → heatmap of where the information lives.
    - Meng et al.: mid-layer MLPs at the last subject token restore factual recall most → basis for ROME model editing.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{x}_\text{clean},\mathbf{x}_\text{corr}$: Clean & corrupted inputs.
- Misc:
    - $\mathbf{h}_{l,t}$: Activation at layer $l$, position $t$ (or 1 head / MLP output).
    - $\mathcal{M}(\cdot)$: Scalar metric on the output, e.g., logit difference $S_\text{correct}-S_\text{wrong}$ or $P(\text{answer})$.

Definition:

$$
\mathcal{M}_\text{patch}(l,t)=\mathcal{M}\Big(f\big(\mathbf{x}_\text{corr}\ \big|\ \mathbf{h}_{l,t}\leftarrow\mathbf{h}_{l,t}(\mathbf{x}_\text{clean})\big)\Big)
$$

Normalized restoration:

$$
R(l,t)=\frac{\mathcal{M}_\text{patch}(l,t)-\mathcal{M}_\text{corr}}{\mathcal{M}_\text{clean}-\mathcal{M}_\text{corr}}
$$

- $R=1$: this activation alone restores the clean behavior. $R=0$: no effect.
- Causal tracing (w/ $\mathcal{M}=P(\text{answer})$): total effect $=\mathcal{M}_\text{clean}-\mathcal{M}_\text{corr}$. Indirect effect of $(l,t)$ $=\mathcal{M}_\text{patch}(l,t)-\mathcal{M}_\text{corr}$.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class ActivationPatcher:
    def __init__(self, model, module):
        self.model, self.module = model, module          ## module whose output gets patched

    @torch.no_grad()
    def __call__(self, x_clean, x_corr, idx):
        cache = {}
        ## 1. clean run: cache the module output
        hook = self.module.register_forward_hook(lambda m, i, o: cache.update(clean=o.clone()))
        self.model(x_clean)
        hook.remove()

        ## 2. corrupted run: overwrite only the slice idx w/ its clean value
        def patch(m, i, o):
            o = o.clone()
            o[idx] = cache['clean'][idx]
            return o                                     ## returning a tensor replaces the output
        hook = self.module.register_forward_hook(patch)
        out = self.model(x_corr)
        hook.remove()
        return out

## Example
torch.manual_seed(0)
model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
x_clean, x_corr = torch.randn(1, 4), torch.randn(1, 4)
metric = lambda out: (out[0, 0] - out[0, 1]).item()     ## logit difference
with torch.no_grad():
    m_clean, m_corr = metric(model(x_clean)), metric(model(x_corr))
patcher = ActivationPatcher(model, model[1])            ## patch the hidden ReLU layer
for idx in [(slice(None), slice(0, 4)), (slice(None), slice(None))]:   ## half the neurons, all neurons
    m = metric(patcher(x_clean, x_corr, idx))
    print((m - m_corr) / (m_clean - m_corr))            ## half: partial; all: 1.0
```
````

```{attention} Q&A
:class: dropdown
*Denoising vs noising?*
- **Denoising** (clean → corrupted run): is this component *sufficient* to restore the behavior?
- **Noising** (corrupted → clean run): is it *necessary*?
- Answers differ under redundancy: backup components take over when 1 is knocked out.

*How should you corrupt?*
- Gaussian noise on embeddings → off-distribution activations → can exaggerate or misplace effects.
- Token replacement ("Eiffel Tower" → "Colosseum") → stays in-distribution & changes only the target fact.

*Which metric?*
- Probability saturates & is nonlinear → hides small changes.
- Logit difference is linear in each component's direct contribution to the residual stream (up to the final LayerNorm scale) → preferred.

*Pitfalls?*
- Patching 1 component at a time misses distributed or redundant mechanisms.
- The component w/ the largest restoration is not necessarily the best place to edit or intervene.

*Cost & shortcut?*
- 1 forward pass per $(l,t)$ → expensive for large sweeps.
- **Attribution patching**: 1st-order approximation $\Delta\mathcal{M}\approx(\mathbf{h}_\text{clean}-\mathbf{h}_\text{corr})^T\nabla_\mathbf{h}\mathcal{M}|_\text{corr}$ → all components from 1 backward pass.
    - = gradient × input on activations. [IG](#integrated-gradients) = its path-integrated version.

*Patching vs probing?*
- Probe: is the information *present*? Patching: does the model *use* it here?
```

&nbsp;

### Activation Steering
- **What**: Add a fixed vector to a layer's activations at inference to shift the model's behavior. {cite:p}`subramani2022extracting`
    - Origin contested. Subramani et al. = earliest LM steering vectors cited here, found by gradient optimization per target sentence on a frozen GPT-2. Optimization-free contrastive variants came later → [ActAdd](#actadd), [CAA](#caa), [RepE](#repe).
- **Why**: Control behavior (sentiment, honesty, refusal, sycophancy) w/o finetuning or prompt changes.
    - Also a causal test of the [LRH](#linear-representation-hypothesis): adding a concept direction changes the concept → the model uses that direction.
- **How**:
    1. Get a steering vector $\mathbf{v}$ at layer $l$:
        - **Optimized**: gradient descent on $\mathbf{v}$ (model frozen) until the model emits a target.
        - **Contrastive**: activation difference between positive & negative inputs ("Love" vs "Hate").
    2. Inference: $\mathbf{h}_{l,t}\leftarrow\mathbf{h}_{l,t}+\alpha\mathbf{v}$ at chosen positions.
    3. Tune layer $l$ & coefficient $\alpha$. Too small → no effect. Too large → incoherent text.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathcal{P}=\{(\mathbf{x}^+,\mathbf{x}^-)\}$: Contrast pairs.
- Hyperparams:
    - $\alpha$: Steering coeff. $\alpha<0$ → push away from the concept.

Contrastive steering vector:

$$
\mathbf{v}=\frac{1}{|\mathcal{P}|}\sum_{(\mathbf{x}^+,\mathbf{x}^-)\in\mathcal{P}}\Big(\mathbf{h}_l(\mathbf{x}^+)-\mathbf{h}_l(\mathbf{x}^-)\Big)
$$

Inference:

$$
\tilde{\mathbf{h}}_{l,t}=\mathbf{h}_{l,t}+\alpha\mathbf{v}
$$

Directional ablation (remove the concept):

$$
\tilde{\mathbf{h}}_{l,t}=\mathbf{h}_{l,t}-\hat{\mathbf{v}}\hat{\mathbf{v}}^T\mathbf{h}_{l,t},\qquad\hat{\mathbf{v}}=\mathbf{v}/\|\mathbf{v}\|
$$
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class SteeringVector:
    def __init__(self, module):
        self.module, self.v, self.handle = module, None, None

    def fit(self, h_pos, h_neg):                 ## (P, d) activations at layer l on contrast pairs
        self.v = (h_pos - h_neg).mean(0)         ## mean difference: ActAdd = 1 pair, CAA = many
        return self

    def attach(self, alpha):
        ## a forward hook that returns a tensor replaces the output: h <- h + alpha * v
        self.handle = self.module.register_forward_hook(lambda m, i, o: o + alpha * self.v)

    def detach(self):
        self.handle.remove()

## Example
torch.manual_seed(0)
model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
x_pos, x_neg = torch.randn(16, 4) + 1, torch.randn(16, 4) - 1   ## contrast inputs ("Love" vs "Hate")
with torch.no_grad():
    sv = SteeringVector(model[1]).fit(model[:2](x_pos), model[:2](x_neg))   ## v in R^8
    x = torch.randn(1, 4)
    before = model(x)
    sv.attach(alpha=2.0)
    after = model(x)
    sv.detach()
print(after - before)   ## = alpha * W_2 v: same shift for every x (linear readout here)
```
````

```{dropdown} Table: Steering Variants
| Method | Vector | Where added |
|:--|:--|:--|
| Latent steering vector | Optimized per target sentence | Hidden states; 1st or every timestep |
| [ActAdd](#actadd) | Difference of 1 prompt pair, per position | User-prompt residual stream at layer $l$, aligned at the front |
| [CAA](#caa) | Mean difference over many A/B pairs, at the answer-letter position | All positions after the user prompt, $\pm$ multiplier |
| [RepE](#repe) | Reading vector (1st PC of activation differences) or contrast vector | Monitoring (read) & control (write) |
```

```{attention} Q&A
:class: dropdown
*Why does it work?*
- LRH: concept ≈ direction → adding $\alpha\mathbf{v}$ moves $\mathbf{h}$ along the concept.
- Other features ≈ nearly orthogonal → mostly unaffected.

*Steering vs prompting vs finetuning?*
- ✅ ❌Training, cheap, adjustable strength, composable, reversible at inference.
- ❌ Brittle: layer & $\alpha$ need tuning. Large $\alpha$ hurts fluency. Off-target side effects. Effect varies across inputs.

*Mean difference vs probe direction?*
- Probe weights maximize separability → can exploit correlated low-variance directions → good for reading, ❌necessarily for writing.
- Mean difference = the direction the concept actually shifts activations.
- LRH view: the right writing direction = embedding representation $\bar{\boldsymbol\lambda}_Z$, ❌the probe-like $\bar{\boldsymbol\gamma}_Z$. They coincide only under the causal inner product.

*Relation to SAEs?*
- Clamping SAE latent $i$ = steering along its decoder direction $\mathbf{d}_i$.

*Relation to activation patching?*
- Patching copies a whole activation from another run. Steering adds 1 direction to every run.
```

&nbsp;

#### ActAdd
- **Name**: Activation Addition {cite:p}`turner2023steering`
- **What**: Steering vector = activation difference of 1 prompt pair. ❌Optimization.
- **Why**: Optimized steering vectors need gradient descent per target → slow iteration.
    - ActAdd: 2 forward passes → rapid trial & error.
- **How**:
    1. Pad the prompt pair $(p_+,p_-)$ (e.g., "Love", "Hate") to the same token length.
    2. Forward both → $\mathbf{h}_+^l-\mathbf{h}_-^l$: 1 difference vector per position.
    3. Scale by the injection coeff $c$. Add to the user prompt's residual stream at layer $l$, aligned at the 1st token.
    4. Continue the forward pass → steered generation.

```{attention} Q&A
:class: dropdown
*Pros?*
- 1 pair of data points. ❌Gradients. ❌Weight change.
- Preserves performance on off-target tasks in the original evaluations.

*Cons?*
- 1 pair → the vector carries prompt-specific content, ❌only the concept.
- Extra hyperparams: layer $l$, coeff $c$, alignment position.
- Intervening on the very last residual stream position reliably breaks syntax.
```

&nbsp;

#### CAA
- **Name**: Contrastive Activation Addition {cite:p}`rimsky2024steering`
- **What**: Mean activation difference over many multiple-choice contrast pairs, added at all positions after the user prompt.
- **Why**: 1 pair (ActAdd) → noisy. Averaging over many pairs → prompt-specific content cancels, the behavior direction stays.
- **How**:
    1. Pairs: the same A/B question, ending w/ the answer letter of the behavior vs its opposite.
    2. Cache the residual stream at layer $l$, at the answer-letter position.
    3. Mean difference → $\mathbf{v}_\text{MD}$.
    4. Inference: add $\pm$ multiplier $\times\mathbf{v}_\text{MD}$ at every position after the instruction.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $(\mathbf{p},\mathbf{c}_p,\mathbf{c}_n)\in\mathcal{D}$: Prompt, positive answer letter, negative answer letter.
- Misc:
    - $\mathbf{h}_l(\mathbf{p},\mathbf{c})$: Residual stream at layer $l$, at the answer-letter position of prompt $\mathbf{p}$ + answer $\mathbf{c}$.

Steering vector:

$$
\mathbf{v}_\text{MD}=\frac{1}{|\mathcal{D}|}\sum_{(\mathbf{p},\mathbf{c}_p,\mathbf{c}_n)\in\mathcal{D}}\Big(\mathbf{h}_l(\mathbf{p},\mathbf{c}_p)-\mathbf{h}_l(\mathbf{p},\mathbf{c}_n)\Big)
$$
```

```{attention} Q&A
:class: dropdown
*Why the answer-letter position?*
- Same question in both prompts → the only difference = the chosen answer → the activation difference isolates the behavior.

*Pros?*
- On Llama 2 Chat: steers behaviors such as sycophancy & hallucination, works on top of finetuning & system prompts, minimal capability loss.

*Cons?*
- Needs a dataset of contrast pairs per behavior.
- Same layer & multiplier tuning as every steering method. Large multipliers → ⬇️fluency.
```

&nbsp;

#### RepE
- **Name**: Representation Engineering {cite:p}`zou2023representation`
- **What**: Top-down transparency: read & control population-level representations of high-level concepts (honesty, power-seeking). ❌Neurons, ❌circuits.
- **Why**: Bottom-up analysis (neurons, circuits) struggles to reach high-level cognitive phenomena → study concept directions directly.
- **How**:
    - **Reading** (Linear Artificial Tomography, LAT):
        1. Design stimuli that elicit the concept.
        2. Collect activations.
        3. PCA on differences of activation pairs → 1st PC = **reading vector** $\mathbf{v}$. Score: $\mathbf{h}^T\mathbf{v}$.
    - **Control**:
        - Reading vector: add/subtract $\mathbf{v}$. Same perturbation for every input.
        - **Contrast vector**: run the *same* input w/ 2 contrastive prompts → difference = input-dependent controller. Stronger baseline.
        - **LoRRA** (Low-Rank Representation Adaptation): fine-tune low-rank adapters so representations move toward a target (e.g., the contrast vector).

```{attention} Q&A
:class: dropdown
*RepE vs mechanistic interpretability?*
- Mech interp: bottom-up, from components to behavior → [Circuit](#circuit).
- RepE: top-down, from a concept to its representation → coarser, but scales to abstract concepts.

*Why PCA on differences, ❌on raw activations?*
- Differences cancel content shared by both stimuli → the top PC tracks the concept.

*Cons?*
- Reading vector ≈ a probe → same caveat as the [Linear Probe](#linear-probe): decodable ≠ used.
- Stimulus design decides what the vector captures → confounds.
```

&nbsp;
