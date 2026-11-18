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
# Energy-Based Models
Models that assign a scalar energy to each configuration: low energy = likely. Covers classical energy networks & training w/o the partition function. Score-based training & annealed sampling → [Score-Based View](diffusion.md#score-based-view).

Notations:
- $x$: Data sample ($\mathbb{R}^d$ or $\{0,1\}^d$)
- $d$: Data dim
- $\theta$: Params
- $E_\theta(x)$: Energy
- $Z_\theta$: Partition function (normalizer)
- $p_\theta(x)=e^{-E_\theta(x)}/Z_\theta$: Model density
- $p_\text{data}$: Data distribution
- $N$: #units (Hopfield, Boltzmann)
- $W$: Symmetric weight matrix, entries $w_{ij}$, $w_{ii}=0$
- $b$: Bias vector
- $\sigma(\cdot)$: Logistic sigmoid
- $\langle\cdot\rangle_\text{data}$: Expectation w/ visible units clamped to data
- $\langle\cdot\rangle_\text{model}$: Expectation under the free-running model

Overrides of global notation: $x$, $b$ = whole vectors (non-bold). $i,j$ = unit idx, ❌sample/feature idx.

&nbsp;

## EBM
- **Name**: Energy-Based Model {cite:p}`lecun2006a`
- **What**: Unnormalized density $p_\theta(x)\propto\exp(-E_\theta(x))$ w/ any scalar function $E_\theta$.
- **Why**: Flexibility.
    - Any scalar NN w/ integrable $e^{-E_\theta}$ (finite $Z_\theta$) = valid model. ❌Invertibility (flows), ❌encoder (VAE), ❌normalized outputs.
    - Composable: sum of energies = product of densities → AND of constraints.
- **How**:
    1. Learn $E_\theta$: low on data, high elsewhere.
    2. **Inference**: minimize energy over the unknown variables, e.g., $\hat y=\arg\min_yE_\theta(x,y)$.
    3. **Sampling**: MCMC on $E_\theta$, e.g., [Langevin Dynamics](#langevin-dynamics).
    4. **Training**: push $E$ down on data, push $E$ up on model samples ← MLE gradient.

```{note} Math
:class: dropdown
Definition:

$$
p_\theta(x)=\frac{\exp\big(-E_\theta(x)\big)}{Z_\theta},\qquad Z_\theta=\int\exp\big(-E_\theta(x)\big)\,dx
$$

- Discrete $x$: $\int\to\sum$ over all configurations.
- $Z_\theta$: integral over all of $\mathbb{R}^d$ (or $2^d$ terms) → intractable for an NN energy.
- Temperature $T$: $p\propto\exp(-E/T)$. $T$⬆️ → flatter.

MLE gradient:

$$
\nabla_\theta\left[-\log p_\theta(x)\right]=\nabla_\theta E_\theta(x)-\mathbb{E}_{x'\sim p_\theta}\left[\nabla_\theta E_\theta(x')\right]
$$

- **Positive phase** (1st term): lowers $E$ at data.
- **Negative phase** (2nd term): raises $E$ at model samples → needs samples from $p_\theta$ → MCMC.

Score:

$$
\nabla_x\log p_\theta(x)=-\nabla_xE_\theta(x)
$$

- ← $Z_\theta$ constant in $x$.
```

```{tip} Derivation
:class: dropdown
*Why does $\nabla_\theta\log Z_\theta$ become a model expectation?*
1. $\nabla_\theta\log Z_\theta=\dfrac{1}{Z_\theta}\displaystyle\int\nabla_\theta e^{-E_\theta(x)}\,dx=\dfrac{1}{Z_\theta}\displaystyle\int-\nabla_\theta E_\theta(x)\,e^{-E_\theta(x)}\,dx$.
2. $e^{-E_\theta}/Z_\theta=p_\theta$ → $\nabla_\theta\log Z_\theta=-\mathbb{E}_{p_\theta}[\nabla_\theta E_\theta]$.
3. $-\log p_\theta(x)=E_\theta(x)+\log Z_\theta$ → differentiate → MLE gradient above.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class EBMLoss(nn.Module):
    def __init__(self, energy):
        super().__init__()
        self.E = energy

    def forward(self, x_data, x_model):
        ## grad = E_data[dE/dθ] - E_model[dE/dθ]; log Z never computed
        ## x_model: MCMC samples -> detach, the sampler is not differentiated
        return self.E(x_data).mean() - self.E(x_model.detach()).mean()

## Example: Gaussian EBM E = theta * x^2 / 2 (precision theta), exact model samples
theta = torch.tensor(1.0, requires_grad=True)
E = lambda x: 0.5 * theta * x.pow(2).sum(-1)
x_data = 2 * torch.randn(100000, 1)                  ## true precision 0.25
x_model = torch.randn(100000, 1) / theta.sqrt()      ## samples from p_theta
EBMLoss(E)(x_data, x_model).backward()
print(theta.grad)   ## ≈ 1.5 = E_data[x^2]/2 - 1/(2 theta) > 0 -> theta ⬇️ -> model widens
```
````

```{dropdown} Table: Training EBMs
| Method | Avoids $Z$ by | Needs | Weakness |
|:--|:--|:--|:--|
| MLE + MCMC | Sampling the negative phase | Samples from $p_\theta$ | Slow mixing → biased grads |
| [Contrastive Divergence](#contrastive-divergence) | Short chains started at data | $k$ MCMC steps | Biased; spurious modes far from data survive |
| [Score Matching](diffusion.md#score-matching) | Matching $\nabla_x\log p$ | Differentiable $x$; Jacobian trace or noise ([DSM](diffusion.md#dsm)) | ❌Discrete data; blind to mode weights |
| [NCE](#nce) | Classifying data vs noise | Known noise density | Weak signal if noise is far from data |
```

```{attention} Q&A
:class: dropdown
*Why is $Z_\theta$ intractable?*
- Sum over $2^d$ configurations, or an integral over $\mathbb{R}^d$. No closed form for an NN energy.

*What can you do w/o $Z_\theta$?*
- Compare 2 inputs: $p(x_1)/p(x_2)=e^{E(x_2)-E(x_1)}$.
- MAP inference: $\arg\min E$.
- MCMC sampling: uses only ratios or $\nabla_xE$.
- Score: $-\nabla_xE$.
- ❌Absolute likelihood → ❌direct model comparison by NLL.

*Pros?*
- Most flexible density model: any architecture.
- Composable: add energies of separate models → conjunction of concepts.
- MLE-based → mode covering.

*Cons?*
- Sampling = MCMC → slow, mixing problems.
- Training unstable; negative-phase quality limits the model.
- ❌Exact likelihood.

*Is a softmax classifier an EBM?*
- Yes, over labels: $E(x,y)=-\text{logit}_y(x)$. $Z(x)=\sum_y e^{\text{logit}_y}$ → tractable ← few labels.
- EBM hardness appears when the variable being modeled is high-dim ($x$ itself).

*How do EBMs relate to diffusion?*
- A score model $s_\theta(x)$ = $-\nabla_xE$ if it is a conservative field.
- Diffusion learns scores at many noise levels by regression → ❌$Z$, ❌MCMC in training → [Score Matching](diffusion.md#score-matching), [NCSN](diffusion.md#ncsn).
```

&nbsp;

## Hopfield Network
- **What**: Recurrent net of binary units w/ symmetric weights; asynchronous updates descend an energy → stored patterns = local minima. {cite:p}`hopfield1982neural`
- **Why**: Content-addressable (associative) memory: partial or noisy cue → full stored pattern.
- **How**:
    1. **Store** (Hebbian): $w_{ij}\propto\sum_\mu\xi^\mu_i\xi^\mu_j$ → each pattern ≈ local energy minimum. Exact minimum only at low load.
    2. **Recall**: set the state to the cue. Update 1 unit at a time: $s_i\leftarrow\text{sign}(\sum_jw_{ij}s_j+b_i)$.
    3. Energy never increases → converges to a fixed point (attractor) ≈ nearest stored pattern.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $s\in\{-1,+1\}^N$: State.
- Params:
    - $\xi^\mu\in\{-1,+1\}^N$: $\mu$-th stored pattern.
- Misc:
    - $P$: #stored patterns.
    - $h_i$: Local field of unit $i$.

Energy:

$$
E(s)=-\frac12s^TWs-b^Ts=-\frac12\sum_{i,j}w_{ij}s_is_j-\sum_ib_is_i
$$

Hebbian storage:

$$
W=\frac1N\sum_{\mu=1}^{P}\xi^\mu\xi^{\mu T}-\frac PNI
$$

- $-\frac PNI$: zeroes the diagonal → $w_{ii}=0$.

Process (asynchronous, 1 unit $i$ at a time):

$$
h_i=\sum_jw_{ij}s_j+b_i,\qquad s_i\leftarrow\begin{cases}+1&h_i>0\\-1&h_i<0\\s_i&h_i=0\end{cases}
$$

Capacity (random patterns, Hebbian, $N\to\infty$):

$$
P_\text{max}\approx0.14\,N
$$

- Critical load for associative retrieval {cite:p}`amit1985storing`. Below it: attractors w/ a small fraction of wrong bits near each pattern, ❌exact patterns. Hopfield's simulations: ~$0.15N$.
- Exact recall of most patterns: $P\le N/(2\ln N)$. Of every pattern: $P\le N/(4\ln N)$ (McEliece et al., 1987).
```

```{tip} Derivation
:class: dropdown
*Why does energy never increase under asynchronous updates?*
1. Collect the terms of $E$ that contain $s_i$. Use $w_{ij}=w_{ji}$ & $w_{ii}=0$: $E=-s_ih_i+(\text{terms w/o }s_i)$.
2. Flip $s_i\to s_i'$: $\Delta E=-(s_i'-s_i)\,h_i$.
3. Update rule → $s_i'$ has the sign of $h_i$ → $(s_i'-s_i)$ is $0$ or has the sign of $h_i$ → $\Delta E\le0$.
4. $E$ bounded below & finite #states → no infinite descent → reaches a fixed point.

*Why are stored patterns stable?*
1. Set $s=\xi^\nu$, $b=0$: $h_i=\frac1N\sum_\mu\xi^\mu_i\sum_{j\ne i}\xi^\mu_j\xi^\nu_j$.
2. Split $\mu=\nu$ from $\mu\ne\nu$: $h_i=\frac{N-1}{N}\xi^\nu_i+C_i$.
3. Crosstalk $C_i$: sum of ~$(P-1)(N-1)$ random $\pm\frac1N$ terms → mean 0, variance ≈ $P/N$.
4. Stable if $|C_i|<1$ for all $i$ → signal-to-noise ≈ $\sqrt{N/P}$ → capacity $\propto N$.
5. Exact constant $0.14$ needs the full statistical-mechanics analysis: errors cascade through the dynamics.
```

````{important} Code
:class: dropdown
```python
import torch

class Hopfield:
    def __init__(self, patterns):                  ## (P, N) in {-1, +1}
        N = patterns.shape[1]
        self.W = patterns.T @ patterns / N         ## Hebbian: units that agree -> positive weight
        self.W.fill_diagonal_(0)                   ## w_ii = 0 -> energy descent guaranteed

    def energy(self, s):
        return -0.5 * s @ self.W @ s

    def recall(self, s, n_sweeps=5):
        s = s.clone()
        for _ in range(n_sweeps):
            for i in torch.randperm(len(s)):       ## asynchronous: 1 unit at a time
                h = self.W[i] @ s
                if h != 0:
                    s[i] = torch.sign(h)           ## each flip: ΔE <= 0
        return s

## Example: 3 patterns in 100 units (P/N = 0.03 << 0.14), cue = pattern 0 w/ 8 bits flipped
torch.manual_seed(0)
xi = torch.randint(0, 2, (3, 100)).float() * 2 - 1
net = Hopfield(xi)
cue = xi[0].clone()
cue[:8] *= -1
out = net.recall(cue)
print((out == xi[0]).all(), net.energy(cue) > net.energy(out))   ## True True
```
````

```{attention} Q&A
:class: dropdown
*Why symmetric $W$ & $w_{ii}=0$?*
- The descent proof needs both.
- Asymmetric $W$ → ❌energy function → cycles & chaotic dynamics possible.

*Why asynchronous updates?*
- Synchronous (all units at once) → descent proof fails → can oscillate in 2-cycles.

*What are spurious states?*
- Sign-flipped patterns $-\xi^\mu$ ← $E(s)=E(-s)$ at $b=0$.
- Mixtures of an odd # of patterns, e.g., $\text{sign}(\xi^1+\xi^2+\xi^3)$.
- Spin-glass states unrelated to any pattern (high load).

*What happens past capacity?*
- Correlated patterns → capacity ≪ $0.14N$.
- Load $P/N$ past the critical value → retrieval states vanish → recall of all patterns fails, ❌graceful decay.

*Hopfield vs Boltzmann machine?*
- Hopfield = deterministic, ❌hidden units → memory.
- [Boltzmann Machine](#boltzmann-machine) = stochastic version w/ hidden units → generative model. Hopfield ≈ its $T\to0$ limit w/o hidden units.

*What does this tell us about AC?*
- Attractor dynamics = model of memory & pattern completion in recurrent brain circuits.
- Capacity ∝ $N$ bounds the information a Hopfield-like system can hold → used in the integration paradox → [Consciousness as a State of Matter](../../ac/theories.md#consciousness-as-a-state-of-matter).
```

&nbsp;

### Modern Hopfield Network
- **What**: Continuous-state Hopfield net w/ log-sum-exp energy; 1 update = softmax attention over stored patterns. {cite:p}`ramsauer2021hopfield`
- **Why**: Classical capacity ∝ $N$ & many spurious states.
    - Exponential interaction → sharp energy wells → capacity exponential in the state dim; retrieval in 1 update.
    - Continuous state → differentiable → usable as a deep-net layer.
- **How**:
    1. Store $P$ patterns as columns of $X$.
    2. Similarity of state $\xi$ to each pattern: $X^T\xi$.
    3. Softmax w/ inverse temperature $\beta$ → weights over patterns.
    4. New state = weighted average of patterns. $\beta$⬆️ → nearest pattern.
    5. Query ← state, keys = values ← stored patterns, $\beta=1/\sqrt{d_k}$ → [Scaled Dot-Product Attention](../../nlp/attn.md#scaled-dot-product-attention).

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\xi\in\mathbb{R}^d$: State (query).
- Params:
    - $X=(x^1,\dots,x^P)\in\mathbb{R}^{d\times P}$: Stored patterns as columns.
- Hyperparams:
    - $\beta>0$: Inverse temperature.
- Misc:
    - $P$: #stored patterns.
    - $\text{lse}(\beta,\mathbf{v})=\beta^{-1}\log\sum_{\mu=1}^{P}\exp(\beta v_\mu)$: Log-sum-exp.
    - $M=\max_\mu\|x^\mu\|$: Largest pattern norm.

Energy:

$$
E(\xi)=-\text{lse}(\beta,X^T\xi)+\frac12\xi^T\xi+\beta^{-1}\log P+\frac12M^2
$$

- Last 2 terms: constants → $E\ge0$.

Process (update):

$$
\xi^\text{new}=X\,\text{softmax}\left(\beta X^T\xi\right)
$$

- Converges; $E(\xi^\text{new})\le E(\xi)$.

Attention form (row vectors, many queries):

$$
\text{softmax}\left(\beta\,QK^T\right)V,\qquad Q\leftarrow\xi^T,\ K=V\leftarrow X^T
$$

- Transformer: separate projections for $Q,K,V$ & $\beta=1/\sqrt{d_k}$.
```

```{tip} Derivation
:class: dropdown
*Why is the update a guaranteed energy descent?*
1. Split $E$: $\frac12\xi^T\xi$ is convex. $-\text{lse}$ is concave ← lse is convex.
2. Concave-convex procedure (CCCP): set the gradient of the convex part at $\xi^\text{new}$ equal to minus the gradient of the concave part at $\xi$.
3. $\nabla_\xi\tfrac12\xi^T\xi=\xi$. $\nabla_\xi\,\text{lse}(\beta,X^T\xi)=X\,\text{softmax}(\beta X^T\xi)$.
4. → $\xi^\text{new}=X\,\text{softmax}(\beta X^T\xi)$.
5. CCCP minimizes a convex upper bound of $E$ that touches $E$ at $\xi$ → $E$ never increases.
```

````{important} Code
:class: dropdown
```python
import torch

## Reuses ScaledDotProductAttention from nlp/attn.md
class ModernHopfield:
    def __init__(self, X, beta):                   ## X: (P, d) stored patterns as rows
        self.X, self.beta = X, beta

    def update(self, xi):                          ## xi: (B, d)
        ## similarity -> softmax over patterns -> weighted average of patterns
        return torch.softmax(self.beta * xi @ self.X.T, dim=-1) @ self.X

    def energy(self, xi):                          ## constants dropped
        return -torch.logsumexp(self.beta * xi @ self.X.T, dim=-1) / self.beta + 0.5 * (xi ** 2).sum(-1)

## Example: 8 random patterns in d = 64, noisy query near pattern 2
torch.manual_seed(0)
X = torch.randn(8, 64)
xi = X[2:3] + 0.3 * torch.randn(1, 64)
net = ModernHopfield(X, beta=2.0)
out = net.update(xi)
print(torch.allclose(out, X[2:3], atol=1e-3), net.energy(xi) > net.energy(out))   ## True tensor([True])
## 1 update w/ beta = 1/sqrt(d) == attention w/ query = xi, keys = values = X
attn = ScaledDotProductAttention()(xi, X, X)
print(torch.allclose(ModernHopfield(X, beta=64 ** -0.5).update(xi), attn, atol=1e-6))   ## True
```
````

```{attention} Q&A
:class: dropdown
*Is attention = Hopfield retrieval?*
- Same update formula: 1 attention step = 1 update w/ the query as state.
- Differences in transformers: $K\ne V$ (separate projections) → value ≠ stored pattern. 1 step, ❌iteration to a fixed point. Patterns = current context, ❌fixed memory.
- → An interpretation of attention, ❌a claim that transformers run associative-memory dynamics.

*Role of $\beta$?*
- $\beta$⬆️ → softmax ≈ one-hot → retrieve 1 pattern.
- $\beta$⬇️ → near-uniform weights → fixed point ≈ average of many patterns (metastable state).

*Why log-sum-exp?*
- $\text{lse}\approx\max$ for large $\beta$ → each pattern gets a sharp well → little crosstalk.
- Classical quadratic energy = sum of squared overlaps → crosstalk from all patterns adds up.

*Predecessors?*
- Dense associative memory (Krotov & Hopfield): polynomial interaction → capacity $\propto N^{n-1}$ for degree $n$.
- Demircigil et al.: exponential interaction → exponential capacity. Both binary states.
- Ramsauer et al.: continuous states → differentiable attention-like update.
```

&nbsp;

## Boltzmann Machine
- **What**: Stochastic Hopfield-style network w/ hidden units that samples states from $p(s)\propto e^{-E(s)}$. {cite:p}`ackley1985a`
- **Why**: Hopfield = deterministic, ❌hidden units → stores patterns, ❌models a distribution.
    - Hidden units → latent features & higher-order correlations among visible units.
    - Stochastic units → generative probability model w/ a local learning rule.
- **How**:
    1. Units $s=(v,h)$, binary. Energy as in Hopfield.
    2. **Sample**: Gibbs. Turn unit $i$ on w/ prob $\sigma(\text{local field}/T)$.
    3. **Learn**: $\Delta w_{ij}\propto\langle s_is_j\rangle_\text{data}-\langle s_is_j\rangle_\text{model}$.
        - Data phase: clamp visibles to a data vector, sample hiddens.
        - Model phase: sample all units freely.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $s=(v,h)\in\{0,1\}^N$: State.
    - $v$: Visible units.
    - $h$: Hidden units.
- Hyperparams:
    - $T$: Temperature.

Model:

$$
E(s)=-\frac12s^TWs-b^Ts,\qquad p(s)=\frac{e^{-E(s)/T}}{Z},\qquad p(v)=\sum_hp(v,h)
$$

Inference (Gibbs update of unit $i$):

$$
p(s_i=1\,|\,s_{-i})=\sigma\left(\frac{\sum_jw_{ij}s_j+b_i}{T}\right)
$$

- $s_{-i}$: All units except $i$.

Training ($T=1$):

$$
\frac{\partial\log p(v)}{\partial w_{ij}}=\langle s_is_j\rangle_\text{data}-\langle s_is_j\rangle_\text{model}
$$
```

```{tip} Derivation
:class: dropdown
*Why a sigmoid for the Gibbs update?*
1. Terms of $E$ that contain $s_i$: $-s_i\left(\sum_jw_{ij}s_j+b_i\right)$ ← symmetric $W$, $w_{ii}=0$.
2. Energy gap: $\Delta E_i=E(s_i=0)-E(s_i=1)=\sum_jw_{ij}s_j+b_i$.
3. $\dfrac{p(s_i=1|s_{-i})}{p(s_i=0|s_{-i})}=e^{\Delta E_i/T}$ → $p(s_i=1|s_{-i})=\sigma(\Delta E_i/T)$.

*Why the "data minus model" learning rule?*
1. Apply the [EBM](#ebm) MLE gradient to $p(v)$ w/ latent $h$: $\nabla\log p(v)=\mathbb{E}_{h|v}[-\nabla E]-\mathbb{E}_{p(v,h)}[-\nabla E]$.
2. $\partial E/\partial w_{ij}=-s_is_j$.
3. → $\langle s_is_j\rangle_\text{data}-\langle s_is_j\rangle_\text{model}$.
```

```{attention} Q&A
:class: dropdown
*Why is it hard to train?*
- Both phases need equilibrium samples. Fully connected → Gibbs updates 1 unit at a time → slow mixing.
- Correlations estimated from noisy samples → noisy grads. The original paper uses simulated annealing to reach equilibrium.
- → [RBM](#rbm) restricts the graph.

*Why is the learning rule appealing?*
- Local: $\Delta w_{ij}$ uses only stats of units $i$ & $j$.
- Hebbian (data phase) − anti-Hebbian (model phase). Often read as "wake" & "sleep".

*Where does the name come from?*
- $p\propto e^{-E/T}$ = Boltzmann distribution of statistical mechanics.
```

&nbsp;

### RBM
- **Name**: Restricted Boltzmann Machine {cite:p}`smolensky1986information`
- **What**: Boltzmann machine on a bipartite graph: visible–hidden edges only, ❌within-layer edges.
- **Why**: Full Boltzmann machine → sequential, slow Gibbs sampling.
    - Bipartite → units in 1 layer conditionally independent given the other layer → **block Gibbs**: 2 parallel steps per sweep.
    - Data phase exact: $p(h|v)$ factorizes.
- **How**:
    1. $p(h|v)$: each $h_j$ = independent Bernoulli w/ prob $\sigma(b_j+\sum_iv_iw_{ij})$.
    2. $p(v|h)$: same form for each $v_i$.
    3. Block Gibbs: $v\to h\to v\to\dots$, all units of a layer at once.
    4. Train w/ [Contrastive Divergence](#contrastive-divergence).

```{note} Math
:class: dropdown
Notations:
- IO:
    - $v\in\{0,1\}^{d_v}$: Visible units.
    - $h\in\{0,1\}^{d_h}$: Hidden units.
- Params:
    - $W\in\mathbb{R}^{d_v\times d_h}$: Visible–hidden weights. Overrides the page-level square $W$.
    - $a\in\mathbb{R}^{d_v}$: Visible bias.
    - $b\in\mathbb{R}^{d_h}$: Hidden bias.

Model:

$$
E(v,h)=-a^Tv-b^Th-v^TWh
$$

Inference (conditionals):

$$
p(h_j=1\,|\,v)=\sigma\Big(b_j+\sum_iv_iw_{ij}\Big),\qquad p(v_i=1\,|\,h)=\sigma\Big(a_i+\sum_jw_{ij}h_j\Big)
$$

Free energy:

$$
F(v)=-a^Tv-\sum_{j=1}^{d_h}\log\Big(1+\exp\big(b_j+\textstyle\sum_iv_iw_{ij}\big)\Big),\qquad p(v)=\frac{e^{-F(v)}}{Z}
$$

Training:

$$
\frac{\partial\log p(v)}{\partial w_{ij}}=\langle v_ih_j\rangle_\text{data}-\langle v_ih_j\rangle_\text{model}
$$

- Data term exact: $\langle v_ih_j\rangle_\text{data}=v_i\,p(h_j=1|v)$.
- Model term: sum over $2^{d_v+d_h}$ states → intractable → [CD](#contrastive-divergence).
```

```{tip} Derivation
:class: dropdown
*Why are hidden units conditionally independent given $v$?*
1. Fix $v$: $-E=a^Tv+\sum_jh_j\big(b_j+\sum_iv_iw_{ij}\big)$ → a sum of separate terms per $h_j$.
2. $e^{-E}$ → product over $j$ → $p(h|v)=\prod_jp(h_j|v)$.
3. Each factor: $p(h_j=1|v)=\dfrac{e^{c_j}}{1+e^{c_j}}=\sigma(c_j)$, $c_j=b_j+\sum_iv_iw_{ij}$.

*Why does the free energy have a closed form?*
1. $\sum_he^{-E(v,h)}=e^{a^Tv}\prod_j\sum_{h_j\in\{0,1\}}e^{h_jc_j}=e^{a^Tv}\prod_j(1+e^{c_j})$.
2. $F(v)=-\log\sum_he^{-E(v,h)}$ → $-a^Tv-\sum_j\log(1+e^{c_j})$.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class RBM(nn.Module):
    def __init__(self, d_v, d_h):
        super().__init__()
        self.W = nn.Parameter(0.01 * torch.randn(d_v, d_h))
        self.a = nn.Parameter(torch.zeros(d_v))          ## visible bias
        self.b = nn.Parameter(torch.zeros(d_h))          ## hidden bias

    def p_h(self, v):                                    ## all h_j independent given v -> 1 matmul
        return torch.sigmoid(v @ self.W + self.b)

    def p_v(self, h):
        return torch.sigmoid(h @ self.W.T + self.a)

    def free_energy(self, v):                            ## h summed out analytically
        return -v @ self.a - F.softplus(v @ self.W + self.b).sum(-1)

    def gibbs(self, v, k):
        for _ in range(k):                               ## block Gibbs: whole layer at once
            h = torch.bernoulli(self.p_h(v))
            v = torch.bernoulli(self.p_v(h))
        return v

## Example: free energy == -log sum_h exp(-E(v, h)) by brute force over all 2^3 hidden states
torch.manual_seed(0)
rbm = RBM(6, 3)
v = torch.bernoulli(0.5 * torch.ones(4, 6))
h_all = torch.cartesian_prod(*[torch.tensor([0.0, 1.0])] * 3)            ## (8, 3)
E = -(v @ rbm.a)[:, None] - (h_all @ rbm.b)[None] - v @ rbm.W @ h_all.T  ## (4, 8)
print(torch.allclose(rbm.free_energy(v), -torch.logsumexp(-E, dim=1)))   ## True
print(rbm.gibbs(v, k=1).shape)                                           ## torch.Size([4, 6])
```
````

```{attention} Q&A
:class: dropdown
*Why "restricted"?*
- ❌Visible–visible & ❌hidden–hidden edges.

*Is the RBM tractable now?*
- Conditionals & free energy: ✅.
- $Z$ & the model term: ❌ → [CD](#contrastive-divergence), PCD.

*How is an RBM a product of experts?*
- $p(v)\propto e^{a^Tv}\prod_j\big(1+e^{c_j(v)}\big)$ → each hidden unit = 1 expert factor.
- Experts multiply → each vetoes configurations it dislikes → sharp distributions.

*Real-valued data?*
- Gaussian–Bernoulli RBM: Gaussian visible units, binary hidden units.

*Historical role?*
- Smolensky's "harmonium" → renamed RBM later.
- Stacked RBMs → deep belief nets, greedy layer-wise pretraining (Hinton et al., 2006) → part of the 2006 deep-learning revival.
- Later replaced by end-to-end backprop w/ ReLU, good init & normalization.
```

&nbsp;

## Training

&nbsp;

### Contrastive Divergence
- **What**: Approximate MLE gradient whose negative samples come from $k$ MCMC steps started at the data. {cite:p}`hinton2002training`
- **Why**: Exact negative phase needs equilibrium samples from $p_\theta$ → long chains per update.
    - Start at data → chain already in high-density regions → few steps show where the model drifts away from data.
- **How**:
    1. $v^{(0)}$ = data batch.
    2. Run $k$ steps of block Gibbs → $v^{(k)}$ (CD-$k$, often $k=1$).
    3. Update: lower $F$ at $v^{(0)}$, raise $F$ at $v^{(k)}$.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $k$: #Gibbs steps.
    - $\eta$: Learning rate.
- Misc:
    - $v^{(k)}$: Visible state after $k$ Gibbs steps from data.
    - $p_k$: Distribution of $v^{(k)}$. $p_0=p_\text{data}$, $p_\infty=p_\theta$.

Objective:

$$
\text{CD}_k=\text{KL}(p_0\,\|\,p_\theta)-\text{KL}(p_k\,\|\,p_\theta)
$$

Update (RBM, column vectors):

$$\begin{align*}
&\Delta W=\eta\left(v^{(0)}\,p(h|v^{(0)})^T-v^{(k)}\,p(h|v^{(k)})^T\right) \\
&\Delta a=\eta\left(v^{(0)}-v^{(k)}\right),\qquad\Delta b=\eta\left(p(h|v^{(0)})-p(h|v^{(k)})\right)
\end{align*}$$

- $p(h|v)\in[0,1]^{d_h}$: Vector of $p(h_j=1|v)$.
- $k\to\infty$ → $p_k\to p_\theta$ → exact MLE gradient.
```

```{tip} Derivation
:class: dropdown
*Why does the update approximate the CD objective's gradient?*
1. For an EBM w/ marginal energy $F$: $\nabla_\theta\text{KL}(q\|p_\theta)=\langle\nabla_\theta F\rangle_q-\langle\nabla_\theta F\rangle_{p_\theta}$ for any fixed $q$.
2. Apply to $q=p_0$ & $q=p_k$ & subtract → the intractable $\langle\nabla_\theta F\rangle_{p_\theta}$ cancels.
3. Leftover: $\langle\nabla_\theta F\rangle_{p_0}-\langle\nabla_\theta F\rangle_{p_k}$, plus a term from $p_k$ depending on $\theta$.
4. CD drops that extra term (small in practice) → the update above.

*Why does $k=1$ work at all?*
- If $p_\theta=p_\text{data}$: Gibbs from data stays at the data distribution → update = 0.
- → Data distribution is a fixed point (if the model family contains it). Any 1-step drift away from data = a mismatch signal.
```

````{important} Code
:class: dropdown
```python
import torch

## Reuses RBM from RBM above
class CDk:
    def __init__(self, rbm, k=1, lr=0.1):
        self.rbm, self.k = rbm, k
        self.opt = torch.optim.SGD(rbm.parameters(), lr=lr)

    def loss(self, v0):
        vk = self.rbm.gibbs(v0, self.k).detach()        ## chain starts AT the data
        ## grad of F(v0) - F(vk): positive phase - negative phase
        return self.rbm.free_energy(v0).mean() - self.rbm.free_energy(vk).mean(), vk

    def step(self, v0):
        loss, _ = self.loss(v0)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

## Example: autograd of the loss == classic CD-1 rule  dW ∝ v0 p(h|v0)^T - vk p(h|vk)^T
torch.manual_seed(0)
cd = CDk(RBM(6, 3))
v0 = torch.bernoulli(0.5 * torch.ones(4, 6))
loss, vk = cd.loss(v0)
loss.backward()
rule = (v0.T @ cd.rbm.p_h(v0) - vk.T @ cd.rbm.p_h(vk)) / 4       ## (6, 3), batch-averaged
print(torch.allclose(-cd.rbm.W.grad, rule, atol=1e-6))           ## True: SGD on the loss = CD-1 update
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- Cheap: $k=1$ often enough to learn useful features.
- Low variance ← chains start at data.

*Cons?*
- Biased: fixed points ≠ MLE in general (Carreira-Perpiñán & Hinton, 2005).
- ❌Gradient of any objective in general (Sutskever & Tieleman, 2010).
- Spurious modes far from data: short chains never reach them → never pushed up → bad samples from long chains.

*What is PCD?*
- **Persistent CD**: keep the negative chains across updates, ❌restart at data {cite:p}`tieleman2008training`.
- $\theta$ changes slowly → chains track $p_\theta$ → closer to MLE → better likelihood & samples.
- Deep EBMs use the same idea as a replay buffer of past samples.

*CD for continuous EBMs?*
- Replace Gibbs w/ a few [Langevin](#langevin-dynamics) steps from data or a buffer.
```

&nbsp;

### Langevin Dynamics
- **What**: MCMC that steps along the score & adds Gaussian noise: $x\leftarrow x-\frac\eta2\nabla_xE(x)+\sqrt\eta\,z$.
- **Why**: Sample $p\propto e^{-E}$ in continuous high-dim space w/ only $\nabla_xE$ → ❌$Z$.
    - Gradient → moves toward high density → faster than random-walk Metropolis.
    - Noise → explores → ❌collapse to the energy minimum (that would be gradient descent, i.e., MAP).
- **How**:
    1. Init $x$: noise, data, or a replay buffer.
    2. Repeat: step down the energy gradient + inject noise w/ variance = step size.
    3. Small $\eta$, many steps → samples from $p$.
    4. **ULA** (unadjusted Langevin algorithm): ❌accept/reject → bias $O(\eta)$. **MALA**: add a Metropolis–Hastings accept/reject → exact stationary distribution.
    - Deep EBMs: negative samples from short-run Langevin chains + replay buffer {cite:p}`du2019implicit`.
    - Learn the score directly instead of $E$ → [Score Matching](diffusion.md#score-matching). Multi-noise annealed version → [NCSN](diffusion.md#ncsn).

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\eta$: Step size.
- Misc:
    - $z^{(t)}\sim\mathcal{N}(0,I)$: Fresh noise at step $t$.
    - $x^{(t)}$: Chain state at step $t$.
    - $W_\tau$: Standard Brownian motion in continuous time $\tau$.

Overdamped Langevin SDE:

$$
dx=-\nabla_xE(x)\,d\tau+\sqrt2\,dW_\tau
$$

- Stationary distribution: $p\propto e^{-E}$.

Process (ULA):

$$
x^{(t+1)}=x^{(t)}-\frac\eta2\nabla_xE\big(x^{(t)}\big)+\sqrt\eta\,z^{(t)}=x^{(t)}+\frac\eta2\nabla_x\log p\big(x^{(t)}\big)+\sqrt\eta\,z^{(t)}
$$

- = Euler–Maruyama step of the SDE w/ $d\tau=\eta/2$.
- Same form as [NCSN](diffusion.md#ncsn) sampling w/ $\delta=\eta$.
- $\eta\to0$ & #steps $\to\infty$ → exact. Fixed $\eta$ → stationary distribution ≠ $p$, bias $O(\eta)$.
```

```{tip} Derivation
:class: dropdown
*Why is $p\propto e^{-E}$ stationary?*
1. Fokker–Planck of the SDE: $\partial_\tau p=\nabla\cdot(p\,\nabla E)+\Delta p=\nabla\cdot(p\,\nabla E+\nabla p)$.
2. $p\propto e^{-E}$ → $\nabla p=-p\,\nabla E$ → probability flux $p\nabla E+\nabla p=0$.
3. → $\partial_\tau p=0$.

*How big is the ULA bias? Gaussian case.*
1. $E=x^2/2$ (1-D) → target $\mathcal{N}(0,1)$.
2. ULA: $x'=(1-\eta/2)\,x+\sqrt\eta\,z$ → linear Gaussian chain.
3. Stationary variance $v$: $v=(1-\eta/2)^2v+\eta$ → $v=\dfrac{1}{1-\eta/4}>1$.
4. → Bias $O(\eta)$: too wide for any fixed $\eta>0$.

*Why noise $\sqrt\eta$, ❌$\eta$?*
- Brownian displacement std ∝ $\sqrt{\text{time}}$.
- Noise ∝ $\eta$ → vanishes faster than the drift as $\eta\to0$ → gradient descent → collapses to the mode.
```

````{important} Code
:class: dropdown
```python
import torch

class Langevin:
    def __init__(self, energy, eta=1e-2, n_steps=2000):
        self.E, self.eta, self.n = energy, eta, n_steps

    def sample(self, x):
        for _ in range(self.n):
            x = x.detach().requires_grad_(True)
            grad = torch.autograd.grad(self.E(x).sum(), x)[0]    ## ∇E = -score; Z never needed
            x = x - 0.5 * self.eta * grad + self.eta ** 0.5 * torch.randn_like(x)
        return x.detach()

## Example: E = ||x - 3||^2 / 2 -> target N(3, I)
x = Langevin(lambda x: 0.5 * ((x - 3) ** 2).sum(-1)).sample(torch.zeros(5000, 2))
print(x.mean(0), x.std(0))   ## ≈ [3, 3], ≈ [1, 1] (exact ULA std = 1/sqrt(1 - eta/4) ≈ 1.001)
```
````

```{attention} Q&A
:class: dropdown
*ULA vs MALA vs SGLD?*
- ULA: cheap, biased $O(\eta)$.
- MALA: + accept/reject → exact, but rejections ⬆️ in high dim unless $\eta$⬇️.
- SGLD: minibatch grads of a log posterior + Langevin noise → Bayesian posterior sampling over params (Welling & Teh, 2011).

*When does Langevin fail?*
- Modes separated by low-density regions → crossing time exponential in the barrier height → wrong mode proportions.
- Fix: anneal from high noise / temperature → [NCSN](diffusion.md#ncsn).

*How is it used in deep EBM training in practice?*
- Short chains (tens of steps) from a replay buffer → negative samples.
- Common heuristic: noise scale smaller than $\sqrt\eta$ → ❌valid Langevin → the learned $E$ adapts to this sampler, ❌a faithful density.

*Langevin & diffusion sampling?*
- DDPM ancestral sampling ≈ annealed Langevin across noise levels.
- Score SDE predictor-corrector: corrector = a few Langevin steps at fixed noise level → [Score SDE](diffusion.md#score-sde).
```

&nbsp;

### NCE
- **Name**: Noise-Contrastive Estimation {cite:p}`gutmann2010noise`
- **What**: Fit an unnormalized model by logistic regression: data vs samples from a known noise distribution; $\log Z$ = a learned param.
- **Why**: MLE needs $Z$ or MCMC. Score matching needs $\nabla_x$ (❌discrete data).
    - NCE: ❌$Z$, ❌MCMC, works for discrete data.
- **How**:
    1. Draw $\nu$ noise samples per data sample from $p_n$ (known density, easy to sample).
    2. Classifier logit = $\log p_\theta(u)-\log p_n(u)-\log\nu$, w/ $\log p_\theta=-E_\theta-c$.
    3. Train binary logistic regression (data = 1, noise = 0) over $\theta$ & $c$.
    4. At the optimum: $p_\theta=p_\text{data}$ & $c=\log Z_\theta$ → self-normalized model.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $u$: Generic input (data or noise sample).
    - $\tilde x\sim p_n$: Noise sample.
- Params:
    - $c$: Learned log-normalizer ($\approx\log Z_\theta$).
- Hyperparams:
    - $p_n$: Noise distribution.
    - $\nu$: #noise samples per data sample.
- Misc:
    - $G(u)$: Classifier logit.

Model:

$$
\log p_\theta(u)=-E_\theta(u)-c
$$

Classifier:

$$
P(\text{data}\,|\,u)=\sigma\big(G(u)\big),\qquad G(u)=\log p_\theta(u)-\log p_n(u)-\log\nu
$$

Objective (maximize):

$$
J(\theta,c)=\mathbb{E}_{x\sim p_\text{data}}\left[\log\sigma\big(G(x)\big)\right]+\nu\,\mathbb{E}_{\tilde x\sim p_n}\left[\log\sigma\big(-G(\tilde x)\big)\right]
$$

- Requires $p_n(u)>0$ wherever $p_\text{data}(u)>0$.
```

```{tip} Derivation
:class: dropdown
*Why is the optimum $p_\theta=p_\text{data}$?*
1. Mix data & noise at ratio $1:\nu$. Bayes-optimal posterior: $P(\text{data}|u)=\dfrac{p_\text{data}(u)}{p_\text{data}(u)+\nu\,p_n(u)}=\sigma\big(\log p_\text{data}(u)-\log p_n(u)-\log\nu\big)$.
2. Logistic loss = proper scoring rule → maximized by the true posterior.
3. → $G(u)=\log p_\text{data}(u)-\log p_n(u)-\log\nu$ → $\log p_\theta=\log p_\text{data}$.
4. $c$ free → absorbs the normalizer → $c=\log Z_\theta$ (assumes the model family contains $p_\text{data}$).
```

````{important} Code
:class: dropdown
```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class NCELoss(nn.Module):
    def __init__(self, energy, noise, nu=1):
        super().__init__()
        self.E, self.noise, self.nu = energy, noise, nu   ## noise: torch.distributions object
        self.c = nn.Parameter(torch.zeros(()))            ## learned log Z

    def logit(self, u):                                   ## log p_theta - log p_n - log nu
        return -self.E(u) - self.c - self.noise.log_prob(u) - math.log(self.nu)

    def forward(self, x):
        x_noise = self.noise.sample((self.nu * x.shape[0],))
        ## binary logistic regression: data -> 1, noise -> 0; sums / #data match J(θ, c)
        return -(F.logsigmoid(self.logit(x)).sum() + F.logsigmoid(-self.logit(x_noise)).sum()) / x.shape[0]

## Example: true energy x^2/2 fixed; learn only c -> should reach log Z = log sqrt(2π) ≈ 0.919
torch.manual_seed(0)
nce = NCELoss(lambda x: 0.5 * x ** 2, torch.distributions.Normal(0.0, 2.0))
opt = torch.optim.Adam(nce.parameters(), lr=0.05)
for _ in range(500):
    opt.zero_grad()
    nce(torch.randn(4096)).backward()
    opt.step()
print(nce.c.item())   ## ≈ 0.92
```
````

```{attention} Q&A
:class: dropdown
*Why can NCE learn $c$, but MLE cannot?*
- MLE w/ free $c$: $\log p_\theta=-E-c$ → $c\to-\infty$ raises every density → unbounded likelihood.
- NCE: high $p_\theta$ on noise samples → misclassification → penalized → $c$ pinned at $\log Z$.

*How do you choose the noise?*
- Close to data → hard classification → informative grads → efficient.
- Far from data → classifier wins on trivial features → weak signal for $\theta$.
- $\nu$⬆️ → variance ⬇️. As $\nu\to\infty$, efficiency approaches MLE.

*NCE vs InfoNCE?*
- NCE: binary classification per sample → consistent density estimator, absolute normalizer $c$.
- [InfoNCE](../../ml/obj.md#infonce): $(\nu+1)$-way softmax → pick the positive among negatives → learns a density ratio up to a constant, ❌absolute normalizer.

*NCE vs word2vec negative sampling?*
- Negative sampling = NCE w/ $\nu\,p_n(u)=1$ inside the logit → ❌consistent density estimator, but good embeddings.

*NCE vs GAN?*
- NCE: fixed noise distribution. The classifier is built from the model itself.
- [GAN](gan.md#gan): noise distribution = a learned generator. Discriminator = a separate net.
```

&nbsp;
