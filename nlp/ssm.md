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
# State Space Models
Linear recurrent sequence layers derived from continuous-time state spaces: S4 → Mamba-3. Linear attention & DeltaNet: [Attention](attn.md#linear-attention). Production hybrids: [Models](models.md#mamba-hybrids).

Notations:
- $m$: #Tokens in sequence
- $d$: Model dim (#channels)
- $N$: State dim
- $t$: Time step idx
- $x_t\in\mathbb{R}$: Input at step $t$ (1 channel)
- $y_t\in\mathbb{R}$: Output at step $t$ (1 channel)
- $\mathbf{h}_t\in\mathbb{R}^N$: Hidden state at step $t$
- $A\in\mathbb{R}^{N\times N}$: State matrix
- $B\in\mathbb{R}^{N\times 1}$: Input matrix
- $C\in\mathbb{R}^{1\times N}$: Output matrix
- $\Delta>0$: Step size
- $\bar{A}$: Discretized state matrix
- $\bar{B}$: Discretized input matrix

Override: $m$ = #tokens (global: #samples), matching [Attention](attn.md).

&nbsp;

## Basics

&nbsp;

### State Space Model
- **What**: Linear ODE: input signal → $N$-dim latent state → output signal.
- **Why**: Combine the strengths of RNNs & CNNs.
    - [RNN](../dl/foundation/module.md#rnn): $\mathcal{O}(1)$ per-step inference, ❌parallel training.
    - CNN: parallel training, ❌unbounded context.
    - Transformer: exact recall, but $\mathcal{O}(m^2)$ training & $\mathcal{O}(m)$ KV cache.
    - Linear SSM = recurrence AND convolution → parallel training + constant-size inference state.
- **How**:
    1. Treat each channel as a 1-D signal $x(t)$.
    2. Drive an $N$-dim state $\mathbf{h}(t)$ w/ a linear ODE.
    3. Read the output as a linear projection of the state.
    4. [Discretize](#discretization) w/ step size $\Delta$ → recurrence over tokens.
    5. Stack layers: SSM per channel → nonlinearity & channel mixing between SSMs.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $D\in\mathbb{R}$: Skip param.
- Misc:
    - $x(t),y(t)$: Continuous-time input & output.
    - $\mathbf{h}(t)\in\mathbb{R}^N$: Continuous-time state.

Definition:

$$
\mathbf{h}'(t)=A\mathbf{h}(t)+Bx(t),\qquad y(t)=C\mathbf{h}(t)+Dx(t)
$$

- $Dx(t)$: Skip term. Often dropped (≈ residual connection).

Properties:
- **LTI** (linear time-invariant): $A,B,C$ fixed over time.
- **SISO** (single-input single-output): 1 channel in → 1 channel out. Layer w/ $d$ channels = $d$ SSMs.
- Stable ⟺ every eigenvalue of $A$ has negative real part.
- Change of basis $(A,B,C)\to(V^{-1}AV,\ V^{-1}B,\ CV)$ → same map $x\mapsto y$.
```

```{attention} Q&A
:class: dropdown
*Where is the nonlinearity?*
- Not in the SSM. It sits between SSMs: activations, gates, MLPs.
- Linear core → closed-form discretization, convolution & scan.

*How does it relate to an RNN?*
- RNN: $\mathbf{h}_t=\tanh(W\mathbf{h}_{t-1}+U\mathbf{x}_t)$ → nonlinear in $\mathbf{h}$ → strictly sequential.
- SSM: linear in $\mathbf{h}$ → unroll in closed form → convolution or parallel scan.

*How does it relate to classical SSMs (control, Kalman filter)?*
- Same equations.
- Classical: hand-specified $A,B,C$, noise model, goal = infer the hidden state.
- Deep SSM: learned $A,B,C$, deterministic, goal = sequence-to-sequence map.

*Why continuous time?*
- $\Delta$ = learned timescale. New sampling rate → rescale $\Delta$, keep $A,B,C$.
- Continuous theory ([HiPPO](#hippo)) gives an $A$ w/ provable long memory.
- Not strictly required: a linear, diagonal, well-initialized RNN (LRU) matches deep SSMs on Long Range Arena w/o any discretization. {cite:p}`orvieto2023resurrecting`
```

&nbsp;

### Discretization
- **What**: Map continuous $(A,B)$ & step $\Delta$ to discrete $(\bar{A},\bar{B})$ for the recurrence $\mathbf{h}_t=\bar{A}\mathbf{h}_{t-1}+\bar{B}x_t$.
- **Why**: Tokens are samples, not functions → ODE must become a recurrence.
- **How**:
    - **ZOH** (zero-order hold): hold $x$ constant over each step → solve the ODE exactly.
    - **Bilinear**: trapezoid rule on $\mathbf{h}'$.
    - **Forward Euler**: first-order step $\mathbf{h}_t\approx\mathbf{h}_{t-1}+\Delta\mathbf{h}'$.
    - $C$ stays unchanged: $y_t=C\mathbf{h}_t$.

```{note} Math
:class: dropdown
ZOH:

$$
\bar{A}=\exp(\Delta A),\qquad \bar{B}=(\Delta A)^{-1}\left(\exp(\Delta A)-I\right)\Delta B
$$

- Closed form needs invertible $A$. General: $\bar{B}=\left(\int_0^\Delta e^{sA}ds\right)B$ (e.g., $A=0$ → $\bar{B}=\Delta B$).

Bilinear:

$$
\bar{A}=\left(I-\tfrac{\Delta}{2}A\right)^{-1}\left(I+\tfrac{\Delta}{2}A\right),\qquad \bar{B}=\left(I-\tfrac{\Delta}{2}A\right)^{-1}\Delta B
$$

Forward Euler:

$$
\bar{A}=I+\Delta A,\qquad \bar{B}=\Delta B
$$
```

```{tip} Derivation
:class: dropdown
*Why $\exp(\Delta A)$ under ZOH?*
1. Exact solution of the linear ODE over one step: $\mathbf{h}(\tau_0+\Delta)=e^{\Delta A}\mathbf{h}(\tau_0)+\int_0^\Delta e^{(\Delta-\tau)A}Bx(\tau_0+\tau)\,d\tau$.
2. ZOH: $x(\tau_0+\tau)=x_t$ on the whole step → pull $Bx_t$ out of the integral.
3. Substitute $s=\Delta-\tau$: $\int_0^\Delta e^{sA}ds=A^{-1}\left(e^{\Delta A}-I\right)$.
4. → $\bar{A}=e^{\Delta A}$, $\bar{B}=A^{-1}(e^{\Delta A}-I)B=(\Delta A)^{-1}(e^{\Delta A}-I)\Delta B$.

*Where does the bilinear form come from?*
1. Trapezoid rule on the state: $\mathbf{h}_t-\mathbf{h}_{t-1}=\frac{\Delta}{2}A(\mathbf{h}_t+\mathbf{h}_{t-1})+\Delta Bx_t$.
2. Collect $\mathbf{h}_t$: $\left(I-\frac{\Delta}{2}A\right)\mathbf{h}_t=\left(I+\frac{\Delta}{2}A\right)\mathbf{h}_{t-1}+\Delta Bx_t$.
3. Multiply by $\left(I-\frac{\Delta}{2}A\right)^{-1}$.
```

````{important} Code
:class: dropdown
```python
import torch

class Discretize:
    def __init__(self, A, B):
        self.A, self.B = A, B  ## A: (N, N), invertible | B: (N, 1)

    def zoh(self, dt):
        ## Exact for piecewise-constant input
        I = torch.eye(self.A.shape[0], dtype=self.A.dtype)
        A_bar = torch.linalg.matrix_exp(dt * self.A)
        B_bar = torch.linalg.solve(self.A, (A_bar - I) @ self.B)  ## A^{-1}(e^{dt A} - I) B
        return A_bar, B_bar

    def bilinear(self, dt):
        ## Trapezoid rule on h' -> maps Re(lambda) < 0 into the unit disk
        I = torch.eye(self.A.shape[0], dtype=self.A.dtype)
        inv = torch.linalg.inv(I - dt / 2 * self.A)
        return inv @ (I + dt / 2 * self.A), inv @ (dt * self.B)

## Example: A = -1, B = 1, dt = 0.5
A = torch.tensor([[-1.0]], dtype=torch.float64)
B = torch.tensor([[1.0]], dtype=torch.float64)
print([v.item() for v in Discretize(A, B).zoh(0.5)])       ## [0.6065..., 0.3934...] = e^{-0.5}, 1 - e^{-0.5}
print([v.item() for v in Discretize(A, B).bilinear(0.5)])  ## [0.6, 0.4]
```
````

```{attention} Q&A
:class: dropdown
*What does $\Delta$ control?*
- $\Delta\to0$: $\bar{A}\to I$, $\bar{B}\to0$ → keep the state, ignore the input.
- $\Delta\to\infty$ (ZOH, stable $A$): $\bar{A}\to0$ → forget the state, focus on the curr input.
    - Bilinear: $\bar{A}\to-I$ instead. Forward Euler: diverges.
- → $\Delta$ = timescale = gate. [Mamba](#mamba) makes it input-dependent.

*Which methods preserve stability?*
- ZOH: eigenvalue $\lambda$ of $A$ → $e^{\Delta\lambda}$. $\mathrm{Re}\,\lambda<0$ → $|e^{\Delta\lambda}|<1$ for every $\Delta$ ✅.
- Bilinear: maps the left half-plane into the unit disk for every $\Delta$ ✅.
- Forward Euler: $|1+\Delta\lambda|<1$ only for small $\Delta$ → ❌large steps.

*Which method does each model use?*
- S4: bilinear. S4D: either (base = bilinear; little empirical difference). DSS & S5: ZOH.
- Mamba-1/2: paper states ZOH; released code uses $\exp(\Delta A)$ & $\bar{B}=\Delta B$ (exponential-Euler).
- Mamba-3: exponential-trapezoidal → [Mamba-3](#mamba-3).
```

&nbsp;

### Convolutional View
- **What**: Unrolled LTI recurrence = causal convolution of the input w/ a kernel $\bar{K}$.
- **Why**: Recurrence → $m$ sequential steps. Convolution → parallel via FFT in $\mathcal{O}(m\log m)$.
- **How**:
    1. Unroll $\mathbf{h}_t$ from $\mathbf{h}_0=\mathbf{0}$.
    2. Each output = weighted sum of past inputs. Weight depends only on the lag $t-s$.
    3. **Training**: compute $\bar{K}$ once per param update → FFT convolution.
    4. **Inference**: switch back to the recurrence → constant cost & memory per token.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\ell$: Lag idx, $0\le\ell<m$.
    - $\bar{K}\in\mathbb{R}^m$: SSM convolution kernel.

Forward:

$$
y_t=\sum_{s=1}^{t}\bar{K}_{t-s}\,x_s,\qquad \bar{K}_\ell=C\bar{A}^{\ell}\bar{B}
$$
```

```{tip} Derivation
:class: dropdown
1. $\mathbf{h}_1=\bar{B}x_1$.
2. $\mathbf{h}_2=\bar{A}\bar{B}x_1+\bar{B}x_2$.
3. Induction: $\mathbf{h}_t=\sum_{s=1}^{t}\bar{A}^{t-s}\bar{B}x_s$.
4. $y_t=C\mathbf{h}_t=\sum_{s=1}^{t}C\bar{A}^{t-s}\bar{B}\,x_s$ → kernel $\bar{K}_\ell=C\bar{A}^\ell\bar{B}$.
```

````{important} Code
:class: dropdown
```python
import torch

## Reuses Discretize from this page
class SSMConv:
    def __init__(self, A_bar, B_bar, C):
        self.A_bar, self.B_bar, self.C = A_bar, B_bar, C  ## (N, N), (N, 1), (1, N)

    def kernel(self, m):
        ## K_l = C A_bar^l B_bar -> depends on params only, NOT on the input
        K, v = [], self.B_bar
        for _ in range(m):
            K.append((self.C @ v).squeeze())
            v = self.A_bar @ v
        return torch.stack(K)

    def __call__(self, x):
        ## Causal conv via FFT; pad to 2m so circular conv = linear conv
        m = x.shape[-1]
        Kf = torch.fft.rfft(self.kernel(m), n=2 * m)
        return torch.fft.irfft(torch.fft.rfft(x, n=2 * m) * Kf, n=2 * m)[..., :m]

## Example: convolution output == step-by-step recurrence
torch.manual_seed(0)
N, m = 3, 8
A = -torch.eye(N, dtype=torch.float64) + 0.1 * torch.randn(N, N, dtype=torch.float64)
B, C = torch.randn(N, 1, dtype=torch.float64), torch.randn(1, N, dtype=torch.float64)
A_bar, B_bar = Discretize(A, B).zoh(0.1)
x = torch.randn(m, dtype=torch.float64)
h, y_rec = torch.zeros(N, 1, dtype=torch.float64), []
for t in range(m):
    h = A_bar @ h + B_bar * x[t]
    y_rec.append((C @ h).squeeze())
print(torch.allclose(SSMConv(A_bar, B_bar, C)(x), torch.stack(y_rec)))  ## True
```
````

```{attention} Q&A
:class: dropdown
*Why does it need LTI?*
- $\bar{K}$ depends only on $(\bar{A},\bar{B},C)$.
- Input-dependent params → different kernel at every position → ❌single convolution → [Parallel Scan](#parallel-scan).

*What is the bottleneck?*
- Naive kernel: $m$ powers of $\bar{A}$ → $\mathcal{O}(N^2m)$ ops & $\mathcal{O}(Nm)$ memory.
- Structured $A$ removes it → [S4](#s4), [S4D](#s4d).

*Training cost vs attention?*
- FFT convolution: $\mathcal{O}(m\log m)$ per channel.
- Attention: $\mathcal{O}(m^2)$.
```

&nbsp;

### Parallel Scan
- **What**: All prefixes of an associative operator in $\mathcal{O}(\log m)$ parallel depth.
- **Why**: Time-varying recurrence $\mathbf{h}_t=\bar{A}_t\mathbf{h}_{t-1}+\bar{B}_tx_t$ → ❌convolution. Sequential loop → $m$ steps, GPU idle.
- **How**:
    1. Encode step $t$ as an affine map $(a_t,b_t)$: $h\mapsto a_th+b_t$.
    2. Composition of affine maps is associative.
    3. Combine neighbors, then neighbors of neighbors → tree w/ $\lceil\log_2m\rceil$ levels.
    4. **Hillis–Steele**: $\log_2m$ depth, $\mathcal{O}(m\log m)$ work. **Blelloch**: up-sweep + down-sweep → $2\log_2m$ depth, $\mathcal{O}(m)$ work.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $a_t$: Transition at step $t$ (1 diagonal entry of $\bar{A}_t$).
    - $b_t$: Input at step $t$ (1 entry of $\bar{B}_tx_t$).
    - $\bullet$: Scan operator.

Operator (apply 1, then 2):

$$
(a_1,b_1)\bullet(a_2,b_2)=(a_2a_1,\ a_2b_1+b_2)
$$

Result:

$$
(a_1,b_1)\bullet\cdots\bullet(a_t,b_t)=\left(\textstyle\prod_{r=1}^{t}a_r,\ h_t\right)
$$
```

```{tip} Derivation
:class: dropdown
*Why is $\bullet$ associative?*
1. $((1\bullet2)\bullet3)=(a_3a_2a_1,\ a_3(a_2b_1+b_2)+b_3)$.
2. $(1\bullet(2\bullet3))=(a_3a_2a_1,\ a_3a_2b_1+a_3b_2+b_3)$.
3. Equal → any bracketing works → tree-parallel.
```

````{important} Code
:class: dropdown
```python
import torch

class LinearScan:
    def __call__(self, a, b):
        ## Solves h_t = a_t * h_{t-1} + b_t (h_{-1} = 0) in ceil(log2 m) parallel steps
        m, k = a.shape[0], 1
        while k < m:
            ## Hillis-Steele: every t >= k absorbs the block ending at t-k, all t at once
            a_new, b_new = a.clone(), b.clone()
            b_new[k:] = a[k:] * b[:-k] + b[k:]  ## earlier map first, later map second
            a_new[k:] = a[k:] * a[:-k]
            a, b, k = a_new, b_new, 2 * k
        return b

## Example: matches the sequential loop
torch.manual_seed(0)
a, b = torch.rand(10, dtype=torch.float64), torch.randn(10, dtype=torch.float64)
h, ref = 0.0, []
for t in range(10):
    h = a[t] * h + b[t]
    ref.append(h)
print(torch.allclose(LinearScan()(a, b), torch.stack(ref)))  ## True
```
````

```{attention} Q&A
:class: dropdown
*Why does diagonal $\bar{A}$ matter?*
- Dense $\bar{A}_t$: each combine multiplies $N\times N$ matrices → $\mathcal{O}(N^3)$.
- Diagonal: elementwise → $\mathcal{O}(N)$, $N$ independent scalar scans.

*Why not always scan, even for LTI?*
- Scan = elementwise ops → memory-bound, ❌tensor cores.
- LTI → FFT convolution is often faster → S4 uses it. [S5](#s5) & [Mamba](#mamba) use the scan.
```

&nbsp;

## Structured SSMs
- **What**: LTI SSMs w/ a structured $A$ for long memory & fast compute.

&nbsp;

### HiPPO
- **Name**: High-order Polynomial Projection Operators {cite:p}`gu2020hippo`
- **What**: Online projection of the input history onto $N$ orthogonal polynomials, updated by a linear ODE.
- **Why**: Memory w/ guarantees.
    - RNNs forget ← vanishing grads. Gates ([LSTM](../dl/foundation/module.md#lstm)) help only heuristically.
    - HiPPO state = optimal $N$-coefficient approximation of the history → $A,B$ in closed form.
- **How**:
    1. Choose a measure: the weight of each past time. **LegS**: uniform over all of $[0,t]$.
    2. Project $x_{\le t}$ onto Legendre polynomials under that measure → $N$ coefficients.
    3. Differentiate the projection in $t$ → coefficients follow a linear ODE.
    4. Reconstruct the history at any time from the coefficients.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\mathbf{c}(t)\in\mathbb{R}^N$: Projection coefficients (= state).
    - $P_n$: $n$-th Legendre polynomial.
    - $n,k$: Coefficient idx, $0\le n,k<N$ (overrides global $k$ = class idx).

Definition (HiPPO-LegS):

$$
\mathbf{c}'(t)=-\frac{1}{t}A\mathbf{c}(t)+\frac{1}{t}Bx(t)
$$

$$
A_{nk}=\begin{cases}(2n+1)^{1/2}(2k+1)^{1/2}&n>k\\ n+1&n=k\\ 0&n<k\end{cases},\qquad B_n=(2n+1)^{1/2}
$$

- Sign convention: $A$ enters w/ a minus sign. S4 uses $-A$ as its state matrix & drops $\frac{1}{t}$ (LTI).

Discrete (forward Euler, step $t\ge1$):

$$
\mathbf{c}_{t+1}=\left(I-\frac{A}{t}\right)\mathbf{c}_t+\frac{1}{t}Bx_t
$$

Reconstruction, $s\in[0,t]$:

$$
x(s)\approx\sum_{n=0}^{N-1}c_n(t)\,(2n+1)^{1/2}P_n\!\left(\tfrac{2s}{t}-1\right)
$$

Properties:
- $A$ lower-triangular → eigenvalues of $-A$ = $-(n+1)$ → stable.
- LegS window grows w/ $t$ → ❌timescale hyperparam.
```

````{important} Code
:class: dropdown
```python
import math
import torch

class HiPPOLegS:
    def __init__(self, N):
        n = torch.arange(N, dtype=torch.float64)
        r = torch.sqrt(2 * n + 1)
        ## r_n r_k below the diagonal, n+1 on it, 0 above -> lower-triangular
        self.A = torch.tril(torch.outer(r, r), -1) + torch.diag(n + 1)
        self.B = r
        self.N = N

    def __call__(self, f):
        ## Online: after step t, c = Legendre coeffs of the WHOLE prefix f_1..f_t
        c = torch.zeros(self.N, dtype=torch.float64)
        for t in range(1, len(f) + 1):
            c = c - self.A @ c / t + self.B * f[t - 1] / t
        return c

    def reconstruct(self, c, s):
        ## s in (0, 1] = fraction of the history; Legendre via 3-term recurrence
        z = 2 * s - 1
        P = [torch.ones_like(z), z]
        for n in range(1, self.N - 1):
            P.append(((2 * n + 1) * z * P[n] - n * P[n - 1]) / (n + 1))
        return sum(c[n] * self.B[n] * P[n] for n in range(self.N))

## Example: compress 1000 samples into N numbers
L = 1000
s = torch.arange(1, L + 1, dtype=torch.float64) / L
f = torch.sin(2 * math.pi * s)
for N in (4, 8):
    hippo = HiPPOLegS(N)
    rmse = (hippo.reconstruct(hippo(f), s) - f).pow(2).mean().sqrt()
    print(N, rmse.item())  ## 4 -> ~0.066 | 8 -> ~0.0036 (floor = Euler discretization error)
```
````

```{attention} Q&A
:class: dropdown
*Why Legendre polynomials?*
- Orthogonal on an interval → independent coefficients → closed-form linear ODE.
- Other measures → other bases. **LegT**: sliding window of fixed length. **LagT**: exponentially decaying weights (Laguerre).

*Why LegS?*
- Window = whole history, rescaled w/ $t$ → no window-length hyperparam.
- Robust to input timescale. Bounded grads.

*Does HiPPO matter for LMs?*
- Mamba default init S4D-Real ($-(n+1)$) comes from HiPPO.
- On language, random init also works. → HiPPO matters most on long-range continuous signals.

*How does it relate to RNN gates?*
- Gating = the lowest-order extreme of HiPPO.
- Exponentially decaying measure (LagT) w/ $N=1$: $c'=-c+x$ → Euler: $c_t=(1-\Delta)c_{t-1}+\Delta x_t$.
- Choose $\Delta$ adaptively from the input & state → gated RNN update.
```

&nbsp;

### S4
- **Name**: Structured State Space sequence model {cite:p}`gu2022efficiently`
- **What**: Deep SSM layer w/ a HiPPO-initialized normal-plus-low-rank $A$ & a fast convolution kernel.
- **Why**: HiPPO gives long memory, but the naive kernel is too expensive.
    - $m$ powers of $\bar{A}$ → $\mathcal{O}(N^2m)$ ops, $\mathcal{O}(Nm)$ memory (LSSL, the predecessor).
    - Direct diagonalization of HiPPO is unstable ← eigenvector entries grow exponentially in $N$.
- **How**:
    1. Write HiPPO $A$ as **normal plus low-rank** (NPLR).
    2. Conjugate by a unitary $V$ → **diagonal plus low-rank** (DPLR). Stable ← unitary.
    3. Evaluate the kernel's generating function at roots of unity. → Matrix inverse replaces matrix powers.
    4. Low-rank term → Woodbury identity. Diagonal term → Cauchy kernel.
    5. Inverse FFT → $\bar{K}$. **Training**: FFT conv. **Generation**: recurrence.
    6. **Layer**: $d$ independent SISO SSMs (1 per channel) + skip $D$ → GELU → position-wise linear mixing.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\Lambda$: Diagonal matrix (eigenvalues of the normal part).
    - $P,Q\in\mathbb{R}^{N\times r}$: Real low-rank factors ($r=1$ for LegS).
- Misc:
    - $V\in\mathbb{C}^{N\times N}$: Unitary matrix.
    - $\omega$: $m$-th root of unity, $\omega^m=1$.
    - $\hat{K}(z)$: Generating function of $\bar{K}$.

NPLR:

$$
A=V\Lambda V^*-PQ^\top=V\left(\Lambda-(V^*P)(V^*Q)^*\right)V^*
$$

Generating function at roots of unity:

$$
\hat{K}(\omega^j)=\sum_{\ell=0}^{m-1}\bar{K}_\ell\,\omega^{j\ell}=\tilde{C}\left(I-\bar{A}\omega^j\right)^{-1}\bar{B},\qquad \tilde{C}=C\left(I-\bar{A}^m\right)
$$

- $(I-\bar{A}\omega^j)^{-1}$ w/ DPLR $\bar{A}$ → Woodbury + Cauchy kernel.
- Cost: $\tilde{\mathcal{O}}(N+m)$ ops, $\mathcal{O}(N+m)$ memory.
```

```{tip} Derivation
:class: dropdown
*Why do roots of unity remove the matrix powers?*
1. Geometric series: $\sum_{\ell=0}^{m-1}(\bar{A}z)^\ell=\left(I-\bar{A}^mz^m\right)(I-\bar{A}z)^{-1}$.
2. $z=\omega^j$ → $z^m=1$ → factor $(I-\bar{A}^m)$ no longer depends on $j$ → fold it into $\tilde{C}$.
3. → $m$ evaluations of 1 resolvent $(I-\bar{A}z)^{-1}$, then 1 inverse FFT → $\bar{K}$.
```

```{attention} Q&A
:class: dropdown
*Pros?*
- First model to solve LRA Path-X (length 16,384): 88% vs 50% (chance) for all prior work.
- SoTA on every Long Range Arena task. 91% sequential CIFAR-10 w/o augmentation.
- Generation 60× faster than Transformers ← recurrent mode.
- vs LSSL: up to 30× faster, 400× less memory.

*Cons?*
- Complex algorithm: Cauchy kernel, custom kernels.
- LTI → ❌content-based selection → large gap to Transformers on language.

*Why complex numbers?*
- Normal matrices diagonalize over $\mathbb{C}$. Eigenvalues of HiPPO's normal part are complex.
- Real input → real output ← states come in conjugate pairs: eigenvalues AND matching $B,C$ entries.

*How did later work simplify it?*
- Drop the low-rank term → [S4D](#s4d). Replace convolution w/ scan → [S5](#s5). Add selection → [Mamba](#mamba).
```

&nbsp;

#### S4D
- **What**: S4 w/ a diagonal state matrix. {cite:p}`gu2022parameterization`
- **Why**: S4's NPLR algorithm is hard to implement.
    - DSS: diagonal $A$ w/ an S4-derived init matches S4. {cite:p}`gupta2022diagonal`
- **How**:
    1. $A=\mathrm{diag}(\lambda_1,\dots,\lambda_N)$, complex. Store 1 state per conjugate pair $(\lambda_n,B_n,C_n)$ → $N/2$ states, real output.
    2. Discretize per state dim (elementwise): bilinear (base S4D) or ZOH. Little empirical difference.
    3. Kernel = Vandermonde product → 2 lines of code, $\mathcal{O}(Nm)$.
    4. Init from HiPPO: S4D-LegS (diagonal of HiPPO's normal part), S4D-Inv, S4D-Lin; or S4D-Real.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\lambda_n\in\mathbb{C}$: $n$-th diagonal entry of $A$, $\mathrm{Re}\,\lambda_n<0$.
    - $B_n,C_n\in\mathbb{C}$: $n$-th entries of $B,C$.

Forward (ZOH variant):

$$
\bar{K}_\ell=2\,\mathrm{Re}\sum_{n=0}^{N/2-1}C_n\bar{B}_n\,e^{\ell\Delta\lambda_n},\qquad \bar{B}_n=\lambda_n^{-1}\left(e^{\Delta\lambda_n}-1\right)B_n
$$

Init ($n=0,1,\dots$):
- **S4D-Lin**: $\lambda_n=-\frac{1}{2}+i\pi n$.
- **S4D-Inv**: $\lambda_n=-\frac{1}{2}+i\frac{N}{\pi}\left(\frac{N}{2n+1}-1\right)$.
- **S4D-Real**: $\lambda_n=-(n+1)$.
```

````{important} Code
:class: dropdown
```python
import math
import torch
import torch.nn as nn

class S4DKernel(nn.Module):
    def __init__(self, d, N, dt_min=1e-3, dt_max=1e-1):
        super().__init__()
        ## 1 timescale per channel, log-uniform in [dt_min, dt_max]
        self.log_dt = nn.Parameter(torch.rand(d) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        ## S4D-Lin: lambda_n = -1/2 + i*pi*n; Re stored as log -> Re < 0 always -> stable
        self.log_A_real = nn.Parameter(torch.full((d, N // 2), math.log(0.5)))
        self.A_imag = nn.Parameter(math.pi * torch.arange(N // 2).float().repeat(d, 1))
        self.C = nn.Parameter(torch.randn(d, N // 2, dtype=torch.cfloat))  ## B = 1, folded into C

    def forward(self, m):
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag  ## (d, N/2)
        dtA = self.log_dt.exp()[:, None] * A                ## (d, N/2)
        C = self.C * (torch.exp(dtA) - 1) / A               ## ZOH: C_n * B_bar_n
        ## Vandermonde: K[c, l] = 2 Re sum_n C[c, n] exp(l * dtA[c, n])
        return 2 * torch.einsum("dn,dnl->dl", C, torch.exp(dtA[..., None] * torch.arange(m))).real

## Example: kernels for 2 channels, length 16 -> feed to an FFT conv (see SSMConv)
print(S4DKernel(d=2, N=8)(16).shape)  ## torch.Size([2, 16])
```
````

```{attention} Q&A
:class: dropdown
*Why is diagonal enough?*
- Almost every matrix is diagonalizable over $\mathbb{C}$ → conjugation keeps the same map → ~no loss in expressivity.
- The hard part is the init. HiPPO-LegS's normal part w/ input $B/2$ recovers the S4-LegS kernel basis as $N\to\infty$.
- DSS drops the low-rank term but keeps $B$ → noisy approximation of S4 → usually slightly worse.

*Diagonal vs DPLR in practice?*
- Comparable in almost all settings. S4D: 85% avg on LRA.
- Small init changes that break the eigenvalue structure degrade performance.

*Why conjugate pairs?*
- Real input → real kernel ⟸ states in conjugate pairs: $(\lambda_n,B_n,C_n)$ & $(\lambda_n^*,B_n^*,C_n^*)$.
    - Conjugate eigenvalues alone are not enough. $B,C$ entries must pair too.
    - → Compute $N/2$ states, take $2\,\mathrm{Re}$.

*Why parameterize $\mathrm{Re}\,\lambda$ as $-\exp(\cdot)$?*
- Forces $\mathrm{Re}\,\lambda<0$ during training → $|e^{\Delta\lambda}|<1$ → kernel cannot blow up.
```

&nbsp;

#### S5
- **What**: S4 variant w/ 1 multi-input multi-output (MIMO) diagonal SSM, computed by parallel scan. {cite:p}`smith2023simplified`
- **Why**: S4 = $d$ independent SISO SSMs + per-channel kernels + FFT. 1 MIMO SSM + scan → simpler, same efficiency.
- **How**:
    1. 1 SSM maps $\mathbf{x}_t\in\mathbb{R}^d$ → state $\mathbb{C}^{N}$ → $\mathbf{y}_t\in\mathbb{R}^d$.
    2. Diagonal $A$ w/ HiPPO-based init, ZOH.
    3. Run the recurrence w/ a [parallel scan](#parallel-scan). ❌Convolution kernel.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\Lambda\in\mathbb{C}^{N\times N}$: Diagonal state matrix.
    - $B\in\mathbb{C}^{N\times d}$: Input matrix (overrides page $N\times1$).
    - $C\in\mathbb{C}^{d\times N}$: Output matrix (overrides page $1\times N$).
    - $D\in\mathbb{R}^{d\times d}$: Diagonal skip matrix.

Forward:

$$
\mathbf{h}_t=\bar{\Lambda}\mathbf{h}_{t-1}+\bar{B}\mathbf{x}_t,\qquad \mathbf{y}_t=\mathrm{Re}\left(C\mathbf{h}_t\right)+D\mathbf{x}_t
$$
```

```{attention} Q&A
:class: dropdown
*Pros?*
- LRA avg 87.4%; Path-X 98.5%.
- Scan handles time-varying $\Delta_t$ → irregularly sampled sequences.

*Why does it matter historically?*
- Moved deep SSMs from convolution to scan → the compute path Mamba builds on.

*Cons?*
- Still LTI in practice → ❌content-based selection.
```

&nbsp;

## Selective SSMs
- **What**: SSMs w/ input-dependent $\Delta,B,C$ → linear time-varying (LTV).

&nbsp;

### Mamba
- **What**: Homogeneous attention-free, MLP-free block around a selective SSM (S6). {cite:p}`gu2023mamba`
- **Why**:
    - *Why do we need it?*
        - LTI → same dynamics for every token → ❌content-based reasoning.
        - ❌Selective copying (skip filler tokens). ❌Induction heads (recall the token after a match).
        - → LTI SSMs lag Transformers on language.
    - *Why does it work?*
        - Input-dependent $\Delta_t$ = gate: $\Delta_t$⬆️/⬇️ → reset state & absorb $x_t$ / keep state & ignore $x_t$.
        - Input-dependent $B_t$ / $C_t$ → choose what to write / read per token.
- **How**:
    1. Project $\Delta_t,B_t,C_t$ from the token. $A$ stays a learned param (diagonal, real).
    2. Discretize per token: $\bar{A}_t=\exp(\Delta_tA)$, $\bar{B}_t=\Delta_tB_t$.
    3. **Selective scan**: load params HBM → SRAM, discretize + scan in SRAM, write only outputs.
        - ❌Materialize the $(m,d,N)$ states in HBM. Recompute them in the backward pass.
    4. **Block**: in-proj $d\to2Ed$ → 2 branches.
        - Branch 1: causal depthwise conv → SiLU → selective SSM.
        - Branch 2: SiLU → gate.
        - Multiply branches → out-proj $Ed\to d$.
    5. Stack blocks w/ norm + residual.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $A\in\mathbb{R}^{N\times N}$: Diagonal, negative; 1 per channel.
    - $W_B,W_C\in\mathbb{R}^{N\times d}$: Projections, shared across channels.
    - $W_\Delta\in\mathbb{R}^{d\times d}$: Low-rank projection for $\Delta$.
    - $\mathbf{b}_\Delta\in\mathbb{R}^d$: $\Delta$ bias.
    - $D\in\mathbb{R}$: Skip, 1 per channel.
- Hyperparams:
    - $E$: Expansion factor ($E=2$).
    - $N=16$.
- Misc:
    - $\mathbf{x}_t\in\mathbb{R}^d$: Token at step $t$, all channels. $x_t$ = 1 channel of it.
    - $\Delta_t,B_t,C_t$: Input-dependent step, input & output matrices.

Model (1 channel):

$$
\begin{align*}
B_t&=W_B\mathbf{x}_t,\qquad C_t=(W_C\mathbf{x}_t)^\top,\qquad \Delta_t=\left[\mathrm{softplus}(W_\Delta\mathbf{x}_t+\mathbf{b}_\Delta)\right]_{\text{channel}}\\
\mathbf{h}_t&=\exp(\Delta_tA)\,\mathbf{h}_{t-1}+\Delta_tB_t\,x_t\\
y_t&=C_t\mathbf{h}_t+Dx_t
\end{align*}
$$

- Paper states ZOH for $\bar{B}_t$. Released code uses $\Delta_tB_t$ (exponential-Euler).
```

```{tip} Derivation
:class: dropdown
*Why is selection a generalized RNN gate?*
1. Set $N=1$, $A=-1$, $B=1$, $\Delta_t=\mathrm{softplus}(z_t)$, $z_t=\mathbf{w}^\top\mathbf{x}_t$. Use ZOH.
2. $\bar{A}_t=e^{-\Delta_t}=\frac{1}{1+e^{z_t}}=1-\sigma(z_t)$.
3. $\bar{B}_t=(-\Delta_t)^{-1}(e^{-\Delta_t}-1)\Delta_t=1-e^{-\Delta_t}=\sigma(z_t)$.
4. → $h_t=(1-g_t)h_{t-1}+g_tx_t$, $g_t=\sigma(z_t)$: the classic gated update.
```

````{important} Code
:class: dropdown
```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelectiveSSM(nn.Module):
    def __init__(self, d, N=16):
        super().__init__()
        ## S4D-Real init: A_n = -(n+1); stored as log -> A < 0 always
        self.A_log = nn.Parameter(torch.log(torch.arange(1, N + 1).float()).repeat(d, 1))  ## (d, N)
        self.D = nn.Parameter(torch.ones(d))
        self.proj_B = nn.Linear(d, N, bias=False)
        self.proj_C = nn.Linear(d, N, bias=False)
        self.proj_dt = nn.Linear(d, d)  ## paper: low-rank; full rank here for brevity
        ## Delta bias = softplus^{-1}(dt), dt log-uniform in [1e-3, 0.1] -> long memory at init
        dt = torch.exp(torch.rand(d) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
        with torch.no_grad():
            self.proj_dt.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def forward(self, x):
        ## x: (m, d)
        A = -self.A_log.exp()                              ## (d, N)
        dt = F.softplus(self.proj_dt(x))                   ## (m, d): SELECTION, depends on x_t
        B, C = self.proj_B(x), self.proj_C(x)              ## (m, N): SELECTION, depends on x_t
        A_bar = torch.exp(dt[..., None] * A)               ## (m, d, N)
        Bx = dt[..., None] * B[:, None, :] * x[..., None]  ## (m, d, N): B_bar = dt * B (as in released code)
        h, ys = x.new_zeros(A.shape), []                   ## h: (d, N)
        for t in range(x.shape[0]):
            ## Time-varying -> no conv kernel; real kernels run this as a fused parallel scan
            h = A_bar[t] * h + Bx[t]
            ys.append(h @ C[t])                            ## (d,)
        return torch.stack(ys) + self.D * x

## Example
x = torch.randn(5, 4)
print(SelectiveSSM(d=4)(x).shape)  ## torch.Size([5, 4])
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- Linear-time training. $\mathcal{O}(1)$ per-token decode. ❌KV cache → bigger batches → 4–5× inference throughput vs a same-size Transformer.
- Mamba-3B beats same-size Transformers & matches Transformers 2× its size.
- Induction heads: trained at length 256, generalizes to 1M tokens (4000×).

*Cons?*
- Fixed-size state → lossy → ❌exact copy/retrieval over long context → [hybrids](models.md#mamba-hybrids).
- Scan = elementwise ops → ❌tensor cores. Cost ∝ $N$ → $N$ stuck at 16 → [Mamba-2](#mamba-2).

*Why can't Mamba use the convolution trick?*
- $\Delta_t,B_t,C_t$ change per token → no single kernel → scan.

*Is the recurrent mode too expensive?*
- FLOPs: recurrence $\mathcal{O}(mdN)$ vs convolution $\mathcal{O}(md\log m)$ → recurrence wins for small $N$.
- Real bottleneck = memory IO → fuse into SRAM.

*Why is $A$ not selective?*
- $A$ acts only through $\bar{A}_t=\exp(\Delta_tA)$ → selective $\Delta_t$ already makes $\bar{A}_t$ input-dependent.

*Why real $A$, not complex?*
- Real (S4D-Real) works better on discrete data (text, DNA).
- Complex can help on continuous signals (audio, video).

*Why the short conv?*
- Cheap local mixing of neighboring tokens before the SSM.
- Long treated as essential for recurrent LMs. [Mamba-3](#mamba-3) removes it.

*Decode memory vs KV cache?*
- State per layer = $E\cdot d\cdot N$ numbers (+ small conv buffer), constant in $m$.
- $d=2560$, $E=2$, $N=16$ → 81,920 numbers = MHA KV cache ($2md$) of only $m=16$ tokens.

*Why no positional encoding?*
- Recurrence processes tokens in order → order is built in.
```

&nbsp;

### Mamba-2
- **What**: Mamba w/ a scalar-times-identity state matrix & multi-head state, computed as a structured matrix (SSD). {cite:p}`dao2024transformers`
- **Why**:
    - *Why do we need it?*
        - Mamba-1 scan → ❌tensor cores → slow training. Cost ∝ $N$ → small state.
    - *Why does it work?*
        - Scalar decay per head → SSM = masked attention-like matrix $L\circ CB^\top$: **SSD** (State Space Duality).
        - → Compute it w/ chunked matmuls → tensor cores → large $N$ is cheap.
- **How**:
    1. $\bar{A}_t=a_tI$, $a_t=\exp(\Delta_tA)$ → 1 scalar per head per token.
    2. Head state $H_t\in\mathbb{R}^{N\times P}$: decay by $a_t$, add rank-1 write $B_t\mathbf{x}_t^\top$, read w/ $C_t$.
    3. **Training** (chunked SSD, chunk length $Q$):
        1. Inside each chunk: quadratic attention-like form → matmul.
        2. Each chunk's final state, from zero init → matmul.
        3. Scan across $m/Q$ chunk states → short recurrence.
        4. Add each incoming state's effect to the chunk's outputs → matmul.
    4. **Inference**: plain recurrence, $\mathcal{O}(NP)$ per head per token.
    5. **Block changes**:
        - Project $\mathbf{x},\Delta,B,C$ in parallel at block start (like $Q,K,V$) → tensor parallelism.
        - Extra norm before the out-proj → stability at scale.
        - $B,C$ shared across heads, $\mathbf{x}$ per head (multi-value attention analog).

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $P$: Head dim (64 or 128).
    - $Q$: Chunk length.
- Misc:
    - $a_t\in(0,1)$: Scalar decay per head.
    - $\mathbf{x}_t\in\mathbb{R}^P$: Head input (overrides page scalar $x_t$).
    - $\mathbf{y}_t\in\mathbb{R}^P$: Head output.
    - $B_t,C_t\in\mathbb{R}^N$: Input & output vectors; $\Delta_t$ folded into $B_t$.
    - $H_t\in\mathbb{R}^{N\times P}$: Head state.
    - $X,Y\in\mathbb{R}^{m\times P}$: Stacked inputs & outputs.
    - $B_{1:m},C_{1:m}\in\mathbb{R}^{m\times N}$: Stacked $B_t^\top$, $C_t^\top$.

Recurrent form:

$$
H_t=a_tH_{t-1}+B_t\mathbf{x}_t^\top,\qquad \mathbf{y}_t=H_t^\top C_t
$$

Quadratic (dual) form:

$$
Y=\left(L\circ C_{1:m}B_{1:m}^\top\right)X,\qquad L_{ts}=\begin{cases}\prod_{r=s+1}^{t}a_r&s\le t\\0&s>t\end{cases}
$$

- $\circ$: Elementwise product.
- $L\circ C_{1:m}B_{1:m}^\top$ is **$N$-semiseparable**: every submatrix inside the lower triangle has rank $\le N$.
- $a_r=1$ → $L$ = causal mask → causal [linear attention](attn.md#linear-attention) w/ $Q=C$, $K=B$, $V=X$.
```

```{tip} Derivation
:class: dropdown
*Why are the 2 forms equal?*
1. Unroll: $H_t=\sum_{s\le t}\left(\prod_{r=s+1}^{t}a_r\right)B_s\mathbf{x}_s^\top$.
2. Read: $\mathbf{y}_t^\top=C_t^\top H_t=\sum_{s\le t}L_{ts}\,(C_t^\top B_s)\,\mathbf{x}_s^\top$.
3. Stack rows over $t$ → $Y=\left(L\circ C_{1:m}B_{1:m}^\top\right)X$.

*Why must the decay be scalar?*
- Diagonal (non-scalar) decay → different $L$ per state dim → $N$ masks, ❌1 masked matmul.
```

````{important} Code
:class: dropdown
```python
import torch

class SSD:
    ## 1 head: a: (m,) | B, C: (m, N) | X: (m, P)
    def recurrent(self, a, B, C, X):
        H, Y = X.new_zeros(B.shape[1], X.shape[1]), []  ## H: (N, P)
        for t in range(X.shape[0]):
            H = a[t] * H + torch.outer(B[t], X[t])       ## scalar decay + rank-1 write
            Y.append(C[t] @ H)                           ## (P,)
        return torch.stack(Y)

    def quadratic(self, a, B, C, X):
        ## L[t, s] = a_{s+1} * ... * a_t via log-cumsum differences (real kernels use a stable segsum)
        log_cum = torch.cumsum(torch.log(a), 0)
        L = torch.exp(log_cum[:, None] - log_cum[None, :]).tril()
        return (L * (C @ B.T)) @ X                      ## masked "attention": C ~ Q, B ~ K, X ~ V

## Example: both forms give the same output
torch.manual_seed(0)
m, N, P = 6, 4, 3
a = 0.5 + 0.5 * torch.rand(m, dtype=torch.float64)   ## a_t = exp(dt_t * A) in (0, 1)
B, C = torch.randn(m, N, dtype=torch.float64), torch.randn(m, N, dtype=torch.float64)
X = torch.randn(m, P, dtype=torch.float64)
ssd = SSD()
print(torch.allclose(ssd.recurrent(a, B, C, X), ssd.quadratic(a, B, C, X)))  ## True
```
````

```{dropdown} Table: SSD Instances
| Layer | $L_{ts}$, $s\le t$ | Transition |
|:--|:--|:--|
| Causal linear attention | $1$ | ❌Decay |
| Mamba-2 | $\prod_{r=s+1}^{t}a_r$ | Data-dependent scalar decay |
| Mamba-3 | Mamba-2 decay × 2-band (trapezoid) mask | Data-dependent decay + rotation |
| Gated DeltaNet | ❌Not of this form | Matrix $\alpha_t(I-\beta_t\mathbf{k}_t\mathbf{k}_t^\top)$ |
| Softmax attention | ❌Not of this form | ❌Finite state |
```

```{attention} Q&A
:class: dropdown
*Pros?*
- SSD layer 2–8× faster than Mamba-1's fused scan. Faster than FlashAttention-2 from 2K tokens.
- State 8× larger than Mamba-1 w/ minimal slowdown ($N$=64–256 vs 16).

*Cons?*
- Scalar decay < diagonal decay in expressivity. Trade: expressivity for speed.
- Decay $\in(0,1)$ → state can only shrink, never rotate or flip sign → ❌state tracking (e.g., parity) → [Mamba-3](#mamba-3).
- At short lengths (~2K), a full Transformer can still train faster.

*Are Transformers really SSMs?*
- Only linear (kernel) attention. It is SSD w/ $a_t=1$.
- Softmax: $\exp(\mathbf{q}^\top\mathbf{k})$ needs an infinite-dim feature map → no finite state.

*Why does semiseparability matter?*
- Off-diagonal blocks have rank $\le N$ → each chunk passes all info forward through an $N\times P$ state.
- → Chunked algorithm: matmuls inside chunks, short scan across chunks.

*Mamba-2 vs Gated DeltaNet?*
- [Gated DeltaNet](attn.md#gated-deltanet) = Mamba-2's gated state + delta-rule erase → better associative recall.
```

&nbsp;

#### Mamba-3
- **What**: Mamba-2 + exponential-trapezoidal discretization + complex-valued (RoPE) state + MIMO update. {cite:p}`lahoti2026mamba`
- **Why**: Inference-first design. 3 gaps in Mamba-2:
    - Discretization = unjustified first-order heuristic.
    - Real positive decay → ❌state tracking (parity, modular arithmetic).
    - Decode = memory-bound rank-1 update → compute units idle.
- **How**:
    1. **Exponential-trapezoidal**: write a data-dependent blend of $B_{t-1}\mathbf{x}_{t-1}^\top$ & $B_t\mathbf{x}_t^\top$ → implicit width-2 conv inside the recurrence.
    2. **Complex state**: data-dependent rotation per pair of state dims → equivalent to RoPE w/ data-dependent angles on $B_t$ & $C_t$.
    3. **MIMO**: rank-$R$ matmul state update instead of a rank-1 outer product → more FLOPs per state load, same state size.
    4. **Block**:
        - RMSNorm on $B,C$ (BCNorm, = QK-norm) → stability.
        - Post-gate RMSNorm: dropped in pure Mamba-3 models, kept in hybrids (needed for long-context extrapolation).
        - Learnable head-specific $B,C$ biases.
        - ❌Short conv.
        - Llama-style: Mamba-3 blocks alternate w/ SwiGLU MLP blocks.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $A_t<0$: Continuous decay rate; data-dependent in Mamba-3.
    - $B_t,C_t\in\mathbb{R}^N$: Continuous input & output vectors ($\Delta_t$ NOT folded in).
    - $\lambda_t\in[0,1]$: Data-dependent trapezoid weight.
    - $\alpha_t,\beta_t,\gamma_t$: Discrete coefficients.
    - $\theta_t$: Data-dependent rotation frequency.
    - $\phi_t$: Cumulative angle.
    - $R(\cdot)$: Block-diagonal $2\times2$ rotation matrix.
- Other symbols follow [Mamba-2](#mamba-2).

Exponential-trapezoidal recurrence:

$$
H_t=\alpha_tH_{t-1}+\beta_tB_{t-1}\mathbf{x}_{t-1}^\top+\gamma_tB_t\mathbf{x}_t^\top,\qquad \mathbf{y}_t=H_t^\top C_t
$$

$$
\alpha_t=e^{\Delta_tA_t},\qquad \beta_t=(1-\lambda_t)\Delta_te^{\Delta_tA_t},\qquad \gamma_t=\lambda_t\Delta_t
$$

- $\lambda_t=1$ → exponential-Euler (Mamba-1/2 code).
- $\lambda_t=\frac{1}{2}$ → classical trapezoid rule.

Complex state as data-dependent RoPE (exponential-Euler case):

$$
\phi_t=\sum_{r\le t}\Delta_r\theta_r,\qquad C_t^\top R(\phi_t-\phi_s)B_s=\left(R(\phi_t)^\top C_t\right)^\top\left(R(\phi_s)^\top B_s\right)
$$

- Step $s\to t$ rotates each state pair by $\phi_t-\phi_s$.
- 2-D rotations commute → split into per-position rotations of $C_t$ & $B_s$ → same trick as [RoPE](pos.md#rope) on $Q=C$, $K=B$.
```

````{important} Code
:class: dropdown
```python
import torch

## Reuses SSD from this page
class ExpTrapezoidalSSM:
    ## Real-valued SISO core of Mamba-3 (complex/RoPE & MIMO omitted)
    def __call__(self, dt, A, lam, B, C, X):
        ## dt, A, lam: (m,) | B, C: (m, N) | X: (m, P)
        alpha = torch.exp(dt * A)
        beta = (1 - lam) * dt * alpha
        gamma = lam * dt
        H = X.new_zeros(B.shape[1], X.shape[1])  ## (N, P)
        prev, Y = torch.zeros_like(H), []        ## prev = B_{t-1} x_{t-1}^T
        for t in range(X.shape[0]):
            cur = torch.outer(B[t], X[t])
            ## Trapezoid: blend both interval endpoints -> width-2 conv on B_t x_t inside the recurrence
            H = alpha[t] * H + beta[t] * prev + gamma[t] * cur
            Y.append(C[t] @ H)
            prev = cur
        return torch.stack(Y)

## Example: lam = 1 recovers Mamba-2 (exp-Euler)
torch.manual_seed(0)
m, N, P = 6, 4, 3
dt, A = torch.rand(m, dtype=torch.float64), -torch.rand(m, dtype=torch.float64)
B, C = torch.randn(m, N, dtype=torch.float64), torch.randn(m, N, dtype=torch.float64)
X = torch.randn(m, P, dtype=torch.float64)
y3 = ExpTrapezoidalSSM()(dt, A, torch.ones(m, dtype=torch.float64), B, C, X)
y2 = SSD().recurrent(torch.exp(dt * A), dt[:, None] * B, C, X)
print(torch.allclose(y3, y2))  ## True
```
````

```{dropdown} Table: SSM Lineage
| Model | $A$ structure | Time | Training compute | Discretization |
|:--|:--|:--|:--|:--|
| S4 | NPLR, complex (HiPPO) | LTI | FFT conv (Cauchy kernel) | Bilinear |
| S4D | Diagonal, complex | LTI | FFT conv (Vandermonde) | Bilinear or ZOH |
| S5 | Diagonal, complex, MIMO | LTI | Parallel scan | ZOH |
| Mamba | Diagonal, real | LTV | Fused selective scan | Exp-Euler (code) |
| Mamba-2 | Scalar × $I$ | LTV | Chunked SSD (matmul) | Exp-Euler |
| Mamba-3 | Scalar × $I$ + rotation | LTV | Chunked SSD (matmul) | Exp-trapezoidal |
```

```{attention} Q&A
:class: dropdown
*Results?*
- 1.5B scale: SISO +0.6 pts avg downstream accuracy over Gated DeltaNet. MIMO +1.2 more (+1.8 total).
- MIMO w/ $N=64$ ≈ Mamba-2 w/ $N=128$ in perplexity → half the state.
- MIMO: up to 4× decode FLOPs vs Mamba-2 at similar wall-clock latency.

*Why does a complex state fix parity?*
- Real positive decay → each state dim only shrinks → ❌toggle.
- Rotation by $\pi$ = sign flip → rotate by $\pi$ on each 1-bit → state sign = parity.
- Arithmetic tasks: Mamba-2 & Mamba-3 w/o RoPE ≈ chance; Mamba-3 near-perfect.

*Why can the short conv go?*
- Trapezoid rule = width-2 data-dependent conv on $B_t\mathbf{x}_t^\top$ inside the recurrence.
- $B,C$ biases add data-independent, conv-like terms.
- Empirical result, not a proof.

*Why does MIMO not slow decode?*
- Decode is memory-bound: time ≈ load the state.
- Rank-$R$ update reuses each loaded state $R$× → FLOPs⬆️, latency ≈ same.

*Why not fix $\lambda_t=\frac{1}{2}$?*
- Second-order error $\mathcal{O}(\Delta_t^3)$ needs $\lambda_t=\frac{1}{2}+\mathcal{O}(\Delta_t)$.
- Unconstrained, data-dependent $\lambda_t$ is better empirically.
```

&nbsp;
