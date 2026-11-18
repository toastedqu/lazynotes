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
# Model Compression
Shrink a trained network's memory, bandwidth & compute at ~fixed function: fewer bits (quantization), fewer weights (pruning), fewer ranks (low-rank factorization). Smaller student → [Distillation](../nlp/post/kd.md). Number formats → [Precision](sys.md#precision). KV-cache compression → [Cache Quantization](../nlp/infer/cache.md#quantization).

Notations:
- $W\in\mathbb{R}^{d_\text{out}\times d_\text{in}}$: Weight of a linear layer, $\mathbf{y}=W\mathbf{x}$
- $\hat{W}$: Compressed weight
- $X\in\mathbb{R}^{d_\text{in}\times m}$: Calibration inputs to the layer, 1 column per token
- $H=2XX^T\in\mathbb{R}^{d_\text{in}\times d_\text{in}}$: Hessian of the layer-wise loss $\|WX-\hat{W}X\|_F^2$ w.r.t. 1 row of $\hat{W}$ (same for every row)
- $b$: #Bits per value
- $s$: Quantization scale (grid step)
- $z$: Zero-point
- $q$: Integer code
- $M\in\{0,1\}^{d_\text{out}\times d_\text{in}}$: Pruning mask (1 = keep)

Overrides of global notation: $m$ = #calibration tokens. $X$ stores samples as columns (matches $\mathbf{y}=W\mathbf{x}$). $i$ = row (output-channel) idx of $W$, ❌sample idx. $\hat{\ }$ = compressed approximation, ❌statistical estimator.

&nbsp;

## Quantization
- **What**: Map real values onto a uniform integer grid: $\hat{x}=s(q-z)$.
- **Why**: Bytes.
    - FP16 → INT4: 4× ⬇️weight memory.
    - LLM decode is memory-bound → fewer bytes moved per weight → ⬇️latency.
    - INT8 matmul units → ⬆️math throughput, only if activations are also integer.
- **How**:
    1. Pick a clipping range $[x_\text{min},x_\text{max}]$ for a group of values. Extend it to contain 0.
    2. Scale $s$ = range ÷ #intervals ($2^b-1$). Zero-point $z$ = the integer code of real 0.
    3. Quantize: divide by $s$ → round → shift by $z$ → clamp to the $b$-bit range.
    4. Dequantize: $\hat{x}=s(q-z)$, usually fused into the matmul kernel.
    - **Asymmetric**: $z$ free → grid need not be centered → fits skewed ranges (e.g., post-ReLU).
    - **Symmetric**: $z=0$, range $[-\max|x|,\max|x|]$ → cheaper integer matmul.
    - **Granularity**: 1 $(s,z)$ per tensor / per output channel (row of $W$) / per group of $g$ consecutive weights in a row.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $g$: Group size (#weights sharing 1 $(s,z)$).
- Misc:
    - $[x_\text{min},x_\text{max}]$: Clipping range of the group, w/ $x_\text{min}\leq0\leq x_\text{max}$.

Asymmetric (affine):

$$
s=\frac{x_\text{max}-x_\text{min}}{2^b-1},\qquad z=\text{round}\left(-\frac{x_\text{min}}{s}\right),\qquad q=\text{clamp}\left(\text{round}\left(\frac{x}{s}\right)+z,\ 0,\ 2^b-1\right),\qquad \hat{x}=s(q-z)
$$

Symmetric:

$$
s=\frac{\max|x|}{2^{b-1}-1},\qquad z=0,\qquad q=\text{clamp}\left(\text{round}\left(\frac{x}{s}\right),\ -(2^{b-1}-1),\ 2^{b-1}-1\right)
$$

- Code $-2^{b-1}$ unused → grid symmetric around 0.

Round-trip error:
- $x$ inside the grid range: $|x-\hat{x}|\leq\frac{s}{2}$ (**rounding error**).
- Rounding error $\sim\mathcal{U}\left(-\frac{s}{2},\frac{s}{2}\right)$ → $\mathbb{E}\left[(x-\hat{x})^2\right]=\frac{s^2}{12}$.
- $x$ outside the grid range: $\hat{x}$ = nearest endpoint → error = distance to it, unbounded (**clipping error**).

Storage per weight (FP16 $s$ & $z$):

$$
b+\frac{32}{g}\ \text{bits}
$$

- $b=4$, $g=128$ → 4.25 bits.
```

````{important} Code
:class: dropdown
```python
import torch

class Quantizer:
    def __init__(self, b=4, symmetric=True, group=None):
        ## group: None -> per-tensor, -1 -> per-channel (1 row), g -> g consecutive weights
        self.b, self.sym, self.group = b, symmetric, group

    def _groups(self, w):
        if self.group is None:
            return w.reshape(1, -1)
        return w if self.group == -1 else w.reshape(-1, self.group)

    def __call__(self, w):
        x = self._groups(w)                      ## (#groups, group size)
        if self.sym:
            qmax = 2 ** (self.b - 1) - 1
            s = x.abs().amax(-1, keepdim=True).clamp(min=1e-8) / qmax
            x_hat = s * (x / s).round().clamp(-qmax, qmax)
        else:
            ## range must contain 0 -> real 0 maps to a valid integer code z
            lo = x.amin(-1, keepdim=True).clamp(max=0)
            hi = x.amax(-1, keepdim=True).clamp(min=0)
            s = (hi - lo).clamp(min=1e-8) / (2 ** self.b - 1)
            z = (-lo / s).round()
            q = ((x / s).round() + z).clamp(0, 2 ** self.b - 1)
            x_hat = s * (q - z)
        return x_hat.reshape(w.shape)            ## fake-quantized: on the grid, still float

## Example
torch.manual_seed(0)
W = torch.randn(4, 8)
W[0, 0] = 10.0                                   ## 1 outlier inflates s for its whole group
for g in [None, -1, 4]:
    print(g, f"{(Quantizer(b=4, group=g)(W) - W).abs().mean():.3f}")   ## error shrinks w/ finer groups
```
````

```{attention} Q&A
:class: dropdown
*Why must real 0 be exactly representable?*
- Zero padding & ReLU outputs are exact zeros. A biased 0 → systematic error at every padded position.
- → $z$ is an integer code & the range is extended to contain 0.

*Symmetric vs asymmetric?*
- Symmetric: $z=0$ → integer matmul needs no zero-point cross terms. Wastes ~1 bit of grid on one-signed data.
- Asymmetric: full grid on skewed data. Cross terms cost extra compute.
- Common practice: symmetric weights (≈ zero-centered), asymmetric activations when one-signed.

*Rounding vs clipping error?*
- Range ⬇️ → $s$⬇️ → rounding error ⬇️, clipping error ⬆️.
- Min-max range → 0 clipping error, max rounding error. 1 outlier → $s$⬆️ for the whole group.
- → Pick the range by MSE, percentile, or KL instead of min-max.

*Why per-output-channel scales for weights & per-token scales for activations, ❌per-input-channel, in an INT×INT GEMM?*
- Symmetric: $y_i=\sum_jW_{ij}x_j\approx s^W_i\,s^x\sum_jq^W_{ij}q^x_j$.
- Scale indexed by the output row $i$ or by the token → factors out of $\sum_j$ → pure integer GEMM, then 1 rescale.
- Scale indexed by the input channel $j$ → stays inside $\sum_j$ → ❌integer GEMM.
- Per-group weight scales also sit inside $\sum_j$ → fine for weight-only (dequantize to FP16 first), ❌for INT×INT GEMM.
- Per-input-channel scaling is still possible if folded into an adjacent op → [SmoothQuant](#smoothquant).

*Weight-only vs weight+activation?*
- Weight-only (e.g., W4A16): weights dequantized in the kernel, math in FP16. Win = bytes → helps memory-bound decode.
- W8A8: INT8 GEMM. Win = math throughput → helps compute-bound prefill & large batch.
- Large batch → decode becomes compute-bound → weight-only gives ⬇️speedup.

*Why does rounding noise not destroy the network?*
- Assume rounding errors $e_j$ zero-mean & independent across weights.
- → $\text{Var}\left(\sum_je_jx_j\right)=\frac{s^2}{12}\sum_jx_j^2$: errors add in variance, ❌in magnitude.
- Trained nets tolerate small weight noise. Breaks when 1 term dominates the sum → outliers.

*INT grid vs FP8 / NF4?*
- INT: uniform levels → best for ~uniform values.
- FP8: log-spaced levels → more dynamic range for heavy-tailed tensors → [Precision](sys.md#precision).
- NF4: levels at normal quantiles → equal mass per level for Gaussian weights → [QLoRA](../nlp/post/sft.md#qlora).
```

&nbsp;

### PTQ
- **Name**: Post-Training Quantization
- **What**: Quantize a trained model w/o training, using a small calibration set.
- **Why**: QAT = another training run → too costly at LLM scale.
    - PTQ: ~100s of calibration samples, minutes to hours, ❌labels, ❌gradients.
- **How**:
    1. Run calibration data → record activation ranges & per-layer inputs $X$.
    2. Choose $(s,z)$ per tensor / channel / group.
    3. Optional: reduce the error beyond round-to-nearest (**RTN**).
        - Compensate rounding error w/ other weights → [GPTQ](#gptq).
        - Rescale channels before rounding → [SmoothQuant](#smoothquant), [AWQ](#awq).
    - **Static** activation scales: fixed from calibration. **Dynamic**: recomputed per token at runtime.
    - **LLM outliers**: at ≥6.7B params, a few hidden dims carry magnitudes up to 20× larger, in all layers. {cite:p}`dettmers2022llmint8`
        - → 1 per-tensor / per-token activation scale fits the outliers → all other dims get ~0 levels → INT8 activations fail.
        - LLM.int8(): outlier dims ($|x|\geq6$) in FP16, the rest (>99.9% of values) in INT8.

```{note} Math
:class: dropdown
Objective (layer-wise reconstruction):

$$
\hat{W}=\arg\min_{\hat{W}\in\mathcal{Q}}\|WX-\hat{W}X\|_F^2=\arg\min_{\hat{W}\in\mathcal{Q}}\sum_{i=1}^{d_\text{out}}(\mathbf{w}_i-\hat{\mathbf{w}}_i)^TXX^T(\mathbf{w}_i-\hat{\mathbf{w}}_i)
$$

- $\mathcal{Q}$: Set of matrices whose entries lie on the quantization grid.
- $\mathbf{w}_i^T$: Row $i$ of $W$.
- Rows decouple → $d_\text{out}$ independent quadratics, each w/ Hessian $H=2XX^T$.
- RTN ignores $XX^T$ → optimal only if $XX^T$ is diagonal (uncorrelated inputs) → objective separates per weight.
```

```{dropdown} Table: LLM PTQ Methods
| Method | Format | Quantizes | Mechanism | Calibration use |
|:--|:--|:--|:--|:--|
| RTN | Any | Weights | Round each weight independently | ❌ |
| LLM.int8() | W8A8 + FP16 outlier dims | Weights & activations | Mixed-precision decomposition | ❌ (outlier dims found at runtime, $\lvert x\rvert\geq6$) |
| [SmoothQuant](#smoothquant) | W8A8 | Weights & activations | Move activation outliers into weights | Per-channel max $\lvert x_j\rvert$ |
| [GPTQ](#gptq) | W3–W4, A16 | Weights | 2nd-order error compensation | $H=2XX^T$ |
| [AWQ](#awq) | W4, A16 | Weights | Scale salient input channels up | Per-channel mean $\lvert x_j\rvert$ + grid search |
```

```{attention} Q&A
:class: dropdown
*Pros?*
- Cheap: forward passes over a few hundred samples.
- ❌Training pipeline, ❌labels.

*Cons?*
- Low bits (≤3-bit weights, ≤4-bit activations) → accuracy drops faster than w/ QAT.
- Can overfit the calibration distribution (worse for reconstruction methods).

*Why are weights easier to quantize than activations?*
- Weights: static, known offline, ≈ zero-centered & light-tailed → per-channel/group scales fit well.
- Activations: input-dependent, must be quantized at runtime, w/ persistent outlier channels.

*Why does RTN collapse at 3-bit on large LLMs?*
- Per-weight errors are independent of $X$ → they add up in $WX$.
- 3-bit → $s$ large → total error too large. GPTQ keeps 3-bit usable by cancelling errors through correlated inputs.

*When is PTQ NOT enough?*
- Small, compact models (e.g., MobileNet) → little redundancy to absorb error → QAT.
- Very low bits → QAT or quantization-aware fine-tuning.
```

&nbsp;

#### GPTQ
- **What**: Column-by-column weight rounding w/ 2nd-order error compensation on the not-yet-rounded columns. {cite:p}`frantar2023gptq`
- **Why**: RTN rounds each weight independently.
    - Inputs are correlated → error of 1 weight can be cancelled by adjusting others.
    - Exact greedy [OBS](#obs) per weight (OBQ) → ~1 GPU hour for 25M params → ❌175B.
- **How**:
    1. Collect $X$ (128 random 2048-token C4 segments) → $H=2XX^T$ + damping (1% of mean diagonal).
    2. For column $j$ = left → right: round column $j$ of all rows at once.
    3. Spread its rounding error onto columns $>j$ via the OBS update.
    4. Same column order for every row → 1 shared $H^{-1}$ sequence → precompute it w/ 1 Cholesky.
    5. **Lazy batch updates**: apply updates within a block of $B=128$ columns. Flush to later columns once per block → compute-bound, ❌memory-bound.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $B$: Lazy-update block size.
- Misc:
    - $F$: Indices of not-yet-quantized columns, incl. the curr column $j$.
    - $H_F$: $H$ restricted to rows & columns in $F$.
    - $U$: Upper Cholesky factor, $H^{-1}=U^TU$.
    - $\text{quant}(\cdot)$: RTN onto the fixed grid.

Process (column $j$, every row in parallel):
1. $\hat{W}_{:,j}=\text{quant}(W_{:,j})$
2. OBS update of the remaining columns:

    $$
    W_{:,k}\leftarrow W_{:,k}-\frac{W_{:,j}-\hat{W}_{:,j}}{[H_F^{-1}]_{jj}}[H_F^{-1}]_{jk}\qquad k\in F\setminus\{j\}
    $$

3. Equivalent Cholesky form:

    $$
    W_{:,k}\leftarrow W_{:,k}-\frac{W_{:,j}-\hat{W}_{:,j}}{U_{jj}}U_{jk}\qquad k>j
    $$

4. $F\leftarrow F\setminus\{j\}$.
```

```{tip} Derivation
:class: dropdown
*Why does row $j$ of the Cholesky factor give the OBS update?*
1. Let $A=H^{-1}$. Upper Cholesky $A=U^TU$ → row 0: $U_{0k}=\frac{A_{0k}}{\sqrt{A_{00}}}$.
2. Peel row 0 off: $A-\mathbf{u}_0\mathbf{u}_0^T$, w/ $\mathbf{u}_0^T$ = row 0 of $U$. Its remaining block:

    $$
    A_{FF}-\frac{A_{F0}A_{0F}}{A_{00}}
    $$

3. Block-inverse identity: this Schur complement $=(H_{FF})^{-1}$ = the OBS inverse after removing index 0.
4. Recurse → $U_{jj}=\sqrt{[H_F^{-1}]_{jj}}$ & $U_{jk}=\frac{[H_F^{-1}]_{jk}}{\sqrt{[H_F^{-1}]_{jj}}}$ at step $j$.
5. → $\frac{U_{jk}}{U_{jj}}=\frac{[H_F^{-1}]_{jk}}{[H_F^{-1}]_{jj}}$ → the 2 update forms are identical.
6. → All $d_\text{in}$ inverse updates = 1 Cholesky, precomputed. Numerically more stable than repeated rank-1 updates.
```

````{important} Code
:class: dropdown
```python
import torch

class GPTQ:
    def __init__(self, W, X, damp=0.01):
        self.W = W.clone()                                     ## (d_out, d_in)
        H = 2 * X @ X.T                                        ## (d_in, d_in)
        H += damp * H.diag().mean() * torch.eye(H.shape[0])    ## damping -> H^{-1} well-conditioned
        ## row j of U = OBS update coefs once columns < j are fixed
        self.U = torch.linalg.cholesky(torch.linalg.inv(H), upper=True)

    def __call__(self, b=4):
        W, U = self.W.clone(), self.U
        qmax = 2 ** (b - 1) - 1
        s = W.abs().amax(1) / qmax                             ## per-row grid, fixed BEFORE updates
        Q = torch.zeros_like(W)
        for j in range(W.shape[1]):
            Q[:, j] = (W[:, j] / s).round().clamp(-qmax, qmax) * s   ## round column j, all rows
            err = (W[:, j] - Q[:, j]) / U[j, j]
            ## push the error onto columns not yet rounded (lazy blocking omitted)
            W[:, j + 1:] -= err[:, None] * U[j, j + 1:][None, :]
        return Q

## Example
torch.manual_seed(0)
W = torch.randn(8, 16)
X = torch.randn(16, 4) @ torch.randn(4, 256) + 0.1 * torch.randn(16, 256)   ## correlated inputs
s = W.abs().amax(1, keepdim=True) / 7
rtn = (W / s).round().clamp(-7, 7) * s
gptq = GPTQ(W, X)(b=4)
print(f"RTN {(rtn @ X - W @ X).norm():.2f} vs GPTQ {(gptq @ X - W @ X).norm():.2f}")   ## GPTQ << RTN
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- 175B params in ~4 GPU hours, 3–4-bit weights, negligible accuracy loss.
- 3-bit usable where RTN collapses.
- End-to-end speedup vs FP16: ~3.25× on A100, ~4.5× on A6000 (memory-bound generation).

*Cons?*
- Reconstruction can overfit the calibration set → larger degradation than AWQ under calibration-domain shift.
- Some models (e.g., LLaMA-7B, OPT-66B) need a column-reordering trick to work well.

*Why can column order be arbitrary instead of greedy (as in OBQ)?*
- Empirically, greedy order barely beats a fixed order on large layers.
- Fixed order → all rows share $F$ → share $H_F^{-1}$ → 1 inverse sequence instead of $d_\text{out}$.

*Why is the OBS update exact here?*
- Layer-wise loss is exactly quadratic in $\hat{W}$ → 2nd-order Taylor = the loss, ❌approximation.
- The only approximations: greedy one-column-at-a-time & the fixed grid.

*Why damping?*
- $XX^T$ can be singular (dead or collinear input dims) → $H^{-1}$ explodes → wild updates.
- $+\lambda I$ (1% of mean diag) → stable. Larger models also need the Cholesky form.

*How does grouping interact?*
- Group scales are set when the group's 1st column is reached, from the already-updated weights.
- OPT-175B / BLOOM-176B at 3-bit: $g=128$ (≈ +0.15 bits) → only 0.1–0.3 perplexity off FP16.

*GPTQ vs SparseGPT?*
- Same column-wise OBS loop. Target per weight = grid point (GPTQ) vs 0 (SparseGPT) → [SparseGPT](#sparsegpt).
```

&nbsp;

#### SmoothQuant
- **What**: Per-channel rescaling that moves activation outliers into the weights, enabling W8A8 PTQ. {cite:p}`xiao2023smoothquant`
- **Why**: Activation outliers (~100× typical values) sit in a few fixed input channels.
    - Per-input-channel activation scales would fix it → ❌INT8 GEMM.
    - Weights are easy to quantize → they can absorb some of the range.
- **How**:
    1. Divide input channel $j$ of $\mathbf{x}$ by $c_j$. Multiply column $j$ of $W$ by $c_j$ → exact same output.
    2. $c_j$ balances the max of activation channel $j$ against the max of weight column $j$.
    3. Fold $\text{diag}(\mathbf{c})^{-1}$ into the previous op (LayerNorm / linear) offline → 0 runtime cost.
        - Input from a residual add → ❌foldable → extra scaling on the residual branch.
    4. Quantize weights & activations to INT8 → INT8 GEMM.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\alpha\in[0,1]$: Migration strength. Default 0.5 (OPT, BLOOM). 0.75 for GLM-130B.
- Misc:
    - $\mathbf{c}\in\mathbb{R}^{d_\text{in}}$: Per-input-channel smoothing factor (paper's $\mathbf{s}$).

Equivalent transform:

$$
\mathbf{y}=W\mathbf{x}=\big(W\,\text{diag}(\mathbf{c})\big)\big(\text{diag}(\mathbf{c})^{-1}\mathbf{x}\big)
$$

Smoothing factor:

$$
c_j=\frac{\max|x_j|^{\alpha}}{\max|W_{:,j}|^{1-\alpha}}
$$

- $\max|x_j|$: Over calibration tokens.
- New channel maxima: activation $\max|x_j|^{1-\alpha}\max|W_{:,j}|^{1-\alpha}$, weight $\max|x_j|^{\alpha}\max|W_{:,j}|^{\alpha}$.
- $\alpha=0.5$ → both $=\sqrt{\max|x_j|\max|W_{:,j}|}$ → difficulty split evenly.
```

```{attention} Q&A
:class: dropdown
*Pros?*
- Training-free W8A8 for OPT, BLOOM, Llama, Mistral, Mixtral, … up to 530B.
- Up to 1.56× speedup & 2× memory reduction vs FP16.

*Cons?*
- $\alpha$ is model-dependent: too high → weights become hard to quantize.
- Needs a preceding op to fold $\text{diag}(\mathbf{c})^{-1}$ into.

*Why not keep outliers in FP16 like LLM.int8()?*
- Mixed-precision decomposition → extra kernels & scatter/gather → slower than FP16 in many cases.
- SmoothQuant keeps 1 uniform INT8 GEMM.

*Why per-channel outliers in the first place?*
- Outliers persist in the same channels across tokens → a per-channel issue, invisible to per-token scaling.
```

&nbsp;

#### AWQ
- **Name**: Activation-aware Weight Quantization {cite:p}`lin2024awq`
- **What**: Weight-only PTQ that scales up salient input channels before rounding; salience from activation magnitude.
- **Why**: Salience.
    - ~0.1–1% of weights dominate the quantization loss. Keep them FP16 → OPT-6.7B INT3-g128 perplexity 43.2 → 13.0.
    - Salient = channels w/ large **activations**. Selecting by weight norm ≈ random selection.
    - Mixed precision → hardware-unfriendly → protect salient channels by scaling instead.
- **How**:
    1. Calibration → $\mathbf{c}_X$: mean $|x_j|$ per input channel.
    2. Scale weight column $j$ by $c_j$, input by $1/c_j$ (same equivalent transform as [SmoothQuant](#smoothquant)).
    3. Scaling 1 channel rarely changes the group max → $s$ ≈ unchanged → that channel's relative error ÷$c_j$.
    4. Too large $c_j$ → group max grows → error of other channels ⬆️ → grid-search 1 exponent $\alpha$.
    5. Search weight clipping for MSE → RTN group quantization (e.g., INT4, $g=128$).

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\alpha\in[0,1]$: Scaling exponent, grid-searched.
- Misc:
    - $\mathbf{c}\in\mathbb{R}^{d_\text{in}}$: Per-input-channel scale (paper's $\mathbf{s}$).
    - $\mathbf{c}_X$: Mean $|x_j|$ per input channel over calibration tokens.
    - $Q(\cdot)$: Group-wise RTN quantizer.

Objective:

$$
\mathbf{c}^*=\arg\min_{\mathbf{c}}\left\|Q\big(W\text{diag}(\mathbf{c})\big)\big(\text{diag}(\mathbf{c})^{-1}X\big)-WX\right\|,\qquad \mathbf{c}=\mathbf{c}_X^{\alpha},\qquad \alpha^*=\arg\min_{\alpha\in[0,1]}\mathcal{L}\left(\mathbf{c}_X^{\alpha}\right)
$$

- $\alpha=0$ → $\mathbf{c}=\mathbf{1}$ → RTN. $\alpha=1$ → most aggressive.

Error of 1 weight $w$ (input $x$, group step $s$):
- $\text{RoundErr}(\cdot)$: Absolute rounding error in grid units, ≈ uniform on $[0,0.5]$ → mean ≈ 0.25 regardless of input.
- Before: $s\cdot\text{RoundErr}\left(\frac{w}{s}\right)\cdot x$
- After scaling by $c>1$: $s'\cdot\text{RoundErr}\left(\frac{wc}{s'}\right)\cdot\frac{x}{c}$
- Same mean $\text{RoundErr}$ & $s'\approx s$ → error ratio $\frac{s'}{s}\cdot\frac{1}{c}<1$.
```

````{important} Code
:class: dropdown
```python
import torch

## Reuses Quantizer from ## Quantization above
class AWQ:
    def __init__(self, W, X, quant, n_grid=20):
        self.W, self.X, self.quant, self.n_grid = W, X, quant, n_grid
        self.c_x = X.abs().mean(dim=1)                 ## (d_in,) mean |activation| per input channel

    def __call__(self):
        ref = self.W @ self.X
        best_err, best_W = float("inf"), None
        for alpha in torch.linspace(0, 1, self.n_grid):   ## alpha = 0 -> plain RTN is a candidate
            c = self.c_x.pow(alpha).clamp(min=1e-4)
            ## Q(W diag(c)) diag(c)^-1 X; deployed: 1/c folded into the PREVIOUS op, W stays quantized
            W_hat = self.quant(self.W * c) / c
            err = (W_hat @ self.X - ref).pow(2).mean()
            if err < best_err:
                best_err, best_W = err, W_hat
        return best_W

## Example
torch.manual_seed(0)
W, X = torch.randn(8, 16), torch.randn(16, 256)
X[:2] *= 10                                            ## 2 salient input channels
q = Quantizer(b=3, group=8)
rtn_err = (q(W) @ X - W @ X).pow(2).mean()
awq_err = (AWQ(W, X, q)() @ X - W @ X).pow(2).mean()
print(f"RTN {rtn_err:.3f} vs AWQ {awq_err:.3f}")      ## AWQ <= RTN on the calibration set
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- ❌Backprop, ❌reconstruction → only per-channel means → hard to overfit the calibration set.
- Works on instruction-tuned & multimodal LMs.
- Hardware-friendly: uniform INT4 group format, no mixed precision.

*Cons?*
- Weight-only → no INT math speedup at compute-bound large batch.
- Needs a preceding op to fold $1/\mathbf{c}$ into.

*AWQ vs SmoothQuant?*
- Same equivalent transform. Different goal.
- SmoothQuant: flatten **activations** for W8A8 → $\mathbf{c}$ shrinks outlier channels.
- AWQ: protect salient **weights** for W4A16 → $\mathbf{c}$ enlarges salient channels' weights.

*AWQ vs GPTQ?*
- GPTQ: changes weight values to cancel error (regression on $X$).
- AWQ: changes only per-channel scales → less calibration overfitting under domain shift.
- Orthogonal → can be combined.

*Why activation magnitude, not weight magnitude?*
- Output error of $w_jx_j$ ∝ error in $w_j$ × $|x_j|$.
- Large $|x_j|$ amplifies rounding error of every weight in column $j$.
```

&nbsp;

### QAT
- **Name**: Quantization-Aware Training {cite:p}`jacob2018quantization`
- **What**: Train or fine-tune w/ quantization simulated in the forward pass.
- **Why**: PTQ loses accuracy at low bits & on compact models (e.g., MobileNet).
    - → Let the weights learn to be robust to the rounding they will meet at inference.
- **How**:
    1. Keep FP32 master weights.
    2. Forward: **fake-quantize** weights & activations (quantize → dequantize → float on the grid).
    3. Backward: round has 0 grad almost everywhere → pass grads through via [STE](#ste).
    4. Activation ranges: EMA of observed min/max during training.
    5. Fold BatchNorm into the weights before fake-quantizing → train on the exact inference graph.
    6. Export integer weights + $(s,z)$.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $[x_\text{min},x_\text{max}]$: Clipping range of the quantized group.

Forward (fake quantization):

$$
\hat{W}=s\Big(\text{clamp}\big(\text{round}(W/s)+z,\ 0,\ 2^b-1\big)-z\Big)
$$

Backward (clipped STE):

$$
\frac{\partial\mathcal{L}}{\partial W}=\frac{\partial\mathcal{L}}{\partial\hat{W}}\odot\mathbb{1}\left[x_\text{min}\leq W\leq x_\text{max}\right]
$$

- $\mathbb{1}[\cdot]$: Elementwise indicator. Clipped values get 0 grad.

BatchNorm folding:

$$
W_\text{fold}=\frac{\gamma\,W}{\sqrt{\text{EMA}(\sigma_B^2)+\varepsilon}},\qquad \text{bias}_\text{fold}=\beta+\frac{\gamma\left(\text{bias}-\text{EMA}(\mu_B)\right)}{\sqrt{\text{EMA}(\sigma_B^2)+\varepsilon}}
$$

- $\gamma,\beta$: BN scale & shift, per output channel.
- $\text{EMA}(\mu_B),\text{EMA}(\sigma_B^2)$: Moving averages of the batch mean & variance.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FakeQuantLinear(nn.Linear):
    def __init__(self, d_in, d_out, b=4):
        super().__init__(d_in, d_out)
        self.qmax = 2 ** (b - 1) - 1

    def forward(self, x):
        w = self.weight                                       ## FP32 master weights
        s = w.detach().abs().amax(1, keepdim=True) / self.qmax   ## per-output-channel symmetric scale
        w_q = (w / s).round().clamp(-self.qmax, self.qmax) * s
        ## STE: forward sees w_q, backward sees identity (round's true grad = 0)
        w_ste = w + (w_q - w).detach()
        return F.linear(x, w_ste, self.bias)                  ## weights only; activations omitted

## Example
layer = FakeQuantLinear(4, 3, b=2)
layer(torch.randn(2, 4)).sum().backward()
print(layer.weight.grad.abs().sum() > 0)                     ## tensor(True): grads reach the master weights
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- Best accuracy at low bits ← the loss sees the rounding.
- Learns ranges/weights that sit well on the grid.

*Cons?*
- Full (or partial) training run → ⬆️compute, needs data & pipeline.
- Slower steps: extra quantize/dequantize ops.
- STE grads are biased → weights near rounding boundaries can oscillate.

*Why keep FP32 master weights?*
- 1 SGD step ≪ $s$ → an integer-only weight would never move.
- FP32 accumulates small updates until they cross a rounding boundary.

*Why fold BatchNorm before fake-quantizing?*
- Inference fuses BN into the weights → the quantized tensor is $W_\text{fold}$, ❌$W$.
- Fake-quantizing $W$ instead → train/inference mismatch.

*PTQ vs QAT?*
- PTQ: minutes, no training, good at 8-bit & LLM 4-bit weights.
- QAT: training cost, needed for very low bits or small compact models.
```

&nbsp;

#### STE
- **Name**: Straight-Through Estimator {cite:p}`bengio2013estimating`
- **What**: Backprop through a non-differentiable op as if it were the identity.
- **Why**: round / sign / threshold have 0 grad almost everywhere → no learning signal upstream.
- **How**:
    1. Forward: apply the true op.
    2. Backward: copy the upstream grad unchanged. Clipped variant: 0 grad outside the clipping range.
    - PyTorch idiom: `x + (op(x) - x).detach()`.

```{note} Math
:class: dropdown
Definition (for $\hat{x}=\text{round}(x)$):

$$
\frac{\partial\mathcal{L}}{\partial x}:=\frac{\partial\mathcal{L}}{\partial\hat{x}}\qquad\left(\text{true: }\frac{\partial\hat{x}}{\partial x}=0\ \text{a.e.}\right)
$$

- Biased estimator of the grad.
- For 1 layer of binary threshold units: correct sign. ❌Guaranteed through deeper stacks.
```

```{attention} Q&A
:class: dropdown
*Why does a "wrong" gradient work?*
- $\text{round}(x)=x+e$, $|e|\leq\frac{1}{2}$ → on scales ≫ 1 step, round ≈ identity.
- STE = grad of that coarse-scale trend; ignores the staircase.

*Where else is it used?*
- [VQ-VAE](gen/vae.md#vq-vae) codebook lookup, binary/ternary nets, discrete latents in model-based RL.

*Origin?*
- Proposed in Hinton's 2012 lecture; named & analyzed by Bengio et al. 2013.
```

&nbsp;

## Pruning
- **What**: Set a subset of weights to exactly 0.
- **Why**: Trained nets are over-parameterized → many weights contribute little → remove them.
- **How**: 3 independent choices.
    - **Criterion**: which weights → magnitude, 2nd-order (OBS), activation-aware.
    - **Granularity**: what unit → single weights (unstructured), N:M blocks, whole channels/heads/layers (structured).
    - **Schedule**: one-shot vs iterative; w/ or w/o retraining.

```{dropdown} Table: Pruning Granularity
| | Unstructured | N:M (2:4) | Structured |
|:--|:--|:--|:--|
| Unit | Single weight | ≤N nonzeros per M consecutive weights | Neuron / channel / head / layer |
| Accuracy at equal sparsity | Best | Middle | Worst |
| Storage | Values + indices (CSR) | Values + 2-bit index per value | Smaller dense tensor |
| Speedup on GPUs | Rare (irregular access) | 2× math on Sparse Tensor Cores | Yes, any hardware |
| Sparsity level | Free | Fixed 50% (2:4) | Free |
```

```{attention} Q&A
:class: dropdown
*Why does pruning not just break the network?*
- Many weights have tiny effect on the output.
- Retraining (or OBS-style compensation) lets survivors absorb the removed function.

*Why does sparsity often fail to give speedups?*
- Dense GEMM hardware cannot skip scattered zeros.
- Sparse formats add index metadata (CSR up to 200% overhead at 8-bit values) & irregular memory access.
- → Unstructured sparsity mainly saves storage. Speed needs structure (N:M, structured).

*Pruning vs quantization?*
- Pruning: fewer nonzeros. Quantization: fewer bits per nonzero. Orthogonal → combine (e.g., 50% sparse + 4-bit).
- GPT-scale models: magnitude pruning collapses early → 2nd-order pruners needed → [SparseGPT](#sparsegpt).

*How does it relate to the Lottery Ticket Hypothesis?*
- Iterative magnitude pruning + rewinding the survivors to their initial (or early-training) values → trainable sparse subnetworks → [Lottery Ticket Hypothesis](theory.md#lottery-ticket-hypothesis).
```

&nbsp;

### Magnitude Pruning
- **What**: Remove the smallest-$|w|$ weights, then retrain the survivors. {cite:p}`han2015learning`
- **Why**: $|w|$ = free importance proxy: ❌data, ❌Hessian.
- **How**:
    1. Train dense → learn which connections matter.
    2. Prune weights below a per-layer threshold (Han: quality parameter × std of the layer's weights).
    3. Retrain w/ the mask fixed (pruned weights stay 0).
    4. Repeat 2–3 → higher sparsity.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\tau$: Pruning threshold.

Process:

$$
M_{ij}=\mathbb{1}\left[|W_{ij}|>\tau\right],\qquad \hat{W}=M\odot W
$$

Retraining step:

$$
W\leftarrow W-\eta\left(M\odot\nabla_W\mathcal{L}\right)
$$

- $\eta$: Learning rate.
- Masked grads → pruned weights stay 0.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class MagnitudePruner:
    def __init__(self, layer: nn.Linear, sparsity=0.5):
        w = layer.weight.data
        k = int(sparsity * w.numel())
        ## keep exactly numel - k largest |w| (topk, not a threshold -> robust to ties & k = 0)
        keep = w.abs().flatten().topk(w.numel() - k).indices
        self.mask = torch.zeros(w.numel()).scatter_(0, keep, 1.0).view_as(w)
        layer.weight.data *= self.mask
        ## retraining: zero the grads of pruned weights -> they stay 0
        layer.weight.register_hook(lambda g: g * self.mask)

## Example
layer = nn.Linear(6, 4)
p = MagnitudePruner(layer, sparsity=0.5)
print((layer.weight == 0).float().mean())                  ## tensor(0.5000)
layer(torch.randn(2, 6)).sum().backward()
print((layer.weight.grad[p.mask == 0] == 0).all())         ## tensor(True)
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- Trivial to implement. Scales to any model.
- AlexNet 61M → 6.7M params (9×), VGG-16 138M → 10.3M (13×), no accuracy loss.

*Cons?*
- Ignores curvature & input correlations → a small weight on a sensitive direction can matter.
- Retraining needed. One-shot magnitude pruning on LLMs collapses fast (OPT-175B: fine at 10%, collapse beyond 30%).
- Unstructured → storage win, rarely a speed win.

*Why iterative instead of one-shot?*
- Large single cut → survivors cannot recover. Small cuts + retraining → gradual adaptation.
- AlexNet: 5× → 9× compression w/o accuracy loss.

*Why retrain at all?*
- Removing weights shifts every downstream activation → retraining re-fits the survivors.

*When does $|w|$ fail as importance?*
- Importance = effect on the loss ≈ $\frac{1}{2}H_{jj}w_j^2$ (diag Hessian) → magnitude assumes equal curvature for all weights.
- LLM layers have outlier input channels → equal curvature fails → [OBS](#obs) / activation-aware criteria.
```

&nbsp;

### OBS
- **Name**: Optimal Brain Surgeon {cite:p}`hassibi1992second`
- **What**: Remove the weight whose deletion least increases a 2nd-order loss; update all others to compensate.
- **Why**: Magnitude ignores curvature & correlations.
    - Small weight on a high-curvature direction → large loss increase.
    - Correlated weights can absorb a removed weight's role → compensation.
- **How**:
    1. 2nd-order Taylor expansion of the loss at a trained minimum (grad ≈ 0).
    2. Constrain weight $q$ to 0 → solve the constrained quadratic in closed form.
    3. Prune the weight w/ the smallest saliency $\varepsilon_q$. Apply the update $\boldsymbol{\delta}$ to the rest.
    4. Remove $q$ from $H^{-1}$ (rank-1 update) → repeat.
    - **OBD** = diagonal-$H$ special case → saliency $\frac{1}{2}H_{qq}w_q^2$, ❌compensation.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\mathbf{w}$: Weight vector (1 row of $W$ for the layer-wise loss).
- Misc:
    - $\boldsymbol{\delta}$: Weight change.
    - $\mathbf{e}_q$: $q$-th unit vector.

Overrides: here $H$ = Hessian of the loss w.r.t. $\mathbf{w}$. Layer-wise loss → $H=2XX^T$ & the Taylor expansion is exact. $q$ = idx of the removed weight, ❌integer code.

Objective:

$$
\min_{\boldsymbol{\delta}}\ \frac{1}{2}\boldsymbol{\delta}^TH\boldsymbol{\delta}\quad\text{s.t.}\quad\mathbf{e}_q^T\boldsymbol{\delta}+w_q=0
$$

Solution:

$$
\boldsymbol{\delta}=-\frac{w_q}{[H^{-1}]_{qq}}H^{-1}\mathbf{e}_q,\qquad \varepsilon_q=\frac{w_q^2}{2[H^{-1}]_{qq}}
$$

- $\varepsilon_q$: Loss increase (saliency) of removing $w_q$.
- Target $t\neq0$ (e.g., a grid point): replace $w_q$ by $w_q-t$ → quantization version ([GPTQ](#gptq)).

Inverse update after removing $q$:

$$
H_{-q}^{-1}=\left(H^{-1}-\frac{H^{-1}_{:,q}H^{-1}_{q,:}}{[H^{-1}]_{qq}}\right)_{-q}
$$

- $(\cdot)_{-q}$: Drop row & column $q$.
```

```{tip} Derivation
:class: dropdown
*Where do $\boldsymbol{\delta}$ & $\varepsilon_q$ come from?*
1. Lagrangian: $\mathcal{J}=\frac{1}{2}\boldsymbol{\delta}^TH\boldsymbol{\delta}+\mu\left(\mathbf{e}_q^T\boldsymbol{\delta}+w_q\right)$.
2. $\nabla_{\boldsymbol{\delta}}\mathcal{J}=0$ → $H\boldsymbol{\delta}+\mu\mathbf{e}_q=0$ → $\boldsymbol{\delta}=-\mu H^{-1}\mathbf{e}_q$.
3. Constraint: $-\mu[H^{-1}]_{qq}+w_q=0$ → $\mu=\frac{w_q}{[H^{-1}]_{qq}}$.
4. → $\boldsymbol{\delta}=-\frac{w_q}{[H^{-1}]_{qq}}H^{-1}\mathbf{e}_q$.
5. $\varepsilon_q=\frac{1}{2}\boldsymbol{\delta}^TH\boldsymbol{\delta}=\frac{1}{2}\mu^2\,\mathbf{e}_q^TH^{-1}\mathbf{e}_q=\frac{w_q^2}{2[H^{-1}]_{qq}}$.
```

````{important} Code
:class: dropdown
```python
import torch

class OBS:
    def __init__(self, X, damp=0.01):
        H = 2 * X @ X.T                                       ## layer-wise Hessian (d_in, d_in)
        H += damp * H.diag().mean() * torch.eye(H.shape[0])
        self.Hinv = torch.linalg.inv(H)

    def prune(self, w, k):
        ## greedily zero k weights of 1 row; re-fit the survivors after each removal
        w, Hinv = w.clone(), self.Hinv.clone()
        alive = torch.ones_like(w, dtype=torch.bool)
        for _ in range(k):
            eps = w ** 2 / (2 * Hinv.diag())                  ## saliency of each weight
            eps[~alive] = float("inf")
            q = int(eps.argmin())
            w -= w[q] / Hinv[q, q] * Hinv[:, q]               ## OBS update; sets w[q] = 0 exactly
            Hinv -= torch.outer(Hinv[:, q], Hinv[q, :]) / Hinv[q, q]   ## remove q from H^{-1}
            alive[q] = False
        return w

## Example
torch.manual_seed(0)
X = torch.randn(8, 3) @ torch.randn(3, 128) + 0.1 * torch.randn(8, 128)   ## correlated inputs
w = torch.randn(8)
w_obs = OBS(X).prune(w, k=4)
idx = w.abs().topk(4).indices
w_mag = torch.zeros_like(w)
w_mag[idx] = w[idx]                                           ## magnitude: keep top-4, no compensation
print(f"mag {((w_mag - w) @ X).norm():.2f} vs OBS {((w_obs - w) @ X).norm():.2f}")   ## OBS << mag
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- Accounts for curvature & correlations. Compensates instead of only deleting.
- Exact for quadratic (layer-wise) losses.

*Cons?*
- Full-network $H$: $\mathcal{O}(P^2)$ memory, $\mathcal{O}(P^3)$ inverse → infeasible for modern nets ($P$ = #params).
- Taylor expansion only local → large removals leave its valid region.

*How do modern methods make it tractable?*
- Layer-wise loss → $H=2XX^T$ is only $d_\text{in}\times d_\text{in}$ & shared by all rows.
- Fixed column order + Cholesky → [GPTQ](#gptq), [SparseGPT](#sparsegpt).

*OBS vs OBD?*
- OBD: diagonal $H$ → saliency $\frac{1}{2}H_{qq}w_q^2$, no update to other weights.
- OBS: full $H^{-1}$ → correct saliency under correlations + compensating update.

*What if the weights are not at a minimum?*
- Linear term $\mathbf{g}^T\boldsymbol{\delta}$ ≠ 0 → formula misses it.
- Layer-wise reconstruction around the original $W$: grad = 0 at $\hat{W}=W$ → assumption holds exactly.
```

&nbsp;

#### SparseGPT
- **What**: One-shot OBS-based pruning for LLMs, using GPTQ's column-wise loop. {cite:p}`frantar2023sparsegpt`
- **Why**: Magnitude pruning collapses on GPT-scale models. Exact OBS & retraining → ❌at 100B+.
- **How**:
    1. Same $H$, Cholesky & column loop as [GPTQ](#gptq). Target per weight = 0 (pruned) or itself (kept).
    2. **Adaptive mask**: every $B_s=128$ columns, choose the mask for the next block by OBS error $\frac{w^2}{[H_F^{-1}]_{jj}}$ on the **curr** updated weights.
    3. Only pruned weights create error → their error is pushed onto later columns of the same row.
    4. N:M: set $B_s=M$ → per row, prune the lowest-error weights in each block of $M$.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $p$: Target sparsity.
    - $B_s$: Mask-selection block size.

Process (column $j$, every row in parallel):
1. If $j\bmod B_s=0$: in columns $j{:}j+B_s$, prune fraction $p$ w/ the smallest $\frac{W_{ik}^2}{U_{kk}^2}$ (all rows of the block pooled).
2. Error: $E_{:,j}=(1-M_{:,j})\odot\frac{W_{:,j}}{U_{jj}}$.
3. Update: $W_{:,k}\leftarrow W_{:,k}-E_{:,j}U_{jk}$ for $k>j$.
4. End: $\hat{W}=M\odot W$.

- $U_{kk}^2=[H_F^{-1}]_{kk}$ (see GPTQ Derivation) → score = OBS saliency up to a factor.
- Paper's $H=XX^T$ → same updates & mask ranking; only the absolute saliency changes by 2×.
```

````{important} Code
:class: dropdown
```python
import torch

class SparseGPT:
    def __init__(self, W, X, damp=0.01):
        ## same H^{-1} Cholesky setup as GPTQ
        self.W = W.clone()
        H = 2 * X @ X.T
        H += damp * H.diag().mean() * torch.eye(H.shape[0])
        self.U = torch.linalg.cholesky(torch.linalg.inv(H), upper=True)

    def __call__(self, sparsity=0.5, bs=4):
        W, U = self.W.clone(), self.U
        M = torch.ones_like(W, dtype=torch.bool)                 ## True = keep
        for j in range(W.shape[1]):
            if j % bs == 0:
                ## adaptive mask on the CURRENT (already compensated) weights
                score = W[:, j:j + bs] ** 2 / U.diag()[j:j + bs] ** 2
                drop = score.flatten().topk(int(sparsity * score.numel()), largest=False).indices
                m = torch.ones_like(score, dtype=torch.bool)
                m.view(-1)[drop] = False
                M[:, j:j + bs] = m
            err = (~M[:, j]) * W[:, j] / U[j, j]                 ## kept weights create no error
            W[:, j + 1:] -= err[:, None] * U[j, j + 1:][None, :]
        return W * M

## Example
torch.manual_seed(0)
W = torch.randn(8, 16)
X = torch.randn(16, 4) @ torch.randn(4, 256) + 0.1 * torch.randn(16, 256)
W_sp = SparseGPT(W, X)(sparsity=0.5)
W_mag = W * (W.abs() > W.abs().flatten().kthvalue(64).values)  ## 50% magnitude, no compensation
print((W_sp == 0).float().mean())                             ## tensor(0.5000)
print(f"mag {((W_mag - W) @ X).norm():.2f} vs SparseGPT {((W_sp - W) @ X).norm():.2f}")
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- OPT-175B & BLOOM-176B: 50–60% unstructured sparsity in one shot, <4.5 hours, negligible perplexity increase.
- ❌Retraining. Supports 2:4 & 4:8. Combines w/ quantization in the same loop.

*Cons?*
- 2:4 costs more: OPT-175B perplexity +0.39 (2:4) vs +0.11 (4:8).
- Smaller models lose more at the same sparsity.
- Unstructured 50% → storage win, no dense-GEMM speedup.

*Why select the mask adaptively instead of upfront?*
- Compensation updates change later weights significantly → an upfront mask becomes stale.
- Block-wise mask → sparsity can differ across columns ← LLMs have a few sensitive outlier input features.

*Why do larger models prune more easily?*
- Empirically: OPT-2.7B loses ~1 perplexity at 50%, OPT-66B ~0.
- Consistent w/ more redundancy at scale.

*50% sparse + 4-bit vs 3-bit?*
- Similar memory. OPT-175B perplexity: 8.29 (50% + 4-bit) vs 8.68 (GPTQ 3-bit).
```

&nbsp;

### Structured Pruning
- **What**: Remove whole structures (neurons/channels, attention heads, layers) → smaller dense model.
- **Why**: Speed on any hardware.
    - Unstructured zeros → dense kernels cannot skip them.
    - Removing a structure → smaller dense matrices → fewer FLOPs, less memory, no special kernels.
- **How**:
    1. Score each structure: weight norm, activation statistics, or loss sensitivity.
    2. Remove the lowest-scoring ones. Shrink coupled shapes: hidden unit $i$ of layer $\ell$ ⇔ row $i$ of $W_\ell$ & column $i$ of $W_{\ell+1}$.
    3. Fine-tune or distill to recover.
    - **Heads**: many attention heads can be removed at test time; some layers keep 1 head. {cite:p}`michel2019are`
    - **Layers**: drop whole blocks; feasible because the residual stream carries the signal past them.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $W_1\in\mathbb{R}^{d_h\times d_\text{in}}$: 1st layer weight.
    - $W_2\in\mathbb{R}^{d_\text{out}\times d_h}$: 2nd layer weight.
- Misc:
    - $d_h$: Hidden width.
    - $\phi$: Activation function.
    - $\mathcal{K}$: Kept hidden units, $|\mathcal{K}|=k$.

Model:

$$
\mathbf{y}=W_2\,\phi(W_1\mathbf{x})
$$

Pruned model:

$$
\hat{\mathbf{y}}=W_{2,:,\mathcal{K}}\,\phi(W_{1,\mathcal{K},:}\mathbf{x})
$$

- Removing unit $i$ changes the output by exactly $-W_{2,:,i}\,\phi\left(W_{1,i,:}\mathbf{x}\right)$.
- FLOPs & params ∝ $k$: $d_h(d_\text{in}+d_\text{out})$ → $k(d_\text{in}+d_\text{out})$.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class NeuronPruner:
    def __init__(self, fc1: nn.Linear, fc2: nn.Linear):
        self.fc1, self.fc2 = fc1, fc2

    @torch.no_grad()
    def __call__(self, keep):
        ## score unit i by in-norm * out-norm: both sides decide its effect on y
        score = self.fc1.weight.norm(dim=1) * self.fc2.weight.norm(dim=0)
        idx = score.topk(keep).indices.sort().values
        fc1 = nn.Linear(self.fc1.in_features, keep)
        fc2 = nn.Linear(keep, self.fc2.out_features)
        fc1.weight.copy_(self.fc1.weight[idx])               ## drop rows of W1
        fc1.bias.copy_(self.fc1.bias[idx])
        fc2.weight.copy_(self.fc2.weight[:, idx])            ## drop the matching columns of W2
        fc2.bias.copy_(self.fc2.bias)
        return fc1, fc2

## Example
fc1, fc2 = NeuronPruner(nn.Linear(4, 8), nn.Linear(8, 3))(keep=4)
x = torch.randn(2, 4)
print(fc1.weight.shape, fc2.weight.shape, fc2(torch.relu(fc1(x))).shape)
## torch.Size([4, 4]) torch.Size([3, 4]) torch.Size([2, 3]) -> genuinely smaller dense layers
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- Real speedup & memory saving on any hardware & framework.
- Output = a standard smaller architecture.

*Cons?*
- Coarse units → much larger accuracy loss than unstructured at equal sparsity.
- Usually needs substantial fine-tuning or distillation.
- Coupled shapes (residual dims, GQA groups) constrain what can be removed.

*Why does removing heads often barely hurt?*
- Heads are redundant after training; many contribute little at test time.
- Michel et al. find evidence that multi-head gains come partly from training dynamics, ❌only test-time capacity.

*Structured pruning vs training a small model from scratch?*
- Pruning inherits the large model's learned features → often better per FLOP.
- Pruned + distilled from the original model → common recipe for small LLMs → [Distillation](../nlp/post/kd.md).
```

&nbsp;

### N:M Sparsity
- **What**: At most $N$ nonzeros in every $M$ consecutive weights along the input dim. {cite:p}`mishra2021accelerating`
- **Why**: Unstructured → ❌speedup. Structured → ⬇️accuracy.
    - N:M = fine-grained enough for accuracy, regular enough for hardware.
    - NVIDIA Ampere Sparse Tensor Cores: 2:4 → 2× math throughput.
- **How**:
    1. Split each row of $W$ (GEMM reduction dim) into blocks of $M$.
    2. Keep the $N$ largest $|w|$ per block (or lowest OBS error → [SparseGPT](#sparsegpt)).
    3. Retrain w/ the mask fixed. NVIDIA workflow: repeat the original training run (same optimizer, LR schedule, #epochs).
    4. Store $N$ values + a 2-bit position index per value per block (2:4).
    - Optional: permute input channels before pruning → spread large weights across blocks. Permute the previous layer's output rows to match → same function.

```{note} Math
:class: dropdown
Override: $N$ = #nonzeros per block, $M$ = block width (❌mask).

Constraint:

$$
\left\|\hat{W}_{i,\,kM:(k+1)M}\right\|_0\leq N\qquad\forall i,k
$$

- $k$: Block idx along $d_\text{in}$.

Storage per weight (2:4, $b$-bit values, 2-bit index):

$$
\frac{N}{M}(b+2)=\frac{b+2}{2}\ \text{bits}
$$

- FP16: 9 bits vs 16 → 36 vs 64 bits per block.
- INT8: 5 bits vs 8 → ~38% less memory & bandwidth.

Convention: $N$ = #nonzeros (NVIDIA). SparseGPT's "$n$:$m$" counts zeros. 2:4 is the same either way.
```

````{important} Code
:class: dropdown
```python
import torch

class NMSparsity:
    def __init__(self, n=2, m=4):
        self.n, self.m = n, m

    def mask(self, W):
        d_out, d_in = W.shape
        blocks = W.abs().reshape(d_out, d_in // self.m, self.m)   ## groups along the INPUT dim
        keep = blocks.topk(self.n, dim=-1).indices                ## N largest |w| per block
        M = torch.zeros_like(blocks).scatter_(-1, keep, 1.0)
        return M.reshape(d_out, d_in)

## Example
W = torch.randn(2, 8)
M = NMSparsity(2, 4).mask(W)
print(M.reshape(2, 2, 4).sum(-1))          ## tensor([[2., 2.], [2., 2.]]): exactly 2 of every 4
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- Real 2× math speedup for the sparse GEMM (Ampere & later), w/ a tiny metadata overhead.
- Dense-trained model → prune → retrain → accuracy ≈ dense across a wide range of vision & language tasks.

*Cons?*
- Density fixed at 50% (2:4): extra zeros give no extra storage or throughput saving.
- Speedup only on the sparse matmuls → end-to-end gain < 2× (attention, memory traffic, other ops).
- One-shot 2:4 on LLMs costs more accuracy than 50% unstructured.

*Why along the input dim?*
- The hardware skips multiply-adds inside the dot product → zeros must lie on the reduction (K) dim.
- Weight $W\in\mathbb{R}^{d_\text{out}\times d_\text{in}}$ → reduction dim = $d_\text{in}$ → groups along each row.

*Why 2:4 specifically?*
- 50% → accuracy recoverable w/o hyperparam search.
- 4-value groups → 2-bit index per value → cheap decode & fixed-size compressed rows (no indirection).
```

&nbsp;

## Low-Rank Factorization
- **What**: Replace $W$ by a product $BA$ of rank $r$ → 2 thinner layers.
- **Why**: #params & FLOPs $d_\text{out}d_\text{in}$ → $r(d_\text{out}+d_\text{in})$.
    - Saves iff $r<\frac{d_\text{out}d_\text{in}}{d_\text{out}+d_\text{in}}$.
- **How**:
    1. [SVD](../ml/unsupervised.md#svd): $W=U\Sigma V^T$.
    2. Keep the top-$r$ singular triplets → $W_r$ = best rank-$r$ approximation (**Eckart–Young**).
    3. Split $\Sigma_r$ evenly: $B=U_r\Sigma_r^{1/2}$, $A=\Sigma_r^{1/2}V_r^T$.
    4. Replace the layer: $\mathbf{x}\to A\mathbf{x}\to BA\mathbf{x}$. Fine-tune to recover.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $A\in\mathbb{R}^{r\times d_\text{in}}$: Down-projection.
    - $B\in\mathbb{R}^{d_\text{out}\times r}$: Up-projection.
- Hyperparams:
    - $r$: Rank.
- Misc:
    - $\sigma_i$: $i$-th largest singular value of $W$.
    - $U_r,\Sigma_r,V_r$: Top-$r$ singular vectors & values.
    - $[\cdot]_r$: Truncated SVD to rank $r$.

Model:

$$
\hat{W}=BA=U_r\Sigma_rV_r^T=W_r
$$

Eckart–Young (optimal over all rank-$\leq r$ matrices):

$$
\|W-W_r\|_F^2=\sum_{i>r}\sigma_i^2,\qquad \|W-W_r\|_2=\sigma_{r+1}
$$

Activation-aware variant (output error, $XX^T=LL^T$, $L$ invertible):

$$
\hat{W}=\arg\min_{\text{rank}(\hat{W})\leq r}\left\|(W-\hat{W})X\right\|_F=[WL]_r\,L^{-1}
$$
```

```{tip} Derivation
:class: dropdown
*Why is $[WL]_rL^{-1}$ optimal for the output error?*
1. $\|(W-\hat{W})X\|_F^2=\text{tr}\left((W-\hat{W})XX^T(W-\hat{W})^T\right)=\|(W-\hat{W})L\|_F^2$.
2. $= \|WL-\hat{W}L\|_F^2$.
3. $L$ invertible → $\text{rank}(\hat{W}L)=\text{rank}(\hat{W})$ → $\hat{W}L$ ranges over all rank-$\leq r$ matrices.
4. Eckart–Young on $WL$ → $\hat{W}L=[WL]_r$ → $\hat{W}=[WL]_rL^{-1}$.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class LowRankLinear(nn.Module):
    def __init__(self, linear: nn.Linear, r):
        super().__init__()
        U, S, Vh = torch.linalg.svd(linear.weight.data, full_matrices=False)
        sqrt_s = S[:r].sqrt()                                  ## split Sigma_r evenly between factors
        self.A = nn.Parameter(sqrt_s[:, None] * Vh[:r])        ## (r, d_in)
        self.B = nn.Parameter(U[:, :r] * sqrt_s[None, :])      ## (d_out, r)
        self.bias = linear.bias

    def forward(self, x):
        return (x @ self.A.T) @ self.B.T + self.bias           ## 2 thin matmuls, never forms BA

## Example
lin = nn.Linear(8, 6)
low = LowRankLinear(lin, r=2)
S = torch.linalg.svdvals(lin.weight.data)
err = (low.B @ low.A - lin.weight).norm()
print(torch.allclose(err, S[2:].pow(2).sum().sqrt(), atol=1e-5))   ## True: Eckart-Young error
print(low.A.numel() + low.B.numel(), lin.weight.numel())           ## 28 vs 48
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- Dense, hardware-friendly: 2 ordinary matmuls, no special kernels.
- Closed-form optimal (Frobenius & spectral) per matrix.

*Cons?*
- Trained weight spectra often decay slowly → small $r$ loses a lot → fine-tuning needed.
- Optimal for weight error, ❌output error → use the activation-aware variant.
- Breakeven rank is high: square $d\times d$ → saves only if $r<d/2$.

*Low-rank factorization vs [LoRA](../nlp/post/sft.md#lora)?*
- LoRA: full-rank frozen $W_0$ + low-rank **update** $\Delta W$ → cheap training, ❌inference saving.
- Factorization: $W$ itself replaced by low rank → inference saving, needs $W$ ≈ low rank.
- LoRA's assumption (the fine-tuning update is low-rank) is much weaker than "the weight is low-rank".

*Why split $\Sigma_r$ as $\Sigma_r^{1/2}$ on both sides?*
- Any split gives the same product. Balanced factors → similar scales → better-conditioned fine-tuning.

*Train low-rank from scratch instead?*
- Architectural low-rank (bottlenecks, [MLA](../nlp/infer/cache.md#mla)'s latent KV) avoids the post-hoc approximation error.
```

&nbsp;
