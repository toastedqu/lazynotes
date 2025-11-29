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
# Training
This page covers some common training techniques in DL.

## Weight Initialization
### Zero Initialization
- **What**: All params are 0.
- **Why**: People thought it was simple and unbiased back in the days.

```{attention} Q&A
:class: dropdown
*Pros?*
- Simple, unbiased, good performance with small models.

*Cons?*
- **Failure to break symmetry**: All weights produce the same output and receive the same gradients $\rightarrow$ No learning.
- Vanishing gradients.
- No non-linearity.
- Dead neurons if activated by the ReLU family.
```

&nbsp;

### Random Initialization
- **What**: All params are random.
- **Why**: To **break symmetry** $\rightarrow$ Neurons learn diff features.

```{attention} Q&A
:class: dropdown
*Pros?*
- Simple, cheap & breaks symmetry.

*Cons?*
- Require careful selection of scale/distribution.
- Slower convergence.
- Dead neurons with ReLU are still likely.
- Run-to-run variability.
```

&nbsp;

### He Initialization
- **What**: All params in ReLU nets are constrained to a fixed variance.
- **Why**: To mitigate [vanishing/exploding gradients](../dl/issues.md#vanishingexploding-gradient) in very deep NNs with ReLU activations.

```{note} Math
:class: dropdown
- Distributions:
	\begin{align*}
	&\text{Normal:}  &&W\sim N(0,\sigma^2),\quad\sigma=\sqrt{\frac{2}{\text{fan\_in}}} \\
	&\text{Uniform:} &&W\sim U(-a,a),\quad a=\sqrt{\frac{6}{\text{fan\_in}}}
	\end{align*}
	- $\text{Var}[U(-a,a)]=\frac{a^2}{3}\rightarrow\frac{a^2}{3}=\frac{2}{\text{fan\_in}}$
- Modes:
	- `fan_in`: Preserves forward activation scale (most common) $\rightarrow$ Mitigates vanishing/exploding activations.
	- `fan_out`: Preserves backprop gradient scale $\rightarrow$ Mitigates gradient issues not caused by activation issues.
```

```{attention} Q&A
:class: dropdown
*Pros?*
- Training stability & convergence ⬆️.

*Cons?*
- NOT universally optimal.
- The "right" variant depends on activation, direction, and layer type.
```

&nbsp;

### Xavier/Glorot Initialization
- **What**: "He Init" but preserve variance in **both ways**.
- **Why**: To mitigate [vanishing/exploding gradients](../dl/issues.md#vanishingexploding-gradient) in very deep NNs with ReLU activations.

```{note} Math
:class: dropdown
- Distributions:
	\begin{align*}
	&\text{Normal:}  &&W\sim N(0,\sigma^2),\quad\sigma=\sqrt{\frac{2}{n+m}} \\
	&\text{Uniform:} &&W\sim U(-a,a),\quad a=\sqrt{\frac{6}{n+m}}
	\end{align*}
	- $n=\text{fan\_in},\quad m=\text{fan\_out}$.
- Modes:
	- `fan_in`: Preserves forward activation scale (most common) $\rightarrow$ Mitigates vanishing/exploding activations.
	- `fan_out`: Preserves backprop gradient scale $\rightarrow$ Mitigates gradient issues not caused by activation issues.
```

```{attention} Q&A
:class: dropdown
*Pros?*
- ✅ for most non-ReLU activations.

*Cons?*
- ❌ for ReLU activations.
- Too many assumptions (independence, no normalization, etc.).
```

&nbsp;

## Gradient Control
### Gradient Clipping
- **What**: Cap grads by value/norm.
- **Why**: To mitigate exploding grads.

```{attention} Q&A
:class: dropdown
*Cons?*
- May slow learning & worsen performance if threshold too small.
```

&nbsp;

### Gradient Accumulation
- **What**: Sum grads across several mini-batches, then perform one optimizer update.
- **Why**: To get stability benefits of large-batch training **under memory limits**.
	- Grads are computed per smaller mini-batch.
	- BUT updates are made on the large batch.
- **How**:
	1. Split a large batch into $k$ mini-batches which can fit in memory.
	2. Compute grads for each mini-batch.
	3. Sum grads.
	4. Scale loss by $k$ so the final accumulated grad matches the large-batch average.
	5. Apply one optimizer step AND **zero grads** to clear grad cache.

```{attention} Q&A
:class: dropdown
*Cons?*
- Fewer param updates per epoch.
- Longer training $\leftarrow$ Extra forward/backward passes.
- May mess up other operations that affect grads (e.g., Normalization, Dropout, LR scheduling, etc.).
```

&nbsp;

## Regularization
### Early Stopping
- **What**: Stop when val perf stops improving.
- **Why**: To prevent overfitting.
- **How**:
	1. Choose:
		- Monitored val metric.
		- **Patience**: How many epochs to wait for improvement.
		- **Min Delta**: Minimum threshold to qualify as "improvement".
	2. Train:
		- Every time improvement exceeds Min Delta, reset Patience counter.
		- If Patience runs out, quit & keep best checkpoint.

```{attention} Q&A
:class: dropdown
*Cons?*
- May stop too early $\rightarrow$ Hyperparam tuning, validation data/metric noise, etc.
```

&nbsp;

## Distributed Training
```{dropdown} Table: DDP vs FSDP
| Category | DDP | FSDP |
|:---------|-----|------|
| Memory | ⬆️ | ⬇️ |
| Time | ⬆️ on small/medium models | ⬆️ on large models<br>when memory savings permit larger batch sizes |
| Scalability | #GPUs | #Params per GPU |
| Debugging | Single-GPU semantics | Extra metadata and indirection layers |
```

### DDP
- **What**: Distributed Data Parallel.
	- Copy a model across multi processes (GPUs), each taking a unique mini-batch of data.
- **Why**: Faster than FSDP, used when the model can easily fit in each GPU's memory.
- **How**:
	1. Each process holds a copy of the model.
	2. A distributed sampler ensures each process receives a non-overlapping portion of the input data.
	3. Forward: Each copy processes its mini-batch to compute the loss.
	4. Backward:
		1. Each copy computes its own grads.
		2. The grads are sychronized & averaged across all processes via a ring all-reduce algorithm.
			- Processes are arranged in a logical ring.
			- Grads on each process are split into chunks.
			- Chunks are passed to their immediate neighbors & combined, till every process has the final, identical results.
			- Then averaged.
		3. Each copy uses the averaged grad to update its local model's weights.
			- Since all copy weights are the same initially, they remain the same after each identical update.

### FSDP
- **What**: Fully Sharded Data Parallel.
	- Shard model params, grads, and optimizer states across multi processes (GPUs), each taking a unique mini-batch of data.
- **Why**: More flexible than DDP, used when the model cannot fit in each GPU's memory.
- **How**:
	1. Each process holds an even shard of each param tensor and the metadata needed to reassemble full tensors.
	2. Forward:
		1. Before each layer's forward pass, every process completes an all-gather (i.e., each process temporarily holds a full copy of that layer's params).
		2. Each copy processes its mini-batch.
		3. As soon as the process finishes, discard the full params and ONLY keep the original shards. GPU memory drops back to the sharded footprint.
		4. Move onto the next layer.
		5. Repeat Step 2.1 - 2.4 till loss is computed.
	3. Backward: Repeat Forward but for grads.

&nbsp;

## Mixed Precision Training
### Precision
- **What**: Numerical format + Bit-width.
    - **Bit-width**: #Bits to encode each numerical value in memory.
- **Why**:
    1. It affects params, activations, and grads
    2. $\rightarrow$ It affects memory cost, time cost, model quality, training stabiity
- **How (Bit-width)**: 3 components:
    - **Sign**: 0 = +; 1 = -
    - **Exponent**:
        - Theory: **Actual exponent** $n$ in scientific notation $a \times 2^n$.
        - Practice:
            1. Use a given **#exponent bits** $k$ to calculate **bias** $2^{k-1}-1$.
            2. **Stored exponent** $\leftarrow$ **Actual exponent** + **Bias**.
    - **Mantissa/Fraction**:
        - Theory: $a$ in scientific notation $a \times 2^n$.
        - Practice: 
            1. The first non-zero digit is always immediately to the left of the decimal point.
            2. The only possible non-zero binary digit is 1.
            3. $\rightarrow$ Every base number starts with 1.
            4. $\rightarrow$ We don't need to store it. We ONLY need to store the **fraction**.

```{note} Math
:class: dropdown
Representation Formula:

$$
\text{value} = (-1)^{\text{sign}}1.\text{fraction}\times 2^{\text{actual\_exponent}}
$$

&nbsp;

Value range:
1. Exponent range:

$$\begin{align*}
E_{\text{min}}&=1-\text{bias}=-(2^{k-1}-2) \\
E_{\text{max}}&=(2^{k}-2)-\text{bias}=2^{k-1}-1
\end{align*}$$

2. Binary dynamic range:

$$\begin{align*}
\text{Min positive}&=+1\times2^{E_{\text{min}}} \\
\text{Max positive}&=(2-2^{-k_{\text{mantissa}}})\times2^{E_{\text{max}}}
\end{align*}$$
&nbsp;

Example: 2.5 in FP16
1. Write in normalized binary scientific notation

$$
2.5_{10}\rightarrow 10.1_2\rightarrow 1.01_2\times 2^1
$$

2. Obtain Sign bit:

$$
1.01_2 > 0 \rightarrow \text{Sign bit: }0
$$

3. Obtain Exponent bits:

$$\begin{align*}
&\text{\#Exponent Bits:} && k=5             \\
&\text{Bias:}            && 2^{k-1}-1=15    \\
&\text{Actual Exponent:} && 1               \\
&\text{Exponent Bit:}    && 1+15=16_{10}=10000_2
\end{align*}$$

4. Obtain Mantissa bits:

$$
\text{Fraction: }.01 \rightarrow \text{Mantissa bits (10 bits): }0100000000
$$

5. Concatenate all bits (sign + exponent + mantissa):

$$
0\ \ 10000\ \ 0100000000
$$
```

```{dropdown} Table 1: Precision Comparison
| Format   | Bits | Exponent bits | Mantissa bits | Dynamic Range                | Decimal Precision | Memory per value |
| :------- | :--: | :-----------: | :-----------: | :--------------------------: | :---------------: | :--------------: |
| **FP32** | 32   | 8             | 23            | $\approx[1e^{-38}, 1e^{38}]$ | \~7 digits        | 4 bytes          |
| **FP16** | 16   | 5             | 10            | $\approx[6e^{-5}, 6e^{4}]$   | \~3–4 digits      | 2 bytes          |
| **BF16** | 16   | 8             | 7             | $\approx[1e^{-38}, 1e^{38}]$ | \~3 digits        | 2 bytes          |
| **INT8** | 8    | –             | –             | $=[–128,127]$                | \~2–3 digits      | 1 byte           |
```

```{dropdown} Table 2: Precision Usage
| Format   | Usage |
| :------- |:--------------|
| **FP32** | Gold-standard baseline for training & inference. |
| **FP16** | Mixed-precision training & GPU inference. |
| **BF16** | Mixed-precision training & CPU/GPU inference. |
| **INT8** | Quantized inference with extreme memory limits. |
```

<!-- ## Model Compression
### Pruning
### Quantization
### Knowledge Distillation -->