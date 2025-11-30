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
# Training Toolkit
This page covers training tricks commonly used in all kinds of DL tasks.

This page does NOT cover specialized training methods like transfer learning or post-training. They are discussed separately.

## Weight Initialization
### Zero Initialization
- **What**: All params are 0.
- **Why**: People thought it was simple and unbiased back in the days.

```{attention} Q&A
:class: dropdown
*Pros?*
- Simple, unbiased, good performance with small models.

*Cons?*
- **Failure to break symmetry**: All weights produce the same output and receive the same gradients → No learning.
- Vanishing gradients.
- No non-linearity.
- Dead neurons if activated by the ReLU family.
```

&nbsp;

### Random Initialization
- **What**: All params are random.
- **Why**: To **break symmetry** → Neurons learn diff features.

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
- **Why**: To mitigate [vanishing/exploding gradients](../dl/issues.md) in very deep NNs with ReLU activations.

```{note} Math
:class: dropdown
- Distributions:
	\begin{align*}
	&\text{Normal:}  &&W\sim N(0,\sigma^2),\quad\sigma=\sqrt{\frac{2}{\text{fan\_in}}} \\
	&\text{Uniform:} &&W\sim U(-a,a),\quad a=\sqrt{\frac{6}{\text{fan\_in}}}
	\end{align*}
	- $\text{Var}[U(-a,a)]=\frac{a^2}{3}\rightarrow\frac{a^2}{3}=\frac{2}{\text{fan\_in}}$
- Modes:
	- `fan_in`: Preserves forward activation scale (most common) → Mitigates vanishing/exploding activations.
	- `fan_out`: Preserves backprop gradient scale → Mitigates gradient issues not caused by activation issues.
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
- **Why**: To mitigate [vanishing/exploding gradients](../dl/issues.md) in very deep NNs with ReLU activations.

```{note} Math
:class: dropdown
- Distributions:
	\begin{align*}
	&\text{Normal:}  &&W\sim N(0,\sigma^2),\quad\sigma=\sqrt{\frac{2}{n+m}} \\
	&\text{Uniform:} &&W\sim U(-a,a),\quad a=\sqrt{\frac{6}{n+m}}
	\end{align*}
	- $n=\text{fan\_in},\quad m=\text{fan\_out}$.
- Modes:
	- `fan_in`: Preserves forward activation scale (most common) → Mitigates vanishing/exploding activations.
	- `fan_out`: Preserves backprop gradient scale → Mitigates gradient issues not caused by activation issues.
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
- May stop too early → Hyperparam tuning, validation data/metric noise, etc.
```

&nbsp;

## Gradient Tricks
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
- Longer training ← Extra forward/backward passes.
- May mess up other operations that affect grads (e.g., Normalization, Dropout, LR scheduling, etc.).
```

&nbsp;

### Gradient Checkpointing
- **What**: Store fewer intermediate activations during forward pass → Recompute them during backprop.
- **Why**: To fit larger models/batches into limited GPU memory → Trade compute for memory.
- **How**:
    1. Divide the computational graph into segments.
    2. Forward: Store activations only at segment boundaries (checkpoints).
    3. Backward: Recompute intermediate activations on-demand.
    4. Apply optimizer step.

```{attention} Q&A
:class: dropdown
*Pros?*
- HUGE memory savings (often 30-50% less GPU memory).
- Enables training deeper NNs on larger batches on the same hardware.

*Cons?*
- ⬆️⬆️ Computational cost (~20–30% more FLOPs).
- ⬆️⬆️ Training time.
- ⬆️ Implementation complexity ← Requires careful graph segmentation

*FYI:*
- Gradient checkpointing is **orthogonal** to gradient accumulation → Combine both for MASSIVE memory savings
```

&nbsp;

## Distributed Training
```{dropdown} Table: Parallelism Comparison
| Category                       | DDP                                     | FSDP                                                                | ZeRO                                                     | Pipeline Parallelism                               | Tensor Parallelism                                     |
| :----------------------------- | :-------------------------------------- | :------------------------------------------------------------------ | :------------------------------------------------------- | :------------------------------------------------- | :----------------------------------------------------- |
| **Memory**                     | ⬆️                                      | ⬇️                                                                  | ⬇️⬇️ (Stage-dependent: S1→S3)                            | ↔️ (splits layers, but states still exist)         | ↔️/⬇️ (layer-scope sharding only)                      |
| **Time**                       | ⬆️ on small/medium models               | ⬆️ on large models<br>when memory savings permit larger batch sizes | ⬆️ when memory savings avoid OOM; ⬇️ if comms bottleneck | ↔️/⬆️ if well-balanced; ⬇️ with pipeline bubbles   | ⬆️ with fast interconnect; ⬇️ if comms-heavy per layer |
| **Memory Targets**             | Params+Grads+Opt **replicated**         | Params **sharded**, grads sometimes sharded                         | Opt (S1)<br>Grads (S2)<br>Params (S3) **sharded**     | Layers **partitioned** across devices              | Tensors inside layers **sharded**                      |
| **Scalability**                | #GPUs                                   | #Params per GPU                                                     | #Params total (via sharding all states)                  | #Depth (number of layers/stages)                   | #Width (hidden size / shards per layer)                |
| **Debugging**                  | Single-GPU semantics                    | Extra metadata and indirection layers                               | Runtime-managed state partitioning                       | Cross-stage scheduling and correctness             | Per-op sharding + collectives                          |
| **Communication**              | All-reduce grads (ring)                 | All-gather/Reduce-Scatter per shard                                 | Heavy All-Gather/Reduce-Scatter of params/grads/states   | Activation transfers between stages                | Per-layer collectives (matmul shards)                  |
| **Failure Modes**              | OOM at larger models                    | Comms hotspots<br>Sharding bugs                                       | Network bottlenecks<br>Partition mismatches                | Pipeline bubbles<br>Stage imbalance                  | Synchronization stalls<br>Kernel mismatch                |
| **When to Use**            | Model fits per GPU | Large models that barely fit with sharding                         | Ultra-large models                  | Ultra-deep NNs (e.g., transformers with many blocks) | Ultra-wide NNs (large hidden sizes/heads)            |
| **When NOT to Use**            | Model doesn’t fit per GPU               | Very small models                          | Weak interconnect / Unstable comms                      | Shallow models; hard-to-balance stages             | Small hidden sizes                    |
| **IRL Tools**     | PyTorch DDP                             | PyTorch FSDP                                                        | DeepSpeed ZeRO (Stages 1–3)                              | Megatron/DeepSpeed PP                              | Megatron-LM TP                                         |
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

&nbsp;

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

### ZeRO
- **What**: Zero Redundancy Optimizer.
	- Partition optimizer states, grads, and params across multiple devices instead of replication → Memory optimization
- **Why**: Replication results in redundancy in data storage → Partitioning directly eliminates redundancy in memory storage.
- **How**:
    1. Partitioning:
        - **Stage 1**: Partition optimizer states (e.g., Adam's momentum & variance).
        - **Stage 2**: Partition grads.
        - **Stage 3**: Partition params.
    2. Forward:
        - Each GPU holds ONLY the necessary params for computation.
        - When required, params are fetched on-demand from other GPUs.
    3. Backward:
        - Grads are computed locally, then partitioned and communicated to the GPUs responsible for those params.
    4. Optimizer Step:
        - Each GPU updates ONLY its own opt states & params.

```{attention} Q&A
:class: dropdown
*Cons?*
- Communication overhead:
    - Frequent all-gather/reduce-scatter calls → Network bottlenecks.

*FYI:*
- ZeRO = the backbone of **DeepSpeed**.
- ZeRO does NOT change the math of optimization — only how states are stored and communicated.
- ZeRO is combinable with pipeline/tensor parallelism.
```

&nbsp;

### Pipeline Parallelism
- **What**: Split a model by **layers** into sequential stages across devices.
- **Why**: To train deep NNs that don't fit on 1 GPU.
- **How**:
    1. Partition the network into $K$ stages, each placed on a different device.
    2. Microbatch the global batch into $m$ microbatches: $B \rightarrow {b_1,\dots,b_m}$.
    3. Forward:
        1. Stage 0 runs $b_1$.
        2. Stage 0 runs $b_2$. Stage 1 runs $b_1$.
        3. Stage 0 runs $b_3$. Stage 1 runs $b_2$. Stage 2 runs $b_1$.
        4. ...
    4. Backward:
        - GPipe: Do ALL forwards for ALL batches first, then ALL backwards for ALL batches.
            - Cleaner BUT slower.
        - 1F1B (common): Once the pipeline is filled, each stage alternates:
            - 1 Forward on a newer microbatch.
            - 1 Backward on an older microbatch whose grad is ready.
    5. Apply optimizer step.

```{attention} Q&A
:class: dropdown
*Pros?*
- ⬆️ Device utilization ← Overlapping.
- Combinable with data parallelism.

*Cons?*
- Pipeline bubbles: At start/end, some stages sit idle (waste).
- Communication overhead ← Activations/Grads must be sent between stages every microbatch.
- ❌ Load balancing ← Slowest stage bottlenecks the whole pipeline.
```

&nbsp;

### Tensor Parallelism
- **What**: Split the **tensors inside a single layer** across multiple GPUs.
- **Why**: To train wide NNs that don't fit on 1 GPU.
- **How**: e.g., Transformer blocks,
    1. Choose a sharding axis: hidden/output dimensions are most common.
    2. Shard weights:
        - MLP:
            - Column-parallel: Shard output features → Each GPU produces a slice of activations.
            - Row-parallel: Shard input features → Each GPU consumes a slice of input. Partial results need reduction.
        - Attention:
            - Shard heads: Each GPU owns $\frac{H}{\text{\#GPU}}$ heads.
    3. Forward:
        - **All-Gather** when assembling full activations for the next op.
        - **All-Reduce / Reduce-Scatter** when partial sums must become a correct full result.
    4. Backward:
        - Grads for sharded weights are computed locally & combined via the same collectives.
    5. Optimizer Step:
        - Each GPU updates ONLY its own opt states & params.

```{attention} Q&A
:class: dropdown
*Pros?*
- ⬆️ Device utilization than PP for Transformers.
- **The 3D Parallelism**: DP × PP × TP.

*Cons?*
- Communication overhead ← TP introduces collectives inside almost every layer.
```

&nbsp;

## Mixed Precision
### Precision
- **What**: Numerical format + Bit-width.
    - **Bit-width**: #Bits to encode each numerical value in memory.
- **Why**:
    1. It affects params, activations, and grads
    2. → It affects memory cost, time cost, model quality, training stabiity
- **How (Bit-width)**: 3 components:
    - **Sign**: 0 = +; 1 = -
    - **Exponent**:
        - Theory: **Actual exponent** $n$ in scientific notation $a \times 2^n$.
        - Practice:
            1. Use a given **#exponent bits** $k$ to calculate **bias** $2^{k-1}-1$.
            2. **Stored exponent** ← **Actual exponent** + **Bias**.
    - **Mantissa/Fraction**:
        - Theory: $a$ in scientific notation $a \times 2^n$.
        - Practice:
            1. The first non-zero digit is always immediately to the left of the decimal point.
            2. The only possible non-zero binary digit is 1.
            3. → Every base number starts with 1.
            4. → We don't need to store it. We ONLY need to store the **fraction**.

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

&nbsp;

### Mixed Precision Training
- **What**: Train using a mix of **low & high** precisions for different parts of computation.
- **Why**: To reduce memory usage & speed up training WHILE maintaining model accuracy.
    - FP16 ops are faster & use less memory.
	- FP32 is kept for critical steps (e.g., weight updates) to maintain numerical stability.
- **How**:
    1. Forward:
        1. Compute most operations in FP16.
        2. Multiply loss by a scale factor to avoid underflow in FP16 gradients.
    2. Backward:
		1. Compute grads in FP16.
		2. Unscale grads.
       	3. Convert grads to FP32 for weight updates.
       	4. Apply optimizer step in FP32.

```{attention} Q&A
:class: dropdown
*Pros?*
- ⬆️ Training speed.
- ⬇️ Memory usage → Larger batch sizes become possible.
- ❌ Accuracy loss, if implemented correctly.

*Cons?*
- Very complex implementation & debugging ← Mixed data types
```

&nbsp;

## Performance Metrics
### FLOP
- **What**: Floating Point Operation.
	- = A single arithmetic operation on floating-point numbers (e.g., addition, multiplication).
- **Why**: To measure **computational complexity** of a model.
- **How**:
    1. Count all floating-point operations in forward + backward pass.
    2.  Sum across all layers.
    3.  Report in GFLOPs (billions) or TFLOPs (trillions).

```{dropdown} Table: Example FLOPs
| Module   | FLOPs |
| :------- |:--------------|
| **Linear** | $\text{FLOPs} \approx 2 \times (\text{input size} \times \text{output size})$ |
| **Convolution** | $\text{FLOPs} \approx 2 \times (\text{kernel size}^2 \times \text{input channels} \times \text{output channels} \times \text{output feature map size})$ |
| **Self-Attention** | $\text{FLOPs} \approx 4 \times (\text{seq len} \times \text{hidden size}^2) + 2 \times (\text{seq len}^2 \times \text{hidden size})$ |
```

```{attention} Q&A
:class: dropdown
*Pros?*
- Hardware-agnostic metric for cost computation.
- ✅ Model comparison.
- ✅ Scaling law.
- ✅ Training/Inference time estimation with hardware specs.

*Cons?*
- FLOPs ≠ actual runtime ← Memory bandwidth, parallelism, and hardware optimizations matter.
- ❌ Precision.
- ❌ Sparsity.
- ❌ Data movement cost.

*FYI:*
- FLOPs for training ≈ 2-3× FLOPs for inference (due to backward pass).
- FLOPs ≠ FLOPS:
    - FLOPs = operations count.
    - FLOPS = operations per second (hardware speed).
```

&nbsp;

### Memory Footprint
- **What**: Total #memory (RAM/VRAM) consumed during training/inference.
    - Includes: model params, optimizer states, activations, grads, and buffers.
- **Why**: Memory footprint determines:
    - Whether your model **fits on the hardware**.
    - Max batch size.
    - Whether you need memory save techniques.
    - Whether you can avoid the STINKY OOM error.

```{attention} Q&A
:class: dropdown
*Cons?*
- Hard to estimate precisely for dynamic graphs or custom ops.
- May spike due to **non-obvious buffers** (e.g., fused kernels).
```

&nbsp;

### Throughput
- **What**: #Samples (or Tokens) processed **per unit time** (e.g., per second).
- **Why**: To measure **training** efficiency & hardware utilization → Training time
    - Higher/Lower throughput = Faster/Slower iteration cycles.
- **How**: Measure total samples processed over a fixed time window, then divide.

```{attention} Q&A
:class: dropdown
*Cons?*
- Chasing throughput can harm model quality:
    - Large batch sizes → Poor generalization.
    - Aggressive mixed precision without stability checks.

*FYI:*
- Throughput is NOT constant:
    - Early epochs may have lower throughput due to warm-up.
    - Distributed training overhead (e.g., all-reduce) can reduce effective throughput.
```

&nbsp;

### Latency
- **What**: **Time** taken to complete a single operation.
- **Why**: To monitor **real-time inference**.
    - Lower latency → Faster response for end-users.
    - In training, latency per step affects overall throughput.

```{attention} Q&A
:class: dropdown
*Cons?*
- Reducing latency often requires:
    - Model compression → Accuracy trade-offs.
    - Specialized hardware → Higher cost.

*FYI:*
- Latency ≠ Throughput.
	- Latency is "per request".
	- Throughput is "aggregate over time".
```