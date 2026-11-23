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
# Systems
Hardware-aware training: memory, communication, parallelism, numerics & cost metrics.

Notations:
- $p$: #Devices (GPUs) in 1 communication group
- $p_\text{x}$: Degree of parallelism x, x $\in$ {dp, tp, pp, cp, ep} (e.g., $p_\text{tp}$ = TP group size)
- $V$: Full (unsharded) buffer size of 1 collective (bytes)
- $\alpha$: Latency per message (s)
- $\beta$: Transfer time per byte (s/byte) = 1 / link bandwidth
- $N$: #Model params
- $L$: #Layers
- $d$: Hidden size
- $a$: #Attention heads
- $s$: Sequence length (tokens)
- $b$: Micro-batch size (sequences)
- $F$: Peak FLOP/s per device

Overrides of global notation: in collective & parallelism sections, $i,j,k$ = device / chunk idx, ❌sample / feature / class idx.

&nbsp;

## Memory Saving

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


## Collective Communication
- **What**: Data-exchange primitive in which every device of a group takes part.

```{dropdown} Table: Collectives
| Collective | Input on device $i$ | Output on device $i$ | Ring cost | Typical use |
|:--|:--|:--|:--|:--|
| **All-Reduce** | $x_i$ (size $V$) | $\sum_k x_k$ (size $V$) | $2(p-1)\alpha+2\frac{p-1}{p}V\beta$ | DDP grad sync, TP partial sums |
| **Reduce-Scatter** | $x_i$ (size $V$) | Shard $i$ of $\sum_k x_k$ (size $V/p$) | $(p-1)\alpha+\frac{p-1}{p}V\beta$ | ZeRO-2/3 & FSDP grads, SP |
| **All-Gather** | Shard $x_i$ (size $V/p$) | $[x_1;\dots;x_p]$ (size $V$) | $(p-1)\alpha+\frac{p-1}{p}V\beta$ | FSDP/ZeRO-3 params, SP |
| **All-to-All** | $p$ chunks $x_{i\to j}$ (size $V$) | $p$ chunks $x_{j\to i}$ (size $V$) | $(p-1)\alpha+\frac{p-1}{p}V\beta$ (full-bisection network) | EP token dispatch, DeepSpeed-Ulysses |

- Cost model ($\alpha$-$\beta$): time = #sequential steps × $\alpha$ + bytes sent per device × $\beta$.
```

&nbsp;

### All-Reduce
- **What**: Every device ends with the elementwise reduction (usually sum) of all devices' buffers.
- **Why**: Data-parallel replicas compute different grads → every replica needs the same averaged grad to stay identical.
- **How**:
    - **Naive**: All → 1 root → reduce → broadcast → root link carries $(p-1)V$ each way → bottleneck.
    - **Ring**: Reduce-Scatter + All-Gather → bandwidth-optimal → [Ring All-Reduce](#ring-all-reduce).
    - **Tree** / recursive doubling: $\mathcal{O}(\log p)$ steps → latency-optimal → small messages.

```{note} Math
:class: dropdown
Definition:

$$
y_i=\sum_{k=1}^{p}x_k\quad\forall i\in\{1,\dots,p\}
$$

- $x_k$: Buffer on device $k$ (size $V$).
- $y_i$: Result on device $i$.

Properties:
- Lower bound on bytes sent per device: $2\frac{p-1}{p}V$. Ring reaches it.
- Grad averaging: $\bar g=\frac{1}{p}y_i$ ← global loss = mean over equal-size per-device batches.
```

```{attention} Q&A
:class: dropdown
*Why average, ❌sum?*
- Loss = mean over the global batch → grad = mean of per-device grads (equal per-device batch sizes).
- Sum → effective LR ×$p$.

*Does all-reduce time grow with $p$?*
- Bandwidth term $2\frac{p-1}{p}V\beta\to2V\beta$ → ~constant in $p$ → DP scales.
- Latency term $2(p-1)\alpha$ grows linearly → many tiny messages hurt → bucket grads (PyTorch DDP default bucket: 25 MB).

*How does DDP hide the all-reduce?*
- Grads become ready layer by layer during backward (last layer first).
- Launch an all-reduce per full bucket while backward continues → comm overlaps compute.

*Is DDP training bit-identical to 1-GPU training?*
- ❌. FP addition is not associative → reduction order differs → tiny numeric differences.
- All replicas still agree with each other ← each chunk is reduced once, then copied.

*Multi-node?*
- Hierarchical: Reduce-Scatter in node (NVLink) → All-Reduce across nodes (IB) → All-Gather in node.
- → Slow inter-node link carries only $1/(\text{GPUs per node})$ of the data per GPU.
```

&nbsp;

#### Ring All-Reduce
- **What**: All-Reduce as 2 passes around a logical ring of devices: Reduce-Scatter, then All-Gather.
- **Why**: Bandwidth. Per-device traffic $2\frac{p-1}{p}V\approx2V$, independent of $p$ → ❌hot root link.
- **How**:
    1. Arrange devices in a ring. Split each buffer into $p$ chunks.
    2. **Reduce-Scatter** ($p-1$ steps): each device sends 1 chunk to its right neighbor. The neighbor adds it to its own copy.
    3. → Device $i$ now owns 1 fully reduced chunk.
    4. **All-Gather** ($p-1$ steps): pass the finished chunks around the ring. Each receiver overwrites its copy.
    5. Every link is busy at every step → full bandwidth use.

```{note} Math
:class: dropdown
Process (step $t\in\{0,\dots,p-2\}$, chunk idx mod $p$):
1. Reduce-Scatter: device $i$ sends chunk $i-t$ to device $i+1$, which adds it.
2. After $p-1$ steps: device $i$ holds the full sum of chunk $i+1$.
3. All-Gather: device $i$ sends chunk $i+1-t$ to device $i+1$, which overwrites it.

Cost:

$$
T_\text{ring}=2(p-1)\alpha+2\frac{p-1}{p}V\beta
$$

- $2(p-1)$ steps. Each step: every device sends 1 chunk of $V/p$ bytes.
```

```{note} Example
:class: dropdown
$p=4$ GPUs, $V=1$ GB of grads:
- Ring: each GPU sends $2\cdot\frac{3}{4}\cdot1=1.5$ GB.
- Naive root: root receives 3 GB & sends 3 GB → 2× slower even at the same link speed, worse as $p$⬆️.
```

````{important} Code
:class: dropdown
```python
import torch

class RingAllReduce:
    def __init__(self):
        self.sent = 0  ## elements sent by device 0 (every device sends the same amount)

    def __call__(self, xs):
        p = len(xs)
        c = [list(x.clone().chunk(p)) for x in xs]  ## c[i][j] = chunk j on device i
        ## Phase 1: reduce-scatter. Step t: device i sends chunk (i-t)%p to i+1, which ADDS it
        for t in range(p - 1):
            msgs = [(i, (i - t) % p, c[i][(i - t) % p].clone()) for i in range(p)]  ## snapshot: all sends at once
            for i, j, buf in msgs:
                c[(i + 1) % p][j] = c[(i + 1) % p][j] + buf
            self.sent += msgs[0][2].numel()
        ## now device i holds the full sum of chunk (i+1)%p
        ## Phase 2: all-gather. Step t: device i forwards chunk (i+1-t)%p to i+1, which OVERWRITES it
        for t in range(p - 1):
            msgs = [(i, (i + 1 - t) % p, c[i][(i + 1 - t) % p].clone()) for i in range(p)]
            for i, j, buf in msgs:
                c[(i + 1) % p][j] = buf
            self.sent += msgs[0][2].numel()
        return [torch.cat(ci) for ci in c]

## Example: 4 devices, 8 elements each
xs = [torch.arange(8.) * (r + 1) for r in range(4)]
ar = RingAllReduce()
out = ar(xs)
print(all(torch.equal(o, sum(xs)) for o in out))  ## True
print(ar.sent)  ## 12 = 2(p-1)/p * V = 2*3/4*8
```
````

```{attention} Q&A
:class: dropdown
*Ring vs tree?*
- Ring: bandwidth-optimal, $2(p-1)$ steps → best for large messages.
- Tree: $\mathcal{O}(\log p)$ steps → best for small messages or many nodes.
- NCCL selects the algorithm per message size & topology.

*What limits ring speed?*
- All steps are synchronous → ring speed = slowest link → 1 slow link or straggler slows every device.
- → Build rings along fast links (NVLink in node).

*Why is ring All-Reduce = Reduce-Scatter + All-Gather important?*
- The 2 halves can be split around other work.
- ZeRO-2: Reduce-Scatter grads → update own shard → All-Gather params. Same bytes as DDP, $1/p$ the optimizer memory → [Reduce-Scatter](#reduce-scatter).
```

&nbsp;

### All-Gather
- **What**: Every device ends with the concatenation of all devices' shards.
- **Why**: Sharded state (FSDP/ZeRO-3 params, SP activations) must exist in full right before an op uses it.
- **How**: Ring, $p-1$ steps. Each step, forward the shard received in the previous step → every device sees every shard once.

```{note} Math
:class: dropdown
Definition:

$$
y_i=[x_1;x_2;\dots;x_p]\quad\forall i
$$

- $x_k$: Shard on device $k$ (size $V/p$).
- $y_i$: Gathered buffer on device $i$ (size $V$).

Cost:

$$
T_\text{AG}=(p-1)\alpha+\frac{p-1}{p}V\beta
$$
```

```{attention} Q&A
:class: dropdown
*Memory cost?*
- Output = $p$× input → transient memory peak.
- FSDP: gather 1 layer's params → compute → free at once. Prefetch the next layer's gather → overlap comm & compute.

*What is the backward of All-Gather?*
- Reduce-Scatter. And the backward of Reduce-Scatter = All-Gather.
- ← Shard $x_i$ appears in every device's $y$ → its grad = sum of all devices' grads for slice $i$ → reduce, then keep slice $i$ on device $i$.
```

&nbsp;

### Reduce-Scatter
- **What**: Sum all devices' buffers. Each device keeps only its $1/p$ shard of the sum.
- **Why**: A device that owns only 1 shard (ZeRO-2/3, FSDP) needs only that shard's reduced grad → skip the gather half of All-Reduce → ½ the traffic, $1/p$ the result memory.
- **How**: Ring, $p-1$ steps. Each step, pass a partial-sum chunk to the neighbor, which adds its own contribution.
    - → All-Reduce = Reduce-Scatter + All-Gather.

```{note} Math
:class: dropdown
Definition:

$$
y_i=\Big(\sum_{k=1}^{p}x_k\Big)_{[i]}
$$

- $(\cdot)_{[i]}$: $i$-th of $p$ equal shards.

Cost:

$$
T_\text{RS}=(p-1)\alpha+\frac{p-1}{p}V\beta
$$
```

```{attention} Q&A
:class: dropdown
*Why does ZeRO-2 cost the same comm as DDP?*
- DDP: All-Reduce grads = RS + AG.
- ZeRO-2: RS grads → each device updates its own shard → AG updated params.
- Same 2 collectives, same bytes. Only the optimizer step moves between them.
- ZeRO-3: + AG params in forward & again in backward → 1.5× DDP volume.

*Where else is RS + AG used in place of All-Reduce?*
- [Sequence Parallelism](#sequence-parallelism): TP's All-Reduce → RS + AG → activations between them stay sharded.
```

&nbsp;

### All-to-All
- **What**: Each device sends a distinct chunk to every other device: chunk $j$ of device $i$ → slot $i$ of device $j$.
- **Why**: Re-shard data along a new axis, e.g., tokens grouped by source device → grouped by destination expert.
- **How**: Distributed transpose of a $p\times p$ chunk matrix. Run as $p-1$ pairwise exchanges.

```{note} Math
:class: dropdown
Definition:

$$
y_j[i]=x_i[j]\quad\forall i,j
$$

- $x_i[j]$: Chunk $j$ on device $i$ (size $V/p$).

Cost (full-bisection network, e.g., NVSwitch in 1 node):

$$
T_\text{A2A}=(p-1)\alpha+\frac{p-1}{p}V\beta
$$

- Ring or oversubscribed network → limited by bisection bandwidth → much slower at large $p$.
```

````{important} Code
:class: dropdown
```python
import torch

class AllToAll:
    def __call__(self, xs):
        p = len(xs)
        ## chunk j on device i is addressed to device j
        chunks = [x.chunk(p) for x in xs]
        ## device j collects chunk j from every source i, in source order -> distributed transpose
        return [torch.cat([chunks[i][j] for i in range(p)]) for j in range(p)]

## Example: 3 devices; value = 10*source + destination
xs = [torch.tensor([10. * i + j for j in range(3)]) for i in range(3)]
print(AllToAll()(xs)[1])  ## tensor([ 1., 11., 21.]) -> device 1 got its chunk from devices 0, 1, 2
```
````

```{attention} Q&A
:class: dropdown
*Why is All-to-All harder to scale than All-Reduce?*
- Every pair of devices exchanges data → needs bisection bandwidth, ❌neighbor-only ring.
- Cross-node links (IB) ≪ NVLink → All-to-All across nodes is often comm-bound.

*Where is it used?*
- [Expert Parallelism](#expert-parallelism): dispatch tokens to experts, combine outputs back.
- DeepSpeed-Ulysses: switch sequence-sharding ↔ head-sharding around attention.

*What makes MoE All-to-All irregular?*
- Chunk sizes depend on the router → variable-size All-to-All.
- Hot experts → 1 device receives more → it sets step time.
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

#### ZeRO-Offload
- **What**: ZeRO-2 + move FP32 optimizer states, FP32 master params, FP16 grads & the optimizer step to CPU. {cite:p}`ren2021zerooffload`
- **Why**: GPU memory ≪ CPU memory. Model states (16 bytes/param w/ mixed-precision Adam) dominate GPU memory → offload them → ~10× larger model per GPU.
- **How**:
    - **ZeRO-Offload**:
        1. GPU keeps FP16 params → forward & backward stay on GPU.
        2. Backward: send each FP16 grad to CPU as soon as it is ready → overlaps w/ the rest of backward.
        3. CPU: Adam step on FP32 states (vectorized, multithreaded CPU-Adam).
        4. Copy updated params CPU → GPU in FP16.
        5. Multi-GPU: ZeRO-2 partition first → each GPU offloads only its own shard → CPU work splits across processes.
        6. Optional **DPU** (one-step Delayed Parameter Update): CPU step for batch $t$ overlaps GPU forward/backward of batch $t+1$ → update is 1 step stale.
    - **ZeRO-Infinity**: ZeRO-3 + offload params, grads & optimizer states to CPU **or NVMe** + activation checkpoints to CPU → fine-tune a 1T-param model on 1 DGX-2 node (16 V100). {cite:p}`rajbhandari2021zeroinfinity`

```{note} Math
:class: dropdown
Model states per param (mixed precision + Adam, bytes):

$$
\underbrace{2}_{\text{FP16 param}}+\underbrace{2}_{\text{FP16 grad}}+\underbrace{4}_{\text{FP32 param}}+\underbrace{4}_{\text{momentum}}+\underbrace{4}_{\text{variance}}=16
$$

Placement & traffic per step:

$$\begin{align*}
\text{GPU memory}&=2N \\
\text{CPU memory}&=14N \\
\text{GPU}\leftrightarrow\text{CPU traffic}&=\underbrace{2N}_{\text{FP16 grads}\downarrow}+\underbrace{2N}_{\text{FP16 params}\uparrow}=4N
\end{align*}$$

- GPU model-state memory $16N\to2N$: 8× less.
```

```{tip} Derivation
:class: dropdown
*Why is this placement the unique optimum?*
1. Forward & backward cost $\mathcal{O}(N\cdot\text{batch})$. CPU is orders of magnitude slower → keep them on GPU. Only $\mathcal{O}(N)$ ops (update, norms) may move to CPU.
2. View 1 training step as a data-flow graph. Every node lies on a cycle: params → fwd/bwd → grads → update → params.
3. Any GPU/CPU split cuts ≥ 2 edges. Each edge carries ≥ $2N$ bytes → traffic ≥ $4N$.
4. FP32 states (param, momentum, variance) must sit w/ the update op. Else a $4N$ edge is cut → traffic ≥ $6N$.
5. FP16 params must sit w/ fwd/bwd ← both read them. Else traffic ≥ $6N$.
6. Left to place: FP16 grads & the update block. Both on CPU → max GPU savings (8×) at the minimum $4N$ traffic.
```

```{attention} Q&A
:class: dropdown
*Pros?*
- ~10× larger model on 1 GPU: 13B params on 1 V100. 40 TFLOP/s per V100 for a 10B model.
- ❌Model code change.
- Combines w/ ZeRO-2 data parallelism & tensor parallelism.

*Cons?*
- PCIe bandwidth ≪ HBM → needs a large enough batch per step to hide the $4N$ transfer.
- CPU Adam can bottleneck → DPU → 1-step stale update.
- Needs large CPU RAM: $14N$ bytes (13B params → ~182 GB).

*Why not offload activations too?*
- ZeRO-Offload targets model states only. Activations → [Gradient Checkpointing](#gradient-checkpointing).
- ZeRO-Infinity also offloads activation checkpoints to CPU.

*ZeRO-Offload vs ZeRO-3?*
- ZeRO-3: borrow memory from other GPUs → pay network bandwidth.
- ZeRO-Offload: borrow memory from CPU → pay PCIe bandwidth & CPU compute.
- ZeRO-Infinity = both + NVMe.

*When to use it?*
- Few GPUs (fine-tuning on 1 node or 1 GPU).
- ❌ when enough GPUs exist to shard w/ ZeRO-3/FSDP → faster.
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

### Sequence Parallelism
- **What**: Shard LayerNorm & dropout activations along the sequence dim across the TP group. {cite:p}`korthikanti2022reducing`
- **Why**: TP shards activations inside attention & MLP only. LayerNorm/dropout regions stay replicated on all $p_\text{tp}$ GPUs → their memory does not shrink w/ $p_\text{tp}$ → dominates at large $p_\text{tp}$.
- **How**:
    1. LayerNorm & dropout act per token → run on $s/p_\text{tp}$ tokens per GPU, ❌comm.
    2. Enter a TP region (attention / MLP): **All-Gather** along the sequence dim.
    3. Leave a TP region: **Reduce-Scatter** along the sequence dim (replaces TP's All-Reduce).
    4. All-Reduce = RS + AG → same comm volume as plain TP.

```{note} Math
:class: dropdown
Activation memory per transformer layer (bytes, 16-bit activations, 1-byte dropout masks):

$$\begin{align*}
\text{No parallelism}&:\ sbd\left(34+5\frac{as}{d}\right) \\
\text{TP}&:\ sbd\left(10+\frac{24}{p_\text{tp}}+5\frac{as}{d\,p_\text{tp}}\right) \\
\text{TP + SP}&:\ \frac{sbd}{p_\text{tp}}\left(34+5\frac{as}{d}\right)
\end{align*}$$

- $10\,sbd$: Replicated under TP: 2 LayerNorm inputs, 2 dropout masks, inputs to the 1st linear of attention & MLP.
- $24\,sbd$: Activations inside attention & MLP that TP already shards.
- $5\,as^2b$: Softmax output, softmax dropout mask, softmax dropout output.
```

```{attention} Q&A
:class: dropdown
*Pros?*
- All activations ÷ $p_\text{tp}$, ❌extra comm.
- LayerNorm/dropout compute ÷ $p_\text{tp}$ → small speedup (6% per-layer forward in the paper).
- W/ selective activation recomputation: 5× less activation memory, > 90% of full-recompute overhead removed.
- 530B GPT on 2240 A100s: 54.2% MFU vs 42.1% w/ full recomputation.

*Cons?*
- Tied to the TP group → no effect when $p_\text{tp}=1$.
- Total activations still ∝ $s$ → 1 sequence of 100K+ tokens still overflows → [Context Parallelism](#context-parallelism).

*Why can LayerNorm run on a sequence shard w/o comm?*
- LayerNorm normalizes over the hidden dim of 1 token → tokens are independent.
- ❌ for BatchNorm ← needs statistics across samples.

*What is selective activation recomputation?*
- Recompute only the $5as^2b$ attention-core part: large memory, few FLOPs.
- Keep the $34\,sbd$ part: costly matmuls to recompute.
- → Finer-grained [Gradient Checkpointing](#gradient-checkpointing).

*Naming?*
- "Sequence parallelism" also names other methods: Li et al. 2021 (ring self-attention) & DeepSpeed-Ulysses shard the WHOLE sequence → closer to [Context Parallelism](#context-parallelism).
- Megatron SP shards ONLY the non-TP regions.
```

&nbsp;

### Context Parallelism
- **What**: Shard ALL activations, incl. attention, along the sequence dim; circulate K/V blocks between devices. {cite:p}`liu2023ring`
- **Why**: Long context. Activations ∝ $s$ → at $s\sim10^5$–$10^6$, 1 sequence does not fit on 1 GPU, even w/ TP + SP + FlashAttention.
- **How** (Ring Attention):
    1. Split the sequence into $p_\text{cp}$ blocks. Device $i$ holds block $i$ of Q, K, V.
    2. Non-attention ops (linear, MLP, LayerNorm) act per token → run locally, ❌comm.
    3. Attention: $p_\text{cp}$ steps. Each step, attend local Q to the current K/V block. Then pass that K/V block to the next device in the ring.
    4. Merge partial results w/ the online softmax (as in [FlashAttention](../nlp/attn.md#flashattention)) → exact attention.
    5. Send K/V of the next step during compute of the current step → comm hidden if block compute ≥ block transfer.
    - Causal mask → contiguous blocks are unbalanced (late blocks attend to more keys) → reorder blocks to balance load (e.g., Megatron-Core CP).
- Caveat: naming varies. Ring Attention, Megatron "context parallelism", Li et al. 2021 "sequence parallelism" & DeepSpeed-Ulysses all shard the full sequence w/ different comm patterns.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $c=s/p_\text{cp}$: Block length (tokens per device).

Per device per ring step (forward, $b=1$, 16-bit):

$$\begin{align*}
\text{Compute}&:\ \underbrace{2c^2d}_{QK^T}+\underbrace{2c^2d}_{PV}=4c^2d\ \text{FLOPs} \\
\text{Transfer}&:\ \underbrace{2cd}_{K,V}\times2\ \text{bytes}=4cd\ \text{bytes}
\end{align*}$$

Overlap condition:

$$
\frac{4c^2d}{F}\ge4cd\,\beta\iff c\ge F\beta
$$

- $F\beta$ = device FLOP/s ÷ link bandwidth → min block length to hide comm.
```

````{important} Code
:class: dropdown
```python
import torch

class RingAttention:
    def __init__(self, p):
        self.p = p

    def __call__(self, q, k, v):
        ## q, k, v: (s, d_h); device i owns block i of each
        qs, ks, vs = q.chunk(self.p), k.chunk(self.p), v.chunk(self.p)
        outs = []
        for i in range(self.p):  ## devices run in parallel in reality
            m = torch.full((qs[i].shape[0], 1), float("-inf"))  ## running row max
            l = torch.zeros(qs[i].shape[0], 1)                   ## running softmax denominator
            o = torch.zeros_like(qs[i])                          ## running unnormalized output
            for t in range(self.p):
                j = (i - t) % self.p  ## K/V block that reaches device i at ring step t
                sc = qs[i] @ ks[j].T / q.shape[-1] ** 0.5
                m_new = torch.maximum(m, sc.max(-1, keepdim=True).values)
                ## rescale old stats to the new max -> exact softmax over all blocks seen so far
                r = torch.exp(m - m_new)
                e = torch.exp(sc - m_new)
                l = l * r + e.sum(-1, keepdim=True)
                o = o * r + e @ vs[j]
                m = m_new
            outs.append(o / l)
        return torch.cat(outs)

## Example: 8 tokens over 4 devices
q, k, v = torch.randn(3, 8, 4).unbind(0)
ref = torch.softmax(q @ k.T / 2, dim=-1) @ v
print(torch.allclose(RingAttention(4)(q, k, v), ref, atol=1e-5))  ## True
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- Exact attention, ❌approximation.
- Activation memory per device ∝ $s/p_\text{cp}$ → max context ∝ #devices.
- Comm overlaps compute if $c\ge F\beta$.

*Cons?*
- Extra comm in every attention layer.
- Block too short → comm exposed.
- Causal load imbalance needs block reordering.

*CP vs FlashAttention?*
- FlashAttention: removes the $s^2$ score matrix on 1 device. Activations still ∝ $s$.
- CP: splits that $\propto s$ term across devices. Each device runs FlashAttention kernels on its blocks.
- → Complementary. Same online-softmax math, but tiles live on different devices.

*CP (ring) vs DeepSpeed-Ulysses?*
- Ring: P2P K/V passing. Degree not limited by #heads.
- Ulysses: All-to-All → each device gets the full sequence for $a/p$ heads → local attention → All-to-All back. Degree ≤ #heads.

*CP vs SP?*
- [SP](#sequence-parallelism): shards only LayerNorm/dropout regions, inside the TP group.
- CP: shards the whole layer incl. attention → its own parallel dim.
```

&nbsp;

### Expert Parallelism
- **What**: Place different MoE experts on different devices; route tokens to them via All-to-All. {cite:p}`shazeer2017outrageously`
- **Why**: [MoE](../nlp/models.md#moe) total params ≫ active params → all experts do not fit on 1 device.
    - Each token needs only its top-$K$ experts → move tokens to experts, ❌experts to tokens.
- **How**:
    1. EP group of $p_\text{ep}$ devices. Each device holds $N_e/p_\text{ep}$ experts. Non-expert layers (attention) are replicated → data parallel.
    2. Router picks the top-$K$ experts per token.
    3. **Dispatch**: All-to-All sends each token to the devices of its experts.
    4. Each device runs its experts on the tokens it received.
    5. **Combine**: All-to-All returns the outputs to each token's home device → gate-weighted sum.
    6. Backward: the same 2 All-to-Alls in reverse.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $N_e$: #Experts per MoE layer ($N$ on the MoE page).
    - $K$: #Experts per token (top-$K$).
    - $n_\text{tok}$: #Tokens per device per micro-batch.

Bytes sent per device per All-to-All (balanced routing, 16-bit):

$$
\frac{p_\text{ep}-1}{p_\text{ep}}\cdot K\,n_\text{tok}\,d\cdot2
$$

- Each token sends $K$ copies of its $d$-dim hidden state.
- 4 All-to-Alls per MoE layer per step: dispatch & combine, forward & backward.
```

```{attention} Q&A
:class: dropdown
*Pros?*
- Expert weights not replicated → expert memory per device ÷ $p_\text{ep}$.
- Params scale w/o FLOPs per token scaling.

*Cons?*
- All-to-All sits on the critical path ← the expert FFN needs the dispatched tokens first.
- Load imbalance → hot device sets step time, or tokens over capacity are dropped.
- Cross-node All-to-All is slow.

*Why All-to-All, ❌All-Gather of expert weights?*
- Token sends $K\,d$ values. 1 expert holds ~$3\,d\,d_\text{ff}$ weights (SwiGLU).
- Gather all experts' weights to every device → moves all params → defeats sparsity.

*How do systems limit All-to-All cost?*
- Keep EP inside 1 node (NVLink) when possible.
- Top-1 routing (Switch Transformer) → 1 copy per token → ½ the dispatch bytes of top-2 (GShard).
- Node-limited routing: DeepSeek-V3 sends each token to ≤ 4 nodes {cite:p}`deepseekai2024deepseekv3`.

*EP vs TP for experts?*
- TP: split every expert matrix → All-Reduce per expert → poor for many small experts.
- EP: keep each expert whole on 1 device → better for many experts.
```

&nbsp;

### 3D Parallelism
- **What**: Combine DP × TP × PP over a 3D device mesh: $p=p_\text{dp}\,p_\text{tp}\,p_\text{pp}$. {cite:p}`narayanan2021efficient`
- **Why**: Each scheme alone hits a wall:
    - DP: model must fit on 1 device.
    - TP: All-Reduces in every layer → only fast inside 1 node.
    - PP: bubbles ⬆️ w/ #stages → needs many microbatches.
    - → Combine them. Map each to the network tier that fits its comm pattern.
- **How**:
    1. **TP inside a node** ($p_\text{tp}\le$ #GPUs per node, e.g., 8): most frequent, largest comm → NVLink.
    2. **PP across nodes**: only P2P activations at stage boundaries → tolerates slower IB.
    3. **DP outermost**: 1 grad sync per step, overlapped w/ backward.
    4. Model-parallel size $p_\text{tp}\,p_\text{pp}$ = smallest size that fits the model. Scale the rest w/ DP.
    5. Modern stacks add SP (w/ TP), CP, EP & ZeRO-1 for the DP dim → "4D/5D parallelism".
- Caveat: "3D parallelism" = DeepSpeed's term. Narayanan et al. call it PTD-P.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $m$: #Microbatches per pipeline per step (as in [Pipeline Parallelism](#pipeline-parallelism); overrides global #samples).
    - $v$: #Model chunks per device (interleaved schedule).
    - $B$: Global batch size (sequences).

Layout:

$$
p=p_\text{dp}\,p_\text{tp}\,p_\text{pp},\qquad B=p_\text{dp}\,b\,m
$$

Pipeline bubble fraction (bubble time ÷ ideal compute time):

$$
\frac{p_\text{pp}-1}{m}\quad\text{(GPipe, 1F1B)},\qquad\frac{p_\text{pp}-1}{v\,m}\quad\text{(interleaved 1F1B)}
$$

- Interleaving cuts the bubble by $v$, but sends $v$× more P2P messages.
```

```{tip} Derivation
:class: dropdown
*Why is the bubble fraction $(p_\text{pp}-1)/m$?*
1. Let 1 microbatch take $t_f$ forward & $t_b$ backward per stage.
2. Ideal time per step: $m(t_f+t_b)$.
3. Forward phase: the last microbatch leaves the last stage at $(m+p_\text{pp}-1)\,t_f$ ← $p_\text{pp}-1$ steps to fill the pipe.
4. Backward phase: same w/ $t_b$ → total $(m+p_\text{pp}-1)(t_f+t_b)$.
5. → Bubble $=(p_\text{pp}-1)(t_f+t_b)$ → fraction $=\frac{p_\text{pp}-1}{m}$. 1F1B reorders work but keeps the same bubble.
```

````{important} Code
:class: dropdown
```python
class Mesh3D:
    def __init__(self, p_tp, p_dp, p_pp):
        self.p_tp, self.p_dp, self.p_pp = p_tp, p_dp, p_pp

    def __call__(self, rank):
        ## TP varies fastest -> 1 TP group = consecutive ranks = 1 node (NVLink)
        tp = rank % self.p_tp
        dp = (rank // self.p_tp) % self.p_dp
        ## PP varies slowest -> stages land on different nodes (P2P over IB is cheap)
        pp = rank // (self.p_tp * self.p_dp)
        return tp, dp, pp

## Example: 16 GPUs = 2 nodes x 8; p_tp=4, p_dp=2, p_pp=2
mesh = Mesh3D(4, 2, 2)
print([mesh(r) for r in (0, 3, 4, 8)])  ## [(0, 0, 0), (3, 0, 0), (0, 1, 0), (0, 0, 1)] -> node 0 = stage 0, node 1 = stage 1
```
````

```{dropdown} Table: Parallelism Axes
| Scheme | Shards | Main comm | Typical placement |
|:--|:--|:--|:--|
| DP (DDP / ZeRO / FSDP) | Batch (+ model states for ZeRO/FSDP) | DDP: All-Reduce grads. ZeRO/FSDP: RS grads + AG params | Outermost, across nodes |
| TP | Weight matrices (hidden dim / heads) | 2 All-Reduces per layer forward, 2 backward | Inside 1 node |
| SP | LayerNorm/dropout activations along $s$ | RS + AG (replaces TP All-Reduce) | Same group as TP |
| CP | All activations along $s$ | Ring P2P of K/V | Fast links |
| PP | Layers | P2P activations at stage boundaries | Across nodes |
| EP | Experts | 2 All-to-Alls per MoE layer forward, 2 backward | Inside 1 node preferred |
```

```{attention} Q&A
:class: dropdown
*Pros?*
- Scales to trillions of params: 1T-param GPT at 502 PFLOP/s on 3072 A100s (52% of peak per GPU).
- Each comm pattern runs on the link tier it fits.

*Why TP inside a node, ❌across nodes?*
- TP: 2 All-Reduces of the full $s\,b\,d$ activation per layer forward → on the critical path, hard to overlap.
- Inter-node bandwidth ≪ NVLink → TP across nodes stalls every layer.

*Why not use only PP across nodes?*
- Bubble ∝ $p_\text{pp}/m$ → more stages need more microbatches → larger global batch.
- 1F1B: the 1st stage holds activations of up to $p_\text{pp}$ in-flight microbatches.

*Microbatch size trade-off?*
- Larger $b$ → larger GEMMs → ⬆️GPU efficiency.
- Fixed $B$ → larger $b$ → smaller $m$ → ⬆️bubble.

*How does it relate to the [Parallelism Comparison](#distributed-training) table?*
- The table compares each scheme alone. 3D parallelism stacks them: DP for throughput, TP for width, PP for depth.
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
| **INT8** | 8    | –             | –             | $=[-128,127]$                | \~2–3 digits      | 1 byte           |
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

### FP8 Training
- **What**: Run matmuls in 8-bit floating point w/ per-tensor scaling; keep master weights, optimizer states & accumulation in higher precision. {cite:p}`micikevicius2022fp8`
- **Why**: Speed & memory.
    - FP8 tensor-core peak = 2× BF16 (H100).
    - ½ the bytes of BF16 → ⬇️activation memory & bandwidth.
- **How**:
    1. **2 formats**: E4M3 (more precision) for weights & activations. E5M2 (more range) for grads.
    2. **Per-tensor scaling**: multiply the tensor by $\kappa$ → its amax lands near the FP8 max → cast → GEMM w/ FP32 accumulation → divide output by the scales.
    3. **Delayed scaling**: compute $\kappa$ from the amax history of past steps → ❌extra read of the tensor before the cast. Risk: sudden outlier → clipped.
    4. Keep the rest in BF16/FP32: LayerNorm, softmax, master weights, optimizer states.
    5. **Fine-grained scaling** (DeepSeek-V3): 1 scale per 1×128 activation tile & per 128×128 weight block → outlier hurts only its block → E4M3 for ALL tensors. Partial sums promoted to FP32 every 128 elements {cite:p}`deepseekai2024deepseekv3`.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $X\in\mathbb{R}^{n_\text{tok}\times d_\text{in}}$: Input activations.
    - $Y\in\mathbb{R}^{n_\text{tok}\times d_\text{out}}$: Output.
- Params:
    - $W\in\mathbb{R}^{d_\text{out}\times d_\text{in}}$: Weight.
- Misc:
    - $\kappa_X$: Scale factor of tensor $X$.
    - $q_\text{max}$: Largest finite FP8 value (E4M3: 448. E5M2: 57344).
    - $H$: Amax history length.

Quantize (per tensor):

$$
\hat X=\text{cast}_\text{FP8}\big(\text{clip}(\kappa_XX,\,-q_\text{max},\,q_\text{max})\big),\qquad\kappa_X=\frac{q_\text{max}}{\text{amax}(X)}
$$

- $\text{amax}(X)=\max_{i,j}|X_{ij}|$.

FP8 GEMM:

$$
Y\approx\frac{1}{\kappa_X\kappa_W}\,\hat X\hat W^T\quad\text{(FP32 accumulation)}
$$

Delayed scaling (step $t$):

$$
\kappa_X^{(t)}=\frac{q_\text{max}}{\max_{1\le h\le H}\text{amax}\big(X^{(t-h)}\big)}
$$
```

```{dropdown} Table: FP8 Formats
| | E4M3 | E5M2 |
|:--|:--|:--|
| Exponent / mantissa bits | 4 / 3 | 5 / 2 |
| Exponent bias | 7 | 15 |
| Max normal | 448 | 57344 |
| Min normal | $2^{-6}$ | $2^{-14}$ |
| Min subnormal | $2^{-9}$ | $2^{-16}$ |
| Infinity | ❌ | ✅ |
| NaN encodings | 1 per sign (S.1111.111) | IEEE-style (S.11111.{01,10,11}) |
| Default use | Weights, activations | Grads |
```

````{important} Code
:class: dropdown
```python
import torch

class DelayedScalingFP8:
    def __init__(self, dtype=torch.float8_e4m3fn, history=4):
        self.dtype = dtype
        self.qmax = torch.finfo(dtype).max  ## 448 (E4M3) / 57344 (E5M2)
        self.amax_hist = torch.zeros(history)

    def __call__(self, x):
        ## scale from PAST amaxes -> no extra pass over x before the cast
        amax = self.amax_hist.max()
        kappa = self.qmax / amax if amax > 0 else torch.tensor(1.0)
        ## clip: a stale scale can push new outliers past qmax (E4M3 has no inf)
        x8 = (x * kappa).clamp(-self.qmax, self.qmax).to(self.dtype)
        self.amax_hist = torch.roll(self.amax_hist, 1)
        self.amax_hist[0] = x.abs().max()  ## record the CURRENT amax for future steps
        return x8, kappa

    @staticmethod
    def dequant(x8, kappa):
        return x8.float() / kappa

## Example: amax >> 448
fp8 = DelayedScalingFP8()
x = torch.randn(4, 4) * 1e3
for step in range(2):
    x8, kappa = fp8(x)
    print(step, (fp8.dequant(x8, kappa) - x).abs().max().item())
## step 0: kappa=1 (empty history) -> values > 448 clipped -> large error
## step 1: kappa=448/amax -> error ~ relative FP8 rounding only
```
````

```{attention} Q&A
:class: dropdown
*Why 2 formats?*
- Grads: wide dynamic range, low precision suffices → E5M2.
- Weights & activations: narrower range after scaling, need precision → E4M3.

*Why scale at all?*
- E4M3 covers only $2^{-9}$ to 448 (~18 binades). BF16 covers ~$2^{-133}$ to $2^{128}$.
- → Must map each tensor's own range onto the narrow FP8 window.
- Same idea as loss scaling in [Mixed Precision Training](#mixed-precision-training), but per tensor.

*Delayed vs current scaling?*
- Delayed: scale from past amax → 1 read of the tensor. Stale on spikes.
- Current: amax of the tensor now → exact, 1 extra read.

*What breaks per-tensor scaling?*
- Activation outliers: 1 huge value sets the scale → small values underflow.
- Fix: fine-grained (tile/block) scales.

*Does FP8 cut optimizer memory?*
- ❌. Master weights & Adam states stay FP32.
- Savings: GEMM speed, FP8 activations stored for backward, FP8 comm.

*Why is FP8 harder than FP16 mixed precision?*
- FP16 (5 exp bits) w/ 1 global loss scale suffices.
- FP8 has 2–3 mantissa bits & a narrow range → every tensor needs its own scale, & accumulation must stay high precision.
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

&nbsp;

### MFU
- **Name**: Model FLOPs Utilization {cite:p}`chowdhery2022palm`
- **What**: Achieved model FLOP/s ÷ peak hardware FLOP/s.
- **Why**: Comparable efficiency across models, hardware & implementations.
    - Raw [Throughput](#throughput) (tokens/s) depends on model size & hardware.
    - HFU counts recomputed FLOPs → rewards wasted work.
- **How**:
    1. Model FLOPs per token ≈ $6N$ (2N forward + 4N backward) + attention term.
    2. Model FLOP/s = tokens/s × model FLOPs per token.
    3. Divide by #devices × peak dense FLOP/s of the training dtype.
    - **HFU** (Hardware FLOPs Utilization): count ALL executed FLOPs, incl. activation recomputation → HFU ≥ MFU.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $R$: Observed throughput (tokens/s).
    - $d_h$: Head dim ($a\,d_h=d$ in standard MHA).

Definition:

$$
\text{MFU}=\frac{R\,\left(6N+12\,L\,a\,d_h\,s\right)}{p\,F}
$$

- $6N$: Dense matmul FLOPs per token, forward + backward.
- $12\,L\,a\,d_h\,s$: Attention $QK^T$ & $AV$ FLOPs per token, forward + backward, ❌causal discount.
- HFU w/ full activation recomputation (+1 forward): $6N+12\,L\,a\,d_h\,s\to8N+16\,L\,a\,d_h\,s$.
```

```{tip} Derivation
:class: dropdown
*Why $6N$ FLOPs per token?*
1. Forward: each weight does 1 multiply + 1 add per token → $2N$.
2. Backward: grad w.r.t. layer input ($2N$) + grad w.r.t. weights ($2N$) → $4N$.
3. Total: $6N$.

*Why $12\,L\,a\,d_h\,s$ for attention?*
1. 1 query vs $s$ keys: $QK^T$ costs $2\,s\,a\,d_h$ FLOPs per token per layer.
2. $AV$: another $2\,s\,a\,d_h$ → forward $4\,s\,a\,d_h$.
3. Backward = 2× forward → $12\,s\,a\,d_h$ per layer → ×$L$.
```

```{note} Example
:class: dropdown
PaLM 540B on 6144 TPU v4 chips ($F$ = 275 TFLOP/s each), $R$ = 238.3K tokens/s, $L=118$, $a=48$, $d_h=256$, $s=2048$:
- FLOPs per token: $6\cdot540\text{B}+12\cdot118\cdot48\cdot256\cdot2048\approx3.24\text{T}+0.036\text{T}=3.276\text{T}$.
- MFU $=\frac{238.3\text{K}\times3.276\text{T}}{6144\times275\text{T}}\approx46.2\%$.
- HFU = 57.8% ← recomputes 75% of non-attention forward FLOPs & all attention forward FLOPs → $7.5N+16\,L\,a\,d_h\,s$ per token.
- Same method on GPT-3 (V100): 21.3% MFU.
```

````{important} Code
:class: dropdown
```python
class MFU:
    def __init__(self, n_params, n_layers, n_heads, d_head, seq_len, n_devices, peak_flops):
        ## model FLOPs per token: 6N for dense matmuls + attention QK^T & AV (fwd + bwd)
        self.flops_per_tok = 6 * n_params + 12 * n_layers * n_heads * d_head * seq_len
        self.peak = n_devices * peak_flops

    def __call__(self, tokens_per_s):
        return tokens_per_s * self.flops_per_tok / self.peak

## Example: PaLM 540B
mfu = MFU(540e9, 118, 48, 256, 2048, 6144, 275e12)
print(f"{mfu(238.3e3):.3f}")  ## 0.462
```
````

```{attention} Q&A
:class: dropdown
*Why MFU < 100%?*
- Exposed comm (TP All-Reduce, DP sync, EP All-to-All).
- Memory-bound ops: LayerNorm, softmax, elementwise → [Roofline Model](#roofline-model).
- Pipeline bubbles, recomputation, kernel launch gaps, data loading.

*MFU vs HFU?*
- HFU counts recompute → can rise while real speed falls.
- MFU counts only required FLOPs → fixed by the model & throughput alone.

*Typical values?*
- Dense LLM pretraining on large clusters: ~40–55% (PaLM 540B: 46.2%. Megatron 530B w/ SP: 54.2%).

*Gotchas?*
- Use the dense peak, ❌the sparsity peak (often quoted as 2×).
- Use the peak of the training dtype (BF16 vs FP8 peaks differ 2×).
- MoE: $N$ = active params per token.
- Papers differ on the attention term (causal discount or not) → compare w/ care.
```

&nbsp;

### Roofline Model
- **What**: Attainable FLOP/s = min(peak compute, memory bandwidth × arithmetic intensity). {cite:p}`williams2009roofline`
- **Why**: Shows whether a kernel is compute-bound or memory-bound → which optimization can help.
- **How**:
    1. **Arithmetic intensity** $I$ = FLOPs ÷ bytes moved to/from main memory (HBM).
    2. Plot attainable FLOP/s vs $I$ (log-log): slanted roof (bandwidth), then flat roof (peak).
    3. **Ridge point** $I^*$: where the 2 roofs meet.
    4. $I<I^*$ → memory-bound → raise $I$ (fuse kernels, batch, fewer bytes per value).
    5. $I>I^*$ → compute-bound → fewer FLOPs or faster math (tensor cores, FP8).

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $I$: Arithmetic intensity (FLOPs per byte).
    - $B_\text{mem}$: Memory bandwidth (bytes/s).
    - $e$: Bytes per element.
    - $q$: Matrix side length.

Model:

$$
F_\text{att}(I)=\min\left(F,\;B_\text{mem}\,I\right),\qquad I^*=\frac{F}{B_\text{mem}}
$$

- $F_\text{att}$: Attainable FLOP/s.

Common kernels:

$$\begin{align*}
\text{Square GEMM }(q\times q)(q\times q)&:\ I=\frac{2q^3}{3eq^2}=\frac{2q}{3e} \\
\text{GEMV (batch-1 decode) }(q\times q)\,\mathbf{x}&:\ I\approx\frac{2q^2}{eq^2}=\frac{2}{e} \\
\text{Elementwise add}&:\ I=\frac{1}{3e}
\end{align*}$$
```

```{note} Example
:class: dropdown
A100 80GB SXM, BF16 dense: $F$ = 312 TFLOP/s, $B_\text{mem}$ = 2,039 GB/s → $I^*\approx153$ FLOP/byte.
- Square BF16 GEMM, $q=4096$: $I\approx1365$ → compute-bound.
- Batch-1 decode GEMV: $I\approx1$ → memory-bound → ≤ ~2 TFLOP/s (< 1% of peak) → [Decode](../nlp/infer/cache.md#decode).
- Elementwise add (BF16): $I=1/6$ → deeply memory-bound → fuse into neighbor kernels.
- H100 SXM: 989 TFLOP/s BF16 dense, 3.35 TB/s → $I^*\approx295$ → the ridge moves right each generation.
```

````{important} Code
:class: dropdown
```python
class Roofline:
    def __init__(self, peak_flops, mem_bw):
        self.peak, self.bw = peak_flops, mem_bw
        self.ridge = peak_flops / mem_bw  ## FLOP/byte where the 2 roofs meet

    def __call__(self, flops, bytes_moved):
        I = flops / bytes_moved
        ## left of the ridge, bandwidth caps speed; right of it, compute does
        return min(self.peak, self.bw * I), ("memory" if I < self.ridge else "compute")

## Example: A100 80GB SXM, BF16 dense
a100 = Roofline(312e12, 2039e9)
q = 4096
print(round(a100.ridge))              ## 153
print(a100(2 * q**3, 2 * 3 * q**2))   ## (312000000000000.0, 'compute')  square GEMM
print(a100(2 * q * q, 2 * q * q))     ## (2039000000000.0, 'memory')     batch-1 GEMV
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- 2 numbers per device ($F$, $B_\text{mem}$) + 1 per kernel ($I$) → instant bottleneck diagnosis.
- Says which optimization can help: ❌fewer FLOPs for a memory-bound kernel.

*Cons?*
- Upper bound only. Ignores latency, launch overhead, cache hierarchy (needs 1 roof per memory level), comm.
- Real bytes moved ≥ the minimum → measured $I$ < ideal $I$.

*How to raise $I$?*
- Batch: decode w/ batch $b$ → weights read once for $b$ tokens → $I\propto b$.
- Fuse kernels: [FlashAttention](../nlp/attn.md#flashattention) never writes the $s\times s$ scores to HBM.
- Fewer bytes per value: quantized weights → ⬆️$I$ for memory-bound decode.

*Why does lower precision help both regimes?*
- Memory-bound: fewer bytes for the same FLOPs → ⬆️$I$.
- Compute-bound: low-precision tensor cores → ⬆️$F$ ([FP8 Training](#fp8-training)).

*Is the same idea used for communication?*
- Yes. Compute vs link bandwidth → [Context Parallelism](#context-parallelism) hides K/V transfer iff block length $c\ge F\beta$.

*Operational vs arithmetic intensity?*
- Williams et al. say "operational intensity": FLOPs per byte of DRAM traffic. Same concept.
```
