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
# Distributed Training

## DDP (Distributed Data Parallel)
- **What**: Copy a model across multi processes (GPUs), each taking a unique mini-batch of data.
- **Why**: Faster than FSDP, used when the model can easily fit in each GPU's memory.
- **How**:
	1. Each process hosts a copy of the model.
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

## FSDP (Fully Sharded Data Parallel)
- **What**: Shard model params, grads, and optimizer states across multi processes (GPUs), each taking a unique mini-batch of data.
- **Why**: More flexible than DDP, used when the model cannot fit in each GPU's memory.