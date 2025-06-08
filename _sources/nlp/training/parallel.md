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

## FSDP (Fully Sharded Data Parallel)
- **What**: Shard model params, grads, and optimizer states across multi processes (GPUs), each taking a unique mini-batch of data.
- **Why**: More flexible than DDP, used when the model cannot fit in each GPU's memory.