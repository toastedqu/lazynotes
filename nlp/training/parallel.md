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

## FSDP (Fully Sharded Data Parallel)
- **What**: Shard model params, grads, and optimizer states across multi processes (GPUs), each taking a unique mini-batch of data.