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
# Efficiency
# Knowledge Distillation
- **What**: Large, pre-trained **teacher** model $\xrightarrow{\text{transfer knowledge}}$ Small **student** model
    - Teacher & Student are trained on the same data.
    - Teacher sees true labels (i.e., **hard targets**).
    - Student sees both true labels and teacher's outputs (i.e., **soft targets**).
- **Why**: Model compression + Domain-specific knowledge transfer.