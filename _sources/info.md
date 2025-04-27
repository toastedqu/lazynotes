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
# Information Theory
# Basics
## Information
- **What**: Surprise.
    - Less/More likely events $\rightarrow$ More/Less info
- **Why**: To mathematically analyze, optimize, & understand the fundamental limits of systems that store, process, & transmit data.
    - Data compression
    - Communication
    - Systems
    - Cryptography
    - ML
    - ...
- **How**: To quantify surprise (i.e., self-information), we want
    1. Higher surprise for less probable events.
    2. Zero surprise for deterministic events.
    3. If 2 independent events happen, their surprises should add up.
    - 2 & 3 $\rightarrow$ $\log P$
    - 1 $\rightarrow$ $-\log P$

```{dropdown} ELI5
*What is information?*
- You live in the middle of Egypt.
- Weather forecast: "It will be sunny tmrw."
- You: "lol as expected." (You didn't get much info)
- Weather forecast: "It will snow tmrw."
- You: "Hol up WTF???" (You got huge info)
```

```{admonition} Math
:class: note, dropdown
**Information**:

$$
I(x)=-\log P(x)
$$
```