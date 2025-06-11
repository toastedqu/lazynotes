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
# Precision & Quantization

## Precision
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

```{admonition} Math
:class: note, dropdown
Representation Formula:

$$
\text{value} = (-1)^{\text{sign}}1.\text{fraction}\times 2^{\text{actual\_exponent}}
$$

<br/>

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
<br/>

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

## Quantization
- **What**: High precision $\rightarrow$ Low precision
- **Why**: ⬇️Computational cost (time & memory)
    - BUT it risks model quality & training stability.