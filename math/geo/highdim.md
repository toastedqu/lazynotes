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
# High-Dimensional Geometry
Counter-intuitive geometry of $\mathbb{R}^d$ for large $d$, & how to exploit it: random projection, distributed symbolic codes.
Study notes from {cite:t}`fds` & {cite:t}`hdp`. ML consequences for density & distance estimation → [Curse of Dimensionality](../../ml/misc.md#curse-of-dimensionality).

Notations:
- $d$: Ambient dim.
- $\mathbf{x}\in\mathbb{R}^d$: Point or random vector.
- $\|\cdot\|$: Euclidean norm.
- $\langle\cdot,\cdot\rangle$: Euclidean inner product.
- $B^d$: Unit ball $\{\mathbf{x}:\|\mathbf{x}\|\le1\}$.
- $S^{d-1}$: Unit sphere $\{\mathbf{x}:\|\mathbf{x}\|=1\}$.
- $\mathcal{N}(0,I_d)$: Standard Gaussian in $\mathbb{R}^d$.
- $\mathrm{Unif}(S^{d-1})$: Uniform distribution on $S^{d-1}$.
- $\epsilon$: Tolerance (distortion or inner-product bound).
- $m$: #points.
- $k$: Target dim of a projection.

- **Overrides**: $d$ = ambient dim (Tier-1 $n$). $k$ = target dim, ❌class idx.
- **Conventions**: $\ln$ = natural log. $c,C>0$: absolute constants, independent of $d,m,k$; value can change between lines.

&nbsp;

## Concentration

### Volume Concentration
- **What**: Unit-ball volume → 0 as $d$⬆️; almost all of it lies in a thin shell & near every equator.
- **Why**: Low-dim intuition ("most of a ball is deep inside") fails → wrong expectations about sampling, NN search & latent spaces.
- **How**:
    - **Shell**: Shrink the radius by $(1-\epsilon)$ → volume ×$(1-\epsilon)^d\approx e^{-\epsilon d}$ → almost nothing is left inside.
    - **Equator**: Slab $|x_1|\le c/\sqrt{d-1}$ holds $\ge1-\delta$ of the volume for $c=\sqrt{2\ln(2/\delta)}$, for every choice of "north" ← rotational symmetry.
    - **Shrinking**: $\pi^{d/2}$ grows slower than $\Gamma(d/2+1)$ → $V_d\to0$.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $V_d$: Volume of $B^d$.
    - $A_d$: Surface area of $S^{d-1}$.

Definition:

$$
V_d=\frac{\pi^{d/2}}{\Gamma\left(\frac d2+1\right)},\qquad A_d=dV_d=\frac{2\pi^{d/2}}{\Gamma\left(\frac d2\right)}
$$

- $\Gamma$: Gamma function, $\Gamma(z+1)=z\Gamma(z)$, $\Gamma(1/2)=\sqrt\pi$.
- $V_d$ peaks at $d=5$ ($\approx5.26$). $A_d$ peaks at $d=7$ ($\approx33.07$). Both → 0 superexponentially.
- Stirling: $V_d\approx\frac{1}{\sqrt{\pi d}}\left(\frac{2\pi e}{d}\right)^{d/2}$ → unit-volume ball has radius $\approx\sqrt{d/(2\pi e)}$.

Shell:

$$
\frac{\mathrm{vol}\left(B^d\setminus(1-\epsilon)B^d\right)}{V_d}=1-(1-\epsilon)^d\ge1-e^{-\epsilon d}
$$

Equator ({cite:t}`fds`, Thm 2.7), for $c\ge1$ & $d\ge3$:

$$
\frac{\mathrm{vol}\left\{\mathbf{x}\in B^d:|x_1|\le\frac{c}{\sqrt{d-1}}\right\}}{V_d}\ge1-\frac2ce^{-c^2/2}
$$

- $x_1$: Coordinate along any fixed unit direction.
- Fixed $c$ → fixed fraction $<1$, not "almost all". Target fraction $1-\delta$, $\delta<1$ → choose $c=\sqrt{2\ln(2/\delta)}\ge1$ → bound $=1-\delta/c\ge1-\delta$.

Ball vs cube $[-1,1]^d$:
- $V_d/2^d\to0$: the inscribed ball fills almost none of the cube.
- Cube corner at distance $\sqrt d$; face center at distance 1.
- Uniform cube point: $\mathbb{E}x_j^2=1/3$ → $\|\mathbf{x}\|\approx\sqrt{d/3}$ → outside the inscribed ball for $d>3$.
- Distance to the nearest vertex $\mathrm{sign}(\mathbf{x})$: $\sum_j(1-|x_j|)^2\approx d/3$ → also $\approx\sqrt{d/3}$ → ❌ near the corners.
- Near the boundary ≠ near a vertex: some coordinate is close to $\pm1$, most are not.
```

```{tip} Derivation
:class: dropdown
*Why does the volume formula hold?*

1. Gaussian integral in Cartesian coordinates: product of $d$ 1-D integrals.

    $$
    I=\int_{\mathbb{R}^d}e^{-\|\mathbf{x}\|^2}d\mathbf{x}=\left(\int_{-\infty}^{\infty}e^{-t^2}dt\right)^d=\pi^{d/2}
    $$

2. Same integral in polar coordinates: shell of radius $r$ has area $A_dr^{d-1}$. Substitute $u=r^2$.

    $$
    I=A_d\int_0^\infty r^{d-1}e^{-r^2}dr=\frac{A_d}{2}\Gamma\left(\frac d2\right)
    $$

3. Equate → $A_d=2\pi^{d/2}/\Gamma(d/2)$.
4. Integrate the shells from $r=0$ to $1$: $V_d=A_d\int_0^1r^{d-1}dr=A_d/d$.
5. $\frac d2\Gamma\left(\frac d2\right)=\Gamma\left(\frac d2+1\right)$ → $V_d=\pi^{d/2}/\Gamma\left(\frac d2+1\right)$.

*Why does the shell hold almost everything?*

1. Volume scales as $r^d$ → the inner ball of radius $1-\epsilon$ has fraction $(1-\epsilon)^d$.
2. $1-\epsilon\le e^{-\epsilon}$ → fraction $\le e^{-\epsilon d}$.
```

```{note} Example
:class: dropdown
| $d$ | $V_d$ | $V_d/2^d$ (ball / cube) |
|:--|:--|:--|
| 2 | 3.14 | 0.785 |
| 3 | 4.19 | 0.524 |
| 5 | 5.26 | 0.164 |
| 10 | 2.55 | 0.0025 |
| 20 | 0.026 | $2.5\times10^{-8}$ |
| 100 | $2.4\times10^{-40}$ | $1.9\times10^{-70}$ |

- $d=100$, $\epsilon=0.05$ → outer 5% of the radius holds $1-0.95^{100}\approx99.4\%$ of the volume.
- Rejection sampling of $B^{10}$ from the cube accepts only 0.25% of draws.
```

```{attention} Q&A
:class: dropdown
*How can the mass be near the surface & near the equator at the same time?*

- No conflict: each statement excludes a different small set.
- Typical point: $\|\mathbf{x}\|\approx1$ & $|x_1|\approx1/\sqrt d$ → the remaining coordinates carry the norm.

*Is "volume → 0" a deep fact?*

- ❌ Partly a unit artifact: radius is fixed at 1 while the natural scale grows as $\sqrt d$.
- Ball of radius $\sqrt d$ → volume ⬆️ without bound.
- Scale-free facts (shell & equator fractions) are the real content.

*Consequences for ML?*

- Uniform samples in a box or ball → nearly all near the boundary → every query is near the boundary → [Curse of Dimensionality](../../ml/misc.md#curse-of-dimensionality).
```

&nbsp;

### Gaussian Annulus
- **What**: $\mathbf{x}\sim\mathcal{N}(0,I_d)$ → $\|\mathbf{x}\|=\sqrt d\pm O(1)$: a thin "soap bubble".
- **Why**: Density peaks at the origin, yet almost no mass is near it → controls sampling, latent interpolation & init scales.
- **How**:
    - $\|\mathbf{x}\|^2=\sum_jx_j^2$ = sum of $d$ iid terms, mean 1, variance 2 → mean $d$, std $\sqrt{2d}$.
    - Square root shrinks the spread: $\sqrt{d\pm\sqrt{2d}}\approx\sqrt d\pm1/\sqrt2$.
    - Shell area ⬆️ as $r^{d-1}$ & density ⬇️ as $e^{-r^2/2}$ → product peaks at $r=\sqrt{d-1}$.

```{note} Math
:class: dropdown
Gaussian Annulus Theorem ({cite:t}`fds`, Thm 2.9), for $\beta\le\sqrt d$:

$$
P\left(\left|\|\mathbf{x}\|-\sqrt d\right|\ge\beta\right)\le3e^{-c\beta^2}
$$

- $\beta$: Half-width of the annulus.
- FDS proof gives $c=1/96$.

Sharper, via [Concentration of Measure](#concentration-of-measure) ($\|\cdot\|$ is 1-Lipschitz):

$$
P\left(\left|\|\mathbf{x}\|-\mathbb{E}\|\mathbf{x}\|\right|\ge t\right)\le2e^{-t^2/2},\qquad\sqrt{d-1}\le\mathbb{E}\|\mathbf{x}\|\le\sqrt d
$$

- $t$: Deviation.

Properties:
- $\mathbb{E}\|\mathbf{x}\|^2=d$, $\mathrm{Var}\|\mathbf{x}\|^2=2d$.
- $\|\mathbf{x}\|$ is chi-distributed → $\mathrm{Var}\|\mathbf{x}\|\to1/2$ → shell width $O(1)$, independent of $d$.
- Radial density $\propto r^{d-1}e^{-r^2/2}$ → mode $r=\sqrt{d-1}$.
- $\mathbf{x}/\|\mathbf{x}\|\sim\mathrm{Unif}(S^{d-1})$, independent of $\|\mathbf{x}\|$ → $\mathcal{N}(0,I_d)\approx\mathrm{Unif}(\sqrt dS^{d-1})$.
- Two independent samples: $\|\mathbf{x}-\mathbf{x}'\|\approx\sqrt{2d}$, $\langle\mathbf{x},\mathbf{x}'\rangle$ has mean 0 & std $\sqrt d$ → angle $\approx90°$ ([Near-Orthogonality](#near-orthogonality)).
- $\mathcal{N}(\boldsymbol\mu,\sigma^2I_d)$ → shell of radius $\sigma\sqrt d$ around $\boldsymbol\mu$.
```

```{tip} Derivation
:class: dropdown
*Why $\sqrt{d-1}\le\mathbb{E}\|\mathbf{x}\|\le\sqrt d$?*

1. Upper: Jensen → $\mathbb{E}\|\mathbf{x}\|\le\sqrt{\mathbb{E}\|\mathbf{x}\|^2}=\sqrt d$.
2. Lower: Gaussian Poincaré inequality $\mathrm{Var}f(\mathbf{x})\le\mathbb{E}\|\nabla f(\mathbf{x})\|^2$.
3. $f=\|\cdot\|$ → $\|\nabla f\|=1$ → $\mathrm{Var}\|\mathbf{x}\|\le1$.
4. $(\mathbb{E}\|\mathbf{x}\|)^2=\mathbb{E}\|\mathbf{x}\|^2-\mathrm{Var}\|\mathbf{x}\|\ge d-1$.

*How does the FDS proof work?*

1. $|r-\sqrt d|\ge\beta$, multiply by $r+\sqrt d\ge\sqrt d$ → $|r^2-d|\ge\beta\sqrt d$, with $r=\|\mathbf{x}\|$.
2. $r^2-d=\sum_j(x_j^2-1)$: sum of $d$ iid zero-mean terms w/ bounded moments.
3. Tail bound for such sums → probability $\le3e^{-\beta^2/96}$.

*Why is the radial mode at $\sqrt{d-1}$?*

1. Maximize $\ln\left(r^{d-1}e^{-r^2/2}\right)=(d-1)\ln r-r^2/2$.
2. Derivative $(d-1)/r-r=0$ → $r=\sqrt{d-1}$.
```

````{important} Code
:class: dropdown
```python
import numpy as np

class Slerp:
    ## spherical linear interpolation: moves along the arc, not the chord
    def __call__(self, p0, p1, t):
        cos = p0 @ p1 / (np.linalg.norm(p0) * np.linalg.norm(p1))
        omega = np.arccos(np.clip(cos, -1.0, 1.0))  ## angle between endpoints
        if omega < 1e-6:
            return (1 - t) * p0 + t * p1  ## ~parallel: arc = chord, avoid 0/0
        if np.pi - omega < 1e-6:
            raise ValueError("antipodal endpoints: great circle not unique")
        return (np.sin((1 - t) * omega) * p0 + np.sin(t * omega) * p1) / np.sin(omega)

## Example: GAN-style latent z ~ N(0, I_512)
rng = np.random.default_rng(0)
d = 512
r = np.linalg.norm(rng.standard_normal((10_000, d)), axis=1)
print(r.mean(), r.std())                     ## ~22.6 (= sqrt(512)), ~0.7: thin shell
z0, z1 = rng.standard_normal((2, d))
print(np.linalg.norm((z0 + z1) / 2))         ## ~16.0 = sqrt(d/2): lerp midpoint leaves the shell
print(np.linalg.norm(Slerp()(z0, z1, 0.5)))  ## ~22.6: slerp stays on the shell
```
````

```{attention} Q&A
:class: dropdown
*Mode vs typical set?*

- Density max at $\mathbf{0}$, but $P(\|\mathbf{x}\|\le\sqrt d/2)$ is exponentially small.
- → "Most likely point" ≠ "typical point". Same split as the [typical set](../info.md#typical-sets-and-the-aep) in information theory.
- → A mode-seeking estimate can be atypical of real samples.

*Why interpolate latents with slerp?*

- Lerp midpoint of two samples: $\frac12(\mathbf{z}_0+\mathbf{z}_1)\sim\mathcal{N}(0,\frac12I_d)$ → norm $\approx\sqrt{d/2}$ → off the shell.
- Generator saw almost no latents of that norm → output can degrade. Recommended for GANs by White (2016), "Sampling Generative Networks".
- Slerp keeps the norm $\approx\sqrt d$.
- Caveat: the size of the effect depends on the generator. Some models tolerate lerp.
- Alternative: lerp, then rescale to norm $\sqrt d$.

*Why does a high-dim Gaussian look like a uniform sphere?*

- Direction: exactly uniform. Norm: almost constant.
- Converse (Poincaré–Borel): one coordinate of $\mathrm{Unif}(\sqrt dS^{d-1})$ → $\mathcal{N}(0,1)$ as $d\to\infty$.

*How does this relate to weight init?*

- Pre-activation $\mathbf{w}^\top\mathbf{x}$ w/ iid $\mathcal{N}(0,\sigma^2)$ weights → std $\sigma\|\mathbf{x}\|$.
- $\|\mathbf{x}\|\approx\sqrt{d}\cdot$ RMS entry, $d$ = fan-in → $\sigma\propto1/\sqrt{d}$ keeps the scale (Xavier/He).
```

&nbsp;

### Concentration of Measure
- **What**: Lipschitz function of a high-dim Gaussian or uniform-sphere vector ≈ constant.
    - Gaussian: fluctuation $O(L)$, independent of $d$.
    - Unit sphere: fluctuation $O(L/\sqrt d)$.
- **Why**: One principle behind the annulus, near-orthogonality, JL & the LLN → many weak influences → near-deterministic aggregate.
- **How**:
    - Lipschitz → gap $t$ in output values = distance $\ge t/L$ between level sets $\{f\le M\}$ & $\{f\ge M+t\}$.
    - Gaussian/spherical isoperimetry: set w/ half the measure → points at distance $\ge t/L$ from it have exponentially small measure.
    - Geometric form (Lévy): any set w/ half the measure → its $t$-neighborhood holds almost all the measure.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $f$: $L$-Lipschitz function.
    - $L$: Lipschitz constant.
    - $t$: Deviation.
    - $\sigma$: Uniform probability measure on $S^{d-1}$.
    - $\mathcal{A}_t$: Set of points within distance $t$ of $\mathcal{A}\subseteq S^{d-1}$.

Definition:

$$
|f(\mathbf{x})-f(\mathbf{y})|\le L\|\mathbf{x}-\mathbf{y}\|\quad\forall\mathbf{x},\mathbf{y}
$$

Gaussian concentration (Tsirelson–Ibragimov–Sudakov), $\mathbf{x}\sim\mathcal{N}(0,I_d)$:

$$
P\left(|f(\mathbf{x})-\mathbb{E}f(\mathbf{x})|\ge t\right)\le2e^{-t^2/(2L^2)}
$$

Sphere concentration (Lévy), $\mathbf{u}\sim\mathrm{Unif}(S^{d-1})$:

$$
P\left(|f(\mathbf{u})-\mathbb{E}f(\mathbf{u})|\ge t\right)\le2e^{-cdt^2/L^2}
$$

- {cite:t}`hdp` (Thm 5.1.4) states it on $\sqrt dS^{d-1}$ w/ fluctuation $O(L)$; rescale by $1/\sqrt d$ → form above.

Isoperimetric form:

$$
\sigma(\mathcal{A})\ge\tfrac12\ \Rightarrow\ \sigma(\mathcal{A}_t)\ge1-2e^{-cdt^2}
$$

Special cases:
- $f(\mathbf{x})=\frac1d\sum_jx_j$: $L=1/\sqrt d$ → $2e^{-dt^2/2}$ → LLN w/ an exponential rate.
- $f(\mathbf{x})=\|\mathbf{x}\|$: $L=1$ → [Gaussian Annulus](#gaussian-annulus).
- $f(\mathbf{u})=\langle\mathbf{u},\mathbf{v}\rangle$, $\|\mathbf{v}\|=1$: $L=1$ → [Near-Orthogonality](#near-orthogonality).
- $f(\mathbf{x})=\|\Pi\mathbf{x}\|$, $\Pi$ orthogonal projection onto a $k$-dim subspace: $L=1$ → [JL](#johnson-lindenstrauss-lemma).
- Mean & median of $f$ differ by $O(L)$ (Gaussian) → interchangeable up to constants.
- Bounded independent coordinates: McDiarmid (bounded differences) & Talagrand (convex Lipschitz $f$) give analogous bounds.
```

```{tip} Derivation
:class: dropdown
*Why are the Lipschitz constants of the special cases what they are?*

1. Mean: $\left|\frac1d\langle\mathbf{1},\mathbf{x}-\mathbf{y}\rangle\right|\le\frac{\|\mathbf{1}\|}{d}\|\mathbf{x}-\mathbf{y}\|=\frac{1}{\sqrt d}\|\mathbf{x}-\mathbf{y}\|$ ← Cauchy–Schwarz.
2. Norm: $\big|\|\mathbf{x}\|-\|\mathbf{y}\|\big|\le\|\mathbf{x}-\mathbf{y}\|$ ← triangle inequality.
3. Inner product: $|\langle\mathbf{x}-\mathbf{y},\mathbf{v}\rangle|\le\|\mathbf{x}-\mathbf{y}\|$ for unit $\mathbf{v}$ ← Cauchy–Schwarz.
4. Projection: $\big|\|\Pi\mathbf{x}\|-\|\Pi\mathbf{y}\|\big|\le\|\Pi(\mathbf{x}-\mathbf{y})\|\le\|\mathbf{x}-\mathbf{y}\|$ ← operator norm of $\Pi$ is 1.

*Why does the mean case reproduce the exact Gaussian tail?*

1. $\frac1d\sum_jx_j\sim\mathcal{N}(0,1/d)$.
2. Gaussian tail: $P(|Z|\ge t)\le2e^{-t^2/(2\cdot1/d)}=2e^{-dt^2/2}$.
3. Bound w/ $L=1/\sqrt d$ gives the same → constant $1/2$ is sharp.
```

```{attention} Q&A
:class: dropdown
*Why "Lipschitz" & not just "smooth"?*

- $L$ caps the sensitivity to all coordinates together.
- $\|\mathbf{x}\|^2$ is smooth but not globally Lipschitz → fluctuation $\sqrt{2d}$ grows w/ $d$.
- $\|\mathbf{x}\|$ is 1-Lipschitz → fluctuation $O(1)$.

*How does this differ from the CLT?*

- CLT: shape of fluctuations of sums, asymptotic.
- Concentration: non-asymptotic tail bound for any Lipschitz $f$, also nonlinear ones.

*How does it relate to ML?*

- Generalization bounds: McDiarmid → Rademacher-complexity bounds.
- [JL](#johnson-lindenstrauss-lemma) & [Random Projection](#random-projection).
- Wide nets at random init: averages over many neurons concentrate → kernel (NTK) becomes deterministic as width → ∞.
- Single pre-activations keep nonzero variance → Gaussian-process limit, not a constant.

*What does it tell us about intelligence/AC?*

- Macro-variables of systems w/ many weakly coupled parts are near-deterministic → stable macro-states (thermodynamics).
- Hypothesis: a large population of noisy neurons or units → reliable population-level signals.
- Caveat: correlated noise breaks the independence premise → gains from averaging saturate.
```

&nbsp;

### Near-Orthogonality
- **What**: Independent random directions in $\mathbb{R}^d$ are almost orthogonal: $\cos\approx\pm1/\sqrt d$.
- **Why**: Only $d$ exactly orthogonal directions exist, but $e^{c\epsilon^2d}$ almost orthogonal ones exist → room for exponentially many codes w/ low interference.
- **How**:
    1. Fix $\mathbf{v}$ ← rotational symmetry → $\langle\mathbf{u},\mathbf{v}\rangle$ = one coordinate $u_1$ of a random unit vector.
    2. $\sum_ju_j^2=1$ & symmetry → $\mathbb{E}u_1^2=1/d$ → typical $|\cos|\approx1/\sqrt d$.
    3. Tail $\le e^{-d\epsilon^2/2}$ → union bound over all pairs stays small for $e^{d\epsilon^2/4}$ vectors.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\mathbf{u},\mathbf{v}$: Independent $\mathrm{Unif}(S^{d-1})$ vectors.
    - $N$: #vectors.

Definition: unit vectors $\mathbf{u},\mathbf{v}$ are $\epsilon$-orthogonal ⇔ $|\langle\mathbf{u},\mathbf{v}\rangle|\le\epsilon$.

Properties:
- $\mathbb{E}\langle\mathbf{u},\mathbf{v}\rangle=0$, $\mathrm{Var}\langle\mathbf{u},\mathbf{v}\rangle=1/d$ exactly.
- Density of $s=\langle\mathbf{u},\mathbf{v}\rangle$ on $[-1,1]$: $\propto(1-s^2)^{(d-3)/2}$ → $\sqrt d\,s\to\mathcal{N}(0,1)$.
- Cap bound, $0\le\epsilon<1$:

    $$
    P\left(\langle\mathbf{u},\mathbf{v}\rangle\ge\epsilon\right)\le e^{-d\epsilon^2/2}
    $$

- Packing: some set of $N\ge e^{d\epsilon^2/4}$ unit vectors is pairwise $\epsilon$-orthogonal. Exact orthogonality → $N\le d$.
- Distances: $\|\mathbf{u}-\mathbf{v}\|^2=2-2\langle\mathbf{u},\mathbf{v}\rangle$.
- $N$ independent $\mathrm{Unif}(S^{d-1})$ vectors: $\max_{i\ne i'}|\langle\mathbf{u}_i,\mathbf{u}_{i'}\rangle|=O_P\left(\sqrt{\ln N/d}\right)$ → all pairwise distances → $\sqrt2$ if $\ln N=o(d)$.
- Unnormalized $\mathbf{x},\mathbf{x}'$ w/ mutually independent, zero-mean, unit-variance entries: $\langle\mathbf{x},\mathbf{x}'\rangle$ has mean 0 & std $\sqrt d$.
```

```{tip} Derivation
:class: dropdown
*Why do $e^{d\epsilon^2/4}$ pairwise $\epsilon$-orthogonal unit vectors exist?*

1. Draw $N$ iid $\mathrm{Unif}(S^{d-1})$ vectors.
2. One pair: $P(|\langle\mathbf{u},\mathbf{v}\rangle|\ge\epsilon)\le2e^{-d\epsilon^2/2}$ ← cap bound, both signs.
3. Union over $\binom N2$ pairs: $P(\text{some bad pair})\le N(N-1)e^{-d\epsilon^2/2}<N^2e^{-d\epsilon^2/2}$.
4. $N=e^{d\epsilon^2/4}$ → bound $<1$ → some draw has no bad pair → the set exists (probabilistic method).

*Why is $\mathrm{Var}\langle\mathbf{u},\mathbf{v}\rangle$ exactly $1/d$?*

1. Condition on $\mathbf{v}$; rotate so $\mathbf{v}=\mathbf{e}_1$ → $\langle\mathbf{u},\mathbf{v}\rangle=u_1$.
2. $\sum_j\mathbb{E}u_j^2=\mathbb{E}\|\mathbf{u}\|^2=1$; all $\mathbb{E}u_j^2$ equal ← symmetry → $\mathbb{E}u_1^2=1/d$.
3. $\mathbb{E}u_1=0$ ← symmetry → $\mathrm{Var}=1/d$.
```

```{note} Example
:class: dropdown
| $d$ | std of $\cos$ between random directions |
|:--|:--|
| 3 | 0.58 |
| 100 | 0.10 |
| 768 | 0.036 |
| 10,000 | 0.01 |

- $d=10{,}000$, $\epsilon=0.1$ → at least $e^{25}\approx7.2\times10^{10}$ directions w/ pairwise $|\cos|<0.1$.
- Same $d$: $P(|\cos|\ge0.05)\le2e^{-12.5}\approx7.5\times10^{-6}$ for one random pair.
```

```{attention} Q&A
:class: dropdown
*Why does attention divide by $\sqrt{d_k}$?*

- Query & key w/ mutually independent, zero-mean, unit-variance entries → dot product std $\sqrt{d_k}$ → softmax saturates.
- $\div\sqrt{d_k}$ → unit variance at every head width → [Scaled Dot-Product Attention](../../nlp/attn.md#scaled-dot-product-attention).

*What does it do to nearest-neighbor search?*

- $N$ random directions w/ $\ln N=o(d)$: all pairwise distances → $\sqrt2$ → nearest & farthest neighbor barely differ → [Curse of Dimensionality](../../ml/misc.md#curse-of-dimensionality).
- Real data is not iid across coordinates → low [intrinsic dimension](data.md#intrinsic-dimension) restores contrast.

*How does it enable superposition?*

- Assign features to near-orthogonal directions → #features ≫ $d$ w/ pairwise interference $\le\epsilon$.
- Sparse activation → few features active at once → summed interference stays small → [Superposition](repr.md#superposition).
- Caveat: trained models learn the arrangement; the packing bound shows only that the capacity exists.

*Is the exponential count tight?*

- Near-tight: $N$ pairwise $\epsilon$-orthogonal unit vectors need $d=\Omega\left(\frac{\ln N}{\epsilon^2\ln(1/\epsilon)}\right)$ for $1/\sqrt N\le\epsilon\le1/2$ (Alon).
- → Max $N$ is $e^{\Theta(d\epsilon^2)}$ up to the $\ln(1/\epsilon)$ factor.

*What does it tell us about intelligence/AC?*

- One $d$-dim state can address exponentially many concepts w/ little crosstalk.
- Basis of distributed codes: [HDC](#hyperdimensional-computing) & [HRR](#hrr) use it by design; LLMs appear to use it via superposition.
- Hypothesis: high-dim neural populations exploit the same capacity. Contested as a model of brain coding.
```

&nbsp;

## Random Projection

### Johnson-Lindenstrauss Lemma
- **What**: Any $m$ points in $\mathbb{R}^d$ map linearly into $\mathbb{R}^k$, $k=O(\epsilon^{-2}\log m)$, w/ all pairwise distances kept within $1\pm\epsilon$. {cite:p}`johnson1984extensions`
- **Why**: Distance-based algorithms (NN search, clustering, kernel methods) cost $\propto d$ → cut $d$ to $O(\log m)$, independent of $d$, w/o learning anything.
- **How**:
    1. Draw a random Gaussian matrix, scale by $1/\sqrt k$.
    2. One fixed difference vector: projected squared norm = average of $k$ iid squared Gaussians → concentrates; failure probability $e^{-ck\epsilon^2}$.
    3. Union bound over $\binom m2$ pairs → $k\propto\epsilon^{-2}\ln m$ makes all pairs succeed.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathcal{X}=\{\mathbf{x}_1,\dots,\mathbf{x}_m\}\subset\mathbb{R}^d$: Point set.
- Params:
    - $R\in\mathbb{R}^{k\times d}$: Random matrix, iid $\mathcal{N}(0,1)$ entries.
- Misc:
    - $i'$: Second sample idx.

Theorem: for $0<\epsilon<1$ &

$$
k\ge\frac{4\ln m}{\epsilon^2/2-\epsilon^3/3},
$$

a linear map $f:\mathbb{R}^d\to\mathbb{R}^k$ exists with

$$
(1-\epsilon)\|\mathbf{x}_i-\mathbf{x}_{i'}\|^2\le\|f(\mathbf{x}_i)-f(\mathbf{x}_{i'})\|^2\le(1+\epsilon)\|\mathbf{x}_i-\mathbf{x}_{i'}\|^2\quad\forall i,i'
$$

Construction:

$$
f(\mathbf{x})=\frac{1}{\sqrt k}R\mathbf{x}
$$

- Random $k$-dim subspace scaled by $\sqrt{d/k}$ → same bound.

Properties:
- $k$ independent of $d$.
- Oblivious: draw $f$ independently of data & queries.
- Fixed future queries: include them in $\mathcal{X}$ (count them in $m$) → still valid.
- ❌ Guarantee for queries chosen w/ knowledge of $f$ ← adversary picks $\mathbf{v}\in\ker f$ → $f(\mathbf{x}_1+\mathbf{v})=f(\mathbf{x}_1)$.
- Tight: some $m$-point sets need $k=\Omega(\epsilon^{-2}\log m)$ for any map, even nonlinear (Larsen–Nelson, 2017; for $\epsilon$ not too small).
- Inner products: apply JL to $\{\mathbf{x}_i\pm\mathbf{x}_{i'}\}$ too → $|\langle f(\mathbf{x}_i),f(\mathbf{x}_{i'})\rangle-\langle\mathbf{x}_i,\mathbf{x}_{i'}\rangle|\le\epsilon$ for unit vectors.
```

```{tip} Derivation
:class: dropdown
*Why does $k\ge4\ln m/(\epsilon^2/2-\epsilon^3/3)$ suffice?* Constants from {cite:t}`dasgupta2003elementary` (random subspace); a Gaussian matrix gives the same tails.

1. Fix a pair; $\mathbf{v}=\mathbf{x}_i-\mathbf{x}_{i'}$, WLOG $\|\mathbf{v}\|=1$.
2. Rows of $R$ iid Gaussian → $(R\mathbf{v})_r\sim\mathcal{N}(0,1)$ iid → $Z=\|R\mathbf{v}\|^2\sim\chi^2_k$, with $r$ the row idx.
3. Chernoff on $\chi^2_k$, w/ $\ln(1+\epsilon)\le\epsilon-\epsilon^2/2+\epsilon^3/3$ & $\ln(1-\epsilon)\le-\epsilon-\epsilon^2/2$:

    $$
    P\left(Z\ge(1+\epsilon)k\right)\le e^{-\frac k2\left(\frac{\epsilon^2}{2}-\frac{\epsilon^3}{3}\right)},\qquad P\left(Z\le(1-\epsilon)k\right)\le e^{-\frac{k\epsilon^2}{4}}
    $$

4. Plug in $k$ → each tail $\le e^{-2\ln m}=1/m^2$ → pair fails w.p. $\le2/m^2$.
5. Union over $\binom m2$ pairs → $P(\text{any fail})\le\frac{m-1}{m}$ → success $\ge1/m$ per draw.
6. Check all pairs in poly time; redraw $O(m)$ times → constant success probability.
7. A larger $k$ shrinks the failure probability polynomially in $m$ → see [Random Projection](#random-projection).
```

```{note} Example
:class: dropdown
| $m$ | $\epsilon$ | $k\ge$ |
|:--|:--|:--|
| 100 | 0.5 | 222 |
| $10^6$ | 0.1 | 11,842 |

- Same $k$ for $d=10^4$ or $d=10^9$.
- Worst-case bound is loose: $m=10^6$, $\epsilon=0.1$ → no reduction for 768-dim embeddings. In practice choose $k$ empirically.
- `sklearn.random_projection.johnson_lindenstrauss_min_dim` uses this exact bound.
```

````{important} Code
:class: dropdown
```python
import numpy as np

def jl_min_dim(m, eps):
    ## Dasgupta-Gupta bound
    return int(np.ceil(4 * np.log(m) / (eps**2 / 2 - eps**3 / 3)))

class GaussianJL:
    def __init__(self, d, k, seed=0):
        rng = np.random.default_rng(seed)
        self.R = rng.standard_normal((k, d))  ## data-oblivious: drawn before seeing X
        self.k = k

    def __call__(self, X):
        ## X: (m, d) -> (m, k); 1/sqrt(k) -> E||f(x)||^2 = ||x||^2
        return X @ self.R.T / np.sqrt(self.k)

def sq_dists(Z):
    ## all pairwise squared distances from the Gram matrix
    G = Z @ Z.T
    g = np.diag(G)
    return g[:, None] + g[None, :] - 2 * G

def max_distortion(X, Y):
    ## worst |ratio of squared distances - 1| over all pairs
    i, j = np.triu_indices(len(X), k=1)
    return np.abs(sq_dists(Y)[i, j] / sq_dists(X)[i, j] - 1).max()

## Example
rng = np.random.default_rng(1)
m, d, eps = 100, 10_000, 0.5
X = rng.standard_normal((m, d))
k = jl_min_dim(m, eps)
Y = GaussianJL(d, k)(X)
print(k, max_distortion(X, Y))  ## 222, ~0.4 (< eps): 10,000 dims -> 222 dims
```
````

```{attention} Q&A
:class: dropdown
*JL vs PCA?*

- PCA: data-dependent; best rank-$k$ fit on average; can crush small-variance directions → some pairs distort badly; needs an SVD.
- JL: data-oblivious; worst-case guarantee for every pair; cost $O(dk)$ per point; $k$ fixed by $m$ & $\epsilon$, not by structure.
- Data w/ low [intrinsic dimension](data.md#intrinsic-dimension) → [PCA](../../ml/unsupervised.md#pca) wins w/ a much smaller $k$.

*Why $\log m$ & not $m$?*

- Failure probability per pair is exponentially small in $k\epsilon^2$.
- $m^2$ pairs → $k\epsilon^2\gtrsim\ln m^2$ suffices.

*Why is $k$ independent of $d$?*

- Rotation invariance of the Gaussian → each pair's distortion is distributed as $\chi^2_k/k$, whatever $d$ is.

*Does it work for infinite sets?*

- ❌ w/ a $\log m$ count.
- ✅ for structured sets: a fixed $s$-dim subspace needs $k=O(s/\epsilon^2)$; general sets → $k$ grows w/ their Gaussian width.
- $s$-sparse vectors → restricted isometry → compressed sensing.

*Other norms?*

- ❌ No comparable dimension reduction in $\ell_1$ (Brinkman–Charikar, 2003).
```

&nbsp;

### Random Projection
- **What**: Linear map $\mathbf{x}\mapsto\frac{1}{\sqrt k}R\mathbf{x}$ w/ a data-independent random $R$; sparse $\{\pm1,0\}$ entries suffice. {cite:p}`achlioptas2003database`
- **Why**: Dense Gaussian $R$ → $O(dk)$ float multiplies & storage.
    - Binary or sparse entries → adds & subtracts only → SQL-friendly; sparse → ~3× faster; same JL guarantee.
- **How**:
    - **Gaussian**: $R_{rj}\sim\mathcal{N}(0,1)$.
    - **Binary**: $R_{rj}=\pm1$, w.p. $1/2$ each.
    - **Sparse**: $R_{rj}=\sqrt3\cdot\{+1,0,-1\}$, w.p. $\{1/6,2/3,1/6\}$.
    - All: mean 0, variance 1 → $\mathbb{E}\|f(\mathbf{x})\|^2=\|\mathbf{x}\|^2$.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $R\in\mathbb{R}^{k\times d}$: Random matrix w/ iid entries, mean 0, variance 1.
- Hyperparams:
    - $\beta>0$: Confidence parameter.
- Misc:
    - $r$: Row idx of $R$.

Forward:

$$
f(\mathbf{x})=\frac{1}{\sqrt k}R\mathbf{x}
$$

- Achlioptas writes the row-vector form $\frac{1}{\sqrt k}AR$, w/ data matrix $A\in\mathbb{R}^{m\times d}$ & $R\in\mathbb{R}^{d\times k}$.

Theorem (binary or sparse entries):

$$
k\ge\frac{(4+2\beta)\ln m}{\epsilon^2/2-\epsilon^3/3}
$$

- → w.p. $\ge1-m^{-\beta}$, all pairs within $1\pm\epsilon$ on squared distances (same form as [JL](#johnson-lindenstrauss-lemma)).
- 2/3 is the max zero mass for which Achlioptas's moment lemma holds, at range size 3 & variance 1.

Variance of one projected squared norm:

$$
\mathrm{Var}\|f(\mathbf{x})\|^2=\frac1k\left(2\|\mathbf{x}\|^4+\left(\mathbb{E}R_{rj}^4-3\right)\sum_jx_j^4\right)
$$

- $\mathbb{E}R_{rj}^4$: 3 (Gaussian), 1 (binary), 3 (sparse) → binary & sparse are never worse than Gaussian.
```

```{tip} Derivation
:class: dropdown
*Why unbiased?*

1. $\mathbb{E}(R\mathbf{x})_r^2=\sum_{j,j'}x_jx_{j'}\mathbb{E}R_{rj}R_{rj'}=\sum_jx_j^2$ ← independent, mean 0, variance 1.
2. Sum over $k$ rows, divide by $k$ → $\mathbb{E}\|f(\mathbf{x})\|^2=\|\mathbf{x}\|^2$.

*Where does the variance formula come from?*

1. $W=(R\mathbf{x})_r=\sum_jR_{rj}x_j$.
2. Fourth moment: only terms w/ indices paired survive → $\mathbb{E}W^4=\mathbb{E}R^4\sum_jx_j^4+3\sum_{j\ne j'}x_j^2x_{j'}^2$.
3. $\sum_{j\ne j'}x_j^2x_{j'}^2=\|\mathbf{x}\|^4-\sum_jx_j^4$ → $\mathrm{Var}W^2=2\|\mathbf{x}\|^4+(\mathbb{E}R^4-3)\sum_jx_j^4$.
4. $\|f(\mathbf{x})\|^2$ = average of $k$ iid $W^2$ → divide by $k$.
5. Sparse: $R^4=9$ w.p. $1/3$ → $\mathbb{E}R^4=3$, equal to the Gaussian value.
```

````{important} Code
:class: dropdown
```python
import numpy as np
## Reuses jl_min_dim & max_distortion from Johnson-Lindenstrauss Lemma above

class AchlioptasProjection:
    def __init__(self, d, k, sparse=True, seed=0):
        rng = np.random.default_rng(seed)
        if sparse:
            ## 2/3 zeros -> only 1/3 of the input coordinates touched per output
            R = np.sqrt(3) * rng.choice([1.0, 0.0, -1.0], size=(k, d), p=[1/6, 2/3, 1/6])
        else:
            R = rng.choice([1.0, -1.0], size=(k, d))
        self.R, self.k = R, k

    def __call__(self, X):
        ## X: (m, d) -> (m, k); same scaling as the Gaussian map
        return X @ self.R.T / np.sqrt(self.k)

## Example
rng = np.random.default_rng(1)
m, d, eps = 100, 10_000, 0.5
X = rng.standard_normal((m, d))
proj = AchlioptasProjection(d, jl_min_dim(m, eps))
print((proj.R != 0).mean())               ## ~0.333
print(max_distortion(X, proj(X)))         ## ~0.4 (< eps), like the Gaussian map
```
````

```{dropdown} Table: Random Projection Variants
| Variant | Entries | Cost per point | Note |
|:--|:--|:--|:--|
| Gaussian | $\mathcal{N}(0,1)$ | $O(dk)$ multiplies | Simplest proofs |
| Binary | $\pm1$ | $O(dk)$ adds | No multiplies |
| Sparse (Achlioptas) | $\sqrt3\{\pm1,0\}$ | $O(dk/3)$ adds | Same bound as Gaussian |
| Very sparse (Li et al., 2006) | nonzero w.p. $1/\sqrt d$ | $O(\sqrt dk)$ | $\mathbb{E}R_{rj}^4=\sqrt d$ → higher variance on spiky vectors |
| Fast JL (Ailon–Chazelle) | Random signs + Hadamard + sparse | $O(d\log d)$ + sparse step | Spreads mass first, then samples |
```

```{attention} Q&A
:class: dropdown
*How does it relate to LSH?*

- SimHash (Charikar, 2002): keep only $\mathrm{sign}(R\mathbf{x})$ → $k$ bits.
- $P(\text{bit differs})=\theta/\pi$, $\theta$ = angle between inputs → Hamming distance estimates angle → approximate NN search in buckets.

*How does it relate to compressed sensing?*

- Normalize $A=R/\sqrt k$. Order-$2s$ RIP w/ constant $\delta$: $(1-\delta)\|\mathbf{z}\|^2\le\|A\mathbf{z}\|^2\le(1+\delta)\|\mathbf{z}\|^2$ for all $2s$-sparse $\mathbf{z}$.
- Holds w.h.p. if $k\gtrsim\delta^{-2}s\ln(ed/s)$.
- Proof: JL-type concentration + net argument + union bound (Baraniuk et al., 2008).
- $\delta_{2s}(A)<\sqrt2-1$ → $\ell_1$ minimization recovers every $s$-sparse $\mathbf{x}$ exactly from $A\mathbf{x}$ (Candès, 2008).
- RIP of order $s$ alone is not enough: the difference of two $s$-sparse vectors is $2s$-sparse.

*Where does it appear in deep learning?*

- Train only in a random subspace $\theta=\theta_0+P\mathbf{z}$ → measure the "intrinsic dimension" of a task (Li et al., 2018).
- Fine-tuning of pretrained LMs works in a small random subspace (Aghajanyan et al., 2020) → motivation for [LoRA](../../nlp/post/sft.md#lora).

*Do brains use random projection?*

- Fly olfaction: ~50-dim odor code → sparse binary random projection to ~40× more Kenyon cells → winner-take-all keeps the top ~5% → LSH-like similarity search (Dasgupta, Stevens & Navlakha, 2017).
- Expansion, not reduction: sparse high-dim codes separate similar odors.
- Interpretation by the authors; one circuit, not a general brain principle.

*Pros?*

- No training, data-oblivious, streaming-friendly, worst-case guarantee.

*Cons?*

- Ignores structure → needs a much larger $k$ than PCA on low-rank data.
- Dense $R$ has $kd$ entries → store a seed & regenerate, or use sparse/structured $R$.
```

&nbsp;

## Vector Symbolic Architectures

### Hyperdimensional Computing
- **What**: Compute w/ random ~10,000-dim vectors as symbols & 3 operations: bind, bundle, permute. {cite:p}`kanerva2009hyperdimensional`
- **Why**: Symbolic structure (roles, records, sequences) in one fixed-width distributed vector.
    - Information spreads over all components → errors degrade it gracefully.
    - Similarity is built in → nearest-neighbor lookup replaces exact matching.
- **How**:
    1. **Atoms**: Draw a random bipolar vector per symbol → near-orthogonal to all others ← [Near-Orthogonality](#near-orthogonality).
    2. **Bind** (⊙, XOR for 0/1): role ⊙ filler → dissimilar to both; self-inverse.
    3. **Bundle** (sum + majority): set of vectors → similar to every member.
    4. **Permute** (ρ): mark position or order → dissimilar to the input; invertible.
    5. **Query**: unbind → noisy vector → cleanup memory returns the nearest stored atom.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\mathbf{a},\mathbf{b},\mathbf{c}\in\{-1,+1\}^d$: Atoms, iid uniform entries.
- Misc:
    - $\odot$: Elementwise product.
    - $[\cdot]$: Elementwise sign, random tie-break.
    - $\rho$: Fixed permutation of the coordinates.
    - $K$: #vectors in a bundle.

Definitions:

$$
\text{bind: }\mathbf{a}\odot\mathbf{b},\qquad\text{bundle: }[\mathbf{a}+\mathbf{b}+\mathbf{c}],\qquad\text{permute: }\rho(\mathbf{a}),\qquad\mathrm{sim}(\mathbf{a},\mathbf{b})=\frac{\langle\mathbf{a},\mathbf{b}\rangle}{d}
$$

- Binary $\{0,1\}$ w/ XOR $\Leftrightarrow$ bipolar w/ $\times$. $\mathrm{sim}=1-2\cdot$Hamming$/d$.
- Sequence: $[\rho^2(\mathbf{a})+\rho(\mathbf{b})+\mathbf{c}]$ encodes $(\mathbf{a},\mathbf{b},\mathbf{c})$.

Properties:
- Self-inverse: $\mathbf{a}\odot\mathbf{a}=\mathbf{1}$ → $\mathbf{a}\odot(\mathbf{a}\odot\mathbf{b})=\mathbf{b}$ exactly.
- Distributive: $\mathbf{a}\odot(\mathbf{b}+\mathbf{c})=\mathbf{a}\odot\mathbf{b}+\mathbf{a}\odot\mathbf{c}$.
- Distance-preserving: $\mathrm{sim}(\mathbf{a}\odot\mathbf{x},\mathbf{a}\odot\mathbf{y})=\mathrm{sim}(\mathbf{x},\mathbf{y})$.
- Randomizing: $\mathrm{sim}(\mathbf{a}\odot\mathbf{b},\mathbf{a})\approx0$.
- Bundle of $K$ atoms, $K$ odd: $\mathbb{E}\,\mathrm{sim}(\text{bundle},\text{member})=\binom{K-1}{(K-1)/2}2^{-(K-1)}\approx\sqrt{2/(\pi K)}$.
- Noise floor: $\mathrm{sim}$ of unrelated vectors has std $1/\sqrt d$.
```

```{tip} Derivation
:class: dropdown
*Why is the bundle similar to each member?*

1. Fix member $\mathbf{a}$; the other $K-1$ atoms sum to $S_j$ at coordinate $j$. $K-1$ even → $S_j$ even.
2. $S_j=0$ → $[a_j+S_j]=a_j$ → agreement.
3. $|S_j|\ge2$ → $[a_j+S_j]=\mathrm{sign}(S_j)$, independent of $a_j$ → zero mean contribution.
4. → $\mathbb{E}\,\mathrm{sim}=P(S_j=0)=\binom{K-1}{(K-1)/2}2^{-(K-1)}$. $K=3$ → $0.5$.

*How many items fit in one bundle?*

1. Signal $\approx\sqrt{2/(\pi K)}$; noise per foreign atom $\approx1/\sqrt d$.
2. Retrieval among $N$ atoms needs signal $\gtrsim\sqrt{2\ln N/d}$ ← max of $N$ Gaussians.
3. → $K\lesssim d/(\pi\ln N)$: heuristic, linear in $d$.
```

```{note} Example
:class: dropdown
*What is the dollar of Mexico?* (Kanerva's example)

1. Atoms: roles $\mathbf{X}$ (country), $\mathbf{Y}$ (currency); fillers $\mathbf{U}$ (USA), $\mathbf{M}$ (Mexico), $\mathbf{D}$ (dollar), $\mathbf{P}$ (peso).
2. Records: $\mathbf{A}=[\mathbf{X}\odot\mathbf{U}+\mathbf{Y}\odot\mathbf{D}]$, $\mathbf{B}=[\mathbf{X}\odot\mathbf{M}+\mathbf{Y}\odot\mathbf{P}]$.
3. Role of dollar in USA: $\mathbf{D}\odot\mathbf{A}$ → noisy $\mathbf{Y}$, $\mathbb{E}\,\mathrm{sim}=1/2$.
4. Apply that role to Mexico: $\mathbf{q}=(\mathbf{D}\odot\mathbf{A})\odot\mathbf{B}$ → noisy peso code, $\mathbb{E}\,\mathrm{sim}(\mathbf{q},\mathbf{P})=1/4$ (binarized records, random tie-break, as in the Code).
    - Unbinarized records (sums w/o $[\cdot]$) → $\mathbf{q}=\mathbf{P}$ + 3 noise terms exactly ← distributivity & self-inverse.
5. Cleanup → peso. The variable $\mathbf{Y}$ is never queried explicitly → analogy by algebra.
```

````{important} Code
:class: dropdown
```python
import numpy as np

class HDC:
    def __init__(self, d=10_000, seed=0):
        self.d, self.rng = d, np.random.default_rng(seed)
        self.memory = {}  ## item memory = cleanup memory

    def atom(self, name):
        ## random bipolar vector -> near-orthogonal to every other atom
        self.memory[name] = self.rng.choice([-1, 1], size=self.d)
        return self.memory[name]

    def bind(self, a, b):
        return a * b  ## bipolar product = XOR; self-inverse: a * a = 1

    def bundle(self, *vs):
        s = np.sum(vs, axis=0)
        s = np.where(s == 0, self.rng.choice([-1, 1], size=self.d), s)  ## random tie-break
        return np.sign(s)  ## majority -> similar to every input

    def permute(self, a, n=1):
        return np.roll(a, n)  ## cyclic shift: invertible, output ~orthogonal to a

    def sim(self, a, b):
        return a @ b / self.d  ## cosine for bipolar vectors

    def cleanup(self, q):
        return max(self.memory, key=lambda name: self.sim(q, self.memory[name]))

## Example: "What is the dollar of Mexico?"
hd = HDC()
X, Y, U, D, M, P = (hd.atom(s) for s in ["country", "currency", "USA", "dollar", "Mexico", "peso"])
usa = hd.bundle(hd.bind(X, U), hd.bind(Y, D))
mex = hd.bundle(hd.bind(X, M), hd.bind(Y, P))
q = hd.bind(hd.bind(D, usa), mex)      ## D*usa ~ Y -> unbinds Mexico's currency
print(hd.cleanup(q), hd.sim(q, P))     ## peso ~0.25; every other atom ~0 +- 0.01
print(hd.sim(hd.permute(U), U))        ## ~0: permutation gives a new, dissimilar vector
```
````

```{dropdown} Table: VSA Families
| Family | Atoms | Bind | Unbind | Bundle |
|:--|:--|:--|:--|:--|
| BSC (Kanerva) | Binary $\{0,1\}^d$ | XOR | XOR (exact) | Majority |
| MAP (Gayler) | Bipolar / real | Elementwise × | × (exact for bipolar) | Sum (+ sign) |
| [HRR](#hrr) (Plate) | Real, $\mathcal{N}(0,1/d)$ | Circular convolution | Circular correlation (approx) | Sum |
| FHRR (Plate) | Complex unit phasors | Elementwise × (phase add) | × conjugate (exact) | Sum (+ normalize) |
| Tensor product (Smolensky) | Real | Outer product → $d^2$ | Contraction | Sum |

- BSC: Binary Spatter Code. MAP: Multiply-Add-Permute. FHRR: Fourier HRR.
```

```{attention} Q&A
:class: dropdown
*Why ~10,000 dims?*

- Random pair of 10,000-bit vectors → Hamming distance $5000\pm50$ (binomial).
- Less than a millionth of the space lies closer than $0.476d$ → any random atom is "unrelated" to all others w/ near certainty.
- Not a magic number: choose $d$ for the needed #atoms & bundle size.

*Bind vs bundle?*

- Bind: product → dissimilar to inputs → structure (role-filler pairs).
- Bundle: sum → similar to inputs → sets.

*Why is a cleanup memory needed?*

- Unbinding from a bundle returns filler + crosstalk noise.
- Snap to the nearest stored atom → exact symbol back → next step starts noise-free.

*How does it relate to superposition in LLMs?*

- Both store many items in one vector ← near-orthogonality → [Superposition](repr.md#superposition).
- VSA: random atoms, fixed hand-designed operations. LLMs: learned features; how they bind variables is an open question.
- Linear features in LLMs → [Linear Representation Hypothesis](repr.md#linear-representation-hypothesis).

*What does it tell us about intelligence/AC?*

- Binding problem: connectionist codes must represent "red square + blue circle" w/o mixing attributes. Fodor & Pylyshyn argued NNs lack systematic composition.
- VSA answer: binding = vector operation → compositional structure in a distributed code → a proposed bridge between connectionist & symbolic AI.
- Hypothesis: brains compute w/ similar high-dim distributed codes (Kanerva). Plausible, not established.

*Pros?*

- One-shot: no gradient training. Robust to component noise. Bitwise ops → cheap hardware. Transparent algebra.

*Cons?*

- Capacity limits ← crosstalk. Random atoms carry no similarity unless designed or learned.
- Lags deep learning on perception tasks → usually paired w/ a learned encoder.
```

&nbsp;

#### HRR
- **Name**: Holographic Reduced Representations {cite:p}`plate1995holographic`
- **What**: VSA w/ real Gaussian atoms, binding by circular convolution & unbinding by circular correlation.
- **Why**: Bind real-valued distributed vectors w/o growth in dimension.
    - Tensor-product binding → $d^2$ dims; nested structure → dims explode.
    - Circular convolution = compressed tensor product → stays $d$.
- **How**:
    1. **Atoms**: iid $\mathcal{N}(0,1/d)$ entries → norm $\approx1$.
    2. **Bind**: $\mathbf{a}\circledast\mathbf{b}$ → product of spectra via FFT, $O(d\log d)$.
    3. **Unbind**: convolve w/ the involution $\mathbf{a}^*$ = circular correlation → $\approx\mathbf{b}$ + noise.
    4. **Cleanup**: nearest atom in item memory.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\mathbf{a},\mathbf{b}\in\mathbb{R}^d$: Atoms, iid $\mathcal{N}(0,1/d)$ entries; indices $0..d-1$.
- Misc:
    - $\circledast$: Circular convolution.
    - $\mathbf{a}^*$: Involution, $a^*_j=a_{(-j)\bmod d}$.
    - $\mathcal{F}$: Discrete Fourier transform.

Forward (bind):

$$
(\mathbf{a}\circledast\mathbf{b})_j=\sum_{l=0}^{d-1}a_lb_{(j-l)\bmod d}=\mathcal{F}^{-1}\left(\mathcal{F}\mathbf{a}\odot\mathcal{F}\mathbf{b}\right)_j
$$

- $l$: Summation idx.

Inverse (unbind):

$$
\mathbf{a}^*\circledast(\mathbf{a}\circledast\mathbf{b})=\mathcal{F}^{-1}\left(|\mathcal{F}\mathbf{a}|^2\odot\mathcal{F}\mathbf{b}\right)\approx\mathbf{b}
$$

- $\mathcal{F}\mathbf{a}^*=\overline{\mathcal{F}\mathbf{a}}$ for real $\mathbf{a}$ → unbind = circular correlation.
- Approximate ← $|\mathcal{F}\mathbf{a}|^2$ has mean 1 but fluctuates per frequency.
- Exact inverse $\mathcal{F}^{-1}(1/\mathcal{F}\mathbf{a})$: exists if no frequency is 0; unstable if some $|\mathcal{F}\mathbf{a}|$ is small.
- Unitary atoms ($|\mathcal{F}\mathbf{a}|=1$ at all frequencies) → $\mathbf{a}^*$ is the exact inverse.
- $\circledast$: commutative, associative, bilinear → distributes over $+$.
```

```{tip} Derivation
:class: dropdown
*How noisy is retrieval?*

1. Parseval: $\langle\mathbf{x},\mathbf{y}\rangle=\frac1d\sum_f\mathcal{F}\mathbf{x}_f\overline{\mathcal{F}\mathbf{y}_f}$, $f$ = frequency idx.
2. Atoms iid $\mathcal{N}(0,1/d)$ → for complex frequencies $|\mathcal{F}\mathbf{a}_f|^2\sim\mathrm{Exp}(1)$ → $\mathbb{E}|\mathcal{F}\mathbf{a}_f|^2=1$, $\mathbb{E}|\mathcal{F}\mathbf{a}_f|^4=2$.
3. Signal $\mathbf{s}=\mathbf{a}^*\circledast\mathbf{a}\circledast\mathbf{b}$: $\langle\mathbf{s},\mathbf{b}\rangle\approx\mathbb{E}|\mathcal{F}\mathbf{a}|^2|\mathcal{F}\mathbf{b}|^2=1$; $\|\mathbf{s}\|^2\approx\mathbb{E}|\mathcal{F}\mathbf{a}|^4|\mathcal{F}\mathbf{b}|^2=2$.
4. Trace w/ $K$ pairs → $K-1$ crosstalk terms, each norm$^2\approx1$, $\approx$ orthogonal to $\mathbf{b}$.
5. → $\cos(\text{retrieved},\mathbf{b})\approx1/\sqrt{K+1}$. $K=2$ → $\approx0.58$.
```

````{important} Code
:class: dropdown
```python
import numpy as np

class HRR:
    def __init__(self, d=1024, seed=0):
        self.d, self.rng = d, np.random.default_rng(seed)
        self.memory = {}  ## cleanup memory

    def atom(self, name):
        ## iid N(0, 1/d) entries -> norm ~1
        self.memory[name] = self.rng.normal(0, 1 / np.sqrt(self.d), self.d)
        return self.memory[name]

    def bind(self, a, b):
        ## circular convolution = product of spectra, O(d log d)
        return np.fft.irfft(np.fft.rfft(a) * np.fft.rfft(b), n=self.d)

    def unbind(self, a, t):
        ## circular correlation = convolution w/ involution a*; spectrum of a* = conj(spectrum of a)
        return np.fft.irfft(np.conj(np.fft.rfft(a)) * np.fft.rfft(t), n=self.d)

    def sim(self, a, b):
        return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))

    def cleanup(self, q):
        return max(self.memory, key=lambda name: self.sim(q, self.memory[name]))

## Example: frame {color: red, shape: square} in ONE d-dim vector
hrr = HRR()
color, shape, red, square, blue = (hrr.atom(s) for s in ["color", "shape", "red", "square", "blue"])
frame = hrr.bind(color, red) + hrr.bind(shape, square)
noisy = hrr.unbind(color, frame)
print(hrr.cleanup(noisy), hrr.sim(noisy, red))  ## red ~0.58 (= 1/sqrt(3)); other atoms ~0
```
````

```{attention} Q&A
:class: dropdown
*Why "holographic"?*

- Each component carries information about the whole structure, like an optical hologram.
- Convolution & correlation are the operations of holographic storage → the name.

*Why not plain tensor products?*

- Outer product binding is exact but needs $d^2$ dims; depth-$t$ nesting → $d^{t+1}$.
- HRR sums the wrapped anti-diagonals of the outer product → fixed $d$, at the cost of noise.

*HRR vs bipolar HDC?*

- HRR: real-valued, unbind approximate, bind $O(d\log d)$.
- Bipolar: unbind exact, bind $O(d)$.
- FHRR: HRR in the Fourier domain → elementwise phase ops, $O(d)$, exact inverse.

*What happens w/ nested structures?*

- Noise compounds at each unbinding level → clean up after every level.

*How does it relate to ML?*

- Knowledge-graph embeddings: HolE scores triples w/ circular correlation (Nickel, Rosasco & Poggio, 2016).
- [Linear attention](../../nlp/attn.md#linear-attention) state $\sum_t\mathbf{v}_t\mathbf{k}_t^\top$ = tensor-product key-value memory; read w/ a query → $\approx$ value if keys are near-orthogonal. HRR = its compressed, $d$-dim version.
```

&nbsp;
