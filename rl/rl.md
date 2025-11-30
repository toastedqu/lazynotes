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
# RL
This page contains my study notes for [Sergey Levine's CS 285](https://rail.eecs.berkeley.edu/deeprlcourse/).

- **What**: Agent $\overset{\text{action}}{\underset{\text{reward}}{\rightleftarrows}}$ Environment
- **Why**: To learn decision-making where actions have delayed consequence in dynamic, sequential tasks, from experience.
    - Data w/o Optimization → No innovation to solve new problems
    - Optimization w/o Data → Hard to apply IRL

```{dropdown} Table: 3 Learning Types
|  | Supervised Learning | Unsupervised Learning | Reinforcement Learning |
|:--|------|-------------|------|
| Input | $\mathbf{x}$ | $\mathbf{x}$ | $\mathbf{s}$ |
| Output | $\mathbf{y}$ | (Optional: $\mathbf{z}$) | $\mathbf{a}$ |
| Data | $\mathcal{D}=\{(\mathbf{x}_i,\mathbf{y}_i)\}$<br>Static & Given | $\mathcal{D}=\{\mathbf{x}_i\}$<br>Static & Given | $[\mathbf{s}_1,\mathbf{a}_1,\cdots,\mathbf{s}_T,\mathbf{a}_T]$<br>Dynamic & DIY |
| Goal | Learn $f_\theta(\mathbf{x}_i)$ to minimize $\mathcal{L}(f_\theta(\mathbf{x}_i), \mathbf{y}_i)$ | Learn $f_\theta(\mathbf{x}_i)$ to capture structure in $\mathcal{D}$ | Learn $\pi_\theta: \mathbf{s}_t\rightarrow \mathbf{a}_t$ to maximize $\sum_t r_t$ |
```

```{dropdown} Notations
- $\mathbf{s}$: State.
- $\mathbf{a}$: Action.
- $\mathbf{o}$: Observation.
- $\mathcal{S}$: State space.
- $\mathcal{A}$: Action space.
- $\mathcal{O}$: Observation space.
- $\mathcal{T}$: Transition operator.
- $r(\mathbf{s},\mathbf{a}):\mathcal{S}\times\mathcal{A}\rightarrow\mathbb{R}$: Reward function.
- $p(\mathbf{s}'|\mathbf{s},\mathbf{a})$: Transition probability.
- $\mathcal{E}=p(\mathbf{o}_t|\mathbf{s}_t)$: Emission probability.
- $\pi_\theta(\mathbf{a}_t|\mathbf{o}_t)$: Policy.
- $\pi_\theta(\mathbf{a}_t|\mathbf{s}_t)$: Policy (Fully observed).
```

## MDP (Markov Decision Process)
- **What**: A decision-making process following the Markov Chain.
    - **Markov Chain**: A stochastic process where the probability of transitioning to the next state depends ONLY on the curr state.
- **Why**: Minimal structure of states, actions, transitions, and rewards.

<!-- ```{note} Math
:class: dropdown -->
MDP:

$$
\mathcal{M}=\{\mathcal{S},\mathcal{A},\mathcal{T},r\}
$$

Probability of Observing a full trajectory under Policy $\pi_\theta$:

$$\begin{align*}
p_\theta(\tau)&=p_\theta(s_1, a_1, \cdots, s_T, a_T) \\
&=p(s_1)\prod_{t=1}^T\pi_\theta(a_t|s_t)p(s_{t+1}|s_t,a_t)
\end{align*}$$

Objective of RL: **Cumulative Reward Maximization**

$$\begin{align*}
\theta^*&=\arg\max_\theta\text{E}_{\tau\sim p_\theta(\tau)}\left[\sum_t r(s_t,a_t)\right] && \\
&=\arg\max_\theta\sum_{t=1}^T\text{E}_{(s_t,a_t)\sim p_\theta(s_t,a_t)}[r(s_t,a_t)] && (\text{Finite Horizon}) \\
&=\text{E}_{(s,a)\sim p_\theta(s,a)}[r(s,a)] && (\text{Infinite Horizon})
\end{align*}$$

- Finite Horizon:

    $$
    \theta^*
    $$
    - $p_\theta(s_t,a_t)=p(s_{t+1}|s_t,a_t)\pi_\theta(a_{t+1}|s_{t+1})$: State-action marginal.

<!-- ``` -->

### POMDP (Partially Observed MDP)
- **What**: MDP but the state is not directly observable but provides partial info.
- **Why**: Uncertainty IRL.

```{note} Math
:class: dropdown
MDP:

$$
\mathcal{M}=\{\mathcal{S},\mathcal{A},\mathcal{O},\mathcal{T},\mathcal{E},r\}
$$
```