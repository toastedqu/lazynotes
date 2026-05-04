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
RL learns from **NON-i.i.d.** data, specifically dynamic data where curr output affects next input.

Overview:
$$\text{Agent}
\overset{\text{action}}{\underset{\text{reward}}{\rightleftarrows}}
\text{Environment}$$

All RL algorithms follow this workflow:

```{mermaid}
%%{init: {'flowchart': {'nodeSpacing': 180, 'rankSpacing': 60}}}%%
flowchart LR
  A[Improve policy] --> B[Generate samples]
  B --> C[Estimate return]
  C --> A
```

```{dropdown} Table: Learning Types
:open:
|  | Supervised Learning | Unsupervised Learning | Reinforcement Learning |
|:--|------|-------------|------|
| Input | $\mathbf{x}$ | $\mathbf{x}$ | $\mathbf{s}$ |
| Output | $\mathbf{y}$ | (Optional: $\mathbf{z}$) | $\mathbf{a}$ |
| Data | $\mathcal{D}=\{(\mathbf{x}_i,\mathbf{y}_i)\}$<br>Static & Given | $\mathcal{D}=\{\mathbf{x}_i\}$<br>Static & Given | $(\mathbf{s}_1,\mathbf{a}_1,\cdots,\mathbf{s}_T,\mathbf{a}_T)$<br>Dynamic & DIY |
| Goal | $\arg\min_{f_\theta}\mathcal{L}(f_\theta(\mathbf{x}_i), \mathbf{y}_i)$ | $\arg\max_{f_\theta}P(\mathcal{D})$ | $\arg\max_{\pi_\theta}R(\tau)$ |
```

```{dropdown} Table: Notations
| Concept | Notation |
|------|-----|
| State | $s\in\mathcal{S}$ |
| Action | $a\in\mathcal{A}$ |
| Policy | $\pi(a_t\|s_t):\mathcal{S}\times\mathcal{A}\rightarrow[0,1]$ |
| Reward | $r(s,a):\mathcal{S}\times\mathcal{A}\rightarrow\mathbb{R}$ |
| Transition Probability | $p(s_{t+1}\|s_t,a_t)\in[0,1]$ |
| Discount Factor | $\gamma\in[0,1)$ |
| #Steps | $T\in[0,\infty)$ ($T=\infty$ if endless) |
| Trajectory | $\tau=(s_0,a_0,s_1,a_1,\dots)$ |
| Discounted Return of a trajectory | $R(\tau)=G_0=\sum_{t=0}^T\gamma^tr_t$ |
| Discounted Return at a time step | $G_t=R(\tau_{t:})=\sum_{k=0}^{T-t}\gamma^kr_{t+k}$ |
| State Value Function | $V^\pi(s)=E_\pi[G_t\|s_t=s]$ |
| Action Value Function | $Q^\pi(s,a)=E_\pi[G_t\|s_t=s,a_t=a]$ |
| Advantage | $A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)$ |
| Expected Discounted Return | $J(\pi)=E_{\tau\sim\pi}\left[R(\tau)\right]=E_{s_0\in\mathcal{S}}[V^\pi(s_0)]$ |
```

```{dropdown} Table: Intuition
| Concept | Intuition |
|---------|-----------|
| Policy | How likely the agent chooses a specific action given curr state. |
| Reward | The immediate feedback from the env given curr action & state. |
| Transition Probability | How likely the env moves to a specific state given curr action & state. |
| Discount Factor | How much the agent cares about future vs immediate rewards.<br>(1: yes; 0: no) |
| Discounted Return of a trajectory | How good a full trajectory is. |
| Discounted Return at a time step | How good the future trajectory is from a specific moment. |
| State Value Function | How good it is to be in a specific curr state, following curr policy. |
| Action Value Function | How good it is to take a specific action in a specific curr state, if following curr policy. |
| Advantage | How much a specific action is better/worse than expected. |
| Expected Discounted Return | The overall performance of a policy across all of its possible trajectories.<br>(i.e., **RL's main objective**) |
```

&nbsp;

## MDP (Markov Decision Process)
- **What**: Markov Chain decision-making process.
- **Why**: Minimal structure of states, actions, transitions, and rewards.

```{note} Math
:class: dropdown
MDP:
$$
\mathcal{M}=\{\mathcal{S},\mathcal{A},\mathcal{T},r\}
$$

MDP as Markov Chain:
$$
p(s_{t+1},a_{t+1}|s_t,a_t)=p(s_{t+1}|s_t,a_t)\pi_\theta(a_{t+1}|s_{t+1})
$$

Trajectory Distribution:
$$
p_\theta(\tau)=p(s_0)\prod_{t=0}^T\pi_\theta(a_t|s_t)p(s_{t+1}|s_t,a_t)
$$

Cumulative Reward Maximization:
$$
\theta^*&=\arg\max_\theta\text{E}_{\tau\sim p_\theta(\tau)}\left[\sum_t r(s_t,a_t)\right] && \\
&=\arg\max_\theta\sum_{t=0}^T\text{E}_{(s_t,a_t)\sim p_\theta(s_t,a_t)}[r(s_t,a_t)] && (\text{Finite Horizon}) \\
&=\arg\max_\theta\text{E}_{(s,a)\sim p_\theta(s,a)}[r(s,a)] && (\text{Infinite Horizon})
$$
```

## Algorithms
### Policy Gradients
- **What**

Reference:
- [Sergey Levine's CS 285](https://rail.eecs.berkeley.edu/deeprlcourse/)