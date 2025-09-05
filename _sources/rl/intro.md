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
# Reinforcement Learning
Reinforcement Learning learns through **experience** to make good decisions under uncertainty.
- RL directly interacts with the environment and uses training info that **evaluates** the actions taken.
	- Evaluative feedback depends only on the action taken.
- Supervised Learning uses training info that **instructs** by giving correct actions instead.
	- Instructive feedback is independent of the action taken.

The core of RL is **STATE** - the perceived signals from the environment at each time step. Based on the concept of state, RL consists of 4 elements:
- **Policy**: perceived states -> actions.
- **Reward**: short-term goal, at each step (primary).
- **Value**: long-term goal, from the expected future (secondary).
- **Model**: simulation of environment behavior (optional).
	- **Model-based**: planning
	- **Model-free**: trial-and-error