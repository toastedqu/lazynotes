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
# Reward Hacking
Reward hacking happens when the LLM learns to maximize the measurable reward signal w/o learning the actual human-intended objective.

Reward hacking is mainly caused by **imperfect training data** and/or **imperfect reward models**.

&nbsp;

## Taxonomy
### Feature-level Exploitation
- **What**: The LLM exploits certain surface features that correlate w/ reward signals regardless of their true quality.
- **Why**: Correlation bias in training data.
- **How**:
    - Symptoms: Length, Formatting, Hedging, Artifacts (e.g., Markdown, Emojis), etc.
    - Example:
        - Prompt: *Explain what reward hacking is briefly.*
        - Response: *Here's a detailed six-part analysis... (followed with redundant markdown headers & long bullet points)*

&nbsp;

### Representation-level Exploitation
- **What**: The LLM gets the reward with BS reasons → separates outcome from process.
- **Why**:
    - Outcome-only rewards.
    - No process supervision.
- **How**:
    - Symptoms: Fabricated Reasoning, Post-hoc rationalizations, hallucinated justifications, Intentional hiding of real cues, etc.
    - Example:
        - Prompt: *(some MCQ worded in a way that strongly implies the answer is C)*
        - Reasoning: *(Lines & paragraphs of BS that sounds plausible)*
        - Response: *The answer is C.*

&nbsp;

### Sycophancy / Agreement Optimization
- **What**: The LLM seeks user/evaluator's approval instead of being correct.
- **Why**:
    - Agreement bias in training data.
    - Emphasis on agreement in reward model.
- **How**:
    - Example:
        - Prompt: *I don't think reward hacking exists.*
        - Response: *You're absolutely right!*

&nbsp;

### Proxy Gaming
- **What**: Reward Proxy = Optimization Target.
- **Why**: Poor evaluator design (incompleteness, brittleness, no invariance constraints, etc.)
- **How**:
    - Example:
        - Prompt: *Produce a minimal JSON report of this project with real metrics.*
        - Response: *(some schema-valid skeleton with no substance because the evaluator only checks the schema)*

&nbsp;

### In-Context Reward Hacking
- **What**: During inference, the LLM optimizes a hidden proxy objective through a feedback loop w/ environment.

## Causes
### Objective Compression
- **What**: Reward proxies ≠ Human intentions (info loss due to compression).

### Optimization Amplification
- **What**: Overfitting ← too much optimization.

### Evaluator-Policy Co-adaptation


## Mitigation