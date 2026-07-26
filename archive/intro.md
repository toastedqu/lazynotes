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
# Overview
An AI project consists of 4 parts:

```{image} ../images/ml_system_design_flow.png
:align: center
```

## Problem
1. **Overview**: Define the problem, the goal, and the expected inputs & outputs.
2. **Scope**: Identify constraints.
    - **Data**:
        - **Size**: #samples, #features, etc.
        - **Features**: content, context, user, etc.
        - **Targets**: Explicit (direct), implicit (indirect).
    - **Model constraints**:
        - Priority: Performance / Quality
        - Type: Single general / Multiple specific
        - Interpretability
        - Retrainability
        - ...
    - **Resource constraints**:
        - Time: Training, inference, project duration, etc.
        - Computation: Training, inference, local/cloud, etc.
3. **Evaluation**: Define success measurement.
    - **Automatic metrics**:
        - Offline: MSE, P/R/F1, etc.
        - Online: Usage time, usage frequency, click rate, etc.
    - **Human metrics**: User interaction, recent reports, company intention for users, personalization, etc.

## Data
1. **Data Collection/Availability**
    - **Status**: Available/unavailable, quantity, etc.
    - **Annotation**: Quality, cost, resolving disagreements, feasibility of auto-annotation, etc.
    - **Privacy**: User data accessibility, methods, online/periodic data use, anonymity, etc.
    - **Logistics**: Storage location, structure, biases, etc.
2. **Data Processing**
3. **Feature Engineering**

## Modeling
For each model, specify:
- Why: Motivation
- What: Functionality
- How: Objective and optimization
- Pros & Cons

Procedure:
1. Baseline: Stats (mean, median, mode), random benchmarks, etc.
2. Easy model
3. Hard model
4. Experiment, evaluation & ablation study

## Production
Performance can degrade in production due to:
- **Data Drift**: Production data $\neq$ training data.
- **Feature Drift**: Changes in features or feature transformations.
- **Concept Drift**: Changes in the relationship between features & target variables, especially in a dynamic environment.
- **Data Quality**: Missing values, outliers, noise, etc.
- **Model Versioning**: R&D models $\neq$ deployed models.
- **Scaling & Latency**: Handling large data volumes and fast response times.
- **Ethics**: Adversarial attacks, privacy concerns, regulatory compliance, interpretability, etc.
- **Others**: Random errors (e.g., network issues).

Consider these factors for production:
1. **Inference location**:
    - **Local**: High memory/storage usage, low latency.
    - **Server**: Low memory/storage usage, high latency, privacy concerns.
2. **Feature serving**:
    - **Batch**: Handled offline, served online with periodic data generation/collection.
    - **Real-time**: Handled & served online at request time, prioritize scalability & latency, use feature stores and caching.
3. **Performance Monitoring**: Errors, latency, biases, data drift, CPU load, memory usage, retrain frequency, etc.
