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
# LLM System Design

## Problem
- User
- Task
    - Multimodal
    - Tool usage
- Inputs
- Outputs
    - Structured output
- Constraints
    - Budget
    - Compute
    - Latency
        - p95: 95% of requests are handled at least at this value.
    - Throughput
        - QPS (Queries Per Second)
        - TPS (Tokens Per Second)
    - Availability
        - Time fraction of system serving correctly.
    - Auth/Rate Limit
    - Privacy, Ethics & Compliance
- Metrics

## Workflow
- API Gateway
- Data Pipeline
    - Preprocessing
    - Metadata
    - Quality control: dedup, etc.
- Model Pool
    - Paid frontier vs Self-hosted open-weight
    - RAG vs FT
    - Router (small classifier)
- Prompt Pool
- Retriever, Vector DB, Reranker
- Tools/Functions
- Cache (Prompt, Embedding, Output)
- Guardrails & Error Handling
- Schema Validator
- Tracing

## Evaluation (before/after)
- Offline
    - Automatic
    - Human
- Online A/B
    - Task success rate, hallucination, refusal, unsafe, etc.
- Risks
    - Hallucination on OOD queries
    - Latency spikes
    - Prompt injection
    - Cost blow-up
    - Data leakage

## Deployment
- Unit tests w mock samples
- Red teaming for safety
- User feedback
