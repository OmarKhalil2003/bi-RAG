# Coding Agent Instructions

## Objective

Implement `SPEC.md` exactly enough to satisfy the assessment while minimizing unnecessary complexity.

## Operating principles

- Prefer standard Python over frameworks.
- Keep functions small and testable.
- Keep retrieval and generation separate.
- Keep evaluation deterministic where possible.
- Do not fabricate data, metrics, URLs, citations, or test results.
- Do not silently change the experimental design.
- Do not add a second retrieval improvement.
- Do not optimize for architectural sophistication.

## Experimental integrity

The experiment has two retrieval systems:

### Baseline

`BGE-M3 -> FAISS -> top-k`

### Improved

`BGE-M3 -> FAISS top-20 -> BGE reranker -> top-k`

Everything else must remain constant.

Do not modify the embedding model, corpus, gold questions, final k, or generation configuration between these systems unless the change is explicitly documented and applies equally.

## Evaluation integrity

Retrieval metrics must be computed from retrieved chunk IDs, not from LLM judgments.

Never tune a threshold on the final evaluation set.

If a development split is needed, create it before final evaluation and document it.

## Definition of done

Do not finish until the acceptance checklist in `SPEC.md` is satisfied.

The final README and report must contain actual results produced by the implementation.
