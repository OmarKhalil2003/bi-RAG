# Evaluation Protocol

## Dataset

Use the fixed `evaluation/gold.jsonl`.

Do not modify questions or supporting chunk IDs after running the final comparison unless the change is documented as a new evaluation version.

## Retrieval

For every question, run:

### Baseline

```text
query -> BGE-M3 -> FAISS -> top-k
```

### Improved

```text
query -> BGE-M3 -> FAISS top-20 -> reranker -> top-k
```

Use the same gold chunk IDs.

## Hit@k

```python
hit_at_k = number_of_questions_with_gold_chunk_in_top_k / total_questions
```

Calculate for k:

- 1
- 3
- 5

## MRR

For each query:

```python
rr = 1 / first_relevant_rank
```

If there is no relevant result:

```python
rr = 0
```

Then:

```python
mrr = mean(rr)
```

## Answer correctness

For answerable questions:

```text
2 = fully correct
1 = partially correct
0 = incorrect
```

Primary correctness metric:

```text
fully_correct / answerable_questions
```

Report partial correctness separately.

## Unsupported-answer rate

Review generated answers against the retrieved evidence.

An answer is unsupported if it makes a factual claim that cannot be supported by the retrieved document passages.

```text
unsupported rate =
unsupported answerable responses /
total answerable responses
```

## Refusal accuracy

For unanswerable questions:

```text
correct refusal /
total unanswerable questions
```

## Failure analysis

Select exactly 10 failures that expose distinct or high-impact failure modes.

Do not select only easy-to-explain failures.

For every failure record:

- question
- language
- expected evidence
- baseline retrieval
- improved retrieval
- generated answer
- failure category
- root cause
- next fix

## Reporting

Required result table:

| System | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---:|---:|---:|---:|
| Dense baseline | | | | |
| Dense + reranker | | | | |

Required generation table:

| Metric | Result |
|---|---:|
| Answer correctness | |
| Partial correctness | |
| Unsupported-answer rate | |
| Unanswerable refusal accuracy | |

Never fill these cells with invented values.
