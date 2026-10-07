# Bilingual Document Q&A (RAG) — Assessment Spec Kit

## 0. Mission

Build a small, reproducible, production-shaped **bilingual Arabic/English Document Q&A RAG system** that satisfies the assessment exactly.

The priority order is:

1. Correctness and measured quality
2. Reproducibility
3. Clear engineering decisions
4. Simple architecture
5. Clean documentation

Do NOT build a platform. Do NOT add unnecessary agents, microservices, orchestration frameworks, UI frameworks, Kubernetes, queues, authentication, streaming, or cloud infrastructure.

---

# 1. Required Outcome

The repository must demonstrate:

- At least **20 public documents**, containing both Arabic and English.
- Ingestion and chunking with a documented rationale.
- A **baseline** multilingual dense retriever.
- Exactly **ONE retrieval improvement**.
- Generation in the language of the question.
- Passage citations.
- Refusal when the answer is not supported by the corpus.
- An open-source/local LLM OR genuinely free API tier.
- A gold evaluation set with **>=40 questions**.
- Arabic + English questions.
- **>=8 unanswerable questions**.
- Retrieval Hit@k and MRR for baseline vs improved retrieval.
- Answer correctness.
- Unsupported-answer rate.
- Analysis of **10 failures**.
- FastAPI endpoint with validation, logging, and basic tests.
- Docker packaging.
- README with reproducible setup.
- A concise **2-page report**.

---

# 2. Recommended Minimal Architecture

```text
Public Arabic/English Documents
            |
            v
     Ingestion / Parsing
            |
            v
      Chunking + Metadata
            |
            +------------------------------+
            |                              |
            v                              v
   Dense Embeddings                 Document Metadata
   multilingual model
            |
            v
       Vector DB
            |
            v
       BASELINE RETRIEVER
            |
            |         +-----------------------------+
            |         |                              |
            |         v                              |
            |   ONE IMPROVEMENT: RERANKING           |
            |   (cross-encoder)                      |
            |         |                              |
            +---------+------------------------------+
                      |
                      v
                Top-k Context
                      |
                      v
             Local/Open LLM
                      |
                      v
      Answer + Source Passage Citations
      or explicit "not found" refusal
                      |
                      v
                  FastAPI
```

## Fixed retrieval improvement

Use **reranking** as the single improvement.

Why:

- Easy to explain.
- Easy to measure against a dense baseline.
- Does not require maintaining a second retrieval infrastructure.
- Usually improves precision for semantically similar but not exactly relevant chunks.
- Keeps the architecture simple.

Do NOT combine reranking with hybrid search or query rewriting. That would violate the "exactly one improvement" requirement.

---

# 3. Recommended Technology Stack

Use Python.

### Core

- Python 3.11
- FastAPI
- Pydantic
- Uvicorn
- Docker

### Document ingestion

Prefer:

- `pymupdf` for PDF text extraction
- `python-docx` only if DOCX documents are included
- plain text/HTML handling only if needed

Do not build a universal document parser.

### Chunking

Recommended initial configuration:

- chunk size: **500 tokens**
- overlap: **75 tokens**
- preserve paragraph boundaries where practical
- attach metadata:
  - `document_id`
  - `title`
  - `language`
  - `source_url`
  - `page`
  - `chunk_id`

If token-based chunking is inconvenient, use a character approximation and document the approximation. Prefer token-aware splitting.

### Embeddings

Recommended:

`BAAI/bge-m3`

Reason:

- multilingual
- strong retrieval model
- supports Arabic and English
- open model
- local inference
- no API dependency

Use normalized embeddings and cosine similarity.

### Vector database

Use **FAISS**.

Reason:

- lightweight
- local
- reproducible
- sufficient for a 20+ document assessment corpus
- no unnecessary database server

Persist:

```text
data/index.faiss
data/chunks.jsonl
```

### Reranker

Recommended:

`BAAI/bge-reranker-v2-m3`

Use it only after baseline retrieval.

Example:

```text
query
  -> dense retrieve top 20
  -> rerank those 20
  -> return top 5
```

The baseline should retrieve top 5 directly from dense search.

For fair evaluation, compare:

```text
Baseline:
embedding -> FAISS -> top k

Improved:
embedding -> FAISS top 20 -> reranker -> top k
```

Do not change the embedding model between the two systems.

### Generation

Recommended default:

`Qwen/Qwen2.5-7B-Instruct`

Use a local inference backend that is practical on the available machine.

If hardware is insufficient, use a documented free API tier instead.

The generator is NOT part of the retrieval comparison. Keep generation settings identical for both retrieval systems.

---

# 4. Repository Structure

Use this exact simple structure unless a concrete need requires a small deviation:

```text
bilingual-rag/
├── app/
│   ├── main.py
│   ├── schemas.py
│   ├── rag.py
│   ├── retrieval.py
│   ├── generation.py
│   └── logging_config.py
│
├── src/
│   ├── ingest.py
│   ├── chunk.py
│   ├── embed.py
│   ├── index.py
│   ├── retrieve.py
│   ├── rerank.py
│   └── evaluate.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── index.faiss
│   └── chunks.jsonl
│
├── evaluation/
│   ├── gold.jsonl
│   ├── evaluate_retrieval.py
│   ├── evaluate_answers.py
│   └── failure_analysis.md
│
├── tests/
│   ├── test_api.py
│   ├── test_retrieval.py
│   └── test_chunking.py
│
├── report/
│   └── report.md
│
├── scripts/
│   ├── download_corpus.py
│   ├── build_index.py
│   └── run_evaluation.py
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── README.md
└── SPEC.md
```

Do not create additional services unless required.

---

# 5. Corpus

## Minimum

At least **20 public documents**.

Use a meaningful bilingual corpus, not 20 copies of the same type of document.

Recommended sources:

- government reports
- international organization reports
- public policy documents
- public manuals
- public institutional reports

Prefer documents with stable public URLs.

Target:

- 10+ Arabic documents
- 10+ English documents

The corpus does not need to contain translations of the same documents.

## Corpus manifest

Create:

```text
data/corpus_manifest.csv
```

Columns:

```text
document_id
title
language
source_url
local_path
document_type
```

Every document must have a stable source URL.

## Reproducibility rule

The repository must contain a script that either:

1. downloads the documents automatically, or
2. clearly documents exactly where each document can be downloaded.

If licensing permits, small public documents may be included directly.

Do not commit copyrighted material unnecessarily.

---

# 6. Ingestion

Input:

```text
data/raw/*
```

Output:

```text
data/processed/
```

For every document:

1. identify document metadata
2. extract text
3. preserve page number when available
4. detect/record language
5. remove obvious extraction noise
6. chunk the text
7. write chunks as JSONL

Example chunk:

```json
{
  "chunk_id": "doc_003_p12_c02",
  "document_id": "doc_003",
  "title": "Example Report",
  "language": "ar",
  "page": 12,
  "text": "...",
  "source_url": "https://..."
}
```

Do not perform aggressive normalization that destroys Arabic meaning.

Avoid:

- removing Arabic diacritics unless justified
- transliterating Arabic
- translating all documents into one language
- stemming unless needed

---

# 7. Chunking Decision

Default:

- 500 tokens
- 75 token overlap
- paragraph-aware splitting
- no chunk smaller than necessary
- retain page boundaries when possible

## Why

The assessment needs enough context for policy/report questions while avoiding very large passages that dilute retrieval precision.

Overlap reduces boundary failures where the answer spans two adjacent chunks.

Paragraph-aware splitting preserves semantic units better than blind fixed-length slicing.

## Required experiment

Do not run a giant hyperparameter sweep.

At minimum, document why:

```text
500-token chunks + 75-token overlap
```

was selected.

Optionally test one nearby configuration such as 350/50 if time allows, but this is NOT required for the assessment.

---

# 8. Baseline Retrieval

Implement:

```python
query
 -> bge-m3 embedding
 -> FAISS cosine similarity
 -> top-k chunks
```

Recommended:

```text
candidate_k = 5
```

For evaluation, calculate at least:

- Hit@1
- Hit@3
- Hit@5
- MRR

A retrieved chunk is a hit if its `chunk_id` belongs to the gold supporting chunk set for the question.

## Important

Do not evaluate retrieval using generated answers.

Retrieval evaluation must directly inspect retrieved chunk IDs.

---

# 9. Exactly One Improvement: Reranking

Pipeline:

```text
query
 -> bge-m3
 -> FAISS top 20
 -> bge-reranker-v2-m3
 -> top 5
```

This is the ONLY retrieval improvement.

Do not additionally:

- use BM25
- rewrite the query
- translate the query
- use multiple embedding models
- use reciprocal rank fusion
- use HyDE
- add a second reranker

The report must explicitly state:

> The improved system differs from the baseline only by adding a multilingual cross-encoder reranking stage.

This makes the comparison scientifically interpretable.

---

# 10. Gold Evaluation Set

Create:

```text
evaluation/gold.jsonl
```

Minimum: **40 questions**.

Recommended target:

- 16 Arabic answerable
- 16 English answerable
- 4 Arabic unanswerable
- 4 English unanswerable

Total = 40.

Better if time permits:

- 20 Arabic
- 20 English
- 10+ unanswerable

But 40 with 8+ unanswerable is the hard requirement.

## Gold schema

```json
{
  "id": "q001",
  "language": "en",
  "question": "What is ...?",
  "answerable": true,
  "gold_answer": "....",
  "supporting_chunk_ids": [
    "doc_003_p12_c02"
  ]
}
```

For unanswerable:

```json
{
  "id": "q039",
  "language": "ar",
  "question": "...",
  "answerable": false,
  "gold_answer": null,
  "supporting_chunk_ids": []
}
```

## Question design

Avoid trivial questions only.

Include:

- factual lookup
- definitions
- policy requirements
- numbers
- dates
- comparisons within a document
- multi-sentence answers
- Arabic questions over Arabic documents
- English questions over English documents
- cross-lingual retrieval where useful

At least a few questions should require retrieval from a document whose language differs from the question language, if the corpus supports this.

Do NOT make all questions easy keyword matches.

---

# 11. Generation Contract

The LLM receives:

```text
SYSTEM:
You answer questions only from the provided document passages.

Rules:
1. Answer in the language of the user's question.
2. Use only the provided passages.
3. If the passages do not contain enough information, say that the answer is not available in the provided documents.
4. Never invent facts.
5. Cite every substantive claim using [source: chunk_id].
6. Do not cite a source that does not support the claim.

CONTEXT:
[chunk_id=...]
...

QUESTION:
...
```

Expected output:

```text
The required retention period is 5 years. [source: doc_003_p12_c02]
```

Unanswerable:

```text
The answer is not available in the provided documents.
```

Arabic unanswerable:

```text
الإجابة غير متوفرة في المستندات المقدمة.
```

The exact wording may vary, but the semantic behavior must be consistent.

---

# 12. Refusal / Unsupported Answer Control

Do NOT rely only on the LLM prompt.

Implement a simple retrieval confidence gate.

For example:

```text
if top retrieval score < threshold:
    refuse
else:
    generate
```

The threshold must be determined on a development subset, not tuned on the final test set.

Alternative acceptable implementation:

- reranker score threshold
- combined retrieval confidence threshold

Do not build a classifier unless necessary.

## Critical requirement

The system must distinguish:

```text
"the answer is absent"
```

from:

```text
"the answer is present but difficult"
```

This is part of the unsupported-answer evaluation.

---

# 13. Answer Evaluation

Measure at least:

## Answer correctness

Use a small manually labeled evaluation set.

Recommended labels:

```text
2 = fully correct
1 = partially correct
0 = incorrect
```

Report:

```text
Correctness = fully correct / total answerable questions
```

Also report partial correctness separately if used.

## Unsupported-answer rate

For answerable questions:

```text
unsupported-answer rate =
number of answers containing unsupported factual claims
/
number of answerable questions
```

For unanswerable questions, separately measure:

```text
unanswerable refusal accuracy =
correct refusals / total unanswerable questions
```

Do not merge these metrics.

---

# 14. Retrieval Metrics

For every system:

```text
Baseline Dense
Improved Dense + Reranker
```

Report:

| System | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---:|---:|---:|---:|
| Baseline | | | | |
| Improved | | | | |

Definitions:

### Hit@k

A question counts as a hit if at least one gold supporting chunk appears in top-k.

```text
Hit@k = questions with >=1 relevant chunk in top-k / total questions
```

### MRR

For each question:

```text
1 / rank_of_first_relevant_chunk
```

If no relevant chunk is retrieved:

```text
0
```

Then average over all questions.

---

# 15. Failure Analysis

Analyze exactly **10 representative failures**.

Create:

```text
evaluation/failure_analysis.md
```

For each:

```text
Question:
Language:
Answerable:
Expected evidence:
What baseline retrieved:
What improved retrieval retrieved:
Generation behavior:
Failure category:
Root cause:
Next fix:
```

Use categories such as:

- retrieval miss
- wrong chunk
- chunk boundary problem
- Arabic morphology/tokenization issue
- cross-lingual mismatch
- reranker error
- insufficient context
- generation hallucination
- refusal threshold too aggressive
- ambiguous question
- OCR/PDF extraction problem

Do not invent failures. Use actual evaluation outputs.

---

# 16. FastAPI Service

Expose:

```http
POST /query
```

Request:

```json
{
  "question": "What is the policy requirement?"
}
```

Response:

```json
{
  "answer": "...",
  "sources": [
    {
      "chunk_id": "doc_003_p12_c02",
      "title": "Example Report",
      "page": 12,
      "source_url": "https://..."
    }
  ]
}
```

## Validation

Require:

- non-empty question
- reasonable max length, e.g. 2000 characters

Return HTTP 422 for invalid request bodies.

## Logging

Log:

- request ID
- timestamp
- question length
- retrieved chunk IDs
- retrieval scores
- whether refusal occurred
- latency
- error information

Do NOT log secrets.

Do not log full user questions by default unless explicitly justified.

---

# 17. Basic Tests

Minimum:

### `test_chunking.py`

Verify:

- chunks are non-empty
- metadata exists
- overlap/chunk constraints behave as expected

### `test_retrieval.py`

Verify:

- index loads
- query returns k results
- returned chunks have valid IDs

### `test_api.py`

Verify:

- valid query returns 200
- missing/empty question is rejected
- response has `answer` and `sources`

Do not build a huge test suite.

---

# 18. Docker

The service must run with:

```bash
docker build -t bilingual-rag .
docker run -p 8000:8000 bilingual-rag
```

Recommended Docker command:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

If local LLM inference requires GPU-specific setup, keep the API container architecture documented and provide a CPU-compatible fallback or free API mode.

Do not require Kubernetes.

Do not require Redis.

Do not require PostgreSQL.

Do not require a separate vector database server.

FAISS is enough.

---

# 19. Configuration

Use environment variables only for genuinely variable settings:

```text
MODEL_NAME
RERANKER_NAME
TOP_K
CANDIDATE_K
GENERATION_MODEL
LLM_API_KEY
```

Put examples in:

```text
.env.example
```

Never commit real API keys.

---

# 20. README Requirements

README must contain exactly these practical sections:

1. Project overview
2. Architecture
3. Dataset/corpus
4. Models
5. Chunking strategy
6. Baseline retrieval
7. Reranking improvement
8. Generation and refusal behavior
9. Evaluation methodology
10. Results
11. Failure analysis
12. API usage
13. Docker
14. Reproducibility
15. Limitations

Include commands.

Minimum reproducibility flow:

```bash
git clone <repo>
cd bilingual-rag

python -m venv .venv
source .venv/bin/activate   # Windows equivalent also documented

pip install -r requirements.txt

python scripts/download_corpus.py
python scripts/build_index.py
python scripts/run_evaluation.py

uvicorn app.main:app --reload
```

Docker:

```bash
docker build -t bilingual-rag .
docker run -p 8000:8000 bilingual-rag
```

---

# 21. Two-Page Report

The report should be concise.

Recommended structure:

## Page 1

### 1. Approach

Explain:

- bilingual corpus
- chunking
- multilingual embedding
- FAISS baseline
- reranking improvement
- local/open LLM

### 2. Architecture

Include one simple architecture diagram.

### 3. Design rationale

Explain why:

- BGE-M3
- FAISS
- 500/75 chunking
- reranking

## Page 2

### 4. Evaluation

Show:

- corpus size
- question count
- Arabic/English split
- unanswerable count
- Hit@1/3/5
- MRR
- correctness
- unsupported-answer rate
- refusal accuracy

### 5. Failure analysis

Summarize the 10 failures.

### 6. Next fixes

Prioritize the highest-impact fixes.

### 7. Limitations

Examples:

- small corpus
- limited gold set
- local LLM hardware constraints
- PDF extraction noise
- threshold sensitivity
- manual answer evaluation

Do not fill the report with generic RAG theory.

---

# 22. Acceptance Criteria

The implementation is DONE only when all are true:

## Corpus

- [ ] >=20 public documents
- [ ] Arabic + English
- [ ] manifest exists
- [ ] source URLs recorded
- [ ] ingestion reproducible

## Chunking

- [ ] chunking implemented
- [ ] chunk metadata preserved
- [ ] 500/75 default documented
- [ ] rationale documented

## Retrieval

- [ ] multilingual embedding model used
- [ ] FAISS index works
- [ ] baseline implemented
- [ ] exactly one improvement: reranking
- [ ] baseline and improved systems use identical candidate embeddings

## Generation

- [ ] open/local model or free API
- [ ] answer language follows question language
- [ ] source citations included
- [ ] unsupported questions refused
- [ ] refusal mechanism has a confidence gate

## Evaluation

- [ ] >=40 gold questions
- [ ] Arabic + English
- [ ] >=8 unanswerable
- [ ] Hit@1/3/5
- [ ] MRR
- [ ] correctness
- [ ] unsupported-answer rate
- [ ] unanswerable refusal accuracy
- [ ] exactly 10 failure analyses

## API

- [ ] POST /query
- [ ] Pydantic validation
- [ ] logging
- [ ] basic tests
- [ ] Docker build works

## Deliverables

- [ ] README
- [ ] reproducible commands
- [ ] 2-page report
- [ ] assumptions/limitations
- [ ] no secrets committed

---

# 23. Anti-Overengineering Rules

The coding agent MUST NOT add these unless the assessment literally requires them:

- LangChain
- LangGraph
- LlamaIndex
- Haystack
- Elasticsearch
- OpenSearch
- Weaviate
- Pinecone
- Chroma server
- PostgreSQL
- Redis
- Celery
- Airflow
- Kafka
- Kubernetes
- Terraform
- MLflow
- experiment tracking platforms
- frontend
- authentication
- user accounts
- agent frameworks
- multiple retrieval improvements
- query translation
- multi-agent workflows

Plain Python is preferred.

Use libraries where they remove real implementation work, not to make the project look more sophisticated.

---

# 24. Coding-Agent Rules

The coding agent should work in this order:

1. Inspect repository.
2. Create minimal project structure.
3. Implement corpus manifest/download.
4. Implement extraction and chunking.
5. Build embedding/index pipeline.
6. Implement baseline retrieval.
7. Implement reranking.
8. Create gold evaluation set.
9. Implement retrieval evaluation.
10. Implement generation and refusal.
11. Implement answer evaluation.
12. Implement FastAPI.
13. Add tests.
14. Add Docker.
15. Run end-to-end evaluation.
16. Write README and report from actual results.
17. Verify every acceptance criterion.

Never write fake metrics.

Never write fake evaluation results.

Never claim a test passed unless it was actually executed.

Never fabricate source URLs.

---

# 25. Definition of Done

The final repository must allow a reviewer to answer "yes" to all of these:

1. Can I reproduce the corpus/index?
2. Can I see exactly how chunks were created?
3. Can I run baseline retrieval?
4. Can I run improved retrieval?
5. Is reranking the ONLY retrieval improvement?
6. Can I see quantitative evidence that the improvement helped or did not help?
7. Can I inspect the gold questions?
8. Can I verify retrieval metrics?
9. Can I see how hallucinations/unsupported answers were measured?
10. Can I inspect 10 real failures?
11. Can I call the API?
12. Can I run the project in Docker?
13. Can I understand the architecture in under five minutes?

If any answer is "no", the implementation is not finished.

---

# 26. Expected Final Demo

The README should provide one English example and one Arabic example.

Example:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What are the main objectives of the policy?"}'
```

Arabic:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"ما هي الأهداف الرئيسية للسياسة؟"}'
```

Also demonstrate one unanswerable query:

```text
Question:
What is the population of Mars according to the documents?

Expected behavior:
Refusal / not found in the provided documents.
```

The actual demo question should be chosen based on the real corpus.

---

# 27. Reviewer-Facing Design Summary

Use this concise statement in the submission:

> We implement a bilingual Arabic/English RAG pipeline using BGE-M3 multilingual embeddings and FAISS for dense retrieval. The baseline retrieves the top-k chunks directly from the vector index. We evaluate exactly one improvement: multilingual cross-encoder reranking of the top-20 dense candidates. A local/open instruction-tuned LLM generates answers only from retrieved passages, returns citations to source chunks, and refuses when retrieval confidence or document evidence is insufficient. Retrieval is evaluated independently using Hit@k and MRR, while generation is evaluated for correctness, unsupported claims, and refusal accuracy. The service is exposed through FastAPI, tested, and packaged with Docker.

---

# 28. Assumptions and Limitations to State

Use only those that actually apply.

Likely limitations:

- 20–30 documents are enough for the assessment but not representative of enterprise-scale retrieval.
- Gold-set size limits statistical confidence.
- Manual answer correctness labeling introduces evaluator subjectivity.
- PDF extraction quality can affect retrieval.
- Arabic morphology and formatting can create retrieval edge cases.
- Local LLM quality depends on available hardware.
- A single reranker is not a complete retrieval optimization study.
- Retrieval metrics do not guarantee factual generation.
- Refusal thresholds require calibration.

Do not hide weaknesses. Explain them and identify the next improvement.

---

# 29. Final Quality Gate

Before submission, run:

```bash
python -m pytest -q
python scripts/build_index.py
python scripts/run_evaluation.py
```

Then manually verify:

```text
[ ] 20+ documents
[ ] 40+ questions
[ ] 8+ unanswerable
[ ] Arabic questions
[ ] English questions
[ ] baseline metrics
[ ] reranked metrics
[ ] correctness
[ ] unsupported-answer rate
[ ] refusal accuracy
[ ] 10 failures
[ ] API works
[ ] Docker builds
[ ] README commands work
[ ] report is approximately 2 pages
[ ] no secrets
[ ] no fabricated numbers
```
