# Bilingual Document Q&A (RAG) System

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com/)
[![FAISS](https://img.shields.io/badge/FAISS-CPU%201.15-orange.svg)](https://github.com/facebookresearch/faiss)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A clean, reproducible **Bilingual (Arabic & English) Document Question Answering** system. It uses **BGE-M3** multilingual embeddings, **FAISS** vector search, **BM25Okapi** keyword retrieval fused via **Reciprocal Rank Fusion (RRF)**, and a lightweight LLM generation layer with citation and refusal guardrails.

---

## 1. Quick Highlights

- **Bilingual**: Answers in the exact language of the user's question (Arabic or English).
- **Faithful Citations**: Every factual statement includes its source chunk ID `[source: chunk_id]`.
- **Refusal Guardrails**: Out-of-corpus or unanswerable questions are safely refused without hallucination.
- **Measured Quality**: Evaluated on a 40-question gold benchmark (Arabic, English, and unanswerable).
- **Simple Architecture**: Plain Python without bloated frameworks like LangChain or LlamaIndex.

---

## 2. System Architecture

```text
User Question (Arabic / English)
             │
             ├──► Dense Search (BGE-M3 + FAISS IndexFlatIP)
             └──► Sparse Search (Multilingual BM25Okapi)
                            │
                            ▼
               Reciprocal Rank Fusion (RRF, k=60)
                            │
                            ▼
                 Confidence Check Gate
                (Dense < 0.50 & BM25 < 15)
                 /                      \
        (Low Confidence)         (Sufficient Evidence)
               │                                │
               ▼                                ▼
        Direct Refusal               LLM Context Assembly
   "Answer not available..."           (OpenRouter / Local LLM)
                                                │
                                                ▼
                                     Answer with Citations
```

---

## 3. Corpus & Chunking

- **Corpus**: 24 public documents (12 Arabic, 12 English) covering government policies, technical manuals, and environmental/water reports.
- **Chunk Size**: 500 tokens with 75-token sliding overlap.
- **Paragraph Awareness**: Splits at natural paragraph breaks rather than hard character cuts to keep clauses intact.
- **Context Header**: Prepends `[Document: {title} | Page: {page}]` to every chunk for precise document grounding.

---

## 4. Retrieval Pipeline

### Baseline (Dense Vector Search)
- **Embedding Model**: `BAAI/bge-m3` (1024-dimensional normalized vectors).
- **Vector Index**: Local FAISS `IndexFlatIP` computing exact cosine similarity.
- **Preprocessing**: Cleans Arabic diacritics (Tashkeel), Tatweel, and normalizes Arabic-Indic numerals (`٠-٩` → `0-9`).

### Enhanced Version (Hybrid Search)
- **Single Improvement**: Combines dense semantic search with sparse lexical search using **BM25Okapi**.
- **Reciprocal Rank Fusion (RRF)**:
  $$\text{RRF}(d) = \frac{1}{60 + \text{rank}_{\text{dense}}(d)} + \frac{1}{60 + \text{rank}_{\text{bm25}}(d)}$$
- **Why Hybrid**: Dense vectors capture conceptual meaning across languages; BM25 ensures exact matches for specific codes and numbers (e.g., `ECC-1:2018`, `NIST SP 800-53`, `PM2.5`).

---

## 5. Generation & Refusal

- **Language Matching**: Automatically detects the query language and replies in that same language.
- **Dual-Gate Refusal**:
  1. *Retrieval Gate*: If both dense score (< 0.50) and BM25 score (< 15) are low, refuses immediately without calling the LLM.
  2. *Model Gate*: The LLM is instructed to answer solely from context passages and cite source chunk IDs.
- **Clean Output**: Automatically removes reasoning traces (`<think>` tags) and includes an extractive fallback in case of upstream network issues.

---

## 6. Evaluation Results

Evaluated on a 40-question gold benchmark (`evaluation/gold.jsonl`) with 32 answerable questions (16 Arabic, 16 English) and 8 adversarial unanswerable questions.

### Retrieval Performance

| System | Hit@1 | Hit@3 | Hit@5 | MRR |
| :--- | :---: | :---: | :---: | :---: |
| **Dense Baseline (BGE-M3 + FAISS)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **Enhanced Hybrid (Dense + BM25 RRF)** | **0.9688** | **0.9688** | **0.9688** | **0.9688** |

*Note on Cross-Lingual Search*: In question `q_en_16`, an English prompt asked about an Arabic cloud policy. Dense search retrieved the Arabic passage at Rank 1. BM25 had 0 keyword overlap across languages, favoring unrelated English documents. This highlights the trade-off of unweighted hybrid search on cross-lingual queries.

### Generation Quality

| Metric | Score | Evaluated Queries | Meaning |
| :--- | :---: | :--- | :--- |
| **Answer Correctness** | **68.75% (22/32)** | Answerable queries | Fully accurate answers with exact facts and citations |
| **Partial Correctness** | **18.75% (6/32)** | Answerable queries | Core answer is accurate with minor secondary clause omission |
| **Combined Correctness** | **87.50% (28/32)** | Answerable queries | Factually grounded answers |
| **Unsupported-Answer Rate** | **3.12% (1/32)** | Answerable queries | Claims or numbers without evidence in context (only 1 query) |
| **Refusal Accuracy** | **100.0% (8/8)** | Unanswerable queries | Correctly refused out-of-corpus questions |

Hybrid search boosted full answer correctness to **68.75%** because BM25 precisely provided the exact regulatory codes and numbers directly into the context window.

---

## 7. Failure Analysis (10 Representative Cases)

Here is a summary of the 10 failure cases observed during evaluation:

| # | Question ID & Topic | What Happened | Root Cause | Proposed Fix |
| :-: | :--- | :--- | :--- | :--- |
| **1** | `q_ar_02` (Cybersecurity MFA) | Erroneously refused | High dense score (0.719), but low BM25 score triggered a false refusal | Allow refusal bypass when dense confidence is high ($\ge 0.60$) |
| **2** | `q_ar_03` (AI Ethics) | Scored partial | LLM numbered its list (`1.`, `2.`, `3.`); evaluator flagged digits as ungrounded numbers | Strip list item numbers before checking unevidenced digits |
| **3** | `q_ar_05` (PM2.5 & Buses) | Omitted bus target | PM2.5 was answered, but the electric bus target from page 2 was missed | Aggregate adjacent page chunks for compound questions |
| **4** | `q_ar_10` (Water Strategy) | Omitted leak target | Desalination targets were answered; leakage reduction timeline was omitted | Split multi-part questions into sub-queries |
| **5** | `q_ar_15` (Pipe Repair Time) | Omitted repair hours | Urban water coverage was answered; 2-hour repair timeframe was missed | Add a prompt checklist verifying all query parts are answered |
| **6** | `q_en_04` (OECD AI Principles) | Omitted trigger | All 5 principles were explained, but "high-risk" classification trigger was missed | Enforce concise bullet responses without conversational filler |
| **7** | `q_en_08` (Cloud HSM / BYOK) | Omitted rotation cycle | Cited FIPS 140-2 Level 3 HSMs correctly, but skipped the annual key rotation cycle | Instruct model to extract operational lifecycles alongside ratings |
| **8** | `q_en_09` (GDPR Fine Phrasing) | Scored partial | Mandatory 72h notification was cited; fine clause phrasing differed slightly from gold | Allow flexible phrasing for complex regulatory clauses |
| **9** | `q_en_14` (CISA Incident Playbook) | Scored partial | SHA-256 hashing was correct; minor forensic synonyms scored partial | Add standard technical synonyms to the gold reference |
| **10** | `q_en_16` (Cross-Lingual Cloud) | Erroneously refused | English query over Arabic text had 0 BM25 overlap, causing BM25 to boost irrelevant English docs | Turn off BM25 when query language differs from target document language |

### Key Takeaways & Prioritized Fixes
1. **Dynamic BM25 Weighting**: Set BM25 weight to 0 on cross-lingual queries so multilingual dense vectors handle language bridging.
2. **Sub-Query Splitting**: Decompose compound questions (e.g. "What is X and what is Y?") into separate retrieval steps.
3. **Local Quantized LLM**: Package an offline model (e.g. Qwen2.5-3B) to avoid external API rate limits.

---

## 8. Project Structure

```text
├── app/
│   ├── main.py              # FastAPI application & endpoints (/query, /health)
│   ├── schemas.py           # Pydantic request/response validation schemas
│   ├── rag.py               # RAG pipeline coordinator
│   ├── retrieval.py         # Hybrid search service wrapper
│   ├── generation.py        # LLM prompt, citations, and dual refusal gates
│   └── logging_config.py    # Structured JSON logger
├── src/
│   ├── chunk.py             # 500-token paragraph-aware text chunker
│   ├── embed.py             # BGE-M3 dense embeddings with L2 normalization
│   ├── evaluate.py          # Metric calculations (Hit@k, MRR)
│   ├── hybrid.py            # BM25Okapi store & Reciprocal Rank Fusion (RRF)
│   ├── index.py             # FAISS IndexFlatIP vector store
│   ├── ingest.py            # PDF text extraction and NFKC normalization
│   ├── preprocess.py        # Canonical Arabic & English query preprocessing
│   └── retrieve.py          # Baseline dense FAISS retriever
├── data/
│   ├── corpus_manifest.csv  # 24 documents metadata registry
│   ├── raw/                 # 24 raw PDFs (12 Arabic, 12 English)
│   ├── processed/           # 24 extracted JSON page texts
│   ├── chunks.jsonl         # 72 token-bounded contextual chunks
│   └── index.faiss          # Serialized FAISS IndexFlatIP vector index
├── scripts/
│   ├── build_index.py       # Ingests documents and builds FAISS index
│   ├── create_gold_dataset.py # Generates 40-question gold benchmark
│   ├── download_corpus.py   # Generates 24 bilingual reference PDFs
│   ├── generate_report_docx.py # Builds 2-page assessment DOCX
│   └── run_evaluation.py    # Executes retrieval and generation benchmarks
├── evaluation/
│   ├── gold.jsonl           # 40 bilingual gold questions and answers
│   ├── retrieval_results.json # Dense vs Hybrid retrieval metric output
│   └── answer_evaluation_results.json # Generation correctness & refusal output
├── report/
│   ├── report.docx          # 2-page assessment summary report (Word)
│   └── report.pdf           # 2-page assessment summary report (PDF)
└── tests/
    ├── test_api.py          # FastAPI endpoint integration tests
    ├── test_chunking.py     # Tokenizer and chunk boundary tests
    └── test_retrieval.py    # Dense & Hybrid retrieval tests (13 total)
```

---

## 9. Quickstart Guide

### 1. Installation
```bash
git clone https://github.com/OmarKhalil2003/bi-RAG.git
cd bi-RAG

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Add your OPENROUTER_API_KEY in .env
```

### 3. Build Indexes & Run Tests
```bash
# Ingest documents and build indexes
python scripts/download_corpus.py
python scripts/build_index.py

# Run tests and evaluation
pytest -q
python scripts/run_evaluation.py
```

### 4. Run the API Service
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 5. Query Examples

**English Query**:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"Under NIST SP 800-53 Section AC-2, after how many days of inactivity must system accounts be automatically disabled?"}'
```

**Arabic Query**:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"ما هي الفترة الزمنية الإلزامية للاحتفاظ بسجلات التدقيق الأمني وسجلات الوصول للبيانات وفق سياسة حوكمة البيانات الوطنية؟"}'
```

**Unanswerable Query (Refusal)**:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the statutory minimum salary for commercial airline pilots under civil aviation law?"}'
```

---

## 10. Docker Deployment

```bash
# Build and run with Docker
docker build -t bilingual-rag .
docker run -d -p 8000:8000 --env-file .env --name bilingual-rag-app bilingual-rag

# Or with Docker Compose
docker compose up -d
```
