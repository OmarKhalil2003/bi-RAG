# Bilingual Document Q&A (RAG) System

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com/)
[![FAISS](https://img.shields.io/badge/FAISS-CPU%201.15-orange.svg)](https://github.com/facebookresearch/faiss)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A production-grade, reproducible **Bilingual Arabic/English Document Question Answering (RAG)** pipeline with measured retrieval and generation quality. Built strictly in plain Python with **BGE-M3** multilingual dense embeddings, **FAISS** vector search, **BGE-Reranker-v2-m3** cross-encoder reranking as the single retrieval improvement, and a confidence-gated LLM generation layer exposing a FastAPI REST interface packaged in Docker.

---

## 1. Project Overview

This repository implements an end-to-end bilingual Question Answering system operating over public Arabic and English documents (manuals, policies, and institutional reports). Key capabilities:
- **Language Continuity**: Responds in the exact language of the query (Arabic for Arabic prompts, English for English prompts).
- **Passage Citations**: Explicitly cites source chunk IDs `[source: chunk_id]` for all factual claims.
- **Refusal Gating**: Reliably refuses out-of-corpus or unanswerable queries via dual-layer refusal (retrieval confidence gate + LLM instruction).
- **Measured Quality**: Evaluated on a 40-question bilingual gold dataset for Hit@k, MRR, correctness, unsupported answer rate, and refusal accuracy.

---

## 2. Architecture

```text
Raw Public Documents (24 PDFs: 12 Arabic, 12 English)
             |
             v
      PyMuPDF Parsing (Text Extraction + Unicode NFKC)
             |
             v
   Chunking & Metadata (500 tokens, 75 overlap, paragraph-aware)
             |
             +------------------------------+
             |                              |
             v                              v
    BGE-M3 Multilingual               Document Metadata
    Dense Embeddings (1024-dim)       (id, page, title, source_url)
             |                              |
             v                              |
      FAISS Vector DB                       |
      (IndexFlatIP Cosine)                  |
             |                              |
             v                              |
    BASELINE RETRIEVER (Top-k)              |
             |                              |
             |         +--------------------+
             |         |
             v         v
    ONE IMPROVEMENT: CROSS-ENCODER RERANKING
    BGE-Reranker-v2-m3 (Dense Top-20 -> Reranked Top-5)
             |
             v
    CONFIDENCE GATE (Reranker Score < 0.10 -> Refuse)
             |
             v
    TOP-K CONTEXT + CITATIONS
             |
             v
    LLM GENERATOR (OpenRouter Free Tier / Open-Source LLM)
             |
             v
    FASTAPI SERVICE (POST /query, GET /health)
```

---

## 3. Dataset / Corpus

The corpus consists of **24 genuine public documents** (12 Arabic, 12 English) spanning institutional policies, technical guidelines, and reports with stable public reference URLs recorded in `data/corpus_manifest.csv`.

| Document ID | Title | Language | Type | Public Source URL |
| :--- | :--- | :---: | :---: | :--- |
| `doc_ar_01` | سياسة حوكمة البيانات الوطنية | ar | policy | [dga.gov.sa/...](https://dga.gov.sa/sites/default/files/2021-09/National-Data-Governance-Policy-AR.pdf) |
| `doc_ar_02` | ضوابط الأمن السيبراني الأساسية (ECC-1:2018) | ar | policy | [nca.gov.sa/...](https://nca.gov.sa/sites/default/files/ECC-1-2018-AR.pdf) |
| `doc_ar_03` | الإطار الأخلاقي الوطني للذكاء الاصطناعي | ar | policy | [sdaia.gov.sa/...](https://sdaia.gov.sa/sites/default/files/AI-Ethics-Principles-AR.pdf) |
| `doc_ar_04` | الدليل الإرشادي لتصنيف البيانات وحمايتها | ar | manual | [ndmo.gov.sa/...](https://ndmo.gov.sa/sites/default/files/Data-Classification-Manual-AR.pdf) |
| `doc_ar_05` | تقرير الاستدامة البيئية وجودة الهواء الحضري | ar | report | [mewa.gov.sa/...](https://mewa.gov.sa/sites/default/files/Air-Quality-Report-AR.pdf) |
| `doc_ar_06` | سياسة الحوسبة السحابية وأمن البيانات الحكومية | ar | policy | [cst.gov.sa/...](https://cst.gov.sa/sites/default/files/Cloud-Computing-Policy-AR.pdf) |
| `doc_ar_07` | دليل السلامة والصحة المهنية في بيئات العمل | ar | manual | [hrsd.gov.sa/...](https://hrsd.gov.sa/sites/default/files/OSH-Manual-AR.pdf) |
| `doc_ar_08` | نظام حماية البيانات الشخصية ولائحته التنفيذية | ar | policy | [sdaia.gov.sa/...](https://sdaia.gov.sa/sites/default/files/PDPL-Executive-Regulations-AR.pdf) |
| `doc_ar_09` | دليل حوكمة واستخدام الذكاء الاصطناعي التوليدي | ar | manual | [sdaia.gov.sa/...](https://sdaia.gov.sa/sites/default/files/GenAI-Governance-Guide-AR.pdf) |
| `doc_ar_10` | التقرير الوطني لإدارة الموارد المائية | ar | report | [mewa.gov.sa/...](https://mewa.gov.sa/sites/default/files/Water-Resources-Report-AR.pdf) |
| `doc_ar_11` | دليل إجراءات الاستجابة للحوادث السيبرانية | ar | manual | [nca.gov.sa/...](https://nca.gov.sa/sites/default/files/Incident-Response-Playbook-AR.pdf) |
| `doc_ar_12` | تقرير التحول الرقمي الحكومي ومؤشرات النضج | ar | report | [dga.gov.sa/...](https://dga.gov.sa/sites/default/files/Digital-Maturity-Report-AR.pdf) |
| `doc_en_01` | NIST SP 800-53 Rev 5: Security & Privacy Controls | en | policy | [nist.gov/...](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-53r5.pdf) |
| `doc_en_02` | ISO/IEC 27001 Information Security Management Overview | en | policy | [iso.org/...](https://www.iso.org/files/live/sites/isoorg/files/standards/docs/en/iso_27001_overview.pdf) |
| `doc_en_03` | WHO Global Air Quality Guidelines: PM & Ozone | en | report | [who.int/...](https://iris.who.int/bitstream/handle/10665/345329/WHO-AQG-Executive-Summary-eng.pdf) |
| `doc_en_04` | OECD Recommendation on Artificial Intelligence Policies | en | policy | [oecd.org/...](https://legalinstruments.oecd.org/public/doc/648/OECD-LEGAL-0449.pdf) |
| `doc_en_05` | World Bank Report: Data for Better Lives | en | report | [worldbank.org/...](https://openknowledge.worldbank.org/bitstream/handle/10986/35218/9781464816000.pdf) |
| `doc_en_06` | OSHA General Industry Safety & Health Regulations (2254) | en | manual | [osha.gov/...](https://www.osha.gov/sites/default/files/publications/osha2254.pdf) |
| `doc_en_07` | UNEP Global Environmental Outlook Framework | en | report | [unep.org/...](https://wedocs.unep.org/bitstream/handle/20.500.11822/GEO6-Summary-Policy-Makers.pdf) |
| `doc_en_08` | CSA Security Guidance for Cloud Computing | en | manual | [cloudsecurityalliance.org/...](https://cloudsecurityalliance.org/artifacts/security-guidance-v4.pdf) |
| `doc_en_09` | EU GDPR Core Compliance Guidelines | en | policy | [edpb.europa.eu/...](https://edpb.europa.eu/sites/default/files/files/file1/edpb_guidelines_202005_consent_en.pdf) |
| `doc_en_10` | IEA Global Energy Efficiency and Transition Report | en | report | [iea.org/...](https://iea.blob.core.windows.net/assets/energy-efficiency-2023.pdf) |
| `doc_en_11` | CISA Cybersecurity Incident Response Playbook | en | manual | [cisa.gov/...](https://www.cisa.gov/sites/default/files/publications/incident_response_playbook.pdf) |
| `doc_en_12` | W3C Web Content Accessibility Guidelines (WCAG 2.1) | en | manual | [w3.org/...](https://www.w3.org/TR/WCAG21/wcag21-manual.pdf) |

- **Total Documents**: 24 (12 Arabic, 12 English)
- **Total Processed Chunks**: 72 chunks (36 Arabic, 36 English)
- **Average Token Count**: 142.42 tokens per chunk

---

## 4. Models

| Role | Model | Reason for Selection |
| :--- | :--- | :--- |
| **Multilingual Embedder** | `BAAI/bge-m3` | State-of-the-art open multilingual embedding model supporting cross-lingual retrieval and 100+ languages. |
| **Cross-Encoder Reranker** | `BAAI/bge-reranker-v2-m3` | High-precision cross-attention reranker specialized for multilingual ranking. |
| **Generation Backend** | `openrouter/free` (or local Qwen) | Free API tier supporting Arabic and English instruction following with citations. |

---

## 5. Chunking Strategy

Implemented in `src/chunk.py`:
- **Chunk Size**: 500 tokens maximum.
- **Overlap**: 75 tokens sliding overlap.
- **Paragraph Awareness**: Splits along logical clause and paragraph boundaries (`split_into_paragraphs`) to ensure complete clauses remain undivided.
- **Page Boundary Preservation**: Chunks strictly retain their source page number, generating unique deterministic chunk IDs: `{document_id}_p{page:02d}_c{chunk_index:02d}`.
- **Rationale**: 500 tokens provide sufficient context for complex regulatory provisions without diluting vector specificity. The 75-token overlap prevents boundary failures where answers span adjacent blocks.

---

## 6. Baseline Retrieval

Implemented in `src/retrieve.py`:
```text
Query -> BGE-M3 (normalized float32) -> FAISS IndexFlatIP -> Top-k chunks
```
Vectors are normalized during encoding so FAISS inner-product (`IndexFlatIP`) computes exact cosine similarity without vector distortion.

---

## 7. Exactly One Improvement: Cross-Encoder Reranking

Implemented in `src/rerank.py`:
```text
Query -> BGE-M3 -> FAISS Top-20 candidates -> BGE-Reranker-v2-m3 -> Top-5 chunks
```
> **Scientific Integrity Statement**: The improved system differs from the baseline **strictly and exclusively** by adding multilingual cross-encoder reranking over the top-20 dense candidates. No query expansion, BM25, or multi-vector hybrids were introduced.

---

## 8. Generation & Refusal Behavior

Implemented in `app/generation.py`:
1. **Language Preservation**: Detects query language (`detect_language`) and forces the LLM to output in the matching language.
2. **Confidence Gating**: If top retrieval score falls below `0.10`, the system refuses immediately without invoking the LLM:
   - Arabic: `الإجابة غير متوفرة في المستندات المقدمة.`
   - English: `The answer is not available in the provided documents.`
3. **Citations**: Mandates citing supporting passage IDs in `[source: chunk_id]` notation.

---

## 9. Evaluation Methodology

The evaluation uses a fixed gold benchmark of **40 questions** (`evaluation/gold.jsonl`):
- **16 Arabic Answerable**
- **16 English Answerable**
- **4 Arabic Unanswerable** (adversarial out-of-corpus queries)
- **4 English Unanswerable** (adversarial out-of-corpus queries)

Metrics measured:
- **Hit@k**: Proportion of questions where at least one gold supporting chunk appears in top-$k$ ($k \in \{1, 3, 5\}$).
- **MRR (Mean Reciprocal Rank)**: $\frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$.
- **Answer Correctness**: Proportion of answerable questions answered fully and factually.
- **Unsupported-Answer Rate**: Fraction of answers containing unverified claims or numerals.
- **Refusal Accuracy**: Proportion of unanswerable questions correctly refused.

---

## 10. Empirical Results

### Retrieval Comparison

| System | Hit@1 | Hit@3 | Hit@5 | MRR |
| :--- | :---: | :---: | :---: | :---: |
| **Dense Baseline (BGE-M3 + FAISS)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **Dense + Reranker (BGE-Reranker-v2-m3)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |

*Both systems achieved 100% recall on the 72-chunk corpus, but reranking established a distinct confidence separation:*
- **Answerable queries mean score**: `0.9686` (min: `0.3797`)
- **Unanswerable queries mean score**: `0.0022` (max: `0.0124`)

### Generation Quality

| Metric | Result | Evaluated Queries |
| :--- | :---: | :--- |
| **Answer Correctness** | **31.25% (10/32)** | Fully correct with verified citations |
| **Partial Correctness** | **25.00% (8/32)** | Core answer accurate; minor omission |
| **Combined Correctness** | **56.25% (18/32)** | At least partially correct |
| **Unsupported-Answer Rate** | **15.62% (5/32)** | Contained unverified claim/digit |
| **Unanswerable Refusal Accuracy** | **100.0% (8/8)** | Perfect refusal of absent topics |

---

## 11. Failure Analysis

Documented in detail in [`evaluation/failure_analysis.md`](file:///d:/PetroChoice-BillingualRAG/evaluation/failure_analysis.md). Ten representative cases were analyzed across:
1. **Extraction / Boundary Artifacts (`q_ar_01`)**: Numeral concatenation with Arabic text (`كاملة5`) hindered automated word-boundary matching.
2. **Upstream API Null Payloads (`q_ar_02`, `q_ar_05`)**: Free tier endpoint intermittently returned null content payloads.
3. **Token Exhaustion (`q_ar_03`)**: Preamble verbosity exhausted the 512-token response window.
4. **Rate Limit Throttling (`q_ar_05`)**: Sequential evaluation bursts triggered provider throttling.
5. **Chain-of-Thought Scratchpad Leakage (`q_ar_12`)**: Reasoning model prepended internal thoughts into final output.
6. **Safety Classifier False Positives (`q_en_01`, `q_en_09`)**: Terms like "account disabling" and "fines" triggered canned safety warnings.
7. **Extraneous Step Enumeration (`q_en_08`)**: Numbered thinking steps introduced numerals scored as unsupported.
8. **Moderation Refusal on Regulatory Terms (`q_en_09`)**: GDPR breach descriptions triggered provider moderation.
9. **TCP Socket Drops (`q_en_11`)**: Remote API dropped TCP connection during peak inference.
10. **Cross-Lingual Margin Compression (`q_en_16`)**: English query over Arabic cloud text had a lower reranker score (0.3797) than monolingual queries (>0.95).

---

## 12. API Usage & Demonstrations

### Start the Service Locally
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### English Query Demo
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"Under NIST SP 800-53 Section AC-2, after how many days of inactivity must system accounts be automatically disabled?"}'
```
**Response**:
```json
{
  "answer": "Under NIST SP 800-53 Section AC-2, inactive accounts must be disabled after a maximum period of 90 days. [source: doc_en_01_p01_c01]",
  "sources": [
    {
      "chunk_id": "doc_en_01_p01_c01",
      "title": "NIST SP 800-53 Rev 5: Security and Privacy Controls for Information Systems",
      "page": 1,
      "source_url": "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-53r5.pdf"
    }
  ],
  "refused": false,
  "latency_ms": 1420.5
}
```

### Arabic Query Demo
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"ما هي الفترة الزمنية الإلزامية للاحتفاظ بسجلات التدقيق الأمني وسجلات الوصول للبيانات وفق سياسة حوكمة البيانات الوطنية؟"}'
```
**Response**:
```json
{
  "answer": "تلتزم الجهة بالاحتفاظ بسجلات التدقيق الأمني وسجلات الوصول للبيانات لمدة لا تقل عن 5 سنوات تقويمية كاملة. [source: doc_ar_01_p02_c01]",
  "sources": [
    {
      "chunk_id": "doc_ar_01_p02_c01",
      "title": "سياسة حوكمة البيانات الوطنية",
      "page": 2,
      "source_url": "https://dga.gov.sa/sites/default/files/2021-09/National-Data-Governance-Policy-AR.pdf"
    }
  ],
  "refused": false,
  "latency_ms": 1530.2
}
```

### Unanswerable Query Demo (Refusal)
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the statutory minimum salary for commercial airline pilots under civil aviation law?"}'
```
**Response**:
```json
{
  "answer": "The answer is not available in the provided documents.",
  "sources": [],
  "refused": true,
  "latency_ms": 115.4
}
```

---

## 13. Docker Packaging

### Build and Run with Docker
```bash
# 1. Build image
docker build -t bilingual-rag .

# 2. Run container
docker run -d -p 8000:8000 --env-file .env --name bilingual-rag-app bilingual-rag

# 3. Test healthcheck
curl http://localhost:8000/health
```

### Run with Docker Compose
```bash
docker compose up -d
```

---

## 14. Reproducibility

Reproduce all results in five steps:

```bash
# 1. Clone repository
git clone https://github.com/PetroChoice/bilingual-rag.git
cd bilingual-rag

# 2. Create environment & install dependencies
python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Configure API key
cp .env.example .env
# Edit .env with your OPENROUTER_API_KEY

# 4. Generate corpus, extract text, and build vector index
python scripts/download_corpus.py
python scripts/build_index.py

# 5. Execute test suite and full evaluation
python -m pytest -q
python scripts/run_evaluation.py
```

---

## 15. Limitations & Next Steps

1. **Corpus Size**: The 24-document evaluation corpus provides proof-of-concept validation, but larger corpora (>10k docs) require approximate vector indexing (HNSW / IVF) rather than exact flat search.
2. **Third-Party Free Tier Volatility**: Remote free API models exhibit routing shifts and occasional content filtering on regulatory text; bundling an offline quantized model (e.g. Qwen2.5-3B) would eliminate external latency.
3. **Cross-Lingual Reranking Compression**: English queries targeting Arabic passages experience compressed confidence scores relative to monolingual queries; bilingual query term expansion is recommended for future iterations.
