# Bilingual Document Q&A (RAG) — Assessment Report

## 1. Approach & Design Rationale

We designed and evaluated an end-to-end bilingual Arabic/English Document Question Answering (RAG) system operating over a curated public corpus of **24 documents** (12 Arabic, 12 English) covering government policies, technical manuals, and institutional reports. 

### Document Ingestion & Chunking Rationale
Documents are ingested as multi-page PDFs using `PyMuPDF` with Unicode NFKC normalization to preserve Arabic character semantics while stripping extraction noise. Chunks are generated using a token-constrained, paragraph-aware splitting strategy with **500-token chunks and 75-token overlap**.
- **Context Sufficiency**: Regulatory, environmental, and cybersecurity queries require self-contained clauses, articles, and quantitative thresholds; 500 tokens capture full multi-sentence provisions without truncation.
- **Overlap (75 tokens)**: Mitigates boundary-splitting failures where a question spans the tail of one paragraph and the head of the next.
- **Paragraph Awareness**: Splits on logical structural boundaries (articles, sections, subsections) rather than arbitrary byte boundaries, preserving semantic cohesion.

### Retrieval Architecture & The Single Improvement
- **Baseline Retriever**: Employs `BAAI/bge-m3` multilingual dense embeddings with a local **FAISS `IndexFlatIP`** vector database. Because vectors are $L_2$-normalized upon encoding, inner product search computes exact cosine similarity.
- **Exactly One Retrieval Improvement — Cross-Encoder Reranking**: The improved pipeline retrieves the top-20 dense candidates from FAISS, applies the multilingual cross-encoder `BAAI/bge-reranker-v2-m3` directly over `(query, passage)` pairs, and returns the top-5 candidates. No hybrid search, query translation, or second reranker was added, ensuring experimental interpretability.

### Generation & Refusal Gating
Answers are generated using `openrouter/free` (or local open-weights LLMs) instructed to answer strictly in the language of the prompt, cite supporting passages via `[source: chunk_id]`, and refuse unsupported questions. To prevent hallucination when evidence is absent, a **retrieval confidence gate** (`score < 0.10`) enforces deterministic refusal prior to LLM invocation.

---

## 2. Minimal Architecture Diagram

```text
       Raw Public Documents (24 PDFs: 12 Arabic, 12 English)
                                |
                                v
             PyMuPDF Text Extraction + Unicode NFKC
                                |
                                v
        Paragraph-Aware Chunking (500 tokens, 75 overlap)
                                |
                   +------------+------------+
                   |                         |
                   v                         v
          BGE-M3 Embeddings          Document Metadata
          (1024-dim vectors)         (id, page, title, URL)
                   |                         |
                   v                         |
              FAISS Index                    |
                   |                         |
                   v                         |
          Dense Baseline (Top-k)             |
                   |                         |
                   +------------+------------+
                                |
                                v
                   [One Improvement: Reranking]
                   BGE-Reranker-v2-m3 (Top-20 -> Top-5)
                                |
                                v
                   Confidence Gate (Threshold = 0.10)
                     /                      \
      (Score < 0.10) /                        \ (Score >= 0.10)
                    v                          v
             Direct Refusal             Context Assembly + Citations
       ("الإجابة غير متوفرة...")               |
                                               v
                                        OpenRouter LLM API
                                               |
                                               v
                                     FastAPI Endpoint (/query)
```

---

## 3. Quantitative Evaluation

Evaluation was conducted against a fixed gold dataset of **40 questions** (20 Arabic, 20 English) featuring **32 answerable** queries and **8 unanswerable** adversarial queries across factual lookups, numeric thresholds, regulatory timeframes, and cross-lingual requirements.

### Retrieval Performance Comparison

| System | Hit@1 | Hit@3 | Hit@5 | MRR |
| :--- | :---: | :---: | :---: | :---: |
| **Dense Baseline (BGE-M3 + FAISS)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **Dense + Reranker (BGE-Reranker-v2-m3)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |

*Language breakdown: Arabic MRR = 1.0000, English MRR = 1.0000 across both systems.*

While both systems achieved perfect Top-1 recall over this 72-chunk corpus, cross-encoder reranking produced a **sharp, order-of-magnitude confidence separation**:
- **Answerable queries**: Mean reranker score = **0.9686** (min: 0.3797, max: 0.9998).
- **Unanswerable queries**: Mean reranker score = **0.0022** (min: 0.0001, max: 0.0124).
This separation enabled a 100% accurate confidence gate thresholding at 0.10.

### Generation & Quality Metrics

| Metric | Result | Evaluated Set |
| :--- | :---: | :--- |
| **Answer Correctness** | **31.25% (10/32)** | Fully correct, exact numbers, cites source chunk |
| **Partial Correctness** | **25.00% (8/32)** | Core fact correct; minor formatting/list omission |
| **Combined Correctness** | **56.25% (18/32)** | At least partially correct |
| **Unsupported-Answer Rate** | **15.62% (5/32)** | Contains numeral/claim unverified by evidence |
| **Unanswerable Refusal Accuracy** | **100.0% (8/8)** | Correctly refused absent topics without hallucination |

---

## 4. Failure Analysis Summary (10 Real Cases)

Analysis of the 14 non-perfect instances revealed four predominant failure categories:
1. **Extraction / Boundary Artifacts (e.g., `q_ar_01`)**: RTL numeral concatenation (`كاملة5`) caused automated string evaluators to flag supported numerals as unsupported.
2. **Upstream Free API Transient Errors (e.g., `q_ar_02`, `q_ar_05`, `q_ar_11`)**: High provider load on OpenRouter free endpoints returned null `content` payloads or dropped sockets during peak evaluation hours.
3. **Chain-of-Thought Scratchpad Leakage (e.g., `q_ar_12`, `q_en_08`)**: Reasoning models prepended `Here's a thinking process:...` into user answers, introducing extraneous enumeration numbers.
4. **Safety Filter False Positives (e.g., `q_en_01`, `q_en_09`)**: Regulatory phrases like "account disabling" and "penalties" triggered template safety responses (`User Safety: safe`).
5. **Cross-Lingual Margin Attenuation (e.g., `q_en_16`)**: Cross-lingual retrieval (English query over Arabic cloud policy) yielded a lower reranker score (0.3797) than monolingual queries (>0.95).

---

## 5. Prioritized Next Fixes

1. **Extraction Normalization Filter**: Insert regex spacing around numerals (`[\u0600-\u06FF](\d+)`) during PDF ingestion to prevent word-digit fusion.
2. **Defensive Post-Processing**: Strip chain-of-thought markdown traces (`<think>` or `Here's a thinking process`) before emitting API responses.
3. **Local Small LLM Fallback**: Package a quantised local model (e.g., `Qwen2.5-3B-Instruct` or `0.5B`) to guarantee inference when external free APIs experience throttling.
4. **Bilingual Query Expansion**: Append translated anchor terminology to cross-lingual queries to elevate cross-encoder confidence margins.

---

## 6. Assumptions & Limitations

- **Corpus Scale**: 24 documents (72 chunks) demonstrate architectural correctness and bilingual retrieval viability, but enterprise corpora of 100,000+ documents require approximate indexes (IVFFlat/HNSW) rather than FlatIP.
- **Free API Dependencies**: Free-tier API endpoints exhibit latency jitter, content-safety false positives, and variable model routing.
- **Evaluation Set Size**: 40 questions establish directional metrics but provide a wide confidence interval (±10%).
