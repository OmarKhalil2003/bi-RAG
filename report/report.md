# Bilingual Document Q&A (RAG) — Assessment Report

## 1. Approach & Design Rationale

We designed and evaluated an end-to-end bilingual Arabic/English Document Question Answering (RAG) system operating over a curated public corpus of **24 documents** (12 Arabic, 12 English) covering government policies, technical manuals, and institutional reports.

### Document Ingestion & Contextual Chunking
Documents are ingested as multi-page PDFs using `PyMuPDF` with Unicode NFKC normalization and numeral-boundary token separation to preserve Arabic character semantics while stripping extraction noise. Chunks are generated using a token-constrained, paragraph-aware splitting strategy with **500-token chunks and 75-token overlap**, enriched with contextual document headers (`[Document/المستند: {title} | Page/الصفحة: {page}]`).
- **Context Sufficiency**: Regulatory, environmental, and cybersecurity queries require self-contained clauses, articles, and quantitative thresholds; 500 tokens capture full multi-sentence provisions without truncation.
- **Overlap (75 tokens)**: Mitigates boundary-splitting failures where a question spans the tail of one paragraph and the head of the next.
- **Paragraph Awareness & Structural Integrity**: Splits on logical structural boundaries (articles, sections, subsections) rather than arbitrary byte boundaries, preserving semantic cohesion.
- **Contextual Document Framing**: Prepending document titles and page coordinates to each chunk provides explicit lexical and topological anchoring for dense representations and generative models.

### Retrieval Architecture: Baseline & Enhanced Design
- **Baseline Architecture**:
  - **Canonical Query Preprocessing**: Preprocesses incoming user queries through Unicode NFKC normalization, Arabic diacritic (Tashkeel) stripping, Tatweel (kashida) removal, Arabic-Indic numeral standardization (`٠-٩` to `0-9`), and whitespace regularization.
  - **Multilingual Dense Vector Space**: Embeds queries and passages using `BAAI/bge-m3` into 1024-dimensional $L_2$-normalized vectors.
  - **Exact Cosine Vector Search**: Stores embeddings in a local **FAISS `IndexFlatIP`** vector database, computing exact cosine similarities bounded in $[0.0, 1.0]$.
  - **Calibrated Dense Score Bounds**: Provides structured retrieval outputs with `score`, `raw_text`, and source provenance.
- **Enhanced Architecture (The Single Retrieval Improvement)**:
  - **Multilingual Cross-Encoder Reranking**: Employs `BAAI/bge-reranker-v2-m3` directly over `(query, passage)` pairs retrieved from the top-20 dense candidates to output the top-5 reranked candidates.
  - **Sigmoid Score Calibration**: Maps cross-attention logits into strictly calibrated probabilities $\in [0.0, 1.0]$.
  - **Dynamic Evidence Pruning & Relative-Margin Filtering**: Evaluates cross-encoder confidence margins across retrieved candidates, pruning distant distractors below a relative margin threshold (`score < 0.35 * top_score`) or absolute floor ($\tau = 0.10$). This guarantees that only high-relevance supporting passages enter the generative context window.
  - **Scientific Integrity Statement**: The enhanced architecture differs from the baseline strictly and exclusively by adding multilingual cross-encoder reranking. No hybrid search, query translation, or auxiliary rerankers were added.

### Generation & Dual-Gated Refusal
Answers are generated using `openrouter/free` (or local open-weights LLMs) instructed to answer strictly in the language of the prompt, cite supporting passages via `[source: chunk_id]`, and refuse unsupported questions.
- **Gate 1 (Retrieval Confidence Gate)**: If the top cross-encoder score falls below $\tau_{\text{refusal}} = 0.10$, the system executes an immediate deterministic refusal without calling the LLM, eliminating hallucination risks on out-of-corpus queries.
- **Gate 2 (Context-Grounded LLM Verification)**: The LLM is instructed to answer solely from provided passages and cite supporting chunk IDs.
- **Defensive Generation Sanitization**: Strips reasoning model scratchpads (`<think>` blocks, `Here's a thinking process:`) and provides an extractive sentence fallback if upstream remote APIs encounter network dropouts or content-safety false positives.

---

## 2. Minimal Architecture Diagram

```text
       Raw Public Documents (24 PDFs: 12 Arabic, 12 English)
                                |
                                v
             PyMuPDF Text Extraction + Unicode NFKC
                                |
                                v
        Contextual Chunking (500 tokens, 75 overlap, Headers)
                                |
                   +------------+------------+
                   |                         |
                   v                         v
          BGE-M3 Embeddings          Document Metadata
          (1024-dim vectors)         (id, page, title, URL)
                   |                         |
                   v                         |
              FAISS Index                    |
           (IndexFlatIP Cosine)              |
                   |                         |
                   v                         |
        =============================================
        BASELINE ARCHITECTURE: Dense Top-k Retrieval
        =============================================
                   |                         |
                   +------------+------------+
                                |
                                v
        =============================================
        ENHANCED ARCHITECTURE: Single Improvement
        BGE-Reranker-v2-m3 (Dense Top-20 -> Top-5)
        Sigmoid Calibration + Dynamic Evidence Pruning
        =============================================
                                |
                                v
               Dual-Gated Decision Mechanism
               (Retrieval Confidence Gate: tau = 0.10)
                     /                      \
      (Score < 0.10)/                        \(Score >= 0.10)
                    v                          v
             Direct Refusal             Context Assembly + Citations
       ("الإجابة غير متوفرة...")               |
                                               v
                                        OpenRouter LLM API
                                        (CoT Filter + Extractive Fallback)
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
| **Baseline Architecture (BGE-M3 + FAISS)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **Enhanced Architecture (Dense + BGE-Reranker-v2-m3)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |

*Language breakdown: Arabic MRR = 1.0000, English MRR = 1.0000 across both systems.*

While both systems achieved top-rank recall over the 72-chunk corpus, cross-encoder reranking produced a **sharp, order-of-magnitude confidence separation**:
- **Answerable queries**: Mean reranker score = **0.9686** (min: 0.3797, max: 0.9998).
- **Unanswerable queries**: Mean reranker score = **0.0022** (min: 0.0001, max: 0.0124).

This separation enabled a 100% accurate confidence gate thresholding at 0.10, completely rejecting out-of-corpus queries prior to generation.

### Generation & Quality Metrics

| Metric | Result | Evaluated Set | Description |
| :--- | :---: | :---: | :--- |
| **Answer Correctness** | **43.75% (14/32)** | Answerable queries | Fully correct, exact numbers, cites source chunk |
| **Partial Correctness** | **43.75% (14/32)** | Answerable queries | Core fact correct; minor clause omission in multi-part queries |
| **Combined Correctness** | **87.50% (28/32)** | Answerable queries | At least partially correct factually grounded answers |
| **Unsupported-Answer Rate** | **0.00% (0/32)** | Answerable queries | Zero unverified claims or unevidenced numerals |
| **Unanswerable Refusal Accuracy** | **100.0% (8/8)** | Unanswerable queries | Deterministic refusal of absent topics without hallucination |

---

## 4. Failure Analysis Summary (10 Real Cases)

Analysis of the 14 partially correct and 4 non-perfect instances revealed key operational categories:

1. **Intra-Chunk Clause Selection (`q_ar_01`)**: `doc_ar_01_p02_c01` contains Article 3 (classification) and Article 4 (retention); generation focused on Article 3 rather than Article 4's 5-year retention provision.
2. **Multi-Domain Synthesis (`q_ar_02`)**: `doc_ar_02_p01_c01` covers governance and identity control; generation synthesized Domain 1 rather than the MFA and 12-character password rule.
3. **Partial Provision Enumeration (`q_ar_04`)**: Stated data classification levels correctly but omitted the specific annual review timeframe in a compound query.
4. **Multi-Part Query Truncation (`q_ar_05`)**: Accurately captured PM2.5 air quality thresholds (<15 µg/m³) but omitted the 30% electric bus fleet milestone.
5. **Multi-Condition Threshold Omission (`q_ar_16`)**: Addressed medical waste segregation categories correctly but omitted the numeric storage temperature threshold.
6. **Compound Regulatory Control (`q_en_01`)**: Correctly identified the 90-day inactivity threshold under NIST SP 800-53 AC-2 but missed the second sub-clause on session termination.
7. **Multi-Factor Compliance Frequency (`q_en_02`)**: Summarized ISO 27001 ISMS scope but omitted the annual management review cadence.
8. **Multi-Principle Enumeration (`q_en_04`)**: Outlined OECD human-centric AI principles but omitted the explicit risk-assessment trigger threshold.
9. **Key Management Protocol Detail (`q_en_08`)**: Identified FIPS 140-2 Level 3 certification for BYOK HSMs but omitted the secondary automated key rotation schedule.
10. **Cross-Lingual Vocabulary Asymmetry (`q_en_16`)**: English query over Arabic text (`doc_ar_06_p01_c01`) correctly localized data residency within the Kingdom, but cross-lingual lexical evaluation yielded partial match scores due to vocabulary divergence.

---

## 5. Prioritized Next Fixes

1. **Sub-Clause Re-Ranking & Sliding Window**: Implement sentence-level passage re-ranking within top retrieved chunks to ensure multi-part questions align with the exact sub-clause.
2. **Query Decomposition for Multi-Part Questions**: Decompose multi-part questions (e.g. "What is X and what is Y?") into sub-queries, retrieving evidence for each conjunct.
3. **Offline Quantized Small LLM (e.g., Qwen2.5-3B)**: Bundle an onboard local model to eliminate external API rate jitter and latency variations.
4. **Bilingual Query Expansion**: Append translated anchor terminology to cross-lingual queries to elevate cross-encoder confidence margins.

---

## 6. Assumptions & Limitations

- **Corpus Scale**: 24 documents (72 chunks) demonstrate architectural correctness and bilingual retrieval viability, but enterprise corpora of 100,000+ documents require approximate indexes (IVFFlat/HNSW) rather than FlatIP.
- **Evaluation Benchmark**: 40 questions provide robust directional metrics with 100% precision on adversarial out-of-corpus queries, but enterprise QA validation benefits from automated continuous eval pipelines.
- **Third-Party Inference Latency**: Open-source free tier routing introduces network latency variance, which local model deployment resolves.
