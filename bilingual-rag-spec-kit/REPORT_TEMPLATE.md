# Bilingual Document Q&A — 2-Page Report

## 1. Approach

We built a bilingual Arabic/English RAG system over **[N] public documents**. The corpus contains **[N_AR] Arabic** and **[N_EN] English** documents.

Documents are parsed into page-aware chunks using **500-token chunks with 75-token overlap**. This was selected to preserve enough context for policy/report questions while keeping retrieval passages focused. Paragraph-aware splitting reduces semantic fragmentation.

The baseline uses **BGE-M3** multilingual embeddings with **FAISS** dense retrieval. The only retrieval improvement is **BGE multilingual cross-encoder reranking** of the top-20 dense candidates before selecting the final top-5.

Generation uses **[MODEL]**. The model is instructed to answer only from retrieved passages, answer in the question's language, cite supporting chunk IDs, and refuse when the evidence is insufficient. A retrieval-confidence gate provides an additional refusal mechanism.

## 2. Architecture

```text
Documents
   |
Parsing + Chunking
   |
BGE-M3
   |
FAISS
   |
Baseline --------------------+
   |                         |
   |                    Top-20 candidates
   |                         |
   |                    BGE Reranker
   |                         |
   +-------------------------+
             |
          Top-5
             |
          LLM
             |
 Answer + citations / refusal
             |
          FastAPI
```

## 3. Evaluation

The gold set contains **[N] questions**, including **[N_AR] Arabic**, **[N_EN] English**, and **[N_UNANSWERABLE] unanswerable** questions.

### Retrieval

| System | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---:|---:|---:|---:|
| Dense baseline | | | | |
| Dense + reranker | | | | |

### Generation

| Metric | Result |
|---|---:|
| Answer correctness | |
| Partial correctness | |
| Unsupported-answer rate | |
| Unanswerable refusal accuracy | |

The reranker **[improved/did not improve]** retrieval. The largest change occurred at **[metric]**, indicating **[brief interpretation]**.

## 4. Failure Analysis

Ten failures were analyzed. The main causes were **[cause 1]**, **[cause 2]**, and **[cause 3]**. The most important failure was **[brief example]**, where **[root cause]**.

## 5. Next Fixes

The next highest-impact fixes would be:

1. **[fix]**
2. **[fix]**
3. **[fix]**

These were intentionally not included in the current experiment so that the comparison remained limited to exactly one retrieval improvement.

## 6. Limitations

The corpus is relatively small, the gold set is limited, manual correctness evaluation introduces some subjectivity, and PDF extraction can introduce noise. Local LLM quality also depends on available hardware. Retrieval quality does not guarantee generation faithfulness.
