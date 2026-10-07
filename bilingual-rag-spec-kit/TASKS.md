# Implementation Task List

## Phase 1 — Skeleton

- [ ] Create repository structure from SPEC.md.
- [ ] Create `requirements.txt`.
- [ ] Create `.env.example`.
- [ ] Create basic README.
- [ ] Add `.gitignore`.

## Phase 2 — Corpus

- [ ] Select >=20 public Arabic/English documents.
- [ ] Create `data/corpus_manifest.csv`.
- [ ] Implement `scripts/download_corpus.py`.
- [ ] Download/verify documents.
- [ ] Implement PDF extraction.
- [ ] Preserve page metadata.
- [ ] Implement chunking.
- [ ] Write `chunks.jsonl`.
- [ ] Document 500-token / 75-token overlap choice.

## Phase 3 — Baseline Retrieval

- [ ] Load BGE-M3.
- [ ] Generate normalized embeddings.
- [ ] Build FAISS index.
- [ ] Persist index and chunk metadata.
- [ ] Implement top-k retrieval.
- [ ] Add retrieval unit tests.

## Phase 4 — One Improvement

- [ ] Load BGE reranker v2 multilingual.
- [ ] Retrieve top-20 dense candidates.
- [ ] Rerank.
- [ ] Return top-5.
- [ ] Confirm no other retrieval improvement exists.

## Phase 5 — Gold Set

- [ ] Create >=40 questions.
- [ ] Include Arabic and English.
- [ ] Include >=8 unanswerable questions.
- [ ] Assign supporting chunk IDs.
- [ ] Validate gold-set schema.
- [ ] Keep gold data fixed before final evaluation.

## Phase 6 — Retrieval Evaluation

- [ ] Implement Hit@1.
- [ ] Implement Hit@3.
- [ ] Implement Hit@5.
- [ ] Implement MRR.
- [ ] Run baseline.
- [ ] Run reranked system.
- [ ] Save results.

## Phase 7 — Generation

- [ ] Implement context formatting.
- [ ] Implement language-preserving answer generation.
- [ ] Implement citations.
- [ ] Implement refusal prompt.
- [ ] Implement retrieval confidence gate.
- [ ] Test answerable example.
- [ ] Test unanswerable example.

## Phase 8 — Answer Evaluation

- [ ] Label correctness.
- [ ] Calculate correctness.
- [ ] Calculate unsupported-answer rate.
- [ ] Calculate unanswerable refusal accuracy.
- [ ] Analyze exactly 10 failures.

## Phase 9 — API

- [ ] Implement POST `/query`.
- [ ] Add Pydantic request validation.
- [ ] Add response schema.
- [ ] Add logging.
- [ ] Add request ID.
- [ ] Add API tests.

## Phase 10 — Docker

- [ ] Create Dockerfile.
- [ ] Build image.
- [ ] Run API container.
- [ ] Test `/query` from host.
- [ ] Document limitations around local LLM hardware/API mode.

## Phase 11 — Deliverables

- [ ] Finalize README.
- [ ] Finalize 2-page report.
- [ ] Add architecture diagram.
- [ ] Add metrics table.
- [ ] Add limitations.
- [ ] Verify reproducibility commands.
- [ ] Run complete test/evaluation gate.
