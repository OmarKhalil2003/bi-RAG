# Final Agent Checklist

Before reporting completion, execute and verify:

## Code
- [ ] imports work
- [ ] no dead critical path
- [ ] no hard-coded secret
- [ ] no fake data
- [ ] no fake metrics

## Corpus
- [ ] >=20 documents
- [ ] Arabic + English
- [ ] public source URLs
- [ ] manifest
- [ ] reproducible ingestion

## Retrieval
- [ ] BGE-M3
- [ ] FAISS
- [ ] dense baseline
- [ ] top-20 candidate retrieval
- [ ] BGE reranker
- [ ] top-5 final retrieval
- [ ] no other retrieval enhancement

## Evaluation
- [ ] >=40 questions
- [ ] >=8 unanswerable
- [ ] Arabic + English
- [ ] Hit@1
- [ ] Hit@3
- [ ] Hit@5
- [ ] MRR
- [ ] correctness
- [ ] unsupported-answer rate
- [ ] refusal accuracy
- [ ] exactly 10 failure analyses

## API
- [ ] POST /query
- [ ] validation
- [ ] logging
- [ ] tests
- [ ] Docker

## Documentation
- [ ] README
- [ ] architecture
- [ ] design rationale
- [ ] actual metrics
- [ ] 2-page report
- [ ] assumptions
- [ ] limitations

## Final rule

If a feature does not directly help satisfy an assessment requirement, do not add it.
