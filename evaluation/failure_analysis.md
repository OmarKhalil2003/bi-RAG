# Failure Analysis: 10 Representative Cases
Bilingual Document Q&A (RAG) System

This document analyzes exactly **10 representative failure cases** observed during the empirical evaluation of the baseline and enhanced retrieval pipelines and generation modules on the fixed `evaluation/gold.jsonl` benchmark.

---

### Failure 1: Intra-Chunk Clause Selection Bias
- **Question**: ما هي الفترة الزمنية الإلزامية للاحتفاظ بسجلات التدقيق الأمني وسجلات الوصول للبيانات وفق سياسة حوكمة البيانات الوطنية؟
- **Language**: Arabic (ar)
- **Answerable**: Yes
- **Expected Evidence**: `doc_ar_01_p02_c01` ("تلتزم الجهة بالاحتفاظ بسجلات التدقيق الأمني وسجلات الوصول للبيانات لمدة لا تقل عن 5 سنوات تقويمية كاملة.")
- **Baseline Retrieval**: `doc_ar_01_p02_c01` (Dense score: 0.6827, Rank 1)
- **Enhanced Retrieval**: `doc_ar_01_p02_c01` (Reranker score: 0.9990, Rank 1)
- **Generation Behavior**: The model synthesized Article 3 ("معايير تصنيف البيانات...") from the lead portion of the chunk rather than the subsequent Article 4 clause stating the mandatory 5-year retention period.
- **Failure Category**: Multi-topic chunk attention bias / clause alignment.
- **Root Cause**: The 500-token chunk encompassed two related legal articles (Article 3 and Article 4). The generator attended strongly to the lead topic rather than isolating the second clause addressing audit retention.
- **Next Fix**: Implement sentence-level salience windowing or atomic sub-article chunk splitting during document ingestion.

---

### Failure 2: Multi-Domain Lead Preference
- **Question**: ما هي شروط طول كلمة المرور واستخدام المصادقة متعددة العوامل في ضوابط الأمن السيبراني الأساسية (ECC-1:2018)؟
- **Language**: Arabic (ar)
- **Answerable**: Yes
- **Expected Evidence**: `doc_ar_02_p01_c01` ("يجب تطبيق متطلبات المصادقة متعددة العوامل (MFA)... يجب ألا يقل الحد الأدنى لطول كلمات المرور عن 12 خانة")
- **Baseline Retrieval**: `doc_ar_02_p01_c01` (Dense score: 0.7191, Rank 1)
- **Enhanced Retrieval**: `doc_ar_02_p01_c01` (Reranker score: 0.9984, Rank 1)
- **Generation Behavior**: The model summarized Domain 1 ("حوكمة الأمن السيبراني واستراتيجية الدفاع...") from the top of the passage rather than Domain 2 ("إدارة الهوية والتحكم في الوصول الرقمي").
- **Failure Category**: Intra-chunk lead topic dominance.
- **Root Cause**: Both domains appeared on the same page and were merged into one chunk; the language model focused on the foundational governance mandate.
- **Next Fix**: Enforce query-guided paragraph filtering prior to assembling the final prompt context.

---

### Failure 3: Partial Provision Enumeration in Compound Queries
- **Question**: ما هي المستويات الأربعة لتصنيف البيانات في الدليل الإرشادي وما هي دورية مراجعة التصنيف؟
- **Language**: Arabic (ar)
- **Answerable**: Yes
- **Expected Evidence**: `doc_ar_04_p01_c01` (4 levels: Public, Restricted, Confidential, Top Secret; annual review)
- **Baseline Retrieval**: `doc_ar_04_p01_c01` (Dense score: 0.7154, Rank 1)
- **Enhanced Retrieval**: `doc_ar_04_p01_c01` (Reranker score: 0.9989, Rank 1)
- **Generation Behavior**: Correctly listed the four classification tiers but omitted the periodic annual review schedule.
- **Failure Category**: Incomplete multi-part answer.
- **Root Cause**: The prompt contained two distinct questions connected by "وما هي", and the generator satisfied the dominant list query while dropping the secondary schedule question.
- **Next Fix**: Query decomposition breaking compound prompts into distinct sub-questions for individual verification.

---

### Failure 4: Cross-Page Evidence Distribution
- **Question**: ما هو الحد السنوي المستهدف لتركيز الجسيمات العالقة PM2.5 في تقرير جودة الهواء وما هي مستهدفات الحافلات الكهربائية بحلول 2030؟
- **Language**: Arabic (ar)
- **Answerable**: Yes
- **Expected Evidence**: `doc_ar_05_p01_c01` (PM2.5 < 15 µg/m³) and `doc_ar_05_p02_c01` (30% electric buses)
- **Baseline Retrieval**: `doc_ar_05_p01_c01` (Dense score: 0.6517, Rank 1)
- **Enhanced Retrieval**: `doc_ar_05_p01_c01` (Reranker score: 0.9944, Rank 1)
- **Generation Behavior**: Answered the air quality particulate threshold (<15 µg/m³) accurately, but omitted the electric transit fleet target.
- **Failure Category**: Multi-page distributed evidence.
- **Root Cause**: The two facts resided across page 1 and page 2 of the environmental report; the top-ranked passage contained only page 1 context.
- **Next Fix**: Multi-chunk context aggregation assembling candidate passages across adjacent document pages for compound queries.

---

### Failure 5: Multi-Condition Threshold Omission
- **Question**: أين يجب استضافة البيانات الحكومية المقيدة ومفاتيح تشفيرها وما هو المعيار المعتمد لتشفير البيانات أثناء النقل؟
- **Language**: Arabic (ar)
- **Answerable**: Yes
- **Expected Evidence**: `doc_ar_06_p01_c01` (inside KSA) and `doc_ar_06_p02_c01` (TLS 1.3 / AES-256)
- **Baseline Retrieval**: `doc_ar_06_p01_c01` (Dense score: 0.6880, Rank 1)
- **Enhanced Retrieval**: `doc_ar_06_p01_c01` (Reranker score: 0.9790, Rank 1)
- **Generation Behavior**: Answered sovereign data hosting inside the Kingdom but omitted transit encryption specifications.
- **Failure Category**: Multi-hop architectural question omission.
- **Root Cause**: The question asked about both physical hosting and transit cryptography; the top passage only covered the hosting article.
- **Next Fix**: Sub-query expansion generating secondary search vectors for compound technical requirements.

---

### Failure 6: Missing Quantitative Parameter in Procedure Query
- **Question**: ما هي درجات الحرارة المحددة لحفظ النفايات الطبية الخطرة وفترة التخزين القصوى قبل المعالجة؟
- **Language**: Arabic (ar)
- **Answerable**: Yes
- **Expected Evidence**: `doc_ar_07_p02_c01` (storage below 4°C, maximum 24-48 hours)
- **Baseline Retrieval**: `doc_ar_07_p02_c01` (Dense score: 0.6218, Rank 1)
- **Enhanced Retrieval**: `doc_ar_07_p02_c01` (Reranker score: 0.9891, Rank 1)
- **Generation Behavior**: Outlined safety protocols and storage procedures but omitted the specific 4°C temperature figure.
- **Failure Category**: Quantitative unit omission.
- **Root Cause**: The model prioritized qualitative safety guidelines over numeric measurement thresholds.
- **Next Fix**: Prompt constraint instructing the model to prioritize quantitative specifications and measurement units.

---

### Failure 7: Secondary Clause Dropping in Regulatory Standard
- **Question**: Under NIST SP 800-53 Section AC-2, after how many days of inactivity must system accounts be automatically disabled, and what is the session lock timeout?
- **Language**: English (en)
- **Answerable**: Yes
- **Expected Evidence**: `doc_en_01_p01_c01` (90 days inactivity; 15 minutes session lock)
- **Baseline Retrieval**: `doc_en_01_p01_c01` (Dense score: 0.7635, Rank 1)
- **Enhanced Retrieval**: `doc_en_01_p01_c01` (Reranker score: 0.9994, Rank 1)
- **Generation Behavior**: Correctly cited the 90-day inactivity threshold but dropped the secondary session lock clause.
- **Failure Category**: Compound regulatory inquiry omission.
- **Root Cause**: The model treated the inactivity question as the primary objective, omitting the secondary timing constraint.
- **Next Fix**: Prompt checklist mechanism requiring explicit validation of all numeric parameters mentioned in the query.

---

### Failure 8: Cadence Specification Omission in Framework Summary
- **Question**: What are the three core elements of the ISO/IEC 27001 ISMS scope definition, and how frequently must management reviews occur?
- **Language**: English (en)
- **Answerable**: Yes
- **Expected Evidence**: `doc_en_02_p01_c01` (Internal/external issues, interested party requirements, interfaces/dependencies; annual review)
- **Baseline Retrieval**: `doc_en_02_p01_c01` (Dense score: 0.8095, Rank 1)
- **Enhanced Retrieval**: `doc_en_02_p01_c01` (Reranker score: 0.9984, Rank 1)
- **Generation Behavior**: Correctly listed the three scope boundary elements but omitted the annual review frequency.
- **Failure Category**: Incomplete multi-part answer.
- **Root Cause**: Generating structured lists creates token exhaustion for subsequent paragraph conclusions.
- **Next Fix**: Structure prompt instructions: "If multiple questions are asked, number each answer to match the questions."

---

### Failure 9: Omission of Classification Trigger Phrase
- **Question**: What are the five OECD AI principles, and what risk classification threshold triggers a mandatory human impact assessment?
- **Language**: English (en)
- **Answerable**: Yes
- **Expected Evidence**: `doc_en_04_p01_c01` (Inclusive growth, human-centered values, transparency, robustness/safety, accountability; high-risk threshold)
- **Baseline Retrieval**: `doc_en_04_p01_c01` (Dense score: 0.7677, Rank 1)
- **Enhanced Retrieval**: `doc_en_04_p01_c01` (Reranker score: 0.9949, Rank 1)
- **Generation Behavior**: Detailed all five OECD AI principles but omitted the explicit "high-risk" classification trigger.
- **Failure Category**: Secondary clause omission.
- **Root Cause**: The model expended context summarizing all five principles and terminated before answering the threshold clause.
- **Next Fix**: Enforce concise bullet summaries without repetitive expository preamble.

---

### Failure 10: Cross-Lingual Evaluation Metric Asymmetry
- **Question**: According to the Saudi Cloud Computing Regulatory Policy, where must sovereign and restricted government data and their encryption keys be hosted?
- **Language**: English (en)
- **Answerable**: Yes (Cross-lingual)
- **Expected Evidence**: `doc_ar_06_p01_c01` ("يجب أن تكون مراكز البيانات التي تستضيف بيانات حكومية أو بيانات مقيدة ومفاتيح تشفيرها موجودة جغرافياً داخل حدود المملكة.")
- **Baseline Retrieval**: `doc_ar_06_p01_c01` (Dense score: 0.5766, Rank 1)
- **Enhanced Retrieval**: `doc_ar_06_p01_c01` (Reranker score: 0.3797, Rank 1)
- **Generation Behavior**: The model accurately translated and answered in English that sovereign data must be stored geographically within the Kingdom of Saudi Arabia with citations. However, the automated string evaluator scored it partial due to cross-lingual lexical distance.
- **Failure Category**: Evaluation metric cross-lingual mismatch.
- **Root Cause**: The string evaluator expects direct lexical overlap with the Arabic text or a rigid translation template.
- **Next Fix**: Utilize LLM-as-a-judge or semantic embedding similarity for cross-lingual correctness evaluation.
