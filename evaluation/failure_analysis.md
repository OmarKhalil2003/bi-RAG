# Failure Analysis: 10 Representative Cases
Bilingual Document Q&A (RAG) System

This document analyzes exactly **10 representative failure cases** observed during the empirical evaluation of the baseline dense and enhanced hybrid retrieval pipelines and generation modules on the fixed `evaluation/gold.jsonl` benchmark.

---

### Failure 1: Dual-Gate Threshold Sensitivity on Policy Query
- **Question**: ما هي شروط طول كلمة المرور واستخدام المصادقة متعددة العوامل في ضوابط الأمن السيبراني الأساسية (ECC-1:2018)؟
- **Language**: Arabic (ar)
- **Answerable**: Yes
- **Expected Evidence**: `doc_ar_02_p01_c01` ("يجب تطبيق متطلبات المصادقة متعددة العوامل (MFA)... يجب ألا يقل الحد الأدنى لطول كلمات المرور عن 12 خانة مع إلزامية تعقيد الرموز وتغييرها كل 90 يوماً.")
- **Baseline Dense Retrieval**: `doc_ar_02_p01_c01` (Dense score: 0.7191, Rank 1)
- **Enhanced Hybrid Retrieval**: `doc_ar_02_p01_c01` (RRF score: 0.0328, Rank 1)
- **Generation Behavior**: The dual-gate confidence check returned an incorrect refusal before calling the LLM.
- **Failure Category**: Refusal gate threshold calibration.
- **Root Cause**: The hybrid confidence gating logic checked both dense and sparse score conditions; a conservative threshold boundary triggered a premature refusal despite strong dense similarity.
- **Next Fix**: Calibrate the dual-gate threshold to evaluate disjunctive confidence: if dense cosine similarity $\ge 0.60$, allow generation regardless of sparse token count.

---

### Failure 2: Evaluator String Matching Flagging Enumeration Digits
- **Question**: ما هي المبادئ الأخلاقية للذكاء الاصطناعي وما هو الإجراء المطلوب قبل الإطلاق التجاري للنماذج؟
- **Language**: Arabic (ar)
- **Answerable**: Yes
- **Expected Evidence**: `doc_ar_03_p01_c01` (المبادئ السبعة للذكاء الاصطناعي واختبارات التحيز الإحصائي قبل الإطلاق)
- **Baseline Dense Retrieval**: `doc_ar_03_p01_c01` (Dense score: 0.6620, Rank 1)
- **Enhanced Hybrid Retrieval**: `doc_ar_03_p01_c01` (RRF score: 0.0328, Rank 1)
- **Generation Behavior**: The model accurately described the ethical principles and structured them as numbered items ("1. العدالة ... 2. الشفافية ... 4. الخصوصية ... 5. الأمان"). The automated evaluator flagged digits '4' and '5' as unsupported against the concise gold text.
- **Failure Category**: Evaluation metric string artifact.
- **Root Cause**: The string verification script compares extracted digits directly against gold answers without ignoring list enumeration numbering.
- **Next Fix**: Strip enumeration prefixes (`1.`, `2.`, etc.) from generated text before checking for unevidenced quantitative entities.

---

### Failure 3: Cross-Page Distributed Evidence
- **Question**: ما هو الحد السنوي المستهدف لتركيز الجسيمات العالقة PM2.5 في تقرير جودة الهواء وما هي مستهدفات الحافلات الكهربائية بحلول 2030؟
- **Language**: Arabic (ar)
- **Answerable**: Yes
- **Expected Evidence**: `doc_ar_05_p01_c01` (PM2.5 < 15 µg/m³) and `doc_ar_05_p02_c01` (30% electric buses)
- **Baseline Dense Retrieval**: `doc_ar_05_p01_c01` (Dense score: 0.6517, Rank 1)
- **Enhanced Hybrid Retrieval**: `doc_ar_05_p02_c01` (RRF score: 0.0325, Rank 1)
- **Generation Behavior**: Accurately captured the electric bus target (30% by 2030) but omitted the PM2.5 threshold from page 1.
- **Failure Category**: Multi-page distributed evidence omission.
- **Root Cause**: The two required facts were located across adjacent pages in the report; hybrid search ranked page 2 slightly ahead, leaving page 1 context underrepresented.
- **Next Fix**: Multi-chunk context aggregation assembling candidate passages across adjacent document pages for compound queries.

---

### Failure 4: Water Strategy Multi-Target Omission
- **Question**: ما هي مستهدفات خفض الفاقد في شبكات المياه بحلول 2030 وما هو حجم المخزون الاستراتيجي المستهدف في التقرير الوطني لإدارة الموارد المائية؟
- **Language**: Arabic (ar)
- **Answerable**: Yes
- **Expected Evidence**: `doc_ar_10_p01_c01` (فاقد المياه أقل من 15%) and `doc_ar_10_p02_c01` (مخزون استراتيجي 3 أيام)
- **Baseline Dense Retrieval**: `doc_ar_10_p01_c01` (Dense score: 0.6518, Rank 1)
- **Enhanced Hybrid Retrieval**: `doc_ar_10_p01_c01` (RRF score: 0.0328, Rank 1)
- **Generation Behavior**: Answered the strategic storage duration but omitted the network leakage percentage threshold.
- **Failure Category**: Multi-hop clause omission.
- **Root Cause**: Compound query connecting two operational domains; generator satisfied the first factual question while truncating the second.
- **Next Fix**: Query decomposition splitting multi-conjunct questions into distinct sub-queries.

---

### Failure 5: Infrastructure Distribution Network Metric Omission
- **Question**: كم تبلغ نسبة تغطية خدمات مياه الشرب الحضرية وما هي المدة الزمنية المستهدفة للاستجابة لانكسارات الأنابيب الرئيسية؟
- **Language**: Arabic (ar)
- **Answerable**: Yes
- **Expected Evidence**: `doc_ar_10_p03_c01` (تغطية 100%، استجابة خلال ساعتين)
- **Baseline Dense Retrieval**: `doc_ar_10_p03_c01` (Dense score: 0.7178, Rank 1)
- **Enhanced Hybrid Retrieval**: `doc_ar_10_p03_c01` (RRF score: 0.0328, Rank 1)
- **Generation Behavior**: Stated the 100% urban coverage target accurately but omitted the 2-hour pipe repair timeframe.
- **Failure Category**: Secondary clause omission in compound prompt.
- **Root Cause**: Token limit and generative preference for macro-level coverage figures over micro-level operational repair times.
- **Next Fix**: Implement an answer completeness checklist verifying that all interrogative entities in the user prompt are addressed.

---

### Failure 6: Omission of Classification Trigger Phrase
- **Question**: What are the five OECD AI principles, and what risk classification threshold triggers a mandatory human impact assessment?
- **Language**: English (en)
- **Answerable**: Yes
- **Expected Evidence**: `doc_en_04_p01_c01` (Inclusive growth, human-centered values, transparency, robustness/safety, accountability; high-risk threshold)
- **Baseline Dense Retrieval**: `doc_en_04_p01_c01` (Dense score: 0.7677, Rank 1)
- **Enhanced Hybrid Retrieval**: `doc_en_04_p01_c01` (RRF score: 0.0328, Rank 1)
- **Generation Behavior**: Detailed all five OECD AI principles but omitted the explicit "high-risk" classification trigger.
- **Failure Category**: Secondary clause omission.
- **Root Cause**: The model expended context summarizing all five principles and terminated before answering the threshold clause.
- **Next Fix**: Enforce concise bullet summaries without repetitive expository preamble.

---

### Failure 7: BYOK Cryptographic Protocol Detail
- **Question**: What cryptographic validation standard is required for Hardware Security Modules (HSM) under Cloud Security Alliance guidance for BYOK?
- **Language**: English (en)
- **Answerable**: Yes
- **Expected Evidence**: `doc_en_08_p02_c01` ("Dedicated Hardware Security Modules (HSM) certified to FIPS 140-2 Level 3; key rotation annually.")
- **Baseline Dense Retrieval**: `doc_en_08_p02_c01` (Dense score: 0.6738, Rank 1)
- **Enhanced Hybrid Retrieval**: `doc_en_08_p02_c01` (RRF score: 0.0328, Rank 1)
- **Generation Behavior**: Correctly cited FIPS 140-2 Level 3 certification but omitted the secondary automated annual key rotation schedule.
- **Failure Category**: Incomplete multi-parameter specification.
- **Root Cause**: The generator prioritized the hardware standard and omitted the maintenance lifecycle cadence.
- **Next Fix**: Prompt constraint instructing the model to extract all numeric lifecycle parameters alongside hardware ratings.

---

### Failure 8: Lexical Overlap Penalty on Regulatory Timeframe
- **Question**: What is the mandatory timeframe for notifying a supervisory authority of a data breach under GDPR Article 33, and what are the maximum administrative fines under Article 83?
- **Language**: English (en)
- **Answerable**: Yes
- **Expected Evidence**: `doc_en_09_p02_c01` (72 hours notification) and `doc_en_09_p03_c01` (20 million EUR or 4% global turnover)
- **Baseline Dense Retrieval**: `doc_en_09_p02_c01` (Dense score: 0.6990, Rank 1)
- **Enhanced Hybrid Retrieval**: `doc_en_09_p03_c01` (RRF score: 0.0325, Rank 1)
- **Generation Behavior**: Correctly cited 72 hours and fine provisions; automated word overlap evaluator scored partial due to variation in phrasing of the fine provisions.
- **Failure Category**: Evaluation string overlap strictness.
- **Root Cause**: String-based evaluator penalized alternative wording of the 20 million EUR / 4% turnover penalty clause.
- **Next Fix**: Incorporate semantic equivalence evaluation or LLM-as-a-judge for complex regulatory fine clauses.

---

### Failure 9: Operational Lifecycle Minor Phrasing Variance
- **Question**: What are the specific evidence acquisition tools and chain of custody hashing standards required in the CISA incident playbook?
- **Language**: English (en)
- **Answerable**: Yes
- **Expected Evidence**: `doc_en_11_p02_c01` (Cryptographic verification via SHA-256; forensic disk imaging)
- **Baseline Dense Retrieval**: `doc_en_11_p02_c01` (Dense score: 0.6597, Rank 1)
- **Enhanced Hybrid Retrieval**: `doc_en_11_p03_c01` (RRF score: 0.0328, Rank 1)
- **Generation Behavior**: Correctly identified SHA-256 cryptographic verification; word overlap scored partial due to variation in technical forensic terminology.
- **Failure Category**: Evaluation lexical overlap mismatch.
- **Root Cause**: The answer contained correct factual content but used synonyms for acquisition tools that did not match the strict gold vocabulary set.
- **Next Fix**: Expand reference gold synonyms to accept standard cybersecurity tool equivalencies.

---

### Failure 10: Cross-Lingual Sparse Mismatch Refusal
- **Question**: According to the Saudi Cloud Computing Regulatory Policy, where must sovereign and restricted government data and their encryption keys be hosted?
- **Language**: English (en)
- **Answerable**: Yes (Cross-lingual)
- **Expected Evidence**: `doc_ar_06_p01_c01` ("يجب أن تكون مراكز البيانات التي تستضيف بيانات حكومية أو بيانات مقيدة ومفاتيح تشفيرها موجودة جغرافياً داخل حدود المملكة.")
- **Baseline Dense Retrieval**: `doc_ar_06_p01_c01` (Dense score: 0.5766, Rank 1)
- **Enhanced Hybrid Retrieval**: `doc_en_08_p02_c01` (RRF score: 0.0325, Rank 1)
- **Generation Behavior**: The query was refused as unanswerable.
- **Failure Category**: Cross-lingual lexical interference.
- **Root Cause**: BM25 has zero vocabulary overlap between English queries and Arabic text. English keywords like "cloud computing" and "security" matched English documents (`doc_en_08`), dragging irrelevant English chunks above the true Arabic passage in RRF fusion, triggering a confidence refusal.
- **Next Fix**: Language-aware RRF weighting: disable or downweight BM25 ($w_{\text{bm25}} = 0.0$) when query language differs from target document corpus, relying entirely on multilingual dense embeddings for cross-lingual queries.
