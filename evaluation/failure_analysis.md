# Failure Analysis: 10 Representative Cases
Bilingual Document Q&A (RAG) System

This document analyzes exactly **10 representative failures** observed during the evaluation of the baseline and improved retrieval pipelines and generation modules on the fixed `evaluation/gold.jsonl` benchmark.

---

### Failure 1: Arabic Numeral Extraction Boundary Artifact
- **Question**: ما هي الفترة الزمنية الإلزامية للاحتفاظ بسجلات التدقيق الأمني وسجلات الوصول للبيانات وفق سياسة حوكمة البيانات الوطنية؟
- **Language**: Arabic (ar)
- **Answerable**: Yes
- **Expected evidence**: `doc_ar_01_p02_c01` ("تلتزم الجهة بالاحتفاظ بسجلات التدقيق الأمني وسجلات الوصول للبيانات لمدة لا تقل عن 5 سنوات تقويمية كاملة.")
- **What baseline retrieved**: `doc_ar_01_p02_c01` (Dense score: 0.6827, Rank 1)
- **What improved retrieval retrieved**: `doc_ar_01_p02_c01` (Reranker score: 0.9987, Rank 1)
- **Generation behavior**: The LLM correctly stated the answer ("لا تقل عن 5 سنوات تقويمية كاملة [source: doc_ar_01_p02_c01]"), but automated boundary matching flagged numeral `5` as an unsupported claim.
- **Failure category**: OCR/PDF extraction problem / token boundary issue
- **Root cause**: In the extracted PDF text stream, the digit `5` was concatenated directly to the preceding Arabic text (`كاملة5`) due to bidirectional rendering layout, preventing isolated word-boundary detection (`\b5\b`).
- **Next fix**: Apply regex-based character boundary spacing during PDF text ingestion (`re.sub(r'([\u0600-\u06FF])(\d+)', r'\1 \2', text)`) to ensure clean tokenization around numerals.

---

### Failure 2: Free Tier API Null Content Payload
- **Question**: ما هي شروط طول كلمة المرور واستخدام المصادقة متعددة العوامل في ضوابط الأمن السيبراني الأساسية (ECC-1:2018)؟
- **Language**: Arabic (ar)
- **Answerable**: Yes
- **Expected evidence**: `doc_ar_02_p01_c01` ("يجب تطبيق متطلبات المصادقة متعددة العوامل (MFA)... يجب ألا يقل الحد الأدنى لطول كلمات المرور عن 12 خانة")
- **What baseline retrieved**: `doc_ar_02_p01_c01` (Dense score: 0.7191, Rank 1)
- **What improved retrieval retrieved**: `doc_ar_02_p01_c01` (Reranker score: 0.9982, Rank 1)
- **Generation behavior**: Generation failed with `'NoneType' object has no attribute 'strip'` error.
- **Failure category**: Upstream API transient error
- **Root cause**: OpenRouter free-tier endpoint returned an HTTP 200 response where `choices[0]['message']['content']` was `None` (reasoning model state without final content).
- **Next fix**: Add defensive null-handling for `choices[0]['message']['content']` and fall back to `reasoning_content` or a secondary open-source model fallback.

---

### Failure 3: Incomplete Enumeration via Token Limit
- **Question**: ما هي المبادئ الأخلاقية للذكاء الاصطناعي وما هو الإجراء المطلوب قبل الإطلاق التجاري للنماذج؟
- **Language**: Arabic (ar)
- **Answerable**: Yes
- **Expected evidence**: `doc_ar_03_p01_c01` (المبادئ السبعة للذكاء الاصطناعي واختبارات التحيز الإحصائي قبل الإطلاق)
- **What baseline retrieved**: `doc_ar_03_p01_c01` (Dense score: 0.6620, Rank 1)
- **What improved retrieval retrieved**: `doc_ar_03_p01_c01` (Reranker score: 0.9994, Rank 1)
- **Generation behavior**: The model started generating the list ("المبادئ الأخلاقية الخمسة للذكاء الاصطناعي وفق الإطار الوطني: 1...") but truncated after item 1.
- **Failure category**: Insufficient context / output token exhaustion
- **Root cause**: The `max_tokens` budget (512) was exhausted by preamble and markdown formatting, causing generation truncation.
- **Next fix**: Increase `max_tokens` to 1024 and instruct the prompt to generate concise, bulleted summaries without verbose preambles.

---

### Failure 4: API Rate Throttling under Sequential Evaluation
- **Question**: ما هو الحد السنوي المستهدف لتركيز الجسيمات العالقة PM2.5 في تقرير جودة الهواء وما هي مستهدفات الحافلات الكهربائية بحلول 2030؟
- **Language**: Arabic (ar)
- **Answerable**: Yes
- **Expected evidence**: `doc_ar_05_p01_c01` (PM2.5 أقل من 15 ميكروغرام) and `doc_ar_05_p02_c01` (30% حافلات كهربائية)
- **What baseline retrieved**: `doc_ar_05_p01_c01` (Dense score: 0.6517, Rank 1)
- **What improved retrieval retrieved**: `doc_ar_05_p01_c01` (Reranker score: 0.9944, Rank 1)
- **Generation behavior**: Returned network error payload due to provider rate limiting.
- **Failure category**: Upstream API rate limit
- **Root cause**: Burst evaluation queries exceeded the requests-per-minute quota on the free OpenRouter tier.
- **Next fix**: Implement exponential backoff with jitter and client-side rate limiting (1.5s delay between evaluation requests).

---

### Failure 5: Chain-of-Thought (CoT) Scratchpad Leakage
- **Question**: كم بلغت نسبة النضج الرقمي الحكومي العام ونسبة الخدمات الحكومية المرقمنة وفق تقرير التحول الرقمي؟
- **Language**: Arabic (ar)
- **Answerable**: Yes
- **Expected evidence**: `doc_ar_12_p01_c01` (مؤشر النضج 86.4%، نسبة رقمنة الخدمات 92%)
- **What baseline retrieved**: `doc_ar_12_p01_c01` (Dense score: 0.7899, Rank 1)
- **What improved retrieval retrieved**: `doc_ar_12_p01_c01` (Reranker score: 0.9998, Rank 1)
- **Generation behavior**: Answer included raw reasoning scratchpad ("Here's a thinking process: 1. Analyze User Input...").
- **Failure category**: Generation formatting failure / CoT leakage
- **Root cause**: OpenRouter routed the request to a reasoning model that emits thought tokens directly into the response stream.
- **Next fix**: Add a regex post-filter in `app/generation.py` to strip blocks starting with `Here's a thinking process:` or `<think>...</think>`.

---

### Failure 6: Upstream Safety Guardrail False Positive
- **Question**: Under NIST SP 800-53 Section AC-2, after how many days of inactivity must system accounts be automatically disabled?
- **Language**: English (en)
- **Answerable**: Yes
- **Expected evidence**: `doc_en_01_p01_c01` ("automatically disabling inactive system accounts after a maximum inactivity period of 90 days.")
- **What baseline retrieved**: `doc_en_01_p01_c01` (Dense score: 0.7635, Rank 1)
- **What improved retrieval retrieved**: `doc_en_01_p01_c01` (Reranker score: 0.9994, Rank 1)
- **Generation behavior**: The model returned "User Safety: safe" instead of answering the query.
- **Failure category**: Generation safety classifier false positive
- **Root cause**: Keywords "account disabling" and "inactivity" triggered the upstream provider's automated content moderation template.
- **Next fix**: Neutralize prompt phrasing or use explicit system role markers to frame the task as regulatory audit compliance Q&A.

---

### Failure 7: Reasoning Preamble Introducing Extraneous Numerals
- **Question**: What cryptographic validation standard is required for Hardware Security Modules (HSM) under Cloud Security Alliance guidance for BYOK?
- **Language**: English (en)
- **Answerable**: Yes
- **Expected evidence**: `doc_en_08_p02_c01` ("Dedicated Hardware Security Modules (HSM) certified to FIPS 140-2 Level 3.")
- **What baseline retrieved**: `doc_en_08_p02_c01` (Dense score: 0.6738, Rank 1)
- **What improved retrieval retrieved**: `doc_en_08_p02_c01` (Reranker score: 0.9815, Rank 1)
- **Generation behavior**: Correct answer was provided but preceded by numbered chain-of-thought steps ("1. Analyze User Input...").
- **Failure category**: Prompt leakage / evaluator noise
- **Root cause**: Automated evaluation scored the answer as partially incorrect because numbered reasoning steps introduced digits not in the gold answer.
- **Next fix**: Add prompt rule: "Do not output reasoning steps or thinking process; output only the final direct answer."

---

### Failure 8: Regulatory Penalty Keyword Triggering Generic Moderation
- **Question**: What is the mandatory timeframe for notifying a supervisory authority of a data breach under GDPR Article 33, and what are the maximum administrative fines under Article 83?
- **Language**: English (en)
- **Answerable**: Yes
- **Expected evidence**: `doc_en_09_p02_c01` (72 hours notification) and `doc_en_09_p03_c01` (20 million EUR or 4% turnover)
- **What baseline retrieved**: `doc_en_09_p02_c01` (Dense score: 0.6990, Rank 1)
- **What improved retrieval retrieved**: `doc_en_09_p02_c01` (Reranker score: 0.9852, Rank 1)
- **Generation behavior**: Returned moderation template string ("User Safety: safe").
- **Failure category**: Model moderation refusal
- **Root cause**: Combination of "data breach" and "fines" triggered generic safety template reply on the free tier endpoint.
- **Next fix**: Add an extractive fallback that parses evidence sentences directly when the LLM returns an unexpected non-answer string.

---

### Failure 9: Premature Upstream Socket Termination
- **Question**: What are the five operational phases of the incident response lifecycle in the CISA playbook, and what hashing algorithm is required for forensic images?
- **Language**: English (en)
- **Answerable**: Yes
- **Expected evidence**: `doc_en_11_p01_c01` (Five phases: Preparation, Detection & Analysis, Containment, Eradication & Recovery, Post-Incident) and `doc_en_11_p02_c01` (SHA-256 hashing)
- **What baseline retrieved**: `doc_en_11_p01_c01` (Dense score: 0.6978, Rank 1)
- **What improved retrieval retrieved**: `doc_en_11_p01_c01` (Reranker score: 0.9953, Rank 1)
- **Generation behavior**: Failed with network read timeout.
- **Failure category**: Infrastructure network timeout
- **Root cause**: Free tier API dropped the TCP socket during peak usage hours.
- **Next fix**: Use persistent HTTP session with `HTTPAdapter(max_retries=3)` and fallback to local small LLM (e.g., Qwen1.5-0.5B-Chat).

---

### Failure 10: Cross-Lingual Semantic Distance Attenuation
- **Question**: According to the Saudi Cloud Computing Regulatory Policy, where must sovereign and restricted government data and their encryption keys be hosted?
- **Language**: English (en)
- **Answerable**: Yes (Cross-lingual)
- **Expected evidence**: `doc_ar_06_p01_c01` ("يجب أن تكون مراكز البيانات التي تستضيف بيانات حكومية أو بيانات مقيدة ومفاتيح تشفيرها موجودة جغرافياً داخل حدود المملكة.")
- **What baseline retrieved**: `doc_ar_06_p01_c01` (Dense score: 0.5766, Rank 1)
- **What improved retrieval retrieved**: `doc_ar_06_p01_c01` (Reranker score: 0.3797, Rank 1)
- **Generation behavior**: Top reranker score (0.3797) was significantly lower than monolingual queries (~0.98), and English generation over Arabic context was less assertive.
- **Failure category**: Cross-lingual semantic gap
- **Root cause**: Cross-lingual dense embeddings exhibit slightly larger cosine distance than monolingual equivalents; the cross-encoder gave 0.3797 compared to >0.95 for monolingual pairs.
- **Next fix**: Augment cross-lingual queries by appending bilingual key domain terms (e.g., "استضافة البيانات السحابية / cloud data residency") before dense retrieval and reranking.
