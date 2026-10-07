"""
Generation Module
Bilingual Document Q&A (RAG) System

Handles:
- Canonical language detection (Arabic vs English)
- Retrieval confidence threshold gating
- Dynamic context formatting with chunk provenance
- LLM prompt construction with citation requirements
- Scratchpad / Chain-of-thought sanitization
- OpenRouter API client with defensive extractive fallback
- Source citations parsing and refusal detection
"""

import os
import re
import json
import time
from typing import List, Dict, Any, Optional, Tuple
import requests

DEFAULT_MODEL = os.getenv("GENERATION_MODEL", "openrouter/free")
OPENROUTER_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
CONFIDENCE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.10"))

REFUSAL_MESSAGE_AR = "الإجابة غير متوفرة في المستندات المقدمة."
REFUSAL_MESSAGE_EN = "The answer is not available in the provided documents."


def detect_language(text: str) -> str:
    """Detect if text is primarily Arabic or English."""
    arabic_chars = len(re.findall(r"[\u0600-\u06FF]", text))
    latin_chars = len(re.findall(r"[a-zA-Z]", text))
    return "ar" if arabic_chars >= latin_chars and arabic_chars > 0 else "en"


def format_context(chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into numbered context blocks with chunk IDs."""
    context_blocks = []
    for c in chunks:
        cid = c.get("chunk_id", "unknown")
        title = c.get("title", "")
        text = c.get("text", "")
        context_blocks.append(f"[chunk_id={cid} | title={title}]\n{text}")
    return "\n\n".join(context_blocks)


def build_system_prompt() -> str:
    return (
        "You are an expert bilingual factual Q&A assistant for regulatory and technical documents.\n\n"
        "Instructions:\n"
        "1. Answer strictly in the language of the user's question (Arabic for Arabic questions, English for English questions).\n"
        "2. Use only the provided document passages. Never invent facts or extrapolate beyond the text.\n"
        "3. If the provided passages do not contain sufficient evidence to answer, state clearly that the answer is not available in the provided documents.\n"
        "4. Support every factual claim, numeric figure, or timeframe with an explicit citation in square brackets: [source: chunk_id].\n"
        "5. Output only the direct answer with citations. Do not include conversational filler, meta-announcements, or chain-of-thought scratchpads."
    )


def build_user_prompt(question: str, context: str) -> str:
    return (
        f"DOCUMENT PASSAGES:\n{context}\n\n"
        f"USER QUESTION:\n{question}\n\n"
        "Provide a direct, concise factual answer citing supporting chunk IDs as [source: chunk_id]."
    )


def clean_llm_response(text: str) -> str:
    """Strip chain-of-thought traces, thinking blocks, and safety template strings."""
    if not text:
        return ""

    # Strip XML thinking blocks e.g. <think>...</think>
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)

    # Strip thinking process prefixes
    if "Here's a thinking process:" in text:
        sub_parts = text.split("\n\n")
        clean_parts = [
            p for p in sub_parts
            if not p.strip().startswith("Here's a thinking process:")
            and not re.match(r"^\d+\.\s+\*\*Analyze", p.strip())
            and not re.match(r"^\d+\.\s+\*\*Examine", p.strip())
            and not re.match(r"^\d+\.\s+\*\*Formulate", p.strip())
        ]
        if clean_parts:
            text = "\n\n".join(clean_parts).strip()

    # Strip upstream safety classification strings
    text = re.sub(r"^User Safety:\s*safe\s*", "", text, flags=re.IGNORECASE).strip()

    return text.strip()


def is_refusal(answer_text: str) -> bool:
    """Detect whether the answer is an explicit refusal."""
    refusal_keywords_en = [
        "not available in the provided documents",
        "not mentioned in the provided documents",
        "not found in the provided",
        "does not contain",
        "do not contain",
        "no information provided",
        "cannot be answered from the provided",
        "not available in the documents",
        "not addressed in the provided"
    ]
    refusal_keywords_ar = [
        "غير متوفرة في المستندات",
        "غير مذكورة في المستندات",
        "لا تحتوي المستندات",
        "لا توجد معلومات",
        "غير متوفر في الوثائق",
        "الإجابة غير متوفرة",
        "لم يرد في المستندات"
    ]
    lower_ans = answer_text.lower()
    for kw in refusal_keywords_en:
        if kw in lower_ans:
            return True
    for kw in refusal_keywords_ar:
        if kw in answer_text:
            return True
    return False


def extract_citations(answer_text: str) -> List[str]:
    """Extract cited chunk IDs matching [source: chunk_id] or [chunk_id]."""
    matches = re.findall(r"\[source:\s*([a-zA-Z0-9_\-]+)\]", answer_text, re.IGNORECASE)
    if not matches:
        matches = re.findall(r"\b(doc_[a-z0-9_]+_p\d+_c\d+)\b", answer_text)
    return list(dict.fromkeys(matches))


def synthesize_extractive_fallback(question: str, top_chunk: Dict[str, Any]) -> str:
    """
    Extractive fallback synthesizing a factual sentence from the top chunk
    if the external LLM returns a network timeout, null content, or false safety flag.
    """
    text = top_chunk.get("raw_text") or top_chunk.get("text", "")
    cid = top_chunk.get("chunk_id", "")
    lines = [line.strip() for line in text.split("\n") if line.strip() and not line.startswith("[")]
    if lines:
        best_sentence = lines[0] if len(lines) == 1 else " ".join(lines[:2])
        return f"{best_sentence} [source: {cid}]"
    return f"{text[:200]}... [source: {cid}]"


class Generator:
    """Generator orchestrator with dual confidence gating and OpenRouter client."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        threshold: float = CONFIDENCE_THRESHOLD
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.model = model
        self.threshold = threshold

    def generate(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        End-to-end generation with confidence gate, prompt assembly, and citation mapping.
        """
        start_time = time.time()
        lang = detect_language(question)

        # 1. Retrieval confidence gate (Gate 1)
        top_score = 0.0
        if retrieved_chunks:
            top_score = retrieved_chunks[0].get("reranker_score", retrieved_chunks[0].get("score", 0.0))

        if not retrieved_chunks or top_score < self.threshold:
            refusal_msg = REFUSAL_MESSAGE_AR if lang == "ar" else REFUSAL_MESSAGE_EN
            return {
                "answer": refusal_msg,
                "refused": True,
                "refusal_reason": f"low_retrieval_confidence (top_score={top_score:.4f} < {self.threshold})",
                "citations": [],
                "sources": [],
                "latency_ms": round((time.time() - start_time) * 1000, 2)
            }

        # 2. Format context
        context_str = format_context(retrieved_chunks)
        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(question, context_str)

        # 3. Call OpenRouter API
        raw_answer = self._call_openrouter(system_prompt, user_prompt)
        cleaned_answer = clean_llm_response(raw_answer)

        # 4. Defensive handling for upstream transient failures / safety triggers
        is_api_failure = (
            not cleaned_answer
            or "Generation request failed" in cleaned_answer
            or "Generation network error" in cleaned_answer
            or "Generation request timed out" in cleaned_answer
            or cleaned_answer.lower() == "safe"
        )
        if is_api_failure and top_score >= 0.70 and retrieved_chunks:
            cleaned_answer = synthesize_extractive_fallback(question, retrieved_chunks[0])

        # 5. Check LLM refusal & citations
        refused = is_refusal(cleaned_answer)
        citations = extract_citations(cleaned_answer)

        # If LLM answered without explicit citation, inject top chunk citation if confidence is high
        if not refused and not citations and retrieved_chunks:
            top_cid = retrieved_chunks[0]["chunk_id"]
            cleaned_answer = f"{cleaned_answer} [source: {top_cid}]"
            citations = [top_cid]

        # 6. Map cited sources or return top chunks as sources
        chunk_map = {c["chunk_id"]: c for c in retrieved_chunks}
        sources = []
        if citations:
            for cid in citations:
                if cid in chunk_map:
                    c = chunk_map[cid]
                    sources.append({
                        "chunk_id": cid,
                        "title": c.get("title", ""),
                        "page": c.get("page", 1),
                        "source_url": c.get("source_url", "")
                    })
        if not sources and not refused and retrieved_chunks:
            top_c = retrieved_chunks[0]
            sources.append({
                "chunk_id": top_c.get("chunk_id", ""),
                "title": top_c.get("title", ""),
                "page": top_c.get("page", 1),
                "source_url": top_c.get("source_url", "")
            })

        latency = round((time.time() - start_time) * 1000, 2)
        return {
            "answer": cleaned_answer,
            "refused": refused,
            "refusal_reason": "llm_evidence_insufficient" if refused else None,
            "citations": citations,
            "sources": sources,
            "latency_ms": latency
        }

    def _call_openrouter(self, system_prompt: str, user_prompt: str) -> str:
        """Call OpenRouter Chat Completion API with retry fallback."""
        if not self.api_key:
            return "API key not configured. Refusing generation."

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/PetroChoice/bilingual-rag",
            "X-Title": "Bilingual Document Q&A RAG"
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 1024
        }

        for attempt in range(2):
            try:
                resp = requests.post(
                    f"{OPENROUTER_URL}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                if resp.status_code == 200:
                    data = resp.json()
                    choices = data.get("choices", [])
                    if choices:
                        msg = choices[0].get("message", {})
                        content = msg.get("content") or msg.get("reasoning_content") or ""
                        return content.strip() if content else ""
                    return ""
                elif resp.status_code == 429:
                    time.sleep(2)
                    continue
                else:
                    return f"Generation request failed with status {resp.status_code}."
            except Exception as e:
                if attempt == 1:
                    return f"Generation network error: {str(e)}"
                time.sleep(1)

        return "Generation request timed out."
