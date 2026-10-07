"""
Generation Module
Bilingual Document Q&A (RAG) System

Handles:
- Language detection (Arabic vs English)
- Retrieval confidence threshold gating
- Context formatting with chunk IDs
- LLM prompt construction with citation requirements
- OpenRouter API free tier client
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
        "You answer questions only from the provided document passages.\n\n"
        "Rules:\n"
        "1. Answer in the language of the user's question.\n"
        "2. Use only the provided passages.\n"
        "3. If the passages do not contain enough information, say that the answer is not available in the provided documents.\n"
        "4. Never invent facts.\n"
        "5. Cite every substantive claim using [source: chunk_id].\n"
        "6. Do not cite a source that does not support the claim."
    )


def build_user_prompt(question: str, context: str) -> str:
    return (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{question}\n\n"
        "Provide a concise, direct answer citing the supporting chunk ID in square brackets [source: chunk_id]."
    )


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
        "not available in the documents"
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
        # Fallback to plain chunk IDs
        matches = re.findall(r"\b(doc_[a-z0-9_]+_p\d+_c\d+)\b", answer_text)
    return list(dict.fromkeys(matches))


class Generator:
    """Generator orchestrator with confidence gate and OpenRouter client."""

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

        # 1. Retrieval confidence gate
        top_score = 0.0
        if retrieved_chunks:
            # Check reranker_score or score
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
        answer = self._call_openrouter(system_prompt, user_prompt)
        
        # 4. Check LLM refusal
        refused = is_refusal(answer)
        citations = extract_citations(answer)

        # 5. Map cited sources or return top chunks as sources
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
        # If no citation parsed or refused, provide top chunks as evidence
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
            "answer": answer,
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
            "max_tokens": 512
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
                        # Strip thinking process if emitted by reasoning model
                        if "Here's a thinking process:" in content:
                            sub_parts = content.split("\n\n")
                            # Filter out thinking steps
                            clean_parts = [p for p in sub_parts if not p.strip().startswith("Here's a thinking process:") and not p.strip().startswith("1.  **Analyze") and not p.strip().startswith("2.  **Examine") and not p.strip().startswith("3.  **Formulate")]
                            if clean_parts:
                                content = "\n\n".join(clean_parts).strip()
                        return content.strip() if content else "The answer is not available in the provided documents."
                    return "The answer is not available in the provided documents."
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
