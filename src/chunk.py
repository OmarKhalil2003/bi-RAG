"""
Chunking Module
Bilingual Document Q&A (RAG) System

Implements paragraph-aware, token-constrained chunking with overlap
and metadata preservation (document_id, title, language, page, source_url, chunk_id).
Default: 500 tokens with 75-token overlap.
"""

import re
from typing import List, Dict, Any, Optional

_TOKENIZER = None

def get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        try:
            from transformers import AutoTokenizer
            _TOKENIZER = AutoTokenizer.from_pretrained("BAAI/bge-m3")
        except Exception:
            _TOKENIZER = False
    return _TOKENIZER if _TOKENIZER is not False else None


def count_tokens(text: str) -> int:
    """Count tokens using BGE-M3 tokenizer if available, else approximate."""
    tok = get_tokenizer()
    if tok is not None:
        return len(tok.encode(text, add_special_tokens=False))
    # Fast fallback: ~1 token per 4 chars for EN, ~1 token per 3 chars for AR
    words = text.split()
    return int(len(words) * 1.3) + 1


def split_into_paragraphs(text: str) -> List[str]:
    """Split text into distinct paragraphs or logical sections."""
    lines = [p.strip() for p in text.split("\n") if p.strip()]
    paragraphs = []
    current_para = []
    
    for line in lines:
        # Check if line looks like a header or new article
        is_header = bool(re.match(r"^(المادة|المجال|الفصل|الباب|المحور|القسم|Section|Chapter|Clause|Annex|Principle|Domain|Subpart|Article)", line))
        if is_header and current_para:
            paragraphs.append(" ".join(current_para))
            current_para = [line]
        else:
            current_para.append(line)
            
    if current_para:
        paragraphs.append(" ".join(current_para))
        
    return paragraphs if paragraphs else [text]


def chunk_document_page(
    page_data: Dict[str, Any],
    doc_metadata: Dict[str, Any],
    chunk_size: int = 500,
    overlap: int = 75
) -> List[Dict[str, Any]]:
    """
    Chunk text from a single page respecting paragraph boundaries and overlap.
    Retains page boundary metadata.
    """
    page_num = page_data["page"]
    page_text = page_data["text"]
    doc_id = doc_metadata["document_id"]
    title = doc_metadata["title"]
    language = doc_metadata["language"]
    source_url = doc_metadata["source_url"]

    paragraphs = split_into_paragraphs(page_text)
    chunks = []
    
    current_chunk_paras = []
    current_tokens = 0
    chunk_idx = 1
    
    for para in paragraphs:
        para_tokens = count_tokens(para)
        
        # If adding this paragraph exceeds chunk_size and we already have content
        if current_tokens + para_tokens > chunk_size and current_chunk_paras:
            chunk_text = "\n\n".join(current_chunk_paras).strip()
            chunk_id = f"{doc_id}_p{page_num:02d}_c{chunk_idx:02d}"
            chunks.append({
                "chunk_id": chunk_id,
                "document_id": doc_id,
                "title": title,
                "language": language,
                "page": page_num,
                "text": chunk_text,
                "source_url": source_url,
                "token_count": count_tokens(chunk_text)
            })
            chunk_idx += 1
            
            # Carry over overlap if possible
            overlap_paras = []
            overlap_tokens = 0
            for p in reversed(current_chunk_paras):
                p_tok = count_tokens(p)
                if overlap_tokens + p_tok <= overlap:
                    overlap_paras.insert(0, p)
                    overlap_tokens += p_tok
                else:
                    break
            current_chunk_paras = overlap_paras + [para]
            current_tokens = overlap_tokens + para_tokens
        else:
            current_chunk_paras.append(para)
            current_tokens += para_tokens
            
    if current_chunk_paras:
        chunk_text = "\n\n".join(current_chunk_paras).strip()
        chunk_id = f"{doc_id}_p{page_num:02d}_c{chunk_idx:02d}"
        chunks.append({
            "chunk_id": chunk_id,
            "document_id": doc_id,
            "title": title,
            "language": language,
            "page": page_num,
            "text": chunk_text,
            "source_url": source_url,
            "token_count": count_tokens(chunk_text)
        })
        
    return chunks


def chunk_document(
    pages: List[Dict[str, Any]],
    doc_metadata: Dict[str, Any],
    chunk_size: int = 500,
    overlap: int = 75
) -> List[Dict[str, Any]]:
    """Chunk all pages of a document while preserving page boundaries."""
    doc_chunks = []
    for page in pages:
        page_chunks = chunk_document_page(
            page,
            doc_metadata,
            chunk_size=chunk_size,
            overlap=overlap
        )
        doc_chunks.extend(page_chunks)
    return doc_chunks
