"""
Ingestion Pipeline
Bilingual Document Q&A (RAG) System

Extracts text from raw PDF documents using PyMuPDF (fitz),
performs Unicode NFKC normalization and clean-up,
chunks the text using token-constrained chunking,
and writes chunks to data/chunks.jsonl and data/processed/.
"""

import os
import re
import csv
import json
import unicodedata
from typing import List, Dict, Any, Tuple
import fitz

from src.chunk import chunk_document


def clean_extracted_text(text: str, language: str) -> str:
    """Clean PDF extraction noise while preserving structure and Arabic semantics."""
    # Unicode NFKC normalization to standard forms
    norm_text = unicodedata.normalize("NFKC", text)
    
    # Remove null bytes or invisible control chars
    norm_text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", norm_text)

    # Ensure clean spacing between numerals and Arabic/Latin letters to prevent token concatenation
    norm_text = re.sub(r"([\u0600-\u06FFa-zA-Z])(\d+)", r"\1 \2", norm_text)
    norm_text = re.sub(r"(\d+)([\u0600-\u06FFa-zA-Z])", r"\1 \2", norm_text)
    
    # Clean leading dots or artifacts created by RTL text extraction
    lines = []
    for line in norm_text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Remove header/footer line metadata (e.g. Page X of Y | Document ID...)
        if re.search(r"(كود المستند|Document ID:|Page \d+ of \d+|الصفحة \d+ من \d+)", line):
            continue
        # Strip leading dot/comma from RTL rendering artifact if present at start of line
        if line.startswith(".") and len(line) > 1 and not line.startswith(".."):
            line = line[1:].strip() + "."
        lines.append(line)
        
    cleaned = "\n".join(lines)
    return cleaned


def extract_pages_from_pdf(pdf_path: str, language: str) -> List[Dict[str, Any]]:
    """Extract text from each page of a PDF using PyMuPDF."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
    doc = fitz.open(pdf_path)
    pages = []
    for page_idx, page in enumerate(doc, start=1):
        raw_text = page.get_text("text")
        cleaned_text = clean_extracted_text(raw_text, language)
        if cleaned_text:
            pages.append({
                "page": page_idx,
                "text": cleaned_text
            })
    doc.close()
    return pages


def run_ingestion(
    manifest_path: str = "data/corpus_manifest.csv",
    processed_dir: str = "data/processed",
    output_chunks_path: str = "data/chunks.jsonl",
    chunk_size: int = 500,
    overlap: int = 75
) -> Dict[str, Any]:
    """
    Run full ingestion pipeline over the corpus manifest.
    Writes chunks to chunks.jsonl and per-document JSON in data/processed/.
    """
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_chunks_path), exist_ok=True)

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        manifest_entries = list(reader)

    all_chunks: List[Dict[str, Any]] = []
    doc_stats = []

    for entry in manifest_entries:
        doc_id = entry["document_id"]
        lang = entry["language"]
        local_path = entry["local_path"]

        # Handle path separators cleanly across OS
        normalized_path = os.path.normpath(local_path)
        pages = extract_pages_from_pdf(normalized_path, lang)
        
        # Save processed pages
        processed_doc_file = os.path.join(processed_dir, f"{doc_id}.json")
        with open(processed_doc_file, "w", encoding="utf-8") as f:
            json.dump({
                "document_id": doc_id,
                "title": entry["title"],
                "language": lang,
                "document_type": entry["document_type"],
                "source_url": entry["source_url"],
                "pages": pages
            }, f, ensure_ascii=False, indent=2)

        # Chunk document
        chunks = chunk_document(
            pages=pages,
            doc_metadata=entry,
            chunk_size=chunk_size,
            overlap=overlap
        )
        all_chunks.extend(chunks)
        doc_stats.append({
            "document_id": doc_id,
            "language": lang,
            "pages": len(pages),
            "chunks": len(chunks)
        })

    # Write all chunks to JSONL
    with open(output_chunks_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    ar_chunks = [c for c in all_chunks if c["language"] == "ar"]
    en_chunks = [c for c in all_chunks if c["language"] == "en"]
    avg_tokens = sum(c.get("token_count", 0) for c in all_chunks) / len(all_chunks) if all_chunks else 0

    stats = {
        "total_documents": len(manifest_entries),
        "total_chunks": len(all_chunks),
        "arabic_chunks": len(ar_chunks),
        "english_chunks": len(en_chunks),
        "average_tokens_per_chunk": round(avg_tokens, 2),
        "chunks_output_path": output_chunks_path
    }
    return stats


if __name__ == "__main__":
    print("Starting corpus ingestion...")
    stats = run_ingestion()
    print("Ingestion complete!")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
