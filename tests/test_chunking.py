"""
Unit Tests for Chunking Pipeline
"""

import pytest
from src.chunk import count_tokens, split_into_paragraphs, chunk_document_page, chunk_document


def test_count_tokens():
    en_text = "This is a simple test sentence for token counting."
    ar_text = "هذه جملة اختبارية بسيطة لحساب عدد الرموز في النص العربي."
    
    en_count = count_tokens(en_text)
    ar_count = count_tokens(ar_text)
    
    assert en_count > 0
    assert ar_count > 0
    assert isinstance(en_count, int)
    assert isinstance(ar_count, int)


def test_split_into_paragraphs():
    raw_text = (
        "المادة الأولى: النطاق العام للسياسة\n"
        "تسري السياسة على كافة الوزارات.\n\n"
        "المادة الثانية: فترات الاحتفاظ\n"
        "يجب الاحتفاظ بالسجلات لمدة خمس سنوات."
    )
    paras = split_into_paragraphs(raw_text)
    assert len(paras) == 2
    assert "المادة الأولى" in paras[0]
    assert "المادة الثانية" in paras[1]


def test_chunk_document_page_constraints():
    page_data = {
        "page": 1,
        "text": "المادة الأولى: التعريفات.\nتنطبق هذه الضوابط على كافة الجهات الحكومية والشركات التابعة."
    }
    doc_metadata = {
        "document_id": "doc_test_01",
        "title": "وثيقة اختبارية",
        "language": "ar",
        "source_url": "https://example.com/test.pdf"
    }

    chunks = chunk_document_page(
        page_data=page_data,
        doc_metadata=doc_metadata,
        chunk_size=500,
        overlap=75
    )

    assert len(chunks) >= 1
    chunk = chunks[0]
    assert chunk["chunk_id"] == "doc_test_01_p01_c01"
    assert chunk["document_id"] == "doc_test_01"
    assert chunk["page"] == 1
    assert chunk["language"] == "ar"
    assert len(chunk["text"]) > 0
    assert chunk["token_count"] > 0
    assert "source_url" in chunk


def test_chunk_document_multiple_pages():
    pages = [
        {"page": 1, "text": "Page one text content here."},
        {"page": 2, "text": "Page two text content with more guidelines."}
    ]
    doc_metadata = {
        "document_id": "doc_test_en",
        "title": "Test Document EN",
        "language": "en",
        "source_url": "https://example.com/en.pdf"
    }

    chunks = chunk_document(pages, doc_metadata, chunk_size=500, overlap=75)
    assert len(chunks) == 2
    assert chunks[0]["chunk_id"] == "doc_test_en_p01_c01"
    assert chunks[1]["chunk_id"] == "doc_test_en_p02_c01"
