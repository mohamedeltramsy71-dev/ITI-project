import os
import pytest
from unittest.mock import patch, MagicMock
from src.ingestion import chunk_text, extract_text_from_pdf
from src.retrieval import format_context

def test_chunk_text():
    text = "This is a test document. " * 100
    chunks = chunk_text(text, "test.pdf", chunk_size=200, chunk_overlap=50)
    assert len(chunks) > 0
    assert all(hasattr(c, "page_content") for c in chunks)
    assert all(c.metadata["source"] == "test.pdf" for c in chunks)

def test_chunk_metadata():
    text = "Contract clause. " * 50
    chunks = chunk_text(text, "contract.pdf")
    for i, chunk in enumerate(chunks):
        assert chunk.metadata["chunk_index"] == i
        assert chunk.metadata["total_chunks"] == len(chunks)

def test_format_context():
    from langchain.schema import Document
    docs = [
        (Document(page_content="Clause 1", metadata={"source": "doc.pdf", "chunk_index": 0}), 0.9),
        (Document(page_content="Clause 2", metadata={"source": "doc.pdf", "chunk_index": 1}), 0.7),
    ]
    context, sources = format_context(docs)
    assert "Clause 1" in context
    assert len(sources) == 2
    assert sources[0]["relevance_score"] == 0.9

if __name__ == "__main__":
    pytest.main([__file__, "-v"])