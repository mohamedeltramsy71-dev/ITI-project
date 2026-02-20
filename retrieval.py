from typing import List, Tuple
from langchain_core.documents import Document
from langchain_chroma import Chroma

def retrieve_relevant_chunks(
    vectorstore: Chroma,
    query: str,
    top_k: int = 5,
    score_threshold: float = 0.3,
) -> List[Tuple[Document, float]]:
    """
    Retrieve top-k most relevant chunks using semantic similarity.
    Returns list of (document, score) tuples.
    """
    results = vectorstore.similarity_search_with_relevance_scores(
        query=query,
        k=top_k,
    )
    filtered = [(doc, score) for doc, score in results if score >= score_threshold]
    if not filtered:
        return results[:top_k]
    return filtered

def format_context(
    retrieved_docs: List[Tuple[Document, float]],
    include_sources: bool = True,
) -> Tuple[str, List[dict]]:
    """
    Format retrieved documents into a context string and source citations.
    Returns (context_string, sources_list)
    """
    context_parts = []
    sources = []
    for i, (doc, score) in enumerate(retrieved_docs):
        chunk_text = doc.page_content
        source = doc.metadata.get("source", "Unknown")
        chunk_idx = doc.metadata.get("chunk_index", i)
        context_parts.append(
            f"[Source {i+1} | {source} | Chunk {chunk_idx+1}]\n{chunk_text}"
        )
        sources.append({
            "source_id": i + 1,
            "file": source,
            "chunk": chunk_idx + 1,
            "relevance_score": round(score, 3),
            "excerpt": chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text,
        })
    context = "\n\n---\n\n".join(context_parts)
    return context, sources