import os
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")

MAP_PROMPT = PromptTemplate(
    template="""Summarize this section of a legal/contract document concisely:
{text}
SECTION SUMMARY:""",
    input_variables=["text"],
)

REDUCE_PROMPT = PromptTemplate(
    template="""You are summarizing a legal document. 
Combine these section summaries into one comprehensive document summary.
Include: parties involved, key obligations, important dates/deadlines, 
payment terms, termination clauses, and any red flags.
SECTION SUMMARIES:
{text}
FINAL COMPREHENSIVE SUMMARY:""",
    input_variables=["text"],
)

def summarize_document(text: str, max_words: int = 500) -> dict:
    """
    Summarize a document using map-reduce strategy:
    1. Split into chunks
    2. Summarize each chunk (map)
    3. Combine summaries (reduce)
    """
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=GROQ_API_KEY,
        temperature=0.2,
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=100,
    )
    chunks = splitter.split_text(text)
    if len(chunks) == 1:
        prompt = f"""Summarize this legal/contract document in under {max_words} words.
Include: parties, obligations, key terms, dates, payment info, and red flags.
DOCUMENT:
{text}
SUMMARY:"""
        response = llm.invoke(prompt)
        return {
            "summary": response.content,
            "strategy": "direct",
            "chunks_processed": 1,
        }
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        print(f"   Summarizing chunk {i+1}/{len(chunks)}...")
        prompt = MAP_PROMPT.format(text=chunk)
        response = llm.invoke(prompt)
        chunk_summaries.append(response.content)
    combined = "\n\n---\n\n".join(
        [f"Section {i+1}:\n{s}" for i, s in enumerate(chunk_summaries)]
    )
    final_prompt = REDUCE_PROMPT.format(text=combined)
    final_response = llm.invoke(final_prompt)
    return {
        "summary": final_response.content,
        "strategy": "map-reduce",
        "chunks_processed": len(chunks),
    }