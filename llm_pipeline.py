import os
from typing import List, Tuple, Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_chroma import Chroma
from .retrieval import retrieve_relevant_chunks, format_context
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")

SYSTEM_PROMPT = """You are a professional Smart Contract & Legal Document Assistant.
Your role:
- Answer questions ONLY based on the provided document context.
- Always cite the specific source sections you used.
- If the answer is not found in the context, say "I cannot find this information in the uploaded document."
- Never make up information or use external knowledge to answer contract-specific questions.
- Be precise, professional, and concise.
- Flag any ambiguous or potentially risky clauses.
IMPORTANT GUARDRAILS:
1. Do not provide legal advice - only factual information from the document.
2. Always add disclaimer: "This is not legal advice."
3. If asked about harmful or illegal activities, decline politely.
4. Stick strictly to document content.
Context from the document:
{context}
"""

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

def get_llm(temperature: float = 0.1):
    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=GROQ_API_KEY,
        temperature=temperature,
    )

UNSAFE_PATTERNS = [
    "how to avoid tax", "illegal", "fraud", "money laundering",
    "evade", "bypass law", "illegal clause",
]

def check_guardrails(query: str) -> Tuple[bool, str]:
    query_lower = query.lower()
    for pattern in UNSAFE_PATTERNS:
        if pattern in query_lower:
            return False, (
                "⚠️ I can't assist with that type of request. "
                "Please ask about legitimate contract terms and clauses."
            )
    return True, ""

def answer_question(
    question: str,
    vectorstore: Chroma,
    chat_history: List = None,
    top_k: int = 5,
) -> dict:
    chat_history = chat_history or []
    is_safe, guardrail_msg = check_guardrails(question)
    if not is_safe:
        return {
            "answer": guardrail_msg,
            "sources": [],
            "blocked": True,
            "chunks_retrieved": 0,
        }
    retrieved = retrieve_relevant_chunks(vectorstore, question, top_k=top_k)
    context, sources = format_context(retrieved)
    lc_history = []
    for msg in chat_history:
        if msg["role"] == "user":
            lc_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lc_history.append(AIMessage(content=msg["content"]))
    llm = get_llm()
    prompt = QA_PROMPT.format_messages(
        context=context,
        chat_history=lc_history,
        question=question,
    )
    response = llm.invoke(prompt)
    answer = response.content
    if "not legal advice" not in answer.lower():
        answer += "\n\n*Disclaimer: This is not legal advice. Consult a qualified attorney for legal decisions.*"
    return {
        "answer": answer,
        "sources": sources,
        "blocked": False,
        "chunks_retrieved": len(retrieved),
    }