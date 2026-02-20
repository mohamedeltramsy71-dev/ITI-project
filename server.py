print("API: http://localhost:8000/docs")

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, AsyncIterator
import asyncio
import json
from langserve import add_routes
from langchain_core.runnables import RunnableLambda
from src.ingestion import ingest_document, load_vectorstore, extract_text
from src.llm_pipeline import answer_question
from src.summarizer import summarize_document

app = FastAPI(
    title="Smart Contract Assistant API",
    description="RAG-based contract Q&A system with LangServe Streaming",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

vectorstore_cache: dict = {}
chat_histories: dict = {}
UPLOAD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "uploads"
)
os.makedirs(UPLOAD_DIR, exist_ok=True)

class QuestionRequest(BaseModel):
    question: str
    session_id: str
    collection_name: str = "contracts"
    top_k: int = 5

class QuestionResponse(BaseModel):
    answer: str
    sources: List[dict]
    blocked: bool
    chunks_retrieved: int

class SummarizeRequest(BaseModel):
    collection_name: str = "contracts"
    file_name: str

def _load_vs(collection_name: str):
    """Get vectorstore from cache or load from disk."""
    if collection_name not in vectorstore_cache:
        try:
            vectorstore_cache[collection_name] = load_vectorstore(collection_name)
        except Exception:
            raise ValueError(
                f"No vectorstore found for '{collection_name}'. Please upload a document first."
            )
    return vectorstore_cache[collection_name]

def qa_runnable_fn(inputs: dict) -> dict:
    """
    Input:
        question        (str)
        collection_name (str, default: "contracts")
        chat_history    (list, default: [])
        top_k           (int, default: 5)
    """
    vs = _load_vs(inputs.get("collection_name", "contracts"))
    return answer_question(
        question=inputs["question"],
        vectorstore=vs,
        chat_history=inputs.get("chat_history", []),
        top_k=inputs.get("top_k", 5),
    )

def summarize_runnable_fn(inputs: dict) -> dict:
    """
    Input:
        file_name (str)
    """
    file_path = os.path.join(UPLOAD_DIR, inputs["file_name"])
    if not os.path.exists(file_path):
        raise ValueError(f"File not found: {inputs['file_name']}")
    text = extract_text(file_path)
    return summarize_document(text)

add_routes(
    app,
    RunnableLambda(qa_runnable_fn),
    path="/langserve/ask",
    enabled_endpoints=["invoke", "batch", "stream", "stream_log", "playground"],
)
add_routes(
    app,
    RunnableLambda(summarize_runnable_fn),
    path="/langserve/summarize",
    enabled_endpoints=["invoke", "playground"],
)

async def _stream_answer(req: QuestionRequest) -> AsyncIterator[str]:
    vs = _load_vs(req.collection_name)
    history = chat_histories.get(req.session_id, [])
    loop = asyncio.get_event_loop()
    result: dict = await loop.run_in_executor(
        None,
        lambda: answer_question(
            question=req.question,
            vectorstore=vs,
            chat_history=history,
            top_k=req.top_k,
        ),
    )
    answer: str = result.get("answer", "")
    sources: list = result.get("sources", [])
    blocked: bool = result.get("blocked", False)
    words = answer.split(" ")
    for i, word in enumerate(words):
        chunk = word if i == len(words) - 1 else word + " "
        yield f"data: {json.dumps({'type': 'token', 'data': chunk})}\n\n"
        await asyncio.sleep(0.02) 
    yield f"data: {json.dumps({'type': 'done', 'sources': sources, 'blocked': blocked, 'chunks_retrieved': result.get('chunks_retrieved', 0)})}\n\n"
    history.append({"role": "user", "content": req.question})
    history.append({"role": "assistant", "content": answer})
    chat_histories[req.session_id] = history[-20:]

@app.post("/ask/stream")
async def ask_stream(req: QuestionRequest):
    """
    SSE streaming endpoint. Example JS usage:

        const response = await fetch("/ask/stream", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question: "...", session_id: "abc" }),
        });
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            const lines = decoder.decode(value).split("\\n");
            for (const line of lines) {
                if (line.startsWith("data: ")) {
                    const event = JSON.parse(line.slice(6));
                    if (event.type === "token") process.stdout.write(event.data);
                    if (event.type === "done")  console.log("\\nSources:", event.sources);
                }
            }
        }
    """
    return StreamingResponse(
        _stream_answer(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no", 
        },
    )

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Smart Contract Assistant"}

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    collection_name: str = "contracts",
):
    """Upload and ingest a PDF/DOCX document."""
    if not file.filename.endswith((".pdf", ".docx", ".doc")):
        raise HTTPException(400, "Only PDF and DOCX files are supported.")
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        vs = ingest_document(file_path, collection_name=collection_name)
        vectorstore_cache[collection_name] = vs
        return {
            "status": "success",
            "file": file.filename,
            "collection": collection_name,
            "message": "Document ingested successfully",
        }
    except Exception as e:
        raise HTTPException(500, f"Ingestion failed: {str(e)}")


@app.post("/ask", response_model=QuestionResponse)
async def ask_question(req: QuestionRequest):
    """Non-streaming Q&A endpoint."""
    vs = _load_vs(req.collection_name)
    history = chat_histories.get(req.session_id, [])
    result = answer_question(
        question=req.question,
        vectorstore=vs,
        chat_history=history,
        top_k=req.top_k,
    )
    history.append({"role": "user", "content": req.question})
    history.append({"role": "assistant", "content": result["answer"]})
    chat_histories[req.session_id] = history[-20:]

    return QuestionResponse(**result)

@app.post("/summarize")
async def summarize(req: SummarizeRequest):
    """Summarize an uploaded document."""
    file_path = os.path.join(UPLOAD_DIR, req.file_name)
    if not os.path.exists(file_path):
        raise HTTPException(404, f"File not found: {req.file_name}")
    text = extract_text(file_path)
    return summarize_document(text)

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear chat history for a session."""
    chat_histories.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)