# 📜 Smart Contract Assistant

RAG-based document Q&A system using Google Gemini + LangChain + ChromaDB + Gradio.

## 🚀 Quick Start

### 1. Clone & Setup
\`\`\`bash
git clone <your-repo>
cd smart_contract_assistant
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
\`\`\`

### 2. Configure API Key
\`\`\`bash
cp .env.example .env
# Edit .env and add your Gemini API key:
# GOOGLE_API_KEY=your_key_here
\`\`\`

### 3. Run
\`\`\`bash
python main.py
# API: http://localhost:8000
# UI:  http://localhost:7860
\`\`\`

## 📁 Project Structure
- `src/ingestion.py` - PDF/DOCX parsing + chunking + embedding
- `src/retrieval.py` - Semantic search
- `src/llm_pipeline.py` - Grock + guardrails
- `src/summarizer.py` - Map-reduce summarization
- `src/evaluation.py` - RAG evaluation metrics
- `api/server.py` - FastAPI backend
- `ui/gradio_app.py` - Gradio interface

## 🔑 Get Gemini API Key
1. Go to https://aistudio.google.com/app/apikey
2. Create new API key
3. Add to `.env` file

## 📊 Architecture
User → Gradio UI → FastAPI → LangChain RAG Pipeline → Gemini LLM → Answer + Citations
                                    ↕
                              ChromaDB Vector Store