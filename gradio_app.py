print("UI:  http://localhost:7860")

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import uuid
import json
import gradio as gr
import requests
from dotenv import load_dotenv
from src.evaluation import evaluate_answer

load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8000")
session_id = str(uuid.uuid4())

def upload_file(file):
    if file is None:
        return "Please select a file.", gr.update(visible=False)
    with open(file, "rb") as f:
        response = requests.post(
            f"{API_URL}/upload",
            files={"file": (os.path.basename(file), f)},
        )
    if response.status_code == 200:
        data = response.json()
        return (
            f"**{data['file']}** ingested successfully!\n"
            f"Collection: `{data['collection']}`",
            gr.update(visible=True),
        )
    else:
        return f" Upload failed: {response.text}", gr.update(visible=False)

def summarize_doc(file):
    if file is None:
        return "Please upload a file first."
    response = requests.post(
        f"{API_URL}/summarize",
        json={"file_name": os.path.basename(file)},
    )
    if response.status_code == 200:
        data = response.json()
        strategy = data.get("strategy", "unknown")
        return f"**Document Summary** (Strategy: {strategy})\n\n{data['summary']}"
    else:
        return f"Summarization failed: {response.text}"

def chat(message, history):
    if not message.strip():
        return history, ""
    response = requests.post(
        f"{API_URL}/ask",
        json={
            "question": message,
            "session_id": session_id,
            "top_k": 5,
        },
    )
    
    if response.status_code == 200:
        data = response.json()
        answer = data["answer"]
        if data["sources"] and not data["blocked"]:
            sources_text = "\n\n**Sources:**\n"
            for src in data["sources"]:
                sources_text += (
                    f"-`{src['file']}` | Chunk {src['chunk']} "
                    f"| Relevance: {src['relevance_score']:.2f}\n"
                    f"  *\"{src['excerpt']}\"*\n"
                )
            answer += sources_text
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": answer})
    else:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": f" Error: {response.text}"})
    return history, ""

def clear_chat():
    requests.delete(f"{API_URL}/session/{session_id}")
    return [], ""
def run_evaluation(questions_text, progress=gr.Progress()):
    """Run evaluation on provided questions."""
    if not questions_text.strip():
        return "Please enter questions to evaluate.", "", ""
    
    try:
        try:
            qa_pairs = json.loads(questions_text)
            if not isinstance(qa_pairs, list):
                qa_pairs = [qa_pairs]
        except:
            lines = [l.strip() for l in questions_text.split('\n') if l.strip()]
            qa_pairs = [{"question": q} for q in lines]
        
        if not qa_pairs:
            return " No valid questions found.", "", ""
        progress(0, desc="Starting evaluation...")
        results = []
        for i, pair in enumerate(qa_pairs):
            progress((i + 1) / len(qa_pairs), desc=f"Evaluating {i+1}/{len(qa_pairs)}...")
            
            question = pair["question"]

            response = requests.post(
                f"{API_URL}/ask",
                json={
                    "question": question,
                    "session_id": f"eval-{uuid.uuid4()}",
                    "top_k": 5,
                },
            )
            
            if response.status_code == 200:
                result = response.json()
            else:
                result = {
                    "answer": "Error",
                    "sources": [],
                    "blocked": True,
                }
            scores = evaluate_answer(
                question=question,
                answer=result["answer"],
                context=str(result.get("sources", "")),
                ground_truth=pair.get("ground_truth"),
            )
            
            results.append({
                "question": question,
                "answer": result["answer"],
                "scores": scores,
            })
        
        valid = [r for r in results if "error" not in r["scores"]]
        if valid:
            avg_faithfulness = sum(r["scores"]["faithfulness"] for r in valid) / len(valid)
            avg_relevance = sum(r["scores"]["relevance"] for r in valid) / len(valid)
            avg_completeness = sum(r["scores"]["completeness"] for r in valid) / len(valid)
            avg_overall = (avg_faithfulness + avg_relevance + avg_completeness) / 3
        else:
            avg_faithfulness = avg_relevance = avg_completeness = avg_overall = 0
        
        # Format summary
        summary = f"""## Evaluation Results

**Total Questions:** {len(qa_pairs)}  
**Successfully Evaluated:** {len(valid)}  
**Failed:** {len(qa_pairs) - len(valid)}

### Average Scores (1-5 scale)
- **Faithfulness:** {avg_faithfulness:.2f} / 5.0
- **Relevance:** {avg_relevance:.2f} / 5.0
- **Completeness:** {avg_completeness:.2f} / 5.0
- **Overall:** {avg_overall:.2f} / 5.0
"""
        detailed = "##Detailed Results\n\n"
        for i, r in enumerate(results):
            scores = r["scores"]
            if "error" in scores:
                detailed += f"### Question {i+1}: {r['question']}\n"
                detailed += f"**Error:** {scores['error']}\n\n---\n\n"
            else:
                detailed += f"### Question {i+1}: {r['question']}\n\n"
                detailed += f"**Scores:**\n"
                detailed += f"- Faithfulness: {scores['faithfulness']}/5 - {scores.get('faithfulness_reason', 'N/A')}\n"
                detailed += f"- Relevance: {scores['relevance']}/5 - {scores.get('relevance_reason', 'N/A')}\n"
                detailed += f"- Completeness: {scores['completeness']}/5 - {scores.get('completeness_reason', 'N/A')}\n"
                detailed += f"- **Overall: {scores.get('overall_score', 0):.2f}/5**\n\n"
                detailed += f"**Answer:** {r['answer'][:300]}...\n\n"
                detailed += "---\n\n"

        json_output = json.dumps({
            "summary": {
                "total_questions": len(qa_pairs),
                "evaluated": len(valid),
                "avg_faithfulness": round(avg_faithfulness, 2),
                "avg_relevance": round(avg_relevance, 2),
                "avg_completeness": round(avg_completeness, 2),
                "avg_overall": round(avg_overall, 2),
            },
            "detailed_results": results,
        }, indent=2)
        
        progress(1.0, desc="Evaluation complete!")
        return summary, detailed, json_output
        
    except Exception as e:
        return f"Error during evaluation: {str(e)}", "", ""
with gr.Blocks(title="Smart Contract Assistant") as demo:
    gr.Markdown("""
    Smart Contract & Document Assistant
    **Powered by LangChain + HuggingFace + ChromaDB**
    Upload a PDF or DOCX document, then ask questions about it.
    """)
    with gr.Tabs():
        with gr.Tab("Upload Document"):
            with gr.Row():
                file_input = gr.File(
                    label="Upload PDF or DOCX",
                    file_types=[".pdf", ".docx", ".doc"],
                )
            upload_btn = gr.Button("Process Document", variant="primary")
            upload_status = gr.Markdown()
            summarize_btn = gr.Button("Summarize Document", visible=False)
            summary_output = gr.Markdown()
            upload_btn.click(
                upload_file,
                inputs=[file_input],
                outputs=[upload_status, summarize_btn],
            )
            summarize_btn.click(
                summarize_doc,
                inputs=[file_input],
                outputs=[summary_output],
            )

        with gr.Tab("Ask Questions"):
            chatbot = gr.Chatbot(
                label="Contract Q&A",
                height=500,
            )
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Ask anything about your document...",
                    label="Your Question",
                    scale=4,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
            clear_btn = gr.Button("Clear Chat", variant="secondary")
            gr.Examples(
                examples=[
                    "What are the main obligations of each party?",
                    "What are the payment terms?",
                    "What are the termination clauses?",
                    "Are there any penalty clauses?",
                    "What is the contract duration?",
                    "Summarize the key risks in this contract.",
                ],
                inputs=msg_input,
            )
            send_btn.click(chat, [msg_input, chatbot], [chatbot, msg_input])
            msg_input.submit(chat, [msg_input, chatbot], [chatbot, msg_input])
            clear_btn.click(clear_chat, outputs=[chatbot, msg_input])

        with gr.Tab("Evaluation"):
            with gr.Row():
                with gr.Column(scale=1):
                    questions_input = gr.Textbox(
                        label="Questions",
                        placeholder="Enter questions (one per line)...",
                        lines=12,
                    )
                    eval_btn = gr.Button("Run Evaluation", variant="primary", size="lg")
                
                with gr.Column(scale=2):
                    summary_output_eval = gr.Markdown(label="Summary")
                    
                    with gr.Accordion("Detailed Results", open=False):
                        detailed_output = gr.Markdown()
                    
                    with gr.Accordion("Export JSON", open=False):
                        json_output = gr.Textbox(label="JSON", lines=10)
            eval_btn.click(
                run_evaluation,
                inputs=[questions_input],
                outputs=[summary_output_eval, detailed_output, json_output],
            )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(primary_hue="blue"),
    )