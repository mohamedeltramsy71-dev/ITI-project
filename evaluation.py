import os
import json
from typing import List, Dict
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")

def evaluate_answer(
    question: str,
    answer: str,
    context: str,
    ground_truth: str = None,
) -> Dict:
    """
    Evaluate a RAG answer using LLM-as-judge approach.
    Metrics:
    - Faithfulness: Is the answer grounded in context?
    - Relevance: Does the answer address the question?
    - Completeness: Is the answer complete?
    """
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY,
        temperature=0,
    )
    
    eval_prompt = f"""You are an expert evaluator for RAG systems. Evaluate this answer objectively and fairly.

QUESTION: {question}

CONTEXT PROVIDED:
{context[:2000]}

GENERATED ANSWER:
{answer}

{"GROUND TRUTH (Expected Answer): " + ground_truth if ground_truth else ""}

EVALUATION INSTRUCTIONS:

1. RELEVANCE (Does the answer address the question?)
   Score 5: Directly and completely answers the question
   Score 4: Answers the question with minor extra details
   Score 3: Partially answers, some information is off-topic
   Score 2: Tangentially related but doesn't answer directly
   Score 1: Completely misses the question

2. FAITHFULNESS (Is the answer grounded in context?)
   Score 5: Every fact comes directly from the context
   Score 4: Mostly factual with minimal interpretation
   Score 3: Mix of facts and reasonable inference
   Score 2: Contains unsupported claims
   Score 1: Contradicts or ignores the context

3. COMPLETENESS (Is the answer thorough?)
   Score 5: Comprehensive, covers all aspects
   Score 4: Good coverage, minor details could be added
   Score 3: Adequate but missing some key points
   Score 2: Lacks important information
   Score 1: Minimal or vague response

SCORING GUIDELINES:
- Be FAIR and BALANCED in your evaluation
- A concise but accurate answer deserves high scores
- Focus on QUALITY over LENGTH
- Consider if the answer would actually help the user
- Compare to ground truth if provided

Respond with ONLY valid JSON (no markdown, no explanation):
{{
  "relevance": <integer 1-5>,
  "faithfulness": <integer 1-5>,
  "completeness": <integer 1-5>,
  "relevance_reason": "<concise explanation in 10-15 words>",
  "faithfulness_reason": "<concise explanation in 10-15 words>",
  "completeness_reason": "<concise explanation in 10-15 words>",
  "overall_score": <float, average of three scores>
}}"""
    
    response = llm.invoke(eval_prompt)
    try:
        content = response.content.strip()
        
        # Clean JSON response
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join([l for l in lines if not l.startswith("```")])
        if content.startswith("json"):
            content = content[4:].strip()
        
        result = json.loads(content)
        
        # Ensure overall_score is calculated
        if "overall_score" not in result or result["overall_score"] == 0:
            result["overall_score"] = round(
                (result["relevance"] + result["faithfulness"] + result["completeness"]) / 3, 
                2
            )
        
        return result
        
    except Exception as e:
        return {
            "faithfulness": 0,
            "relevance": 0,
            "completeness": 0,
            "faithfulness_reason": "Parse error",
            "relevance_reason": "Parse error",
            "completeness_reason": "Parse error",
            "overall_score": 0,
            "error": f"Could not parse evaluation: {str(e)}",
            "raw": response.content if 'response' in locals() else "No response",
        }

def run_evaluation_suite(
    qa_pairs: List[Dict],
    answer_fn,
    vectorstore,
) -> Dict:
    """
    Run evaluation on a list of question-answer pairs.
    qa_pairs: [{"question": "...", "ground_truth": "..."}]
    """
    results = []
    for pair in qa_pairs:
        print(f"🔍 Evaluating: {pair['question'][:50]}...")
        result = answer_fn(pair["question"], vectorstore)
        scores = evaluate_answer(
            question=pair["question"],
            answer=result["answer"],
            context=str(result.get("sources", "")),
            ground_truth=pair.get("ground_truth"),
        )
        results.append({
            "question": pair["question"],
            "answer": result["answer"],
            "scores": scores,
        })
    
    valid = [r for r in results if "error" not in r["scores"]]
    if valid:
        avg_faithfulness = sum(r["scores"]["faithfulness"] for r in valid) / len(valid)
        avg_relevance = sum(r["scores"]["relevance"] for r in valid) / len(valid)
        avg_completeness = sum(r["scores"]["completeness"] for r in valid) / len(valid)
    else:
        avg_faithfulness = avg_relevance = avg_completeness = 0
    
    return {
        "total_questions": len(qa_pairs),
        "evaluated": len(valid),
        "avg_faithfulness": round(avg_faithfulness, 2),
        "avg_relevance": round(avg_relevance, 2),
        "avg_completeness": round(avg_completeness, 2),
        "avg_overall": round((avg_faithfulness + avg_relevance + avg_completeness) / 3, 2),
        "detailed_results": results,
    }