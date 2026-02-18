from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import os
import json

from groq import Groq


# ======================
# APP + LLM CLIENT
# ======================

app = FastAPI()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ======================
# SCHEMA
# ======================

class EvalRequest(BaseModel):
    case_type: str

    # USER (Petitioner)
    main_argument: str
    user_reply_to_judge: Optional[str] = ""
    user_rebuttal: Optional[str] = ""

    # CONTEXT
    judge_question: Optional[str] = ""
    respondent_argument: Optional[str] = ""

# ======================
# LLM CALL
# ======================

def call_llm(prompt: str):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You are a strict moot court evaluator. Evaluate ONLY the petitioner."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    message = response.choices[0].message.content

    try:
        return json.loads(message)
    except json.JSONDecodeError:
        return {
            "error": "Evaluator did not return valid JSON",
            "raw_output": message
        }

# # ======================
# # API ENDPOINT
# # ======================

# @app.post("/evaluate")
# def evaluate_petitioner(eval_request: EvalRequest):
#     # 1️⃣ Retrieve context from vector DB
#     retrieved_docs = retrieve_context(
#         case_type=eval_request.case_type,
#         query=eval_request.main_argument
#     )

#     # Log retrieved docs for debugging
#     print("Retrieved documents:")
#     for i, doc in enumerate(retrieved_docs):
#         print(f"Doc {i+1}: {doc[:500]}")  # first 500 chars

#     # 2️⃣ Build prompt including retrieved context
#     prompt = build_evaluation_prompt(
#         main_argument=eval_request.main_argument,
#         judge_question=eval_request.judge_question,
#         user_judge_answer=eval_request.user_reply_to_judge,
#         opponent_argument=eval_request.respondent_argument,
#         user_rebuttal=eval_request.user_rebuttal,
#         rubric_text=RUBRIC_TEXT,
#         retrieved_context="\n".join(retrieved_docs)
#     )

#     # 3️⃣ Call LLM
#     result = call_llm(prompt)
#     return result
