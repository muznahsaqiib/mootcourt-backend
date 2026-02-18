# # backend/app/routes/session.py

# from fastapi import APIRouter, HTTPException, Depends
# from pydantic import BaseModel
# from typing import Optional
# from datetime import datetime
# from bson import ObjectId
# import random
# import asyncio

# from app.database.mongodb import sessions_collection, judge_questions_collection
# from app.routes.auth import get_current_user
# from rag.moot_rag.run_rag import run_opponent_rag
# from eval_rag.evaluator.rubric import RUBRIC_TEXT
# from eval_rag.retrieval.retriever import retrieve_context
# from eval_rag.evaluator.prompt_builder import build_prompt
# from eval_rag.api.main import call_llm

# router = APIRouter(prefix="/session", tags=["Session"])

# # ------------------ MODELS ------------------

# class StartSessionRequest(BaseModel):
#     case_id: str
#     mode: str = "argument"

# class ArgumentRequest(BaseModel):
#     text: Optional[str] = None

# class EvaluationInput(BaseModel):
#     party: str  # petitioner / respondent
#     scores: dict
#     overall_feedback: str

# # ------------------ HELPERS ------------------

# async def get_session(session_id: str):
#     session = await sessions_collection.find_one({"_id": ObjectId(session_id)})
#     if not session:
#         raise HTTPException(404, "Session not found")
#     return session

# async def push_history(session_id: str, role: str, text: str, type_: Optional[str] = None):
#     entry = {"role": role, "text": text, "timestamp": datetime.utcnow()}
#     if type_:
#         entry["type"] = type_
#     await sessions_collection.update_one(
#         {"_id": ObjectId(session_id)},
#         {"$push": {"history": entry}, "$set": {"updated_at": datetime.utcnow()}}
#     )

# async def set_turn(session_id: str, next_turn: Optional[str], current_party: Optional[str]):
#     await sessions_collection.update_one(
#         {"_id": ObjectId(session_id)},
#         {"$set": {"turn": next_turn, "current_party": current_party, "updated_at": datetime.utcnow()}}
#     )

# async def get_judge_question(case_type: str):
#     q = await judge_questions_collection.find_one({"case_type": case_type.lower()})
#     if not q or not q.get("questions"):
#         return None
#     return random.choice(q["questions"])

# # ------------------ SESSION ------------------

# @router.post("/start")
# async def start_session(req: StartSessionRequest, current_user=Depends(get_current_user)):
#     session_doc = {
#         "user_id": str(current_user["_id"]),
#         "username": current_user["username"],
#         "case_id": req.case_id,
#         "mode": req.mode,
#         "history": [],
#         "evaluation_history": [],
#         "turn": "PETITIONER_ARGUMENT",
#         "current_party": "PETITIONER",
#         "original_petitioner_argument": None,
#         "status": "active",
#         "created_at": datetime.utcnow(),
#         "updated_at": datetime.utcnow()
#     }
#     res = await sessions_collection.insert_one(session_doc)
#     return {"session_id": str(res.inserted_id), "next_turn": "PETITIONER_ARGUMENT"}

# @router.get("/{session_id}")
# async def fetch_session(session_id: str, current_user=Depends(get_current_user)):
#     session = await get_session(session_id)
#     if session["user_id"] != str(current_user["_id"]):
#         raise HTTPException(403, "Not allowed")
#     return session

# # ------------------ PETITIONER ------------------

# @router.post("/{session_id}/petitioner/argument")
# async def petitioner_argument(session_id: str, req: ArgumentRequest, current_user=Depends(get_current_user)):
#     session = await get_session(session_id)
#     if session["current_party"] != "PETITIONER":
#         raise HTTPException(400, "Not petitioner's turn")

#     await sessions_collection.update_one(
#         {"_id": ObjectId(session_id)},
#         {"$set": {"original_petitioner_argument": req.text or ""}}
#     )

#     await push_history(session_id, "petitioner", req.text or "", type_="argument")
#     judge_q = await get_judge_question(session.get("case_type", "default"))
#     if judge_q:
#         await push_history(session_id, "judge", judge_q)

#     await set_turn(session_id, "PETITIONER_REPLY_TO_JUDGE", "PETITIONER")
#     return {"judge_question": judge_q, "next_turn": "PETITIONER_REPLY_TO_JUDGE"}

# @router.post("/{session_id}/petitioner/reply")
# async def petitioner_reply(session_id: str, req: ArgumentRequest, current_user=Depends(get_current_user)):
#     session = await get_session(session_id)
#     if session["current_party"] != "PETITIONER":
#         raise HTTPException(400, "Not petitioner's turn")

#     await push_history(session_id, "petitioner", req.text or "", type_="reply_to_judge")
#     await set_turn(session_id, "RESPONDENT_RAG", "RESPONDENT")
#     return {"next_turn": "RESPONDENT_RAG"}

# @router.post("/{session_id}/petitioner/rebuttal")
# async def petitioner_rebuttal(session_id: str, req: ArgumentRequest, current_user=Depends(get_current_user)):
#     session = await get_session(session_id)
#     if session["current_party"] != "PETITIONER":
#         raise HTTPException(400, "Not petitioner's turn")

#     await push_history(session_id, "petitioner", req.text or "", type_="rebuttal")
#     await set_turn(session_id, None, None)
#     await sessions_collection.update_one(
#         {"_id": ObjectId(session_id)},
#         {"$set": {"status": "completed", "ended_at": datetime.utcnow()}}
#     )
#     return {"next_turn": "SESSION_END"}

# # ------------------ RESPONDENT RAG ------------------

# @router.post("/{session_id}/respondent/rag")
# async def respondent_rag(session_id: str, current_user=Depends(get_current_user)):
#     session = await get_session(session_id)
#     original_arg = session.get("original_petitioner_argument", "")
#     history = session.get("history", [])

#     response = await asyncio.to_thread(run_opponent_rag,
#                                        case_key=session["case_id"],
#                                        argument=original_arg,
#                                        history=history)
#     respondent_arg = response.get("response") if isinstance(response, dict) else str(response)
#     await push_history(session_id, "respondent", respondent_arg)

#     judge_q = await get_judge_question(session.get("case_type", "default"))
#     if judge_q:
#         await push_history(session_id, "judge", judge_q)
#         reply_response = await asyncio.to_thread(run_opponent_rag,
#                                                  case_key=session["case_id"],
#                                                  argument=judge_q,
#                                                  history=session.get("history", []))
#         respondent_reply = reply_response.get("response") if isinstance(reply_response, dict) else str(reply_response)
#         await push_history(session_id, "respondent", respondent_reply)
#     else:
#         respondent_reply = None

#     await set_turn(session_id, "PETITIONER_REBUTTAL", "PETITIONER")
#     return {
#         "respondent_argument": respondent_arg,
#         "judge_question": judge_q,
#         "respondent_reply": respondent_reply,
#         "next_turn": "PETITIONER_REBUTTAL"
#     }

# # ------------------ EVALUATION ------------------

# @router.post("/{session_id}/evaluate")
# async def evaluate_session(session_id: str, current_user=Depends(get_current_user)):
#     session = await get_session(session_id)
#     history = session.get("history", [])

#     if session.get("status") != "completed":
#         raise HTTPException(400, "Session not ended yet")

#     main_arg = "\n\n".join([h["text"] for h in history if h["role"]=="petitioner" and h.get("type")=="argument"])
#     judge_responses = "\n\n".join([h["text"] for h in history if h["role"]=="petitioner" and h.get("type")=="reply_to_judge"])
#     rebuttals = "\n\n".join([h["text"] for h in history if h["role"]=="petitioner" and h.get("type")=="rebuttal"])
#     last_judge = next((h["text"] for h in reversed(history) if h["role"]=="judge"), "")
#     last_respondent = next((h["text"] for h in reversed(history) if h["role"]=="respondent"), "")

#     retrieved_context = retrieve_context(main_argument=main_arg,
#                                          judge_question=last_judge,
#                                          case_type=session.get("case_type", "default"))

#     prompt = build_prompt(main_argument=main_arg,
#                           judge_question=last_judge,
#                           user_judge_answer=judge_responses,
#                           opponent_argument=last_respondent,
#                           user_rebuttal=rebuttals,
#                           rubric_text=RUBRIC_TEXT,
#                           retrieved_context=retrieved_context)

#     raw_eval = await asyncio.to_thread(call_llm, prompt)

#     import re, json
#     scores = {}
#     overall_feedback = "Evaluation completed."

#     if raw_eval:
#         if isinstance(raw_eval, dict):
#             eval_obj = raw_eval
#         else:
#             match = re.search(r"\{.*\}", str(raw_eval), re.DOTALL)
#             if match:
#                 try:
#                     eval_obj = json.loads(match.group())
#                 except:
#                     eval_obj = None
#             else:
#                 eval_obj = None
#         if eval_obj:
#             scores = eval_obj.get("scores", {})
#             overall_feedback = eval_obj.get("overall_feedback", overall_feedback)

#     # push to session
#     await sessions_collection.update_one(
#         {"_id": ObjectId(session_id)},
#         {"$push": {"evaluation_history": {"party":"petitioner", "scores": scores,
#                                           "overall_feedback": overall_feedback,
#                                           "timestamp": datetime.utcnow()}}}
#     )

#     return {"session_id": session_id, "evaluation": {"scores": scores, "overall_feedback": overall_feedback}}
