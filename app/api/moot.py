from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from bson import ObjectId
import random
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from app.routes.auth import get_current_user
from app.database.mongodb import live_sessions_collection, judge_questions_collection
from eval_rag.evaluator.rubric import RUBRIC_TEXT
from rag.moot_rag.run_rag import run_opponent_rag
import shutil
from fastapi import UploadFile, File
from rag.moot_rag.audio.stt import speech_to_text
from rag.moot_rag.audio.tts import text_to_speech
from eval_rag.api.main import call_llm
from eval_rag.evaluator.prompt_builder import build_prompt
from eval_rag.retrieval.retriever import retrieve_context

router = APIRouter(prefix="/moot", tags=["Moot"])

# ================= MODELS =================
class SessionRequest(BaseModel):
    case_id: str
    case_type: str

class ArgumentRequest(BaseModel):
    text: Optional[str] = None

# ================= HELPERS =================
async def get_session_by_id(session_id: str, user_id):
    session = await live_sessions_collection.find_one({
        "_id": ObjectId(session_id),
        "user_id": user_id  # already ObjectId
    })
    if not session:
        raise HTTPException(404, "Session not found")
    return session

async def push_history(session_id: str, user_id, role: str, text: str, type: Optional[str] = None):
    entry = {"role": role, "text": text, "timestamp": datetime.utcnow()}
    if type:
        entry["type"] = type
    result = await live_sessions_collection.update_one(
        {"_id": ObjectId(session_id), "user_id": user_id},
        {"$push": {"history": entry}}
    )
    if result.modified_count == 0:
        raise HTTPException(500, "Failed to push history")

async def set_turn(session_id: str, turn: str, party: Optional[str]):
    await live_sessions_collection.update_one(
        {"_id": ObjectId(session_id)},
        {"$set": {
            "next_turn": turn,
            "current_party": party,
            "updated_at": datetime.utcnow()
        }}
    )

async def get_judge_question(case_type: str):
    q = await judge_questions_collection.find_one({"case_type": case_type.lower()})
    if not q or not q.get("questions"):
        return None
    return random.choice(q["questions"])

# ================= SESSION =================
@router.post("/initiate")
async def initiate(req: SessionRequest, current_user=Depends(get_current_user)):
    session = {
        "user_id": current_user["_id"],  # store ObjectId directly
        "case_id": req.case_id,
        "case_type": req.case_type,
        "history": [],
        "evaluation_history": [],
        "original_petitioner_argument": None,
        "next_turn": "PETITIONER_ARGUMENT",
        "current_party": "PETITIONER",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    res = await live_sessions_collection.insert_one(session)
    return {"session_id": str(res.inserted_id), "next_turn": "PETITIONER_ARGUMENT"}

@router.get("/transcript")
async def transcript(session_id: str, current_user=Depends(get_current_user)):
    session = await get_session_by_id(session_id, current_user["_id"])
    return session.get("history", [])

# ================= PETITIONER =================
@router.post("/petitioner/argument")
async def petitioner_argument(req: ArgumentRequest, session_id: str, current_user=Depends(get_current_user)):
    session = await get_session_by_id(session_id, current_user["_id"])
    if session["current_party"] != "PETITIONER":
        raise HTTPException(400, "Not petitioner's turn")

    # Save original argument
    await live_sessions_collection.update_one(
        {"_id": session["_id"]},
        {"$set": {"original_petitioner_argument": req.text or ""}}
    )
    await push_history(session_id, current_user["_id"], "petitioner", req.text or "", type="argument")

    # Judge question
    judge_q = await get_judge_question(session["case_type"])
    if judge_q:
        await push_history(session_id, current_user["_id"], "judge", judge_q)

    await set_turn(session_id, "PETITIONER_REPLY_TO_JUDGE", "PETITIONER")
    return {"judge_question": judge_q, "next_turn": "PETITIONER_REPLY_TO_JUDGE"}

@router.post("/petitioner/reply")
async def petitioner_reply(req: ArgumentRequest, session_id: str, current_user=Depends(get_current_user)):
    session = await get_session_by_id(session_id, current_user["_id"])
    if session["current_party"] != "PETITIONER":
        raise HTTPException(400, "Not petitioner's turn")

    await push_history(session_id, current_user["_id"], "petitioner", req.text or "", type="reply_to_judge")
    await set_turn(session_id, "RESPONDENT_RAG", "RESPONDENT")
    return {"next_turn": "RESPONDENT_RAG"}

# ================= RESPONDENT RAG =================
@router.post("/respondent/rag")
async def respondent_rag(session_id: str, current_user=Depends(get_current_user)):
    session = await get_session_by_id(session_id, current_user["_id"])
    original_arg = session.get("original_petitioner_argument", "")
    history = session.get("history", [])

    # Generate AI argument
    response = await asyncio.to_thread(run_opponent_rag, case_key=session["case_id"], argument=original_arg, history=history)
    respondent_argument = response.get("response") if isinstance(response, dict) else str(response)
    await push_history(session_id, current_user["_id"], "respondent", respondent_argument)

    # Judge question & AI reply
    judge_q = await get_judge_question(session["case_type"])
    if judge_q:
        await push_history(session_id, current_user["_id"], "judge", judge_q)
        updated_session = await get_session_by_id(session_id, current_user["_id"])
        reply_response = await asyncio.to_thread(run_opponent_rag, case_key=session["case_id"], argument=judge_q, history=updated_session["history"])
        respondent_reply = reply_response.get("response") if isinstance(reply_response, dict) else str(reply_response)
        await push_history(session_id, current_user["_id"], "respondent", respondent_reply)
    else:
        respondent_reply = None

    await set_turn(session_id, "PETITIONER_REBUTTAL", "PETITIONER")
    return {
        "respondent_argument": respondent_argument,
        "judge_question": judge_q,
        "respondent_reply": respondent_reply,
        "next_turn": "PETITIONER_REBUTTAL"
    }

# ================= PETITIONER REBUTTAL =================
@router.post("/petitioner/rebut")
async def petitioner_rebut(req: ArgumentRequest, session_id: str, current_user=Depends(get_current_user)):
    session = await get_session_by_id(session_id, current_user["_id"])
    if session["current_party"] != "PETITIONER":
        raise HTTPException(400, "Not petitioner's turn")

    await push_history(session_id, current_user["_id"], "petitioner", req.text or "", type="rebuttal")
    await set_turn(session_id, "SESSION_END", None)
    return {"next_turn": "SESSION_END"}

# ================= EVALUATION =================
@router.post("/evaluate")
async def evaluate_user_only(session_id: str, current_user=Depends(get_current_user)):
    logger.info("🔍 Starting evaluation...")
    logger.info(f"Session ID: {session_id}")
    logger.info(f"User ID: {current_user['_id']}")

    session = await get_session_by_id(session_id, current_user["_id"])

    if session.get("next_turn") != "SESSION_END":
        logger.warning("❌ Session not ended yet")
        raise HTTPException(400, "Session not ended yet")

    history = session.get("history", [])
    logger.info(f"History length: {len(history)}")

    main_argument, judge_responses, rebuttals = [], [], []

    for idx, h in enumerate(history):
        logger.info(f"Processing history item {idx}: {h}")

        if h.get("role") != "petitioner":
            continue

        t = h.get("type") or (
            "argument" if idx == 0 else
            "rebuttal" if idx == len(history)-1 else
            "reply_to_judge"
        )

        if t == "argument":
            main_argument.append(h.get("text", ""))
        elif t == "reply_to_judge":
            judge_responses.append(h.get("text", ""))
        elif t == "rebuttal":
            rebuttals.append(h.get("text", ""))

    main_argument_text = "\n\n".join(main_argument)
    judge_response_text = "\n\n".join(judge_responses)
    rebuttal_text = "\n\n".join(rebuttals)

    logger.info("Main Argument Extracted:")
    logger.info(main_argument_text)

    logger.info("Judge Responses Extracted:")
    logger.info(judge_response_text)

    logger.info("Rebuttal Extracted:")
    logger.info(rebuttal_text)

    last_judge_q = next(
        (h["text"] for h in reversed(history) if h.get("role") == "judge"),
        ""
    )

    last_respondent_arg = next(
        (h["text"] for h in reversed(history) if h.get("role") == "respondent"),
        ""
    )

    logger.info(f"Last Judge Question: {last_judge_q}")
    logger.info(f"Last Respondent Argument: {last_respondent_arg}")

    retrieved_context_text = retrieve_context(
        main_argument=main_argument_text,
        judge_question=last_judge_q,
        case_type=session.get("case_type", "default")
    )

    logger.info("Retrieved Context:")
    logger.info(retrieved_context_text)

    prompt = build_prompt(
        main_argument=main_argument_text,
        judge_question=last_judge_q,
        user_judge_answer=judge_response_text,
        opponent_argument=last_respondent_arg,
        user_rebuttal=rebuttal_text,
        rubric_text=RUBRIC_TEXT,
        retrieved_context=retrieved_context_text
    )

    logger.info("Final Prompt Sent To LLM:")
    logger.info(prompt)

    raw_eval_result = await asyncio.to_thread(call_llm, prompt)

    logger.info("Raw LLM Response:")
    logger.info(raw_eval_result)

    # Parse evaluator output
    import re, json
    scores = {}
    overall_feedback = "Evaluation completed."

    if raw_eval_result:
        if isinstance(raw_eval_result, dict):
            eval_obj = raw_eval_result
        else:
            match = re.search(r"\{.*\}", str(raw_eval_result), re.DOTALL)
            eval_obj = json.loads(match.group()) if match else None

        logger.info(f"Parsed Evaluation Object: {eval_obj}")

        if eval_obj:
            scores = eval_obj.get("scores", {})
            overall_feedback = eval_obj.get("overall_feedback", overall_feedback)

    logger.info(f"Final Scores: {scores}")
    logger.info(f"Overall Feedback: {overall_feedback}")

    await live_sessions_collection.update_one(
        {"_id": session["_id"]},
        {"$push": {
            "evaluation_history": {
                "party": "petitioner",
                "timestamp": datetime.utcnow(),
                "evaluation": {
                    "scores": scores,
                    "overall_feedback": overall_feedback
                }
            }
        }}
    )

    logger.info("✅ Evaluation saved successfully")

    return {
        "session_id": str(session["_id"]),
        "evaluation": {
            "scores": scores,
            "overall_feedback": overall_feedback
        }
    }




@router.post("/petitioner/argument/audio")
async def petitioner_argument_audio(
    session_id: str,
    file: UploadFile = File(...),
    current_user=Depends(get_current_user)
):
    session = await get_session_by_id(session_id, current_user["_id"])
    if session["current_party"] != "PETITIONER":
        raise HTTPException(400, "Not petitioner's turn")

    temp_path = f"temp_{session_id}.wav"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    text = speech_to_text(temp_path)

    await live_sessions_collection.update_one(
        {"_id": session["_id"]},
        {"$set": {"original_petitioner_argument": text}}
    )

    await push_history(session_id, current_user["_id"], "petitioner", text, type="argument")

    judge_q = await get_judge_question(session["case_type"])
    if judge_q:
        await push_history(session_id, current_user["_id"], "judge", judge_q)

    await set_turn(session_id, "PETITIONER_REPLY_TO_JUDGE", "PETITIONER")

    return {
        "transcribed_text": text,
        "judge_question": judge_q,
        "next_turn": "PETITIONER_REPLY_TO_JUDGE"
    }