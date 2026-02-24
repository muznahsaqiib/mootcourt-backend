import io
import subprocess
import logging
import random
import asyncio
import os
from datetime import datetime
from typing import Optional
from pydantic import BaseModel
from bson import ObjectId
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form
import base64
from app.routes.auth import get_current_user
from app.database.mongodb import live_sessions_collection, judge_questions_collection
from rag.moot_rag.audio.stt import speech_to_text
from rag.moot_rag.audio.tts import text_to_speech
from rag.moot_rag.run_rag import run_opponent_rag
from eval_rag.evaluator.rubric import RUBRIC_TEXT
from eval_rag.api.main import call_llm
from eval_rag.evaluator.prompt_builder import build_prompt
from eval_rag.retrieval.retriever import retrieve_context

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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
        "user_id": user_id
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
        "user_id": current_user["_id"],
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

# ================= AUDIO UTILITY =================
async def process_audio(file: UploadFile) -> str:
    """Convert uploaded audio to WAV and transcribe"""
    webm_bytes = await file.read()

    if not webm_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-i", "pipe:0",
        "-ac", "1",
        "-ar", "16000",
        "-f", "wav",
        "pipe:1"
    ]

    try:
        process = subprocess.run(
            ffmpeg_cmd,
            input=webm_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )

        wav_bytes = process.stdout
        wav_stream = io.BytesIO(wav_bytes)
        wav_stream.seek(0)

        # ✅ run STT in thread since it's blocking
        text = await asyncio.to_thread(speech_to_text, wav_stream)

        if not text:
            raise HTTPException(status_code=500, detail="STT transcription failed")

        return text

    except subprocess.CalledProcessError as e:
        err_msg = e.stderr.decode() if e.stderr else str(e)
        raise HTTPException(status_code=500, detail=f"ffmpeg failed: {err_msg}")

def _tts_output_to_bytes(tts_output) -> bytes:
    """
    Normalize TTS output into raw audio bytes.
    text_to_speech may return bytes directly or a file path.
    """
    if tts_output is None:
        return b""

    if isinstance(tts_output, (bytes, bytearray, memoryview)):
        return bytes(tts_output)

    if isinstance(tts_output, str):
        if not os.path.exists(tts_output):
            raise HTTPException(status_code=500, detail=f"TTS file not found: {tts_output}")
        with open(tts_output, "rb") as f:
            return f.read()

    raise HTTPException(
        status_code=500,
        detail=f"Unexpected TTS output type: {type(tts_output).__name__}"
    )
# ================= PETITIONER AUDIO FLOW =================
async def petitioner_audio_flow(file: UploadFile, session_id: str, current_user):
    # 1️⃣ Fetch session
    session = await get_session_by_id(session_id, current_user["_id"])
    if session["current_party"] != "PETITIONER":
        raise HTTPException(status_code=400, detail="Not petitioner's turn")

    # 2️⃣ Convert audio to WAV and transcribe
    text = await process_audio(file)

    # 3️⃣ Push petitioner's argument to history
    await push_history(
        session_id,
        current_user["_id"],
        "petitioner",
        text,
        type="argument"
    )

    # 4️⃣ Get judge question (if any)
    judge_q = await get_judge_question(session["case_type"])
    if judge_q:
        await push_history(session_id, current_user["_id"], "judge", judge_q)

    # 5️⃣ Generate respondent RAG response in a thread
    response = await asyncio.to_thread(
        run_opponent_rag,
        case_key=session["case_id"],
        argument=text,
        history=session.get("history", [])
    )

    respondent_arg = response.get("response") if isinstance(response, dict) else str(response)
    await push_history(session_id, current_user["_id"], "respondent", respondent_arg)

    # 6️⃣ Convert Judge question & Respondent response to audio (TTS)
    # Ensure text_to_speech returns bytes
    judge_tts_output = await asyncio.to_thread(text_to_speech, judge_q) if judge_q else None
    respondent_tts_output = await asyncio.to_thread(text_to_speech, respondent_arg)
    judge_audio_bytes = _tts_output_to_bytes(judge_tts_output) if judge_tts_output else None
    respondent_audio_bytes = _tts_output_to_bytes(respondent_tts_output)

    # 7️⃣ Encode audio as Base64 for frontend
    judge_audio_b64 = base64.b64encode(judge_audio_bytes).decode() if judge_audio_bytes else None
    respondent_audio_b64 = base64.b64encode(respondent_audio_bytes).decode()

    # 8️⃣ Update next turn
    await set_turn(session_id, "PETITIONER_REBUTTAL", "PETITIONER")

    # 9️⃣ Return all results
    return {
        "transcribed_text": text,
        "judge_question": judge_q,
        "respondent_argument": respondent_arg,
        "judge_audio": judge_audio_b64,
        "respondent_audio": respondent_audio_b64,
        "next_turn": "PETITIONER_REBUTTAL"
    }
@router.post("/petitioner/argument/audio")
async def petitioner_argument_audio(
    session_id: str = Form(...),
    file: UploadFile = File(...),
    current_user=Depends(get_current_user)
):
    return await petitioner_audio_flow(file, session_id, current_user)

# ================= PETITIONER TEXT FLOW =================
@router.post("/petitioner/argument")
async def petitioner_argument(req: ArgumentRequest, session_id: str, current_user=Depends(get_current_user)):
    session = await get_session_by_id(session_id, current_user["_id"])
    if session["current_party"] != "PETITIONER":
        raise HTTPException(400, "Not petitioner's turn")

    await push_history(session_id, current_user["_id"], "petitioner", req.text or "", type="argument")
    judge_q = await get_judge_question(session["case_type"])
    if judge_q:
        await push_history(session_id, current_user["_id"], "judge", judge_q)

    await set_turn(session_id, "PETITIONER_REPLY_TO_JUDGE", "PETITIONER")
    return {"judge_question": judge_q, "next_turn": "PETITIONER_REPLY_TO_JUDGE"}

# ================= PETITIONER REPLY =================
@router.post("/petitioner/reply")
async def petitioner_reply(req: ArgumentRequest, session_id: str, current_user=Depends(get_current_user)):
    session = await get_session_by_id(session_id, current_user["_id"])
    if session["current_party"] != "PETITIONER":
        raise HTTPException(400, "Not petitioner's turn")
    await push_history(session_id, current_user["_id"], "petitioner", req.text or "", type="reply_to_judge")
    await set_turn(session_id, "RESPONDENT_RAG", "RESPONDENT")
    return {"next_turn": "RESPONDENT_RAG"}

@router.post("/petitioner/reply/audio")
async def petitioner_reply_audio(
    session_id: str = Form(...),
    file: UploadFile = File(...),
    current_user=Depends(get_current_user)
):
    session = await get_session_by_id(session_id, current_user["_id"])

    if session["current_party"] != "PETITIONER":
        raise HTTPException(status_code=400, detail="Not petitioner's turn")

    # -------- STT --------
    text = await process_audio(file)

    await push_history(
        session_id,
        current_user["_id"],
        "petitioner",
        text,
        type="reply_to_judge"
    )

    # -------- Respondent RAG --------
    response = await asyncio.to_thread(
        run_opponent_rag,
        case_key=session["case_id"],
        argument=text,
        history=session.get("history", [])
    )

    respondent_reply = (
        response.get("response")
        if isinstance(response, dict)
        else str(response)
    )

    await push_history(
        session_id,
        current_user["_id"],
        "respondent",
        respondent_reply
    )

    # -------- TTS --------
    respondent_tts_output = await asyncio.to_thread(text_to_speech, respondent_reply)
    respondent_audio = _tts_output_to_bytes(respondent_tts_output)
    respondent_audio_b64 = base64.b64encode(respondent_audio).decode()

    await set_turn(session_id, "PETITIONER_REBUTTAL", "PETITIONER")

    return {
        "transcribed_text": text,
        "respondent_reply": respondent_reply,
        "respondent_audio": respondent_audio_b64,
        "next_turn": "PETITIONER_REBUTTAL"
    }

    # # Respondent RAG response
    # response = await asyncio.to_thread(run_opponent_rag, case_key=session["case_id"], argument=text, history=session.get("history", []))
    # respondent_reply = response.get("response") if isinstance(response, dict) else str(response)
    # await push_history(session_id, current_user["_id"], "respondent", respondent_reply)

    # # TTS
    # respondent_audio = text_to_speech(respondent_reply)
    # respondent_audio_b64 = respondent_audio.decode("latin1")

    # await set_turn(session_id, "PETITIONER_REBUTTAL", "PETITIONER")
    # return {"transcribed_text": text, "respondent_reply": respondent_reply, "respondent_audio": respondent_audio_b64, "next_turn": "PETITIONER_REBUTTAL"}

# ================= PETITIONER REBUT =================
@router.post("/petitioner/rebut")
async def petitioner_rebut(req: ArgumentRequest, session_id: str, current_user=Depends(get_current_user)):
    session = await get_session_by_id(session_id, current_user["_id"])
    if session["current_party"] != "PETITIONER":
        raise HTTPException(400, "Not petitioner's turn")
    await push_history(session_id, current_user["_id"], "petitioner", req.text or "", type="rebuttal")
    await set_turn(session_id, "SESSION_END", None)
    return {"next_turn": "SESSION_END"}

@router.post("/petitioner/rebut/audio")
async def petitioner_rebut_audio(
    session_id: str = Form(...),
    file: UploadFile = File(...),
    current_user=Depends(get_current_user)
):
    session = await get_session_by_id(session_id, current_user["_id"])

    if session["current_party"] != "PETITIONER":
        raise HTTPException(status_code=400, detail="Not petitioner's turn")

    text = await process_audio(file)

    await push_history(
        session_id,
        current_user["_id"],
        "petitioner",
        text,
        type="rebuttal"
    )

    await set_turn(session_id, "SESSION_END", None)

    return {
        "transcribed_text": text,
        "next_turn": "SESSION_END"
    }

# ================= RESPONDENT RAG =================
@router.post("/respondent/rag")
async def respondent_rag(session_id: str, current_user=Depends(get_current_user)):
    session = await get_session_by_id(session_id, current_user["_id"])
    original_arg = session.get("original_petitioner_argument", "")
    history = session.get("history", [])
    response = await asyncio.to_thread(run_opponent_rag, case_key=session["case_id"], argument=original_arg, history=history)
    respondent_argument = response.get("response") if isinstance(response, dict) else str(response)
    await push_history(session_id, current_user["_id"], "respondent", respondent_argument)

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
    return {"respondent_argument": respondent_argument, "judge_question": judge_q, "respondent_reply": respondent_reply, "next_turn": "PETITIONER_REBUTTAL"}

# ================= EVALUATION =================
@router.post("/evaluate")
async def evaluate_user_only(session_id: str, current_user=Depends(get_current_user)):
    session = await get_session_by_id(session_id, current_user["_id"])
    if session.get("next_turn") != "SESSION_END":
        raise HTTPException(400, "Session not ended yet")

    history = session.get("history", [])
    main_argument, judge_responses, rebuttals = [], [], []

    for idx, h in enumerate(history):
        if h.get("role") != "petitioner":
            continue
        t = h.get("type") or ("argument" if idx == 0 else "rebuttal" if idx == len(history)-1 else "reply_to_judge")
        if t == "argument":
            main_argument.append(h.get("text", ""))
        elif t == "reply_to_judge":
            judge_responses.append(h.get("text", ""))
        elif t == "rebuttal":
            rebuttals.append(h.get("text", ""))

    main_argument_text = "\n\n".join(main_argument)
    judge_response_text = "\n\n".join(judge_responses)
    rebuttal_text = "\n\n".join(rebuttals)

    last_judge_q = next((h["text"] for h in reversed(history) if h.get("role")=="judge"), "")
    last_respondent_arg = next((h["text"] for h in reversed(history) if h.get("role")=="respondent"), "")

    retrieved_context_text = retrieve_context(main_argument_text, last_judge_q, session.get("case_type", "default"))

    prompt = build_prompt(main_argument_text, last_judge_q, judge_response_text, last_respondent_arg, rebuttal_text, RUBRIC_TEXT, retrieved_context_text)
    raw_eval_result = await asyncio.to_thread(call_llm, prompt)

    import re, json
    scores = {}
    overall_feedback = "Evaluation completed."
    if raw_eval_result:
        if isinstance(raw_eval_result, dict):
            eval_obj = raw_eval_result
        else:
            match = re.search(r"\{.*\}", str(raw_eval_result), re.DOTALL)
            eval_obj = json.loads(match.group()) if match else None
        if eval_obj:
            scores = eval_obj.get("scores", {})
            overall_feedback = eval_obj.get("overall_feedback", overall_feedback)

    await live_sessions_collection.update_one(
        {"_id": session["_id"]},
        {"$push": {"evaluation_history": {"party":"petitioner","timestamp":datetime.utcnow(),"evaluation":{"scores":scores,"overall_feedback":overall_feedback}}}}
    )

    return {"session_id": str(session["_id"]), "evaluation":{"scores":scores,"overall_feedback":overall_feedback}}