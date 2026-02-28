import io
import subprocess
import logging
import random
import asyncio
import os
import json
from datetime import datetime
from typing import Optional
from pydantic import BaseModel
from bson import ObjectId
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form
from fastapi.responses import StreamingResponse
import base64
from app.routes.auth import get_current_user
from app.database.mongodb import live_sessions_collection, judge_questions_collection
from rag.moot_rag.audio.stt import speech_to_text
from rag.moot_rag.audio.tts import text_to_speech, tts_to_bytes
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


async def push_history(session_id: str, user_id, role: str, text: str,
                       type: Optional[str] = None, audio_b64: Optional[str] = None):
    entry = {"role": role, "text": text, "timestamp": datetime.utcnow()}
    if type:
        entry["type"] = type
    if audio_b64:
        entry["audio"] = audio_b64
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


# ================= TTS HELPER =================
async def generate_audio_b64(text: str, voice: str = "default") -> Optional[str]:
    if not text or not text.strip():
        return None
    try:
        audio_bytes = await asyncio.to_thread(text_to_speech, text, voice)
        return base64.b64encode(audio_bytes).decode()
    except Exception as e:
        logger.warning(f"TTS failed for voice={voice}: {e}")
        return None


# ================= AUDIO UTILITY =================
async def process_audio(file: UploadFile) -> str:
    webm_bytes = await file.read()
    if not webm_bytes:
        return ""

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
        wav_stream = io.BytesIO(process.stdout)
        wav_stream.seek(0)
        text = await asyncio.to_thread(speech_to_text, wav_stream)
        return text.strip() if text else ""
    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg error: {e.stderr.decode() if e.stderr else str(e)}")
        return ""


# ================= SSE HELPER =================
def sse_event(event: str, data: dict) -> str:
    """Format a single SSE message."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


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


# ================= PETITIONER ARGUMENT (TEXT) =================
@router.post("/petitioner/argument")
async def petitioner_argument(req: ArgumentRequest, session_id: str, current_user=Depends(get_current_user)):
    session = await get_session_by_id(session_id, current_user["_id"])
    if session["current_party"] != "PETITIONER":
        raise HTTPException(400, "Not petitioner's turn")

    text = req.text or ""

    await live_sessions_collection.update_one(
        {"_id": ObjectId(session_id)},
        {"$set": {"original_petitioner_argument": text}}
    )

    await push_history(session_id, current_user["_id"], "petitioner", text, type="argument")

    judge_q = await get_judge_question(session["case_type"])
    if judge_q:
        await push_history(session_id, current_user["_id"], "judge", judge_q)

    await set_turn(session_id, "PETITIONER_REPLY_TO_JUDGE" if judge_q else "RESPONDENT_RAG", "PETITIONER")

    return {
        "judge_question": judge_q,
        "next_turn": "PETITIONER_REPLY_TO_JUDGE" if judge_q else "RESPONDENT_RAG"
    }


# ================= PETITIONER ARGUMENT (AUDIO) =================
@router.post("/petitioner/argument/audio")
async def petitioner_argument_audio(
    session_id: str = Form(...),
    file: UploadFile = File(...),
    current_user=Depends(get_current_user)
):
    session = await get_session_by_id(session_id, current_user["_id"])
    if session["current_party"] != "PETITIONER":
        raise HTTPException(400, "Not petitioner's turn")

    text = await process_audio(file)
    if not text:
        return {
            "transcribed_text": "",
            "message": "Could not detect speech. Please speak again.",
            "judge_question": None,
            "judge_audio": None,
            "next_turn": session.get("next_turn", "PETITIONER_ARGUMENT")
        }

    await live_sessions_collection.update_one(
        {"_id": ObjectId(session_id)},
        {"$set": {"original_petitioner_argument": text}}
    )

    petitioner_audio_b64 = await generate_audio_b64(text, "petitioner")
    await push_history(session_id, current_user["_id"], "petitioner", text,
                       type="argument", audio_b64=petitioner_audio_b64)

    judge_q = await get_judge_question(session["case_type"])
    judge_audio_b64 = None

    if judge_q:
        judge_audio_b64 = await generate_audio_b64(judge_q, "judge")
        await push_history(session_id, current_user["_id"], "judge", judge_q,
                           audio_b64=judge_audio_b64)

    next_turn = "PETITIONER_REPLY_TO_JUDGE" if judge_q else "RESPONDENT_RAG"
    await set_turn(session_id, next_turn, "PETITIONER")

    return {
        "transcribed_text": text,
        "judge_question": judge_q,
        "judge_audio": judge_audio_b64,
        "next_turn": next_turn
    }


# ================= PETITIONER REPLY TO JUDGE (TEXT) =================
@router.post("/petitioner/reply")
async def petitioner_reply(req: ArgumentRequest, session_id: str, current_user=Depends(get_current_user)):
    session = await get_session_by_id(session_id, current_user["_id"])
    if session["current_party"] != "PETITIONER":
        raise HTTPException(400, "Not petitioner's turn")

    text = req.text or ""
    await push_history(session_id, current_user["_id"], "petitioner", text, type="reply_to_judge")
    await set_turn(session_id, "RESPONDENT_RAG", "RESPONDENT")
    return {"next_turn": "RESPONDENT_RAG"}


# ================= PETITIONER REPLY TO JUDGE (AUDIO) =================
@router.post("/petitioner/reply/audio")
async def petitioner_reply_audio(
    session_id: str = Form(...),
    file: UploadFile = File(...),
    current_user=Depends(get_current_user)
):
    session = await get_session_by_id(session_id, current_user["_id"])
    if session["current_party"] != "PETITIONER":
        raise HTTPException(400, "Not petitioner's turn")

    text = await process_audio(file)
    if not text:
        return {"transcribed_text": "", "message": "Could not detect speech.",
                "next_turn": session.get("next_turn")}

    petitioner_audio_b64 = await generate_audio_b64(text, "petitioner")
    await push_history(session_id, current_user["_id"], "petitioner", text,
                       type="reply_to_judge", audio_b64=petitioner_audio_b64)
    await set_turn(session_id, "RESPONDENT_RAG", "RESPONDENT")

    return {"transcribed_text": text, "next_turn": "RESPONDENT_RAG"}


# ================= PETITIONER REBUTTAL (TEXT) =================
@router.post("/petitioner/rebut")
async def petitioner_rebut(req: ArgumentRequest, session_id: str, current_user=Depends(get_current_user)):
    session = await get_session_by_id(session_id, current_user["_id"])
    if session["current_party"] != "PETITIONER":
        raise HTTPException(400, "Not petitioner's turn")

    text = req.text or ""
    await push_history(session_id, current_user["_id"], "petitioner", text, type="rebuttal")
    await set_turn(session_id, "SESSION_END", None)
    return {"next_turn": "SESSION_END"}


# ================= PETITIONER REBUTTAL (AUDIO) =================
@router.post("/petitioner/rebut/audio")
async def petitioner_rebut_audio(
    session_id: str = Form(...),
    file: UploadFile = File(...),
    current_user=Depends(get_current_user)
):
    session = await get_session_by_id(session_id, current_user["_id"])
    if session["current_party"] != "PETITIONER":
        raise HTTPException(400, "Not petitioner's turn")

    text = await process_audio(file)
    petitioner_audio_b64 = await generate_audio_b64(text, "petitioner")

    await push_history(session_id, current_user["_id"], "petitioner", text,
                       type="rebuttal", audio_b64=petitioner_audio_b64)
    await set_turn(session_id, "SESSION_END", None)

    return {
        "transcribed_text": text,
        "petitioner_audio": petitioner_audio_b64,
        "next_turn": "SESSION_END"
    }


# ================= RESPONDENT RAG (SSE STREAMING) =================
@router.get("/respondent/rag/stream")
async def respondent_rag_stream(session_id: str, current_user=Depends(get_current_user)):
    """
    Streams respondent RAG steps as SSE events so the frontend can render
    each piece immediately instead of waiting for the full chain.

    Event sequence:
      1. respondent_argument  — RAG argument text + audio
      2. judge_question       — judge question text + audio  (if any)
      3. respondent_reply     — RAG reply to judge + audio   (if judge asked)
      4. done                 — final next_turn signal
    """
    session = await get_session_by_id(session_id, current_user["_id"])

    async def event_generator():
        original_arg = session.get("original_petitioner_argument", "")
        history = session.get("history", [])
        case_id = session["case_id"]
        case_type = session.get("case_type")

        # ── STEP 1: Respondent RAG argument ──────────────────────────────
        try:
            rag_response = await asyncio.to_thread(
                run_opponent_rag,
                case_key=case_id,
                argument=original_arg,
                history=history,
                case_type=case_type
            )
            respondent_argument = (
                rag_response.get("response")
                if isinstance(rag_response, dict)
                else str(rag_response)
            )
        except Exception as e:
            logger.error(f"Respondent RAG failed: {e}")
            yield sse_event("error", {"message": "Respondent RAG failed."})
            return

        respondent_audio_b64 = await generate_audio_b64(respondent_argument, "respondent")

        await push_history(session_id, current_user["_id"], "respondent",
                           respondent_argument, audio_b64=respondent_audio_b64)

        # ── Emit immediately so frontend shows respondent argument now ──
        yield sse_event("respondent_argument", {
            "text": respondent_argument,
            "audio": respondent_audio_b64,
        })

        # ── STEP 2: Judge question ────────────────────────────────────────
        judge_q = await get_judge_question(case_type)
        judge_audio_b64 = None

        if judge_q:
            judge_audio_b64 = await generate_audio_b64(judge_q, "judge")
            await push_history(session_id, current_user["_id"], "judge",
                               judge_q, audio_b64=judge_audio_b64)

            # ── Emit judge question so frontend shows it now ──
            yield sse_event("judge_question", {
                "text": judge_q,
                "audio": judge_audio_b64,
            })

            # ── STEP 3: Respondent replies to judge ───────────────────────
            updated_session = await get_session_by_id(session_id, current_user["_id"])
            try:
                reply_response = await asyncio.to_thread(
                    run_opponent_rag,
                    case_key=case_id,
                    argument=judge_q,
                    history=updated_session["history"],
                    case_type=case_type
                )
                respondent_reply = (
                    reply_response.get("response")
                    if isinstance(reply_response, dict)
                    else str(reply_response)
                )
            except Exception as e:
                logger.error(f"Respondent reply RAG failed: {e}")
                yield sse_event("error", {"message": "Respondent reply failed."})
                return

            respondent_reply_audio_b64 = await generate_audio_b64(respondent_reply, "respondent")
            await push_history(session_id, current_user["_id"], "respondent",
                               respondent_reply, audio_b64=respondent_reply_audio_b64)

            # ── Emit respondent reply so frontend shows it now ──
            yield sse_event("respondent_reply", {
                "text": respondent_reply,
                "audio": respondent_reply_audio_b64,
            })

        # ── STEP 4: Finalise turn ─────────────────────────────────────────
        await set_turn(session_id, "PETITIONER_REBUTTAL", "PETITIONER")

        yield sse_event("done", {"next_turn": "PETITIONER_REBUTTAL"})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disables nginx buffering
        }
    )


# ── Keep the old POST endpoint as a non-streaming fallback ──────────────────
@router.post("/respondent/rag")
async def respondent_rag(session_id: str, current_user=Depends(get_current_user)):
    """Non-streaming fallback — waits for the full chain before returning."""
    session = await get_session_by_id(session_id, current_user["_id"])
    original_arg = session.get("original_petitioner_argument", "")
    history = session.get("history", [])

    rag_task = asyncio.to_thread(
        run_opponent_rag,
        case_key=session["case_id"],
        argument=original_arg,
        history=history,
        case_type=session.get("case_type")
    )
    judge_task = get_judge_question(session["case_type"])
    rag_response, judge_q = await asyncio.gather(rag_task, judge_task)

    respondent_argument = rag_response.get("response") if isinstance(rag_response, dict) else str(rag_response)

    async def _maybe_tts(text, voice):
        return await generate_audio_b64(text, voice) if text else None

    respondent_audio_b64, judge_audio_b64 = await asyncio.gather(
        _maybe_tts(respondent_argument, "respondent"),
        _maybe_tts(judge_q, "judge"),
    )

    await push_history(session_id, current_user["_id"], "respondent", respondent_argument,
                       audio_b64=respondent_audio_b64)

    respondent_reply = None
    respondent_reply_audio_b64 = None

    if judge_q:
        await push_history(session_id, current_user["_id"], "judge", judge_q,
                           audio_b64=judge_audio_b64)

        updated_session = await get_session_by_id(session_id, current_user["_id"])
        reply_response = await asyncio.to_thread(
            run_opponent_rag,
            case_key=session["case_id"],
            argument=judge_q,
            history=updated_session["history"],
            case_type=session.get("case_type")
        )
        respondent_reply = reply_response.get("response") if isinstance(reply_response, dict) else str(reply_response)
        respondent_reply_audio_b64 = await generate_audio_b64(respondent_reply, "respondent")
        await push_history(session_id, current_user["_id"], "respondent", respondent_reply,
                           audio_b64=respondent_reply_audio_b64)

    await set_turn(session_id, "PETITIONER_REBUTTAL", "PETITIONER")

    return {
        "respondent_argument": respondent_argument,
        "respondent_audio": respondent_audio_b64,
        "judge_question": judge_q,
        "judge_audio": judge_audio_b64,
        "respondent_reply": respondent_reply,
        "respondent_reply_audio": respondent_reply_audio_b64,
        "next_turn": "PETITIONER_REBUTTAL"
    }


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
        t = h.get("type") or ("argument" if idx == 0 else "rebuttal" if idx == len(history) - 1 else "reply_to_judge")
        if t == "argument":
            main_argument.append(h.get("text", ""))
        elif t == "reply_to_judge":
            judge_responses.append(h.get("text", ""))
        elif t == "rebuttal":
            rebuttals.append(h.get("text", ""))

    main_argument_text = "\n\n".join(main_argument)
    judge_response_text = "\n\n".join(judge_responses)
    rebuttal_text = "\n\n".join(rebuttals)

    last_judge_q = next((h["text"] for h in reversed(history) if h.get("role") == "judge"), "")
    last_respondent_arg = next((h["text"] for h in reversed(history) if h.get("role") == "respondent"), "")

    retrieved_context_text = retrieve_context(main_argument_text, last_judge_q, session.get("case_type", "default"))
    prompt = build_prompt(main_argument_text, last_judge_q, judge_response_text,
                          last_respondent_arg, rebuttal_text, RUBRIC_TEXT, retrieved_context_text)
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
        {"$push": {"evaluation_history": {
            "party": "petitioner",
            "timestamp": datetime.utcnow(),
            "evaluation": {"scores": scores, "overall_feedback": overall_feedback}
        }}}
    )

    return {"session_id": str(session["_id"]), "evaluation": {"scores": scores, "overall_feedback": overall_feedback}}