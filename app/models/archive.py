# app/api/archive.py
from fastapi import APIRouter, HTTPException
from datetime import datetime
from bson import ObjectId
from app.database.mongodb import (
    live_sessions_collection,
    case_history_collection
)

router = APIRouter(prefix="/archive", tags=["Archive"])

@router.post("/{session_id}")
async def archive_session(session_id: str):
    session = await live_sessions_collection.find_one(
        {"_id": ObjectId(session_id)}
    )

    if not session or session["status"] != "closed":
        raise HTTPException(status_code=400)

    history_doc = {
        "user_id": session["user_id"],
        "case_id": session["case_id"],
        "case_type": session["case_type"],
        "session_date": datetime.utcnow(),
        "transcript": session["history"],
        "result": session.get("result")
    }

    await case_history_collection.insert_one(history_doc)
    return {"message": "Archived"}
