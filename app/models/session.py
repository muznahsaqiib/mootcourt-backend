# backend/app/models/session.py

from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime


class TranscriptEntry(BaseModel):
    """
    Represents a single turn in the session.
    role: user | opponent | judge
    text: actual text of argument/question/reply
    timestamp: UTC datetime of the turn
    """
    role: str
    text: str
    timestamp: datetime


class SessionResult(BaseModel):
    """
    Stores evaluation results for a session.
    Supports both user and opponent scores, detailed scoring, and feedback.
    """
    user_score: Optional[int]
    opponent_score: Optional[int]
    evaluator_feedback_user: Optional[str]
    evaluator_feedback_opponent: Optional[str]
    detailed_scores_user: Optional[Dict]  # e.g., JSON from evaluator RAG
    detailed_scores_opponent: Optional[Dict]
    evaluated_at: Optional[datetime]  # when evaluation was performed

class LiveSession(BaseModel):
    session_id: str
    user_id: str
    case_id: str
    mode: str
    role: str                      # 🔥 ADD THIS
    history: List[TranscriptEntry] = Field(default_factory=list)
    status: str = "active"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    result: Optional[SessionResult] = None

class CaseHistory(BaseModel):
    session_id: str

    user_id: str
    case_id: str
    case_title: Optional[str]

    role: str
    mode: str

    transcript: List[TranscriptEntry]
    result: Optional[SessionResult]

    duration_seconds: Optional[int]

    started_at: datetime
    ended_at: datetime

    created_at: datetime = Field(default_factory=datetime.utcnow)