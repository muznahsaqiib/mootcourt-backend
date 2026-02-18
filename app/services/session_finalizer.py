from datetime import datetime
from app.models.session import CaseHistory, LiveSession

def finalize_live_session(
    live: LiveSession,
    case_title: str
) -> CaseHistory:

    ended_at = datetime.utcnow()
    duration = int((ended_at - live.created_at).total_seconds())

    return CaseHistory(
        session_id=live.session_id,
        user_id=live.user_id,
        case_id=live.case_id,
        case_title=case_title,
        role=live.role,
        mode=live.mode,
        transcript=live.history,
        result=live.result,
        duration_seconds=duration,
        started_at=live.created_at,
        ended_at=ended_at
    )
