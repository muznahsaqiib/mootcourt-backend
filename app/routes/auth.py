# backend/app/routes/auth.py
from fastapi import APIRouter, HTTPException, Response, Cookie, Depends
from pydantic import BaseModel, EmailStr
from typing import Optional
import uuid
from bson import ObjectId
from datetime import datetime
from passlib.context import CryptContext

from app.database.mongodb import users_collection, sessions_collection, live_sessions_collection

router = APIRouter()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ------------------------------- Pydantic Models -------------------------------
class UserRegister(BaseModel):
    username: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

# ------------------------------- Password Helpers -------------------------------
def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

# ------------------------------- Reusable Dependency -------------------------------
async def get_current_user(session_cookie: Optional[str] = Cookie(default=None)):
    """Get user from auth cookie (renamed to avoid conflict with path params)."""
    if not session_cookie:
        raise HTTPException(status_code=401, detail="Not authenticated")
    session = await sessions_collection.find_one({"session_id": session_cookie})
    if not session:
        raise HTTPException(status_code=401, detail="Invalid session")
    user_id = session.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Session invalid: missing user_id")
    user = await users_collection.find_one({"_id": ObjectId(user_id)})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

# ------------------------------- Registration -------------------------------
@router.post("/auth/register")
async def register_user(user: UserRegister, response: Response):
    if await users_collection.find_one({"username": user.username}):
        raise HTTPException(status_code=400, detail="Username already taken")
    if await users_collection.find_one({"email": user.email}):
        raise HTTPException(status_code=400, detail="Email already registered")

    user_dict = {
        "username": user.username,
        "email": user.email,
        "hashed_password": hash_password(user.password),
        "created_at": datetime.utcnow()
    }

    result = await users_collection.insert_one(user_dict)
    user_id = result.inserted_id

    # Generate session cookie
    session_id = str(uuid.uuid4())
    await sessions_collection.insert_one({
        "session_id": session_id,
        "user_id": user_id,
        "created_at": datetime.utcnow()
    })

    response.set_cookie(
        key="session_cookie",  
        value=session_id,
        httponly=True,
        samesite="lax",
        secure=False,
        path="/"
    )

    return {
        "message": f"Hi {user.username}, your account has been created.",
        "user": {
            "user_id": str(user_id),
            "username": user.username,
            "email": user.email,
        }
    }

# ------------------------------- Login -------------------------------
@router.post("/auth/login")
async def login_user(user: UserLogin, response: Response):
    db_user = await users_collection.find_one({"username": user.username})
    if not db_user or not verify_password(user.password, db_user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    session_id = str(uuid.uuid4())
    await sessions_collection.insert_one({
        "session_id": session_id,
        "user_id": db_user["_id"],
        "created_at": datetime.utcnow()
    })

    response.set_cookie(
        key="session_cookie", 
        value=session_id,
        httponly=True,
        samesite="lax",
        secure=False,
        path="/"
    )

    return {
        "message": f"Hi {db_user['username']}, you’re now logged in.",
        "user": {
            "user_id": str(db_user["_id"]),
            "username": db_user["username"],
            "email": db_user["email"]
        }
    }

# ------------------------------- Logout -------------------------------
@router.post("/auth/logout")
async def logout_user(response: Response, session_cookie: Optional[str] = Cookie(default=None)):
    if session_cookie:
        await sessions_collection.delete_one({"session_id": session_cookie})
        response.delete_cookie(
            key="session_cookie",
            path="/",
            samesite="lax",
            secure=False,
            httponly=True
        )
    return {"message": "Logged out successfully"}

# ------------------------------- Get Current User -------------------------------
@router.get("/auth/me")
async def get_me(current_user=Depends(get_current_user)):
    return {
        "user_id": str(current_user["_id"]),
        "username": current_user["username"],
        "email": current_user["email"]
    }

@router.get("/auth/user/history")
async def get_user_history(current_user=Depends(get_current_user)):
    user_id = current_user["_id"]  

   
    sessions = await live_sessions_collection.find(
        {"user_id": user_id}
    ).sort("created_at", -1).to_list(length=100)

    formatted_sessions = []

    for s in sessions:
        latest_eval = None
        if s.get("evaluation_history"):
            latest_eval = s["evaluation_history"][-1].get("evaluation")

        # Calculate duration in seconds
        duration_seconds = 0
        if s.get("created_at") and s.get("updated_at"):
            duration_seconds = int(
                (s["updated_at"] - s["created_at"]).total_seconds()
            )

        # Extract key metrics safely
        detailed_scores_user = {}
        if latest_eval and latest_eval.get("scores"):
            scores = latest_eval["scores"]
            detailed_scores_user = {
                "clarity": scores.get("Organization & Clarity", {}).get("score"),
                "responsiveness": scores.get("Responsiveness to Judge", {}).get("score"),
                "structure": scores.get("Organization & Clarity", {}).get("score")
            }

        formatted_sessions.append({
            "session_id": str(s["_id"]),
            "case_id": s.get("case_id"),
            "case_title": s.get("case_id"),  
            "case_type": s.get("case_type"),
            "duration_seconds": duration_seconds,
            "next_turn": s.get("next_turn"),
            "current_party": s.get("current_party"),
            "result": {
                "user_score": latest_eval.get("scores", {}).get("overall_score") if latest_eval else None,
                "detailed_scores_user": detailed_scores_user,
                "overall_feedback": latest_eval.get("overall_feedback") if latest_eval else None
            }
        })

    return formatted_sessions