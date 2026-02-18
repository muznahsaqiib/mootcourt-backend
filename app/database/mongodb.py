# app/database/mongodb.py

from motor.motor_asyncio import AsyncIOMotorClient

MONGO_URL = "mongodb://localhost:27017"

client = AsyncIOMotorClient(MONGO_URL)

db = client["mootcourt"]  # ✅ YOUR actual database name
users_collection = db.get_collection("users")
cases_collection = db.get_collection("cases")
moot_problems_collection = db["moot-problems"]
sessions_collection = db["sessions"]  # Required for session-based auth
live_sessions_collection = db["live_sessions"]
case_histories_collection = db["case_histories"]
judge_questions_collection = db["judge_questions"]
session_history_collection = db["session_history"]