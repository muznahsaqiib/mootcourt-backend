from fastapi import APIRouter, HTTPException
from app.database.mongodb import cases_collection, moot_problems_collection

router = APIRouter()

# =========================
# GET ALL CASES
# =========================
@router.get("/cases")
async def get_cases():
    cases = []
    async for doc in cases_collection.find():
        doc["_id"] = str(doc["_id"])  # Convert ObjectId to str
        cases.append(doc)
    return cases  # ✅ Return a list


# =========================
# GET SINGLE CASE BY STRING ID
# =========================
@router.get("/cases/{case_id}")
async def get_case(case_id: str):
    # Clean the ID: remove accidental quotes from URL param
    clean_id = case_id.strip("'\"")
    case = await cases_collection.find_one({"id": clean_id})
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    case["_id"] = str(case.get("_id", ""))
    return case


# =========================
# GET ALL MOOT PROBLEMS
# =========================
@router.get("/moot-problems")
async def get_moot_problems():
    problems = await moot_problems_collection.find({}, {"_id": 0}).to_list(length=100)
    return {"moot_problems": problems}


# =========================
# GET SINGLE MOOT PROBLEM BY STRING ID
# =========================
@router.get("/moot-problems/{problem_id}")
async def get_single_problem(problem_id: str):
    # Clean the ID
    clean_id = problem_id.strip("'\"")
    problem = await moot_problems_collection.find_one({"id": clean_id}, {"_id": 0})
    if not problem:
        raise HTTPException(status_code=404, detail="Moot problem not found")
    return problem
