from fastapi import APIRouter

router = APIRouter()

@router.get("/cases")
async def get_cases():
    return {"cases": []}