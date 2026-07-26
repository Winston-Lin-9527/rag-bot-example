from fastapi import APIRouter, HTTPException

from api.schemas.chat import ChatRequest, ChatResponse
from api.services.chat import answer_chat


router = APIRouter(prefix="/api")


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        return answer_chat(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc