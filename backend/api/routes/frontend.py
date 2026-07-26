from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from api.config import FRONTEND_STATIC_DIR


router = APIRouter()


@router.get("/")
def serve_frontend() -> FileResponse:
    index_path = FRONTEND_STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(index_path)