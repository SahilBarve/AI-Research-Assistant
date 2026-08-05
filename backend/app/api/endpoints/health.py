#Important: This endpoint is only for learning and testing. We'll remove it later.
from fastapi import APIRouter
from app.exceptions.custom_exceptions import DocumentTooLargeException

router = APIRouter(
    prefix="/health",
    tags=["Health"]
)

@router.get("/test-exception")
def test_exception():
    raise DocumentTooLargeException(
        "This is a test exception."
    )   

@router.get("/")
def health_check():
    return {
        "status": "healthy",
        "service": "AI Research Assistant"
    }