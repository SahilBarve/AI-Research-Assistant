import shutil    
from fastapi import APIRouter, File, UploadFile
from pathlib import Path

from app.exceptions.custom_exceptions import InvalidDocumentTypeException
from app.core.config import get_settings
from app.schemas.document import DocumentUploadResponse

router = APIRouter(
    prefix="/documents",
    tags=["Documents"] #Without tags, Swagger shows a long flat list of endpoints.
)

settings = get_settings()

ALLOWED_EXTENSIONS = {".pdf"}
MAX_FILENAME_LENGTH = 100
@router.post( #This creates: POST /documents/upload and tells FastAPI:"The response will follow the DocumentUploadResponse schema."
    "/upload",
    response_model=DocumentUploadResponse
)

async def upload_document(
    file: UploadFile = File(...)
):
    # Validate file extension
    extension = Path(file.filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
      raise InvalidDocumentTypeException()

    #Save uploaded file
    file_path = Path(settings.upload_dir) / file.filename

    with open(file_path, "wb") as buffer:
     shutil.copyfileobj(file.file, buffer)
    return DocumentUploadResponse(
        message="Document uploaded successfully.",
        filename=file.filename
    )



