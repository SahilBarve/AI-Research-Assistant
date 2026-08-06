# Standard Library
import shutil    
from pathlib import Path
# Third-party
from fastapi import APIRouter, File, UploadFile
# Local application
from backend.app.services.document_processor.pdf_processors import PDFProcessor
from app.core.config import get_settings
from app.schemas.document import DocumentUploadResponse
from app.exceptions.custom_exceptions import (
    InvalidDocumentTypeException,
    DocumentTooLargeException,
    DuplicateDocumentException,
)

router = APIRouter(
    prefix="/documents",
    tags=["Documents"] #Without tags, Swagger shows a long flat list of endpoints.
)
# Configuration
settings = get_settings()
# Services
pdf_processor = PDFProcessor()

# Constants
MAX_FILE_SIZE = settings.max_upload_size_mb * 1024 * 1024
ALLOWED_EXTENSIONS = {".pdf"}
MAX_FILENAME_LENGTH = 100

@router.post( #This creates: POST /documents/upload and tells FastAPI:"The response will follow the DocumentUploadResponse schema."
    "/upload",
    response_model=DocumentUploadResponse
)

async def upload_document(
    file: UploadFile = File(...)
):
    # ----------------------------------------
    # Validate file extension
    # ----------------------------------------
    extension = Path(file.filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise InvalidDocumentTypeException()

    # ----------------------------------------
    # Validate file size
    # ----------------------------------------
    content = await file.read()

    if len(content) > MAX_FILE_SIZE:
        raise DocumentTooLargeException(
            f"Maximum allowed file size is {settings.max_upload_size_mb} MB."
        )

    # Reset file pointer
    file.file.seek(0) # while reading the file our pointer reaches at the end of the file and if we start writing from here we will write the wrong file and so this seek sets the pointer back to the start of the file for writing

    # ----------------------------------------
    # Save uploaded file
    # ----------------------------------------
    file_path = Path(settings.upload_dir) / file.filename

    if file_path.exists():
     raise DuplicateDocumentException()
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    extracted_text = pdf_processor.extract_text(str(file_path))

    cleaned_text = pdf_processor.clean_text(extracted_text)

    print("\n========== Cleaned Text ==========\n")
    print(cleaned_text)
    print("\n=================================\n")
    
    print("\n========== Extracted Text ==========\n")
    print(extracted_text)
    print("\n===================================\n")

    return DocumentUploadResponse(
        message="Document uploaded successfully.",
        filename=file.filename
        )
