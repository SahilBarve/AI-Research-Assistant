from pathlib import Path

from fastapi import (
    APIRouter,
    File,
    UploadFile,
)

from app.core.config import get_settings

from app.schemas.document import (
    DocumentUploadResponse,
)

from app.core.dependencies import (
    get_document_service,
)

from app.exceptions.custom_exceptions import (
    InvalidDocumentTypeException,
    DocumentTooLargeException,
)


# =========================================================
# ROUTER
# =========================================================

router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)


# =========================================================
# CONFIGURATION
# =========================================================

settings = get_settings()


# =========================================================
# DOCUMENT SERVICE
# =========================================================

document_service = (
    get_document_service()
)


# =========================================================
# CONSTANTS
# =========================================================

MAX_FILE_SIZE = (
    settings.max_upload_size_mb
    * 1024
    * 1024
)
ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".md",
    ".markdown",
}


# =========================================================
# UPLOAD DOCUMENT
# =========================================================

@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
)
async def upload_document(
    file: UploadFile = File(...),
):

    # -----------------------------------------------------
    # Validate filename
    # -----------------------------------------------------

    if not file.filename:

        raise InvalidDocumentTypeException()


    # -----------------------------------------------------
    # Validate extension
    # -----------------------------------------------------

    extension = (
        Path(file.filename)
        .suffix
        .lower()
    )

    if extension not in ALLOWED_EXTENSIONS:

        raise InvalidDocumentTypeException()


    # -----------------------------------------------------
    # Validate file size
    # -----------------------------------------------------

    content = await file.read()

    if len(content) > MAX_FILE_SIZE:

        raise DocumentTooLargeException(
            f"Maximum allowed file size is "
            f"{settings.max_upload_size_mb} MB."
        )


    # -----------------------------------------------------
    # Reset file pointer
    # -----------------------------------------------------

    file.file.seek(0)


    # -----------------------------------------------------
    # Check duplicate
    # -----------------------------------------------------

    document_service.check_duplicate(
        file.filename
    )


    # -----------------------------------------------------
    # Save document
    # -----------------------------------------------------

    file_path = (
        document_service.save_uploaded_file(
            file
        )
    )


    # -----------------------------------------------------
    # Index document
    # -----------------------------------------------------

    result = (
        document_service.index_document(
            file_path=file_path,
            filename=file.filename,
        )
    )


    # -----------------------------------------------------
    # Response
    # -----------------------------------------------------

    return DocumentUploadResponse(
        message="Document uploaded successfully.",
        filename=file.filename,
    )


# =========================================================
# RE-INDEX DOCUMENT
# =========================================================

@router.post(
    "/{filename}/reindex",
)
async def reindex_document(
    filename: str,
):

    result = (
        document_service.reindex_document(
            filename
        )
    )

    return {
        "message": (
            "Document re-indexed successfully."
        ),
        "filename": result["filename"],
        "chunks": result["chunks"],
    }


# =========================================================
# DELETE DOCUMENT
# =========================================================

@router.delete(
    "/{filename}",
)
async def delete_document(
    filename: str,
):

    result = (
        document_service.delete_document(
            filename
        )
    )

    return {
        "message": (
            "Document deleted successfully."
        ),
        "filename": result["filename"],
    }