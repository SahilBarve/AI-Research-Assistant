"""
Global exception handlers for the AI Research Assistant.
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.logging import logger
from app.exceptions.custom_exceptions import (
    AIResearchAssistantException,
)


def register_exception_handlers(app: FastAPI) -> None:
    """
    Registers all custom exception handlers with the FastAPI application.
    """

    @app.exception_handler(AIResearchAssistantException)
    async def ai_exception_handler(
        request: Request,
        exc: AIResearchAssistantException,
    ):
        """
        Handles all application-specific exceptions.
        """

        logger.error(exc.message)

        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": {
                    "message": exc.message,
                },
            },
        )