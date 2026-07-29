"""
Application logging configuration.

This module configures a centralized logger that can be
used across the application.
"""

import logging
from pathlib import Path #This creates logs/ automatically.


# Create logs directory if it doesn't exist
Path("logs").mkdir(exist_ok=True)


logging.basicConfig(
    level=logging.INFO, # This determines which messages are recorded. Logging levels are:DEBUG,INFO,WARNING,ERROR,CRITICAl

    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/app.log"),#Writes logs into logs/app.log
        logging.StreamHandler(),#Also prints logs inside the terminal.This is useful while developing.
    ],
)


logger = logging.getLogger("ai_research_assistant")


'''By using above statement 
Now anywhere in the project we can write:

from app.core.logging import logger

logger.info("Document uploaded")

instead of

print("uploaded")'''