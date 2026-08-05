from fastapi import FastAPI

from pathlib import Path
from app.core.config import get_settings
from app.core.logging import logger
from app.api.router import api_router
from app.exceptions.handlers import register_exception_handlers

settings = get_settings()

Path(settings.upload_dir).mkdir(
    parents=True,
    exist_ok=True
)

app = FastAPI(  #Everything in our backend revolves around this app object.
    title="AI Research Assistant",
    version="1.0.0",
    description="Production-grade AI Knowledge Intelligence Platform"
)

register_exception_handlers(app)
logger.info("AI Research Assistant started successfully.")

app.include_router(api_router)



@app.get("/") # This means "If someone sends a GET request to /, execute the function below." Whenever someone accesses: GET / FastAPI calls: root() automatically.
def root():
    return {
        "status": "running",
        "message": "Welcome to AI Research Assistant 🚀"
    }