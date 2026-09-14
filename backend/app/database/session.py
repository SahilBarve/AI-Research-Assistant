
"""
Database Session Configuration.

This module creates the SQLAlchemy engine,
database session factory, and declarative base
used by the application's database models.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.core.config import get_settings


# =========================================================
# SETTINGS
# =========================================================

settings = get_settings()


# =========================================================
# DATABASE ENGINE
# =========================================================

engine = create_engine(  #create_engine() creates the Engine object that manages this communication.
    settings.database_url,
    pool_pre_ping=True,
    echo=settings.debug,
)


# =========================================================
# DATABASE SESSION FACTORY
# =========================================================

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


# =========================================================
# SQLALCHEMY BASE
# =========================================================

class Base(DeclarativeBase):
    pass


# =========================================================
# FASTAPI DATABASE DEPENDENCY
# =========================================================

def get_db():
    """
    Create a database session for a request.

    The session is automatically closed after
    the request finishes, even if an exception occurs.
    """

    db = SessionLocal()

    try:
        yield db

    finally:
        db.close()

