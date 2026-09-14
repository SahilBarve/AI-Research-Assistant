
"""
Alembic Environment Configuration.

This file connects Alembic with our SQLAlchemy models
and application database configuration.

Alembic uses this file when creating and running
database migrations.
"""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from app.core.config import get_settings
from app.database.session import Base

# Import models so SQLAlchemy registers their tables
# in Base.metadata.
from app.models.domain import (
    DocumentModel,
    ConversationModel,
    MessageModel,
)


# =========================================================
# ALEMBIC CONFIGURATION
# =========================================================

config = context.config


# =========================================================
# LOGGING
# =========================================================

if config.config_file_name is not None:
    fileConfig(config.config_file_name)


# =========================================================
# DATABASE URL
# =========================================================

settings = get_settings()

config.set_main_option(
    "sqlalchemy.url",
    settings.database_url,
)


# =========================================================
# SQLALCHEMY METADATA
# =========================================================

# Alembic compares this metadata with the actual
# PostgreSQL database schema when using --autogenerate.
target_metadata = Base.metadata


# =========================================================
# OFFLINE MIGRATIONS
# =========================================================

def run_migrations_offline() -> None:
    """
    Run migrations without creating a live database connection.

    Alembic generates SQL statements that can be executed
    against the database later.
    """

    url = config.get_main_option("sqlalchemy.url")

    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={
            "paramstyle": "named",
        },
    )

    with context.begin_transaction():
        context.run_migrations()


# =========================================================
# ONLINE MIGRATIONS
# =========================================================

def run_migrations_online() -> None:
    """
    Run migrations using a live database connection.

    This is the normal mode we will use during development.
    """

    connectable = engine_from_config(
        config.get_section(
            config.config_ini_section,
            {},
        ),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:

        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()


# =========================================================
# MIGRATION MODE
# =========================================================

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

