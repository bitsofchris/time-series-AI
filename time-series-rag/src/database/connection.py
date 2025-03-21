"""
Database connection management for time-series-rag project.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


def get_db_engine(db_url: str = None) -> Engine:
    """
    Get SQLAlchemy engine for database connection.

    Args:
        db_url: SQLAlchemy database URL. If None, uses DATABASE_URL env variable.

    Returns:
        SQLAlchemy Engine instance

    Raises:
        ValueError: If database URL not provided and DATABASE_URL env var not set
    """
    db_url = db_url or os.environ.get("DATABASE_URL")
    if not db_url:
        raise ValueError(
            "Database URL not provided and DATABASE_URL env variable not set"
        )

    return create_engine(db_url)


def get_connection_string() -> str:
    """
    Get database connection string from environment variables.

    Returns:
        Connection string in SQLAlchemy format

    Raises:
        ValueError: If required environment variables are not set
    """
    db_user = os.environ.get("POSTGRES_USER", "timeseriesrag")
    db_password = os.environ.get("POSTGRES_PASSWORD", "devpassword")
    db_host = os.environ.get("DB_HOST", "localhost")
    db_port = os.environ.get("DB_PORT", "5432")
    db_name = os.environ.get("POSTGRES_DB", "market_data")

    return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
