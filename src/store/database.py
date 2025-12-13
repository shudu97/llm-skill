"""
Database engine and shared metadata.

All custom tables are defined here so they can be created together.
Swap the connection string to migrate from SQLite to PostgreSQL.
"""

from sqlalchemy import Column, MetaData, String, Table, create_engine

metadata = MetaData()

conversations = Table(
    "conversations",
    metadata,
    Column("thread_id", String, primary_key=True),
    Column("user_id", String, nullable=False),
    Column("title", String, nullable=False),
    Column("created_at", String, nullable=False),
    Column("updated_at", String, nullable=False),
)


def create_db_engine(db_url: str):
    """Create a SQLAlchemy engine from a connection URL.

    SQLite:     create_db_engine("sqlite:///agent.db")
    PostgreSQL: create_db_engine("postgresql://user:pass@host/dbname")
    """
    engine = create_engine(db_url)
    metadata.create_all(engine)
    return engine
