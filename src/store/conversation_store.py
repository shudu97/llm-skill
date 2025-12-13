"""
Conversation metadata store.

Tracks conversation threads (thread_id, title, timestamps) per user.
Backed by SQLAlchemy Core — swap the engine URL to move from SQLite to PostgreSQL.
"""

from datetime import datetime, timezone

from sqlalchemy import Engine, desc, insert, select, update

from src.store.database import conversations


class ConversationStore:
    def __init__(self, engine: Engine, user_id: str) -> None:
        self.engine = engine
        self.user_id = user_id

    def list(self) -> list[dict]:
        """Return all conversations for this user, most recent first."""
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(conversations)
                .where(conversations.c.user_id == self.user_id)
                .order_by(desc(conversations.c.updated_at))
            ).fetchall()
        return [row._asdict() for row in rows]

    def create(self, thread_id: str, title: str) -> None:
        """Create a new conversation record."""
        now = _now()
        with self.engine.begin() as conn:
            conn.execute(
                insert(conversations).values(
                    thread_id=thread_id,
                    user_id=self.user_id,
                    title=title,
                    created_at=now,
                    updated_at=now,
                )
            )

    def update_title(self, thread_id: str, title: str) -> None:
        """Update the title of a conversation."""
        with self.engine.begin() as conn:
            conn.execute(
                update(conversations)
                .where(conversations.c.thread_id == thread_id)
                .values(title=title, updated_at=_now())
            )

    def touch(self, thread_id: str) -> None:
        """Update updated_at — call after each agent response."""
        with self.engine.begin() as conn:
            conn.execute(
                update(conversations)
                .where(conversations.c.thread_id == thread_id)
                .values(updated_at=_now())
            )


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
