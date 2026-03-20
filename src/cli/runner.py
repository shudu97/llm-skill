"""
CLI runner for the ReAct agent.
Handles the command-line interface and user interaction loop.
"""

import asyncio
import os
import uuid

import questionary

from src.agent.graph import ReActAgent
from src.cli.callbacks import CLICallback
from src.store.conversation_store import ConversationStore
from src.store.database import create_db_engine
from src.utils.logger import logger

_NEW_CONVERSATION = "__new__"


def select_session(store: ConversationStore) -> tuple[str, bool]:
    """Show conversation selector and return (session_id, is_new)."""
    past = store.list()

    if not past:
        return str(uuid.uuid4()), True

    choices = [
        questionary.Choice("New conversation", value=_NEW_CONVERSATION),
        questionary.Separator(),
        *[
            questionary.Choice(
                f"[{r['updated_at'][:16]}]  {r['title']}",
                value=r["thread_id"],
            )
            for r in past
        ],
    ]

    session_id = questionary.select(
        "Select a conversation:",
        choices=choices,
    ).ask()

    if session_id is None:
        raise KeyboardInterrupt

    if session_id == _NEW_CONVERSATION:
        return str(uuid.uuid4()), True

    return session_id, False


def run_cli(session_id: str, is_new: bool) -> None:
    """Run the CLI interface for the ReAct agent."""
    db_path = os.getenv("AGENT_DB_PATH", "data/agent.db")
    db_url = f"sqlite:///{db_path}"
    user_id = os.getenv("AGENT_USER_ID", "cli_user")

    engine = create_db_engine(db_url)
    store = ConversationStore(engine=engine, user_id=user_id)

    callback = CLICallback()
    agent = ReActAgent(
        callback=callback,
        user_id=user_id,
        session_id=session_id if not is_new else None,
    )

    logger.info(f"Session: {session_id}")
    logger.info("Type 'exit' or 'quit' to end the conversation.\n")

    title_updated = not is_new
    db_created = not is_new  # for existing sessions the record already exists

    while True:
        try:
            query_text = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            logger.info("Ending conversation. Goodbye!")
            break

        if query_text.lower() in ["exit", "quit", "q"]:
            logger.info("Ending conversation. Goodbye!")
            break

        if not query_text:
            continue

        try:
            response, new_session_id = asyncio.run(agent.run(query_text))
            print(f"\n{response}\n")

            if new_session_id and new_session_id != session_id:
                session_id = new_session_id

            if not db_created:
                store.create(session_id, title="New conversation")
                db_created = True

            if not title_updated:
                title = query_text[:60] + ("..." if len(query_text) > 60 else "")
                store.update_title(session_id, title)
                title_updated = True

            store.touch(session_id)

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            logger.info("Please try again or type 'exit' to quit.")
