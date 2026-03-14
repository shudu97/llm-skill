"""
CLI runner for the ReAct agent.
Handles the command-line interface and user interaction loop.
"""

import asyncio
import os
import uuid

import questionary
from prompt_toolkit import PromptSession

from src.agent.graph import ReActAgent
from src.cli.callbacks import CLICallback
from src.store.conversation_store import ConversationStore
from src.store.database import create_db_engine
from src.utils.logger import logger

_NEW_CONVERSATION = "__new__"


async def _select_session(store: ConversationStore) -> tuple[str, bool]:
    """Show conversation selector and return (session_id, is_new).

    Returns is_new=True when the user chose to start a fresh conversation.
    """
    past = store.list()

    if not past:
        # No history yet — start fresh silently
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

    session_id = await asyncio.to_thread(
        questionary.select("Select a conversation:", choices=choices).ask
    )

    if session_id is None:
        # User hit Ctrl-C at the selector
        raise KeyboardInterrupt

    if session_id == _NEW_CONVERSATION:
        return str(uuid.uuid4()), True

    return session_id, False


async def run_cli() -> None:
    """Run the CLI interface for the ReAct agent."""
    db_path = os.getenv("AGENT_DB_PATH", "data/agent.db")
    db_url = f"sqlite:///{db_path}"
    user_id = os.getenv("AGENT_USER_ID", "cli_user")

    engine = create_db_engine(db_url)
    store = ConversationStore(engine=engine, user_id=user_id)

    session_id, is_new = await _select_session(store)

    if is_new:
        store.create(session_id, title="New conversation")

    callback = CLICallback()
    agent = ReActAgent(
        callback=callback,
        user_id=user_id,
        session_id=session_id if not is_new else None,
    )

    logger.info(f"Session: {session_id}")
    logger.info("Type 'exit' or 'quit' to end the conversation.\n")

    title_updated = not is_new  # resuming — title already set
    prompt_session = PromptSession()

    while True:
        query_text = (await asyncio.to_thread(
            prompt_session.prompt, "\n>>> ", placeholder="Send a message"
        )).strip()

        if query_text.lower() in ["exit", "quit", "q"]:
            logger.info("Ending conversation. Goodbye!")
            break

        if not query_text:
            continue

        try:
            response, new_session_id = await agent.run(query_text)
            print(f"\n{response}\n")

            # Persist session_id returned by the SDK on first turn
            if new_session_id and new_session_id != session_id:
                session_id = new_session_id

            # Set title from the first user message of a new conversation
            if not title_updated:
                title = query_text[:60] + ("..." if len(query_text) > 60 else "")
                store.update_title(session_id, title)
                title_updated = True

            store.touch(session_id)

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            logger.info("Please try again or type 'exit' to quit.")
