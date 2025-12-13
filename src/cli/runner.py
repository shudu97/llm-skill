"""
CLI runner for the ReAct agent.
Handles the command-line interface and user interaction loop.
"""

import os
import uuid

import questionary
from prompt_toolkit import prompt

from src.agent.graph import ReActAgent
from src.cli.callbacks import CLICallback
from src.store.conversation_store import ConversationStore
from src.store.database import create_db_engine
from src.utils.logger import logger

_NEW_CONVERSATION = "__new__"


def _select_thread(store: ConversationStore) -> tuple[str, bool]:
    """Show conversation selector and return (thread_id, is_new).

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

    thread_id = questionary.select(
        "Select a conversation:",
        choices=choices,
    ).ask()

    if thread_id is None:
        # User hit Ctrl-C at the selector
        raise KeyboardInterrupt

    if thread_id == _NEW_CONVERSATION:
        return str(uuid.uuid4()), True

    return thread_id, False


def run_cli(model_name: str = "qwen3:8b") -> None:
    """Run the CLI interface for the ReAct agent."""
    db_path = os.getenv("AGENT_DB_PATH", "data/agent.db")
    db_url = f"sqlite:///{db_path}"
    user_id = os.getenv("AGENT_USER_ID", "cli_user")

    engine = create_db_engine(db_url)
    store = ConversationStore(engine=engine, user_id=user_id)

    thread_id, is_new = _select_thread(store)

    if is_new:
        store.create(thread_id, title="New conversation")

    callback = CLICallback()
    agent = ReActAgent(
        callback=callback,
        user_id=user_id,
        thread_id=thread_id,
        model_name=model_name,
        db_path=db_path,
    )

    logger.info(f"Thread: {thread_id}")
    logger.info("Type 'exit' or 'quit' to end the conversation.\n")

    title_updated = not is_new  # resuming — title already set

    while True:
        query = prompt("\n>>> ", placeholder="Send a message").strip()

        if query.lower() in ["exit", "quit", "q"]:
            logger.info("Ending conversation. Goodbye!")
            break

        if not query:
            continue

        try:
            response = agent.run(query)
            print(f"\n{response}\n")

            # Set title from the first user message of a new conversation
            if not title_updated:
                title = query[:60] + ("..." if len(query) > 60 else "")
                store.update_title(thread_id, title)
                title_updated = True

            store.touch(thread_id)

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            logger.info("Please try again or type 'exit' to quit.")
