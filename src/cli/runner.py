"""
CLI runner for the ReAct agent.
Handles the command-line interface and user interaction loop.
"""

import asyncio
import os
import re
import uuid

import questionary

from claude_agent_sdk import get_session_messages

from src.agent.graph import ReActAgent
from src.cli.callbacks import CLICallback
from src.store.conversation_store import ConversationStore
from src.store.database import create_db_engine
from src.utils.logger import logger

_NEW_CONVERSATION = "__new__"
_ABS_PATH_RE = re.compile(r"(/(?:[^\s/]+/)*[^\s/]+)")


def _linkify_paths(text: str) -> str:
    """Wrap absolute paths in OSC 8 terminal hyperlinks so they're clickable."""
    def _replace(m: re.Match) -> str:
        path = m.group(1)
        url = f"file://{path}"
        return f"\033]8;;{url}\033\\{path}\033]8;;\033\\"

    return _ABS_PATH_RE.sub(_replace, text)


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


def _print_conversation_history(session_id: str) -> None:
    """Print previous conversation messages to the terminal."""
    messages = get_session_messages(session_id)
    if not messages:
        return

    print("\n\033[90m─── Conversation History ───\033[0m")
    for msg in messages:
        role = msg.type  # "user" or "assistant"
        content = msg.message.get("content", "")

        # Extract text from content (may be str or list of blocks)
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block["text"])
            text = "\n".join(parts)
        else:
            continue

        text = text.strip()
        if not text:
            continue

        if role == "user":
            print(f"\n>>> {text}")
        else:
            print(f"\n{_linkify_paths(text)}\n")

    print("\033[90m─── End of History ───\033[0m")


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

    if not is_new:
        _print_conversation_history(session_id)

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
            print(f"\n{_linkify_paths(response)}\n")

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
