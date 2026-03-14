"""
CLI entry point for the ReAct agent
"""

import atexit
import os
import subprocess
import sys
import time

from dotenv import load_dotenv
from phoenix.otel import register

from src.cli.runner import run_cli, select_session
from src.store.conversation_store import ConversationStore
from src.store.database import create_db_engine
from src.utils.logger import logger

# Load environment variables
load_dotenv()

_LITELLM_HOST = "127.0.0.1"
_LITELLM_PORT = 4000


def _start_litellm_proxy() -> None:
    litellm_bin = os.path.join(os.path.dirname(sys.executable), "litellm")
    log = open("data/litellm.log", "w")
    proc = subprocess.Popen(
        [litellm_bin, "--config", "litellm_config.yaml", "--port", str(_LITELLM_PORT)],
        stdout=log,
        stderr=log,
    )
    atexit.register(proc.terminate)
    time.sleep(2)
    if proc.poll() is not None:
        raise RuntimeError(f"LiteLLM proxy exited immediately with code {proc.returncode}")
    logger.info(f"LiteLLM proxy running on port {_LITELLM_PORT}")


# Start the proxy and point the Claude CLI at it
_start_litellm_proxy()
os.environ["ANTHROPIC_BASE_URL"] = f"http://{_LITELLM_HOST}:{_LITELLM_PORT}"

# Get Phoenix endpoint from environment
phoenix_endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://0.0.0.0:6006")

# Register Phoenix tracing
register(
    project_name="default",
    endpoint=f"{phoenix_endpoint}/v1/traces",
)

logger.info(f"Phoenix configured at: {phoenix_endpoint}")
logger.info("Traces will be sent to Phoenix - check http://0.0.0.0:6006/projects")


if __name__ == "__main__":
    db_path = os.getenv("AGENT_DB_PATH", "data/agent.db")
    user_id = os.getenv("AGENT_USER_ID", "cli_user")
    engine = create_db_engine(f"sqlite:///{db_path}")
    store = ConversationStore(engine=engine, user_id=user_id)

    session_id, is_new = select_session(store)
    run_cli(session_id, is_new)
