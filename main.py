"""
CLI entry point for the ReAct agent
"""

import asyncio
import atexit
import os
import subprocess
import sys
import time

from dotenv import load_dotenv
from phoenix.otel import register

from src.cli.runner import run_cli
from src.utils.logger import logger

# Load environment variables
load_dotenv()

_LITELLM_HOST = "127.0.0.1"
_LITELLM_PORT = 4000


def _start_litellm_proxy() -> None:
    litellm_bin = os.path.join(os.path.dirname(sys.executable), "litellm")
    proc = subprocess.Popen(
        [litellm_bin, "--config", "litellm_config.yaml", "--port", str(_LITELLM_PORT)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    atexit.register(proc.terminate)


# Start the proxy and point the Claude CLI at it
_start_litellm_proxy()
os.environ["ANTHROPIC_BASE_URL"] = f"http://{_LITELLM_HOST}:{_LITELLM_PORT}"
time.sleep(2)  # give the proxy a moment to bind

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
    asyncio.run(run_cli())
