"""
CLI entry point for the ReAct agent
"""

import asyncio
import os
import threading
import time

import uvicorn
from dotenv import load_dotenv
from phoenix.otel import register

from src.cli.runner import run_cli
from src.utils.logger import logger

# Load environment variables
load_dotenv()

_LITELLM_HOST = "127.0.0.1"
_LITELLM_PORT = 4000


def _start_litellm_proxy() -> None:
    from litellm.proxy.proxy_server import app, ProxyConfig

    proxy_config = ProxyConfig()
    asyncio.run(proxy_config.load_config(config="litellm_config.yaml"))
    uvicorn.run(app, host=_LITELLM_HOST, port=_LITELLM_PORT, log_level="error")


# Start the proxy and point the Claude CLI at it
threading.Thread(target=_start_litellm_proxy, daemon=True).start()
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
