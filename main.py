"""
CLI entry point for the ReAct agent
"""

import asyncio
import os

from dotenv import load_dotenv
from phoenix.otel import register

from src.cli.runner import run_cli
from src.utils.logger import logger

# Load environment variables
load_dotenv()

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
