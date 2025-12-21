"""
Simple example of using the ReAct agent
"""

import os

from dotenv import load_dotenv
from openinference.instrumentation.langchain import LangChainInstrumentor
from phoenix.otel import register

from src.agent.graph import ReActAgent
from src.utils.logger import logger

# Load environment variables
load_dotenv()

# Get Phoenix endpoint from environment
phoenix_endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://0.0.0.0:6006")

# Register Phoenix tracing
trace_provider = register(
    project_name="default",
    endpoint=f"{phoenix_endpoint}/v1/traces",
)

# Explicitly instrument LangChain
LangChainInstrumentor().instrument(tracer_provider=trace_provider)

logger.info(f"Phoenix configured at: {phoenix_endpoint}")
logger.info("LangChain instrumentation enabled")
logger.info("Traces will be sent to Phoenix - check http://0.0.0.0:6006/projects")


def main():
    # Initialize the agent (make sure Ollama is running with llama3.2 model)
    agent = ReActAgent(model_name="qwen3:8b")

    # Get query from user input
    query = input("Enter your query: ")

    # Get the response
    response = agent.run(query)

    return response


if __name__ == "__main__":
    response = main()
    logger.info(f"Response: {response}")
