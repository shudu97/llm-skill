"""
Simple example of using the ReAct agent
"""

import os

from dotenv import load_dotenv
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor

from src.agent.graph import ReActAgent

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

print(f"Phoenix configured at: {phoenix_endpoint}")
print("LangChain instrumentation enabled")
print("Traces will be sent to Phoenix - check http://0.0.0.0:6006/projects")


def main():
    # Initialize the agent (make sure Ollama is running with llama3.2 model)
    agent = ReActAgent(model_name="qwen3:8b")

    # Simple example query
    query = "Generate a report for CCAR internal market shock"

    print(f"Question: {query}\n")

    # Get the response
    response = agent.run(query)

    print(f"Answer: {response}")


if __name__ == "__main__":
    main()
