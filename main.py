"""
Simple example of using the ReAct agent
"""

from src.agent.graph import ReActAgent


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
