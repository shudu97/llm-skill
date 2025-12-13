"""
Simple example of using the ReAct agent
"""
from react_agent import ReActAgent


def main():
    # Initialize the agent (make sure Ollama is running with llama3.2 model)
    agent = ReActAgent(model_name="llama3.2")

    # Simple example query
    query = "What is 15 times 7?"

    print(f"Question: {query}\n")

    # Get the response
    response = agent.run(query)

    print(f"Answer: {response}")


if __name__ == "__main__":
    main()
