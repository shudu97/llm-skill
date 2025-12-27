"""
Bash execution middleware for the agent.

This middleware provides bash command execution functionality,
exposing the bash tool for running shell commands.
"""

import subprocess

from langchain_core.tools import tool
from langchain.agents.middleware.types import AgentMiddleware


class BashMiddleware(AgentMiddleware):
    """Provides bash command execution capability.

    This middleware adds the bash tool for executing shell commands.

    Example:
        ```python
        from langchain.agents import create_agent
        from src.agent.middleware.bash import BashMiddleware

        bash_middleware = BashMiddleware()

        agent = create_agent(
            model=model,
            tools=[],
            middleware=[bash_middleware],
        )
        ```
    """

    def __init__(self, timeout: int = 30) -> None:
        """Initialize the bash middleware.

        Args:
            timeout: Command execution timeout in seconds (default: 30)
        """
        super().__init__()
        self.timeout = timeout

        # Create bash tool as a closure that captures self
        @tool
        def bash(command: str) -> str:
            """Execute a bash command and return the output.

            This tool runs any bash command in a shell environment, similar to running commands in a terminal.
            Use this to execute system commands, file operations, process management, or any bash operations.

            Args:
                command: The bash command to execute (e.g., "ls -la", "cat file.txt", "python script.py")

            Returns:
                The output from the executed command or any error messages
            """
            try:
                # Execute the command with shell=True to enable bash functionality
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )

                # For failed commands: return exit code and error messages
                if result.returncode != 0:
                    error_output = result.stderr if result.stderr else result.stdout
                    return f"Command failed with exit code {result.returncode}\nError: {error_output}"

                # For successful commands: only return stdout (actual output), ignore stderr (logs/warnings)
                return result.stdout if result.stdout else "Command executed successfully (no output)"

            except subprocess.TimeoutExpired:
                return f"Error: Command execution timed out ({self.timeout} second limit)"
            except Exception as e:
                return f"Error executing command: {str(e)}"

        self.tools = [bash]
