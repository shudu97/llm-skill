import subprocess

from langchain_core.tools import tool


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
            timeout=30,  # 30 second timeout to prevent hanging
        )

        # Combine stdout and stderr
        output = result.stdout

        if result.stderr:
            output += f"\nErrors/Warnings:\n{result.stderr}"

        if result.returncode != 0:
            return f"Command exited with code {result.returncode}\n{output}"

        return output if output else "Command executed successfully (no output)"

    except subprocess.TimeoutExpired:
        return "Error: Command execution timed out (30 second limit)"
    except Exception as e:
        return f"Error executing command: {str(e)}"
