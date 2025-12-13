"""
Tools for the ReAct agent
"""

import subprocess
from pathlib import Path
from langchain_core.tools import tool

from src.skills import SkillManager

# Global skill manager instance
_skill_manager = SkillManager()


@tool
def execute_script(file_path: str, args: str = "", interpreter: str = "") -> str:
    """Execute a script file and return the output.

    This tool runs script files from the filesystem, similar to running scripts in bash.
    It automatically detects the script type based on file extension, or you can specify an interpreter.
    Use this to execute programs, perform calculations, data processing, or any scripted operations.

    Args:
        file_path: Path to the script file to execute (e.g., "scripts/process.py", "scripts/analyze.sh")
        args: Optional command-line arguments to pass to the script (e.g., "--input data.csv")
        interpreter: Optional interpreter to use (e.g., "python", "bash", "node"). If not provided,
                    it will be auto-detected from file extension.

    Returns:
        The output from the executed script or any error messages
    """
    try:
        # Determine the interpreter if not provided
        if not interpreter:
            file_ext = Path(file_path).suffix.lower()
            interpreter_map = {
                '.py': 'python',
                '.sh': 'bash',
                '.js': 'node',
                '.rb': 'ruby',
                '.pl': 'perl',
            }
            interpreter = interpreter_map.get(file_ext, 'python')

        # Build the command
        cmd = [interpreter, file_path]

        # Add arguments if provided
        if args:
            cmd.extend(args.split())

        # Execute the script
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30  # 30 second timeout to prevent hanging
        )

        # Combine stdout and stderr
        output = result.stdout
        if result.stderr:
            output += f"\nErrors/Warnings:\n{result.stderr}"

        if result.returncode != 0:
            return f"Script exited with code {result.returncode}\n{output}"

        return output if output else "Script executed successfully (no output)"

    except subprocess.TimeoutExpired:
        return "Error: Script execution timed out (30 second limit)"
    except FileNotFoundError as e:
        return f"Error: File or interpreter not found: {str(e)}"
    except Exception as e:
        return f"Error executing script: {str(e)}"


@tool
def view_skill(skill_id: str) -> str:
    """Load and view the full content of a specific skill.

    When you need detailed instructions for a skill, use this tool to load the complete
    skill definition. The skill summaries are already available to you, but this tool
    provides the full instructions including which scripts to execute, parameters to use,
    and step-by-step procedures.

    Args:
        skill_id: The ID of the skill to load (e.g., "ccar_ims"). Use the skill ID shown
                 in the available skills list.

    Returns:
        The full content of the skill's SKILL.md file with detailed instructions
    """
    return _skill_manager.load_skill(skill_id)
