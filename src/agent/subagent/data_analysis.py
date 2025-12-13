"""
Data Analysis subagent.

This subagent writes and executes Python scripts for data analysis tasks.
"""

import json
import os
import sqlite3
import subprocess
import tempfile

import pandas as pd
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite import SqliteSaver

from src.agent.middleware.bash import BashMiddleware
from src.agent.state import DataAnalysisState

DATA_ANALYSIS_SYSTEM_PROMPT = """You are a data analysis assistant that writes and runs Python scripts.

Your job is to:
1. Understand the data analysis task
2. Write a Python script using the provided dataframe info
3. Use run_script tool to execute your script
4. If there's an error, use bash with `cat <script_path>` to read the script, fix it, and try again

ENVIRONMENT:
- pandas and numpy are already installed
- The dataframe structure is provided below, use this info to write your script
- Load the data from the file path provided

Guidelines:
- Write simple, clean Python code
- Always include print statements to show results
- Use the exact column names from the provided schema

DEBUGGING:
- If run_script returns an error, it will include the script path
- Use bash tool: cat /path/to/script.py to see your code
- Fix the issue and call run_script with corrected code
"""


class DataAnalysisSubagent:
    """Subagent for data analysis tasks.

    Creates a tool that delegates data analysis tasks to a specialized
    subagent which writes and executes Python scripts.
    """

    def __init__(
        self,
        bash_middleware: BashMiddleware,
        model_name: str = "llama3.2",
        timeout: int = 60,
        privacy_mode: bool = False,
        db_path: str | None = None,
    ) -> None:
        """Initialize the data analysis subagent.

        Args:
            bash_middleware: Shared bash middleware for file reading
            model_name: LLM model for the subagent
            timeout: Script execution timeout in seconds
            privacy_mode: If True, hide actual data values from subagent
            db_path: Path to the SQLite database file (defaults to AGENT_DB_PATH env var or agent.db)
        """
        self.bash_middleware = bash_middleware
        self.model_name = model_name
        self.timeout = timeout
        self.privacy_mode = privacy_mode
        self.db_path = db_path or os.getenv("AGENT_DB_PATH", "data/agent.db")
        self._temp_files: list[str] = []

    def get_tool(self):
        """Return the data_analysis tool for use in create_agent."""
        subagent = self

        @tool
        def data_analysis(file_path: str, task: str) -> str:
            """Delegate a data analysis task to a specialized subagent.

            Use this tool when you need to:
            - Analyze CSV/JSON/Excel data files
            - Generate statistics or summaries
            - Transform or clean data

            Args:
                file_path: Path to the data file (CSV, JSON, or Excel)
                task: Description of the data analysis task.

            Returns:
                The analysis results
            """
            return subagent._run_subagent(file_path, task)

        return data_analysis

    def _create_run_script_tool(self):
        """Create a tool that writes and runs a Python script."""
        timeout = self.timeout
        temp_files = self._temp_files

        @tool
        def run_script(script: str) -> str:
            """Write a Python script to a temp file and execute it.

            Args:
                script: The complete Python script code to run

            Returns:
                The script output on success.
                On error: the script path and error message.
            """
            fd, script_path = tempfile.mkstemp(suffix=".py", prefix="analysis_")
            temp_files.append(script_path)

            with os.fdopen(fd, "w") as f:
                f.write(script)

            try:
                result = subprocess.run(
                    ["python", script_path],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )

                if result.returncode != 0:
                    return f"ERROR in {script_path}\n{result.stderr}"

                os.unlink(script_path)
                temp_files.remove(script_path)
                return result.stdout if result.stdout else "Success (no output)"

            except subprocess.TimeoutExpired:
                return f"TIMEOUT in {script_path} - exceeded {timeout}s"
            except Exception as e:
                return f"ERROR in {script_path}\n{str(e)}"

        return run_script

    def _cleanup_temp_files(self):
        """Clean up any remaining temp files."""
        for path in self._temp_files:
            if os.path.exists(path):
                os.unlink(path)
        self._temp_files.clear()

    def _analyze_dataframe(self, file_path: str) -> str:
        """Pre-analyze dataframe structure without loading full data into context."""
        try:
            # Detect file type and load
            if file_path.endswith(".csv"):
                df = pd.read_csv(file_path)
            elif file_path.endswith(".json"):
                df = pd.read_json(file_path)
            elif file_path.endswith((".xls", ".xlsx")):
                df = pd.read_excel(file_path)
            else:
                return json.dumps({"error": f"Unsupported file type: {file_path}"})

            # Convert string columns that look like dates to datetime
            for col in df.columns:
                if df[col].dtype == "object":
                    try:
                        df[col] = pd.to_datetime(df[col], errors="coerce")
                    except (ValueError, TypeError):
                        pass  # Not a date column, keep as string

            # Build column info
            columns = {col: str(dtype) for col, dtype in df.dtypes.items()}

            # Build structure info as JSON
            info = {
                "file_path": file_path,
                "shape": {"rows": df.shape[0], "columns": df.shape[1]},
                "columns": columns,
            }

            # Add sample data only if not in privacy mode
            if not self.privacy_mode:
                info["sample"] = df.head(3).to_dict(orient="records")

            return json.dumps(info, indent=2, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _run_subagent(self, file_path: str, task: str) -> str:
        """Run the data analysis subagent."""
        # Pre-analyze dataframe structure
        df_info = self._analyze_dataframe(file_path)

        # Build the task prompt with dataframe info
        task_prompt = f"""DATAFRAME INFO:
            {df_info}

            TASK:
            {task}
        """

        llm = ChatOllama(model=self.model_name, temperature=0)

        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        subagent = create_agent(
            model=llm,
            system_prompt=SystemMessage(content=DATA_ANALYSIS_SYSTEM_PROMPT),
            state_schema=DataAnalysisState,
            tools=[
                self._create_run_script_tool(),
                *self.bash_middleware.tools,
            ],
            checkpointer=SqliteSaver(conn),
        )

        config = {"configurable": {"thread_id": f"data_analysis_{id(self)}"}}

        result = None
        for event in subagent.stream(
            {"messages": [HumanMessage(content=task_prompt)]},
            config=config,
            stream_mode="values",
        ):
            result = event

        self._cleanup_temp_files()

        if result and result.get("messages"):
            return result["messages"][-1].content
        return "Subagent failed to produce a result"
