"""
Skill management middleware for the agent.

This middleware provides skill discovery and loading functionality,
exposing the view_skill tool and skill summaries for system prompt injection.
"""

import re
from pathlib import Path
from typing import Dict, List

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.tools import tool


class SkillManager:
    """Manages skill loading and discovery"""

    def __init__(self, skills_dir: str = "src/skills/skills"):
        """Initialize the skill manager.

        Args:
            skills_dir: Directory containing skill folders
        """
        self.skills_dir = Path(skills_dir)
        self.skills: Dict[str, Path] = {}
        self._discover_skills()

    def _discover_skills(self) -> None:
        """Discover all SKILL.md files in the skills directory."""
        if not self.skills_dir.exists():
            return

        # Find all SKILL.md files in subdirectories
        for skill_file in self.skills_dir.glob("*/SKILL.md"):
            skill_name = skill_file.parent.name
            self.skills[skill_name] = skill_file

    def get_skill_summaries(self) -> str:
        """Get summaries (name and description) of all available skills.

        Returns:
            Formatted string with all skill summaries
        """
        if not self.skills:
            return "No skills available."

        summaries = []

        for skill_name, skill_file in sorted(self.skills.items()):
            try:
                summary = self._extract_summary(skill_file)
                summaries.append(f"## {summary['name']}")
                summaries.append(f"**ID**: `{skill_name}`")
                summaries.append(f"**Description**: {summary['description']}\n")
            except Exception as e:
                summaries.append(f"## {skill_name}")
                summaries.append(f"**Error loading summary**: {str(e)}\n")

        return "\n".join(summaries)

    def _extract_summary(self, skill_file: Path) -> Dict[str, str]:
        """Extract name and description from a SKILL.md file.

        Args:
            skill_file: Path to the SKILL.md file

        Returns:
            Dictionary with 'name' and 'description' keys
        """
        with open(skill_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Parse YAML frontmatter
        frontmatter_match = re.match(r"^---\s*\n(.*?\n)---\s*\n", content, re.DOTALL)

        if not frontmatter_match:
            raise ValueError("No frontmatter found in SKILL.md")

        frontmatter = frontmatter_match.group(1)

        # Extract name and description
        name_match = re.search(r"^name:\s*(.+)$", frontmatter, re.MULTILINE)
        desc_match = re.search(r"^description:\s*(.+)$", frontmatter, re.MULTILINE)

        return {
            "name": name_match.group(1).strip() if name_match else "Unknown",
            "description": desc_match.group(1).strip()
            if desc_match
            else "No description",
        }

    def load_skill(self, skill_id: str, file: str = "SKILL.md") -> str:
        """Load content from a skill folder.

        Args:
            skill_id: The skill ID (folder name)
            file: File to read within the skill folder (default: SKILL.md)

        Returns:
            Content of the requested file. For SKILL.md, YAML frontmatter is removed.
        """
        if skill_id not in self.skills:
            available = ", ".join(sorted(self.skills.keys()))
            return f"Error: Skill '{skill_id}' not found. Available skills: {available}"

        try:
            skill_folder = self.skills[skill_id].parent
            file_path = skill_folder / file

            if not file_path.exists():
                return f"Error: File '{file}' not found in skill '{skill_id}'"

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Strip YAML frontmatter only for SKILL.md
            if file == "SKILL.md":
                frontmatter_match = re.match(r"^---\s*\n(.*?\n)---\s*\n", content, re.DOTALL)
                if frontmatter_match:
                    content = content[frontmatter_match.end() :]

            return content
        except Exception as e:
            return f"Error loading '{file}' from skill '{skill_id}': {str(e)}"

    def list_skills(self) -> List[str]:
        """Get a list of all available skill IDs.

        Returns:
            List of skill IDs
        """
        return sorted(self.skills.keys())


class SkillMiddleware(AgentMiddleware):
    """Provides skill management tools and summaries.

    This middleware adds the view_skill tool and exposes skill summaries
    that can be injected into the agent's system prompt.

    Example:
        ```python
        from langchain.agents import create_agent
        from src.agent.middleware.skill_middleware import SkillMiddleware

        skill_middleware = SkillMiddleware(skills_dir="src/skills/skills")

        agent = create_agent(
            model=model,
            tools=[],
            system_prompt=f"You are an agent. Skills: {skill_middleware.skill_summaries}",
            middleware=[skill_middleware],
        )
        ```
    """

    def __init__(self, skills_dir: str = "src/skills/skills") -> None:
        """Initialize the skill middleware.

        Args:
            skills_dir: Directory containing skill folders (each with SKILL.md)
        """
        super().__init__()
        self.skill_manager = SkillManager(skills_dir)

        # Create view_skill tool as a closure that captures self
        @tool
        def view_skill(skill_id: str, file: str = "SKILL.md") -> str:
            """Load content from a skill folder.

            Args:
                skill_id: The skill ID (e.g., "ccar_ims")
                file: File to read (default: SKILL.md). Use for reference files like "reference/example.md"

            Returns:
                The file content
            """
            return self.skill_manager.load_skill(skill_id, file)

        self.tools = [view_skill]

    @property
    def skill_summaries(self) -> str:
        """Get formatted skill summaries for system prompt injection.

        Returns:
            Formatted string with all skill summaries (name, ID, description)
        """
        return self.skill_manager.get_skill_summaries()
