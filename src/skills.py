"""
Skills management for the ReAct agent
Handles skill discovery, loading, and progressive disclosure
"""

import re
from pathlib import Path
from typing import Dict, List


class SkillManager:
    """Manages skill loading and discovery"""

    def __init__(self, skills_dir: str = "src/skills"):
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

        summaries = ["# Available Skills\n"]

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

        print(name_match)

        return {
            "name": name_match.group(1).strip() if name_match else "Unknown",
            "description": desc_match.group(1).strip()
            if desc_match
            else "No description",
        }

    def load_skill(self, skill_id: str) -> str:
        """Load the full content of a specific skill.

        Args:
            skill_id: The skill ID (folder name)

        Returns:
            Full content of the SKILL.md file
        """
        if skill_id not in self.skills:
            available = ", ".join(sorted(self.skills.keys()))
            return f"Error: Skill '{skill_id}' not found. Available skills: {available}"

        try:
            with open(self.skills[skill_id], "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"Error loading skill '{skill_id}': {str(e)}"

    def list_skills(self) -> List[str]:
        """Get a list of all available skill IDs.

        Returns:
            List of skill IDs
        """
        return sorted(self.skills.keys())
