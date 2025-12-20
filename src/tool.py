"""
Tools for the ReAct agent
"""

from langchain_core.tools import tool

from src.skills import SkillManager

# Global skill manager instance
_skill_manager = SkillManager()


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
