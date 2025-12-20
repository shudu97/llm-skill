"""
System prompts for the ReAct agent
"""

from pathlib import Path


def get_system_prompt(skill_summaries: str) -> str:
    """Generate the system prompt for the agent.

    Args:
        skill_summaries: Formatted string containing all available skill summaries

    Returns:
        The complete system prompt
    """
    # Read the system prompt template from Agent.md
    agent_md_path = Path(__file__).parent.parent.parent / "Agent.md"
    prompt_template = agent_md_path.read_text()

    # Insert skill_summaries into the template
    prompt = prompt_template.format(skill_summaries=skill_summaries)

    return prompt
