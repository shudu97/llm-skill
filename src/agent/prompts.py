"""
System prompts for the ReAct agent
"""


def get_system_prompt(skill_summaries: str) -> str:
    """Generate the system prompt for the agent.

    Args:
        skill_summaries: Formatted string containing all available skill summaries

    Returns:
        The complete system prompt
    """

    prompt = f"""
        You are a helpful assistant with access to specialized skills.

        {skill_summaries}

        When a user asks for something that matches a skill, use the view_skill tool to load the detailed instructions for that skill, then follow those instructions to complete the task.
    """

    return prompt
