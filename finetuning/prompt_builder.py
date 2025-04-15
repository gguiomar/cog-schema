from typing import Dict, Any


class TemporalPromptBuilder:
    """
    Builds prompts for temporal reasoning tasks.
    """
    SYSTEM_PROMPT = """
    You are a temporal reasoning assistant. When answering, follow the exact format below.

    Example:
    <reasoning>
    I thought about the sequence of rounds and noted that quadrant [ANSWER] was the biased one.
    </reasoning>
    <answer>
    [ANSWER]
    </answer>

    Now, given the game log, provide your chain-of-thought within <reasoning> tags and your final answer (the winning quadrant) within <answer> tags.
    """

    @classmethod
    def build_prompt(cls, log: Dict[str, Any]) -> str:
        """
        Build a prompt from a game log.
        
        Args:
            log (Dict[str, Any]): Parsed game log.
        
        Returns:
            str: Formatted prompt for temporal reasoning.
        """
        history_lines = []
        for idx, choice in enumerate(log["choices"]):
            choice_val = choice.get("choice", "?")
            color_val = choice.get("color", "?")
            quad = choice.get("quadrant")
            quadrant_str = f"{quad+1}" if quad is not None else "unknown"
            line = f"Round {idx+1}: Chose {choice_val} (Color: {color_val}) from quadrant {quadrant_str}"
            history_lines.append(line)
        
        history = "\n".join(history_lines)
        prompt = f"{cls.SYSTEM_PROMPT}\nGame Log:\n{history}\nBased on the above rounds, which quadrant was the biased one?"
        return prompt