import re
from typing import List

from .response_extractor import ResponseExtractor


class RewardFunctions:
    """
    Reward functions for GRPO training.
    """
    @staticmethod
    def correctness_reward_func(prompts, completions, answer, **kwargs) -> List[float]:
        """
        Reward based on correctness of the answer.
        """
        responses = [completion for completion in completions]
        extracted_responses = [ResponseExtractor.extract_xml_answer(r) for r in responses]
        return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

    @staticmethod
    def int_reward_func(completions, **kwargs) -> List[float]:
        """
        Reward for providing an integer answer.
        """
        responses = [completion for completion in completions]
        extracted_responses = [ResponseExtractor.extract_xml_answer(r) for r in responses]
        return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

    @staticmethod
    def strict_format_reward_func(completions, **kwargs) -> List[float]:
        """
        Reward for strict XML format.
        """
        pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
        responses = [completion for completion in completions]
        matches = [re.match(pattern, r) for r in responses]
        return [0.5 if match else 0.0 for match in matches]

    @staticmethod
    def soft_format_reward_func(completions, **kwargs) -> List[float]:
        """
        Reward for soft XML format.
        """
        pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
        responses = [completion for completion in completions]
        matches = [re.match(pattern, r) for r in responses]
        return [0.5 if match else 0.0 for match in matches]

    @staticmethod
    def xmlcount_reward_func(completions, **kwargs) -> List[float]:
        """
        Reward based on XML tag count and structure.
        """
        def count_xml(text: str) -> float:
            count = 0.0
            if text.count("<reasoning>\n") == 1:
                count += 0.125
            if text.count("\n</reasoning>\n") == 1:
                count += 0.125
            if text.count("\n<answer>\n") == 1:
                count += 0.125
                count -= len(text.split("\n</answer>\n")[-1]) * 0.001
            if text.count("\n</answer>") == 1:
                count += 0.125
                count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
            return count

        return [count_xml(c) for c in completions]