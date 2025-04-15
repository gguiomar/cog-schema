import re


class ResponseExtractor:
    """
    Extracts answers from model completions.
    """
    @staticmethod
    def extract_xml_answer(text: str) -> str:
        """
        Extract answer from XML-formatted text.
        
        Args:
            text (str): Full model completion text.
        
        Returns:
            str: Extracted answer from <answer> tags.
        """
        match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
        return match.group(1).strip() if match else ""