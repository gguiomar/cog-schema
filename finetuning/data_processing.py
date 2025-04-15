import os
import json
from typing import List, Dict, Any


class GameLogProcessor:
    """
    Handles processing of game log files for temporal reasoning tasks.
    """
    @staticmethod
    def parse_game_log(file_path: str) -> Dict[str, Any]:
        """
        Parse a single game log JSON file.
        
        Args:
            file_path (str): Path to the game log JSON file.
        
        Returns:
            Dict containing parsed game log information.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        normalized_choices = []
        for choice in data.get("choices", []):
            choice_key = choice.get("cue_name") or choice.get("choice")
            normalized_choices.append({
                "round": choice.get("round"),
                "quadrant": choice.get("quadrant"),
                "choice": choice_key,
                "color": choice.get("color"),
                "timestamp": choice.get("timestamp") or choice.get("client_timestamp")
            })
        
        final_choice = data.get("final_choice") or {}
        success = data.get("success") if data.get("success") is not None else final_choice.get("correct")
        
        return {
            "game_id": data.get("game_id"),
            "start_time": data.get("start_time"),
            "choices": normalized_choices,
            "final_choice": final_choice,
            "completion_time": data.get("completion_time"),
            "total_duration": data.get("total_duration"),
            "success": success
        }
    
    @classmethod
    def load_game_logs(cls, folder_path: str) -> List[Dict[str, Any]]:
        """
        Load game logs from a specified folder.
        
        Args:
            folder_path (str): Path to the folder containing game log files.
        
        Returns:
            List of parsed game logs.
        """
        logs = []
        for filename in os.listdir(folder_path):
            if filename.endswith('.json'):
                file_path = os.path.join(folder_path, filename)
                logs.append(cls.parse_game_log(file_path))
        return logs


class TemporalDatasetPreparator:
    """
    Prepares dataset for temporal reasoning fine-tuning.
    """
    @staticmethod
    def prepare_dataset(
        game_logs: List[Dict[str, Any]], 
        prompt_builder_cls: Any
    ) -> List[Dict[str, str]]:
        """
        Prepare training examples from game logs.
        
        Args:
            game_logs (List[Dict[str, Any]]): List of parsed game logs.
            prompt_builder_cls (Any): Prompt builder class to generate prompts.
        
        Returns:
            List of training examples with prompts and answers.
        """
        examples = []
        for log in game_logs:
            prompt = prompt_builder_cls.build_prompt(log)
            answer = str(log["final_choice"].get("chosen_quadrant", "?"))
            examples.append({"prompt": prompt, "answer": answer})
        return examples