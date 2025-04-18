import os
import transformers
from transformers.utils import logging
import torch
import time
import anthropic
from typing import Optional, List, Union
from openai import OpenAI

class LLMagent:
    # Define reasoning models as a class variable
    REASONING_MODELS = ["Deepseek_R1_1.5B_Qwen",
                        "Deepseek_R1_7B_Qwen",
                        "Deepseek_R1_8B_Llama",
                        "Deepseek_R1_14B_Qwen",
                        "Deepseek_R1_32B_Qwen"]
    
    def __init__(self, 
                 model_name: str, 
                 device_map: str = "cpu", 
                 max_seq_length: int = 32768, 
                 load_in_4bit: bool = True, 
                 use_unsloth: bool = False,
                 openai_api_key: Optional[str] = None,
                 anthropic_api_key: Optional[str] = None,
                 reasoning_mode: str = "time",  # 'time' or 'tokens'
                 min_thinking_time: float = 5.0,
                 max_thinking_time: float = 10.0,
                 min_thinking_tokens: int = 200,
                 max_thinking_tokens: int = 500):
        """
        Initialize LLM agent with specified model and backend.
        
        Parameters:
        - model_name: str, the name or path of the model.
        - device_map: str, device to use (e.g., 'cpu', 'cuda:0').
        - max_seq_length: int, maximum sequence length for the model.
        - load_in_4bit: bool, whether to load the model in 4-bit precision (unsloth only).
        - use_unsloth: bool, whether to use unsloth or transformers.
        - openai_api_key: Optional[str], if provided, the OpenAI API is used with this API key.
        - anthropic_api_key: Optional[str], if provided, the Anthropic API is used with this API key.
        - reasoning_mode: str, 'time' or 'tokens', controls how reasoning is limited
        - min_thinking_time: float, minimum thinking time in seconds (used when reasoning_mode='time')
        - max_thinking_time: float, maximum thinking time in seconds (used when reasoning_mode='time')
        - min_thinking_tokens: int, minimum number of thinking tokens (used when reasoning_mode='tokens')
        - max_thinking_tokens: int, maximum number of thinking tokens (used when reasoning_mode='tokens')
        
        If neither API key is provided and use_unsloth is False, then a local model (via transformers) is used.
        """
        self.openai_api_key = openai_api_key
        self.anthropic_api_key = anthropic_api_key
        self.openai_flag, self.anthropic_flag = False, False
        self.thinking_time = 0  # Track thinking time for reasoning models
        self.last_thinking_tokens = ""  # Store thinking tokens for reasoning models
        self.reasoning_mode = reasoning_mode
        self.min_thinking_time = min_thinking_time
        self.max_thinking_time = max_thinking_time
        self.min_thinking_tokens = min_thinking_tokens
        self.max_thinking_tokens = max_thinking_tokens
        self.token_count = 0  # Track token count for token-based reasoning

        # Map friendly model names to their corresponding Hugging Face repository strings.
        model_aliases = {
            "Deepseek_R1_1.5B_Qwen": "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-unsloth-bnb-4bit",
            "Deepseek_R1_7B_Qwen" : "unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit",
            "Deepseek_R1_8B_Llama": "unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit", 
            "Deepseek_R1_14B_Qwen": "unsloth/DeepSeek-R1-Distill-Qwen-14B-unsloth-bnb-4bit",
            "Deepseek_R1_32B_Qwen": "unsloth/DeepSeek-R1-Distill-Qwen-32B-bnb-4bit",
            "Qwen_0.5B": "unsloth/Qwen2.5-0.5B-bnb-4bit",
            "Qwen_1.5B": "unsloth/Qwen2.5-1.5B-bnb-4bit",
            "Qwen_3B": "unsloth/Qwen2.5-3B-bnb-4bit",
            "Qwen_7B": "unsloth/Qwen2.5-7B-bnb-4bit",
            "Qwen_14B": "unsloth/Qwen2.5-14B-bnb-4bit",
            "Qwen_32B": "unsloth/Qwen2.5-32B-bnb-4bit",
            "Qwen_0.5B_Instruct": "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit",
            "Qwen_1.5B_Instruct": "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
            "Qwen_3B_Instruct": "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
            "Qwen_7B_Instruct": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
            "Qwen_14B_Instruct": "unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
            "Qwen_32B_Instruct": "unsloth/Qwen2.5-32B-Instruct-bnb-4bit",
            "Centaur_8B":    "marcelbinz/Llama-3.1-Centaur-8B-adapter",
            "Mistral_7B_Instruct": "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
            "Mistral_7B": "unsloth/mistral-7b-v0.3-bnb-4bit",
            "Phi_4_8B": "unsloth/phi-4-bnb-4bit",
            "Phi_3.5_mini_Instruct": "unsloth/Phi-3.5-mini-instruct-bnb-4bit",
            "Phi_3_mini_Instruct": "unsloth/Phi-3-mini-4k-instruct-bnb-4bit",
            "Phi_3_medium_Instruct": "unsloth/Phi-3-medium-4k-instruct-bnb-4bit",
            "Gemma_2B": "unsloth/gemma-2-2b-bnb-4bit",
            "Gemma_9B": "unsloth/gemma-2-9b-bnb-4bit",
            "Gemma_27B": "unsloth/gemma-2-27b-bnb-4bit",
            "Gemma_2B_Instruct": "unsloth/gemma-2-2b-it-bnb-4bit",
            "Gemma_9B_Instruct": "unsloth/gemma-2-9b-it-bnb-4bit",
            "Gemma_27B_Instruct": "unsloth/gemma-2-27b-it-bnb-4bit",
        }

        model_aliases_mps = {
            "Deepseek_R1_1.5B_Qwen": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "Deepseek_R1_7B_Qwen" : "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "Deepseek_R1_8B_Llama": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            "Deepseek_R1_14B_Qwen": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            "Deepseek_R1_32B_Qwen": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            "Qwen_0.5B": "Qwen/Qwen2.5-0.5B",
            "Qwen_1.5B": "Qwen/Qwen2.5-1.5B",
            "Qwen_3B": "Qwen/Qwen2.5-3B",
            "Qwen_7B": "Qwen/Qwen2.5-7B",
            "Qwen_14B": "Qwen/Qwen2.5-14B",
            "Qwen_32B": "Qwen/Qwen2.5-32B",
            "Qwen_0.5B_Instruct": "Qwen/Qwen2.5-0.5B-Instruct",
            "Qwen_1.5B_Instruct": "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen_3B_Instruct": "Qwen/Qwen2.5-3B-Instruct",
            "Qwen_7B_Instruct": "Qwen/Qwen2.5-7B-Instruct",
            "Qwen_14B_Instruct": "Qwen/Qwen2.5-14B-Instruct",
            "Qwen_32B_Instruct": "Qwen/Qwen2.5-32B-Instruct",
            "Centaur_8B": "marcelbinz/Llama-3.1-Centaur-8B",
            "Mistral_7B_Instruct": "mistralai/Mistral-7B-Instruct-v0.3",
            "Mistral_7B": "mistralai/Mistral-7B-v0.3",
            "Phi_4_8B": "microsoft/phi-4",
            "Phi_3.5_mini_Instruct": "microsoft/Phi-3.5-mini-instruct",
            "Phi_3_mini_Instruct": "microsoft/Phi-3-mini-4k-instruct",
            "Phi_3_medium_Instruct": "microsoft/Phi-3-medium-4k-instruct",
            "Gemma_2B": "google/gemma-2b",
            "Gemma_9B": "google/gemma-9b",
            "Gemma_27B": "google/gemma-27b",
            "Gemma_2B_Instruct": "google/gemma-2b-it",
            "Gemma_9B_Instruct": "google/gemma-9b-it",
            "Gemma_27B_Instruct": "google/gemma-27b-it",
        }

        model_openai = {
            "gpt4o": "gpt-4o",
            "gpt4o-mini": "gpt-4o-mini",
            "o1-mini": "o1-mini"
        }

        model_anthropic = {
            "sonnet": "claude-3-5-sonnet-latest",
            "haiku": "claude-3-5-haiku-latest"
        }


        # Check if this is a reasoning model
        self.is_reasoning_model = model_name in self.REASONING_MODELS
        self.model_name = model_name  # Store the friendly model name

        # Set this so that it doesn't print the device at every inference
        logging.set_verbosity_error()

        if model_name in model_openai:
            self.openai_flag = True
            model_name = model_openai[model_name]
            self.api_model_name = model_name
            print("Using OpenAI API")
            self.client = OpenAI(api_key=self.openai_api_key)
        elif model_name in model_anthropic:
            self.anthropic_flag = True
            model_name = model_anthropic[model_name]
            self.api_model_name = model_name
            print("Using Anthropic API")
            self.client = anthropic.Anthropic(api_key=self.anthropic_api_key)
        elif model_name in model_aliases and device_map == "cuda:0" and use_unsloth:
            from unsloth import FastLanguageModel
            print("Using unsloth with GPU")
            model_alias = model_aliases[model_name]
            if "qwen" in model_alias.lower():
                max_seq_length = 4096  # adjust if Qwen requires a different sequence length
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(model_name=model_alias, max_seq_length=max_seq_length)
            FastLanguageModel.for_inference(self.model)
        elif model_name in model_aliases and device_map == "cpu": 
            model_alias = model_aliases[model_name]
            print("Using transformers with CPU")
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                model_alias, 
                device_map=device_map
            )
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_alias)
        elif device_map == "mps" and model_name in model_aliases_mps:
            model_alias = model_aliases_mps[model_name]
            print("Using transformers with mps")
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                model_alias,
                device_map=device_map
            )
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_alias)
        else:
            raise ValueError("Unsupported model or configuration")


    @classmethod
    def get_reasoning_models(cls) -> List[str]:
        """Return the list of models that support internal reasoning."""
        return cls.REASONING_MODELS

    def get_response(self, prompt: str) -> str:
        """Get response from the LLM."""
        self.thinking_time = 0  # Reset thinking time for this prompt
        self.last_thinking_tokens = ""  # Reset thinking tokens
        self.token_count = 0  # Reset token count

        if self.openai_flag:
            # Use the OpenAI API for chat completions
            # Build a dictionary of parameters for the API call
            params = {
                "model": self.api_model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 1.0,
                "stop": ["<<"],
            }
            
            # Select the appropriate token parameter based on the model name
            if self.api_model_name in ["gpt-4o", "gpt-4o-mini"]:
                params["max_tokens"] = 1  # Use max_tokens for GPT-4o models
            elif self.api_model_name in ["o1-mini", "o3-mini"]:
                params["max_completion_tokens"] = 1  # Use max_completion_tokens for O-models
            else:
                # Fallback behavior if the model name isn't one of the above.
                # You can choose to default to one of the parameters or raise an error.
                params["max_tokens"] = 1

            response = self.client.chat.completions.create(**params)
            generated_text = response.choices[0].message.content.strip()
        
        elif self.anthropic_flag:
            # Use the Anthropic API for completions
            response = self.client.messages.create(
                model=self.api_model_name,
                max_tokens=1,  # adjust based on your needs
                temperature=1.0,
                stop_sequences=["<<"],
                messages=[{"role": "user", "content": prompt}],
            )
            if response.content and len(response.content) > 0:
                generated_text = response.content[0].text.strip()
            else:
                generated_text = "X" # Anthropic API returned an empty response.

        elif self.is_reasoning_model:
            tokenized_prompt = self.tokenizer.encode(f"<｜begin▁of▁sentence｜><｜User｜>{prompt}<｜Assistant｜>", return_tensors="pt", add_special_tokens=False).to(self.model.device)

            ## REASONING PHASE
            batch_tokens = 20        # Generate tokens in small batches
            replacement_text = "Wait"  # Replacement text if end is prematurely detected
            end_think_token = self.tokenizer.encode("</think>", add_special_tokens=False)[0]
            replacement_ids = self.tokenizer.encode(replacement_text, add_special_tokens=False)

            is_thinking = True       # Are we in the thinking phase?
            finished = False
            final_tokens = tokenized_prompt

            start_time = time.time()  # Record the start time

            # Choose between time-based and token-based reasoning
            if self.reasoning_mode == "time":
                while not finished:
                    outputs = self.model.generate(input_ids=final_tokens, max_new_tokens=batch_tokens, do_sample=True, temperature=0.6, top_p=0.95)
                    new_tokens = outputs[:, final_tokens.shape[1]:]  # Only newly generated tokens
                    for token in new_tokens[0]:  # Iterate over tokens
                        elapsed_time = time.time() - start_time  # Check elapsed time
                        if elapsed_time >= self.max_thinking_time:  # Maximum thinking time reached
                            final_tokens = torch.cat([final_tokens, torch.tensor([[end_think_token]]).to(final_tokens.device)], dim=1)
                            finished = True
                            break
                        token_id = token.item()
                        if token_id == end_think_token and is_thinking:  # Detect </think> token
                            elapsed_time = time.time() - start_time
                            if elapsed_time < self.min_thinking_time:  # If we haven't "thought" long enough
                                final_tokens = torch.cat([final_tokens, torch.tensor([replacement_ids]).to(final_tokens.device)], dim=1)
                                continue
                            else:  # If we have "thought" long enough
                                is_thinking = False
                                finished = True
                        final_tokens = torch.cat([final_tokens, token.unsqueeze(0).unsqueeze(0)], dim=1)
                        if token_id == end_think_token:
                            finished = True
                            break
                    if finished:
                        break
            
            elif self.reasoning_mode == "tokens":
                # Token-based reasoning
                while not finished:
                    outputs = self.model.generate(input_ids=final_tokens, max_new_tokens=batch_tokens, do_sample=True, temperature=0.6, top_p=0.95)
                    new_tokens = outputs[:, final_tokens.shape[1]:]  # Only newly generated tokens
                    for token in new_tokens[0]:  # Iterate over tokens
                        self.token_count += 1  # Increment token count
                        if self.token_count >= self.max_thinking_tokens:  # Maximum token count reached
                            final_tokens = torch.cat([final_tokens, torch.tensor([[end_think_token]]).to(final_tokens.device)], dim=1)
                            finished = True
                            break
                        token_id = token.item()
                        if token_id == end_think_token and is_thinking:  # Detect </think> token
                            if self.token_count < self.min_thinking_tokens:  # If we haven't generated enough tokens
                                final_tokens = torch.cat([final_tokens, torch.tensor([replacement_ids]).to(final_tokens.device)], dim=1)
                                continue
                            else:  # If we have generated enough tokens
                                is_thinking = False
                                finished = True
                        final_tokens = torch.cat([final_tokens, token.unsqueeze(0).unsqueeze(0)], dim=1)
                        if token_id == end_think_token:
                            finished = True
                            break
                    if finished:
                        break
            
            output_text = self.tokenizer.decode(final_tokens[0], skip_special_tokens=True)
            
            # Extract thinking tokens by removing the prompt
            # First, format the prompt in the same way as it would appear in output_text
            formatted_prompt = f"<｜User｜>{prompt}<｜Assistant｜>"
            
            # Extract thinking tokens
            if formatted_prompt in output_text:
                self.last_thinking_tokens = output_text[len(formatted_prompt):]
            else:
                # Try a simpler approach if the formatted prompt isn't found exactly
                self.last_thinking_tokens = output_text[output_text.find("<｜Assistant｜>") + len("<｜Assistant｜>"):]
            
            # Further strip any <think> or </think> tags
            self.last_thinking_tokens = self.last_thinking_tokens.replace("<think>", "").replace("</think>", "").strip()

            # Record the thinking time
            self.thinking_time = time.time() - start_time

            ## GENERATION PHASE
            # Use the output_text as the new prompt to generate a single answer token.
            conclusive_text = "\nThe answer is <<"
            input_ids = self.tokenizer.encode(output_text+conclusive_text, return_tensors="pt").to(final_tokens.device)
            answer_output = self.model.generate(input_ids=input_ids, max_new_tokens=2, do_sample=True, temperature=0.6, top_p=0.95)
            generated_text = self.tokenizer.decode(answer_output[0, -2], skip_special_tokens=True)

        else:
            self.pipe = transformers.pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                trust_remote_code=True,
                pad_token_id=0,
                do_sample=True,
                temperature=1.0,
                max_new_tokens=1,
            )

            # Use the local pipeline (unsloth or transformers)
            generated_text = self.pipe(prompt)[0]["generated_text"][len(prompt):].strip()

        return generated_text
        
    def get_thinking_tokens(self):
        """Get the most recent thinking tokens for reasoning models."""
        return self.last_thinking_tokens
        
    def get_reasoning_parameters(self):
        """Get the current reasoning parameters."""
        if self.is_reasoning_model:
            if self.reasoning_mode == "time":
                return {
                    "reasoning_mode": self.reasoning_mode,
                    "min_thinking_time": self.min_thinking_time,
                    "max_thinking_time": self.max_thinking_time,
                    "min_thinking_tokens": None,
                    "max_thinking_tokens": None
                }
            else:  # token mode
                return {
                    "reasoning_mode": self.reasoning_mode,
                    "min_thinking_time": None,
                    "max_thinking_time": None,
                    "min_thinking_tokens": self.min_thinking_tokens,
                    "max_thinking_tokens": self.max_thinking_tokens
                }
        else:
            return {
                "reasoning_mode": None,
                "min_thinking_time": None,
                "max_thinking_time": None,
                "min_thinking_tokens": None,
                "max_thinking_tokens": None
            }