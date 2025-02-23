import os
import transformers
import torch
import time
import anthropic
from typing import Optional
from openai import OpenAI
from unsloth import FastLanguageModel

class LLMagent:
    def __init__(self, 
                 model_name: str, 
                 device_map: str = "cpu", 
                 max_seq_length: int = 32768, 
                 load_in_4bit: bool = True, 
                 use_unsloth: bool = False,
                 openai_api_key: Optional[str] = None,
                 anthropic_api_key: Optional[str] = None):
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
        
        If neither API key is provided and use_unsloth is False, then a local model (via transformers) is used.
        """
        self.openai_api_key = openai_api_key
        self.anthropic_api_key = anthropic_api_key
        self.openai_flag, self.anthropic_flag = False, False

        # Map friendly model names to their corresponding Hugging Face repository strings.
        model_aliases = {
            "Deepseek_R1_1B_Qwen": "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-unsloth-bnb-4bit",
            "Deepseek_R1_7B_Qwen" : "unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit",
            "Deepseek_R1_8B_Llama": "unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit",
            "Qwen_1B": "Qwen/Qwen2.5-1.5B",
            "Qwen_3B": "Qwen/Qwen2.5-3B",
            "Qwen_7B": "Qwen/Qwen2.5-7B",
            "Qwen_1B_Instruct": "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen_3B_Instruct": "Qwen/Qwen2.5-3B-Instruct",
            "Qwen_7B_Instruct": "Qwen/Qwen2.5-7B-Instruct",
            "Centaur_8B":    "marcelbinz/Llama-3.1-Centaur-8B-adapter"
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

        # Define which models are expected to support internal chain-of-thought
        reasoning_models = ["Deepseek_R1_1B_Qwen", "Deepseek_R1_7B_Qwen", "Deepseek_R1_8B_Llama"]
        self.is_reasoning_model = model_name in reasoning_models

        if model_name in model_openai:
            self.openai_flag = True
            model_name = model_openai[model_name]
            self.model_name = model_name
            print("Using OpenAI API")
            self.client = OpenAI(api_key=self.openai_api_key)
        elif model_name in model_anthropic:
            self.anthropic_flag = True
            model_name = model_anthropic[model_name]
            self.model_name = model_name
            print("Using Anthropic API")
            self.client = anthropic.Anthropic(api_key=self.anthropic_api_key)
        elif model_name in model_aliases and device_map == "cuda:0" and use_unsloth:
            print("Using unsloth with GPU")
            model_name = model_aliases[model_name]
            if "qwen" in model_name.lower():
                max_seq_length = 4096  # adjust if Qwen requires a different sequence length
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(model_name=model_name, max_seq_length=max_seq_length)
            FastLanguageModel.for_inference(self.model)

        elif model_name in model_aliases and device_map == "cpu": 
            model_name = model_aliases[model_name]
            print("Using transformers with CPU")
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                model_name, 
                device_map=device_map
            )
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        
        self.conversation_history = ""

    def get_response(self, prompt: str) -> str:
        """Get response from the LLM."""
        # Combine conversation history with the current prompt
        full_prompt = self.conversation_history + prompt

        if self.openai_flag:
            # Use the OpenAI API for chat completions
            # Build a dictionary of parameters for the API call
            params = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": full_prompt}],
                "temperature": 1.0,
                "stop": ["<<"],
            }
            
            # Select the appropriate token parameter based on the model name
            if self.model_name in ["gpt-4o", "gpt-4o-mini"]:
                params["max_tokens"] = 1  # Use max_tokens for GPT-4o models
            elif self.model_name in ["o1-mini", "o3-mini"]:
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
                model=self.model_name,
                max_tokens=1,  # adjust based on your needs
                temperature=1.0,
                stop_sequences=["<<"],
                messages=[{"role": "user", "content": full_prompt}],
            )
            if response.content and len(response.content) > 0:
                generated_text = response.content[0].text.strip()
            else:
                generated_text = "X" # Anthropic API returned an empty response.

        elif self.is_reasoning_model:
            tokenized_prompt = self.tokenizer.encode(f"<｜begin▁of▁sentence｜><｜User｜>{full_prompt}<｜Assistant｜>", return_tensors="pt", add_special_tokens=False).to(self.model.device)

            ## REASONING PHASE
            min_thinking_time = 10   # minimum thinking time in seconds
            max_thinking_time = 20   # maximum thinking time in seconds
            batch_tokens = 20        # Generate tokens in small batches
            replacement_text = "Wait"  # Replacement text if end is prematurely detected
            #start_think_token = self.tokenizer.encode("<think>", add_special_tokens=False)[0]
            end_think_token = self.tokenizer.encode("</think>", add_special_tokens=False)[0]
            replacement_ids = self.tokenizer.encode(replacement_text, add_special_tokens=False)

            is_thinking = True       # Are we in the thinking phase?
            finished = False
            final_tokens = tokenized_prompt

            start_time = time.time()  # Record the start time

            while not finished:
                outputs = self.model.generate(input_ids=final_tokens,max_new_tokens=batch_tokens,do_sample=True,temperature=0.6,top_p=0.95)
                new_tokens = outputs[:, final_tokens.shape[1]:]  # Only newly generated tokens
                for token in new_tokens[0]:  # Iterate over tokens
                    elapsed_time = time.time() - start_time # Check if the maximum thinking time has been reached
                    if elapsed_time >= max_thinking_time: # Append end-of-think token and break out of the loop
                        final_tokens = torch.cat([final_tokens, torch.tensor([[end_think_token]]).to(final_tokens.device)], dim=1)
                        finished = True
                        break
                    token_id = token.item()
                    if token_id == end_think_token and is_thinking:  # Detect </think> token
                        elapsed_time = time.time() - start_time
                        if elapsed_time < min_thinking_time:  # If we haven't "thought" long enough
                            final_tokens = torch.cat([final_tokens, torch.tensor([replacement_ids]).to(final_tokens.device)], dim=1) # Replace premature </think> with replacement text.
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

            output_text = self.tokenizer.decode(final_tokens[0], skip_special_tokens=True)

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
            generated_text = self.pipe(full_prompt)[0]["generated_text"][len(full_prompt):].strip()

        return generated_text


    def update_history(self, text: str):
        """Update conversation history."""
        self.conversation_history += text

    def reset_history(self):
        """Reset conversation history."""
        self.conversation_history = ""
