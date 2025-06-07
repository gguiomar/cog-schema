import os
import transformers
from transformers.utils import logging
import torch
import time
import anthropic
from typing import Optional, List, Union
from openai import OpenAI
import numpy as np
class LLMagent:

    INSTRUCT_MODELS = ["Mistral_7B_Instruct",
                        "Llama-3.2-3B-Instruct",
                        "Qwen_7B_Instruct",
                        "Gemma_2B_Instruct",
                        "Qwen_0.5B_Instruct",
                        "Qwen_1.5B_Instruct",
                        "Qwen_3B_Instruct",
                        "Qwen_14B_Instruct",
                        "Phi_4_mini_Instruct",
                        "Qwen3-4B",
                        "Qwen3-8B",
                        "Qwen3-1.7B",
                        "Qwen3-0.6B",
                        "Gemma_3B_Instruct",
                        "Gemma_12B_Instruct",
                        "Gemma_1B_Instruct",
                        "Llama-3.2-1B-Instruct"]
    
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
                 use_unsloth: bool = True,
                 openai_api_key: Optional[str] = None,
                 anthropic_api_key: Optional[str] = None,
                 reasoning_mode: str = "time",  # 'time' or 'tokens'
                 min_thinking_time: float = 5.0,
                 max_thinking_time: float = 10.0,
                 min_thinking_tokens: int = 200,
                 max_thinking_tokens: int = 500,
                 add_padding: bool = False):
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
        self.add_padding = add_padding

        # Map friendly model names to their corresponding Hugging Face repository strings.
        model_aliases = {
            "Deepseek_R1_1.5B_Qwen": "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-unsloth-bnb-4bit",
            "Deepseek_R1_7B_Qwen" : "unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit",
            "Deepseek_R1_8B_Llama": "unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit", 
            "Deepseek_R1_14B_Qwen": "unsloth/DeepSeek-R1-Distill-Qwen-14B-unsloth-bnb-4bit",
            "Deepseek_R1_32B_Qwen": "unsloth/DeepSeek-R1-Distill-Qwen-32B-bnb-4bit",
            "Qwen_3B_quantized": "unsloth/Qwen2.5-3B-bnb-4bit",
            "Qwen_7B_quantized": "unsloth/Qwen2.5-7B-bnb-4bit",
            "Qwen_14B_quantized": "unsloth/Qwen2.5-14B-bnb-4bit",
            "Qwen_32B_quantized": "unsloth/Qwen2.5-32B-bnb-4bit",
            "Qwen_3B_Instruct_quantized": "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
            "Qwen_7B_Instruct_quantized": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
            "Qwen_14B_Instruct_quantized": "unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
            "Qwen_32B_Instruct_quantized": "unsloth/Qwen2.5-32B-Instruct-bnb-4bit",
            "Qwen_0.5B": "unsloth/Qwen2.5-0.5B",
            "Qwen3-0.6B": "unsloth/Qwen3-0.6B",
            "Qwen_1.5B": "unsloth/Qwen2.5-1.5B",
            "Qwen3_1.7B": "unsloth/Qwen3-1.7B",
            "Qwen_3B": "unsloth/Qwen2.5-3B",
            "Qwen3_4B": "unsloth/Qwen3-4B",
            "Qwen_7B": "unsloth/Qwen2.5-7B",
            "Qwen3_8B": "unsloth/Qwen3-8B",
            "Qwen_0.5B_Instruct": "unsloth/Qwen2.5-0.5B-Instruct",
            "Qwen_1.5B_Instruct": "unsloth/Qwen2.5-1.5B-Instruct",
            "Qwen_3B_Instruct": "unsloth/Qwen2.5-3B-Instruct",
            "Qwen_7B_Instruct": "unsloth/Qwen2.5-7B-Instruct",
            "Qwen_14B_Instruct": "unsloth/Qwen2.5-14B-Instruct",
            "Centaur_8B": "marcelbinz/Llama-3.1-Centaur-8B-adapter",
            "Mistral_7B_Instruct_quantized": "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
            "Mistral_7B_quantized": "unsloth/mistral-7b-v0.3-bnb-4bit",
            "Mistral_7B_Instruct": "unsloth/mistral-7b-instruct-v0.3",
            "Mistral_7B": "unsloth/mistral-7b-v0.3",
            "Phi_4_8B": "unsloth/phi-4-bnb-4bit",
            "Phi_3.5_mini_Instruct": "unsloth/Phi-3.5-mini-instruct-bnb-4bit",
            "Phi_3_mini_Instruct": "unsloth/Phi-3-mini-4k-instruct-bnb-4bit",
            "Phi_3_medium_Instruct": "unsloth/Phi-3-medium-4k-instruct-bnb-4bit",
            "Phi_4_mini_Instruct": "microsoft/Phi-4-mini-instruct",
            "Gemma_2B": "unsloth/gemma-2-2b",
            "Gemma_2B_Instruct_quantized": "unsloth/gemma-2-2b-it-bnb-4bit",
            "Gemma_9B_Instruct_quantized": "unsloth/gemma-2-9b-it-bnb-4bit",
            "Gemma_27B_Instruct_quantized": "unsloth/gemma-2-27b-it-bnb-4bit",
            "Gemma_1B_Instruct": "unsloth/gemma-3-1b-it",
            "Gemma_2B_Instruct": "unsloth/gemma-2-2b-it",
            "Gemma_3B_Instruct": "unsloth/gemma-3-4b-it",
            "Gemma_12B_Instruct": "unsloth/gemma-3-12b-it",
            "Gemma_9B_quantized": "unsloth/gemma-2-9b-bnb-4bit",
            "Gemma_27B_quantized": "unsloth/gemma-2-27b-bnb-4bit",
            "Llama-3.2-1B-Instruct": "unsloth/Llama-3.2-1B-Instruct",
            "Llama-3.2-3B-Instruct": "unsloth/Llama-3.2-3B-Instruct",
            "Llama-3.1-8B-Instruct": "unsloth/Llama-3.1-8B-Instruct",
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
            "Mistral_7B_Instruct": "alokabhishek/Mistral-7B-Instruct-v0.2-bnb-4bit", # mistralai/Mistral-7B-Instruct-v0.3
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
            "Gemma3_4B_Instruct": "google/gemma-3-4b-it", 
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
        self.is_instruct_model = model_name in self.INSTRUCT_MODELS
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
        elif model_name in model_aliases and device_map == "cuda:0" and not use_unsloth:
            model_alias = model_aliases_mps[model_name]
            print("Using transformers with GPU")
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                model_alias,
                device_map=device_map,
            )
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_alias)
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

        if self.add_padding:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))


    @classmethod
    def get_reasoning_models(cls) -> List[str]:
        """Return the list of models that support internal reasoning."""
        return cls.REASONING_MODELS
    
    @classmethod
    def get_instruct_models(cls) -> List[str]:
        """Return the list of models that were finetuned for instruction/chat."""
        return cls.INSTRUCT_MODELS

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
            
            # allowed_ids = [self.tokenizer.encode(ch, add_special_tokens=False)[0]
            #    for ch in ["A","B","C","D"]]
            
            def only_abcd(batch_id, input_ids):
                return allowed_ids
            
            generated_text = self.generate_with_logits(prompt, only_abcd)

        return generated_text
        
    def get_thinking_tokens(self):
        """Get the most recent thinking tokens for reasoning models."""
        return self.last_thinking_tokens
    
    
    def generate_with_logits(self, prompt, only_abcd: callable): 
        """Generate a single token and return the logits for all tokens in the vocabulary."""
        
        if not isinstance(prompt, str):        
            text = self.tokenizer.apply_chat_template(
                    prompt,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
            )
            
            inputs = self.tokenizer(
                [text],
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        else:
            # Tokenize and move to model device
            inputs = self.tokenizer(
                prompt + "\nYou choose:", 
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        

        # Generate exactly one new token, returning raw scores
        gen_out = self.model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,  # Disable sampling to always pick highest probability
            temperature=1.0,  # Set temperature to 0 for deterministic output
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=self.tokenizer.eos_token_id,
            #prefix_allowed_tokens_fn=only_abcd
        )
        
        full_sequence = gen_out.sequences[0]  # shape: (input_len + 1,)
        input_len = inputs["input_ids"].shape[-1]

        # grab only the new token IDs
        new_token_ids = full_sequence[input_len:]
        # decode them to a string and strip whitespace
        generated_text = self.tokenizer.decode(new_token_ids, 
                                            skip_special_tokens=True).strip()

        # Extract the single-step logits
        logits = gen_out.scores[0][0].detach().cpu().numpy()  # shape (vocab_size,)

        # Convert to probabilities
        probs = torch.softmax(torch.tensor(logits), dim=0).numpy()

        # Filter indices by probability threshold
        threshold = 0.01
        idxs = np.where(probs > threshold)[0]
        probs_filtered = probs[idxs]
        # Sort token IDs by probability descending
        order = np.argsort(probs_filtered)[::-1]
        top_idxs = idxs[order]
        top_probs = probs_filtered[order]

        # Decode tokens into human-readable strings
        top_tokens = [self.tokenizer.decode([int(i)]) for i in top_idxs]
        
        # Create dictionary of tokens and their probabilities
        token_probs = {tok: float(prob) for tok, prob in zip(top_tokens, top_probs)}
        
        # Store the token-probability dictionary
        self.last_logits = token_probs
        
        # Print the probabilities
        # print("Top tokens >0.01:")
        # for tok, p in token_probs.items():
        #     print(f"{tok!r}: {p:.4f}")
        
        return generated_text
        
    def get_last_logits(self):
        """Get the logit distribution from the last response."""
        return self.last_logits
        
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