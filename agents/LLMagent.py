import os
import transformers
import anthropic
from typing import Optional
from openai import OpenAI
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

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
            #"Deepseek_R1_1B_Qwen": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "Deepseek_R1_7B_Qwen" : "unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit",
            #"Deepseek_R1_7B_Qwen": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "Deepseek_R1_8B_Llama": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
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
        reasoning_models = ["Deepseek_R1_1B_Qwen", "Deepseek_R1_7B_Qwen", "Deepseek_R1_8B_Llama", "o1-mini"]
        self.is_reasoning_model = model_name in reasoning_models
        #self.SYSTEM_PROMPT = """Respond in the following format:<think>...</think><answer>...</answer>"""

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
            # Optionally adjust parameters based on model (e.g., max_seq_length)
            if "qwen" in model_name.lower():
                max_seq_length = 4096  # adjust if Qwen requires a different sequence length

            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=max_seq_length,
                fast_inference=True,
                dtype=None,
            )

            #FastLanguageModel.for_inference(model)

            #self.pipe = transformers.pipeline(
                #"text-generation",
                #model=model,
                #tokenizer=tokenizer,
                #trust_remote_code=True,
                #pad_token_id=0,
                #do_sample=True,
                #temperature=1.0,
                #max_new_tokens=1,
            #)

        elif model_name in model_aliases and device_map == "cpu": 
            model_name = model_aliases[model_name]
            print("Using transformers with CPU")
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_name, 
                device_map=device_map
            )
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
            self.pipe = transformers.pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                trust_remote_code=True,
                pad_token_id=0,
                do_sample=True,
                temperature=1.0,
                max_new_tokens=1,
            )
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
    
        else:
            #tokenized_prompt = self.tokenizer.apply_chat_template([
                #{"role" : "user", "content" : full_prompt},
            #], tokenize = False, add_generation_prompt = True)

            tokenized_prompt = f"<｜begin▁of▁sentence｜><｜User｜>{full_prompt}<｜Assistant｜>"

            from vllm import SamplingParams
            sampling_params = SamplingParams(
                temperature = 0.8,
                top_p = 0.95,
                max_tokens = 1000,
            )

            generated_text = self.model.fast_generate(
                tokenized_prompt,
                sampling_params = sampling_params,
            )[0].outputs[0].text

            print(generated_text)

            # Use the local pipeline (unsloth or transformers)
            #generated_text = self.pipe(full_prompt)[0]["generated_text"][len(full_prompt):].strip()

        return generated_text


    def update_history(self, text: str):
        """Update conversation history."""
        self.conversation_history += text

    def reset_history(self):
        """Reset conversation history."""
        self.conversation_history = ""
