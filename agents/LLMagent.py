import os
import transformers
from typing import Optional

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
          (For OpenAI API, e.g. "gpt-3.5-turbo" or "gpt-4o-mini"; for Anthropic API, e.g. "claude-v1")
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
        
        if self.openai_api_key:
            print("Using OpenAI API for GPT model")
            self.model_name = model_name  # e.g., "gpt-3.5-turbo", "gpt-4o-mini", etc.
            # Instantiate the OpenAI client using the provided API key
            from openai import OpenAI
            self.client = OpenAI(api_key=self.openai_api_key)
        elif self.anthropic_api_key:
            print("Using Anthropic API for GPT model")
            self.model_name = model_name  # e.g., "claude-v1"
            # Delay the import until it's needed
            import anthropic
            # Pass the API key to the Anthropic client (ensure your anthropic package supports this)
            self.client = anthropic.Anthropic(api_key=self.anthropic_api_key)
        elif use_unsloth:
            from unsloth import FastLanguageModel
            print("Using unsloth with GPU")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=max_seq_length,
                dtype=None,
                load_in_4bit=load_in_4bit,
            )
            FastLanguageModel.for_inference(model)
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
        else:
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
        
        if self.openai_api_key:
            # Use the OpenAI API for chat completions
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=1.0,
                max_tokens=1,     # adjust if necessary
                stop=["<<"]       # customize stop tokens if needed
            )
            generated_text = response.choices[0].message.content.strip()
        elif self.anthropic_api_key:
            # Use the Anthropic API for completions
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=1,  # adjust based on your needs
                temperature=1.0,
                stop_sequences=["<<"],
                messages=[{"role": "user", "content": full_prompt}],
            )    
            generated_text = response.content[0].text.strip()
        else:
            # Use the local pipeline (unsloth or transformers)
            generated_text = self.pipe(full_prompt)[0]["generated_text"][len(full_prompt):].strip()
        return generated_text

    def update_history(self, text: str):
        """Update conversation history."""
        self.conversation_history += text

    def reset_history(self):
        """Reset conversation history."""
        self.conversation_history = ""
