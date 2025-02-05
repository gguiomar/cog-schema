import os
import transformers

class LLMagent:
    def __init__(self, 
                 model_name: str, 
                 device_map: str = "cpu", 
                 max_seq_length: int = 32768, 
                 load_in_4bit: bool = True, 
                 use_unsloth: bool = False,
                 use_openai_api: bool = False,
                 use_anthropic_api: bool = False):
        """
        Initialize LLM agent with specified model and backend.
        
        Parameters:
        - model_name: str, the name or path of the model. 
          (For OpenAI API, e.g. "gpt-3.5-turbo" or "gpt-4o-mini"; for Anthropic API, e.g. "claude-v1")
        - device_map: str, device to use (e.g., 'cpu', 'cuda:0').
        - max_seq_length: int, maximum sequence length for the model.
        - load_in_4bit: bool, whether to load the model in 4-bit precision (unsloth only).
        - use_unsloth: bool, whether to use unsloth or transformers.
        - use_openai_api: bool, whether to use the OpenAI API.
        - use_anthropic_api: bool, whether to use the Anthropic API.
        """
        self.use_openai_api = use_openai_api
        self.use_anthropic_api = use_anthropic_api

        if self.use_openai_api:
            print("Using OpenAI API for GPT model")
            self.model_name = model_name  # e.g., "gpt-3.5-turbo", "gpt-4o-mini", etc.
            # Instantiate the new OpenAI client using your API key
            from openai import OpenAI
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif self.use_anthropic_api:
            print("Using Anthropic API for GPT model")
            self.model_name = model_name  # e.g., "claude-v1"
            # Delay the import until it's needed
            import anthropic
            self.client = anthropic.Anthropic()
        elif use_unsloth:
            # Delay the unsloth import until it's needed
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
        
        if self.use_openai_api:
            # Use the new OpenAI client interface for chat completions
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=1.0,
                max_tokens=1,     # adjust if necessary; the local pipeline was set to generate 1 token
                stop=["<<"]       # customize stop tokens if needed
            )
            # Extract the generated text from the response
            generated_text = response.choices[0].message.content.strip()
        elif self.use_anthropic_api:
            # Set a system prompt as needed for your application.
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=1,  # adjust based on your needs
                temperature=1.0,
                stop_sequences=["<<"],
                messages=[{"role": "user", "content": full_prompt}],
            )    
            generated_text = response.content[0].text.strip()
        else:
            # Use local pipeline (unsloth or transformers)
            generated_text = self.pipe(full_prompt)[0]["generated_text"][len(full_prompt):].strip()
        return generated_text

    def update_history(self, text: str):
        """Update conversation history."""
        self.conversation_history += text

    def reset_history(self):
        """Reset conversation history."""
        self.conversation_history = ""
