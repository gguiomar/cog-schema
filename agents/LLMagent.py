import transformers

class LLMagent:
    def __init__(self, model_name: str, device_map: str = "cpu"):
        """Initialize LLM agent with specified model."""
        if model_name == "centaur8b":
            model_path = "agents/llama_centaur_adapter/"
            
        # Initialize model and tokenizer
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map = device_map
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        
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
        """Get response from LLM."""
        full_prompt = self.conversation_history + prompt
        response = self.pipe(full_prompt)[0]['generated_text'][len(full_prompt):].strip()
        return response.upper()
    
    def update_history(self, text: str):
        """Update conversation history."""
        self.conversation_history += text
        
    def reset_history(self):
        """Reset conversation history."""
        self.conversation_history = ""
