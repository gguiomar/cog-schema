from unsloth import FastLanguageModel
import transformers


class LLMagent:
    def __init__(self, 
                 model_name: str, 
                 device_map: str = "cpu", 
                 max_seq_length: int = 32768, 
                 load_in_4bit: bool = True, 
                 use_unsloth: bool = False):
        """
        Initialize LLM agent with specified model and backend.
        
        Parameters:
        - model_name: str, the name or path of the model.
        - device_map: str, device to use (e.g., 'cpu', 'cuda:0').
        - max_seq_length: int, maximum sequence length for the model.
        - load_in_4bit: bool, whether to load the model in 4-bit precision (unsloth only).
        - use_unsloth: bool, whether to use unsloth or transformers.
        """
        if use_unsloth:
            # Load model and tokenizer using unsloth
            print("Using unsloth with GPU")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=max_seq_length,
                dtype=None,
                load_in_4bit=load_in_4bit,
            )
            # Prepare the model for inference
            FastLanguageModel.for_inference(model)
        else:
            # Load model and tokenizer using transformers
            print("Using transformers with CPU")
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_name, 
                device_map=device_map
            )
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

        # Create the text-generation pipeline
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
        # Combine conversation history with the current prompt
        full_prompt = self.conversation_history + prompt
        # Generate a response
        response = self.pipe(full_prompt)[0]["generated_text"][len(full_prompt):].strip()
        return response

    def update_history(self, text: str):
        """Update conversation history."""
        self.conversation_history += text

    def reset_history(self):
        """Reset conversation history."""
        self.conversation_history = ""
