import transformers


class LLMAgent:
    def __init__(self, model_name: str):
        if model_name == "centaur8b":
            model_path = "agents/llama_centaur_adapter/"
            model_path = "agents/llama_centaur_adapter/"

        # initialize model and tokenizer
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map="cpu"
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)

        pipe = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        trust_remote_code=True,
        pad_token_id=0,
        do_sample=True,
        temperature=1.0,
        max_new_tokens=1,
    )

    return pipe



