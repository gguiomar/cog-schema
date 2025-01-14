import transformers
from agents.LLMTaskRunner import LLMTaskRunner

model_path = "agents/llama_centaur_adapter/"
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

# Run the task with detailed output
runner = DetailedLLMTaskRunner(nrounds=5, num_quadrants=2, pipe=pipe)
stats = runner.run_with_output()