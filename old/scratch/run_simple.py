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

# Run the task
runner = LLMTaskRunner(nrounds=5, num_quadrants=2, pipe=pipe)
stats = runner.run()

# Print results
print("\nGame Statistics:")
print(f"Correct quadrant was: {stats['correct_quadrant']}")
print(f"LLM chose: {stats['final_choice']}")
print(f"Success: {stats['success']}")
print("\nRound History:")
for i, round_data in enumerate(stats['rounds'], 1):
    print(f"Round {i}:")
    print(f"  Squares shown: {round_data['squares_shown']}")
    print(f"  Choice: {round_data['choice']}")
    print(f"  Result: {round_data['result']}")