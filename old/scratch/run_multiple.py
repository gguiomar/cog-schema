from quad.quadtext_min import ModifiedTask
from agents.LLMTaskRunner import LLMTaskRunner
from manager.SimulationManager import SimulationManager
import transformers

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

manager = SimulationManager(
    n_simulations=10,  # Number of simulations to run
    nrounds=5,          # Rounds per simulation
    num_quadrants=2,    # Number of quadrants
    pipe=pipe
)

metrics = manager.run_simulations()

# Print summary results
print("\n=== Simulation Results ===")
print(f"Success Rate: {metrics['success_rate']:.2%}")
print("\nQuadrant Distribution:")
for quadrant, data in metrics['quadrant_distribution'].items():
    print(f"{quadrant}:")
    print(f"  Times chosen: {data['times_chosen']}")
    print(f"  Times correct: {data['times_correct']}")