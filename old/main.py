from tasks.VSTtask import VSTtask
from agents.LLMagent import LLMagent
from manager.TaskManager import TaskManager

# Run Single Instance
model_name = "marcelbinz/Llama-3.1-Centaur-8B-adapter"
pipe = LLMagent(
    model_name=model_name, 
    device_map="cpu", 
    max_seq_length=32768, 
    load_in_4bit=True, 
    use_unsloth=False
)

manager = TaskManager(n_simulations=100, nrounds=10, num_quadrants=2, num_queues=1, pipe=pipe, verbose=False)
metrics = manager.run_simulations()




