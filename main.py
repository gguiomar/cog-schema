from tasks.VSTtask import VSTtask
import agents.LLMagent as LLMagent
from manager.TaskManager import TaskManager

pipe = LLMagent.LLMAgent("centaur8b")
manager = TaskManager(n_simulations=10, nrounds=5, num_quadrants=2, pipe=pipe)
metrics = manager.run_simulations()