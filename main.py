from tasks.VSTtask import VSTtask
from agents.LLMagent import LLMagent
from manager.TaskManager import TaskManager

pipe = LLMagent("centaur8b")
manager = TaskManager(n_simulations=1, nrounds=5, num_quadrants=2, pipe=pipe, verbose=True)
metrics = manager.run_simulations()