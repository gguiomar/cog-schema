from quad.quadtext_min import ModifiedTask
import agents.LLMagent as LLMagent
from manager.SimulationManager import SimulationManager
import transformers

pipe = LLMagent.LLMsAgent("centaur8b")
manager = SimulationManager(n_simulations = 10, nrounds = 5, num_quadrants = 2, pipe = pipe)
metrics = manager.run_simulations()

# Print summary results
print("\n=== Simulation Results ===")
print(f"Success Rate: {metrics['success_rate']:.2%}")
print("\nQuadrant Distribution:")
for quadrant, data in metrics['quadrant_distribution'].items():
    print(f"{quadrant}:")
    print(f"  Times chosen: {data['times_chosen']}")
    print(f"  Times correct: {data['times_correct']}")