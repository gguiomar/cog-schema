import numpy as np

class RandomPolicyAgent:
    """Agent that makes random decisions."""
    
    def __init__(self, k: int, rng: np.random.Generator):
        """
        Initialize the random policy agent.
        
        Parameters:
        -----------
        k : int
            Number of possible locations/targets
        rng : np.random.Generator
            Random number generator
        """
        self.k = k
        self.rng = rng
   
    def get_decision(self, learned_posterior):
        """Make a random decision regardless of posterior."""
        return self.rng.integers(self.k)
   
    def is_correct(self, decision, true_z):
        """Check if decision matches true target."""
        return int(decision == true_z)
