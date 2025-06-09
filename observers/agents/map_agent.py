import numpy as np

class MAPAgent:
    """Agent that makes Maximum A Posteriori decisions."""
    
    def __init__(self, k: int):
        """
        Initialize the MAP agent.
        
        Parameters:
        -----------
        k : int
            Number of possible locations/targets
        """
        self.k = k
   
    def get_decision(self, learned_posterior):
        """Make decision based on maximum posterior probability."""
        return np.argmax(learned_posterior)
   
    def is_correct(self, decision, true_z):
        """Check if decision matches true target."""
        return int(decision == true_z)
   
    def get_map_probability(self, learned_posterior):
        """Get the maximum posterior probability."""
        return np.max(learned_posterior)
