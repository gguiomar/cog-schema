import numpy as np

class TemporalReasoningEnvironment:
    """Environment for temporal reasoning tasks with biased cues."""
    
    def __init__(self, k: int, p_t: float, p_f: float, rng: np.random.Generator):
        """
        Initialize the environment.
        
        Parameters:
        -----------
        k : int
            Number of possible locations/targets
        p_t : float
            Probability of correct color when cue matches true target
        p_f : float
            Probability of correct color when cue doesn't match true target
        rng : np.random.Generator
            Random number generator
        """
        self.k = k
        self.p_t = p_t
        self.p_f = p_f
        self.rng = rng

    def start_trial(self) -> int:
        """Start a new trial by randomly selecting a true target location."""
        return self.rng.integers(self.k)

    def sample_round(self, true_z: int):
        """
        Sample a cue and color for a round.
        
        Parameters:
        -----------
        true_z : int
            The true target location
            
        Returns:
        --------
        tuple
            (cue, color) where cue is the location and color is 0 or 1
        """
        cue = self.rng.integers(self.k)
        if cue == true_z:
            p_color_1 = self.p_t
        else:
            p_color_1 = self.p_f
        color = int(self.rng.random() < p_color_1)
        return cue, color
