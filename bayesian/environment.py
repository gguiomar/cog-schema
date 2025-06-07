import numpy as np

class TemporalReasoningEnvironment:
    """Environment for temporal reasoning tasks with biased cues."""
    
    def __init__(self, k: int, p_t: float, p_f: float, rng: np.random.Generator,
                 use_hidden_cues: bool = False, min_available_cues: int = None, 
                 max_available_cues: int = None):
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
        use_hidden_cues : bool
            Whether to use hidden cues (subset of cues available each round)
        min_available_cues : int
            Minimum number of cues available per round (default: 1)
        max_available_cues : int
            Maximum number of cues available per round (default: k)
        """
        self.k = k
        self.p_t = p_t
        self.p_f = p_f
        self.rng = rng
        self.use_hidden_cues = use_hidden_cues
        
        # Set defaults for cue availability
        self.min_available_cues = min_available_cues if min_available_cues is not None else 1
        self.max_available_cues = max_available_cues if max_available_cues is not None else k
        
        # Validate parameters
        if self.min_available_cues < 1:
            raise ValueError("min_available_cues must be at least 1")
        if self.max_available_cues > k:
            raise ValueError("max_available_cues cannot exceed k")
        if self.min_available_cues > self.max_available_cues:
            raise ValueError("min_available_cues cannot exceed max_available_cues")

    def start_trial(self) -> int:
        """Start a new trial by randomly selecting a true target location."""
        return self.rng.integers(self.k)

    def sample_round(self, true_z: int):
        """
        Sample a cue and color for a round, with optional hidden cues.
        
        Parameters:
        -----------
        true_z : int
            The true target location
            
        Returns:
        --------
        tuple
            (cue, color, available_cues) where:
            - cue is the location that was sampled
            - color is 0 or 1
            - available_cues is a list of cues that were available this round
        """
        # Determine available cues for this round
        if self.use_hidden_cues:
            # Sample number of available cues
            n_available = self.rng.integers(self.min_available_cues, self.max_available_cues + 1)
            # Randomly select which cues are available
            available_cues = sorted(self.rng.choice(self.k, size=n_available, replace=False))
        else:
            # All cues are available
            available_cues = list(range(self.k))
        
        # Sample a cue from the available ones
        cue = self.rng.choice(available_cues)
        
        # Determine color probability based on whether cue matches true target
        if cue == true_z:
            p_color_1 = self.p_t
        else:
            p_color_1 = self.p_f
        
        # Sample color
        color = int(self.rng.random() < p_color_1)
        
        return cue, color, available_cues
