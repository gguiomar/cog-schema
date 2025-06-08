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


class RLEnvironmentWrapper:
    """Wrapper for TemporalReasoningEnvironment that adds reward functionality for RL agents."""
    
    def __init__(self, base_env: TemporalReasoningEnvironment, reward_value: float = 1.0):
        """
        Initialize the RL environment wrapper.
        
        Parameters:
        -----------
        base_env : TemporalReasoningEnvironment
            The base environment to wrap
        reward_value : float
            Reward value for correct final decisions
        """
        self.base_env = base_env
        self.reward_value = reward_value
        self.current_trial_target = None
        self.episode_step = 0
        self.max_episode_length = None
    
    def start_trial(self, max_episode_length: int = None) -> int:
        """
        Start a new trial and return the true target.
        
        Parameters:
        -----------
        max_episode_length : int, optional
            Maximum number of steps in this episode
            
        Returns:
        --------
        int
            The true target location
        """
        self.current_trial_target = self.base_env.start_trial()
        self.episode_step = 0
        self.max_episode_length = max_episode_length
        return self.current_trial_target
    
    def step(self, action: int, is_final_step: bool = False):
        """
        Take a step in the environment with reward calculation.
        
        Parameters:
        -----------
        action : int
            The action taken by the agent (cue selection or final decision)
        is_final_step : bool
            Whether this is the final decision step
            
        Returns:
        --------
        tuple
            (cue, color, available_cues, reward, done) where:
            - cue is the sampled cue location
            - color is the observed outcome (0 or 1)
            - available_cues is the list of available cues
            - reward is the reward signal (0 during sampling, R for correct final decision)
            - done is whether the episode is finished
        """
        self.episode_step += 1
        
        if is_final_step:
            # Final decision step - no environment sampling, just reward calculation
            reward = self.reward_value if action == self.current_trial_target else 0.0
            done = True
            # Return dummy values for cue and color since this is decision step
            return None, None, list(range(self.base_env.k)), reward, done
        else:
            # Sampling step - get environment response
            cue, color, available_cues = self.base_env.sample_round(self.current_trial_target)
            reward = 0.0  # No reward during sampling
            
            # Check if episode should end (if max length specified)
            done = False
            if self.max_episode_length is not None and self.episode_step >= self.max_episode_length:
                done = True
                
            return cue, color, available_cues, reward, done
    
    def sample_round(self, true_z: int):
        """
        Compatibility method for non-RL agents - delegates to base environment.
        
        Parameters:
        -----------
        true_z : int
            The true target location
            
        Returns:
        --------
        tuple
            (cue, color, available_cues)
        """
        return self.base_env.sample_round(true_z)
    
    @property
    def k(self):
        """Number of possible locations."""
        return self.base_env.k
    
    @property
    def p_t(self):
        """Probability of correct color when cue matches true target."""
        return self.base_env.p_t
    
    @property
    def p_f(self):
        """Probability of correct color when cue doesn't match true target."""
        return self.base_env.p_f
    
    @property
    def use_hidden_cues(self):
        """Whether hidden cues are enabled."""
        return self.base_env.use_hidden_cues
    
    @property
    def min_available_cues(self):
        """Minimum number of available cues per round."""
        return self.base_env.min_available_cues
    
    @property
    def max_available_cues(self):
        """Maximum number of available cues per round."""
        return self.base_env.max_available_cues
