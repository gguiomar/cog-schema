import numpy as np

class RLAgent:
    """Pure Q-learning agent for cue selection."""
    
    def __init__(self, k: int, p_t: float, p_f: float, alpha: float = 0.1, 
                 gamma: float = 0.9, beta: float = 1.0,
                 state_discretization: int = 10):
        """
        Initialize the RL agent.
        
        Parameters:
        -----------
        k : int
            Number of possible locations/targets
        p_t : float
            Probability of correct color when cue matches true target
        p_f : float
            Probability of correct color when cue doesn't match true target
        alpha : float
            Learning rate for Q-learning
        gamma : float
            Discount factor
        beta : float
            Inverse temperature for action selection
        state_discretization : int
            Number of bins for discretizing belief states
        """
        self.k = k
        self.p_t = p_t
        self.p_f = p_f
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.state_discretization = state_discretization
        
        # Initialize Q-table
        # State space: discretized belief states
        # Action space: k possible cues
        self.q_values = np.zeros((state_discretization ** k, k))
        
        # Bayesian belief tracking
        self.posterior = np.full(k, 1.0 / k)
        
    def discretize_state(self, posterior):
        """
        Convert continuous posterior to discrete state index.
        
        Parameters:
        -----------
        posterior : np.array
            Current posterior distribution
            
        Returns:
        --------
        int
            Discrete state index
        """
        # Simple discretization: bin each probability dimension
        discretized = np.floor(posterior * self.state_discretization).astype(int)
        discretized = np.clip(discretized, 0, self.state_discretization - 1)
        
        # Convert to single index
        state_index = 0
        for i, val in enumerate(discretized):
            state_index += val * (self.state_discretization ** i)
        
        return min(state_index, self.q_values.shape[0] - 1)
    
    def select_action(self, available_cues: list, state_index: int = None):
        """
        Select action using Q-values and softmax policy.
        
        Parameters:
        -----------
        available_cues : list
            List of available cues
        state_index : int, optional
            Current state index (computed if not provided)
            
        Returns:
        --------
        int
            Selected action
        """
        if not available_cues:
            raise ValueError("No available cues")
        
        if state_index is None:
            state_index = self.discretize_state(self.posterior)
        
        # Get Q-values for available actions
        q_values = []
        for action in available_cues:
            q_values.append(self.q_values[state_index, action])
        
        q_values = np.array(q_values)
        
        # Softmax action selection
        exp_values = np.exp(self.beta * q_values)
        probabilities = exp_values / np.sum(exp_values)
        
        # Sample action
        action_idx = np.random.choice(len(available_cues), p=probabilities)
        return available_cues[action_idx]
    
    def update_q_values(self, state: int, action: int, reward: float, 
                       next_state: int, done: bool):
        """
        Update Q-values using TD learning.
        
        Parameters:
        -----------
        state : int
            Current state index
        action : int
            Action taken
        reward : float
            Reward received
        next_state : int
            Next state index
        done : bool
            Whether episode is finished
        """
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_values[next_state, :])
        
        td_error = target - self.q_values[state, action]
        self.q_values[state, action] += self.alpha * td_error
    
    def update(self, cue: int, color: int):
        """
        Update posterior beliefs using Bayesian updating.
        
        Parameters:
        -----------
        cue : int
            The cue location
        color : int
            The observed color (0 or 1)
            
        Returns:
        --------
        tuple
            (shannon_surprise, bayesian_surprise, predictive_prob)
        """
        prior = self.posterior.copy()
        likelihood = np.zeros(self.k)
        
        for z in range(self.k):
            if z == cue:
                likelihood[z] = self.p_t if color == 1 else (1.0 - self.p_t)
            else:
                likelihood[z] = self.p_f if color == 1 else (1.0 - self.p_f)
        
        # Calculate predictive probability
        predictive_prob = np.dot(likelihood, prior)
        
        # Update posterior using Bayes rule
        self.posterior = likelihood * prior / predictive_prob
        
        # Calculate surprises
        shannon_surprise = -np.log(predictive_prob)
        bayesian_surprise = np.sum(self.posterior * np.log((self.posterior + 1e-12) / (prior + 1e-12)))
        
        return shannon_surprise, bayesian_surprise, predictive_prob
    
    def get_decision(self, learned_posterior=None):
        """Make final decision using MAP estimation."""
        posterior = learned_posterior if learned_posterior is not None else self.posterior
        return np.argmax(posterior)
    
    def is_correct(self, decision, true_z):
        """Check if decision matches true target."""
        return int(decision == true_z)
    
    @property
    def entropy(self):
        """Calculate the entropy of the current posterior."""
        p = np.clip(self.posterior, 1e-12, 1.0)
        return -np.sum(p * np.log(p))
    
    def reset(self):
        """Reset the agent to uniform prior."""
        self.posterior = np.full(self.k, 1.0 / self.k)
