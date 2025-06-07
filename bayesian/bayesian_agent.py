import numpy as np

class BayesAgent:
    """Exact Bayesian updater in probability space; returns surprise metrics."""
    
    def __init__(self, k: int, p_t: float, p_f: float):
        """
        Initialize the Bayesian agent.
        
        Parameters:
        -----------
        k : int
            Number of possible locations/targets
        p_t : float
            Probability of correct color when cue matches true target
        p_f : float
            Probability of correct color when cue doesn't match true target
        """
        self.k = k
        self.p_t = p_t
        self.p_f = p_f
        self.posterior = np.full(k, 1.0 / k)

    def update(self, cue: int, color: int):
        """
        Update posterior beliefs and calculate surprise metrics.
        
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
        shannon_surprise = -np.log(predictive_prob)  # Shannon surprise
        bayesian_surprise = np.sum(self.posterior * np.log((self.posterior + 1e-12) / (prior + 1e-12)))  # KL surprise
        
        return shannon_surprise, bayesian_surprise, predictive_prob

    @property
    def entropy(self):
        """Calculate the entropy of the current posterior."""
        p = np.clip(self.posterior, 1e-12, 1.0)
        return -np.sum(p * np.log(p))

    def get_decision(self, learned_posterior=None):
        """Make final decision using MAP estimation."""
        posterior = learned_posterior if learned_posterior is not None else self.posterior
        return np.argmax(posterior)
    
    def is_correct(self, decision, true_z):
        """Check if decision matches true target."""
        return int(decision == true_z)

    def reset(self):
        """Reset the agent to uniform prior."""
        self.posterior = np.full(self.k, 1.0 / self.k)


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


class ActiveInferenceAgent:
    """Agent that minimizes expected surprise (active inference)."""
    
    def __init__(self, k: int, p_t: float, p_f: float, beta: float = 1.0):
        """
        Initialize the active inference agent.
        
        Parameters:
        -----------
        k : int
            Number of possible locations/targets
        p_t : float
            Probability of correct color when cue matches true target
        p_f : float
            Probability of correct color when cue doesn't match true target
        beta : float
            Inverse temperature parameter for action selection
        """
        self.k = k
        self.p_t = p_t
        self.p_f = p_f
        self.beta = beta
        self.posterior = np.full(k, 1.0 / k)
        
    def calculate_expected_surprise(self, action: int, available_cues: list):
        """
        Calculate expected surprise for a given action.
        
        Parameters:
        -----------
        action : int
            The cue to potentially select
        available_cues : list
            List of available cues
            
        Returns:
        --------
        float
            Expected surprise (entropy of predictive distribution)
        """
        if action not in available_cues:
            return np.inf  # Invalid action
        
        # Calculate predictive probabilities for both outcomes
        p_x1 = 0.0  # P(X=1 | action)
        p_x0 = 0.0  # P(X=0 | action)
        
        for z in range(self.k):
            if z == action:
                p_x1 += self.posterior[z] * self.p_t
                p_x0 += self.posterior[z] * (1.0 - self.p_t)
            else:
                p_x1 += self.posterior[z] * self.p_f
                p_x0 += self.posterior[z] * (1.0 - self.p_f)
        
        # Calculate entropy of predictive distribution
        entropy = 0.0
        if p_x1 > 0:
            entropy -= p_x1 * np.log(p_x1)
        if p_x0 > 0:
            entropy -= p_x0 * np.log(p_x0)
            
        return entropy
    
    def select_action(self, available_cues: list):
        """
        Select action using surprise-minimizing policy.
        
        Parameters:
        -----------
        available_cues : list
            List of available cues
            
        Returns:
        --------
        int
            Selected cue
        """
        if not available_cues:
            raise ValueError("No available cues")
        
        # Calculate expected surprise for each available action
        surprises = []
        for action in available_cues:
            surprise = self.calculate_expected_surprise(action, available_cues)
            surprises.append(surprise)
        
        surprises = np.array(surprises)
        
        # Softmax policy with negative surprise (minimize surprise)
        exp_values = np.exp(-self.beta * surprises)
        probabilities = exp_values / np.sum(exp_values)
        
        # Sample action according to probabilities
        return np.random.choice(available_cues, p=probabilities)
    
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


class RLAgent:
    """Reinforcement Learning agent with dual Q-functions for reward and surprise."""
    
    def __init__(self, k: int, p_t: float, p_f: float, alpha: float = 0.1, 
                 gamma: float = 0.9, beta: float = 1.0, lambda_surprise: float = 0.1,
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
        lambda_surprise : float
            Weight for surprise component in combined objective
        state_discretization : int
            Number of bins for discretizing belief states
        """
        self.k = k
        self.p_t = p_t
        self.p_f = p_f
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.lambda_surprise = lambda_surprise
        self.state_discretization = state_discretization
        
        # Initialize Q-tables
        # State space: discretized belief states
        # Action space: k possible cues
        self.q_reward = np.zeros((state_discretization ** k, k))
        self.q_surprise = np.zeros((state_discretization ** k, k))
        
        # Bayesian belief tracking
        self.posterior = np.full(k, 1.0 / k)
        self.previous_posterior = np.full(k, 1.0 / k)
        
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
        
        return min(state_index, self.q_reward.shape[0] - 1)
    
    def calculate_bayesian_surprise(self):
        """
        Calculate Bayesian surprise as KL divergence between current and previous posterior.
        
        Returns:
        --------
        float
            Bayesian surprise (KL divergence)
        """
        # KL divergence: sum(p * log(p/q))
        kl_div = 0.0
        for i in range(self.k):
            if self.posterior[i] > 1e-12 and self.previous_posterior[i] > 1e-12:
                kl_div += self.posterior[i] * np.log(self.posterior[i] / self.previous_posterior[i])
        return kl_div
    
    def select_action(self, available_cues: list, state_index: int = None):
        """
        Select action using combined Q-function and softmax policy.
        
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
        
        # Calculate combined Q-values for available actions
        q_values = []
        for action in available_cues:
            q_r = self.q_reward[state_index, action]
            q_s = self.q_surprise[state_index, action]
            combined_q = q_r + self.lambda_surprise * q_s
            q_values.append(combined_q)
        
        q_values = np.array(q_values)
        
        # Softmax action selection
        exp_values = np.exp(self.beta * q_values)
        probabilities = exp_values / np.sum(exp_values)
        
        # Sample action
        action_idx = np.random.choice(len(available_cues), p=probabilities)
        return available_cues[action_idx]
    
    def update_q_values(self, state: int, action: int, reward: float, 
                       surprise_reward: float, next_state: int, done: bool):
        """
        Update Q-values using TD learning.
        
        Parameters:
        -----------
        state : int
            Current state index
        action : int
            Action taken
        reward : float
            External reward received
        surprise_reward : float
            Surprise-based auxiliary reward
        next_state : int
            Next state index
        done : bool
            Whether episode is finished
        """
        # Update reward Q-function
        if done:
            target_r = reward
        else:
            target_r = reward + self.gamma * np.max(self.q_reward[next_state, :])
        
        td_error_r = target_r - self.q_reward[state, action]
        self.q_reward[state, action] += self.alpha * td_error_r
        
        # Update surprise Q-function
        if done:
            target_s = surprise_reward
        else:
            target_s = surprise_reward + self.gamma * np.max(self.q_surprise[next_state, :])
        
        td_error_s = target_s - self.q_surprise[state, action]
        self.q_surprise[state, action] += self.alpha * td_error_s
    
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
        self.previous_posterior = self.posterior.copy()
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
        bayesian_surprise = self.calculate_bayesian_surprise()
        
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
        self.previous_posterior = np.full(self.k, 1.0 / self.k)
