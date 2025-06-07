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
