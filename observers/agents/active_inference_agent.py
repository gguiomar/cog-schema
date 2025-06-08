import numpy as np

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
        """Make final decision using entropy minimization."""
        posterior = learned_posterior if learned_posterior is not None else self.posterior
        return self._entropy_minimizing_decision(posterior)
    
    def _entropy_minimizing_decision(self, posterior):
        """
        Select decision that minimizes expected entropy.
        
        For each possible decision d, calculate the expected entropy
        if that decision were chosen, then select the decision with
        minimum expected entropy.
        
        Parameters:
        -----------
        posterior : np.array
            Current posterior distribution
            
        Returns:
        --------
        int
            Decision that minimizes expected entropy
        """
        min_entropy = float('inf')
        best_decision = 0
        
        for decision in range(self.k):
            # Calculate expected entropy if this decision were chosen
            expected_entropy = self._calculate_expected_entropy_for_decision(decision, posterior)
            
            if expected_entropy < min_entropy:
                min_entropy = expected_entropy
                best_decision = decision
        
        return best_decision
    
    def _calculate_expected_entropy_for_decision(self, decision, posterior):
        """
        Calculate expected entropy if a particular decision were chosen.
        
        This represents the expected uncertainty remaining after making
        this decision, weighted by the probability that this decision
        is correct given the current posterior.
        
        Parameters:
        -----------
        decision : int
            The decision (location) being considered
        posterior : np.array
            Current posterior distribution
            
        Returns:
        --------
        float
            Expected entropy for this decision
        """
        # The expected entropy is the entropy of the posterior weighted by
        # the probability that this decision is correct
        
        # If we choose this decision, the "surprise" or entropy depends on
        # how confident we are in this choice
        decision_confidence = posterior[decision]
        
        # Calculate entropy of the posterior distribution
        entropy = -np.sum(posterior * np.log(posterior + 1e-12))
        
        # Weight the entropy by our uncertainty about this decision
        # Higher confidence (higher posterior[decision]) should lead to lower expected entropy
        expected_entropy = entropy * (1.0 - decision_confidence)
        
        return expected_entropy
    
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
