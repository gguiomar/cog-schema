#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp

rng = np.random.default_rng(0)
K, bias_true, bias_false, n_trials = 4, 0.9, 0.5, 500
rounds = np.arange(2, 13)
mean_post, mean_entropy = [], []

for n in rounds:
    posts, ents = [], []
    for _ in range(n_trials):
        true_h = rng.integers(K)
        lp = np.full(K, -np.log(K))
        for _ in range(n):
            cue = rng.integers(K)
            p = bias_true if cue == true_h else bias_false
            color = rng.random() < p
            ll = np.log(p if color else 1 - p)
            lp += np.where(np.arange(K) == cue, ll, np.log(bias_false if color else 1 - bias_false))
            lp -= logsumexp(lp)
        p_post = np.exp(lp)
        posts.append(p_post[true_h])
        ents.append(-np.sum(p_post * lp))
    mean_post.append(np.mean(posts))
    mean_entropy.append(np.mean(ents))

plt.plot(rounds, mean_post, 'o-')
plt.xlabel('Rounds'); plt.ylabel('Mean Posterior'); plt.title('Posterior vs Rounds'); plt.grid()
plt.figure()
plt.plot(rounds, mean_entropy, 'o-')
plt.xlabel('Rounds'); plt.ylabel('Mean Entropy'); plt.title('Entropy vs Rounds'); plt.grid()
plt.show()

#%%