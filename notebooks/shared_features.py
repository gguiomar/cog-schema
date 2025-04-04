import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

# Load SAE (replace with your SAE implementation)
class SparseAutoencoder(torch.nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = torch.nn.Linear(input_dim, latent_dim)
        self.decoder = torch.nn.Linear(latent_dim, input_dim)

    def encode(self, x):
        return F.relu(self.encoder(x))

    def decode(self, x):
        return self.decoder(x)

sae = SparseAutoencoder(input_dim=4096, latent_dim=65536).to(model.device)

# Experiment 1: Get token activations
def get_token_activations(token: str, num_samples: int = 10):
    token_id = tokenizer.convert_tokens_to_ids(token)
    samples = [f"This is a test with {token}.", f"The letter {token} appears here."]
    
    all_activations = []
    for text in samples:
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            hidden_states = model(**inputs, output_hidden_states=True).hidden_states[-1]  # Last layer
            activations = sae.encode(hidden_states.float())
        all_activations.append(activations)
    
    return torch.cat(all_activations).mean(dim=0)

# Experiment 2: Visualize high-dimensional features
def visualize_features(activations, labels):
    tsne = TSNE(n_components=2, perplexity=15)
    reduced = tsne.fit_transform(activations.cpu().numpy())
    plt.scatter(reduced[:, 0], reduced[:, 1], c=labels)
    plt.title("t-SNE of Token Activations")
    plt.show()

# Experiment 3: Shared feature space analysis
def feature_overlap(a, b, threshold=0.5):
    a_features = (a > threshold).nonzero().flatten()
    b_features = (b > threshold).nonzero().flatten()
    intersection = len(np.intersect1d(a_features, b_features))
    return intersection / len(a_features)

def plot_feature_clustering(activations, labels):
    Z = linkage(activations.cpu().numpy(), 'ward')
    dendrogram(Z, labels=labels)
    plt.title("Hierarchical Clustering of Token Features")
    plt.show()

# Experiment 4: Intervention experiments
def generate_with_intervention(token: str, feature_idx: int, intervention: float = 5.0):
    inputs = tokenizer(f"This is a test with {token}.", return_tensors="pt").to(model.device)
    with torch.no_grad():
        hidden_states = model(**inputs, output_hidden_states=True).hidden_states[-1]
        activations = sae.encode(hidden_states.float())
        activations[:, :, feature_idx] += intervention
        modified_hidden = sae.decode(activations)
        outputs = model.generate(inputs.input_ids, hidden_states=modified_hidden, max_new_tokens=20)
    return tokenizer.decode(outputs[0])

# Main script
if __name__ == "__main__":
    # Get activations for tokens
    tokens = ["ĠK", "ĠL", "ĠM"]
    activations = [get_token_activations(token) for token in tokens]
    labels = [i for i, token in enumerate(tokens) for _ in range(len(activations[0]))]

    # Experiment 1: Print top features for each token
    for token, act in zip(tokens, activations):
        top_features = act.topk(5)
        print(f"Top features for {token}: {top_features.indices.tolist()}")

    # Experiment 2: Visualize features with t-SNE
    all_activations = torch.stack(activations)
    visualize_features(all_activations, labels=[0, 1, 2])

    # Experiment 3: Shared feature analysis
    print(f"K-L feature overlap: {feature_overlap(activations[0], activations[1]):.2f}")
    print(f"K-M feature overlap: {feature_overlap(activations[0], activations[2]):.2f}")
    plot_feature_clustering(all_activations, tokens)

    # Experiment 4: Intervention on top feature of "K"
    top_k_feature = activations[0].topk(1).indices.item()
    print("\nIntervention on top K feature:")
    print(generate_with_intervention("K", top_k_feature, intervention=5.0))