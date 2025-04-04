%pip install torch transformers accelerate

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, expansion_factor: float = 16):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = int(input_dim * expansion_factor)
        self.decoder = nn.Linear(self.latent_dim, input_dim, bias=True)
        self.encoder = nn.Linear(input_dim, self.latent_dim, bias=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = F.relu(self.encoder(x))
        decoded = self.decoder(encoded)
        return decoded, encoded

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return F.relu(self.encoder(x))

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.decoder(x)

    @classmethod
    def from_pretrained(cls, path: str, input_dim: int, expansion_factor: float = 16, device: str = "cuda") -> "SparseAutoencoder":
        model = cls(input_dim=input_dim, expansion_factor=expansion_factor)
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        return model



from huggingface_hub import hf_hub_download, notebook_login
sae_name = "DeepSeek-R1-Distill-Llama-8B-SAE-l19"
# sae_name = "Llama-3.2-1B-Instruct-SAE-l9"
# sae_name = "DeepSeek-R1-Distill-Llama-70B-SAE-l48"
if sae_name == "Llama-3.2-1B-Instruct-SAE-l9":
    notebook_login()
file_path = hf_hub_download(
    repo_id=f"qresearch/{sae_name}",
    filename=f"{sae_name}.pt",
    repo_type="model"
)

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B" if sae_name == "DeepSeek-R1-Distill-Llama-8B-SAE-l19"
              else "meta-llama/Llama-3.2-1B-Instruct" if sae_name == "Llama-3.2-1B-Instruct-SAE-l9"
              else "deepseek-ai/DeepSeek-R1-Distill-Llama-70B")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="bfloat16", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

expansion_factor = 8 if sae_name == "DeepSeek-R1-Distill-Llama-70B-SAE-l48" else 16
sae = SparseAutoencoder.from_pretrained(
    path=file_path,
    input_dim=model.config.hidden_size,
    expansion_factor=8,
    device="cuda"
)

inputs = tokenizer.apply_chat_template(
    [
        {"role": "user", "content": "Hello, how are you?"},
    ],
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")
outputs = model.generate(input_ids=inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))


def gather_residual_activations(model, target_layer, inputs):
    target_act = None
    def gather_target_act_hook(mod, inputs, outputs):
        nonlocal target_act
        target_act = inputs[0]  # Get residual stream from layer input
        return outputs

    handle = model.model.layers[target_layer].register_forward_hook(gather_target_act_hook)
    with torch.no_grad():
        _ = model(inputs)
    handle.remove()
    return target_act


    layer_id = (19 if sae_name == "DeepSeek-R1-Distill-Llama-8B-SAE-l19"
            else 9 if sae_name == "Llama-3.2-1B-Instruct-SAE-l9"
            else 48)

target_act = gather_residual_activations(model, layer_id, inputs)


def ensure_same_device(sae, target_act):
    """Ensure SAE and activations are on the same device"""
    model_device = target_act.device
    sae = sae.to(model_device)
    return sae, target_act.to(model_device)

sae, target_act = ensure_same_device(sae, target_act)
sae_acts = sae.encode(target_act.to(torch.float32))
recon = sae.decode(sae_acts)

var_explained = 1 - torch.mean((recon - target_act.to(torch.float32)) ** 2) / torch.var(target_act.to(torch.float32))
print(f"Variance explained: {var_explained:.3f}")

inputs = tokenizer.apply_chat_template(
    [
        {"role": "user", "content": "Roleplay as a pirate"},
        {"role": "assistant", "content": "Yarr, I'll be speakin' like a true seafarer from here on out! Got me sea legs ready and me vocabulary set to proper pirate speak. What can I help ye with, me hearty?"},
    ],
    return_tensors="pt",
).to("cuda")

import numpy as np
# Get activations
target_act = gather_residual_activations(model, layer_id, inputs)
sae_acts = sae.encode(target_act.to(torch.float32))

# Get token IDs and decode them for reference
tokens = inputs[0].cpu().numpy()
token_texts = tokenizer.convert_ids_to_tokens(tokens)

# Find which tokens are part of the assistant's response
token_ids = inputs[0].cpu().numpy()
is_special = (token_ids >= 128000) & (token_ids <= 128255)
special_positions = np.where(is_special)[0]

assistant_start = special_positions[-2] + 1
assistant_tokens = slice(assistant_start, None)

# Get activation statistics for assistant's response
assistant_activations = sae_acts[0, assistant_tokens]
mean_activations = assistant_activations.mean(dim=0)

# Find top activated features during pirate speech
num_top_features = 20
top_features = mean_activations.topk(num_top_features)

print("Top activated features during pirate speech:")
for idx, value in zip(top_features.indices, top_features.values):
    print(f"Feature {idx}: {value:.3f}")

# Look at how these features activate across different tokens
print("\nActivation patterns across tokens:")
for i, (token, acts) in enumerate(zip(token_texts[assistant_tokens], assistant_activations)):
    top_acts = acts[top_features.indices]
    if top_acts.max() > 0.2:  # Only show tokens with significant activation
        print(f"\nToken: {token}")
        for feat_idx, act_val in zip(top_features.indices, top_acts):
            if act_val > 0.2:  # Threshold for "active" features
                print(f"  Feature {feat_idx}: {act_val:.3f}")


def generate_with_intervention(
    model,
    tokenizer,
    sae,
    messages: list[dict],
    feature_idx: int,
    intervention: float = 3.0,
    target_layer: int = 9,
    max_new_tokens: int = 50
):
    modified_activations = None

    def intervention_hook(module, inputs, outputs):
        nonlocal modified_activations
        activations = inputs[0]

        features = sae.encode(activations.to(torch.float32))
        reconstructed = sae.decode(features)
        error = activations.to(torch.float32) - reconstructed

        features[:, :, feature_idx] += intervention

        modified = sae.decode(features) + error
        modified_activations = modified
        modified_activations = modified.to(torch.bfloat16)

        return outputs

    def output_hook(module, inputs, outputs):
        nonlocal modified_activations
        if modified_activations is not None:
            return (modified_activations,) + outputs[1:] if len(outputs) > 1 else (modified_activations,)
        return outputs

    handles = [
        model.model.layers[target_layer].register_forward_hook(intervention_hook),
        model.model.layers[target_layer].register_forward_hook(output_hook)
    ]

    try:
        input_tokens = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        outputs = model.generate(
            input_tokens,
            max_new_tokens=max_new_tokens,
            do_sample=False  # Use greedy decoding for consistency
        )

        generated_text = tokenizer.decode(outputs[0])

    finally:
        for handle in handles:
            handle.remove()

    return generated_text

messages = [
    {"role": "user", "content": "How are you doing?"}
]
feature_to_modify = 7560

print("Original generation:")
input_tokens = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)
outputs = model.generate(input_tokens, max_new_tokens=1000, do_sample=False)
print(tokenizer.decode(outputs[0]))

print("\nGeneration with modified feature:")
modified_text = generate_with_intervention(
    model=model,
    tokenizer=tokenizer,
    sae=sae,
    messages=messages,
    feature_idx=feature_to_modify,
    intervention=10,
    target_layer=layer_id,
    max_new_tokens=100
)
print(modified_text)



print("\nGeneration with modified feature:")

messages = [
    {"role": "user", "content": "How many Rs in strawberry?"}
]

modified_text = generate_with_intervention(
    model=model,
    tokenizer=tokenizer,
    sae=sae,
    messages=messages,
    feature_idx=feature_to_modify,
    intervention=8,
    target_layer=layer_id,
    max_new_tokens=1000
)
print(modified_text)