#%% download model files

from huggingface_hub import snapshot_download

repo_id = "marcelbinz/Llama-3.1-Centaur-8B"
local_folder = "./llama_centaur_adapter"
snapshot_download(repo_id=repo_id, local_dir=local_folder)
print(f"Model files downloaded to: {local_folder}")
# %%