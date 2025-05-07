import wandb
import torch
from functools import partial
import os
import json
from tqdm import tqdm

def init_wandb(wandb_cfg, cfg):
    return wandb.init(entity="gibbon-sae", project=wandb_cfg["project"], name=wandb_cfg["name"], config=cfg, reinit=True)

def log_batch_wandb(output, wandb_run):
    metrics_to_log = ["loss", "l2_loss", "l1_loss", "l0_norm", "l1_norm", "aux_loss", "num_dead_features"]
    log_dict = {k: output[k].item() for k in metrics_to_log if k in output}
    log_dict["n_dead_in_batch"] = (output["feature_acts"].sum(0) == 0).sum().item()

    wandb_run.log(log_dict)

def save_checkpoint(wandb_run, sae, cfg, wandb_cfg, epoch):
    save_dir = f"checkpoints/{wandb_cfg['name']}_{epoch}"
    os.makedirs(save_dir, exist_ok=True)

    # Save model state
    sae_path = os.path.join(save_dir, "sae.pt")
    torch.save(sae.state_dict(), sae_path)

    # Prepare config for JSON serialization
    json_safe_cfg = {}
    for key, value in cfg.items():
        if isinstance(value, (int, float, str, bool, type(None))):
            json_safe_cfg[key] = value
        elif isinstance(value, (torch.dtype, type)):
            json_safe_cfg[key] = str(value)
        else:
            json_safe_cfg[key] = str(value)

    # Save config
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(json_safe_cfg, f, indent=4)

    # Create and log artifact
    artifact = wandb.Artifact(
        name=f"{wandb_cfg['name']}_{epoch}",
        type="model",
        description=f"Model checkpoint at epoch {epoch}",
    )
    artifact.add_file(sae_path)
    artifact.add_file(config_path)
    wandb_run.log_artifact(artifact)

    tqdm.write(f"Model and config saved as artifact at epoch {epoch}")

def load_checkpoint(run_id, checkpoint_name, device="cuda"):
    api = wandb.Api()
    wandb_run = api.run(f"gibbon-sae/SAE training/{run_id}")
    artifact = wandb_run.use_artifact(checkpoint_name, type='model')
    artifact_dir = artifact.download()
    model_path = os.path.join(artifact_dir, "sae.pt")
    config_path = os.path.join(artifact_dir, "config.json")

    with open(config_path, "r") as f:
        cfg = json.load(f)

    model = torch.load(model_path, map_location=device)
    return model, cfg