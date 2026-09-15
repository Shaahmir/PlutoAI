import torch
import torch.nn as nn
from pathlib import Path

def load_pretrain_weights(model: nn.Module, checkpoint_path: Path, device: torch.device):

    if not checkpoint_path.exists():
        raise FileNotFoundError("Pretrain Checkpoint not found!")

    checkpoint = torch.load(
        checkpoint_path,
        map_location = device,
        weights_only = False
    )

    if "model" not in checkpoint:
        raise KeyError("Checkpoint does not contain a 'model' state")

    state_dict = checkpoint["model"]

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[len("module."):]

        cleaned_state_dict[key] = value

    missing, unexpected = model.load_state_dict(
        cleaned_state_dict,
        strict = True
    )

    if missing:
        raise RuntimeError(f"Missing model keys: {missing}")

    if unexpected:
        raise RuntimeError(f"Unexpected model keys: {unexpected}")

    print(f"Loaded pretrain weights: {checkpoint_path}")
