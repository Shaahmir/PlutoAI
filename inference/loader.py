import torch
from pathlib import Path
from model.gpt import GPT
from tokenizer.tokenizer import BPETokenizer

def load_model(checkpoint_path: Path, device: torch.device) -> GPT:

    model = GPT().to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location = device,
        weights_only = False
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    model.eval()

    return model

def load_tokenizer(tokenizer_path) -> BPETokenizer:
    return BPETokenizer(
        tokenizer_path
    )
