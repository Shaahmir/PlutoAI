import os
import json
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

def unwrap_model(model):
    if hasattr(model, "module"):
        return model.module

    return model

def get_learning_rate(optimizer: optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])

def parameter_dtype(model: nn.Module) -> torch.dtype:
    
    for parameter in model.parameters():
        return parameter.dtype
    
    return torch.float32

def save_rng_state() -> dict:

    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state()
    }

    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()

    return state

def restore_rng_state(state: dict):

    if not state:
        return
    
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])

    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])

def all_reduce_sum(value: torch.Tensor) -> torch.Tensor:

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(
            value,
            op = dist.ReduceOp.SUM
        )

    return value

def count_valid_tokens(labels: torch.Tensor) -> int:
    return int((labels != -100).sum().item())

def append_jsonl(path, record: dict):
    
    with open(path, "a", encoding = "utf-8") as f:
        f.write(
            json.dumps(
                record,
                ensure_ascii = False
            )
        )

        f.write("\n")
