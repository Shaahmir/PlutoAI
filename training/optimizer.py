import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import optimizer

def build_optimizer(model: nn.Module, learning_rate: float, weight_decay: float, betas: tuple[float, float], eps: float) -> optim.Optimizer:

    decay_parameters = []
    no_decay_parameters = []

    for _, parameter in model.named_parameters():

        if not parameter.requires_grad:
            continue

        if parameter.ndim >= 2:
            decay_parameters.append(parameter)

        else:
            no_decay_parameters.append(parameter)

        
    parameter_groups = [
        {
            "params": decay_parameters,
            "weight_decay": weight_decay
        },
        {
            "params": no_decay_parameters,
            "weight_decay": 0.0
        }
    ]

    optimizer = optim.AdamW(
        parameter_groups,
        lr = learning_rate,
        betas = betas,
        eps = eps,
        fused = torch.cuda.is_available()
    )

    return optimizer
