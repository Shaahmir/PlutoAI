import torch
import torch.nn as nn

from config import CONFIG

class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6):

        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(
            torch.ones(dim)
        )

    def forward(self, x: torch.Tensor):
        
        input_dtype = x.dtype
        x_float = x.float()
        
        variance = x_float.pow(2).mean(
            dim = -1,
            keepdim = True
        )

        x_normalised = x_float * torch.rsqrt(
            variance + self.eps
        )

        return (
            self.weight * x_normalised
        ).to(input_dtype)