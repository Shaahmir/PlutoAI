import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CONFIG

class SwiGLU(nn.Module):

    def __init__(self):
        
        super().__init__()

        self.gate_proj = nn.Linear(
            CONFIG.D_MODEL,
            CONFIG.D_FF,
            bias = CONFIG.USE_BIAS
        )

        self.up_proj = nn.Linear(
            CONFIG.D_MODEL,
            CONFIG.D_FF,
            bias = CONFIG.USE_BIAS
        )

        self.down_proj = nn.Linear(
            CONFIG.D_FF,
            CONFIG.D_MODEL,
            bias = CONFIG.USE_BIAS
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)

        return self.down_proj(
            gate * up
        )
