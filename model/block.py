import torch
import torch.nn as nn

from config import CONFIG
from model.attention import GQA
from model.mlp import SwiGLU
from model.rmsnorm import RMSNorm

class TransformerBlock(nn.Module):

    def __init__(self):

        super().__init__()

        self.input_norm = RMSNorm(
            CONFIG.D_MODEL
        )

        self.attention = GQA()

        self.post_attention_norm = RMSNorm(
            CONFIG.D_MODEL
        )

        self.mlp = SwiGLU()

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:

        x = x + self.attention(
            self.input_norm(x),
            attention_mask = attention_mask
        )

        x = x + self.mlp(
            self.post_attention_norm(x)
        )

        return x
