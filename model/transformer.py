import torch
import torch.nn as nn

from config import CONFIG
from model.block import TransformerBlock
from model.rmsnorm import RMSNorm

class Transformer(nn.Module):

    def __init__(self):

        super().__init__()

        self.layers = nn.ModuleList(
            [TransformerBlock() for _ in range(CONFIG.N_LAYERS)]
        )

        self.final_norm = RMSNorm(
            CONFIG.D_MODEL
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:

        prepared_mask = attention_mask

        if attention_mask is not None:

            if attention_mask.ndim != 2 or attention_mask.shape != x.shape[:2]:
                raise ValueError(f"attention_mask must be [B, S] = {tuple(x.shape[:2])} got {tuple(attention_mask.shape)}")

            sequence_length = x.shape[1]
            causal = torch.tril(
                torch.ones(
                    (sequence_length, sequence_length),
                    dtype = torch.bool,
                    device = x.device
                )
            )

            prepared_mask = causal[None, None, :, :] & attention_mask[:, None, None, :].bool()

        for layer in self.layers:
            x = layer(
                x,
                attention_mask = prepared_mask
            )
        
        return self.final_norm(x)
