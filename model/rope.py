from __future__ import annotations

import torch
import torch.nn as nn

class RotaryEmbedding(nn.Module):

    def __init__(self, head_dim: int, max_position_embeddings: int, theta: float =  10_000.0):

        super().__init__()

        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE!")

        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.theta = theta

        inverse_frequency = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype = torch.float32) / head_dim))
        #Alternative formula: theta ** (-torch.arange(0, head_dim, 2, dtype = torch.float32) / head_dim)

        self.register_buffer(
            "inverse_frequency",
            inverse_frequency,
            persistent = False
        )
        
        self._build_cache(max_position_embeddings)

    def _build_cache(self, sequence_length: int, device: torch.device | None = None):

        if device is None:
            device = self.inverse_frequency.device

        positions = torch.arange(
            sequence_length,
            device = device,
            dtype = torch.float32
        )

        frequencies = torch.outer(
            positions,
            self.inverse_frequency.to(device)
        )

        cos = frequencies.cos()
        sin = frequencies.sin()

        cos = torch.repeat_interleave(
            cos,
            repeats = 2,
            dim = -1
        )

        sin = torch.repeat_interleave(
            sin,
            repeats = 2,
            dim = -1
        )

        self.register_buffer(
            "cos_cached",
            cos[None, None, :, :],
            persistent = False
        )

        self.register_buffer(
            "sin_cached",
            sin[None, None, :, :],
            persistent = False
        )

    def _rotary_half(self, x: torch.Tensor) -> torch.Tensor:

        x1 = x[..., ::2]
        x2 = x[..., 1::2]

        return torch.stack(
            (-x2, x1),
            dim = -1
        ).flatten(-2)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        sequence_length = q.shape[-2]

        if sequence_length > self.cos_cached.shape[-2]:
            self._build_cache(
                sequence_length,
                q.device
            )

        cos = self.cos_cached[..., :sequence_length, :].to(
            device = q.device,
            dtype = q.dtype
        )

        sin = self.sin_cached[..., :sequence_length, :].to(
            device = q.device,
            dtype = q.dtype
        )

        q = q * cos + self._rotary_half(q) * sin
        k = k * cos[..., :k.shape[-2], :] + self._rotary_half(k) * sin[..., :k.shape[-2], :]

        return q, k
