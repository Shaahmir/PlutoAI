import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CONFIG
from model.rope import RotaryEmbedding

class GQA(nn.Module):

    def __init__(self):
        
        super().__init__()

        self.d_model = CONFIG.D_MODEL
        self.n_heads = CONFIG.N_HEADS
        self.n_kv_heads = CONFIG.N_KV_HEADS
        self.head_dim = CONFIG.HEAD_DIM
        self.n_rep = CONFIG.N_REP
        self.dropout = CONFIG.DROPOUT

        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")

        self.q_proj = nn.Linear(
            CONFIG.D_MODEL,
            CONFIG.N_HEADS * CONFIG.HEAD_DIM,
            bias = CONFIG.USE_BIAS
        )

        self.k_proj = nn.Linear(
            CONFIG.D_MODEL,
            CONFIG.N_KV_HEADS * CONFIG.HEAD_DIM,
            bias = CONFIG.USE_BIAS
        )

        self.v_proj = nn.Linear(
            CONFIG.D_MODEL,
            CONFIG.N_KV_HEADS * CONFIG.HEAD_DIM,
            bias = CONFIG.USE_BIAS
        )

        self.o_proj = nn.Linear(
            CONFIG.N_HEADS * CONFIG.HEAD_DIM,
            CONFIG.D_MODEL,
            bias = CONFIG.USE_BIAS
        )

        self.rope = RotaryEmbedding(
            head_dim = CONFIG.HEAD_DIM,
            max_position_embeddings = CONFIG.CONTEXT_LENGTH,
            theta = CONFIG.ROPE_THETA
        )

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:

        if self.n_rep == 1:
            return x

        batch_size = x.shape[0]
        n_kv_heads = x.shape[1]
        sequence_length = x.shape[2]
        head_dim = x.shape[3]

        x = x.unsqueeze(2)
        x = x.expand(
            batch_size,
            n_kv_heads,
            self.n_rep,
            sequence_length,
            head_dim
        )

        return x.reshape(
            batch_size,
            self.n_kv_heads * self.n_rep,
            sequence_length,
            head_dim
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:

        batch_size, sequence_length, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(
            batch_size,
            sequence_length,
            self.n_heads,
            self.head_dim
        )

        k = k.view(
            batch_size,
            sequence_length,
            self.n_kv_heads,
            self.head_dim
        )

        v = v.view(
            batch_size,
            sequence_length,
            self.n_kv_heads,
            self.head_dim
        )

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q, k = self.rope(
            q, k
        )

        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        if attention_mask is None:

            attention_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask = None,
                dropout_p = self.dropout if self.training else 0.0,
                is_causal = True
            )

        else:
            if attention_mask.ndim == 2:

                causal = torch.tril(
                    torch.ones(
                        (sequence_length, sequence_length),
                        dtype = torch.bool,
                        device = x.device
                    )
                )

                allowed = causal[None, None, :, :] & attention_mask[:, None, None, :].bool()

            elif attention_mask.ndim == 4 and attention_mask.shape[0] == batch_size and attention_mask.shape[-2:] == (sequence_length, sequence_length):
                allowed = attention_mask.bool()

            else:
                raise ValueError(f"attention_mask must be [B, T] or prepared [B, 1, T, T] got {tuple(attention_mask.shape)}!")

            attention_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask = allowed,
                dropout_p = self.dropout if self.training else 0.0,
                is_causal = False
            )

        attention_output = attention_output.transpose(1, 2)
        attention_output = attention_output.contiguous()
        
        attention_output = attention_output.view(
            batch_size,
            sequence_length,
            self.n_heads * self.head_dim
        )

        return self.o_proj(attention_output)
