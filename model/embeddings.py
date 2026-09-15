import torch
import torch.nn as nn
from config import CONFIG

class TokenEmbedding(nn.Module):

    def __init__(self):

        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings = CONFIG.VOCAB_SIZE,
            embedding_dim = CONFIG.D_MODEL
        )

        self.reset_parameters()

    def reset_parameters(self):

        nn.init.normal_(
            self.embedding.weight,
            mean = 0.0,
            std = 0.02
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:

        return self.embedding(
            input_ids
        )