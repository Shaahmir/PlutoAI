import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CONFIG
from model.embeddings import TokenEmbedding
from model.transformer import Transformer

class GPT(nn.Module):

    def __init__(self):

        super().__init__()
        
        self.config = CONFIG
        self.token_embedding = TokenEmbedding()
        self.transformer = Transformer()

        self.lm_head = nn.Linear(
            CONFIG.D_MODEL,
            CONFIG.VOCAB_SIZE,
            bias = CONFIG.USE_BIAS
        )   

        if CONFIG.USE_WEIGHT_TYING:
            self.lm_head.weight = self.token_embedding.embedding.weight

        self.apply(
            self._init_weight
        )

        if CONFIG.USE_WEIGHT_TYING:
            self._reset_tied_weight()

    def _init_weight(self, module: nn.Module):

        if isinstance(module, nn.Linear):

            std = 0.02

            if module is not self.lm_head:
                if module.out_features == CONFIG.D_MODEL:
                    std = 0.02 / math.sqrt(2 * CONFIG.N_LAYERS)

            nn.init.normal_(
                module.weight,
                mean = 0.0,
                std = std
            )

            if module.bias is not None:

                nn.init.zeros_(
                    module.bias
                )

        elif isinstance(module, nn.Embedding):

            nn.init.normal_(
                module.weight,
                mean = 0.0,
                std = 0.02
            )

    def _reset_tied_weight(self):
        self.lm_head.weight = self.token_embedding.embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None

    ) -> tuple[torch.Tensor, torch.Tensor | None]:

        x = self.token_embedding(
            input_ids
        )

        x = self.transformer(
            x,
            attention_mask = attention_mask
        )

        logits = self.lm_head(x)
        loss = None

        if labels is not None:

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index = -100
            )

        return logits, loss
