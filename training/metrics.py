import math
import time
import torch

class TrainingMetrics: 

    def __init__(self):

        self.loss_sum = 0.0
        self.tokens = 0

        self.start_time = time.perf_counter()

    def update(self, loss: torch.Tensor, token_count: int):

        self.loss_sum += loss.detach().float().item() * token_count
        self.tokens += token_count

    @property
    def mean_loss(self) -> float:

        if self.tokens == 0:
            return 0.0

        return self.loss_sum / self.tokens

    @property
    def perplexity(self) -> float:

        loss = self.mean_loss

        if loss > 20:
            return float("inf")

        return math.exp(loss)

    def tokens_per_second(self) -> float:

        elapsed_time = time.perf_counter() - self.start_time

        if elapsed_time <= 0:
            return 0.0

        return self.tokens / elapsed_time

    def reset(self):

        self.loss_sum = 0.0
        self.tokens = 0

        self.start_time = time.perf_counter()
