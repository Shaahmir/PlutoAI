import math
import torch
import torch.optim as optim

class WarmupCosineScheduler:

    def __init__(self, optimizer: optim.Optimizer, warmup_steps: int, total_steps: int, min_lr: float, max_lr: float):

        if warmup_steps < 0:
            raise ValueError("warmup_steps cannot be negative!")

        if total_steps <= 0:
            raise ValueError("total_steps must be positive!")
        
        if warmup_steps >= total_steps:
            raise ValueError("warmup_steps must be less than total_steps!")

        self.optimizer = optimizer
        
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

        self.min_lr = min_lr
        self.max_lr = max_lr

        self.step_count = 0
        self._set_lr(0.0)

    def _set_lr(self, learning_rate: float):

        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
    
    def get_lr(self) -> float:

        if self.step_count < self.warmup_steps:
            return self.max_lr * self.step_count / self.warmup_steps

        decay_steps = self.total_steps - self.warmup_steps

        if decay_steps <= 0:
            return self.min_lr

        progress = (self.step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = min(max(progress, 0.0), 1.0)

        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

        return self.min_lr + (self.max_lr - self.min_lr) * cosine

    def step(self) -> float:

        self.step_count += 1
        learning_rate = self.get_lr()
        self._set_lr(learning_rate)

        return learning_rate

    def state_dict(self) -> dict:
        return {
            "step_count": self.step_count
        }

    def load_state_dict(self, state_dict: dict):

        self.step_count = int(state_dict["step_count"])
        self._set_lr(
            self.get_lr()
        )
