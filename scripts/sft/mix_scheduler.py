import random
from dataclasses import dataclass

@dataclass(frozen = True)
class DatasetBudget:
    name: str
    target_tokens: int

class TokenMixScheduler:

    def __init__(self, total_tokens: int, ratios: dict[str, float], seed: int):

        if total_tokens <= 0:
            raise ValueError("total_tokens must be positive!")

        if not ratios:
            raise ValueError("ratios cannot be empty!")
            
        if any(value < 0.0 for value in ratios.values()):
            raise ValueError(f"Negative dataset ratios: {ratios}")

        self.total_tokens = total_tokens
        self.ratios = dict(ratios)
        self.random = random.Random(seed)
        self.budgets = {
            name: int(total_tokens * ratio) for name, ratio in ratios.items()
        }

        allocated = sum(self.budgets.values())
        remainder = total_tokens - allocated

        largest_name = max(self.budgets, key = self.budgets.get)

        self.budgets[largest_name] += remainder

    def get_budget(self, name: str) -> int:

        if name not in self.budgets:
            raise KeyError(name)

        return self.budgets[name]

    def all_budgets(self) -> dict[str, int]:
        return dict(self.budgets)

    def next_score(self, remaining: dict[str, int]) -> str | None:

        available = [name for name, tokens in remaining.items() if tokens > 0]

        if not available:
            return None

        weights = [remaining[name] for name in available]

        return self.random.choices(
            available,
            weights = weights,
            k = 1
        )[0]
