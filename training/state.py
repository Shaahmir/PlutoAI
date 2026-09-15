from dataclasses import dataclass

@dataclass
class TrainingState:

    GLOBAL_STEP: int = 0
    EPOCH: int = 0
    BATCH_IN_EPOCH: int = 0
    OPTIMIZER_STEP: int = 0
    SEEN_TOKENS: int = 0
    BEST_VALID_LOSS: float | None = None
