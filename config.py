import torch
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen = True)
class Config:

    # Paths Configurations

    ROOT: Path = Path(__file__).resolve().parent

    # Raw Paths

    RAW_DATA: Path = ROOT / "data" / "raw"
    RAW_PRETRAIN_DIR: Path = RAW_DATA / "pretrain"
    RAW_SFT_DIR: Path = RAW_DATA / "sft"

    # Tokenizer Path

    TOKENIZER_DIR: Path = ROOT / "data" / "tokenizer"

    # Processed Paths

    PROCESSED_DATA: Path = ROOT / "data" / "processed"
    TRAIN_DIR: Path = PROCESSED_DATA / "train"
    VALID_DIR: Path = PROCESSED_DATA / "valid"

    TRAIN_PRETRAIN_DIR: Path = TRAIN_DIR / "pretrain"
    TRAIN_SFT_DIR: Path = TRAIN_DIR / "sft"
    TRAIN_GRPO_DIR: Path = TRAIN_DIR / "grpo"

    VALID_PRETRAIN_DIR: Path = VALID_DIR / "pretrain"
    VALID_SFT_DIR: Path = VALID_DIR / "sft"
    VALID_GRPO_DIR: Path = VALID_DIR / "grpo"

    # Other Paths

    CHECKPOINT_DIR: Path = ROOT / "checkpoints"
    LOG_DIR = ROOT / "logs"

    PRETRAIN_CHECKPOINT_DIR: Path = CHECKPOINT_DIR / "pretrain"
    SFT_CHECKPOINT_DIR: Path = CHECKPOINT_DIR / "sft"
    GRPO_CHECKPOINT_DIR: Path = CHECKPOINT_DIR / "grpo"
    GRPO_SFT_CHECKPOINT: Path = SFT_CHECKPOINT_DIR / "best.pt"

    PRETRAIN_LOG_DIR: Path =  LOG_DIR / "pretrain"
    SFT_LOG_DIR: Path =  LOG_DIR / "sft"
    GRPO_LOG_DIR: Path = LOG_DIR / "grpo"

    # Modes Configurations

    DATASET_MODE: str = "pretrain"
    DATASET_MODES: tuple = ("pretrain", "sft", "grpo")

    # Validation Configurations

    VALIDATION_SPLIT: float = 0.01
    SFT_VALIDATION_SPLIT: float = 0.10

    # Pretrain Configurations

    PRETRAIN_MIN_SCORE: float = 2.5
    PRETRAIN_MIN_TOKEN_COUNT: int = 128
    PRETRAIN_MIN_LANG_SCORE: float = 0.85

    # Device Configurations

    DEVICE: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    NUM_GPUS: int = torch.cuda.device_count()
    USE_CUDA: bool = torch.cuda.is_available()
    USE_DDP: bool = USE_CUDA and NUM_GPUS > 1

    # Tokenizer Configurations

    VOCAB_SIZE: int = 32000

    SPECIAL_TAGS = {
        "PAD": "<|pad|>",
        "UNK": "<|unk|>",
        "BOS": "<|bos|>",
        "EOS": "<|eos|>",
        "START_HEAD": "<|start_header_id|>",
        "END_HEAD": "<|end_header_id|>",
        "EOT": "<|end_turn|>",
        "THINK": "<|think|>",
        "END_THINK": "<|end_think|>"
    }

    SPECIAL_TOKENS: tuple = tuple(SPECIAL_TAGS.values())


    TOKENIZER_NAME: str = "pluto_ai_bpe"
    TOKENIZER_FILE: str = "tokenizer.json"
    TOKENIZER_PATH: Path = TOKENIZER_DIR / TOKENIZER_FILE

    # Sequence Configurations
    
    CONTEXT_LENGTH: int = 1024
    RESERVED_TOKENS: int = len(SPECIAL_TOKENS)

    # Model Configurations

    N_LAYERS: int = 12
    D_MODEL: int = 768
    N_HEADS: int = 12
    N_KV_HEADS: int = 4
    D_FF: int = 2944

    # D_FF Calculation:
    # Modern SwiGLU architectures adjust the traditional 4x hidden multiplier 
    # down to ~8/3 * d_model to keep parameter counts balanced. 
    # 2944 is chosen because it scales up from the base ratio and is a clean 
    # multiple of 128 (2944 / 128 = 23) for optimal GPU tensor core efficiency.
    
    DROPOUT: float = 0.0
    USE_BIAS: bool = False
    USE_WEIGHT_TYING: bool = True

    # Positional Encoding Configurations

    ROPE_THETA: float = 10_000.0

    # Attention Configurations

    USE_FLASH_ATTENTION: bool = True

    # Training Configurations

    BATCH_SIZE: int = 4
    GRADIENT_ACCUMULATION_STEPS: int = 8
    MAX_GRAD_NORM: float = 1.0
    NUM_EPOCHS: int = 1
    MAX_TRAIN_STEPS: int | None = None
    VALIDATION_STEPS: int | None = None
    ACTIVATION_FUNCTION: str = "swiglu"

    # Optimizer Configurations

    LEARNING_RATE: float = 3e-4
    MIN_LEARNING_RATE: float = 3e-5
    WEIGHT_DECAY: float = 0.1
    ADAM_BETAS: tuple = (0.9, 0.95)
    ADAM_EPS: float = 1e-8

    # Scheduler Configurations

    WARMUP_STEPS: int = 2000
    LR_SCHEDULER: str = "cosine"

    # Mixed Precision Configurations

    USE_AMP: bool = USE_CUDA
    AMP_DTYPE: torch.dtype = torch.float16

    # Memory Configurations

    USE_GRADIENT_CHECKPOINTING: bool = False

    # Dataloader Configurations

    NUM_WORKERS: int = 4
    PIN_MEMORY: bool = USE_CUDA
    PERSISTENT_WORKERS: bool = True
    PREFETCH_FACTOR: int = 2
    DROP_LAST: bool = True

    # Dataset Preprocessing Configurations

    # TOKENS_PER_SHARD: int = 50_000_000 / 1_024 (context Length) = 48_828.125 -> 48_828 x _024 = 49_999_872
    TOKENS_PER_SHARD: int = 49_999_872
    SEQUENCE_PACKING: bool = True
    STORE_LOSS_MASK: bool = True
    TOKEN_DTYPE: str = "uint16"
    LOSS_MASK_DTYPE: str = "uint8"
    INDEX_OFFSET_DTYPE: str = "uint64"
    INDEX_LENGTH_DTYPE: str = "uint32"
    MIN_TURN_COUNT: int = 1
    MAX_TURN_COUNT: int | None = None

    # Checkpoint Configurations

    SAVE_EVERY_STEPS: int = 1000
    SAVE_EVERY_EPOCHS: int = 1
    KEEP_LAST_N_CHECKPOINTS: int = 2
    
    LATEST_CHECKPOINT: Path = CHECKPOINT_DIR / DATASET_MODE / "latest.pt"
    BEST_CHECKPOINT: Path = CHECKPOINT_DIR / DATASET_MODE / "best.pt"

    RESUME_FROM_CHECKPOINTS: Path = None
    # RESUME_FROM_CHECKPOINTS: Path = CHECKPOINT_DIR / "pretrain" / "latest.pt"
    BASE_MODEL_CHECKPOINTS: Path = CHECKPOINT_DIR / "pretrain" / "best.pt"

    # Logging Configurations

    LOG_EVERY_STEPS: int = 10
    LOG_TOKEN_THROUGHPUT: bool = True
    EVAL_EVERY_STEPS: int = 500
    TENSORBOARD_DIR: Path = LOG_DIR / "tensorboard"

    # Training Runtime Configurations

    SYNC_DDP_EVERY_STEP: bool = True
    USE_TF32: bool = False

    # SFT Configurations

    SFT_D1_EXAMPLES: int = 100_000

    SFT_D1_TOKEN_RATIO: float = 0.50
    SFT_D2_TOKEN_RATIO: float = 0.40
    SFT_REPLAY_TOKEN_RATIO: float = 0.10
    
    SFT_TOKENS: int = 100_000_000
    SFT_VALID_TOKENS: int = 10_000_000

    SFT_EPOCHS: int = 3
    SFT_BATCH_SIZE: int = 4
    SFT_LR: float = 1e-5
    SFT_MIN_LR: float = 1e-6
    SFT_WARMUP_STEPS: int = 0

    SFT_EVAL_EVERY_STEPS: int = 250
    SFT_SAVE_EVERY_STEPS: int = 500
    SFT_KEEP_LAST_N: int = 3

    SFT_GRADIENT_ACCUMULATION_STEPS: int = 4
    SFT_WEIGHT_DECAY: float = 0.1
    SFT_ADAM_BETAS: tuple = (0.9, 0.95)
    SFT_EPS: float = 1e-8
    SFT_GRAD_CLIP: float = 1.0

    SFT_REPLAY_VALID: int = 64

    # Inference Configurations

    TEMPERATURE: float = 0.8
    TOP_K: int = 50
    TOP_P: float = 0.95
    MIN_P: float = 0.05
    REPETITION_PENALTY: float = 1.2
    FREQUENCY_PENALTY: float = 0.0
    PRESENCE_PENALTY: float = 0.0
    REPETITION_WINDOW: int = 256
    MAX_NEW_TOKENS: int = 1024

    # Reproducibitlity Configurations

    DETERMINISTIC: bool = False
    SEED: int = 42

    # Derived Configurations

    HEAD_DIM: int = D_MODEL // N_HEADS
    N_REP: int = N_HEADS // N_KV_HEADS
    GLOBAL_BATCH_SIZE: int = BATCH_SIZE * max(NUM_GPUS, 1) * GRADIENT_ACCUMULATION_STEPS

    # Validation

    def validate(self):

        if not 0.0 < self.VALIDATION_SPLIT < 1.0 or not 0.0 < self.SFT_VALIDATION_SPLIT < 1.0 :
            raise ValueError("vALIDATION_SPLIT must be between 0 and 1!")

        if self.D_MODEL % self.N_HEADS != 0:
            raise ValueError("D_MODEL must be divisible by N_HEADS")

        RATIO_SUM = (
            self.SFT_D1_TOKEN_RATIO +
            self.SFT_D2_TOKEN_RATIO +
            self.SFT_REPLAY_TOKEN_RATIO
        )

        if abs(RATIO_SUM - 1.0) > 1e-8:
            raise ValueError("SFT dataset ratios must sum to 1.0!")

CONFIG = Config()
CONFIG.validate()
