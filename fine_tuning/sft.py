import math
from pathlib import Path
from config import CONFIG

import torch
from torch.nn.parallel import DistributedDataParallel
from data.loader import create_dataloader

from model.gpt import GPT
from fine_tuning.collator_sft import SFTCollator
from training.distributed import is_distributed, is_main_process, setup_distributed, cleanup_distributed, set_seed
from training.logger import TensorBoardLogger
from training.optimizer import build_optimizer
from training.scheduler import WarmupCosineScheduler
from training.checkpoint import CheckpointManager
from training.state import TrainingState
from training.trainer import Trainer

from fine_tuning.load_pretrain import load_pretrain_weights
from training.utils import restore_rng_state

def sft_trainer(device: torch.device) -> Trainer:

    # Loading Data

    _, train_loader, train_sampler = create_dataloader(
        split_dir = CONFIG.TRAIN_SFT_DIR,
        batch_size = CONFIG.SFT_BATCH_SIZE,
        shuffle = True,
        collate_fn = SFTCollator
    )

    _, valid_loader, valid_sampler = create_dataloader(
        split_dir = CONFIG.VALID_SFT_DIR,
        batch_size = CONFIG.SFT_BATCH_SIZE,
        shuffle = False,
        collate_fn = SFTCollator
    )

    # Model

    model = GPT().to(device)

    # Loading Pretrain Weights

    if not Path(CONFIG.BASE_MODEL_CHECKPOINTS).exists():
        raise FileNotFoundError(f"SFT base checkpoint not found {CONFIG.BASE_MODEL_CHECKPOINTS}!")

    if is_main_process():
        print(f"Loading pretrain weights from {CONFIG.BASE_MODEL_CHECKPOINTS} ...")

    load_pretrain_weights(
        model = model,
        checkpoint_path = CONFIG.BASE_MODEL_CHECKPOINTS,
        device = device
    )

    # Distrubuted Sampler

    if is_distributed():

        model = DistributedDataParallel(
            model,
            device_ids = [device.index],
            output_device = device.index,
            broadcast_buffers = False,
            find_unused_parameters = False
        )

    # Optimizer

    optimizer = build_optimizer(
        model = model,
        learning_rate = CONFIG.SFT_LR,
        weight_decay = CONFIG.SFT_WEIGHT_DECAY,
        betas = CONFIG.SFT_ADAM_BETAS,
        eps = CONFIG.SFT_EPS
    )

    # Scheduler

    batches_per_epoch = len(train_loader)
    updates_per_epoch = math.ceil(batches_per_epoch / CONFIG.SFT_GRADIENT_ACCUMULATION_STEPS)
    total_steps = updates_per_epoch * CONFIG.SFT_EPOCHS

    scheduler = WarmupCosineScheduler(
        optimizer = optimizer,
        warmup_steps = CONFIG.SFT_WARMUP_STEPS,
        total_steps = total_steps,
        max_lr = CONFIG.SFT_LR,
        min_lr = CONFIG.SFT_MIN_LR
    )

    # Scaler
    
    scaler = torch.amp.GradScaler(
        CONFIG.DEVICE.type,
        enabled = CONFIG.USE_AMP
    )

    # Checkpoint Manager

    checkpoint_manager = CheckpointManager(
        directory = CONFIG.SFT_CHECKPOINT_DIR,
        keep_last_n = CONFIG.SFT_KEEP_LAST_N
    )

    # Logger

    logger = TensorBoardLogger(
        log_dir = CONFIG.SFT_LOG_DIR,
        enabled = is_main_process()
    )

    # State

    state = TrainingState()

    if CONFIG.LATEST_CHECKPOINT.exists():

        checkpoint = checkpoint_manager.load(
            checkpoint_path = CONFIG.LATEST_CHECKPOINT,
            model = model,
            optimizer = optimizer,
            scheduler = scheduler,
            scaler = scaler,
            map_location = device
        )

        training_state = checkpoint["training_state"]
        state.GLOBAL_STEP = training_state["global_step"]
        state.EPOCH = training_state["epoch"]
        state.BATCH_IN_EPOCH = training_state["batch_in_epoch"]
        state.OPTIMIZER_STEP = training_state["optimizer_step"]
        state.SEEN_TOKENS = training_state["seen_tokens"]
        state.BEST_VALID_LOSS = training_state["best_validation_loss"]

        if is_main_process():
            print(f"Resumed from: {CONFIG.LATEST_CHECKPOINT}")
            print(f"Global Step: {state.GLOBAL_STEP:,}")
            print(f"Epoch: {state.EPOCH}")
            print(f"Batch in Epoch: {state.BATCH_IN_EPOCH}")
            print(f"Resuming SFT Training from {state.EPOCH} ...")

        try:
            restore_rng_state(
                checkpoint.get("rng_state", {})
            )
        except Exception as e:
            if is_main_process():
                print(f"WARNING: Could not restore rng state ({e})")

    else:

        state.GLOBAL_STEP = 0
        state.EPOCH = 0
        state.BATCH_IN_EPOCH = 0
        state.OPTIMIZER_STEP = 0
        state.SEEN_TOKENS = 0
        state.BEST_VALID_LOSS = None

        if is_main_process():
            print("Initialized SFT State, optimizer and scheduler!")

    # Config

    config = {
        "stage": "SFT",
        "dataset_mode": CONFIG.DATASET_MODE,
        "model": {
            "vocab_size": model.module.config.VOCAB_SIZE if hasattr(model, "module") else model.config.VOCAB_SIZE
        }
    }

    # Trainer

    return Trainer(
        model = model,
        train_loader = train_loader,
        train_sampler = train_sampler,
        valid_loader = valid_loader,
        valid_sampler = valid_sampler,
        optimizer = optimizer,
        scheduler = scheduler,
        scaler = scaler,
        device = device,
        checkpoint_manager = checkpoint_manager,
        logger = logger,
        state = state,
        config = config,
        total_steps = total_steps,
        num_epochs = CONFIG.SFT_EPOCHS
    )

def main():

    rank, world_size, device = setup_distributed()
    set_seed(CONFIG.SEED, rank)

    try:

        if CONFIG.USE_TF32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        if rank == 0:
            print("=" * 50)
            print("Supervised Fine Tuning")

            print(f"World Size: {world_size}")
            print(f"Batch Size: {CONFIG.SFT_BATCH_SIZE}")
            print(f"Gradient Accumulation: {CONFIG.SFT_GRADIENT_ACCUMULATION_STEPS}")
            print(f"Learning Rate: {CONFIG.SFT_LR}")
            print(f"Epochs: {CONFIG.SFT_EPOCHS}")


        trainer = sft_trainer(device)
        trainer.train()

    finally:
        cleanup_distributed()

if __name__ == "__main__":
    main()
