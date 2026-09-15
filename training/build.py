from config import CONFIG

import math
import torch
from torch.nn.parallel import DistributedDataParallel
from pathlib import Path

from data.loader import create_dataloader
from model.gpt import GPT

from training.checkpoint import CheckpointManager
from training.logger import TensorBoardLogger
from training.optimizer import build_optimizer
from training.scheduler import WarmupCosineScheduler
from training.state import TrainingState
from training.trainer import Trainer
from training.distributed import is_main_process, is_distributed
from training.utils import restore_rng_state

def build_trainer(device: torch.device) -> Trainer:

    model = GPT().to(device)

    optimizer = build_optimizer(
        model = model,
        learning_rate = CONFIG.LEARNING_RATE,
        weight_decay = CONFIG.WEIGHT_DECAY,
        betas = CONFIG.ADAM_BETAS,
        eps = CONFIG.ADAM_EPS
    )

    total_steps = CONFIG.MAX_TRAIN_STEPS

    if total_steps is None:

        _, train_loader, train_sampler = create_dataloader(
            split_dir = CONFIG.TRAIN_PRETRAIN_DIR,
            batch_size = CONFIG.BATCH_SIZE,
            shuffle = True
        )

        batches_per_epoch = len(train_loader)
        updates_per_epoch = math.ceil(batches_per_epoch / CONFIG.GRADIENT_ACCUMULATION_STEPS)

        total_steps = updates_per_epoch * CONFIG.NUM_EPOCHS

    else:

        _, train_loader, train_sampler = create_dataloader(
            split_dir = CONFIG.TRAIN_PRETRAIN_DIR,
            batch_size = CONFIG.BATCH_SIZE,
            shuffle = True
        )
    
    _, valid_loader, valid_sampler = create_dataloader(
        split_dir = CONFIG.VALID_PRETRAIN_DIR,
        batch_size = CONFIG.BATCH_SIZE,
        shuffle = False
    )

    scheduler = WarmupCosineScheduler(
        optimizer = optimizer,
        warmup_steps = CONFIG.WARMUP_STEPS,
        total_steps = total_steps,
        max_lr = CONFIG.LEARNING_RATE,
        min_lr = CONFIG.MIN_LEARNING_RATE
    )

    scaler = torch.amp.GradScaler(
        CONFIG.DEVICE.type,
        enabled = CONFIG.USE_AMP
    )

    if is_distributed():

        model = DistributedDataParallel(
            model,
            device_ids = [device.index],
            output_device = device.index,
            broadcast_buffers = False,
            find_unused_parameters = False
        )

    checkpoint_manager = CheckpointManager(
        directory = CONFIG.CHECKPOINT_DIR,
        keep_last_n = CONFIG.KEEP_LAST_N_CHECKPOINTS
    )

    logger = TensorBoardLogger(
        log_dir = CONFIG.TENSORBOARD_DIR,
        enabled = is_main_process()
    )

    state = TrainingState()

    if CONFIG.RESUME_FROM_CHECKPOINTS is not None:

        checkpoint = checkpoint_manager.load(
            checkpoint_path = CONFIG.RESUME_FROM_CHECKPOINTS,
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
            print(f"Resumed from: {CONFIG.RESUME_FROM_CHECKPOINTS}")
            print(f"Global Step: {state.GLOBAL_STEP:,}")
            print(f"Epoch: {state.EPOCH}")
            print(f"Batch in Epoch: {state.BATCH_IN_EPOCH}")

        restore_rng_state(
            checkpoint.get("rng_state", {})
        )

    config = {
        "dataset_mode": CONFIG.DATASET_MODE,
        "model": {
            "vocab_size": model.module.config.VOCAB_SIZE if hasattr(model, "module") else model.config.VOCAB_SIZE
        }
    }

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
        total_steps = total_steps
    )
