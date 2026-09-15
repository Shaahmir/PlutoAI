import math
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from contextlib import nullcontext
from pathlib import Path
from config import CONFIG
from tqdm import tqdm

from model.gpt import GPT

from training.checkpoint import CheckpointManager
from training.evaluate import evaluate, perplexity
from training.logger import TensorBoardLogger
from training.state import TrainingState
from training.distributed import get_rank, get_world_size, is_distributed, is_main_process
from training.utils import save_rng_state, count_valid_tokens

class Trainer:

    def __init__(
        self,
        model: GPT,
        train_loader,
        train_sampler,
        valid_loader,
        valid_sampler,
        optimizer,
        scheduler,
        scaler,
        device: torch.device,
        checkpoint_manager: CheckpointManager,
        logger: TensorBoardLogger,
        state: TrainingState,
        config: dict,
        total_steps: int | None = None,
        num_epochs: int | None = None,
        valid_loaders: dict | None = None,
        valid_samplers: dict | None = None
    ):

        self.model = model

        self.train_loader = train_loader
        self.train_sampler = train_sampler
        self.valid_loader = valid_loader
        self.valid_sampler = valid_sampler

        self.valid_loaders = valid_loaders or {"combined": valid_loader}
        self.valid_samplers = valid_samplers or {"combined": valid_sampler}

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.device = device

        self.checkpoint_manager = checkpoint_manager
        self.logger = logger
        self.state = state

        self.config = config
        self.total_steps = total_steps
        self.num_epochs = num_epochs if num_epochs is not None else CONFIG.NUM_EPOCHS
        self.grad_clip = CONFIG.MAX_GRAD_NORM if CONFIG.DATASET_MODE == "pretrain" else CONFIG.SFT_GRAD_CLIP

        self.eval_every_step = CONFIG.EVAL_EVERY_STEPS if CONFIG.DATASET_MODE == "pretrain" else CONFIG.SFT_EVAL_EVERY_STEPS
        self.save_every_step = CONFIG.SAVE_EVERY_STEPS if CONFIG.DATASET_MODE == "pretrain" else CONFIG.SFT_SAVE_EVERY_STEPS

        self.rank = get_rank()
        self.world_size = get_world_size()

    def _autocast_context(self):

        if not CONFIG.USE_AMP:
            return nullcontext()

        return torch.autocast(
            device_type = CONFIG.DEVICE.type,
            dtype = CONFIG.AMP_DTYPE
        )

    def _distributed_loss_statistics(self, loss: torch.Tensor, labels: torch.Tensor) -> tuple[float, int]:

        valid_tokens = (labels != -100).sum()

        loss_sum = loss.detach().double() * valid_tokens
        token_count = valid_tokens.detach().double()

        if is_distributed():

            dist.all_reduce(
                loss_sum,
                op = dist.ReduceOp.SUM
            )

            dist.all_reduce(
                token_count,
                op = dist.ReduceOp.SUM
            )

        global_tokens = int(token_count.item())
        global_loss = loss_sum.item() / max(global_tokens, 1)

        return (
            global_loss,
            global_tokens
        )

    def _save_checkpoint(self, filename: str) -> Path:

        return self.checkpoint_manager.save(
            filename = filename,
            model = self.model,
            optimizer = self.optimizer,
            scheduler = self.scheduler,
            scaler = self.scaler,
            state = self.state,
            config = self.config,
            rng_state = save_rng_state()
        )

    def _run_validation(self) -> float:

        if self.valid_sampler is not None:
            self.valid_sampler.set_epoch(
                self.state.EPOCH
            )
        
        validation_loss, token_count = evaluate(
            model = self.model,
            dataloader = self.valid_loader,
            device = self.device,
            max_steps = CONFIG.VALIDATION_STEPS
        )

        validation_ppl = perplexity(validation_loss)

        if is_main_process():
            print(f"Validation | Loss: {validation_loss:.6f} | Perplexity: {validation_ppl:.4f} | Tokens: {token_count:,}")
        
            self.logger.scalar(
                "validation/loss",
                validation_loss,
                self.state.GLOBAL_STEP
            )

            self.logger.scalar(
                "validation/perplexity",
                validation_ppl,
                self.state.GLOBAL_STEP
            )

        self.model.train()

        return validation_loss

    def train(self):

        self.model.train()
        self.optimizer.zero_grad(
            set_to_none = True
        )

        total_steps = self.total_steps

        if total_steps is None:
            total_steps = CONFIG.MAX_TRAIN_STEPS

        if total_steps is None:

            total_batches = len(self.train_loader)
            steps_per_epoch = max(1, math.ceil(total_batches / CONFIG.GRADIENT_ACCUMULATION_STEPS))
            total_steps = steps_per_epoch * self.num_epochs
        
        start_time = time.perf_counter()

        accumulation_tokens = 0
        accumulation_steps = 0

        resume_epoch = self.state.EPOCH
        resume_batch = self.state.BATCH_IN_EPOCH

        for epoch in range(self.state.EPOCH, self.num_epochs):

            self.state.EPOCH = epoch

            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            pbar = tqdm(
                enumerate(self.train_loader),
                total = len(self.train_loader),
                disable = not is_main_process(),
                dynamic_ncols = True,
                desc = f"Epoch: {epoch + 1}",
                unit = "it"
            )

            for batch_index, batch in pbar:

                if epoch == resume_epoch and batch_index < resume_batch:
                    continue

                if total_steps is not None and self.state.GLOBAL_STEP >= total_steps:
                    break

                input_ids = batch["input_ids"].to(self.device, non_blocking = True)
                labels = batch["labels"].to(self.device, non_blocking = True)

                is_accumulation_boundary = (batch_index + 1) % CONFIG.GRADIENT_ACCUMULATION_STEPS == 0
                is_last_batch = batch_index == len(self.train_loader) - 1

                should_optimizer_step = is_accumulation_boundary or is_last_batch
                should_sync = not is_distributed() or should_optimizer_step

                if isinstance(self.model, DistributedDataParallel) and not should_sync:
                    sync_context = self.model.no_sync()

                else:
                    sync_context = nullcontext()

                with sync_context:

                    with self._autocast_context():

                        _, loss = self.model(
                            input_ids,
                            labels
                        )

                        if loss is None:
                            raise RuntimeError("Model did not return loss!")

                        remaining_batches = len(self.train_loader) - (batch_index - (batch_index % CONFIG.GRADIENT_ACCUMULATION_STEPS))

                        micro_batches_this_update = min(
                            CONFIG.GRADIENT_ACCUMULATION_STEPS,
                            remaining_batches
                        )

                        loss_for_backward = loss / micro_batches_this_update

                    if CONFIG.USE_AMP:
                        self.scaler.scale(
                            loss_for_backward
                        ).backward()

                    else:
                        loss_for_backward.backward()
                
                accumulation_steps += 1
                accumulation_tokens += count_valid_tokens(labels)
                self.state.BATCH_IN_EPOCH = batch_index + 1

                if not should_optimizer_step:
                    continue

                if CONFIG.USE_AMP:
                    self.scaler.unscale_(
                        self.optimizer
                    )

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.grad_clip
                )

                if CONFIG.USE_AMP:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                else:
                    self.optimizer.step()

                self.optimizer.zero_grad(
                    set_to_none = True
                )

                learning_rate = self.scheduler.step()
                
                self.state.GLOBAL_STEP += 1
                self.state.OPTIMIZER_STEP += 1
                self.state.SEEN_TOKENS += accumulation_tokens * self.world_size

                global_loss, global_tokens = self._distributed_loss_statistics(
                    loss,
                    labels
                )

                accumulation_tokens = 0
                accumulation_steps = 0

                if is_main_process():

                    elapsed_time = time.perf_counter() - start_time
                    tokens_per_second = self.state.SEEN_TOKENS / max(elapsed_time, 1e-6)

                    perplexity_value = perplexity(
                        global_loss
                    )

                    pbar.set_postfix(
                        loss = f"{global_loss:.5f}",
                        ppl = f"{perplexity_value:.3f}",
                        lr = f"{learning_rate:.3e}",
                        grad = f"{float(grad_norm):.3f}",
                        tps = f"{tokens_per_second:,.0f}"
                    )

                    self.logger.scalar(
                        "train/loss",
                        global_loss,
                        self.state.GLOBAL_STEP
                    )

                    self.logger.scalar(
                        "train/perplexity",
                        perplexity_value,
                        self.state.GLOBAL_STEP
                    )
                    
                    self.logger.scalar(
                        "train/learning_rate",
                        learning_rate,
                        self.state.GLOBAL_STEP
                    )

                    self.logger.scalar(
                        "train/grad_norm",
                        float(grad_norm),
                        self.state.GLOBAL_STEP
                    )

                    self.logger.scalar(
                        "train/tokens_per_second",
                        tokens_per_second,
                        self.state.GLOBAL_STEP
                    )

                    self.logger.scalar(
                        "train/tokens_seen",
                        self.state.SEEN_TOKENS,
                        self.state.GLOBAL_STEP
                    )

                if self.eval_every_step > 0 and self.state.GLOBAL_STEP % self.eval_every_step == 0:
                    validation_loss = self._run_validation()

                    if self.state.BEST_VALID_LOSS is None or validation_loss < self.state.BEST_VALID_LOSS:
                        self.state.BEST_VALID_LOSS = validation_loss

                        if is_main_process():
                            self._save_checkpoint(
                                "best.pt"
                            )

                if self.save_every_step > 0 and self.state.GLOBAL_STEP % self.save_every_step == 0:
                    if is_main_process():
                        
                        self._save_checkpoint(
                            f"step_{self.state.GLOBAL_STEP:08d}.pt"
                        )

                        self.checkpoint_manager.cleanup_step_checkpoints()

                        self._save_checkpoint(
                            "latest.pt"
                        )

            self.state.BATCH_IN_EPOCH = 0
            self.state.EPOCH = epoch + 1

            if is_main_process():
                self._save_checkpoint(
                    f"epoch_{self.state.EPOCH:04d}.pt"
                )

        self.logger.flush()
        self.logger.close()
