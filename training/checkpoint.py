import torch
from pathlib import Path

class CheckpointManager:

    def __init__(self, directory: Path, keep_last_n: int = 3):

        self.directory = Path(directory)
        self.directory.mkdir(parents = True, exist_ok = True)

        self.keep_last_n = keep_last_n

    def save(
        self,
        filename: str,
        model,
        optimizer,
        scheduler,
        scaler, 
        state,
        config: dict,
        rng_state: dict
    ):

        target_path = self.directory / filename
        temporary_path = self.directory / f".{filename}.tmp"

        model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()

        checkpoint = {
            "format_version": 1,
            "model": model_state,
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,

            "training_state": {
                "global_step": state.GLOBAL_STEP,
                "epoch": state.EPOCH,
                "batch_in_epoch": state.BATCH_IN_EPOCH,
                "optimizer_step": state.OPTIMIZER_STEP,
                "seen_tokens": state.SEEN_TOKENS,
                "best_validation_loss": state.BEST_VALID_LOSS,
            },

            "config": config,
            "rng_state": rng_state
        }

        torch.save(
            checkpoint,
            temporary_path
        )

        temporary_path.replace(
            target_path
        )

        if filename.startswith("step_"):
            self.cleanup_step_checkpoints()

        if filename.startswith("epoch_"):
            self.cleanup_epoch_checkpoints()

        return target_path

    def load(
        self,
        checkpoint_path: Path,
        model,
        optimizer = None,
        scheduler = None,
        scaler = None,
        map_location = "cpu"
    ):

        checkpoint = torch.load(
            checkpoint_path,
            map_location = map_location,
            weights_only = False
        )

        target_model = model.module if hasattr(model, "module") else model
        target_model.load_state_dict(checkpoint["model"])

        if optimizer is not None and checkpoint.get("optimizer") is not None:
            optimizer.load_state_dict(
                checkpoint["optimizer"]
            )

        if scheduler is not None and checkpoint.get("scheduler") is not None:
            scheduler.load_state_dict(
                checkpoint["scheduler"]
            )

        if scaler is not None and checkpoint.get("scaler") is not None:
            scaler.load_state_dict(
                checkpoint["scaler"]
            )

        return checkpoint

    def _cleanup_checkpoints(self, pattern: str):

        checkpoints = sorted(
            self.directory.glob(pattern),
            key = lambda path: path.stat().st_mtime
        )

        if len(checkpoints) <= self.keep_last_n:
            return

        checkpoints_to_delete = checkpoints[:-self.keep_last_n]

        for checkpoint in checkpoints_to_delete:
            checkpoint.unlink()

    def cleanup_step_checkpoints(self):
        self._cleanup_checkpoints("step_*.pt")

    def cleanup_epoch_checkpoints(self):
        self._cleanup_checkpoints("epoch_*.pt")
