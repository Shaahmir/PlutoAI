import torch

from config import CONFIG
from training.build import build_trainer
from training.distributed import cleanup_distributed, set_seed, setup_distributed

def main():

    rank, world_size, device = setup_distributed()

    try:
        set_seed(
            CONFIG.SEED,
            rank
        )

        if CONFIG.USE_TF32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        trainer = build_trainer(device)

        if rank == 0:

            print("="*50)
            print("Pre-Training")

            print(f"World Size: {world_size}")
            print(f"Device: {device}")
            print(f"Dataset Mode: {CONFIG.DATASET_MODE}")
            print(f"Batch/GPU: {CONFIG.BATCH_SIZE}")
            print(f"Gradient Accumulation: {CONFIG.GRADIENT_ACCUMULATION_STEPS}")
            print(f"Global Batch: {CONFIG.GLOBAL_BATCH_SIZE}")
            print(f"Context: {CONFIG.CONTEXT_LENGTH}")
            print(f"AMP: {CONFIG.USE_AMP}")

        trainer.train()
    
    finally:
        cleanup_distributed()

if __name__ == "__main__":
    main()
