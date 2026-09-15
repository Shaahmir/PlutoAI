import torch
from tqdm import tqdm
from config import CONFIG
from data.loader import create_dataloader
from training.distributed import cleanup_distributed, setup_distributed

def main():

    rank, world_size, device = setup_distributed()
    MAX_BATCHES = 5

    try:
        _, loader, sampler = create_dataloader(
            split_dir = CONFIG.TRAIN_PRETRAIN_DIR,
            batch_size = CONFIG.BATCH_SIZE,
            shuffle = True
        )

        if sampler is not None:
            sampler.set_epoch(0)

        loader_iter = tqdm(iter(loader), desc = f"Testing Dataloader [Rank {rank}]", total = MAX_BATCHES) if rank == 0 else loader

        for i, batch in enumerate(loader_iter):

            if i >= MAX_BATCHES:
                break

            assert batch["input_ids"].shape[1] == CONFIG.CONTEXT_LENGTH - 1
            assert "loss_mask" not in batch

        print(f"Rank: {rank}/{world_size}")
        print(f"Device: {device}")

        print(f"Input: ", tuple(batch['input_ids'].shape))
        print(f"Labels: {tuple(batch['labels'].shape)}")
            
        print(f"Input dtype: {batch['input_ids'].dtype}")
        print(f"Labels dtype: {batch['labels'].dtype}")

        print(f"Rank {rank}: Dataloader looks Good")
    
    finally:
        cleanup_distributed()

if __name__ == "__main__":
    main()

# !torchrun --nproc_per_node=2 testing/test_loader.py
# Change: NUM_WORKERS = 0 on CPU and run using "python testing/test_loader.py"