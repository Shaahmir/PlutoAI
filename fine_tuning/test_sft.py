from config import CONFIG
from data.loader import create_dataloader
from fine_tuning.collator_sft import SFTCollator
from training.distributed import setup_distributed, cleanup_distributed, set_seed
from fine_tuning.sft import sft_trainer

SMOKE_TEST_STEPS = 20

def verify_batch(rank: int):

    _, train_loader, train_sampler = create_dataloader(
        split_dir = CONFIG.TRAIN_SFT_DIR,
        batch_size = CONFIG.BATCH_SIZE,
        shuffle = True,
        collate_fn = SFTCollator
    )

    first_batch = next(iter(train_loader))

    input_ids = first_batch["input_ids"]
    labels = first_batch["labels"]

    if input_ids.ndim != 2:
        raise RuntimeError(f"Expected input_ids 2D got {input_ids.shape}!")

    if labels.shape != input_ids.shape:
        raise RuntimeError("input_ids and labels have different shapes!")

    ignored = (labels == -100).sum().item()
    trainable = (labels != -100).sum().item()

    if trainable == 0:
        raise RuntimeError("Entire SFT batch is masked")

    if rank == 0:
        print("First Batch")
        print(f"input_ids: {tuple(input_ids.shape)}")
        print(f"labels: {tuple(labels.shape)}")
        print(f"Ignored: {ignored:,}")
        print(f"Trainable: {trainable:,}")

def main():

    rank, world_size, device = setup_distributed()
    set_seed(CONFIG.SEED)

    try:
        if rank == 0:
            print("=" * 50)
            print("SFT Smoke Test")

            print(f"World Size: {world_size}")
            print(f"Device: {device}")
            print(f"Dataset Mode: {CONFIG.DATASET_MODE}")
            print(f"Batch/GPU: {CONFIG.BATCH_SIZE}")
            print(f"Gradient Accumulation: {CONFIG.GRADIENT_ACCUMULATION_STEPS}")
            print(f"Global Batch: {CONFIG.GLOBAL_BATCH_SIZE}")
            print(f"Context: {CONFIG.CONTEXT_LENGTH}")
            print(f"AMP: {CONFIG.USE_AMP}")

        verify_batch(rank)

        CONFIG.MAX_TRAIN_STEPS = SMOKE_TEST_STEPS
        trainer = sft_trainer(device)
        trainer.train()

        print("Training Looks Good!")
    
    finally:
        cleanup_distributed()

if __name__ == "__main__":
    main()
