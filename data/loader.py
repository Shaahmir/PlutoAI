import torch
from torch.utils.data import DataLoader, DistributedSampler

from pathlib import Path
from config import CONFIG

from data.collator import Collator
from data.dataset import PackedTokenDataset
from training.distributed import get_rank, get_world_size, is_distributed

def create_dataloader(split_dir: Path, batch_size: int, shuffle: bool, collate_fn = Collator) -> tuple[PackedTokenDataset, DataLoader, DistributedSampler | None]:

    dataset = PackedTokenDataset(split_dir)
    sampler = None

    if is_distributed():
        sampler = DistributedSampler(
            dataset,
            num_replicas = get_world_size(),
            rank = get_rank(),
            shuffle = shuffle,
            drop_last = CONFIG.DROP_LAST
        )

    loader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = shuffle if sampler is None else False,
        sampler = sampler,
        collate_fn = collate_fn(),
        num_workers = CONFIG.NUM_WORKERS,
        pin_memory = CONFIG.PIN_MEMORY,
        persistent_workers = CONFIG.PERSISTENT_WORKERS if CONFIG.NUM_WORKERS > 0 else False,
        prefetch_factor = CONFIG.PREFETCH_FACTOR if CONFIG.NUM_WORKERS > 0 else None,
        drop_last = CONFIG.DROP_LAST
    )

    return (
        dataset,
        loader,
        sampler
    )
