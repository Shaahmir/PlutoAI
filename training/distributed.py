import os
import random

from config import CONFIG

import numpy as np
import torch
import torch.distributed as dist

def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0

def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1

def is_main_process() -> bool:
    return get_rank() == 0

def setup_distributed() -> tuple[int, int, torch.device]:

    if "RANK" not in os.environ:

        device = CONFIG.DEVICE
        return 0, 1, device

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for distributed CUDA training!")

    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend = "nccl",
        rank = rank,
        world_size = world_size
    )

    device = torch.device("cuda", local_rank)

    return rank, world_size, device

def cleanup_distributed():
    if is_distributed():
        dist.destroy_process_group()

def barrier():
    if is_distributed():
        dist.barrier()

def broadcast_object(obj, src: int = 0):

    if not is_distributed():
        return obj

    objects = [obj]

    dist.broadcast_object_list(
        objects,
        src = src
    )

    return objects[0]

def set_seed(seed: int, rank: int = 0):

    final_seed = seed + rank
    random.seed(final_seed)
    np.random.seed(final_seed)
    torch.manual_seed(final_seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(final_seed)
        torch.cuda.manual_seed_all(final_seed)
