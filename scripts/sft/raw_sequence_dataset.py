import json
import numpy as np

import torch
from torch.utils.data import Dataset
from bisect import bisect_right
from pathlib import Path

from data.dataset import MappedShard

class RawPackedDataset(Dataset):

    def __init__(self, split_dir: Path):

        metadata_path = split_dir / "metadata.json"

        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata.json: {metadata_path}!")
        
        with metadata_path.open("r", encoding = "utf-8") as f:
            self.metadata = json.load(f)
        
        self.shards = []
        self.cumulative_sizes = []

        total = 0

        for shard_info in self.metadata["shards"]:

            shard = MappedShard(
                split_dir,
                shard_info
            )

            self.shards.append(shard)

            total += shard.sequence_count

            self.cumulative_sizes.append(total)

        self.total_sequences = total

    def __len__(self):
        return self.total_sequences

    def _locate(self, index: int) -> tuple[int, int]:

        if index < 0:
            index += len(self)

        if index < 0 or index >= len(self):
            raise ValueError(index)

        shard_index = bisect_right(
            self.cumulative_sizes,
            index
        )

        previous = 0 if shard_index == 0 else self.cumulative_sizes[shard_index - 1]

        local_index = index - previous

        return (
            shard_index,
            local_index
        )

    def __getitem__(self, index: int) -> torch.Tensor:

        shard_index, local_index = self._locate(index)
        tokens, _ = self.shards[shard_index].get(local_index)

        if hasattr(tokens, "input_ids"):
            tokens = tokens.input_ids

        if isinstance(tokens, torch.Tensor):
            return tokens.detach().clone().long()

        if isinstance(tokens, np.ndarray):
            if not tokens.flags.writeable:
                tokens = tokens.copy()
            return torch.from_numpy(tokens).long()

        return torch.tensor(
            tokens,
            dtype = torch.long
        )
