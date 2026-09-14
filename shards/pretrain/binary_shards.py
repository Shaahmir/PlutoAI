import json
import struct
import numpy as np

import torch
from torch.utils.data import Dataset
from pathlib import Path

MAGIC = b"PLUTOSHARD01"
DTYPE = np.int32
HEADER_FORMAT = "<12sIIII"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

class BinaryShard:

    def __init__(self, path: Path):

        self.path = path

        with path.open("rb") as f:
            header = f.read(HEADER_SIZE)

        (magic, self.context_length, self.examples, self.inputs_bytes, self.label_bytes) = struct.unpack(
            HEADER_FORMAT,
            header
        )

        if magic != MAGIC:
            raise ValueError("Invalid Shard!")

        self.input_offset = HEADER_SIZE
        self.label_offset = HEADER_SIZE + self.inputs_bytes

        self.input_array = np.memmap(
            path,
            dtype = DTYPE,
            mode = "r",
            offset = self.input_offset,
            shape = (self.examples, self.context_length)
        )

        self.label_array = np.memmap(
            path,
            dtype = DTYPE,
            mode = "r",
            offset = self.label_offset,
            shape = (self.examples, self.context_length)
        )

    def get(self, index: int):
        
        return (
            self.input_array[index],
            self.label_array[index]
        )

class BinaryShardDataset(Dataset):
    
    def __init__(self, directory: Path):

        self.directory = Path(directory)
        metadata_path = self.directory / "metadata.json"

        if not metadata_path.exists():
            raise FileNotFoundError("Missing Metadata!")

        with metadata_path.open("r", encoding = "utf-8") as f:
            self.metadata = json.load(f)
        
        self.shards: list[BinaryShard] = []
        self.cumulative_sizes = []

        total = 0

        for shard_info in self.metadata["shards"]:
            
            shard = BinaryShard(self.directory / shard_info["file"])
            self.shards.append(shard)
            total += shard.examples
            self.cumulative_sizes.append(total)

        self.total_examples = total

    def __len__(self):
        return self.total_examples

    def _find_shard(self, index: int):
        
        if index < 0:
            index += len(self)

        if index < 0 or index >= len(self):
            raise IndexError(index)

        shard_index = 0
        previous_size = 0

        for cumulative_size in self.cumulative_sizes:

            if index < cumulative_size:
                local_index = index - previous_size
                return (shard_index, local_index)

            previous_size = cumulative_size
            shard_index += 1

        raise IndexError(index)

    def __getitem__(self, index: int):

        shard_index, local_idnex = self._find_shard(index)
        input_ids, labels = self.shards[shard_index].get(local_idnex)

        return {
            "input_ids": torch.from_numpy(input_ids.copy()).long(),
            "labels": torch.from_numpy(labels.copy()).long()
        }