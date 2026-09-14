import json
import struct
from pathlib import Path
from config import CONFIG

import torch
import numpy as np
from torch.utils.data import Dataset
from bisect import bisect_right

TOKEN_MAGIC = b"PLUTOAIBIN01"
MASK_MAGIC = b"PLUTOAIMSK01"
INDEX_MAGIC = b"PLUTOAIIDX01"

TOKEN_HEADER_FORMAT = "<12sII"
MASK_HEADER_FORMAT = "<12sII"
INDEX_HEADER_FORMAT = "<12sI"

TOKEN_HEADER_SIZE = struct.calcsize(TOKEN_HEADER_FORMAT)
MASK_HEADER_SIZE = struct.calcsize(MASK_HEADER_FORMAT)
INDEX_HEADER_SIZE = struct.calcsize(INDEX_HEADER_FORMAT)

class MappedShard:

    def __init__(self, directory: Path, shard_metadata: dict):
        
        self.directory = directory

        self.token_path = directory / shard_metadata["token_file"]
        self.index_path = directory / shard_metadata["index_file"]
        self.mask_path = None

        if "mask_file" in shard_metadata:
            self.mask_path = directory / shard_metadata["mask_file"]

        self.token_count = shard_metadata["tokens"]
        self.sequence_count = shard_metadata["sequences"]

        self._load_header()
        self._load_index()
        self._map_arrays()

    def _load_header(self):
        
        # TOKEN

        with self.token_path.open("rb") as f:
            header = f.read(TOKEN_HEADER_SIZE)

        magic, version, token_count = struct.unpack(
            TOKEN_HEADER_FORMAT,
            header
        )
        
        if magic != TOKEN_MAGIC:
            raise ValueError(f"Invalid token shard {self.token_path}!")

        if version != 1:
            raise ValueError(f"Unsupported token shard version {version}!")

        if token_count != self.token_count:
            raise ValueError("Token count mismatch!")

        # MASK

        if self.mask_path is not None:

            with self.mask_path.open("rb") as f:
                header = f.read(MASK_HEADER_SIZE)

            magic, version, mask_count = struct.unpack(
                MASK_HEADER_FORMAT,
                header
            )
            
            if magic != MASK_MAGIC:
                raise ValueError(f"Invalid mask shard {self.mask_path}!")

            if version != 1:
                raise ValueError(f"Unsupported mask shard version {version}!")

            if mask_count != self.token_count:
                raise ValueError("Mask/Token count mismatch!")

    def _load_index(self):

        with self.index_path.open("rb") as f:
            header = f.read(INDEX_HEADER_SIZE)

            magic, sequence_count = struct.unpack(
                INDEX_HEADER_FORMAT,
                header
            )
            
            if magic != INDEX_MAGIC:
                raise ValueError(f"Invalid index shard {self.index_path}!")

            if sequence_count != self.sequence_count:
                raise ValueError("Sequence count mismatch!")

            self.offsets = np.fromfile(
                f,
                dtype = np.uint64,
                count = sequence_count
            )

            self.lengths = np.fromfile(
                f,
                dtype = np.uint32,
                count = sequence_count
            )

    def _map_arrays(self):

        self.tokens = np.memmap(
            self.token_path,
            dtype = np.uint16,
            mode = "r",
            offset = TOKEN_HEADER_SIZE,
            shape = (self.token_count, )
        )

        self.loss_mask = None

        if self.mask_path is not None:

            self.loss_mask = np.memmap(
                self.mask_path,
                dtype = np.uint8,
                mode = "r",
                offset = MASK_HEADER_SIZE,
                shape = (self.token_count, )
            )

    def get(self, index: int):

        offset = int(self.offsets[index])
        length = int(self.lengths[index])

        end = offset + length
        tokens = self.tokens[offset:end]

        if self.loss_mask is None:
            return (
                tokens,
                None
            )

        return (
            tokens,
            self.loss_mask[offset:end]
        )

class PackedTokenDataset(Dataset):

    def __init__(self, split_dir: Path):

        self.split_dir = Path(split_dir)
        metadata_path = self.split_dir / "metadata.json"

        if not metadata_path.exists():
            raise FileNotFoundError("metadata.json file not found!")

        with metadata_path.open("r", encoding = "utf-8") as f:
            self.metadata = json.load(f)

        self.dataset_mode = self.metadata["dataset_mode"]

        if self.dataset_mode not in CONFIG.DATASET_MODES:
            raise ValueError(f"Unsupported dataset mode: {self.dataset_mode}")

        self.shards: list[MappedShard] = []
        self.cumulative_sizes: list[int] = []

        total_sequences = 0

        for shard_metadata in self.metadata["shards"]:

            shard = MappedShard(
                directory = self.split_dir,
                shard_metadata = shard_metadata
            )

            self.shards.append(shard)
            total_sequences += shard.sequence_count
            self.cumulative_sizes.append(total_sequences)

        self.total_sequences = total_sequences

    def __len__(self):
        return self.total_sequences

    def _locate(self, index: int) -> tuple[int, int]:

        if index < 0:
            index += len(self)

        if index < 0 or index >= len(self):
            raise IndexError(index)

        shard_index = bisect_right(
            self.cumulative_sizes,
            index
        )

        previous_size = 0 if shard_index == 0 else self.cumulative_sizes[shard_index - 1]
        local_index = index - previous_size

        return shard_index, local_index

    def __getitem__(self, index: int):

        shard_index, local_index = self._locate(index)
        tokens, loss_mask = self.shards[shard_index].get(local_index)

        tokens = torch.from_numpy(
            np.array(tokens, copy = True)
        ).long()

        result = {
            "input_ids": tokens[:-1],
            "labels": tokens[1:]
        }

        if loss_mask is not None:

            loss_mask = torch.from_numpy(
                np.array(loss_mask, copy = True)
            ).bool()

            shifted_mask = loss_mask[1:]

            if shifted_mask.numel() == 0 or not bool(shifted_mask.any()):
                raise RuntimeError(f"SFT sequence {index} has zero supervised tokens!")

            result["loss_mask"] = shifted_mask

        return result
