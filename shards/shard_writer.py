import json
import struct
import numpy as np
from pathlib import Path
from config import CONFIG

TOKEN_MAGIC = b"PLUTOAIBIN01"
MASK_MAGIC = b"PLUTOAIMSK01"
INDEX_MAGIC = b"PLUTOAIIDX01"

TOKEN_HEADER_FORMAT = "<12sII"
MASK_HEADER_FORMAT = "<12sII"
INDEX_HEADER_FORMAT = "<12sI"

TOKEN_HEADER_SIZE = struct.calcsize(TOKEN_HEADER_FORMAT)
MASK_HEADER_SIZE = struct.calcsize(MASK_HEADER_FORMAT)
INDEX_HEADER_SIZE = struct.calcsize(INDEX_HEADER_FORMAT)

class PackedShardWriter:

    def __init__(self, output_dir: Path, split: str, tokens_per_shard: int, context_length: int, dataset_mode: str = "pretrain"):

        if tokens_per_shard % context_length != 0:
            raise ValueError("tokens_per_shards must be divisible by context_length")

        if dataset_mode not in CONFIG.DATASET_MODES:
            raise ValueError(f"Unsupported mode: {dataset_mode}")

        self.output_dir = output_dir
        self.output_dir.mkdir(parents = True, exist_ok = True)

        self.split = split
        self.tokens_per_shard = tokens_per_shard
        self.context_length = context_length

        self.token_buffer = bytearray()
        self.mask_buffer = bytearray() if dataset_mode == "sft" else None

        self.sequence_offsets: list[int] = []
        self.sequence_lengths: list[int] = []

        self.shard_index = 0

        self.total_tokens = 0
        self.total_sequences = 0

        self.shard_metadata = []

        self.dataset_mode = dataset_mode

    def _current_token_count(self):
        return len(self.token_buffer) // 2

    def add_tokens(self, token_ids: list[int], loss_mask: list[int] | None = None):

        if not token_ids:
            return
            
        if len(token_ids) > self.tokens_per_shard:
            raise ValueError("A sequence exceeded tokens_per_shard capacity!")

        if self.dataset_mode == "sft":

            if loss_mask is None:
                raise ValueError("SFT mode requires loss_mask!")

            if not any(loss_mask):
                raise ValueError("SFT Sequence has no supervised tokens!")
            
            if len(token_ids) != len(loss_mask):
                raise ValueError("input_ids length does not match loss_mask!")

            if any(mask not in (0, 1) for mask in loss_mask):
                raise ValueError("loss_mask must be 0 or 1.")

        if any(token_id < 0 or token_id > 65535 for token_id in token_ids):
            raise ValueError("token_id exceeded uint16 range!")

        current_tokens = self._current_token_count()

        if current_tokens + len(token_ids) > self.tokens_per_shard:
            self.flush()

        self._append(
            token_ids,
            loss_mask
        )

    def add(self, token_ids: list[int], loss_mask: list[int] | None = None):

        if not token_ids:
            return

        self.add_tokens(
            token_ids,
            loss_mask
        )

    def _append(self, token_ids: list[int], loss_mask: list[int] | None = None):

        offset = self._current_token_count()
        self.sequence_offsets.append(offset)
        self.sequence_lengths.append(len(token_ids))
        
        tokens = np.asarray(
            token_ids,
            dtype = np.uint16
        )

        self.token_buffer.extend(
            tokens.tobytes()
        )
        
        if self.dataset_mode == "sft":

            mask = np.asarray(
                loss_mask,
                dtype = np.uint8
            )

            self.mask_buffer.extend(
                mask.tobytes()
            )

        self.total_tokens += len(token_ids)
        self.total_sequences += 1

    def _write_index(self, index_path: Path):

        offsets = np.asarray(
            self.sequence_offsets,
            dtype = np.uint64
        )

        lengths = np.asarray(
            self.sequence_lengths,
            dtype = np.uint32
        )

        with index_path.open("wb") as f:

            f.write(
                struct.pack(
                    INDEX_HEADER_FORMAT,
                    INDEX_MAGIC,
                    len(offsets)
                )
            )

            f.write(offsets.tobytes())
            f.write(lengths.tobytes())

    def flush(self):

        if not self.token_buffer:
            return

        shard_name = f"shard_{self.shard_index:05d}"

        token_path = self.output_dir / f"{shard_name}.bin"
        index_path = self.output_dir / f"{shard_name}.idx"

        token_count = self._current_token_count()
        sequence_count = len(self.sequence_offsets)

        with token_path.open("wb") as f:

            f.write(
                struct.pack(
                    TOKEN_HEADER_FORMAT,
                    TOKEN_MAGIC,
                    1,
                    token_count
                )
            )

            f.write(self.token_buffer)

        self._write_index(index_path)
        
        shard_metadata = {
            "shard": shard_name,
            "tokens": token_count,
            "sequences": sequence_count,
            "token_file": token_path.name,
            "index_file": index_path.name,
            "token_dtype": "uint16"
        }

        if self.dataset_mode == "sft":

            if self.mask_buffer is None:
                raise ValueError("SFT mask_buffer is missing!")

            mask_path = self.output_dir / f"{shard_name}.mask"

            with mask_path.open("wb") as f:

                f.write(
                    struct.pack(
                        MASK_HEADER_FORMAT,
                        MASK_MAGIC,
                        1,
                        token_count
                    )
                )

                f.write(self.mask_buffer)

                shard_metadata.update({
                    "mask_file": mask_path.name,
                    "mask_dtype": "uint8"
                })

        self.shard_metadata.append(shard_metadata)
        self.shard_index += 1

        self.token_buffer.clear()

        if self.mask_buffer is not None:
            self.mask_buffer.clear()

        self.sequence_offsets.clear()
        self.sequence_lengths.clear()

    def close(self):

        self.flush()
        metadata_path = self.output_dir / "metadata.json"

        metadata = {
            "format": "PLUTO_AI_PACKED_SHARDS_V1",
            "dataset_mode": self.dataset_mode,
            "split": self.split,
            "context_length": self.context_length,
            "tokens_per_shard": self.tokens_per_shard,
            "total_tokens": self.total_tokens,
            "total_sequences": self.total_sequences,
            "num_shards": len(self.shard_metadata),
            "token_dtype": "uint16",
            "index_offset_dtype": "uint64",
            "index_length_dtype": "uint32",
            "shards": self.shard_metadata
        }

        if self.dataset_mode == "sft":
            metadata["mask_dtype"] = "uint8"

        with metadata_path.open("w", encoding = "utf-8") as f:
            json.dump(
                metadata,
                f,
                indent = 4
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
