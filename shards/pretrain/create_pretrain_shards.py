import json
from pathlib import Path
from tqdm import tqdm
from config import CONFIG
from collections import defaultdict

from scripts.pretrain.pretrain_processor import iter_pretrain_rows, load_pretrain_manifest
from shards.sequence_packer import StreamingSequencePacker
from shards.shard_writer import PackedShardWriter
from tokenizer.tokenizer import BPETokenizer

BATCH_SIZE = 5000
manifest = load_pretrain_manifest()
    
def process_document(tokenizer: BPETokenizer, writer: PackedShardWriter, split: str, manifest: dict[str, str]):

    statistics = defaultdict(int)
        
    eos_id = tokenizer.token_to_id("<|eos|>")

    if eos_id is None:
        raise RuntimeError("EOS is missing from tokenizer!")

    packer = StreamingSequencePacker(CONFIG.CONTEXT_LENGTH)

    file_paths = sorted((CONFIG.RAW_PRETRAIN_DIR).glob("*.parquet"))

    for file_path in file_paths:
        print(f"Processing {file_path.name}...")

        file_documents = 0
        file_tokens = 0
        file_sequences = 0

        texts = []

        for row in tqdm(iter_pretrain_rows(file_path, split = split, manifest = manifest), desc = f"Processing {file_path.name}", leave = False, unit = "row"):

            texts.append(row["text"])

            if len(texts) >= BATCH_SIZE:

                batch_ids = tokenizer.encode_batch(texts)

                for token_ids in batch_ids:

                    token_ids.append(eos_id)
                    statistics["documents"] += 1
                    file_documents += 1

                    token_count = len(token_ids)
                    statistics["tokenized_tokens"] += token_count
                    file_tokens += token_count

                    for packed_tokens, _ in packer.add(token_ids):
                        writer.add(packed_tokens)
                        statistics["sequences"] += 1
                        file_sequences += 1

                texts.clear()

        if texts:
            batch_ids = tokenizer.encode_batch(texts)

            for token_ids in batch_ids:

                token_ids.append(eos_id)
                statistics["documents"] += 1
                file_documents += 1

                token_count = len(token_ids)
                statistics["tokenized_tokens"] += token_count
                file_tokens += token_count

                for packed_tokens, _ in packer.add(token_ids):
                    writer.add(packed_tokens)
                    statistics["sequences"] += 1
                    file_sequences += 1

        print(f"Documents: {file_documents}")
        print(f"Tokens: {file_tokens}")
        print(f"Sequences: {file_sequences}")

    statistics["discarded_tokens"] = packer.discard_remainder()
            
    return dict(statistics)

def save_statistics(statistics: dict, output_path: Path):
    
    with output_path.open("w", encoding = "utf-8") as f:
        json.dump(
            statistics,
            f,
            indent = 4
        )

def main():

    if CONFIG.DATASET_MODE != "pretrain":
        raise RuntimeError("DATASET_MODE must be pretrain to create pretrain shards!")

    CONFIG.TRAIN_PRETRAIN_DIR.mkdir(parents = True, exist_ok = True)
    CONFIG.VALID_PRETRAIN_DIR.mkdir(parents = True, exist_ok = True)

    tokenizer = BPETokenizer(CONFIG.TOKENIZER_PATH)

    print("=" * 80)
    print(f"Creating {CONFIG.DATASET_MODE} Binary shards...")

    for split in ("train", "valid"):

        split_dir = CONFIG.PROCESSED_DATA / split / "pretrain"

        with PackedShardWriter(
            output_dir = split_dir,
            split = split,
            tokens_per_shard = CONFIG.TOKENS_PER_SHARD,
            context_length = CONFIG.CONTEXT_LENGTH,
            dataset_mode = CONFIG.DATASET_MODE
        ) as writer:

            statistics = process_document(
                tokenizer = tokenizer,
                writer = writer,
                split = split,
                manifest = manifest
            )

    statistics_path = CONFIG.TRAIN_PRETRAIN_DIR / "statistics.json"
    save_statistics(statistics, statistics_path)
    
    print("=" * 70)
    print("Pretrain Binary Shards Creation Signing off!")

    print(f"Documents: {statistics['documents']:,}")
    print(f"Tokens: {statistics['tokenized_tokens']:,}")
    print(f"Sequences: {statistics['sequences']:,}")

if __name__ == "__main__":
    main()