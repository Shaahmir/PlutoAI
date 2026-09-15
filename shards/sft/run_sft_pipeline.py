import random
from pathlib import Path
from config import CONFIG

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from scripts.sft.instruct_selector import is_valid_example
from scripts.sft.instruct_processor import iter_sft_rows as iter_instruct

from scripts.sft.raw_sequence_dataset import RawPackedDataset
from scripts.sft.mix_scheduler import TokenMixScheduler

from shards.sft.sft_dataset_builder import build_sft, TokenizedItem, encode_example
from tokenizer.tokenizer import BPETokenizer

MAX_WORKERS = 8
_BATCH_ROW = 512

def _init_worker(tokenizer_path):
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = BPETokenizer(tokenizer_path)

def _encode_row_helper(row):
    return encode_example(row, _WORKER_TOKENIZER)

def _streams_rows(file_paths, iterator_factory):

    paths = list(file_paths)

    rng = random.Random(CONFIG.SEED)
    rng.shuffle(paths)

    for path in paths:
        yield from iterator_factory(path)

def load_sft_pool(file_paths, iterator_factory, target_tokens: int, source, require_no_reasoning = False):

    if not file_paths:
        raise RuntimeError(f"{source}: no source files found!")

    examples = []
    accumulated = 0
    rows_seen = 0
    rows_kept = 0
    batch = []

    pbar = tqdm(
        total = target_tokens,
        desc = f"Loading {source}",
        unit = "tok",
        unit_scale = True,
        dynamic_ncols = True
    )

    def flush(pool_executer):

        nonlocal accumulated, rows_kept

        if not batch:
            return

        encoded = pool_executer.map(
            _encode_row_helper,
            batch,
            chunksize = 32
        )

        for row, (token_ids, loss_mask) in zip(batch, encoded):

            if accumulated >= target_tokens:
                break

            if not token_ids or len(token_ids) > CONFIG.CONTEXT_LENGTH:
                continue

            if not any(loss_mask):
                continue

            examples.append(row)
            accumulated += len(token_ids)
            rows_kept += 1

            pbar.update(len(token_ids))
            pbar.set_postfix(rows_seen = rows_seen, rows_kept = rows_kept, refresh = False)

    with ProcessPoolExecutor(max_workers = MAX_WORKERS, initializer = _init_worker, initargs = (CONFIG.TOKENIZER_PATH,)) as pool:

        for row in _streams_rows(file_paths, iterator_factory):

            if accumulated >= target_tokens:
                break

            if not is_valid_example(row):
                continue

            if require_no_reasoning and getattr(row, "reasoning", ""):
                continue

            batch.append(row)
            rows_seen += 1

            if len(batch) >= _BATCH_ROW:
                flush(pool)
                batch.clear()

        flush(pool)

    pbar.close()

    if not examples:
        raise RuntimeError(f"{source}: yields zero usable examples!")

    print(f"{source}: row_checked = {rows_seen:,} selected = {rows_kept:,} tokens = {accumulated:,}")
    return examples

def load_train_replay_pool(target_tokens: int) -> list[TokenizedItem]:

    print("Loading pretrain replay pool ...")

    dataset = RawPackedDataset(
        split_dir = CONFIG.TRAIN_PRETRAIN_DIR
    )

    if len(dataset) == 0:
        raise RuntimeError("Pretrain dataset is empty!")

    indices = list(range(len(dataset)))

    rng = random.Random(CONFIG.SEED + 1)
    rng.shuffle(indices)

    pool: list[TokenizedItem] = []
    total = 0

    pbar = tqdm(
        total = target_tokens,
        desc = f"Loading Replay",
        unit = "tok",
        unit_scale = True,
        dynamic_ncols = True
    )

    for index in indices:
        
        if total >= target_tokens:
            break

        tokens = dataset[index]
        
        if hasattr(tokens, "tolist"):
            tokens = tokens.tolist()

        tokens = list(tokens)

        if not tokens:
            continue

        take = min(len(tokens), target_tokens - total)
        tokens = tokens[:take]

        pool.append(
            TokenizedItem(
                source = "Replay",
                token_ids = tokens,
                loss_mask = [1] * len(tokens)
            )
        )

        total += len(tokens)
        pbar.update(len(tokens))

    pbar.close()

    if total == 0:
        raise RuntimeError("Replay Pool produced zero tokens!")

    print(f"Loaded {len(pool):,} replay items with {total:,} tokens into training pool.")
    return pool

def main():

    print("=" * 50)
    print("SFT Shards Builder")

    # Token Mix Scheduler

    scheduler = TokenMixScheduler(
        total_tokens = CONFIG.SFT_TOKENS,
        ratios = {
            "d1": CONFIG.SFT_D1_TOKEN_RATIO,
            "d2": CONFIG.SFT_D2_TOKEN_RATIO,
            # "d3": CONFIG.SFT_D3_TOKEN_RATIO,
            "replay": CONFIG.SFT_REPLAY_TOKEN_RATIO
        },
        seed = CONFIG.SEED
    )

    # Budgets

    budgets = scheduler.all_budgets()

    print("Token Budgets: ")
    
    for name, budget in budgets.items():
        print(f"- {name.capitalize()}: {budget:,}")

    RAW_SFT_DIR = CONFIG.RAW_SFT_DIR

    # Tokenizer
    
    tokenizer = BPETokenizer(
        CONFIG.TOKENIZER_PATH
    )

    # Dataset 1

    d1_files_path = sorted(Path(RAW_SFT_DIR / "dataset1").glob("*.parquet"))
    d1_target = int(budgets["d1"] / (1.0 - CONFIG.SFT_VALIDATION_SPLIT))
    d1_examples = load_sft_pool(d1_files_path, iter_instruct, d1_target, "Dataset1")

    # Dataset 2

    d2_files_path = sorted(Path(RAW_SFT_DIR / "dataset2").glob("*.parquet"))
    d2_target = int(budgets["d2"] / (1.0 - CONFIG.SFT_VALIDATION_SPLIT))
    d2_examples = load_sft_pool(d2_files_path, iter_instruct, d2_target, "Dataset2")

    # Pretrain Replay

    replay_train_pool = load_train_replay_pool(
        target_tokens = budgets["replay"]
    )

    # Shard Building

    build_sft(
        d1_examples = d1_examples,
        d2_examples = d2_examples,
        replay_train_pool = replay_train_pool,
        tokenizer = tokenizer
    )

    print("SFT Shards builder signing off!")

if __name__ == "__main__":
    main()
