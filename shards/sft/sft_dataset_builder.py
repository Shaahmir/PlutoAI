import json
import random
from dataclasses import dataclass
from typing import Iterable, Sequence

from pathlib import Path
from config import CONFIG
from tqdm import tqdm

from scripts.sft.raw_sequence_dataset import RawPackedDataset
from scripts.sft.sft_data import SFTData
from scripts.sft.tokenize_sft import tokenize_example
from scripts.sft.sft_processor import fit_sequence

from shards.shard_writer import PackedShardWriter
from shards.sft.sft_split import stable_bucket, create_example_id

from tokenizer.tokenizer import BPETokenizer
from concurrent.futures import ProcessPoolExecutor

MAX_WORKERS = 8
_WORKER_TOKENIZER = None

def _init_worker(tokenizer_path):
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = BPETokenizer(tokenizer_path)

def _encode_worker(example):
    global _WORKER_TOKENIZER
    return encode_example(
        example,
        _WORKER_TOKENIZER
    )

@dataclass(frozen = True)
class SourceItem:
    source: str
    example: object

@dataclass
class TokenizedItem:
    source: str
    token_ids: list[int]
    loss_mask: list[int]

def handle_multiturn(messages: list[dict[str, str]], tokenizer: BPETokenizer, max_len: int) -> tuple[list[int], list[int]]:

    history = list(messages)

    while history and history[-1]["role"] != "assistant":
        history.pop()

    if not history:
        return (
            [],
            []
        )

    start_head = CONFIG.SPECIAL_TAGS["START_HEAD"]
    end_head = CONFIG.SPECIAL_TAGS["END_HEAD"]
    eot = CONFIG.SPECIAL_TAGS["EOT"]
    eos = CONFIG.SPECIAL_TAGS["EOS"]
    start_think = CONFIG.SPECIAL_TAGS["THINK"]
    end_think = CONFIG.SPECIAL_TAGS["END_THINK"]

    token_ids: list[int] = []
    loss_mask: list[int] = []

    while len(history) >= 2:

        token_ids = []
        loss_mask = []

        for index, message in enumerate(history):

            role = message["role"]
            content = message.get("content", "").strip()
            reasoning = message.get("reasoning", "").strip()

            is_last = (index == len(history) - 1)

            if role == "assistant" and reasoning:
                body = f"{start_think}\n{reasoning}\n{end_think}\n{content}"

            else:
                body = content

            turn_text = f"{start_head}{role}{end_head}\n{body}\n{eot}\n"

            if is_last and role == "assistant":
                turn_text += f"{eos}"

            
            turn_tokens = tokenizer.encode(turn_text)
            token_ids.extend(turn_tokens)

            if role == "assistant":
                loss_mask.extend([1] * len(turn_tokens))

            else:
                loss_mask.extend([0] * len(turn_tokens))

        if len(token_ids) <= max_len:
            return (
                token_ids,
                loss_mask
            )

        has_system = (history[0]["role"] == "system")

        if has_system and len(history) >= 3 and history[1]["role"] == "user" and history[2]["role"] == "assistant":
            history = [history[0]] + history[3:]

        elif not has_system and len(history) >= 2 and history[0]["role"] == "user" and history[1]["role"] == "assistant":
            history = history[2:]

        else:
            pop_index = 1 if (has_system and len(history) > 1) else 0
            history.pop(pop_index)

    fitted = fit_sequence(
        token_ids,
        loss_mask
    )

    if fitted is not None:
        return fitted

    return (
        [],
        []
    )

def deterministic_split(examples: Sequence[object], validation_fraction, seed: int) -> tuple[list[object], list[object]]:

    ranked = []

    for example in examples:
        key = create_example_id(example)
        score = stable_bucket(key, seed)
        ranked.append((score, key, example))
    
    ranked.sort(key = lambda x: (x[0], x[1]))
    valid_count = max(1, int(round(len(ranked) * validation_fraction))) if len(ranked) > 0 else 0
    
    valid = [item[2] for item in ranked[:valid_count]]
    train = [item[2] for item in ranked[valid_count:]]

    return train, valid

def encode_example(example: SFTData, tokenizer: BPETokenizer) -> tuple[list[int], list[int]]:

    if getattr(example, "messages", None):
        token_ids, loss_mask = handle_multiturn(
            example.messages,
            tokenizer,
            CONFIG.CONTEXT_LENGTH
        )

    elif isinstance(example, dict) and "messages" in example:
        token_ids, loss_mask = handle_multiturn(
            example["messages"],
            tokenizer,
            CONFIG.CONTEXT_LENGTH
        )

    else:
        token_ids, loss_mask = tokenize_example(
            example,
            tokenizer
        )

    if len(token_ids) != len(loss_mask):
        raise RuntimeError("token_ids length must match loss_mask length!")

    return token_ids, loss_mask

def validate_tokenized(token_ids: Sequence[int], loss_mask: Sequence[int]):

    if len(token_ids) > CONFIG.CONTEXT_LENGTH:
        raise RuntimeError(f"SFT example exceeded context length! {len(token_ids)} > {CONFIG.CONTEXT_LENGTH}")

    if any(value not in (0, 1) for value in loss_mask):
        raise RuntimeError("Invalid loss_mask found!")

def shuffle_example(examples: Sequence[object], seed: int) -> list[object]:

    items = list(examples)

    rng = random.Random(seed)
    rng.shuffle(items)

    return items

def tokenized_pool(source: str, examples: Sequence[object], tokenizer: BPETokenizer) -> list[TokenizedItem]:

    result: list[TokenizedItem] = []
    skipped_empty = 0
    skipped_oversized = 0

    with ProcessPoolExecutor(max_workers = MAX_WORKERS, initializer = _init_worker, initargs = (CONFIG.TOKENIZER_PATH,)) as pool:

        encoded_batch = pool.map(_encode_worker, [example for example in examples], chunksize = 500)
    
        for token_ids, loss_mask in tqdm(encoded_batch, desc = f"Tokenizing {source}", total = len(examples)):

            if len(token_ids) == 0:
                skipped_empty += 1
                continue

            if len(token_ids) > CONFIG.CONTEXT_LENGTH:
                skipped_oversized += 1
                continue

            validate_tokenized(
                token_ids,
                loss_mask
            )

            if sum(loss_mask) == 0:
                skipped_unsupervised += 1
                continue

            result.append(
                TokenizedItem(
                    source = source,
                    token_ids = token_ids,
                    loss_mask = loss_mask
                )
            )

    print(f"[{source}] tokenized [{len(result)}] examples and skipped empty: {skipped_empty} oversized: {skipped_oversized}")

    return result

def source_budget(total_tokens: int) -> dict[str, int]:

    ratios = {
        "Dataset1": CONFIG.SFT_D1_TOKEN_RATIO,
        "Dataset2": CONFIG.SFT_D2_TOKEN_RATIO,
        "Replay": CONFIG.SFT_REPLAY_TOKEN_RATIO
    }


    budgets = {
        source: int(total_tokens * ratio) for source, ratio in ratios.items()
    }

    remainder = total_tokens - sum(budgets.values())
    budgets["Dataset1"] += remainder

    return budgets

def budget_sampler(pool: Sequence[TokenizedItem], token_budget: int, seed: int, allow_reuse: bool, max_reuse: int | None = None) -> list[TokenizedItem]:

    if token_budget <= 0:
        return []

    if len(pool) == 0:
        raise RuntimeError("Cannot sample from empty source pool!")

    if max_reuse is not None and max_reuse < 1:
        raise ValueError("max reuse must be >= 1")

    order = list(range(len(pool)))

    rng = random.Random(seed)
    rng.shuffle(order)

    reuse  = [0] * len(pool)
    output: list[TokenizedItem] = []
    total = 0
    cursor = 0
    min_item_tokens = min(len(item.token_ids) for item in pool if len(item.token_ids) > 0)

    with tqdm(total = token_budget, desc = "Sampling budget", unit = "tok") as pbar:
        while total < token_budget:

            remaining = token_budget - total

            if remaining < min_item_tokens:
                break

            if cursor >= len(order):

                if not allow_reuse:
                    break

                if max_reuse is not None and all(count >= max_reuse for count in reuse):
                    break

                rng.shuffle(order)
                cursor = 0

            idx = order[cursor]
            cursor += 1

            if max_reuse is not None and reuse[idx] >= max_reuse:
                continue

            item = pool[idx]
            item_tokens = len(item.token_ids)

            if item_tokens <= 0 or item_tokens > remaining:
                continue

            output.append(item)
            reuse[idx] += 1
            total += item_tokens
            pbar.update(item_tokens)

    return output

def shard_writer(items: Iterable[TokenizedItem], split: str,  output_dir: Path, dataset_name: str) -> dict:

    output_dir.mkdir(parents = True, exist_ok = True)

    writer = PackedShardWriter(
        output_dir = output_dir,
        split = split,
        tokens_per_shard = CONFIG.TOKENS_PER_SHARD,
        context_length = CONFIG.CONTEXT_LENGTH,
        dataset_mode = CONFIG.DATASET_MODE
    )

    counts: dict[str, int] = {}
    token_counts: dict[str, int] = {}
    supervised_counts: dict = {}

    total_examples = 0
    total_tokens = 0
    total_supervised = 0

    for item in items:
        
        writer.add_tokens(
            item.token_ids,
            item.loss_mask
        )

        n = len(item.token_ids)
        s = sum(item.loss_mask)

        counts[item.source] = counts.get(item.source, 0) + 1
        token_counts[item.source] = token_counts.get(item.source, 0) + n
        supervised_counts[item.source] = supervised_counts.get(item.source, 0) + s

        total_examples += 1
        total_tokens += n
        total_supervised += s

    writer.close()

    statistics = {
        "dataset": dataset_name,
        "examples": total_examples,
        "tokens": total_tokens,
        "supervised_tokens": total_supervised,
        "source_examples": counts,
        "source_tokens": token_counts,
        "source_supervised_tokens": supervised_counts
    }

    print()
    print(f"{dataset_name} build complete!")
    print(f"Examples: {total_examples:,}")
    print(f"Tokens: {total_tokens:,}")

    for source in sorted(counts):
        print(f"{source:10s} | examples = {counts[source]:,} | tokens = {token_counts[source]:,}")

    return statistics

def deterministic_valid_items(pool: Sequence[TokenizedItem], budget: int, seed: int) -> list[TokenizedItem]:
    return budget_sampler(pool, budget, seed, allow_reuse = False)

def load_valid_replay() -> list[TokenizedItem]:

    dataset = RawPackedDataset(
        split_dir = CONFIG.VALID_PRETRAIN_DIR
    )

    if len(dataset) == 0 or CONFIG.SFT_REPLAY_VALID == 0:
        return []

    indices = list(range(len(dataset)))
    random.Random(CONFIG.SEED + 1).shuffle(indices)

    result: list[TokenizedItem] = []
    limit = min(CONFIG.SFT_REPLAY_VALID, len(dataset))

    for index in indices[:limit]:

        tokens = dataset[index]

        if hasattr(tokens, "tolist"):
            tokens = tokens.tolist()

        if len(tokens) != CONFIG.CONTEXT_LENGTH:
            raise RuntimeError(f"Unexpected pretrain replay length: {len(tokens)}")

        loss_mask = [1] * len(tokens)

        result.append(
            TokenizedItem(
                source = "Replay",
                token_ids = list(tokens),
                loss_mask = loss_mask
            )
        )

    print(f"Pretrain valid replay sequences: {len(result):,}")
    return result

def build_sft(
    d1_examples: Sequence[SFTData],
    d2_examples: Sequence[SFTData],
    replay_train_pool: Sequence[TokenizedItem],
    tokenizer: BPETokenizer
):

    print("=" * 50)
    print("SFT - Deterministic Data Split")

    
    d1_train, d1_valid = deterministic_split(
        d1_examples,
        CONFIG.SFT_VALIDATION_SPLIT,
        CONFIG.SEED
    )
    
    d2_train, d2_valid = deterministic_split(
        d2_examples,
        CONFIG.SFT_VALIDATION_SPLIT,
        CONFIG.SEED + 1
    )

    print(f"Dataset 1: train = {len(d1_train):,} valid = {len(d1_valid):,}")
    print(f"Dataset 2: train = {len(d2_train):,} valid = {len(d2_valid):,}")

    for name, pool in [("Dataset1", d1_train), ("Dataset2", d2_train)]: #("Dataset3", d3_train)
        if len(pool) == 0:
            raise RuntimeError(f"{name} train split is empty")

    # Training

    print()
    print("=" * 50)
    print("SFT - Tokenizing Train Pool")

    d1_train_pool = tokenized_pool(
        "Dataset1",
        d1_train,
        tokenizer
    )

    d2_train_pool = tokenized_pool(
        "Dataset2",
        d2_train,
        tokenizer
    )

    budgets = source_budget(
        CONFIG.SFT_TOKENS
    )

    print()
    print("=" * 50)
    print("SFT - Train Token Budgets")

    for source, budget in budgets.items():
        print(f"{source:10s}: {budget:,} tokens")

    selected = []

    selected.extend(
        budget_sampler(
            d1_train_pool,
            budgets["Dataset1"],
            CONFIG.SEED,
            allow_reuse = False
        )
    )

    selected.extend(
        budget_sampler(
            d2_train_pool,
            budgets["Dataset2"],
            CONFIG.SEED + 1,
            allow_reuse = False
        )
    )

    selected.extend(
        budget_sampler(
            replay_train_pool,
            budgets["Replay"],
            CONFIG.SEED + 3,
            allow_reuse = False
        )
    )

    shortfall = CONFIG.SFT_TOKENS - sum(len(x.token_ids) for x in selected)

    if shortfall > 0:

        extra_d1 = budget_sampler(
            d1_train_pool,
            shortfall // 2,
            CONFIG.SEED + 201,
            allow_reuse = True,
            max_reuse = 2
        )

        remaining = shortfall - sum(len(x.token_ids) for x in extra_d1)

        extra_d2 = budget_sampler(
            d2_train_pool,
            max(0, remaining),
            CONFIG.SEED + 202,
            allow_reuse = True,
            max_reuse = 2
        )

        selected.extend(extra_d1)
        selected.extend(extra_d2)

    rng = random.Random(CONFIG.SEED)
    rng.shuffle(selected)

    train_statistics = shard_writer(
        items = selected,
        split = "train",
        output_dir = CONFIG.TRAIN_SFT_DIR,
        dataset_name = "SFT_train" 
    )

    # Validation

    print()
    print("=" * 50)
    print("SFT - Tokenizing Valid Pool")

    d1_valid_pool = tokenized_pool(
        "Dataset1",
        d1_valid,
        tokenizer
    )

    d2_valid_pool = tokenized_pool(
        "Dataset2",
        d2_valid,
        tokenizer
    )

    replay_valid_pool = load_valid_replay()

    valid_budgets = {
        "Dataset1": int(CONFIG.SFT_VALID_TOKENS * CONFIG.SFT_D1_TOKEN_RATIO),
        "Dataset2": int(CONFIG.SFT_VALID_TOKENS * CONFIG.SFT_D2_TOKEN_RATIO)
    }

    valid_items = []

    valid_items.extend(
        deterministic_valid_items(
            d1_valid_pool,
            valid_budgets["Dataset1"],
            CONFIG.SEED
        )
    )

    valid_items.extend(
        deterministic_valid_items(
            d2_valid_pool,
            valid_budgets["Dataset2"],
            CONFIG.SEED
        )
    )
    
    valid_items.extend(
        replay_valid_pool
    )

    random.Random(CONFIG.SEED).shuffle(valid_items)

    valid_statistics = shard_writer(
        valid_items,
        split = "valid",
        output_dir = CONFIG.VALID_SFT_DIR,
        dataset_name = "SFT_valid" 
    )

    valid_sets = {
        "Dataset1": deterministic_valid_items(d1_valid_pool, valid_budgets["Dataset1"], CONFIG.SEED + 101),
        "Dataset2": deterministic_valid_items(d2_valid_pool, valid_budgets["Dataset2"], CONFIG.SEED + 102),
        "Replay": replay_valid_pool
    }

    valid_dirs = {}

    for name, items in valid_sets.items():
        output = CONFIG.VALID_SFT_DIR / name
        valid_dirs[name] = str(output)
        shard_writer(
            items,
            "valid",
            output,
            f"SFT_valid_{name}"
        )

    with (CONFIG.TRAIN_SFT_DIR / "statistics.json").open("w", encoding = "utf-8") as f:
        json.dump(
            train_statistics,
            f,
            indent = 2
        )

    with (CONFIG.VALID_SFT_DIR / "statistics.json").open("w", encoding = "utf-8") as f:
        json.dump(
            valid_statistics,
            f,
            indent = 2
    )

if __name__ == "__main__":

    raise SystemError(
        "Import build_sft from your dataset build script after loading SFT dataset source"
    )
