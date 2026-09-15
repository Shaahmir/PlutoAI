from collections import defaultdict
from config import CONFIG

from shards.sequence_packer import StreamingSequencePacker
from shards.shard_writer import PackedShardWriter

from scripts.sft.sft_processor import fit_sequence
from shards.sft.sft_dataset_builder import encode_example as encode_sft_example

from tokenizer.tokenizer import BPETokenizer

def encode_example(example, tokenizer: BPETokenizer):
    
    token_ids, loss_mask = encode_sft_example(
        example,
        tokenizer
    )

    return fit_sequence(
        token_ids = token_ids,
        loss_mask = loss_mask
    )

def add_example(
    writer: PackedShardWriter,
    packer: StreamingSequencePacker,
    token_ids: list[int],
    loss_mask: list[int],
    statistics: dict,
    source: str
):

    statistics[f"{source}_packed_tokens"] += len(token_ids)

    packed_sequences = packer.add(
        token_ids,
        loss_mask
    )

    for packed_tokens, packed_mask in packed_sequences:

        writer.add_tokens(
            packed_tokens,
            packed_mask
        )

        statistics["packed_sequences"] += 1
        statistics["packed_tokens"] += len(packed_tokens)

def save_split(output_dir, split, tokenizer: BPETokenizer, sources, replay_sampler, budgets) -> dict:

    statistics = defaultdict(int)
    consecutive_failures = defaultdict(int)

    packer = StreamingSequencePacker(
        CONFIG.CONTEXT_LENGTH
    )

    remaining = dict(
        budgets
    )

    with PackedShardWriter(
        output_dir = output_dir,
        split = split,
        tokens_per_shard = CONFIG.TOKENS_PER_SHARD,
        context_length = CONFIG.CONTEXT_LENGTH,
        dataset_mode = CONFIG.DATASET_MODE
    ) as writer:

        while any(value > 0 for value in remaining.values()):

            active = [name for name, value in remaining.items() if value > 0]

            if not active:
                break

            source = active[statistics["selection_step"] % len(active)]
            statistics["selection_step"] += 1

            if source == "replay":
                token_ids, loss_mask = replay_sampler.sample()

            else:
                examples = sources[source]

                if not examples:
                    remaining[source] = 0
                    continue

                if consecutive_failures[source] >= len(examples):
                    remaining[source] = 0
                    continue

                index = statistics[f"{source}_index"] % len(examples)
                statistics[f"{source}_index"] += 1

                tokenized = encode_example(
                    examples[index],
                    tokenizer
                )

                if tokenized is None:
                    consecutive_failures[source] += 1
                    continue

                consecutive_failures[source] = 0
                token_ids, loss_mask = tokenized

            budget = remaining[source]

            if len(token_ids) > budget:

                token_ids = token_ids[:budget]
                loss_mask = loss_mask[:budget]

            if not token_ids:
                continue

            remaining[source] -= len(token_ids)
            statistics[f"{source}_tokens"] += len(token_ids)

            add_example(
                packer = packer,
                writer = writer,
                token_ids = token_ids,
                loss_mask = loss_mask,
                statistics = statistics,
                source = source
            )

    statistics["discarded_tokens"] = packer.discard_remainder()
    return dict(statistics)
