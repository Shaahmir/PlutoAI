import random
import hashlib
from typing import Iterator
from config import CONFIG
from scripts.sft.instruct_processor import iter_sft_rows
from scripts.sft.sft_data import SFTData

MIN_PROMPT_LEN = 1
MIN_RESPONSE_LEN = 1

think_tags = ("<think>", "</think>")

def stable_score(example: SFTData) -> float:

    text = example.prompt + "\n" +  example.response

    digest = hashlib.sha256(
        text.encode("utf-8")
    ).digest()

    value = int.from_bytes(
        digest[:8],
        byteorder = "big",
        signed = False
    )

    return value / 2 ** 64

def is_valid_example(example: SFTData) -> bool:

    prompt = example.prompt.strip()
    response = example.response.strip()

    if len(prompt) < MIN_PROMPT_LEN:
        return False

    if len(response) < MIN_RESPONSE_LEN:
        return False

    if any(tag in prompt for tag in think_tags):
        return False

    if any(tag in response for tag in think_tags):
        return False

    return True

def select_instruct(file_path, target_examples: int) -> Iterator[SFTData]:

    candidates = [
        example for example in iter_sft_rows(file_path) if is_valid_example(example)
    ]

    rng = random.Random(CONFIG.SEED)
    rng.shuffle(candidates)

    for example in candidates[:target_examples]:
        yield example
