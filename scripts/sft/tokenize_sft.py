from scripts.sft.sft_serializer import serialize_sft_example
from tokenizer.tokenizer import BPETokenizer

def tokenize_example(example, tokenizer: BPETokenizer) -> tuple[list[int], list[int]]:

    user_text, assistant_text = serialize_sft_example(
        example
    )

    user_ids = tokenizer.encode(
        user_text
    )

    assistant_ids = tokenizer.encode(
        assistant_text
    )

    token_ids = user_ids + assistant_ids

    if not token_ids:
        raise ValueError("SFT example produced no tokens.")

    loss_mask = [0] * len(user_ids) + [1] * len(assistant_ids)

    return (
        token_ids,
        loss_mask
    )
