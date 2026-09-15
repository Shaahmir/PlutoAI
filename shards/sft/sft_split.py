import hashlib
from scripts.sft.sft_data import SFTData

def stable_bucket(value: str, seed: int) -> int:

    value = (
        f"{seed}|{value}"
    ).encode("utf-8")

    digest = hashlib.sha256(
        value
    ).digest()

    integer = int.from_bytes(
        digest[:8],
        byteorder = "big",
        signed = False
    )

    return integer

def deterministic_validation(key: str, validation_ratio, seed) -> bool:

    value = (
        f"{seed}:{key}"
    ).encode("utf-8")

    digest = hashlib.sha256(
        value
    ).digest()

    integer = int.from_bytes(
        digest[:8],
        byteorder = "big",
        signed = False
    )

    bucket = integer / 2 ** 64

    return bucket < validation_ratio

def create_example_id(example: SFTData) -> str:

    if isinstance(example, dict):
        metadata = example.get("metadata") or {}
        source = example.get("source", "")
        prompt = example.get("prompt", "")
        response = example.get("response", "")
        reasoning = example.get("reasoning", "")
        messages = example.get("messages", "")

    else:
        metadata = getattr(example, "metadata", None) or {}
        source = getattr(example, "source", "")
        prompt = getattr(example, "prompt", "")
        response = getattr(example, "response", "")
        reasoning = getattr(example, "reasoning", "")
        messages = getattr(example, "messages", "")

    keys = (
        "id",
        "conversation_id",
        "conversation_hash",
        "uuid",
        "uid"
    )

    if isinstance(metadata, dict):

        for key in keys:
            value = metadata.get(key)
            if value is not None:
                return f"{source}:{value}"

    elif metadata is not None:

        for key in keys:
            if hasattr(metadata, key):
                value = getattr(metadata, key)
                if value is not None:
                    return f"{source}:{value}"


    if messages:
        content = f"{source}\n{messages}\n"

    else:
        content = (
            f"{source}\n"
            f"{prompt}\n"
            f"{response}\n"
            f"{reasoning or ''}\n"
        )

    digest = hashlib.sha256(
        content.encode(
            "utf-8"
        )
    ).hexdigest()

    return f"{source}:{digest}"
