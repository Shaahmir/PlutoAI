from config import CONFIG

def fit_sequence(token_ids: list[int], loss_mask: list[int]) -> tuple[list[int], list[int]] | None:

    if len(token_ids) != len(loss_mask):
        raise ValueError("token_ids and loss_mask must have equal lengths!")

    context_length = CONFIG.CONTEXT_LENGTH

    if len(token_ids) <= context_length:
        return token_ids, loss_mask
        
    first_training = next((idx for idx, value in enumerate(loss_mask) if value == 1), None)

    if first_training is None:
        return None

    assistant_tokens = token_ids[first_training:]
    assistant_mask = loss_mask[first_training:]

    if len(assistant_tokens) >= context_length:

        return (
            assistant_tokens[-context_length:],
            assistant_mask[-context_length:]
        )

    remaining = context_length - len(assistant_tokens)

    user_tokens = token_ids[max(0, first_training - remaining) : first_training]
    user_mask = loss_mask[max(0, first_training - remaining) : first_training]

    return (
        user_tokens + assistant_tokens,
        user_mask + assistant_mask
    )
