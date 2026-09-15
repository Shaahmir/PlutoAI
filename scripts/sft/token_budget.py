
def consume_to_budget(token_ids: list[int], loss_mask: list[int], remaining_budget: int) -> tuple[list[int], ...]:

    if len(token_ids) != len(loss_mask):
        raise ValueError("token_ids length must match loss_mask length!")

    if remaining_budget <= 0:
        return (
            [],
            [],
            token_ids,
            loss_mask
        )

    take = min(len(token_ids), remaining_budget)

    consumed_tokens = token_ids[:take]
    consumed_mask = loss_mask[:take]

    remaining_tokens = token_ids[take:]
    remaining_mask = loss_mask[take:]

    return (
        consumed_tokens,
        consumed_mask,
        remaining_tokens,
        remaining_mask
    )
