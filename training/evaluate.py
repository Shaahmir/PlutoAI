import math
import torch
import torch.distributed as dist

@torch.no_grad()
def evaluate(model, dataloader, device: torch.device, max_steps: int | None = None) -> tuple[float, int]:

    model.eval()

    local_loss_sum = torch.zeros(
        1,
        device = device,
        dtype = torch.float64
    )

    local_token_count = torch.zeros(
        1,
        device = device,
        dtype = torch.float64
    )

    for step, batch in enumerate(dataloader):

        if max_steps is not None and step >= max_steps:
            break

        input_ids = batch["input_ids"].to(device, non_blocking = True)
        labels = batch["labels"].to(device, non_blocking = True)
        attention_mask = batch.get("attention_mask")

        if attention_mask is not None:
            attention_mask = attention_mask.to(device, non_blocking = True)

        logits, loss = model(
            input_ids,
            labels,
            attention_mask = attention_mask
        )

        if loss is None:
            raise RuntimeError("Model did not return loss!")

        valid_tokens = (labels != -100).sum()

        local_loss_sum += loss.detach().double() * valid_tokens
        local_token_count += valid_tokens

    if dist.is_available() and dist.is_initialized():

        dist.all_reduce(
            local_loss_sum,
            op = dist.ReduceOp.SUM
        )

        dist.all_reduce(
            local_token_count,
            op = dist.ReduceOp.SUM
        )

    total_tokens = local_token_count.item()

    if total_tokens == 0:
        raise RuntimeError("Validation contains zero valid tokens")

    mean_loss = local_loss_sum.item() / total_tokens

    return (
        float(mean_loss),
        int(total_tokens)
    )

def perplexity(loss: float):

    if loss > 20.0:
        return float("inf")

    return math.exp(loss)
