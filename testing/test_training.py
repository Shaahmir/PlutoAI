import torch
from config import CONFIG
from model.gpt import GPT

def main():

    device = CONFIG.DEVICE
    model = GPT().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = CONFIG.LEARNING_RATE,
        betas = CONFIG.ADAM_BETAS,
        eps = CONFIG.ADAM_EPS,
        weight_decay = CONFIG.WEIGHT_DECAY
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled = CONFIG.USE_AMP
    )

    input_ids = torch.randint(
        0,
        CONFIG.VOCAB_SIZE,
        (CONFIG.BATCH_SIZE, CONFIG.CONTEXT_LENGTH - 1),
        device = device
    )

    labels = torch.randint(
        0,
        CONFIG.VOCAB_SIZE,
        (CONFIG.BATCH_SIZE, CONFIG.CONTEXT_LENGTH - 1),
        device = device
    )

    optimizer.zero_grad(
        set_to_none = True
    )

    with torch.autocast(
        device_type = "cuda",
        dtype = CONFIG.AMP_DTYPE,
        enabled = CONFIG.USE_AMP
    ):
        logits, loss = model(
            input_ids,
            labels
        )

    if loss is None:
        raise RuntimeError("Loss was not returned!")

    if CONFIG.USE_AMP:

        scaler.scale(
            loss
        ).backward()

        scaler.unscale_(
            optimizer
        )

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            CONFIG.MAX_GRAD_NORM
        )

        scaler.step(
            optimizer
        )

        scaler.update()

    else:

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            CONFIG.MAX_GRAD_NORM
        )

        optimizer.step()

    print(f"Loss: {loss.item():.6f}")
    print("Training Looks Good!")

if __name__ == "__main__":
    main()
