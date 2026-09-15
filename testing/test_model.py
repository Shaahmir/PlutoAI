import torch
import torch.nn as nn
from config import CONFIG
from model.gpt import GPT

def count_parameters(model: nn.Module):
    return sum(parameter.numel() for parameter in model.parameters())

def count_trainable_parameters(model: nn.Module):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

def parameter_breakdown(model: GPT):

    breakdown = {
        "token_embedding": sum(parameter.numel() for parameter in model.token_embedding.parameters()),
        "transformers": sum(parameter.numel() for parameter in model.transformer.parameters())
    }

    if not CONFIG.USE_WEIGHT_TYING:
        breakdown["lm_head"] = sum(parameter.numel() for parameter in model.lm_head.parameters())

    return breakdown

def main():

    device = CONFIG.DEVICE
    batch_size = CONFIG.BATCH_SIZE
    sequence_length = CONFIG.CONTEXT_LENGTH

    model = GPT().to(device)
    model.eval()

    input_ids = torch.randint(
        0,
        CONFIG.VOCAB_SIZE,
        (batch_size, sequence_length),
        device = device
    )

    with torch.no_grad():
        logits, _ = model(input_ids)

    parameters = count_parameters(model)
    trainable_parameters = count_trainable_parameters(model)
    breakdown = parameter_breakdown(model)

    print("Model Test")
    print(f"Device: {device}")
    print(f"Vocabulary Size:  {CONFIG.VOCAB_SIZE:,}")
    print(f"CONTEXT LENGTH :  {CONFIG.CONTEXT_LENGTH:,}")
    print(f"Layers:  {CONFIG.N_LAYERS:,}")
    print(f"Hidden Size:  {CONFIG.D_MODEL:,}")
    print(f"Attention Heads:  {CONFIG.N_HEADS:,}")
    print(f"Head Dimension:  {CONFIG.HEAD_DIM:,}")
    print(f"SwiGGLU Dimensions:  {CONFIG.D_FF:,}")
    print(f"RoPE Theta:  {CONFIG.ROPE_THETA:,}")
    print(f"Input Shape: {tuple(input_ids.shape)}")
    print(f"Logits Shape: {tuple(logits.shape)}")
    print(f"Parameters: {parameters / 1e6:.2f}M")
    print(f"Trainable Parameters: {trainable_parameters / 1e6:.2f}M")

    for name, count in breakdown.items():
        print(f"{name.replace('_', ' ').capitalize()}: {count:,}")

    expected_shape = batch_size, sequence_length, CONFIG.VOCAB_SIZE
    assert logits.shape == expected_shape

    print("Model look good!")

if __name__ == "__main__":
    main()
