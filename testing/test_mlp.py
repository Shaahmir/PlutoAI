import torch
from config import CONFIG
from model.mlp import SwiGLU

def main():

    device = CONFIG.DEVICE
    mlp = SwiGLU().to(device)

    batch_size = CONFIG.BATCH_SIZE
    sequence_length = CONFIG.CONTEXT_LENGTH

    x = torch.randn(
        batch_size,
        sequence_length,
        CONFIG.D_MODEL,
        device = device
    )

    with torch.no_grad():
        output = mlp(x)

    print(f"Input Shape: {tuple(x.shape)}")
    print(f"Output Shape: {tuple(output.shape)}")
    print(f"FNN Dimesnions: {CONFIG.D_FF}")

    assert output.shape == x.shape

    print("SwiGLU look good!")

if __name__ == "__main__":
    main()
