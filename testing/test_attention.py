import torch
from config import CONFIG
from model.attention import GQA

def main():

    device = CONFIG.DEVICE
    attention = GQA().to(device)
    
    batch_size = CONFIG.BATCH_SIZE
    sequence_length = CONFIG.CONTEXT_LENGTH

    x = torch.randn(
        batch_size,
        sequence_length,
        CONFIG.D_MODEL,
        device = device
    )

    with torch.no_grad():
        output = attention(x)

    print(f"Input Shape: {tuple(x.shape)}")
    print(f"Output Shape: {tuple(output.shape)}")
    print(f"Q Heads: {CONFIG.N_HEADS}")
    print(f"KV Heads: {CONFIG.N_KV_HEADS}")
    print(f"Head Dimension: {CONFIG.HEAD_DIM}")
    print(f"GQA Replication: {CONFIG.N_REP}")

    assert output.shape == x.shape
    assert CONFIG.N_HEADS % CONFIG.N_KV_HEADS == 0

    print("GQA look good!")

if __name__ == "__main__":
    main()
