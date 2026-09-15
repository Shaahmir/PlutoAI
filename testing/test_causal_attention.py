import torch
from config import CONFIG
from model.attention import GQA

def main():

    device = CONFIG.DEVICE
    sequence_length = 16
    attention = GQA().to(device)

    x = torch.randn(
        1,
        sequence_length,
        CONFIG.D_MODEL,
        device = device
    )

    x_modified = x.clone()
    x_modified[:, -1, :] += 100.0

    with torch.no_grad():
        output = attention(x)
        modified_output = attention(x_modified)
    
    difference = (output[:, :-1] - modified_output[:, :-1]).abs().max().item()
    print(f"Maximum Difference in earlier position {difference:.8e}")

    if difference > 1e-5:
        raise RuntimeError("Causal Attention failed. Earlier positions changed on future token modification!")

    print("Causal Attention looks good!")

if __name__ == "__main__":
    main()
