import torch 
from config import CONFIG
from model.rmsnorm import RMSNorm

def main():

    rmsnorm = RMSNorm(
        CONFIG.D_MODEL
    )

    x = torch.randn(
        2,
        CONFIG.CONTEXT_LENGTH,
        CONFIG.D_MODEL
    )

    output = rmsnorm(x)

    print(f"Input Shape: {tuple(x.shape)}")
    print(f"Output Shape: {tuple(output.shape)}")
    print(f"Output dtype: {output.dtype}")
    print(f"Weight Shape: {tuple(rmsnorm.weight.shape)}")

    assert output.shape == x.shape
    assert rmsnorm.weight.shape == (CONFIG.D_MODEL, )

    print("RMSNorm look good!")

if __name__ == "__main__":
    main()
