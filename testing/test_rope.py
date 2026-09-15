import torch 
from config import CONFIG
from model.rope import RotaryEmbedding

def main():

    rope = RotaryEmbedding(
        head_dim = CONFIG.HEAD_DIM,
        max_position_embeddings = CONFIG.CONTEXT_LENGTH,
        theta = CONFIG.ROPE_THETA
    )

    q = torch.randn(
        CONFIG.BATCH_SIZE,
        CONFIG.N_HEADS,
        CONFIG.CONTEXT_LENGTH,
        CONFIG.HEAD_DIM
    )

    k = torch.randn(
        CONFIG.BATCH_SIZE,
        CONFIG.N_KV_HEADS,
        CONFIG.CONTEXT_LENGTH,
        CONFIG.HEAD_DIM
    )

    rotated_q, rotated_k = rope(q, k)

    print(f"Q Input: {tuple(q.shape)}")
    print(f"K Input: {tuple(k.shape)}")
    print(f"Q Output: {tuple(rotated_q.shape)}")
    print(f"K Output: {tuple(rotated_k.shape)}")

    assert rotated_q.shape == q.shape
    assert rotated_k.shape == k.shape

    print("RoPE look good!")

if __name__ == "__main__":
    main()
