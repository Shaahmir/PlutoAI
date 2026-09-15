from config import CONFIG
from model.gpt import GPT

def main():

    device = CONFIG.DEVICE
    model = GPT().to(device)

    embedding_weight = model.token_embedding.embedding.weight
    lm_head_weight = model.lm_head.weight

    print(f"Embedding shape: {tuple(embedding_weight.shape)}")
    print(f"LM Head shape: {tuple(lm_head_weight.shape)}")
    print(f"Same Storage: {embedding_weight.data_ptr() == lm_head_weight.data_ptr()}")

    assert embedding_weight.data_ptr() == lm_head_weight.data_ptr()
    print("Weight Tying looks good!")

if __name__ == "__main__":
    main()
