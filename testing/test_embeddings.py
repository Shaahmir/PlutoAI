import torch
from model.embeddings import TokenEmbedding
from config import CONFIG

def main():

    model = TokenEmbedding()

    input_ids = torch.randint(
        0,
        CONFIG.VOCAB_SIZE,
        (2, CONFIG.CONTEXT_LENGTH)
    )

    output = model(input_ids)

    print(f"Input Shape: {tuple(input_ids.shape)}")
    print(f"Output Shape: {tuple(output.shape)}")
    print(f"Output dtype: {output.dtype}")

    expected_shape = 2, CONFIG.CONTEXT_LENGTH, CONFIG.D_MODEL
    assert output.shape == expected_shape

    print("Embedding look good!")

if __name__ == "__main__":
    main()
