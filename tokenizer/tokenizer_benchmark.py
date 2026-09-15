from config import CONFIG
from tqdm import tqdm
from scripts.pretrain.pretrain_processor import iter_pretrain_rows
from tokenizer.tokenizer import BPETokenizer

def main():

    tokenizer = BPETokenizer(CONFIG.TOKENIZER_PATH)

    documents = 0
    source_tokens = 0
    bpe_tokens = 0

    max_bpe_length = 0
    file_paths = sorted((CONFIG.RAW_PRETRAIN_DIR).glob("*.parquet"))

    BATCH_SIZE = 1000

    for file_path in file_paths:

        print(f"Benchmarking {file_path.name}")

        texts = []
        counts = []

        for row in tqdm(iter_pretrain_rows(file_path), desc = f"Reading {file_path.name}", leave = False, unit = "row"):

            texts.append(row["text"])
            counts.append(int(row["token_count"]))

            if len(texts) >= BATCH_SIZE:

                lengths = [len(ids) for ids in tokenizer.encode_batch(texts)]

                documents += len(texts)
                source_tokens += sum(counts)
                bpe_tokens += sum(lengths)
                max_bpe_length = max(max_bpe_length, max(lengths))

                texts.clear()
                counts.clear()

        if texts:
            lengths = [len(ids) for ids in tokenizer.encode_batch(texts)]
            
            documents += len(texts)
            source_tokens += sum(counts)
            bpe_tokens += sum(lengths)
            max_bpe_length = max(max_bpe_length, max(lengths))

    if source_tokens == 0:
        raise RuntimeError("No tokens were processed!")
    
    compression = source_tokens / bpe_tokens
    tokens_per_word = bpe_tokens / documents

    print("=" * 50)
    print("TOKENIZER BENCHMARK")

    print(f"Documents: {documents:,}")
    print(f"Source tokenizer tokens: {source_tokens:,}")
    print(f"BPE tokens: {bpe_tokens:,}")
    print(f"Compression Ratio: {compression:,.2f}")
    print(f"Average BPE: {tokens_per_word:,}")
    print(f"Maximum BPE: {max_bpe_length:,}")

if __name__ == "__main__":
    main()