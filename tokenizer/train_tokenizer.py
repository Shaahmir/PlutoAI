import gc
import tempfile

import json
import pyarrow.parquet as pq
from config import CONFIG
from pathlib import Path
from tqdm import tqdm

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer

from scripts.pretrain.pretrain_processor import iter_pretrain_text

def write_to_disk(corpus_path: Path, buffer: int = 1024 * 1024):

    line_count = 0

    with corpus_path.open("w", encoding = "utf-8", buffering = buffer) as f:

        for text in tqdm(iter_pretrain_text(split = "train"), desc = "Writing corpus", unit = "lines"):
            f.writelines(text.replace("\n", " ") + "\n")
            line_count += 1

    gc.collect()
    return line_count

def build_tokenizer(corpus_path: Path):

    tokenizer = Tokenizer(
        BPE(
            unk_token = "<|unk|>",
            byte_fallback = True
        )
    )

    tokenizer.pre_tokenizer = ByteLevel(
        add_prefix_space = False,
        trim_offsets = True,
        use_regex = True
    )

    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size = CONFIG.VOCAB_SIZE,
        min_frequency = 2,
        special_tokens = list(CONFIG.SPECIAL_TOKENS),
        show_progress = True,
        initial_alphabet = ByteLevel.alphabet()
    )

    tokenizer.train(
        files = [str(corpus_path)],
        trainer = trainer
    )

    return tokenizer

def verify_special_tokens(tokenizer: Tokenizer):

    missing_tokens = [token for token in CONFIG.SPECIAL_TOKENS if tokenizer.token_to_id(token) is None]

    if missing_tokens:
        raise RuntimeError(f"The following special tokens are not tokenized {missing_tokens}!")

def verify_vocab_size(tokenizer: Tokenizer):

    vocab_size = tokenizer.get_vocab_size()

    if vocab_size != CONFIG.VOCAB_SIZE:
        raise RuntimeError(f"Expected {CONFIG.VOCAB_SIZE} got {vocab_size}!")

def save_metadata(tokenizer: Tokenizer):

    metadata = {
        "tokenizer_type": "BPE",
        "tokenizer_name": CONFIG.TOKENIZER_NAME,
        "vocab_size": tokenizer.get_vocab_size(),
        "requested_vocab_size": CONFIG.VOCAB_SIZE,
        "special_tokens": list(CONFIG.SPECIAL_TOKENS),
        "context_length": CONFIG.CONTEXT_LENGTH,
        "byte_level": True,
        "byte_fallback": True,
        "min_frequency": 2,
        "source_files": list(sorted(f.name for f in (CONFIG.RAW_DATA).glob("*.parquet"))),
        "source_split": "train",
        "filters": {
            "score": f"Min Score: {CONFIG.PRETRAIN_MIN_SCORE}",
            "token_count": f"Min Token Count: {CONFIG.PRETRAIN_MIN_TOKEN_COUNT}",
            "language_score": f"Min Language Score: {CONFIG.PRETRAIN_MIN_LANG_SCORE}",
        }
    }

    metadata_path = CONFIG.TOKENIZER_DIR / "metadata.json"
    with metadata_path.open("w", encoding = "utf-8") as f:
        json.dump(
            metadata,
            f,
            indent = 4,
            ensure_ascii = True
        )

def test_tokenizer(tokenizer: Tokenizer):

    test_text = '<|start_header_id|>Hi there!<|end_header_id|>'

    encoding = tokenizer.encode(test_text)
    decoding = tokenizer.decode(encoding.ids, skip_special_tokens = False)

    print(f"IDs: {encoding.ids}")
    print(f"Decoded: {decoding}")

    if decoding != test_text:
        raise RuntimeError("Tokenizer's test failed!")

def main():

    CONFIG.TOKENIZER_DIR.mkdir(
        parents = True,
        exist_ok = True
    )

    print("="*50)
    print("Training Tokenizer")

    print(f"Vocabulary target: {CONFIG.VOCAB_SIZE:,}")
    print(f"Special Tokens: {len(CONFIG.SPECIAL_TOKENS)}")

    with tempfile.TemporaryDirectory(dir = CONFIG.TOKENIZER_DIR) as temp_dir:

        corpus_path = Path(temp_dir) / "corpus.txt"
        print("Writing corpus to disk ...")
        n_lines = write_to_disk(corpus_path)
        print(f"Wrote {n_lines:,} conversations to {corpus_path}")

        tokenizer = build_tokenizer(corpus_path)

    verify_special_tokens(tokenizer)
    verify_vocab_size(tokenizer)

    tokenizer_path = CONFIG.TOKENIZER_DIR / CONFIG.TOKENIZER_FILE
    tokenizer.save(str(tokenizer_path))

    save_metadata(tokenizer)
    test_tokenizer(tokenizer)

    print("="*50)
    print("Tokenizer Training Complete")

    print(f"Vocabulary Size: {tokenizer.get_vocab_size()}")
    print(f"Tokenizer Directory {tokenizer_path}")

if __name__ == "__main__":
    
    main()

    # tokenizer = Tokenizer.from_file(str(CONFIG.TOKENIZER_DIR / CONFIG.TOKENIZER_FILE))
    # test_tokenizer(tokenizer)