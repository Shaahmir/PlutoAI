from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

class BPETokenizer:

    def __init__(self, tokenizer_path: Path):

        if not tokenizer_path.exists():
            raise FileNotFoundError("No tokenizer file found!")

        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.decoder = ByteLevelDecoder()

    def encode(self, text: str, add_special_tokens = False):

        encoding = self.tokenizer.encode(
            text,
            add_special_tokens = add_special_tokens
        )

        return encoding.ids

    def decode(self, ids: list[int], skip_special_tokens = False):
        
        return self.tokenizer.decode(
            ids,
            skip_special_tokens = skip_special_tokens
        )

    def encode_batch(self, texts: list[str], add_special_tokens = False):

        encodings = self.tokenizer.encode_batch_fast(
            texts,
            add_special_tokens = add_special_tokens
        )

        return [encoding.ids for encoding in encodings]

    def decode_batch(self, batch_ids: list[list[int]], skip_special_tokens = False):

        return self.tokenizer.decode_batch(
            batch_ids,
            skip_special_tokens = skip_special_tokens
        )

    def token_to_id(self, token: str):
        return self.tokenizer.token_to_id(token)

    def id_to_token(self, token_id: int):
        return self.tokenizer.id_to_token(token_id)

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()