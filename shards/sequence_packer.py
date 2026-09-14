from collections import deque
from typing import Iterable

class StreamingSequencePacker:

    def __init__ (self, sequence_length: int):

        if sequence_length <= 0:
            raise ValueError("sequence_length must be greater than zero!")

        self.sequence_length = sequence_length

        self.token_chunks = deque()
        self.mask_chunks = deque()

        self.token_count = 0

    def add(self, token_ids: Iterable[int], loss_mask: Iterable[int] | None = None):

        tokens = list(token_ids)

        if not tokens:
            return

        if loss_mask is not None:
            mask = list(loss_mask)

            if len(tokens) != len(mask):
                raise ValueError("tokens must have same length as loss_mask!")
        else:
            mask = None

        self.token_chunks.append(tokens)
        self.mask_chunks.append(mask)

        self.token_count += len(tokens)

        while self.token_count >= self.sequence_length:

            yield self._build_sequence(
                self.sequence_length
            )
        
    def _build_sequence(self, length: int):

        tokens_out: list[int] = []
        mask_out: list[int] | None = None

        if self.mask_chunks[0] is not None:
            mask_out = []

        remaining = length

        while remaining > 0:
            
            tokens = self.token_chunks[0]
            mask = self.mask_chunks[0]

            take = min(remaining, len(tokens))
            tokens_out.extend(tokens[:take])

            if mask_out is not None:
                if mask is None:
                    raise RuntimeError("Mask State Mismatch")

                mask_out.extend(mask[:take])

            if take == len(tokens):
                self.token_chunks.popleft()
                self.mask_chunks.popleft()

            else:
                self.token_chunks[0] = tokens[take:]

                if mask is not None:
                    self.mask_chunks[0] = mask[take:]

            self.token_count -= take
            remaining -= take

        return (tokens_out, mask_out)

    def buffered_tokens(self):
        return self.token_count

    def discard_remainder(self):

        discarded = self.token_count

        self.token_chunks.clear()
        self.mask_chunks.clear()

        self.token_count = 0

        return discarded