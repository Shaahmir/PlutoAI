import torch
import torch.nn.functional as F

from model.gpt import GPT
from tokenizer.tokenizer import BPETokenizer
from config import CONFIG

def banned_ngrams(generated_ids: list[int], n: int) -> set[int]:

    if n <= 1 or len(generated_ids) < n - 1:
        return set()

    prefix = tuple(generated_ids[-(n - 1):])
    banned = set()

    for i in range(len(generated_ids) - n + 1):
        if tuple(generated_ids[i : i + n - 1]) == prefix:
            banned.add(generated_ids[i + n - 1])

    return banned

@torch.inference_mode
def generate(
    model: GPT,
    tokenizer: BPETokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    repetition_window: int = 256,
    no_repeat_ngram_size: int = 4
) -> str:

    model.eval()

    input_ids = tokenizer.encode(prompt)

    if not input_ids:
        raise RuntimeError("Prompt produced zero tokens!")

    if temperature <= 0:
        raise ValueError("Temperature must be greater than zero!")

    input_ids = torch.tensor(
        input_ids,
        dtype = torch.long,
        device = device
    ).unsqueeze(0)

    eos_id = tokenizer.token_to_id(CONFIG.SPECIAL_TAGS["EOS"])
    eot_id = tokenizer.token_to_id(CONFIG.SPECIAL_TAGS["EOT"])

    previous_text = tokenizer.decode(
        input_ids[0].tolist(),
        skip_special_tokens = False
    )

    prompt_length = input_ids.size(1)

    for _ in range(max_new_tokens):

        context_ids = input_ids[:, -model.config.CONTEXT_LENGTH:]
        logits, _ = model(context_ids)

        next_token_logits = logits[:, -1, :].clone()

        currently_generated = input_ids[:, prompt_length:]

        if no_repeat_ngram_size > 1 and currently_generated.size(1) > 0:

            banned_ids = banned_ngrams(
                currently_generated[0].tolist(),
                no_repeat_ngram_size
            )

            if banned_ids:
                next_token_logits[0, list(banned_ids)] = float("-inf")

        if currently_generated.size(1) > 0:

            recent_ids = currently_generated[:, -repetition_window : ]

            if repetition_penalty != 1.0 or frequency_penalty != 0.0 or presence_penalty != 0.0:

                unique_ids, counts = torch.unique(
                    recent_ids[0],
                    return_counts = True
                )

                if repetition_penalty != 0.0:
                    selected = next_token_logits[0, unique_ids]
                    penalized = torch.where(
                        selected > 0,
                        selected / repetition_penalty,
                        selected * repetition_penalty
                    )
                    next_token_logits[0, unique_ids] = penalized

                if frequency_penalty != 0.0:
                    next_token_logits[0, unique_ids] -= frequency_penalty * counts.to(next_token_logits.dtype)

                if presence_penalty != 0.0:
                    next_token_logits[0, unique_ids] -= presence_penalty

        next_token_logits = next_token_logits / temperature

        if top_k > 0:

            top_k = min(top_k, next_token_logits.size(-1))

            values, _ = torch.topk(
                next_token_logits,
                top_k
            )

            minimum_value = values[:, -1, None]

            next_token_logits = torch.where(

                next_token_logits < minimum_value,
                torch.full_like(next_token_logits, float("-inf")),
                next_token_logits

            )

        if 0.0 < top_p < 1.0:

            sorted_logits, sorted_indices = torch.sort(
                next_token_logits,
                descending = True
            )

            sorted_probabilities = F.softmax(
                sorted_logits,
                dim = -1
            )

            cumulative_probabilities = torch.cumsum(
                sorted_probabilities,
                dim = -1
            )

            remove = cumulative_probabilities > top_p
            remove[:, 0] = False

            sorted_logits = sorted_logits.masked_fill(
                remove,
                float("-inf")
            )

            next_token_logits = torch.full_like(
                next_token_logits,
                float("-inf")
            )

            next_token_logits.scatter_(
                1,
                sorted_indices,
                sorted_logits
            )

        if 0.0 < min_p < 1.0:

            probabilities_min_p = F.softmax(
                next_token_logits,
                dim = -1
            )

            top_probability = probabilities_min_p.max(
                dim = 1,
                keepdim = True
            ).values

            threshold = min_p * top_probability
            remove_low_prob = probabilities_min_p < threshold

            next_token_logits = next_token_logits.masked_fill(
                remove_low_prob,
                float("-inf")
            )

        probabilities = F.softmax(
            next_token_logits,
            dim = -1
        )

        next_token = torch.multinomial(
            probabilities,
            num_samples = 1
        )

        input_ids = torch.cat(
            [input_ids, next_token],
            dim = 1
        )

        current_text = tokenizer.decode(
            input_ids[0].tolist(),
            skip_special_tokens = False
        )

        new_chunk = current_text[len(previous_text):]
        previous_text = current_text

        if new_chunk:
            yield new_chunk

        if (eos_id is not None and next_token.item() == eos_id) or (eot_id is not None and next_token.item() == eot_id):
            break
