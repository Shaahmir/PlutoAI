from config import CONFIG
from inference.pretrain.generate import generate
from inference.loader import load_model, load_tokenizer

GRAY = "\033[90m"
RED = "\033[91m"
YELLOW = "\033[93m"
WHITE = "\033[97m"
RESET = "\033[0m"

def main():

    device = CONFIG.DEVICE

    checkpoint_path = CONFIG.PRETRAIN_CHECKPOINT_DIR / "best.pt"

    if not checkpoint_path.exists():
        checkpoint_path = CONFIG.PRETRAIN_CHECKPOINT_DIR / "latest.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError("No checkpoint found!")  

    tokenizer_path = CONFIG.TOKENIZER_PATH

    max_new_tokens = CONFIG.MAX_NEW_TOKENS
    temperature = CONFIG.TEMPERATURE
    top_k = CONFIG.TOP_K
    top_p = CONFIG.TOP_P
    min_p = CONFIG.MIN_P
    repetition_penalty = CONFIG.REPETITION_PENALTY
    frequency_penalty = CONFIG.FREQUENCY_PENALTY
    presence_penalty = CONFIG.PRESENCE_PENALTY
    repetition_window = CONFIG.REPETITION_WINDOW

    print("Loading PlutoAI ...")

    tokenizer = load_tokenizer(
        tokenizer_path
    )

    model = load_model(
        checkpoint_path,
        device
    )

    print("PlutoAI Loaded Successfully!")
    print("INFO: Type /exit to quit")

    while True:

        try:
            prompt = input(f"\n{RED}Prompt: {WHITE}").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if prompt.lower() == "/exit":
            break

        if not prompt:
            continue

        print(f"\n{YELLOW}Assistant: {RESET}", end = "", flush = True)
        response = ""

        if response == "":
            print(prompt, end = "", flush = True)

        for chunk in generate(
            model = model,
            tokenizer = tokenizer,
            prompt = prompt,
            device = device,
            max_new_tokens = max_new_tokens,
            temperature = temperature,
            top_k = top_k,
            top_p = top_p,
            min_p = min_p,
            repetition_penalty = repetition_penalty,
            frequency_penalty = frequency_penalty,
            presence_penalty = presence_penalty,
            repetition_window = repetition_window
        ):
            print(chunk, end = "", flush = True)
            response += chunk

        print()

        EOT = CONFIG.SPECIAL_TAGS["EOT"]
        EOS = CONFIG.SPECIAL_TAGS["EOS"]

        if response.endswith(EOT):
            response = response[: -len(EOT)].strip()
        
        if response.endswith(EOS):
            response = response[: -len(EOS)].strip()

if __name__ == "__main__":
    main()
