from config import CONFIG
from scripts.sft.sft_data import SFTData

def serialize_sft_example(example: SFTData) -> tuple[str, str]:

    user_text = (
        f"{CONFIG.SPECIAL_TAGS['START_HEAD']}user{CONFIG.SPECIAL_TAGS['END_HEAD']}\n"
        f"{example.prompt.strip()}\n"
        f"{CONFIG.SPECIAL_TAGS['EOT']}\n"
    )

    if example.reasoning.strip():

        assistant_text = (
            f"{CONFIG.SPECIAL_TAGS['START_HEAD']}assistant{CONFIG.SPECIAL_TAGS['END_HEAD']}\n"
            f"{CONFIG.SPECIAL_TAGS['THINK']}\n"
            f"{example.reasoning.strip()}\n"
            f"{CONFIG.SPECIAL_TAGS['END_THINK']}\n"
            f"{example.response.strip()}\n"
            f"{CONFIG.SPECIAL_TAGS['EOT']}\n"
            f"{CONFIG.SPECIAL_TAGS['EOS']}"
        )

    else:

        assistant_text = (
            f"{CONFIG.SPECIAL_TAGS['START_HEAD']}assistant{CONFIG.SPECIAL_TAGS['END_HEAD']}\n"
            f"{example.response.strip()}\n"
            f"{CONFIG.SPECIAL_TAGS['EOT']}\n"
            f"{CONFIG.SPECIAL_TAGS['EOS']}"
        )

    return (
        user_text,
        assistant_text
    )
