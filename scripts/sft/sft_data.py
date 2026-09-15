from dataclasses import dataclass

@dataclass(frozen = True)
class SFTData:
    source: str
    prompt: str
    response: str
    reasoning: str = ""
    metadata: dict | None = None
    messages: list[dict] | None = None
