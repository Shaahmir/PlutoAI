from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq
from scripts.sft.sft_data import SFTData

def _get_first(row: dict, keys: tuple):

    for key in keys:

        value = row.get(key)

        if value is not None:
            return value

    return None

def _extract_content(messages: list) -> tuple[list[dict], str, str, str] | None:

    if not isinstance(messages, list):
        return None

    cleaned_messages = []

    for message in messages:

        if not isinstance(message, dict):
            continue

        role = str(message.get("role", "")).strip().lower()
        content = message.get("content")

        if isinstance(content, list):

            parts = []

            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])

            content = "\n".join(parts)

        if content is None:
            continue

        content = str(content).strip()

        if not content or role not in ("system", "user", "assistant"):
            continue

        turn = {
            "role": role,
            "content": content,
            "reasoning": ""
        }

        if role == "assistant" and "<think>" in content and "</think>" in content:

            pre, remaining = content.split("<think>", 1)
            reasoning, post = remaining.split("</think>", 1)
            turn["content"] = f"{pre} {post}".strip() if pre.strip() else post.strip()
            turn["reasoning"] = reasoning.strip()

        elif role == "assistant" and "<|think|>" in content and "<|end_think|>" in content:
            pre, remaining = content.split("<|think|>", 1)
            reasoning, post = remaining.split("<|end_think|>", 1)
            turn["content"] = f"{pre} {post}".strip() if pre.strip() else post.strip()
            turn["reasoning"] = reasoning.strip()

        elif role == "assistant" and "<think>" in content and "</think>" not in content:
            continue

        elif  role == "assistant" and "<|think|>" in content and "<|end_think|>" not in content:
            continue

        cleaned_messages.append(turn)
    
    if not cleaned_messages:
        return None

    if sum(message["role"] == "system" for message in cleaned_messages) > 1:
        return None

    start_index = 1 if cleaned_messages[0]["role"] == "system" else 0
    turns = cleaned_messages[start_index:]

    if len(turns) < 2:
        return None

    for index, turn in enumerate(turns):

        expected = "user" if index % 2 == 0 else "assistant"

        if turn["role"] != expected:
            return None

    if cleaned_messages[-1]["role"] != "assistant":
        return None

    if len(cleaned_messages) > 64:
        return None

    last_user = next((message["content"] for message in reversed(cleaned_messages) if message["role"] == "user"), "")
    last_assistant = next((message for message in reversed(cleaned_messages) if message["role"] == "assistant"), {})

    if not last_user or last_assistant is None or not last_assistant["content"]:
        return None

    return (
        cleaned_messages,
        last_user,
        last_assistant["content"],
        last_assistant.get("reasoning", "")
    )

def normalize_row(row: dict) -> SFTData | None:

    messages = _get_first(
        row,
        ("messages", "conversation", "conversations", "chat")
    )

    if messages is not None:

        extracted = _extract_content(
            messages
        )

        if extracted is not None:

            cleaned_messages, prompt, response, reasoning = extracted

            return SFTData(
                source = "allenai/Dolci-Think-SFT-7B",
                prompt = prompt,
                response = response,
                reasoning = reasoning,
                metadata = row,
                messages = cleaned_messages
            )
    
    prompt = _get_first(
        row,
        ("prompt", "question", "user", "instruction")
    )

    response = _get_first(
        row,
        ("response", "answer", "assistant", "output", "completion")
    )

    if prompt is None or response is None:
        return None

    return SFTData(
        source = "allenai/Dolci-Think-SFT-7B",
        prompt = str(prompt).strip(),
        response = str(response).strip(),
        reasoning = "",
        metadata = row,
        messages = None
    )

def iter_sft_rows(file_path: Path) -> Iterator[SFTData]:

    parquet_file = pq.ParquetFile(file_path)

    for batch in parquet_file.iter_batches(batch_size = 2048):

        for row in batch.to_pylist():

            example = normalize_row(row)

            if example is None:
                continue

            if not example.prompt or not example.response:
                continue

            yield example
