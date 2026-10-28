"""Generate a small synthetic dataset for evaluating AWS-related prompts.

Run from this directory with:
    python 001_prompt_evals.py

Requires ANTHROPIC_API_KEY in the environment or in a local .env file.
"""

import ast
import json
import os
from pathlib import Path
from typing import Any

from anthropic import Anthropic
from anthropic.types import MessageParam, TextBlock
from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parent / ".env")
api_key = os.getenv("ANTHROPIC_API_KEY")
base_url = os.getenv("ANTHROPIC_BASE_URL")
client = (
    Anthropic(api_key=api_key, base_url=base_url)
    if base_url
    else Anthropic(api_key=api_key)
)
model = os.getenv("ANTHROPIC_MODEL", "ollama/llama3.2")


def add_user_message(messages: list[MessageParam], text: str) -> None:
    """Append a user turn to a conversation's message history."""
    messages.append({"role": "user", "content": text})


def add_assistant_message(messages: list[MessageParam], text: str) -> None:
    """Append an assistant turn to a conversation's message history."""
    messages.append({"role": "assistant", "content": text})


def chat(
    messages: list[MessageParam],
    system: str | None = None,
    stop_sequences: list[str] | None = None,
) -> str:
    """Send a conversation to Claude and return its text response.

    The installed Anthropic SDK's Messages API does not expose a temperature
    argument, so this function leaves it out rather than passing an unsupported
    parameter.
    """
    request = {
        "model": model,
        "max_tokens": 1000,
        "messages": messages,
        "stop_sequences": stop_sequences or [],
    }
    if system:
        response = client.messages.create(**request, system=system)
    else:
        response = client.messages.create(**request)

    for block in response.content:
        if isinstance(block, TextBlock):
            return block.text.strip()
    raise ValueError("Claude returned no text content.")


def generate_dataset() -> list[dict[str, Any]]:
    """Ask Claude to create three concise AWS coding/structured-output tasks."""
    prompt = """Generate a dataset for evaluating prompts that produce Python,
JSON, or regular expressions for AWS-related tasks.

Return only a valid JSON array containing exactly 3 objects. Each object must
have one string field named "task". Keep each task small enough to solve with
one Python function, one JSON object, or one regular expression. Do not use
Markdown fences or trailing commas.
"""

    raw_response = chat([{"role": "user", "content": prompt}])
    # Tolerate Markdown fences if the model includes them despite the instruction.
    cleaned = raw_response.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```")
        cleaned = cleaned.removesuffix("```").strip()

    try:
        dataset = json.loads(cleaned)
    except json.JSONDecodeError:
        # Models sometimes add an introduction before otherwise valid JSON.
        decoder = json.JSONDecoder()
        dataset = None
        for index, character in enumerate(cleaned):
            if character == "[":
                try:
                    candidate, _ = decoder.raw_decode(cleaned, index)
                except json.JSONDecodeError:
                    continue
                if isinstance(candidate, list):
                    dataset = candidate
                    break

        # Some local models return Python code containing a literal `tasks` list
        # instead of the requested JSON. Parse only that literal; never execute it.
        if dataset is None:
            try:
                response_ast = ast.parse(cleaned)
            except SyntaxError as exc:
                raise ValueError(
                    "The model did not return valid JSON or parseable Python data. "
                    f"Response preview: {raw_response[:300]!r}"
                ) from exc

            for node in ast.walk(response_ast):
                if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "tasks"
                    for target in node.targets
                ):
                    try:
                        dataset = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        continue
                    break

        if dataset is None:
            raise ValueError(
                "The model response was not JSON and did not contain a literal "
                f"'tasks' list. Response preview: {raw_response[:300]!r}"
            )

    if not isinstance(dataset, list) or not all(
        isinstance(item, dict) and isinstance(item.get("task"), str)
        for item in dataset
    ):
        raise ValueError('Expected a JSON array of objects with a string "task" field.')
    return dataset


def main() -> None:
    """Generate the dataset and print it as formatted JSON."""
    dataset = generate_dataset()
    print(json.dumps(dataset, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
