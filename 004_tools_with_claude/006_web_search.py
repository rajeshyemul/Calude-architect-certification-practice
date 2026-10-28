"""006_web_search.py

Converted from 006_web_search.ipynb.
Baseline introduction to Anthropic's built-in Web Search Tool.
"""

import os
from pathlib import Path
from typing import Any

from anthropic import Anthropic
from anthropic.types import Message, TextBlock
from dotenv import load_dotenv

# ==============================================================================
# 1. Environment & Client Setup
# ==============================================================================
load_dotenv(Path(__file__).resolve().parent / ".env")

api_key = os.getenv("ANTHROPIC_API_KEY")
base_url = os.getenv("ANTHROPIC_BASE_URL")
client = (
    Anthropic(api_key=api_key, base_url=base_url)
    if base_url
    else Anthropic(api_key=api_key)
)
model = os.getenv("ANTHROPIC_MODEL", "ollama/llama3.2")

print(f"=== 006: Web Search Tool — Baseline (Model: {model}) ===\n")


# ==============================================================================
# 2. Conversation Helper Functions (from 006_web_search.ipynb)
# ==============================================================================
def add_user_message(messages: list[dict[str, Any]], message: Any) -> None:
    """Append a user message, extracting content if passed a Message object."""
    user_message = {
        "role": "user",
        "content": message.content if isinstance(message, Message) else message,
    }
    messages.append(user_message)


def add_assistant_message(messages: list[dict[str, Any]], message: Any) -> None:
    """Append an assistant message, extracting content if passed a Message object."""
    assistant_message = {
        "role": "assistant",
        "content": message.content if isinstance(message, Message) else message,
    }
    messages.append(assistant_message)


def chat(
    messages: list[dict[str, Any]],
    system: str | None = None,
    temperature: float = 0.0,
    stop_sequences: list[str] | None = None,
    tools: list[Any] | None = None,
    max_tokens: int = 1000,
) -> Any:
    """Send conversation parameters to Claude and return the complete Message object."""
    params: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
        "temperature": temperature,
        "stop_sequences": stop_sequences or [],
    }
    if tools:
        params["tools"] = tools
    if system:
        params["system"] = system

    return client.messages.create(**params)


def text_from_message(message: Any) -> str:
    """Extract and concatenate all text blocks from a message."""
    return "\n".join([block.text for block in message.content if getattr(block, "type", None) == "text"])


# ==============================================================================
# 3. Main Execution
# ==============================================================================
def main() -> None:
    messages: list[dict[str, Any]] = []

    user_query = "What's the best exercise for gaining leg muscle?"
    print(f"User Query:\n\"{user_query}\"\n")
    add_user_message(messages, user_query)

    print("Sending query to Claude (baseline without web search tool)...")
    response = chat(messages)
    add_assistant_message(messages, response)

    print("\nClaude's Response:")
    print(text_from_message(response))


if __name__ == "__main__":
    main()
