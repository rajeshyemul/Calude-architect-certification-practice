"""006_web_search_complete.py

Converted from 006_web_search_complete.ipynb.
Demonstrates:
1. Anthropic's Native Server-Side Web Search Tool (`type: "web_search_20250305"`).
2. Domain filtering with `allowed_domains=["nih.gov"]` (ensuring scientific rigor and preventing prompt injection).
3. Search limits with `max_uses=5` to prevent unbounded search loops.
4. Parsing model response, web search results, and citations.
"""

import json
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

print(f"=== 006: Server-Side Web Search with Domain Filtering (Model: {model}) ===\n")


# ==============================================================================
# 2. Conversation Helper Functions (from 006_web_search_complete.ipynb)
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
    max_tokens: int = 1500,
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
# 3. Anthropic Server-Side Web Search Tool Definition
# ==============================================================================
web_search_schema = {
    "type": "web_search_20250305",
    "name": "web_search",
    "max_uses": 5,
    "allowed_domains": ["nih.gov"],  # Constrains research to trusted National Institutes of Health domain
}


# ==============================================================================
# 4. Main Execution
# ==============================================================================
def main() -> None:
    messages: list[dict[str, Any]] = []

    user_query = "What's the best exercise for gaining leg muscle?"
    print("=" * 70)
    print(f"User Query:\n\"{user_query}\"")
    print(f"Tool Configuration: allowed_domains=['nih.gov'], max_uses=5")
    print("=" * 70 + "\n")

    add_user_message(messages, user_query)

    print("Sending request with Anthropic Server-Side Web Search Tool...")
    try:
        response = chat(messages, tools=[web_search_schema])
        add_assistant_message(messages, response)

        print("\n--- Inspecting Response Content Blocks ---")
        for i, block in enumerate(response.content, 1):
            block_type = getattr(block, "type", "unknown")
            print(f"Block #{i} [Type: {block_type}]")
            if block_type == "text":
                print(block.text)
                # Check for citations if present
                citations = getattr(block, "citations", None)
                if citations:
                    print(f"\n[Citations Attached: {len(citations)}]")
                    for c in citations:
                        print(f"  - Cited URL: {getattr(c, 'url', 'N/A')}")
            else:
                print(f"  Details: {block}")

    except Exception as exc:
        print(f"\n[Note on Execution]: {exc}")
        print("\nArchitect Context:")
        print("Anthropic's `web_search_20250305` is a managed server-side tool hosted by Anthropic.")
        print("It requires an official Anthropic API key with web search access.")
        print("If running through a local LiteLLM/Ollama proxy, server-side tools are intercepted")
        print("unless proxy pass-through is configured.")

    # ==============================================================================
    # 5. Architectural Takeaways
    # ==============================================================================
    print("\n" + "=" * 70)
    print("ARCHITECT NOTE: Server-Side vs Client-Side Tools")
    print("1. Server-Side Execution Model:")
    print("   Unlike standard tools where the client must execute code locally and return `tool_result`,")
    print("   Anthropic's Web Search is entirely server-managed. Anthropic handles the query execution,")
    print("   HTTP fetching, text summarization, and citation generation within their cloud pipeline.")
    print("2. Domain Whitelisting as a Security Boundary:")
    print("   `allowed_domains=['nih.gov']` (or `['docs.aws.amazon.com']`) creates a critical guardrail:")
    print("   - Prevents prompt injection from untrusted external websites.")
    print("   - Eliminates hallucinated facts by grounding answers strictly in vetted domains.")
    print("3. Controlling Latency and Costs with `max_uses`:")
    print("   Web searches introduce network latency (~1-3s per search). Limiting `max_uses` (e.g. 3-5)")
    print("   prevents Claude from entering deep rabbit-holes or incurring high search fees.")
    print("=" * 70)


if __name__ == "__main__":
    main()
