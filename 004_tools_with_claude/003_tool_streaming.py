"""003_tool_streaming.py

Converted from 003_tool_streaming.ipynb.
Demonstrates:
1. Streaming both natural language thoughts and tool arguments in real-time.
2. Unconstrained tool selection with streaming: Claude decides to invoke `save_article`.
3. Event handling for `text`, `content_block_start`, `input_json`, and `content_block_stop`.
4. Assembling and executing tool results through `run_tools(response)`.
"""

import json
import os
from pathlib import Path
from typing import Any

from anthropic import Anthropic
from anthropic.types import Message, TextBlock, ToolParam, ToolUseBlock
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
model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")

print(f"=== 003_tool_streaming: Real-Time Tool & Thought Streaming (Model: {model}) ===\n")


# ==============================================================================
# 2. Conversation Helper Functions (from 003_tool_streaming.ipynb)
# ==============================================================================
def add_user_message(messages: list[dict[str, Any]], message: Any) -> None:
    """Append a user message, formatting text strings into standard content blocks."""
    if isinstance(message, list):
        user_message = {
            "role": "user",
            "content": message,
        }
    else:
        user_message = {
            "role": "user",
            "content": [{"type": "text", "text": message}],
        }
    messages.append(user_message)


def add_assistant_message(messages: list[dict[str, Any]], message: Any) -> None:
    """Append an assistant response, properly extracting content blocks from Message objects."""
    if isinstance(message, list):
        assistant_message = {
            "role": "assistant",
            "content": message,
        }
    elif hasattr(message, "content"):
        content_list = []
        for block in message.content:
            if getattr(block, "type", None) == "text":
                content_list.append({"type": "text", "text": block.text})
            elif getattr(block, "type", None) == "tool_use":
                content_list.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )
        assistant_message = {
            "role": "assistant",
            "content": content_list,
        }
    else:
        assistant_message = {
            "role": "assistant",
            "content": [{"type": "text", "text": message}],
        }
    messages.append(assistant_message)


def chat_stream(
    messages: list[dict[str, Any]],
    system: str | None = None,
    temperature: float = 0.0,
    stop_sequences: list[str] | None = None,
    tools: list[Any] | None = None,
    tool_choice: dict[str, Any] | None = None,
    betas: list[str] | None = None,
    max_tokens: int = 1000,
) -> Any:
    """Create a streaming context manager for Claude Messages API."""
    params: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
        "temperature": temperature,
        "stop_sequences": stop_sequences or [],
    }

    if tool_choice:
        params["tool_choice"] = tool_choice

    if tools:
        params["tools"] = tools

    if system:
        params["system"] = system

    if betas:
        try:
            return client.beta.messages.stream(**params, betas=betas)
        except Exception:
            pass

    return client.messages.stream(**params)


def text_from_message(message: Any) -> str:
    """Extract concatenated text from message content."""
    return "\n".join([block.text for block in message.content if getattr(block, "type", None) == "text"])


# ==============================================================================
# 3. Tool Implementations & Schemas
# ==============================================================================
save_article_schema = ToolParam(
    {
        "name": "save_article",
        "description": "Saves a scholarly journal article with abstract and metadata review.",
        "input_schema": {
            "type": "object",
            "properties": {
                "abstract": {
                    "type": "string",
                    "description": "Abstract of the article. One short sentence max.",
                },
                "meta": {
                    "type": "object",
                    "properties": {
                        "word_count": {
                            "type": "integer",
                            "description": "Word count estimate.",
                        },
                        "review": {
                            "type": "string",
                            "description": "Eight sentence review of the paper.",
                        },
                    },
                    "required": ["word_count", "review"],
                },
            },
            "required": ["abstract", "meta"],
        },
    }
)


def save_article(**kwargs) -> str:
    """Local execution function for saving an article."""
    return "Article saved!"


def run_tool(tool_name: str, tool_input: dict[str, Any]) -> Any:
    """Execute tool by name."""
    if tool_name == "save_article":
        return save_article(**tool_input)
    raise ValueError(f"Unknown tool: '{tool_name}'")


def run_tools(message: Any) -> list[dict[str, Any]]:
    """Inspects assistant message for tool_use blocks and generates tool_result blocks."""
    tool_requests = [block for block in message.content if getattr(block, "type", None) == "tool_use"]
    tool_result_blocks: list[dict[str, Any]] = []

    for tool_request in tool_requests:
        try:
            tool_output = run_tool(tool_request.name, tool_request.input)
            tool_result_blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_request.id,
                    "content": json.dumps(tool_output),
                    "is_error": False,
                }
            )
        except Exception as e:
            tool_result_blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_request.id,
                    "content": f"Error: {e}",
                    "is_error": True,
                }
            )

    return tool_result_blocks


# ==============================================================================
# 4. Streaming Conversation Runner
# ==============================================================================
def run_conversation(
    messages: list[dict[str, Any]],
    tools: list[Any] | None = None,
    tool_choice: dict[str, Any] | None = None,
    fine_grained: bool = False,
    max_turns: int = 5,
) -> list[dict[str, Any]]:
    """
    Executes tool streaming:
    - Streams plain text tokens as they arrive.
    - Streams tool invocation starts (`content_block_start`).
    - Streams partial JSON argument chunks (`input_json`).
    - Obtains final assembled message and processes tool results.
    """
    turn = 1
    active_tools = tools or []

    while turn <= max_turns:
        print(f"--- [Turn {turn}] Streaming Response from Claude ---")

        with chat_stream(
            messages,
            tools=active_tools,
            betas=["fine-grained-tool-streaming-2025-05-14"] if fine_grained else [],
            tool_choice=tool_choice,
        ) as stream:
            for chunk in stream:
                # Text token streaming
                if chunk.type == "text":
                    print(chunk.text, end="", flush=True)

                # Tool use block started
                elif chunk.type == "content_block_start":
                    if getattr(chunk.content_block, "type", None) == "tool_use":
                        print(f"\n>>> Tool Call: \"{chunk.content_block.name}\"")
                        print(">>> Streaming Input JSON: ", end="", flush=True)

                # Streaming partial JSON argument tokens
                elif chunk.type == "input_json" and chunk.partial_json:
                    print(chunk.partial_json, end="", flush=True)

                # Content block completed
                elif chunk.type == "content_block_stop":
                    print("\n")

            response = stream.get_final_message()

        add_assistant_message(messages, response)

        # Stop when Claude is finished with tool calls
        if response.stop_reason != "tool_use":
            break

        tool_results = run_tools(response)
        add_user_message(messages, tool_results)

        if tool_choice:
            break

        turn += 1

    return messages


# ==============================================================================
# 5. Main Execution (from 003_tool_streaming.ipynb)
# ==============================================================================
def main() -> None:
    messages: list[dict[str, Any]] = []

    user_query = "Create and save a fake computer science article"
    print("=" * 70)
    print(f"User Query:\n\"{user_query}\"")
    print("=" * 70 + "\n")

    add_user_message(messages, user_query)

    run_conversation(
        messages,
        tools=[save_article_schema],
    )


if __name__ == "__main__":
    main()
