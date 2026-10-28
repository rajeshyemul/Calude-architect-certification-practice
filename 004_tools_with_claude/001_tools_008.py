"""001_tools_008.py

Converted from 001_tools_008.ipynb.
Demonstrates:
1. Generic tool dispatching (`run_tool` and `run_tools`).
2. Parallel and multi-turn tool handling in a robust `run_conversation` loop.
3. Catching and reporting tool errors gracefully via `is_error: True`.
4. Testing multiple concurrent/sequential tool calls (e.g. requesting multiple time formats).
"""

import json
import os
from datetime import datetime, timedelta
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
model = os.getenv("ANTHROPIC_MODEL", "ollama/llama3.2")

print(f"=== 001_tools_008: Generic Tool Execution Engine (Model: {model}) ===\n")


# ==============================================================================
# 2. Conversation Helper Functions (from 001_tools_008.ipynb)
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
# 3. Tool Implementations
# ==============================================================================
def get_current_datetime(date_format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Returns the current system date and time formatted according to date_format."""
    if not date_format:
        raise ValueError("date_format cannot be empty")
    return datetime.now().strftime(date_format)


def add_duration_to_datetime(
    datetime_str: str,
    duration: float = 0,
    unit: str = "days",
    input_format: str = "%Y-%m-%d",
) -> str:
    """Adds a duration to a datetime string and returns a formatted result."""
    formats_to_try = [input_format, "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"]
    date = None
    for fmt in formats_to_try:
        try:
            date = datetime.strptime(datetime_str.strip(), fmt)
            break
        except ValueError:
            continue

    if date is None:
        date = datetime.strptime(datetime_str[:10], "%Y-%m-%d")

    if unit == "seconds":
        new_date = date + timedelta(seconds=duration)
    elif unit == "minutes":
        new_date = date + timedelta(minutes=duration)
    elif unit == "hours":
        new_date = date + timedelta(hours=duration)
    elif unit == "days":
        new_date = date + timedelta(days=duration)
    elif unit == "weeks":
        new_date = date + timedelta(weeks=duration)
    elif unit == "months":
        month = date.month + int(duration)
        year = date.year + month // 12
        month = month % 12
        if month == 0:
            month = 12
            year -= 1
        day = min(
            date.day,
            [
                31,
                29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28,
                31,
                30,
                31,
                30,
                31,
                31,
                30,
                31,
                30,
                31,
            ][month - 1],
        )
        new_date = date.replace(year=year, month=month, day=day)
    elif unit == "years":
        new_date = date.replace(year=date.year + int(duration))
    else:
        raise ValueError(f"Unsupported time unit: {unit}")

    return new_date.strftime("%A, %B %d, %Y %I:%M:%S %p")


def set_reminder(content: str, timestamp: str) -> str:
    """Creates a timed reminder event."""
    print(f"\n[EVENT] Setting reminder for {timestamp}: {content}\n")
    return f"Reminder successfully registered for '{content}' at {timestamp}."


# ==============================================================================
# 4. Tool Schemas
# ==============================================================================
get_current_datetime_schema = ToolParam(
    {
        "name": "get_current_datetime",
        "description": (
            "Returns the current date and time formatted according to the specified format string. "
            "Use this tool when you need to know the current date and time, such as for timestamping records, "
            "calculating time differences, or displaying the current time to users. The default format returns "
            "the date and time in ISO-like format (YYYY-MM-DD HH:MM:SS)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "date_format": {
                    "type": "string",
                    "description": (
                        "A string specifying the format of the returned datetime using strftime codes "
                        "(e.g., '%H:%M' for hour/minute, '%S' for seconds, '%Y-%m-%d' for date)."
                    ),
                    "default": "%Y-%m-%d %H:%M:%S",
                }
            },
            "required": [],
        },
    }
)

add_duration_to_datetime_schema = {
    "name": "add_duration_to_datetime",
    "description": "Adds a duration to a datetime string and returns a formatted result.",
    "input_schema": {
        "type": "object",
        "properties": {
            "datetime_str": {"type": "string", "description": "The input datetime string."},
            "duration": {"type": "number", "description": "Amount of time to add."},
            "unit": {
                "type": "string",
                "enum": ["seconds", "minutes", "hours", "days", "weeks", "months", "years"],
                "description": "Time unit.",
            },
            "input_format": {"type": "string", "description": "strptime format string."},
        },
        "required": ["datetime_str"],
    },
}

set_reminder_schema = {
    "name": "set_reminder",
    "description": "Creates a timed reminder that will notify the user at the specified time with content.",
    "input_schema": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "Reminder text."},
            "timestamp": {"type": "string", "description": "Target date and time."},
        },
        "required": ["content", "timestamp"],
    },
}

# Master list of all registered tools
ALL_TOOLS = [
    get_current_datetime_schema,
    add_duration_to_datetime_schema,
    set_reminder_schema,
]


# ==============================================================================
# 5. Generic Tool Dispatcher & Execution Engine (from 001_tools_008.ipynb)
# ==============================================================================
def run_tool(tool_name: str, tool_input: dict[str, Any]) -> Any:
    """Dispatches any requested tool to its corresponding Python function."""
    if tool_name == "get_current_datetime":
        return get_current_datetime(**tool_input)
    elif tool_name == "add_duration_to_datetime":
        return add_duration_to_datetime(**tool_input)
    elif tool_name == "set_reminder":
        return set_reminder(**tool_input)
    else:
        raise ValueError(f"Unknown tool: '{tool_name}'")


def run_tools(message: Any) -> list[dict[str, Any]]:
    """
    Extracts all tool_use blocks from the assistant's message, executes each one,
    and returns a list of tool_result blocks with error handling.
    """
    tool_requests = [block for block in message.content if getattr(block, "type", None) == "tool_use"]
    tool_result_blocks: list[dict[str, Any]] = []

    for tool_request in tool_requests:
        print(f"  -> Executing Tool: {tool_request.name} (ID: {tool_request.id})")
        print(f"     Inputs: {json.dumps(tool_request.input)}")
        try:
            tool_output = run_tool(tool_request.name, tool_request.input)
            print(f"     Output: {tool_output}")
            tool_result_block = {
                "type": "tool_result",
                "tool_use_id": tool_request.id,
                "content": json.dumps(tool_output),
                "is_error": False,
            }
        except Exception as e:
            print(f"     Error: {e}")
            tool_result_block = {
                "type": "tool_result",
                "tool_use_id": tool_request.id,
                "content": f"Error: {e}",
                "is_error": True,
            }

        tool_result_blocks.append(tool_result_block)

    return tool_result_blocks


# ==============================================================================
# 6. Autonomous Conversation Engine (from 001_tools_008.ipynb)
# ==============================================================================
def run_conversation(
    messages: list[dict[str, Any]],
    tools: list[Any] | None = None,
    max_turns: int = 6,
) -> list[dict[str, Any]]:
    """
    Executes a multi-turn conversation loop until Claude stops requesting tools
    or max_turns is reached.
    """
    turn = 1
    active_tools = tools if tools is not None else ALL_TOOLS

    while turn <= max_turns:
        print(f"\n--- [Turn {turn}] Sending to Claude ---")
        response = chat(messages, tools=active_tools, temperature=0.0)

        add_assistant_message(messages, response)
        thought_text = text_from_message(response)
        if thought_text:
            print(f"Claude: {thought_text}")

        # Stop when Claude has finished using tools
        if response.stop_reason != "tool_use":
            break

        # Execute all requested tools and feed back results
        tool_results = run_tools(response)
        add_user_message(messages, tool_results)
        turn += 1

    return messages


# ==============================================================================
# 7. Demonstration
# ==============================================================================
def main() -> None:
    # Test Case from 001_tools_008.ipynb:
    # Requires Claude to execute multiple tool calls for different formats!
    messages: list[dict[str, Any]] = []

    user_query = "What is the current time in HH:MM format? Also, what is the current time in SS format?"
    print(f"User Query:\n\"{user_query}\"")
    add_user_message(messages, user_query)

    # Run conversation using get_current_datetime_schema
    run_conversation(messages, tools=[get_current_datetime_schema])

    # ==============================================================================
    # 8. Architectural Takeaways
    # ==============================================================================
    print("\n" + "=" * 70)
    print("ARCHITECT NOTE: Generic Dispatchers & Error Boundaries")
    print("1. Generic Tool Dispatching (`run_tools`):")
    print("   Rather than hardcoding conditional if-else statements for every single tool call,")
    print("   a generic dispatcher decouples tool invocation from model turn management.")
    print("2. Parallel Tool Use:")
    print("   Claude can emit multiple `ToolUseBlock`s in a single response turn.")
    print("   The client application loops over all tool_use blocks, executes them (in parallel or sequence),")
    print("   and packs all corresponding `tool_result` blocks into a single user response turn.")
    print("3. Graceful Error Recovery via `is_error=True`:")
    print("   When an external API or database throws an exception, returning `is_error: True` in")
    print("   the `tool_result` block allows Claude to reason about the failure, explain the issue")
    print("   to the user, or attempt alternative parameters, rather than crashing the system.")
    print("=" * 70)


if __name__ == "__main__":
    main()
