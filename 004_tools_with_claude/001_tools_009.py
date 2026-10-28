"""001_tools_009.py

Converted from 001_tools_009.ipynb.
Demonstrates:
1. Equipping Claude with the full tool belt (multiple tools provided simultaneously).
2. Autonomous Tool Selection: Claude evaluates all schemas and selects ONLY the relevant tools.
3. Multi-turn Sequential Chaining:
   - Turn 1: `add_duration_to_datetime` calculates future appointment date from base date.
   - Turn 2: `set_reminder` registers the reminder using the calculated date.
   - Turn 3: Claude synthesizes final confirmation to the user.
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

print(f"=== 001_tools_009: Autonomous Multi-Tool Selection & Chaining (Model: {model}) ===\n")


# ==============================================================================
# 2. Conversation Helper Functions (from 001_tools_009.ipynb)
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
    """Returns the current system date and time."""
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
    print(f"----")
    print(f"Setting the following reminder for {timestamp}:")
    print(f"{content}")
    print(f"----")
    return f"Reminder successfully set for '{content}' at {timestamp}."


# ==============================================================================
# 4. Tool Schemas
# ==============================================================================
get_current_datetime_schema = ToolParam(
    {
        "name": "get_current_datetime",
        "description": "Returns the current date and time formatted according to the specified format string.",
        "input_schema": {
            "type": "object",
            "properties": {
                "date_format": {
                    "type": "string",
                    "description": "strftime format string. Defaults to '%Y-%m-%d %H:%M:%S'.",
                    "default": "%Y-%m-%d %H:%M:%S",
                }
            },
            "required": [],
        },
    }
)

add_duration_to_datetime_schema = {
    "name": "add_duration_to_datetime",
    "description": (
        "Adds a specified duration to a datetime string and returns the resulting datetime in a detailed format. "
        "Handles seconds, minutes, hours, days, weeks, months, and years."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "datetime_str": {
                "type": "string",
                "description": "The input datetime string to which the duration will be added.",
            },
            "duration": {
                "type": "number",
                "description": "The amount of time to add (can be positive or negative).",
            },
            "unit": {
                "type": "string",
                "enum": ["seconds", "minutes", "hours", "days", "weeks", "months", "years"],
                "description": "The unit of time for the duration. Defaults to 'days'.",
            },
            "input_format": {
                "type": "string",
                "description": "Format string for parsing datetime_str. Defaults to '%Y-%m-%d'.",
            },
        },
        "required": ["datetime_str"],
    },
}

set_reminder_schema = {
    "name": "set_reminder",
    "description": "Creates a timed reminder that will notify the user at the specified time with the provided content.",
    "input_schema": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The message text to display in the reminder.",
            },
            "timestamp": {
                "type": "string",
                "description": "The exact date and time when the reminder should trigger (e.g., ISO 8601 YYYY-MM-DDTHH:MM:SS).",
            },
        },
        "required": ["content", "timestamp"],
    },
}

batch_tool_schema = {
    "name": "batch_tool",
    "description": "Invoke multiple other tool calls simultaneously",
    "input_schema": {
        "type": "object",
        "properties": {
            "invocations": {
                "type": "array",
                "description": "The tool calls to invoke",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "arguments": {"type": "string"},
                    },
                    "required": ["name", "arguments"],
                },
            }
        },
        "required": ["invocations"],
    },
}

# The Full Tool Belt available to Claude
FULL_TOOL_BELT = [
    get_current_datetime_schema,
    add_duration_to_datetime_schema,
    set_reminder_schema,
]


# ==============================================================================
# 5. Tool Dispatcher & Execution Engine (from 001_tools_009.ipynb)
# ==============================================================================
def run_tool(tool_name: str, tool_input: dict[str, Any]) -> Any:
    """Dispatches tool calls by name."""
    if tool_name == "get_current_datetime":
        return get_current_datetime(**tool_input)
    elif tool_name == "add_duration_to_datetime":
        return add_duration_to_datetime(**tool_input)
    elif tool_name == "set_reminder":
        return set_reminder(**tool_input)
    else:
        raise ValueError(f"Unknown tool name: {tool_name}")


def run_tools(message: Any) -> list[dict[str, Any]]:
    """Inspects assistant message for tool_use blocks, executes them, and returns tool_results."""
    tool_requests = [block for block in message.content if getattr(block, "type", None) == "tool_use"]
    tool_result_blocks: list[dict[str, Any]] = []

    for tool_request in tool_requests:
        print(f"  [Tool Request] Name: '{tool_request.name}' (ID: {tool_request.id})")
        print(f"  [Arguments]    {json.dumps(tool_request.input)}")
        try:
            tool_output = run_tool(tool_request.name, tool_request.input)
            print(f"  [Result]       {tool_output}\n")
            tool_result_block = {
                "type": "tool_result",
                "tool_use_id": tool_request.id,
                "content": json.dumps(tool_output),
                "is_error": False,
            }
        except Exception as e:
            print(f"  [Tool Error]   {e}\n")
            tool_result_block = {
                "type": "tool_result",
                "tool_use_id": tool_request.id,
                "content": f"Error: {e}",
                "is_error": True,
            }

        tool_result_blocks.append(tool_result_block)

    return tool_result_blocks


# ==============================================================================
# 6. Autonomous Conversation Engine (from 001_tools_009.ipynb)
# ==============================================================================
def run_conversation(messages: list[dict[str, Any]], max_turns: int = 6) -> list[dict[str, Any]]:
    """
    Executes a multi-turn conversation loop equipped with the full tool belt.
    Runs until Claude reaches completion (stop_reason != 'tool_use') or max_turns is exceeded.
    """
    turn = 1
    while turn <= max_turns:
        print(f"--- [Turn {turn}] Calling Claude with Full Tool Belt ---")
        response = chat(messages, tools=FULL_TOOL_BELT, temperature=0.0)

        add_assistant_message(messages, response)
        thought_text = text_from_message(response)
        if thought_text:
            print(f"Claude: {thought_text}\n")

        if response.stop_reason != "tool_use":
            # Claude is finished using tools and provided the final response
            break

        tool_results = run_tools(response)
        add_user_message(messages, tool_results)
        turn += 1

    return messages


# ==============================================================================
# 7. Main Demonstration
# ==============================================================================
def main() -> None:
    # Test case from 001_tools_009.ipynb:
    # Demonstrates tool selection (ignoring get_current_datetime because base date is given),
    # date arithmetic calculation, and reminder registration.
    messages: list[dict[str, Any]] = []

    user_query = "Set a reminder for my doctors appointment. Its 177 days after Jan 1st, 2050."
    print("=" * 70)
    print(f"User Query:\n\"{user_query}\"")
    print("=" * 70 + "\n")

    add_user_message(messages, user_query)
    run_conversation(messages)

    # ==============================================================================
    # 8. Architectural Takeaways
    # ==============================================================================
    print("=" * 70)
    print("ARCHITECT NOTE: Tool Selection & Parameter Translation")
    print("1. Autonomous Tool Pruning:")
    print("   Even though `get_current_datetime` was available in the tool belt, Claude recognized")
    print("   that the query provided an explicit starting anchor ('Jan 1st, 2050') and correctly")
    print("   bypassed the current time tool, going straight to `add_duration_to_datetime`.")
    print("2. Intermediate Parameter Translation:")
    print("   `add_duration_to_datetime` returned human-readable text: 'Monday, June 27, 2050 12:00:00 AM'.")
    print("   Claude parsed this text and automatically mapped it into the ISO timestamp format")
    print("   required by `set_reminder` ('2050-06-27T00:00:00').")
    print("   This translation of intermediate tool representations is a key strength of LLM agents.")
    print("3. Stateless Loop Management:")
    print("   The client loop remains completely stateless between turns; all intermediate state is")
    print("   preserved strictly in `messages[]` through standard user/assistant turn interleaving.")
    print("=" * 70)


if __name__ == "__main__":
    main()
