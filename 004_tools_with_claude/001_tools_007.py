"""001_tools_007.py

Demonstrates:
1. Dynamic time-awareness via `get_current_datetime` tool (grounding the model in current time).
2. Date arithmetic with `add_duration_to_datetime`.
3. Notification triggering with `set_reminder`.
4. Multi-step autonomous tool chaining:
   - Turn 1: LLM calls `get_current_datetime` to determine the present date.
   - Turn 2: LLM uses the result to calculate future date and trigger reminders.
   - Turn 3: LLM synthesizes a final natural language answer.
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from anthropic import Anthropic
from anthropic.types import Message, MessageParam, TextBlock, ToolParam, ToolUseBlock
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

print(f"=== 001_tools_007: Multi-Tool Chaining & Time Grounding (Model: {model}) ===\n")


# ==============================================================================
# 2. Conversation Helper Functions (from 001_tools_007.ipynb)
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
# 3. Python Tool Implementations
# ==============================================================================
def get_current_datetime(date_format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Returns the current system date and time formatted according to format."""
    if not date_format:
        raise ValueError("date_format cannot be empty")
    now_str = datetime.now().strftime(date_format)
    print(f"  [Tool: get_current_datetime] -> Current system time: {now_str}")
    return now_str


def add_duration_to_datetime(
    datetime_str: str,
    duration: float = 0,
    unit: str = "days",
    input_format: str = "%Y-%m-%d",
) -> str:
    """Adds a duration to a datetime string and returns a formatted result."""
    # Attempt parsing with multiple common date formats for resilience
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

    result_str = new_date.strftime("%A, %B %d, %Y %I:%M:%S %p")
    print(f"  [Tool: add_duration_to_datetime] {datetime_str} + {duration} {unit} -> {result_str}")
    return result_str


def set_reminder(content: str, timestamp: str) -> str:
    """Creates a timed reminder event."""
    print(f"\n  [Tool: set_reminder Event Triggered]")
    print(f"    Target Time: {timestamp}")
    print(f"    Message:     {content}\n")
    return f"Reminder successfully registered for '{content}' at {timestamp}."


# ==============================================================================
# 4. Tool Schemas (from 001_tools_007.ipynb)
# ==============================================================================
get_current_datetime_schema = ToolParam(
    {
        "name": "get_current_datetime",
        "description": (
            "Returns the current date and time formatted according to the specified format string. "
            "Use this tool when you need to know the current real-world date and time, such as for "
            "timestamping records, calculating relative dates from today, or scheduling future tasks."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "date_format": {
                    "type": "string",
                    "description": "Python strftime format. Defaults to '%Y-%m-%d %H:%M:%S'.",
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
                "description": "The unit of time. Defaults to 'days'.",
            },
            "input_format": {
                "type": "string",
                "description": "Python strptime format string. Defaults to '%Y-%m-%d'.",
            },
        },
        "required": ["datetime_str"],
    },
}

set_reminder_schema = {
    "name": "set_reminder",
    "description": (
        "Creates a timed reminder that will notify the user at the specified time with the provided content."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The message text that will be displayed in the reminder notification.",
            },
            "timestamp": {
                "type": "string",
                "description": "The exact date and time formatted as ISO 8601 (YYYY-MM-DDTHH:MM:SS) or formatted date.",
            },
        },
        "required": ["content", "timestamp"],
    },
}

batch_tool_schema = {
    "name": "batch_tool",
    "description": "Invoke multiple other tool calls simultaneously.",
    "input_schema": {
        "type": "object",
        "properties": {
            "invocations": {
                "type": "array",
                "description": "The tool calls to invoke.",
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

# The active tool suite
TOOLS = [
    get_current_datetime_schema,
    add_duration_to_datetime_schema,
    set_reminder_schema,
]


# ==============================================================================
# 5. Tool Dispatcher
# ==============================================================================
def execute_tool(name: str, arguments: dict[str, Any]) -> str:
    """Dispatches tool calls to the matching Python function."""
    if name == "get_current_datetime":
        fmt = arguments.get("date_format", "%Y-%m-%d %H:%M:%S")
        return get_current_datetime(date_format=fmt)
    elif name == "add_duration_to_datetime":
        return add_duration_to_datetime(
            datetime_str=arguments["datetime_str"],
            duration=arguments.get("duration", 0),
            unit=arguments.get("unit", "days"),
            input_format=arguments.get("input_format", "%Y-%m-%d"),
        )
    elif name == "set_reminder":
        return set_reminder(
            content=arguments["content"],
            timestamp=arguments["timestamp"],
        )
    else:
        raise ValueError(f"Unknown tool name: {name}")


# ==============================================================================
# 6. Autonomous Multi-Step Tool Chaining Loop
# ==============================================================================
def run_autonomous_agent(user_query: str, max_turns: int = 5) -> None:
    """
    Executes an autonomous agent loop:
    1. Sends user message to Claude with available tools.
    2. While Claude requests tool use, executes tools locally and feeds results back.
    3. Exits when Claude produces final text (stop_reason != 'tool_use') or max_turns is reached.
    """
    print("=" * 70)
    print(f"User Goal: \"{user_query}\"")
    print("=" * 70)

    messages: list[dict[str, Any]] = []
    add_user_message(messages, user_query)

    turn = 1
    while turn <= max_turns:
        print(f"\n--- [Turn {turn}] Sending to Claude ---")
        response = chat(messages, tools=TOOLS, temperature=0.0)
        add_assistant_message(messages, response)

        # Check if Claude wants to use tools
        tool_blocks = [b for b in response.content if isinstance(b, ToolUseBlock)]

        if not tool_blocks or response.stop_reason != "tool_use":
            # No further tools needed: Claude provided the final answer!
            print("\nFinal Response from Claude:")
            print(text_from_message(response))
            break

        # Process all tool calls requested in this turn
        tool_results: list[dict[str, Any]] = []
        for block in tool_blocks:
            print(f"-> Claude requested tool: '{block.name}'")
            print(f"   Arguments: {json.dumps(block.input)}")

            try:
                output = execute_tool(block.name, block.input)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(output),
                    }
                )
            except Exception as e:
                print(f"   [Error executing tool]: {e}")
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"Error: {e}",
                        "is_error": True,
                    }
                )

        # Feed tool results back to Claude
        add_user_message(messages, tool_results)
        turn += 1

    if turn > max_turns:
        print("\nWarning: Reached maximum turn limit before completion.")

    # ==============================================================================
    # 7. Architectural Takeaway
    # ==============================================================================
    print("\n" + "=" * 70)
    print("ARCHITECT NOTE: Dynamic Temporal Grounding & Multi-Turn Chaining")
    print("1. The Temporal Cutoff Problem:")
    print("   LLMs do not know 'today's' date because model weights are frozen at training time.")
    print("   Providing a simple `get_current_datetime` tool anchors the agent in real time,")
    print("   preventing hallucinated dates without needing to hardcode timestamps into system prompts.")
    print("2. Multi-Step Chaining (Emergent Workflows):")
    print("   Notice that Claude autonomously chained 3 separate tools:")
    print("   get_current_datetime -> add_duration_to_datetime -> set_reminder.")
    print("   The client did not hardcode this sequence; Claude planned and executed it step-by-step.")
    print("3. State Growth & Compounding Latency:")
    print("   Each turn appends tool_use and tool_result blocks to `messages[]`.")
    print("   In complex agentic loops with 10+ turns, prompt caching or history summarization")
    print("   becomes critical to control input token costs.")
    print("=" * 70)


def main() -> None:
    # A query requiring real-time knowledge, date arithmetic, and action:
    query = (
        "What is the current date and time today? Based on right now, "
        "what will the date be exactly 10 days from today? "
        "Please schedule a reminder on that date at 15:00:00 for 'Review sprint deliverables'."
    )
    run_autonomous_agent(query)


if __name__ == "__main__":
    main()
