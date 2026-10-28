"""001_tools.py

Converted from 001_tools.ipynb.
Demonstrates:
1. Defining Python tool implementations (datetime calculation, reminders).
2. Authoring standard Anthropic JSON Schemas for tool definitions.
3. Sending tools to Claude via the Messages API (`tools=[...]`).
4. Parsing ToolUseBlock and dispatching local execution.
5. Returning tool_result blocks to Claude in a multi-turn conversation.
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, cast

from anthropic import Anthropic
from anthropic.types import MessageParam, TextBlock, ToolParam, ToolResultBlockParam, ToolUseBlock
from dotenv import load_dotenv

# ==============================================================================
# 1. Environment & Client Initialization
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

print(f"=== Anthropic Tool Use with Claude (Model: {model}) ===\n")


# ==============================================================================
# 2. Conversation Helper Functions
# ==============================================================================
def add_user_message(messages: list[MessageParam], text: str) -> None:
    """Append a user message to conversation history."""
    messages.append({"role": "user", "content": text})


def add_assistant_message(messages: list[MessageParam], content: Any) -> None:
    """Append an assistant response to conversation history."""
    messages.append({"role": "assistant", "content": content})


# ==============================================================================
# 3. Python Tool Implementations (Local Execution)
# ==============================================================================
def add_duration_to_datetime(
    datetime_str: str,
    duration: float = 0,
    unit: str = "days",
    input_format: str = "%Y-%m-%d",
) -> str:
    """Adds a duration to a datetime string and returns a formatted string."""
    date = datetime.strptime(datetime_str, input_format)

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
    """Creates a timed reminder for the user."""
    print(f"\n[REMINDER SYSTEM EVENT]")
    print(f"  Timestamp: {timestamp}")
    print(f"  Content:   {content}\n")
    return f"Reminder successfully set for '{content}' at {timestamp}."


# ==============================================================================
# 4. Tool Schemas (JSON Schema Specification for Claude)
# ==============================================================================
add_duration_to_datetime_schema = {
    "name": "add_duration_to_datetime",
    "description": (
        "Adds a specified duration to a datetime string and returns the resulting "
        "datetime in a detailed format. Handles seconds, minutes, hours, days, weeks, "
        "months, and years, accounting for month lengths and leap years."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "datetime_str": {
                "type": "string",
                "description": "The input datetime string (e.g., '2025-04-03').",
            },
            "duration": {
                "type": "number",
                "description": "Amount of time to add (can be positive or negative).",
            },
            "unit": {
                "type": "string",
                "enum": ["seconds", "minutes", "hours", "days", "weeks", "months", "years"],
                "description": "Unit of time. Defaults to 'days'.",
            },
            "input_format": {
                "type": "string",
                "description": "strptime format string. Defaults to '%Y-%m-%d'.",
            },
        },
        "required": ["datetime_str"],
    },
}

set_reminder_schema = {
    "name": "set_reminder",
    "description": (
        "Creates a timed reminder that will notify the user at the specified time "
        "with the provided content."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The message text for the notification.",
            },
            "timestamp": {
                "type": "string",
                "description": "ISO 8601 formatted date/time string (e.g., '2025-04-24T09:00:00').",
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
                        "name": {
                            "type": "string",
                            "description": "The name of the tool to invoke.",
                        },
                        "arguments": {
                            "type": "string",
                            "description": "The arguments to the tool, encoded as a JSON string.",
                        },
                    },
                    "required": ["name", "arguments"],
                },
            }
        },
        "required": ["invocations"],
    },
}

# Registered tools list
TOOLS: list[ToolParam] = [
    cast(ToolParam, add_duration_to_datetime_schema),
    cast(ToolParam, set_reminder_schema),
]


# ==============================================================================
# 5. Local Tool Dispatcher
# ==============================================================================
def execute_tool(name: str, arguments: dict[str, Any]) -> str:
    """Execute the local Python function corresponding to the tool name."""
    if name == "add_duration_to_datetime":
        duration_value = arguments.get("duration", 0)
        if isinstance(duration_value, str):
            duration_value = float(duration_value)

        return add_duration_to_datetime(
            datetime_str=arguments["datetime_str"],
            duration=float(duration_value),
            unit=arguments.get("unit", "days"),
            input_format=arguments.get("input_format", "%Y-%m-%d"),
        )
    elif name == "set_reminder":
        return set_reminder(
            content=arguments["content"],
            timestamp=arguments["timestamp"],
        )
    else:
        raise ValueError(f"Unknown tool requested: {name}")


# ==============================================================================
# 6. End-to-End Tool Calling Loop Demonstration
# ==============================================================================
def run_tool_use_demo() -> None:
    messages: list[MessageParam] = []

    user_query = (
        "Today is 2025-04-03. What will the date be exactly 3 weeks from today? "
        "Also, please schedule a reminder for that date at 09:00:00 to submit my quarterly taxes."
    )
    print(f"User Request:\n\"{user_query}\"\n")
    add_user_message(messages, user_query)

    # Turn 1: Send query with tools to Claude
    print("--- [Turn 1] Sending Request to Claude with Tools ---")
    response = client.messages.create(
        model=model,
        max_tokens=1000,
        tools=TOOLS,
        messages=messages,
    )

    print(f"Stop Reason: {response.stop_reason}")

    # Claude returns its proposed action (may include text explanation and tool use blocks)
    # Important: append the full assistant response content to maintain conversation state
    messages.append({"role": "assistant", "content": response.content})

    tool_results: list[ToolResultBlockParam] = []

    for block in response.content:
        if isinstance(block, TextBlock):
            print(f"Claude Thought: {block.text}")
        elif isinstance(block, ToolUseBlock):
            print(f"\nClaude invoked tool: '{block.name}' (ID: {block.id})")
            print(f"Tool Arguments: {json.dumps(block.input, indent=2)}")

            # Execute the tool locally
            try:
                result_output = execute_tool(block.name, block.input)
                print(f"Execution Output: {result_output}")
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(result_output),
                    }
                )
            except Exception as exc:
                print(f"Execution Error: {exc}")
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"Error executing tool: {exc}",
                        "is_error": True,
                    }
                )

    # Turn 2: Send tool results back to Claude for final synthesis
    if tool_results:
        print("\n--- [Turn 2] Sending Tool Results back to Claude ---")
        messages.append({"role": "user", "content": tool_results})

        follow_up = client.messages.create(
            model=model,
            max_tokens=1000,
            tools=TOOLS,
            messages=messages,
        )

        print("\nClaude's Final Response:")
        for block in follow_up.content:
            if isinstance(block, TextBlock):
                print(block.text)

    # ==============================================================================
    # 7. Architectural Takeaway
    # ==============================================================================
    print("\n" + "=" * 70)
    print("ARCHITECT NOTE: The Tool Use Lifecycle")
    print("1. Who Executes the Tool?")
    print("   Claude NEVER executes external functions or touches local environments.")
    print("   Claude acts as a reasoning engine that emits a structured ToolUseBlock.")
    print("   The client application remains the sole executor and gatekeeper of side effects.")
    print("2. The Turn-Taking Protocol:")
    print("   - Turn 1: User message -> Model emits ToolUseBlock (stop_reason='tool_use').")
    print("   - Turn 2: Client sends tool_result (with matching tool_use_id) -> Model generates final answer.")
    print("3. Security & Human-in-the-Loop:")
    print("   Because execution happens client-side, architects can insert validation,")
    print("   authorization gates, or confirmation dialogues before executing destructive tools.")
    print("=" * 70)


if __name__ == "__main__":
    run_tool_use_demo()
