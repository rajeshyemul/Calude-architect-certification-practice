"""005_text_editor_tool.py

Converted from 005_text_editor_tool.ipynb.
Demonstrates:
1. Implementation of Anthropic's reference TextEditorTool (`str_replace_editor`).
2. Directory sandboxing (`_validate_path`) and automated backup rollback (`.backups/`).
3. Core file operations:
   - `create`: Safe new file creation (refuses to overwrite existing files).
   - `view`: Numbered line viewing and directory inspection.
   - `str_replace`: Unambiguous string replacement (enforces exactly 1 match).
   - `insert`: Line-indexed text insertion.
   - `undo_edit`: Restoring previous edits from timestamped backups.
4. Autonomous agentic loop performing live file creation, inspection, and bug fixing.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Any, List, Optional

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

print(f"=== 005: Anthropic Text Editor Tool (str_replace_editor) [Model: {model}] ===\n")


# ==============================================================================
# 2. Conversation Helper Functions (from notebook)
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
# 3. TextEditorTool Implementation (from 005_text_editor_tool.ipynb)
# ==============================================================================
class TextEditorTool:
    """
    Anthropic reference TextEditorTool implementation.
    Provides sandboxed file system operations with automated backups and exact-match safety.
    """

    def __init__(self, base_dir: str = "", backup_dir: str = ""):
        self.base_dir = os.path.abspath(base_dir or os.getcwd())
        self.backup_dir = backup_dir or os.path.join(self.base_dir, ".backups")
        os.makedirs(self.backup_dir, exist_ok=True)

    def _validate_path(self, file_path: str) -> str:
        """Prevent directory traversal attacks (e.g. ../../etc/passwd)."""
        abs_path = os.path.normpath(os.path.join(self.base_dir, file_path))
        if not abs_path.startswith(self.base_dir):
            raise ValueError(
                f"Access denied: Path '{file_path}' is outside the allowed directory"
            )
        return abs_path

    def _backup_file(self, file_path: str) -> str:
        """Create timestamped copy in backup directory before modifying."""
        if not os.path.exists(file_path):
            return ""
        file_name = os.path.basename(file_path)
        backup_path = os.path.join(
            self.backup_dir, f"{file_name}.{os.path.getmtime(file_path):.0f}"
        )
        shutil.copy2(file_path, backup_path)
        return backup_path

    def _restore_backup(self, file_path: str) -> str:
        """Restore the most recent backup."""
        file_name = os.path.basename(file_path)
        backups = [
            f for f in os.listdir(self.backup_dir) if f.startswith(file_name + ".")
        ]
        if not backups:
            raise FileNotFoundError(f"No backups found for {file_path}")

        latest_backup = sorted(backups, reverse=True)[0]
        backup_path = os.path.join(self.backup_dir, latest_backup)
        shutil.copy2(backup_path, file_path)
        return f"Successfully restored {file_path} from backup"

    def _count_matches(self, content: str, old_str: str) -> int:
        return content.count(old_str)

    def view(self, file_path: str, view_range: Optional[List[int]] = None) -> str:
        """View file contents with line numbers, or list directory entries."""
        try:
            abs_path = self._validate_path(file_path)

            if os.path.isdir(abs_path):
                entries = sorted(os.listdir(abs_path))
                return "\n".join(entries) if entries else "[Empty Directory]"

            if not os.path.exists(abs_path):
                raise FileNotFoundError("File not found")

            with open(abs_path, "r", encoding="utf-8") as f:
                content = f.read()

            lines = content.split("\n")
            if view_range:
                start, end = view_range
                if end == -1:
                    end = len(lines)
                selected_lines = lines[start - 1 : end]
                result = [f"{i}: {line}" for i, line in enumerate(selected_lines, start)]
                return "\n".join(result)
            else:
                result = [f"{i}: {line}" for i, line in enumerate(lines, 1)]
                return "\n".join(result)

        except Exception as e:
            raise type(e)(str(e))

    def str_replace(self, file_path: str, old_str: str, new_str: str) -> str:
        """Replace old_str with new_str. Enforces exactly ONE unique match."""
        try:
            abs_path = self._validate_path(file_path)

            if not os.path.exists(abs_path):
                raise FileNotFoundError("File not found")

            with open(abs_path, "r", encoding="utf-8") as f:
                content = f.read()

            match_count = self._count_matches(content, old_str)

            if match_count == 0:
                raise ValueError(
                    "No match found for replacement. Please check your text and try again."
                )
            elif match_count > 1:
                raise ValueError(
                    f"Found {match_count} matches for replacement text. "
                    "Please provide more context lines to make a unique match."
                )

            # Create backup before modifying
            self._backup_file(abs_path)

            # Perform unique replacement
            new_content = content.replace(old_str, new_str)
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            return "Successfully replaced text at exactly one location."

        except Exception as e:
            raise type(e)(str(e))

    def create(self, file_path: str, file_text: str) -> str:
        """Create a new file with file_text (fails if already exists)."""
        try:
            abs_path = self._validate_path(file_path)

            if os.path.exists(abs_path):
                raise FileExistsError(
                    "File already exists. Use str_replace to modify it."
                )

            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(file_text)

            return f"Successfully created {file_path}"

        except Exception as e:
            raise type(e)(str(e))

    def insert(self, file_path: str, insert_line: int, new_str: str) -> str:
        """Insert new_str after insert_line."""
        try:
            abs_path = self._validate_path(file_path)

            if not os.path.exists(abs_path):
                raise FileNotFoundError("File not found")

            self._backup_file(abs_path)

            with open(abs_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            if lines and not lines[-1].endswith("\n"):
                new_str = "\n" + new_str

            if insert_line == 0:
                lines.insert(0, new_str + "\n")
            elif 0 < insert_line <= len(lines):
                lines.insert(insert_line, new_str + "\n")
            else:
                raise IndexError(
                    f"Line number {insert_line} is out of range. File has {len(lines)} lines."
                )

            with open(abs_path, "w", encoding="utf-8") as f:
                f.writelines(lines)

            return f"Successfully inserted text after line {insert_line}"

        except Exception as e:
            raise type(e)(str(e))

    def undo_edit(self, file_path: str) -> str:
        """Undo last modification by restoring the latest backup."""
        try:
            abs_path = self._validate_path(file_path)
            if not os.path.exists(abs_path):
                raise FileNotFoundError("File not found")
            return self._restore_backup(abs_path)
        except Exception as e:
            raise type(e)(str(e))


# Initialize editor tool in current working directory
text_editor_tool = TextEditorTool(base_dir=str(Path(__file__).resolve().parent))


# ==============================================================================
# 4. Tool Schema Definition
# ==============================================================================
# Compatible with both Anthropic Cloud API and local Ollama/LiteLLM models
str_replace_editor_schema = {
    "name": "str_replace_editor",
    "description": (
        "Custom editing tool for viewing, creating, and editing files.\n"
        "* If `path` is a directory, `view` lists top-level files and directories.\n"
        "* If `path` is a file, `view` displays the file with numbered lines.\n"
        "* `create` creates a new file at `path`.\n"
        "* `str_replace` replaces `old_str` with `new_str` (must match exactly once).\n"
        "* `insert` inserts `new_str` after `insert_line`.\n"
        "* `undo_edit` restores the previous version of `path` from backup."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "enum": ["view", "create", "str_replace", "insert", "undo_edit"],
                "description": "The command to run.",
            },
            "path": {
                "type": "string",
                "description": "The path to the file or directory.",
            },
            "file_text": {
                "type": "string",
                "description": "Required parameter for `create`. The content of the file.",
            },
            "old_str": {
                "type": "string",
                "description": "Required parameter for `str_replace`. The string to be replaced.",
            },
            "new_str": {
                "type": "string",
                "description": "Optional parameter for `str_replace` or `insert`.",
            },
            "insert_line": {
                "type": "integer",
                "description": "Required parameter for `insert`. The line number after which to insert.",
            },
            "view_range": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Optional parameter for `view`. Line range [start_line, end_line].",
            },
        },
        "required": ["command", "path"],
    },
}

# Anthropic beta built-in schema format
def get_text_edit_schema(model_name: str) -> dict[str, Any]:
    # If using local LiteLLM/Ollama, return the standard schema
    if "ollama" in model_name.lower():
        return str_replace_editor_schema
    return {
        "type": "text_editor_20250728",
        "name": "str_replace_based_edit_tool",
    }


# ==============================================================================
# 5. Tool Dispatcher & Execution (from notebook)
# ==============================================================================
def run_tool(tool_name: str, tool_input: dict[str, Any]) -> Any:
    """Dispatches text editor commands."""
    if tool_name in ["str_replace_editor", "str_replace_based_edit_tool"]:
        command = tool_input["command"]
        path = tool_input["path"]

        if command == "view":
            return text_editor_tool.view(path, tool_input.get("view_range"))
        elif command == "str_replace":
            return text_editor_tool.str_replace(path, tool_input["old_str"], tool_input["new_str"])
        elif command == "create":
            return text_editor_tool.create(path, tool_input["file_text"])
        elif command == "insert":
            return text_editor_tool.insert(path, tool_input["insert_line"], tool_input["new_str"])
        elif command == "undo_edit":
            return text_editor_tool.undo_edit(path)
        else:
            raise ValueError(f"Unknown text editor command: {command}")
    else:
        raise ValueError(f"Unknown tool name: {tool_name}")


def run_tools(message: Any) -> list[dict[str, Any]]:
    """Execute all tool_use blocks and return tool_result blocks."""
    tool_requests = [block for block in message.content if getattr(block, "type", None) == "tool_use"]
    tool_result_blocks: list[dict[str, Any]] = []

    for tool_request in tool_requests:
        print(f"  -> Tool Call: {tool_request.name} [{tool_request.input.get('command')}]")
        print(f"     Target Path: {tool_request.input.get('path')}")
        try:
            tool_output = run_tool(tool_request.name, tool_request.input)
            print(f"     Result: {repr(str(tool_output)[:100])}...\n")
            tool_result_blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_request.id,
                    "content": json.dumps(tool_output),
                    "is_error": False,
                }
            )
        except Exception as e:
            print(f"     Error: {e}\n")
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
# 6. Autonomous Conversation Engine (from notebook)
# ==============================================================================
def run_conversation(messages: list[dict[str, Any]], max_turns: int = 8) -> list[dict[str, Any]]:
    """Runs autonomous conversation loop until task completion."""
    turn = 1
    tool_spec = str_replace_editor_schema

    while turn <= max_turns:
        print(f"--- [Turn {turn}] Sending to Claude ---")
        response = chat(messages, tools=[tool_spec], temperature=0.0)

        add_assistant_message(messages, response)
        thought = text_from_message(response)
        if thought:
            print(f"Claude Thought: {thought}\n")

        if response.stop_reason != "tool_use":
            break

        tool_results = run_tools(response)
        add_user_message(messages, tool_results)
        turn += 1

    return messages


# ==============================================================================
# 7. Main Demonstration
# ==============================================================================
def main() -> None:
    messages: list[dict[str, Any]] = []

    user_prompt = """
    Please perform the following programming task:
    1. Create a Python file named 'sample_service.py' with a function 'calculate_discount(price, discount_pct)'.
       Include a bug where it divides by 0 if discount_pct is 0.
    2. View the file to inspect the line numbers.
    3. Use str_replace to fix the bug so that if discount_pct <= 0, it simply returns the original price.
    4. View the file again to confirm the fix was applied cleanly.
    """

    print("=" * 70)
    print(f"Goal:\n{user_prompt.strip()}")
    print("=" * 70 + "\n")

    add_user_message(messages, user_prompt.strip())
    run_conversation(messages)

    # Clean up generated demonstration file and backups
    sample_file = Path(__file__).resolve().parent / "sample_service.py"
    if sample_file.exists():
        print(f"\nFinal File Content of '{sample_file.name}':")
        print("-" * 50)
        print(sample_file.read_text(encoding="utf-8"))
        print("-" * 50)

    # ==============================================================================
    # 8. Architectural Takeaways
    # ==============================================================================
    print("\n" + "=" * 70)
    print("ARCHITECT NOTE: The Design of Anthropic's str_replace_editor")
    print("1. Why String Replacement Over Full File Rewrites:")
    print("   For large files (500+ lines), rewriting the entire file costs hundreds of")
    print("   thousands of output tokens and introduces subtle hallucinated deletions.")
    print("   `str_replace` modifies only the relevant lines, reducing token cost by >95%.")
    print("2. The 'Exactly One Match' Invariant:")
    print("   Enforcing `match_count == 1` is critical for AI code editing. If `old_str` matches")
    print("   0 times (hallucination) or 2+ times (ambiguous location), the tool fails fast.")
    print("   This forces Claude to provide surrounding context lines to disambiguate the edit.")
    print("3. Sandboxing & Safe Rollbacks:")
    print("   - `_validate_path` prevents traversal out of `base_dir`.")
    print("   - `_backup_file` creates automated snapshots before every destructive mutation,")
    print("     enabling instant rollback via `undo_edit`.")
    print("=" * 70)


if __name__ == "__main__":
    main()
