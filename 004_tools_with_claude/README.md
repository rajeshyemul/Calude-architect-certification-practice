# 004 - Tool Use with Claude

This directory covers **Tool Use** (Function Calling) with Claude: defining JSON schemas, parsing model tool calls, managing client-side tool execution, and returning results through the multi-turn Messages API.

---

## 📂 Capability Files

| File | Meaningful Capability | Architectural Concept |
| :--- | :--- | :--- |
| **[`001_tools.py`](file:///Users/rajesh.yemul/Claude%20Certification/004_tools_with_claude/001_tools.py)** | Tool Calling & Execution Lifecycle | Defining JSON Schemas, inspecting `ToolUseBlock`, local execution dispatch, returning `tool_result` blocks, and multi-turn state. |
| **[`001_tools_007.py`](file:///Users/rajesh.yemul/Claude%20Certification/004_tools_with_claude/001_tools_007.py)** | Multi-Tool Chaining & Temporal Grounding | Dynamic time anchoring via `get_current_datetime`, multi-step autonomous tool chaining (`get_datetime` -> `add_duration` -> `set_reminder`). |
| **[`001_tools_008.py`](file:///Users/rajesh.yemul/Claude%20Certification/004_tools_with_claude/001_tools_008.py)** | Generic Dispatcher & Parallel Tool Calling | Reusable `run_tool`/`run_tools` execution engine, parallel tool invocation, and graceful `is_error` recovery. |
| **[`001_tools_009.py`](file:///Users/rajesh.yemul/Claude%20Certification/004_tools_with_claude/001_tools_009.py)** | Autonomous Tool Pruning & Parameter Translation | Full tool belt reasoning: pruning unneeded tools (`get_current_datetime`), intermediate date calculation, and automatic schema translation into `set_reminder`. |
| **[`003_tool_streaming.py`](file:///Users/rajesh.yemul/Claude%20Certification/004_tools_with_claude/003_tool_streaming.py)** | Real-Time Tool & Thought Streaming | Streaming `text`, `content_block_start`, `input_json` (partial JSON tokens), and `content_block_stop`. |
| **[`003_tool_streaming_completed.py`](file:///Users/rajesh.yemul/Claude%20Certification/004_tools_with_claude/003_tool_streaming_completed.py)** | Forced Tool Streaming (`tool_choice`) | Enforcing schema execution with `tool_choice`, streaming partial inputs, and graceful turn termination. |
| **[`005_text_editor_tool.py`](file:///Users/rajesh.yemul/Claude%20Certification/004_tools_with_claude/005_text_editor_tool.py)** | Anthropic Text Editor (`str_replace_editor`) | Sandboxed file system operations, automatic `.backups/` snapshotting, rollback (`undo_edit`), and exact 1-match string replacement (`str_replace`). |
| **[`006_web_search.py`](file:///Users/rajesh.yemul/Claude%20Certification/004_tools_with_claude/006_web_search.py)** | Web Search Baseline | Standard query execution without external search tools. |
| **[`006_web_search_complete.py`](file:///Users/rajesh.yemul/Claude%20Certification/004_tools_with_claude/006_web_search_complete.py)** | Native Server-Side Web Search | Managed Anthropic web search (`web_search_20250305`), domain whitelist filtering (`allowed_domains`), and source citations. |

---

## 🧠 The 5 Architect Questions

### 1. What did I learn?
Claude **does not execute tools**. Claude is purely a reasoning engine that identifies when a tool is needed, decides which tool to call, and formats valid JSON parameters matching your schema. The **client application** is the custodian that actually executes the code and feeds the output back into Claude.

### 2. What is the implementation architecture?
```text
┌──────────────────────────────────────────────────────────────────┐
│                       Client Application                         │
│                                                                  │
│  1. User Query: "Calculate 3 weeks from 2025-04-03"              │
│  2. Attach tool schemas: tools = [add_duration_schema, ...]      │
└─────────────────────────────────┬────────────────────────────────┘
                                  │ HTTP POST /v1/messages
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Claude Inference Engine                       │
│                                                                  │
│  - Decides external tool is required.                            │
│  - Generates ToolUseBlock (tool_use_id, name, input JSON).       │
│  - Stop reason: "tool_use".                                      │
└─────────────────────────────────┬────────────────────────────────┘
                                  │ Returns ToolUseBlock
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│                       Client Application                         │
│                                                                  │
│  3. Dispatches local Python function: add_duration_to_datetime() │
│  4. Formats tool_result block:                                   │
│     {"type": "tool_result", "tool_use_id": "...", "content": ...}│
└─────────────────────────────────┬────────────────────────────────┘
                                  │ HTTP POST /v1/messages (Turn 2)
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Claude Inference Engine                       │
│                                                                  │
│  5. Reads tool_result and synthesizes natural language response. │
└─────────────────────────────────┬────────────────────────────────┘
                                  │
                                  ▼
                      Final Answer to User
```

### 3. What did I actually experiment with?
- Defining structured JSON Schemas with types, descriptions, and required fields.
- Handling `ToolUseBlock` objects and parsing `block.input` dictionaries.
- Executing local functions (`add_duration_to_datetime` and `set_reminder`).
- Appending the assistant's `ToolUseBlock` and user's `tool_result` to maintain exact turn ordering.

### 4. What can go wrong? (Edge Cases & Failure Modes)
- **ID Mismatch**: Omitting or mismatching the `tool_use_id` in the `tool_result` will trigger an immediate 400 API error.
- **Uncaught Tool Exceptions**: If your local Python function crashes, pass `is_error=True` in the `tool_result` block so Claude can gracefully apologize or attempt an alternative approach, rather than crashing the whole system.
- **Vague Schema Descriptions**: If your tool description does not explain *when* and *how* to use the tool, Claude will fail to call it or pass hallucinated arguments.
- **Unbounded Tool Loops**: In autonomous agentic loops, Claude may repeatedly invoke failing tools unless you enforce a maximum turn limit (e.g., max 5 tool turns).

### 5. What would an Architect need to know?
- **Security Boundary**: Because tools execute on your servers or devices, you have full control over permissions, sandboxing, and security validation before any side-effect occurs.
- **Human-in-the-Loop (HITL)**: For destructive tools (e.g., `delete_database`, `transfer_funds`), the client can pause execution, prompt the user for confirmation, and only resume the tool loop once approved.
- **Prompt Caching on Tool Schemas**: Tool definitions are static across user sessions. Placing `cache_control` on the final tool schema allows Anthropic to cache the entire tool definition block, drastically cutting per-request token costs and latency.

---

## 🚀 How to Run

In your terminal:
```zsh
cd "/Users/rajesh.yemul/Claude Certification/004_tools_with_claude"
source ../001_claude_api/.venv/bin/activate

python 001_tools.py
python 001_tools_007.py
python 001_tools_008.py
python 001_tools_009.py
python 003_tool_streaming.py
python 003_tool_streaming_completed.py
python 005_text_editor_tool.py
python 006_web_search.py
python 006_web_search_complete.py
```
