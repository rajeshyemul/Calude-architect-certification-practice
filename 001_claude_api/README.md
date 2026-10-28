# 001 - Claude API & Conversation State Architecture

This directory establishes the foundational primitives for interacting with Claude through the Anthropic Messages API (`/v1/messages`), focusing on client-side state ownership, parameters, streaming, and deterministic structured extraction.

---

## 📂 Capability Files

| File | Meaningful Capability | Architectural Concept |
| :--- | :--- | :--- |
| **[`01_first_request.py`](file:///Users/rajesh.yemul/Claude%20Certification/001_claude_api/01_first_request.py)** | Basic Request / Response | SDK initialization, model selection, content block inspection (`TextBlock`). |
| **[`02_multi_turn.py`](file:///Users/rajesh.yemul/Claude%20Certification/001_claude_api/02_multi_turn.py)** | Conversation State Management | Client-side state accumulator (`messages[]`), role alternation, history preservation. |
| **[`03_system_prompt.py`](file:///Users/rajesh.yemul/Claude%20Certification/001_claude_api/03_system_prompt.py)** | Behavioral Steering | Top-level `system` parameter vs in-message prompt injection, persona enforcement. |
| **[`04_temperature.py`](file:///Users/rajesh.yemul/Claude%20Certification/001_claude_api/04_temperature.py)** | Determinism vs Creativity | Temperature $0.0$ (deterministic/code) vs $1.0$ (stochastic/brainstorming). |
| **[`05_streaming.py`](file:///Users/rajesh.yemul/Claude%20Certification/001_claude_api/05_streaming.py)** | Latency & Event Streaming | Time-To-First-Token (TTFT) profiling, `client.messages.stream()`, final usage stats. |
| **[`06_structured_output.py`](file:///Users/rajesh.yemul/Claude%20Certification/001_claude_api/06_structured_output.py)** | Schema-Enforced Extraction | Tool definition + `tool_choice` for guaranteed, validated JSON payloads without markdown noise. |
| **[`07_structured_output.py`](file:///Users/rajesh.yemul/Claude%20Certification/001_claude_api/07_structured_output.py)** | Assistant Prefill & Stop Sequences | Prompt prefilling (`assistant: ```json\n`) + `stop_sequences=["```"]` for clean copy-ready code/JSON. |

---

## 🧠 The 5 Architect Questions

### 1. What did I learn?
The Claude Messages API is strictly **stateless**. Claude has no internal session memory or persistent conversation database. The client application is the sole owner and manager of conversation state across multiple turns.

### 2. What is the implementation architecture?
```text
┌─────────────────────────────────────────────────────────┐
│                   Client Application                    │
│                                                         │
│  ┌────────────────┐               ┌──────────────────┐  │
│  │ User Input     │ ──append──>   │ messages = [     │  │
│  └────────────────┘               │   {"role":"user"}│  │
│                                   │ ]                │  │
│                                   └─────────┬────────┘  │
└─────────────────────────────────────────────┼───────────┘
                                              │ HTTP POST /v1/messages
                                              ▼
┌─────────────────────────────────────────────────────────┐
│                      Anthropic API                      │
│                                                         │
│               Claude Inference Engine                   │
│                                                         │
└─────────────────────────────────────────────┬───────────┘
                                              │ Response (Content Blocks)
                                              ▼
┌─────────────────────────────────────────────────────────┐
│                   Client Application                    │
│                                                         │
│  ┌────────────────┐               ┌──────────────────┐  │
│  │ Assistant      │ ──append──>   │ messages.append( │  │
│  │ TextBlock      │               │   {"role":       │  │
│  │                │               │    "assistant"}  │  │
│  └────────────────┘               │ )                │  │
│                                   └──────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### 3. What did I actually experiment with?
1. **Response Blocks**: Inspected `message.content` as an iterable array of typed blocks (`TextBlock`, `ToolUseBlock`), rather than raw strings.
2. **Conversation History**: Appended turns sequentially to test how Claude resolves contextual references (e.g., pronouns like *"Can you explain **it** in more detail?"*).
3. **Behavioral Divergence**: Tested identical user queries with and without a security-architect `system` prompt to observe constraint compliance.
4. **Sampling Variations**: Verified repeatable output identity at `temperature=0.0` vs high variability at `temperature=1.0`.
5. **Streaming Telemetry**: Captured exact Time-To-First-Token (TTFT) vs total end-to-end response duration.
6. **Tool-Enforced JSON**: Replaced fragile regex/string parsing with schema-enforced tool calling (`tool_choice={"type": "tool", "name": ...}`).

### 4. What can go wrong? (Edge Cases & Failure Modes)
- **Role Order Desynchronization**: Sending two consecutive `"user"` messages or alternating incorrectly results in an HTTP 400 `invalid_request_error`.
- **Token Inflation & Compounding**: As conversation length grows, re-sending the entire history on every request quadratically increases input token consumption and latency.
- **Silent Truncation**: When response reaches `max_tokens`, Claude cuts off mid-sentence. Checking `message.stop_reason == "max_tokens"` is required to avoid corrupt downstream data.
- **Parsing Crashes on Non-Text Blocks**: Assuming `message.content[0].text` will raise `AttributeError` when Claude returns a `ToolUseBlock` or thinking block.
- **Markdown Hallucination in JSON**: Using plain text prompts for JSON often yields markdown fences (` ```json `) or polite conversational filler.

### 5. What would an Architect need to know?
- **State Windowing & Summarization Strategy**: For production multi-turn chats, unbounded history causes cost spikes and context exhaustion (200k limit). Implement sliding-window truncation or periodic LLM summarization.
- **Prompt Caching Economics**: System prompts and static few-shot examples should be placed at the top of requests and marked with `cache_control` to reduce input token latency by up to 85% and cost by up to 90%.
- **Streaming vs Non-Streaming Trade-off**:
  - *Stream* for human-in-the-loop interfaces (chatbots, IDE assistants) to minimize perceived wait time (TTFT).
  - *Do NOT stream* for inter-service microservices, automated pipelines, or tool calling loops where payloads must be validated atomically.
- **Structured Data Reliability**: Never rely on "Please respond in JSON format" in enterprise pipelines. Use Tool Calling (`tools` + `tool_choice`) to enforce schema-level validation at the inference layer.

---

## 🚀 How to Run

1. **Activate Environment**:
   ```zsh
   source .venv/bin/activate
   ```
2. **Run Capability Scripts**:
   ```zsh
   python 01_first_request.py
   python 02_multi_turn.py
   python 03_system_prompt.py
   python 04_temperature.py
   python 05_streaming.py
   python 06_structured_output.py
   python 07_structured_output.py
   ```
