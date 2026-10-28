import os
import json
from dotenv import load_dotenv
from anthropic import Anthropic
from anthropic.types import AssistantMessageParam, MessageParam, TextBlock, UserMessageParam

# 1. Environment & Client Setup
load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")
base_url = os.getenv("ANTHROPIC_BASE_URL")
client = Anthropic(api_key=api_key, base_url=base_url) if base_url else Anthropic(api_key=api_key)
model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")

print(f"=== 07: Structured Output via Assistant Prefilling & Stop Sequences (Model: {model}) ===\n")

user_prompt = "Generate a very short AWS EventBridge rule as JSON that captures EC2 instance state changes when instances start running."

# ==============================================================================
# Experiment A: The Problem — Default Prompting (Verbose Commentary + Markdown)
# ==============================================================================
print("--- [A] Default Behavior (Without Prefill) ---")
default_response = client.messages.create(
    model=model,
    max_tokens=300,
    messages=[
        {"role": "user", "content": user_prompt}
    ]
)

raw_default_text = ""
for block in default_response.content:
    if isinstance(block, TextBlock):
        raw_default_text += block.text

print(raw_default_text)
print("\n[Problem Observed]: Notice the markdown backticks and conversational text")
print("surrounding the JSON. A web application cannot safely parse this without regex extraction.")

# ==============================================================================
# Experiment B: The Solution — Assistant Prefill + Stop Sequence
# ==============================================================================
print("\n" + "="*70)
print("--- [B] Solution: Assistant Prefill ('```json') + Stop Sequence ('```') ---")

# Step 1: User asks for the JSON
# Step 2: Client 'prefills' the assistant turn with '```json'
# Claude is tricked into believing it already began the markdown block.
messages: list[MessageParam] = [
    UserMessageParam(role="user", content=user_prompt),
    AssistantMessageParam(role="assistant", content="```json\n"),
]

prefilled_response = client.messages.create(
    model=model,
    max_tokens=300,
    stop_sequences=["```"],  # Halts generation the exact millisecond Claude tries to close the block
    messages=messages
)

prefilled_text = ""
for block in prefilled_response.content:
    if isinstance(block, TextBlock):
        prefilled_text += block.text

print("Raw Output Received from Claude:")
print(repr(prefilled_text))  # Inspect newlines and formatting

print("\nRendered Text:")
print(prefilled_text.strip())

# Verify stop reason and stop sequence
print(f"\nStop Reason:   {prefilled_response.stop_reason}")      # Expected: 'stop_sequence'
print(f"Stop Sequence: {prefilled_response.stop_sequence}")  # Expected: '```'

# ==============================================================================
# Experiment C: Ingestion & Validation
# ==============================================================================
print("\n--- [C] Ingesting and Parsing into Python Dictionary ---")
try:
    clean_json = json.loads(prefilled_text.strip())
    print("Successfully parsed into Python dict without regex or string stripping!")
    print(f"EventBridge Source: {clean_json.get('source')}")
    print(f"Detail Type:        {clean_json.get('detail-type')}")
    print(f"Target State:       {clean_json.get('detail', {}).get('state')}")
except json.JSONDecodeError as e:
    print(f"JSON Parsing Error: {e}")

# ==============================================================================
# Architectural Takeaway: Prefilling vs Tool Calling
# ==============================================================================
print("\n" + "="*70)
print("ARCHITECT NOTE: Prefilling vs Tool Calling Comparison")
print("1. How Assistant Prefilling Works:")
print("   By supplying a partial assistant response, you steer Claude's next-token")
print("   probability to continue directly from that token prefix.")
print("2. The Role of Stop Sequences:")
print("   Without stop_sequences=['```'], Claude would close the block and append")
print("   unwanted commentary ('This rule matches...'). The stop sequence cuts the stream immediately.")
print("3. Architectural Comparison:")
print("   - Technique A: Assistant Prefill + Stop Sequence (This file, 07)")
print("     -> Best for: Code generation, raw markdown, CSV, bullet lists, copy-to-clipboard UI.")
print("     -> Limitation: Does not strictly enforce a JSON Schema (keys/types can still vary).")
print("   - Technique B: Tool Calling with Forced Choice (06_structured_output.py)")
print("     -> Best for: Machine-to-machine APIs, DB insertion, backend pipelines.")
print("     -> Advantage: Enforces exact schema types, required fields, and enums.")
print("="*70)
