import os
import json
from dotenv import load_dotenv
from anthropic import Anthropic
from anthropic.types import ToolParam, ToolUseBlock

# 1. Environment & Client Setup
load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")
base_url = os.getenv("ANTHROPIC_BASE_URL")
client = Anthropic(api_key=api_key, base_url=base_url) if base_url else Anthropic(api_key=api_key)
model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")

print(f"=== 06: Structured Output via Schema Enforcement (Model: {model}) ===\n")

unstructured_incident = """
At 14:23 UTC on Oct 22, database cluster prod-db-04 suffered high CPU saturation (99%).
API gateway returned HTTP 504 gateway timeouts to roughly 12% of checkout requests.
The on-call SRE (Dave K.) rolled back migration #402, and services recovered by 14:38 UTC.
Severity was classified as Sev-2.
"""

# ==============================================================================
# Architectural Pattern: Enforced Schema via Tool Calling Definition
# In Claude, the most robust way to force valid JSON output is defining a Tool
# and setting tool_choice to force Claude to invoke it.
# ==============================================================================
incident_schema_tool: ToolParam = {
    "name": "record_incident_report",
    "description": "Extract structured post-mortem telemetry from an unstructured incident log.",
    "input_schema": {
        "type": "object",
        "properties": {
            "incident_timestamp": {
                "type": "string",
                "description": "UTC timestamp of incident start"
            },
            "severity": {
                "type": "string",
                "enum": ["Sev-1", "Sev-2", "Sev-3", "Sev-4"],
                "description": "Incident severity tier"
            },
            "affected_system": {
                "type": "string",
                "description": "Specific host or service affected"
            },
            "impact_percentage": {
                "type": "number",
                "description": "Estimated percentage of failed requests"
            },
            "resolution_action": {
                "type": "string",
                "description": "Immediate mitigation step applied"
            },
            "resolved_within_minutes": {
                "type": "integer",
                "description": "Total duration in minutes from start to recovery"
            }
        },
        "required": [
            "incident_timestamp",
            "severity",
            "affected_system",
            "resolution_action",
            "resolved_within_minutes"
        ]
    }
}

print("--- Sending Extraction Request with Forced Tool Choice ---")
response = client.messages.create(
    model=model,
    max_tokens=600,
    tools=[incident_schema_tool],
    tool_choice={"type": "tool", "name": "record_incident_report"},
    messages=[
        {
            "role": "user",
            "content": f"Extract telemetry from this log:\n{unstructured_incident}"
        }
    ]
)

# Extract tool call input payload
extracted_data = None
for block in response.content:
    if isinstance(block, ToolUseBlock):
        print(f"Tool Invoked: {block.name}")
        extracted_data = block.input
        print("\nExtracted JSON Payload:")
        print(json.dumps(extracted_data, indent=2))

# ==============================================================================
# Architectural Takeaway
# ==============================================================================
print("\n" + "="*70)
print("ARCHITECT NOTE:")
print("1. Why Tool Choice over Prompting ('Respond in JSON'):")
print("   Free-form text prompting with 'Output only valid JSON' suffers from:")
print("   - Markdown wrapping (```json ... ```) requiring regex cleanup.")
print("   - Conversational filler ('Sure! Here is your JSON:').")
print("   - Schema drift (missing required keys or type hallucinations).")
print("2. Tool Choice Guarantees:")
print("   Setting tool_choice={'type': 'tool', 'name': ...} forces Claude to output")
print("   pure structured parameters validated directly against your JSON Schema.")
print("3. Downstream Integration:")
print("   The result can be deserialized directly into Pydantic models or DB records")
print("   without fragile post-processing strings.")
print("="*70)
