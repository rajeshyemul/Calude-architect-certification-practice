import os
from dotenv import load_dotenv
from anthropic import Anthropic
from anthropic.types import TextBlock

# 1. Environment & Client Setup
load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")
base_url = os.getenv("ANTHROPIC_BASE_URL")
client = Anthropic(api_key=api_key, base_url=base_url) if base_url else Anthropic(api_key=api_key)
model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")

print(f"=== 03: System Prompts & Behavioral Steering (Model: {model}) ===\n")

user_query = "Should we store API keys directly in the frontend repository for convenience?"

# ==============================================================================
# Experiment A: Baseline request without a system prompt
# ==============================================================================
print("--- [A] Request WITHOUT System Prompt ---")
baseline_response = client.messages.create(
    model=model,
    max_tokens=300,
    messages=[{"role": "user", "content": user_query}]
)

for block in baseline_response.content:
    if isinstance(block, TextBlock):
        print(block.text.strip())

# ==============================================================================
# Experiment B: Request WITH a strict Architectural Persona via 'system' parameter
# ==============================================================================
print("\n--- [B] Request WITH Strict System Persona ---")
system_prompt = (
    "You are a strict Zero-Trust Cloud Security Architect. "
    "Rules:\n"
    "1. Respond ONLY with a risk rating (CRITICAL, HIGH, MEDIUM, LOW) followed by a 2-bullet mitigation plan.\n"
    "2. Tone must be professional, direct, and uncompromising.\n"
    "3. Never provide verbose conversational filler."
)

steered_response = client.messages.create(
    model=model,
    system=system_prompt,
    max_tokens=300,
    messages=[{"role": "user", "content": user_query}]
)

for block in steered_response.content:
    if isinstance(block, TextBlock):
        print(block.text.strip())

# ==============================================================================
# Architectural Takeaway
# ==============================================================================
print("\n" + "="*70)
print("ARCHITECT NOTE:")
print("1. Anthropic places 'system' as a top-level parameter, NOT inside messages[].")
print("2. System prompts provide out-of-band behavioral conditioning that survives multi-turn dialogues.")
print("3. In production, system prompts are static across requests, making them prime candidates")
print("   for Anthropic Prompt Caching (saving up to 90% within input token costs).")
print("="*70)
