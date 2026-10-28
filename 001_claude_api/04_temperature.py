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

print(f"=== 04: Temperature & Determinism Trade-offs (Model: {model}) ===\n")

prompt = "Generate exactly 3 brand names for an autonomous software testing AI."

def test_temperature(temp_value: float, run_num: int):
    response = client.messages.create(
        model=model,
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}],
        extra_body={"temperature": temp_value}
    )
    text = ""
    for block in response.content:
        if isinstance(block, TextBlock):
            text += block.text
    return text.strip().replace("\n", " | ")

# ==============================================================================
# Experiment A: Temperature = 0.0 (High Determinism / Low Entropy)
# ==============================================================================
print("--- [A] Testing Temperature = 0.0 (Greedy Decoding / Deterministic) ---")
run1_t0 = test_temperature(0.0, 1)
run2_t0 = test_temperature(0.0, 2)
print(f"Run 1: {run1_t0}")
print(f"Run 2: {run2_t0}")
is_identical = (run1_t0 == run2_t0)
print(f"Are outputs identical?: {is_identical} (Expected: True or near-identical)")

# ==============================================================================
# Experiment B: Temperature = 1.0 (High Entropy / Creative Sampling)
# ==============================================================================
print("\n--- [B] Testing Temperature = 1.0 (Stochastic Sampling) ---")
run1_t1 = test_temperature(1.0, 1)
run2_t1 = test_temperature(1.0, 2)
print(f"Run 1: {run1_t1}")
print(f"Run 2: {run2_t1}")

# ==============================================================================
# Architectural Takeaway
# ==============================================================================
print("\n" + "="*70)
print("ARCHITECT NOTE:")
print("1. Temperature controls token sampling probability distribution.")
print("2. Use temperature=0.0 for: Code generation, RAG answer extraction, JSON output,")
print("   classification, and evaluation benchmarks (reproducibility is critical).")
print("3. Use temperature=0.7 - 1.0 for: Brainstorming, diverse synthetic data generation,")
print("   and open-ended roleplay.")
print("4. Caution: Even at temperature=0.0, floating-point GPU non-determinism can occasionally")
print("   yield minor token shifts over time across distributed clusters.")
print("="*70)
