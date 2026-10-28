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

print(f"=== 01: Why Evals Matter — Beyond 'Vibe Checking' (Model: {model}) ===\n")

# ==============================================================================
# The Scenario: Production Support Ticket Routing
# Allowed categories: [BILLING, TECHNICAL, ACCOUNT, FEATURE_REQUEST]
# ==============================================================================

# Golden Test Dataset (Diverse edge cases representing real-world users)
TEST_SUITE = [
    {
        "id": 1,
        "input": "My app keeps crashing whenever I open the analytics dashboard.",
        "expected": "TECHNICAL",
        "type": "Standard clear-cut"
    },
    {
        "id": 2,
        "input": "I was charged twice on my credit card for the monthly subscription.",
        "expected": "BILLING",
        "type": "Standard clear-cut"
    },
    {
        "id": 3,
        "input": "Could you add a dark mode toggle to the settings page?",
        "expected": "FEATURE_REQUEST",
        "type": "Standard clear-cut"
    },
    {
        "id": 4,
        "input": "I can't log in because password reset email never arrives, and now my card was billed anyway!",
        "expected": "ACCOUNT",
        "type": "Multi-intent / Ambiguous (Primary blocker is Account access)"
    },
    {
        "id": 5,
        "input": "Great job guys, your server is down again on a Monday morning.",
        "expected": "TECHNICAL",
        "type": "Sarcasm / Sentiment noise"
    }
]

# ==============================================================================
# Path 1 & 2: The "Vibe Check" Prompt (Drafted and tested casually)
# ==============================================================================
def prompt_v1_naive(ticket_text: str) -> str:
    """Naive prompt without constraints or strict formatting."""
    response = client.messages.create(
        model=model,
        max_tokens=100,
        messages=[
            {
                "role": "user",
                "content": f"Categorize this support ticket into BILLING, TECHNICAL, ACCOUNT, or FEATURE_REQUEST:\n{ticket_text}"
            }
        ]
    )
    for block in response.content:
        if isinstance(block, TextBlock):
            return block.text.strip()
    return ""

# ==============================================================================
# Path 3: The Systematic Evaluation-First Prompt
# Structured with clear disambiguation hierarchy and assistant prefilling
# ==============================================================================
def prompt_v2_engineered(ticket_text: str) -> str:
    """Engineered prompt with priority hierarchy and constrained output."""
    system_prompt = (
        "You are an automated support ticket triage classifier. "
        "Classify the ticket into exactly one of: BILLING, TECHNICAL, ACCOUNT, FEATURE_REQUEST.\n"
        "Rules:\n"
        "- If login/authentication prevents access, classify as ACCOUNT even if billing is mentioned.\n"
        "- Sarcasm or complaints about system outage must be classified as TECHNICAL.\n"
        "- Respond with ONLY the single category name in capital letters. No explanations."
    )
    
    response = client.messages.create(
        model=model,
        system=system_prompt,
        max_tokens=20,
        extra_body={"temperature": 0.0},  # Deterministic decoding for classification
        messages=[
            {
                "role": "user",
                "content": f"<ticket>\n{ticket_text}\n</ticket>"
            }
        ]
    )
    for block in response.content:
        if isinstance(block, TextBlock):
            # Clean up punctuation/whitespace
            return block.text.strip().replace(".", "")
    return ""

# ==============================================================================
# Evaluation Harness: Objective Scoring Pipeline
# ==============================================================================
def run_evaluation(prompt_func, prompt_name: str):
    print(f"\n--- Running Evaluation Pipeline for: {prompt_name} ---")
    passed = 0
    total = len(TEST_SUITE)
    
    for test in TEST_SUITE:
        raw_output = prompt_func(test["input"])
        # Code-based deterministic grading: Does the output exactly match or contain the expected label?
        is_pass = (test["expected"] == raw_output)
        status_icon = "PASS" if is_pass else "FAIL"
        if is_pass:
            passed += 1
            
        print(f"[{status_icon}] Test #{test['id']} ({test['type']})")
        print(f"       Expected: {test['expected']}")
        print(f"       Received: {raw_output}")
    
    accuracy = (passed / total) * 100
    print(f"\n>> Result for {prompt_name}: {passed}/{total} Passed ({accuracy:.1f}% Accuracy)")
    return accuracy

# ==============================================================================
# Execution: Comparing Vibe Check vs Evaluation-Driven Refinement
# ==============================================================================
print("Trap 1: Testing on a single sample gives a false sense of security.")
sample_output = prompt_v1_naive(TEST_SUITE[0]["input"])
print(f"Single test: Input='{TEST_SUITE[0]['input']}' -> Output='{sample_output}' (Looks fine!)\n")

print("Trap 2: But real users provide multi-intent, sarcasm, and conversational nuance.")
print("Now running full automated evaluation across both prompt candidates...\n")

acc_v1 = run_evaluation(prompt_v1_naive, "Prompt V1 (Naive / Vibe Check)")
acc_v2 = run_evaluation(prompt_v2_engineered, "Prompt V2 (Eval-Refined)")

# ==============================================================================
# Architectural Takeaway
# ==============================================================================
print("\n" + "="*70)
print("ARCHITECT NOTE:")
print("1. Prompt Engineering vs Prompt Evaluation:")
print("   - Prompt engineering is how you WRITE prompts (XML tags, few-shot, system steering).")
print("   - Prompt evaluation is how you MEASURE if changes actually improve the system.")
print("2. The 'Whack-a-Mole' Trap (Prompt Regression):")
print("   Fixing a prompt for one edge case often silently breaks another previously-working case.")
print("   Without an automated eval harness, you cannot detect regressions.")
print("3. Evals are the Unit Tests of Generative AI Architecture:")
print("   Never deploy prompt changes to production based on intuition ('vibe checks').")
print("   Every prompt refinement must be backed by benchmark metrics on a golden dataset.")
print("="*70)
