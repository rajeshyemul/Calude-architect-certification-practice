import os
import json
from dotenv import load_dotenv
from anthropic import Anthropic
from anthropic.types import TextBlock

# 1. Environment & Client Setup
load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")
base_url = os.getenv("ANTHROPIC_BASE_URL")
client = Anthropic(api_key=api_key, base_url=base_url) if base_url else Anthropic(api_key=api_key)
model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")

print(f"=== 02: A Typical Eval Workflow — Closed-Loop Benchmark (Model: {model}) ===\n")

# ==============================================================================
# Step 2: Evaluation Dataset
# Represents real-world sample inputs covering diverse domains
# ==============================================================================
EVAL_DATASET = [
    {
        "id": "math_basic",
        "question": "What's 2+2?",
        "expected_trait": "Direct, concise mathematical correctness"
    },
    {
        "id": "recipe_procedural",
        "question": "How do I make oatmeal?",
        "expected_trait": "Clear step-by-step cooking steps, ingredient ratios, and serving tips"
    },
    {
        "id": "scientific_factual",
        "question": "How far away is the Moon?",
        "expected_trait": "Accurate distance with context (average ~238,855 miles / 384,400 km)"
    }
]

# ==============================================================================
# Step 1: Draft Candidate Prompts (Baseline V1 vs Iterated V2)
# ==============================================================================
def prompt_v1_baseline(question: str) -> str:
    """Step 1 (Baseline): Simple and naive prompt."""
    return f"Please answer the user's question:\n\n{question}"

def prompt_v2_detailed(question: str) -> str:
    """Step 5 (Iteration): Enhanced prompt with explicit guidance for depth & clarity."""
    return (
        f"Please answer the user's question:\n\n{question}\n\n"
        f"Answer the question with ample detail, clear step-by-step instructions where applicable, "
        f"and accurate context."
    )

# ==============================================================================
# Step 3: Feed Through Claude (Generator)
# ==============================================================================
def generate_answer(prompt_text: str) -> str:
    """Sends candidate prompt to Claude and extracts response text."""
    response = client.messages.create(
        model=model,
        max_tokens=400,
        messages=[{"role": "user", "content": prompt_text}]
    )
    for block in response.content:
        if isinstance(block, TextBlock):
            return block.text.strip()
    return ""

# ==============================================================================
# Step 4: Feed Through a Grader (LLM-as-a-Judge)
# Evaluates question + answer on an objective 1 to 10 scale
# ==============================================================================
def grade_answer(question: str, generated_answer: str) -> dict:
    """
    Grader evaluates accuracy, completeness, and clarity on a 1-10 scale.
    Uses assistant prefilling ('{') to encourage clean JSON extraction.
    The installed Anthropic SDK's Messages API does not expose temperature.
    """
    judge_prompt = f"""
You are an expert AI response evaluator.
Evaluate the following generated answer to the user's question.

User Question:
{question}

Generated Answer:
{generated_answer}

Scoring Rubric (1 to 10):
- 10: Perfect answer. Accurate, comprehensive, clear, and perfectly tailored.
- 7-9: Good to very good. Accurate with minor omissions in detail or context.
- 4-6: Mediocre. Partially correct, overly sparse, or lacking necessary steps/depth.
- 1-3: Poor. Factually incorrect, confusing, or completely unhelpful.

Respond ONLY with a valid JSON object with exactly two keys:
"score": an integer from 1 to 10
"reasoning": a concise 1-2 sentence explanation of the score
"""

    response = client.messages.create(
        model=model,
        max_tokens=150,
        stop_sequences=["}"],
        messages=[
            {"role": "user", "content": judge_prompt},
            {"role": "assistant", "content": "{"}  # Prefill to enforce JSON
        ]
    )
    
    grader_text = next(
        (block.text.strip() for block in response.content if isinstance(block, TextBlock)),
        "",
    )
    raw_text = "{" + grader_text + "}"
    try:
        parsed = json.loads(raw_text)
        return {
            "score": int(parsed.get("score", 5)),
            "reasoning": str(parsed.get("reasoning", "No reason provided."))
        }
    except Exception:
        # Fallback heuristic if JSON parsing fails
        return {"score": 5, "reasoning": "Fallback grade due to JSON parse error."}

# ==============================================================================
# Step 5: Run Benchmark, Score, and Compare (The Iteration Loop)
# ==============================================================================
def run_eval_suite(prompt_func, prompt_label: str):
    print(f"\n" + "="*70)
    print(f"Running Eval Suite for: {prompt_label}")
    print("="*70)
    
    results = []
    total_score = 0
    
    for item in EVAL_DATASET:
        q = item["question"]
        print(f"\n[Test Case: {item['id']}] Q: \"{q}\"")
        
        # 1. Generate
        full_prompt = prompt_func(q)
        answer = generate_answer(full_prompt)
        print(f"-> Generated Answer Preview: {repr(answer[:90])}...")
        
        # 2. Grade
        grade_result = grade_answer(q, answer)
        score = grade_result["score"]
        reason = grade_result["reasoning"]
        
        print(f"-> Grader Score: {score}/10 | Reason: {reason}")
        
        total_score += score
        results.append({
            "id": item["id"],
            "question": q,
            "score": score,
            "reasoning": reason,
            "answer": answer
        })
        
    avg_score = total_score / len(EVAL_DATASET)
    print(f"\n>>> Average Score for {prompt_label}: {avg_score:.2f} / 10.00")
    return avg_score, results

# Execute Baseline (V1)
v1_avg, v1_results = run_eval_suite(prompt_v1_baseline, "Prompt V1 (Baseline)")

# Execute Iterated (V2)
v2_avg, v2_results = run_eval_suite(prompt_v2_detailed, "Prompt V2 (Enhanced Detail)")

# ==============================================================================
# Comparison Report & Delta Calculation
# ==============================================================================
print("\n" + "#"*70)
print("📊 PROMPT EVALUATION COMPARISON SUMMARY")
print("#"*70)
print(f"{'Test ID':<20} | {'V1 Score':<10} | {'V2 Score':<10} | {'Delta':<8}")
print("-" * 56)

for r1, r2 in zip(v1_results, v2_results):
    delta = r2["score"] - r1["score"]
    delta_str = f"+{delta}" if delta > 0 else f"{delta}"
    print(f"{r1['id']:<20} | {r1['score']:<10} | {r2['score']:<10} | {delta_str:<8}")

print("-" * 56)
overall_delta = v2_avg - v1_avg
sign = "+" if overall_delta > 0 else ""
print(f"Overall Average:       | {v1_avg:.2f}      | {v2_avg:.2f}      | {sign}{overall_delta:.2f}")

pct_gain = ((v2_avg - v1_avg) / v1_avg) * 100 if v1_avg > 0 else 0
print(f"Net Improvement:       | {pct_gain:+.1f}%")

# ==============================================================================
# Architectural Takeaway
# ==============================================================================
print("\n" + "="*70)
print("ARCHITECT NOTE:")
print("1. Closed-Loop Optimization:")
print("   Prompt engineering without automated scoring is subjective opinion.")
print("   By quantifying output quality (V1: {:.2f} vs V2: {:.2f}), teams make data-driven decisions.".format(v1_avg, v2_avg))
print("2. The Dual-Call Cost Equation:")
print("   Running an eval requires 2 LLM invocations per test case:")
print("   Cost = (N_test_cases) * [Cost(Generator) + Cost(Judge)].")
print("   Architect Tip: Use a faster/cheaper model (e.g. Claude 3.5 Haiku) for high-volume")
print("   generation, and a frontier model (Claude 3.5 Sonnet) as the impartial judge.")
print("3. Judge Calibration:")
print("   Supply a calibrated rubric (1-10 anchor points) to improve scoring consistency.")
print("   The installed Anthropic SDK does not expose temperature for Messages.create,")
print("   so repeated judge scores may still vary.")
print("="*70)
