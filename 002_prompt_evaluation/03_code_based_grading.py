"""Code-Based Syntax Grading + Model-Based Content Grading.

Production-grade evaluation script:
1. Formats: Validates Python, JSON, and Regex using deterministic AST/parser checks.
2. Valid Syntax: AST parse, JSON decoder, Re compile.
3. Content & Task Following: Evaluated via LLM-as-a-judge.
4. Combined Score: Average of (Model Score + Syntax Score).
"""

import ast
import json
import os
import re
from pathlib import Path
from statistics import mean
from typing import Any

from anthropic import Anthropic
from anthropic.types import MessageParam, TextBlock
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
model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")

print(f"=== Code-Based Grading & Composite Evaluation (Model: {model}) ===\n")


# ==============================================================================
# 2. Conversation Helper Functions
# ==============================================================================
def add_user_message(messages: list[MessageParam], text: str) -> None:
    """Append a user turn to the conversation history."""
    messages.append({"role": "user", "content": text})


def add_assistant_message(messages: list[MessageParam], text: str) -> None:
    """Append an assistant turn (for prefilling or multi-turn history)."""
    messages.append({"role": "assistant", "content": text})


def chat(
    messages: list[MessageParam],
    system: str | None = None,
    stop_sequences: list[str] | None = None,
    max_tokens: int = 1000,
) -> str:
    """Send a conversation to the configured model and extract text output."""
    request_params: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
        "stop_sequences": stop_sequences or [],
    }
    if system:
        request_params["system"] = system

    response = client.messages.create(**request_params)

    for block in response.content:
        if isinstance(block, TextBlock):
            return block.text.strip()
    return ""


# ==============================================================================
# 3. Code-Based Syntax Validation (Deterministic Grader)
# Fast, free, non-flaky checks for technical validity
# ==============================================================================
def validate_json(text: str) -> int:
    """Check if text is syntactically valid JSON. Returns 10 if valid, else 0."""
    try:
        json.loads(text.strip())
        return 10
    except (json.JSONDecodeError, ValueError):
        return 0


def validate_python(text: str) -> int:
    """Check if text parses into a valid Python Abstract Syntax Tree (AST)."""
    try:
        ast.parse(text.strip())
        return 10
    except SyntaxError:
        return 0


def validate_regex(text: str) -> int:
    """Check if text compiles into a valid regular expression pattern."""
    try:
        re.compile(text.strip())
        return 10
    except re.error:
        return 0


def grade_syntax(response: str, test_case: dict[str, str]) -> int:
    """Route the generated text to the appropriate syntax validator."""
    fmt = test_case.get("format", "").lower()
    if fmt == "json":
        return validate_json(response)
    elif fmt == "python":
        return validate_python(response)
    elif fmt == "regex":
        return validate_regex(response)
    else:
        print(f"Warning: Unknown format '{fmt}'. Skipping syntax score.")
        return 0


# ==============================================================================
# 4. System Under Test: Generating Pure Code via Assistant Prefill
# ==============================================================================
def run_prompt(test_case: dict[str, str]) -> str:
    """
    Executes the prompt against Claude.
    Uses assistant prefilling with ```code and stop sequence ``` to guarantee
    raw code generation without conversational preamble.
    """
    prompt = f"""
Please solve the following task:

{test_case["task"]}

* Respond only with Python, JSON, or a plain Regex
* Do not add any comments or commentary or explanation
"""
    messages: list[MessageParam] = []
    add_user_message(messages, prompt.strip())
    add_assistant_message(messages, "```code\n")

    output = chat(messages, stop_sequences=["```"])
    return output


# ==============================================================================
# 5. Model-Based Grader (LLM-as-a-Judge for Semantic Quality)
# ==============================================================================
def grade_by_model(test_case: dict[str, str], output: str) -> dict[str, Any]:
    """Evaluates task following, correctness, and completeness via Claude."""
    eval_prompt = f"""
You are an expert AWS code reviewer. Your task is to evaluate the following AI-generated solution.

Original Task:
<task>
{test_case["task"]}
</task>

Expected Format: {test_case["format"]}

Solution to Evaluate:
<solution>
{output}
</solution>

Output Format:
Provide your evaluation as a structured JSON object with these keys:
- "strengths": string[] (1-3 key strengths)
- "weaknesses": string[] (1-3 key areas for improvement)
- "reasoning": string (concise assessment)
- "score": number between 1-10

Respond ONLY with valid JSON.
"""
    messages: list[MessageParam] = []
    add_user_message(messages, eval_prompt.strip())
    add_assistant_message(messages, "{\n")

    eval_text = chat(messages, stop_sequences=["}"])
    full_json_str = "{\n" + eval_text + "\n}"

    try:
        parsed = json.loads(full_json_str)
        return {
            "score": float(parsed.get("score", 5.0)),
            "reasoning": str(parsed.get("reasoning", "No reasoning provided.")),
            "strengths": parsed.get("strengths", []),
            "weaknesses": parsed.get("weaknesses", []),
        }
    except Exception:
        # Fallback regex extraction if full JSON parsing encounters an edge case
        match = re.search(r'"score"\s*:\s*(\d+(?:\.\d+)?)', full_json_str)
        score = float(match.group(1)) if match else 5.0
        return {
            "score": score,
            "reasoning": "Parsed score from model review.",
            "strengths": [],
            "weaknesses": [],
        }


# ==============================================================================
# 6. Hybrid Test Execution & Composite Scoring
# ==============================================================================
def run_test_case(test_case: dict[str, str]) -> dict[str, Any]:
    """Generates code, runs code syntax grading, runs model grading, and combines scores."""
    output = run_prompt(test_case)

    # 1. Deterministic Code-Based Syntax Score (0 or 10)
    syntax_score = grade_syntax(output, test_case)

    # 2. Semantic Model-Based Score (1 to 10)
    model_grade = grade_by_model(test_case, output)
    model_score = model_grade["score"]

    # 3. Composite Score: 50% technical correctness + 50% task adherence
    composite_score = (model_score + syntax_score) / 2

    return {
        "test_case": test_case,
        "output": output,
        "syntax_score": syntax_score,
        "model_score": model_score,
        "composite_score": composite_score,
        "reasoning": model_grade["reasoning"],
    }


def run_eval(dataset: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Runs evaluation over the entire dataset and prints a structured benchmark table."""
    results = []

    print(f"{'Task / Format':<45} | {'Syntax':<8} | {'Model':<8} | {'Final':<8}")
    print("-" * 75)

    for case in dataset:
        result = run_test_case(case)
        results.append(result)

        task_label = f"[{case['format'].upper()}] {case['task'][:35]}..."
        print(
            f"{task_label:<45} | "
            f"{result['syntax_score']:<8.1f} | "
            f"{result['model_score']:<8.1f} | "
            f"{result['composite_score']:<8.1f}"
        )

    avg_syntax = mean(r["syntax_score"] for r in results)
    avg_model = mean(r["model_score"] for r in results)
    avg_composite = mean(r["composite_score"] for r in results)

    print("-" * 75)
    print(f"{'AVERAGES':<45} | {avg_syntax:<8.1f} | {avg_model:<8.1f} | {avg_composite:<8.1f}")
    return results


# ==============================================================================
# 7. Dataset Loading & Main Entrypoint
# ==============================================================================
def load_or_create_dataset() -> list[dict[str, str]]:
    """Load existing dataset.json or fall back to standard AWS tasks."""
    dataset_file = Path(__file__).resolve().parent / "dataset.json"
    if dataset_file.exists():
        try:
            return json.loads(dataset_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Built-in fallback dataset matching course syllabus
    return [
        {
            "task": "Create a JSON configuration for an AWS Lambda function that sets environment variables for database connection",
            "format": "json",
        },
        {
            "task": "Write a Python function to extract the AWS region from an ARN (Amazon Resource Name) string",
            "format": "python",
        },
        {
            "task": "Design a regular expression to validate a valid AWS EC2 instance ID (format: i-[alphanumeric characters])",
            "format": "regex",
        },
    ]


def main() -> None:
    dataset = load_or_create_dataset()
    results = run_eval(dataset)

    # Print one sample detailed output
    sample = results[0]
    print("\n" + "=" * 70)
    print("SAMPLE DETAILED EVALUATION")
    print("=" * 70)
    print(f"Task:    {sample['test_case']['task']}")
    print(f"Format:  {sample['test_case']['format']}")
    print(f"Syntax Score: {sample['syntax_score']}/10")
    print(f"Model Score:  {sample['model_score']}/10 (Final: {sample['composite_score']}/10)")
    print(f"Judge Reasoning: {sample['reasoning']}")
    print("\nGenerated Code:")
    print(sample["output"].strip())

    # Architectural Takeaway
    print("\n" + "=" * 70)
    print("ARCHITECT NOTE:")
    print("1. Why Hybrid Grading (Code + Model):")
    print("   - Code graders are FAST, FREE, and 100% DETERMINISTIC (AST parsing never hallucinates).")
    print("   - Model graders are FLEXIBLE and SEMANTIC (understanding if the code actually solves the task).")
    print("2. The 'Gating Function' Pattern:")
    print("   In high-volume CI/CD pipelines, use the code grader as a gating check:")
    print("   If syntax_score == 0, immediately fail the test with Score=0 and SKIP the model grader call.")
    print("   This cuts evaluation API costs by 50% for broken candidates.")
    print("3. Composite Weighting:")
    print("   Depending on system requirements, adjust weights:")
    print("   e.g., Score = (0.7 * Syntax) + (0.3 * Model) for strict execution sandboxes.")
    print("=" * 70)


if __name__ == "__main__":
    main()
