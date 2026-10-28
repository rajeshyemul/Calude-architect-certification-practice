"""Generate AWS-focused prompt tests, run a model, and grade its output.

The model and Anthropic-compatible local proxy are read from this folder's .env.
"""

import json
import os
import re
from pathlib import Path
from statistics import mean
from typing import Any

from anthropic import Anthropic
from anthropic.types import MessageParam, TextBlock
from dotenv import load_dotenv


# ============================================================
# 1. Environment & Client
# ============================================================

load_dotenv(Path(__file__).resolve().parent / ".env")
api_key = os.getenv("ANTHROPIC_API_KEY")
base_url = os.getenv("ANTHROPIC_BASE_URL")
client = (
    Anthropic(api_key=api_key, base_url=base_url)
    if base_url
    else Anthropic(api_key=api_key)
)
model = os.getenv("ANTHROPIC_MODEL", "ollama/llama3.2")


# ============================================================
# 2. Helper Functions
# ============================================================

def add_user_message(messages: list[MessageParam], text: str) -> None:
    """Append a user message to a conversation."""
    messages.append({"role": "user", "content": text})


def add_assistant_message(messages: list[MessageParam], text: str) -> None:
    """Append an assistant message to a conversation."""
    messages.append({"role": "assistant", "content": text})


def chat(
    messages: list[MessageParam],
    system: str | None = None,
    max_tokens: int = 1000,
) -> str:
    """Call the configured local model and return its text response."""
    if system:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=messages,
            system=system,
        )
    else:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=messages,
        )
    if response.stop_reason == "max_tokens":
        raise ValueError(
            f"The model response reached its {max_tokens}-token limit and was truncated. "
            "Shorten the requested output or increase max_tokens."
        )
    for block in response.content:
        if isinstance(block, TextBlock):
            return block.text.strip()
    raise ValueError("The model response did not contain a text block.")


def parse_json_response(text: str, expected_type: type) -> Any:
    """Parse the expected JSON object/array despite model-added prose or fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```")
        cleaned = cleaned.removesuffix("```").strip()

    try:
        value = json.loads(cleaned)
        if isinstance(value, expected_type):
            return value
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    opening = "{" if expected_type is dict else "["
    for index, character in enumerate(cleaned):
        if character == opening:
            try:
                value, _ = decoder.raw_decode(cleaned, index)
            except json.JSONDecodeError:
                continue
            if isinstance(value, expected_type):
                return value

    raise ValueError(f"Model did not return valid JSON. Response: {text[:300]!r}")


def chat_json(
    messages: list[MessageParam],
    expected_type: type,
    format_hint: str,
    max_tokens: int = 2000,
    attempts: int = 3,
) -> Any:
    """Call the model until it returns a complete JSON value of the right type."""
    conversation = list(messages)
    expected_name = "JSON object" if expected_type is dict else "JSON array"
    last_error: ValueError | None = None

    for attempt in range(1, attempts + 1):
        response_text = chat(conversation, max_tokens=max_tokens)
        try:
            return parse_json_response(response_text, expected_type)
        except ValueError as exc:
            last_error = exc
            if attempt == attempts:
                break

            print(
                f"Invalid or incomplete {expected_name} from {model}; "
                f"retrying ({attempt + 1}/{attempts})..."
            )
            conversation.extend(
                [
                    {"role": "assistant", "content": response_text},
                    {
                        "role": "user",
                        "content": (
                            f"Your previous response was invalid or incomplete. "
                            f"Return a complete, valid {expected_name} only. "
                            f"{format_hint} Do not add Markdown or explanatory text."
                        ),
                    },
                ]
            )

    raise ValueError(
        f"Could not get a valid {expected_name} after {attempts} attempts. "
        f"Last parser error: {last_error}"
    ) from last_error


# ============================================================
# 3. Dataset Generation
# ============================================================

def generate_dataset() -> list[dict[str, str]]:
    """Generate three AWS-related tasks and their expected output formats."""
    prompt = """
Generate an evaluation dataset for testing prompts that produce Python, JSON,
or regular expressions for AWS-related tasks.

Return only a JSON array containing exactly 3 objects. Each object must have
string fields named "task" and "format". Each task should be small enough to
solve as one Python function, one JSON object, or one regular expression.
Do not include Markdown fences or text outside the JSON.
"""
    dataset = chat_json(
        [{"role": "user", "content": prompt}],
        list,
        'Each array item must have string fields "task" and "format".',
    )
    if not isinstance(dataset, list) or not all(
        isinstance(item, dict)
        and isinstance(item.get("task"), str)
        and isinstance(item.get("format"), str)
        for item in dataset
    ):
        raise ValueError('Dataset must be a JSON array with string "task" and "format" fields.')
    return dataset


# ============================================================
# 4. Model Grader
# ============================================================

def grade_by_model(test_case: dict[str, str], output: str) -> dict[str, Any]:
    """Ask the local model for a compact numeric score and validate it."""
    eval_prompt = f"""
You are an expert AWS code reviewer. Grade the solution for the task below.

Task:
<task>
{test_case["task"]}
</task>

Solution:
<solution>
{output}
</solution>

Evaluate correctness, completeness, and clarity. Return ONLY one integer from 1 to 10.
Do not include an explanation, Markdown, or any other text.
"""
    grader_response = chat(
        [{"role": "user", "content": eval_prompt}], max_tokens=64
    )
    score_match = re.search(
        r"\b(?:score\s*[:=]?\s*)?(10|[1-9])(?:\s*/\s*10)?\b",
        grader_response,
        re.IGNORECASE,
    )
    if not score_match:
        raise ValueError(
            "The local grader did not return a score from 1 to 10. "
            f"Response: {grader_response[:200]!r}"
        )

    score = int(score_match.group(1))
    return {
        "score": score,
        "reasoning": f"Score-only evaluation from {model}.",
    }


# ============================================================
# 5. System Under Test
# ============================================================

def run_prompt(test_case: dict[str, str]) -> str:
    """Ask the configured model to solve one generated task."""
    prompt = f"Please solve this task:\n\n{test_case['task']}"
    return chat([{"role": "user", "content": prompt}])


# ============================================================
# 6. Execute One Test Case
# ============================================================

def run_test_case(test_case: dict[str, str]) -> dict[str, Any]:
    """Generate and grade a response for one test case."""
    output = run_prompt(test_case)
    model_grade = grade_by_model(test_case, output)
    return {
        "output": output,
        "test_case": test_case,
        "score": model_grade["score"],
        "reasoning": model_grade["reasoning"],
    }


# ============================================================
# 7. Run Evaluation
# ============================================================

def run_eval(dataset: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Run all cases, print the average score, and return detailed results."""
    if not dataset:
        raise ValueError("Cannot run evaluation: the dataset is empty.")

    results = [run_test_case(test_case) for test_case in dataset]
    average_score = mean(result["score"] for result in results)
    print(f"Average score ({model}): {average_score:.2f}/10")
    return results


# ============================================================
# 8. Main Execution
# ============================================================

def main() -> None:
    """Generate tasks, evaluate them, and save outputs beside this script."""
    output_dir = Path(__file__).resolve().parent
    dataset = generate_dataset()
    (output_dir / "dataset.json").write_text(
        json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    results = run_eval(dataset)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()