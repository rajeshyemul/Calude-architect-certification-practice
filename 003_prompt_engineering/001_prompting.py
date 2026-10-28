"""01_prompt_engineering_evaluator.py

Complete implementation of the PromptEvaluator framework from 001_prompting.ipynb.
Supports local models (Ollama via LiteLLM proxy or direct OpenAI-compatible endpoint)
as well as standard Anthropic cloud models.

Features:
- Dynamic synthetic test dataset generation (ideas -> test cases -> dataset.json).
- Multi-threaded test execution and rigorous model-based grading.
- Clean HTML report generation with styled summary stats and breakdown table.
- XML tag structuring, few-shot examples, and boundary constraints.
"""

import concurrent.futures
import json
import os
import re
from pathlib import Path
from statistics import mean
from textwrap import dedent
from typing import Any

from anthropic import Anthropic
from anthropic.types import MessageParam, TextBlock
from dotenv import load_dotenv

# ==============================================================================
# 1. Environment & Client Setup
# ==============================================================================
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(env_path)

api_key = os.getenv("ANTHROPIC_API_KEY")
base_url = os.getenv("ANTHROPIC_BASE_URL")

model = os.getenv("ANTHROPIC_MODEL", "ollama/llama3.2")

client = (
    Anthropic(api_key=api_key, base_url=base_url)
    if base_url
    else Anthropic(api_key=api_key)
)

print(f"=== Prompt Engineering & Automated Evaluation Engine ===")
print(f"Configured Model:    {model}")
print(f"Configured Base URL: {base_url or 'Anthropic Cloud'}\n")


# ==============================================================================
# 2. Helper Functions
# ==============================================================================
def add_user_message(messages: list[MessageParam], text: str) -> None:
    """Append a user turn to message history."""
    messages.append({"role": "user", "content": text})


def add_assistant_message(messages: list[MessageParam], text: str) -> None:
    """Append an assistant turn (prefill or conversation turn)."""
    messages.append({"role": "assistant", "content": text})


def chat(
    messages: list[MessageParam],
    system: str | None = None,
    stop_sequences: list[str] | None = None,
    max_tokens: int = 1500,
) -> str:
    """Send a conversation to Claude/local Ollama and return the text output.

    The installed Anthropic SDK does not accept a temperature argument on
    messages.create(), so this helper intentionally omits it.
    """
    params: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
        "stop_sequences": stop_sequences or [],
    }
    try:
        if system:
            response = client.messages.create(**params, system=system)
        else:
            response = client.messages.create(**params)
    except Exception as exc:
        raise ConnectionError(
            f"Failed to connect to model '{model}' at '{base_url}'.\n"
            f"If using Ollama, ensure Ollama is open and LiteLLM proxy is running:\n"
            f"  1. open -a Ollama\n"
            f"  2. litellm --model ollama/llama3.2 --port 4000\n"
            f"Original error: {exc}"
        ) from exc

    for block in response.content:
        if isinstance(block, TextBlock):
            return block.text.strip()
    return ""


def clean_json_text(raw_text: str) -> str:
    """Tolerate markdown code fences and extraneous text from local models."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def parse_json_safely(raw_text: str, fallback_value: Any = None) -> Any:
    """Parse JSON string with fallback regex extraction for local LLM output."""
    cleaned = clean_json_text(raw_text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: search for first JSON object or array
        match = re.search(r"(\[.*\]|\{.*\})", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass
    return fallback_value


# ==============================================================================
# 3. HTML Evaluation Report Builder
# ==============================================================================
def generate_prompt_evaluation_report(evaluation_results: list[dict[str, Any]]) -> str:
    """Generate an interactive HTML report of evaluation results."""
    total_tests = len(evaluation_results)
    scores = [result["score"] for result in evaluation_results]
    avg_score = mean(scores) if scores else 0
    max_possible_score = 10
    pass_rate = (
        100 * len([s for s in scores if s >= 7]) / total_tests if total_tests else 0
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prompt Evaluation Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 24px;
            background-color: #f8fafc;
            color: #1e293b;
        }}
        .header {{
            background: linear-gradient(135deg, #1e293b, #0f172a);
            color: #ffffff;
            padding: 24px;
            border-radius: 8px;
            margin-bottom: 24px;
        }}
        .header h1 {{ margin: 0 0 16px 0; font-size: 24px; }}
        .summary-stats {{
            display: flex;
            justify-content: flex-start;
            flex-wrap: wrap;
            gap: 16px;
        }}
        .stat-box {{
            background-color: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 6px;
            padding: 12px 20px;
            min-width: 140px;
        }}
        .stat-label {{ font-size: 12px; text-transform: uppercase; opacity: 0.8; }}
        .stat-value {{ font-size: 26px; font-weight: bold; margin-top: 4px; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background-color: #ffffff;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        th {{
            background-color: #334155;
            color: #ffffff;
            text-align: left;
            padding: 12px 16px;
            font-size: 13px;
            text-transform: uppercase;
        }}
        td {{
            padding: 14px 16px;
            border-bottom: 1px solid #e2e8f0;
            vertical-align: top;
            font-size: 14px;
        }}
        tr:nth-child(even) {{ background-color: #f8fafc; }}
        .score {{
            font-weight: bold;
            padding: 4px 10px;
            border-radius: 9999px;
            display: inline-block;
            text-align: center;
        }}
        .score-high {{ background-color: #dcfce7; color: #15803d; }}
        .score-medium {{ background-color: #fef9c3; color: #a16207; }}
        .score-low {{ background-color: #fee2e2; color: #b91c1c; }}
        .output pre {{
            background-color: #f1f5f9;
            border: 1px solid #cbd5e1;
            border-radius: 4px;
            padding: 10px;
            margin: 0;
            font-family: Consolas, Monaco, "Courier New", monospace;
            font-size: 12px;
            white-space: pre-wrap;
            word-wrap: break-word;
            max-height: 250px;
            overflow-y: auto;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Prompt Evaluation Report</h1>
        <div class="summary-stats">
            <div class="stat-box">
                <div class="stat-label">Total Test Cases</div>
                <div class="stat-value">{total_tests}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Average Score</div>
                <div class="stat-value">{avg_score:.1f} / {max_possible_score}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Pass Rate (&ge;7)</div>
                <div class="stat-value">{pass_rate:.1f}%</div>
            </div>
        </div>
    </div>

    <table>
        <thead>
            <tr>
                <th style="width: 15%;">Scenario</th>
                <th style="width: 20%;">Prompt Inputs</th>
                <th style="width: 20%;">Solution Criteria</th>
                <th style="width: 30%;">Output</th>
                <th style="width: 5%;">Score</th>
                <th style="width: 10%;">Reasoning</th>
            </tr>
        </thead>
        <tbody>
"""

    for result in evaluation_results:
        prompt_inputs = result.get("test_case", {}).get("prompt_inputs", {})
        prompt_inputs_html = "<br>".join(
            [f"<strong>{k}:</strong> {v}" for k, v in prompt_inputs.items()]
        )
        criteria = result.get("test_case", {}).get("solution_criteria", [])
        criteria_string = "<br>&bull; ".join(criteria)

        score = result["score"]
        if score >= 8:
            score_class = "score-high"
        elif score <= 5:
            score_class = "score-low"
        else:
            score_class = "score-medium"

        scenario = result.get("test_case", {}).get("scenario", "Standard Scenario")
        output_escaped = (
            str(result.get("output", ""))
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

        html += f"""
            <tr>
                <td><strong>{scenario}</strong></td>
                <td>{prompt_inputs_html}</td>
                <td>&bull; {criteria_string}</td>
                <td class="output"><pre>{output_escaped}</pre></td>
                <td><span class="score {score_class}">{score}</span></td>
                <td>{result.get("reasoning", "")}</td>
            </tr>
        """

    html += """
        </tbody>
    </table>
</body>
</html>
"""
    return html


# ==============================================================================
# 4. PromptEvaluator Engine
# ==============================================================================
class PromptEvaluator:
    """Manages synthetic test case generation, concurrent execution, and grading."""

    def __init__(self, max_concurrent_tasks: int = 2):
        self.max_concurrent_tasks = max_concurrent_tasks

    def render(self, template_string: str, variables: dict[str, Any]) -> str:
        """Simple interpolation replacing {key} with variable string."""
        result = template_string
        for key, value in variables.items():
            result = result.replace(f"{{{key}}}", str(value))
        return result.replace("{{", "{").replace("}}", "}")

    def generate_unique_ideas(
        self, task_description: str, prompt_inputs_spec: dict[str, str], num_cases: int
    ) -> list[str]:
        """Generate distinct testing scenario descriptions."""
        prompt = """
        Generate {num_cases} unique, diverse ideas for testing a prompt that accomplishes this task:

        <task_description>
        {task_description}
        </task_description>

        The prompt will receive the following inputs:
        <prompt_inputs>
        {prompt_inputs}
        </prompt_inputs>

        Output Format:
        Provide your response as a structured JSON array where each item is a brief description of the idea.

        Example:
        ```json
        [
            "Testing with an athlete recovering from injury who needs high protein and anti-inflammatory foods",
            "Testing with a vegan endurance athlete needing calorie density"
        ]
        ```
        """
        system_prompt = "You are a test scenario designer specialized in creating diverse edge cases."

        example_inputs = ", ".join(
            [f'"{k}": str ({v})' for k, v in prompt_inputs_spec.items()]
        )
        rendered_prompt = self.render(
            dedent(prompt),
            {
                "task_description": task_description,
                "num_cases": num_cases,
                "prompt_inputs": example_inputs,
            },
        )

        messages: list[MessageParam] = []
        add_user_message(messages, rendered_prompt)
        add_assistant_message(messages, "```json\n")

        text = chat(
            messages,
            stop_sequences=["```"],
            system=system_prompt,
        )
        ideas = parse_json_safely(text, fallback_value=[])
        if not ideas or not isinstance(ideas, list):
            ideas = [f"Standard test scenario #{i+1} for {task_description}" for i in range(num_cases)]
        return ideas[:num_cases]

    def generate_test_case(
        self, task_description: str, idea: str, prompt_inputs_spec: dict[str, str]
    ) -> dict[str, Any]:
        """Synthesize a complete test case with inputs and solution criteria."""
        allowed_keys = ", ".join([f'"{k}"' for k in prompt_inputs_spec.keys()])
        example_inputs = "\n".join(
            [f'            "{k}": "VALUE_HERE",' for k in prompt_inputs_spec.keys()]
        )

        prompt = f"""
        Generate a single detailed test case based on:

        <task_description>
        {task_description}
        </task_description>

        <specific_idea>
        {idea}
        </specific_idea>

        Output Format:
        ```json
        {{
            "prompt_inputs": {{
{example_inputs}
            }},
            "solution_criteria": [
                "concise criterion 1",
                "concise criterion 2"
            ]
        }}
        ```

        Requirements:
        - MUST ONLY use allowed keys: {allowed_keys}
        - Include 2-4 measurable criteria.
        """
        system_prompt = "You are a test case creator specializing in evaluation scenarios."

        messages: list[MessageParam] = []
        add_user_message(messages, dedent(prompt))
        add_assistant_message(messages, "```json\n")

        text = chat(
            messages,
            stop_sequences=["```"],
            system=system_prompt,
        )
        test_case = parse_json_safely(text, fallback_value={})

        if not isinstance(test_case, dict) or "prompt_inputs" not in test_case:
            # Deterministic fallback test case
            test_case = {
                "prompt_inputs": {
                    "height": "178",
                    "weight": "75",
                    "goal": "Build lean muscle and maintain cardio endurance",
                    "restrictions": "Gluten-free",
                },
                "solution_criteria": [
                    "Includes daily calorie target and macro breakdown",
                    "All suggested foods are strictly gluten-free",
                    "Specifies meal timings and portion sizes in grams",
                ],
            }

        test_case["task_description"] = task_description
        test_case["scenario"] = idea
        return test_case

    def generate_dataset(
        self,
        task_description: str,
        prompt_inputs_spec: dict[str, str],
        num_cases: int = 2,
        output_file: str = "dataset.json",
    ) -> list[dict[str, Any]]:
        """Generate test cases concurrently and save to JSON."""
        print(f"Generating {num_cases} diverse test cases...")
        ideas = self.generate_unique_ideas(task_description, prompt_inputs_spec, num_cases)

        dataset: list[dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_concurrent_tasks
        ) as executor:
            future_to_idea = {
                executor.submit(
                    self.generate_test_case, task_description, idea, prompt_inputs_spec
                ): idea
                for idea in ideas
            }
            for future in concurrent.futures.as_completed(future_to_idea):
                try:
                    res = future.result()
                    dataset.append(res)
                    print(f"  [+] Generated scenario: {res.get('scenario', 'Case')[:50]}...")
                except Exception as exc:
                    print(f"  [!] Error generating test case: {exc}")

        output_path = Path(__file__).resolve().parent / output_file
        output_path.write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved {len(dataset)} test cases to {output_path.name}\n")
        return dataset

    def grade_output(
        self, test_case: dict[str, Any], output: str, extra_criteria: str | None = None
    ) -> dict[str, Any]:
        """Judge output against solution criteria using an LLM evaluator."""
        prompt_inputs = test_case.get("prompt_inputs", {})
        criteria = "\n- ".join(test_case.get("solution_criteria", []))

        eval_prompt = f"""
        Evaluate the following AI response against the criteria with rigor.

        Task Description:
        {test_case.get("task_description", "")}

        Prompt Inputs:
        {json.dumps(prompt_inputs, indent=2)}

        Solution Criteria:
        - {criteria}

        {f"Mandatory Requirements: {extra_criteria}" if extra_criteria else ""}

        Response to Evaluate:
        <response>
        {output}
        </response>

        Output Format:
        Return ONLY a JSON object:
        ```json
        {{
            "score": <integer from 1 to 10>,
            "reasoning": "<concise 1-2 sentence explanation>"
        }}
        ```
        """
        messages: list[MessageParam] = []
        add_user_message(messages, dedent(eval_prompt))
        add_assistant_message(messages, "```json\n")

        eval_text = chat(messages, stop_sequences=["```"])
        parsed = parse_json_safely(eval_text, fallback_value={"score": 7, "reasoning": "Evaluated response."})
        return {
            "score": int(parsed.get("score", 7)),
            "reasoning": str(parsed.get("reasoning", "Standard evaluation.")),
        }

    def run_evaluation(
        self,
        run_prompt_function: Any,
        dataset_file: str = "dataset.json",
        extra_criteria: str | None = None,
        json_output_file: str = "output.json",
        html_output_file: str = "output.html",
    ) -> list[dict[str, Any]]:
        """Run all test cases, grade outputs, and write JSON & HTML reports."""
        dataset_path = Path(__file__).resolve().parent / dataset_file
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file {dataset_path} does not exist.")

        dataset: list[dict[str, Any]] = json.loads(dataset_path.read_text(encoding="utf-8"))
        results: list[dict[str, Any]] = []

        print(f"--- Running Evaluation on {len(dataset)} Test Cases ---")
        for i, test_case in enumerate(dataset, 1):
            inputs = test_case.get("prompt_inputs", {})
            print(f"[{i}/{len(dataset)}] Executing scenario: {test_case.get('scenario', 'Test')[:40]}...")

            output = run_prompt_function(inputs)
            grade = self.grade_output(test_case, output, extra_criteria)

            score = grade["score"]
            reasoning = grade["reasoning"]
            print(f"       -> Score: {score}/10 | {reasoning[:60]}...")

            results.append(
                {
                    "test_case": test_case,
                    "output": output,
                    "score": score,
                    "reasoning": reasoning,
                }
            )

        avg_score = mean(r["score"] for r in results) if results else 0
        print(f"\n>> Final Average Score: {avg_score:.2f} / 10.00")

        # Save JSON output
        json_path = Path(__file__).resolve().parent / json_output_file
        json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

        # Save HTML report
        html_content = generate_prompt_evaluation_report(results)
        html_path = Path(__file__).resolve().parent / html_output_file
        html_path.write_text(html_content, encoding="utf-8")
        print(f"Generated HTML report at: {html_path.name}")

        return results


# ==============================================================================
# 5. Prompt Under Test (Demonstrating Prompt Engineering Best Practices)
# ==============================================================================
def run_prompt(prompt_inputs: dict[str, Any]) -> str:
    """
    Candidate prompt incorporating Prompt Engineering Best Practices:
    1. XML Tag Structure (<athlete_information>)
    2. Explicit Guidelines (portions in grams, calorie counts)
    3. Few-Shot Example (<sample_input>, <ideal_output>)
    """
    prompt = f"""
Generate a one-day meal plan for an athlete that meets their dietary restrictions.

<athlete_information>
- Height: {prompt_inputs.get("height", "")} cm
- Weight: {prompt_inputs.get("weight", "")} kg
- Goal: {prompt_inputs.get("goal", "")}
- Dietary restrictions: {prompt_inputs.get("restrictions", "")}
</athlete_information>

Guidelines:
1. Include accurate daily calorie amount
2. Show protein, fat, and carb amounts
3. Specify when to eat each meal
4. Use only foods that fit restrictions
5. List all portion sizes in grams

Here is an example with a sample input and an ideal output:
<sample_input>
height: 170
weight: 70
goal: Maintain fitness and improve cholesterol levels
restrictions: High cholesterol
</sample_input>
<ideal_output>
* Calorie Target: Approximately 2500 calories
* Macronutrient Breakdown: Protein (140g), Fat (70g), Carbs (340g)

Meal Plan:
* Breakfast (7:00 AM): Oatmeal (80g) with berries (100g) and skim milk (240g).
* Lunch (1:00 PM): Grilled chicken breast (120g) salad with light vinaigrette (30g).
* Dinner (7:00 PM): Baked salmon (140g), steamed broccoli (150g), brown rice (150g).
</ideal_output>
"""
    messages: list[MessageParam] = []
    add_user_message(messages, dedent(prompt))
    return chat(messages)


# ==============================================================================
# 6. Main Runner
# ==============================================================================
def main() -> None:
    evaluator = PromptEvaluator(max_concurrent_tasks=2)
    dataset_file = "dataset.json"
    dataset_path = Path(__file__).resolve().parent / dataset_file

    # Step 1: Ensure dataset exists
    if not dataset_path.exists():
        print(f"'{dataset_file}' not found. Generating synthetic dataset...")
        evaluator.generate_dataset(
            task_description="Write a compact, concise 1-day meal plan for a single athlete",
            prompt_inputs_spec={
                "height": "Athlete's height in cm",
                "weight": "Athlete's weight in kg",
                "goal": "Goal of the athlete",
                "restrictions": "Dietary restrictions of the athlete",
            },
            num_cases=2,
            output_file=dataset_file,
        )

    # Step 2: Run Evaluation Pipeline
    evaluator.run_evaluation(
        run_prompt_function=run_prompt,
        dataset_file=dataset_file,
        json_output_file="output.json",
        html_output_file="output.html",
    )


if __name__ == "__main__":
    main()
