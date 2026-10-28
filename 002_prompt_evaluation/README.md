# 002 - Prompt Evaluation & Guardrails

This directory covers systematic prompt evaluation methodologies, benchmarking harnesses, dataset generation, and grading strategies (code-based vs model-based) to build resilient AI pipelines.

---

## 📂 Capability Files

| File | Meaningful Capability | Architectural Concept |
| :--- | :--- | :--- |
| **[`01_why_evals_matter.py`](file:///Users/rajesh.yemul/Claude%20Certification/002_prompt_evaluation/01_why_evals_matter.py)** | Vibe Check vs Eval Harness | Escaping Option 1 & Option 2 traps; automated scoring on a golden dataset; preventing prompt regression. |
| **[`02_eval_workflow.py`](file:///Users/rajesh.yemul/Claude%20Certification/002_prompt_evaluation/02_eval_workflow.py)** | Closed-Loop Eval Workflow | 5-step cycle: Draft -> Dataset -> Inference -> LLM-as-a-Judge (1-10) -> Delta Comparison. |
| **[`002_prompt_evals_grader.py`](file:///Users/rajesh.yemul/Claude%20Certification/002_prompt_evaluation/002_prompt_evals_grader.py)** | Model-Based Grading | LLM-as-a-judge scoring code correctness, completeness, and clarity. |
| **[`03_code_based_grading.py`](file:///Users/rajesh.yemul/Claude%20Certification/002_prompt_evaluation/03_code_based_grading.py)** | Code Syntax + Model Composite | Deterministic AST/JSON/Regex validation (0/10) + LLM judge, combined composite score. |
| *(Upcoming)* `04_running_evals.py` | Running Batch Evals | Concurrency, rate-limiting, error handling, collecting eval traces. |
| *(Upcoming)* `05_prompt_eval_exercise.py` | End-to-End Eval Exercise | Iterative prompt refinement driven by measurable accuracy benchmarks. |

---

## 🧠 The 5 Architect Questions

### 1. What did I learn?
Prompt engineering without automated prompt evaluation is just "vibe checking." Without an evaluation pipeline, tweaking a prompt to solve one edge case often silently causes regressions on other cases (the "whack-a-mole" problem).

### 2. What is the implementation architecture?
```text
┌─────────────────────────────────────────────────────────────┐
│                    Evaluation Harness                       │
│                                                             │
│   ┌────────────────────┐          ┌──────────────────────┐  │
│   │   Golden Dataset   │ ───────> │  Candidate Prompt    │  │
│   │ (Inputs & Expected)│          │  (Version A / B)     │  │
│   └────────────────────┘          └──────────┬───────────┘  │
│                                              │              │
│                                              ▼              │
│                                   ┌──────────────────────┐  │
│                                   │ Claude API Inference │  │
│                                   └──────────┬───────────┘  │
│                                              │ Raw Output   │
│                                              ▼              │
│   ┌────────────────────┐          ┌──────────────────────┐  │
│   │  Aggregate Report  │ <─────── │ Grader (Code/Model)  │  │
│   │ (Pass/Fail/Metrics)│          │ (Pass/Fail vs Target)│  │
│   └────────────────────┘          └──────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 3. What did I actually experiment with?
- Tested a naive prompt that passed a single test case but degraded on ambiguous, multi-intent, and sarcastic real-world inputs.
- Built a multi-turn evaluation harness that runs candidate prompts against a diverse test suite.
- Calculated objective pass-rate metrics to quantify prompt quality before deploying to production.

### 4. What can go wrong?
- **Prompt Regression (The Whack-a-Mole Trap)**: Modifying a prompt to fix an issue reported by a single user can silently break 10 other common scenarios.
- **Evaluation Set Contamination**: Testing prompts on the exact same inputs used during prompt tuning leads to overfitting.
- **Flaky Graders**: Graders that look for exact substrings may fail due to minor punctuation or whitespace variations.
- **Cost of Continuous Evals**: Running hundreds of test cases through large frontier models (`claude-3-5-sonnet`) can become expensive; architects balance test suite size and leverage smaller models or prompt caching.

### 5. What would an Architect need to know?
- **Evals are Unit Tests for LLMs**: In traditional software, code is accompanied by unit and integration tests. In AI systems, prompts must be accompanied by an evaluation suite in CI/CD.
- **Code-Based vs Model-Based Grading**:
  - Use **Code-Based Grading** (regex, JSON parsing, exact match, enums) whenever outputs have structured or deterministic formats. It is fast, deterministic, and free.
  - Use **Model-Based Grading** (LLM-as-a-judge) for open-ended summaries, tone, and subjective reasoning, but calibrate the judge prompt carefully.
- **Production Drift Monitoring**: Real user queries change over time. An architect must set up telemetry to sample production logs into new eval cases to keep the golden test suite updated.

---

## 🚀 How to Run

```zsh
cd "/Users/rajesh.yemul/Claude Certification/002_prompt_evaluation"
source ../001_claude_api/.venv/bin/activate
python 01_why_evals_matter.py
python 02_eval_workflow.py
python 03_code_based_grading.py
```
