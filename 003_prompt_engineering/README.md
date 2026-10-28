# 003 - Prompt Engineering & Systematic Prompt Evaluation

This directory covers advanced prompt engineering techniques (XML tagging, few-shot conditioning, strict negative constraints, and persona steering) coupled with an automated evaluation engine that synthesizes test datasets, executes parallel runs, and generates styled HTML reports.

---

## 📂 Capability Files

| File | Meaningful Capability | Architectural Concept |
| :--- | :--- | :--- |
| **[`001_prompting.py`](file:///Users/rajesh.yemul/Claude%20Certification/003_prompt_engineering/001_prompting.py)** | Comprehensive Prompt Engineering & Eval Engine | XML structuring, few-shot examples, dynamic synthetic dataset generation, concurrent evaluation, and HTML reporting (`output.html`). |

---

## 🧠 The 5 Architect Questions

### 1. What did I learn?
Prompt engineering is not about crafting clever single-shot sentences; it is about **context architecture**:
- **XML Tag Isolation**: Wrapping inputs in XML tags (`<athlete_information>`, `<sample_input>`, `<ideal_output>`) prevents prompt injection and disambiguates user data from system instructions.
- **Few-Shot Conditioning**: Giving exact input/output pairs anchors model output formatting, tone, and level of detail more reliably than paragraphs of rules.
- **Automated Synthetic Generation**: Using Claude/Ollama to generate diverse edge-case scenarios eliminates human bias in test dataset creation.

### 2. What is the implementation architecture?
```text
┌─────────────────────────────────────────────────────────────┐
│                    PromptEvaluator Engine                   │
│                                                             │
│   Task Description + Inputs Spec                            │
│                 │                                           │
│                 ▼                                           │
│   1. Generate Unique Ideas (JSON array)                     │
│                 │                                           │
│                 ▼                                           │
│   2. Synthesize Test Cases (prompt_inputs + criteria)       │
│                 │                                           │
│                 ▼                                           │
│   3. Save `dataset.json`                                    │
│                 │                                           │
│                 ▼                                           │
│   4. Concurrent Worker Pool (ThreadPoolExecutor)            │
│         ├── Test 1 ──> Candidate Prompt ──> Claude/Ollama   │
│         └── Test 2 ──> Candidate Prompt ──> Claude/Ollama   │
│                 │                                           │
│                 ▼                                           │
│   5. Grader (LLM-as-a-Judge against criteria)               │
│                 │                                           │
│                 ▼                                           │
│   6. Output Artifacts: `output.json` & `output.html`        │
└─────────────────────────────────────────────────────────────┘
```

### 3. What did I actually experiment with?
- Structuring prompt inputs inside `<athlete_information>` XML tags.
- Adding a few-shot demonstration with `<sample_input>` and `<ideal_output>`.
- Setting up concurrent execution workers via `concurrent.futures.ThreadPoolExecutor`.
- Generating rich HTML reports displaying color-coded pass rates, criteria breakdown, and model reasoning.

### 4. What can go wrong? (Edge Cases & Failure Modes)
- **Local Model Format Drift**: Smaller local models (like `llama3.2` on Ollama) may occasionally include markdown code fences or conversational preambles when returning JSON. Safe parsing with fallback regex is necessary.
- **Rate Limiting on Local & Cloud Endpoints**: High concurrency with local models can exhaust GPU VRAM or queue threads; with cloud APIs it triggers HTTP 429 `rate_limit_error`. Keep `max_concurrent_tasks` conservative (1 to 3).
- **Overfitting to Few-Shot Examples**: If the few-shot example is overly specific, the model may parrot the example's exact numbers or values rather than calculating new ones for the target inputs.

### 5. What would an Architect need to know?
- **Why XML Tags are Standard for Claude**: Anthropic models are pre-trained and fine-tuned specifically to recognize XML structures. Parsing XML tags gives Claude clear structural semantic boundaries that resist adversarial prompt injection.
- **Local Model (Ollama) vs Frontier Model (Claude)**:
  - Local models (via Ollama/LiteLLM) are ideal for private, offline testing and zero token cost.
  - Frontier models (Claude 3.5 Sonnet) should be used for nuanced grading, complex instruction following, and production workloads.
- **HTML Artifacts for Stakeholder Buy-in**: Evaluating prompts through numbers alone can be abstract; generating visually styled HTML reports with scenario tables and color-coded scores makes model performance tangible to non-technical product owners.

---

## 🚀 How to Run

### Step 1: Ensure Ollama & LiteLLM are Running (if using local model)
In a separate terminal window:
```zsh
# Start Ollama
open -a Ollama

# Start LiteLLM proxy pointing to your local Ollama model
litellm --model ollama/llama3.2 --port 4000
```

*(Alternatively, to run against official Anthropic Claude, update `.env` with `ANTHROPIC_API_KEY="sk-ant-..."` and remove `ANTHROPIC_BASE_URL`)*.

### Step 2: Run the Evaluator
```zsh
cd "/Users/rajesh.yemul/Claude Certification/003_prompt_engineering"
source ../001_claude_api/.venv/bin/activate

python 001_prompting.py
```

After running, inspect:
- `dataset.json` — Generated test cases.
- `output.json` — Raw evaluation scores and reasonings.
- `output.html` — Interactive visual evaluation report (open in browser!).
