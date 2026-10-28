# Claude Certified Architect — Hands-On Portfolio

This repository tracks hands-on implementations, architectural trade-offs, failure modes, and mental models required for the **Claude Certified Architect** exam and real-world system design.

---

## 🗺️ 10-Module Architect Progression

| Module | Focus Area | Key Architectural Competencies |
| :--- | :--- | :--- |
| **[`001_claude_api/`](file:///Users/rajesh.yemul/Claude%20Certification/001_claude_api)** | **Core API & State** | Client state ownership, system prompts, temperature trade-offs, streaming TTFT, structured outputs |
| **[`002_prompt_evaluation/`](file:///Users/rajesh.yemul/Claude%20Certification/002_prompt_evaluation)** | **Evaluation & Guardrails** | Deterministic vs LLM-as-a-judge, golden datasets, grading pipelines, prompt drift detection |
| **[`003_prompt_engineering/`](file:///Users/rajesh.yemul/Claude%20Certification/003_prompt_engineering)** | **Context Shaping** | Few-shot conditioning, XML tags (`<context>`, `<rules>`), prefilling assistant responses, chain-of-thought |
| **[`004_tools_with_claude/`](file:///Users/rajesh.yemul/Claude%20Certification/004_tools_with_claude)** | **Tool Calling** | Tool schemas, JSON validation, single/multi-tool execution loops, handling tool call failures & retries |
| **`005_rag/`** | **Retrieval-Augmented Generation** | Semantic embeddings vs BM25 hybrid search, chunking strategies, reranking, context stuffing trade-offs |
| **`006_claude_features/`** | **Advanced Claude Features** | Extended thinking (budget tokens), prompt caching (`cache_control`), vision API, PDF document processing |
| **`007_mcp/`** | **Model Context Protocol** | Architecture of MCP hosts, clients, servers; Tools, Resources, and Prompts; security isolation boundaries |
| **`008_claude_code/`** | **CLI & Computer Use** | Headless agent execution, terminal tools, environment isolation, computer use API |
| **`009_agents_workflows/`** | **Agentic Architectures** | Prompt chaining, routing, parallelization, orchestrator-workers, evaluator-optimizer loops |
| **`010_final_capstone/`** | **Production Agent System** | End-to-end agent integrating MCP tools, hybrid RAG, prompt caching, evaluation harness, and observability |

---

## 🎯 The Rule of Every Capability File

To develop architect-level instincts, every capability implementation follows the **Discipline Loop**:

```text
Concept Implementation
         ↓
    Experiment
         ↓
Failure / Edge Case
         ↓
Improved Implementation
         ↓
Architectural Notes (5 Questions)
```

### The 5 Questions Every Architect Must Answer
1. **What did I learn?** (Core capability)
2. **What is the implementation?** (Data flow and component responsibilities)
3. **What did I experiment with?** (Parameters and behavior changes)
4. **What can go wrong?** (Edge cases, errors, rate limits, silent degradations)
5. **What would an Architect need to know?** (Cost, latency, state, security, system trade-offs)
