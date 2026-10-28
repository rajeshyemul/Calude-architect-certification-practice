import os
import time
from dotenv import load_dotenv
from anthropic import Anthropic

# 1. Environment & Client Setup
load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")
base_url = os.getenv("ANTHROPIC_BASE_URL")
client = Anthropic(api_key=api_key, base_url=base_url) if base_url else Anthropic(api_key=api_key)
model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")

print(f"=== 05: Streaming & Latency Profiling (Model: {model}) ===\n")

prompt = "Explain how event-driven architecture handles backpressure in 3 brief paragraphs."

print("--- Streaming Tokens in Real-Time ---")
start_time = time.time()
ttft = None
first_token_received = False
total_tokens_printed = 0

with client.messages.stream(
    model=model,
    max_tokens=600,
    messages=[{"role": "user", "content": prompt}]
) as stream:
    for text in stream.text_stream:
        if not first_token_received:
            ttft = time.time() - start_time
            first_token_received = True
        print(text, end="", flush=True)
        total_tokens_printed += 1

total_duration = time.time() - start_time
final_message = stream.get_final_message()

print("\n\n" + "-"*50)
print(f"Time-To-First-Token (TTFT): {ttft:.2f} seconds" if ttft else "TTFT: N/A")
print(f"Total Response Duration:    {total_duration:.2f} seconds")
print(f"Usage Input Tokens:         {final_message.usage.input_tokens}")
print(f"Usage Output Tokens:        {final_message.usage.output_tokens}")
print(f"Stop Reason:                {final_message.stop_reason}")
print("-"*50)

# ==============================================================================
# Architectural Takeaway
# ==============================================================================
print("\n" + "="*70)
print("ARCHITECT NOTE:")
print("1. Perceived Latency (TTFT) vs Computational Latency:")
print("   Streaming does NOT reduce model compute time, but it cuts human perceived wait")
print("   time from multiple seconds down to a fraction of a second.")
print("2. stream.get_final_message():")
print("   Essential for capturing token usage (`usage.input_tokens`, `usage.output_tokens`)")
print("   and stop reasons (`end_turn`, `max_tokens`) after streaming completes.")
print("3. When NOT to Stream:")
print("   - Automated pipelines, batch jobs, and internal microservice-to-microservice APIs")
print("     where payload validation and downstream consumption must be atomic.")
print("   - Tool execution loops where you need the complete tool input JSON before taking action.")
print("="*70)
