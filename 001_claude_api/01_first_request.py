import os
from dotenv import load_dotenv
from anthropic import Anthropic
from anthropic.types import TextBlock

# 1. Load environment variables from .env file
load_dotenv()

# Verify API key is present
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key or api_key == "your-api-key-here":
    print("Warning: Please set a valid ANTHROPIC_API_KEY in your .env file.")

# 2. Initialize Anthropic client
# (Supports local proxies like LiteLLM if ANTHROPIC_BASE_URL is set in .env)
base_url = os.getenv("ANTHROPIC_BASE_URL")
client = Anthropic(api_key=api_key, base_url=base_url) if base_url else Anthropic(api_key=api_key)

# 3. Specify the model
# Common production models include "claude-3-5-sonnet-latest", "claude-3-7-sonnet-latest", "claude-3-haiku-20240307"
model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")

# 4. Create the message request
print(f"Sending request using model: {model}...")
message = client.messages.create(
    model=model,
    max_tokens=1000,
    messages=[
        {
            "role": "user",
            "content": "What is quantum computing? Answer in one sentence."
        }
    ]
)

# 5. Extract and print the response
print("\nClaude's Response:")
for block in message.content:
    if isinstance(block, TextBlock):
        print(block.text)

