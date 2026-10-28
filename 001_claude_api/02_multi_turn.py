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

def add_user_message(messages, content):
    """Helper function to add a user message to the messages list."""
    messages.append({"role": "user", "content": content})

def add_assistant_message(messages, content):
    """Helper function to add an assistant message to the messages list."""
    messages.append({"role": "assistant", "content": content})

# 4. Create the message request
print(f"Sending request using model: {model}...")
def chat(messages):
    """Function to send messages to the model and receive a response."""
    message = client.messages.create(
        model=model,
        max_tokens=1000,
        messages=messages
    )
    for block in message.content:
        if isinstance(block, TextBlock):
            return block.text
    return ""

# 5. Example usage
messages = []

""" Add a user message to the conversation."""
add_user_message(messages, "What is quantum computing? Answer in one sentence.")
chat_response = chat(messages)
print(f"\nResponse from Claude: {chat_response}")

""" Add the assistant's response to the conversation history."""
add_assistant_message(messages, chat_response)
print("\nConversation history:")
for msg in messages:
    print(f"{msg['role'].capitalize()}: {msg['content']}") 

""" Add another user message to the conversation."""
add_user_message(messages, "Can you explain it in more detail?")
chat_response = chat(messages)
print(f"\nResponse from Claude: {chat_response}")

