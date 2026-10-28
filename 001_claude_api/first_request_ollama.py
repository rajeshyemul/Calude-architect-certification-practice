"""
Direct Ollama example (No Anthropic API Key required).
Uses standard library urllib so no additional dependencies are needed.
"""
import json
import urllib.request
import urllib.error

# Pick any model you have downloaded locally in Ollama (e.g., llama3.2, llama3, mistral, qwen2.5)
MODEL = "llama3.2"

def main():
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": "What is quantum computing? Answer in one sentence."
            }
        ],
        "stream": False
    }

    print(f"Connecting to local Ollama on port 11434 (model: {MODEL})...")
    
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            print("\nOllama's Response:")
            print(result["message"]["content"])
    except urllib.error.URLError as e:
        print(f"\n[Error] Unable to reach Ollama: {e}")
        print("Please ensure Ollama is running:")
        print("  1. Launch the Ollama app: open -a Ollama")
        print(f"  2. Ensure the model is downloaded: /usr/local/bin/ollama pull {MODEL}")

if __name__ == "__main__":
    main()

