import requests

def ollama_chat(model: str, prompt: str):
    url = "http://localhost:11434/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}  # Non-streaming

    response = requests.post(url, json=payload)

    if response.status_code == 200:
        return response.json()["response"]  # Return full response
    else:
        return f"Error: {response.status_code}, {response.text}"

print(ollama_chat("llama2", "Tell me about black holes."))
