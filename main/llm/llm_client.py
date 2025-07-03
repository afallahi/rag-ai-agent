import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral"


def generate_answer(prompt: str, model: str = MODEL_NAME) -> str:
    """
    Calls the local Ollama model with a given prompt and returns the response.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        return response.json()["response"]
    except requests.RequestException as e:
        print(f"[ERROR] Failed to call Ollama: {e}")
        return "LLM error: could not generate response"