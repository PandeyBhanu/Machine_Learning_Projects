# app/services/ollama_client.py
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "deepseek-coder"


def run_llm(prompt: str) -> str:
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False}
    res = requests.post(OLLAMA_URL, json=payload)
    res.raise_for_status()
    return res.json()["response"]
