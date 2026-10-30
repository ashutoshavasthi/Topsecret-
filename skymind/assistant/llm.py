"""Minimal client for a local Ollama server (https://ollama.com), stdlib only.

Install Ollama, then pull the models once (they are then fully offline and free):

    ollama pull qwen2.5:3b      # planner: turns requests into drone actions (~2 GB)
    ollama pull moondream       # vision: describes camera images (~1.7 GB)
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from typing import Any


class LLMError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, url: str = "http://127.0.0.1:11434", timeout: float = 60.0):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            self.url + path, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ConnectionError) as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc

    def models(self) -> list[str]:
        try:
            with urllib.request.urlopen(self.url + "/api/tags", timeout=3.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ConnectionError, OSError):
            return []
        return [m.get("name", "") for m in data.get("models", [])]

    def has_model(self, name: str | None) -> bool:
        if not name:
            return False
        names = self.models()
        if ":" in name:
            return name in names
        return any(n.split(":")[0] == name for n in names)  # untagged name: any tag of that model

    def chat(self, model: str, messages: list[dict], schema: dict | None = None, temperature: float = 0.2) -> str:
        payload: dict[str, Any] = {"model": model, "messages": messages, "stream": False, "options": {"temperature": temperature}}
        if schema is not None:
            payload["format"] = schema  # structured output: the reply is guaranteed to parse as this JSON
        data = self._post("/api/chat", payload)
        try:
            return data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise LLMError(f"unexpected Ollama response: {str(data)[:200]}") from exc

    def chat_json(self, model: str, messages: list[dict], schema: dict) -> dict:
        text = self.chat(model, messages, schema)
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMError(f"model did not return JSON: {text[:200]}") from exc

    def describe_image(self, model: str, jpeg: bytes, prompt: str) -> str:
        msg = {"role": "user", "content": prompt, "images": [base64.b64encode(jpeg).decode("ascii")]}
        return self.chat(model, [msg], temperature=0.1).strip()
