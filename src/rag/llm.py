from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass(frozen=True)
class LLMConfig:
    api_key: Optional[str]
    model: str = "gpt-4o-mini"


class BaseLLM:
    def complete_json(self, *, system: str, user: str) -> Dict[str, Any]:
        raise NotImplementedError

    def complete_markdown(self, *, system: str, user: str) -> str:
        raise NotImplementedError


class OpenAILLM(BaseLLM):
    def __init__(self, cfg: LLMConfig):
        if not cfg.api_key:
            raise ValueError("OPENAI_API_KEY missing")
        self._client = OpenAI(api_key=cfg.api_key)
        self._model = cfg.model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.7, min=0.7, max=6))
    def complete_json(self, *, system: str, user: str) -> Dict[str, Any]:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        content = resp.choices[0].message.content or "{}"
        return json.loads(content)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.7, min=0.7, max=6))
    def complete_markdown(self, *, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
        )
        return (resp.choices[0].message.content or "").strip()


class OllamaLLM(BaseLLM):
    """
    Simple wrapper around a local Ollama server.
    Uses the /api/chat endpoint and returns the final message content.
    """

    def __init__(self, model: str = "llama3.1", base_url: str = "http://127.0.0.1:11434"):
        self._model = model
        self._base_url = base_url.rstrip("/")

    def _chat(self, *, system: str, user: str) -> str:
        url = f"{self._base_url}/api/chat"
        payload = {
            "model": self._model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        # Ollama's /api/chat returns {"message": {"role": "...", "content": "..."}, ...}
        return (data.get("message", {}) or {}).get("content", "") or ""

    def complete_markdown(self, *, system: str, user: str) -> str:
        return self._chat(system=system, user=user).strip()

    def complete_json(self, *, system: str, user: str) -> Dict[str, Any]:
        text = self._chat(system=system, user=user).strip()
        try:
            return json.loads(text)
        except Exception:
            # If the model doesn't strictly obey JSON, wrap as best-effort.
            return {"raw": text}

