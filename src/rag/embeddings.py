from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import List, Optional

from openai import OpenAI, RateLimitError


@dataclass(frozen=True)
class EmbeddingConfig:
    api_key: Optional[str] = None
    openai_embed_model: str = "text-embedding-3-small"
    local_dim: int = 384


class Embedder:
    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError


_TOKEN_RE = re.compile(r"[a-z0-9\+\#\.\-]{2,}", re.IGNORECASE)


class LocalHashingEmbedder(Embedder):
    """
    Pure-Python hashing embeddings (fast, dependency-light).
    Not SOTA, but robust on fresh Python versions and adequate for a prototype.
    Produces L2-normalized non-negative vectors, so dot(vec, vec) ~ cosine similarity in [0,1].
    """

    def __init__(self, dim: int = 384):
        self._dim = dim

    def _embed_one(self, text: str) -> List[float]:
        v = [0.0] * self._dim
        tokens = _TOKEN_RE.findall(text.lower())
        # Unigrams + light bigrams for better matching.
        for i, tok in enumerate(tokens):
            idx = self._hash_to_idx(tok)
            v[idx] += 1.0
            if i + 1 < len(tokens):
                bi = f"{tok}_{tokens[i+1]}"
                v[self._hash_to_idx(bi)] += 0.5

        # L2 normalize
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def _hash_to_idx(self, s: str) -> int:
        h = hashlib.blake2b(s.encode("utf-8", errors="ignore"), digest_size=8).digest()
        n = int.from_bytes(h, "little", signed=False)
        return n % self._dim

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_one(t) for t in texts]


class OpenAIEmbedder(Embedder):
    def __init__(self, api_key: str, model: str):
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._fallback = LocalHashingEmbedder()

    def embed(self, texts: List[str]) -> List[List[float]]:
        try:
            resp = self._client.embeddings.create(model=self._model, input=texts)
            return [d.embedding for d in resp.data]
        except RateLimitError:
            return self._fallback.embed(texts)
        except Exception:
            return self._fallback.embed(texts)


def make_embedder(cfg: EmbeddingConfig) -> Embedder:
    if cfg.api_key:
        return OpenAIEmbedder(api_key=cfg.api_key, model=cfg.openai_embed_model)
    return LocalHashingEmbedder(dim=cfg.local_dim)

