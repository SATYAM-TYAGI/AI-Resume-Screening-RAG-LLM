from __future__ import annotations

import json
import os
import sqlite3
from array import array
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class VectorStoreConfig:
    persist_dir: str
    collection_name: str


class SQLiteVectorStore:
    def __init__(self, cfg: VectorStoreConfig):
        self._cfg = cfg
        os.makedirs(cfg.persist_dir, exist_ok=True)
        self._db_path = os.path.join(cfg.persist_dir, f"{cfg.collection_name}.sqlite3")
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
              id TEXT PRIMARY KEY,
              doc_name TEXT NOT NULL,
              text TEXT NOT NULL,
              meta_json TEXT NOT NULL,
              embedding BLOB NOT NULL
            );
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_name ON chunks(doc_name);")
        self._conn.commit()

    def reset(self) -> None:
        self._conn.execute("DELETE FROM chunks;")
        self._conn.commit()

    def upsert(
        self,
        *,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict],
    ) -> None:
        rows = []
        for _id, emb, doc, meta in zip(ids, embeddings, documents, metadatas):
            doc_name = meta.get("doc_name") or meta.get("source") or "unknown"
            meta_json = json.dumps(meta, ensure_ascii=False)
            blob = array("f", [float(x) for x in emb]).tobytes()
            rows.append((_id, doc_name, doc, meta_json, blob))

        self._conn.executemany(
            """
            INSERT OR REPLACE INTO chunks (id, doc_name, text, meta_json, embedding)
            VALUES (?, ?, ?, ?, ?);
            """,
            rows,
        )
        self._conn.commit()

    def list_doc_names(self) -> List[str]:
        cur = self._conn.execute("SELECT DISTINCT doc_name FROM chunks ORDER BY doc_name;")
        return [r[0] for r in cur.fetchall()]

    def query(
        self,
        *,
        query_embedding: List[float],
        top_k: int,
        where: Optional[Dict] = None,
    ) -> List[Tuple[str, str, Dict, float]]:
        """
        Returns list of (id, document_text, metadata, score) where score is cosine similarity in [0,1].
        """
        q = array("f", [float(x) for x in query_embedding])

        sql = "SELECT id, text, meta_json, embedding FROM chunks"
        params: List = []
        if where and "doc_name" in where:
            sql += " WHERE doc_name = ?"
            params.append(where["doc_name"])
        cur = self._conn.execute(sql + ";", params)

        scored: List[Tuple[str, str, Dict, float]] = []
        for _id, text, meta_json, blob in cur.fetchall():
            emb = array("f")
            emb.frombytes(blob)
            # embeddings are normalized => dot product approximates cosine similarity
            score = 0.0
            # safe min length
            n = min(len(q), len(emb))
            for i in range(n):
                score += q[i] * emb[i]
            meta = json.loads(meta_json) if meta_json else {}
            scored.append((_id, text, meta, max(0.0, min(1.0, float(score)))))

        scored.sort(key=lambda x: x[3], reverse=True)
        return scored[:top_k]

