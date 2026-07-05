"""RAG vector store backed by ChromaDB and OpenAI embeddings.

Two collections are maintained:

* ``categories`` — one entry per category path, used to suggest consistent
  existing categories when classifying a new file.
* ``files`` — the content of every organized file, enabling Q&A search over
  everything the organizer has processed.
"""

from __future__ import annotations

from typing import Optional

import chromadb
from openai import OpenAI

from .oai_retry import with_retries


class RagStore:
    def __init__(self, persist_dir: str, client: OpenAI, embed_model: str) -> None:
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._oa = client
        self._embed_model = embed_model
        # Ensure the collections exist up front.
        self._collection("files")
        self._collection("categories")

    def _collection(self, name: str):
        """Get (or lazily recreate) a collection by name.

        Fetching by name on each access keeps things robust if a collection is
        deleted out from under us (e.g. by clear() or another process).
        """
        return self._client.get_or_create_collection(
            name, metadata={"hnsw:space": "cosine"}
        )

    @property
    def files(self):
        return self._collection("files")

    @property
    def categories(self):
        return self._collection("categories")

    # --- embeddings --------------------------------------------------------
    def _embed(self, texts: list[str]) -> list[list[float]]:
        safe = [t if t and t.strip() else " " for t in texts]
        response = with_retries(
            lambda: self._oa.embeddings.create(model=self._embed_model, input=safe)
        )
        return [item.embedding for item in response.data]

    # --- categories --------------------------------------------------------
    def suggest_categories(self, text: str, n: int = 6) -> list[str]:
        """Return existing category paths most similar to ``text``."""
        count = self.categories.count()
        if count == 0:
            return []
        embedding = self._embed([text])[0]
        result = self.categories.query(
            query_embeddings=[embedding], n_results=min(n, count)
        )
        docs = result.get("documents") or [[]]
        return docs[0]

    def remember_category(self, path: str, description: str) -> None:
        embedding = self._embed([f"{path}. {description}"])[0]
        self.categories.upsert(
            ids=[path],
            embeddings=[embedding],
            documents=[path],
            metadatas=[{"description": description[:500]}],
        )

    # --- files -------------------------------------------------------------
    def add_file(self, file_id: str, content: str, metadata: dict) -> None:
        text = content if content and content.strip() else metadata.get("filename", "")
        embedding = self._embed([text[:6000]])[0]
        self.files.upsert(
            ids=[file_id],
            embeddings=[embedding],
            documents=[text[:6000]],
            metadatas=[metadata],
        )

    def search(self, query: str, n: int = 5) -> list[dict]:
        """Return the file entries most relevant to ``query``."""
        count = self.files.count()
        if count == 0:
            return []
        embedding = self._embed([query])[0]
        result = self.files.query(
            query_embeddings=[embedding], n_results=min(n, count)
        )
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        return [{"content": d, "metadata": m} for d, m in zip(docs, metas)]

    def file_count(self) -> int:
        return self.files.count()

    def clear(self) -> None:
        """Delete everything indexed (files and remembered categories).

        The collections are recreated lazily on next access via the properties.
        """
        for name in ("files", "categories"):
            try:
                self._client.delete_collection(name)
            except Exception:  # noqa: BLE001 - collection may not exist
                pass


def answer_question(
    client: OpenAI, model: str, question: str, sources: list[dict]
) -> str:
    """Answer ``question`` grounded in the retrieved file ``sources``."""
    if not sources:
        return "No organized files are indexed yet, so I have nothing to search."

    context_blocks = []
    for i, src in enumerate(sources, start=1):
        meta = src.get("metadata", {})
        context_blocks.append(
            f"[{i}] file: {meta.get('filename', 'unknown')} "
            f"(category: {meta.get('category', 'n/a')})\n"
            f"{(src.get('content') or '')[:1500]}"
        )
    context = "\n\n".join(context_blocks)

    kwargs: dict = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You answer questions about a user's organized files using "
                    "only the provided context. Cite files by name when useful. "
                    "If the answer is not in the context, say so."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
    }
    # Reasoning models (gpt-5, o1/o3/o4) only accept the default temperature.
    lowered = model.lower()
    if not (lowered.startswith("gpt-5") or lowered.startswith(("o1", "o3", "o4"))):
        kwargs["temperature"] = 0

    response = with_retries(lambda: client.chat.completions.create(**kwargs))
    return response.choices[0].message.content or ""
