"""RAG (Retrieval-Augmented Generation) engine (Module 6 knowledge base).

Indexes the NCERT/JEE knowledge base — the formula sheet
(``data/knowledge_base/formula_sheet.json``) plus any ingested NCERT
textbook / JEE-formula PDFs — into a persistent ChromaDB collection
named ``math_knowledge``, and exposes similarity search over it for
:mod:`src.mathbot`'s ``explain_concept`` and ``formula_lookup`` tools.

Embeddings default to a small, free, local ``sentence-transformers``
model (no API key required) so RAG works fully offline; this can be
swapped for a Gemini/OpenAI embedding model later without changing
the public interface.

This module is intentionally decoupled from any LLM-framework abstractions
(it drives the ``chromadb`` client directly) so it stays lightweight
and testable; :mod:`src.mathbot` uses it to ground Gemini's answers.

Example:
    >>> from src.rag_engine import RAGEngine
    >>> engine = RAGEngine()
    >>> engine.ingest_formula_sheet()
    >>> results = engine.query("quadratic formula", top_k=2)
    >>> results[0].metadata["name"]
    'Quadratic Formula'
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from src.config import settings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "math_knowledge"


class RAGEngineError(Exception):
    """Raised when the vector store or embedding backend is unavailable."""


@dataclass
class RetrievedDocument:
    """A single retrieval hit.

    Attributes:
        text: The document chunk's raw text.
        metadata: Arbitrary metadata (topic, source, class_level, etc.).
        distance: Vector distance (lower = more similar); ``None`` if
            the backend doesn't expose it.
    """

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    distance: Optional[float] = None


class RAGEngine:
    """Thin wrapper around a persistent ChromaDB collection for math knowledge."""

    def __init__(
        self,
        persist_dir: Optional[Path] = None,
        embedding_model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        """Create a RAG engine.

        Args:
            persist_dir: On-disk directory for the ChromaDB store.
                Defaults to ``settings.chromadb_dir``.
            embedding_model_name: A ``sentence-transformers`` model name.
                The default is small (~80MB), fast on CPU, and free.
        """
        self.persist_dir = persist_dir or settings.chromadb_dir
        self.embedding_model_name = embedding_model_name
        self._client = None
        self._collection = None
        self._embedder = None

    # ------------------------------------------------------------------
    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            import chromadb
        except ImportError as exc:
            raise RAGEngineError(
                "RAG engine requires 'chromadb'. Install with: pip install chromadb"
            ) from exc

        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._collection = self._client.get_or_create_collection(COLLECTION_NAME)

    def _ensure_embedder(self) -> None:
        if self._embedder is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RAGEngineError(
                "RAG engine requires 'sentence-transformers'. "
                "Install with: pip install sentence-transformers"
            ) from exc
        logger.info("Loading embedding model: %s", self.embedding_model_name)
        self._embedder = SentenceTransformer(self.embedding_model_name)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        self._ensure_embedder()
        return self._embedder.encode(texts, show_progress_bar=False).tolist()

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def ingest_formula_sheet(self, path: Optional[Path] = None) -> int:
        """Load ``formula_sheet.json`` and index each formula as a document.

        Args:
            path: Override for the formula sheet's location.

        Returns:
            The number of formula documents indexed.
        """
        self._ensure_client()
        path = path or (settings.knowledge_base_dir / "formula_sheet.json")
        if not path.exists():
            raise FileNotFoundError(f"Formula sheet not found: {path}")

        with open(path, encoding="utf-8") as fh:
            formula_sheet = json.load(fh)

        ids, documents, metadatas = [], [], []
        for category, formulas in formula_sheet.items():
            for i, formula in enumerate(formulas):
                doc_text = (
                    f"{formula['name']}: {formula['latex']}. {formula['description']}"
                )
                ids.append(f"formula::{category}::{i}")
                documents.append(doc_text)
                metadatas.append(
                    {
                        "type": "formula",
                        "category": category,
                        "name": formula["name"],
                        "latex": formula["latex"],
                        "class_level": formula.get("class_level", ""),
                    }
                )

        if not documents:
            logger.warning("No formulas found in %s", path)
            return 0

        embeddings = self._embed(documents)
        self._collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
        logger.info("Ingested %d formulas into '%s'", len(documents), COLLECTION_NAME)
        return len(documents)

    def ingest_text_chunks(
        self,
        chunks: list[str],
        source: str,
        chunk_metadata: Optional[list[dict[str, Any]]] = None,
    ) -> int:
        """Index arbitrary pre-chunked text (e.g. from an NCERT PDF).

        Args:
            chunks: List of text chunks (already split to a reasonable
                size, e.g. by :mod:`scripts.ingest_knowledge`).
            source: A label identifying where these chunks came from
                (e.g. ``"ncert_class10.pdf"``), stored in metadata.
            chunk_metadata: Optional per-chunk metadata overrides.

        Returns:
            The number of chunks indexed.
        """
        self._ensure_client()
        if not chunks:
            return 0

        ids = [f"{source}::chunk::{i}" for i in range(len(chunks))]
        metadatas = chunk_metadata or [{"type": "textbook", "source": source} for _ in chunks]
        embeddings = self._embed(chunks)
        self._collection.upsert(ids=ids, documents=chunks, metadatas=metadatas, embeddings=embeddings)
        logger.info("Ingested %d chunks from %s", len(chunks), source)
        return len(chunks)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def query(
        self, question: str, top_k: int = 3, where: Optional[dict[str, Any]] = None
    ) -> list[RetrievedDocument]:
        """Retrieve the most relevant documents for a natural-language question.

        Args:
            question: e.g. "quadratic formula" or "why do we use the discriminant".
            top_k: Number of results to return.
            where: Optional ChromaDB metadata filter, e.g. ``{"category": "algebra"}``.

        Returns:
            List of :class:`RetrievedDocument`, best match first.
        """
        self._ensure_client()
        query_embedding = self._embed([question])[0]

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0] if "distances" in results else [None] * len(documents)

        return [
            RetrievedDocument(text=doc, metadata=meta or {}, distance=dist)
            for doc, meta, dist in zip(documents, metadatas, distances)
        ]

    def formula_lookup(self, name_or_topic: str, top_k: int = 1) -> Optional[RetrievedDocument]:
        """Convenience wrapper for MathBot's ``formula_lookup`` tool.

        Args:
            name_or_topic: Formula name or topic keyword, e.g. "quadratic formula".
            top_k: How many candidates to consider before returning the best.

        Returns:
            The single best-matching :class:`RetrievedDocument`, or ``None``
            if the collection is empty / unavailable.
        """
        try:
            results = self.query(name_or_topic, top_k=top_k, where={"type": "formula"})
            return results[0] if results else None
        except RAGEngineError:
            logger.warning("RAG engine unavailable for formula lookup", exc_info=True)
            return None

    def count(self) -> int:
        """Return the number of documents currently indexed."""
        self._ensure_client()
        return self._collection.count()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engine = RAGEngine()
    try:
        n = engine.ingest_formula_sheet()
        print(f"Ingested {n} formulas. Total documents: {engine.count()}")

        for q in ["quadratic formula", "how to differentiate x^n", "probability of an event"]:
            hits = engine.query(q, top_k=1)
            if hits:
                print(f"Q: {q!r} -> {hits[0].metadata.get('name', hits[0].text[:60])}")
    except RAGEngineError as e:
        print(f"RAG engine unavailable in this environment: {e}")
