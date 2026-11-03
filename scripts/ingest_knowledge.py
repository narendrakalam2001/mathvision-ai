#!/usr/bin/env python3
"""Ingest the NCERT/JEE knowledge base into the ChromaDB RAG store.

Two sources are ingested:
    1. ``data/knowledge_base/formula_sheet.json`` — always available,
       ships with the repo.
    2. Any NCERT/JEE textbook PDFs dropped into
       ``data/knowledge_base/*.pdf`` (e.g. ``ncert_class10.pdf``,
       ``jee_formulas.pdf``) — extracted with PyMuPDF and chunked into
       ~500-character passages with overlap.

Usage:
    python scripts/ingest_knowledge.py                # formula sheet + any PDFs present
    python scripts/ingest_knowledge.py --formulas-only # skip PDF extraction
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python scripts/<name>.py` from any directory: put the project root on sys.path
# (Python only adds the scripts/ folder itself, so `import src` would fail otherwise).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import logging
from pathlib import Path

from src.config import settings
from src.rag_engine import RAGEngine, RAGEngineError

logger = logging.getLogger(__name__)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping fixed-size chunks for embedding.

    Args:
        text: Full extracted document text.
        chunk_size: Target characters per chunk.
        overlap: Characters of overlap between consecutive chunks, so
            a fact split across a chunk boundary isn't lost entirely.

    Returns:
        List of non-empty text chunks.
    """
    text = " ".join(text.split())  # normalize whitespace
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract all text from a PDF using PyMuPDF.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        The concatenated text of all pages.

    Raises:
        ImportError: If PyMuPDF (``fitz``) isn't installed.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise ImportError(
            "PDF ingestion requires PyMuPDF. Install with: pip install pymupdf"
        ) from exc

    text_parts = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text_parts.append(page.get_text())
    return "\n".join(text_parts)


def ingest_all(formulas_only: bool = False) -> dict[str, int]:
    """Run the full ingestion pipeline.

    Args:
        formulas_only: If True, skip PDF discovery/extraction entirely.

    Returns:
        A dict of {source_name: num_chunks_ingested} for reporting.
    """
    engine = RAGEngine()
    report: dict[str, int] = {}

    n_formulas = engine.ingest_formula_sheet()
    report["formula_sheet.json"] = n_formulas

    if formulas_only:
        return report

    pdf_paths = sorted(settings.knowledge_base_dir.glob("*.pdf"))
    if not pdf_paths:
        logger.info("No PDFs found in %s — only the formula sheet was ingested.", settings.knowledge_base_dir)
        return report

    for pdf_path in pdf_paths:
        try:
            raw_text = extract_pdf_text(pdf_path)
            chunks = chunk_text(raw_text)
            metadata = [
                {"type": "textbook", "source": pdf_path.name, "chunk_index": i}
                for i in range(len(chunks))
            ]
            n = engine.ingest_text_chunks(chunks, source=pdf_path.stem, chunk_metadata=metadata)
            report[pdf_path.name] = n
        except ImportError as exc:
            logger.error(str(exc))
            report[pdf_path.name] = 0
        except Exception:  # noqa: BLE001
            logger.error("Failed to ingest %s", pdf_path, exc_info=True)
            report[pdf_path.name] = 0

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the NCERT/JEE knowledge base into ChromaDB.")
    parser.add_argument("--formulas-only", action="store_true", help="Skip PDF extraction entirely")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    try:
        report = ingest_all(formulas_only=args.formulas_only)
    except RAGEngineError as exc:
        print(f"RAG engine unavailable: {exc}")
        print("Install with: pip install chromadb sentence-transformers")
        raise SystemExit(1) from exc

    print("Ingestion report:")
    for source, count in report.items():
        print(f"  {source}: {count} documents/chunks")


if __name__ == "__main__":
    main()
