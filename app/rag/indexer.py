"""
RAG Indexer — Load knowledge base files, generate embeddings, build FAISS index.

Loads all data files from the data/ directory at startup:
- JSON files: indexed per-key (each zodiac sign, each planet, etc.)
- TXT files: indexed per-line (each guidance entry)

Uses sentence-transformers for embedding generation and FAISS for vector search.
"""

import json
import os
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """A single indexed chunk of knowledge."""
    content: str
    source_file: str
    chunk_key: str  # e.g., "Aries", "Sun", or line number
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class KnowledgeIndexer:
    """
    Builds and manages the FAISS vector index over the knowledge corpus.

    Supports JSON files (per-key indexing) and TXT files (per-line indexing).
    """

    def __init__(self, data_dir: str = "data", model_name: str = "all-MiniLM-L6-v2"):
        """
        Args:
            data_dir: Path to the data directory containing knowledge files.
            model_name: Sentence-transformer model name for embeddings.
        """
        self.data_dir = data_dir
        self.model_name = model_name
        self.chunks: List[DocumentChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index = None  # FAISS index
        self._model = None

    def _get_model(self):
        """Lazy-load the sentence transformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _load_json_file(self, filepath: str, filename: str) -> List[DocumentChunk]:
        """Load a JSON file and create chunks per key."""
        chunks = []
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            source_name = os.path.splitext(filename)[0]

            for key, value in data.items():
                if isinstance(value, dict):
                    # Structured entry (e.g., zodiac traits, planetary impacts)
                    content_parts = [f"{key}:"]
                    for sub_key, sub_val in value.items():
                        content_parts.append(f"  {sub_key}: {sub_val}")
                    content = "\n".join(content_parts)
                elif isinstance(value, str):
                    content = f"{key}: {value}"
                else:
                    content = f"{key}: {json.dumps(value)}"

                chunks.append(DocumentChunk(
                    content=content,
                    source_file=source_name,
                    chunk_key=key,
                    metadata={"type": "json", "key": key},
                ))

            logger.info(f"Loaded {len(chunks)} chunks from {filename}")
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")

        return chunks

    def _load_txt_file(self, filepath: str, filename: str) -> List[DocumentChunk]:
        """Load a TXT file and create chunks per line/entry."""
        chunks = []
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()

            source_name = os.path.splitext(filename)[0]

            for i, line in enumerate(lines):
                line = line.strip().lstrip("- ").strip()
                if not line:
                    continue

                chunks.append(DocumentChunk(
                    content=line,
                    source_file=source_name,
                    chunk_key=f"line_{i+1}",
                    metadata={"type": "txt", "line": i + 1},
                ))

            logger.info(f"Loaded {len(chunks)} chunks from {filename}")
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")

        return chunks

    def load_all_documents(self) -> int:
        """
        Load all documents from the data directory.

        Returns:
            Number of chunks loaded.
        """
        self.chunks = []

        if not os.path.exists(self.data_dir):
            logger.warning(f"Data directory '{self.data_dir}' not found.")
            return 0

        for filename in sorted(os.listdir(self.data_dir)):
            filepath = os.path.join(self.data_dir, filename)
            if not os.path.isfile(filepath):
                continue

            if filename.endswith(".json"):
                self.chunks.extend(self._load_json_file(filepath, filename))
            elif filename.endswith(".txt"):
                self.chunks.extend(self._load_txt_file(filepath, filename))
            else:
                logger.debug(f"Skipping unsupported file: {filename}")

        logger.info(f"Total chunks loaded: {len(self.chunks)}")
        return len(self.chunks)

    def build_index(self) -> None:
        """
        Generate embeddings for all chunks and build the FAISS index.

        Must call load_all_documents() first.
        """
        import faiss

        if not self.chunks:
            logger.warning("No chunks to index. Call load_all_documents() first.")
            return

        model = self._get_model()
        texts = [chunk.content for chunk in self.chunks]

        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        self.embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        # Normalize for cosine similarity
        faiss.normalize_L2(self.embeddings)

        # Build FAISS index (Inner Product = cosine similarity after normalization)
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.embeddings)

        logger.info(f"FAISS index built with {self.index.ntotal} vectors (dim={dimension}).")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """
        Search the FAISS index for chunks similar to the query.

        Args:
            query: The search query string.
            top_k: Number of top results to return.

        Returns:
            List of (DocumentChunk, similarity_score) tuples, sorted by relevance.
        """
        import faiss

        if self.index is None:
            logger.warning("Index not built. Call build_index() first.")
            return []

        model = self._get_model()
        query_embedding = model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            results.append((self.chunks[idx], float(score)))

        return results
