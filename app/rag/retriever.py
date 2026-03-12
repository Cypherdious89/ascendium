"""
RAG Retriever — Semantic search over the FAISS index with filtering.

Wraps the indexer's search with:
- Configurable similarity threshold
- Source metadata extraction
- Formatted context for LLM prompts
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from app.rag.indexer import KnowledgeIndexer, DocumentChunk

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """A single retrieval result with content, source info, and score."""
    content: str
    source: str
    chunk_key: str
    score: float


class KnowledgeRetriever:
    """
    Retrieves relevant knowledge chunks for a given query.

    Features:
    - Similarity threshold filtering
    - Source tracking for the API response
    - Formatted context string for LLM prompts
    """

    def __init__(
        self,
        indexer: KnowledgeIndexer,
        similarity_threshold: float = 0.25,
        top_k: int = 5,
    ):
        """
        Args:
            indexer: The KnowledgeIndexer with a built FAISS index.
            similarity_threshold: Minimum similarity score to include a result.
            top_k: Maximum number of results to retrieve.
        """
        self.indexer = indexer
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """
        Retrieve relevant knowledge chunks for the query.

        Args:
            query: The user's question or message.
            top_k: Override default top_k for this query.

        Returns:
            List of RetrievalResult objects, sorted by relevance.
        """
        k = top_k or self.top_k
        raw_results = self.indexer.search(query, top_k=k)

        results = []
        for chunk, score in raw_results:
            if score < self.similarity_threshold:
                continue
            results.append(RetrievalResult(
                content=chunk.content,
                source=chunk.source_file,
                chunk_key=chunk.chunk_key,
                score=score,
            ))

        logger.info(
            f"Retrieved {len(results)} chunks for query: '{query[:50]}...' "
            f"(threshold={self.similarity_threshold})"
        )
        return results

    def get_context_sources(self, results: List[RetrievalResult]) -> List[str]:
        """
        Extract unique source names from retrieval results.

        Returns:
            Deduplicated list of source file names.
        """
        seen = set()
        sources = []
        for r in results:
            source_key = f"{r.source}_{r.chunk_key}" if r.chunk_key.startswith("line_") else r.chunk_key
            display_source = r.source if r.chunk_key.startswith("line_") else f"{r.chunk_key.lower()}_traits"
            if r.source not in seen:
                seen.add(r.source)
                sources.append(r.source)
        return sources

    def format_context(self, results: List[RetrievalResult]) -> str:
        """
        Format retrieval results into a context string for the LLM prompt.

        Returns:
            Formatted context string with source annotations.
        """
        if not results:
            return ""

        context_parts = ["### Retrieved Astrological Knowledge:\n"]
        for i, r in enumerate(results, 1):
            context_parts.append(
                f"[Source: {r.source} | Key: {r.chunk_key} | Relevance: {r.score:.2f}]\n"
                f"{r.content}\n"
            )

        return "\n".join(context_parts)
