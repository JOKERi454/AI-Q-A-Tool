"""
Embedding generation and Chroma vector store management.
Uses local sentence-transformers with HuggingFace mirror (GFW-compatible).
"""

import os
from typing import List, Optional
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from core.config import (
    LOCAL_EMBEDDING_MODEL,
    HF_ENDPOINT,
    CHROMA_PERSIST_DIR,
    RETRIEVAL_K,
    SEARCH_TYPE,
)


class EmbeddingStore:
    """Manages embeddings and Chroma vector store for PDF document chunks."""

    COLLECTION_NAME = "pdf_documents"

    def __init__(self):
        self._embedding_function = None
        self._vector_store: Optional[Chroma] = None

    def _get_embedding_function(self):
        """Lazy-init the embedding function based on configuration."""
        if self._embedding_function is not None:
            return self._embedding_function

        # Set HuggingFace mirror for users behind GFW
        if HF_ENDPOINT and "HF_ENDPOINT" not in os.environ:
            os.environ["HF_ENDPOINT"] = HF_ENDPOINT

        self._embedding_function = HuggingFaceEmbeddings(
            model_name=LOCAL_EMBEDDING_MODEL,
        )
        return self._embedding_function

    def get_vector_store(self) -> Chroma:
        """Get or create the Chroma vector store instance."""
        if self._vector_store is None:
            embedding = self._get_embedding_function()
            self._vector_store = Chroma(
                collection_name=self.COLLECTION_NAME,
                embedding_function=embedding,
                persist_directory=str(CHROMA_PERSIST_DIR),
            )
        return self._vector_store

    def ingest_documents(self, chunks: List[Document], source_name: str) -> int:
        """
        Add document chunks to the vector store.
        Clears existing chunks from the same source first (dedup).

        Args:
            chunks: List of LangChain Document objects to ingest.
            source_name: The source filename (used for dedup).

        Returns:
            Number of chunks successfully ingested.
        """
        if not chunks:
            return 0

        store = self.get_vector_store()

        # Remove existing documents from the same source to avoid duplicates
        try:
            existing_ids = store.get(where={"source": source_name})
            if existing_ids and existing_ids.get("ids"):
                store.delete(ids=existing_ids["ids"])
        except Exception:
            pass  # Collection might be empty or filter not supported yet

        # Add new chunks; Chroma handles embedding internally
        ids = store.add_documents(chunks)
        return len(ids)

    def similarity_search(self, query: str, k: int = RETRIEVAL_K,
                          source_filter: Optional[str] = None) -> List[Document]:
        """
        Retrieve the top-k most similar document chunks for a query.

        Args:
            query: The search query string.
            k: Number of results to return.
            source_filter: Optional source filename to restrict search scope.

        Returns:
            List of Document objects with full metadata.
        """
        store = self.get_vector_store()

        search_kwargs = {"k": k}
        if source_filter:
            search_kwargs["filter"] = {"source": source_filter}

        retriever = store.as_retriever(
            search_type=SEARCH_TYPE,
            search_kwargs=search_kwargs,
        )
        return retriever.invoke(query)

    def get_available_sources(self) -> List[str]:
        """
        Return a list of unique source filenames currently in the store.

        Returns:
            List of source filename strings.
        """
        store = self.get_vector_store()
        try:
            results = store.get()
            if results and results.get("metadatas"):
                sources = set(
                    m.get("source", "unknown")
                    for m in results["metadatas"]
                    if m
                )
                return sorted(sources)
        except Exception:
            pass
        return []

    def clear_source(self, source_name: str):
        """Remove all documents from a specific source file."""
        store = self.get_vector_store()
        try:
            existing = store.get(where={"source": source_name})
            if existing and existing.get("ids"):
                store.delete(ids=existing["ids"])
        except Exception:
            pass

    def clear_all(self):
        """Remove all documents from the vector store."""
        store = self.get_vector_store()
        try:
            all_ids = store.get()
            if all_ids and all_ids.get("ids"):
                store.delete(ids=all_ids["ids"])
        except Exception:
            pass

    def get_document_count(self) -> int:
        """Return the total number of chunks in the store."""
        store = self.get_vector_store()
        try:
            results = store.get()
            if results and results.get("ids"):
                return len(results["ids"])
        except Exception:
            pass
        return 0
