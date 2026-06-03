"""
Text chunking using LangChain's RecursiveCharacterTextSplitter.
Converts raw page dicts into LangChain Document objects with rich metadata.
"""

from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from core.config import CHUNK_SIZE, CHUNK_OVERLAP


class DocumentChunker:
    """Split extracted PDF pages into chunked LangChain Documents."""

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def chunk_documents(self, loaded_pages: List[Dict], source_name: str) -> List[Document]:
        """
        Convert loaded page dicts into chunked Documents.

        Args:
            loaded_pages: Output from PDFLoader.load_pdf().
            source_name: The source filename (e.g., "lecture5.pdf").

        Returns:
            List of LangChain Document objects, each with metadata:
                source, page, total_pages, chunk_id.
        """
        all_chunks: List[Document] = []
        total_pages = loaded_pages[0]["total_pages"] if loaded_pages else 0

        for page_data in loaded_pages:
            page_num = page_data["page_number"]
            text = page_data["text"]

            if not text.strip():
                continue  # skip empty pages

            # Create a base document for this page
            base_doc = Document(
                page_content=text,
                metadata={
                    "source": source_name,
                    "page": page_num,
                    "total_pages": total_pages,
                }
            )

            # Split into chunks; each inherits the base metadata
            chunks = self.splitter.split_documents([base_doc])

            # Add unique chunk IDs
            for idx, chunk in enumerate(chunks):
                chunk.metadata["chunk_id"] = f"{source_name}_p{page_num}_c{idx}"

            all_chunks.extend(chunks)

        return all_chunks

    @staticmethod
    def get_chunk_stats(chunks: List[Document]) -> Dict:
        """
        Return summary statistics for the chunked documents.

        Returns:
            Dict with total_chunks, avg_chunk_size, pages_covered.
        """
        if not chunks:
            return {"total_chunks": 0, "avg_chunk_size": 0, "pages_covered": []}

        pages = sorted(set(chunk.metadata.get("page", 0) for chunk in chunks))
        avg_size = sum(len(chunk.page_content) for chunk in chunks) / len(chunks)

        return {
            "total_chunks": len(chunks),
            "avg_chunk_size": round(avg_size),
            "pages_covered": pages,
        }
