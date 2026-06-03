"""
Citation tracking: extract, format, and validate source citations
from retrieved documents and LLM responses.
"""

import re
from typing import List, Dict, Set
from langchain_core.documents import Document


class CitationTracker:
    """Extract and format source citations throughout the RAG pipeline."""

    # Regex to match [Page N] or [Page N, M, ...] patterns in LLM output
    CITATION_PATTERN = re.compile(r"\[Page\s+(\d+(?:,\s*\d+)*)\]")

    @staticmethod
    def format_docs_for_prompt(docs: List[Document]) -> str:
        """
        Format retrieved documents for the LLM context prompt,
        prefixing each chunk with its source and page number.

        Args:
            docs: Retrieved Document objects with metadata.

        Returns:
            Formatted string ready for inclusion in the prompt.
        """
        formatted_parts = []
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            content = doc.page_content.strip()
            formatted_parts.append(
                f"[Source: {source}, Page: {page}]\n{content}"
            )
        return "\n\n".join(formatted_parts)

    @staticmethod
    def extract_citations(docs: List[Document]) -> List[Dict]:
        """
        Extract unique citation metadata from retrieved documents.

        Args:
            docs: Retrieved Document objects.

        Returns:
            List of citation dicts with source, page, snippet.
            Deduplicated by (source, page).
        """
        seen: Set[str] = set()
        citations: List[Dict] = []

        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            key = f"{source}_p{page}"

            if key not in seen:
                seen.add(key)
                snippet = doc.page_content[:300].strip()
                citations.append({
                    "source": source,
                    "page": page,
                    "snippet": snippet,
                })

        # Sort by page number
        citations.sort(key=lambda c: c["page"] if isinstance(c["page"], int) else 0)
        return citations

    @staticmethod
    def parse_llm_citations(answer_text: str) -> List[int]:
        """
        Parse page numbers cited by the LLM in its answer.

        Args:
            answer_text: The LLM's full answer string.

        Returns:
            Sorted list of unique page numbers cited, e.g. [3, 5, 7].
        """
        matches = CitationTracker.CITATION_PATTERN.findall(answer_text)
        pages: Set[int] = set()
        for match in matches:
            for num in match.split(","):
                try:
                    pages.add(int(num.strip()))
                except ValueError:
                    continue
        return sorted(pages)

    @staticmethod
    def validate_citations(cited_pages: List[int], retrieved_pages: List[int]) -> Dict:
        """
        Check whether LLM-cited pages actually appeared in the retrieved context.

        Args:
            cited_pages: Pages cited by the LLM.
            retrieved_pages: Pages present in retrieved documents.

        Returns:
            Dict with matched, hallucinated, and missing lists.
        """
        retrieved_set = set(retrieved_pages)
        matched = [p for p in cited_pages if p in retrieved_set]
        hallucinated = [p for p in cited_pages if p not in retrieved_set]
        return {
            "matched": matched,
            "hallucinated": hallucinated,
            "all_valid": len(hallucinated) == 0,
        }
