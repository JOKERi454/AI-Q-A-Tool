"""
RAG (Retrieval-Augmented Generation) chain assembly.
Wires together Chroma retrieval, history-aware rephrasing, and DeepSeek LLM generation.
"""

from typing import List, Dict, Optional
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_classic.chains import create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain

from core.config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    RETRIEVAL_K,
    SEARCH_TYPE,
)
from core.citation_tracker import CitationTracker
from core.embeddings_store import EmbeddingStore
from core.memory_manager import MemoryManager


# ── Prompt Templates ──────────────────────────────────────────────

# System prompt for rephrasing a follow-up question into a standalone question
CONTEXTUALIZE_SYSTEM_PROMPT = """\
Given a chat history and the latest user question which might reference \
context in the chat history, formulate a standalone question which can be \
understood without the chat history. Do NOT answer the question, just \
reformulate it if needed and otherwise return it as is."""

# System prompt for generating answers with citations
QA_SYSTEM_PROMPT = """\
You are a helpful academic assistant answering questions about PDF documents \
(course slides, research papers, etc.).

Answer the question based ONLY on the provided document context below. \
If the context does not contain enough information to answer, \
say "I could not find enough information in the uploaded documents to answer this question."

Guidelines:
- For every factual claim, cite the source page number in brackets: [Page N].
- If multiple pages support a claim, list them: [Page 3, 5].
- Keep answers clear, concise, and well-structured.
- Use bullet points for lists when helpful.

Document Context:
{context}"""

# ── Chain Builder ──────────────────────────────────────────────────


class RAGChain:
    """
    Full RAG pipeline: history-aware retrieval + document-grounded generation.
    """

    def __init__(self, embedding_store: EmbeddingStore, memory_manager: MemoryManager):
        self.embedding_store = embedding_store
        self.memory_manager = memory_manager
        self._llm = None
        self._chain = None
        self._retriever = None

    def _get_llm(self) -> ChatOpenAI:
        """Lazy-init the DeepSeek LLM via OpenAI-compatible interface."""
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=LLM_MODEL,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                openai_api_key=DEEPSEEK_API_KEY,
                openai_api_base=f"{DEEPSEEK_BASE_URL}/v1",
            )
        return self._llm

    def _get_retriever(self, source_filter: Optional[str] = None):
        """Get a Chroma retriever, optionally filtered by source."""
        store = self.embedding_store.get_vector_store()
        search_kwargs = {"k": RETRIEVAL_K}
        if source_filter:
            search_kwargs["filter"] = {"source": source_filter}

        return store.as_retriever(
            search_type=SEARCH_TYPE,
            search_kwargs=search_kwargs,
        )

    def _build_chain(self, source_filter: Optional[str] = None):
        """
        Assemble the LangChain RAG pipeline:
        1. History-aware retriever (rephrase question with chat context)
        2. Stuff-documents chain (generate answer from retrieved docs)
        3. Full retrieval chain (retrieve + generate)
        """
        llm = self._get_llm()
        retriever = self._get_retriever(source_filter)

        # Step 1: History-aware retriever
        contextualize_prompt = ChatPromptTemplate.from_messages([
            ("system", CONTEXTUALIZE_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        history_aware_retriever = create_history_aware_retriever(
            llm=llm,
            retriever=retriever,
            prompt=contextualize_prompt,
        )

        # Step 2: QA chain
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", QA_SYSTEM_PROMPT),
            ("human", "{input}"),
        ])
        question_answer_chain = create_stuff_documents_chain(
            llm=llm,
            prompt=qa_prompt,
        )

        # Step 3: Full retrieval chain
        self._chain = create_retrieval_chain(
            retriever=history_aware_retriever,
            combine_docs_chain=question_answer_chain,
        )

    def ask(self, question: str, source_filter: Optional[str] = None) -> Dict:
        """
        Process a user question through the RAG pipeline.

        Args:
            question: The user's question string.
            source_filter: Optional source filename to restrict search to.

        Returns:
            Dict with keys:
                - answer: The generated answer string.
                - source_documents: Retrieved Document objects.
                - citations: List of citation dicts.
                - question: The original question (echoed back).
        """
        # Rebuild chain if source filter changed
        self._build_chain(source_filter)

        # Get chat history for the history-aware retriever
        chat_history = self.memory_manager.get_history_for_langchain()

        # Invoke the chain
        result = self._chain.invoke({
            "input": question,
            "chat_history": chat_history,
        })

        answer = result.get("answer", "")
        source_docs = result.get("context", [])

        # Extract citations
        citations = CitationTracker.extract_citations(source_docs)

        # Store in memory for next turn
        self.memory_manager.add_exchange(question, answer)

        return {
            "answer": answer,
            "source_documents": source_docs,
            "citations": citations,
            "question": question,
        }

    def ask_stream(self, question: str, source_filter: Optional[str] = None):
        """
        Streaming version — yields answer tokens one at a time.
        Currently uses the non-streaming chain and yields the full result.
        For true token-level streaming, a different LangChain pattern is needed.
        """
        # Rebuild chain if needed
        self._build_chain(source_filter)

        chat_history = self.memory_manager.get_history_for_langchain()

        result = self._chain.invoke({
            "input": question,
            "chat_history": chat_history,
        })

        answer = result.get("answer", "")
        source_docs = result.get("context", [])

        # Store in memory
        self.memory_manager.add_exchange(question, answer)

        yield {
            "answer": answer,
            "source_documents": source_docs,
            "citations": CitationTracker.extract_citations(source_docs),
            "question": question,
        }
