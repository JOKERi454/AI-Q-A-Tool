"""
Streamlit UI for the RAG-based PDF Q&A Tool.
Supports PDF upload, multi-turn chat with source citations.
"""

import os
import tempfile
import streamlit as st

from core.config import validate_config
from core.pdf_loader import PDFLoader
from core.chunker import DocumentChunker
from core.embeddings_store import EmbeddingStore
from core.memory_manager import MemoryManager
from core.rag_chain import RAGChain
from core.citation_tracker import CitationTracker

# ── Page Configuration ──────────────────────────────────────────

st.set_page_config(
    page_title="PDF Q&A Tool",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────

st.markdown("""
<style>
    .citation-card {
        background-color: #f0f2f6;
        border-left: 3px solid #4a90d9;
        padding: 0.5rem 1rem;
        margin: 0.25rem 0;
        border-radius: 4px;
        font-size: 0.9rem;
    }
    .citation-card .page-badge {
        background-color: #4a90d9;
        color: white;
        padding: 0.1rem 0.5rem;
        border-radius: 10px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .processing-stats {
        background-color: #e8f5e9;
        padding: 0.75rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-size: 0.85rem;
    }
    .warning-box {
        background-color: #fff3e0;
        padding: 0.75rem;
        border-radius: 8px;
        border-left: 3px solid #ff9800;
    }
</style>
""", unsafe_allow_html=True)

# ── Session State Initialization ────────────────────────────────

DEFAULT_SESSION = {
    "messages": [],           # Chat messages: [{"role": "user"|"assistant", "content": ..., "citations": [...]}]
    "processing_complete": False,
    "loaded_sources": [],     # List of source filenames in the vector store
    "total_chunks": 0,
    "memory": [],             # Serialized MemoryManager state
    "embedding_store": None,  # EmbeddingStore instance (cached)
    "rag_chain": None,        # RAGChain instance (cached)
}

for key, default in DEFAULT_SESSION.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Helper Functions ─────────────────────────────────────────────

def get_embedding_store() -> EmbeddingStore:
    """Get or create the cached EmbeddingStore instance."""
    if st.session_state.embedding_store is None:
        st.session_state.embedding_store = EmbeddingStore()
    return st.session_state.embedding_store


def get_memory_manager() -> MemoryManager:
    """Get or restore the MemoryManager from session state."""
    mm = MemoryManager()
    if st.session_state.memory:
        mm.from_dict_list(st.session_state.memory)
    return mm


def get_rag_chain() -> RAGChain:
    """Get or create the cached RAGChain instance."""
    if st.session_state.rag_chain is None:
        embedding_store = get_embedding_store()
        memory_manager = get_memory_manager()
        st.session_state.rag_chain = RAGChain(embedding_store, memory_manager)
    # Always sync latest memory into the chain
    st.session_state.rag_chain.memory_manager = get_memory_manager()
    return st.session_state.rag_chain


def process_pdf(uploaded_file) -> tuple:
    """
    Process an uploaded PDF through the full ingestion pipeline.
    Returns (success: bool, message: str, stats: dict | None).
    """
    # Save uploaded file to a temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        # Step 1: Load PDF
        pages = PDFLoader.load_pdf(tmp_path)
        if not pages:
            return False, "No pages extracted from the PDF.", None

        source_name = uploaded_file.name

        # Step 2: Chunk
        chunker = DocumentChunker()
        chunks = chunker.chunk_documents(pages, source_name)
        if not chunks:
            return False, "No text chunks generated. The PDF may be image-only.", None

        # Step 3: Ingest into vector store
        embedding_store = get_embedding_store()
        count = embedding_store.ingest_documents(chunks, source_name)

        # Update session state
        if source_name not in st.session_state.loaded_sources:
            st.session_state.loaded_sources.append(source_name)
        st.session_state.processing_complete = True
        st.session_state.total_chunks = embedding_store.get_document_count()

        stats = {
            "source": source_name,
            "pages": len(pages),
            "chunks": count,
            "total_chunks": st.session_state.total_chunks,
        }

        return True, f"Successfully processed '{source_name}'.", stats

    except ValueError as e:
        return False, str(e), None
    except Exception as e:
        return False, f"Error processing PDF: {str(e)}", None
    finally:
        # Cleanup temp file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def clear_chat():
    """Clear chat messages and conversation memory."""
    st.session_state.messages = []
    st.session_state.memory = []


def clear_all_data():
    """Clear everything including the vector store."""
    clear_chat()
    embedding_store = get_embedding_store()
    embedding_store.clear_all()
    st.session_state.processing_complete = False
    st.session_state.loaded_sources = []
    st.session_state.total_chunks = 0
    st.session_state.embedding_store = None
    st.session_state.rag_chain = None


def display_citations(citations: list):
    """Render citation cards in the chat."""
    if not citations:
        return

    with st.expander(f"📚 View Sources ({len(citations)} pages cited)", expanded=False):
        for i, citation in enumerate(citations):
            source = citation.get("source", "Unknown")
            page = citation.get("page", "?")
            snippet = citation.get("snippet", "")

            st.markdown(f"""
            <div class="citation-card">
                <span class="page-badge">📄 {source}, Page {page}</span><br>
                <em>"{snippet}..."</em>
            </div>
            """, unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────

with st.sidebar:
    st.title("📄 PDF Q&A Tool")
    st.caption("Upload PDFs and ask questions about their content.")

    # ── Configuration status ──
    config_issues = validate_config()
    if config_issues:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.warning("⚠️ Configuration Issues:")
        for issue in config_issues:
            st.markdown(f"- {issue}")
        st.info(
            "Set your `DEEPSEEK_API_KEY` in a `.env` file. "
            "Copy `.env.example` to `.env` and add your key."
        )
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # ── PDF Upload ──
    st.subheader("📤 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Choose PDF files (slides, papers, etc.)",
        type=["pdf"],
        accept_multiple_files=True,
        disabled=bool(config_issues),
        key="pdf_uploader",
    )

    if uploaded_files:
        if st.button("🔄 Process PDFs", use_container_width=True, disabled=bool(config_issues)):
            with st.status("Processing PDFs...", expanded=True) as status:
                success_count = 0
                for uploaded_file in uploaded_files:
                    st.write(f"Processing: **{uploaded_file.name}**...")
                    success, message, stats = process_pdf(uploaded_file)
                    if success:
                        success_count += 1
                        st.write(f"  ✅ {message}")
                        if stats:
                            st.write(f"  📊 {stats['pages']} pages → {stats['chunks']} chunks")
                    else:
                        st.write(f"  ❌ {message}")

                if success_count > 0:
                    status.update(label=f"✅ {success_count} PDF(s) processed successfully!", state="complete")
                else:
                    status.update(label="❌ No PDFs could be processed.", state="error")

    st.divider()

    # ── Processing Status ──
    if st.session_state.processing_complete:
        st.subheader("📊 Status")
        st.markdown(f"""
        <div class="processing-stats">
            <strong>Documents loaded:</strong> {len(st.session_state.loaded_sources)}<br>
            <strong>Total chunks:</strong> {st.session_state.total_chunks}<br>
            <strong>Sources:</strong> {', '.join(st.session_state.loaded_sources) if st.session_state.loaded_sources else 'None'}
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Source Filter ──
    if st.session_state.loaded_sources:
        st.subheader("🔍 Filter by Document")
        source_filter = st.selectbox(
            "Limit search to:",
            options=["All Documents"] + st.session_state.loaded_sources,
            key="source_filter",
        )
    else:
        source_filter = "All Documents"

    st.divider()

    # ── Actions ──
    st.subheader("⚙️ Actions")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            clear_chat()
            st.rerun()
    with col2:
        if st.button("⚠️ Clear All", use_container_width=True):
            clear_all_data()
            st.rerun()

# ── Main Chat Area ───────────────────────────────────────────────

st.title("💬 Chat with your PDFs")

if not st.session_state.processing_complete:
    st.info(
        "👋 **Welcome!** Upload one or more PDF files in the sidebar "
        "and click **Process PDFs** to get started. "
        "You can then ask questions about their content."
    )
    # Show example placeholder
    with st.chat_message("assistant"):
        st.markdown(
            "I'm ready to help you understand your PDF documents. "
            "Upload course slides, research papers, or any text-based PDF, "
            "and I'll answer your questions with **source citations** showing "
            "exactly where the information came from.\n\n"
            "Try asking things like:\n"
            "- \"Summarize the key points of this paper\"\n"
            "- \"What does page 5 say about the methodology?\"\n"
            "- \"Explain the concept introduced in section 3\""
        )

# ── Render Chat Messages ──
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("citations"):
            display_citations(message["citations"])

# ── Chat Input ──
if prompt := st.chat_input(
    "Ask a question about the uploaded documents...",
    disabled=not st.session_state.processing_complete or bool(config_issues),
):
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "citations": [],
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                chain = get_rag_chain()
                active_filter = (
                    st.session_state.get("source_filter")
                    if st.session_state.get("source_filter") != "All Documents"
                    else None
                )
                result = chain.ask(prompt, source_filter=active_filter)

                answer = result["answer"]
                citations = result["citations"]

                # Display answer
                st.markdown(answer)

                # Display citations
                if citations:
                    display_citations(citations)

                # Save to session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "citations": citations,
                })

                # Persist memory state
                st.session_state.memory = chain.memory_manager.to_dict_list()

            except Exception as e:
                error_msg = f"❌ **Error:** {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "citations": [],
                })

# ── Footer ───────────────────────────────────────────────────────

st.divider()
st.caption(
    "Built with LangChain + DeepSeek + Chroma | "
    f"Documents in store: {st.session_state.total_chunks} chunks"
)
