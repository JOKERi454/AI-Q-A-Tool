"""
Multi-turn conversation memory manager.
Maintains a sliding window of Q&A exchanges with LangChain-compatible formatting.
"""

from typing import List, Dict
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from core.config import MAX_HISTORY


class MemoryManager:
    """
    Lightweight conversation memory with serialization support.
    Designed for Streamlit's st.session_state rerun model.
    """

    def __init__(self, max_history: int = MAX_HISTORY):
        self.max_history = max_history
        self._history: List[Dict[str, str]] = []

    def add_exchange(self, question: str, answer: str):
        """Record a Q&A exchange and trim to max_history."""
        self._history.append({"question": question, "answer": answer})
        if len(self._history) > self.max_history:
            self._history = self._history[-self.max_history:]

    def get_history_for_langchain(self) -> List[BaseMessage]:
        """
        Convert stored history to LangChain message format.
        Used by the history-aware retriever chain.

        Returns:
            List alternating HumanMessage and AIMessage objects.
        """
        messages: List[BaseMessage] = []
        for exchange in self._history:
            messages.append(HumanMessage(content=exchange["question"]))
            messages.append(AIMessage(content=exchange["answer"]))
        return messages

    def get_history_as_text(self) -> str:
        """
        Format history as a readable text block (used for debugging/display).

        Returns:
            Newline-separated conversation text.
        """
        if not self._history:
            return "(No conversation history)"

        lines = []
        for i, ex in enumerate(self._history, 1):
            lines.append(f"Q{i}: {ex['question']}")
            lines.append(f"A{i}: {ex['answer'][:200]}{'...' if len(ex['answer']) > 200 else ''}")
        return "\n".join(lines)

    def to_dict_list(self) -> List[Dict[str, str]]:
        """Serialize history for Streamlit session state storage."""
        return list(self._history)

    def from_dict_list(self, history: List[Dict[str, str]]):
        """Restore history from serialized data."""
        self._history = list(history)

    def clear(self):
        """Reset all conversation history."""
        self._history = []

    @property
    def is_empty(self) -> bool:
        return len(self._history) == 0

    @property
    def exchange_count(self) -> int:
        return len(self._history)

    def __len__(self) -> int:
        return len(self._history)
