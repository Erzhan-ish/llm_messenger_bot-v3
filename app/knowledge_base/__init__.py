"""
Knowledge base (KB) module.

- Builds searchable text chunks from a SOURCE knowledge base.
- Retrieves the most relevant chunks for a query.
"""

from .kb import KnowledgeBase, KBChunk

__all__ = ["KnowledgeBase", "KBChunk"]
