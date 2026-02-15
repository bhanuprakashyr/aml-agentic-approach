"""
FAISS-based retrieval for similar case lookup.
"""

from .faiss_index import FAISSIndex
from .retriever import Retriever

__all__ = ['FAISSIndex', 'Retriever']
