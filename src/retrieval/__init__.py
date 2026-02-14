"""
FAISS-based retrieval for similar case lookup.
"""

from .faiss_index import FAISSIndex
from .retriever import CaseRetriever

__all__ = ['FAISSIndex', 'CaseRetriever']
