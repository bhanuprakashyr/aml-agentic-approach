"""
FAISS Index for efficient similarity search.

Provides fast nearest neighbor search for case retrieval.
"""

import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
import faiss


class FAISSIndex:
    """
    FAISS index for efficient similarity search.
    
    Wraps FAISS to provide:
    - Building index from embeddings
    - Fast k-NN search
    - Saving/loading index to disk
    
    Args:
        embedding_dim: Dimension of embeddings
        index_type: Type of index ('flat', 'ivf', 'hnsw')
        metric: Distance metric ('l2', 'cosine')
        
    Usage:
        index = FAISSIndex(embedding_dim=64)
        index.build(embeddings)
        distances, indices = index.search(query_embedding, k=5)
    """
    
    def __init__(
        self,
        embedding_dim: int = 64,
        index_type: str = 'flat',
        metric: str = 'l2'
    ):
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.metric = metric
        
        self.index: Optional[faiss.Index] = None
        self._num_vectors = 0
        self._is_trained = False
        
    def build(
        self,
        embeddings: np.ndarray,
        nlist: int = 100
    ):
        """
        Build the FAISS index from embeddings.
        
        Args:
            embeddings: Array of shape (n_samples, embedding_dim)
            nlist: Number of clusters for IVF index
        """
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        
        n_samples, dim = embeddings.shape
        
        if dim != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embedding_dim}, got {dim}"
            )
        
        # Normalize for cosine similarity
        if self.metric == 'cosine':
            faiss.normalize_L2(embeddings)
        
        # Create index based on type
        if self.index_type == 'flat':
            # Exact search - best for small datasets (<10K)
            if self.metric == 'cosine':
                self.index = faiss.IndexFlatIP(dim)  # Inner product for cosine
            else:
                self.index = faiss.IndexFlatL2(dim)
        
        elif self.index_type == 'ivf':
            # Approximate search with inverted file
            quantizer = faiss.IndexFlatL2(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            self.index.train(embeddings)
        
        elif self.index_type == 'hnsw':
            # Hierarchical Navigable Small World graph
            self.index = faiss.IndexHNSWFlat(dim, 32)  # 32 = M parameter
        
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        # Add vectors to index
        self.index.add(embeddings)
        self._num_vectors = n_samples
        self._is_trained = True
        
        print(f"Built FAISS index: {n_samples} vectors, dim={dim}, type={self.index_type}")
    
    def search(
        self,
        query: np.ndarray,
        k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors.
        
        Args:
            query: Query embedding(s) of shape (n_queries, embedding_dim) or (embedding_dim,)
            k: Number of neighbors to return
            
        Returns:
            Tuple of (distances, indices)
            - distances: Shape (n_queries, k)
            - indices: Shape (n_queries, k)
        """
        if self.index is None:
            raise ValueError("Index not built. Call build() first.")
        
        # Handle single query
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        if query.dtype != np.float32:
            query = query.astype(np.float32)
        
        # Normalize for cosine similarity
        if self.metric == 'cosine':
            faiss.normalize_L2(query)
        
        # Search
        distances, indices = self.index.search(query, k)
        
        return distances, indices
    
    def search_single(
        self,
        query: np.ndarray,
        k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Search for a single query and return results as list.
        
        Args:
            query: Query embedding of shape (embedding_dim,)
            k: Number of neighbors
            
        Returns:
            List of (index, distance) tuples
        """
        distances, indices = self.search(query, k)
        return list(zip(indices[0].tolist(), distances[0].tolist()))
    
    def add(self, embeddings: np.ndarray):
        """
        Add new embeddings to the index.
        
        Args:
            embeddings: New embeddings to add
        """
        if self.index is None:
            raise ValueError("Index not built. Call build() first.")
        
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        
        if self.metric == 'cosine':
            faiss.normalize_L2(embeddings)
        
        self.index.add(embeddings)
        self._num_vectors += len(embeddings)
    
    def save(self, path: str):
        """Save index to disk."""
        if self.index is None:
            raise ValueError("No index to save.")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))
        print(f"Saved FAISS index to {path}")
    
    def load(self, path: str):
        """Load index from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Index file not found: {path}")
        
        self.index = faiss.read_index(str(path))
        self._num_vectors = self.index.ntotal
        self._is_trained = True
        print(f"Loaded FAISS index from {path}: {self._num_vectors} vectors")
    
    @property
    def num_vectors(self) -> int:
        """Number of vectors in the index."""
        return self._num_vectors
    
    @property
    def is_trained(self) -> bool:
        """Whether the index is trained and ready."""
        return self._is_trained


class CaseRetriever:
    """
    Retrieves similar cases using FAISS index.
    
    Combines FAISS index with case memory for full retrieval pipeline.
    
    Args:
        faiss_index: FAISSIndex instance
        case_ids: List of case IDs corresponding to index vectors
        
    Usage:
        retriever = CaseRetriever(faiss_index, case_ids)
        similar_case_ids = retriever.retrieve(query_embedding, k=3)
    """
    
    def __init__(
        self,
        faiss_index: FAISSIndex,
        case_ids: List[int]
    ):
        self.faiss_index = faiss_index
        self.case_ids = case_ids
        
        if len(case_ids) != faiss_index.num_vectors:
            raise ValueError(
                f"Mismatch: {len(case_ids)} case IDs but {faiss_index.num_vectors} vectors in index"
            )
    
    def retrieve(
        self,
        query: np.ndarray,
        k: int = 5,
        return_distances: bool = False
    ) -> List[int]:
        """
        Retrieve k most similar case IDs.
        
        Args:
            query: Query embedding
            k: Number of cases to retrieve
            return_distances: Whether to return distances too
            
        Returns:
            List of case IDs (or list of (case_id, distance) if return_distances=True)
        """
        distances, indices = self.faiss_index.search(query, k)
        
        # Map indices to case IDs
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.case_ids):
                case_id = self.case_ids[idx]
                if return_distances:
                    results.append((case_id, distances[0][i]))
                else:
                    results.append(case_id)
        
        return results
    
    def retrieve_with_scores(
        self,
        query: np.ndarray,
        k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Retrieve k most similar cases with similarity scores.
        
        Args:
            query: Query embedding
            k: Number of cases
            
        Returns:
            List of (case_id, similarity_score) tuples
        """
        return self.retrieve(query, k, return_distances=True)


def build_faiss_index(
    embeddings: np.ndarray,
    index_type: str = 'flat',
    metric: str = 'l2',
    save_path: Optional[str] = None
) -> FAISSIndex:
    """
    Convenience function to build a FAISS index.
    
    Args:
        embeddings: Embeddings array
        index_type: Type of index
        metric: Distance metric
        save_path: Optional path to save index
        
    Returns:
        Built FAISSIndex instance
    """
    embedding_dim = embeddings.shape[1]
    index = FAISSIndex(embedding_dim, index_type, metric)
    index.build(embeddings)
    
    if save_path:
        index.save(save_path)
    
    return index


if __name__ == '__main__':
    # Test FAISS index
    print("Testing FAISS Index")
    print("="*50)
    
    # Create dummy data
    np.random.seed(42)
    n_samples = 1000
    embedding_dim = 64
    
    embeddings = np.random.randn(n_samples, embedding_dim).astype(np.float32)
    
    # Test different index types
    for index_type in ['flat', 'hnsw']:
        print(f"\nTesting {index_type} index:")
        
        index = FAISSIndex(embedding_dim, index_type=index_type)
        index.build(embeddings)
        
        # Search
        query = np.random.randn(embedding_dim).astype(np.float32)
        distances, indices = index.search(query, k=5)
        
        print(f"  Top 5 neighbors: {indices[0].tolist()}")
        print(f"  Distances: {distances[0].tolist()}")
    
    # Test retriever
    print("\nTesting CaseRetriever:")
    
    index = FAISSIndex(embedding_dim)
    index.build(embeddings)
    
    case_ids = list(range(n_samples))
    retriever = CaseRetriever(index, case_ids)
    
    query = np.random.randn(embedding_dim).astype(np.float32)
    results = retriever.retrieve_with_scores(query, k=3)
    
    print(f"  Retrieved cases: {results}")
