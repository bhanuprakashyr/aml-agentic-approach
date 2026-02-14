"""
Case Retriever: Full retrieval pipeline combining FAISS and Case Memory.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from .faiss_index import FAISSIndex


class Retriever:
    """
    Full case retrieval pipeline.
    
    Combines:
    - FAISS index for fast similarity search
    - Case memory for retrieving full case details
    
    Args:
        faiss_index: FAISSIndex instance
        case_memory: CaseMemory instance
        
    Usage:
        retriever = Retriever(faiss_index, case_memory)
        similar_cases = retriever.retrieve(query_embedding, k=3)
    """
    
    def __init__(
        self,
        faiss_index: FAISSIndex,
        case_memory  # CaseMemory instance
    ):
        self.faiss_index = faiss_index
        self.case_memory = case_memory
        
        # Build mapping from case ID to index position
        self._case_id_to_idx = {
            case.case_id: i for i, case in enumerate(case_memory.cases)
        }
        self._idx_to_case_id = {
            i: case.case_id for i, case in enumerate(case_memory.cases)
        }
    
    def retrieve(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filter_label: Optional[str] = None,
        min_similarity: Optional[float] = None
    ) -> List[Dict]:
        """
        Retrieve k most similar cases.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of cases to retrieve
            filter_label: Only return cases with this label
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of case dictionaries with similarity scores
        """
        # Search for more than k in case we need to filter
        search_k = k * 3 if filter_label else k
        
        distances, indices = self.faiss_index.search(
            query_embedding.reshape(1, -1), search_k
        )
        
        results = []
        
        for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if idx >= len(self._idx_to_case_id):
                continue
            
            case_id = self._idx_to_case_id[idx]
            case = self.case_memory.get_case(case_id)
            
            if case is None:
                continue
            
            # Apply filters
            if filter_label and case.true_label != filter_label:
                continue
            
            # Convert distance to similarity (for L2)
            similarity = 1 / (1 + dist)
            
            if min_similarity and similarity < min_similarity:
                continue
            
            results.append({
                'case': case,
                'case_id': case_id,
                'similarity': similarity,
                'distance': float(dist),
                'rank': len(results) + 1
            })
            
            if len(results) >= k:
                break
        
        return results
    
    def retrieve_for_icl(
        self,
        query_embedding: np.ndarray,
        k: int = 3,
        include_both_labels: bool = True
    ) -> List[Dict]:
        """
        Retrieve cases optimized for In-Context Learning.
        
        Ensures a mix of illicit and licit examples for better LLM reasoning.
        
        Args:
            query_embedding: Query embedding
            k: Total number of cases to retrieve
            include_both_labels: Whether to include both illicit and licit cases
            
        Returns:
            List of case dictionaries
        """
        if not include_both_labels:
            return self.retrieve(query_embedding, k)
        
        # Get some of each type
        n_per_label = max(1, k // 2)
        
        illicit_cases = self.retrieve(
            query_embedding, n_per_label, filter_label='illicit'
        )
        licit_cases = self.retrieve(
            query_embedding, n_per_label, filter_label='licit'
        )
        
        # Combine and sort by similarity
        all_cases = illicit_cases + licit_cases
        all_cases.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Re-rank
        for i, case in enumerate(all_cases[:k]):
            case['rank'] = i + 1
        
        return all_cases[:k]
    
    def get_retrieval_summary(
        self,
        results: List[Dict]
    ) -> str:
        """Generate summary of retrieval results."""
        if not results:
            return "No similar cases found."
        
        lines = [
            f"Retrieved {len(results)} similar cases:",
            ""
        ]
        
        for r in results:
            case = r['case']
            lines.append(
                f"  #{r['rank']}: Case {r['case_id']} "
                f"(similarity: {r['similarity']:.3f}, label: {case.true_label})"
            )
        
        return "\n".join(lines)


def create_retriever(case_memory, index_type: str = 'flat', metric: str = 'l2'):
    """
    Create retriever from case memory.
    
    Args:
        case_memory: CaseMemory instance
        index_type: FAISS index type
        metric: Distance metric
        
    Returns:
        Retriever instance
    """
    embeddings = case_memory.get_embeddings()
    
    if len(embeddings) == 0:
        raise ValueError("Case memory is empty")
    
    embedding_dim = embeddings.shape[1]
    
    # Build FAISS index
    faiss_index = FAISSIndex(embedding_dim, index_type, metric)
    faiss_index.build(embeddings)
    
    return Retriever(faiss_index, case_memory)


if __name__ == '__main__':
    print("Retriever module - use with CaseMemory")
