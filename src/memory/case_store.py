"""
Case Memory: Storage for labeled and explained transactions.

Stores curated cases with:
- Node embeddings (for FAISS)
- GNNExplainer outputs (for ICL prompts)
- Metadata (labels, scores, etc.)
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import pickle


@dataclass
class Case:
    """A single case in the memory."""
    case_id: int
    node_idx: int
    original_tx_id: Optional[str]
    
    # Prediction info
    fraud_score: float
    predicted_label: str
    true_label: str
    
    # Embedding for FAISS
    embedding: List[float]
    
    # Explanation
    explanation: Dict
    
    # Metadata
    time_step: int
    in_degree: int
    out_degree: int
    metadata: Dict
    
    created_at: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Case':
        """Create from dictionary."""
        return cls(**data)


class CaseMemory:
    """
    Storage for labeled and explained fraud cases.
    
    Manages a collection of cases with their embeddings and explanations.
    Embeddings are stored separately for FAISS indexing.
    
    Args:
        storage_dir: Directory to store cases
        
    Usage:
        memory = CaseMemory(storage_dir='./case_memory')
        memory.add_case(case)
        memory.save()
        
        # Later
        memory = CaseMemory.load(storage_dir='./case_memory')
        cases = memory.get_cases(label='illicit', top_k=10)
    """
    
    def __init__(self, storage_dir: str = './case_memory'):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.cases: List[Case] = []
        self._embeddings: Optional[np.ndarray] = None
        self._next_id = 0
        
    def add_case(
        self,
        node_idx: int,
        embedding: np.ndarray,
        fraud_score: float,
        predicted_label: str,
        true_label: str,
        explanation: Dict,
        time_step: int = 0,
        in_degree: int = 0,
        out_degree: int = 0,
        original_tx_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Case:
        """
        Add a case to memory.
        
        Args:
            node_idx: Node index in the graph
            embedding: Node embedding vector
            fraud_score: Predicted fraud probability
            predicted_label: Predicted class
            true_label: Ground truth label
            explanation: GNNExplainer output
            time_step: Transaction time step
            in_degree: Number of incoming edges
            out_degree: Number of outgoing edges
            original_tx_id: Original transaction ID
            metadata: Additional metadata
            
        Returns:
            Created Case object
        """
        case = Case(
            case_id=self._next_id,
            node_idx=node_idx,
            original_tx_id=original_tx_id,
            fraud_score=fraud_score,
            predicted_label=predicted_label,
            true_label=true_label,
            embedding=embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
            explanation=explanation,
            time_step=time_step,
            in_degree=in_degree,
            out_degree=out_degree,
            metadata=metadata or {},
            created_at=datetime.now().isoformat()
        )
        
        self.cases.append(case)
        self._next_id += 1
        self._embeddings = None  # Invalidate cached embeddings
        
        return case
    
    def get_case(self, case_id: int) -> Optional[Case]:
        """Get a case by ID."""
        for case in self.cases:
            if case.case_id == case_id:
                return case
        return None
    
    def get_cases(
        self,
        label: Optional[str] = None,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None,
        time_step: Optional[int] = None,
        top_k: Optional[int] = None
    ) -> List[Case]:
        """
        Get cases matching criteria.
        
        Args:
            label: Filter by true label ('illicit', 'licit')
            min_score: Minimum fraud score
            max_score: Maximum fraud score
            time_step: Filter by time step
            top_k: Return only top-k cases (by fraud score)
            
        Returns:
            List of matching cases
        """
        filtered = self.cases.copy()
        
        if label is not None:
            filtered = [c for c in filtered if c.true_label == label]
        
        if min_score is not None:
            filtered = [c for c in filtered if c.fraud_score >= min_score]
        
        if max_score is not None:
            filtered = [c for c in filtered if c.fraud_score <= max_score]
        
        if time_step is not None:
            filtered = [c for c in filtered if c.time_step == time_step]
        
        if top_k is not None:
            filtered.sort(key=lambda c: c.fraud_score, reverse=True)
            filtered = filtered[:top_k]
        
        return filtered
    
    def get_embeddings(self) -> np.ndarray:
        """
        Get all embeddings as numpy array.
        
        Returns:
            Array of shape (num_cases, embedding_dim)
        """
        if self._embeddings is None:
            if len(self.cases) == 0:
                return np.array([])
            self._embeddings = np.array([c.embedding for c in self.cases], dtype=np.float32)
        return self._embeddings
    
    def get_case_ids(self) -> List[int]:
        """Get all case IDs in order."""
        return [c.case_id for c in self.cases]
    
    @property
    def num_cases(self) -> int:
        """Number of cases in memory."""
        return len(self.cases)
    
    @property
    def embedding_dim(self) -> int:
        """Embedding dimension."""
        if len(self.cases) == 0:
            return 0
        return len(self.cases[0].embedding)
    
    def get_statistics(self) -> Dict:
        """Get statistics about the case memory."""
        if len(self.cases) == 0:
            return {'num_cases': 0}
        
        labels = [c.true_label for c in self.cases]
        scores = [c.fraud_score for c in self.cases]
        
        return {
            'num_cases': len(self.cases),
            'num_illicit': labels.count('illicit'),
            'num_licit': labels.count('licit'),
            'avg_fraud_score': np.mean(scores),
            'min_fraud_score': np.min(scores),
            'max_fraud_score': np.max(scores),
            'embedding_dim': self.embedding_dim
        }
    
    def save(self):
        """Save case memory to disk."""
        # Save cases as JSON
        cases_path = self.storage_dir / 'cases.json'
        cases_data = [c.to_dict() for c in self.cases]
        with open(cases_path, 'w') as f:
            json.dump(cases_data, f, indent=2)
        
        # Save embeddings as numpy array (more efficient for FAISS)
        embeddings_path = self.storage_dir / 'embeddings.npy'
        np.save(embeddings_path, self.get_embeddings())
        
        # Save metadata
        meta_path = self.storage_dir / 'metadata.json'
        metadata = {
            'num_cases': self.num_cases,
            'embedding_dim': self.embedding_dim,
            'next_id': self._next_id,
            'saved_at': datetime.now().isoformat(),
            'statistics': self.get_statistics()
        }
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved {self.num_cases} cases to {self.storage_dir}")
    
    @classmethod
    def load(cls, storage_dir: str) -> 'CaseMemory':
        """
        Load case memory from disk.
        
        Args:
            storage_dir: Directory containing saved cases
            
        Returns:
            Loaded CaseMemory instance
        """
        memory = cls(storage_dir)
        
        cases_path = memory.storage_dir / 'cases.json'
        if not cases_path.exists():
            print(f"No cases found at {cases_path}")
            return memory
        
        # Load cases
        with open(cases_path, 'r') as f:
            cases_data = json.load(f)
        
        memory.cases = [Case.from_dict(c) for c in cases_data]
        
        # Load metadata
        meta_path = memory.storage_dir / 'metadata.json'
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            memory._next_id = metadata.get('next_id', len(memory.cases))
        else:
            memory._next_id = len(memory.cases)
        
        # Load embeddings (optional, will be recomputed if not found)
        embeddings_path = memory.storage_dir / 'embeddings.npy'
        if embeddings_path.exists():
            memory._embeddings = np.load(embeddings_path)
        
        print(f"Loaded {memory.num_cases} cases from {storage_dir}")
        return memory


class CaseSelector:
    """
    Selects cases for the case memory based on model predictions.
    
    Selects a diverse set of cases:
    - High-confidence illicit cases
    - High-confidence licit cases
    - Edge cases (model uncertain)
    
    Args:
        num_illicit: Number of illicit cases to select
        num_licit: Number of licit cases to select
        num_edge: Number of edge cases to select
        illicit_threshold: Min fraud score for illicit selection
        licit_threshold: Max fraud score for licit selection
        edge_range: (min, max) fraud score range for edge cases
    """
    
    def __init__(
        self,
        num_illicit: int = 350,
        num_licit: int = 350,
        num_edge: int = 300,
        illicit_threshold: float = 0.85,
        licit_threshold: float = 0.15,
        edge_range: Tuple[float, float] = (0.4, 0.6)
    ):
        self.num_illicit = num_illicit
        self.num_licit = num_licit
        self.num_edge = num_edge
        self.illicit_threshold = illicit_threshold
        self.licit_threshold = licit_threshold
        self.edge_range = edge_range
    
    def select(
        self,
        labels: np.ndarray,
        fraud_scores: np.ndarray,
        embeddings: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        Select cases for memory.
        
        Args:
            labels: True labels (0=licit, 1=illicit, -1=unknown)
            fraud_scores: Predicted fraud probabilities
            embeddings: Node embeddings
            
        Returns:
            Tuple of (selected_indices, selection_stats)
        """
        selected = []
        stats = {}
        
        # High-confidence illicit cases
        illicit_mask = (labels == 1) & (fraud_scores >= self.illicit_threshold)
        illicit_indices = np.where(illicit_mask)[0]
        
        if len(illicit_indices) >= self.num_illicit:
            # Select diverse cases using embedding clustering
            selected_illicit = self._select_diverse(
                illicit_indices, embeddings, self.num_illicit
            )
        else:
            selected_illicit = illicit_indices
        
        selected.extend(selected_illicit)
        stats['illicit'] = len(selected_illicit)
        
        # High-confidence licit cases
        licit_mask = (labels == 0) & (fraud_scores <= self.licit_threshold)
        licit_indices = np.where(licit_mask)[0]
        
        if len(licit_indices) >= self.num_licit:
            selected_licit = self._select_diverse(
                licit_indices, embeddings, self.num_licit
            )
        else:
            selected_licit = licit_indices
        
        selected.extend(selected_licit)
        stats['licit'] = len(selected_licit)
        
        # Edge cases (uncertain predictions)
        labeled_mask = labels != -1
        edge_mask = (
            labeled_mask &
            (fraud_scores >= self.edge_range[0]) &
            (fraud_scores <= self.edge_range[1])
        )
        edge_indices = np.where(edge_mask)[0]
        
        if len(edge_indices) >= self.num_edge:
            selected_edge = self._select_diverse(
                edge_indices, embeddings, self.num_edge
            )
        else:
            selected_edge = edge_indices
        
        selected.extend(selected_edge)
        stats['edge'] = len(selected_edge)
        stats['total'] = len(selected)
        
        return np.array(selected), stats
    
    def _select_diverse(
        self,
        indices: np.ndarray,
        embeddings: np.ndarray,
        n: int
    ) -> np.ndarray:
        """Select diverse cases using random sampling (simple approach)."""
        if len(indices) <= n:
            return indices
        
        # Simple random selection (could use k-means for more diversity)
        selected = np.random.choice(indices, size=n, replace=False)
        return selected


if __name__ == '__main__':
    # Test case memory
    print("Testing CaseMemory")
    print("="*50)
    
    # Create memory
    memory = CaseMemory(storage_dir='./test_case_memory')
    
    # Add some test cases
    for i in range(10):
        memory.add_case(
            node_idx=i,
            embedding=np.random.randn(64),
            fraud_score=np.random.random(),
            predicted_label='illicit' if np.random.random() > 0.5 else 'licit',
            true_label='illicit' if i < 5 else 'licit',
            explanation={'test': True},
            time_step=i % 5
        )
    
    # Get statistics
    stats = memory.get_statistics()
    print(f"\nStatistics: {stats}")
    
    # Get embeddings
    embeddings = memory.get_embeddings()
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Filter cases
    illicit_cases = memory.get_cases(label='illicit')
    print(f"Illicit cases: {len(illicit_cases)}")
    
    # Save and reload
    memory.save()
    
    loaded_memory = CaseMemory.load('./test_case_memory')
    print(f"Loaded {loaded_memory.num_cases} cases")
    
    # Cleanup
    import shutil
    shutil.rmtree('./test_case_memory')
