"""
Case Selector: Select representative cases for case memory.

Implements selection strategies to curate diverse, high-quality cases.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.cluster import KMeans


class CaseSelector:
    """
    Selects cases for the case memory based on model predictions.
    
    Implements multiple selection strategies:
    1. Confidence-based: High-confidence predictions
    2. Diversity-based: K-means clustering on embeddings
    3. Stratified: Balanced selection across classes
    
    Args:
        num_illicit: Number of illicit cases to select
        num_licit: Number of licit cases to select
        num_edge: Number of edge cases to select
        illicit_threshold: Min fraud score for illicit selection
        licit_threshold: Max fraud score for licit selection
        edge_range: (min, max) fraud score range for edge cases
        diversity_weight: Weight for diversity in selection (0-1)
    """
    
    def __init__(
        self,
        num_illicit: int = 350,
        num_licit: int = 350,
        num_edge: int = 300,
        illicit_threshold: float = 0.85,
        licit_threshold: float = 0.15,
        edge_range: Tuple[float, float] = (0.4, 0.6),
        diversity_weight: float = 0.5
    ):
        self.num_illicit = num_illicit
        self.num_licit = num_licit
        self.num_edge = num_edge
        self.illicit_threshold = illicit_threshold
        self.licit_threshold = licit_threshold
        self.edge_range = edge_range
        self.diversity_weight = diversity_weight
    
    def select(
        self,
        labels: np.ndarray,
        fraud_scores: np.ndarray,
        embeddings: np.ndarray,
        strategy: str = 'balanced'
    ) -> Tuple[np.ndarray, Dict]:
        """
        Select cases for memory.
        
        Args:
            labels: True labels (0=licit, 1=illicit, -1=unknown)
            fraud_scores: Predicted fraud probabilities
            embeddings: Node embeddings
            strategy: Selection strategy ('balanced', 'diverse', 'confident')
            
        Returns:
            Tuple of (selected_indices, selection_stats)
        """
        if strategy == 'balanced':
            return self._select_balanced(labels, fraud_scores, embeddings)
        elif strategy == 'diverse':
            return self._select_diverse(labels, fraud_scores, embeddings)
        elif strategy == 'confident':
            return self._select_confident(labels, fraud_scores, embeddings)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _select_balanced(
        self,
        labels: np.ndarray,
        fraud_scores: np.ndarray,
        embeddings: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """Balanced selection: equal parts illicit, licit, and edge cases."""
        selected = []
        stats = {}
        
        # High-confidence illicit cases
        illicit_mask = (labels == 1) & (fraud_scores >= self.illicit_threshold)
        illicit_indices = np.where(illicit_mask)[0]
        selected_illicit = self._select_from_pool(
            illicit_indices, embeddings, self.num_illicit
        )
        selected.extend(selected_illicit)
        stats['illicit'] = len(selected_illicit)
        stats['illicit_available'] = len(illicit_indices)
        
        # High-confidence licit cases
        licit_mask = (labels == 0) & (fraud_scores <= self.licit_threshold)
        licit_indices = np.where(licit_mask)[0]
        selected_licit = self._select_from_pool(
            licit_indices, embeddings, self.num_licit
        )
        selected.extend(selected_licit)
        stats['licit'] = len(selected_licit)
        stats['licit_available'] = len(licit_indices)
        
        # Edge cases
        labeled_mask = labels != -1
        edge_mask = (
            labeled_mask &
            (fraud_scores >= self.edge_range[0]) &
            (fraud_scores <= self.edge_range[1])
        )
        edge_indices = np.where(edge_mask)[0]
        selected_edge = self._select_from_pool(
            edge_indices, embeddings, self.num_edge
        )
        selected.extend(selected_edge)
        stats['edge'] = len(selected_edge)
        stats['edge_available'] = len(edge_indices)
        
        stats['total'] = len(selected)
        
        return np.array(selected), stats
    
    def _select_diverse(
        self,
        labels: np.ndarray,
        fraud_scores: np.ndarray,
        embeddings: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """Diversity-focused selection using clustering."""
        selected = []
        stats = {}
        
        # Only consider labeled data
        labeled_mask = labels != -1
        labeled_indices = np.where(labeled_mask)[0]
        
        if len(labeled_indices) == 0:
            return np.array([]), {'total': 0}
        
        # Get embeddings for labeled nodes
        labeled_embeddings = embeddings[labeled_indices]
        
        # Cluster embeddings
        total_cases = self.num_illicit + self.num_licit + self.num_edge
        n_clusters = min(total_cases, len(labeled_indices))
        
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(labeled_embeddings)
            
            # Select one case per cluster, prioritizing high-confidence predictions
            for cluster_id in range(n_clusters):
                cluster_mask = cluster_labels == cluster_id
                cluster_indices = labeled_indices[cluster_mask]
                
                if len(cluster_indices) == 0:
                    continue
                
                # Score by confidence (distance from 0.5)
                cluster_scores = fraud_scores[cluster_indices]
                confidence_scores = np.abs(cluster_scores - 0.5)
                
                # Select most confident case in cluster
                best_idx = cluster_indices[np.argmax(confidence_scores)]
                selected.append(best_idx)
        else:
            selected = labeled_indices.tolist()
        
        stats['total'] = len(selected)
        stats['n_clusters'] = n_clusters
        
        return np.array(selected), stats
    
    def _select_confident(
        self,
        labels: np.ndarray,
        fraud_scores: np.ndarray,
        embeddings: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """Confidence-focused selection: top predictions only."""
        selected = []
        stats = {}
        
        # Only labeled data
        labeled_mask = labels != -1
        
        # Sort by confidence (distance from 0.5)
        confidence = np.abs(fraud_scores - 0.5)
        confidence[~labeled_mask] = -1  # Exclude unlabeled
        
        # Get top confident predictions
        total_cases = self.num_illicit + self.num_licit + self.num_edge
        top_indices = np.argsort(confidence)[-total_cases:][::-1]
        
        # Filter to only include labeled
        selected = [i for i in top_indices if labeled_mask[i]]
        
        stats['total'] = len(selected)
        
        return np.array(selected), stats
    
    def _select_from_pool(
        self,
        indices: np.ndarray,
        embeddings: np.ndarray,
        n: int
    ) -> List[int]:
        """
        Select n cases from a pool using embedding diversity.
        
        Uses a greedy approach: iteratively select the case that is
        most different from already selected cases.
        """
        if len(indices) == 0:
            return []
        
        if len(indices) <= n:
            return indices.tolist()
        
        # Mix of random and diverse selection
        n_random = int(n * (1 - self.diversity_weight))
        n_diverse = n - n_random
        
        selected = []
        
        # Random selection
        if n_random > 0:
            random_idx = np.random.choice(len(indices), size=n_random, replace=False)
            selected.extend(indices[random_idx].tolist())
        
        # Diverse selection
        if n_diverse > 0 and len(indices) > len(selected):
            remaining = [i for i in range(len(indices)) if indices[i] not in selected]
            
            if len(remaining) > 0:
                # Get embeddings for remaining candidates
                remaining_indices = indices[remaining]
                remaining_embeddings = embeddings[remaining_indices]
                
                # Greedy diverse selection
                for _ in range(min(n_diverse, len(remaining))):
                    if len(selected) == 0:
                        # Select random first case
                        idx = np.random.randint(len(remaining))
                    else:
                        # Select case most different from selected
                        selected_embeddings = embeddings[selected]
                        mean_selected = np.mean(selected_embeddings, axis=0)
                        
                        distances = np.linalg.norm(
                            remaining_embeddings - mean_selected, axis=1
                        )
                        idx = np.argmax(distances)
                    
                    selected.append(remaining_indices[idx])
                    
                    # Remove from remaining
                    remaining_embeddings = np.delete(remaining_embeddings, idx, axis=0)
                    remaining_indices = np.delete(remaining_indices, idx)
                    
                    if len(remaining_indices) == 0:
                        break
        
        return selected[:n]
    
    def get_selection_summary(self, stats: Dict) -> str:
        """Generate human-readable selection summary."""
        lines = [
            "Case Selection Summary",
            "=" * 40,
            f"Total cases selected: {stats.get('total', 0)}",
            "",
            "Breakdown:",
        ]
        
        if 'illicit' in stats:
            lines.append(
                f"  Illicit: {stats['illicit']} / {stats.get('illicit_available', '?')} available"
            )
        if 'licit' in stats:
            lines.append(
                f"  Licit: {stats['licit']} / {stats.get('licit_available', '?')} available"
            )
        if 'edge' in stats:
            lines.append(
                f"  Edge cases: {stats['edge']} / {stats.get('edge_available', '?')} available"
            )
        
        return "\n".join(lines)


def select_cases_for_memory(
    labels: np.ndarray,
    fraud_scores: np.ndarray,
    embeddings: np.ndarray,
    num_cases: int = 1000,
    strategy: str = 'balanced'
) -> Tuple[np.ndarray, Dict]:
    """
    Convenience function to select cases for memory.
    
    Args:
        labels: True labels
        fraud_scores: Predicted fraud probabilities
        embeddings: Node embeddings
        num_cases: Total number of cases to select
        strategy: Selection strategy
        
    Returns:
        Tuple of (selected_indices, stats)
    """
    # Calculate proportions
    n_illicit = int(num_cases * 0.35)
    n_licit = int(num_cases * 0.35)
    n_edge = num_cases - n_illicit - n_licit
    
    selector = CaseSelector(
        num_illicit=n_illicit,
        num_licit=n_licit,
        num_edge=n_edge
    )
    
    return selector.select(labels, fraud_scores, embeddings, strategy)


if __name__ == '__main__':
    # Test case selector
    print("Testing CaseSelector")
    print("="*50)
    
    # Create dummy data
    np.random.seed(42)
    n = 1000
    
    labels = np.random.choice([-1, 0, 1], size=n, p=[0.7, 0.2, 0.1])
    fraud_scores = np.random.random(n)
    embeddings = np.random.randn(n, 64)
    
    # Test selection
    selector = CaseSelector(num_illicit=20, num_licit=20, num_edge=10)
    
    for strategy in ['balanced', 'diverse', 'confident']:
        print(f"\nStrategy: {strategy}")
        indices, stats = selector.select(labels, fraud_scores, embeddings, strategy)
        print(selector.get_selection_summary(stats))
