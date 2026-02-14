"""
GNN Explainer for Fraud Detection

Provides interpretable explanations for GNN predictions using:
- GNNExplainer: Learns important subgraph and features
- Attention-based explanation (for GAT models)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer as PyGExplainer
from torch_geometric.utils import k_hop_subgraph
from dataclasses import dataclass
from datetime import datetime
import json


@dataclass
class ExplanationResult:
    """Container for explanation results."""
    node_idx: int
    fraud_score: float
    predicted_label: str
    true_label: Optional[str]
    
    # Subgraph info
    subgraph_nodes: List[int]
    subgraph_edges: List[Tuple[int, int]]
    edge_importance: List[float]
    
    # Feature importance
    feature_importance: List[float]
    top_features: List[Dict]
    
    # Narrative
    narrative: str
    timestamp: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'node_idx': self.node_idx,
            'fraud_score': self.fraud_score,
            'predicted_label': self.predicted_label,
            'true_label': self.true_label,
            'subgraph_nodes': self.subgraph_nodes,
            'subgraph_edges': self.subgraph_edges,
            'edge_importance': self.edge_importance,
            'feature_importance': self.feature_importance,
            'top_features': self.top_features,
            'narrative': self.narrative,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ExplanationResult':
        """Create from dictionary."""
        return cls(**data)


class FraudExplainer:
    """
    Explainer for fraud detection GNN models.
    
    Uses GNNExplainer to identify important subgraphs and features
    that contribute to fraud predictions.
    
    Args:
        model: Trained GNN model
        data: PyG Data object
        device: Device to use
        
    Usage:
        explainer = FraudExplainer(model, data, device='cuda')
        explanation = explainer.explain_node(node_idx=12345)
        print(explanation.narrative)
    """
    
    def __init__(
        self,
        model: nn.Module,
        data: Data,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.data = data.to(device)
        self.device = device
        
        # Setup explainer
        self.explainer = Explainer(
            model=model,
            algorithm=PyGExplainer(epochs=200),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(
                mode='multiclass_classification',
                task_level='node',
                return_type='raw'
            )
        )
        
        # Feature names (Elliptic doesn't provide names, so we use indices)
        self.feature_names = [f'feature_{i}' for i in range(data.x.size(1))]
        
    def explain_node(
        self,
        node_idx: int,
        num_hops: int = 2,
        top_k_features: int = 10,
        include_narrative: bool = True
    ) -> ExplanationResult:
        """
        Generate explanation for a single node prediction.
        
        Args:
            node_idx: Index of node to explain
            num_hops: Number of hops for subgraph extraction
            top_k_features: Number of top features to include
            include_narrative: Whether to generate narrative explanation
            
        Returns:
            ExplanationResult with all explanation details
        """
        self.model.eval()
        
        # Get prediction for this node
        with torch.no_grad():
            logits, _ = self.model(self.data.x, self.data.edge_index)
            probs = torch.softmax(logits, dim=1)
            fraud_score = probs[node_idx, 1].item()
            pred_label = 'illicit' if fraud_score > 0.5 else 'licit'
        
        # Get true label if available
        true_label_idx = self.data.y[node_idx].item()
        true_label = {-1: 'unknown', 0: 'licit', 1: 'illicit'}.get(true_label_idx, 'unknown')
        
        # Generate explanation
        explanation = self.explainer(
            self.data.x,
            self.data.edge_index,
            index=node_idx
        )
        
        # Extract subgraph
        subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx=node_idx,
            num_hops=num_hops,
            edge_index=self.data.edge_index,
            relabel_nodes=False,
            num_nodes=self.data.num_nodes
        )
        
        subgraph_nodes = subset.cpu().tolist()
        subgraph_edges = list(zip(
            sub_edge_index[0].cpu().tolist(),
            sub_edge_index[1].cpu().tolist()
        ))
        
        # Get edge importance from explanation
        if explanation.edge_mask is not None:
            # Map edge mask to subgraph edges
            full_edge_mask = explanation.edge_mask.cpu().numpy()
            edge_importance = full_edge_mask[edge_mask.cpu().numpy()].tolist()
        else:
            edge_importance = [0.5] * len(subgraph_edges)
        
        # Get feature importance
        if explanation.node_mask is not None:
            feature_importance = explanation.node_mask[node_idx].cpu().numpy().tolist()
        else:
            feature_importance = [0.0] * len(self.feature_names)
        
        # Get top features
        top_indices = np.argsort(feature_importance)[-top_k_features:][::-1]
        top_features = [
            {
                'index': int(idx),
                'name': self.feature_names[idx],
                'importance': feature_importance[idx],
                'value': self.data.x[node_idx, idx].item()
            }
            for idx in top_indices
        ]
        
        # Generate narrative
        if include_narrative:
            narrative = self._generate_narrative(
                node_idx=node_idx,
                fraud_score=fraud_score,
                pred_label=pred_label,
                top_features=top_features,
                subgraph_nodes=subgraph_nodes,
                edge_importance=edge_importance
            )
        else:
            narrative = ""
        
        return ExplanationResult(
            node_idx=node_idx,
            fraud_score=fraud_score,
            predicted_label=pred_label,
            true_label=true_label,
            subgraph_nodes=subgraph_nodes,
            subgraph_edges=subgraph_edges,
            edge_importance=edge_importance,
            feature_importance=feature_importance,
            top_features=top_features,
            narrative=narrative,
            timestamp=datetime.now().isoformat()
        )
    
    def _generate_narrative(
        self,
        node_idx: int,
        fraud_score: float,
        pred_label: str,
        top_features: List[Dict],
        subgraph_nodes: List[int],
        edge_importance: List[float]
    ) -> str:
        """Generate human-readable narrative explanation."""
        
        # Risk level
        if fraud_score > 0.9:
            risk_level = "very high"
        elif fraud_score > 0.7:
            risk_level = "high"
        elif fraud_score > 0.5:
            risk_level = "moderate"
        elif fraud_score > 0.3:
            risk_level = "low"
        else:
            risk_level = "very low"
        
        # Build narrative
        narrative_parts = [
            f"Transaction #{node_idx} Analysis",
            f"=" * 40,
            f"",
            f"Risk Assessment: {risk_level.upper()} ({fraud_score:.1%} fraud probability)",
            f"Classification: {pred_label.upper()}",
            f"",
            f"Key Factors:",
        ]
        
        # Top features
        for i, feat in enumerate(top_features[:5], 1):
            narrative_parts.append(
                f"  {i}. {feat['name']}: importance={feat['importance']:.3f}, value={feat['value']:.3f}"
            )
        
        # Network context
        num_neighbors = len(subgraph_nodes) - 1
        important_edges = sum(1 for imp in edge_importance if imp > 0.5)
        
        narrative_parts.extend([
            f"",
            f"Network Context:",
            f"  - Connected to {num_neighbors} transactions in 2-hop neighborhood",
            f"  - {important_edges} connections flagged as significant by the model",
        ])
        
        # Risk summary
        narrative_parts.extend([
            f"",
            f"Summary:",
            f"  This transaction shows {'suspicious' if fraud_score > 0.5 else 'normal'} patterns "
            f"based on its features and network connections."
        ])
        
        return "\n".join(narrative_parts)
    
    def explain_batch(
        self,
        node_indices: List[int],
        num_hops: int = 2,
        top_k_features: int = 10,
        verbose: bool = True
    ) -> List[ExplanationResult]:
        """
        Generate explanations for multiple nodes.
        
        Args:
            node_indices: List of node indices to explain
            num_hops: Number of hops for subgraph
            top_k_features: Number of top features
            verbose: Whether to show progress
            
        Returns:
            List of ExplanationResult objects
        """
        from tqdm import tqdm
        
        explanations = []
        iterator = tqdm(node_indices, desc="Generating explanations") if verbose else node_indices
        
        for node_idx in iterator:
            try:
                exp = self.explain_node(
                    node_idx=node_idx,
                    num_hops=num_hops,
                    top_k_features=top_k_features
                )
                explanations.append(exp)
            except Exception as e:
                if verbose:
                    print(f"Warning: Failed to explain node {node_idx}: {e}")
                continue
        
        return explanations
    
    def get_important_neighbors(
        self,
        node_idx: int,
        explanation: Optional[ExplanationResult] = None,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Get the most important neighboring nodes.
        
        Args:
            node_idx: Central node index
            explanation: Pre-computed explanation (optional)
            top_k: Number of top neighbors to return
            
        Returns:
            List of dicts with neighbor info
        """
        if explanation is None:
            explanation = self.explain_node(node_idx)
        
        # Get edges involving this node
        edge_index = self.data.edge_index.cpu().numpy()
        
        # Incoming edges
        incoming_mask = edge_index[1] == node_idx
        incoming_sources = edge_index[0][incoming_mask]
        
        # Outgoing edges
        outgoing_mask = edge_index[0] == node_idx
        outgoing_targets = edge_index[1][outgoing_mask]
        
        neighbors = []
        
        # Process incoming
        for src in incoming_sources:
            # Find edge in explanation
            edge_idx = None
            for i, (s, t) in enumerate(explanation.subgraph_edges):
                if s == src and t == node_idx:
                    edge_idx = i
                    break
            
            importance = explanation.edge_importance[edge_idx] if edge_idx is not None else 0.0
            
            neighbors.append({
                'node_idx': int(src),
                'direction': 'incoming',
                'importance': importance,
                'label': {-1: 'unknown', 0: 'licit', 1: 'illicit'}.get(
                    self.data.y[src].item(), 'unknown'
                )
            })
        
        # Process outgoing
        for tgt in outgoing_targets:
            edge_idx = None
            for i, (s, t) in enumerate(explanation.subgraph_edges):
                if s == node_idx and t == tgt:
                    edge_idx = i
                    break
            
            importance = explanation.edge_importance[edge_idx] if edge_idx is not None else 0.0
            
            neighbors.append({
                'node_idx': int(tgt),
                'direction': 'outgoing',
                'importance': importance,
                'label': {-1: 'unknown', 0: 'licit', 1: 'illicit'}.get(
                    self.data.y[tgt].item(), 'unknown'
                )
            })
        
        # Sort by importance and return top-k
        neighbors.sort(key=lambda x: x['importance'], reverse=True)
        return neighbors[:top_k]


def explain_prediction(
    model: nn.Module,
    data: Data,
    node_idx: int,
    device: str = 'cpu'
) -> ExplanationResult:
    """
    Convenience function to explain a single prediction.
    
    Args:
        model: Trained GNN model
        data: PyG Data object
        node_idx: Node to explain
        device: Device to use
        
    Returns:
        ExplanationResult
    """
    explainer = FraudExplainer(model, data, device)
    return explainer.explain_node(node_idx)


if __name__ == '__main__':
    # Test with dummy data
    print("Testing FraudExplainer")
    print("="*50)
    
    import sys
    sys.path.append('..')
    from models.graphsage import GraphSAGEModel
    
    # Create dummy data
    num_nodes = 100
    num_features = 166
    num_edges = 500
    
    x = torch.randn(num_nodes, num_features)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    y = torch.randint(-1, 2, (num_nodes,))
    
    data = Data(x=x, edge_index=edge_index, y=y)
    
    # Create model
    model = GraphSAGEModel(num_features=166, num_classes=2)
    
    # Test explainer
    explainer = FraudExplainer(model, data, device='cpu')
    
    # Explain a node
    result = explainer.explain_node(node_idx=0)
    
    print("\nExplanation Result:")
    print(f"Node: {result.node_idx}")
    print(f"Fraud Score: {result.fraud_score:.4f}")
    print(f"Predicted: {result.predicted_label}")
    print(f"Subgraph nodes: {len(result.subgraph_nodes)}")
    print(f"\nNarrative:\n{result.narrative}")
