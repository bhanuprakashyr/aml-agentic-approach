"""
GNN model architectures for fraud detection.
"""

from .graphsage import GraphSAGEModel
from .gat import GATModel

__all__ = ['GraphSAGEModel', 'GATModel']
