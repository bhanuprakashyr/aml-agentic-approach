"""
Graph Attention Network (GAT) Model for Fraud Detection

Implements GAT architecture with multi-head attention for node classification.
GAT learns to weight neighbor importance via attention mechanism, which can
provide better interpretability compared to GraphSAGE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, BatchNorm
from typing import Tuple, Optional


class GATModel(nn.Module):
    """
    Graph Attention Network for fraud detection.
    
    Architecture:
    - Multi-head attention layers with configurable heads
    - Each layer learns attention weights over neighbors
    - Returns both predictions and embeddings
    
    The attention weights can be extracted for interpretability,
    showing which neighbors influenced each prediction.
    
    Args:
        num_features: Number of input features per node
        num_classes: Number of output classes
        hidden_dims: List of hidden layer dimensions
        embedding_dim: Dimension of output embeddings
        heads: Number of attention heads per layer
        dropout: Dropout probability
        attention_dropout: Dropout on attention weights
        
    Usage:
        model = GATModel(
            num_features=166,
            num_classes=2,
            hidden_dims=[256, 128],
            embedding_dim=64,
            heads=[4, 4]
        )
        logits, embeddings = model(x, edge_index)
    """
    
    def __init__(
        self,
        num_features: int,
        num_classes: int = 2,
        hidden_dims: list = [256, 128],
        embedding_dim: int = 64,
        heads: list = [4, 4],
        dropout: float = 0.3,
        attention_dropout: float = 0.3,
        use_batch_norm: bool = True
    ):
        super(GATModel, self).__init__()
        
        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.embedding_dim = embedding_dim
        self.heads = heads if len(heads) == len(hidden_dims) else [heads[0]] * len(hidden_dims)
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.use_batch_norm = use_batch_norm
        
        # Build layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        
        # Input layer
        # Output dim = hidden_dims[0] * heads[0] (concatenation of attention heads)
        self.convs.append(GATConv(
            num_features,
            hidden_dims[0],
            heads=self.heads[0],
            dropout=attention_dropout,
            concat=True  # Concatenate attention heads
        ))
        if use_batch_norm:
            self.batch_norms.append(BatchNorm(hidden_dims[0] * self.heads[0]))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            # Input dim includes all attention heads from previous layer
            in_dim = hidden_dims[i] * self.heads[i]
            self.convs.append(GATConv(
                in_dim,
                hidden_dims[i+1],
                heads=self.heads[i+1] if i+1 < len(self.heads) else 1,
                dropout=attention_dropout,
                concat=True
            ))
            if use_batch_norm:
                out_dim = hidden_dims[i+1] * (self.heads[i+1] if i+1 < len(self.heads) else 1)
                self.batch_norms.append(BatchNorm(out_dim))
        
        # Embedding layer (single head, no concatenation)
        last_dim = hidden_dims[-1] * self.heads[-1] if len(self.heads) == len(hidden_dims) else hidden_dims[-1]
        self.embedding_layer = GATConv(
            last_dim,
            embedding_dim,
            heads=1,
            dropout=attention_dropout,
            concat=False  # Average attention heads
        )
        
        # Classifier head
        self.classifier = nn.Linear(embedding_dim, num_classes)
        
        # For storing attention weights
        self._attention_weights = None
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_embeddings: bool = True,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Node features (num_nodes, num_features)
            edge_index: Graph connectivity (2, num_edges)
            return_embeddings: Whether to return embeddings
            return_attention: Whether to store attention weights
            
        Returns:
            Tuple of (logits, embeddings)
        """
        attention_weights = [] if return_attention else None
        
        # Pass through convolutional layers
        for i, conv in enumerate(self.convs):
            if return_attention:
                x, (edge_index_out, alpha) = conv(x, edge_index, return_attention_weights=True)
                attention_weights.append(alpha)
            else:
                x = conv(x, edge_index)
                
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            x = F.elu(x)  # ELU often works better with GAT
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Embedding layer
        if return_attention:
            embeddings, (_, alpha) = self.embedding_layer(x, edge_index, return_attention_weights=True)
            attention_weights.append(alpha)
            self._attention_weights = attention_weights
        else:
            embeddings = self.embedding_layer(x, edge_index)
        
        # Classification head
        logits = self.classifier(embeddings)
        
        if return_embeddings:
            return logits, embeddings
        else:
            return logits, None
    
    def get_attention_weights(self) -> Optional[list]:
        """
        Get attention weights from the last forward pass.
        
        Must call forward() with return_attention=True first.
        
        Returns:
            List of attention weight tensors for each layer
        """
        return self._attention_weights
    
    def get_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Get only the node embeddings (for FAISS indexing)."""
        self.eval()
        with torch.no_grad():
            _, embeddings = self.forward(x, edge_index, return_embeddings=True)
        return embeddings
    
    def predict_proba(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Get probability predictions."""
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(x, edge_index, return_embeddings=False)
            probs = F.softmax(logits, dim=1)
        return probs
    
    def predict(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Get class predictions."""
        probs = self.predict_proba(x, edge_index)
        return probs.argmax(dim=1)
    
    def get_neighbor_attention(
        self,
        node_idx: int,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> dict:
        """
        Get attention weights for a specific node's neighbors.
        
        Useful for interpretability - shows which neighbors the model
        pays attention to when making predictions.
        
        Args:
            node_idx: Index of the node to analyze
            x: Node features
            edge_index: Graph connectivity
            
        Returns:
            Dict with neighbor indices and their attention weights
        """
        self.eval()
        with torch.no_grad():
            _ = self.forward(x, edge_index, return_attention=True)
        
        if self._attention_weights is None:
            raise ValueError("No attention weights available. Run forward with return_attention=True")
        
        # Get edges involving this node
        mask = edge_index[1] == node_idx
        neighbor_indices = edge_index[0][mask].cpu().numpy()
        
        # Get attention weights for last layer
        last_attention = self._attention_weights[-1]
        neighbor_attention = last_attention[mask].cpu().numpy()
        
        return {
            'node_idx': node_idx,
            'neighbors': neighbor_indices.tolist(),
            'attention_weights': neighbor_attention.tolist()
        }


class GATv2Model(nn.Module):
    """
    GATv2 - Improved Graph Attention Network.
    
    Uses GATv2Conv which fixes the static attention problem in the original GAT.
    In GATv2, attention is dynamic and depends on both source and target features.
    """
    
    def __init__(
        self,
        num_features: int,
        num_classes: int = 2,
        hidden_dims: list = [256, 128],
        embedding_dim: int = 64,
        heads: int = 4,
        dropout: float = 0.3
    ):
        super(GATv2Model, self).__init__()
        
        from torch_geometric.nn import GATv2Conv
        
        self.conv1 = GATv2Conv(num_features, hidden_dims[0], heads=heads, dropout=dropout, concat=True)
        self.bn1 = BatchNorm(hidden_dims[0] * heads)
        
        self.conv2 = GATv2Conv(hidden_dims[0] * heads, hidden_dims[1], heads=heads, dropout=dropout, concat=True)
        self.bn2 = BatchNorm(hidden_dims[1] * heads)
        
        self.conv3 = GATv2Conv(hidden_dims[1] * heads, embedding_dim, heads=1, dropout=dropout, concat=False)
        
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.dropout = dropout
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_embeddings: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass."""
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        embeddings = self.conv3(x, edge_index)
        logits = self.classifier(embeddings)
        
        if return_embeddings:
            return logits, embeddings
        return logits, None


def create_gat_model(
    num_features: int,
    num_classes: int = 2,
    model_size: str = 'medium',
    heads: int = 4,
    dropout: float = 0.3,
    use_v2: bool = False
) -> nn.Module:
    """
    Factory function to create GAT models.
    
    Args:
        num_features: Number of input features
        num_classes: Number of output classes
        model_size: One of 'small', 'medium', 'large'
        heads: Number of attention heads
        dropout: Dropout probability
        use_v2: Whether to use GATv2 (improved version)
        
    Returns:
        GAT model instance
    """
    configs = {
        'small': {'hidden_dims': [128, 64], 'embedding_dim': 32},
        'medium': {'hidden_dims': [256, 128], 'embedding_dim': 64},
        'large': {'hidden_dims': [512, 256], 'embedding_dim': 64}
    }
    
    if model_size not in configs:
        raise ValueError(f"model_size must be one of {list(configs.keys())}")
    
    config = configs[model_size]
    
    if use_v2:
        return GATv2Model(
            num_features=num_features,
            num_classes=num_classes,
            hidden_dims=config['hidden_dims'],
            embedding_dim=config['embedding_dim'],
            heads=heads,
            dropout=dropout
        )
    else:
        return GATModel(
            num_features=num_features,
            num_classes=num_classes,
            hidden_dims=config['hidden_dims'],
            embedding_dim=config['embedding_dim'],
            heads=[heads] * len(config['hidden_dims']),
            dropout=dropout
        )


if __name__ == '__main__':
    # Test the model
    print("Testing GAT Model")
    print("="*50)
    
    # Create dummy data
    num_nodes = 1000
    num_features = 166
    num_edges = 5000
    
    x = torch.randn(num_nodes, num_features)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # Test standard model
    model = GATModel(
        num_features=num_features,
        num_classes=2,
        hidden_dims=[256, 128],
        embedding_dim=64,
        heads=[4, 4]
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward pass
    model.train()
    logits, embeddings = model(x, edge_index)
    
    print(f"Input shape: {x.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Test attention weights
    print("\nTesting attention extraction")
    model.eval()
    _, _ = model(x, edge_index, return_attention=True)
    attention = model.get_attention_weights()
    print(f"Number of attention layers: {len(attention)}")
    for i, att in enumerate(attention):
        print(f"  Layer {i}: {att.shape}")
