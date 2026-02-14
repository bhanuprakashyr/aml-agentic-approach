"""
GraphSAGE Model for Fraud Detection

Implements GraphSAGE architecture for node classification on the Elliptic dataset.
GraphSAGE is chosen for its inductive learning capability - it can handle new,
unseen nodes at inference time by sampling and aggregating neighbor features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, BatchNorm
from typing import Tuple, Optional


class GraphSAGEModel(nn.Module):
    """
    GraphSAGE model for fraud detection.
    
    Architecture:
    - Input layer: num_features -> hidden_dims[0]
    - Hidden layers: hidden_dims[i] -> hidden_dims[i+1]
    - Embedding layer: hidden_dims[-1] -> embedding_dim
    - Classifier: embedding_dim -> num_classes
    
    The model returns both class logits (for classification) and
    node embeddings (for FAISS similarity search).
    
    Args:
        num_features: Number of input features per node
        num_classes: Number of output classes
        hidden_dims: List of hidden layer dimensions
        embedding_dim: Dimension of output embeddings (for FAISS)
        dropout: Dropout probability
        use_batch_norm: Whether to use batch normalization
        
    Usage:
        model = GraphSAGEModel(
            num_features=166,
            num_classes=2,
            hidden_dims=[256, 128],
            embedding_dim=64
        )
        logits, embeddings = model(x, edge_index)
    """
    
    def __init__(
        self,
        num_features: int,
        num_classes: int = 2,
        hidden_dims: list = [256, 128],
        embedding_dim: int = 64,
        dropout: float = 0.3,
        use_batch_norm: bool = True,
        aggregator: str = 'mean'
    ):
        super(GraphSAGEModel, self).__init__()
        
        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        
        # Build layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        
        # Input layer
        self.convs.append(SAGEConv(num_features, hidden_dims[0], aggr=aggregator))
        if use_batch_norm:
            self.batch_norms.append(BatchNorm(hidden_dims[0]))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            self.convs.append(SAGEConv(hidden_dims[i], hidden_dims[i+1], aggr=aggregator))
            if use_batch_norm:
                self.batch_norms.append(BatchNorm(hidden_dims[i+1]))
        
        # Embedding layer (no batch norm on this one for cleaner embeddings)
        self.embedding_layer = SAGEConv(hidden_dims[-1], embedding_dim, aggr=aggregator)
        
        # Classifier head
        self.classifier = nn.Linear(embedding_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_embeddings: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Node features (num_nodes, num_features)
            edge_index: Graph connectivity (2, num_edges)
            return_embeddings: Whether to return embeddings
            
        Returns:
            Tuple of (logits, embeddings)
            - logits: Class predictions (num_nodes, num_classes)
            - embeddings: Node embeddings (num_nodes, embedding_dim) or None
        """
        # Pass through convolutional layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Embedding layer
        embeddings = self.embedding_layer(x, edge_index)
        
        # Classification head
        logits = self.classifier(embeddings)
        
        if return_embeddings:
            return logits, embeddings
        else:
            return logits, None
    
    def get_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Get only the node embeddings (for FAISS indexing).
        
        Args:
            x: Node features (num_nodes, num_features)
            edge_index: Graph connectivity (2, num_edges)
            
        Returns:
            Node embeddings (num_nodes, embedding_dim)
        """
        self.eval()
        with torch.no_grad():
            _, embeddings = self.forward(x, edge_index, return_embeddings=True)
        return embeddings
    
    def predict_proba(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Get probability predictions.
        
        Args:
            x: Node features
            edge_index: Graph connectivity
            
        Returns:
            Probabilities (num_nodes, num_classes)
        """
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
        """
        Get class predictions.
        
        Args:
            x: Node features
            edge_index: Graph connectivity
            
        Returns:
            Predicted classes (num_nodes,)
        """
        probs = self.predict_proba(x, edge_index)
        return probs.argmax(dim=1)


class GraphSAGEWithSkip(nn.Module):
    """
    GraphSAGE with skip connections for better gradient flow.
    
    Same as GraphSAGEModel but adds residual connections between layers.
    Useful for deeper networks.
    """
    
    def __init__(
        self,
        num_features: int,
        num_classes: int = 2,
        hidden_dims: list = [256, 128],
        embedding_dim: int = 64,
        dropout: float = 0.3
    ):
        super(GraphSAGEWithSkip, self).__init__()
        
        self.num_features = num_features
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        
        # Input projection (for skip connection)
        self.input_proj = nn.Linear(num_features, hidden_dims[0])
        
        # Convolutional layers
        self.conv1 = SAGEConv(num_features, hidden_dims[0])
        self.conv2 = SAGEConv(hidden_dims[0], hidden_dims[1] if len(hidden_dims) > 1 else hidden_dims[0])
        self.conv3 = SAGEConv(hidden_dims[-1], embedding_dim)
        
        # Skip connection projections
        if len(hidden_dims) > 1:
            self.skip_proj = nn.Linear(hidden_dims[0], hidden_dims[1])
        else:
            self.skip_proj = None
            
        # Batch norms
        self.bn1 = BatchNorm(hidden_dims[0])
        self.bn2 = BatchNorm(hidden_dims[1] if len(hidden_dims) > 1 else hidden_dims[0])
        
        # Classifier
        self.classifier = nn.Linear(embedding_dim, num_classes)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_embeddings: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with skip connections."""
        
        # First layer
        identity = self.input_proj(x)
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x + identity)  # Skip connection
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second layer
        if self.skip_proj is not None:
            identity = self.skip_proj(x)
            x = self.conv2(x, edge_index)
            x = self.bn2(x)
            x = F.relu(x + identity)  # Skip connection
        else:
            x = self.conv2(x, edge_index)
            x = self.bn2(x)
            x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Embedding layer
        embeddings = self.conv3(x, edge_index)
        
        # Classifier
        logits = self.classifier(embeddings)
        
        if return_embeddings:
            return logits, embeddings
        return logits, None


def create_graphsage_model(
    num_features: int,
    num_classes: int = 2,
    model_size: str = 'medium',
    dropout: float = 0.3,
    use_skip: bool = False
) -> nn.Module:
    """
    Factory function to create GraphSAGE models.
    
    Args:
        num_features: Number of input features
        num_classes: Number of output classes
        model_size: One of 'small', 'medium', 'large'
        dropout: Dropout probability
        use_skip: Whether to use skip connections
        
    Returns:
        GraphSAGE model instance
    """
    configs = {
        'small': {'hidden_dims': [128, 64], 'embedding_dim': 32},
        'medium': {'hidden_dims': [256, 128], 'embedding_dim': 64},
        'large': {'hidden_dims': [512, 256, 128], 'embedding_dim': 64}
    }
    
    if model_size not in configs:
        raise ValueError(f"model_size must be one of {list(configs.keys())}")
    
    config = configs[model_size]
    
    ModelClass = GraphSAGEWithSkip if use_skip else GraphSAGEModel
    
    return ModelClass(
        num_features=num_features,
        num_classes=num_classes,
        hidden_dims=config['hidden_dims'],
        embedding_dim=config['embedding_dim'],
        dropout=dropout
    )


if __name__ == '__main__':
    # Test the model
    print("Testing GraphSAGE Model")
    print("="*50)
    
    # Create dummy data
    num_nodes = 1000
    num_features = 166
    num_edges = 5000
    
    x = torch.randn(num_nodes, num_features)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # Test standard model
    model = GraphSAGEModel(
        num_features=num_features,
        num_classes=2,
        hidden_dims=[256, 128],
        embedding_dim=64
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward pass
    model.train()
    logits, embeddings = model(x, edge_index)
    
    print(f"Input shape: {x.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Test factory function
    print("\nTesting factory function")
    for size in ['small', 'medium', 'large']:
        m = create_graphsage_model(num_features, model_size=size)
        params = sum(p.numel() for p in m.parameters())
        print(f"  {size}: {params:,} parameters")
