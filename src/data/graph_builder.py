"""
Graph Builder for PyTorch Geometric

Converts raw Elliptic data into PyG Data objects for GNN training.
"""

import torch
import numpy as np
from typing import Optional, Dict, Tuple
from torch_geometric.data import Data
from .elliptic_loader import EllipticDataLoader


class GraphBuilder:
    """
    Builds PyTorch Geometric Data objects from Elliptic dataset.
    
    Creates graph data structures suitable for GNN training, including:
    - Node features tensor
    - Edge index tensor (COO format)
    - Labels tensor with mask for known labels
    - Train/val/test masks based on temporal splits
    
    Usage:
        builder = GraphBuilder(data_dir='./data/elliptic_bitcoin_dataset')
        data = builder.build()  # Returns PyG Data object
        
        # Or with custom loader
        loader = EllipticDataLoader(data_dir)
        loader.load()
        builder = GraphBuilder.from_loader(loader)
        data = builder.build()
    """
    
    def __init__(self, data_dir: str = './data/elliptic_bitcoin_dataset'):
        """
        Initialize graph builder.
        
        Args:
            data_dir: Path to Elliptic dataset directory
        """
        self.loader = EllipticDataLoader(data_dir)
        self._data_loaded = False
        
    @classmethod
    def from_loader(cls, loader: EllipticDataLoader) -> 'GraphBuilder':
        """
        Create GraphBuilder from an existing loader.
        
        Args:
            loader: Initialized EllipticDataLoader with data loaded
            
        Returns:
            GraphBuilder instance
        """
        builder = cls.__new__(cls)
        builder.loader = loader
        builder._data_loaded = True
        return builder
    
    def build(
        self,
        normalize: bool = True,
        add_self_loops: bool = False,
        to_undirected: bool = False
    ) -> Data:
        """
        Build PyG Data object from Elliptic dataset.
        
        Args:
            normalize: Whether to normalize features
            add_self_loops: Whether to add self-loops to the graph
            to_undirected: Whether to convert directed edges to undirected
            
        Returns:
            PyG Data object with:
            - x: Node features (num_nodes, num_features)
            - edge_index: Graph connectivity (2, num_edges)
            - y: Node labels (num_nodes,) with -1 for unknown
            - train_mask: Boolean mask for training nodes
            - val_mask: Boolean mask for validation nodes
            - test_mask: Boolean mask for test nodes
            - labeled_mask: Boolean mask for all labeled nodes
            - time_steps: Time step for each node
        """
        # Load data if not already loaded
        if not self._data_loaded:
            features, labels, edge_index = self.loader.load(normalize=normalize)
            self._data_loaded = True
        else:
            features = self.loader._features
            labels = self.loader._labels
            edge_index = self.loader._edge_index
            
        # Convert to tensors
        x = torch.tensor(features, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.long)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        time_steps = torch.tensor(self.loader._time_steps, dtype=torch.long)
        
        # Add self-loops if requested
        if add_self_loops:
            num_nodes = x.size(0)
            self_loops = torch.arange(num_nodes, dtype=torch.long).unsqueeze(0).repeat(2, 1)
            edge_index = torch.cat([edge_index, self_loops], dim=1)
            
        # Convert to undirected if requested
        if to_undirected:
            # Add reverse edges
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            # Remove duplicates
            edge_index = torch.unique(edge_index, dim=1)
        
        # Get temporal splits
        train_split, val_split, test_split = self.loader.get_temporal_splits()
        
        # Create masks
        train_mask = torch.tensor(train_split['node_mask'], dtype=torch.bool)
        val_mask = torch.tensor(val_split['node_mask'], dtype=torch.bool)
        test_mask = torch.tensor(test_split['node_mask'], dtype=torch.bool)
        labeled_mask = y != -1
        
        # For training, we only use labeled nodes in the training time period
        train_mask = train_mask & labeled_mask
        val_mask = val_mask & labeled_mask
        test_mask = test_mask & labeled_mask
        
        # Get class weights
        class_weights = torch.tensor(self.loader.get_class_weights(), dtype=torch.float32)
        
        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            labeled_mask=labeled_mask,
            time_steps=time_steps,
            class_weights=class_weights
        )
        
        # Store additional metadata
        data.num_classes = 2
        data.num_features = x.size(1)
        
        print("\n" + "="*50)
        print("PyG Data Object Created")
        print("="*50)
        print(f"Nodes: {data.num_nodes:,}")
        print(f"Edges: {data.num_edges:,}")
        print(f"Features: {data.num_features}")
        print(f"Train nodes: {train_mask.sum():,}")
        print(f"Val nodes: {val_mask.sum():,}")
        print(f"Test nodes: {test_mask.sum():,}")
        
        return data
    
    def build_temporal_snapshots(
        self,
        normalize: bool = True
    ) -> Dict[int, Data]:
        """
        Build separate PyG Data objects for each time step.
        
        Useful for temporal GNN training where each time step is
        processed separately.
        
        Args:
            normalize: Whether to normalize features
            
        Returns:
            Dictionary mapping time_step -> PyG Data object
        """
        # Load data if not already loaded
        if not self._data_loaded:
            features, labels, edge_index = self.loader.load(normalize=normalize)
            self._data_loaded = True
        else:
            features = self.loader._features
            labels = self.loader._labels
            edge_index = self.loader._edge_index
            
        time_steps = self.loader._time_steps
        unique_times = sorted(np.unique(time_steps))
        
        snapshots = {}
        
        for t in unique_times:
            # Get nodes in this time step
            node_mask = time_steps == t
            node_indices = np.where(node_mask)[0]
            
            # Create local index mapping
            global_to_local = {g: l for l, g in enumerate(node_indices)}
            
            # Get features and labels for this time step
            x_t = torch.tensor(features[node_mask], dtype=torch.float32)
            y_t = torch.tensor(labels[node_mask], dtype=torch.long)
            
            # Get edges within this time step
            src_in_t = np.isin(edge_index[0], node_indices)
            dst_in_t = np.isin(edge_index[1], node_indices)
            edge_mask = src_in_t & dst_in_t
            
            if edge_mask.sum() > 0:
                edges_t = edge_index[:, edge_mask]
                # Remap to local indices
                edges_local = np.array([
                    [global_to_local[e] for e in edges_t[0]],
                    [global_to_local[e] for e in edges_t[1]]
                ])
                edge_index_t = torch.tensor(edges_local, dtype=torch.long)
            else:
                edge_index_t = torch.zeros((2, 0), dtype=torch.long)
            
            # Create Data object for this time step
            data_t = Data(
                x=x_t,
                edge_index=edge_index_t,
                y=y_t,
                labeled_mask=(y_t != -1),
                time_step=t,
                global_indices=torch.tensor(node_indices, dtype=torch.long)
            )
            
            snapshots[t] = data_t
            
        print(f"\nCreated {len(snapshots)} temporal snapshots")
        print(f"Time steps: {min(unique_times)} to {max(unique_times)}")
        
        return snapshots
    
    def get_subgraph(
        self,
        node_idx: int,
        num_hops: int = 2,
        data: Optional[Data] = None
    ) -> Tuple[Data, Dict[int, int]]:
        """
        Extract k-hop subgraph around a node.
        
        Useful for GNNExplainer and visualization.
        
        Args:
            node_idx: Central node index
            num_hops: Number of hops to include
            data: PyG Data object (will build if not provided)
            
        Returns:
            Tuple of (subgraph_data, node_mapping)
            - subgraph_data: PyG Data object for subgraph
            - node_mapping: Dict mapping subgraph indices to original indices
        """
        from torch_geometric.utils import k_hop_subgraph
        
        if data is None:
            data = self.build()
            
        # Get k-hop subgraph
        subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx=node_idx,
            num_hops=num_hops,
            edge_index=data.edge_index,
            relabel_nodes=True,
            num_nodes=data.num_nodes
        )
        
        # Create subgraph Data object
        subgraph = Data(
            x=data.x[subset],
            edge_index=sub_edge_index,
            y=data.y[subset],
            center_idx=mapping,  # Index of the center node in subgraph
            original_indices=subset
        )
        
        # Create mapping from subgraph to original indices
        node_mapping = {i: int(subset[i]) for i in range(len(subset))}
        
        return subgraph, node_mapping


def build_pyg_data(
    data_dir: str = './data/elliptic_bitcoin_dataset',
    normalize: bool = True,
    add_self_loops: bool = False,
    to_undirected: bool = False
) -> Data:
    """
    Convenience function to build PyG Data object.
    
    Args:
        data_dir: Path to dataset directory
        normalize: Whether to normalize features
        add_self_loops: Whether to add self-loops
        to_undirected: Whether to make graph undirected
        
    Returns:
        PyG Data object ready for training
    """
    builder = GraphBuilder(data_dir)
    return builder.build(
        normalize=normalize,
        add_self_loops=add_self_loops,
        to_undirected=to_undirected
    )


if __name__ == '__main__':
    # Test the builder
    builder = GraphBuilder('../data/elliptic_bitcoin_dataset')
    data = builder.build()
    
    print("\n" + "="*50)
    print("Testing subgraph extraction")
    print("="*50)
    
    # Test subgraph extraction for a random labeled node
    labeled_indices = torch.where(data.labeled_mask)[0]
    test_node = int(labeled_indices[0])
    subgraph, mapping = builder.get_subgraph(test_node, num_hops=2, data=data)
    
    print(f"Subgraph for node {test_node}:")
    print(f"  Nodes: {subgraph.num_nodes}")
    print(f"  Edges: {subgraph.num_edges}")
    print(f"  Center node index in subgraph: {subgraph.center_idx}")
