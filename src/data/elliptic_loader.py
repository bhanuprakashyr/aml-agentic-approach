"""
Elliptic Dataset Loader

Loads and preprocesses the Elliptic Bitcoin dataset:
- Features: 166 features per transaction (time step + 165 attributes)
- Labels: illicit, licit, unknown
- Edges: Directed payment flows between transactions
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


class EllipticDataLoader:
    """
    Loads the Elliptic Bitcoin dataset and provides preprocessed data.
    
    The Elliptic dataset contains:
    - 203,769 Bitcoin transactions (nodes)
    - 234,355 payment flows (directed edges)
    - 166 features per node (1 time step + 93 local + 72 aggregated)
    - Labels: illicit (4,545), licit (42,019), unknown (157,205)
    
    Usage:
        loader = EllipticDataLoader(data_dir='./data/elliptic_bitcoin_dataset')
        features, labels, edges = loader.load()
        train_data, val_data, test_data = loader.get_temporal_splits()
    """
    
    # Class label mapping
    LABEL_MAP = {
        '1': 'illicit',
        '2': 'licit', 
        'unknown': 'unknown'
    }
    
    # Numeric label encoding for model
    LABEL_ENCODING = {
        'illicit': 1,
        'licit': 0,
        'unknown': -1  # Will be masked during training
    }
    
    def __init__(self, data_dir: str = './data/elliptic_bitcoin_dataset'):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Path to the directory containing Elliptic CSV files
        """
        self.data_dir = Path(data_dir)
        self._validate_data_dir()
        
        # Data containers (loaded lazily)
        self._features_df: Optional[pd.DataFrame] = None
        self._labels_df: Optional[pd.DataFrame] = None
        self._edges_df: Optional[pd.DataFrame] = None
        
        # Processed data
        self._node_id_map: Optional[Dict[int, int]] = None
        self._features: Optional[np.ndarray] = None
        self._labels: Optional[np.ndarray] = None
        self._time_steps: Optional[np.ndarray] = None
        self._edge_index: Optional[np.ndarray] = None
        
        # Scaler for feature normalization
        self._scaler: Optional[StandardScaler] = None
        
    def _validate_data_dir(self):
        """Validate that all required files exist."""
        required_files = [
            'elliptic_txs_features.csv',
            'elliptic_txs_classes.csv',
            'elliptic_txs_edgelist.csv'
        ]
        for fname in required_files:
            fpath = self.data_dir / fname
            if not fpath.exists():
                raise FileNotFoundError(
                    f"Required file not found: {fpath}\n"
                    f"Please download the Elliptic dataset from Kaggle and place it in {self.data_dir}"
                )
    
    def load(self, normalize: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and preprocess all data.
        
        Args:
            normalize: Whether to normalize features using StandardScaler
            
        Returns:
            Tuple of (features, labels, edge_index)
            - features: (num_nodes, 166) array of node features
            - labels: (num_nodes,) array of labels (-1 for unknown, 0 for licit, 1 for illicit)
            - edge_index: (2, num_edges) array of edges in COO format
        """
        self._load_raw_data()
        self._build_node_mapping()
        self._process_features(normalize=normalize)
        self._process_labels()
        self._process_edges()
        
        return self._features, self._labels, self._edge_index
    
    def _load_raw_data(self):
        """Load raw CSV files into DataFrames."""
        print("Loading Elliptic dataset...")
        
        # Load features (no header - column 0 is node ID, column 1 is time step)
        self._features_df = pd.read_csv(
            self.data_dir / 'elliptic_txs_features.csv',
            header=None
        )
        print(f"  Features: {self._features_df.shape[0]:,} nodes, {self._features_df.shape[1]} columns")
        
        # Load labels
        self._labels_df = pd.read_csv(
            self.data_dir / 'elliptic_txs_classes.csv'
        )
        print(f"  Labels: {self._labels_df.shape[0]:,} entries")
        
        # Load edges
        self._edges_df = pd.read_csv(
            self.data_dir / 'elliptic_txs_edgelist.csv',
            header=None
        )
        print(f"  Edges: {self._edges_df.shape[0]:,} directed edges")
        
    def _build_node_mapping(self):
        """Build mapping from original node IDs to contiguous indices."""
        # Original node IDs from features (column 0)
        original_ids = self._features_df.iloc[:, 0].values
        
        # Create mapping: original_id -> contiguous_index
        self._node_id_map = {int(orig_id): idx for idx, orig_id in enumerate(original_ids)}
        self._reverse_node_map = {idx: orig_id for orig_id, idx in self._node_id_map.items()}
        
        print(f"  Node mapping: {len(self._node_id_map):,} unique nodes")
        
    def _process_features(self, normalize: bool = True):
        """Extract and optionally normalize features."""
        # Column 0 = node ID, Column 1 = time step, Columns 2-166 = features
        # But we keep time step as a feature (total 166 features including time)
        
        # Extract time steps (column 1)
        self._time_steps = self._features_df.iloc[:, 1].values.astype(np.int32)
        
        # Extract features (columns 1 onwards, including time step)
        self._features = self._features_df.iloc[:, 1:].values.astype(np.float32)
        
        if normalize:
            print("  Normalizing features...")
            self._scaler = StandardScaler()
            self._features = self._scaler.fit_transform(self._features).astype(np.float32)
            
        print(f"  Features shape: {self._features.shape}")
        print(f"  Time steps: {self._time_steps.min()} to {self._time_steps.max()} ({len(np.unique(self._time_steps))} unique)")
        
    def _process_labels(self):
        """Process labels and encode them numerically."""
        # Create label array initialized with -1 (unknown)
        num_nodes = len(self._node_id_map)
        self._labels = np.full(num_nodes, -1, dtype=np.int64)
        
        # Map labels to nodes
        for _, row in self._labels_df.iterrows():
            node_id = int(row['txId'])
            label_str = str(row['class'])
            
            if node_id in self._node_id_map:
                idx = self._node_id_map[node_id]
                label_name = self.LABEL_MAP.get(label_str, 'unknown')
                self._labels[idx] = self.LABEL_ENCODING[label_name]
        
        # Print label distribution
        unique, counts = np.unique(self._labels, return_counts=True)
        print("  Label distribution:")
        label_names = {-1: 'unknown', 0: 'licit', 1: 'illicit'}
        for label, count in zip(unique, counts):
            print(f"    {label_names[label]}: {count:,} ({count/num_nodes*100:.2f}%)")
            
    def _process_edges(self):
        """Process edges and convert to COO format with mapped indices."""
        edges_list = []
        skipped = 0
        
        for _, row in self._edges_df.iterrows():
            src = int(row[0])
            dst = int(row[1])
            
            # Map to contiguous indices
            if src in self._node_id_map and dst in self._node_id_map:
                edges_list.append([
                    self._node_id_map[src],
                    self._node_id_map[dst]
                ])
            else:
                skipped += 1
                
        self._edge_index = np.array(edges_list, dtype=np.int64).T
        print(f"  Edge index shape: {self._edge_index.shape}")
        if skipped > 0:
            print(f"  Warning: Skipped {skipped} edges with unknown nodes")
            
    def get_temporal_splits(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ) -> Tuple[Dict, Dict, Dict]:
        """
        Split data temporally by time steps.
        
        The Elliptic dataset has 49 time steps (2 weeks each).
        Default split:
        - Train: Time steps 1-34 (~70%)
        - Val: Time steps 35-42 (~15%)
        - Test: Time steps 43-49 (~15%)
        
        Args:
            train_ratio: Fraction of time steps for training
            val_ratio: Fraction of time steps for validation
            
        Returns:
            Tuple of (train_data, val_data, test_data) dicts containing:
            - 'node_mask': Boolean mask for nodes in this split
            - 'time_steps': List of time steps in this split
            - 'num_nodes': Number of nodes in this split
            - 'num_labeled': Number of labeled nodes in this split
        """
        if self._time_steps is None:
            raise ValueError("Data not loaded. Call load() first.")
            
        unique_times = sorted(np.unique(self._time_steps))
        num_times = len(unique_times)
        
        # Calculate split points
        train_end = int(num_times * train_ratio)
        val_end = int(num_times * (train_ratio + val_ratio))
        
        train_times = unique_times[:train_end]
        val_times = unique_times[train_end:val_end]
        test_times = unique_times[val_end:]
        
        def create_split_data(time_list):
            mask = np.isin(self._time_steps, time_list)
            labeled_mask = mask & (self._labels != -1)
            return {
                'node_mask': mask,
                'time_steps': list(time_list),
                'num_nodes': mask.sum(),
                'num_labeled': labeled_mask.sum(),
                'num_illicit': ((self._labels == 1) & mask).sum(),
                'num_licit': ((self._labels == 0) & mask).sum()
            }
        
        train_data = create_split_data(train_times)
        val_data = create_split_data(val_times)
        test_data = create_split_data(test_times)
        
        print("\nTemporal splits:")
        print(f"  Train: {len(train_times)} time steps, {train_data['num_nodes']:,} nodes "
              f"({train_data['num_labeled']:,} labeled)")
        print(f"  Val:   {len(val_times)} time steps, {val_data['num_nodes']:,} nodes "
              f"({val_data['num_labeled']:,} labeled)")
        print(f"  Test:  {len(test_times)} time steps, {test_data['num_nodes']:,} nodes "
              f"({test_data['num_labeled']:,} labeled)")
        
        return train_data, val_data, test_data
    
    def get_class_weights(self) -> np.ndarray:
        """
        Calculate class weights for handling imbalanced data.
        
        Returns:
            Array of weights [weight_licit, weight_illicit]
        """
        if self._labels is None:
            raise ValueError("Data not loaded. Call load() first.")
            
        # Only consider labeled nodes
        labeled_mask = self._labels != -1
        labels_subset = self._labels[labeled_mask]
        
        # Count classes
        n_licit = (labels_subset == 0).sum()
        n_illicit = (labels_subset == 1).sum()
        n_total = n_licit + n_illicit
        
        # Inverse frequency weighting
        weight_licit = n_total / (2 * n_licit)
        weight_illicit = n_total / (2 * n_illicit)
        
        weights = np.array([weight_licit, weight_illicit], dtype=np.float32)
        print(f"\nClass weights: licit={weight_licit:.4f}, illicit={weight_illicit:.4f}")
        
        return weights
    
    @property
    def num_nodes(self) -> int:
        """Number of nodes in the graph."""
        return len(self._node_id_map) if self._node_id_map else 0
    
    @property
    def num_edges(self) -> int:
        """Number of edges in the graph."""
        return self._edge_index.shape[1] if self._edge_index is not None else 0
    
    @property
    def num_features(self) -> int:
        """Number of features per node."""
        return self._features.shape[1] if self._features is not None else 0
    
    @property
    def num_classes(self) -> int:
        """Number of classes (excluding unknown)."""
        return 2  # illicit, licit
    
    def get_node_info(self, node_idx: int) -> Dict:
        """
        Get detailed information about a specific node.
        
        Args:
            node_idx: Contiguous node index
            
        Returns:
            Dictionary with node information
        """
        if self._features is None:
            raise ValueError("Data not loaded. Call load() first.")
            
        label_names = {-1: 'unknown', 0: 'licit', 1: 'illicit'}
        original_id = self._reverse_node_map.get(node_idx, 'unknown')
        
        return {
            'node_idx': node_idx,
            'original_id': original_id,
            'time_step': int(self._time_steps[node_idx]),
            'label': label_names[self._labels[node_idx]],
            'label_encoded': int(self._labels[node_idx]),
            'features': self._features[node_idx]
        }


# Convenience function for quick loading
def load_elliptic_data(data_dir: str = './data/elliptic_bitcoin_dataset', normalize: bool = True):
    """
    Quick function to load Elliptic dataset.
    
    Args:
        data_dir: Path to dataset directory
        normalize: Whether to normalize features
        
    Returns:
        Tuple of (loader, features, labels, edge_index)
    """
    loader = EllipticDataLoader(data_dir)
    features, labels, edge_index = loader.load(normalize=normalize)
    return loader, features, labels, edge_index


if __name__ == '__main__':
    # Test the loader
    loader = EllipticDataLoader('../data/elliptic_bitcoin_dataset')
    features, labels, edge_index = loader.load()
    train_data, val_data, test_data = loader.get_temporal_splits()
    weights = loader.get_class_weights()
    
    print("\n" + "="*50)
    print("Dataset Summary")
    print("="*50)
    print(f"Nodes: {loader.num_nodes:,}")
    print(f"Edges: {loader.num_edges:,}")
    print(f"Features: {loader.num_features}")
    print(f"Classes: {loader.num_classes}")
