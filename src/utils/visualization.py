"""
Visualization utilities for graph analysis and training.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import networkx as nx

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5)
):
    """
    Plot training and validation curves.
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_f1', 'val_f1', etc.
        save_path: Path to save the figure (optional)
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # F1 Score
    axes[1].plot(epochs, history['train_f1'], 'b-', label='Train', linewidth=2)
    axes[1].plot(epochs, history['val_f1'], 'r-', label='Val', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('F1 Score')
    axes[1].set_title('Training and Validation F1')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # AUC
    if 'train_auc' in history:
        axes[2].plot(epochs, history['train_auc'], 'b-', label='Train', linewidth=2)
        axes[2].plot(epochs, history['val_auc'], 'r-', label='Val', linewidth=2)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('AUC-ROC')
        axes[2].set_title('Training and Validation AUC')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str] = ['Licit', 'Illicit'],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
):
    """
    Plot confusion matrix heatmap.
    
    Args:
        cm: Confusion matrix array
        labels: Class labels
        save_path: Path to save figure
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={'size': 14}
    )
    
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_subgraph(
    edge_index: np.ndarray,
    node_features: Optional[np.ndarray] = None,
    node_labels: Optional[np.ndarray] = None,
    node_scores: Optional[np.ndarray] = None,
    center_node: Optional[int] = None,
    edge_weights: Optional[np.ndarray] = None,
    title: str = 'Subgraph Visualization',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10)
):
    """
    Visualize a subgraph with optional node colors and edge weights.
    
    Useful for GNNExplainer output visualization.
    
    Args:
        edge_index: Edge indices (2, num_edges)
        node_features: Node features (optional)
        node_labels: Node labels for coloring (optional)
        node_scores: Node fraud scores for coloring (optional)
        center_node: Central node to highlight (optional)
        edge_weights: Edge importance weights (optional)
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    # Create networkx graph
    G = nx.DiGraph()
    
    # Get unique nodes
    unique_nodes = np.unique(edge_index)
    G.add_nodes_from(unique_nodes)
    
    # Add edges
    edges = list(zip(edge_index[0], edge_index[1]))
    G.add_edges_from(edges)
    
    # Layout
    pos = nx.spring_layout(G, seed=42, k=2)
    
    plt.figure(figsize=figsize)
    
    # Node colors
    if node_scores is not None:
        node_colors = [node_scores[n] if n < len(node_scores) else 0.5 for n in G.nodes()]
        cmap = 'RdYlGn_r'  # Red for high fraud score
    elif node_labels is not None:
        label_colors = {-1: 'gray', 0: 'green', 1: 'red'}
        node_colors = [label_colors.get(node_labels[n] if n < len(node_labels) else -1, 'gray') for n in G.nodes()]
        cmap = None
    else:
        node_colors = 'lightblue'
        cmap = None
    
    # Node sizes
    node_sizes = [500 if n == center_node else 300 for n in G.nodes()]
    
    # Draw nodes
    if cmap:
        nodes = nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=node_sizes,
            cmap=plt.cm.get_cmap(cmap),
            vmin=0, vmax=1
        )
        plt.colorbar(nodes, label='Fraud Score')
    else:
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=node_sizes
        )
    
    # Draw edges
    if edge_weights is not None:
        # Normalize weights for visualization
        weights_norm = (edge_weights - edge_weights.min()) / (edge_weights.max() - edge_weights.min() + 1e-8)
        edge_colors = plt.cm.Reds(weights_norm)
        
        nx.draw_networkx_edges(
            G, pos,
            edge_color=edge_colors,
            width=[1 + 3 * w for w in weights_norm],
            alpha=0.7,
            arrows=True,
            arrowsize=15
        )
    else:
        nx.draw_networkx_edges(
            G, pos,
            edge_color='gray',
            width=1.5,
            alpha=0.5,
            arrows=True,
            arrowsize=15
        )
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    # Highlight center node
    if center_node is not None and center_node in G.nodes():
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=[center_node],
            node_color='none',
            node_size=600,
            edgecolors='black',
            linewidths=3
        )
    
    plt.title(title, fontsize=14)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_feature_importance(
    feature_importance: np.ndarray,
    feature_names: Optional[List[str]] = None,
    top_k: int = 20,
    title: str = 'Top Feature Importance',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
):
    """
    Plot feature importance bar chart.
    
    Args:
        feature_importance: Importance scores for each feature
        feature_names: Names of features (optional)
        top_k: Number of top features to show
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    # Get top-k features
    top_indices = np.argsort(feature_importance)[-top_k:][::-1]
    top_importance = feature_importance[top_indices]
    
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(len(feature_importance))]
    
    top_names = [feature_names[i] for i in top_indices]
    
    plt.figure(figsize=figsize)
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_k))
    
    plt.barh(range(top_k), top_importance[::-1], color=colors)
    plt.yticks(range(top_k), top_names[::-1])
    plt.xlabel('Importance Score')
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_embedding_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    title: str = 't-SNE Visualization of Node Embeddings',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    sample_size: int = 5000
):
    """
    Plot t-SNE visualization of embeddings.
    
    Args:
        embeddings: Node embeddings (num_nodes, embedding_dim)
        labels: Node labels
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        sample_size: Number of samples to plot (for performance)
    """
    from sklearn.manifold import TSNE
    
    # Sample if too many points
    if len(embeddings) > sample_size:
        indices = np.random.choice(len(embeddings), sample_size, replace=False)
        embeddings = embeddings[indices]
        labels = labels[indices]
    
    # Run t-SNE
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=figsize)
    
    # Define colors
    color_map = {-1: 'gray', 0: 'green', 1: 'red'}
    label_names = {-1: 'Unknown', 0: 'Licit', 1: 'Illicit'}
    
    for label in sorted(np.unique(labels)):
        mask = labels == label
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=color_map.get(label, 'blue'),
            label=label_names.get(label, f'Class {label}'),
            alpha=0.5,
            s=20
        )
    
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title(title)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_fraud_score_distribution(
    fraud_scores: np.ndarray,
    labels: np.ndarray,
    title: str = 'Fraud Score Distribution by Class',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Plot distribution of fraud scores by class.
    
    Args:
        fraud_scores: Predicted fraud probabilities
        labels: True labels
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Only plot labeled data
    labeled_mask = labels != -1
    scores_labeled = fraud_scores[labeled_mask]
    labels_labeled = labels[labeled_mask]
    
    # Plot histograms
    plt.hist(
        scores_labeled[labels_labeled == 0],
        bins=50, alpha=0.5, label='Licit', color='green', density=True
    )
    plt.hist(
        scores_labeled[labels_labeled == 1],
        bins=50, alpha=0.5, label='Illicit', color='red', density=True
    )
    
    plt.xlabel('Fraud Score')
    plt.ylabel('Density')
    plt.title(title)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


if __name__ == '__main__':
    # Test visualizations
    print("Testing visualization utilities")
    print("="*50)
    
    # Test training curves
    history = {
        'train_loss': np.random.rand(20).cumsum()[::-1].tolist(),
        'val_loss': (np.random.rand(20).cumsum()[::-1] + 0.1).tolist(),
        'train_f1': np.linspace(0.5, 0.9, 20).tolist(),
        'val_f1': np.linspace(0.45, 0.85, 20).tolist(),
        'train_auc': np.linspace(0.6, 0.95, 20).tolist(),
        'val_auc': np.linspace(0.55, 0.90, 20).tolist()
    }
    
    print("Plotting training curves...")
    plot_training_curves(history)
