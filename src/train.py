"""
Training script for GNN fraud detection models.

Provides training loop, evaluation, and model checkpointing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch_geometric.data import Data
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from tqdm import tqdm
import json
from datetime import datetime

from .utils.metrics import compute_metrics


class Trainer:
    """
    Trainer for GNN models.
    
    Handles training loop, validation, early stopping, and checkpointing.
    
    Args:
        model: GNN model to train
        data: PyG Data object with train/val/test masks
        device: Device to train on ('cpu' or 'cuda')
        checkpoint_dir: Directory to save checkpoints
        
    Usage:
        trainer = Trainer(model, data, device='cuda')
        history = trainer.train(epochs=200, lr=1e-3)
        results = trainer.evaluate()
    """
    
    def __init__(
        self,
        model: nn.Module,
        data: Data,
        device: str = 'cpu',
        checkpoint_dir: str = './baseline/checkpoints'
    ):
        self.model = model.to(device)
        self.data = data.to(device)
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.optimizer = None
        self.scheduler = None
        self.best_val_f1 = 0.0
        self.best_epoch = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_f1': [],
            'val_f1': [],
            'train_auc': [],
            'val_auc': []
        }
        
    def train(
        self,
        epochs: int = 200,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 20,
        min_delta: float = 1e-4,
        scheduler_type: str = 'plateau',
        use_class_weights: bool = True,
        verbose: bool = True
    ) -> Dict:
        """
        Train the model.
        
        Args:
            epochs: Maximum number of epochs
            lr: Learning rate
            weight_decay: L2 regularization
            patience: Early stopping patience
            min_delta: Minimum improvement for early stopping
            scheduler_type: 'plateau' or 'cosine'
            use_class_weights: Whether to use class weights in loss
            verbose: Whether to print progress
            
        Returns:
            Training history dictionary
        """
        # Setup optimizer
        self.optimizer = Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Setup scheduler
        if scheduler_type == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=10,
                verbose=verbose
            )
        else:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=epochs,
                eta_min=1e-6
            )
        
        # Setup loss function
        if use_class_weights and hasattr(self.data, 'class_weights'):
            weight = self.data.class_weights.to(self.device)
        else:
            weight = None
        criterion = nn.CrossEntropyLoss(weight=weight)
        
        # Early stopping
        epochs_without_improvement = 0
        
        # Training loop
        pbar = tqdm(range(epochs), desc='Training', disable=not verbose)
        
        for epoch in pbar:
            # Train
            train_loss, train_metrics = self._train_epoch(criterion)
            
            # Validate
            val_loss, val_metrics = self._validate_epoch(criterion)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_f1'].append(train_metrics['f1'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['train_auc'].append(train_metrics['auc'])
            self.history['val_auc'].append(val_metrics['auc'])
            
            # Update scheduler
            if scheduler_type == 'plateau':
                self.scheduler.step(val_metrics['f1'])
            else:
                self.scheduler.step()
            
            # Check for improvement
            if val_metrics['f1'] > self.best_val_f1 + min_delta:
                self.best_val_f1 = val_metrics['f1']
                self.best_epoch = epoch
                epochs_without_improvement = 0
                self._save_checkpoint('best_model.pt')
            else:
                epochs_without_improvement += 1
            
            # Update progress bar
            pbar.set_postfix({
                'train_loss': f'{train_loss:.4f}',
                'val_f1': f'{val_metrics["f1"]:.4f}',
                'best_f1': f'{self.best_val_f1:.4f}'
            })
            
            # Early stopping
            if epochs_without_improvement >= patience:
                if verbose:
                    print(f'\nEarly stopping at epoch {epoch}. Best F1: {self.best_val_f1:.4f} at epoch {self.best_epoch}')
                break
        
        # Load best model
        self._load_checkpoint('best_model.pt')
        
        # Save training history
        self._save_history()
        
        return self.history
    
    def _train_epoch(self, criterion: nn.Module) -> Tuple[float, Dict]:
        """Run one training epoch."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        logits, _ = self.model(self.data.x, self.data.edge_index)
        
        # Only compute loss on training nodes
        train_logits = logits[self.data.train_mask]
        train_labels = self.data.y[self.data.train_mask]
        
        loss = criterion(train_logits, train_labels)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            train_probs = F.softmax(train_logits, dim=1)
            metrics = compute_metrics(
                train_labels.cpu().numpy(),
                train_probs[:, 1].cpu().numpy(),
                train_probs.argmax(dim=1).cpu().numpy()
            )
        
        return loss.item(), metrics
    
    @torch.no_grad()
    def _validate_epoch(self, criterion: nn.Module) -> Tuple[float, Dict]:
        """Run validation."""
        self.model.eval()
        
        # Forward pass
        logits, _ = self.model(self.data.x, self.data.edge_index)
        
        # Only compute on validation nodes
        val_logits = logits[self.data.val_mask]
        val_labels = self.data.y[self.data.val_mask]
        
        loss = criterion(val_logits, val_labels)
        
        # Compute metrics
        val_probs = F.softmax(val_logits, dim=1)
        metrics = compute_metrics(
            val_labels.cpu().numpy(),
            val_probs[:, 1].cpu().numpy(),
            val_probs.argmax(dim=1).cpu().numpy()
        )
        
        return loss.item(), metrics
    
    @torch.no_grad()
    def evaluate(self, mask_type: str = 'test') -> Dict:
        """
        Evaluate model on specified split.
        
        Args:
            mask_type: One of 'train', 'val', 'test'
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        
        # Get mask
        if mask_type == 'train':
            mask = self.data.train_mask
        elif mask_type == 'val':
            mask = self.data.val_mask
        else:
            mask = self.data.test_mask
        
        # Forward pass
        logits, embeddings = self.model(self.data.x, self.data.edge_index)
        
        # Get predictions for masked nodes
        test_logits = logits[mask]
        test_labels = self.data.y[mask]
        test_probs = F.softmax(test_logits, dim=1)
        
        # Compute metrics
        metrics = compute_metrics(
            test_labels.cpu().numpy(),
            test_probs[:, 1].cpu().numpy(),
            test_probs.argmax(dim=1).cpu().numpy()
        )
        
        print(f"\n{'='*50}")
        print(f"Evaluation Results ({mask_type} set)")
        print(f"{'='*50}")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")
        print(f"AUC-ROC:   {metrics['auc']:.4f}")
        
        return metrics
    
    def get_predictions(self, mask_type: str = 'all') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get model predictions and embeddings.
        
        Args:
            mask_type: 'train', 'val', 'test', or 'all'
            
        Returns:
            Tuple of (predictions, probabilities, embeddings)
        """
        self.model.eval()
        
        with torch.no_grad():
            logits, embeddings = self.model(self.data.x, self.data.edge_index)
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
        
        if mask_type == 'all':
            return (
                preds.cpu().numpy(),
                probs.cpu().numpy(),
                embeddings.cpu().numpy()
            )
        
        # Get mask
        if mask_type == 'train':
            mask = self.data.train_mask
        elif mask_type == 'val':
            mask = self.data.val_mask
        else:
            mask = self.data.test_mask
        
        return (
            preds[mask].cpu().numpy(),
            probs[mask].cpu().numpy(),
            embeddings[mask].cpu().numpy()
        )
    
    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.checkpoint_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'best_val_f1': self.best_val_f1,
            'best_epoch': self.best_epoch,
            'history': self.history
        }, path)
    
    def _load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = self.checkpoint_dir / filename
        if path.exists():
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.best_val_f1 = checkpoint.get('best_val_f1', 0.0)
            self.best_epoch = checkpoint.get('best_epoch', 0)
    
    def _save_history(self):
        """Save training history to JSON."""
        path = self.checkpoint_dir / 'training_history.json'
        history_serializable = {k: [float(v) for v in vals] for k, vals in self.history.items()}
        history_serializable['best_val_f1'] = float(self.best_val_f1)
        history_serializable['best_epoch'] = int(self.best_epoch)
        history_serializable['timestamp'] = datetime.now().isoformat()
        
        with open(path, 'w') as f:
            json.dump(history_serializable, f, indent=2)


def train_model(
    model: nn.Module,
    data: Data,
    epochs: int = 200,
    lr: float = 1e-3,
    device: str = 'cpu',
    checkpoint_dir: str = './baseline/checkpoints',
    **kwargs
) -> Tuple[nn.Module, Dict]:
    """
    Convenience function to train a model.
    
    Args:
        model: GNN model to train
        data: PyG Data object
        epochs: Number of epochs
        lr: Learning rate
        device: Device to use
        checkpoint_dir: Where to save checkpoints
        **kwargs: Additional arguments for Trainer.train()
        
    Returns:
        Tuple of (trained_model, history)
    """
    trainer = Trainer(model, data, device, checkpoint_dir)
    history = trainer.train(epochs=epochs, lr=lr, **kwargs)
    
    return trainer.model, history


if __name__ == '__main__':
    # Test with dummy data
    from torch_geometric.data import Data
    from .models.graphsage import GraphSAGEModel
    
    print("Testing Trainer")
    print("="*50)
    
    # Create dummy data
    num_nodes = 1000
    num_features = 166
    num_edges = 5000
    
    x = torch.randn(num_nodes, num_features)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    y = torch.randint(0, 2, (num_nodes,))
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[:600] = True
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask[600:800] = True
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[800:] = True
    
    data = Data(
        x=x, edge_index=edge_index, y=y,
        train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
        class_weights=torch.tensor([0.5, 4.5])
    )
    
    # Create model
    model = GraphSAGEModel(num_features=166, num_classes=2)
    
    # Train
    trainer = Trainer(model, data, device='cpu')
    history = trainer.train(epochs=10, lr=1e-3, patience=5)
    
    # Evaluate
    results = trainer.evaluate('test')
