"""
End-to-End Inference Pipeline

Combines all components for full fraud analysis:
1. GNN prediction
2. GNNExplainer explanation
3. FAISS case retrieval
4. ICL prompt generation
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from torch_geometric.data import Data


class InferencePipeline:
    """
    End-to-end inference pipeline for fraud analysis.
    
    Combines:
    - GNN model for predictions and embeddings
    - GNNExplainer for interpretability
    - FAISS retrieval for similar cases
    - ICL prompt builder for LLM analysis
    
    Args:
        model: Trained GNN model
        data: PyG Data object
        case_memory: CaseMemory with historical cases
        faiss_index: FAISS index built from case embeddings
        device: Device for inference
        
    Usage:
        pipeline = InferencePipeline(model, data, case_memory, faiss_index)
        result = pipeline.analyze(node_idx=12345)
        print(result['prompt'])
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        data: Data,
        case_memory,  # CaseMemory
        faiss_index,  # FAISSIndex
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.data = data.to(device)
        self.case_memory = case_memory
        self.faiss_index = faiss_index
        self.device = device
        
        # Initialize explainer lazily
        self._explainer = None
        
        # Initialize prompt builder
        from ..prompts.icl_constructor import ICLPromptBuilder
        self.prompt_builder = ICLPromptBuilder()
    
    @property
    def explainer(self):
        """Lazy initialization of explainer."""
        if self._explainer is None:
            from ..explainer.gnn_explainer import FraudExplainer
            self._explainer = FraudExplainer(self.model, self.data, self.device)
        return self._explainer
    
    def analyze(
        self,
        node_idx: int,
        k_cases: int = 3,
        generate_explanation: bool = True,
        generate_prompt: bool = True
    ) -> Dict:
        """
        Run full analysis on a transaction.
        
        Args:
            node_idx: Index of transaction to analyze
            k_cases: Number of similar cases to retrieve
            generate_explanation: Whether to run GNNExplainer
            generate_prompt: Whether to generate ICL prompt
            
        Returns:
            Dictionary containing:
            - prediction: Model prediction info
            - embedding: Node embedding
            - explanation: GNNExplainer output (if requested)
            - similar_cases: Retrieved similar cases
            - prompt: ICL prompt for LLM (if requested)
        """
        result = {}
        
        # 1. Get model prediction
        self.model.eval()
        with torch.no_grad():
            logits, embeddings = self.model(self.data.x, self.data.edge_index)
            probs = torch.softmax(logits, dim=1)
            
            fraud_score = probs[node_idx, 1].item()
            predicted_label = 'illicit' if fraud_score > 0.5 else 'licit'
            embedding = embeddings[node_idx].cpu().numpy()
        
        # Get true label if available
        true_label_idx = self.data.y[node_idx].item()
        true_label = {-1: 'unknown', 0: 'licit', 1: 'illicit'}.get(true_label_idx, 'unknown')
        
        result['prediction'] = {
            'node_idx': node_idx,
            'fraud_score': fraud_score,
            'predicted_label': predicted_label,
            'true_label': true_label
        }
        result['embedding'] = embedding
        
        # 2. Generate explanation
        if generate_explanation:
            explanation = self.explainer.explain_node(node_idx)
            result['explanation'] = explanation
        else:
            explanation = None
        
        # 3. Retrieve similar cases
        distances, indices = self.faiss_index.search(
            embedding.reshape(1, -1).astype(np.float32), k_cases
        )
        
        similar_cases = []
        for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if idx < len(self.case_memory.cases):
                case = self.case_memory.cases[idx]
                similar_cases.append({
                    'case': case,
                    'case_id': case.case_id,
                    'similarity': 1 / (1 + dist),
                    'distance': float(dist),
                    'rank': i + 1
                })
        
        result['similar_cases'] = similar_cases
        
        # 4. Generate ICL prompt
        if generate_prompt:
            from ..prompts.icl_constructor import TransactionContext
            
            # Build transaction context
            top_features = []
            if explanation:
                top_features = explanation.top_features
            
            # Calculate network stats
            edge_index = self.data.edge_index.cpu().numpy()
            in_degree = (edge_index[1] == node_idx).sum()
            out_degree = (edge_index[0] == node_idx).sum()
            
            transaction = TransactionContext(
                node_idx=node_idx,
                fraud_score=fraud_score,
                predicted_label=predicted_label,
                embedding=embedding.tolist(),
                top_features=top_features,
                feature_summary="",
                num_neighbors=in_degree + out_degree,
                in_degree=int(in_degree),
                out_degree=int(out_degree),
                subgraph_summary=f"Connected to {in_degree + out_degree} transactions",
                explanation_narrative=explanation.narrative if explanation else ""
            )
            
            prompt = self.prompt_builder.build_prompt(transaction, similar_cases)
            result['prompt'] = prompt
        
        return result
    
    def batch_analyze(
        self,
        node_indices: List[int],
        k_cases: int = 3,
        verbose: bool = True
    ) -> List[Dict]:
        """
        Analyze multiple transactions.
        
        Args:
            node_indices: List of node indices
            k_cases: Number of similar cases per transaction
            verbose: Whether to show progress
            
        Returns:
            List of analysis results
        """
        from tqdm import tqdm
        
        results = []
        iterator = tqdm(node_indices, desc="Analyzing") if verbose else node_indices
        
        for node_idx in iterator:
            try:
                result = self.analyze(
                    node_idx,
                    k_cases=k_cases,
                    generate_explanation=False,  # Skip for batch
                    generate_prompt=False
                )
                results.append(result)
            except Exception as e:
                if verbose:
                    print(f"Warning: Failed to analyze node {node_idx}: {e}")
                continue
        
        return results
    
    def get_top_suspicious(
        self,
        top_k: int = 100,
        exclude_labeled: bool = False
    ) -> List[Dict]:
        """
        Get the most suspicious transactions.
        
        Args:
            top_k: Number of transactions to return
            exclude_labeled: Whether to exclude already labeled transactions
            
        Returns:
            List of transaction info sorted by fraud score
        """
        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(self.data.x, self.data.edge_index)
            probs = torch.softmax(logits, dim=1)
            fraud_scores = probs[:, 1].cpu().numpy()
        
        # Get indices sorted by fraud score
        sorted_indices = np.argsort(fraud_scores)[::-1]
        
        results = []
        for idx in sorted_indices:
            if len(results) >= top_k:
                break
            
            idx = int(idx)
            
            # Skip if labeled and exclude_labeled is True
            if exclude_labeled:
                label = self.data.y[idx].item()
                if label != -1:
                    continue
            
            results.append({
                'node_idx': idx,
                'fraud_score': fraud_scores[idx],
                'true_label': {-1: 'unknown', 0: 'licit', 1: 'illicit'}.get(
                    self.data.y[idx].item(), 'unknown'
                )
            })
        
        return results


def create_pipeline(
    model_path: str,
    data_path: str,
    case_memory_path: str,
    faiss_index_path: str,
    device: str = 'cpu'
) -> InferencePipeline:
    """
    Factory function to create inference pipeline from saved components.
    
    Args:
        model_path: Path to saved model checkpoint
        data_path: Path to saved PyG data
        case_memory_path: Path to case memory directory
        faiss_index_path: Path to FAISS index file
        device: Device for inference
        
    Returns:
        Configured InferencePipeline
    """
    import torch
    from ..models.graphsage import GraphSAGEModel
    from ..memory.case_store import CaseMemory
    from ..retrieval.faiss_index import FAISSIndex
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = GraphSAGEModel(num_features=166, num_classes=2)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load data
    data = torch.load(data_path, map_location=device)
    
    # Load case memory
    case_memory = CaseMemory.load(case_memory_path)
    
    # Load FAISS index
    faiss_index = FAISSIndex(embedding_dim=64)
    faiss_index.load(faiss_index_path)
    
    return InferencePipeline(model, data, case_memory, faiss_index, device)


if __name__ == '__main__':
    print("Inference Pipeline - Use with trained model and case memory")
