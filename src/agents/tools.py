"""
AML Investigation Tools

Wraps existing pipeline components as callable tools for the Coordinator Agent.
Each tool has a name, description, and callable function.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field


@dataclass
class Tool:
    """Definition of a callable tool."""
    name: str
    description: str
    func: Callable
    parameters: Dict[str, str] = field(default_factory=dict)
    
    def __call__(self, **kwargs) -> Any:
        """Execute the tool."""
        return self.func(**kwargs)


class AMLTools:
    """
    Collection of tools for AML investigation.
    
    Wraps existing components (GNN model, explainer, FAISS, case memory)
    as callable tools that the Coordinator Agent can invoke.
    
    Available Tools:
        - get_fraud_score: Get ML fraud probability for a transaction
        - retrieve_similar_cases: Find similar historical cases
        - explain_prediction: Get feature importance explanation
        - get_network_context: Get transaction neighborhood statistics
        - lookup_case: Get details of a specific case from memory
    
    Usage:
        tools = AMLTools(model, data, case_memory, faiss_index)
        result = tools.execute("get_fraud_score", node_idx=12345)
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        data,  # PyG Data object
        case_memory,  # CaseMemory
        faiss_index,  # FAISSIndex
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.data = data.to(device)
        self.case_memory = case_memory
        self.faiss_index = faiss_index
        self.device = device
        
        # Cache embeddings for efficiency
        self._embeddings = None
        self._fraud_scores = None
        self._compute_embeddings()
        
        # Register tools
        self._tools = self._register_tools()
    
    def _compute_embeddings(self):
        """Compute and cache all node embeddings."""
        self.model.eval()
        with torch.no_grad():
            logits, embeddings = self.model(self.data.x, self.data.edge_index)
            probs = torch.softmax(logits, dim=1)
            self._fraud_scores = probs[:, 1].cpu().numpy()
            self._embeddings = embeddings.cpu().numpy()
    
    def _register_tools(self) -> Dict[str, Tool]:
        """Register all available tools."""
        return {
            "get_fraud_score": Tool(
                name="get_fraud_score",
                description="Get the ML model's fraud probability score for a transaction. Returns score between 0 (licit) and 1 (illicit).",
                func=self._get_fraud_score,
                parameters={"node_idx": "Transaction node index (integer)"}
            ),
            "retrieve_similar_cases": Tool(
                name="retrieve_similar_cases",
                description="Find similar historical cases from case memory using embedding similarity. Returns list of similar cases with their verdicts.",
                func=self._retrieve_similar_cases,
                parameters={
                    "node_idx": "Transaction node index (integer)",
                    "k": "Number of similar cases to retrieve (default: 3)"
                }
            ),
            "explain_prediction": Tool(
                name="explain_prediction",
                description="Get feature importance explanation for why the model flagged this transaction. Returns top contributing features.",
                func=self._explain_prediction,
                parameters={"node_idx": "Transaction node index (integer)"}
            ),
            "get_network_context": Tool(
                name="get_network_context",
                description="Get network/graph context for a transaction including neighbor count, connectivity patterns, and nearby suspicious nodes.",
                func=self._get_network_context,
                parameters={"node_idx": "Transaction node index (integer)"}
            ),
            "lookup_case": Tool(
                name="lookup_case",
                description="Get detailed information about a specific case from case memory by its ID.",
                func=self._lookup_case,
                parameters={"case_id": "Case ID (integer)"}
            ),
            "get_transaction_features": Tool(
                name="get_transaction_features",
                description="Get the raw feature values for a transaction node.",
                func=self._get_transaction_features,
                parameters={"node_idx": "Transaction node index (integer)"}
            ),
        }
    
    @property
    def tool_descriptions(self) -> str:
        """Get formatted descriptions of all tools for the agent prompt."""
        lines = ["Available Tools:"]
        for name, tool in self._tools.items():
            params = ", ".join([f"{k}: {v}" for k, v in tool.parameters.items()])
            lines.append(f"\n- {name}: {tool.description}")
            lines.append(f"  Parameters: {params}")
        return "\n".join(lines)
    
    @property
    def tool_names(self) -> List[str]:
        """Get list of tool names."""
        return list(self._tools.keys())
    
    def execute(self, tool_name: str, **kwargs) -> Dict:
        """
        Execute a tool by name.
        
        Args:
            tool_name: Name of tool to execute
            **kwargs: Parameters for the tool
            
        Returns:
            Tool result as dictionary
        """
        if tool_name not in self._tools:
            return {
                "error": f"Unknown tool: {tool_name}",
                "available_tools": self.tool_names
            }
        
        try:
            result = self._tools[tool_name](**kwargs)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # ==================== Tool Implementations ====================
    
    def _get_fraud_score(self, node_idx: int) -> Dict:
        """Get fraud score for a transaction."""
        if node_idx < 0 or node_idx >= len(self._fraud_scores):
            return {"error": f"Invalid node_idx: {node_idx}"}
        
        score = float(self._fraud_scores[node_idx])
        true_label_idx = int(self.data.y[node_idx].item())
        true_label = {-1: "unknown", 0: "licit", 1: "illicit"}.get(true_label_idx, "unknown")
        
        # Determine risk level
        if score >= 0.8:
            risk_level = "CRITICAL"
        elif score >= 0.6:
            risk_level = "HIGH"
        elif score >= 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return {
            "node_idx": node_idx,
            "fraud_score": round(score, 4),
            "fraud_percentage": f"{score * 100:.1f}%",
            "risk_level": risk_level,
            "predicted_label": "illicit" if score > 0.5 else "licit",
            "true_label": true_label
        }
    
    def _retrieve_similar_cases(self, node_idx: int, k: int = 3) -> Dict:
        """Retrieve similar cases from case memory."""
        if node_idx < 0 or node_idx >= len(self._embeddings):
            return {"error": f"Invalid node_idx: {node_idx}"}
        
        # Get embedding for query node
        query_embedding = self._embeddings[node_idx].reshape(1, -1).astype(np.float32)
        
        # Search FAISS index
        distances, indices = self.faiss_index.search(query_embedding, k=k)
        
        # Build results
        similar_cases = []
        for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if idx < len(self.case_memory.cases):
                case = self.case_memory.cases[idx]
                similarity = 1 / (1 + dist)  # Convert distance to similarity
                
                similar_cases.append({
                    "rank": i + 1,
                    "case_id": case.case_id,
                    "similarity": round(similarity, 4),
                    "similarity_percentage": f"{similarity * 100:.1f}%",
                    "true_label": case.true_label,
                    "fraud_score": round(case.fraud_score, 4),
                    "summary": f"Case {case.case_id}: {case.true_label.upper()} transaction with {case.fraud_score*100:.0f}% fraud score"
                })
        
        # Aggregate statistics
        illicit_count = sum(1 for c in similar_cases if c["true_label"] == "illicit")
        licit_count = sum(1 for c in similar_cases if c["true_label"] == "licit")
        
        return {
            "query_node": node_idx,
            "num_retrieved": len(similar_cases),
            "similar_cases": similar_cases,
            "label_distribution": {
                "illicit": illicit_count,
                "licit": licit_count
            },
            "majority_verdict": "illicit" if illicit_count > licit_count else "licit"
        }
    
    def _explain_prediction(self, node_idx: int) -> Dict:
        """Get explanation for prediction using feature importance."""
        if node_idx < 0 or node_idx >= self.data.num_nodes:
            return {"error": f"Invalid node_idx: {node_idx}"}
        
        # Get node features
        features = self.data.x[node_idx].cpu().numpy()
        
        # Simple feature importance: magnitude-based
        # (In production, this would use GNNExplainer)
        feature_magnitudes = np.abs(features)
        top_indices = np.argsort(feature_magnitudes)[-10:][::-1]
        
        top_features = []
        for idx in top_indices:
            top_features.append({
                "feature_index": int(idx),
                "feature_name": f"feature_{idx}",
                "value": round(float(features[idx]), 4),
                "magnitude": round(float(feature_magnitudes[idx]), 4)
            })
        
        # Check if we have GNNExplainer results in case memory
        gnn_explanation = None
        for case in self.case_memory.cases:
            if case.node_idx == node_idx and case.explanation:
                gnn_explanation = case.explanation
                break
        
        return {
            "node_idx": node_idx,
            "method": "gnn_explainer" if gnn_explanation else "feature_magnitude",
            "top_features": top_features[:5],
            "explanation_summary": f"Top contributing features: {[f['feature_index'] for f in top_features[:5]]}",
            "gnn_explainer_available": gnn_explanation is not None
        }
    
    def _get_network_context(self, node_idx: int) -> Dict:
        """Get network context for a transaction."""
        if node_idx < 0 or node_idx >= self.data.num_nodes:
            return {"error": f"Invalid node_idx: {node_idx}"}
        
        edge_index = self.data.edge_index.cpu().numpy()
        
        # Calculate degree
        out_edges = np.sum(edge_index[0] == node_idx)
        in_edges = np.sum(edge_index[1] == node_idx)
        
        # Find 1-hop neighbors
        out_neighbors = edge_index[1][edge_index[0] == node_idx].tolist()
        in_neighbors = edge_index[0][edge_index[1] == node_idx].tolist()
        all_neighbors = list(set(out_neighbors + in_neighbors))
        
        # Check neighbor risk levels
        neighbor_fraud_scores = [float(self._fraud_scores[n]) for n in all_neighbors if n < len(self._fraud_scores)]
        high_risk_neighbors = sum(1 for s in neighbor_fraud_scores if s > 0.7)
        
        # Get labels for neighbors
        neighbor_labels = []
        for n in all_neighbors[:10]:  # Limit to first 10
            label_idx = int(self.data.y[n].item())
            label = {-1: "unknown", 0: "licit", 1: "illicit"}.get(label_idx, "unknown")
            neighbor_labels.append(label)
        
        illicit_neighbors = neighbor_labels.count("illicit")
        
        return {
            "node_idx": node_idx,
            "out_degree": int(out_edges),
            "in_degree": int(in_edges),
            "total_degree": int(out_edges + in_edges),
            "num_neighbors": len(all_neighbors),
            "high_risk_neighbors": high_risk_neighbors,
            "known_illicit_neighbors": illicit_neighbors,
            "avg_neighbor_fraud_score": round(np.mean(neighbor_fraud_scores), 4) if neighbor_fraud_scores else 0,
            "network_risk": "HIGH" if high_risk_neighbors >= 2 or illicit_neighbors >= 1 else "MEDIUM" if high_risk_neighbors >= 1 else "LOW",
            "summary": f"Connected to {len(all_neighbors)} nodes, {high_risk_neighbors} high-risk, {illicit_neighbors} known illicit"
        }
    
    def _lookup_case(self, case_id: int) -> Dict:
        """Lookup a specific case from case memory."""
        for case in self.case_memory.cases:
            if case.case_id == case_id:
                return {
                    "case_id": case.case_id,
                    "node_idx": case.node_idx,
                    "true_label": case.true_label,
                    "predicted_label": case.predicted_label,
                    "fraud_score": round(case.fraud_score, 4),
                    "time_step": case.time_step,
                    "in_degree": case.in_degree,
                    "out_degree": case.out_degree,
                    "has_explanation": bool(case.explanation),
                    "explanation_method": case.explanation.get("method", "unknown") if case.explanation else None
                }
        
        return {"error": f"Case {case_id} not found in memory"}
    
    def _get_transaction_features(self, node_idx: int) -> Dict:
        """Get raw features for a transaction."""
        if node_idx < 0 or node_idx >= self.data.num_nodes:
            return {"error": f"Invalid node_idx: {node_idx}"}
        
        features = self.data.x[node_idx].cpu().numpy()
        
        # Get basic statistics
        return {
            "node_idx": node_idx,
            "num_features": len(features),
            "feature_mean": round(float(np.mean(features)), 4),
            "feature_std": round(float(np.std(features)), 4),
            "feature_min": round(float(np.min(features)), 4),
            "feature_max": round(float(np.max(features)), 4),
            "non_zero_features": int(np.sum(features != 0)),
            "top_5_values": [round(float(v), 4) for v in sorted(features, reverse=True)[:5]]
        }
