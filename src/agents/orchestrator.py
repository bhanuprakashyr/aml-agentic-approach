"""
AML Orchestrator

Coordinates the Hybrid 2-Agent system:
1. Coordinator Agent gathers evidence
2. Analyst Agent produces verdict

Provides a simple interface for transaction investigation.
"""

import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .tools import AMLTools
from .coordinator import CoordinatorAgent, InvestigationState
from .analyst import AnalystAgent, Verdict
from ..llm.client import LLMClient


@dataclass
class InvestigationResult:
    """Complete investigation result."""
    node_idx: int
    state: InvestigationState
    verdict: Verdict
    duration_seconds: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "node_idx": self.node_idx,
            "investigation": {
                "steps": len(self.state.steps),
                "evidence_collected": list(self.state.evidence.keys()),
                "status": self.state.status,
                "started_at": self.state.started_at,
                "completed_at": self.state.completed_at
            },
            "verdict": self.verdict.to_dict(),
            "duration_seconds": self.duration_seconds
        }
    
    def save(self, filepath: str):
        """Save result to JSON file."""
        import numpy as np
        
        def convert_numpy(obj):
            """Convert numpy types to Python native types for JSON serialization."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        with open(filepath, 'w') as f:
            json.dump(convert_numpy(self.to_dict()), f, indent=2)


class AMLOrchestrator:
    """
    Orchestrator for Agentic AML Investigation.
    
    Coordinates the two-agent investigation process:
    1. CoordinatorAgent plans and gathers evidence using tools
    2. AnalystAgent reviews evidence and produces verdict
    
    This class provides the main interface for running investigations.
    
    Args:
        model: Trained GNN model
        data: PyG Data object
        case_memory: CaseMemory instance
        faiss_index: FAISSIndex instance
        llm_provider: LLM provider ("openai", "anthropic", "ollama")
        llm_model: Specific model to use (optional)
        device: Device for model inference
        verbose: Print investigation progress
        
    Usage:
        # Initialize
        orchestrator = AMLOrchestrator(
            model=model,
            data=data,
            case_memory=case_memory,
            faiss_index=faiss_index,
            llm_provider="ollama"
        )
        
        # Single investigation
        result = orchestrator.investigate(node_idx=12345)
        print(result.verdict.to_report())
        
        # Batch investigation
        results = orchestrator.batch_investigate([12345, 67890, 11111])
    """
    
    def __init__(
        self,
        model,
        data,
        case_memory,
        faiss_index,
        llm_provider: str = "ollama",
        llm_model: Optional[str] = None,
        device: str = 'cpu',
        verbose: bool = True,
        max_steps: int = 5
    ):
        # Initialize LLM client
        self.llm = LLMClient(provider=llm_provider, model=llm_model)
        
        # Initialize tools
        self.tools = AMLTools(
            model=model,
            data=data,
            case_memory=case_memory,
            faiss_index=faiss_index,
            device=device
        )
        
        # Initialize agents
        self.coordinator = CoordinatorAgent(
            tools=self.tools,
            llm=self.llm,
            max_steps=max_steps,
            verbose=verbose
        )
        
        self.analyst = AnalystAgent(
            llm=self.llm,
            verbose=verbose
        )
        
        self.verbose = verbose
        self._investigation_history: List[InvestigationResult] = []
    
    def investigate(self, node_idx: int) -> InvestigationResult:
        """
        Run complete investigation on a transaction.
        
        Args:
            node_idx: Transaction node index
            
        Returns:
            InvestigationResult with evidence and verdict
        """
        start_time = datetime.now()
        
        if self.verbose:
            print(f"\n{'#'*70}")
            print(f"#  AGENTIC AML INVESTIGATION - Transaction #{node_idx}")
            print(f"{'#'*70}")
        
        # Phase 1: Coordinator gathers evidence
        if self.verbose:
            print("\n[PHASE 1: EVIDENCE GATHERING]")
        
        state = self.coordinator.investigate(node_idx)
        
        # Phase 2: Analyst produces verdict
        if self.verbose:
            print("\n[PHASE 2: RISK ASSESSMENT]")
        
        verdict = self.analyst.assess(state)
        
        # Calculate duration
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Create result
        result = InvestigationResult(
            node_idx=node_idx,
            state=state,
            verdict=verdict,
            duration_seconds=duration
        )
        
        # Store in history
        self._investigation_history.append(result)
        
        if self.verbose:
            print(f"\n{'#'*70}")
            print(f"#  INVESTIGATION COMPLETE")
            print(f"#  Duration: {duration:.1f} seconds")
            print(f"#  Verdict: {verdict.risk_level.value} ({verdict.confidence*100:.0f}% confidence)")
            print(f"{'#'*70}")
        
        return result
    
    def batch_investigate(
        self,
        node_indices: List[int],
        save_results: bool = False,
        output_dir: str = "./investigation_results"
    ) -> List[InvestigationResult]:
        """
        Run investigations on multiple transactions.
        
        Args:
            node_indices: List of transaction node indices
            save_results: Whether to save individual results to files
            output_dir: Directory for saving results
            
        Returns:
            List of InvestigationResult objects
        """
        results = []
        
        if save_results:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for i, node_idx in enumerate(node_indices, 1):
            if self.verbose:
                print(f"\n[Batch Progress: {i}/{len(node_indices)}]")
            
            result = self.investigate(node_idx)
            results.append(result)
            
            if save_results:
                filepath = f"{output_dir}/investigation_{node_idx}.json"
                result.save(filepath)
                if self.verbose:
                    print(f"Saved: {filepath}")
        
        return results
    
    def get_summary(self) -> Dict:
        """Get summary of all investigations."""
        if not self._investigation_history:
            return {"message": "No investigations conducted yet."}
        
        verdicts = [r.verdict for r in self._investigation_history]
        
        risk_distribution = {
            "CRITICAL": sum(1 for v in verdicts if v.risk_level.value == "CRITICAL"),
            "HIGH": sum(1 for v in verdicts if v.risk_level.value == "HIGH"),
            "MEDIUM": sum(1 for v in verdicts if v.risk_level.value == "MEDIUM"),
            "LOW": sum(1 for v in verdicts if v.risk_level.value == "LOW")
        }
        
        recommendation_distribution = {
            "FLAG_IMMEDIATE": sum(1 for v in verdicts if v.recommendation.value == "FLAG_IMMEDIATE"),
            "INVESTIGATE": sum(1 for v in verdicts if v.recommendation.value == "INVESTIGATE"),
            "MONITOR": sum(1 for v in verdicts if v.recommendation.value == "MONITOR"),
            "CLEAR": sum(1 for v in verdicts if v.recommendation.value == "CLEAR")
        }
        
        avg_confidence = sum(v.confidence for v in verdicts) / len(verdicts)
        avg_duration = sum(r.duration_seconds for r in self._investigation_history) / len(self._investigation_history)
        
        return {
            "total_investigations": len(self._investigation_history),
            "risk_distribution": risk_distribution,
            "recommendation_distribution": recommendation_distribution,
            "average_confidence": f"{avg_confidence * 100:.1f}%",
            "average_duration_seconds": round(avg_duration, 2),
            "high_risk_transactions": [
                r.node_idx for r in self._investigation_history 
                if r.verdict.risk_level.value in ["CRITICAL", "HIGH"]
            ]
        }
    
    def print_report(self, result: InvestigationResult):
        """Print formatted investigation report."""
        print(result.verdict.to_report())
    
    def test_llm_connection(self) -> bool:
        """Test if LLM is accessible."""
        return self.llm.test_connection()
