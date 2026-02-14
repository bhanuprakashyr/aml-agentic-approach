"""
ICL (In-Context Learning) Prompt Constructor

Builds prompts for LLM analysis using retrieved similar cases.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from .templates import PromptTemplates


@dataclass
class TransactionContext:
    """Context for the transaction being analyzed."""
    node_idx: int
    fraud_score: float
    predicted_label: str
    embedding: List[float]
    
    # Feature summary
    top_features: List[Dict]
    feature_summary: str
    
    # Network context
    num_neighbors: int
    in_degree: int
    out_degree: int
    subgraph_summary: str
    
    # Explanation
    explanation_narrative: str


class ICLPromptBuilder:
    """
    Builds In-Context Learning prompts for LLM analysis.
    
    Constructs prompts that include:
    1. System instruction for the AML analyst role
    2. Current transaction analysis
    3. Similar historical cases (from FAISS retrieval)
    4. Task instruction
    
    Args:
        templates: PromptTemplates instance (optional, uses defaults)
        max_cases: Maximum number of similar cases to include
        
    Usage:
        builder = ICLPromptBuilder()
        prompt = builder.build_prompt(
            transaction=current_tx,
            similar_cases=retrieved_cases
        )
    """
    
    def __init__(
        self,
        templates: Optional[PromptTemplates] = None,
        max_cases: int = 3
    ):
        self.templates = templates or PromptTemplates()
        self.max_cases = max_cases
    
    def build_prompt(
        self,
        transaction: TransactionContext,
        similar_cases: List[Dict],
        task: str = 'full_analysis'
    ) -> str:
        """
        Build complete ICL prompt.
        
        Args:
            transaction: Current transaction context
            similar_cases: Retrieved similar cases from FAISS
            task: Type of task ('full_analysis', 'risk_only', 'recommendation_only')
            
        Returns:
            Complete prompt string
        """
        sections = []
        
        # 1. System instruction
        sections.append(self.templates.get_system_prompt())
        
        # 2. Current transaction analysis
        sections.append(self._build_transaction_section(transaction))
        
        # 3. Similar historical cases
        if similar_cases:
            sections.append(self._build_cases_section(similar_cases[:self.max_cases]))
        
        # 4. Task instruction
        sections.append(self._build_task_section(task))
        
        return "\n\n".join(sections)
    
    def _build_transaction_section(self, transaction: TransactionContext) -> str:
        """Build the current transaction section."""
        risk_level = self._get_risk_level(transaction.fraud_score)
        
        lines = [
            "## Current Transaction Analysis",
            "",
            f"**Transaction ID:** #{transaction.node_idx}",
            f"**Fraud Score:** {transaction.fraud_score:.1%}",
            f"**Risk Level:** {risk_level}",
            f"**Model Prediction:** {transaction.predicted_label.upper()}",
            "",
            "### Key Risk Indicators",
        ]
        
        # Add top features
        for i, feat in enumerate(transaction.top_features[:5], 1):
            lines.append(f"{i}. **{feat.get('name', f'Feature {feat.get(\"index\", i)}')}**: "
                        f"importance={feat.get('importance', 0):.3f}, "
                        f"value={feat.get('value', 0):.3f}")
        
        # Add network context
        lines.extend([
            "",
            "### Network Context",
            f"- Connected to {transaction.num_neighbors} transactions in neighborhood",
            f"- Incoming connections: {transaction.in_degree}",
            f"- Outgoing connections: {transaction.out_degree}",
        ])
        
        if transaction.subgraph_summary:
            lines.append(f"- {transaction.subgraph_summary}")
        
        return "\n".join(lines)
    
    def _build_cases_section(self, cases: List[Dict]) -> str:
        """Build the similar cases section."""
        lines = [
            "## Similar Historical Cases",
            "",
            "The following cases are most similar to the current transaction based on their",
            "feature patterns and network structure:",
            ""
        ]
        
        for i, case_data in enumerate(cases, 1):
            case = case_data.get('case')
            similarity = case_data.get('similarity', 0)
            
            lines.extend([
                f"### Case {i} (Similarity: {similarity:.2%})",
                f"**Verdict:** {case.true_label.upper()}",
                f"**Fraud Score:** {case.fraud_score:.1%}",
            ])
            
            # Add explanation if available
            explanation = case.explanation
            if explanation:
                if isinstance(explanation, dict):
                    narrative = explanation.get('narrative', '')
                    if narrative:
                        lines.append(f"**Analysis:** {narrative[:500]}...")
                    
                    top_features = explanation.get('important_features', [])
                    if top_features:
                        lines.append("**Key Features:**")
                        for feat in top_features[:3]:
                            if isinstance(feat, dict):
                                lines.append(f"  - {feat.get('name', 'Feature')}: {feat.get('importance', 0):.3f}")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def _build_task_section(self, task: str) -> str:
        """Build the task instruction section."""
        if task == 'risk_only':
            return self.templates.get_risk_assessment_task()
        elif task == 'recommendation_only':
            return self.templates.get_recommendation_task()
        else:
            return self.templates.get_full_analysis_task()
    
    def _get_risk_level(self, fraud_score: float) -> str:
        """Convert fraud score to risk level."""
        if fraud_score >= 0.9:
            return "VERY HIGH 🔴"
        elif fraud_score >= 0.7:
            return "HIGH 🟠"
        elif fraud_score >= 0.5:
            return "MODERATE 🟡"
        elif fraud_score >= 0.3:
            return "LOW 🟢"
        else:
            return "VERY LOW ⚪"
    
    def build_simple_prompt(
        self,
        node_idx: int,
        fraud_score: float,
        predicted_label: str,
        similar_cases: List[Dict]
    ) -> str:
        """
        Build a simplified prompt without full transaction context.
        
        Useful for quick analysis or when full context is not available.
        """
        lines = [
            self.templates.get_system_prompt(),
            "",
            "## Current Transaction",
            f"- ID: #{node_idx}",
            f"- Fraud Score: {fraud_score:.1%}",
            f"- Prediction: {predicted_label}",
            "",
            "## Similar Historical Cases",
        ]
        
        for i, case_data in enumerate(similar_cases[:self.max_cases], 1):
            case = case_data.get('case')
            similarity = case_data.get('similarity', 0)
            lines.append(
                f"{i}. Case {case.case_id}: {case.true_label} "
                f"(similarity: {similarity:.2%}, score: {case.fraud_score:.1%})"
            )
        
        lines.extend([
            "",
            self.templates.get_full_analysis_task()
        ])
        
        return "\n".join(lines)


def build_icl_prompt(
    node_idx: int,
    fraud_score: float,
    predicted_label: str,
    similar_cases: List[Dict],
    top_features: Optional[List[Dict]] = None,
    **kwargs
) -> str:
    """
    Convenience function to build ICL prompt.
    
    Args:
        node_idx: Transaction node index
        fraud_score: Model's fraud prediction
        predicted_label: Predicted class
        similar_cases: Retrieved similar cases
        top_features: Important features (optional)
        **kwargs: Additional context
        
    Returns:
        Complete prompt string
    """
    builder = ICLPromptBuilder()
    
    # Create transaction context
    transaction = TransactionContext(
        node_idx=node_idx,
        fraud_score=fraud_score,
        predicted_label=predicted_label,
        embedding=[],
        top_features=top_features or [],
        feature_summary="",
        num_neighbors=kwargs.get('num_neighbors', 0),
        in_degree=kwargs.get('in_degree', 0),
        out_degree=kwargs.get('out_degree', 0),
        subgraph_summary=kwargs.get('subgraph_summary', ''),
        explanation_narrative=kwargs.get('explanation', '')
    )
    
    return builder.build_prompt(transaction, similar_cases)


if __name__ == '__main__':
    # Test prompt builder
    print("Testing ICL Prompt Builder")
    print("="*50)
    
    # Create mock transaction
    transaction = TransactionContext(
        node_idx=12345,
        fraud_score=0.87,
        predicted_label='illicit',
        embedding=[],
        top_features=[
            {'name': 'transaction_volume', 'importance': 0.92, 'value': 847000},
            {'name': 'time_gap', 'importance': 0.85, 'value': 3},
            {'name': 'fan_out_degree', 'importance': 0.78, 'value': 47}
        ],
        feature_summary="High volume with rapid fan-out pattern",
        num_neighbors=15,
        in_degree=3,
        out_degree=12,
        subgraph_summary="Connected to known mixer service",
        explanation_narrative="Transaction shows layering behavior"
    )
    
    # Create mock similar cases
    from dataclasses import dataclass
    
    @dataclass
    class MockCase:
        case_id: int
        true_label: str
        fraud_score: float
        explanation: dict
    
    similar_cases = [
        {
            'case': MockCase(42, 'illicit', 0.92, {'narrative': 'Known mixer involvement'}),
            'similarity': 0.95
        },
        {
            'case': MockCase(187, 'licit', 0.12, {'narrative': 'Regular exchange deposit'}),
            'similarity': 0.88
        }
    ]
    
    # Build prompt
    builder = ICLPromptBuilder()
    prompt = builder.build_prompt(transaction, similar_cases)
    
    print("\nGenerated Prompt:")
    print("-"*50)
    print(prompt)
