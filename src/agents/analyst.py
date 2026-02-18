"""
Analyst Agent

Receives evidence from Coordinator and produces final risk assessment.
Generates human-readable verdicts with confidence scores and recommendations.
"""

import re
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .coordinator import InvestigationState
from ..llm.client import LLMClient


class RiskLevel(str, Enum):
    """Risk classification levels."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class Recommendation(str, Enum):
    """Investigation recommendations."""
    FLAG_IMMEDIATE = "FLAG_IMMEDIATE"  # Immediate escalation
    INVESTIGATE = "INVESTIGATE"         # Needs human review
    MONITOR = "MONITOR"                 # Add to watchlist
    CLEAR = "CLEAR"                     # No action needed


@dataclass
class Verdict:
    """Final verdict from Analyst Agent."""
    node_idx: int
    risk_level: RiskLevel
    confidence: float  # 0.0 to 1.0
    recommendation: Recommendation
    reasoning: str
    key_factors: List[str]
    similar_case_summary: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "node_idx": self.node_idx,
            "risk_level": self.risk_level.value,
            "confidence": self.confidence,
            "confidence_percentage": f"{self.confidence * 100:.0f}%",
            "recommendation": self.recommendation.value,
            "reasoning": self.reasoning,
            "key_factors": self.key_factors,
            "similar_case_summary": self.similar_case_summary,
            "timestamp": self.timestamp
        }
    
    def to_report(self) -> str:
        """Generate human-readable report."""
        emoji_map = {
            RiskLevel.CRITICAL: "🔴",
            RiskLevel.HIGH: "🟠",
            RiskLevel.MEDIUM: "🟡",
            RiskLevel.LOW: "🟢"
        }
        
        rec_action = {
            Recommendation.FLAG_IMMEDIATE: "ESCALATE IMMEDIATELY to compliance team",
            Recommendation.INVESTIGATE: "ASSIGN to analyst for detailed review",
            Recommendation.MONITOR: "ADD to watchlist for ongoing monitoring",
            Recommendation.CLEAR: "NO ACTION required at this time"
        }
        
        report = f"""
{'='*70}
TRANSACTION RISK ASSESSMENT REPORT
{'='*70}

Transaction ID: #{self.node_idx}
Assessment Date: {self.timestamp}

{emoji_map[self.risk_level]} RISK LEVEL: {self.risk_level.value}
Confidence: {self.confidence * 100:.0f}%

RECOMMENDATION: {self.recommendation.value}
Action: {rec_action[self.recommendation]}

KEY RISK FACTORS:
{chr(10).join(f'  • {factor}' for factor in self.key_factors)}

SIMILAR CASE ANALYSIS:
{self.similar_case_summary}

DETAILED REASONING:
{self.reasoning}

{'='*70}
"""
        return report


class AnalystAgent:
    """
    Analyst Agent for AML Risk Assessment.
    
    Receives evidence collected by the Coordinator Agent and produces
    a final risk assessment with:
    - Risk level classification (CRITICAL/HIGH/MEDIUM/LOW)
    - Confidence score (0-100%)
    - Recommendation (FLAG/INVESTIGATE/MONITOR/CLEAR)
    - Human-readable reasoning
    
    Args:
        llm: LLMClient for reasoning
        
    Usage:
        analyst = AnalystAgent(llm)
        verdict = analyst.assess(investigation_state)
        print(verdict.to_report())
    """
    
    SYSTEM_PROMPT = """You are a Senior AML (Anti-Money Laundering) Analyst.

Your role is to review evidence collected during transaction investigations and provide:
1. A risk level assessment (CRITICAL, HIGH, MEDIUM, or LOW)
2. A confidence score (0-100%)
3. A recommendation (FLAG_IMMEDIATE, INVESTIGATE, MONITOR, or CLEAR)
4. Clear reasoning explaining your decision

Guidelines:
- CRITICAL (90-100% confidence): Clear evidence of illicit activity, multiple red flags
- HIGH (70-89%): Strong indicators of suspicious activity, similar to known illicit cases
- MEDIUM (40-69%): Some concerning patterns but mixed signals
- LOW (0-39%): Minimal risk indicators, consistent with legitimate activity

Recommendations:
- FLAG_IMMEDIATE: CRITICAL or HIGH risk with high confidence → immediate escalation
- INVESTIGATE: MEDIUM-HIGH risk needing human review
- MONITOR: LOW-MEDIUM risk, add to watchlist
- CLEAR: LOW risk with high confidence, no action needed

Be objective and base your assessment solely on the evidence provided.
"""

    ASSESSMENT_PROMPT = """Review the following evidence for Transaction #{node_idx}:

=== EVIDENCE SUMMARY ===
{evidence_summary}

=== INVESTIGATION STEPS ===
{investigation_steps}

Based on this evidence, provide your risk assessment.

Respond in the following JSON format:
{{
    "risk_level": "CRITICAL|HIGH|MEDIUM|LOW",
    "confidence": 0.85,
    "recommendation": "FLAG_IMMEDIATE|INVESTIGATE|MONITOR|CLEAR",
    "key_factors": ["factor 1", "factor 2", "factor 3"],
    "similar_case_summary": "Summary of how similar cases were resolved",
    "reasoning": "Detailed explanation of your assessment"
}}
"""

    def __init__(self, llm: LLMClient, verbose: bool = True):
        self.llm = llm
        self.verbose = verbose
    
    def assess(self, state: InvestigationState) -> Verdict:
        """
        Produce risk assessment from investigation evidence.
        
        Args:
            state: InvestigationState from Coordinator Agent
            
        Returns:
            Verdict with risk level, confidence, and recommendation
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"ANALYST: Reviewing evidence for Transaction #{state.node_idx}")
            print(f"{'='*60}")
        
        # Build evidence summary
        evidence_summary = self._build_evidence_summary(state)
        investigation_steps = self._build_steps_summary(state)
        
        # Get LLM assessment
        prompt = self.ASSESSMENT_PROMPT.format(
            node_idx=state.node_idx,
            evidence_summary=evidence_summary,
            investigation_steps=investigation_steps
        )
        
        response = self.llm.call(prompt, system_prompt=self.SYSTEM_PROMPT)
        
        # Parse response
        verdict = self._parse_verdict(state.node_idx, response.content, state)
        
        if self.verbose:
            print(f"\nRisk Level: {verdict.risk_level.value}")
            print(f"Confidence: {verdict.confidence * 100:.0f}%")
            print(f"Recommendation: {verdict.recommendation.value}")
        
        return verdict
    
    def _build_evidence_summary(self, state: InvestigationState) -> str:
        """Build formatted evidence summary."""
        lines = []
        
        # Fraud score
        if 'get_fraud_score' in state.evidence:
            fs = state.evidence['get_fraud_score']
            lines.append(f"FRAUD SCORE: {fs.get('fraud_percentage', 'N/A')} ({fs.get('risk_level', 'N/A')} risk)")
            lines.append(f"  - Model prediction: {fs.get('predicted_label', 'N/A')}")
            lines.append(f"  - True label (if known): {fs.get('true_label', 'unknown')}")
        
        # Similar cases
        if 'retrieve_similar_cases' in state.evidence:
            sc = state.evidence['retrieve_similar_cases']
            dist = sc.get('label_distribution', {})
            lines.append(f"\nSIMILAR CASES: {sc.get('num_retrieved', 0)} retrieved")
            lines.append(f"  - Illicit matches: {dist.get('illicit', 0)}")
            lines.append(f"  - Licit matches: {dist.get('licit', 0)}")
            lines.append(f"  - Majority verdict: {sc.get('majority_verdict', 'N/A')}")
            
            for case in sc.get('similar_cases', [])[:3]:
                lines.append(f"  - Case {case['case_id']}: {case['true_label']} ({case['similarity_percentage']} similar)")
        
        # Network context
        if 'get_network_context' in state.evidence:
            nc = state.evidence['get_network_context']
            lines.append(f"\nNETWORK CONTEXT:")
            lines.append(f"  - Total connections: {nc.get('num_neighbors', 0)}")
            lines.append(f"  - High-risk neighbors: {nc.get('high_risk_neighbors', 0)}")
            lines.append(f"  - Known illicit neighbors: {nc.get('known_illicit_neighbors', 0)}")
            lines.append(f"  - Network risk: {nc.get('network_risk', 'N/A')}")
        
        # Feature explanation
        if 'explain_prediction' in state.evidence:
            exp = state.evidence['explain_prediction']
            lines.append(f"\nFEATURE ANALYSIS:")
            lines.append(f"  - Method: {exp.get('method', 'N/A')}")
            lines.append(f"  - Top features: {exp.get('explanation_summary', 'N/A')}")
        
        return "\n".join(lines) if lines else "No evidence collected."
    
    def _build_steps_summary(self, state: InvestigationState) -> str:
        """Build summary of investigation steps."""
        if not state.steps:
            return "No investigation steps recorded."
        
        lines = []
        for step in state.steps:
            lines.append(f"Step {step.step_num}: {step.action}")
            lines.append(f"  Thought: {step.thought[:100]}...")
        
        return "\n".join(lines)
    
    def _parse_verdict(
        self,
        node_idx: int,
        response: str,
        state: InvestigationState
    ) -> Verdict:
        """Parse LLM response into Verdict object."""
        # Try to extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response)
        
        if json_match:
            try:
                data = json.loads(json_match.group())
                return Verdict(
                    node_idx=node_idx,
                    risk_level=RiskLevel(data.get('risk_level', 'MEDIUM')),
                    confidence=float(data.get('confidence', 0.5)),
                    recommendation=Recommendation(data.get('recommendation', 'INVESTIGATE')),
                    reasoning=data.get('reasoning', 'Assessment based on collected evidence.'),
                    key_factors=data.get('key_factors', ['ML fraud score', 'Similar case analysis']),
                    similar_case_summary=data.get('similar_case_summary', 'See similar cases in evidence.')
                )
            except (json.JSONDecodeError, ValueError, KeyError):
                pass
        
        # Fallback: rule-based assessment from evidence
        return self._rule_based_verdict(node_idx, state)
    
    def _rule_based_verdict(self, node_idx: int, state: InvestigationState) -> Verdict:
        """Generate verdict using rule-based logic as fallback."""
        fraud_score = 0.5
        risk_level = RiskLevel.MEDIUM
        key_factors = []
        
        # Get fraud score
        if 'get_fraud_score' in state.evidence:
            fs = state.evidence['get_fraud_score']
            fraud_score = fs.get('fraud_score', 0.5)
            if fraud_score >= 0.8:
                risk_level = RiskLevel.CRITICAL
                key_factors.append(f"Very high fraud score ({fraud_score*100:.0f}%)")
            elif fraud_score >= 0.6:
                risk_level = RiskLevel.HIGH
                key_factors.append(f"High fraud score ({fraud_score*100:.0f}%)")
            elif fraud_score >= 0.4:
                risk_level = RiskLevel.MEDIUM
                key_factors.append(f"Moderate fraud score ({fraud_score*100:.0f}%)")
            else:
                risk_level = RiskLevel.LOW
                key_factors.append(f"Low fraud score ({fraud_score*100:.0f}%)")
        
        # Check similar cases
        similar_summary = "No similar cases retrieved."
        if 'retrieve_similar_cases' in state.evidence:
            sc = state.evidence['retrieve_similar_cases']
            dist = sc.get('label_distribution', {})
            illicit = dist.get('illicit', 0)
            licit = dist.get('licit', 0)
            
            if illicit > licit:
                key_factors.append(f"Majority of similar cases ({illicit}/{illicit+licit}) were illicit")
                if risk_level == RiskLevel.MEDIUM:
                    risk_level = RiskLevel.HIGH
            elif licit > illicit:
                key_factors.append(f"Majority of similar cases ({licit}/{illicit+licit}) were licit")
            
            similar_summary = f"{illicit} illicit and {licit} licit similar cases found."
        
        # Check network context
        if 'get_network_context' in state.evidence:
            nc = state.evidence['get_network_context']
            if nc.get('known_illicit_neighbors', 0) > 0:
                key_factors.append(f"Connected to {nc['known_illicit_neighbors']} known illicit transaction(s)")
                if risk_level in [RiskLevel.MEDIUM, RiskLevel.LOW]:
                    risk_level = RiskLevel.HIGH
            if nc.get('high_risk_neighbors', 0) >= 2:
                key_factors.append(f"Multiple high-risk neighbors ({nc['high_risk_neighbors']})")
        
        # Determine recommendation
        if risk_level == RiskLevel.CRITICAL:
            recommendation = Recommendation.FLAG_IMMEDIATE
        elif risk_level == RiskLevel.HIGH:
            recommendation = Recommendation.INVESTIGATE
        elif risk_level == RiskLevel.MEDIUM:
            recommendation = Recommendation.MONITOR
        else:
            recommendation = Recommendation.CLEAR
        
        # Calculate confidence
        evidence_count = len(state.evidence)
        confidence = min(0.9, 0.5 + (evidence_count * 0.1))
        
        return Verdict(
            node_idx=node_idx,
            risk_level=risk_level,
            confidence=confidence,
            recommendation=recommendation,
            reasoning=f"Assessment based on {evidence_count} evidence sources. " + 
                      f"Fraud score: {fraud_score*100:.0f}%. " +
                      f"Similar cases suggest {state.evidence.get('retrieve_similar_cases', {}).get('majority_verdict', 'mixed')} pattern.",
            key_factors=key_factors if key_factors else ["Insufficient evidence for detailed analysis"],
            similar_case_summary=similar_summary
        )
