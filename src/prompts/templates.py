"""
Prompt Templates for AML Analysis

Contains all prompt templates used for LLM-based fraud analysis.
"""


class PromptTemplates:
    """
    Collection of prompt templates for AML analysis.
    
    Provides templates for:
    - System prompts (analyst persona)
    - Task-specific instructions
    - Output format specifications
    """
    
    @staticmethod
    def get_system_prompt() -> str:
        """Get the system prompt defining the AML analyst role."""
        return """# AML Transaction Analyst

You are an expert Anti-Money Laundering (AML) analyst specializing in cryptocurrency 
transaction analysis. Your role is to:

1. **Analyze** suspicious Bitcoin transactions flagged by our ML system
2. **Compare** with historical cases to identify patterns
3. **Assess** the likelihood of illicit activity
4. **Recommend** appropriate actions

## Your Expertise:
- Deep knowledge of money laundering typologies (layering, structuring, mixing)
- Understanding of Bitcoin transaction patterns
- Experience with regulatory requirements (BSA/AML, FATF guidelines)
- Pattern recognition across historical fraud cases

## Important Guidelines:
- Base your analysis on the provided data and similar cases
- Provide clear reasoning for your assessment
- Consider both false positive and false negative risks
- Be specific about which patterns inform your decision"""

    @staticmethod
    def get_full_analysis_task() -> str:
        """Get the full analysis task instruction."""
        return """## Your Task

Based on the current transaction's features and the similar historical cases provided:

1. **Risk Assessment**: Evaluate the likelihood this transaction is illicit (High/Medium/Low)

2. **Pattern Analysis**: Identify which patterns from similar cases are present:
   - Transaction volume anomalies
   - Timing patterns (rapid succession, unusual hours)
   - Network patterns (fan-out, mixing, layering)
   - Connection to known illicit addresses

3. **Reasoning**: Explain your thought process, referencing:
   - Specific features that raised/lowered concern
   - How similar cases inform your assessment
   - Any contradicting evidence

4. **Recommendation**: Choose one and justify:
   - **FLAG** - High confidence illicit, immediate review required
   - **INVESTIGATE** - Suspicious patterns, needs additional analysis
   - **CLEAR** - Low risk, consistent with legitimate behavior
   - **MONITOR** - Borderline, add to watchlist

5. **Confidence**: Rate your confidence (1-10) and explain uncertainty factors

Please provide your analysis in a structured format."""

    @staticmethod
    def get_risk_assessment_task() -> str:
        """Get the risk-only task instruction."""
        return """## Your Task

Provide a quick risk assessment:

1. **Risk Level**: HIGH / MEDIUM / LOW
2. **Primary Reason**: One sentence explaining the main risk factor
3. **Confidence**: 1-10

Keep your response brief and focused."""

    @staticmethod
    def get_recommendation_task() -> str:
        """Get the recommendation-only task instruction."""
        return """## Your Task

Based on the analysis, provide your recommendation:

**Recommendation**: FLAG / INVESTIGATE / CLEAR / MONITOR

**Justification**: 2-3 sentences explaining your decision

**Next Steps**: What additional information would strengthen your assessment?"""

    @staticmethod
    def get_explanation_task() -> str:
        """Get the explanation generation task."""
        return """## Your Task

Generate a human-readable explanation for why this transaction was flagged.
The explanation should be:

1. **Clear**: Understandable by non-technical compliance officers
2. **Specific**: Reference actual patterns and features
3. **Actionable**: Suggest what to look for in manual review

Format your response as a brief investigation summary."""

    @staticmethod
    def get_comparison_prompt(case1: dict, case2: dict) -> str:
        """Get prompt for comparing two cases."""
        return f"""## Case Comparison Task

Compare these two transactions and explain their similarities and differences:

**Case A** (Current):
- Fraud Score: {case1.get('fraud_score', 'N/A')}
- Features: {case1.get('features', 'N/A')}

**Case B** (Historical):
- Verdict: {case2.get('label', 'N/A')}
- Fraud Score: {case2.get('fraud_score', 'N/A')}
- Features: {case2.get('features', 'N/A')}

Questions:
1. What patterns do these cases share?
2. What are the key differences?
3. Does Case B inform the risk assessment of Case A?"""

    @staticmethod
    def get_output_format() -> str:
        """Get the expected output format specification."""
        return """## Expected Output Format

```
RISK ASSESSMENT
===============
Level: [HIGH/MEDIUM/LOW]
Score: [Fraud probability interpretation]
Confidence: [1-10]

KEY FINDINGS
============
1. [Finding 1]
2. [Finding 2]
3. [Finding 3]

PATTERN ANALYSIS
================
[Comparison with similar cases]

RECOMMENDATION
==============
Action: [FLAG/INVESTIGATE/CLEAR/MONITOR]
Rationale: [2-3 sentences]

NEXT STEPS
==========
[Suggested follow-up actions]
```"""

    @staticmethod
    def format_case_for_prompt(case: dict) -> str:
        """Format a case object for inclusion in prompts."""
        lines = [
            f"**Case ID:** {case.get('case_id', 'N/A')}",
            f"**True Label:** {case.get('true_label', 'N/A')}",
            f"**Fraud Score:** {case.get('fraud_score', 0):.1%}",
        ]
        
        explanation = case.get('explanation', {})
        if explanation:
            if isinstance(explanation, dict):
                narrative = explanation.get('narrative', '')
                if narrative:
                    lines.append(f"**Analysis:** {narrative}")
        
        return "\n".join(lines)


# Singleton instance for easy access
TEMPLATES = PromptTemplates()


if __name__ == '__main__':
    # Print example templates
    print("System Prompt:")
    print("="*50)
    print(TEMPLATES.get_system_prompt())
    print("\n")
    
    print("Full Analysis Task:")
    print("="*50)
    print(TEMPLATES.get_full_analysis_task())
