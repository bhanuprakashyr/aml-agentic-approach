"""
Coordinator Agent

Plans and executes AML investigation using available tools.
Implements ReAct (Reason-Act-Observe) pattern for autonomous investigation.
"""

import re
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

from .tools import AMLTools
from ..llm.client import LLMClient


@dataclass
class InvestigationStep:
    """Record of a single investigation step."""
    step_num: int
    thought: str
    action: str
    action_input: Dict
    observation: Dict
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass 
class InvestigationState:
    """State of an ongoing investigation."""
    node_idx: int
    steps: List[InvestigationStep] = field(default_factory=list)
    evidence: Dict = field(default_factory=dict)
    status: str = "in_progress"  # in_progress, complete, error
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None


class CoordinatorAgent:
    """
    Coordinator Agent for AML Investigation.
    
    Plans the investigation strategy, selects appropriate tools,
    and gathers evidence through iterative reasoning.
    
    Implements ReAct pattern:
        1. THOUGHT: Reason about what information is needed
        2. ACTION: Select and invoke a tool
        3. OBSERVATION: Process tool output
        4. Repeat until sufficient evidence gathered
    
    Args:
        tools: AMLTools instance with registered tools
        llm: LLMClient for reasoning
        max_steps: Maximum reasoning steps (default: 5)
        
    Usage:
        coordinator = CoordinatorAgent(tools, llm)
        evidence = coordinator.investigate(node_idx=12345)
    """
    
    SYSTEM_PROMPT = """You are an AML (Anti-Money Laundering) Investigation Coordinator.

Your role is to systematically investigate suspicious Bitcoin transactions by:
1. Gathering relevant evidence using available tools
2. Building a comprehensive risk profile
3. Collecting enough information for a final verdict

You follow the ReAct framework:
- THOUGHT: Reason about what information you need next
- ACTION: Choose a tool to gather that information
- OBSERVATION: Analyze the tool's output

Be methodical and thorough. Always start with the fraud score, then gather supporting evidence.
When you have enough evidence (fraud score, similar cases, network context), indicate you are DONE.

{tool_descriptions}

Response Format:
THOUGHT: [Your reasoning about what to do next]
ACTION: [tool_name]
ACTION_INPUT: {{"param": "value"}}

When investigation is complete:
THOUGHT: [Summary of evidence gathered]
ACTION: DONE
ACTION_INPUT: {{}}
"""

    PLANNING_PROMPT = """You are investigating transaction #{node_idx}.

Current Evidence Collected:
{evidence_summary}

Investigation History:
{history_summary}

What should you do next? Remember to use the available tools to gather evidence.
If you have collected: (1) fraud score, (2) similar cases, and (3) network context, 
you likely have enough evidence and should indicate DONE.

{tool_descriptions}

Respond with your next step:
THOUGHT: [Your reasoning]
ACTION: [tool_name or DONE]
ACTION_INPUT: {{"param": "value"}}
"""

    def __init__(
        self,
        tools: AMLTools,
        llm: LLMClient,
        max_steps: int = 5,
        verbose: bool = True
    ):
        self.tools = tools
        self.llm = llm
        self.max_steps = max_steps
        self.verbose = verbose
    
    def investigate(self, node_idx: int) -> InvestigationState:
        """
        Run full investigation on a transaction.
        
        Args:
            node_idx: Transaction node index to investigate
            
        Returns:
            InvestigationState with all collected evidence
        """
        state = InvestigationState(node_idx=node_idx)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"COORDINATOR: Starting investigation of Transaction #{node_idx}")
            print(f"{'='*60}")
        
        for step_num in range(1, self.max_steps + 1):
            if self.verbose:
                print(f"\n--- Step {step_num}/{self.max_steps} ---")
            
            # Get next action from LLM
            prompt = self._build_prompt(state)
            response = self.llm.call(
                prompt,
                system_prompt=self.SYSTEM_PROMPT.format(
                    tool_descriptions=self.tools.tool_descriptions
                )
            )
            
            # Parse the response
            parsed = self._parse_response(response.content)
            
            if self.verbose:
                print(f"THOUGHT: {parsed['thought']}")
                print(f"ACTION: {parsed['action']}")
            
            # Check if investigation is complete
            if parsed['action'].upper() == 'DONE':
                state.status = "complete"
                state.completed_at = datetime.now().isoformat()
                if self.verbose:
                    print("\nCOORDINATOR: Investigation complete. Passing to Analyst.")
                break
            
            # Execute the tool
            tool_result = self.tools.execute(
                parsed['action'],
                **parsed['action_input']
            )
            
            if self.verbose:
                print(f"OBSERVATION: {self._summarize_observation(tool_result)}")
            
            # Record the step
            step = InvestigationStep(
                step_num=step_num,
                thought=parsed['thought'],
                action=parsed['action'],
                action_input=parsed['action_input'],
                observation=tool_result
            )
            state.steps.append(step)
            
            # Update evidence
            if tool_result.get('success', False):
                state.evidence[parsed['action']] = tool_result['result']
        
        # If max steps reached without DONE
        if state.status != "complete":
            state.status = "complete"
            state.completed_at = datetime.now().isoformat()
            if self.verbose:
                print("\nCOORDINATOR: Max steps reached. Proceeding with available evidence.")
        
        return state
    
    def _build_prompt(self, state: InvestigationState) -> str:
        """Build prompt for the next reasoning step."""
        # Summarize current evidence
        evidence_lines = []
        for tool_name, result in state.evidence.items():
            evidence_lines.append(f"- {tool_name}: {self._summarize_evidence(result)}")
        
        evidence_summary = "\n".join(evidence_lines) if evidence_lines else "No evidence collected yet."
        
        # Summarize history
        history_lines = []
        for step in state.steps[-3:]:  # Last 3 steps
            history_lines.append(f"Step {step.step_num}: {step.action} → {self._summarize_observation(step.observation)}")
        
        history_summary = "\n".join(history_lines) if history_lines else "No previous steps."
        
        return self.PLANNING_PROMPT.format(
            node_idx=state.node_idx,
            evidence_summary=evidence_summary,
            history_summary=history_summary,
            tool_descriptions=self.tools.tool_descriptions
        )
    
    def _parse_response(self, response: str) -> Dict:
        """Parse LLM response into structured action."""
        # Default values
        parsed = {
            'thought': '',
            'action': 'DONE',
            'action_input': {}
        }
        
        # Extract THOUGHT
        thought_match = re.search(r'THOUGHT:\s*(.+?)(?=ACTION:|$)', response, re.DOTALL | re.IGNORECASE)
        if thought_match:
            parsed['thought'] = thought_match.group(1).strip()
        
        # Extract ACTION
        action_match = re.search(r'ACTION:\s*(\w+)', response, re.IGNORECASE)
        if action_match:
            parsed['action'] = action_match.group(1).strip()
        
        # Extract ACTION_INPUT
        input_match = re.search(r'ACTION_INPUT:\s*(\{.+?\})', response, re.DOTALL | re.IGNORECASE)
        if input_match:
            try:
                parsed['action_input'] = json.loads(input_match.group(1))
            except json.JSONDecodeError:
                # Try to extract simple parameters
                if 'node_idx' in response.lower():
                    idx_match = re.search(r'node_idx["\s:]+(\d+)', response)
                    if idx_match:
                        parsed['action_input'] = {'node_idx': int(idx_match.group(1))}
        
        # If no action_input but we have node_idx in state, use it
        if not parsed['action_input'] and parsed['action'] != 'DONE':
            # This will be filled by the investigate method
            pass
        
        return parsed
    
    def _summarize_evidence(self, result: Dict) -> str:
        """Create brief summary of evidence."""
        if 'fraud_score' in result:
            return f"Score: {result['fraud_score']}, Risk: {result.get('risk_level', 'N/A')}"
        elif 'similar_cases' in result:
            dist = result.get('label_distribution', {})
            return f"{dist.get('illicit', 0)} illicit, {dist.get('licit', 0)} licit matches"
        elif 'network_risk' in result:
            return f"Network risk: {result['network_risk']}, {result.get('num_neighbors', 0)} neighbors"
        elif 'top_features' in result:
            return f"Top features: {[f['feature_index'] for f in result['top_features'][:3]]}"
        else:
            return str(result)[:100]
    
    def _summarize_observation(self, observation: Dict) -> str:
        """Create brief summary of tool observation."""
        if observation.get('success'):
            result = observation.get('result', {})
            return self._summarize_evidence(result)
        else:
            return f"Error: {observation.get('error', 'Unknown error')}"
