"""
STRAND: Semantic Topological Reasoning and Accelerated Narrative Defense
Official Implementation for NAACL 2026 Submission (Paper ID: NAACL-2026-STRAND v12.0)

Modules:
    Module 1: KineticBeliefTracker (Section 2.1, Equations 1 & 2)
    Module 2: CausalDAGAuditor & causal_supp (Section 2.2, Equation 3, Algorithm 1)
    Module 3: EpistemicInoculator (Section 2.3)
    Pipeline: STRANDPipeline (Figure 1)
"""

from .belief_tracker import KineticBeliefTracker, KineticStepTelemetry
from .causal_dag import CausalDAGAuditor, CausalSuppVerdict, TypedRelation, causal_supp
from .counter_prompt import EpistemicInoculator, InoculationPayload
from .guard_pipeline import STRANDPipeline, STRANDDecision

__all__ = [
    "KineticBeliefTracker",
    "KineticStepTelemetry",
    "CausalDAGAuditor",
    "CausalSuppVerdict",
    "TypedRelation",
    "causal_supp",
    "EpistemicInoculator",
    "InoculationPayload",
    "STRANDPipeline",
    "STRANDDecision",
]
