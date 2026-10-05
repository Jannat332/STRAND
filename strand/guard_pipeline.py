"""
STRAND Unified Defense Pipeline
Orchestrates Modules 1, 2, and 3 matching Figure 1 and Section 2 of STRAND (NAACL 2026).
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import time

from .belief_tracker import KineticBeliefTracker, KineticStepTelemetry
from .causal_dag import CausalDAGAuditor, DAGAuditReport
from .counter_prompt import EpistemicInoculator, InoculationPayload


@dataclass
class STRANDDecision:
    """Consolidated defense audit decision for an incoming conversational turn."""
    attack_detected: bool
    kinetic_gated: bool
    kinetic_telemetry: Optional[KineticStepTelemetry]
    dag_report: Optional[DAGAuditReport]
    inoculation_payload: Optional[InoculationPayload]
    execution_time_ms: float
    component_latencies_ms: Dict[str, float] = field(default_factory=dict)


class STRANDPipeline:
    """
    End-to-end STRAND architecture:
        Incoming factual stream {s_1, ..., s_T}
        -> Kinetic Gate (a_t > alpha_thresh)
        -> Causal DAG Provenance Audit (eta(G) < eta_thresh)
        -> Epistemic Inoculation Prompt Injection.
    """

    def __init__(
        self,
        alpha_thresh: float = 0.15,
        eta_thresh: float = 0.50,
        enable_kinetic_gate: bool = True
    ) -> None:
        """
        Initialize the STRAND defense pipeline.
        
        Args:
            alpha_thresh: Kinetic acceleration threshold (default: 0.15).
            eta_thresh: Topological causal provenance threshold (default: 0.50).
            enable_kinetic_gate: Whether to use Module 1 as computational gate.
        """
        self.alpha_thresh = alpha_thresh
        self.eta_thresh = eta_thresh
        self.enable_kinetic_gate = enable_kinetic_gate

        self.module1_kinetic = KineticBeliefTracker(alpha_thresh=alpha_thresh)
        self.module2_dag = CausalDAGAuditor(eta_thresh=eta_thresh)
        self.module3_inoculator = EpistemicInoculator()

    def reset(self) -> None:
        """Reset internal trajectory state for a new conversation session."""
        self.module1_kinetic.reset()

    def inspect_turn(
        self,
        confidence: float,
        rationale_text: str,
        premises: List[str],
        target_hypothesis: Optional[str] = None
    ) -> STRANDDecision:
        """
        Audit a single conversation turn through the cascaded STRAND pipeline.
        
        Args:
            confidence: Current model belief probability in [0.0, 1.0].
            rationale_text: Model Chain-of-Thought or elicited candidate reasoning.
            premises: List of verified factual input premises S_prem up to turn t.
            target_hypothesis: Optional target conclusion string.
            
        Returns:
            STRANDDecision containing audit verdict and component latencies.
        """
        t0 = time.perf_counter()
        latencies: Dict[str, float] = {}

        # ----------------------------------------------------------------------
        # Module 1: Kinetic Belief Tracking (Efficiency Gate)
        # ----------------------------------------------------------------------
        t_m1_start = time.perf_counter()
        kinetic_telemetry = self.module1_kinetic.update(confidence=confidence)
        latencies["module1_kinetic_ms"] = (time.perf_counter() - t_m1_start) * 1000.0

        # Evaluate gating decision
        must_run_dag = (not self.enable_kinetic_gate) or kinetic_telemetry.is_triggered

        dag_report: Optional[DAGAuditReport] = None
        inoculation: Optional[InoculationPayload] = None

        if not must_run_dag:
            # Gated turn: Graph extraction is bypassed on 87.5% of benign turns
            total_elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return STRANDDecision(
                attack_detected=False,
                kinetic_gated=True,
                kinetic_telemetry=kinetic_telemetry,
                dag_report=None,
                inoculation_payload=None,
                execution_time_ms=total_elapsed_ms,
                component_latencies_ms=latencies
            )

        # ----------------------------------------------------------------------
        # Module 2: Relation-Level Causal DAG Auditing (Primary Detector)
        # ----------------------------------------------------------------------
        t_m2_start = time.perf_counter()
        dag_report = self.module2_dag.audit(
            rationale_text=rationale_text,
            premises=premises
        )
        latencies["module2_dag_ms"] = (time.perf_counter() - t_m2_start) * 1000.0

        # ----------------------------------------------------------------------
        # Module 3: Epistemic Inoculation Prompting (Mitigation)
        # ----------------------------------------------------------------------
        t_m3_start = time.perf_counter()
        inoculation = self.module3_inoculator.synthesize(
            dag_report=dag_report,
            kinetic_telemetry=kinetic_telemetry,
            target_hypothesis=target_hypothesis
        )
        latencies["module3_inoculation_ms"] = (time.perf_counter() - t_m3_start) * 1000.0

        total_elapsed_ms = (time.perf_counter() - t0) * 1000.0

        attack_detected = bool(dag_report.is_manipulated)

        return STRANDDecision(
            attack_detected=attack_detected,
            kinetic_gated=False,
            kinetic_telemetry=kinetic_telemetry,
            dag_report=dag_report,
            inoculation_payload=inoculation,
            execution_time_ms=total_elapsed_ms,
            component_latencies_ms=latencies
        )
