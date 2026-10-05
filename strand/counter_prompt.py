"""
Module 3: Epistemic Inoculation Prompting (Mitigation)
Corresponds to Section 2.3 in STRAND (NAACL 2026).

When ungrounded edges are detected (eta(G) < eta_thresh), STRAND injects
an epistemic inoculation constraint to force instruction-compliant models
to reject ungrounded causal leaps.
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass, field
from .causal_dag import DAGAuditReport
from .belief_tracker import KineticStepTelemetry


@dataclass(frozen=True)
class InoculationPayload:
    """Inoculation output payload containing synthesized constraint and metadata."""
    is_triggered: bool
    trigger_reason: str
    inoculation_constraint: str
    unsupported_edges: List[Tuple[str, str]] = field(default_factory=list)


class EpistemicInoculator:
    """
    Implements Module 3: Context-preserving epistemic inoculation synthesis.
    
    Grounded in cognitive inoculation theory (Roozenbeek and van der Linden, 2019),
    this prompt constraint compels the LLM to separate verified premise facts from
    unsupported inductive leaps.
    """

    EXACT_PAPER_TEMPLATE: str = (
        "[STRAND Audit: The preceding inputs establish isolated factual statements "
        "{{s_i}}, but do not substantiate causal dependency {unsupported_dependency}. "
        "Formulate deductions strictly grounded in verified edges.]"
    )

    def __init__(self, prefix_tag: str = "[STRAND Audit]") -> None:
        self.prefix_tag = prefix_tag

    def synthesize(
        self,
        dag_report: Optional[DAGAuditReport] = None,
        kinetic_telemetry: Optional[KineticStepTelemetry] = None,
        target_hypothesis: Optional[str] = None
    ) -> InoculationPayload:
        """
        Synthesize epistemic inoculation constraint prompt.
        
        Args:
            dag_report: Audit report from Module 2.
            kinetic_telemetry: Telemetry from Module 1.
            target_hypothesis: Optional string of candidate target hypothesis.
            
        Returns:
            InoculationPayload with constraint string ready for prompt injection.
        """
        reasons = []
        unsupported = []

        if kinetic_telemetry and kinetic_telemetry.is_triggered:
            reasons.append(
                f"Acceleration surge detected (a_t={kinetic_telemetry.acceleration_at:.3f} > alpha_thresh)"
            )

        if dag_report and dag_report.is_manipulated:
            reasons.append(
                f"Topological provenance deficit: eta(G)={dag_report.provenance_ratio:.2f} < eta_thresh (0.50)"
            )
            unsupported = list(dag_report.pruned_edges)

        is_triggered = bool(dag_report and dag_report.is_manipulated)

        if not is_triggered:
            return InoculationPayload(
                is_triggered=False,
                trigger_reason="Provenance audit passed (eta(G) >= eta_thresh). No inoculation required.",
                inoculation_constraint="",
                unsupported_edges=[]
            )

        # Formulate explicit dependency representation
        if unsupported:
            u_sample, v_sample = unsupported[0]
            dep_str = f"'{u_sample[:40]}...' => '{v_sample[:40]}...'"
        elif target_hypothesis:
            dep_str = f"premises => '{target_hypothesis}'"
        else:
            dep_str = "u => v"

        constraint = self.EXACT_PAPER_TEMPLATE.format(unsupported_dependency=dep_str)

        return InoculationPayload(
            is_triggered=True,
            trigger_reason="; ".join(reasons),
            inoculation_constraint=constraint,
            unsupported_edges=unsupported
        )
