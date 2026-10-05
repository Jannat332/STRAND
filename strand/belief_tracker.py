"""
Module 1: Kinetic Belief Tracking (Efficiency Gate)
Corresponds to Section 2.1, Equations (1) & (2) in STRAND (NAACL 2026).

Mathematical Formulation:
    Normalized stance:
        p_t = 2 * P_hat(H | S_<=t) - 1 in [-1.0, 1.0]
    Backward finite differences (Delta t = 1 turn):
        v_t = p_t - p_{t-1}
        a_t = p_t - 2*p_{t-1} + p_{t-2}
    Kinetic Anomaly Gating Function:
        I_kinetic(t) = I[a_t > alpha_thresh and v_t > 0.15]
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass(frozen=True)
class KineticStepTelemetry:
    """Telemetry record for a single discrete conversation turn t."""
    turn_index: int
    raw_confidence: float
    stance_pt: float
    velocity_vt: float
    acceleration_at: float
    is_triggered: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class KineticBeliefTracker:
    """
    Implements Module 1: Discrete Kinetic Belief Tracking.
    
    Serves strictly as an efficiency-oriented computational gate (Section 2.1)
    to bypass expensive Causal DAG extraction on steady-state benign turns.
    """

    def __init__(
        self,
        alpha_thresh: float = 0.15,
        velocity_floor: float = 0.15
    ) -> None:
        """
        Initialize the discrete kinetic belief tracker.
        
        Args:
            alpha_thresh: Calibrated second-order acceleration threshold (default: 0.15).
            velocity_floor: Velocity directional lower bound (default: 0.15).
        """
        self.alpha_thresh = float(alpha_thresh)
        self.velocity_floor = float(velocity_floor)
        self.trajectory: List[KineticStepTelemetry] = []

    def reset(self) -> None:
        """Clear sequential stance history for a new conversation stream."""
        self.trajectory.clear()

    @staticmethod
    def confidence_to_stance(confidence: float) -> float:
        """
        Normalize posterior probability into symmetric epistemic stance:
        p_t = 2 * P_hat(H | S_<=t) - 1 in [-1.0, 1.0]
        """
        prob = max(0.0, min(1.0, float(confidence)))
        return 2.0 * prob - 1.0

    def update(
        self,
        confidence: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> KineticStepTelemetry:
        """
        Evaluate discrete derivatives for turn t via backward finite differences.
        
        Args:
            confidence: Estimated model posterior P_hat(H | S_<=t) in [0.0, 1.0].
            metadata: Optional dictionary with turn provenance metadata.
            
        Returns:
            KineticStepTelemetry with computed (p_t, v_t, a_t, I_kinetic).
        """
        turn_t = len(self.trajectory)
        p_t = self.confidence_to_stance(confidence)
        meta = metadata or {}

        if turn_t == 0:
            v_t = 0.0
            a_t = 0.0
        elif turn_t == 1:
            p_prev = self.trajectory[0].stance_pt
            v_t = p_t - p_prev
            a_t = 0.0
        else:
            p_prev = self.trajectory[-1].stance_pt
            p_prev2 = self.trajectory[-2].stance_pt
            # Equation (1): Backward finite differences
            v_t = p_t - p_prev
            a_t = p_t - (2.0 * p_prev) + p_prev2

        # Equation (2): Operational kinetic anomaly trigger
        # I_kinetic(t) = I[a_t > alpha_thresh and v_t > 0.15]
        is_triggered = bool((a_t > self.alpha_thresh) and (v_t > self.velocity_floor))

        telemetry = KineticStepTelemetry(
            turn_index=turn_t,
            raw_confidence=float(confidence),
            stance_pt=float(p_t),
            velocity_vt=float(v_t),
            acceleration_at=float(a_t),
            is_triggered=is_triggered,
            metadata=meta
        )
        self.trajectory.append(telemetry)
        return telemetry

    @property
    def latest(self) -> Optional[KineticStepTelemetry]:
        """Return telemetry from the most recent turn."""
        return self.trajectory[-1] if self.trajectory else None
