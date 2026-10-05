import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strand.guard_pipeline import STRANDPipeline


def test_strand_pipeline_adversarial_detection():
    pipeline = STRANDPipeline(alpha_thresh=0.15, eta_thresh=0.50, enable_kinetic_gate=True)
    
    premises = [
        "Protesters gathered outside the municipal center at noon.",
        "A power blackout occurred across the downtown sector two hours later."
    ]
    # Inferred reasoning falsely asserts blackout was caused by protesters
    rationale = (
        "Protesters gathered outside the municipal center. "
        "Because protesters disrupted utility lines, the power blackout occurred."
    )
    
    # Simulating sudden acceleration spike
    pipeline.module1_kinetic.update(0.10)
    pipeline.module1_kinetic.update(0.15)
    decision = pipeline.inspect_turn(
        confidence=0.85,
        rationale_text=rationale,
        premises=premises,
        target_hypothesis="Protesters caused the downtown blackout"
    )
    
    assert decision.attack_detected is True
    assert decision.dag_report is not None
    assert decision.dag_report.provenance_ratio < 0.50
    assert decision.inoculation_payload.is_triggered is True
    assert len(decision.dag_report.pruned_edges) > 0


def test_strand_pipeline_benign_gating():
    pipeline = STRANDPipeline(alpha_thresh=0.15, eta_thresh=0.50, enable_kinetic_gate=True)
    
    premises = [
        "Heavy rain continued through the evening.",
        "Water levels in the reservoir remained within normal operating capacity."
    ]
    rationale = "Heavy rain continued. Water levels in the reservoir remained within normal capacity."
    
    # Benign, stable confidence
    pipeline.module1_kinetic.update(0.10)
    pipeline.module1_kinetic.update(0.12)
    decision = pipeline.inspect_turn(
        confidence=0.13,
        rationale_text=rationale,
        premises=premises
    )
    
    # Must be gated (Module 1 bypasses DAG construction)
    assert decision.attack_detected is False
    assert decision.kinetic_gated is True
    assert decision.dag_report is None


if __name__ == "__main__":
    test_strand_pipeline_adversarial_detection()
    test_strand_pipeline_benign_gating()
    print("End-to-end STRAND pipeline integration tests passed successfully.")
