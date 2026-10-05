import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strand.causal_dag import causal_supp, TypedRelation


def test_explicit_causal_support():
    # Template A: "B occurred because A took place." -> CAUSAL-SUPPORT (Gr = 1)
    premise = "The river flooded because heavy rainfall inundated the valley."
    edge = ("heavy rainfall inundated the valley", "river flooded")
    
    verdict = causal_supp(edge=edge, premises=[premise])
    assert verdict.is_grounded == 1
    assert verdict.relation_type == TypedRelation.CAUSAL_SUPPORT


def test_temporal_only_relation():
    # Template C: "A occurred strictly before B." -> TEMPORAL-ONLY (Gr = 0)
    premise = "The warning siren sounded before the tremor shook the district."
    edge = ("warning siren sounded", "tremor shook the district")
    
    verdict = causal_supp(edge=edge, premises=[premise])
    assert verdict.is_grounded == 0
    assert verdict.relation_type == TypedRelation.TEMPORAL_ONLY


def test_correlational_relation():
    # Template D: "A and B occurred concurrently." -> CORRELATIONAL (Gr = 0)
    premise = "Traffic delays increased and temperatures dropped across the city."
    edge = ("temperatures dropped", "traffic delays increased")
    
    verdict = causal_supp(edge=edge, premises=[premise])
    assert verdict.is_grounded == 0
    assert verdict.relation_type == TypedRelation.CORRELATIONAL


def test_unsupported_causal_leap():
    # Template E: Post-hoc leap without connector in source premises -> UNSUPPORTED (Gr = 0)
    premise = "Officer Ahmed was stationed outside the building."
    edge = ("Ahmed was stationed outside", "Ahmed was the primary conspirator")
    
    verdict = causal_supp(edge=edge, premises=[premise])
    assert verdict.is_grounded == 0
    assert verdict.relation_type == TypedRelation.UNSUPPORTED


if __name__ == "__main__":
    test_explicit_causal_support()
    test_temporal_only_relation()
    test_correlational_relation()
    test_unsupported_causal_leap()
    print("CausalSupp unit tests passed successfully.")
