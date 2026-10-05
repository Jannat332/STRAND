import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strand.belief_tracker import KineticBeliefTracker


def test_finite_difference_derivatives():
    tracker = KineticBeliefTracker(alpha_thresh=0.15, velocity_floor=0.15)
    
    # Sequence of beliefs P_hat: 0.50, 0.55, 0.90
    # Stances p_t: 2*0.5-1 = 0.0; 2*0.55-1 = 0.10; 2*0.90-1 = 0.80
    t0 = tracker.update(0.50)
    assert t0.stance_pt == 0.0
    assert t0.velocity_vt == 0.0
    assert t0.acceleration_at == 0.0
    assert not t0.is_triggered

    t1 = tracker.update(0.55)
    assert round(t1.stance_pt, 2) == 0.10
    assert round(t1.velocity_vt, 2) == 0.10
    assert t1.acceleration_at == 0.0
    assert not t1.is_triggered

    t2 = tracker.update(0.90)
    # v_2 = 0.80 - 0.10 = 0.70
    # a_2 = 0.80 - 2*(0.10) + 0.0 = 0.60
    assert round(t2.stance_pt, 2) == 0.80
    assert round(t2.velocity_vt, 2) == 0.70
    assert round(t2.acceleration_at, 2) == 0.60
    # a_2 = 0.60 > 0.15 and v_2 = 0.70 > 0.15 -> trigger
    assert t2.is_triggered is True


def test_gradual_creeping_update_bypasses_gate():
    tracker = KineticBeliefTracker(alpha_thresh=0.15, velocity_floor=0.15)
    # Gradual, slow linear movement (Attack A in Section 3.5 & Table 5)
    # Stances: 0.0, 0.05, 0.10, 0.15
    for conf in [0.50, 0.525, 0.55, 0.575]:
        t = tracker.update(conf)
        # a_t should be approximately 0.0 <= alpha_thresh
        assert abs(t.acceleration_at) <= 0.05
        assert t.is_triggered is False


if __name__ == "__main__":
    test_finite_difference_derivatives()
    test_gradual_creeping_update_bypasses_gate()
    print("Kinetic tracking unit tests passed successfully.")
