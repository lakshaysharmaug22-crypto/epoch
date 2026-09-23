import numpy as np

from epoch.evolution.pareto import hypervolume, non_dominated, normalize
from epoch.evolution.race import summarize, trials_to
from epoch.objectives import Constraint, Objective


def test_non_dominated_and_hv_2d():
    pts = np.array([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1], [0.6, 0.6]])
    assert non_dominated(pts).tolist() == [True, True, True, False]
    # ref 1.1: exact union of rectangles
    expected = (1.1 - 0.1) * (1.1 - 0.9) + (1.1 - 0.5) * (0.9 - 0.5) + (1.1 - 0.9) * (0.5 - 0.1)
    assert abs(hypervolume(pts) - expected) < 1e-12


def test_hv_3d_monotone():
    rng = np.random.default_rng(0)
    pts = rng.random((30, 3))
    assert hypervolume(pts[:10]) <= hypervolume(pts[:20]) <= hypervolume(pts) + 1e-12


def test_normalize_directions_and_log():
    objs = [Objective("q", "max", "q", lo=0.5, hi=1.0), Objective("lat", "min", "l", lo=1, hi=1000, log=True)]
    n = normalize(np.array([[1.0, 1.0], [0.5, 1000.0]]), objs)
    assert np.allclose(n, [[0.0, 0.0], [1.0, 1.0]])


def test_constraint_violation_sign():
    c = Constraint("p95", "<=", 100)
    assert c.violation({"p95": 50}) < 0 < c.violation({"p95": 150})
    assert Constraint("f1", ">=", 0.8).violation({"f1": 0.7}) > 0


def test_race_summary_headline():
    curves = {"nsga2": [[0.1, 0.2, 0.5, 0.6]], "agent": [[0.2, 0.6, 0.6, 0.7]]}
    s = summarize(curves, budget=4)
    assert trials_to([0.1, 0.5], 0.5) == 2 and trials_to([0.1], 0.5) is None
    assert s["headline"]["agent_trials_to_nsga2_final"] == 2
    assert s["headline"]["nsga2_trials_to_own_final"] == 4
    assert s["headline"]["pct_fewer_trials"] == 50.0
