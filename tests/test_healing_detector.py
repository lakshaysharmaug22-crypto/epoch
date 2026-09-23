import numpy as np

from epoch.healing.detector import Cusum
from epoch.healing.drift import char_hist, js_divergence


def test_cusum_rarely_false_alarms_and_fires_fast_on_shift():
    false_alarms, delays = 0, []
    for seed in range(200):
        rng = np.random.default_rng(seed)
        d = Cusum("p95", "up").fit(list(10 + rng.normal(0, 1, 40)))
        false_alarms += any(d.update(x)[1] for x in 10 + rng.normal(0, 1, 40))
        d.reset()
        fired = [d.update(x)[1] for x in 13 + rng.normal(0, 1, 10)]
        delays.append(fired.index(True) if any(fired) else 99)
    assert false_alarms / 200 <= 0.05  # ≈2–3% expected over 40 in-control windows with a 40-window baseline
    assert np.median(delays) <= 4  # a 3σ shift is caught within a few windows


def test_cusum_down_direction():
    d = Cusum("f1", "down").fit([0.9, 0.91, 0.89, 0.9, 0.9, 0.92, 0.88])
    assert any(d.update(0.75)[1] for _ in range(5))


def test_js_divergence_detects_new_phrasing():
    a = ["disbursement pending for loan", "kyc documents missing"] * 20
    b = ["paisa nahi aaya abhi tak", "kaunse docs chahiye salaried"] * 20
    assert js_divergence(char_hist(a), char_hist(a)) < 1e-6
    assert js_divergence(char_hist(a), char_hist(b)) > 0.2
