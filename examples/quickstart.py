"""Two synthetic retraining loops: one that settles, one that blows up.

Run:  python examples/quickstart.py
"""

from lyapwatch import EmbeddingDriftTracker, Regime, StabilityMonitor


def converging_loop():
    """Loss decays, eval climbs, embeddings settle — healthy retraining."""
    mon = StabilityMonitor({"loss": "min", "eval": "max", "drift": "min"})
    drift = EmbeddingDriftTracker()
    centroid = [1.0, 0.0, 0.0]
    print("=== converging loop ===")
    for k in range(24):
        # embeddings move less and less each round
        centroid = [centroid[0], centroid[1] + 0.3 * 0.7**k, 0.0]
        obs = {
            "loss": 0.9 * 0.85**k,
            "eval": 1.0 - 0.4 * 0.85**k,
            "drift": drift.update(centroid),
        }
        r = mon.update(obs)
        if k % 6 == 0 or r.regime is Regime.DIVERGING:
            print(f"  {r}")
    print(f"  final -> {r.regime.value} (rho={r.contraction_rate:.3f})\n")


def diverging_loop():
    """Loss creeps up geometrically — the monitor calls it early, with a horizon."""
    mon = StabilityMonitor({"loss": "min"}, v_threshold=4.0)
    print("=== diverging loop ===")
    first_alarm = None
    for k in range(22):
        r = mon.update({"loss": 0.25 * 1.18**k})
        if r.regime is Regime.DIVERGING and first_alarm is None:
            first_alarm = k
            print(f"  ALARM at step {k}: {r.message}")
    print(f"  first divergence alarm fired at step {first_alarm}")
    print(f"  final -> {r.regime.value} (rho={r.contraction_rate:.3f})")


if __name__ == "__main__":
    converging_loop()
    diverging_loop()
