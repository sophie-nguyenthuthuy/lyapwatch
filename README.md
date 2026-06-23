# lyapwatch

**Lyapunov stability monitoring for retraining loops.** Model a retraining
pipeline as a discrete dynamical system, estimate a Lyapunov function from its
metric stream, and raise a divergence alarm — with a projected horizon —
*before* the loop loses stability.

> Bounded delegation: a retraining loop is delegated authority to update a model
> on its own. `lyapwatch` is the bound. It certifies the loop is still
> contracting toward a fixed point and pulls the andon cord when it isn't.

Zero heavyweight dependencies — just `numpy`.

## Why

Autonomous retraining loops (RLHF, self-distillation, continual fine-tuning,
agentic data flywheels) can slide from *converging* into *diverging* — model
collapse, reward hacking, representational drift — and the usual per-step
metric thresholds only fire once damage is visible. Stability is not a level,
it's a **trend in an energy function**. `lyapwatch` watches that trend online.

## Install

```bash
pip install -e .          # from this repo
```

## Quickstart

```python
from lyapwatch import StabilityMonitor, Regime

mon = StabilityMonitor(
    channels={"loss": "min", "eval": "max", "drift": "min"},
    window=20,            # steps used for the contraction fit
    v_threshold=4.0,      # energy at which the loop is "lost"
)

for step in retraining_loop:
    report = mon.update({"loss": loss, "eval": eval_score, "drift": emb_drift})
    print(report)                                   # [step 12] CONVERGING: ...
    if report.regime is Regime.DIVERGING:
        stop_and_rollback(horizon=report.horizon)   # ~N steps to instability
```

Each `update()` returns a `StabilityReport`:

| field | meaning |
|-------|---------|
| `regime` | `WARMING_UP` · `CONVERGING` · `STALLED` · `DIVERGING` |
| `energy` | current Lyapunov energy `V_k ≥ 0` (0 at the targets) |
| `contraction_rate` | estimated `ρ` in `V_k ≈ V_0·ρ^k`; `<1` settling, `>1` blowing up |
| `drift` | mean one-step Lyapunov drift `ΔV` over the window |
| `confidence` | `R²` of the contraction fit, `[0,1]` |
| `horizon` | projected steps until `V` crosses `v_threshold` (only when diverging) |

`report.regime.is_actionable` is `True` for `STALLED` and `DIVERGING`.

## How it works

1. **Energy.** Each channel declares a direction (`"min"` / `"max"`) and an
   equilibrium target (default `0.0` / `1.0`). The one-sided error to target is
   normalized by the first step's error, and the energy is the weighted sum of
   squared normalized errors:

   ```
   V_k = Σ_i  w_i · ( error_i(m) / error_i(m₀) )²
   ```

   `V_k ≥ 0`, zero exactly at the targets — a valid candidate Lyapunov function.

2. **Contraction.** Fit `log V_k = a + s·k` over the trailing window;
   `ρ = exp(s)` is the per-step contraction factor (a discrete Lyapunov
   exponent). A drift-sign test guards against noisy single-window reads.

3. **Verdict.** A dead-band around `ρ = 1` separates converging / stalled /
   diverging. When diverging, solve `V·ρ^k = v_threshold` for the **horizon**.

## CLI

Replay a metrics CSV — useful as an Airflow / CI gate (non-zero exit when the
loop ends stalled or diverging):

```bash
lyapwatch run metrics.csv --min loss,drift --max eval --final
```

## Embedding drift

```python
from lyapwatch import EmbeddingDriftTracker

drift = EmbeddingDriftTracker(metric="cosine")
for ckpt in checkpoints:
    d = drift.update(probe_set_mean_embedding(ckpt))   # feed d as a "min" channel
```

## Run the tests / demo

```bash
pytest
python examples/quickstart.py
```

## License

MIT
