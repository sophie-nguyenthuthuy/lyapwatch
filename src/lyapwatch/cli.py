"""Command-line front-end: replay a metrics CSV through the monitor.

    lyapwatch run metrics.csv --min loss,drift --max eval

The CSV must have a header row. Every named channel column is read as a float;
all other columns are ignored. A verdict is printed per step (or just the final
verdict with ``--final``), and the exit code is non-zero if the loop ended in a
stalled or diverging regime — handy as a CI / Airflow gate.
"""

from __future__ import annotations

import argparse
import csv
import sys
from typing import Iterable, Optional, Sequence

from . import __version__
from .monitor import StabilityMonitor
from .regime import Regime


def _split(spec: Optional[str]) -> list[str]:
    return [c.strip() for c in spec.split(",") if c.strip()] if spec else []


def _build_monitor(args: argparse.Namespace) -> StabilityMonitor:
    channels: dict[str, str] = {}
    for name in _split(args.min):
        channels[name] = "min"
    for name in _split(args.max):
        channels[name] = "max"
    if not channels:
        raise SystemExit("error: declare at least one channel via --min / --max")
    return StabilityMonitor(
        channels,
        window=args.window,
        v_threshold=args.threshold,
    )


def _rows(path: str, channels: Iterable[str]) -> Iterable[dict[str, float]]:
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        missing = set(channels) - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"error: CSV is missing columns: {sorted(missing)}")
        for i, row in enumerate(reader, 1):
            try:
                yield {c: float(row[c]) for c in channels}
            except (TypeError, ValueError) as exc:
                raise SystemExit(f"error: non-numeric value on data row {i}: {exc}")


def run(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="lyapwatch",
        description="Replay a retraining metrics CSV through the stability monitor.",
    )
    parser.add_argument("--version", action="version", version=f"lyapwatch {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run the monitor over a metrics CSV")
    run_p.add_argument("csv", help="path to a metrics CSV with a header row")
    run_p.add_argument("--min", help="comma-separated lower-is-better columns (e.g. loss,drift)")
    run_p.add_argument("--max", help="comma-separated higher-is-better columns (e.g. eval)")
    run_p.add_argument("--window", type=int, default=20, help="contraction-fit window (default 20)")
    run_p.add_argument("--threshold", type=float, default=4.0, help="instability energy threshold")
    run_p.add_argument("--final", action="store_true", help="print only the final verdict")

    args = parser.parse_args(argv)

    monitor = _build_monitor(args)
    last = None
    for obs in _rows(args.csv, monitor.channels):
        last = monitor.update(obs)
        if not args.final:
            print(last)

    if last is None:
        raise SystemExit("error: CSV had no data rows")

    if args.final:
        print(last)

    return 1 if last.regime.is_actionable else 0


def main() -> None:  # console-script entry point
    sys.exit(run())


if __name__ == "__main__":  # pragma: no cover
    main()
