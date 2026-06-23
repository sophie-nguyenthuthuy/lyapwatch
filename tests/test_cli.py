import textwrap

from lyapwatch.cli import run


def _write_csv(path, rows, header="step,loss,eval"):
    lines = [header] + rows
    path.write_text("\n".join(lines) + "\n")


def test_cli_converging_exits_zero(tmp_path, capsys):
    csv = tmp_path / "m.csv"
    rows = [f"{k},{0.85**k:.6f},{1 - 0.5 * 0.85**k:.6f}" for k in range(25)]
    _write_csv(csv, rows)
    code = run(["run", str(csv), "--min", "loss", "--max", "eval", "--final"])
    out = capsys.readouterr().out
    assert code == 0
    assert "CONVERGING" in out


def test_cli_diverging_exits_nonzero(tmp_path, capsys):
    csv = tmp_path / "m.csv"
    rows = [f"{k},{0.2 * 1.2**k:.6f}" for k in range(20)]
    _write_csv(csv, rows, header="step,loss")
    code = run(["run", str(csv), "--min", "loss", "--final"])
    out = capsys.readouterr().out
    assert code == 1
    assert "DIVERGING" in out


def test_cli_missing_column_errors(tmp_path):
    csv = tmp_path / "m.csv"
    _write_csv(csv, ["0,1.0"], header="step,loss")
    try:
        run(["run", str(csv), "--min", "drift"])
    except SystemExit as exc:
        assert "missing columns" in str(exc.code)
    else:
        raise AssertionError("expected SystemExit")
