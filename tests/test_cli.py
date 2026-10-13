from quantlab.cli import build_parser, main


def test_parser_subcommands():
    p = build_parser()
    args = p.parse_args(["run", "--config", "x.yaml", "--hypothesis", "1", "--walk-forward"])
    assert args.cmd == "run" and args.hypothesis == 1 and args.walk_forward


def test_hypothesis_add_and_list(tmp_path, capsys):
    reg = str(tmp_path / "exp.duckdb")
    rationale = "x" * 60
    assert (
        main(["--registry", reg, "hypothesis", "add", "--name", "H", "--rationale", rationale]) == 0
    )
    assert main(["--registry", reg, "hypothesis", "list"]) == 0
    out = capsys.readouterr().out
    assert "id 1" in out and "[1] H" in out


def test_runs_empty(tmp_path, capsys):
    assert main(["--registry", str(tmp_path / "e.duckdb"), "runs"]) == 0
    assert "Aucun run" in capsys.readouterr().out
