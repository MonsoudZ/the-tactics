"""Tests for the `python -m tactics` front door. Offline: the runner is injected.

What matters here is not argparse. It is that the defaults are the safe ones,
that the repository is not written to unless asked, and that the two mistakes
which silently measure the wrong thing get warned about.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest

from tactics.cli import build_parser, main
from tactics.playbooks.agent_sdk import AgentRun


def _repo(tmp_path, name="repo") -> str:
    repo = tmp_path / name
    repo.mkdir()
    run = lambda *a: subprocess.run(["git", *a], cwd=repo, capture_output=True, check=True)
    run("init", "-q"); run("config", "user.email", "t@t"); run("config", "user.name", "t")
    (repo / "seed.txt").write_text("seed\n")
    run("add", "-A"); run("commit", "-qm", "seed")
    return str(repo)


def _runner(filename="work.txt", body="done\n"):
    """Stands in for a real agent: writes into whichever worktree it is handed."""

    def runner(brief, spec, bridge, ws):
        run = AgentRun(cost_usd=0.02)
        if bridge.decide("Write", {"file_path": filename})[0]:
            pathlib.Path(ws.path, filename).write_text(body)
        run.tools_used.append("Write")
        run.denied = list(bridge.denied)
        return run

    return runner


def _argv(repo, *extra):
    return [repo, "do the thing", "--check", "test -f work.txt",
            "--no-learn", "--no-persist", *extra]


# --- the defaults are the safe ones -------------------------------------------


def test_your_repository_is_not_written_to_without_apply(tmp_path, capsys):
    repo = _repo(tmp_path)
    assert main(_argv(repo, "--yes"), runner=_runner()) == 0
    assert not pathlib.Path(repo, "work.txt").exists()
    out = capsys.readouterr().out
    assert "your repository is untouched" in out
    assert "--apply" in out                       # and it says how to change that


def test_apply_lands_the_patch(tmp_path, capsys):
    repo = _repo(tmp_path)
    assert main(_argv(repo, "--yes", "--apply"), runner=_runner()) == 0
    assert pathlib.Path(repo, "work.txt").read_text() == "done\n"
    assert "landed" in capsys.readouterr().out


def test_a_dry_run_writes_nothing_anywhere(tmp_path, capsys):
    repo = _repo(tmp_path)
    code = main(_argv(repo, "--dry-run"), runner=_runner())
    assert code == 1                              # nothing produced, so nothing to apply
    assert not pathlib.Path(repo, "work.txt").exists()
    assert "0 candidate patch" in capsys.readouterr().out


def test_the_default_posture_escalates_rather_than_approving(tmp_path):
    # No --yes: a push or an rm -rf must not simply go through.
    from tactics.core.approval import PolicyGate

    parser = build_parser()
    args = parser.parse_args([".", "t"])
    assert not args.yes and not args.dry_run and not args.apply
    gate = PolicyGate(escalate=lambda p, c: False, allow_risk=("low", "medium"))
    from tactics.playbooks.agent_sdk import classify_tool_call

    reversible, risk = classify_tool_call("Bash", {"command": "git push"})
    assert gate.decide(type("P", (), {"reversible": reversible, "risk": risk})(), None) is False


def test_worktrees_are_reaped_even_though_the_repo_is_untouched(tmp_path):
    repo = _repo(tmp_path)
    main(_argv(repo, "--yes"), runner=_runner())
    listed = subprocess.run(["git", "-C", repo, "worktree", "list"],
                            capture_output=True, text=True).stdout
    assert listed.strip().count("\n") == 0        # only the main tree remains


# --- refusing to start beats failing halfway ----------------------------------


def test_a_missing_directory_is_refused(tmp_path, capsys):
    assert main(_argv(str(tmp_path / "nope"))) == 2
    assert "not a directory" in capsys.readouterr().err


def test_a_directory_that_is_not_a_repo_is_refused(tmp_path, capsys):
    plain = tmp_path / "plain"
    plain.mkdir()
    assert main(_argv(str(plain))) == 2
    assert "not a git repository" in capsys.readouterr().err


def test_a_repo_with_no_commits_is_refused(tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=empty, check=True)
    assert main(_argv(str(empty))) == 2
    assert "no commits" in capsys.readouterr().err


# --- the two mistakes that silently measure the wrong thing -------------------


def test_it_warns_about_a_src_layout_with_no_pythonpath(tmp_path, capsys):
    # The trap that bit this project's own experiments: the check runs inside the
    # worktree, so an editable install measures the main tree instead.
    repo = _repo(tmp_path)
    pathlib.Path(repo, "src").mkdir()
    main([repo, "t", "--check", "pytest -q", "--no-learn", "--no-persist", "--dry-run"],
         runner=_runner())
    assert "src/ layout with no pythonpath" in capsys.readouterr().err


def test_no_such_warning_when_pythonpath_is_configured(tmp_path, capsys):
    repo = _repo(tmp_path)
    pathlib.Path(repo, "src").mkdir()
    pathlib.Path(repo, "pytest.ini").write_text("[pytest]\npythonpath = src\n")
    subprocess.run(["git", "-C", repo, "add", "-A"], check=True, capture_output=True)
    subprocess.run(["git", "-C", repo, "commit", "-qm", "cfg"], check=True, capture_output=True)
    main([repo, "t", "--check", "pytest -q", "--no-learn", "--no-persist", "--dry-run"],
         runner=_runner())
    assert "pythonpath" not in capsys.readouterr().err


def test_it_warns_that_uncommitted_work_is_invisible_to_the_agents(tmp_path, capsys):
    repo = _repo(tmp_path)
    pathlib.Path(repo, "seed.txt").write_text("edited but not committed\n")
    main(_argv(repo, "--dry-run"), runner=_runner())
    assert "worktrees are cut from HEAD" in capsys.readouterr().err


def test_the_check_is_echoed_because_it_is_the_reward(tmp_path, capsys):
    repo = _repo(tmp_path)
    main(_argv(repo, "--dry-run"), runner=_runner())
    assert "this decides the reward" in capsys.readouterr().out


# --- the knobs actually reach the machinery -----------------------------------


def test_parallel_agents_each_get_their_own_worktree(tmp_path, capsys):
    repo = _repo(tmp_path)
    calls = []

    def runner(brief, spec, bridge, ws):
        calls.append(ws.path)
        return _runner()(brief, spec, bridge, ws)

    main(_argv(repo, "--yes", "--agents", "3"), runner=runner)
    assert len(calls) == 3 and len(set(calls)) == 3


def test_max_turns_reaches_the_briefs(tmp_path):
    repo = _repo(tmp_path)
    seen = []

    def runner(brief, spec, bridge, ws):
        seen.append(spec.max_turns)
        return _runner()(brief, spec, bridge, ws)

    main(_argv(repo, "--yes", "--max-turns", "4"), runner=runner)
    assert seen == [4]


def test_persistence_is_opt_out_and_lands_in_the_repo(tmp_path):
    repo = _repo(tmp_path)
    main([repo, "t", "--check", "test -f work.txt", "--yes", "--no-learn"], runner=_runner())
    assert pathlib.Path(repo, ".tactics", "agent_sdk_memory.json").exists()


def test_no_persist_leaves_nothing_behind(tmp_path):
    repo = _repo(tmp_path)
    main(_argv(repo, "--yes"), runner=_runner())
    assert not pathlib.Path(repo, ".tactics").exists()


def test_the_budget_is_a_real_ceiling(tmp_path, capsys):
    repo = _repo(tmp_path)
    calls = []

    def expensive(brief, spec, bridge, ws):
        run = _runner()(brief, spec, bridge, ws)
        run.cost_usd = 99.0
        calls.append(brief)
        return run

    # The check can never pass, so nothing but the budget can stop the rounds:
    # a satisfied goal would end the run for the wrong reason and the ceiling
    # would go untested (without it: 8 calls, with it: 2).
    main([repo, "do the thing", "--check", "test -f never.txt", "--no-learn",
          "--no-persist", "--yes", "--agents", "2", "--rounds", "4",
          "--budget", "1.00"], runner=expensive)
    # One round of two ants blows the $1 ceiling, so no further round starts.
    # Asserted on the agent calls themselves, not on what the report prints.
    assert len(calls) == 2
    assert "$198.00 spent" in capsys.readouterr().out


def test_yes_and_dry_run_cannot_both_be_asked_for():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["r", "t", "--yes", "--dry-run"])


def test_it_warns_when_a_repo_has_no_gitignore(tmp_path, capsys):
    # Found live: with no .gitignore, an agent that ran the tests itself put
    # .pyc files in its own patch, and the patch then would not apply.
    repo = _repo(tmp_path)
    main(_argv(repo, "--dry-run"), runner=_runner())
    assert "no .gitignore" in capsys.readouterr().err


def test_no_such_warning_when_the_repo_ignores_its_artifacts(tmp_path, capsys):
    repo = _repo(tmp_path)
    pathlib.Path(repo, ".gitignore").write_text("__pycache__/\n*.pyc\n")
    subprocess.run(["git", "-C", repo, "add", "-A"], check=True, capture_output=True)
    subprocess.run(["git", "-C", repo, "commit", "-qm", "ignore"], check=True, capture_output=True)
    main(_argv(repo, "--dry-run"), runner=_runner())
    assert "no .gitignore" not in capsys.readouterr().err


# --- you cannot review what you cannot see ------------------------------------


def test_show_diff_prints_the_patch_body(tmp_path, capsys):
    repo = _repo(tmp_path)
    main(_argv(repo, "--yes", "--show-diff"),
         runner=_runner("work.txt", "a distinctive line\n"))
    out = capsys.readouterr().out
    assert "patch from SingleAgentNarrow" in out
    assert "a distinctive line" in out            # the actual content, not a count
    assert "+++ b/work.txt" in out                # and it is a real unified diff


def test_the_diff_is_not_truncated(tmp_path, capsys):
    # The worktree is destroyed when the run ends, so the printed copy is the
    # only one; truncating it would lose the tail permanently.
    body = "".join(f"line {i}\n" for i in range(300))
    repo = _repo(tmp_path)
    main(_argv(repo, "--yes", "--show-diff"), runner=_runner("big.txt", body))
    out = capsys.readouterr().out
    assert "line 0" in out and "line 299" in out


def test_without_the_flag_only_the_summary_is_printed(tmp_path, capsys):
    repo = _repo(tmp_path)
    main(_argv(repo, "--yes"), runner=_runner("work.txt", "a distinctive line\n"))
    out = capsys.readouterr().out
    assert "1 candidate patch" in out
    assert "a distinctive line" not in out
    assert "patch from" not in out


def test_each_candidate_gets_its_own_labelled_section(tmp_path, capsys):
    repo = _repo(tmp_path)
    main(_argv(repo, "--yes", "--agents", "2", "--show-diff"), runner=_runner())
    assert capsys.readouterr().out.count("patch from") == 2


def test_show_diff_on_a_run_that_produced_nothing_says_nothing(tmp_path, capsys):
    repo = _repo(tmp_path)
    main(_argv(repo, "--dry-run", "--show-diff"), runner=_runner())
    out = capsys.readouterr().out
    assert "0 candidate patch" in out and "patch from" not in out


def test_the_report_says_whether_the_check_actually_passed(tmp_path, capsys):
    # Listing candidate patches without their verdict reads as though every one
    # of them works — which a live run showed is not true.
    repo = _repo(tmp_path)
    main(_argv(repo, "--yes"), runner=_runner())
    out = capsys.readouterr().out
    assert "verification" in out and "confirms the fix" in out


def test_a_patch_that_does_not_fix_anything_says_so(tmp_path, capsys):
    repo = _repo(tmp_path)
    # The check looks for a file the agent never writes.
    main([repo, "t", "--check", "test -f never.txt", "--yes", "--no-learn", "--no-persist"],
         runner=_runner("something_else.txt"))
    assert "still failing" in capsys.readouterr().out
