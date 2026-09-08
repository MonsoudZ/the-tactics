"""Tests for the `python -m tactics` front door. Offline: the runner is injected.

What matters here is not argparse. It is that the defaults are the safe ones,
that the repository is not written to unless asked, and that the two mistakes
which silently measure the wrong thing get warned about.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile

import pytest

from tactics.cli import build_parser, main
from tactics.playbooks.agent_sdk import AgentRun


@pytest.fixture(autouse=True)
def _keep_temporary_files_out_of_the_real_tmp(tmp_path, monkeypatch):
    """A --no-persist run archives its patches under the system temp directory.

    That is right in production and litter in a test suite, so point the whole
    module somewhere disposable. Worktree roots follow it too, which is a bonus.
    """
    home = tmp_path / "tmp"
    home.mkdir(exist_ok=True)
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(home))


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


# --- the gate has to be askable ------------------------------------------------


class _Proposal:
    def __init__(self, action="agent tool call: Bash", **detail):
        self.action = action
        self.detail = detail
        self.reversible = False
        self.risk = "high"


def test_with_no_terminal_the_escalation_refuses(tmp_path, monkeypatch, capsys):
    from tactics.cli import _ask

    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    assert _ask(_Proposal(command="git push --force"), None) is False
    assert "refused (no terminal to ask)" in capsys.readouterr().err


def test_the_prompt_names_the_command_not_just_the_tool(tmp_path, monkeypatch):
    # "approve? agent tool call: Bash" is not a question anybody can answer.
    from tactics.cli import _ask

    asked = []
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr("builtins.input", lambda prompt: asked.append(prompt) or "y")
    assert _ask(_Proposal(command="rm -rf build/"), None) is True
    assert "rm -rf build/" in asked[0]


@pytest.mark.parametrize("answer,approved", [("y", True), ("yes", True), ("Y", True),
                                             ("n", False), ("", False), ("maybe", False)])
def test_only_yes_means_yes(monkeypatch, answer, approved):
    from tactics.cli import _ask

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr("builtins.input", lambda prompt: answer)
    assert _ask(_Proposal(), None) is approved


def test_closed_stdin_refuses_rather_than_guessing(monkeypatch, capsys):
    from tactics.cli import _ask

    def closed(prompt):
        raise EOFError

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr("builtins.input", closed)
    assert _ask(_Proposal(), None) is False
    assert "stdin closed" in capsys.readouterr().err


def test_parallel_agents_do_not_share_one_prompt(monkeypatch):
    # This is reached from each ant's own thread. Unserialized, several agents
    # read the same stdin at once and an approval meant for one arrives at
    # another — a yes given for the wrong action.
    import threading
    import time

    from tactics.cli import _ask

    inside = []
    overlapped = []

    def slow_input(prompt):
        inside.append(prompt)
        if len(inside) > 1:
            overlapped.append(prompt)
        time.sleep(0.02)
        inside.pop()
        return "y"

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr("builtins.input", slow_input)
    threads = [threading.Thread(target=_ask, args=(_Proposal(command=f"cmd {i}"), None))
               for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert overlapped == []


# --- a failure has to look like one -------------------------------------------


def _broken_runner(message):
    def runner(brief, spec, bridge, ws):
        return AgentRun(cost_usd=0.0, error=message)

    return runner


def test_a_run_that_never_started_says_why(tmp_path, capsys):
    # The first-run state for most people: the SDK is not installed. This used
    # to print "1 run, $0.00, 0 file(s)" and "nothing to apply" — indistinguishable
    # from an agent that looked around and found nothing to do.
    repo = _repo(tmp_path)
    code = main(_argv(repo, "--yes"),
                runner=_broken_runner("claude-agent-sdk not installed: No module named X"))
    out = capsys.readouterr().out
    assert "claude-agent-sdk not installed" in out
    assert "setup problem, not a result" in out
    assert "pip install 'tactics[agent-sdk]'" in out
    assert code == 2                              # not 1: nothing ran at all


def test_an_auth_failure_says_what_to_do_about_it(tmp_path, capsys):
    repo = _repo(tmp_path)
    main(_argv(repo, "--yes"), runner=_broken_runner("401 unauthorized"))
    assert "authenticate the Claude CLI" in capsys.readouterr().out


def test_an_unrecognised_failure_is_still_shown_verbatim(tmp_path, capsys):
    # No hint is better than a wrong hint, but the error itself always prints.
    repo = _repo(tmp_path)
    main(_argv(repo, "--yes"), runner=_broken_runner("the moon was in the wrong phase"))
    out = capsys.readouterr().out
    assert "the moon was in the wrong phase" in out
    assert "pip install" not in out


def test_one_failure_among_several_is_not_a_setup_problem(tmp_path, capsys):
    repo = _repo(tmp_path)
    calls = []

    def flaky(brief, spec, bridge, ws):
        calls.append(brief)
        if len(calls) == 1:
            return AgentRun(cost_usd=0.0, error="transient explosion")
        return _runner()(brief, spec, bridge, ws)

    code = main(_argv(repo, "--yes", "--agents", "2"), runner=flaky)
    out = capsys.readouterr().out
    assert "transient explosion" in out           # still reported
    assert "setup problem" not in out             # but not diagnosed as one
    assert code == 0                              # the other agent delivered


def test_a_check_that_passes_over_an_empty_diff_does_not_claim_a_fix(tmp_path, capsys):
    # An already-green repo. Every ant "confirming the fix" here is the exact
    # self-report this playbook refuses to trust, just phrased politely.
    repo = _repo(tmp_path)

    def idle(brief, spec, bridge, ws):
        return AgentRun(cost_usd=0.01)            # touches nothing

    main([repo, "do the thing", "--check", "true", "--no-learn", "--no-persist",
          "--yes"], runner=idle)
    out = capsys.readouterr().out
    assert "check passes, but the agent changed nothing" in out
    assert "confirms the fix" not in out


# --- and the leavings do not ---------------------------------------------------


def test_the_cli_reaps_worktrees_a_killed_run_left_behind(tmp_path, capsys):
    from tactics.playbooks.agent_sdk import AgentWorkspace

    repo = _repo(tmp_path)
    dead = AgentWorkspace(repo, isolate=True)
    orphan = dead.session(None).path
    dead._owner_lock.close()          # what SIGKILL does: no cleanup, no lock
    assert pathlib.Path(orphan).exists()

    main(_argv(repo, "--yes"), runner=_runner())
    assert not pathlib.Path(orphan).exists()
    assert "cleaned up 1 worktree(s) abandoned by an earlier run" in capsys.readouterr().out


def test_nothing_is_said_when_there_is_nothing_to_clean_up(tmp_path, capsys):
    repo = _repo(tmp_path)
    main(_argv(repo, "--yes"), runner=_runner())
    assert "abandoned" not in capsys.readouterr().out


def test_a_terminate_signal_unwinds_instead_of_leaking(tmp_path):
    # SIGKILL cannot be caught, but SIGTERM is how `timeout` and a plain `kill`
    # end a run — and the default disposition dies before any cleanup runs.
    import signal as signal_module

    from tactics.cli import _catch_terminate, _restore_terminate

    before = signal_module.getsignal(signal_module.SIGTERM)
    previous = _catch_terminate()
    installed = signal_module.getsignal(signal_module.SIGTERM)
    try:
        with pytest.raises(SystemExit):
            installed(signal_module.SIGTERM, None)
    finally:
        _restore_terminate(previous)
    assert signal_module.getsignal(signal_module.SIGTERM) is before


# --- the candidates outlive the run -------------------------------------------


def test_patches_are_archived_under_the_repo_by_default(tmp_path, capsys):
    repo = _repo(tmp_path)
    main([repo, "do the thing", "--check", "test -f work.txt", "--no-learn", "--yes"],
         runner=_runner())
    saved = sorted(pathlib.Path(repo, ".tactics", "patches").rglob("*.patch"))
    assert len(saved) == 1
    assert "work.txt" in saved[0].read_text()
    out = capsys.readouterr().out
    assert "saved to" in out and "git apply --3way" in out


def test_no_persist_still_keeps_the_work_just_not_in_your_repo(tmp_path, capsys):
    # --no-persist is about not writing your repository. Throwing the candidates
    # away instead would be a different, worse promise.
    repo = _repo(tmp_path)
    main(_argv(repo, "--yes"), runner=_runner())
    assert not pathlib.Path(repo, ".tactics").exists()
    saved = sorted((tmp_path / "tmp").rglob("*.patch"))
    assert len(saved) == 1 and "work.txt" in saved[0].read_text()
    assert f"saved to {saved[0].parent}" in capsys.readouterr().out


def test_patch_dir_puts_them_where_you_asked(tmp_path, capsys):
    repo = _repo(tmp_path)
    where = tmp_path / "elsewhere"
    main(_argv(repo, "--yes", "--patch-dir", str(where)), runner=_runner())
    assert [f.name for f in sorted(where.glob("*.patch"))] == ["001-SingleAgentNarrow.patch"]
    assert str(where) in capsys.readouterr().out


def test_an_archived_patch_applies_with_plain_git(tmp_path):
    # The recovery path for a run that died: nothing survives but the file.
    repo = _repo(tmp_path)
    where = tmp_path / "elsewhere"
    main(_argv(repo, "--yes", "--patch-dir", str(where)), runner=_runner())
    saved = sorted(where.glob("*.patch"))[0]
    done = subprocess.run(["git", "-C", repo, "apply", "--3way", str(saved)],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert pathlib.Path(repo, "work.txt").read_text() == "done\n"


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
