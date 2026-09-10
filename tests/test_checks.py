"""Working out how a repo is tested, and proving it before anything is spent.

The check *is* the reward, so a command that cannot execute is the most
dangerous thing this framework can be handed: it fails looking exactly like a
repository in need of repair, and every safeguard downstream agrees with it.
"""

from __future__ import annotations

import pathlib

from tactics.playbooks.checks import Attempt, Check, attempt, choose, propose


def _tree(tmp_path, files: dict[str, str]) -> str:
    for name, body in files.items():
        path = pathlib.Path(tmp_path, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    return str(tmp_path)


def _attempt(code: int, output: str = "") -> Attempt:
    return Attempt(check=Check(["x"], "", "test"), code=code, output=output)


# --- ran, or never started ----------------------------------------------------


def test_a_silent_failure_is_still_a_check(tmp_path):
    # The first version of this demanded test-shaped output and rejected
    # `test -f built.txt` — a perfectly good check that exits 1 with nothing to
    # say. The burden belongs on failure-to-start, not on proof of running.
    assert _attempt(1, "").usable


def test_a_red_suite_is_a_check(tmp_path):
    assert _attempt(1, "3 examples, 2 failures").usable


def test_command_not_found_is_not_a_check():
    # 127 and 126 are OS conventions no test runner uses for a failing suite.
    assert not _attempt(127, "bash: pytest: command not found").usable
    assert not _attempt(126, "permission denied").usable


def test_a_process_that_starts_and_still_cannot_measure_is_not_a_check():
    # These exit 1 like a failing suite, so the exit code alone is not enough.
    assert not _attempt(1, "npm ERR! Missing script: \"test\"").usable
    assert not _attempt(1, "LoadError: cannot load such file -- rspec").usable
    assert not _attempt(1, "ModuleNotFoundError: No module named 'pytest'").usable


def test_passing_is_obviously_a_check():
    assert _attempt(0, "").usable


# --- proposing ----------------------------------------------------------------


def test_what_the_project_declares_beats_what_we_would_guess(tmp_path):
    # A declared script is the author telling you; anything else is inference.
    root = _tree(tmp_path, {
        "package.json": '{"scripts": {"test": "vitest run"}}',
        "Makefile": "test:\n\techo hi\n",
    })
    kinds = [c.kind for c in propose(root)]
    assert kinds and kinds[0] == "npm"
    assert "vitest run" in propose(root)[0].why      # the evidence, not a guess


def test_a_src_layout_gets_the_pythonpath_it_needs_first(tmp_path):
    # The trap this whole framework has documented since the first live run:
    # inside a worktree an editable install resolves to the *main* tree, so the
    # check measures code the agent never touched.
    root = _tree(tmp_path, {"pyproject.toml": "[project]\n", "src/pkg/__init__.py": "",
                            "tests/test_x.py": "def test_x(): pass\n"})
    first = propose(root)[0]
    assert first.argv[:2] == ["env", "PYTHONPATH=src"]
    assert "measures the main tree" in first.why


def test_a_configured_pythonpath_needs_no_such_help(tmp_path):
    root = _tree(tmp_path, {"pyproject.toml": "[tool.pytest.ini_options]\npythonpath = ['src']\n",
                            "src/pkg/__init__.py": "", "tests/test_x.py": ""})
    assert all("PYTHONPATH" not in c.command for c in propose(root))


def test_ruby_carries_a_fallback_for_a_gem_home_without_binstubs(tmp_path):
    # Found live: `bundle exec rspec` exits 127 on a container whose gem home
    # has no bin/ directory, while rspec-core is plainly installed. Invoking the
    # runner through Ruby needs no executable on PATH.
    root = _tree(tmp_path, {"Gemfile": "gem 'rspec-rails'\n", "spec/x_spec.rb": ""})
    commands = [c.command for c in propose(root)]
    assert any("bundle exec rspec" in c for c in commands)
    assert any("RSpec::Core::Runner" in c for c in commands)


def test_a_repo_that_suggests_nothing_proposes_nothing(tmp_path):
    assert propose(_tree(tmp_path, {"README.md": "hello\n"})) == []


# --- proving ------------------------------------------------------------------


def test_a_candidate_is_run_not_assumed(tmp_path):
    root = _tree(tmp_path, {"ok.txt": ""})
    assert attempt(root, Check(["test", "-f", "ok.txt"], "", "t")).usable
    assert not attempt(root, Check(["definitely-not-a-real-binary"], "", "t")).usable


def test_choose_falls_through_to_what_works(tmp_path):
    root = _tree(tmp_path, {"marker": ""})
    broken = Check(["definitely-not-a-real-binary"], "first guess", "t")
    working = Check(["test", "-f", "marker"], "second guess", "t")

    import tactics.playbooks.checks as mod

    original = mod.propose
    mod.propose = lambda _root: [broken, working]
    try:
        chosen, attempts = choose(root)
    finally:
        mod.propose = original

    assert chosen is working
    assert len(attempts) == 2 and not attempts[0].usable


def test_nothing_usable_is_a_result_with_its_reasons(tmp_path):
    # Refusing to start beats running a swarm against a check that measures
    # nothing, and the refusal has to be able to say what it tried.
    root = _tree(tmp_path, {"README.md": ""})
    chosen, attempts = choose(root)
    assert chosen is None and attempts == []


def test_the_chosen_check_records_how_long_it_took(tmp_path):
    # Printed to the user, because a check is run once per agent per round and
    # a slow one is a cost they should see before committing to it.
    root = _tree(tmp_path, {"ok.txt": ""})
    check = Check(["test", "-f", "ok.txt"], "", "t")
    attempt(root, check)
    assert check.seconds >= 0 and check.detail == "passed"


def test_a_check_that_collects_nothing_is_not_a_check(tmp_path):
    # Found live: `pytest -q` in a repo with no tests exits 5 and was accepted.
    # An agent can satisfy an empty check by adding any passing test at all,
    # which is never the task — so it measures nothing and must be refused.
    assert not _attempt(5, "no tests ran in 0.01s").usable
    assert not _attempt(1, "collected 0 items").usable
    assert not _attempt(1, "0 examples, 0 failures").usable
    assert _attempt(1, "collected 12 items / 1 failed").usable


def test_the_empty_check_says_what_was_wrong_with_it(tmp_path):
    root = _tree(tmp_path, {"pyproject.toml": "[project]\n"})
    check = Check(["python3", "-m", "pytest", "-q"], "", "pytest")
    attempt(root, check)
    assert check.ran is False
    assert "collected no tests" in check.detail


def test_the_ruby_fallback_names_the_spec_directory(tmp_path):
    # `RSpec::Core::Runner.run([])` exits 0 having run nothing: the binary
    # defaults to spec/, the API does not. That check was declared "proven"
    # against a real app before the empty-check rule caught it reporting
    # success over zero examples.
    root = _tree(tmp_path, {"Gemfile": "gem 'rspec'\n", "spec/x_spec.rb": ""})
    fallback = next(c for c in propose(root) if "RSpec::Core::Runner" in c.command)
    assert '["spec"]' in fallback.command


# --- a dependency that is not answering ---------------------------------------
#
# The tzdata failure wearing different clothes: a check that goes red for a
# reason having nothing to do with the repository, handed to a swarm that will
# duly "fix" production code to accommodate it.


#: Trimmed from a real run: focusmate-api's suite against a DATABASE_URL naming
#: a database that does not exist.
_NO_DATABASE = """An error occurred while loading ./spec/models/user_spec.rb.
Failure/Error: ActiveRecord::Migration.maintain_test_schema!

ActiveRecord::NoDatabaseError:
  We could not find your database: definitely_not_a_database_here.
# --- Caused by: ---
# PG::ConnectionBad:
#   connection to server on socket "/var/run/postgresql/.s.PGSQL.5432" failed:
#   FATAL:  database "definitely_not_a_database_here" does not exist

0 examples, 0 failures, 21 errors occurred outside of examples
"""


def test_a_database_that_is_not_there_is_not_a_red_suite():
    a = _attempt(1, _NO_DATABASE)
    assert not a.usable
    assert a.unavailable


def test_the_reason_given_is_the_real_one(tmp_path):
    # It was already refused before this existed, but by accident: `_EMPTY`'s
    # `^0 examples, 0 failures` prefix-matched `0 examples, 0 failures, 21
    # errors occurred outside of examples`, so the answer was "collected no
    # tests". Right refusal, wrong reason — and the reason is what a user acts
    # on. They would have gone looking for missing specs.
    script = pathlib.Path(tmp_path, "fake_suite.sh")
    script.write_text(f"#!/bin/sh\ncat <<'EOF'\n{_NO_DATABASE}EOF\nexit 1\n")
    script.chmod(0o755)

    check = Check([str(script)], "", "rspec")
    attempt(str(tmp_path), check)
    assert check.ran is False
    assert "could not reach a service" in check.detail
    assert "collected no tests" not in check.detail


def test_a_suite_that_half_ran_and_lost_its_database_is_still_not_a_check():
    # The case the accident did not cover, and the dangerous one: examples ran,
    # so there is no "0 examples" line anywhere. Under the old rule this was a
    # red suite — a repository in need of repair — and agents would have been
    # pointed at application code to fix a connection drop.
    output = ("40 examples, 21 failures\n"
              "Failure/Error: PG::ConnectionBad: could not connect to server")
    assert not _attempt(1, output).usable


def test_a_suite_that_never_loaded_is_not_a_check_whatever_the_cause():
    # The general form. The runner is saying the failures happened before any
    # of the repository's code was reached.
    assert not _attempt(1, "12 errors occurred outside of examples").usable
    assert not _attempt(2, "ERROR collecting tests/test_x.py").usable


def test_a_suite_testing_error_handling_is_left_alone(tmp_path):
    # The cry-wolf guard. A repository whose specs cover a refused connection
    # will print exactly the words a naive rule would key on, and disqualifying
    # its check would be worse than the bug this fixes. So each pattern names a
    # driver failing to reach its server, never a bare network phrase.
    output = ("Errno::ECONNREFUSED: Connection refused - connect(2)\n"
              "expected the client to retry\n"
              "31 examples, 1 failure")
    assert _attempt(1, output).usable


def test_a_passing_run_is_not_second_guessed():
    # `unavailable` only ever refuses a check; it never selects one, and it must
    # not overturn a suite that went green.
    assert _attempt(0, "80 examples, 0 failures").usable
