"""A green check cannot tell a real change from a no-op.

On a repository whose check already passes, "the check passes after the run"
proves only that nothing broke — and the agent wrote the test that grades its
own work. These cover the measurement that replaces the assumption.
"""

from __future__ import annotations

import pathlib
import subprocess

from tactics.playbooks.agent_sdk import AgentWorkspace, Patch, _diff_subset


def _repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    run = lambda *a: subprocess.run(["git", *a], cwd=repo, capture_output=True, check=True)
    run("init", "-q"); run("config", "user.email", "t@t"); run("config", "user.name", "t")
    (repo / "lib.py").write_text("def add(a, b):\n    return a + b\n")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_lib.py").write_text(
        "from lib import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n")
    run("add", "-A"); run("commit", "-qm", "seed")
    return str(repo)


def _patch_from(repo, edits) -> Patch:
    """Whatever `edits` writes into a worktree, as a Patch."""
    ws = AgentWorkspace(repo, isolate=True, check=["python3", "-m", "pytest", "-q"])
    try:
        session = ws.session(None)
        edits(pathlib.Path(session.path))
        ws.release(session)
        return ws.patches[0]
    finally:
        ws.cleanup()


def _ws(repo):
    return AgentWorkspace(repo, isolate=True, check=["python3", "-m", "pytest", "-q"])


def test_a_test_that_needs_the_code_proves_the_patch(tmp_path):
    def edits(tree):
        (tree / "lib.py").write_text("def add(a, b):\n    return a + b\n\n\n"
                                     "def mul(a, b):\n    return a * b\n")
        (tree / "tests" / "test_mul.py").write_text(
            "from lib import mul\n\n\ndef test_mul():\n    assert mul(2, 3) == 6\n")

    repo = _repo(tmp_path)
    patch = _patch_from(repo, edits)
    ws = _ws(repo)
    try:
        proof = ws.proves_itself(patch)
        assert proof.ok, proof.detail          # the test goes red without mul()
    finally:
        ws.cleanup()


def test_a_test_that_passes_without_the_code_proves_nothing(tmp_path):
    # The failure this exists to catch: a patch whose test would pass on an
    # empty change, so the green check after the run measured nothing.
    def edits(tree):
        (tree / "lib.py").write_text("def add(a, b):\n    return a + b\n\n\n"
                                     "def unused():\n    return 1\n")
        (tree / "tests" / "test_vacuous.py").write_text(
            "def test_arithmetic_still_works():\n    assert 1 + 1 == 2\n")

    repo = _repo(tmp_path)
    patch = _patch_from(repo, edits)
    ws = _ws(repo)
    try:
        proof = ws.proves_itself(patch)
        assert not proof.ok
        assert "without the rest of the patch" in proof.detail
    finally:
        ws.cleanup()


def test_a_patch_with_no_tests_at_all_is_told_so(tmp_path):
    repo = _repo(tmp_path)
    patch = _patch_from(repo, lambda tree: (tree / "lib.py").write_text(
        "def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n"))
    ws = _ws(repo)
    try:
        proof = ws.proves_itself(patch)
        assert not proof.ok and "no test files" in proof.detail
    finally:
        ws.cleanup()


def test_the_main_repository_is_never_touched_by_the_proof(tmp_path):
    def edits(tree):
        (tree / "tests" / "test_new.py").write_text("def test_x():\n    assert True\n")

    repo = _repo(tmp_path)
    patch = _patch_from(repo, edits)
    ws = _ws(repo)
    try:
        ws.proves_itself(patch)
        assert not pathlib.Path(repo, "tests", "test_new.py").exists()
        listed = subprocess.run(["git", "-C", repo, "worktree", "list"],
                                capture_output=True, text=True).stdout
        assert listed.strip().count("\n") == 0        # the scratch tree is reaped
    finally:
        ws.cleanup()


def test_the_diff_subset_keeps_only_the_files_asked_for():
    diff = (
        "diff --git a/lib.py b/lib.py\n--- a/lib.py\n+++ b/lib.py\n@@ -1 +1 @@\n-old\n+new\n"
        "diff --git a/tests/test_x.py b/tests/test_x.py\n--- a/tests/test_x.py\n"
        "+++ b/tests/test_x.py\n@@ -1 +1 @@\n-a\n+b\n"
    )
    only = _diff_subset(diff, ["tests/test_x.py"])
    assert "tests/test_x.py" in only
    assert "lib.py\n--- a/lib.py" not in only        # the code half is gone
