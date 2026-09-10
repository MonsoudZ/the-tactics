"""Per-ant resources: the half of isolation a git worktree cannot provide.

A worktree isolates files. Two ants running a suite against one database
truncate each other's fixtures mid-run and produce failures belonging to no ant
in particular — the same corruption worktrees exist to prevent, one layer down.
"""

from __future__ import annotations

import os
import pathlib
import subprocess

import pytest

from tactics.playbooks.agent_sdk import AgentWorkspace, Resource
from tactics.playbooks.resources import env_per_ant, postgres_per_ant


def _repo(tmp_path) -> str:
    repo = tmp_path / "repo"
    repo.mkdir()
    run = lambda *a: subprocess.run(["git", *a], cwd=repo, capture_output=True, check=True)
    run("init", "-q"); run("config", "user.email", "t@t"); run("config", "user.name", "t")
    (repo / "seed.txt").write_text("seed\n")
    run("add", "-A"); run("commit", "-qm", "seed")
    return str(repo)


# --- the mechanism, domain-free -----------------------------------------------


def test_each_ant_gets_its_own_environment(tmp_path):
    ws = AgentWorkspace(_repo(tmp_path), isolate=True,
                        provision=env_per_ant(SLOT="{n}"))
    try:
        first, second, third = ws.session(None), ws.session(None), ws.session(None)
        assert [first.env["SLOT"], second.env["SLOT"], third.env["SLOT"]] == ["1", "2", "3"]
    finally:
        ws.cleanup()


def test_the_environment_reaches_what_the_ant_runs(tmp_path):
    # The point of merging rather than replacing: the check picks this up
    # without knowing the mechanism exists.
    ws = AgentWorkspace(_repo(tmp_path), isolate=True, provision=env_per_ant(SLOT="{n}"))
    try:
        session = ws.session(None)
        code, out = session.run(["sh", "-c", "echo $SLOT-$HOME"])
        assert code == 0
        assert out.strip().startswith("1-") and len(out.strip()) > 2   # PATH etc. survive
    finally:
        ws.cleanup()


def test_the_main_workspace_has_no_resource_of_its_own(tmp_path):
    ws = AgentWorkspace(_repo(tmp_path), isolate=True, provision=env_per_ant(SLOT="{n}"))
    try:
        assert ws.env == {}
    finally:
        ws.cleanup()


def test_a_resource_is_given_back_when_the_ant_is_released(tmp_path):
    released = []
    ws = AgentWorkspace(_repo(tmp_path), isolate=True,
                        provision=lambda i, p: Resource(env={"X": str(i)},
                                                        release=lambda: released.append(i)))
    try:
        session = ws.session(None)
        pathlib.Path(session.path, "work.txt").write_text("done\n")
        ws.release(session)
        assert released == [1]
    finally:
        ws.cleanup()


def test_a_release_that_throws_does_not_cost_the_patch(tmp_path):
    # A leaked database is a nuisance; a lost patch is the run's whole output.
    def explode():
        raise RuntimeError("could not drop")

    ws = AgentWorkspace(_repo(tmp_path), isolate=True,
                        provision=lambda i, p: Resource(release=explode))
    try:
        session = ws.session(None)
        pathlib.Path(session.path, "work.txt").write_text("done\n")
        ws.release(session)
        assert len(ws.patches) == 1 and "work.txt" in ws.patches[0].text
    finally:
        ws.cleanup()


def test_a_scratch_tree_inherits_the_ants_environment(tmp_path):
    # `trial` and `proves_itself` cut further worktrees to re-verify a patch. If
    # those ran against the default database the verification would measure a
    # different world than the run did.
    ws = AgentWorkspace(_repo(tmp_path), isolate=True, provision=env_per_ant(SLOT="{n}"))
    try:
        session = ws.session(None)
        scratch = session._add_worktree(prefix="trial")
        assert scratch.env == session.env
        session.run(["git", "worktree", "remove", "--force", scratch.path])
    finally:
        ws.cleanup()


def test_without_provisioning_nothing_changes(tmp_path):
    ws = AgentWorkspace(_repo(tmp_path), isolate=True)
    try:
        assert ws.session(None).env == {}
    finally:
        ws.cleanup()


# --- the Postgres provisioner -------------------------------------------------


def test_an_unsafe_database_name_is_refused_rather_than_escaped():
    # The one string here that reaches a SQL statement. It comes from config
    # rather than a user, and is still validated instead of trusted.
    with pytest.raises(ValueError, match="unsafe database name"):
        postgres_per_ant('test"; DROP DATABASE production; --')


def test_two_ants_never_share_a_database_name(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr("tactics.playbooks.resources._psql",
                        lambda sql, **kw: (seen.append(sql), (0, ""))[1])
    provision = postgres_per_ant("app_test")
    first, second = provision(1, "/a"), provision(1, "/b")   # same index on purpose
    assert first.env["DATABASE_URL"] != second.env["DATABASE_URL"]
    assert all("TEMPLATE \"app_test\"" in s for s in seen)


def test_a_clone_that_fails_leaves_the_ant_unconfigured_rather_than_broken(monkeypatch):
    # Pointing at a database that does not exist would fail every check for a
    # reason that has nothing to do with the agent. Sharing the default is
    # wrong, but visibly so.
    monkeypatch.setattr("tactics.playbooks.resources._psql",
                        lambda sql, **kw: (1, "FATAL: no such template"))
    resource = postgres_per_ant("app_test")(1, "/a")
    assert resource.env == {}
    assert "could not clone" in resource.describe


def test_release_drops_the_clone(monkeypatch):
    statements = []
    monkeypatch.setattr("tactics.playbooks.resources._psql",
                        lambda sql, **kw: (statements.append(sql), (0, ""))[1])
    resource = postgres_per_ant("app_test")(1, "/a")
    resource.release()
    assert any(s.startswith("DROP DATABASE IF EXISTS") for s in statements)
    assert any("FORCE" in s for s in statements)     # a dead run leaves connections
