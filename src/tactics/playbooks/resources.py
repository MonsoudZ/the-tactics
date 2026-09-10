"""Give each parallel ant its own copy of the things that live outside the tree.

A git worktree isolates files. It does not isolate a database, and that is the
difference between fan-out working on a toy repo and fan-out working on an
application. Pointed at a real Rails API, every run in this project has been
`--agents 1` for exactly one reason: three ants running the suite against one
Postgres truncate each other's fixtures mid-run, and the failures belong to no
ant in particular — the same corruption `Target.session` exists to prevent, one
layer down.

The mechanism is in `AgentWorkspace` and is domain-free: a session may carry a
:class:`Resource`, whose ``env`` is merged into everything that session runs.
The knowledge of *what* to provision is here, because "clone a Postgres
template database" is not something the core should ever know.

    from tactics.playbooks.resources import postgres_per_ant
    ws = AgentWorkspace(repo, isolate=True,
                        provision=postgres_per_ant("intentia_api_test"))

Each ant then gets `DATABASE_URL` pointing at its own clone, made with
`CREATE DATABASE … TEMPLATE …` so the schema arrives without re-running
migrations, and dropped when the ant is released.
"""

from __future__ import annotations

import os
import re
import subprocess
import uuid
from typing import Callable

from .agent_sdk import Resource

#: Postgres identifiers: what we are willing to interpolate into SQL. Names come
#: from a caller's config, not a user's input, but a database name is still the
#: one string here that reaches a statement — so it is validated rather than
#: trusted, and anything unexpected is refused instead of escaped.
_SAFE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")


def _psql(sql: str, *, database: str = "postgres", timeout: int = 120) -> tuple[int, str]:
    try:
        done = subprocess.run(["psql", "-d", database, "-v", "ON_ERROR_STOP=1", "-c", sql],
                              capture_output=True, text=True, timeout=timeout)
        return done.returncode, (done.stdout or "") + (done.stderr or "")
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, repr(exc)


def postgres_available() -> bool:
    code, _out = _psql("select 1")
    return code == 0


def postgres_per_ant(
    template: str,
    *,
    url_for: Callable[[str], str] | None = None,
    variable: str = "DATABASE_URL",
    prefix: str = "tactics_ant",
) -> Callable[[int, str], Resource]:
    """Provision one Postgres database per ant, cloned from ``template``.

    `CREATE DATABASE … TEMPLATE …` copies a prepared schema in about a second,
    which matters: the alternative is running migrations once per ant per round.

    The clone is dropped on release, and named with a random suffix as well as
    the ant's index — two runs of the same repo overlap in time more often than
    you would think, and a name collision would hand two ants the same database,
    which is the failure this exists to prevent wearing a fix's clothes.

    If the template cannot be cloned the ant gets **no** environment override
    rather than a broken one: it then shares the default database, which is
    wrong but visibly so, instead of pointing at a database that does not exist
    and failing every check for a reason that has nothing to do with the agent.
    """
    if not _SAFE_NAME.match(template):
        raise ValueError(f"unsafe database name: {template!r}")

    def provision(index: int, _path: str) -> Resource:
        name = f"{prefix}_{index}_{uuid.uuid4().hex[:8]}"
        code, out = _psql(f'CREATE DATABASE "{name}" TEMPLATE "{template}"')
        if code != 0:
            return Resource(describe=f"could not clone {template}: {out.strip()[-160:]}")

        def release() -> None:
            # FORCE disconnects anything still attached; without it a suite that
            # died mid-run leaves a connection that blocks the drop forever.
            dropped, _ = _psql(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
            if dropped != 0:                        # older servers have no FORCE
                _psql(f'DROP DATABASE IF EXISTS "{name}"')

        url = url_for(name) if url_for else f"postgres:///{name}"
        return Resource(env={variable: url}, release=release,
                        describe=f"{variable}={url} (clone of {template})")

    return provision


def env_per_ant(**template: str) -> Callable[[int, str], Resource]:
    """Give each ant environment variables of its own, `{n}` filled with its index.

    The general case behind the Postgres one: a port, a cache namespace, a
    scratch directory. `env_per_ant(REDIS_URL="redis://localhost/{n}")` hands
    ant 3 database 3.
    """
    def provision(index: int, path: str) -> Resource:
        env = {k: v.format(n=index, path=path) for k, v in template.items()}
        return Resource(env=env, describe=", ".join(f"{k}={v}" for k, v in env.items()))

    return provision
