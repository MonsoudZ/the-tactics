"""Point the brain at a real repo and audit it. Runs against this repo by default.

    python examples/repo_audit_demo.py            # audits "."
    python examples/repo_audit_demo.py /path/to/repo
"""

from __future__ import annotations

import sys

from tactics.playbooks.repo_health import CodeRepo, audit_goal, build_audit_colony


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "."
    colony = build_audit_colony(CodeRepo(path), max_workers=3)
    result = colony.run(audit_goal())

    print(f"\nRepo health audit of {path!r}")
    print(result.summary())
    print()
    for task in result.board.tasks:
        if task.result is None:
            continue
        o = task.result
        flag = "✓" if o.success else "✗"
        print(f"  {flag} {task.payload.get('check'):8s} {o.notes}")
        for hit in o.metrics.get("hits", []):
            print(f"        ! {hit['kind']} in {hit['file']}:{hit['line']}")


if __name__ == "__main__":
    main()
