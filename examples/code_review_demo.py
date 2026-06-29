"""The brain reviewing code and reporting findings (scripted offline).

Shows the shape of an LLM code review: per-file findings with severity + line.
The scripted client stands in for Claude so this runs with no key. Live, swap in
ClaudeClient() and point `files` at your real source.

Run it:  python examples/code_review_demo.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from tactics.llm import ScriptedClient
from tactics.playbooks.code_review import build_review_colony, review_goal
from tactics.playbooks.repo_health import CodeRepo

# What Claude would return per file (scripted for an offline demo).
SCRIPT = [
    '{"findings": ['
    '{"severity": "high", "kind": "security", "line": 3, "message": "SQL built via string interpolation — injection risk"},'
    '{"severity": "medium", "kind": "dead_code", "line": 7, "message": "helper() is never called"}'
    '], "summary": "One injection risk and some dead code."}',
    '{"findings": [], "summary": "Clean — clear names, no obvious issues."}',
]


def main() -> None:
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "orders.py").write_text("def find(uid):\n    q = '...'\n    return run(q)\n", encoding="utf-8")
        (Path(d) / "utils.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")

        colony = build_review_colony(
            CodeRepo(str(d)), ScriptedClient(SCRIPT),
            files=["orders.py", "utils.py"], max_workers=1,
        )
        result = colony.run(review_goal())

        print("Code review\n")
        print(result.summary(), "\n")
        for t in result.board.tasks:
            if not t.result:
                continue
            o = t.result
            print(f"  {t.payload['file']}  — {o.notes}")
            for f in o.metrics.get("findings", []):
                print(f"      [{f['severity'].upper():6}] line {f['line']}: {f['message']}")
        print("\nLive: build_review_colony(CodeRepo('.'), ClaudeClient()) reviews your real files.")


if __name__ == "__main__":
    main()
