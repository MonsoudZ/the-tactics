"""Tests for the LLM code-review playbook (offline via ScriptedClient)."""

from __future__ import annotations

from tactics import Context, Goal
from tactics.colony.blackboard import Task
from tactics.llm import ScriptedClient
from tactics.playbooks.code_review import CodeReview, build_review_colony, review_goal
from tactics.playbooks.repo_health import CodeRepo


def _ctx(target, file):
    return Context(target=target, goal=Goal(name="g"),
                   task=Task(id="t", description="", payload={"file": file}))


def test_review_reports_findings(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    client = ScriptedClient([
        '{"findings": [{"severity": "high", "kind": "bug", "line": 1, "message": "boom"}],'
        ' "summary": "one bug"}'
    ])
    out = CodeReview("rev", client).execute(_ctx(CodeRepo(str(tmp_path)), "a.py"))
    assert out.success is False           # a high-severity finding fails the file
    assert out.reward < 1.0
    assert out.metrics["findings"][0]["kind"] == "bug"
    assert out.metrics["file"] == "a.py"
    assert out.cost > 0                    # tokens charged to the Budget


def test_review_clean_file_scores_one(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    client = ScriptedClient(['{"findings": [], "summary": "looks clean"}'])
    out = CodeReview("rev", client).execute(_ctx(CodeRepo(str(tmp_path)), "a.py"))
    assert out.success is True
    assert out.reward == 1.0


def test_review_prompt_includes_numbered_code(tmp_path):
    (tmp_path / "a.py").write_text("first\nsecond\n", encoding="utf-8")
    client = ScriptedClient(['{"findings": [], "summary": ""}'])
    CodeReview("rev", client).execute(_ctx(CodeRepo(str(tmp_path)), "a.py"))
    prompt = client.calls[0]["prompt"]
    assert "1: first" in prompt and "2: second" in prompt  # line numbers for citing


def test_review_colony_fans_out_over_files(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("y = 2\n", encoding="utf-8")
    client = ScriptedClient(['{"findings": [], "summary": "clean"}'])  # repeats for both
    colony = build_review_colony(
        CodeRepo(str(tmp_path)), client, files=["a.py", "b.py"], max_workers=1
    )
    result = colony.run(review_goal())
    reviewed = {t.result.metrics["file"] for t in result.board.tasks if t.result}
    assert reviewed == {"a.py", "b.py"}
