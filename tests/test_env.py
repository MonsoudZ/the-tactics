"""Tests for the .env loader."""

from __future__ import annotations

import os

from tactics import load_env


def test_load_env_parses_and_respects_existing(tmp_path, monkeypatch):
    p = tmp_path / ".env"
    p.write_text(
        '# a comment\nANTHROPIC_API_KEY=sk-test-123\nQUOTED="two"\nEXISTING=fromfile\n\n',
        encoding="utf-8",
    )
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("EXISTING", "fromenv")

    parsed = load_env(str(p))

    assert os.environ["ANTHROPIC_API_KEY"] == "sk-test-123"
    assert os.environ["QUOTED"] == "two"  # surrounding quotes stripped
    assert os.environ["EXISTING"] == "fromenv"  # real env not clobbered
    assert parsed["EXISTING"] == "fromfile"  # ...but still reported


def test_load_env_override(tmp_path, monkeypatch):
    p = tmp_path / ".env"
    p.write_text("K=fromfile\n", encoding="utf-8")
    monkeypatch.setenv("K", "fromenv")
    load_env(str(p), override=True)
    assert os.environ["K"] == "fromfile"


def test_load_env_missing_file_is_noop(tmp_path):
    assert load_env(str(tmp_path / "nope.env")) == {}
