"""Tiny, dependency-free .env loader so secrets stay out of code.

Keep your keys in a gitignored ``.env`` file:

    ANTHROPIC_API_KEY=sk-ant-...
    GOOGLE_PLACES_API_KEY=...

then call ``load_env()`` once at startup, before constructing any client:

    from tactics import load_env
    load_env()
    from tactics.llm import ClaudeClient
    client = ClaudeClient()   # picks up ANTHROPIC_API_KEY from the environment

Real environment variables win over the file by default, so CI / shell secrets
are never clobbered. No third-party dependency (no python-dotenv needed).
"""

from __future__ import annotations

import os


def load_env(path: str = ".env", *, override: bool = False) -> dict[str, str]:
    """Load ``KEY=VALUE`` lines from ``path`` into ``os.environ``.

    Blank lines and ``#`` comments are ignored; surrounding quotes are stripped.
    Existing env vars are kept unless ``override=True``. Returns the parsed
    key/values (whether or not they were applied). Missing file is a no-op.
    """
    parsed: dict[str, str] = {}
    if not os.path.exists(path):
        return parsed
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if not key:
                continue
            parsed[key] = val
            if override or key not in os.environ:
                os.environ[key] = val
    return parsed
