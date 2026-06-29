"""What a focumate Rails audit looks like (simulated offline).

The real audit runs in a session scoped to the focumate repo:

    from tactics.playbooks.repo_health import CodeRepo
    from tactics.playbooks.focumate import build_rails_audit, prod_ready_goal
    build_rails_audit(CodeRepo(".")).run(prod_ready_goal("rails"))

Here we feed a scripted runner + a planted repo so you can see the report shape —
including the checks firing on real problems.

Run it:  python examples/focumate_audit_demo.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from tactics.playbooks.focumate import build_rails_audit, prod_ready_goal
from tactics.playbooks.repo_health import CodeRepo


def main() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "config").mkdir()
        (root / "app").mkdir()
        (root / "config" / "master.key").write_text("xxxx", encoding="utf-8")  # committed secret!
        planted = "AIza" + "B" * 35  # a fake Google key, assembled so this file is clean
        (root / "app" / "payment.rb").write_text(f'KEY = "{planted}"\n', encoding="utf-8")

        def runner(cmd):
            if cmd[:1] == ["git"]:
                return 0, "config/master.key\napp/payment.rb\nGemfile"
            if "brakeman" in cmd:
                return 1, "1 security warning (possible SQL injection in UsersController)"
            if "bundler-audit" in cmd:
                return 1, "Vulnerable gem: nokogiri 1.13.0 (CVE-2022-...)"
            return 0, "ok"  # rspec, rubocop, migrations pass

        target = CodeRepo(str(root), runner=runner)
        result = build_rails_audit(target, max_workers=4).run(prod_ready_goal("rails"))

        print("focumate Rails — production-readiness audit (simulated)\n")
        print(result.summary(), "\n")
        for t in result.board.tasks:
            if t.result is None:
                continue
            o = t.result
            flag = "✓" if o.success else "✗"
            print(f"  {flag} {t.payload['check']:18s} {o.notes.splitlines()[0][:70]}")
            for hit in o.metrics.get("hits", []):
                print(f"        ! {hit['kind']} in {hit['file']}:{hit['line']}")
            for f in o.metrics.get("tracked", []):
                print(f"        ! committed: {f}")

        print("\nGreen the ✗ rows and focumate is a long step closer to prod.")
        print("Real run: scope a session to the focumate repo, then build_rails_audit(CodeRepo('.')).")


if __name__ == "__main__":
    main()
