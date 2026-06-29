"""Downstream of lead_finder.rb: turn leads.csv into reviewed outreach.

Your Ruby tool finds + audits + ranks leads into leads.csv. This takes that file
and drafts a specific, credible hook per lead, lets the critic kill the weak
ones, and proposes each send through a DryRun gate (nothing sends — you review).

Run it:  python examples/places_outreach_demo.py

Against your real file, replace the sample path with your leads.csv. For
Claude-written hooks: drafter=llm_drafter(ClaudeClient()).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from tactics.playbooks.lead_finder import (
    LeadFinder,
    build_places_colony,
    load_places_csv,
    win_clients,
)

SAMPLE = (
    "date_found,email_sent,confidence,score,name,phone,website,rating,reviews,issues,address,place_id\n"
    '2026-06-01,,high,18.4,Ace Roofing,303-555-0100,http://aceroofing.example,4.8,210,'
    '"no HTTPS; not mobile-responsive (no viewport); slow load (3.2s)",Parker CO,pid_ace\n'
    "2026-06-01,,needs_verification,12.0,Birmingham Dental,205-555-0200,(none),4.9,150,"
    "no website at all,Birmingham AL,pid_dent\n"
    '2026-06-01,,high,7.5,Evergreen HVAC,303-555-0300,http://evergreenhvac.example,4.5,60,'
    '"outdated (©2019); no meta description (weak SEO)",Evergreen CO,pid_hvac\n'
    "2026-05-01,2026-05-02,high,9.0,Already Contacted Co,303-555-0400,http://done.example,4.4,40,"
    "no HTTPS,Denver CO,pid_done\n"
)


def main() -> None:
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "leads.csv"
        path.write_text(SAMPLE, encoding="utf-8")

        sites = load_places_csv(str(path))  # skips the already-emailed row
        print(f"Loaded {len(sites)} fresh leads (already-contacted ones skipped).\n")

        target = LeadFinder(sites)               # no mailer -> dry-run stub
        result = build_places_colony(target).run(win_clients())

        print(result.summary())
        print(f"Emails sent: {len(target.sent)} (DryRun — you review first)\n")

        print("Drafted outreach, awaiting your approval:")
        done = [t for t in result.board.tasks if t.status == "done" and t.result]
        for t in sorted(done, key=lambda t: -t.result.reward):
            o = t.result
            print(f"\n  ── {o.metrics.get('subject')}")
            print(f"     {o.notes}")
        print("\n(Run with drafter=llm_drafter(ClaudeClient()) to have Claude write each hook.)")


if __name__ == "__main__":
    main()
