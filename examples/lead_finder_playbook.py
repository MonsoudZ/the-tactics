"""The lead-finder playbook running end-to-end — offline, dry-run.

It qualifies a list of sites, drafts outreach for the strong leads, and *proposes*
sending each (held by the DryRun gate — nothing actually sends). The deliverable
is the review: which leads it would contact and with what hook.

Run it:  python examples/lead_finder_playbook.py

To go live: pass a real `mailer` to LeadFinder, swap the gate to CallbackGate
(your approval), and optionally swap `drafter`/`scorer` for LLM versions
(`from tactics.llm import ClaudeClient`).
"""

from __future__ import annotations

from tactics.playbooks.lead_finder import LeadFinder, Site, build_colony, win_clients


def main() -> None:
    sites = [
        Site("sloppy-plumbing.example", "Sloppy Plumbing",
             {"no_https": True, "mobile_broken": True, "slow": True}),
        Site("corner-cafe.example", "Corner Cafe",
             {"no_https": True, "mobile_broken": True, "no_contact_form": True}),
        Site("polished-law.example", "Polished Law Firm",
             {"no_https": False, "mobile_broken": False, "slow": False}),  # strong site — skip
        Site("dusty-garage.example", "Dusty Garage",
             {"no_https": True, "outdated_design": True, "slow": True, "no_contact_form": True}),
    ]

    target = LeadFinder(sites)            # no mailer -> dry-run stub
    colony = build_colony(target)         # DryRun gate by default
    result = colony.run(win_clients())

    print(result.summary())
    print(f"\nEmails actually sent: {len(target.sent)} (DryRun — nothing left the building)\n")

    print("Would contact (held for your approval):")
    for e in result.journal.of_kind("gate.hold"):
        print(f"  • {e.data['action']}")

    print("\nQualified + drafted (verified by the critic):")
    for f in result.findings:
        if f.kind == "task_done":
            print(f"  • {f.detail['task']}  (reward {f.detail['reward']})")

    print("\nSkipped as weak leads:")
    for f in result.findings:
        if f.kind == "rejected":
            print(f"  • {f.detail['task']}  — {f.detail['reason']}")


if __name__ == "__main__":
    main()
