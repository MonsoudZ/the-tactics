"""Tests for the lead-finder playbook."""

from __future__ import annotations

from tactics import AutoApprove, DryRun, InMemoryStore
from tactics.playbooks.lead_finder import (
    LeadFinder,
    Site,
    build_colony,
    heuristic_scorer,
    template_drafter,
    win_clients,
)

STRONG = Site("weak.example", "Weak Co", {"no_https": True, "mobile_broken": True, "slow": True})
MEH = Site("ok.example", "OK Co", {"no_https": True, "mobile_broken": False, "slow": False})
GOOD_SITE = Site("good.example", "Good Co", {"no_https": False, "mobile_broken": False})


# --- seams -------------------------------------------------------------------


def test_heuristic_scorer_rates_weakness():
    assert heuristic_scorer(STRONG) == 1.0
    assert round(heuristic_scorer(MEH), 3) == round(1 / 3, 3)
    assert heuristic_scorer(GOOD_SITE) == 0.0
    assert heuristic_scorer(Site("x", "X")) == 0.0  # no signals


def test_template_drafter_is_specific_and_scores_quality():
    draft = template_drafter(STRONG)
    assert "Weak Co" in draft.body
    assert "no https" in draft.body  # leads with a concrete issue
    assert 0.0 < draft.quality <= 1.0


# --- colony, dry-run ---------------------------------------------------------


def test_dry_run_sends_nothing_but_qualifies_leads():
    target = LeadFinder([STRONG, MEH, GOOD_SITE])
    result = build_colony(target, gate=DryRun()).run(win_clients())
    assert target.sent == {}  # DryRun: nothing left the building
    # strong lead accepted, the weak lead and strong site rejected
    done = [f for f in result.findings if f.kind == "task_done"]
    rejected = [f for f in result.findings if f.kind == "rejected"]
    assert len(done) == 1
    assert len(rejected) == 2
    # the one email that *would* send is recorded for review (by business name)
    holds = result.journal.of_kind("gate.hold")
    assert any("Weak Co" in e.data["action"] for e in holds)


def test_colony_terminates_on_no_open_work():
    target = LeadFinder([STRONG, MEH])
    result = build_colony(target).run(win_clients())
    assert result.stop_reason == "no open work"


# --- colony, live (approved) -------------------------------------------------


def test_approved_gate_delivers_through_the_mailer():
    delivered = []

    def mailer(site, subject, body):
        delivered.append(site.url)
        return True

    target = LeadFinder([STRONG], mailer=mailer)
    build_colony(target, gate=AutoApprove(), memory=InMemoryStore()).run(win_clients())
    assert delivered == ["weak.example"]
    assert "weak.example" in target.sent


def test_learning_is_recorded_for_verified_leads():
    target = LeadFinder([STRONG])
    memory = InMemoryStore()
    build_colony(target, memory=memory).run(win_clients())
    entries = list(memory.entries())
    assert entries
    assert all(e.tactic == "work_lead" for e in entries)
