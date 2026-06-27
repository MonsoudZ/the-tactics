"""Tests for the lead-finder playbook."""

from __future__ import annotations

from tactics import AutoApprove, Budget, DryRun, InMemoryStore
from tactics.llm import ScriptedClient
from tactics.playbooks.lead_finder import (
    LeadFinder,
    Site,
    build_colony,
    heuristic_scorer,
    llm_drafter,
    llm_scorer,
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


# --- LLM seams (offline via ScriptedClient) ----------------------------------


def test_llm_scorer_parses_and_clamps():
    score = llm_scorer(ScriptedClient(['{"weakness": 1.5, "reason": "broken"}']))
    assert score(STRONG) == 1.0  # clamped into [0, 1]


def test_llm_scorer_fails_soft_to_zero():
    score = llm_scorer(ScriptedClient(["the model said no json"]))
    assert score(STRONG) == 0.0  # hiccup skips the lead, doesn't crash


def test_llm_drafter_parses_and_charges_tokens():
    client = ScriptedClient(['{"subject": "S", "body": "B", "quality": 0.9}'])
    draft = llm_drafter(client)(STRONG)
    assert draft.subject == "S" and draft.body == "B" and draft.quality == 0.9
    assert draft.cost > 0  # tokens flow into the Budget


def test_llm_drafter_fails_closed_on_garbage():
    import pytest

    with pytest.raises(ValueError):
        llm_drafter(ScriptedClient(["not json — never send this"]))(STRONG)


def test_colony_with_llm_seams_and_budget():
    target = LeadFinder([STRONG])
    scorer = llm_scorer(ScriptedClient(['{"weakness": 0.9, "reason": "weak"}']))
    drafter = llm_drafter(ScriptedClient(['{"subject": "Fix your site", "body": "...", "quality": 0.8}']))
    budget = Budget(max_cost=10_000)
    result = build_colony(target, scorer=scorer, drafter=drafter, budget=budget).run(win_clients())
    done = [f for f in result.findings if f.kind == "task_done"]
    assert len(done) == 1
    assert budget.spent > 0  # LLM drafting tokens were charged

