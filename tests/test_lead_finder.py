"""Tests for the lead-finder playbook."""

from __future__ import annotations

from tactics import AutoApprove, Budget, DryRun, InMemoryStore
from tactics.llm import ScriptedClient
from tactics.playbooks.lead_finder import (
    LeadFinder,
    Site,
    build_colony,
    build_places_colony,
    heuristic_scorer,
    llm_drafter,
    llm_scorer,
    load_places_csv,
    load_sites,
    mark_emailed,
    places_drafter,
    places_scorer,
    template_drafter,
    win_clients,
)

# A sample in the exact schema lead_finder.rb writes.
_PLACES_HEADER = (
    "date_found,email_sent,confidence,score,name,phone,website,rating,reviews,issues,address,place_id\n"
)
_PLACES_ROWS = (
    '2026-06-01,,high,18.4,Ace Roofing,303-555-0100,http://aceroofing.example,4.8,210,'
    '"no HTTPS; not mobile-responsive (no viewport); slow load (3.2s)",Parker CO,pid_ace\n'
    "2026-06-01,,needs_verification,12.0,Best Dentistry,303-555-0200,(none),4.9,150,"
    "no website at all,Denver CO,pid_best\n"
    "2026-05-01,2026-05-02,high,9.0,Old Spa,205-555-0300,http://oldspa.example,4.4,40,"
    "outdated (©2019),Mobile AL,pid_old\n"
)


def _write_places_csv(tmp_path):
    p = tmp_path / "leads.csv"
    p.write_text(_PLACES_HEADER + _PLACES_ROWS, encoding="utf-8")
    return str(p)

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


def test_load_sites_from_csv(tmp_path):
    p = tmp_path / "leads.csv"
    p.write_text(
        "url,business,no_https,mobile_broken\n"
        "http://a.example,Alpha,true,yes\n"
        "http://b.example,,0,1\n",
        encoding="utf-8",
    )
    sites = load_sites(str(p))
    assert len(sites) == 2
    assert sites[0].business == "Alpha"
    assert sites[0].signals == {"no_https": True, "mobile_broken": True}
    assert sites[1].business == "B"  # derived from the domain
    assert sites[1].signals == {"no_https": False, "mobile_broken": True}


def test_load_sites_from_jsonl_with_nested_signals(tmp_path):
    p = tmp_path / "leads.jsonl"
    p.write_text(
        '{"url": "http://x.example", "business": "X Co", "signals": {"slow": true}}\n'
        '{"url": "http://y.example"}\n',
        encoding="utf-8",
    )
    sites = load_sites(str(p))
    assert sites[0].signals == {"slow": True}
    assert sites[1].business == "Y"


def test_load_sites_json_with_field_overrides(tmp_path):
    p = tmp_path / "leads.json"
    p.write_text(
        '[{"domain": "http://z.example", "name": "Z Co", "ssl": "false", "mobile": "ok"}]',
        encoding="utf-8",
    )
    sites = load_sites(
        str(p), url_field="domain", business_field="name",
        signal_fields={"ssl": "no_https", "mobile": "mobile_broken"},
    )
    assert sites[0].url == "http://z.example"
    assert sites[0].business == "Z Co"
    assert sites[0].signals == {"no_https": False, "mobile_broken": True}


# --- Places pipeline adapter (the Ruby leads.csv) ----------------------------


def test_load_places_csv_skips_emailed_and_parses_issues(tmp_path):
    sites = load_places_csv(_write_places_csv(tmp_path))
    assert len(sites) == 2  # the already-emailed "Old Spa" row is skipped
    by_name = {s.business: s for s in sites}
    assert by_name["Ace Roofing"].signals == {"no_https": True, "not_mobile": True, "slow": True}
    assert by_name["Ace Roofing"].meta["rating"] == "4.8"


def test_load_places_csv_handles_no_website_lead(tmp_path):
    sites = load_places_csv(_write_places_csv(tmp_path))
    best = next(s for s in sites if s.business == "Best Dentistry")
    assert best.signals.get("no_website") is True
    assert best.url == "noweb:pid_best"  # unique identity, no collision
    assert best.meta["website"] == "(none)"


def test_places_scorer_normalizes_ranking(tmp_path):
    sites = load_places_csv(_write_places_csv(tmp_path))
    ace = next(s for s in sites if s.business == "Ace Roofing")
    assert places_scorer(ace) == 1.0  # top score in the set


def test_places_drafter_is_specific(tmp_path):
    sites = load_places_csv(_write_places_csv(tmp_path))
    ace = next(s for s in sites if s.business == "Ace Roofing")
    draft = places_drafter(ace)
    assert "Ace Roofing" in draft.body
    assert "210 reviews" in draft.body  # leads with the real reputation
    best = next(s for s in sites if s.business == "Best Dentistry")
    assert "website" in places_drafter(best).body.lower()


def test_mark_emailed_closes_the_loop(tmp_path):
    path = _write_places_csv(tmp_path)
    n = mark_emailed(path, ["pid_ace"], when="2026-06-28")
    assert n == 1
    reloaded = load_places_csv(path)  # pid_ace now skipped as emailed
    assert all(s.meta["place_id"] != "pid_ace" for s in reloaded)


def test_places_colony_drafts_top_leads_dry_run(tmp_path):
    target = LeadFinder(load_places_csv(_write_places_csv(tmp_path)))
    result = build_places_colony(target).run(win_clients())
    assert target.sent == {}  # DryRun
    done = [f for f in result.findings if f.kind == "task_done"]
    assert len(done) == 2  # both fresh leads drafted + verified
    assert len(result.journal.of_kind("gate.hold")) == 2  # both proposed for review


def test_colony_with_llm_seams_and_budget():
    target = LeadFinder([STRONG])
    scorer = llm_scorer(ScriptedClient(['{"weakness": 0.9, "reason": "weak"}']))
    drafter = llm_drafter(ScriptedClient(['{"subject": "Fix your site", "body": "...", "quality": 0.8}']))
    budget = Budget(max_cost=10_000)
    result = build_colony(target, scorer=scorer, drafter=drafter, budget=budget).run(win_clients())
    done = [f for f in result.findings if f.kind == "task_done"]
    assert len(done) == 1
    assert budget.spent > 0  # LLM drafting tokens were charged

