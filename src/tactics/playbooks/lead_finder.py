"""Lead-finder playbook — score weak sites, draft outreach, propose sending.

The revenue loop: qualify a lead -> draft a hook -> (gated) send -> learn from
results -> get better at which sites and which hooks convert.

It's built from clean **seams** so you wire your real world to it piece by piece —
the parts you don't have yet stay as safe stubs:

  * ``sites``  — the candidate list (url + business + observed weakness signals).
  * ``scorer(site) -> float`` in [0,1] — how weak / likely-a-good-lead the site is.
                 Default: a heuristic over the signals. Swap in an LLM scorer.
  * ``drafter(site) -> Draft`` — the outreach. Default: a template. Swap in an LLM.
  * ``mailer(site, subject, body) -> bool`` — actually send. Default: None, a
                 dry-run stub that records intent but sends nothing.

Safety: sending is irreversible, so the tactic **proposes** it through the
approval gate. Run with ``DryRun`` first and review ``result.journal`` for every
email it *would* send; switch the gate to ``CallbackGate``/``PolicyGate`` to go live.

Reward honesty: until real reply/deal data exists, reward is a **proxy**
(lead weakness x draft quality). When you learn a real outcome (a reply, a deal),
record it against ``work_lead`` for that situation so learning reflects truth, not
the proxy — that's the difference between the right win and the quick win.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from typing import Any, Callable

from ..colony import Colony, FunctionCritic, FunctionPlanner, Verdict
from ..core.approval import ApprovalGate, DryRun, Proposal
from ..core.budget import Budget
from ..core.goal import Goal
from ..core.memory import InMemoryStore, MemoryStore
from ..core.outcome import Outcome
from ..core.tactic import Tactic
from ..core.target import Target


@dataclass
class Site:
    url: str
    business: str
    # Observed weakness signals, e.g. {"no_https": True, "mobile_broken": True}.
    signals: dict[str, bool] = field(default_factory=dict)
    # Free-form context for richer drafting (rating, reviews, raw issue text,
    # phone, place_id…). Populated by load_places_csv; empty for simple loaders.
    meta: dict[str, Any] = field(default_factory=dict)


# --- loading your lead source (a file your tool produces) ---------------------

_TRUE = {"1", "true", "t", "yes", "y", "x", "on"}
_FALSE = {"", "0", "false", "f", "no", "n", "off", "none", "null"}


def _truthy(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, str):
        s = v.strip().lower()
        if s in _TRUE:
            return True
        if s in _FALSE:
            return False
        return bool(s)  # any other non-empty string counts as present
    return bool(v)


def _business_from_url(url: str) -> str:
    host = url.split("//", 1)[-1].split("/", 1)[0].split("?", 1)[0]
    host = host.removeprefix("www.")
    name = host.split(":")[0].split(".")[0]
    return name.replace("-", " ").replace("_", " ").title() or url


def _record_to_site(rec: dict, url_field: str, business_field: str, signal_fields) -> Site | None:
    url = str(rec.get(url_field, "")).strip()
    if not url:
        return None
    business = str(rec.get(business_field) or "").strip() or _business_from_url(url)

    signals: dict[str, bool] = {}
    nested = rec.get("signals")
    if isinstance(nested, dict):  # a JSON record that already nests signals
        signals.update({str(k): _truthy(v) for k, v in nested.items()})
    if signal_fields is not None:  # explicit list/dict of which columns are signals
        mapping = signal_fields if isinstance(signal_fields, dict) else {f: f for f in signal_fields}
        for src, name in mapping.items():
            if src in rec:
                signals[name] = _truthy(rec[src])
    elif not signals:  # infer: extra boolean-ish columns become signals
        skip = {url_field, business_field, "signals"}
        for k, v in rec.items():
            if k in skip:
                continue
            if isinstance(v, bool) or (isinstance(v, str) and v.strip().lower() in (_TRUE | _FALSE)):
                signals[k] = _truthy(v)
    return Site(url=url, business=business, signals=signals)


def load_sites(
    path: str,
    *,
    url_field: str = "url",
    business_field: str = "business",
    signal_fields=None,
) -> list[Site]:
    """Load candidate sites from your lead-source file into Site objects.

    Supports ``.json`` (a list, or ``{"sites": [...]}``), ``.jsonl`` (one JSON
    object per line), and ``.csv`` (header row). Each record needs at least a URL;
    the business name is taken from ``business_field`` or derived from the domain.

    Signals (the weakness flags the scorer reads) come from, in order:
      * a nested ``signals`` object on the record, if present;
      * ``signal_fields`` — a list of columns to treat as signals, or a
        ``{column: signal_name}`` dict to rename them;
      * otherwise, any extra boolean-ish columns are inferred as signals.

    If your file is just URLs with no weakness data, load it anyway and use the
    LLM scorer (``llm_scorer``), or add an audit step that fills in signals.
    """
    if path.endswith(".jsonl"):
        records = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    elif path.endswith(".json"):
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        records = data["sites"] if isinstance(data, dict) and "sites" in data else data
    elif path.endswith(".csv"):
        with open(path, encoding="utf-8", newline="") as fh:
            records = list(csv.DictReader(fh))
    else:
        raise ValueError(f"unsupported lead-source format: {path} (use .json/.jsonl/.csv)")

    sites = [_record_to_site(r, url_field, business_field, signal_fields) for r in records]
    return [s for s in sites if s is not None]


@dataclass
class Draft:
    subject: str
    body: str
    quality: float  # 0..1 self-rated strength of the hook
    cost: float = 0.0  # tokens spent drafting (flows into Budget); 0 for templates


# --- default seams (offline, no LLM, no network) -----------------------------


def heuristic_scorer(site: Site) -> float:
    """Weakness = fraction of signals that are bad. A weak site is a good lead."""
    if not site.signals:
        return 0.0
    bad = sum(1 for v in site.signals.values() if v)
    return bad / len(site.signals)


def template_drafter(site: Site) -> Draft:
    """A concrete, specific hook beats a generic one — quality grows with specifics."""
    issues = [k.replace("_", " ") for k, v in site.signals.items() if v]
    lead_issue = issues[0] if issues else "a few issues"
    subject = f"quick note about {site.business}'s website"
    body = (
        f"Hi {site.business} team — I took a look at your site and noticed {lead_issue}. "
        f"It's likely costing you customers. I help local businesses fix this fast — "
        f"want a free 5-minute audit?"
    )
    quality = min(1.0, 0.4 + 0.2 * len(issues))
    return Draft(subject=subject, body=body, quality=round(quality, 3))


# --- LLM seams (Claude judges weakness and writes the hook) -------------------
#
# Same callable shape as the heuristics, so they drop straight into build_colony.
# Offline-testable: pass a ScriptedClient. Live: pass ClaudeClient() (needs a key).


def llm_scorer(client, *, max_tokens: int = 512) -> Callable[[Site], float]:
    """Build a scorer that asks the model how good a lead the site is.

    Fail-soft: a malformed/failed response scores 0.0 (skip the lead), never
    crashes the run — a transient hiccup shouldn't blow up the swarm.
    """
    from ..llm.client import extract_json  # local import keeps anthropic optional

    schema = {
        "type": "object",
        "properties": {"weakness": {"type": "number"}, "reason": {"type": "string"}},
        "required": ["weakness", "reason"],
        "additionalProperties": False,
    }
    system = (
        "You qualify sales leads for a web-improvement service. A weak, broken, "
        "insecure, or outdated site is a STRONG lead. Respond ONLY with JSON."
    )

    def score(site: Site) -> float:
        prompt = (
            f"Business: {site.business}\nURL: {site.url}\n"
            f"Observed signals: {site.signals}\n\n"
            'How good a lead is this? Return {"weakness": <0..1>, "reason": "<short>"}.'
        )
        try:
            resp = client.complete(prompt, system=system, schema=schema, max_tokens=max_tokens)
            data = extract_json(resp.text)
            return max(0.0, min(1.0, float(data.get("weakness", 0.0))))
        except (ValueError, KeyError, TypeError):
            return 0.0

    return score


def llm_drafter(client, *, max_tokens: int = 1024) -> Callable[[Site], Draft]:
    """Build a drafter that asks the model for a specific, credible outreach hook.

    Fail-closed: a malformed response raises, so the colony's failure isolation
    turns it into a loss and **no broken email is ever proposed for sending**.
    Drafting tokens are reported as the Draft's cost, so a Budget caps spend.
    """
    from ..llm.client import extract_json

    schema = {
        "type": "object",
        "properties": {
            "subject": {"type": "string"},
            "body": {"type": "string"},
            "quality": {"type": "number"},
        },
        "required": ["subject", "body", "quality"],
        "additionalProperties": False,
    }
    system = (
        "You write concise, specific, credible cold outreach to win web-services "
        "clients. Never generic or spammy. Respond ONLY with JSON."
    )

    def draft(site: Site) -> Draft:
        issues = site.meta.get("issues") or ", ".join(
            k.replace("_", " ") for k, v in site.signals.items() if v
        ) or "unclear"
        context = ""
        if site.meta.get("rating"):
            context = (
                f"\nReputation: {site.meta.get('rating')} stars, "
                f"{site.meta.get('reviews')} reviews (lead with this — it's real and flattering)."
            )
        prompt = (
            f"Write cold outreach to win {site.business} as a web-services client.\n"
            f"Their site issues: {issues}{context}\n\n"
            'Return {"subject": "...", "body": "...", "quality": <0..1 self-rating>}.'
        )
        resp = client.complete(prompt, system=system, schema=schema, max_tokens=max_tokens)
        data = extract_json(resp.text)  # raises on garbage -> fail closed
        return Draft(
            subject=str(data["subject"]),
            body=str(data["body"]),
            quality=max(0.0, min(1.0, float(data.get("quality", 0.5)))),
            cost=float(resp.tokens),
        )

    return draft


# --- adapter for the Ruby pipeline's leads.csv -------------------------------
#
# Your lead_finder.rb already finds, audits, scores, and ranks leads into a CSV
# with columns: date_found, email_sent, confidence, score, name, phone, website,
# rating, reviews, issues, address, place_id. This adapter reads that file so the
# brain can do the part the Ruby tool doesn't: draft + verify + (gated) send.

_ISSUE_SIGNALS = [
    ("no website", "no_website"),
    ("unreachable", "unreachable"),
    ("no https", "no_https"),
    ("viewport", "not_mobile"),
    ("mobile", "not_mobile"),
    ("title", "missing_title"),
    ("meta description", "no_meta_description"),
    ("slow", "slow"),
    ("outdated", "outdated"),
]


def _parse_issues(text: str) -> dict[str, bool]:
    """Turn the Ruby auditor's ``issues`` text into boolean signals."""
    t = (text or "").lower()
    return {name: True for needle, name in _ISSUE_SIGNALS if needle in t}


def load_places_csv(path: str, *, only_unsent: bool = True, min_score: float = 0.0) -> list[Site]:
    """Load leads from the Ruby pipeline's ``leads.csv``.

    Keeps the rich context (rating, reviews, raw issues, phone, place_id) in
    ``Site.meta`` so the drafter can write something specific. ``score`` is
    normalized into ``meta["score_norm"]`` (0..1) for prioritization. By default
    rows already marked ``email_sent`` are skipped, so you only work fresh leads.
    """
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    scores = [float(r.get("score") or 0) for r in rows]
    max_score = max(scores) if scores else 0.0

    sites: list[Site] = []
    for r in rows:
        if only_unsent and (r.get("email_sent") or "").strip():
            continue
        score = float(r.get("score") or 0)
        if score < min_score:
            continue

        website = (r.get("website") or "").strip()
        has_site = website not in ("", "(none)")
        name = (r.get("name") or "").strip() or (_business_from_url(website) if has_site else "Unknown")
        issues = (r.get("issues") or "").strip()
        signals = _parse_issues(issues)
        if not has_site:
            signals["no_website"] = True

        place_id = (r.get("place_id") or "").strip()
        identity = website if has_site else f"noweb:{place_id or name}"
        meta = {
            "rating": r.get("rating"),
            "reviews": r.get("reviews"),
            "score": score,
            "score_norm": round(score / max_score, 3) if max_score else 0.0,
            "confidence": r.get("confidence"),
            "phone": r.get("phone"),
            "address": r.get("address"),
            "place_id": place_id,
            "website": website,
            "issues": issues,
        }
        sites.append(Site(url=identity, business=name, signals=signals, meta=meta))
    return sites


def mark_emailed(path: str, place_ids, *, when: str) -> int:
    """Stamp ``email_sent=<when>`` for these place_ids in leads.csv, closing the
    loop with the Ruby tool (its next run preserves already-emailed leads).

    Returns how many rows were updated. Only fills empty ``email_sent`` cells.
    """
    ids = set(place_ids)
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    updated = 0
    for r in rows:
        if r.get("place_id") in ids and not (r.get("email_sent") or "").strip():
            r["email_sent"] = when
            updated += 1
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return updated


def places_scorer(site: Site) -> float:
    """Trust the Ruby pipeline's ranking — reward higher-scored leads more."""
    return float(site.meta.get("score_norm", 0.0))


def places_drafter(site: Site) -> Draft:
    """A specific, credible hook built from the lead's real reputation + issues."""
    m = site.meta
    rating, reviews = m.get("rating"), m.get("reviews")
    cred = f"a {rating}★ reputation with {reviews} reviews" if rating else "a great reputation"
    if site.signals.get("no_website"):
        body = (
            f"Hi {site.business} — you've earned {cred}, but I couldn't find a website. "
            f"Customers who search for you are finding nothing (or your competitors). "
            f"I build fast, mobile-first sites for local businesses — worth a quick chat?"
        )
        quality = 0.85
    else:
        first_issue = (m.get("issues") or "").split(";")[0].strip() or "a few fixable issues"
        body = (
            f"Hi {site.business} — impressive: {cred}. I took a look at your site and noticed "
            f"{first_issue}, which quietly costs you customers who check you out online. "
            f"I help local businesses fix exactly this — want a free 5-minute audit?"
        )
        quality = min(1.0, 0.55 + 0.1 * len(site.signals))
    return Draft(
        subject=f"quick note about {site.business}'s online presence",
        body=body,
        quality=round(quality, 3),
    )


# --- target ------------------------------------------------------------------


class LeadFinder(Target):
    name = "lead_finder"

    def __init__(self, sites: list[Site], *, mailer: Callable[[Site, str, str], bool] | None = None):
        self.sites = {s.url: s for s in sites}
        self.sent: dict[str, Draft] = {}
        self.mailer = mailer

    def observe(self) -> dict:
        return {"total": len(self.sites), "sent": len(self.sent)}

    def deliver(self, url: str, draft: Draft) -> bool:
        """The irreversible action — only ever called through the approval gate."""
        self.sent[url] = draft
        if self.mailer is not None:
            return bool(self.mailer(self.sites[url], draft.subject, draft.body))
        return True  # no mailer wired: record intent, report 'would send'


# --- tactic ------------------------------------------------------------------


class WorkLead(Tactic):
    """Qualify one site, and if it's a good lead, draft and propose outreach."""

    def __init__(
        self,
        *,
        scorer: Callable[[Site], float] = heuristic_scorer,
        drafter: Callable[[Site], Draft] = template_drafter,
        min_weakness: float = 0.34,
    ) -> None:
        super().__init__(name="work_lead")
        self.scorer = scorer
        self.drafter = drafter
        self.min_weakness = min_weakness

    def execute(self, ctx) -> Outcome:  # noqa: ANN001
        url = ctx.task.payload["url"]
        site = ctx.target.sites[url]
        weakness = float(self.scorer(site))
        if weakness < self.min_weakness:
            return Outcome(success=False, reward=round(weakness, 3),
                           metrics={"weakness": round(weakness, 3), "url": url},
                           notes="not a strong enough lead")

        draft = self.drafter(site)
        result = ctx.gate.submit(
            Proposal(
                action=f"email {site.business} <{url}>",
                commit=lambda: ctx.target.deliver(url, draft),
                reversible=False,
                risk="medium",
                detail={"subject": draft.subject},
            ),
            ctx,
        )
        reward = round(weakness * draft.quality, 3)
        return Outcome(
            success=True,
            reward=reward,
            cost=draft.cost,  # LLM drafting tokens (0 for the template drafter)
            metrics={
                "weakness": round(weakness, 3),
                "quality": draft.quality,
                "sent": result.committed,
                "subject": draft.subject,
                "url": url,
            },
            notes=draft.body,
        )


# --- planner & critic --------------------------------------------------------


def lead_planner(goal: Goal, board, target) -> list:  # noqa: ANN001
    if board.tasks:
        return []
    return [
        board.post_task(f"lead:{s.business}", payload={"url": s.url}, signal="outreach")
        for s in target.sites.values()
    ]


def quality_critic(min_reward: float = 0.25) -> FunctionCritic:
    """Trust only strong leads with a solid hook; reject weak/thin ones."""

    def check(outcome: Outcome, ctx) -> Verdict:  # noqa: ANN001
        ok = outcome.success and outcome.reward >= min_reward
        return Verdict(accepted=ok, reason="good lead + hook" if ok else "weak lead or thin hook")

    return FunctionCritic(check)


# --- convenience wiring ------------------------------------------------------


def win_clients() -> Goal:
    """Open-ended goal — the colony stops when every lead is worked (no open tasks)."""
    return Goal(name="win_clients", description="Qualify weak sites and send outreach")


def build_colony(
    target: LeadFinder,
    *,
    gate: ApprovalGate | None = None,
    memory: MemoryStore | None = None,
    scorer: Callable[[Site], float] = heuristic_scorer,
    drafter: Callable[[Site], Draft] = template_drafter,
    critic: FunctionCritic | None = None,
    budget: Budget | None = None,
    max_workers: int = 1,
    max_rounds: int = 50,
) -> Colony:
    """Wire a ready-to-run lead-finder colony. Defaults are safe: DryRun gate (no
    email actually sends) and one attempt per lead (deterministic scoring)."""
    return Colony(
        target,
        [WorkLead(scorer=scorer, drafter=drafter)],
        FunctionPlanner(lead_planner),
        gate=gate or DryRun(),
        memory=memory or InMemoryStore(),
        critic=critic or quality_critic(),
        budget=budget or Budget(max_attempts_per_task=1),
        max_workers=max_workers,
        max_rounds=max_rounds,
    )


def build_places_colony(
    target: LeadFinder,
    *,
    gate: ApprovalGate | None = None,
    memory: MemoryStore | None = None,
    drafter: Callable[[Site], Draft] = places_drafter,
    critic: FunctionCritic | None = None,
    budget: Budget | None = None,
    max_workers: int = 1,
    max_rounds: int = 500,
) -> Colony:
    """Wire a colony for leads loaded from the Ruby pipeline's CSV: trust its
    ranking (``places_scorer``), draft from the rich context, gate sends (DryRun
    by default). Pass ``drafter=llm_drafter(client)`` for Claude-written hooks."""
    return Colony(
        target,
        [WorkLead(scorer=places_scorer, drafter=drafter, min_weakness=0.1)],
        FunctionPlanner(lead_planner),
        gate=gate or DryRun(),
        memory=memory or InMemoryStore(),
        critic=critic or quality_critic(min_reward=0.15),
        budget=budget or Budget(max_attempts_per_task=1),
        max_workers=max_workers,
        max_rounds=max_rounds,
    )
