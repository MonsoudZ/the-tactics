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

from dataclasses import dataclass, field
from typing import Callable

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


@dataclass
class Draft:
    subject: str
    body: str
    quality: float  # 0..1 self-rated strength of the hook


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
