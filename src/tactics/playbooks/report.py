"""Read a repository and say what is in it — feature by feature, with the gaps.

This is the front of the whole system: point it at a repo and get back what the
thing *does*, what each part is supposed to do, what is missing, and what can
go. Everything else here — tasks, agents, verification — hangs off that report.

**Every line of it traces to a file.** That constraint is the design. A report
that invents a feature is worse than no report, because it is indistinguishable
from one that read carefully, and it will be believed. So this module only
*extracts*: routes are read from the router, features are grouped by the
controllers that serve them, and a gap is the absence of a file that the repo's
own conventions say should exist. Nothing here asks a model what it thinks.

The optional second pass (:func:`describe`) *does* ask a model, and is kept
separate for exactly that reason: it is given the extracted facts and asked to
name and summarise them, never to add. What it returns is marked as narration
in the rendered report, so a reader always knows which half they are reading.

    from tactics.playbooks.report import inventory, render
    print(render(inventory("/path/to/repo")))
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .survey import code_files, is_generated, line_count, survey

# --- what we found ------------------------------------------------------------


@dataclass
class Endpoint:
    """One route the application answers, and what backs it."""

    verb: str
    path: str
    controller: str
    action: str
    specs: list[str] = field(default_factory=list)
    authorized: bool | None = None      # None: the repo has no such convention

    @property
    def name(self) -> str:
        return f"{self.verb} {self.path}"


@dataclass
class Gap:
    """Something the repository's own conventions say should be here and is not.

    Never "this code is bad" — only "this file has no test", "this action opts
    out of the check its siblings use". Absence is checkable; taste is not.
    """

    kind: str
    detail: str
    where: str
    severity: str = "medium"            # low | medium | high


@dataclass
class Feature:
    """A coherent slice of the application, grouped by what serves it."""

    name: str
    endpoints: list[Endpoint] = field(default_factory=list)
    models: list[str] = field(default_factory=list)
    services: list[str] = field(default_factory=list)
    jobs: list[str] = field(default_factory=list)
    policies: list[str] = field(default_factory=list)
    specs: list[str] = field(default_factory=list)
    gaps: list[Gap] = field(default_factory=list)
    narration: str = ""                 # filled only by describe(); never extracted

    @property
    def tested(self) -> bool:
        return bool(self.specs)


@dataclass
class Report:
    root: str
    kind: str                           # "rails" | "python" | "unknown"
    features: list[Feature] = field(default_factory=list)
    gaps: list[Gap] = field(default_factory=list)          # repo-wide
    missing: list[Gap] = field(default_factory=list)        # what is not there yet
    removable: list[Any] = field(default_factory=list)     # survey Candidates
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def all_gaps(self) -> list[Gap]:
        return self.gaps + [g for f in self.features for g in f.gaps]


# --- recognising the repository -----------------------------------------------


def detect(root: str) -> str:
    if os.path.isfile(os.path.join(root, "config", "routes.rb")):
        return "rails"
    if os.path.isfile(os.path.join(root, "pyproject.toml")) or os.path.isdir(
            os.path.join(root, "src")):
        return "python"
    return "unknown"


def _read(root: str, path: str) -> str:
    try:
        with open(os.path.join(root, path), encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return ""


def _under(root: str, *parts: str) -> list[str]:
    """Files under a directory, relative to the repo, skipping generated ones."""
    base = os.path.join(*parts)
    start = os.path.join(root, base)
    if not os.path.isdir(start):
        return []
    out = []
    for dirpath, _dirs, names in os.walk(start):
        for name in names:
            rel = os.path.relpath(os.path.join(dirpath, name), root)
            if name.endswith((".rb", ".py")) and not is_generated(root, rel):
                out.append(rel)
    return sorted(out)


# --- routes: asked of the router, not guessed ---------------------------------

_RAILS_ROUTES = (
    'require "./config/environment"; '
    "Rails.application.routes.routes.each { |r| "
    "c = r.defaults[:controller]; next unless c; "
    "puts({verb: r.verb.to_s, path: r.path.spec.to_s.sub('(.:format)',''), "
    "controller: c, action: r.defaults[:action].to_s}.to_json) }"
)


def rails_routes(root: str, *, timeout: int = 300) -> list[Endpoint]:
    """Every route the application actually serves.

    Booted rather than parsed: `resources :tasks` expands to seven routes and
    a regex over routes.rb would report one. If the app cannot boot here (no
    database, missing gems) the result is empty and the report says the routes
    are unknown, rather than inventing a plausible list.
    """
    try:
        done = subprocess.run(["bundle", "exec", "ruby", "-e", _RAILS_ROUTES],
                              cwd=root, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return []
    out = []
    for line in done.stdout.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        out.append(Endpoint(verb=row["verb"] or "ANY", path=row["path"],
                            controller=row["controller"], action=row["action"]))
    return out


# --- grouping into features ---------------------------------------------------

# Structural path components, not names. "app/services/task_update_service.rb"
# is the task-update feature, not the "app services task update" feature.
_STOP = {"api", "v1", "v2", "base", "application", "concerns",
         "app", "src", "lib", "controllers", "models", "services", "jobs",
         "policies", "serializers", "mailers", "channels", "spec", "specs",
         "test", "tests"}     # not "requests": friend_requests is a resource


def feature_name(identifier: str) -> str:
    """"api/v1/friend_requests" -> "friend requests"; "TaskUpdateService" -> "task"."""
    identifier = re.sub(r"\.(rb|py)$", "", identifier)
    parts = [p for p in re.split(r"[/_]", identifier) if p]
    parts = [p for p in parts if p.lower() not in _STOP]
    if not parts:
        return "core"
    tail = parts[-1]
    for suffix in ("controller", "service", "job", "policy", "serializer", "mailer"):
        if tail.lower().endswith(suffix) and len(tail) > len(suffix):
            tail = tail[: -len(suffix)]
        elif tail.lower() == suffix:
            parts = parts[:-1]
            tail = parts[-1] if parts else "core"
    name = " ".join(parts[:-1] + [tail]).strip().replace("  ", " ")
    return re.sub(r"(?<!^)(?=[A-Z])", " ", name).lower().strip() or "core"


def _singular(word: str) -> str:
    for plural, single in (("ies", "y"), ("ses", "s"), ("s", "")):
        if word.endswith(plural) and len(word) > len(plural):
            return word[: -len(plural)] + single
    return word


def _matches(feature: str, path: str) -> bool:
    """Does this file belong to this feature? Stem match, both directions."""
    stem = _singular(feature.replace(" ", "_"))
    target = _singular(os.path.splitext(os.path.basename(path))[0])
    return bool(stem) and (stem in target or target.startswith(stem))


# --- the gaps -----------------------------------------------------------------


def _spec_index(root: str) -> dict[str, str]:
    """Every spec/test file mapped to its text, for looking up what is covered."""
    files = _under(root, "spec") + _under(root, "test") + _under(root, "tests")
    return {p: _read(root, p) for p in files if re.search(r"_(spec|test)\.(rb|py)$", p)
            or re.search(r"(^|/)test_[^/]+\.py$", p)}


def rails_gaps(root: str, endpoints: list[Endpoint], specs: dict[str, str]) -> list[Gap]:
    """Absences the repo's own conventions make visible."""
    gaps: list[Gap] = []

    # An endpoint no spec mentions. Each :param becomes a wildcard so the route
    # "/api/v1/lists/:list_id/tasks/:id/complete" matches a spec writing
    # "/api/v1/lists/#{list.id}/tasks/#{task.id}/complete". Collapsing params to
    # nothing instead produced "//" and reported eight tested endpoints as
    # untested — a gap list full of false positives is worse than no gap list.
    for endpoint in endpoints:
        if not endpoint.path or endpoint.controller.startswith("rails/"):
            continue
        pattern = re.compile("".join(
            r'[^/"\s]+' if part.startswith(":") else re.escape(part)
            for part in re.split(r"(:[a-z_]+)", endpoint.path)))
        hits = [p for p, text in specs.items() if pattern.search(text)]
        endpoint.specs = hits
        if not hits:
            gaps.append(Gap("untested endpoint",
                            f"{endpoint.name} → {endpoint.controller}#{endpoint.action} "
                            "is served but no spec mentions its path",
                            endpoint.controller, severity="high"))

    # Opting out of the authorization check the siblings use.
    for path in _under(root, "app", "controllers"):
        text = _read(root, path)
        for match in re.finditer(r"skip_after_action\s+:verify_authorized([^\n]*)", text):
            scope = match.group(1).strip() or " (all actions)"
            gaps.append(Gap("authorization opt-out",
                            f"skips verify_authorized{scope} — the enforcement its "
                            "siblings rely on; confirm the scope guards it instead",
                            path, severity="medium"))

    # A model with no policy in an app that has policies.
    policies = {os.path.basename(p).replace("_policy.rb", "") for p in _under(root, "app", "policies")}
    if policies:
        for path in _under(root, "app", "models"):
            name = os.path.basename(path).replace(".rb", "")
            if name not in policies and name not in ("application_record",):
                gaps.append(Gap("model without policy",
                                f"{name} has no {name}_policy.rb in an app that "
                                "authorizes through Pundit",
                                path, severity="low"))
    return gaps


def python_gaps(root: str, specs: dict[str, str]) -> list[Gap]:
    """A module nothing imports in a test is a module nothing tests."""
    gaps = []
    for path in code_files(root, suffixes=(".py",)):
        if re.search(r"(^|/)(tests?|docs|examples)/", path) or path.endswith("__init__.py"):
            continue
        module = os.path.splitext(os.path.basename(path))[0]
        if not any(re.search(rf"\b{re.escape(module)}\b", text) for text in specs.values()):
            gaps.append(Gap("untested module",
                            f"{module} is not mentioned by any test file",
                            path, severity="high"))
    return gaps


# --- what is not there yet ----------------------------------------------------
#
# The hardest third, and the one where inventing is most tempting. The rule that
# keeps it honest: a missing feature is the repository *contradicting itself* —
# a resource whose siblings all offer delete and this one does not, a column
# stored and never returned, an association nothing serves. The repo says what
# it expects by what it does elsewhere, and that is checkable. "You should add
# rate limiting" is taste, and taste belongs in narration where it is labelled.

CRUD = ("index", "show", "create", "update", "destroy")


def _schema_columns(root: str) -> dict[str, list[str]]:
    """Table -> columns, read from the Rails schema. Generated, and authoritative."""
    text = _read(root, os.path.join("db", "schema.rb"))
    tables: dict[str, list[str]] = {}
    table = None
    for line in text.splitlines():
        created = re.match(r'\s*create_table "(\w+)"', line)
        if created:
            table = created.group(1)
            tables[table] = []
            continue
        column = re.match(r'\s*t\.\w+ "(\w+)"', line)
        if table and column:
            tables[table].append(column.group(1))
        elif line.strip() == "end":
            table = None
    return tables


#: Columns that are plumbing, not data anyone asked for. The second group is
#: Devise's own schema: listing a user's failed_attempts as an unreturned
#: feature buries the one that matters (current_streak) in framework noise.
_INTERNAL = re.compile(r"^(id|created_at|updated_at|.*_id|encrypted_.*|.*_digest|"
                       r".*_token|.*_count|deleted_at|discarded_at|lock_version|"
                       r"confirmation_.*|confirmed_at|unconfirmed_email|jti|"
                       r"reset_password_.*|remember_created_at|failed_attempts|"
                       r"locked_at|current_sign_in_.*|last_sign_in_.*|sign_in_.*)$")


def find_missing(root: str, endpoints: list[Endpoint]) -> list[Gap]:
    """Absences the repository's own shape makes visible."""
    out: list[Gap] = []

    # 1. A resource whose siblings all offer an action and it does not. Only
    #    for controllers already doing three of the five, so a single-purpose
    #    controller is not badgered into being a CRUD resource.
    by_controller: dict[str, set[str]] = defaultdict(set)
    for endpoint in endpoints:
        if not endpoint.controller.startswith("rails/"):
            by_controller[endpoint.controller].add(endpoint.action)
    for controller, actions in sorted(by_controller.items()):
        present = [a for a in CRUD if a in actions]
        if len(present) < 3:
            continue
        for action in CRUD:
            if action not in actions:
                out.append(Gap("incomplete resource",
                               f"serves {', '.join(present)} but not {action}",
                               controller, severity="low"))

    # 2. A column stored and never returned. Data captured and unreachable is
    #    either a feature that was never finished or a column to drop.
    tables = _schema_columns(root)
    serializers = {os.path.basename(p).replace("_serializer.rb", ""): _read(root, p)
                   for p in _under(root, "app", "serializers")}
    for name, text in serializers.items():
        columns = tables.get(name + "s") or tables.get(name) or []
        unexposed = [c for c in columns
                     if not _INTERNAL.match(c) and not re.search(rf"\b{re.escape(c)}\b", text)]
        if unexposed:
            out.append(Gap("stored but never returned",
                           f"{name}: {', '.join(sorted(unexposed))} "
                           f"{'is' if len(unexposed) == 1 else 'are'} in the table and in no "
                           "serializer — an unfinished feature, or a column to drop",
                           f"app/serializers/{name}_serializer.rb", severity="medium"))

    # There was a third finder here — "has_many :x with no route serving x" —
    # and it is gone because association names and route names need not
    # correspond. `has_many :recent_searches` is served at /searches/recent,
    # `calendar_events` at /calendar, `friendships` at /friends. Matching
    # literally reported eighteen absences of which roughly none were real,
    # and most of the rest were aliases (`class_name: "List"`) or join tables.
    # The mapping is not mechanically derivable, so this belongs in narration.
    return out


# --- assembling ---------------------------------------------------------------


def inventory(root: str, *, routes: bool = True) -> Report:
    """Read the repository. No model, no network, no cost — just files.

    ``routes=False`` skips booting the application, which is the slow part and
    the part that needs a working environment.
    """
    root = os.path.abspath(root)
    kind = detect(root)
    report = Report(root=root, kind=kind)
    specs = _spec_index(root)

    endpoints = rails_routes(root) if (kind == "rails" and routes) else []
    if kind == "rails":
        report.gaps = rails_gaps(root, endpoints, specs)
        buckets: dict[str, Feature] = {}
        for endpoint in endpoints:
            if endpoint.controller.startswith("rails/"):
                continue
            name = feature_name(endpoint.controller)
            feature = buckets.setdefault(name, Feature(name=name))
            feature.endpoints.append(endpoint)
        # The routes define the features: they are what the application *does*.
        # Everything else attaches to one, and what attaches to none is support
        # code listed as such — inventing a feature per service file produced
        # 136 "features" for an app with 25 controllers, which is a filing
        # system pretending to be an understanding.
        support = Feature(name="supporting code (attached to no endpoint)")
        for kinds, attr in (("models", "models"), ("services", "services"),
                            ("jobs", "jobs"), ("policies", "policies")):
            for path in _under(root, "app", kinds):
                home = next((f for n, f in buckets.items() if _matches(n, path)), None)
                getattr(home or support, attr).append(path)
        for name, feature in buckets.items():
            feature.specs = sorted(p for p in specs if _matches(name, p))
            feature.gaps = [g for g in report.gaps
                            if feature_name(g.where) == name]
        report.gaps = [g for g in report.gaps
                       if g not in [x for f in buckets.values() for x in f.gaps]]
        report.features = sorted(buckets.values(), key=lambda f: -len(f.endpoints))
        if any([support.models, support.services, support.jobs, support.policies]):
            report.features.append(support)
    else:
        report.gaps = python_gaps(root, specs) if kind == "python" else []
        buckets = {}
        for path in code_files(root, suffixes=(".py",)):
            if re.search(r"(^|/)(tests?|docs|examples)/", path):
                continue
            name = feature_name(path)
            feature = buckets.setdefault(name, Feature(name=name))
            feature.models.append(path)
        for name, feature in buckets.items():
            feature.specs = sorted(p for p in specs if _matches(name, p))
        report.features = sorted(buckets.values(), key=lambda f: f.name)

    if kind == "rails":
        report.missing = find_missing(root, endpoints)
    report.removable = survey(root)
    report.stats = {
        "files": len(code_files(root)),
        "lines": int(sum(line_count(root, p) for p in code_files(root))),
        "endpoints": len(endpoints),
        "features": len(report.features),
        "spec files": len(specs),
        "gaps": len(report.all_gaps),
    }
    return report


# --- rendering ----------------------------------------------------------------


def render(report: Report, *, limit: int = 0) -> str:
    """The report as markdown. Extracted facts only, unless describe() ran."""
    out: list[str] = []
    w = out.append

    w(f"# {os.path.basename(report.root)} — repository report")
    w("")
    w(f"*{report.kind} project · " + " · ".join(
        f"{v} {k}" for k, v in report.stats.items()) + "*")
    w("")
    w("Everything below is read from the repository. Nothing is inferred about "
      "intent except where a line is marked *narration*.")
    w("")

    w("## What it does")
    w("")
    if not report.features:
        w("No features could be identified — the layout is not one this knows how to read.")
    for feature in report.features[:limit or len(report.features)]:
        w(f"### {feature.name}")
        if feature.narration:
            w("")
            w(f"*narration:* {feature.narration}")
        if feature.endpoints:
            w("")
            w("| endpoint | action | specs |")
            w("|---|---|---|")
            for e in feature.endpoints:
                w(f"| `{e.name}` | {e.controller}#{e.action} | "
                  f"{len(e.specs) or '**none**'} |")
        for label, items in (("models", feature.models), ("services", feature.services),
                             ("jobs", feature.jobs), ("policies", feature.policies)):
            if items:
                w("")
                w(f"- **{label}**: " + ", ".join(f"`{os.path.basename(i)}`" for i in items))
        if not feature.tested:
            w("")
            w("- ⚠ **no spec file matches this feature by name**")
        w("")

    w("## What is missing")
    w("")
    gaps = sorted(report.all_gaps, key=lambda g: {"high": 0, "medium": 1, "low": 2}[g.severity])
    if not gaps:
        w("Nothing the repository's own conventions call for is absent.")
    else:
        w("| severity | kind | where | detail |")
        w("|---|---|---|---|")
        for gap in gaps:
            w(f"| {gap.severity} | {gap.kind} | `{gap.where}` | {gap.detail} |")
    w("")

    w("## What is not there yet")
    w("")
    w("Each of these is the repository contradicting itself — an action its "
      "siblings offer, a column it stores and never returns, an association "
      "nothing serves. What *ought* to exist beyond that is a matter of "
      "judgement, and appears only as narration.")
    w("")
    if not report.missing:
        w("Nothing. Every resource is complete, every column is reachable, every "
          "association is served.")
    else:
        w("| kind | where | detail |")
        w("|---|---|---|")
        for gap in sorted(report.missing, key=lambda g: g.kind):
            w(f"| {gap.kind} | `{gap.where}` | {gap.detail} |")
    w("")

    w("## What could go")
    w("")
    if not report.removable:
        w("Nothing measurable. (Dead-code detection is opt-in: it cannot see "
          "convention-based wiring and will call a whole Rails app dead.)")
    else:
        w("| metric | now | target |")
        w("|---|---|---|")
        for c in report.removable:
            w(f"| {c.metric} | {c.before:g} | {c.target:g} |")
    w("")
    return "\n".join(out)


# --- the optional narration ---------------------------------------------------

_DESCRIBE = (
    "You are given facts extracted from a repository: a feature's endpoints, "
    "models, services and specs. In two or three sentences, say what this "
    "feature does and what it is evidently *for*.\n\n"
    "Rules: describe only what the facts show. Do not speculate about features "
    "that are not listed, do not recommend anything, and if the facts are too "
    "thin to tell, say exactly that.\n\n"
)


def describe(report: Report, client: Any, *, limit: int = 12) -> Report:
    """Add one narration line per feature, from a model, from the facts only.

    Separate from :func:`inventory` on purpose. Extraction is checkable and free;
    narration is neither, so it is opt-in, marked in the output, and given
    nothing but what was extracted. A feature it cannot describe from the facts
    is left blank rather than filled in.
    """
    for feature in report.features[:limit]:
        facts = {
            "feature": feature.name,
            "endpoints": [f"{e.name} -> {e.controller}#{e.action}" for e in feature.endpoints],
            "models": [os.path.basename(m) for m in feature.models],
            "services": [os.path.basename(s) for s in feature.services],
            "jobs": [os.path.basename(j) for j in feature.jobs],
            "specs": [os.path.basename(s) for s in feature.specs],
        }
        try:
            reply = client.complete(_DESCRIBE + json.dumps(facts, indent=2))
        except Exception:  # noqa: BLE001 - a missing narration is not a failed report
            continue
        feature.narration = reply.text.strip()
    return report
