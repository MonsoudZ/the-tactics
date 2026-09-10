"""Survey a repository for work worth doing — with the number that says it worked.

The rest of this framework waits to be told what to do. This finds the work, and
the reason it can is that the interesting kind of cleanup is *more* measurable
than feature work, not less.

Feature work has a self-grading problem: the agent writes the test that judges
it, so a green check after the run can mean very little (hence
``AgentWorkspace.proves_itself``). A refactor has no such problem. The agent
cannot write the tests that grade it, because they already exist — the reward is
**a number moving in the right direction while the suite you already had stays
green**. Nothing self-reported anywhere in that sentence.

So every candidate this module posts arrives with a `measure` and a `target`.
"Make it cleaner" cannot be expressed here, which is the point: a proposal that
cannot fail cannot be scored, cannot be rejected by a critic, and teaches the
policy nothing. What *can* be expressed:

  * a file that has grown past the point anyone reads it whole (``oversized``)
  * the same block of code living in several places (``duplication``)
  * something nothing refers to any more (``unreferenced``)
  * work that was started and left (``unfinished``)

Each is a `Candidate` whose `measure(tree)` recomputes the metric in whichever
worktree is being judged, so the reward is re-measured against the ant's own
tree exactly like the check command is.

    from tactics.playbooks.survey import survey
    for c in survey("."):
        print(c.metric, c.before, "->", c.target, c.description)
"""

from __future__ import annotations

import hashlib
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Iterable

# Text files worth reading. Everything else is noise or binary.
CODE_SUFFIXES = (".py", ".rb", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java",
                 ".swift", ".kt", ".c", ".h", ".cc", ".cpp", ".cs", ".php", ".ex", ".exs")

# Directories that are never the author's own code.
SKIP_DIRS = {".git", "node_modules", "vendor", "tmp", "log", "build", "dist", "coverage",
             "__pycache__", ".venv", "venv", ".bundle", "storage", ".tactics", "public",
             # Migrations are write-once and already run in production. A long
             # `change` method is normal there and refactoring one is work
             # nobody does — the Rails survey offered three of them as its top
             # findings. Django and Rails both use these names.
             "migrate", "migrations"}

# A line that is only punctuation or a keyword carries no duplication signal.
_TRIVIAL = re.compile(r"^[\s\}\)\]\{\(\[;,]*$|^\s*(end|else|fi|done|\}\selse\s\{)\s*$")

# The marker must be *in a comment*. Two false positives from the first run on
# this repo made the case: this module's own pattern matched itself, and prose
# in a docstring counted. And `raise NotImplementedError` is gone entirely —
# it is the ordinary way to spell an abstract method in Python, so it finds
# more base classes than unfinished work. A finder that cries wolf gets ignored.
_UNFINISHED = re.compile(r"(?:#|//|--|/\*)\s*\b(TODO|FIXME|HACK|XXX)\b", re.IGNORECASE)


@dataclass
class Candidate:
    """One piece of work found by surveying, and how to tell whether it landed.

    ``measure`` is re-run inside whichever tree is being judged: the metric is
    measured, never remembered, for the same reason the check command is re-run
    rather than trusted. ``ok(tree)`` is the whole definition of done, and a
    candidate that cannot answer it has no business being posted.
    """

    kind: str
    description: str
    metric: str
    before: float
    target: float
    measure: Callable[[str], float] = field(repr=False)
    paths: list[str] = field(default_factory=list)
    #: True when the metric should go up rather than down. Nothing does yet;
    #: it exists so a coverage-style candidate does not need a special case.
    higher_is_better: bool = False

    def value(self, tree: str) -> float:
        return self.measure(tree)

    def ok(self, tree: str) -> tuple[bool, str]:
        """Did this land? Measured in ``tree``, never assumed."""
        now = self.value(tree)
        hit = now >= self.target if self.higher_is_better else now <= self.target
        arrow = "≥" if self.higher_is_better else "≤"
        return hit, f"{self.metric}: {now:g} (was {self.before:g}, needs {arrow} {self.target:g})"

    @property
    def headline(self) -> str:
        return f"[{self.kind}] {self.description}"


# --- reading the tree ---------------------------------------------------------


def code_files(tree: str, *, suffixes: tuple[str, ...] = CODE_SUFFIXES) -> list[str]:
    """Every source file under ``tree``, as paths relative to it."""
    found: list[str] = []
    for root, dirs, names in os.walk(tree):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        for name in names:
            if name.endswith(suffixes):
                found.append(os.path.relpath(os.path.join(root, name), tree))
    return sorted(p for p in found if not is_generated(tree, p))


# Files a tool writes and a person does not. Refactoring one is work the next
# `rails db:migrate` or `npm install` silently throws away — found by surveying
# a Rails app, which offered db/schema.rb as its largest file.
GENERATED_NAMES = ("schema.rb", "structure.sql", "Gemfile.lock", "package-lock.json",
                   "yarn.lock", "poetry.lock", "Cargo.lock", "go.sum")
_GENERATED_HEADER = re.compile(
    r"auto[- ]?generated|automatically generated|do not (edit|modify)|generated by",
    re.IGNORECASE)


def is_generated(tree: str, path: str) -> bool:
    """Does this file say a tool wrote it? Its own header is the authority."""
    if os.path.basename(path) in GENERATED_NAMES:
        return True
    with_head = _read(tree, path)[:2000]
    head = "\n".join(with_head.splitlines()[:12])
    return bool(_GENERATED_HEADER.search(head))


def _read(tree: str, path: str) -> str:
    try:
        with open(os.path.join(tree, path), encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return ""


def line_count(tree: str, path: str) -> float:
    text = _read(tree, path)
    return float(text.count("\n") + (1 if text and not text.endswith("\n") else 0))


# --- the finders --------------------------------------------------------------


def find_oversized(tree: str, *, limit: int = 600, target_ratio: float = 0.7) -> list[Candidate]:
    """Files long enough that nobody reads them whole.

    The target is a *ratio* rather than a fixed number because "under 400 lines"
    means something different for a 700-line file and a 3000-line one. Splitting
    is left to the agent; this only says how far it has to get.
    """
    out = []
    for path in code_files(tree):
        lines = line_count(tree, path)
        if lines < limit:
            continue
        target = round(lines * target_ratio)
        out.append(Candidate(
            kind="oversized",
            description=(f"{path} is {lines:g} lines. Split it into coherent units so no "
                         f"file exceeds {target:g} lines, without changing behaviour."),
            metric=f"lines in {path}",
            before=lines,
            target=float(target),
            measure=lambda t, p=path: line_count(t, p),
            paths=[path],
        ))
    return sorted(out, key=lambda c: -c.before)


def _blocks(text: str, window: int) -> Iterable[tuple[str, int]]:
    """Normalised sliding windows of non-trivial lines, with their start line."""
    lines = [(i, ln.strip()) for i, ln in enumerate(text.splitlines(), 1)]
    lines = [(i, ln) for i, ln in lines if ln and not _TRIVIAL.match(ln)]
    for start in range(0, max(0, len(lines) - window + 1)):
        chunk = lines[start:start + window]
        digest = hashlib.sha1("\n".join(ln for _, ln in chunk).encode()).hexdigest()
        yield digest, chunk[0][0]


def find_duplication(tree: str, *, window: int = 12, min_copies: int = 3) -> list[Candidate]:
    """The same block of code living in several places.

    Whole-line hashing over a sliding window: crude next to a real clone
    detector, and deliberately so — it reports only exact repetition, which is
    the kind nobody argues about. ``min_copies`` defaults to three because two
    copies are frequently a coincidence and three rarely are.
    """
    seen: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for path in code_files(tree):
        for digest, line in _blocks(_read(tree, path), window):
            seen[digest].append((path, line))

    out = []
    for digest, hits in seen.items():
        files = sorted({p for p, _ in hits})
        if len(hits) < min_copies or len(files) < 2:
            continue
        out.append(Candidate(
            kind="duplication",
            description=(f"the same {window}-line block appears {len(hits)} times across "
                         f"{', '.join(files[:4])}{'…' if len(files) > 4 else ''}. Extract it "
                         f"into one definition and call it from each site."),
            metric=f"copies of block {digest[:8]}",
            before=float(len(hits)),
            target=1.0,
            measure=lambda t, d=digest, w=window: float(sum(
                1 for p in code_files(t) for dig, _ in _blocks(_read(t, p), w) if dig == d)),
            paths=files,
        ))
    # Longest-carrying first: the block in the most places is the best trade.
    return sorted(out, key=lambda c: -c.before)[:20]


def find_unfinished(tree: str) -> list[Candidate]:
    """Work that was started and left — TODOs, FIXMEs, unimplemented stubs.

    One candidate per file rather than per marker: a file with six TODOs is one
    piece of work, and six tasks racing on one file would collide.
    """
    out = []
    for path in code_files(tree):
        marks = _UNFINISHED.findall(_read(tree, path))
        if not marks:
            continue
        count = float(len(marks))
        out.append(Candidate(
            kind="unfinished",
            description=(f"{path} carries {count:g} unfinished marker(s) (TODO/FIXME/"
                         f"not-implemented). Finish the work or delete the marker if it is "
                         f"no longer true — do not simply remove the comment."),
            metric=f"unfinished markers in {path}",
            before=count,
            target=0.0,
            measure=lambda t, p=path: float(len(_UNFINISHED.findall(_read(t, p)))),
            paths=[path],
        ))
    return sorted(out, key=lambda c: -c.before)


# Wired by convention, not by name. Rails routes say `resources :friends` and
# never mention FriendsController; Django, Spring and Rails all autoload whole
# directories. A name search over such a tree concludes the entire application
# is dead — surveying focusmate-api reported 188 unreferenced definitions, one
# for every controller, job and serializer in it.
CONVENTION_DIRS = ("app/", "config/", "db/", "lib/tasks/", "migrations/")
CONVENTION_SUFFIXES = ("Controller", "Job", "Mailer", "Serializer", "Channel",
                       "Policy", "Middleware", "Migration", "Helper")

_DEF = {
    ".py": re.compile(r"^\s*(?:def|class)\s+([A-Za-z_]\w*)", re.M),
    ".rb": re.compile(r"^\s*(?:def|class|module)\s+([A-Za-z_]\w*)", re.M),
}


def find_unreferenced(tree: str, *, ignore: tuple[str, ...] = ("_", "test", "spec")) -> list[Candidate]:
    """Definitions nothing else mentions — the dead ends. **Opt-in, and here is why.**

    A name search cannot see convention-based wiring, and the first run of this
    finder against a real Rails application returned 188 candidates: every
    controller, job and serializer in the app, because `resources :friends`
    never spells `FriendsController`. Acting on that would have deleted a
    production API. So this finder is *not* in the default survey — ask for it
    by name — and it skips the places frameworks autoload from and the class
    names they dispatch to.

    What survives those filters is still a heuristic. `send`, `const_get`,
    string dispatch and reflection all defeat it, which is why the candidate
    says "confirm it is unreachable" rather than "delete it", and why the suite
    staying green is what makes acting on one safe. This points; it does not
    conclude.
    """
    names: dict[str, str] = {}
    for path in code_files(tree, suffixes=tuple(_DEF)):
        if path.startswith(CONVENTION_DIRS):
            continue                      # the framework wires these, not the code
        pattern = _DEF[os.path.splitext(path)[1]]
        for name in pattern.findall(_read(tree, path)):
            if name.startswith(ignore) or name.endswith(CONVENTION_SUFFIXES):
                continue
            names.setdefault(name, path)

    corpus = {p: _read(tree, p) for p in code_files(tree)}
    out = []
    for name, path in names.items():
        uses = sum(len(re.findall(rf"\b{re.escape(name)}\b", text))
                   for p, text in corpus.items())
        if uses > 1:                      # its own definition is one mention
            continue
        out.append(Candidate(
            kind="unreferenced",
            description=(f"`{name}` in {path} is defined and never mentioned anywhere else. "
                         f"Confirm it is unreachable — dynamic dispatch defeats a name search "
                         f"— and remove it and anything it alone kept alive."),
            metric=f"mentions of {name}",
            before=1.0,
            target=0.0,
            measure=lambda t, n=name: float(sum(
                len(re.findall(rf"\b{re.escape(n)}\b", _read(t, p))) for p in code_files(t))),
            paths=[path],
        ))
    return out


# --- units that are hard to hold in your head ---------------------------------
#
# The classic refactor shape, and the one that scores most honestly: a number
# comes down while the tests you already had stay green, so nothing in the
# reward is self-reported. Three measurements, all objective, all with an
# obvious after-state.
#
# Every one of them is measured **per file, on the worst case in it**, never on
# a named function. Naming one invites the two ways this gets gamed: rename it
# and the metric cannot find it, delete it and the metric reads zero. "The
# longest function in this file" survives both — it can only improve by the
# file actually getting easier to read.

#: Where a function begins, for languages with no parser here. Ruby only: its
#: `def` plus indentation is a strong enough convention to measure *length*.
#: Nesting and argument counts are not attempted from text — the first version
#: tried and reported a wrapped call continuation as 56 columns of nesting,
#: which is line-wrapping, not depth.
_FUNCTION = {".rb": re.compile(r"^(?P<indent>\s*)def\s+\w+")}

#: Block statements that genuinely nest. Used against Python's own parser, so
#: this is exact rather than a guess.
_NESTS = ("If", "For", "AsyncFor", "While", "With", "AsyncWith", "Try", "Match",
          "ExceptHandler")


def _required(args) -> int:
    """How many arguments a caller *must* supply, in order.

    Not how many the signature has. `Colony.__init__` declares fourteen — three
    positional and eleven keyword-only options with defaults — and
    `Colony(target, tactics, planner, max_workers=1)` is not a call site anyone
    struggles with. Counting all fourteen offered this framework's own
    configuration constructor as work, and the refactor would have made it
    worse. A named option with a default costs the caller nothing; what has to
    be held in your head is the run of positionals you cannot skip, cannot name,
    and must get the order of. `*args`/`**kwargs` are optional by definition.
    """
    positional = args.posonlyargs + args.args
    supplied = [a for a in positional if a.arg not in ("self", "cls")]
    without_default = supplied[:len(supplied) - len(args.defaults)] if args.defaults else supplied
    mandatory_keywords = sum(1 for d in args.kw_defaults if d is None)
    return len(without_default) + mandatory_keywords


def _python_units(text: str) -> list[tuple[str, int, int, int]]:
    """(name, length, nesting depth, argument count) for each Python function.

    Parsed, not pattern-matched. Python ships a parser and using anything else
    here would be choosing to be wrong: an annotation like
    `Callable[[str, int], None]` reads as three parameters to a comma split, and
    a wrapped argument list reads as deep nesting to an indentation count. Both
    were live false positives before this was rewritten.
    """
    import ast

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []       # not our file to judge

    def depth(node, level=0):
        deepest = level
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue        # a nested def starts its own reckoning
            step = 1 if type(child).__name__ in _NESTS else 0
            deepest = max(deepest, depth(child, level + step))
        return deepest

    out = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        length = (node.end_lineno or node.lineno) - node.lineno + 1
        out.append((node.name, length, depth(node), _required(node.args)))
    return out


def _ruby_lengths(text: str) -> list[int]:
    """Function lengths in Ruby, by `def` and indentation. Length only."""
    lines = text.splitlines()
    out = []
    for i, line in enumerate(lines):
        match = _FUNCTION[".rb"].match(line)
        if not match:
            continue
        base = len(match.group("indent"))
        body = 0
        for follower in lines[i + 1:]:
            if follower.strip() and len(follower) - len(follower.lstrip()) <= base:
                break
            body += 1
        out.append(body + 1)
    return out


def _longest_function(tree: str, path: str) -> float:
    if path.endswith(".py"):
        return float(max((length for _n, length, _d, _a in _python_units(_read(tree, path))),
                         default=0))
    return float(max(_ruby_lengths(_read(tree, path)), default=0))


def find_long_functions(tree: str, *, limit: int = 60, target_ratio: float = 0.6) -> list[Candidate]:
    """Files containing a function too long to read in one sitting.

    Measured on the *longest* function in the file, never on a named one. Naming
    invites the two ways this gets gamed: rename it and the metric cannot find
    it, delete it and the metric reads zero. The file's worst case can only
    improve by the file genuinely getting easier to read.
    """
    out = []
    for path in code_files(tree, suffixes=(".py", ".rb")):
        longest = _longest_function(tree, path)
        if longest < limit:
            continue
        out.append(Candidate(
            kind="long function",
            description=(f"the longest function in {path} runs {longest:g} lines. Break it "
                         f"into named pieces so none exceeds "
                         f"{round(longest * target_ratio)}, without changing behaviour — "
                         "the tests you already have are the proof."),
            metric=f"longest function in {path}",
            before=longest,
            target=float(round(longest * target_ratio)),
            measure=lambda t, p=path: _longest_function(t, p),
            paths=[path],
        ))
    return sorted(out, key=lambda c: -c.before)


def find_deep_nesting(tree: str, *, limit: int = 5) -> list[Candidate]:
    """Python files with a function nested deeper than anyone tracks comfortably.

    Depth in *block levels*, from Python's own parser — an `if` inside a `for`
    inside a `with` is three. Python only: nothing here can parse Ruby, and
    counting indentation instead measured line-wrapping.
    """
    out = []
    for path in code_files(tree, suffixes=(".py",)):
        units = _python_units(_read(tree, path))
        deepest = max((d for _n, _l, d, _a in units), default=0)
        if deepest < limit:
            continue
        out.append(Candidate(
            kind="deep nesting",
            description=(f"a function in {path} nests {deepest} block levels deep. Flatten "
                         "it — early returns, guard clauses, an extracted predicate — "
                         "without changing behaviour."),
            metric=f"deepest nesting in {path}",
            before=float(deepest),
            target=float(limit - 1),
            measure=lambda t, p=path: float(max(
                (d for _n, _l, d, _a in _python_units(_read(t, p))), default=0)),
            paths=[path],
        ))
    return sorted(out, key=lambda c: -c.before)


def find_long_signatures(tree: str, *, limit: int = 6) -> list[Candidate]:
    """Python functions demanding more *required* arguments than a caller can hold.

    Two narrowings, both from being wrong on this repository. From the parser,
    so `Callable[[str, int], None]` is one parameter rather than the three a
    comma split reported. And only what the caller must supply, in order — see
    :func:`_required`, which is where a fourteen-argument finding turned out to
    be a keyword-only constructor nobody has ever mis-called.
    """
    out = []
    for path in code_files(tree, suffixes=(".py",)):
        widest = max((a for _n, _l, _d, a in _python_units(_read(tree, path))), default=0)
        if widest <= limit:
            continue
        out.append(Candidate(
            kind="long signature",
            description=(f"a function in {path} requires {widest} arguments before a caller "
                         f"can name anything. Group the related ones into an object, give "
                         f"the incidental ones defaults, or split the function, so none "
                         f"requires more than {limit}."),
            metric=f"most required arguments in {path}",
            before=float(widest),
            target=float(limit),
            measure=lambda t, p=path: float(max(
                (a for _n, _l, _d, a in _python_units(_read(t, p))), default=0)),
            paths=[path],
        ))
    return sorted(out, key=lambda c: -c.before)


#: What a plain ``survey()`` runs. The last three read the same in every
#: language — a long file is long, a repeated block is repeated, a TODO is a
#: TODO — and the first three need a parser, so they measure only what they can
#: measure exactly and stay silent elsewhere. A finder that guesses on a
#: language it cannot read is worse than one that skips it.
FINDERS: dict[str, Callable[..., list[Candidate]]] = {
    "long function": find_long_functions,
    "deep nesting": find_deep_nesting,
    "long signature": find_long_signatures,
    "oversized": find_oversized,
    "duplication": find_duplication,
    "unfinished": find_unfinished,
}

#: Ask for these by name. ``unreferenced`` is accurate on a codebase that wires
#: itself explicitly and wildly wrong on one that wires itself by convention,
#: and the difference is not something this module can detect for you.
OPTIONAL_FINDERS: dict[str, Callable[..., list[Candidate]]] = {
    "unreferenced": find_unreferenced,
}


def survey(tree: str, *, kinds: Iterable[str] | None = None) -> list[Candidate]:
    """Everything worth doing here, biggest win first.

    Pure reading: no shell, no model, no cost. A survey that spent money to tell
    you what it found would not get run often enough to be useful.
    """
    wanted = list(kinds) if kinds else list(FINDERS)
    found: list[Candidate] = []
    for kind in wanted:
        finder = FINDERS.get(kind) or OPTIONAL_FINDERS.get(kind)
        if finder:
            found.extend(finder(tree))
    # Rank by how much of the metric is on the table, normalised so line counts
    # do not drown out duplication counts.
    return sorted(found, key=lambda c: -(c.before - c.target) / max(c.before, 1.0))
