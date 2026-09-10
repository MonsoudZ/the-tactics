"""Finding the work, with the number that says it landed.

A refactor is the *easiest* thing here to score honestly: the agent cannot write
the tests that grade it, because they already exist. So every candidate carries a
metric, and these check the metrics are real — including the two false positives
the first live survey of this repo produced.
"""

from __future__ import annotations

import pathlib

from tactics.playbooks.survey import (
    Candidate,
    code_files,
    find_deep_nesting,
    find_duplication,
    find_long_functions,
    find_long_signatures,
    find_oversized,
    find_unfinished,
    find_unreferenced,
    survey,
)


def _tree(tmp_path, files: dict[str, str]) -> str:
    for name, body in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    return str(tmp_path)


# --- the metric is measured, never remembered ---------------------------------


def test_a_candidate_is_judged_against_the_tree_it_is_given(tmp_path):
    # The whole contract: `before` is history, `ok()` is a fresh measurement in
    # whichever worktree is being judged — the same discipline as the check.
    big = "\n".join(f"line {i}" for i in range(800))
    tree = _tree(tmp_path / "before", {"big.py": big})
    candidate = find_oversized(tree)[0]
    assert candidate.before == 800
    assert not candidate.ok(tree)[0]

    after = _tree(tmp_path / "after", {"big.py": "\n".join(f"line {i}" for i in range(300))})
    passed, detail = candidate.ok(after)
    assert passed and "300" in detail


def test_the_reason_is_reported_in_the_units_of_the_metric(tmp_path):
    tree = _tree(tmp_path, {"big.py": "\n".join(str(i) for i in range(700))})
    _ok, detail = find_oversized(tree)[0].ok(tree)
    assert "lines in big.py" in detail and "was 700" in detail


# --- oversized ----------------------------------------------------------------


def test_a_short_file_is_not_work(tmp_path):
    assert find_oversized(_tree(tmp_path, {"small.py": "x = 1\n"})) == []


def test_the_target_scales_with_the_file(tmp_path):
    # "Under 400 lines" means different things for a 700-line file and a 3000-
    # line one, so the target is a ratio.
    tree = _tree(tmp_path, {"a.py": "\n".join(["x"] * 1000), "b.py": "\n".join(["y"] * 700)})
    targets = {c.metric: c.target for c in find_oversized(tree)}
    assert targets["lines in a.py"] == 700 and targets["lines in b.py"] == 490


# --- duplication --------------------------------------------------------------


def test_the_same_block_in_three_places_is_found(tmp_path):
    block = "\n".join(f"    step_{i}()" for i in range(14))
    tree = _tree(tmp_path, {
        "a.py": f"def one():\n{block}\n",
        "b.py": f"def two():\n{block}\n",
        "c.py": f"def three():\n{block}\n",
    })
    found = find_duplication(tree)
    assert found, "three identical blocks should be reported"
    assert found[0].target == 1.0
    assert {"a.py", "b.py", "c.py"} <= set(found[0].paths)


def test_one_copy_is_not_duplication(tmp_path):
    block = "\n".join(f"    step_{i}()" for i in range(14))
    assert find_duplication(_tree(tmp_path, {"a.py": f"def one():\n{block}\n"})) == []


def test_extracting_the_block_satisfies_the_candidate(tmp_path):
    block = "\n".join(f"    step_{i}()" for i in range(14))
    before = _tree(tmp_path / "before", {
        "a.py": f"def one():\n{block}\n", "b.py": f"def two():\n{block}\n",
        "c.py": f"def three():\n{block}\n"})
    candidate = find_duplication(before)[0]

    after = _tree(tmp_path / "after", {
        "shared.py": f"def shared():\n{block}\n",
        "a.py": "from shared import shared\n\n\ndef one():\n    shared()\n",
        "b.py": "from shared import shared\n\n\ndef two():\n    shared()\n",
        "c.py": "from shared import shared\n\n\ndef three():\n    shared()\n"})
    assert candidate.ok(after)[0]


# --- unfinished: both false positives the first live survey produced ----------


def test_a_marker_only_counts_inside_a_comment(tmp_path):
    # Found live: this module's own regex matched itself, and prose in a
    # docstring counted as unfinished work.
    tree = _tree(tmp_path, {
        "real.py": "# TODO: finish this\nx = 1\n",
        "pattern.py": 'MARKER = r"\\b(TODO|FIXME)\\b"\n',
        "prose.py": '"""Handles TODOs and FIXMEs written by other people."""\n',
    })
    assert [c.metric for c in find_unfinished(tree)] == ["unfinished markers in real.py"]


def test_an_abstract_method_is_not_unfinished_work(tmp_path):
    # `raise NotImplementedError` is how Python spells an abstract method, so
    # counting it found base classes rather than unfinished work.
    tree = _tree(tmp_path, {"base.py": "class T:\n    def observe(self):\n"
                                       "        raise NotImplementedError\n"})
    assert find_unfinished(tree) == []


def test_finishing_the_work_satisfies_the_candidate(tmp_path):
    before = _tree(tmp_path / "before", {"a.py": "# FIXME: handle the empty case\nx = 1\n"})
    candidate = find_unfinished(before)[0]
    after = _tree(tmp_path / "after", {"a.py": "x = 1 if True else 0\n"})
    assert candidate.ok(after)[0]


# --- unreferenced -------------------------------------------------------------


def test_a_definition_nothing_mentions_is_a_dead_end(tmp_path):
    tree = _tree(tmp_path, {
        "lib.py": "def used():\n    return 1\n\n\ndef orphan():\n    return 2\n",
        "app.py": "from lib import used\n\nprint(used())\n",
    })
    assert [c.paths for c in find_unreferenced(tree)] == [["lib.py"]]
    assert "orphan" in find_unreferenced(tree)[0].metric


def test_private_and_test_names_are_left_alone(tmp_path):
    tree = _tree(tmp_path, {"lib.py": "def _helper():\n    return 1\n\n\n"
                                      "def test_thing():\n    return 2\n"})
    assert find_unreferenced(tree) == []


# --- the walk -----------------------------------------------------------------


def test_vendored_and_generated_trees_are_not_your_code(tmp_path):
    tree = _tree(tmp_path, {
        "app.py": "x = 1\n",
        "node_modules/pkg/index.js": "y = 2\n",
        "vendor/bundle/thing.rb": "z = 3\n",
        "coverage/report.py": "w = 4\n",
    })
    assert code_files(tree) == ["app.py"]


def test_a_survey_ranks_by_how_much_is_on_the_table(tmp_path):
    tree = _tree(tmp_path, {
        "huge.py": "\n".join(["x"] * 900),
        "note.py": "# TODO: one thing\n",
    })
    kinds = [c.kind for c in survey(tree)]
    # The TODO is 100% removable; the file can only shed 30%. Proportion, not size.
    assert kinds.index("unfinished") < kinds.index("oversized")


def test_asking_for_one_kind_returns_only_that_kind(tmp_path):
    tree = _tree(tmp_path, {"huge.py": "\n".join(["x"] * 900), "note.py": "# TODO: x\n"})
    assert {c.kind for c in survey(tree, kinds=["oversized"])} == {"oversized"}


# --- what a name search cannot see --------------------------------------------


def test_a_convention_wired_app_is_not_reported_as_dead(tmp_path):
    # Found live: surveying a Rails API returned 188 unreferenced definitions —
    # every controller in it — because `resources :friends` never spells
    # FriendsController. Acting on that would have deleted a production API.
    tree = _tree(tmp_path, {
        "app/controllers/friends_controller.rb": "class FriendsController\n  def index\n  end\nend\n",
        "app/jobs/sync_job.rb": "class SyncJob\n  def perform\n  end\nend\n",
        "config/routes.rb": "Rails.application.routes.draw do\n  resources :friends\nend\n",
    })
    assert find_unreferenced(tree) == []


def test_the_dead_end_finder_is_not_in_the_default_survey(tmp_path):
    # Its accuracy depends on whether the codebase wires itself by name or by
    # convention, which this module cannot detect — so it is asked for, never
    # assumed.
    from tactics.playbooks.survey import FINDERS, OPTIONAL_FINDERS

    assert "unreferenced" not in FINDERS
    assert "unreferenced" in OPTIONAL_FINDERS

    tree = _tree(tmp_path, {"lib.py": "def orphan():\n    return 1\n"})
    assert survey(tree) == []
    assert [c.kind for c in survey(tree, kinds=["unreferenced"])] == ["unreferenced"]


def test_a_generated_file_is_not_work_for_a_person(tmp_path):
    # Found live: the largest file in a Rails app was db/schema.rb, which the
    # next migration rewrites. Its own header is the authority.
    from tactics.playbooks.survey import is_generated

    tree = _tree(tmp_path, {
        "db/schema.rb": "\n".join(["x"] * 900),
        "gen.py": "# This file was auto-generated. DO NOT EDIT.\n" + "\n".join(["y"] * 900),
        "mine.py": "\n".join(["z"] * 900),
    })
    assert is_generated(tree, "db/schema.rb") and is_generated(tree, "gen.py")
    assert not is_generated(tree, "mine.py")
    assert [c.paths for c in survey(tree)] == [["mine.py"]]


# --- units that are hard to hold in your head ---------------------------------


def test_a_long_function_is_measured_on_the_files_worst_case(tmp_path):
    # Never on a named function: rename it and the metric cannot find it,
    # delete it and the metric reads zero. The file's worst case can only
    # improve by the file genuinely getting easier to read.
    body = "\n".join(f"    step_{i} = {i}" for i in range(80))
    before = _tree(tmp_path / "before", {"a.py": f"def big():\n{body}\n"})
    candidate = find_long_functions(before)[0]
    assert candidate.before == 81

    after = _tree(tmp_path / "after", {"a.py": "def big():\n    return helper()\n\n\n"
                                              "def helper():\n    return 1\n"})
    assert candidate.ok(after)[0]


def test_deleting_the_function_does_not_win(tmp_path):
    body = "\n".join(f"    step_{i} = {i}" for i in range(80))
    before = _tree(tmp_path / "before", {"a.py": f"def big():\n{body}\n\n\ndef other():\n"
                                                 f"{body}\n"})
    candidate = find_long_functions(before)[0]
    # One of the two removed; the other is still enormous, so nothing improved.
    after = _tree(tmp_path / "after", {"a.py": f"def other():\n{body}\n"})
    assert not candidate.ok(after)[0]


def test_an_annotation_is_one_parameter_not_three(tmp_path):
    # Found live: a comma split read `Callable[[str, BriefSpec, X], AgentRun]`
    # as three parameters and reported a 10-argument function as taking 20.
    tree = _tree(tmp_path, {"a.py": "from typing import Callable\n\n\n"
                                    "def f(a, cb: Callable[[str, int, float], None], c, d):\n"
                                    "    return a\n"})
    assert find_long_signatures(tree, limit=2)[0].before == 4


def test_an_option_with_a_default_is_not_something_a_caller_holds(tmp_path):
    # Found live on this repository: `Colony.__init__` declares fourteen
    # arguments — three positional and eleven keyword-only with defaults — and
    # was offered as the top piece of work. `Colony(target, tactics, planner,
    # max_workers=1)` is not a call site anyone has ever mis-typed, and the
    # refactor would have made the framework worse. A finder that fires on good
    # design gets ignored, which costs the findings that were real.
    tree = _tree(tmp_path, {"a.py": "def f(a, b, *, c=1, d=2, e=3, f=4, g=5, h=6, i=7):\n"
                                    "    return a\n"})
    assert find_long_signatures(tree, limit=3) == []


def test_a_keyword_only_argument_with_no_default_still_counts(tmp_path):
    # It cannot be skipped, only named. Excluding it because it is keyword-only
    # would let a genuinely demanding signature hide behind a `*`.
    tree = _tree(tmp_path, {"a.py": "def f(a, *, b, c, d):\n    return a\n"})
    assert find_long_signatures(tree, limit=3)[0].before == 4


def test_star_args_are_optional_by_definition(tmp_path):
    tree = _tree(tmp_path, {"a.py": "def f(a, b, *rest, **kw):\n    return a\n"})
    assert find_long_signatures(tree, limit=2) == []


def test_self_is_not_an_argument_a_caller_passes(tmp_path):
    tree = _tree(tmp_path, {"a.py": "class T:\n    def f(self, a, b, c):\n        return a\n"})
    assert find_long_signatures(tree, limit=2)[0].before == 3


def test_a_wrapped_call_is_not_deep_nesting(tmp_path):
    # Found live: an argument list aligned under an open paren was reported as
    # 56 columns of nesting. That is line-wrapping, and the parser knows the
    # difference where an indentation count cannot.
    tree = _tree(tmp_path, {"a.py": "def f():\n"
                                    "    thing = call(one,\n"
                                    "                 two,\n"
                                    "                 three)\n"
                                    "    return thing\n"})
    assert find_deep_nesting(tree, limit=2) == []


def test_real_nesting_is_counted_in_block_levels(tmp_path):
    tree = _tree(tmp_path, {"a.py": "def f(xs):\n"
                                    "    for x in xs:\n"
                                    "        if x:\n"
                                    "            with open(x) as fh:\n"
                                    "                return fh\n"})
    assert find_deep_nesting(tree, limit=3)[0].before == 3


def test_a_nested_def_starts_its_own_reckoning(tmp_path):
    # An inner function is not "deeper" code; it is its own unit.
    tree = _tree(tmp_path, {"a.py": "def outer():\n"
                                    "    def inner():\n"
                                    "        if True:\n"
                                    "            return 1\n"
                                    "    return inner\n"})
    assert find_deep_nesting(tree, limit=2) == []


def test_a_file_python_cannot_parse_is_not_ours_to_judge(tmp_path):
    tree = _tree(tmp_path, {"broken.py": "def f(:\n  this is not python\n"})
    assert find_deep_nesting(tree) == [] and find_long_signatures(tree) == []


def test_ruby_gets_length_only(tmp_path):
    # Nothing here parses Ruby, so it is measured for length and left alone for
    # the rest rather than guessed at.
    body = "\n".join(f"    step_{i} = {i}" for i in range(70))
    tree = _tree(tmp_path, {"a.rb": f"def big\n{body}\nend\n"})
    assert find_long_functions(tree)[0].before >= 70
    assert find_deep_nesting(tree) == [] and find_long_signatures(tree) == []


def test_migrations_are_not_work_for_a_person(tmp_path):
    # Found live: three of the Rails survey's top five long functions were
    # migrations — write-once, already run in production, never refactored.
    body = "\n".join(f"    step_{i} = {i}" for i in range(80))
    tree = _tree(tmp_path, {
        "db/migrate/20260101_create_things.rb": f"def change\n{body}\nend\n",
        "app/models/thing.rb": f"def process\n{body}\nend\n",
    })
    assert [c.paths for c in find_long_functions(tree)] == [["app/models/thing.rb"]]
