"""The repository report — extracted facts only.

The rule these enforce: a report that invents a feature is worse than no report,
because it reads exactly like one that was careful, and it will be believed. So
everything traces to a file, gaps are absences rather than opinions, and the two
false-positive classes the first real run produced stay fixed.
"""

from __future__ import annotations

import pathlib

from tactics.playbooks.report import (
    Endpoint,
    Feature,
    Gap,
    Report,
    describe,
    detect,
    feature_name,
    inventory,
    rails_gaps,
    rails_routes,
    render,
)


def _tree(tmp_path, files: dict[str, str]) -> str:
    for name, body in files.items():
        path = pathlib.Path(tmp_path, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    return str(tmp_path)


# --- recognising the shape ----------------------------------------------------


def test_a_rails_app_is_recognised_by_its_router(tmp_path):
    assert detect(_tree(tmp_path, {"config/routes.rb": "Rails.application.routes.draw\n"})) == "rails"


def test_a_python_project_is_recognised_by_its_manifest(tmp_path):
    assert detect(_tree(tmp_path, {"pyproject.toml": "[project]\n"})) == "python"


def test_something_unrecognised_says_so(tmp_path):
    assert detect(_tree(tmp_path, {"main.c": "int main(){}\n"})) == "unknown"


def test_routes_are_empty_rather_than_invented_when_the_app_cannot_boot(tmp_path):
    # The alternative is a plausible list of routes that do not exist, which is
    # exactly the failure this module is built to avoid.
    assert rails_routes(_tree(tmp_path, {"config/routes.rb": "x\n"}), timeout=5) == []


# --- naming -------------------------------------------------------------------


def test_a_controller_names_its_feature():
    assert feature_name("api/v1/friend_requests_controller") == "friend requests"
    assert feature_name("api/v1/tasks_controller") == "tasks"


def test_the_api_version_is_not_part_of_the_name():
    assert "v1" not in feature_name("api/v1/lists_controller")


# --- gaps are absences, and the absence has to be real ------------------------


def _specs(text: str) -> dict[str, str]:
    return {"spec/requests/thing_spec.rb": text}


def test_a_route_with_parameters_is_matched_against_how_specs_write_it(tmp_path):
    # Found live: collapsing ":id" to nothing produced "/lists//tasks//complete",
    # which no spec contains, so eight thoroughly tested endpoints were reported
    # as untested. A gap list full of false positives is worse than none.
    endpoint = Endpoint("PATCH", "/api/v1/lists/:list_id/tasks/:id/complete",
                        "api/v1/tasks", "complete")
    specs = _specs('auth_patch "/api/v1/lists/#{list.id}/tasks/#{task.id}/complete", user: user')
    assert rails_gaps(str(tmp_path), [endpoint], specs) == []
    assert endpoint.specs == ["spec/requests/thing_spec.rb"]


def test_an_endpoint_no_spec_mentions_is_a_gap(tmp_path):
    endpoint = Endpoint("POST", "/api/v1/tasks/batch", "api/v1/tasks", "batch")
    gaps = rails_gaps(str(tmp_path), [endpoint], _specs("something else entirely"))
    assert [g.kind for g in gaps] == ["untested endpoint"]
    assert gaps[0].severity == "high"


def test_framework_routes_are_not_the_authors_problem(tmp_path):
    endpoint = Endpoint("GET", "/rails/info/routes", "rails/info", "routes")
    assert rails_gaps(str(tmp_path), [endpoint], _specs("")) == []


def test_opting_out_of_the_authorization_check_is_reported(tmp_path):
    root = _tree(tmp_path, {
        "app/controllers/api/v1/sync_controller.rb":
            "class SyncController\n  skip_after_action :verify_authorized\nend\n"})
    gaps = rails_gaps(root, [], {})
    assert [g.kind for g in gaps] == ["authorization opt-out"]
    assert "sync_controller" in gaps[0].where


def test_a_model_without_a_policy_is_only_a_gap_where_policies_are_the_convention(tmp_path):
    without = _tree(tmp_path / "a", {"app/models/task.rb": "class Task; end\n"})
    assert rails_gaps(without, [], {}) == []

    with_policies = _tree(tmp_path / "b", {
        "app/models/task.rb": "class Task; end\n",
        "app/models/list.rb": "class List; end\n",
        "app/policies/list_policy.rb": "class ListPolicy; end\n"})
    kinds = [(g.kind, g.where) for g in rails_gaps(with_policies, [], {})]
    assert ("model without policy", "app/models/task.rb") in kinds
    assert not any(w.endswith("list.rb") for _k, w in kinds)


# --- grouping -----------------------------------------------------------------


def test_the_routes_define_the_features_and_the_rest_attaches(tmp_path):
    # Found live: naming a feature after every service file produced 136
    # "features" for an app with 25 controllers — a filing system pretending to
    # be an understanding.
    root = _tree(tmp_path, {
        "config/routes.rb": "Rails.application.routes.draw\n",
        "app/services/task_update_service.rb": "class TaskUpdateService; end\n",
        "app/services/task_creation_service.rb": "class TaskCreationService; end\n",
        "app/models/task.rb": "class Task; end\n",
        "app/jobs/lonely_job.rb": "class LonelyJob; end\n",
    })
    report = inventory(root, routes=False)
    report.features = []                     # no routes booted, so group by hand
    feature = Feature(name="tasks")
    assert feature_name("app/services/task_update_service.rb").startswith("task")

    # With a route present, the services land under it rather than beside it.
    from tactics.playbooks.report import _matches

    assert _matches("tasks", "app/services/task_update_service.rb")
    assert _matches("tasks", "app/models/task.rb")
    assert not _matches("tasks", "app/jobs/lonely_job.rb")


def test_a_python_project_reports_untested_modules(tmp_path):
    root = _tree(tmp_path, {
        "pyproject.toml": "[project]\n",
        "src/thing/engine.py": "def go():\n    return 1\n",
        "src/thing/lonely.py": "def solo():\n    return 2\n",
        "tests/test_engine.py": "from thing.engine import go\n\n\ndef test_go():\n    assert go()\n",
    })
    report = inventory(root)
    untested = [g.detail for g in report.all_gaps if g.kind == "untested module"]
    assert any("lonely" in d for d in untested)
    assert not any("engine" in d for d in untested)


# --- rendering ----------------------------------------------------------------


def _report() -> Report:
    feature = Feature(name="tasks", endpoints=[Endpoint("GET", "/tasks", "tasks", "index")])
    return Report(root="/repo", kind="rails", features=[feature],
                  gaps=[Gap("untested endpoint", "no spec mentions it", "tasks", "high")],
                  stats={"endpoints": 1})


def test_the_report_says_which_half_is_extracted(tmp_path):
    text = render(_report())
    assert "Everything below is read from the repository" in text
    assert "## What it does" in text and "## What is missing" in text
    assert "## What could go" in text


def test_narration_is_labelled_as_narration():
    class _Client:
        def complete(self, prompt, **kw):
            return type("R", (), {"text": "Tasks are created and completed."})()

    text = render(describe(_report(), _Client()))
    assert "*narration:* Tasks are created and completed." in text


def test_a_model_that_cannot_answer_leaves_the_feature_undescribed():
    # Silence beats invention: an unavailable model must not become an empty
    # confident sentence.
    class _Broken:
        def complete(self, prompt, **kw):
            raise RuntimeError("no")

    report = describe(_report(), _Broken())
    assert report.features[0].narration == ""
    assert "*narration:*" not in render(report)


def test_the_narrator_is_told_not_to_invent():
    seen = {}

    class _Client:
        def complete(self, prompt, **kw):
            seen["prompt"] = prompt
            return type("R", (), {"text": "ok"})()

    describe(_report(), _Client())
    assert "Do not speculate" in seen["prompt"]
    assert "too thin to tell" in seen["prompt"]


# --- what is not there yet ----------------------------------------------------


def test_a_resource_missing_what_its_siblings_offer(tmp_path):
    endpoints = [Endpoint(v, "/x", "api/v1/things", a) for v, a in
                 [("GET", "index"), ("POST", "create"), ("PATCH", "update"), ("DELETE", "destroy")]]
    from tactics.playbooks.report import find_missing

    gaps = [g for g in find_missing(str(tmp_path), endpoints) if g.kind == "incomplete resource"]
    assert [g.detail for g in gaps] == ["serves index, create, update, destroy but not show"]


def test_a_single_purpose_controller_is_not_badgered_into_being_a_resource(tmp_path):
    # Two actions is a controller doing a job, not a half-built CRUD resource.
    from tactics.playbooks.report import find_missing

    endpoints = [Endpoint("GET", "/sync", "api/v1/sync", "index"),
                 Endpoint("POST", "/sync", "api/v1/sync", "create")]
    assert [g for g in find_missing(str(tmp_path), endpoints)
            if g.kind == "incomplete resource"] == []


def test_a_column_stored_and_never_returned_is_an_unfinished_feature(tmp_path):
    # The real find on a live app: three streak columns, a service that
    # maintains them and a job that runs it — and no serializer returning any
    # of it, so no client could ever see the feature.
    from tactics.playbooks.report import find_missing

    root = _tree(tmp_path, {
        "db/schema.rb": (
            'create_table "users", force: :cascade do |t|\n'
            '  t.string "email"\n  t.integer "current_streak"\n'
            '  t.datetime "created_at", null: false\n  t.bigint "list_id"\n'
            "end\n"),
        "app/serializers/user_serializer.rb": "class UserSerializer\n  def as_json\n"
                                              "    { email: user.email }\n  end\nend\n",
    })
    gaps = [g for g in find_missing(root, []) if g.kind == "stored but never returned"]
    assert len(gaps) == 1
    assert "current_streak" in gaps[0].detail
    assert "email" not in gaps[0].detail          # it is returned
    assert "created_at" not in gaps[0].detail     # plumbing
    assert "list_id" not in gaps[0].detail        # plumbing


def test_the_frameworks_own_columns_are_not_unfinished_features(tmp_path):
    # Devise's schema buried the one column that mattered under six of its own.
    from tactics.playbooks.report import find_missing

    root = _tree(tmp_path, {
        "db/schema.rb": (
            'create_table "users", force: :cascade do |t|\n'
            '  t.integer "failed_attempts"\n  t.datetime "confirmed_at"\n'
            '  t.string "reset_password_token"\n  t.integer "sign_in_count"\n'
            "end\n"),
        "app/serializers/user_serializer.rb": "class UserSerializer; end\n",
    })
    assert [g for g in find_missing(root, []) if g.kind == "stored but never returned"] == []


def test_the_report_separates_missing_from_removable():
    text = render(Report(root="/r", kind="rails",
                         missing=[Gap("incomplete resource", "no destroy", "things")]))
    assert "## What is not there yet" in text
    assert "the repository contradicting itself" in text
    assert "no destroy" in text
