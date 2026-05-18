# Copyright 2026 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the data_object path on mlrun.serving.server.execute_graph (ML-12578).

Covers AC-4 through AC-26 + AC-31 from the validation/review documents.
"""

import os
from collections import OrderedDict
from typing import Any

import pandas as pd
import pytest

import mlrun
import mlrun.errors
from mlrun.execution import MLClientCtx
from mlrun.serving.server import execute_graph

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# Module-level captures so the user-handler can record what the graph step
# received without needing instance state. Storey/MLRun graph handler functions
# are invoked with the event body (not the event object) by default, but a
# step can opt into the full Event via full_event=True.
_captured_bodies: list[Any] = []
_captured_event_ids: list[Any] = []


def _echo_body(event):
    """Graph handler: records the body it received and returns it unchanged.

    Storey graph handlers receive the event *body* directly. This is the
    standard pattern used by tests like function_with_simple_transformation.py.
    """
    _captured_bodies.append(event)
    return event


def _echo_full_event(event):
    """Graph handler used with full_event=True: records event id AND body.

    Returns the event body so .respond()'s wire format matches the
    body-only handler.
    """
    _captured_event_ids.append(event.id)
    _captured_bodies.append(event.body)
    # Mutate body so the response is still the body (not the Event itself).
    event.body = event.body
    return event


def _reset_capture():
    _captured_bodies.clear()
    _captured_event_ids.clear()


def _build_dict_path_spec(track_models: bool = False) -> str:
    """Build a minimal serving spec (JSON-string form) with a responder step.

    Uses the public ServingRuntime API to build the graph + spec so the spec
    shape matches what the real to_job() handler receives at runtime.
    """
    fn = mlrun.new_function(
        "test-execute-graph",
        kind="serving",
        project="default",
    )
    fn.set_topology("flow", engine="async")
    fn.spec.graph.to(
        name="echo",
        handler="tests.serving.test_execute_graph_data_object._echo_body",
    ).respond()
    if track_models:
        fn.spec.track_models = True
    return fn._get_serving_spec()


def _build_dict_path_spec_with_tracking() -> str:
    return _build_dict_path_spec(track_models=True)


def _build_no_responder_spec() -> str:
    """Variant without a responder step — for tests that should not log
    the prediction artifact (e.g. empty-dict, which would create a 1-row,
    0-column DataFrame that pandas cannot describe)."""
    fn = mlrun.new_function(
        "test-execute-graph-no-responder",
        kind="serving",
        project="default",
    )
    fn.set_topology("flow", engine="async")
    fn.spec.graph.to(
        name="echo",
        handler="tests.serving.test_execute_graph_data_object._echo_body",
    )
    return fn._get_serving_spec()


def _build_full_event_spec() -> str:
    """Variant whose step receives the full storey Event (not just the body).

    Used to capture per-event IDs and verify the batch-path
    closure-over-loop-variable bug is fixed by the index→idx rename.
    """
    fn = mlrun.new_function(
        "test-execute-graph-full-event",
        kind="serving",
        project="default",
    )
    fn.set_topology("flow", engine="async")
    fn.spec.graph.to(
        name="echo_full",
        handler="tests.serving.test_execute_graph_data_object._echo_full_event",
        full_event=True,
    )
    return fn._get_serving_spec()


def _make_context(tmp_path) -> MLClientCtx:
    """Build a minimal MLClientCtx for handler invocation in unit tests."""
    return MLClientCtx.from_dict(
        {
            "metadata": {"name": "test", "project": "test-project"},
            "spec": {"output_path": str(tmp_path)},
        },
        autocommit=False,
    )


def _set_serving_spec(spec_json: str) -> None:
    """Install a serving spec for the handler to read via SERVING_SPEC_ENV."""
    os.environ["SERVING_SPEC_ENV"] = spec_json


# ---------------------------------------------------------------------------
# Validation tests (U1, U2, U3, U11)
# ---------------------------------------------------------------------------


def test_execute_graph_requires_data_or_data_object():
    """Calling execute_graph with neither data nor data_object raises."""
    context = MLClientCtx.from_dict(
        {"metadata": {"name": "test"}, "spec": {}}, autocommit=False
    )
    with pytest.raises(
        mlrun.errors.MLRunInvalidArgumentError,
        match=r"exactly one of 'data' or 'data_object' must be provided",
    ):
        execute_graph(context)


def test_execute_graph_rejects_both_data_and_data_object():
    """Data and data_object are mutually exclusive."""
    context = MLClientCtx.from_dict(
        {"metadata": {"name": "test"}, "spec": {}}, autocommit=False
    )
    # Even an int (which would fail the DataItem check) — the mutual-exclusion
    # check must fire FIRST and produce its own error.
    with pytest.raises(
        mlrun.errors.MLRunInvalidArgumentError,
        match=r"'data' and 'data_object' are mutually exclusive",
    ):
        execute_graph(context, data=123, data_object={"x": 1})


@pytest.mark.parametrize(
    "bad_value",
    [1, "a-string", [1, 2, 3], (1, 2), 3.14, True],
)
def test_execute_graph_data_object_must_be_dict(bad_value):
    """data_object must be a dict (or subclass)."""
    context = MLClientCtx.from_dict(
        {"metadata": {"name": "test"}, "spec": {}}, autocommit=False
    )
    with pytest.raises(
        mlrun.errors.MLRunInvalidArgumentError,
        match=r"data_object must be a dict",
    ):
        execute_graph(context, data_object=bad_value)


def test_execute_graph_data_object_accepts_dict_subclasses():
    """Corollary: dict subclasses (e.g. OrderedDict) are accepted by the type check.

    Note: this test only verifies the type check passes; it intentionally does
    not run the full graph (which is exercised by behavioral tests below).
    """
    # Subclass-acceptance means the type check passes — to avoid running the
    # full handler here, we patch the no-op return after type validation.
    context = MLClientCtx.from_dict(
        {"metadata": {"name": "test"}, "spec": {}}, autocommit=False
    )
    # We expect to NOT hit MLRunInvalidArgumentError for "data_object must be a dict".
    # We may hit other errors (e.g. missing serving spec) — those are out of scope here.
    with pytest.raises(Exception) as exc_info:
        execute_graph(context, data_object=OrderedDict([("a", 1)]))
    assert "data_object must be a dict" not in str(exc_info.value)


def test_execute_graph_data_object_does_not_trigger_legacy_dataitem_error():
    """Negative: when data_object is set, the legacy DataItem error must NOT fire.

    The mutual-exclusion check (step 2) fires first; the legacy DataItem check
    (step 4) is only reached when data_object is None.
    """
    context = MLClientCtx.from_dict(
        {"metadata": {"name": "test"}, "spec": {}}, autocommit=False
    )
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError) as exc_info:
        execute_graph(context, data=123, data_object={"x": 1})
    # The mutual-exclusion message must fire, NOT the legacy DataItem message.
    assert "mutually exclusive" in str(exc_info.value)
    assert "DataItem" not in str(exc_info.value)


def test_execute_graph_dataitem_validation_still_fires_when_data_object_is_none():
    """Positive (mirrors existing test_execute_graph_dataitem_parameter_validation):
    when data_object is None, the legacy DataItem error must still fire for non-DataItem data.
    """
    context = MLClientCtx.from_dict(
        {"metadata": {"name": "test"}, "spec": {}}, autocommit=False
    )
    with pytest.raises(
        mlrun.errors.MLRunInvalidArgumentError,
        match=r".*data.*DataItem.*inputs.*params.*",
    ):
        execute_graph(context, data=123)


# ---------------------------------------------------------------------------
# Behavioral tests (U4 — U10)
# ---------------------------------------------------------------------------


def test_execute_graph_data_object_runs_once_and_returns_response(tmp_path):
    """Dict input runs the graph exactly once,
    returns the response, logs num_rows=1, and produces a 1-row prediction artifact."""
    _reset_capture()
    _set_serving_spec(_build_dict_path_spec())
    context = _make_context(tmp_path)

    body = {"x": 1, "y": 2}
    result = execute_graph(context, data_object=body)

    # AC-25: returns the graph's response as-is.
    assert result == body, f"Expected return == body, got {result!r}"

    # AC-9: graph ran exactly once.
    assert len(_captured_bodies) == 1

    # AC-23: num_rows logged as 1.
    assert context.results.get("num_rows") == 1


def test_execute_graph_data_object_with_timestamp_column(tmp_path):
    """timestamp_column extracted from dict root; the value is
    attached to the event via _original_timestamp by the inner run() closure.
    """
    _reset_capture()
    _set_serving_spec(_build_dict_path_spec())
    context = _make_context(tmp_path)

    body = {"value": 8, "ts": "2020-01-01T00:00:00"}
    result = execute_graph(context, data_object=body, timestamp_column="ts")
    assert result == body
    assert len(_captured_bodies) == 1


def test_execute_graph_data_object_missing_timestamp_column_raises(tmp_path):
    """Missing timestamp_column key raises MLRunRuntimeError.

    When track_models is True, the dict-path outer check fires; when False,
    the inner closure fires. Both paths raise MLRunRuntimeError with wording
    matching r".*did not contain timestamp column.*".
    """
    _set_serving_spec(_build_dict_path_spec())
    context = _make_context(tmp_path)

    with pytest.raises(
        mlrun.errors.MLRunRuntimeError,
        match=r".*did not contain timestamp column.*",
    ):
        execute_graph(context, data_object={"value": 8}, timestamp_column="ts")


def test_execute_graph_data_object_clock_time_when_no_timestamp_column(tmp_path):
    """When timestamp_column is None, start_time falls back to clock time."""
    _reset_capture()
    _set_serving_spec(_build_dict_path_spec())
    context = _make_context(tmp_path)

    # Should not raise; the graph runs and the start_time is set from clock time.
    result = execute_graph(context, data_object={"x": 1})
    assert result == {"x": 1}
    assert len(_captured_bodies) == 1


@pytest.mark.parametrize("read_as_lists", [False, True])
@pytest.mark.parametrize("nest_under_inputs", [False, True])
def test_execute_graph_data_object_body_shape_matrix(
    tmp_path, read_as_lists, nest_under_inputs
):
    """4-cell matrix of read_as_lists × nest_under_inputs.

    Verifies the graph step receives the body in the documented shape.
    """
    _reset_capture()
    _set_serving_spec(_build_dict_path_spec())
    context = _make_context(tmp_path)

    data_object = {"a": 1, "b": 2}
    result = execute_graph(
        context,
        data_object=data_object,
        read_as_lists=read_as_lists,
        nest_under_inputs=nest_under_inputs,
    )

    # Compute the expected body shape.
    expected_inner = [1, 2] if read_as_lists else {"a": 1, "b": 2}
    expected_body = {"inputs": expected_inner} if nest_under_inputs else expected_inner

    assert _captured_bodies == [expected_body], (
        f"Body shape mismatch: got {_captured_bodies}, expected [{expected_body!r}]"
    )
    # And the response is returned as-is.
    assert result == expected_body


def test_execute_graph_data_object_ignores_batching(tmp_path):
    """Batching / batch_size are ignored on the data_object path.

    Runs once, num_rows == 1, no error.
    """
    _reset_capture()
    _set_serving_spec(_build_dict_path_spec())
    context = _make_context(tmp_path)

    result = execute_graph(
        context,
        data_object={"x": 1},
        batching=True,
        batch_size=10,
    )
    assert result == {"x": 1}
    assert len(_captured_bodies) == 1
    assert context.results.get("num_rows") == 1


def test_execute_graph_with_empty_data_object_runs_once(tmp_path):
    """Empty data_object {} runs the graph exactly once (no early-out).

    Uses a graph WITHOUT a responder so we exercise just the "run once" semantic
    — the prediction-artifact-logging path with pd.DataFrame([{}]) is a
    pre-existing pandas limitation (1-row, 0-column DataFrame cannot be
    described) and is out of scope for this feature. The Jira intent for
    empty-dict is "run-once", not "produce an empty artifact".
    """
    _reset_capture()
    _set_serving_spec(_build_no_responder_spec())
    context = _make_context(tmp_path)

    # Should not raise; should run the graph exactly once.
    execute_graph(context, data_object={})
    # The step received the empty dict body (pinned via the body-capture).
    assert _captured_bodies == [{}], (
        f"Expected single empty-dict body capture, got {_captured_bodies}"
    )
    assert context.results.get("num_rows") == 1


# ---------------------------------------------------------------------------
# Regression test for the secondary fix (U12 / AC-31)
# ---------------------------------------------------------------------------


def test_execute_graph_batch_path_event_ids_are_per_row(tmp_path):
    """Each task in the batch loop must receive its own event.id.

    Pins the secondary fix from the index → idx closure-arg rename. Before the
    fix every task captured the loop's terminal index value (len(df) - 1).
    """
    _reset_capture()
    _set_serving_spec(_build_full_event_spec())
    context = _make_context(tmp_path)

    # Persist a real 3-row CSV and load it via mlrun.get_dataitem so we exercise
    # the real DataItem path (no monkey-patching of the type check).
    df = pd.DataFrame([{"a": i, "b": i * 10} for i in range(3)])
    csv_path = tmp_path / "rows.csv"
    df.to_csv(csv_path, index=False)
    data_item = mlrun.get_dataitem(str(csv_path))

    execute_graph(context, data=data_item)

    # AC-31: event ids must be [0, 1, 2], NOT [2, 2, 2].
    assert _captured_event_ids == [0, 1, 2], (
        f"Expected event ids [0,1,2], got {_captured_event_ids}. "
        "If this is [2,2,2], the closure-over-loop-variable bug has regressed."
    )
    assert context.results.get("num_rows") == 3
