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
"""L2 behavioural parity between the ``_v1`` (pydantic.v1) and ``_v2`` (native pydantic 2)
schema faces (ML-12900).

L1 (``test_schemas_parity.py``, ML-12891) pins the declared *shape* of every model — same
fields, same required-ness, same defaults. It cannot express *behaviour*: how a value is
coerced, serialised, or rejected. L2 closes that gap with a fixed, reviewable corpus of
inputs, each carrying an explicit expected verdict — ``accept`` or ``reject``.

The verdict is never inferred from "do the two faces agree" (a both-agree assertion would
silently pass if an ``accept`` case decayed into a mutual ``reject`` as schemas evolve,
quietly shrinking the corpus to rejection-only). Each case is asserted against its pinned
verdict independently.

* ``accept`` — both faces must accept the input, and round-trip identically: the JSON each
  produces, parsed back to a plain value, must be equal, AND each face must be able to
  parse the *other* face's JSON output (the actual wire scenario — a ``_v1`` client and a
  ``_v2`` server exchanging requests/responses, per Backend HLD §2.3).
* ``reject`` — both faces must reject, and the set of error loci (offending field paths)
  must match. Only loci are compared, never messages or ``type`` codes — those differ
  between Pydantic majors by construction (Backend HLD §3.1).

Like L1, this runs on the Pydantic-2-only baseline and imports ``_v1``/``_v2`` directly,
independent of the ``MLRUN_IS_API_SERVER`` dispatch gate.
"""

import json

import pydantic as v2
import pydantic.v1 as v1
import pytest

if int(v2.VERSION.split(".")[0]) < 2:
    pytest.skip(
        "L2 behavioural parity requires the pydantic 2 baseline",
        allow_module_level=True,
    )

import mlrun.common.schemas._v1.alert as v1_alert
import mlrun.common.schemas._v1.client_spec as v1_client_spec
import mlrun.common.schemas._v1.common as v1_common
import mlrun.common.schemas._v1.model_monitoring.model_endpoints as v1_mep
import mlrun.common.schemas._v1.schedule as v1_schedule
import mlrun.common.schemas._v2.alert as v2_alert
import mlrun.common.schemas._v2.client_spec as v2_client_spec
import mlrun.common.schemas._v2.common as v2_common
import mlrun.common.schemas._v2.model_monitoring.model_endpoints as v2_mep
import mlrun.common.schemas._v2.schedule as v2_schedule

# (case_id, v1 model, v2 model, input payload). Grounded in real hard-conversion sites
# found in the merged _v1/_v2 schemas, per Backend HLD §3.1's seed list (implicit-optional
# fields, const->Literal, engine-sensitive regex, json_encoders->serializer, coercion).
ACCEPT_CASES = [
    (
        "schedule_cron_trigger_all_optional_omitted",
        v1_schedule.ScheduleCronTrigger,
        v2_schedule.ScheduleCronTrigger,
        {},
    ),
    (
        "schedule_cron_trigger_iso_string_start_date",
        v1_schedule.ScheduleCronTrigger,
        v2_schedule.ScheduleCronTrigger,
        {"start_date": "2026-01-01T00:00:00"},
    ),
    (
        # regression guard for the historical bare-`Any` v1 bug (ML-12736 POC): a v1
        # client omitting `scheduled_object` must keep working under `_v2`.
        "schedule_input_missing_scheduled_object",
        v1_schedule.ScheduleInput,
        v2_schedule.ScheduleInput,
        {"name": "s1", "kind": "job", "cron_trigger": "0 0 * * *"},
    ),
    (
        "schedule_identifier_correct_kind_literal",
        v1_schedule.ScheduleIdentifier,
        v2_schedule.ScheduleIdentifier,
        {"kind": "schedule", "name": "s1"},
    ),
    (
        # v1 implicitly coerces non-str scalars into str-typed fields; _v2's before-
        # validator on ClientSpec must reproduce that exactly (else existing v1 clients
        # that send e.g. a bool for a str-typed setting would get a different value back).
        "client_spec_bool_coerced_to_str",
        v1_client_spec.ClientSpec,
        v2_client_spec.ClientSpec,
        {"scrape_metrics": True},
    ),
    (
        "model_endpoint_metadata_valid_project_pattern",
        v1_mep.ModelEndpointMetadata,
        v2_mep.ModelEndpointMetadata,
        {"name": "ep1", "project": "my-proj-1"},
    ),
    (
        # PROJECT_PATTERN/MODEL_ENDPOINT_ID_PATTERN use a bare `$` anchor. Python's `re`
        # module treats `$` as matching just before a trailing newline, not only at the
        # true end of string; Pydantic 2's default (Rust) regex engine does not. `_v2`
        # sets `regex_engine="python-re"` specifically to keep this quirk wire-identical
        # to `_v1` (which always uses Python's `re`) -- this case guards that choice.
        "model_endpoint_instruction_name_trailing_newline",
        v1_mep.ModelEndpointInstruction,
        v2_mep.ModelEndpointInstruction,
        {"name": "myname\n"},
    ),
    (
        "alert_event_entities_single_id",
        v1_alert.EventEntities,
        v2_alert.EventEntities,
        {"kind": "job", "project": "p1", "ids": ["x"]},
    ),
    (
        # v1's Union[str, dict[str, str|None], list[str]] coerces a non-str dict value
        # (int -> str) before its custom validator runs; v2's default "after"-mode
        # validator type-checks against the Union first and rejected this, until fixed
        # to mode="before" (this case is the regression guard for that fix).
        "labels_model_dict_with_int_value",
        v1_common.LabelsModel,
        v2_common.LabelsModel,
        {"labels": {"label1": 1}},
    ),
]

REJECT_CASES = [
    (
        "schedule_identifier_wrong_kind_literal",
        v1_schedule.ScheduleIdentifier,
        v2_schedule.ScheduleIdentifier,
        {"kind": "not-a-schedule", "name": "s1"},
        {("kind",)},
    ),
    (
        "model_endpoint_metadata_invalid_project_pattern",
        v1_mep.ModelEndpointMetadata,
        v2_mep.ModelEndpointMetadata,
        {"name": "ep1", "project": "Bad_Project!"},
        {("project",)},
    ),
    (
        "model_endpoint_instruction_name_with_space",
        v1_mep.ModelEndpointInstruction,
        v2_mep.ModelEndpointInstruction,
        {"name": "bad name"},
        {("name",)},
    ),
    (
        # conlist(min_items/max_items=1) (_v1) vs conlist(min_length/max_length=1) (_v2):
        # an empty list must violate the lower bound on both.
        "alert_event_entities_empty_ids",
        v1_alert.EventEntities,
        v2_alert.EventEntities,
        {"kind": "job", "project": "p1", "ids": []},
        {("ids",)},
    ),
    (
        "alert_event_entities_too_many_ids",
        v1_alert.EventEntities,
        v2_alert.EventEntities,
        {"kind": "job", "project": "p1", "ids": ["a", "b"]},
        {("ids",)},
    ),
]


def _error_loci(exc) -> set:
    return {tuple(error["loc"]) for error in exc.errors()}


@pytest.mark.parametrize(
    "case_id,v1_model,v2_model,payload",
    ACCEPT_CASES,
    ids=[case[0] for case in ACCEPT_CASES],
)
def test_accept_case_round_trips_both_directions(case_id, v1_model, v2_model, payload):
    try:
        v1_obj = v1_model.parse_obj(payload)
    except v1.ValidationError as exc:
        pytest.fail(f"{case_id}: _v1 rejected an accept-verdict input: {exc}")

    try:
        v2_obj = v2_model.model_validate(payload)
    except v2.ValidationError as exc:
        pytest.fail(f"{case_id}: _v2 rejected an accept-verdict input: {exc}")

    v1_json = v1_obj.json()
    v2_json = v2_obj.model_dump_json()
    # Compared as parsed JSON, not raw strings: `.json()` (v1) and `.model_dump_json()`
    # (v2) differ in incidental whitespace by default (v1 inserts separators, v2's Rust
    # serializer is compact) -- neither reflects an actual wire encoding (FastAPI's
    # JSONResponse and the client's HTTP layer both re-encode compactly regardless), so
    # whitespace is not a wire-parity signal. Content is. Known blind spot: Python's `==`
    # treats `True`/`1`/`1.0` as equal, so a face divergence that only swaps JSON `true`
    # for `1` would slip through here -- none of the current corpus exercises that shape.
    assert json.loads(v1_json) == json.loads(v2_json), (
        f"{case_id}: dumps differ\n v1={v1_json}\n v2={v2_json}"
    )

    # both directions: each face must accept the JSON the *other* face produced -- this is
    # the actual wire scenario (v1 client <-> v2 server, Backend HLD §2.3).
    try:
        v2_model.model_validate_json(v1_json)
    except v2.ValidationError as exc:
        pytest.fail(f"{case_id}: _v2 rejected _v1's JSON output: {exc}")
    try:
        v1_model.parse_raw(v2_json)
    except v1.ValidationError as exc:
        pytest.fail(f"{case_id}: _v1 rejected _v2's JSON output: {exc}")


@pytest.mark.parametrize(
    "case_id,v1_model,v2_model,payload,expected_loci",
    REJECT_CASES,
    ids=[case[0] for case in REJECT_CASES],
)
def test_reject_case_matches_error_loci(
    case_id, v1_model, v2_model, payload, expected_loci
):
    try:
        v1_model.parse_obj(payload)
        pytest.fail(f"{case_id}: _v1 accepted a reject-verdict input")
    except v1.ValidationError as exc:
        v1_loci = _error_loci(exc)

    try:
        v2_model.model_validate(payload)
        pytest.fail(f"{case_id}: _v2 accepted a reject-verdict input")
    except v2.ValidationError as exc:
        v2_loci = _error_loci(exc)

    assert v1_loci == expected_loci, (
        f"{case_id}: _v1 error loci {v1_loci} != expected {expected_loci}"
    )
    assert v2_loci == expected_loci, (
        f"{case_id}: _v2 error loci {v2_loci} != expected {expected_loci}"
    )


def test_schedule_record_callable_scheduled_object_serializes_to_empty_object():
    # The one case in this file that isn't an ACCEPT_CASES/REJECT_CASES entry: it has no
    # _v1 side to compare against (see below), so it can't fit the accept/reject corpus
    # shape and is asserted directly instead.
    # _v2 only, not a _v1-vs-_v2 comparison: a `local_function` schedule holds a live
    # Python callable server-side. The pre-migration stack rendered it as `{}` via
    # *FastAPI's* `jsonable_encoder` (its fallback for an unencodable object), never via
    # pydantic v1's own `.json()` -- calling that directly on a raw function raises
    # TypeError (confirmed: pydantic v1 has no encoder for a bare function), so there is
    # no genuine v1 behaviour to compare against here. `_v2`'s `field_serializer` on
    # `scheduled_object` exists specifically to reproduce the old FastAPI-layer behaviour
    # now that the server dumps via `.model_dump_json()` directly; this guards that it
    # still does.
    def _scheduled_object():
        pass

    v2_obj = v2_schedule.ScheduleRecord(
        name="s1",
        kind="local_function",
        scheduled_object=_scheduled_object,
        cron_trigger="0 0 * * *",
        creation_time="2026-01-01T00:00:00",
        project="p1",
    )

    assert json.loads(v2_obj.model_dump_json())["scheduled_object"] == {}
