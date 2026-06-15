# Copyright 2023 Iguazio
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

import pydantic.v1
import pytest

import mlrun.common.schemas.common
import mlrun.common.schemas.project
from mlrun.common.schemas.model_monitoring.constants import StreamTarget, TSDBTarget


@pytest.mark.parametrize(
    "labels,expected",
    [
        (None, []),
        ({}, []),
        ([], []),
        (["label1", "label2"], ["label1", "label2"]),
        (["label1"], ["label1"]),
        ({"label1": "value1", "label2": "value2"}, ["label1=value1", "label2=value2"]),
        ({"label1": 1}, ["label1=1"]),
        (["label1=value1"], ["label1=value1"]),
        ({"label1": "value1", "label2": None}, ["label1=value1", "label2"]),
        ("label1=value1,label2", ["label1=value1", "label2"]),
    ],
)
def test_labels_validation(labels, expected):
    labels_result = mlrun.common.schemas.common.LabelsModel(labels=labels).labels
    assert labels_result == expected


# --- ProjectMonitoringSpec (ML-12543) ---------------------------------------


def test_project_monitoring_spec_defaults():
    """Bare ProjectMonitoringSpec has both flags False and no type set."""
    spec = mlrun.common.schemas.project.ProjectMonitoringSpec()
    assert spec.enabled is False
    assert spec.otlp_enabled is False
    assert spec.stream_type is None
    assert spec.tsdb_type is None


def test_project_monitoring_spec_round_trip():
    """A populated ProjectMonitoringSpec round-trips through dict / parse_obj."""
    original = mlrun.common.schemas.project.ProjectMonitoringSpec(
        enabled=True,
        otlp_enabled=True,
        stream_type=StreamTarget.KAFKA,
        tsdb_type=TSDBTarget.TimescaleDB,
    )
    parsed = mlrun.common.schemas.project.ProjectMonitoringSpec.parse_obj(
        original.dict()
    )
    assert parsed.enabled is True
    assert parsed.otlp_enabled is True
    assert parsed.stream_type is StreamTarget.KAFKA
    assert parsed.tsdb_type is TSDBTarget.TimescaleDB


@pytest.mark.parametrize(
    "field, value",
    [("stream_type", "not-a-real-stream"), ("tsdb_type", "not-a-real-tsdb")],
)
def test_project_monitoring_spec_rejects_invalid_enum(field, value):
    """Invalid stream/tsdb type strings must fail validation."""
    with pytest.raises(pydantic.v1.ValidationError):
        mlrun.common.schemas.project.ProjectMonitoringSpec(**{field: value})


def test_project_spec_model_monitoring_default_is_populated():
    """ProjectSpec.model_monitoring always returns a default-populated struct so
    callers never have to None-guard. Reading individual flags before
    enable_model_monitoring/set_credentials yields the all-default values.
    """
    project_spec = mlrun.common.schemas.project.ProjectSpec()
    assert project_spec.model_monitoring is not None
    assert project_spec.model_monitoring.enabled is False
    assert project_spec.model_monitoring.otlp_enabled is False
    assert project_spec.model_monitoring.stream_type is None
    assert project_spec.model_monitoring.tsdb_type is None


def test_project_spec_nests_model_monitoring():
    """Setting model_monitoring on ProjectSpec round-trips with the nested fields intact."""
    project_spec = mlrun.common.schemas.project.ProjectSpec(
        model_monitoring=mlrun.common.schemas.project.ProjectMonitoringSpec(
            enabled=True, otlp_enabled=True
        )
    )
    reparsed = mlrun.common.schemas.project.ProjectSpec.parse_obj(project_spec.dict())
    assert reparsed.model_monitoring is not None
    assert reparsed.model_monitoring.enabled is True
    assert reparsed.model_monitoring.otlp_enabled is True


def test_project_spec_out_model_monitoring_default_is_populated():
    """ProjectSpecOut mirrors ProjectSpec's default-populated model_monitoring
    so the GET-project response always carries a usable struct. Parsing a
    legacy payload without a `model_monitoring` key must fill it in.
    """
    spec_out = mlrun.common.schemas.project.ProjectSpecOut()
    assert spec_out.model_monitoring is not None
    assert spec_out.model_monitoring.enabled is False
    assert spec_out.model_monitoring.otlp_enabled is False

    # Legacy payloads (pre-ML-12543) have no model_monitoring section — they
    # must deserialize into a default-populated struct rather than None.
    legacy = mlrun.common.schemas.project.ProjectSpecOut.parse_obj({"params": {}})
    assert legacy.model_monitoring is not None
    assert legacy.model_monitoring.enabled is False
    assert legacy.model_monitoring.otlp_enabled is False
    assert legacy.model_monitoring.stream_type is None
    assert legacy.model_monitoring.tsdb_type is None
