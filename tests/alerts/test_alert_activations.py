# Copyright 2024 Iguazio
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

import datetime
from datetime import UTC

import pytest

import mlrun.common.schemas
import mlrun.common.schemas.project


@pytest.fixture
def sample_alert_activations():
    return mlrun.common.schemas.AlertActivations(
        activations=[
            mlrun.common.schemas.AlertActivation(
                id=1,
                name="alert1",
                project="project1",
                severity=mlrun.common.schemas.alert.AlertSeverity.LOW,
                activation_time=datetime.datetime.now(UTC),
                entity_id="123456",
                entity_kind=mlrun.common.schemas.alert.EventEntityKind.MODEL_ENDPOINT_RESULT,
                event_kind=mlrun.common.schemas.alert.EventKind.DATA_DRIFT_SUSPECTED,
                number_of_events=1,
                notifications=[],
                criteria=mlrun.common.schemas.alert.AlertCriteria(count=1),
            ),
            mlrun.common.schemas.AlertActivation(
                id=2,
                name="alert1",
                project="project2",
                severity=mlrun.common.schemas.alert.AlertSeverity.LOW,
                activation_time=datetime.datetime.now(UTC),
                entity_id="123456",
                entity_kind=mlrun.common.schemas.alert.EventEntityKind.MODEL_ENDPOINT_RESULT,
                event_kind=mlrun.common.schemas.alert.EventKind.DATA_DRIFT_DETECTED,
                number_of_events=2,
                notifications=[],
                criteria=mlrun.common.schemas.alert.AlertCriteria(count=2),
            ),
            mlrun.common.schemas.AlertActivation(
                id=3,
                name="alert2",
                project="project3",
                severity=mlrun.common.schemas.alert.AlertSeverity.HIGH,
                activation_time=datetime.datetime.now(UTC),
                entity_id="1234",
                entity_kind=mlrun.common.schemas.alert.EventEntityKind.JOB,
                event_kind=mlrun.common.schemas.alert.EventKind.FAILED,
                number_of_events=1,
                notifications=[],
                criteria=mlrun.common.schemas.alert.AlertCriteria(count=1),
            ),
            mlrun.common.schemas.AlertActivation(
                id=4,
                name="alert3",
                project="project3",
                severity=mlrun.common.schemas.alert.AlertSeverity.MEDIUM,
                activation_time=datetime.datetime.now(UTC),
                entity_id="1234",
                entity_kind=mlrun.common.schemas.alert.EventEntityKind.JOB,
                event_kind=mlrun.common.schemas.alert.EventKind.FAILED,
                number_of_events=1,
                notifications=[],
                criteria=mlrun.common.schemas.alert.AlertCriteria(count=1),
            ),
        ]
    )


def test_group_by_severity(sample_alert_activations):
    grouped = sample_alert_activations.group_by("severity")
    assert len(grouped) == 3
    assert len(grouped[mlrun.common.schemas.alert.AlertSeverity.LOW]) == 2
    assert len(grouped[mlrun.common.schemas.alert.AlertSeverity.HIGH]) == 1
    assert len(grouped[mlrun.common.schemas.alert.AlertSeverity.MEDIUM]) == 1


def test_group_by_event_and_entity_kind(sample_alert_activations):
    grouped = sample_alert_activations.group_by("event_kind", "entity_kind")
    assert len(grouped) == 3
    assert (
        len(
            grouped[
                (
                    mlrun.common.schemas.alert.EventKind.DATA_DRIFT_DETECTED,
                    mlrun.common.schemas.alert.EventEntityKind.MODEL_ENDPOINT_RESULT,
                )
            ]
        )
        == 1
    )
    assert (
        len(
            grouped[
                (
                    mlrun.common.schemas.alert.EventKind.DATA_DRIFT_SUSPECTED,
                    mlrun.common.schemas.alert.EventEntityKind.MODEL_ENDPOINT_RESULT,
                )
            ]
        )
        == 1
    )
    assert (
        len(
            grouped[
                (
                    mlrun.common.schemas.alert.EventKind.FAILED,
                    mlrun.common.schemas.alert.EventEntityKind.JOB,
                )
            ]
        )
        == 2
    )


def test_aggregate_by_severity(sample_alert_activations):
    aggregated = sample_alert_activations.aggregate_by(
        ["severity"],
        lambda activations: sum(
            activation.number_of_events for activation in activations
        ),
    )
    assert aggregated == {
        (mlrun.common.schemas.alert.AlertSeverity.HIGH): 1,
        (mlrun.common.schemas.alert.AlertSeverity.LOW): 3,
        (mlrun.common.schemas.alert.AlertSeverity.MEDIUM): 1,
    }


def test_all_entity_kinds_have_project_summary_counter():
    """Ensure every EventEntityKind has a corresponding *_alerts_count field in ProjectSummary.

    If a new EventEntityKind is added without a matching counter, this test will fail,
    reminding the developer to update _calculate_alert_activations_counters and ProjectSummary.
    """
    entity_kind_to_counter_field = {
        mlrun.common.schemas.alert.EventEntityKind.MODEL_ENDPOINT_RESULT: "endpoint_alerts_count",
        mlrun.common.schemas.alert.EventEntityKind.JOB: "job_alerts_count",
        mlrun.common.schemas.alert.EventEntityKind.MODEL_MONITORING_APPLICATION: "application_alerts_count",
        mlrun.common.schemas.alert.EventEntityKind.MODEL_MONITORING_INFRA: "infra_alerts_count",
    }
    all_entity_kinds = set(mlrun.common.schemas.alert.EventEntityKind)
    mapped_entity_kinds = set(entity_kind_to_counter_field.keys())
    assert all_entity_kinds == mapped_entity_kinds, (
        f"EventEntityKind values not mapped to ProjectSummary counter fields: "
        f"{all_entity_kinds - mapped_entity_kinds}. "
        f"Update _calculate_alert_activations_counters in db.py and ProjectSummary schema."
    )
    summary_fields = set(mlrun.common.schemas.project.ProjectSummary.__fields__.keys())
    for entity_kind, counter_field in entity_kind_to_counter_field.items():
        assert counter_field in summary_fields, (
            f"Counter field '{counter_field}' for {entity_kind} missing from ProjectSummary"
        )


def test_aggregate_by_event_and_entity_kind(sample_alert_activations):
    aggregated = sample_alert_activations.aggregate_by(
        ["name", "entity_id"], lambda activations: len(activations)
    )
    assert aggregated == {
        ("alert1", "123456"): 2,
        ("alert2", "1234"): 1,
        ("alert3", "1234"): 1,
    }
    aggregated = sample_alert_activations.aggregate_by(
        ["name", "entity_id", "project"], lambda activations: len(activations)
    )
    assert aggregated == {
        ("alert1", "123456", "project1"): 1,
        ("alert1", "123456", "project2"): 1,
        ("alert2", "1234", "project3"): 1,
        ("alert3", "1234", "project3"): 1,
    }
