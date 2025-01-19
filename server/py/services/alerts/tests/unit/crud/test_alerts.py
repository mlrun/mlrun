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
#
import unittest
import unittest.mock
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from datetime import timezone

import fastapi.concurrency
import pytest
import sqlalchemy.orm

import mlrun.common.schemas.alert
import mlrun.common.schemas.alert as alert_objects

import framework.utils.singletons.db
import services.alerts.crud
import services.alerts.tests.unit.crud.utils
from framework.tests.unit.common_fixtures import K8sSecretsMock
from services.alerts.tests.unit.conftest import TestAlertsBase


@pytest.fixture
def reset_alert_caches():
    yield
    services.alerts.crud.Alerts()._alert_cache.cache_clear()
    services.alerts.crud.Alerts()._alert_state_cache.cache_clear()


class TestAlerts(TestAlertsBase):
    @pytest.mark.asyncio
    @unittest.mock.patch.object(
        framework.utils.singletons.db.SQLDB,
        "update_alert_activation",
        return_value=None,
    )
    @unittest.mock.patch.object(
        services.alerts.crud.AlertActivation,
        "store_alert_activation",
        return_value=None,
    )
    async def test_process_event_no_cache(
        self,
        mocked_update_alert_activation,
        mocked_store_alert_activation,
        db: sqlalchemy.orm.Session,
        k8s_secrets_mock: K8sSecretsMock,
    ):
        project = "project-name"
        alert_name = "my-alert"
        alert_summary = "testing 1 2 3"
        alert_reset_policy = alert_objects.ResetPolicy.MANUAL
        alert_entity = alert_objects.EventEntities(
            kind=alert_objects.EventEntityKind.MODEL_ENDPOINT_RESULT,
            project=project,
            ids=[123],
        )
        event_kind = alert_objects.EventKind.DATA_DRIFT_SUSPECTED

        alert_data = services.alerts.tests.unit.crud.utils.generate_alert_data(
            project=project,
            name=alert_name,
            entity=alert_entity,
            summary=alert_summary,
            event_kind=event_kind,
            reset_policy=alert_reset_policy,
        )

        services.alerts.crud.Alerts().store_alert(
            session=db,
            project=project,
            name=alert_name,
            alert_data=alert_data,
        )

        event = alert_objects.Event(kind=event_kind, entity=alert_entity)

        await fastapi.concurrency.run_in_threadpool(
            services.alerts.crud.Alerts().process_event_no_cache,
            db,
            event.kind,
            event,
        )

        alert = services.alerts.crud.Alerts().get_alert(
            session=db,
            project=project,
            name=alert_name,
        )
        assert alert.state == alert_objects.AlertActiveState.ACTIVE

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "alert_name, expectation",
        [
            ("invalid_name?", pytest.raises(mlrun.errors.MLRunBadRequestError)),
            ("name with spaces", pytest.raises(mlrun.errors.MLRunBadRequestError)),
            ("invalid/name", pytest.raises(mlrun.errors.MLRunBadRequestError)),
            ("invalid@name", pytest.raises(mlrun.errors.MLRunBadRequestError)),
            ("invalid_name", pytest.raises(mlrun.errors.MLRunBadRequestError)),
            ("$indalid_name", pytest.raises(mlrun.errors.MLRunBadRequestError)),
            ("valid-name", does_not_raise()),
            ("valid-name-123", does_not_raise()),
        ],
    )
    @unittest.mock.patch.object(
        framework.utils.singletons.db.SQLDB,
        "update_alert_activation",
        return_value=None,
    )
    @unittest.mock.patch.object(
        services.alerts.crud.AlertActivation,
        "store_alert_activation",
        return_value=None,
    )
    async def test_validate_alert_name(
        self,
        mocked_update_alert_activation,
        mocked_store_alert_activation,
        db: sqlalchemy.orm.Session,
        k8s_secrets_mock: K8sSecretsMock,
        alert_name: str,
        expectation: AbstractContextManager,
    ):
        project = "project-name"
        alert_summary = "The job has failed"
        alert_entity = alert_objects.EventEntities(
            kind=alert_objects.EventEntityKind.JOB,
            project=project,
            ids=[123],
        )
        event_kind = alert_objects.EventKind.FAILED

        alert_data = services.alerts.tests.unit.crud.utils.generate_alert_data(
            project=project,
            name=alert_name,
            entity=alert_entity,
            summary=alert_summary,
            event_kind=event_kind,
        )
        with expectation:
            services.alerts.crud.Alerts().store_alert(
                session=db,
                project=project,
                name=alert_name,
                alert_data=alert_data,
            )

    @pytest.mark.parametrize(
        "modify_field, modified_value, should_reset",
        [
            # Non-functional fields:
            ("summary", "The job has failed again", False),
            ("description", "Job failure detected", False),
            ("severity", alert_objects.AlertSeverity.HIGH, False),
            (
                "notifications",
                [
                    alert_objects.AlertNotification(
                        notification=mlrun.common.schemas.Notification(
                            kind="webhook",
                            name="webhook_notification",
                            params={
                                "url": "some-webhook-url",
                            },
                        )
                    )
                ],
                False,
            ),
            ("reset_policy", alert_objects.ResetPolicy.AUTO, True),
            # Functional fields:
            (
                "entities",
                alert_objects.EventEntities(
                    kind=alert_objects.EventEntityKind.JOB,
                    project="project-name",
                    ids=[456],
                ),
                True,
            ),
            (
                "entities",
                alert_objects.EventEntities(
                    kind=alert_objects.EventEntityKind.MODEL_ENDPOINT_RESULT,
                    project="project-name",
                    ids=[123],
                ),
                True,
            ),
            (
                "trigger",
                alert_objects.AlertTrigger(
                    events=[alert_objects.EventKind.DATA_DRIFT_DETECTED]
                ),
                True,
            ),
            (
                "criteria",
                alert_objects.AlertCriteria(
                    count=5,
                    period="10m",
                ),
                True,
            ),
            # Test multiple modifications
            (
                ["summary", "severity"],
                [
                    "Job has failed again",
                    alert_objects.AlertSeverity.HIGH,
                ],
                False,
            ),
            (
                ["summary", "severity", "reset_policy"],
                [
                    "Job has failed again",
                    alert_objects.AlertSeverity.HIGH,
                    alert_objects.ResetPolicy.AUTO,
                ],
                True,
            ),
            (
                ["summary", "severity", "trigger"],
                [
                    "Job has failed again",
                    alert_objects.AlertSeverity.HIGH,
                    alert_objects.AlertTrigger(
                        events=[alert_objects.EventKind.DATA_DRIFT_SUSPECTED]
                    ),
                ],
                True,
            ),
            (
                ["criteria", "trigger"],
                [
                    alert_objects.AlertCriteria(
                        count=3,
                        period="10m",
                    ),
                    alert_objects.AlertTrigger(
                        events=[alert_objects.EventKind.DATA_DRIFT_SUSPECTED]
                    ),
                ],
                True,
            ),
        ],
    )
    @pytest.mark.parametrize(
        "force_reset",
        [False, True],
    )
    @unittest.mock.patch.object(
        framework.utils.singletons.db.SQLDB,
        "update_alert_activation",
        return_value=None,
    )
    @unittest.mock.patch.object(
        services.alerts.crud.AlertActivation,
        "store_alert_activation",
        return_value=None,
    )
    async def test_alert_reset_with_fields_updates(
        self,
        mocked_update_alert_activation,
        mocked_store_alert_activation,
        db: sqlalchemy.orm.Session,
        modify_field,
        modified_value,
        should_reset,
        force_reset,
        k8s_secrets_mock: K8sSecretsMock,
        reset_alert_caches,
    ):
        project = "project-name"
        alert_name = "failed-alert"
        alert_summary = "The job has failed"
        alert_reset_policy = alert_objects.ResetPolicy.MANUAL
        alert_entity = alert_objects.EventEntities(
            kind=alert_objects.EventEntityKind.JOB,
            project=project,
            ids=[123],
        )
        event_kind = alert_objects.EventKind.FAILED

        alert_data = services.alerts.tests.unit.crud.utils.generate_alert_data(
            project=project,
            name=alert_name,
            entity=alert_entity,
            summary=alert_summary,
            event_kind=event_kind,
            reset_policy=alert_reset_policy,
        )

        # store the initial alert
        services.alerts.crud.Alerts().store_alert(
            session=db,
            project=project,
            name=alert_name,
            alert_data=alert_data,
        )

        # activate the alert
        event = alert_objects.Event(kind=event_kind, entity=alert_entity)
        await fastapi.concurrency.run_in_threadpool(
            services.alerts.crud.Alerts().process_event_no_cache,
            db,
            event.kind,
            event,
        )
        alert = services.alerts.crud.Alerts().get_alert(
            session=db,
            project=project,
            name=alert_name,
        )
        assert alert.state == alert_objects.AlertActiveState.ACTIVE

        # modify the alert data based on the parameterized field
        if isinstance(modify_field, list):
            for field, value in zip(modify_field, modified_value):
                setattr(alert_data, field, value)
        else:
            setattr(alert_data, modify_field, modified_value)

        # store the modified alert
        services.alerts.crud.Alerts().store_alert(
            session=db,
            project=project,
            name=alert_name,
            alert_data=alert_data,
            force_reset=force_reset,
        )

        # fetch the updated alert
        alert = services.alerts.crud.Alerts().get_alert(
            session=db,
            project=project,
            name=alert_name,
        )

        # validate the state based on whether it should have reset
        expected_state = (
            alert_objects.AlertActiveState.INACTIVE
            if should_reset or force_reset
            else alert_objects.AlertActiveState.ACTIVE
        )
        assert alert.state == expected_state

    @pytest.mark.asyncio
    @unittest.mock.patch.object(
        framework.utils.singletons.db.SQLDB,
        "update_alert_activation",
        return_value=None,
    )
    @unittest.mock.patch.object(
        services.alerts.crud.AlertActivation,
        "store_alert_activation",
        return_value=None,
    )
    async def test_store_alert_update_time(
        self,
        mocked_update_alert_activation,
        mocked_store_alert_activation,
        db: sqlalchemy.orm.Session,
        k8s_secrets_mock: K8sSecretsMock,
    ):
        project = "project-name"
        alert_name = "my-alert"
        alert_summary = "testing 1 2 3"
        alert_entity = alert_objects.EventEntities(
            kind=alert_objects.EventEntityKind.MODEL_ENDPOINT_RESULT,
            project=project,
            ids=[123],
        )
        event_kind = alert_objects.EventKind.DATA_DRIFT_SUSPECTED

        alert_data = services.alerts.tests.unit.crud.utils.generate_alert_data(
            project=project,
            name=alert_name,
            entity=alert_entity,
            summary=alert_summary,
            event_kind=event_kind,
        )

        services.alerts.crud.Alerts().store_alert(
            session=db,
            project=project,
            name=alert_name,
            alert_data=alert_data,
        )

        alert = services.alerts.crud.Alerts().get_alert(
            session=db,
            project=project,
            name=alert_name,
        )
        assert alert.updated is not None
        assert alert.updated == alert.created

        # modify the alert and store again and validate that the updated field is modified
        alert_data.summary = "new summary"
        services.alerts.crud.Alerts().store_alert(
            session=db,
            project=project,
            name=alert_name,
            alert_data=alert_data,
        )

        alert = services.alerts.crud.Alerts().get_alert(
            session=db,
            project=project,
            name=alert_name,
        )
        assert alert.updated is not None
        assert alert.updated > alert.created.replace(tzinfo=timezone.utc)

    @pytest.mark.asyncio
    @unittest.mock.patch.object(
        framework.utils.singletons.db.SQLDB,
        "update_alert_activation",
        return_value=None,
    )
    @unittest.mock.patch.object(
        services.alerts.crud.AlertActivation,
        "store_alert_activation",
        return_value=None,
    )
    async def test_get_alerts_exclude_updated(
        self,
        mocked_update_alert_activation,
        mocked_store_alert_activation,
        db: sqlalchemy.orm.Session,
        k8s_secrets_mock: K8sSecretsMock,
    ):
        project = "project-name"
        alert_name = "my-alert"

        alert_data = services.alerts.tests.unit.crud.utils.generate_alert_data(
            project=project,
            name=alert_name,
            entity=services.alerts.tests.unit.crud.utils.generate_alert_entity(
                project=project
            ),
        )

        services.alerts.crud.Alerts().store_alert(
            session=db,
            project=project,
            name=alert_name,
            alert_data=alert_data,
        )

        # retrieve alerts without excluding the `updated` field (default behavior)
        alerts = services.alerts.crud.Alerts().list_alerts(
            session=db,
            project=project,
        )
        assert alerts[0].updated is not None

        # retrieve alerts with excluding the `updated` field
        alerts = services.alerts.crud.Alerts().list_alerts(
            session=db,
            project=project,
            exclude_updated=True,
        )
        assert alerts[0].updated is None

        # retrieve a specific alert with excluding the `updated` field
        alert = services.alerts.crud.Alerts().get_alert(
            session=db, project=project, name=alert_name, exclude_updated=True
        )
        assert alert.updated is None
