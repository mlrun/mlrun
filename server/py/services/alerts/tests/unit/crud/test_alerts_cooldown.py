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

import unittest
import unittest.mock
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise

import fastapi.concurrency
import pytest
import sqlalchemy.orm

import mlrun
import mlrun.common.schemas.alert as alert_objects
import mlrun.errors

import framework.utils.singletons.db
import services.alerts.crud
import services.alerts.tests.unit.crud.utils
from framework.tests.unit.common_fixtures import K8sSecretsMock
from services.alerts.tests.unit.conftest import TestAlertsBase


class TestAlertsCooldown(TestAlertsBase):
    @pytest.mark.parametrize(
        "cooldown_period, reset_policy, expectation",
        [
            # Valid: cooldown_period with reset_policy=auto
            (
                "1m",
                alert_objects.ResetPolicy.AUTO,
                does_not_raise(),
            ),
            # Valid: no cooldown_period (always allowed)
            (
                None,
                alert_objects.ResetPolicy.AUTO,
                does_not_raise(),
            ),
            (
                None,
                alert_objects.ResetPolicy.MANUAL,
                does_not_raise(),
            ),
            # Invalid: cooldown_period with reset_policy=manual
            (
                "1m",
                alert_objects.ResetPolicy.MANUAL,
                pytest.raises(mlrun.errors.MLRunBadRequestError),
            ),
            # Invalid: malformed duration string
            (
                "not-a-duration",
                alert_objects.ResetPolicy.AUTO,
                pytest.raises(mlrun.errors.MLRunBadRequestError),
            ),
            # Invalid: cooldown_period below the minimum (one second under cooldown_reset_interval=15s)
            (
                "14s",
                alert_objects.ResetPolicy.AUTO,
                pytest.raises(mlrun.errors.MLRunBadRequestError),
            ),
            # Valid: exactly at the minimum
            (
                "15s",
                alert_objects.ResetPolicy.AUTO,
                does_not_raise(),
            ),
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
    async def test_validate_cooldown_period(
        self,
        mocked_update_alert_activation,
        mocked_store_alert_activation,
        db: sqlalchemy.orm.Session,
        k8s_secrets_mock: K8sSecretsMock,
        monkeypatch,
        cooldown_period: str | None,
        reset_policy: alert_objects.ResetPolicy,
        expectation: AbstractContextManager,
    ):
        """Validates that cooldown_period is accepted or rejected correctly based on its value and reset_policy.

        cooldown_reset_interval is pinned to 15s so the boundary cases are stable regardless of config changes.
        """
        monkeypatch.setattr(mlrun.mlconf.alerts, "cooldown_reset_interval", 15)
        project = "project-name"
        alert_name = "cooldown-alert"
        alert_entity = services.alerts.tests.unit.crud.utils.generate_alert_entity(
            project=project,
        )

        alert_data = services.alerts.tests.unit.crud.utils.generate_alert_data(
            project=project,
            name=alert_name,
            entity=alert_entity,
            reset_policy=reset_policy,
            cooldown_period=cooldown_period,
        )

        with expectation:
            services.alerts.crud.Alerts().store_alert(
                session=db,
                project=project,
                name=alert_name,
                alert_data=alert_data,
            )

    @pytest.mark.parametrize(
        "new_cooldown_period",
        [
            # Removing cooldown_period is a functional change, alert should reset
            None,
            # Changing cooldown_period to a different value is a functional change, alert should reset
            "5m",
        ],
    )
    async def test_cooldown_period_change_resets_alert(
        self,
        db: sqlalchemy.orm.Session,
        k8s_secrets_mock: K8sSecretsMock,
        new_cooldown_period: str | None,
        reset_alert_caches,
    ):
        """Updating cooldown_period on an active alert resets it, since it is a functional field."""
        project = "project-name"
        alert_name = "cooldown-reset-alert"

        alert_data = services.alerts.tests.unit.crud.utils.generate_alert_data(
            project=project,
            name=alert_name,
            entity=services.alerts.tests.unit.crud.utils.generate_alert_entity(
                project=project,
            ),
            reset_policy=alert_objects.ResetPolicy.AUTO,
            cooldown_period="1m",
        )

        stored_alert = services.alerts.crud.Alerts().store_alert(
            session=db,
            project=project,
            name=alert_name,
            alert_data=alert_data,
        )
        framework.utils.singletons.db.get_db().store_alert_state(
            session=db,
            project=project,
            name=alert_name,
            last_updated=None,
            active=True,
            alert_id=stored_alert.id,
        )

        alert_data.cooldown_period = new_cooldown_period
        services.alerts.crud.Alerts().store_alert(
            session=db,
            project=project,
            name=alert_name,
            alert_data=alert_data,
        )

        alert = services.alerts.crud.Alerts().get_alert(
            session=db, project=project, name=alert_name
        )
        assert alert.state == alert_objects.AlertActiveState.INACTIVE

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
    async def test_alert_active_during_cooldown(
        self,
        mocked_update_alert_activation,
        mocked_store_alert_activation,
        db: sqlalchemy.orm.Session,
        k8s_secrets_mock: K8sSecretsMock,
        reset_alert_caches,
    ):
        """Alert with cooldown_period stays ACTIVE after being triggered and store_alert_state is called
        with cooldown_end_time."""
        project = "project-name"
        alert_name = "cooldown-active-alert"
        alert_entity = services.alerts.tests.unit.crud.utils.generate_alert_entity(
            project=project,
        )
        event_kind = alert_objects.EventKind.FAILED

        alert_data = services.alerts.tests.unit.crud.utils.generate_alert_data(
            project=project,
            name=alert_name,
            entity=alert_entity,
            event_kind=event_kind,
            reset_policy=alert_objects.ResetPolicy.AUTO,
            cooldown_period="1m",
        )

        services.alerts.crud.Alerts().store_alert(
            session=db,
            project=project,
            name=alert_name,
            alert_data=alert_data,
        )

        with unittest.mock.patch.object(
            framework.utils.singletons.db.SQLDB,
            "store_alert_state",
            wraps=framework.utils.singletons.db.get_db().store_alert_state,
        ) as mock_store_state:
            event = alert_objects.Event(kind=event_kind, entity=alert_entity)
            await fastapi.concurrency.run_in_threadpool(
                services.alerts.crud.Alerts().process_event_no_cache,
                db,
                event.kind,
                event,
            )

            alert = services.alerts.crud.Alerts().get_alert(
                session=db, project=project, name=alert_name
            )
            assert alert.state == alert_objects.AlertActiveState.ACTIVE

            store_state_calls = mock_store_state.call_args_list
            cooldown_calls = [
                c for c in store_state_calls if c.kwargs.get("cooldown_end_time")
            ]
            assert len(cooldown_calls) > 0, (
                "store_alert_state should be called with cooldown_end_time"
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
    async def test_auto_without_cooldown_resets_immediately(
        self,
        mocked_update_alert_activation,
        mocked_store_alert_activation,
        db: sqlalchemy.orm.Session,
        k8s_secrets_mock: K8sSecretsMock,
        reset_alert_caches,
    ):
        """AUTO policy with no cooldown_period resets the alert immediately after notification is sent."""
        project = "project-name"
        alert_name = "no-cooldown-alert"
        alert_entity = services.alerts.tests.unit.crud.utils.generate_alert_entity(
            project=project,
        )
        event_kind = alert_objects.EventKind.FAILED

        alert_data = services.alerts.tests.unit.crud.utils.generate_alert_data(
            project=project,
            name=alert_name,
            entity=alert_entity,
            event_kind=event_kind,
            reset_policy=alert_objects.ResetPolicy.AUTO,
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
            session=db, project=project, name=alert_name
        )
        assert alert.state == alert_objects.AlertActiveState.INACTIVE

    async def test_reset_alert_clears_cooldown(
        self,
        db: sqlalchemy.orm.Session,
        k8s_secrets_mock: K8sSecretsMock,
        reset_alert_caches,
    ):
        """reset_alert() calls store_alert_state with clear_cooldown=True and obj={} to clear cooldown_end_time
        and stale full_object JSON."""
        project = "project-name"
        alert_name = "cooldown-reset-test"

        alert_data = services.alerts.tests.unit.crud.utils.generate_alert_data(
            project=project,
            name=alert_name,
            entity=services.alerts.tests.unit.crud.utils.generate_alert_entity(
                project=project,
            ),
            reset_policy=alert_objects.ResetPolicy.AUTO,
            cooldown_period="1m",
        )

        stored_alert = services.alerts.crud.Alerts().store_alert(
            session=db,
            project=project,
            name=alert_name,
            alert_data=alert_data,
        )
        framework.utils.singletons.db.get_db().store_alert_state(
            session=db,
            project=project,
            name=alert_name,
            last_updated=None,
            active=True,
            alert_id=stored_alert.id,
        )

        with unittest.mock.patch.object(
            framework.utils.singletons.db.SQLDB,
            "store_alert_state",
            wraps=framework.utils.singletons.db.get_db().store_alert_state,
        ) as mock_store_state:
            services.alerts.crud.Alerts().reset_alert(
                session=db,
                project=project,
                name=alert_name,
            )

            mock_store_state.assert_called_once()
            call_kwargs = mock_store_state.call_args.kwargs
            assert call_kwargs.get("clear_cooldown") is True
            assert call_kwargs.get("obj") == {}

        alert = services.alerts.crud.Alerts().get_alert(
            session=db, project=project, name=alert_name
        )
        assert alert.state == alert_objects.AlertActiveState.INACTIVE
