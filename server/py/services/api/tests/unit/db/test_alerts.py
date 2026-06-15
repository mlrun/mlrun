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

from datetime import UTC, datetime, timedelta

import mlrun
import mlrun.common.schemas
import mlrun.common.schemas.alert as alert_objects

from framework.tests.unit.db.common_fixtures import TestDatabaseBase


class TestAlertStateDB(TestDatabaseBase):
    def test_store_alert_state_sets_cooldown_end_time(self):
        """store_alert_state with cooldown_end_time persists the value in the DB row."""
        alert_id = self._create_alert(self._db, self._db_session)
        cooldown_end_time = datetime.now(UTC) + timedelta(minutes=1)

        self._db.store_alert_state(
            session=self._db_session,
            project="project",
            name="alert",
            last_updated=datetime.now(UTC),
            active=True,
            alert_id=alert_id,
            cooldown_end_time=cooldown_end_time,
        )

        state = self._db.get_alert_state(self._db_session, alert_id)
        assert state.cooldown_end_time is not None
        assert (
            abs(
                (
                    state.cooldown_end_time - cooldown_end_time.replace(tzinfo=None)
                ).total_seconds()
            )
            < 1
        )

    def test_store_alert_state_clear_cooldown(self):
        """store_alert_state with clear_cooldown=True sets cooldown_end_time to NULL in the DB row."""
        alert_id = self._create_alert(self._db, self._db_session)

        # first set a cooldown_end_time
        self._db.store_alert_state(
            session=self._db_session,
            project="project",
            name="alert",
            last_updated=datetime.now(UTC),
            active=True,
            alert_id=alert_id,
            cooldown_end_time=datetime.now(UTC) + timedelta(minutes=1),
        )
        assert (
            self._db.get_alert_state(self._db_session, alert_id).cooldown_end_time
            is not None
        )

        # now clear it
        self._db.store_alert_state(
            session=self._db_session,
            project="project",
            name="alert",
            last_updated=datetime.now(UTC),
            active=False,
            alert_id=alert_id,
            clear_cooldown=True,
        )

        assert (
            self._db.get_alert_state(self._db_session, alert_id).cooldown_end_time
            is None
        )

    def test_list_alerts_pending_cooldown_reset(self):
        """list_alerts_pending_cooldown_reset returns only alerts whose cooldown has elapsed."""
        # alert whose cooldown has already passed
        elapsed_id = self._create_alert(
            self._db, self._db_session, name="elapsed-alert"
        )
        self._db.store_alert_state(
            session=self._db_session,
            project="project",
            name="elapsed-alert",
            last_updated=datetime.now(UTC),
            active=True,
            alert_id=elapsed_id,
            cooldown_end_time=datetime.now(UTC) - timedelta(seconds=1),
        )

        # alert still within cooldown
        active_id = self._create_alert(
            self._db, self._db_session, name="active-cooldown-alert"
        )
        self._db.store_alert_state(
            session=self._db_session,
            project="project",
            name="active-cooldown-alert",
            last_updated=datetime.now(UTC),
            active=True,
            alert_id=active_id,
            cooldown_end_time=datetime.now(UTC) + timedelta(minutes=5),
        )

        # alert with no cooldown set
        no_cooldown_id = self._create_alert(
            self._db, self._db_session, name="no-cooldown-alert"
        )
        self._db.store_alert_state(
            session=self._db_session,
            project="project",
            name="no-cooldown-alert",
            last_updated=datetime.now(UTC),
            active=True,
            alert_id=no_cooldown_id,
        )

        # alert with elapsed cooldown but active=False — should NOT be reset
        inactive_id = self._create_alert(
            self._db, self._db_session, name="inactive-alert"
        )
        self._db.store_alert_state(
            session=self._db_session,
            project="project",
            name="inactive-alert",
            last_updated=datetime.now(UTC),
            active=False,
            alert_id=inactive_id,
            cooldown_end_time=datetime.now(UTC) - timedelta(seconds=1),
        )

        pending = self._db.list_alerts_pending_cooldown_reset(self._db_session)
        pending_names = [alert.name for alert in pending]
        assert "elapsed-alert" in pending_names
        assert "active-cooldown-alert" not in pending_names
        assert "no-cooldown-alert" not in pending_names
        assert "inactive-alert" not in pending_names

    def _create_alert(self, db, session, project="project", name="alert") -> int:
        alert = alert_objects.AlertConfig(
            project=project,
            name=name,
            summary="test",
            severity=alert_objects.AlertSeverity.LOW,
            entities=alert_objects.EventEntities(
                kind=alert_objects.EventEntityKind.JOB,
                project=project,
                ids=["1"],
            ),
            trigger=alert_objects.AlertTrigger(events=[alert_objects.EventKind.FAILED]),
            notifications=[
                alert_objects.AlertNotification(
                    notification=mlrun.common.schemas.Notification(
                        kind="slack",
                        name="slack",
                        secret_params={"webhook": "https://slack.com/api/api.test"},
                    )
                )
            ],
            reset_policy=alert_objects.ResetPolicy.AUTO,
        )
        stored = db.store_alert(session, alert)
        return stored.id
