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

from datetime import datetime, timezone

from sqlalchemy.orm import Session

import mlrun.common.schemas
import mlrun.model
from server.api.db.base import DBInterface


def test_store_alert_created_time(db: DBInterface, db_session: Session):
    project = "project"

    notification = mlrun.model.Notification(
        kind="slack",
        when=["error"],
        name="slack-notification",
        message="test-message",
        condition="",
        severity="info",
        params={"some-param": "some-value"},
    ).to_dict()

    new_alert = mlrun.common.schemas.AlertConfig(
        project=project,
        name="test_alert",
        summary="drift detected on the model",
        severity=mlrun.common.schemas.alert.AlertSeverity.HIGH,
        entities={
            "kind": mlrun.common.schemas.alert.EventEntityKind.MODEL_ENDPOINT_RESULT,
            "project": project,
            "ids": [1234],
        },
        trigger={"events": [mlrun.common.schemas.alert.EventKind.DATA_DRIFT_DETECTED]},
        notifications=[{"notification": notification}],
        reset_policy=mlrun.common.schemas.alert.ResetPolicy.MANUAL,
    )
    db.store_alert(db_session, new_alert)
    alerts = db.list_alerts(db_session, project)
    assert len(alerts) == 1

    end_time = datetime.now(tz=timezone.utc)

    assert alerts[0].created.replace(tzinfo=timezone.utc) < end_time
