from http import HTTPStatus

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from mlrun.api import schemas
from mlrun.api.utils.singletons.db import get_db
from mlrun.config import config


async def do_nothing():
    pass


def test_list_schedules(db: Session, client: TestClient) -> None:
    resp = client.get("projects/default/schedules")
    assert resp.status_code == HTTPStatus.OK.value, "status"
    assert "schedules" in resp.json(), "no schedules"

    labels_1 = {
        "label1": "value1",
    }
    cron_trigger = schemas.ScheduleCronTrigger(year="1999")
    schedule_name = "schedule-name"
    project = config.default_project
    get_db().create_schedule(
        db,
        project,
        schedule_name,
        schemas.ScheduleKinds.local_function,
        do_nothing,
        cron_trigger,
        config.httpdb.scheduling.default_concurrency_limit,
        labels_1,
    )

    labels_2 = {
        "label2": "value2",
    }
    schedule_name_2 = "schedule-name-2"
    get_db().create_schedule(
        db,
        project,
        schedule_name_2,
        schemas.ScheduleKinds.local_function,
        do_nothing,
        cron_trigger,
        config.httpdb.scheduling.default_concurrency_limit,
        labels_2,
    )

    _get_and_assert_single_schedule(client, {"labels": "label1"}, schedule_name)
    _get_and_assert_single_schedule(client, {"labels": "label2"}, schedule_name_2)
    _get_and_assert_single_schedule(client, {"labels": "label1=value1"}, schedule_name)
    _get_and_assert_single_schedule(
        client, {"labels": "label2=value2"}, schedule_name_2
    )


def _get_and_assert_single_schedule(
    client: TestClient, get_params: dict, schedule_name: str
):
    resp = client.get("projects/default/schedules", params=get_params)
    assert resp.status_code == HTTPStatus.OK.value, "status"
    result = resp.json()["schedules"]
    assert len(result) == 1
    assert result[0]["name"] == schedule_name
