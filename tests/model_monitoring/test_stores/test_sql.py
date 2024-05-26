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
from typing import Optional, Union
from zoneinfo import ZoneInfo

import pytest

from mlrun.model_monitoring.db.stores.sqldb import models
from mlrun.model_monitoring.db.stores.sqldb.sql_store import SQLStoreBase


@pytest.mark.parametrize(
    ("connection_string", "expected_table"),
    [
        (None, models.SQLiteApplicationResultTable),
        ("sqlite://", models.SQLiteApplicationResultTable),
        (
            "mysql+pymysql://<username>:<password>@<host>/<dbname>",
            models.MySQLApplicationResultTable,
        ),
    ],
)
def test_get_app_metrics_table(
    connection_string: Optional[str], expected_table: type
) -> None:
    assert (
        models._get_application_result_table(connection_string=connection_string)
        == expected_table
    ), "The metrics table is different than expected"


@pytest.mark.parametrize(
    "time",
    [
        datetime.datetime.now(tz=ZoneInfo("Asia/Jerusalem")),
        "2020-05-22T08:59:54.279435+00:00",
    ],
)
def test_convert_to_datetime(time: Union[str, datetime.datetime]) -> None:
    time_key = "time"
    event = {time_key: time}
    SQLStoreBase._convert_to_datetime(event=event, key=time_key)
    new_time = event[time_key]
    assert isinstance(new_time, datetime.datetime)
    assert new_time.tzinfo == datetime.timezone.utc
