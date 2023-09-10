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

from typing import Type
from uuid import UUID

import pytest

from mlrun.model_monitoring.writer import (
    ModelMonitoringWriter,
    RawEvent,
    WriterEvent,
    _WriterEventTypeError,
    _WriterEventValueError,
    application_result_key,
)


@pytest.mark.parametrize("endpoint_ids", [[UUID(int=4), "flsdkl2210kd"], ["justepid"]])
@pytest.mark.parametrize("app_names", [["app_1", "app_2"]])
def test_unique_kv_keys(endpoint_ids: list[str], app_names: list[str]) -> None:
    keys = [
        application_result_key(endpoint_id=ep_id, app_name=app_name)
        for ep_id in endpoint_ids
        for app_name in app_names
    ]
    assert len(set(keys)) == len(keys), "Some keys are not unique"


@pytest.mark.parametrize(
    ("event", "exception"),
    [
        ("key1:val1,key2:val2", _WriterEventTypeError),
        ({WriterEvent.ENDPOINT_ID: "ep2211"}, _WriterEventValueError),
    ],
)
def test_reconstruct_event_error(event: RawEvent, exception: Type[Exception]) -> None:
    with pytest.raises(exception):
        ModelMonitoringWriter._reconstruct_event(event)
