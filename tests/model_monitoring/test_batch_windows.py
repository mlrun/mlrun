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

import datetime

import pytest

from mlrun.common.schemas.model_monitoring import EventFieldType
from mlrun.model_monitoring.model_monitoring_batch import BatchWindower


@pytest.mark.parametrize(
    ("batch_dict", "interval_delta"),
    [
        (
            {
                EventFieldType.MINUTES: 0,
                EventFieldType.HOURS: 1,
                EventFieldType.DAYS: 0,
            },
            datetime.timedelta(hours=1),
        ),
    ],
)
def test_get_interval_len(batch_dict: dict, interval_delta: datetime.timedelta) -> None:
    assert BatchWindower.batch_dict2window_len(batch_dict) == interval_delta
