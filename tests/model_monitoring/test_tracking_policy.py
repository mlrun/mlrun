# Copyright 2018 Iguazio
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

import mlrun.api.crud
import mlrun.utils.model_monitoring


def test_batch_intervals():
    # Check batch interval for a simple tracking policy object
    tracking_policy = mlrun.utils.model_monitoring.TrackingPolicy(
        default_batch_intervals="0 */2 * * *"
    )
    assert tracking_policy.default_batch_intervals.minute == 0
    assert tracking_policy.default_batch_intervals.hour == "*/2"

    # Check get batching interval param function
    interval_list = mlrun.api.crud.ModelEndpoints()._get_batching_interval_param(
        [0, "*/1", None]
    )
    assert interval_list == (0.0, 1.0, 0.0)
    interval_list = mlrun.api.crud.ModelEndpoints()._get_batching_interval_param(
        ["3/2", "*/1", 1]
    )
    assert interval_list == (2.0, 1.0, 0.0)
