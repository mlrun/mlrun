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

import pytest

import mlrun.track
from mlrun.track import Tracker
from mlrun.track.tracker_manager import TrackerManager
from mlrun.track.trackers.mlflow_tracker import MLFlowTracker


class TrackerExample(Tracker):
    # just some random module
    TRACKED_MODULE_NAME = "os"

    def is_enabled(self):
        return True

    def pre_run(self, context) -> dict:
        return context.to_dict()

    def post_run(self, context):
        return True


# see that the manager adds each tracker by themselves and then all together
@pytest.mark.parametrize(
    "tracker_list",
    [
        [MLFlowTracker, TrackerExample],
        [MLFlowTracker],
        [TrackerExample],
    ],
)
def test_add_tracker(tracker_list):
    # enable tracking in config for inspection
    mlrun.mlconf.external_platform_tracking.enabled = True
    trackers_manager = TrackerManager()
    for tracker in tracker_list:
        trackers_manager.add_tracker(tracker)
        assert trackers_manager._trackers[-1] == tracker
    assert len(trackers_manager._trackers) == len(tracker_list)


def test_get_trackers_manager(rundb_mock):
    # enable tracking in config for inspection
    mlrun.mlconf.external_platform_tracking.enabled = True
    trackers_manager = mlrun.track.get_trackers_manager()
    assert type(trackers_manager) is TrackerManager
    # from here we need to change after we add more trackers
    # see that added trackers correspond to project configs
    assert len(trackers_manager._trackers) == 1
