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

from .base_tracker import BaseTracker
from .tracker import Tracker
from .tracker_manager import TrackerManager
from .trackers.mlflow_tracker import MLFlowTracker


def get_trackers_manager():
    """
    initiates an TrackerManager, looks for all relevant trackers and adds them
    :return: instance of TrackerManager with all relevant trackers
    """
    trackers_manager = TrackerManager()
    if MLFlowTracker.is_enabled():  # if mlflow is in env and enabled
        trackers_manager.add_tracker(MLFlowTracker)
    return trackers_manager


TRACKERS_MANAGER = get_trackers_manager()
