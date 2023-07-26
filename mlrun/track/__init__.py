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
# flake8: noqa  - this is until we take care of the F401 violations with respect to __all__ & sphinx

import mlrun

from .base_tracker import BaseTracker
from .tracker import Tracker
from .tracker_manager import TrackerManager
from .trackers import MLFlowTracker

# Add a tracker to this list for it to be added into the global tracker manager:
_AVAILABLE_TRACKERS = [MLFlowTracker]
_TRACKERS_MANAGER = TrackerManager()


def get_trackers_manager() -> TrackerManager:
    """
    Initialize a `TrackerManager`, looking for all relevant trackers and adds them to it if not empty
    :return: instance of TrackerManager with all relevant trackers
    """
    global _TRACKERS_MANAGER
    # check general config for tracking usage, if false we return an empty manager
    if not mlrun.mlconf.tracking.enabled:
        return _TRACKERS_MANAGER
    # else, if manager is empty we add all relevant and enabled trackers
    if not len(_TRACKERS_MANAGER.trackers):
        for tracker in _AVAILABLE_TRACKERS:
            if tracker.is_enabled():
                _TRACKERS_MANAGER.add_tracker(tracker)
    return _TRACKERS_MANAGER
