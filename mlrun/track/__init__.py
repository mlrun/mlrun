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

from ..utils import logger
from .base_tracker import BaseTracker
from .tracker import Tracker
from .tracker_manager import TrackerManager
from .trackers import MLFlowTracker

# Add a tracker to this list for it to be added into the global tracker manager:
_AVAILABLE_TRACKERS = [MLFlowTracker]
trackers_manager = TrackerManager()


def get_trackers_manager():
    """
    Initialize a `TrackerManager`, looking for all relevant trackers and adds them to it if not empty
    :return: instance of TrackerManager with all relevant trackers
    """
    global trackers_manager
    if not len(trackers_manager.trackers):
        #  Go over the available trackers list and add them to the manager:
        for tracker in _AVAILABLE_TRACKERS:
            if tracker.is_enabled():
                logger.debug(f"Added tracker of type: {tracker}")
                trackers_manager.add_tracker(tracker)
    return trackers_manager




