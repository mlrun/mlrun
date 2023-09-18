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
import importlib
import inspect
from typing import Type, List

from mlrun.config import config as mlconf

from .tracker import Tracker
from .tracker_manager import TrackerManager

# Add a tracker to this list for it to be added into the global tracker manager (only if available in the interpreter):
_TRACKERS = ["mlflow"]

# A list for the available trackers during runtime. It will be setup at the beginning of the run by the function
# `_collect_available_trackers`:
_AVAILABLE_TRACKERS: List[Type[Tracker]] = None

# The global singleton trackers manager:
_TRACKERS_MANAGER = TrackerManager(_stale=True)


def _collect_available_trackers():
    """
    Set up the `_AVAILABLE_TRACKERS` list with trackers that were able to be imported. The tracked modules are not in
    MLRun's requirements and so it trys to import the module file of each and only if it succeeds (not raising
    `ModuleNotFoundError`) it collects it as an available tracker.
    """
    global _AVAILABLE_TRACKERS

    # Initialize an empty list:
    _AVAILABLE_TRACKERS = []

    # Go over the trackers and try to collect them:
    for tracker_module_name in _TRACKERS:
        # Try to import:
        try:
            tracker_module = importlib.import_module(f"mlrun.track.tracker.{tracker_module_name}_tracker")
        except ModuleNotFoundError:
            continue
        # Look for `Tracker` classes inside:
        _AVAILABLE_TRACKERS += [
            member
            for _, member in inspect.getmembers(
                tracker_module,
                lambda m: (
                    # Validate it is declared in the module:
                    hasattr(m, "__module__")
                    and m.__module__ == tracker_module.__name__
                    # Validate it is a `Tracker`:
                    and isinstance(m, type)
                    and issubclass(m, Tracker)
                    # Validate it is not a "protected" `Tracker`:
                    and not m.__name__.startswith("_")
                ),
            )
        ]


def get_trackers_manager() -> TrackerManager:
    """
    Get a trackers manager. The returned manager will be the global manager of this runtime. It is initialized once per
    runtime.

    :return: The trackers manager.
    """
    global _TRACKERS_MANAGER, _AVAILABLE_TRACKERS

    # If the manager was already initialized, return it:
    if not _TRACKERS_MANAGER.is_stale():
        return _TRACKERS_MANAGER

    # Initialize a new empty tracker manager:
    _TRACKERS_MANAGER = TrackerManager()

    # Check general config for tracking usage, if false we return an empty manager
    if not mlconf.external_platform_tracking.enabled:
        return _TRACKERS_MANAGER

    # Check if the available trackers were collected:
    if _AVAILABLE_TRACKERS is None:
        _collect_available_trackers()

    # Add relevant trackers (enabled ones by the configuration) to be managed:
    for available_tracker in _AVAILABLE_TRACKERS:
        if available_tracker.is_enabled():
            _TRACKERS_MANAGER.add_tracker(tracker=available_tracker)

    return _TRACKERS_MANAGER
