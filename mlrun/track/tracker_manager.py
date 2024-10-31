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

import importlib
import inspect
from typing import Union

import mlrun.errors
from mlrun.config import config as mlconf
from mlrun.execution import MLClientCtx
from mlrun.track.tracker import Tracker
from mlrun.utils import logger
from mlrun.utils.singleton import Singleton

# Add a tracker to this list for it to be added into the global tracker manager (only if available in the interpreter):
_TRACKERS = ["mlflow"]

# A list for the available trackers during runtime. It will be setup at the beginning of the run by the function
# `_collect_available_trackers`:
_AVAILABLE_TRACKERS: list[Tracker] = None


class TrackerManager(metaclass=Singleton):
    """
    A class for handling multiple `Tracker` instances during a run. The manager is a singleton class and should be
    retrieved using the `mlrun.track.get_trackers_manager` function.
    """

    def __init__(self):
        """
        Initialize a new empty tracker manager.
        """
        self._trackers: list[Tracker] = []

        # Check general config for tracking usage, if false we return an empty manager
        if mlconf.external_platform_tracking.enabled:
            # Check if the available trackers were collected:
            if _AVAILABLE_TRACKERS is None:
                self._collect_available_trackers()

            # Add relevant trackers (enabled ones by the configuration) to be managed:
            for available_tracker in _AVAILABLE_TRACKERS:
                if available_tracker.is_enabled():
                    self.add_tracker(tracker=available_tracker())

    def add_tracker(self, tracker: Tracker):
        """
        Adds a Tracker to trackers list.

        :param tracker: The tracker class to add
        """
        # Check if tracker of same type already in manager
        if any(isinstance(t, type(tracker)) for t in self._trackers):
            logger.warn("Tracker already in manager", tracker=tracker)

        self._trackers.append(tracker)
        logger.debug("Added tracker", tracker=tracker.__class__.__name__)

    def pre_run(self, context: MLClientCtx):
        """
        Goes over all the managed trackers and calls their `pre_run` method.

        :param context: Active mlrun context.
        """
        for tracker in self._trackers:
            tracker.pre_run(context=context)

    def post_run(self, context: Union[MLClientCtx, dict]) -> Union[MLClientCtx, dict]:
        """
        Goes over all the managed trackers and calls their `post_run` method.

        :param context: Active mlrun context.

        :return: The context updated with the trackers products.
        """
        if not self._trackers:
            return context

        # Check if the context received is a dict to initialize it as an `MLClientCtx` object:
        is_context_dict = isinstance(context, dict)
        if is_context_dict:
            context = MLClientCtx.from_dict(
                context, include_status=True, store_run=False
            )

        # Go over the trackers and call the `post_run` method:
        for tracker in self._trackers:
            try:
                tracker.post_run(context)
            except Exception as e:
                logger.warn(
                    f"Tracker {tracker.__class__.__name__} failed in post run with the following exception",
                    exception=mlrun.errors.err_to_str(e),
                )

        # Commit changes:
        context.commit()

        # Return the context (cast to dict if received as a dict):
        return context.to_dict() if is_context_dict else context

    def _collect_available_trackers(self):
        """
        Set up the `_AVAILABLE_TRACKERS` list with trackers that were able to be imported.
        The tracked modules are not in MLRun's requirements and so it trys to import the module file of each and
        only if it succeeds (not raising `ModuleNotFoundError`) it collects it as an available tracker.
        """
        global _AVAILABLE_TRACKERS

        # Initialize an empty list:
        _AVAILABLE_TRACKERS = []

        # Go over the trackers and try to collect them:
        for tracker_module_name in _TRACKERS:
            # Try to import:
            try:
                tracker_module = importlib.import_module(
                    f"mlrun.track.trackers.{tracker_module_name}_tracker"
                )
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
