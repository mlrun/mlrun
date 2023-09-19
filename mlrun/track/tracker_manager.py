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

from typing import List, Type, Union

from mlrun.execution import MLClientCtx
from mlrun.utils import logger

from .tracker import Tracker


class TrackerManager:
    """
    A class for handling multiple `Tracker` instances during a run. The manager is a singleton class and should be
    retrieved using the `mlrun.track.get_trackers_manager` function.
    """

    def __init__(self, _stale: bool = False):
        """
        Initialize a new empty tracker manager.

        :param _stale: An inner attribute to init a trackers manager in a specific staleness state.
        """
        self._trackers: List[Type[Tracker]] = []
        self._stale = _stale

    def add_tracker(self, tracker: Type[Tracker]):
        """
        Adds a Tracker to trackers list.

        :param tracker: The tracker class to add
        """
        self._trackers.append(tracker)
        logger.debug("Added tracker", tracker=tracker)

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
        # Check if the context received is a dict to initialize it as an `MLClientCtx` object:

        is_context_dict = False
        if isinstance(context, dict):
            is_context_dict = True
            context = MLClientCtx.from_dict(
                context, include_status=True, store_run=False
            )

        # Go over the trackers and call the `post_run` method:
        for tracker in self._trackers:
            tracker.post_run(context)

        # Commit changes:
        context.commit()

        # Mark the manager as stale, so it can be re-initialized next run:
        self._stale = True

        # Return the context (cast to dict if received as a dict):
        return context.to_dict() if is_context_dict else context

    def is_stale(self) -> bool:
        """
        Get the manager's staleness to check if a new one should be initialized instead.

        :return: The staleness property.
        """
        return self._stale
