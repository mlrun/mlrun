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

from os import environ
from typing import Type, Union

from mlrun.execution import MLClientCtx
from mlrun.utils import logger

from .tracker import Tracker


class TrackerManager:
    def __init__(self):
        self._trackers = []

    def add_tracker(self, tracker: Type[Tracker]):
        """
        adds a Tracker to tracking list
        :param tracker: The tracker class to add
        """
        self._trackers.append(tracker())
        logger.debug("Added tracker", tracker=tracker)

    def clear_trackers(self):
        """
        removes all trackers from the tracking list
        """
        self._trackers.clear()

    def pre_run(self, context: MLClientCtx):
        """
        goes over all trackers and calls their pre_run function
        :param context: current mlrun context
        """
        logger.debug("Number of trackers", Number=len(self._trackers))
        for tracker_num, tracker in enumerate(self._trackers):
            if tracker.is_enabled():
                logger.debug("Tracking run number", number=tracker_num)
                env = tracker.pre_run(context=context)
                environ.update(env)  # needed in order to set 3rd party experiment id

    def post_run(self, context: Union[MLClientCtx, dict]):
        """
        goes over all trackers and calls there post_run function
        :param context: current mlrun context
        """
        if isinstance(context, dict):
            context = MLClientCtx.from_dict(
                context, include_status=True, store_run=False
            )

        for tracker in self._trackers:
            if tracker.is_enabled():
                tracker.post_run(context)
        self.clear_trackers()

    @property
    def trackers(self):
        return self._trackers
