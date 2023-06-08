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

import typing
from .platform_trackers import MLFlowTracker
from .tracker import Tracker
from mlrun.execution import MLClientCtx


class TrackerManager:
    def __init__(self):
        self._trackers = []

    def add_tracker(self, tracker: Tracker):
        '''
        adds a Tracker to tracking list
        '''
        self._trackers.append(tracker())

    def pre_run(
        self, context: MLClientCtx, mode: str = None, env: dict = None
    ) -> (dict, dict):
        """
        goes over all trackers and calls there pre_run function

         Returns:
            two dictionaries, env and args containing environment and tracking data for all trackers
        """
        env = env or {}
        args = {"mode": mode}
        for tracker in self._trackers:
            if tracker.is_enabled():
                env, args = tracker.pre_run(context, env, args)
        return env, args

    def post_run(self, context: typing.Union[MLClientCtx, dict], args: dict, db = None) -> dict:
        """
        goes over all trackers and calls there post_run function
        """
        for tracker in self._trackers:
            if tracker.is_enabled():
                if isinstance(context, dict):
                    context = MLClientCtx.from_dict(context)
                tracker.post_run(context, args, db=db)

        if isinstance(context, dict):
            return context
        context.commit()
        return context.to_dict()


tracking_services = TrackerManager()
if MLFlowTracker.is_relevant():  # if mlflow is in env and enabled
    tracking_services.add_tracker(MLFlowTracker)
