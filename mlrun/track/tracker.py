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

import abc
import typing

from mlrun.execution import MLClientCtx


class Tracker(abc.ABC):
    """
    general tracker class, includes basic demands for tracker classes,
    in order to log 3rd party vendor's artifacts into MLRun
    '
    """
    MODULE_NAME = ...

    def __init__(self):
        self._utils = None
        self._tracked_platform = None  # assuming only one is being used every time
        self._client = None

    def utils(self):
        if self._utils:
            return self._utils
        from mlrun.frameworks._common import CommonUtils  # needed to avoid import issues later
        self._utils = CommonUtils
        return self._utils

    def is_enabled(self):
        """
        Checks if tracker is enabled.

        Returns:
            bool: True if the feature is enabled, False otherwise.
        """
        return True

    @abc.abstractmethod
    def pre_run(self, context: MLClientCtx, env: dict, args: dict) -> (dict, dict):
        """
        Initializes the tracking system for a 3rd party module.

        This function sets up the necessary components and resources to enable tracking of params, artifacts,
        or metrics within the module.
        Returns:
            two dictionaries, env and args containing environment and tracking data
        """
        pass

    @abc.abstractmethod
    def post_run(self, context: typing.Union[MLClientCtx, dict], args: dict):
        """
        Performs post-run tasks of logging 3rd party artifacts generated during the run.
        """
        pass
