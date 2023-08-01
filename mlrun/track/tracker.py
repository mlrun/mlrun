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
from typing import Union

from mlrun.execution import MLClientCtx


class Tracker(abc.ABC):
    """
    Abstract tracker class, describes the interface for tracker classes,
    in order to log 3rd party vendor's artifacts into MLRun
    """

    def __init__(self):
        self._client = None

    @abc.abstractmethod
    def is_enabled(self):
        """
        Checks if tracker is enabled.
        :return: True if the feature is enabled, False otherwise.
        """
        pass

    @abc.abstractmethod
    def pre_run(self, context: MLClientCtx) -> dict:
        """
        Initializes the tracking system for a 3rd party module.
        This function sets up the necessary components and resources to enable tracking of params, artifacts,
        or metrics within the module.
        :param context: current mlrun context

        :return: env containing environment data to log and track 3-rd party runs
        """
        pass

    @abc.abstractmethod
    def post_run(self, context: Union[MLClientCtx, dict]):
        """
        Performs post-run tasks of logging 3rd party artifacts generated during the run.
        :param context: current mlrun context
        """
        pass
