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

from mlrun.config import config as mlconf
from .tracker import Tracker

from mlrun.execution import MLClientCtx


class BaseTracker(Tracker):
    MODULE_NAME = ...

    def is_enabled(self) -> bool:
        """
        Checks if tracker is enabled. only checks in config.
        :return: True if the feature is enabled, False otherwise.
        """
        return mlconf.trackers.is_enabled

    @abc.abstractmethod
    def log_model(self, model_uri, context):
        pass
