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
from abc import abstractmethod

import mlrun

from ..utils import logger
from .tracker import Tracker


class BaseTracker(Tracker):
    TRACKED_MODULE_NAME = ...

    def __init__(self):
        super().__init__()
        self._tracked_platform = importlib.import_module(self.TRACKED_MODULE_NAME)
        self._run_track_kwargs = {}
        self._artifacts = {}

    @classmethod
    def is_enabled(cls) -> bool:
        is_module_found = False
        # Check for user configuration - defaulted to False if not configured:
        if (
            mlrun.mlconf.tracking.enabled
            and getattr(mlrun.mlconf.tracking, cls.TRACKED_MODULE_NAME).mode
            == "enabled"
        ):
            try:
                # Check if the module is available - can be imported:
                importlib.import_module(cls.TRACKED_MODULE_NAME)

                is_module_found = True
            except (ModuleNotFoundError, ImportError) as e:
                logger.warning(
                    f"Tracking enabled for {cls.TRACKED_MODULE_NAME}, but unable to import module "
                    f"due to the following exception: {str(e)}"
                )

        return is_module_found

    @abstractmethod
    def log_model(self, model_uri, context):
        """
        zips model dir and logs it and all artifacts
        :param model_uri: uri of model to log
        :param context: run context in which we log the model
        """
        pass

    @abstractmethod
    def log_artifact(self, context, full_path, artifact):
        """
        logs 3rd party artifacts, turns into mlrun artifacts and then stores them in list
        :param context: run context in which we log the model
        :param full_path:
        :param artifact:
        """
        pass

    @abstractmethod
    def log_dataset(self, dataset_path, context):
        """
        zips model dir and logs it and all artifacts
        :param model_uri: uri of model to log
        :param context: run context in which we log the model
        """
        pass
