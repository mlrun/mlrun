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
from mlrun.execution import MLClientCtx

from ..utils import logger
from .tracker import Tracker


class BaseTracker(Tracker):
    """
    a base class for specific trackers, starts to implement class functions
    """

    # The module name this tracker will look for.
    TRACKED_MODULE_NAME = ...

    def __init__(self):
        super().__init__()
        self._tracked_platform = importlib.import_module(self.TRACKED_MODULE_NAME)
        self._run_track_kwargs = {}
        # will add artifacts in mlrun format to here
        self._artifacts = {}

    @classmethod
    def is_enabled(cls) -> bool:
        # Check the tracker's configuration:
        if getattr(mlrun.mlconf.tracking, cls.TRACKED_MODULE_NAME).mode != "enabled":
            return False

        # Check if the module to track is available in the interpreter:
        try:
            importlib.import_module(cls.TRACKED_MODULE_NAME)
        except ModuleNotFoundError as module_not_found_error:
            # The module is missing in the interpreter:
            logger.warning(
                f"Tracking enabled for {cls.TRACKED_MODULE_NAME}, but the module was not found  "
                f"in the interpreter: {str(module_not_found_error)}"
            )
            return False
        except ImportError as import_error:
            # There was an error during the import of the module:
            logger.warning(
                f"Tracking enabled for {cls.TRACKED_MODULE_NAME}, but MLRun was unable to import the "
                f"module due to the following error: {str(import_error)}"
            )
            return False

        return True

    @abstractmethod
    def log_model(self, model_uri: str, context: MLClientCtx):
        """
        zips model dir and logs it and all artifacts
        :param model_uri: uri of model to log
        :param context: run context in which we log the model
        """
        pass

    @abstractmethod
    def log_artifact(self, context: MLClientCtx, local_path: str, artifact):
        # todo add hint for TRACKED_MODULE_NAME artifact
        """
        logs 3rd party artifacts, turns into mlrun artifacts and then stores them in list to log with the model later
        :param context: run context in which we log the model
        :param local_path: path to artifact to log
        :param artifact:  artifact of module self.TRACKED_MODULE_NAME
        """
        pass

    @abstractmethod
    def log_dataset(self, dataset_path: str, context: MLClientCtx):
        """
        zips model dir and logs it and all artifacts
        :param dataset_path: dataset_path of dataset to log
        :param context: run context in which we log the model
        """
        pass
