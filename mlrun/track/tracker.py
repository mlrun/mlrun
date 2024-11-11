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

from abc import ABC, abstractmethod
from typing import Any, Optional, Union

from mlrun.artifacts import Artifact, ModelArtifact
from mlrun.execution import MLClientCtx
from mlrun.model import RunObject
from mlrun.projects import MlrunProject


class Tracker(ABC):
    """
    Abstract tracker class, describes the interface for a tracker: A class for tracking 3rd party vendor's for logging
    their artifacts and experiments into MLRun. There are two tracking modes:

    * Online: While running an MLRun function, a tracker will be used via the `pre_run` and `post_run` methods to track
      a run from the tracked vendor into MLRun.
    * Offline: Manually importing models and artifacts into an MLRun project using the `import_x` methods.
    """

    @staticmethod
    @abstractmethod
    def is_enabled() -> bool:
        """
        Checks if tracker is enabled.

        :return: True if the tracking configuration is enabled, False otherwise.
        """
        pass

    def pre_run(self, context: MLClientCtx):
        """
        Initializes the tracking system for a 3rd party module. This function sets up the necessary components and
        resources to enable tracking of the module.

        :param context: Current mlrun context
        """
        pass

    def post_run(self, context: Union[MLClientCtx, dict]):
        """
        Performs post-run tasks for logging 3rd party artifacts generated during the run.

        :param context: Current mlrun context
        """
        pass

    def import_run(
        self,
        project: MlrunProject,
        reference_id: Any,
        function_name: str,
        handler: Optional[str] = None,
        **kwargs,
    ) -> RunObject:
        """
        Import a previous run from a 3rd party vendor to MLRun.

        :param project:       The MLRun project to import the run to.
        :param reference_id:  A reference for the run to import from the 3rd party vendor.
        :param function_name: The MLRun function to assign this run to.
        :param handler:       The handler for MLRun's RunObject

        :return: The newly imported run object.
        """
        pass

    def import_model(
        self, project: MlrunProject, reference_id: Any, **kwargs
    ) -> ModelArtifact:
        """
        Import a model from a 3rd party vendor to MLRun.

        :param project:      The MLRun project to import the model to.
        :param reference_id: A reference for the model to import from the 3rd party vendor.

        :return: The newly imported model artifact.
        """
        pass

    def import_artifact(
        self, project: MlrunProject, reference_id: Any, **kwargs
    ) -> Artifact:
        """
        Import an artifact from a 3rd party vendor to MLRun.

        :param project:      The MLRun project to import the artifact to.
        :param reference_id: A reference for the artifact to import from the 3rd party vendor.

        :return: The newly imported artifact.
        """
        pass
