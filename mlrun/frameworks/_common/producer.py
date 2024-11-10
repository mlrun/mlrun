# Copyright 2023 Iguazio
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
#

from typing import Optional

import mlrun
from mlrun.artifacts import Artifact

from .plan import Plan
from .utils import LoggingMode


class Producer:
    """
    Class for handling production of artifact plans during a run.
    """

    def __init__(
        self,
        context: mlrun.MLClientCtx = None,
        plans: Optional[list[Plan]] = None,
    ):
        """
        Initialize a producer with the given plans. The producer will log the produced artifacts using the given
        context.

        :param context: The context to log with.
        :param plans:   The plans the producer will manage.
        """
        # Store the context and plans:
        self._context = context
        self._plans = plans if plans is not None else []

        # Set up the logger's mode (default:  Training):
        self._mode = LoggingMode.TRAINING

        # Prepare the dictionaries to hold the artifacts. Once they are logged they will be moved from one to another:
        self._logged_artifacts = {}  # type: Dict[str, Artifact]
        self._not_logged_artifacts = {}  # type: Dict[str, Artifact]

    @property
    def mode(self) -> LoggingMode:
        """
        Get the logger's mode.

        :return: The logger mode.
        """
        return self._mode

    @property
    def context(self) -> mlrun.MLClientCtx:
        """
        Get the logger's MLRun context.

        :return: The logger's MLRun context.
        """
        return self._context

    @property
    def artifacts(self) -> dict[str, Artifact]:
        """
        Get the logged artifacts.

        :return: The logged artifacts.
        """
        return self._logged_artifacts

    def set_mode(self, mode: LoggingMode):
        """
        Set the producer's mode.

        :param mode: The mode to set.
        """
        self._mode = mode

    def set_context(self, context: mlrun.MLClientCtx):
        """
        Set the context this logger will log with.

        :param context: The to be set MLRun context.
        """
        self._context = context

    def set_plans(self, plans: list[Plan]):
        """
        Update the plans of this logger to the given list of plans here.

        :param plans: The list of plans to override the current one.
        """
        self._plans = plans

    def produce_stage(self, stage, **kwargs):
        """
        Produce the artifacts ready at the given stage and log them.

        :param stage:  The current stage to log at.
        :param kwargs: All the required produce arguments to pass onto the plans.
        """
        # Produce all the artifacts according to the given stage:
        self._produce_artifacts(stage=stage, **kwargs)

        # Log if a context is available:
        if self._context is not None:
            # Log the artifacts in queue:
            self._log_artifacts()
            # Commit:
            self._context.commit(completed=False)

    def _produce_artifacts(self, stage, **kwargs):
        """
        Go through the plans and check if they are ready to be produced in the given stage of the run. If they are,
        the logger will pass all the arguments to the 'plan.produce' method and collect the returned artifact.

        :param stage:            The stage to produce the artifact to check if its ready.
        :param kwargs:           All of the required produce arguments to pass onto the plans.
        """
        # Initialize a new list of plans for all the plans that will still need to be produced:
        plans = []

        # Go ver the plans to produce their artifacts:
        for plan in self._plans:
            # Check if the plan is ready:
            if plan.is_ready(stage=stage):
                # Produce the artifact:
                self._not_logged_artifacts = {
                    **self._not_logged_artifacts,
                    **plan.produce(**kwargs),
                }
                # If the plan should not be produced again, continue to the next one so it won't be collected:
                if not plan.is_reproducible():
                    continue
            # Collect the plan to produce it later (or again if reproducible):
            plans.append(plan)

        # Clear the old plans:
        self._plans = plans

        # Add evaluation prefix if in Evaluation mode:
        if self._mode == LoggingMode.EVALUATION:
            self._not_logged_artifacts = {
                f"evaluation-{key}": value
                for key, value in self._not_logged_artifacts.items()
            }
            for artifact in self._not_logged_artifacts.values():
                artifact.key = f"evaluation-{artifact.key}"

    def _log_artifacts(self):
        """
        Log the produced plans artifacts using the logger's context.
        """
        # Use the context to log each artifact:
        for artifact in self._not_logged_artifacts.values():
            self._context.log_artifact(artifact)

        # Collect the logged artifacts:
        self._logged_artifacts = {
            **self._logged_artifacts,
            **self._not_logged_artifacts,
        }

        # Clean the not logged artifacts dictionary:
        self._not_logged_artifacts = {}
