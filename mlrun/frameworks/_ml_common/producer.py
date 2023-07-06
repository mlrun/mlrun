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
#
from .._common import LoggingMode, Producer
from .plan import MLPlanStages


class MLProducer(Producer):
    """
    Class for handling production of artifact plans during a run.
    """

    def is_probabilities_required(self) -> bool:
        """
        Check if probabilities are required in order to produce some of the artifacts.

        :return: True if probabilities are required by at least one plan and False otherwise.
        """
        return any(plan.need_probabilities for plan in self._plans)

    def produce_stage(
        self, stage: MLPlanStages, is_probabilities: bool = False, **kwargs
    ):
        """
        Produce the artifacts ready at the given stage and log them.

        :param stage:            The current stage to log at.
        :param is_probabilities: True if the 'y_pred' is a prediction of probabilities (from 'predict_proba') and False
                                 if not. Default: False.
        :param kwargs:           All of the required produce arguments to pass onto the plans.
        """
        # Produce all the artifacts according to the given stage:
        self._produce_artifacts(
            stage=stage, is_probabilities=is_probabilities, **kwargs
        )

        # Log if a context is available:
        if self._context is not None:
            # Log the artifacts in queue:
            self._log_artifacts()
            # Commit:
            self._context.commit(completed=False)

    def _produce_artifacts(
        self, stage: MLPlanStages, is_probabilities: bool = False, **kwargs
    ):
        """
        Go through the plans and check if they are ready to be produced in the given stage of the run. If they are,
        the logger will pass all the arguments to the 'plan.produce' method and collect the returned artifact.

        :param stage:            The stage to produce the artifact to check if its ready.
        :param is_probabilities: True if the 'y_pred' that will be sent to 'produce' is a prediction of probabilities
                                 (from 'predict_proba') and False if not. Default: False.
        :param kwargs:           All of the required produce arguments to pass onto the plans.
        """
        # Initialize a new list of plans for all the plans that will still need to be produced:
        plans = []

        # Go ver the plans to produce their artifacts:
        for plan in self._plans:
            # Check if the plan is ready:
            if plan.is_ready(stage=stage, is_probabilities=is_probabilities):
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
