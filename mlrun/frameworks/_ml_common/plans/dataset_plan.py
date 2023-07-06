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
from enum import Enum
from typing import Dict, Union

import mlrun.errors
from mlrun.artifacts import Artifact, DatasetArtifact

from ..plan import MLPlan, MLPlanStages
from ..utils import MLTypes, MLUtils


class DatasetPlan(MLPlan):
    """
    A dataset plan for creating a dataset artifact. The plan can generate the artifact with a specific dataset according
    to the provided dataset split purpose: train, validation and test.
    """

    class Purposes(Enum):
        """
        All the dataset split purposes.
        """

        OTHER = "other"
        TRAIN = "train"
        VALIDATION = "validation"
        TEST = "test"

    # Default name dictionary to map a purpose to a default dataset artifact name:
    _DEFAULT_NAMES = {
        Purposes.OTHER: "dataset",
        Purposes.TRAIN: "train_set",
        Purposes.VALIDATION: "validation_set",
        Purposes.TEST: "test_set",
    }

    def __init__(
        self,
        purpose: Union[Purposes, str] = Purposes.OTHER,
        name: str = None,
        preview: int = None,
        stats: bool = False,
        fmt: str = "parquet",
    ):
        """
        Initialize a new dataset plan.

        :param purpose: The purpose of this dataset, can be one of DatasetPlan.Purposes:
                        * "other" will manually log the given dataset.
                        * "train" will log the training set (the one who sent to 'fit').
                        * "validation" will log the validation set (the one who set in 'apply_mlrun' as 'x_test' and
                          'y_test'.
                        * "test" will log the test set (the one who sent to 'predict').
        :param name:    The name to store the dataset as. Will be defaulted according to the purpose to be "dataset"
                        for "any", "train_set" for "train" and "test_set" for "test".
        :param preview: Number of lines to store as preview in the artifact metadata.
        :param stats:   Calculate and store dataset stats in the artifact metadata. Will be replaced soon with dataset
                        artifacts.
        :param fmt:     Format to use for saving the dataset. Can be one of {"csv", "parquet", "pq", "tsdb", "kv"}.

        :raise MLRunInvalidArgumentError: If either one of the arguments is invalid.
        """
        # Validate format:
        if fmt is not None and fmt not in DatasetArtifact.SUPPORTED_FORMATS:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"The given format: '{fmt}' is not supported. Supporting only the following formats: "
                f"{', '.join(DatasetArtifact.SUPPORTED_FORMATS)}"
            )

        # Store the configurations:
        self._purpose = (
            self.Purposes(purpose)
            if not isinstance(purpose, self.Purposes)
            else purpose
        )
        self._name = name if name is not None else self._DEFAULT_NAMES[self._purpose]
        self._preview = preview
        self._stats = stats
        self._fmt = fmt
        self._plans = (
            {}
        )  # TODO: Implement DatasetPlansLibrary with dataset specific artifacts plans.

        # Continue initializing the plan:
        super(DatasetPlan, self).__init__(need_probabilities=False)

    def is_ready(self, stage: MLPlanStages, is_probabilities: bool) -> bool:
        """
        Check whether or not the plan is fit for production by the given stage.

        :param stage:            The stage to check if the plan is ready.
        :param is_probabilities: True if the 'y_pred' that will be sent to 'produce' is a prediction of probabilities
                                 (from 'predict_proba') and False if not.

        :return: True if the plan is producible and False otherwise.
        """
        # For training set:
        if self._purpose == self.Purposes.TRAIN:
            return stage == MLPlanStages.PRE_FIT

        # For validation and test sets:
        if (
            self._purpose == self.Purposes.VALIDATION
            or self._purpose == self.Purposes.TEST
        ):
            return stage == MLPlanStages.PRE_PREDICT

        # For other:
        return True

    def produce(
        self,
        x: MLTypes.DatasetType,
        y: MLTypes.DatasetType = None,
        target_columns_names: MLTypes.TargetColumnsNamesType = None,
        **kwargs,
    ) -> Dict[str, Artifact]:
        """
        Produce the dataset artifact according to this plan.

        :param x:                    A collection of inputs to a model.
        :param y:                    A collection of ground truth labels corresponding to the inputs.
        :param target_columns_names: List of names or indices to give the columns of the ground truth labels.

        :return: The produced dataset artifact.

        :raise MLRunInvalidArgumentError: If no dataset parameters were passed.
        """
        # Merge x and y into a single dataset:
        dataset, y_columns = MLUtils.concatenate_x_y(
            x=x, y=y, target_columns_names=target_columns_names
        )

        # Create the dataset artifact:
        dataset_artifact = DatasetArtifact(
            key=self._name,
            df=dataset,
            preview=self._preview,
            format=self._fmt,
            stats=self._stats,
        )

        # Add the purpose as a label:
        if self._purpose != self.Purposes.OTHER:
            dataset_artifact.labels["Purpose"] = self._purpose.value

        # TODO: Add the target columns names as an additional artifact (save as a json for example)

        # Store it:
        self._artifacts[self._name] = dataset_artifact

        return self._artifacts

    def _cli_display(self):
        """
        How the plan's products would be presented on a cli kernel.
        """
        print(self._artifacts[self._name].df)

    def _gui_display(self):
        """
        How the plan's products would be presented on a graphic IPython kernel (like a Jupyter notebook).
        """
        from IPython.display import display

        display(self._artifacts[self._name].df)
