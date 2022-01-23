from typing import Dict, List, Union

import numpy as np
import pandas as pd

import mlrun.errors
from mlrun.artifacts import Artifact, DatasetArtifact

from ..plan import MLPlan, MLPlanStages
from ..utils import concatenate_x_y


class DatasetPlan(MLPlan):
    """
    A dataset plan for creating a dataset artifact. The plan can generate the artifact with a specific dataset according
    to the provided dataset split purpose: train, validation and test.
    """

    class Purposes:
        """
        All of the dataset split purposes.
        """

        OTHER = "other"
        TRAIN = "train"
        VALIDATION = "validation"
        TEST = "test"

        @staticmethod
        def get_all_purposes() -> List[str]:
            """
            Get all the purposes found in this class.

            :return: A list of all the purposes.
            """
            return [
                getattr(DatasetPlan.Purposes, purpose)
                for purpose in dir(DatasetPlan.Purposes)
                if not purpose.startswith("_")
            ]

    # Default name dictionary to map a purpose to a default dataset artifact name:
    _DEFAULT_NAMES = {
        Purposes.OTHER: "dataset",
        Purposes.TRAIN: "train_set",
        Purposes.VALIDATION: "validation_set",
        Purposes.TEST: "test_set",
    }

    def __init__(
        self,
        purpose: str = Purposes.OTHER,
        name: str = None,
        preview: int = None,
        stats: bool = False,
        fmt: str = "parquet",
    ):
        """
        Initialize a new dataset plan.

        :param purpose: The purpose of this dataset, can be one of DatasetPlan.Purposes:
                        * "other" will look for 'x', 'y' parameters in kwargs of produce.
                        * "train" will look for 'x_train', 'y_train' parameters in kwargs of produce.
                        * "validation" will look for 'x_validation', 'y_validation' parameters in kwargs of produce.
                        * "test" will look for 'x_test', 'y_test' parameters in kwargs of produce.
        :param name:    The name to store the dataset as. Will be defaulted according to the purpose to be "dataset"
                        for "any", "train_set" for "train" and "test_set" for "test".
        :param preview: Number of lines to store as preview in the artifact metadata.
        :param stats:   Calculate and store dataset stats in the artifact metadata. Will be replaced soon with dataset
                        artifacts.
        :param fmt:     Format to use for saving the dataset. Can be one of {"csv", "parquet", "pq", "tsdb", "kv"}.

        :raise MLRunInvalidArgumentError: If either one of the arguments is invalid.
        """
        # Validate purpose:
        if purpose not in self.Purposes.get_all_purposes():
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"The given dataset purpose: '{purpose}' is not recognized. Supporting only the following: "
                f"{', '.join(self.Purposes.get_all_purposes())}"
            )

        # Validate format:
        if fmt is not None and fmt not in DatasetArtifact.SUPPORTED_FORMATS:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"The given format: '{fmt}' is not supported. Supporting only the following formats: "
                f"{', '.join(DatasetArtifact.SUPPORTED_FORMATS)}"
            )

        # Store the configurations:
        self._purpose = purpose
        self._name = name if name is not None else self._DEFAULT_NAMES[purpose]
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
        return True

    def produce(
        self,
        x: Union[list, dict, np.ndarray, pd.DataFrame, pd.Series],
        y: Union[list, dict, np.ndarray, pd.DataFrame, pd.Series] = None,
        y_columns: Union[List[str], List[int]] = None,
        **kwargs,
    ) -> Dict[str, Artifact]:
        """
        Produce the dataset artifact according to this plan.

        :param x:         A collection of inputs to a model.
        :param y:         A collection of ground truth labels corresponding to the inputs.
        :param y_columns: List of names or indices to give the columns of the ground truth labels.

        :return: The produced dataset artifact.

        :raise MLRunInvalidArgumentError: If no dataset parameters were passed.
        """
        # Merge x and y into a single dataset:
        dataset, y_columns = concatenate_x_y(x=x, y=y, y_columns=y_columns)

        # Create the dataset artifact:
        dataset_artifact = DatasetArtifact(
            key=self._name,
            df=dataset,
            preview=self._preview,
            format=self._fmt,
            stats=self._stats,
            **kwargs,
        )

        # Add the purpose as a label:
        if self._purpose != self.Purposes.OTHER:
            dataset_artifact.labels["Purpose"] = self._purpose

        # TODO: Add the y columns as an additional artifact (save as a json for example)

        # Store it:
        self._artifacts[self._name] = dataset_artifact

        return self._artifacts

    def _cli_display(self):
        """
        How the plan's products would be presented on a cli kernel.
        """
        print(self._artifacts[self._name].df)

    def _visual_display(self):
        """
        How the plan's products would be presented on a graphic IPython kernel (like a Jupyter notebook).
        """
        from IPython.display import display

        display(self._artifacts[self._name].df)
