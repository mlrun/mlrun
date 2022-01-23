from abc import ABC
from typing import List

from .._common import ModelType
from .._common.artifacts_library import ArtifactsLibrary, Plan
from .plans import (  # FeatureImportancePlan,; LearningCurvesPlan,; ROCCurvePlan,
    ConfusionMatrixPlan,
    DatasetPlan,
)
from .utils import AlgorithmFunctionality, DatasetType


class MLArtifactsLibrary(ArtifactsLibrary, ABC):
    """
    An abstract class for a ML framework artifacts library. Each ML framework should have an artifacts library for
    knowing what artifacts can be produced and their configurations. The default method must be implemented for when the
    user do not pass any plans.

    To add a plan to the library, simply write its name in the library class as a class variable pointing to the plan's
    class 'init_artifact' class method:

    some_artifact = SomeArtifactPlan
    """

    dataset = DatasetPlan
    # feature_importance = FeatureImportancePlan
    confusion_matrix = ConfusionMatrixPlan
    # roc_curve = ROCCurvePlan
    # learning_curves = LearningCurvesPlan

    @classmethod
    def default(
        cls, model: ModelType, y: DatasetType = None, *args, **kwargs
    ) -> List[Plan]:
        """
        Get the default artifacts plans list of this framework's library.

        :param model: The model to check if its a regression model or a classification model.
        :param y:     The ground truth values to check if its multiclass and / or multi output.

        :return: The default artifacts plans list.
        """
        # Discover the algorithm functionality of the provided model:
        algorithm_functionality = AlgorithmFunctionality.get_algorithm_functionality(
            model=model, y=y
        )

        # Initialize the plans list:
        plans = [DatasetPlan(purpose=DatasetPlan.Purposes.TEST)]

        # Add classification plans:
        if algorithm_functionality.is_classification():
            # Add binary classification plans:
            if algorithm_functionality.is_binary_classification():
                plans += [
                    # FeatureImportancePlan(),
                    ConfusionMatrixPlan(),
                    # ROCCurvePlan(),
                    # LearningCurvesPlan(),
                ]

        # Add regression plans:
        if algorithm_functionality.is_regression():
            # Add single output regression plans:
            if algorithm_functionality.is_single_output():
                # plans += [FeatureImportancePlan()]
                pass  # TODO: Wait for Alex to correct the plans and uncomment all of these.
        return plans
