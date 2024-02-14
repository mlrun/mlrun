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
from abc import ABC

from .._common.artifacts_library import ArtifactsLibrary, Plan
from .plans import (
    CalibrationCurvePlan,
    ConfusionMatrixPlan,
    DatasetPlan,
    FeatureImportancePlan,
    ROCCurvePlan,
)
from .utils import MLTypes, MLUtils


class MLArtifactsLibrary(ArtifactsLibrary, ABC):
    """
    An abstract class for a ML framework artifacts library. Each ML framework should have an artifacts library for
    knowing what artifacts can be produced and their configurations. The default method must be implemented for when the
    user do not pass any plans.

    To add a plan to the library, simply write its name in the library class as a class variable pointing to the plan's
    class 'init_artifact' class method:

    some_artifact = SomeArtifactPlan
    """

    calibration_curve = CalibrationCurvePlan
    confusion_matrix = ConfusionMatrixPlan
    dataset = DatasetPlan
    feature_importance = FeatureImportancePlan
    roc_curve = ROCCurvePlan

    @classmethod
    def default(
        cls, model: MLTypes.ModelType, y: MLTypes.DatasetType = None, *args, **kwargs
    ) -> list[Plan]:
        """
        Get the default artifacts plans list of this framework's library.

        :param model: The model to check if its a regression model or a classification model.
        :param y:     The ground truth values to check if its multiclass and / or multi output.

        :return: The default artifacts plans list.
        """
        # Discover the algorithm functionality of the provided model:
        algorithm_functionality = MLUtils.get_algorithm_functionality(model=model, y=y)

        # Initialize the plans list:
        plans = [DatasetPlan(purpose=DatasetPlan.Purposes.TEST)]

        # Add classification plans:
        if algorithm_functionality.is_classification():
            if algorithm_functionality.is_single_output():
                plans += [
                    FeatureImportancePlan(),
                    ConfusionMatrixPlan(),
                    ROCCurvePlan(),
                ]
                if algorithm_functionality.is_binary_classification():
                    plans += [CalibrationCurvePlan()]

        # Add regression plans:
        if algorithm_functionality.is_regression():
            if algorithm_functionality.is_single_output():
                plans += [FeatureImportancePlan()]

        # Filter out the plans by probabilities requirement:
        if not hasattr(model, "predict_proba"):
            plans = [plan for plan in plans if not plan.need_probabilities]

        return plans
