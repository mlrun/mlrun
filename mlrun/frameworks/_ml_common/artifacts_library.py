from abc import ABC, abstractmethod
from enum import Enum
from typing import List

from sklearn.base import is_classifier, is_regressor

import mlrun.errors
from .._common import ModelType
from .._common.artifacts_library import ArtifactLibrary, Plan


class MLPlanStages(Enum):
    """
    Stages for a machine learning plan to be produced.
    """

    PRE_FIT = "pre_fit"
    POST_FIT = "post_fit"
    PRE_PREDICT = "pre_predict"
    POST_PREDICT = "post_predict"


class MLPlan(Plan, ABC):
    """
    An abstract class for describing a ML plan. A ML plan is used to produce artifact in a given time using the MLLogger
    or a direct call from the user using the MLArtifactsLibrary.
    """

    def __init__(
        self, need_probabilities: bool = False, auto_produce: bool = True, **produce_arguments
    ):
        """
        Initialize a new ML plan. The plan will be automatically produced if all of the required arguments to the
        produce method are given.

        :param need_probabilities: Whether this plan will need the predictions return from 'model.predict()' or
                                   'model.predict_proba()'. True means predict_proba and False predict. Defaulted to
                                   False.
        :param auto_produce:       Whether to automatically produce the artifact if all of the required arguments are
                                   given. Defaulted to True.
        :param produce_arguments:  The provided arguments to the produce method in kwargs style.
        """
        self._need_probabilities = need_probabilities
        super(MLPlan, self).__init__(auto_produce=auto_produce, **produce_arguments)

    @property
    def need_probabilities(self) -> bool:
        """
        Whether this plan require predictions returned from 'model.predict()' or 'model.predict_proba()'.

        :return: True if predict_proba and False if predict.
        """
        return self._need_probabilities

    @abstractmethod
    def is_ready(self, stage: MLPlanStages) -> bool:
        """
        Check whether or not the plan is fit for production by the given stage.

        :return: True if the plan is producible and False otherwise.
        """
        pass


class MLArtifactLibrary(ArtifactLibrary, ABC):
    """
    An abstract class for a ML framework artifacts library. Each ML framework should have an artifacts library for
    knowing what artifacts can be produced and their configurations. The default method must be implemented for when the
    user do not pass any plans.

    To add a plan to the library, simply write its name in the library class as a class variable pointing to the plan's
    class 'init_artifact' class method:

    some_artifact = SomeArtifactPlan
    """

    @classmethod
    def default(cls, model: ModelType, **kwargs) -> List[Plan]:
        """
        Get the default artifacts plans list of this framework's library.

        :return: The default artifacts plans list.
        """
        if is_classifier(model):
            return cls._default_classification()
        if is_regressor(model):
            return cls._default_regression()
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"Could not figure out if the given model '{type(model)}' is a classifier or regressor. Please contact us "
            f"on GitHub at https://github.com/mlrun/mlrun with the type of model that failed being recognized. You can "
            f"also use an explicit list of desired artifacts instead of calling the default method."
        )

    @classmethod
    @abstractmethod
    def _default_classification(cls) -> List[Plan]:
        """
        Get the default artifacts plans list of this framework's library for classification model.

        :return: The default artifacts plans list.
        """
        pass

    @classmethod
    @abstractmethod
    def _default_regression(cls) -> List[Plan]:
        """
        Get the default artifacts plans list of this framework's library for regression model.

        :return: The default artifacts plans list.
        """
        pass
