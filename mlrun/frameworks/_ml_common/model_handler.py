from abc import ABC
from typing import Dict, List, Union

from mlrun.artifacts import Artifact
from mlrun.features import Feature

from .._common import ExtraDataType, IOSampleType, ModelHandler


class MLModelHandler(ModelHandler, ABC):
    """
    Abstract class for a machine learning framework model handling, enabling loading, saving and logging it during runs.
    """

    def log(
        self,
        tag: str = "",
        labels: Dict[str, Union[str, int, float]] = None,
        parameters: Dict[str, Union[str, int, float]] = None,
        inputs: List[Feature] = None,
        outputs: List[Feature] = None,
        metrics: Dict[str, Union[int, float]] = None,
        artifacts: Dict[str, Artifact] = None,
        extra_data: Dict[str, ExtraDataType] = None,
        algorithm: str = None,
        training_set: IOSampleType = None,
        label_column: Union[str, List[str]] = None,
        feature_vector: str = None,
        feature_weights: List[float] = None,
    ):
        """
        Log the model held by this handler into the MLRun context provided.

        :param tag:             Tag of a version to give to the logged model.
        :param labels:          Labels to log the model with.
        :param parameters:      Parameters to log with the model.
        :param inputs:          A list of features this model expects to receive - the model's input ports.
        :param outputs:         A list of features this model expects to return - the model's output ports.
        :param metrics:         Metrics results to log with the model.
        :param artifacts:       Artifacts to log the model with. Will be added to the extra data.
        :param extra_data:      Extra data to log with the model.
        :param algorithm:       The algorithm of this model. If None it will be read as the model's class name.
        :param training_set:    Training set to use for infer the model and get its inputs and outputs. Do not pass both
                                training set and inputs / outputs.
        :param label_column:    The y (ground truth) labels names.
        :param feature_vector:  Feature store feature vector uri (store://feature-vectors/<project>/<name>[:tag])
        :param feature_weights: List of feature weights, one per input column.

        :raise MLRunRuntimeError:         In case is no model in this handler.
        :raise MLRunInvalidArgumentError: If a context is missing.
        """
        super(MLModelHandler, self).log(
            tag=tag,
            labels=labels,
            parameters=parameters,
            inputs=inputs,
            outputs=outputs,
            metrics=metrics,
            artifacts=artifacts,
            extra_data=extra_data,
            algorithm=algorithm,
            training_set=training_set,
            label_column=label_column,
            feature_vector=feature_vector,
            feature_weights=feature_weights,
        )
