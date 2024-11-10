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
from typing import Optional, Union

import mlrun
from mlrun.artifacts import Artifact
from mlrun.datastore import is_store_uri
from mlrun.features import Feature

from .._common import ModelHandler
from .utils import MLTypes, MLUtils


class MLModelHandler(ModelHandler, ABC):
    """
    Abstract class for a machine learning framework model handling, enabling loading, saving and logging it during runs.
    """

    def __init__(
        self,
        model: MLTypes.ModelType = None,
        model_path: MLTypes.PathType = None,
        model_name: Optional[str] = None,
        modules_map: Union[
            dict[str, Union[None, str, list[str]]], MLTypes.PathType
        ] = None,
        custom_objects_map: Union[
            dict[str, Union[str, list[str]]], MLTypes.PathType
        ] = None,
        custom_objects_directory: MLTypes.PathType = None,
        context: mlrun.MLClientCtx = None,
        **kwargs,
    ):
        """
        Initialize the handler. The model can be set here so it won't require loading. Note you must provide at least
        one of 'model' and 'model_path'. If a model is not given, the files in the model path will be collected
        automatically to be ready for loading.

        :param model:                    Model to handle or None in case a loading parameters were supplied.
        :param model_path:               Path to the directory with the model files. Can be passed as a model object
                                         path in the following format:
                                         'store://models/<PROJECT_NAME>/<MODEL_NAME>:<VERSION>'
        :param model_name:               The model name for saving and logging the model:
                                         * Mandatory for loading the model from a local path.
                                         * If given a logged model (store model path) it will be read from the artifact.
                                         * If given a loaded model object and the model name is None, the name will be
                                           set to the model's object name / class.
        :param modules_map:              A dictionary of all the modules required for loading the model. Each key
                                         is a path to a module and its value is the object name to import from it. All
                                         the modules will be imported globally. If multiple objects needed to be
                                         imported from the same module a list can be given. The map can be passed as a
                                         path to a json file as well. For example:
                                         {
                                             "module1": None,  # => import module1
                                             "module2": ["func1", "func2"],  # => from module2 import func1, func2
                                             "module3.sub_module": "func3",  # => from module3.sub_module import func3
                                         }
                                         If the model path given is of a store object, the modules map will be read from
                                         the logged modules map artifact of the model.
        :param custom_objects_map:       A dictionary of all the custom objects required for loading the model. Each key
                                         is a path to a python file and its value is the custom object name to import
                                         from it. If multiple objects needed to be imported from the same py file a list
                                         can be given. The map can be passed as a path to a json file as well. For
                                         example:
                                         {
                                             "/.../custom_optimizer.py": "optimizer",
                                             "/.../custom_layers.py": ["layer1", "layer2"]
                                         }
                                         All the paths will be accessed from the given 'custom_objects_directory',
                                         meaning each py file will be read from 'custom_objects_directory/<MAP VALUE>'.
                                         If the model path given is of a store object, the custom objects map will be
                                         read from the logged custom object map artifact of the model.
                                         Notice: The custom objects will be imported in the order they came in this
                                         dictionary (or json). If a custom object is depended on another, make sure to
                                         put it below the one it relies on.
        :param custom_objects_directory: Path to the directory with all the python files required for the custom
                                         objects. Can be passed as a zip file as well (will be extracted during the run
                                         before loading the model). If the model path given is of a store object, the
                                         custom objects files will be read from the logged custom object artifact of the
                                         model.
        :param context:                  MLRun context to work with for logging the model.
        :param model_format:             The format to use for saving and loading the model. Should be passed as a
                                         member of the class 'ModelFormats'. Default: 'ModelFormats.PKL'.

        :raise MLRunInvalidArgumentError: In case one of the given parameters are invalid.
        """
        # Setup additional properties for logging a ml model into a ModelArtifact:
        self._algorithm = None  # type: str
        self._sample_set = None  # type: MLTypes.DatasetType
        self._target_columns = None  # type: MLTypes.TargetColumnsNamesType
        self._feature_vector = None  # type: str
        self._feature_weights = None  # type: List[float]

        # Continue the initialization:
        super().__init__(
            model=model,
            model_path=model_path,
            model_name=model_name,
            modules_map=modules_map,
            custom_objects_map=custom_objects_map,
            custom_objects_directory=custom_objects_directory,
            context=context,
            **kwargs,
        )

    @property
    def algorithm(self) -> str:
        """
        Get the model algorithm set in this handler.

        :return: The handler's model algorithm.
        """
        return self._algorithm

    @property
    def sample_set(self) -> MLTypes.DatasetType:
        """
        Get the sample dataset set in this handler.

        :return: The handler's sample dataset.
        """
        return self._sample_set

    @property
    def target_columns(self) -> MLTypes.TargetColumnsNamesType:
        """
        Get the sample dataset target columns set in this handler.

        :return: The handler's sample dataset target columns names.
        """
        return self._target_columns

    @property
    def feature_vector(self) -> str:
        """
        Get the feature vector set in this handler.

        :return: The handler's feature vector.
        """
        return self._feature_vector

    @property
    def feature_weights(self) -> list[float]:
        """
        Get the feature weights set in this handler.

        :return: The handler's feature weights.
        """
        return self._feature_weights

    def set_algorithm(self, algorithm: str):
        """
        Set the algorithm this model will be logged with.

        :param algorithm: The algorithm to set.
        """
        self._algorithm = algorithm

    def set_sample_set(
        self, sample_set: Union[MLTypes.DatasetType, mlrun.DataItem, str]
    ):
        """
        Set the sample set this model will be logged with. The sample set will be casted to a pd.DataFrame. Can be sent
        as a DataItem and as a store object string.

        :param sample_set: The sample set to set.

        :raise MLRunInvalidArgumentError: In case the sample set store uri is incorrect or if the sample set type is not
                                          supported.
        """
        # Parse the store uri to a DataItem:
        if isinstance(sample_set, str):
            if not is_store_uri(sample_set):
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"The provided sample set string is an invalid store uri: {sample_set}"
                )
            sample_set = mlrun.get_dataitem(sample_set)

        # Parse the DataItem to a DataFrame:
        if isinstance(sample_set, mlrun.DataItem):
            sample_set = sample_set.as_df()

        # Set the sample set casting it to a DataFrame:
        self._sample_set = MLUtils.to_dataframe(dataset=sample_set)

    def set_target_columns(self, target_columns: MLTypes.TargetColumnsNamesType):
        """
        Set the ground truth column names of the sample set this model will be logged with.

        :param target_columns: The ground truth (y) columns names to set.
        """
        self._target_columns = target_columns

    def set_feature_vector(self, feature_vector: str):
        """
        Set the feature vector this model will be logged with.

        :param feature_vector: The feature store feature vector uri to set
                               (store://feature-vectors/<project>/<name>[:tag]).
        """
        self._feature_vector = feature_vector

    def set_feature_weights(self, feature_weights: list[float]):
        """
        Set the feature weights this model will be logged with.

        :param feature_weights: The feature weights to set, one per input column.
        """
        self._feature_weights = feature_weights

    def log(
        self,
        tag: str = "",
        labels: Optional[dict[str, Union[str, int, float]]] = None,
        parameters: Optional[dict[str, Union[str, int, float]]] = None,
        inputs: Optional[list[Feature]] = None,
        outputs: Optional[list[Feature]] = None,
        metrics: Optional[dict[str, Union[int, float]]] = None,
        artifacts: Optional[dict[str, Artifact]] = None,
        extra_data: Optional[dict[str, MLTypes.ExtraDataType]] = None,
        algorithm: Optional[str] = None,
        sample_set: MLTypes.DatasetType = None,
        target_columns: MLTypes.TargetColumnsNamesType = None,
        feature_vector: Optional[str] = None,
        feature_weights: Optional[list[float]] = None,
    ):
        """
        Log the model held by this handler into the MLRun context provided.

        :param tag:             Tag of a version to give to the logged model. Will override the stored tag in this
                                handler.
        :param labels:          Labels to log the model with. Will be joined to the labels set.
        :param parameters:      Parameters to log with the model. Will be joined to the parameters set.
        :param inputs:          A list of features this model expects to receive - the model's input ports. If already
                                set, will be overridden by the inputs given here.
        :param outputs:         A list of features this model expects to return - the model's output ports. If already
                                set, will be overridden by the outputs given here.
        :param metrics:         Metrics results to log with the model.
        :param artifacts:       Artifacts to log the model with. Will be joined to the registered artifacts and added to
                                the extra data.
        :param extra_data:      Extra data to log with the model.
        :param algorithm:       The algorithm of this model. If None it will be read as the model's class name.
        :param sample_set:      Sample set to use for getting the model's inputs, outputs and base stats for model
                                monitoring. Do not pass both sample set and inputs / outputs.
        :param target_columns:  The ground truth (y) labels names.
        :param feature_vector:  Feature store feature vector uri (store://feature-vectors/<project>/<name>[:tag])
        :param feature_weights: List of feature weights, one per input column.

        :raise MLRunRuntimeError:         In case is no model in this handler.
        :raise MLRunInvalidArgumentError: If a context is missing.
        """
        # Update the algorithm (set default if both values are None):
        if algorithm is not None:
            self._algorithm = algorithm
        elif self._algorithm is None:
            self.set_algorithm(algorithm=self._model.__class__.__name__)

        # Update the sample set:
        if sample_set is not None:
            self.set_sample_set(sample_set=sample_set)
        if target_columns is not None:
            self.set_target_columns(target_columns=target_columns)

        # Update the feature parameters:
        if feature_vector is not None:
            self.set_feature_vector(feature_vector=feature_vector)
        if feature_weights is not None:
            self.set_feature_weights(feature_weights=feature_weights)

        # Continue with the handler logging:
        super().log(
            tag=tag,
            labels=labels,
            parameters=parameters,
            inputs=inputs,
            outputs=outputs,
            metrics=metrics,
            artifacts=artifacts,
            extra_data=extra_data,
            algorithm=self._algorithm,
            sample_set=self._sample_set,
            target_columns=self._target_columns,
            feature_vector=self._feature_vector,
            feature_weights=self._feature_weights,
        )

    def update(
        self,
        labels: Optional[dict[str, Union[str, int, float]]] = None,
        parameters: Optional[dict[str, Union[str, int, float]]] = None,
        inputs: Optional[list[Feature]] = None,
        outputs: Optional[list[Feature]] = None,
        metrics: Optional[dict[str, Union[int, float]]] = None,
        artifacts: Optional[dict[str, Artifact]] = None,
        extra_data: Optional[dict[str, MLTypes.ExtraDataType]] = None,
        feature_vector: Optional[str] = None,
        feature_weights: Optional[list[float]] = None,
    ):
        """
        Update the model held by this handler into the MLRun context provided, updating the model's artifact properties
        in the same model path provided.

        :param labels:          Labels to update or add to the model.
        :param parameters:      Parameters to update or add to the model.
        :param inputs:          A list of features this model expects to receive - the model's input ports. If already
                                set, will be overridden by the inputs given here.
        :param outputs:         A list of features this model expects to return - the model's output ports. If already
                                set, will be overridden by the outputs given here.
        :param metrics:         Metrics results to log with the model.
        :param artifacts:       Artifacts to update or add to the model. Will be joined to the registered artifacts and
                                added to the extra data.
        :param extra_data:      Extra data to update or add to the model.
        :param feature_vector:  Feature store feature vector uri (store://feature-vectors/<project>/<name>[:tag])
        :param feature_weights: List of feature weights, one per input column.

        :raise MLRunInvalidArgumentError: In case a context is missing or the model path in this handler is missing or
                                          not of a store object.
        """
        # Update the feature parameters:
        if feature_vector is not None:
            self._feature_vector = feature_vector
        if feature_weights is not None:
            self._feature_weights = feature_weights

        # Continue with the handler update:
        super().update(
            labels=labels,
            parameters=parameters,
            inputs=inputs,
            outputs=outputs,
            metrics=metrics,
            artifacts=artifacts,
            extra_data=extra_data,
            feature_vector=self._feature_vector,
            feature_weights=self._feature_weights,
        )
