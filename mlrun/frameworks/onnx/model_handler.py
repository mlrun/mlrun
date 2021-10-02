import os
from typing import Any, Dict, List, Union

import onnx
import onnxoptimizer

import mlrun
from mlrun.artifacts import Artifact
from mlrun.frameworks._common import ModelHandler


class ONNXModelHandler(ModelHandler):
    """
    Class for handling an ONNX model, enabling loading and saving it during runs.
    """

    def __init__(
        self,
        model_name: str,
        model_path: str = None,
        model: onnx.ModelProto = None,
        context: mlrun.MLClientCtx = None,
    ):
        """
        Initialize the handler. The model can be set here so it won't require loading. Notice that if the model path
        given is of a previously logged model (store model object path), all of the other configurations will be loaded
        automatically as they were logged with the model, hence they are optional.

        :param model_name: The model name for saving and logging the model.
        :param model_path: Path to the model's directory to load it from. The onnx file must start with the given model
                           name and the directory must contain the onnx file. The model path can be also passed as a
                           model object path in the following format:
                           'store://models/<PROJECT_NAME>/<MODEL_NAME>:<VERSION>'.
        :param model:      Model to handle or None in case a loading parameters were supplied.
        :param context:    MLRun context to work with for logging the model.

        :raise ValueError: There was no model or model directory supplied.
        """
        # Setup the base handler class:
        super(ONNXModelHandler, self).__init__(
            model_name=model_name, model_path=model_path, model=model, context=context,
        )

    # TODO: output_path won't work well with logging artifacts. Need to look into changing the logic of 'log_artifact'.
    def save(
        self, output_path: str = None, *args, **kwargs
    ) -> Union[Dict[str, Artifact], None]:
        """
        Save the handled model at the given output path. If a MLRun context is available, the saved model files will be
        logged and returned as artifacts.

        :param output_path: The full path to the directory to save the handled model at. If not given, the context
                            stored will be used to save the model in the defaulted artifacts location.

        :return The saved model artifacts dictionary if context is available and None otherwise.
        """
        super(ONNXModelHandler, self).save(output_path=output_path)

        # Setup the returning model artifacts list:
        artifacts = {}  # type: Dict[str, Artifact]
        model_file = None  # type: str

        # Set the output path:
        if output_path is None:
            output_path = os.path.join(self._context.artifact_path, self._model_name)

        # Save the model:
        model_file = "{}.onnx".format(self._model_name)
        onnx.save(self._model, model_file)

        # Update the paths and log artifacts if context is available:
        self._model_file = model_file
        if self._context is not None:
            artifacts[
                self._get_model_file_artifact_name()
            ] = self._context.log_artifact(
                model_file,
                local_path=model_file,
                artifact_path=output_path,
                db_key=False,
            )

        return artifacts if self._context is not None else None

    def load(self, *args, **kwargs):
        """
        Load the specified model in this handler.
        """
        super(ONNXModelHandler, self).load()

        # Check that the model is well formed:
        onnx.checker.check_model(self._model_file)

        # Load the ONNX model:
        self._model = onnx.load(self._model_file)

    def log(
        self,
        labels: Dict[str, Union[str, int, float]] = None,
        parameters: Dict[str, Union[str, int, float]] = None,
        extra_data: Dict[str, Any] = None,
        artifacts: Dict[str, Artifact] = None,
    ):
        """
        Log the model held by this handler into the MLRun context provided.

        :param labels:     Labels to log the model with.
        :param parameters: Parameters to log with the model.
        :param extra_data: Extra data to log with the model.
        :param artifacts:  Artifacts to log the model with. Will be added to the extra data.

        :raise ValueError: In case a context is missing or there is no model in this handler.
        """
        super(ONNXModelHandler, self).log(
            labels=labels,
            parameters=parameters,
            extra_data=extra_data,
            artifacts=artifacts,
        )

        # Set default values:
        labels = {} if labels is None else labels
        parameters = {} if parameters is None else parameters
        extra_data = {} if extra_data is None else extra_data
        artifacts = {} if artifacts is None else artifacts

        # Save the model:
        model_artifacts = self.save()

        # Log the model:
        self._context.log_model(
            self._model_name,
            db_key=self._model_name,
            model_file=self._model_file,
            framework="onnx",
            labels=labels,
            parameters=parameters,
            metrics=self._context.results,
            extra_data={**model_artifacts, **artifacts, **extra_data},
        )

    def optimize(self, optimizations: List[str] = None, fixed_point: bool = False):
        """
        Use ONNX optimizer to optimize the ONNX model. The optimizations supported can be seen by calling
        'onnxoptimizer.get_available_passes()'

        :param optimizations: List of possible optimizations. If None, all of the optimizations will be used. Defaulted
                              to None.
        :param fixed_point:   Optimize the weights using fixed point. Defaulted to False.
        """
        # Set the ONNX optimizations list:
        onnx_optimizations = onnxoptimizer.get_fuse_and_elimination_passes()
        if optimizations is None:
            # Set to all optimizations:
            optimizations = onnx_optimizations

        # Optimize the model:
        self._model = onnxoptimizer.optimize(
            self._model, passes=optimizations, fixed_point=fixed_point
        )

    def to_onnx(self, *args, **kwargs) -> onnx.ModelProto:
        """
        Convert the model in this handler to an ONNX model. In this case the handled ONNX model will simply be returned.

        :return: The current handled ONNX model as there is nothing to convert.
        """
        return self._model

    def _collect_files_from_store_object(self):
        """
        If the model path given is of a store object, collect the needed model files into this handler for later loading
        the model.
        """
        # Get the artifact and model file along with its extra data:
        (
            self._model_file,
            self._model_artifact,
            self._extra_data,
        ) = mlrun.artifacts.get_model(self._model_path)

        # Get the model file:
        if self._model_file.endswith(".pkl"):
            self._model_file = self._extra_data[
                self._get_model_file_artifact_name()
            ].local()

    def _collect_files_from_local_path(self):
        """
        If the model path given is of a local path, search for the needed model files and collect them into this handler
        for later loading the model.
        """
        self._model_file = os.path.join(
            self._model_path, "{}.onnx".format(self._model_name)
        )
        if not os.path.exists(self._model_file):
            raise FileNotFoundError(
                "The model file '{}.onnx' was not found within the given 'model_path': "
                "'{}'".format(self._model_name, self._model_path)
            )
