import os
from typing import Dict, List, Union

import onnx
import onnxoptimizer

import mlrun
from mlrun.artifacts import Artifact

from .._common import ModelHandler


class ONNXModelHandler(ModelHandler):
    """
    Class for handling an ONNX model, enabling loading and saving it during runs.
    """

    # Framework name:
    FRAMEWORK_NAME = "onnx"

    def __init__(
        self,
        model: onnx.ModelProto = None,
        model_path: str = None,
        model_name: str = None,
        context: mlrun.MLClientCtx = None,
        **kwargs,
    ):
        """
        Initialize the handler. The model can be set here so it won't require loading. Notice that if the model path
        given is of a previously logged model (store model object path), all of the other configurations will be loaded
        automatically as they were logged with the model, hence they are optional.

        :param model:      Model to handle or None in case a loading parameters were supplied.
        :param model_path: Path to the model's directory to load it from. The onnx file must start with the given model
                           name and the directory must contain the onnx file. The model path can be also passed as a
                           model object path in the following format:
                           'store://models/<PROJECT_NAME>/<MODEL_NAME>:<VERSION>'.
        :param model_name: The model name for saving and logging the model:
                           * Mandatory for loading the model from a local path.
                           * If given a logged model (store model path) it will be read from the artifact.
                           * If given a loaded model object and the model name is None, the name will be set to the
                             model's object name / class.
        :param context:    MLRun context to work with for logging the model.

        :raise MLRunInvalidArgumentError: There was no model or model directory supplied.
        """
        # Setup the base handler class:
        super(ONNXModelHandler, self).__init__(
            model=model,
            model_path=model_path,
            model_name=model_name,
            context=context,
            **kwargs,
        )

    # TODO: output_path won't work well with logging artifacts. Need to look into changing the logic of 'log_artifact'.
    def save(
        self, output_path: str = None, **kwargs
    ) -> Union[Dict[str, Artifact], None]:
        """
        Save the handled model at the given output path. If a MLRun context is available, the saved model files will be
        logged and returned as artifacts.

        :param output_path: The full path to the directory to save the handled model at. If not given, the context
                            stored will be used to save the model in the defaulted artifacts location.

        :return The saved model additional artifacts (if needed) dictionary if context is available and None otherwise.
        """
        super(ONNXModelHandler, self).save(output_path=output_path)

        # Set the output path:
        if output_path is None:
            output_path = os.path.join(self._context.artifact_path, self._model_name)

        # Save the model:
        self._model_file = f"{self._model_name}.onnx"
        onnx.save(self._model, self._model_file)

        return None

    def load(self, **kwargs):
        """
        Load the specified model in this handler.
        """
        super(ONNXModelHandler, self).load()

        # Check that the model is well formed:
        onnx.checker.check_model(self._model_file)

        # Load the ONNX model:
        self._model = onnx.load(self._model_file)

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

    def _collect_files_from_local_path(self):
        """
        If the model path given is of a local path, search for the needed model files and collect them into this handler
        for later loading the model.

        :raise MLRunNotFoundError: If the onnx file was not found.
        """
        self._model_file = os.path.join(self._model_path, f"{self._model_name}.onnx")
        if not os.path.exists(self._model_file):
            raise mlrun.errors.MLRunNotFoundError(
                f"The model file '{self._model_name}.onnx' was not found within the given "
                f"'model_path': '{self._model_path}'"
            )
