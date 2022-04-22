from typing import Any, Dict, List, Tuple, Union

import numpy as np
import onnx
import onnxruntime

import mlrun
from mlrun.serving.v2_serving import V2ModelServer

from .model_handler import ONNXModelHandler


class ONNXModelServer(V2ModelServer):
    """
    ONNX Model serving class, inheriting the V2ModelServer class for being initialized automatically by the model server
    and be able to run locally as part of a nuclio serverless function, or as part of a real-time pipeline.
    """

    def __init__(
        self,
        context: mlrun.MLClientCtx,
        name: str,
        model: onnx.ModelProto = None,
        model_path: str = None,
        model_name: str = None,
        execution_providers: List[Union[str, Tuple[str, Dict[str, Any]]]] = None,
        protocol: str = None,
        **class_args,
    ):
        """
        Initialize a serving class for an onnx.ModelProto model.

        :param context:             The mlrun context to work with.
        :param name:                The model name to be served.
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
        :param execution_providers: List of the execution providers. The first provider in the list will be the most
                                    preferred. For example, a CUDA execution provider with configurations and a CPU
                                    execution provider:
                                    [
                                        (
                                            'CUDAExecutionProvider',
                                            {
                                                'device_id': 0,
                                                'arena_extend_strategy': 'kNextPowerOfTwo',
                                                'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                                                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                                                'do_copy_in_default_stream': True,
                                            }
                                        ),
                                        'CPUExecutionProvider'
                                    ]
                                    Defaulted to None - will prefer CUDA Execution Provider over CPU Execution Provider.
        :param protocol:            -
        :param class_args:          -
        """
        super(ONNXModelServer, self).__init__(
            context=context,
            name=name,
            model_path=model_path,
            model=model,
            protocol=protocol,
            **class_args,
        )

        # Set up a model handler:
        self._model_handler = ONNXModelHandler(
            model_name=model_name,
            model_path=model_path,
            model=model,
            context=self.context,
        )

        # Set the execution providers (default will prefer CUDA Execution Provider over CPU Execution Provider):
        self._execution_providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if execution_providers is None
            else execution_providers
        )

        # Prepare inference parameters:
        self._inference_session = None  # type: onnxruntime.InferenceSession
        self._input_layers = None  # type: List[str]
        self._output_layers = None  # type: List[str]

    def load(self):
        """
        Use the model handler to get the model file path and initialize an ONNX run time inference session.
        """
        # Load the model:
        if self._model_handler.model is None:
            self._model_handler.load()
        self.model = self._model_handler.model

        # initialize the onnx run time session:
        self._inference_session = onnxruntime.InferenceSession(
            onnx._serialize(self._model_handler.model),
            providers=self._execution_providers,
        )

        # Get the input layers names:
        self._input_layers = [
            input_layer.name for input_layer in self._inference_session.get_inputs()
        ]

        # Get the outputs layers names:
        self._output_layers = [
            output_layer.name for output_layer in self._inference_session.get_outputs()
        ]

    def predict(self, request: Dict[str, Any]) -> np.ndarray:
        """
        Infer the inputs through the model using ONNXRunTime and return its output. The inferred data will be
        read from the "inputs" key of the request.

        :param request: The request to the model. The input to the model will be read from the "inputs" key.

        :return: The ONNXRunTime session returned output on the given inputs.
        """
        # Read the inputs from the request:
        inputs = request["inputs"]

        # Infer the inputs through the model:
        return self._inference_session.run(
            output_names=self._output_layers,
            input_feed={
                input_layer: data
                for input_layer, data in zip(self._input_layers, inputs)
            },
        )

    def explain(self, request: Dict[str, Any]) -> str:
        """
        Return a string explaining what model is being serve in this serving function and the function name.

        :param request: A given request.

        :return: Explanation string.
        """
        return f"The '{self.model.name}' model serving function named '{self.name}'"
