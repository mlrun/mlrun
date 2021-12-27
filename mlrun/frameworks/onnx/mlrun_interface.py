from typing import Callable, List

import numpy as np
import onnx
import onnxruntime

import mlrun

from .dataset import ONNXDataset


# TODO: Finish evaluation and prediction.
class ONNXMLRunInterface:
    """
    An interface for enabling convenient MLRun features for the ONNX framework.
    """

    # MLRun's context default name:
    DEFAULT_CONTEXT_NAME = "mlrun-onnx"

    def __init__(
        self,
        model: onnx.ModelProto,
        execution_providers: List[str] = None,
        context: mlrun.MLClientCtx = None,
    ):
        # Set the context:
        self._context = (
            context
            if context is not None
            else mlrun.get_or_create_ctx(self.DEFAULT_CONTEXT_NAME)
        )

        # Store the model:
        self._model = model

        # Set the execution providers (default will prefer CUDA Execution Provider over CPU Execution Provider):
        self._execution_providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if execution_providers is None
            else execution_providers
        )

        # initialize the onnx run time session:
        self._inference_session = onnxruntime.InferenceSession(
            onnx._serialize(model), providers=self._execution_providers,
        )

        # Get the input layers names:
        self._input_layers = [
            input_layer.name for input_layer in self._inference_session.get_inputs()
        ]

        # Get the outputs layers names:
        self._output_layers = [
            output_layer.name for output_layer in self._inference_session.get_outputs()
        ]

    def evaluate(
        self,
        dataset: ONNXDataset,
        metrics: List[Callable[[np.ndarray, np.ndarray], float]],
    ):
        pass

    def predict(self, inputs: np.ndarray):
        pass
