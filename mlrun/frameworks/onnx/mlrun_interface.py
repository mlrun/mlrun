# Copyright 2018 Iguazio
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
            onnx._serialize(model),
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

    def evaluate(
        self,
        dataset: ONNXDataset,
        metrics: List[Callable[[np.ndarray, np.ndarray], float]],
    ):
        pass

    def predict(self, inputs: np.ndarray):
        pass
