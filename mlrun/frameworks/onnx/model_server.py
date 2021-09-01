from typing import Any, Dict

import onnx
import onnxruntime as ort

import mlrun
from mlrun.frameworks.onnx.model_handler import ONNXModelHandler
from mlrun.serving.v2_serving import V2ModelServer


class ONNXModelServer(V2ModelServer):
    """
    ONNX Model serving class, inheriting the V2ModelServer class for being initialized automatically by the model server
    and be able to run locally as part of a nuclio serverless function, or as part of a real-time pipeline.
    """

    def __init__(
        self,
        context: mlrun.MLClientCtx,
        name: str,
        model_path: str = None,
        model: onnx.ModelProto = None,
        protocol: str = None,
        **class_args,
    ):
        """
        Initialize a serving class for an onnx.ModelProto model.

        :param context:    The mlrun context to work with.
        :param name:       The model name to be served.
        :param model_path: Path to the model directory to load. Can be passed as a store model object.
        :param model:      The model to use.
        :param protocol:   -
        :param class_args: -
        """
        super(ONNXModelServer, self).__init__(
            context=context,
            name=name,
            model_path=model_path,
            model=model,
            protocol=protocol,
            **class_args,
        )
        self._model_handler = ONNXModelHandler(
            model_name=name,
            model_path=model_path,
            model=model,
            context=self.context,
        )

    def load(self):
        """
        Use the model handler to load the model.
        """
        self._model_handler.load()
        self.model = self._model_handler.model

    def predict(self, request: Dict[str, Any]) -> list:
        """
        Infer the inputs through the model using ONNXRunTime and return its output. The inferred data will be
        read from the "inputs" key of the request.

        :param request: The request of the model. The input to the model will be read from the "inputs" key.

        :return: The ONNXRunTime session returned output on the given inputs.
        """
        images = request["inputs"]

        # TODO: Implement a session of ort
        predicted_probability = None
        return predicted_probability

    def explain(self, request: Dict[str, Any]) -> str:
        """
        Return a string explaining what model is being serve in this serving function and the function name.

        :param request: A given request.

        :return: Explanation string.
        """
        return "The '{}' model serving function named '{}'".format(
            self.model.name, self.name
        )
