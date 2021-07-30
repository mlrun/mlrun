from typing import Any, Dict

import torch

import mlrun
from mlrun.frameworks.pytorch.model_handler import PyTorchModelHandler
from mlrun.serving.v2_serving import V2ModelServer


class PyTorchModelServer(V2ModelServer):
    """
    Tensorflow.keras Model serving class, inheriting the V2ModelServer class for being initialized automatically by the
    model server and be able to run locally as part of a nuclio serverless function, or as part of a real-time pipeline.
    """

    def __init__(
        self,
        context: mlrun.MLClientCtx,
        name: str,
        model_path: str = None,
        model: torch.Module = None,
        protocol: str = None,
        **class_args,
    ):
        """
        Initialize a serving class for a tf.keras model.

        :param context:      The mlrun context to work with.
        :param name:         The name of this server to be initialized.
        :param model_path:   Path to the model directory to load. Can be passed as a store model object.
        :param model:        The model to use.
        :param protocol:     -
        :param class_args:   -
        """
        super(PyTorchModelServer, self).__init__(
            context=context,
            name=name,
            model_path=model_path,
            model=model,
            protocol=protocol,
            **class_args,
        )

        self._model_handler = PyTorchModelHandler(
            context=self.context,
            model_name="",
            model_class="",
            model_path=model_path,
            model=model,
            custom_objects_directory="",
            custom_objects_map={},
        )

        raise NotImplementedError

    def load(self):
        """
        Use the model handler to load the model.
        """
        self._model_handler.load()
        self.model = self._model_handler.model

    def predict(self, request: Dict[str, Any]) -> list:
        """
        Infer the inputs through the model using 'torch.Module.__call__' and return its output. The inferred data will
        be read from the "inputs" key of the request.

        :param request: The request of the model. The input to the model will be read from the "inputs" key.

        :return: The 'torch.Module.__call__' returned output on the given inputs.
        """
        images = request["inputs"]
        predicted_probability = self.model(images)
        return predicted_probability.tolist()

    def explain(self, request: Dict[str, Any]) -> str:
        """
        Return a string explaining what model is being serve in this serving function and the function name.

        :param request: A given request.

        :return: Explanation string.
        """
        return "The '{}' model serving function named '{}'".format(
            type(self.model), self.name
        )
