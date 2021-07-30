from typing import Any, Dict

from tensorflow import keras

import mlrun
from mlrun.frameworks.keras.model_handler import KerasModelHandler
from mlrun.serving.v2_serving import V2ModelServer


class KerasModelServer(V2ModelServer):
    """
    Tensorflow.keras Model serving class, inheriting the V2ModelServer class for being initialized automatically by the
    model server and be able to run locally as part of a nuclio serverless function, or as part of a real-time pipeline.
    """

    def __init__(
        self,
        context: mlrun.MLClientCtx,
        name: str,
        model_path: str = None,
        model: keras.Model = None,
        protocol: str = None,
        model_format: str = KerasModelHandler.ModelFormats.H5,
        **class_args,
    ):
        """
        Initialize a serving class for a tf.keras model.

        :param context:      The mlrun context to work with.
        :param name:         The model name to be served.
        :param model_path:   Path to the model directory to load. Can be passed as a store model object.
        :param model:        The model to use.
        :param protocol:     -
        :param model_format: The format used to save the model. One of the members of the KerasModelHandler.ModelFormats
                             class.
        :param class_args:   -
        """
        super(KerasModelServer, self).__init__(
            context=context,
            name=name,
            model_path=model_path,
            model=model,
            protocol=protocol,
            **class_args,
        )
        self._model_handler = KerasModelHandler(
            model_name=name,
            context=self.context,
            model_path=model_path,
            model=model,
            model_format=model_format,
        )

    def load(self):
        """
        Use the model handler to load the model.
        """
        self._model_handler.load()
        self.model = self._model_handler.model

    def predict(self, request: Dict[str, Any]) -> list:
        """
        Infer the inputs through the model using 'keras.Model.predict' and return its output. The inferred data will be
        read from the "inputs" key of the request.

        :param request: The request of the model. The input to the model will be read from the "inputs" key.

        :return: The 'keras.Model.predict' returned output on the given inputs.
        """
        images = request["inputs"]
        predicted_probability = self.model.predict(images)
        return predicted_probability.tolist()

    def explain(self, request: Dict[str, Any]) -> str:
        """
        Return a string explaining what model is being serve in this serving function and the function name.

        :param request: A given request.

        :return: Explanation string.
        """
        return "The '{}' model serving function named '{}'".format(
            self.model.name, self.name
        )
