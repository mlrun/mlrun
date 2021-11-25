from typing import Any, Dict, List, Union

import numpy as np
from tensorflow import keras

import mlrun
from mlrun.serving.v2_serving import V2ModelServer

from .model_handler import TFKerasModelHandler


class TFKerasModelServer(V2ModelServer):
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
        custom_objects_map: Union[Dict[str, Union[str, List[str]]], str] = None,
        custom_objects_directory: str = None,
        protocol: str = None,
        model_format: str = TFKerasModelHandler.ModelFormats.H5,
        **class_args,
    ):
        """
        Initialize a serving class for a tf.keras model.

        :param context:                  The mlrun context to work with.
        :param name:                     The model name to be served.
        :param model_path:               Path to the model directory to load. Can be passed as a store model object.
        :param model:                    The model to use.
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
        :param protocol:                 -
        :param model_format:             The format used to save the model. One of the members of the
                                         TFKerasModelHandler.ModelFormats class.
        :param class_args:               -
        """
        super(TFKerasModelServer, self).__init__(
            context=context,
            name=name,
            model_path=model_path,
            model=model,
            protocol=protocol,
            **class_args,
        )
        self._model_handler = TFKerasModelHandler(
            model_name=name,
            model_path=model_path,
            model=model,
            custom_objects_map=custom_objects_map,
            custom_objects_directory=custom_objects_directory,
            model_format=model_format,
            context=self.context,
        )

    def load(self):
        """
        Use the model handler to load the model.
        """
        self._model_handler.load()
        self.model = self._model_handler.model

    def predict(self, request: Dict[str, Any]) -> np.ndarray:
        """
        Infer the inputs through the model using 'keras.Model.predict' and return its output. The inferred data will be
        read from the "inputs" key of the request.

        :param request: The request to the model. The input to the model will be read from the "inputs" key.

        :return: The 'keras.Model.predict' returned output on the given inputs.
        """
        inputs = request["inputs"]
        return self.model.predict(inputs)

    def explain(self, request: Dict[str, Any]) -> str:
        """
        Return a string explaining what model is being serve in this serving function and the function name.

        :param request: A given request.

        :return: Explanation string.
        """
        return f"The '{self.model.name}' model serving function named '{self.name}'"
