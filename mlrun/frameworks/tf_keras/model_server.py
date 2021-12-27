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
        model: keras.Model = None,
        model_path: str = None,
        model_name: str = None,
        modules_map: Union[Dict[str, Union[None, str, List[str]]], str] = None,
        custom_objects_map: Union[Dict[str, Union[str, List[str]]], str] = None,
        custom_objects_directory: str = None,
        model_format: str = TFKerasModelHandler.ModelFormats.SAVED_MODEL,
        to_list: bool = False,
        protocol: str = None,
        **class_args,
    ):
        """
        Initialize a serving class for a tf.keras model.

        :param context:                  The mlrun context to work with.
        :param name:                     The model name to be served.
        :param model:                    Model to handle or None in case a loading parameters were supplied.
        :param model_path:               Path to the model's directory to load it from. The model files must start with
                                         the given model name and the directory must contain based on the given model
                                         formats:
                                         * SavedModel - A zip file 'model_name.zip' or a directory named 'model_name'.
                                         * H5 - A h5 file 'model_name.h5'.
                                         * Architecture and weights - The json file 'model_name.json' and h5 weight file
                                           'model_name.h5'.
                                         The model path can be also passed as a model object path in the following
                                         format: 'store://models/<PROJECT_NAME>/<MODEL_NAME>:<VERSION>'.
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
        :param model_format:             The format used to save the model. One of the members of the
                                         TFKerasModelHandler.ModelFormats class. Defaulted to SavedModel.
        :param to_list:                  Whether to return a list instead of a numpy.ndarray. Defaulted to False.
        :param protocol:                 -
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

        # Set up a model handler:
        self._model_handler = TFKerasModelHandler(
            model=model,
            model_path=model_path,
            model_name=model_name,
            modules_map=modules_map,
            custom_objects_map=custom_objects_map,
            custom_objects_directory=custom_objects_directory,
            model_format=model_format,
            context=self.context,
        )

        # Store additional configurations:
        self._to_list = to_list

    def load(self):
        """
        Use the model handler to load the model.
        """
        if self._model_handler.model is None:
            self._model_handler.load()
        self.model = self._model_handler.model

    def predict(self, request: Dict[str, Any]) -> Union[np.ndarray, list]:
        """
        Infer the inputs through the model using 'keras.Model.predict' and return its output. The inferred data will be
        read from the "inputs" key of the request.

        :param request: The request to the model. The input to the model will be read from the "inputs" key.

        :return: The 'keras.Model.predict' returned output on the given inputs. If 'to_list' was set to True in
                 initialization, a list will be returned instead of a numpy.ndarray.
        """
        # Get the inputs:
        inputs = request["inputs"]

        # Predict:
        prediction = self.model.predict(inputs)

        # Return as list if required:
        return prediction if not self._to_list else prediction.tolist()

    def explain(self, request: Dict[str, Any]) -> str:
        """
        Return a string explaining what model is being serve in this serving function and the function name.

        :param request: A given request.

        :return: Explanation string.
        """
        return f"The '{self.model.name}' model serving function named '{self.name}'"
