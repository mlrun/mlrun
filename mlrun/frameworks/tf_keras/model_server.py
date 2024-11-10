# Copyright 2023 Iguazio
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
from typing import Any, Optional, Union

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
        context: mlrun.MLClientCtx = None,
        name: Optional[str] = None,
        model: keras.Model = None,
        model_path: Optional[str] = None,
        model_name: Optional[str] = None,
        modules_map: Optional[
            Union[dict[str, Union[None, str, list[str]]], str]
        ] = None,
        custom_objects_map: Optional[
            Union[dict[str, Union[str, list[str]]], str]
        ] = None,
        custom_objects_directory: Optional[str] = None,
        model_format: str = TFKerasModelHandler.ModelFormats.SAVED_MODEL,
        to_list: bool = False,
        protocol: Optional[str] = None,
        **class_args,
    ):
        """
        Initialize a serving class for a tf.keras model.

        :param context:                  For internal use (passed in init).
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
                                         TFKerasModelHandler.ModelFormats class. Default: SavedModel.
        :param to_list:                  Whether to return a list instead of a numpy.ndarray. Default: False.
        :param protocol:                 -
        :param class_args:               -
        """
        super().__init__(
            context=context,
            name=name,
            model_path=model_path,
            model=model,
            protocol=protocol,
            **class_args,
        )

        # Store the model handler attributes:
        self.model_name = model_name
        self.modules_map = modules_map
        self.custom_objects_map = custom_objects_map
        self.custom_objects_directory = custom_objects_directory
        self.model_format = model_format

        # Store additional configurations:
        self.to_list = to_list

        # Set up a model handler:
        self._model_handler: TFKerasModelHandler = None

    def load(self):
        """
        Use the model handler to load the model.
        """
        # Initialize the model handler:
        self._model_handler = TFKerasModelHandler(
            model=self.model,
            model_path=self.model_path,
            model_name=self.model_name,
            modules_map=self.modules_map,
            custom_objects_map=self.custom_objects_map,
            custom_objects_directory=self.custom_objects_directory,
            model_format=self.model_format,
            context=self.context,
        )

        # Load the model:
        if self._model_handler.model is None:
            self._model_handler.load()
        self.model = self._model_handler.model

    def predict(self, request: dict[str, Any]) -> Union[np.ndarray, list]:
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
        return prediction if not self.to_list else prediction.tolist()

    def explain(self, request: dict[str, Any]) -> str:
        """
        Return a string explaining what model is being serve in this serving function and the function name.

        :param request: A given request.

        :return: Explanation string.
        """
        return f"The '{self.model.name}' model serving function named '{self.name}'"
