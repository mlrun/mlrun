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
from typing import Any, Dict, List, Type, Union

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module

import mlrun
from mlrun.serving.v2_serving import V2ModelServer

from .mlrun_interface import PyTorchMLRunInterface
from .model_handler import PyTorchModelHandler


class PyTorchModelServer(V2ModelServer):
    """
    PyTorch Model serving class, inheriting the V2ModelServer class for being initialized automatically by the model
    server and be able to run locally as part of a nuclio serverless function, or as part of a real-time pipeline.
    """

    def __init__(
        self,
        context: mlrun.MLClientCtx = None,
        name: str = None,
        model: Module = None,
        model_path: str = None,
        model_name: str = None,
        model_class: Union[Type[Module], str] = None,
        modules_map: Union[Dict[str, Union[None, str, List[str]]], str] = None,
        custom_objects_map: Union[Dict[str, Union[str, List[str]]], str] = None,
        custom_objects_directory: str = None,
        use_cuda: bool = True,
        to_list: bool = False,
        protocol: str = None,
        **class_args,
    ):
        """
        Initialize a serving class for a torch model.

        :param context:                  For internal use (passed in init).
        :param name:                     The name of this server to be initialized.
        :param model:                    Model to handle or None in case a loading parameters were supplied.
        :param model_path:               Path to the model's directory with the saved '.pt' file. The file must start
                                         with the given model name. The model path can be also passed as a model object
                                         path in the following format:
                                         'store://models/<PROJECT_NAME>/<MODEL_NAME>:<VERSION>'.
        :param model_name:               The model name for saving and logging the model:
                                         * Mandatory for loading the model from a local path.
                                         * If given a logged model (store model path) it will be read from the artifact.
                                         * If given a loaded model object and the model name is None, the name will be
                                           set to the model's object name / class.
        :param model_class:              The model's class type object. Can be passed as the class's name (string) as
                                         well. The model class must appear in the custom objects / modules map
                                         dictionary / json. If the model path given is of a store object, this model
                                         class name will be read from the logged label of the model.
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
        :param use_cuda:                 Whether or not to use cuda. Only relevant if cuda is available. Default:
                                         True.
        :param to_list:                  Whether to return a list instead of a torch.Tensor. Default: False.
        :param protocol:                 -
        :param class_args:               -
        """
        super(PyTorchModelServer, self).__init__(
            context=context,
            name=name,
            model_path=model_path,
            model=model,
            protocol=protocol,
            **class_args,
        )

        # Store the model handler attributes:
        self.model_name = model_name
        self.model_class = model_class
        self.modules_map = modules_map
        self.custom_objects_map = custom_objects_map
        self.custom_objects_directory = custom_objects_directory

        # Store the preferences:
        self.use_cuda = use_cuda
        self.to_list = to_list

        # Set up a model handler:
        self._model_handler: PyTorchModelHandler = None

        # Prepare inference parameters:
        self._pytorch_interface: PyTorchMLRunInterface = None

    def load(self):
        """
        Use the model handler to load the model.
        """
        # Initialize the model handler:
        self._model_handler = PyTorchModelHandler(
            model_path=self.model_path,
            model=self.model,
            model_name=self.model_name,
            model_class=self.model_class,
            modules_map=self.modules_map,
            custom_objects_map=self.custom_objects_map,
            custom_objects_directory=self.custom_objects_directory,
            context=self.context,
        )

        # Load the model:
        if self._model_handler.model is None:
            self._model_handler.load()
        self.model = self._model_handler.model

        # Initialize the MLRun interface:
        self._pytorch_interface = PyTorchMLRunInterface(
            model=self._model_handler.model, context=self.context
        )

    def predict(self, request: Dict[str, Any]) -> Union[Tensor, list]:
        """
        Infer the inputs through the model using MLRun's PyTorch interface and return its output. The inferred data will
        be read from the "inputs" key of the request.

        :param request: The request to the model. The input to the model will be read from the "inputs" key.

        :return: The model's prediction on the given input. If 'to_list' was set to True in initialization, a list will
                 be returned instead of a torch.Tensor.
        """
        # Get the inputs:
        inputs = request["inputs"]

        # Convert input from numpy arrays to torch tensors if needed:
        if isinstance(inputs[0], np.ndarray):
            inputs = [torch.from_numpy(array_input) for array_input in inputs]

        # Predict:
        predictions = self._pytorch_interface.predict(
            inputs=inputs, use_cuda=self.use_cuda
        )

        # Return as list if required:
        return predictions if not self.to_list else predictions.tolist()

    def explain(self, request: Dict[str, Any]) -> str:
        """
        Return a string explaining what model is being serve in this serving function and the function name.

        :param request: A given request.

        :return: Explanation string.
        """
        return f"The '{type(self.model)}' model serving function named '{self.name}'"
