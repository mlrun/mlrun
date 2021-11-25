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
        context: mlrun.MLClientCtx,
        name: str,
        model_class: Union[Type[Module], str] = None,
        model_path: str = None,
        model: Module = None,
        custom_objects_map: Union[Dict[str, Union[str, List[str]]], str] = None,
        custom_objects_directory: str = None,
        use_cuda: bool = True,
        protocol: str = None,
        **class_args,
    ):
        """
        Initialize a serving class for a torch model.

        :param context:                  The mlrun context to work with.
        :param name:                     The name of this server to be initialized.
        :param model_class:              The model's class type object. Can be passed as the class's name (string) as
                                         well. The model class must appear in the custom objects map dictionary / json.
                                         If the model path given is of a store object, this model class name will be
                                         read from the logged label of the model.
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
        :param use_cuda:                 Whether or not to use cuda. Only relevant if cuda is available. Defaulted to
                                         True.
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
        self._use_cuda = use_cuda
        self._model_handler = PyTorchModelHandler(
            context=self.context,
            model_name=name,
            model_class=model_class,
            model_path=model_path,
            model=model,
            custom_objects_map=custom_objects_map,
            custom_objects_directory=custom_objects_directory,
        )
        self._pytorch_interface = None  # type: PyTorchMLRunInterface

    def load(self):
        """
        Use the model handler to load the model.
        """
        self._model_handler.load()
        self.model = self._model_handler.model
        self._pytorch_interface = PyTorchMLRunInterface(
            model=self._model_handler.model, context=self.context
        )

    def predict(self, request: Dict[str, Any]) -> Tensor:
        """
        Infer the inputs through the model using MLRun's PyTorch interface and return its output. The inferred data will
        be read from the "inputs" key of the request.

        :param request: The request to the model. The input to the model will be read from the "inputs" key.

        :return: The model's prediction on the given input.
        """
        # Get the inputs:
        inputs = request["inputs"]

        # Parse the inputs:
        if isinstance(inputs[0], np.ndarray):
            inputs = torch.from_numpy(np.stack(inputs))

        # Predict:
        predictions = self._pytorch_interface.predict(
            inputs=inputs, use_cuda=self._use_cuda
        )

        return predictions

    def explain(self, request: Dict[str, Any]) -> str:
        """
        Return a string explaining what model is being serve in this serving function and the function name.

        :param request: A given request.

        :return: Explanation string.
        """
        return f"The '{type(self.model)}' model serving function named '{self.name}'"
