from typing import Any, Dict, List, Union
import numpy as np

import xgboost as xgb

import mlrun
from mlrun.frameworks._ml_common import MLModelHandler
import pandas as pd
from cloudpickle import dumps
from mlrun.features import Feature
from mlrun.artifacts import Artifact


class XGBoostModelHandler(MLModelHandler):
    """
    Class for handling a XGBoost model, enabling loading and saving it during runs.
    """

    # Framework name:
    _FRAMEWORK_NAME = "xgboost"

    # Declare a type of an input sample:
    IOSample = Union[xgb.DMatrix, pd.DataFrame, np.ndarray]

    def __init__(
        self,
        model_name: str,
        model_path: str = None,
        model: xgb.XGBModel = None,
        modules_map: Union[Dict[str, Union[None, str, List[str]]], str] = None,
        custom_objects_map: Union[Dict[str, Union[str, List[str]]], str] = None,
        custom_objects_directory: str = None,
        context: mlrun.MLClientCtx = None,
    ):
        """
        Initialize the handler. The model can be set here so it won't require loading. Note you must provide at least
        one of 'model' and 'model_path'. If a model is not given, the files in the model path will be collected
        automatically to be ready for loading.

        :param model_name:               The model name for saving and logging the model.
        :param model_path:               Path to the directory with the model files. Can be passed as a model object
                                         path in the following format:
                                         'store://models/<PROJECT_NAME>/<MODEL_NAME>:<VERSION>'
        :param model:                    Model to handle or None in case a loading parameters were supplied.
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
        :param context:                  MLRun context to work with for logging the model.

        :raise MLRunInvalidArgumentError: In case one of the given parameters are invalid.
        """
        model_name = model_name or "model"
        self._training_set = None
        self._label_column = None

        super(XGBoostModelHandler, self).__init__(
            model_name=model_name,
            model_path=model_path,
            model=model,
            modules_map=modules_map,
            custom_objects_map=custom_objects_map,
            custom_objects_directory=custom_objects_directory,
            context=context,
        )

    def set_dataset(self, training_set, label_column):
        self._training_set = training_set
        self._label_column = label_column

    def log(
        self,
        labels: Dict[str, Union[str, int, float]] = None,
        parameters: Dict[str, Union[str, int, float]] = None,
        inputs: List[Feature] = None,
        outputs: List[Feature] = None,
        metrics: Dict[str, Union[int, float]] = None,
        artifacts: Dict[str, Artifact] = None,
        extra_data: Dict[str, Any] = None,
    ):
        self._context.log_model(
            self._model_name,
            db_key=self._model_name,
            body=dumps(self.model),
            labels=labels if labels is not None else {},
            artifact_path=self._context.artifact_subpath("models"),
            framework=f"{str(self._model.__module__).split('.')[0]}",
            algorithm=str(self._model.__class__.__name__),
            model_file=f"{str(self._model.__class__.__name__)}.pkl",
            metrics=self._context.results,
            format="pkl",
            training_set=self._training_set,
            label_column=self._label_column,
            extra_data=self._committed_artifacts,
        )

    def _collect_files_from_local_path(self):
        pass

    def _collect_files_from_store_object(self):
        pass

    def load(self):
        pass

    def save(self, output_path: str = None, **kwargs):
        pass

    def to_onnx(self, model_name: str = None, optimize: bool = True, **kwargs):
        pass
