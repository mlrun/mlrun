from typing import Callable, Dict, List, Tuple, Union

import torch
from torch import Tensor

import mlrun
from mlrun.artifacts import Artifact

from ..._dl_common.loggers import LoggerMode, MLRunLogger, TrackableType
from ..model_handler import PyTorchModelHandler
from .logging_callback import LoggingCallback


class MLRunLoggingCallback(LoggingCallback):
    """
    Callback for logging data during training / validation via mlrun's context. Each tracked hyperparameter and metrics
    results will be logged per epoch and at the end of the run the model will be saved and logged as well. Some plots
    will be available as well. To summerize, the available data in mlrun will be:

    * For each epoch:

      * Tracking table: epoch, static hyperparameters, dynamic hyperparameters, training metrics, validation metrics.
      * Per iteration (batch) chart artifacts for the training and validation metrics.

    * At the end of the run:

      * Per epoch chart artifacts for the validation summaries and dynamic hyperparameters.
      * Model is logged with all of the files and artifacts.

    All the collected data will be available in this callback post the training / validation process and can be accessed
    via the 'training_results', 'validation_results', 'static_hyperparameters', 'dynamic_hyperparameters' and
    'summaries' properties.
    """

    def __init__(
        self,
        context: mlrun.MLClientCtx,
        modules_map: Union[Dict[str, Union[None, str, List[str]]], str] = None,
        custom_objects_map: Union[Dict[str, Union[str, List[str]]], str] = None,
        custom_objects_directory: str = None,
        model_name: str = None,
        model_path: str = None,
        input_sample: PyTorchModelHandler.IOSample = None,
        output_sample: PyTorchModelHandler.IOSample = None,
        log_model_labels: Dict[str, TrackableType] = None,
        log_model_parameters: Dict[str, TrackableType] = None,
        log_model_extra_data: Dict[str, Union[TrackableType, Artifact]] = None,
        dynamic_hyperparameters: Dict[
            str, Tuple[str, Union[List[Union[str, int]], Callable[[], TrackableType]]]
        ] = None,
        static_hyperparameters: Dict[
            str, Union[TrackableType, Tuple[str, List[Union[str, int]]]]
        ] = None,
        auto_log: bool = False,
    ):
        """
        Initialize an mlrun logging callback with the given hyperparameters and logging configurations. Notice: In order
        to log the model, its class (torch.Module) must be in the custom objects map or the modules map.

        :param context:                  MLRun context to log to. Its parameters will be logged automatically  if
                                         'auto_log' is True.
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
        :param model_name:               The model name to use for storing the model artifact. If not given, the model's
                                         class name will be used.
        :param model_path:               The model's store object path. Mandatory for evaluation (to know which model to
                                         update).
        :param input_sample:             Input sample to the model for logging additional data regarding the input ports
                                         of the model. If None, the input sample will be read automatically from the
                                         training / evaluation process.
        :param output_sample:            Output sample of the model for logging additional data regarding the output
                                         ports of the model. If None, the input sample will be read automatically from
                                         the training / evaluation process.
        :param log_model_labels:         Labels to log with the model.
        :param log_model_parameters:     Parameters to log with the model.
        :param log_model_extra_data:     Extra data to log with the model.
        :param dynamic_hyperparameters:  If needed to track a hyperparameter dynamically (sample it each epoch) it
                                         should be passed here. The parameter expects a dictionary where the keys are
                                         the hyperparameter chosen names and the values are tuples of object key and a
                                         list with the key chain. A key chain is a list of keys and indices to know how
                                         to access the needed hyperparameter. If the hyperparameter is not of accessible
                                         from any of the HyperparametersKeys, a custom callable method can be passed in
                                         the tuple instead of the key chain when providing the word
                                         HyperparametersKeys.CUSTOM. For example, to track the 'lr' attribute of
                                         an optimizer and a custom parameter, one should pass:
                                         {
                                             "lr": (HyperparametersKeys.OPTIMIZER, ["param_groups", 0, "lr"]),
                                             "custom parameter": (HyperparametersKeys.CUSTOM, get_custom_parameter)
                                         }
        :param static_hyperparameters:   If needed to track a hyperparameter one time per run it should be passed here.
                                         The parameter expects a dictionary where the keys are the
                                         hyperparameter chosen names and the values are the hyperparameter static value
                                         or a tuple of object key and a list with the key chain just like the dynamic
                                         hyperparameter. For example, to track the 'epochs' of an experiment run, one
                                         should pass:
                                         {
                                             "epochs": 7
                                         }
        :param auto_log:                 Whether or not to enable auto logging for logging the context parameters and
                                         trying to track common static and dynamic hyperparameters.
        """
        super(MLRunLoggingCallback, self).__init__(
            dynamic_hyperparameters=dynamic_hyperparameters,
            static_hyperparameters=static_hyperparameters,
            auto_log=auto_log,
        )

        # Replace the logger with an MLRunLogger:
        del self._logger
        self._logger = MLRunLogger(
            context=context,
            log_model_labels=log_model_labels,
            log_model_parameters=log_model_parameters,
            log_model_extra_data=log_model_extra_data,
        )

        # Store the additional PyTorchModelHandler parameters for logging the model later:
        self._model_name = model_name
        self._model_path = model_path
        self._modules_map = modules_map
        self._custom_objects_map = custom_objects_map
        self._custom_objects_directory = custom_objects_directory
        self._input_sample = input_sample
        self._output_sample = output_sample

    def on_run_end(self):
        """
        Before the run ends, this method will be called to log the model and the run summaries charts.
        """
        # Check if the logger is in evaluation mode, if so, log the last epoch
        if self._logger.mode == LoggerMode.EVALUATION:
            self._logger.log_epoch_to_context(epoch=1)

        # Set the model name:
        self._model_name = (
            type(self._objects[self._ObjectKeys.MODEL]).__name__
            if self._model_name is None
            else self._model_name
        )

        # Create the model handler:
        model_handler = PyTorchModelHandler(
            model_name=self._model_name,
            model_path=self._model_path,
            modules_map=self._modules_map,
            custom_objects_map=self._custom_objects_map,
            custom_objects_directory=self._custom_objects_directory,
            model=self._objects[self._ObjectKeys.MODEL],
        )

        # Set the inputs and outputs:
        model_handler.set_inputs(from_sample=self._input_sample)
        model_handler.set_outputs(from_sample=self._output_sample)

        # End the run:
        self._logger.log_run(model_handler=model_handler)

    def on_epoch_end(self, epoch: int):
        """
        Before the given epoch ends, this method will be called to log the dynamic hyperparameters and results of this
        epoch via the stored context.

        :param epoch: The epoch that has just ended.
        """
        super(MLRunLoggingCallback, self).on_epoch_end(epoch=epoch)

        # Create child context to hold the current epoch's results:
        self._logger.log_epoch_to_context(epoch=epoch)

    def on_inference_begin(self, x: Tensor):
        """
        Before the inference of the current batch sample into the model, this method will be called to save an input
        sample - a zeros tensor with the same properties of the 'x' input.

        :param x: The input of the current batch.
        """
        if self._input_sample is None:
            self._input_sample = torch.zeros(size=x.shape, dtype=x.dtype)

    def on_inference_end(self, y_pred: Tensor, y_true: Tensor):
        """
        After the inference of the current batch sample, this method will be called to save an output sample - a zeros
        tensor with the same properties of the 'y_pred' output.

        :param y_pred: The prediction (output) of the model for this batch's input ('x').
        :param y_true: The ground truth value of the current batch.
        """
        if self._output_sample is None:
            self._output_sample = torch.zeros(size=y_pred.shape, dtype=y_pred.dtype)
