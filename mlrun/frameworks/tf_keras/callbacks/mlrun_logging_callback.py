from typing import Callable, Dict, List, Union

import mlrun
from mlrun.artifacts import Artifact

from ..._dl_common.loggers import LoggerMode, MLRunLogger, TrackableType
from ..model_handler import TFKerasModelHandler
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
        model_name: str = None,
        model_path: str = None,
        modules_map: Union[Dict[str, Union[None, str, List[str]]], str] = None,
        custom_objects_map: Union[Dict[str, Union[str, List[str]]], str] = None,
        custom_objects_directory: str = None,
        model_format: str = TFKerasModelHandler.ModelFormats.SAVED_MODEL,
        save_traces: bool = False,
        input_sample: TFKerasModelHandler.IOSample = None,
        output_sample: TFKerasModelHandler.IOSample = None,
        log_model_labels: Dict[str, TrackableType] = None,
        log_model_parameters: Dict[str, TrackableType] = None,
        log_model_extra_data: Dict[str, Union[TrackableType, Artifact]] = None,
        dynamic_hyperparameters: Dict[
            str, Union[List[Union[str, int]], Callable[[], TrackableType]]
        ] = None,
        static_hyperparameters: Dict[
            str, Union[TrackableType, List[Union[str, int]]]
        ] = None,
        auto_log: bool = False,
    ):
        """
        Initialize an mlrun logging callback with the given hyperparameters and logging configurations.

        :param context:                  MLRun context to log to. Its parameters will be logged automatically  if
                                         'auto_log' is True.
        :param model_name:               The model name to use for storing the model artifact. If not given, the
                                         tf.keras.Model.name will be used.
        :param model_path:               The model's store object path. Mandatory for evaluation (to know which model to
                                         update).
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
        :param model_format:             The format to use for saving and loading the model. Should be passed as a
                                         member of the class 'ModelFormats'. Defaulted to
                                         'ModelFormats.JSON_ARCHITECTURE_H5'.
        :param save_traces:              Whether or not to use functions saving (only available for the 'SavedModel'
                                         format) for loading the model later without the custom objects dictionary. Only
                                         from tensorflow version >= 2.4.0. Using this setting will increase the model
                                         saving size.
        :param input_sample:             Input sample to the model for logging additional data regarding the input ports
                                         of the model. If None, the input sample will be read automatically from the
                                         model.
        :param output_sample:            Output sample of the model for logging additional data regarding the output
                                         ports of the model. If None, the input sample will be read automatically from
                                         the model.
        :param log_model_labels:         Labels to log with the model.
        :param log_model_parameters:     Parameters to log with the model.
        :param log_model_extra_data:     Extra data to log with the model.
        :param dynamic_hyperparameters:  If needed to track a hyperparameter dynamically (sample it each epoch) it
                                         should be passed here. The parameter expects a dictionary where the keys are
                                         the hyperparameter chosen names and the values are a key chain from the model.
                                         A key chain is a list of keys and indices to know how to access the needed
                                         hyperparameter from the model. If the hyperparameter is not of accessible from
                                         the model, a custom callable method can be passed. For example, to track the
                                         'lr' attribute of an optimizer and a custom parameter, one should pass:
                                         {
                                             "learning rate": ["optimizer", "lr"],
                                             "custom_parameter": get_custom_parameter
                                         }
        :param static_hyperparameters:   If needed to track a hyperparameter one time per run it should be passed here.
                                         The parameter expects a dictionary where the keys are the
                                         hyperparameter chosen names and the values are the hyperparameter static value
                                         or a key chain just like the dynamic hyperparameter. For example, to track the
                                         'epochs' of an experiment run, one should pass:
                                         {
                                             "epochs": 7
                                         }
        :param auto_log:                 Whether or not to enable auto logging for logging the context parameters and
                                         trying to track common static and dynamic hyperparameters such as learning
                                         rate.
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

        # Store the additional TFKerasModelHandler parameters for logging the model later:
        self._model_name = model_name
        self._model_path = model_path
        self._modules_map = modules_map
        self._custom_objects_map = custom_objects_map
        self._custom_objects_directory = custom_objects_directory
        self._model_format = model_format
        self._save_traces = save_traces
        self._input_sample = input_sample
        self._output_sample = output_sample

    def on_train_end(self, logs: dict = None):
        """
        Called at the end of training, logging the model and the summaries of this run.

        :param logs: Currently the output of the last call to `on_epoch_end()` is passed to this argument for this
                     method but that may change in the future.
        """
        self._end_run()

    def on_test_end(self, logs: dict = None):
        """
        Called at the end of evaluation or validation. Will be called on each epoch according to the validation
        per epoch configuration. The recent evaluation / validation results will be summarized and logged. If the logger
        is in evaluation mode, the model artifact will be updated.

        :param logs: Currently no data is passed to this argument for this method but that may change in the
                     future.
        """
        super(MLRunLoggingCallback, self).on_test_end(logs=logs)

        # Check if its part of evaluation. If so, end the run:
        if self._logger.mode == LoggerMode.EVALUATION:
            self._logger.log_epoch_to_context(epoch=1)
            self._end_run()

    def on_epoch_end(self, epoch: int, logs: dict = None):
        """
        Called at the end of an epoch, logging the dynamic hyperparameters and results of this epoch via the stored
        context.

        :param epoch: Integer, index of epoch.
        :param logs:  Dict, metric results for this training epoch, and for the validation epoch if validation is
                      performed. Validation result keys are prefixed with `val_`. For training epoch, the values of the
                      `Model`'s metrics are returned. Example : `{'loss': 0.2, 'acc': 0.7}`.
        """
        super(MLRunLoggingCallback, self).on_epoch_end(epoch=epoch)

        # Log the current epoch's results:
        self._logger.log_epoch_to_context(epoch=epoch)

    def _end_run(self):
        """
        End the run, logging the collected artifacts.
        """
        # Set the model name:
        self._model_name = (
            self.model.name if self._model_name is None else self._model_name
        )

        # Create the model handler:
        model_handler = TFKerasModelHandler(
            model_name=self._model_name,
            model_path=self._model_path,
            model=self.model,
            modules_map=self._modules_map,
            custom_objects_map=self._custom_objects_map,
            custom_objects_directory=self._custom_objects_directory,
            model_format=self._model_format,
            save_traces=self._save_traces,
        )

        # Set the inputs information if available:
        if self._input_sample is not None:
            model_handler.set_inputs(from_sample=self._input_sample)
        else:
            model_handler.read_inputs_from_model()

        # Set the outputs information if available:
        if self._output_sample is not None:
            model_handler.set_outputs(from_sample=self._output_sample)
        else:
            model_handler.read_outputs_from_model()

        # Log the model:
        self._logger.log_run(model_handler=model_handler)
