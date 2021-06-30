from typing import Callable, Dict, List, Tuple, Union

import mlrun
from mlrun.artifacts import Artifact
from mlrun.frameworks._common.loggers import MLRunLogger, TrackableType
from mlrun.frameworks.pytorch.callbacks.logging_callback import LoggingCallback
from mlrun.frameworks.pytorch.model_handler import PyTorchModelHandler


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
        log_model_labels: Dict[str, TrackableType] = None,
        log_model_parameters: Dict[str, TrackableType] = None,
        log_model_extra_data: Dict[str, Union[TrackableType, Artifact]] = None,
        custom_objects: Dict[Union[str, List[str]], str] = None,
        dynamic_hyperparameters: Dict[
            str, Tuple[str, Union[List[Union[str, int]], Callable[[], TrackableType]]]
        ] = None,
        static_hyperparameters: Dict[
            str, Union[TrackableType, Tuple[str, List[Union[str, int]]]]
        ] = None,
        auto_log: bool = False,
    ):
        """
        Initialize an mlrun logging callback with the given hyperparameters and logging configurations.

        :param context:                 MLRun context to log to. Its parameters will be logged automatically  if
                                        'auto_log' is True.
        :param log_model_labels:        Labels to log with the model.
        :param log_model_parameters:    Parameters to log with the model.
        :param log_model_extra_data:    Extra data to log with the model.
        :param custom_objects:          Custom objects the model is using. Expecting a dictionary with the classes names
                                        to import as keys (if multiple classes needed to be imported from the same py
                                        file a list can be given) and the python file from where to import them as their
                                        values. The model class itself must be specified in order to properly save it
                                        for later being loaded with a handler. For example:
                                        {
                                            "class_name": "/path/to/model.py",
                                            ["layer1", "layer2"]: "/path/to/custom_layers.py"
                                        }
        :param dynamic_hyperparameters: If needed to track a hyperparameter dynamically (sample it each epoch) it should
                                        be passed here. The parameter expects a dictionary where the keys are the
                                        hyperparameter chosen names and the values are tuples of object key and a list
                                        with the key chain. A key chain is a list of keys and indices to know how to
                                        access the needed hyperparameter. If the hyperparameter is not of accessible
                                        from any of the HyperparametersKeys, a custom callable method can be passed in
                                        the tuple instead of the key chain when providing the word
                                        HyperparametersKeys.CUSTOM. For example, to track the 'lr' attribute of
                                        an optimizer and a custom parameter, one should pass:
                                        {
                                            "learning rate": (HyperparametersKeys.OPTIMIZER, ["param_groups", 0, "lr"]),
                                            "custom parameter": (HyperparametersKeys.CUSTOM, get_custom_parameter)
                                        }
        :param static_hyperparameters:  If needed to track a hyperparameter one time per run it should be passed here.
                                        The parameter expects a dictionary where the keys are the
                                        hyperparameter chosen names and the values are the hyperparameter static value
                                        or a tuple of object key and a list with the key chain just like the dynamic
                                        hyperparameter. For example, to track the 'epochs' of an experiment run, one
                                        should pass:
                                        {
                                            "epochs": 7
                                        }
        :param auto_log:                Whether or not to enable auto logging for logging the context parameters and
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

        # Store the custom objects:
        self._custom_objects = custom_objects

    def on_run_end(self):
        """
        Before the run ends, this method will be called to log the model and the run summaries charts.
        """
        model = self._objects[self._ObjectKeys.MODEL]
        self._logger.log_run(
            model_handler=PyTorchModelHandler(
                model_class=type(model),
                custom_objects=self._custom_objects,
                model=model,
            )
        )

    def on_epoch_end(self, epoch: int):
        """
        Before the given epoch ends, this method will be called to log the dynamic hyperparameters and results of this
        epoch via the stored context.

        :param epoch: The epoch that has just ended.
        """
        super(MLRunLoggingCallback, self).on_epoch_end(epoch=epoch)

        # Create child context to hold the current epoch's results:
        self._logger.log_epoch_to_context(epoch=epoch)
