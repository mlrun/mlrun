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

from typing import Optional

import mlrun

from ..._ml_common.loggers import MLRunLogger
from .callback import CallbackEnv
from .logging_callback import LoggingCallback


class MLRunLoggingCallback(LoggingCallback):
    """
    A logging callback to collect training data into MLRun. The logging includes:

    * Per iteration chart artifacts for the metrics results.
    * Per iteration chart artifacts for the dynamic hyperparameters values.
    * Results table of the training including the static hyperparameters, and the last iteration dynamic hyperparameters
      values and metrics results.
    """

    def __init__(
        self,
        context: mlrun.MLClientCtx,
        dynamic_hyperparameters: Optional[list[str]] = None,
        static_hyperparameters: Optional[list[str]] = None,
        logging_frequency: int = 100,
    ):
        """
        Initialize an MLRun logging callback with the given configuration. All the metrics data will be collected but
        the hyperparameters to log must be given. The hyperparameters will be taken from the `params` of the model in
        each iteration.

        :param context:                 MLRun context to log to. The context parameters can be logged as static
                                        hyperparameters.
        :param dynamic_hyperparameters: If needed to track a hyperparameter dynamically (sample it each iteration) it
                                        should be passed here. The parameter expects a list of all the hyperparameters
                                        names to track our of the `params` dictionary.
        :param static_hyperparameters:  If needed to track a hyperparameter one time per run it should be passed here.
                                        The parameter expects a list of all the hyperparameters names to track our of
                                        the `params` dictionary.
        :param logging_frequency:       Per how many iterations to write the logs to MLRun (create the plots and log
                                        them and the results to MLRun). Two low frequency may slow the training time.
                                        Default: 100.
        """
        super().__init__(
            dynamic_hyperparameters=dynamic_hyperparameters,
            static_hyperparameters=static_hyperparameters,
        )

        # Replace the logger with an MLRun logger:
        del self._logger
        self._logger = MLRunLogger(context=context)

        # Store the logging frequency, it will be compared with the iteration received in the `CallbackEnv` tuple.
        self._logging_frequency = logging_frequency

    def __call__(self, env: CallbackEnv):
        """
        Log the iteration that ended and all the results it calculated.

        :param env: A named tuple passed ad the end of each iteration containing the metrics results. For more
                    information check the `Callback` doc string.
        """
        # Log the results and parameters:
        super().__call__(env=env)

        # Produce the artifacts (post iteration stage):
        if env.iteration % self._logging_frequency == 0:
            self._logger.log_iteration_to_context()

    def on_train_begin(self):
        """
        Log the context parameters when training begins.
        """
        self._logger.log_context_parameters()

    def on_train_end(self):
        """
        Log the last iteration training data into MLRun.
        """
        self._logger.log_iteration_to_context()
