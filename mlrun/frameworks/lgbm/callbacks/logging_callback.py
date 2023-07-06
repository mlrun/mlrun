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
from typing import List

from ..._ml_common.loggers import Logger
from ..utils import LGBMTypes
from .callback import Callback, CallbackEnv


class LoggingCallback(Callback):
    """
    A logging callback to collect training data.
    """

    def __init__(
        self,
        dynamic_hyperparameters: List[str] = None,
        static_hyperparameters: List[str] = None,
    ):
        """
        Initialize the logging callback with the given configuration. All the metrics data will be collected but the
        hyperparameters to log must be given. The hyperparameters will be taken from the `params` of the model in each
        iteration.

        :param dynamic_hyperparameters: If needed to track a hyperparameter dynamically (sample it each iteration) it
                                        should be passed here. The parameter expects a list of all the hyperparameters
                                        names to track our of the `params` dictionary.
        :param static_hyperparameters:  If needed to track a hyperparameter one time per run it should be passed here.
                                        The parameter expects a list of all the hyperparameters names to track our of
                                        the `params` dictionary.
        """
        super(LoggingCallback, self).__init__()
        self._logger = Logger()
        self._dynamic_hyperparameters_keys = (
            dynamic_hyperparameters if dynamic_hyperparameters is not None else {}
        )
        self._static_hyperparameters_keys = (
            static_hyperparameters if static_hyperparameters is not None else {}
        )

    @property
    def logger(self) -> Logger:
        """
        Get the logger of the callback. In the logger you may access the collected training data.

        :return: The logger.
        """
        return self._logger

    def __call__(self, env: CallbackEnv):
        """
        Log the iteration that ended and all the results it calculated.

        :param env: A named tuple passed ad the end of each iteration containing the metrics results. For more
                    information check the `Callback` doc string.
        """
        # Log the iteration:
        self._logger.log_iteration()

        # Log the metrics results out of the `evaluation_result_list` field:
        self._log_results(evaluation_result_list=env.evaluation_result_list)

        # Log the hyperparameters out of the `params` field:
        self._log_hyperparameters(parameters=env.params)

    def _log_results(
        self, evaluation_result_list: List[LGBMTypes.EvaluationResultType]
    ):
        """
        Log the callback environment results data into the logger.

        :param evaluation_result_list: The metrics results as provided by the callback environment of LightGBM.
        """
        for evaluation_result in evaluation_result_list:
            # Check what results were given, from `lightgbm.train` or `lightgbm.cv`:
            if len(evaluation_result) == 4:
                # `lightgbm.train` is used:
                self._logger.log_result(
                    validation_set_name=evaluation_result[0],
                    metric_name=evaluation_result[1],
                    result=evaluation_result[2],
                )
            else:
                # `lightgbm.cv` is used, unpack both mean and stdv scores:
                self._logger.log_result(
                    validation_set_name=evaluation_result[0],
                    metric_name=f"{evaluation_result[1]}_mean",
                    result=evaluation_result[2],
                )
                self._logger.log_result(
                    validation_set_name=evaluation_result[0],
                    metric_name=f"{evaluation_result[1]}_stdv",
                    result=evaluation_result[4],
                )

    def _log_hyperparameters(self, parameters: dict):
        """
        Log the callback environment parameters into the logger.

        :param parameters: The parameters as provided by the callback environment of LightGBM.
        """
        for parameter_name, value in parameters.items():
            if parameter_name in self._dynamic_hyperparameters_keys:
                self._logger.log_dynamic_hyperparameter(
                    parameter_name=parameter_name, value=value
                )
                continue
            if parameter_name in self._static_hyperparameters_keys:
                self._logger.log_static_hyperparameter(
                    parameter_name=parameter_name, value=value
                )
