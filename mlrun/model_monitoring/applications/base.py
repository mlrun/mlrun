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

import socket
from abc import ABC, abstractmethod
from collections.abc import Iterator
from datetime import datetime, timedelta
from typing import Any, Optional, Union, cast

import pandas as pd

import mlrun
import mlrun.common.constants as mlrun_constants
import mlrun.common.schemas.model_monitoring.constants as mm_constants
import mlrun.errors
import mlrun.model_monitoring.api as mm_api
import mlrun.model_monitoring.applications.context as mm_context
import mlrun.model_monitoring.applications.results as mm_results
from mlrun.serving.utils import MonitoringApplicationToDict


class ModelMonitoringApplicationBase(MonitoringApplicationToDict, ABC):
    """
    The base class for a model monitoring application.
    Inherit from this class to create a custom model monitoring application.

    For example, :code:`MyApp` below is a simplistic custom application::

        class MyApp(ModelMonitoringApplicationBase):
            def do_tracking(
                self,
                monitoring_context: mm_context.MonitoringApplicationContext,
            ) -> ModelMonitoringApplicationResult:
                monitoring_context.log_artifact(
                    TableArtifact(
                        "sample_df_stats", df=self.dict_to_histogram(sample_df_stats)
                    )
                )
                return ModelMonitoringApplicationResult(
                    name="data_drift_test",
                    value=0.5,
                    kind=mm_constant.ResultKindApp.data_drift,
                    status=mm_constant.ResultStatusApp.detected,
                )
    """

    kind = "monitoring_application"

    def do(
        self, monitoring_context: mm_context.MonitoringApplicationContext
    ) -> tuple[
        list[
            Union[
                mm_results.ModelMonitoringApplicationResult,
                mm_results.ModelMonitoringApplicationMetric,
            ]
        ],
        mm_context.MonitoringApplicationContext,
    ]:
        """
        Process the monitoring event and return application results & metrics.
        Note: this method is internal and should not be called directly or overridden.

        :param monitoring_context:   (MonitoringApplicationContext) The monitoring application context.
        :returns:                    A tuple of:
                                        [0] = list of application results that can be either from type
                                        `ModelMonitoringApplicationResult`
                                        or from type `ModelMonitoringApplicationResult`.
                                        [1] = the original application event, wrapped in `MonitoringApplicationContext`
                                         object
        """
        results = self.do_tracking(monitoring_context=monitoring_context)
        if isinstance(results, dict):
            results = [
                mm_results.ModelMonitoringApplicationMetric(name=key, value=value)
                for key, value in results.items()
            ]
        results = results if isinstance(results, list) else [results]
        return results, monitoring_context

    def _handler(
        self,
        context: "mlrun.MLClientCtx",
        sample_data: Optional[pd.DataFrame] = None,
        reference_data: Optional[pd.DataFrame] = None,
        endpoints: Optional[list[tuple[str, str]]] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        base_period: Optional[int] = None,
    ):
        """
        A custom handler that wraps the application's logic implemented in
        :py:meth:`~mlrun.model_monitoring.applications.ModelMonitoringApplicationBase.do_tracking`
        for an MLRun job.
        This method should not be called directly.
        """
        feature_stats = (
            mm_api.get_sample_set_statistics(reference_data)
            if reference_data is not None
            else None
        )

        def call_do_tracking(event: Optional[dict] = None):
            if event is None:
                event = {}
            monitoring_context = mm_context.MonitoringApplicationContext._from_ml_ctx(
                event=event,
                application_name=self.__class__.__name__,
                context=context,
                sample_df=sample_data,
                feature_stats=feature_stats,
            )
            return self.do_tracking(monitoring_context)

        if endpoints is not None:
            start, end = self._validate_times(start, end, base_period)
            for window_start, window_end in self._window_generator(
                start, end, base_period
            ):
                for endpoint_name, endpoint_id in endpoints:
                    result = call_do_tracking(
                        event={
                            mm_constants.ApplicationEvent.ENDPOINT_NAME: endpoint_name,
                            mm_constants.ApplicationEvent.ENDPOINT_ID: endpoint_id,
                            mm_constants.ApplicationEvent.START_INFER_TIME: window_start,
                            mm_constants.ApplicationEvent.END_INFER_TIME: window_end,
                        }
                    )
                    context.log_result(
                        f"{endpoint_name}_{window_start.isoformat()}_{window_end.isoformat()}",
                        result,
                    )
        else:
            return call_do_tracking()

    @staticmethod
    def _validate_times(
        start: Optional[datetime],
        end: Optional[datetime],
        base_period: Optional[int],
    ) -> tuple[datetime, datetime]:
        if (start is None) or (end is None):
            raise mlrun.errors.MLRunValueError(
                "When `endpoint_names` is provided, you must also pass the start and end times"
            )
        if (base_period is not None) and not (
            isinstance(base_period, int) and base_period > 0
        ):
            raise mlrun.errors.MLRunValueError(
                "`base_period` must be a nonnegative integer - the number of minutes in a monitoring window"
            )
        return start, end

    @staticmethod
    def _window_generator(
        start: datetime, end: datetime, base_period: Optional[int]
    ) -> Iterator[tuple[datetime, datetime]]:
        if base_period is None:
            yield start, end
            return

        window_length = timedelta(minutes=base_period)
        current_start_time = start
        while current_start_time < end:
            current_end_time = min(current_start_time + window_length, end)
            yield current_start_time, current_end_time
            current_start_time = current_end_time

    @classmethod
    def deploy(
        cls,
        func_name: str,
        func_path: Optional[str] = None,
        image: Optional[str] = None,
        handler: Optional[str] = None,
        with_repo: Optional[bool] = False,
        tag: Optional[str] = None,
        requirements: Optional[Union[str, list[str]]] = None,
        requirements_file: str = "",
        **application_kwargs,
    ) -> None:
        """
        Set the application to the current project and deploy it as a Nuclio serving function.
        Required for your model monitoring application to work as a part of the model monitoring framework.

        :param func_name: The name of the function.
        :param func_path: The path of the function, :code:`None` refers to the current Jupyter notebook.

        For the other arguments, refer to
        :py:meth:`~mlrun.projects.MlrunProject.set_model_monitoring_function`.
        """
        project = cast("mlrun.MlrunProject", mlrun.get_current_project())
        function = project.set_model_monitoring_function(
            name=func_name,
            func=func_path,
            application_class=cls.__name__,
            handler=handler,
            image=image,
            with_repo=with_repo,
            requirements=requirements,
            requirements_file=requirements_file,
            tag=tag,
            **application_kwargs,
        )
        function.deploy()

    @classmethod
    def evaluate(
        cls,
        func_path: Optional[str] = None,
        func_name: Optional[str] = None,
        *,
        tag: Optional[str] = None,
        run_local: bool = True,
        sample_data: Optional[pd.DataFrame] = None,
        reference_data: Optional[pd.DataFrame] = None,
        image: Optional[str] = None,
        with_repo: Optional[bool] = False,
        requirements: Optional[Union[str, list[str]]] = None,
        requirements_file: str = "",
        endpoints: Optional[list[tuple[str, str]]] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        base_period: Optional[int] = None,
    ) -> "mlrun.RunObject":
        """
        Call this function to run the application's
        :py:meth:`~mlrun.model_monitoring.applications.ModelMonitoringApplicationBase.do_tracking`
        model monitoring logic as a :py:class:`~mlrun.runtimes.KubejobRuntime`, which is an MLRun function.

        This method has default values for all of its arguments. You should be change them when you want to pass
        data to the application.

        :param func_path:         The path to the function. If ``None``, the current notebook is used.
        :param func_name:         The name of the function. If not ``None``, the class name is used.
        :param tag:               Tag for the function.
        :param run_local:         Whether to run the function locally or remotely.
        :param sample_data:       Pandas data-frame as the current dataset.
                                  When set, it replaces the data read from the model endpoint's offline source.
        :param reference_data:    Pandas data-frame of the reference dataset.
                                  When set, its statistics override the model endpoint's feature statistics.
        :param image:             Docker image to run the job on.
        :param with_repo:         Whether to clone the current repo to the build source.
        :param requirements:      List of Python requirements to be installed in the image.
        :param requirements_file: Path to a Python requirements file to be installed in the image.
        :param endpoints:         A list of tuples of the model endpoint (name, uid) to get the data from.
                                  If provided, you have to provide also the start and end times of the data to analyze.
        :param start:             The start time of the sample data.
        :param end:               The end time of the sample data.
        :param base_period:       The window length in minutes. If ``None``, the whole window from ``start`` to ``end``
                                  is taken. If an integer is specified, the application is run from ``start`` to ``end``
                                  in ``base_period`` length windows, except for the last window that ends at ``end`` and
                                  therefore may be shorter.

        :returns: The output of the
                  :py:meth:`~mlrun.model_monitoring.applications.ModelMonitoringApplicationBase.do_tracking`
                  method with the given parameters and inputs, wrapped in a :py:class:`~mlrun.model.RunObject`.
        """
        project = cast("mlrun.MlrunProject", mlrun.get_current_project())
        class_name = cls.__name__
        job_name = func_name if func_name is not None else class_name
        handler = f"{class_name}::{cls._handler.__name__}"

        job = cast(
            mlrun.runtimes.KubejobRuntime,
            project.set_function(
                func=func_path,
                name=job_name,
                kind=mlrun.runtimes.KubejobRuntime.kind,
                handler=handler,
                tag=tag,
                image=image,
                with_repo=with_repo,
                requirements=requirements,
                requirements_file=requirements_file,
            ),
        )

        params: dict[str, Union[list[tuple[str, str]], datetime, int, None]] = {}
        if endpoints:
            start, end = cls._validate_times(start, end, base_period)
            params["endpoints"] = endpoints
            params["start"] = start
            params["end"] = end
            params["base_period"] = base_period
        elif start or end or base_period:
            raise mlrun.errors.MLRunValueError(
                "Custom start and end times or base_period are supported only with endpoints data"
            )

        inputs: dict[str, str] = {}
        for data, identifier in [
            (sample_data, "sample_data"),
            (reference_data, "reference_data"),
        ]:
            if data is not None:
                key = f"{job_name}_{identifier}"
                inputs[identifier] = project.log_dataset(
                    key,
                    data,
                    labels={
                        mlrun_constants.MLRunInternalLabels.runner_pod: socket.gethostname(),
                        mlrun_constants.MLRunInternalLabels.producer_type: "model-monitoring-job",
                        mlrun_constants.MLRunInternalLabels.app_name: class_name,
                    },
                ).uri

        run_result = job.run(local=run_local, params=params, inputs=inputs)
        return run_result

    @abstractmethod
    def do_tracking(
        self,
        monitoring_context: mm_context.MonitoringApplicationContext,
    ) -> Union[
        mm_results.ModelMonitoringApplicationResult,
        list[
            Union[
                mm_results.ModelMonitoringApplicationResult,
                mm_results.ModelMonitoringApplicationMetric,
            ]
        ],
        dict[str, Any],
    ]:
        """
        Implement this method with your custom monitoring logic.

        :param monitoring_context:      (MonitoringApplicationContext) The monitoring context to process.

        :returns:                       (ModelMonitoringApplicationResult) or
                                        (list[Union[ModelMonitoringApplicationResult,
                                        ModelMonitoringApplicationMetric]])
                                        or dict that contains the application metrics only (in this case the name of
                                        each metric name is the key and the metric value is the corresponding value).
        """
        raise NotImplementedError
