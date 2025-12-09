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

import json
import socket
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, Optional, Union, cast

import pandas as pd

import mlrun
import mlrun.common.constants as mlrun_constants
import mlrun.common.helpers
import mlrun.common.schemas.model_monitoring.constants as mm_constants
import mlrun.common.types
import mlrun.datastore.datastore_profile as ds_profile
import mlrun.errors
import mlrun.model_monitoring.api as mm_api
import mlrun.model_monitoring.applications.context as mm_context
import mlrun.model_monitoring.applications.results as mm_results
import mlrun.model_monitoring.db._schedules as mm_schedules
import mlrun.model_monitoring.helpers as mm_helpers
import mlrun.utils
from mlrun.serving.utils import MonitoringApplicationToDict
from mlrun.utils import logger


class ExistingDataHandling(mlrun.common.types.StrEnum):
    fail_on_overlap = "fail_on_overlap"
    skip_overlap = "skip_overlap"
    delete_all = "delete_all"


def _serialize_context_and_result(
    *,
    context: mm_context.MonitoringApplicationContext,
    result: Union[
        mm_results.ModelMonitoringApplicationResult,
        mm_results.ModelMonitoringApplicationMetric,
        mm_results._ModelMonitoringApplicationStats,
    ],
) -> dict[mm_constants.WriterEvent, str]:
    """
    Serialize the returned result from a model monitoring application and its context
    for the writer.
    """
    writer_event = {
        mm_constants.WriterEvent.ENDPOINT_NAME: context.endpoint_name,
        mm_constants.WriterEvent.APPLICATION_NAME: context.application_name,
        mm_constants.WriterEvent.ENDPOINT_ID: context.endpoint_id,
        mm_constants.WriterEvent.START_INFER_TIME: context.start_infer_time.isoformat(
            sep=" ", timespec="microseconds"
        ),
        mm_constants.WriterEvent.END_INFER_TIME: context.end_infer_time.isoformat(
            sep=" ", timespec="microseconds"
        ),
    }

    if isinstance(result, mm_results.ModelMonitoringApplicationResult):
        writer_event[mm_constants.WriterEvent.EVENT_KIND] = (
            mm_constants.WriterEventKind.RESULT
        )
    elif isinstance(result, mm_results._ModelMonitoringApplicationStats):
        writer_event[mm_constants.WriterEvent.EVENT_KIND] = (
            mm_constants.WriterEventKind.STATS
        )
    else:
        writer_event[mm_constants.WriterEvent.EVENT_KIND] = (
            mm_constants.WriterEventKind.METRIC
        )
    writer_event[mm_constants.WriterEvent.DATA] = json.dumps(result.to_dict())

    return writer_event


class ModelMonitoringApplicationBase(MonitoringApplicationToDict, ABC):
    """
    The base class for a model monitoring application.
    Inherit from this class to create a custom model monitoring application.

    For example, :code:`MyApp` below is a simplistic custom application::

        from mlrun.common.schemas.model_monitoring.constants import (
            ResultKindApp,
            ResultStatusApp,
        )
        from mlrun.model_monitoring.applications import (
            ModelMonitoringApplicationBase,
            ModelMonitoringApplicationResult,
            MonitoringApplicationContext,
        )


        class MyApp(ModelMonitoringApplicationBase):
            def do_tracking(
                self, monitoring_context: MonitoringApplicationContext
            ) -> ModelMonitoringApplicationResult:
                monitoring_context.logger.info(
                    "Running application",
                    application_name=monitoring_context.application_name,
                )
                return ModelMonitoringApplicationResult(
                    name="data_drift_test",
                    value=0.5,
                    kind=ResultKindApp.data_drift,
                    status=ResultStatusApp.detected,
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

    @staticmethod
    def _flatten_data_result(
        result: Union[
            list[mm_results._ModelMonitoringApplicationDataRes],
            mm_results._ModelMonitoringApplicationDataRes,
        ],
    ) -> Union[list[dict], dict]:
        """Flatten result/metric objects to dictionaries"""
        if isinstance(result, mm_results._ModelMonitoringApplicationDataRes):
            return result.to_dict()
        if isinstance(result, list):
            return [
                element.to_dict()
                if isinstance(element, mm_results._ModelMonitoringApplicationDataRes)
                else element
                for element in result
            ]
        return result

    @staticmethod
    def _check_writer_is_up(project: "mlrun.MlrunProject") -> None:
        try:
            project.get_function(
                mm_constants.MonitoringFunctionNames.WRITER, ignore_cache=True
            )
        except mlrun.errors.MLRunNotFoundError:
            raise mlrun.errors.MLRunValueError(
                "Writing outputs to the databases is blocked as the model monitoring infrastructure is disabled.\n"
                "To unblock, enable model monitoring with `project.enable_model_monitoring()`."
            )

    @classmethod
    @contextmanager
    def _push_to_writer(
        cls,
        *,
        write_output: bool,
        application_name: str,
        artifact_path: str,
        stream_profile: Optional[ds_profile.DatastoreProfile],
        project: "mlrun.MlrunProject",
    ) -> Iterator[
        tuple[
            dict[str, list[tuple]],
            Optional[mm_schedules.ModelMonitoringSchedulesFileApplication],
        ]
    ]:
        endpoints_output: dict[
            str,
            list[
                tuple[
                    mm_context.MonitoringApplicationContext,
                    Union[
                        mm_results.ModelMonitoringApplicationResult,
                        mm_results.ModelMonitoringApplicationMetric,
                        list[
                            Union[
                                mm_results.ModelMonitoringApplicationResult,
                                mm_results.ModelMonitoringApplicationMetric,
                                mm_results._ModelMonitoringApplicationStats,
                            ]
                        ],
                    ],
                ]
            ],
        ] = defaultdict(list)
        application_schedules = nullcontext()
        if write_output:
            cls._check_writer_is_up(project)
            application_schedules = (
                mm_schedules.ModelMonitoringSchedulesFileApplication(
                    artifact_path, application=application_name
                )
            )
        try:
            yield endpoints_output, application_schedules.__enter__()
        finally:
            if write_output and any(endpoints_output.values()):
                logger.debug(
                    "Pushing model monitoring application job data to the writer stream",
                    passed_stream_profile=str(stream_profile),
                )
                project_name = (
                    mlrun.mlconf.active_project or mlrun.get_current_project().name
                )
                writer_stream = mm_helpers.get_output_stream(
                    project=project_name,
                    function_name=mm_constants.MonitoringFunctionNames.WRITER,
                    profile=stream_profile,
                )
                for endpoint_id, outputs in endpoints_output.items():
                    writer_events = []
                    for ctx, res in outputs:
                        if isinstance(res, list):
                            writer_events.extend(
                                _serialize_context_and_result(
                                    context=ctx, result=sub_res
                                )
                                for sub_res in res
                            )
                        else:
                            writer_events.append(
                                _serialize_context_and_result(context=ctx, result=res)
                            )
                    writer_stream.push(
                        writer_events,
                        partition_key=endpoint_id,
                    )
                logger.debug(
                    "Pushed the data to all the relevant model endpoints successfully",
                    endpoints_output=endpoints_output,
                )

                logger.debug(
                    "Saving the application schedules",
                    application_name=application_name,
                )
                application_schedules.__exit__(None, None, None)

    @classmethod
    def _get_application_name(cls, context: "mlrun.MLClientCtx") -> str:
        """Get the application name from the context via the function URI"""
        _, application_name, _, _ = mlrun.common.helpers.parse_versioned_object_uri(
            context.to_dict().get("spec", {}).get("function", "")
        )
        return application_name

    def _handler(
        self,
        context: "mlrun.MLClientCtx",
        sample_data: Optional[pd.DataFrame] = None,
        reference_data: Optional[pd.DataFrame] = None,
        endpoints: Union[
            list[tuple[str, str]], list[list[str]], list[str], Literal["all"], None
        ] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        base_period: Optional[int] = None,
        write_output: bool = False,
        existing_data_handling: ExistingDataHandling = ExistingDataHandling.fail_on_overlap,
        stream_profile: Optional[ds_profile.DatastoreProfile] = None,
    ):
        """
        A custom handler that wraps the application's logic implemented in
        :py:meth:`~mlrun.model_monitoring.applications.ModelMonitoringApplicationBase.do_tracking`
        for an MLRun job.
        This method should not be called directly.
        """
        project = context.get_project_object()
        if not project:
            raise mlrun.errors.MLRunValueError("Could not load project from context")

        if write_output and (
            not endpoints or sample_data is not None or reference_data is not None
        ):
            raise mlrun.errors.MLRunValueError(
                "Writing the results of an application to the TSDB is possible only when "
                "working with endpoints, without any custom data-frame input"
            )

        application_name = self._get_application_name(context)

        feature_stats = (
            mm_api.get_sample_set_statistics(reference_data)
            if reference_data is not None
            else None
        )

        with self._push_to_writer(
            write_output=write_output,
            stream_profile=stream_profile,
            application_name=application_name,
            artifact_path=context.artifact_path,
            project=project,
        ) as (endpoints_output, application_schedules):

            def call_do_tracking(
                monitoring_context: mm_context.MonitoringApplicationContext,
            ):
                nonlocal endpoints_output

                result = self.do_tracking(monitoring_context)
                endpoints_output[monitoring_context.endpoint_id].append(
                    (monitoring_context, result)
                )
                return result

            if endpoints is not None:
                resolved_endpoints = self._normalize_and_validate_endpoints(
                    project=project, endpoints=endpoints
                )
                if (
                    write_output
                    and existing_data_handling == ExistingDataHandling.delete_all
                ):
                    endpoint_ids = [
                        endpoint_id for _, endpoint_id in resolved_endpoints
                    ]
                    context.logger.info(
                        "Deleting all the application data before running the application",
                        application_name=application_name,
                        endpoint_ids=endpoint_ids,
                    )
                    self._delete_application_data(
                        project_name=project.name,
                        application_name=application_name,
                        endpoint_ids=endpoint_ids,
                        application_schedules=application_schedules,
                    )
                for endpoint_name, endpoint_id in resolved_endpoints:
                    for monitoring_ctx in self._window_generator(
                        start=start,
                        end=end,
                        base_period=base_period,
                        application_schedules=application_schedules,
                        endpoint_id=endpoint_id,
                        endpoint_name=endpoint_name,
                        application_name=application_name,
                        existing_data_handling=existing_data_handling,
                        sample_data=sample_data,
                        context=context,
                        project=project,
                    ):
                        result = call_do_tracking(monitoring_ctx)
                        result_key = (
                            f"{endpoint_name}-{endpoint_id}_{monitoring_ctx.start_infer_time.isoformat()}_{monitoring_ctx.end_infer_time.isoformat()}"
                            if monitoring_ctx.start_infer_time
                            and monitoring_ctx.end_infer_time
                            else f"{endpoint_name}-{endpoint_id}"
                        )

                        context.log_result(
                            result_key, self._flatten_data_result(result)
                        )
                # Check if no result was produced for any endpoint (e.g., due to no data in all windows)
                if not any(endpoints_output.values()):
                    context.logger.warning(
                        "No data was found for any of the specified endpoints. "
                        "No results were produced",
                        application_name=application_name,
                        endpoints=endpoints,
                        start=start,
                        end=end,
                    )
            else:
                result = call_do_tracking(
                    mm_context.MonitoringApplicationContext._from_ml_ctx(
                        context=context,
                        project=project,
                        application_name=application_name,
                        event={},
                        sample_df=sample_data,
                        feature_stats=feature_stats,
                    )
                )
                return self._flatten_data_result(result)

    @staticmethod
    def _check_endpoints_first_request(
        endpoints: list[mlrun.common.schemas.ModelEndpoint],
    ) -> None:
        """Make sure that all the endpoints have had at least one request"""
        endpoints_no_requests = [
            (endpoint.metadata.name, endpoint.metadata.uid)
            for endpoint in endpoints
            if not endpoint.status.first_request
        ]
        if endpoints_no_requests:
            raise mlrun.errors.MLRunValueError(
                "The following model endpoints have not had any requests yet and "
                "have no data, cannot run the model monitoring application on them: "
                f"{endpoints_no_requests}"
            )

    @classmethod
    def _normalize_and_validate_endpoints(
        cls,
        project: "mlrun.MlrunProject",
        endpoints: Union[
            list[tuple[str, str]], list[list[str]], list[str], Literal["all"]
        ],
    ) -> list[tuple[str, str]]:
        if isinstance(endpoints, list):
            if all(
                isinstance(endpoint, (tuple, list)) and len(endpoint) == 2
                for endpoint in endpoints
            ):
                # A list of [(name, uid), ...] / [[name, uid], ...] tuples/lists
                endpoint_uids_to_names = {
                    endpoint[1]: endpoint[0] for endpoint in endpoints
                }
                endpoints_list = project.list_model_endpoints(
                    uids=list(endpoint_uids_to_names.keys()), latest_only=True
                ).endpoints

                # Check for missing endpoint uids or name/uid mismatches
                for endpoint in endpoints_list:
                    if (
                        endpoint_uids_to_names[cast(str, endpoint.metadata.uid)]
                        != endpoint.metadata.name
                    ):
                        raise mlrun.errors.MLRunNotFoundError(
                            "Could not find model endpoint with name "
                            f"'{endpoint_uids_to_names[cast(str, endpoint.metadata.uid)]}' "
                            f"and uid '{endpoint.metadata.uid}'"
                        )
                missing = set(endpoint_uids_to_names.keys()) - {
                    cast(str, endpoint.metadata.uid) for endpoint in endpoints_list
                }
                if missing:
                    raise mlrun.errors.MLRunNotFoundError(
                        "Could not find model endpoints with the following uids: "
                        f"{missing}"
                    )

            elif all(isinstance(endpoint, str) for endpoint in endpoints):
                # A list of [name, ...] strings
                endpoint_names = cast(list[str], endpoints)
                endpoints_list = project.list_model_endpoints(
                    names=endpoint_names, latest_only=True
                ).endpoints

                # Check for missing endpoint names
                missing = set(endpoints) - {
                    endpoint.metadata.name for endpoint in endpoints_list
                }
                if missing:
                    logger.warning(
                        "Could not list all the required endpoints",
                        missing_endpoints=missing,
                        endpoints_list=endpoints_list,
                    )
            else:
                raise mlrun.errors.MLRunValueError(
                    "Could not resolve the following list as a list of endpoints:\n"
                    f"{endpoints}\n"
                    "The list must be either a list of (name, uid) tuples/lists or a list of names."
                )
        elif endpoints == "all":
            endpoints_list = project.list_model_endpoints(latest_only=True).endpoints
        elif isinstance(endpoints, str):
            raise mlrun.errors.MLRunValueError(
                'A string input for `endpoints` can only be "all" for all the model endpoints in '
                "the project. If you want to select a single model endpoint with the given name, "
                f'use a list: `endpoints=["{endpoints}"]`.'
            )
        else:
            raise mlrun.errors.MLRunValueError(
                "Could not resolve the `endpoints` parameter. The parameter must be either:\n"
                "- a list of (name, uid) tuples/lists\n"
                "- a list of names\n"
                '- the string "all" for all the model endpoints in the project.'
            )

        if not endpoints_list:
            raise mlrun.errors.MLRunNotFoundError(
                f"Did not find any model endpoints {endpoints=}"
            )

        cls._check_endpoints_first_request(endpoints_list)

        return [
            (endpoint.metadata.name, cast(str, endpoint.metadata.uid))
            for endpoint in endpoints_list
        ]

    @staticmethod
    def _validate_and_get_window_length(
        *, base_period: int, start_dt: datetime, end_dt: datetime
    ) -> timedelta:
        if not isinstance(base_period, int) or base_period <= 0:
            raise mlrun.errors.MLRunValueError(
                "`base_period` must be a nonnegative integer - the number of minutes in a monitoring window"
            )

        window_length = timedelta(minutes=base_period)

        full_interval_length = end_dt - start_dt
        remainder = full_interval_length % window_length
        if remainder:
            if full_interval_length < window_length:
                extra_msg = (
                    "The `base_period` is longer than the difference between `end` and `start`: "
                    f"{full_interval_length}. Consider not specifying `base_period`."
                )
            else:
                extra_msg = (
                    f"Consider changing the `end` time to `end`={end_dt - remainder}"
                )
            raise mlrun.errors.MLRunValueError(
                "The difference between `end` and `start` must be a multiple of `base_period`: "
                f"`base_period`={window_length}, `start`={start_dt}, `end`={end_dt}. "
                f"{extra_msg}"
            )
        return window_length

    @staticmethod
    def _validate_monotonically_increasing_data(
        *,
        application_schedules: Optional[
            mm_schedules.ModelMonitoringSchedulesFileApplication
        ],
        endpoint_id: str,
        start_dt: datetime,
        end_dt: datetime,
        base_period: Optional[int],
        application_name: str,
        existing_data_handling: ExistingDataHandling,
    ) -> datetime:
        """Make sure that the (app, endpoint) pair doesn't write output before the last analyzed window"""
        if application_schedules:
            last_analyzed = application_schedules.get_endpoint_last_analyzed(
                endpoint_id
            )
            if last_analyzed:
                if start_dt < last_analyzed:
                    if existing_data_handling == ExistingDataHandling.skip_overlap:
                        if last_analyzed < end_dt and base_period is None:
                            logger.warn(
                                "Setting the start time to last_analyzed since the original start time precedes "
                                "last_analyzed",
                                original_start=start_dt,
                                new_start=last_analyzed,
                                application_name=application_name,
                                endpoint_id=endpoint_id,
                            )
                            start_dt = last_analyzed
                        else:
                            raise mlrun.errors.MLRunValueError(
                                "The start time for the application and endpoint precedes the last analyzed time: "
                                f"start_dt='{start_dt}', last_analyzed='{last_analyzed}', {application_name=}, "
                                f"{endpoint_id=}. "
                                "Writing data out of order is not supported, and the start time could not be "
                                "dynamically reset, as last_analyzed is later than the given end time or that "
                                f"base_period was specified (end_dt='{end_dt}', {base_period=})."
                            )
                    else:
                        raise mlrun.errors.MLRunValueError(
                            "The start time for the application and endpoint precedes the last analyzed time: "
                            f"start_dt='{start_dt}', last_analyzed='{last_analyzed}', {application_name=}, "
                            f"{endpoint_id=}. "
                            "Writing data out of order is not supported. You should change the start time to "
                            f"'{last_analyzed}' or later."
                        )
            else:
                logger.debug(
                    "The application is running on the endpoint for the first time",
                    endpoint_id=endpoint_id,
                    start_dt=start_dt,
                    application_name=application_name,
                )
        return start_dt

    @staticmethod
    def _delete_application_data(
        project_name: str,
        application_name: str,
        endpoint_ids: list[str],
        application_schedules: Optional[
            mm_schedules.ModelMonitoringSchedulesFileApplication
        ],
    ) -> None:
        mlrun.get_run_db().delete_model_monitoring_metrics(
            project=project_name,
            application_name=application_name,
            endpoint_ids=endpoint_ids,
        )
        if application_schedules:
            application_schedules.delete_endpoints_last_analyzed(
                endpoint_uids=endpoint_ids
            )

    @classmethod
    def _window_generator(
        cls,
        *,
        start: Optional[str],
        end: Optional[str],
        base_period: Optional[int],
        application_schedules: Optional[
            mm_schedules.ModelMonitoringSchedulesFileApplication
        ],
        endpoint_name: str,
        endpoint_id: str,
        application_name: str,
        existing_data_handling: ExistingDataHandling,
        context: "mlrun.MLClientCtx",
        project: "mlrun.MlrunProject",
        sample_data: Optional[pd.DataFrame],
    ) -> Iterator[mm_context.MonitoringApplicationContext]:
        def yield_monitoring_ctx(
            window_start: Optional[datetime], window_end: Optional[datetime]
        ) -> Iterator[mm_context.MonitoringApplicationContext]:
            ctx = mm_context.MonitoringApplicationContext._from_ml_ctx(
                event={
                    mm_constants.ApplicationEvent.ENDPOINT_NAME: endpoint_name,
                    mm_constants.ApplicationEvent.ENDPOINT_ID: endpoint_id,
                    mm_constants.ApplicationEvent.START_INFER_TIME: window_start,
                    mm_constants.ApplicationEvent.END_INFER_TIME: window_end,
                },
                application_name=application_name,
                context=context,
                project=project,
                sample_df=sample_data,
            )

            if ctx.sample_df.empty:
                # The current sample is empty
                context.logger.debug(
                    "No sample data available for tracking",
                    application_name=application_name,
                    endpoint_id=ctx.endpoint_id,
                    start_time=ctx.start_infer_time,
                    end_time=ctx.end_infer_time,
                )
                return

            yield ctx

            if application_schedules and window_end:
                application_schedules.update_endpoint_last_analyzed(
                    endpoint_uid=endpoint_id, last_analyzed=window_end
                )

        if start is None or end is None:
            # A single window based on the `sample_data` input - see `_handler`.
            yield from yield_monitoring_ctx(None, None)
            return

        start_dt = datetime.fromisoformat(start)
        end_dt = datetime.fromisoformat(end)

        # If `start_dt` and `end_dt` do not include time zone information - change them to UTC
        if (start_dt.tzinfo is None) and (end_dt.tzinfo is None):
            start_dt = start_dt.replace(tzinfo=timezone.utc)
            end_dt = end_dt.replace(tzinfo=timezone.utc)
        elif (start_dt.tzinfo is None) or (end_dt.tzinfo is None):
            raise mlrun.errors.MLRunValueError(
                "The start and end times must either both include time zone information or both be naive (no time "
                f"zone). Asserting the above failed, aborting the evaluate request: start={start}, end={end}."
            )

        if existing_data_handling != ExistingDataHandling.delete_all:
            start_dt = cls._validate_monotonically_increasing_data(
                application_schedules=application_schedules,
                endpoint_id=endpoint_id,
                start_dt=start_dt,
                end_dt=end_dt,
                base_period=base_period,
                application_name=application_name,
                existing_data_handling=existing_data_handling,
            )

        if base_period is None:
            yield from yield_monitoring_ctx(start_dt, end_dt)
            return

        window_length = cls._validate_and_get_window_length(
            base_period=base_period, start_dt=start_dt, end_dt=end_dt
        )

        current_start_time = start_dt
        while current_start_time < end_dt:
            current_end_time = min(current_start_time + window_length, end_dt)
            yield from yield_monitoring_ctx(current_start_time, current_end_time)
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
    def get_job_handler(cls, handler_to_class: str) -> str:
        """
        A helper function to get the handler to the application job ``_handler``.

        :param handler_to_class: The handler to the application class, e.g. ``my_package.sub_module1.MonitoringApp1``.
        :returns:                The handler to the job of the application class.
        """
        return f"{handler_to_class}::{cls._handler.__name__}"

    @classmethod
    def _determine_job_name(
        cls,
        *,
        func_name: Optional[str],
        class_handler: Optional[str],
        handler_to_class: str,
    ) -> str:
        """
        Determine the batch app's job name. This name is used also as the application name,
        which is retrieved in `_get_application_name`.
        """
        if func_name:
            job_name = func_name
        else:
            if not class_handler:
                class_name = cls.__name__
            else:
                class_name = handler_to_class.split(".")[-1].split("::")[0]

            job_name = mlrun.utils.normalize_name(class_name)

        if not mm_constants.APP_NAME_REGEX.fullmatch(job_name):
            raise mlrun.errors.MLRunValueError(
                "The function name does not comply with the required pattern "
                f"`{mm_constants.APP_NAME_REGEX.pattern}`. "
                "Please choose another `func_name`."
            )
        job_name, was_renamed, suffix = mlrun.utils.helpers.ensure_batch_job_suffix(
            job_name
        )
        if was_renamed:
            mlrun.utils.logger.info(
                f'Changing function name - adding `"{suffix}"` suffix',
                func_name=job_name,
            )

        return job_name

    @classmethod
    def to_job(
        cls,
        *,
        class_handler: Optional[str] = None,
        func_path: Optional[str] = None,
        func_name: Optional[str] = None,
        tag: Optional[str] = None,
        image: Optional[str] = None,
        with_repo: Optional[bool] = False,
        requirements: Optional[Union[str, list[str]]] = None,
        requirements_file: str = "",
        project: Optional["mlrun.MlrunProject"] = None,
    ) -> mlrun.runtimes.KubejobRuntime:
        """
        Get the application's :py:meth:`~mlrun.model_monitoring.applications.ModelMonitoringApplicationBase.do_tracking`
        model monitoring logic as a :py:class:`~mlrun.runtimes.KubejobRuntime`.

        The returned job can be run as any MLRun job with the relevant inputs and params to your application:

        .. code-block:: python

            job = ModelMonitoringApplicationBase.to_job(
                class_handler="package.module.AppClass"
            )
            job.run(inputs={}, params={}, local=False)  # Add the relevant inputs and params

        Optional inputs:

        * ``sample_data``, ``pd.DataFrame``
        * ``reference_data``, ``pd.DataFrame``

        Optional params:

        * ``endpoints``, ``list[tuple[str, str]]``
        * ``start``, ``datetime``
        * ``end``, ``datetime``
        * ``base_period``, ``int``
        * ``write_output``, ``bool``
        * ``existing_data_handling``, ``str``
        * ``_init_args``, ``dict`` - the arguments for the application class constructor
          (equivalent to ``class_arguments``)

        See :py:meth:`~ModelMonitoringApplicationBase.evaluate` for more details
        about these inputs and params.

        For Git sources, add the source archive to the returned job and change the handler:

        .. code-block:: python

            handler = ModelMonitoringApplicationBase.get_job_handler("module.AppClass")
            job.with_source_archive(
                "git://github.com/owner/repo.git#branch-category/specific-task",
                workdir="path/to/application/folder",
                handler=handler,
            )

        :param class_handler:     The handler to the class, e.g. ``path.to.module::MonitoringApplication``,
                                  useful when using Git sources or code from images.
                                  If ``None``, the current class, deriving from
                                  :py:class:`~mlrun.model_monitoring.applications.ModelMonitoringApplicationBase`,
                                  is used.
        :param func_path:         The path to the function. If ``None``, the current notebook is used.
        :param func_name:         The name of the function. If ``None``, the normalized class name is used
                                  (:py:meth:`mlrun.utils.helpers.normalize_name`).
                                  A ``"-batch"`` suffix is guaranteed to be added if not already there.
                                  The function name is also used as the application name to use for the results.
        :param tag:               Tag for the function.
        :param image:             Docker image to run the job on (when running remotely).
        :param with_repo:         Whether to clone the current repo to the build source.
        :param requirements:      List of Python requirements to be installed in the image.
        :param requirements_file: Path to a Python requirements file to be installed in the image.
        :param project:           The current project to set the function to. If not set, the current project is used.

        :returns: The :py:class:`~mlrun.runtimes.KubejobRuntime` job that wraps the model monitoring application's
                  logic.
        """
        project = project or cast("mlrun.MlrunProject", mlrun.get_current_project())

        if not class_handler and cls == ModelMonitoringApplicationBase:
            raise ValueError(
                "You must provide a handler to the model monitoring application class"
            )

        handler_to_class = class_handler or cls.__name__
        handler = cls.get_job_handler(handler_to_class)

        job_name = cls._determine_job_name(
            func_name=func_name,
            class_handler=class_handler,
            handler_to_class=handler_to_class,
        )

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
        return job

    @classmethod
    def evaluate(
        cls,
        func_path: Optional[str] = None,
        func_name: Optional[str] = None,
        *,
        tag: Optional[str] = None,
        run_local: bool = True,
        auto_build: bool = True,
        sample_data: Optional[Union[pd.DataFrame, str]] = None,
        reference_data: Optional[Union[pd.DataFrame, str]] = None,
        image: Optional[str] = None,
        with_repo: Optional[bool] = False,
        class_handler: Optional[str] = None,
        class_arguments: Optional[dict[str, Any]] = None,
        requirements: Optional[Union[str, list[str]]] = None,
        requirements_file: str = "",
        endpoints: Union[list[tuple[str, str]], list[str], Literal["all"], None] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        base_period: Optional[int] = None,
        write_output: bool = False,
        existing_data_handling: ExistingDataHandling = ExistingDataHandling.fail_on_overlap,
        stream_profile: Optional[ds_profile.DatastoreProfile] = None,
    ) -> "mlrun.RunObject":
        """
        Call this function to run the application's
        :py:meth:`~mlrun.model_monitoring.applications.ModelMonitoringApplicationBase.do_tracking`
        model monitoring logic as a :py:class:`~mlrun.runtimes.KubejobRuntime`, which is an MLRun function.

        This function has default values for all of its arguments. You should change them when you want to pass
        data to the application.

        :param func_path:         The path to the function. If ``None``, the current notebook is used.
        :param func_name:         The name of the function. If ``None``, the normalized class name is used
                                  (:py:meth:`mlrun.utils.helpers.normalize_name`).
                                  A ``"-batch"`` suffix is guaranteed to be added if not already there.
                                  The function name is also used as the application name to use for the results.
        :param tag:               Tag for the function.
        :param run_local:         Whether to run the function locally or remotely.
        :param auto_build:        Whether to auto build the function.
        :param sample_data:       Pandas data-frame or :py:class:`~mlrun.artifacts.dataset.DatasetArtifact` URI as
                                  the current dataset.
                                  When set, it replaces the data read from the model endpoint's offline source.
        :param reference_data:    Pandas data-frame or :py:class:`~mlrun.artifacts.dataset.DatasetArtifact` URI as
                                  the reference dataset.
                                  When set, its statistics override the model endpoint's feature statistics.
                                  You do not need to have a model endpoint to use this option.
        :param image:             Docker image to run the job on (when running remotely).
        :param with_repo:         Whether to clone the current repo to the build source.
        :param class_handler:     The relative path to the application class, useful when using Git sources or code
                                  from images.
        :param class_arguments:   The arguments for the application class constructor. These are passed to the
                                  class ``__init__``. The values must be JSON-serializable.
        :param requirements:      List of Python requirements to be installed in the image.
        :param requirements_file: Path to a Python requirements file to be installed in the image.
        :param endpoints:         The model endpoints to get the data from. The options are:

                                  - a list of tuples of the model endpoints ``[(name, uid), ...]``
                                  - a list of model endpoint names ``[name, ...]``
                                  - ``"all"`` for all the project's model endpoints

                                  Note: a model endpoint name retrieves all the active model endpoints using this
                                  name, which may be more than one per name when the same name is used across
                                  multiple serving functions.

                                  If provided, and ``sample_data`` is not ``None``, you have to provide also the
                                  ``start`` and ``end`` times of the data to analyze from the model endpoints.
        :param start:             The start time of the endpoint's data, not included.
                                  If you want the model endpoint's data at ``start`` included, you need to subtract a
                                  small ``datetime.timedelta`` from it.
                                  Make sure to include the time zone when constructing ``datetime.datetime`` objects
                                  manually. When both ``start`` and ``end`` times do not include a time zone, they will
                                  be treated as UTC.
        :param end:               The end time of the endpoint's data, included.
                                  Please note: when ``start`` and ``end`` are set, they create a left-open time interval
                                  ("window") :math:`(\\operatorname{start}, \\operatorname{end}]` that excludes the
                                  endpoint's data at ``start`` and includes the data at ``end``:
                                  :math:`\\operatorname{start} < t \\leq \\operatorname{end}`, :math:`t` is the time
                                  taken in the window's data.
        :param base_period:       The window length in minutes. If ``None``, the whole window from ``start`` to ``end``
                                  is taken. If an integer is specified, the application is run from ``start`` to ``end``
                                  in ``base_period`` length windows:
                                  :math:`(\\operatorname{start}, \\operatorname{start} + \\operatorname{base\\_period}],
                                  (\\operatorname{start} + \\operatorname{base\\_period},
                                  \\operatorname{start} + 2\\cdot\\operatorname{base\\_period}],
                                  ..., (\\operatorname{start} +
                                  (m - 1)\\cdot\\operatorname{base\\_period}, \\operatorname{end}]`,
                                  where :math:`m` is a positive integer and :math:`\\operatorname{end} =
                                  \\operatorname{start} + m\\cdot\\operatorname{base\\_period}`.
                                  Please note that the difference between ``end`` and ``start`` must be a multiple of
                                  ``base_period``.
        :param write_output:      Whether to write the results and metrics to the time-series DB. Can be ``True`` only
                                  if ``endpoints`` are passed.
                                  Note: the model monitoring infrastructure must be up for the writing to work.
        :param existing_data_handling:
                                  How to handle the existing application data for the model endpoints when writing
                                  new data whose requested ``start`` time precedes the ``end`` time of a previous run
                                  that also wrote to the database. Relevant only when ``write_output=True``.
                                  The options are:

                                  - ``"fail_on_overlap"``: Default. An error is raised.
                                  - ``"skip_overlap"``:  the overlapping data is ignored and the
                                    time window is cut so that it starts at the earliest possible time after ``start``.
                                  - ``"delete_all"``: delete all the data that was written by the application to the
                                    model endpoints, regardless of the time window, and write the new data.

        :param stream_profile:    The stream datastore profile. It should be provided only when running locally and
                                  writing the outputs to the database (i.e., when both ``run_local`` and
                                  ``write_output`` are set to ``True``).
                                  For more details on configuring the stream profile, see
                                  :py:meth:`~mlrun.projects.MlrunProject.set_model_monitoring_credentials`.

        :returns: The output of the
                  :py:meth:`~mlrun.model_monitoring.applications.ModelMonitoringApplicationBase.do_tracking`
                  method with the given parameters and inputs, wrapped in a :py:class:`~mlrun.model.RunObject`.
        """
        project = cast("mlrun.MlrunProject", mlrun.get_current_project())

        job = cls.to_job(
            func_path=func_path,
            func_name=func_name,
            class_handler=class_handler,
            tag=tag,
            image=image,
            with_repo=with_repo,
            requirements=requirements,
            requirements_file=requirements_file,
            project=project,
        )

        params: dict[
            str, Union[list, dict, str, int, None, ds_profile.DatastoreProfile]
        ] = {}
        if endpoints:
            params["endpoints"] = endpoints
            if sample_data is None:
                if start is None or end is None:
                    raise mlrun.errors.MLRunValueError(
                        "`start` and `end` times must be provided when `endpoints` "
                        "is provided without `sample_data`"
                    )
                params["start"] = (
                    start.isoformat() if isinstance(start, datetime) else start
                )
                params["end"] = end.isoformat() if isinstance(end, datetime) else end
                params["base_period"] = base_period
        elif start or end or base_period:
            raise mlrun.errors.MLRunValueError(
                "Custom `start` and `end` times or base_period are supported only with endpoints data"
            )
        elif write_output or stream_profile:
            raise mlrun.errors.MLRunValueError(
                "Writing the application output or passing `stream_profile` are supported only with endpoints data"
            )

        params["write_output"] = write_output
        params["existing_data_handling"] = existing_data_handling
        if stream_profile:
            if not run_local:
                raise mlrun.errors.MLRunValueError(
                    "Passing a `stream_profile` is relevant only when running locally"
                )
            if not write_output:
                raise mlrun.errors.MLRunValueError(
                    "Passing a `stream_profile` is relevant only when writing the outputs"
                )
        params["stream_profile"] = stream_profile

        if class_arguments:
            params["_init_args"] = class_arguments

        inputs: dict[str, str] = {}
        for data, identifier in [
            (sample_data, "sample_data"),
            (reference_data, "reference_data"),
        ]:
            if isinstance(data, str):
                inputs[identifier] = data
            elif data is not None:
                key = f"{job.metadata.name}_{identifier}"
                inputs[identifier] = project.log_dataset(
                    key,
                    data,
                    labels={
                        mlrun_constants.MLRunInternalLabels.runner_pod: socket.gethostname(),
                        mlrun_constants.MLRunInternalLabels.producer_type: "model-monitoring-job",
                        mlrun_constants.MLRunInternalLabels.app_name: func_name
                        or cls.__name__,
                    },
                ).uri

        run_result = job.run(
            local=run_local, auto_build=auto_build, params=params, inputs=inputs
        )
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
