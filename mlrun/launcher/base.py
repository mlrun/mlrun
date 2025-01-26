# Copyright 2023 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import abc
import ast
import copy
import os
import uuid
from typing import Any, Callable, Optional, Union

import mlrun.common.schemas
import mlrun.config
import mlrun.errors
import mlrun.lists
import mlrun.model
import mlrun.runtimes
import mlrun.utils.regex
import mlrun_pipelines.common.ops
from mlrun.utils import logger

run_modes = ["pass"]


class BaseLauncher(abc.ABC):
    """
    Abstract class for managing and running functions in different contexts
    This class is designed to encapsulate the logic of running a function in different contexts
    i.e. running a function locally, remotely or in a server
    Each context will have its own implementation of the abstract methods while the common logic resides in this class
    """

    def __init__(self, **kwargs):
        pass

    @abc.abstractmethod
    def launch(
        self,
        runtime: "mlrun.runtimes.BaseRuntime",
        task: Optional[
            Union["mlrun.run.RunTemplate", "mlrun.run.RunObject", dict]
        ] = None,
        handler: Optional[Union[str, Callable]] = None,
        name: Optional[str] = "",
        project: Optional[str] = "",
        params: Optional[dict] = None,
        inputs: Optional[dict[str, str]] = None,
        out_path: Optional[str] = "",
        workdir: Optional[str] = "",
        artifact_path: Optional[str] = "",
        watch: Optional[bool] = True,
        schedule: Optional[
            Union[str, mlrun.common.schemas.schedule.ScheduleCronTrigger]
        ] = None,
        hyperparams: Optional[dict[str, list]] = None,
        hyper_param_options: Optional[mlrun.model.HyperParamOptions] = None,
        verbose: Optional[bool] = None,
        scrape_metrics: Optional[bool] = None,
        local_code_path: Optional[str] = None,
        auto_build: Optional[bool] = None,
        param_file_secrets: Optional[dict[str, str]] = None,
        notifications: Optional[list[mlrun.model.Notification]] = None,
        returns: Optional[list[Union[str, dict[str, str]]]] = None,
        state_thresholds: Optional[dict[str, int]] = None,
    ) -> "mlrun.run.RunObject":
        """run the function from the server/client[local/remote]"""
        pass

    @abc.abstractmethod
    def enrich_runtime(
        self,
        runtime: "mlrun.runtimes.base.BaseRuntime",
        project_name: Optional[str] = "",
        full: bool = True,
    ):
        pass

    @staticmethod
    @abc.abstractmethod
    def _store_function(
        runtime: "mlrun.runtimes.BaseRuntime", run: "mlrun.run.RunObject"
    ):
        pass

    def save_function(
        self,
        runtime: "mlrun.runtimes.BaseRuntime",
        tag: str = "",
        versioned: bool = False,
        refresh: bool = False,
    ) -> str:
        """
        store the function to the db
        :param runtime:     runtime object
        :param tag:         function tag to store
        :param versioned:   whether we want to version this function object so that it will queryable by its hash key
        :param refresh:     refresh function metadata

        :return:            function uri
        """
        db = runtime._get_db()
        if not db:
            raise mlrun.errors.MLRunPreconditionFailedError(
                "Database connection is not configured"
            )

        if refresh:
            self._refresh_function_metadata(runtime)

        tag = tag or runtime.metadata.tag

        obj = runtime.to_dict()
        logger.debug("Saving function", runtime_name=runtime.metadata.name, tag=tag)
        hash_key = db.store_function(
            obj, runtime.metadata.name, runtime.metadata.project, tag, versioned
        )
        hash_key = hash_key if versioned else None
        return "db://" + runtime._function_uri(hash_key=hash_key, tag=tag)

    @staticmethod
    def prepare_image_for_deploy(runtime: "mlrun.runtimes.BaseRuntime"):
        """Check if the runtime requires to build the image and updates the spec accordingly"""
        pass

    def _validate_runtime(
        self,
        runtime: "mlrun.runtimes.BaseRuntime",
        run: "mlrun.run.RunObject",
    ):
        mlrun.utils.helpers.verify_dict_items_type(
            "Inputs", run.spec.inputs, [str], [str]
        )

        if runtime.spec.mode and runtime.spec.mode not in run_modes:
            raise ValueError(f'run mode can only be {",".join(run_modes)}')

        self._validate_run_params(run.spec.parameters)
        self._validate_output_path(runtime, run)

    @staticmethod
    def _validate_output_path(
        runtime: "mlrun.runtimes.BaseRuntime",
        run: "mlrun.run.RunObject",
    ):
        if not run.spec.output_path or "://" not in run.spec.output_path:
            message = ""
            if not os.path.isabs(run.spec.output_path):
                message = (
                    "Artifact/output path is not defined or is local and relative,"
                    " artifacts will not be visible in the UI"
                )
                if mlrun.runtimes.RuntimeKinds.requires_absolute_artifacts_path(
                    runtime.kind
                ):
                    raise mlrun.errors.MLRunPreconditionFailedError(
                        "Artifact path (`artifact_path`) must be absolute for remote tasks"
                    )
            elif (
                hasattr(runtime.spec, "volume_mounts")
                and not runtime.spec.volume_mounts
            ):
                message = (
                    "Artifact output path is local while no volume mount is specified. "
                    "Artifacts would not be visible via UI."
                )
            if message:
                logger.warning(message, output_path=run.spec.output_path)

    def _validate_run_params(self, parameters: dict[str, Any]):
        for param_name, param_value in parameters.items():
            if isinstance(param_value, dict):
                # if the parameter is a dict, we might have some nested parameters,
                # in this case we need to verify them as well recursively
                self._validate_run_params(param_value)
            self._validate_run_single_param(
                param_name=param_name, param_value=param_value
            )

    @classmethod
    def _validate_run_single_param(cls, param_name, param_value):
        # verify that integer parameters don't exceed a int64
        if isinstance(param_value, int) and abs(param_value) >= 2**63:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Parameter {param_name} value {param_value} exceeds int64"
            )

    @staticmethod
    def _create_run_object(task):
        valid_task_types = (dict, mlrun.run.RunTemplate, mlrun.run.RunObject)

        if not task:
            # if task passed generate default RunObject
            return mlrun.run.RunObject.from_dict(task)

        # deepcopy user's task, so we don't modify / enrich the user's object
        task = copy.deepcopy(task)

        if isinstance(task, str):
            task = ast.literal_eval(task)

        if not isinstance(task, valid_task_types):
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Task is not a valid object, type={type(task)}, expected types={valid_task_types}"
            )

        if isinstance(task, mlrun.run.RunTemplate):
            return mlrun.run.RunObject.from_template(task)
        elif isinstance(task, dict):
            return mlrun.run.RunObject.from_dict(task)

        # task is already a RunObject
        return task

    @staticmethod
    def _enrich_run(
        runtime,
        run,
        handler=None,
        project_name=None,
        name=None,
        params=None,
        inputs=None,
        returns=None,
        hyperparams=None,
        hyper_param_options=None,
        verbose=None,
        scrape_metrics=None,
        out_path=None,
        artifact_path=None,
        workdir=None,
        notifications: Optional[list[mlrun.model.Notification]] = None,
        state_thresholds: Optional[dict[str, int]] = None,
    ):
        run.spec.handler = (
            handler or run.spec.handler or runtime.spec.default_handler or ""
        )
        # callable handlers are valid for handler and dask runtimes,
        # for other runtimes we need to convert the handler to a string
        if run.spec.handler and runtime.kind not in ["handler", "dask"]:
            run.spec.handler = run.spec.handler_name

        def_name = runtime.metadata.name
        if run.spec.handler_name:
            short_name = run.spec.handler_name
            for separator in ["#", "::", "."]:
                # drop paths, module or class name from short name
                if separator in short_name:
                    short_name = short_name.split(separator)[-1]
            def_name += "-" + short_name

        run.metadata.name = mlrun.utils.normalize_name(
            name=name or run.metadata.name or def_name,
            # if name or runspec.metadata.name are set then it means that is user defined name and we want to warn the
            # user that the passed name needs to be set without underscore, if its not user defined but rather enriched
            # from the handler(function) name then we replace the underscore without warning the user.
            # most of the time handlers will have `_` in the handler name (python convention is to separate function
            # words with `_`), therefore we don't want to be noisy when normalizing the run name
            verbose=bool(name or run.metadata.name),
        )
        mlrun.utils.verify_field_regex(
            "run.metadata.name", run.metadata.name, mlrun.utils.regex.run_name
        )
        run.metadata.project = (
            project_name
            or run.metadata.project
            or runtime.metadata.project
            or mlrun.mlconf.default_project
        )
        run.spec.parameters = params or run.spec.parameters
        run.spec.inputs = inputs or run.spec.inputs
        run.spec.returns = returns or run.spec.returns
        run.spec.hyperparams = hyperparams or run.spec.hyperparams
        run.spec.hyper_param_options = (
            hyper_param_options or run.spec.hyper_param_options
        )
        run.spec.verbose = verbose or run.spec.verbose
        if scrape_metrics is None:
            if run.spec.scrape_metrics is None:
                scrape_metrics = mlrun.mlconf.scrape_metrics
            else:
                scrape_metrics = run.spec.scrape_metrics
        run.spec.scrape_metrics = scrape_metrics
        run.spec.input_path = workdir or run.spec.input_path or runtime.spec.workdir
        if runtime.spec.allow_empty_resources:
            run.spec.allow_empty_resources = runtime.spec.allow_empty_resources

        spec = run.spec
        if spec.secret_sources:
            runtime._secrets = mlrun.secrets.SecretsStore.from_list(spec.secret_sources)

        # update run metadata (uid, labels) and store in DB
        meta = run.metadata
        meta.uid = meta.uid or uuid.uuid4().hex

        run.spec.output_path = out_path or artifact_path or run.spec.output_path

        if not run.spec.output_path:
            if run.metadata.project:
                if (
                    mlrun.pipeline_context.project
                    and run.metadata.project
                    == mlrun.pipeline_context.project.metadata.name
                ):
                    run.spec.output_path = (
                        mlrun.pipeline_context.project.spec.artifact_path
                        or mlrun.pipeline_context.workflow_artifact_path
                    )

                # get_db might be None when no rundb is set on runtime
                if not run.spec.output_path and runtime._get_db():
                    try:
                        # not passing or loading the DB before the enrichment on purpose, because we want to enrich the
                        # spec first as get_db() depends on it
                        project = runtime._get_db().get_project(run.metadata.project)
                        # this is mainly for tests, so we won't need to mock get_project for so many tests
                        # in normal use cases if no project is found we will get an error
                        if project:
                            run.spec.output_path = project.spec.artifact_path
                    except mlrun.errors.MLRunNotFoundError:
                        logger.warning(
                            f"Project {project_name} is not saved in DB yet, "
                            f"enriching output path with default artifact path: {mlrun.mlconf.artifact_path}"
                        )

            if not run.spec.output_path:
                run.spec.output_path = mlrun.mlconf.artifact_path

        if run.spec.output_path:
            run.spec.output_path = mlrun.utils.helpers.template_artifact_path(
                run.spec.output_path, run.metadata.project, meta.uid
            )

        notifications = notifications or run.spec.notifications or []
        mlrun.model.Notification.validate_notification_uniqueness(notifications)
        for notification in notifications:
            notification.validate_notification()

        run.spec.notifications = notifications

        state_thresholds = (
            state_thresholds
            or run.spec.state_thresholds
            or getattr(runtime.spec, "state_thresholds", {})
            or {}
        )
        state_thresholds = (
            mlrun.mlconf.function.spec.state_thresholds.default.to_dict()
            | state_thresholds
        )
        run.spec.state_thresholds = state_thresholds or run.spec.state_thresholds
        return run

    @staticmethod
    def _run_has_valid_notifications(runobj) -> bool:
        if not runobj.spec.notifications:
            logger.debug(
                "No notifications to push for run", run_uid=runobj.metadata.uid
            )
            return False

        # TODO: add support for other notifications per run iteration
        if runobj.metadata.iteration and runobj.metadata.iteration > 0:
            logger.debug(
                "Notifications per iteration are not supported, skipping",
                run_uid=runobj.metadata.uid,
            )
            return False

        return True

    def _wrap_run_result(
        self,
        runtime: "mlrun.runtimes.BaseRuntime",
        result: dict,
        run: "mlrun.run.RunObject",
        schedule: Optional[mlrun.common.schemas.ScheduleCronTrigger] = None,
        err: Optional[Exception] = None,
    ):
        # if the purpose was to schedule (and not to run) nothing to wrap
        if schedule:
            return

        if result and runtime.kfp and err is None:
            mlrun_pipelines.common.ops.write_kfpmeta(result)

        self._log_track_results(runtime.is_child, result, run)

        if result:
            run = mlrun.run.RunObject.from_dict(result)
            logger.info(
                "Run execution finished",
                status=run.status.state,
                name=run.metadata.name,
            )
            if (
                run.status.state
                in mlrun.common.runtimes.constants.RunStates.error_and_abortion_states()
            ):
                if runtime._is_remote and not runtime.is_child:
                    logger.error(
                        "Run did not finish successfully",
                        state=run.status.state,
                        status=run.status.to_dict(),
                    )
                raise mlrun.runtimes.utils.RunError(run.error)
            return run

        return None

    @staticmethod
    def _refresh_function_metadata(runtime: "mlrun.runtimes.BaseRuntime"):
        pass

    @staticmethod
    def _log_track_results(
        runtime: "mlrun.runtimes.BaseRuntime", result: dict, run: "mlrun.run.RunObject"
    ):
        pass
