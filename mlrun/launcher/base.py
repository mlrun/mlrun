# Copyright 2023 MLRun Authors
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
from typing import Any, Dict, List, Optional, Union

import mlrun.errors
import mlrun.model
import mlrun.runtimes
from mlrun.model import RunObject
from mlrun.utils import logger

run_modes = ["pass"]


class BaseLauncher(abc.ABC):
    """
    Abstract class for managing and running functions in different contexts
    This class is designed to encapsulate the logic of running a function in different contexts
    i.e. running a function locally, remotely or in a server
    Each context will have its own implementation of the abstract methods while the common logic resides in this class
    """

    def __init__(self):
        self._db = None

    @property
    def db(self) -> mlrun.db.base.RunDBInterface:
        if not self._db:
            self._db = mlrun.db.get_run_db()
        return self._db

    @staticmethod
    @abc.abstractmethod
    def verify_base_image(runtime):
        """resolves and sets the build base image if build is needed"""
        pass

    @staticmethod
    @abc.abstractmethod
    def save(runtime):
        """store the function to the db"""
        pass

    def launch(
        self,
        runtime: mlrun.runtimes.BaseRuntime,
        task: Optional[Union[mlrun.run.RunTemplate, mlrun.run.RunObject]] = None,
        handler: Optional[str] = None,
        name: Optional[str] = "",
        project: Optional[str] = "",
        params: Optional[dict] = None,
        inputs: Optional[Dict[str, str]] = None,
        out_path: Optional[str] = "",
        workdir: Optional[str] = "",
        artifact_path: Optional[str] = "",
        watch: Optional[bool] = True,
        # TODO: don't use schedule from API schemas but rather from mlrun client
        schedule: Optional[
            Union[str, mlrun.api.schemas.schedule.ScheduleCronTrigger]
        ] = None,
        hyperparams: Dict[str, list] = None,
        hyper_param_options: Optional[mlrun.model.HyperParamOptions] = None,
        verbose: Optional[bool] = None,
        scrape_metrics: Optional[bool] = None,
        local: Optional[bool] = False,
        local_code_path: Optional[str] = None,
        auto_build: Optional[bool] = None,
        param_file_secrets: Optional[Dict[str, str]] = None,
        notifications: Optional[List[mlrun.model.Notification]] = None,
        returns: Optional[List[Union[str, Dict[str, str]]]] = None,
    ) -> mlrun.run.RunObject:
        """run the function from the server/client[local/remote]"""
        pass

    @staticmethod
    @abc.abstractmethod
    def _enrich_runtime(runtime):
        pass

    def _validate_runtime(
        self,
        runtime: mlrun.runtimes.BaseRuntime,
        run: RunObject,
    ):
        mlrun.utils.helpers.verify_dict_items_type(
            "Inputs", run.spec.inputs, [str], [str]
        )

        if runtime.spec.mode and runtime.spec.mode not in run_modes:
            raise ValueError(f'run mode can only be {",".join(run_modes)}')

        self._verify_run_params(run.spec.parameters)
        self._validate_output_path(runtime, run)

    @staticmethod
    def _validate_output_path(
        runtime: mlrun.runtimes.BaseRuntime,
        run: RunObject,
    ):
        if not run.spec.output_path or "://" not in run.spec.output_path:
            message = ""
            if not os.path.isabs(run.spec.output_path):
                message = (
                    "artifact/output path is not defined or is local and relative,"
                    " artifacts will not be visible in the UI"
                )
                if mlrun.runtimes.RuntimeKinds.requires_absolute_artifacts_path(
                    runtime.kind
                ):
                    raise mlrun.errors.MLRunPreconditionFailedError(
                        "artifact path (`artifact_path`) must be absolute for remote tasks"
                    )
            elif (
                hasattr(runtime.spec, "volume_mounts")
                and not runtime.spec.volume_mounts
            ):
                message = (
                    "artifact output path is local while no volume mount is specified. "
                    "artifacts would not be visible via UI."
                )
            if message:
                logger.warning(message, output_path=run.spec.output_path)

    def _verify_run_params(self, parameters: Dict[str, Any]):
        for param_name, param_value in parameters.items():

            if isinstance(param_value, dict):
                # if the parameter is a dict, we might have some nested parameters,
                # in this case we need to verify them as well recursively
                self._verify_run_params(param_value)

            # verify that integer parameters don't exceed a int64
            if isinstance(param_value, int) and abs(param_value) >= 2**63:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"parameter {param_name} value {param_value} exceeds int64"
                )

    @abc.abstractmethod
    def _save_or_push_notifications(self, runobj):
        pass

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

    @staticmethod
    def _enrich_run(
        runtime,
        runspec,
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
        notifications: List[mlrun.model.Notification] = None,
    ):
        runspec.spec.handler = (
            handler or runspec.spec.handler or runtime.spec.default_handler or ""
        )
        if runspec.spec.handler and runtime.kind not in ["handler", "dask"]:
            runspec.spec.handler = runspec.spec.handler_name

        def_name = runtime.metadata.name
        if runspec.spec.handler_name:
            short_name = runspec.spec.handler_name
            for separator in ["#", "::", "."]:
                # drop paths, module or class name from short name
                if separator in short_name:
                    short_name = short_name.split(separator)[-1]
            def_name += "-" + short_name

        runspec.metadata.name = mlrun.utils.normalize_name(
            name=name or runspec.metadata.name or def_name,
            # if name or runspec.metadata.name are set then it means that is user defined name and we want to warn the
            # user that the passed name needs to be set without underscore, if its not user defined but rather enriched
            # from the handler(function) name then we replace the underscore without warning the user.
            # most of the time handlers will have `_` in the handler name (python convention is to separate function
            # words with `_`), therefore we don't want to be noisy when normalizing the run name
            verbose=bool(name or runspec.metadata.name),
        )
        mlrun.utils.verify_field_regex(
            "run.metadata.name", runspec.metadata.name, mlrun.utils.regex.run_name
        )
        runspec.metadata.project = (
            project_name
            or runspec.metadata.project
            or runtime.metadata.project
            or mlrun.mlconf.default_project
        )
        runspec.spec.parameters = params or runspec.spec.parameters
        runspec.spec.inputs = inputs or runspec.spec.inputs
        runspec.spec.returns = returns or runspec.spec.returns
        runspec.spec.hyperparams = hyperparams or runspec.spec.hyperparams
        runspec.spec.hyper_param_options = (
            hyper_param_options or runspec.spec.hyper_param_options
        )
        runspec.spec.verbose = verbose or runspec.spec.verbose
        if scrape_metrics is None:
            if runspec.spec.scrape_metrics is None:
                scrape_metrics = mlrun.mlconf.scrape_metrics
            else:
                scrape_metrics = runspec.spec.scrape_metrics
        runspec.spec.scrape_metrics = scrape_metrics
        runspec.spec.input_path = (
            workdir or runspec.spec.input_path or runtime.spec.workdir
        )
        if runtime.spec.allow_empty_resources:
            runspec.spec.allow_empty_resources = runtime.spec.allow_empty_resources

        spec = runspec.spec
        if spec.secret_sources:
            runtime._secrets = mlrun.secrets.SecretsStore.from_list(spec.secret_sources)

        # update run metadata (uid, labels) and store in DB
        meta = runspec.metadata
        meta.uid = meta.uid or uuid.uuid4().hex

        runspec.spec.output_path = out_path or artifact_path or runspec.spec.output_path

        if not runspec.spec.output_path:
            if runspec.metadata.project:
                if (
                    mlrun.pipeline_context.project
                    and runspec.metadata.project
                    == mlrun.pipeline_context.project.metadata.name
                ):
                    runspec.spec.output_path = (
                        mlrun.pipeline_context.project.spec.artifact_path
                        or mlrun.pipeline_context.workflow_artifact_path
                    )

                if not runspec.spec.output_path and runtime._get_db():
                    try:
                        # not passing or loading the DB before the enrichment on purpose, because we want to enrich the
                        # spec first as get_db() depends on it
                        project = runtime._get_db().get_project(
                            runspec.metadata.project
                        )
                        # this is mainly for tests, so we won't need to mock get_project for so many tests
                        # in normal use cases if no project is found we will get an error
                        if project:
                            runspec.spec.output_path = project.spec.artifact_path
                    except mlrun.errors.MLRunNotFoundError:
                        logger.warning(
                            f"project {project_name} is not saved in DB yet, "
                            f"enriching output path with default artifact path: {mlrun.mlconf.artifact_path}"
                        )

            if not runspec.spec.output_path:
                runspec.spec.output_path = mlrun.mlconf.artifact_path

        if runspec.spec.output_path:
            runspec.spec.output_path = runspec.spec.output_path.replace(
                "{{run.uid}}", meta.uid
            )
            runspec.spec.output_path = mlrun.utils.helpers.fill_artifact_path_template(
                runspec.spec.output_path, runspec.metadata.project
            )

        runspec.spec.notifications = notifications or runspec.spec.notifications or []
        return runspec

    @staticmethod
    def _are_valid_notifications(runobj) -> bool:
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
