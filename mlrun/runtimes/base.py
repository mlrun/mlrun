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
import enum
import http
import re
import typing
import warnings
from os import environ
from typing import Callable, Optional, Union

import requests.exceptions
from nuclio.build import mlrun_footer

import mlrun.common.constants
import mlrun.common.constants as mlrun_constants
import mlrun.common.formatters
import mlrun.common.runtimes
import mlrun.common.schemas
import mlrun.common.schemas.model_monitoring.constants as mm_constants
import mlrun.errors
import mlrun.launcher.factory
import mlrun.utils.helpers
import mlrun.utils.notifications
import mlrun.utils.regex
from mlrun.model import (
    BaseMetadata,
    HyperParamOptions,
    ImageBuilder,
    ModelObj,
    RunObject,
    RunTemplate,
)
from mlrun.utils.helpers import generate_object_uri, verify_field_regex
from mlrun_pipelines.common.ops import mlrun_op

from ..config import config
from ..datastore import store_manager
from ..errors import err_to_str
from ..lists import RunList
from ..utils import (
    dict_to_json,
    dict_to_yaml,
    enrich_image_url,
    get_in,
    get_parsed_docker_registry,
    logger,
    now_date,
    update_in,
)
from .funcdoc import update_function_entry_points
from .utils import RunError, calc_hash

spec_fields = [
    "command",
    "args",
    "image",
    "mode",
    "build",
    "entry_points",
    "description",
    "workdir",
    "default_handler",
    "pythonpath",
    "disable_auto_mount",
    "allow_empty_resources",
    "clone_target_dir",
    "reset_on_run",
]


class RuntimeClassMode(enum.Enum):
    """
    Runtime class mode
    Currently there are two modes:
    * run - the runtime class is used to run a function
    * build - the runtime class is used to build a function

    The runtime class mode is used to determine what should be the name of the runtime class, each runtime might have a
    different name for each mode and some might not have both modes.
    """

    run = "run"
    build = "build"


class FunctionStatus(ModelObj):
    def __init__(self, state=None, build_pod=None):
        self.state = state
        self.build_pod = build_pod


class FunctionSpec(ModelObj):
    _dict_fields = spec_fields
    _default_fields_to_strip = []

    def __init__(
        self,
        command=None,
        args=None,
        image=None,
        mode=None,
        build=None,
        entry_points=None,
        description=None,
        workdir=None,
        default_handler=None,
        pythonpath=None,
        disable_auto_mount=False,
        clone_target_dir=None,
    ):
        self.command = command or ""
        self.image = image or ""
        self.mode = mode
        self.args = args or []
        self.rundb = None
        self.description = description or ""
        self.workdir = workdir
        self.pythonpath = pythonpath

        self._build = None
        self.build = build
        self.default_handler = default_handler
        self.entry_points = entry_points or {}
        self.disable_auto_mount = disable_auto_mount
        self.allow_empty_resources = None
        # The build.source is cloned/extracted to the specified clone_target_dir
        # if a relative path is specified, it will be enriched with a temp dir path
        self._clone_target_dir = clone_target_dir or None

    @property
    def build(self) -> ImageBuilder:
        return self._build

    @build.setter
    def build(self, build):
        self._build = self._verify_dict(build, "build", ImageBuilder)

    @property
    def clone_target_dir(self):
        # TODO: remove this property in 1.10.0
        if self.build.source_code_target_dir:
            warnings.warn(
                "The clone_target_dir attribute is deprecated in 1.6.2 and will be removed in 1.10.0. "
                "Use spec.build.source_code_target_dir instead.",
                FutureWarning,
            )
        return self.build.source_code_target_dir

    @clone_target_dir.setter
    def clone_target_dir(self, clone_target_dir):
        # TODO: remove this property in 1.10.0
        if clone_target_dir:
            warnings.warn(
                "The clone_target_dir attribute is deprecated in 1.6.2 and will be removed in 1.10.0. "
                "Use spec.build.source_code_target_dir instead.",
                FutureWarning,
            )
        self.build.source_code_target_dir = clone_target_dir

    def enrich_function_preemption_spec(self):
        pass

    def validate_service_account(self, allowed_service_accounts):
        pass


class BaseRuntime(ModelObj):
    kind = "base"
    _is_nested = False
    _is_remote = False
    _dict_fields = ["kind", "metadata", "spec", "status", "verbose"]
    _default_fields_to_strip = ModelObj._default_fields_to_strip + [
        "status",  # Function status describes the state rather than configuration
    ]

    def __init__(self, metadata=None, spec=None):
        self._metadata = None
        self.metadata = metadata
        self.kfp = None
        self._spec = None
        self.spec = spec
        self._db_conn = None
        self._secrets = None
        self._k8s = None
        self._is_built = False
        self.is_child = False
        self._status = None
        self.status = None
        self.verbose = False
        self._enriched_image = False

    def set_db_connection(self, conn):
        if not self._db_conn:
            self._db_conn = conn

    @property
    def metadata(self) -> BaseMetadata:
        return self._metadata

    @metadata.setter
    def metadata(self, metadata):
        self._metadata = self._verify_dict(metadata, "metadata", BaseMetadata)

    @property
    def spec(self) -> FunctionSpec:
        return self._spec

    @spec.setter
    def spec(self, spec):
        self._spec = self._verify_dict(spec, "spec", FunctionSpec)

    @property
    def status(self) -> FunctionStatus:
        return self._status

    @status.setter
    def status(self, status):
        self._status = self._verify_dict(status, "status", FunctionStatus)

    def set_label(self, key, value):
        self.metadata.labels[key] = str(value)
        return self

    def set_categories(self, categories: list[str]):
        self.metadata.categories = mlrun.utils.helpers.as_list(categories)

    @property
    def uri(self):
        return self._function_uri()

    def is_deployed(self):
        return True

    def is_model_monitoring_function(self):
        return (
            self.metadata.labels.get(mm_constants.ModelMonitoringAppLabel.KEY, "")
            == mm_constants.ModelMonitoringAppLabel.VAL
        )

    def _is_remote_api(self):
        db = self._get_db()
        if db and db.kind == "http":
            return True
        return False

    def _function_uri(self, tag=None, hash_key=None):
        return generate_object_uri(
            self.metadata.project,
            self.metadata.name,
            tag=tag or self.metadata.tag,
            hash_key=hash_key,
        )

    def _ensure_run_db(self):
        self.spec.rundb = self.spec.rundb or mlrun.db.get_or_set_dburl()

    def _get_db(self):
        # TODO: remove this function and use the launcher db instead
        self._ensure_run_db()
        if not self._db_conn:
            if self.spec.rundb:
                self._db_conn = mlrun.db.get_run_db(
                    self.spec.rundb, secrets=self._secrets
                )
        return self._db_conn

    # This function is different than the auto_mount function, as it mounts to runtimes based on the configuration.
    # That's why it's named differently.
    def try_auto_mount_based_on_config(self):
        pass

    def validate_and_enrich_service_account(
        self, allowed_service_account, default_service_account
    ):
        pass

    def _fill_credentials(self):
        """
        If access key is not mask (starts with secret prefix) then fill $generate so that the API will handle filling
         of the credentials.
        We rely on the HTTPDB to send the access key session through the request header and that the API will mask
         the access key, that way we won't even store any plain access key in the function.
        """
        if self.metadata.credentials.access_key and (
            # if contains secret reference or $generate then no need to overwrite the access key
            self.metadata.credentials.access_key.startswith(
                mlrun.model.Credentials.secret_reference_prefix
            )
            or self.metadata.credentials.access_key.startswith(
                mlrun.model.Credentials.generate_access_key
            )
        ):
            return
        self.metadata.credentials.access_key = (
            mlrun.model.Credentials.generate_access_key
        )

    def generate_runtime_k8s_env(self, runobj: RunObject = None) -> list[dict]:
        """
        Prepares a runtime environment as it's expected by kubernetes.models.V1Container

        :param runobj: Run context object (RunObject) with run metadata and status
        :return: List of dicts with the structure {"name": "var_name", "value": "var_value"}
        """
        return [
            {"name": k, "value": v}
            for k, v in self._generate_runtime_env(runobj).items()
        ]

    def run(
        self,
        runspec: Optional[
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
        schedule: Optional[Union[str, mlrun.common.schemas.ScheduleCronTrigger]] = None,
        hyperparams: Optional[dict[str, list]] = None,
        hyper_param_options: Optional[HyperParamOptions] = None,
        verbose: Optional[bool] = None,
        scrape_metrics: Optional[bool] = None,
        local: Optional[bool] = False,
        local_code_path: Optional[str] = None,
        auto_build: Optional[bool] = None,
        param_file_secrets: Optional[dict[str, str]] = None,
        notifications: Optional[list[mlrun.model.Notification]] = None,
        returns: Optional[list[Union[str, dict[str, str]]]] = None,
        state_thresholds: Optional[dict[str, int]] = None,
        reset_on_run: Optional[bool] = None,
        **launcher_kwargs,
    ) -> RunObject:
        """
        Run a local or remote task.

        :param runspec:        The run spec to generate the RunObject from. Can be RunTemplate | RunObject | dict.
        :param handler:        Pointer or name of a function handler.
        :param name:           Execution name.
        :param project:        Project name.
        :param params:         Input parameters (dict).
        :param inputs:         Input objects to pass to the handler. Type hints can be given so the input will be parsed
                               during runtime from `mlrun.DataItem` to the given type hint. The type hint can be given
                               in the key field of the dictionary after a colon, e.g: "<key> : <type_hint>".
        :param out_path:       Default artifact output path.
        :param artifact_path:  Default artifact output path (will replace out_path).
        :param workdir:        Default input artifacts path.
        :param watch:          Watch/follow run log.
        :param schedule:       ScheduleCronTrigger class instance or a standard crontab expression string
                               (which will be converted to the class using its `from_crontab` constructor),
                               see this link for help:
                               https://apscheduler.readthedocs.io/en/3.x/modules/triggers/cron.html#module-apscheduler.triggers.cron
        :param hyperparams:    Dict of param name and list of values to be enumerated.
                               The default strategy is grid search and uses e.g. {"p1": [1,2,3]}.
                               (Can be specified as a JSON file)
                               For list, lists must be of equal length, e.g. {"p1": [1], "p2": [2]}.
                               (Can be specified as JSON file or as a CSV file listing the parameter values
                               per iteration.)
                               You can specify strategy of type grid, list, random,
                               and other options in the hyper_param_options parameter.
        :param hyper_param_options: Dict or :py:class:`~mlrun.model.HyperParamOptions` struct of hyperparameter options.
        :param verbose:             Add verbose prints/logs.
        :param scrape_metrics:      Whether to add the `mlrun/scrape-metrics` label to this run's resources.
        :param local:               Run the function locally vs on the runtime/cluster.
        :param local_code_path:     Path of the code for local runs & debug.
        :param auto_build:          When set to True and the function require build it will be built on the first
                                    function run, use only if you don't plan on changing the build config between runs.
        :param param_file_secrets:  Dictionary of secrets to be used only for accessing the hyper-param parameter file.
                                    These secrets are only used locally and will not be stored anywhere
        :param notifications:       List of notifications to push when the run is completed
        :param returns: List of log hints - configurations for how to log the returning values from the handler's run
                        (as artifacts or results). The list's length must be equal to the amount of returning objects. A
                        log hint may be given as:

                        * A string of the key to use to log the returning value as result or as an artifact. To specify
                          The artifact type, it is possible to pass a string in the following structure:
                          "<key> : <type>". Available artifact types can be seen in `mlrun.ArtifactType`. If no
                          artifact type is specified, the object's default artifact type will be used.
                        * A dictionary of configurations to use when logging. Further info per object type and artifact
                          type can be given there. The artifact key must appear in the dictionary as "key": "the_key".
        :param state_thresholds:    Dictionary of states to time thresholds. The state will be matched against the
                k8s resource's status. The threshold should be a time string that conforms to timelength python package
                standards and is at least 1 minute (-1 for infinite).
                If the phase is active for longer than the threshold, the run will be aborted.
                See mlconf.function.spec.state_thresholds for the state options and default values.
        :param reset_on_run: When True, function python modules would reload prior to code execution.
                             This ensures latest code changes are executed. This argument must be used in
                             conjunction with the local=True argument.
        :return: Run context object (RunObject) with run metadata, results and status
        """
        launcher = mlrun.launcher.factory.LauncherFactory().create_launcher(
            self._is_remote, local=local, **launcher_kwargs
        )
        return launcher.launch(
            runtime=self,
            task=runspec,
            handler=handler,
            name=name,
            project=project,
            params=params,
            inputs=inputs,
            out_path=out_path,
            workdir=workdir,
            artifact_path=artifact_path,
            watch=watch,
            schedule=schedule,
            hyperparams=hyperparams,
            hyper_param_options=hyper_param_options,
            verbose=verbose,
            scrape_metrics=scrape_metrics,
            local_code_path=local_code_path,
            auto_build=auto_build,
            param_file_secrets=param_file_secrets,
            notifications=notifications,
            returns=returns,
            state_thresholds=state_thresholds,
            reset_on_run=reset_on_run,
        )

    def _get_db_run(
        self,
        task: RunObject = None,
        run_format: mlrun.common.formatters.RunFormat = mlrun.common.formatters.RunFormat.full,
    ):
        if self._get_db() and task:
            project = task.metadata.project
            uid = task.metadata.uid
            iter = task.metadata.iteration
            try:
                return self._get_db().read_run(
                    uid, project, iter=iter, format_=run_format
                )
            except mlrun.db.RunDBError:
                return None
        if task:
            return task.to_dict()

    def _generate_runtime_env(self, runobj: RunObject = None) -> dict:
        """
        Prepares all available environment variables for usage on a runtime
        Data will be extracted from several sources and most of them are not guaranteed to be available

        :param runobj: Run context object (RunObject) with run metadata and status
        :return: Dictionary with all the variables that could be parsed
        """
        runtime_env = {
            "MLRUN_DEFAULT_PROJECT": self.metadata.project or config.default_project
        }
        if runobj:
            runtime_env["MLRUN_EXEC_CONFIG"] = runobj.to_json(
                exclude_notifications_params=True
            )
            if runobj.metadata.project:
                runtime_env["MLRUN_DEFAULT_PROJECT"] = runobj.metadata.project
            if runobj.spec.verbose:
                runtime_env["MLRUN_LOG_LEVEL"] = "DEBUG"
        if config.httpdb.api_url:
            runtime_env["MLRUN_DBPATH"] = config.httpdb.api_url
        if self.metadata.namespace or config.namespace:
            runtime_env["MLRUN_NAMESPACE"] = self.metadata.namespace or config.namespace
        return runtime_env

    @staticmethod
    def _handle_submit_job_http_error(error: requests.HTTPError):
        # if we receive a 400 status code, this means the request was invalid and the run wasn't created in the DB.
        # so we don't need to update the run state and we can just raise the error.
        # more status code handling can be added here if needed
        if error.response.status_code == http.HTTPStatus.BAD_REQUEST.value:
            raise mlrun.errors.MLRunBadRequestError(
                f"Bad request to mlrun api: {error.response.text}"
            )

    def _store_function(self, runspec, meta, db):
        meta.labels["kind"] = self.kind
        mlrun.runtimes.utils.enrich_run_labels(
            meta.labels, [mlrun_constants.MLRunInternalLabels.owner]
        )
        if runspec.spec.output_path:
            runspec.spec.output_path = runspec.spec.output_path.replace(
                "{{run.user}}", meta.labels[mlrun_constants.MLRunInternalLabels.owner]
            )

        if db and self.kind != "handler":
            struct = self.to_dict()
            hash_key = db.store_function(
                struct, self.metadata.name, self.metadata.project, versioned=True
            )
            runspec.spec.function = self._function_uri(hash_key=hash_key)

    def _pre_run(self, runspec: RunObject, execution):
        pass

    def _post_run(self, results, execution):
        pass

    def _run(self, runobj: RunObject, execution) -> dict:
        pass

    def _run_many(self, generator, execution, runobj: RunObject) -> RunList:
        results = RunList()
        num_errors = 0
        tasks = generator.generate(runobj)
        for task in tasks:
            try:
                self.store_run(task)
                resp = self._run(task, execution)
                resp = self._update_run_state(resp, task=task)
                run_results = resp["status"].get("results", {})
                if generator.eval_stop_condition(run_results):
                    logger.info(
                        f"Reached early stop condition ({generator.options.stop_condition}), stopping iterations!"
                    )
                    results.append(resp)
                    break

            except RunError as err:
                task.status.state = "error"
                error_string = err_to_str(err)
                task.status.error = error_string
                resp = self._update_run_state(task=task, err=error_string)
                num_errors += 1
                if num_errors > generator.max_errors:
                    logger.error("Too many errors, stopping iterations!")
                    results.append(resp)
                    break

            results.append(resp)

        return results

    def store_run(self, runobj: RunObject):
        if self._get_db() and runobj:
            project = runobj.metadata.project
            uid = runobj.metadata.uid
            iter = runobj.metadata.iteration
            self._get_db().store_run(runobj.to_dict(), uid, project, iter=iter)

    def _store_run_dict(self, rundict: dict):
        if self._get_db() and rundict:
            project = get_in(rundict, "metadata.project", "")
            uid = get_in(rundict, "metadata.uid")
            iter = get_in(rundict, "metadata.iteration", 0)
            self._get_db().store_run(rundict, uid, project, iter=iter)

    def _update_run_state(
        self,
        resp: Optional[dict] = None,
        task: RunObject = None,
        err: Optional[Union[Exception, str]] = None,
        run_format: mlrun.common.formatters.RunFormat = mlrun.common.formatters.RunFormat.full,
    ) -> typing.Optional[dict]:
        """update the task state in the DB"""
        was_none = False
        if resp is None and task:
            was_none = True
            resp = self._get_db_run(task, run_format)

            if not resp:
                self.store_run(task)
                return task.to_dict()

            if task.status.status_text:
                update_in(resp, "status.status_text", task.status.status_text)

        if resp is None:
            return None

        if not isinstance(resp, dict):
            raise ValueError(f"post_run called with type {type(resp)}")

        updates = None
        last_state = get_in(resp, "status.state", "")
        kind = get_in(resp, "metadata.labels.kind", "")
        if last_state == "error" or err:
            updates = {
                "status.last_update": now_date().isoformat(),
                "status.state": "error",
            }
            update_in(resp, "status.state", "error")
            if err:
                update_in(resp, "status.error", err_to_str(err))
            err = get_in(resp, "status.error")
            if err:
                updates["status.error"] = err_to_str(err)

        elif (
            not was_none
            and last_state != mlrun.common.runtimes.constants.RunStates.completed
            and last_state
            not in mlrun.common.runtimes.constants.RunStates.error_and_abortion_states()
        ):
            try:
                runtime_cls = mlrun.runtimes.get_runtime_class(kind)
                updates = runtime_cls._get_run_completion_updates(resp)
            except KeyError:
                updates = self._get_run_completion_updates(resp)

        uid = get_in(resp, "metadata.uid")
        logger.debug(
            "Run updates",
            name=get_in(resp, "metadata.name"),
            uid=uid,
            kind=kind,
            last_state=last_state,
            updates=updates,
        )
        if self._get_db() and updates:
            project = get_in(resp, "metadata.project")
            iter = get_in(resp, "metadata.iteration", 0)
            self._get_db().update_run(updates, uid, project, iter=iter)

        return resp

    def _force_handler(self, handler):
        if not handler:
            raise RunError(f"Handler must be provided for {self.kind} runtime")

    def _has_pipeline_param(self) -> bool:
        # check if the runtime has pipeline parameters
        # https://www.kubeflow.org/docs/components/pipelines/v1/sdk/parameters/
        matches = re.findall(mlrun.utils.regex.pipeline_param[0], self.to_json())
        return bool(matches)

    @staticmethod
    def _get_run_completion_updates(run: dict) -> dict:
        """
        Get the required updates for the run object when it's completed and update the run object state
        Override this if the run completion is not resolved by a single execution
        """
        updates = {
            "status.last_update": now_date().isoformat(),
            "status.state": "completed",
        }
        update_in(run, "status.state", "completed")
        return updates

    def full_image_path(
        self,
        image=None,
        client_version: Optional[str] = None,
        client_python_version: Optional[str] = None,
    ):
        image = image or self.spec.image or ""

        image = enrich_image_url(image, client_version, client_python_version)
        if not image.startswith(
            mlrun.common.constants.IMAGE_NAME_ENRICH_REGISTRY_PREFIX
        ):
            return image
        registry, repository = get_parsed_docker_registry()
        if registry:
            if repository and repository not in image:
                return f"{registry}/{repository}/{image[1:]}"
            return f"{registry}/{image[1:]}"
        namespace_domain = environ.get("IGZ_NAMESPACE_DOMAIN", None)
        if namespace_domain is not None:
            return f"docker-registry.{namespace_domain}:80/{image[1:]}"
        raise RunError("Local container registry is not defined")

    def as_step(
        self,
        runspec: Union[RunObject, RunTemplate] = None,
        handler=None,
        name: str = "",
        project: str = "",
        params: Optional[dict] = None,
        hyperparams=None,
        selector="",
        hyper_param_options: HyperParamOptions = None,
        inputs: Optional[dict] = None,
        outputs: Optional[list] = None,
        workdir: str = "",
        artifact_path: str = "",
        image: str = "",
        labels: Optional[dict] = None,
        use_db=True,
        verbose=None,
        scrape_metrics=False,
        returns: Optional[list[Union[str, dict[str, str]]]] = None,
        auto_build: bool = False,
    ):
        """Run a local or remote task.

        :param runspec:         run template object or dict (see RunTemplate)
        :param handler:         name of the function handler
        :param name:            execution name
        :param project:         project name
        :param params:          input parameters (dict)
        :param hyperparams:     hyper parameters
        :param selector:        selection criteria for hyper params
        :param hyper_param_options:  hyper param options (selector, early stop, strategy, ..)
                            see: :py:class:`~mlrun.model.HyperParamOptions`
        :param inputs:          Input objects to pass to the handler. Type hints can be given so the input will be
                                parsed during runtime from `mlrun.DataItem` to the given type hint. The type hint can be
                                given in the key field of the dictionary after a colon, e.g: "<key> : <type_hint>".
        :param outputs:         list of outputs which can pass in the workflow
        :param artifact_path:   default artifact output path (replace out_path)
        :param workdir:         default input artifacts path
        :param image:           container image to use
        :param labels:          labels to tag the job/run with ({key:val, ..})
        :param use_db:          save function spec in the db (vs the workflow file)
        :param verbose:         add verbose prints/logs
        :param scrape_metrics:  whether to add the `mlrun/scrape-metrics` label to this run's resources
        :param returns:         List of configurations for how to log the returning values from the handler's run
                                (as artifacts or results). The list's length must be equal to the amount of returning
                                objects. A configuration may be given as:

                                * A string of the key to use to log the returning value as result or as an artifact.
                                  To specify The artifact type, it is possible to pass a string in the following
                                  structure:
                                  "<key> : <type>". Available artifact types can be seen in `mlrun.ArtifactType`. If no
                                  artifact type is specified, the object's default artifact type will be used.
                                * A dictionary of configurations to use when logging. Further info per object type and
                                  artifact type can be given there. The artifact key must appear in the dictionary as
                                  "key": "the_key".
        :param auto_build:      when set to True and the function require build it will be built on the first
                                function run, use only if you dont plan on changing the build config between runs
        :return: mlrun_pipelines.models.PipelineNodeWrapper
        """

        # if the function contain KFP PipelineParams (futures) pass the full spec to the
        # PipelineNodeWrapper this way KFP will substitute the params with previous step outputs
        if use_db and not self._has_pipeline_param():
            # if the same function is built as part of the pipeline we do not use the versioned function
            # rather the latest function w the same tag so we can pick up the updated image/status
            versioned = False if hasattr(self, "_build_in_pipeline") else True
            url = self.save(versioned=versioned, refresh=True)
        else:
            url = None

        if runspec is not None:
            verify_field_regex(
                "run.metadata.name", runspec.metadata.name, mlrun.utils.regex.run_name
            )

        return mlrun_op(
            name,
            project,
            function=self,
            func_url=url,
            runobj=runspec,
            handler=handler,
            params=params,
            hyperparams=hyperparams,
            selector=selector,
            hyper_param_options=hyper_param_options,
            inputs=inputs,
            returns=returns,
            outputs=outputs,
            job_image=image,
            labels=labels,
            out_path=artifact_path,
            in_path=workdir,
            verbose=verbose,
            scrape_metrics=scrape_metrics,
            auto_build=auto_build,
        )

    def with_code(self, from_file="", body=None, with_doc=True):
        """Update the function code
        This function eliminates the need to build container images every time we edit the code

        :param from_file:   blank for current notebook, or path to .py/.ipynb file
        :param body:        will use the body as the function code
        :param with_doc:    update the document of the function parameters

        :return: function object
        """
        if body and from_file:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "must provide either body or from_file argument. not both"
            )

        if (not body and not from_file) or (from_file and from_file.endswith(".ipynb")):
            from nuclio import build_file

            _, _, body = build_file(from_file, name=self.metadata.name)

        else:
            if from_file:
                with open(from_file) as fp:
                    body = fp.read()
            if self.kind == mlrun.runtimes.RuntimeKinds.serving:
                body = body + mlrun_footer.format(
                    mlrun.runtimes.nuclio.serving.serving_subkind
                )

        self.spec.build.functionSourceCode = mlrun.utils.helpers.encode_user_code(body)
        if with_doc:
            update_function_entry_points(self, body)
        return self

    def with_requirements(
        self,
        requirements: Optional[list[str]] = None,
        overwrite: bool = False,
        prepare_image_for_deploy: bool = True,
        requirements_file: Optional[str] = "",
    ):
        """add package requirements from file or list to build spec.

        :param requirements:                a list of python packages
        :param requirements_file:           a local python requirements file path
        :param overwrite:                   overwrite existing requirements
        :param prepare_image_for_deploy:    prepare the image/base_image spec for deployment
        :return: function object
        """
        self.spec.build.with_requirements(requirements, requirements_file, overwrite)

        if prepare_image_for_deploy:
            self.prepare_image_for_deploy()

        return self

    def with_commands(
        self,
        commands: list[str],
        overwrite: bool = False,
        prepare_image_for_deploy: bool = True,
    ):
        """add commands to build spec.

        :param commands:                    list of commands to run during build
        :param overwrite:                   overwrite existing commands
        :param prepare_image_for_deploy:    prepare the image/base_image spec for deployment

        :return: function object
        """
        self.spec.build.with_commands(commands, overwrite)

        if prepare_image_for_deploy:
            self.prepare_image_for_deploy()
        return self

    def clean_build_params(self):
        # when using `with_requirements` we also execute `prepare_image_for_deploy` which adds the base image
        # and cleans the spec.image, so we need to restore the image back
        if self.spec.build.base_image and not self.spec.image:
            self.spec.image = self.spec.build.base_image

        self.spec.build = {}
        return self

    def requires_build(self) -> bool:
        build = self.spec.build
        return bool(
            build.commands
            or build.requirements
            or (build.source and not build.load_source_on_run)
        )

    def enrich_runtime_spec(
        self,
        project_node_selector: dict[str, str],
    ):
        pass

    def prepare_image_for_deploy(self):
        """
        if a function has a 'spec.image' it is considered to be deployed,
        but because we allow the user to set 'spec.image' for usability purposes,
        we need to check whether this is a built image or it requires to be built on top.
        """
        launcher = mlrun.launcher.factory.LauncherFactory().create_launcher(
            is_remote=self._is_remote
        )
        launcher.prepare_image_for_deploy(self)

    def export(self, target="", format=".yaml", secrets=None, strip=True):
        """save function spec to a local/remote path (default to./function.yaml)

        :param target:   target path/url
        :param format:   `.yaml` (default) or `.json`
        :param secrets:  optional secrets dict/object for target path (e.g. s3)
        :param strip:    strip status data

        :returns: self
        """
        if self.kind == "handler":
            raise ValueError(
                "cannot export local handler function, use "
                + "code_to_function() to serialize your function"
            )
        calc_hash(self)
        struct = self.to_dict(strip=strip)
        if format == ".yaml":
            data = dict_to_yaml(struct)
        else:
            data = dict_to_json(struct)
        stores = store_manager.set(secrets)
        target = target or "function.yaml"
        datastore, subpath, url = stores.get_or_create_store(target)
        datastore.put(subpath, data)
        logger.info(f"function spec saved to path: {target}")
        return self

    def save(self, tag="", versioned=False, refresh=False) -> str:
        launcher = mlrun.launcher.factory.LauncherFactory().create_launcher(
            is_remote=self._is_remote
        )
        return launcher.save_function(
            self, tag=tag, versioned=versioned, refresh=refresh
        )

    def doc(self):
        print("function:", self.metadata.name)
        print(self.spec.description)
        if self.spec.default_handler:
            print("default handler:", self.spec.default_handler)
        if self.spec.entry_points:
            print("entry points:")
            for name, entry in self.spec.entry_points.items():
                print(f"  {name}: {entry.get('doc', '')}")
                params = entry.get("parameters")
                if params:
                    for p in params:
                        line = p["name"]
                        if "type" in p:
                            line += f"({p['type']})"
                        line += "  - " + p.get("doc", "")
                        if "default" in p:
                            line += f", default={p['default']}"
                        print("    " + line)

    def skip_image_enrichment(self):
        return False
