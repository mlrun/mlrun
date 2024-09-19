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
import abc
import builtins
import http
import importlib.util as imputil
import os
import tempfile
import typing
import uuid

import mlrun_pipelines.common.models
import mlrun_pipelines.patcher
import mlrun_pipelines.utils

import mlrun
import mlrun.common.runtimes.constants
import mlrun.common.schemas
import mlrun.utils.notifications
from mlrun.errors import err_to_str
from mlrun.utils import (
    get_ui_url,
    logger,
    normalize_workflow_name,
    retry_until_successful,
)

from ..common.helpers import parse_versioned_object_uri
from ..config import config
from ..run import _run_pipeline, wait_for_pipeline_completion
from ..runtimes.pod import AutoMountType


def get_workflow_engine(engine_kind, local=False):
    if pipeline_context.is_run_local(local):
        if engine_kind == "kfp":
            logger.warning(
                "Running kubeflow pipeline locally, note some ops may not run locally!"
            )
        elif engine_kind == "remote":
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Cannot run a remote pipeline locally using `kind='remote'` and `local=True`. "
                "in order to run a local pipeline remotely, please use `engine='remote:local'` instead"
            )
        return _LocalRunner
    if not engine_kind or engine_kind == "kfp":
        return _KFPRunner
    if engine_kind == "local":
        return _LocalRunner
    if engine_kind == "remote":
        return _RemoteRunner
    raise mlrun.errors.MLRunInvalidArgumentError(
        f"Provided workflow engine is not supported. engine_kind={engine_kind}"
    )


class WorkflowSpec(mlrun.model.ModelObj):
    """workflow spec and helpers"""

    def __init__(
        self,
        engine: typing.Optional[str] = None,
        code: typing.Optional[str] = None,
        path: typing.Optional[str] = None,
        args: typing.Optional[dict] = None,
        name: typing.Optional[str] = None,
        handler: typing.Optional[str] = None,
        args_schema: typing.Optional[dict] = None,
        schedule: typing.Union[str, mlrun.common.schemas.ScheduleCronTrigger] = None,
        cleanup_ttl: typing.Optional[int] = None,
        image: typing.Optional[str] = None,
        workflow_runner_node_selector: typing.Optional[dict[str, str]] = None,
    ):
        self.engine = engine
        self.code = code
        self.path = path
        self.args = args
        self.name = name
        self.handler = handler
        self.cleanup_ttl = cleanup_ttl
        self.args_schema = args_schema
        self.run_local = False
        self._tmp_path = None
        self.schedule = schedule
        self.image = image
        self.workflow_runner_node_selector = workflow_runner_node_selector

    def get_source_file(self, context=""):
        if not self.code and not self.path:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "workflow must have code or path properties"
            )
        if self.code:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as workflow_fh:
                workflow_fh.write(self.code)
                self._tmp_path = workflow_path = workflow_fh.name
        else:
            workflow_path = self.path or ""
            if (
                context
                and not workflow_path.startswith("/")
                # since the user may provide a path the includes the context,
                # we need to make sure we don't add it twice
                and not workflow_path.startswith(context)
            ):
                workflow_path = os.path.join(context, workflow_path.lstrip("./"))
        return workflow_path

    def merge_args(self, extra_args):
        self.args = self.args or {}
        required = []
        if self.args_schema:
            for schema in self.args_schema:
                name = schema.get("name")
                if name not in self.args:
                    self.args[name] = schema.get("default")
                if schema.get("required"):
                    required.append(name)
        if extra_args:
            for k, v in extra_args.items():
                self.args[k] = v
                if k in required:
                    required.remove(k)
        if required:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Workflow argument(s) {','.join(required)} are required and were not specified"
            )

    def clear_tmp(self):
        if self._tmp_path:
            os.remove(self._tmp_path)


class FunctionsDict:
    """Virtual dictionary hosting the project functions, cached or in the DB"""

    def __init__(self, project, decorator=None):
        self.project = project
        self._decorator = decorator

    @property
    def _functions(self):
        return self.project.spec._function_objects

    def enrich(self, function, key):
        enriched_function = enrich_function_object(
            self.project, function, self._decorator
        )
        self._functions[key] = enriched_function  # update the cache
        return self._functions[key]

    def load_or_set_function(self, key, default=None) -> mlrun.runtimes.BaseRuntime:
        try:
            function = self.project.get_function(key, sync=False)
        except Exception as e:
            if not default:
                raise e
            function = default

        return self.enrich(function, key)

    def get(self, key, default=None) -> mlrun.runtimes.BaseRuntime:
        return self.load_or_set_function(key, default)

    def __getitem__(self, key) -> mlrun.runtimes.BaseRuntime:
        return self.load_or_set_function(key)

    def __setitem__(self, key, val):
        self._functions[key] = val

    def values(self):
        return [self.enrich(function, key) for key, function in self._functions.items()]

    def keys(self):
        return self._functions.keys()

    def items(self):
        return [
            (key, self.enrich(function, key))
            for key, function in self._functions.items()
        ]

    def __len__(self):
        return len(self._functions)

    def __iter__(self):
        yield from self._functions.keys()

    def __delitem__(self, key):
        del self._functions[key]


class _PipelineContext:
    """current (running) pipeline context"""

    def __init__(self):
        self.project = None
        self.workflow = None
        self.functions = FunctionsDict(None)
        self.workflow_id = None
        self.workflow_artifact_path = None
        self.runs_map = {}

    def is_run_local(self, local=None):
        if local is not None:
            # if the user specified an explicit value in local we use it
            return local
        force_run_local = mlrun.mlconf.force_run_local
        if force_run_local is None or force_run_local == "auto":
            force_run_local = not mlrun.mlconf.is_api_running_on_k8s()
            if not mlrun.mlconf.kfp_url:
                logger.debug("Kubeflow pipeline URL is not set, running locally")
                force_run_local = True

        if self.workflow:
            force_run_local = force_run_local or self.workflow.run_local

        return force_run_local

    def set(self, project, workflow=None):
        self.project = project
        self.workflow = workflow
        self.functions.project = project
        self.runs_map = {}

    def clear(self, with_project=False):
        if with_project:
            self.project = None
            self.functions.project = None
        self.workflow = None
        self.runs_map = {}
        self.workflow_id = None
        self.workflow_artifact_path = None

    def is_initialized(self, raise_exception=False):
        if self.project:
            return True
        if raise_exception:
            raise ValueError(
                "Pipeline context is not initialized, must be used inside a pipeline"
            )
        return False


pipeline_context = _PipelineContext()


def _set_function_attribute_on_kfp_pod(
    kfp_pod_template, function, pod_template_key, function_spec_key
):
    try:
        if kfp_pod_template.get("name").startswith(function.metadata.name):
            attribute_value = getattr(function.spec, function_spec_key, None)
            if attribute_value:
                kfp_pod_template[pod_template_key] = attribute_value
    except Exception as err:
        kfp_pod_name = kfp_pod_template.get("name")
        logger.warning(
            f"Unable to set function attribute on kfp pod {kfp_pod_name}",
            function_spec_key=function_spec_key,
            pod_template_key=pod_template_key,
            error=err_to_str(err),
        )


def _enrich_kfp_pod_security_context(kfp_pod_template, function):
    if (
        mlrun.runtimes.RuntimeKinds.is_local_runtime(function.kind)
        or mlrun.mlconf.function.spec.security_context.enrichment_mode
        == mlrun.common.schemas.SecurityContextEnrichmentModes.disabled.value
    ):
        return

    # ensure kfp pod user id is not None or 0 (root)
    if not mlrun.mlconf.function.spec.security_context.pipelines.kfp_pod_user_unix_id:
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"Kubeflow pipeline pod user id is invalid: "
            f"{mlrun.mlconf.function.spec.security_context.pipelines.kfp_pod_user_unix_id}, "
            f"it must be an integer greater than 0. "
            f"See mlrun.config.function.spec.security_context.pipelines.kfp_pod_user_unix_id for more details."
        )

    kfp_pod_user_unix_id = int(
        mlrun.mlconf.function.spec.security_context.pipelines.kfp_pod_user_unix_id
    )
    kfp_pod_template["SecurityContext"] = {
        "runAsUser": kfp_pod_user_unix_id,
        "runAsGroup": mlrun.mlconf.get_security_context_enrichment_group_id(
            kfp_pod_user_unix_id
        ),
    }


def get_db_function(project, key) -> mlrun.runtimes.BaseRuntime:
    project_instance, name, tag, hash_key = parse_versioned_object_uri(
        key, project.metadata.name
    )
    runtime = mlrun.get_run_db().get_function(name, project_instance, tag, hash_key)
    return mlrun.new_function(runtime=runtime)


def enrich_function_object(
    project, function, decorator=None, copy_function=True, try_auto_mount=True
) -> mlrun.runtimes.BaseRuntime:
    if hasattr(function, "_enriched"):
        return function
    f = function.copy() if copy_function else function
    f.metadata.project = project.metadata.name
    setattr(f, "_enriched", True)

    # set project default image if defined and function does not have an image specified
    if project.spec.default_image and not f.spec.image:
        f._enriched_image = True
        f.spec.image = project.spec.default_image

    src = f.spec.build.source
    if src and src in [".", "./"]:
        if not project.spec.source and not project.spec.mountdir:
            logger.warning(
                "project.spec.source should be specified when function is using code from project context"
            )

        if project.spec.mountdir:
            f.spec.workdir = project.spec.mountdir
            f.spec.build.source = ""
        else:
            f.spec.build.source = project.spec.source
            f.spec.build.load_source_on_run = project.spec.load_source_on_run
            f.spec.build.source_code_target_dir = (
                project.spec.build.source_code_target_dir
            )
            f.spec.workdir = project.spec.workdir or project.spec.subpath
            f.prepare_image_for_deploy()

    if project.spec.default_requirements:
        f.with_requirements(project.spec.default_requirements)
    if decorator:
        decorator(f)

    if project.spec.default_function_node_selector:
        f.enrich_runtime_spec(
            project.spec.default_function_node_selector,
        )

    if try_auto_mount:
        if (
            decorator and AutoMountType.is_auto_modifier(decorator)
        ) or project.spec.disable_auto_mount:
            f.spec.disable_auto_mount = True
        f.try_auto_mount_based_on_config()

    return f


class _PipelineRunStatus:
    """pipeline run result (status)"""

    def __init__(
        self,
        run_id: str,
        engine: type["_PipelineRunner"],
        project: "mlrun.projects.MlrunProject",
        workflow: WorkflowSpec = None,
        state: mlrun_pipelines.common.models.RunStatuses = "",
        exc: Exception = None,
    ):
        """
        :param run_id:      unique id of the pipeline run
        :param engine:      pipeline runner
        :param project:     mlrun project
        :param workflow:    workflow with spec on how to run the pipeline
        :param state:       the current state of the pipeline run
        :param exc:         exception that was raised during the pipeline run
        """
        self.run_id = run_id
        self.project = project
        self.workflow = workflow
        self._engine = engine
        self._state = state
        self._exc = exc

    @property
    def state(self):
        if (
            self._state
            not in mlrun_pipelines.common.models.RunStatuses.stable_statuses()
        ):
            self._state = self._engine.get_state(self.run_id, self.project)
        return self._state

    @property
    def exc(self):
        return self._exc

    def wait_for_completion(self, timeout=None, expected_statuses=None):
        returned_state = self._engine.wait_for_completion(
            self,
            project=self.project,
            timeout=timeout,
            expected_statuses=expected_statuses,
        )
        # TODO: returning a state is optional until all runners implement wait_for_completion
        if returned_state:
            self._state = returned_state
        return self._state

    def __str__(self):
        return str(self.run_id)

    def __repr__(self):
        return str(self.run_id)


class _PipelineRunner(abc.ABC):
    """abstract pipeline runner class"""

    engine = ""

    @classmethod
    @abc.abstractmethod
    def save(cls, project, workflow_spec: WorkflowSpec, target, artifact_path=None):
        raise NotImplementedError(
            f"Save operation not supported in {cls.engine} pipeline engine"
        )

    @classmethod
    @abc.abstractmethod
    def run(
        cls,
        project,
        workflow_spec: WorkflowSpec,
        name=None,
        workflow_handler=None,
        secrets=None,
        artifact_path=None,
        namespace=None,
        source=None,
        notifications: list[mlrun.model.Notification] = None,
    ) -> _PipelineRunStatus:
        pass

    @staticmethod
    @abc.abstractmethod
    def wait_for_completion(run_id, project=None, timeout=None, expected_statuses=None):
        pass

    @staticmethod
    @abc.abstractmethod
    def get_state(run_id, project=None):
        pass

    @staticmethod
    def get_run_status(
        project,
        run: _PipelineRunStatus,
        timeout=None,
        expected_statuses=None,
        notifiers: mlrun.utils.notifications.CustomNotificationPusher = None,
        **kwargs,
    ):
        timeout = timeout or 60 * 60
        raise_error = None
        state = ""
        try:
            if timeout:
                state = run.wait_for_completion(
                    timeout=timeout, expected_statuses=expected_statuses
                )
        except RuntimeError as exc:
            # push runs table also when we have errors
            raise_error = exc

        mldb = mlrun.db.get_run_db(secrets=project._secrets)
        runs = mldb.list_runs(project=project.name, labels=f"workflow={run.run_id}")

        # TODO: The below section duplicates notifiers.push_pipeline_run_results() logic. We should use it instead.
        errors_counter = 0
        for r in runs:
            if r["status"].get("state", "") == "error":
                errors_counter += 1

        text = _PipelineRunner._generate_workflow_finished_message(
            run.run_id, errors_counter, run._state
        )

        notifiers = notifiers or project.notifiers
        if notifiers:
            notifiers.push(text, "info", runs)

        if raise_error:
            raise raise_error
        return state or run._state, errors_counter, text

    @staticmethod
    def _get_handler(workflow_handler, workflow_spec, project, secrets):
        if not (workflow_handler and callable(workflow_handler)):
            workflow_file = workflow_spec.get_source_file(project.spec.get_code_path())
            workflow_handler = create_pipeline(
                project,
                workflow_file,
                pipeline_context.functions,
                secrets,
                handler=workflow_handler or workflow_spec.handler,
            )
        else:
            builtins.funcs = pipeline_context.functions
        return workflow_handler

    @staticmethod
    def _generate_workflow_finished_message(run_id, errors_counter, state):
        text = f"Workflow {run_id} finished"
        if errors_counter:
            text += f" with {errors_counter} errors"
        if state:
            text += f", state={state}"
        return text


class _KFPRunner(_PipelineRunner):
    """Kubeflow pipelines runner"""

    engine = "kfp"

    @classmethod
    def save(cls, project, workflow_spec: WorkflowSpec, target, artifact_path=None):
        pipeline_context.set(project, workflow_spec)
        workflow_file = workflow_spec.get_source_file(project.spec.get_code_path())
        functions = FunctionsDict(project)
        pipeline = create_pipeline(
            project,
            workflow_file,
            functions,
            secrets=project._secrets,
        )
        mlrun_pipelines.utils.compile_pipeline(
            artifact_path=artifact_path or project.spec.artifact_path,
            cleanup_ttl=workflow_spec.cleanup_ttl,
            ops=None,
            pipeline=pipeline,
            pipe_file=target,
            type_check=True,
        )
        workflow_spec.clear_tmp()
        pipeline_context.clear()

    @classmethod
    def run(
        cls,
        project: "mlrun.projects.MlrunProject",
        workflow_spec: WorkflowSpec,
        name=None,
        workflow_handler=None,
        secrets=None,
        artifact_path=None,
        namespace=None,
        source=None,
        notifications: list[mlrun.model.Notification] = None,
    ) -> _PipelineRunStatus:
        pipeline_context.set(project, workflow_spec)
        workflow_handler = _PipelineRunner._get_handler(
            workflow_handler, workflow_spec, project, secrets
        )
        if source:
            project.set_source(source=source)

        namespace = namespace or config.namespace

        # fallback to old notification behavior
        if notifications:
            logger.warning(
                "Setting notifications on kfp pipeline runner uses old notification behavior. "
                "Notifications will only be sent if you wait for pipeline completion. "
                "To use the new notification behavior, use the remote pipeline runner."
            )
            # for start message, fallback to old notification behavior
            for notification in notifications or []:
                project.notifiers.add_notification(
                    notification.kind, notification.params
                )

        run_id = _run_pipeline(
            workflow_handler,
            project=project.metadata.name,
            arguments=workflow_spec.args,
            experiment=name or workflow_spec.name,
            namespace=namespace,
            artifact_path=artifact_path,
            cleanup_ttl=workflow_spec.cleanup_ttl,
            timeout=int(mlrun.mlconf.workflows.timeouts.kfp),
        )

        # The user provided workflow code might have made changes to function specs that require cleanup
        for func in project.spec._function_objects.values():
            try:
                func.spec.discard_changes()
            except AttributeError:
                logger.debug(
                    "Function does not require a field rollback", func_type=type(func)
                )
            except Exception as exc:
                logger.warning(
                    "Failed to rollback spec fields for function",
                    project=project,
                    func_name=func.metadata.name,
                    exc_info=err_to_str(exc),
                )
        project.notifiers.push_pipeline_start_message(
            project.metadata.name,
            project.get_param("commit_id", None),
            run_id,
            True,
        )
        pipeline_context.clear()
        return _PipelineRunStatus(run_id, cls, project=project, workflow=workflow_spec)

    @staticmethod
    def wait_for_completion(run, project=None, timeout=None, expected_statuses=None):
        logger.info(
            "Waiting for pipeline run completion", run_id=run.run_id, project=project
        )
        timeout = timeout or 60 * 60
        project_name = project.metadata.name if project else ""
        run_info = wait_for_pipeline_completion(
            run.run_id,
            timeout=timeout,
            expected_statuses=expected_statuses,
            project=project_name,
        )
        status = ""
        if run_info:
            status = run_info["run"].get("status")
        return status

    @staticmethod
    def get_state(run_id, project=None):
        project_name = project.metadata.name if project else ""
        resp = mlrun.run.get_pipeline(run_id, project=project_name)
        if resp:
            return resp["run"].get("status", "")
        return ""


class _LocalRunner(_PipelineRunner):
    """local pipelines runner"""

    engine = "local"

    @classmethod
    def run(
        cls,
        project,
        workflow_spec: WorkflowSpec,
        name=None,
        workflow_handler=None,
        secrets=None,
        artifact_path=None,
        namespace=None,
        source=None,
        notifications: list[mlrun.model.Notification] = None,
    ) -> _PipelineRunStatus:
        pipeline_context.set(project, workflow_spec)
        workflow_handler = _PipelineRunner._get_handler(
            workflow_handler, workflow_spec, project, secrets
        )

        # fallback to old notification behavior
        for notification in notifications or []:
            project.notifiers.add_notification(notification.kind, notification.params)

        workflow_id = uuid.uuid4().hex
        pipeline_context.workflow_id = workflow_id
        # When using KFP, it would do this replacement. When running locally, we need to take care of it.
        if artifact_path:
            artifact_path = artifact_path.replace("{{workflow.uid}}", workflow_id)
        original_source = None
        if source:
            original_source = project.spec.source
            project.set_source(source=source)
        pipeline_context.workflow_artifact_path = artifact_path

        project.notifiers.push_pipeline_start_message(
            project.metadata.name, pipeline_id=workflow_id
        )
        err = None
        try:
            workflow_handler(**workflow_spec.args)
            state = mlrun_pipelines.common.models.RunStatuses.succeeded
        except Exception as exc:
            err = exc
            logger.exception("Workflow run failed")
            project.notifiers.push(
                f":x: Workflow {workflow_id} run failed!, error: {err_to_str(exc)}",
                mlrun.common.schemas.NotificationSeverity.ERROR,
            )
            state = mlrun_pipelines.common.models.RunStatuses.failed
        mlrun.run.wait_for_runs_completion(pipeline_context.runs_map.values())
        project.notifiers.push_pipeline_run_results(
            pipeline_context.runs_map.values(), state=state
        )
        pipeline_context.clear()

        # Setting the source back to the original in the project object
        if original_source:
            project.set_source(source=original_source)
        return _PipelineRunStatus(
            workflow_id,
            cls,
            project=project,
            workflow=workflow_spec,
            state=state,
            exc=err,
        )

    @staticmethod
    def get_state(run_id, project=None):
        return ""

    @staticmethod
    def wait_for_completion(run, project=None, timeout=None, expected_statuses=None):
        # TODO: local runner blocks for the duration of the pipeline.
        # Therefore usually there will be nothing to wait for.
        # However, users may run functions with watch=False and then it can be useful to wait for the runs here.
        pass


class _RemoteRunner(_PipelineRunner):
    """remote pipelines runner"""

    engine = "remote"

    @classmethod
    def run(
        cls,
        project: "mlrun.projects.MlrunProject",
        workflow_spec: WorkflowSpec,
        name: str = None,
        workflow_handler: typing.Union[str, typing.Callable] = None,
        secrets: mlrun.secrets.SecretsStore = None,
        artifact_path: str = None,
        namespace: str = None,
        source: str = None,
        notifications: list[mlrun.model.Notification] = None,
    ) -> typing.Optional[_PipelineRunStatus]:
        workflow_name = normalize_workflow_name(name=name, project_name=project.name)
        workflow_id = None

        # The returned engine for this runner is the engine of the workflow.
        # In this way wait_for_completion/get_run_status would be executed by the correct pipeline runner.
        inner_engine = get_workflow_engine(workflow_spec.engine)
        run_db = mlrun.get_run_db()
        err = None
        try:
            logger.info(
                "Submitting remote workflow",
                workflow_engine=workflow_spec.engine,
                schedule=workflow_spec.schedule,
                project_name=project.name,
            )

            # set it relative to project path
            # as the runner pod will mount and use `load_and_run` which will use the project context
            # to load the workflow file to.
            # e.g.
            # /path/to/project/workflow.py -> ./workflow.py
            # /path/to/project/subdir/workflow.py -> ./workflow.py
            if workflow_spec.path:
                prefix = project.spec.get_code_path()
                if workflow_spec.path.startswith(prefix):
                    workflow_spec.path = workflow_spec.path.removeprefix(prefix)
                    relative_prefix = "."
                    if not workflow_spec.path.startswith("/"):
                        relative_prefix += "/"
                    workflow_spec.path = f"{relative_prefix}{workflow_spec.path}"

            workflow_response = run_db.submit_workflow(
                project=project.name,
                name=workflow_name,
                workflow_spec=workflow_spec,
                artifact_path=artifact_path,
                source=source,
                run_name=config.workflows.default_workflow_runner_name.format(
                    workflow_name
                ),
                namespace=namespace,
                notifications=notifications,
            )
            if workflow_spec.schedule:
                logger.info(
                    "Workflow scheduled successfully",
                    workflow_response=workflow_response,
                )
                return

            get_workflow_id_timeout = max(
                int(mlrun.mlconf.workflows.timeouts.remote),
                int(getattr(mlrun.mlconf.workflows.timeouts, inner_engine.engine)),
            )

            logger.debug(
                "Workflow submitted, waiting for pipeline run to start",
                workflow_name=workflow_response.name,
                get_workflow_id_timeout=get_workflow_id_timeout,
            )

            def _get_workflow_id_or_bail():
                try:
                    return run_db.get_workflow_id(
                        project=project.name,
                        name=workflow_response.name,
                        run_id=workflow_response.run_id,
                        engine=workflow_spec.engine,
                    )
                except mlrun.errors.MLRunHTTPStatusError as get_wf_exc:
                    # fail fast on specific errors
                    if get_wf_exc.error_status_code in [
                        http.HTTPStatus.PRECONDITION_FAILED
                    ]:
                        raise mlrun.errors.MLRunFatalFailureError(
                            original_exception=get_wf_exc
                        )

                    # raise for a retry (on other errors)
                    raise

            # Getting workflow id from run:
            response = retry_until_successful(
                1,
                get_workflow_id_timeout,
                logger,
                False,
                _get_workflow_id_or_bail,
            )
            workflow_id = response.workflow_id
            # After fetching the workflow_id the workflow executed successfully

        except Exception as exc:
            err = exc
            logger.exception("Workflow run failed")
            project.notifiers.push(
                f":x: Workflow {workflow_name} run failed!, error: {err_to_str(exc)}",
                mlrun.common.schemas.NotificationSeverity.ERROR,
            )
            state = mlrun_pipelines.common.models.RunStatuses.failed
        else:
            state = mlrun_pipelines.common.models.RunStatuses.running
            pipeline_context.clear()
        return _PipelineRunStatus(
            run_id=workflow_id,
            engine=inner_engine,
            project=project,
            workflow=workflow_spec,
            state=state,
            exc=err,
        )

    @staticmethod
    def get_run_status(
        project,
        run: _PipelineRunStatus,
        timeout=None,
        expected_statuses=None,
        notifiers: mlrun.utils.notifications.CustomNotificationPusher = None,
        inner_engine: type[_PipelineRunner] = None,
    ):
        inner_engine = inner_engine or _KFPRunner
        if inner_engine.engine == _KFPRunner.engine:
            # ignore notifiers for remote notifications, as they are handled by the remote pipeline notifications,
            # so overriding with CustomNotificationPusher with empty list of notifiers or only local notifiers
            local_project_notifiers = list(
                set(mlrun.utils.notifications.NotificationTypes.local()).intersection(
                    set(project.notifiers.notifications.keys())
                )
            )
            notifiers = mlrun.utils.notifications.CustomNotificationPusher(
                local_project_notifiers
            )
            return _KFPRunner.get_run_status(
                project,
                run,
                timeout,
                expected_statuses,
                notifiers=notifiers,
            )

        elif inner_engine.engine == _LocalRunner.engine:
            mldb = mlrun.db.get_run_db(secrets=project._secrets)
            pipeline_runner_run = mldb.read_run(run.run_id, project=project.name)

            pipeline_runner_run = mlrun.run.RunObject.from_dict(pipeline_runner_run)

            # here we are waiting for the pipeline run to complete and refreshing after that the pipeline run from the
            # db
            # TODO: do it with timeout
            pipeline_runner_run.logs(db=mldb)
            pipeline_runner_run.refresh()
            run._state = mlrun.common.runtimes.constants.RunStates.run_state_to_pipeline_run_status(
                pipeline_runner_run.status.state
            )
            run._exc = pipeline_runner_run.status.error
            return _LocalRunner.get_run_status(
                project,
                run,
                timeout,
                expected_statuses,
                notifiers=notifiers,
            )

        else:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Unsupported inner runner engine: {inner_engine.engine}"
            )


def create_pipeline(project, pipeline, functions, secrets=None, handler=None):
    spec = imputil.spec_from_file_location("workflow", pipeline)
    if spec is None:
        raise ImportError(f"Cannot import workflow {pipeline}")
    mod = imputil.module_from_spec(spec)
    spec.loader.exec_module(mod)

    setattr(mod, "funcs", functions)  # should be replaced with "functions" in future
    setattr(mod, "functions", functions)
    setattr(mod, "this_project", project)

    # verify all functions are in this project (init_functions may add new functions)
    for f in functions.values():
        f.metadata.project = project.metadata.name

    if not handler and hasattr(mod, "kfpipeline"):
        handler = "kfpipeline"
    if not handler and hasattr(mod, "pipeline"):
        handler = "pipeline"
    if not handler or not hasattr(mod, handler):
        raise ValueError(
            f"'workflow_handler' is not defined. "
            f"Either provide it as set_workflow argument, or include a function named"
            f" '{handler or 'pipeline'}' in your workflow .py file."
        )

    return getattr(mod, handler)


def github_webhook(request):
    signature = request.headers.get("X-Hub-Signature")
    data = request.data
    print("sig:", signature)
    print("headers:", request.headers)
    print("data:", data)
    print("json:", request.get_json())

    if request.headers.get("X-GitHub-Event") == "ping":
        return {"msg": "Ok"}

    return {"msg": "pushed"}


def load_and_run(
    context: mlrun.execution.MLClientCtx,
    url: str = None,
    project_name: str = "",
    init_git: bool = None,
    subpath: str = None,
    clone: bool = False,
    save: bool = True,
    workflow_name: str = None,
    workflow_path: str = None,
    workflow_arguments: dict[str, typing.Any] = None,
    artifact_path: str = None,
    workflow_handler: typing.Union[str, typing.Callable] = None,
    namespace: str = None,
    sync: bool = False,
    dirty: bool = False,
    engine: str = None,
    local: bool = None,
    schedule: typing.Union[str, mlrun.common.schemas.ScheduleCronTrigger] = None,
    cleanup_ttl: int = None,
    load_only: bool = False,
    wait_for_completion: bool = False,
    project_context: str = None,
):
    """
    Auxiliary function that the RemoteRunner run once or run every schedule.
    This function loads a project from a given remote source and then runs the workflow.

    :param context:             mlrun context.
    :param url:                 remote url that represents the project's source.
                                See 'mlrun.load_project()' for details
    :param project_name:        project name
    :param init_git:            if True, will git init the context dir
    :param subpath:             project subpath (within the archive)
    :param clone:               if True, always clone (delete any existing content)
    :param save:                whether to save the created project and artifact in the DB
    :param workflow_name:       name of the workflow
    :param workflow_path:       url to a workflow file, if not a project workflow
    :param workflow_arguments:  kubeflow pipelines arguments (parameters)
    :param artifact_path:       target path/url for workflow artifacts, the string
                                '{{workflow.uid}}' will be replaced by workflow id
    :param workflow_handler:    workflow function handler (for running workflow function directly)
    :param namespace:           kubernetes namespace if other than default
    :param sync:                force functions sync before run
    :param dirty:               allow running the workflow when the git repo is dirty
    :param engine:              workflow engine running the workflow.
                                supported values are 'kfp' (default) or 'local'
    :param local:               run local pipeline with local functions (set local=True in function.run())
    :param schedule:            ScheduleCronTrigger class instance or a standard crontab expression string
    :param cleanup_ttl:         pipeline cleanup ttl in secs (time to wait after workflow completion, at which point the
                                workflow and all its resources are deleted)
    :param load_only:           for just loading the project, inner use.
    :param wait_for_completion: wait for workflow completion before returning
    :param project_context:     project context path (used for loading the project)
    """
    try:
        project = mlrun.load_project(
            context=project_context or f"./{project_name}",
            url=url,
            name=project_name,
            init_git=init_git,
            subpath=subpath,
            clone=clone,
            save=save,
            sync_functions=True,
        )
    except Exception as error:
        if schedule:
            notification_pusher = mlrun.utils.notifications.CustomNotificationPusher(
                ["slack"]
            )
            url = get_ui_url(project_name, context.uid)
            link = f"<{url}|*view workflow job details*>"
            message = (
                f":x: Failed to run scheduled workflow {workflow_name} in Project {project_name} !\n"
                f"error: ```{error}```\n{link}"
            )
            # Sending Slack Notification without losing the original error:
            try:
                notification_pusher.push(
                    message=message,
                    severity=mlrun.common.schemas.NotificationSeverity.ERROR,
                )

            except Exception as exc:
                logger.error("Failed to send slack notification", exc=err_to_str(exc))

        raise error

    context.logger.info(f"Loaded project {project.name} successfully")

    if load_only:
        return

    # extract "start" notification if exists
    start_notifications = [
        notification
        for notification in context.get_notifications()
        if "running" in notification.when
    ]

    workflow_log_message = workflow_name or workflow_path
    context.logger.info(f"Running workflow {workflow_log_message} from remote")
    run = project.run(
        name=workflow_name,
        workflow_path=workflow_path,
        arguments=workflow_arguments,
        artifact_path=artifact_path,
        workflow_handler=workflow_handler,
        namespace=namespace,
        sync=sync,
        watch=False,  # Required for fetching the workflow_id
        dirty=dirty,
        cleanup_ttl=cleanup_ttl,
        engine=engine,
        local=local,
        notifications=start_notifications,
    )
    context.log_result(key="workflow_id", value=run.run_id)
    context.log_result(key="engine", value=run._engine.engine, commit=True)

    if run.state == mlrun_pipelines.common.models.RunStatuses.failed:
        raise RuntimeError(f"Workflow {workflow_log_message} failed") from run.exc

    if wait_for_completion:
        try:
            run.wait_for_completion()
        except Exception as exc:
            logger.error(
                "Failed waiting for workflow completion",
                workflow=workflow_log_message,
                exc=err_to_str(exc),
            )

        pipeline_state, _, _ = project.get_run_status(run)
        context.log_result(key="workflow_state", value=pipeline_state, commit=True)
        if pipeline_state != mlrun_pipelines.common.models.RunStatuses.succeeded:
            raise RuntimeError(
                f"Workflow {workflow_log_message} failed, state={pipeline_state}"
            )
