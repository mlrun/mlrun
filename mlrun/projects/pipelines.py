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
import abc
import builtins
import importlib.util as imputil
import os
import tempfile
import time
import traceback
import typing
import uuid
import warnings

import kfp.compiler
from kfp import dsl
from kfp.compiler import compiler

import mlrun
import mlrun.api.schemas
import mlrun.utils.notifications
from mlrun.errors import err_to_str
from mlrun.utils import (
    get_ui_url,
    logger,
    new_pipe_metadata,
    parse_versioned_object_uri,
)

from ..config import config
from ..run import run_pipeline, wait_for_pipeline_completion
from ..runtimes.pod import AutoMountType


def get_workflow_engine(engine_kind, local=False):
    if pipeline_context.is_run_local(local):
        if engine_kind == "kfp":
            logger.warning(
                "running kubeflow pipeline locally, note some ops may not run locally!"
            )
        elif engine_kind == "remote":
            raise mlrun.errors.MLRunInvalidArgumentError(
                "cannot run a remote pipeline locally using `kind='remote'` and `local=True`. "
                "in order to run a local pipeline remotely, please use `engine='remote: local'` instead"
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
        engine=None,
        code=None,
        path=None,
        args=None,
        name=None,
        handler=None,
        # TODO: deprecated, remove in 1.5.0
        ttl=None,
        args_schema: dict = None,
        schedule: typing.Union[str, mlrun.api.schemas.ScheduleCronTrigger] = None,
        override: bool = None,
        cleanup_ttl: int = None,
    ):
        if ttl:
            warnings.warn(
                "'ttl' is deprecated, use 'cleanup_ttl' instead. "
                "This will be removed in 1.5.0",
                # TODO: Remove this in 1.5.0
                FutureWarning,
            )

        self.engine = engine
        self.code = code
        self.path = path
        self.args = args
        self.name = name
        self.handler = handler
        self.ttl = cleanup_ttl or ttl
        self.cleanup_ttl = cleanup_ttl or ttl
        self.args_schema = args_schema
        self.run_local = False
        self._tmp_path = None
        self.schedule = schedule
        self.override = override

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
            if context and not workflow_path.startswith("/"):
                workflow_path = os.path.join(context, workflow_path)
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
                f"workflow argument(s) {','.join(required)} are required and were not specified"
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
        return {
            key: self.enrich(function, key) for key, function in self._functions.items()
        }

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
            kfp_url = mlrun.mlconf.resolve_kfp_url()
            if not kfp_url:
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
                "pipeline context is not initialized, must be used inside a pipeline"
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
        == mlrun.api.schemas.SecurityContextEnrichmentModes.disabled.value
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


# When we run pipelines, the kfp.compile.Compile.compile() method takes the decorated function with @dsl.pipeline and
# converts it to a k8s object. As part of the flow in the Compile.compile() method,
# we call _create_and_write_workflow, which builds a dictionary from the workflow and then writes it to a file.
# Unfortunately, the kfp sdk does not provide an API for configuring priority_class_name and other attributes.
# I ran across the following problem when seeking for a method to set the priority_class_name:
# https://github.com/kubeflow/pipelines/issues/3594
# When we patch the _create_and_write_workflow, we can eventually obtain the dictionary right before we write it
# to a file and enrich it with argo compatible fields, make sure you looking for the same argo version we use
# https://github.com/argoproj/argo-workflows/blob/release-2.7/pkg/apis/workflow/v1alpha1/workflow_types.go
def _create_enriched_mlrun_workflow(
    self,
    pipeline_func: typing.Callable,
    pipeline_name: typing.Optional[typing.Text] = None,
    pipeline_description: typing.Optional[typing.Text] = None,
    params_list: typing.Optional[typing.List[dsl.PipelineParam]] = None,
    pipeline_conf: typing.Optional[dsl.PipelineConf] = None,
):
    """Call internal implementation of create_workflow and enrich with mlrun functions attributes"""
    workflow = self._original_create_workflow(
        pipeline_func, pipeline_name, pipeline_description, params_list, pipeline_conf
    )
    # We don't want to interrupt the original flow and don't know all the scenarios the function could be called.
    # that's why we have try/except on all the code of the enrichment and also specific try/except for errors that
    # we know can be raised.
    try:
        functions = []
        if pipeline_context.functions:
            try:
                functions = pipeline_context.functions.values()
            except Exception as err:
                logger.debug(
                    "Unable to retrieve project functions, not enriching workflow with mlrun",
                    error=err_to_str(err),
                )
                return workflow

        # enrich each pipeline step with your desire k8s attribute
        for kfp_step_template in workflow["spec"]["templates"]:
            if kfp_step_template.get("container"):
                for function_obj in functions:
                    # we condition within each function since the comparison between the function and
                    # the kfp pod may change depending on the attribute type.
                    _set_function_attribute_on_kfp_pod(
                        kfp_step_template,
                        function_obj,
                        "PriorityClassName",
                        "priority_class_name",
                    )
                    _enrich_kfp_pod_security_context(
                        kfp_step_template,
                        function_obj,
                    )
    except mlrun.errors.MLRunInvalidArgumentError:
        raise
    except Exception as err:
        logger.debug(
            "Something in the enrichment of kfp pods failed", error=err_to_str(err)
        )
    return workflow


# patching function as class method
kfp.compiler.Compiler._original_create_workflow = kfp.compiler.Compiler._create_workflow
kfp.compiler.Compiler._create_workflow = _create_enriched_mlrun_workflow


def get_db_function(project, key) -> mlrun.runtimes.BaseRuntime:
    project_instance, name, tag, hash_key = parse_versioned_object_uri(
        key, project.metadata.name
    )
    runtime = mlrun.get_run_db().get_function(name, project_instance, tag, hash_key)
    return mlrun.new_function(runtime=runtime)


def enrich_function_object(
    project, function, decorator=None, copy_function=True
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
            f.spec.workdir = project.spec.workdir or project.spec.subpath
            f.verify_base_image()

    if project.spec.default_requirements:
        f.with_requirements(project.spec.default_requirements)
    if decorator:
        decorator(f)

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
        engine: typing.Type["_PipelineRunner"],
        project: "mlrun.projects.MlrunProject",
        workflow: WorkflowSpec = None,
        state: str = "",
    ):
        """
        :param run_id:      unique id of the pipeline run
        :param engine:      pipeline runner
        :param project:     mlrun project
        :param workflow:    workflow with spec on how to run the pipeline
        :param state:       the current state of the pipeline run
        """
        self.run_id = run_id
        self.project = project
        self.workflow = workflow
        self._engine = engine
        self._state = state

    @property
    def state(self):
        if self._state not in mlrun.run.RunStatuses.stable_statuses():
            self._state = self._engine.get_state(self.run_id, self.project)
        return self._state

    def wait_for_completion(self, timeout=None, expected_statuses=None):
        self._state = self._engine.wait_for_completion(
            self.run_id,
            project=self.project,
            timeout=timeout,
            expected_statuses=expected_statuses,
        )
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
            f"save operation not supported in {cls.engine} pipeline engine"
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
    def _get_handler(workflow_handler, workflow_spec, project, secrets):
        if not (workflow_handler and callable(workflow_handler)):
            workflow_file = workflow_spec.get_source_file(project.spec.context)
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
    @abc.abstractmethod
    def get_run_status(
        project,
        run,
        timeout=None,
        expected_statuses=None,
        notifiers: mlrun.utils.notifications.CustomNotificationPusher = None,
    ):
        pass


class _KFPRunner(_PipelineRunner):
    """Kubeflow pipelines runner"""

    engine = "kfp"

    @classmethod
    def save(cls, project, workflow_spec: WorkflowSpec, target, artifact_path=None):
        pipeline_context.set(project, workflow_spec)
        workflow_file = workflow_spec.get_source_file(project.spec.context)
        functions = FunctionsDict(project)
        pipeline = create_pipeline(
            project,
            workflow_file,
            functions,
            secrets=project._secrets,
        )
        artifact_path = artifact_path or project.spec.artifact_path

        conf = new_pipe_metadata(
            artifact_path=artifact_path,
            cleanup_ttl=workflow_spec.cleanup_ttl or workflow_spec.ttl,
        )
        compiler.Compiler().compile(pipeline, target, pipeline_conf=conf)
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
    ) -> _PipelineRunStatus:
        pipeline_context.set(project, workflow_spec)
        workflow_handler = _PipelineRunner._get_handler(
            workflow_handler, workflow_spec, project, secrets
        )
        if source:
            project.set_source(source=source)

        namespace = namespace or config.namespace
        id = run_pipeline(
            workflow_handler,
            project=project.metadata.name,
            arguments=workflow_spec.args,
            experiment=name or workflow_spec.name,
            namespace=namespace,
            artifact_path=artifact_path,
            cleanup_ttl=workflow_spec.cleanup_ttl or workflow_spec.ttl,
        )
        project.notifiers.push_pipeline_start_message(
            project.metadata.name,
            project.get_param("commit_id", None),
            id,
            True,
        )
        pipeline_context.clear()
        return _PipelineRunStatus(id, cls, project=project, workflow=workflow_spec)

    @staticmethod
    def wait_for_completion(run_id, project=None, timeout=None, expected_statuses=None):
        if timeout is None:
            timeout = 60 * 60
        project_name = project.metadata.name if project else ""
        run_info = wait_for_pipeline_completion(
            run_id,
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

    @staticmethod
    def get_run_status(
        project,
        run,
        timeout=None,
        expected_statuses=None,
        notifiers: mlrun.utils.notifications.CustomNotificationPusher = None,
    ):
        if timeout is None:
            timeout = 60 * 60
        state = ""
        raise_error = None
        try:
            if timeout:
                logger.info("waiting for pipeline run completion")
                state = run.wait_for_completion(
                    timeout=timeout, expected_statuses=expected_statuses
                )
        except RuntimeError as exc:
            # push runs table also when we have errors
            raise_error = exc

        mldb = mlrun.db.get_run_db(secrets=project._secrets)
        runs = mldb.list_runs(project=project.name, labels=f"workflow={run.run_id}")

        had_errors = 0
        for r in runs:
            if r["status"].get("state", "") == "error":
                had_errors += 1

        text = f"Workflow {run.run_id} finished"
        if had_errors:
            text += f" with {had_errors} errors"
        if state:
            text += f", state={state}"

        notifiers = notifiers or project.notifiers
        notifiers.push(text, "info", runs)

        if raise_error:
            raise raise_error
        return state, had_errors, text


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
    ) -> _PipelineRunStatus:
        pipeline_context.set(project, workflow_spec)
        workflow_handler = _PipelineRunner._get_handler(
            workflow_handler, workflow_spec, project, secrets
        )

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
        try:
            workflow_handler(**workflow_spec.args)
            state = mlrun.run.RunStatuses.succeeded
        except Exception as e:
            trace = traceback.format_exc()
            logger.error(trace)
            project.notifiers.push(
                f"Workflow {workflow_id} run failed!, error: {e}\n{trace}", "error"
            )
            state = mlrun.run.RunStatuses.failed
        mlrun.run.wait_for_runs_completion(pipeline_context.runs_map.values())
        project.notifiers.push_pipeline_run_results(
            pipeline_context.runs_map.values(), state=state
        )
        pipeline_context.clear()

        # Setting the source back to the original in the project object
        if original_source:
            project.set_source(source=original_source)
        return _PipelineRunStatus(
            workflow_id, cls, project=project, workflow=workflow_spec, state=state
        )

    @staticmethod
    def get_state(run_id, project=None):
        return ""

    @staticmethod
    def wait_for_completion(run_id, project=None, timeout=None, expected_statuses=None):
        pass

    @staticmethod
    def get_run_status(
        project,
        run,
        timeout=None,
        expected_statuses=None,
        notifiers: mlrun.utils.notifications.CustomNotificationPusher = None,
    ):
        pass


class _RemoteRunner(_PipelineRunner):
    """remote pipelines runner"""

    engine = "remote"

    @staticmethod
    def _prepare_load_and_run_function(
        source: str,
        project_name: str,
        save: bool,
        workflow_name: str,
        workflow_spec: WorkflowSpec,
        artifact_path: str,
        workflow_handler: str,
        namespace: str,
    ) -> typing.Tuple[mlrun.runtimes.RemoteRuntime, "mlrun.RunObject"]:
        """
        Helper function for creating the runspec of the load and run function.
        For internal use only.
        :param source:              The source of the project.
        :param project_name:        project name
        :param save:                either to save the project in the DB
        :param workflow_name:       workflow name
        :param workflow_spec:       workflow to run
        :param artifact_path:       path to store artifacts
        :param workflow_handler:    workflow function handler (for running workflow function directly)
        :param namespace:           kubernetes namespace if other than default
        :return:
        """
        # Creating the load project and workflow running function:
        load_and_run_fn = mlrun.new_function(
            name=mlrun.mlconf.default_workflow_runner_name.format(workflow_name),
            project=project_name,
            kind="job",
            image=mlrun.mlconf.default_base_image,
        )
        runspec = mlrun.RunObject(
            spec=mlrun.model.RunSpec(
                parameters={
                    "url": source,
                    "project_name": project_name,
                    "save": save,
                    "workflow_name": workflow_name or workflow_spec.name,
                    "workflow_path": workflow_spec.path,
                    "workflow_arguments": workflow_spec.args,
                    "artifact_path": artifact_path,
                    "workflow_handler": workflow_handler or workflow_spec.handler,
                    "namespace": namespace,
                    "ttl": workflow_spec.cleanup_ttl or workflow_spec.ttl,
                    "engine": workflow_spec.engine,
                    "local": workflow_spec.run_local,
                    "schedule": workflow_spec.schedule,
                },
                handler="mlrun.projects.load_and_run",
            ),
            metadata=mlrun.model.RunMetadata(name=workflow_name),
        )

        runspec = runspec.set_label("job-type", "workflow-runner").set_label(
            "workflow", workflow_name
        )
        return load_and_run_fn, runspec

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
    ) -> typing.Optional[_PipelineRunStatus]:
        workflow_name = name.split("-")[-1] if f"{project.name}-" in name else name

        run_id = None

        # If the user provided a source we want to load the project from the source
        # (like from a specific commit/branch from git repo) without changing the source of the project (save=False).
        save, current_source = (
            (False, source) if source else (True, project.spec.source)
        )
        if "://" not in current_source:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"remote workflows can only be performed by a project with remote source,"
                f" the given source '{current_source}' is not remote"
            )

        # Creating the load project and workflow running function:
        load_and_run_fn, runspec = cls._prepare_load_and_run_function(
            source=current_source,
            project_name=project.name,
            save=save,
            workflow_name=workflow_name,
            workflow_spec=workflow_spec,
            artifact_path=artifact_path,
            workflow_handler=workflow_handler,
            namespace=namespace,
        )

        if workflow_spec.schedule:
            is_scheduled = True
            schedule_name = runspec.spec.parameters.get("workflow_name")
            run_db = mlrun.get_run_db()

            try:
                run_db.get_schedule(project.name, schedule_name)
            except mlrun.errors.MLRunNotFoundError:
                is_scheduled = False

            if workflow_spec.override:
                if is_scheduled:
                    logger.info(f"Deleting schedule {schedule_name}")
                    run_db.delete_schedule(project.name, schedule_name)
                else:
                    logger.info(
                        f"No schedule by name '{schedule_name}' was found, nothing to override."
                    )
            elif is_scheduled:
                raise mlrun.errors.MLRunConflictError(
                    f"There is already a schedule for workflow {schedule_name}."
                    " If you want to override this schedule use 'override=True' (SDK) or '--override-workflow' (CLI)"
                )

        # The returned engine for this runner is the engine of the workflow.
        # In this way wait_for_completion/get_run_status would be executed by the correct pipeline runner.
        inner_engine = get_workflow_engine(workflow_spec.engine)

        msg = "executing workflow"
        if workflow_spec.schedule:
            msg += " scheduling"
        logger.info(
            f"{msg} '{load_and_run_fn.metadata.name}' remotely with {workflow_spec.engine} engine"
        )

        try:
            run = load_and_run_fn.run(
                runspec=runspec,
                local=False,
                schedule=workflow_spec.schedule,
                artifact_path=artifact_path,
            )
            if workflow_spec.schedule:
                return
            # Fetching workflow id:
            while not run_id:
                run.refresh()
                run_id = run.status.results.get("workflow_id", None)
                time.sleep(1)
            # After fetching the workflow_id the workflow executed successfully
            state = mlrun.run.RunStatuses.succeeded

        except Exception as e:
            trace = traceback.format_exc()
            logger.error(trace)
            project.notifiers.push(
                f"Workflow {workflow_name} run failed!, error: {e}\n{trace}", "error"
            )
            state = mlrun.run.RunStatuses.failed
            return _PipelineRunStatus(
                run_id,
                inner_engine,
                project=project,
                workflow=workflow_spec,
                state=state,
            )

        project.notifiers.push_pipeline_start_message(
            project.metadata.name,
        )
        pipeline_context.clear()
        return _PipelineRunStatus(
            run_id,
            inner_engine,
            project=project,
            workflow=workflow_spec,
            state=state,
        )


def create_pipeline(project, pipeline, functions, secrets=None, handler=None):
    spec = imputil.spec_from_file_location("workflow", pipeline)
    if spec is None:
        raise ImportError(f"cannot import workflow {pipeline}")
    mod = imputil.module_from_spec(spec)
    spec.loader.exec_module(mod)

    setattr(mod, "funcs", functions)  # should be replaced with "functions" in future
    setattr(mod, "functions", functions)
    setattr(mod, "this_project", project)

    if hasattr(mod, "init_functions"):

        # TODO: remove in 1.5.0
        warnings.warn(
            "'init_functions' is deprecated in 1.3.0 and will be removed in 1.5.0. "
            "Place function initialization in the pipeline code.",
            FutureWarning,
        )
        getattr(mod, "init_functions")(functions, project, secrets)

    # verify all functions are in this project (init_functions may add new functions)
    for f in functions.values():
        f.metadata.project = project.metadata.name

    if not handler and hasattr(mod, "kfpipeline"):
        handler = "kfpipeline"
    if not handler and hasattr(mod, "pipeline"):
        handler = "pipeline"
    if not handler or not hasattr(mod, handler):
        raise ValueError(f"pipeline function ({handler or 'pipeline'}) not found")

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
    context,
    url: str = None,
    project_name: str = "",
    init_git: bool = None,
    subpath: str = None,
    clone: bool = False,
    save: bool = True,
    workflow_name: str = None,
    workflow_path: str = None,
    workflow_arguments: typing.Dict[str, typing.Any] = None,
    artifact_path: str = None,
    workflow_handler: typing.Union[str, typing.Callable] = None,
    namespace: str = None,
    sync: bool = False,
    dirty: bool = False,
    # TODO: deprecated, remove in 1.5.0
    ttl: int = None,
    engine: str = None,
    local: bool = None,
    schedule: typing.Union[str, mlrun.api.schemas.ScheduleCronTrigger] = None,
    cleanup_ttl: int = None,
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
    :param ttl:                 pipeline cleanup ttl in secs (time to wait after workflow completion, at which point the
                                workflow and all its resources are deleted) (deprecated, use cleanup_ttl instead)
    :param engine:              workflow engine running the workflow.
                                supported values are 'kfp' (default) or 'local'
    :param local:               run local pipeline with local functions (set local=True in function.run())
    :param schedule:            ScheduleCronTrigger class instance or a standard crontab expression string
    :param cleanup_ttl:         pipeline cleanup ttl in secs (time to wait after workflow completion, at which point the
                                workflow and all its resources are deleted)
    """
    if ttl:
        warnings.warn(
            "'ttl' is deprecated, use 'cleanup_ttl' instead. "
            "This will be removed in 1.5.0",
            # TODO: Remove this in 1.5.0
            FutureWarning,
        )

    try:
        project = mlrun.load_project(
            context=f"./{project_name}",
            url=url,
            name=project_name,
            init_git=init_git,
            subpath=subpath,
            clone=clone,
            save=save,
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
                    severity=mlrun.utils.notifications.NotificationSeverity.ERROR,
                )

            except Exception as exc:
                logger.error("Failed to send slack notification", exc=exc)

        raise error

    context.logger.info(f"Loaded project {project.name} from remote successfully")

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
        cleanup_ttl=cleanup_ttl or ttl,
        engine=engine,
        local=local,
    )
    context.log_result(key="workflow_id", value=run.run_id)

    context.log_result(key="engine", value=run._engine.engine, commit=True)
