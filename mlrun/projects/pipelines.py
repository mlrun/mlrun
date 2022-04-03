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
import traceback
import types
import typing
import uuid

import kfp
import kfp.compiler
from kfp.compiler import compiler

import mlrun
from mlrun.utils import logger, new_pipe_meta, parse_versioned_object_uri

from ..config import config
from ..run import run_pipeline, wait_for_pipeline_completion
from ..runtimes.pod import AutoMountType


def get_workflow_engine(engine_kind, local=False):
    if local:
        if engine_kind == "kfp":
            logger.warning(
                "running kubeflow pipeline locally, note some ops may not run locally!"
            )
        return _LocalRunner
    if not engine_kind or engine_kind == "kfp":
        return _KFPRunner
    if engine_kind == "local":
        return _LocalRunner
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
        ttl=None,
        args_schema: dict = None,
    ):
        self.engine = engine
        self.code = code
        self.path = path
        self.args = args
        self.name = name
        self.handler = handler
        self.ttl = ttl
        self.args_schema = args_schema
        self.run_local = False
        self._tmp_path = None

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
            function = self.project.get_function(key)
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
{
    "kind": "project",
    "metadata": {"name": "sk-project-dask"},
    "spec": {
        "functions": [
            {
                "name": "my-trainer",
                "spec": {
                    "kind": "job",
                    "metadata": {
                        "name": "my-trainer",
                        "tag": "",
                        "project": "sk-project-dask",
                    },
                    "spec": {
                        "command": "",
                        "args": [],
                        "image": "",
                        "build": {
                            "functionSourceCode": "IyBHZW5lcmF0ZWQgYnkgbnVjbGlvLmV4cG9ydC5OdWNsaW9FeHBvcnRlcgoKaW1wb3J0IG1scnVuIAoKaW1wb3J0IHRpbWUKaW1wb3J0IHBhbmRhcyBhcyBwZApmcm9tIG1scnVuLmFydGlmYWN0cyBpbXBvcnQgZ2V0X21vZGVsLCB1cGRhdGVfbW9kZWwKCmRlZiB0cmFpbmluZygKICAgIGNvbnRleHQsCiAgICBwMTogaW50ID0gMSwKICAgIHAyOiBpbnQgPSAyCikgLT4gTm9uZToKICAgICIiIlRyYWluIGEgbW9kZWwuCgogICAgOnBhcmFtIGNvbnRleHQ6IFRoZSBydW50aW1lIGNvbnRleHQgb2JqZWN0LgogICAgOnBhcmFtIHAxOiBBIG1vZGVsIHBhcmFtZXRlci4KICAgIDpwYXJhbSBwMjogQW5vdGhlciBtb2RlbCBwYXJhbWV0ZXIuCiAgICAiIiIKICAgICMgYWNjZXNzIGlucHV0IG1ldGFkYXRhLCB2YWx1ZXMsIGFuZCBpbnB1dHMKICAgIHByaW50KGYnUnVuOiB7Y29udGV4dC5uYW1lfSAodWlkPXtjb250ZXh0LnVpZH0pJykKICAgIHByaW50KGYnUGFyYW1zOiBwMT17cDF9LCBwMj17cDJ9JykKICAgIGNvbnRleHQubG9nZ2VyLmluZm8oJ3N0YXJ0ZWQgdHJhaW5pbmcnKQogICAgCiAgICAjIDxpbnNlcnQgdHJhaW5pbmcgY29kZSBoZXJlPgogICAgCiAgICAjIGxvZyB0aGUgcnVuIHJlc3VsdHMgKHNjYWxhciB2YWx1ZXMpCiAgICBjb250ZXh0LmxvZ19yZXN1bHQoJ2FjY3VyYWN5JywgcDEgKiAyKQogICAgY29udGV4dC5sb2dfcmVzdWx0KCdsb3NzJywgcDEgKiAzKQogICAgCiAgICAjIGFkZCBhIGxhYmVsL3RhZyB0byB0aGlzIHJ1biAKICAgIGNvbnRleHQuc2V0X2xhYmVsKCdjYXRlZ29yeScsICd0ZXN0cycpCiAgICAKICAgICMgbG9nIGEgc2ltcGxlIGFydGlmYWN0ICsgbGFiZWwgdGhlIGFydGlmYWN0IAogICAgIyBJZiB5b3Ugd2FudCB0byB1cGxvYWQgYSBsb2NhbCBmaWxlIHRvIHRoZSBhcnRpZmFjdCByZXBvIGFkZCBzcmNfcGF0aD08bG9jYWwtcGF0aD4KICAgIGNvbnRleHQubG9nX2FydGlmYWN0KCdzb21lZmlsZScsIAogICAgICAgICAgICAgICAgICAgICAgICAgIGJvZHk9YidhYmMgaXMgMTIzJywgCiAgICAgICAgICAgICAgICAgICAgICAgICAgbG9jYWxfcGF0aD0nbXlmaWxlLnR4dCcpCiAgICAKICAgICMgY3JlYXRlIGEgZGF0YWZyYW1lIGFydGlmYWN0IAogICAgZGYgPSBwZC5EYXRhRnJhbWUoW3snQSc6MTAsICdCJzoxMDB9LCB7J0EnOjExLCdCJzoxMTB9LCB7J0EnOjEyLCdCJzoxMjB9XSkKICAgIGNvbnRleHQubG9nX2RhdGFzZXQoJ215ZGYnLCBkZj1kZikKICAgIAogICAgIyBMb2cgYW4gTUwgTW9kZWwgYXJ0aWZhY3QsIGFkZCBtZXRyaWNzLCBwYXJhbXMsIGFuZCBsYWJlbHMgdG8gaXQKICAgICMgYW5kIHBsYWNlIGl0IGluIGEgc3ViZGlyICgnbW9kZWxzJykgdW5kZXIgYXJ0aWZhY3RzIHBhdGggCiAgICBjb250ZXh0LmxvZ19tb2RlbCgnbXltb2RlbCcsIGJvZHk9YidhYmMgaXMgMTIzJywgCiAgICAgICAgICAgICAgICAgICAgICBtb2RlbF9maWxlPSdtb2RlbC50eHQnLCAKICAgICAgICAgICAgICAgICAgICAgIG1ldHJpY3M9eydhY2N1cmFjeSc6MC44NX0sIHBhcmFtZXRlcnM9eyd4eCc6J2FiYyd9LAogICAgICAgICAgICAgICAgICAgICAgbGFiZWxzPXsnZnJhbWV3b3JrJzogJ3hnYm9vc3QnfSwKICAgICAgICAgICAgICAgICAgICAgIGFydGlmYWN0X3BhdGg9Y29udGV4dC5hcnRpZmFjdF9zdWJwYXRoKCdtb2RlbHMnKSkKCmRlZiB2YWxpZGF0aW9uKAogICAgY29udGV4dCwKICAgIG1vZGVsOiBtbHJ1bi5EYXRhSXRlbQopIC0+IE5vbmU6CiAgICAiIiJNb2RlbCB2YWxpZGF0aW9uLgogICAgCiAgICBEdW1teSB2YWxpZGF0aW9uIGZ1bmN0aW9uLgogICAgCiAgICA6cGFyYW0gY29udGV4dDogVGhlIHJ1bnRpbWUgY29udGV4dCBvYmplY3QuCiAgICA6cGFyYW0gbW9kZWw6IFRoZSBleHRpbWF0ZWQgbW9kZWwgb2JqZWN0LgogICAgIiIiCiAgICAjIGFjY2VzcyBpbnB1dCBtZXRhZGF0YSwgdmFsdWVzLCBmaWxlcywgYW5kIHNlY3JldHMgKHBhc3N3b3JkcykKICAgIHByaW50KGYnUnVuOiB7Y29udGV4dC5uYW1lfSAodWlkPXtjb250ZXh0LnVpZH0pJykKICAgIGNvbnRleHQubG9nZ2VyLmluZm8oJ3N0YXJ0ZWQgdmFsaWRhdGlvbicpCiAgICAKICAgICMgZ2V0IHRoZSBtb2RlbCBmaWxlLCBjbGFzcyAobWV0YWRhdGEpLCBhbmQgZXh0cmFfZGF0YSAoZGljdCBvZiBrZXk6IERhdGFJdGVtKQogICAgbW9kZWxfZmlsZSwgbW9kZWxfb2JqLCBfID0gZ2V0X21vZGVsKG1vZGVsKQoKICAgICMgdXBkYXRlIG1vZGVsIG9iamVjdCBlbGVtZW50cyBhbmQgZGF0YQogICAgdXBkYXRlX21vZGVsKG1vZGVsX29iaiwgcGFyYW1ldGVycz17J29uZV9tb3JlJzogNX0pCgogICAgcHJpbnQoZidwYXRoIHRvIGxvY2FsIGNvcHkgb2YgbW9kZWwgZmlsZSAtIHttb2RlbF9maWxlfScpCiAgICBwcmludCgncGFyYW1ldGVyczonLCBtb2RlbF9vYmoucGFyYW1ldGVycykKICAgIHByaW50KCdtZXRyaWNzOicsIG1vZGVsX29iai5tZXRyaWNzKQogICAgY29udGV4dC5sb2dfYXJ0aWZhY3QoJ3ZhbGlkYXRpb24nLCAKICAgICAgICAgICAgICAgICAgICAgICAgIGJvZHk9Yic8Yj4gdmFsaWRhdGVkIDwvYj4nLCAKICAgICAgICAgICAgICAgICAgICAgICAgIGZvcm1hdD0naHRtbCcpCgo=",
                            "base_image": "mlrun/mlrun",
                            "commands": ["python -m pip install pandas"],
                            "origin_filename": "my-trainer.ipynb",
                        },
                        "entry_points": {
                            "training": {
                                "name": "training",
                                "doc": "Train a model.",
                                "parameters": [
                                    {
                                        "name": "context",
                                        "doc": "The runtime context object.",
                                        "default": "",
                                    },
                                    {
                                        "name": "p1",
                                        "type": "int",
                                        "doc": "A model parameter.",
                                        "default": 1,
                                    },
                                    {
                                        "name": "p2",
                                        "type": "int",
                                        "doc": "Another model parameter.",
                                        "default": 2,
                                    },
                                ],
                                "outputs": [{"default": ""}],
                                "lineno": 9,
                            },
                            "validation": {
                                "name": "validation",
                                "doc": "Model validation.\n\nDummy validation function.",
                                "parameters": [
                                    {
                                        "name": "context",
                                        "doc": "The runtime context object.",
                                        "default": "",
                                    },
                                    {
                                        "name": "model",
                                        "type": "DataItem",
                                        "doc": "The extimated model object.",
                                        "default": "",
                                    },
                                ],
                                "outputs": [{"default": ""}],
                                "lineno": 52,
                            },
                        },
                        "description": "",
                        "default_handler": "",
                        "disable_auto_mount": False,
                        "env": [
                            {"name": "V3IO_API", "value": ""},
                            {"name": "V3IO_USERNAME", "value": ""},
                            {"name": "V3IO_ACCESS_KEY", "value": ""},
                            {"name": "V3IO_FRAMESD", "value": ""},
                        ],
                        "node_selector": {"tainted": "yes"},
                        "priority_class_name": "igz-workload-medium",
                        "affinity": None,
                        "tolerations": [
                            {
                                "effect": "NoSchedule",
                                "key": "key1",
                                "operator": "Exists",
                            }
                        ],
                    },
                    "verbose": False,
                },
            }
        ],
        "workflows": [
            {
                "name": "bbl",
                "path": "workflow.py",
                "handler": "job_pipeline",
                "engine": None,
            }
        ],
        "artifacts": [],
        "source": "",
        "subpath": "",
        "origin_url": "",
        "desired_state": "online",
        "disable_auto_mount": False,
    },
}


def _create_and_write_workflow(
    self,
    pipeline_func,
    pipeline_name=None,
    pipeline_description=None,
    params_list=None,
    pipeline_conf=None,
    package_path=None,
) -> None:
    """Compile the given pipeline function and dump it to specified file
    format."""
    workflow = self._create_workflow(
        pipeline_func, pipeline_name, pipeline_description, params_list, pipeline_conf
    )
    import mlrun.config

    print(pipeline_context.functions.items())
    print(pipeline_context.functions.get("my-trainer"))
    print(pipeline_context.project)
    print(pipeline_context)
    for pod in workflow["spec"]["templates"]:
        if pod.get("container"):
            pod[
                "PriorityClassName"
            ] = mlrun.config.config.default_function_priority_class_name

    print(mlrun.config.config.default_function_priority_class_name)
    logger.info("im hereee")
    # workflow["spec"][
    #     "PodPriorityClassName"
    # ] = mlrun.config.config.default_function_priority_class_name
    # workflow["spec"]["PodPriority"] = 1
    print(workflow)
    self._write_workflow(workflow, package_path)
    kfp.compiler.compiler._validate_workflow(workflow)


kfp.compiler.Compiler._create_and_write_workflow = _create_and_write_workflow


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
    src = f.spec.build.source
    if src and src in [".", "./"]:

        if not project.spec.source:
            raise ValueError(
                "project source must be specified when cloning context to a function"
            )

        if project.spec.mountdir:
            f.spec.workdir = project.spec.mountdir
            f.spec.build.source = ""
        else:
            f.spec.build.source = project.spec.source
            f.spec.build.load_source_on_run = project.spec.load_source_on_run
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

    def __init__(self, run_id, engine, project, workflow=None, state=""):
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
    ) -> _PipelineRunStatus:
        return None

    @staticmethod
    @abc.abstractmethod
    def wait_for_completion(run_id, project=None, timeout=None, expected_statuses=None):
        return ""

    @staticmethod
    @abc.abstractmethod
    def get_state(run_id, project=None):
        return ""

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


class _KFPRunner(_PipelineRunner):
    """Kubeflow pipelines runner"""

    engine = "kfp"

    @classmethod
    def save(cls, project, workflow_spec: WorkflowSpec, target, artifact_path=None):
        workflow_file = workflow_spec.get_source_file(project.spec.context)
        functions = FunctionsDict(project)
        pipeline = create_pipeline(
            project,
            workflow_file,
            functions,
            secrets=project._secrets,
        )
        artifact_path = artifact_path or project.spec.artifact_path

        conf = new_pipe_meta(artifact_path, ttl=workflow_spec.ttl)
        compiler.Compiler._create_and_write_workflow = _create_and_write_workflow
        compiler.Compiler().compile(pipeline, target, pipeline_conf=conf)
        workflow_spec.clear_tmp()

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
    ) -> _PipelineRunStatus:
        pipeline_context.set(project, workflow_spec)
        workflow_handler = _PipelineRunner._get_handler(
            workflow_handler, workflow_spec, project, secrets
        )

        namespace = namespace or config.namespace
        id = run_pipeline(
            workflow_handler,
            project=project.metadata.name,
            arguments=workflow_spec.args,
            experiment=name or workflow_spec.name,
            namespace=namespace,
            artifact_path=artifact_path,
            ttl=workflow_spec.ttl,
        )
        project.notifiers.push_start_message(
            project.metadata.name,
            project.get_param("commit_id", None),
            id,
            True,
        )
        pipeline_context.clear()
        return _PipelineRunStatus(id, cls, project=project, workflow=workflow_spec)

    @staticmethod
    def wait_for_completion(run_id, project=None, timeout=None, expected_statuses=None):
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
        pipeline_context.workflow_artifact_path = artifact_path
        project.notifiers.push_start_message(project.metadata.name, id=workflow_id)
        try:
            workflow_handler(**workflow_spec.args)
            state = mlrun.run.RunStatuses.succeeded
        except Exception as e:
            trace = traceback.format_exc()
            logger.error(trace)
            project.notifiers.push(
                f"Workflow {workflow_id} run failed!, error: {e}\n{trace}"
            )
            state = mlrun.run.RunStatuses.failed

        mlrun.run.wait_for_runs_completion(pipeline_context.runs_map.values())
        project.notifiers.push_run_results(
            pipeline_context.runs_map.values(), state=state
        )
        pipeline_context.clear()
        return _PipelineRunStatus(
            workflow_id, cls, project=project, workflow=workflow_spec, state=state
        )

    @staticmethod
    def get_state(run_id, project=None):
        return ""


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
