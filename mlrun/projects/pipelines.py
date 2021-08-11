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


import importlib.util as imputil
import os
from tempfile import mktemp

from kfp.compiler import compiler

import mlrun
from mlrun.utils import new_pipe_meta, parse_versioned_object_uri

from ..config import config
from ..run import run_pipeline, wait_for_pipeline_completion


class _PipelineContext:
    """current (running) pipeline context"""

    def __init__(self):
        self.project = None
        self.workflow = None
        self.functions = None
        self.workflow_id = None

    def set(self, project, functions, workflow=None):
        self.project = project
        self.workflow = workflow
        self.functions = functions

    def is_initialized(self, raise_exception=False):
        if self.project:
            return True
        if raise_exception:
            raise ValueError(
                "pipeline context is not initialized, must be used inside a pipeline"
            )
        return False

    def enrich_function(self, function):
        self.is_initialized(raise_exception=True)
        return self.functions._enrich(function)


pipeline_context = _PipelineContext()


def get_workflow_engine(engine_kind):
    if not engine_kind or engine_kind == "kfp":
        return _KFPRunner
    # todo: add engines
    else:
        raise ValueError(f"unsupported workflow engine {engine_kind}")


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
    ):
        self.engine = engine
        self.code = code
        self.path = path
        self.args = args
        self.name = name
        self.handler = handler
        self.ttl = ttl
        self._tmp_path = None

    def get_source_file(self, context=""):
        if self.code:
            workflow_path = mktemp(".py")
            with open(workflow_path, "w") as wf:
                wf.write(self.code)
            self._tmp_path = workflow_path
        else:
            workflow_path = self.path or ""
            if context and not workflow_path.startswith("/"):
                workflow_path = os.path.join(context, workflow_path)
        return workflow_path

    def merge_args(self, extra_args):
        self.args = self.args or {}
        if extra_args:
            for k, v in extra_args.items():
                self.args[k] = v

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

    def _enrich(self, function):
        return enrich_function_object(self.project, function, self._decorator)

    def load_or_set_function(self, key, default=None):
        try:
            function = self.project.func(key)
        except Exception as e:
            if not default:
                raise e
            function = default

        self._functions[key] = self._enrich(function)
        return self._functions[key]

    def get(self, key, default=None):
        return self.load_or_set_function(key, default)

    def __getitem__(self, key):
        return self.load_or_set_function(key)

    def __setitem__(self, key, val):
        self._functions[key] = val

    def values(self):
        return self._functions.values()

    def keys(self):
        return self._functions.keys()

    def items(self):
        return self._functions.items()

    def __len__(self):
        return len(self._functions)

    def __iter__(self):
        yield from self._functions.keys()

    def __delitem__(self, key):
        del self._functions[key]


def get_db_function(project, key):
    project_instance, name, tag, hash_key = parse_versioned_object_uri(
        key, project.metadata.name
    )
    runtime = mlrun.get_run_db().get_function(name, project_instance, tag, hash_key)
    return mlrun.new_function(runtime=runtime)


def enrich_function_object(
    project, function, decorator=None
) -> "mlrun.runtimes.BaseRuntime":
    if hasattr(function, "_enriched"):
        return function
    f = function.copy()
    f.metadata.project = project.metadata.name
    setattr(f, "_enriched", True)
    src = f.spec.build.source
    if project.spec.source and src and src in [".", "./"]:
        if project.spec.mountdir:
            f.spec.workdir = project.spec.mountdir
            f.spec.build.source = ""
        else:
            f.spec.build.source = project.spec.source
            f.spec.build.load_source_on_run = project.spec.load_source_on_run
    f.try_auto_mount_based_on_config()
    if decorator:
        decorator(f)
    return f


class _PipelineRunStatus:
    """pipeline run result (status)"""

    def __init__(self, run_id, engine, project, workflow=None):
        self.run_id = run_id
        self.project = project
        self.workflow = workflow
        self._engine = engine

    def wait_for_completion(self, run_id, timeout=None, expected_statuses=None):
        return self._engine.wait_for_completion(
            run_id,
            project=self.project,
            timeout=timeout,
            expected_statuses=expected_statuses,
        )

    def __str__(self):
        return str(self.run_id)

    def __repr__(self):
        return str(self.run_id)


class _PipelineRunner:
    """abstract pipeline runner class"""

    engine = ""

    @classmethod
    def save(cls, project, workflow_spec: WorkflowSpec, target, artifact_path=None):
        raise NotImplementedError(
            f"save operation not supported in {cls.engine} pipeline engine"
        )

    @classmethod
    def run(
        cls,
        project,
        name,
        workflow_spec: WorkflowSpec,
        secrets=None,
        artifact_path=None,
        namespace=None,
    ):
        return

    @staticmethod
    def wait_for_completion(run_id, project=None, timeout=None, expected_statuses=None):
        return ""


class _KFPRunner(_PipelineRunner):
    """Kubeflow pipelines runner"""

    engine = "kfp"

    @classmethod
    def save(cls, project, workflow_spec: WorkflowSpec, target, artifact_path=None):
        workflow_file = workflow_spec.get_source_file(project.spec.context)
        functions = FunctionsDict(project)
        pipeline = create_pipeline(
            project, workflow_file, functions, secrets=project._secrets,
        )
        artifact_path = artifact_path or project.spec.artifact_path

        conf = new_pipe_meta(artifact_path, ttl=workflow_spec.ttl)
        compiler.Compiler().compile(pipeline, target, pipeline_conf=conf)
        workflow_spec.clear_tmp()

    @classmethod
    def run(
        cls,
        project,
        workflow_spec: WorkflowSpec,
        name=None,
        secrets=None,
        artifact_path=None,
        namespace=None,
    ):
        workflow_file = workflow_spec.get_source_file(project.spec.context)
        functions = FunctionsDict(project)
        pipeline_context.set(project, functions, workflow_spec)
        kfpipeline = create_pipeline(project, workflow_file, functions, secrets)

        namespace = namespace or config.namespace
        id = run_pipeline(
            kfpipeline,
            project=project.metadata.name,
            arguments=workflow_spec.args,
            experiment=name or workflow_spec.name,
            namespace=namespace,
            artifact_path=artifact_path,
            ttl=workflow_spec.ttl,
        )
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


def create_pipeline(project, pipeline, functions, secrets=None, handler=None):
    spec = imputil.spec_from_file_location("workflow", pipeline)
    if spec is None:
        raise ImportError(f"cannot import workflow {pipeline}")
    mod = imputil.module_from_spec(spec)
    spec.loader.exec_module(mod)

    setattr(mod, "funcs", functions)
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
