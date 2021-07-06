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
from mlrun.utils import new_pipe_meta

from ..config import config
from ..run import run_pipeline, wait_for_pipeline_completion


def get_workflow_engine(engine_kind):
    if engine_kind == "kfp":
        return _KFPRunner
    elif engine_kind == "local":
        return _LocalRunner
    else:
        raise ValueError(f"unsupported workflow engine {engine_kind}")


def run_project_pipeline(
    engine,
    project,
    name,
    workflow_spec,
    functions,
    secrets=None,
    artifact_path=None,
    namespace=None,
):
    engine = get_workflow_engine(engine)
    return engine.run(
        name,
        project,
        workflow_spec,
        functions=functions,
        secrets=secrets,
        artifact_path=artifact_path,
        namespace=namespace,
    )


class WorkflowSpec(mlrun.model.ModelObj):
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
            if self.context and not workflow_path.startswith("/"):
                workflow_path = os.path.join(self.context, workflow_path)
        return workflow_path

    def clear_tmp(self):
        if self._tmp_path:
            os.remove(self._tmp_path)


class _PipelineRunStatus:
    def __init__(self, run_id, engine, args=None):
        self.run_id = run_id
        self._engine = engine
        self._args = args

    def wait_for_completion(self, run_id, timeout=None, expected_statuses=None):
        return self._engine.wait_for_completion(
            run_id, timeout=timeout, expected_statuses=expected_statuses, **self._args
        )

    def __str__(self):
        return str(self.run_id)

    def __repr__(self):
        return str(self.run_id)


class _PipelineRunner:
    engine = ""

    @classmethod
    def save(cls, project, workflow_spec: WorkflowSpec, target, artifact_path=None):
        raise NotImplementedError(
            f"save operation not supported in {cls.engine} pipeline engine"
        )

    @classmethod
    def run(
        cls,
        name,
        project,
        workflow_spec: WorkflowSpec,
        functions,
        secrets=None,
        artifact_path=None,
        namespace=None,
    ):
        return

    @staticmethod
    def wait_for_completion(run_id, timeout=None, expected_statuses=None):
        return ""


class _KFPRunner(_PipelineRunner):
    engine = "kfp"

    @classmethod
    def save(cls, project, workflow_spec: WorkflowSpec, target, artifact_path=None):
        workflow_file = workflow_spec.get_source_file(project.spec.context)
        pipeline = create_pipeline(
            project,
            workflow_file,
            project.spec._function_objects,
            secrets=project._secrets,
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
        functions,
        name=None,
        secrets=None,
        artifact_path=None,
        namespace=None,
    ):
        workflow_file = workflow_spec.get_source_file(project.spec.context)
        kfpipeline = create_pipeline(
            project, workflow_file, functions, secrets, arguments=workflow_spec.args
        )

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
        return _PipelineRunStatus(id, cls)

    @staticmethod
    def wait_for_completion(run_id, timeout=None, expected_statuses=None):
        run_info = wait_for_pipeline_completion(
            run_id, timeout=timeout, expected_statuses=expected_statuses
        )
        status = ""
        if run_info:
            status = run_info["run"].get("status")
        return status


class _LocalRunner:
    engine = "local"

    @classmethod
    def run(
        cls,
        name,
        project,
        workflow_spec: WorkflowSpec,
        functions,
        secrets=None,
        artifact_path=None,
        namespace=None,
    ):
        workflow_file = workflow_spec.get_source_file(project.spec.context)
        run = _run_local_pipeline(
            project,
            name,
            workflow_file,
            functions,
            secrets=secrets,
            arguments=workflow_spec.args,
            artifact_path=artifact_path,
            namespace=namespace,
            ttl=workflow_spec.ttl,
        )
        return _PipelineRunStatus(run, cls)

    @staticmethod
    def wait_for_completion(run_id, timeout=None, expected_statuses=None):
        return ""


def _run_kf_pipeline(
    project,
    name,
    pipeline,
    functions,
    secrets=None,
    arguments=None,
    artifact_path=None,
    namespace=None,
    ttl=None,
):
    kfpipeline = create_pipeline(
        project, pipeline, functions, secrets, arguments=arguments
    )

    namespace = namespace or config.namespace
    id = run_pipeline(
        kfpipeline,
        project=project.metadata.name,
        arguments=arguments,
        experiment=name,
        namespace=namespace,
        artifact_path=artifact_path,
        ttl=ttl,
    )
    return id


def _run_local_pipeline(
    project,
    name,
    pipeline,
    functions,
    secrets=None,
    arguments=None,
    artifact_path=None,
    namespace=None,
    ttl=None,
):
    runs = {}

    def run_decorator(func):
        def wrapped(*args, **kw):
            labels = kw.get("labels", {})
            labels["workflow-name"] = name
            kw["labels"] = labels
            run: mlrun.RunObject = func._old_run(*args, **kw)
            if run:
                run._function = func
                run._notified = False
                runs[run.uid()] = run
            return run

        func._old_run = func.run
        return wrapped

    for f in functions.values():
        f.run = run_decorator(f)

    pipeline = create_pipeline(
        project, pipeline, functions, secrets, arguments=arguments
    )
    project.notifiers.push_start_message(project.metadata.name)
    artifact_path = artifact_path or mlrun.mlconf.artifact_path
    try:
        pipeline(
            project,
            functions=functions,
            arguments=arguments,
            secrets=secrets,
            artifact_path=artifact_path,
        )
    except Exception as e:
        project.notifiers.push(f"Pipeline run failed!, error: {e}")

    mlrun.run.wait_for_runs_completion(runs.values())
    project.notifiers.push_run_results(runs.values())
    return ""


def create_pipeline(
    project, pipeline, funcs, secrets=None, arguments=None, handler=None
):
    functions = enrich_functions_source(project, funcs)
    spec = imputil.spec_from_file_location("workflow", pipeline)
    if spec is None:
        raise ImportError(f"cannot import workflow {pipeline}")
    mod = imputil.module_from_spec(spec)
    spec.loader.exec_module(mod)

    setattr(mod, "funcs", functions)
    setattr(mod, "this_project", project)

    if hasattr(mod, "init_workflow"):
        getattr(mod, "init_workflow")(functions, project, secrets, arguments)
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


def enrich_functions_source(project, funcs):
    functions = {}
    for name, func in funcs.items():
        f = func.copy()
        f.metadata.project = project.metadata.name
        src = f.spec.build.source
        if project.spec.source and src and src in [".", "./"]:
            if project.spec.mountdir:
                f.spec.workdir = project.spec.mountdir
                f.spec.build.source = ""
            else:
                f.spec.build.source = project.spec.source
                f.spec.build.load_source_on_run = project.spec.load_source_on_run

        functions[name] = f
    return functions
