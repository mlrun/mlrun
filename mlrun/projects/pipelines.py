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
import traceback
from tempfile import mktemp

from kfp.compiler import compiler

import mlrun
from mlrun.utils import logger, new_pipe_meta, parse_versioned_object_uri

from ..config import config
from ..run import run_pipeline, wait_for_pipeline_completion


class _PipelineContext:
    def __init__(self):
        self.project = None
        self.workflow = None
        self.functions = None
        self.workflow_id = None

    def set(self, project, workflow, functions):
        self.project = project
        self.workflow = workflow
        self.functions = functions

    def get_function(self, key):
        if self.functions:
            return self.functions.load_or_set_function(key)


pipeline_context = _PipelineContext()


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
            if context and not workflow_path.startswith("/"):
                workflow_path = os.path.join(context, workflow_path)
        return workflow_path

    def clear_tmp(self):
        if self._tmp_path:
            os.remove(self._tmp_path)


class FunctionsDict(dict):
    def __init__(self, project, functions=None, decorator=None, db=None):
        dict.__init__(self, functions)
        self.project = project
        self._db = db or mlrun.get_run_db()
        self._decorator = decorator

    def _enrich(self, function):
        function = _enrich_function_source(self.project, function)
        if self._decorator:
            self._decorator(function)
        return function

    def load_or_set_function(self, key, default=None):
        if key in dict.keys(self):
            return self._enrich(dict.__getitem__(self, key))

        if default:
            function = default
        else:
            project_name = self.project.metadata.name
            project_instance, name, tag, hash_key = parse_versioned_object_uri(
                key, project_name
            )
            runtime = self._db.get_function(name, project_instance, tag, hash_key)
            function = mlrun.new_function(runtime=runtime)

        dict.__setitem__(self, key, function)
        print(f"GET func[{key}] = {function}")
        return self._enrich(function)

    def get(self, key, default=None):
        return self.load_or_set_function(key, default)

    def __getitem__(self, key):
        return self.load_or_set_function(key)

    def __setitem__(self, key, val):
        print(f"SET [{key}] = {val}")
        dict.__setitem__(self, key, val)


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
        functions = FunctionsDict(project, project.spec._function_objects)
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
        functions,
        name=None,
        secrets=None,
        artifact_path=None,
        namespace=None,
    ):
        workflow_file = workflow_spec.get_source_file(project.spec.context)
        functions = FunctionsDict(project, functions)
        pipeline_context.set(project, workflow_spec, functions)
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
        runs = {}
        workflow_id = name

        def run_decorator(func):
            def wrapped(*args, **kw):
                labels = kw.get("labels") or {}
                labels["workflow"] = workflow_id
                kw["labels"] = labels
                run: mlrun.RunObject = func._old_run(*args, **kw)
                if run:
                    run._function = func
                    run._notified = False
                    runs[run.uid()] = run
                return run

            func._old_run = func.run
            func.run = wrapped

        for f in functions.values():
            run_decorator(f)

        functions = FunctionsDict(project, functions, decorator=run_decorator)
        pipeline_context.set(project, workflow_spec, functions)
        pipeline = create_pipeline(
            project, workflow_file, functions, secrets, arguments=workflow_spec.args
        )
        project.notifiers.push_start_message(project.metadata.name)
        artifact_path = artifact_path or mlrun.mlconf.artifact_path
        try:
            pipeline(
                project,
                functions=functions,
                arguments=workflow_spec.args,
                secrets=secrets,
                artifact_path=artifact_path,
            )
        except Exception as e:
            trace = traceback.format_exc()
            logger.error(trace)
            project.notifiers.push(f"Pipeline run failed!, error: {e}\n{trace}")

        mlrun.run.wait_for_runs_completion(runs.values())
        project.notifiers.push_run_results(runs.values())
        return _PipelineRunStatus(workflow_id, cls)

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
    functions = FunctionsDict(project, functions)
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


def create_pipeline(
    project, pipeline, functions, secrets=None, arguments=None, handler=None
):
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


def _enrich_function_source(project, func):
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
    return f
