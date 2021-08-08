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

import mlrun

from ..config import config
from ..run import run_pipeline
from ..utils import parse_versioned_object_uri


class _PipelineContext:
    def __init__(self):
        self.project = None
        self.functions_dict = None

    def set(self, project, functions_dict):
        self.project = project
        self.functions_dict = functions_dict

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
        return self.functions_dict._enrich(function)


pipeline_context = _PipelineContext()


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
            self._functions[key] = function

        return self._enrich(function)

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
    if decorator:
        decorator(f)
    return f


def _create_pipeline(project, pipeline, funcs, secrets=None, handler=None):
    functions = FunctionsDict(project)
    pipeline_context.set(project, functions)
    spec = imputil.spec_from_file_location("workflow", pipeline)
    if spec is None:
        raise ImportError(f"cannot import workflow {pipeline}")
    mod = imputil.module_from_spec(spec)
    spec.loader.exec_module(mod)

    setattr(mod, "funcs", functions)
    setattr(mod, "this_project", project)

    if hasattr(mod, "init_functions"):
        getattr(mod, "init_functions")(functions, project, secrets)

    # verify all functions are in this project
    for f in functions.values():
        f.metadata.project = project.metadata.name

    if not handler and hasattr(mod, "kfpipeline"):
        handler = "kfpipeline"
    if not handler and hasattr(mod, "pipeline"):
        handler = "pipeline"
    if not handler or not hasattr(mod, handler):
        raise ValueError(f"pipeline function ({handler or 'kfpipeline'}) not found")

    return getattr(mod, handler)


def _run_pipeline(
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
    kfpipeline = _create_pipeline(project, pipeline, functions, secrets)

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
