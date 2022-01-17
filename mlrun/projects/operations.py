from typing import List, Union

import kfp

import mlrun
from mlrun.utils import hub_prefix

from .pipelines import enrich_function_object, pipeline_context


def _get_engine_and_function(function, project=None):
    is_function_object = not isinstance(function, str)
    project = project or pipeline_context.project
    if not is_function_object:
        if function.startswith(hub_prefix):
            function = mlrun.import_function(function)
            if project:
                function = enrich_function_object(project, function)
        else:
            if not project:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "function name (str) can only be used in a project context, you must create, "
                    "load or get a project first or provide function object instead of its name"
                )
            function = pipeline_context.functions[function]
    elif project:
        function = enrich_function_object(project, function)

    if not pipeline_context.workflow:
        return "local", function

    return pipeline_context.workflow.engine, function


def run_function(
    function: Union[str, mlrun.runtimes.BaseRuntime],
    handler: str = None,
    name: str = "",
    params: dict = None,
    hyperparams: dict = None,
    hyper_param_options: mlrun.model.HyperParamOptions = None,
    inputs: dict = None,
    outputs: List[str] = None,
    workdir: str = "",
    labels: dict = None,
    base_task: mlrun.model.RunTemplate = None,
    watch: bool = True,
    local: bool = False,
    verbose: bool = None,
    project_object=None,
) -> Union[mlrun.model.RunObject, kfp.dsl.ContainerOp]:
    """Run a local or remote task as part of a local/kubeflow pipeline

    run_function() allow you to execute a function locally, on a remote cluster, or as part of an automated workflow
    function can be specified as an object or by name (str), when the function is specified by name it is looked up
    in the current project eliminating the need to redefine/edit functions.

    when functions run as part of a workflow/pipeline (project.run()) some attributes can be set at the run level,
    e.g. local=True will run all the functions locally, setting artifact_path will direct all outputs to the same path.
    project runs provide additional notifications/reporting and exception handling.
    inside a Kubeflow pipeline (KFP) run_function() generates KFP "ContainerOps" which are used to form a DAG
    some behavior may differ between regular runs and deferred KFP runs.

    example (use with function object)::

        function = mlrun.import_function("hub://sklearn_classifier")
        run1 = run_function(function, params={"data": url})

    example (use with project)::

        # create a project with two functions (local and from marketplace)
        project = mlrun.new_project(project_name, "./proj)
        project.set_function("mycode.py", "myfunc", image="mlrun/mlrun")
        project.set_function("hub://sklearn_classifier", "train")

        # run functions (refer to them by name)
        run1 = run_function("myfunc", params={"x": 7})
        run2 = run_function("train", params={"data": run1.outputs["data"]})

    example (use in pipeline)::

        @dsl.pipeline(name="test pipeline", description="test")
        def my_pipe(url=""):
            run1 = run_function("loaddata", params={"url": url})
            run2 = run_function("train", params={"data": run1.outputs["data"]})

        project.run(workflow_handler=my_pipe, arguments={"param1": 7})

    :param function:        name of the function (in the project) or function object
    :param handler:         name of the function handler
    :param name:            execution name
    :param params:          input parameters (dict)
    :param hyperparams:     hyper parameters
    :param hyper_param_options:  hyper param options (selector, early stop, strategy, ..)
                            see: :py:class:`~mlrun.model.HyperParamOptions`
    :param inputs:          input objects (dict of key: path)
    :param outputs:         list of outputs which can pass in the workflow
    :param workdir:         default input artifacts path
    :param labels:          labels to tag the job/run with ({key:val, ..})
    :param base_task:       task object to use as base
    :param watch:           watch/follow run log, True by default
    :param local:           run the function locally vs on the runtime/cluster
    :param verbose:         add verbose prints/logs

    :return: MLRun RunObject or KubeFlow containerOp
    """
    engine, function = _get_engine_and_function(function, project_object)
    task = mlrun.new_task(
        name,
        handler=handler,
        params=params,
        hyper_params=hyperparams,
        hyper_param_options=hyper_param_options,
        inputs=inputs,
        base=base_task,
    )
    task.spec.verbose = task.spec.verbose or verbose

    if engine == "kfp":
        return function.as_step(
            runspec=task, workdir=workdir, outputs=outputs, labels=labels
        )
    else:
        if pipeline_context.workflow:
            local = local or pipeline_context.workflow.run_local
        task.metadata.labels = task.metadata.labels or labels or {}
        task.metadata.labels["workflow"] = pipeline_context.workflow_id
        run_result = function.run(
            runspec=task,
            workdir=workdir,
            verbose=verbose,
            watch=watch,
            local=local,
            artifact_path=pipeline_context.workflow_artifact_path,
        )
        if run_result:
            run_result._notified = False
            pipeline_context.runs_map[run_result.uid()] = run_result
            run_result.after = (
                lambda x: run_result
            )  # emulate KFP op, .after() will be ignored
        return run_result


class BuildStatus:
    """returned status from build operation"""

    def __init__(self, ready, outputs={}):
        self.ready = ready
        self.outputs = outputs

    def after(self, step):
        """nil function, for KFP compatibility"""
        return self

    def __repr__(self):
        return f"BuildStatus(ready={self.ready}, outputs={self.outputs})"


def build_function(
    function: Union[str, mlrun.runtimes.BaseRuntime],
    with_mlrun: bool = True,
    skip_deployed: bool = False,
    image=None,
    base_image=None,
    commands: list = None,
    secret_name="",
    requirements: Union[str, List[str]] = None,
    mlrun_version_specifier=None,
    builder_env: dict = None,
    project_object=None,
):
    """deploy ML function, build container with its dependencies

    :param function:        name of the function (in the project) or function object
    :param with_mlrun:      add the current mlrun package to the container build
    :param skip_deployed:   skip the build if we already have an image for the function
    :param image:           target image name/path
    :param base_image:      base image name/path (commands and source code will be added to it)
    :param commands:        list of docker build (RUN) commands e.g. ['pip install pandas']
    :param secret_name:     k8s secret for accessing the docker registry
    :param requirements:    list of python packages or pip requirements file path, defaults to None
    :param mlrun_version_specifier:  which mlrun package version to include (if not current)
    :param builder_env:     Kaniko builder pod env vars dict (for config/credentials)
                            e.g. builder_env={"GIT_TOKEN": token}, does not work yet in KFP
    """
    engine, function = _get_engine_and_function(function, project_object)
    if requirements:
        function.with_requirements(requirements)
        if commands and function.spec.build.commands:
            # merge requirements with commands
            for command in function.spec.build.commands:
                if command not in commands:
                    commands.append(command)

    if function.kind in mlrun.runtimes.RuntimeKinds.nuclio_runtimes():
        raise mlrun.errors.MLRunInvalidArgumentError(
            "cannot build use deploy_function()"
        )
    if engine == "kfp":
        return function.deploy_step(
            image=image,
            base_image=base_image,
            commands=commands,
            secret_name=secret_name,
            with_mlrun=with_mlrun,
            skip_deployed=skip_deployed,
        )
    else:
        function.build_config(
            image=image, base_image=base_image, commands=commands, secret=secret_name
        )
        ready = function.deploy(
            watch=True,
            with_mlrun=with_mlrun,
            skip_deployed=skip_deployed,
            mlrun_version_specifier=mlrun_version_specifier,
            builder_env=builder_env,
        )
        # return object with the same outputs as the KFP op (allow using the same pipeline)
        return BuildStatus(ready, {"image": function.spec.image})


class DeployStatus:
    """returned status from deploy operation"""

    def __init__(self, state, outputs={}):
        self.state = state
        self.outputs = outputs

    def after(self, step):
        """nil function, for KFP compatibility"""
        return self

    def __repr__(self):
        return f"DeployStatus(state={self.state}, outputs={self.outputs})"


def deploy_function(
    function: Union[str, mlrun.runtimes.BaseRuntime],
    dashboard: str = "",
    models: list = None,
    env: dict = None,
    tag: str = None,
    verbose: bool = None,
    project_object=None,
):
    """deploy real-time (nuclio based) functions

    :param function:   name of the function (in the project) or function object
    :param dashboard:  url of the remore Nuclio dashboard (when not local)
    :param models:     list of model items
    :param env:        dict of extra environment variables
    :param tag:        extra version tag
    :param verbose     add verbose prints/logs
    """
    engine, function = _get_engine_and_function(function, project_object)
    if function.kind not in mlrun.runtimes.RuntimeKinds.nuclio_runtimes():
        raise mlrun.errors.MLRunInvalidArgumentError(
            "deploy is used with real-time functions, for other kinds use build_function()"
        )
    if engine == "kfp":
        return function.deploy_step(
            dashboard=dashboard, models=models, env=env, tag=tag, verbose=verbose
        )
    else:
        if env:
            function.set_envs(env)
        if models:
            for model_args in models:
                function.add_model(**model_args)
        address = function.deploy(dashboard=dashboard, tag=tag, verbose=verbose)
        # return object with the same outputs as the KFP op (allow using the same pipeline)
        return DeployStatus(
            state=function.status.state,
            outputs={"endpoint": address, "name": function.status.nuclio_name},
        )
