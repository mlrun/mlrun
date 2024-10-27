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
#
import warnings
from typing import Optional, Union

import mlrun_pipelines.common.models
import mlrun_pipelines.models

import mlrun
import mlrun.common.constants as mlrun_constants
import mlrun.common.schemas.function
import mlrun.common.schemas.workflow
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
            # we don't want to use a copy of the function object, we want to use the actual object
            # so changes on it will be reflected in the project and will persist for future use of the function
            function = project.get_function(
                function, sync=False, enrich=True, copy_function=False
            )
    elif project:
        # if a user provide the function object we enrich in-place so build, deploy, etc.
        # will update the original function object status/image, and not the copy (may fail fn.run())
        function = enrich_function_object(project, function, copy_function=False)

    if not pipeline_context.workflow:
        return mlrun.common.schemas.workflow.EngineType.LOCAL, function

    return pipeline_context.workflow.engine, function


def run_function(
    function: Union[str, mlrun.runtimes.BaseRuntime],
    handler: str = None,
    name: str = "",
    params: dict = None,
    hyperparams: dict = None,
    hyper_param_options: mlrun.model.HyperParamOptions = None,
    inputs: dict = None,
    outputs: list[str] = None,
    workdir: str = "",
    labels: dict = None,
    base_task: mlrun.model.RunTemplate = None,
    watch: bool = True,
    local: bool = None,
    verbose: bool = None,
    selector: str = None,
    project_object=None,
    auto_build: bool = None,
    schedule: Union[str, mlrun.common.schemas.ScheduleCronTrigger] = None,
    artifact_path: str = None,
    notifications: list[mlrun.model.Notification] = None,
    returns: Optional[list[Union[str, dict[str, str]]]] = None,
    builder_env: Optional[list] = None,
    reset_on_run: Optional[bool] = None,
) -> Union[mlrun.model.RunObject, mlrun_pipelines.models.PipelineNodeWrapper]:
    """Run a local or remote task as part of a local/kubeflow pipeline

    run_function() allow you to execute a function locally, on a remote cluster, or as part of an automated workflow
    function can be specified as an object or by name (str), when the function is specified by name it is looked up
    in the current project eliminating the need to redefine/edit functions.

    when functions run as part of a workflow/pipeline (project.run()) some attributes can be set at the run level,
    e.g. local=True will run all the functions locally, setting artifact_path will direct all outputs to the same path.
    project runs provide additional notifications/reporting and exception handling.
    inside a Kubeflow pipeline (KFP) run_function() generates KFP node (see PipelineNodeWrapper) which forms a DAG
    some behavior may differ between regular runs and deferred KFP runs.

    example (use with function object)::

        LABELS = "is_error"
        MODEL_CLASS = "sklearn.ensemble.RandomForestClassifier"
        DATA_PATH = "s3://bigdata/data.parquet"
        function = mlrun.import_function("hub://auto-trainer")
        run1 = run_function(
            function,
            params={"label_columns": LABELS, "model_class": MODEL_CLASS},
            inputs={"dataset": DATA_PATH},
        )

    example (use with project)::

        # create a project with two functions (local and from hub)
        project = mlrun.new_project(project_name, "./proj)
        project.set_function("mycode.py", "myfunc", image="mlrun/mlrun")
        project.set_function("hub://auto-trainer", "train")

        # run functions (refer to them by name)
        run1 = run_function("myfunc", params={"x": 7})
        run2 = run_function("train", params={"label_columns": LABELS, "model_class": MODEL_CLASS},
                                     inputs={"dataset": run1.outputs["data"]})

    example (use in pipeline)::

        @dsl.pipeline(name="test pipeline", description="test")
        def my_pipe(url=""):
            run1 = run_function("loaddata", params={"url": url}, outputs=["data"])
            run2 = run_function(
                "train",
                params={"label_columns": LABELS, "model_class": MODEL_CLASS},
                inputs={"dataset": run1.outputs["data"]},
            )


        project.run(workflow_handler=my_pipe, arguments={"param1": 7})

    :param function:        name of the function (in the project) or function object
    :param handler:         name of the function handler
    :param name:            execution name
    :param params:          input parameters (dict)
    :param hyperparams:     hyper parameters
    :param selector:        selection criteria for hyper params e.g. "max.accuracy"
    :param hyper_param_options:  hyper param options (selector, early stop, strategy, ..)
                            see: :py:class:`~mlrun.model.HyperParamOptions`
    :param inputs:          Input objects to pass to the handler. Type hints can be given so the input will be parsed
                            during runtime from `mlrun.DataItem` to the given type hint. The type hint can be given
                            in the key field of the dictionary after a colon, e.g: "<key> : <type_hint>".
    :param outputs:         list of outputs which can pass in the workflow
    :param workdir:         default input artifacts path
    :param labels:          labels to tag the job/run with ({key:val, ..})
    :param base_task:       task object to use as base
    :param watch:           watch/follow run log, True by default
    :param local:           run the function locally vs on the runtime/cluster
    :param verbose:         add verbose prints/logs
    :param project_object:  override the project object to use, will default to the project set in the runtime context.
    :param auto_build:      when set to True and the function require build it will be built on the first
                            function run, use only if you do not plan on changing the build config between runs
    :param schedule:        ScheduleCronTrigger class instance or a standard crontab expression string
                            (which will be converted to the class using its `from_crontab` constructor),
                            see this link for help:
                            https://apscheduler.readthedocs.io/en/3.x/modules/triggers/cron.html#module-apscheduler.triggers.cron
    :param artifact_path:   path to store artifacts, when running in a workflow this will be set automatically
    :param notifications:   list of notifications to push when the run is completed
    :param returns:         List of log hints - configurations for how to log the returning values from the handler's
                            run (as artifacts or results). The list's length must be equal to the amount of returning
                            objects. A log hint may be given as:

                            * A string of the key to use to log the returning value as result or as an artifact. To
                              specify The artifact type, it is possible to pass a string in the following structure:
                              "<key> : <type>". Available artifact types can be seen in `mlrun.ArtifactType`. If no
                              artifact type is specified, the object's default artifact type will be used.
                            * A dictionary of configurations to use when logging. Further info per object type and
                              artifact type can be given there. The artifact key must appear in the dictionary as
                              "key": "the_key".
    :param builder_env:     env vars dict for source archive config/credentials e.g. builder_env={"GIT_TOKEN": token}
    :param reset_on_run:    When True, function python modules would reload prior to code execution.
                            This ensures latest code changes are executed. This argument must be used in
                            conjunction with the local=True argument.
    :return: MLRun RunObject or PipelineNodeWrapper
    """
    engine, function = _get_engine_and_function(function, project_object)
    task = mlrun.new_task(
        handler=handler,
        params=params,
        hyper_params=hyperparams,
        hyper_param_options=hyper_param_options,
        inputs=inputs,
        returns=returns,
        base=base_task,
        selector=selector,
    )
    task.spec.verbose = task.spec.verbose or verbose

    if engine == mlrun.common.schemas.workflow.EngineType.KFP:
        if schedule:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Scheduling jobs is not supported when running a workflow with the kfp engine."
            )
        return function.as_step(
            name=name, runspec=task, workdir=workdir, outputs=outputs, labels=labels
        )
    else:
        project = project_object or pipeline_context.project
        local = pipeline_context.is_run_local(local)
        task.metadata.labels = task.metadata.labels or labels or {}
        if pipeline_context.workflow_id:
            task.metadata.labels[mlrun_constants.MLRunInternalLabels.workflow] = (
                pipeline_context.workflow_id
            )
        if function.kind == "local":
            command, function = mlrun.run.load_func_code(function)
            function.spec.command = command
        if local and project and function.spec.build.source:
            workdir = workdir or project.spec.get_code_path()
        run_result = function.run(
            name=name,
            runspec=task,
            workdir=workdir,
            verbose=verbose,
            watch=watch,
            local=local,
            artifact_path=artifact_path
            # workflow artifact_path has precedence over the project artifact_path equivalent to
            # passing artifact_path to function.run() has precedence over the project.artifact_path and the default one
            or pipeline_context.workflow_artifact_path
            or (project.artifact_path if project else None),
            auto_build=auto_build,
            schedule=schedule,
            notifications=notifications,
            builder_env=builder_env,
            reset_on_run=reset_on_run,
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

    def __init__(self, ready, outputs=None, function=None):
        self.ready = ready
        self.outputs = outputs or {}
        self.function = function

    def after(self, step):
        """nil function, for KFP compatibility"""
        return self

    def __repr__(self):
        return f"BuildStatus(ready={self.ready}, outputs={self.outputs})"


def build_function(
    function: Union[str, mlrun.runtimes.BaseRuntime],
    with_mlrun: bool = None,
    skip_deployed: bool = False,
    image=None,
    base_image=None,
    commands: list = None,
    secret_name=None,
    requirements: Union[str, list[str]] = None,
    requirements_file: str = None,
    mlrun_version_specifier=None,
    builder_env: dict = None,
    project_object=None,
    overwrite_build_params: bool = False,
    extra_args: str = None,
    force_build: bool = False,
) -> Union[BuildStatus, mlrun_pipelines.models.PipelineNodeWrapper]:
    """deploy ML function, build container with its dependencies

    :param function:        Name of the function (in the project) or function object
    :param with_mlrun:      Add the current mlrun package to the container build
    :param skip_deployed:   Skip the build if we already have an image for the function
    :param image:           Target image name/path
    :param base_image:      Base image name/path (commands and source code will be added to it)
    :param commands:        List of docker build (RUN) commands e.g. ['pip install pandas']
    :param secret_name:     K8s secret for accessing the docker registry
    :param requirements:    List of python packages, defaults to None
    :param requirements_file:    pip requirements file path, defaults to None
    :param mlrun_version_specifier:  which mlrun package version to include (if not current)
    :param builder_env:     Kaniko builder pod env vars dict (for config/credentials)
        e.g. builder_env={"GIT_TOKEN": token}, does not work yet in KFP
    :param project_object:  Override the project object to use, will default to the project set in the runtime context.
    :param overwrite_build_params:  Overwrite existing build configuration (currently applies to
        requirements and commands)
        * False: The new params are merged with the existing
        * True: The existing params are replaced by the new ones
    :param extra_args:  A string containing additional builder arguments in the format of command-line options,
        e.g. extra_args="--skip-tls-verify --build-arg A=val"
    :param force_build: Force building the image, even when no changes were made
    """
    if not overwrite_build_params:
        # TODO: change overwrite_build_params default to True in 1.8.0
        warnings.warn(
            "The `overwrite_build_params` parameter default will change from 'False' to 'True' in 1.8.0.",
            mlrun.utils.OverwriteBuildParamsWarning,
        )

    engine, function = _get_engine_and_function(function, project_object)
    if function.kind in mlrun.runtimes.RuntimeKinds.nuclio_runtimes():
        raise mlrun.errors.MLRunInvalidArgumentError(
            "Cannot build use deploy_function()"
        )
    if engine == mlrun.common.schemas.workflow.EngineType.KFP:
        if overwrite_build_params:
            function.spec.build.commands = None
        if requirements or requirements_file:
            function.with_requirements(
                requirements=requirements,
                requirements_file=requirements_file,
                overwrite=True,
            )
        if commands:
            function.with_commands(commands)
        return function.deploy_step(
            image=image,
            base_image=base_image,
            commands=commands,
            secret_name=secret_name,
            with_mlrun=with_mlrun,
            skip_deployed=skip_deployed,
        )
    else:
        # TODO: remove filter once overwrite_build_params default is changed to True in 1.8.0
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore", category=mlrun.utils.OverwriteBuildParamsWarning
            )

            function.build_config(
                image=image,
                base_image=base_image,
                commands=commands,
                secret=secret_name,
                requirements=requirements,
                requirements_file=requirements_file,
                overwrite=overwrite_build_params,
                extra_args=extra_args,
            )
        ready = function.deploy(
            watch=True,
            with_mlrun=with_mlrun,
            skip_deployed=skip_deployed,
            mlrun_version_specifier=mlrun_version_specifier,
            builder_env=builder_env,
            force_build=force_build,
        )
        # return object with the same outputs as the KFP op (allow using the same pipeline)
        return BuildStatus(ready, {"image": function.spec.image}, function=function)


class DeployStatus:
    """returned status from deploy operation"""

    def __init__(self, state, outputs=None, function=None):
        self.state = state
        self.outputs = outputs or {}
        self.function = function

    def after(self, step):
        """nil function, for KFP compatibility"""
        return self

    def __repr__(self):
        return f"DeployStatus(state={self.state}, outputs={self.outputs})"


def deploy_function(
    function: Union[str, mlrun.runtimes.BaseRuntime],
    models: list = None,
    env: dict = None,
    tag: str = None,
    verbose: bool = None,
    builder_env: dict = None,
    project_object=None,
    mock: bool = None,
) -> Union[DeployStatus, mlrun_pipelines.models.PipelineNodeWrapper]:
    """deploy real-time (nuclio based) functions

    :param function:   name of the function (in the project) or function object
    :param models:     list of model items
    :param env:        dict of extra environment variables
    :param tag:        extra version tag
    :param verbose:    add verbose prints/logs
    :param builder_env: env vars dict for source archive config/credentials e.g. builder_env={"GIT_TOKEN": token}
    :param mock:       deploy mock server vs a real Nuclio function (for local simulations)
    :param project_object:  override the project object to use, will default to the project set in the runtime context.
    """
    engine, function = _get_engine_and_function(function, project_object)
    if function.kind not in mlrun.runtimes.RuntimeKinds.nuclio_runtimes():
        raise mlrun.errors.MLRunInvalidArgumentError(
            "deploy is used with real-time functions, for other kinds use build_function()"
        )
    if engine == mlrun.common.schemas.workflow.EngineType.KFP:
        return function.deploy_step(models=models, env=env, tag=tag, verbose=verbose)
    else:
        if env:
            function.set_envs(env)
        if models:
            for model_args in models:
                function.add_model(**model_args)

        mock = mlrun.mlconf.use_nuclio_mock(mock)
        function._set_as_mock(mock)
        if mock:
            # make sure the latest ver is saved in the DB (same as in function.deploy())
            function.save()
            return DeployStatus(
                state="ready",
                outputs={"endpoint": "Mock", "name": function.metadata.name},
                function=function,
            )

        address = function.deploy(tag=tag, verbose=verbose, builder_env=builder_env)
        # return object with the same outputs as the KFP op (allow using the same pipeline)
        return DeployStatus(
            state=function.status.state,
            outputs={"endpoint": address, "name": function.status.nuclio_name},
            function=function,
        )
