from mlrun.runtimes import RuntimeKinds

import mlrun
from .pipelines import pipeline_context


def _get_engine():
    if not pipeline_context.workflow:
        raise ValueError("not running inside a workflow")
    return pipeline_context.workflow.engine


def run_function(
    function_name,
    handler=None,
    name: str = "",
    params: dict = None,
    hyperparams=None,
    hyper_param_options: mlrun.model.HyperParamOptions = None,
    inputs: dict = None,
    outputs: dict = None,
    workdir: str = "",
    labels: dict = None,
    base_task = None,
    watch=True,
    local=True,
    verbose=None,
):
    """Run a local or remote task as part of a local/kubeflow pipeline

    example::

        @dsl.pipeline(name="test pipeline", description="test")
        def my_pipe(url=""):
            run1 = run_function("loaddata", params={"url": url})
            run2 = run_function("train", params={"data": run1.outputs["data"]})

    :param function_name    name of the function (in the project)
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

    :return: KubeFlow containerOp
    """
    function = pipeline_context.functions[function_name]
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

    engine = _get_engine()
    if engine == "kfp":
        return function.as_step(
            runspec=task,
            workdir=workdir,
            outputs=outputs,
            labels=labels
        )
    else:
        if pipeline_context.workflow:
            local = local or pipeline_context.workflow.run_local
        task.metadata.labels = task.metadata.labels or labels or {}
        task.metadata.labels["workflow"] = pipeline_context.workflow_id
        run = function.run(
            runspec=task,
            workdir=workdir,
            verbose=verbose,
            watch=watch,
            local=local,
        )
        if run:
            run._notified = False
            pipeline_context.runs_map[run.uid()] = run
        return run


def build_function(
        function_name,
        with_mlrun=True,
        skip_deployed=False,
        image=None,
        base_image=None,
        commands: list = None,
        secret_name="",
        mlrun_version_specifier=None,
        builder_env: dict = None,
):
    """deploy ML function, build container with its dependencies

    :param function_name    name of the function (in the project)
    :param with_mlrun:      add the current mlrun package to the container build
    :param skip_deployed:   skip the build if we already have an image for the function
    :param image:           target image name/path
    :param base_image:      base image name/path (commands and source code will be added to it)
    :param commands:        list of docker build (RUN) commands e.g. ['pip install pandas']
    :param secret_name:     k8s secret for accessing the docker registry
    :param mlrun_version_specifier:  which mlrun package version to include (if not current)
    :param builder_env:     Kaniko builder pod env vars dict (for config/credentials)
                            e.g. builder_env={"GIT_TOKEN": token}, does not work yet in KFP
    """
    function = pipeline_context.functions[function_name]
    if function.kind in mlrun.runtimes.RuntimeKinds.nuclio_runtimes():
        raise mlrun.errors.MLRunInvalidArgumentError(
            "cannot build use deploy_function()"
        )
    engine = _get_engine()
    if engine == "kfp":
        function.deploy_step(
            image=image,
            base_image=base_image,
            commands = commands,
            secret_name = secret_name,
            with_mlrun = with_mlrun,
            skip_deployed = skip_deployed,
        )
    else:
        function.build_config(image=image, base_image=base_image, commands=commands, secret=secret_name)
        function.deploy(
            watch=True,
            with_mlrun=with_mlrun,
            skip_deployed=skip_deployed,
            mlrun_version_specifier=mlrun_version_specifier,
            builder_env = builder_env,
        )


def deploy_function(
        function_name,
        dashboard="",
        models=None,
        env=None,
        tag=None,
        verbose=None,
):
    """deploy real-time (nuclio based) functions

    :param dashboard:  url of the remore Nuclio dashboard (when not local)
    :param models:     list of model items
    :param env:        dict of extra environment variables
    :param tag:        extra version tag
    :param verbose     add verbose prints/logs
    """
    function = pipeline_context.functions[function_name]
    if function.kind not in mlrun.runtimes.RuntimeKinds.nuclio_runtimes():
        raise mlrun.errors.MLRunInvalidArgumentError(
            "deploy is used with real-time functions, for other kinds use build_function()"
        )
    engine = _get_engine()
    if engine == "kfp":
        function.deploy_step(
            dashboard=dashboard,
            models=models,
            env=env,
            tag=tag,
            verbose=verbose
        )
    else:
        if env:
            function.set_envs(env)
        function.deploy(
            dashboard=dashboard,
            tag=tag,
            verbose=verbose
        )

