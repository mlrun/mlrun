import mlrun
from .pipelines import pipeline_context


def _is_kfp():
    return pipeline_context.workflow


def run_function(
    function_uri,
    handler=None,
    name: str = "",
    params: dict = None,
    hyper_params=None,
    hyper_param_options: mlrun.model.HyperParamOptions = None,
    inputs: dict = None,
    outputs: dict = None,
    workdir: str = "",
    labels: dict = None,
    verbose=None,
):
    """Run a local or remote task.

    :param handler:         name of the function handler
    :param name:            execution name
    :param params:          input parameters (dict)
    :param hyperparams:     hyper parameters
    :param selector:        selection criteria for hyper params
    :param inputs:          input objects (dict of key: path)
    :param outputs:         list of outputs which can pass in the workflow
    :param artifact_path:   default artifact output path (replace out_path)
    :param workdir:         default input artifacts path
    :param image:           container image to use
    :param labels:          labels to tag the job/run with ({key:val, ..})
    :param verbose:         add verbose prints/logs

    :return: KubeFlow containerOp
    """
    function = pipeline_context.functiond[function_uri]
    task = mlrun.new_task(
        name,
        handler=handler,
        params=params,
        hyper_params=hyper_params,
        hyper_param_options=hyper_param_options,
    )
    if _is_kfp():
        function.as_step()

    return function.run(
        handler=handler,
        name=name,
        params=params,
        hyperparams=hyper_params,
        hyper_param_options=hyper_param_options,
        inputs=inputs,
        workdir=workdir,
        labels=labels,
        verbose=verbose,
        local=True,
    )


def as_step(
    self,
    runspec: RunObject = None,
    handler=None,
    name: str = "",
    project: str = "",
    params: dict = None,
    hyperparams=None,
    selector="",
    hyper_param_options: HyperParamOptions = None,
    inputs: dict = None,
    outputs: dict = None,
    workdir: str = "",
    artifact_path: str = "",
    image: str = "",
    labels: dict = None,
    use_db=True,
    verbose=None,
    scrape_metrics=False,
):
    """Run a local or remote task.

    :param runspec:         run template object or dict (see RunTemplate)
    :param handler:         name of the function handler
    :param name:            execution name
    :param project:         project name
    :param params:          input parameters (dict)
    :param hyperparams:     hyper parameters
    :param selector:        selection criteria for hyper params
    :param inputs:          input objects (dict of key: path)
    :param outputs:         list of outputs which can pass in the workflow
    :param artifact_path:   default artifact output path (replace out_path)
    :param workdir:         default input artifacts path
    :param image:           container image to use
    :param labels:          labels to tag the job/run with ({key:val, ..})
    :param use_db:          save function spec in the db (vs the workflow file)
    :param verbose:         add verbose prints/logs
    :param scrape_metrics:  whether to add the `mlrun/scrape-metrics` label to this run's resources

    :return: KubeFlow containerOp
    """
