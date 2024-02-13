from typing import Dict, List, Union

from mlrun.pipelines.common.helpers import (
    function_annotation,
    project_annotation,
    run_annotation,
)
from mlrun.utils import get_in


def generate_kfp_dag_and_resolve_project(run, project=None):
    workflow = run.workflow_manifest
    if not workflow:
        return None, project, None

    templates = {}
    for template in workflow["spec"]["templates"]:
        project = project or get_in(
            template, ["metadata", "annotations", project_annotation], ""
        )
        name = template["name"]
        templates[name] = {
            "run_type": get_in(
                template, ["metadata", "annotations", run_annotation], ""
            ),
            "function": get_in(
                template, ["metadata", "annotations", function_annotation], ""
            ),
        }

    # nodes = workflow["status"].get("nodes", {})
    nodes = run["run_details"]["task_details"]
    dag = {}
    for node in nodes:
        name = node["display_name"]
        record = {
            "phase": node["state"],
            "started_at": node["create_time"],
            "finished_at": node["end_time"],
            "id": node["display_name"],
            "parent": node.get("display_name", ""),
            "name": name,
            "type": "DAG" if node["child_tasks"] else "Pod",
            "children": [c["pod_name"] for c in node["child_tasks"] or []],
        }

        if name in templates:
            record["function"] = templates[name].get("function")
            record["run_type"] = templates[name].get("run_type")
        dag[node["display_name"]] = record

    # TODO: find workflow exit message on the KFP 2.0 API and use it instead of "state"
    return dag, project, run["state"]


def show_kfp_run(run, clear_output=False):
    return


def mlrun_op(
    name: str = "",
    project: str = "",
    function=None,
    func_url=None,
    image: str = "",
    runobj=None,
    command: str = "",
    secrets: list = None,
    params: dict = None,
    job_image=None,
    hyperparams: dict = None,
    param_file: str = "",
    labels: dict = None,
    selector: str = "",
    inputs: dict = None,
    outputs: list = None,
    in_path: str = "",
    out_path: str = "",
    rundb: str = "",
    mode: str = "",
    handler: str = "",
    more_args: list = None,
    hyper_param_options=None,
    verbose=None,
    scrape_metrics=False,
    returns: List[Union[str, Dict[str, str]]] = None,
    auto_build: bool = False,
):
    """mlrun KubeFlow pipelines operator, use to form pipeline steps

    when using kubeflow pipelines, each step is wrapped in an mlrun_op
    one step can pass state and data to the next step, see example below.

    :param name:    name used for the step
    :param project: optional, project name
    :param image:   optional, run container image (will be executing the step)
                    the container should host all required packages + code
                    for the run, alternatively user can mount packages/code via
                    shared file volumes like v3io (see example below)
    :param function: optional, function object
    :param func_url: optional, function object url
    :param command: exec command (or URL for functions)
    :param secrets: extra secrets specs, will be injected into the runtime
                    e.g. ['file=<filename>', 'env=ENV_KEY1,ENV_KEY2']
    :param params:  dictionary of run parameters and values
    :param hyperparams: dictionary of hyper parameters and list values, each
                        hyperparam holds a list of values, the run will be
                        executed for every parameter combination (GridSearch)
    :param param_file:  a csv/json file with parameter combinations, first csv row hold
                        the parameter names, following rows hold param values
    :param selector: selection criteria for hyperparams e.g. "max.accuracy"
    :param hyper_param_options: hyper param options class, see: :py:class:`~mlrun.model.HyperParamOptions`
    :param labels:   labels to tag the job/run with ({key:val, ..})
    :param inputs:   dictionary of input objects + optional paths (if path is
                     omitted the path will be the in_path/key.
    :param outputs:  dictionary of output objects + optional paths (if path is
                     omitted the path will be the out_path/key.
    :param in_path:  default input path/url (prefix) for inputs
    :param out_path: default output path/url (prefix) for artifacts
    :param rundb:    path for rundb (or use 'MLRUN_DBPATH' env instead)
    :param mode:     run mode, e.g. 'pass' for using the command without mlrun wrapper
    :param handler   code entry-point/handler name
    :param job_image name of the image user for the job
    :param verbose:  add verbose prints/logs
    :param scrape_metrics:  whether to add the `mlrun/scrape-metrics` label to this run's resources
    :param returns: List of configurations for how to log the returning values from the handler's run (as artifacts or
                    results). The list's length must be equal to the amount of returning objects. A configuration may be
                    given as:

                    * A string of the key to use to log the returning value as result or as an artifact. To specify
                      The artifact type, it is possible to pass a string in the following structure:
                      "<key> : <type>". Available artifact types can be seen in `mlrun.ArtifactType`. If no artifact
                      type is specified, the object's default artifact type will be used.
                    * A dictionary of configurations to use when logging. Further info per object type and artifact
                      type can be given there. The artifact key must appear in the dictionary as "key": "the_key".
    :param auto_build: when set to True and the function require build it will be built on the first
                       function run, use only if you dont plan on changing the build config between runs

    :returns: KFP step operation

    Example:
    from kfp import dsl
    from mlrun import mlrun_op
    from mlrun.platforms import mount_v3io

    def mlrun_train(p1, p2):
    return mlrun_op('training',
                    command = '/User/kubeflow/training.py',
                    params = {'p1':p1, 'p2':p2},
                    outputs = {'model.txt':'', 'dataset.csv':''},
                    out_path ='v3io:///projects/my-proj/mlrun/{{workflow.uid}}/',
                    rundb = '/User/kubeflow')

    # use data from the first step
    def mlrun_validate(modelfile):
        return mlrun_op('validation',
                    command = '/User/kubeflow/validation.py',
                    inputs = {'model.txt':modelfile},
                    out_path ='v3io:///projects/my-proj/{{workflow.uid}}/',
                    rundb = '/User/kubeflow')

    @dsl.pipeline(
        name='My MLRUN pipeline', description='Shows how to use mlrun.'
    )
    def mlrun_pipeline(
        p1 = 5 , p2 = '"text"'
    ):
        # run training, mount_v3io will mount "/User" into the pipeline step
        train = mlrun_train(p1, p2).apply(mount_v3io())

        # feed 1st step results into the second step
        validate = mlrun_validate(
            train.outputs['model-txt']).apply(mount_v3io())

    """
    return


def build_op(
    name,
    function=None,
    func_url=None,
    image=None,
    base_image=None,
    commands: list = None,
    secret_name="",
    with_mlrun=True,
    skip_deployed=False,
):
    """build Docker image."""
    return


def deploy_op(
    name,
    function,
    func_url=None,
    source="",
    project="",
    models: list = None,
    env: dict = None,
    tag="",
    verbose=False,
):
    return
